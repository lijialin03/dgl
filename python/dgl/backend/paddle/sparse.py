import logging

import paddle

from ..._sparse_ops import _bwd_segment_cmp
from ..._sparse_ops import _csrmask
from ..._sparse_ops import _csrmm
from ..._sparse_ops import _csrsum
from ..._sparse_ops import _edge_softmax_backward
from ..._sparse_ops import _edge_softmax_forward
from ..._sparse_ops import _gather_mm
from ..._sparse_ops import _gather_mm_scatter
from ..._sparse_ops import _gsddmm
from ..._sparse_ops import _gsddmm_hetero
from ..._sparse_ops import _gspmm
from ..._sparse_ops import _gspmm_hetero
from ..._sparse_ops import _scatter_add
from ..._sparse_ops import _segment_mm
from ..._sparse_ops import _segment_mm_backward_B
from ..._sparse_ops import _segment_reduce
from ..._sparse_ops import _update_grad_minmax_hetero
from ...base import ALL
from ...base import is_all
from ...heterograph_index import create_unitgraph_from_csr
from .utils import transpose_aux_func

logger = logging.getLogger(__name__)

__all__ = [
    "gspmm",
    "gsddmm",
    "gspmm_hetero",
    "gsddmm_hetero",
    "edge_softmax",
    "edge_softmax_hetero",
    "segment_reduce",
    "scatter_add",
    "csrmm",
    "csrsum",
    "csrmask",
    "gather_mm",
    "segment_mm",
]


def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension
    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.

    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor

    Returns
    -------
    Tensor
    """
    grad_shape = tuple(grad.shape)[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = paddle.nonzero(
        x=paddle.to_tensor(data=grad_shape) - paddle.to_tensor(data=in_shape),
        as_tuple=False,
    )
    reduce_idx += 1
    if len(reduce_idx) > 0:
        grad = grad.sum(axis=tuple(reduce_idx), keepdim=True)
    return grad.view(-1, *shape[1:])


def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    if ufeat is None or efeat is None:
        return False
    ushp = tuple(ufeat.shape)
    eshp = tuple(efeat.shape)
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1


def _expand(x, shape):
    return x.expand(shape=[-1, *shape])


def spmm_cache_X(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache X in SpMM forward stage."""
    if binary_op != "copy_lhs" and req_grad_Y:
        if reduce_op == "sum":
            return True
        elif binary_op == "mul":
            return True
    return False


def spmm_cache_Y(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache Y in SpMM forward stage."""
    if binary_op != "copy_rhs" and req_grad_X:
        if reduce_op == "sum":
            if binary_op in ["mul", "add"]:
                return True
        elif binary_op == "mul":
            return True
    return False


def spmm_cache_argX(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache argX in SpMM forward stage."""
    if req_grad_X or req_grad_Y:
        if reduce_op in ["min", "max"]:
            return True
    return False


def spmm_cache_argY(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache argY in SpMM forward stage."""
    if req_grad_X or req_grad_Y:
        if reduce_op in ["min", "max"]:
            return True
    return False


def _disable_autocast_if_enabled():
    return paddle.amp.auto_cast(enable=False)


class GSpMM(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidx, op, reduce_op, X, Y):
        out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
        reduce_last = _need_reduce_last_dim(X, Y)
        X_shape = tuple(X.shape) if X is not None else None
        Y_shape = tuple(Y.shape) if Y is not None else None
        dtype = X.dtype if X is not None else Y.dtype
        device = X.place if X is not None else Y.place
        ctx.backward_cache = (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
        )
        req_grad_X = not X.stop_gradient if X is not None else False
        req_grad_Y = not Y.stop_gradient if Y is not None else False
        if not spmm_cache_X(op, reduce_op, req_grad_X, req_grad_Y):
            X = None
        if not spmm_cache_Y(op, reduce_op, req_grad_X, req_grad_Y):
            Y = None
        if not spmm_cache_argX(op, reduce_op, req_grad_X, req_grad_Y):
            argX = None
        if not spmm_cache_argY(op, reduce_op, req_grad_X, req_grad_Y):
            argY = None
        ctx.save_for_backward(X, Y, argX, argY)
        return out

    @staticmethod
    def backward(ctx, dZ):
        (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
        ) = ctx.backward_cache
        X, Y, argX, argY = ctx.saved_tensor()
        if op != "copy_rhs" and ctx.needs_input_grad[3]:
            g_rev = gidx.reverse()
            if reduce_op == "sum":
                if op == "mul":
                    dX = gspmm(g_rev, "mul", "sum", dZ, Y)
                elif op == "add":
                    dX = gspmm(g_rev, "copy_lhs", "sum", dZ, Y)
                elif op == "copy_lhs":
                    dX = gspmm(g_rev, "copy_lhs", "sum", dZ, None)
            else:
                dX = paddle.zeros(shape=(X_shape[0],) + tuple(dZ.shape)[1:], dtype=dtype)
                if op == "mul":
                    grad = (
                        _expand(Y, tuple(dZ.shape)[1:]).take_along_axis(
                            axis=0,
                            indices=argY.astype(dtype="int64"),
                            broadcast=False,
                        )
                        * dZ
                    )
                    dX.put_along_axis_(
                        axis=0,
                        indices=argX.astype(dtype="int64"),
                        values=grad,
                        reduce="add",
                    )
                elif op in ["add", "copy_lhs"]:
                    dX.put_along_axis_(
                        axis=0,
                        indices=argX.astype(dtype="int64"),
                        values=dZ,
                        reduce="add",
                    )
            dX = _reduce_grad(dX, X_shape)
        else:
            dX = None
        if op != "copy_lhs" and ctx.needs_input_grad[4]:
            if reduce_op == "sum":
                if op == "mul" and reduce_last:
                    dY = gsddmm(gidx, "dot", X, dZ)
                elif op == "mul":
                    dY = gsddmm(gidx, "mul", X, dZ)
                elif op in ["add", "copy_rhs"]:
                    dY = gsddmm(gidx, "copy_rhs", X, dZ)
            else:
                dY = paddle.zeros(shape=(Y_shape[0],) + tuple(dZ.shape)[1:], dtype=dtype)
                if op == "mul":
                    grad = (
                        _expand(X, tuple(dZ.shape)[1:]).take_along_axis(
                            axis=0,
                            indices=argX.astype(dtype="int64"),
                            broadcast=False,
                        )
                        * dZ
                    )
                    dY.put_along_axis_(
                        axis=0,
                        indices=argY.astype(dtype="int64"),
                        values=grad,
                        reduce="add",
                    )
                elif op in ["add", "copy_rhs"]:
                    dY.put_along_axis_(
                        axis=0,
                        indices=argY.astype(dtype="int64"),
                        values=dZ,
                        reduce="add",
                    )
            dY = _reduce_grad(dY, Y_shape)
        else:
            dY = None
        return None, None, None, dX, dY


class GSpMM_hetero(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidx, op, reduce_op, X_len, *feats):
        out, (argX, argY, argX_ntype, argY_etype) = _gspmm_hetero(gidx, op, reduce_op, X_len, feats)
        X, Y = feats[:X_len], feats[X_len:]
        src_id, dst_id = gidx.metagraph.find_edge(0)
        reduce_last = _need_reduce_last_dim(X[src_id], Y[dst_id])
        X_shape = tuple([(tuple(X[i].shape) if X[i] is not None else None) for i in range(X_len)])
        Y_shape = tuple([(tuple(Y[i].shape) if Y[i] is not None else None) for i in range(len(Y))])
        dtype = X[src_id].dtype if X[src_id] is not None else Y[dst_id].dtype
        device = X[src_id].place if X[src_id] is not None else Y[dst_id].place
        ctx.backward_cache = (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
            X_len,
        )
        req_grad_X = tuple([(not X[i].stop_gradient if X[i] is not None else False) for i in range(X_len)])
        req_grad_Y = tuple([(not Y[i].stop_gradient if Y[i] is not None else False) for i in range(len(Y))])
        if not spmm_cache_argX(op, reduce_op, req_grad_X[src_id], req_grad_Y[dst_id]):
            argX = tuple([None] * len(X))
        if not spmm_cache_argY(op, reduce_op, req_grad_X[src_id], req_grad_Y[dst_id]):
            argY = tuple([None] * len(X))
        ctx.save_for_backward(*feats, *argX, *argX_ntype, *argY, *argY_etype)
        return out

    @staticmethod
    def backward(ctx, *dZ):
        (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
            X_len,
        ) = ctx.backward_cache
        saved_tensors = ctx.saved_tensor()
        num_ntypes = gidx.number_of_ntypes()
        indices = [len(saved_tensors) - (i * num_ntypes) for i in range(5, 0, -1)]
        feats = saved_tensors[: indices[0]]
        argX = saved_tensors[indices[0] : indices[1]]
        argX_ntype = saved_tensors[indices[1] : indices[2]]
        argY = saved_tensors[indices[2] : indices[3]]
        argY_etype = saved_tensors[indices[3] :]
        X, Y = feats[:X_len], feats[X_len:]
        if op != "copy_rhs" and any([(x is not None) for x in X]):
            g_rev = gidx.reverse()
            if reduce_op == "sum":
                if op == "mul":
                    dX = gspmm_hetero(g_rev, "mul", "sum", len(X), *tuple(dZ + Y))
                elif op == "add":
                    dX = gspmm_hetero(g_rev, "copy_lhs", "sum", len(X), *tuple(dZ + Y))
                elif op == "copy_lhs":
                    tpl_None = tuple([None] * len(Y))
                    dX = gspmm_hetero(g_rev, "copy_lhs", "sum", len(X), *tuple(dZ + tpl_None))
            else:
                src_id, dst_id = gidx.metagraph.find_edge(0)
                dX = tuple(
                    [
                        (
                            paddle.zeros(
                                shape=(X_shape[i][0],) + tuple(dZ[dst_id].shape)[1:],
                                dtype=dtype,
                            )
                            if X[i] is not None
                            else None
                        )
                        for i in range(len(X))
                    ]
                )
                if op == "mul":
                    grad = (
                        _expand(Y, tuple(dZ.shape)[1:]).take_along_axis(
                            axis=0,
                            indices=argY.astype(dtype="int64"),
                            broadcast=False,
                        )
                        * dZ
                    )
                    dX.put_along_axis_(
                        axis=0,
                        indices=argX.astype(dtype="int64"),
                        values=grad,
                        reduce="add",
                    )
                elif op in ["add", "copy_lhs"]:
                    dX = _update_grad_minmax_hetero(g_rev, op, dZ, argX, argX_ntype, dX)
            dX = tuple([(_reduce_grad(dX[i], X_shape[i]) if X[i] is not None else None) for i in range(len(X))])
        else:
            dX = tuple([None] * len(X))
        if op != "copy_lhs" and any([(y is not None) for y in Y]):
            if reduce_op == "sum":
                tpl_dZ = tuple([(dZ[i] if dZ[i] is not None else None) for i in range(len(dZ))])
                tpl_X_dZ = tuple(X + tpl_dZ)
                if op == "mul" and reduce_last:
                    dY = gsddmm_hetero(gidx, "dot", X_len, "u", "v", *tpl_X_dZ)
                elif op == "mul":
                    dY = gsddmm_hetero(gidx, "mul", X_len, "u", "v", *tpl_X_dZ)
                elif op in ["add", "copy_rhs"]:
                    dY = gsddmm_hetero(gidx, "copy_rhs", X_len, "u", "v", *tpl_X_dZ)
            else:
                src_id, dst_id = gidx.metagraph.find_edge(0)
                dY = tuple(
                    [
                        (
                            paddle.zeros(
                                shape=(Y_shape[i][0],) + tuple(dZ[dst_id].shape)[1:],
                                dtype=dtype,
                            )
                            if Y[i] is not None
                            else None
                        )
                        for i in range(len(Y))
                    ]
                )
                if op == "mul":
                    grad = (
                        _expand(X, tuple(dZ.shape)[1:]).take_along_axis(
                            axis=0,
                            indices=argX.astype(dtype="int64"),
                            broadcast=False,
                        )
                        * dZ
                    )
                    dY.put_along_axis_(
                        axis=0,
                        indices=argY.astype(dtype="int64"),
                        values=grad,
                        reduce="add",
                    )
                elif op in ["add", "copy_rhs"]:
                    dY = _update_grad_minmax_hetero(gidx.reverse(), op, dZ, argY, argY_etype, dY)
            dY = tuple([(_reduce_grad(dY[i], Y_shape[i]) if dY[i] is not None else None) for i in range(len(dY))])
        else:
            dY = tuple([None] * len(Y))
        return (None, None, None, None) + dX + dY


def sddmm_cache_X(op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache X in SDDMM forward stage."""
    if op in ["mul", "dot"] and req_grad_Y:
        return True
    return False


def sddmm_cache_Y(op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache Y in SDDMM forward stage."""
    if op in ["mul", "dot"] and req_grad_X:
        return True
    return False


class GSDDMM(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidx, op, X, Y, lhs_target, rhs_target):
        out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
        X_shape = tuple(X.shape) if X is not None else None
        Y_shape = tuple(Y.shape) if Y is not None else None
        ctx.backward_cache = gidx, op, lhs_target, rhs_target, X_shape, Y_shape
        req_grad_X = not X.stop_gradient if X is not None else False
        req_grad_Y = not Y.stop_gradient if Y is not None else False
        if not sddmm_cache_X(op, req_grad_X, req_grad_Y):
            X = None
        if not sddmm_cache_Y(op, req_grad_X, req_grad_Y):
            Y = None
        ctx.save_for_backward(X, Y)
        return out

    @staticmethod
    def backward(ctx, dZ):
        gidx, op, lhs_target, rhs_target, X_shape, Y_shape = ctx.backward_cache
        X, Y = ctx.saved_tensor()
        if op != "copy_rhs" and ctx.needs_input_grad[2]:
            if lhs_target in ["u", "v"]:
                _gidx = gidx if lhs_target == "v" else gidx.reverse()
                if op in ["add", "copy_lhs"]:
                    dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ)
                elif rhs_target == lhs_target:
                    dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ) * Y
                elif rhs_target == "e":
                    dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ * Y)
                else:
                    dX = gspmm(_gidx, "mul", "sum", Y, dZ)
            elif op in ["add", "copy_lhs"]:
                dX = dZ
            else:
                dX = gsddmm(gidx, "mul", dZ, Y, "e", rhs_target)
            dX = _reduce_grad(dX, X_shape)
        else:
            dX = None
        if op != "copy_lhs" and ctx.needs_input_grad[3]:
            if rhs_target in ["u", "v"]:
                _gidx = gidx if rhs_target == "v" else gidx.reverse()
                if op in ["add", "copy_rhs"]:
                    dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ)
                elif lhs_target == rhs_target:
                    dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ) * X
                elif lhs_target == "e":
                    dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ * X)
                else:
                    dY = gspmm(_gidx, "mul", "sum", X, dZ)
            elif op in ["add", "copy_rhs"]:
                dY = dZ
            else:
                dY = gsddmm(gidx, "mul", dZ, X, "e", lhs_target)
            dY = _reduce_grad(dY, Y_shape)
        else:
            dY = None
        return None, None, dX, dY, None, None


class GSDDMM_hetero(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidx, op, X_len, lhs_target, rhs_target, *feats):
        out = _gsddmm_hetero(gidx, op, X_len, lhs_target, rhs_target, feats)
        X, Y = feats[:X_len], feats[X_len:]
        X_shape = tuple([(tuple(X[i].shape) if X[i] is not None else None) for i in range(len(X))])
        Y_shape = tuple([(tuple(Y[i].shape) if Y[i] is not None else None) for i in range(len(Y))])
        ctx.backward_cache = (
            gidx,
            op,
            lhs_target,
            rhs_target,
            X_shape,
            Y_shape,
            X_len,
        )
        req_grad_X = tuple([(not X[i].stop_gradient if X[i] is not None else False) for i in range(len(X))])
        req_grad_Y = tuple([(not Y[i].stop_gradient if Y[i] is not None else False) for i in range(len(Y))])
        # Log the information
        logger.debug("Requires grad for X: %s", req_grad_X)
        logger.debug("Requires grad for Y: %s", req_grad_Y)
        ctx.save_for_backward(*feats)
        return out

    @staticmethod
    def backward(ctx, *dZ):
        (
            gidx,
            op,
            lhs_target,
            rhs_target,
            X_shape,
            Y_shape,
            X_len,
        ) = ctx.backward_cache
        feats = ctx.saved_tensor()
        X, Y = feats[:X_len], feats[X_len:]
        if op != "copy_rhs" and any([(x is not None) for x in X]):
            if lhs_target in ["u", "v"]:
                _gidx = gidx if lhs_target == "v" else gidx.reverse()
                tpl_of_None = tuple([None] * len(X))
                if op in ["add", "copy_lhs"]:
                    dX = gspmm_hetero(_gidx, "copy_rhs", "sum", len(X), *tuple(tpl_of_None + dZ))
                elif rhs_target == lhs_target:
                    dX = gspmm_hetero(_gidx, "copy_rhs", "sum", len(X), *tuple(tpl_of_None + dZ)) * Y
                elif rhs_target == "e":
                    dZ_mul_Y = tuple([(dZ[i] * Y[i] if dZ[i] is not None else None) for i in range(len(Y))])
                    dX = gspmm_hetero(_gidx, "copy_rhs", "sum", len(X), *tuple(tpl_of_None + dZ_mul_Y))
                else:
                    dX = gspmm_hetero(_gidx, "mul", "sum", len(X), *tuple(Y + dZ))
            elif op in ["add", "copy_lhs"]:
                dX = dZ
            else:
                num_etype = gidx.number_of_etypes()
                dX = gsddmm_hetero(gidx, "mul", num_etype, "e", rhs_target, *tuple(dZ + Y))
            dX = tuple([(_reduce_grad(dX[i], X_shape[i]) if X[i] is not None else None) for i in range(len(X))])
        else:
            dX = tuple([None] * len(X))
        if op != "copy_lhs" and any([(y is not None) for y in Y]):
            if rhs_target in ["u", "v"]:
                _gidx = gidx if rhs_target == "v" else gidx.reverse()
                tpl_of_None = tuple([None] * len(X))
                if op in ["add", "copy_rhs"]:
                    dY = gspmm_hetero(_gidx, "copy_rhs", "sum", len(X), *tuple(tpl_of_None + dZ))
                elif lhs_target == rhs_target:
                    dY = gspmm_hetero(_gidx, "copy_rhs", "sum", len(X), *tuple(tpl_of_None + dZ)) * X
                elif lhs_target == "e":
                    dZ_mul_X = tuple([(dZ[i] * X[i] if dZ[i] is not None else None) for i in range(len(X))])
                    dY = gspmm_hetero(_gidx, "copy_rhs", "sum", len(X), *tuple(tpl_of_None + dZ_mul_X))
                else:
                    dY = gspmm_hetero(_gidx, "mul", "sum", len(X), *tuple(X + dZ))
            elif op in ["add", "copy_rhs"]:
                dY = tuple([(dZ[i] if dZ[i] is not None else None) for i in range(len(dZ))])
            else:
                num_etype = gidx.number_of_etypes()
                dY = gsddmm_hetero(gidx, "mul", num_etype, "e", lhs_target, *tuple(dZ + X))
            dY = tuple([(_reduce_grad(dY[i], Y_shape[i]) if Y[i] is not None else None) for i in range(len(Y))])
        else:
            dY = tuple([None] * len(Y))
        return (None, None, None, None, None) + dX + dY


class EdgeSoftmax(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidx, score, eids, norm_by):
        """Forward function.

        Pseudo-code:

        .. code:: python

            score = dgl.EData(g, score)
            score_max = score.dst_max()  # of type dgl.NData
            score = score - score_max  # edge_sub_dst, ret dgl.EData
            score_sum = score.dst_sum()  # of type dgl.NData
            out = score / score_sum    # edge_div_dst, ret dgl.EData
            return out.data
        """
        if not is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph
        if norm_by == "src":
            gidx = gidx.reverse()
        if "gpu" in str(score.place):
            score_max = _gspmm(gidx, "copy_rhs", "max", None, score)[0]
            score = paddle.exp(x=_gsddmm(gidx, "sub", score, score_max, "e", "v"))
            score_sum = _gspmm(gidx, "copy_rhs", "sum", None, score)[0]
            out = _gsddmm(gidx, "div", score, score_sum, "e", "v")
        else:
            out = _edge_softmax_forward(gidx, score, "copy_rhs")
        ctx.backward_cache = gidx
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward function.

        Pseudo-code:

        .. code:: python

            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - out * sds_sum  # multiple expressions
            return grad_score.data
        """
        gidx = ctx.backward_cache
        (out,) = ctx.saved_tensor()
        sds = out * grad_out
        if "gpu" in str(out.place):
            accum = gspmm(gidx, "copy_rhs", "sum", None, sds)
            grad_score = sds - gsddmm(gidx, "mul", out, accum, "e", "v")
        else:
            grad_score = _edge_softmax_backward(gidx, out, sds)
        return None, grad_score, None, None


class EdgeSoftmax_hetero(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidx, eids, norm_by, *score):
        """Forward function.

        Pseudo-code:

        .. code:: python

            score = dgl.EData(g, score)
            score_max = score.dst_max()  # of type dgl.NData
            score = score - score_max  # edge_sub_dst, ret dgl.EData
            score_sum = score.dst_sum()  # of type dgl.NData
            out = score / score_sum    # edge_div_dst, ret dgl.EData
            return out.data
        """
        if not is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph
        if norm_by == "src":
            gidx = gidx.reverse()
        u_len = gidx.number_of_ntypes()
        e_len = gidx.number_of_etypes()
        lhs = [None] * u_len
        feats = tuple(lhs + list(score))
        score_max = _gspmm_hetero(gidx, "copy_rhs", "max", u_len, feats)[0]
        out_tmp = _gsddmm_hetero(gidx, "sub", e_len, "e", "v", tuple(list(score) + list(score_max)))
        score = tuple([(paddle.exp(x=out_tmp[i]) if out_tmp[i] is not None else None) for i in range(len(out_tmp))])
        score_sum = _gspmm_hetero(gidx, "copy_rhs", "sum", u_len, tuple(lhs + list(score)))[0]
        out = _gsddmm_hetero(gidx, "div", e_len, "e", "v", tuple(list(score) + list(score_sum)))
        ctx.backward_cache = gidx
        ctx.save_for_backward(*out)
        return out

    @staticmethod
    def backward(ctx, *grad_out):
        """Backward function.

        Pseudo-code:

        .. code:: python

            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - out * sds_sum  # multiple expressions
            return grad_score.data
        """
        gidx = ctx.backward_cache
        u_len = gidx.number_of_ntypes()
        e_len = gidx.number_of_etypes()
        lhs = [None] * u_len
        out = ctx.saved_tensor()
        sds = tuple([(out[i] * grad_out[i]) for i in range(len(out))])
        accum = _gspmm_hetero(gidx, "copy_rhs", "sum", u_len, tuple(lhs + list(sds)))[0]
        out_sddmm = _gsddmm_hetero(gidx, "mul", e_len, "e", "v", tuple(list(out) + list(accum)))
        grad_score = tuple([(sds[i] - out_sddmm[i]) for i in range(len(sds))])
        return (None, None, None) + grad_score


class SegmentReduce(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, op, x, offsets):
        y, arg = _segment_reduce(op, x, offsets)
        ctx.save_for_backward(arg, offsets)
        ctx.backward_cache = op
        return y

    @staticmethod
    def backward(ctx, dy):
        op = ctx.backward_cache
        arg, offsets = ctx.saved_tensor()
        m = offsets[-1].item()
        if op == "sum":
            offsets = offsets[1:]
            indices = paddle.zeros(shape=(m + 1,), dtype=offsets.dtype)
            indices.put_along_axis_(
                axis=0,
                indices=offsets,
                values=paddle.ones_like(x=offsets),
                reduce="add",
            )
            indices = paddle.cumsum(x=indices, axis=-1)[:-1]
            dx = dy[indices]
        else:
            dx = _bwd_segment_cmp(dy, arg, m)
        return (dx,)


class ScatterAdd(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, idx, m):
        y = _scatter_add(x, idx, m)
        ctx.save_for_backward(idx)
        return y

    @staticmethod
    def backward(ctx, dy):
        idx = ctx.saved_tensor()
        return dy[idx], None, None


class CSRMM(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidxA, A_weights, gidxB, B_weights, num_vtypes):
        gidxC, C_weights = _csrmm(gidxA, A_weights, gidxB, B_weights, num_vtypes)
        (
            nrows,
            ncols,
            C_indptr,
            C_indices,
            C_eids,
        ) = gidxC.adjacency_matrix_tensors(0, False, "csr")
        ctx.backward_cache = gidxA, gidxB, gidxC
        ctx.save_for_backward(A_weights, B_weights)
        return (
            paddle.to_tensor(data=nrows),
            paddle.to_tensor(data=ncols),
            C_indptr,
            C_indices,
            C_eids,
            C_weights,
        )

    @staticmethod
    def backward(ctx, dnrows, dncols, dC_indptr, dC_indices, dC_eids, dC_weights):
        gidxA, gidxB, gidxC = ctx.backward_cache
        A_weights, B_weights = ctx.saved_tensor()
        dgidxA, dA_weights = csrmm(
            gidxC,
            dC_weights,
            gidxB.reverse(),
            B_weights,
            gidxA.number_of_ntypes(),
        )
        dgidxB, dB_weights = csrmm(
            gidxA.reverse(),
            A_weights,
            gidxC,
            dC_weights,
            gidxB.number_of_ntypes(),
        )
        dA_weights = csrmask(dgidxA, dA_weights, gidxA)
        dB_weights = csrmask(dgidxB, dB_weights, gidxB)
        return None, dA_weights, None, dB_weights, None


class CSRSum(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidxs, *weights):
        gidxC, C_weights = _csrsum(gidxs, weights)
        (
            nrows,
            ncols,
            C_indptr,
            C_indices,
            C_eids,
        ) = gidxC.adjacency_matrix_tensors(0, False, "csr")
        ctx.backward_cache = gidxs, gidxC
        return (
            paddle.to_tensor(data=nrows),
            paddle.to_tensor(data=ncols),
            C_indptr,
            C_indices,
            C_eids,
            C_weights,
        )

    @staticmethod
    def backward(ctx, dnrows, dncols, dC_indptr, dC_indices, dC_eids, dC_weights):
        gidxs, gidxC = ctx.backward_cache
        return (None,) + tuple(csrmask(gidxC, dC_weights, gidx) for gidx in gidxs)


class CSRMask(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, gidxA, A_weights, gidxB):
        ctx.backward_cache = gidxA, gidxB
        return _csrmask(gidxA, A_weights, gidxB)

    @staticmethod
    def backward(ctx, dB_weights):
        gidxA, gidxB = ctx.backward_cache
        return None, csrmask(gidxB, dB_weights, gidxA), None


class SEGMENTMM(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, A, B, seglen_A):
        if B.dim() != 3:
            raise ValueError("segment_mm expects B to be a 3D tensor.")
        C = paddle.empty(shape=(tuple(A.shape)[0], tuple(B.shape)[2]), dtype=A.dtype)
        C = _segment_mm(A, B, C, seglen_A)
        ctx.backward_cache = A, B, seglen_A
        return C

    @staticmethod
    def backward(ctx, dZ):
        A, B, seglen_A = ctx.backward_cache
        A_grad = B_grad = None
        if ctx.needs_input_grad[0]:
            A_grad = paddle.empty(shape=tuple(A.shape), dtype=A.dtype)
            A_grad = _segment_mm(dZ, B, A_grad, seglen_A, b_trans=True)
        if ctx.needs_input_grad[1]:
            B_grad = paddle.empty(shape=tuple(B.shape), dtype=B.dtype)
            B_grad = _segment_mm_backward_B(A, dZ, B_grad, seglen_A)
        return A_grad, B_grad, None


class GATHERMM(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, A, B, idx_a, idx_b):
        if B.dim() != 3:
            raise ValueError("Expected dimension of B is 3. Got " + str(B.dim()))
        N = len(idx_b) if idx_a is None else len(idx_a)
        C = paddle.zeros(shape=(N, tuple(B.shape)[2]), dtype=A.dtype)
        C = _gather_mm(A, B, C, idx_a, idx_b)
        ctx.backward_cache = A, B, idx_a, idx_b
        return C

    @staticmethod
    def backward(ctx, dZ):
        A, B, idx_a, idx_b = ctx.backward_cache
        A_grad = B_grad = None
        if ctx.needs_input_grad[0]:
            A_grad = paddle.zeros(shape=tuple(A.shape), dtype=A.dtype)
            A_grad = _gather_mm_scatter(
                dZ,
                B.transpose(perm=transpose_aux_func(B.ndim, 1, 2)),
                A_grad,
                idx_b=idx_b,
                idx_c=idx_a,
            )
        if ctx.needs_input_grad[1]:
            B_grad = paddle.zeros(shape=tuple(B.shape), dtype=B.dtype)
            B_grad = _gather_mm_scatter(A, dZ, B_grad, idx_a=idx_a, idx_c=idx_b)
        return A_grad, B_grad, None, None


def gspmm(gidx, op, reduce_op, lhs_data, rhs_data):
    if op == "sub":
        op = "add"
        rhs_data = -rhs_data
    if op == "div":
        op = "mul"
        rhs_data = 1.0 / rhs_data
    return GSpMM.apply(gidx, op, reduce_op, lhs_data, rhs_data)


def gsddmm(gidx, op, lhs_data, rhs_data, lhs_target="u", rhs_target="v"):
    if op == "sub":
        op = "add"
        rhs_data = -rhs_data
    if op == "div":
        op = "mul"
        rhs_data = 1.0 / rhs_data
    return GSDDMM.apply(gidx, op, lhs_data, rhs_data, lhs_target, rhs_target)


def gspmm_hetero(g, op, reduce_op, lhs_len, *lhs_and_rhs_tuple):
    lhs_tuple, rhs_tuple = (
        lhs_and_rhs_tuple[:lhs_len],
        lhs_and_rhs_tuple[lhs_len:],
    )
    if op == "sub":
        op = "add"
        rhs_tuple = tuple([(-rhs_tuple[i] if rhs_tuple[i] is not None else None) for i in range(len(rhs_tuple))])
    if op == "div":
        op = "mul"
        rhs_tuple = tuple([(1.0 / rhs_tuple[i] if rhs_tuple[i] is not None else None) for i in range(len(rhs_tuple))])
    if op in ["add", "mul"]:
        lhs_and_rhs_tuple = tuple(list(lhs_tuple) + list(rhs_tuple))
    return GSpMM_hetero.apply(g, op, reduce_op, lhs_len, *lhs_and_rhs_tuple)


def gsddmm_hetero(g, op, lhs_len, lhs_target="u", rhs_target="v", *lhs_and_rhs_tuple):
    lhs_tuple, rhs_tuple = (
        lhs_and_rhs_tuple[:lhs_len],
        lhs_and_rhs_tuple[lhs_len:],
    )
    if op == "sub":
        op = "add"
        rhs_tuple = tuple([(-rhs_tuple[i] if rhs_tuple[i] is not None else None) for i in range(len(rhs_tuple))])
    if op == "div":
        op = "mul"
        rhs_tuple = tuple([(1.0 / rhs_tuple[i] if rhs_tuple[i] is not None else None) for i in range(len(rhs_tuple))])
    if op in ["add", "mul"]:
        lhs_and_rhs_tuple = tuple(list(lhs_tuple) + list(rhs_tuple))
    return GSDDMM_hetero.apply(g, op, lhs_len, lhs_target, rhs_target, *lhs_and_rhs_tuple)


def edge_softmax(gidx, logits, eids=ALL, norm_by="dst"):
    return EdgeSoftmax.apply(gidx, logits, eids, norm_by)


def edge_softmax_hetero(gidx, eids=ALL, norm_by="dst", *logits):
    return EdgeSoftmax_hetero.apply(gidx, eids, norm_by, *logits)


def segment_reduce(op, x, offsets):
    return SegmentReduce.apply(op, x, offsets)


def scatter_add(x, idx, m):
    return ScatterAdd.apply(x, idx, m)


def csrmm(gidxA, A_weights, gidxB, B_weights, num_vtypes):
    nrows, ncols, C_indptr, C_indices, C_eids, C_weights = CSRMM.apply(gidxA, A_weights, gidxB, B_weights, num_vtypes)
    gidxC = create_unitgraph_from_csr(
        num_vtypes,
        nrows.item(),
        ncols.item(),
        C_indptr,
        C_indices,
        C_eids,
        ["coo", "csr", "csc"],
    )
    return gidxC, C_weights


def csrsum(gidxs, weights):
    nrows, ncols, C_indptr, C_indices, C_eids, C_weights = CSRSum.apply(gidxs, *weights)
    gidxC = create_unitgraph_from_csr(
        gidxs[0].number_of_ntypes(),
        nrows.item(),
        ncols.item(),
        C_indptr,
        C_indices,
        C_eids,
        ["coo", "csr", "csc"],
    )
    return gidxC, C_weights


def csrmask(gidxA, A_weights, gidxB):
    return CSRMask.apply(gidxA, A_weights, gidxB)


def segment_mm(A, B, seglen_A):
    if A.device.type == "cpu":
        C = []
        off = 0
        for i in range(tuple(B.shape)[0]):
            C.append(A[off : off + seglen_A[i]] @ B[i])
            off += seglen_A[i]
        return paddle.concat(x=C)
    else:
        return SEGMENTMM.apply(A, B, seglen_A)


def gather_mm(A, B, idx_A=None, idx_B=None):
    if A.device.type == "cpu":
        A = A[idx_A] if idx_A is not None else A
        B = B[idx_B] if idx_B is not None else B
        return paddle.bmm(x=A.unsqueeze(axis=1), y=B).squeeze(axis=1)
    else:
        return GATHERMM.apply(A, B, idx_A, idx_B)
