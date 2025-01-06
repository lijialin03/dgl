"""Torch modules for graph attention networks(GAT)."""
import paddle

from .... import function as fn
from ....backend.paddle.utils import transpose_aux_func
from ....base import DGLError
from ....utils import expand_as_pair
from ...functional import edge_softmax
from ..utils import Identity


class GATConv(paddle.nn.Layer):
    """Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__

    .. math::
        h_i^{(l+1)} = \\sum_{j\\in \\mathcal{N}(i)} \\alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \\alpha_{ij}^{l} &= \\mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \\mathrm{LeakyReLU}\\left(\\vec{a}^T [W h_{i} \\| W h_{j}]\\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GATConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> gatconv = GATConv(10, 2, num_heads=3)
    >>> res = gatconv(g, feat)
    >>> res
    tensor([[[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]]], grad_fn=<BinaryReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatconv = GATConv((5,10), 2, 3)
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    tensor([[[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]],
            [[ 0.0268,  1.0783],
            [ 0.5041, -1.3025],
            [ 0.6568,  0.7048]],
            [[-0.2688,  1.0543],
            [-0.0315, -0.9016],
            [ 0.3943,  0.5347]],
            [[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = paddle.nn.Linear(
                in_features=self._in_src_feats,
                out_features=out_feats * num_heads,
                bias_attr=False,
            )
            self.fc_dst = paddle.nn.Linear(
                in_features=self._in_dst_feats,
                out_features=out_feats * num_heads,
                bias_attr=False,
            )
        else:
            self.fc = paddle.nn.Linear(
                in_features=self._in_src_feats,
                out_features=out_feats * num_heads,
                bias_attr=False,
            )
        self.attn_l = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=(1, num_heads, out_feats), dtype="float32")
        )
        self.attn_r = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=(1, num_heads, out_feats), dtype="float32")
        )
        self.feat_drop = paddle.nn.Dropout(p=feat_drop)
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.leaky_relu = paddle.nn.LeakyReLU(negative_slope=negative_slope)
        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = paddle.nn.Linear(
                    in_features=self._in_dst_feats,
                    out_features=num_heads * out_feats,
                    bias_attr=bias,
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer(name="res_fc", tensor=None)
        if bias and not self.has_linear_res:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=(num_heads * out_feats,), dtype="float32")
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer(name="bias", tensor=None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = paddle.nn.initializer.calculate_gain(nonlinearity="relu")
        if hasattr(self, "fc"):
            init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
            init_XavierNormal(self.fc.weight)
        else:
            init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
            init_XavierNormal(self.fc_src.weight)
            init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
            init_XavierNormal(self.fc_dst.weight)
        init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
        init_XavierNormal(self.attn_l)
        init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
        init_XavierNormal(self.attn_r)
        if self.has_explicit_bias:
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(self.bias)
        if isinstance(self.res_fc, paddle.nn.Linear):
            init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
            init_XavierNormal(self.res_fc.weight)
            if self.res_fc.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(self.res_fc.bias)

    def set_allow_zero_in_degree(self, set_value):
        """

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        """

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).astype("bool").any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            if isinstance(feat, tuple):
                src_prefix_shape = tuple(feat[0].shape)[:-1]
                dst_prefix_shape = tuple(feat[1].shape)[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = tuple(feat.shape)[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            el = (feat_src * self.attn_l).sum(axis=-1).unsqueeze(axis=-1)
            er = (feat_dst * self.attn_r).sum(axis=-1).unsqueeze(axis=-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(repeat_times=[1, self._num_heads, 1]).transpose(
                    perm=transpose_aux_func(
                        edge_weight.tile(repeat_times=[1, self._num_heads, 1]).ndim,
                        0,
                        2,
                    )
                )
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
