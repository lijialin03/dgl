"""Paddle Module for GraphSAGE layer"""

import paddle

from .... import function as fn
from ....base import DGLError
from ....utils import check_eq_shape
from ....utils import expand_as_pair


class SAGEConv(paddle.nn.Layer):
    """GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__

    .. math::
        h_{\\mathcal{N}(i)}^{(l+1)} &= \\mathrm{aggregate}
        \\left(\\{h_{j}^{l}, \\forall j \\in \\mathcal{N}(i) \\}\\right)

        h_{i}^{(l+1)} &= \\sigma \\left(W \\cdot \\mathrm{concat}
        (h_{i}^{l}, h_{\\mathcal{N}(i)}^{l+1}) \\right)

        h_{i}^{(l+1)} &= \\mathrm{norm}(h_{i}^{(l+1)})

    If a weight tensor on each edge is provided, the aggregation becomes:

    .. math::
        h_{\\mathcal{N}(i)}^{(l+1)} = \\mathrm{aggregate}
        \\left(\\{e_{ji} h_{j}^{l}, \\forall j \\in \\mathcal{N}(i) \\}\\right)

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that :math:`e_{ji}` is broadcastable with :math:`h_j^{l}`.

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.

        SAGEConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer applies on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    feat_drop : float
        Dropout rate on features, default: ``0``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import paddle as th
    >>> from dgl.nn import SAGEConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = SAGEConv(10, 2, 'pool')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099]], grad_fn=<AddBackward0>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('_N', '_E', '_N'):(u, v)})
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 10)
    >>> conv = SAGEConv((5, 10), 2, 'mean')
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[ 0.3163,  3.1166],
            [ 0.3866,  2.5398],
            [ 0.5873,  1.6597],
            [-0.2502,  2.8068]], grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = paddle.nn.Dropout(p=feat_drop)
        self.activation = activation
        if aggregator_type == "pool":
            self.fc_pool = paddle.nn.Linear(
                in_features=self._in_src_feats, out_features=self._in_src_feats
            )
        if aggregator_type == "lstm":
            self.lstm = paddle.nn.LSTM(
                input_size=self._in_src_feats,
                hidden_size=self._in_src_feats,
                time_major=not True,
            )
        self.fc_neigh = paddle.nn.Linear(
            in_features=self._in_src_feats,
            out_features=out_feats,
            bias_attr=False,
        )
        if aggregator_type != "gcn":
            self.fc_self = paddle.nn.Linear(
                in_features=self._in_dst_feats,
                out_features=out_feats,
                bias_attr=bias,
            )
        elif bias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=self._out_feats)
            )
        else:
            self.register_buffer(name="bias", tensor=None)
        self.reset_parameters()

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = paddle.nn.initializer.calculate_gain(nonlinearity="relu")
        if self._aggre_type == "pool":
            init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=gain)
            init_XavierUniform(self.fc_pool.weight)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=gain)
            init_XavierUniform(self.fc_self.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=gain)
        init_XavierUniform(self.fc_neigh.weight)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]
        batch_size = tuple(m.shape)[0]
        h = paddle.zeros(
            shape=(1, batch_size, self._in_src_feats), dtype=m.dtype
        ), paddle.zeros(
            shape=(1, batch_size, self._in_src_feats), dtype=m.dtype
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(axis=0)}

    def forward(self, graph, feat, edge_weight=None):
        """

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : paddle.Tensor or pair of paddle.Tensor
            If a paddle.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of paddle.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : paddle.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        paddle.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert tuple(edge_weight.shape)[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")
            h_self = feat_dst
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = paddle.zeros(
                    shape=[tuple(feat_dst.shape)[0], self._in_src_feats]
                ).to(feat_dst)
            lin_before_mp = self._in_src_feats > self._out_feats
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                elif graph.is_block:
                    graph.dstdata["h"] = graph.srcdata["h"][
                        : graph.num_dst_nodes()
                    ]
                else:
                    graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(axis=-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = paddle.nn.functional.relu(
                    x=self.fc_pool(feat_src)
                )
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )
            if self._aggre_type == "gcn":
                rst = h_neigh
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh
            if self.activation is not None:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
