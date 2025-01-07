"""Paddle modules for graph attention networks(GAT)."""

import paddle

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair
from ...functional import edge_softmax


class EdgeGATConv(paddle.nn.Layer):
    """Graph attention layer with edge features from `SCENE
    <https://arxiv.org/pdf/2301.03512.pdf>`__

    .. math::

        \\mathbf{v}_i^\\prime = \\mathbf{\\Theta}_\\mathrm{s} \\cdot \\mathbf{v}_i +
        \\sum\\limits_{j \\in \\mathcal{N}(v_i)} \\alpha_{j, i} \\left( \\mathbf{\\Theta}_\\mathrm{n}
        \\cdot \\mathbf{v}_j + \\mathbf{\\Theta}_\\mathrm{e} \\cdot \\mathbf{e}_{j,i} \\right)

    where :math:`\\mathbf{\\Theta}` is used to denote learnable weight matrices
    for the transformation of features of the node to update (s=self),
    neighboring nodes (n=neighbor) and edge features (e=edge).
    Attention weights are obtained by

    .. math::

        \\alpha_{j, i} = \\mathrm{softmax}_i \\Big( \\mathrm{LeakyReLU} \\big( \\mathbf{a}^T
        [ \\mathbf{\\Theta}_\\mathrm{n} \\cdot \\mathbf{v}_i || \\mathbf{\\Theta}_\\mathrm{n}
        \\cdot \\mathbf{v}_j || \\mathbf{\\Theta}_\\mathrm{e} \\cdot \\mathbf{e}_{j,i} ] \\big) \\Big)

    with :math:`\\mathbf{a}` corresponding to a learnable vector.
    :math:`\\mathrm{softmax_i}` stands for the normalization by all incoming edges of node :math:`i`.

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`\\mathbf{v}_i`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    edge_feats: int
        Edge feature size; i.e., the number of dimensions of :math:\\mathbf{e}_{j,i}`.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`\\mathbf{v}_i^\\prime`.
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
    ----------
    >>> import dgl
    >>> import numpy as np
    >>> import paddle as th
    >>> from dgl.nn import EdgeGATConv

    >>> # Case 1: Homogeneous graph.
    >>> num_nodes, num_edges = 8, 30
    >>> # Generate a graph.
    >>> graph = dgl.rand_graph(num_nodes,num_edges)
    >>> node_feats = th.rand((num_nodes, 20))
    >>> edge_feats = th.rand((num_edges, 12))
    >>> edge_gat = EdgeGATConv(
    ...     in_feats=20,
    ...     edge_feats=12,
    ...     out_feats=15,
    ...     num_heads=3,
    ... )
    >>> # Forward pass.
    >>> new_node_feats = edge_gat(graph, node_feats, edge_feats)
    >>> new_node_feats.shape
    (8, 3, 15) (30, 3, 10)

    >>> # Case 2: Unidirectional bipartite graph.
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    >>> u_feat = th.tensor(np.random.rand(2, 25).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 30).astype(np.float32))
    >>> nfeats = (u_feat,v_feat)
    >>> efeats = th.tensor(np.random.rand(5, 15).astype(np.float32))
    >>> in_feats = (25,30)
    >>> edge_feats = 15
    >>> out_feats = 10
    >>> num_heads = 3
    >>> egat_model =  EdgeGATConv(
    ...     in_feats,
    ...     edge_feats,
    ...     out_feats,
    ...     num_heads,
    ... )
    >>> # Forward pass.
    >>> new_node_feats, attention_weights = egat_model(g, nfeats, efeats, get_attention=True)
    >>> new_node_feats.shape, attention_weights.shape
    ((4, 3, 10), (5, 3, 1))
    """

    def __init__(
        self,
        in_feats,
        edge_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(EdgeGATConv, self).__init__()
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
            tensor=paddle.empty(
                shape=(1, num_heads, out_feats), dtype="float32"
            )
        )
        self.attn_r = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(
                shape=(1, num_heads, out_feats), dtype="float32"
            )
        )
        self.feat_drop = paddle.nn.Dropout(p=feat_drop)
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.leaky_relu = paddle.nn.LeakyReLU(negative_slope=negative_slope)
        if bias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(
                    shape=(num_heads * out_feats,), dtype="float32"
                )
            )
        else:
            self.register_buffer(name="bias", tensor=None)
        if residual:
            self.res_fc = paddle.nn.Linear(
                in_features=self._in_dst_feats,
                out_features=num_heads * out_feats,
                bias_attr=False,
            )
        else:
            self.register_buffer(name="res_fc", tensor=None)
        self._edge_feats = edge_feats
        self.fc_edge = paddle.nn.Linear(
            in_features=edge_feats,
            out_features=out_feats * num_heads,
            bias_attr=False,
        )
        self.attn_edge = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(
                shape=(1, num_heads, out_feats), dtype="float32"
            )
        )
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`\\mathbf{\\Theta}` are and the
        attention weights are using xavier initialization method.
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
        init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
        init_XavierNormal(self.fc_edge.weight)
        init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
        init_XavierNormal(self.attn_edge)
        if self.bias is not None:
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(self.bias)
        if isinstance(self.res_fc, paddle.nn.Linear):
            init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=gain)
            init_XavierNormal(self.res_fc.weight)

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

    def forward(self, graph, feat, edge_feat, get_attention=False):
        """

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : paddle.Tensor or pair of paddle.Tensor
            If a paddle.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of paddle.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_feat : paddle.Tensor
            The input edge feature of shape :math:`(E, D_{in_{edge}})`,
            where :math:`E` is the number of edges and :math:`D_{in_{edge}}`
            the size of the edge features.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        paddle.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        paddle.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`. This is returned only
            when :attr:`get_attention` is ``True``.

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
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = tuple(feat.shape)[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            n_edges = tuple(edge_feat.shape)[:-1]
            feat_edge = self.fc_edge(edge_feat).view(
                *n_edges, self._num_heads, self._out_feats
            )
            graph.edata["ft_edge"] = feat_edge
            el = (feat_src * self.attn_l).sum(axis=-1).unsqueeze(axis=-1)
            er = (feat_dst * self.attn_r).sum(axis=-1).unsqueeze(axis=-1)
            ee = (feat_edge * self.attn_edge).sum(axis=-1).unsqueeze(axis=-1)
            graph.edata["ee"] = ee
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e_tmp"))
            graph.edata["e"] = graph.edata["e_tmp"] + graph.edata["ee"]
            graph.apply_edges(fn.u_add_e("ft", "ft_edge", "ft_combined"))
            e = self.leaky_relu(graph.edata.pop("e"))
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            graph.edata["m_combined"] = (
                graph.edata["ft_combined"] * graph.edata["a"]
            )
            graph.update_all(fn.copy_e("m_combined", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
