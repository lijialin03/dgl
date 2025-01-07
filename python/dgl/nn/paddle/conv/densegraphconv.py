import paddle

"""Paddle Module for DenseGraphConv"""


class DenseGraphConv(paddle.nn.Layer):
    """Graph Convolutional layer from `Semi-Supervised Classification with Graph
    Convolutional Networks <https://arxiv.org/abs/1609.02907>`__

    We recommend user to use this module when applying graph convolution on
    dense graphs.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Notes
    -----
    Zero in-degree nodes will lead to all-zero output. A common practice
    to avoid this is to add a self-loop for each node in the graph,
    which can be achieved by setting the diagonal of the adjacency matrix to be 1.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import paddle as th
    >>> from dgl.nn import DenseGraphConv
    >>>
    >>> feat = th.ones(6, 10)
    >>> adj = th.tensor([[0., 0., 1., 0., 0., 0.],
    ...         [1., 0., 0., 0., 0., 0.],
    ...         [0., 1., 0., 0., 0., 0.],
    ...         [0., 0., 1., 0., 0., 1.],
    ...         [0., 0., 0., 1., 0., 0.],
    ...         [0., 0., 0., 0., 0., 0.]])
    >>> conv = DenseGraphConv(10, 2)
    >>> res = conv(adj, feat)
    >>> res
    tensor([[0.2159, 1.9027],
            [0.3053, 2.6908],
            [0.3053, 2.6908],
            [0.3685, 3.2481],
            [0.3053, 2.6908],
            [0.0000, 0.0000]], grad_fn=<AddBackward0>)

    See also
    --------
    `GraphConv <https://docs.dgl.ai/api/python/nn.paddle.html#graphconv>`__
    """

    def __init__(
        self, in_feats, out_feats, norm="both", bias=True, activation=None
    ):
        super(DenseGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[in_feats, out_feats])
        )
        if bias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(data=out_feats)
            )
        else:
            self.register_buffer(name="bias", tensor=None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.weight)
        if self.bias is not None:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.bias)

    def forward(self, adj, feat):
        """Compute (Dense) Graph Convolution layer.

        Parameters
        ----------
        adj : paddle.Tensor
            The adjacency matrix of the graph to apply Graph Convolution on, when
            applied to a unidirectional bipartite graph, ``adj`` should be of shape
            should be of shape :math:`(N_{out}, N_{in})`; when applied to a homo
            graph, ``adj`` should be of shape :math:`(N, N)`. In both cases,
            a row represents a destination node while a column represents a source
            node.
        feat : paddle.Tensor
            The input feature.

        Returns
        -------
        paddle.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        adj = adj.to(feat)
        src_degrees = adj.sum(axis=0).clip(min=1)
        dst_degrees = adj.sum(axis=1).clip(min=1)
        feat_src = feat
        if self._norm == "both":
            norm_src = paddle.pow(x=src_degrees, y=-0.5)
            shp = tuple(norm_src.shape) + (1,) * (feat.dim() - 1)
            norm_src = paddle.reshape(x=norm_src, shape=shp).to(feat.place)
            feat_src = feat_src * norm_src
        if self._in_feats > self._out_feats:
            feat_src = paddle.matmul(x=feat_src, y=self.weight)
            rst = adj @ feat_src
        else:
            rst = adj @ feat_src
            rst = paddle.matmul(x=rst, y=self.weight)
        if self._norm != "none":
            if self._norm == "both":
                norm_dst = paddle.pow(x=dst_degrees, y=-0.5)
            else:
                norm_dst = 1.0 / dst_degrees
            shp = tuple(norm_dst.shape) + (1,) * (feat.dim() - 1)
            norm_dst = paddle.reshape(x=norm_dst, shape=shp).to(feat.place)
            rst = rst * norm_dst
        if self.bias is not None:
            rst = rst + self.bias
        if self._activation is not None:
            rst = self._activation(rst)
        return rst
