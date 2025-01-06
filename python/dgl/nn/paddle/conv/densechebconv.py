"""Torch Module for DenseChebConv"""
import paddle


class DenseChebConv(paddle.nn.Layer):
    """Chebyshev Spectral Graph Convolution layer from `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__

    We recommend to use this module when applying ChebConv on dense graphs.

    Parameters
    ----------
    in_feats: int
        Dimension of input features :math:`h_i^{(l)}`.
    out_feats: int
        Dimension of output features :math:`h_i^{(l+1)}`.
    k : int
        Chebyshev filter size.
    activation : function, optional
        Activation function, default is ReLu.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import DenseChebConv
    >>>
    >>> feat = th.ones(6, 10)
    >>> adj = th.tensor([[0., 0., 1., 0., 0., 0.],
    ...         [1., 0., 0., 0., 0., 0.],
    ...         [0., 1., 0., 0., 0., 0.],
    ...         [0., 0., 1., 0., 0., 1.],
    ...         [0., 0., 0., 1., 0., 0.],
    ...         [0., 0., 0., 0., 0., 0.]])
    >>> conv = DenseChebConv(10, 2, 2)
    >>> res = conv(adj, feat)
    >>> res
    tensor([[-3.3516, -2.4797],
            [-3.3516, -2.4797],
            [-3.3516, -2.4797],
            [-4.5192, -3.0835],
            [-2.5259, -2.0527],
            [-0.5327, -1.0219]], grad_fn=<AddBackward0>)

    See also
    --------
    `ChebConv <https://docs.dgl.ai/api/python/nn.pytorch.html#chebconv>`__
    """

    def __init__(self, in_feats, out_feats, k, bias=True):
        super(DenseChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        self.W = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.empty(shape=[k, in_feats, out_feats]))
        if bias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.to_tensor(data=out_feats))
        else:
            self.register_buffer(name="bias", tensor=None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bias is not None:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.bias)
        for i in range(self._k):
            init_XavierNormal = paddle.nn.initializer.XavierNormal(
                gain=paddle.nn.initializer.calculate_gain(nonlinearity="relu")
            )
            init_XavierNormal(self.W[i])

    def forward(self, adj, feat, lambda_max=None):
        """Compute (Dense) Chebyshev Spectral Graph Convolution layer

        Parameters
        ----------
        adj : torch.Tensor
            The adjacency matrix of the graph to apply Graph Convolution on,
            should be of shape :math:`(N, N)`, where a row represents the destination
            and a column represents the source.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : float or None, optional
            A float value indicates the largest eigenvalue of given graph.
            Default: None.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        A = adj.to(feat)
        num_nodes = tuple(A.shape)[0]
        in_degree = 1 / A.sum(axis=1).clip(min=1).sqrt()
        D_invsqrt = paddle.diag(x=in_degree)
        I = paddle.eye(num_rows=num_nodes).to(A)  # noqa: E741
        L = I - D_invsqrt @ A @ D_invsqrt
        if lambda_max is None:
            lambda_ = paddle.linalg.eig(L)[0][:, 0]
            lambda_max = lambda_.max()
        L_hat = 2 * L / lambda_max - I
        Z = [paddle.eye(num_rows=num_nodes).to(A)]
        for i in range(1, self._k):
            if i == 1:
                Z.append(L_hat)
            else:
                Z.append(2 * L_hat @ Z[-1] - Z[-2])
        Zs = paddle.stack(x=Z, axis=0)
        Zh = Zs @ feat.unsqueeze(axis=0) @ self.W
        Zh = Zh.sum(axis=0)
        if self.bias is not None:
            Zh = Zh + self.bias
        return Zh
