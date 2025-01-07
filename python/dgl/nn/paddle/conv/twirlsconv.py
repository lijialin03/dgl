"""Paddle modules for TWIRLS"""

import paddle

from .... import function as fn


class TWIRLSConv(paddle.nn.Layer):
    """Convolution together with iteratively reweighting least squre from
    `Graph Neural Networks Inspired by Classical Iterative Algorithms
    <https://arxiv.org/pdf/2103.06064.pdf>`__

    Parameters
    ----------
    input_d : int
        Number of input features.
    output_d : int
        Number of output features.
    hidden_d : int
        Size of hidden layers.
    prop_step : int
        Number of propagation steps
    num_mlp_before : int
        Number of mlp layers before propagation. Default: ``1``.
    num_mlp_after : int
        Number of mlp layers after propagation.  Default: ``1``.
    norm : str
        The type of norm layers inside mlp layers. Can be ``'batch'``, ``'layer'`` or ``'none'``.
        Default: ``'none'``
    precond : str
        If True, use pre conditioning and unormalized laplacian, else not use pre conditioning
        and use normalized laplacian. Default: ``True``
    alp : float
        The :math:`\\alpha` in paper. If equal to :math:`0`, will be automatically decided based
        on other hyper prameters. Default: ``0``.
    lam : float
        The :math:`\\lambda` in paper. Default: ``1``.
    attention : bool
        If ``True``, add an attention layer inside propagations. Default: ``False``.
    tau : float
        The :math:`\\tau` in paper. Default: ``0.2``.
    T : float
        The :math:`T` in paper. If < 0, :math:`T` will be set to `\\infty`. Default: ``-1``.
    p : float
        The :math:`p` in paper. Default: ``1``.
    use_eta : bool
        If ``True``, add a learnable weight on each dimension in attention. Default: ``False``.
    attn_bef : bool
        If ``True``, add another attention layer before propagation. Default: ``False``.
    dropout : float
        The dropout rate in mlp layers. Default: ``0.0``.
    attn_dropout : float
        The dropout rate of attention values. Default: ``0.0``.
    inp_dropout : float
        The dropout rate on input features. Default: ``0.0``.


    Note
    ----
     ``add_self_loop`` will be automatically called before propagation.

    Example
    -------
    >>> import dgl
    >>> from dgl.nn import TWIRLSConv
    >>> import paddle as th

    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = TWIRLSConv(10, 2, 128, prop_step = 64)
    >>> res = conv(g , feat)
    >>> res.size()
    (6, 2)
    """

    def __init__(
        self,
        input_d,
        output_d,
        hidden_d,
        prop_step,
        num_mlp_before=1,
        num_mlp_after=1,
        norm="none",
        precond=True,
        alp=0,
        lam=1,
        attention=False,
        tau=0.2,
        T=-1,
        p=1,
        use_eta=False,
        attn_bef=False,
        dropout=0.0,
        attn_dropout=0.0,
        inp_dropout=0.0,
    ):
        super().__init__()
        self.input_d = input_d
        self.output_d = output_d
        self.hidden_d = hidden_d
        self.prop_step = prop_step
        self.num_mlp_before = num_mlp_before
        self.num_mlp_after = num_mlp_after
        self.norm = norm
        self.precond = precond
        self.attention = attention
        self.alp = alp
        self.lam = lam
        self.tau = tau
        self.T = T
        self.p = p
        self.use_eta = use_eta
        self.init_att = attn_bef
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.inp_dropout = inp_dropout
        self.attn_aft = prop_step // 2 if attention else -1
        self.cacheable = (
            not self.attention
            and self.num_mlp_before == 0
            and self.inp_dropout <= 0
        )
        if self.cacheable:
            self.cached_unfolding = None
        self.size_bef_unf = self.hidden_d
        self.size_aft_unf = self.hidden_d
        if self.num_mlp_before == 0:
            self.size_aft_unf = self.input_d
        if self.num_mlp_after == 0:
            self.size_bef_unf = self.output_d
        self.mlp_bef = MLP(
            self.input_d,
            self.hidden_d,
            self.size_bef_unf,
            self.num_mlp_before,
            self.dropout,
            self.norm,
            init_activate=False,
        )
        self.unfolding = TWIRLSUnfoldingAndAttention(
            self.hidden_d,
            self.alp,
            self.lam,
            self.prop_step,
            self.attn_aft,
            self.tau,
            self.T,
            self.p,
            self.use_eta,
            self.init_att,
            self.attn_dropout,
            self.precond,
        )
        self.mlp_aft = MLP(
            self.size_aft_unf,
            self.hidden_d,
            self.output_d,
            self.num_mlp_after,
            self.dropout,
            self.norm,
            init_activate=self.num_mlp_before > 0 and self.num_mlp_after > 0,
        )

    def forward(self, graph, feat):
        """

        Description
        -----------
        Run TWIRLS forward.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : paddle.Tensor
            The initial node features.
        Returns
        -------
        paddle.Tensor
            The output feature

        Note
        ----
        * Input shape: :math:`(N, \\text{input_d})` where :math:`N` is the number of nodes.
        * Output shape: :math:`(N, \\text{output_d})`.
        """
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        x = feat
        if self.cacheable:
            if self.cached_unfolding is None:
                self.cached_unfolding = self.unfolding(graph, x)
            x = self.cached_unfolding
        else:
            if self.inp_dropout > 0:
                x = paddle.nn.functional.dropout(
                    x=x, p=self.inp_dropout, training=self.training
                )
            x = self.mlp_bef(x)
            x = self.unfolding(graph, x)
        x = self.mlp_aft(x)
        return x


class Propagate(paddle.nn.Layer):
    """

    Description
    -----------
    The propagation method which is with pre-conditioning and reparameterizing. Correspond to
    eq.28 in the paper.

    """

    def __init__(self):
        super().__init__()

    def _prop(self, graph, Y, lam):
        """propagation part."""
        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)
        Y = AX(graph, Y)
        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)
        return Y

    def forward(self, graph, Y, X, alp, lam):
        """

        Description
        -----------
        Propagation forward.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        Y : paddle.Tensor
            The feature under propagation. Corresponds to :math:`Z^{(k)}` in eq.28 in the paper.
        X : paddle.Tensor
            The original feature. Corresponds to :math:`Z^{(0)}` in eq.28 in the paper.
        alp : float
            The step size. Corresponds to :math:`\\alpha` in the paper.
        lam : paddle.Tensor
            The coefficient of smoothing term. Corresponds to :math:`\\lambda` in the paper.
        Returns
        -------
        paddle.Tensor
            Propagated feature. :math:`Z^{(k+1)}` in eq.28 in the paper.
        """
        return (
            (1 - alp) * Y
            + alp * lam * self._prop(graph, Y, lam)
            + alp * D_power_bias_X(graph, X, -1, lam, 1 - lam)
        )


class PropagateNoPrecond(paddle.nn.Layer):
    """

    Description
    -----------
    The propagation method which is without pre-conditioning and reparameterizing and using
    normalized laplacian.
    Correspond to eq.30 in the paper.
    """

    def __init__(self):
        super().__init__()

    def forward(self, graph, Y, X, alp, lam):
        """

        Description
        -----------
        Propagation forward.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        Y : paddle.Tensor
            The feature under propagation. Corresponds to :math:`Y^{(k)}` in eq.30 in the paper.
        X : paddle.Tensor
            The original feature. Corresponds to :math:`Y^{(0)}` in eq.30 in the paper.
        alp : float
            The step size. Corresponds to :math:`\\alpha` in the paper.
        lam : paddle.Tensor
            The coefficient of smoothing term. Corresponds to :math:`\\lambda` in the paper.
        Returns
        -------
        paddle.Tensor
            Propagated feature. :math:`Y^{(k+1)}` in eq.30 in the paper.
        """
        return (
            (1 - alp * lam - alp) * Y
            + alp * lam * normalized_AX(graph, Y)
            + alp * X
        )


class Attention(paddle.nn.Layer):
    """

    Description
    -----------
    The attention function. Correspond to :math:`s` in eq.27 the paper.

    Parameters
    ----------
    tau : float
        The lower thresholding parameter. Correspond to :math:`\\tau` in the paper.
    T : float
        The upper thresholding parameter. Correspond to :math:`T` in the paper.
    p : float
        Correspond to :math:`\\rho` in the paper..
    attn_dropout : float
        the dropout rate of attention value. Default: ``0.0``.

    Returns
    -------
    paddle.Tensor
        The output feature
    """

    def __init__(self, tau, T, p, attn_dropout=0.0):
        super().__init__()
        self.tau = tau
        self.T = T
        self.p = p
        self.attn_dropout = attn_dropout

    def reweighting(self, graph):
        """Compute graph edge weight. Would be stored in ``graph.edata['w']``"""
        w = graph.edata["w"]
        w = paddle.nn.functional.relu(x=w) + 1e-07
        w = paddle.pow(x=w, y=1 - 0.5 * self.p)
        w[w < self.tau] = self.tau
        if self.T > 0:
            w[w > self.T] = float("inf")
        w = 1 / w
        graph.edata["w"] = w + 1e-09

    def forward(self, graph, Y, etas=None):
        """

        Description
        -----------
        Attention forward. Will update ``graph.edata['w']`` and ``graph.ndata['deg']``.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        Y : paddle.Tensor
            The feature to compute attention.
        etas : float
            The weight of each dimension. If ``None``, then weight of each dimension is 1.
            Default: ``None``.

        Returns
        -------
        DGLGraph
            The graph.
        """
        if etas is not None:
            Y = Y * etas.view(-1)
        graph.srcdata["h"] = Y
        graph.srcdata["h_norm"] = (Y**2).sum(axis=-1)
        graph.apply_edges(fn.u_dot_v("h", "h", "dot_"))
        graph.apply_edges(fn.u_add_v("h_norm", "h_norm", "norm_"))
        graph.edata["dot_"] = graph.edata["dot_"].view(-1)
        graph.edata["norm_"] = graph.edata["norm_"].view(-1)
        graph.edata["w"] = graph.edata["norm_"] - 2 * graph.edata["dot_"]
        self.reweighting(graph)
        graph.update_all(fn.copy_e("w", "m"), fn.sum("m", "deg"))
        graph.ndata["deg"] = graph.ndata["deg"].view(-1)
        if self.attn_dropout > 0:
            graph.edata["w"] = paddle.nn.functional.dropout(
                x=graph.edata["w"], p=self.attn_dropout, training=self.training
            )
        return graph


def normalized_AX(graph, X):
    """Y = D^{-1/2}AD^{-1/2}X"""
    Y = D_power_X(graph, X, -0.5)
    Y = AX(graph, Y)
    Y = D_power_X(graph, Y, -0.5)
    return Y


def AX(graph, X):
    """Y = AX"""
    graph.srcdata["h"] = X
    graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
    Y = graph.dstdata["h"]
    return Y


def D_power_X(graph, X, power):
    """Y = D^{power}X"""
    degs = graph.ndata["deg"]
    norm = paddle.pow(x=degs, y=power)
    Y = X * norm.view(X.shape[0], 1)
    return Y


def D_power_bias_X(graph, X, power, coeff, bias):
    """Y = (coeff*D + bias*I)^{power} X"""
    degs = graph.ndata["deg"]
    degs = coeff * degs + bias
    norm = paddle.pow(x=degs, y=power)
    Y = X * norm.view(X.shape[0], 1)
    return Y


class TWIRLSUnfoldingAndAttention(paddle.nn.Layer):
    """

    Description
    -----------
    Combine propagation and attention together.

    Parameters
    ----------
    d : int
        Size of graph feature.
    alp : float
        Step size. :math:`\\alpha` in ther paper.
    lam : int
        Coefficient of graph smooth term. :math:`\\lambda` in ther paper.
    prop_step : int
        Number of propagation steps
    attn_aft : int
        Where to put attention layer. i.e. number of propagation steps before attention.
        If set to ``-1``, then no attention.
    tau : float
        The lower thresholding parameter. Correspond to :math:`\\tau` in the paper.
    T : float
        The upper thresholding parameter. Correspond to :math:`T` in the paper.
    p : float
        Correspond to :math:`\\rho` in the paper..
    use_eta : bool
        If `True`, learn a weight vector for each dimension when doing attention.
    init_att : bool
        If ``True``, add an extra attention layer before propagation.
    attn_dropout : float
        the dropout rate of attention value. Default: ``0.0``.
    precond : bool
        If ``True``, use pre-conditioned & reparameterized version propagation (eq.28), else use
        normalized laplacian (eq.30).

    Example
    -------
    >>> import dgl
    >>> from dgl.nn import TWIRLSUnfoldingAndAttention
    >>> import paddle as th

    >>> g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3])).add_self_loop()
    >>> feat = th.ones(6,5)
    >>> prop = TWIRLSUnfoldingAndAttention(10, 1, 1, prop_step=3)
    >>> res = prop(g,feat)
    >>> res
    tensor([[2.5000, 2.5000, 2.5000, 2.5000, 2.5000],
            [2.5000, 2.5000, 2.5000, 2.5000, 2.5000],
            [2.5000, 2.5000, 2.5000, 2.5000, 2.5000],
            [3.7656, 3.7656, 3.7656, 3.7656, 3.7656],
            [2.5217, 2.5217, 2.5217, 2.5217, 2.5217],
            [4.0000, 4.0000, 4.0000, 4.0000, 4.0000]])

    """

    def __init__(
        self,
        d,
        alp,
        lam,
        prop_step,
        attn_aft=-1,
        tau=0.2,
        T=-1,
        p=1,
        use_eta=False,
        init_att=False,
        attn_dropout=0,
        precond=True,
    ):
        super().__init__()
        self.d = d
        self.alp = alp if alp > 0 else 1 / (lam + 1)
        self.lam = lam
        self.tau = tau
        self.p = p
        self.prop_step = prop_step
        self.attn_aft = attn_aft
        self.use_eta = use_eta
        self.init_att = init_att
        prop_method = Propagate if precond else PropagateNoPrecond
        self.prop_layers = paddle.nn.LayerList(
            sublayers=[prop_method() for _ in range(prop_step)]
        )
        self.init_attn = (
            Attention(tau, T, p, attn_dropout) if self.init_att else None
        )
        self.attn_layer = (
            Attention(tau, T, p, attn_dropout) if self.attn_aft >= 0 else None
        )
        self.etas = (
            paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.ones(shape=d)
            )
            if self.use_eta
            else None
        )

    def forward(self, g, X):
        """

        Description
        -----------
        Compute forward pass of propagation & attention.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        X : paddle.Tensor
            Init features.

        Returns
        -------
        paddle.Tensor
            The graph.
        """
        Y = X
        g.edata["w"] = paddle.ones(shape=[g.num_edges(), 1])
        g.ndata["deg"] = g.in_degrees().to(X)
        if self.init_att:
            g = self.init_attn(g, Y, self.etas)
        for k, layer in enumerate(self.prop_layers):
            Y = layer(g, Y, X, self.alp, self.lam)
            if k == self.attn_aft - 1:
                g = self.attn_layer(g, Y, self.etas)
        return Y


class MLP(paddle.nn.Layer):
    """

    Description
    -----------
    An MLP module.

    Parameters
    ----------
    input_d : int
        Number of input features.
    output_d : int
        Number of output features.
    hidden_d : int
        Size of hidden layers.
    num_layers : int
        Number of mlp layers.
    dropout : float
        The dropout rate in mlp layers.
    norm : str
        The type of norm layers inside mlp layers. Can be ``'batch'``, ``'layer'`` or ``'none'``.
    init_activate : bool
        If add a relu at the beginning.

    """

    def __init__(
        self,
        input_d,
        hidden_d,
        output_d,
        num_layers,
        dropout,
        norm,
        init_activate,
    ):
        super().__init__()
        self.init_activate = init_activate
        self.norm = norm
        self.dropout = dropout
        self.layers = paddle.nn.LayerList(sublayers=[])
        if num_layers == 1:
            self.layers.append(
                paddle.nn.Linear(in_features=input_d, out_features=output_d)
            )
        elif num_layers > 1:
            self.layers.append(
                paddle.nn.Linear(in_features=input_d, out_features=hidden_d)
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    paddle.nn.Linear(
                        in_features=hidden_d, out_features=hidden_d
                    )
                )
            self.layers.append(
                paddle.nn.Linear(in_features=hidden_d, out_features=output_d)
            )
        self.norm_cnt = num_layers - 1 + int(init_activate)
        if norm == "batch":
            self.norms = paddle.nn.LayerList(
                sublayers=[
                    paddle.nn.BatchNorm1D(num_features=hidden_d)
                    for _ in range(self.norm_cnt)
                ]
            )
        elif norm == "layer":
            self.norms = paddle.nn.LayerList(
                sublayers=[
                    paddle.nn.LayerNorm(normalized_shape=hidden_d)
                    for _ in range(self.norm_cnt)
                ]
            )
        self.reset_params()

    def reset_params(self):
        """reset mlp parameters using xavier_norm"""
        for layer in self.layers:
            init_XavierNormal = paddle.nn.initializer.XavierNormal()
            init_XavierNormal(layer.weight.data)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(layer.bias.data)

    def activate(self, x):
        """do normlaization and activation"""
        if self.norm != "none":
            x = self.norms[self.cur_norm_idx](x)
            self.cur_norm_idx += 1
        x = paddle.nn.functional.relu(x=x)
        x = paddle.nn.functional.dropout(
            x=x, p=self.dropout, training=self.training
        )
        return x

    def forward(self, x):
        """The forward pass of mlp."""
        self.cur_norm_idx = 0
        if self.init_activate:
            x = self.activate(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activate(x)
        return x
