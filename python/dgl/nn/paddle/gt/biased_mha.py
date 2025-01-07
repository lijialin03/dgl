"""Biased Multi-head Attention"""

import paddle

from ....backend.paddle.utils import transpose_aux_func


class BiasedMHA(paddle.nn.Layer):
    """Dense Multi-Head Attention Module with Graph Attention Bias.

    Compute attention between nodes with attention bias obtained from graph
    structures, as introduced in `Do Transformers Really Perform Bad for
    Graph Representation? <https://arxiv.org/pdf/2106.05234>`__

    .. math::

        \\text{Attn}=\\text{softmax}(\\dfrac{QK^T}{\\sqrt{d}} \\circ b)

    :math:`Q` and :math:`K` are feature representations of nodes. :math:`d`
    is the corresponding :attr:`feat_size`. :math:`b` is attention bias, which
    can be additive or multiplicative according to the operator :math:`\\circ`.

    Parameters
    ----------
    feat_size : int
        Feature size.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    bias : bool, optional
        If True, it uses bias for linear projection. Default: True.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    attn_drop : float, optional
        Dropout probability on attention weights. Defalt: 0.1.

    Examples
    --------
    >>> import paddle as th
    >>> from dgl.nn import BiasedMHA

    >>> ndata = th.rand(16, 100, 512)
    >>> bias = th.rand(16, 100, 100, 8)
    >>> net = BiasedMHA(feat_size=512, num_heads=8)
    >>> out = net(ndata, bias)
    """

    def __init__(
        self,
        feat_size,
        num_heads,
        bias=True,
        attn_bias_type="add",
        attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
            self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.attn_bias_type = attn_bias_type
        self.q_proj = paddle.nn.Linear(
            in_features=feat_size, out_features=feat_size, bias_attr=bias
        )
        self.k_proj = paddle.nn.Linear(
            in_features=feat_size, out_features=feat_size, bias_attr=bias
        )
        self.v_proj = paddle.nn.Linear(
            in_features=feat_size, out_features=feat_size, bias_attr=bias
        )
        self.out_proj = paddle.nn.Linear(
            in_features=feat_size, out_features=feat_size, bias_attr=bias
        )
        self.dropout = paddle.nn.Dropout(p=attn_drop)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters of projection matrices, the same settings as in
        the original implementation of the paper.
        """
        init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=2**-0.5)
        init_XavierUniform(self.q_proj.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=2**-0.5)
        init_XavierUniform(self.k_proj.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=2**-0.5)
        init_XavierUniform(self.v_proj.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.out_proj.weight)
        if self.out_proj.bias is not None:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.out_proj.bias)

    def forward(self, ndata, attn_bias=None, attn_mask=None):
        """Forward computation.

        Parameters
        ----------
        ndata : paddle.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where
            N is the maximum number of nodes.
        attn_bias : paddle.Tensor, optional
            The attention bias used for attention modification. Shape:
            (batch_size, N, N, :attr:`num_heads`).
        attn_mask : paddle.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where invalid positions are indicated by `True` values.
            Shape: (batch_size, N, N). Note: For rows corresponding to
            unexisting nodes, make sure at least one entry is set to `False` to
            prevent obtaining NaNs with softmax.

        Returns
        -------
        y : paddle.Tensor
            The output tensor. Shape: (batch_size, N, :attr:`feat_size`)
        """
        q_h = self.q_proj(ndata).transpose(
            perm=transpose_aux_func(self.q_proj(ndata).ndim, 0, 1)
        )
        k_h = self.k_proj(ndata).transpose(
            perm=transpose_aux_func(self.k_proj(ndata).ndim, 0, 1)
        )
        v_h = self.v_proj(ndata).transpose(
            perm=transpose_aux_func(self.v_proj(ndata).ndim, 0, 1)
        )
        bsz, N, _ = tuple(ndata.shape)
        q_h = (
            q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(
                perm=transpose_aux_func(
                    q_h.reshape(N, bsz * self.num_heads, self.head_dim).ndim,
                    0,
                    1,
                )
            )
            * self.scaling
        )
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(
            perm=[1, 2, 0]
        )
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(
            perm=transpose_aux_func(
                v_h.reshape(N, bsz * self.num_heads, self.head_dim).ndim, 0, 1
            )
        )
        attn_weights = (
            paddle.bmm(x=q_h, y=k_h)
            .transpose(
                perm=transpose_aux_func(paddle.bmm(x=q_h, y=k_h).ndim, 0, 2)
            )
            .reshape(N, N, bsz, self.num_heads)
            .transpose(
                perm=transpose_aux_func(
                    paddle.bmm(x=q_h, y=k_h)
                    .transpose(
                        perm=transpose_aux_func(
                            paddle.bmm(x=q_h, y=k_h).ndim, 0, 2
                        )
                    )
                    .reshape(N, N, bsz, self.num_heads)
                    .ndim,
                    0,
                    2,
                )
            )
        )
        if attn_bias is not None:
            if self.attn_bias_type == "add":
                attn_weights += attn_bias
            else:
                attn_weights *= attn_bias
        if attn_mask is not None:
            attn_weights[attn_mask.to("bool")] = float("-inf")
        attn_weights = paddle.nn.functional.softmax(
            x=attn_weights.transpose(
                perm=transpose_aux_func(attn_weights.ndim, 0, 2)
            )
            .reshape(N, N, bsz * self.num_heads)
            .transpose(
                perm=transpose_aux_func(
                    attn_weights.transpose(
                        perm=transpose_aux_func(attn_weights.ndim, 0, 2)
                    )
                    .reshape(N, N, bsz * self.num_heads)
                    .ndim,
                    0,
                    2,
                )
            ),
            axis=2,
        )
        attn_weights = self.dropout(attn_weights)
        attn = paddle.bmm(x=attn_weights, y=v_h).transpose(
            perm=transpose_aux_func(
                paddle.bmm(x=attn_weights, y=v_h).ndim, 0, 1
            )
        )
        attn = self.out_proj(
            attn.reshape(N, bsz, self.feat_size).transpose(
                perm=transpose_aux_func(
                    attn.reshape(N, bsz, self.feat_size).ndim, 0, 1
                )
            )
        )
        return attn
