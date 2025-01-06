"""EGT Layer"""
import paddle


class EGTLayer(paddle.nn.Layer):
    """EGTLayer for Edge-augmented Graph Transformer (EGT), as introduced in
    `Global Self-Attention as a Replacement for Graph Convolution
    Reference `<https://arxiv.org/pdf/2108.03348.pdf>`_

    Parameters
    ----------
    feat_size : int
        Node feature size.
    edge_feat_size : int
        Edge feature size.
    num_heads : int
        Number of attention heads, by which :attr: `feat_size` is divisible.
    num_virtual_nodes : int
        Number of virtual nodes.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.0.
    activation : callable activation layer, optional
        Activation function. Default: nn.ELU().
    edge_update : bool, optional
        Whether to update the edge embedding. Default: True.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import EGTLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> feat_size, edge_feat_size = 128, 32
    >>> nfeat = th.rand(batch_size, num_nodes, feat_size)
    >>> efeat = th.rand(batch_size, num_nodes, num_nodes, edge_feat_size)
    >>> net = EGTLayer(
            feat_size=feat_size,
            edge_feat_size=edge_feat_size,
            num_heads=8,
            num_virtual_nodes=4,
        )
    >>> out = net(nfeat, efeat)
    """

    def __init__(
        self,
        feat_size,
        edge_feat_size,
        num_heads,
        num_virtual_nodes,
        dropout=0,
        attn_dropout=0,
        activation=paddle.nn.ELU(),
        edge_update=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_virtual_nodes = num_virtual_nodes
        self.edge_update = edge_update
        assert feat_size % num_heads == 0, "feat_size must be divisible by num_heads"
        self.dot_dim = feat_size // num_heads
        self.mha_ln_h = paddle.nn.LayerNorm(normalized_shape=feat_size)
        self.mha_ln_e = paddle.nn.LayerNorm(normalized_shape=edge_feat_size)
        self.edge_input = paddle.nn.Linear(in_features=edge_feat_size, out_features=num_heads)
        self.qkv_proj = paddle.nn.Linear(in_features=feat_size, out_features=feat_size * 3)
        self.gate = paddle.nn.Linear(in_features=edge_feat_size, out_features=num_heads)
        self.attn_dropout = paddle.nn.Dropout(p=attn_dropout)
        self.node_output = paddle.nn.Linear(in_features=feat_size, out_features=feat_size)
        self.mha_dropout_h = paddle.nn.Dropout(p=dropout)
        self.node_ffn = paddle.nn.Sequential(
            paddle.nn.LayerNorm(normalized_shape=feat_size),
            paddle.nn.Linear(in_features=feat_size, out_features=feat_size),
            activation,
            paddle.nn.Linear(in_features=feat_size, out_features=feat_size),
            paddle.nn.Dropout(p=dropout),
        )
        if self.edge_update:
            self.edge_output = paddle.nn.Linear(in_features=num_heads, out_features=edge_feat_size)
            self.mha_dropout_e = paddle.nn.Dropout(p=dropout)
            self.edge_ffn = paddle.nn.Sequential(
                paddle.nn.LayerNorm(normalized_shape=edge_feat_size),
                paddle.nn.Linear(in_features=edge_feat_size, out_features=edge_feat_size),
                activation,
                paddle.nn.Linear(in_features=edge_feat_size, out_features=edge_feat_size),
                paddle.nn.Dropout(p=dropout),
            )

    def forward(self, nfeat, efeat, mask=None):
        """Forward computation. Note: :attr:`nfeat` and :attr:`efeat` should be
        padded with embedding of virtual nodes if :attr:`num_virtual_nodes` > 0,
        while :attr:`mask` should be padded with `0` values for virtual nodes.
        The padding should be put at the beginning.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where N
            is the sum of the maximum number of nodes and the number of virtual nodes.
        efeat : torch.Tensor
            Edge embedding used for attention computation and self update.
            Shape: (batch_size, N, N, :attr:`edge_feat_size`).
        mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where valid positions are indicated by `0` and
            invalid positions are indicated by `-inf`.
            Shape: (batch_size, N, N). Default: None.

        Returns
        -------
        nfeat : torch.Tensor
            The output node embedding. Shape: (batch_size, N, :attr:`feat_size`).
        efeat : torch.Tensor, optional
            The output edge embedding. Shape: (batch_size, N, N, :attr:`edge_feat_size`).
            It is returned only if :attr:`edge_update` is True.
        """
        nfeat_r1 = nfeat
        efeat_r1 = efeat
        nfeat_ln = self.mha_ln_h(nfeat)
        efeat_ln = self.mha_ln_e(efeat)
        qkv = self.qkv_proj(nfeat_ln)
        e_bias = self.edge_input(efeat_ln)
        gates = self.gate(efeat_ln)
        bsz, N, _ = tuple(qkv.shape)
        q_h, k_h, v_h = qkv.view(bsz, N, -1, self.num_heads).split(self.dot_dim, dim=2)
        attn_hat = paddle.einsum("bldh,bmdh->blmh", q_h, k_h)
        attn_hat = attn_hat.clip(min=-5, max=5) + e_bias
        if mask is None:
            gates = paddle.nn.functional.sigmoid(x=gates)
            attn_tild = paddle.nn.functional.softmax(x=attn_hat, axis=2) * gates
        else:
            gates = paddle.nn.functional.sigmoid(x=gates + mask.unsqueeze(axis=-1))
            attn_tild = paddle.nn.functional.softmax(x=attn_hat + mask.unsqueeze(axis=-1), axis=2) * gates
        attn_tild = self.attn_dropout(attn_tild)
        v_attn = paddle.einsum("blmh,bmkh->blkh", attn_tild, v_h)
        degrees = paddle.sum(x=gates, axis=2, keepdim=True)
        degree_scalers = paddle.log(x=1 + degrees)
        degree_scalers[:, : self.num_virtual_nodes] = 1.0
        v_attn = v_attn * degree_scalers
        v_attn = v_attn.reshape(bsz, N, self.num_heads * self.dot_dim)
        nfeat = self.node_output(v_attn)
        nfeat = self.mha_dropout_h(nfeat)
        nfeat.add_(y=paddle.to_tensor(nfeat_r1))
        nfeat_r2 = nfeat
        nfeat = self.node_ffn(nfeat)
        nfeat.add_(y=paddle.to_tensor(nfeat_r2))
        if self.edge_update:
            efeat = self.edge_output(attn_hat)
            efeat = self.mha_dropout_e(efeat)
            efeat.add_(y=paddle.to_tensor(efeat_r1))
            efeat_r2 = efeat
            efeat = self.edge_ffn(efeat)
            efeat.add_(y=paddle.to_tensor(efeat_r2))
            return nfeat, efeat
        return nfeat
