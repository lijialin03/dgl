import paddle

"""Laplacian Positional Encoder"""


class LapPosEncoder(paddle.nn.Layer):
    """Laplacian Positional Encoder (LPE), as introduced in
    `GraphGPS: General Powerful Scalable Graph Transformers
    <https://arxiv.org/abs/2205.12454>`__

    This module is a learned laplacian positional encoding module using
    Transformer or DeepSet.

    Parameters
    ----------
    model_type : str
        Encoder model type for LPE, can only be "Transformer" or "DeepSet".
    num_layer : int
        Number of layers in Transformer/DeepSet Encoder.
    k : int
        Number of smallest non-trivial eigenvectors.
    dim : int
        Output size of final laplacian encoding.
    n_head : int, optional
        Number of heads in Transformer Encoder.
        Default : 1.
    batch_norm : bool, optional
        If True, apply batch normalization on raw laplacian positional
        encoding. Default : False.
    num_post_layer : int, optional
        If num_post_layer > 0, apply an MLP of ``num_post_layer`` layers after
        pooling. Default : 0.

    Example
    -------
    >>> import dgl
    >>> from dgl import LapPE
    >>> from dgl.nn import LapPosEncoder

    >>> transform = LapPE(k=5, feat_name='eigvec', eigval_name='eigval', padding=True)
    >>> g = dgl.graph(([0,1,2,3,4,2,3,1,4,0], [2,3,1,4,0,0,1,2,3,4]))
    >>> g = transform(g)
    >>> eigvals, eigvecs = g.ndata['eigval'], g.ndata['eigvec']
    >>> transformer_encoder = LapPosEncoder(
            model_type="Transformer", num_layer=3, k=5, dim=16, n_head=4
        )
    >>> pos_encoding = transformer_encoder(eigvals, eigvecs)
    >>> deepset_encoder = LapPosEncoder(
            model_type="DeepSet", num_layer=3, k=5, dim=16, num_post_layer=2
        )
    >>> pos_encoding = deepset_encoder(eigvals, eigvecs)
    """

    def __init__(
        self,
        model_type,
        num_layer,
        k,
        dim,
        n_head=1,
        batch_norm=False,
        num_post_layer=0,
    ):
        super(LapPosEncoder, self).__init__()
        self.model_type = model_type
        self.linear = paddle.nn.Linear(in_features=2, out_features=dim)
        if self.model_type == "Transformer":
            encoder_layer = paddle.nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_head, batch_first=True
            )
            self.pe_encoder = paddle.nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=num_layer
            )
        elif self.model_type == "DeepSet":
            layers = []
            if num_layer == 1:
                layers.append(paddle.nn.ReLU())
            else:
                self.linear = paddle.nn.Linear(
                    in_features=2, out_features=2 * dim
                )
                layers.append(paddle.nn.ReLU())
                for _ in range(num_layer - 2):
                    layers.append(
                        paddle.nn.Linear(
                            in_features=2 * dim, out_features=2 * dim
                        )
                    )
                    layers.append(paddle.nn.ReLU())
                layers.append(
                    paddle.nn.Linear(in_features=2 * dim, out_features=dim)
                )
                layers.append(paddle.nn.ReLU())
            self.pe_encoder = paddle.nn.Sequential(*layers)
        else:
            raise ValueError(
                f"model_type '{model_type}' is not allowed, must be 'Transformer' or 'DeepSet'."
            )
        if batch_norm:
            self.raw_norm = paddle.nn.BatchNorm1D(num_features=k)
        else:
            self.raw_norm = None
        if num_post_layer > 0:
            layers = []
            if num_post_layer == 1:
                layers.append(
                    paddle.nn.Linear(in_features=dim, out_features=dim)
                )
                layers.append(paddle.nn.ReLU())
            else:
                layers.append(
                    paddle.nn.Linear(in_features=dim, out_features=2 * dim)
                )
                layers.append(paddle.nn.ReLU())
                for _ in range(num_post_layer - 2):
                    layers.append(
                        paddle.nn.Linear(
                            in_features=2 * dim, out_features=2 * dim
                        )
                    )
                    layers.append(paddle.nn.ReLU())
                layers.append(
                    paddle.nn.Linear(in_features=2 * dim, out_features=dim)
                )
                layers.append(paddle.nn.ReLU())
            self.post_mlp = paddle.nn.Sequential(*layers)
        else:
            self.post_mlp = None

    def forward(self, eigvals, eigvecs):
        """
        Parameters
        ----------
        eigvals : Tensor
            Laplacian Eigenvalues of shape :math:`(N, k)`, k different
            eigenvalues repeat N times, can be obtained by using `LaplacianPE`.
        eigvecs : Tensor
            Laplacian Eigenvectors of shape :math:`(N, k)`, can be obtained by
            using `LaplacianPE`.

        Returns
        -------
        Tensor
            Return the laplacian positional encodings of shape :math:`(N, d)`,
            where :math:`N` is the number of nodes in the input graph,
            :math:`d` is :attr:`dim`.
        """
        pos_encoding = paddle.concat(
            x=(eigvecs.unsqueeze(axis=2), eigvals.unsqueeze(axis=2)), axis=2
        ).astype(dtype="float32")
        empty_mask = paddle.isnan(x=pos_encoding)
        pos_encoding[empty_mask] = 0
        if self.raw_norm:
            pos_encoding = self.raw_norm(pos_encoding)
        pos_encoding = self.linear(pos_encoding)
        if self.model_type == "Transformer":
            pos_encoding = self.pe_encoder(
                src=pos_encoding, src_key_padding_mask=empty_mask[:, :, 1]
            )
        else:
            pos_encoding = self.pe_encoder(pos_encoding)
        pos_encoding[empty_mask[:, :, 1]] = 0
        pos_encoding = paddle.sum(x=pos_encoding, axis=1, keepdim=False)
        if self.post_mlp:
            pos_encoding = self.post_mlp(pos_encoding)
        return pos_encoding
