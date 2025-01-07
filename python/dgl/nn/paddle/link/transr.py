"""TransR."""

import paddle


class TransR(paddle.nn.Layer):
    """Similarity measure from
    `Learning entity and relation embeddings for knowledge graph completion
    <https://ojs.aaai.org/index.php/AAAI/article/view/9491>`__

    Mathematically, it is defined as follows:

    .. math::

        - {\\| M_r h + r - M_r t \\|}_p

    where :math:`M_r` is a relation-specific projection matrix, :math:`h` is the
    head embedding, :math:`r` is the relation embedding, and :math:`t` is the tail embedding.

    Parameters
    ----------
    num_rels : int
        Number of relation types.
    rfeats : int
        Relation embedding size.
    nfeats : int
        Entity embedding size.
    p : int, optional
        The p to use for Lp norm, which can be 1 or 2.

    Attributes
    ----------
    rel_emb : paddle.nn.Embedding
        The learnable relation type embedding.
    rel_project : paddle.nn.Embedding
        The learnable relation-type-specific projection.

    Examples
    --------
    >>> import dgl
    >>> import paddle as th
    >>> from dgl.nn import TransR

    >>> # input features
    >>> num_nodes = 10
    >>> num_edges = 30
    >>> num_rels = 3
    >>> feats = 4

    >>> scorer = TransR(num_rels=num_rels, rfeats=2, nfeats=feats)
    >>> g = dgl.rand_graph(num_nodes=num_nodes, num_edges=num_edges)
    >>> src, dst = g.edges()
    >>> h = th.randn(num_nodes, feats)
    >>> h_head = h[src]
    >>> h_tail = h[dst]
    >>> # Randomly initialize edge relation types for demonstration
    >>> rels = th.randint(low=0, high=num_rels, size=(num_edges,))
    >>> scorer(h_head, h_tail, rels).shape
    (30,)
    """

    def __init__(self, num_rels, rfeats, nfeats, p=1):
        super(TransR, self).__init__()
        self.rel_emb = paddle.nn.Embedding(
            num_embeddings=num_rels, embedding_dim=rfeats
        )
        self.rel_project = paddle.nn.Embedding(
            num_embeddings=num_rels, embedding_dim=nfeats * rfeats
        )
        self.rfeats = rfeats
        self.nfeats = nfeats
        self.p = p

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.
        """
        self.rel_emb.reset_parameters()
        self.rel_project.reset_parameters()

    def forward(self, h_head, h_tail, rels):
        """
        Score triples.

        Parameters
        ----------
        h_head : paddle.Tensor
            Head entity features. The tensor is of shape :math:`(E, D)`, where
            :math:`E` is the number of triples, and :math:`D` is the feature size.
        h_tail : paddle.Tensor
            Tail entity features. The tensor is of shape :math:`(E, D)`, where
            :math:`E` is the number of triples, and :math:`D` is the feature size.
        rels : paddle.Tensor
            Relation types. It is a LongTensor of shape :math:`(E)`, where
            :math:`E` is the number of triples.

        Returns
        -------
        paddle.Tensor
            The triple scores. The tensor is of shape :math:`(E)`.
        """
        h_rel = self.rel_emb(rels)
        proj_rel = self.rel_project(rels).reshape(-1, self.nfeats, self.rfeats)
        h_head = (h_head.unsqueeze(axis=1) @ proj_rel).squeeze(axis=1)
        h_tail = (h_tail.unsqueeze(axis=1) @ proj_rel).squeeze(axis=1)
        return -paddle.linalg.norm(x=h_head + h_rel - h_tail, p=self.p, axis=-1)
