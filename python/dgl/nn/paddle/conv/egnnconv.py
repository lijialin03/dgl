"""Torch Module for E(n) Equivariant Graph Convolutional Layer"""
import paddle

from .... import function as fn


class EGNNConv(paddle.nn.Layer):
    """Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::

        m_{ij}=\\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})

        x_i^{l+1} = x_i^l + C\\sum_{j\\in\\mathcal{N}(i)}(x_i^l-x_j^l)\\phi_x(m_{ij})

        m_i = \\sum_{j\\in\\mathcal{N}(i)} m_{ij}

        h_i^{l+1} = \\phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\\phi_e`, :math:`\\phi_h`, and
    :math:`\\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\\phi_e, \\phi_x, \\phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGNNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> node_feat, coord_feat, edge_feat = th.ones(6, 10), th.ones(6, 3), th.ones(6, 2)
    >>> conv = EGNNConv(10, 10, 10, 2)
    >>> h, x = conv(g, node_feat, coord_feat, edge_feat)
    """

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNConv, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = paddle.nn.Silu()
        self.edge_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=in_size * 2 + edge_feat_size + 1,
                out_features=hidden_size,
            ),
            act_fn,
            paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            act_fn,
        )
        self.node_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=in_size + hidden_size, out_features=hidden_size),
            act_fn,
            paddle.nn.Linear(in_features=hidden_size, out_features=out_size),
        )
        self.coord_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            act_fn,
            paddle.nn.Linear(in_features=hidden_size, out_features=1, bias_attr=False),
        )

    def message(self, edges):
        """message function for EGNN"""
        if self.edge_feat_size > 0:
            f = paddle.concat(
                x=[
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["radial"],
                    edges.data["a"],
                ],
                axis=-1,
            )
        else:
            f = paddle.concat(
                x=[edges.src["h"], edges.dst["h"], edges.data["radial"]],
                axis=-1,
            )
        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data["x_diff"]
        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        """
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        with graph.local_scope():
            graph.ndata["h"] = node_feat
            graph.ndata["x"] = coord_feat
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["radial"] = graph.edata["x_diff"].square().sum(axis=1).unsqueeze(axis=-1)
            graph.edata["x_diff"] = graph.edata["x_diff"] / (graph.edata["radial"].sqrt() + 1e-30)
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))
            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]
            h = self.node_mlp(paddle.concat(x=[node_feat, h_neigh], axis=-1))
            x = coord_feat + x_neigh
            return h, x
