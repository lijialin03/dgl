"""Heterogeneous Graph Transformer"""
import math

import paddle

from .... import function as fn
from ..linear import TypedLinear
from ..softmax import edge_softmax


class HGTConv(paddle.nn.Layer):
    """Heterogeneous graph transformer convolution from `Heterogeneous Graph Transformer
    <https://arxiv.org/abs/2003.01332>`__

    Given a graph :math:`G(V, E)` and input node features :math:`H^{(l-1)}`,
    it computes the new node features as follows:

    Compute a multi-head attention score for each edge :math:`(s, e, t)` in the graph:

    .. math::

      Attention(s, e, t) = \\text{Softmax}\\left(||_{i\\in[1,h]}ATT-head^i(s, e, t)\\right) \\\\
      ATT-head^i(s, e, t) = \\left(K^i(s)W^{ATT}_{\\phi(e)}Q^i(t)^{\\top}\\right)\\cdot
        \\frac{\\mu_{(\\tau(s),\\phi(e),\\tau(t)}}{\\sqrt{d}} \\\\
      K^i(s) = \\text{K-Linear}^i_{\\tau(s)}(H^{(l-1)}[s]) \\\\
      Q^i(t) = \\text{Q-Linear}^i_{\\tau(t)}(H^{(l-1)}[t]) \\\\

    Compute the message to send on each edge :math:`(s, e, t)`:

    .. math::

      Message(s, e, t) = ||_{i\\in[1, h]} MSG-head^i(s, e, t) \\\\
      MSG-head^i(s, e, t) = \\text{M-Linear}^i_{\\tau(s)}(H^{(l-1)}[s])W^{MSG}_{\\phi(e)} \\\\

    Send messages to target nodes :math:`t` and aggregate:

    .. math::

      \\tilde{H}^{(l)}[t] = \\sum_{\\forall s\\in \\mathcal{N}(t)}\\left( Attention(s,e,t)
      \\cdot Message(s,e,t)\\right)

    Compute new node features:

    .. math::

      H^{(l)}[t]=\\text{A-Linear}_{\\tau(t)}(\\sigma(\\tilde(H)^{(l)}[t])) + H^{(l-1)}[t]

    Parameters
    ----------
    in_size : int
        Input node feature size.
    head_size : int
        Output head size. The output node feature size is ``head_size * num_heads``.
    num_heads : int
        Number of heads. The output node feature size is ``head_size * num_heads``.
    num_ntypes : int
        Number of node types.
    num_etypes : int
        Number of edge types.
    dropout : optional, float
        Dropout rate.
    use_norm : optiona, bool
        If true, apply a layer norm on the output node feature.

    Examples
    --------
    """

    def __init__(
        self,
        in_size,
        head_size,
        num_heads,
        num_ntypes,
        num_etypes,
        dropout=0.2,
        use_norm=False,
    ):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.sqrt_d = math.sqrt(head_size)
        self.use_norm = use_norm
        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_a = TypedLinear(head_size * num_heads, head_size * num_heads, num_ntypes)
        self.relation_pri = paddle.nn.ParameterList(
            parameters=[
                paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=num_etypes))
                for i in range(num_heads)
            ]
        )
        self.relation_att = paddle.nn.LayerList(
            sublayers=[TypedLinear(head_size, head_size, num_etypes) for i in range(num_heads)]
        )
        self.relation_msg = paddle.nn.LayerList(
            sublayers=[TypedLinear(head_size, head_size, num_etypes) for i in range(num_heads)]
        )
        self.skip = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=num_ntypes))
        self.drop = paddle.nn.Dropout(p=dropout)
        if use_norm:
            self.norm = paddle.nn.LayerNorm(normalized_shape=head_size * num_heads)
        if in_size != head_size * num_heads:
            self.residual_w = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[in_size, head_size * num_heads])
            )
            init_XavierUniform = paddle.nn.initializer.XavierUniform()
            init_XavierUniform(self.residual_w)

    def forward(self, g, x, ntype, etype, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        x : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        ntype : torch.Tensor
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        etype : torch.Tensor
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether *both* the nodes and the edges of the input graph have been sorted by
            their types. Forward on pre-sorted graph may be faster. Graphs created by
            :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
        """
        self.presorted = presorted
        if g.is_block:
            x_src = x
            x_dst = x[: g.num_dst_nodes()]
            srcntype = ntype
            dstntype = ntype[: g.num_dst_nodes()]
        else:
            x_src = x
            x_dst = x
            srcntype = ntype
            dstntype = ntype
        with g.local_scope():
            k = self.linear_k(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
            q = self.linear_q(x_dst, dstntype, presorted).view(-1, self.num_heads, self.head_size)
            v = self.linear_v(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
            g.srcdata["k"] = k
            g.dstdata["q"] = q
            g.srcdata["v"] = v
            g.edata["etype"] = etype
            g.apply_edges(self.message)
            g.edata["m"] = g.edata["m"] * edge_softmax(g, g.edata["a"]).unsqueeze(-1)
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))
            h = g.dstdata["h"].view(-1, self.num_heads * self.head_size)
            h = self.drop(self.linear_a(h, dstntype, presorted))
            alpha = paddle.nn.functional.sigmoid(x=self.skip[dstntype]).unsqueeze(axis=-1)
            if tuple(x_dst.shape) != tuple(h.shape):
                h = h * alpha + x_dst @ self.residual_w * (1 - alpha)
            else:
                h = h * alpha + x_dst * (1 - alpha)
            if self.use_norm:
                h = self.norm(h)
            return h

    def message(self, edges):
        """Message function."""
        a, m = [], []
        etype = edges.data["etype"]
        k = paddle.unbind(input=edges.src["k"], axis=1)
        q = paddle.unbind(input=edges.dst["q"], axis=1)
        v = paddle.unbind(input=edges.src["v"], axis=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)
            a.append((kw * q[i]).sum(axis=-1) * self.relation_pri[i][etype] / self.sqrt_d)
            m.append(self.relation_msg[i](v[i], etype, self.presorted))
        return {"a": paddle.stack(x=a, axis=1), "m": paddle.stack(x=m, axis=1)}
