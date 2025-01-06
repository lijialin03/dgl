"""Network Embedding NN Modules"""
import random

import paddle
from tqdm.auto import trange

from ...base import NID
from ...convert import to_heterogeneous
from ...convert import to_homogeneous
from ...random import choice
from ...sampling import random_walk

__all__ = ["DeepWalk", "MetaPath2Vec"]


class DeepWalk(paddle.nn.Layer):
    """DeepWalk module from `DeepWalk: Online Learning of Social Representations
    <https://arxiv.org/abs/1403.6652>`__

    For a graph, it learns the node representations from scratch by maximizing the similarity of
    node pairs that are nearby (positive node pairs) and minimizing the similarity of other
    random node pairs (negative node pairs).

    Parameters
    ----------
    g : DGLGraph
        Graph for learning node embeddings
    emb_dim : int, optional
        Size of each embedding vector. Default: 128
    walk_length : int, optional
        Number of nodes in a random walk sequence. Default: 40
    window_size : int, optional
        In a random walk :attr:`w`, a node :attr:`w[j]` is considered close to a node
        :attr:`w[i]` if :attr:`i - window_size <= j <= i + window_size`. Default: 5
    neg_weight : float, optional
        Weight of the loss term for negative samples in the total loss. Default: 1.0
    negative_size : int, optional
        Number of negative samples to use for each positive sample. Default: 5
    fast_neg : bool, optional
        If True, it samples negative node pairs within a batch of random walks. Default: True
    sparse : bool, optional
        If True, gradients with respect to the learnable weights will be sparse.
        Default: True

    Attributes
    ----------
    node_embed : nn.Embedding
        Embedding table of the nodes

    Examples
    --------

    >>> import torch
    >>> from dgl.data import CoraGraphDataset
    >>> from dgl.nn import DeepWalk
    >>> from torch.optim import SparseAdam
    >>> from torch.utils.data import DataLoader
    >>> from sklearn.linear_model import LogisticRegression

    >>> dataset = CoraGraphDataset()
    >>> g = dataset[0]
    >>> model = DeepWalk(g)
    >>> dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=128,
    ...                         shuffle=True, collate_fn=model.sample)
    >>> optimizer = SparseAdam(model.parameters(), lr=0.01)
    >>> num_epochs = 5

    >>> for epoch in range(num_epochs):
    ...     for batch_walk in dataloader:
    ...         loss = model(batch_walk)
    ...         optimizer.zero_grad()
    ...         loss.backward()
    ...         optimizer.step()

    >>> train_mask = g.ndata['train_mask']
    >>> test_mask = g.ndata['test_mask']
    >>> X = model.node_embed.weight.detach()
    >>> y = g.ndata['label']
    >>> clf = LogisticRegression().fit(X[train_mask].numpy(), y[train_mask].numpy())
    >>> clf.score(X[test_mask].numpy(), y[test_mask].numpy())
    """

    def __init__(
        self,
        g,
        emb_dim=128,
        walk_length=40,
        window_size=5,
        neg_weight=1,
        negative_size=5,
        fast_neg=True,
        sparse=True,
    ):
        super().__init__()
        assert (
            walk_length >= window_size + 1
        ), f"Expect walk_length >= window_size + 1, got {walk_length} and {window_size + 1}"
        self.g = g
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.walk_length = walk_length
        self.neg_weight = neg_weight
        self.negative_size = negative_size
        self.fast_neg = fast_neg
        num_nodes = g.num_nodes()
        self.node_embed = paddle.nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_dim, sparse=sparse)
        self.context_embed = paddle.nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_dim, sparse=sparse)
        self.reset_parameters()
        if not fast_neg:
            neg_prob = g.out_degrees().pow(y=0.75)
            self.neg_prob = neg_prob / neg_prob.sum()
        idx_list_src = []
        idx_list_dst = []
        for i in range(walk_length):
            for j in range(max(0, i - window_size), i):
                idx_list_src.append(j)
                idx_list_dst.append(i)
            for j in range(i + 1, min(walk_length, i + 1 + window_size)):
                idx_list_src.append(j)
                idx_list_dst.append(i)
        self.idx_list_src = paddle.to_tensor(data=idx_list_src, dtype="int64")
        self.idx_list_dst = paddle.to_tensor(data=idx_list_dst, dtype="int64")

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        init_range = 1.0 / self.emb_dim
        init_Uniform = paddle.nn.initializer.Uniform(low=-init_range, high=init_range)
        init_Uniform(self.node_embed.weight.data)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.context_embed.weight.data)

    def sample(self, indices):
        """Sample random walks

        Parameters
        ----------
        indices : torch.Tensor
            Nodes from which we perform random walk

        Returns
        -------
        torch.Tensor
            Random walks in the form of node ID sequences. The Tensor
            is of shape :attr:`(len(indices), walk_length)`.
        """
        return random_walk(self.g, indices, length=self.walk_length - 1)[0]

    def forward(self, batch_walk):
        """Compute the loss for the batch of random walks

        Parameters
        ----------
        batch_walk : torch.Tensor
            Random walks in the form of node ID sequences. The Tensor
            is of shape :attr:`(batch_size, walk_length)`.

        Returns
        -------
        torch.Tensor
            Loss value
        """
        batch_size = len(batch_walk)
        device = batch_walk.place
        batch_node_embed = self.node_embed(batch_walk).view(-1, self.emb_dim)
        batch_context_embed = self.context_embed(batch_walk).view(-1, self.emb_dim)
        batch_idx_list_offset = paddle.arange(end=batch_size) * self.walk_length
        batch_idx_list_offset = batch_idx_list_offset.unsqueeze(axis=1)
        idx_list_src = batch_idx_list_offset + self.idx_list_src.unsqueeze(axis=0)
        idx_list_dst = batch_idx_list_offset + self.idx_list_dst.unsqueeze(axis=0)
        idx_list_src = idx_list_src.view(-1).to(device)
        idx_list_dst = idx_list_dst.view(-1).to(device)
        pos_src_emb = batch_node_embed[idx_list_src]
        pos_dst_emb = batch_context_embed[idx_list_dst]
        neg_idx_list_src = idx_list_dst.unsqueeze(axis=1) + paddle.zeros(shape=self.negative_size).unsqueeze(axis=0).to(
            device
        )
        neg_idx_list_src = neg_idx_list_src.view(-1)
        neg_src_emb = batch_node_embed[neg_idx_list_src.astype(dtype="int64")]
        if self.fast_neg:
            neg_idx_list_dst = list(range(batch_size * self.walk_length)) * (self.negative_size * self.window_size * 2)
            random.shuffle(neg_idx_list_dst)
            neg_idx_list_dst = neg_idx_list_dst[: len(neg_idx_list_src)]
            neg_idx_list_dst = paddle.to_tensor(data=neg_idx_list_dst, dtype="int64").to(device)
            neg_dst_emb = batch_context_embed[neg_idx_list_dst]
        else:
            neg_dst = choice(self.g.num_nodes(), size=len(neg_src_emb), prob=self.neg_prob)
            neg_dst_emb = self.context_embed(neg_dst.to(device))
        pos_score = paddle.sum(
            x=paddle.multiply(x=pos_src_emb, y=paddle.to_tensor(pos_dst_emb)),
            axis=1,
        )
        pos_score = paddle.clip(x=pos_score, max=6, min=-6)
        pos_score = paddle.mean(x=-paddle.nn.functional.log_sigmoid(x=pos_score))
        neg_score = paddle.sum(
            x=paddle.multiply(x=neg_src_emb, y=paddle.to_tensor(neg_dst_emb)),
            axis=1,
        )
        neg_score = paddle.clip(x=neg_score, max=6, min=-6)
        neg_score = (
            paddle.mean(x=-paddle.nn.functional.log_sigmoid(x=-neg_score)) * self.negative_size * self.neg_weight
        )
        return paddle.mean(x=pos_score + neg_score)


class MetaPath2Vec(paddle.nn.Layer):
    """metapath2vec module from `metapath2vec: Scalable Representation Learning for
    Heterogeneous Networks <https://dl.acm.org/doi/pdf/10.1145/3097983.3098036>`__

    To achieve efficient optimization, we leverage the negative sampling technique for the
    training process. Repeatedly for each node in meta-path, we treat it as the center node
    and sample nearby positive nodes within context size and draw negative samples among all
    types of nodes from all meta-paths. Then we can use the center-context paired nodes and
    context-negative paired nodes to update the network.

    Parameters
    ----------
    g : DGLGraph
        Graph for learning node embeddings. Two different canonical edge types
        :attr:`(utype, etype, vtype)` are not allowed to have same :attr:`etype`.
    metapath : list[str]
        A sequence of edge types in the form of a string. It defines a new edge type by composing
        multiple edge types in order. Note that the start node type and the end one are commonly
        the same.
    window_size : int
        In a random walk :attr:`w`, a node :attr:`w[j]` is considered close to a node
        :attr:`w[i]` if :attr:`i - window_size <= j <= i + window_size`.
    emb_dim : int, optional
        Size of each embedding vector. Default: 128
    negative_size : int, optional
        Number of negative samples to use for each positive sample. Default: 5
    sparse : bool, optional
        If True, gradients with respect to the learnable weights will be sparse.
        Default: True

    Attributes
    ----------
    node_embed : nn.Embedding
        Embedding table of all nodes
    local_to_global_nid : dict[str, list]
        Mapping from type-specific node IDs to global node IDs

    Examples
    --------

    >>> import torch
    >>> import dgl
    >>> from torch.optim import SparseAdam
    >>> from torch.utils.data import DataLoader
    >>> from dgl.nn.pytorch import MetaPath2Vec

    >>> # Define a model
    >>> g = dgl.heterograph({
    ...     ('user', 'uc', 'company'): dgl.rand_graph(100, 1000).edges(),
    ...     ('company', 'cp', 'product'): dgl.rand_graph(100, 1000).edges(),
    ...     ('company', 'cu', 'user'): dgl.rand_graph(100, 1000).edges(),
    ...     ('product', 'pc', 'company'): dgl.rand_graph(100, 1000).edges()
    ... })
    >>> model = MetaPath2Vec(g, ['uc', 'cu'], window_size=1)

    >>> # Use the source node type of etype 'uc'
    >>> dataloader = DataLoader(torch.arange(g.num_nodes('user')), batch_size=128,
    ...                         shuffle=True, collate_fn=model.sample)
    >>> optimizer = SparseAdam(model.parameters(), lr=0.025)

    >>> for (pos_u, pos_v, neg_v) in dataloader:
    ...     loss = model(pos_u, pos_v, neg_v)
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()

    >>> # Get the embeddings of all user nodes
    >>> user_nids = torch.LongTensor(model.local_to_global_nid['user'])
    >>> user_emb = model.node_embed(user_nids)
    """

    def __init__(
        self,
        g,
        metapath,
        window_size,
        emb_dim=128,
        negative_size=5,
        sparse=True,
    ):
        super().__init__()
        assert (
            len(metapath) + 1 >= window_size
        ), f"Expect len(metapath) >= window_size - 1, got {metapath} and {window_size}"
        self.hg = g
        self.emb_dim = emb_dim
        self.metapath = metapath
        self.window_size = window_size
        self.negative_size = negative_size
        src_type, _, _ = g.to_canonical_etype(metapath[0])
        node_metapath = [src_type]
        for etype in metapath:
            _, _, dst_type = g.to_canonical_etype(etype)
            node_metapath.append(dst_type)
        self.node_metapath = node_metapath
        g = to_homogeneous(g)
        hg = to_heterogeneous(g, self.hg.ntypes, self.hg.etypes)
        local_to_global_nid = hg.ndata[NID]
        for key, val in local_to_global_nid.items():
            local_to_global_nid[key] = list(val.cpu().numpy())
        self.local_to_global_nid = local_to_global_nid
        num_nodes_total = hg.num_nodes()
        node_frequency = paddle.zeros(shape=num_nodes_total)
        for idx in trange(hg.num_nodes(node_metapath[0])):
            traces, _ = random_walk(g=hg, nodes=[idx], metapath=metapath)
            for tr in traces.cpu().numpy():
                tr_nids = [self.local_to_global_nid[node_metapath[i]][tr[i]] for i in range(len(tr))]
                node_frequency[paddle.to_tensor(data=tr_nids, dtype="int64")] += 1
        neg_prob = node_frequency.pow(y=0.75)
        self.neg_prob = neg_prob / neg_prob.sum()
        self.node_embed = paddle.nn.Embedding(
            num_embeddings=num_nodes_total,
            embedding_dim=self.emb_dim,
            sparse=sparse,
        )
        self.context_embed = paddle.nn.Embedding(
            num_embeddings=num_nodes_total,
            embedding_dim=self.emb_dim,
            sparse=sparse,
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        init_range = 1.0 / self.emb_dim
        init_Uniform = paddle.nn.initializer.Uniform(low=-init_range, high=init_range)
        init_Uniform(self.node_embed.weight.data)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.context_embed.weight.data)

    def sample(self, indices):
        """Sample positive and negative samples

        Parameters
        ----------
        indices : torch.Tensor
            Node IDs of the source node type from which we perform random walks

        Returns
        -------
        torch.Tensor
            Positive center nodes
        torch.Tensor
            Positive context nodes
        torch.Tensor
            Negative context nodes
        """
        traces, _ = random_walk(g=self.hg, nodes=indices, metapath=self.metapath)
        u_list = []
        v_list = []
        for tr in traces.cpu().numpy():
            tr_nids = [self.local_to_global_nid[self.node_metapath[i]][tr[i]] for i in range(len(tr))]
            for i, u in enumerate(tr_nids):
                for j, v in enumerate(tr_nids[max(i - self.window_size, 0) : i + self.window_size]):
                    if i == j:
                        continue
                    u_list.append(u)
                    v_list.append(v)
        neg_v = choice(
            self.hg.num_nodes(),
            size=len(u_list) * self.negative_size,
            prob=self.neg_prob,
        ).reshape(len(u_list), self.negative_size)
        return (
            paddle.to_tensor(data=u_list, dtype="int64"),
            paddle.to_tensor(data=v_list, dtype="int64"),
            neg_v,
        )

    def forward(self, pos_u, pos_v, neg_v):
        """Compute the loss for the batch of positive and negative samples

        Parameters
        ----------
        pos_u : torch.Tensor
            Positive center nodes
        pos_v : torch.Tensor
            Positive context nodes
        neg_v : torch.Tensor
            Negative context nodes

        Returns
        -------
        torch.Tensor
            Loss value
        """
        emb_u = self.node_embed(pos_u)
        emb_v = self.context_embed(pos_v)
        emb_neg_v = self.context_embed(neg_v)
        score = paddle.sum(x=paddle.multiply(x=emb_u, y=paddle.to_tensor(emb_v)), axis=1)
        score = paddle.clip(x=score, max=10, min=-10)
        score = -paddle.nn.functional.log_sigmoid(x=score)
        neg_score = paddle.bmm(x=emb_neg_v, y=emb_u.unsqueeze(axis=2)).squeeze()
        neg_score = paddle.clip(x=neg_score, max=10, min=-10)
        neg_score = -paddle.sum(x=paddle.nn.functional.log_sigmoid(x=-neg_score), axis=1)
        return paddle.mean(x=score + neg_score)
