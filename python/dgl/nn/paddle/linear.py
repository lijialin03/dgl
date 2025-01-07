"""Various commonly used linear modules"""

import math

import paddle

from ...ops import gather_mm
from ...ops import segment_mm

__all__ = ["TypedLinear"]


class TypedLinear(paddle.nn.Layer):
    """Linear transformation according to types.

    For each sample of the input batch :math:`x \\in X`, apply linear transformation
    :math:`xW_t`, where :math:`t` is the type of :math:`x`.

    The module supports two regularization methods (basis-decomposition and
    block-diagonal-decomposition) proposed by "`Modeling Relational Data
    with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"

    The basis regularization decomposes :math:`W_t` by:

    .. math::

       W_t^{(l)} = \\sum_{b=1}^B a_{tb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{tb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_t` into :math:`B`
    block-diagonal matrices. We refer to :math:`B` as the number of bases:

    .. math::

       W_t^{(l)} = \\oplus_{b=1}^B Q_{tb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{tb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)\\times(d^{l}/B)}`.

    Parameters
    ----------
    in_size : int
        Input feature size.
    out_size : int
        Output feature size.
    num_types : int
        Total number of types.
    regularizer : str, optional
        Which weight regularizer to use "basis" or "bdd":

         - "basis" is short for basis-decomposition.
         - "bdd" is short for block-diagonal-decomposition.

        Default applies no regularization.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Typically smaller
        than ``num_types``.
        Default: ``None``.

    Examples
    --------

    No regularization.

    >>> from dgl.nn import TypedLinear
    >>> import paddle
    >>>
    >>> x = paddle.randn(100, 32)
    >>> x_type = paddle.randint(0, 5, (100,))
    >>> m = TypedLinear(32, 64, 5)
    >>> y = m(x, x_type)
    >>> print(y.shape)
    (100, 64)

    With basis regularization

    >>> x = paddle.randn(100, 32)
    >>> x_type = paddle.randint(0, 5, (100,))
    >>> m = TypedLinear(32, 64, 5, regularizer='basis', num_bases=4)
    >>> y = m(x, x_type)
    >>> print(y.shape)
    (100, 64)
    """

    def __init__(
        self, in_size, out_size, num_types, regularizer=None, num_bases=None
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_types = num_types
        if regularizer is None:
            self.W = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[num_types, in_size, out_size])
            )
        elif regularizer == "basis":
            if num_bases is None:
                raise ValueError(
                    'Missing "num_bases" for basis regularization.'
                )
            self.W = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[num_bases, in_size, out_size])
            )
            self.coeff = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[num_types, num_bases])
            )
            self.num_bases = num_bases
        elif regularizer == "bdd":
            if num_bases is None:
                raise ValueError('Missing "num_bases" for bdd regularization.')
            if in_size % num_bases != 0 or out_size % num_bases != 0:
                raise ValueError(
                    "Input and output sizes must be divisible by num_bases."
                )
            self.submat_in = in_size // num_bases
            self.submat_out = out_size // num_bases
            self.W = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(
                    shape=[
                        num_types,
                        num_bases * self.submat_in * self.submat_out,
                    ]
                )
            )
            self.num_bases = num_bases
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}'
            )
        self.regularizer = regularizer
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters"""
        with paddle.no_grad():
            if self.regularizer is None:
                init_Uniform = paddle.nn.initializer.Uniform(
                    low=-1 / math.sqrt(self.in_size),
                    high=1 / math.sqrt(self.in_size),
                )
                init_Uniform(self.W)
            elif self.regularizer == "basis":
                init_Uniform = paddle.nn.initializer.Uniform(
                    low=-1 / math.sqrt(self.in_size),
                    high=1 / math.sqrt(self.in_size),
                )
                init_Uniform(self.W)
                init_XavierUniform = paddle.nn.initializer.XavierUniform(
                    gain=paddle.nn.initializer.calculate_gain(
                        nonlinearity="relu"
                    )
                )
                init_XavierUniform(self.coeff)
            elif self.regularizer == "bdd":
                init_Uniform = paddle.nn.initializer.Uniform(
                    low=-1 / math.sqrt(self.submat_in),
                    high=1 / math.sqrt(self.submat_in),
                )
                init_Uniform(self.W)
            else:
                raise ValueError(
                    f'Supported regularizer options: "basis", "bdd", but got {self.regularizer}'
                )

    def get_weight(self):
        """Get type-wise weight"""
        if self.regularizer is None:
            return self.W
        elif self.regularizer == "basis":
            W = self.W.view(self.num_bases, self.in_size * self.out_size)
            return (self.coeff @ W).view(
                self.num_types, self.in_size, self.out_size
            )
        elif self.regularizer == "bdd":
            return self.W
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {self.regularizer}'
            )

    def forward(self, x, x_type, sorted_by_type=False):
        """Forward computation.

        Parameters
        ----------
        x : paddle.Tensor
            A 2D input tensor. Shape: (N, D1)
        x_type : paddle.Tensor
            A 1D integer tensor storing the type of the elements in ``x`` with one-to-one
            correspondenc. Shape: (N,)
        sorted_by_type : bool, optional
            Whether the inputs have been sorted by the types. Forward on pre-sorted inputs may
            be faster.

        Returns
        -------
        y : paddle.Tensor
            The transformed output tensor. Shape: (N, D2)
        """
        w = self.get_weight()
        if self.regularizer == "bdd":
            w = w.index_select(axis=0, index=x_type).view(
                -1, self.submat_in, self.submat_out
            )
            x = x.view(-1, 1, self.submat_in)
            return paddle.bmm(x=x, y=w).view(-1, self.out_size)
        elif sorted_by_type:
            pos_l = paddle.searchsorted(
                sorted_sequence=x_type, values=paddle.arange(end=self.num_types)
            )
            pos_r = paddle.concat(
                x=[
                    pos_l[1:],
                    paddle.to_tensor(data=[len(x_type)], place=x.place),
                ]
            )
            seglen = (pos_r - pos_l).cpu()
            return segment_mm(x, w, seglen_a=seglen)
        else:
            return gather_mm(x, w, idx_b=x_type)

    def __repr__(self):
        if self.regularizer is None:
            return (
                f"TypedLinear(in_size={self.in_size}, out_size={self.out_size}, "
                f"num_types={self.num_types})"
            )
        else:
            return (
                f"TypedLinear(in_size={self.in_size}, out_size={self.out_size}, "
                f"num_types={self.num_types}, regularizer={self.regularizer}, "
                f"num_bases={self.num_bases})"
            )
