import paddle

"""An abstract base class for cugraph-ops nn module."""


class CuGraphBaseConv(paddle.nn.Layer):
    """An abstract base class for cugraph-ops nn module."""

    def __init__(self):
        super().__init__()
        self._cached_offsets_fg = None

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        raise NotImplementedError

    def forward(self, *args):
        """Runs the forward pass of the module."""
        raise NotImplementedError

    def pad_offsets(self, offsets: paddle.Tensor, size: int) -> paddle.Tensor:
        """Pad zero-in-degree nodes to the end of offsets to reach size.

        cugraph-ops often provides two variants of aggregation functions for a
        specific model: one intended for sampled-graph use cases, one for
        full-graph ones. The former is in general more performant, however, it
        only works when the sample size (the max of in-degrees) is small (<200),
        due to the limit of GPU shared memory. For graphs with a larger max
        in-degree, we need to fall back to the full-graph option, which requires
        to convert a DGL block to a full graph. With the csc-representation,
        this is equivalent to pad zero-in-degree nodes to the end of the offsets
        array (also called indptr or colptr).

        Parameters
        ----------
        offsets :
            The (monotonically increasing) index pointer array in a CSC-format
            graph.
        size : int
            The length of offsets after padding.

        Returns
        -------
        paddle.Tensor
            The augmented offsets array.
        """
        if self._cached_offsets_fg is None:
            self._cached_offsets_fg = paddle.empty(
                shape=size, dtype=offsets.dtype
            )
        elif self._cached_offsets_fg.size < size:
            self._cached_offsets_fg.resize_([size])
        self._cached_offsets_fg[: offsets.size] = offsets
        self._cached_offsets_fg[offsets.size : size] = offsets[-1]
        return self._cached_offsets_fg[:size]
