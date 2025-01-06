from __future__ import absolute_import

import builtins
import numbers

import numpy as np
import paddle

from ... import ndarray as nd
from ...utils import version
from .utils import transpose_aux_func

if version.parse(paddle.__version__) < version.parse("2.1.0"):
    raise RuntimeError("DGL requires PyTorch >= 2.1.0")


def data_type_dict():
    return {
        "bfloat16": paddle.bfloat16,
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
        "uint8": paddle.uint8,
        "int8": paddle.int8,
        "int16": paddle.int16,
        "int32": paddle.int32,
        "int64": paddle.int64,
        "bool": paddle.bool,
    }


def cpu():
    return paddle.CPUPlace()


def tensor(data, dtype=None):
    if isinstance(data, numbers.Number):
        data = [data]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], paddle.Tensor):
        if data[0].ndim == 0:
            return paddle.stack(x=data)
    if isinstance(data, paddle.Tensor):
        return paddle.to_tensor(data=data, dtype=dtype, place=data.place)
    else:
        return paddle.to_tensor(data=data, dtype=dtype)


def as_scalar(data):
    return data.item()


def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "coo"


def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt != "coo":
        raise TypeError("Pytorch backend only supports COO format. But got %s." % fmt)
    spmat = paddle.sparse.sparse_coo_tensor(indices=index[1], values=data, shape=shape)
    return spmat, None


def sparse_matrix_indices(spmat):
    return "coo", spmat._indices()


def is_tensor(obj):
    return isinstance(obj, paddle.Tensor)


def shape(input):
    return tuple(input.shape)


def dtype(input):
    return input.dtype


def ndim(input):
    return input.dim()


def context(input):
    return input.place


def device_type(ctx):
    return paddle.get_device().split(":")[0]


def device_id(ctx):
    ctx = paddle.get_device()
    if ":" in ctx:
        return int(ctx.split(":")[1])
    else:
        return 0


def to_backend_ctx(dglctx):
    dev_type = dglctx.device_type
    if dev_type == 1:
        return paddle.CPUPlace()
    elif dev_type == 2:
        return paddle.CUDAPlace(dglctx.device_id)
    else:
        raise ValueError("Unsupported DGL device context:", dglctx)


def astype(input, ty):
    return input.astype(ty)


def asnumpy(input):
    if isinstance(input, paddle.Tensor) and input.is_sparse():
        return input.to_dense().numpy()
    else:
        return input.numpy()


def copy_to(input, ctx, **kwargs):
    if isinstance(ctx, paddle.CPUPlace):
        return input.cpu()
    elif isinstance(ctx, paddle.CUDAPlace):
        paddle.set_device(f"gpu:{ctx.get_device_id()}")
        return input.cuda()
    else:
        raise RuntimeError("Invalid context", ctx)


def is_pinned(input):
    return "pinned" in str(input.place)


def sum(input, dim, keepdims=False):
    return paddle.sum(x=input, axis=dim, keepdim=keepdims)


def floor_div(in1, in2):
    return in1 // in2


def reduce_sum(input):
    return input.sum()


def cumsum(input, dim):
    return paddle.cumsum(x=input, axis=dim)


def mean(input, dim):
    return paddle.mean(x=input, axis=dim)


def reduce_mean(input):
    return input.mean()


def max(input, dim):
    return (paddle.max(x=input, axis=dim), paddle.argmax(x=input, axis=dim))[0]


def reduce_max(input):
    return input.max()


def min(input, dim):
    return (paddle.min(x=input, axis=dim), paddle.argmin(x=input, axis=dim))[0]


def reduce_min(input):
    return input.min()


def argsort(input, dim, descending):
    return paddle.argsort(x=input, axis=dim, descending=descending)


def topk(input, k, dim, descending=True):
    return paddle.topk(k=k, largest=descending, x=input, axis=dim)[0]


def argtopk(input, k, dim, descending=True):
    return paddle.topk(k=k, largest=descending, x=input, axis=dim)[1]


def exp(input):
    return paddle.exp(x=input)


def inverse(input):
    return paddle.linalg.inv(x=input)


def sqrt(input):
    return paddle.sqrt(x=input)


def softmax(input, dim=-1):
    return paddle.nn.functional.softmax(x=input, axis=dim)


def cat(seq, dim):
    return paddle.concat(x=seq, axis=dim)


def stack(seq, dim):
    return paddle.stack(x=seq, axis=dim)


def split(input, sizes_or_sections, dim):
    if isinstance(sizes_or_sections, int):
        return paddle.split(input, input.shape[dim] // sizes_or_sections, dim)
    else:
        return paddle.split(input, sizes_or_sections, dim)


def repeat(input, repeats, dim):
    return paddle.repeat_interleave(x=input, repeats=repeats, axis=dim)


def gather_row(data, row_index):
    return paddle.index_select(x=data, axis=0, index=row_index.astype(dtype="int64"))


def slice_axis(data, axis, begin, end):
    start_0 = data.shape[axis] + begin if begin < 0 else begin
    return paddle.slice(data, [axis], [start_0], [start_0 + (end - begin)])


def take(data, indices, dim):
    new_shape = tuple(data.shape)[:dim] + tuple(indices.shape) + tuple(data.shape)[dim + 1 :]
    return paddle.index_select(x=data, axis=dim, index=indices.view(-1)).view(new_shape)


def narrow_row(x, start, stop):
    return x[start:stop]


def index_add_inplace(data, row_idx, value):
    data.index_add_(axis=0, index=row_idx, value=value)


def scatter_row(data, row_index, value):
    row_index = paddle.cast(row_index, "int64")
    updated_data = paddle.scatter_(data, row_index, value, overwrite=True)
    return updated_data


def scatter_row_inplace(data, row_index, value):
    data[row_index.astype(dtype="int64")] = value


def squeeze(input, dim):
    return paddle.squeeze(x=input, axis=dim)


def unsqueeze(input, dim):
    return paddle.unsqueeze(x=input, axis=dim)


def reshape(input, shape):
    return paddle.reshape(x=input, shape=shape)


def swapaxes(input, axis1, axis2):
    return paddle.transpose(x=input, perm=transpose_aux_func(input.ndim, axis1, axis2))


def empty(shape, dtype, ctx):
    return paddle.empty(shape=shape, dtype=dtype)


def zeros(shape, dtype, ctx):
    return paddle.zeros(shape=shape, dtype=dtype)


def zeros_like(input):
    return paddle.zeros_like(x=input)


def ones(shape, dtype, ctx):
    return paddle.ones(shape=shape, dtype=dtype)


def uniform(shape, dtype, ctx, low, high):
    return paddle.empty(shape=shape, dtype=dtype).uniform_(min=low, max=high)


def randint(shape, dtype, ctx, low, high):
    return paddle.randint(low=low, high=high, shape=shape, dtype=dtype)


def pad_packed_tensor(input, lengths, value, l_min=None):
    old_shape = tuple(input.shape)
    device = input.place
    if not is_tensor(lengths):
        lengths = paddle.to_tensor(data=lengths, dtype="int64", place=device)
    else:
        lengths = lengths.to(device)
    max_len = as_scalar(lengths.max())
    if l_min is not None:
        max_len = builtins.max(max_len, l_min)
    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value=value)
    index = paddle.ones(shape=len(input), dtype="int64")
    cum_lengths = paddle.cumsum(x=lengths, axis=0)
    index[cum_lengths[:-1]] += max_len - lengths[:-1]
    index = paddle.cumsum(x=index, axis=0) - 1
    x[index] = input
    return x.view(batch_size, max_len, *old_shape[1:])


def pack_padded_tensor(input, lengths):
    max_len = tuple(input.shape)[1]
    device = input.place
    if not is_tensor(lengths):
        lengths = paddle.to_tensor(data=lengths, dtype="int64", place=device)
    else:
        lengths = lengths.to(device)
    input = input.view(-1, *tuple(input.shape)[2:])
    out_len = lengths.sum().item()
    index = paddle.ones(shape=out_len, dtype="int64")
    cum_lengths = paddle.cumsum(x=lengths, axis=0)
    index[cum_lengths[:-1]] += max_len - lengths[:-1]
    index = paddle.cumsum(x=index, axis=0) - 1
    return input[index]


def boolean_mask(input, mask):
    if "bool" not in str(mask.dtype):
        mask = paddle.to_tensor(data=mask, dtype="bool")
    return input[mask]


def equal(x, y):
    return x == y


def allclose(x, y, rtol=0.0001, atol=0.0001):
    return paddle.allclose(x=x, y=y, rtol=rtol, atol=atol).item()


def logical_not(input):
    return ~input


def logical_and(input1, input2):
    return input1 & input2


def clone(input):
    return input.clone()


def clamp(data, min_val, max_val):
    return paddle.clip(x=data, min=min_val, max=max_val)


def replace_inf_with_zero(x):
    return paddle.masked_fill(x=x, mask=paddle.isinf(x=x), value=0)


def count_nonzero(input):
    return np.count_nonzero(input)


def unique(input, return_inverse=False, return_counts=False):
    if input.dtype == "bool":
        input = input.astype("int8")
    return paddle.unique(x=input, return_inverse=return_inverse, return_counts=return_counts)


def full_1d(length, fill_value, dtype, ctx):
    return paddle.full(shape=(length,), fill_value=fill_value, dtype=dtype)


def nonzero_1d(input):
    paddle.utils.try_import("warnings").warn("Now, the return shape is inconsistent with torch when as_tuple is True")
    x = paddle.nonzero(x=input, as_tuple=False).squeeze()
    return x if x.dim() == 1 else x.view(-1)


def sort_1d(input):
    return paddle.sort(x=input), paddle.argsort(x=input)


def arange(start, stop, dtype="int64", ctx=None):
    return paddle.arange(start=start, end=stop, dtype=dtype)


def rand_shuffle(arr):
    idx = paddle.randperm(n=len(arr))
    return arr[idx]


def zerocopy_to_dlpack(input):
    return paddle.utils.dlpack.to_dlpack(x=input.contiguous())


def zerocopy_from_dlpack(dlpack_tensor):
    return paddle.utils.dlpack.from_dlpack(dlpack=dlpack_tensor)


def zerocopy_to_numpy(input):
    return asnumpy(input)


def zerocopy_from_numpy(np_array):
    return paddle.to_tensor(data=np_array)


def zerocopy_to_dgl_ndarray(data):
    if data.dtype == "bool":
        data = data.astype(dtype="uint8")
    return nd.from_dlpack(paddle.utils.dlpack.to_dlpack(x=data.contiguous()))


def zerocopy_to_dgl_ndarray_for_write(input):
    if input.numel() > 0:
        assert input.is_contiguous(), (
            "Cannot convert non-contiguous tensors " "to dgl ndarray for write. Call Tensor.contiguous() first."
        )
    return zerocopy_to_dgl_ndarray(input)


def zerocopy_from_dgl_ndarray(data):
    if tuple(data.shape) == (0,):
        return paddle.to_tensor(data=[], dtype=data.dtype, place=to_backend_ctx(data.ctx))
    elif len(tuple(data.shape)) == 0 or builtins.min(tuple(data.shape)) == 0:
        return paddle.empty(shape=tuple(data.shape), dtype=data.dtype)
    else:
        return paddle.utils.dlpack.from_dlpack(dlpack=data.to_dlpack())


def sync():
    pass


def attach_grad(x):
    if x.grad is not None:
        x.grad.zero_()
        return x
    else:
        out_0 = x
        out_0.stop_gradient = not True
        return out_0


def backward(x, head_gradient=None):
    if head_gradient is not None and tuple(head_gradient.shape)[0] == 1 and len(tuple(head_gradient.shape)) == 1:
        head_gradient = paddle.to_tensor(data=head_gradient.item()).to(head_gradient.place)
    x.backward(grad_tensor=head_gradient)


def grad(x):
    x.retain_grads()
    return x.grad


def is_no_grad(x):
    return x.grad is None or (x.grad == 0).astype("bool").all()


def is_recording():
    return paddle.is_grad_enabled()


class record_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


no_grad = paddle.no_grad
