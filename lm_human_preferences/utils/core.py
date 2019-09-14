"""Utilities."""

import collections
import contextlib
import inspect
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache, partial, wraps
from typing import Any, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.contrib import summary

try:
    import horovod.tensorflow as hvd
    hvd.init()
except:
    hvd = None


nest = tf.contrib.framework.nest


def nvidia_gpu_count():
    """
    Count the GPUs on this machine.
    """
    if shutil.which('nvidia-smi') is None:
        return 0
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'])
    except subprocess.CalledProcessError:
        # Probably no GPUs / no driver running.
        return 0
    return max(0, len(output.split(b'\n')) - 2)


def get_local_rank_size(comm):
    """
    Returns the rank of each process on its machine
    The processes on a given machine will be assigned ranks
        0, 1, 2, ..., N-1,
    where N is the number of processes on this machine.
    Useful if you want to assign one gpu per machine
    """
    this_node = platform.node()
    ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
    node2rankssofar = collections.defaultdict(int)
    local_rank = None
    for (rank, node) in ranks_nodes:
        if rank == comm.Get_rank():
            local_rank = node2rankssofar[node]
        node2rankssofar[node] += 1
    assert local_rank is not None
    return local_rank, node2rankssofar[this_node]


@lru_cache()
def gpu_devices():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('CUDA_VISIBLE_DEVICES should not be set (it will cause nccl slowdowns).  Use VISIBLE_DEVICES instead!')
    devices_str = os.environ.get('VISIBLE_DEVICES')
    if devices_str is not None:
        return list(map(int, filter(len, devices_str.split(','))))
    else:
        return list(range(nvidia_gpu_count()))

@lru_cache()
def gpu_count():
    return len(gpu_devices()) or None


@lru_cache()
def _our_gpu():
    """Figure out which GPU we should be using in an MPI context."""
    gpus = gpu_devices()
    if not gpus:
        return None
    rank = MPI.COMM_WORLD.Get_rank()
    local_rank, local_size = get_local_rank_size(MPI.COMM_WORLD)
    if gpu_count() not in (0, local_size):
        raise ValueError('Expected one GPU per rank, got gpus %s, local size %d' % (gpus, local_size))
    gpu = gpus[local_rank]
    print('rank %d: gpus = %s, our gpu = %d' % (rank, gpus, gpu))
    return gpu


def mpi_session_config():
    """Make a tf.ConfigProto to use only the GPU assigned to this MPI session."""
    config = tf.ConfigProto()
    gpu = _our_gpu()
    if gpu is not None:
        config.gpu_options.visible_device_list = str(gpu)
    config.gpu_options.allow_growth = True
    return config


def mpi_session():
    """Create a session using only the GPU assigned to this MPI process."""
    return tf.Session(config=mpi_session_config())


def set_mpi_seed(seed: Optional[int]):
    if seed is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        seed = seed + rank * 100003  # Prime (kept for backwards compatibility even though it does nothing)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def exact_div(a, b):
    q = a // b
    if tf.contrib.framework.is_tensor(q):
        with tf.control_dependencies([tf.debugging.Assert(tf.equal(a, q * b), [a, b])]):
            return tf.identity(q)
    else:
        if a != q * b:
            raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
        return q


def ceil_div(a, b):
    return (a - 1) // b + 1


def expand_tile(value, size, *, axis, name=None):
    """Add a new axis of given size."""
    with tf.name_scope(name, 'expand_tile', [value, size, axis]) as scope:
        value = tf.convert_to_tensor(value, name='value')
        size = tf.convert_to_tensor(size, name='size')
        ndims = value.shape.rank
        if axis < 0:
            axis += ndims + 1
        return tf.tile(tf.expand_dims(value, axis=axis), [1]*axis + [size] + [1]*(ndims - axis), name=scope)


def index_each(a, ix):
    """Do a batched indexing operation: index row i of a by ix[i]

    In the simple case (a is >=2D and ix is 1D), returns [row[i] for row, i in zip(a, ix)].

    If ix has more dimensions, multiple lookups will be done at each batch index.
    For instance, if ix is 2D, returns [[row[i] for i in ix_row] for row, ix_row in zip(a, ix)].

    Always indexes into dimension 1 of a.
    """
    a = tf.convert_to_tensor(a, name='a')
    ix = tf.convert_to_tensor(ix, name='ix', dtype=tf.int32)
    with tf.name_scope('index_each', values=[a, ix]) as scope:
        a.shape[:1].assert_is_compatible_with(ix.shape[:1])
        i0 = tf.range(tf.shape(a)[0], dtype=ix.dtype)
        if ix.shape.rank > 1:
            i0 = tf.tile(tf.reshape(i0, (-1,) + (1,)*(ix.shape.rank - 1)), tf.concat([[1], tf.shape(ix)[1:]], axis=0))
        return tf.gather_nd(a, tf.stack([i0, ix], axis=-1), name=scope)

def cumulative_max(x):
    """Takes the (inclusive) cumulative maximum along the last axis of x. (Not efficient.)"""
    x = tf.convert_to_tensor(x)
    with tf.name_scope('cumulative_max', values=[x]) as scope:
        repeated = tf.tile(
            tf.expand_dims(x, axis=-1),
            tf.concat([tf.ones(x.shape.rank, dtype=tf.int32), tf.shape(x)[-1:]], axis=0))
        trues = tf.ones_like(repeated, dtype=tf.bool)
        upper_triangle = tf.matrix_band_part(trues, 0, -1)
        neg_inf = tf.ones_like(repeated) * tf.dtypes.saturate_cast(-np.inf, dtype=x.dtype)
        prefixes = tf.where(upper_triangle, repeated, neg_inf)
        return tf.math.reduce_max(prefixes, axis=-2, name=scope)


def flatten_dict(nested, sep='.'):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat

@dataclass
class Schema:
    dtype: Any
    shape: Tuple[Optional[int],...]


def add_batch_dim(schemas, batch_size=None):
    def add_dim(schema):
        return Schema(dtype=schema.dtype, shape=(batch_size,)+schema.shape)
    return nest.map_structure(add_dim, schemas)


class SampleBuffer:
    """A circular buffer for storing and sampling data.

    Data can be added to the buffer with `add`, and old data will be dropped.  If you need to
    control where the buffer is stored, wrap the constructor call in a `with tf.device` block:

        with tf.device('cpu:0'):
            buffer = SampleBuffer(...)
    """

    def __init__(self, *, capacity: int, schemas: Dict[str,Schema], name=None) -> None:
        with tf.variable_scope(name, 'buffer', use_resource=True, initializer=tf.zeros_initializer):
            self._capacity = tf.constant(capacity, dtype=tf.int32, name='capacity')
            self._total = tf.get_variable(
                'total', dtype=tf.int32, shape=(), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES],
            )
            self._vars = {
                n: tf.get_variable(
                    n, dtype=s.dtype, shape=(capacity,) + s.shape, trainable=False,
                    collections=[tf.GraphKeys.LOCAL_VARIABLES],
                )
                for n,s in schemas.items()
            }

    def add(self, **data):
        """Add new data to the end of the buffer, dropping old data if we exceed capacity."""
        # Check input shapes
        if data.keys() != self._vars.keys():
            raise ValueError('data.keys() = %s != %s' % (sorted(data.keys()), sorted(self._vars.keys())))
        first = next(iter(data.values()))
        pre = first.shape[:1]
        for k, d in data.items():
            try:
                d.shape.assert_is_compatible_with(pre.concatenate(self._vars[k].shape[1:]))
            except ValueError as e:
                raise ValueError('%s, key %s' % (e, k))
        # Enqueue
        n = tf.shape(first)[0]
        capacity = self._capacity
        i0 = (self._total.assign_add(n) - n) % capacity
        i0n = i0 + n
        i1 = tf.minimum(i0n, capacity)
        i2 = i1 % capacity
        i3 = i0n % capacity
        slices = slice(i0, i1), slice(i2, i3)
        sizes = tf.stack([i1 - i0, i3 - i2])
        assigns = [self._vars[k][s].assign(part)
                   for k,d in data.items()
                   for s, part in zip(slices, tf.split(d, sizes))]
        return tf.group(assigns)

    def total(self):
        """Total number of entries ever added, including those already discarded."""
        return self._total.read_value()

    def size(self):
        """Current number of entries."""
        return tf.minimum(self.total(), self._capacity)

    def read(self, indices):
        """indices: A 1-D Tensor of indices to read from. Each index must be less than
        capacity."""
        return {k: v.sparse_read(indices) for k,v in self._vars.items()}

    def data(self):
        return {k: v[:self.size()] for k,v in self._vars.items()}

    def sample(self, n, seed=None):
        """Sample n entries with replacement."""
        size = self.size()
        indices = tf.random_uniform([n], maxval=size, dtype=tf.int32, seed=seed)
        return self.read(indices)

    def write(self, indices, updates):
        """
        indices: A 1-D Tensor of indices to write to. Each index must be less than `capacity`.
        update: A dictionary of new values, where each entry is a tensor with the same length as `indices`.
        """
        ops = []
        for k, v in updates.items():
            ops.append(self._vars[k].scatter_update(tf.IndexedSlices(v, tf.cast(indices, dtype=tf.int32))))
        return tf.group(*ops)

    def write_add(self, indices, deltas):
        ops = []
        for k, d in deltas.items():
            ops.append(self._vars[k].scatter_add(tf.IndexedSlices(d, tf.cast(indices, dtype=tf.int32))))
        return tf.group(*ops)


def entropy_from_logits(logits):
    pd = tf.nn.softmax(logits, axis=-1)
    return tf.math.reduce_logsumexp(logits, axis=-1) - tf.reduce_sum(pd*logits, axis=-1)


def logprobs_from_logits(*, logits, labels):
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def sample_from_logits(logits, dtype=tf.int32):
    with tf.name_scope('sample_from_logits', values=[logits]) as scope:
        shape = tf.shape(logits)
        flat_logits = tf.reshape(logits, [-1, shape[-1]])
        flat_samples = tf.random.categorical(flat_logits, num_samples=1, dtype=dtype)
        return tf.reshape(flat_samples, shape[:-1], name=scope)


def take_top_k_logits(logits, k):
    values, _ = tf.nn.top_k(logits, k=k)
    min_values = values[:, :, -1, tf.newaxis]
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def take_top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, sequence, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch)[:, tf.newaxis],
        tf.range(0, sequence)[tf.newaxis, :],
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def whiten(values, shift_mean=True):
    mean, var = tf.nn.moments(values, axes=list(range(values.shape.rank)))
    whitened = (values - mean) * tf.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened



def where(cond, true, false, name=None):
    """Similar to tf.where, but broadcasts scalar values."""
    with tf.name_scope(name, 'where', [cond, true, false]) as name:
        cond = tf.convert_to_tensor(cond, name='cond', dtype=tf.bool)
        true = tf.convert_to_tensor(true, name='true',
                                    dtype=false.dtype if isinstance(false, tf.Tensor) else None)
        false = tf.convert_to_tensor(false, name='false', dtype=true.dtype)
        if true.shape.rank == false.shape.rank == 0:
            shape = tf.shape(cond)
            true = tf.fill(shape, true)
            false = tf.fill(shape, false)
        elif true.shape.rank == 0:
            true = tf.fill(tf.shape(false), true)
        elif false.shape.rank == 0:
            false = tf.fill(tf.shape(true), false)
        return tf.where(cond, true, false, name=name)


def map_flat(f, values):
    """Apply the function f to flattened, concatenated values, then split and reshape back to original shapes."""
    values = tuple(values)
    for v in values:
        assert not isinstance(v, tf.IndexedSlices)
    values = [tf.convert_to_tensor(v) for v in values]
    flat = tf.concat([tf.reshape(v, [-1]) for v in values], axis=0)
    flat = f(flat)
    parts = tf.split(flat, [tf.size(v) for v in values])
    return [tf.reshape(p, tf.shape(v)) for p, v in zip(parts, values)]


def map_flat_chunked(f, values, *, limit=1<<29):
    """
    Apply the function f to chunked, flattened, concatenated values, then split and reshape back to original shapes.
    """
    values = tuple(values)
    for v in values:
        assert not isinstance(v, tf.IndexedSlices)
    values = [tf.convert_to_tensor(v) for v in values]
    chunks = chunk_tensors(values, limit=limit)
    mapped_values = [v for chunk in chunks for v in map_flat(f, chunk)]
    return mapped_values


def map_flat_bits(f, values):
    """Apply the function f to bit-concatenated values, then convert back to original shapes and dtypes."""
    values = [tf.convert_to_tensor(v) for v in values]
    def maybe_bitcast(v, dtype):
        cast = tf.cast if tf.bool in (v.dtype, dtype) else tf.bitcast
        return cast(v, dtype)
    bits = [maybe_bitcast(v, tf.uint8) for v in values]
    flat = tf.concat([tf.reshape(b, [-1]) for b in bits], axis=0)
    flat = f(flat)
    parts = tf.split(flat, [tf.size(b) for b in bits])
    return [maybe_bitcast(tf.reshape(p, tf.shape(b)), v.dtype)
            for p, v, b in zip(parts, values, bits)]

def mpi_bcast_tensor_dict(d, comm):
    sorted_keys = sorted(d.keys())
    values = map_flat_bits(partial(mpi_bcast, comm), [d[k] for k in sorted_keys])
    return {k: v for k, v in zip(sorted_keys, values)}

def mpi_bcast(comm, value, root=0):
    """Broadcast value from root to other processes via a TensorFlow py_func."""
    value = tf.convert_to_tensor(value)
    if comm.Get_size() == 1:
        return value
    comm = comm.Dup()  # Allow parallelism at graph execution time
    if comm.Get_rank() == root:
        out = tf.py_func(partial(comm.bcast, root=root), [value], value.dtype)
    else:
        out = tf.py_func(partial(comm.bcast, None, root=root), [], value.dtype)
    out.set_shape(value.shape)
    return out


def chunk_tensors(tensors, *, limit=1 << 28):
    """Chunk the list of tensors into groups of size at most `limit` bytes.

    The tensors must have a static shape.
    """
    total = 0
    batches = []
    for v in tensors:
        size = v.dtype.size * v.shape.num_elements()
        if not batches or total + size > limit:
            total = 0
            batches.append([])
        total += size
        batches[-1].append(v)
    return batches


def variable_synchronizer(comm, vars, *, limit=1<<28):
    """Synchronize `vars` from the root to other processs"""
    if comm.Get_size() == 1:
        return tf.no_op()

    # Split vars into chunks so that no chunk is over limit bytes
    batches = chunk_tensors(sorted(vars, key=lambda v: v.name), limit=limit)

    # Synchronize each batch, using a separate communicator to ensure safety
    prev = tf.no_op()
    for batch in batches:
        with tf.control_dependencies([prev]):
            assigns = []
            values = map_flat_bits(partial(mpi_bcast, comm), batch)
            for var, value in zip(batch, values):
                assigns.append(var.assign(value))
            prev = tf.group(*assigns)
    return prev


def mpi_read_file(comm, path):
    """Read a file on rank 0 and broadcast the contents to all machines."""
    if comm.Get_rank() == 0:
        with tf.gfile.Open(path, 'rb') as fh:
            data = fh.read()
        comm.bcast(data)
    else:
        data = comm.bcast(None)
    return data


def mpi_allreduce_sum(values, *, comm):
    if comm.Get_size() == 1:
        return values
    orig_dtype = values.dtype
    if hvd is None:
        orig_shape = values.shape
        def _allreduce(vals):
            buf = np.zeros(vals.shape, np.float32)
            comm.Allreduce(vals, buf, op=MPI.SUM)
            return buf
        values = tf.py_func(_allreduce, [values], tf.float32)
        values.set_shape(orig_shape)
    else:
        values = hvd.mpi_ops._allreduce(values)
    return tf.cast(values, dtype=orig_dtype)


def mpi_allreduce_mean(values, *, comm):
    scale = 1 / comm.Get_size()
    values = mpi_allreduce_sum(values, comm=comm)
    return values if scale == 1 else scale * values


class FlatStats:
    """A bunch of statistics stored as a single flat tensor."""

    def __init__(self, keys, flat):
        keys = tuple(keys)
        flat = tf.convert_to_tensor(flat, dtype=tf.float32, name='flat')
        assert [len(keys)] == flat.shape.as_list()
        self.keys = keys
        self.flat = flat

    @staticmethod
    def from_dict(stats):
        for k, v in stats.items():
            if v.dtype != tf.float32:
                raise ValueError('Statistic %s has dtype %r, expected %r' % (k, v.dtype, tf.float32))
        keys = tuple(sorted(stats.keys()))
        flat = tf.stack([stats[k] for k in keys])
        return FlatStats(keys, flat)

    def concat(self, more):
        dups = set(self.keys) & set(more.keys)
        if dups:
            raise ValueError('Duplicate statistics: %s' % ', '.join(dups))
        return FlatStats(self.keys + more.keys, tf.concat([self.flat, more.flat], axis=0))

    def as_dict(self):
        flat = tf.unstack(self.flat, num=len(self.keys))
        return dict(safe_zip(self.keys, flat))

    def with_values(self, flat):
        return FlatStats(self.keys, flat)

    def map_flat(self, f):
        return FlatStats(self.keys, f(self.flat))


def find_trainable_variables(key):
    return [v for v in tf.trainable_variables() if v.op.name.startswith(key + '/')]


def variables_on_gpu():
    """Prevent variables from accidentally being placed on the CPU.

    This dodges an obscure bug in tf.train.init_from_checkpoint.
    """
    if _our_gpu() is None:
        return contextlib.suppress()
    def device(op):
        return '/gpu:0' if op.type == 'VarHandleOp' else ''
    return tf.device(device)



def graph_function(**schemas: Schema):
    def decorate(make_op):
        def make_ph(path, schema):
            return tf.placeholder(name=f'arg_{make_op.__name__}_{path}', shape=schema.shape, dtype=schema.dtype)
        phs = nest.map_structure_with_paths(make_ph, schemas)
        op = make_op(**phs)
        sig = inspect.signature(make_op)
        @wraps(make_op)
        def run(*args, **kwargs):
            bound: inspect.BoundArguments = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            arg_dict = bound.arguments
            for name, param in sig.parameters.items():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    kwargs = arg_dict[name]
                    arg_dict.update(kwargs)
                    del arg_dict[name]
            flat_phs = nest.flatten(phs)
            flat_arguments = nest.flatten_up_to(phs, bound.arguments)
            feed = {ph: arg for ph, arg in zip(flat_phs, flat_arguments)}
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

            return tf.get_default_session().run(op, feed_dict=feed, options=run_options, run_metadata=None)
        return run
    return decorate



def pearson_r(x: tf.Tensor, y: tf.Tensor):
    assert x.shape.rank == 1
    assert y.shape.rank == 1
    x_mean, x_var = tf.nn.moments(x, axes=[0])
    y_mean, y_var = tf.nn.moments(y, axes=[0])
    cov = tf.reduce_mean((x - x_mean)*(y - y_mean), axis=0)
    return cov / tf.sqrt(x_var * y_var)

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def safe_zip(*args):
    """Zip, but require all sequences to be the same length."""
    args = tuple(map(tuple, args))
    for a in args[1:]:
        if len(args[0]) != len(a):
            raise ValueError(f'Lengths do not match: {[len(a) for a in args]}')
    return zip(*args)


def get_summary_writer(save_dir, subdir='', comm=MPI.COMM_WORLD):
    if comm.Get_rank() != 0:
        return None
    if save_dir is None:
        return None
    with tf.init_scope():
        return summary.create_file_writer(os.path.join(save_dir, 'tb', subdir))


def record_stats(*, stats, summary_writer, step, log_interval, name=None, comm=MPI.COMM_WORLD):
    def log_stats(step, *stat_values):
        if comm.Get_rank() != 0 or step % log_interval != 0:
            return

        for k, v in safe_zip(stats.keys(), stat_values):
            print('k = ', k, ', v = ', v)

    summary_ops = [tf.py_func(log_stats, [step] + list(stats.values()), [])]
    if summary_writer:
        with summary_writer.as_default(), summary.always_record_summaries():
            for key, value in stats.items():
                summary_ops.append(summary.scalar(key, value, step=step))
    return tf.group(*summary_ops, name=name)


def minimize(*, loss, params, lr, name=None, comm=MPI.COMM_WORLD):
    with tf.name_scope(name, 'minimize'):
        with tf.name_scope('grads'):
            grads = tf.gradients(loss, params)
        grads, params = zip(*[(g, v) for g, v in zip(grads, params) if g is not None])
        grads = map_flat_chunked(partial(mpi_allreduce_mean, comm=comm), grads)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5, name='adam')
        opt_op = optimizer.apply_gradients(zip(grads, params), name=name)
        return opt_op
