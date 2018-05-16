import copy
import ctypes
import threading
import multiprocessing
from multiprocessing import Lock, Array
import numexpr as ne
import warnings
import os
import six
import numpy as np

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer.training.updaters import standard_updater

import cupy as xp
from cupy.cuda import profiler

try:
    from cupy.cuda import nccl
    _available = True
except ImportError:
    _available = False


class _Worker(multiprocessing.Process):

    def __init__(self, proc_id, pipe, master):
        super(_Worker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.converter = master.converter
        self.model = copy.deepcopy(master._master)
        self.device = master._devices[proc_id]
        self.iterator = master._mpu_iterators[proc_id]
        self.n_devices = len(master._devices)
        self.devices = master._devices
        self.optimizer = copy.deepcopy(master._optimizer)

        # data prefetcher
        self.data_prefetcher = None

        self.need_sync_param = master.need_sync_param
        self._stream = None

        self.cpu_mode = master.cpu_mode
        self.debug = False  # turn off debug for workers
        self.fine_granularity = master.fine_granularity

        # recompute orders because they are not copied
        order = 0
        for _, param in sorted(self.model.namedparams()):
            param._node.order = order
            order += 1

        # for cpu_mode
        if self.cpu_mode:
            self.cbcl = CPUBasedLWR(self.model, master.global_grad_name,
                                    master.done_workers, master.locks,
                                    self.devices, self.device, self.debug)

    def setup(self):
        if self.cpu_mode:
            self.cbcl.setup()
        else:
            _, comm_id = self.pipe.recv()
            self.comm = nccl.NcclCommunicator(self.n_devices, comm_id,
                                              self.proc_id)
            self._stream = cuda.Stream(non_blocking=True)

        self.optimizer.setup(self.model)
        self.model.to_gpu(self.device)
        self.reporter = reporter.Reporter()
        self.reporter.add_observer('main', self.model)

        # prepare a worker to do data prefetch
        self.data_prefetcher = _DataPrefetcher(self.iterator,
                                               self.converter,
                                               self.device)
        self.data_prefetcher.daemon = True
        self.data_prefetcher.start()

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        if self.cpu_mode:
            self.pipe.recv()  # wait for the master finished its setup
        self.setup()
        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                dev.synchronize()
                self.finalize()
                break
            if job == 'update':
                if data == "new_epoch":  # new epoch
                    self.optimizer.new_epoch()

                # For reducing memory
                self.model.cleargrads()

                # batch = self.converter(self.iterator.next(), self.device)
                batch = self.data_prefetcher.batch()

                observation = {}
                with self.reporter.scope(observation):
                    loss = _calc_loss(self.model, batch)

                self.model.cleargrads()

                with chainer.layer_wise_reduce_mode(self):
                    loss.backward()
                del loss
                self.optimizer.update()

                if self.need_sync_param and not self.cpu_mode:
                    cuda.Stream.null.synchronize()
                    gp = gather_params(self.model)
                    self.comm.bcast(gp.data.ptr, gp.size, nccl.NCCL_FLOAT, 0,
                                    cuda.Stream.null.ptr)
                    scatter_params(self.model, gp)
                    gp = None
                    self.need_sync_param = False

        profiler.stop()

    # a callback function, worker
    def run_vnode(self, vnodes=None):
        if len(vnodes) == 0:
            return

        if self.cpu_mode:
            event = cuda.Stream.null.record()
            event.synchronize()
            orders = [x.order for x in vnodes]
            self.cbcl.allreduce(orders)
        else:
            event = cuda.Stream.null.record()
            self._stream.wait_event(event)
            for vnode in vnodes:
                if self.debug:
                    print("allreduce vnode {} size {}"
                          .format(vnode.rank, vnode.grad.size))

                self.comm.allReduce(vnode.grad.data.ptr,
                                    vnode.grad.data.ptr,
                                    vnode.grad.size,
                                    nccl.NCCL_FLOAT,
                                    nccl.NCCL_SUM,
                                    self._stream.ptr)

    # a callback function
    def start_backward(self):
        pass

    # a callback function, worker
    def end_backward(self):
        if self.cpu_mode:
            self.cbcl.wait_for_gradients()
        else:
            self._stream.synchronize()

    def finalize(self):
        if self.cpu_mode:
            if self.cbcl is not None:
                self.cbcl.finalize()
        # stop the process of reading data
        self.data_prefetcher.finalize()
        self.data_prefetcher.join()


class LWRParallelUpdater(standard_updater.StandardUpdater):

    """Implementation of a layer-wise reduction training which is based on
    the multiprocess parallel GPU Updater. Gradients are accumulated as soon as
    they are available instead of being accumulated at the end of the backward
    phase.

    This is an implementation of :class:`Updater` that uses multiple GPUs
    with multi-process data parallelism. It uses Nvidia NCCL for communication
    between multiple GPUs.

    It behaves similarly to :class:`~chainer.training.StandardUpdater`. The
    update routine is modified to support data-parallel computation on multiple
    GPUs in one machine. It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.

    It does not transfer the values collected by :class:`Reporter` in the sub
    devices to the main device. So you can only see the reported values in
    the main device.

    Args:
        iterators: List of dataset iterator for the training dataset. The
            number of the iterators must be same to the number of GPUs you use.
        optimizer: Optimizer to update parameters. The model should be attached
            to the optimizer.
        converter: Converter function to build input arrays. Each batch
            extracted by the iterator is split equally between the devices and
            then passed with corresponding ``device`` option to this function.
            :func:`~chainer.dataset.concat_examples` is used by default.
        devices: Dictionary or list of devices to which the training data is
            sent. The master device will be the first one in the list or the
            value attached to the key ``'main'``.
        fine_granularity: True if doing accumulation after each variable's
            backward phase finished. False, otherwise.
        cpu_mode: True if using CPU for gradient accumulation. False if using
            GPU for gradient accumulation.
        debug: Display debug information or not

    """

    def __init__(self, iterators, optimizer, converter=convert.concat_examples,
                 devices=None, fine_granularity=True, cpu_mode=False,
                 debug=False):
        if not cpu_mode and not LWRParallelUpdater.available():
            raise Exception(
                'NCCL is not enabled. LWRParallelUpdater '
                'requires NCCL.\n'
                'Please reinstall chainer after you install NCCL.\n'
                '(see https://github.com/chainer/chainer#installation).')

        assert len(iterators) == len(devices)
        for iterator in iterators[1:]:
            assert len(iterator.dataset) == len(iterators[0].dataset)

        # Correct optimizer parameters for new minibatch size
        optim = optimizer.__class__.__name__
        if optim in ('Adam', 'AdaGrad', 'RMSprop'):
            optimizer.eps *= len(devices)
            warnings.warn('optimizer.eps is changed to {} '
                          'by LWRParallelUpdater for new batch size.'.
                          format(optimizer.eps))
        elif optim in ('RMSpropGraves', 'AdaDelta'):
            optimizer.eps *= len(devices) ** 2  # not quite right for AdaDelta
            warnings.warn('optimizer.eps is changed to {} '
                          'by LWRParallelUpdater for new batch size.'.
                          format(optimizer.eps))
        elif hasattr(optimizer, 'lr'):
            optimizer.lr /= len(devices)
            warnings.warn('optimizer.lr is changed to {} '
                          'by LWRParallelUpdater for new batch size.'.
                          format(optimizer.lr))

        super(LWRParallelUpdater, self).__init__(
            iterator=iterators[0],
            optimizer=optimizer,
            converter=converter
        )

        if isinstance(devices, dict):
            main = devices.pop('main')
            devices = list(six.itervalues(devices))
            devices = [main] + devices
        if devices is None or any(device is None for device in devices):
            raise ValueError('must specify GPU devices')

        self._master = optimizer.target
        self._devices = devices
        self._mpu_iterators = iterators
        self._initialized = False
        self._optimizer = optimizer

        self._pipes = []
        self._workers = []
        self.comm = None

        # data prefetcher
        self.data_prefetcher = None

        self.need_sync_param = False
        self._stream = None

        self.cpu_mode = cpu_mode
        self.debug = debug
        self.fine_granularity = fine_granularity

        # for cpu_mode
        if cpu_mode:
            # shared, global variables
            self.global_grad_name = "gg_mem"
            self.done_workers = None
            self.locks = None
            self.cbcl = None

    @staticmethod
    def available():
        return _available

    def _send_message(self, message):
        for pipe in self._pipes:
            pipe.send(message)

    def setup_workers(self):
        if self._initialized:
            return
        self._initialized = True

        if self.cpu_mode:
            # CPU mode requires every param is initialized at the beginining
            # e.g. declare: self.conv2 = L.Convolution2D(96, 256,  5, pad=2)
            # instead of self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            order = 0
            for name, param in sorted(self._master.namedparams()):
                assert param.data is not None
                param._node.order = order
                if self.debug:
                    print("order {} size {}".format(order, param.data.size))
                order += 1
        else:
            for _, param in sorted(self._master.namedparams()):
                if param.data is None:
                    self.need_sync_param = True
                    break

        if self.cpu_mode:
            x = _setup_global_variables(self._master)
            self.done_workers = x[0]
            self.locks = x[1]

            self.cbcl = CPUBasedLWR(self._master, self.global_grad_name,
                                    self.done_workers, self.locks,
                                    self._devices, self._devices[0],
                                    self.debug)

        # create workers
        self._master.cleargrads()
        for i in six.moves.range(1, len(self._devices)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with cuda.Device(self._devices[0]):
            # prepare a worker to do data prefetch
            self.data_prefetcher = _DataPrefetcher(self._iterators['main'],
                                                   self.converter,
                                                   self._devices[0])
            self.data_prefetcher.daemon = True
            self.data_prefetcher.start()

            self._master.to_gpu(self._devices[0])
            if len(self._devices) > 1:
                if self.cpu_mode:
                    g_nbytes = 0
                    for _, param in sorted(self._master.namedparams()):
                        g_nbytes += param.data.nbytes
                    self.global_grad_mem = xp.cuda.pinned_memory.SharedPinnedMemory(
                        "gg_mem", g_nbytes)
                    self.cbcl.setup()
                else:
                    comm_id = nccl.get_unique_id()
                    self._send_message(("set comm_id", comm_id))
                    self.comm = nccl.NcclCommunicator(len(self._devices),
                                                      comm_id, 0)
                    self._stream = cuda.Stream(non_blocking=True)

        if self.cpu_mode:
            self._send_message("Setup on master finished")

    def update_core(self):
        self.setup_workers()

        data = ("new_epoch" if self._iterators['main'].is_new_epoch else None)
        self._send_message(('update', data))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            self._master.cleargrads()

            optimizer = self.get_optimizer('main')

            # batch = self.get_iterator('main').next()
            # batch = self.converter(batch, self._devices[0])
            batch = self.data_prefetcher.batch()

            loss = _calc_loss(self._master, batch)

            self._master.cleargrads()
            with chainer.layer_wise_reduce_mode(self):
                loss.backward()
            optimizer.update()

            if self.need_sync_param and not self.cpu_mode:
                cuda.Stream.null.synchronize()
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, nccl.NCCL_FLOAT,
                                0, cuda.Stream.null.ptr)
                self.need_sync_param = False

    # a callback function
    def run_vnode(self, vnodes=None):
        if len(vnodes) == 0:
            return

        if self.cpu_mode:
            event = cuda.Stream.null.record()
            event.synchronize()
            orders = [x.order for x in vnodes]
            self.cbcl.allreduce(orders)
        else:
            event = cuda.Stream.null.record()
            self._stream.wait_event(event)
            for vnode in vnodes:
                if self.debug:
                    print("allreduce vnode {} size {}"
                          .format(vnode.rank, vnode.grad.size))

                self.comm.allReduce(vnode.grad.data.ptr,
                                    vnode.grad.data.ptr,
                                    vnode.grad.size,
                                    nccl.NCCL_FLOAT,
                                    nccl.NCCL_SUM,
                                    self._stream.ptr)

    # a callback function
    def start_backward(self):
        pass

    # a callback function
    def end_backward(self):
        if self.cpu_mode:
            self.cbcl.wait_for_gradients()
        else:
            self._stream.synchronize()

    def finalize(self):
        if self.cpu_mode:
            if self.cbcl is not None:
                self.cbcl.finalize()
        # stop the process of reading data
        self.data_prefetcher.finalize()
        self.data_prefetcher.join()

        self._send_message(('finalize', None))
        for worker in self._workers:
            worker.join()

        profiler.stop()


def _setup_global_variables(model):
    # global shared gradients on host
    locks = []
    nparams = 0
    for _, param in sorted(model.namedparams()):
        # locks for gradients
        locks.append(Lock())
        nparams += 1

    mp_done_workers = Array(ctypes.c_int, 1*nparams, lock=False)
    done_workers = np.frombuffer(mp_done_workers, dtype=np.int32)
    return (done_workers, locks)


def _calc_loss(model, in_arrays):
    if isinstance(in_arrays, tuple):
        return model(*in_arrays)
    elif isinstance(in_arrays, dict):
        return model(**in_arrays)
    else:
        return model(in_arrays)


class CPUBasedLWR(object):
    def __init__(self, model, global_grad_name, done_workers, locks,
                 devices, device, debug):
        self.model = model
        self.global_grad_name = global_grad_name
        self.done_workers = done_workers
        self.locks = locks
        self.devices = devices
        self.device = device

        self.nstages = len(locks)
        self.nstages_done = 0

        self.offsets = {}  # [order => offset in the local grad]
        self.grad_accumulator = None
        self.work_queue = None
        self.ready_input_grads = six.moves.queue.Queue()
        self.ready_output_grads = None

        self.global_grad = None
        self.d2h_stream = None
        self.h2d_stream = None

        self.debug = debug

    def setup(self):
        self.d2h_stream = cuda.Stream(non_blocking=True)
        self.h2d_stream = cuda.Stream(non_blocking=True)

        # prepare an array (pinned memory) for local gradients
        g_size = 0
        g_dtype = None
        for _, param in sorted(self.model.namedparams()):
            if g_dtype is None:
                g_dtype = param.data.dtype
            g_size += param.data.size
        buf = xp.cuda.alloc_pinned_memory(g_size * g_dtype.itemsize)
        self.local_grad = np.frombuffer(buf, g_dtype, g_size)
        self.local_grad.shape = (g_size,)

        mem = xp.cuda.pinned_memory.SharedPinnedMemory(
            self.global_grad_name, g_size * g_dtype.itemsize)
        mem_ptr = xp.cuda.pinned_memory.PinnedMemoryPointer(mem, 0)
        self.global_grad = np.frombuffer(mem_ptr, g_dtype, g_size)
        self.global_grad.shape = (g_size,)

        # prepare workers to accumulate gradients for layers
        offset = 0
        self.ready_output_grads = multiprocessing.Queue()
        for idx, param in sorted(self.model.namedparams()):
            size = param.data.size
            self.offsets[param._node.order] = (offset, param)
            offset += size

        self.work_queue = multiprocessing.Queue()
        self.grad_accumulator = GradAccumulator(
            self.global_grad, self.local_grad,
            self.done_workers, self.ready_output_grads,
            self.work_queue, self.locks, len(self.devices))
        self.grad_accumulator.daemon = True
        self.grad_accumulator.start()

    def allreduce(self, orders):
        offset, param = self.offsets[orders[0]]
        for order in orders:
            offset, p = self.offsets[order]
            self.ready_input_grads.put((order, p.grad.size, offset))
            ptr = self.local_grad[offset:].ctypes.data_as(ctypes.c_void_p)
            p.grad.data.copy_to_host_async(
                ptr, p.grad.nbytes, self.d2h_stream)

            if self.debug:
                print("allreduce vnode {} order {} size {}"
                      .format(p._node.rank, order, p.grad.size))

        # accumulate the current gradients on host
        self.d2h_stream.add_callback(self._accumulate_grad(), None)

        if self.ready_output_grads.empty():
            return
        stage = self.ready_output_grads.get()
        self._grads_to_gpu(stage)
        self.nstages_done += 1

    def wait_for_gradients(self):
        if self.d2h_stream is not None:
            self.d2h_stream.synchronize()

        # remaining gradients
        while (self.nstages_done != self.nstages):
            if self.ready_output_grads.empty():
                continue
            stage = self.ready_output_grads.get()
            self._grads_to_gpu(stage)
            self.nstages_done += 1
        self.nstages_done = 0

        if self.h2d_stream is not None:
            self.h2d_stream.synchronize()

    def _accumulate_grad(self):
        while not self.ready_input_grads.empty():
            order, size, offset = self.ready_input_grads.get()
            # pass to another process to process
            self.work_queue.put((order, size, offset))

    def _grads_to_gpu(self, stage):
        offset, param = self.offsets[stage]
        ptr = self.global_grad[offset:].ctypes.data_as(ctypes.c_void_p)
        param.grad.data.copy_from_host_async(
            ptr, param.grad.nbytes, stream=self.h2d_stream)

    def finalize(self):
        # stop the workers that accumulates gradients
        self.work_queue.put((-1, -1, -1))  # kill the worker
        self.grad_accumulator.join()


def _create_shared_data(iterator):
    sample_image = iterator.dataset[0]
    batch_size = iterator.batch_size
    data_nbytes = sample_image[0].nbytes * batch_size
    label_nbytes = sample_image[1].nbytes * batch_size
    data_shape = (batch_size,) + sample_image[0].shape
    label_shape = (batch_size,)
    mp_shared_data = xp.cuda.alloc_pinned_memory(data_nbytes)
    mp_shared_label = xp.cuda.alloc_pinned_memory(label_nbytes)

    read_arraysize_data = data_nbytes / np.dtype(np.float32).itemsize
    read_arraysize_label = label_nbytes / np.dtype(np.int32).itemsize
    shared_data = np.frombuffer(mp_shared_data,
                                dtype=np.float32,
                                count=read_arraysize_data).reshape(
                                    data_shape)
    shared_label = np.frombuffer(mp_shared_label,
                                 dtype=np.int32,
                                 count=read_arraysize_label).reshape(
                                          label_shape)
    return (shared_data, shared_label)


class _DataPrefetcher(threading.Thread):
    def __init__(self, iterator, converter, device):
        super(_DataPrefetcher, self).__init__()
        self.iterator = iterator
        self.converter = converter
        self.device = device

        self.free_queue = six.moves.queue.Queue()
        self.full_queue = six.moves.queue.Queue()
        self.shared_data = None
        self.shared_label = None

    def setup(self):
        # prepare a worker to do data prefetch
        sample_image = self.iterator.dataset[0]
        batch_size = self.iterator.batch_size
        data_nbytes = sample_image[0].nbytes * batch_size
        label_nbytes = sample_image[1].nbytes * batch_size
        data_shape = (batch_size,) + sample_image[0].shape
        label_shape = (batch_size,)
        mp_shared_data = xp.cuda.alloc_pinned_memory(data_nbytes)
        mp_shared_label = xp.cuda.alloc_pinned_memory(label_nbytes)

        read_arraysize_data = int(data_nbytes / np.dtype(np.float32).itemsize)
        read_arraysize_label = int(label_nbytes / np.dtype(np.int32).itemsize)
        self.shared_data = np.frombuffer(mp_shared_data,
                                         dtype=np.float32,
                                         count=read_arraysize_data).reshape(
                                             data_shape)
        self.shared_label = np.frombuffer(mp_shared_label,
                                          dtype=np.int32,
                                          count=read_arraysize_label).reshape(
                                              label_shape)
        self.free_queue.put(True)  # prefetch the first batch

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        self.setup()

        while True:
            if self.free_queue.empty():
                continue
            flag = self.free_queue.get()
            if flag:
                next_batch = self.iterator.next()
                cpu_batch = self.converter(next_batch, device=-1)
                np.copyto(self.shared_data, cpu_batch[0])
                np.copyto(self.shared_label, cpu_batch[1])
                self.full_queue.put(True)
            else:
                break

    def batch(self):
        self.full_queue.get()
        data = xp.empty_like(self.shared_data)
        data.set(self.shared_data, cuda.Stream.null)
        label = xp.empty_like(self.shared_label)
        label.set(self.shared_label, cuda.Stream.null)
        self.free_queue.put(True)
        return (data, label)

    def finalize(self):
        # stop the process of reading data
        self.free_queue.put(False)


class GradAccumulator(multiprocessing.Process):
    def __init__(self, global_grad, local_grad,
                 done_workers, ready_output_grads,
                 work_queue, locks, nworkers):
        super(GradAccumulator, self).__init__()

        self.work_queue = work_queue
        self.locks = locks
        self.nworkers = nworkers
        self.done_workers = done_workers
        self.ready_output_grads = ready_output_grads

        self.threshold = 2000000
        self.done_queue = six.moves.queue.Queue()

        self.local_grad = local_grad
        self.global_grad = global_grad

    def setup(self):
        numexpr_num_threads = os.getenv("NUMEXPR_NUM_THREADS")
        if numexpr_num_threads is not None:
            ne.set_num_threads(int(numexpr_num_threads))
        else:
            ne.set_num_threads(16)

    def run(self):
        self.setup()
        while True:
            if not self.done_queue.empty():
                done_stage = self.done_queue.get()
                x = self.done_workers[done_stage]
                if x == self.nworkers:
                    self.ready_output_grads.put(done_stage)
                else:
                    self.done_queue.put(done_stage)

            if self.work_queue.empty():
                continue

            stage, size, offset = self.work_queue.get()
            if stage < 0:
                break
            local_grad_stage = self.local_grad[offset:(offset+size)]
            global_grad_stage = self.global_grad[offset:(offset+size)]
            model_grad = local_grad_stage.copy()
            self.locks[stage].acquire()
            if self.done_workers[stage] == self.nworkers:
                np.copyto(global_grad_stage, model_grad)
                self.done_workers[stage] = 0
            else:
                if size >= self.threshold:
                    tmp = global_grad_stage
                    ne.evaluate("tmp + model_grad", out=global_grad_stage)
                else:
                    global_grad_stage += model_grad
            self.done_workers[stage] += 1
            self.locks[stage].release()
            self.done_queue.put(stage)

        self.work_queue.close()
        self.work_queue.join_thread()
        self.finalize()

    def finalize(self):
        pass


def size_num_grads(link):
    """Count total size of all gradient arrays of a given link

    Args:
        link (chainer.link.Link): Target link object.
    """
    size = 0
    num = 0
    for param in link.params():
        if param.size == 0:
            continue
        size += param.size
        num += 1
    return size, num


def _batch_memcpy():
    return cuda.cupy.ElementwiseKernel(
        'raw T ptrs, raw X info',
        'raw float32 dst',
        '''
            int id_min = id_pre;
            int id_max = num_src;
            while (id_max - id_min > 1) {
                int id = (id_max + id_min) / 2;
                if (i < info[id]) id_max = id;
                else              id_min = id;
            }
            int id = id_min;
            float *src = (float *)(ptrs[id]);
            int i_dst = i;
            int i_src = i;
            if (id > 0) i_src -= info[id];
            dst[i_dst] = 0;
            if (src != NULL) {
                dst[i_dst] = src[i_src];
            }
            id_pre = id;
        ''',
        'batch_memcpy',
        loop_prep='''
                int num_src = info[0];
                int id_pre = 0;
            ''')


def _gather(link, target):
    size, num = size_num_grads(link)

    ptrs = np.empty(num, dtype=np.uint64)
    info = np.empty(num + 1, dtype=np.int32)
    info[0] = 0
    i = 0
    for _, param in sorted(link.namedparams()):
        if param.size == 0:
            continue
        ptrs[i] = 0  # NULL pointer
        d = getattr(param, target)
        if d is not None:
            ptrs[i] = d.data.ptr
        info[i + 1] = info[i] + param.size
        i += 1
    info[0] = num

    ptrs = cuda.to_gpu(ptrs, stream=cuda.Stream.null)
    info = cuda.to_gpu(info, stream=cuda.Stream.null)

    return _batch_memcpy()(ptrs, info, size=size)


def gather_params(link):
    """Put together all gradient arrays and make a single array

    Args:
        link (chainer.link.Link): Target link object.
    Return:
        cupy.ndarray
    """
    if link.xp is np:
        raise RuntimeError('Link.gather_params works only on GPU.')
    return _gather(link, "data")


def scatter_params(link, array):
    """Put back contents of the specified array to the related gradient arrays

    Args:
        link (chainer.link.Link): Target link object.
        array (cupy.ndarray): gathered array created by gather_params()
    """
    offset = 0
    for _, param in sorted(link.namedparams()):
        next_offset = offset + param.size
        param.data = array[offset:next_offset].reshape(param.data.shape)
        offset = next_offset
    assert array.size == offset
