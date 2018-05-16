import functools

from chainer import cuda
from chainer.training import extension
from chainer.training.extensions.evaluator import Evaluator
from cupy.cuda import cudnn_util
_log_enabled = False

def log(msg):
    if _log_enabled:
        print(msg, file=sys.stderr, flush=True)


class WorkspaceAllocator(extension.Extension):

    """Trainer extension to accelerate forward and backward computation in
    convolutional networks.

    This extension finds size of memory space required by algorithms provided by
    cuDNN library and allocates it automatically.  Memory spaces will be
    allocated for the process which calls this object.

    Args:
        devices: Device list where memory spaces are allocated.  Negative values
            are not supported.
        ratio: Ratio of the size of the allocated memory space to the size of
            free avaiable GPU memory. It should be equal to or more than
            0 and equal to or less than 1.

    Attributes:
        devices: Device list where memory spaces are allocated.
        ratio: Ratio of the size of the allocated memory space to the size of
            free avaiable GPU memory.
    """
    def __init__(self, devices, ratio=0.9):
        log('WorkspaceAllocator::__init__ called')
        self.devices = devices
        self.ratio = ratio

    def __call__(self, trainer=None):
        """Allocates memory space for cuDNN functions.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        """
        log('WorkspaceAllocator::__call__ called')
        cudnn_util.allocate_workspace(self.devices, self.ratio)

    def finalize(self):
        """Free memory space allocated for cuDNN functions.
        """
        log('WorkspaceAllocator::finalize called')
        for device in self.devices:
            with cuda.get_device(device):
                cudnn_util.free_workspace(device)


class WorkspaceEvaluator(Evaluator):

    """Trainer extension to evaluate models with acceleration of
    convoluational forward and backward.

    This extension allocates memory space used by algorithms provided by
    cuDNN library. This class is based on
    :class:`~chainer.training.extensions.evaluator.Evaluator`

    Args:
       iterator: Dataset iterator for the validation dataset. It can also be
            a dictionary of iterators. If this is just an iterator, the
            iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        device: Device to which the training data is sent and memory space for
            cuDNN functions are allocated. Negative values are not supported.
        ratio: Ratio of the size of the allocated memory space to the size of
            free avaiable GPU memory. It should be equal to or more than
            0 and equal to or less than 1.
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.

    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent and memory space for
            cuDNN functions are allocated.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        ratio: Ratio of the size of the allocated memory space to the size of
            free avaiable GPU memory.
    """
    def __init__(self, iterator, target, device, ratio=0.9, **kwargs):
        log('WorkspaceEvaluator::__init__ called')
        super(WorkspaceEvaluator, self).__init__(iterator, target,
                                                 device=device, **kwargs)
        self.ratio = ratio
        eval_func = self.eval_func or self._targets['main']
        self.eval_func = functools.partial(self._eval, eval_func)

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method invokes the evaluate function of
        :class:`~chainer.training.extensions.evaluator.Evaluator`

        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.
        """
        log('WorkspaceEvaluator::evaluate called')
        return super(WorkspaceEvaluator, self).evaluate()

    def _eval(self, eval_func, *args, **kwargs):
        log('WorkspaceEvaluator::_eval called')
        cudnn_util.allocate_workspace([self.device], self.ratio)
        eval_func(*args, **kwargs)
