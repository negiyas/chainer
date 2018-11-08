import sys
import warnings

from chainer.backends import cuda
from chainer import function_hook
from chainer import variable
import chainer


class LMS(function_hook.FunctionHook):
    """Function hook that prints debug information.
    This function hook outputs the debug information of input arguments of
    ``forward`` and ``backward`` methods involved in the hooked functions
    at preprocessing time (that is, just before each method is called).
    Unlike simple "debug print" technique, where users insert print functions
    at every function to be inspected, we can show the information
    of all functions involved with single ``with`` statement.
    Further, this hook enables us to show the information of
    ``backward`` methods without inserting print functions into
    Chainer's library code.
    Args:
        end: Character to be added at the end of print function.
        file: Output file_like object that that redirect to.
        flush: If ``True``, this hook forcibly flushes the text stream
            at the end of preprocessing.
    .. admonition:: Example
        The basic usage is to use it with ``with`` statement.
        >>> from chainer import function_hooks
        >>> l = L.Linear(10, 10)
        >>> x = chainer.Variable(np.zeros((1, 10), np.float32))
        >>> with chainer.function_hooks.OutOfCore():
        ...     y = l(x)
        ...     z = F.sum(y)
        ...     z.backward() # doctest:+SKIP
        In this example, ``PrintHook`` shows the debug information of
        forward propagation of ``LinearFunction`` (which is implicitly
        called by ``l``) and ``Sum`` (called by ``F.sum``)
        and backward propagation of ``z`` and ``y``.
    """

    name = 'LMS'

    def __init__(self, logfile=None, maxswap=100, numoverlap=1,
                 minswapsize=8192,
                 swap_functions=[ "LinearFunction", "Convolution2DFunction",
                                  "MaxPooling2D"],
                 excl_functions=[ "SoftmaxCrossEntropy", "Accuracy",
                                  "LinearGradData", "LinearGradWeight",
                                  "ReLUGrad2", "Reshape", "Dropout",
                                  "ReLU", "Sum", "DropoutGrad",
                                  "Deconvolution2DFunction",
                                  "Convolution2DGradW", "MaxPooling2DGrad" ]):
        self.maxswap = maxswap
        self.numoverlap = numoverlap
        self.minswapsize = minswapsize
        self.swap_functions = swap_functions # []
        self.excl_functions = excl_functions
        self.numfunc = 0
        self.swapped = []
        self.logfile = logfile

    def log(self, msg):
        self.logfile.write(msg + '\n')
        self.logfile.flush()

    def forward_postprocess(self, function, in_data):
        if not chainer.config.train:
            return
        if self.logfile is not None:
            self.log('FORWARD: function {}'.format(function.label))
        if (self.swap_functions and function.label in self.swap_functions) or \
           (function.label not in self.excl_functions):
            if len(self.swapped) < self.maxswap:
                for d in in_data:
                    if d is not None and (d.size * 4) >= self.minswapsize:
                        if self.logfile is not None:
                            self.log("SWAPOUT: {} {} {} {} {} {}"
                                     .format(function.label, hex(id(d)),
                                             d.size * 4, hex(d.data.mem.ptr),
                                             str(d.shape).replace(" ", ""),
                                             d.dtype))
                        d.swapout()
                self.swapped.append((function.label, in_data))
            self.numfunc += 1

    def backward_preprocess(self, function, in_data, out_grad):
        if not chainer.config.train:
            return
        if self.logfile is not None:
            self.log('BACKWARD: function {}'.format(function.label))
        if (self.swap_functions and function.label in self.swap_functions) or \
           (function.label not in self.excl_functions):
            self.numfunc -= 1
        while len(self.swapped) > 0 and \
          (self.numfunc - self.numoverlap) < len(self.swapped):
            swap_function, swap_data = self.swapped.pop()
            for d in swap_data:
                if d is not None and d.is_swapout:
                    if self.logfile is not None:
                        self.log("SWAPIN: {} {} {} {} {}"
                                 .format(swap_function, hex(id(d)), d.size * 4,
                                         str(d.shape).replace(" ", ""), d.dtype))
                    d.swapin()

    def backward_postprocess(self, function, in_data, out_grad):
        if not chainer.config.train:
            return
        if self.logfile is not None:
            self.log('DELETE: function {}'.format(function.label))
        for d in in_data:
            if d is not None:
                if self.logfile is not None:
                    self.log("DELETE: {} {} {} {} {}"
                            .format(function.label, hex(id(d)), d.size * 4,
                                    str(d.shape).replace(" ", ""), d.dtype))
                del d
        del in_data
