import sys
import warnings

from chainer.backends import cuda
from chainer import function_hook
from chainer import variable
import chainer


class LMS(function_hook.FunctionHook):
    """Function hook that prints debug information.

    This function hook enables '`large model support,``  called LMS.
    LMS swap-out GPU memories for ndarrays not to be used soon to CPU memory,
    and swap-in it before using them again.
    With LMS, models cusuming larger memory than GPU's can be executed.

    Args:
        logfile: File name for log
        maxswap: maxinum number of layers to be swap-out
        numoverlap: number of layers to start swap-in the ndarray before using
        swap_functions: set of function names to be swap-out
        excl_functions: set of function names not to be swap-out

    .. admonition:: Example

        The basic usage is to use it with ``with`` statement.

        >>> from chainer import function_hooks
        >>> l = L.Linear(10, 10)
        >>> x = chainer.Variable(np.zeros((1, 10), np.float32))
        >>> with chainer.function_hooks.LMS():
        ...     y = l(x)
        ...     z = F.sum(y)
        ...     z.backward() # doctest:+SKIP

    """

    name = 'LMS'

    def __init__(self, logfile=None, maxswap=100, numoverlap=1, numsustain=1,
                 minswapsize=8192,
                 swap_functions=["LinearFunction", "Convolution2DFunction",
                                 "MaxPooling2D"],
                 excl_functions=["SoftmaxCrossEntropy", "Accuracy",
                                 "LinearGradData", "LinearGradWeight",
                                 "ReLUGrad2", "Reshape", "Dropout",
                                 "ReLU", "Sum", "DropoutGrad",
                                 "Deconvolution2DFunction",
                                 "Convolution2DGradW", "MaxPooling2DGrad"]):
        self.maxswap = maxswap
        self.numoverlap = numoverlap
        self.numsustain = numsustain
        self.minswapsize = minswapsize
        self.swap_functions = swap_functions # []
        #self.excl_functions = excl_functions
        self.numfunc = 0
        self.swapped = []
        self.using = []
        self.logfile = logfile

    def log(self, msg):
        if self.logfile is not None:
            self.logfile.write(msg + '\n')
            self.logfile.flush()

    def forward_preprocess(self, function, in_data):
        if not chainer.config.train:
            return
        self.log('FORWARD-PRE: function {} #{} rank={}'
                 .format(function.label, self.numfunc, function.rank))
        
    def forward_postprocess(self, function, in_data):
        if not chainer.config.train:
            return
        self.log('FORWARD-POST: function {} #{} rank={}'
                 .format(function.label, self.numfunc, function.rank))
        if (self.swap_functions and function.label in self.swap_functions):
            # or  (function.label not in self.excl_functions):
            if len(self.swapped) < self.maxswap:
                for d in in_data:
                #for d in function.get_retained_inputs():
                    #self.log("AAAA: ={}".format(type(d)))
                    if d is not None and (d.size * 4) >= self.minswapsize:
                        self.log("SWAPOUT: {} {} {} {} {} {}"
                                 .format(function.label, hex(id(d)),
                                         d.size * 4, hex(d.data.mem.ptr),
                                         str(d.shape).replace(" ", ""),
                                         d.dtype))
                        d.swapout()
                self.swapped.append((function.label, in_data))
            self.numfunc += 1
            self.log('FORWARD LAYER: function {} #{} rank={}'
                     .format(function.label, self.numfunc, function.rank))

    def backward_preprocess(self, function, in_data, out_grad):
        if not chainer.config.train:
            return
        self.log('BACKWARD-PRE: function {} #{} rank={}'
                 .format(function.label, self.numfunc, function.rank))
        if (self.swap_functions and function.label in self.swap_functions):
            # or (function.label not in self.excl_functions):
            self.log('BACKWARD LAYER: function {} #{} rank={}'
                     .format(function.label, self.numfunc, function.rank))
            self.numfunc -= 1
        while len(self.swapped) > 0 and \
          (self.numfunc - self.numoverlap) < len(self.swapped):
            swap_function, swap_data = self.swapped.pop()
            #swap_function, swap_data = self.swapped[len(self.swapped) - 1]
            #del self.swapped[len(self.swapped) - 1]
        #if True:
        #    swap_function swap_data = "XXXXX", \
        #                              list(function.get_retained_inputs())
        #    outputs = function.get_retained_outputs()
        #    if outputs is None:
        #        self.log("BACKWARD-PRE: END (NO OUTPUTS)")
        #        return
        #    swap_data.extend(outputs)
            for d in swap_data:
                if d is not None and d.is_swapout:
                    self.log("SWAPIN: {} {} {} {} {}"
                             .format(swap_function, hex(id(d)), d.size * 4,
                                     str(d.shape).replace(" ", ""), d.dtype))
                    d.swapin()
                    # del d
            #del swap_function
            #del swap_data

    def backward_postprocess(self, function, in_data, out_grad):
        if not chainer.config.train:
            return
        self.log('BACKWARD-POST: function {} #{} rank={}'
                 .format(function.label, self.numfunc, function.rank))
                
        #while (len(self.swapped) + len(self.using) > (self.numfunc + 3)):
        #    using_function, using_data = self.using.pop()
        #    for d in using_data:
        if (self.swap_functions and function.label in self.swap_functions):
            self.using.append((function.label, in_data))
            if (len(self.using) > self.numsustain):
                using_function, using_data = self.using.pop(0)
                n = 0
                for d in using_data:
                    if d is not None:
                        self.log("DELETE: {}:{} {} {} {} {}"
                                 .format(using_function, n, hex(id(d)),
                                         d.size * 4,
                                         str(d.shape).replace(" ", ""),
                                         d.dtype))
                        d.deldata()
                        n += 1
                        break
