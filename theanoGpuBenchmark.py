# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:24:04 2016

@author: lynx

Simple gpu benchmark with theano

http://deeplearning.net/software/theano/tutorial/using_gpu.html
"""
import theano
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

def benchmark_1():
    """
    Computes the exp function over a random array
    """
    print '\n\nBenchmark 1. Computes the exp function over a random array'
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 10000
    
    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
    return t1-t0
    
def benchmark_2():
    """
    Convolution benchmark
    """
    print '\n\nBenchmark 2. Convolution benchmark'
    iters = 1000
    
    """
    images_shape((batch_size, channels, rows, cols))
    filters_shape.append((nkern, channels, rows, cols))
    """
    image_shape = (10,3,100,100)
    filter_shape = (16,3,5,5)
    
    # initialize weights with random weights
    rng = numpy.random.RandomState(22)
    W_bound = 1
    W = theano.shared(
        numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    
    
    rng.uniform()
    x = shared(numpy.asarray(rng.uniform(size=image_shape), config.floatX))
    conv_out = theano.tensor.nnet.conv.conv2d(
                    input=x,
                    filters=W,
                    filter_shape=filter_shape,
                    image_shape=image_shape
                )
    f = function([], conv_out)
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
    return t1-t0
        
if __name__ == '__main__':
    benchmark_1()
    benchmark_2()