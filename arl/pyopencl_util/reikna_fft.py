import numpy


from reikna.fft import FFT,FFTShift
import reikna.cluda.dtypes as dtypes
import reikna.cluda as cluda


 
def reikna_fftn(arr, inverse = 0):
    """
    If ``0``, ``output`` contains the forward FFT of ``input``,
    if ``1``, the inverse one.
    """
    api = cluda.ocl_api()
    thr = api.Thread.create()
    result = numpy.empty_like(arr)

    arr = arr.astype(numpy.complex64)

    if (len(arr.shape) == 4):
        result = reikna_fft4d(arr,result,thr,inverse)
    else:
        result = reikna_fft2d(arr,thr,inverse)
        
    thr.release()
    return result
 
    
def reikna_fftshift(thr, arr,arr_dev, axes =None):   
    shift = FFTShift(arr,axes=axes)
    shiftc = shift.compile(thr)
    shiftc(arr_dev, arr_dev)

def reikna_fft2(thr,arr,arr_dev, arr_ret, axes = None, inverse = 0):
    fft = FFT(arr_dev, axes)
    fftc = fft.compile(thr)
    fftc(arr_ret, arr_dev, inverse)



def reikna_fft4d(arr,result,thr,inverse = 0):
    a,b,c,d = arr.shape
  
    for m in range (a):
        for n in range (b):
            tmp = arr[m,n,:,:]
            result[m,n,:,:] = reikna_fft2d(tmp,thr,inverse)
    return result
 
def reikna_fft2d(arr,thr, inverse = 0):

    arr_dev = thr.to_device(arr)
    #arr_ret = thr.array(arr.shape, dtype=my_type)
    arr_ret = arr_dev

    axes = None
    reikna_fftshift(thr, arr, arr_dev, axes)
    reikna_fft2(thr,arr,arr_dev, arr_ret, axes,inverse)
    reikna_fftshift(thr, arr, arr_ret, axes)
    return arr_ret.get()
