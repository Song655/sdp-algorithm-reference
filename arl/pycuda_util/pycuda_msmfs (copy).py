# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
# Initialize the CUDA device
import pycuda.autoinit
 
import numpy
import logging
import math

from arl.image.cleaners import *

log = logging.getLogger(__name__)

def cuda_compile(source_string, function_name):
  print ("Compiling a CUDA kernel...")
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

cuda_hogbom_kernel_source = """
#define BLOCK_SIZE  512
#define LOCAL_M_SIZE 5  // the value should >= nmoments

/**
find the max value in the array
g_idata: input array
g_oValue: output max value array
g_oIndex: output max value index array
size :  the size of the input array
abs: True of False. Ture mean to get the absolute max value
*/
__global__ void findMax_kernel(double *g_idata, double *g_oValue, unsigned int *g_oIndex, unsigned int size, int abs) {
    __shared__ volatile double s_value[BLOCK_SIZE];
    __shared__ volatile unsigned int s_index[BLOCK_SIZE];  

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x* BLOCK_SIZE *2 + tid;
    unsigned int gridSize = BLOCK_SIZE*2*gridDim.x;

    double temp1, temp2;

    while ((i+BLOCK_SIZE ) < size) {
        if(abs){
            temp1 = fabs(g_idata[i+BLOCK_SIZE]);
            temp2 = fabs(g_idata[i]);
        }
        else{
            temp1 = g_idata[i+BLOCK_SIZE];
            temp2 = g_idata[i];
        }

	if(temp1 > temp2){
		s_value[tid]= temp1;
		s_index[tid] = i+BLOCK_SIZE;
	}
	else{
		s_value[tid] = temp2;
		s_index[tid] = i;
	}
	i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
	    if(s_value[tid+256] > s_value[tid]) {
		s_value[tid] = s_value[tid+256];
		s_index[tid] = s_index[tid+256];
	    }
        } __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
	    if(s_value[tid+128]  > s_value[tid]) {
		s_value[tid] = s_value[tid+128];
		s_index[tid] = s_index[tid+128];
	    }
        } __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
	    if(s_value[tid+64] > s_value[tid]) {
		s_value[tid] = s_value[tid+64];
		s_index[tid] = s_index[tid+64];
	    }
        } __syncthreads();
    }
    if (tid < 32) {
        if(s_value[tid+32] > s_value[tid]) {
	    s_value[tid] = s_value[tid+32];
	    s_index[tid] = s_index[tid+32];
        } 
  
        if(s_value[tid+16] > s_value[tid]) {
	    s_value[tid] = s_value[tid+16];
	    s_index[tid] = s_index[tid+16];
        } 

        if(s_value[tid+8] > s_value[tid]) {
	    s_value[tid] = s_value[tid+8];
	    s_index[tid] = s_index[tid+8];
        }

        if(s_value[tid+4] > s_value[tid]) {
	    s_value[tid] = s_value[tid+4];
	    s_index[tid] = s_index[tid+4];
        } 

        if(s_value[tid+2] > s_value[tid]) {
 	    s_value[tid] = s_value[tid+2];
	    s_index[tid] = s_index[tid+2];
        } 

        if(s_value[tid+1] > s_value[tid]) {
	    s_value[tid] = s_value[tid+1];
	    s_index[tid] = s_index[tid+1];
        } 
    
    }
    // write result for this block to global mem
    if (tid == 0) {
	g_oValue[blockIdx.x] = s_value[0];
	g_oIndex[blockIdx.x] = s_index[0];
    }
}


/**
ihsmmpsf: nscales, nmoments, nmoments
smresidual: nscales, nmoments, nx, ny
smpsol = numpy.einsum("smn,smxy->snxy", ihsmmpsf, smresidual)

here we do: 
  -smpsol = numpy.einsum("mn,mxy->nxy", ihsmmpsf, smresidual)
  -smpsol = smpsol*smresidual
*/
__global__ void calculate_scale_moment_principal_solution_kernel(double *ihsmmpsf, double *smresidual, double *smpsol, unsigned int nmoments, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*BLOCK_SIZE + tid;

    extern __shared__ volatile double s_ihsmmpsf[];
    double l_smresidual[LOCAL_M_SIZE];
    
    if(tid < nmoments * nmoments)
    {   
       s_ihsmmpsf[tid] = ihsmmpsf[tid];
    }
    __syncthreads();

    unsigned i,j;
    double tempsum;
    if (idx < size)
    {
        for(j=0; j<nmoments; j++)
        {
           l_smresidual[j] = smresidual[j*size + idx];
        }

        for(j=0; j<nmoments; j++)
        { 
            tempsum = 0 ;
            for(i=0; i<nmoments; i++)
            {
                tempsum += s_ihsmmpsf[i*nmoments+j] * l_smresidual[i];
                
            }
            smpsol[j*size + idx] = tempsum * l_smresidual[j];          
        }
    }
}

//[moments, nx, ny]  --> [moments]
//shape[0-2] = moments, nx, ny
//shape[3,4] = mx, my (max vale index)
__global__ void get_mval_kernel(double *smpsol, double *smresidual, double* mval,unsigned int *shape)
{
    unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x;

    if (ix < shape[0])
    {
        unsigned int id1 = ix*shape[1]*shape[2] + shape[3]*shape[2] + shape[4];
        mval[ix] = smpsol[id1]/smresidual[id1];
    }

}


/**
    para[0],para[1]    #res_lowerx,res_lowery 
    para[2],para[3]   #res_upperx,res_uppery 
    para[4],para[5]   #psf_lowerx,psf_lowery 
    para[6],para[7]   #psf_upperx,psf_uppery 
*/
__global__ void update_moment_model_residual_kernel(double *m_model, double* scalestack, double* smresidual, double *ssmmpsf,
double* mval, unsigned int* para, double gain, unsigned int nmoments, unsigned int nscales,unsigned int mscale, unsigned int px, unsigned int py, unsigned int nx, unsigned int ny )
{
    unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int res_idx =ix + para[0];
    unsigned int res_idy =iy + para[1];

    unsigned int psf_idx =ix + para[4];
    unsigned int psf_idy =iy + para[5];

    unsigned int rex_Id = res_idx*ny + res_idy;
    unsigned int psf_Id = psf_idx*py + psf_idy;

    int i,j,k,d0;
    double temp;

    double l_mval[LOCAL_M_SIZE];
     
    if (ix+para[0] < para[2] && iy+para[1] < para[3]) 
    {
         for (i = 0; i< nmoments; i++)
         {
             l_mval[i] = mval[i];
         }
    
         for (i = 0; i< nmoments; i++)
         {
             m_model[i*nx*ny + rex_Id] += scalestack[mscale*px*py + psf_Id]*gain*l_mval[i];
             for (j = 0; j < nscales; j++)
             {
                 temp = 0; 
                 for (k = 0; k< nmoments; k++)
                 {
                     d0 = mscale* nscales*nmoments*nmoments*px*py + j*nmoments*nmoments*px*py + i*nmoments*px*py + k*px*py;
                     temp += ssmmpsf[d0+ psf_Id] * l_mval[k];
                 }
                 smresidual[j*nmoments*nx*ny + i*nx*ny + rex_Id] -= temp*gain;
             }
         }
    }

}

__global__ void pmax_residual_kernel(double* smresidual, double *re_smres, double pmax, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*BLOCK_SIZE + tid;

    if (idx < size)
    {
        re_smres[idx] = smresidual[idx]*pmax;
    }
}

"""
cuda_hogbom_kernel = nvcc.SourceModule(cuda_hogbom_kernel_source,options=['-O3'])
cuda_findmax_core = cuda_hogbom_kernel.get_function("findMax_kernel")
cuda_calculate_scale_moment_principal_solution_core = cuda_hogbom_kernel.get_function("calculate_scale_moment_principal_solution_kernel")
cuda_get_mval_core = cuda_hogbom_kernel.get_function("get_mval_kernel")
cuda_update_moment_model_residual_core = cuda_hogbom_kernel.get_function("update_moment_model_residual_kernel")
cuda_pmax_residual_core = cuda_hogbom_kernel.get_function("pmax_residual_kernel") 


THREAD_NUM = 512
THREAD_X = 32
THREAD_Y = 32

def idxToPos_2D(idx,width):
    x = idx // width
    y = idx % width
    return x,y

def cuda_findMax_2D(gpu_array,arraySize, absmax, getIndex=False):
    BLOCK_Count  =math.ceil(arraySize/(THREAD_NUM*2.0));
    cpu_value = numpy.zeros(BLOCK_Count,dtype=numpy.float64,order='C')
    cpu_index = numpy.zeros(BLOCK_Count, dtype=numpy.uint32,order='C')  
    if absmax:
        absmax = numpy.uint32(1)
    else:
        absmax = numpy.uint32(0)

    cuda_findmax_core(gpu_array,cuda.Out(cpu_value),cuda.Out(cpu_index),arraySize, absmax, block=(THREAD_NUM,1,1),grid=(BLOCK_Count,1))

    if getIndex:  
        pos = numpy.unravel_index(cpu_value.argmax(), cpu_value.shape)
        mx,my = idxToPos_2D(cpu_index[pos],gpu_array.shape[1]) 
        return cpu_value[pos], mx, my 
    else:
        return cpu_value.max(), 0, 0

def cuda_findMax(A, absmax=False, getIndex=False):
    nm,nx,ny = A.shape
    sn = 0
    sx = 0
    sy = 0
    maxv = 0.0
 
    for i in range (nm):
        gpu_A = gpuarray.to_gpu(A[i,:,:])
        mv, mx, my = cuda_findMax_2D(gpu_A,numpy.uint32(nx*ny),absmax,getIndex)
        if mv > maxv:
            maxv = mv
            sx = mx
            sy = my
            sn = i
    return maxv, sn, sx, sy

def cuda_find_global_optimum_default(ihsmmpsf, smresidual, windowstack):
    s,m,n = ihsmmpsf.shape
    s,m,x,y = smresidual.shape
    gridx  = math.ceil(x*y*1.0/THREAD_NUM)   
    sn = 0
    sx = 0
    sy = 0
    maxv = 0.0
    gpu_mavl = gpuarray.zeros((n),dtype=numpy.float64)
    gpu_smpsol = gpuarray.zeros((n,x,y),dtype=numpy.float64)   
    for i in range (s):
        gpu_ihsmmpsf = ihsmmpsf[i,:,:]
        #gpu_smresidual = gpuarray.to_gpu(smresidual[i,:,:,:]) 
        gpu_smresidual = smresidual[i,:,:,:]   
        cuda_calculate_scale_moment_principal_solution_core(gpu_ihsmmpsf, gpu_smresidual, gpu_smpsol, numpy.uint32(m),numpy.uint32(x*y),block=(THREAD_NUM,1,1),grid=(gridx,1),shared = m*n)
        mv, mx, my = cuda_findMax_2D(gpu_smpsol[0,:,:], numpy.uint32(x*y),absmax=True, getIndex=True)
        if mv > maxv:
            maxv = mv
            sx = mx
            sy = my
            sn = i
            shape_a = numpy.array([n,x,y,mx,my]).astype(numpy.uint32)
            gridxx  = math.ceil(n*1.0/THREAD_X)
            cuda_get_mval_core(gpu_smpsol, gpu_smresidual, gpu_mavl,cuda.In(shape_a),block=(THREAD_X,1,1),grid=(gridxx,1))
    return sn, sx, sy, gpu_mavl

def cuda_find_global_optimum(hsmmpsf, ihsmmpsf, smresidual, windowstack, findpeak):
    """Find the optimum peak using one of a number of algorithms

    """
    if findpeak == 'Algorithm1':
        # Calculate the principal solution in moment-moment axes. This decouples the moments
        smpsol = calculate_scale_moment_principal_solution(smresidual, ihsmmpsf)
        # Now find the location and scale
        mx, my, mscale = find_optimum_scale_zero_moment(smpsol, windowstack)
        mval = smpsol[mscale, :, mx, my]
    elif findpeak == 'CASA':
        # CASA 4.7 version
        smpsol = calculate_scale_moment_principal_solution(smresidual, ihsmmpsf)
        #        smpsol = calculate_scale_moment_approximate_principal_solution(smresidual, hsmmpsf)
        nscales, nmoments, nx, ny = smpsol.shape
        dchisq = numpy.zeros([nscales, 1, nx, ny])
        for scale in range(nscales):
            for moment1 in range(nmoments):
                dchisq[scale, 0, ...] += 2.0 * smpsol[scale, moment1, ...] * smresidual[scale, moment1, ...]
                for moment2 in range(nmoments):
                    dchisq[scale, 0, ...] -= hsmmpsf[scale, moment1, moment2] * \
                                             smpsol[scale, moment1, ...] * smpsol[scale, moment2, ...]
        
        mx, my, mscale = find_optimum_scale_zero_moment(dchisq, windowstack)
        mval = smpsol[mscale, :, mx, my]
    
    else:
        mscale, mx, my, mval = cuda_find_global_optimum_default(ihsmmpsf, smresidual, windowstack)
    
    return mscale, mx, my, mval 

def cuda_overlapIndices(res, psf, peakx, peaky):
    nx, ny = res.shape[0], res.shape[1]
    psfwidthx, psfwidthy = psf.shape[0] // 2, psf.shape[1] // 2
    psfpeakx, psfpeaky = psf.shape[0] // 2, psf.shape[1] // 2
    # Step 1 line up the coordinate ignoring limits
    para = numpy.zeros(8,dtype=numpy.int32,order='C')
    para[0], para[1] = (max(0, peakx - psfwidthx), max(0, peaky - psfwidthy))
    para[2], para[3] = res_upper = (min(nx, peakx + psfwidthx), min(peaky + psfwidthy, ny))
    para[4], para[5] = psf_lower = (max(0, psfpeakx + (para[0] - peakx)), max(0, psfpeaky + (para[1] - peaky)))
    para[6], para[7] =  psf_upper = (
        min(psf.shape[0], psfpeakx + (para[2] - peakx)), min(psfpeaky + (para[3] - peaky), psf.shape[1]))
    
    return para



def cuda_update_moment_model_residual(gpu_m_model, gpu_scalestack, gpu_smresidual, gpu_ssmmpsf, para, gain, mscale, gpu_mval):
    nscales, nmoments, nx, ny = gpu_smresidual.shape
    _,_,_,_,px,py = gpu_ssmmpsf.shape
    thread1 = THREAD_X
    thread2 = THREAD_Y 
    grid1 = math.ceil(1.0*(para[2]-para[0])/thread1)
    grid2 = math.ceil(1.0*(para[3]-para[1])/thread2)
    
    cuda_update_moment_model_residual_core(gpu_m_model, gpu_scalestack, gpu_smresidual, gpu_ssmmpsf, gpu_mval, cuda.In(para),\
    numpy.float64(gain), numpy.uint32(nmoments), numpy.uint32(nscales), numpy.uint32(mscale), numpy.uint32(px), numpy.uint32(py), numpy.uint32(nx),numpy.uint32(ny),\
    block=(thread1,thread2,1),grid=(grid1,grid2))
                

def cuda_return_res_zero(gpu_smresidual, pmax):
    nscales, nmoments, nx, ny = gpu_smresidual.shape
    thread1 = THREAD_NUM
    asize = nx*ny*nmoments
    grid1 = math.ceil(1.0*asize/thread1)
    gpu_res = gpuarray.zeros((nmoments, nx, ny),dtype=numpy.float64)
    cuda_pmax_residual_core(gpu_smresidual[0,:,:,:], gpu_res, numpy.float64(pmax), numpy.uint32(asize), block=(thread1,1,1),grid=(grid1,1))
    return gpu_res.get_async()


def cuda_msmfsclean(dirty, psf, window, gain, thresh, niter, scales, fracthresh, findpeak='CASA'):
    """ Perform image plane multiscale multi frequency clean

    This algorithm is documented as Algorithm 1 in: U. Rau and T. J. Cornwell, 鈥淎 multi-scale multi-frequency
    deconvolution algorithm for synthesis imaging in radio interferometry,鈥?A&A 532, A71 (2011). Note that
    this is only the image plane parts.

    Specific code is linked to specific lines in that algorithm description.

    This version operates on numpy arrays that have been converted to moments on the last axis.

    :param fracthresh:
    :param dirty: The dirty image, i.e., the image to be deconvolved
    :param psf: The point spread-function
    :param window: Regions where clean components are allowed. If True, all of the dirty image is allowed
    :param gain: The "loop gain", i.e., the fraction of the brightest pixel that is removed in each iteration
    :param thresh: Cleaning stops when the maximum of the absolute deviation of the residual is less than this value
    :param niter: Maximum number of components to make if the threshold "thresh" is not hit
    :param scales: Scales (in pixels width) to be used
    :param fracthres: Fractional stopping threshold
    :param ntaylor: Number of Taylor terms
    :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'CASA'|'ARL', Default is ARL.
    :return: clean component image, residual image
    """

    assert 0.0 < gain < 2.0
    assert niter > 0
    assert len(scales) > 0
    log.info("cuda_msmfsclean")
    gpu_m_model  = gpuarray.zeros(dirty.shape,dtype=numpy.float64)
    
    nscales = len(scales)
   
    pmax,pmx,pmy,pmz = cuda_findMax(psf, absmax=True, getIndex=True)
    assert pmax > 0.0
    log.info("msmfsclean: Peak of PSF = %s at %s" % (pmax, (pmx,pmy,pmz)))
 
    dmax,dmx,dmy,dmz = cuda_findMax(dirty, absmax=True, getIndex=True)
    log.info("msmfsclean: Peak of Dirty = %s at %s" % (dmax, (dmx,dmy,dmz)))
    lpsf = psf / pmax
    ldirty = dirty / pmax
    
    nmoments, ny, nx = dirty.shape
    assert psf.shape[0] == 2 * nmoments
    
    # Create the "scale basis functions" in Algorithm 1
    scaleshape = [nscales, ldirty.shape[1], ldirty.shape[2]]
    scalestack = create_scalestack(scaleshape, scales, norm=True)

    pscaleshape = [nscales, lpsf.shape[1], lpsf.shape[2]]
    pscalestack = create_scalestack(pscaleshape, scales, norm=True)

    gpu_pscalestack= gpuarray.to_gpu_async(pscalestack)
    
    # Calculate scale convolutions of moment residuals
    smresidual = calculate_scale_moment_residual(ldirty, scalestack)

    gpu_smresidual = gpuarray.to_gpu_async(smresidual)
    
    # Calculate scale scale moment moment psf, Hessian, and inverse of Hessian
    # scale scale moment moment psf is needed for update of scale-moment residuals
    # Hessian is needed in calculation of optimum for any iteration
    # Inverse Hessian is needed to calculate principal soluation in moment-space    
    ssmmpsf = calculate_scale_scale_moment_moment_psf(lpsf, pscalestack)
    hsmmpsf, ihsmmpsf = calculate_scale_inverse_moment_moment_hessian(ssmmpsf)
    
    gpu_ssmmpsf = gpuarray.to_gpu_async(ssmmpsf)
    gpu_ihsmmpsf = gpuarray.to_gpu_async(ihsmmpsf)
    
    for scale in range(nscales):
        log.info("msmfsclean: Moment-moment coupling matrix[scale %d] =\n %s" % (scale, hsmmpsf[scale]))
    
    # The window is scale dependent - we form it by smoothing and thresholding
    # the input window. This prevents components being placed too close to the
    # edge of the Image.
    
    if window is None:
        windowstack = None
    else:
        windowstack = numpy.zeros_like(scalestack)
        windowstack[convolve_scalestack(scalestack, window) > 0.9] = 1.0

    smresidual_max,_,_ = cuda_findMax_2D(gpu_smresidual[0,0,:,:],arraySize=numpy.uint32(nx*ny), absmax=True, getIndex=False)

    absolutethresh = max(thresh, fracthresh * smresidual_max)

    log.info("msmfsclean: Max abs in dirty Image = %.6f" % smresidual_max)  
    log.info("msmfsclean: Start of minor cycle")
    log.info("msmfsclean: This minor cycle will stop at %d iterations or peak < %s" % (niter, absolutethresh))
    
    # Start iterations
    scale_counts = numpy.zeros(nscales, dtype='int')
    scale_flux = numpy.zeros(nscales)

    for i in range(niter):
        
        # Find the optimum scale and location.
        mscale, mx, my, gpu_mval = cuda_find_global_optimum(hsmmpsf, gpu_ihsmmpsf, gpu_smresidual, windowstack, findpeak) 
        mval = gpu_mval.get()
        scale_counts[mscale] += 1
        scale_flux[mscale] += mval[0]
        
        # Report on progress
        #raw_mval = smresidual[mscale, :, mx, my]
        if niter < 10 or i % (niter // 10) == 0:
            log.info("msmfsclean: Minor cycle %d, peak %s at [%d, %d, %d]" % (i, mval, mx, my, mscale))
        
        # Are we ready to stop yet?
        peak = numpy.max(numpy.fabs(mval))
        if peak < absolutethresh:
            log.info("msmfsclean: Absolute value of peak %.6f is below stopping threshold %.6f" \
                     % (peak, absolutethresh))
            break
        
        # Calculate indices needed for lhs and rhs of updates to model and residual
        para = cuda_overlapIndices(ldirty[0, ...], psf[0, ...], mx, my)
        
        # Update model and residual image
        
        cuda_update_moment_model_residual(gpu_m_model, gpu_pscalestack, gpu_smresidual, gpu_ssmmpsf, para, gain, mscale, gpu_mval)

    log.info("msmfsclean: End of minor cycles")
    
    log.info("msmfsclean: Scale counts %s" % (scale_counts))
    log.info("msmfsclean: Scale flux %s" % (scale_flux))

    m_model = gpu_m_model.get_async()
    res_smre = cuda_return_res_zero(gpu_smresidual,pmax)
    
    return m_model, res_smre




