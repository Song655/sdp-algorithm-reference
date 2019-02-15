# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
# Initialize the CUDA device
import pycuda.autoinit
 
import numpy
import logging
import math
import scipy.special

log = logging.getLogger(__name__)

cuda_grid_kernel_source = """
#include <cuComplex.h>
#define Data_Type float

__device__ inline void atAddComplex(cuComplex* a, cuComplex b){
  atomicAdd(&(a->x), b.x);
  atomicAdd(&(a->y), b.y);
}

__device__ inline cuComplex ComplexScale(cuComplex a, float s)
{
    cuComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

__device__ inline cuComplex ComplexMul(cuComplex a, cuComplex b)
{
    cuComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

__device__ inline int around(float a)
{
   if(a!=-0.5 && a!=0.5){
       return round(a);
   }
   else return 0;
}

__global__ void griding_kernel(cuComplex *uvgrid,
                        Data_Type *wts, 
                        cuComplex *vis, 
                        cuComplex * kernels0,
                        Data_Type *vuvwmap0,
                        Data_Type *vuvwmap1,
                        int *vfrequencymap, 
                        Data_Type *sumwt,
                        unsigned int kernel_oversampling,
                        unsigned int nx, 
                        unsigned int ny,
                        unsigned int gh,
                        unsigned int gw,
                        unsigned int npol,
						unsigned int size) 
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size){
		Data_Type x = nx/2 + vuvwmap0[idx]*nx;
		Data_Type flx = floor(x + 0.5/kernel_oversampling);
		int xxf = around((x-flx)*kernel_oversampling);
		int xx = flx - gw/2;

		Data_Type y = ny/2 + vuvwmap1[idx]*ny;
		Data_Type fly = floor(y + 0.5/kernel_oversampling);
		int yyf = around((y-fly)*kernel_oversampling);
		int yy = fly - gh/2;

		int ichan = vfrequencymap[idx];

		int i = 0;
		int j = 0;
		int k = 0;

		for (i = 0; i< npol; i++){
		    Data_Type vwt = wts[idx*npol + i];
			cuComplex v = ComplexScale(vis[idx*npol+i],vwt);
			for (j = 0; j < gh; j++){
				for(k =0; k < gw; k++){
					int id1 = ichan*npol*ny*nx + i*ny*nx + (yy+j)*nx + xx+k;
					int id2 = yyf*kernel_oversampling*gh*gw + xxf*gh*gw + j*gw + k;         
					atAddComplex(&uvgrid[id1],ComplexMul(kernels0[id2],v));
				}
			}
			atomicAdd(&sumwt[ichan * npol + i],vwt);
		}
	}
}
"""

cuda_grid_kernel = nvcc.SourceModule(cuda_grid_kernel_source,options=['-O3'])
cuda_gridding_core = cuda_grid_kernel.get_function("griding_kernel") 
Data_Type = numpy.float32
THREAD_NUM = 512

def cuda_convolutional_grid(kernel_list, uvgrid, vis, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None):
    print("---cuda_convolutional_grid---")
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape

    asize = len(vfrequencymap)  ## size = Baseline*Channel*Time (166*165/2 * 8 * 10)
    assert numpy.array(vuvwmap[:,0-1] >= -0.5).all() and numpy.array(vuvwmap[:,0-1] < 0.5).all(), "Cellsize is too large: uv overflows grid uv"
    
    thread1 = THREAD_NUM
    grid1 = math.ceil(1.0*asize/thread1)

    gpu_sumwt = gpuarray.zeros((inchan, inpol),dtype = Data_Type)
    #for pol in range(inpol):
        
    gpu_uvgrid = gpuarray.to_gpu(uvgrid.astype(numpy.complex64))
    gpu_wts = gpuarray.to_gpu(visweights.astype(numpy.float32))
    gpu_vis = gpuarray.to_gpu(vis.astype(numpy.complex64))
    gpu_kernels0 = gpuarray.to_gpu(kernels[0].astype(numpy.complex64))
    gpu_vuvwmap0 = gpuarray.to_gpu(vuvwmap[:, 0].astype(numpy.float32))
    gpu_vuvwmap1 = gpuarray.to_gpu(vuvwmap[:, 1].astype(numpy.float32))
    gpu_vfrequencymap = gpuarray.to_gpu(numpy.asarray(vfrequencymap).astype(numpy.int32))
	
    cuda_gridding_core(gpu_uvgrid, gpu_wts,gpu_vis, gpu_kernels0, gpu_vuvwmap0,gpu_vuvwmap1,gpu_vfrequencymap,gpu_sumwt,numpy.uint32(kernel_oversampling),numpy.uint32(nx),numpy.uint32(ny),numpy.uint32(gh),numpy.uint32(gw),numpy.uint32(inpol),numpy.uint32(asize),block=(thread1,1,1),grid=(grid1,1))

    uvgrid = gpu_uvgrid.get().astype(numpy.complex128)
    return uvgrid, gpu_sumwt.get().astype(numpy.float64)

	
	
	
	
	