#from __future__ import absolute_import, print_function
#import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

import numpy
import logging
import math
import os
import logging

#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

platforms = cl.get_platforms()
ctx = cl.Context(dev_type=cl.device_type.GPU)
#ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])


queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
DATA_TPYE = 'FLOAT'

if DATA_TPYE == 'FLOAT': 
    prg = cl.Program(ctx, """
    # include <pyopencl-complex.h>
 
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

	inline void atomic_add_float(__global float *ptr, const float value)
	{
	    __global int *addr = (__global int *) ptr;
	    int old, new;
	    do {
	        old = addr[0];
	        new = as_int(value + as_float(old));
	    } while (atom_cmpxchg(addr, old, new) != old);
	}

	inline void atAddComplex(__global cfloat_t *ptr,const cfloat_t value)
	{
	    atomic_add_float(&(ptr->real),value.real);
	    atomic_add_float(&(ptr->imag),value.imag); 
	    
	}

	inline int around( float a)
	{

	   if(a!= -0.5f && a!= 0.5f){
	       return round(a);
	   }
	   else return 0;
	}

	__kernel void griding_kernel(
	                        __global cfloat_t *uvgrid,
	                        __global const float *wts, 
	                        __global const cfloat_t *vis, 
	                        __global const cfloat_t * kernels0,
	                        __global const float *vuvwmap0,
	                        __global const float *vuvwmap1,
	                        __global const int *vfrequencymap, 
	                        __global float *sumwt,
	                        const unsigned int kernel_oversampling,
	                        const unsigned int nx, 
	                        const unsigned int ny,
	                        const unsigned int gh,
	                        const unsigned int gw,
	                        const unsigned int npol,
				    const unsigned int size) 
	{
	       unsigned idx = get_global_id(0);

		if (idx < size){
			float x = nx/2 + vuvwmap0[idx]*nx;
			float flx = floor(x + 0.5f/kernel_oversampling);
			int xxf = around((x-flx)*kernel_oversampling);
			int xx =flx - gw/2;

			float y = ny/2 + vuvwmap1[idx]*ny;
			float fly = floor(y + 0.5f/kernel_oversampling);
			int yyf = around((y-fly)*kernel_oversampling);
			int yy =fly - gh/2;

			int ichan = vfrequencymap[idx];

			int i = 0;
			int j = 0;
			int k = 0;

			for (i = 0; i< npol; i++){
			       float vwt = wts[idx*npol + i];
				cfloat_t v = cfloat_mulr(vis[idx*npol+i], vwt);
				for (j = 0; j < gh; j++){
					for(k =0; k < gw; k++){
						int id1 = ichan*npol*ny*nx + i*ny*nx + (yy+j)*nx + xx+k;
						int id2 = yyf*kernel_oversampling*gh*gw + xxf*gh*gw + j*gw + k;         
						atAddComplex(&uvgrid[id1], cfloat_mul(kernels0[id2],v));
					}
				}
				atomic_add_float(&sumwt[ichan * npol + i], vwt);
			}
		}
	}

	"""
	).build(options=['-O3'])

    def ocl_convolutional_grid(kernel_list, uvgrid, vis, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None):
        kernel_indices, kernels = kernel_list
        kernel_oversampling, _, gh, gw = kernels[0].shape
        assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
        assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
        inchan, inpol, ny, nx = uvgrid.shape

        print("ocl_convolutional_grid")

        map_arr = numpy.asarray(vfrequencymap).astype(numpy.int32)
        asize = len(vfrequencymap)  ## size = Baseline*Channel*Time (166*165/2 * 8 * 10)
        assert numpy.array(vuvwmap[:,0-1] >= -0.5).all() and numpy.array(vuvwmap[:,0-1] < 0.5).all(), "Cellsize is too large: uv overflows grid uv"

        """
        gpu_sumwt = cl_array.zeros(queue,(inchan,inpol),dtype = numpy.float64)
        gpu_uvgrid = cl_array.to_device(queue, uvgrid.astype(numpy.complex128))
        gpu_wts = cl_array.to_device(queue, visweights.astype(numpy.float64))
        gpu_vis = cl_array.to_device(queue, vis.astype(numpy.complex128))
        gpu_kernels0 = cl_array.to_device(queue, kernels[0].astype(numpy.complex128))
        gpu_vuvwmap0 = cl_array.to_device(queue, vuvwmap[:, 0].astype(numpy.float64))
        gpu_vuvwmap1 = cl_array.to_device(queue, vuvwmap[:, 1].astype(numpy.float64))
        gpu_vfrequencymap = cl_array.to_device(queue, map_arr)
        queue.finish()

        gridding = prg.griding_kernel  
        gridding.set_args(gpu_uvgrid.data, gpu_wts.data,gpu_vis.data, gpu_kernels0.data, gpu_vuvwmap0.data,gpu_vuvwmap1.data,gpu_vfrequencymap.data,gpu_sumwt.data,numpy.uint32(kernel_oversampling),numpy.uint32(nx),numpy.uint32(ny),numpy.uint32(gh),numpy.uint32(gw),numpy.uint32(inpol),numpy.uint32(asize))
                     
        ev = cl.enqueue_nd_range_kernel(queue, gridding, map_arr.shape,None)
        queue.finish() 
	 """

        sumwt = numpy.zeros((inchan,inpol),dtype = numpy.float32)
        gpu_sumwt = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sumwt)
    
        uvgrid_64 = uvgrid.astype(numpy.complex64) 
        gpu_uvgrid = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=uvgrid_64)

        gpu_wts = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=visweights.astype(numpy.float32))
        gpu_vis = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vis.astype(numpy.complex64))
        gpu_kernels0 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernels[0].astype(numpy.complex64))
        gpu_vuvwmap0 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vuvwmap[:, 0].astype(numpy.float32))
        gpu_vuvwmap1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vuvwmap[:, 1].astype(numpy.float32))
        gpu_vfrequencymap = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=map_arr)

        gridding = prg.griding_kernel
        gridding.set_args(gpu_uvgrid, gpu_wts,gpu_vis, gpu_kernels0, gpu_vuvwmap0,gpu_vuvwmap1,gpu_vfrequencymap,gpu_sumwt,numpy.uint32(kernel_oversampling),numpy.uint32(nx),numpy.uint32(ny),numpy.uint32(gh),numpy.uint32(gw),numpy.uint32(inpol),numpy.uint32(asize))
    
        ev = cl.enqueue_nd_range_kernel(queue, gridding, map_arr.shape,None)
        ev.wait()
   
        cl.enqueue_copy(queue, uvgrid_64, gpu_uvgrid)
        cl.enqueue_copy(queue, sumwt, gpu_sumwt) 
        queue.finish() 
   
        uvgrid = uvgrid_64.astype(numpy.complex128)
        return uvgrid, sumwt.astype(numpy.float64)


if DATA_TPYE == 'DOUBLE': 
    prg = cl.Program(ctx, """
    #define PYOPENCL_DEFINE_CDOUBLE
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
    # include <pyopencl-complex.h>

    inline void atomic_add_double(__global double *ptr, const double value)
    {
        __global ulong *addr = (__global ulong *) ptr;
        ulong old, new;
        do {  
            old = addr[0];
            new = as_ulong(value + as_double(old));
        } while (atom_cmpxchg(addr, old, new) != old);
    }

    inline void atAddComplex(__global cdouble_t *ptr,const cdouble_t value)
    {
        atomic_add_double(&(ptr->real),value.real);
        atomic_add_double(&(ptr->imag),value.imag); 
        
    }

    inline int around( double a)
    {

       if(a!= -0.5 && a!= 0.5){
           return round(a);
       }
       else return 0;
    }

    __kernel void griding_kernel(
                            __global cdouble_t *uvgrid,
                            __global const double *wts, 
                            __global const cdouble_t *vis, 
                            __global const cdouble_t * kernels0,
                            __global const double *vuvwmap0,
                            __global const double *vuvwmap1,
                            __global const int *vfrequencymap, 
                            __global double *sumwt,
                            const unsigned int kernel_oversampling,
                            const unsigned int nx, 
                            const unsigned int ny,
                            const unsigned int gh,
                            const unsigned int gw,
                            const unsigned int npol,
    			        const unsigned int size) 
    {
       unsigned idx = get_global_id(0);

    	if (idx < size){
    		double x = nx/2 + vuvwmap0[idx]*nx;
    		double flx = floor(x + 0.5f/kernel_oversampling);
    		int xxf = around((x-flx)*kernel_oversampling);
    		int xx =flx - gw/2;

    		double y = ny/2 + vuvwmap1[idx]*ny;
    		double fly = floor(y + 0.5f/kernel_oversampling);
    		int yyf = around((y-fly)*kernel_oversampling);
    		int yy =fly - gh/2;

    		int ichan = vfrequencymap[idx];

    		int i = 0;
    		int j = 0;
    		int k = 0;

    		for (i = 0; i< npol; i++){
    		       double vwt = wts[idx*npol + i];
    			cdouble_t v = cdouble_mulr(vis[idx*npol+i], vwt);
    			for (j = 0; j < gh; j++){
    				for(k =0; k < gw; k++){
    					int id1 = ichan*npol*ny*nx + i*ny*nx + (yy+j)*nx + xx+k;
    					int id2 = yyf*kernel_oversampling*gh*gw + xxf*gh*gw + j*gw + k;         
    					atAddComplex(&uvgrid[id1], cdouble_mul(kernels0[id2],v));
    				}
    			}
    			atomic_add_double(&sumwt[ichan * npol + i], vwt);
    		}
    	}
    }

    """
    ).build(options=['-O3'])

    def ocl_convolutional_grid(kernel_list, uvgrid, vis, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None):
        kernel_indices, kernels = kernel_list
        kernel_oversampling, _, gh, gw = kernels[0].shape
        assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
        assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
        inchan, inpol, ny, nx = uvgrid.shape

        map_arr = numpy.asarray(vfrequencymap).astype(numpy.int32)
        asize = len(vfrequencymap)  ## size = Baseline*Channel*Time (166*165/2 * 8 * 10)
        assert numpy.array(vuvwmap[:,0-1] >= -0.5).all() and numpy.array(vuvwmap[:,0-1] < 0.5).all(), "Cellsize is too large: uv overflows grid uv"

        """
        gpu_sumwt = cl_array.zeros(queue,(inchan,inpol),dtype = numpy.float64)
        gpu_uvgrid = cl_array.to_device(queue, uvgrid.astype(numpy.complex128))
        gpu_wts = cl_array.to_device(queue, visweights.astype(numpy.float64))
        gpu_vis = cl_array.to_device(queue, vis.astype(numpy.complex128))
        gpu_kernels0 = cl_array.to_device(queue, kernels[0].astype(numpy.complex128))
        gpu_vuvwmap0 = cl_array.to_device(queue, vuvwmap[:, 0].astype(numpy.float64))
        gpu_vuvwmap1 = cl_array.to_device(queue, vuvwmap[:, 1].astype(numpy.float64))
        gpu_vfrequencymap = cl_array.to_device(queue, map_arr)
        queue.finish()

        gridding = prg.griding_kernel  
        gridding.set_args(gpu_uvgrid.data, gpu_wts.data,gpu_vis.data, gpu_kernels0.data, gpu_vuvwmap0.data,gpu_vuvwmap1.data,gpu_vfrequencymap.data,gpu_sumwt.data,numpy.uint32(kernel_oversampling),numpy.uint32(nx),numpy.uint32(ny),numpy.uint32(gh),numpy.uint32(gw),numpy.uint32(inpol),numpy.uint32(asize))
                     
        ev = cl.enqueue_nd_range_kernel(queue, gridding, map_arr.shape,None)
        queue.finish() 
	 """

        sumwt = numpy.zeros((inchan,inpol),dtype = numpy.float64)
        gpu_sumwt = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sumwt)
    
        gpu_uvgrid = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=uvgrid)

        gpu_wts = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=visweights.astype(numpy.float64))
        gpu_vis = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vis.astype(numpy.complex128))
        gpu_kernels0 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernels[0].astype(numpy.complex128))
        gpu_vuvwmap0 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vuvwmap[:, 0].astype(numpy.float64))
        gpu_vuvwmap1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vuvwmap[:, 1].astype(numpy.float64))
        gpu_vfrequencymap = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=map_arr)

        gridding = prg.griding_kernel
        gridding.set_args(gpu_uvgrid, gpu_wts,gpu_vis, gpu_kernels0, gpu_vuvwmap0,gpu_vuvwmap1,gpu_vfrequencymap,gpu_sumwt,numpy.uint32(kernel_oversampling),numpy.uint32(nx),numpy.uint32(ny),numpy.uint32(gh),numpy.uint32(gw),numpy.uint32(inpol),numpy.uint32(asize))
    
        ev = cl.enqueue_nd_range_kernel(queue, gridding, map_arr.shape,None)
        ev.wait()
   
        cl.enqueue_copy(queue, uvgrid, gpu_uvgrid)
        cl.enqueue_copy(queue, sumwt, gpu_sumwt) 
        queue.finish() 
        """
        for iy in range(ny):
            for ix in range(nx):
                print(uvgrid[0,0,iy,ix])
        """
        
        return uvgrid, sumwt

