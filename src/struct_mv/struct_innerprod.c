/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Structured inner product routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

#define minimo(a,b) a<b?a:b
#define MAXBLOCKS 32
#define NTHREADS 256 // must be a power of 2

__global__ void dot (HYPRE_Real * a, HYPRE_Real * b, HYPRE_Real *c, HYPRE_Int hypre__tot,
                     HYPRE_Int *loop_size_cuda,HYPRE_Int ndim,
                     HYPRE_Int *stride_cuda1,HYPRE_Int *start_cuda1,HYPRE_Int *dboxmin1,HYPRE_Int *dboxmax1,
                     HYPRE_Int *stride_cuda2,HYPRE_Int *start_cuda2,HYPRE_Int *dboxmin2,HYPRE_Int *dboxmax2)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	HYPRE_Int local_idx;
    HYPRE_Int d,idx_local = id;
    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;
    HYPRE_Int i1 = 0, i2 = 0;
    
    //// reducted output
    __shared__ HYPRE_Real shared_cache [NTHREADS];
	HYPRE_Real sum = 0;
    
    for (d = 0;d < ndim;d ++)
    {
        local_idx  = idx_local % loop_size_cuda[d];
        idx_local  = idx_local / loop_size_cuda[d];
        i1 += (local_idx*stride_cuda1[d] + start_cuda1[d] - dboxmin1[d]) * hypre_boxD1;
        hypre_boxD1 *= hypre_max(0, dboxmax1[d] - dboxmin1[d] + 1);
        i2 += (local_idx*stride_cuda2[d] + start_cuda2[d] - dboxmin2[d]) * hypre_boxD2;
        hypre_boxD2 *= hypre_max(0, dboxmax2[d] - dboxmin2[d] + 1);
    }
    //for (;id < size ;){
    //    sum += (*(a+id)) * (*(b+id));
    //    id+= nextid;
    //}
	if (id < hypre__tot)
		sum = a[i1] * hypre_conj(b[i2]);
    *(shared_cache + threadIdx.x) = sum;
	
    __syncthreads();
	
    ///////// sum of internal cache
	
    int i;    
    
    for (i=(NTHREADS /2); i>0 ; i= i/2){
		if (threadIdx.x < i){
			*(shared_cache + threadIdx.x) += *(shared_cache + threadIdx.x + i);
		}
		__syncthreads();
    }
	
    if ( threadIdx.x == 0){
        *(c+ blockIdx.x) = shared_cache[0];
    }
}

/*--------------------------------------------------------------------------
 * hypre_StructInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_StructInnerProd( hypre_StructVector *x,
                       hypre_StructVector *y )
{
   HYPRE_Real       final_innerprod_result;
   HYPRE_Real       process_result;
                   
   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;
                   
   HYPRE_Int        xi;
   HYPRE_Int        yi;
                   
   HYPRE_Complex   *xp;
   HYPRE_Complex   *yp;
                   
   hypre_BoxArray  *boxes;
   hypre_Box       *box;
    hypre_Index      loop_size;
    hypre_IndexRef   start;
    hypre_Index      unit_stride;
    
   HYPRE_Int         ndim = hypre_StructVectorNDim(x);               
   HYPRE_Int        i, d;

//#ifdef HYPRE_USE_RAJA
//   const size_t block_size = 256;
//   ReduceSum< cuda_reduce<block_size>, HYPRE_Real> local_result(0.0);
   HYPRE_Real       local_result;
   local_result = 0.0;
   //zypre_Reductioninit(local_result);
   process_result = 0.0;
   
   hypre_SetIndex(unit_stride, 1);
   
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
   {
      box   = hypre_BoxArrayBox(boxes, i);
	  start = hypre_BoxIMin(box);
	  
       x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
       y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
       
      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);
       
       hypre_BoxGetSize(box, loop_size);
	   cudaDeviceSynchronize();
       {
           zypre_BoxLoopCUDAInit(ndim,loop_size);
           zypre_newBoxLoopInitK(ndim,x_data_box,loop_size,start,unit_stride,1);
           zypre_newBoxLoopInitK(ndim,y_data_box,loop_size,start,unit_stride,2);
           int n_blocks = (hypre__tot+NTHREADS-1)/NTHREADS;
           HYPRE_Real *d_c;
		   HYPRE_Real * c = new HYPRE_Real[n_blocks];
           cudaMalloc((void**) &d_c, n_blocks * sizeof(HYPRE_Real));
		   //cudaMallocManaged((void**)&d_c,sizeof(HYPRE_Real)*n_blocks, cudaMemAttachGlobal);
           dot<<< n_blocks ,NTHREADS>>>(xp,yp,d_c,hypre__tot,loop_size_cuda,ndim,
                     stride_cuda1,start_cuda1,dboxmin1,dboxmax1,
                     stride_cuda2,start_cuda2,dboxmin2,dboxmax2);
		   cudaMemcpy(c,d_c,n_blocks*sizeof(HYPRE_Real),cudaMemcpyDeviceToHost);
		   cudaDeviceSynchronize();
		   for (int j = 0 ; j< n_blocks ; ++j){
			   local_result += c[j];
		   }
       }
	   //int n_blocks = minimo( MAXBLOCKS, ((n+NTHREADS-1)/NTHREADS));
	   //cudaMalloc((void**) &d_c, n_blocks * sizeof(int));
	   //dot<<< n_blocks ,NTHREADS>>>(d_a,d_b,d_c,n);
	   //// final sum on host
       // int final_result = 0;
       // for (int i=0 ; i< n_blocks ; ++i){
       //     final_result += *(c+i);
       // }
	   //printf("local_result = %f\n",xp[0]);
       
   }
   process_result = local_result;
   
       /*
       hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                           x_data_box, start, unit_stride, xi,
                           y_data_box, start, unit_stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,xi,yi) reduction(+:local_result) HYPRE_SMP_SCHEDULE
#endif
       hypre_BoxLoop2For(xi, yi)
       {
		   printf("local_result = %f\n",local_result);
		   
           local_result += xp[xi] * hypre_conj(yp[yi]);
       }
       hypre_BoxLoop2End(xi, yi);
   }
    process_result = local_result;
	   */
/*
#ifdef HYPRE_BOX_PRIVATE_VAR
#undef HYPRE_BOX_PRIVATE_VAR
#endif
#define HYPRE_BOX_PRIVATE_VAR xi,yi
#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif
#define HYPRE_BOX_REDUCTION reduction(+:local_result)
	   
	  zypre_newBoxLoop2ReductionBegin(ndim, loop_size,
									  x_data_box, start, unit_stride, xi,
									  y_data_box, start, unit_stride, yi,local_result);
      {
         local_result += xp[xi] * hypre_conj(yp[yi]);		 
      }
      zypre_newBoxLoop2ReductionEnd(xi, yi, local_result);
   }
 
   printf("get here\n");//process_result = (double) (local_result);
   cudaDeviceSynchronize();
   process_result = static_cast<double>(local_result);
   printf("Cast\n");
*/
   hypre_MPI_Allreduce(&process_result, &final_innerprod_result, 1,
                       HYPRE_MPI_REAL, hypre_MPI_SUM, hypre_StructVectorComm(x));

   hypre_IncFLOPCount(2*hypre_StructVectorGlobalSize(x));
   
   return final_innerprod_result;
}
