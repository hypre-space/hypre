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

#include "_hypre_utilities.h"
#include "../seq_mv/seq_mv.h"

#if defined(HYPRE_USING_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

/******************************************************************************
 *
 * hypre initialization
 *
 *****************************************************************************/

void
HYPRE_Init( hypre_int argc, char *argv[] )
{
   /*
   HYPRE_Int  num_procs, myid;

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   */
#if defined(HYPRE_USING_KOKKOS)
   /*
   Kokkos::InitArguments args;
   args.num_threads = 10;
   Kokkos::initialize (args);
   */
   Kokkos::initialize (argc, argv);
#endif

#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && defined(HYPRE_USING_CUDA)
   /*
   if (!cuda_reduce_buffer)
   {
      cuda_reduce_buffer = hypre_TAlloc(HYPRE_double6, 1024, HYPRE_MEMORY_DEVICE);
   }
   */
#endif

#if defined(HYPRE_USING_UNIFIED_MEMORY)
   hypre_GPUInit(-1);
#endif

   /* hypre_InitMemoryDebug(myid); */

#if defined(HYPRE_USING_DEVICE_OPENMP)
   /*
   hypre__offload_device_num = omp_get_initial_device();
   hypre__offload_host_num   = omp_get_initial_device();
   */
   HYPRE_OMPOffloadOn();
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_device_csr_handle = hypre_DeviceCSRHandleCreate();
#endif
}

/******************************************************************************
 *
 * hypre finalization
 *
 *****************************************************************************/

/* declared in "struct_communication.c" */
extern HYPRE_Complex *global_recv_buffer, *global_send_buffer;
extern HYPRE_Int      global_recv_size, global_send_size;

void
HYPRE_Finalize()
{
#if defined(HYPRE_USING_UNIFIED_MEMORY)
   hypre_GPUFinalize();
#endif

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif

#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && defined(HYPRE_USING_CUDA)
   hypre_TFree(cuda_reduce_buffer, HYPRE_MEMORY_DEVICE);
#endif

   hypre_TFree(global_send_buffer, HYPRE_MEMORY_DEVICE);
   hypre_TFree(global_recv_buffer, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CUDA)
   hypre_DeviceCSRHandleDestroy(hypre_device_csr_handle);
#endif
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
/* Initialize GPU branch of Hypre AMG */
/* use_device =-1 */
/* Application passes device number it is using or -1 to let Hypre decide on which device to use */
void hypre_GPUInit(hypre_int use_device)
{
   //char pciBusId[80];
   HYPRE_Int myid, nproc;
   hypre_int nDevices;

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &nproc);

   if (!HYPRE_GPU_HANDLE)
   {
      HYPRE_GPU_HANDLE = 1;
      HYPRE_DEVICE = 0;
      HYPRE_CUDA_CALL(cudaGetDeviceCount(&nDevices));
      HYPRE_DEVICE_COUNT = nDevices;

      hypre_MPI_Comm node_comm;
      hypre_MPI_Comm_split_type(hypre_MPI_COMM_WORLD, hypre_MPI_COMM_TYPE_SHARED, myid, MPI_INFO_NULL, &node_comm);
      HYPRE_Int myNodeid, NodeSize;
      hypre_MPI_Comm_rank(node_comm, &myNodeid);
      hypre_MPI_Comm_size(node_comm, &NodeSize);
      hypre_MPI_Comm_free(&node_comm);

      if (use_device < 0)
      {
         HYPRE_DEVICE = myNodeid % nDevices;
      }
      else
      {
         HYPRE_DEVICE = use_device;
      }

      HYPRE_CUDA_CALL(cudaSetDevice(HYPRE_DEVICE));

      hypre_int device_id;
      HYPRE_CUDA_CALL(cudaGetDevice(&device_id));
      printf("Proc [global %d/%d, local %d/%d] can see %d GPUs and is running on %d\n",
              myid, nproc, myNodeid, NodeSize, nDevices, device_id);

#if defined(HYPRE_USING_OPENMP_OFFLOAD) || defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
      omp_set_default_device(HYPRE_DEVICE);
      //printf("Set OMP Default device to %d \n",HYPRE_DEVICE);
#endif

      /* Create NVTX domain for all the nvtx calls in HYPRE */
      HYPRE_DOMAIN=nvtxDomainCreateA("Hypre");

      /* Initialize streams */
      hypre_int jj;
      for(jj=0;jj<MAX_HGS_ELEMENTS;jj++)
      {
         //HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&(HYPRE_STREAM(jj)),cudaStreamNonBlocking));
         HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&(HYPRE_STREAM(jj)), cudaStreamDefault));
      }

      /* Initialize the library handles and streams */
      HYPRE_CUSPARSE_CALL(cusparseCreate(&(HYPRE_CUSPARSE_HANDLE)));
      HYPRE_CUSPARSE_CALL(cusparseSetStream(HYPRE_CUSPARSE_HANDLE, HYPRE_STREAM(4)));
      //HYPRE_CUSPARSE_CALL(cusparseSetStream(HYPRE_CUSPARSE_HANDLE,0)); // Cusparse MxV happens in default stream
      HYPRE_CUSPARSE_CALL(cusparseCreateMatDescr(&(HYPRE_CUSPARSE_MAT_DESCR)));
      HYPRE_CUSPARSE_CALL(cusparseSetMatType(HYPRE_CUSPARSE_MAT_DESCR,CUSPARSE_MATRIX_TYPE_GENERAL));
      HYPRE_CUSPARSE_CALL(cusparseSetMatIndexBase(HYPRE_CUSPARSE_MAT_DESCR,CUSPARSE_INDEX_BASE_ZERO));

      //if (!checkDeviceProps()) hypre_printf("WARNING:: Concurrent memory access not allowed\n");

      /* Check if the arch flags used for compiling the cuda kernels match the device */
#if defined(HYPRE_USING_GPU)
      hypre_CudaCompileFlagCheck();
#endif
   }
}


void hypre_GPUFinalize()
{
   HYPRE_CUSPARSE_CALL(cusparseDestroy(HYPRE_CUSPARSE_HANDLE));

   /* Destroy streams */
   hypre_int jj;
   for(jj=0;jj<MAX_HGS_ELEMENTS;jj++)
   {
      HYPRE_CUDA_CALL(cudaStreamDestroy(HYPRE_STREAM(jj)));
   }

   cudaError_t cudaerr = cudaGetLastError() ;
   if (cudaerr != cudaSuccess)
   {
      hypre_printf("CUDA error: %s\n",cudaGetErrorString(cudaerr));
   }

}
#endif

