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

/******************************************************************************
 *
 * hypre initialization
 *
 *****************************************************************************/

void
hypre_init()
{
   /*
   HYPRE_Int  num_procs, myid;
   
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   */

#if defined(HYPRE_USE_KOKKOS)
   Kokkos::InitArguments args;
   args.num_threads = 10;
   Kokkos::initialize (args);
#endif

#if defined(HYPRE_USE_CUDA)
   initCudaReductionMemBlock();
#endif

#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
   hypre_GPUInit(-1);
#endif

   /* hypre_InitMemoryDebug(myid); */

#ifdef HYPRE_USE_OMP45
   HYPRE_OMPOffloadOn();
#endif
}

/******************************************************************************
 *
 * hypre finalization
 *
 *****************************************************************************/

void
hypre_finalize()
{
#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
   hypre_GPUFinalize();
#endif

#if defined(HYPRE_USE_KOKKOS)
   Kokkos::finalize ();
#endif

#if defined(HYPRE_USE_CUDA)
   freeCudaReductionMemBlock();
#endif
}

