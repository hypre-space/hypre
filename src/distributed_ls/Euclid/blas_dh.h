/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef THREADED_BLAS_DH
#define THREADED_BLAS_DH

/* notes: 1. all calls are threaded with OpenMP.
          2. for mpi MatVec, see "Mat_dhMatvec()" in Mat_dh.h
          3. MPI calls use hypre_MPI_COMM_WORLD for the communicator,
             where applicable.
*/

/* #include "euclid_common.h" */

#ifdef SEQUENTIAL_MODE
#define MatVec       matvec_euclid_seq
#endif

extern void matvec_euclid_seq(HYPRE_Int n, HYPRE_Int *rp, HYPRE_Int *cval, HYPRE_Real *aval, HYPRE_Real *x, HYPRE_Real *y);
extern HYPRE_Real InnerProd(HYPRE_Int local_n, HYPRE_Real *x, HYPRE_Real *y);
extern HYPRE_Real Norm2(HYPRE_Int local_n, HYPRE_Real *x);
extern void Axpy(HYPRE_Int n, HYPRE_Real alpha, HYPRE_Real *x, HYPRE_Real *y);
extern HYPRE_Real Norm2(HYPRE_Int n, HYPRE_Real *x);
extern void CopyVec(HYPRE_Int n, HYPRE_Real *xIN, HYPRE_Real *yOUT);
extern void ScaleVec(HYPRE_Int n, HYPRE_Real alpha, HYPRE_Real *x);

#endif
