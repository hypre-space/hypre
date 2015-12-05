/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#ifndef THREADED_BLAS_DH
#define THREADED_BLAS_DH

/* notes: 1. all calls are threaded with OpenMP.
          2. for mpi MatVec, see "Mat_dhMatvec()" in Mat_dh.h
          3. MPI calls use hypre_MPI_COMM_WORLD for the communicator,
             where applicable.
*/

#include "euclid_common.h"

#ifdef SEQUENTIAL_MODE
#define MatVec       matvec_euclid_seq
#endif

extern void matvec_euclid_seq(HYPRE_Int n, HYPRE_Int *rp, HYPRE_Int *cval, double *aval, double *x, double *y);
extern double InnerProd(HYPRE_Int local_n, double *x, double *y);
extern double Norm2(HYPRE_Int local_n, double *x);
extern void Axpy(HYPRE_Int n, double alpha, double *x, double *y);
extern double Norm2(HYPRE_Int n, double *x);
extern void CopyVec(HYPRE_Int n, double *xIN, double *yOUT);
extern void ScaleVec(HYPRE_Int n, double alpha, double *x);

#endif
