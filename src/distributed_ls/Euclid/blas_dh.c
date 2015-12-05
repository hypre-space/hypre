/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




#include "blas_dh.h"

#undef __FUNC__
#define __FUNC__ "matvec_euclid_seq"
void matvec_euclid_seq(HYPRE_Int n, HYPRE_Int *rp, HYPRE_Int *cval, double *aval, double *x, double *y)
{
  START_FUNC_DH
  HYPRE_Int i, j;
  HYPRE_Int from, to, col;
  double sum;
 
  if (np_dh > 1) SET_V_ERROR("only for sequential case!\n");

#ifdef USING_OPENMP_DH
#pragma omp parallel private(j, col, sum, from, to) \
                default(shared) \
                firstprivate(n, rp, cval, aval, x, y) 
#endif
  {
#ifdef USING_OPENMP_DH
#pragma omp for schedule(static)       
#endif
      for (i=0; i<n; ++i) {
        sum = 0.0;
        from = rp[i]; 
        to = rp[i+1];
        for (j=from; j<to; ++j) {
          col = cval[j];
          sum += (aval[j]*x[col]);
        }
        y[i] = sum;
      }
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Axpy"
void Axpy(HYPRE_Int n, double alpha, double *x, double *y)
{
  START_FUNC_DH
  HYPRE_Int i;

#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(alpha, x, y) \
             private(i) 
#endif
  for (i=0; i<n; ++i) {
    y[i] = alpha*x[i] + y[i];
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "CopyVec"
void CopyVec(HYPRE_Int n, double *xIN, double *yOUT)
{
  START_FUNC_DH
  HYPRE_Int i;

#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(yOUT, xIN) \
             private(i)
#endif
  for (i=0; i<n; ++i) {
    yOUT[i] = xIN[i];
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "ScaleVec"
void ScaleVec(HYPRE_Int n, double alpha, double *x)
{
  START_FUNC_DH
  HYPRE_Int i;

#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(alpha, x) \
             private(i)
#endif
  for (i=0; i<n; ++i) {
    x[i] *= alpha;
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "InnerProd"
double InnerProd(HYPRE_Int n, double *x, double *y)
{
  START_FUNC_DH
  double result, local_result = 0.0;

  HYPRE_Int i;

#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(x, y) \
             private(i) \
             reduction(+:local_result)
#endif
    for (i=0; i<n; ++i) {
      local_result += x[i] * y[i];
    }

    if (np_dh > 1) {
      hypre_MPI_Allreduce(&local_result, &result, 1, hypre_MPI_DOUBLE, hypre_MPI_SUM, comm_dh);
    } else {
      result = local_result;
    }

  END_FUNC_VAL(result)
}

#undef __FUNC__
#define __FUNC__ "Norm2"
double Norm2(HYPRE_Int n, double *x)
{
  START_FUNC_DH
  double result, local_result = 0.0;
  HYPRE_Int i;

#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(x) \
             private(i) \
             reduction(+:local_result)
#endif
  for (i=0; i<n; ++i) {
    local_result += (x[i]*x[i]);
  }

  if (np_dh > 1) {
    hypre_MPI_Allreduce(&local_result, &result, 1, hypre_MPI_DOUBLE, hypre_MPI_SUM, comm_dh);
  } else {
    result = local_result;
  }
  result = sqrt(result);
  END_FUNC_VAL(result)
}
