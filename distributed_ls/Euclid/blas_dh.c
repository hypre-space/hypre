/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include "blas_dh.h"

#undef __FUNC__
#define __FUNC__ "matvec_euclid_seq"
void matvec_euclid_seq(int n, int *rp, int *cval, double *aval, double *x, double *y)
{
  START_FUNC_DH
  int i, j;
  int from, to, col;
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
void Axpy(int n, double alpha, double *x, double *y)
{
  START_FUNC_DH
  int i;

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
void CopyVec(int n, double *xIN, double *yOUT)
{
  START_FUNC_DH
  int i;

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
void ScaleVec(int n, double alpha, double *x)
{
  START_FUNC_DH
  int i;

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
double InnerProd(int n, double *x, double *y)
{
  START_FUNC_DH
  double result, local_result = 0.0;

  int i;

#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(x, y) \
             private(i) \
             reduction(+:local_result)
#endif
    for (i=0; i<n; ++i) {
      local_result += x[i] * y[i];
    }

    if (np_dh > 1) {
      MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, comm_dh);
    } else {
      result = local_result;
    }

  END_FUNC_VAL(result)
}

#undef __FUNC__
#define __FUNC__ "Norm2"
double Norm2(int n, double *x)
{
  START_FUNC_DH
  double result, local_result = 0.0;
  int i;

#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(x) \
             private(i) \
             reduction(+:local_result)
#endif
  for (i=0; i<n; ++i) {
    local_result += (x[i]*x[i]);
  }

  if (np_dh > 1) {
    MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, comm_dh);
  } else {
    result = local_result;
  }
  result = sqrt(result);
  END_FUNC_VAL(result)
}
