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

  #pragma omp parallel private(j, col, sum, from, to) \
                default(shared) \
                firstprivate(n, rp, cval, aval, x, y) 
  {
    #pragma omp for schedule(static)       
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

  #pragma omp parallel for schedule(static) firstprivate(alpha, x, y) \
             private(i) 
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

  #pragma omp parallel for schedule(static) firstprivate(yOUT, xIN) \
             private(i)
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

  #pragma omp parallel for schedule(static) firstprivate(alpha, x) \
             private(i)
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

  #pragma omp parallel for schedule(static) firstprivate(x, y) \
             private(i) \
             reduction(+:local_result)
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

  #pragma omp parallel for schedule(static) firstprivate(x) \
             private(i) \
             reduction(+:local_result)
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
