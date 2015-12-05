#ifndef THREADED_BLAS_DH
#define THREADED_BLAS_DH

/* notes: 1. all calls are threaded with OpenMP.
          2. for mpi MatVec, see "Mat_dhMatvec()" in Mat_dh.h
          3. MPI calls use MPI_COMM_WORLD for the communicator,
             where applicable.
*/

#include "euclid_common.h"

#ifdef SEQUENTIAL_MODE
#define MatVec       matvec_euclid_seq
#endif

extern void matvec_euclid_seq(int n, int *rp, int *cval, double *aval, double *x, double *y);
extern double InnerProd(int local_n, double *x, double *y);
extern double Norm2(int local_n, double *x);
extern void Axpy(int n, double alpha, double *x, double *y);
extern double Norm2(int n, double *x);
extern void CopyVec(int n, double *xIN, double *yOUT);
extern void ScaleVec(int n, double alpha, double *x);

#endif
