#ifndef THREADED_KRYLOV_H
#define THREADED_KRYLOV_H

#include "blas_dh.h"

void bicgstab_euclid(Mat_dh A, Euclid_dh ctx, double *x, double *b, int *itsOUT);
void cg_euclid(Mat_dh A, Euclid_dh ctx, double *x, double *b, int *itsOUT);

#endif
