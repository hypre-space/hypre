#ifndef THREADED_KRYLOV_H
#define THREADED_KRYLOV_H

#include "blas_dh.h"

extern void bicgstab_euclid(Mat_dh A, Euclid_dh ctx, double *x, double *b, 
                                                              int *itsOUT);

extern void cg_euclid(Mat_dh A, Euclid_dh ctx, double *x, double *b, 
                                                              int *itsOUT);

#endif
