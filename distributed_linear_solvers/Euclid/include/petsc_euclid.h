#ifndef PETSC_EUCLID_H
#define PETSC_EUCLID_H

#ifdef PETSC_MODE

#include "euclid_common.h"


      /* constructs a vector with random values in [0,1] */
int buildRandVec(int globalRows, int localRows, int beg_row, Vec *Xout);


void buildPetscMat(int globalRows, int localRows, int beg_row, 
             int* rp, int* cval, double* aval, Mat *Aout);

void buildEuclidFromPetscMat(Mat Ain, Mat_dh *Aout);


  /* For ExtractMat, caller is responsible for allocating memory for n2o_row 
     and n2o_col.
     extractMat allocates memory for rp, cval, and aval; caller
     is responsible for freeing via calls to FREE_DH(ptr).
     It is permissible to pass a null ptrs for n2o_row and n2o_col,
     in which case any switch "-mat_ordering_type" is ignored.
     At present, ordering only works for single-mpi task runs.
   */         
void extractMat(Mat Ain, int *globalRows, int *localRows, int *beg_row,
               int **rp, int **cval, double **aval,
               int *n2o_row, int *n2o_col);

#endif /* #ifdef PETSC_MODE */
#endif
