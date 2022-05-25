/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the MLI_Matrix data structure
 *
 *****************************************************************************/

#ifndef __MLIMATRIXH__
#define __MLIMATRIXH__

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include "_hypre_utilities.h"

#include "mli_vector.h"
#include "mli_utils.h"

/*--------------------------------------------------------------------------
 * MLI_Matrix data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_Matrix
{
   char   name_[100];
   int    gNRows_, maxNNZ_, minNNZ_, totNNZ_;
   double maxVal_, minVal_, dtotNNZ_;
   void   *matrix_;
   int    (*destroyFunc_)(void *);
   int    subMatrixLength_;
   int    *subMatrixEqnList_;

public :

   MLI_Matrix( void *, char *, MLI_Function *func);
   ~MLI_Matrix();
   void       setSubMatrixEqnList(int leng, int *list);
   void       *getMatrix();
   void	      *takeMatrix();
   char       *getName();
   int        apply(double, MLI_Vector *, double, MLI_Vector *, MLI_Vector *);
   MLI_Vector *createVector();
   int        getMatrixInfo(char *, int &, double &);
   int        print(char *);
};

extern int MLI_Matrix_ComputePtAP(MLI_Matrix *P,MLI_Matrix *A,MLI_Matrix **);
extern int MLI_Matrix_FormJacobi(MLI_Matrix *A, double alpha, MLI_Matrix **J);
extern int MLI_Matrix_Compress(MLI_Matrix *A, int blksize, MLI_Matrix **A2);
extern int MLI_Matrix_GetSubMatrix(MLI_Matrix *A, int nRows, int *rowIndices,
                      int *newNRows, double **newAA);
extern int MLI_Matrix_GetOverlappedMatrix(MLI_Matrix *, int *offNRows, 
                      int **offRowLengs, int **offCols, double **offVals);
#endif

