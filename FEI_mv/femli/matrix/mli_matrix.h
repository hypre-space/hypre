/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

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

#include "utilities/utilities.h"

#include "vector/mli_vector.h"
#include "util/mli_utils.h"

/*--------------------------------------------------------------------------
 * MLI_Matrix data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_Matrix
{
   char   name[100];
   int    g_nrows_, max_nnz_, min_nnz_, tot_nnz_;
   double max_val_, min_val_;
   void   *matrix;
   int    (*destroy_func)(void *);

public :

   MLI_Matrix( void *, char *, MLI_Function *func);
   ~MLI_Matrix();
   void       *getMatrix()                          { return matrix; }
   char       *getName()                            { return name; }
   int        apply( double, MLI_Vector *, double, MLI_Vector *, MLI_Vector * );
   MLI_Vector *createVector();
   int        getMatrixInfo(char *, int &, double &);
   int        print(char *);
};

extern int MLI_Matrix_ComputePtAP(MLI_Matrix *P,MLI_Matrix *A,MLI_Matrix **RAP);
extern int MLI_Matrix_FormJacobi(MLI_Matrix *A, double alpha, MLI_Matrix **J);
extern int MLI_Matrix_Compress(MLI_Matrix *A, int blksize, MLI_Matrix **A2);

#endif

