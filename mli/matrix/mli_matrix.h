/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Parallel Matrix data structures
 *
 *****************************************************************************/

#ifndef __MLI_MATRIX__
#define __MLI_MATRIX__

#include "utilities.h"

/*--------------------------------------------------------------------------
 * MLI_Matrix 
 *--------------------------------------------------------------------------*/

typedef struct
{
   char  name[100];
   void  *matrix;
   void  (*destroy_func)(void *);
} 
MLI_Matrix;

/*--------------------------------------------------------------------------
 * constructor and destructor functions for the MLI_Matrix 
 *--------------------------------------------------------------------------*/

MLI_Matrix *MLI_Matrix_Create(void *in_matrix, char *name,
                              void (*mat_destroyer)(void *))
{
   MLI_Matrix *mli_matrix = (MLI_Matrix *) calloc(MLI_Matrix, 1);
   mli_matrix->matrix = in_matrix;
   mli_matrix->destroy_func = mat_destroyer;
   strncpy(mli_matrix->name, name, 100);
   return mli_matrix;
}

void *MLI_Matrix_Destroy(MLI_Matrix *mli_matrix)
{
   if ( mli_matrix != NULL )
   {
      if ( mli_matrix->matrix != NULL && mli_matrix->destroy_func != NULL )
               mli_matrix->destroy_func(mli_matrix->matrix);
            mli_matrix->matrix = NULL;
            mli_matrix->destroy_func = NULL;
            free( mli_matrix );
   }
}

#endif

