/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include "mli_matrix.h"

/*--------------------------------------------------------------------------
 * constructor functions for the MLI_Matrix 
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

/*--------------------------------------------------------------------------
 * destructor functions for the MLI_Matrix 
 *--------------------------------------------------------------------------*/

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

