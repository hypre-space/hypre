/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Parallel Vector data structures
 *
 *****************************************************************************/

#ifndef __MLI_VECTOR__
#define __MLI_VECTOR__

#include "utilities.h"

/*--------------------------------------------------------------------------
 * MLI_Vector 
 *--------------------------------------------------------------------------*/

typedef struct
{
   char  name[100];
   void  *vector;
   void  (*destroy_func)(void*);
} 
MLI_Vector;

/*--------------------------------------------------------------------------
 * constructor and destructor functions for the MLI_Vector 
 *--------------------------------------------------------------------------*/

MLI_Vector *MLI_Vector_Create(void *in_vec, char *name,
                              void (*vec_destroyer)(void*))
{
   MLI_Vector *mli_vector = hypre_CTAlloc(MLI_Vector, 1);
   mli_vector->vector = in_vec;
   mli_vector->destroy_func = vec_destroyer;
   strncpy(mli_vector->name, name, 100);
   return mli_vector;
}

void *MLI_Vector_Destroy(MLI_Vector *mli_vector)
{
   if ( mli_vector != NULL )
   {
      if ( mli_vector->vector != NULL && mli_vector->destroy_func != NULL )
         mli_vector->destroy_func(mli_vector->vector);
      mli_vector->vector = NULL;
      mli_vector->destroy_func = NULL;
      hypre_TFree( mli_vector );
   }
}

#endif

