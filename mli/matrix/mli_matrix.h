/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for MLI Matrix data structures
 *
 *****************************************************************************/

#ifndef __MLI_MATRIX__
#define __MLI_MATRIX__

#include "utilities.h"

/*--------------------------------------------------------------------------
 * MLI_Matrix declaration
 *--------------------------------------------------------------------------*/

typedef struct
{
   char  name[100];
   void  *matrix;
   void  (*destroy_func)(void *);
} 
MLI_Matrix;

/*--------------------------------------------------------------------------
 * functions for the MLI_Matrix 
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
{
#endif

MLI_Matrix *MLI_Matrix_Create(void *in_matrix, char *name,
                              void (*mat_destroyer)(void *));

void *MLI_Matrix_Destroy(MLI_Matrix *mli_matrix);

#ifdef __cplusplus
}
#endif

#endif

