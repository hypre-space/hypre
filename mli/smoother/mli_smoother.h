/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SMOOTHER__
#define __MLI_SMOOTHER__

#include "utilities.h"
#include "parcsr_mv.h"

/******************************************************************************
 * declaration of MLI_Smoother class
 *****************************************************************************/

typedef struct MLI_Smoother_Struct
{
   void *object;
   void (*destroy_func)(void *);
   int  (*apply_func)(void *smoo_obj, MLI_Vector *f, MLI_Vector *u);
} MLI_Smoother;

/******************************************************************************
 * constructor and destructor
 *****************************************************************************/

int MLI_Smoother_Create( void **smoo_obj )
{
   MLI_Smoother *smoother = hypre_CTAlloc( MLI_Smoother, 1);
   if ( smoother == NULL ) { (*smoo_obj) = NULL; return 1; }
   smoother->object = NULL;
   smoother->destroy_func = NULL;
   smoother->apply_func = NULL;
   (*smoo_obj) = smoother;
   return 0;
}
 
int MLI_Smoother_Destroy( void *smoo_obj )
{
   MLI_Smoother *smoother = (MLI_Smoother *) smoo_obj;
   if ( smoother->object != NULL && smoother->destroy_func != NULL )
      smoother->destroy_func( smoother->object );
   smoother->apply_func = NULL;
   hypre_TFree( smoother );
   return 0;
}

#endif

