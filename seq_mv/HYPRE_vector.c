/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_Vector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_CreateVector
 *--------------------------------------------------------------------------*/

HYPRE_Vector
HYPRE_CreateVector( int size )
{
   return ( (HYPRE_Vector) hypre_CreateVector(size) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DestroyVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_DestroyVector( HYPRE_Vector vector )
{
   return ( hypre_DestroyVector( (hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_InitializeVector( HYPRE_Vector vector )
{
   return ( hypre_InitializeVector( (hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintVector
 *--------------------------------------------------------------------------*/

int
HYPRE_PrintVector( HYPRE_Vector  vector,
                   char         *file_name )
{
   return ( hypre_PrintVector( (hypre_Vector *) vector,
                      file_name ) );
}

