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
 * HYPRE_VectorCreate
 *--------------------------------------------------------------------------*/

HYPRE_Vector
HYPRE_VectorCreate( int size )
{
   return ( (HYPRE_Vector) hypre_SeqVectorCreate(size) );
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_VectorDestroy( HYPRE_Vector vector )
{
   return ( hypre_SeqVectorDestroy( (hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_VectorInitialize( HYPRE_Vector vector )
{
   return ( hypre_SeqVectorInitialize( (hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorPrint
 *--------------------------------------------------------------------------*/

int
HYPRE_VectorPrint( HYPRE_Vector  vector,
                   char         *file_name )
{
   return ( hypre_SeqVectorPrint( (hypre_Vector *) vector,
                      file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorRead
 *--------------------------------------------------------------------------*/

HYPRE_Vector
HYPRE_VectorRead( char         *file_name )
{
   return ( (HYPRE_Vector) hypre_SeqVectorRead( file_name ) );
}
