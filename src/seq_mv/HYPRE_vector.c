/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_Vector interface
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_VectorCreate
 *--------------------------------------------------------------------------*/

HYPRE_Vector
HYPRE_VectorCreate( HYPRE_Int size )
{
   return ( (HYPRE_Vector) hypre_SeqVectorCreate(size) );
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_VectorDestroy( HYPRE_Vector vector )
{
   return ( hypre_SeqVectorDestroy( (hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_VectorInitialize( HYPRE_Vector vector )
{
   return ( hypre_SeqVectorInitialize( (hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
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
