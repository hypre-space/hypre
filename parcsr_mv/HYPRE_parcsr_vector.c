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
 * HYPRE_ParVector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_CreateParVector
 *--------------------------------------------------------------------------*/

HYPRE_ParVector
HYPRE_CreateParVector( MPI_Comm comm,
                       int      global_size, 
                       int     *partitioning )
{
   return ( (HYPRE_ParVector) hypre_CreateParVector(comm, global_size,
                                                    partitioning) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DestroyParVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_DestroyParVector( HYPRE_ParVector vector )
{
   return ( hypre_DestroyParVector( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeParVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_InitializeParVector( HYPRE_ParVector vector )
{
   return ( hypre_InitializeParVector( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintParVector
 *--------------------------------------------------------------------------*/

int
HYPRE_PrintParVector( HYPRE_ParVector  vector,
                      char         *file_name )
{
   return ( hypre_PrintParVector( (hypre_ParVector *) vector,
                                  file_name ) );
}

