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

/*--------------------------------------------------------------------------
 * HYPRE_SetParVectorConstantValues
 *--------------------------------------------------------------------------*/

int
HYPRE_SetParVectorConstantValues( HYPRE_ParVector  vector,
                      		  double	   value )
{
   return ( hypre_SetParVectorConstantValues( (hypre_ParVector *) vector,
                                  value ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetParVectorRandomValues
 *--------------------------------------------------------------------------*/

int
HYPRE_SetParVectorRandomValues( HYPRE_ParVector  vector,
                      		int	         seed  )
{
   return ( hypre_SetParVectorRandomValues( (hypre_ParVector *) vector,
                                  seed ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CopyParVector
 *--------------------------------------------------------------------------*/

int
HYPRE_CopyParVector( HYPRE_ParVector x, HYPRE_ParVector y)
{
   return ( hypre_CopyParVector( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ScaleParVector
 *--------------------------------------------------------------------------*/

int
HYPRE_ScaleParVector( double value, HYPRE_ParVector x)
{
   return ( hypre_ScaleParVector( value, (hypre_ParVector *) x) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParInnerProd
 *--------------------------------------------------------------------------*/

double
HYPRE_ParInnerProd( HYPRE_ParVector x, HYPRE_ParVector y)
{
   return ( hypre_ParInnerProd( (hypre_ParVector *) x, 
				(hypre_ParVector *) y) );
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorToParVector
 *--------------------------------------------------------------------------*/

HYPRE_ParVector
HYPRE_VectorToParVector( MPI_Comm comm, HYPRE_Vector b, int *partitioning)
{
   return ( (HYPRE_ParVector) hypre_VectorToParVector (comm, 
		(hypre_Vector *) b, partitioning ));
}
