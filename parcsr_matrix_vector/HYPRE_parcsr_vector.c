/*BHEADER*********************************************************************
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
 * HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorCreate( MPI_Comm comm,
                       int      global_size, 
                       int     *partitioning,
		       HYPRE_ParVector *vector )
{
   *vector = (HYPRE_ParVector) hypre_ParVectorCreate(comm, global_size,
                                                    partitioning) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParVectorDestroy( HYPRE_ParVector vector )
{
   return ( hypre_ParVectorDestroy( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParVectorInitialize( HYPRE_ParVector vector )
{
   return ( hypre_ParVectorInitialize( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorRead( MPI_Comm  comm,
                     char     *file_name, 
		     HYPRE_ParVector *vector)
{
   *vector = (HYPRE_ParVector) hypre_ParVectorRead( comm, file_name ) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorPrint( HYPRE_ParVector  vector,
                      char         *file_name )
{
   return ( hypre_ParVectorPrint( (hypre_ParVector *) vector,
                                  file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorSetConstantValues( HYPRE_ParVector  vector,
                      		  double	   value )
{
   return ( hypre_ParVectorSetConstantValues( (hypre_ParVector *) vector,
                                  value ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorSetRandomValues( HYPRE_ParVector  vector,
                      		int	         seed  )
{
   return ( hypre_ParVectorSetRandomValues( (hypre_ParVector *) vector,
                                  seed ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorCopy( HYPRE_ParVector x, HYPRE_ParVector y)
{
   return ( hypre_ParVectorCopy( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorScale
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorScale( double value, HYPRE_ParVector x)
{
   return ( hypre_ParVectorScale( value, (hypre_ParVector *) x) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorInnerProd( HYPRE_ParVector x, HYPRE_ParVector y, double *prod)
{
   *prod = hypre_ParVectorInnerProd( (hypre_ParVector *) x, 
			(hypre_ParVector *) y) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorToParVector
 *--------------------------------------------------------------------------*/

int
HYPRE_VectorToParVector( MPI_Comm comm, HYPRE_Vector b, int *partitioning,
			 HYPRE_ParVector *vector)
{
   *vector = (HYPRE_ParVector) hypre_VectorToParVector (comm, 
		(hypre_Vector *) b, partitioning );
   return 0;
}
