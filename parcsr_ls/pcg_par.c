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
 * Struct matrix-vector implementation of PCG interface routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_ParKrylovCAlloc( int count,
                 int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovFree
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovFree( char *ptr )
{
   int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovCreateVector( void *vvector )
{
   hypre_ParVector *vector = vvector;
   hypre_ParVector *new_vector;

   new_vector = hypre_ParVectorCreate( hypre_ParVectorComm(vector),
				       hypre_ParVectorGlobalSize(vector),	
                                       hypre_ParVectorPartitioning(vector) );
   hypre_ParVectorSetPartitioningOwner(new_vector,0);
   hypre_ParVectorInitialize(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovCreateVectorArray(int n, void *vvector )
{
   hypre_ParVector *vector = vvector;
   hypre_ParVector **new_vector;
   int i;

   new_vector = hypre_CTAlloc(hypre_ParVector*,n);
   for (i=0; i < n; i++)
   {
      new_vector[i] = hypre_ParVectorCreate( hypre_ParVectorComm(vector),
				       hypre_ParVectorGlobalSize(vector),	
                                       hypre_ParVectorPartitioning(vector) );
      hypre_ParVectorSetPartitioningOwner(new_vector[i],0);
      hypre_ParVectorInitialize(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovDestroyVector
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovDestroyVector( void *vvector )
{
   hypre_ParVector *vector = vvector;

   return( hypre_ParVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovMatvecCreate( void   *A,
                       void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvec
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovMatvec( void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_ParCSRMatrixMatvec ( alpha,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) x,
                               beta,
                              (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecT
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovMatvecT(void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_ParCSRMatrixMatvecT( alpha,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) x,
                               beta,
                              (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_ParKrylovInnerProd( void *x, 
                    void *y )
{
   return ( hypre_ParVectorInnerProd( (hypre_ParVector *) x,
                                (hypre_ParVector *) y ) );
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovCopyVector
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovCopyVector( void *x, 
                     void *y )
{
   return ( hypre_ParVectorCopy( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovClearVector
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovClearVector( void *x )
{
   return ( hypre_ParVectorSetConstantValues( (hypre_ParVector *) x, 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovScaleVector
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovScaleVector( double  alpha,
                      void   *x     )
{
   return ( hypre_ParVectorScale( alpha, (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovAxpy
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovAxpy( double alpha,
               void   *x,
               void   *y )
{
   return ( hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x,
                              (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCommInfo
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovCommInfo( void   *A, int *my_id, int *num_procs)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm ( (hypre_ParCSRMatrix *) A);
   MPI_Comm_size(comm,num_procs);
   MPI_Comm_rank(comm,my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovIdentitySetup
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovIdentitySetup( void *vdata,
                        void *A,
                        void *b,
                        void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovIdentity
 *--------------------------------------------------------------------------*/

int
hypre_ParKrylovIdentity( void *vdata,
                   void *A,
                   void *b,
                   void *x     )

{
   return( hypre_ParKrylovCopyVector( b, x ) );
}

