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
 * hypre_KrylovCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_KrylovCAlloc( int count,
                 int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovFree
 *--------------------------------------------------------------------------*/

int
hypre_KrylovFree( char *ptr )
{
   int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_KrylovCreateVector( void *vvector )
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
 * hypre_KrylovCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_KrylovCreateVectorArray(int n, void *vvector )
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
 * hypre_KrylovDestroyVector
 *--------------------------------------------------------------------------*/

int
hypre_KrylovDestroyVector( void *vvector )
{
   hypre_ParVector *vector = vvector;

   return( hypre_ParVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_KrylovMatvecCreate( void   *A,
                       void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovMatvec
 *--------------------------------------------------------------------------*/

int
hypre_KrylovMatvec( void   *matvec_data,
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
 * hypre_KrylovMatvecT
 *--------------------------------------------------------------------------*/

int
hypre_KrylovMatvecT(void   *matvec_data,
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
 * hypre_KrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_KrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_KrylovInnerProd( void *x, 
                    void *y )
{
   return ( hypre_ParVectorInnerProd( (hypre_ParVector *) x,
                                (hypre_ParVector *) y ) );
}


/*--------------------------------------------------------------------------
 * hypre_KrylovCopyVector
 *--------------------------------------------------------------------------*/

int
hypre_KrylovCopyVector( void *x, 
                     void *y )
{
   return ( hypre_ParVectorCopy( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovClearVector
 *--------------------------------------------------------------------------*/

int
hypre_KrylovClearVector( void *x )
{
   return ( hypre_ParVectorSetConstantValues( (hypre_ParVector *) x, 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovScaleVector
 *--------------------------------------------------------------------------*/

int
hypre_KrylovScaleVector( double  alpha,
                      void   *x     )
{
   return ( hypre_ParVectorScale( alpha, (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovAxpy
 *--------------------------------------------------------------------------*/

int
hypre_KrylovAxpy( double alpha,
               void   *x,
               void   *y )
{
   return ( hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x,
                              (hypre_ParVector *) y ) );
}

