/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * SStruct matrix-vector implementation of Krylov interface routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_SStructKrylovCAlloc( int count,
                    int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovFree
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovFree( char *ptr )
{
   int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovCreateVector( void *vvector )
{
   hypre_SStructVector *vector = vvector;
   hypre_SStructVector *new_vector;

   HYPRE_SStructVectorCreate(hypre_SStructVectorComm(vector),
                             hypre_SStructVectorGrid(vector),
                             &new_vector);
   HYPRE_SStructVectorInitialize(new_vector);
   HYPRE_SStructVectorAssemble(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovCreateVectorArray(int n, void *vvector )
{
   hypre_SStructVector *vector = vvector;
   hypre_SStructVector **new_vector;
   int i;

   new_vector = hypre_CTAlloc(hypre_SStructVector*,n);
   for (i=0; i < n; i++)
   {
      HYPRE_SStructVectorCreate(hypre_SStructVectorComm(vector),
                                hypre_SStructVectorGrid(vector),
                                &new_vector[i]);
      HYPRE_SStructVectorInitialize(new_vector[i]);
      HYPRE_SStructVectorAssemble(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovDestroyVector
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovDestroyVector( void *vvector )
{
   hypre_SStructVector *vector = vvector;

   return( HYPRE_SStructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovMatvecCreate( void   *A,
                          void   *x )
{
   void *matvec_data;

   hypre_SStructMatvecCreate( &matvec_data );
   hypre_SStructMatvecSetup( matvec_data,
                             (hypre_SStructMatrix *) A,
                             (hypre_SStructVector *) x );

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovMatvec
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovMatvec( void   *matvec_data,
                    double  alpha,
                    void   *A,
                    void   *x,
                    double  beta,
                    void   *y )
{
   return ( hypre_SStructMatvec( alpha,
                                 (hypre_SStructMatrix *) A,
                                 (hypre_SStructVector *) x,
                                 beta,
                                 (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovMatvecDestroy( void *matvec_data )
{
   return ( hypre_SStructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_SStructKrylovInnerProd( void *x, 
                       void *y )
{
   double result;

   hypre_SStructInnerProd( (hypre_SStructVector *) x,
                           (hypre_SStructVector *) y, &result );

   return result;
}


/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCopyVector
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovCopyVector( void *x, 
                        void *y )
{
   return ( hypre_SStructCopy( (hypre_SStructVector *) x,
                               (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovClearVector
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovClearVector( void *x )
{
   return ( hypre_SStructVectorSetConstantValues( (hypre_SStructVector *) x,
                                                  0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovScaleVector
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovScaleVector( double  alpha,
                         void   *x )
{
   return ( hypre_SStructScale( alpha, (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovAxpy
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovAxpy( double alpha,
                  void   *x,
                  void   *y )
{
   return ( hypre_SStructAxpy( alpha, (hypre_SStructVector *) x,
                                     (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCommInfo
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovCommInfo( void  *A,
                      int   *my_id,
                      int   *num_procs )
{
   MPI_Comm comm = hypre_SStructMatrixComm((hypre_SStructMatrix *) A);
   MPI_Comm_size(comm,num_procs);
   MPI_Comm_rank(comm,my_id);
   return 0;
}

