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
 * hypre_KrylovCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_KrylovCreateVectorArray(int n, void *vvector )
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
 * hypre_KrylovDestroyVector
 *--------------------------------------------------------------------------*/

int
hypre_KrylovDestroyVector( void *vvector )
{
   hypre_SStructVector *vector = vvector;

   return( HYPRE_SStructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_KrylovMatvecCreate( void   *A,
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
 * hypre_KrylovMatvec
 *--------------------------------------------------------------------------*/

int
hypre_KrylovMatvec( void   *matvec_data,
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
 * hypre_KrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_KrylovMatvecDestroy( void *matvec_data )
{
   return ( hypre_SStructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_KrylovInnerProd( void *x, 
                       void *y )
{
   double result;

   hypre_SStructInnerProd( (hypre_SStructVector *) x,
                           (hypre_SStructVector *) y, &result );

   return result;
}


/*--------------------------------------------------------------------------
 * hypre_KrylovCopyVector
 *--------------------------------------------------------------------------*/

int
hypre_KrylovCopyVector( void *x, 
                        void *y )
{
   return ( hypre_SStructCopy( (hypre_SStructVector *) x,
                               (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovClearVector
 *--------------------------------------------------------------------------*/

int
hypre_KrylovClearVector( void *x )
{
   return ( hypre_SStructVectorSetConstantValues( (hypre_SStructVector *) x,
                                                  0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovScaleVector
 *--------------------------------------------------------------------------*/

int
hypre_KrylovScaleVector( double  alpha,
                         void   *x )
{
   return ( hypre_SStructScale( alpha, (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovAxpy
 *--------------------------------------------------------------------------*/

int
hypre_KrylovAxpy( double alpha,
                  void   *x,
                  void   *y )
{
   return ( hypre_SStructAxpy( alpha, (hypre_SStructVector *) x,
                                     (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovCommInfo
 *--------------------------------------------------------------------------*/

int
hypre_KrylovCommInfo( void  *A,
                      int   *my_id,
                      int   *num_procs )
{
   MPI_Comm comm = hypre_SStructMatrixComm((hypre_SStructMatrix *) A);
   MPI_Comm_size(comm,num_procs);
   MPI_Comm_rank(comm,my_id);
   return 0;
}

