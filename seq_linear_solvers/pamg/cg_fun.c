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
 * csr matrix-vector implementation of PCG interface routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CGCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_CGCAlloc( int count,
                 int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGFree
 *--------------------------------------------------------------------------*/

int
hypre_CGFree( char *ptr )
{
   int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_CGCreateVector( void *vvector )
{
   hypre_Vector *vector = vvector;
   hypre_Vector *new_vector;

   new_vector = hypre_VectorCreate( hypre_VectorSize(vector));
   hypre_VectorInitialize(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_CGCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_CGCreateVectorArray(int n, void *vvector )
{
   hypre_Vector *vector = vvector;
   hypre_Vector **new_vector;
   int i;

   new_vector = hypre_CTAlloc(hypre_Vector*,n);
   for (i=0; i < n; i++)
   {
      new_vector[i] = hypre_VectorCreate( hypre_VectorSize(vector));
      hypre_VectorInitialize(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_CGDestroyVector
 *--------------------------------------------------------------------------*/

int
hypre_CGDestroyVector( void *vvector )
{
   hypre_Vector *vector = vvector;

   return( hypre_VectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_CGMatvecCreate( void   *A,
                       void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_CGMatvec
 *--------------------------------------------------------------------------*/

int
hypre_CGMatvec( void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_CSRMatrixMatvec ( alpha,
                              (hypre_CSRMatrix *) A,
                              (hypre_Vector *) x,
                               beta,
                              (hypre_Vector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGMatvecT
 *--------------------------------------------------------------------------*/

int
hypre_CGMatvecT(void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_CSRMatrixMatvecT( alpha,
                              (hypre_CSRMatrix *) A,
                              (hypre_Vector *) x,
                               beta,
                              (hypre_Vector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_CGMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CGInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_CGInnerProd( void *x, 
                    void *y )
{
   return ( hypre_VectorInnerProd( (hypre_Vector *) x,
                                (hypre_Vector *) y ) );
}


/*--------------------------------------------------------------------------
 * hypre_CGCopyVector
 *--------------------------------------------------------------------------*/

int
hypre_CGCopyVector( void *x, 
                     void *y )
{
   return ( hypre_VectorCopy( (hypre_Vector *) x,
                                 (hypre_Vector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGClearVector
 *--------------------------------------------------------------------------*/

int
hypre_CGClearVector( void *x )
{
   return ( hypre_VectorSetConstantValues( (hypre_Vector *) x, 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGScaleVector
 *--------------------------------------------------------------------------*/

int
hypre_CGScaleVector( double  alpha,
                      void   *x     )
{
   return ( hypre_VectorScale( alpha, (hypre_Vector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGAxpy
 *--------------------------------------------------------------------------*/

int
hypre_CGAxpy( double alpha,
               void   *x,
               void   *y )
{
   return ( hypre_VectorAxpy( alpha, (hypre_Vector *) x,
                              (hypre_Vector *) y ) );
}


