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
 * hypre_PCGCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_PCGCAlloc( int count,
                 int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGFree
 *--------------------------------------------------------------------------*/

int
hypre_PCGFree( char *ptr )
{
   int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_PCGCreateVector( void *vvector )
{
   hypre_StructVector *vector = vvector;
   hypre_StructVector *new_vector;

   new_vector = hypre_StructVectorCreate( hypre_StructVectorComm(vector),
                                          hypre_StructVectorGrid(vector) );
   hypre_StructVectorInitialize(new_vector);
   hypre_StructVectorAssemble(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_PCGDestroyVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGDestroyVector( void *vvector )
{
   hypre_StructVector *vector = vvector;

   return( hypre_StructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_PCGMatvecCreate( void   *A,
                       void   *x )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup(matvec_data, A, x);

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_PCGMatvec
 *--------------------------------------------------------------------------*/

int
hypre_PCGMatvec( void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_StructMatvecCompute( matvec_data,
                                       alpha,
                                       (hypre_StructMatrix *) A,
                                       (hypre_StructVector *) x,
                                       beta,
                                       (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_PCGMatvecDestroy( void *matvec_data )
{
   return ( hypre_StructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_PCGInnerProd( void *x, 
                    void *y )
{
   return ( hypre_StructInnerProd( (hypre_StructVector *) x,
                                   (hypre_StructVector *) y ) );
}


/*--------------------------------------------------------------------------
 * hypre_PCGCopyVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGCopyVector( void *x, 
                     void *y )
{
   return ( hypre_StructCopy( (hypre_StructVector *) x,
                              (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGClearVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGClearVector( void *x )
{
   return ( hypre_StructVectorSetConstantValues( (hypre_StructVector *) x,
                                                 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGScaleVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGScaleVector( double  alpha,
                      void   *x     )
{
   return ( hypre_StructScale( alpha, (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGAxpy
 *--------------------------------------------------------------------------*/

int
hypre_PCGAxpy( double alpha,
               void   *x,
               void   *y )
{
   return ( hypre_StructAxpy( alpha, (hypre_StructVector *) x,
                              (hypre_StructVector *) y ) );
}

