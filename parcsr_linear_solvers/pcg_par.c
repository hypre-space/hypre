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
   hypre_ParVector *vector = vvector;
   hypre_ParVector *new_vector;

   new_vector = hypre_CreateParVector( hypre_ParVectorComm(vector),
				       hypre_ParVectorGlobalSize(vector),	
                                       hypre_ParVectorPartitioning(vector) );
   hypre_InitializeParVector(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_PCGCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_PCGCreateVectorArray(int n, void *vvector )
{
   hypre_ParVector *vector = vvector;
   hypre_ParVector **new_vector;
   int i;

   new_vector = hypre_CTAlloc(hypre_ParVector*,n);
   for (i=0; i < n; i++)
   {
      new_vector[i] = hypre_CreateParVector( hypre_ParVectorComm(vector),
				       hypre_ParVectorGlobalSize(vector),	
                                       hypre_ParVectorPartitioning(vector) );
      hypre_InitializeParVector(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_PCGDestroyVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGDestroyVector( void *vvector )
{
   hypre_ParVector *vector = vvector;

   return( hypre_DestroyParVector( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_PCGMatvecCreate( void   *A,
                       void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

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
   return ( hypre_ParMatvec ( alpha,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) x,
                               beta,
                              (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGMatvecT
 *--------------------------------------------------------------------------*/

int
hypre_PCGMatvecT(void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_ParMatvecT( alpha,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) x,
                               beta,
                              (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_PCGMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_PCGInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_PCGInnerProd( void *x, 
                    void *y )
{
   return ( hypre_ParInnerProd( (hypre_ParVector *) x,
                                (hypre_ParVector *) y ) );
}


/*--------------------------------------------------------------------------
 * hypre_PCGCopyVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGCopyVector( void *x, 
                     void *y )
{
   return ( hypre_CopyParVector( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGClearVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGClearVector( void *x )
{
   return ( hypre_SetParVectorConstantValues( (hypre_ParVector *) x, 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGScaleVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGScaleVector( double  alpha,
                      void   *x     )
{
   return ( hypre_ScaleParVector( alpha, (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGAxpy
 *--------------------------------------------------------------------------*/

int
hypre_PCGAxpy( double alpha,
               void   *x,
               void   *y )
{
   return ( hypre_ParAxpy( alpha, (hypre_ParVector *) x,
                              (hypre_ParVector *) y ) );
}

