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
 * hypre_PCGNewVector
 *--------------------------------------------------------------------------*/

void *
hypre_PCGNewVector( void *vvector )
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
 * hypre_PCGFreeVector
 *--------------------------------------------------------------------------*/

int
hypre_PCGFreeVector( void *vvector )
{
   hypre_ParVector *vector = vvector;

   return( hypre_DestroyParVector( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGMatvecInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_PCGMatvecInitialize( void   *A,
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
 * hypre_PCGMatvecFinalize
 *--------------------------------------------------------------------------*/

int
hypre_PCGMatvecFinalize( void *matvec_data )
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

