#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDPCGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDPCGCreate( HYPRE_Solver *solver )
{
   hypre_PCGFunctions * pcg_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_ParAMGDDKrylovFree, hypre_ParAMGDDKrylovCommInfo,
         hypre_ParAMGDDKrylovCreateVector,
         hypre_ParAMGDDKrylovDestroyVector, hypre_ParAMGDDKrylovMatvecCreate,
         hypre_ParAMGDDKrylovMatvec, hypre_ParAMGDDKrylovMatvecDestroy,
         hypre_ParAMGDDKrylovInnerProd, hypre_ParAMGDDKrylovCopyVector,
         hypre_ParAMGDDKrylovClearVector,
         hypre_ParAMGDDKrylovScaleVector, hypre_ParAMGDDKrylovAxpy,
         hypre_ParAMGDDKrylovIdentitySetup, hypre_ParAMGDDKrylovIdentity );
   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDPCGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParAMGDDPCGDestroy( HYPRE_Solver solver )
{
   return( hypre_PCGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDPCGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParAMGDDPCGSetup( HYPRE_Solver solver,
                      hypre_ParCompGridMatrix *A,
                      hypre_ParCompGridVector *b,
                      hypre_ParCompGridVector *x      )
{
   return( HYPRE_PCGSetup( solver,
                           (HYPRE_Matrix) A,
                           (HYPRE_Vector) b,
                           (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDPCGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParAMGDDPCGSolve( HYPRE_Solver solver,
                      hypre_ParCompGridMatrix *A,
                      hypre_ParCompGridVector *b,
                      hypre_ParCompGridVector *x      )
{
   return( HYPRE_PCGSolve( solver,
                           (HYPRE_Matrix) A,
                           (HYPRE_Vector) b,
                           (HYPRE_Vector) x ) );
}
/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovCAlloc 
 *--------------------------------------------------------------------------*/

void *
hypre_ParAMGDDKrylovCAlloc( HYPRE_Int count,
                       HYPRE_Int elt_size )
{
   return( (void*) hypre_CTAlloc( char, count * elt_size , HYPRE_MEMORY_HOST) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovFree 
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovFree( void *ptr )
{
   HYPRE_Int ierr = 0;

   hypre_Free( ptr , HYPRE_MEMORY_HOST);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_ParAMGDDKrylovCreateVector( void *vvector )
{
   hypre_ParCompGridVector *vector = (hypre_ParCompGridVector *) vvector;
   hypre_ParCompGridVector *new_vector;

   new_vector = hypre_ParCompGridVectorCreate(); 
   hypre_ParCompGridVectorInitialize(new_vector, hypre_VectorSize(hypre_ParCompGridVectorOwned(vector)), hypre_VectorSize(hypre_ParCompGridVectorNonOwned(vector)), hypre_ParCompGridVectorNumReal(vector));

   return ( (void *) new_vector );
}


/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovDestroyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovDestroyVector( void *vvector )
{
   hypre_ParCompGridVector *vector = (hypre_ParCompGridVector *) vvector;

   return( hypre_ParCompGridVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovMatvecCreate 
 *--------------------------------------------------------------------------*/

void *
hypre_ParAMGDDKrylovMatvecCreate( void   *A,
                             void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovMatvec( void   *matvec_data,
                       HYPRE_Complex  alpha,
                       void   *A,
                       void   *x,
                       HYPRE_Complex  beta,
                       void   *y           )
{
   return ( hypre_ParCompGridRealMatvec ( alpha,
                                       (hypre_ParCompGridMatrix *) A,
                                       (hypre_ParCompGridVector *) x,
                                       beta,
                                       (hypre_ParCompGridVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovMatvecDestroy 
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_ParAMGDDKrylovInnerProd( void *x, 
                          void *y )
{
   return ( hypre_ParCompGridVectorRealInnerProd( (hypre_ParCompGridVector *) x,
                                      (hypre_ParCompGridVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovCopyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovCopyVector( void *x, 
                           void *y )
{
   return ( hypre_ParCompGridVectorRealCopy( (hypre_ParCompGridVector *) x,
                                 (hypre_ParCompGridVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovClearVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovClearVector( void *x )
{
   return ( hypre_ParCompGridVectorRealSetConstantValues( (hypre_ParCompGridVector *) x, 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovScaleVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovScaleVector( HYPRE_Complex  alpha,
                            void   *x     )
{
   return ( hypre_ParCompGridVectorRealScale( alpha, (hypre_ParCompGridVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovAxpy( HYPRE_Complex alpha,
                     void   *x,
                     void   *y )
{
   return ( hypre_ParCompGridVectorRealAxpy( alpha, (hypre_ParCompGridVector *) x,
                                 (hypre_ParCompGridVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovCommInfo
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovCommInfo( void   *A, HYPRE_Int *my_id, HYPRE_Int *num_procs)
{
   MPI_Comm comm = hypre_MPI_COMM_SELF;
   hypre_MPI_Comm_size(comm,num_procs);
   hypre_MPI_Comm_rank(comm,my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovIdentitySetup 
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovIdentitySetup( void *vdata,
                              void *A,
                              void *b,
                              void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDKrylovIdentity
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParAMGDDKrylovIdentity( void *vdata,
                         void *A,
                         void *b,
                         void *x     )

{
   return( hypre_ParAMGDDKrylovCopyVector( b, x ) );
}

