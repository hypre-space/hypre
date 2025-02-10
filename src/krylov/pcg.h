/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Preconditioned conjugate gradient (Omin) headers
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_PCG_HEADER
#define hypre_KRYLOV_PCG_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic PCG Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic PCG linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_PCGData and hypre_PCGFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name PCG structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_PCGSFunctions} object ...
 **/

typedef struct
{
   void *       (*CAlloc)        ( size_t count, size_t elt_size );
   HYPRE_Int    (*Free)          ( void *ptr );
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   HYPRE_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   HYPRE_Real   (*InnerProd)     ( void *x, void *y );
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x );
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y );

   HYPRE_Int    (*precond)();
   HYPRE_Int    (*precond_setup)();

} hypre_PCGFunctions;

/**
 * The {\tt hypre\_PCGData} object ...
 **/

/*
 Summary of Parameters to Control Stopping Test:
 - Standard (default) error tolerance: |delta-residual|/|right-hand-side|<tol
 where the norm is an energy norm wrt preconditioner, |r|=sqrt(<Cr,r>).
 - two_norm!=0 means: the norm is the L2 norm, |r|=sqrt(<r,r>)
 - rel_change!=0 means: if pass the other stopping criteria, also check the
 relative change in the solution x.  Pass iff this relative change is small.
 - tol = relative error tolerance, as above
 -a_tol = absolute convergence tolerance (default is 0.0)
   If one desires the convergence test to check the absolute
   convergence tolerance *only*, then set the relative convergence
   tolerance to 0.0.  (The default convergence test is  <C*r,r> <=
   max(relative_tolerance^2 * <C*b, b>, absolute_tolerance^2)
- cf_tol = convergence factor tolerance; if >0 used for special test
  for slow convergence
- stop_crit!=0 means (TO BE PHASED OUT):
  pure absolute error tolerance rather than a pure relative
  error tolerance on the residual.  Never applies if rel_change!=0 or atolf!=0.
 - atolf = absolute error tolerance factor to be used _together_ with the
 relative error tolerance, |delta-residual| / ( atolf + |right-hand-side| ) < tol
  (To BE PHASED OUT)
 - recompute_residual means: when the iteration seems to be converged, recompute the
 residual from scratch (r=b-Ax) and use this new residual to repeat the convergence test.
 This can be expensive, use this only if you have seen a problem with the regular
 residual computation.
 - recompute_residual_p means: recompute the residual from scratch (r=b-Ax)
 every "recompute_residual_p" iterations.  This can be expensive and degrade the
 convergence. Use it only if you have seen a problem with the regular residual
 computation.
 - skip_break means that cg will not stop for very small alpha and gamma. default: 0
*/

typedef struct
{
   HYPRE_Real   tol;
   HYPRE_Real   atolf;
   HYPRE_Real   cf_tol;
   HYPRE_Real   a_tol;
   HYPRE_Real   rtol;
   HYPRE_Int    max_iter;
   HYPRE_Int    two_norm;
   HYPRE_Int    rel_change;
   HYPRE_Int    recompute_residual;
   HYPRE_Int    recompute_residual_p;
   HYPRE_Int    stop_crit;
   HYPRE_Int    converged;
   HYPRE_Int    hybrid;
   HYPRE_Int    skip_break;
   HYPRE_Int    flex;

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                  If that is ever changed, it still must be kept if logging>1 */
   void    *r_old; /* old residual needed for flexible CG, PR method */
   void    *v; /* work vector only needed if recompute_residual_p uis used */

   HYPRE_Int  owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void      *matvec_data;
   void      *precond_data;

   hypre_PCGFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int    num_iterations;
   HYPRE_Real   rel_residual_norm;

   HYPRE_Int    print_level; /* printing when print_level>0 */
   HYPRE_Int    logging;  /* extra computations for logging when logging>0 */
   HYPRE_Real  *norms;
   HYPRE_Real  *rel_norms;

} hypre_PCGData;

#define hypre_PCGDataOwnsMatvecData(pcgdata)  ((pcgdata) -> owns_matvec_data)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic PCG Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_PCGFunctions *
hypre_PCGFunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( void *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
);

/**
 * Description...
 *
 * @param param [IN] ...
 **/

void *
hypre_PCGCreate( hypre_PCGFunctions *pcg_functions );

#ifdef __cplusplus
}
#endif

#endif
