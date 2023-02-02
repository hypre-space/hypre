/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * GMRES gmres
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_GMRES_HEADER
#define hypre_KRYLOV_GMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic GMRES Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic GMRES linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_GMRESData and hypre_GMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name GMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_GMRESFunctions} object ...
 **/

typedef struct
{
   void *       (*CAlloc)        ( size_t count, size_t elt_size );
   HYPRE_Int    (*Free)          ( void *ptr );
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors );
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

   HYPRE_Int    (*precond)       ();
   HYPRE_Int    (*precond_setup) ();

} hypre_GMRESFunctions;

/**
 * The {\tt hypre\_GMRESData} object ...
 **/

typedef struct
{
   HYPRE_Int      k_dim;
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      rel_change;
   HYPRE_Int      skip_real_r_check;
   HYPRE_Int      stop_crit;
   HYPRE_Int      converged;
   HYPRE_Int      hybrid;
   HYPRE_Real   tol;
   HYPRE_Real   cf_tol;
   HYPRE_Real   a_tol;
   HYPRE_Real   rel_residual_norm;

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   hypre_GMRESFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;

   HYPRE_Int     print_level; /* printing when print_level>0 */
   HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_GMRESData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic GMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_GMRESFunctions *
hypre_GMRESFunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( void *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
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
hypre_GMRESCreate( hypre_GMRESFunctions *gmres_functions );

#ifdef __cplusplus
}
#endif
#endif
