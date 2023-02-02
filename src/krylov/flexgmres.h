/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * FLEXGMRES flexible gmres
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_FLEXGMRES_HEADER
#define hypre_KRYLOV_FLEXGMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic FlexGMRES Interface
 *
 * A general description of the interface goes here...
 *
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESData and hypre_FlexGMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name FlexGMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_FlexGMRESFunctions} object ...
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

   HYPRE_Int    (*precond)(void *vdata, void *A, void *b, void *x );
   HYPRE_Int    (*precond_setup)(void *vdata, void *A, void *b, void *x );

   HYPRE_Int    (*modify_pc)(void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm );

} hypre_FlexGMRESFunctions;

/**
 * The {\tt hypre\_FlexGMRESData} object ...
 **/

typedef struct
{
   HYPRE_Int      k_dim;
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      rel_change;
   HYPRE_Int      stop_crit;
   HYPRE_Int      converged;
   HYPRE_Real   tol;
   HYPRE_Real   cf_tol;
   HYPRE_Real   a_tol;
   HYPRE_Real   rel_residual_norm;

   void   **pre_vecs;

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   hypre_FlexGMRESFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;

   HYPRE_Int     print_level; /* printing when print_level>0 */
   HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_FlexGMRESData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic FlexGMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_FlexGMRESFunctions *
hypre_FlexGMRESFunctionsCreate(
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
hypre_FlexGMRESCreate( hypre_FlexGMRESFunctions *fgmres_functions );

#ifdef __cplusplus
}
#endif
#endif
