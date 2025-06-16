/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * LGMRES lgmres
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_LGMRES_HEADER
#define hypre_KRYLOV_LGMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic LGMRES Interface
 *
 * A general description of the interface goes here...
 *
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_LGMRESData and hypre_LGMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name LGMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_LGMRESFunctions} object ...
 **/

typedef struct
{
   hypre_KrylovPtrToCAlloc             CAlloc;
   hypre_KrylovPtrToFree               Free;
   hypre_KrylovPtrToCommInfo           CommInfo;
   hypre_KrylovPtrToCreateVector       CreateVector;
   hypre_KrylovPtrToCreateVectorArray  CreateVectorArray;
   hypre_KrylovPtrToDestroyVector      DestroyVector;
   hypre_KrylovPtrToMatvecCreate       MatvecCreate;
   hypre_KrylovPtrToMatvec             Matvec;
   hypre_KrylovPtrToMatvecDestroy      MatvecDestroy;
   hypre_KrylovPtrToInnerProd          InnerProd;
   hypre_KrylovPtrToCopyVector         CopyVector;
   hypre_KrylovPtrToClearVector        ClearVector;
   hypre_KrylovPtrToScaleVector        ScaleVector;
   hypre_KrylovPtrToAxpy               Axpy;
   hypre_KrylovPtrToPrecond            precond;
   hypre_KrylovPtrToPrecondSetup       precond_setup;

} hypre_LGMRESFunctions;

/**
 * The {\tt hypre\_LGMRESData} object ...
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

   /*lgmres specific stuff */
   HYPRE_Int      aug_dim;
   HYPRE_Int      approx_constant;
   void   **aug_vecs;
   HYPRE_Int     *aug_order;
   void   **a_aug_vecs;
   /*---*/

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   hypre_LGMRESFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;

   HYPRE_Int     print_level; /* printing when print_level>0 */
   HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_LGMRESData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic LGMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_LGMRESFunctions *
hypre_LGMRESFunctionsCreate(
   hypre_KrylovPtrToCAlloc             CAlloc,
   hypre_KrylovPtrToFree               Free,
   hypre_KrylovPtrToCommInfo           CommInfo,
   hypre_KrylovPtrToCreateVector       CreateVector,
   hypre_KrylovPtrToCreateVectorArray  CreateVectorArray,
   hypre_KrylovPtrToDestroyVector      DestroyVector,
   hypre_KrylovPtrToMatvecCreate       MatvecCreate,
   hypre_KrylovPtrToMatvec             Matvec,
   hypre_KrylovPtrToMatvecDestroy      MatvecDestroy,
   hypre_KrylovPtrToInnerProd          InnerProd,
   hypre_KrylovPtrToCopyVector         CopyVector,
   hypre_KrylovPtrToClearVector        ClearVector,
   hypre_KrylovPtrToScaleVector        ScaleVector,
   hypre_KrylovPtrToAxpy               Axpy,
   hypre_KrylovPtrToPrecondSetup       PrecondSetup,
   hypre_KrylovPtrToPrecond            Precond
);

/**
 * Description...
 *
 * @param param [IN] ...
 **/

#ifdef __cplusplus
}
#endif
#endif
