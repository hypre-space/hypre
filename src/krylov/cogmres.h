/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * COGMRES cogmres
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_COGMRES_HEADER
#define hypre_KRYLOV_COGMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic COGMRES Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic COGMRES linear solver interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_COGMRESData and hypre_COGMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name COGMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_COGMRESFunctions} object ...
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
   hypre_KrylovPtrToMassInnerProd      MassInnerProd;
   hypre_KrylovPtrToMassDotpTwo        MassDotpTwo;
   hypre_KrylovPtrToCopyVector         CopyVector;
   hypre_KrylovPtrToClearVector        ClearVector;
   hypre_KrylovPtrToScaleVector        ScaleVector;
   hypre_KrylovPtrToAxpy               Axpy;
   hypre_KrylovPtrToMassAxpy           MassAxpy;
   hypre_KrylovPtrToPrecond            precond;
   hypre_KrylovPtrToPrecondSetup       precond_setup;
   hypre_KrylovPtrToModifyPC           modify_pc;

} hypre_COGMRESFunctions;

/**
 * The {\tt hypre\_COGMRESData} object ...
 **/

typedef struct
{
   HYPRE_Int      k_dim;
   HYPRE_Int      unroll;
   HYPRE_Int      cgs;
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      rel_change;
   HYPRE_Int      skip_real_r_check;
   HYPRE_Int      stop_crit;
   HYPRE_Int      converged;
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

   hypre_COGMRESFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;

   HYPRE_Int     print_level; /* printing when print_level>0 */
   HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_COGMRESData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic COGMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_COGMRESFunctions *
hypre_COGMRESFunctionsCreate(
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
   hypre_KrylovPtrToMassInnerProd      MassInnerProd,
   hypre_KrylovPtrToMassDotpTwo        MassDotpTwo,
   hypre_KrylovPtrToCopyVector         CopyVector,
   hypre_KrylovPtrToClearVector        ClearVector,
   hypre_KrylovPtrToScaleVector        ScaleVector,
   hypre_KrylovPtrToAxpy               Axpy,
   hypre_KrylovPtrToMassAxpy           MassAxpy,
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
