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
   hypre_KrylovPtrToCAlloc             CAlloc;
   hypre_KrylovPtrToFree               Free;
   hypre_KrylovPtrToCommInfo           CommInfo;
   hypre_KrylovPtrToCreateVector       CreateVector;
   hypre_KrylovPtrToCreateVectorArray  CreateVectorArray;
   hypre_KrylovPtrToDestroyVector      DestroyVector;
   hypre_KrylovPtrToMatvecCreate       MatvecCreate;
   hypre_KrylovPtrToMatvec             Matvec;
   hypre_KrylovPtrToMatvecDestroy      MatvecDestroy;
   hypre_KrylovPtrToInnerProdTagged    InnerProd;
   hypre_KrylovPtrToCopyVector         CopyVector;
   hypre_KrylovPtrToClearVector        ClearVector;
   hypre_KrylovPtrToScaleVector        ScaleVector;
   hypre_KrylovPtrToAxpy               Axpy;
   hypre_KrylovPtrToPrecond            precond;
   hypre_KrylovPtrToPrecondSetup       precond_setup;

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
   void  *w_3;
   void  **p;
   void  *xref;   /* reference solution for error computation */

   void    *matvec_data;
   void    *precond_data;
   void    *precond_Mat;

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
   hypre_KrylovPtrToCAlloc             CAlloc,
   hypre_KrylovPtrToFree               Free,
   hypre_KrylovPtrToCommInfo           CommInfo,
   hypre_KrylovPtrToCreateVector       CreateVector,
   hypre_KrylovPtrToCreateVectorArray  CreateVectorArray,
   hypre_KrylovPtrToDestroyVector      DestroyVector,
   hypre_KrylovPtrToMatvecCreate       MatvecCreate,
   hypre_KrylovPtrToMatvec             Matvec,
   hypre_KrylovPtrToMatvecDestroy      MatvecDestroy,
   hypre_KrylovPtrToInnerProdTagged    InnerProd,
   hypre_KrylovPtrToCopyVector         CopyVector,
   hypre_KrylovPtrToClearVector        ClearVector,
   hypre_KrylovPtrToScaleVector        ScaleVector,
   hypre_KrylovPtrToAxpy               Axpy,
   hypre_KrylovPtrToPrecond            Precond,
   hypre_KrylovPtrToPrecondSetup       PrecondSetup
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
