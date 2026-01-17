/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * BiCGSTAB bicgstab
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_BiCGSTAB_HEADER
#define hypre_KRYLOV_BiCGSTAB_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic BiCGSTAB Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic BiCGSTAB linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABData and hypre_BiCGSTABFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name BiCGSTAB structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_BiCGSTABSFunctions} object ...
 **/

typedef struct
{
   hypre_KrylovPtrToCreateVector       CreateVector;
   hypre_KrylovPtrToDestroyVector      DestroyVector;
   hypre_KrylovPtrToMatvecCreate       MatvecCreate;
   hypre_KrylovPtrToMatvec             Matvec;
   hypre_KrylovPtrToMatvecDestroy      MatvecDestroy;
   hypre_KrylovPtrToInnerProd          InnerProd;
   hypre_KrylovPtrToCopyVector         CopyVector;
   hypre_KrylovPtrToClearVector        ClearVector;
   hypre_KrylovPtrToScaleVector        ScaleVector;
   hypre_KrylovPtrToAxpy               Axpy;
   hypre_KrylovPtrToCommInfo           CommInfo;
   hypre_KrylovPtrToPrecond            precond;
   hypre_KrylovPtrToPrecondSetup       precond_setup;

} hypre_BiCGSTABFunctions;

/**
 * The {\tt hypre\_BiCGSTABData} object ...
 **/

typedef struct
{
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      stop_crit;
   HYPRE_Int      converged;
   HYPRE_Int      hybrid;
   HYPRE_Real     tol;
   HYPRE_Real     cf_tol;
   HYPRE_Real     rel_residual_norm;
   HYPRE_Real     a_tol;


   void  *A;
   void  *r;
   void  *r0;
   void  *s;
   void  *v;
   void  *p;
   void  *q;

   void  *matvec_data;
   void  *precond_data;
   void  *precond_Mat;

   hypre_BiCGSTABFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int      logging;
   HYPRE_Int      print_level;
   HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_BiCGSTABData;

#define hypre_BiCGSTABDataHybrid(pcgdata)  ((pcgdata) -> hybrid)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic BiCGSTAB Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_BiCGSTABFunctions *
hypre_BiCGSTABFunctionsCreate(
   hypre_KrylovPtrToCreateVector  CreateVector,
   hypre_KrylovPtrToDestroyVector DestroyVector,
   hypre_KrylovPtrToMatvecCreate  MatvecCreate,
   hypre_KrylovPtrToMatvec        Matvec,
   hypre_KrylovPtrToMatvecDestroy MatvecDestroy,
   hypre_KrylovPtrToInnerProd     InnerProd,
   hypre_KrylovPtrToCopyVector    CopyVector,
   hypre_KrylovPtrToClearVector   ClearVector,
   hypre_KrylovPtrToScaleVector   ScaleVector,
   hypre_KrylovPtrToAxpy          Axpy,
   hypre_KrylovPtrToCommInfo      CommInfo,
   hypre_KrylovPtrToPrecondSetup  PrecondSetup,
   hypre_KrylovPtrToPrecond       Precond
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
