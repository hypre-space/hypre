/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * cgnr (conjugate gradient on the normal equations A^TAx = A^Tb) functions
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_CGNR_HEADER
#define hypre_KRYLOV_CGNR_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic CGNR Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic CGNR linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_CGNRData and hypre_CGNRFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name CGNR structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_CGNRSFunctions} object ...
 **/

typedef struct
{
   hypre_KrylovPtrToCommInfo           CommInfo;
   hypre_KrylovPtrToCreateVector       CreateVector;
   hypre_KrylovPtrToDestroyVector      DestroyVector;
   hypre_KrylovPtrToMatvecCreate       MatvecCreate;
   hypre_KrylovPtrToMatvec             Matvec;
   hypre_KrylovPtrToMatvecT            MatvecT;
   hypre_KrylovPtrToMatvecDestroy      MatvecDestroy;
   hypre_KrylovPtrToInnerProd          InnerProd;
   hypre_KrylovPtrToCopyVector         CopyVector;
   hypre_KrylovPtrToClearVector        ClearVector;
   hypre_KrylovPtrToScaleVector        ScaleVector;
   hypre_KrylovPtrToAxpy               Axpy;
   hypre_KrylovPtrToPrecondSetup       precond_setup;
   hypre_KrylovPtrToPrecond            precond;
   hypre_KrylovPtrToPrecondT           precondT;

} hypre_CGNRFunctions;

/**
 * The {\tt hypre\_CGNRData} object ...
 **/

typedef struct
{
   HYPRE_Real   tol;
   HYPRE_Real   rel_residual_norm;
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      stop_crit;

   void    *A;
   void    *p;
   void    *q;
   void    *r;
   void    *t;

   void    *matvec_data;
   void    *precond_data;

   hypre_CGNRFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int      logging;
   HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_CGNRData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic CGNR Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
hypre_CGNRFunctions *
hypre_CGNRFunctionsCreate(
   hypre_KrylovPtrToCommInfo           CommInfo,
   hypre_KrylovPtrToCreateVector       CreateVector,
   hypre_KrylovPtrToDestroyVector      DestroyVector,
   hypre_KrylovPtrToMatvecCreate       MatvecCreate,
   hypre_KrylovPtrToMatvec             Matvec,
   hypre_KrylovPtrToMatvecT            MatvecT,
   hypre_KrylovPtrToMatvecDestroy      MatvecDestroy,
   hypre_KrylovPtrToInnerProd          InnerProd,
   hypre_KrylovPtrToCopyVector         CopyVector,
   hypre_KrylovPtrToClearVector        ClearVector,
   hypre_KrylovPtrToScaleVector        ScaleVector,
   hypre_KrylovPtrToAxpy               Axpy,
   hypre_KrylovPtrToPrecondSetup       PrecondSetup,
   hypre_KrylovPtrToPrecond            Precond,
   hypre_KrylovPtrToPrecondT           PrecondT
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
