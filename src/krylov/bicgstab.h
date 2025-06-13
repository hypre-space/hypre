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
   void *     (*CreateVector)  ( void *vvector );
   HYPRE_Int  (*DestroyVector) ( void *vvector );
   void *     (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int  (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                 void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int  (*MatvecDestroy) ( void *matvec_data );
   HYPRE_Real (*InnerProd)     ( void *x, void *y );
   HYPRE_Int  (*CopyVector)    ( void *x, void *y );
   HYPRE_Int  (*ClearVector)   ( void *x );
   HYPRE_Int  (*ScaleVector)   ( HYPRE_Complex alpha, void *x );
   HYPRE_Int  (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y );
   HYPRE_Int  (*CommInfo)      ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );
   HYPRE_Int  (*precond_setup) (void *vdata, void *A, void *b, void *x);
   HYPRE_Int  (*precond)       (void *vdata, void *A, void *b, void *x);

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
   void *     (*CreateVector)  ( void *vvector ),
   HYPRE_Int  (*DestroyVector) ( void *vvector ),
   void *     (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int  (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                 void *x, HYPRE_Complex beta, void *y ),
   HYPRE_Int  (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int  (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int  (*ClearVector)   ( void *x ),
   HYPRE_Int  (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int  (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   HYPRE_Int  (*CommInfo)      ( void *A, HYPRE_Int *my_id,
                                 HYPRE_Int *num_procs ),
   HYPRE_Int  (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int  (*Precond)       ( void *vdata, void *A, void *b, void *x )
);

/**
 * Description...
 *
 * @param param [IN] ...
 **/

void *
hypre_BiCGSTABCreate( hypre_BiCGSTABFunctions * bicgstab_functions );

#ifdef __cplusplus
}
#endif

#endif
