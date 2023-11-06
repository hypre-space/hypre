/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "HYPRE_krylov.h"

#ifndef hypre_KRYLOV_HEADER
#define hypre_KRYLOV_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"

#define hypre_CTAllocF(type, count, funcs, location) \
  ( (type *)(*(funcs->CAlloc))((size_t)(count), (size_t)sizeof(type), location) )

#define hypre_TFreeF( ptr, funcs ) ( (*(funcs->Free))((void *)ptr), ptr = NULL )

#ifdef __cplusplus
extern "C" {
#endif

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

/* functions in pcg_struct.c which aren't used here:
   void *hypre_ParKrylovCAlloc( HYPRE_Int count , HYPRE_Int elt_size );
   HYPRE_Int hypre_ParKrylovFree( void *ptr );
   void *hypre_ParKrylovCreateVectorArray( HYPRE_Int n , void *vvector );
   HYPRE_Int hypre_ParKrylovMatvecT( void *matvec_data , HYPRE_Real alpha , void *A , void *x , HYPRE_Real beta , void *y );
   */
/* functions in pcg_struct.c which are used here:
   void *hypre_ParKrylovCreateVector( void *vvector );
   HYPRE_Int hypre_ParKrylovDestroyVector( void *vvector );
   void *hypre_ParKrylovMatvecCreate( void *A , void *x );
   HYPRE_Int hypre_ParKrylovMatvec( void *matvec_data , HYPRE_Real alpha , void *A , void *x , HYPRE_Real beta , void *y );
   HYPRE_Int hypre_ParKrylovMatvecDestroy( void *matvec_data );
   HYPRE_Real hypre_ParKrylovInnerProd( void *x , void *y );
   HYPRE_Int hypre_ParKrylovCopyVector( void *x , void *y );
   HYPRE_Int hypre_ParKrylovClearVector( void *x );
   HYPRE_Int hypre_ParKrylovScaleVector( HYPRE_Real alpha , void *x );
   HYPRE_Int hypre_ParKrylovAxpy( HYPRE_Real alpha , void *x , void *y );
   HYPRE_Int hypre_ParKrylovCommInfo( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs );
   HYPRE_Int hypre_ParKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
   HYPRE_Int hypre_ParKrylovIdentity( void *vdata , void *A , void *b , void *x );
   */

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
   HYPRE_Real   tol;
   HYPRE_Real   cf_tol;
   HYPRE_Real   rel_residual_norm;
   HYPRE_Real   a_tol;


   void  *A;
   void  *r;
   void  *r0;
   void  *s;
   void  *v;
   void  *p;
   void  *q;

   void  *matvec_data;
   void    *precond_data;

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
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   HYPRE_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int    (*MatvecT)       ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   HYPRE_Real   (*InnerProd)     ( void *x, void *y );
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x );
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y );
   HYPRE_Int    (*precond_setup) ( void *vdata, void *A, void *b, void *x );
   HYPRE_Int    (*precond)       ( void *vdata, void *A, void *b, void *x );
   HYPRE_Int    (*precondT)      ( void *vdata, void *A, void *b, void *x );

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
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *       (*CreateVector)  ( void *vector ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *       (*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                      void *x, HYPRE_Complex beta, void *y ),
      HYPRE_Int    (*MatvecT)       ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                      void *x, HYPRE_Complex beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*PrecondT)      ( void *vdata, void *A, void *b, void *x )
   );

   /**
    * Description...
    *
    * @param param [IN] ...
    **/

   void *
   hypre_CGNRCreate( hypre_CGNRFunctions *cgnr_functions );

#ifdef __cplusplus
}
#endif

#endif

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
   void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
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

   HYPRE_Int    (*precond)       (void *vdata, void *A, void *b, void *x);
   HYPRE_Int    (*precond_setup) (void *vdata, void *A, void *b, void *x);

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

#define hypre_GMRESDataHybrid(pcgdata)  ((pcgdata) -> hybrid)

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
      void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
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

/***********KS code ****************/
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
 * hypre_COGMRESData and hypre_COGMRESFunctions
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
   void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
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
   HYPRE_Int    (*MassInnerProd) ( void *x, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result);
   HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, HYPRE_Int k, HYPRE_Int unroll,
                                   void *result_x, void *result_y);
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x );
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y );
   HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex * alpha, void **x, void *y, HYPRE_Int k,
                                   HYPRE_Int unroll);
   HYPRE_Int    (*precond)       (void *vdata, void *A, void *b, void *x);
   HYPRE_Int    (*precond_setup) (void *vdata, void *A, void *b, void *x);

   HYPRE_Int    (*modify_pc)( void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm);


} hypre_COGMRESFunctions;

/**
 * The {\tt hypre\_GMRESData} object ...
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

   hypre_COGMRESFunctions *
   hypre_COGMRESFunctionsCreate(
      void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *       (*CreateVector)  ( void *vector ),
      void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *       (*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A, void *x,
                                      HYPRE_Complex beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*MassInnerProd) ( void *x, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result),
      HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, HYPRE_Int k, HYPRE_Int unroll,
                                      void *result_x, void *result_y),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
      HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex *alpha, void **x, void *y, HYPRE_Int k,
                                      HYPRE_Int unroll),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );

   /**
    * Description...
    *
    * @param param [IN] ...
    **/

   void *
   hypre_COGMRESCreate( hypre_COGMRESFunctions *gmres_functions );

#ifdef __cplusplus
}
#endif
#endif



/***********end of KS code *********/



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
   void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
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

   HYPRE_Int    (*precond)       (void *vdata, void *A, void *b, void *x);
   HYPRE_Int    (*precond_setup) (void *vdata, void *A, void *b, void *x);

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
      void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location),
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
   hypre_LGMRESCreate( hypre_LGMRESFunctions *lgmres_functions );

#ifdef __cplusplus
}
#endif
#endif

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
   void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
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

   HYPRE_Int    (*modify_pc)( void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm);

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
      void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
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
   void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
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

   HYPRE_Int    (*precond)(void *vdata, void *A, void *b, void *x);
   HYPRE_Int    (*precond_setup)(void *vdata, void *A, void *b, void *x);

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
   */

typedef struct
{
   HYPRE_Real   tol;
   HYPRE_Real   atolf;
   HYPRE_Real   cf_tol;
   HYPRE_Real   a_tol;
   HYPRE_Real   rtol;
   HYPRE_Int      max_iter;
   HYPRE_Int      two_norm;
   HYPRE_Int      rel_change;
   HYPRE_Int      recompute_residual;
   HYPRE_Int      recompute_residual_p;
   HYPRE_Int      stop_crit;
   HYPRE_Int      converged;
   HYPRE_Int      hybrid;
   HYPRE_Int      skip_break;
   HYPRE_Int      flex;

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                   If that is ever changed, it still must be kept if logging>1 */
   void    *r_old; /* only needed for flexible CG */
   void    *v; /* work vector; only needed if recompute_residual_p is set */

   HYPRE_Int      owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void    *matvec_data;
   void    *precond_data;

   hypre_PCGFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;
   HYPRE_Real   rel_residual_norm;

   HYPRE_Int     print_level; /* printing when print_level>0 */
   HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   HYPRE_Real  *norms;
   HYPRE_Real  *rel_norms;

} hypre_PCGData;

#define hypre_PCGDataOwnsMatvecData(pcgdata)  ((pcgdata) -> owns_matvec_data)
#define hypre_PCGDataHybrid(pcgdata)  ((pcgdata) -> hybrid)

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
      void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
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

/* bicgstab.c */
void *hypre_BiCGSTABCreate ( hypre_BiCGSTABFunctions *bicgstab_functions );
HYPRE_Int hypre_BiCGSTABDestroy ( void *bicgstab_vdata );
HYPRE_Int hypre_BiCGSTABSetup ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSolve ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSetTol ( void *bicgstab_vdata, HYPRE_Real tol );
HYPRE_Int hypre_BiCGSTABSetAbsoluteTol ( void *bicgstab_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_BiCGSTABSetConvergenceFactorTol ( void *bicgstab_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_BiCGSTABSetMinIter ( void *bicgstab_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_BiCGSTABSetMaxIter ( void *bicgstab_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_BiCGSTABSetStopCrit ( void *bicgstab_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_BiCGSTABSetPrecond ( void *bicgstab_vdata, HYPRE_Int (*precond )(void*, void*,
                                                                                 void*,
                                                                                 void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_BiCGSTABGetPrecond ( void *bicgstab_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_BiCGSTABSetLogging ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetHybrid ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetPrintLevel ( void *bicgstab_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_BiCGSTABGetConverged ( void *bicgstab_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_BiCGSTABGetNumIterations ( void *bicgstab_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BiCGSTABGetFinalRelativeResidualNorm ( void *bicgstab_vdata,
                                                       HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_BiCGSTABGetResidual ( void *bicgstab_vdata, void **residual );

/* cgnr.c */
void *hypre_CGNRCreate ( hypre_CGNRFunctions *cgnr_functions );
HYPRE_Int hypre_CGNRDestroy ( void *cgnr_vdata );
HYPRE_Int hypre_CGNRSetup ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSolve ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSetTol ( void *cgnr_vdata, HYPRE_Real tol );
HYPRE_Int hypre_CGNRSetMinIter ( void *cgnr_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_CGNRSetMaxIter ( void *cgnr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_CGNRSetStopCrit ( void *cgnr_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_CGNRSetPrecond ( void *cgnr_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                         void*),
                                 HYPRE_Int (*precondT )(void*, void*, void*, void*), HYPRE_Int (*precond_setup )(void*, void*, void*,
                                       void*), void *precond_data );
HYPRE_Int hypre_CGNRGetPrecond ( void *cgnr_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_CGNRSetLogging ( void *cgnr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_CGNRGetNumIterations ( void *cgnr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_CGNRGetFinalRelativeResidualNorm ( void *cgnr_vdata,
                                                   HYPRE_Real *relative_residual_norm );

/* gmres.c */
void *hypre_GMRESCreate ( hypre_GMRESFunctions *gmres_functions );
HYPRE_Int hypre_GMRESDestroy ( void *gmres_vdata );
HYPRE_Int hypre_GMRESGetResidual ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_GMRESSetup ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSolve ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSetKDim ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_GMRESGetKDim ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_GMRESSetTol ( void *gmres_vdata, HYPRE_Real tol );
HYPRE_Int hypre_GMRESGetTol ( void *gmres_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_GMRESSetAbsoluteTol ( void *gmres_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_GMRESGetAbsoluteTol ( void *gmres_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_GMRESSetConvergenceFactorTol ( void *gmres_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_GMRESGetConvergenceFactorTol ( void *gmres_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_GMRESSetMinIter ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_GMRESGetMinIter ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_GMRESSetMaxIter ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_GMRESGetMaxIter ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_GMRESSetRelChange ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_GMRESGetRelChange ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_GMRESSetSkipRealResidualCheck ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_GMRESGetSkipRealResidualCheck ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_GMRESSetStopCrit ( void *gmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_GMRESGetStopCrit ( void *gmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_GMRESSetPrecond ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                           void*),
                                  HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_GMRESGetPrecond ( void *gmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_GMRESSetPrintLevel ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESGetPrintLevel ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESSetLogging ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESGetLogging ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESSetHybrid ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESGetNumIterations ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_GMRESGetConverged ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_GMRESGetFinalRelativeResidualNorm ( void *gmres_vdata,
                                                    HYPRE_Real *relative_residual_norm );

/* cogmres.c */
void *hypre_COGMRESCreate ( hypre_COGMRESFunctions *gmres_functions );
HYPRE_Int hypre_COGMRESDestroy ( void *gmres_vdata );
HYPRE_Int hypre_COGMRESGetResidual ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_COGMRESSetup ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSolve ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSetKDim ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_COGMRESGetKDim ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_COGMRESSetUnroll ( void *gmres_vdata, HYPRE_Int unroll );
HYPRE_Int hypre_COGMRESGetUnroll ( void *gmres_vdata, HYPRE_Int *unroll );
HYPRE_Int hypre_COGMRESSetCGS ( void *gmres_vdata, HYPRE_Int cgs );
HYPRE_Int hypre_COGMRESGetCGS ( void *gmres_vdata, HYPRE_Int *cgs );
HYPRE_Int hypre_COGMRESSetTol ( void *gmres_vdata, HYPRE_Real tol );
HYPRE_Int hypre_COGMRESGetTol ( void *gmres_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_COGMRESSetAbsoluteTol ( void *gmres_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_COGMRESGetAbsoluteTol ( void *gmres_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_COGMRESSetConvergenceFactorTol ( void *gmres_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_COGMRESGetConvergenceFactorTol ( void *gmres_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_COGMRESSetMinIter ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_COGMRESGetMinIter ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_COGMRESSetMaxIter ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_COGMRESGetMaxIter ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_COGMRESSetRelChange ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_COGMRESGetRelChange ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_COGMRESSetSkipRealResidualCheck ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_COGMRESGetSkipRealResidualCheck ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_COGMRESSetPrecond ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_COGMRESGetPrecond ( void *gmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_COGMRESSetPrintLevel ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESGetPrintLevel ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESSetLogging ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESGetLogging ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetNumIterations ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_COGMRESGetConverged ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_COGMRESGetFinalRelativeResidualNorm ( void *gmres_vdata,
                                                      HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_COGMRESSetModifyPC ( void *fgmres_vdata, HYPRE_Int (*modify_pc )(void *precond_data,
                                                                                 HYPRE_Int iteration, HYPRE_Real rel_residual_norm));



/* flexgmres.c */
void *hypre_FlexGMRESCreate ( hypre_FlexGMRESFunctions *fgmres_functions );
HYPRE_Int hypre_FlexGMRESDestroy ( void *fgmres_vdata );
HYPRE_Int hypre_FlexGMRESGetResidual ( void *fgmres_vdata, void **residual );
HYPRE_Int hypre_FlexGMRESSetup ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSolve ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSetKDim ( void *fgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_FlexGMRESGetKDim ( void *fgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_FlexGMRESSetTol ( void *fgmres_vdata, HYPRE_Real tol );
HYPRE_Int hypre_FlexGMRESGetTol ( void *fgmres_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_FlexGMRESSetAbsoluteTol ( void *fgmres_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_FlexGMRESGetAbsoluteTol ( void *fgmres_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_FlexGMRESSetConvergenceFactorTol ( void *fgmres_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_FlexGMRESGetConvergenceFactorTol ( void *fgmres_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_FlexGMRESSetMinIter ( void *fgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_FlexGMRESGetMinIter ( void *fgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_FlexGMRESSetMaxIter ( void *fgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_FlexGMRESGetMaxIter ( void *fgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_FlexGMRESSetStopCrit ( void *fgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_FlexGMRESGetStopCrit ( void *fgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_FlexGMRESSetPrecond ( void *fgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                                void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_FlexGMRESGetPrecond ( void *fgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_FlexGMRESSetPrintLevel ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESGetPrintLevel ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESSetLogging ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESGetLogging ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetNumIterations ( void *fgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FlexGMRESGetConverged ( void *fgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_FlexGMRESGetFinalRelativeResidualNorm ( void *fgmres_vdata,
                                                        HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_FlexGMRESSetModifyPC ( void *fgmres_vdata,
                                       HYPRE_Int (*modify_pc )(void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm));
HYPRE_Int hypre_FlexGMRESModifyPCDefault ( void *precond_data, HYPRE_Int iteration,
                                           HYPRE_Real rel_residual_norm );

/* lgmres.c */
void *hypre_LGMRESCreate ( hypre_LGMRESFunctions *lgmres_functions );
HYPRE_Int hypre_LGMRESDestroy ( void *lgmres_vdata );
HYPRE_Int hypre_LGMRESGetResidual ( void *lgmres_vdata, void **residual );
HYPRE_Int hypre_LGMRESSetup ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSolve ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSetKDim ( void *lgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_LGMRESGetKDim ( void *lgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_LGMRESSetAugDim ( void *lgmres_vdata, HYPRE_Int aug_dim );
HYPRE_Int hypre_LGMRESGetAugDim ( void *lgmres_vdata, HYPRE_Int *aug_dim );
HYPRE_Int hypre_LGMRESSetTol ( void *lgmres_vdata, HYPRE_Real tol );
HYPRE_Int hypre_LGMRESGetTol ( void *lgmres_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_LGMRESSetAbsoluteTol ( void *lgmres_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_LGMRESGetAbsoluteTol ( void *lgmres_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_LGMRESSetConvergenceFactorTol ( void *lgmres_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_LGMRESGetConvergenceFactorTol ( void *lgmres_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_LGMRESSetMinIter ( void *lgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_LGMRESGetMinIter ( void *lgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_LGMRESSetMaxIter ( void *lgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_LGMRESGetMaxIter ( void *lgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_LGMRESSetStopCrit ( void *lgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_LGMRESGetStopCrit ( void *lgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_LGMRESSetPrecond ( void *lgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_LGMRESGetPrecond ( void *lgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_LGMRESSetPrintLevel ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESGetPrintLevel ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESSetLogging ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESGetLogging ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetNumIterations ( void *lgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_LGMRESGetConverged ( void *lgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_LGMRESGetFinalRelativeResidualNorm ( void *lgmres_vdata,
                                                     HYPRE_Real *relative_residual_norm );

/* HYPRE_bicgstab.c */
HYPRE_Int HYPRE_BiCGSTABDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BiCGSTABSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_BiCGSTABSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_BiCGSTABSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_BiCGSTABSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BiCGSTABSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BiCGSTABSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_BiCGSTABSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                     HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_BiCGSTABGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_BiCGSTABSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BiCGSTABSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BiCGSTABGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BiCGSTABGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_BiCGSTABGetResidual ( HYPRE_Solver solver, void *residual );

/* HYPRE_cgnr.c */
HYPRE_Int HYPRE_CGNRDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_CGNRSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_CGNRSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_CGNRSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_CGNRSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_CGNRSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                 HYPRE_PtrToSolverFcn precondT, HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_CGNRGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_CGNRSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_CGNRGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_CGNRGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );

/* HYPRE_gmres.c */
HYPRE_Int HYPRE_GMRESSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_GMRESGetKDim ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_GMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_GMRESGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_GMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_GMRESGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_GMRESSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_GMRESGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_GMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_GMRESGetMinIter ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_GMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_GMRESGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_GMRESSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_GMRESGetStopCrit ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_GMRESSetRelChange ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_GMRESGetRelChange ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_GMRESSetSkipRealResidualCheck ( HYPRE_Solver solver, HYPRE_Int skip_real_r_check );
HYPRE_Int HYPRE_GMRESGetSkipRealResidualCheck ( HYPRE_Solver solver, HYPRE_Int *skip_real_r_check );
HYPRE_Int HYPRE_GMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                  HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_GMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_GMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_GMRESGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_GMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_GMRESGetResidual ( HYPRE_Solver solver, void *residual );

/* HYPRE_cogmres.c */
HYPRE_Int HYPRE_COGMRESSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_COGMRESGetKDim ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_COGMRESSetUnroll ( HYPRE_Solver solver, HYPRE_Int unroll );
HYPRE_Int HYPRE_COGMRESGetUnroll ( HYPRE_Solver solver, HYPRE_Int *unroll );
HYPRE_Int HYPRE_COGMRESSetCGS ( HYPRE_Solver solver, HYPRE_Int cgs );
HYPRE_Int HYPRE_COGMRESGetCGS ( HYPRE_Solver solver, HYPRE_Int *cgs );
HYPRE_Int HYPRE_COGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_COGMRESGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_COGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_COGMRESGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_COGMRESSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_COGMRESGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_COGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_COGMRESGetMinIter ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_COGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_COGMRESGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_COGMRESSetRelChange ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_COGMRESGetRelChange ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_COGMRESSetSkipRealResidualCheck ( HYPRE_Solver solver,
                                                  HYPRE_Int skip_real_r_check );
HYPRE_Int HYPRE_COGMRESGetSkipRealResidualCheck ( HYPRE_Solver solver,
                                                  HYPRE_Int *skip_real_r_check );
HYPRE_Int HYPRE_COGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                    HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_COGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_COGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_COGMRESGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_COGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_COGMRESGetResidual ( HYPRE_Solver solver, void *residual );

/* HYPRE_flexgmres.c */
HYPRE_Int HYPRE_FlexGMRESSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_FlexGMRESGetKDim ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_FlexGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_FlexGMRESGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_FlexGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_FlexGMRESGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_FlexGMRESSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_FlexGMRESGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_FlexGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_FlexGMRESGetMinIter ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_FlexGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_FlexGMRESGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_FlexGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                      HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_FlexGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_FlexGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_FlexGMRESGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_FlexGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_FlexGMRESGetResidual ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_FlexGMRESSetModifyPC ( HYPRE_Solver solver, HYPRE_Int (*modify_pc )(HYPRE_Solver,
                                                                                    HYPRE_Int, HYPRE_Real ));

/* HYPRE_lgmres.c */
HYPRE_Int HYPRE_LGMRESSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_LGMRESGetKDim ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_LGMRESSetAugDim ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_LGMRESGetAugDim ( HYPRE_Solver solver, HYPRE_Int *aug_dim );
HYPRE_Int HYPRE_LGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_LGMRESGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_LGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_LGMRESGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_LGMRESSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_LGMRESGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_LGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_LGMRESGetMinIter ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_LGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_LGMRESGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_LGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                   HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_LGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_LGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_LGMRESGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_LGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_LGMRESGetResidual ( HYPRE_Solver solver, void *residual );

/* HYPRE_pcg.c */
HYPRE_Int HYPRE_PCGSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_PCGGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_PCGGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTolFactor ( HYPRE_Solver solver, HYPRE_Real abstolf );
HYPRE_Int HYPRE_PCGGetAbsoluteTolFactor ( HYPRE_Solver solver, HYPRE_Real *abstolf );
HYPRE_Int HYPRE_PCGSetResidualTol ( HYPRE_Solver solver, HYPRE_Real rtol );
HYPRE_Int HYPRE_PCGGetResidualTol ( HYPRE_Solver solver, HYPRE_Real *rtol );
HYPRE_Int HYPRE_PCGSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_PCGGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_PCGSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_PCGGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_PCGSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_PCGGetStopCrit ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_PCGSetTwoNorm ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_PCGGetTwoNorm ( HYPRE_Solver solver, HYPRE_Int *two_norm );
HYPRE_Int HYPRE_PCGSetRelChange ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_PCGGetRelChange ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_PCGSetRecomputeResidual ( HYPRE_Solver solver, HYPRE_Int recompute_residual );
HYPRE_Int HYPRE_PCGGetRecomputeResidual ( HYPRE_Solver solver, HYPRE_Int *recompute_residual );
HYPRE_Int HYPRE_PCGSetRecomputeResidualP ( HYPRE_Solver solver, HYPRE_Int recompute_residual_p );
HYPRE_Int HYPRE_PCGGetRecomputeResidualP ( HYPRE_Solver solver, HYPRE_Int *recompute_residual_p );
HYPRE_Int HYPRE_PCGSetSkipBreak ( HYPRE_Solver solver, HYPRE_Int skip_break );
HYPRE_Int HYPRE_PCGGetSkipBreak ( HYPRE_Solver solver, HYPRE_Int *skip_break );
HYPRE_Int HYPRE_PCGSetFlex ( HYPRE_Solver solver, HYPRE_Int flex );
HYPRE_Int HYPRE_PCGGetFlex ( HYPRE_Solver solver, HYPRE_Int *flex );
HYPRE_Int HYPRE_PCGSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_PCGSetPreconditioner ( HYPRE_Solver solver, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_PCGGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_PCGSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_PCGGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_PCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_PCGGetResidual ( HYPRE_Solver solver, void *residual );

/* pcg.c */
void *hypre_PCGCreate ( hypre_PCGFunctions *pcg_functions );
HYPRE_Int hypre_PCGDestroy ( void *pcg_vdata );
HYPRE_Int hypre_PCGGetResidual ( void *pcg_vdata, void **residual );
HYPRE_Int hypre_PCGSetup ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSolve ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSetTol ( void *pcg_vdata, HYPRE_Real tol );
HYPRE_Int hypre_PCGGetTol ( void *pcg_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_PCGSetAbsoluteTol ( void *pcg_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_PCGGetAbsoluteTol ( void *pcg_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_PCGSetAbsoluteTolFactor ( void *pcg_vdata, HYPRE_Real atolf );
HYPRE_Int hypre_PCGGetAbsoluteTolFactor ( void *pcg_vdata, HYPRE_Real *atolf );
HYPRE_Int hypre_PCGSetResidualTol ( void *pcg_vdata, HYPRE_Real rtol );
HYPRE_Int hypre_PCGGetResidualTol ( void *pcg_vdata, HYPRE_Real *rtol );
HYPRE_Int hypre_PCGSetConvergenceFactorTol ( void *pcg_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_PCGGetConvergenceFactorTol ( void *pcg_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_PCGSetMaxIter ( void *pcg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PCGGetMaxIter ( void *pcg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PCGSetTwoNorm ( void *pcg_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_PCGGetTwoNorm ( void *pcg_vdata, HYPRE_Int *two_norm );
HYPRE_Int hypre_PCGSetRelChange ( void *pcg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PCGGetRelChange ( void *pcg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PCGSetRecomputeResidual ( void *pcg_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_PCGGetRecomputeResidual ( void *pcg_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_PCGSetRecomputeResidualP ( void *pcg_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_PCGGetRecomputeResidualP ( void *pcg_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_PCGSetStopCrit ( void *pcg_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_PCGGetStopCrit ( void *pcg_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_PCGSetSkipBreak ( void *pcg_vdata, HYPRE_Int skip_break );
HYPRE_Int hypre_PCGGetSkipBreak ( void *pcg_vdata, HYPRE_Int *skip_break );
HYPRE_Int hypre_PCGSetFlex ( void *pcg_vdata, HYPRE_Int flex );
HYPRE_Int hypre_PCGGetFlex ( void *pcg_vdata, HYPRE_Int *flex );
HYPRE_Int hypre_PCGGetPrecond ( void *pcg_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_PCGSetPrecond ( void *pcg_vdata,
                                HYPRE_Int (*precond )(void*, void*, void*, void*),
                                HYPRE_Int (*precond_setup )(void*, void*, void*, void*),
                                void *precond_data );
HYPRE_Int hypre_PCGSetPreconditioner ( void *pcg_vdata, void *precond_data );
HYPRE_Int hypre_PCGSetPrintLevel ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGGetPrintLevel ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGSetLogging ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGGetLogging ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGSetHybrid ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGGetNumIterations ( void *pcg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PCGGetConverged ( void *pcg_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_PCGPrintLogging ( void *pcg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PCGGetFinalRelativeResidualNorm ( void *pcg_vdata,
                                                  HYPRE_Real *relative_residual_norm );

#ifdef __cplusplus
}
#endif

#endif
