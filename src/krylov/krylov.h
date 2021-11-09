
#include "HYPRE_krylov.h"

#ifndef hypre_KRYLOV_HEADER
#define hypre_KRYLOV_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"

#define hypre_CTAllocF(type, count, funcs) ( (type *)(*(funcs->CAlloc))((size_t)(count), (size_t)sizeof(type)) )

#define hypre_TFreeF( ptr, funcs ) ( (*(funcs->Free))((char *)ptr), ptr = NULL )

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

/* functions in pcg_struct.c which aren't used here:
char *hypre_ParKrylovCAlloc( HYPRE_Int count , HYPRE_Int elt_size );
HYPRE_Int hypre_ParKrylovFree( char *ptr );
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
      HYPRE_Int  (*PrecondSetup)  (void *vdata, void *A, void *b, void *x ),
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
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
   hypre_LGMRESCreate( hypre_LGMRESFunctions *lgmres_functions );

#ifdef __cplusplus
}
#endif
#endif
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                  If that is ever changed, it still must be kept if logging>1 */

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

/* bicgstab.c */

/* cgnr.c */

/* gmres.c */

/* flexgmres.c */

/* lgmres.c */

/* HYPRE_bicgstab.c */

/* HYPRE_cgnr.c */

/* HYPRE_gmres.c */

/* HYPRE_flexgmres.c */

/* HYPRE_lgmres.c */

/* HYPRE_pcg.c */

/* pcg.c */

#ifdef __cplusplus
}
#endif

#endif

