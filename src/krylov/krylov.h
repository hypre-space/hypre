/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.40 $
 ***********************************************************************EHEADER*/

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

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.40 $
 ***********************************************************************EHEADER*/




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
HYPRE_Int hypre_ParKrylovMatvecT( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
*/
/* functions in pcg_struct.c which are used here:
  void *hypre_ParKrylovCreateVector( void *vvector );
  HYPRE_Int hypre_ParKrylovDestroyVector( void *vvector );
  void *hypre_ParKrylovMatvecCreate( void *A , void *x );
  HYPRE_Int hypre_ParKrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
  HYPRE_Int hypre_ParKrylovMatvecDestroy( void *matvec_data );
  double hypre_ParKrylovInnerProd( void *x , void *y );
  HYPRE_Int hypre_ParKrylovCopyVector( void *x , void *y );
  HYPRE_Int hypre_ParKrylovClearVector( void *x );
  HYPRE_Int hypre_ParKrylovScaleVector( double alpha , void *x );
  HYPRE_Int hypre_ParKrylovAxpy( double alpha , void *x , void *y );
  HYPRE_Int hypre_ParKrylovCommInfo( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs );
  HYPRE_Int hypre_ParKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
  HYPRE_Int hypre_ParKrylovIdentity( void *vdata , void *A , void *b , void *x );
*/

typedef struct
{
  void *(*CreateVector)( void *vvector );
  HYPRE_Int (*DestroyVector)( void *vvector );
  void *(*MatvecCreate)( void *A , void *x );
  HYPRE_Int (*Matvec)( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
  HYPRE_Int (*MatvecDestroy)( void *matvec_data );
  double (*InnerProd)( void *x , void *y );
  HYPRE_Int (*CopyVector)( void *x , void *y );
  HYPRE_Int (*ClearVector)( void *x );
  HYPRE_Int (*ScaleVector)( double alpha , void *x );
  HYPRE_Int (*Axpy)( double alpha , void *x , void *y );
  HYPRE_Int (*CommInfo)( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs );
  HYPRE_Int (*precond_setup)();
  HYPRE_Int (*precond)();

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
   double   tol;
   double   cf_tol;
   double   rel_residual_norm;
   double   a_tol;
   

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
   double  *norms;
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
      void *(*CreateVector)( void *vvector ),
      HYPRE_Int (*DestroyVector)( void *vvector ),
      void *(*MatvecCreate)( void *A , void *x ),
      HYPRE_Int (*Matvec)( void *matvec_data , double alpha , void *A , void *x , double beta , void *y ),
      HYPRE_Int (*MatvecDestroy)( void *matvec_data ),
      double (*InnerProd)( void *x , void *y ),
      HYPRE_Int (*CopyVector)( void *x , void *y ),
      HYPRE_Int (*ClearVector)( void *x ),
      HYPRE_Int (*ScaleVector)( double alpha , void *x ),
      HYPRE_Int (*Axpy)( double alpha , void *x , void *y ),
      HYPRE_Int (*CommInfo)( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs ),
      HYPRE_Int (*PrecondSetup) (void *vdata, void *A, void *b, void *x ),
      HYPRE_Int (*Precond)  ( void *vdata, void *A, void *b, void *x )
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
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.40 $
 ***********************************************************************EHEADER*/





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
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   HYPRE_Int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   HYPRE_Int    (*MatvecT)       ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x );
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y );
   HYPRE_Int    (*precond_setup) ( void *vdata, void *A, void *b, void *x );
   HYPRE_Int    (*precond)       ( void *vdata, void *A, void *b, void *x );
   HYPRE_Int    (*precondT)       ( void *vdata, void *A, void *b, void *x );
} hypre_CGNRFunctions;

/**
 * The {\tt hypre\_CGNRData} object ...
 **/

typedef struct
{
   double   tol;
   double   rel_residual_norm;
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
   double  *norms;
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
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecT)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y ),
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*PrecondT)       ( void *vdata, void *A, void *b, void *x )
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
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.40 $
 ***********************************************************************EHEADER*/




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
   char * (*CAlloc)        ( size_t count, size_t elt_size );
   HYPRE_Int    (*Free)          ( char *ptr );
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   void * (*CreateVectorArray)  ( HYPRE_Int size, void *vectors );
   HYPRE_Int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x );
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y );

   HYPRE_Int    (*precond)();
   HYPRE_Int    (*precond_setup)();

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
   double   tol;
   double   cf_tol;
   double   a_tol;
   double   rel_residual_norm;

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
   double  *norms;
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
   char * (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( char *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y ),
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
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.40 $
 ***********************************************************************EHEADER*/




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
   char * (*CAlloc)        ( size_t count, size_t elt_size );
   HYPRE_Int    (*Free)          ( char *ptr );
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   void * (*CreateVectorArray)  ( HYPRE_Int size, void *vectors );
   HYPRE_Int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x );
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y );

   HYPRE_Int    (*precond)();
   HYPRE_Int    (*precond_setup)();

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
   double   tol;
   double   cf_tol;
   double   a_tol;
   double   rel_residual_norm;

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
   double  *norms;
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
   char * (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( char *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y ),
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
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.40 $
 ***********************************************************************EHEADER*/





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
   char * (*CAlloc)        ( size_t count, size_t elt_size );
   HYPRE_Int    (*Free)          ( char *ptr );
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   void * (*CreateVectorArray)  ( HYPRE_Int size, void *vectors );
   HYPRE_Int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x );
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y );

   HYPRE_Int    (*precond)();
   HYPRE_Int    (*precond_setup)();

   HYPRE_Int    (*modify_pc)();
   

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
   double   tol;
   double   cf_tol;
   double   a_tol;
   double   rel_residual_norm;

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
   double  *norms;
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
   char * (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( char *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y ),
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
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.40 $
 ***********************************************************************EHEADER*/





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
   char * (*CAlloc)        ( size_t count, size_t elt_size );
   HYPRE_Int    (*Free)          ( char *ptr );
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   HYPRE_Int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x );
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y );

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
   double   tol;
   double   atolf;
   double   cf_tol;
   double   a_tol;
   double   rtol;
   HYPRE_Int      max_iter;
   HYPRE_Int      two_norm;
   HYPRE_Int      rel_change;
   HYPRE_Int      recompute_residual;
   HYPRE_Int      recompute_residual_p;
   HYPRE_Int      stop_crit;
   HYPRE_Int      converged;

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                  If that is ever changed, it still must be kept if logging>1 */

   HYPRE_Int      owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void    *matvec_data;
   void    *precond_data;

   hypre_PCGFunctions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;
   double   rel_residual_norm;

   HYPRE_Int     print_level; /* printing when print_level>0 */
   HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   double  *norms;
   double  *rel_norms;

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
   char * (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( char *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y ),
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
hypre_BiCGSTABFunctions *hypre_BiCGSTABFunctionsCreate ( void *(*CreateVector )(void *vvector ), HYPRE_Int (*DestroyVector )(void *vvector ), void *(*MatvecCreate )(void *A ,void *x ), HYPRE_Int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), HYPRE_Int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), HYPRE_Int (*CopyVector )(void *x ,void *y ), HYPRE_Int (*ClearVector )(void *x ), HYPRE_Int (*ScaleVector )(double alpha ,void *x ), HYPRE_Int (*Axpy )(double alpha ,void *x ,void *y ), HYPRE_Int (*CommInfo )(void *A ,HYPRE_Int *my_id ,HYPRE_Int *num_procs ), HYPRE_Int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), HYPRE_Int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_BiCGSTABCreate ( hypre_BiCGSTABFunctions *bicgstab_functions );
HYPRE_Int hypre_BiCGSTABDestroy ( void *bicgstab_vdata );
HYPRE_Int hypre_BiCGSTABSetup ( void *bicgstab_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_BiCGSTABSolve ( void *bicgstab_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_BiCGSTABSetTol ( void *bicgstab_vdata , double tol );
HYPRE_Int hypre_BiCGSTABSetAbsoluteTol ( void *bicgstab_vdata , double a_tol );
HYPRE_Int hypre_BiCGSTABSetConvergenceFactorTol ( void *bicgstab_vdata , double cf_tol );
HYPRE_Int hypre_BiCGSTABSetMinIter ( void *bicgstab_vdata , HYPRE_Int min_iter );
HYPRE_Int hypre_BiCGSTABSetMaxIter ( void *bicgstab_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_BiCGSTABSetStopCrit ( void *bicgstab_vdata , HYPRE_Int stop_crit );
HYPRE_Int hypre_BiCGSTABSetPrecond ( void *bicgstab_vdata , HYPRE_Int (*precond )(), HYPRE_Int (*precond_setup )(), void *precond_data );
HYPRE_Int hypre_BiCGSTABGetPrecond ( void *bicgstab_vdata , HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_BiCGSTABSetLogging ( void *bicgstab_vdata , HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetPrintLevel ( void *bicgstab_vdata , HYPRE_Int print_level );
HYPRE_Int hypre_BiCGSTABGetConverged ( void *bicgstab_vdata , HYPRE_Int *converged );
HYPRE_Int hypre_BiCGSTABGetNumIterations ( void *bicgstab_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_BiCGSTABGetFinalRelativeResidualNorm ( void *bicgstab_vdata , double *relative_residual_norm );
HYPRE_Int hypre_BiCGSTABGetResidual ( void *bicgstab_vdata , void **residual );

/* cgnr.c */
hypre_CGNRFunctions *hypre_CGNRFunctionsCreate ( HYPRE_Int (*CommInfo )(void *A ,HYPRE_Int *my_id ,HYPRE_Int *num_procs ), void *(*CreateVector )(void *vector ), HYPRE_Int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), HYPRE_Int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), HYPRE_Int (*MatvecT )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), HYPRE_Int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), HYPRE_Int (*CopyVector )(void *x ,void *y ), HYPRE_Int (*ClearVector )(void *x ), HYPRE_Int (*ScaleVector )(double alpha ,void *x ), HYPRE_Int (*Axpy )(double alpha ,void *x ,void *y ), HYPRE_Int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), HYPRE_Int (*Precond )(void *vdata ,void *A ,void *b ,void *x ), HYPRE_Int (*PrecondT )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_CGNRCreate ( hypre_CGNRFunctions *cgnr_functions );
HYPRE_Int hypre_CGNRDestroy ( void *cgnr_vdata );
HYPRE_Int hypre_CGNRSetup ( void *cgnr_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_CGNRSolve ( void *cgnr_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_CGNRSetTol ( void *cgnr_vdata , double tol );
HYPRE_Int hypre_CGNRSetMinIter ( void *cgnr_vdata , HYPRE_Int min_iter );
HYPRE_Int hypre_CGNRSetMaxIter ( void *cgnr_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_CGNRSetStopCrit ( void *cgnr_vdata , HYPRE_Int stop_crit );
HYPRE_Int hypre_CGNRSetPrecond ( void *cgnr_vdata , HYPRE_Int (*precond )(), HYPRE_Int (*precondT )(), HYPRE_Int (*precond_setup )(), void *precond_data );
HYPRE_Int hypre_CGNRGetPrecond ( void *cgnr_vdata , HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_CGNRSetLogging ( void *cgnr_vdata , HYPRE_Int logging );
HYPRE_Int hypre_CGNRGetNumIterations ( void *cgnr_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_CGNRGetFinalRelativeResidualNorm ( void *cgnr_vdata , double *relative_residual_norm );

/* gmres.c */
hypre_GMRESFunctions *hypre_GMRESFunctionsCreate ( char *(*CAlloc )(size_t count ,size_t elt_size ), HYPRE_Int (*Free )(char *ptr ), HYPRE_Int (*CommInfo )(void *A ,HYPRE_Int *my_id ,HYPRE_Int *num_procs ), void *(*CreateVector )(void *vector ), void *(*CreateVectorArray )(HYPRE_Int size ,void *vectors ), HYPRE_Int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), HYPRE_Int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), HYPRE_Int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), HYPRE_Int (*CopyVector )(void *x ,void *y ), HYPRE_Int (*ClearVector )(void *x ), HYPRE_Int (*ScaleVector )(double alpha ,void *x ), HYPRE_Int (*Axpy )(double alpha ,void *x ,void *y ), HYPRE_Int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), HYPRE_Int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_GMRESCreate ( hypre_GMRESFunctions *gmres_functions );
HYPRE_Int hypre_GMRESDestroy ( void *gmres_vdata );
HYPRE_Int hypre_GMRESGetResidual ( void *gmres_vdata , void **residual );
HYPRE_Int hypre_GMRESSetup ( void *gmres_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_GMRESSolve ( void *gmres_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_GMRESSetKDim ( void *gmres_vdata , HYPRE_Int k_dim );
HYPRE_Int hypre_GMRESGetKDim ( void *gmres_vdata , HYPRE_Int *k_dim );
HYPRE_Int hypre_GMRESSetTol ( void *gmres_vdata , double tol );
HYPRE_Int hypre_GMRESGetTol ( void *gmres_vdata , double *tol );
HYPRE_Int hypre_GMRESSetAbsoluteTol ( void *gmres_vdata , double a_tol );
HYPRE_Int hypre_GMRESGetAbsoluteTol ( void *gmres_vdata , double *a_tol );
HYPRE_Int hypre_GMRESSetConvergenceFactorTol ( void *gmres_vdata , double cf_tol );
HYPRE_Int hypre_GMRESGetConvergenceFactorTol ( void *gmres_vdata , double *cf_tol );
HYPRE_Int hypre_GMRESSetMinIter ( void *gmres_vdata , HYPRE_Int min_iter );
HYPRE_Int hypre_GMRESGetMinIter ( void *gmres_vdata , HYPRE_Int *min_iter );
HYPRE_Int hypre_GMRESSetMaxIter ( void *gmres_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_GMRESGetMaxIter ( void *gmres_vdata , HYPRE_Int *max_iter );
HYPRE_Int hypre_GMRESSetRelChange ( void *gmres_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_GMRESGetRelChange ( void *gmres_vdata , HYPRE_Int *rel_change );
HYPRE_Int hypre_GMRESSetSkipRealResidualCheck ( void *gmres_vdata , HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_GMRESGetSkipRealResidualCheck ( void *gmres_vdata , HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_GMRESSetStopCrit ( void *gmres_vdata , HYPRE_Int stop_crit );
HYPRE_Int hypre_GMRESGetStopCrit ( void *gmres_vdata , HYPRE_Int *stop_crit );
HYPRE_Int hypre_GMRESSetPrecond ( void *gmres_vdata , HYPRE_Int (*precond )(), HYPRE_Int (*precond_setup )(), void *precond_data );
HYPRE_Int hypre_GMRESGetPrecond ( void *gmres_vdata , HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_GMRESSetPrintLevel ( void *gmres_vdata , HYPRE_Int level );
HYPRE_Int hypre_GMRESGetPrintLevel ( void *gmres_vdata , HYPRE_Int *level );
HYPRE_Int hypre_GMRESSetLogging ( void *gmres_vdata , HYPRE_Int level );
HYPRE_Int hypre_GMRESGetLogging ( void *gmres_vdata , HYPRE_Int *level );
HYPRE_Int hypre_GMRESGetNumIterations ( void *gmres_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_GMRESGetConverged ( void *gmres_vdata , HYPRE_Int *converged );
HYPRE_Int hypre_GMRESGetFinalRelativeResidualNorm ( void *gmres_vdata , double *relative_residual_norm );

/* flexgmres.c */
hypre_FlexGMRESFunctions *hypre_FlexGMRESFunctionsCreate ( char *(*CAlloc )(size_t count ,size_t elt_size ), HYPRE_Int (*Free )(char *ptr ), HYPRE_Int (*CommInfo )(void *A ,HYPRE_Int *my_id ,HYPRE_Int *num_procs ), void *(*CreateVector )(void *vector ), void *(*CreateVectorArray )(HYPRE_Int size ,void *vectors ), HYPRE_Int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), HYPRE_Int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), HYPRE_Int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), HYPRE_Int (*CopyVector )(void *x ,void *y ), HYPRE_Int (*ClearVector )(void *x ), HYPRE_Int (*ScaleVector )(double alpha ,void *x ), HYPRE_Int (*Axpy )(double alpha ,void *x ,void *y ), HYPRE_Int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), HYPRE_Int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_FlexGMRESCreate ( hypre_FlexGMRESFunctions *fgmres_functions );
HYPRE_Int hypre_FlexGMRESDestroy ( void *fgmres_vdata );
HYPRE_Int hypre_FlexGMRESGetResidual ( void *fgmres_vdata , void **residual );
HYPRE_Int hypre_FlexGMRESSetup ( void *fgmres_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_FlexGMRESSolve ( void *fgmres_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_FlexGMRESSetKDim ( void *fgmres_vdata , HYPRE_Int k_dim );
HYPRE_Int hypre_FlexGMRESGetKDim ( void *fgmres_vdata , HYPRE_Int *k_dim );
HYPRE_Int hypre_FlexGMRESSetTol ( void *fgmres_vdata , double tol );
HYPRE_Int hypre_FlexGMRESGetTol ( void *fgmres_vdata , double *tol );
HYPRE_Int hypre_FlexGMRESSetAbsoluteTol ( void *fgmres_vdata , double a_tol );
HYPRE_Int hypre_FlexGMRESGetAbsoluteTol ( void *fgmres_vdata , double *a_tol );
HYPRE_Int hypre_FlexGMRESSetConvergenceFactorTol ( void *fgmres_vdata , double cf_tol );
HYPRE_Int hypre_FlexGMRESGetConvergenceFactorTol ( void *fgmres_vdata , double *cf_tol );
HYPRE_Int hypre_FlexGMRESSetMinIter ( void *fgmres_vdata , HYPRE_Int min_iter );
HYPRE_Int hypre_FlexGMRESGetMinIter ( void *fgmres_vdata , HYPRE_Int *min_iter );
HYPRE_Int hypre_FlexGMRESSetMaxIter ( void *fgmres_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_FlexGMRESGetMaxIter ( void *fgmres_vdata , HYPRE_Int *max_iter );
HYPRE_Int hypre_FlexGMRESSetStopCrit ( void *fgmres_vdata , HYPRE_Int stop_crit );
HYPRE_Int hypre_FlexGMRESGetStopCrit ( void *fgmres_vdata , HYPRE_Int *stop_crit );
HYPRE_Int hypre_FlexGMRESSetPrecond ( void *fgmres_vdata , HYPRE_Int (*precond )(), HYPRE_Int (*precond_setup )(), void *precond_data );
HYPRE_Int hypre_FlexGMRESGetPrecond ( void *fgmres_vdata , HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_FlexGMRESSetPrintLevel ( void *fgmres_vdata , HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESGetPrintLevel ( void *fgmres_vdata , HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESSetLogging ( void *fgmres_vdata , HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESGetLogging ( void *fgmres_vdata , HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetNumIterations ( void *fgmres_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_FlexGMRESGetConverged ( void *fgmres_vdata , HYPRE_Int *converged );
HYPRE_Int hypre_FlexGMRESGetFinalRelativeResidualNorm ( void *fgmres_vdata , double *relative_residual_norm );
HYPRE_Int hypre_FlexGMRESSetModifyPC ( void *fgmres_vdata , HYPRE_Int (*modify_pc )());
HYPRE_Int hypre_FlexGMRESModifyPCDefault ( void *precond_data , HYPRE_Int iteration , double rel_residual_norm );

/* lgmres.c */
hypre_LGMRESFunctions *hypre_LGMRESFunctionsCreate ( char *(*CAlloc )(size_t count ,size_t elt_size ), HYPRE_Int (*Free )(char *ptr ), HYPRE_Int (*CommInfo )(void *A ,HYPRE_Int *my_id ,HYPRE_Int *num_procs ), void *(*CreateVector )(void *vector ), void *(*CreateVectorArray )(HYPRE_Int size ,void *vectors ), HYPRE_Int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), HYPRE_Int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), HYPRE_Int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), HYPRE_Int (*CopyVector )(void *x ,void *y ), HYPRE_Int (*ClearVector )(void *x ), HYPRE_Int (*ScaleVector )(double alpha ,void *x ), HYPRE_Int (*Axpy )(double alpha ,void *x ,void *y ), HYPRE_Int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), HYPRE_Int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_LGMRESCreate ( hypre_LGMRESFunctions *lgmres_functions );
HYPRE_Int hypre_LGMRESDestroy ( void *lgmres_vdata );
HYPRE_Int hypre_LGMRESGetResidual ( void *lgmres_vdata , void **residual );
HYPRE_Int hypre_LGMRESSetup ( void *lgmres_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_LGMRESSolve ( void *lgmres_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_LGMRESSetKDim ( void *lgmres_vdata , HYPRE_Int k_dim );
HYPRE_Int hypre_LGMRESGetKDim ( void *lgmres_vdata , HYPRE_Int *k_dim );
HYPRE_Int hypre_LGMRESSetAugDim ( void *lgmres_vdata , HYPRE_Int aug_dim );
HYPRE_Int hypre_LGMRESGetAugDim ( void *lgmres_vdata , HYPRE_Int *aug_dim );
HYPRE_Int hypre_LGMRESSetTol ( void *lgmres_vdata , double tol );
HYPRE_Int hypre_LGMRESGetTol ( void *lgmres_vdata , double *tol );
HYPRE_Int hypre_LGMRESSetAbsoluteTol ( void *lgmres_vdata , double a_tol );
HYPRE_Int hypre_LGMRESGetAbsoluteTol ( void *lgmres_vdata , double *a_tol );
HYPRE_Int hypre_LGMRESSetConvergenceFactorTol ( void *lgmres_vdata , double cf_tol );
HYPRE_Int hypre_LGMRESGetConvergenceFactorTol ( void *lgmres_vdata , double *cf_tol );
HYPRE_Int hypre_LGMRESSetMinIter ( void *lgmres_vdata , HYPRE_Int min_iter );
HYPRE_Int hypre_LGMRESGetMinIter ( void *lgmres_vdata , HYPRE_Int *min_iter );
HYPRE_Int hypre_LGMRESSetMaxIter ( void *lgmres_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_LGMRESGetMaxIter ( void *lgmres_vdata , HYPRE_Int *max_iter );
HYPRE_Int hypre_LGMRESSetStopCrit ( void *lgmres_vdata , HYPRE_Int stop_crit );
HYPRE_Int hypre_LGMRESGetStopCrit ( void *lgmres_vdata , HYPRE_Int *stop_crit );
HYPRE_Int hypre_LGMRESSetPrecond ( void *lgmres_vdata , HYPRE_Int (*precond )(), HYPRE_Int (*precond_setup )(), void *precond_data );
HYPRE_Int hypre_LGMRESGetPrecond ( void *lgmres_vdata , HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_LGMRESSetPrintLevel ( void *lgmres_vdata , HYPRE_Int level );
HYPRE_Int hypre_LGMRESGetPrintLevel ( void *lgmres_vdata , HYPRE_Int *level );
HYPRE_Int hypre_LGMRESSetLogging ( void *lgmres_vdata , HYPRE_Int level );
HYPRE_Int hypre_LGMRESGetLogging ( void *lgmres_vdata , HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetNumIterations ( void *lgmres_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_LGMRESGetConverged ( void *lgmres_vdata , HYPRE_Int *converged );
HYPRE_Int hypre_LGMRESGetFinalRelativeResidualNorm ( void *lgmres_vdata , double *relative_residual_norm );

/* HYPRE_bicgstab.c */
HYPRE_Int HYPRE_BiCGSTABDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BiCGSTABSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_BiCGSTABSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
HYPRE_Int HYPRE_BiCGSTABSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
HYPRE_Int HYPRE_BiCGSTABSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_BiCGSTABSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_BiCGSTABSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_BiCGSTABSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_BiCGSTABGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_BiCGSTABSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_BiCGSTABSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_BiCGSTABGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BiCGSTABGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
HYPRE_Int HYPRE_BiCGSTABGetResidual ( HYPRE_Solver solver , void **residual );

/* HYPRE_cgnr.c */
HYPRE_Int HYPRE_CGNRDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_CGNRSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_CGNRSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_CGNRSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_CGNRSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_CGNRSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precondT , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_CGNRGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_CGNRSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_CGNRGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_CGNRGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );

/* HYPRE_gmres.c */
HYPRE_Int HYPRE_GMRESSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_GMRESGetKDim ( HYPRE_Solver solver , HYPRE_Int *k_dim );
HYPRE_Int HYPRE_GMRESSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_GMRESGetTol ( HYPRE_Solver solver , double *tol );
HYPRE_Int HYPRE_GMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
HYPRE_Int HYPRE_GMRESGetAbsoluteTol ( HYPRE_Solver solver , double *a_tol );
HYPRE_Int HYPRE_GMRESSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
HYPRE_Int HYPRE_GMRESGetConvergenceFactorTol ( HYPRE_Solver solver , double *cf_tol );
HYPRE_Int HYPRE_GMRESSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_GMRESGetMinIter ( HYPRE_Solver solver , HYPRE_Int *min_iter );
HYPRE_Int HYPRE_GMRESSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_GMRESGetMaxIter ( HYPRE_Solver solver , HYPRE_Int *max_iter );
HYPRE_Int HYPRE_GMRESSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_GMRESGetStopCrit ( HYPRE_Solver solver , HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_GMRESSetRelChange ( HYPRE_Solver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_GMRESGetRelChange ( HYPRE_Solver solver , HYPRE_Int *rel_change );
HYPRE_Int HYPRE_GMRESSetSkipRealResidualCheck ( HYPRE_Solver solver , HYPRE_Int skip_real_r_check );
HYPRE_Int HYPRE_GMRESGetSkipRealResidualCheck ( HYPRE_Solver solver , HYPRE_Int *skip_real_r_check );
HYPRE_Int HYPRE_GMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_GMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_GMRESSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_GMRESGetPrintLevel ( HYPRE_Solver solver , HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESSetLogging ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_GMRESGetLogging ( HYPRE_Solver solver , HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_GMRESGetConverged ( HYPRE_Solver solver , HYPRE_Int *converged );
HYPRE_Int HYPRE_GMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
HYPRE_Int HYPRE_GMRESGetResidual ( HYPRE_Solver solver , void **residual );

/* HYPRE_flexgmres.c */
HYPRE_Int HYPRE_FlexGMRESSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_FlexGMRESGetKDim ( HYPRE_Solver solver , HYPRE_Int *k_dim );
HYPRE_Int HYPRE_FlexGMRESSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_FlexGMRESGetTol ( HYPRE_Solver solver , double *tol );
HYPRE_Int HYPRE_FlexGMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
HYPRE_Int HYPRE_FlexGMRESGetAbsoluteTol ( HYPRE_Solver solver , double *a_tol );
HYPRE_Int HYPRE_FlexGMRESSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
HYPRE_Int HYPRE_FlexGMRESGetConvergenceFactorTol ( HYPRE_Solver solver , double *cf_tol );
HYPRE_Int HYPRE_FlexGMRESSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_FlexGMRESGetMinIter ( HYPRE_Solver solver , HYPRE_Int *min_iter );
HYPRE_Int HYPRE_FlexGMRESSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_FlexGMRESGetMaxIter ( HYPRE_Solver solver , HYPRE_Int *max_iter );
HYPRE_Int HYPRE_FlexGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_FlexGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_FlexGMRESSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESGetPrintLevel ( HYPRE_Solver solver , HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESSetLogging ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESGetLogging ( HYPRE_Solver solver , HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_FlexGMRESGetConverged ( HYPRE_Solver solver , HYPRE_Int *converged );
HYPRE_Int HYPRE_FlexGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
HYPRE_Int HYPRE_FlexGMRESGetResidual ( HYPRE_Solver solver , void **residual );
HYPRE_Int HYPRE_FlexGMRESSetModifyPC ( HYPRE_Solver solver , HYPRE_Int (*modify_pc )(HYPRE_Solver ,HYPRE_Int ,double ));

/* HYPRE_lgmres.c */
HYPRE_Int HYPRE_LGMRESSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_LGMRESGetKDim ( HYPRE_Solver solver , HYPRE_Int *k_dim );
HYPRE_Int HYPRE_LGMRESSetAugDim ( HYPRE_Solver solver , HYPRE_Int aug_dim );
HYPRE_Int HYPRE_LGMRESGetAugDim ( HYPRE_Solver solver , HYPRE_Int *aug_dim );
HYPRE_Int HYPRE_LGMRESSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_LGMRESGetTol ( HYPRE_Solver solver , double *tol );
HYPRE_Int HYPRE_LGMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
HYPRE_Int HYPRE_LGMRESGetAbsoluteTol ( HYPRE_Solver solver , double *a_tol );
HYPRE_Int HYPRE_LGMRESSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
HYPRE_Int HYPRE_LGMRESGetConvergenceFactorTol ( HYPRE_Solver solver , double *cf_tol );
HYPRE_Int HYPRE_LGMRESSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_LGMRESGetMinIter ( HYPRE_Solver solver , HYPRE_Int *min_iter );
HYPRE_Int HYPRE_LGMRESSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_LGMRESGetMaxIter ( HYPRE_Solver solver , HYPRE_Int *max_iter );
HYPRE_Int HYPRE_LGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_LGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_LGMRESSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESGetPrintLevel ( HYPRE_Solver solver , HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESSetLogging ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESGetLogging ( HYPRE_Solver solver , HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_LGMRESGetConverged ( HYPRE_Solver solver , HYPRE_Int *converged );
HYPRE_Int HYPRE_LGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
HYPRE_Int HYPRE_LGMRESGetResidual ( HYPRE_Solver solver , void **residual );

/* HYPRE_pcg.c */
HYPRE_Int HYPRE_PCGSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_PCGGetTol ( HYPRE_Solver solver , double *tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
HYPRE_Int HYPRE_PCGGetAbsoluteTol ( HYPRE_Solver solver , double *a_tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTolFactor ( HYPRE_Solver solver , double abstolf );
HYPRE_Int HYPRE_PCGGetAbsoluteTolFactor ( HYPRE_Solver solver , double *abstolf );
HYPRE_Int HYPRE_PCGSetResidualTol ( HYPRE_Solver solver , double rtol );
HYPRE_Int HYPRE_PCGGetResidualTol ( HYPRE_Solver solver , double *rtol );
HYPRE_Int HYPRE_PCGSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
HYPRE_Int HYPRE_PCGGetConvergenceFactorTol ( HYPRE_Solver solver , double *cf_tol );
HYPRE_Int HYPRE_PCGSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_PCGGetMaxIter ( HYPRE_Solver solver , HYPRE_Int *max_iter );
HYPRE_Int HYPRE_PCGSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_PCGGetStopCrit ( HYPRE_Solver solver , HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_PCGSetTwoNorm ( HYPRE_Solver solver , HYPRE_Int two_norm );
HYPRE_Int HYPRE_PCGGetTwoNorm ( HYPRE_Solver solver , HYPRE_Int *two_norm );
HYPRE_Int HYPRE_PCGSetRelChange ( HYPRE_Solver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_PCGGetRelChange ( HYPRE_Solver solver , HYPRE_Int *rel_change );
HYPRE_Int HYPRE_PCGSetRecomputeResidual ( HYPRE_Solver solver , HYPRE_Int recompute_residual );
HYPRE_Int HYPRE_PCGGetRecomputeResidual ( HYPRE_Solver solver , HYPRE_Int *recompute_residual );
HYPRE_Int HYPRE_PCGSetRecomputeResidualP ( HYPRE_Solver solver , HYPRE_Int recompute_residual_p );
HYPRE_Int HYPRE_PCGGetRecomputeResidualP ( HYPRE_Solver solver , HYPRE_Int *recompute_residual_p );
HYPRE_Int HYPRE_PCGSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_PCGGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_PCGSetLogging ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_PCGGetLogging ( HYPRE_Solver solver , HYPRE_Int *level );
HYPRE_Int HYPRE_PCGSetPrintLevel ( HYPRE_Solver solver , HYPRE_Int level );
HYPRE_Int HYPRE_PCGGetPrintLevel ( HYPRE_Solver solver , HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_PCGGetConverged ( HYPRE_Solver solver , HYPRE_Int *converged );
HYPRE_Int HYPRE_PCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
HYPRE_Int HYPRE_PCGGetResidual ( HYPRE_Solver solver , void **residual );

/* pcg.c */
hypre_PCGFunctions *hypre_PCGFunctionsCreate ( char *(*CAlloc )(size_t count ,size_t elt_size ), HYPRE_Int (*Free )(char *ptr ), HYPRE_Int (*CommInfo )(void *A ,HYPRE_Int *my_id ,HYPRE_Int *num_procs ), void *(*CreateVector )(void *vector ), HYPRE_Int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), HYPRE_Int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), HYPRE_Int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), HYPRE_Int (*CopyVector )(void *x ,void *y ), HYPRE_Int (*ClearVector )(void *x ), HYPRE_Int (*ScaleVector )(double alpha ,void *x ), HYPRE_Int (*Axpy )(double alpha ,void *x ,void *y ), HYPRE_Int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), HYPRE_Int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_PCGCreate ( hypre_PCGFunctions *pcg_functions );
HYPRE_Int hypre_PCGDestroy ( void *pcg_vdata );
HYPRE_Int hypre_PCGGetResidual ( void *pcg_vdata , void **residual );
HYPRE_Int hypre_PCGSetup ( void *pcg_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_PCGSolve ( void *pcg_vdata , void *A , void *b , void *x );
HYPRE_Int hypre_PCGSetTol ( void *pcg_vdata , double tol );
HYPRE_Int hypre_PCGGetTol ( void *pcg_vdata , double *tol );
HYPRE_Int hypre_PCGSetAbsoluteTol ( void *pcg_vdata , double a_tol );
HYPRE_Int hypre_PCGGetAbsoluteTol ( void *pcg_vdata , double *a_tol );
HYPRE_Int hypre_PCGSetAbsoluteTolFactor ( void *pcg_vdata , double atolf );
HYPRE_Int hypre_PCGGetAbsoluteTolFactor ( void *pcg_vdata , double *atolf );
HYPRE_Int hypre_PCGSetResidualTol ( void *pcg_vdata , double rtol );
HYPRE_Int hypre_PCGGetResidualTol ( void *pcg_vdata , double *rtol );
HYPRE_Int hypre_PCGSetConvergenceFactorTol ( void *pcg_vdata , double cf_tol );
HYPRE_Int hypre_PCGGetConvergenceFactorTol ( void *pcg_vdata , double *cf_tol );
HYPRE_Int hypre_PCGSetMaxIter ( void *pcg_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_PCGGetMaxIter ( void *pcg_vdata , HYPRE_Int *max_iter );
HYPRE_Int hypre_PCGSetTwoNorm ( void *pcg_vdata , HYPRE_Int two_norm );
HYPRE_Int hypre_PCGGetTwoNorm ( void *pcg_vdata , HYPRE_Int *two_norm );
HYPRE_Int hypre_PCGSetRelChange ( void *pcg_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_PCGGetRelChange ( void *pcg_vdata , HYPRE_Int *rel_change );
HYPRE_Int hypre_PCGSetRecomputeResidual ( void *pcg_vdata , HYPRE_Int recompute_residual );
HYPRE_Int hypre_PCGGetRecomputeResidual ( void *pcg_vdata , HYPRE_Int *recompute_residual );
HYPRE_Int hypre_PCGSetRecomputeResidualP ( void *pcg_vdata , HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_PCGGetRecomputeResidualP ( void *pcg_vdata , HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_PCGSetStopCrit ( void *pcg_vdata , HYPRE_Int stop_crit );
HYPRE_Int hypre_PCGGetStopCrit ( void *pcg_vdata , HYPRE_Int *stop_crit );
HYPRE_Int hypre_PCGGetPrecond ( void *pcg_vdata , HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_PCGSetPrecond ( void *pcg_vdata , HYPRE_Int (*precond )(), HYPRE_Int (*precond_setup )(), void *precond_data );
HYPRE_Int hypre_PCGSetPrintLevel ( void *pcg_vdata , HYPRE_Int level );
HYPRE_Int hypre_PCGGetPrintLevel ( void *pcg_vdata , HYPRE_Int *level );
HYPRE_Int hypre_PCGSetLogging ( void *pcg_vdata , HYPRE_Int level );
HYPRE_Int hypre_PCGGetLogging ( void *pcg_vdata , HYPRE_Int *level );
HYPRE_Int hypre_PCGGetNumIterations ( void *pcg_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_PCGGetConverged ( void *pcg_vdata , HYPRE_Int *converged );
HYPRE_Int hypre_PCGPrintLogging ( void *pcg_vdata , HYPRE_Int myid );
HYPRE_Int hypre_PCGGetFinalRelativeResidualNorm ( void *pcg_vdata , double *relative_residual_norm );

#ifdef __cplusplus
}
#endif

#endif

