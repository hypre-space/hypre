/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "HYPRE_krylov.h"

#ifndef hypre_KRYLOV_HEADER
#define hypre_KRYLOV_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"

#define hypre_CTAllocF(type, count, funcs) ( (type *)(*(funcs->CAlloc))((unsigned int)(count), (unsigned int)sizeof(type)) )

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
 * $Revision$
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
char *hypre_ParKrylovCAlloc( int count , int elt_size );
int hypre_ParKrylovFree( char *ptr );
void *hypre_ParKrylovCreateVectorArray( int n , void *vvector );
int hypre_ParKrylovMatvecT( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_ParKrylovClearVector( void *x );
*/
/* functions in pcg_struct.c which are used here:
  void *hypre_ParKrylovCreateVector( void *vvector );
  int hypre_ParKrylovDestroyVector( void *vvector );
  void *hypre_ParKrylovMatvecCreate( void *A , void *x );
  int hypre_ParKrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
  int hypre_ParKrylovMatvecDestroy( void *matvec_data );
  double hypre_ParKrylovInnerProd( void *x , void *y );
  int hypre_ParKrylovCopyVector( void *x , void *y );
  int hypre_ParKrylovScaleVector( double alpha , void *x );
  int hypre_ParKrylovAxpy( double alpha , void *x , void *y );
  int hypre_ParKrylovCommInfo( void *A , int *my_id , int *num_procs );
  int hypre_ParKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
  int hypre_ParKrylovIdentity( void *vdata , void *A , void *b , void *x );
*/

typedef struct
{
  void *(*CreateVector)( void *vvector );
  int (*DestroyVector)( void *vvector );
  void *(*MatvecCreate)( void *A , void *x );
  int (*Matvec)( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
  int (*MatvecDestroy)( void *matvec_data );
  double (*InnerProd)( void *x , void *y );
  int (*CopyVector)( void *x , void *y );
  int (*ScaleVector)( double alpha , void *x );
  int (*Axpy)( double alpha , void *x , void *y );
  int (*CommInfo)( void *A , int *my_id , int *num_procs );
  int (*precond_setup)();
  int (*precond)();

} hypre_BiCGSTABFunctions;

/**
 * The {\tt hypre\_BiCGSTABData} object ...
 **/

typedef struct
{
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   int      converged;
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
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   int      print_level;
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
      int (*DestroyVector)( void *vvector ),
      void *(*MatvecCreate)( void *A , void *x ),
      int (*Matvec)( void *matvec_data , double alpha , void *A , void *x , double beta , void *y ),
      int (*MatvecDestroy)( void *matvec_data ),
      double (*InnerProd)( void *x , void *y ),
      int (*CopyVector)( void *x , void *y ),
      int (*ScaleVector)( double alpha , void *x ),
      int (*Axpy)( double alpha , void *x , void *y ),
      int (*CommInfo)( void *A , int *my_id , int *num_procs ),
      int (*PrecondSetup) (void *vdata, void *A, void *b, void *x ),
      int (*Precond)  ( void *vdata, void *A, void *b, void *x )
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
 * $Revision$
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
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   int    (*MatvecT)       ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   int    (*CopyVector)    ( void *x, void *y );
   int    (*ClearVector)   ( void *x );
   int    (*ScaleVector)   ( double alpha, void *x );
   int    (*Axpy)          ( double alpha, void *x, void *y );
   int    (*precond_setup) ( void *vdata, void *A, void *b, void *x );
   int    (*precond)       ( void *vdata, void *A, void *b, void *x );
   int    (*precondT)       ( void *vdata, void *A, void *b, void *x );
} hypre_CGNRFunctions;

/**
 * The {\tt hypre\_CGNRData} object ...
 **/

typedef struct
{
   double   tol;
   double   rel_residual_norm;
   int      min_iter;
   int      max_iter;
   int      stop_crit;

   void    *A;
   void    *p;
   void    *q;
   void    *r;
   void    *t;

   void    *matvec_data;
   void    *precond_data;

   hypre_CGNRFunctions * functions;

   /* log info (always logged) */
   int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   int      logging;
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
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecT)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   int    (*CopyVector)    ( void *x, void *y ),
   int    (*ClearVector)   ( void *x ),
   int    (*ScaleVector)   ( double alpha, void *x ),
   int    (*Axpy)          ( double alpha, void *x, void *y ),
   int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
   int    (*PrecondT)       ( void *vdata, void *A, void *b, void *x )
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
 * $Revision$
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
   int    (*Free)          ( char *ptr );
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   void * (*CreateVectorArray)  ( int size, void *vectors );
   int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   int    (*CopyVector)    ( void *x, void *y );
   int    (*ClearVector)   ( void *x );
   int    (*ScaleVector)   ( double alpha, void *x );
   int    (*Axpy)          ( double alpha, void *x, void *y );

   int    (*precond)();
   int    (*precond_setup)();

} hypre_GMRESFunctions;

/**
 * The {\tt hypre\_GMRESData} object ...
 **/


typedef struct
{
   int      k_dim;
   int      min_iter;
   int      max_iter;
   int      rel_change;
   int      stop_crit;
   int      converged;
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
   int      num_iterations;
 
   int     print_level; /* printing when print_level>0 */
   int     logging;  /* extra computations for logging when logging>0 */
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
   int    (*Free)          ( char *ptr ),
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( int size, void *vectors ),
   int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   int    (*CopyVector)    ( void *x, void *y ),
   int    (*ClearVector)   ( void *x ),
   int    (*ScaleVector)   ( double alpha, void *x ),
   int    (*Axpy)          ( double alpha, void *x, void *y ),
   int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
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
 * $Revision$
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
   int    (*Free)          ( char *ptr );
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   void * (*CreateVectorArray)  ( int size, void *vectors );
   int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   int    (*CopyVector)    ( void *x, void *y );
   int    (*ClearVector)   ( void *x );
   int    (*ScaleVector)   ( double alpha, void *x );
   int    (*Axpy)          ( double alpha, void *x, void *y );

   int    (*precond)();
   int    (*precond_setup)();

} hypre_LGMRESFunctions;

/**
 * The {\tt hypre\_LGMRESData} object ...
 **/



typedef struct
{
   int      k_dim;
   int      min_iter;
   int      max_iter;
   int      rel_change;
   int      stop_crit;
   int      converged;
   double   tol;
   double   cf_tol;
   double   a_tol;
   double   rel_residual_norm;

/*lgmres specific stuff */
   int      aug_dim;
   int      approx_constant;
   void   **aug_vecs;
   int     *aug_order;
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
   int      num_iterations;
 
   int     print_level; /* printing when print_level>0 */
   int     logging;  /* extra computations for logging when logging>0 */
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
   int    (*Free)          ( char *ptr ),
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( int size, void *vectors ),
   int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   int    (*CopyVector)    ( void *x, void *y ),
   int    (*ClearVector)   ( void *x ),
   int    (*ScaleVector)   ( double alpha, void *x ),
   int    (*Axpy)          ( double alpha, void *x, void *y ),
   int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
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
 * $Revision$
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
   int    (*Free)          ( char *ptr );
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   void * (*CreateVectorArray)  ( int size, void *vectors );
   int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   int    (*CopyVector)    ( void *x, void *y );
   int    (*ClearVector)   ( void *x );
   int    (*ScaleVector)   ( double alpha, void *x );
   int    (*Axpy)          ( double alpha, void *x, void *y );

   int    (*precond)();
   int    (*precond_setup)();

   int    (*modify_pc)();
   

} hypre_FlexGMRESFunctions;

/**
 * The {\tt hypre\_FlexGMRESData} object ...
 **/



typedef struct
{
   int      k_dim;
   int      min_iter;
   int      max_iter;
   int      rel_change;
   int      stop_crit;
   int      converged;
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
   int      num_iterations;
 
   int     print_level; /* printing when print_level>0 */
   int     logging;  /* extra computations for logging when logging>0 */
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
   int    (*Free)          ( char *ptr ),
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( int size, void *vectors ),
   int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   int    (*CopyVector)    ( void *x, void *y ),
   int    (*ClearVector)   ( void *x ),
   int    (*ScaleVector)   ( double alpha, void *x ),
   int    (*Axpy)          ( double alpha, void *x, void *y ),
   int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
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
 * $Revision$
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
   int    (*Free)          ( char *ptr );
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   int    (*CopyVector)    ( void *x, void *y );
   int    (*ClearVector)   ( void *x );
   int    (*ScaleVector)   ( double alpha, void *x );
   int    (*Axpy)          ( double alpha, void *x, void *y );

   int    (*precond)();
   int    (*precond_setup)();
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
*/

typedef struct
{
   double   tol;
   double   atolf;
   double   cf_tol;
   double   a_tol;
   int      max_iter;
   int      two_norm;
   int      rel_change;
   int      recompute_residual;
   int      stop_crit;
   int      converged;

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                  If that is ever changed, it still must be kept if logging>1 */

   int      owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void    *matvec_data;
   void    *precond_data;

   hypre_PCGFunctions * functions;

   /* log info (always logged) */
   int      num_iterations;
   double   rel_residual_norm;

   int     print_level; /* printing when print_level>0 */
   int     logging;  /* extra computations for logging when logging>0 */
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
   int    (*Free)          ( char *ptr ),
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   int    (*CopyVector)    ( void *x, void *y ),
   int    (*ClearVector)   ( void *x ),
   int    (*ScaleVector)   ( double alpha, void *x ),
   int    (*Axpy)          ( double alpha, void *x, void *y ),
   int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
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
hypre_BiCGSTABFunctions *hypre_BiCGSTABFunctionsCreate ( void *(*CreateVector )(void *vvector ), int (*DestroyVector )(void *vvector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_BiCGSTABCreate ( hypre_BiCGSTABFunctions *bicgstab_functions );
int hypre_BiCGSTABDestroy ( void *bicgstab_vdata );
int hypre_BiCGSTABSetup ( void *bicgstab_vdata , void *A , void *b , void *x );
int hypre_BiCGSTABSolve ( void *bicgstab_vdata , void *A , void *b , void *x );
int hypre_BiCGSTABSetTol ( void *bicgstab_vdata , double tol );
int hypre_BiCGSTABSetAbsoluteTol ( void *bicgstab_vdata , double a_tol );
int hypre_BiCGSTABSetConvergenceFactorTol ( void *bicgstab_vdata , double cf_tol );
int hypre_BiCGSTABSetMinIter ( void *bicgstab_vdata , int min_iter );
int hypre_BiCGSTABSetMaxIter ( void *bicgstab_vdata , int max_iter );
int hypre_BiCGSTABSetStopCrit ( void *bicgstab_vdata , int stop_crit );
int hypre_BiCGSTABSetPrecond ( void *bicgstab_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_BiCGSTABGetPrecond ( void *bicgstab_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_BiCGSTABSetLogging ( void *bicgstab_vdata , int logging );
int hypre_BiCGSTABSetPrintLevel ( void *bicgstab_vdata , int print_level );
int hypre_BiCGSTABGetConverged ( void *bicgstab_vdata , int *converged );
int hypre_BiCGSTABGetNumIterations ( void *bicgstab_vdata , int *num_iterations );
int hypre_BiCGSTABGetFinalRelativeResidualNorm ( void *bicgstab_vdata , double *relative_residual_norm );
int hypre_BiCGSTABGetResidual ( void *bicgstab_vdata , void **residual );

/* cgnr.c */
hypre_CGNRFunctions *hypre_CGNRFunctionsCreate ( int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecT )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ), int (*PrecondT )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_CGNRCreate ( hypre_CGNRFunctions *cgnr_functions );
int hypre_CGNRDestroy ( void *cgnr_vdata );
int hypre_CGNRSetup ( void *cgnr_vdata , void *A , void *b , void *x );
int hypre_CGNRSolve ( void *cgnr_vdata , void *A , void *b , void *x );
int hypre_CGNRSetTol ( void *cgnr_vdata , double tol );
int hypre_CGNRSetMinIter ( void *cgnr_vdata , int min_iter );
int hypre_CGNRSetMaxIter ( void *cgnr_vdata , int max_iter );
int hypre_CGNRSetStopCrit ( void *cgnr_vdata , int stop_crit );
int hypre_CGNRSetPrecond ( void *cgnr_vdata , int (*precond )(), int (*precondT )(), int (*precond_setup )(), void *precond_data );
int hypre_CGNRGetPrecond ( void *cgnr_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_CGNRSetLogging ( void *cgnr_vdata , int logging );
int hypre_CGNRGetNumIterations ( void *cgnr_vdata , int *num_iterations );
int hypre_CGNRGetFinalRelativeResidualNorm ( void *cgnr_vdata , double *relative_residual_norm );

/* gmres.c */
hypre_GMRESFunctions *hypre_GMRESFunctionsCreate ( char *(*CAlloc )(size_t count ,size_t elt_size ), int (*Free )(char *ptr ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), void *(*CreateVectorArray )(int size ,void *vectors ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_GMRESCreate ( hypre_GMRESFunctions *gmres_functions );
int hypre_GMRESDestroy ( void *gmres_vdata );
int hypre_GMRESGetResidual ( void *gmres_vdata , void **residual );
int hypre_GMRESSetup ( void *gmres_vdata , void *A , void *b , void *x );
int hypre_GMRESSolve ( void *gmres_vdata , void *A , void *b , void *x );
int hypre_GMRESSetKDim ( void *gmres_vdata , int k_dim );
int hypre_GMRESGetKDim ( void *gmres_vdata , int *k_dim );
int hypre_GMRESSetTol ( void *gmres_vdata , double tol );
int hypre_GMRESGetTol ( void *gmres_vdata , double *tol );
int hypre_GMRESSetAbsoluteTol ( void *gmres_vdata , double a_tol );
int hypre_GMRESGetAbsoluteTol ( void *gmres_vdata , double *a_tol );
int hypre_GMRESSetConvergenceFactorTol ( void *gmres_vdata , double cf_tol );
int hypre_GMRESGetConvergenceFactorTol ( void *gmres_vdata , double *cf_tol );
int hypre_GMRESSetMinIter ( void *gmres_vdata , int min_iter );
int hypre_GMRESGetMinIter ( void *gmres_vdata , int *min_iter );
int hypre_GMRESSetMaxIter ( void *gmres_vdata , int max_iter );
int hypre_GMRESGetMaxIter ( void *gmres_vdata , int *max_iter );
int hypre_GMRESSetRelChange ( void *gmres_vdata , int rel_change );
int hypre_GMRESGetRelChange ( void *gmres_vdata , int *rel_change );
int hypre_GMRESSetStopCrit ( void *gmres_vdata , int stop_crit );
int hypre_GMRESGetStopCrit ( void *gmres_vdata , int *stop_crit );
int hypre_GMRESSetPrecond ( void *gmres_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_GMRESGetPrecond ( void *gmres_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_GMRESSetPrintLevel ( void *gmres_vdata , int level );
int hypre_GMRESGetPrintLevel ( void *gmres_vdata , int *level );
int hypre_GMRESSetLogging ( void *gmres_vdata , int level );
int hypre_GMRESGetLogging ( void *gmres_vdata , int *level );
int hypre_GMRESGetNumIterations ( void *gmres_vdata , int *num_iterations );
int hypre_GMRESGetConverged ( void *gmres_vdata , int *converged );
int hypre_GMRESGetFinalRelativeResidualNorm ( void *gmres_vdata , double *relative_residual_norm );

/* flexgmres.c */
hypre_FlexGMRESFunctions *hypre_FlexGMRESFunctionsCreate ( char *(*CAlloc )(size_t count ,size_t elt_size ), int (*Free )(char *ptr ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), void *(*CreateVectorArray )(int size ,void *vectors ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_FlexGMRESCreate ( hypre_FlexGMRESFunctions *fgmres_functions );
int hypre_FlexGMRESDestroy ( void *fgmres_vdata );
int hypre_FlexGMRESGetResidual ( void *fgmres_vdata , void **residual );
int hypre_FlexGMRESSetup ( void *fgmres_vdata , void *A , void *b , void *x );
int hypre_FlexGMRESSolve ( void *fgmres_vdata , void *A , void *b , void *x );
int hypre_FlexGMRESSetKDim ( void *fgmres_vdata , int k_dim );
int hypre_FlexGMRESGetKDim ( void *fgmres_vdata , int *k_dim );
int hypre_FlexGMRESSetTol ( void *fgmres_vdata , double tol );
int hypre_FlexGMRESGetTol ( void *fgmres_vdata , double *tol );
int hypre_FlexGMRESSetAbsoluteTol ( void *fgmres_vdata , double a_tol );
int hypre_FlexGMRESGetAbsoluteTol ( void *fgmres_vdata , double *a_tol );
int hypre_FlexGMRESSetConvergenceFactorTol ( void *fgmres_vdata , double cf_tol );
int hypre_FlexGMRESGetConvergenceFactorTol ( void *fgmres_vdata , double *cf_tol );
int hypre_FlexGMRESSetMinIter ( void *fgmres_vdata , int min_iter );
int hypre_FlexGMRESGetMinIter ( void *fgmres_vdata , int *min_iter );
int hypre_FlexGMRESSetMaxIter ( void *fgmres_vdata , int max_iter );
int hypre_FlexGMRESGetMaxIter ( void *fgmres_vdata , int *max_iter );
int hypre_FlexGMRESSetStopCrit ( void *fgmres_vdata , int stop_crit );
int hypre_FlexGMRESGetStopCrit ( void *fgmres_vdata , int *stop_crit );
int hypre_FlexGMRESSetPrecond ( void *fgmres_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_FlexGMRESGetPrecond ( void *fgmres_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_FlexGMRESSetPrintLevel ( void *fgmres_vdata , int level );
int hypre_FlexGMRESGetPrintLevel ( void *fgmres_vdata , int *level );
int hypre_FlexGMRESSetLogging ( void *fgmres_vdata , int level );
int hypre_FlexGMRESGetLogging ( void *fgmres_vdata , int *level );
int hypre_FlexGMRESGetNumIterations ( void *fgmres_vdata , int *num_iterations );
int hypre_FlexGMRESGetConverged ( void *fgmres_vdata , int *converged );
int hypre_FlexGMRESGetFinalRelativeResidualNorm ( void *fgmres_vdata , double *relative_residual_norm );
int hypre_FlexGMRESSetModifyPC ( void *fgmres_vdata , int (*modify_pc )());
int hypre_FlexGMRESModifyPCDefault ( void *precond_data , int iteration , double rel_residual_norm );
int hypre_FlexGMRESModifyPCAMGExample ( void *precond_data , int iterations , double rel_residual_norm );

/* lgmres.c */
hypre_LGMRESFunctions *hypre_LGMRESFunctionsCreate ( char *(*CAlloc )(size_t count ,size_t elt_size ), int (*Free )(char *ptr ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), void *(*CreateVectorArray )(int size ,void *vectors ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_LGMRESCreate ( hypre_LGMRESFunctions *lgmres_functions );
int hypre_LGMRESDestroy ( void *lgmres_vdata );
int hypre_LGMRESGetResidual ( void *lgmres_vdata , void **residual );
int hypre_LGMRESSetup ( void *lgmres_vdata , void *A , void *b , void *x );
int hypre_LGMRESSolve ( void *lgmres_vdata , void *A , void *b , void *x );
int hypre_LGMRESSetKDim ( void *lgmres_vdata , int k_dim );
int hypre_LGMRESGetKDim ( void *lgmres_vdata , int *k_dim );
int hypre_LGMRESSetAugDim ( void *lgmres_vdata , int aug_dim );
int hypre_LGMRESGetAugDim ( void *lgmres_vdata , int *aug_dim );
int hypre_LGMRESSetTol ( void *lgmres_vdata , double tol );
int hypre_LGMRESGetTol ( void *lgmres_vdata , double *tol );
int hypre_LGMRESSetAbsoluteTol ( void *lgmres_vdata , double a_tol );
int hypre_LGMRESGetAbsoluteTol ( void *lgmres_vdata , double *a_tol );
int hypre_LGMRESSetConvergenceFactorTol ( void *lgmres_vdata , double cf_tol );
int hypre_LGMRESGetConvergenceFactorTol ( void *lgmres_vdata , double *cf_tol );
int hypre_LGMRESSetMinIter ( void *lgmres_vdata , int min_iter );
int hypre_LGMRESGetMinIter ( void *lgmres_vdata , int *min_iter );
int hypre_LGMRESSetMaxIter ( void *lgmres_vdata , int max_iter );
int hypre_LGMRESGetMaxIter ( void *lgmres_vdata , int *max_iter );
int hypre_LGMRESSetStopCrit ( void *lgmres_vdata , int stop_crit );
int hypre_LGMRESGetStopCrit ( void *lgmres_vdata , int *stop_crit );
int hypre_LGMRESSetPrecond ( void *lgmres_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_LGMRESGetPrecond ( void *lgmres_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_LGMRESSetPrintLevel ( void *lgmres_vdata , int level );
int hypre_LGMRESGetPrintLevel ( void *lgmres_vdata , int *level );
int hypre_LGMRESSetLogging ( void *lgmres_vdata , int level );
int hypre_LGMRESGetLogging ( void *lgmres_vdata , int *level );
int hypre_LGMRESGetNumIterations ( void *lgmres_vdata , int *num_iterations );
int hypre_LGMRESGetConverged ( void *lgmres_vdata , int *converged );
int hypre_LGMRESGetFinalRelativeResidualNorm ( void *lgmres_vdata , double *relative_residual_norm );

/* HYPRE_bicgstab.c */
int HYPRE_BiCGSTABDestroy ( HYPRE_Solver solver );
int HYPRE_BiCGSTABSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_BiCGSTABSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_BiCGSTABSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_BiCGSTABSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_BiCGSTABSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
int HYPRE_BiCGSTABSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_BiCGSTABSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_BiCGSTABSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_BiCGSTABSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_BiCGSTABGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_BiCGSTABSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_BiCGSTABSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_BiCGSTABGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_BiCGSTABGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
int HYPRE_BiCGSTABGetResidual ( HYPRE_Solver solver , void **residual );

/* HYPRE_cgnr.c */
int HYPRE_CGNRDestroy ( HYPRE_Solver solver );
int HYPRE_CGNRSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CGNRSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CGNRSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_CGNRSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_CGNRSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_CGNRSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_CGNRSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precondT , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_CGNRGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_CGNRSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_CGNRGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_CGNRGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );

/* HYPRE_gmres.c */
int HYPRE_GMRESSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_GMRESSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_GMRESSetKDim ( HYPRE_Solver solver , int k_dim );
int HYPRE_GMRESGetKDim ( HYPRE_Solver solver , int *k_dim );
int HYPRE_GMRESSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_GMRESGetTol ( HYPRE_Solver solver , double *tol );
int HYPRE_GMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_GMRESGetAbsoluteTol ( HYPRE_Solver solver , double *a_tol );
int HYPRE_GMRESSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
int HYPRE_GMRESGetConvergenceFactorTol ( HYPRE_Solver solver , double *cf_tol );
int HYPRE_GMRESSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_GMRESGetMinIter ( HYPRE_Solver solver , int *min_iter );
int HYPRE_GMRESSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_GMRESGetMaxIter ( HYPRE_Solver solver , int *max_iter );
int HYPRE_GMRESSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_GMRESGetStopCrit ( HYPRE_Solver solver , int *stop_crit );
int HYPRE_GMRESSetRelChange ( HYPRE_Solver solver , int rel_change );
int HYPRE_GMRESGetRelChange ( HYPRE_Solver solver , int *rel_change );
int HYPRE_GMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_GMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_GMRESSetPrintLevel ( HYPRE_Solver solver , int level );
int HYPRE_GMRESGetPrintLevel ( HYPRE_Solver solver , int *level );
int HYPRE_GMRESSetLogging ( HYPRE_Solver solver , int level );
int HYPRE_GMRESGetLogging ( HYPRE_Solver solver , int *level );
int HYPRE_GMRESGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_GMRESGetConverged ( HYPRE_Solver solver , int *converged );
int HYPRE_GMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
int HYPRE_GMRESGetResidual ( HYPRE_Solver solver , void **residual );

/* HYPRE_flexgmres.c */
int HYPRE_FlexGMRESSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_FlexGMRESSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_FlexGMRESSetKDim ( HYPRE_Solver solver , int k_dim );
int HYPRE_FlexGMRESGetKDim ( HYPRE_Solver solver , int *k_dim );
int HYPRE_FlexGMRESSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_FlexGMRESGetTol ( HYPRE_Solver solver , double *tol );
int HYPRE_FlexGMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_FlexGMRESGetAbsoluteTol ( HYPRE_Solver solver , double *a_tol );
int HYPRE_FlexGMRESSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
int HYPRE_FlexGMRESGetConvergenceFactorTol ( HYPRE_Solver solver , double *cf_tol );
int HYPRE_FlexGMRESSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_FlexGMRESGetMinIter ( HYPRE_Solver solver , int *min_iter );
int HYPRE_FlexGMRESSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_FlexGMRESGetMaxIter ( HYPRE_Solver solver , int *max_iter );
int HYPRE_FlexGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_FlexGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_FlexGMRESSetPrintLevel ( HYPRE_Solver solver , int level );
int HYPRE_FlexGMRESGetPrintLevel ( HYPRE_Solver solver , int *level );
int HYPRE_FlexGMRESSetLogging ( HYPRE_Solver solver , int level );
int HYPRE_FlexGMRESGetLogging ( HYPRE_Solver solver , int *level );
int HYPRE_FlexGMRESGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_FlexGMRESGetConverged ( HYPRE_Solver solver , int *converged );
int HYPRE_FlexGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
int HYPRE_FlexGMRESGetResidual ( HYPRE_Solver solver , void **residual );
int HYPRE_FlexGMRESSetModifyPC ( HYPRE_Solver solver , int (*modify_pc )(HYPRE_Solver ,int ,double ));

/* HYPRE_lgmres.c */
int HYPRE_LGMRESSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_LGMRESSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_LGMRESSetKDim ( HYPRE_Solver solver , int k_dim );
int HYPRE_LGMRESGetKDim ( HYPRE_Solver solver , int *k_dim );
int HYPRE_LGMRESSetAugDim ( HYPRE_Solver solver , int aug_dim );
int HYPRE_LGMRESGetAugDim ( HYPRE_Solver solver , int *aug_dim );
int HYPRE_LGMRESSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_LGMRESGetTol ( HYPRE_Solver solver , double *tol );
int HYPRE_LGMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_LGMRESGetAbsoluteTol ( HYPRE_Solver solver , double *a_tol );
int HYPRE_LGMRESSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
int HYPRE_LGMRESGetConvergenceFactorTol ( HYPRE_Solver solver , double *cf_tol );
int HYPRE_LGMRESSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_LGMRESGetMinIter ( HYPRE_Solver solver , int *min_iter );
int HYPRE_LGMRESSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_LGMRESGetMaxIter ( HYPRE_Solver solver , int *max_iter );
int HYPRE_LGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_LGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_LGMRESSetPrintLevel ( HYPRE_Solver solver , int level );
int HYPRE_LGMRESGetPrintLevel ( HYPRE_Solver solver , int *level );
int HYPRE_LGMRESSetLogging ( HYPRE_Solver solver , int level );
int HYPRE_LGMRESGetLogging ( HYPRE_Solver solver , int *level );
int HYPRE_LGMRESGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_LGMRESGetConverged ( HYPRE_Solver solver , int *converged );
int HYPRE_LGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
int HYPRE_LGMRESGetResidual ( HYPRE_Solver solver , void **residual );

/* HYPRE_pcg.c */
int HYPRE_PCGSetup ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_PCGSolve ( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_PCGSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_PCGGetTol ( HYPRE_Solver solver , double *tol );
int HYPRE_PCGSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_PCGGetAbsoluteTol ( HYPRE_Solver solver , double *a_tol );
int HYPRE_PCGSetAbsoluteTolFactor ( HYPRE_Solver solver , double abstolf );
int HYPRE_PCGGetAbsoluteTolFactor ( HYPRE_Solver solver , double *abstolf );
int HYPRE_PCGSetConvergenceFactorTol ( HYPRE_Solver solver , double cf_tol );
int HYPRE_PCGGetConvergenceFactorTol ( HYPRE_Solver solver , double *cf_tol );
int HYPRE_PCGSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_PCGGetMaxIter ( HYPRE_Solver solver , int *max_iter );
int HYPRE_PCGSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_PCGGetStopCrit ( HYPRE_Solver solver , int *stop_crit );
int HYPRE_PCGSetTwoNorm ( HYPRE_Solver solver , int two_norm );
int HYPRE_PCGGetTwoNorm ( HYPRE_Solver solver , int *two_norm );
int HYPRE_PCGSetRelChange ( HYPRE_Solver solver , int rel_change );
int HYPRE_PCGGetRelChange ( HYPRE_Solver solver , int *rel_change );
int HYPRE_PCGSetRecomputeResidual ( HYPRE_Solver solver , int recompute_residual );
int HYPRE_PCGGetRecomputeResidual ( HYPRE_Solver solver , int *recompute_residual );
int HYPRE_PCGSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_PCGGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_PCGSetLogging ( HYPRE_Solver solver , int level );
int HYPRE_PCGGetLogging ( HYPRE_Solver solver , int *level );
int HYPRE_PCGSetPrintLevel ( HYPRE_Solver solver , int level );
int HYPRE_PCGGetPrintLevel ( HYPRE_Solver solver , int *level );
int HYPRE_PCGGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_PCGGetConverged ( HYPRE_Solver solver , int *converged );
int HYPRE_PCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
int HYPRE_PCGGetResidual ( HYPRE_Solver solver , void **residual );

/* pcg.c */
hypre_PCGFunctions *hypre_PCGFunctionsCreate ( char *(*CAlloc )(size_t count ,size_t elt_size ), int (*Free )(char *ptr ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_PCGCreate ( hypre_PCGFunctions *pcg_functions );
int hypre_PCGDestroy ( void *pcg_vdata );
int hypre_PCGGetResidual ( void *pcg_vdata , void **residual );
int hypre_PCGSetup ( void *pcg_vdata , void *A , void *b , void *x );
int hypre_PCGSolve ( void *pcg_vdata , void *A , void *b , void *x );
int hypre_PCGSetTol ( void *pcg_vdata , double tol );
int hypre_PCGGetTol ( void *pcg_vdata , double *tol );
int hypre_PCGSetAbsoluteTol ( void *pcg_vdata , double a_tol );
int hypre_PCGGetAbsoluteTol ( void *pcg_vdata , double *a_tol );
int hypre_PCGSetAbsoluteTolFactor ( void *pcg_vdata , double atolf );
int hypre_PCGGetAbsoluteTolFactor ( void *pcg_vdata , double *atolf );
int hypre_PCGSetConvergenceFactorTol ( void *pcg_vdata , double cf_tol );
int hypre_PCGGetConvergenceFactorTol ( void *pcg_vdata , double *cf_tol );
int hypre_PCGSetMaxIter ( void *pcg_vdata , int max_iter );
int hypre_PCGGetMaxIter ( void *pcg_vdata , int *max_iter );
int hypre_PCGSetTwoNorm ( void *pcg_vdata , int two_norm );
int hypre_PCGGetTwoNorm ( void *pcg_vdata , int *two_norm );
int hypre_PCGSetRelChange ( void *pcg_vdata , int rel_change );
int hypre_PCGGetRelChange ( void *pcg_vdata , int *rel_change );
int hypre_PCGSetRecomputeResidual ( void *pcg_vdata , int recompute_residual );
int hypre_PCGGetRecomputeResidual ( void *pcg_vdata , int *recompute_residual );
int hypre_PCGSetStopCrit ( void *pcg_vdata , int stop_crit );
int hypre_PCGGetStopCrit ( void *pcg_vdata , int *stop_crit );
int hypre_PCGGetPrecond ( void *pcg_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_PCGSetPrecond ( void *pcg_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_PCGSetPrintLevel ( void *pcg_vdata , int level );
int hypre_PCGGetPrintLevel ( void *pcg_vdata , int *level );
int hypre_PCGSetLogging ( void *pcg_vdata , int level );
int hypre_PCGGetLogging ( void *pcg_vdata , int *level );
int hypre_PCGGetNumIterations ( void *pcg_vdata , int *num_iterations );
int hypre_PCGGetConverged ( void *pcg_vdata , int *converged );
int hypre_PCGPrintLogging ( void *pcg_vdata , int myid );
int hypre_PCGGetFinalRelativeResidualNorm ( void *pcg_vdata , double *relative_residual_norm );

#ifdef __cplusplus
}
#endif

#endif

