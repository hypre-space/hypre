/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * krylov solver headers
 *
 *****************************************************************************/

#ifndef HYPRE_ALL_KRYLOV_HEADER
#define HYPRE_ALL_KRYLOV_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif

#define hypre_CTAllocF(type, count, funcs) \
( (type *)(*(funcs->CAlloc))\
((unsigned int)(count), (unsigned int)sizeof(type)) )

#define hypre_TFreeF( ptr, funcs ) \
( (*(funcs->Free))((char *)ptr), ptr = NULL )

/* A pointer to a type which is never defined, sort of works like void* ... */
#ifndef HYPRE_SOLVER_STRUCT
#define HYPRE_SOLVER_STRUCT
struct hypre_Solver_struct;
typedef struct hypre_Solver_struct *HYPRE_Solver;
/* similar pseudo-void* for Matrix and Vector: */
#endif
#ifndef HYPRE_MATRIX_STRUCT
#define HYPRE_MATRIX_STRUCT
struct hypre_Matrix_struct;
typedef struct hypre_Matrix_struct *HYPRE_Matrix;
#endif
#ifndef HYPRE_VECTOR_STRUCT
#define HYPRE_VECTOR_STRUCT
struct hypre_Vector_struct;
typedef struct hypre_Vector_struct *HYPRE_Vector;
#endif

typedef int (*HYPRE_PtrToSolverFcn)(HYPRE_Solver,
                                    HYPRE_Matrix,
                                    HYPRE_Vector,
                                    HYPRE_Vector);

#endif
/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * BiCGSTAB bicgstab
 *
 *****************************************************************************/

#ifndef HYPRE_KRYLOV_BiCGSTAB_HEADER
#define HYPRE_KRYLOV_BiCGSTAB_HEADER

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
  int (*IdentitySetup)( void *vdata , void *A , void *b , void *x );
  int (*Identity)( void *vdata , void *A , void *b , void *x );

   int    (*precond)();
   int    (*precond_setup)();

} hypre_BiCGSTABFunctions;

/**
 * The {\tt hypre\_BiCGSTABData} object ...
 **/

typedef struct
{
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

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
      int    (*precond)(),
      int    (*precond_setup)()
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
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * cgnr (conjugate gradient on the normal equations A^TAx = A^Tb) functions
 *
 *****************************************************************************/

#ifndef HYPRE_KRYLOV_CGNR_HEADER
#define HYPRE_KRYLOV_CGNR_HEADER

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
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * GMRES gmres
 *
 *****************************************************************************/

#ifndef HYPRE_KRYLOV_GMRES_HEADER
#define HYPRE_KRYLOV_GMRES_HEADER

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
   char * (*CAlloc)        ( int count, int elt_size );
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
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *w;
   void  **p;

   void  *matvec_data;
   void    *precond_data;

   hypre_GMRESFunctions * functions;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
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
   char * (*CAlloc)        ( int count, int elt_size ),
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
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Preconditioned conjugate gradient (Omin) headers
 *
 *****************************************************************************/

#ifndef HYPRE_KRYLOV_PCG_HEADER
#define HYPRE_KRYLOV_PCG_HEADER

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
   char * (*CAlloc)        ( int count, int elt_size );
   int    (*Free)          ( char *ptr );
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

/* rel_change!=0 means: if pass the other stopping criteria,
 also check the relative change in the solution x.
   stop_crit!=0 means: absolute error tolerance rather than
 the usual relative error tolerance on the residual.  Never
 applies if rel_change!=0.
*/

typedef struct
{
   double   tol;
   double   cf_tol;
   int      max_iter;
   int      two_norm;
   int      rel_change;
   int      stop_crit;

   void    *A;
   void    *p;
   void    *s;
   void    *r;

   void    *matvec_data;
   void    *precond_data;

   hypre_PCGFunctions * functions;

   /* log info (always logged) */
   int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   double  *rel_norms;

} hypre_PCGData;

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
   char * (*CAlloc)        ( int count, int elt_size ),
   int    (*Free)          ( char *ptr ),
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

#ifndef hypre_KRYLOV_HEADER
#define hypre_KRYLOV_HEADER

#ifdef __cplusplus
extern "C" {
#endif


/* HYPRE_bicgstab.c */
int HYPRE_BiCGSTABDestroy( HYPRE_Solver solver );
int HYPRE_BiCGSTABSetup( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_BiCGSTABSolve( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_BiCGSTABSetTol( HYPRE_Solver solver , double tol );
int HYPRE_BiCGSTABSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_BiCGSTABSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_BiCGSTABSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_BiCGSTABSetPrecond( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_BiCGSTABGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_BiCGSTABSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_BiCGSTABGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_BiCGSTABGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_cgnr.c */
int HYPRE_CGNRDestroy( HYPRE_Solver solver );
int HYPRE_CGNRSetup( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CGNRSolve( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CGNRSetTol( HYPRE_Solver solver , double tol );
int HYPRE_CGNRSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_CGNRSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_CGNRSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_CGNRSetPrecond( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precondT , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_CGNRGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_CGNRSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_CGNRGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_CGNRGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_gmres.c */
int HYPRE_GMRESSetup( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_GMRESSolve( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_GMRESSetKDim( HYPRE_Solver solver , int k_dim );
int HYPRE_GMRESSetTol( HYPRE_Solver solver , double tol );
int HYPRE_GMRESSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_GMRESSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_GMRESSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_GMRESSetPrecond( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_GMRESGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_GMRESSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_GMRESGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_GMRESGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_pcg.c */
int HYPRE_PCGSetup( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_PCGSolve( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_PCGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_PCGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_PCGSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_PCGSetTwoNorm( HYPRE_Solver solver , int two_norm );
int HYPRE_PCGSetRelChange( HYPRE_Solver solver , int rel_change );
int HYPRE_PCGSetPrecond( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_PCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_PCGSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_PCGGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_PCGGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* bicgstab.c */
hypre_BiCGSTABFunctions *hypre_BiCGSTABFunctionsCreate( void *(*CreateVector )(void *vvector ), int (*DestroyVector )(void *vvector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), int (*precond )(), int (*precond_setup )());
void *hypre_BiCGSTABCreate( hypre_BiCGSTABFunctions *bicgstab_functions );
int hypre_BiCGSTABDestroy( void *bicgstab_vdata );
int hypre_BiCGSTABSetup( void *bicgstab_vdata , void *A , void *b , void *x );
int hypre_BiCGSTABSolve( void *bicgstab_vdata , void *A , void *b , void *x );
int hypre_BiCGSTABSetTol( void *bicgstab_vdata , double tol );
int hypre_BiCGSTABSetMinIter( void *bicgstab_vdata , int min_iter );
int hypre_BiCGSTABSetMaxIter( void *bicgstab_vdata , int max_iter );
int hypre_BiCGSTABSetStopCrit( void *bicgstab_vdata , double stop_crit );
int hypre_BiCGSTABSetPrecond( void *bicgstab_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_BiCGSTABGetPrecond( void *bicgstab_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_BiCGSTABSetLogging( void *bicgstab_vdata , int logging );
int hypre_BiCGSTABGetNumIterations( void *bicgstab_vdata , int *num_iterations );
int hypre_BiCGSTABGetFinalRelativeResidualNorm( void *bicgstab_vdata , double *relative_residual_norm );

/* cgnr.c */
hypre_CGNRFunctions *hypre_CGNRFunctionsCreate( int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecT )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ), int (*PrecondT )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_CGNRCreate( hypre_CGNRFunctions *cgnr_functions );
int hypre_CGNRDestroy( void *cgnr_vdata );
int hypre_CGNRSetup( void *cgnr_vdata , void *A , void *b , void *x );
int hypre_CGNRSolve( void *cgnr_vdata , void *A , void *b , void *x );
int hypre_CGNRSetTol( void *cgnr_vdata , double tol );
int hypre_CGNRSetMinIter( void *cgnr_vdata , int min_iter );
int hypre_CGNRSetMaxIter( void *cgnr_vdata , int max_iter );
int hypre_CGNRSetStopCrit( void *cgnr_vdata , int stop_crit );
int hypre_CGNRSetPrecond( void *cgnr_vdata , int (*precond )(), int (*precondT )(), int (*precond_setup )(), void *precond_data );
int hypre_CGNRGetPrecond( void *cgnr_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_CGNRSetLogging( void *cgnr_vdata , int logging );
int hypre_CGNRGetNumIterations( void *cgnr_vdata , int *num_iterations );
int hypre_CGNRGetFinalRelativeResidualNorm( void *cgnr_vdata , double *relative_residual_norm );

/* gmres.c */
hypre_GMRESFunctions *hypre_GMRESFunctionsCreate( char *(*CAlloc )(int count ,int elt_size ), int (*Free )(char *ptr ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), void *(*CreateVectorArray )(int size ,void *vectors ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_GMRESCreate( hypre_GMRESFunctions *gmres_functions );
int hypre_GMRESDestroy( void *gmres_vdata );
int hypre_GMRESSetup( void *gmres_vdata , void *A , void *b , void *x );
int hypre_GMRESSolve( void *gmres_vdata , void *A , void *b , void *x );
int hypre_GMRESSetKDim( void *gmres_vdata , int k_dim );
int hypre_GMRESSetTol( void *gmres_vdata , double tol );
int hypre_GMRESSetMinIter( void *gmres_vdata , int min_iter );
int hypre_GMRESSetMaxIter( void *gmres_vdata , int max_iter );
int hypre_GMRESSetStopCrit( void *gmres_vdata , double stop_crit );
int hypre_GMRESSetPrecond( void *gmres_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_GMRESGetPrecond( void *gmres_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_GMRESSetLogging( void *gmres_vdata , int logging );
int hypre_GMRESGetNumIterations( void *gmres_vdata , int *num_iterations );
int hypre_GMRESGetFinalRelativeResidualNorm( void *gmres_vdata , double *relative_residual_norm );

/* pcg.c */
hypre_PCGFunctions *hypre_PCGFunctionsCreate( char *(*CAlloc )(int count ,int elt_size ), int (*Free )(char *ptr ), void *(*CreateVector )(void *vector ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_PCGCreate( hypre_PCGFunctions *pcg_functions );
int hypre_PCGDestroy( void *pcg_vdata );
int hypre_PCGSetup( void *pcg_vdata , void *A , void *b , void *x );
int hypre_PCGSolve( void *pcg_vdata , void *A , void *b , void *x );
int hypre_PCGSetTol( void *pcg_vdata , double tol );
int hypre_PCGSetConvergenceFactorTol( void *pcg_vdata , double cf_tol );
int hypre_PCGSetMaxIter( void *pcg_vdata , int max_iter );
int hypre_PCGSetTwoNorm( void *pcg_vdata , int two_norm );
int hypre_PCGSetRelChange( void *pcg_vdata , int rel_change );
int hypre_PCGSetStopCrit( void *pcg_vdata , int stop_crit );
int hypre_PCGGetPrecond( void *pcg_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_PCGSetPrecond( void *pcg_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_PCGSetLogging( void *pcg_vdata , int logging );
int hypre_PCGGetNumIterations( void *pcg_vdata , int *num_iterations );
int hypre_PCGPrintLogging( void *pcg_vdata , int myid );
int hypre_PCGGetFinalRelativeResidualNorm( void *pcg_vdata , double *relative_residual_norm );


#ifdef __cplusplus
}
#endif

#endif

