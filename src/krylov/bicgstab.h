/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
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
