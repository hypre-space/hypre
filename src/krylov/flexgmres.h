/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
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
