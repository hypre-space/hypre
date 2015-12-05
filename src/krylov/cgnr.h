/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.3 $
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
