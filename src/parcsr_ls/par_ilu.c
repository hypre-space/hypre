/*BHEADER**********************************************************************
 * Copyright (c) 2015,  Lawrence Livermore National Security, LLC.
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
 * Incomplete LU factorization smoother
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_ilu.h"
#include <assert.h>

/* Create */
void *
hypre_ILUCreate()
{
   hypre_ParILUData  *ilu_data;

   ilu_data = hypre_CTAlloc(hypre_ParILUData,  1, HYPRE_MEMORY_HOST);

   /* general data */
   (ilu_data -> global_solver) = 0;
   (ilu_data -> matA) = NULL;
   (ilu_data -> matL) = NULL;
   (ilu_data -> matD) = NULL;
   (ilu_data -> matU) = NULL;
   (ilu_data -> matS) = NULL;
   (ilu_data -> schur_solver) = NULL;
   (ilu_data -> schur_precond) = NULL;
   (ilu_data -> rhs) = NULL;
   (ilu_data -> x) = NULL;
  
   (ilu_data -> droptol) = hypre_TAlloc(HYPRE_Real,3,HYPRE_MEMORY_HOST);
   (ilu_data -> own_droptol_data) = 1;
   (ilu_data -> droptol)[0] = 1.0e-02;/* droptol for B */
   (ilu_data -> droptol)[1] = 1.0e-02;/* droptol for E and F */
   (ilu_data -> droptol)[2] = 1.0e-02;/* droptol for S */
   (ilu_data -> lfil) = 0;
   (ilu_data -> maxRowNnz) = 1000;
   (ilu_data -> CF_marker_array) = NULL;
   (ilu_data -> perm) = NULL;

   (ilu_data -> F) = NULL;
   (ilu_data -> U) = NULL;
   (ilu_data -> Utemp) = NULL;
   (ilu_data -> Ftemp) = NULL;
   (ilu_data -> residual) = NULL;
   (ilu_data -> rel_res_norms) = NULL;

   (ilu_data -> num_iterations) = 0;

   (ilu_data -> max_iter) = 20;
   (ilu_data -> tol) = 1.0e-7;

   (ilu_data -> logging) = 0;
   (ilu_data -> print_level) = 0;

   (ilu_data -> l1_norms) = NULL;
  
   (ilu_data -> operator_complexity) = 0.;
  
   (ilu_data -> ilu_type) = 0;
   (ilu_data -> nLU) = 0;
   
   /* see hypre_ILUSetType for more default values */
   
   return (void *) ilu_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* Destroy */
HYPRE_Int
hypre_ILUDestroy( void *data )
{
  hypre_ParILUData * ilu_data = (hypre_ParILUData*) data;
 
  /* final residual vector */
  if((ilu_data -> residual))
  {
    hypre_ParVectorDestroy( (ilu_data -> residual) );
    (ilu_data -> residual) = NULL;
  }
  if((ilu_data -> rel_res_norms))
  {
    hypre_TFree( (ilu_data -> rel_res_norms) , HYPRE_MEMORY_HOST);
    (ilu_data -> rel_res_norms) = NULL;
  }
  /* temp vectors for solve phase */
  if((ilu_data -> Utemp))
  {
    hypre_ParVectorDestroy( (ilu_data -> Utemp) );
    (ilu_data -> Utemp) = NULL;
  }
  if((ilu_data -> Ftemp))
  {
    hypre_ParVectorDestroy( (ilu_data -> Ftemp) );
    (ilu_data -> Ftemp) = NULL;
  }
  if((ilu_data -> rhs))
  {
    hypre_ParVectorDestroy( (ilu_data -> rhs) );
    (ilu_data -> rhs) = NULL;
  }
  if((ilu_data -> x))
  {
    hypre_ParVectorDestroy( (ilu_data -> x) );
    (ilu_data -> x) = NULL;
  }
  /* l1_norms */
  if ((ilu_data -> l1_norms))
  {
    hypre_TFree((ilu_data -> l1_norms), HYPRE_MEMORY_HOST);
    (ilu_data -> l1_norms) = NULL;
  }

  /* Factors */
  if(ilu_data -> matL)
  {
    hypre_ParCSRMatrixDestroy((ilu_data -> matL));
    (ilu_data -> matL) = NULL;
  }
  if(ilu_data -> matU)
  {
    hypre_ParCSRMatrixDestroy((ilu_data -> matU));
    (ilu_data -> matU) = NULL;
  }
  if(ilu_data -> matD)
  {
    hypre_TFree((ilu_data -> matD), HYPRE_MEMORY_HOST);
    (ilu_data -> matD) = NULL;
  }
  if(ilu_data -> matS)
  {
    hypre_ParCSRMatrixDestroy((ilu_data -> matS));
    (ilu_data -> matS) = NULL;
  }     
  if(ilu_data -> schur_solver)
  {
     switch(ilu_data -> ilu_type){
      case 10: case 11: 
         HYPRE_ParCSRGMRESDestroy(ilu_data -> schur_solver); //GMRES for Schur
         break;
      case 20: case 21:
         hypre_NSHDestroy(hypre_ParILUDataSchurSolver(ilu_data));
         break;
      default:
         break;
        }
     (ilu_data -> schur_solver) = NULL;
  } 
  if(ilu_data -> schur_precond)
  {
     switch(ilu_data -> ilu_type){
      case 10: case 11: 
         HYPRE_ILUDestroy(ilu_data -> schur_precond); //ILU as precond for Schur
         break;
      default:
         break;
        }
     (ilu_data -> schur_precond) = NULL;
  } 
  /* CF marker array */
  if((ilu_data -> CF_marker_array))
  {
    hypre_TFree((ilu_data -> CF_marker_array), HYPRE_MEMORY_HOST);
    (ilu_data -> CF_marker_array) = NULL;
  }
  /* permutation array */
  if((ilu_data -> perm))
  {
    hypre_TFree((ilu_data -> perm), HYPRE_MEMORY_HOST);
    (ilu_data -> perm) = NULL;
  }
  /* droptol array */
  if((ilu_data -> own_droptol_data))
  {
     hypre_TFree((ilu_data -> droptol), HYPRE_MEMORY_HOST);
     (ilu_data -> own_droptol_data) = 0;
     (ilu_data -> droptol) = NULL;
  }
  if((ilu_data -> sp_own_droptol_data))
  {
     hypre_TFree((ilu_data -> sp_ilu_droptol), HYPRE_MEMORY_HOST);
     (ilu_data -> sp_own_droptol_data) = 0;
     (ilu_data -> sp_ilu_droptol) = NULL;
  }
  /* ilu data */
  hypre_TFree(ilu_data, HYPRE_MEMORY_HOST);

  return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* set fill level (for ilu(k)) */
HYPRE_Int
hypre_ILUSetLevelOfFill( void *ilu_vdata, HYPRE_Int lfil )
{
  hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
  (ilu_data -> lfil) = lfil;
  return hypre_error_flag;
}
/* set max non-zeros per row in factors (for ilut) */
HYPRE_Int
hypre_ILUSetMaxNnzPerRow( void *ilu_vdata, HYPRE_Int nzmax )
{
  hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
  (ilu_data -> maxRowNnz) = nzmax;
  return hypre_error_flag;
}
/* set threshold for dropping in LU factors (for ilut) */
HYPRE_Int
hypre_ILUSetDropThreshold( void *ilu_vdata, HYPRE_Real threshold )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> droptol)[0] = threshold;
   (ilu_data -> droptol)[1] = threshold;
   (ilu_data -> droptol)[2] = threshold;
   return hypre_error_flag;
}
/* set array of threshold for dropping in LU factors (for ilut) */
HYPRE_Int
hypre_ILUSetDropThresholdArray( void *ilu_vdata, HYPRE_Real *threshold )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
   /* need to free memory if we own droptol array before */
   if((ilu_data -> own_droptol_data))
   {
      hypre_TFree((ilu_data -> droptol), HYPRE_MEMORY_HOST);
      (ilu_data -> own_droptol_data) = 0;
   }
   (ilu_data -> droptol) = threshold;
   return hypre_error_flag;
}
/* set if owns threshold data (for ilut) */
HYPRE_Int
hypre_ILUSetOwnDropThreshold( void *ilu_vdata, HYPRE_Int own_droptol_data )
{
  hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
  (ilu_data -> own_droptol_data) = own_droptol_data;
  return hypre_error_flag;
}
/* set ILU factorization type */
HYPRE_Int
hypre_ILUSetType( void *ilu_vdata, HYPRE_Int ilu_type )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> ilu_type) = ilu_type;
   /* reset default value, not a large cost 
    * assume we won't change back from
    */
   switch(ilu_type)
   {
      /* NSH type */
      case 20: case 21:
         /* set NSH Solver parameters */
         hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data) = 5;
         hypre_ParILUDataSchurNSHSolveTol(ilu_data) = 1.0e-09;
         hypre_ParILUDataSchurSolverLogging(ilu_data) = 0;
         hypre_ParILUDataSchurSolverPrintLevel(ilu_data) = 0;
         if(hypre_ParILUDataSchurNSHOwnDroptolData(ilu_data))
         {
            hypre_TFree(hypre_ParILUDataSchurNSHDroptol(ilu_data), HYPRE_MEMORY_HOST);
         }
         hypre_ParILUDataSchurNSHDroptol(ilu_data) = hypre_ParILUDataDroptol(ilu_data);
         hypre_ParILUDataSchurNSHOwnDroptolData(ilu_data) = 0;
         
         /* set NHS inverse parameters */
         hypre_ParILUDataSchurNSHMaxNumIter(ilu_data) = 2;/* kDim */
         hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data) = 1000;
         hypre_ParILUDataSchurNSHTol(ilu_data) = 1.0e-09;
         
         /* set MR inverse parameters */
         hypre_ParILUDataSchurMRMaxIter(ilu_data) = 5;
         hypre_ParILUDataSchurMRColVersion(ilu_data) = 0;/* sp_lfil */
         hypre_ParILUDataSchurMRMaxRowNnz(ilu_data) = 200;
         hypre_ParILUDataSchurMRTol(ilu_data) = 1.0e-09;
         break;
      case 10: case 11:
         /* default data for schur solver */
         hypre_ParILUDataSchurGMRESKDim(ilu_data) = 5;
         hypre_ParILUDataSchurGMRESTol(ilu_data)=1.0e-09;
         hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data) = 0.0;
         hypre_ParILUDataSchurSolverLogging(ilu_data) = 0;
         hypre_ParILUDataSchurSolverPrintLevel(ilu_data) = 0;
         hypre_ParILUDataSchurGMRESRelChange(ilu_data) = 0;
         
         /* default data for schur precond 
          * default ILU0
          */
         hypre_ParILUDataSchurPrecondIluType(ilu_data) = 0;
         hypre_ParILUDataSchurPrecondIluLfil(ilu_data) = 0;
         hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data) = 1000;
         if(hypre_ParILUDataSchurPrecondOwnDroptolData(ilu_data))
         {
            hypre_TFree(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data), HYPRE_MEMORY_HOST);
         }
         hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = hypre_ParILUDataDroptol(ilu_data);/* use same droptol */
         hypre_ParILUDataSchurPrecondOwnDroptolData(ilu_data) = 0;
         hypre_ParILUDataSchurPrecondPrintLevel(ilu_data) = 0;
         hypre_ParILUDataSchurPrecondMaxIter(ilu_data) = 1;
         hypre_ParILUDataSchurPrecondTol(ilu_data) = 1.0e-09;
         break;
      default:
         break;
   }

   return hypre_error_flag;
}
/* Set max number of iterations for ILU solver */
HYPRE_Int
hypre_ILUSetMaxIter( void *ilu_vdata, HYPRE_Int max_iter )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> max_iter) = max_iter;
   return hypre_error_flag;
}
/* Set convergence tolerance for ILU solver */
HYPRE_Int
hypre_ILUSetTol( void *ilu_vdata, HYPRE_Real tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> tol) = tol;
   return hypre_error_flag;
}
/* Set print level for ilu solver */
HYPRE_Int
hypre_ILUSetPrintLevel( void *ilu_vdata, HYPRE_Int print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> print_level) = print_level;
   return hypre_error_flag;
}
/* Set print level for ilu solver */
HYPRE_Int
hypre_ILUSetLogging( void *ilu_vdata, HYPRE_Int logging )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> logging) = logging;
   return hypre_error_flag;
}
/* Set KDim (for GMRES) for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverKDIM( void *ilu_vdata, HYPRE_Int ss_kDim )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> ss_kDim) = ss_kDim;
   return hypre_error_flag;
}
/* Set max iteration for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverMaxIter( void *ilu_vdata, HYPRE_Int ss_max_iter )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   switch(hypre_ParILUDataIluType(ilu_data))
   {
      case 10: case 11:
         /* GMRES 
          * To avoid restart, GMRES kDim is equal to max num iter
          */
         hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_max_iter;
         break;
      case 20: case 21:
         /* set max num iter if use NSH solve */
         hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data) = ss_max_iter;
         break;
      default:
         /* warning */
         printf("Current type has no Schur System\n");
         break;
   }
   return hypre_error_flag;
}
/* Set convergence tolerance for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverTol( void *ilu_vdata, HYPRE_Real ss_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> ss_tol) = ss_tol;
   return hypre_error_flag;
}
/* Set absolute tolerance for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverAbsoluteTol( void *ilu_vdata, HYPRE_Real ss_absolute_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> ss_absolute_tol) = ss_absolute_tol;
   return hypre_error_flag;
}
/* Set logging for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverLogging( void *ilu_vdata, HYPRE_Int ss_logging )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> ss_logging) = ss_logging;
   return hypre_error_flag;
}
/* Set print level for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverPrintLevel( void *ilu_vdata, HYPRE_Int ss_print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> ss_print_level) = ss_print_level;
   return hypre_error_flag;
}
/* Set rel change (for GMRES) for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverRelChange( void *ilu_vdata, HYPRE_Int ss_rel_change )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> ss_rel_change) = ss_rel_change;
   return hypre_error_flag;
}
/* Set IUL type for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILUType( void *ilu_vdata, HYPRE_Int sp_ilu_type )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> sp_ilu_type) = sp_ilu_type;
   return hypre_error_flag;
}
/* Set IUL level of fill for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, HYPRE_Int sp_ilu_lfil )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> sp_ilu_lfil) = sp_ilu_lfil;
   return hypre_error_flag;
}
/* Set IUL max nonzeros per row for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILUMaxNnzPerRow( void *ilu_vdata, HYPRE_Int sp_ilu_max_row_nnz )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> sp_ilu_max_row_nnz) = sp_ilu_max_row_nnz;
   return hypre_error_flag;
}
/* Set IUL drop threshold for ILUT for Precond of Schur System 
 * We don't want to influence the original ILU, so create new array if not own data
 */
HYPRE_Int
hypre_ILUSetSchurPrecondILUDropThreshold( void *ilu_vdata, HYPRE_Real sp_ilu_droptol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(hypre_ParILUDataSchurPrecondOwnDroptolData(ilu_data))
   {
      /* if we own data, just change our own data */
      (ilu_data -> sp_ilu_droptol)[0] = sp_ilu_droptol;
      (ilu_data -> sp_ilu_droptol)[1] = sp_ilu_droptol;
      (ilu_data -> sp_ilu_droptol)[2] = sp_ilu_droptol;
   }
   else
   {
      /* if we share data with other, create new one 
       * becuase as default we use data from ILU, so we don't want to change it
       */
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
      hypre_ParILUDataSchurPrecondOwnDroptolData(ilu_data) = 1;
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[0] = sp_ilu_droptol;
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[1] = sp_ilu_droptol;
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[2] = sp_ilu_droptol;
   }
   return hypre_error_flag;
}
/* Set array of IUL drop threshold for ILUT for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILUDropThresholdArray( void *ilu_vdata, HYPRE_Real *sp_ilu_droptol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   /* need to free memory if we own droptol array before */
   if((ilu_data -> sp_own_droptol_data))
   {
      hypre_TFree((ilu_data -> sp_ilu_droptol), HYPRE_MEMORY_HOST);
      (ilu_data -> sp_own_droptol_data) = 0;
   }
   (ilu_data -> sp_ilu_droptol) = sp_ilu_droptol;
   return hypre_error_flag;
}
/* Set if owns drop threshold array for ILUT for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILUOwnDropThreshold( void *ilu_vdata, HYPRE_Int sp_own_droptol_data )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> sp_own_droptol_data) = sp_own_droptol_data;
   return hypre_error_flag;
}
/* Set print level for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondPrintLevel( void *ilu_vdata, HYPRE_Int sp_print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> sp_print_level) = sp_print_level;
   return hypre_error_flag;
}
/* Set max number of iterations for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondMaxIter( void *ilu_vdata, HYPRE_Int sp_max_iter )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> sp_max_iter) = sp_max_iter;
   return hypre_error_flag;
}
/* Set onvergence tolerance for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondTol( void *ilu_vdata, HYPRE_Int sp_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> sp_tol) = sp_tol;
   return hypre_error_flag;
}
/* Set tolorance for NSH for Schur System 
 * We don't want to influence the original ILU, so create new array if not own data
 */
HYPRE_Int
hypre_ILUSetSchurNSHDropThreshold( void *ilu_vdata, HYPRE_Real threshold)
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(hypre_ParILUDataSchurNSHOwnDroptolData(ilu_data))
   {
      hypre_ParILUDataSchurNSHDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 2, HYPRE_MEMORY_HOST);
      hypre_ParILUDataSchurNSHOwnDroptolData(ilu_data) = 1;
      hypre_ParILUDataSchurNSHDroptol(ilu_data)[0] = threshold;
      hypre_ParILUDataSchurNSHDroptol(ilu_data)[1] = threshold;
   }
   else
   {
      hypre_ParILUDataSchurNSHDroptol(ilu_data)[0] = threshold;
      hypre_ParILUDataSchurNSHDroptol(ilu_data)[1] = threshold;
   }
   return hypre_error_flag;
}
/* Set tolorance array for NSH for Schur System */
HYPRE_Int
hypre_ILUSetSchurNSHDropThresholdArray( void *ilu_vdata, HYPRE_Real *threshold)
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(hypre_ParILUDataSchurNSHOwnDroptolData(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataSchurNSHDroptol(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataSchurNSHOwnDroptolData(ilu_data) = 0;
   }
   hypre_ParILUDataSchurNSHDroptol(ilu_data) = threshold;
   return hypre_error_flag;
}

/* Get number of iterations for ILU solver */
HYPRE_Int
hypre_ILUGetNumIterations( void *ilu_vdata, HYPRE_Int *num_iterations )
{
   hypre_ParILUData  *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!ilu_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *num_iterations = ilu_data->num_iterations;

   return hypre_error_flag;
}
/* Get residual norms for ILU solver */
HYPRE_Int
hypre_ILUGetFinalRelativeResidualNorm( void *ilu_vdata, HYPRE_Real *res_norm )
{
   hypre_ParILUData  *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!ilu_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *res_norm = ilu_data->final_rel_residual_norm;

   return hypre_error_flag;
}
/* 
 * Quicksort of the elements in a from low to high. 
 * The elements in b are permuted according to the sorted a. 
 * The elements in iw are permuted reverse according to the sorted a as it's index
 *   ie, iw[a1] and iw[a2] will be switched if a1 and a2 are switched
 * lo and hi are the extents of the region of the array a, that is to be sorted.
*/
HYPRE_Int 
hypre_quickSortIR (HYPRE_Int *a, HYPRE_Real *b, HYPRE_Int *iw, const HYPRE_Int lo, const HYPRE_Int hi)
{
   HYPRE_Int i=lo, j=hi;
   HYPRE_Int v;
   HYPRE_Int mid = (lo+hi)>>1;
   HYPRE_Int x=ceil(a[mid]);
   HYPRE_Real q;
   //  partition
   do
   {
      while (a[i]<x) i++;
      while (a[j]>x) j--;
      if (i<=j)
      {
          v=a[i]; a[i]=a[j]; a[j]=v;
          q=b[i]; b[i]=b[j]; b[j]=q;
          v=iw[a[i]];iw[a[i]]=iw[a[j]];iw[a[j]]=v;
          i++; j--;
      }
   } while (i<=j);
   //  recursion
   if (lo<j) hypre_quickSortIR(a, b, iw, lo, j);
   if (i<hi) hypre_quickSortIR(a, b, iw, i, hi);
   
   return hypre_error_flag;
}
/* Print solver params */
HYPRE_Int
hypre_ILUWriteSolverParams(void *ilu_vdata)
{
   hypre_ParILUData  *ilu_data = (hypre_ParILUData*) ilu_vdata;      
   hypre_printf("ILU Setup parameters: \n");   
   hypre_printf("ILU factorization type: %d : ", (ilu_data -> ilu_type));
   switch(ilu_data -> ilu_type){
      case 0: hypre_printf("Block Jacobi with ILU(%d) \n", (ilu_data -> lfil));
              hypre_printf("Operator Complexity (Fill factor) = %f \n", (ilu_data -> operator_complexity));
         break;
      case 1: hypre_printf("Block Jacobi with ILUT \n");
              hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", hypre_ParILUDataDroptol(ilu_data)[0],hypre_ParILUDataDroptol(ilu_data)[1],hypre_ParILUDataDroptol(ilu_data)[2]);
              hypre_printf("Max nnz per row = %d \n", (ilu_data -> maxRowNnz));
              hypre_printf("Operator Complexity (Fill factor) = %f \n", (ilu_data -> operator_complexity));
         break;
      case 10: 
              hypre_printf("ILU-GMRES with ILU(%d) \n", (ilu_data -> lfil));
              hypre_printf("Operator Complexity (Fill factor) = %f \n", (ilu_data -> operator_complexity));
         break;
      case 11: 
              hypre_printf("ILU-GMRES with ILUT \n");
              hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", hypre_ParILUDataDroptol(ilu_data)[0],hypre_ParILUDataDroptol(ilu_data)[1],hypre_ParILUDataDroptol(ilu_data)[2]);
              hypre_printf("Max nnz per row = %d \n", (ilu_data -> maxRowNnz));
              hypre_printf("Operator Complexity (Fill factor) = %f \n", (ilu_data -> operator_complexity));
         break;
      case 20: 
              hypre_printf("Newton–Schulz–Hotelling with ILU(%d) \n", (ilu_data -> lfil));
              hypre_printf("Operator Complexity (Fill factor) = %f \n", (ilu_data -> operator_complexity));
         break;
      case 21: 
              hypre_printf("Newton–Schulz–Hotelling with ILUT \n");
              hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", hypre_ParILUDataDroptol(ilu_data)[0],hypre_ParILUDataDroptol(ilu_data)[1],hypre_ParILUDataDroptol(ilu_data)[2]);
              hypre_printf("Max nnz per row = %d \n", (ilu_data -> maxRowNnz));
              hypre_printf("Operator Complexity (Fill factor) = %f \n", (ilu_data -> operator_complexity));
         break;  
      default: hypre_printf("Unknown type \n");
         break;
   }
     
   hypre_printf("\n ILU Solver Parameters: \n");  
   hypre_printf("Max number of iterations: %d\n", (ilu_data -> max_iter));
   hypre_printf("Stopping tolerance: %e\n", (ilu_data -> tol));
   
   return hypre_error_flag;
}

/* helper functions */
/*
 * Add an element to the heap
 * I means HYPRE_Int
 * R means HYPRE_Real
 * max/min heap
 * r means heap goes from 0 to -1, -2 instead of 0 1 2
 * Ii and Ri means orderd by value of heap, like iw for ILU
 * heap: array of that heap
 * len: the current length of the heap
 * WARNING: You should first put that element to the end of the heap
 *    and add the length of heap by one before call this function.
 * the reason is that we don't want to change something outside the
 *    heap, so left it to the user
 */
HYPRE_Int
hypre_ILUMinHeapAddI(HYPRE_Int *heap, HYPRE_Int len)
{
   /* parent, left, right */
   int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(heap,p,len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUMinHeapAddIIIi(HYPRE_Int *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(Ii1,heap[p],heap[len]);
         hypre_swap2i(heap,I1,p,len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUMinHeapAddIRIi(HYPRE_Int *heap, HYPRE_Real *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(Ii1,heap[p],heap[len]);
         hypre_swap2(heap,I1,p,len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUMaxHeapAddRabsIIi(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(fabs(heap[p]) < fabs(heap[len]))
      {
         /* this is smaller */
         hypre_swap(Ii1,heap[p],heap[len]);
         hypre_swap2(I1,heap,p,len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUMaxrHeapAddRabsI(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int len)
{
   /* parent, left, right */
   int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(fabs(heap[-p]) < fabs(heap[-len]))
      {
         /* this is smaller */
         hypre_swap2(I1,heap,-p,-len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/*
 * Swap the first element with the last element of the heap,
 *    reduce size by one, and maintain the heap structure
 * I means HYPRE_Int
 * R means HYPRE_Real
 * max/min heap
 * r means heap goes from 0 to -1, -2 instead of 0 1 2
 * Ii and Ri means orderd by value of heap, like iw for ILU
 * heap: aray of that heap
 * len: current length of the heap
 * WARNING: Remember to change the len youself
 */
HYPRE_Int
hypre_ILUMinHeapRemoveI(HYPRE_Int *heap, HYPRE_Int len)
{
   /* parent, left, right */
   int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap(heap,0,len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l]<heap[r] ? l : r;
      if(heap[l]<heap[p])
      {
         hypre_swap(heap,l,p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUMinHeapRemoveIIIi(HYPRE_Int *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap(Ii1,heap[0],heap[len]);
   hypre_swap2i(heap,I1,0,len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l]<heap[r] ? l : r;
      if(heap[l]<heap[p])
      {
         hypre_swap(Ii1,heap[p],heap[l]);
         hypre_swap2i(heap,I1,l,p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUMinHeapRemoveIRIi(HYPRE_Int *heap, HYPRE_Real *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap(Ii1,heap[0],heap[len]);
   hypre_swap2(heap,I1,0,len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l]<heap[r] ? l : r;
      if(heap[l]<heap[p])
      {
         hypre_swap(Ii1,heap[p],heap[l]);
         hypre_swap2(heap,I1,l,p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUMaxHeapRemoveRabsIIi(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap(Ii1,heap[0],heap[len]);
   hypre_swap2(I1,heap,0,len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || fabs(heap[l])>fabs(heap[r]) ? l : r;
      if(fabs(heap[l])>fabs(heap[p]))
      {
         hypre_swap(Ii1,heap[p],heap[l]);
         hypre_swap2(I1,heap,l,p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUMaxrHeapRemoveRabsI(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int len)
{
   /* parent, left, right */
   int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap2(I1,heap,0,-len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || fabs(heap[-l])>fabs(heap[-r]) ? l : r;
      if(fabs(heap[-l])>fabs(heap[-p]))
      {
         hypre_swap2(I1,heap,-l,-p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* Split based on quick sort algorithm
 * find the largest k elements out of original array
 * array: input array for compare
 * I: integer array bind with array
 * k: largest k elements
 * len: length of the array
 */
HYPRE_Int
hypre_ILUMaxQSplitRabsI(HYPRE_Real *array, HYPRE_Int *I, HYPRE_Int left, HYPRE_Int bound, HYPRE_Int right)
{
   HYPRE_Int i, last;
   if (left >= right)
   {
      return hypre_error_flag;
   }
   hypre_swap2(I,array,left,(left+right)/2);
   last = left;
   for(i = left + 1 ; i <= right ; i ++)
   {
      if(fabs(array[i]) > fabs(array[left]))
      {
         hypre_swap2(I,array,++last,i);
      }
   }
   hypre_swap2(I,array,left,last);
   hypre_ILUMaxQSplitRabsI(array,I,left,bound,last-1);
   if(bound > last)
   {
       hypre_ILUMaxQSplitRabsI(array,I,last+1,bound,right);
   }
   
   return hypre_error_flag;
}

/*
 * Get perm array from parcsr matrix based on diag and offdiag matrix
 * Just simply loop through the rows of offd of A, check for nonzero rows
 * Put interial nodes at the beginning
 * A: parcer matrix
 * perm: permutation array
 * nLU: number of interial nodes
 */
HYPRE_Int
hypre_ILUGetPerm(hypre_ParCSRMatrix *A, HYPRE_Int **perm, HYPRE_Int *nLU)
{
   /* get basic information of A */
   HYPRE_Int n = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int i, j, first, last, start, end;
   HYPRE_Int num_sends, send_map_start, send_map_end, col;
   hypre_CSRMatrix *A_offd;
   HYPRE_Int *A_offd_i;
   A_offd = hypre_ParCSRMatrixOffd(A);
   A_offd_i = hypre_CSRMatrixI(A_offd);
   first = 0;
   last = n - 1;
   HYPRE_Int *temp_perm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   HYPRE_Int *marker = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   
   /* first get col nonzero from com_pkg */
   /* get comm_pkg, craete one if we not yet have one */
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A); 
      comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }
   
   /* now directly take adavantage of comm_pkg */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   for( i = 0 ; i < num_sends ; i ++ )
   {
      send_map_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
      send_map_end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1);
      for ( j = send_map_start ; j < send_map_end ; j ++)
      {
         col = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         if(marker[col] == 0)
         {
            temp_perm[last--] = col;
            marker[col] = -1;
         }
      }
   }
   
   /* now deal with the row */
   for( i = 0 ; i < n ; i ++)
   {
      if(marker[i] == 0)
      {
         start = A_offd_i[i];
         end = A_offd_i[i+1];
         if(start == end)
         {
            temp_perm[first++] = i;
         }
         else
         {
            temp_perm[last--] = i;
         }
      }
   }
   
   /* set out values */
   *nLU = first;
   if(*perm != NULL) hypre_TFree(*perm,HYPRE_MEMORY_HOST);
   *perm = temp_perm;
   
   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}




/* NSH create and solve and help functions */

/* Create */
void *
hypre_NSHCreate()
{
   hypre_ParNSHData  *nsh_data;

   nsh_data = hypre_CTAlloc(hypre_ParNSHData,  1, HYPRE_MEMORY_HOST);
   
   /* general data */
   hypre_ParNSHDataMatA(nsh_data) = NULL;
   hypre_ParNSHDataMatM(nsh_data) = NULL;
   hypre_ParNSHDataF(nsh_data) = NULL;
   hypre_ParNSHDataU(nsh_data) = NULL;
   hypre_ParNSHDataResidual(nsh_data) = NULL;
   hypre_ParNSHDataRelResNorms(nsh_data) = NULL;
   hypre_ParNSHDataNumIterations(nsh_data) = 0;
   hypre_ParNSHDataL1Norms(nsh_data) = NULL;
   hypre_ParNSHDataFinalRelResidualNorm(nsh_data) = 0.0;
   hypre_ParNSHDataTol(nsh_data) = 1e-09;
   hypre_ParNSHDataLogging(nsh_data) = 2;
   hypre_ParNSHDataPrintLevel(nsh_data) = 2;
   hypre_ParNSHDataMaxIter(nsh_data) = 5;
   
   hypre_ParNSHDataOperatorComplexity(nsh_data) = 0.0;
   hypre_ParNSHDataDroptol(nsh_data) = hypre_TAlloc(HYPRE_Real,2,HYPRE_MEMORY_HOST);
   hypre_ParNSHDataOwnDroptolData(nsh_data) = 1;
   hypre_ParNSHDataDroptol(nsh_data)[0] = 1.0e-02;/* droptol for MR */
   hypre_ParNSHDataDroptol(nsh_data)[1] = 1.0e-02;/* droptol for NSH */
   hypre_ParNSHDataUTemp(nsh_data) = NULL;
   hypre_ParNSHDataFTemp(nsh_data) = NULL;
   
   /* MR data */
   hypre_ParNSHDataMRMaxIter(nsh_data) = 5;
   hypre_ParNSHDataMRTol(nsh_data) = 1e-09;
   hypre_ParNSHDataMRMaxRowNnz(nsh_data) = 800;
   hypre_ParNSHDataMRColVersion(nsh_data) = 0;
   
   /* NSH data */
   hypre_ParNSHDataNSHMaxIter(nsh_data) = 2;
   hypre_ParNSHDataNSHTol(nsh_data) = 1e-09;
   hypre_ParNSHDataNSHMaxRowNnz(nsh_data) = 1000;
   
   return (void *) nsh_data;
}

/* Destroy */
HYPRE_Int
hypre_NSHDestroy( void *data )
{
   hypre_ParNSHData * nsh_data = (hypre_ParNSHData*) data;
   
   /* residual */ 
   if(hypre_ParNSHDataResidual(nsh_data))
   {
      hypre_ParVectorDestroy( hypre_ParNSHDataResidual(nsh_data) );
      hypre_ParNSHDataResidual(nsh_data) = NULL;
   }
   
   /* residual norms */
   if(hypre_ParNSHDataRelResNorms(nsh_data))
   {
      hypre_TFree( hypre_ParNSHDataRelResNorms(nsh_data), HYPRE_MEMORY_HOST );
      hypre_ParNSHDataRelResNorms(nsh_data) = NULL;
   }
   
   /* l1 norms */
   if(hypre_ParNSHDataL1Norms(nsh_data))
   {
      hypre_TFree( hypre_ParNSHDataL1Norms(nsh_data), HYPRE_MEMORY_HOST );
      hypre_ParNSHDataL1Norms(nsh_data) = NULL;
   }
   
   /* temp arrays */ 
   if(hypre_ParNSHDataUTemp(nsh_data))
   {
      hypre_ParVectorDestroy( hypre_ParNSHDataUTemp(nsh_data) );
      hypre_ParNSHDataUTemp(nsh_data) = NULL;
   }
   if(hypre_ParNSHDataFTemp(nsh_data))
   {
      hypre_ParVectorDestroy( hypre_ParNSHDataFTemp(nsh_data) );
      hypre_ParNSHDataFTemp(nsh_data) = NULL;
   }
   
   /* approx inverse matrix */
   if(hypre_ParNSHDataMatM(nsh_data))
   {
      hypre_ParCSRMatrixDestroy( hypre_ParNSHDataMatM(nsh_data) );
      hypre_ParNSHDataMatM(nsh_data) = NULL;
   }
   
   /* droptol array */
  if(hypre_ParNSHDataOwnDroptolData(nsh_data))
  {
     hypre_TFree(hypre_ParNSHDataDroptol(nsh_data), HYPRE_MEMORY_HOST);
     hypre_ParNSHDataOwnDroptolData(nsh_data) = 0;
     hypre_ParNSHDataDroptol(nsh_data) = NULL;
  }

   /* nsh data */
   hypre_TFree(nsh_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* Print solver params */
HYPRE_Int
hypre_NSHWriteSolverParams(void *nsh_vdata)
{
   hypre_ParNSHData  *nsh_data = (hypre_ParNSHData*) nsh_vdata;      
   hypre_printf("Newton–Schulz–Hotelling Setup parameters: \n");
   hypre_printf("NSH max iterations = %d \n", hypre_ParNSHDataNSHMaxIter(nsh_data));
   hypre_printf("NSH drop tolerance = %e \n", hypre_ParNSHDataDroptol(nsh_data)[1]);
   hypre_printf("NSH max nnz per row = %d \n", hypre_ParNSHDataNSHMaxRowNnz(nsh_data));
   hypre_printf("MR max iterations = %d \n", hypre_ParNSHDataMRMaxIter(nsh_data));
   hypre_printf("MR drop tolerance = %e \n", hypre_ParNSHDataDroptol(nsh_data)[0]);
   hypre_printf("MR max nnz per row = %d \n", hypre_ParNSHDataMRMaxRowNnz(nsh_data));
   hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParNSHDataOperatorComplexity(nsh_data));
   hypre_printf("\n Newton–Schulz–Hotelling Solver Parameters: \n");  
   hypre_printf("Max number of iterations: %d\n", hypre_ParNSHDataMaxIter(nsh_data));
   hypre_printf("Stopping tolerance: %e\n", hypre_ParNSHDataTol(nsh_data));
   
   return hypre_error_flag;
}

/* set print level */
HYPRE_Int
hypre_NSHSetPrintLevel( void *nsh_vdata, HYPRE_Int print_level )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataPrintLevel(nsh_data) = print_level;
   return hypre_error_flag;
}
/* set logging level */
HYPRE_Int
hypre_NSHSetLogging( void *nsh_vdata, HYPRE_Int logging )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataLogging(nsh_data) = logging;
   return hypre_error_flag;
}
/* set max iteration */
HYPRE_Int
hypre_NSHSetMaxIter( void *nsh_vdata, HYPRE_Int max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMaxIter(nsh_data) = max_iter;
   return hypre_error_flag;
}
/* set solver iteration tol */
HYPRE_Int
hypre_NSHSetTol( void *nsh_vdata, HYPRE_Real tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataTol(nsh_data) = tol;
   return hypre_error_flag;
}
/* set global solver */
HYPRE_Int
hypre_NSHSetGlobalSolver( void *nsh_vdata, HYPRE_Int global_solver )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataGlobalSolver(nsh_data) = global_solver;
   return hypre_error_flag;
}
/* set all droptols */
HYPRE_Int
hypre_NSHSetDropThreshold( void *nsh_vdata, HYPRE_Real droptol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataDroptol(nsh_data)[0] = droptol;
   hypre_ParNSHDataDroptol(nsh_data)[1] = droptol;
   return hypre_error_flag;
}
/* set array of droptols */
HYPRE_Int
hypre_NSHSetDropThresholdArray( void *nsh_vdata, HYPRE_Real *droptol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   if(hypre_ParNSHDataOwnDroptolData(nsh_data))
   {
      hypre_TFree(hypre_ParNSHDataDroptol(nsh_data),HYPRE_MEMORY_HOST);
      hypre_ParNSHDataOwnDroptolData(nsh_data) = 0;
   }
   hypre_ParNSHDataDroptol(nsh_data) = droptol;
   return hypre_error_flag;
}
/* set own data */
HYPRE_Int
hypre_NSHSetOwnDroptolData( void *nsh_vdata, HYPRE_Int own_droptol_data )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataOwnDroptolData(nsh_data) = own_droptol_data;
   return hypre_error_flag;
}
/* set MR max iter */
HYPRE_Int
hypre_NSHSetMRMaxIter( void *nsh_vdata, HYPRE_Int mr_max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRMaxIter(nsh_data) = mr_max_iter;
   return hypre_error_flag;
}
/* set MR tol */
HYPRE_Int
hypre_NSHSetMRTol( void *nsh_vdata, HYPRE_Real mr_tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRTol(nsh_data) = mr_tol;
   return hypre_error_flag;
}
/* set MR max nonzeros of a row */
HYPRE_Int
hypre_NSHSetMRMaxRowNnz( void *nsh_vdata, HYPRE_Int mr_max_row_nnz )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRMaxRowNnz(nsh_data) = mr_max_row_nnz;
   return hypre_error_flag;
}
/* set MR version, column version or global version */
HYPRE_Int
hypre_NSHSetColVersion( void *nsh_vdata, HYPRE_Int mr_col_version )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRColVersion(nsh_data) = mr_col_version;
   return hypre_error_flag;
}
/* set NSH max iter */
HYPRE_Int
hypre_NSHSetNSHMaxIter( void *nsh_vdata, HYPRE_Int nsh_max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHMaxIter(nsh_data) = nsh_max_iter;
   return hypre_error_flag;
}
/* set NSH tol */
HYPRE_Int
hypre_NSHSetNSHTol( void *nsh_vdata, HYPRE_Real nsh_tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHTol(nsh_data) = nsh_tol;
   return hypre_error_flag;
}
/* set NSH max nonzeros of a row */
HYPRE_Int
hypre_NSHSetNSHMaxRowNnz( void *nsh_vdata, HYPRE_Int nsh_max_row_nnz )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHMaxRowNnz(nsh_data) = nsh_max_row_nnz;
   return hypre_error_flag;
}


/* Compute the F norm of CSR matrix
 * A: the target CSR matrix
 * norm_io: output
 */
HYPRE_Int
hypre_CSRMatrixNormFro(hypre_CSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real norm = 0.0;
   HYPRE_Real *data = hypre_CSRMatrixData(A);
   HYPRE_Int i,k;
   k = hypre_CSRMatrixNumNonzeros(A);
   /* main loop */
   for(i = 0 ; i < k ; i ++)
   {
      norm += data[i] * data[i];
   }
   *norm_io = sqrt(norm);
   return hypre_error_flag;
   
}

/* Compute the norm of I-A where I is identity matrix and A is a CSR matrix
 * A: the target CSR matrix
 * norm_io: the output
 */
HYPRE_Int
hypre_CSRMatrixResNormFro(hypre_CSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real norm = 0.0, value;
   HYPRE_Int i, j, k1, k2, n;
   HYPRE_Int *idx = hypre_CSRMatrixI(A);
   HYPRE_Int *cols = hypre_CSRMatrixJ(A);
   HYPRE_Real *data = hypre_CSRMatrixData(A);
   n = hypre_CSRMatrixNumRows(A);
   /* main loop to sum up data */
   for(i = 0 ; i < n ; i ++)
   {
      k1 = idx[i];
      k2 = idx[i+1];
      /* check if we have diagonal in A */
      if(k2 > k1)
      {
         if(cols[k1] == i)
         {
            /* reduce 1 on diagonal */
            value = data[k1] - 1.0;
            norm += value * value;
         }
         else
         {
            /* we don't have diagonal in A, so we need to add 1 to norm */
            norm += 1.0;
            norm += data[k1] * data[k1];
         }
      }
      else
      {
         /* we don't have diagonal in A, so we need to add 1 to norm */
         norm += 1.0;
      }
      /* and the rest of the code */
      for(j = k1 + 1 ; j < k2 ; j ++)
      {
         norm += data[j] * data[j];
      }
   }
   *norm_io = sqrt(norm);
   return hypre_error_flag;
}

/* Compute the F norm of ParCSR matrix
 * A: the target CSR matrix
 */
HYPRE_Int
hypre_ParCSRMatrixNormFro(hypre_ParCSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real local_norm = 0.0;
   HYPRE_Real global_norm;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   
   hypre_CSRMatrixNormFro(A_diag, &local_norm);
   /* use global_norm to store offd for now */
   hypre_CSRMatrixNormFro(A_offd, &global_norm);
   
   /* square and sum them */
   local_norm *= local_norm;
   local_norm += global_norm*global_norm;
   
   /* do communication to get global total sum */
   hypre_MPI_Allreduce(&local_norm, &global_norm, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

   *norm_io = sqrt(global_norm);
   return hypre_error_flag;
   
}

/* Compute the F norm of ParCSR matrix
 * Norm of I-A
 * A: the target CSR matrix
 */
HYPRE_Int
hypre_ParCSRMatrixResNormFro(hypre_ParCSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real local_norm = 0.0;
   HYPRE_Real global_norm;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   
   /* compute I-A for diagonal */
   hypre_CSRMatrixResNormFro(A_diag, &local_norm);
   /* use global_norm to store offd for now */
   hypre_CSRMatrixNormFro(A_offd, &global_norm);
   
   /* square and sum them */
   local_norm *= local_norm;
   local_norm += global_norm*global_norm;
   
   /* do communication to get global total sum */
   hypre_MPI_Allreduce(&local_norm, &global_norm, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

   *norm_io = sqrt(global_norm);
   return hypre_error_flag;
   
}

/* Compute the trace of CSR matrix
 * A: the target CSR matrix
 * trace_io: the output trace
 */
HYPRE_Int
hypre_CSRMatrixTrace(hypre_CSRMatrix *A, HYPRE_Real *trace_io)
{
   HYPRE_Real trace = 0.0;
   HYPRE_Int *idx = hypre_CSRMatrixI(A);
   HYPRE_Int *cols = hypre_CSRMatrixJ(A);
   HYPRE_Real *data = hypre_CSRMatrixData(A);
   HYPRE_Int i,k1,k2,n;
   n = hypre_CSRMatrixNumRows(A);
   for(i = 0 ; i < n ; i ++)
   {
      k1 = idx[i];
      k2 = idx[i+1];
      if(cols[k1] == i && k2 > k1)
      {
         /* only add when diagonal is nonzero */
         trace += data[k1];
      }
   }
   
   *trace_io = trace;
   return hypre_error_flag;
   
}

/* Scale CSR matrix A = scalar * A
 * A: the target CSR matrix
 * scalar: real number
 */
HYPRE_Int
hypre_CSRMatrixScale(hypre_CSRMatrix *A, HYPRE_Real scalar)
{
   HYPRE_Real *data = hypre_CSRMatrixData(A);
   HYPRE_Int i,k;
   k = hypre_CSRMatrixNumNonzeros(A);
   for(i = 0 ; i < k ; i ++)
   {
      data[i] *= scalar;
   }
   return hypre_error_flag;
}

/* Scale ParCSR matrix A = scalar * A
 * A: the target CSR matrix
 * scalar: real number
 */
HYPRE_Int
hypre_ParCSRMatrixScale(hypre_ParCSRMatrix *A, HYPRE_Real scalar)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   /* each thread scale local diag and offd */
   hypre_CSRMatrixScale(A_diag, scalar);
   hypre_CSRMatrixScale(A_offd, scalar);
   return hypre_error_flag;
}

/* Apply dropping to CSR matrix
 * A: the target CSR matrix
 * droptol: all entries have smaller absolute value than this will be dropped
 * max_row_nnz: max nonzoers allowed for each row, only largest max_row_nnz kept
 * we NEVER drop diagonal entry if exists
 */
HYPRE_Int
hypre_CSRMatrixDropInplace(hypre_CSRMatrix *A, HYPRE_Real droptol, HYPRE_Int max_row_nnz)
{
   HYPRE_Int      i, j, k1, k2, has_diag;
   HYPRE_Int      *idx, len, drop_len;
   HYPRE_Real     *data, value, itol, norm;
   
   /* info of matrix A */
   HYPRE_Int      n = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      m = hypre_CSRMatrixNumCols(A);
   HYPRE_Int      *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int      *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real     *A_data = hypre_CSRMatrixData(A);
   HYPRE_Real     nnzA = hypre_CSRMatrixNumNonzeros(A);
   
   /* new data */
   HYPRE_Int      *new_i;
   HYPRE_Int      *new_j;
   HYPRE_Real     *new_data;
   
   /* memory */
   HYPRE_Int      capacity;
   HYPRE_Int      ctrA;
   
   /* setup */
   capacity = nnzA*0.3+1;
   ctrA = 0;
   new_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   new_j = hypre_TAlloc(HYPRE_Int, capacity, HYPRE_MEMORY_HOST);
   new_data = hypre_TAlloc(HYPRE_Real, capacity, HYPRE_MEMORY_HOST);
   
   idx = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_HOST);
   data = hypre_TAlloc(HYPRE_Real, m, HYPRE_MEMORY_HOST);
   
   /* start of main loop */
   new_i[0] = 0;
   for(i = 0 ; i < n ; i ++)
   {
      len = 0;
      k1 = A_i[i];
      k2 = A_i[i+1];
      /* compute droptol for current row */
      norm = 0.0;
      for(j = k1 ; j < k2 ; j ++)
      {
         norm += fabs(A_data[j]);
      }
      if(k2 > k1)
      {
         norm /= (HYPRE_Real)(k2 - k1);
      }
      itol = droptol * norm;
      /* we don't want to drop the diagonal entry, so use an if statement here */
      if(A_j[k1] == i)
      {
         /* we have diagonal entry, skip it */
         idx[len] = A_j[k1];
         data[len++] = A_data[k1];
         for(j = k1 + 1 ; j < k2 ; j ++)
         {
            value = A_data[j];
            if(fabs(value) < itol)
            {
               /* skip small element */
               continue;
            }
            idx[len] = A_j[j];
            data[len++] = A_data[j];
         }
         
         /* now apply drop on length */
         if(len > max_row_nnz)
         {
            drop_len = max_row_nnz;
            hypre_ILUMaxQSplitRabsI( data + 1, idx + 1, 0, drop_len - 1 , len - 2);
         }
         else
         {
            /* don't need to sort, we keep all of them */
            drop_len = len;
         }
         
         /* copy data */
         while(ctrA + drop_len > capacity)
         {
            capacity = capacity * EXPAND_FACT + 1;
            new_j = hypre_TReAlloc(new_j, HYPRE_Int, capacity, HYPRE_MEMORY_HOST);
            new_data = hypre_TReAlloc(new_data, HYPRE_Real, capacity, HYPRE_MEMORY_HOST);
         }
         hypre_TMemcpy( new_j + ctrA, idx,HYPRE_Int, drop_len, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_TMemcpy( new_data + ctrA, data,HYPRE_Real, drop_len, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         ctrA += drop_len;
         new_i[i+1] = ctrA;
      }
      else
      {
         /* we don't have diagonal entry */
         for(j = k1 ; j < k2 ; j ++)
         {
            value = A_data[j];
            if(fabs(value)<itol)
            {
               /* skip small element */
               continue;
            }
            idx[len] = A_j[j];
            data[len++] = A_data[j];
         }
         
         /* now apply drop on length */
         if(len > max_row_nnz)
         {
            drop_len = max_row_nnz;
            hypre_ILUMaxQSplitRabsI( data, idx, 0, drop_len, len - 1);
         }
         else
         {
            /* don't need to sort, we keep all of them */
            drop_len = len;
         }
         
         /* copy data */
         while(ctrA + drop_len > capacity)
         {
            capacity = capacity * EXPAND_FACT + 1;
            new_j = hypre_TReAlloc(new_j, HYPRE_Int, capacity, HYPRE_MEMORY_HOST);
            new_data = hypre_TReAlloc(new_data, HYPRE_Real, capacity, HYPRE_MEMORY_HOST);
         }
         hypre_TMemcpy( new_j + ctrA, idx,HYPRE_Int, drop_len, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_TMemcpy( new_data + ctrA, data,HYPRE_Real, drop_len, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         ctrA += drop_len;
         new_i[i+1] = ctrA;
      }
   }/* end of main loop */
   /* destory data if A own them */
   if(hypre_CSRMatrixOwnsData(A))
   {
      hypre_TFree(A_i, HYPRE_MEMORY_HOST);
      hypre_TFree(A_j, HYPRE_MEMORY_HOST);
      hypre_TFree(A_data, HYPRE_MEMORY_HOST);
   }
   
   hypre_CSRMatrixI(A) = new_i;
   hypre_CSRMatrixJ(A) = new_j;
   hypre_CSRMatrixData(A) = new_data;
   hypre_CSRMatrixNumNonzeros(A) = ctrA;
   hypre_CSRMatrixOwnsData(A) = 1;
   
   hypre_TFree(idx, HYPRE_MEMORY_HOST);
   hypre_TFree(data, HYPRE_MEMORY_HOST);
   
   return hypre_error_flag;
}

/* Compute the inverse with MR of original CSR matrix
 * Global(not by each column) and out place version
 * A: the input matrix
 * M: the output matrix
 * droptol: the dropping tolorance
 * tol: when to stop the iteration
 * eps_tol: to avoid divide by 0
 * max_row_nnz: max number of nonzeros per row
 * max_iter: max number of iterations
 * print_level: the print level of this algorithm
 */
HYPRE_Int
hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(hypre_CSRMatrix *matA, hypre_CSRMatrix **M, HYPRE_Real droptol, 
                                               HYPRE_Real tol, HYPRE_Real eps_tol, HYPRE_Int max_row_nnz, HYPRE_Int max_iter, 
                                               HYPRE_Int print_level )
{
   HYPRE_Int         i, k1, k2, j;
   HYPRE_Real        value, trace1, trace2, alpha, r_norm, z_norm;
   
   /* martix A */
   HYPRE_Int         *A_i = hypre_CSRMatrixI(matA);
   HYPRE_Int         *A_j = hypre_CSRMatrixJ(matA);
   HYPRE_Real        *A_data = hypre_CSRMatrixData(matA);
   
   /* complexity */
   HYPRE_Real        nnzA = hypre_CSRMatrixNumNonzeros(matA);
   HYPRE_Real        nnzM;
   
   /* inverse matrix */
   hypre_CSRMatrix   *inM = *M;
   hypre_CSRMatrix   *matM;
   HYPRE_Int         *M_i;
   HYPRE_Int         *M_j;
   HYPRE_Real        *M_data;
   
   /* idendity matrix */
   hypre_CSRMatrix   *matI;
   HYPRE_Int         *I_i;
   HYPRE_Int         *I_j;
   HYPRE_Real        *I_data;
   
   /* helper matrices */
   hypre_CSRMatrix   *matR;
   hypre_CSRMatrix   *matR_temp;
   hypre_CSRMatrix   *matZ;
   hypre_CSRMatrix   *matC;
   hypre_CSRMatrix   *matW;
   
   HYPRE_Real        time_s, time_e;
   
   HYPRE_Int         n = hypre_CSRMatrixNumRows(matA);
   
   if(n == 0)
   {
      *M = NULL;
      return hypre_error_flag;
   }
   
   /* create initial guess and matrix I */
   matM = hypre_CSRMatrixCreate(n,n,n);
   M_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   M_j = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   M_data = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   
   matI = hypre_CSRMatrixCreate(n,n,n);
   I_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   I_j = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   I_data = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   
   /* now loop to create initial guess */
   M_i[0] = 0;
   I_i[0] = 0;
   for(i = 0 ; i < n ; i ++)
   {
      M_i[i+1] = i+1;
      M_j[i] = i;
      k1 = A_i[i];
      k2 = A_i[i+1];
      if(k2 > k1)
      {
         if(A_j[k1] == i)
         {
            value = A_data[k1];
            if(fabs(value) < MAT_TOL)
            {
               value = 1.0;
            }
            M_data[i] = 1.0/value;
         }
         else
         {
            M_data[i] = 1.0;
         }
      }
      else
      {
         M_data[i] = 1.0;
      }
      I_i[i+1] = i+1;
      I_j[i] = i;
      I_data[i] = 1.0;
   }
   
   hypre_CSRMatrixI(matM) = M_i;
   hypre_CSRMatrixJ(matM) = M_j;
   hypre_CSRMatrixData(matM) = M_data;
   hypre_CSRMatrixOwnsData(matM) = 1;
   
   hypre_CSRMatrixI(matI) = I_i;
   hypre_CSRMatrixJ(matI) = I_j;
   hypre_CSRMatrixData(matI) = I_data;
   hypre_CSRMatrixOwnsData(matI) = 1;
   
   /* now start the main loop */
   if(print_level > 1)
   {
      /* time the iteration */
      time_s = hypre_MPI_Wtime();
   }
   
   /* main loop */
   for(i = 0 ; i < max_iter ; i ++)
   {
      nnzM = hypre_CSRMatrixNumNonzeros(matM);
      /* R = I - AM */
      matR_temp = hypre_CSRMatrixMultiply(matA,matM);
      
      hypre_CSRMatrixScale(matR_temp, -1.0);
      
      matR = hypre_CSRMatrixAdd(matI,matR_temp);
      hypre_CSRMatrixDestroy(matR_temp);
      
      /* r_norm */
      hypre_CSRMatrixNormFro(matR, &r_norm);
      if(r_norm < tol)
      {
         break;
      }
      
      /* Z = MR and dropping */
      matZ = hypre_CSRMatrixMultiply(matM, matR);
      //hypre_CSRMatrixNormFro(matZ, &z_norm);
      hypre_CSRMatrixDropInplace(matZ, droptol, max_row_nnz);
      
      /* C = A*Z */
      matC = hypre_CSRMatrixMultiply(matA, matZ);
      
      /* W = R' * C */
      hypre_CSRMatrixTranspose(matR,&matR_temp,1);
      matW = hypre_CSRMatrixMultiply(matR_temp,matC);
      
      /* trace and alpha */
      hypre_CSRMatrixTrace(matW, &trace1);
      hypre_CSRMatrixNormFro(matC, &trace2);
      trace2 *= trace2;
      
      if(fabs(trace2) < eps_tol)
      {
         break;
      }
      
      alpha = trace1 / trace2;
      
      /* M - M + alpha * Z */
      hypre_CSRMatrixScale(matZ, alpha);
      
      hypre_CSRMatrixDestroy(matR);
      matR = hypre_CSRMatrixAdd(matM, matZ);
      hypre_CSRMatrixDestroy(matM);
      matM = matR;
      
      hypre_CSRMatrixDestroy(matZ);
      hypre_CSRMatrixDestroy(matW);
      hypre_CSRMatrixDestroy(matC);
      hypre_CSRMatrixDestroy(matR_temp);
      
   }/* end of main loop i for compute inverse matrix */
   
   /* time if we need to print */
   if(print_level > 1)
   {
      time_e = hypre_MPI_Wtime();
      if(i == 0)
      {
         i = 1;
      }
      printf("matrix size %5d\nfinal norm at loop %5d is %16.12f, time per iteration is %16.12f, complexity is %16.12f out of maximum %16.12f\n",n,i,r_norm, (time_e-time_s)/i, nnzM/nnzA, n/nnzA*n);
   }
   
   hypre_CSRMatrixDestroy(matI);
   if(inM)
   {
      hypre_CSRMatrixDestroy(inM);
   }
   *M = matM;
   
   return hypre_error_flag;
   
}

/* Compute inverse with NSH method
 * Use MR to get local initial guess
 * A: input matrix
 * M: output matrix
 * droptol: droptol array. droptol[0] for MR and droptol[1] for NSH.
 * mr_tol: tol for stop iteration for MR
 * nsh_tol: tol for stop iteration for NSH
 * esp_tol: tol for avoid divide by 0
 * mr_max_row_nnz: max number of nonzeros for MR
 * nsh_max_row_nnz: max number of nonzeros for NSH
 * mr_max_iter: max number of iterations for MR
 * nsh_max_iter: max number of iterations for NSH
 * mr_col_version: column version of global version
 */
HYPRE_Int
hypre_ILUParCSRInverseNSH(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **M, HYPRE_Real *droptol, HYPRE_Real mr_tol, 
                            HYPRE_Real nsh_tol, HYPRE_Real eps_tol, HYPRE_Int mr_max_row_nnz, HYPRE_Int nsh_max_row_nnz, 
                            HYPRE_Int mr_max_iter, HYPRE_Int nsh_max_iter, HYPRE_Int mr_col_version,
                            HYPRE_Int print_level)
{
   HYPRE_Int               i;
   
   /* data slots for matrices */
   hypre_ParCSRMatrix      *matM = NULL;
   hypre_ParCSRMatrix      *inM = *M;
   hypre_ParCSRMatrix      *AM,*MAM;
   HYPRE_Real              norm, s_norm;
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int               myid;
   
   
   hypre_CSRMatrix         *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *M_diag = NULL;
   hypre_CSRMatrix         *M_offd;
   HYPRE_Int               *M_offd_i;
   
   HYPRE_Real              time_s, time_e;
   
   HYPRE_Int               n = hypre_CSRMatrixNumRows(A_diag);
   
   /* setup */
   hypre_MPI_Comm_rank(comm, &myid);
   
   M_offd_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   
   if(mr_col_version)
   {
      printf("Column version is not yet support, switch to global version\n");
   }
   
   /* call MR to build loacl initial matrix 
    * droptol here should be larger
    * we want same number for MR and NSH to let user set them eaiser
    * but we don't want a too dense MR initial guess
    */
   hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(A_diag, &M_diag, droptol[0] * 5.0, mr_tol, eps_tol, mr_max_row_nnz, mr_max_iter, print_level );
   
   /* create empty offdiagonal */
   for(i = 0 ; i <= n ; i ++)
   {
      M_offd_i[i] = 0;
   }
   
   /* create parCSR matM */
   matM = hypre_ParCSRMatrixCreate( comm,
                       hypre_ParCSRMatrixGlobalNumRows(A),
                       hypre_ParCSRMatrixGlobalNumRows(A),
                       hypre_ParCSRMatrixRowStarts(A),
                       hypre_ParCSRMatrixColStarts(A),
                       0,
                       hypre_CSRMatrixNumNonzeros(M_diag),
                       0 );

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matM));
   hypre_ParCSRMatrixDiag(matM) = M_diag;
   
   M_offd = hypre_ParCSRMatrixOffd(matM);
   hypre_CSRMatrixI(M_offd) = M_offd_i;
   hypre_CSRMatrixOwnsData(M_offd) = 1;
   
   hypre_ParCSRMatrixSetColStartsOwner(matM,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matM,0);
   
   /* now start NSH 
    * Mj+1 = 2Mj - MjAMj
    */
   
   AM = hypre_ParMatmul(A, matM);
   hypre_ParCSRMatrixResNormFro(AM, &norm);
   s_norm = norm;
   hypre_ParCSRMatrixDestroy(AM);
   if(print_level > 1)
   {
      if(myid == 0)
      {
         printf("before NSH the norm is %16.12f\n", norm);
      }
      time_s = hypre_MPI_Wtime();
   }
   
   for(i = 0 ; i < nsh_max_iter ; i ++)
   {
      /* compute XjAXj */
      AM = hypre_ParMatmul(A, matM);
      hypre_ParCSRMatrixResNormFro(AM, &norm);
      if(norm < nsh_tol)
      {
         break;
      }
      MAM = hypre_ParMatmul(matM, AM);
      hypre_ParCSRMatrixDestroy(AM);
      
      /* apply dropping */
      //hypre_ParCSRMatrixNormFro(MAM, &norm);
      /* this function has already built with norm */
      hypre_ParCSRMatrixDropSmallEntries(MAM, droptol[1]);
      
      /* update Mj+1 = 2Mj - MjAMj 
       * the result holds it own start/end data!
       */
      hypre_ParcsrAdd(2.0, matM,-1.0, MAM, &AM);
      hypre_ParCSRMatrixDestroy(matM);
      matM = AM;
      
      /* destroy */
      hypre_ParCSRMatrixDestroy(MAM);
   }
   
   if(print_level > 1)
   {
      time_e = hypre_MPI_Wtime();
      /* at this point of time, norm has to be already computed */
      if(i == 0)
      {
         i = 1;
      }
      if(myid == 0)
      {
         printf("after %5d NSH iterations the norm is %16.12f, time per iteration is %16.12f\n", i, norm, (time_e-time_s)/i);
      }
   }
   
   if(s_norm < norm)
   {
      /* the residual norm increase after NSH iteration, need to let user know */
      if(myid == 0)
      {
         printf("Warning: NSH divergence, probably bad approximate invese matrix.\n");
      }
   }
   
   if(inM)
   {
      hypre_ParCSRMatrixDestroy(inM);
   }
   *M = matM;
   
   return hypre_error_flag;
}
