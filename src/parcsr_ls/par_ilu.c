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
  (ilu_data -> droptol) = 1.0e-3;
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
  (ilu_data -> droptol) = threshold;

  return hypre_error_flag;
}
/* set ILU factorization type */
HYPRE_Int
hypre_ILUSetType( void *ilu_vdata, HYPRE_Int ilu_type )
{
  hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
  (ilu_data -> ilu_type) = ilu_type;

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
   Quicksort of the elements in a from low to high. The elements
   in b are permuted according to the sorted a. lo and hi are the 
   extents of the region of the array a, that is to be sorted.
*/
/* commented out to use current version hypre_qsort1(...)
HYPRE_Int 
hypre_quickSortIR (HYPRE_Int *a, HYPRE_Real *b, const HYPRE_Int lo, const HYPRE_Int hi)
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
          i++; j--;
      }
   } while (i<=j);
   //  recursion
   if (lo<j) quickSortIR(a, b, lo, j);
   if (i<hi) quickSortIR(a, b, i, hi);
   
   return hypre_error_flag;
}
*/
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
              hypre_printf("drop tolerance = %e \n", (ilu_data -> droptol));
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

