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
   (ilu_data -> droptol)[0] = 1.0e-03;/* droptol for B */
   (ilu_data -> droptol)[1] = 1.0e-03;/* droptol for E and F */
   (ilu_data -> droptol)[2] = 1.0e-03;/* droptol for S */
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
  
   /* default data for schur solver */
   (ilu_data -> ss_kDim) = 5;
   (ilu_data -> ss_max_iter) = 5;
   (ilu_data -> ss_tol) = 0.0e-00;
   (ilu_data -> ss_absolute_tol) = 0.0;
   (ilu_data -> ss_logging) = 1;
   (ilu_data -> ss_print_level) = 0;
   (ilu_data -> ss_rel_change) = 0;
   
   /* default data for schur precond 
    * default ILUT
    */
   (ilu_data -> sp_ilu_type) = 1;
   (ilu_data -> sp_ilu_lfil) = 3;
   (ilu_data -> sp_ilu_max_row_nnz) = 1000;
   (ilu_data -> sp_own_droptol_data) = 0;
   (ilu_data -> sp_ilu_droptol) = (ilu_data -> droptol);/* use same droptol */
   (ilu_data -> sp_print_level) = 0;
   (ilu_data -> sp_max_iter) = 1;
   (ilu_data -> sp_tol) = 0.0e-00;
   
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
   (ilu_data -> ss_max_iter) = ss_max_iter;
   /* avoid restart */
   if((ilu_data -> ss_kDim) < ss_max_iter)
   {
      (ilu_data -> ss_kDim) = ss_max_iter;
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
/* Set IUL drop threshold for ILUT for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILUDropThreshold( void *ilu_vdata, HYPRE_Real sp_ilu_droptol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   (ilu_data -> sp_ilu_droptol)[0] = sp_ilu_droptol;
   (ilu_data -> sp_ilu_droptol)[1] = sp_ilu_droptol;
   (ilu_data -> sp_ilu_droptol)[2] = sp_ilu_droptol;
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
      case 10: 
              hypre_printf("GMRES with ILU(%d) \n", (ilu_data -> lfil));
              hypre_printf("Operator Complexity (Fill factor) = %f \n", (ilu_data -> operator_complexity));
         break;
      case 11: 
              hypre_printf("GMRES with ILUT \n");
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

/* Extract a sub SCR matrix from the original matrix
 * row_start and row_end: array on length 2, start and end of sub matrix's row/col
 * B: the return matrix
 * HAVENT FINISHED YET
 */
HYPRE_Int
hypre_CSRMatrixExtractSubMatrix(hypre_CSRMatrix *A, HYPRE_Int *row_start, HYPRE_Int *col_start, hypre_CSRMatrix **B)
{
   /* first get basic information of the submatrix */
   HYPRE_Int num_rows = row_start[1] - row_start[0];
   HYPRE_Int num_cols = col_start[1] - col_start[0];
   HYPRE_Int num_nonzeors = 0;
   hypre_CSRMatrix *outmat;
   //outmat = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   *B = outmat;
   return hypre_error_flag;
}
