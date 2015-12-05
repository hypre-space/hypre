/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Member functions for hypre_AuxParCSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "aux_parcsr_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AuxParCSRMatrixCreate( hypre_AuxParCSRMatrix **aux_matrix,
			     HYPRE_Int  local_num_rows,
                       	     HYPRE_Int  local_num_cols,
			     HYPRE_Int *sizes)
{
   hypre_AuxParCSRMatrix  *matrix;
   
   matrix = hypre_CTAlloc(hypre_AuxParCSRMatrix, 1);
  
   hypre_AuxParCSRMatrixLocalNumRows(matrix) = local_num_rows;
   hypre_AuxParCSRMatrixLocalNumCols(matrix) = local_num_cols;

   if (sizes)
   {
      hypre_AuxParCSRMatrixRowSpace(matrix) = sizes;
   }
   else
   {
      hypre_AuxParCSRMatrixRowSpace(matrix) = NULL;
   }

   /* set defaults */
   hypre_AuxParCSRMatrixNeedAux(matrix) = 1;
   hypre_AuxParCSRMatrixMaxOffProcElmts(matrix) = 0;
   hypre_AuxParCSRMatrixCurrentNumElmts(matrix) = 0;
   hypre_AuxParCSRMatrixOffProcIIndx(matrix) = 0;
   hypre_AuxParCSRMatrixRowLength(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxJ(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxData(matrix) = NULL;
   hypre_AuxParCSRMatrixIndxDiag(matrix) = NULL;
   hypre_AuxParCSRMatrixIndxOffd(matrix) = NULL;
   /* stash for setting or adding off processor values */
   hypre_AuxParCSRMatrixOffProcI(matrix) = NULL;
   hypre_AuxParCSRMatrixOffProcJ(matrix) = NULL;
   hypre_AuxParCSRMatrixOffProcData(matrix) = NULL;


   *aux_matrix = matrix;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_AuxParCSRMatrixDestroy( hypre_AuxParCSRMatrix *matrix )
{
   HYPRE_Int ierr=0;
   HYPRE_Int i;
   HYPRE_Int num_rows;

   if (matrix)
   {
      num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);
      if (hypre_AuxParCSRMatrixRowLength(matrix))
         hypre_TFree(hypre_AuxParCSRMatrixRowLength(matrix));
      if (hypre_AuxParCSRMatrixRowSpace(matrix))
         hypre_TFree(hypre_AuxParCSRMatrixRowSpace(matrix));
      if (hypre_AuxParCSRMatrixAuxJ(matrix))
      {
         for (i=0; i < num_rows; i++)
	    hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix)[i]);
	 hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix));
      }
      if (hypre_AuxParCSRMatrixAuxData(matrix))
      {
         for (i=0; i < num_rows; i++)
            hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix)[i]);
	 hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix));
      }
      if (hypre_AuxParCSRMatrixIndxDiag(matrix))
            hypre_TFree(hypre_AuxParCSRMatrixIndxDiag(matrix));
      if (hypre_AuxParCSRMatrixIndxOffd(matrix))
            hypre_TFree(hypre_AuxParCSRMatrixIndxOffd(matrix));
      if (hypre_AuxParCSRMatrixOffProcI(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcI(matrix));
      if (hypre_AuxParCSRMatrixOffProcJ(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcJ(matrix));
      if (hypre_AuxParCSRMatrixOffProcData(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcData(matrix));
      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_AuxParCSRMatrixInitialize( hypre_AuxParCSRMatrix *matrix )
{
   HYPRE_Int local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);
   HYPRE_Int *row_space = hypre_AuxParCSRMatrixRowSpace(matrix);
   HYPRE_Int max_off_proc_elmts = hypre_AuxParCSRMatrixMaxOffProcElmts(matrix);
   HYPRE_Int **aux_j;
   double **aux_data;
   HYPRE_Int i;

   if (local_num_rows < 0) 
      return -1;
   if (local_num_rows == 0) 
      return 0;
   /* allocate stash for setting or adding off processor values */
   if (max_off_proc_elmts > 0)
   {
      hypre_AuxParCSRMatrixOffProcI(matrix) = hypre_CTAlloc(HYPRE_Int,
		2*max_off_proc_elmts);
      hypre_AuxParCSRMatrixOffProcJ(matrix) = hypre_CTAlloc(HYPRE_Int,
		max_off_proc_elmts);
      hypre_AuxParCSRMatrixOffProcData(matrix) = hypre_CTAlloc(double,
		max_off_proc_elmts);
   }
   if (hypre_AuxParCSRMatrixNeedAux(matrix))
   {
      aux_j = hypre_CTAlloc(HYPRE_Int *, local_num_rows);
      aux_data = hypre_CTAlloc(double *, local_num_rows);
      if (!hypre_AuxParCSRMatrixRowLength(matrix))
         hypre_AuxParCSRMatrixRowLength(matrix) = 
  		hypre_CTAlloc(HYPRE_Int, local_num_rows);
      if (row_space)
      {
         for (i=0; i < local_num_rows; i++)
         {
            aux_j[i] = hypre_CTAlloc(HYPRE_Int, row_space[i]);
            aux_data[i] = hypre_CTAlloc(double, row_space[i]);
         }
      }
      else
      {
         row_space = hypre_CTAlloc(HYPRE_Int, local_num_rows);
         for (i=0; i < local_num_rows; i++)
         {
            row_space[i] = 30;
            aux_j[i] = hypre_CTAlloc(HYPRE_Int, 30);
            aux_data[i] = hypre_CTAlloc(double, 30);
         }
         hypre_AuxParCSRMatrixRowSpace(matrix) = row_space;
      }
      hypre_AuxParCSRMatrixAuxJ(matrix) = aux_j;
      hypre_AuxParCSRMatrixAuxData(matrix) = aux_data;
   }
   else
   {
      hypre_AuxParCSRMatrixIndxDiag(matrix) = hypre_CTAlloc(HYPRE_Int,local_num_rows);
      hypre_AuxParCSRMatrixIndxOffd(matrix) = hypre_CTAlloc(HYPRE_Int,local_num_rows);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_AuxParCSRMatrixSetMaxOffPRocElmts( hypre_AuxParCSRMatrix *matrix,
					 HYPRE_Int max_off_proc_elmts )
{
   HYPRE_Int ierr = 0;
   hypre_AuxParCSRMatrixMaxOffProcElmts(matrix) = max_off_proc_elmts;
   return ierr;
}

