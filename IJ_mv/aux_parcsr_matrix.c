
/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_AuxParCSRMatrix class.
 *
 *****************************************************************************/

#include "IJ_mv.h"
#include "aux_parcsr_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

int
hypre_AuxParCSRMatrixCreate( hypre_AuxParCSRMatrix **aux_matrix,
			     int  local_num_rows,
                       	     int  local_num_cols,
			     int *sizes)
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
   hypre_AuxParCSRMatrixMaxOffProcElmtsSet(matrix) = 0;
   hypre_AuxParCSRMatrixCurrentNumElmtsSet(matrix) = 0;
   hypre_AuxParCSRMatrixOffProcIIndxSet(matrix) = 0;
   hypre_AuxParCSRMatrixMaxOffProcElmtsAdd(matrix) = 0;
   hypre_AuxParCSRMatrixCurrentNumElmtsAdd(matrix) = 0;
   hypre_AuxParCSRMatrixOffProcIIndxAdd(matrix) = 0;
   hypre_AuxParCSRMatrixRowLength(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxJ(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxData(matrix) = NULL;
   hypre_AuxParCSRMatrixIndxDiag(matrix) = NULL;
   hypre_AuxParCSRMatrixIndxOffd(matrix) = NULL;
   /* stash for setting off processor values */
   hypre_AuxParCSRMatrixOffProcISet(matrix) = NULL;
   hypre_AuxParCSRMatrixOffProcJSet(matrix) = NULL;
   hypre_AuxParCSRMatrixOffProcDataSet(matrix) = NULL;
   /* stash for adding to off processor values */
   hypre_AuxParCSRMatrixOffProcIAdd(matrix) = NULL;
   hypre_AuxParCSRMatrixOffProcJAdd(matrix) = NULL;
   hypre_AuxParCSRMatrixOffProcDataAdd(matrix) = NULL;


   *aux_matrix = matrix;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParCSRMatrixDestroy( hypre_AuxParCSRMatrix *matrix )
{
   int ierr=0;
   int i;
   int num_rows;

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
      if (hypre_AuxParCSRMatrixOffProcISet(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcISet(matrix));
      if (hypre_AuxParCSRMatrixOffProcJSet(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcJSet(matrix));
      if (hypre_AuxParCSRMatrixOffProcDataSet(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcDataSet(matrix));
      if (hypre_AuxParCSRMatrixOffProcIAdd(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcIAdd(matrix));
      if (hypre_AuxParCSRMatrixOffProcJAdd(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcJAdd(matrix));
      if (hypre_AuxParCSRMatrixOffProcDataAdd(matrix))
      	    hypre_TFree(hypre_AuxParCSRMatrixOffProcDataAdd(matrix));
      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParCSRMatrixInitialize( hypre_AuxParCSRMatrix *matrix )
{
   int local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);
   int *row_space = hypre_AuxParCSRMatrixRowSpace(matrix);
   int max_off_proc_elmts_set = hypre_AuxParCSRMatrixMaxOffProcElmtsSet(matrix);
   int max_off_proc_elmts_add = hypre_AuxParCSRMatrixMaxOffProcElmtsAdd(matrix);
   int **aux_j;
   double **aux_data;
   int i;

   if (local_num_rows < 0) 
      return -1;
   if (local_num_rows == 0) 
      return 0;
   /* allocate stash for setting off processor values */
   if (max_off_proc_elmts_set > 0)
   {
      hypre_AuxParCSRMatrixOffProcISet(matrix) = hypre_CTAlloc(int,
		2*max_off_proc_elmts_set);
      hypre_AuxParCSRMatrixOffProcJSet(matrix) = hypre_CTAlloc(int,
		max_off_proc_elmts_set);
      hypre_AuxParCSRMatrixOffProcDataSet(matrix) = hypre_CTAlloc(double,
		max_off_proc_elmts_set);
   }
   /* allocate stash for adding to off processor values */
   if (max_off_proc_elmts_add > 0)
   {
      hypre_AuxParCSRMatrixOffProcIAdd(matrix) = hypre_CTAlloc(int,
		2*max_off_proc_elmts_add);
      hypre_AuxParCSRMatrixOffProcJAdd(matrix) = hypre_CTAlloc(int,
		max_off_proc_elmts_add);
      hypre_AuxParCSRMatrixOffProcDataAdd(matrix) = hypre_CTAlloc(double,
		max_off_proc_elmts_add);
   }
   if (hypre_AuxParCSRMatrixNeedAux(matrix))
   {
      aux_j = hypre_CTAlloc(int *, local_num_rows);
      aux_data = hypre_CTAlloc(double *, local_num_rows);
      if (!hypre_AuxParCSRMatrixRowLength(matrix))
         hypre_AuxParCSRMatrixRowLength(matrix) = 
  		hypre_CTAlloc(int, local_num_rows);
      if (row_space)
      {
         for (i=0; i < local_num_rows; i++)
         {
            aux_j[i] = hypre_CTAlloc(int, row_space[i]);
            aux_data[i] = hypre_CTAlloc(double, row_space[i]);
         }
      }
      else
      {
         row_space = hypre_CTAlloc(int, local_num_rows);
         for (i=0; i < local_num_rows; i++)
         {
            row_space[i] = 30;
            aux_j[i] = hypre_CTAlloc(int, 30);
            aux_data[i] = hypre_CTAlloc(double, 30);
         }
         hypre_AuxParCSRMatrixRowSpace(matrix) = row_space;
      }
      hypre_AuxParCSRMatrixAuxJ(matrix) = aux_j;
      hypre_AuxParCSRMatrixAuxData(matrix) = aux_data;
   }
   else
   {
      hypre_AuxParCSRMatrixIndxDiag(matrix) = hypre_CTAlloc(int,local_num_rows);
      hypre_AuxParCSRMatrixIndxOffd(matrix) = hypre_CTAlloc(int,local_num_rows);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixSetMaxOffProcElmtsSet
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParCSRMatrixSetMaxOffPRocElmtsSet( hypre_AuxParCSRMatrix *matrix,
					    int max_off_proc_elmts_set )
{
   int ierr = 0;
   hypre_AuxParCSRMatrixMaxOffProcElmtsSet(matrix) = max_off_proc_elmts_set;
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixSetMaxOffProcElmtsAdd
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParCSRMatrixSetMaxOffPRocElmtsAdd( hypre_AuxParCSRMatrix *matrix,
					    int max_off_proc_elmts_add )
{
   int ierr = 0;
   hypre_AuxParCSRMatrixMaxOffProcElmtsAdd(matrix) = max_off_proc_elmts_add;
   return ierr;
}
