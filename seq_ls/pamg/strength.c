
/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

#include "headers.h"

/*==========================================================================*/
/*==========================================================================*/
/**
  Creates strength matrix S

  {\bf Input files:}
  headers.h

  @return Error code.
  
  @param A [IN]
  coefficient matrix
  @param strength_threshold [IN]
  threshold parameter used to define strength
  @param S_ptr [OUT]
  strength matrix
  
  @see */
/*--------------------------------------------------------------------------*/

/**************************************************************
 *
 *      Creates strength matrix
 *
 **************************************************************/
int
hypre_AMGCreateS( hypre_CSRMatrix    *A,
                  double              strength_threshold,
                  int		      mode,
                  int		     *dof_func,
                  hypre_CSRMatrix   **S_ptr              )
{
   int             *A_i           = hypre_CSRMatrixI(A);
   int             *A_j           = hypre_CSRMatrixJ(A);
   double          *A_data        = hypre_CSRMatrixData(A);
   int              num_variables = hypre_CSRMatrixNumRows(A);
                  
   hypre_CSRMatrix *S;
   int             *S_i;
   int             *S_j;
   double          *S_data;
                 
   double           diag, row_scale;
   int              i, j, k, jA, jS, kS, ig;

   int              ierr = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   S = hypre_CSRMatrixCreate(num_variables, num_variables,
                             A_i[num_variables]);
   hypre_CSRMatrixInitialize(S);

   S_i           = hypre_CSRMatrixI(S);
   S_j           = hypre_CSRMatrixJ(S);
   S_data        = hypre_CSRMatrixData(S);

   /* give S same nonzero structure as A */
   for (i = 0; i < num_variables; i++)
   {
      S_i[i] = A_i[i];
      for (jA = A_i[i]; jA < A_i[i+1]; jA++)
      {
         S_j[jA] = A_j[jA];
      }
   }
   S_i[num_variables] = A_i[num_variables];

   if (mode)
   {
   for (i = 0; i < num_variables; i++)
   {
      diag = A_data[A_i[i]];

      /* compute scaling factor */
      row_scale = 0.0;
      for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
      {
	 if (dof_func[i] == dof_func[A_j[jA]])
            row_scale = hypre_max(row_scale, fabs(A_data[jA]));
      }

      /* compute row entries of S */
      S_data[A_i[i]] = 0;
      for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
      {
         S_data[jA] = 0;
         if (fabs(A_data[jA]) > strength_threshold * row_scale
		&& dof_func[i] == dof_func[A_j[jA]])            
            S_data[jA] = -1;
      }
   }
   }
   else
   {
   for (i = 0; i < num_variables; i++)
   {
      diag = A_data[A_i[i]];

      /* compute scaling factor */
      row_scale = 0.0;
      if (diag < 0)
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
	   if (dof_func[i] == dof_func[A_j[jA]])
            row_scale = hypre_max(row_scale, A_data[jA]);
         }
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
	   if (dof_func[i] == dof_func[A_j[jA]])
            row_scale = hypre_min(row_scale, A_data[jA]);
         }
      }

      /* compute row entries of S */
      S_data[A_i[i]] = 0;
      if (diag < 0) 
      { 
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            S_data[jA] = 0;
            if (A_data[jA] > strength_threshold * row_scale
		&& dof_func[i] == dof_func[A_j[jA]])            {
               S_data[jA] = -1;
            }
         }
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            S_data[jA] = 0;
            if (A_data[jA] < strength_threshold * row_scale
		&&dof_func[i] == dof_func[A_j[jA]])
            {
               S_data[jA] = -1;
            }
         }
      }
   }
   }


   /*--------------------------------------------------------------
    * "Compress" the strength matrix.
    *
    * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
    *
    * NOTE: This "compression" section of code may be removed, and
    * coarsening will still be done correctly.  However, the routine
    * that builds interpolation would have to be modified first.
    *----------------------------------------------------------------*/

   jS = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_i[i] = jS;
      for (jA = A_i[i]; jA < A_i[i+1]; jA++)
      {
         if (S_data[jA])
         {
            S_j[jS]    = S_j[jA];
            S_data[jS] = S_data[jA];
            jS++;
         }
      }
   }
   S_i[num_variables] = jS;
   hypre_CSRMatrixNumNonzeros(S) = jS;

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   *S_ptr           = S;

   return (ierr);
}

int
hypre_AMGCompressS( hypre_CSRMatrix    *S,
                    int		      num_path)
{
   int *S_i = hypre_CSRMatrixI(S);
   int *S_j = hypre_CSRMatrixJ(S);
   double *S_data = hypre_CSRMatrixData(S);
   
   double dnum_path = (double) num_path;
   double dat;
   int num_rows = hypre_CSRMatrixNumRows(S); 
   int i, j;
   int col, cnt;

   cnt = 0;
   for (i=0; i < num_rows; i++)
   {
      for (j= S_i[i]; j < S_i[i+1]; j++)
      {
	 col = S_j[j];
	 dat = fabs(S_data[j]);
         if (dat >= dnum_path && col != i)
         {
	    S_data[cnt] = -dat;
	    S_j[cnt++] = S_j[j];
         }
      }
      S_i[i] = cnt;
   }

   for (i=num_rows; i > 0; i--)
      S_i[i] = S_i[i-1];

   S_i[0] = 0; 

   hypre_CSRMatrixNumNonzeros(S) = S_i[num_rows];
   return 0;
}
