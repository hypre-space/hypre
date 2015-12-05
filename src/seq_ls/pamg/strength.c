/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





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
HYPRE_Int
hypre_AMGCreateS( hypre_CSRMatrix    *A,
                  double              strength_threshold,
                  HYPRE_Int		      mode,
                  HYPRE_Int		     *dof_func,
                  hypre_CSRMatrix   **S_ptr              )
{
   HYPRE_Int             *A_i           = hypre_CSRMatrixI(A);
   HYPRE_Int             *A_j           = hypre_CSRMatrixJ(A);
   double          *A_data        = hypre_CSRMatrixData(A);
   HYPRE_Int              num_variables = hypre_CSRMatrixNumRows(A);
                  
   hypre_CSRMatrix *S;
   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;
   double          *S_data;
                 
   double           diag, row_scale;
   HYPRE_Int              i, jA, jS;

   HYPRE_Int              ierr = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
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

HYPRE_Int
hypre_AMGCompressS( hypre_CSRMatrix    *S,
                    HYPRE_Int		      num_path)
{
   HYPRE_Int *S_i = hypre_CSRMatrixI(S);
   HYPRE_Int *S_j = hypre_CSRMatrixJ(S);
   double *S_data = hypre_CSRMatrixData(S);
   
   double dnum_path = (double) num_path;
   double dat;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(S); 
   HYPRE_Int i, j;
   HYPRE_Int col, cnt;

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


HYPRE_Int
hypre_AMGCreate2ndS( hypre_CSRMatrix *A, HYPRE_Int n_coarse,
              HYPRE_Int *CF_marker, HYPRE_Int num_paths, hypre_CSRMatrix **S_ptr)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   hypre_CSRMatrix *S;
   double     *S_data;
   HYPRE_Int	      *S_i;
   HYPRE_Int        *S_j;

   HYPRE_Int         ia, ib, ic, ja, jb, num_nonzeros=0;
   HYPRE_Int	       row_start, cnt, S_cnt;
   HYPRE_Int	       i, j, jcol, col;
   double      a_entry, b_entry, d_num_paths;
   HYPRE_Int         *B_marker;
   HYPRE_Int         *fine_to_coarse;


   B_marker = hypre_CTAlloc(HYPRE_Int, nrows_A);
   S_i = hypre_CTAlloc(HYPRE_Int, n_coarse+1);
   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, nrows_A);
   d_num_paths = (double) num_paths;

   for (ib = 0; ib < nrows_A; ib++)
   {
	B_marker[ib] = -1;
	fine_to_coarse[ib] = -1;
   }

   cnt = 0;
   S_i[0] = 0;

   for (ic = 0; ic < nrows_A; ic++)
   {
      if (CF_marker[ic] > 0)
      {
	fine_to_coarse[ic] = cnt;
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
		jcol = A_j[ia];
                if (CF_marker[jcol] > 0)
	        {
		   B_marker[jcol] = ic;
		   num_nonzeros++;
	        }
	}
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
		ja = A_j[ia];
		for (ib = A_i[ja]; ib < A_i[ja+1]; ib++)
		{
			jb = A_j[ib];
			if (CF_marker[jb] > 0 && B_marker[jb] != ic)
			{
				B_marker[jb] = ic;
				num_nonzeros++;
			}
		}
   	}
	S_i[++cnt] = num_nonzeros;
      }
   }

   S = hypre_CSRMatrixCreate(n_coarse, n_coarse, num_nonzeros);
   hypre_CSRMatrixI(S) = S_i;
   hypre_CSRMatrixInitialize(S);
   S_j = hypre_CSRMatrixJ(S);
   S_data = hypre_CSRMatrixData(S);

   for (ib = 0; ib < nrows_A; ib++)
	B_marker[ib] = -1;

   cnt = 0;
   S_cnt = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      if (CF_marker[ic] > 0)
      {
	row_start = S_i[S_cnt++];
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
		jcol = A_j[ia];
	 	if (CF_marker[jcol] > 0)
		{
		   S_j[cnt] = fine_to_coarse[jcol];
		   S_data[cnt] = 2*A_data[ia];
		   B_marker[jcol] = cnt;
		   cnt++;
		}
	}
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
		ja = A_j[ia];
		a_entry = A_data[ia];
		for (ib = A_i[ja]; ib < A_i[ja+1]; ib++)
		{
			jb = A_j[ib];
			b_entry = A_data[ib];
			if (CF_marker[jb] > 0)
			{
			   if (B_marker[jb] < row_start)
			   {
				B_marker[jb] = cnt;
				S_j[B_marker[jb]] = fine_to_coarse[jb];
				S_data[B_marker[jb]] = -a_entry*b_entry;
				cnt++;
			   }
			   else
				S_data[B_marker[jb]] -= a_entry*b_entry;
			}
				 
		}
	}
      }
   }
   hypre_TFree(B_marker);
   hypre_TFree(fine_to_coarse);
   cnt = 0;
   for (i=0; i < n_coarse; i++)
   {
      for (j= S_i[i]; j < S_i[i+1]; j++)
      {
	 col = S_j[j];
	 a_entry = fabs(S_data[j]);
         if (a_entry >= d_num_paths && col != i)
         {
	    S_data[cnt] = -a_entry;
	    S_j[cnt++] = S_j[j];
         }
      }
      S_i[i] = cnt;
   }

   for (i=n_coarse; i > 0; i--)
      S_i[i] = S_i[i-1];

   S_i[0] = 0; 

   hypre_CSRMatrixNumNonzeros(S) = S_i[n_coarse];

   *S_ptr = S;

   return 0;
}

HYPRE_Int
hypre_AMGCorrectCFMarker(HYPRE_Int *CF_marker, HYPRE_Int num_var, HYPRE_Int *new_CF_marker)
{
   HYPRE_Int i, cnt;

   cnt = 0;
   for (i=0; i < num_var; i++)
   {
      if (CF_marker[i] > 0)
	 CF_marker[i] = new_CF_marker[cnt++]; 
   }

   return 0;
}
   	
