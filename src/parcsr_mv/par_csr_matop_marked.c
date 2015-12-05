/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/





#include "headers.h"
#include "assert.h"

void hypre_ParCSRMatrixCopy_C( hypre_ParCSRMatrix * P,
                               hypre_ParCSRMatrix * C, HYPRE_Int * CF_marker );
void hypre_ParCSRMatrixZero_F( hypre_ParCSRMatrix * P, HYPRE_Int * CF_marker );

void hypre_ParMatmul_RowSizes_Marked
( HYPRE_Int ** C_diag_i, HYPRE_Int ** C_offd_i, HYPRE_Int ** B_marker,
  HYPRE_Int * A_diag_i, HYPRE_Int * A_diag_j, HYPRE_Int * A_offd_i, HYPRE_Int * A_offd_j,
  HYPRE_Int * B_diag_i, HYPRE_Int * B_diag_j, HYPRE_Int * B_offd_i, HYPRE_Int * B_offd_j,
  HYPRE_Int * B_ext_diag_i, HYPRE_Int * B_ext_diag_j, 
  HYPRE_Int * B_ext_offd_i, HYPRE_Int * B_ext_offd_j, HYPRE_Int * map_B_to_C,
  HYPRE_Int *C_diag_size, HYPRE_Int *C_offd_size,
  HYPRE_Int num_rows_diag_A, HYPRE_Int num_cols_offd_A, HYPRE_Int allsquare,
  HYPRE_Int num_cols_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_Int num_cols_offd_C,
  HYPRE_Int * CF_marker, HYPRE_Int * dof_func, HYPRE_Int * dof_func_offd
   )
 /* Compute row sizes of result of a matrix multiplication A*B.
   But we only consider rows designated by CF_marker(i)<0 ("Fine" rows).
   This function is the same as hypre_ParMatmul_RowSizes,but with a little code
   added to use the marker array.
   Input arguments like num_rows_diag_A should refer to the full size matrix A,
   not just the "Fine" part.  The principle here is that A and B have coarse+fine
   data, C only has fine data.  But C is the full size of the product A*B.
 */
{
   HYPRE_Int i1, i2, i3, jj2, jj3;
   HYPRE_Int jj_count_diag, jj_count_offd, jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int start_indexing = 0; /* start indexing for C_data at 0 */
   /* First pass begins here.  Computes sizes of marked C rows.
      Arrays computed: C_diag_i, C_offd_i, B_marker
      Arrays needed: (11, all HYPRE_Int*)
        A_diag_i, A_diag_j, A_offd_i, A_offd_j,
        B_diag_i, B_diag_j, B_offd_i, B_offd_j,
        B_ext_i, B_ext_j, col_map_offd_B,
        col_map_offd_B, B_offd_i, B_offd_j, B_ext_i, B_ext_j,
      Scalars computed: C_diag_size, C_offd_size
      Scalars needed:
      num_rows_diag_A, num_rows_diag_A, num_cols_offd_A, allsquare,
      first_col_diag_B, n_cols_B, num_cols_offd_B, num_cols_diag_B
   */

   *C_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A+1);
   *C_offd_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A+1);
   /* ... CTAlloc initializes to 0, so entries ignored due to CF_marker will be
      returned as 0 */

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < num_cols_diag_B+num_cols_offd_C; i1++)
   {      
      (*B_marker)[i1] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over rows of A
    *-----------------------------------------------------------------------*/
   
   for (i1 = 0; i1 < num_rows_diag_A; i1++)
      if ( CF_marker[i1] >= 0 ) /* Coarse row */
      {
         /* To make an empty C row i1, its begin point should be the same as for i1: */
         /* (*C_diag_i)[i1] = jj_count_diag;
            (*C_offd_i)[i1] = jj_count_offd;*/
         /* To make the C row i1 the same size as the B row i1: */
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         jj_count_diag += B_diag_i[i1+1] - B_diag_i[i1];
         jj_count_offd += B_offd_i[i1+1] - B_offd_i[i1];
         (*C_diag_i)[i1] = jj_row_begin_diag;
         (*C_offd_i)[i1] = jj_row_begin_offd;
      }
      else
   {
      /* This block, most of of this function, is unchanged from hypre_ParMatmul_Row_Sizes
         (except for the dof_func checks, which are effectively gone if you set
         dof_func=NULL); maybe it can be spun off into a separate shared function.*/      
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, C_{i1,i1} (for square matrices). 
       *--------------------------------------------------------------------*/
 
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare ) {
         (*B_marker)[i1] = jj_count_diag;
         jj_count_diag++;
      }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
	 if (num_cols_offd_A)
	 {
            for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
            {
               i2 = A_offd_j[jj2];
 
               if ( dof_func==NULL || dof_func[i1] == dof_func_offd[i2] )
               {/* interpolate only like "functions" */
                  /*-----------------------------------------------------------
                   *  Loop over entries in row i2 of B_ext.
                   *-----------------------------------------------------------*/
 
                  for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2+1]; jj3++)
                  {
                     i3 = num_cols_diag_B+B_ext_offd_j[jj3];
                  
                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, mark it and increment
                      *  counter.
                      *--------------------------------------------------------*/

                     if ((*B_marker)[i3] < jj_row_begin_offd)
                     {
                     	(*B_marker)[i3] = jj_count_offd;
                     	jj_count_offd++;
                     } 
                  }
                  for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2+1]; jj3++)
                  {
                     i3 = B_ext_diag_j[jj3];
                  
                     if ((*B_marker)[i3] < jj_row_begin_diag)
                     {
                  	(*B_marker)[i3] = jj_count_diag;
                     	jj_count_diag++;
                     } 
                  }
               }
            }
         }
         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
 
            if( dof_func==NULL || dof_func[i1] == dof_func[i2] )
            { /* interpolate only like "functions" */
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_diag.
                *-----------------------------------------------------------*/
 
               for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
               {
                  i3 = B_diag_j[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
 
                  if ((*B_marker)[i3] < jj_row_begin_diag)
                  {
                     (*B_marker)[i3] = jj_count_diag;
                     jj_count_diag++;
                  }
               }
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_offd.
                *-----------------------------------------------------------*/

	       if (num_cols_offd_B)
	       { 
                  for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
                  {
                     i3 = num_cols_diag_B+map_B_to_C[B_offd_j[jj3]];
                  
                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, mark it and increment
                      *  counter.
                      *--------------------------------------------------------*/
 
                     if ((*B_marker)[i3] < jj_row_begin_offd)
                     {
                        (*B_marker)[i3] = jj_count_offd;
                        jj_count_offd++;
                     }
                  }
               }
            }
         }
            
      /*--------------------------------------------------------------------
       * Set C_diag_i and C_offd_i for this row.
       *--------------------------------------------------------------------*/
 
      (*C_diag_i)[i1] = jj_row_begin_diag;
      (*C_offd_i)[i1] = jj_row_begin_offd;
      
   }
  
   (*C_diag_i)[num_rows_diag_A] = jj_count_diag;
   (*C_offd_i)[num_rows_diag_A] = jj_count_offd;
 
   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   *C_diag_size = jj_count_diag;
   *C_offd_size = jj_count_offd;

   /* End of First Pass */
}





hypre_ParCSRMatrix * hypre_ParMatmul_FC(
   hypre_ParCSRMatrix * A, hypre_ParCSRMatrix * P, HYPRE_Int * CF_marker,
   HYPRE_Int * dof_func, HYPRE_Int * dof_func_offd )
/* hypre_parMatmul_FC creates and returns the "Fine"-designated rows of the
   matrix product A*P.  A's size is (nC+nF)*(nC+nF), P's size is (nC+nF)*nC
   where nC is the number of coarse rows/columns, nF the number of fine
   rows/columns.  The size of C=A*P is (nC+nF)*nC, even though not all rows
   of C are actually computed.  If we were to construct a matrix consisting
   only of the computed rows of C, its size would be nF*nC.
   "Fine" is defined solely by the marker array, and for example could be
   a proper subset of the fine points of a multigrid hierarchy.
*/
{
   /* To compute a submatrix of C containing only the computed data, i.e.
      only "Fine" rows, we would have to do a lot of computational work,
      with a lot of communication.  The communication is because such a
      matrix would need global information that depends on which rows are
      "Fine".
   */

   MPI_Comm 	   comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   
   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   
   double          *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int	num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int	num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int	num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   
   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   
   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int             *P_diag_j = hypre_CSRMatrixJ(P_diag);

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int		   *col_map_offd_P = hypre_ParCSRMatrixColMapOffd(P);
   
   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int             *P_offd_j = hypre_CSRMatrixJ(P_offd);

   HYPRE_Int	first_col_diag_P = hypre_ParCSRMatrixFirstColDiag(P);
   HYPRE_Int	last_col_diag_P;
   HYPRE_Int *col_starts_P = hypre_ParCSRMatrixColStarts(P);
   HYPRE_Int	num_rows_diag_P = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int	num_cols_diag_P = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int	num_cols_offd_P = hypre_CSRMatrixNumCols(P_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_Int		      *col_map_offd_C;
   HYPRE_Int		      *map_P_to_C;

   hypre_CSRMatrix *C_diag;

   double          *C_diag_data;
   HYPRE_Int             *C_diag_i;
   HYPRE_Int             *C_diag_j;

   hypre_CSRMatrix *C_offd;

   double          *C_offd_data=NULL;
   HYPRE_Int             *C_offd_i=NULL;
   HYPRE_Int             *C_offd_j=NULL;

   HYPRE_Int              C_diag_size;
   HYPRE_Int              C_offd_size;
   HYPRE_Int		    num_cols_offd_C = 0;
   
   hypre_CSRMatrix *Ps_ext;
   
   double          *Ps_ext_data;
   HYPRE_Int             *Ps_ext_i;
   HYPRE_Int             *Ps_ext_j;

   double          *P_ext_diag_data;
   HYPRE_Int             *P_ext_diag_i;
   HYPRE_Int             *P_ext_diag_j;
   HYPRE_Int              P_ext_diag_size;

   double          *P_ext_offd_data;
   HYPRE_Int             *P_ext_offd_i;
   HYPRE_Int             *P_ext_offd_j;
   HYPRE_Int              P_ext_offd_size;

   HYPRE_Int		   *P_marker;
   HYPRE_Int		   *temp;

   HYPRE_Int              i, j;
   HYPRE_Int              i1, i2, i3;
   HYPRE_Int              jj2, jj3;
   
   HYPRE_Int              jj_count_diag, jj_count_offd;
   HYPRE_Int              jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int              start_indexing = 0; /* start indexing for C_data at 0 */
   HYPRE_Int		    n_rows_A_global, n_cols_A_global;
   HYPRE_Int		    n_rows_P_global, n_cols_P_global;
   HYPRE_Int              allsquare = 0;
   HYPRE_Int              cnt, cnt_offd, cnt_diag;
   HYPRE_Int 		    num_procs;
   HYPRE_Int 		    value;

   double           a_entry;
   double           a_b_product;
   
   n_rows_A_global = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A_global = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_P_global = hypre_ParCSRMatrixGlobalNumRows(P);
   n_cols_P_global = hypre_ParCSRMatrixGlobalNumCols(P);

   if (n_cols_A_global != n_rows_P_global || num_cols_diag_A != num_rows_diag_P)
   {
	hypre_printf(" Error! Incompatible matrix dimensions!\n");
	return NULL;
   }
   /* if (num_rows_A==num_cols_P) allsquare = 1; */

   /*-----------------------------------------------------------------------
    *  Extract P_ext, i.e. portion of P that is stored on neighbor procs
    *  and needed locally for matrix matrix product 
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);

   if (num_procs > 1)
   {
       /*---------------------------------------------------------------------
    	* If there exists no CommPkg for A, a CommPkg is generated using
    	* equally load balanced partitionings within 
	* hypre_ParCSRMatrixExtractBExt
    	*--------------------------------------------------------------------*/
   	Ps_ext = hypre_ParCSRMatrixExtractBExt(P,A,1);
   	Ps_ext_data = hypre_CSRMatrixData(Ps_ext);
   	Ps_ext_i    = hypre_CSRMatrixI(Ps_ext);
   	Ps_ext_j    = hypre_CSRMatrixJ(Ps_ext);
   }
   P_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A+1);
   P_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A+1);
   P_ext_diag_size = 0;
   P_ext_offd_size = 0;
   last_col_diag_P = first_col_diag_P + num_cols_diag_P -1;

   for (i=0; i < num_cols_offd_A; i++)
   {
      for (j=Ps_ext_i[i]; j < Ps_ext_i[i+1]; j++)
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
            P_ext_offd_size++;
         else
            P_ext_diag_size++;
      P_ext_diag_i[i+1] = P_ext_diag_size;
      P_ext_offd_i[i+1] = P_ext_offd_size;
   }

   if (P_ext_diag_size)
   {
      P_ext_diag_j = hypre_CTAlloc(HYPRE_Int, P_ext_diag_size);
      P_ext_diag_data = hypre_CTAlloc(double, P_ext_diag_size);
   }
   if (P_ext_offd_size)
   {
      P_ext_offd_j = hypre_CTAlloc(HYPRE_Int, P_ext_offd_size);
      P_ext_offd_data = hypre_CTAlloc(double, P_ext_offd_size);
   }

   cnt_offd = 0;
   cnt_diag = 0;
   for (i=0; i < num_cols_offd_A; i++)
   {
      for (j=Ps_ext_i[i]; j < Ps_ext_i[i+1]; j++)
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
         {
            P_ext_offd_j[cnt_offd] = Ps_ext_j[j];
            P_ext_offd_data[cnt_offd++] = Ps_ext_data[j];
         }
         else
         {
            P_ext_diag_j[cnt_diag] = Ps_ext_j[j] - first_col_diag_P;
            P_ext_diag_data[cnt_diag++] = Ps_ext_data[j];
         }
   }

   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(Ps_ext);
      Ps_ext = NULL;
   }

   cnt = 0;
   if (P_ext_offd_size || num_cols_offd_P)
   {
      temp = hypre_CTAlloc(HYPRE_Int, P_ext_offd_size+num_cols_offd_P);
      for (i=0; i < P_ext_offd_size; i++)
         temp[i] = P_ext_offd_j[i];
      cnt = P_ext_offd_size;
      for (i=0; i < num_cols_offd_P; i++)
         temp[cnt++] = col_map_offd_P[i];
   }
   if (cnt)
   {
      qsort0(temp, 0, cnt-1);

      num_cols_offd_C = 1;
      value = temp[0];
      for (i=1; i < cnt; i++)
      {
         if (temp[i] > value)
         {
            value = temp[i];
            temp[num_cols_offd_C++] = value;
         }
      }
   }

   if (num_cols_offd_C)
        col_map_offd_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_C);

   for (i=0; i < num_cols_offd_C; i++)
      col_map_offd_C[i] = temp[i];

   if (P_ext_offd_size || num_cols_offd_P)
      hypre_TFree(temp);

   for (i=0 ; i < P_ext_offd_size; i++)
      P_ext_offd_j[i] = hypre_BinarySearch(col_map_offd_C,
                                           P_ext_offd_j[i],
                                           num_cols_offd_C);
   if (num_cols_offd_P)
   {
      map_P_to_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_P);

      cnt = 0;
      for (i=0; i < num_cols_offd_C; i++)
         if (col_map_offd_C[i] == col_map_offd_P[cnt])
         {
            map_P_to_C[cnt++] = i;
            if (cnt == num_cols_offd_P) break;
         }
   }

   /*-----------------------------------------------------------------------
   *  Allocate marker array.
    *-----------------------------------------------------------------------*/

   P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_diag_P+num_cols_offd_C);

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   for (i1 = 0; i1 < num_cols_diag_P+num_cols_offd_C; i1++)
   {      
      P_marker[i1] = -1;
   }


/* no changes for the marked version above this point */
   /* This function call is the first pass: */
   hypre_ParMatmul_RowSizes_Marked(
      &C_diag_i, &C_offd_i, &P_marker,
      A_diag_i, A_diag_j, A_offd_i, A_offd_j,
      P_diag_i, P_diag_j, P_offd_i, P_offd_j,
      P_ext_diag_i, P_ext_diag_j, P_ext_offd_i, P_ext_offd_j,
      map_P_to_C,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_cols_offd_A, allsquare,
      num_cols_diag_P, num_cols_offd_P,
      num_cols_offd_C, CF_marker, dof_func, dof_func_offd
      );

   /* The above call of hypre_ParMatmul_RowSizes_Marked computed
      two scalars: C_diag_size, C_offd_size,
      and two arrays: C_diag_i, C_offd_i
      ( P_marker is also computed, but only used internally )
   */

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   last_col_diag_P = first_col_diag_P + num_cols_diag_P - 1;
   C_diag_data = hypre_CTAlloc(double, C_diag_size);
   C_diag_j    = hypre_CTAlloc(HYPRE_Int, C_diag_size);
   if (C_offd_size)
   { 
   	C_offd_data = hypre_CTAlloc(double, C_offd_size);
   	C_offd_j    = hypre_CTAlloc(HYPRE_Int, C_offd_size);
   } 


   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_data and C_diag_j.
    *  Second Pass: Fill in C_offd_data and C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < num_cols_diag_P+num_cols_offd_C; i1++)
   {      
      P_marker[i1] = -1;
   }
   
   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/
    
   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {

      if ( CF_marker[i1] < 0 )  /* i1 is a fine row */
         /* ... This and the coarse row code are the only parts between first pass
            and near the end where
            hypre_ParMatmul_FC is different from the regular hypre_ParMatmul */
      {

         /*--------------------------------------------------------------------
          *  Create diagonal entry, C_{i1,i1} 
          *--------------------------------------------------------------------*/

         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
	 if (num_cols_offd_A)
	 {
            for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
            {
               i2 = A_offd_j[jj2];
               if( dof_func==NULL || dof_func[i1] == dof_func_offd[i2] )
               {  /* interpolate only like "functions" */
                  a_entry = A_offd_data[jj2];
            
                  /*-----------------------------------------------------------
                   *  Loop over entries in row i2 of P_ext.
                   *-----------------------------------------------------------*/

                  for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3++)
                  {
                     i3 = num_cols_diag_P+P_ext_offd_j[jj3];
                     a_b_product = a_entry * P_ext_offd_data[jj3];
                  
                     /*--------------------------------------------------------
                      *  Check P_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution.
                      *--------------------------------------------------------*/
                     if (P_marker[i3] < jj_row_begin_offd)
                     {
                        P_marker[i3] = jj_count_offd;
                        C_offd_data[jj_count_offd] = a_b_product;
                        C_offd_j[jj_count_offd] = i3-num_cols_diag_P;
                        jj_count_offd++;
                     }
                     else
                        C_offd_data[P_marker[i3]] += a_b_product;
                  }
                  for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3++)
                  {
                     i3 = P_ext_diag_j[jj3];
                     a_b_product = a_entry * P_ext_diag_data[jj3];

                     if (P_marker[i3] < jj_row_begin_diag)
                     {
                        P_marker[i3] = jj_count_diag;
                        C_diag_data[jj_count_diag] = a_b_product;
                        C_diag_j[jj_count_diag] = i3;
                        jj_count_diag++;
                     }
                     else
                        C_diag_data[P_marker[i3]] += a_b_product;
                  }
               }
               else
               {  /* Interpolation mat should be 0 where i1 and i2 correspond to
                     different "functions".  As we haven't created an entry for
                     C(i1,i2), nothing needs to be done. */
               }

            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            if( dof_func==NULL || dof_func[i1] == dof_func[i2] )
            {  /* interpolate only like "functions" */
               a_entry = A_diag_data[jj2];
            
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/

               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3];
                  a_b_product = a_entry * P_diag_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_diag)
                  {
                     P_marker[i3] = jj_count_diag;
                     C_diag_data[jj_count_diag] = a_b_product;
                     C_diag_j[jj_count_diag] = i3;
                     jj_count_diag++;
                  }
                  else
                  {
                     C_diag_data[P_marker[i3]] += a_b_product;
                  }
               }
               if (num_cols_offd_P)
	       {
                  for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
                  {
                     i3 = num_cols_diag_P+map_P_to_C[P_offd_j[jj3]];
                     a_b_product = a_entry * P_offd_data[jj3];
                  
                     /*--------------------------------------------------------
                      *  Check P_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution.
                      *--------------------------------------------------------*/

                     if (P_marker[i3] < jj_row_begin_offd)
                     {
                        P_marker[i3] = jj_count_offd;
                        C_offd_data[jj_count_offd] = a_b_product;
                        C_offd_j[jj_count_offd] = i3-num_cols_diag_P;
                        jj_count_offd++;
                     }
                     else
                     {
                        C_offd_data[P_marker[i3]] += a_b_product;
                     }
                  }
               }
            }
            else
            {  /* Interpolation mat should be 0 where i1 and i2 correspond to
                  different "functions".  As we haven't created an entry for
                  C(i1,i2), nothing needs to be done. */
            }
         }
      }
      else  /* i1 is a coarse row.*/
         /* Copy P coarse-row values to C.  This is useful if C is meant to
            become a replacement for P */
      {
	 if (num_cols_offd_P)
	 {
            for (jj2 = P_offd_i[i1]; jj2 < P_offd_i[i1+1]; jj2++)
            {
               C_offd_j[jj_count_offd] = P_offd_j[jj_count_offd];
               C_offd_data[jj_count_offd] = P_offd_data[jj_count_offd];
               ++jj_count_offd;
            }
         }
         for (jj2 = P_diag_i[i1]; jj2 < P_diag_i[i1+1]; jj2++)
         {
            C_diag_j[jj_count_diag] = P_diag_j[jj2];
            C_diag_data[jj_count_diag] = P_diag_data[jj2];
            ++jj_count_diag;
         }
      }
   }

   C = hypre_ParCSRMatrixCreate(
      comm, n_rows_A_global, n_cols_P_global,
      row_starts_A, col_starts_P, num_cols_offd_C, C_diag_size, C_offd_size );

   /* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRMatrixSetColStartsOwner(C,0);

   C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrixData(C_diag) = C_diag_data; 
   hypre_CSRMatrixI(C_diag) = C_diag_i; 
   hypre_CSRMatrixJ(C_diag) = C_diag_j; 

   C_offd = hypre_ParCSRMatrixOffd(C);
   hypre_CSRMatrixI(C_offd) = C_offd_i; 
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_CSRMatrixData(C_offd) = C_offd_data; 
      hypre_CSRMatrixJ(C_offd) = C_offd_j; 
      hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   }

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   hypre_TFree(P_marker);   
   hypre_TFree(P_ext_diag_i);
   if (P_ext_diag_size)
   {
      hypre_TFree(P_ext_diag_j);
      hypre_TFree(P_ext_diag_data);
   }
   hypre_TFree(P_ext_offd_i);
   if (P_ext_offd_size)
   {
      hypre_TFree(P_ext_offd_j);
      hypre_TFree(P_ext_offd_data);
   }
   if (num_cols_offd_P) hypre_TFree(map_P_to_C);

   return C;
   
}

void hypre_ParMatScaleDiagInv_F(
   hypre_ParCSRMatrix * C, hypre_ParCSRMatrix * A, double weight, HYPRE_Int * CF_marker )
   /* hypre_ParMatScaleDiagInv scales certain rows of its first
    * argument by premultiplying with a submatrix of the inverse of
    * the diagonal of its second argument; and _also_ multiplying by the scalar
    * third argument.
    * The marker array determines rows are changed and which diagonal elements
    * are used.
    */
{
   /*
     If A=(Aij),C=(Cik), i&j in Fine+Coarse, k in Coarse, we want
        new Cik = (1/aii)*Cik, for Fine i only, all k.
     Unlike a matmul, this computation is purely local, only the diag
     blocks are involved.
   */

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrix *C_offd = hypre_ParCSRMatrixOffd(C);

   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);
   double          *C_diag_data = hypre_CSRMatrixData(C_diag);
   double          *C_offd_data = hypre_CSRMatrixData(C_offd);
   HYPRE_Int             *C_diag_i = hypre_CSRMatrixI(C_diag);
   HYPRE_Int             *C_offd_i = hypre_CSRMatrixI(C_offd);


   HYPRE_Int	num_rows_diag_C = hypre_CSRMatrixNumRows(C_diag);
   HYPRE_Int	num_cols_offd_C = hypre_CSRMatrixNumCols(C_offd);

   HYPRE_Int              i1, i2;
   HYPRE_Int              jj2, jj3;
   double           a_entry;

   /*-----------------------------------------------------------------------
    *  Loop over C_diag rows.
    *-----------------------------------------------------------------------*/
    
   for (i1 = 0; i1 < num_rows_diag_C; i1++)
   {
      
      if ( CF_marker[i1] < 0 )  /* Fine data only */
      {

         /*-----------------------------------------------------------------
          *  Loop over A_diag data
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            if ( i1==i2 )  /* diagonal of A only */
            {
               a_entry = A_diag_data[jj2] * weight;
            
               /*-----------------------------------------------------------
                *  Loop over entries in current row of C_diag.
                *-----------------------------------------------------------*/

               for (jj3 = C_diag_i[i2]; jj3 < C_diag_i[i2+1]; jj3++)
               {
                  C_diag_data[jj3] = C_diag_data[jj3] / a_entry;
                  
               }
               /*-----------------------------------------------------------
                *  Loop over entries in current row of C_offd.
                *-----------------------------------------------------------*/

               if ( num_cols_offd_C )
               {
                  for (jj3 = C_offd_i[i2]; jj3 < C_offd_i[i2+1]; jj3++)
                  {
                     C_offd_data[jj3] = C_offd_data[jj3] / a_entry;
                  }
               }
            }
         }
      }
   }

}

hypre_ParCSRMatrix * hypre_ParMatMinus_F(
   hypre_ParCSRMatrix * P, hypre_ParCSRMatrix * C, HYPRE_Int * CF_marker )
/* hypre_ParMatMinus_F subtracts selected rows of its second argument
   from selected rows of its first argument.  The marker array
   determines which rows are affected - those for which CF_marker<0.
   The result is returned as a new matrix.
*/
{
   /*
     If P=(Pik),C=(Cik), i in Fine+Coarse, k in Coarse, we want
        new Pik = Pik - Cik, for Fine i only, all k.
     This computation is purely local.
   */
   /* This is _not_ a general-purpose matrix subtraction function.
      This is written for an interpolation problem where it is known that C(i,k)
      exists whenever P(i,k) does (because C=A*P where A has nonzero diagonal elements).
   */

   hypre_ParCSRMatrix *Pnew;
   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrix *C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrix *C_offd = hypre_ParCSRMatrixOffd(C);
   hypre_CSRMatrix *Pnew_diag;
   hypre_CSRMatrix *Pnew_offd;

   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int             *P_diag_j = hypre_CSRMatrixJ(P_diag);
   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int             *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Int             *P_col_map_offd = hypre_ParCSRMatrixColMapOffd( P );
   double          *C_diag_data = hypre_CSRMatrixData(C_diag);
   HYPRE_Int             *C_diag_i = hypre_CSRMatrixI(C_diag);
   HYPRE_Int             *C_diag_j = hypre_CSRMatrixJ(C_diag);
   double          *C_offd_data = hypre_CSRMatrixData(C_offd);
   HYPRE_Int             *C_offd_i = hypre_CSRMatrixI(C_offd);
   HYPRE_Int             *C_offd_j = hypre_CSRMatrixJ(C_offd);
   HYPRE_Int             *C_col_map_offd = hypre_ParCSRMatrixColMapOffd( C );
   HYPRE_Int             *Pnew_diag_i;
   HYPRE_Int             *Pnew_diag_j;
   double          *Pnew_diag_data;
   HYPRE_Int             *Pnew_offd_i;
   HYPRE_Int             *Pnew_offd_j;
   double          *Pnew_offd_data;
   HYPRE_Int             *Pnew_j2m;
   HYPRE_Int             *Pnew_col_map_offd;

   HYPRE_Int	num_rows_diag_C = hypre_CSRMatrixNumRows(C_diag);
   /* HYPRE_Int	num_rows_offd_C = hypre_CSRMatrixNumRows(C_offd); */
   HYPRE_Int	num_cols_offd_C = hypre_CSRMatrixNumCols(C_offd);
   HYPRE_Int	num_cols_offd_P = hypre_CSRMatrixNumCols(P_offd);
   HYPRE_Int  num_cols_offd_Pnew, num_rows_offd_Pnew;
   
   HYPRE_Int              i1, jmin, jmax, jrange, jrangem1;
   HYPRE_Int              j, m, mc, mp, jc, jp, jP, jC, jg, jCg, jPg;
   double           dc, dp;

/*   Pnew = hypre_ParCSRMatrixCompleteClone( C );*/

   Pnew = hypre_ParCSRMatrixUnion( C, P );
;
   hypre_ParCSRMatrixZero_F( Pnew, CF_marker );  /* fine rows of Pnew set to 0 */
   hypre_ParCSRMatrixCopy_C( Pnew, C, CF_marker ); /* coarse rows of Pnew copied from C (or P) */
   /* ...Zero_F may not be needed depending on how Pnew is made */
   Pnew_diag = hypre_ParCSRMatrixDiag(Pnew);
   Pnew_offd = hypre_ParCSRMatrixOffd(Pnew);
   Pnew_diag_i = hypre_CSRMatrixI(Pnew_diag);
   Pnew_diag_j = hypre_CSRMatrixJ(Pnew_diag);
   Pnew_offd_i = hypre_CSRMatrixI(Pnew_offd);
   Pnew_offd_j = hypre_CSRMatrixJ(Pnew_offd);
   Pnew_diag_data = hypre_CSRMatrixData(Pnew_diag);
   Pnew_offd_data = hypre_CSRMatrixData(Pnew_offd);
   Pnew_col_map_offd = hypre_ParCSRMatrixColMapOffd( Pnew );
   num_rows_offd_Pnew = hypre_CSRMatrixNumRows(Pnew_offd);
   num_cols_offd_Pnew = hypre_CSRMatrixNumCols(Pnew_offd);


   /* Find the j-ranges, needed to allocate a "reverse lookup" array. */
   /* This is the max j - min j over P and Pnew (which here is a copy of C).
      Only the diag block is considered. */
   /* For scalability reasons (jrange can get big) this won't work for the offd block.
      Also, indexing is more complicated in the offd block (c.f. col_map_offd).
      It's not clear, though whether the "quadratic" algorithm I'm using for the offd
      block is really any slower than the more complicated "linear" algorithm here. */
   jrange = 0;
   jrangem1=-1;
   for ( i1 = 0; i1 < num_rows_diag_C; i1++ )
   {
      if ( CF_marker[i1]<0 && hypre_CSRMatrixNumNonzeros(Pnew_diag)>0 )  /* only Fine rows matter */
      {
         jmin = Pnew_diag_j[ Pnew_diag_i[i1] ];
         jmax = Pnew_diag_j[ Pnew_diag_i[i1+1]-1 ];
         jrangem1 = jmax-jmin;
         jrange = hypre_max(jrange,jrangem1+1);
         /* If columns (of a given row) were in increasing order, the above would be sufficient.
            If not, the following would be necessary (and sufficient) */
         jmin = Pnew_diag_j[ Pnew_diag_i[i1] ];
         jmax = Pnew_diag_j[ Pnew_diag_i[i1] ];
         for ( m=Pnew_diag_i[i1]+1; m<Pnew_diag_i[i1+1]; ++m )
         {
            j = Pnew_diag_j[m];
            jmin = hypre_min( jmin, j );
            jmax = hypre_max( jmax, j );
         }
         for ( m=P_diag_i[i1]; m<P_diag_i[i1+1]; ++m )
         {
            j = P_diag_j[m];
            jmin = hypre_min( jmin, j );
            jmax = hypre_max( jmax, j );
         }
         jrangem1 = jmax-jmin;
         jrange = hypre_max(jrange,jrangem1+1);
      }
   }


   /*-----------------------------------------------------------------------
    *  Loop over Pnew_diag rows.  Construct a temporary reverse array:
    *  If j is a column number, Pnew_j2m[j] is the array index for j, i.e.
    *  Pnew_diag_j[ Pnew_j2m[j] ] = j
    *-----------------------------------------------------------------------*/

   Pnew_j2m = hypre_CTAlloc( HYPRE_Int, jrange );

   for ( i1 = 0; i1 < num_rows_diag_C; i1++ )
   {
      if ( CF_marker[i1]<0 && hypre_CSRMatrixNumNonzeros(Pnew_diag)>0 )  /* Fine data only */
      {
         /* just needed for an assertion below... */
         for ( j=0; j<jrange; ++j ) Pnew_j2m[j] = -1;
         jmin = Pnew_diag_j[ Pnew_diag_i[i1] ];
            /* If columns (of a given row) were in increasing order, the above line would be sufficient.
               If not, the following loop would have to be added (or store the jmin computed above )*/
         for ( m=Pnew_diag_i[i1]+1; m<Pnew_diag_i[i1+1]; ++m )
         {
            j = Pnew_diag_j[m];
            jmin = hypre_min( jmin, j );
         }
         for ( m=P_diag_i[i1]; m<P_diag_i[i1+1]; ++m )
         {
            j = P_diag_j[m];
            jmin = hypre_min( jmin, j );
         }
         for ( m = Pnew_diag_i[i1]; m<Pnew_diag_i[i1+1]; ++m )
         {
            j = Pnew_diag_j[m];
            hypre_assert( j-jmin>=0 );
            hypre_assert( j-jmin<jrange );
            Pnew_j2m[ j-jmin ] = m;
         }

         /*-----------------------------------------------------------------------
          *  Loop over C_diag data for the current row.
          *  Subtract each C data entry from the corresponding Pnew entry.
          *-----------------------------------------------------------------------*/

         for ( mc=C_diag_i[i1]; mc<C_diag_i[i1+1]; ++mc )
         {
            jc = C_diag_j[mc];
            dc = C_diag_data[mc];
            m = Pnew_j2m[jc-jmin];
            hypre_assert( m>=0 );
            Pnew_diag_data[m] -= dc;
         }

         /*-----------------------------------------------------------------------
          *  Loop over P_diag data for the current row.
          *  Add each P data entry from the corresponding Pnew entry.
          *-----------------------------------------------------------------------*/

         for ( mp=P_diag_i[i1]; mp<P_diag_i[i1+1]; ++mp )
         {
            jp = P_diag_j[mp];
            dp = P_diag_data[mp];
            m = Pnew_j2m[jp-jmin];
            hypre_assert( m>=0 );
            Pnew_diag_data[m] += dp;
         }
      }
   }

         /*-----------------------------------------------------------------------
          * Repeat for the offd block.
          *-----------------------------------------------------------------------*/

   for ( i1 = 0; i1 < num_rows_offd_Pnew; i1++ )
   {
      if ( CF_marker[i1]<0 && hypre_CSRMatrixNumNonzeros(Pnew_offd)>0 )  /* Fine data only */
      {
         if ( num_cols_offd_Pnew )
         {
            /*  This is a simple quadratic algorithm.  If necessary I may try
               to implement the ideas used on the diag block later. */
            for ( m = Pnew_offd_i[i1]; m<Pnew_offd_i[i1+1]; ++m )
            {
               j = Pnew_offd_j[m];
               jg = Pnew_col_map_offd[j];
               Pnew_offd_data[m] = 0;
               if ( num_cols_offd_C )
                  for ( mc=C_offd_i[i1]; mc<C_offd_i[i1+1]; ++mc )
                  {
                     jC = C_offd_j[mc];
                     jCg = C_col_map_offd[jC];
                     if ( jCg==jg ) Pnew_offd_data[m] -= C_offd_data[mc];
                  }
               if ( num_cols_offd_P )
                  for ( mp=P_offd_i[i1]; mp<P_offd_i[i1+1]; ++mp )
                  {
                     jP = P_offd_j[mp];
                     jPg = P_col_map_offd[jP];
                     if ( jPg==jg ) Pnew_offd_data[m] += P_offd_data[mp];
                  }
            }
         }
      }
   }


   hypre_TFree(Pnew_j2m);

   return Pnew;
}


  /* fine (marked <0 ) rows of Pnew set to 0 */
void  hypre_ParCSRMatrixZero_F( hypre_ParCSRMatrix * P, HYPRE_Int * CF_marker )
{
   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);

   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int	num_rows_diag_P = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int	num_rows_offd_P = hypre_CSRMatrixNumRows(P_offd);
   HYPRE_Int	num_cols_offd_P = hypre_CSRMatrixNumCols(P_offd);
   HYPRE_Int i1,  m;

   for ( i1= 0; i1 < num_rows_diag_P; i1++ )
   {
      if ( CF_marker[i1] < 0 )  /* Fine rows only */
      {
         for ( m=P_diag_i[i1]; m<P_diag_i[i1+1]; ++m )
         {
            P_diag_data[m] = 0;
         }
      }
   }
   if ( num_cols_offd_P )
      for ( i1= 0; i1 < num_rows_offd_P; i1++ )
      {
         if ( CF_marker[i1] < 0 )  /* Fine rows only */
         {
            for ( m=P_offd_i[i1]; m<P_offd_i[i1+1]; ++m )
            {
               P_offd_data[m] = 0;
            }
         }
      }

}


 /* coarse (marked >=0) rows of P copied from C Both matrices have the same sizes. */
void hypre_ParCSRMatrixCopy_C( hypre_ParCSRMatrix * P,
                               hypre_ParCSRMatrix * C, HYPRE_Int * CF_marker )
{
   hypre_CSRMatrix *C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrix *C_offd = hypre_ParCSRMatrixOffd(C);
   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);

   double          *C_diag_data = hypre_CSRMatrixData(C_diag);
   HYPRE_Int             *C_diag_i = hypre_CSRMatrixI(C_diag);
   double          *C_offd_data = hypre_CSRMatrixData(C_offd);
   HYPRE_Int             *C_offd_i = hypre_CSRMatrixI(C_offd);
   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int	num_rows_diag_C = hypre_CSRMatrixNumRows(C_diag);
   HYPRE_Int	num_rows_offd_C = hypre_CSRMatrixNumRows(C_offd);
   HYPRE_Int	num_cols_offd_C = hypre_CSRMatrixNumCols(C_offd);

   HYPRE_Int i1, m;

   for ( i1= 0; i1 < num_rows_diag_C; i1++ )
   {
      if ( CF_marker[i1] >= 0 )  /* Coarse rows only */
      {
         for ( m=C_diag_i[i1]; m<C_diag_i[i1+1]; ++m )
         {
            P_diag_data[m] = C_diag_data[m];
         }
      }
   }
   if ( num_cols_offd_C )
      for ( i1= 0; i1 < num_rows_offd_C; i1++ )
      {
         if ( CF_marker[i1] >= 0 )  /* Coarse rows only */
         {
            for ( m=C_offd_i[i1]; m<C_offd_i[i1+1]; ++m )
            {
               P_offd_data[m] = C_offd_data[m];
            }
         }
      }

}

/* Delete any matrix entry C(i,j) for which the corresponding entry P(i,j) doesn't exist -
   but only for "fine" rows C(i)<0
   This is done as a purely local computation - C and P must have the same data distribution
   (among processors).
*/
void hypre_ParCSRMatrixDropEntries( hypre_ParCSRMatrix * C,
                                    hypre_ParCSRMatrix * P, HYPRE_Int * CF_marker )
{
   hypre_CSRMatrix *C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrix *C_offd = hypre_ParCSRMatrixOffd(C);
   double          *C_diag_data = hypre_CSRMatrixData(C_diag);
   HYPRE_Int             *C_diag_i = hypre_CSRMatrixI(C_diag);
   HYPRE_Int             *C_diag_j = hypre_CSRMatrixJ(C_diag);
   double          *C_offd_data = hypre_CSRMatrixData(C_offd);
   HYPRE_Int             *C_offd_i = hypre_CSRMatrixI(C_offd);
   HYPRE_Int             *C_offd_j = hypre_CSRMatrixJ(C_offd);
   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int             *P_diag_j = hypre_CSRMatrixJ(P_diag);
   HYPRE_Int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int             *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Int             *new_C_diag_i;
   HYPRE_Int             *new_C_offd_i;
   HYPRE_Int	num_rows_diag_C = hypre_CSRMatrixNumRows(C_diag);
   HYPRE_Int	num_rows_offd_C = hypre_CSRMatrixNumCols(C_offd);
   HYPRE_Int num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(C_diag);
   HYPRE_Int num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(C_offd);
   double vmax = 0.0;
   double vmin = 0.0;
   double v, old_sum, new_sum, scale;
   HYPRE_Int i1, m, m1d, m1o, jC, mP, keep;

   /* Repack the i,j,and data arrays of C so as to discard those elements for which
      there is no corresponding element in P.
      Elements of Coarse rows (CF_marker>=0) are always kept.
      The arrays are not re-allocated, so there will generally be unused space
      at the ends of the arrays. */
   new_C_diag_i = hypre_CTAlloc( HYPRE_Int, num_rows_diag_C+1 );
   new_C_offd_i = hypre_CTAlloc( HYPRE_Int, num_rows_offd_C+1 );
   m1d = C_diag_i[0];
   m1o = C_offd_i[0];
   for ( i1 = 0; i1 < num_rows_diag_C; i1++ )
   {
      old_sum = 0;
      new_sum = 0;
      for ( m=C_diag_i[i1]; m<C_diag_i[i1+1]; ++m )
      {
         v = C_diag_data[m];
         jC = C_diag_j[m];
         old_sum += v;
         /* Do we know anything about the order of P_diag_j?  It would be better
            not to search through it all here.  If we know nothing, some ordering or
            index scheme will be needed for efficiency (worth doing iff this function
            gets called at all ) (may2006: this function is no longer called) */
         keep=0;
         for ( mP=P_diag_i[i1]; mP<P_diag_i[i1+1]; ++mP )
         {
            if ( jC==P_diag_j[m] )
            {
               keep=1;
               break;
            }
         }
         if ( CF_marker[i1]>=0 || keep==1 )
         {  /* keep v in C */
            new_sum += v;
            C_diag_j[m1d] = C_diag_j[m];
            C_diag_data[m1d] = C_diag_data[m];
            ++m1d;
         }
         else
         {  /* discard v */
            --num_nonzeros_diag;
         }
      }
      for ( m=C_offd_i[i1]; m<C_offd_i[i1+1]; ++m )
      {
         v = C_offd_data[m];
         jC = C_diag_j[m];
         old_sum += v;
         keep=0;
         for ( mP=P_offd_i[i1]; mP<P_offd_i[i1+1]; ++mP )
         {
            if ( jC==P_offd_j[m] )
            {
               keep=1;
               break;
            }
         }
         if ( CF_marker[i1]>=0 || v>=vmax || v<=vmin )
         {  /* keep v in C */
            new_sum += v;
            C_offd_j[m1o] = C_offd_j[m];
            C_offd_data[m1o] = C_offd_data[m];
            ++m1o;
         }
         else
         {  /* discard v */
            --num_nonzeros_offd;
         }
      }

      new_C_diag_i[i1+1] = m1d;
      if ( i1<num_rows_offd_C ) new_C_offd_i[i1+1] = m1o;

      /* rescale to keep row sum the same */
      if (new_sum!=0) scale = old_sum/new_sum; else scale = 1.0;
      for ( m=new_C_diag_i[i1]; m<new_C_diag_i[i1+1]; ++m )
         C_diag_data[m] *= scale;
      if ( i1<num_rows_offd_C ) /* this test fails when there is no offd block */
         for ( m=new_C_offd_i[i1]; m<new_C_offd_i[i1+1]; ++m )
            C_offd_data[m] *= scale;

   }

   for ( i1 = 1; i1 <= num_rows_diag_C; i1++ )
   {
      C_diag_i[i1] = new_C_diag_i[i1];
      if ( i1<num_rows_offd_C ) C_offd_i[i1] = new_C_offd_i[i1];
   }
   hypre_TFree( new_C_diag_i );
   if ( num_rows_offd_C>0 ) hypre_TFree( new_C_offd_i );

   hypre_CSRMatrixNumNonzeros(C_diag) = num_nonzeros_diag;
   hypre_CSRMatrixNumNonzeros(C_offd) = num_nonzeros_offd;
   /*  SetNumNonzeros, SetDNumNonzeros are global, need hypre_MPI_Allreduce.
       I suspect, but don't know, that other parts of hypre do not assume that
       the correct values have been set.
     hypre_ParCSRMatrixSetNumNonzeros( C );
     hypre_ParCSRMatrixSetDNumNonzeros( C );*/
   hypre_ParCSRMatrixNumNonzeros( C ) = 0;
   hypre_ParCSRMatrixDNumNonzeros( C ) = 0.0;

}

