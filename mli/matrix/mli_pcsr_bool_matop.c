/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#include "headers.h"

MLI_ParCSRBooleanMatrix *MLI_ParBooleanMatmul
( MLI_ParCSRBooleanMatrix *A,  MLI_ParCSRBooleanMatrix  *B )
{
   MPI_Comm 	   comm = MLI_ParCSRBooleanMatrix_Get_Comm(A);

   MLI_CSRBooleanMatrix *A_diag = MLI_ParCSRBooleanMatrix_Get_Diag(A);
   int             *A_diag_i = MLI_CSRBooleanMatrix_Get_I(A_diag);
   int             *A_diag_j = MLI_CSRBooleanMatrix_Get_J(A_diag);

   MLI_CSRBooleanMatrix *A_offd = MLI_ParCSRBooleanMatrix_Get_Offd(A);
   int             *A_offd_i = MLI_CSRBooleanMatrix_Get_I(A_offd);
   int             *A_offd_j = MLI_CSRBooleanMatrix_Get_J(A_offd);

   int *row_starts_A = MLI_ParCSRBooleanMatrix_Get_RowStarts(A);
   int	num_rows_diag_A = MLI_CSRBooleanMatrix_Get_NRows(A_diag);
   int	num_cols_offd_A = MLI_CSRBooleanMatrix_Get_NCols(A_offd);
   
   MLI_CSRBooleanMatrix *B_diag = MLI_ParCSRBooleanMatrix_Get_Diag(B);
   int             *B_diag_i = MLI_CSRBooleanMatrix_Get_I(B_diag);
   int             *B_diag_j = MLI_CSRBooleanMatrix_Get_J(B_diag);

   MLI_CSRBooleanMatrix *B_offd = MLI_ParCSRBooleanMatrix_Get_Offd(B);
   int		   *col_map_offd_B = MLI_ParCSRBooleanMatrix_Get_ColMapOffd(B);
   int             *B_offd_i = MLI_CSRBooleanMatrix_Get_I(B_offd);
   int             *B_offd_j = MLI_CSRBooleanMatrix_Get_J(B_offd);

   int	first_col_diag_B = MLI_ParCSRBooleanMatrix_Get_FirstColDiag(B);
   int *col_starts_B = MLI_ParCSRBooleanMatrix_Get_ColStarts(B);
   int	num_cols_diag_B = MLI_CSRBooleanMatrix_Get_NCols(B_diag);
   int	num_cols_offd_B = MLI_CSRBooleanMatrix_Get_NCols(B_offd);

   MLI_ParCSRBooleanMatrix *C;
   int		      *col_map_offd_C;

   MLI_CSRBooleanMatrix *C_diag;
   int             *C_diag_i;
   int             *C_diag_j;

   MLI_CSRBooleanMatrix *C_offd;
   int             *C_offd_i;
   int             *C_offd_j;

   int              C_diag_size;
   int              C_offd_size;
   int		    last_col_diag_C;
   int		    num_cols_offd_C;
   
   MLI_CSRBooleanMatrix *B_ext;
   int             *B_ext_i;
   int             *B_ext_j;

   int		   *B_marker;

   int              i;
   int              i1, i2, i3;
   int              jj2, jj3;
   
   int              jj_count_diag, jj_count_offd;
   int              jj_row_begin_diag, jj_row_begin_offd;
   int              start_indexing = 0; /* start indexing for C_data at 0 */
   int		    count;
   int		    n_rows_A, n_cols_A;
   int		    n_rows_B, n_cols_B;


   n_rows_A = MLI_ParCSRBooleanMatrix_Get_GlobalNRows(A);
   n_cols_A = MLI_ParCSRBooleanMatrix_Get_GlobalNCols(A);
   n_rows_B = MLI_ParCSRBooleanMatrix_Get_GlobalNRows(B);
   n_cols_B = MLI_ParCSRBooleanMatrix_Get_GlobalNCols(B);

   if (n_cols_A != n_rows_B)
   {
	printf(" Error! Incompatible matrix dimensions!\n");
	return NULL;
   }
   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product 
    *-----------------------------------------------------------------------*/

   if (num_rows_diag_A != n_rows_A) 
   {
       /*---------------------------------------------------------------------
    	* If there exists no CommPkg for A, a CommPkg is generated using
    	* equally load balanced partitionings
    	*--------------------------------------------------------------------*/
   	if (!MLI_ParCSRBooleanMatrix_Get_CommPkg(A))
   	{
        	hypre_MatvecCommPkgCreate(A);
   	}

   	B_ext = MLI_ParCSRBooleanMatrixExtractBExt(B,A);
   	B_ext_i    = MLI_CSRBooleanMatrix_Get_I(B_ext);
   	B_ext_j    = MLI_CSRBooleanMatrix_Get_J(B_ext);
   }
   /*-----------------------------------------------------------------------
   *  Allocate marker array.
    *-----------------------------------------------------------------------*/

   B_marker = hypre_CTAlloc(int, n_cols_B);

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   for (i1 = 0; i1 < n_cols_B; i1++)
   {      
      B_marker[i1] = -1;
   }


   hypre_ParMatmul_RowSizes(
      &C_diag_i, &C_offd_i, &B_marker,
      A_diag_i, A_diag_j, A_offd_i, A_offd_j,
      B_diag_i, B_diag_j, B_offd_i, B_offd_j,
      B_ext_i, B_ext_j, col_map_offd_B,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_cols_offd_A,
      first_col_diag_B, n_cols_B, num_cols_offd_B, num_cols_diag_B
      );


   /*-----------------------------------------------------------------------
    *  Allocate C_diag_j arrays.
    *  Allocate C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   C_diag_j    = hypre_CTAlloc(int, C_diag_size);
   if (C_offd_size)
   { 
   	C_offd_j    = hypre_CTAlloc(int, C_offd_size);
   } 


   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_j.
    *  Second Pass: Fill in C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < n_cols_B; i1++)
   {      
      B_marker[i1] = -1;
   }
   
   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/
    
   for (i1 = 0; i1 < num_cols_diag_B; i1++)
   {
      
      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1} 
       *--------------------------------------------------------------------*/

      B_marker[i1+first_col_diag_B] = jj_count_diag;
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      C_diag_j[jj_count_diag] = i1;
      jj_count_diag++;

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
	 if (num_cols_offd_A)
	 {
	  for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
          {
            i2 = A_offd_j[jj2];
            
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_ext.
                *-----------------------------------------------------------*/

               for (jj3 = B_ext_i[i2]; jj3 < B_ext_i[i2+1]; jj3++)
               {
                  i3 = B_ext_j[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/
		  if (i3 < first_col_diag_B || i3 > last_col_diag_C)
		  {
                     if (B_marker[i3] < jj_row_begin_offd)
                     {
                     	B_marker[i3] = jj_count_offd;
                     	C_offd_j[jj_count_offd] = i3;
                     	jj_count_offd++;
		     }
                  }
                  else
                  {
                     if (B_marker[i3] < jj_row_begin_diag)
                     {
                     	B_marker[i3] = jj_count_diag;
                     	C_diag_j[jj_count_diag] = i3-first_col_diag_B;
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
            
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_diag.
                *-----------------------------------------------------------*/

               for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
               {
                  i3 = B_diag_j[jj3]+first_col_diag_B;
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_j[jj_count_diag] = B_diag_j[jj3];
                     jj_count_diag++;
                  }
               }
               if (num_cols_offd_B)
	       {
		for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
                {
                  i3 = col_map_offd_B[B_offd_j[jj3]];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     C_offd_j[jj_count_offd] = i3;
                     jj_count_offd++;
                  }
                }
               }
         }
   }

   /*-----------------------------------------------------------------------
    *  Delete 0-columns in C_offd, i.e. generate col_map_offd and reset
    *  C_offd_j.
    *-----------------------------------------------------------------------*/

   for (i=0; i < C_offd_size; i++)
	B_marker[C_offd_j[i]] = -2;

   num_cols_offd_C = 0;
   for (i=0; i < n_cols_B; i++)
	if (B_marker[i] == -2) 
		num_cols_offd_C++;

   if (num_cols_offd_C)
	col_map_offd_C = hypre_CTAlloc(int,num_cols_offd_C);

   count = 0;
   for (i=0; i < n_cols_B; i++)
	if (B_marker[i] == -2) 
	{
		col_map_offd_C[count] = i;
		B_marker[i] = count;
		count++;
	}

   for (i=0; i < C_offd_size; i++)
	C_offd_j[i] = B_marker[C_offd_j[i]];

   C = MLI_ParCSRBooleanMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
	col_starts_B, num_cols_offd_C, C_diag_size, C_offd_size);

/* Note that C does not own the partitionings */
   MLI_ParCSRBooleanMatrixSetRowStartsOwner(C,0);
   MLI_ParCSRBooleanMatrixSetColStartsOwner(C,0);

   C_diag = MLI_ParCSRBooleanMatrix_Get_Diag(C);
   MLI_CSRBooleanMatrix_Get_I(C_diag) = C_diag_i; 
   MLI_CSRBooleanMatrix_Get_J(C_diag) = C_diag_j; 

   if (num_cols_offd_C)
   {
      C_offd = MLI_ParCSRBooleanMatrix_Get_Offd(C);
      MLI_CSRBooleanMatrix_Get_I(C_offd) = C_offd_i; 
      MLI_CSRBooleanMatrix_Get_J(C_offd) = C_offd_j; 
      MLI_ParCSRBooleanMatrix_Get_Offd(C) = C_offd;
      MLI_ParCSRBooleanMatrix_Get_ColMapOffd(C) = col_map_offd_C;

   }
   else
	hypre_TFree(C_offd_i);

   /*-----------------------------------------------------------------------
    *  Free B_ext and marker array.
    *-----------------------------------------------------------------------*/

   if (num_cols_offd_A)
   {
      MLI_CSRBooleanMatrixDestroy(B_ext);
      B_ext = NULL;
   }
   hypre_TFree(B_marker);   

   return C;
   
}            



/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixExtractBExt :
 * extracts rows from B which are located on other
 * processors and needed for multiplication with A locally. The rows
 * are returned as CSRBooleanMatrix.
 *--------------------------------------------------------------------------*/

MLI_CSRBooleanMatrix * 
MLI_ParCSRBooleanMatrixExtractBExt
( MLI_ParCSRBooleanMatrix *B, MLI_ParCSRBooleanMatrix *A )
{
   MPI_Comm comm = MLI_ParCSRBooleanMatrix_Get_Comm(B);
   int first_col_diag = MLI_ParCSRBooleanMatrix_Get_FirstColDiag(B);
   int *col_map_offd = MLI_ParCSRBooleanMatrix_Get_ColMapOffd(B);

   hypre_ParCSRCommPkg *comm_pkg = MLI_ParCSRBooleanMatrix_Get_CommPkg(A);
   int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   MLI_CSRBooleanMatrix *diag = MLI_ParCSRBooleanMatrix_Get_Diag(B);
   int *diag_i = MLI_CSRBooleanMatrix_Get_I(diag);
   int *diag_j = MLI_CSRBooleanMatrix_Get_J(diag);

   MLI_CSRBooleanMatrix *offd = MLI_ParCSRBooleanMatrix_Get_Offd(B);
   int *offd_i = MLI_CSRBooleanMatrix_Get_I(offd);
   int *offd_j = MLI_CSRBooleanMatrix_Get_J(offd);

   int num_cols_B, num_nonzeros;
   int num_rows_B_ext;

   MLI_CSRBooleanMatrix *B_ext;
   int *B_ext_i;
   int *B_ext_j;

   int *B_ext_data, *diag_data=NULL, *offd_data=NULL;
   /* ... not referenced, but needed for function call */
 
   num_cols_B = MLI_ParCSRBooleanMatrix_Get_GlobalNCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   hypre_ParCSRMatrixExtractBExt_Arrays
      ( &B_ext_i, &B_ext_j, &B_ext_data,
        &num_nonzeros,
        0, comm, comm_pkg,
        num_cols_B, num_recvs, num_sends,
        first_col_diag,
        recv_vec_starts, send_map_starts, send_map_elmts,
        diag_i, diag_j, offd_i, offd_j, col_map_offd,
        diag_data, offd_data
         );

   B_ext = MLI_CSRBooleanMatrixCreate(num_rows_B_ext,num_cols_B,num_nonzeros);
   MLI_CSRBooleanMatrix_Get_I(B_ext) = B_ext_i;
   MLI_CSRBooleanMatrix_Get_J(B_ext) = B_ext_j;

   return B_ext;
}
