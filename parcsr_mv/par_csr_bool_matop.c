/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#include "headers.h"

hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul
( hypre_ParCSRBooleanMatrix *A,  hypre_ParCSRBooleanMatrix  *B )
{
   MPI_Comm 	   comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);

   hypre_CSRBooleanMatrix *A_diag = hypre_ParCSRBooleanMatrix_Get_Diag(A);
   int             *A_diag_i = hypre_CSRBooleanMatrix_Get_I(A_diag);
   int             *A_diag_j = hypre_CSRBooleanMatrix_Get_J(A_diag);

   hypre_CSRBooleanMatrix *A_offd = hypre_ParCSRBooleanMatrix_Get_Offd(A);
   int             *A_offd_i = hypre_CSRBooleanMatrix_Get_I(A_offd);
   int             *A_offd_j = hypre_CSRBooleanMatrix_Get_J(A_offd);

   int *row_starts_A = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   int	num_rows_diag_A = hypre_CSRBooleanMatrix_Get_NRows(A_diag);
   int	num_cols_offd_A = hypre_CSRBooleanMatrix_Get_NCols(A_offd);
   
   hypre_CSRBooleanMatrix *B_diag = hypre_ParCSRBooleanMatrix_Get_Diag(B);
   int             *B_diag_i = hypre_CSRBooleanMatrix_Get_I(B_diag);
   int             *B_diag_j = hypre_CSRBooleanMatrix_Get_J(B_diag);

   hypre_CSRBooleanMatrix *B_offd = hypre_ParCSRBooleanMatrix_Get_Offd(B);
   int		   *col_map_offd_B = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(B);
   int             *B_offd_i = hypre_CSRBooleanMatrix_Get_I(B_offd);
   int             *B_offd_j = hypre_CSRBooleanMatrix_Get_J(B_offd);

   int	first_col_diag_B = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(B);
   int *col_starts_B = hypre_ParCSRBooleanMatrix_Get_ColStarts(B);
   int	num_cols_diag_B = hypre_CSRBooleanMatrix_Get_NCols(B_diag);
   int	num_cols_offd_B = hypre_CSRBooleanMatrix_Get_NCols(B_offd);

   hypre_ParCSRBooleanMatrix *C;
   int		      *col_map_offd_C;

   hypre_CSRBooleanMatrix *C_diag;
   int             *C_diag_i;
   int             *C_diag_j;

   hypre_CSRBooleanMatrix *C_offd;
   int             *C_offd_i=NULL;
   int             *C_offd_j=NULL;

   int              C_diag_size;
   int              C_offd_size;
   int		    last_col_diag_C;
   int		    num_cols_offd_C;
   
   hypre_CSRBooleanMatrix *B_ext;
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
   int              allsquare = 0;

   n_rows_A = hypre_ParCSRBooleanMatrix_Get_GlobalNRows(A);
   n_cols_A = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);
   n_rows_B = hypre_ParCSRBooleanMatrix_Get_GlobalNRows(B);
   n_cols_B = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(B);

   if (n_cols_A != n_rows_B)
   {
	printf(" Error! Incompatible matrix dimensions!\n");
	return NULL;
   }
   if ( n_rows_A==n_cols_A && n_rows_B==n_cols_B ) allsquare = 1;

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
   	if (!hypre_ParCSRBooleanMatrix_Get_CommPkg(A))
   	{
        	hypre_BooleanMatvecCommPkgCreate(A);
   	}

   	B_ext = hypre_ParCSRBooleanMatrixExtractBExt(B,A);
   	B_ext_i    = hypre_CSRBooleanMatrix_Get_I(B_ext);
   	B_ext_j    = hypre_CSRBooleanMatrix_Get_J(B_ext);
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
      num_rows_diag_A, num_cols_offd_A, allsquare,
      first_col_diag_B, n_cols_B, num_cols_offd_B, num_cols_diag_B
      );


   /*-----------------------------------------------------------------------
    *  Allocate C_diag_j arrays.
    *  Allocate C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   last_col_diag_C = first_col_diag_B + num_cols_diag_B - 1;
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
    
   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {
      
      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1} 
       *--------------------------------------------------------------------*/

      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare ) {
         B_marker[i1+first_col_diag_B] = jj_count_diag;
         C_diag_j[jj_count_diag] = i1;
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

   C = hypre_ParCSRBooleanMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
	col_starts_B, num_cols_offd_C, C_diag_size, C_offd_size);

/* Note that C does not own the partitionings */
   hypre_ParCSRBooleanMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRBooleanMatrixSetColStartsOwner(C,0);

   C_diag = hypre_ParCSRBooleanMatrix_Get_Diag(C);
   hypre_CSRBooleanMatrix_Get_I(C_diag) = C_diag_i; 
   hypre_CSRBooleanMatrix_Get_J(C_diag) = C_diag_j; 

   if (num_cols_offd_C)
   {
      C_offd = hypre_ParCSRBooleanMatrix_Get_Offd(C);
      hypre_CSRBooleanMatrix_Get_I(C_offd) = C_offd_i; 
      hypre_CSRBooleanMatrix_Get_J(C_offd) = C_offd_j; 
      hypre_ParCSRBooleanMatrix_Get_Offd(C) = C_offd;
      hypre_ParCSRBooleanMatrix_Get_ColMapOffd(C) = col_map_offd_C;

   }
   else
	hypre_TFree(C_offd_i);

   /*-----------------------------------------------------------------------
    *  Free B_ext and marker array.
    *-----------------------------------------------------------------------*/

   if (num_cols_offd_A)
   {
      hypre_CSRBooleanMatrixDestroy(B_ext);
      B_ext = NULL;
   }
   hypre_TFree(B_marker);   

   return C;
   
}            



/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixExtractBExt :
 * extracts rows from B which are located on other
 * processors and needed for multiplication with A locally. The rows
 * are returned as CSRBooleanMatrix.
 *--------------------------------------------------------------------------*/

hypre_CSRBooleanMatrix * 
hypre_ParCSRBooleanMatrixExtractBExt
( hypre_ParCSRBooleanMatrix *B, hypre_ParCSRBooleanMatrix *A )
{
   MPI_Comm comm = hypre_ParCSRBooleanMatrix_Get_Comm(B);
   int first_col_diag = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(B);
   int first_row_index = hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(B);
   int *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRBooleanMatrix_Get_CommPkg(A);
   int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   hypre_CSRBooleanMatrix *diag = hypre_ParCSRBooleanMatrix_Get_Diag(B);
   int *diag_i = hypre_CSRBooleanMatrix_Get_I(diag);
   int *diag_j = hypre_CSRBooleanMatrix_Get_J(diag);

   hypre_CSRBooleanMatrix *offd = hypre_ParCSRBooleanMatrix_Get_Offd(B);
   int *offd_i = hypre_CSRBooleanMatrix_Get_I(offd);
   int *offd_j = hypre_CSRBooleanMatrix_Get_J(offd);

   int num_cols_B, num_nonzeros;
   int num_rows_B_ext;

   hypre_CSRBooleanMatrix *B_ext;
   int *B_ext_i;
   int *B_ext_j;

   double *B_ext_data=NULL, *diag_data=NULL, *offd_data=NULL;
   int *B_ext_row_map=NULL;
   /* ... not referenced, but needed for function call */
 
   num_cols_B = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   hypre_ParCSRMatrixExtractBExt_Arrays
      ( &B_ext_i, &B_ext_j, &B_ext_data, &B_ext_row_map,
        &num_nonzeros,
        0, 0, comm, comm_pkg,
        num_cols_B, num_recvs, num_sends,
        first_col_diag, first_row_index,
        recv_vec_starts, send_map_starts, send_map_elmts,
        diag_i, diag_j, offd_i, offd_j, col_map_offd,
        diag_data, offd_data
         );

   B_ext = hypre_CSRBooleanMatrixCreate(num_rows_B_ext,num_cols_B,num_nonzeros);
   hypre_CSRBooleanMatrix_Get_I(B_ext) = B_ext_i;
   hypre_CSRBooleanMatrix_Get_J(B_ext) = B_ext_j;

   return B_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixExtractAExt : extracts rows from A which are located on other
 * processors and needed for multiplying A^T with the local part of A. The rows
 * are returned as CSRBooleanMatrix.  A row map for A_ext (like the ParCSRColMap) is
 * returned through the third argument.
 *--------------------------------------------------------------------------*/

hypre_CSRBooleanMatrix * 
hypre_ParCSRBooleanMatrixExtractAExt( hypre_ParCSRBooleanMatrix *A,
                                    int ** pA_ext_row_map )
{
   /* Note that A's role as the first factor in A*A^T is used only
      through ...CommPkgT(A), which basically says which rows of A
      (columns of A^T) are needed.  In all the other places where A
      serves as an input, it is through its role as A^T, the matrix
      whose data needs to be passed between processors. */
   MPI_Comm comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);
   int first_col_diag = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   int first_row_index = hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(A);
   int *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRBooleanMatrix_Get_CommPkgT(A);
   /* ... CommPkgT(A) should identify all rows of A^T needed for A*A^T (that is
    * generally a bigger set than ...CommPkg(A), the rows of B needed for A*B) */
   int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   hypre_CSRBooleanMatrix *diag = hypre_ParCSRBooleanMatrix_Get_Diag(A);

   int *diag_i = hypre_CSRMatrixI(diag);
   int *diag_j = hypre_CSRMatrixJ(diag);

   hypre_CSRBooleanMatrix *offd = hypre_ParCSRBooleanMatrix_Get_Offd(A);

   int *offd_i = hypre_CSRMatrixI(offd);
   int *offd_j = hypre_CSRMatrixJ(offd);

   int num_cols_A, num_nonzeros;
   int num_rows_A_ext;

   hypre_CSRBooleanMatrix *A_ext;

   int *A_ext_i;
   int *A_ext_j;

   int data = 0;
   double *A_ext_data = NULL, *diag_data=NULL, *offd_data=NULL;
   /* ... not referenced, but needed for function call */
 
   num_cols_A = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);
   num_rows_A_ext = recv_vec_starts[num_recvs];

   hypre_ParCSRMatrixExtractBExt_Arrays
      ( &A_ext_i, &A_ext_j, &A_ext_data, pA_ext_row_map,
        &num_nonzeros,
        data, 1, comm, comm_pkg,
        num_cols_A, num_recvs, num_sends,
        first_col_diag, first_row_index,
        recv_vec_starts, send_map_starts, send_map_elmts,
        diag_i, diag_j, offd_i, offd_j, col_map_offd,
        diag_data, offd_data
         );

   A_ext = hypre_CSRBooleanMatrixCreate(num_rows_A_ext,num_cols_A,num_nonzeros);
   hypre_CSRBooleanMatrix_Get_I(A_ext) = A_ext_i;
   hypre_CSRBooleanMatrix_Get_J(A_ext) = A_ext_j;

   return A_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParBooleanAAT : multiplies hypre_ParCSRBooleanMatrix A by its transpose,
 * A*A^T, and returns the product in hypre_ParCSRBooleanMatrix C
 * Note that C does not own the partitionings
 * This is based on hypre_ParCSRAAt.
 *--------------------------------------------------------------------------*/

hypre_ParCSRBooleanMatrix * hypre_ParBooleanAAt( hypre_ParCSRBooleanMatrix  * A )
{
   MPI_Comm 	   comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);

   hypre_CSRBooleanMatrix *A_diag = hypre_ParCSRBooleanMatrix_Get_Diag(A);
   
   int             *A_diag_i = hypre_CSRBooleanMatrix_Get_I(A_diag);
   int             *A_diag_j = hypre_CSRBooleanMatrix_Get_J(A_diag);

   hypre_CSRBooleanMatrix *A_offd = hypre_ParCSRBooleanMatrix_Get_Offd(A);
   int             *A_offd_i = hypre_CSRBooleanMatrix_Get_I(A_offd);
   int             *A_offd_j = hypre_CSRBooleanMatrix_Get_J(A_offd);

   int             *A_col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   int             * A_ext_row_map;

   int *row_starts_A = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   int	num_rows_diag_A = hypre_CSRBooleanMatrix_Get_NRows(A_diag);
   int	num_cols_offd_A = hypre_CSRBooleanMatrix_Get_NCols(A_offd);
   
   hypre_ParCSRBooleanMatrix *C;
   int		      *col_map_offd_C;

   hypre_CSRBooleanMatrix *C_diag;

   int             *C_diag_i;
   int             *C_diag_j;

   hypre_CSRBooleanMatrix *C_offd;

   int             *C_offd_i=NULL;
   int             *C_offd_j=NULL;
   int             *new_C_offd_j;

   int              C_diag_size;
   int              C_offd_size;
   int		    last_col_diag_C;
   int		    num_cols_offd_C;
   
   hypre_CSRBooleanMatrix *A_ext;
   
   int             *A_ext_i;
   int             *A_ext_j;
   int             num_rows_A_ext=0;

   int	first_row_index_A = hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(A);
   int	first_col_diag_A = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   int		   *B_marker;

   int              i;
   int              i1, i2, i3;
   int              jj2, jj3;
   
   int              jj_count_diag, jj_count_offd;
   int              jj_row_begin_diag, jj_row_begin_offd;
   int              start_indexing = 0; /* start indexing for C_data at 0 */
   int		    count;
   int		    n_rows_A, n_cols_A;

   n_rows_A = hypre_ParCSRBooleanMatrix_Get_GlobalNRows(A);
   n_cols_A = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);

   if (n_cols_A != n_rows_A)
   {
	printf(" Error! Incompatible matrix dimensions!\n");
	return NULL;
   }
   /*-----------------------------------------------------------------------
    *  Extract A_ext, i.e. portion of A that is stored on neighbor procs
    *  and needed locally for A^T in the matrix matrix product A*A^T
    *-----------------------------------------------------------------------*/

   if (num_rows_diag_A != n_rows_A) 
   {
       /*---------------------------------------------------------------------
    	* If there exists no CommPkg for A, a CommPkg is generated using
    	* equally load balanced partitionings
    	*--------------------------------------------------------------------*/
   	if (!hypre_ParCSRBooleanMatrix_Get_CommPkg(A))
   	{
        	hypre_BooleanMatTCommPkgCreate(A);
   	}

   	A_ext = hypre_ParCSRBooleanMatrixExtractAExt( A, &A_ext_row_map );
   	A_ext_i    = hypre_CSRBooleanMatrix_Get_I(A_ext);
   	A_ext_j    = hypre_CSRBooleanMatrix_Get_J(A_ext);
        num_rows_A_ext = hypre_CSRBooleanMatrix_Get_NRows(A_ext);
   }
   /*-----------------------------------------------------------------------
   *  Allocate marker array.
    *-----------------------------------------------------------------------*/

   B_marker = hypre_CTAlloc(int, num_rows_diag_A+num_rows_A_ext );

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   for ( i1=0; i1<num_rows_diag_A+num_rows_A_ext; ++i1 )
   {      
      B_marker[i1] = -1;
   }


   hypre_ParAat_RowSizes(
      &C_diag_i, &C_offd_i, B_marker,
      A_diag_i, A_diag_j,
      A_offd_i, A_offd_j, A_col_map_offd,
      A_ext_i, A_ext_j, A_ext_row_map,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_cols_offd_A,
      num_rows_A_ext,
      first_col_diag_A, first_row_index_A
      );

#if 0
/* debugging output: */
   printf("A_ext_row_map (%i):",num_rows_A_ext);
   for ( i1=0; i1<num_rows_A_ext; ++i1 ) printf(" %i",A_ext_row_map[i1] );
   printf("\nC_diag_i (%i):",C_diag_size);
   for ( i1=0; i1<=num_rows_diag_A; ++i1 ) printf(" %i",C_diag_i[i1] );
   printf("\nC_offd_i (%i):",C_offd_size);
   for ( i1=0; i1<=num_rows_diag_A; ++i1 ) printf(" %i",C_offd_i[i1] );
   printf("\n");
#endif

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_j arrays.
    *  Allocate C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   last_col_diag_C = first_row_index_A + num_rows_diag_A - 1;
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
   for ( i1=0; i1<num_rows_diag_A+num_rows_A_ext; ++i1 )
   {      
      B_marker[i1] = -1;
   }
   
   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/
    
   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {
      
      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1} 
       *--------------------------------------------------------------------*/

      B_marker[i1] = jj_count_diag;
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      C_diag_j[jj_count_diag] = i1;
      jj_count_diag++;

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/
         
      /* There are 3 CSRMatrix or CSRBooleanMatrix objects here:
         A_diag, A_offd, and A_ext.  That's 9 possiable X*Y combinations.  But
         ext*ext, ext*diag, and ext*offd belong to another processor.
         diag*offd and offd*diag don't count - never share a column by definition.
         So we have to do 4 cases:
         diag*ext, offd*ext, diag*diag, and offd*offd.
      */

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
      {
         i2 = A_diag_j[jj2];
            
         /* diag*ext */
         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
          *  That is, rows i3 having a column i2 of A_ext.
          *  For now, for each row i3 of A_ext we crudely check _all_
          *  columns to see whether one matches i2.
          *  For each entry (i2,i3) of (A_ext)^T, A(i1,i2)*A(i3,i2) defines
          *  C(i1,i3) .  This contributes to both the diag and offd
          *  blocks of C.
          *-----------------------------------------------------------*/

         for ( i3=0; i3<num_rows_A_ext; i3++ ) {
            for ( jj3=A_ext_i[i3]; jj3<A_ext_i[i3+1]; jj3++ ) {
               if ( A_ext_j[jj3]==i2+first_col_diag_A ) {
                  /* row i3, column i2 of A_ext; or,
                     row i2, column i3 of (A_ext)^T */

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *--------------------------------------------------------*/

                  if ( A_ext_row_map[i3] < first_row_index_A ||
                       A_ext_row_map[i3] > last_col_diag_C ) { /* offd */
                     if (B_marker[i3+num_rows_diag_A] < jj_row_begin_offd) {
                        B_marker[i3+num_rows_diag_A] = jj_count_offd;
                        C_offd_j[jj_count_offd] = i3;
                        jj_count_offd++;
                     }
                  }
                  else {                                              /* diag */
                     if (B_marker[i3+num_rows_diag_A] < jj_row_begin_diag) {
                        B_marker[i3+num_rows_diag_A] = jj_count_diag;
                        C_diag_j[jj_count_diag] = i3-first_col_diag_A;
                        jj_count_diag++;
                     }
                  } 
               }
            }
         }
      }

      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];
            
            /* offd * ext */
            /*-----------------------------------------------------------
             *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
             *  That is, rows i3 having a column i2 of A_ext.
             *  For now, for each row i3 of A_ext we crudely check _all_
             *  columns to see whether one matches i2.
             *  For each entry (i2,i3) of (A_ext)^T, A(i1,i2)*A(i3,i2) defines
             *  C(i1,i3) .  This contributes to both the diag and offd
             *  blocks of C.
             *-----------------------------------------------------------*/

            for ( i3=0; i3<num_rows_A_ext; i3++ ) {
               for ( jj3=A_ext_i[i3]; jj3<A_ext_i[i3+1]; jj3++ ) {
                  if ( A_ext_j[jj3]==A_col_map_offd[i2] ) {
                     /* row i3, column i2 of A_ext; or,
                        row i2, column i3 of (A_ext)^T */

                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution.
                      *--------------------------------------------------------*/

                     if ( A_ext_row_map[i3] < first_row_index_A ||
                          A_ext_row_map[i3] > last_col_diag_C ) { /* offd */
                        if (B_marker[i3+num_rows_diag_A] < jj_row_begin_offd) {
                           B_marker[i3+num_rows_diag_A] = jj_count_offd;
                           C_offd_j[jj_count_offd] = i3;
                           jj_count_offd++;
                        }
                     }
                     else {                                              /* diag */
                        if (B_marker[i3+num_rows_diag_A] < jj_row_begin_diag) {
                           B_marker[i3+num_rows_diag_A] = jj_count_diag;
                           C_diag_j[jj_count_diag] = i3-first_row_index_A;
                           jj_count_diag++;
                        }
                     } 
                  }
               }
            }
         }
      }

      /* diag * diag */
      /*-----------------------------------------------------------------
       *  Loop over entries (columns) i2 in row i1 of A_diag.
       *  For each such column we will find the contributions of the
       *  corresponding rows i2 of A^T to C=A*A^T .  Now we only look
       *  at the local part of A^T - with columns (rows of A) living
       *  on this processor.
       *-----------------------------------------------------------------*/
         
      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
      {
         i2 = A_diag_j[jj2];
 
         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of A^T
          *  That is, rows i3 having a column i2 of A (local part).
          *  For now, for each row i3 of A we crudely check _all_
          *  columns to see whether one matches i2.
          *  This i3-loop is for the diagonal block of A.
          *  It contributes to the diagonal block of C.
          *  For each entry (i2,i3) of A^T, A(i1,i2)*A(i3,i2) defines
          *  to C(i1,i3)
          *-----------------------------------------------------------*/
         for ( i3=0; i3<num_rows_diag_A; i3++ ) {
            for ( jj3=A_diag_i[i3]; jj3<A_diag_i[i3+1]; jj3++ ) {
               if ( A_diag_j[jj3]==i2 ) {
                  /* row i3, column i2 of A; or,
                     row i2, column i3 of A^T */

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_j[jj_count_diag] = i3;
                     jj_count_diag++;
                  }
               }
            }
         } /* end of i3 loop */
      } /* end of third i2 loop */


         /* offd * offd */
         /*-----------------------------------------------------------
          *  Loop over offd columns i2 of A in A*A^T.  Then
          *  loop over offd entries (columns) i3 in row i2 of A^T
          *  That is, rows i3 having a column i2 of A (local part).
          *  For now, for each row i3 of A we crudely check _all_
          *  columns to see whether one matches i2.
          *  This i3-loop is for the off-diagonal block of A.
          *  It contributes to the diag block of C.
          *  For each entry (i2,i3) of A^T, A*A^T defines C
          *-----------------------------------------------------------*/
      if (num_cols_offd_A) {

         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];
            
            for ( i3=0; i3<num_rows_diag_A; i3++ ) {
               /* ... note that num_rows_diag_A == num_rows_offd_A */
               for ( jj3=A_offd_i[i3]; jj3<A_offd_i[i3+1]; jj3++ ) {
                     if ( A_offd_j[jj3]==i2 ) {
                     /* row i3, column i2 of A; or,
                        row i2, column i3 of A^T */

                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution
                      *--------------------------------------------------------*/
 
                     if (B_marker[i3] < jj_row_begin_diag)
                     {
                        B_marker[i3] = jj_count_diag;
                        C_diag_j[jj_count_diag] = i3;
                        jj_count_diag++;
                     }
                  }
               }
            }  /* end of last i3 loop */
         }     /* end of if (num_cols_offd_A) */

      }        /* end of fourth and last i2 loop */
#if 0          /* debugging printout */
         printf("end of i1 loop: i1=%i jj_count_diag=%i\n", i1, jj_count_diag );
         printf("  C_diag_j=");
         for ( jj3=0; jj3<jj_count_diag; ++jj3) printf("%i ",C_diag_j[jj3]);
         printf("\n");
         printf("  C_offd_j=");
         for ( jj3=0; jj3<jj_count_offd; ++jj3) printf("%i ",C_offd_j[jj3]);
         printf("\n");
         printf( "  B_marker =" );
         for ( it=0; it<num_rows_diag_A+num_rows_A_ext; ++it )
            printf(" %i", B_marker[it] );
         printf( "\n" );
#endif
   }           /* end of i1 loop */

   /*-----------------------------------------------------------------------
    *  Delete 0-columns in C_offd, i.e. generate col_map_offd and reset
    *  C_offd_j.  Note that (with the indexing we have coming into this
    *  block) col_map_offd_C[i3]==A_ext_row_map[i3].
    *-----------------------------------------------------------------------*/

   for ( i=0; i<num_rows_diag_A+num_rows_A_ext; ++i )
      B_marker[i] = -1;
   for ( i=0; i<C_offd_size; i++ )
      B_marker[ C_offd_j[i] ] = -2;

   count = 0;
   for (i=0; i < num_rows_diag_A + num_rows_A_ext; i++) {
      if (B_marker[i] == -2) {
         B_marker[i] = count;
         count++;
      }
   }
   num_cols_offd_C = count;

   if (num_cols_offd_C) {
      col_map_offd_C = hypre_CTAlloc(int,num_cols_offd_C);
      new_C_offd_j = hypre_CTAlloc(int,C_offd_size);
      /* ... a bit big, but num_cols_offd_C is too small.  It might be worth
         computing the correct size, which is sum( no. columns in row i, over all rows i )
      */

      for (i=0; i < C_offd_size; i++) {
         new_C_offd_j[i] = B_marker[C_offd_j[i]];
         col_map_offd_C[ new_C_offd_j[i] ] = A_ext_row_map[ C_offd_j[i] ];
      }

      hypre_TFree(C_offd_j);
      C_offd_j = new_C_offd_j;

   }

   /*---------------------------------------------------------------- 
    * Create C
    *----------------------------------------------------------------*/

   C = hypre_ParCSRBooleanMatrixCreate(comm, n_rows_A, n_rows_A, row_starts_A,
	row_starts_A, num_cols_offd_C, C_diag_size, C_offd_size);

/* Note that C does not own the partitionings */
   hypre_ParCSRBooleanMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRBooleanMatrixSetColStartsOwner(C,0);

   C_diag = hypre_ParCSRBooleanMatrix_Get_Diag(C);
   hypre_CSRBooleanMatrix_Get_I(C_diag) = C_diag_i; 
   hypre_CSRBooleanMatrix_Get_J(C_diag) = C_diag_j; 

   if (num_cols_offd_C)
   {
      C_offd = hypre_ParCSRBooleanMatrix_Get_Offd(C);
      hypre_CSRBooleanMatrix_Get_I(C_offd) = C_offd_i; 
      hypre_CSRBooleanMatrix_Get_J(C_offd) = C_offd_j; 
      hypre_ParCSRBooleanMatrix_Get_Offd(C) = C_offd;
      hypre_ParCSRBooleanMatrix_Get_ColMapOffd(C) = col_map_offd_C;

   }
   else
	hypre_TFree(C_offd_i);

   /*-----------------------------------------------------------------------
    *  Free B_ext and marker array.
    *-----------------------------------------------------------------------*/

   if (num_cols_offd_A)
   {
      hypre_CSRBooleanMatrixDestroy(A_ext);
      A_ext = NULL;
   }
   hypre_TFree(B_marker);
   if ( num_rows_diag_A != n_rows_A )
      hypre_TFree(A_ext_row_map);

   return C;
   
}            


/* ----------------------------------------------------------------------
 * hypre_BooleanMatTCommPkgCreate
 * generates a special comm_pkg for a Boolean matrix A - for use in multiplying
 * by its transpose, A * A^T
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d 
 * ---------------------------------------------------------------------*/

int
hypre_BooleanMatTCommPkgCreate ( hypre_ParCSRBooleanMatrix *A)
{
   hypre_ParCSRCommPkg	*comm_pkg;
   
   MPI_Comm             comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);
/*   MPI_Datatype         *recv_mpi_types;
   MPI_Datatype         *send_mpi_types;
*/
   int			num_sends;
   int			*send_procs;
   int			*send_map_starts;
   int			*send_map_elmts;
   int			num_recvs;
   int			*recv_procs;
   int			*recv_vec_starts;
   
   int  *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   int  first_col_diag = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   int  *col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(A);

   int	ierr = 0;
   int	num_rows_diag = hypre_CSRBooleanMatrix_Get_NRows(hypre_ParCSRBooleanMatrix_Get_Diag(A));
   int	num_cols_diag = hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Diag(A));
   int	num_cols_offd = hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Offd(A));
   int * row_starts = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);

   hypre_MatTCommPkgCreate_core (
      comm, col_map_offd, first_col_diag, col_starts,
      num_rows_diag, num_cols_diag, num_cols_offd, row_starts,
      hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A),
      hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A),
      hypre_CSRBooleanMatrix_Get_I( hypre_ParCSRBooleanMatrix_Get_Diag(A) ),
      hypre_CSRBooleanMatrix_Get_J( hypre_ParCSRBooleanMatrix_Get_Diag(A) ),
      hypre_CSRBooleanMatrix_Get_I( hypre_ParCSRBooleanMatrix_Get_Offd(A) ),
      hypre_CSRBooleanMatrix_Get_J( hypre_ParCSRBooleanMatrix_Get_Offd(A) ),
      0,
      &num_recvs, &recv_procs, &recv_vec_starts,
      &num_sends, &send_procs, &send_map_starts,
      &send_map_elmts
      );

   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;

   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts;

   hypre_ParCSRBooleanMatrix_Get_CommPkgT(A) = comm_pkg;

   return ierr;
}


/* ----------------------------------------------------------------------
 * hypre_BooleanMatvecCommPkgCreate
 * generates the comm_pkg for a Boolean matrix A , to be used for A*B.
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d 
 * ---------------------------------------------------------------------*/

int
hypre_BooleanMatvecCommPkgCreate ( hypre_ParCSRBooleanMatrix *A)
{
   hypre_ParCSRCommPkg	*comm_pkg;
   
   MPI_Comm             comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);
/*   MPI_Datatype         *recv_mpi_types;
   MPI_Datatype         *send_mpi_types;
*/
   int			num_sends;
   int			*send_procs;
   int			*send_map_starts;
   int			*send_map_elmts;
   int			num_recvs;
   int			*recv_procs;
   int			*recv_vec_starts;
   
   int  *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   int  first_col_diag = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   int  *col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(A);

   int	ierr = 0;
   int	num_cols_diag = hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Diag(A));
   int	num_cols_offd = hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Offd(A));

   hypre_MatvecCommPkgCreate_core
      (
         comm, col_map_offd, first_col_diag, col_starts,
         num_cols_diag, num_cols_offd,
         first_col_diag, col_map_offd,
         1,
         &num_recvs, &recv_procs, &recv_vec_starts,
         &num_sends, &send_procs, &send_map_starts,
         &send_map_elmts
         );

   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;

   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   /* hypre_ParCSRCommPkgRecvMPITypes(comm_pkg) = recv_mpi_types; */

   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts;
   /* hypre_ParCSRCommPkgSendMPITypes(comm_pkg) = send_mpi_types; */

   hypre_ParCSRBooleanMatrix_Get_CommPkg(A) = comm_pkg;

   return ierr;
}
