/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#include "headers.h"

/* The following function was formerly part of hypre_ParMatmul
   but was removed so it can also be used for multiplication of
   Boolean matrices
*/

void hypre_ParMatmul_RowSizes
( int ** C_diag_i, int ** C_offd_i, int ** B_marker,
  int * A_diag_i, int * A_diag_j, int * A_offd_i, int * A_offd_j,
  int * B_diag_i, int * B_diag_j, int * B_offd_i, int * B_offd_j,
  int * B_ext_i, int * B_ext_j, int * col_map_offd_B,
  int *C_diag_size, int *C_offd_size,
  int num_rows_diag_A, int num_cols_offd_A,
  int first_col_diag_B, int n_cols_B, int num_cols_offd_B, int num_cols_diag_B
   )
{
   int i1, i2, i3, jj2, jj3;
   int jj_count_diag, jj_count_offd, jj_row_begin_diag, jj_row_begin_offd;
   int last_col_diag_C;
   int start_indexing = 0; /* start indexing for C_data at 0 */
   /* First pass begins here.  Computes sizes of C rows.
      Arrays computed: C_diag_i, C_offd_i, B_marker
      Arrays needed: (11, all int*)
        A_diag_i, A_diag_j, A_offd_i, A_offd_j,
        B_diag_i, B_diag_j, B_offd_i, B_offd_j,
        B_ext_i, B_ext_j, col_map_offd_B,
        col_map_offd_B, B_offd_i, B_offd_j, B_ext_i, B_ext_j,
      Scalars computed: C_diag_size, C_offd_size
      Scalars needed:
      num_rows_diag_A, num_rows_diag_A, num_cols_offd_A,
      first_col_diag_B, n_cols_B, num_cols_offd_B, num_cols_diag_B
   */

   *C_diag_i = hypre_CTAlloc(int, num_cols_diag_B+1);
   *C_offd_i = hypre_CTAlloc(int, num_cols_diag_B+1);

   last_col_diag_C = first_col_diag_B + num_cols_diag_B - 1;

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < n_cols_B; i1++)
   {      
      (*B_marker)[i1] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over rows of A
    *-----------------------------------------------------------------------*/
   
   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {
      
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, C_{i1,i1}. 
       *--------------------------------------------------------------------*/
 
      (*B_marker)[i1+first_col_diag_B] = jj_count_diag;
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
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
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

		  if (i3 < first_col_diag_B || i3 > last_col_diag_C)
		  { 
                  	if ((*B_marker)[i3] < jj_row_begin_offd)
                  	{
                     		(*B_marker)[i3] = jj_count_offd;
                     		jj_count_offd++;
                  	}
		  } 
		  else
		  { 
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
 
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_diag.
                *-----------------------------------------------------------*/
 
               for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
               {
                  i3 = B_diag_j[jj3]+first_col_diag_B;
                  
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
                  i3 = col_map_offd_B[B_offd_j[jj3]];
                  
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
            
      /*--------------------------------------------------------------------
       * Set C_diag_i and C_offd_i for this row.
       *--------------------------------------------------------------------*/
 
      (*C_diag_i)[i1] = jj_row_begin_diag;
      (*C_offd_i)[i1] = jj_row_begin_offd;
      
   }
  
   (*C_diag_i)[num_cols_diag_B] = jj_count_diag;
   (*C_offd_i)[num_cols_diag_B] = jj_count_offd;
 
   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   *C_diag_size = jj_count_diag;
   *C_offd_size = jj_count_offd;

   /* End of First Pass */
}


/*--------------------------------------------------------------------------
 * hypre_ParMatmul : multiplies two ParCSRMatrices A and B and returns
 * the product in ParCSRMatrix C
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *hypre_ParMatmul( hypre_ParCSRMatrix  *A,
				     hypre_ParCSRMatrix  *B)
{
   MPI_Comm 	   comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   
   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   
   double          *A_offd_data = hypre_CSRMatrixData(A_offd);
   int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

   int *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   int	num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   int	num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   
   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
   
   double          *B_diag_data = hypre_CSRMatrixData(B_diag);
   int             *B_diag_i = hypre_CSRMatrixI(B_diag);
   int             *B_diag_j = hypre_CSRMatrixJ(B_diag);

   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   int		   *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);
   
   double          *B_offd_data = hypre_CSRMatrixData(B_offd);
   int             *B_offd_i = hypre_CSRMatrixI(B_offd);
   int             *B_offd_j = hypre_CSRMatrixJ(B_offd);

   int	first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   int *col_starts_B = hypre_ParCSRMatrixColStarts(B);
   int	num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
   int	num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   int		      *col_map_offd_C;

   hypre_CSRMatrix *C_diag;

   double          *C_diag_data;
   int             *C_diag_i;
   int             *C_diag_j;

   hypre_CSRMatrix *C_offd;

   double          *C_offd_data;
   int             *C_offd_i;
   int             *C_offd_j;

   int              C_diag_size;
   int              C_offd_size;
   int		    last_col_diag_C;
   int		    num_cols_offd_C;
   
   hypre_CSRMatrix *B_ext;
   
   double          *B_ext_data;
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

   double           a_entry;
   double           a_b_product;
   
   double           zero = 0.0;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

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
   	if (!hypre_ParCSRMatrixCommPkg(A))
   	{
        	hypre_MatvecCommPkgCreate(A);
   	}

   	B_ext = hypre_ParCSRMatrixExtractBExt(B,A,1);
   	B_ext_data = hypre_CSRMatrixData(B_ext);
   	B_ext_i    = hypre_CSRMatrixI(B_ext);
   	B_ext_j    = hypre_CSRMatrixJ(B_ext);
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
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   C_diag_data = hypre_CTAlloc(double, C_diag_size);
   C_diag_j    = hypre_CTAlloc(int, C_diag_size);
   if (C_offd_size)
   { 
   	C_offd_data = hypre_CTAlloc(double, C_offd_size);
   	C_offd_j    = hypre_CTAlloc(int, C_offd_size);
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
      C_diag_data[jj_count_diag] = zero;
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
            a_entry = A_offd_data[jj2];
            
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_ext.
                *-----------------------------------------------------------*/

               for (jj3 = B_ext_i[i2]; jj3 < B_ext_i[i2+1]; jj3++)
               {
                  i3 = B_ext_j[jj3];
                  a_b_product = a_entry * B_ext_data[jj3];
                  
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
                     	C_offd_data[jj_count_offd] = a_b_product;
                     	C_offd_j[jj_count_offd] = i3;
                     	jj_count_offd++;
		     }
		     else
                     	C_offd_data[B_marker[i3]] += a_b_product;
                  }
                  else
                  {
                     if (B_marker[i3] < jj_row_begin_diag)
                     {
                     	B_marker[i3] = jj_count_diag;
                     	C_diag_data[jj_count_diag] = a_b_product;
                     	C_diag_j[jj_count_diag] = i3-first_col_diag_B;
                     	jj_count_diag++;
		     }
		     else
                     	C_diag_data[B_marker[i3]] += a_b_product;
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
            a_entry = A_diag_data[jj2];
            
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_diag.
                *-----------------------------------------------------------*/

               for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
               {
                  i3 = B_diag_j[jj3]+first_col_diag_B;
                  a_b_product = a_entry * B_diag_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_data[jj_count_diag] = a_b_product;
                     C_diag_j[jj_count_diag] = B_diag_j[jj3];
                     jj_count_diag++;
                  }
                  else
                  {
                     C_diag_data[B_marker[i3]] += a_b_product;
                  }
               }
               if (num_cols_offd_B)
	       {
		for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
                {
                  i3 = col_map_offd_B[B_offd_j[jj3]];
                  a_b_product = a_entry * B_offd_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     C_offd_data[jj_count_offd] = a_b_product;
                     C_offd_j[jj_count_offd] = i3;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_offd_data[B_marker[i3]] += a_b_product;
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

   C = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
	col_starts_B, num_cols_offd_C, C_diag_size, C_offd_size);

/* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRMatrixSetColStartsOwner(C,0);

   C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrixData(C_diag) = C_diag_data; 
   hypre_CSRMatrixI(C_diag) = C_diag_i; 
   hypre_CSRMatrixJ(C_diag) = C_diag_j; 

   if (num_cols_offd_C)
   {
      C_offd = hypre_ParCSRMatrixOffd(C);
      hypre_CSRMatrixData(C_offd) = C_offd_data; 
      hypre_CSRMatrixI(C_offd) = C_offd_i; 
      hypre_CSRMatrixJ(C_offd) = C_offd_j; 
      hypre_ParCSRMatrixOffd(C) = C_offd;
      hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   }
   else
	hypre_TFree(C_offd_i);

   /*-----------------------------------------------------------------------
    *  Free B_ext and marker array.
    *-----------------------------------------------------------------------*/

   if (num_cols_offd_A)
   {
      hypre_CSRMatrixDestroy(B_ext);
      B_ext = NULL;
   }
   hypre_TFree(B_marker);   

   return C;
   
}            


/* The following function was formerly part of hypre_ParCSRMatrixExtractBExt
   but the code was removed so it can be used for a corresponding function
   for Boolean matrices
*/

void hypre_ParCSRMatrixExtractBExt_Arrays
( int ** pB_ext_i, int ** pB_ext_j, double ** pB_ext_data,
  int * num_nonzeros,
  int data, MPI_Comm comm, hypre_ParCSRCommPkg * comm_pkg,
  int num_cols_B, int num_recvs, int num_sends,
  int first_col_diag,
  int * recv_vec_starts, int * send_map_starts, int * send_map_elmts,
  int * diag_i, int * diag_j, int * offd_i, int * offd_j, int * col_map_offd,
  double * diag_data, double * offd_data
  )
{
/* begin generic part.
 inputs:
    int data
    MPI_Comm comm
    hypre_ParCSRCommPkg *comm_pkg
    int num_cols_B, num_rows_B_ext, num_recvs, num_sends
    int first_col_diag,
    int* recv_vec_starts, send_map_starts, send_map_elmts
    int* diag_i, diag_j, offd_i, offd_j, diag_i, col_map_offd
 inputs if data!=0:
    double* diag_data, offd_data
 outputs:
    int num_nonzeros
    int* B_ext_i, B_ext_j
 outputs if data!=0:
    double* B_ext_data;
*/

   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *tmp_comm_pkg;

   int *B_int_i;
   int *B_int_j;
   int *B_ext_i;
   int * B_ext_j;
   double * B_ext_data;
   double * B_int_data;

   int num_procs, my_id;

/*   MPI_Datatype *recv_matrix_types;
   MPI_Datatype *send_matrix_types; */

   int *jdata_recv_vec_starts;
   int *jdata_send_map_starts;
 
   int i, j, k, counter;
   int start_index;
   int j_cnt, jrow;
   int num_rows_B_ext;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   num_rows_B_ext = recv_vec_starts[num_recvs];
   B_int_i = hypre_CTAlloc(int, send_map_starts[num_sends]+1);
   *pB_ext_i = hypre_CTAlloc(int, num_rows_B_ext+1);
   B_ext_i = *pB_ext_i;
/*   send_matrix_types = hypre_CTAlloc(MPI_Datatype, num_sends);
   recv_matrix_types = hypre_CTAlloc(MPI_Datatype, num_recvs);
*/  
/*--------------------------------------------------------------------------
 * generate B_int_i through adding number of row-elements of offd and diag
 * for corresponding rows. B_int_i[j+1] contains the number of elements of
 * a row j (which is determined through send_map_elmts) 
 *--------------------------------------------------------------------------*/
   B_int_i[0] = 0;
   j_cnt = 0;
   *num_nonzeros = 0;
   for (i=0; i < num_sends; i++)
   {
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
	{
	    jrow = send_map_elmts[j];
	    B_int_i[++j_cnt] = offd_i[jrow+1] - offd_i[jrow]
			  + diag_i[jrow+1] - diag_i[jrow];
	    *num_nonzeros += B_int_i[j_cnt];
	}
   }

/*--------------------------------------------------------------------------
 * initialize communication 
 *--------------------------------------------------------------------------*/
   comm_handle = hypre_ParCSRCommHandleCreate(11,comm_pkg,
		&B_int_i[1],&(*pB_ext_i)[1]);

   B_int_j = hypre_CTAlloc(int, *num_nonzeros);
   if (data) B_int_data = hypre_CTAlloc(double, *num_nonzeros);

   jdata_send_map_starts = hypre_CTAlloc(int, num_sends+1);
   jdata_recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);
   start_index = B_int_i[0];
   jdata_send_map_starts[0] = start_index;
   counter = 0;
   for (i=0; i < num_sends; i++)
   {
	*num_nonzeros = counter;
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
	{
	    jrow = send_map_elmts[j];
	    for (k=diag_i[jrow]; k < diag_i[jrow+1]; k++) 
	    {
		B_int_j[counter] = diag_j[k]+first_col_diag;
		if (data) B_int_data[counter] = diag_data[k];
		counter++;
  	    }
	    for (k=offd_i[jrow]; k < offd_i[jrow+1]; k++) 
	    {
		B_int_j[counter] = col_map_offd[offd_j[k]];
		if (data) B_int_data[counter] = offd_data[k];
		counter++;
  	    }
	   
	}
	*num_nonzeros = counter - *num_nonzeros;
	/* if (data) 
	{
		hypre_BuildCSRJDataType(*num_nonzeros, 
			  &B_int_data[start_index], 
			  &B_int_j[start_index], 
			  &send_matrix_types[i]);	
	}
	else
	{
		MPI_Aint displ[1];
		MPI_Datatype type[1];
		type[0] = MPI_INT;
		MPI_Address(&B_int_j[start_index], &displ[0]);
		MPI_Type_struct(1,num_nonzeros,displ,type,
			&send_matrix_types[i]);
		MPI_Type_commit(&send_matrix_types[i]);
	} */
	start_index += *num_nonzeros;
        jdata_send_map_starts[i+1] = start_index;
   }

   tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
   hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
/*   hypre_ParCSRCommPkgSendMPITypes(tmp_comm_pkg) = send_matrix_types;	 */
   hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_send_map_starts; 

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

/*--------------------------------------------------------------------------
 * after communication exchange B_ext_i[j+1] contains the number of elements
 * of a row j ! 
 * evaluate B_ext_i and compute *num_nonzeros for B_ext 
 *--------------------------------------------------------------------------*/

   for (i=0; i < num_recvs; i++)
	for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
		B_ext_i[j+1] += B_ext_i[j];

   *num_nonzeros = B_ext_i[num_rows_B_ext];

   *pB_ext_j = hypre_CTAlloc(int, *num_nonzeros);
   B_ext_j = *pB_ext_j;
   if (data) {
      *pB_ext_data = hypre_CTAlloc(double, *num_nonzeros);
      B_ext_data = *pB_ext_data;
   };

   for (i=0; i < num_recvs; i++)
   {
	start_index = B_ext_i[recv_vec_starts[i]];
	*num_nonzeros = B_ext_i[recv_vec_starts[i+1]]-start_index;
	jdata_recv_vec_starts[i+1] = B_ext_i[recv_vec_starts[i+1]];
/* 	if (data)
	{
		hypre_BuildCSRJDataType(*num_nonzeros, 
			  &B_ext_data[start_index], 
			  &B_ext_j[start_index], 
			  &recv_matrix_types[i]);	
	}
	else
	{
		MPI_Aint displ[1];
		MPI_Datatype type[1];
		type[0] = MPI_INT;
		MPI_Address(&B_ext_j[start_index], &displ[0]);
		MPI_Type_struct(1,num_nonzeros,displ,type,
			&recv_matrix_types[i]);
		MPI_Type_commit(&recv_matrix_types[i]);
	} */
   }

   /* hypre_ParCSRCommPkgRecvMPITypes(tmp_comm_pkg) = recv_matrix_types;

   comm_handle = hypre_ParCSRCommHandleCreate(0,tmp_comm_pkg,NULL,NULL); */

   hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_recv_vec_starts;

   comm_handle = hypre_ParCSRCommHandleCreate(11,tmp_comm_pkg,B_int_j,B_ext_j);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   if (data)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(1,tmp_comm_pkg,B_int_data,
						B_ext_data);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

/*
   for (i=0; i < num_sends; i++)
	MPI_Type_free(&send_matrix_types[i]);

   for (i=0; i < num_recvs; i++)
	MPI_Type_free(&recv_matrix_types[i]);

   hypre_TFree(send_matrix_types);
   hypre_TFree(recv_matrix_types); */

   hypre_TFree(jdata_send_map_starts);
   hypre_TFree(jdata_recv_vec_starts);
   hypre_TFree(tmp_comm_pkg);
   hypre_TFree(B_int_i);
   hypre_TFree(B_int_j);
   if (data) hypre_TFree(B_int_data);

/* end generic part */
}


/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBExt : extracts rows from B which are located on other
 * processors and needed for multiplication with A locally. The rows
 * are returned as CSRMatrix.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix * 
hypre_ParCSRMatrixExtractBExt( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A, int data)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(B);
   int first_col_diag = hypre_ParCSRMatrixFirstColDiag(B);
   int *col_map_offd = hypre_ParCSRMatrixColMapOffd(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(B);

   int *diag_i = hypre_CSRMatrixI(diag);
   int *diag_j = hypre_CSRMatrixJ(diag);
   double *diag_data = hypre_CSRMatrixData(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(B);

   int *offd_i = hypre_CSRMatrixI(offd);
   int *offd_j = hypre_CSRMatrixJ(offd);
   double *offd_data = hypre_CSRMatrixData(offd);

   int num_cols_B, num_nonzeros;
   int num_rows_B_ext;

   hypre_CSRMatrix *B_ext;

   int *B_ext_i;
   int *B_ext_j;
   double *B_ext_data;
 
   num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   hypre_ParCSRMatrixExtractBExt_Arrays
      ( &B_ext_i, &B_ext_j, &B_ext_data,
        &num_nonzeros,
        data, comm, comm_pkg,
        num_cols_B, num_recvs, num_sends,
        first_col_diag,
        recv_vec_starts, send_map_starts, send_map_elmts,
        diag_i, diag_j, offd_i, offd_j, col_map_offd,
        diag_data, offd_data
         );

   B_ext = hypre_CSRMatrixCreate(num_rows_B_ext,num_cols_B,num_nonzeros);
   hypre_CSRMatrixI(B_ext) = B_ext_i;
   hypre_CSRMatrixJ(B_ext) = B_ext_j;
   if (data) hypre_CSRMatrixData(B_ext) = B_ext_data;

   return B_ext;
}

