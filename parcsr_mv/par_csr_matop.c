/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#include "headers.h"

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
   
   int              jj_counter, jj_count_diag, jj_count_offd;
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
        	hypre_GenerateMatvecCommunicationInfo(A);
   	}

   	B_ext = hypre_ExtractBExt(B,A,1);
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

   jj_counter = start_indexing;
   for (i1 = 0; i1 < n_cols_B; i1++)
   {      
      B_marker[i1] = -1;
   }

   C_diag_i = hypre_CTAlloc(int, num_cols_diag_B+1);
   C_offd_i = hypre_CTAlloc(int, num_cols_diag_B+1);

   last_col_diag_C = first_col_diag_B + num_cols_diag_B - 1;

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
    *  Loop over rows of A
    *-----------------------------------------------------------------------*/
   
   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {
      
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, C_{i1,i1}. 
       *--------------------------------------------------------------------*/
 
      B_marker[i1+first_col_diag_B] = jj_count_diag;
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
                  	if (B_marker[i3] < jj_row_begin_offd)
                  	{
                     		B_marker[i3] = jj_count_offd;
                     		jj_count_offd++;
                  	}
		  } 
		  else
		  { 
                  	if (B_marker[i3] < jj_row_begin_diag)
                  	{
                     		B_marker[i3] = jj_count_diag;
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
 
                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
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
 
                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
                 }
            }
      }
            
      /*--------------------------------------------------------------------
       * Set C_diag_i and C_offd_i for this row.
       *--------------------------------------------------------------------*/
 
      C_diag_i[i1] = jj_row_begin_diag;
      C_offd_i[i1] = jj_row_begin_offd;
      
   }
  
   C_diag_i[num_cols_diag_B] = jj_count_diag;
   C_offd_i[num_cols_diag_B] = jj_count_offd;
 
   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   C_diag_size = jj_count_diag;
   C_diag_data = hypre_CTAlloc(double, C_diag_size);
   C_diag_j    = hypre_CTAlloc(int, C_diag_size);
 
   C_offd_size = jj_count_offd;
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

   C = hypre_CreateParCSRMatrix(comm, n_rows_A, n_cols_B, row_starts_A,
	col_starts_B, num_cols_offd_C, C_diag_size, C_offd_size);

/* Note that C does not own the partitionings */
   hypre_SetParCSRMatrixRowStartsOwner(C,0);
   hypre_SetParCSRMatrixColStartsOwner(C,0);

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

   if (num_cols_offd_A) hypre_DestroyCSRMatrix(B_ext);
   hypre_TFree(B_marker);   

   return C;
   
}            

/*--------------------------------------------------------------------------
 * hypre_ExtractBExt : extracts rows from B which are located on other
 * processors and needed for multiplication with A locally. The rows
 * are returned as CSRMatrix.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix * 
hypre_ExtractBExt( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A, int data)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(B);
   int first_col_diag = hypre_ParCSRMatrixFirstColDiag(B);
   int *col_map_offd = hypre_ParCSRMatrixColMapOffd(B);

   hypre_CommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   int num_recvs = hypre_CommPkgNumRecvs(comm_pkg);
   int *recv_vec_starts = hypre_CommPkgRecvVecStarts(comm_pkg);
   int num_sends = hypre_CommPkgNumSends(comm_pkg);
   int *send_map_starts = hypre_CommPkgSendMapStarts(comm_pkg);
   int *send_map_elmts = hypre_CommPkgSendMapElmts(comm_pkg);
 
   MPI_Datatype *recv_matrix_types;
   MPI_Datatype *send_matrix_types;
   hypre_CommHandle *comm_handle;
   hypre_CommPkg *tmp_comm_pkg;

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(B);

   int *diag_i = hypre_CSRMatrixI(diag);
   int *diag_j = hypre_CSRMatrixJ(diag);
   double *diag_data = hypre_CSRMatrixData(diag);

   int num_cols_diag = hypre_CSRMatrixNumCols(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(B);

   int *offd_i = hypre_CSRMatrixI(offd);
   int *offd_j = hypre_CSRMatrixJ(offd);
   double *offd_data = hypre_CSRMatrixData(offd);

   int num_cols_offd = hypre_CSRMatrixNumCols(offd);

   int *B_int_i;
   int *B_int_j;
   double *B_int_data;

   int num_cols_B, num_nonzeros;
   int num_rows_B_ext;
   int num_procs, my_id;

   hypre_CSRMatrix *B_ext;

   int *B_ext_i;
   int *B_ext_j;
   double *B_ext_data;
  
   int i, j, k, counter;
   int start_index;
   int j_cnt, jrow;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];
   B_int_i = hypre_CTAlloc(int, send_map_starts[num_sends]+1);
   send_matrix_types = hypre_CTAlloc(MPI_Datatype, num_sends);
   B_ext_i = hypre_CTAlloc(int, num_rows_B_ext+1);
   recv_matrix_types = hypre_CTAlloc(MPI_Datatype, num_recvs);
  
/*--------------------------------------------------------------------------
 * generate B_int_i through adding number of row-elements of offd and diag
 * for corresponding rows. B_int_i[j+1] contains the number of elements of
 * a row j (which is determined through send_map_elmts) 
 *--------------------------------------------------------------------------*/
   B_int_i[0] = 0;
   j_cnt = 0;
   num_nonzeros = 0;
   for (i=0; i < num_sends; i++)
   {
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
	{
	    jrow = send_map_elmts[j];
	    B_int_i[++j_cnt] = offd_i[jrow+1] - offd_i[jrow]
			  + diag_i[jrow+1] - diag_i[jrow];
	    num_nonzeros += B_int_i[j_cnt];
	}
   }

/*--------------------------------------------------------------------------
 * initialize communication 
 *--------------------------------------------------------------------------*/
   comm_handle = hypre_InitializeCommunication(11,comm_pkg,
		&B_int_i[1],&B_ext_i[1]);

   B_int_j = hypre_CTAlloc(int, num_nonzeros);
   if (data) B_int_data = hypre_CTAlloc(double, num_nonzeros);

   start_index = B_int_i[0];
   counter = 0;
   for (i=0; i < num_sends; i++)
   {
	num_nonzeros = counter;
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
	num_nonzeros = counter - num_nonzeros;
	if (data) 
	{
		hypre_BuildCSRJDataType(num_nonzeros, 
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
		MPI_Type_struct(1,&num_nonzeros,displ,type,
			&send_matrix_types[i]);
		MPI_Type_commit(&send_matrix_types[i]);
	}
	start_index += num_nonzeros;
   }

   tmp_comm_pkg = hypre_CTAlloc(hypre_CommPkg,1);
   hypre_CommPkgComm(tmp_comm_pkg) = comm;
   hypre_CommPkgNumSends(tmp_comm_pkg) = num_sends;
   hypre_CommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
   hypre_CommPkgSendProcs(tmp_comm_pkg) = hypre_CommPkgSendProcs(comm_pkg);
   hypre_CommPkgRecvProcs(tmp_comm_pkg) = hypre_CommPkgRecvProcs(comm_pkg);
   hypre_CommPkgSendMPITypes(tmp_comm_pkg) = send_matrix_types;	

   hypre_FinalizeCommunication(comm_handle);

/*--------------------------------------------------------------------------
 * after communication exchange B_ext_i[j+1] contains the number of elements
 * of a row j ! 
 * evaluate B_ext_i and compute num_nonzeros for B_ext 
 *--------------------------------------------------------------------------*/

   for (i=0; i < num_recvs; i++)
	for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
		B_ext_i[j+1] += B_ext_i[j];

   num_nonzeros = B_ext_i[num_rows_B_ext];

   B_ext = hypre_CreateCSRMatrix(num_rows_B_ext,num_cols_B,num_nonzeros);
   B_ext_j = hypre_CTAlloc(int, num_nonzeros);
   if (data) B_ext_data = hypre_CTAlloc(double, num_nonzeros);

   for (i=0; i < num_recvs; i++)
   {
	start_index = B_ext_i[recv_vec_starts[i]];
	num_nonzeros = B_ext_i[recv_vec_starts[i+1]]-start_index;
	if (data)
	{
		hypre_BuildCSRJDataType(num_nonzeros, 
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
		MPI_Type_struct(1,&num_nonzeros,displ,type,
			&recv_matrix_types[i]);
		MPI_Type_commit(&recv_matrix_types[i]);
	}
   }

   hypre_CommPkgRecvMPITypes(tmp_comm_pkg) = recv_matrix_types;	

   comm_handle = hypre_InitializeCommunication(0,tmp_comm_pkg,NULL,NULL);

   hypre_CSRMatrixI(B_ext) = B_ext_i;
   hypre_CSRMatrixJ(B_ext) = B_ext_j;
   if (data) hypre_CSRMatrixData(B_ext) = B_ext_data;

   hypre_FinalizeCommunication(comm_handle); 

   hypre_TFree(B_int_i);
   hypre_TFree(B_int_j);
   if (data) hypre_TFree(B_int_data);

   for (i=0; i < num_sends; i++)
	MPI_Type_free(&send_matrix_types[i]);

   for (i=0; i < num_recvs; i++)
	MPI_Type_free(&recv_matrix_types[i]);

   hypre_TFree(send_matrix_types);
   hypre_TFree(recv_matrix_types);
   hypre_TFree(tmp_comm_pkg);

   return B_ext;
}

