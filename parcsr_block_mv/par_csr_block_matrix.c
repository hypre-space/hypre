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
 * Member functions for hypre_ParCSRBlockMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixCreate(  MPI_Comm comm,
				int block_size,
				int global_num_rows,
				int global_num_cols,
				int *row_starts,
				int *col_starts,
				int num_cols_offd,
				int num_nonzeros_diag,
				int num_nonzeros_offd)
{
   hypre_ParCSRBlockMatrix  *matrix;
   int  num_procs, my_id;
   int local_num_rows, local_num_cols;
   int first_row_index, first_col_diag;

   matrix = hypre_CTAlloc(hypre_ParCSRBlockMatrix, 1);

   MPI_Comm_rank(comm,&my_id);
   MPI_Comm_size(comm,&num_procs);

   if (!row_starts)
   {
        hypre_GeneratePartitioning(global_num_rows,num_procs,&row_starts);
   }

   if (!col_starts)
   {
      if (global_num_rows == global_num_cols)
      {
        col_starts = row_starts;
      }
      else
      {
        hypre_GeneratePartitioning(global_num_cols,num_procs,&col_starts);
      }
   }

   first_row_index = row_starts[my_id];
   local_num_rows = row_starts[my_id+1]-first_row_index;
   first_col_diag = col_starts[my_id];
   local_num_cols = col_starts[my_id+1]-first_col_diag;
   hypre_ParCSRBlockMatrixComm(matrix) = comm;
   hypre_ParCSRBlockMatrixDiag(matrix) = hypre_CSRBlockMatrixCreate(block_size, local_num_rows, local_num_cols, num_nonzeros_diag);
   hypre_ParCSRBlockMatrixOffd(matrix) = hypre_CSRBlockMatrixCreate(block_size, local_num_rows, num_cols_offd, num_nonzeros_offd);
   hypre_ParCSRBlockMatrixBlockSize(matrix) = block_size;
   hypre_ParCSRBlockMatrixGlobalNumRows(matrix) = global_num_rows;
   hypre_ParCSRBlockMatrixGlobalNumCols(matrix) = global_num_cols;
   hypre_ParCSRBlockMatrixFirstRowIndex(matrix) = first_row_index;
   hypre_ParCSRBlockMatrixFirstColDiag(matrix) = first_col_diag;
   hypre_ParCSRBlockMatrixColMapOffd(matrix) = NULL;
   hypre_ParCSRBlockMatrixRowStarts(matrix) = row_starts;
   hypre_ParCSRBlockMatrixColStarts(matrix) = col_starts;
   hypre_ParCSRBlockMatrixCommPkg(matrix) = NULL;
   hypre_ParCSRBlockMatrixCommPkgT(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRBlockMatrixOwnsData(matrix) = 1;
   hypre_ParCSRBlockMatrixOwnsRowStarts(matrix) = 1;
   hypre_ParCSRBlockMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
        hypre_ParCSRBlockMatrixOwnsColStarts(matrix) = 0;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRBlockMatrixDestroy( hypre_ParCSRBlockMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      if ( hypre_ParCSRBlockMatrixOwnsData(matrix) )
      {
	hypre_CSRBlockMatrixDestroy(hypre_ParCSRBlockMatrixDiag(matrix));
	hypre_CSRBlockMatrixDestroy(hypre_ParCSRBlockMatrixOffd(matrix));
	if (hypre_ParCSRBlockMatrixColMapOffd(matrix))
	  hypre_TFree(hypre_ParCSRBlockMatrixColMapOffd(matrix));
	if (hypre_ParCSRBlockMatrixCommPkg(matrix))
	  hypre_MatvecCommPkgDestroy(hypre_ParCSRBlockMatrixCommPkg(matrix));
	if (hypre_ParCSRBlockMatrixCommPkgT(matrix))
	  hypre_MatvecCommPkgDestroy(hypre_ParCSRBlockMatrixCommPkgT(matrix));
      }
      if ( hypre_ParCSRBlockMatrixOwnsRowStarts(matrix) )
              hypre_TFree(hypre_ParCSRBlockMatrixRowStarts(matrix));
      if ( hypre_ParCSRBlockMatrixOwnsColStarts(matrix) )
              hypre_TFree(hypre_ParCSRBlockMatrixColStarts(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRBlockMatrixInitialize( hypre_ParCSRBlockMatrix *matrix )
{
   int  ierr=0;

   hypre_CSRBlockMatrixInitialize(hypre_ParCSRBlockMatrixDiag(matrix));
   hypre_CSRBlockMatrixInitialize(hypre_ParCSRBlockMatrixOffd(matrix));
   hypre_ParCSRBlockMatrixColMapOffd(matrix) = hypre_CTAlloc(int, hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixOffd(matrix)));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixSetNumNonzeros
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRBlockMatrixSetNumNonzeros( hypre_ParCSRBlockMatrix *matrix)
{
   MPI_Comm comm = hypre_ParCSRBlockMatrixComm(matrix);
   hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(matrix);
   int *diag_i = hypre_CSRBlockMatrixI(diag);
   hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(matrix);
   int *offd_i = hypre_CSRBlockMatrixI(offd);
   int local_num_rows = hypre_CSRBlockMatrixNumRows(diag);
   int total_num_nonzeros;
   int local_num_nonzeros;
   int ierr = 0;

   local_num_nonzeros = diag_i[local_num_rows] + offd_i[local_num_rows];
   MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, MPI_INT,
        MPI_SUM, comm);
   hypre_ParCSRBlockMatrixNumNonzeros(matrix) = total_num_nonzeros;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRBlockMatrixSetDataOwner( hypre_ParCSRBlockMatrix *matrix,
				     int              owns_data )
{
   int    ierr=0;

   hypre_ParCSRBlockMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixSetRowStartsOwner
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRBlockMatrixSetRowStartsOwner( hypre_ParCSRBlockMatrix *matrix,
					  int owns_row_starts )
{
   int    ierr=0;

   hypre_ParCSRBlockMatrixOwnsRowStarts(matrix) = owns_row_starts;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixSetColStartsOwner
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRBlockMatrixSetColStartsOwner( hypre_ParCSRBlockMatrix *matrix,
					  int owns_col_starts )
{
   int    ierr=0;

   hypre_ParCSRBlockMatrixOwnsColStarts(matrix) = owns_col_starts;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixCompress
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_ParCSRBlockMatrixCompress( hypre_ParCSRBlockMatrix *matrix )
{
  MPI_Comm comm = hypre_ParCSRBlockMatrixComm(matrix);
  hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(matrix);
  hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(matrix);
  int global_num_rows = hypre_ParCSRBlockMatrixGlobalNumRows(matrix);
  int global_num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(matrix);
  int *row_starts = hypre_ParCSRBlockMatrixRowStarts(matrix);
  int *col_starts = hypre_ParCSRBlockMatrixColStarts(matrix);
  int num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);
  int num_nonzeros_diag = hypre_CSRBlockMatrixNumNonzeros(diag);
  int num_nonzeros_offd = hypre_CSRBlockMatrixNumNonzeros(offd);

  hypre_ParCSRMatrix *matrix_C;

  int i;

  matrix_C = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols, row_starts, col_starts, num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
  hypre_ParCSRMatrixInitialize(matrix_C);

  hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matrix_C));
  hypre_ParCSRMatrixDiag(matrix_C) = hypre_CSRBlockMatrixCompress(diag);
  hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(matrix_C));
  hypre_ParCSRMatrixOffd(matrix_C) = hypre_CSRBlockMatrixCompress(offd);

  for(i = 0; i < num_cols_offd; i++) {
    hypre_ParCSRMatrixColMapOffd(matrix_C)[i] = hypre_ParCSRBlockMatrixColMapOffd(matrix)[i];
  }

  return matrix_C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixConvertToParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_ParCSRBlockMatrixConvertToParCSRMatrix( hypre_ParCSRBlockMatrix *matrix )
{
  MPI_Comm comm = hypre_ParCSRBlockMatrixComm(matrix);
  hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(matrix);
  hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(matrix);
  int block_size = hypre_ParCSRBlockMatrixBlockSize(matrix);
  int global_num_rows = hypre_ParCSRBlockMatrixGlobalNumRows(matrix);
  int global_num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(matrix);
  int *row_starts = hypre_ParCSRBlockMatrixRowStarts(matrix);
  int *col_starts = hypre_ParCSRBlockMatrixColStarts(matrix);
  int num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);
  int num_nonzeros_diag = hypre_CSRBlockMatrixNumNonzeros(diag);
  int num_nonzeros_offd = hypre_CSRBlockMatrixNumNonzeros(offd);

  hypre_ParCSRMatrix *matrix_C;
  int *matrix_C_row_starts;
  int *matrix_C_col_starts;

  int num_procs, i, j;

  MPI_Comm_size(comm,&num_procs);
  matrix_C_row_starts = hypre_CTAlloc(int, num_procs + 1);
  matrix_C_col_starts = hypre_CTAlloc(int, num_procs + 1);
  for(i = 0; i < num_procs + 1; i++) {
    matrix_C_row_starts[i] = row_starts[i]*block_size;
    matrix_C_col_starts[i] = col_starts[i]*block_size;
  }

  matrix_C = hypre_ParCSRMatrixCreate(comm, global_num_rows*block_size, global_num_cols*block_size, matrix_C_row_starts, matrix_C_col_starts, num_cols_offd*block_size, num_nonzeros_diag*block_size*block_size, num_nonzeros_offd*block_size*block_size);
  hypre_ParCSRMatrixInitialize(matrix_C);

  hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matrix_C));
  hypre_ParCSRMatrixDiag(matrix_C) = hypre_CSRBlockMatrixConvertToCSRMatrix(diag);
  hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(matrix_C));
  hypre_ParCSRMatrixOffd(matrix_C) = hypre_CSRBlockMatrixConvertToCSRMatrix(offd);

 
  for(i = 0; i < num_cols_offd; i++) {
    for(j = 0; j < block_size; j++) {
      hypre_ParCSRMatrixColMapOffd(matrix_C)[i*block_size + j] = hypre_ParCSRBlockMatrixColMapOffd(matrix)[i]*block_size + j;
    }
  }

  return matrix_C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixConvertFromParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixConvertFromParCSRMatrix( hypre_ParCSRMatrix *matrix, int matrix_C_block_size )
{
  MPI_Comm comm = hypre_ParCSRMatrixComm(matrix);
  hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
  hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);
  int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(matrix);
  int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(matrix);
  int *row_starts = hypre_ParCSRMatrixRowStarts(matrix);
  int *col_starts = hypre_ParCSRMatrixColStarts(matrix);
  int num_cols_offd = hypre_CSRMatrixNumCols(offd);

  hypre_ParCSRBlockMatrix *matrix_C;
  int *matrix_C_row_starts;
  int *matrix_C_col_starts;
  hypre_CSRBlockMatrix *matrix_C_diag;
  hypre_CSRBlockMatrix *matrix_C_offd;

  int num_procs, i;

  MPI_Comm_size(comm,&num_procs);
  matrix_C_row_starts = hypre_CTAlloc(int, num_procs + 1);
  matrix_C_col_starts = hypre_CTAlloc(int, num_procs + 1);
  for(i = 0; i < num_procs + 1; i++) {
    matrix_C_row_starts[i] = row_starts[i]/matrix_C_block_size;
    matrix_C_col_starts[i] = col_starts[i]/matrix_C_block_size;
  }

  matrix_C_diag = hypre_CSRBlockMatrixConvertFromCSRMatrix(diag, matrix_C_block_size);
  matrix_C_offd = hypre_CSRBlockMatrixConvertFromCSRMatrix(offd, matrix_C_block_size);

  matrix_C = hypre_ParCSRBlockMatrixCreate(comm, matrix_C_block_size, global_num_rows/matrix_C_block_size, global_num_cols/matrix_C_block_size, matrix_C_row_starts, matrix_C_col_starts, num_cols_offd/matrix_C_block_size, hypre_CSRBlockMatrixNumNonzeros(matrix_C_diag), hypre_CSRBlockMatrixNumNonzeros(matrix_C_offd));
  hypre_ParCSRBlockMatrixInitialize(matrix_C);

  hypre_CSRBlockMatrixDestroy(hypre_ParCSRMatrixDiag(matrix_C));
  hypre_ParCSRBlockMatrixDiag(matrix_C) = matrix_C_diag;
  hypre_CSRBlockMatrixDestroy(hypre_ParCSRMatrixOffd(matrix_C));
  hypre_ParCSRBlockMatrixOffd(matrix_C) = matrix_C_offd;

  for(i = 0; i < num_cols_offd/matrix_C_block_size; i++) {
    hypre_ParCSRBlockMatrixColMapOffd(matrix_C)[i] = hypre_ParCSRMatrixColMapOffd(matrix)[i*matrix_C_block_size]/matrix_C_block_size;
  }

  return matrix_C;
}

/* ----------------------------------------------------------------------
 * hypre_BlockMatvecCommPkgCreate
 * ---------------------------------------------------------------------*/

int
hypre_BlockMatvecCommPkgCreate ( hypre_ParCSRBlockMatrix *A )
{
   hypre_ParCSRCommPkg	*comm_pkg;
   
   MPI_Comm             comm = hypre_ParCSRBlockMatrixComm(A);

   int			num_sends;
   int			*send_procs;
   int			*send_map_starts;
   int			*send_map_elmts;
   int			num_recvs;
   int			*recv_procs;
   int			*recv_vec_starts;
   
   int  *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);
   int  first_col_diag = hypre_ParCSRBlockMatrixFirstColDiag(A);
   int  *col_starts = hypre_ParCSRBlockMatrixColStarts(A);

   int	ierr = 0;
   int	num_cols_diag = hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixDiag(A));
   int	num_cols_offd = hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixOffd(A));

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

   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts;

   hypre_ParCSRMatrixBlockCommPkg(A) = comm_pkg;

   return ierr;
}

/* ----------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixExtractBExt
 * ---------------------------------------------------------------------*/

hypre_CSRBlockMatrix * 
hypre_ParCSRBlockMatrixExtractBExt( hypre_ParCSRBlockMatrix *B, hypre_ParCSRBlockMatrix *A, int data)
{
   MPI_Comm comm = hypre_ParCSRBlockMatrixComm(B);
   int first_col_diag = hypre_ParCSRBlockMatrixFirstColDiag(B);
   int *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(B);
   int block_size = hypre_ParCSRBlockMatrixBlockSize(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *tmp_comm_pkg;

   hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(B);

   int *diag_i = hypre_CSRBlockMatrixI(diag);
   int *diag_j = hypre_CSRBlockMatrixJ(diag);
   double *diag_data = hypre_CSRBlockMatrixData(diag);

   hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(B);

   int *offd_i = hypre_CSRBlockMatrixI(offd);
   int *offd_j = hypre_CSRBlockMatrixJ(offd);
   double *offd_data = hypre_CSRBlockMatrixData(offd);

   int *B_int_i;
   int *B_int_j;
   double *B_int_data;

   int num_cols_B, num_nonzeros;
   int num_rows_B_ext;
   int num_procs, my_id;

   hypre_CSRBlockMatrix *B_ext;

   int *B_ext_i;
   int *B_ext_j;
   double *B_ext_data;
 
   int *jdata_recv_vec_starts;
   int *jdata_send_map_starts;
 
   int i, j, k, l, counter;
   int start_index;
   int j_cnt, jrow;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];
   B_int_i = hypre_CTAlloc(int, send_map_starts[num_sends]+1);
   B_ext_i = hypre_CTAlloc(int, num_rows_B_ext+1);
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
   comm_handle = hypre_ParCSRCommHandleCreate(11,comm_pkg,
		&B_int_i[1],&B_ext_i[1]);

   B_int_j = hypre_CTAlloc(int, num_nonzeros);
   if (data) B_int_data = hypre_CTAlloc(double, num_nonzeros*block_size*block_size);

   jdata_send_map_starts = hypre_CTAlloc(int, num_sends+1);
   jdata_recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);
   start_index = B_int_i[0];
   jdata_send_map_starts[0] = start_index;
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
		if (data) {
		  for(l = 0; l < block_size*block_size; l++) {
		    B_int_data[counter*block_size*block_size + l] = diag_data[k*block_size*block_size + l];
		  }
		}
		counter++;
  	    }
	    for (k=offd_i[jrow]; k < offd_i[jrow+1]; k++) 
	    {
		B_int_j[counter] = col_map_offd[offd_j[k]];
		if (data) {
		  for(l = 0; l < block_size*block_size; l++) {
		    B_int_data[counter*block_size*block_size + l] = offd_data[k*block_size*block_size + l];
		  }
		}
		counter++;
  	    }
	   
	}
	num_nonzeros = counter - num_nonzeros;
	start_index += num_nonzeros;
        jdata_send_map_starts[i+1] = start_index;
   }

   tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
   hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_send_map_starts; 

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

/*--------------------------------------------------------------------------
 * after communication exchange B_ext_i[j+1] contains the number of elements
 * of a row j ! 
 * evaluate B_ext_i and compute num_nonzeros for B_ext 
 *--------------------------------------------------------------------------*/

   for (i=0; i < num_recvs; i++)
	for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
		B_ext_i[j+1] += B_ext_i[j];

   num_nonzeros = B_ext_i[num_rows_B_ext];

   B_ext = hypre_CSRBlockMatrixCreate(block_size, num_rows_B_ext, num_cols_B, num_nonzeros);
   B_ext_j = hypre_CTAlloc(int, num_nonzeros);
   if (data) B_ext_data = hypre_CTAlloc(double, num_nonzeros*block_size*block_size);

   for (i=0; i < num_recvs; i++)
   {
	start_index = B_ext_i[recv_vec_starts[i]];
	num_nonzeros = B_ext_i[recv_vec_starts[i+1]]-start_index;
	jdata_recv_vec_starts[i+1] = B_ext_i[recv_vec_starts[i+1]];
   }

   hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_recv_vec_starts;

   comm_handle = hypre_ParCSRCommHandleCreate(11,tmp_comm_pkg,B_int_j,B_ext_j);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   if (data)
   {
     for(i = 0; i < num_recvs; i++) jdata_recv_vec_starts[i] = jdata_recv_vec_starts[i]*block_size*block_size;
     hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_recv_vec_starts;
     for(i = 0; i < num_sends; i++) jdata_send_map_starts[i] = jdata_send_map_starts[i]*block_size*block_size;
     hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_send_map_starts; 

      comm_handle = hypre_ParCSRCommHandleCreate(1,tmp_comm_pkg,B_int_data,
						B_ext_data);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   hypre_CSRBlockMatrixI(B_ext) = B_ext_i;
   hypre_CSRBlockMatrixJ(B_ext) = B_ext_j;
   if (data) hypre_CSRBlockMatrixData(B_ext) = B_ext_data;

   hypre_TFree(B_int_i);
   hypre_TFree(B_int_j);
   if (data) hypre_TFree(B_int_data);
   hypre_TFree(jdata_send_map_starts);
   hypre_TFree(jdata_recv_vec_starts);
   hypre_TFree(tmp_comm_pkg);

   return B_ext;
}
