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

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildBlockInterp
 *--------------------------------------------------------------------------*/

// RYAN -- Do I really want to take in num_functions and dof_func?

int
hypre_BoomerAMGBuildBlockInterp( hypre_ParCSRBlockMatrix   *A,
                         int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         int                  *num_cpts_global,
                         int                   num_functions,
                         int                  *dof_func,
                         int                   debug_flag,
                         double                trunc_factor,
                         hypre_ParCSRBlockMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRBlockMatrixComm(A);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRBlockMatrix *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   double          *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   int             *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   int             *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   int              block_size = hypre_ParCSRBlockMatrixBlockSize(A);

   hypre_CSRBlockMatrix *A_offd = hypre_ParCSRBlockMatrixOffd(A);   
   double          *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   int             *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   int             *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
   int              num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   int             *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRBlockMatrix *P;
   int		      *col_map_offd_P;

   int             *CF_marker_offd;
   int             *dof_func_offd = NULL;

   hypre_CSRBlockMatrix *A_ext;
   
   double          *A_ext_data;
   int             *A_ext_i;
   int             *A_ext_j;

   hypre_CSRBlockMatrix    *P_diag;
   hypre_CSRBlockMatrix    *P_offd;   

   double          *P_diag_data;
   int             *P_diag_i;
   int             *P_diag_j;
   double          *P_offd_data;
   int             *P_offd_i;
   int             *P_offd_j;

   int              P_diag_size, P_offd_size;
   
   int             *P_marker, *P_marker_offd;

   int              jj_counter,jj_counter_offd;
   int             *jj_count, *jj_count_offd;
   int              jj_begin_row,jj_begin_row_offd;
   int              jj_end_row,jj_end_row_offd;
   
   int              start_indexing = 0; /* start indexing for P_data at 0 */

   int              n_fine = hypre_CSRBlockMatrixNumRows(A_diag);

   int              strong_f_marker;

   int             *fine_to_coarse;
   int             *fine_to_coarse_offd;
   int             *coarse_counter;
   int              coarse_shift;
   int              total_global_cpts;
   int              num_cols_P_offd,my_first_cpt;

   int              i,i1,i2;
   int              j,jl,jj,jj1;
   int              start;
   int              sgn;
   int              c_num;
   /*int              bin = 0;*/
   
   double*           diagonal;
   double*           sum;
   double*           distribute;          
   
   double           zero = 0.0;
   double           one  = 1.0;
   
   int              my_id;
   int              num_procs;
   int              num_threads;
   int              num_sends;
   int              index;
   int              ns, ne, size, rest;
   int             *int_buf_data;

   double           max_coef;
   double           row_sum, scale;
   int              next_open,now_checking,num_lost,start_j;
   int              next_open_offd,now_checking_offd,num_lost_offd;

   int col_1 = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   int local_numrows = hypre_CSRBlockMatrixNumRows(A_diag);
   int col_n = col_1 + local_numrows;

   double           wall_time;  /* for debugging instrumentation  */

   int ib, jb, kb; /* loop variables for blocks */

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];

   /* initialise memory for block entries */
   diagonal = hypre_CTAlloc(double, block_size*block_size);
   sum = hypre_CTAlloc(double, block_size*block_size);
   distribute = hypre_CTAlloc(double, block_size*block_size);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   CF_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);
   if (num_functions > 1 && num_cols_A_offd)
	dof_func_offd = hypre_CTAlloc(int, num_cols_A_offd);

   if (!comm_pkg)
   {
	hypre_BlockMatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
	
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	CF_marker_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);   

   // RYAN -- THIS NEXT BLOCK OF CODE NEEDS TO CHANGE SINCE dof_func WON'T LOOK THE SAME.

   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
   }

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRBlockMatrixExtractBExt(A,A,1);
      A_ext_i    = hypre_CSRBlockMatrixI(A_ext);
      A_ext_j    = hypre_CSRBlockMatrixJ(A_ext);
      A_ext_data = hypre_CSRBlockMatrixData(A_ext);
   }
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(int, num_threads);
   jj_count = hypre_CTAlloc(int, num_threads);
   jj_count_offd = hypre_CTAlloc(int, num_threads);

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

/* RDF: this looks a little tricky, but doable */
#define HYPRE_SMP_PRIVATE i,j,i1,jj,ns,ne,size,rest
#include "../utilities/hypre_smp_forloop.h"
   for (j = 0; j < num_threads; j++)
   {
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (j < rest)
     {
        ns = j*size+j;
        ne = (j+1)*size+j+1;
     }
     else
     {
        ns = j*size+rest;
        ne = (j+1)*size+rest;
     }
     for (i = ns; i < ne; i++)
     {
      
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         jj_count[j]++;
         fine_to_coarse[i] = coarse_counter[j];
         coarse_counter[j]++;
      }
      
      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_count[j]++;
            }
         }

         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_count_offd[j]++;
               }
            }
         }
      }
    }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   for (i=0; i < num_threads-1; i++)
   {
      coarse_counter[i+1] += coarse_counter[i];
      jj_count[i+1] += jj_count[i];
      jj_count_offd[i+1] += jj_count_offd[i];
   }
   i = num_threads-1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size*block_size*block_size);

   P_diag_i[n_fine] = jj_counter; 


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size*block_size*block_size);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(int, num_cols_A_offd); 

#define HYPRE_SMP_PRIVATE i,j,ns,ne,size,rest,coarse_shift
#include "../utilities/hypre_smp_forloop.h"
   for (j = 0; j < num_threads; j++)
   {
     coarse_shift = 0;
     if (j > 0) coarse_shift = coarse_counter[j-1];
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (j < rest)
     {
        ns = j*size+j;
        ne = (j+1)*size+j+1;
     }
     else
     {
        ns = j*size+rest;
        ne = (j+1)*size+rest;
     }
     for (i = ns; i < ne; i++)
	fine_to_coarse[i] += my_first_cpt+coarse_shift;
   }
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
	
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	fine_to_coarse_offd);  

   hypre_ParCSRCommHandleDestroy(comm_handle);   

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
    
#define HYPRE_SMP_PRIVATE i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,diagonal,distribute,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd
#include "../utilities/hypre_smp_forloop.h"
   for (jl = 0; jl < num_threads; jl++)
   {
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (jl < rest)
     {
        ns = jl*size+jl;
        ne = (jl+1)*size+jl+1;
     }
     else
     {
        ns = jl*size+rest;
        ne = (jl+1)*size+rest;
     }
     jj_counter = 0;
     if (jl > 0) jj_counter = jj_count[jl-1];
     jj_counter_offd = 0;
     if (jl > 0) jj_counter_offd = jj_count_offd[jl-1];

     P_marker = hypre_CTAlloc(int, n_fine);
     P_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);

     for (i = 0; i < n_fine; i++)
     {      
        P_marker[i] = -1;
     }
     for (i = 0; i < num_cols_A_offd; i++)
     {      
        P_marker_offd[i] = -1;
     }
     strong_f_marker = -2;
 
     for (i = ns; i < ne; i++)
     {
             
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
      
      if (CF_marker[i] >= 0)
      {
         P_diag_i[i] = jj_counter;
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         /*P_diag_data[jj_counter] = one;*/
	 for(ib = 0; ib < block_size; ib++) {
	   for(jb = 0; jb < block_size; jb++) {
	     if(ib == jb) P_diag_data[jj_counter*block_size*block_size + ib*block_size + jb] = one;
	     else P_diag_data[jj_counter*block_size*block_size + ib*block_size + jb] = zero;
	   }
	 }
         jj_counter++;
      }
      
      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else
      {         
         /* Diagonal part of P */
         P_diag_i[i] = jj_counter;
         jj_begin_row = jj_counter;

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];   

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1] = jj_counter;
               P_diag_j[jj_counter]    = fine_to_coarse[i1];
               /*P_diag_data[jj_counter] = zero;*/
	       for(ib = 0; ib < block_size; ib++) {
		 for(jb = 0; jb < block_size; jb++) {
		   P_diag_data[jj_counter*block_size*block_size + ib*block_size + jb] = zero;
		 }
	       }
               jj_counter++;
            }

            /*--------------------------------------------------------------
             * If neighbor i1 is an F-point, mark it as a strong F-point
             * whose connection needs to be distributed.
             *--------------------------------------------------------------*/

            else if (CF_marker[i1] != -3)
            {
               P_marker[i1] = strong_f_marker;
            }            
         }
         jj_end_row = jj_counter;

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         jj_begin_row_offd = jj_counter_offd;


         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];
                  /*P_offd_data[jj_counter_offd] = zero;*/
		  for(ib = 0; ib < block_size; ib++) {
		    for(jb = 0; jb < block_size; jb++) {
		      P_diag_data[jj_counter_offd*block_size*block_size + ib*block_size + jb] = zero;
		    }
		  }
                  jj_counter_offd++;
               }

               /*-----------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
                  P_marker_offd[i1] = strong_f_marker;
               }            
            }
         }
      
         jj_end_row_offd = jj_counter_offd;
         
         /*diagonal = A_diag_data[A_diag_i[i]];*/
	 for(ib = 0; ib < block_size; ib++) {
	   for(jb = 0; jb < block_size; jb++) {
	     diagonal[ib*block_size + jb] = A_diag_data[A_diag_i[i] + ib*block_size + jb];
	   }
	 }

     
         /* Loop over ith row of A.  First, the diagonal part of A */

         for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
         {
            i1 = A_diag_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row)
            {
	      /*P_diag_data[P_marker[i1]] += A_diag_data[jj];*/
	      for(ib = 0; ib < block_size; ib++) {
		for(jb = 0; jb < block_size; jb++) {
		  P_diag_data[P_marker[i1]*block_size*block_size + ib*block+size + jb] += A_diag_data[jj*block_size*block_size + ib*block_size + jb];
		}
	      }
            }

            /*--------------------------------------------------------------
             * Case 2: neighbor i1 is an F-point and strongly influences i,
             * distribute a_{i,i1} to C-points that strongly infuence i.
             * Note: currently no distribution to the diagonal in this case.
             *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
	      /*sum = zero;*/
	      for(ib = 0; ib < block_size; ib++) {
		for(jb = 0; jb < block_size; jb++) {
		  sum[ib*block+size + jb] = zero;
		}
	      }
               
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *-----------------------------------------------------------*/
	       /*sgn = 1;
	       if (A_diag_data[A_diag_i[i1]] < 0) sgn = -1;*/
               /* Diagonal block part of row i1 */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  /*if (P_marker[i2] >= jj_begin_row && 
					(sgn*A_diag_data[jj1]) < 0)
                  {
                     sum += A_diag_data[jj1];
		  }*/
		  if(P_marker[i2] >= jj_begin_row) {
		    for(ib = 0; ib < block_size; ib++) {
		      for(jb = 0; jb < block_size; jb++) {
			sum[ib*block_size + jb] += A_diag_data[jj1*block_size*block_size + ib*block_size + jb];
		      }
		    }
		  }
               }

               /* Off-Diagonal block part of row i1 */ 
               if (num_procs > 1)
               {              
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     /*if (P_marker_offd[i2] >= jj_begin_row_offd
				&& (sgn*A_offd_data[jj1]) < 0)
                     {
                        sum += A_offd_data[jj1];
		     }*/
		     if(P_marker_offd[i2] >= jj_begin_row_offd) {
		       for(ib = 0; ib < block_size; ib++) {
			 for(jb = 0; jb < block_size; jb++) {
			   sum[ib*block_size + jb] += A_offd_data[jj1*block_size*block_size + ib*block_size + jb];
			 }
		       }
		     }
                  }
               } 

               /*if (sum != 0)
	       {*/
	       // RYAN -- This is where I need to be able to "divide" by a block;  also should check if order of operations is correct
	       /*distribute = A_diag_data[jj] / sum;*/
 
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and do the distribution.
                *-----------------------------------------------------------*/

               /* Diagonal block part of row i1 */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  /*if (P_marker[i2] >= jj_begin_row 
				&& (sgn*A_diag_data[jj1]) < 0)
                  {
                     P_diag_data[P_marker[i2]]
                                  += distribute * A_diag_data[jj1];
		  }*/
		  if(P_marker[i2] >= jj_begin_row) {
		    for(ib = 0; ib < block_size; ib++) {
		      for(jb = 0; jb < block_size; jb++) {
			for(kb = 0; kb < block_size; kb++) {
			  P_diag_data[P_marker[i2]*block_size*block_size + ib*block_size + jb] += distribute[ib*block_size + kb]*A_diag_data[jj1*block_size*block_size + kb*block_size + jb];
			}
		      }
		    }
		  }
               }

               /* Off-Diagonal block part of row i1 */
               if (num_procs > 1)
               {
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     /*if (P_marker_offd[i2] >= jj_begin_row_offd
				&& (sgn*A_offd_data[jj1]) < 0)
                     {
                         P_offd_data[P_marker_offd[i2]]    
                                  += distribute * A_offd_data[jj1]; 
		     }*/
		     if(P_marker_offd[i2] >= jj_begin_row_offd) {
		       for(ib = 0; ib < block_size; ib++) {
			 for(jb = 0; jb < block_size; jb++) {
			   for(kb = 0; kb < block_size; kb++) {
			     P_offd_data[P_marker[i2]*block_size*block_size + ib*block_size + jb] += distribute[ib*block_size + kb]*A_offd_data[jj1*block_size*block_size + kb*block_size + jb];
			   }
			 }
		       }
		     }
                  }
               }
               /*}
               else
               {
		  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     diagonal += A_diag_data[jj];
	       }*/
            }
            
            /*--------------------------------------------------------------
             * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal.
             *--------------------------------------------------------------*/

            else if (CF_marker[i1] != -3)
            {
	      /*if (num_functions == 1 || dof_func[i] == dof_func[i1])
		 diagonal += A_diag_data[jj];*/
	      for(ib = 0; ib < block_size; ib++) {
		for(jb = 0; jb < block_size; jb++) {
		  diagonal[ib*block_size + jb] += A_diag_data[jj*block_size*block_size + ib*block_size + jb];
		}
	      }
            } 

         }    
       

          /*----------------------------------------------------------------
           * Still looping over ith row of A. Next, loop over the 
           * off-diagonal part of A 
           *---------------------------------------------------------------*/

         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
            {
               i1 = A_offd_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
		 /*P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];*/
		 for(ib = 0; ib < block_size; ib++) {
		   for(jb = 0; jb < block_size; jb++) {
		     P_offd_data[P_marker_offd[i1]*block_size*block_size + ib*block_size + jb] += A_offd_data[jj*block_size*block_size + ib*block_size + jb];
		   }
		 }
               }

               /*------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *-----------------------------------------------------------*/
            
               else if (P_marker_offd[i1] == strong_f_marker)
               {
		 /*sum = zero;*/
		 for(ib = 0; ib < block_size; ib++) {
		   for(jb = 0; jb < block_size; jb++) {
		     sum[ib*block+size + jb] = zero;
		   }
		 }
               
               /*---------------------------------------------------------
                * Loop over row of A_ext for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *---------------------------------------------------------*/

                  /* find row number */
                  c_num = A_offd_j[jj];

		  /*sgn = 1;
		    if (A_ext_data[A_ext_i[c_num]] < 0) sgn = -1;*/
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];
                                         
                     if (i2 >= col_1 && i2 < col_n)    
                     {                            
                                           /* in the diagonal block */
		       /*if (P_marker[i2-col_1] >= jj_begin_row
				&& (sgn*A_ext_data[jj1]) < 0)
                        {
                           sum += A_ext_data[jj1];
			}*/
		       if(P_marker[i2-col_1] >= jj_begin_row) {
			 for(ib = 0; ib < block_size; ib++) {
			   for(jb = 0; jb < block_size; jb++) {
			     sum[ib*block_size + jb] += A_ext_data[jj1*block_size*block_size + ib*block_size + jb];
			   }
			 }
		       }
                     }
                     else                       
                     {                          
                                           /* in the off_diagonal block  */
			if (i2 > -1)
			{
                           /*bin++;*/
                           j = hypre_BinarySearch(col_map_offd,i2,num_cols_A_offd);
			   if (j != -1)
			      i2 = -j-2;
			   else 
			      i2 = -1;
			   A_ext_j[jj1] = i2;
		        }
                        if (i2 != -1)
                        { 
			  /*if (P_marker_offd[-i2-2] >= jj_begin_row_offd
				&& (sgn*A_ext_data[jj1]) < 0)
                           {
			      sum += A_ext_data[jj1];
			   }*/
			  if(P_marker_offd[-i2-2] >= jj_begin_row_offd) {
			    for(ib = 0; ib < block_size; ib++) {
			      for(jb = 0; jb < block_size; jb++) {
				sum[ib*block_szie + jb] += A_ext_data[jj1*block_size*block_size + ib*block_size + jb];
			      }
			    }
			  }
                        }
 
                     }

                  }

                  /*if (sum != 0)
		  {*/
		  //RYAN -- need division again.
		  //distribute = A_offd_data[jj] / sum;   
                  /*---------------------------------------------------------
                   * Loop over row of A_ext for point i1 and do 
                   * the distribution.
                   *--------------------------------------------------------*/

                  /* Diagonal block part of row i1 */
                          
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];

                     if (i2 >= col_1 && i2 < col_n) /* in the diagonal block */           
                     {
		       /*if (P_marker[i2-col_1] >= jj_begin_row
				&& (sgn*A_ext_data[jj1]) < 0)
                        {
                           P_diag_data[P_marker[i2-col_1]]
                                     += distribute * A_ext_data[jj1];
			}*/
		       if(P_marker[i2-col_1] >= jj_begin_row) {
			 for(ib = 0; ib < block_size; ib++) {
			   for(jb = 0; jb < block_size; jb++) {
			     for(kb = 0; kb < block_size; kb++) {
			       P_diag_data[P_marker[i2]*block_size*block_size + ib*block_size + jb] += distribute[ib*block_size + kb]*A_ext_data[jj1*block_size*block_size + kb*block_size + jb];
			     }
			   }
			 }
		       }
                     }
                     else
                     {
                        /* check to see if it is in the off_diagonal block  */
                        if (i2 > -1)
	                {
                           /*bin++;*/
                           j = hypre_BinarySearch(col_map_offd,i2,num_cols_A_offd);
			   if (j != -1)
			      i2 = -1;
			   else
			      i2 = -j-2;
			   A_ext_j[jj1] = i2;
	                }
                        if (i2 != -1)
                        { 
			  /*if (P_marker_offd[-i2-2] >= jj_begin_row_offd
				&& (sgn*A_ext_data[jj1]) < 0)
                                  P_offd_data[P_marker_offd[-i2-2]]
				     += distribute * A_ext_data[jj1];*/
			  if(P_marker_offd[-i2-2] >= jj_begin_row_offd) {
			    for(ib = 0; ib < block_size; ib++) {
			      for(jb = 0; jb < block_size; jb++) {
				for(kb = 0; kb < block_size; kb++) {
				  P_offd_data[P_marker[-i2-2]*block_size*block_size + ib*block_size + jb] += distribute[ib*block_size + kb]*A_ext_data[jj1*block_size*block_size + kb*block_size + jb];
				}
			      }
			    }
			  }
                        }
                     }
                  }
                  /*}
		  else
                  {
	             if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        diagonal += A_offd_data[jj];
		  }*/
               }
            
               /*-----------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
		 /*if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
		     diagonal += A_offd_data[jj];*/
		 for(ib = 0; ib < block_size; ib++) {
		   for(jb = 0; jb < block_size; jb++) {
		     diagonal[ib*block_size + jb] += A_offd_data[jj*block_size*block_size + ib*block_size + jb];
		   }
		 }
               } 

            }
         }           

        /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         for (jj = jj_begin_row; jj < jj_end_row; jj++)
         {
	   // RYAN -- division
	   //P_diag_data[jj] /= -diagonal;
         }

         for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
         {
	   // RYAN -- division
	   //P_offd_data[jj] /= -diagonal;
         }
           
      }

      strong_f_marker--; 

      P_offd_i[i+1] = jj_counter_offd;
     }
     hypre_TFree(P_marker);
     hypre_TFree(P_marker_offd);
   }

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   // RYAN -- I am going to remove this block and hope I don't need it.

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(int, P_offd_size);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < P_offd_size; i++)
	 P_marker[i] = P_offd_j[i];

      qsort0(P_marker, 0, P_offd_size-1);

      num_cols_P_offd = 1;
      index = P_marker[0];
      for (i=1; i < P_offd_size; i++)
      {
	if (P_marker[i] > index)
	{
 	  index = P_marker[i];
 	  P_marker[num_cols_P_offd++] = index;
  	}
      }

      col_map_offd_P = hypre_CTAlloc(int,num_cols_P_offd);

      for (i=0; i < num_cols_P_offd; i++)
         col_map_offd_P[i] = P_marker[i];

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < P_offd_size; i++)
	P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
					 P_offd_j[i],
					 num_cols_P_offd);
      hypre_TFree(P_marker); 
   }

   // RYAN -- here

   P = hypre_ParCSRBlockMatrixCreate(comm, block_size, 
                                hypre_ParCSRBlockMatrixGlobalNumRows(A), 
                                total_global_cpts,
                                hypre_ParCSRBlockMatrixColStarts(A),
                                num_cpts_global,
                                num_cols_P_offd, 
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);

   P_diag = hypre_ParCSRBlockMatrixDiag(P);
   hypre_CSRBlockMatrixData(P_diag) = P_diag_data; 
   hypre_CSRBlockMatrixI(P_diag) = P_diag_i; 
   hypre_CSRBlockMatrixJ(P_diag) = P_diag_j; 
   hypre_ParCSRBlockMatrixOwnsRowStarts(P) = 0; 

   /*-------------------------------------------------------------------
    * The following block was originally in an 
    *
    *           if (num_cols_P_offd)
    *
    * block, which has been eliminated to ensure that the code 
    * runs on one processor.
    *
    *-------------------------------------------------------------------*/

   for (i=0; i < n_fine; i++)
      if (CF_marker[i] == -3) CF_marker[i] = -1;

   P_offd = hypre_ParCSRBlockMatrixOffd(P);
   hypre_CSRBlockMatrixI(P_offd) = P_offd_i; 
   if (num_cols_P_offd)
   { 
	hypre_CSRBlockMatrixData(P_offd) = P_offd_data; 
   	hypre_CSRBlockMatrixJ(P_offd) = P_offd_j; 
   	hypre_ParCSRBlockMatrixColMapOffd(P) = col_map_offd_P;
   } 
   hypre_ParCSRBlockMatrixOffd(P) = P_offd;
   // RYAN -- what the heck is the next line?  I think I'll need to make a call to this after converting back to a non-block matrix.
   //hypre_GetCommPkgRTFromCommPkgA(P,A);

   *P_ptr = P;

   hypre_TFree(CF_marker_offd);
   hypre_TFree(dof_func_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   hypre_TFree(diagonal);
   hypre_TFree(sum);
   hypre_TFree(distribute);

   if (num_procs > 1) hypre_CSRBlockMatrixDestroy(A_ext);
   /* printf (" bin count : %d\n", bin);*/

   return(0);  

}            
