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
 * Member functions for hypre_ParCSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CreateParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_CreateParCSRMatrix( MPI_Comm comm,
                       	  int global_num_rows,
                       	  int global_num_cols,
			  int first_row_index,
			  int first_col_diag,
                       	  int local_num_rows,
                       	  int local_num_cols,
                       	  int num_cols_offd,
			  int num_nonzeros_diag,
			  int num_nonzeros_offd)
{
   hypre_ParCSRMatrix  *matrix;
   int	num_procs, my_id;
   
   matrix = hypre_CTAlloc(hypre_ParCSRMatrix, 1);

   if (!first_row_index && !first_col_diag && !local_num_rows 
		&& !local_num_cols )
   {
   	MPI_Comm_rank(comm,&my_id);
   	MPI_Comm_size(comm,&num_procs);
   	MPE_Decomp1d(global_num_rows,num_procs,my_id,&first_row_index,
			&local_num_rows);
	first_row_index--;
   	local_num_rows -= first_row_index;
   	MPE_Decomp1d(global_num_cols,num_procs,my_id,&first_col_diag,
			&local_num_cols);
	first_col_diag--;
   	local_num_cols -= first_col_diag;
   }

   hypre_ParCSRMatrixComm(matrix) = comm;
   hypre_ParCSRMatrixDiag(matrix) = hypre_CreateCSRMatrix(local_num_rows,
		local_num_cols,num_nonzeros_diag);
   hypre_ParCSRMatrixOffd(matrix) = hypre_CreateCSRMatrix(local_num_rows,
		num_cols_offd,num_nonzeros_offd);
   hypre_ParCSRMatrixGlobalNumRows(matrix) = global_num_rows;
   hypre_ParCSRMatrixGlobalNumCols(matrix) = global_num_cols;
   hypre_ParCSRMatrixFirstRowIndex(matrix) = first_row_index;
   hypre_ParCSRMatrixFirstColDiag(matrix) = first_col_diag;
   hypre_ParCSRMatrixColMapOffd(matrix) = NULL;
   hypre_ParCSRMatrixCommPkg(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(matrix) = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_DestroyParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_DestroyParCSRMatrix( hypre_ParCSRMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      if ( hypre_CSRMatrixOwnsData(matrix) )
      {
         hypre_DestroyCSRMatrix(hypre_ParCSRMatrixDiag(matrix));
         hypre_DestroyCSRMatrix(hypre_ParCSRMatrixOffd(matrix));
      	 if (hypre_ParCSRMatrixColMapOffd(matrix))
      	      hypre_TFree(hypre_ParCSRMatrixColMapOffd(matrix));
         if (hypre_ParCSRMatrixCommPkg(matrix))
	      hypre_DestroyMatvecCommPkg(hypre_ParCSRMatrixCommPkg(matrix));
      }
      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeParCSRMatrix( hypre_ParCSRMatrix *matrix )
{
   int  ierr=0;

   if ( !hypre_ParCSRMatrixDiag(matrix) )
   	hypre_InitializeCSRMatrix(hypre_ParCSRMatrixDiag(matrix));
   if ( !hypre_ParCSRMatrixOffd(matrix) )
   {
   	hypre_InitializeCSRMatrix(hypre_ParCSRMatrixOffd(matrix));
	hypre_ParCSRMatrixColMapOffd(matrix) = 
		hypre_CTAlloc(int,hypre_CSRMatrixNumCols(
		hypre_ParCSRMatrixOffd(matrix)));
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetParCSRMatrixDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_SetParCSRMatrixDataOwner( hypre_ParCSRMatrix *matrix,
                                int              owns_data )
{
   int    ierr=0;

   hypre_ParCSRMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ReadParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_ReadParCSRMatrix( MPI_Comm comm, char *file_name )
{
   hypre_ParCSRMatrix  *matrix;
   hypre_CSRMatrix  *diag;
   hypre_CSRMatrix  *offd;
   int	my_id, i;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   int  global_num_rows, global_num_cols, num_cols_offd;
   int  num_nonzeros_diag, num_nonzeros_offd;
   int  first_row_index, first_col_diag;
   int  local_num_rows, local_num_cols;
   int  *col_map_offd;
   FILE *fp;

   MPI_Comm_rank(comm,&my_id);
   sprintf(new_file_d,"%s.D.%d",file_name,my_id);
   sprintf(new_file_o,"%s.O.%d",file_name,my_id);
   sprintf(new_file_info,"%s.INFO.%d",file_name,my_id);
   fp = fopen(new_file_info, "r");
   fscanf(fp, "%d", &global_num_rows);
   fscanf(fp, "%d", &global_num_cols);
   fscanf(fp, "%d", &num_cols_offd);
   fscanf(fp, "%d", &first_row_index);
   fscanf(fp, "%d", &first_col_diag);
   col_map_offd = hypre_CTAlloc(int, num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
   	fscanf(fp, "%d", &col_map_offd[i]);
	
   fclose(fp);

   diag = hypre_ReadCSRMatrix(new_file_d);
   local_num_rows = hypre_CSRMatrixNumRows(diag);
   local_num_cols = hypre_CSRMatrixNumCols(diag);
   num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(diag);

   if (num_cols_offd != 0)
   {
	offd = hypre_ReadCSRMatrix(new_file_o);
        num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(offd);
   }
   else
        num_nonzeros_offd = 0;
	
   matrix = hypre_CreateParCSRMatrix (comm, global_num_rows,
		global_num_cols, first_row_index, first_col_diag,
		local_num_rows, local_num_cols, num_cols_offd,
		num_nonzeros_diag, num_nonzeros_offd);

   hypre_ParCSRMatrixDiag(matrix) = diag;
   if (num_cols_offd != 0)
   	hypre_ParCSRMatrixOffd(matrix) = offd;
   else
   	hypre_ParCSRMatrixOffd(matrix) = NULL;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_PrintParCSRMatrix
 *--------------------------------------------------------------------------*/

int
hypre_PrintParCSRMatrix( hypre_ParCSRMatrix *matrix, 
                         char            *file_name )
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(matrix);
   int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(matrix);
   int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(matrix);
   int *col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix);
   int	my_id, i;
   char	  new_file_d[80], new_file_o[80], new_file_info[80];
   int  ierr = 0;
   FILE *fp;
   int num_cols_offd = 0;

   if (hypre_ParCSRMatrixOffd(matrix))
   	num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(matrix));

   MPI_Comm_rank(comm, &my_id);
   
   sprintf(new_file_d,"%s.D.%d",file_name,my_id);
   sprintf(new_file_o,"%s.O.%d",file_name,my_id);
   sprintf(new_file_info,"%s.INFO.%d",file_name,my_id);
   hypre_PrintCSRMatrix(hypre_ParCSRMatrixDiag(matrix),new_file_d);
   if (num_cols_offd != 0)
	hypre_PrintCSRMatrix(hypre_ParCSRMatrixOffd(matrix),new_file_o);
  
   fp = fopen(new_file_info, "w");
   fprintf(fp, "%d\n", global_num_rows);
   fprintf(fp, "%d\n", global_num_cols);
   fprintf(fp, "%d\n", num_cols_offd);
   fprintf(fp, "%d\n", hypre_ParCSRMatrixFirstRowIndex(matrix));
   fprintf(fp, "%d\n", hypre_ParCSRMatrixFirstColDiag(matrix));
   for (i=0; i < num_cols_offd; i++)
   	fprintf(fp, "%d\n", col_map_offd[i]);
   fclose(fp);

   return ierr;
}

hypre_ParCSRMatrix *
hypre_CSRMatrixToParCSRMatrix( MPI_Comm comm, hypre_CSRMatrix *A,
			       int **row_starts_ptr,
                               int **col_starts_ptr )
{
   int		global_data[2];
   int		global_num_rows;
   int		global_num_cols;
   int		*local_num_rows;

   int		num_procs, my_id;
   int		*local_num_nonzeros;
   int		num_nonzeros;
  
   double	*a_data;
   int		*a_i;
   int		*a_j;
  
   hypre_CSRMatrix *local_A;

   MPI_Request  *requests;
   MPI_Status	*status, status0;
   MPI_Datatype *csr_matrix_datatypes;

   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   hypre_ParCSRMatrix *par_matrix;

   int 		first_col_diag;
   int 		last_col_diag;
   int *row_starts;
   int *col_starts;
 
   int i, j, ind;
   int no_row_starts = 0;
   int no_col_starts = 0;


   row_starts = *row_starts_ptr;
   col_starts = *col_starts_ptr;

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);

   if (my_id == 0) 
   {
   	global_data[0] = hypre_CSRMatrixNumRows(A);
	global_data[1] = hypre_CSRMatrixNumCols(A);
   	a_data = hypre_CSRMatrixData(A);
   	a_i = hypre_CSRMatrixI(A);
   	a_j = hypre_CSRMatrixJ(A);
   }
   MPI_Bcast(global_data,2,MPI_INT,0,comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];

   local_num_rows = hypre_CTAlloc(int, num_procs);
   csr_matrix_datatypes = hypre_CTAlloc(MPI_Datatype, num_procs);

   if (!row_starts)
   {
	no_row_starts = 1;
	row_starts = hypre_CTAlloc(int, num_procs+1);
   	for (i=0; i < num_procs; i++)
   	{
	 MPE_Decomp1d(global_num_rows, num_procs, i, &row_starts[i], 
		&local_num_rows[i]);
	 row_starts[i]--;
   	}
	row_starts[num_procs] = global_num_rows;
   }
   for (i=0; i < num_procs; i++)
	 local_num_rows[i] = row_starts[i+1] - row_starts[i];

   if (my_id == 0)
   {
   	local_num_nonzeros = hypre_CTAlloc(int, num_procs);
   	for (i=0; i < num_procs-1; i++)
		local_num_nonzeros[i] = a_i[row_starts[i+1]] 
				- a_i[row_starts[i]];
   	local_num_nonzeros[num_procs-1] = a_i[global_num_rows] 
				- a_i[row_starts[num_procs-1]];
   }
   MPI_Scatter(local_num_nonzeros,1,MPI_INT,&num_nonzeros,1,MPI_INT,0,comm);

   if (my_id == 0) num_nonzeros = local_num_nonzeros[0];

   local_A = hypre_CreateCSRMatrix(local_num_rows[my_id], global_num_cols,
		num_nonzeros);
   if (my_id == 0)
   {
	requests = hypre_CTAlloc (MPI_Request, num_procs-1);
	status = hypre_CTAlloc(MPI_Status, num_procs-1);
	j=0;
	for (i=1; i < num_procs; i++)
	{
		ind = a_i[row_starts[i]];
		BuildCSRMatrixMPIDataType(local_num_nonzeros[i], 
			local_num_rows[i],
			&a_data[ind],
			&a_i[row_starts[i]],
			&a_j[ind],
			&csr_matrix_datatypes[i]);
		MPI_Isend(MPI_BOTTOM, 1, csr_matrix_datatypes[i], i, 0, comm,
			&requests[j++]);
    		MPI_Type_free(&csr_matrix_datatypes[i]);
	}
   	hypre_CSRMatrixData(local_A) = a_data;
   	hypre_CSRMatrixI(local_A) = a_i;
   	hypre_CSRMatrixJ(local_A) = a_j;
     	hypre_CSRMatrixOwnsData(local_A) = 0;
	MPI_Waitall(num_procs-1,requests,status);
	hypre_TFree(requests);
	hypre_TFree(status);
	hypre_TFree(local_num_nonzeros);
    }
   else
   {
	hypre_InitializeCSRMatrix(local_A);
	BuildCSRMatrixMPIDataType(num_nonzeros, 
			local_num_rows[my_id],
			hypre_CSRMatrixData(local_A),
			hypre_CSRMatrixI(local_A),
			hypre_CSRMatrixJ(local_A),
			csr_matrix_datatypes);
	MPI_Recv(MPI_BOTTOM,1,csr_matrix_datatypes[0],0,0,comm,&status0);
	MPI_Type_free(csr_matrix_datatypes);
   }

   if (!col_starts)
   {
	no_col_starts = 1;
	col_starts = hypre_CTAlloc(int,num_procs+1);
	for (i=0; i < num_procs; i++)
   	{
    	    MPE_Decomp1d(global_num_cols, num_procs, i, &col_starts[i],
		&last_col_diag);
	    col_starts[i]--;
   	}
	col_starts[num_procs] = global_num_cols;
   }
   
   {
	first_col_diag = col_starts[my_id];
	last_col_diag = col_starts[my_id+1]-1;
   }

   par_matrix = hypre_CreateParCSRMatrix (comm, global_num_rows,
	global_num_cols,row_starts[my_id],first_col_diag,local_num_rows[my_id],
	last_col_diag-first_col_diag+1,0,0,0);
   diag = hypre_ParCSRMatrixDiag(par_matrix);
   offd = hypre_ParCSRMatrixOffd(par_matrix);

   GenerateDiagAndOffd(local_A, par_matrix, first_col_diag, last_col_diag);


   hypre_DestroyCSRMatrix(local_A);
   hypre_TFree(local_num_rows);
   hypre_TFree(csr_matrix_datatypes);
/*   if (no_row_starts) hypre_TFree(row_starts);
   if (no_col_starts) hypre_TFree(col_starts);
*/
   *row_starts_ptr = row_starts;
   *col_starts_ptr = col_starts;

   return par_matrix;
}

int
GenerateDiagAndOffd(hypre_CSRMatrix *A,
		    hypre_ParCSRMatrix *matrix,
		    int	first_col_diag,
		    int last_col_diag)
{
   int  i, j;
   int  jo, jd;
   int  ierr = 0;
   int	num_rows = hypre_CSRMatrixNumRows(A);
   int  num_cols = hypre_CSRMatrixNumCols(A);
   double *a_data = hypre_CSRMatrixData(A);
   int *a_i = hypre_CSRMatrixI(A);
   int *a_j = hypre_CSRMatrixJ(A);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);

   int	*col_map_offd;

   double *diag_data, *offd_data;
   int 	*diag_i, *offd_i;
   int 	*diag_j, *offd_j;
   int 	*marker;
   int num_cols_diag, num_cols_offd;
   int first_elmt = a_i[0];
   int num_nonzeros = a_i[num_rows]-first_elmt;
   int counter;

   num_cols_diag = last_col_diag - first_col_diag +1;
   num_cols_offd = 0;

   if ((num_cols - num_cols_diag) != 0)
   {
	hypre_InitializeCSRMatrix(diag);
   	diag_i = hypre_CSRMatrixI(diag);

	hypre_InitializeCSRMatrix(offd);
   	offd_i = hypre_CSRMatrixI(offd);
   	marker = hypre_CTAlloc(int,num_cols);

	for (i=0; i < num_cols; i++)
		marker[i] = 0;
	
   	jo = 0;
   	jd = 0;
   	for (i=0; i < num_rows; i++)
   	{
	    offd_i[i] = jo;
	    diag_i[i] = jd;
   
	    for (j=a_i[i]-first_elmt; j < a_i[i+1]-first_elmt; j++)
	   	if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
	   	{
			if (!marker[a_j[j]])
			{
				marker[a_j[j]] = 1;
				num_cols_offd++;
			}
			jo++;
	   	}
	   	else
		{
			jd++;
	   	}
   	}
   	offd_i[num_rows] = jo;
   	diag_i[num_rows] = jd;

  	hypre_ParCSRMatrixColMapOffd(matrix) = hypre_CTAlloc(int,num_cols_offd);
	col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix);

	counter = 0;
	for (i=0; i < num_cols; i++)
		if (marker[i])
		{
			col_map_offd[counter] = i;
			marker[i] = counter;
			counter++;
		}

   	hypre_CSRMatrixNumNonzeros(diag) = jd;
 	hypre_InitializeCSRMatrix(diag);
   	diag_data = hypre_CSRMatrixData(diag);
   	diag_j = hypre_CSRMatrixJ(diag);

   	hypre_CSRMatrixNumNonzeros(offd) = jo;
   	hypre_CSRMatrixNumCols(offd) = num_cols_offd;
 	hypre_InitializeCSRMatrix(offd);
   	offd_data = hypre_CSRMatrixData(offd);
   	offd_j = hypre_CSRMatrixJ(offd);

   	jo = 0;
   	jd = 0;
   	for (i=0; i < num_rows; i++)
   	{
	    for (j=a_i[i]-first_elmt; j < a_i[i+1]-first_elmt; j++)
	   	if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
	   	{
			offd_data[jo] = a_data[j];
			offd_j[jo++] = marker[a_j[j]];
	   	}
	   	else
		{
			diag_data[jd] = a_data[j];
			diag_j[jd++] = a_j[j]-first_col_diag;
	   	}
   	}
	hypre_TFree(marker);
   }
   else 
   {
   	hypre_CSRMatrixNumNonzeros(diag) = num_nonzeros;
 	hypre_InitializeCSRMatrix(diag);
   	diag_data = hypre_CSRMatrixData(diag);
   	diag_i = hypre_CSRMatrixI(diag);
   	diag_j = hypre_CSRMatrixJ(diag);

	for (i=0; i < num_nonzeros; i++)
   	{
		diag_data[i] = a_data[i];
		diag_j[i] = a_j[i];
       	}
	for (i=0; i < num_rows+1; i++)
		diag_i[i] = a_i[i];

	hypre_CSRMatrixNumCols(offd) = 0;
   }
   
   return ierr;
}
/*
hypre_CSRMatrix *
hypre_MergeDiagAndOffd(hypre_ParCSRMatrix *par_matrix)
{
   hypre_CSRMatrix  *diag = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix  *offd = hypre_ParCSRMatrixOffd(par_matrix);
   hypre_CSRMatrix  *matrix;

   int 		num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   int 		first_col_diag = hypre_ParCSRMatrixFirstColDiag(par_matrix);
   int 		num_rows = hypre_CSRMatrixNumRows(diag);
   int 		num_cols_diag = hypre_CSRMatrixNumCols(diag);

   int		*diag_i = hypre_CSRMatrixI(diag);
   int		*diag_j = hypre_CSRMatrixJ(diag);
   double	*diag_data = hypre_CSRMatrixData(diag);
   int		*offd_i = hypre_CSRMatrixI(offd);
   int		*offd_j = hypre_CSRMatrixJ(offd);
   double	*offd_data = hypre_CSRMatrixData(offd);

   int		*matrix_i;
   int		*matrix_j;
   double	*matrix_data;

   int		num_nonzeros, i, j;

   num_cols_offd = num_cols - num_cols_diag;
   if (num_cols_offd != 0)
	num_nonzeros = diag_i[num_rows] + offd_i[num_rows];
   else
	num_nonzeros = diag_i[num_rows];

   matrix = hypre_CreateCSRMatrix(num_rows,num_cols,num_nonzeros);
   hypre_InitializeCSRMatrix(matrix);

   for (i=0; i < num_rows; i++)
   {
	for (j=diag_i[i]; j < diag_i[i+1]-1; j++)
	{
		matrix_data[count] = diag_data[j];
		matrix_j[count] = diag_j[j]+first_col_diag;
 	}
*/	
