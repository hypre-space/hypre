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

#include "../seq_matrix_vector/HYPRE_seq_mv.h"
/* In addition to publically accessible interface in HYPRE_mv.h, the implementation
   in this file uses accessor macros into the sequential matrix structure, and
   so includes the .h that defines that structure. Should those accessor functions
   become proper functions at some later date, this will not be necessary. AJC 4/99
*/
#include "../seq_matrix_vector/csr_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_CreateParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_CreateParCSRMatrix( MPI_Comm comm,
                       	  int global_num_rows,
                       	  int global_num_cols,
			  int *row_starts,
			  int *col_starts,
                       	  int num_cols_offd,
			  int num_nonzeros_diag,
			  int num_nonzeros_offd)
{
   hypre_ParCSRMatrix  *matrix;
   int	num_procs, my_id;
   int local_num_rows, local_num_cols;
   int first_row_index, first_col_diag;
   
   matrix = hypre_CTAlloc(hypre_ParCSRMatrix, 1);

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
   hypre_ParCSRMatrixRowStarts(matrix) = row_starts;
   hypre_ParCSRMatrixColStarts(matrix) = col_starts;
   hypre_ParCSRMatrixCommPkg(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(matrix) = 1;
   hypre_ParCSRMatrixOwnsRowStarts(matrix) = 1;
   hypre_ParCSRMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
   	hypre_ParCSRMatrixOwnsColStarts(matrix) = 0;

   hypre_ParCSRMatrixRowindices(matrix) = NULL;
   hypre_ParCSRMatrixRowvalues(matrix) = NULL;
   hypre_ParCSRMatrixGetrowactive(matrix) = 0;

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
      if ( hypre_ParCSRMatrixOwnsData(matrix) )
      {
         hypre_DestroyCSRMatrix(hypre_ParCSRMatrixDiag(matrix));
         hypre_DestroyCSRMatrix(hypre_ParCSRMatrixOffd(matrix));
      	 if (hypre_ParCSRMatrixColMapOffd(matrix))
      	      hypre_TFree(hypre_ParCSRMatrixColMapOffd(matrix));
         if (hypre_ParCSRMatrixCommPkg(matrix))
	      hypre_DestroyMatvecCommPkg(hypre_ParCSRMatrixCommPkg(matrix));
      }
      if ( hypre_ParCSRMatrixOwnsRowStarts(matrix) )
      	      hypre_TFree(hypre_ParCSRMatrixRowStarts(matrix));
      if ( hypre_ParCSRMatrixOwnsColStarts(matrix) )
      	      hypre_TFree(hypre_ParCSRMatrixColStarts(matrix));

      hypre_TFree(hypre_ParCSRMatrixRowindices(matrix));
      hypre_TFree(hypre_ParCSRMatrixRowvalues(matrix));

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

   hypre_InitializeCSRMatrix(hypre_ParCSRMatrixDiag(matrix));
   hypre_InitializeCSRMatrix(hypre_ParCSRMatrixOffd(matrix));
   hypre_ParCSRMatrixColMapOffd(matrix) = 
		hypre_CTAlloc(int,hypre_CSRMatrixNumCols(
		hypre_ParCSRMatrixOffd(matrix)));
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetParCSRMatrixNumNonzeros
 *--------------------------------------------------------------------------*/

int 
hypre_SetParCSRMatrixNumNonzeros( hypre_ParCSRMatrix *matrix)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(matrix);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
   int *diag_i = hypre_CSRMatrixI(diag);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);
   int *offd_i = hypre_CSRMatrixI(offd);
   int local_num_rows = hypre_CSRMatrixNumRows(diag);
   int total_num_nonzeros;
   int local_num_nonzeros;
   int ierr = 0;

   local_num_nonzeros = diag_i[local_num_rows] + offd_i[local_num_rows];
   MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, MPI_INT,
	MPI_SUM, comm);
   hypre_ParCSRMatrixNumNonzeros(matrix) = total_num_nonzeros;
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
 * hypre_SetParCSRMatrixRowStartsOwner
 *--------------------------------------------------------------------------*/

int 
hypre_SetParCSRMatrixRowStartsOwner( hypre_ParCSRMatrix *matrix,
                                     int owns_row_starts )
{
   int    ierr=0;

   hypre_ParCSRMatrixOwnsRowStarts(matrix) = owns_row_starts;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetParCSRMatrixColStartsOwner
 *--------------------------------------------------------------------------*/

int 
hypre_SetParCSRMatrixColStartsOwner( hypre_ParCSRMatrix *matrix,
                                     int owns_col_starts )
{
   int    ierr=0;

   hypre_ParCSRMatrixOwnsColStarts(matrix) = owns_col_starts;

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
   int	my_id, i, num_procs;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   int  global_num_rows, global_num_cols, num_cols_offd;
   int  num_nonzeros_diag, num_nonzeros_offd;
   int  local_num_rows, local_num_cols;
   int  *row_starts;
   int  *col_starts;
   int  *col_map_offd;
   FILE *fp;
   int equal = 1;

   MPI_Comm_rank(comm,&my_id);
   MPI_Comm_size(comm,&num_procs);
   row_starts = hypre_CTAlloc(int, num_procs+1);
   col_starts = hypre_CTAlloc(int, num_procs+1);
   sprintf(new_file_d,"%s.D.%d",file_name,my_id);
   sprintf(new_file_o,"%s.O.%d",file_name,my_id);
   sprintf(new_file_info,"%s.INFO.%d",file_name,my_id);
   fp = fopen(new_file_info, "r");
   fscanf(fp, "%d", &global_num_rows);
   fscanf(fp, "%d", &global_num_cols);
   fscanf(fp, "%d", &num_cols_offd);
   for (i=0; i < num_procs; i++)
	   fscanf(fp, "%d %d", &row_starts[i], &col_starts[i]);
   row_starts[num_procs] = global_num_rows;
   col_starts[num_procs] = global_num_cols;
   col_map_offd = hypre_CTAlloc(int, num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
   	fscanf(fp, "%d", &col_map_offd[i]);
	
   fclose(fp);

   for (i=num_procs; i >= 0; i--)
	if (row_starts[i] != col_starts[i])
	{
		equal = 0;
		break;
	}

   if (equal)
   {
	hypre_TFree(col_starts);
	col_starts = row_starts;
   }
   
   diag = hypre_ReadCSRMatrix(new_file_d);
   local_num_rows = hypre_CSRMatrixNumRows(diag);
   local_num_cols = hypre_CSRMatrixNumCols(diag);
   num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(diag);

   if (num_cols_offd)
   {
	offd = hypre_ReadCSRMatrix(new_file_o);
        num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(offd);
   }
   else
	offd = hypre_CreateCSRMatrix(local_num_rows,0,0);

	
   matrix = hypre_CTAlloc(hypre_ParCSRMatrix, 1);
   
   hypre_ParCSRMatrixComm(matrix) = comm;
   hypre_ParCSRMatrixGlobalNumRows(matrix) = global_num_rows;
   hypre_ParCSRMatrixGlobalNumCols(matrix) = global_num_cols;
   hypre_ParCSRMatrixFirstRowIndex(matrix) = row_starts[my_id];
   hypre_ParCSRMatrixFirstColDiag(matrix) = col_starts[my_id];
   hypre_ParCSRMatrixRowStarts(matrix) = row_starts;
   hypre_ParCSRMatrixColStarts(matrix) = col_starts;
   hypre_ParCSRMatrixCommPkg(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(matrix) = 1;
   hypre_ParCSRMatrixOwnsRowStarts(matrix) = 1;
   hypre_ParCSRMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
   	hypre_ParCSRMatrixOwnsColStarts(matrix) = 0;

   hypre_ParCSRMatrixDiag(matrix) = diag;
   hypre_ParCSRMatrixOffd(matrix) = offd;
   if (num_cols_offd)
   	hypre_ParCSRMatrixColMapOffd(matrix) = col_map_offd;
   else
   	hypre_ParCSRMatrixColMapOffd(matrix) = NULL;

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
   int *row_starts = hypre_ParCSRMatrixRowStarts(matrix);
   int *col_starts = hypre_ParCSRMatrixColStarts(matrix);
   int	my_id, i, num_procs;
   char	  new_file_d[80], new_file_o[80], new_file_info[80];
   int  ierr = 0;
   FILE *fp;
   int num_cols_offd = 0;

   if (hypre_ParCSRMatrixOffd(matrix))
   	num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(matrix));

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);
   
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
   for (i=0; i < num_procs; i++)
   	fprintf(fp, "%d %d\n", row_starts[i], col_starts[i]);
   for (i=0; i < num_cols_offd; i++)
   	fprintf(fp, "%d\n", col_map_offd[i]);
   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GetLocalRangeParCSRMatrix
 * returns the row numbers of the rows stored on this processor.
 * "End" is actually the row number of the last row on this processor.
 *--------------------------------------------------------------------------*/

int 
hypre_GetLocalRangeParCSRMatrix( hypre_ParCSRMatrix *matrix,
                         int               *row_start,
                         int               *row_end,
                         int               *col_start,
                         int               *col_end )
{  
   int ierr=0;
   int my_id;

   MPI_Comm_rank( hypre_ParCSRMatrixComm(matrix), &my_id );

   *row_start = hypre_ParCSRMatrixRowStarts(matrix)[ my_id ];
   *row_end = hypre_ParCSRMatrixRowStarts(matrix)[ my_id + 1 ]-1;
   *col_start = hypre_ParCSRMatrixColStarts(matrix)[ my_id ];
   *col_end = hypre_ParCSRMatrixColStarts(matrix)[ my_id + 1 ]-1;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_GetRowParCSRMatrix
 * Returns global column indices and/or values for a given row in the global
 * matrix. Global row number is used, but the row must be stored locally or
 * an error is returned. This implementation copies from the two matrices that
 * store the local data, storing them in the hypre_ParCSRMatrix structure.
 * Only a single row can be accessed via this function at any one time; the
 * corresponding RestoreRow function must be called, to avoid bleeding memory,
 * and to be able to look at another row.
 * Either one of col_ind and values can be left null, and those values will
 * not be returned.
 * All indices are returned in 0-based indexing, no matter what is used under
 * the hood. EXCEPTION: currently this only works if the local CSR matrices
 * use 0-based indexing.
 * This code, semantics, implementation, etc., are all based on PETSc's MPI_AIJ
 * matrix code, adjusted for our data and software structures.
 * AJC 4/99.
 *--------------------------------------------------------------------------*/

int 
hypre_GetRowParCSRMatrix( hypre_ParCSRMatrix  *mat,
                         int                row,
                         int               *size,
                         int               **col_ind,
                         double            **values )
{  
   int ierr=0;
   int my_id;
   int row_start, row_end;
   hypre_CSRMatrix *Aa = (hypre_CSRMatrix *) hypre_ParCSRMatrixDiag(mat);
   hypre_CSRMatrix *Ba = (hypre_CSRMatrix *) hypre_ParCSRMatrixOffd(mat);
   

   if (hypre_ParCSRMatrixGetrowactive(mat)) return(-1);

   MPI_Comm_rank( hypre_ParCSRMatrixComm(mat), &my_id );

   hypre_ParCSRMatrixGetrowactive(mat) = 1;

   row_end = hypre_ParCSRMatrixRowStarts(mat)[ my_id + 1 ];
   row_start = hypre_ParCSRMatrixRowStarts(mat)[ my_id ];

   if (row < row_start || row >= row_end) return(-1);

   /* if buffer is not allocated and some information is requested,
      allocate buffer */
   if (!hypre_ParCSRMatrixRowvalues(mat) && ( col_ind || values )) 
   {
    /*
        allocate enough space to hold information from the longest row.
    */
     int     max = 1,tmp;
     int i;
     int     m = row_end-row_start;

     for ( i=0; i<m; i++ ) {
       tmp = hypre_CSRMatrixI(Aa)[i+1] - hypre_CSRMatrixI(Aa)[i] + 
             hypre_CSRMatrixI(Ba)[i+1] - hypre_CSRMatrixI(Ba)[i];
       if (max < tmp) { max = tmp; }
     }

     hypre_ParCSRMatrixRowvalues(mat) = (double *) hypre_CTAlloc
                             ( double, max ); 
     hypre_ParCSRMatrixRowindices(mat) = (int *) hypre_CTAlloc
                             ( int, max ); 
   }

   /* Copy from dual sequential matrices into buffer */
   {
   double     *vworkA, *vworkB, *v_p;
   int        i, ierr, *cworkA, *cworkB, 
              cstart = hypre_ParCSRMatrixFirstColDiag(mat);
   int        nztot, nzA, nzB, lrow=row-row_start;
   int        *cmap, *idx_p;

   nzA = hypre_CSRMatrixI(Aa)[lrow+1]-hypre_CSRMatrixI(Aa)[lrow];
   cworkA = &( hypre_CSRMatrixJ(Aa)[ hypre_CSRMatrixI(Aa)[lrow] ] );
   vworkA = &( hypre_CSRMatrixData(Aa)[ hypre_CSRMatrixI(Aa)[lrow] ] );

   nzB = hypre_CSRMatrixI(Ba)[lrow+1]-hypre_CSRMatrixI(Ba)[lrow];
   cworkB = &( hypre_CSRMatrixJ(Ba)[ hypre_CSRMatrixI(Ba)[lrow] ] );
   vworkB = &( hypre_CSRMatrixData(Ba)[ hypre_CSRMatrixI(Ba)[lrow] ] );

   nztot = nzA + nzB;

   cmap  = hypre_ParCSRMatrixColMapOffd(mat);

   if (values  || col_ind) {
     if (nztot) {
       /* Sort by increasing column numbers, assuming A and B already sorted */
       int imark = -1;
       if (values) {
         *values = v_p = hypre_ParCSRMatrixRowvalues(mat);
         for ( i=0; i<nzB; i++ ) {
           if (cmap[cworkB[i]] < cstart)   v_p[i] = vworkB[i];
           else break;
         }
         imark = i;
         for ( i=0; i<nzA; i++ )     v_p[imark+i] = vworkA[i];
         for ( i=imark; i<nzB; i++ ) v_p[nzA+i]   = vworkB[i];
       }
       if (col_ind) {
         *col_ind = idx_p = hypre_ParCSRMatrixRowindices(mat);
         if (imark > -1) {
           for ( i=0; i<imark; i++ ) {
             idx_p[i] = cmap[cworkB[i]];
           }
         } else {
           for ( i=0; i<nzB; i++ ) {
             if (cmap[cworkB[i]] < cstart)   idx_p[i] = cmap[cworkB[i]];
             else break;
           }
           imark = i;
         }
         for ( i=0; i<nzA; i++ )     idx_p[imark+i] = cstart + cworkA[i];
         for ( i=imark; i<nzB; i++ ) idx_p[nzA+i]   = cmap[cworkB[i]];
       } 
     } 
     else {
       if (col_ind) *col_ind = 0; 
       if (values)   *values   = 0;
     }
   }
   *size = nztot;

   } /* End of copy */


   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_RestoreRowParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_RestoreRowParCSRMatrix( hypre_ParCSRMatrix *matrix,
                         int                row,
                         int               *size,
                         int               **col_ind,
                         double            **values )
{  
   int ierr=0;

  if (!hypre_ParCSRMatrixGetrowactive(matrix)) {
    return( -1 );
  }

  hypre_ParCSRMatrixGetrowactive(matrix)=0;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixToParCSRMatrix:
 * generates a ParCSRMatrix distributed across the processors in comm
 * from a CSRMatrix on proc 0 .
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_CSRMatrixToParCSRMatrix( MPI_Comm comm, hypre_CSRMatrix *A,
			       int *row_starts,
                               int *col_starts )
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
 
   int i, j, ind;

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

   par_matrix = hypre_CreateParCSRMatrix (comm, global_num_rows,
	global_num_cols,row_starts,col_starts,0,0,0);

   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   col_starts = hypre_ParCSRMatrixColStarts(par_matrix);

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
		hypre_BuildCSRMatrixMPIDataType(local_num_nonzeros[i], 
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
	hypre_BuildCSRMatrixMPIDataType(num_nonzeros, 
			local_num_rows[my_id],
			hypre_CSRMatrixData(local_A),
			hypre_CSRMatrixI(local_A),
			hypre_CSRMatrixJ(local_A),
			csr_matrix_datatypes);
	MPI_Recv(MPI_BOTTOM,1,csr_matrix_datatypes[0],0,0,comm,&status0);
	MPI_Type_free(csr_matrix_datatypes);
   }

   first_col_diag = col_starts[my_id];
   last_col_diag = col_starts[my_id+1]-1;

   diag = hypre_ParCSRMatrixDiag(par_matrix);
   offd = hypre_ParCSRMatrixOffd(par_matrix);

   GenerateDiagAndOffd(local_A, par_matrix, first_col_diag, last_col_diag);


   hypre_DestroyCSRMatrix(local_A);
   hypre_TFree(local_num_rows);
   hypre_TFree(csr_matrix_datatypes);

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

   if (num_cols - num_cols_diag)
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
        offd_i = hypre_CTAlloc(int, num_rows+1);

	for (i=0; i < num_rows+1; i++)
	{
		diag_i[i] = a_i[i];
		offd_i[i] = 0;
 	}

	hypre_CSRMatrixNumCols(offd) = 0;
	hypre_CSRMatrixI(offd) = offd_i;
   }
   
   return ierr;
}

hypre_CSRMatrix *
hypre_MergeDiagAndOffd(hypre_ParCSRMatrix *par_matrix)
{
   hypre_CSRMatrix  *diag = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix  *offd = hypre_ParCSRMatrixOffd(par_matrix);
   hypre_CSRMatrix  *matrix;

   int 		num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   int 		first_col_diag = hypre_ParCSRMatrixFirstColDiag(par_matrix);
   int		*col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
   int 		num_rows = hypre_CSRMatrixNumRows(diag);
   int 		num_cols_offd = hypre_CSRMatrixNumCols(offd);

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
   int		count;

   if (!num_cols_offd)
   {
   	matrix = hypre_CreateCSRMatrix(num_rows,num_cols,diag_i[num_rows]);
	hypre_CSRMatrixOwnsData(matrix) = 0;
	hypre_CSRMatrixI(matrix) = diag_i;
	hypre_CSRMatrixJ(matrix) = diag_j;
	hypre_CSRMatrixData(matrix) = diag_data;
 	return matrix;
   }

   num_nonzeros = diag_i[num_rows] + offd_i[num_rows];

   matrix = hypre_CreateCSRMatrix(num_rows,num_cols,num_nonzeros);
   hypre_InitializeCSRMatrix(matrix);

   matrix_i = hypre_CSRMatrixI(matrix);
   matrix_j = hypre_CSRMatrixJ(matrix);
   matrix_data = hypre_CSRMatrixData(matrix);

   count = 0;
   matrix_i[0] = 0;
   for (i=0; i < num_rows; i++)
   {
	for (j=diag_i[i]; j < diag_i[i+1]; j++)
	{
		matrix_data[count] = diag_data[j];
		matrix_j[count++] = diag_j[j]+first_col_diag;
 	}
	for (j=offd_i[i]; j < offd_i[i+1]; j++)
	{
		matrix_data[count] = offd_data[j];
		matrix_j[count++] = col_map_offd[offd_j[j]];
 	}
	matrix_i[i+1] = count;
   }

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixToCSRMatrixAll:
 * generates a CSRMatrix from a ParCSRMatrix on all processors that have
 * parts of the ParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_ParCSRMatrixToCSRMatrixAll(hypre_ParCSRMatrix *par_matrix)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(par_matrix);
   hypre_CSRMatrix *matrix;
   hypre_CSRMatrix *local_matrix;
   int num_rows = hypre_ParCSRMatrixGlobalNumRows(par_matrix);
   int num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   int *row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
  
   int *matrix_i;
   int *matrix_j;
   double *matrix_data;
  
   int *local_matrix_i;
   int *local_matrix_j;
   double *local_matrix_data;
  
   int i, j;
   int local_num_rows;
   int local_num_nonzeros;
   int num_nonzeros;
   int num_data;
   int num_requests;
   int vec_len, offset;
   int start_index;
   int proc_id;
   int num_procs, my_id;
   int num_types;
   int *used_procs;

   MPI_Request *requests;
   MPI_Status *status;
   MPI_Datatype *data_type;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

   local_num_rows = row_starts[my_id+1] - row_starts[my_id];

/* if my_id contains no data, return NULL */
 
   if (!local_num_rows)
        return NULL;
 
   local_matrix = hypre_MergeDiagAndOffd(par_matrix);
   local_matrix_i = hypre_CSRMatrixI(local_matrix);
   local_matrix_j = hypre_CSRMatrixJ(local_matrix);
   local_matrix_data = hypre_CSRMatrixData(local_matrix);

   matrix_i = hypre_CTAlloc(int, num_rows+1);

/* determine procs that have vector data and store their ids in used_procs */

   num_types = 0;
   for (i=0; i < num_procs; i++)
	if (row_starts[i+1]-row_starts[i] && i-my_id)
		num_types++;
   num_requests = 2*num_types;

   used_procs = hypre_CTAlloc(int, num_types);
   j = 0;
   for (i=0; i < num_procs; i++)
	if (row_starts[i+1]-row_starts[i] && i-my_id)
		used_procs[j++] = i;

   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);
   data_type = hypre_CTAlloc(MPI_Datatype, num_types+1);

/* exchange contents of local_matrix_i */

   j = 0;
   for (i = 0; i < num_types; i++)
   {
	proc_id = used_procs[i];
	vec_len = row_starts[proc_id+1] - row_starts[proc_id];
	MPI_Irecv(&matrix_i[row_starts[proc_id]+1], vec_len, MPI_INT,
				proc_id, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
	proc_id = used_procs[i];
	MPI_Isend(&local_matrix_i[1], local_num_rows, MPI_INT,
				proc_id, 0, comm, &requests[j++]);
   }

   vec_len = row_starts[my_id+1] - row_starts[my_id];
   for (i=1; i <= vec_len; i++)
	matrix_i[row_starts[my_id]+i] = local_matrix_i[i];

   MPI_Waitall(num_requests, requests, status);

/* generate matrix_i from received data */

   offset = matrix_i[row_starts[1]];
   for (i=1; i < num_procs; i++)
   {
	for (j = row_starts[i]; j < row_starts[i+1]; j++)
		matrix_i[j+1] += offset;
	offset = matrix_i[row_starts[i+1]];
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CreateCSRMatrix(num_rows, num_cols, num_nonzeros);
   hypre_CSRMatrixI(matrix) = matrix_i;
   hypre_InitializeCSRMatrix(matrix);
   matrix_j = hypre_CSRMatrixJ(matrix);
   matrix_data = hypre_CSRMatrixData(matrix);

/* generate datatypes for further data exchange and exchange remaining
   data, i.e. column info and actual data */

   for (i = 0; i < num_types; i++)
   {
	proc_id = used_procs[i];
	start_index = matrix_i[row_starts[proc_id]];
	num_data = matrix_i[row_starts[proc_id+1]] - start_index; 
	hypre_BuildCSRJDataType(num_data,
			  &matrix_data[start_index],
			  &matrix_j[start_index],
			  &data_type[i]);
   }
   local_num_nonzeros = local_matrix_i[local_num_rows];
   hypre_BuildCSRJDataType(local_num_nonzeros,
		     local_matrix_data,
		     local_matrix_j,
		     &data_type[num_types]);
   j = 0;
   for (i=0; i < num_types; i++)
	MPI_Irecv(MPI_BOTTOM, 1, data_type[i],
			used_procs[i], 0, comm, &requests[j++]);
   for (i=0; i < num_types; i++)
	MPI_Isend(MPI_BOTTOM, 1, data_type[num_types],
			used_procs[i], 0, comm, &requests[j++]);

   start_index = matrix_i[row_starts[my_id]];
   for (i=0; i < local_num_nonzeros; i++)
   {
	matrix_j[start_index+i] = local_matrix_j[i];
	matrix_data[start_index+i] = local_matrix_data[i];
   }
   MPI_Waitall(num_requests, requests, status);

   for (i=0; i <= num_types; i++)
	MPI_Type_free(&data_type[i]);

   if (hypre_CSRMatrixOwnsData(local_matrix))
	hypre_DestroyCSRMatrix(local_matrix);
   else
	hypre_TFree(local_matrix);

   if (num_requests)
   {
	hypre_TFree(requests);
   	hypre_TFree(status);
   	hypre_TFree(used_procs);
   }
   hypre_TFree(data_type);

   return matrix;
}
    
/*--------------------------------------------------------------------------
 * hypre_CopyParCSRMatrix,
 * copies B to A,
 * if copy_data = 0, only the structure of A is copied to B
 * the routine does not check whether the dimensions of A and B are compatible
 *--------------------------------------------------------------------------*/

int 
hypre_CopyParCSRMatrix( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B, 
			int copy_data )
{
   int  ierr=0;
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   int *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   int *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);
   int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   int i;

   hypre_CopyCSRMatrix(A_diag, B_diag, copy_data);
   hypre_CopyCSRMatrix(A_offd, B_offd, copy_data);
   for (i = 0; i < num_cols_offd; i++)
	col_map_offd_B[i] = col_map_offd_A[i];
	
   return ierr;
}
