/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * IJMatrix_Parcsr interface
 *
 *****************************************************************************/
 
#include "IJ_matrix_vector.h"
#include "aux_parcsr_matrix.h"

int
hypre_SetIJMatrixLocalSizeParcsr(hypre_IJMatrix *matrix,
			   	 int     	 local_m,
			   	 int     	 local_n)
{
   int ierr = 0;
   hypre_AuxParCSRMatrix *aux_data;
   aux_data = hypre_CTAlloc(hypre_AuxParCSRMatrix, 1);
   hypre_AuxParCSRMatrixLocalNumRows(aux_data) = local_m;
   hypre_AuxParCSRMatrixLocalNumCols(aux_data) = local_n;
   hypre_IJMatrixTranslator(matrix) = aux_data;
   return ierr;
}

int 
hypre_NewIJMatrixParcsr(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   int global_m = hypre_IJMatrixM(matrix); 
   int global_n = hypre_IJMatrixN(matrix); 
   hypre_AuxParCSRMatrix *aux_data = hypre_IJMatrixTranslator(matrix);
   int local_m = hypre_IJMatrixLocalNumRows(aux_data);   
   int local_n = hypre_IJMatrixLocalNumCols(aux_data);   
   int ierr = 0;
   int *row_starts;
   int *col_starts;
   int num_cols_offd = 0;
   int num_nonzeros_diag = 0;
   int num_nonzeros_offd = 0;
   int num_procs, my_id;
   int equal;
   int i;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

   if (aux_data)
   {
      local_m = hypre_IJMatrixLocalNumRows(aux_data);   
      local_n = hypre_IJMatrixLocalNumCols(aux_data);   
      row_starts = hypre_CTAlloc(int,num_procs+1);
      col_starts = hypre_CTAlloc(int,num_procs+1);

      if (my_id == 0 & local_m == global_m)
      {
         row_starts[1] = local_m;
      }
      else
      {
         MPI_Allgather(&local_m,1,MPI_INT,&row_starts[1],1,MPI_INT,comm);
      }

      if (my_id == 0 & local_n == global_n)
      {
         col_starts[1] = local_n;
      }
      else
      {
         MPI_Allgather(&local_n,1,MPI_INT,&col_starts[1],1,MPI_INT,comm);
      }

      equal = 1;
      for (i=0; i < num_procs; i++)
      {
         row_starts[i+1] += row_starts[i];
         col_starts[i+1] += col_starts[i];
         if (row_starts[i+1] != col_starts[i+1])
	 equal = 0;
      }
      if (equal)
      {
         hypre_TFree(col_starts);
         col_starts = row_starts;
      }
   }
   else
   {
      row_starts = NULL;
      col_starts = NULL;
   }
   hypre_IJMatrixLocalStorage(matrix) = hypre_CreateParCSRMatrix(comm,global_m,
		global_n,row_starts, col_starts, num_cols_offd, 
		num_nonzeros_diag, num_nonzeros_offd);
   return ierr;
}

int
hypre_SetIJMatrixDiagRowSizesParcsr(hypre_IJMatrix *matrix,
			      	    int	     	   *sizes)
{
   int ierr = 0;
   int *diag_i;
   int num_rows;
   int i;
   hypre_ParCSRMatrix *par_matrix;
   hypre_AuxParCSRMatrix *aux_matrix;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(par_matrix));
   diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix));
   diag_i[0] = 0;
   for (i = 0; i < num_rows; i++)
   {
      diag_i[i+1] = diag_i[i] + sizes[i];
   }
   hypre_AuxParCSRMatrixDiagSize(aux_matrix) = diag_i[num_rows];
   return ierr;
}

int
hypre_SetIJMatrixOffDiagRowSizesParcsr(hypre_IJMatrix *matrix,
			               int            *sizes)
{
   int ierr = 0;
   int *offd_i;
   int num_rows;
   int i;
   hypre_ParCSRMatrix *par_matrix;
   hypre_AuxParCSRMatrix *aux_matrix;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixOffd(par_matrix));
   offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix));
   offd_i[0] = 0;
   for (i = 0; i < num_rows; i++)
   {
      offd_i[i+1] = offd_i[i] + sizes[i];
   }
   hypre_AuxParCSRMatrixOffdSize(aux_matrix) = offd_i[num_rows];

   return ierr;
}

int
hypre_SetIJMatrixTotalSizeParcsr(hypre_IJMatrix *matrix,
			   	 int     	 size)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   hypre_ParCSRMatrixNumNonzeros(par_matrix) = size;
   return ierr;
}

int
hypre_InitializeIJMatrixParcsr(hypre_IJMatrix *matrix)
{
   int ierr = 0;
   return hypre_IntializeParCSRMatrix(hypre_IJMatrixLocalStorage( matrix ));
}

int
hypre_SetIJMatrixBlockParcsr(hypre_IJMatrix *matrix,
		       	     int	     m,
		             int	     n,
		             int	    *rows,
		             int	    *cols,
		             double	    *coeffs)
{
   int ierr = 0;
   int i, in;
   for (i=0; i < m; i++)
   {
      in = i*n;
      hypre_InsertIJMatrixRowParcsr(matrix,n,rows[i],&cols[in],&coeffs[in]);
   }
   return ierr;
}

int
hypre_AddBlockToIJMatrixParcsr(hypre_IJMatrix *matrix,
		       	       int	       m,
		               int	       n,
		               int	      *rows,
		               int	      *cols,
		               double	      *coeffs)
{
   int ierr = 0;
   int i, in;
   for (i=0; i < m; i++)
   {
      in = i*n;
      hypre_AddIJMatrixRowParcsr(matrix,n,rows[i],&cols[in],&coeffs[in]);
   }
   return ierr;
}

int
hypre_InsertIJMatrixRowParcsr(hypre_IJMatrix *matrix,
		              int	      n,
		              int	      row,
		              int	     *indices,
		              double         *coeffs)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_data;
   int *row_starts;
   int *col_starts;
   MPI_Comm comm;
   int num_procs, my_id;
   int row_local;
   int indx_offd, indx_diag;
   int indx_diag_old;
   int col_0, col_n;
   int i, temp;
   int *diag_i, *offd_i;
   int *aux_diag_j, *aux_offd_j;
   double *aux_diag_data, *aux_offd_data;
   int *diag_j, *offd_j;
   double *diag_data, *offd_data;
   int diag_size, offd_size;
   int diag_space, offd_space;
   int check_diag_space = 0;
   int check_offd_space = 0;
   int *row_start_diag, *row_end_diag;
   int *row_start_offd, *row_end_offd;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   par_matrix = hypre_IJMatrixLocalStorage( matrix );
   aux_data = hypre_IJMatrixTranslator(matrix);
   diag_size = hypre_AuxParCSRMatrixDiagSize(aux_data);
   offd_size = hypre_AuxParCSRMatrixOffdSize(aux_data);
   col_n = hypre_ParCSRMatrixFirstColDiag(par_matrix);
   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
   col_0 = col_starts[my_id];
   col_n = col_starts[my_id+1]-1;
   if (row >= row_starts[my_id] && row < row_starts[my_id+1])
   {
      row_local = row - row_starts[my_id];
      if (diag_size >= 0)
      {
      	 indx_diag = hypre_AuxParCSRMatrixIndxDiag(aux_data);
      	 diag_space = diag_size - indx_diag;
 	 if (diag_space >= n) check_diag_space = 0;
      	 row_start_diag[row_local] = indx_diag;
	 aux_diag_j = hypre_AuxParCSRMatrixAuxDiagJ(aux_data);
	 aux_diag_data = hypre_AuxParCSRMatrixAuxDiagData(aux_data);
      }
      else 
      {
	 diag = hypre_ParCSRMatrixDiag(par_matrix);
	 diag_i = hypre_CSRMatrixI(diag);
	 indx_diag = diag_i[row_local];
	 aux_diag_j = hypre_CSRMatrixJ(diag);
	 aux_diag_data = hypre_CSRMatrixData(diag);
      }
      if (offd_size >= 0)
      {
      	 indx_offd = hypre_AuxParCSRMatrixIndxOffd(aux_data);
      	 offd_space = offd_size - indx_offd;
 	 if (offd_space >= n) check_offd_space = 0;
      	 row_start_offd[row_local] = indx_offd;
	 aux_offd_j = hypre_AuxParCSRMatrixAuxOffdJ(aux_data);
	 aux_offd_data = hypre_AuxParCSRMatrixAuxOffdData(aux_data);
      }
      else 
      {
	 offd = hypre_ParCSRMatrixOffd(par_matrix);
	 offd_i = hypre_CSRMatrixI(offd);
	 indx_offd = offd_i[row_local];
	 aux_offd_j = hypre_CSRMatrixJ(offd);
	 aux_offd_data = hypre_CSRMatrixData(offd);
      }
      if (check_offd_space || check_diag_space)
      {
         for (i=0; i < n; i++)
         {
   	    if (indices[i] < col_0 || indices[i] > col_n)
	    {
	       offd_space--;	
            }
	    else
	    {
	       diag_space--;	
            }
         }
      }
      if (offd_space <= 0)
      {
	 offd_j = hypre_CTAlloc(int,offd_size+n );
	 offd_data = hypre_CTAlloc(double,offd_size+n );
	 for (i=0; i < indx_offd; i++)
	 {
	    offd_j[i] = aux_offd_j[i];
	    offd_data[i] = aux_offd_data[i];
	 }
	 hypre_TFree(aux_offd_j);
	 hypre_TFree(aux_offd_data);
	 aux_offd_j = offd_j;
	 aux_offd_data = offd_data;
      }
      if (diag_space <= 0)
      {
	 diag_j = hypre_CTAlloc(int,diag_size+n );
	 diag_data = hypre_CTAlloc(double,diag_size+n );
	 for (i=0; i < indx_diag; i++)
	 {
	    diag_j[i] = aux_diag_j[i];
	    diag_data[i] = aux_diag_data[i];
	 }
	 hypre_TFree(aux_diag_j);
	 hypre_TFree(aux_diag_data);
	 aux_diag_j = diag_j;
	 aux_diag_data = diag_data;
      }

      indx_diag_old = indx_diag;	
      for (i=0; i < n; i++)
      {
   	 if (indices[i] < col_0 || indices[i] > col_n)
	 {
	    aux_offd_j[indx_offd] = indices[i];
	    aux_offd_data[indx_offd++] = coeffs[i];
	 }
	 else
	 {
	    aux_diag_j[indx_diag] = indices[i] - col_0;
	    aux_diag_data[indx_diag++] = coeffs[i];
	 } 
      } 
/* make sure first element is diagonal element */
      if (aux_diag_j[indx_diag_old] != row_local)
      {
         for (i=indx_diag_old; i < indx_diag; i++)
  	 {
	    if (aux_diag_j[i] == row_local)
	    {
		aux_diag_j[i] = aux_diag_j[indx_diag_old];
		aux_diag_j[indx_diag_old] = row_local;
		temp = aux_diag_data[indx_diag_old];
		aux_diag_data[indx_diag_old] = aux_diag_data[i];
		aux_diag_data[i] = temp;
		break;
	    }
  	 }
      }
      if (diag_size >= 0)
      {
         row_end_diag[row_local] = indx_diag;
      	 hypre_AuxParCSRMatrixIndxDiag(aux_data) = indx_diag;
      }
      if (offd_size >= 0)
      {
         row_end_offd[row_local] = indx_offd;
      	 hypre_AuxParCSRMatrixIndxOffd(aux_data) = indx_offd;
      }
   }
   return ierr;
}

int
hypre_AddIJMatrixRowParcsr(hypre_IJMatrix *matrix,
	                   int	           n,
		           int	           row,
		           int	          *indices,
		           double         *coeffs)
{
   int ierr = 0;

   return ierr;
}

int
hypre_AssembleIJMatrixParcsr(hypre_IJMatrix *matrix)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixLocalStorage(matrix);
   hypre_AuxParCSRMatrix *aux_data = hypre_IJMatrixTranslator(matrix);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(par_matrix);   
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(par_matrix); 
   int *diag_i = hypre_CSRMatrixI(diag);
   int *offd_i = hypre_CSRMatrixI(offd);
   int *diag_j;
   int *offd_j;
   double *diag_data;
   double *offd_data;
   int num_rows = hypre_CSRMatrixNumRows(diag);
   int j_indx, cnt, i, j;
   int num_cols_offd;
   int *col_map_offd;
   int diag_size = hypre_AuxParCSRMatrixDiagSize(aux_data);
   int *row_start_diag = hypre_AuxParCSRMatrixRowStartDiag(aux_data);
   int *row_end_diag = hypre_AuxParCSRMatrixRowEndDiag(aux_data);
   int *aux_diag_j = hypre_AuxParCSRMatrixAuxDiagJ(aux_data);
   double *aux_diag_data = hypre_AuxParCSRMatrixAuxDiagData(aux_data);
   int offd_size = hypre_AuxParCSRMatrixOffdSize(aux_data);
   int *row_start_offd = hypre_AuxParCSRMatrixRowStartOffd(aux_data);
   int *row_end_offd = hypre_AuxParCSRMatrixRowEndOffd(aux_data);
   int *aux_offd_j = hypre_AuxParCSRMatrixAuxOffdJ(aux_data);
   double *aux_offd_data = hypre_AuxParCSRMatrixAuxOffdData(aux_data);
  
   if (diag_size > 0)
   {
      diag_j = hypre_CTAlloc(int,diag_size);
      diag_data = hypre_CTAlloc(double,diag_size);
      j_indx = 0;
      for (i=0; i < num_rows; i++)
      {
	 diag_i[i] = j_indx;
	 for (j=row_start_diag[i]; j < row_end_diag[i]; j++)
	 {
	    diag_j[j_indx] = aux_diag_j[j];
	    diag_data[j_indx++] = aux_diag_data[j];
	 }
      }
      diag_i[num_rows] = diag_size;
      hypre_CSRMatrixJ(diag) = diag_j;      
      hypre_CSRMatrixData(diag) = diag_data;      
   }

   if (offd_size > 0)
   {
      offd_j = hypre_CTAlloc(int,offd_size);
      offd_data = hypre_CTAlloc(double,offd_size);
      j_indx = 0;
      for (i=0; i < num_rows; i++)
      {
	 offd_i[i] = j_indx;
	 for (j=row_start_offd[i]; j < row_end_offd[i]; j++)
	 {
	    offd_j[j_indx] = aux_offd_j[j];
	    offd_data[j_indx++] = aux_offd_data[j];
	 }
      }
      offd_i[num_rows] = offd_size;
/*  generate col_map_offd */
      qsort0(aux_offd_j,0,offd_size-1);
      num_cols_offd = 1;
      for (i=0; i < offd_size-1; i++)
      {
	if (aux_offd_j[i+1] > aux_offd_j[i])
		num_cols_offd++;
      }
      col_map_offd = hypre_CTAlloc(int,num_cols_offd);
      col_map_offd[0] = aux_offd_j[0];
      cnt = 0;
      for (i=1; i < offd_size; i++)
      {
	if (aux_offd_j[i] > col_map_offd[cnt])
  	{
	   cnt++;
	   col_map_offd[cnt] = aux_offd_j[i];
  	}
      }
      for (i=0; i < offd_size; i++)
      {
	offd_j[i] = binsearch(col_map_offd,offd_j[i],num_cols_offd);
      }
      hypre_CSRMatrixJ(offd) = offd_j;      
      hypre_CSRMatrixData(offd) = offd_data;  
      hypre_ParCSRMatrixColMapOffd(par_matrix) = col_map_offd;    
   }

   hypre_DestroyAuxParCSRMatrix(aux_data);
   return ierr;
}

int
hypre_DistributeIJMatrixParcsr(hypre_IJMatrix *matrix,
			       int	      *row_starts,
			       int	      *col_starts)
{
   int ierr = 0;
   hypre_ParCSRMatrix *old_matrix = hypre_IJMatrixLocalStorage(matrix);
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(old_matrix);
   par_matrix = hypre_CSRMatrixToParCSRMatrix(hypre_ParCSRMatrixComm(old_matrix)
		, diag, row_starts, col_starts);
   ierr = hypre_DestroyParCSRMatrix(old_matrix);
   hypre_IJMatrixLocalStorage(matrix) = par_matrix;
   return ierr;
}

int
hypre_ApplyIJMatrixParcsr(hypre_IJMatrix  *matrix,
		    	  hypre_ParVector *x,
		          hypre_ParVector *b)
{
   int ierr = 0;

   return ierr;
}

int
hypre_FreeIJMatrixParcsr(hypre_IJMatrix *matrix)
{
   return hypre_DestroyParCSRMatrix(hypre_IJMatrixLocalStorage(matrix));
}

