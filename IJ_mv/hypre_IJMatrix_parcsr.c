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

/******************************************************************************
 *
 * hypre_SetIJMatrixLocalSizeParcsr
 *
 * sets local number of rows and number of columns of diagonal matrix on
 * current processor.
 *
 *****************************************************************************/

int
hypre_SetIJMatrixLocalSizeParcsr(hypre_IJMatrix *matrix,
			   	 int     	 local_m,
			   	 int     	 local_n)
{
   int ierr = 0;
   hypre_AuxParCSRMatrix *aux_data;
   aux_data = hypre_IJMatrixTranslator(matrix);
   if (aux_data)
   {
      hypre_AuxParCSRMatrixLocalNumRows(aux_data) = local_m;
      hypre_AuxParCSRMatrixLocalNumCols(aux_data) = local_n;
   }
   else
   {
      hypre_IJMatrixTranslator(matrix) = 
			hypre_CreateAuxParCSRMatrix(local_m,local_n,-1,-1);
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_NewIJMatrixParcsr
 *
 * creates AuxParCSRMatrix and ParCSRMatrix if necessary,
 * generates arrays row_starts and col_starts using either previously
 * set data local_m and local_n (user defined) or generates them evenly
 * distributed if not previously defined by user.
 *
 *****************************************************************************/
int
hypre_NewIJMatrixParcsr(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   int global_m = hypre_IJMatrixM(matrix); 
   int global_n = hypre_IJMatrixN(matrix); 
   hypre_AuxParCSRMatrix *aux_data = hypre_IJMatrixTranslator(matrix);
   int local_m;   
   int local_n;   
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
   }
   else
   {
      aux_data = hypre_CreateAuxParCSRMatrix(-1,-1,-1,-1);
      local_m = -1;
      local_n = -1;
      hypre_IJMatrixTranslator(matrix) = aux_data;
   }

   if (local_m < 0)
   {
      row_starts = NULL;
   }
   else
   {
      row_starts = hypre_CTAlloc(int,num_procs+1);

      if (my_id == 0 && local_m == global_m)
      {
         row_starts[1] = local_m;
      }
      else
      {
         MPI_Allgather(&local_m,1,MPI_INT,&row_starts[1],1,MPI_INT,comm);
      }

   }
   if (local_n < 0)
   {
      col_starts = NULL;
   }
   else
   {
      col_starts = hypre_CTAlloc(int,num_procs+1);

      if (my_id == 0 && local_n == global_n)
      {
         col_starts[1] = local_n;
      }
      else
      {
         MPI_Allgather(&local_n,1,MPI_INT,&col_starts[1],1,MPI_INT,comm);
      }
   }

   if (row_starts && col_starts)
   {
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

   hypre_IJMatrixLocalStorage(matrix) = hypre_CreateParCSRMatrix(comm,global_m,
		global_n,row_starts, col_starts, num_cols_offd, 
		num_nonzeros_diag, num_nonzeros_offd);
   return ierr;
}

/******************************************************************************
 *
 * hypre_SetIJMatrixDiagRowSizesParcsr
 *
 * This routine assumes that sizes contains the correct sizes for Diag , 
 * no estimates 
 * it sets diag_i , and makes the use of row_start_diag, row_end_diag,
 * aux_diag_j, aux_diag_data inside AuxParCSRMatrix unnecessary.
 * to indicate this , diag_size is set to be -2.
 *
 *****************************************************************************/
int
hypre_SetIJMatrixDiagRowSizesParcsr(hypre_IJMatrix *matrix,
			      	    int	           *sizes)
{
   int ierr = 0;
   int *diag_i;
   int local_num_rows;
   int i;
   hypre_ParCSRMatrix *par_matrix;
   hypre_AuxParCSRMatrix *aux_matrix;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   if (aux_matrix)
      local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(aux_matrix);
   else
      return -1;
   if (par_matrix)
   {
      diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix));
      if (!diag_i)
      {
          diag_i = hypre_CTAlloc(int,local_num_rows+1);
          hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix)) = diag_i;
      }
   }
   else
   {
      ierr = hypre_NewIJMatrixParcsr(matrix);
      par_matrix = hypre_IJMatrixLocalStorage(matrix);
      diag_i = hypre_CTAlloc(int, local_num_rows+1);
      hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix)) = diag_i;
   }
   diag_i[0] = 0;
   for (i = 0; i < local_num_rows; i++)
      diag_i[i+1] = diag_i[i] + sizes[i];
   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(par_matrix)) 
		= diag_i[local_num_rows];
   hypre_AuxParCSRMatrixDiagSize(aux_matrix) = -2;
   return ierr;
}

/******************************************************************************
 *
 * hypre_SetIJMatrixOffDiagRowSizesParcsr
 *
 * This routine assumes that sizes contains the correct sizes for Offd , 
 * no estimates 
 * it sets offd_i , and makes the use of row_start_offd, row_end_offd,
 * aux_offd_j, aux_offd_data inside AuxParCSRMatrix unnecessary.
 * to indicate this , offd_size is set to be -2.
 *
 *****************************************************************************/
int
hypre_SetIJMatrixOffDiagRowSizesParcsr(hypre_IJMatrix *matrix,
			      	       int            *sizes)
{
   int ierr = 0;
   int *offd_i;
   int local_num_rows;
   int i;
   hypre_ParCSRMatrix *par_matrix;
   hypre_AuxParCSRMatrix *aux_matrix;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   if (aux_matrix)
      local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(aux_matrix);
   else
      return -1;
   if (par_matrix)
   {
      offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix));
      if (!offd_i)
      {
          offd_i = hypre_CTAlloc(int,local_num_rows+1);
          hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix)) = offd_i;
      }
   }
   else
   {
      ierr = hypre_NewIJMatrixParcsr(matrix);
      par_matrix = hypre_IJMatrixLocalStorage(matrix);
      offd_i = hypre_CTAlloc(int, local_num_rows+1);
      hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix)) = offd_i;
   }
   offd_i[0] = 0;
   for (i = 0; i < local_num_rows; i++)
      offd_i[i+1] = offd_i[i] + sizes[i];
   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(par_matrix) )
		= offd_i[local_num_rows];
   hypre_AuxParCSRMatrixOffdSize(aux_matrix) = -2;
   return ierr;
}

/******************************************************************************
 *
 * hypre_InitializeIJMatrixParcsr
 *
 * initializes AuxParCSRMatrix and ParCSRMatrix as necessary
 *
 *****************************************************************************/

int
hypre_InitializeIJMatrixParcsr(hypre_IJMatrix *matrix)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixLocalStorage(matrix);
   hypre_AuxParCSRMatrix *aux_data = hypre_IJMatrixTranslator(matrix);
   int local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(aux_data);
   int local_num_cols = hypre_AuxParCSRMatrixLocalNumCols(aux_data);
   int diag_size = hypre_AuxParCSRMatrixDiagSize(aux_data);
   int offd_size = hypre_AuxParCSRMatrixOffdSize(aux_data);
   int num_nonzeros = hypre_ParCSRMatrixNumNonzeros(par_matrix);
   int local_nnz;
   int num_procs, my_id;
   MPI_Comm  comm = hypre_IJMatrixComm(matrix);
   int global_num_rows = hypre_IJMatrixM(matrix);

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);
   
   local_nnz = (num_nonzeros/global_num_rows+1)*local_num_rows;
   if (local_num_rows < 0)
      hypre_AuxParCSRMatrixLocalNumRows(aux_data) = 
		hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(par_matrix));
   if (local_num_cols < 0)
      hypre_AuxParCSRMatrixLocalNumCols(aux_data) = 
		hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(par_matrix));
   if (diag_size == -1)
   {
      if (num_nonzeros > 0)
         hypre_AuxParCSRMatrixDiagSize(aux_data) = local_nnz;
      else
         hypre_AuxParCSRMatrixDiagSize(aux_data) = local_num_rows*20;
   }
   if (offd_size == -1)
   {
      if (num_nonzeros > 0)
         hypre_AuxParCSRMatrixOffdSize(aux_data) = local_nnz/4; 
      else  
         hypre_AuxParCSRMatrixOffdSize(aux_data) = local_num_rows*5; 
   }   
   ierr = hypre_IntializeAuxParCSRMatrix(aux_data);
   ierr += hypre_IntializeParCSRMatrix(par_matrix);
   return ierr;
}

/******************************************************************************
 *
 * hypre_SetIJMatrixParcsr
 *
 * inserts a block of values into an IJMatrix, currently it just uses
 * SetIJMatrixRowParcsr
 *
 *****************************************************************************/
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
/******************************************************************************
 *
 * hypre_AddIJMatrixParcsr
 *
 * adds a block of values to an IJMatrix, currently it just uses
 * AddIJMatrixRowParcsr
 *
 *****************************************************************************/

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

/******************************************************************************
 *
 * hypre_InsertIJMatrixRowParcsr
 *
 * inserts a row into an IJMatrix, 
 * if diag_i and offd_i are known, those values are inserted directly
 * into the ParCSRMatrix,
 * if they are not known, an auxiliary structure, AuxParCSRMatrix is used
 *
 *****************************************************************************/
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
      row_local = row - row_starts[my_id]; /* compute local row number */
      if (diag_size >= 0)  /* data are written to aux_data */
      {
      	 indx_diag = hypre_AuxParCSRMatrixIndxDiag(aux_data);
      	 diag_space = diag_size - indx_diag;
      	 row_start_diag[row_local] = indx_diag;
	 aux_diag_j = hypre_AuxParCSRMatrixAuxDiagJ(aux_data);
	 aux_diag_data = hypre_AuxParCSRMatrixAuxDiagData(aux_data);
      }
      else  /* data are written to par_matrix */
      {
	 diag = hypre_ParCSRMatrixDiag(par_matrix);
	 diag_i = hypre_CSRMatrixI(diag);
	 indx_diag = diag_i[row_local];
	 aux_diag_j = hypre_CSRMatrixJ(diag);
	 aux_diag_data = hypre_CSRMatrixData(diag);
      	 diag_space = 1;
      }

    /* check if there is enough space left in aux_diag_j(data) */
      if (diag_space >= n) check_diag_space = 0;

      if (offd_size >= 0)  /* data are written to aux_data */
      {
      	 indx_offd = hypre_AuxParCSRMatrixIndxOffd(aux_data);
      	 offd_space = offd_size - indx_offd;
      	 row_start_offd[row_local] = indx_offd;
	 aux_offd_j = hypre_AuxParCSRMatrixAuxOffdJ(aux_data);
	 aux_offd_data = hypre_AuxParCSRMatrixAuxOffdData(aux_data);
      }
      else  /* data are written to par_matrix */
      {
	 offd = hypre_ParCSRMatrixOffd(par_matrix);
	 offd_i = hypre_CSRMatrixI(offd);
	 indx_offd = offd_i[row_local];
	 aux_offd_j = hypre_CSRMatrixJ(offd);
	 aux_offd_data = hypre_CSRMatrixData(offd);
      	 offd_space = 1;
      }
    /* check if there is enough space left in aux_offd_j(data) */
      if (offd_space >= n) check_offd_space = 0;

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

   /* allocate more space if necessary */

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
/* make sure first element is diagonal element, if not, find it and
   exchange it with first element */
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
      if (diag_size >= 0) /* save indx_diag */
      {
         row_end_diag[row_local] = indx_diag;
      	 hypre_AuxParCSRMatrixIndxDiag(aux_data) = indx_diag;
      }
      if (offd_size >= 0) /* save indx_offd */
      {
         row_end_offd[row_local] = indx_offd;
      	 hypre_AuxParCSRMatrixIndxOffd(aux_data) = indx_offd;
      }
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_AddIJMatrixRowParcsr
 *
 * adds a row to an IJMatrix, 
 * NOT IMPLEMENTED YET!
 *****************************************************************************/
int
hypre_AddIJMatrixRowParcsr(hypre_IJMatrix *matrix,
	                   int	           n,
		           int	           row,
		           int	          *indices,
		           double         *coeffs)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixLocalStorage(matrix);
/*   not implemented yet */

   return ierr;
}

/******************************************************************************
 *
 * hypre_AssembleIJMatrixParcsr
 *
 * assembles IJMAtrix from AuxParCSRMatrix auxiliary structure
 *****************************************************************************/
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
   int *offd_j = hypre_CSRMatrixJ(offd);
   double *diag_data;
   double *offd_data;
   int num_rows = hypre_CSRMatrixNumRows(diag);
   int j_indx, cnt, i, j;
   int num_cols_offd;
   int *col_map_offd;
   int diag_size = hypre_AuxParCSRMatrixDiagSize(aux_data);
   int indx_diag = hypre_AuxParCSRMatrixIndxDiag(aux_data);
   int *row_start_diag = hypre_AuxParCSRMatrixRowStartDiag(aux_data);
   int *row_end_diag = hypre_AuxParCSRMatrixRowEndDiag(aux_data);
   int *aux_diag_j = hypre_AuxParCSRMatrixAuxDiagJ(aux_data);
   double *aux_diag_data = hypre_AuxParCSRMatrixAuxDiagData(aux_data);
   int offd_size = hypre_AuxParCSRMatrixOffdSize(aux_data);
   int indx_offd = hypre_AuxParCSRMatrixIndxOffd(aux_data);
   int *row_start_offd = hypre_AuxParCSRMatrixRowStartOffd(aux_data);
   int *row_end_offd = hypre_AuxParCSRMatrixRowEndOffd(aux_data);
   int *aux_offd_j = hypre_AuxParCSRMatrixAuxOffdJ(aux_data);
   double *aux_offd_data = hypre_AuxParCSRMatrixAuxOffdData(aux_data);
 
/* move data into ParCSRMatrix if not there already */ 
   if (diag_size > 0)
   {
      diag_j = hypre_CTAlloc(int,indx_diag);
      diag_data = hypre_CTAlloc(double,indx_diag);
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
      diag_i[num_rows] = indx_diag;
      hypre_CSRMatrixJ(diag) = diag_j;      
      hypre_CSRMatrixData(diag) = diag_data;      
   }

   if (offd_size > 0)
   {
      offd_j = hypre_CTAlloc(int,indx_offd);
      offd_data = hypre_CTAlloc(double,indx_offd);
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
      offd_i[num_rows] = indx_offd;
      hypre_CSRMatrixJ(offd) = offd_j;      
      hypre_CSRMatrixData(offd) = offd_data;  
   }
   else
   {
      indx_offd = offd_i[num_rows];
      aux_offd_j = hypre_CTAlloc(int, indx_offd);
      for (i=0; i < offd_size; i++)
         aux_offd_j[i] = offd_j[i];
   }


/*  generate col_map_offd */
   qsort0(aux_offd_j,0,indx_offd-1);
   num_cols_offd = 1;
   for (i=0; i < indx_offd-1; i++)
   {
      if (aux_offd_j[i+1] > aux_offd_j[i])
      num_cols_offd++;
   }
   col_map_offd = hypre_CTAlloc(int,num_cols_offd);
   col_map_offd[0] = aux_offd_j[0];
   cnt = 0;
   for (i=1; i < indx_offd; i++)
   {
      if (aux_offd_j[i] > col_map_offd[cnt])
      {
	 cnt++;
	 col_map_offd[cnt] = aux_offd_j[i];
      }
   }
   for (i=0; i < indx_offd; i++)
   {
      offd_j[i] = binsearch(col_map_offd,offd_j[i],num_cols_offd);
   }
   hypre_ParCSRMatrixColMapOffd(par_matrix) = col_map_offd;    
   hypre_CSRMatrixNumCols(offd) = num_cols_offd;    

   hypre_DestroyAuxParCSRMatrix(aux_data);
   if (offd_size == -2)
      hypre_TFree(aux_offd_j);

   return ierr;
}

/******************************************************************************
 *
 * hypre_DistributeIJMatrixParcsr
 *
 * takes an IJMatrix generated for one processor and distributes it
 * across many processors according to row_starts and col_starts,
 * if row_starts and/or col_starts NULL, it distributes them evenly.
 *
 *****************************************************************************/
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

/******************************************************************************
 *
 * hypre_ApplyIJMatrixParcsr
 *
 * NOT IMPLEMENTED YET
 *
 *****************************************************************************/
int
hypre_ApplyIJMatrixParcsr(hypre_IJMatrix  *matrix,
		    	  hypre_ParVector *x,
		          hypre_ParVector *b)
{
   int ierr = 0;

   return ierr;
}

/******************************************************************************
 *
 * hypre_FreeIJMatrixParcsr
 *
 * frees an IJMatrix
 *
 *****************************************************************************/
int
hypre_FreeIJMatrixParcsr(hypre_IJMatrix *matrix)
{
   return hypre_DestroyParCSRMatrix(hypre_IJMatrixLocalStorage(matrix));
}

/******************************************************************************
 *
 * hypre_SetIJMatrixTotalSizeParcsr
 *
 * sets the total number of nonzeros of matrix, can be somewhat useful
 * for storage estimates
 *
 *****************************************************************************/
int
hypre_SetIJMatrixTotalSizeParcsr(hypre_IJMatrix *matrix,
			   	 int     	 size)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   if (!par_matrix)
   {
      ierr = hypre_NewIJMatrixParcsr(matrix);
      par_matrix = hypre_IJMatrixLocalStorage(matrix);
   }
   hypre_ParCSRMatrixNumNonzeros(par_matrix) = size;
   return ierr;
}

