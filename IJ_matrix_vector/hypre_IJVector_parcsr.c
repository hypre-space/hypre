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
 * IJVector_Par interface
 *
 *****************************************************************************/
 
#include "IJ_matrix_vector.h"
#include "aux_parcsr_matrix.h"

/******************************************************************************
 *
 * hypre_SetIJVectorLocalSizePar
 *
 * sets local number of rows and number of columns of diagonal matrix on
 * current processor.
 *
 *****************************************************************************/

int
hypre_SetIJVectorLocalSizePar(hypre_IJVector *vector,
                             int             local_n)
{
   int ierr = 0;
   hypre_AuxParVector *aux_data;
   aux_data = hypre_IJVectorTranslator(vector);
   if (aux_data)
   {
      hypre_AuxParVectorLocalNumRows(aux_data) = local_n;
   }
   else
   {
      hypre_IJVectorTranslator(vector) = 
			hypre_CreateAuxParVector(local_n, NULL);
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_NewIJVectorPar
 *
 * creates ParVector if necessary,
 * generates arrays row_starts and col_starts using either previously
 * set data local_m and local_n (user defined) or generates them evenly
 * distributed if not previously defined by user.
 *
 *****************************************************************************/
int
hypre_NewIJVectorPar(hypre_IJVector *vector)
{
   MPI_Comm comm = hypre_IJVectorContext(vector);
   int global_n = hypre_IJVectorN(vector); 
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
   int local_n;   
   int ierr = 0;

   int *row_starts;
   int num_procs, my_id;
   int equal;
   int i;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

   if (aux_vector)
   {
      local_n = hypre_AuxParVectorLocalNumRows(aux_vector);
   }
   else
   {
      aux_vector = hypre_CreateAuxParVector(-1,-1,NULL);
      local_n = -1;
      hypre_IJVectorTranslator(matrix) = aux_vector;
   }

   if (local_n < 0)
   {
      row_starts = NULL;
   }
   else
   {
      row_starts = hypre_CTAlloc(int,num_procs+1);

      if (my_id == 0 && local_n == global_n)
      {
         row_starts[1] = local_n;
      }
      else
      {
         MPI_Allgather(&local_n,1,MPI_INT,&row_starts[1],1,MPI_INT,comm);
      }

   }

   hypre_IJVectorLocalStorage(matrix) = hypre_CreateParVector(comm,
		global_n,row_starts); 
   return ierr;
}

/******************************************************************************
 *
 * hypre_InitializeIJVectorPar
 *
 * initializes ParVector
 *
 *****************************************************************************/

int
hypre_InitializeIJVectorPar(hypre_IJVector *vector)
{
   int ierr = 0;
   hypre_ParVector *par_vector = hypre_IJVectorLocalStorage(vector);
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
   int local_num_rows = hypre_AuxParVectorLocalNumRows(aux_vector);
   int num_procs, my_id;
   MPI_Comm  comm = hypre_IJVectorContext(vector);

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);
   
   if (local_num_rows < 0)
      hypre_AuxParVectorLocalNumRows(aux_vector) = 
		hypre_ParVectorNumRows(par_vector);
   ierr  = hypre_InitializeAuxParVector(aux_vector);
   ierr += hypre_InitializeParVector(par_vector);

   return ierr;
}

/******************************************************************************
 *
 * hypre_InsertIJVectorRowsPar
 *
 * inserts a set of rows into an IJVector, currently it just uses
 * InsertIJVectorRowPar
 *
 *****************************************************************************/
int
hypre_InsertIJVectorRowsPar(hypre_IJVector *vector,
		            int	            n,
		            int	           *rows,
		            double         *vals)
{
   int ierr = 0;
   int i, in;
   for (i=0; i < m; i++)
   {
      in = i*n;
      hypre_InsertIJVectorRowPar(vector,n,rows[i],&vals[in]);
   }
   return ierr;
}
/******************************************************************************
 *
 * hypre_AddRowsToIJVectorPar
 *
 * adds a set of rows to an IJVector, currently it just uses
 * AddIJVectorRowParCSR
 *
 *****************************************************************************/

int
hypre_AddRowsToIJVectorPar(hypre_IJVector *vector,
		              int          n,
		              int         *rows,
		              double      *values )
{
   int ierr = 0;
   int i, in;
   for (i=0; i < n; i++)
   {
      in = i;
      hypre_AddIJVectorRowPar(vector,n,rows[i],&cols[in],&coeffs[in]);
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_AddIJVectorRowPar
 *
 * adds a row to an IJVector before assembly, 
 * 
 *****************************************************************************/
int
hypre_AddIJVectorRowPar(hypre_IJVector *vector,
	                int	        n,
		        int	        row,
		        int	       *indices,
		        double         *coeffs   )
{
   int ierr = 0;
   hypre_ParVector *par_vector;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParVector *aux_vector;
   int *row_starts;
   int *col_starts;
   MPI_Comm comm = hypre_IJMatrixContext(matrix);
   int num_procs, my_id;
   int row_local;
   int col_0, col_n;
   int i, temp;
   int *indx_diag, *indx_offd;
   int **aux_j;
   int *local_j;
   int *tmp_j, *tmp2_j;
   double **aux_data;
   double *local_data;
   double *tmp_data, *tmp2_data;
   int diag_space, offd_space;
   int *row_length, *row_space;
   int need_aux;
   int tmp_indx, indx;
   int size, old_size;
   int cnt, cnt_diag, cnt_offd, indx_0;
   int offd_indx, diag_indx;
   int *diag_i;
   int *diag_j;
   double *diag_data;
   int *offd_i;
   int *offd_j;
   double *offd_data;
   int *tmp_diag_i;
   int *tmp_diag_j;
   double *tmp_diag_data;
   int *tmp_offd_i;
   int *tmp_offd_j;
   double *tmp_offd_data;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   par_matrix = hypre_IJMatrixLocalStorage( matrix );
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   col_0 = col_starts[my_id];
   col_n = col_starts[my_id+1]-1;
   need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);

   if (row >= row_starts[my_id] && row < row_starts[my_id+1])
   {
      if (need_aux)
      {
         row_local = row - row_starts[my_id]; /* compute local row number */
         aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
         local_j = aux_j[row_local];
         local_data = aux_data[row_local];
	 tmp_j = hypre_CTAlloc(int,n);
	 tmp_data = hypre_CTAlloc(double,n);
	 tmp_indx = 0;
         for (i=0; i < n; i++)
	 {
	    if (indices[i] == row)
	       local_data[0] += coeffs[i];
	    else
	    {
	       tmp_j[tmp_indx] = indices[i];
	       tmp_data[tmp_indx++] = coeffs[i];
	    }
	 }
	 qsort1(tmp_j,tmp_data,0,tmp_indx-1);
	 indx = 0;
	 size = 0;
	 for (i=1; i < row_length[row_local]; i++)
	 {
	    while (local_j[i] > tmp_j[indx])
	    {
	       size++;
	       indx++;
	    }
	    if (local_j[i] == tmp_j[indx])
	    {
	       size++;
	       indx++;
	    }
	 }
	 size += tmp_indx-indx;
	    
         old_size = row_length[row_local];   
         row_length[row_local] = size;
         
         if ( row_space[row_local] < size)
         {
   	    tmp2_j = hypre_CTAlloc(int,size);
   	    tmp2_data = hypre_CTAlloc(double,size);
	    for (i=0; i < old_size; i++)
	    {
	       tmp2_j[i] = local_j[i];
	       tmp2_data[i] = local_data[i];
	    }
   	    hypre_TFree(local_j);
   	    hypre_TFree(local_data);
	    local_j = tmp2_j;
	    local_data = tmp2_data;
            row_space[row_local] = n;
         }
        /* merge local and tmp into local */

         indx = 0; 
	 cnt = row_length[row_local];

	 for (i=1; i < old_size; i++)
	 {
	    while (local_j[i] > tmp_j[indx])
	    {
	       local_j[cnt] = tmp_j[indx];
	       local_data[cnt++] = tmp_data[indx++];
	    }
	    if (local_j[i] == tmp_j[indx])
	    {
	       local_j[i] += tmp_j[indx];
	       local_data[i] += tmp_data[indx++];
	    }
	 }
         for (i=indx; i < tmp_indx; i++)
         {
   	    local_j[cnt] = tmp_j[i];
   	    local_data[cnt++] = tmp_data[i];
         }
   
      /* sort data according to column indices, except for first element */

         qsort1(local_j,local_data,1,n-1);
	 hypre_TFree(tmp_j); 
	 hypre_TFree(tmp_data); 
      }
      else /* insert immediately into data into ParCSRMatrix structure */
      {
	 offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	 diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
         diag = hypre_ParCSRMatrixDiag(par_matrix);
         diag_i = hypre_CSRMatrixI(diag);
         diag_j = hypre_CSRMatrixJ(diag);
         diag_data = hypre_CSRMatrixData(diag);
         offd = hypre_ParCSRMatrixOffd(par_matrix);
         offd_i = hypre_CSRMatrixI(offd);
         offd_j = hypre_CSRMatrixJ(offd);
         offd_data = hypre_CSRMatrixData(offd);

	 indx_0 = diag_i[row_local];
	 diag_indx = indx_0+1;
	 
	 tmp_diag_j = hypre_CTAlloc(int,n);
	 tmp_diag_data = hypre_CTAlloc(double,n);
	 cnt_diag = 0;
	 tmp_offd_j = hypre_CTAlloc(int,n);
	 tmp_offd_data = hypre_CTAlloc(double,n);
	 cnt_offd = 0;
  	 for (i=0; i < n; i++)
	 {
	    if (indices[i] < col_0 || indices[i] > col_n)/* insert into offd */	
	    {
	       tmp_offd_j[cnt_offd] = indices[i];
	       tmp_offd_data[cnt_offd++] = coeffs[i];
	    }
	    else if (indices[i] == row) /* diagonal element */
	    {
	       diag_j[indx_0] = indices[i] - col_0;
	       diag_data[indx_0] += coeffs[i];
	    }
	    else  /* insert into diag */
	    {
	       tmp_diag_j[cnt_diag] = indices[i] - col_0;
	       tmp_diag_data[cnt_diag++] = coeffs[i];
	    }
	 }
	 qsort1(tmp_diag_j,tmp_diag_data,0,cnt_diag-1);
	 qsort1(tmp_offd_j,tmp_offd_data,0,cnt_offd-1);

         diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
	 cnt = diag_indx;
	 indx = 0;
	 for (i=diag_i[row_local]+1; i < diag_indx; i++)
	 {
	    while (diag_j[i] > tmp_diag_j[indx])
	    {
	       diag_j[cnt] = tmp_diag_j[indx];
	       diag_data[cnt++] = tmp_diag_data[indx++];
	    }
	    if (diag_j[i] == tmp_diag_j[indx])
	    {
	       diag_j[i] += tmp_diag_j[indx];
	       diag_data[i] += tmp_diag_data[indx++];
	    }
	 }
         for (i=indx; i < cnt_diag; i++)
         {
   	    diag_j[cnt] = tmp_diag_j[i];
   	    diag_data[cnt++] = tmp_diag_data[i];
         }
   
      /* sort data according to column indices, except for first element */

         qsort1(diag_j,diag_data,1,cnt-1);
	 hypre_TFree(tmp_diag_j); 
	 hypre_TFree(tmp_diag_data); 

	 hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt;

         offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	 cnt = offd_indx;
	 indx = 0;
	 for (i=offd_i[row_local]+1; i < offd_indx; i++)
	 {
	    while (offd_j[i] > tmp_offd_j[indx])
	    {
	       offd_j[cnt] = tmp_offd_j[indx];
	       offd_data[cnt++] = tmp_offd_data[indx++];
	    }
	    if (offd_j[i] == tmp_offd_j[indx])
	    {
	       offd_j[i] += tmp_offd_j[indx];
	       offd_data[i] += tmp_offd_data[indx++];
	    }
	 }
         for (i=indx; i < cnt_offd; i++)
         {
   	    offd_j[cnt] = tmp_offd_j[i];
   	    offd_data[cnt++] = tmp_offd_data[i];
         }
   
      /* sort data according to column indices, except for first element */

         qsort1(offd_j,offd_data,1,cnt-1);
	 hypre_TFree(tmp_offd_j); 
	 hypre_TFree(tmp_offd_data); 

	 hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt;
      }
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_AssembleIJVectorPar
 *
 * assembles IJVector from AuxParVector auxiliary structure
 *****************************************************************************/
int
hypre_AssembleIJVectorPar(hypre_IJVector *vector)
{
   int ierr = 0;
   MPI_Comm comm = hypre_IJVectorContext(vector);
   hypre_ParCSRVector *par_vector = hypre_IJVectorLocalStorage(vector);
   hypre_AuxParCSRMatrix *aux_vector = hypre_IJVectorTranslator(vector);
   int *diag_i;
   int *offd_i;
   int *diag_j;
   int *offd_j;
   double *diag_data;
   double *offd_data;
   int *row_starts = hypre_ParVectorRowStarts(par_matrix);
   int cnt, i;
   double **aux_data;
   int *indx_diag;
   int *indx_offd;
   int need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
   int my_id, num_procs;
   int num_rows;
   int i_diag, i_offd;
   int *local_j;
   double *local_data;
   int col_0, col_n;
   int nnz_offd;
   int *aux_offd_j;

   MPI_Comm_size(comm, &num_procs); 
   MPI_Comm_rank(comm, &my_id);
   num_rows = row_starts[my_id+1] - row_starts[my_id]; 
/* move data into ParCSRMatrix if not there already */ 
   if (need_aux)
   {
      col_0 = col_starts[my_id];
      col_n = col_starts[my_id+1]-1;
      i_diag = 0;
      i_offd = 0;
      for (i=0; i < num_rows; i++)
      {
	 local_j = aux_j[i];
	 local_data = aux_data[i];
	 for (j=0; j < row_length[i]; j++)
	 {
	    if (local_j[j] < col_0 || local_j[j] > col_n)
	       i_offd++;
	    else
	       i_diag++;
	 }
	 diag_i[i] = i_diag;
	 offd_i[i] = i_offd;
      }
      diag_j = hypre_CTAlloc(int,i_diag);
      diag_data = hypre_CTAlloc(double,i_diag);
      offd_j = hypre_CTAlloc(int,i_offd);
      offd_data = hypre_CTAlloc(double,i_offd);
      i_diag = 0;
      i_offd = 0;
      for (i=0; i < num_rows; i++)
      {
	 local_j = aux_j[i];
	 local_data = aux_data[i];
	 for (j=0; j < row_length[i]; j++)
	 {
	    if (local_j[j] < col_0 || local_j[j] > col_n)
	    {
	       offd_j[i_offd] = local_j[j];
	       offd_data[i_offd++] = local_data[j];
	    }
	    else
	    {
	       diag_j[i_diag] = local_j[j];
	       diag_data[i_diag++] = local_data[j];
	    }
	 }
      }
      hypre_CSRMatrixJ(diag) = diag_j;      
      hypre_CSRMatrixData(diag) = diag_data;      
      hypre_CSRMatrixNumNonzeros(diag) = diag_i[num_rows];      
      hypre_CSRMatrixJ(offd) = offd_j;      
      hypre_CSRMatrixData(offd) = offd_data;      
      hypre_CSRMatrixNumNonzeros(offd) = offd_i[num_rows];      
   }

/*  generate col_map_offd */
   nnz_offd = offd_i[num_rows];
   aux_offd_j = hypre_CTAlloc(int, nnz_offd);
   for (i=0; i < nnz_offd; i++)
      aux_offd_j[i] = offd_j[i];
   qsort0(aux_offd_j,0,nnz_offd-1);
   num_cols_offd = 1;
   for (i=0; i < nnz_offd-1; i++)
   {
      if (aux_offd_j[i+1] > aux_offd_j[i])
      num_cols_offd++;
   }
   col_map_offd = hypre_CTAlloc(int,num_cols_offd);
   col_map_offd[0] = aux_offd_j[0];
   cnt = 0;
   for (i=1; i < nnz_offd; i++)
   {
      if (aux_offd_j[i] > col_map_offd[cnt])
      {
	 cnt++;
	 col_map_offd[cnt] = aux_offd_j[i];
      }
   }
   for (i=0; i < nnz_offd; i++)
   {
      offd_j[i] = hypre_BinarySearch(col_map_offd,offd_j[i],num_cols_offd);
   }
   hypre_ParCSRMatrixColMapOffd(par_matrix) = col_map_offd;    
   hypre_CSRMatrixNumCols(offd) = num_cols_offd;    

   hypre_DestroyAuxParCSRMatrix(aux_matrix);
   hypre_TFree(aux_offd_j);

   return ierr;
}

/******************************************************************************
 *
 * hypre_DistributeIJVectorPar
 *
 * takes an IJVector generated for one processor and distributes it
 * across many processors according to row_starts,
 * if row_starts is NULL, it distributes them evenly.
 *
 *****************************************************************************/
int
hypre_DistributeIJVectorPar(hypre_IJVector *vector,
			    int	           *row_starts)
{
   int ierr = 0;

   hypre_ParVector *old_vector = hypre_IJVectorLocalStorage(vector);
   hypre_ParVector *par_vector;
   hypre_Vector *diag = hypre_ParVectorDiag(old_vector);
   par_vector = hypre_VectorToParVector(hypre_ParVectorComm(old_vector)
		, diag, row_starts);
   ierr = hypre_DestroyParVector(old_vector);
   hypre_IJVectorLocalStorage(vector) = par_vector;

   return ierr;
}

/******************************************************************************
 *
 * hypre_FreeIJVectorPar
 *
 * frees ParVector local storage of an IJVector 
 *
 *****************************************************************************/
int
hypre_FreeIJVectorPar(hypre_IJVector *vector)
{
   return hypre_DestroyParVector(hypre_IJVectorLocalStorage(vector));
}

