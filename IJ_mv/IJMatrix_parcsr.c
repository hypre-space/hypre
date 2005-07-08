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
 * IJMatrix_ParCSR interface
 *
 *****************************************************************************/
 
#include "headers.h"

#include "../HYPRE.h"

/******************************************************************************
 *
 * hypre_IJMatrixCreateParCSR
 *
 *****************************************************************************/

int
hypre_IJMatrixCreateParCSR(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   int *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   int *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   hypre_ParCSRMatrix *par_matrix;
   int *row_starts;
   int *col_starts;
   int num_procs;
   int i;
   int ierr = 0;

   MPI_Comm_size(comm,&num_procs);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts = hypre_CTAlloc(int,2);
   if (hypre_IJMatrixGlobalFirstRow(matrix))
      for (i=0; i < 2; i++)
	 row_starts[i] = row_partitioning[i]- hypre_IJMatrixGlobalFirstRow(matrix);
   else
      for (i=0; i < 2; i++)
	 row_starts[i] = row_partitioning[i];
   if (row_partitioning != col_partitioning)
   {
      col_starts = hypre_CTAlloc(int,2);
      if (hypre_IJMatrixGlobalFirstCol(matrix))
	 for (i=0; i < 2; i++)
	    col_starts[i] = col_partitioning[i]-hypre_IJMatrixGlobalFirstCol(matrix);
      else
	 for (i=0; i < 2; i++)
	    col_starts[i] = col_partitioning[i];
   }
   else
      col_starts = row_starts;

   par_matrix = hypre_ParCSRMatrixCreate(comm, hypre_IJMatrixGlobalNumRows(matrix),
		 hypre_IJMatrixGlobalNumCols(matrix),row_starts, col_starts, 0, 0, 0);

#else
   row_starts = hypre_CTAlloc(int,num_procs+1);
   if (row_partitioning[0])
	 for (i=0; i < num_procs+1; i++)
	 row_starts[i] = row_partitioning[i]-row_partitioning[0];
   else
	 for (i=0; i < num_procs+1; i++)
	 row_starts[i] = row_partitioning[i];
   if (row_partitioning != col_partitioning)
   {
	 col_starts = hypre_CTAlloc(int,num_procs+1);
      if (col_partitioning[0])
	 for (i=0; i < num_procs+1; i++)
	    col_starts[i] = col_partitioning[i]-col_partitioning[0];
      else
	 for (i=0; i < num_procs+1; i++)
	    col_starts[i] = col_partitioning[i];
   }
   else
	 col_starts = row_starts;
   par_matrix = hypre_ParCSRMatrixCreate(comm,row_starts[num_procs],
		col_starts[num_procs],row_starts, col_starts, 0, 0, 0);
#endif

   hypre_IJMatrixObject(matrix) = par_matrix;

   return ierr;
}


/******************************************************************************
 *
 * hypre_IJMatrixSetRowSizesParCSR
 *
 *****************************************************************************/
int
hypre_IJMatrixSetRowSizesParCSR(hypre_IJMatrix *matrix,
			      	const int      *sizes)
{
   int local_num_rows, local_num_cols;
   int i, my_id;
   int ierr = 0;
   int *row_space;
   int *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   int *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   hypre_AuxParCSRMatrix *aux_matrix;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);

   MPI_Comm_rank(comm,&my_id);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   local_num_rows = row_partitioning[1]-row_partitioning[0];
   local_num_cols = col_partitioning[1]-col_partitioning[0];
#else
   local_num_rows = row_partitioning[my_id+1]-row_partitioning[my_id];
   local_num_cols = col_partitioning[my_id+1]-col_partitioning[my_id];
#endif
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   row_space = NULL;
   if (aux_matrix)
      row_space =  hypre_AuxParCSRMatrixRowSpace(aux_matrix);
   if (!row_space)
      row_space = hypre_CTAlloc(int, local_num_rows);
   for (i = 0; i < local_num_rows; i++)
      row_space[i] = sizes[i];
   if (!aux_matrix)
   {
      ierr = hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
				local_num_cols, row_space);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixRowSpace(aux_matrix) = row_space;
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetDiagOffdSizesParCSR
 * sets diag_i inside the diag part of the ParCSRMatrix
 * and offd_i inside the offd part,
 * requires exact row sizes for diag and offd
 *
 *****************************************************************************/
int
hypre_IJMatrixSetDiagOffdSizesParCSR(hypre_IJMatrix *matrix,
			      	     const int	   *diag_sizes,
			      	     const int	   *offdiag_sizes)
{
   int local_num_rows;
   int i, ierr = 0;
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   int *diag_i;
   int *offd_i;

   if (!par_matrix)
   {
      hypre_IJMatrixCreateParCSR(matrix);
      par_matrix = hypre_IJMatrixObject(matrix);
   }
   
   diag =  hypre_ParCSRMatrixDiag(par_matrix);
   diag_i =  hypre_CSRMatrixI(diag); 
   local_num_rows = hypre_CSRMatrixNumRows(diag); 
   if (!diag_i) 
      diag_i = hypre_CTAlloc(int, local_num_rows+1); 
   for (i = 0; i < local_num_rows; i++) 
      diag_i[i+1] = diag_i[i] + diag_sizes[i]; 
   hypre_CSRMatrixI(diag) = diag_i; 
   hypre_CSRMatrixNumNonzeros(diag) = diag_i[local_num_rows]; 
   offd =  hypre_ParCSRMatrixOffd(par_matrix); 
   offd_i =  hypre_CSRMatrixI(offd); 
   if (!offd_i)
      offd_i = hypre_CTAlloc(int, local_num_rows+1);
   for (i = 0; i < local_num_rows; i++)
      offd_i[i+1] = offd_i[i] + offdiag_sizes[i];
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixNumNonzeros(offd) = offd_i[local_num_rows];
   if (!aux_matrix)
   {
      ierr = hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
				hypre_CSRMatrixNumCols(diag), NULL);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetMaxOffProcElmtsParCSR
 *
 *****************************************************************************/

int
hypre_IJMatrixSetMaxOffProcElmtsParCSR(hypre_IJMatrix *matrix,
			      	       int max_off_proc_elmts)
{
   int ierr = 0;
   hypre_AuxParCSRMatrix *aux_matrix;
   int local_num_rows, local_num_cols, my_id;
   int *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   int *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   MPI_Comm comm = hypre_IJMatrixComm(matrix);

   MPI_Comm_rank(comm,&my_id);
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   if (!aux_matrix)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      local_num_rows = row_partitioning[1]-row_partitioning[0];
      local_num_cols = col_partitioning[1]-col_partitioning[0];
#else
      local_num_rows = row_partitioning[my_id+1]-row_partitioning[my_id];
      local_num_cols = col_partitioning[my_id+1]-col_partitioning[my_id];
#endif
      ierr = hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
				local_num_cols, NULL);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixInitializeParCSR
 *
 * initializes AuxParCSRMatrix and ParCSRMatrix as necessary
 *
 *****************************************************************************/

int
hypre_IJMatrixInitializeParCSR(hypre_IJMatrix *matrix)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   int local_num_rows;

   if (hypre_IJMatrixAssembleFlag(matrix) == 0)
   {
      if (!par_matrix)
      {
         hypre_IJMatrixCreateParCSR(matrix);
         par_matrix = hypre_IJMatrixObject(matrix);
      }
      local_num_rows = 
		hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(par_matrix));
      if (!aux_matrix)
      {
         ierr = hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
		hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(par_matrix)), 
		NULL);
         hypre_IJMatrixTranslator(matrix) = aux_matrix;
      }
     
      ierr += hypre_ParCSRMatrixInitialize(par_matrix);
      ierr += hypre_AuxParCSRMatrixInitialize(aux_matrix);
      if (! hypre_AuxParCSRMatrixNeedAux(aux_matrix))
      {
         int i, *indx_diag, *indx_offd, *diag_i, *offd_i;
         diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix));
         offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix));
         indx_diag = hypre_AuxParCSRMatrixIndxDiag(aux_matrix);
 
         indx_offd = hypre_AuxParCSRMatrixIndxOffd(aux_matrix);
         for (i=0; i < local_num_rows; i++)
         {
	    indx_diag[i] = diag_i[i];
	    indx_offd[i] = offd_i[i];
         }
      }
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixGetRowCountsParCSR
 *
 * gets the number of columns for rows specified by the user
 * 
 *****************************************************************************/
int
hypre_IJMatrixGetRowCountsParCSR( hypre_IJMatrix *matrix,
                               int	       nrows,
                               int            *rows,
                               int	      *ncols)
{
   int ierr = 0;
   int row_index;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);

   int *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(par_matrix);
   int *diag_i = hypre_CSRMatrixI(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(par_matrix);
   int *offd_i = hypre_CSRMatrixI(offd);

   int i, my_id;

   MPI_Comm_rank(comm,&my_id);

   for (i=0; i < nrows; i++)
   {
      row_index = rows[i];
#ifdef HYPRE_NO_GLOBAL_PARTITION
      if (row_index >= row_partitioning[0] && 
		row_index < row_partitioning[1])
      {
   	/* compute local row number */
         row_index -= row_partitioning[0]; 
#else
      if (row_index >= row_partitioning[my_id] && 
		row_index < row_partitioning[my_id+1])
      {
   	/* compute local row number */
         row_index -= row_partitioning[my_id]; 
#endif
         ncols[i] = diag_i[row_index+1]-diag_i[row_index]+offd_i[row_index+1]
		-offd_i[row_index];
      }
      else
      {
         ncols[i] = 0;
	 printf ("Warning! Row %d is not on Proc. %d!\n", row_index, my_id);
      }
   }
   
   return ierr;
}
/******************************************************************************
 *
 * hypre_IJMatrixGetValuesParCSR
 *
 * gets values of an IJMatrix
 * 
 *****************************************************************************/
int
hypre_IJMatrixGetValuesParCSR( hypre_IJMatrix *matrix,
                               int	       nrows,
                               int	      *ncols,
                               int            *rows,
                               int            *cols,
                               double         *values)
{
   int ierr = 0;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);
   int assemble_flag = hypre_IJMatrixAssembleFlag(matrix);

   hypre_CSRMatrix *diag;
   int *diag_i;
   int *diag_j;
   double *diag_data;

   hypre_CSRMatrix *offd;
   int *offd_i;
   int *offd_j;
   double *offd_data;

   int *col_map_offd;
   int *col_starts = hypre_ParCSRMatrixColStarts(par_matrix);

   int *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);

#ifndef HYPRE_NO_GLOBAL_PARTITION
   int *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
#endif

   int i, j, n, ii, indx, col_indx;
   int num_procs, my_id;
   int col_0, col_n, row, row_local, row_size;
   int warning = 0;
   int *counter;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (assemble_flag == 0)
   {
      printf("Error! Matrix not assembled yet! HYPRE_IJMatrixGetValues\n");
      exit(1);
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   col_0 = col_starts[0];
   col_n = col_starts[1]-1;
#else
   col_0 = col_starts[my_id];
   col_n = col_starts[my_id+1]-1;
#endif

   diag = hypre_ParCSRMatrixDiag(par_matrix);
   diag_i = hypre_CSRMatrixI(diag);
   diag_j = hypre_CSRMatrixJ(diag);
   diag_data = hypre_CSRMatrixData(diag);
   
   offd = hypre_ParCSRMatrixOffd(par_matrix);
   offd_i = hypre_CSRMatrixI(offd);
   if (num_procs > 1)
   {
      offd_j = hypre_CSRMatrixJ(offd);
      offd_data = hypre_CSRMatrixData(offd);
      col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
   }

   if (nrows < 0)
   {
      nrows = -nrows;
      
      counter = hypre_CTAlloc(int,nrows+1);
      counter[0] = 0;
      for (i=0; i < nrows; i++)
         counter[i+1] = counter[i]+ncols[i];

      indx = 0;   
      for (i=0; i < nrows; i++)
      {
         row = rows[i];
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = row - row_partitioning[0]; 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = row - row_partitioning[my_id]; 
#endif
            row_size = diag_i[row_local+1]-diag_i[row_local]+
			offd_i[row_local+1]-offd_i[row_local];
            if (counter[i]+row_size > counter[nrows])
	    {
	       printf ("Error! Not enough memory! HYPRE_IJMatrixGetValues\n");
	       exit(1);
	    }
            if (ncols[i] < row_size)
		warning = 1;
	    for (j = diag_i[row_local]; j < diag_i[row_local+1]; j++)
 	    {
	       cols[indx] = diag_j[j] + col_0;
	       values[indx++] = diag_data[j];
	    }
	    for (j = offd_i[row_local]; j < offd_i[row_local+1]; j++)
 	    {
	       cols[indx] = col_map_offd[offd_j[j]];
	       values[indx++] = offd_data[j];
	    }
	    counter[i+1] = indx;
         }
         else
	    printf ("Warning! Row %d is not on Proc. %d!\n", row, my_id);
      }
      if (warning)
      {
         for (i=0; i < nrows; i++)
	    ncols[i] = counter[i+1] - counter[i];
	 printf ("Warning!  ncols has been changed!\n");
      }
      hypre_TFree(counter);
   }
   else
   {
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = row - row_partitioning[0]; 
            /* compute local row number */
     	    for (i=0; i < n; i++)
   	    {
   	       col_indx = cols[indx] -  hypre_IJMatrixGlobalFirstCol(matrix);

#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = row - row_partitioning[my_id]; 
   				/* compute local row number */
     	    for (i=0; i < n; i++)
   	    {
   	       col_indx = cols[indx] - col_partitioning[0];
#endif



   	       values[indx] = 0.0;
   	       if (col_indx < col_0 || col_indx > col_n)
   				/* search in offd */	
   	       {
   	          for (j=offd_i[row_local]; j < offd_i[row_local+1]; j++)
   	          {
   		     if (col_map_offd[offd_j[j]] == col_indx)
   		     {
                        values[indx] = offd_data[j];
   		        break;
   		     }
   	          }
	       }
	       else  /* search in diag */
	       {
   	          col_indx = col_indx - col_0;
	          for (j=diag_i[row_local]; j < diag_i[row_local+1]; j++)
	          {
		     if (diag_j[j] == col_indx)
		     {
                        values[indx] = diag_data[j];
		        break;
		     }
	          } 
	       }
	       indx++;
	    }
         }
         else
	    printf ("Warning! Row %d is not on Proc. %d!\n", row, my_id);
      }
   }

   return ierr;
}
/******************************************************************************
 *
 * hypre_IJMatrixSetValuesParCSR
 *
 * sets or adds row values to an IJMatrix before assembly, 
 * 
 *****************************************************************************/
int
hypre_IJMatrixSetValuesParCSR( hypre_IJMatrix *matrix,
                               int	       nrows,
                               int	      *ncols,
                               const int      *rows,
                               const int      *cols,
                               const double   *values)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   int *row_partitioning;
   int *col_partitioning;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   int num_procs, my_id;
   int row_local, row;
   int col_0, col_n;
   int i, ii, j, n, not_found;
   int **aux_j;
   int *local_j;
   int *tmp_j;
   double **aux_data;
   double *local_data;
   double *tmp_data;
   int diag_space, offd_space;
   int *row_length, *row_space;
   int need_aux;
   int tmp_indx, indx;
   int space, size, old_size;
   int cnt, cnt_diag, cnt_offd;
   int pos_diag, pos_offd;
   int len_diag, len_offd;
   int offd_indx, diag_indx;
   int *diag_i;
   int *diag_j;
   double *diag_data;
   int *offd_i;
   int *offd_j;
   double *offd_data;
   int first;
   int current_num_elmts;
   int max_off_proc_elmts;
   int off_proc_i_indx;
   int *off_proc_i;
   int *off_proc_j;
   double *off_proc_data;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   par_matrix = hypre_IJMatrixObject( matrix );
   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1]-1;
   first =  hypre_IJMatrixGlobalFirstCol(matrix);
#else
   col_0 = col_partitioning[my_id];
   col_n = col_partitioning[my_id+1]-1;
   first = col_partitioning[0];
#endif
   if (nrows < 0)
   {
      printf("Error! nrows negative! HYPRE_IJMatrixSetValues\n");
      exit(1);
   }

   if (hypre_IJMatrixAssembleFlag(matrix))
   {
      int *col_map_offd;
      int num_cols_offd;
      int j_offd;
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = row - row_partitioning[0]; 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = row - row_partitioning[my_id]; 
#endif

				/* compute local row number */
            diag = hypre_ParCSRMatrixDiag(par_matrix);
            diag_i = hypre_CSRMatrixI(diag);
            diag_j = hypre_CSRMatrixJ(diag);
            diag_data = hypre_CSRMatrixData(diag);
            offd = hypre_ParCSRMatrixOffd(par_matrix);
            offd_i = hypre_CSRMatrixI(offd);
            num_cols_offd = hypre_CSRMatrixNumCols(offd);
            if (num_cols_offd)
            {
               col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
               offd_j = hypre_CSRMatrixJ(offd);
               offd_data = hypre_CSRMatrixData(offd);
            }
            size = diag_i[row_local+1] - diag_i[row_local]
      			+ offd_i[row_local+1] - offd_i[row_local];
      
            if (n > size)
            {
      	       printf (" row %d too long! \n", row);
      	       return -1;
            }
       
            pos_diag = diag_i[row_local];
            pos_offd = offd_i[row_local];
            len_diag = diag_i[row_local+1];
            len_offd = offd_i[row_local+1];
            not_found = 1;
      	
            for (i=0; i < n; i++)
            {
               if (cols[indx] < col_0 || cols[indx] > col_n)
				/* insert into offd */	
               {
      	          j_offd = hypre_BinarySearch(col_map_offd,cols[indx]-first,
						num_cols_offd);
      	          if (j_offd == -1)
      	          {
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
      	             return -1;
      	          }
      	          for (j=pos_offd; j < len_offd; j++)
      	          {
      	             if (offd_j[j] == j_offd)
      	             {
                        offd_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
      	             return -1;
      	          }
      	          not_found = 1;
               }
               /* diagonal element */
      	       else if (cols[indx] == row)
      	       {
      	          if (diag_j[pos_diag] != row_local)
      	          {
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
      	             return -1;
      	          }
      	          diag_data[pos_diag] = values[indx];
      	       }
               else  /* insert into diag */
               {
      	          for (j=pos_diag; j < len_diag; j++)
      	          {
      	             if (diag_j[j] == (cols[indx]-col_0))
      	             {
                        diag_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
      	             return -1;
      	          }
               }
               indx++;
            }
         }
         else
	 {
   	    if (!aux_matrix)
            {
#ifdef HYPRE_NO_GLOBAL_PARTITION
               size = row_partitioning[1]-row_partitioning[0];
#else
               size = row_partitioning[my_id+1]-row_partitioning[my_id];
#endif
	       hypre_AuxParCSRMatrixCreate(&aux_matrix, size, size, NULL);
      	       hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
      	       hypre_IJMatrixTranslator(matrix) = aux_matrix;
            }
   	    current_num_elmts 
		= hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts
		= hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = hypre_max(n,1000);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
	       		max_off_proc_elmts;
   	       hypre_AuxParCSRMatrixOffProcI(aux_matrix)
			= hypre_CTAlloc(int,2*max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
			= hypre_CTAlloc(int,max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcData(aux_matrix)
			= hypre_CTAlloc(double,max_off_proc_elmts);
   	       off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = hypre_TReAlloc(off_proc_i,int,2*max_off_proc_elmts);
               off_proc_j = hypre_TReAlloc(off_proc_j,int,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
				max_off_proc_elmts);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
   	    		= max_off_proc_elmts;
	       hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = row; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix)
   	    	= current_num_elmts; 
	 }
      }
   }
   else
   {
      aux_matrix = hypre_IJMatrixTranslator(matrix);
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = row - row_partitioning[0]; 
#else
            if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = row - row_partitioning[my_id]; 

#endif
			/* compute local row number */
            if (need_aux)
            {
               aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
               aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
               local_j = aux_j[row_local];
               local_data = aux_data[row_local];
   	       space = row_space[row_local]; 
   	       old_size = row_length[row_local]; 
   	       size = space - old_size;
   	       if (size < n)
      	       {
      	          size = n - size;
      	          tmp_j = hypre_CTAlloc(int,size);
      	          tmp_data = hypre_CTAlloc(double,size);
      	       }
      	       else
      	       {
      	          tmp_j = NULL;
      	       }
      	       tmp_indx = 0;
      	       not_found = 1;
      	       size = old_size;
               for (i=0; i < n; i++)
      	       {
      	          for (j=0; j < old_size; j++)
      	          {
      	             if (local_j[j] == cols[indx])
      	             {
                        local_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             if (size < space)
      	             {
      	                local_j[size] = cols[indx];
      	                local_data[size++] = values[indx];
      	             }
      	             else
      	             {
      	                tmp_j[tmp_indx] = cols[indx];
      	                tmp_data[tmp_indx++] = values[indx];
      	             }
      	          }
      	          not_found = 1;
        	  indx++;
      	       }
      	    
               row_length[row_local] = size+tmp_indx;
               
               if (tmp_indx)
               {
   	          aux_j[row_local] = hypre_TReAlloc(aux_j[row_local],int,
   				size+tmp_indx);
   	          aux_data[row_local] = hypre_TReAlloc(aux_data[row_local],
   					double,size+tmp_indx);
                  row_space[row_local] = size+tmp_indx;
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
               }
   
   	       cnt = size; 
   
   	       for (i=0; i < tmp_indx; i++)
   	       {
   	          local_j[cnt] = tmp_j[i];
   	          local_data[cnt++] = tmp_data[i];
	       }
  
	       if (tmp_j)
	       { 
	          hypre_TFree(tmp_j); 
	          hypre_TFree(tmp_data); 
	       } 
            }
            else /* insert immediately into data in ParCSRMatrix structure */
            {
	       offd_indx =hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	       diag_indx =hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
               diag = hypre_ParCSRMatrixDiag(par_matrix);
               diag_i = hypre_CSRMatrixI(diag);
               diag_j = hypre_CSRMatrixJ(diag);
               diag_data = hypre_CSRMatrixData(diag);
               offd = hypre_ParCSRMatrixOffd(par_matrix);
               offd_i = hypre_CSRMatrixI(offd);
               if (num_procs > 1)
	       {
	          offd_j = hypre_CSRMatrixJ(offd);
                  offd_data = hypre_CSRMatrixData(offd);
               }
	       cnt_diag = diag_indx;
	       cnt_offd = offd_indx;
	       diag_space = diag_i[row_local+1];
	       offd_space = offd_i[row_local+1];
	       not_found = 1;
  	       for (i=0; i < n; i++)
	       {
	          if (cols[indx] < col_0 || cols[indx] > col_n)
				/* insert into offd */	
	          {
	             for (j=offd_i[row_local]; j < offd_indx; j++)
	             {
		        if (offd_j[j] == cols[indx])
		        {
                           offd_data[j] = values[indx];
		           not_found = 0;
		           break;
		        }
	             }
	             if (not_found)
	             { 
	                if (cnt_offd < offd_space) 
	                { 
	                   offd_j[cnt_offd] = cols[indx];
	                   offd_data[cnt_offd++] = values[indx];
	                } 
	                else 
	 	        {
	    	           printf("Error in row %d ! Too many elements!\n", 
				row);
	    	           return 1;
	 	        }
	             }  
	             not_found = 1;
	          }
	          else  /* insert into diag */
	          {
	             for (j=diag_i[row_local]; j < diag_indx; j++)
	             {
		        if (diag_j[j] == cols[indx])
		        {
                           diag_data[j] = values[indx];
		           not_found = 0;
		           break;
		        }
	             } 
	             if (not_found)
	             { 
	                if (cnt_diag < diag_space) 
	                { 
	                   diag_j[cnt_diag] = cols[indx];
	                   diag_data[cnt_diag++] = values[indx];
	                } 
	                else 
	 	        {
	    	           printf("Error in row %d ! Too many elements !\n", 
				row);
	    	           return 1;
	 	        }
	             } 
	             not_found = 1;
	          }
	          indx++;
	       }

               hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
               hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

            }
         }
         else
	 {
   	    current_num_elmts 
		= hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts
		= hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = hypre_max(n,1000);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
	       		max_off_proc_elmts;
   	       hypre_AuxParCSRMatrixOffProcI(aux_matrix)
			= hypre_CTAlloc(int,2*max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
			= hypre_CTAlloc(int,max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcData(aux_matrix)
			= hypre_CTAlloc(double,max_off_proc_elmts);
   	       off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = hypre_TReAlloc(off_proc_i,int,2*max_off_proc_elmts);
               off_proc_j = hypre_TReAlloc(off_proc_j,int,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
				max_off_proc_elmts);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
   	    		= max_off_proc_elmts;
	       hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = row; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix)
   	    	= current_num_elmts; 
	 }
      }
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixAddToValuesParCSR
 *
 * adds row values to an IJMatrix 
 * 
 *****************************************************************************/
int
hypre_IJMatrixAddToValuesParCSR( hypre_IJMatrix *matrix,
                                 int	       nrows,
                                 int	      *ncols,
                                 const int      *rows,
                                 const int      *cols,
                                 const double   *values)
{
   int ierr = 0;
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   int *row_partitioning;
   int *col_partitioning;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   int num_procs, my_id;
   int row_local, row;
   int col_0, col_n;
   int i, ii, j, n, not_found;
   int **aux_j;
   int *local_j;
   int *tmp_j;
   double **aux_data;
   double *local_data;
   double *tmp_data;
   int diag_space, offd_space;
   int *row_length, *row_space;
   int need_aux;
   int tmp_indx, indx;
   int space, size, old_size;
   int cnt, cnt_diag, cnt_offd;
   int pos_diag, pos_offd;
   int len_diag, len_offd;
   int offd_indx, diag_indx;
   int first;
   int *diag_i;
   int *diag_j;
   double *diag_data;
   int *offd_i;
   int *offd_j;
   double *offd_data;
   int current_num_elmts;
   int max_off_proc_elmts;
   int off_proc_i_indx;
   int *off_proc_i;
   int *off_proc_j;
   double *off_proc_data;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   par_matrix = hypre_IJMatrixObject( matrix );
   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1]-1;
   first = hypre_IJMatrixGlobalFirstCol(matrix);
#else
   col_0 = col_partitioning[my_id];
   col_n = col_partitioning[my_id+1]-1;
   first = col_partitioning[0];
#endif
   if (hypre_IJMatrixAssembleFlag(matrix))
   {
      int num_cols_offd;
      int *col_map_offd;
      int j_offd;
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
#ifdef HYPRE_NO_GLOBAL_PARTITION
  if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = row - row_partitioning[0]; 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = row - row_partitioning[my_id]; 
#endif
				/* compute local row number */
            diag = hypre_ParCSRMatrixDiag(par_matrix);
            diag_i = hypre_CSRMatrixI(diag);
            diag_j = hypre_CSRMatrixJ(diag);
            diag_data = hypre_CSRMatrixData(diag);
            offd = hypre_ParCSRMatrixOffd(par_matrix);
            offd_i = hypre_CSRMatrixI(offd);
            num_cols_offd = hypre_CSRMatrixNumCols(offd);
            if (num_cols_offd)
            {
               col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
               offd_j = hypre_CSRMatrixJ(offd);
               offd_data = hypre_CSRMatrixData(offd);
            }
            size = diag_i[row_local+1] - diag_i[row_local]
      			+ offd_i[row_local+1] - offd_i[row_local];
      
            if (n > size)
            {
      	       printf (" row %d too long! \n", row);
      	       return -1;
            }
       
            pos_diag = diag_i[row_local];
            pos_offd = offd_i[row_local];
            len_diag = diag_i[row_local+1];
            len_offd = offd_i[row_local+1];
            not_found = 1;
      	
            for (i=0; i < n; i++)
            {
               if (cols[indx] < col_0 || cols[indx] > col_n)
				/* insert into offd */	
               {
      	          j_offd = hypre_BinarySearch(col_map_offd,cols[indx]-first,
						num_cols_offd);
      	          if (j_offd == -1)
      	          {
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
      	             return -1;
      	          }
      	          for (j=pos_offd; j < len_offd; j++)
      	          {
      	             if (offd_j[j] == j_offd)
      	             {
                        offd_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
      	             return -1;
      	          }
      	          not_found = 1;
               }
               /* diagonal element */
      	       else if (cols[indx] == row)
      	       {
      	          if (diag_j[pos_diag] != row_local)
      	          {
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
      	             return -1;
      	          }
      	          diag_data[pos_diag] += values[indx];
      	       }
               else  /* insert into diag */
               {
      	          for (j=pos_diag; j < len_diag; j++)
      	          {
      	             if (diag_j[j] == (cols[indx]-col_0))
      	             {
                        diag_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
      	             return -1;
      	          }
               }
            indx++;
            }
         }
         else
	 {
   	    if (!aux_matrix)
            {
#ifdef HYPRE_NO_GLOBAL_PARTITION
               size = row_partitioning[1]-row_partitioning[0];
#else
               size = row_partitioning[my_id+1]-row_partitioning[my_id];
#endif
	       hypre_AuxParCSRMatrixCreate(&aux_matrix, size, size, NULL);
      	       hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
      	       hypre_IJMatrixTranslator(matrix) = aux_matrix;
            }
   	    current_num_elmts 
		= hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts
		= hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = hypre_max(n,1000);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
	       		max_off_proc_elmts;
   	       hypre_AuxParCSRMatrixOffProcI(aux_matrix)
			= hypre_CTAlloc(int,2*max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
			= hypre_CTAlloc(int,max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcData(aux_matrix)
			= hypre_CTAlloc(double,max_off_proc_elmts);
   	       off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = hypre_TReAlloc(off_proc_i,int,2*max_off_proc_elmts);
               off_proc_j = hypre_TReAlloc(off_proc_j,int,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
				max_off_proc_elmts);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
   	    		= max_off_proc_elmts;
	       hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = row; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix)
   	    	= current_num_elmts; 
	 }
      }
   }
   else
   {
      aux_matrix = hypre_IJMatrixTranslator(matrix);
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
#ifdef HYPRE_NO_GLOBAL_PARTITION
  if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = row - row_partitioning[0]; 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = row - row_partitioning[my_id]; 
#endif
			/* compute local row number */
            if (need_aux)
            {
               aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
               aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
               local_j = aux_j[row_local];
               local_data = aux_data[row_local];
   	       space = row_space[row_local]; 
   	       old_size = row_length[row_local]; 
   	       size = space - old_size;
   	       if (size < n)
      	       {
      	          size = n - size;
      	          tmp_j = hypre_CTAlloc(int,size);
      	          tmp_data = hypre_CTAlloc(double,size);
      	       }
      	       else
      	       {
      	          tmp_j = NULL;
      	       }
      	       tmp_indx = 0;
      	       not_found = 1;
      	       size = old_size;
               for (i=0; i < n; i++)
      	       {
      	          for (j=0; j < old_size; j++)
      	          {
      	             if (local_j[j] == cols[indx])
      	             {
                        local_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             if (size < space)
      	             {
      	                local_j[size] = cols[indx];
      	                local_data[size++] = values[indx];
      	             }
      	             else
      	             {
      	                tmp_j[tmp_indx] = cols[indx];
      	                tmp_data[tmp_indx++] = values[indx];
      	             }
      	          }
      	          not_found = 1;
        	  indx++;
      	       }
      	    
               row_length[row_local] = size+tmp_indx;
               
               if (tmp_indx)
               {
   	          aux_j[row_local] = hypre_TReAlloc(aux_j[row_local],int,
   				size+tmp_indx);
   	          aux_data[row_local] = hypre_TReAlloc(aux_data[row_local],
   					double,size+tmp_indx);
                  row_space[row_local] = size+tmp_indx;
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
               }
   
   	       cnt = size; 
   
   	       for (i=0; i < tmp_indx; i++)
   	       {
   	          local_j[cnt] = tmp_j[i];
   	          local_data[cnt++] = tmp_data[i];
	       }
  
	       if (tmp_j)
	       { 
	          hypre_TFree(tmp_j); 
	          hypre_TFree(tmp_data); 
	       } 
            }
            else /* insert immediately into data in ParCSRMatrix structure */
            {
	       offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	       diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
               diag = hypre_ParCSRMatrixDiag(par_matrix);
               diag_i = hypre_CSRMatrixI(diag);
               diag_j = hypre_CSRMatrixJ(diag);
               diag_data = hypre_CSRMatrixData(diag);
               offd = hypre_ParCSRMatrixOffd(par_matrix);
               offd_i = hypre_CSRMatrixI(offd);
               if (num_procs > 1)
	       {
	          offd_j = hypre_CSRMatrixJ(offd);
                  offd_data = hypre_CSRMatrixData(offd);
               }
	       cnt_diag = diag_indx;
	       cnt_offd = offd_indx;
	       diag_space = diag_i[row_local+1];
	       offd_space = offd_i[row_local+1];
	       not_found = 1;
  	       for (i=0; i < n; i++)
	       {
	          if (cols[indx] < col_0 || cols[indx] > col_n)
				/* insert into offd */	
	          {
	             for (j=offd_i[row_local]; j < offd_indx; j++)
	             {
		        if (offd_j[j] == cols[indx])
		        {
                           offd_data[j] += values[indx];
		           not_found = 0;
		           break;
		        }
	             }
	             if (not_found)
	             { 
	                if (cnt_offd < offd_space) 
	                { 
	                   offd_j[cnt_offd] = cols[indx];
	                   offd_data[cnt_offd++] = values[indx];
	                } 
	                else 
	 	        {
	    	           printf("Error in row %d ! Too many elements!\n", 
				row);
	    	           return 1;
	 	        }
	             }  
	             not_found = 1;
	          }
	          else  /* insert into diag */
	          {
	             for (j=diag_i[row_local]; j < diag_indx; j++)
	             {
		        if (diag_j[j] == cols[indx])
		        {
                           diag_data[j] += values[indx];
		           not_found = 0;
		           break;
		        }
	             } 
	             if (not_found)
	             { 
	                if (cnt_diag < diag_space) 
	                { 
	                   diag_j[cnt_diag] = cols[indx];
	                   diag_data[cnt_diag++] = values[indx];
	                } 
	                else 
	 	        {
	    	           printf("Error in row %d ! Too many elements !\n", 
				row);
	    	           return 1;
	 	        }
	             } 
	             not_found = 1;
	          }
	          indx++;
	       }

               hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
               hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

            }
         }
         else
         {
   	    current_num_elmts 
		= hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts
		= hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = hypre_max(n,1000);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
	       		max_off_proc_elmts;
   	       hypre_AuxParCSRMatrixOffProcI(aux_matrix)
			= hypre_CTAlloc(int,2*max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
			= hypre_CTAlloc(int,max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcData(aux_matrix)
			= hypre_CTAlloc(double,max_off_proc_elmts);
   	       off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = hypre_TReAlloc(off_proc_i,int,2*max_off_proc_elmts);
               off_proc_j = hypre_TReAlloc(off_proc_j,int,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
				max_off_proc_elmts);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
   	    		= max_off_proc_elmts;
	       hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = -row-1; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix)
   	    	= current_num_elmts; 
         }
      }
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixAssembleParCSR
 *
 * assembles IJMatrix from AuxParCSRMatrix auxiliary structure
 *****************************************************************************/
int
hypre_IJMatrixAssembleParCSR(hypre_IJMatrix *matrix)
{
   int ierr = 0;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   int *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   int *col_partitioning = hypre_IJMatrixColPartitioning(matrix);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(par_matrix);
   int *diag_i = hypre_CSRMatrixI(diag);
   int *offd_i = hypre_CSRMatrixI(offd);
   int *diag_j;
   int *offd_j;
   double *diag_data;
   double *offd_data;
   int cnt, i, j, j0;
   int num_cols_offd;
   int *diag_pos;
   int *col_map_offd;
   int *row_length;
   int **aux_j;
   double **aux_data;
   int my_id, num_procs;
   int num_rows;
   int i_diag, i_offd;
   int *local_j;
   double *local_data;
   int col_0, col_n;
   int nnz_offd;
   int *aux_offd_j;
   double temp; 
#ifdef HYPRE_NO_GLOBAL_PARTITION
   int base = hypre_IJMatrixGlobalFirstCol(matrix);
#else
   int base = col_partitioning[0];
#endif
   int off_proc_i_indx;
   int max_off_proc_elmts;
   int current_num_elmts;
   int *off_proc_i;
   int *off_proc_j;
   double *off_proc_data;
   int offd_proc_elmts;

   if (aux_matrix)
   {
      off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
      MPI_Allreduce(&off_proc_i_indx,&offd_proc_elmts,1,MPI_INT, MPI_SUM,comm);
      if (offd_proc_elmts)
      {
          max_off_proc_elmts=hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
          current_num_elmts=hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
          off_proc_i=hypre_AuxParCSRMatrixOffProcI(aux_matrix);
          off_proc_j=hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
          off_proc_data=hypre_AuxParCSRMatrixOffProcData(aux_matrix);
          hypre_IJMatrixAssembleOffProcValsParCSR(matrix,off_proc_i_indx,
		max_off_proc_elmts, current_num_elmts, off_proc_i,
		off_proc_j, off_proc_data);
      }
   }

   if (hypre_IJMatrixAssembleFlag(matrix) == 0)
   {
      MPI_Comm_size(comm, &num_procs); 
      MPI_Comm_rank(comm, &my_id);
#ifdef HYPRE_NO_GLOBAL_PARTITION
      num_rows = row_partitioning[1] - row_partitioning[0]; 
      col_0 = col_partitioning[0];
      col_n = col_partitioning[1]-1;
#else
      num_rows = row_partitioning[my_id+1] - row_partitioning[my_id]; 
      col_0 = col_partitioning[my_id];
      col_n = col_partitioning[my_id+1]-1;
#endif
   /* move data into ParCSRMatrix if not there already */ 
      if (hypre_AuxParCSRMatrixNeedAux(aux_matrix))
      {
         aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
         row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
         diag_pos = hypre_CTAlloc(int, num_rows);
         i_diag = 0;
         i_offd = 0;
         for (i=0; i < num_rows; i++)
         {
   	    local_j = aux_j[i];
   	    local_data = aux_data[i];
   	    diag_pos[i] = -1;
   	    for (j=0; j < row_length[i]; j++)
   	    {
   	       if (local_j[j] < col_0 || local_j[j] > col_n)
   	          i_offd++;
   	       else
   	       {
   	          i_diag++;
   	          if (local_j[j]-col_0 == i) diag_pos[i] = j;
   	       }
   	    }
   	    diag_i[i+1] = i_diag;
   	    offd_i[i+1] = i_offd;
         }
         if (hypre_CSRMatrixJ(diag))
            hypre_TFree(hypre_CSRMatrixJ(diag));
         if (hypre_CSRMatrixData(diag))
            hypre_TFree(hypre_CSRMatrixData(diag));
         if (hypre_CSRMatrixJ(offd))
            hypre_TFree(hypre_CSRMatrixJ(offd));
         if (hypre_CSRMatrixData(offd))
            hypre_TFree(hypre_CSRMatrixData(offd));
         diag_j = hypre_CTAlloc(int,i_diag);
         diag_data = hypre_CTAlloc(double,i_diag);
         if (i_offd > 0)
         {
    	    offd_j = hypre_CTAlloc(int,i_offd);
            offd_data = hypre_CTAlloc(double,i_offd);
         }
   
         i_diag = 0;
         i_offd = 0;
         for (i=0; i < num_rows; i++)
         {
   	    local_j = aux_j[i];
   	    local_data = aux_data[i];
            if (diag_pos[i] > -1)
            {
   	       diag_j[i_diag] = local_j[diag_pos[i]] - col_0;
               diag_data[i_diag++] = local_data[diag_pos[i]];
            }
   	    for (j=0; j < row_length[i]; j++)
   	    {
   	       if (local_j[j] < col_0 || local_j[j] > col_n)
   	       {
   	          offd_j[i_offd] = local_j[j];
   	          offd_data[i_offd++] = local_data[j];
   	       }
   	       else if (j != diag_pos[i])
   	       {
   	          diag_j[i_diag] = local_j[j] - col_0;
   	          diag_data[i_diag++] = local_data[j];
   	       }
   	    }
         }
         hypre_CSRMatrixJ(diag) = diag_j;      
         hypre_CSRMatrixData(diag) = diag_data;      
         hypre_CSRMatrixNumNonzeros(diag) = diag_i[num_rows];      
         if (i_offd > 0)
         {
            hypre_CSRMatrixJ(offd) = offd_j;      
            hypre_CSRMatrixData(offd) = offd_data;      
         }
         hypre_CSRMatrixNumNonzeros(offd) = offd_i[num_rows];      
         hypre_TFree(diag_pos);
      }
      else
      {
         /* move diagonal element into first space */
   
         diag_j = hypre_CSRMatrixJ(diag);
         diag_data = hypre_CSRMatrixData(diag);
         for (i=0; i < num_rows; i++)
         {
   	    j0 = diag_i[i];
   	    for (j=j0; j < diag_i[i+1]; j++)
   	    {
   	       diag_j[j] -= col_0;
   	       if (diag_j[j] == i)
   	       {
   	          temp = diag_data[j0];
   	          diag_data[j0] = diag_data[j];
   	          diag_data[j] = temp;
   	          diag_j[j] = diag_j[j0];
   	          diag_j[j0] = i;
   	       }
   	    }
         }
   
         offd_j = hypre_CSRMatrixJ(offd);
      }

            /*  generate the nonzero rows inside offd and diag by calling */

      hypre_CSRMatrixSetRownnz(diag);
      hypre_CSRMatrixSetRownnz(offd);
   
   /*  generate col_map_offd */
      nnz_offd = offd_i[num_rows];
      if (nnz_offd)
      {
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
            offd_j[i]=hypre_BinarySearch(col_map_offd,offd_j[i],num_cols_offd);
         }
 	 if (base)
 	 {
	    for (i=0; i < num_cols_offd; i++)
	       col_map_offd[i] -= base;
	 } 
         hypre_ParCSRMatrixColMapOffd(par_matrix) = col_map_offd;    
         hypre_CSRMatrixNumCols(offd) = num_cols_offd;    
         hypre_TFree(aux_offd_j);
      }
      hypre_AuxParCSRMatrixDestroy(aux_matrix);
      hypre_IJMatrixTranslator(matrix) = NULL;
      hypre_IJMatrixAssembleFlag(matrix) = 1;
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixDestroyParCSR
 *
 * frees an IJMatrix
 *
 *****************************************************************************/
int
hypre_IJMatrixDestroyParCSR(hypre_IJMatrix *matrix)
{
   int ierr = 0;
   ierr = hypre_ParCSRMatrixDestroy(hypre_IJMatrixObject(matrix));
   ierr += hypre_AuxParCSRMatrixDestroy(hypre_IJMatrixTranslator(matrix));
   return ierr;
}




int
hypre_IJMatrixAssembleOffProcValsParCSR( hypre_IJMatrix *matrix, 
   					 int off_proc_i_indx,
   					 int max_off_proc_elmts,
   					 int current_num_elmts,
   					 int *off_proc_i,
   					 int *off_proc_j,
   					 double *off_proc_data)
{
   int ierr = 0;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   MPI_Request *requests;
   MPI_Status *status;
   int i, ii, j, j2, jj, n, row;
   int iii, iid, indx, ip;
   int proc_id, num_procs, my_id;
   int num_sends, num_sends3;
   int num_recvs;
   int num_requests;
   int vec_start, vec_len;
   int *send_procs;
   int *chunks;
   int *send_i;
   int *send_map_starts;
   int *dbl_send_map_starts;
   int *recv_procs;
   int *recv_chunks;
   int *recv_i;
   int *recv_vec_starts;
   int *dbl_recv_vec_starts;
   int *info;
   int *int_buffer;
   int *proc_id_mem;
   int *partitioning;
   int *displs;
   int *recv_buf;
   double *send_data;
   double *recv_data;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm, &my_id);
   partitioning = hypre_IJMatrixRowPartitioning(matrix);

   info = hypre_CTAlloc(int,num_procs);  
   chunks = hypre_CTAlloc(int,num_procs);  
   proc_id_mem = hypre_CTAlloc(int,off_proc_i_indx/2);
   j=0;
   for (i=0; i < off_proc_i_indx; i++)
   {
      row = off_proc_i[i++];
      if (row < 0) row = -row-1; 
      n = off_proc_i[i];
      proc_id = hypre_FindProc(partitioning,row,num_procs);
      proc_id_mem[j++] = proc_id; 
      info[proc_id] += n;
      chunks[proc_id]++;
   }

   /* determine send_procs and amount of data to be sent */   
   num_sends = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         num_sends++;
      }
   }
   send_procs =  hypre_CTAlloc(int,num_sends);
   send_map_starts =  hypre_CTAlloc(int,num_sends+1);
   dbl_send_map_starts =  hypre_CTAlloc(int,num_sends+1);
   num_sends3 = 3*num_sends;
   int_buffer =  hypre_CTAlloc(int,3*num_sends);
   j = 0;
   j2 = 0;
   send_map_starts[0] = 0;
   dbl_send_map_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         send_procs[j++] = i;
         send_map_starts[j] = send_map_starts[j-1]+2*chunks[i]+info[i];
         dbl_send_map_starts[j] = dbl_send_map_starts[j-1]+info[i];
         int_buffer[j2++] = i;
	 int_buffer[j2++] = chunks[i];
	 int_buffer[j2++] = info[i];
      }
   }

   hypre_TFree(chunks);

   MPI_Allgather(&num_sends3,1,MPI_INT,info,1,MPI_INT,comm);

   displs = hypre_CTAlloc(int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
        displs[i] = displs[i-1]+info[i-1];
   recv_buf = hypre_CTAlloc(int, displs[num_procs]);

   MPI_Allgatherv(int_buffer,num_sends3,MPI_INT,recv_buf,info,displs,
			MPI_INT,comm);

   hypre_TFree(int_buffer);
   hypre_TFree(info);

   /* determine recv procs and amount of data to be received */
   num_recvs = 0;
   for (j=0; j < displs[num_procs]; j+=3)
   {
      if (recv_buf[j] == my_id)
	 num_recvs++;
   }

   recv_procs = hypre_CTAlloc(int,num_recvs);
   recv_chunks = hypre_CTAlloc(int,num_recvs);
   recv_vec_starts = hypre_CTAlloc(int,num_recvs+1);
   dbl_recv_vec_starts = hypre_CTAlloc(int,num_recvs+1);

   j2 = 0;
   recv_vec_starts[0] = 0;
   dbl_recv_vec_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      for (j=displs[i]; j < displs[i+1]; j+=3)
      {
         if (recv_buf[j] == my_id)
         {
	    recv_procs[j2] = i;
	    recv_chunks[j2++] = recv_buf[j+1];
	    recv_vec_starts[j2] = recv_vec_starts[j2-1]+2*recv_buf[j+1]
			+recv_buf[j+2];
	    dbl_recv_vec_starts[j2] = dbl_recv_vec_starts[j2-1]+recv_buf[j+2];
         }
         if (j2 == num_recvs) break;
      }
   }
   hypre_TFree(recv_buf);
   hypre_TFree(displs);

   /* set up data to be sent to send procs */
   /* send_i contains for each send proc : row no., no. of elmts and column
      indices, send_data contains corresponding values */
      
   send_i = hypre_CTAlloc(int,send_map_starts[num_sends]);
   send_data = hypre_CTAlloc(double,dbl_send_map_starts[num_sends]);
   recv_i = hypre_CTAlloc(int,recv_vec_starts[num_recvs]);
   recv_data = hypre_CTAlloc(double,dbl_recv_vec_starts[num_recvs]);
    
   j=0;
   jj=0;
   for (i=0; i < off_proc_i_indx; i++)
   {
      row = off_proc_i[i++]; 
      n = off_proc_i[i];
      proc_id = proc_id_mem[i/2];
      indx = hypre_BinarySearch(send_procs,proc_id,num_sends);
      iii = send_map_starts[indx];
      iid = dbl_send_map_starts[indx];
      send_i[iii++] = row;
      send_i[iii++] = n;
      for (ii = 0; ii < n; ii++)
      {
          send_i[iii++] = off_proc_j[jj];
          send_data[iid++] = off_proc_data[jj++];
      }
      send_map_starts[indx] = iii;
      dbl_send_map_starts[indx] = iid;
   }

   hypre_TFree(proc_id_mem);

   for (i=num_sends; i > 0; i--)
   {
      send_map_starts[i] = send_map_starts[i-1];
      dbl_send_map_starts[i] = dbl_send_map_starts[i-1];
   }
   send_map_starts[0] = 0;
   dbl_send_map_starts[0] = 0;

   num_requests = num_recvs+num_sends;

   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);

   j=0; 
   for (i=0; i < num_recvs; i++)
   {
       vec_start = recv_vec_starts[i];
       vec_len = recv_vec_starts[i+1] - vec_start;
       ip = recv_procs[i];
       MPI_Irecv(&recv_i[vec_start], vec_len, MPI_INT, ip, 0, comm, 
			&requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
       vec_start = send_map_starts[i];
       vec_len = send_map_starts[i+1] - vec_start;
       ip = send_procs[i];
       MPI_Isend(&send_i[vec_start], vec_len, MPI_INT, ip, 0, comm, 
			&requests[j++]);
   }
  
   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
   }

   j=0;
   for (i=0; i < num_recvs; i++)
   {
       vec_start = dbl_recv_vec_starts[i];
       vec_len = dbl_recv_vec_starts[i+1] - vec_start;
       ip = recv_procs[i];
       MPI_Irecv(&recv_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, 
			&requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
       vec_start = dbl_send_map_starts[i];
       vec_len = dbl_send_map_starts[i+1] - vec_start;
       ip = send_procs[i];
       MPI_Isend(&send_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, 
			&requests[j++]);
   }
  
   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }

   hypre_TFree(send_i);
   hypre_TFree(send_data);
   hypre_TFree(send_procs);
   hypre_TFree(send_map_starts);
   hypre_TFree(dbl_send_map_starts);
   hypre_TFree(recv_procs);
   hypre_TFree(recv_vec_starts);
   hypre_TFree(dbl_recv_vec_starts);

   j = 0;
   j2 = 0;
   for (i=0; i < num_recvs; i++)
   {
      for (ii=0; ii < recv_chunks[i]; ii++)
      {
         row = recv_i[j];
	 if (row < 0) 
	 {
	    row = -row-1;
 	    hypre_IJMatrixAddToValuesParCSR(matrix,1,&recv_i[j+1],&row,
		&recv_i[j+2],&recv_data[j2]);
	    j2 += recv_i[j+1]; 
	    j += recv_i[j+1]+2; 
	 }
	 else
	 {
 	    hypre_IJMatrixSetValuesParCSR(matrix,1,&recv_i[j+1],&row,
		&recv_i[j+2],&recv_data[j2]);
	    j2 += recv_i[j+1]; 
	    j += recv_i[j+1]+2; 
	 }
      }
   }
   hypre_TFree(recv_chunks);
   hypre_TFree(recv_i);
   hypre_TFree(recv_data);

   return ierr;
}


int hypre_FindProc(int *list, int value, int list_length)
{
   int low, high, m;

   low = 0;
   high = list_length;
   if (value >= list[high] || value < list[low])
      return -1;
   else
   {
      while (low+1 < high)
      {
         m = (low + high) / 2;
         if (value < list[m])
         {
            high = m;
         }
         else if (value >= list[m])
         {
            low = m;
         }
      }
      return low;
   }
}
 
