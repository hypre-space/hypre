/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for MLI_CSRBooleanMatrix and MLI_ParCSRBooleanMatrix class.
 *
 *****************************************************************************/

/* jfp
#include "mli_pcsr_bool_mat.h"
#include "utilities.h"
*/
#include "headers.h"

/*--------------------------------------------------------------------------
 * MLI_CSRBooleanMatrixCreate
 *--------------------------------------------------------------------------*/

MLI_CSRBooleanMatrix *MLI_CSRBooleanMatrixCreate(int num_rows,int num_cols,
                                                 int num_nonzeros )
{
   MLI_CSRBooleanMatrix *matrix;

   matrix = hypre_CTAlloc(MLI_CSRBooleanMatrix, 1);

   MLI_CSRBooleanMatrix_Get_I(matrix)     = NULL;
   MLI_CSRBooleanMatrix_Get_J(matrix)     = NULL;
   MLI_CSRBooleanMatrix_Get_NRows(matrix) = num_rows;
   MLI_CSRBooleanMatrix_Get_NCols(matrix) = num_cols;
   MLI_CSRBooleanMatrix_Get_NNZ(matrix)   = num_nonzeros;
   MLI_CSRBooleanMatrix_Get_OwnsData(matrix) = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * MLI_CSRBooleanMatrixDestroy
 *--------------------------------------------------------------------------*/

int MLI_CSRBooleanMatrixDestroy( MLI_CSRBooleanMatrix *matrix )
{
   if (matrix)
   {
      hypre_TFree(MLI_CSRBooleanMatrix_Get_I(matrix));
      if ( MLI_CSRBooleanMatrix_Get_OwnsData(matrix) )
         hypre_TFree(MLI_CSRBooleanMatrix_Get_J(matrix));
      hypre_TFree(matrix);
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_CSRBooleanMatrixInitialize
 *--------------------------------------------------------------------------*/

int MLI_CSRBooleanMatrixInitialize( MLI_CSRBooleanMatrix *matrix )
{
   int  num_rows     = MLI_CSRBooleanMatrix_Get_NRows(matrix);
   int  num_nonzeros = MLI_CSRBooleanMatrix_Get_NNZ(matrix);

   if ( ! MLI_CSRBooleanMatrix_Get_I(matrix) )
      MLI_CSRBooleanMatrix_Get_I(matrix) = hypre_CTAlloc(int, num_rows + 1);
   if ( ! MLI_CSRBooleanMatrix_Get_J(matrix) )
      MLI_CSRBooleanMatrix_Get_J(matrix) = hypre_CTAlloc(int, num_nonzeros);

   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_CSRBooleanMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

int MLI_CSRBooleanMatrixSetDataOwner( MLI_CSRBooleanMatrix *matrix,
                                      int owns_data )
{
   MLI_CSRBooleanMatrix_Get_OwnsData(matrix) = owns_data;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_CSRBooleanMatrixRead
 *--------------------------------------------------------------------------*/

MLI_CSRBooleanMatrix *
MLI_CSRBooleanMatrixRead( char *file_name )
{
   MLI_CSRBooleanMatrix  *matrix;

   FILE    *fp;

   int     *matrix_i;
   int     *matrix_j;
   int      num_rows;
   int      num_nonzeros;
   int      max_col = 0;

   int      file_base = 1;
   
   int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &num_rows);

   matrix_i = hypre_CTAlloc(int, num_rows + 1);
   for (j = 0; j < num_rows+1; j++)
   {
      fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = MLI_CSRBooleanMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   MLI_CSRBooleanMatrix_Get_I(matrix) = matrix_i;
   MLI_CSRBooleanMatrixInitialize(matrix);

   matrix_j = MLI_CSRBooleanMatrix_Get_J(matrix);
   for (j = 0; j < num_nonzeros; j++)
   {
      fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   fclose(fp);

   MLI_CSRBooleanMatrix_Get_NNZ(matrix) = num_nonzeros;
   MLI_CSRBooleanMatrix_Get_NCols(matrix) = ++max_col;

   return matrix;
}

/*--------------------------------------------------------------------------
 * MLI_CSRBooleanMatrixPrint
 *--------------------------------------------------------------------------*/

int
MLI_CSRBooleanMatrixPrint( MLI_CSRBooleanMatrix *matrix,
                           char            *file_name )
{
   FILE    *fp;

   int     *matrix_i;
   int     *matrix_j;
   int      num_rows;
   
   int      file_base = 1;
   
   int      j;

   int      ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   matrix_i    = MLI_CSRBooleanMatrix_Get_I(matrix);
   matrix_j    = MLI_CSRBooleanMatrix_Get_J(matrix);
   num_rows    = MLI_CSRBooleanMatrix_Get_NRows(matrix);

   fp = fopen(file_name, "w");

   fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j++)
   {
      fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      fprintf(fp, "%d\n", matrix_j[j] + file_base);
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixCreate
 *--------------------------------------------------------------------------*/

MLI_ParCSRBooleanMatrix *MLI_ParCSRBooleanMatrixCreate( MPI_Comm comm,
                               int global_num_rows, int global_num_cols,
                               int *row_starts, int *col_starts,
                               int num_cols_offd, int num_nonzeros_diag,
                               int num_nonzeros_offd)
{
   MLI_ParCSRBooleanMatrix *matrix;
   int                     num_procs, my_id;
   int                     local_num_rows, local_num_cols;
   int                     first_row_index, first_col_diag;
   
   matrix = hypre_CTAlloc(MLI_ParCSRBooleanMatrix, 1);

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
   MLI_ParCSRBooleanMatrix_Get_Comm(matrix) = comm;
   MLI_ParCSRBooleanMatrix_Get_Diag(matrix) = 
          MLI_CSRBooleanMatrixCreate(local_num_rows, local_num_cols,
                                     num_nonzeros_diag);
   MLI_ParCSRBooleanMatrix_Get_Offd(matrix) = 
          MLI_CSRBooleanMatrixCreate(local_num_rows, num_cols_offd,
                                     num_nonzeros_offd);
   MLI_ParCSRBooleanMatrix_Get_GlobalNRows(matrix) = global_num_rows;
   MLI_ParCSRBooleanMatrix_Get_GlobalNCols(matrix) = global_num_cols;
   MLI_ParCSRBooleanMatrix_Get_StartRow(matrix) = first_row_index;
   MLI_ParCSRBooleanMatrix_Get_FirstColDiag(matrix) = first_col_diag;
   MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = NULL;
   MLI_ParCSRBooleanMatrix_Get_RowStarts(matrix) = row_starts;
   MLI_ParCSRBooleanMatrix_Get_ColStarts(matrix) = col_starts;
   MLI_ParCSRBooleanMatrix_Get_CommPkg(matrix) = NULL;

   MLI_ParCSRBooleanMatrix_Get_OwnsData(matrix)      = 1;
   MLI_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = 1;
   MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
      MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 0;

   MLI_ParCSRBooleanMatrix_Get_Rowindices(matrix)   = NULL;
   MLI_ParCSRBooleanMatrix_Get_Getrowactive(matrix) = 0;

   return matrix;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixDestroy
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixDestroy( MLI_ParCSRBooleanMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      if ( MLI_ParCSRBooleanMatrix_Get_OwnsData(matrix) )
      {
         MLI_CSRBooleanMatrixDestroy(MLI_ParCSRBooleanMatrix_Get_Diag(matrix));
         MLI_CSRBooleanMatrixDestroy(MLI_ParCSRBooleanMatrix_Get_Offd(matrix));
         if (MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix))
            hypre_TFree(MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix));
         if (MLI_ParCSRBooleanMatrix_Get_CommPkg(matrix))
            hypre_MatvecCommPkgDestroy(MLI_ParCSRBooleanMatrix_Get_CommPkg(matrix));
      }
      if ( MLI_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) )
         hypre_TFree(MLI_ParCSRBooleanMatrix_Get_RowStarts(matrix));
      if ( MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) )
         hypre_TFree(MLI_ParCSRBooleanMatrix_Get_ColStarts(matrix));

      hypre_TFree(MLI_ParCSRBooleanMatrix_Get_Rowindices(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixInitialize
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixInitialize( MLI_ParCSRBooleanMatrix *matrix )
{
   int  ierr=0;

   MLI_CSRBooleanMatrixInitialize(MLI_ParCSRBooleanMatrix_Get_Diag(matrix));
   MLI_CSRBooleanMatrixInitialize(MLI_ParCSRBooleanMatrix_Get_Offd(matrix));
   MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = 
                hypre_CTAlloc(int,MLI_CSRBooleanMatrix_Get_NCols(
                MLI_ParCSRBooleanMatrix_Get_Offd(matrix)));
   return ierr;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixSetNNZ
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixSetNNZ( MLI_ParCSRBooleanMatrix *matrix)
{
   MPI_Comm comm = MLI_ParCSRBooleanMatrix_Get_Comm(matrix);
   MLI_CSRBooleanMatrix *diag = MLI_ParCSRBooleanMatrix_Get_Diag(matrix);
   int *diag_i = MLI_CSRBooleanMatrix_Get_I(diag);
   MLI_CSRBooleanMatrix *offd = MLI_ParCSRBooleanMatrix_Get_Offd(matrix);
   int *offd_i = MLI_CSRBooleanMatrix_Get_I(offd);
   int local_num_rows = MLI_CSRBooleanMatrix_Get_NRows(diag);
   int total_num_nonzeros;
   int local_num_nonzeros;
   int ierr = 0;

   local_num_nonzeros = diag_i[local_num_rows] + offd_i[local_num_rows];
   MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, MPI_INT,
        MPI_SUM, comm);
   MLI_ParCSRBooleanMatrix_Get_NNZ(matrix) = total_num_nonzeros;
   return ierr;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixSetDataOwner(MLI_ParCSRBooleanMatrix *matrix,
                                        int owns_data )
{
   MLI_ParCSRBooleanMatrix_Get_OwnsData(matrix) = owns_data;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixSetRowStartsOwner
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixSetRowStartsOwner(MLI_ParCSRBooleanMatrix *matrix,
                                             int owns_row_starts )
{
   MLI_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = owns_row_starts;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixSetColStartsOwner
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixSetColStartsOwner(MLI_ParCSRBooleanMatrix *matrix,
                                             int owns_col_starts )
{
   MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = owns_col_starts;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixRead
 *--------------------------------------------------------------------------*/

MLI_ParCSRBooleanMatrix *
MLI_ParCSRBooleanMatrixRead( MPI_Comm comm, char *file_name )
{
   MLI_ParCSRBooleanMatrix  *matrix;
   MLI_CSRBooleanMatrix  *diag;
   MLI_CSRBooleanMatrix  *offd;
   int  my_id, i, num_procs;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   int  global_num_rows, global_num_cols, num_cols_offd;
   int  local_num_rows;
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
   
   diag = MLI_CSRBooleanMatrixRead(new_file_d);
   local_num_rows = MLI_CSRBooleanMatrix_Get_NRows(diag);

   if (num_cols_offd)
   {
        offd = MLI_CSRBooleanMatrixRead(new_file_o);
   }
   else
        offd = MLI_CSRBooleanMatrixCreate(local_num_rows,0,0);

        
   matrix = hypre_CTAlloc(MLI_ParCSRBooleanMatrix, 1);
   
   MLI_ParCSRBooleanMatrix_Get_Comm(matrix) = comm;
   MLI_ParCSRBooleanMatrix_Get_GlobalNRows(matrix) = global_num_rows;
   MLI_ParCSRBooleanMatrix_Get_GlobalNCols(matrix) = global_num_cols;
   MLI_ParCSRBooleanMatrix_Get_StartRow(matrix) = row_starts[my_id];
   MLI_ParCSRBooleanMatrix_Get_FirstColDiag(matrix) = col_starts[my_id];
   MLI_ParCSRBooleanMatrix_Get_RowStarts(matrix) = row_starts;
   MLI_ParCSRBooleanMatrix_Get_ColStarts(matrix) = col_starts;
   MLI_ParCSRBooleanMatrix_Get_CommPkg(matrix) = NULL;

   /* set defaults */
   MLI_ParCSRBooleanMatrix_Get_OwnsData(matrix) = 1;
   MLI_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = 1;
   MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
        MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 0;

   MLI_ParCSRBooleanMatrix_Get_Diag(matrix) = diag;
   MLI_ParCSRBooleanMatrix_Get_Offd(matrix) = offd;
   if (num_cols_offd)
        MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = col_map_offd;
   else
        MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = NULL;

   return matrix;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixPrint
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixPrint( MLI_ParCSRBooleanMatrix *matrix, 
                                  char *file_name )
{
   MPI_Comm comm = MLI_ParCSRBooleanMatrix_Get_Comm(matrix);
   int global_num_rows = MLI_ParCSRBooleanMatrix_Get_GlobalNRows(matrix);
   int global_num_cols = MLI_ParCSRBooleanMatrix_Get_GlobalNCols(matrix);
   int *col_map_offd = MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);
   int *row_starts = MLI_ParCSRBooleanMatrix_Get_RowStarts(matrix);
   int *col_starts = MLI_ParCSRBooleanMatrix_Get_ColStarts(matrix);
   int  my_id, i, num_procs;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   int  ierr = 0;
   FILE *fp;
   int  num_cols_offd = 0;

   if (MLI_ParCSRBooleanMatrix_Get_Offd(matrix)) num_cols_offd = 
      MLI_CSRBooleanMatrix_Get_NCols(MLI_ParCSRBooleanMatrix_Get_Offd(matrix));

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);
   
   sprintf(new_file_d,"%s.D.%d",file_name,my_id);
   sprintf(new_file_o,"%s.O.%d",file_name,my_id);
   sprintf(new_file_info,"%s.INFO.%d",file_name,my_id);
   MLI_CSRBooleanMatrixPrint(MLI_ParCSRBooleanMatrix_Get_Diag(matrix),new_file_d);
   if (num_cols_offd != 0)
      MLI_CSRBooleanMatrixPrint(MLI_ParCSRBooleanMatrix_Get_Offd(matrix),
                                new_file_o);
  
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
 * MLI_ParCSRBooleanMatrixPrintIJ
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixPrintIJ( MLI_ParCSRBooleanMatrix *matrix, 
                                    char *filename )
{
   MPI_Comm comm = MLI_ParCSRBooleanMatrix_Get_Comm(matrix);
   int      global_num_rows = MLI_ParCSRBooleanMatrix_Get_GlobalNRows(matrix);
   int      global_num_cols = MLI_ParCSRBooleanMatrix_Get_GlobalNCols(matrix);
   int      first_row_index = MLI_ParCSRBooleanMatrix_Get_StartRow(matrix);
   int      first_col_diag  = MLI_ParCSRBooleanMatrix_Get_FirstColDiag(matrix);
   int     *col_map_offd    = MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);
   int      num_rows        = MLI_ParCSRBooleanMatrix_Get_NRows(matrix);
   int     *diag_i;
   int     *diag_j;
   int     *offd_i;
   int     *offd_j;
   int      myid, i, j, I, J;
   int      ierr = 0;
   char     new_filename[255];
   FILE    *file;
   MLI_CSRBooleanMatrix *diag = MLI_ParCSRBooleanMatrix_Get_Diag(matrix);
   MLI_CSRBooleanMatrix *offd = MLI_ParCSRBooleanMatrix_Get_Offd(matrix);

   MPI_Comm_rank(comm, &myid);
   
   sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   fprintf(file, "%d, %d\n", global_num_rows, global_num_cols);
   fprintf(file, "%d\n", num_rows);

   diag_i    = MLI_CSRBooleanMatrix_Get_I(diag);
   diag_j    = MLI_CSRBooleanMatrix_Get_J(diag);
   if (offd)
   {
      offd_i    = MLI_CSRBooleanMatrix_Get_I(offd);
      offd_j    = MLI_CSRBooleanMatrix_Get_J(offd);
   }
   for (i = 0; i < num_rows; i++)
   {
      I = first_row_index + i;

      /* print diag columns */
      for (j = diag_i[i]; j < diag_i[i+1]; j++)
      {
         J = first_col_diag + diag_j[j];
         fprintf(file, "%d, %d\n", I, J );
      }

      /* print offd columns */
      if (offd)
      {
         for (j = offd_i[i]; j < offd_i[i+1]; j++)
         {
            J = col_map_offd[offd_j[j]];
            fprintf(file, "%d, %d \n", I, J);
         }
      }
   }

   fclose(file);

   return ierr;
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixGetLocalRange
 * returns the row numbers of the rows stored on this processor.
 * "End" is actually the row number of the last row on this processor.
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixGetLocalRange(MLI_ParCSRBooleanMatrix *matrix,
                                         int *row_start, int *row_end,
                                         int *col_start, int *col_end )
{  
   int ierr=0;
   int my_id;

   MPI_Comm_rank( MLI_ParCSRBooleanMatrix_Get_Comm(matrix), &my_id );

   *row_start = MLI_ParCSRBooleanMatrix_Get_RowStarts(matrix)[ my_id ];
   *row_end   = MLI_ParCSRBooleanMatrix_Get_RowStarts(matrix)[ my_id + 1 ]-1;
   *col_start = MLI_ParCSRBooleanMatrix_Get_ColStarts(matrix)[ my_id ];
   *col_end   = MLI_ParCSRBooleanMatrix_Get_ColStarts(matrix)[ my_id + 1 ]-1;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixGetRow
 * Returns global column indices for a given row in the global matrix.
 * Global row number is used, but the row must be stored locally or
 * an error is returned. This implementation copies from the two matrices that
 * store the local data, storing them in the MLI_ParCSRBooleanMatrix structure.
 * Only a single row can be accessed via this function at any one time; the
 * corresponding RestoreRow function must be called, to avoid bleeding memory,
 * and to be able to look at another row.  All indices are returned in 0-based 
 * indexing, no matter what is used under the hood. 
 * EXCEPTION: currently this only works if the local CSR matrices
 * use 0-based indexing.
 * This code, semantics, implementation, etc., are all based on PETSc's MPI_AIJ
 * matrix code, adjusted for our data and software structures.
 * AJC 4/99.
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixGetRow(MLI_ParCSRBooleanMatrix  *mat,
                                  int row, int *size, int **col_ind)
{  
   int    i, m, ierr=0, max=1, tmp, my_id, row_start, row_end;
   int    *cworkA, *cworkB; 
   int    cstart, nztot, nzA, nzB, lrow;
   int    *cmap, *idx_p;
   MLI_CSRBooleanMatrix *Aa, *Ba;

   Aa = (MLI_CSRBooleanMatrix *) MLI_ParCSRBooleanMatrix_Get_Diag(mat);
   Ba = (MLI_CSRBooleanMatrix *) MLI_ParCSRBooleanMatrix_Get_Offd(mat);
   
   if (MLI_ParCSRBooleanMatrix_Get_Getrowactive(mat)) return(-1);

   MPI_Comm_rank( MLI_ParCSRBooleanMatrix_Get_Comm(mat), &my_id );

   MLI_ParCSRBooleanMatrix_Get_Getrowactive(mat) = 1;

   row_end   = MLI_ParCSRBooleanMatrix_Get_RowStarts(mat)[ my_id + 1 ];
   row_start = MLI_ParCSRBooleanMatrix_Get_RowStarts(mat)[ my_id ];
   lrow      = row - row_start;

   if (row < row_start || row >= row_end) return(-1);

   if ( col_ind ) 
   {
      m = row_end-row_start;
      for ( i=0; i<m; i++ ) 
      {
        tmp = MLI_CSRBooleanMatrix_Get_I(Aa)[i+1] - 
              MLI_CSRBooleanMatrix_Get_I(Aa)[i] + 
              MLI_CSRBooleanMatrix_Get_I(Ba)[i+1] - 
              MLI_CSRBooleanMatrix_Get_I(Ba)[i];
        if (max < tmp) { max = tmp; }
      }
      MLI_ParCSRBooleanMatrix_Get_Rowindices(mat) = (int *) hypre_CTAlloc(int,max); 
   }

   cstart = MLI_ParCSRBooleanMatrix_Get_FirstColDiag(mat);

   nzA = MLI_CSRBooleanMatrix_Get_I(Aa)[lrow+1] -
         MLI_CSRBooleanMatrix_Get_I(Aa)[lrow];
   cworkA= &(MLI_CSRBooleanMatrix_Get_J(Aa)[MLI_CSRBooleanMatrix_Get_I(Aa)[lrow]]);

   nzB = MLI_CSRBooleanMatrix_Get_I(Ba)[lrow+1] -
         MLI_CSRBooleanMatrix_Get_I(Ba)[lrow];
   cworkB= &(MLI_CSRBooleanMatrix_Get_J(Ba)[MLI_CSRBooleanMatrix_Get_I(Ba)[lrow]]);

   nztot = nzA + nzB;

   cmap  = MLI_ParCSRBooleanMatrix_Get_ColMapOffd(mat);

   if (col_ind) 
   {
      if (nztot) 
      {
         int imark = -1;
         if (col_ind) 
         {
            *col_ind = idx_p = MLI_ParCSRBooleanMatrix_Get_Rowindices(mat);
            if (imark > -1) 
            {
               for ( i=0; i<imark; i++ ) idx_p[i] = cmap[cworkB[i]];
            } 
            else 
            {
               for ( i=0; i<nzB; i++ ) 
               {
                  if (cmap[cworkB[i]] < cstart) idx_p[i] = cmap[cworkB[i]];
                  else break;
               }
               imark = i;
            }
            for ( i=0; i<nzA; i++ )     idx_p[imark+i] = cstart + cworkA[i];
            for ( i=imark; i<nzB; i++ ) idx_p[nzA+i]   = cmap[cworkB[i]];
         } 
      } 
      else 
      {
         if (col_ind) *col_ind = 0; 
      }
   }
   *size = nztot;
   return( ierr );
}

/*--------------------------------------------------------------------------
 * MLI_ParCSRBooleanMatrixRestoreRow
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixRestoreRow( MLI_ParCSRBooleanMatrix *matrix,
                                       int row, int *size, int **col_ind)
{  

   if (!MLI_ParCSRBooleanMatrix_Get_Getrowactive(matrix)) return( -1 );

   MLI_ParCSRBooleanMatrix_Get_Getrowactive(matrix) = 0;

   return( 0 );
}


/*--------------------------------------------------------------------------
 * MLI_BuildCSRBooleanMatrixMPIDataType
 *--------------------------------------------------------------------------*/

int
MLI_BuildCSRBooleanMatrixMPIDataType(
   int num_nonzeros, int num_rows, int *a_i, int *a_j, 
   MPI_Datatype *csr_matrix_datatype )
{
   int		block_lens[2];
   MPI_Aint	displ[2];
   MPI_Datatype	types[2];
   int		ierr = 0;

   block_lens[0] = num_rows+1;
   block_lens[1] = num_nonzeros;

   types[0] = MPI_INT;
   types[1] = MPI_INT;

   MPI_Address(a_i, &displ[0]);
   MPI_Address(a_j, &displ[1]);
   MPI_Type_struct(2,block_lens,displ,types,csr_matrix_datatype);
   MPI_Type_commit(csr_matrix_datatype);

   return ierr;
}

/*--------------------------------------------------------------------------
 * MLI_CSRBooleanMatrixToParCSRBooleanMatrix:
 * generates a ParCSRBooleanMatrix distributed across the processors in comm
 * from a CSRBooleanMatrix on proc 0 .
 *--------------------------------------------------------------------------*/

MLI_ParCSRBooleanMatrix *
MLI_CSRBooleanMatrixToParCSRBooleanMatrix
( MPI_Comm comm, MLI_CSRBooleanMatrix *A,
  int *row_starts, int *col_starts )
{
   int          global_data[2];
   int          global_num_rows;
   int          global_num_cols;
   int          *local_num_rows;

   int          num_procs, my_id;
   int          *local_num_nonzeros;
   int          num_nonzeros;
  
   int          *a_i;
   int          *a_j;
  
   MLI_CSRBooleanMatrix *local_A;

   MPI_Request  *requests;
   MPI_Status   *status, status0;
   MPI_Datatype *csr_matrix_datatypes;

   MLI_ParCSRBooleanMatrix *par_matrix;

   int          first_col_diag;
   int          last_col_diag;
 
   int i, j, ind;

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);

   if (my_id == 0) 
   {
        global_data[0] = MLI_CSRBooleanMatrix_Get_NRows(A);
        global_data[1] = MLI_CSRBooleanMatrix_Get_NCols(A);
        a_i = MLI_CSRBooleanMatrix_Get_I(A);
        a_j = MLI_CSRBooleanMatrix_Get_J(A);
   }
   MPI_Bcast(global_data,2,MPI_INT,0,comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];

   local_num_rows = hypre_CTAlloc(int, num_procs);
   csr_matrix_datatypes = hypre_CTAlloc(MPI_Datatype, num_procs);

   par_matrix = MLI_ParCSRBooleanMatrixCreate (comm, global_num_rows,
        global_num_cols,row_starts,col_starts,0,0,0);

   row_starts = MLI_ParCSRBooleanMatrix_Get_RowStarts(par_matrix);
   col_starts = MLI_ParCSRBooleanMatrix_Get_ColStarts(par_matrix);

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

   local_A = MLI_CSRBooleanMatrixCreate(local_num_rows[my_id], global_num_cols,
                num_nonzeros);
   if (my_id == 0)
   {
        requests = hypre_CTAlloc (MPI_Request, num_procs-1);
        status = hypre_CTAlloc(MPI_Status, num_procs-1);
        j=0;
        for (i=1; i < num_procs; i++)
        {
                ind = a_i[row_starts[i]];
                MLI_BuildCSRBooleanMatrixMPIDataType(local_num_nonzeros[i], 
                        local_num_rows[i],
                        &a_i[row_starts[i]],
                        &a_j[ind],
                        &csr_matrix_datatypes[i]);
                MPI_Isend(MPI_BOTTOM, 1, csr_matrix_datatypes[i], i, 0, comm,
                        &requests[j++]);
                MPI_Type_free(&csr_matrix_datatypes[i]);
        }
        MLI_CSRBooleanMatrix_Get_I(local_A) = a_i;
        MLI_CSRBooleanMatrix_Get_J(local_A) = a_j;
        MPI_Waitall(num_procs-1,requests,status);
        hypre_TFree(requests);
        hypre_TFree(status);
        hypre_TFree(local_num_nonzeros);
    }
   else
   {
        MLI_CSRBooleanMatrixInitialize(local_A);
        MLI_BuildCSRBooleanMatrixMPIDataType(num_nonzeros, 
                        local_num_rows[my_id],
                        MLI_CSRBooleanMatrix_Get_I(local_A),
                        MLI_CSRBooleanMatrix_Get_J(local_A),
                        csr_matrix_datatypes);
        MPI_Recv(MPI_BOTTOM,1,csr_matrix_datatypes[0],0,0,comm,&status0);
        MPI_Type_free(csr_matrix_datatypes);
   }

   first_col_diag = col_starts[my_id];
   last_col_diag = col_starts[my_id+1]-1;

   BooleanGenerateDiagAndOffd(local_A, par_matrix, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {      
      MLI_CSRBooleanMatrix_Get_I(local_A) = NULL;
      MLI_CSRBooleanMatrix_Get_J(local_A) = NULL; 
   }      
   MLI_CSRBooleanMatrixDestroy(local_A);
   hypre_TFree(local_num_rows);
   hypre_TFree(csr_matrix_datatypes);

   return par_matrix;
}

int
BooleanGenerateDiagAndOffd(MLI_CSRBooleanMatrix *A,
                    MLI_ParCSRBooleanMatrix *matrix,
                    int first_col_diag,
                    int last_col_diag)
{
   int  i, j;
   int  jo, jd;
   int  ierr = 0;
   int  num_rows = MLI_CSRBooleanMatrix_Get_NRows(A);
   int  num_cols = MLI_CSRBooleanMatrix_Get_NCols(A);
   int *a_i = MLI_CSRBooleanMatrix_Get_I(A);
   int *a_j = MLI_CSRBooleanMatrix_Get_J(A);

   MLI_CSRBooleanMatrix *diag = MLI_ParCSRBooleanMatrix_Get_Diag(matrix);
   MLI_CSRBooleanMatrix *offd = MLI_ParCSRBooleanMatrix_Get_Offd(matrix);

   int  *col_map_offd;

   int  *diag_i, *offd_i;
   int  *diag_j, *offd_j;
   int  *marker;
   int num_cols_diag, num_cols_offd;
   int first_elmt = a_i[0];
   int num_nonzeros = a_i[num_rows]-first_elmt;
   int counter;

   num_cols_diag = last_col_diag - first_col_diag +1;
   num_cols_offd = 0;

   if (num_cols - num_cols_diag)
   {
        MLI_CSRBooleanMatrixInitialize(diag);
        diag_i = MLI_CSRBooleanMatrix_Get_I(diag);

        MLI_CSRBooleanMatrixInitialize(offd);
        offd_i = MLI_CSRBooleanMatrix_Get_I(offd);
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

        MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) =
           hypre_CTAlloc(int,num_cols_offd);
        col_map_offd = MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);

        counter = 0;
        for (i=0; i < num_cols; i++)
                if (marker[i])
                {
                        col_map_offd[counter] = i;
                        marker[i] = counter;
                        counter++;
                }

        MLI_CSRBooleanMatrix_Get_NNZ(diag) = jd;
        MLI_CSRBooleanMatrixInitialize(diag);
        diag_j = MLI_CSRBooleanMatrix_Get_J(diag);

        MLI_CSRBooleanMatrix_Get_NNZ(offd) = jo;
        MLI_CSRBooleanMatrix_Get_NCols(offd) = num_cols_offd;
        MLI_CSRBooleanMatrixInitialize(offd);
        offd_j = MLI_CSRBooleanMatrix_Get_J(offd);

        jo = 0;
        jd = 0;
        for (i=0; i < num_rows; i++)
        {
            for (j=a_i[i]-first_elmt; j < a_i[i+1]-first_elmt; j++)
                if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
                {
                        offd_j[jo++] = marker[a_j[j]];
                }
                else
                {
                        diag_j[jd++] = a_j[j]-first_col_diag;
                }
        }
        hypre_TFree(marker);
   }
   else 
   {
        MLI_CSRBooleanMatrix_Get_NNZ(diag) = num_nonzeros;
        MLI_CSRBooleanMatrixInitialize(diag);
        diag_i = MLI_CSRBooleanMatrix_Get_I(diag);
        diag_j = MLI_CSRBooleanMatrix_Get_J(diag);

        for (i=0; i < num_nonzeros; i++)
        {
                diag_j[i] = a_j[i];
        }
        offd_i = hypre_CTAlloc(int, num_rows+1);

        for (i=0; i < num_rows+1; i++)
        {
                diag_i[i] = a_i[i];
                offd_i[i] = 0;
        }

        MLI_CSRBooleanMatrix_Get_NCols(offd) = 0;
        MLI_CSRBooleanMatrix_Get_I(offd) = offd_i;
   }
   
   return ierr;
}


