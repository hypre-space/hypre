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

#include "mli_pcsr_bool_mat.h"
#include "utilities.h"

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
      if ( MLI_CSRMatrix_Get_OwnsData(matrix) )
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
      if ( MLI_ParCSRMatrix_Get_OwnsData(matrix) )
      {
         MLI_CSRMatrixBooleanDestroy(MLI_ParCSRMatrixBoolean_Get_Diag(matrix));
         MLI_CSRMatrixBooleanDestroy(MLI_ParCSRMatrixBoolean_Get_Offd(matrix));
         if (MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix))
            hypre_TFree(MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix));
         if (MLI_ParCSRBooleanMatrix_Get_CommPkg(matrix))
            MLI_MatvecCommPkgDestroy(MLI_ParCSRBooleanMatrix_Get_CommPkg(matrix));
      }
      if ( MLI_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) )
         MLI_TFree(MLI_ParCSRBooleanMatrix_Get_RowStarts(matrix));
      if ( MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) )
         MLI_TFree(MLI_ParCSRBooleanMatrix_Get_ColStarts(matrix));

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
   MPI_Comm comm = MLI_ParCSRMatrix_Get_Comm(matrix);
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
 * MLI_ParCSRMatrixSetColStartsOwner
 *--------------------------------------------------------------------------*/

int MLI_ParCSRBooleanMatrixSetColStartsOwner(MLI_ParCSRBooleanMatrix *matrix,
                                             int owns_col_starts )
{
   MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = owns_col_starts;
   return 0;
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
   MPI_Comm comm = MLI_ParCSRMatrix_Get_Comm(matrix);
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

   cstart = MLI_ParCSRMatrixFirstColDiag(mat);

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

