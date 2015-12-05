/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Member functions for hypre_CSRBooleanMatrix and hypre_ParCSRBooleanMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CSRBooleanMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate(HYPRE_Int num_rows,HYPRE_Int num_cols,
                                                 HYPRE_Int num_nonzeros )
{
   hypre_CSRBooleanMatrix *matrix;

   matrix = hypre_CTAlloc(hypre_CSRBooleanMatrix, 1);

   hypre_CSRBooleanMatrix_Get_I(matrix)     = NULL;
   hypre_CSRBooleanMatrix_Get_J(matrix)     = NULL;
   hypre_CSRBooleanMatrix_Get_NRows(matrix) = num_rows;
   hypre_CSRBooleanMatrix_Get_NCols(matrix) = num_cols;
   hypre_CSRBooleanMatrix_Get_NNZ(matrix)   = num_nonzeros;
   hypre_CSRBooleanMatrix_Get_OwnsData(matrix) = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBooleanMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_CSRBooleanMatrixDestroy( hypre_CSRBooleanMatrix *matrix )
{
   if (matrix)
   {
      hypre_TFree(hypre_CSRBooleanMatrix_Get_I(matrix));
      if ( hypre_CSRBooleanMatrix_Get_OwnsData(matrix) )
         hypre_TFree(hypre_CSRBooleanMatrix_Get_J(matrix));
      hypre_TFree(matrix);
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBooleanMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_CSRBooleanMatrixInitialize( hypre_CSRBooleanMatrix *matrix )
{
   HYPRE_Int  num_rows     = hypre_CSRBooleanMatrix_Get_NRows(matrix);
   HYPRE_Int  num_nonzeros = hypre_CSRBooleanMatrix_Get_NNZ(matrix);

   if ( ! hypre_CSRBooleanMatrix_Get_I(matrix) )
      hypre_CSRBooleanMatrix_Get_I(matrix) = hypre_CTAlloc(HYPRE_Int, num_rows + 1);
   if ( ! hypre_CSRBooleanMatrix_Get_J(matrix) )
      hypre_CSRBooleanMatrix_Get_J(matrix) = hypre_CTAlloc(HYPRE_Int, num_nonzeros);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBooleanMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_CSRBooleanMatrixSetDataOwner( hypre_CSRBooleanMatrix *matrix,
                                      HYPRE_Int owns_data )
{
   hypre_CSRBooleanMatrix_Get_OwnsData(matrix) = owns_data;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBooleanMatrixRead
 *--------------------------------------------------------------------------*/

hypre_CSRBooleanMatrix *
hypre_CSRBooleanMatrixRead( const char *file_name )
{
   hypre_CSRBooleanMatrix  *matrix;

   FILE    *fp;

   HYPRE_Int     *matrix_i;
   HYPRE_Int     *matrix_j;
   HYPRE_Int      num_rows;
   HYPRE_Int      num_nonzeros;
   HYPRE_Int      max_col = 0;

   HYPRE_Int      file_base = 1;
   
   HYPRE_Int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   hypre_fscanf(fp, "%d", &num_rows);

   matrix_i = hypre_CTAlloc(HYPRE_Int, num_rows + 1);
   for (j = 0; j < num_rows+1; j++)
   {
      hypre_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CSRBooleanMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   hypre_CSRBooleanMatrix_Get_I(matrix) = matrix_i;
   hypre_CSRBooleanMatrixInitialize(matrix);

   matrix_j = hypre_CSRBooleanMatrix_Get_J(matrix);
   for (j = 0; j < num_nonzeros; j++)
   {
      hypre_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   fclose(fp);

   hypre_CSRBooleanMatrix_Get_NNZ(matrix) = num_nonzeros;
   hypre_CSRBooleanMatrix_Get_NCols(matrix) = ++max_col;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBooleanMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRBooleanMatrixPrint( hypre_CSRBooleanMatrix *matrix,
                             const char             *file_name )
{
   FILE    *fp;

   HYPRE_Int     *matrix_i;
   HYPRE_Int     *matrix_j;
   HYPRE_Int      num_rows;
   
   HYPRE_Int      file_base = 1;
   
   HYPRE_Int      j;

   HYPRE_Int      ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   matrix_i    = hypre_CSRBooleanMatrix_Get_I(matrix);
   matrix_j    = hypre_CSRBooleanMatrix_Get_J(matrix);
   num_rows    = hypre_CSRBooleanMatrix_Get_NRows(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j++)
   {
      hypre_fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      hypre_fprintf(fp, "%d\n", matrix_j[j] + file_base);
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate( MPI_Comm comm,
                               HYPRE_Int global_num_rows, HYPRE_Int global_num_cols,
                               HYPRE_Int *row_starts, HYPRE_Int *col_starts,
                               HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag,
                               HYPRE_Int num_nonzeros_offd)
{
   hypre_ParCSRBooleanMatrix *matrix;
   HYPRE_Int                     num_procs, my_id;
   HYPRE_Int                     local_num_rows, local_num_cols;
   HYPRE_Int                     first_row_index, first_col_diag;
   
   matrix = hypre_CTAlloc(hypre_ParCSRBooleanMatrix, 1);

   hypre_MPI_Comm_rank(comm,&my_id);
   hypre_MPI_Comm_size(comm,&num_procs);

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
   hypre_ParCSRBooleanMatrix_Get_Comm(matrix) = comm;
   hypre_ParCSRBooleanMatrix_Get_Diag(matrix) = 
          hypre_CSRBooleanMatrixCreate(local_num_rows, local_num_cols,
                                     num_nonzeros_diag);
   hypre_ParCSRBooleanMatrix_Get_Offd(matrix) = 
          hypre_CSRBooleanMatrixCreate(local_num_rows, num_cols_offd,
                                     num_nonzeros_offd);
   hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix) = global_num_rows;
   hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix) = global_num_cols;
   hypre_ParCSRBooleanMatrix_Get_StartRow(matrix) = first_row_index;
   hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix) = first_col_diag;
   hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = NULL;
   hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix) = row_starts;
   hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix) = col_starts;
   hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix) = NULL;

   hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix)      = 1;
   hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = 1;
   hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
      hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 0;

   hypre_ParCSRBooleanMatrix_Get_Rowindices(matrix)   = NULL;
   hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix) = 0;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixDestroy( hypre_ParCSRBooleanMatrix *matrix )
{
   HYPRE_Int  ierr=0;

   if (matrix)
   {
      if ( hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix) )
      {
         hypre_CSRBooleanMatrixDestroy(hypre_ParCSRBooleanMatrix_Get_Diag(matrix));
         hypre_CSRBooleanMatrixDestroy(hypre_ParCSRBooleanMatrix_Get_Offd(matrix));
         if (hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix))
            hypre_TFree(hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix));
         if (hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix))
            hypre_MatvecCommPkgDestroy(hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix));
      }
      if ( hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) )
         hypre_TFree(hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix));
      if ( hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) )
         hypre_TFree(hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix));

      hypre_TFree(hypre_ParCSRBooleanMatrix_Get_Rowindices(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixInitialize( hypre_ParCSRBooleanMatrix *matrix )
{
   HYPRE_Int  ierr=0;

   hypre_CSRBooleanMatrixInitialize(hypre_ParCSRBooleanMatrix_Get_Diag(matrix));
   hypre_CSRBooleanMatrixInitialize(hypre_ParCSRBooleanMatrix_Get_Offd(matrix));
   hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = 
                hypre_CTAlloc(HYPRE_Int,hypre_CSRBooleanMatrix_Get_NCols(
                hypre_ParCSRBooleanMatrix_Get_Offd(matrix)));
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixSetNNZ
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixSetNNZ( hypre_ParCSRBooleanMatrix *matrix)
{
   MPI_Comm comm = hypre_ParCSRBooleanMatrix_Get_Comm(matrix);
   hypre_CSRBooleanMatrix *diag = hypre_ParCSRBooleanMatrix_Get_Diag(matrix);
   HYPRE_Int *diag_i = hypre_CSRBooleanMatrix_Get_I(diag);
   hypre_CSRBooleanMatrix *offd = hypre_ParCSRBooleanMatrix_Get_Offd(matrix);
   HYPRE_Int *offd_i = hypre_CSRBooleanMatrix_Get_I(offd);
   HYPRE_Int local_num_rows = hypre_CSRBooleanMatrix_Get_NRows(diag);
   HYPRE_Int total_num_nonzeros;
   HYPRE_Int local_num_nonzeros;
   HYPRE_Int ierr = 0;

   local_num_nonzeros = diag_i[local_num_rows] + offd_i[local_num_rows];
   hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, HYPRE_MPI_INT,
        hypre_MPI_SUM, comm);
   hypre_ParCSRBooleanMatrix_Get_NNZ(matrix) = total_num_nonzeros;
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixSetDataOwner(hypre_ParCSRBooleanMatrix *matrix,
                                        HYPRE_Int owns_data )
{
   hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix) = owns_data;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixSetRowStartsOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixSetRowStartsOwner(hypre_ParCSRBooleanMatrix *matrix,
                                             HYPRE_Int owns_row_starts )
{
   hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = owns_row_starts;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixSetColStartsOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixSetColStartsOwner(hypre_ParCSRBooleanMatrix *matrix,
                                             HYPRE_Int owns_col_starts )
{
   hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = owns_col_starts;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixRead
 *--------------------------------------------------------------------------*/

hypre_ParCSRBooleanMatrix *
hypre_ParCSRBooleanMatrixRead( MPI_Comm comm, const char *file_name )
{
   hypre_ParCSRBooleanMatrix  *matrix;
   hypre_CSRBooleanMatrix  *diag;
   hypre_CSRBooleanMatrix  *offd;
   HYPRE_Int  my_id, i, num_procs;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   HYPRE_Int  global_num_rows, global_num_cols, num_cols_offd;
   HYPRE_Int  local_num_rows;
   HYPRE_Int  *row_starts;
   HYPRE_Int  *col_starts;
   HYPRE_Int  *col_map_offd;
   FILE *fp;
   HYPRE_Int equal = 1;

   hypre_MPI_Comm_rank(comm,&my_id);
   hypre_MPI_Comm_size(comm,&num_procs);
   row_starts = hypre_CTAlloc(HYPRE_Int, num_procs+1);
   col_starts = hypre_CTAlloc(HYPRE_Int, num_procs+1);
   hypre_sprintf(new_file_d,"%s.D.%d",file_name,my_id);
   hypre_sprintf(new_file_o,"%s.O.%d",file_name,my_id);
   hypre_sprintf(new_file_info,"%s.INFO.%d",file_name,my_id);
   fp = fopen(new_file_info, "r");
   hypre_fscanf(fp, "%d", &global_num_rows);
   hypre_fscanf(fp, "%d", &global_num_cols);
   hypre_fscanf(fp, "%d", &num_cols_offd);
   for (i=0; i < num_procs; i++)
           hypre_fscanf(fp, "%d %d", &row_starts[i], &col_starts[i]);
   row_starts[num_procs] = global_num_rows;
   col_starts[num_procs] = global_num_cols;
   col_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
        hypre_fscanf(fp, "%d", &col_map_offd[i]);
        
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
   
   diag = hypre_CSRBooleanMatrixRead(new_file_d);
   local_num_rows = hypre_CSRBooleanMatrix_Get_NRows(diag);

   if (num_cols_offd)
   {
        offd = hypre_CSRBooleanMatrixRead(new_file_o);
   }
   else
        offd = hypre_CSRBooleanMatrixCreate(local_num_rows,0,0);

        
   matrix = hypre_CTAlloc(hypre_ParCSRBooleanMatrix, 1);
   
   hypre_ParCSRBooleanMatrix_Get_Comm(matrix) = comm;
   hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix) = global_num_rows;
   hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix) = global_num_cols;
   hypre_ParCSRBooleanMatrix_Get_StartRow(matrix) = row_starts[my_id];
   hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix) = col_starts[my_id];
   hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix) = row_starts;
   hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix) = col_starts;
   hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix) = 1;
   hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = 1;
   hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
        hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 0;

   hypre_ParCSRBooleanMatrix_Get_Diag(matrix) = diag;
   hypre_ParCSRBooleanMatrix_Get_Offd(matrix) = offd;
   if (num_cols_offd)
        hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = col_map_offd;
   else
        hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = NULL;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixPrint( hypre_ParCSRBooleanMatrix *matrix, 
                                    const char                *file_name )
{
   MPI_Comm comm = hypre_ParCSRBooleanMatrix_Get_Comm(matrix);
   HYPRE_Int global_num_rows = hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix);
   HYPRE_Int global_num_cols = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix);
   HYPRE_Int *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);
   HYPRE_Int *row_starts = hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix);
   HYPRE_Int *col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix);
   HYPRE_Int  my_id, i, num_procs;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   HYPRE_Int  ierr = 0;
   FILE *fp;
   HYPRE_Int  num_cols_offd = 0;

   if (hypre_ParCSRBooleanMatrix_Get_Offd(matrix)) num_cols_offd = 
      hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Offd(matrix));

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);
   
   hypre_sprintf(new_file_d,"%s.D.%d",file_name,my_id);
   hypre_sprintf(new_file_o,"%s.O.%d",file_name,my_id);
   hypre_sprintf(new_file_info,"%s.INFO.%d",file_name,my_id);
   hypre_CSRBooleanMatrixPrint(hypre_ParCSRBooleanMatrix_Get_Diag(matrix),new_file_d);
   if (num_cols_offd != 0)
      hypre_CSRBooleanMatrixPrint(hypre_ParCSRBooleanMatrix_Get_Offd(matrix),
                                new_file_o);
  
   fp = fopen(new_file_info, "w");
   hypre_fprintf(fp, "%d\n", global_num_rows);
   hypre_fprintf(fp, "%d\n", global_num_cols);
   hypre_fprintf(fp, "%d\n", num_cols_offd);
   for (i=0; i < num_procs; i++)
      hypre_fprintf(fp, "%d %d\n", row_starts[i], col_starts[i]);
   for (i=0; i < num_cols_offd; i++)
      hypre_fprintf(fp, "%d\n", col_map_offd[i]);
   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixPrintIJ
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixPrintIJ( hypre_ParCSRBooleanMatrix *matrix, 
                                      const char                *filename )
{
   MPI_Comm comm = hypre_ParCSRBooleanMatrix_Get_Comm(matrix);
   HYPRE_Int      global_num_rows = hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix);
   HYPRE_Int      global_num_cols = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix);
   HYPRE_Int      first_row_index = hypre_ParCSRBooleanMatrix_Get_StartRow(matrix);
   HYPRE_Int      first_col_diag  = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix);
   HYPRE_Int     *col_map_offd    = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);
   HYPRE_Int      num_rows        = hypre_ParCSRBooleanMatrix_Get_NRows(matrix);
   HYPRE_Int     *diag_i;
   HYPRE_Int     *diag_j;
   HYPRE_Int     *offd_i;
   HYPRE_Int     *offd_j;
   HYPRE_Int      myid, i, j, I, J;
   HYPRE_Int      ierr = 0;
   char     new_filename[255];
   FILE    *file;
   hypre_CSRBooleanMatrix *diag = hypre_ParCSRBooleanMatrix_Get_Diag(matrix);
   hypre_CSRBooleanMatrix *offd = hypre_ParCSRBooleanMatrix_Get_Offd(matrix);
   HYPRE_Int  num_cols_offd = 0;

   if (offd) num_cols_offd = 
      hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Offd(matrix));

   hypre_MPI_Comm_rank(comm, &myid);
   
   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   hypre_fprintf(file, "%d, %d\n", global_num_rows, global_num_cols);
   hypre_fprintf(file, "%d\n", num_rows);

   diag_i    = hypre_CSRBooleanMatrix_Get_I(diag);
   diag_j    = hypre_CSRBooleanMatrix_Get_J(diag);
   if (num_cols_offd)
   {
      offd_i    = hypre_CSRBooleanMatrix_Get_I(offd);
      offd_j    = hypre_CSRBooleanMatrix_Get_J(offd);
   }
   for (i = 0; i < num_rows; i++)
   {
      I = first_row_index + i;

      /* print diag columns */
      for (j = diag_i[i]; j < diag_i[i+1]; j++)
      {
         J = first_col_diag + diag_j[j];
         hypre_fprintf(file, "%d, %d\n", I, J );
      }

      /* print offd columns */
      if (num_cols_offd)
      {
         for (j = offd_i[i]; j < offd_i[i+1]; j++)
         {
            J = col_map_offd[offd_j[j]];
            hypre_fprintf(file, "%d, %d \n", I, J);
         }
      }
   }

   fclose(file);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixGetLocalRange
 * returns the row numbers of the rows stored on this processor.
 * "End" is actually the row number of the last row on this processor.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixGetLocalRange(hypre_ParCSRBooleanMatrix *matrix,
                                         HYPRE_Int *row_start, HYPRE_Int *row_end,
                                         HYPRE_Int *col_start, HYPRE_Int *col_end )
{  
   HYPRE_Int ierr=0;
   HYPRE_Int my_id;

   hypre_MPI_Comm_rank( hypre_ParCSRBooleanMatrix_Get_Comm(matrix), &my_id );

   *row_start = hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix)[ my_id ];
   *row_end   = hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix)[ my_id + 1 ]-1;
   *col_start = hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix)[ my_id ];
   *col_end   = hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix)[ my_id + 1 ]-1;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixGetRow
 * Returns global column indices for a given row in the global matrix.
 * Global row number is used, but the row must be stored locally or
 * an error is returned. This implementation copies from the two matrices that
 * store the local data, storing them in the hypre_ParCSRBooleanMatrix structure.
 * Only a single row can be accessed via this function at any one time; the
 * corresponding RestoreRow function must be called, to avoid bleeding memory,
 * and to be able to look at another row.  All indices are returned in 0-based 
 * indexing, no matter what is used under the hood. 
 * EXCEPTION: currently this only works if the local CSR matrices
 * use 0-based indexing.
 * This code, semantics, implementation, etc., are all based on PETSc's hypre_MPI_AIJ
 * matrix code, adjusted for our data and software structures.
 * AJC 4/99.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixGetRow(hypre_ParCSRBooleanMatrix  *mat,
                                  HYPRE_Int row, HYPRE_Int *size, HYPRE_Int **col_ind)
{  
   HYPRE_Int    i, m, ierr=0, max=1, tmp, my_id, row_start, row_end;
   HYPRE_Int    *cworkA, *cworkB; 
   HYPRE_Int    cstart, nztot, nzA, nzB, lrow;
   HYPRE_Int    *cmap, *idx_p;
   hypre_CSRBooleanMatrix *Aa, *Ba;

   Aa = (hypre_CSRBooleanMatrix *) hypre_ParCSRBooleanMatrix_Get_Diag(mat);
   Ba = (hypre_CSRBooleanMatrix *) hypre_ParCSRBooleanMatrix_Get_Offd(mat);
   
   if (hypre_ParCSRBooleanMatrix_Get_Getrowactive(mat)) return(-1);

   hypre_MPI_Comm_rank( hypre_ParCSRBooleanMatrix_Get_Comm(mat), &my_id );

   hypre_ParCSRBooleanMatrix_Get_Getrowactive(mat) = 1;

   row_end   = hypre_ParCSRBooleanMatrix_Get_RowStarts(mat)[ my_id + 1 ];
   row_start = hypre_ParCSRBooleanMatrix_Get_RowStarts(mat)[ my_id ];
   lrow      = row - row_start;

   if (row < row_start || row >= row_end) return(-1);

   if ( col_ind ) 
   {
      m = row_end-row_start;
      for ( i=0; i<m; i++ ) 
      {
        tmp = hypre_CSRBooleanMatrix_Get_I(Aa)[i+1] - 
              hypre_CSRBooleanMatrix_Get_I(Aa)[i] + 
              hypre_CSRBooleanMatrix_Get_I(Ba)[i+1] - 
              hypre_CSRBooleanMatrix_Get_I(Ba)[i];
        if (max < tmp) { max = tmp; }
      }
      hypre_ParCSRBooleanMatrix_Get_Rowindices(mat) = (HYPRE_Int *) hypre_CTAlloc(HYPRE_Int,max); 
   }

   cstart = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(mat);

   nzA = hypre_CSRBooleanMatrix_Get_I(Aa)[lrow+1] -
         hypre_CSRBooleanMatrix_Get_I(Aa)[lrow];
   cworkA= &(hypre_CSRBooleanMatrix_Get_J(Aa)[hypre_CSRBooleanMatrix_Get_I(Aa)[lrow]]);

   nzB = hypre_CSRBooleanMatrix_Get_I(Ba)[lrow+1] -
         hypre_CSRBooleanMatrix_Get_I(Ba)[lrow];
   cworkB= &(hypre_CSRBooleanMatrix_Get_J(Ba)[hypre_CSRBooleanMatrix_Get_I(Ba)[lrow]]);

   nztot = nzA + nzB;

   cmap  = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(mat);

   if (col_ind) 
   {
      if (nztot) 
      {
         HYPRE_Int imark = -1;
         if (col_ind) 
         {
            *col_ind = idx_p = hypre_ParCSRBooleanMatrix_Get_Rowindices(mat);
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
 * hypre_ParCSRBooleanMatrixRestoreRow
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRBooleanMatrixRestoreRow( hypre_ParCSRBooleanMatrix *matrix,
                                       HYPRE_Int row, HYPRE_Int *size, HYPRE_Int **col_ind)
{  

   if (!hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix)) return( -1 );

   hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix) = 0;

   return( 0 );
}


/*--------------------------------------------------------------------------
 * hypre_BuildCSRBooleanMatrixMPIDataType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BuildCSRBooleanMatrixMPIDataType(
   HYPRE_Int num_nonzeros, HYPRE_Int num_rows, HYPRE_Int *a_i, HYPRE_Int *a_j, 
   hypre_MPI_Datatype *csr_matrix_datatype )
{
   HYPRE_Int		block_lens[2];
   hypre_MPI_Aint	displ[2];
   hypre_MPI_Datatype	types[2];
   HYPRE_Int		ierr = 0;

   block_lens[0] = num_rows+1;
   block_lens[1] = num_nonzeros;

   types[0] = HYPRE_MPI_INT;
   types[1] = HYPRE_MPI_INT;

   hypre_MPI_Address(a_i, &displ[0]);
   hypre_MPI_Address(a_j, &displ[1]);
   hypre_MPI_Type_struct(2,block_lens,displ,types,csr_matrix_datatype);
   hypre_MPI_Type_commit(csr_matrix_datatype);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBooleanMatrixToParCSRBooleanMatrix:
 * generates a ParCSRBooleanMatrix distributed across the processors in comm
 * from a CSRBooleanMatrix on proc 0 .
 *--------------------------------------------------------------------------*/

hypre_ParCSRBooleanMatrix *
hypre_CSRBooleanMatrixToParCSRBooleanMatrix
( MPI_Comm comm, hypre_CSRBooleanMatrix *A,
  HYPRE_Int *row_starts, HYPRE_Int *col_starts )
{
   HYPRE_Int          global_data[2];
   HYPRE_Int          global_num_rows;
   HYPRE_Int          global_num_cols;
   HYPRE_Int          *local_num_rows;

   HYPRE_Int          num_procs, my_id;
   HYPRE_Int          *local_num_nonzeros;
   HYPRE_Int          num_nonzeros;
  
   HYPRE_Int          *a_i;
   HYPRE_Int          *a_j;
  
   hypre_CSRBooleanMatrix *local_A;

   hypre_MPI_Request  *requests;
   hypre_MPI_Status   *status, status0;
   hypre_MPI_Datatype *csr_matrix_datatypes;

   hypre_ParCSRBooleanMatrix *par_matrix;

   HYPRE_Int          first_col_diag;
   HYPRE_Int          last_col_diag;
 
   HYPRE_Int i, j, ind;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   if (my_id == 0) 
   {
        global_data[0] = hypre_CSRBooleanMatrix_Get_NRows(A);
        global_data[1] = hypre_CSRBooleanMatrix_Get_NCols(A);
        a_i = hypre_CSRBooleanMatrix_Get_I(A);
        a_j = hypre_CSRBooleanMatrix_Get_J(A);
   }
   hypre_MPI_Bcast(global_data,2,HYPRE_MPI_INT,0,comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];

   local_num_rows = hypre_CTAlloc(HYPRE_Int, num_procs);
   csr_matrix_datatypes = hypre_CTAlloc(hypre_MPI_Datatype, num_procs);

   par_matrix = hypre_ParCSRBooleanMatrixCreate (comm, global_num_rows,
        global_num_cols,row_starts,col_starts,0,0,0);

   row_starts = hypre_ParCSRBooleanMatrix_Get_RowStarts(par_matrix);
   col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(par_matrix);

   for (i=0; i < num_procs; i++)
         local_num_rows[i] = row_starts[i+1] - row_starts[i];

   if (my_id == 0)
   {
        local_num_nonzeros = hypre_CTAlloc(HYPRE_Int, num_procs);
        for (i=0; i < num_procs-1; i++)
                local_num_nonzeros[i] = a_i[row_starts[i+1]] 
                                - a_i[row_starts[i]];
        local_num_nonzeros[num_procs-1] = a_i[global_num_rows] 
                                - a_i[row_starts[num_procs-1]];
   }
   hypre_MPI_Scatter(local_num_nonzeros,1,HYPRE_MPI_INT,&num_nonzeros,1,HYPRE_MPI_INT,0,comm);

   if (my_id == 0) num_nonzeros = local_num_nonzeros[0];

   local_A = hypre_CSRBooleanMatrixCreate(local_num_rows[my_id], global_num_cols,
                num_nonzeros);
   if (my_id == 0)
   {
        requests = hypre_CTAlloc (hypre_MPI_Request, num_procs-1);
        status = hypre_CTAlloc(hypre_MPI_Status, num_procs-1);
        j=0;
        for (i=1; i < num_procs; i++)
        {
                ind = a_i[row_starts[i]];
                hypre_BuildCSRBooleanMatrixMPIDataType(local_num_nonzeros[i], 
                        local_num_rows[i],
                        &a_i[row_starts[i]],
                        &a_j[ind],
                        &csr_matrix_datatypes[i]);
                hypre_MPI_Isend(hypre_MPI_BOTTOM, 1, csr_matrix_datatypes[i], i, 0, comm,
                        &requests[j++]);
                hypre_MPI_Type_free(&csr_matrix_datatypes[i]);
        }
        hypre_CSRBooleanMatrix_Get_I(local_A) = a_i;
        hypre_CSRBooleanMatrix_Get_J(local_A) = a_j;
        hypre_MPI_Waitall(num_procs-1,requests,status);
        hypre_TFree(requests);
        hypre_TFree(status);
        hypre_TFree(local_num_nonzeros);
    }
   else
   {
        hypre_CSRBooleanMatrixInitialize(local_A);
        hypre_BuildCSRBooleanMatrixMPIDataType(num_nonzeros, 
                        local_num_rows[my_id],
                        hypre_CSRBooleanMatrix_Get_I(local_A),
                        hypre_CSRBooleanMatrix_Get_J(local_A),
                        csr_matrix_datatypes);
        hypre_MPI_Recv(hypre_MPI_BOTTOM,1,csr_matrix_datatypes[0],0,0,comm,&status0);
        hypre_MPI_Type_free(csr_matrix_datatypes);
   }

   first_col_diag = col_starts[my_id];
   last_col_diag = col_starts[my_id+1]-1;

   BooleanGenerateDiagAndOffd(local_A, par_matrix, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {      
      hypre_CSRBooleanMatrix_Get_I(local_A) = NULL;
      hypre_CSRBooleanMatrix_Get_J(local_A) = NULL; 
   }      
   hypre_CSRBooleanMatrixDestroy(local_A);
   hypre_TFree(local_num_rows);
   hypre_TFree(csr_matrix_datatypes);

   return par_matrix;
}

HYPRE_Int
BooleanGenerateDiagAndOffd(hypre_CSRBooleanMatrix *A,
                    hypre_ParCSRBooleanMatrix *matrix,
                    HYPRE_Int first_col_diag,
                    HYPRE_Int last_col_diag)
{
   HYPRE_Int  i, j;
   HYPRE_Int  jo, jd;
   HYPRE_Int  ierr = 0;
   HYPRE_Int  num_rows = hypre_CSRBooleanMatrix_Get_NRows(A);
   HYPRE_Int  num_cols = hypre_CSRBooleanMatrix_Get_NCols(A);
   HYPRE_Int *a_i = hypre_CSRBooleanMatrix_Get_I(A);
   HYPRE_Int *a_j = hypre_CSRBooleanMatrix_Get_J(A);

   hypre_CSRBooleanMatrix *diag = hypre_ParCSRBooleanMatrix_Get_Diag(matrix);
   hypre_CSRBooleanMatrix *offd = hypre_ParCSRBooleanMatrix_Get_Offd(matrix);

   HYPRE_Int  *col_map_offd;

   HYPRE_Int  *diag_i, *offd_i;
   HYPRE_Int  *diag_j, *offd_j;
   HYPRE_Int  *marker;
   HYPRE_Int num_cols_diag, num_cols_offd;
   HYPRE_Int first_elmt = a_i[0];
   HYPRE_Int num_nonzeros = a_i[num_rows]-first_elmt;
   HYPRE_Int counter;

   num_cols_diag = last_col_diag - first_col_diag +1;
   num_cols_offd = 0;

   if (num_cols - num_cols_diag)
   {
        hypre_CSRBooleanMatrixInitialize(diag);
        diag_i = hypre_CSRBooleanMatrix_Get_I(diag);

        hypre_CSRBooleanMatrixInitialize(offd);
        offd_i = hypre_CSRBooleanMatrix_Get_I(offd);
        marker = hypre_CTAlloc(HYPRE_Int,num_cols);

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

        hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) =
           hypre_CTAlloc(HYPRE_Int,num_cols_offd);
        col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);

        counter = 0;
        for (i=0; i < num_cols; i++)
                if (marker[i])
                {
                        col_map_offd[counter] = i;
                        marker[i] = counter;
                        counter++;
                }

        hypre_CSRBooleanMatrix_Get_NNZ(diag) = jd;
        hypre_CSRBooleanMatrixInitialize(diag);
        diag_j = hypre_CSRBooleanMatrix_Get_J(diag);

        hypre_CSRBooleanMatrix_Get_NNZ(offd) = jo;
        hypre_CSRBooleanMatrix_Get_NCols(offd) = num_cols_offd;
        hypre_CSRBooleanMatrixInitialize(offd);
        offd_j = hypre_CSRBooleanMatrix_Get_J(offd);

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
        hypre_CSRBooleanMatrix_Get_NNZ(diag) = num_nonzeros;
        hypre_CSRBooleanMatrixInitialize(diag);
        diag_i = hypre_CSRBooleanMatrix_Get_I(diag);
        diag_j = hypre_CSRBooleanMatrix_Get_J(diag);

        for (i=0; i < num_nonzeros; i++)
        {
                diag_j[i] = a_j[i];
        }
        offd_i = hypre_CTAlloc(HYPRE_Int, num_rows+1);

        for (i=0; i < num_rows+1; i++)
        {
                diag_i[i] = a_i[i];
                offd_i[i] = 0;
        }

        hypre_CSRBooleanMatrix_Get_NCols(offd) = 0;
        hypre_CSRBooleanMatrix_Get_I(offd) = offd_i;
   }
   
   return ierr;
}


