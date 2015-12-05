/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface: matvec with multivectors
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   hypre_CSRMatrix     *matrix;
   hypre_CSRMatrix     *matrix1;
   hypre_ParCSRMatrix  *par_matrix;
   hypre_Vector        *x_local;
   hypre_Vector        *y_local;
   hypre_Vector        *y2_local;
   hypre_ParVector     *x;
   hypre_ParVector     *x2;
   hypre_ParVector     *y;
   hypre_ParVector     *y2;

   HYPRE_Int          vecstride_x, idxstride_x, vecstride_y, idxstride_y;
   HYPRE_Int          num_procs, my_id;
   HYPRE_Int		local_size;
   HYPRE_Int          num_vectors;
   HYPRE_Int		global_num_rows, global_num_cols;
   HYPRE_Int		first_index;
   HYPRE_Int 		i, j, ierr=0;
   double 	*data, *data2;
   HYPRE_Int 		*row_starts, *col_starts;
   char		file_name[80];
   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id);

   hypre_printf(" my_id: %d num_procs: %d\n", my_id, num_procs);
 
   if (my_id == 0) 
   {
	matrix = hypre_CSRMatrixRead("input");
   	hypre_printf(" read input\n");
   }
   row_starts = NULL;
   col_starts = NULL; 
   par_matrix = hypre_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, matrix, 
		row_starts, col_starts);
   hypre_printf(" converted\n");

   matrix1 = hypre_ParCSRMatrixToCSRMatrixAll(par_matrix);

   hypre_sprintf(file_name,"matrix1.%d",my_id);

   if (matrix1) hypre_CSRMatrixPrint(matrix1, file_name);

   hypre_ParCSRMatrixPrint(par_matrix,"matrix");
   hypre_ParCSRMatrixPrintIJ(par_matrix,0,0,"matrixIJ");

   par_matrix = hypre_ParCSRMatrixRead(hypre_MPI_COMM_WORLD,"matrix");

   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   hypre_printf(" global_num_cols %d\n", global_num_cols);
   global_num_rows = hypre_ParCSRMatrixGlobalNumRows(par_matrix);
 
   col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
   first_index = col_starts[my_id];
   local_size = col_starts[my_id+1] - first_index;

   num_vectors = 3;

   x = hypre_ParMultiVectorCreate( hypre_MPI_COMM_WORLD, global_num_cols,
                                         col_starts, num_vectors );
   hypre_ParVectorSetPartitioningOwner(x,0);
   hypre_ParVectorInitialize(x);
   x_local = hypre_ParVectorLocalVector(x);
   data = hypre_VectorData(x_local);
   vecstride_x = hypre_VectorVectorStride(x_local);
   idxstride_x = hypre_VectorIndexStride(x_local);
   for ( j=0; j<num_vectors; ++j )
      for (i=0; i < local_size; i++)
         data[i*idxstride_x + j*vecstride_x] = first_index+i+1 + 100*j;

   x2 = hypre_ParMultiVectorCreate( hypre_MPI_COMM_WORLD, global_num_cols,
                                    col_starts, num_vectors );
   hypre_ParVectorSetPartitioningOwner(x2,0);
   hypre_ParVectorInitialize(x2);
   hypre_ParVectorSetConstantValues(x2,2.0);

   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   first_index = row_starts[my_id];
   local_size = row_starts[my_id+1] - first_index;
   y = hypre_ParMultiVectorCreate( hypre_MPI_COMM_WORLD, global_num_rows,
                                   row_starts, num_vectors );
   hypre_ParVectorSetPartitioningOwner(y,0);
   hypre_ParVectorInitialize(y);
   y_local = hypre_ParVectorLocalVector(y);

   y2 = hypre_ParMultiVectorCreate( hypre_MPI_COMM_WORLD, global_num_rows,
                                    row_starts, num_vectors );
   hypre_ParVectorSetPartitioningOwner(y2,0);
   hypre_ParVectorInitialize(y2);
   y2_local = hypre_ParVectorLocalVector(y2);
   data2 = hypre_VectorData(y2_local);
   vecstride_y = hypre_VectorVectorStride(y2_local);
   idxstride_y = hypre_VectorIndexStride(y2_local);
 
   for ( j=0; j<num_vectors; ++j )
      for (i=0; i < local_size; i++)
         data2[i*idxstride_y+j*vecstride_y] = first_index+i+1 + 100*j;

   hypre_ParVectorSetConstantValues(y,1.0);
   hypre_printf(" initialized vectors, first_index=%i\n", first_index);

   hypre_ParVectorPrint(x, "vectorx");
   hypre_ParVectorPrint(y, "vectory");

   hypre_MatvecCommPkgCreate(par_matrix);

   hypre_ParCSRMatrixMatvec ( 1.0, par_matrix, x, 1.0, y);
   hypre_printf(" did matvec\n");

   hypre_ParVectorPrint(y, "result");

   ierr = hypre_ParCSRMatrixMatvecT ( 1.0, par_matrix, y2, 1.0, x2);
   hypre_printf(" did matvecT %d\n", ierr);

   hypre_ParVectorPrint(x2, "transp"); 

   hypre_ParCSRMatrixDestroy(par_matrix);
   hypre_ParVectorDestroy(x);
   hypre_ParVectorDestroy(x2);
   hypre_ParVectorDestroy(y);
   hypre_ParVectorDestroy(y2);
   if (my_id == 0) hypre_CSRMatrixDestroy(matrix);
   if (matrix1) hypre_CSRMatrixDestroy(matrix1);

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return 0;
}

