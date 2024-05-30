/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_DenseBlockMatrix class.
 *
 *****************************************************************************/

#include "_hypre_seq_block_mv.h"

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_DenseBlockMatrix *
hypre_DenseBlockMatrixCreate( HYPRE_Int  row_major,
                              HYPRE_Int  num_rows,
                              HYPRE_Int  num_cols,
                              HYPRE_Int  num_rows_block,
                              HYPRE_Int  num_cols_block )
{
   hypre_DenseBlockMatrix  *A;
   HYPRE_Int                num_blocks[2];

   /* Compute number of blocks */
   num_blocks[0] = hypre_ceildiv(num_rows, num_rows_block);
   num_blocks[1] = hypre_ceildiv(num_cols, num_cols_block);
   if (num_blocks[0] != num_blocks[1])
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid number of blocks!");
      return NULL;
   }

   /* Allocate memory */
   A = hypre_TAlloc(hypre_DenseBlockMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_DenseBlockMatrixRowMajor(A)         = row_major;
   hypre_DenseBlockMatrixNumRowsBlock(A)     = num_rows_block;
   hypre_DenseBlockMatrixNumColsBlock(A)     = num_cols_block;
   hypre_DenseBlockMatrixNumBlocks(A)        = num_blocks[0];
   hypre_DenseBlockMatrixNumRows(A)          = num_blocks[0] * hypre_DenseBlockMatrixNumRowsBlock(A);
   hypre_DenseBlockMatrixNumCols(A)          = num_blocks[0] * hypre_DenseBlockMatrixNumColsBlock(A);
   hypre_DenseBlockMatrixNumNonzerosBlock(A) = hypre_DenseBlockMatrixNumRowsBlock(A) *
                                               hypre_DenseBlockMatrixNumColsBlock(A);
   hypre_DenseBlockMatrixNumNonzeros(A)      = num_blocks[0] *
                                               hypre_DenseBlockMatrixNumNonzerosBlock(A);
   hypre_DenseBlockMatrixOwnsData(A)         = 0;
   hypre_DenseBlockMatrixData(A)             = NULL;
   hypre_DenseBlockMatrixDataAOP(A)          = NULL;
   hypre_DenseBlockMatrixMemoryLocation(A)   = hypre_HandleMemoryLocation(hypre_handle());

   if (row_major)
   {
      hypre_DenseBlockMatrixRowStride(A)     = 1;
      hypre_DenseBlockMatrixColStride(A)     = hypre_DenseBlockMatrixNumColsBlock(A);
   }
   else
   {
      hypre_DenseBlockMatrixRowStride(A)     = hypre_DenseBlockMatrixNumRowsBlock(A);
      hypre_DenseBlockMatrixColStride(A)     = 1;
   }

   return A;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixCreateByBlock
 *--------------------------------------------------------------------------*/

hypre_DenseBlockMatrix *
hypre_DenseBlockMatrixCreateByBlock( HYPRE_Int  row_major,
                                     HYPRE_Int  num_blocks,
                                     HYPRE_Int  num_rows_block,
                                     HYPRE_Int  num_cols_block )
{
   return hypre_DenseBlockMatrixCreate(row_major,
                                       num_blocks * num_rows_block,
                                       num_blocks * num_cols_block,
                                       num_rows_block,
                                       num_cols_block);
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixClone
 *--------------------------------------------------------------------------*/

hypre_DenseBlockMatrix*
hypre_DenseBlockMatrixClone( hypre_DenseBlockMatrix *A,
                             HYPRE_Int               copy_data )
{
   HYPRE_Int row_major      = hypre_DenseBlockMatrixRowMajor(A);
   HYPRE_Int num_rows       = hypre_DenseBlockMatrixNumRows(A);
   HYPRE_Int num_cols       = hypre_DenseBlockMatrixNumCols(A);
   HYPRE_Int num_rows_block = hypre_DenseBlockMatrixNumRowsBlock(A);
   HYPRE_Int num_cols_block = hypre_DenseBlockMatrixNumColsBlock(A);

   hypre_DenseBlockMatrix  *B;

   /* Create new matrix */
   B = hypre_DenseBlockMatrixCreate(row_major,
                                    num_rows, num_cols,
                                    num_rows_block, num_cols_block);

   /* Initialize matrix */
   hypre_DenseBlockMatrixInitializeOn(B, hypre_DenseBlockMatrixMemoryLocation(A));

   /* Copy data array */
   if (copy_data)
   {
      hypre_DenseBlockMatrixCopy(A, B);
   }

   return B;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixDestroy( hypre_DenseBlockMatrix *A )
{
   if (A)
   {
      HYPRE_MemoryLocation memory_location = hypre_DenseBlockMatrixMemoryLocation(A);

      if (hypre_DenseBlockMatrixOwnsData(A))
      {
         hypre_TFree(hypre_DenseBlockMatrixData(A), memory_location);
      }

      /* data_aop is always owned by a hypre_DenseBlockMatrix */
      hypre_TFree(hypre_DenseBlockMatrixDataAOP(A), memory_location);

      /* Free matrix pointer */
      hypre_TFree(A, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixInitializeOn
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixInitializeOn( hypre_DenseBlockMatrix  *A,
                                    HYPRE_MemoryLocation     memory_location )
{
   hypre_DenseBlockMatrixMemoryLocation(A) = memory_location;

   /* Allocate memory for data */
   if (!hypre_DenseBlockMatrixData(A) && hypre_DenseBlockMatrixNumNonzeros(A))
   {
      hypre_DenseBlockMatrixData(A) = hypre_CTAlloc(HYPRE_Complex,
                                                    hypre_DenseBlockMatrixNumNonzeros(A),
                                                    memory_location);
      hypre_DenseBlockMatrixOwnsData(A) = 1;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixInitialize( hypre_DenseBlockMatrix *A )
{
   return hypre_DenseBlockMatrixInitializeOn(A, hypre_DenseBlockMatrixMemoryLocation(A));
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixBuildAOP
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixBuildAOP( hypre_DenseBlockMatrix *A )
{
   HYPRE_MemoryLocation memory_location = hypre_DenseBlockMatrixMemoryLocation(A);

   /* Allocate memory if we need */
   if (!hypre_DenseBlockMatrixDataAOP(A))
   {
      hypre_DenseBlockMatrixDataAOP(A) = hypre_TAlloc(HYPRE_Complex *,
                                                      hypre_DenseBlockMatrixNumBlocks(A),
                                                      memory_location);
   }

   /* Build array of pointers to the matrix data */
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypreDevice_ComplexArrayToArrayOfPtrs(hypre_DenseBlockMatrixNumBlocks(A),
                                            hypre_DenseBlockMatrixNumNonzerosBlock(A),
                                            hypre_DenseBlockMatrixData(A),
                                            hypre_DenseBlockMatrixDataAOP(A));
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixCopy( hypre_DenseBlockMatrix *A,
                            hypre_DenseBlockMatrix *B )
{
   /* Copy coeficients from matrix A to B */
   hypre_TMemcpy(hypre_DenseBlockMatrixData(B),
                 hypre_DenseBlockMatrixData(A),
                 HYPRE_Complex,
                 hypre_DenseBlockMatrixNumNonzeros(A),
                 hypre_DenseBlockMatrixMemoryLocation(B),
                 hypre_DenseBlockMatrixMemoryLocation(A));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixMigrate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixMigrate( hypre_DenseBlockMatrix *A,
                               HYPRE_MemoryLocation    memory_location )
{
   /* Input matrix info */
   HYPRE_MemoryLocation   old_memory_location = hypre_DenseBlockMatrixMemoryLocation(A);
   HYPRE_Int              num_nonzeros        = hypre_DenseBlockMatrixNumNonzeros(A);
   HYPRE_Complex         *A_data              = hypre_DenseBlockMatrixData(A);

   /* Output matrix info */
   HYPRE_Complex         *B_data;

   /* Update A's memory location */
   hypre_DenseBlockMatrixMemoryLocation(A) = memory_location;

   if ( hypre_GetActualMemLocation(memory_location) !=
        hypre_GetActualMemLocation(old_memory_location) )
   {
      if (A_data)
      {
         B_data = hypre_TAlloc(HYPRE_Complex, num_nonzeros, memory_location);
         hypre_TMemcpy(B_data, A_data, HYPRE_Complex, num_nonzeros,
                       memory_location, old_memory_location);
         hypre_TFree(A_data, old_memory_location);
         hypre_DenseBlockMatrixData(A) = B_data;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixPrint( MPI_Comm                comm,
                             hypre_DenseBlockMatrix *A,
                             const char*             filename )
{
   /* Input matrix info */
   HYPRE_MemoryLocation   memory_location = hypre_DenseBlockMatrixMemoryLocation(A);

   /* Local variables */
   char                   new_filename[HYPRE_MAX_FILE_NAME_LEN];
   HYPRE_Int              myid, ib, i, j;
   FILE                  *file;

   /* Move matrix to host */
   hypre_DenseBlockMatrixMigrate(A, HYPRE_MEMORY_HOST);

   /* Open file */
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Cannot open output file!");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Write the header
    *---------------------------------------------*/

   /* 1st header line: matrix info */
   hypre_fprintf(file, "%d %d\n",
                 hypre_DenseBlockMatrixNumRows(A),
                 hypre_DenseBlockMatrixNumCols(A));

   /* 2nd header line: local block info */
   hypre_fprintf(file, "%d %d %d %d\n",
                 hypre_DenseBlockMatrixRowMajor(A),
                 hypre_DenseBlockMatrixNumBlocks(A),
                 hypre_DenseBlockMatrixNumRowsBlock(A),
                 hypre_DenseBlockMatrixNumColsBlock(A));

   /*---------------------------------------------
    * Write coefficients
    *---------------------------------------------*/

   for (ib = 0; ib < hypre_DenseBlockMatrixNumBlocks(A); ib++)
   {
      for (i = 0; i < hypre_DenseBlockMatrixNumRowsBlock(A); i++)
      {
         hypre_fprintf(file, "%d", ib);

         for (j = 0; j < hypre_DenseBlockMatrixNumColsBlock(A); j++)
         {
            hypre_fprintf(file, " %.15e", hypre_DenseBlockMatrixDataBIJ(A, ib, i, j));
         }
         hypre_fprintf(file, "\n");
      }
   }

   fclose(file);

   /* Move matrix back to original lcoation */
   hypre_DenseBlockMatrixMigrate(A, memory_location);

   return hypre_error_flag;
}
