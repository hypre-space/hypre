/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

#ifdef HYPRE_PROFILE
HYPRE_Real hypre_profile_times[HYPRE_TIMER_ID_COUNT] = { 0 };
#endif

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixCreate( HYPRE_Int num_rows,
                       HYPRE_Int num_cols,
                       HYPRE_Int num_nonzeros )
{
   hypre_CSRMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_CSRMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_CSRMatrixData(matrix)           = NULL;
   hypre_CSRMatrixI(matrix)              = NULL;
   hypre_CSRMatrixJ(matrix)              = NULL;
   hypre_CSRMatrixBigJ(matrix)           = NULL;
   hypre_CSRMatrixRownnz(matrix)         = NULL;
   hypre_CSRMatrixNumRows(matrix)        = num_rows;
   hypre_CSRMatrixNumRownnz(matrix)      = num_rows;
   hypre_CSRMatrixNumCols(matrix)        = num_cols;
   hypre_CSRMatrixNumNonzeros(matrix)    = num_nonzeros;
   hypre_CSRMatrixMemoryLocation(matrix) = hypre_HandleMemoryLocation(hypre_handle());

   /* set defaults */
   hypre_CSRMatrixOwnsData(matrix)       = 1;

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE) || defined(HYPRE_USING_ONEMKLSPARSE)
   hypre_CSRMatrixSortedJ(matrix)        = NULL;
   hypre_CSRMatrixSortedData(matrix)     = NULL;
   hypre_CSRMatrixCsrsvData(matrix)      = NULL;
#endif

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixDestroy( hypre_CSRMatrix *matrix )
{
   if (matrix)
   {
      HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);

      hypre_TFree(hypre_CSRMatrixI(matrix),      memory_location);
      hypre_TFree(hypre_CSRMatrixRownnz(matrix), memory_location);

      if ( hypre_CSRMatrixOwnsData(matrix) )
      {
         hypre_TFree(hypre_CSRMatrixData(matrix), memory_location);
         hypre_TFree(hypre_CSRMatrixJ(matrix),    memory_location);
         /* RL: TODO There might be cases BigJ cannot be freed FIXME
          * Not so clear how to do it */
         hypre_TFree(hypre_CSRMatrixBigJ(matrix), memory_location);
      }

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE) || defined(HYPRE_USING_ONEMKLSPARSE)
      hypre_TFree(hypre_CSRMatrixSortedData(matrix), memory_location);
      hypre_TFree(hypre_CSRMatrixSortedJ(matrix), memory_location);
      hypre_CsrsvDataDestroy(hypre_CSRMatrixCsrsvData(matrix));
      hypre_GpuMatDataDestroy(hypre_CSRMatrixGPUMatData(matrix));
#endif

      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixInitialize_v2( hypre_CSRMatrix      *matrix,
                              HYPRE_Int             bigInit,
                              HYPRE_MemoryLocation  memory_location )
{
   HYPRE_Int  num_rows     = hypre_CSRMatrixNumRows(matrix);
   HYPRE_Int  num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   /* HYPRE_Int  num_rownnz = hypre_CSRMatrixNumRownnz(matrix); */

   hypre_CSRMatrixMemoryLocation(matrix) = memory_location;

   /* Caveat: for pre-existing i, j, data, their memory location must be guaranteed to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered when being used, and freed */

   if ( !hypre_CSRMatrixData(matrix) && num_nonzeros )
   {
      hypre_CSRMatrixData(matrix) = hypre_CTAlloc(HYPRE_Complex, num_nonzeros, memory_location);
   }
   /*
   else
   {
     //if (PointerAttributes(hypre_CSRMatrixData(matrix))==HYPRE_HOST_POINTER) printf("MATREIX INITIAL WITH JHOST DATA\n");
   }
   */

   if ( !hypre_CSRMatrixI(matrix) )
   {
      hypre_CSRMatrixI(matrix) = hypre_CTAlloc(HYPRE_Int, num_rows + 1, memory_location);
   }

   /*
   if (!hypre_CSRMatrixRownnz(matrix))
   {
      hypre_CSRMatrixRownnz(matrix) = hypre_CTAlloc(HYPRE_Int,  num_rownnz, memory_location);
   }
   */

   if (bigInit)
   {
      if ( !hypre_CSRMatrixBigJ(matrix) && num_nonzeros )
      {
         hypre_CSRMatrixBigJ(matrix) = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros, memory_location);
      }
   }
   else
   {
      if ( !hypre_CSRMatrixJ(matrix) && num_nonzeros )
      {
         hypre_CSRMatrixJ(matrix) = hypre_CTAlloc(HYPRE_Int, num_nonzeros, memory_location);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixInitialize( hypre_CSRMatrix *matrix )
{
   return hypre_CSRMatrixInitialize_v2( matrix, 0, hypre_CSRMatrixMemoryLocation(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixResize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixResize( hypre_CSRMatrix *matrix,
                       HYPRE_Int        new_num_rows,
                       HYPRE_Int        new_num_cols,
                       HYPRE_Int        new_num_nonzeros )
{
   HYPRE_MemoryLocation memory_location  = hypre_CSRMatrixMemoryLocation(matrix);
   HYPRE_Int            old_num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   HYPRE_Int            old_num_rows     = hypre_CSRMatrixNumRows(matrix);

   if (!hypre_CSRMatrixOwnsData(matrix))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Error: called hypre_CSRMatrixResize on a matrix that doesn't own the data\n");
      return hypre_error_flag;
   }

   hypre_CSRMatrixNumCols(matrix) = new_num_cols;

   if (new_num_nonzeros != hypre_CSRMatrixNumNonzeros(matrix))
   {
      hypre_CSRMatrixNumNonzeros(matrix) = new_num_nonzeros;

      if (!hypre_CSRMatrixData(matrix))
      {
         hypre_CSRMatrixData(matrix) = hypre_CTAlloc(HYPRE_Complex, new_num_nonzeros, memory_location);
      }
      else
      {
         hypre_CSRMatrixData(matrix) = hypre_TReAlloc_v2(hypre_CSRMatrixData(matrix),
                                                         HYPRE_Complex, old_num_nonzeros,
                                                         HYPRE_Complex, new_num_nonzeros,
                                                         memory_location);
      }

      if (hypre_CSRMatrixBigJ(matrix))
      {
         hypre_CSRMatrixBigJ(matrix) = hypre_TReAlloc_v2(hypre_CSRMatrixBigJ(matrix),
                                                         HYPRE_BigInt, old_num_nonzeros,
                                                         HYPRE_BigInt, new_num_nonzeros,
                                                         memory_location);
      }
      else
      {
         if (!hypre_CSRMatrixJ(matrix))
         {
            hypre_CSRMatrixJ(matrix) = hypre_CTAlloc(HYPRE_Int, new_num_nonzeros, memory_location);
         }
         else
         {
            hypre_CSRMatrixJ(matrix) = hypre_TReAlloc_v2(hypre_CSRMatrixJ(matrix),
                                                         HYPRE_Int, old_num_nonzeros,
                                                         HYPRE_Int, new_num_nonzeros,
                                                         memory_location);
         }
      }
   }

   if (new_num_rows != hypre_CSRMatrixNumRows(matrix))
   {
      hypre_CSRMatrixNumRows(matrix) = new_num_rows;

      if (!hypre_CSRMatrixI(matrix))
      {
         hypre_CSRMatrixI(matrix) = hypre_CTAlloc(HYPRE_Int, new_num_rows + 1, memory_location);
      }
      else
      {
         hypre_CSRMatrixI(matrix) = hypre_TReAlloc_v2(hypre_CSRMatrixI(matrix),
                                                      HYPRE_Int, old_num_rows + 1,
                                                      HYPRE_Int, new_num_rows + 1,
                                                      memory_location);
      }
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixBigInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixBigInitialize( hypre_CSRMatrix *matrix )
{
   return hypre_CSRMatrixInitialize_v2( matrix, 1, hypre_CSRMatrixMemoryLocation(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixBigJtoJ
 * RL: TODO GPU impl.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixBigJtoJ( hypre_CSRMatrix *matrix )
{
   HYPRE_Int     num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   HYPRE_BigInt *matrix_big_j = hypre_CSRMatrixBigJ(matrix);
   HYPRE_Int    *matrix_j = NULL;

   if (num_nonzeros && matrix_big_j)
   {
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
      HYPRE_Int i;
      matrix_j = hypre_TAlloc(HYPRE_Int, num_nonzeros, hypre_CSRMatrixMemoryLocation(matrix));
      for (i = 0; i < num_nonzeros; i++)
      {
         matrix_j[i] = (HYPRE_Int) matrix_big_j[i];
      }
      hypre_TFree(matrix_big_j, hypre_CSRMatrixMemoryLocation(matrix));
#else
      hypre_assert(sizeof(HYPRE_Int) == sizeof(HYPRE_BigInt));
      matrix_j = (HYPRE_Int *) matrix_big_j;
#endif
      hypre_CSRMatrixJ(matrix) = matrix_j;
      hypre_CSRMatrixBigJ(matrix) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixJtoBigJ
 * RL: TODO GPU impl.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixJtoBigJ( hypre_CSRMatrix *matrix )
{
   HYPRE_Int     num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   HYPRE_Int    *matrix_j = hypre_CSRMatrixJ(matrix);
   HYPRE_BigInt *matrix_big_j = NULL;

   if (num_nonzeros && matrix_j)
   {
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
      HYPRE_Int i;
      matrix_big_j = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, hypre_CSRMatrixMemoryLocation(matrix));
      for (i = 0; i < num_nonzeros; i++)
      {
         matrix_big_j[i] = (HYPRE_BigInt) matrix_j[i];
      }
      hypre_TFree(matrix_j, hypre_CSRMatrixMemoryLocation(matrix));
#else
      hypre_assert(sizeof(HYPRE_Int) == sizeof(HYPRE_BigInt));
      matrix_big_j = (HYPRE_BigInt *) matrix_j;
#endif
      hypre_CSRMatrixBigJ(matrix) = matrix_big_j;
      hypre_CSRMatrixJ(matrix) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSetDataOwner( hypre_CSRMatrix *matrix,
                             HYPRE_Int        owns_data )
{
   hypre_CSRMatrixOwnsData(matrix) = owns_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetPatternOnly
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSetPatternOnly( hypre_CSRMatrix *matrix,
                               HYPRE_Int        pattern_only )
{
   hypre_CSRMatrixPatternOnly(matrix) = pattern_only;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetRownnzHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSetRownnzHost( hypre_CSRMatrix *matrix )
{
   HYPRE_MemoryLocation  memory_location = hypre_CSRMatrixMemoryLocation(matrix);
   HYPRE_Int             num_rows = hypre_CSRMatrixNumRows(matrix);
   HYPRE_Int            *A_i = hypre_CSRMatrixI(matrix);
   HYPRE_Int            *Arownnz = hypre_CSRMatrixRownnz(matrix);

   HYPRE_Int             i, irownnz = 0;

   for (i = 0; i < num_rows; i++)
   {
      if ((A_i[i + 1] - A_i[i]) > 0)
      {
         irownnz++;
      }
   }

   hypre_CSRMatrixNumRownnz(matrix) = irownnz;

   /* Free old rownnz pointer */
   hypre_TFree(Arownnz, memory_location);

   /* Set new rownnz pointer */
   if (irownnz == 0 || irownnz == num_rows)
   {
      hypre_CSRMatrixRownnz(matrix) = NULL;
   }
   else
   {
      Arownnz = hypre_CTAlloc(HYPRE_Int, irownnz, memory_location);
      irownnz = 0;
      for (i = 0; i < num_rows; i++)
      {
         if ((A_i[i + 1] - A_i[i]) > 0)
         {
            Arownnz[irownnz++] = i;
         }
      }
      hypre_CSRMatrixRownnz(matrix) = Arownnz;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetRownnz
 *
 * function to set the substructure rownnz and num_rowsnnz inside the CSRMatrix
 * it needs the A_i substructure of CSRMatrix to find the nonzero rows.
 * It runs after the create CSR and when A_i is known..It does not check for
 * the existence of A_i or of the CSR matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSetRownnz( hypre_CSRMatrix *matrix )
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(matrix) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      // TODO RL: there's no need currently for having rownnz on GPUs
   }
   else
#endif
   {
      hypre_CSRMatrixSetRownnzHost(matrix);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCheckSetNumNonzeros
 *
 * check if numnonzeros was properly set to be ia[nrow]
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixCheckSetNumNonzeros( hypre_CSRMatrix *matrix )
{
   if (!matrix)
   {
      return 0;
   }

   HYPRE_Int nnz, ierr = 0;

   hypre_TMemcpy(&nnz, hypre_CSRMatrixI(matrix) + hypre_CSRMatrixNumRows(matrix),
                 HYPRE_Int, 1, HYPRE_MEMORY_HOST, hypre_CSRMatrixMemoryLocation(matrix));

   if (hypre_CSRMatrixNumNonzeros(matrix) != nnz)
   {
      ierr = 1;
      hypre_printf("warning: CSR matrix nnz was not set properly (!= ia[nrow], %d %d)\n",
                   hypre_CSRMatrixNumNonzeros(matrix), nnz );
      hypre_assert(0);
      hypre_CSRMatrixNumNonzeros(matrix) = nnz;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixRead
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixRead( char *file_name )
{
   hypre_CSRMatrix  *matrix;

   FILE    *fp;

   HYPRE_Complex *matrix_data;
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

   matrix_i = hypre_CTAlloc(HYPRE_Int, num_rows + 1, HYPRE_MEMORY_HOST);
   for (j = 0; j < num_rows + 1; j++)
   {
      hypre_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;
   hypre_CSRMatrixInitialize_v2(matrix, 0, HYPRE_MEMORY_HOST);
   matrix_j = hypre_CSRMatrixJ(matrix);

   for (j = 0; j < num_nonzeros; j++)
   {
      hypre_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = hypre_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      hypre_fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   hypre_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   hypre_CSRMatrixNumCols(matrix) = ++max_col;
   hypre_CSRMatrixSetRownnz(matrix);

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPrint( hypre_CSRMatrix *matrix,
                      const char      *file_name )
{
   FILE    *fp;

   HYPRE_Complex *matrix_data;
   HYPRE_Int     *matrix_i;
   HYPRE_Int     *matrix_j;
   HYPRE_BigInt  *matrix_bigj;
   HYPRE_Int      num_rows;

   HYPRE_Int      file_base = 1;

   HYPRE_Int      j;

   HYPRE_Int      ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   matrix_data = hypre_CSRMatrixData(matrix);
   matrix_i    = hypre_CSRMatrixI(matrix);
   matrix_j    = hypre_CSRMatrixJ(matrix);
   matrix_bigj = hypre_CSRMatrixBigJ(matrix);
   num_rows    = hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j++)
   {
      hypre_fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   if (matrix_j)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
         hypre_fprintf(fp, "%d\n", matrix_j[j] + file_base);
      }
   }

   if (matrix_bigj)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
         hypre_fprintf(fp, "%d\n", matrix_bigj[j] + file_base);
      }
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
#ifdef HYPRE_COMPLEX
         hypre_fprintf(fp, "%.14e , %.14e\n",
                       hypre_creal(matrix_data[j]), hypre_cimag(matrix_data[j]));
#else
         hypre_fprintf(fp, "%.14e\n", matrix_data[j]);
#endif
      }
   }
   else
   {
      hypre_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrintIJ
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPrintIJ( hypre_CSRMatrix  *matrix,
                        HYPRE_Int         base_i,
                        HYPRE_Int         base_j,
                        char             *filename )
{
   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);
   hypre_CSRMatrix     *h_matrix;

   HYPRE_Int            patt_only;
   HYPRE_Int            num_rows;
   HYPRE_Int            num_cols;
   HYPRE_Int           *matrix_i;
   HYPRE_Int           *matrix_j;
   HYPRE_BigInt        *matrix_bj;
   HYPRE_Complex       *matrix_a;

   HYPRE_Int            i, j, ii, jj;
   HYPRE_Int            ilower, iupper, jlower, jupper;
   FILE                *file;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* Create temporary matrix on host memory if needed */
   h_matrix = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
              hypre_CSRMatrixClone_v2(matrix, 1, HYPRE_MEMORY_HOST) : matrix;

   /* Set matrix info */
   patt_only = hypre_CSRMatrixPatternOnly(h_matrix);
   num_rows  = hypre_CSRMatrixNumRows(h_matrix);
   num_cols  = hypre_CSRMatrixNumCols(h_matrix);
   matrix_i  = hypre_CSRMatrixI(h_matrix);
   matrix_j  = hypre_CSRMatrixJ(h_matrix);
   matrix_bj = hypre_CSRMatrixBigJ(h_matrix);
   matrix_a  = hypre_CSRMatrixData(h_matrix);

   if ((file = fopen(filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   /* Print matrix bounds */
   ilower = base_i;
   iupper = num_rows + base_i - 1;
   jlower = base_j;
   jupper = num_cols + base_j - 1;
   hypre_fprintf(file, "%b %b %b %b\n", ilower, iupper, jlower, jupper);

   for (i = 0; i < num_rows; i++)
   {
      ii = i + base_i;

      /* print diag columns */
      for (j = matrix_i[i]; j < matrix_i[i + 1]; j++)
      {
         jj = (matrix_bj) ? (matrix_bj[j] + base_j) : (matrix_j[j] + base_j);

         if (!patt_only)
         {
#ifdef HYPRE_COMPLEX
            hypre_fprintf(file, "%b %b %.14e , %.14e\n", ii, jj,
                          hypre_creal(matrix_a[j]), hypre_cimag(matrix_a[j]));
#else
            hypre_fprintf(file, "%b %b %.14e\n", ii, jj, matrix_a[j]);
#endif
         }
         else
         {
            hypre_fprintf(file, "%b %b\n", ii, jj);
         }
      }
   }

   fclose(file);

   /* Free temporary matrix */
   if (h_matrix != matrix)
   {
      hypre_CSRMatrixDestroy(h_matrix);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrintMM
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPrintMM( hypre_CSRMatrix *matrix,
                        HYPRE_Int        basei,
                        HYPRE_Int        basej,
                        HYPRE_Int        trans,
                        const char      *file_name )
{
   FILE *fp = file_name ? fopen(file_name, "w") : stdout;

   if (!fp)
   {
      hypre_error_w_msg(1, "Cannot open output file");
      return hypre_error_flag;
   }

   const HYPRE_Complex *matrix_data = hypre_CSRMatrixData(matrix);
   const HYPRE_Int     *matrix_i    = hypre_CSRMatrixI(matrix);
   const HYPRE_Int     *matrix_j    = hypre_CSRMatrixJ(matrix);

   hypre_assert(hypre_CSRMatrixI(matrix)[hypre_CSRMatrixNumRows(matrix)] ==
                hypre_CSRMatrixNumNonzeros(matrix));

   if (matrix_data)
   {
      hypre_fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
   }
   else
   {
      hypre_fprintf(fp, "%%%%MatrixMarket matrix coordinate pattern general\n");
   }

   hypre_fprintf(fp, "%d %d %d\n",
                 trans ? hypre_CSRMatrixNumCols(matrix) : hypre_CSRMatrixNumRows(matrix),
                 trans ? hypre_CSRMatrixNumRows(matrix) : hypre_CSRMatrixNumCols(matrix),
                 hypre_CSRMatrixNumNonzeros(matrix));

   HYPRE_Int i, j;

   for (i = 0; i < hypre_CSRMatrixNumRows(matrix); i++)
   {
      for (j = matrix_i[i]; j < matrix_i[i + 1]; j++)
      {
         const HYPRE_Int row = (trans ? matrix_j[j] : i) + basei;
         const HYPRE_Int col = (trans ? i : matrix_j[j]) + basej;
         if (matrix_data)
         {
            hypre_fprintf(fp, "%d %d %.15e\n", row, col, matrix_data[j]);
         }
         else
         {
            hypre_fprintf(fp, "%d %d\n", row, col);
         }
      }
   }

   if (file_name)
   {
      fclose(fp);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrintHB:
 *
 * Print a CSRMatrix in Harwell-Boeing format
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPrintHB( hypre_CSRMatrix *matrix_input,
                        char            *file_name )
{
   FILE            *fp;
   hypre_CSRMatrix *matrix;
   HYPRE_Complex   *matrix_data;
   HYPRE_Int       *matrix_i;
   HYPRE_Int       *matrix_j;
   HYPRE_Int        num_rows;
   HYPRE_Int        file_base = 1;
   HYPRE_Int        j, totcrd, ptrcrd, indcrd, valcrd, rhscrd;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   /* First transpose the input matrix, since HB is in CSC format */
   hypre_CSRMatrixTranspose(matrix_input, &matrix, 1);

   matrix_data = hypre_CSRMatrixData(matrix);
   matrix_i    = hypre_CSRMatrixI(matrix);
   matrix_j    = hypre_CSRMatrixJ(matrix);
   num_rows    = hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%-70s  Key     \n", "Title");
   ptrcrd = num_rows;
   indcrd = matrix_i[num_rows];
   valcrd = matrix_i[num_rows];
   rhscrd = 0;
   totcrd = ptrcrd + indcrd + valcrd + rhscrd;
   hypre_fprintf (fp, "%14d%14d%14d%14d%14d\n",
                  totcrd, ptrcrd, indcrd, valcrd, rhscrd);
   hypre_fprintf (fp, "%-14s%14i%14i%14i%14i\n", "RUA",
                  num_rows, num_rows, valcrd, 0);
   hypre_fprintf (fp, "%-16s%-16s%-16s%26s\n", "(1I8)", "(1I8)", "(1E16.8)", "");

   for (j = 0; j <= num_rows; j++)
   {
      hypre_fprintf(fp, "%8d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      hypre_fprintf(fp, "%8d\n", matrix_j[j] + file_base);
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
#ifdef HYPRE_COMPLEX
         hypre_fprintf(fp, "%16.8e , %16.8e\n",
                       hypre_creal(matrix_data[j]), hypre_cimag(matrix_data[j]));
#else
         hypre_fprintf(fp, "%16.8e\n", matrix_data[j]);
#endif
      }
   }
   else
   {
      hypre_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   hypre_CSRMatrixDestroy(matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCopy: copy A to B,
 *
 * if copy_data = 0 only the structure of A is copied to B.
 * the routine does not check if the dimensions/sparsity of A and B match !!!
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixCopy( hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int copy_data )
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros(A);

   HYPRE_Int     *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_BigInt  *A_bigj   = hypre_CSRMatrixBigJ(A);
   HYPRE_Int     *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Complex *A_data;

   HYPRE_Int     *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int     *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_BigInt  *B_bigj   = hypre_CSRMatrixBigJ(B);
   HYPRE_Int     *B_rownnz = hypre_CSRMatrixRownnz(B);
   HYPRE_Complex *B_data;

   HYPRE_MemoryLocation memory_location_A = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation memory_location_B = hypre_CSRMatrixMemoryLocation(B);

   hypre_TMemcpy(B_i, A_i, HYPRE_Int, num_rows + 1, memory_location_B, memory_location_A);

   if (A_rownnz)
   {
      if (!B_rownnz)
      {
         B_rownnz = hypre_TAlloc(HYPRE_Int,
                                 hypre_CSRMatrixNumRownnz(A),
                                 memory_location_B);
         hypre_CSRMatrixRownnz(B) = B_rownnz;
      }
      hypre_TMemcpy(B_rownnz, A_rownnz,
                    HYPRE_Int, hypre_CSRMatrixNumRownnz(A),
                    memory_location_B, memory_location_A);
   }
   hypre_CSRMatrixNumRownnz(B) = hypre_CSRMatrixNumRownnz(A);

   if (A_j && B_j)
   {
      hypre_TMemcpy(B_j, A_j, HYPRE_Int, num_nonzeros, memory_location_B, memory_location_A);
   }

   if (A_bigj && B_bigj)
   {
      hypre_TMemcpy(B_bigj, A_bigj, HYPRE_BigInt, num_nonzeros,
                    memory_location_B, memory_location_A);
   }

   if (copy_data)
   {
      A_data = hypre_CSRMatrixData(A);
      B_data = hypre_CSRMatrixData(B);
      hypre_TMemcpy(B_data, A_data, HYPRE_Complex, num_nonzeros,
                    memory_location_B, memory_location_A);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMigrate
 *
 * Migrates matrix row pointer, column indices and data to memory_location
 * if it is different to the current one.
 *
 * Note: Does not move rownnz array.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMigrate( hypre_CSRMatrix     *A,
                        HYPRE_MemoryLocation memory_location )
{
   /* Input matrix info */
   HYPRE_Int       num_rows     = hypre_CSRMatrixNumRows(A);
   HYPRE_Int       num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int      *A_ri         = hypre_CSRMatrixRownnz(A);
   HYPRE_Int      *A_i          = hypre_CSRMatrixI(A);
   HYPRE_Int      *A_j          = hypre_CSRMatrixJ(A);
   HYPRE_BigInt   *A_big_j      = hypre_CSRMatrixBigJ(A);
   HYPRE_Complex  *A_data       = hypre_CSRMatrixData(A);

   HYPRE_MemoryLocation old_memory_location = hypre_CSRMatrixMemoryLocation(A);

   /* Output matrix info */
   HYPRE_Int      *B_i;
   HYPRE_Int      *B_j;
   HYPRE_BigInt   *B_big_j;
   HYPRE_Complex  *B_data;
   HYPRE_Int      *B_ri;

   /* Check pointer locations in debug mode */
#if defined(HYPRE_DEBUG)
   hypre_CheckMemoryLocation((void*) A_ri,    hypre_GetActualMemLocation(old_memory_location));
   hypre_CheckMemoryLocation((void*) A_i,     hypre_GetActualMemLocation(old_memory_location));
   hypre_CheckMemoryLocation((void*) A_j,     hypre_GetActualMemLocation(old_memory_location));
   hypre_CheckMemoryLocation((void*) A_big_j, hypre_GetActualMemLocation(old_memory_location));
   hypre_CheckMemoryLocation((void*) A_data,  hypre_GetActualMemLocation(old_memory_location));
#endif

   /* Update A's memory location */
   hypre_CSRMatrixMemoryLocation(A) = memory_location;

   if ( hypre_GetActualMemLocation(memory_location) !=
        hypre_GetActualMemLocation(old_memory_location) )
   {
      if (A_ri)
      {
         B_ri = hypre_TAlloc(HYPRE_Int, num_rows, memory_location);
         hypre_TMemcpy(B_ri, A_ri, HYPRE_Int, num_rows,
                       memory_location, old_memory_location);
         hypre_TFree(A_ri, old_memory_location);
         hypre_CSRMatrixRownnz(A) = B_ri;
      }

      if (A_i)
      {
         B_i = hypre_TAlloc(HYPRE_Int, num_rows + 1, memory_location);
         hypre_TMemcpy(B_i, A_i, HYPRE_Int, num_rows + 1,
                       memory_location, old_memory_location);
         hypre_TFree(A_i, old_memory_location);
         hypre_CSRMatrixI(A) = B_i;
      }

      if (A_j)
      {
         B_j = hypre_TAlloc(HYPRE_Int, num_nonzeros, memory_location);
         hypre_TMemcpy(B_j, A_j, HYPRE_Int, num_nonzeros,
                       memory_location, old_memory_location);
         hypre_TFree(A_j, old_memory_location);
         hypre_CSRMatrixJ(A) = B_j;
      }

      if (A_big_j)
      {
         B_big_j = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, memory_location);
         hypre_TMemcpy(B_big_j, A_big_j, HYPRE_BigInt, num_nonzeros,
                       memory_location, old_memory_location);
         hypre_TFree(A_big_j, old_memory_location);
         hypre_CSRMatrixBigJ(A) = B_big_j;
      }

      if (A_data)
      {
         B_data = hypre_TAlloc(HYPRE_Complex, num_nonzeros, memory_location);
         hypre_TMemcpy(B_data, A_data, HYPRE_Complex, num_nonzeros,
                       memory_location, old_memory_location);
         hypre_TFree(A_data, old_memory_location);
         hypre_CSRMatrixData(A) = B_data;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixClone_v2
 *
 * This function does the same job as hypre_CSRMatrixClone; however, here
 * the user can specify the memory location of the resulting matrix.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixClone_v2( hypre_CSRMatrix *A, HYPRE_Int copy_data,
                         HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);
   HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros(A);

   hypre_CSRMatrix *B = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);

   HYPRE_Int bigInit = hypre_CSRMatrixBigJ(A) != NULL;

   hypre_CSRMatrixInitialize_v2(B, bigInit, memory_location);

   hypre_CSRMatrixCopy(A, B, copy_data);

   return B;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixClone
 *
 * Creates and returns a new copy of the argument, A.
 * Performs a deep copy of information (no pointers are copied);
 * New arrays are created where necessary.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixClone( hypre_CSRMatrix *A, HYPRE_Int copy_data )
{
   return hypre_CSRMatrixClone_v2(A, copy_data, hypre_CSRMatrixMemoryLocation(A));
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPermuteHost
 *
 * See hypre_CSRMatrixPermute. TODO (VPM): OpenMP implementation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPermuteHost( hypre_CSRMatrix  *A,
                            HYPRE_Int        *perm,
                            HYPRE_Int        *rqperm,
                            hypre_CSRMatrix  *B )
{
   /* Input variables */
   HYPRE_Int         num_rows     = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int        *A_i          = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j          = hypre_CSRMatrixJ(A);
   HYPRE_Complex    *A_a          = hypre_CSRMatrixData(A);
   HYPRE_Int        *B_i          = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j          = hypre_CSRMatrixJ(B);
   HYPRE_Complex    *B_a          = hypre_CSRMatrixData(B);

   /* Local variables */
   HYPRE_Int         i, j, k;

   /* Build B = A(perm, qperm) */
   k = 0;
   for (i = 0; i < num_rows; i++)
   {
      B_i[i] = k;
      for (j = A_i[perm[i]]; j < A_i[perm[i] + 1]; j++)
      {
         B_j[k] = rqperm[A_j[j]];
         B_a[k++] = A_a[j];
      }
   }
   B_i[num_rows] = k;
   hypre_assert(k == num_nonzeros);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPermute
 *
 * Reorder a CSRMatrix according to a row-permutation array (perm) and
 * reverse column-permutation array (rqperm).
 *
 * Notes:
 *  1) This function does not move the diagonal to the first entry of a row
 *  2) When perm == rqperm == NULL, B is a deep copy of A.
 *
 * TODO (VPM): add check for permutation arrays under HYPRE_DEBUG
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPermute( hypre_CSRMatrix  *A,
                        HYPRE_Int        *perm,
                        HYPRE_Int        *rqperm,
                        hypre_CSRMatrix **B_ptr )
{
   HYPRE_Int          num_rows     = hypre_CSRMatrixNumRows(A);
   HYPRE_Int          num_cols     = hypre_CSRMatrixNumCols(A);
   HYPRE_Int          num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   hypre_CSRMatrix   *B;

   hypre_GpuProfilingPushRange("CSRMatrixPermute");

   /* Special case: one of the permutation vectors are not provided, then B = A */
   if (!perm || !rqperm)
   {
      *B_ptr = hypre_CSRMatrixClone(A, 1);
      hypre_GpuProfilingPopRange();

      return hypre_error_flag;
   }

   /* Create output matrix B */
   B = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   hypre_CSRMatrixInitialize_v2(B, 0, hypre_CSRMatrixMemoryLocation(A));

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_CSRMatrixPermuteDevice(A, perm, rqperm, B);
   }
   else
#endif
   {
      hypre_CSRMatrixPermuteHost(A, perm, rqperm, B);
   }

   hypre_GpuProfilingPopRange();

   /* Set output pointer */
   *B_ptr = B;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixUnion
 * Creates and returns a matrix whose elements are the union of those of A and B.
 * Data is not computed, only structural information is created.
 * A and B must have the same numbers of rows.
 * Nothing is done about Rownnz.
 *
 * If col_map_offd_A and col_map_offd_B are zero, A and B are expected to have
 * the same column indexing.  Otherwise, col_map_offd_A, col_map_offd_B should
 * be the arrays of that name from two ParCSRMatrices of which A and B are the
 * offd blocks.
 *
 * The algorithm can be expected to have reasonable efficiency only for very
 * sparse matrices (many rows, few nonzeros per row).
 * The nonzeros of a computed row are NOT necessarily in any particular order.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixUnion( hypre_CSRMatrix *A,
                      hypre_CSRMatrix *B,
                      HYPRE_BigInt *col_map_offd_A,
                      HYPRE_BigInt *col_map_offd_B,
                      HYPRE_BigInt **col_map_offd_C )
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows( A );
   HYPRE_Int num_cols_A = hypre_CSRMatrixNumCols( A );
   HYPRE_Int num_cols_B = hypre_CSRMatrixNumCols( B );
   HYPRE_Int num_cols;
   HYPRE_Int num_nonzeros;
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Int *B_i = hypre_CSRMatrixI(B);
   HYPRE_Int *B_j = hypre_CSRMatrixJ(B);
   HYPRE_Int *C_i;
   HYPRE_Int *C_j;
   HYPRE_Int *jC = NULL;
   HYPRE_BigInt jBg, big_jA = -1, big_jB = -1;
   HYPRE_Int i, jA, jB;
   HYPRE_Int ma, mb, mc, ma_min, ma_max, match;
   hypre_CSRMatrix* C;

   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(A);

   hypre_assert( num_rows == hypre_CSRMatrixNumRows(B) );

   if ( col_map_offd_B )
   {
      hypre_assert( col_map_offd_A );
   }

   if ( col_map_offd_A )
   {
      hypre_assert( col_map_offd_B );
   }

   /* ==== First, go through the columns of A and B to count the columns of C. */
   if ( col_map_offd_A == 0 )
   {
      /* The matrices are diagonal blocks.
         Normally num_cols_A==num_cols_B, col_starts is the same, etc.
      */
      num_cols = hypre_max( num_cols_A, num_cols_B );
   }
   else
   {
      /* The matrices are offdiagonal blocks. */
      jC = hypre_CTAlloc(HYPRE_Int, num_cols_B, HYPRE_MEMORY_HOST);
      num_cols = num_cols_A;  /* initialization; we'll compute the actual value */
      for ( jB = 0; jB < num_cols_B; ++jB )
      {
         match = 0;
         jBg = col_map_offd_B[jB];
         for ( ma = 0; ma < num_cols_A; ++ma )
         {
            if ( col_map_offd_A[ma] == jBg )
            {
               match = 1;
            }
         }
         if ( match == 0 )
         {
            jC[jB] = num_cols;
            ++num_cols;
         }
      }
   }

   /* ==== If we're working on a ParCSRMatrix's offd block,
      make and load col_map_offd_C */
   if ( col_map_offd_A )
   {
      *col_map_offd_C = hypre_CTAlloc( HYPRE_BigInt, num_cols, HYPRE_MEMORY_HOST);
      for ( jA = 0; jA < num_cols_A; ++jA )
      {
         (*col_map_offd_C)[jA] = col_map_offd_A[jA];
      }
      for ( jB = 0; jB < num_cols_B; ++jB )
      {
         match = 0;
         jBg = col_map_offd_B[jB];
         for ( ma = 0; ma < num_cols_A; ++ma )
         {
            if ( col_map_offd_A[ma] == jBg )
            {
               match = 1;
            }
         }
         if ( match == 0 )
         {
            (*col_map_offd_C)[ jC[jB] ] = jBg;
         }
      }
   }


   /* ==== The first run through A and B is to count the number of nonzero elements,
      without HYPRE_Complex-counting duplicates.  Then we can create C. */
   num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   for ( i = 0; i < num_rows; ++i )
   {
      ma_min = A_i[i];  ma_max = A_i[i + 1];
      for ( mb = B_i[i]; mb < B_i[i + 1]; ++mb )
      {
         jB = B_j[mb];
         if ( col_map_offd_B )
         {
            big_jB = col_map_offd_B[jB];
         }
         match = 0;
         for ( ma = ma_min; ma < ma_max; ++ma )
         {
            jA = A_j[ma];
            if ( col_map_offd_A )
            {
               big_jA = col_map_offd_A[jA];
            }
            if ( big_jB == big_jA )
            {
               match = 1;
               if ( ma == ma_min )
               {
                  ++ma_min;
               }
               break;
            }
         }
         if ( match == 0 )
         {
            ++num_nonzeros;
         }
      }
   }

   C = hypre_CSRMatrixCreate( num_rows, num_cols, num_nonzeros );
   hypre_CSRMatrixInitialize_v2( C, 0, memory_location );

   /* ==== The second run through A and B is to pick out the column numbers
      for each row, and put them in C. */
   C_i = hypre_CSRMatrixI(C);
   C_i[0] = 0;
   C_j = hypre_CSRMatrixJ(C);
   mc = 0;
   for ( i = 0; i < num_rows; ++i )
   {
      ma_min = A_i[i];
      ma_max = A_i[i + 1];
      for ( ma = ma_min; ma < ma_max; ++ma )
      {
         C_j[mc] = A_j[ma];
         ++mc;
      }
      for ( mb = B_i[i]; mb < B_i[i + 1]; ++mb )
      {
         jB = B_j[mb];
         if ( col_map_offd_B )
         {
            big_jB = col_map_offd_B[jB];
         }
         match = 0;
         for ( ma = ma_min; ma < ma_max; ++ma )
         {
            jA = A_j[ma];
            if ( col_map_offd_A )
            {
               big_jA = col_map_offd_A[jA];
            }
            if ( big_jB == big_jA )
            {
               match = 1;
               if ( ma == ma_min )
               {
                  ++ma_min;
               }
               break;
            }
         }
         if ( match == 0 )
         {
            if ( col_map_offd_A )
            {
               C_j[mc] = jC[ B_j[mb] ];
            }
            else
            {
               C_j[mc] = B_j[mb];
            }
            /* ... I don't know whether column indices are required to be in any
               particular order.  If so, we'll need to sort. */
            ++mc;
         }
      }
      C_i[i + 1] = mc;
   }

   hypre_assert( mc == num_nonzeros );

   hypre_TFree(jC, HYPRE_MEMORY_HOST);

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixGetLoadBalancedPartitionBoundary
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_CSRMatrixGetLoadBalancedPartitionBoundary(hypre_CSRMatrix *A,
                                                HYPRE_Int        idx)
{
   HYPRE_Int num_nonzerosA = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int num_rowsA = hypre_CSRMatrixNumRows(A);
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);

   HYPRE_Int num_threads = hypre_NumActiveThreads();

   HYPRE_Int nonzeros_per_thread = (num_nonzerosA + num_threads - 1) / num_threads;

   if (idx <= 0)
   {
      return 0;
   }
   else if (idx >= num_threads)
   {
      return num_rowsA;
   }
   else
   {
      return (HYPRE_Int)(hypre_LowerBound(A_i, A_i + num_rowsA, nonzeros_per_thread * idx) - A_i);
   }
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixGetLoadBalancedPartitionBegin
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixGetLoadBalancedPartitionBegin(hypre_CSRMatrix *A)
{
   return hypre_CSRMatrixGetLoadBalancedPartitionBoundary(A, hypre_GetThreadNum());
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixGetLoadBalancedPartitionEnd
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixGetLoadBalancedPartitionEnd(hypre_CSRMatrix *A)
{
   return hypre_CSRMatrixGetLoadBalancedPartitionBoundary(A, hypre_GetThreadNum() + 1);
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrefetch
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPrefetch( hypre_CSRMatrix      *A,
                         HYPRE_MemoryLocation  memory_location )
{
#if defined(HYPRE_USING_UNIFIED_MEMORY)
   if (hypre_CSRMatrixMemoryLocation(A) != HYPRE_MEMORY_DEVICE)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "A is not at HYPRE_MEMORY_DEVICE");
      return hypre_error_flag;
   }

   HYPRE_Complex *data = hypre_CSRMatrixData(A);
   HYPRE_Int     *ia   = hypre_CSRMatrixI(A);
   HYPRE_Int     *ja   = hypre_CSRMatrixJ(A);
   HYPRE_Int      nrow = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      nnzA = hypre_CSRMatrixNumNonzeros(A);

   hypre_MemPrefetch(data, sizeof(HYPRE_Complex)*nnzA, memory_location);
   hypre_MemPrefetch(ia,   sizeof(HYPRE_Int) * (nrow + 1), memory_location);
   hypre_MemPrefetch(ja,   sizeof(HYPRE_Int)*nnzA,     memory_location);

#else
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(memory_location);
#endif

   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)  ||\
    defined(HYPRE_USING_ROCSPARSE) ||\
    defined(HYPRE_USING_ONEMKLSPARSE)
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixGetGPUMatData
 *--------------------------------------------------------------------------*/

hypre_GpuMatData*
hypre_CSRMatrixGetGPUMatData(hypre_CSRMatrix *matrix)
{
   if (!matrix)
   {
      return NULL;
   }

   if (!hypre_CSRMatrixGPUMatData(matrix))
   {
      hypre_CSRMatrixGPUMatData(matrix) = hypre_GpuMatDataCreate();
      hypre_GPUMatDataSetCSRData(matrix);
   }

   return hypre_CSRMatrixGPUMatData(matrix);
}
#endif
