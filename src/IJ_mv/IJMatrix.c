/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre_IJMatrix interface
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetRowPartitioning
 *
 * Returns a pointer to the row partitioning of an IJMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJMatrixGetRowPartitioning( HYPRE_IJMatrix matrix,
                                  HYPRE_BigInt **row_partitioning )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Variable ijmatrix is NULL -- hypre_IJMatrixGetRowPartitioning\n");
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixRowPartitioning(ijmatrix))
   {
      *row_partitioning = hypre_IJMatrixRowPartitioning(ijmatrix);
   }
   else
   {
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetColPartitioning
 *
 * Returns a pointer to the column partitioning of an IJMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJMatrixGetColPartitioning( HYPRE_IJMatrix matrix,
                                  HYPRE_BigInt **col_partitioning )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Variable ijmatrix is NULL -- hypre_IJMatrixGetColPartitioning\n");
      return hypre_error_flag;
   }

   if ( hypre_IJMatrixColPartitioning(ijmatrix))
   {
      *col_partitioning = hypre_IJMatrixColPartitioning(ijmatrix);
   }
   else
   {
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJMatrixSetObject( HYPRE_IJMatrix  matrix,
                         void           *object )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (hypre_IJMatrixObject(ijmatrix) != NULL)
   {
      /*hypre_printf("Referencing a new IJMatrix object can orphan an old -- ");
      hypre_printf("hypre_IJMatrixSetObject\n");*/
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   hypre_IJMatrixObject(ijmatrix) = object;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IJMatrixRead
 *
 * Reads a matrix from file, HYPRE's IJ format or MM format. The resulting
 * IJMatrix is stored on host memory.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJMatrixRead( const char     *filename,
                    MPI_Comm        comm,
                    HYPRE_Int       type,
                    HYPRE_IJMatrix *matrix_ptr,
                    HYPRE_Int       is_mm )
{
   HYPRE_IJMatrix  matrix;
   HYPRE_BigInt    ilower, iupper, jlower, jupper;
   HYPRE_BigInt    I, J;
   HYPRE_Int       ncols;
   HYPRE_Complex   value;
   HYPRE_Int       myid, ret;
   HYPRE_Int       isSym = 0;
   char            new_filename[255];
   FILE           *file;

   hypre_MPI_Comm_rank(comm, &myid);

   if (is_mm)
   {
      hypre_sprintf(new_filename, "%s", filename);
   }
   else
   {
      hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   }

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (is_mm)
   {
      MM_typecode matcode;
      HYPRE_Int nrow, ncol, nnz;

      if (hypre_mm_read_banner(file, &matcode) != 0)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not process Matrix Market banner.");
         return hypre_error_flag;
      }

      if (!hypre_mm_is_valid(matcode))
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid Matrix Market file.");
         return hypre_error_flag;
      }

      if ( !( (hypre_mm_is_real(matcode) || hypre_mm_is_integer(matcode)) &&
              hypre_mm_is_coordinate(matcode) && hypre_mm_is_sparse(matcode) ) )
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Only sparse real-valued/integer coordinate matrices are supported");
         return hypre_error_flag;
      }

      if (hypre_mm_is_symmetric(matcode))
      {
         isSym = 1;
      }

      if (hypre_mm_read_mtx_crd_size(file, &nrow, &ncol, &nnz) != 0)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "MM read size error !");
         return hypre_error_flag;
      }

      ilower = 0;
      iupper = ilower + nrow - 1;
      jlower = 0;
      jupper = jlower + ncol - 1;
   }
   else
   {
      hypre_fscanf(file, "%b %b %b %b", &ilower, &iupper, &jlower, &jupper);
   }

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &matrix);

   HYPRE_IJMatrixSetObjectType(matrix, type);

   HYPRE_IJMatrixInitialize_v2(matrix, HYPRE_MEMORY_HOST);

   /* It is important to ensure that whitespace follows the index value to help
    * catch mistakes in the input file.  See comments in IJVectorRead(). */
   ncols = 1;
   while ( (ret = hypre_fscanf(file, "%b %b%*[ \t]%le", &I, &J, &value)) != EOF )
   {
      if (ret != 3)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error in IJ matrix input file.");
         return hypre_error_flag;
      }

      if (is_mm)
      {
         I --;
         J --;
      }

      if (I < ilower || I > iupper)
      {
         HYPRE_IJMatrixAddToValues(matrix, 1, &ncols, &I, &J, &value);
      }
      else
      {
         HYPRE_IJMatrixSetValues(matrix, 1, &ncols, &I, &J, &value);
      }

      if (isSym && I != J)
      {
         if (J < ilower || J > iupper)
         {
            HYPRE_IJMatrixAddToValues(matrix, 1, &ncols, &J, &I, &value);
         }
         else
         {
            HYPRE_IJMatrixSetValues(matrix, 1, &ncols, &J, &I, &value);
         }
      }
   }

   HYPRE_IJMatrixAssemble(matrix);

   fclose(file);

   *matrix_ptr = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IJMatrixReadBinary
 *
 * Reads a matrix from file stored in binary format. The resulting IJMatrix
 * is stored on host memory. For information about the metadata contents
 * contained in the file header, see hypre_ParCSRMatrixPrintBinaryIJ.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJMatrixReadBinary( const char      *prefixname,
                          MPI_Comm         comm,
                          HYPRE_Int        type,
                          HYPRE_IJMatrix  *matrix_ptr )
{
   HYPRE_IJMatrix  matrix;

   /* Local buffers */
   hypre_uint32   *i32buffer = NULL;
   hypre_uint64   *i64buffer = NULL;
   hypre_float    *f32buffer = NULL;
   hypre_double   *f64buffer = NULL;

   /* Matrix buffers */
   HYPRE_Int       num_nonzeros;
   HYPRE_BigInt   *rows;
   HYPRE_BigInt   *cols;
   HYPRE_Complex  *vals;

   /* Local variables */
   HYPRE_Int       one = 1;
   HYPRE_Int       myid;
   char            filename[1024], msg[1024];
   HYPRE_BigInt    i, ilower, iupper, jlower, jupper;
   size_t          count;
   hypre_uint64    header[11];
   FILE           *fp;

   /* Exit if trying to read from big-endian machine */
   if ((*(char*)&one) == 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Support to big-endian machines is incomplete!");
      return hypre_error_flag;
   }

   /* Set filename */
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(filename, "%s.%05d.bin", prefixname, myid);

   /* Open file */
   if ((fp = fopen(filename, "rb")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not open input file\n");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Read header (88 bytes) from file
    *---------------------------------------------*/

   count = 11;
   if (fread(header, sizeof(hypre_uint64), count, fp) != count)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read header entries\n");
      return EXIT_FAILURE;
   }

   /* Check for header version */
   if (header[0] != 1)
   {
      hypre_sprintf(msg, "Unsupported header version: %d", header[0]);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }

   /* Check for integer overflow */
   if (header[6] > HYPRE_INT_MAX)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Detected integer overflow at 7th header entry");
      return hypre_error_flag;
   }
   num_nonzeros = (HYPRE_Int) header[6];

   /* Set variables */
   ilower = (HYPRE_BigInt) header[7];
   iupper = (HYPRE_BigInt) header[8];
   jlower = (HYPRE_BigInt) header[9];
   jupper = (HYPRE_BigInt) header[10];

   /* Allocate memory for row/col buffers */
   if (header[1] == sizeof(hypre_uint32))
   {
      i32buffer = hypre_TAlloc(hypre_uint32, num_nonzeros, HYPRE_MEMORY_HOST);
   }
   else if (header[1] == sizeof(hypre_uint64))
   {
      i64buffer = hypre_TAlloc(hypre_uint64, num_nonzeros, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported data type for row/column indices");
      return hypre_error_flag;
   }

   /* Allocate memory for buffers */
   if (header[2] == sizeof(hypre_float))
   {
      f32buffer = hypre_TAlloc(hypre_float, num_nonzeros, HYPRE_MEMORY_HOST);
   }
   else if (header[2] == sizeof(hypre_double))
   {
      f64buffer = hypre_TAlloc(hypre_double, num_nonzeros, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported data type for matrix coefficients");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Read indices from file
    *---------------------------------------------*/

   count = (size_t) num_nonzeros;
   rows = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_HOST);
   if (i32buffer)
   {
      if (fread(i32buffer, sizeof(hypre_uint32), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all row indices");
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_nonzeros; i++)
      {
         rows[i] = (HYPRE_BigInt) i32buffer[i];
      }
   }
   else
   {
      if (fread(i64buffer, sizeof(hypre_uint64), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all row indices");
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_nonzeros; i++)
      {
         rows[i] = (HYPRE_BigInt) i64buffer[i];
      }
   }

   /*---------------------------------------------
    * Read column indices from file
    *---------------------------------------------*/

   count = (size_t) num_nonzeros;
   cols = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_HOST);
   if (i32buffer)
   {
      if (fread(i32buffer, sizeof(hypre_uint32), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all column indices");
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_nonzeros; i++)
      {
         cols[i] = (HYPRE_BigInt) i32buffer[i];
      }
   }
   else
   {
      if (fread(i64buffer, sizeof(hypre_uint64), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all column indices");
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_nonzeros; i++)
      {
         cols[i] = (HYPRE_BigInt) i64buffer[i];
      }
   }

   /* Free integer buffers */
   hypre_TFree(i32buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(i64buffer, HYPRE_MEMORY_HOST);

   /*---------------------------------------------
    * Read matrix coefficients from file
    *---------------------------------------------*/

   vals = hypre_TAlloc(HYPRE_Complex, num_nonzeros, HYPRE_MEMORY_HOST);
   if (f32buffer)
   {
      if (fread(f32buffer, sizeof(hypre_float), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all matrix coefficients");
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_nonzeros; i++)
      {
         vals[i] = (HYPRE_Complex) f32buffer[i];
      }
   }
   else
   {
      if (fread(f64buffer, sizeof(hypre_double), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all matrix coefficients");
         return hypre_error_flag;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_nonzeros; i++)
      {
         vals[i] = (HYPRE_Complex) f64buffer[i];
      }
   }

   /* Close file stream */
   fclose(fp);

   /* Free floating-point buffers */
   hypre_TFree(f32buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(f64buffer, HYPRE_MEMORY_HOST);

   /*---------------------------------------------
    * Build IJMatrix
    *---------------------------------------------*/

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &matrix);
   HYPRE_IJMatrixSetObjectType(matrix, type);
   HYPRE_IJMatrixInitialize_v2(matrix, HYPRE_MEMORY_HOST);
   HYPRE_IJMatrixSetValues(matrix, num_nonzeros, NULL, rows, cols, vals);
   HYPRE_IJMatrixAssemble(matrix);

   /* Set output pointer */
   *matrix_ptr = matrix;

   /* Free memory */
   hypre_TFree(rows, HYPRE_MEMORY_HOST);
   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
