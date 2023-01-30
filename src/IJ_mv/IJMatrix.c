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
 * is stored on host memory.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJMatrixReadBinary( const char      *prefixname,
                          MPI_Comm         comm,
                          HYPRE_Int        type,
                          HYPRE_IJMatrix  *matrix_ptr )
{
   HYPRE_IJMatrix  matrix;

   HYPRE_BigInt    ilower, iupper, jlower, jupper;
   HYPRE_BigInt    I, J;

   size_t          count;
   uint64_t        header[8];

   /* Local buffers */
   uint32_t             *i32rows = NULL;
   uint64_t             *i64rows = NULL;
   uint32_t             *i32cols = NULL;
   uint64_t             *i64cols = NULL;
   float                *f32vals = NULL;
   hypre_double         *f64vals = NULL;

   HYPRE_Int       myid;
   char            filename[1024];
   FILE           *fp;

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
    * Read header (64 bytes) from file
    *---------------------------------------------*/

   count = 8;
   if (fread(header, sizeof(uint64_t), count, fp) != count)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read header entries\n");
      return EXIT_FAILURE;
   }

   /* Exit if trying to read from big-endian machine */
   if ((*(char*)&header[7]) == 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Support to big-endian machines is incomplete!\n");
      return hypre_error_flag;
   }

   /* Set variables */
   ilower = (HYPRE_BigInt) header[3];
   iupper = (HYPRE_BigInt) header[4];
   jlower = (HYPRE_BigInt) header[5];
   jupper = (HYPRE_BigInt) header[6];

   /* Allocate memory for row/col buffers */
   if (header[0] == sizeof(uint32_t))
   {
      i32rows = hypre_TAlloc(uint32_t, header[2], HYPRE_MEMORY_HOST);
      i32cols = hypre_TAlloc(uint32_t, header[2], HYPRE_MEMORY_HOST);
   }
   else if (header[0] == sizeof(uint64_t))
   {
      i64rows = hypre_TAlloc(uint64_t, header[2], HYPRE_MEMORY_HOST);
      i64cols = hypre_TAlloc(uint64_t, header[2], HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported data type for row/column indices\n");
      return hypre_error_flag;
   }

   /* Allocate memory for buffers */
   if (header[1] == sizeof(float))
   {
      f32vals = hypre_TAlloc(float, header[2], HYPRE_MEMORY_HOST);
   }
   else if (header[1] == sizeof(hypre_double))
   {
      f64vals = hypre_TAlloc(hypre_double, header[2], HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported data type for matrix coefficients\n");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Read row and column indices from file
    *---------------------------------------------*/

   count = (size_t) header[2];
   if (i32rows)
   {
      if (fread(i32rows, sizeof(uint32_t), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all row indices\n");
         return hypre_error_flag;
      }

      if (fread(i32cols, sizeof(uint32_t), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all column indices\n");
         return hypre_error_flag;
      }
   }
   else
   {
      if (fread(i64rows, sizeof(uint64_t), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all row indices\n");
         return hypre_error_flag;
      }

      if (fread(i64cols, sizeof(uint64_t), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all column indices\n");
         return hypre_error_flag;
      }
   }

   /*---------------------------------------------
    * Read matrix coefficients from file
    *---------------------------------------------*/

   if (f32vals)
   {
      if (fread(f32vals, sizeof(float), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all matrix coefficients\n");
         return hypre_error_flag;
      }
   }
   else
   {
      if (fread(f64vals, sizeof(hypre_double), count, fp) != count)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not read all matrix coefficients\n");
         return hypre_error_flag;
      }
   }

   /* Close file stream */
   fclose(fp);

   /*---------------------------------------------
    * Build IJMatrix
    *---------------------------------------------*/

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &matrix);
   HYPRE_IJMatrixSetObjectType(matrix, type);
   HYPRE_IJMatrixInitialize_v2(matrix, HYPRE_MEMORY_HOST);


   if (i32rows && i32cols && f32vals)
   {
      HYPRE_IJMatrixSetValues(matrix, header[2], NULL, i32rows, i32cols, f32vals);
   }
   else if (i32rows && i32cols && f64vals)
   {
      HYPRE_IJMatrixSetValues(matrix, header[2], NULL, i32rows, i32cols, f64vals);
   }
   else if (i64rows && i64cols && f32vals)
   {
      HYPRE_IJMatrixSetValues(matrix, header[2], NULL, i64rows, i64cols, f32vals);
   }
   else
   {
      HYPRE_IJMatrixSetValues(matrix, header[2], NULL, i64rows, i64cols, f64vals);
   }


   HYPRE_IJMatrixAssemble(matrix);
   *matrix_ptr = matrix;

   return hypre_error_flag;
}
