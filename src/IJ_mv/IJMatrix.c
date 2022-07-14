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

#include "./_hypre_IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the row partitioning

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

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
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the column partitioning

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

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
 * hypre_IJMatrixRead: Read from file, HYPRE's IJ format or MM format
 * create IJMatrix on host memory
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IJMatrixRead( const char     *filename,
                    MPI_Comm        comm,
                    HYPRE_Int       type,
                    HYPRE_IJMatrix *matrix_ptr,
                    HYPRE_Int       is_mm       /* if is a Matrix-Market file */)
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

