/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_CSRMatrix interface
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_CSRMatrix
HYPRE_CSRMatrixCreate( HYPRE_Int  num_rows,
                       HYPRE_Int  num_cols,
                       HYPRE_Int *row_sizes )
{
   hypre_CSRMatrix *matrix;
   HYPRE_Int             *matrix_i;
   HYPRE_Int              i;

   matrix_i = hypre_CTAlloc(HYPRE_Int,  num_rows + 1, HYPRE_MEMORY_HOST);
   matrix_i[0] = 0;
   for (i = 0; i < num_rows; i++)
   {
      matrix_i[i+1] = matrix_i[i] + row_sizes[i];
   }

   matrix = hypre_CSRMatrixCreate(num_rows, num_cols, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;

   return ( (HYPRE_CSRMatrix) matrix );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRMatrixDestroy( HYPRE_CSRMatrix matrix )
{
   return( hypre_CSRMatrixDestroy( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRMatrixInitialize( HYPRE_CSRMatrix matrix )
{
   return ( hypre_CSRMatrixInitialize( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixRead
 *--------------------------------------------------------------------------*/

HYPRE_CSRMatrix
HYPRE_CSRMatrixRead( char            *file_name )
{
   return ( (HYPRE_CSRMatrix) hypre_CSRMatrixRead( file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

void
HYPRE_CSRMatrixPrint( HYPRE_CSRMatrix  matrix,
                      char            *file_name )
{
   hypre_CSRMatrixPrint( (hypre_CSRMatrix *) matrix,
                         file_name );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixGetNumRows
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRMatrixGetNumRows( HYPRE_CSRMatrix matrix, HYPRE_Int *num_rows )
{
   hypre_CSRMatrix *csr_matrix = (hypre_CSRMatrix *) matrix;

   *num_rows =  hypre_CSRMatrixNumRows( csr_matrix );

   return 0;
}

HYPRE_Int
HYPRE_CSRMatrixDeviceSpGemmSetRownnzEstimateMethod( HYPRE_Int value )
{
#if defined(HYPRE_USING_CUDA)
   if (hypre_handle == NULL)
   {
      return -1;
   }

   if (value == 1 || value == 2 || value == 3)
   {
      hypre_handle->spgemm_rownnz_estimate_method = value;
   }
   else
   {
      return -1;
   }
#endif

   return 0;
}

HYPRE_Int
HYPRE_CSRMatrixDeviceSpGemmSetRownnzEstimateNSamples( HYPRE_Int value )
{
#if defined(HYPRE_USING_CUDA)
   if (hypre_handle == NULL)
   {
      return -1;
   }

   hypre_handle->spgemm_rownnz_estimate_nsamples = value;
#endif

   return 0;
}

HYPRE_Int
HYPRE_CSRMatrixDeviceSpGemmSetRownnzEstimateMultFactor( HYPRE_Real value )
{
#if defined(HYPRE_USING_CUDA)
   if (hypre_handle == NULL)
   {
      return -1;
   }

   if (value > 0.0)
   {
      hypre_handle->spgemm_rownnz_estimate_mult_factor = value;
   }
   else
   {
      return -1;
   }
#endif

   return 0;
}

HYPRE_Int
HYPRE_CSRMatrixDeviceSpGemmSetHashType( char value )
{
#if defined(HYPRE_USING_CUDA)
   if (hypre_handle == NULL)
   {
      return -1;
   }

   if (value == 'L' || value == 'Q' || value == 'D')
   {
      hypre_handle->spgemm_hash_type = value;
   }
#endif

   return 0;
}

HYPRE_Int
HYPRE_CSRMatrixDeviceSpGemmSetUseCusparse( HYPRE_Int value )
{
#if defined(HYPRE_USING_CUDA)
   if (hypre_handle == NULL)
   {
      return -1;
   }

   hypre_handle->spgemm_use_cusparse = value != 0;
#endif

   return 0;
}

