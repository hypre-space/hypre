/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre seq_mv mixed-precision interface
 *
 *****************************************************************************/

#include "_hypre_seq_mv.h"

#if defined(HYPRE_MIXED_PRECISION)

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Mixed precision hypre_SeqVectorCopy
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SeqVectorCopy_mp( hypre_Vector *x,
                        hypre_Vector *y )
{
   HYPRE_Int      size;
   /* Generic pointer type */
   void               *xp, *yp;

   /* Call standard vector copy if precisions match. */
   if (hypre_VectorPrecision (y) == hypre_VectorPrecision (x))
   {
      return HYPRE_VectorCopy_pre(hypre_VectorPrecision (y), (HYPRE_Vector)x, (HYPRE_Vector)y);
   }

   size = hypre_min(hypre_VectorSize(x), hypre_VectorSize(y)) * hypre_VectorNumVectors(x);

   xp = hypre_VectorData(x);
   yp = hypre_VectorData(y);
   /* copy data */
   hypre_RealArrayCopy_mp(hypre_VectorPrecision (x), xp, hypre_VectorMemoryLocation(y),
   			  hypre_VectorPrecision (y), yp, hypre_VectorMemoryLocation(y), size);
   
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Mixed-precision hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpy_mp( hypre_long_double alpha,
                        hypre_Vector *x,
                        hypre_Vector *y     )
{
   void               *xp, *yp;

   HYPRE_Int      size   = hypre_VectorSize(x);

   /* Call standard vector axpy if precisions match. */
   if (hypre_VectorPrecision (y) == hypre_VectorPrecision (x))
   {
      return HYPRE_VectorAxpy_pre(hypre_VectorPrecision (y), alpha, (HYPRE_Vector)x, (HYPRE_Vector)y);
   }

   size *= hypre_VectorNumVectors(x);

   /* Implicit conversion to generic data type (void pointer) */
   xp = hypre_VectorData(x);
   yp = hypre_VectorData(y);
   
   /* Call mixed-precision axpy on vector data */
   return hypre_RealArrayAxpyn_mp(hypre_VectorPrecision (x), xp, hypre_VectorPrecision (y), yp,
		        hypre_VectorMemoryLocation(y), size, alpha);
}

/*--------------------------------------------------------------------------
 * Convert precision in a mixed precision vector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorConvert_mp (hypre_Vector *v,
                           HYPRE_Precision new_precision)
{
   HYPRE_Precision data_precision = hypre_VectorPrecision (v);
   void *data = hypre_VectorData(v);
   void *data_mp = NULL;
   HYPRE_Int size = hypre_VectorSize(v) * hypre_VectorNumVectors(v);

   HYPRE_MemoryLocation data_location = hypre_VectorMemoryLocation(v);

   if (new_precision == data_precision)
   {
      return hypre_error_flag;
   }
   else
   {
      /* clone vector data and convert to new precision type */
      data_mp = hypre_RealArrayClone_mp(data_precision, data, data_location, new_precision, data_location, size);

      /* reset data pointer for vector */
      hypre_SeqVectorSetData_pre(new_precision, v, data_mp);
      /* Note:
       * SeqVectorSetData() frees old vector data and resets ownership to 0.
       * We need to set data ownership here to ensure new data memory is cleaned up later.
       */
      hypre_SeqVectorSetDataOwner(v, 1);
      /* Update precision */
      hypre_VectorPrecision(v) = new_precision;
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Convert precision in a mixed precision matrix
 *
 * 1. Save matrix data pointer
 * 2. Set the matrix data pointer to NULL
 * 3. Call ResetData() to allocate new data in new precision
 * 4. Copy data
 * 5. Free pointer to old data  and update precision
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixConvert_mp (hypre_CSRMatrix *A,
                           HYPRE_Precision new_precision)
{
   HYPRE_Precision data_precision = hypre_CSRMatrixPrecision (A);
   void *data, *data_mp;
   HYPRE_Int size = hypre_CSRMatrixI(A)[hypre_CSRMatrixNumRows(A)];
   HYPRE_MemoryLocation data_location = hypre_CSRMatrixMemoryLocation(A);

   if (new_precision == data_precision)
   {
      return hypre_error_flag;
   }
   else
   {
      /* Set pointer to current data */
      data = hypre_CSRMatrixData(A);
      /* Set matrix data pointer to NULL */
      hypre_CSRMatrixData(A) = NULL;
      
      /* reset matrix A's data storage to match new precision */
      hypre_CSRMatrixResetData_pre(new_precision, A);

      /* copy data to newly reset storage */
      data_mp = hypre_CSRMatrixData(A);
      hypre_RealArrayCopy_mp(data_precision, data, data_location,
   			  new_precision, data_mp, data_location, size);

      /* Now free old data */
      hypre_Free(data, data_location);
      /* Update precision */
      hypre_CSRMatrixPrecision(A) = new_precision;
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Mixed precision matrix copy.
 * NOTE: This copies the entire matrix and not just the structure.
 *	 For structure only, use hypre_CSRMatrixCopy(A, B, 0);
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_CSRMatrixCopy_mp( hypre_CSRMatrix *A, hypre_CSRMatrix *B)
{
   HYPRE_Precision precision_A = hypre_CSRMatrixPrecision (A);
   HYPRE_Precision precision_B = hypre_CSRMatrixPrecision (B);
   HYPRE_Int size = hypre_CSRMatrixI(A)[hypre_CSRMatrixNumRows(A)];
   
   /* Implicit conversion to generic data type (void pointer) */
   void *Ap = hypre_CSRMatrixData(A);
   void *Bp = hypre_CSRMatrixData(B);

   /* Call standard vector copy if precisions match. */
   if (precision_A == precision_B)
   {
      hypre_CSRMatrixCopy_pre(precision_A, A, B, 1);
   }
   
   /* Copy structure of A to B.
    * Note: We are only copying structure here so we 
    *       can use the default function call
   */
   hypre_CSRMatrixCopy(A, B, 0);
      
   /* Now copy data from A to B */
   hypre_RealArrayCopy_mp(precision_A, Ap, hypre_CSRMatrixMemoryLocation(A),
   			  precision_B, Bp, hypre_CSRMatrixMemoryLocation(B), size); 

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixClone_mp
 * Clone matrix A to a new_precision matrix at the same memory location.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixClone_mp( hypre_CSRMatrix *A, HYPRE_Precision new_precision )
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);
   HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(A);
   
   hypre_CSRMatrix *B = NULL;

   HYPRE_Int bigInit = hypre_CSRMatrixBigJ(A) != NULL;

   /* Create and initialize new matrix B in new precision */
   B = hypre_CSRMatrixCreate_pre(new_precision, num_rows, num_cols, num_nonzeros);
   hypre_CSRMatrixInitialize_v2_pre(new_precision, B, bigInit, memory_location);   

   /* Call mixed-precision copy */
   hypre_CSRMatrixCopy_mp(A, B);

   return B;
}

#endif
