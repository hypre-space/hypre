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
hypre_SeqVectorAxpy_mp( hypre_double alpha,
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
      data_mp = hypre_RealArrayClone_mp(data_precision, data, data_location, new_precision, data_location,
                                        size);

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
 *  For structure only, use hypre_CSRMatrixCopy(A, B, 0);
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
      return hypre_CSRMatrixCopy_pre(precision_A, A, B, 1);
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

/*--------------------------------------------------------------------------*
* hypre_RealArrayCopyHost_mp: copy n array contents from x to y.
* Assumes arrays x and y are both on host memory.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayCopyHost_mp(HYPRE_Precision precision_x, void *x,
                           HYPRE_Precision precision_y, void *y, HYPRE_Int n)
{
   HYPRE_Int      i;

   /* Mixed-precision copy of data */
   switch (precision_x)
   {
      case HYPRE_REAL_SINGLE:
         switch (precision_y)
         {
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_double *)y)[i] = (hypre_double)((hypre_float *)x)[i];
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_long_double *)y)[i] = (hypre_long_double)((hypre_float *)x)[i];
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_DOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_float *)y)[i] = (hypre_float)((hypre_double *)x)[i];
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_long_double *)y)[i] = (hypre_long_double)((hypre_double *)x)[i];
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_LONGDOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_float *)y)[i] = (hypre_float)((hypre_long_double *)x)[i];
               }
               break;
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_double *)y)[i] = (hypre_double)((hypre_long_double *)x)[i];
               }
               break;
            default:
               break;
         }
         break;
      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: Undefined precision type for array Copy!\n");
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------*
* hypre_RealArrayCopy_mp: copy n array contents from x to y.
* Arrays x and y need not have the same memory location.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayCopy_mp(HYPRE_Precision precision_x, void *x, HYPRE_MemoryLocation location_x,
                       HYPRE_Precision precision_y, void *y, HYPRE_MemoryLocation location_y, HYPRE_Int n)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   hypre_GpuProfilingPushRange("RealArrayCopy");

   HYPRE_Int     nbytes;
   size_t     sizeof_x;

   /* tmp pointer for data copy */
   void               *xp = NULL;

   /* get sizeof x data */
   sizeof_x = hypre_GetSizeOfReal_pre(precision_x);

   nbytes = n * sizeof_x;
   /* Call standard memory copy if precisions match. */
   if (precision_x == precision_y)
   {
      hypre_Memcpy(y, x, nbytes, location_y, location_x);

      return hypre_error_flag;
   }

   /* Check memory location */
   if (location_x != location_y)
   {
      /* Allocate memory and copy x to y's memory location */
      xp = hypre_CAlloc(n, sizeof_x, location_y);
      hypre_Memcpy(xp, x, nbytes, location_y, location_x);
   }
   else
   {
      xp = x;
   }

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(location_y);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_RealArrayCopyDevice_mp(precision_x, xp, precision_y, y, n);
   }
   else
#endif
   {
      hypre_RealArrayCopyHost_mp(precision_x, xp, precision_y, y, n);
   }
   /* free xp if allocated */
   if (location_x != location_y)
   {
      hypre_TFree(xp, location_y);
   }
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------*
* hypre_RealArrayClone_mp: Clone array x.
*--------------------------------------------------------------------------*/
void *
hypre_RealArrayClone_mp(HYPRE_Precision precision_x, void *x, HYPRE_MemoryLocation location_x,
                        HYPRE_Precision new_precision, HYPRE_MemoryLocation new_location, HYPRE_Int n)
{
   /* cloned data */
   void           *y = NULL;
   size_t     sizeof_y;

   /* get sizeof new_precision data */
   sizeof_y = hypre_GetSizeOfReal_pre(new_precision);
   /* Allocate memory for cloned data */
   y = hypre_CAlloc(n, sizeof_y, new_location);

   /* Copy from x to y */
   hypre_RealArrayCopy_mp(precision_x, x, location_x,
                          new_precision, y, new_location, n);
   return y;
}

/*--------------------------------------------------------------------------*
* hypre_RealArrayAxpynHost_mp: Axpy on n array contents into y.
* Assumes arrays x and y are both on host memory.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayAxpynHost_mp(HYPRE_Precision precision_x, hypre_long_double alpha, void *x,
                            HYPRE_Precision precision_y, void *y, HYPRE_Int n)
{
   HYPRE_Int      i;

   /* Mixed-precision copy of data */
   switch (precision_x)
   {
      case HYPRE_REAL_SINGLE:
         switch (precision_y)
         {
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_double *)y)[i] += (hypre_double)((hypre_float)alpha * ((hypre_float *)x)[i]);
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_long_double *)y)[i] += (hypre_long_double)((hypre_float)alpha * ((hypre_float *)x)[i]);
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_DOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_float *)y)[i] += (hypre_float)((hypre_double)alpha * ((hypre_double *)x)[i]);
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_long_double *)y)[i] += (hypre_long_double)((hypre_double)alpha * ((hypre_double *)x)[i]);
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_LONGDOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_float *)y)[i] += (hypre_float)((hypre_long_double)alpha * ((hypre_long_double *)x)[i]);
               }
               break;
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
               {
                  ((hypre_double *)y)[i] += (hypre_double)((hypre_long_double)alpha * ((hypre_long_double *)x)[i]);
               }
               break;
            default:
               break;
         }
         break;
      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: Undefined precision type for array Axpyn!\n");
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------*
* hypre_RealArrayAxpyn_mp: Axpy on n array contents into y.
* Assumes arrays x and y have the same memory location.
*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_RealArrayAxpyn_mp(HYPRE_Precision precision_x, void *x, HYPRE_Precision precision_y, void *y,
                        HYPRE_MemoryLocation location, HYPRE_Int n, hypre_long_double alpha)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   hypre_GpuProfilingPushRange("RealArrayAxpyn");

   /* Call standard memory copy if precisions match. */
   if (precision_x == precision_y)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: Not Implemented!\n");
      /*
      #if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
            HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(location);

            hypre_long_double d_alpha = (hypre_long_double)(*alpha);

            if (exec == HYPRE_EXEC_DEVICE)
            {
               hypreDevice_ComplexDeviceArrayAxpyn_pre(precision_y, d_alpha, x, y, n);
            }
            else
      #endif
            {
               HYPRE_Int inc = 1;
               hypre_daxpy_pre(precision_y, n, alpha, x, inc, y, inc);

            }
            return hypre_error_flag;
      */
   }

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(location);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_RealArrayAxpynDevice_mp(precision_x, alpha, x, precision_y, y, n);
   }
   else
#endif
   {
      hypre_RealArrayAxpynHost_mp(precision_x, alpha, x, precision_y, y, n);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif
