/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJMatrix_ParCSR interface
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"

#include "../HYPRE.h"

/* The preferred interface for GPU */
HYPRE_Int
hypre_IJMatrixSetAddValuesParCSRDevice0( hypre_IJMatrix       *matrix,
                                         HYPRE_Int             nelms,
                                         const HYPRE_BigInt   *rows,
                                         const HYPRE_BigInt   *cols,
                                         const HYPRE_Complex  *values,
                                         const char           *action)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_BigInt row_start = row_partitioning[0];
   HYPRE_BigInt row_end   = row_partitioning[1];
   HYPRE_BigInt col_start = col_partitioning[0];
   HYPRE_BigInt col_end   = col_partitioning[1];
#else
   HYPRE_BigInt row_start = row_partitioning[my_id];
   HYPRE_BigInt row_end   = row_partitioning[my_id+1];
   HYPRE_BigInt col_start = col_partitioning[my_id];
   HYPRE_BigInt col_end   = col_partitioning[my_id+1];
#endif
   HYPRE_Int nrows = row_end - row_start;
   HYPRE_Int ncols = col_end - col_start;

   in_range<HYPRE_BigInt> pred(row_start, row_end-1);
   HYPRE_Int nelms_on = HYPRE_THRUST_CALL(count_if, rows, rows+nelms, pred);
   HYPRE_Int nelms_off = nelms - nelms_on;
   const char SorA = action[0] == 's' ? 1 : 0;
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);

   if (!aux_matrix)
   {
      hypre_AuxParCSRMatrixCreate(&aux_matrix, nrows, ncols, NULL);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }

   hypre_AuxParCSRMatrixMemoryLocation(aux_matrix) = HYPRE_MEMORY_DEVICE;

   /* on proc */
   if (nelms_on)
   {
      if ( hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix) < hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) + nelms_on )
      {
         HYPRE_Int size, size0;
         size0 = hypre_AuxParCSRMatrixUsrOnProcSize(aux_matrix) >= 0 ? hypre_AuxParCSRMatrixUsrOnProcSize(aux_matrix) :
                                                                       hypre_AuxParCSRMatrixInitAllocFactor(aux_matrix) * nrows;
         size = hypre_max( hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) + nelms_on, size0);
         size = hypre_max( hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix) * hypre_AuxParCSRMatrixGrowFactor(aux_matrix), size );

         hypre_AuxParCSRMatrixOnProcI(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOnProcI(aux_matrix),    HYPRE_BigInt,  hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix),
                            HYPRE_BigInt,  size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOnProcJ(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOnProcJ(aux_matrix),    HYPRE_BigInt,  hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix),
                            HYPRE_BigInt,  size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOnProcData(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOnProcData(aux_matrix), HYPRE_Complex, hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix),
                            HYPRE_Complex, size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOnProcSorA(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOnProcSorA(aux_matrix), char,          hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix),
                            char,          size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix) = size;
      }

      auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_zip_iterator(thrust::make_tuple(rows,         cols,         values        )),  /* first */
            thrust::make_zip_iterator(thrust::make_tuple(rows + nelms, cols + nelms, values + nelms)),  /* last */
            rows,                                                                                       /* stencil */
            thrust::make_zip_iterator(thrust::make_tuple(
                                      hypre_AuxParCSRMatrixOnProcI(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOnProcJ(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOnProcData(aux_matrix) + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix))),
                                                                                                        /* result */
            pred);

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) - hypre_AuxParCSRMatrixOnProcI(aux_matrix) ==
                    hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) + nelms_on);

      HYPRE_THRUST_CALL(fill_n, hypre_AuxParCSRMatrixOnProcSorA(aux_matrix) + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                        nelms_on, SorA);

      hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) += nelms_on;
   }

   /* off proc */
   if (nelms_off)
   {
      if ( hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) < hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) + nelms_off )
      {
         HYPRE_Int size, size0;
         size0 = hypre_AuxParCSRMatrixUsrOffProcSize(aux_matrix) >= 0 ? hypre_AuxParCSRMatrixUsrOffProcSize(aux_matrix) :
                                                                        hypre_AuxParCSRMatrixInitAllocFactor(aux_matrix) * nrows;
         size = hypre_max( hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) + nelms_off, size0 );
         size = hypre_max( hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) * hypre_AuxParCSRMatrixGrowFactor(aux_matrix), size );

         hypre_AuxParCSRMatrixOffProcI(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOffProcI(aux_matrix),    HYPRE_BigInt,  hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix),
                            HYPRE_BigInt,  size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOffProcJ(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOffProcJ(aux_matrix),    HYPRE_BigInt,  hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix),
                            HYPRE_BigInt,  size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOffProcData(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOffProcData(aux_matrix), HYPRE_Complex, hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix),
                            HYPRE_Complex, size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOffProcSorA(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOffProcSorA(aux_matrix), char,          hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix),
                            char,          size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = size;
      }

      auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_zip_iterator(thrust::make_tuple(rows,         cols,         values        )),  /* first */
            thrust::make_zip_iterator(thrust::make_tuple(rows + nelms, cols + nelms, values + nelms)),  /* last */
            rows,                                                                                       /* stencil */
            thrust::make_zip_iterator(thrust::make_tuple(
                                      hypre_AuxParCSRMatrixOffProcI(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOffProcJ(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOffProcData(aux_matrix) + hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix))),
                                                                                                        /* result */
            thrust::not1(pred));

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) - hypre_AuxParCSRMatrixOffProcI(aux_matrix) ==
                    hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) + nelms_off);

      HYPRE_THRUST_CALL(fill_n, hypre_AuxParCSRMatrixOffProcSorA(aux_matrix) + hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix),
                        nelms_off, SorA);

      hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) += nelms_off;
   }

   return hypre_error_flag;
}

__global__ void
hypreCUDAKernel_IJMatrixValues_dev1(HYPRE_Int n, HYPRE_Int *rowind, HYPRE_Int *row_ptr, HYPRE_Int *row_len, HYPRE_Int *mark)
{
   HYPRE_Int global_thread_id = hypre_cuda_get_grid_thread_id<1,1>();

   if (global_thread_id < n)
   {
      HYPRE_Int row = rowind[global_thread_id];
      if (global_thread_id < read_only_load(&row_ptr[row]) + read_only_load(&row_len[row]))
      {
         mark[global_thread_id] = 0;
      }
      else
      {
         mark[global_thread_id] = -1;
      }
   }
}

/* E.g. nrows = 3
 *      ncols = 2 3 4
 *      rows  = 10 20 30
 *      rows_indexes = 0 4 9
 *              (0 1 2 3 | 4 5 6 7 8 | 9 10 11 12 13)
 *      cols   = x x ! ! | * * * ! ! | +  +  +  +  !
 *      values = . . ! !   . . . ! !   .  .  .  .  !
 */
HYPRE_Int
hypre_IJMatrixSetAddValuesParCSRDevice( hypre_IJMatrix       *matrix,
                                        HYPRE_Int             nrows,
                                        HYPRE_Int            *ncols,        /* if NULL, == all ones */
                                        const HYPRE_BigInt   *rows,
                                        const HYPRE_Int      *row_indexes,  /* if NULL, == ex_scan of ncols, i.e, no gap */
                                        const HYPRE_BigInt   *cols,
                                        const HYPRE_Complex  *values,
                                        const char           *action        /* set or add */)
{
   HYPRE_Int nnz;
   HYPRE_BigInt *rows2, *cols2;
   HYPRE_Complex *values2;

   /* Caveat: the memory that all the pointers referring to must be compatible with the memory location of matrix */
   HYPRE_Int memory_location = hypre_IJMatrixMemoryLocation(matrix);

   /* expand rows into full expansion of rows based on ncols
    * if ncols == NULL, ncols is all ones, so rows are indeed full expansion */
   if (ncols)
   {
      HYPRE_Int *row_ptr = hypre_TAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(row_ptr, ncols, HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE, memory_location);
      hypreDevice_IntegerExclusiveScan(nrows + 1, row_ptr);
      hypre_TMemcpy(&nnz, row_ptr+nrows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, memory_location);

      rows2 = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE);
      hypreDevice_CsrRowPtrsToIndicesWithRowNum(nrows, nnz, row_ptr, (HYPRE_BigInt *) rows, rows2);
      hypre_TFree(row_ptr, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      rows2 = (HYPRE_BigInt *) rows;
      nnz = hypreDevice_IntegerReduceSum(nrows, ncols);
   }

   if (row_indexes)
   {
      cols2 = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE);
      values2 = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);

      HYPRE_Int len, len1;
      hypre_TMemcpy(&len1, &row_indexes[nrows-1], HYPRE_Int, 1, HYPRE_MEMORY_HOST, memory_location);
      hypre_TMemcpy(&len,  &ncols[nrows-1],       HYPRE_Int, 1, HYPRE_MEMORY_HOST, memory_location);
      /* this is the *effective* length of cols and values */
      len += len1;
      HYPRE_Int *indicator = hypre_CTAlloc(HYPRE_Int, len, HYPRE_MEMORY_DEVICE);
      hypreDevice_CsrRowPtrsToIndices_v2(nrows-1, len1, (HYPRE_Int *) row_indexes, indicator);
      /* wanted elements are marked as -1 */
      dim3 bDim = hypre_GetDefaultCUDABlockDimension();
      dim3 gDim = hypre_GetDefaultCUDAGridDimension(len1, "thread", bDim);
      HYPRE_CUDA_LAUNCH( hypreCUDAKernel_IJMatrixValues_dev1, gDim, bDim, len1, indicator, (HYPRE_Int *) row_indexes, ncols, indicator );

      auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_zip_iterator(thrust::make_tuple(cols,       values)),
            thrust::make_zip_iterator(thrust::make_tuple(cols + len, values + len)),
            indicator,
            thrust::make_zip_iterator(thrust::make_tuple(cols2,      values2)),
            is_nonnegative<HYPRE_Int>() );

      HYPRE_Int nnz_tmp = thrust::get<0>(new_end.get_iterator_tuple()) - cols2;

      hypre_assert(nnz_tmp == nnz);

      hypre_TFree(indicator, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      cols2 = (HYPRE_BigInt *) cols;
      values2 = (HYPRE_Complex *) values;
   }

   hypre_IJMatrixSetAddValuesParCSRDevice0(matrix, nnz, rows2, cols2, values2, action);

   if (rows2 != rows)
   {
      hypre_TFree(rows2, HYPRE_MEMORY_DEVICE);
   }

   if (cols2 != cols)
   {
      hypre_TFree(cols2, HYPRE_MEMORY_DEVICE);
   }

   if (values2 != values)
   {
      hypre_TFree(values2, HYPRE_MEMORY_DEVICE);
   }

   return hypre_error_flag;
}

/*
__global__ void
hypreCUDAKernel_IJMatrixAssembleSortAndReduce1(HYPRE_Int n, HYPRE_BigInt *I, HYPRE_BigInt *J, char *X, HYPRE_Complex *A)
{
   HYPRE_Int t = hypre_cuda_get_grid_thread_id<1,1>();
   if (t >= n)
   {
      return;
   }
   HYPRE_BigInt i = read_only_load(I+t);
   HYPRE_BigInt j = read_only_load(J+t);
   for (HYPRE_Int k=t+1; k < n && read_only_load(I+k) == i && read_only_load(J+k) == j; k++)
   {
      if (read_only_load(X+k) == 1)
      {
         A[t] = 0.0;
         break;
      }
   }
}
*/

template<typename T1, typename T2>
struct hypre_IJMatrixAssembleFunctor : public thrust::binary_function< thrust::tuple<T1, T2>, thrust::tuple<T1, T2>, thrust::tuple<T1, T2> >
{
   typedef thrust::tuple<T1, T2> Tuple;

   __device__ Tuple operator()(const Tuple& x, const Tuple& y )
   {
      return thrust::make_tuple( hypre_max(thrust::get<0>(x), thrust::get<0>(y)), thrust::get<1>(x) + thrust::get<1>(y) );
   }
};

/* helper routine used in hypre_IJMatrixAssembleParCSRDevice:
 * 1. sort (X0, A0) with key (I0, J0)
 *    [put the diagonal first; see the comments in hypre_cuda_utils.c]
 * 2. for each segment in (I0, J0), zero out in A0 all before the last `set'
 * 3. reduce A0 [with sum] and reduce X0 [with max]
 * N0: input size; N1: size after reduction (<= N0)
 * Note: (I1, J1, X1, A1) are not resized to N1 but have size N0
 */
HYPRE_Int
hypre_IJMatrixAssembleSortAndReduce1(HYPRE_Int  N0, HYPRE_BigInt  *I0, HYPRE_BigInt  *J0, char  *X0, HYPRE_Complex  *A0,
                                     HYPRE_Int *N1, HYPRE_BigInt **I1, HYPRE_BigInt **J1, char **X1, HYPRE_Complex **A1 )
{
   hypreDevice_StableSortTupleByTupleKey(N0, I0, J0, X0, A0, 2);

   HYPRE_BigInt  *I = hypre_TAlloc(HYPRE_BigInt,  N0, HYPRE_MEMORY_DEVICE);
   HYPRE_BigInt  *J = hypre_TAlloc(HYPRE_BigInt,  N0, HYPRE_MEMORY_DEVICE);
   char          *X = hypre_TAlloc(char,          N0, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *A = hypre_TAlloc(HYPRE_Complex, N0, HYPRE_MEMORY_DEVICE);

   /*
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(N0, "thread", bDim);
   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_IJMatrixAssembleSortAndReduce1, gDim, bDim, N0, I0, J0, X0, A0 );
   */

   /* output X: 1: keep, 0: zero-out */
   HYPRE_THRUST_CALL(
         exclusive_scan_by_key,
         make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(I0+N0, J0+N0))),
         make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(I0,    J0))),
         make_reverse_iterator(thrust::device_pointer_cast<char>(X0)+N0),
         make_reverse_iterator(thrust::device_pointer_cast<char>(X) +N0),
         0,
         thrust::equal_to< thrust::tuple<HYPRE_BigInt, HYPRE_BigInt> >(),
         thrust::maximum<char>() );

   HYPRE_THRUST_CALL(replace_if, A0, A0 + N0, X, thrust::identity<char>(), 0.0);

   auto new_end = HYPRE_THRUST_CALL(
         reduce_by_key,
         thrust::make_zip_iterator(thrust::make_tuple(I0,      J0     )), /* keys_first */
         thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0)), /* keys_last */
         thrust::make_zip_iterator(thrust::make_tuple(X0,      A0     )), /* values_first */
         thrust::make_zip_iterator(thrust::make_tuple(I,       J      )), /* keys_output */
         thrust::make_zip_iterator(thrust::make_tuple(X,       A      )), /* values_output */
         thrust::equal_to< thrust::tuple<HYPRE_BigInt, HYPRE_BigInt> >(), /* binary_pred */
         hypre_IJMatrixAssembleFunctor<char, HYPRE_Complex>()             /* binary_op */);

   *N1 = thrust::get<0>(new_end.first.get_iterator_tuple()) - I;
   *I1 = I;
   *J1 = J;
   *X1 = X;
   *A1 = A;

   return hypre_error_flag;
}

template<typename T1, typename T2>
struct hypre_IJMatrixAssembleFunctor2 : public thrust::binary_function< thrust::tuple<T1, T2>, thrust::tuple<T1, T2>, thrust::tuple<T1, T2> >
{
   typedef thrust::tuple<T1, T2> Tuple;

   __device__ Tuple operator()(const Tuple& x, const Tuple& y)
   {
      const char          tx = thrust::get<0>(x);
      const char          ty = thrust::get<0>(y);
      const HYPRE_Complex vx = thrust::get<1>(x);
      const HYPRE_Complex vy = thrust::get<1>(y);
      const HYPRE_Complex vz = tx == 0 && ty == 0 ? vx + vy : tx ? vx : vy;
      return thrust::make_tuple(0, vz);
   }
};

HYPRE_Int
hypre_IJMatrixAssembleSortAndReduce2(HYPRE_Int  N0, HYPRE_Int  *I0, HYPRE_Int  *J0, char  *X0, HYPRE_Complex  *A0,
                                     HYPRE_Int *N1, HYPRE_Int **I1, HYPRE_Int **J1,            HYPRE_Complex **A1,
                                     HYPRE_Int  opt )
{
   hypreDevice_StableSortTupleByTupleKey(N0, I0, J0, X0, A0, opt);

   HYPRE_Int     *I = hypre_TAlloc(HYPRE_Int,     N0, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *J = hypre_TAlloc(HYPRE_Int,     N0, HYPRE_MEMORY_DEVICE);
   char          *X = hypre_TAlloc(char,          N0, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *A = hypre_TAlloc(HYPRE_Complex, N0, HYPRE_MEMORY_DEVICE);

   auto new_end = HYPRE_THRUST_CALL(
         reduce_by_key,
         thrust::make_zip_iterator(thrust::make_tuple(I0,      J0     )), /* keys_first */
         thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0)), /* keys_last */
         thrust::make_zip_iterator(thrust::make_tuple(X0,      A0     )), /* values_first */
         thrust::make_zip_iterator(thrust::make_tuple(I,       J      )), /* keys_output */
         thrust::make_zip_iterator(thrust::make_tuple(X,       A      )), /* values_output */
         thrust::equal_to< thrust::tuple<HYPRE_Int, HYPRE_Int> >(),       /* binary_pred */
         hypre_IJMatrixAssembleFunctor2<char, HYPRE_Complex>()            /* binary_op */);

   *N1 = thrust::get<0>(new_end.first.get_iterator_tuple()) - I;
   *I1 = I;
   *J1 = J;
   *A1 = A;

   hypre_TFree(X, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

HYPRE_Int
hypre_IJMatrixAssembleParCSRDevice(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_BigInt row_start = row_partitioning[0];
   HYPRE_BigInt row_end   = row_partitioning[1];
   HYPRE_BigInt col_start = col_partitioning[0];
   HYPRE_BigInt col_end   = col_partitioning[1];
#else
   HYPRE_BigInt row_start = row_partitioning[my_id];
   HYPRE_BigInt row_end   = row_partitioning[my_id+1];
   HYPRE_BigInt col_start = col_partitioning[my_id];
   HYPRE_BigInt col_end   = col_partitioning[my_id+1];
#endif
   HYPRE_Int nrows = row_end - row_start;
   HYPRE_Int ncols = col_end - col_start;

   hypre_ParCSRMatrix    *par_matrix = (hypre_ParCSRMatrix*)    hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix*) hypre_IJMatrixTranslator(matrix);

   HYPRE_Int      nelms_on     = hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix);
   HYPRE_BigInt  *on_proc_i    = hypre_AuxParCSRMatrixOnProcI(aux_matrix);
   HYPRE_BigInt  *on_proc_j    = hypre_AuxParCSRMatrixOnProcJ(aux_matrix);
   HYPRE_Complex *on_proc_data = hypre_AuxParCSRMatrixOnProcData(aux_matrix);
   char          *on_proc_SorA = hypre_AuxParCSRMatrixOnProcSorA(aux_matrix);

   HYPRE_Int memory_location = hypre_GetActualMemLocation(hypre_IJMatrixMemoryLocation(matrix));

   if (!aux_matrix)
   {
      return hypre_error_flag;
   }

   if (!par_matrix)
   {
      return hypre_error_flag;
   }

   /* communicate for aux off-proc and add to aux on-proc */
   if ( hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) )
   {
   }

   /* compress */
   if (nelms_on)
   {
      HYPRE_Int      new_nnz;
      HYPRE_BigInt  *new_i;
      HYPRE_BigInt  *new_j;
      HYPRE_Complex *new_data;
      char          *new_SorA;

      /* sort and reduce */
      hypre_IJMatrixAssembleSortAndReduce1(nelms_on, on_proc_i, on_proc_j, on_proc_SorA, on_proc_data,
                                           &new_nnz, &new_i, &new_j, &new_SorA, &new_data);

      /* adjust row indices from global to local */
      HYPRE_Int *new_i_local = hypre_TAlloc(HYPRE_Int, new_nnz, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL(transform, new_i, new_i + new_nnz, new_i_local, _1 - row_start);

      HYPRE_Int      num_cols_offd_new;
      HYPRE_BigInt  *col_map_offd_new;
      HYPRE_Int     *col_map_offd_map;
      HYPRE_Int      diag_nnz_new;
      HYPRE_Int     *diag_i_new;
      HYPRE_Int     *diag_j_new;
      HYPRE_Complex *diag_a_new;
      HYPRE_Int      offd_nnz_new;
      HYPRE_Int     *offd_i_new;
      HYPRE_Int     *offd_j_new;
      HYPRE_Complex *offd_a_new;
      char          *diag_SorA_new = NULL;
      char          *offd_SorA_new = NULL;

      HYPRE_Int diag_nnz_existed = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(par_matrix));
      HYPRE_Int offd_nnz_existed = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(par_matrix));

      /* split IJ into diag and offd */
      hypre_CSRMatrixSplitDevice_core( nrows, new_nnz,
                                       new_i_local, new_j, new_data,
                                       diag_nnz_existed || offd_nnz_existed ? new_SorA : NULL,
                                       col_start, col_end-1,
                                       hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(par_matrix)),
                                       hypre_ParCSRMatrixDeviceColMapOffd(par_matrix),
                                       &col_map_offd_map,
                                       &num_cols_offd_new, &col_map_offd_new,
                                       &diag_nnz_new, &diag_i_new, &diag_j_new, &diag_a_new,
                                       diag_nnz_existed ? &diag_SorA_new : NULL,
                                       &offd_nnz_new, &offd_i_new, &offd_j_new, &offd_a_new,
                                       offd_nnz_existed ? &offd_SorA_new : NULL );

      hypre_TFree(new_i,       HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_j,       HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_data,    HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_SorA,    HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_i_local, HYPRE_MEMORY_DEVICE);

      /* expand the existing diag/offd and compress with the new one */
      if (diag_nnz_existed)
      {
         HYPRE_Int     *tmp_i = hypre_TAlloc(HYPRE_Int,     diag_nnz_existed + diag_nnz_new, HYPRE_MEMORY_DEVICE);
         HYPRE_Int     *tmp_j = hypre_TAlloc(HYPRE_Int,     diag_nnz_existed + diag_nnz_new, HYPRE_MEMORY_DEVICE);
         char          *tmp_x = hypre_TAlloc(char,          diag_nnz_existed + diag_nnz_new, HYPRE_MEMORY_DEVICE);
         HYPRE_Complex *tmp_a = hypre_TAlloc(HYPRE_Complex, diag_nnz_existed + diag_nnz_new, HYPRE_MEMORY_DEVICE);

         hypreDevice_CsrRowPtrsToIndices_v2(nrows, diag_nnz_existed,
                                            hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix)), tmp_i);
         hypre_TMemcpy(tmp_j, hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(par_matrix)), HYPRE_Int,
                       diag_nnz_existed, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         HYPRE_THRUST_CALL(fill_n, tmp_x, diag_nnz_existed, 0);
         hypre_TMemcpy(tmp_a, hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(par_matrix)), HYPRE_Complex,
                       diag_nnz_existed, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         hypre_TMemcpy(tmp_i + diag_nnz_existed, diag_i_new, HYPRE_Int, diag_nnz_new,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(tmp_j + diag_nnz_existed, diag_j_new, HYPRE_Int, diag_nnz_new,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(tmp_x + diag_nnz_existed, diag_SorA_new, char, diag_nnz_new,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(tmp_a + diag_nnz_existed, diag_a_new, HYPRE_Complex, diag_nnz_new,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         hypre_TFree(diag_i_new,    HYPRE_MEMORY_DEVICE);
         hypre_TFree(diag_j_new,    HYPRE_MEMORY_DEVICE);
         hypre_TFree(diag_SorA_new, HYPRE_MEMORY_DEVICE);
         hypre_TFree(diag_a_new,    HYPRE_MEMORY_DEVICE);

         HYPRE_Int nnz_new;
         hypre_IJMatrixAssembleSortAndReduce2(diag_nnz_existed + diag_nnz_new, tmp_i, tmp_j, tmp_x, tmp_a,
                                              &nnz_new, &diag_i_new, &diag_j_new, &diag_a_new, 2);

         hypre_TFree(tmp_i, HYPRE_MEMORY_DEVICE);
         hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);
         hypre_TFree(tmp_x, HYPRE_MEMORY_DEVICE);
         hypre_TFree(tmp_a, HYPRE_MEMORY_DEVICE);

         diag_j_new = hypre_TReAlloc_v2(diag_j_new, HYPRE_Int,     diag_nnz_existed + diag_nnz_new, HYPRE_Int,
                                        nnz_new, HYPRE_MEMORY_DEVICE);
         diag_a_new = hypre_TReAlloc_v2(diag_a_new, HYPRE_Complex, diag_nnz_existed + diag_nnz_new, HYPRE_Complex,
                                        nnz_new, HYPRE_MEMORY_DEVICE);
         diag_nnz_new = nnz_new;
      }

      if (offd_nnz_existed)
      {
         HYPRE_Int     *tmp_i = hypre_TAlloc(HYPRE_Int,     offd_nnz_existed + offd_nnz_new, HYPRE_MEMORY_DEVICE);
         HYPRE_Int     *tmp_j = hypre_TAlloc(HYPRE_Int,     offd_nnz_existed + offd_nnz_new, HYPRE_MEMORY_DEVICE);
         char          *tmp_x = hypre_TAlloc(char,          offd_nnz_existed + offd_nnz_new, HYPRE_MEMORY_DEVICE);
         HYPRE_Complex *tmp_a = hypre_TAlloc(HYPRE_Complex, offd_nnz_existed + offd_nnz_new, HYPRE_MEMORY_DEVICE);

         hypreDevice_CsrRowPtrsToIndices_v2(nrows, offd_nnz_existed,
                                            hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix)), tmp_i);
         HYPRE_THRUST_CALL(gather,
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix)) + offd_nnz_existed,
                           col_map_offd_map, tmp_j);
         HYPRE_THRUST_CALL(fill_n, tmp_x, offd_nnz_existed, 0);
         hypre_TMemcpy(tmp_a, hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(par_matrix)), HYPRE_Complex,
                       offd_nnz_existed, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         hypre_TMemcpy(tmp_i + offd_nnz_existed, offd_i_new, HYPRE_Int, offd_nnz_new,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(tmp_j + offd_nnz_existed, offd_j_new, HYPRE_Int, offd_nnz_new,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(tmp_x + offd_nnz_existed, offd_SorA_new, char, offd_nnz_new,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(tmp_a + offd_nnz_existed, offd_a_new, HYPRE_Complex, offd_nnz_new,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         hypre_TFree(offd_i_new,    HYPRE_MEMORY_DEVICE);
         hypre_TFree(offd_j_new,    HYPRE_MEMORY_DEVICE);
         hypre_TFree(offd_SorA_new, HYPRE_MEMORY_DEVICE);
         hypre_TFree(offd_a_new,    HYPRE_MEMORY_DEVICE);

         HYPRE_Int nnz_new;
         hypre_IJMatrixAssembleSortAndReduce2(offd_nnz_existed + offd_nnz_new, tmp_i, tmp_j, tmp_x, tmp_a,
                                              &nnz_new, &offd_i_new, &offd_j_new, &offd_a_new, 0);

         hypre_TFree(tmp_i, HYPRE_MEMORY_DEVICE);
         hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);
         hypre_TFree(tmp_x, HYPRE_MEMORY_DEVICE);
         hypre_TFree(tmp_a, HYPRE_MEMORY_DEVICE);

         offd_j_new = hypre_TReAlloc_v2(offd_j_new, HYPRE_Int,     offd_nnz_existed + offd_nnz_new, HYPRE_Int,
                                        nnz_new, HYPRE_MEMORY_DEVICE);
         offd_a_new = hypre_TReAlloc_v2(offd_a_new, HYPRE_Complex, offd_nnz_existed + offd_nnz_new, HYPRE_Complex,
                                        nnz_new, HYPRE_MEMORY_DEVICE);
         offd_nnz_new = nnz_new;
      }

      hypre_TFree(col_map_offd_map, HYPRE_MEMORY_DEVICE);

      /* convert to CSR's */
      hypre_CSRMatrix *diag               = hypre_CSRMatrixCreate(nrows, ncols, diag_nnz_new);
      hypre_CSRMatrixI(diag)              = hypreDevice_CsrRowIndicesToPtrs(nrows, diag_nnz_new, diag_i_new);
      hypre_CSRMatrixJ(diag)              = diag_j_new;
      hypre_CSRMatrixData(diag)           = diag_a_new;
      hypre_CSRMatrixMemoryLocation(diag) = HYPRE_MEMORY_DEVICE;

      hypre_TFree(diag_i_new, HYPRE_MEMORY_DEVICE);

      hypre_CSRMatrix *offd               = hypre_CSRMatrixCreate(nrows, num_cols_offd_new, offd_nnz_new);
      hypre_CSRMatrixI(offd)              = hypreDevice_CsrRowIndicesToPtrs(nrows, offd_nnz_new, offd_i_new);
      hypre_CSRMatrixJ(offd)              = offd_j_new;
      hypre_CSRMatrixData(offd)           = offd_a_new;
      hypre_CSRMatrixMemoryLocation(offd) = HYPRE_MEMORY_DEVICE;

      hypre_TFree(offd_i_new, HYPRE_MEMORY_DEVICE);

      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(par_matrix));
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(par_matrix));

      /* [copy to UM] and save in par_csr matrix */
      if (memory_location != HYPRE_MEMORY_DEVICE)
      {
         hypre_CSRMatrix *new_diag = hypre_CSRMatrixClone(diag, 1);
         hypre_CSRMatrixDestroy(diag);
         hypre_ParCSRMatrixDiag(par_matrix) = new_diag;
         hypre_CSRMatrix *new_offd = hypre_CSRMatrixClone(offd, 1);
         hypre_CSRMatrixDestroy(offd);
         hypre_ParCSRMatrixOffd(par_matrix) = new_offd;
      }
      else
      {
         hypre_ParCSRMatrixDiag(par_matrix) = diag;
         hypre_ParCSRMatrixOffd(par_matrix) = offd;
      }

      hypre_TFree(hypre_ParCSRMatrixDeviceColMapOffd(par_matrix), HYPRE_MEMORY_DEVICE);
      hypre_ParCSRMatrixDeviceColMapOffd(par_matrix) = col_map_offd_new;

      hypre_TFree(hypre_ParCSRMatrixColMapOffd(par_matrix), HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixColMapOffd(par_matrix) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_new, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(par_matrix), col_map_offd_new, HYPRE_BigInt, num_cols_offd_new,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   hypre_IJMatrixAssembleFlag(matrix) = 1;
   hypre_AuxParCSRMatrixDestroy(aux_matrix);
   hypre_IJMatrixTranslator(matrix) = NULL;

   return hypre_error_flag;
}


/*
HYPRE_Int nnz = hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix);
printf("nnz %d\n", nnz);
char          *SorA = hypre_TAlloc(char,          nnz, HYPRE_MEMORY_HOST);
HYPRE_Int     *tmpi = hypre_TAlloc(HYPRE_Int,     nnz, HYPRE_MEMORY_HOST);
HYPRE_Int     *tmpj = hypre_TAlloc(HYPRE_Int,     nnz, HYPRE_MEMORY_HOST);
HYPRE_Complex *tmpd = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_HOST);
hypre_TMemcpy(tmpi, hypre_AuxParCSRMatrixOnProcI(aux_matrix),    HYPRE_Int,     nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
hypre_TMemcpy(tmpj, hypre_AuxParCSRMatrixOnProcJ(aux_matrix),    HYPRE_Int,     nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
hypre_TMemcpy(tmpd, hypre_AuxParCSRMatrixOnProcData(aux_matrix), HYPRE_Complex, nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
hypre_TMemcpy(SorA, hypre_AuxParCSRMatrixOnProcSorA(aux_matrix), char,          nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
for (int i=0; i<nnz; i++)
{
   printf("%d ", SorA[i]);
   printf("%d ", tmpi[i]);
   printf("%d ", tmpj[i]);
   printf("%e \n", tmpd[i]);
}
HYPRE_CUDA_CALL( cudaGetLastError() );
exit(0);
*/
