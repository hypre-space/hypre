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

__global__ void
hypreCUDAKernel_IJMatrixValues_dev1(HYPRE_Int n, HYPRE_Int *rowind, HYPRE_Int *row_ptr, HYPRE_Int *row_len, HYPRE_Int *mark);

typedef thrust::tuple<HYPRE_BigInt, HYPRE_BigInt, HYPRE_Complex> Tuple3;

struct hypre_IJMatrixSetAddValuesFunctor : public thrust::unary_function<Tuple3, Tuple3>
{
   __device__
   Tuple3 operator()(const Tuple3& t) const
   {
      const HYPRE_BigInt  r = thrust::get<0>(t);
      const HYPRE_BigInt  c = thrust::get<1>(t);
      const HYPRE_Complex v = thrust::get<2>(t);
      return thrust::make_tuple(r, -c-1, v);
   }
};

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
#ifdef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_BigInt row_start = row_partitioning[0];
   HYPRE_BigInt row_end   = row_partitioning[1];
#else
   HYPRE_BigInt row_start = row_partitioning[my_id];
   HYPRE_BigInt row_end   = row_partitioning[my_id+1];
#endif
   HYPRE_Int nrows = row_end - row_start;
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);

   in_range<HYPRE_BigInt> pred(row_start, row_end-1);
   HYPRE_Int nelms_on = HYPRE_THRUST_CALL(count_if, rows, rows+nelms, pred);
   HYPRE_Int nelms_off = nelms - nelms_on;

   /* on proc */
   if (nelms_on)
   {
      if ( hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix) < hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) + nelms_on )
      {
         HYPRE_Int size;
         size = hypre_max( hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) + nelms_on,
                           hypre_AuxParCSRMatrixInitAllocFactor(aux_matrix) * nrows );
         size = hypre_max( hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix) * hypre_AuxParCSRMatrixGrowFactor(aux_matrix),
                           size );
         hypre_AuxParCSRMatrixOnProcI(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOnProcI(aux_matrix),    HYPRE_BigInt,  hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix),
                            HYPRE_BigInt,  size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOnProcJ(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOnProcJ(aux_matrix),    HYPRE_BigInt,  hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix),
                            HYPRE_BigInt,  size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOnProcData(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOnProcData(aux_matrix), HYPRE_Complex, hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix),
                            HYPRE_Complex, size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixMaxOnProcElmts(aux_matrix) = size;
      }

      if (action[0] == 's')
      {
/*
HYPRE_Int nnz = nelms;
printf("nnz %d\n", nnz);
HYPRE_Int *tmpi = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_HOST);
HYPRE_Int *tmpj = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_HOST);
HYPRE_Complex *tmpd = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_HOST);
hypre_TMemcpy(tmpi, rows, HYPRE_Int, nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
hypre_TMemcpy(tmpj, cols, HYPRE_Int, nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
hypre_TMemcpy(tmpd, values, HYPRE_Complex, nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
for (int i=0; i<nnz; i++)
{
   printf("%d ", tmpi[i]);
   printf("%d ", tmpj[i]);
   printf("%e \n", tmpd[i]);
}
printf("--> %d %p %p %p\n", hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOnProcI(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOnProcJ(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOnProcData(aux_matrix) + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix)
      );
HYPRE_CUDA_CALL( cudaGetLastError() );
exit(0);
*/
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
      }
      else if (action[0] == 'a')
      {
         auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(rows,         cols,         values        )),
                                             hypre_IJMatrixSetAddValuesFunctor() ),                     /* first */
            thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(rows + nelms, cols + nelms, values + nelms)),
                                             hypre_IJMatrixSetAddValuesFunctor() ),                     /* last */
            rows,                                                                                       /* stencil */
            thrust::make_zip_iterator(thrust::make_tuple(
                                      hypre_AuxParCSRMatrixOnProcI(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOnProcJ(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOnProcData(aux_matrix) + hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix))),
                                                                                                        /* result */
            pred);

         hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) - hypre_AuxParCSRMatrixOnProcI(aux_matrix) ==
                       hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) + nelms_on);
      }

      hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) += nelms_on;
   }

   /* off proc */
   if (nelms_off)
   {
      if ( hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) < hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) + nelms_off )
      {
         HYPRE_Int size;
         size = hypre_max( hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) + nelms_off,
                           hypre_AuxParCSRMatrixInitAllocFactor(aux_matrix) * nrows );
         size = hypre_max( hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) * hypre_AuxParCSRMatrixGrowFactor(aux_matrix),
                           size );
         hypre_AuxParCSRMatrixOffProcI(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOffProcI(aux_matrix),    HYPRE_BigInt,  hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix),
                            HYPRE_BigInt,  size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOffProcJ(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOffProcJ(aux_matrix),    HYPRE_BigInt,  hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix),
                            HYPRE_BigInt,  size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixOffProcData(aux_matrix) =
         hypre_TReAlloc_v2( hypre_AuxParCSRMatrixOffProcData(aux_matrix), HYPRE_Complex, hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix),
                            HYPRE_Complex, size, HYPRE_MEMORY_DEVICE );
         hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = size;
      }

      if (action[0] == 's')
      {
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
      }
      else if (action[0] == 'a')
      {
         auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(rows,         cols,         values        )),
                                             hypre_IJMatrixSetAddValuesFunctor() ),                     /* first */
            thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(rows + nelms, cols + nelms, values + nelms)),
                                             hypre_IJMatrixSetAddValuesFunctor() ),                     /* last */
            rows,                                                                                       /* stencil */
            thrust::make_zip_iterator(thrust::make_tuple(
                                      hypre_AuxParCSRMatrixOffProcI(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOffProcJ(aux_matrix)    + hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix),
                                      hypre_AuxParCSRMatrixOffProcData(aux_matrix) + hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix))),
                                                                                                        /* result */
            thrust::not1(pred));

         hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) - hypre_AuxParCSRMatrixOffProcI(aux_matrix) ==
                       hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) + nelms_off);
      }

      hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) += nelms_off;
   }

   return hypre_error_flag;
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


HYPRE_Int
hypre_IJMatrixAssembleParCSRDevice(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   hypre_ParCSRMatrix    *par_matrix = (hypre_ParCSRMatrix*)    hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix*) hypre_IJMatrixTranslator(matrix);

   if (!aux_matrix)
   {
      if (!par_matrix)
      {
      }

      return hypre_error_flag;
   }

   /* communicate for aux off-proc and add to aux on-proc */
   if ( hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) )
   {
   }

   /* expand the exisiting csr and add to aux on-proc */
   if (par_matrix)
   {
   }

   /* compress aux on-proc */
/*
   if ( hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix) )
   {
      hypreDevice_StableSortByTupleKey( hypre_AuxParCSRMatrixCurrentOnProcElmts(aux_matrix),
                                        hypre_AuxParCSRMatrixOnProcI(aux_matrix),
                                        hypre_AuxParCSRMatrixOnProcJ(aux_matrix),
                                        hypre_AuxParCSRMatrixOnProcData(aux_matrix),
                                        1 );

      auto new_end = HYPRE_THRUST_CALL(
         reduce_by_key,



   }
*/
   return hypre_error_flag;
}

