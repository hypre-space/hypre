/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJVector_ParCSR interface
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_SYCL)

template<typename T1, typename T2>
struct hypre_IJVectorAssembleFunctor : public thrust::binary_function< std::tuple<T1, T2>, std::tuple<T1, T2>, std::tuple<T1, T2> >
{
   typedef std::tuple<T1, T2> Tuple;

   Tuple operator()(const Tuple& x, const Tuple& y )
   {
      return std::make_tuple( hypre_max(std::get<0>(x), std::get<0>(y)), std::get<1>(x) + std::get<1>(y) );
   }
};

HYPRE_Int hypre_IJVectorAssembleSortAndReduce3(HYPRE_Int N0, HYPRE_BigInt *I0, char *X0, HYPRE_Complex *A0, HYPRE_Int *N1, HYPRE_BigInt **I1, HYPRE_Complex **A1);

HYPRE_Int hypre_IJVectorAssembleSortAndReduce1(HYPRE_Int N0, HYPRE_BigInt *I0, char *X0, HYPRE_Complex *A0, HYPRE_Int *N1, HYPRE_BigInt **I1, char **X1, HYPRE_Complex **A1 );

void hypreSYCLKernel_IJVectorAssemblePar(HYPRE_Int n, HYPRE_Complex *x, HYPRE_BigInt *map, HYPRE_BigInt offset, char *SorA, HYPRE_Complex *y);

/*
 */
HYPRE_Int
hypre_IJVectorSetAddValuesParDevice(hypre_IJVector       *vector,
                                    HYPRE_Int             num_values,
                                    const HYPRE_BigInt   *indices,
                                    const HYPRE_Complex  *values,
                                    const char           *action)
{
   HYPRE_BigInt *IJpartitioning = hypre_IJVectorPartitioning(vector);
   HYPRE_BigInt  vec_start, vec_stop;
   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1]-1;
   HYPRE_Int nrows = vec_stop - vec_start + 1;
   const char SorA = action[0] == 's' ? 1 : 0;

   if (num_values <= 0)
   {
      return hypre_error_flag;
   }

   /* this is a special use to set/add local values */
   if (!indices)
   {
      hypre_ParVector *par_vector = (hypre_ParVector*) hypre_IJVectorObject(vector);
      hypre_Vector    *local_vector = hypre_ParVectorLocalVector(par_vector);
      HYPRE_Int        num_values2 = hypre_min( hypre_VectorSize(local_vector), num_values );
      HYPRE_BigInt    *indices2 = hypre_TAlloc(HYPRE_BigInt, num_values2, HYPRE_MEMORY_DEVICE);
      sycl_iota(indices2, indices2 + num_values2, vec_start);

      hypre_IJVectorSetAddValuesParDevice(vector, num_values2, indices2, values, action);

      hypre_TFree(indices2, HYPRE_MEMORY_DEVICE);

      return hypre_error_flag;
   }

   hypre_AuxParVector *aux_vector = (hypre_AuxParVector*) hypre_IJVectorTranslator(vector);

   if (!aux_vector)
   {
      hypre_AuxParVectorCreate(&aux_vector);
      hypre_AuxParVectorInitialize_v2(aux_vector, HYPRE_MEMORY_DEVICE);
      hypre_IJVectorTranslator(vector) = aux_vector;
   }

   HYPRE_Int      stack_elmts_max      = hypre_AuxParVectorMaxStackElmts(aux_vector);
   HYPRE_Int      stack_elmts_current  = hypre_AuxParVectorCurrentStackElmts(aux_vector);
   HYPRE_Int      stack_elmts_required = stack_elmts_current + num_values;
   HYPRE_BigInt  *stack_i              = hypre_AuxParVectorStackI(aux_vector);
   HYPRE_Complex *stack_data           = hypre_AuxParVectorStackData(aux_vector);
   char          *stack_sora           = hypre_AuxParVectorStackSorA(aux_vector);

   if ( stack_elmts_max < stack_elmts_required )
   {
      HYPRE_Int stack_elmts_max_new = nrows * hypre_AuxParVectorInitAllocFactor(aux_vector);
      if (hypre_AuxParVectorUsrOffProcElmts(aux_vector) >= 0)
      {
         stack_elmts_max_new += hypre_AuxParVectorUsrOffProcElmts(aux_vector);
      }
      stack_elmts_max_new = hypre_max(stack_elmts_max * hypre_AuxParVectorGrowFactor(aux_vector), stack_elmts_max_new);
      stack_elmts_max_new = hypre_max(stack_elmts_required, stack_elmts_max_new);

      hypre_AuxParVectorStackI(aux_vector)    = stack_i    =
         hypre_TReAlloc_v2(stack_i,    HYPRE_BigInt,  stack_elmts_max, HYPRE_BigInt,  stack_elmts_max_new, HYPRE_MEMORY_DEVICE);
      hypre_AuxParVectorStackData(aux_vector) = stack_data =
         hypre_TReAlloc_v2(stack_data, HYPRE_Complex, stack_elmts_max, HYPRE_Complex, stack_elmts_max_new, HYPRE_MEMORY_DEVICE);
      hypre_AuxParVectorStackSorA(aux_vector) = stack_sora =
         hypre_TReAlloc_v2(stack_sora,          char, stack_elmts_max,          char, stack_elmts_max_new, HYPRE_MEMORY_DEVICE);

      hypre_AuxParVectorMaxStackElmts(aux_vector) = stack_elmts_max_new;
   }

   HYPRE_ONEDPL_CALL(std::fill_n, stack_sora + stack_elmts_current, num_values, SorA);

   hypre_TMemcpy(stack_i    + stack_elmts_current, indices, HYPRE_BigInt,  num_values, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(stack_data + stack_elmts_current, values,  HYPRE_Complex, num_values, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_AuxParVectorCurrentStackElmts(aux_vector) += num_values;

   return hypre_error_flag;
}

/******************************************************************************
 *
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJVectorAssembleParDevice(hypre_IJVector *vector)
{
   MPI_Comm comm = hypre_IJVectorComm(vector);
   HYPRE_BigInt *IJpartitioning = hypre_IJVectorPartitioning(vector);
   HYPRE_BigInt  vec_start, vec_stop;
   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1]-1;
   hypre_ParVector *par_vector = (hypre_ParVector*) hypre_IJVectorObject(vector);
   hypre_AuxParVector *aux_vector = (hypre_AuxParVector*) hypre_IJVectorTranslator(vector);

   if (!aux_vector)
   {
      return hypre_error_flag;
   }

   if (!par_vector)
   {
      return hypre_error_flag;
   }

   HYPRE_Int      nelms      = hypre_AuxParVectorCurrentStackElmts(aux_vector);
   HYPRE_BigInt  *stack_i    = hypre_AuxParVectorStackI(aux_vector);
   HYPRE_Complex *stack_data = hypre_AuxParVectorStackData(aux_vector);
   char          *stack_sora = hypre_AuxParVectorStackSorA(aux_vector);

   in_range<HYPRE_BigInt> pred(vec_start, vec_stop);
   HYPRE_Int nelms_on = HYPRE_ONEDPL_CALL(std::count_if, stack_i, stack_i+nelms, pred);
   HYPRE_Int nelms_off = nelms - nelms_on;
   HYPRE_Int nelms_off_max;
   hypre_MPI_Allreduce(&nelms_off, &nelms_off_max, 1, HYPRE_MPI_INT, hypre_MPI_MAX, comm);

   /* communicate for aux off-proc and add to remote aux on-proc */
   if (nelms_off_max)
   {
      HYPRE_Int      new_nnz  = 0;
      HYPRE_BigInt  *new_i    = NULL;
      HYPRE_Complex *new_data = NULL;

      if (nelms_off)
      {
         /* copy off-proc entries out of stack and remove from stack */
         HYPRE_BigInt  *off_proc_i    = hypre_TAlloc(HYPRE_BigInt,  nelms_off, HYPRE_MEMORY_DEVICE);
         HYPRE_Complex *off_proc_data = hypre_TAlloc(HYPRE_Complex, nelms_off, HYPRE_MEMORY_DEVICE);
         char          *off_proc_sora = hypre_TAlloc(char,          nelms_off, HYPRE_MEMORY_DEVICE);
         char          *is_on_proc    = hypre_TAlloc(char,          nelms,     HYPRE_MEMORY_DEVICE);

         HYPRE_ONEDPL_CALL(std::transform, stack_i, stack_i + nelms, is_on_proc, pred);

         auto new_end1 = HYPRE_ONEDPL_CALL(
	   std::copy_if,
	   oneapi::dpl::make_zip_iterator(std::make_tuple(stack_i,         stack_data,         stack_sora        )),  /* first */
	   oneapi::dpl::make_zip_iterator(std::make_tuple(stack_i + nelms, stack_data + nelms, stack_sora + nelms)),  /* last */
	   is_on_proc,                                                                                                /* stencil */
	   oneapi::dpl::make_zip_iterator(std::make_tuple(off_proc_i,      off_proc_data,      off_proc_sora)),       /* result */
	   std::not1(oneapi::dpl::identity()) );

         hypre_assert(std::get<0>(new_end1.get_iterator_tuple()) - off_proc_i == nelms_off);

         /* remove off-proc entries from stack */
         auto new_end2 = HYPRE_ONEDPL_CALL(
	   dpct::remove_if,
	   oneapi::dpl::make_zip_iterator(std::make_tuple(stack_i,         stack_data,         stack_sora        )),  /* first */
	   oneapi::dpl::make_zip_iterator(std::make_tuple(stack_i + nelms, stack_data + nelms, stack_sora + nelms)),  /* last */
	   is_on_proc,                                                                                                /* stencil */
	   std::not1(oneapi::dpl::identity()) );

         hypre_assert(std::get<0>(new_end2.get_iterator_tuple()) - stack_i == nelms_on);

         hypre_AuxParVectorCurrentStackElmts(aux_vector) = nelms_on;

         hypre_TFree(is_on_proc, HYPRE_MEMORY_DEVICE);

         /* sort and reduce */
         hypre_IJVectorAssembleSortAndReduce3(nelms_off, off_proc_i, off_proc_sora, off_proc_data, &new_nnz, &new_i, &new_data);

         hypre_TFree(off_proc_i,    HYPRE_MEMORY_DEVICE);
         hypre_TFree(off_proc_data, HYPRE_MEMORY_DEVICE);
         hypre_TFree(off_proc_sora, HYPRE_MEMORY_DEVICE);
      }

      /* send new_i/data to remote processes and the receivers call addtovalues */
      hypre_IJVectorAssembleOffProcValsPar(vector, -1, new_nnz, HYPRE_MEMORY_DEVICE, new_i, new_data);

      hypre_TFree(new_i,    HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_data, HYPRE_MEMORY_DEVICE);
   }

   /* Note: the stack might have been changed in hypre_IJVectorAssembleOffProcValsPar,
    * so must get the size and the pointers again */
   nelms      = hypre_AuxParVectorCurrentStackElmts(aux_vector);
   stack_i    = hypre_AuxParVectorStackI(aux_vector);
   stack_data = hypre_AuxParVectorStackData(aux_vector);
   stack_sora = hypre_AuxParVectorStackSorA(aux_vector);

#ifdef HYPRE_DEBUG
   /* the stack should only have on-proc elements now */
   HYPRE_Int tmp = HYPRE_ONEDPL_CALL(std::count_if, stack_i, stack_i+nelms, pred);
   hypre_assert(nelms == tmp);
#endif

   if (nelms)
   {
      HYPRE_Int      new_nnz;
      HYPRE_BigInt  *new_i;
      HYPRE_Complex *new_data;
      char          *new_sora;

      /* sort and reduce */
      hypre_IJVectorAssembleSortAndReduce1(nelms, stack_i, stack_sora, stack_data, &new_nnz, &new_i, &new_sora, &new_data);

      /* set/add to local vector */
      sycl::range<1> bDim = hypre_GetDefaultSYCLWorkgroupDimension();
      sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(new_nnz, "thread", bDim);
      HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_IJVectorAssemblePar, gDim, bDim, new_nnz, new_data, new_i, vec_start, new_sora,
			    hypre_VectorData(hypre_ParVectorLocalVector(par_vector)) );

      hypre_TFree(new_i,    HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_data, HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_sora, HYPRE_MEMORY_DEVICE);
   }

   hypre_AuxParVectorDestroy(aux_vector);
   hypre_IJVectorTranslator(vector) = NULL;

   return hypre_error_flag;
}

/* helper routine used in hypre_IJVectorAssembleParCSRDevice:
 * 1. sort (X0, A0) with key I0
 * 2. for each segment in I0, zero out in A0 all before the last `set'
 * 3. reduce A0 [with sum] and reduce X0 [with max]
 * N0: input size; N1: size after reduction (<= N0)
 * Note: (I1, X1, A1) are not resized to N1 but have size N0
 */
HYPRE_Int
hypre_IJVectorAssembleSortAndReduce1(HYPRE_Int  N0, HYPRE_BigInt  *I0, char  *X0, HYPRE_Complex  *A0,
                                     HYPRE_Int *N1, HYPRE_BigInt **I1, char **X1, HYPRE_Complex **A1 )
{
   auto vals_begin = oneapi::dpl::make_zip_iterator(std::make_tuple(X0, A0));
   auto zipped_begin = oneapi::dpl::make_zip_iterator(I0, vals_begin);
   HYPRE_ONEDPL_CALL( std::stable_sort, zipped_begin, zipped_begin + N0 );

   HYPRE_BigInt  *I = hypre_TAlloc(HYPRE_BigInt,  N0, HYPRE_MEMORY_DEVICE);
   char          *X = hypre_TAlloc(char,          N0, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *A = hypre_TAlloc(HYPRE_Complex, N0, HYPRE_MEMORY_DEVICE);

   /* output X: 0: keep, 1: zero-out */
   HYPRE_ONEDPL_CALL(
     oneapi::dpl::exclusive_scan_by_segment,
     std::make_reverse_iterator(thrust::device_pointer_cast<HYPRE_BigInt>(I0)+N0),  /* key begin */
     std::make_reverse_iterator(thrust::device_pointer_cast<HYPRE_BigInt>(I0)),     /* key end */
     std::make_reverse_iterator(thrust::device_pointer_cast<char>(X0)+N0),          /* input value begin */
     std::make_reverse_iterator(thrust::device_pointer_cast<char>(X) +N0),          /* output value begin */
     0,                                                                             /* init */
     std::equal_to<HYPRE_BigInt>(),
     oneapi::dpl::maximum<char>() );

   HYPRE_ONEDPL_CALL(std::transform, //replace_if
                     A0, A0 + N0, X, A0, 
                     [](HYPRE_Complex input, char mask) { return mask ? 0.0 : input; } );
   //HYPRE_ONEDPL_CALL(dpct::replace_if, A0, A0 + N0, X, oneapi::dpl::identity(), 0.0);

   auto new_end = HYPRE_ONEDPL_CALL(
     oneapi::dpl::reduce_by_segment,
     I0,                                                                /* keys_first */
     I0 + N0,                                                           /* keys_last */
     oneapi::dpl::make_zip_iterator(std::make_tuple(X0,      A0     )), /* values_first */
     I,                                                                 /* keys_output */
     oneapi::dpl::make_zip_iterator(std::make_tuple(X,       A      )), /* values_output */
     std::equal_to<HYPRE_BigInt>(),                                     /* binary_pred */
     hypre_IJVectorAssembleFunctor<char, HYPRE_Complex>()               /* binary_op */ );

   *N1 = new_end.first - I;
   *I1 = I;
   *X1 = X;
   *A1 = A;

   return hypre_error_flag;
}

HYPRE_Int
hypre_IJVectorAssembleSortAndReduce3(HYPRE_Int  N0, HYPRE_BigInt  *I0, char *X0, HYPRE_Complex  *A0,
                                     HYPRE_Int *N1, HYPRE_BigInt **I1,           HYPRE_Complex **A1)
{
   auto vals_begin = oneapi::dpl::make_zip_iterator(std::make_tuple(X0, A0));
   auto zipped_begin = oneapi::dpl::make_zip_iterator(I0, vals_begin);
   HYPRE_ONEDPL_CALL( std::stable_sort, zipped_begin, zipped_begin + N0 );

   HYPRE_Int     *I = hypre_TAlloc(HYPRE_Int,     N0, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *A = hypre_TAlloc(HYPRE_Complex, N0, HYPRE_MEMORY_DEVICE);

   /* output in X0: 0: keep, 1: zero-out */
   HYPRE_ONEDPL_CALL(
     oneapi::dpl::inclusive_scan_by_segment,
     std::make_reverse_iterator(thrust::device_pointer_cast<HYPRE_BigInt>(I0)+N0), /* key begin */
     std::make_reverse_iterator(thrust::device_pointer_cast<HYPRE_BigInt>(I0)),    /* key end */
     std::make_reverse_iterator(thrust::device_pointer_cast<char>(X0)+N0),         /* input value begin */
     std::make_reverse_iterator(thrust::device_pointer_cast<char>(X0)+N0),         /* output value begin */
     std::equal_to<HYPRE_BigInt>(),
     oneapi::dpl::maximum<char>() );

   HYPRE_ONEDPL_CALL(std::transform, //replace_if
                     A0, A0 + N0, X0, A0, 
                     [](HYPRE_Complex input, char mask) { return mask ? 0.0 : input; } );
   //HYPRE_ONEDPL_CALL(dpct::replace_if, A0, A0 + N0, X0, oneapi::dpl::identity(), 0.0);

   auto new_end = HYPRE_ONEDPL_CALL(
     oneapi::dpl::reduce_by_segment,
     I0,      /* keys_first */
     I0 + N0, /* keys_last */
     A0,      /* values_first */
     I,       /* keys_output */
     A        /* values_output */);

   *N1 = new_end.second - A;
   *I1 = I;
   *A1 = A;

   return hypre_error_flag;
}

/* y[map[i]-offset] = x[i] or y[map[i]] += x[i] depending on SorA,
 * same index cannot appear more than once in map */
void
hypreSYCLKernel_IJVectorAssemblePar(sycl::nd_item<1>& item, HYPRE_Int n, HYPRE_Complex *x,
				    HYPRE_BigInt *map, HYPRE_BigInt offset, char *SorA, HYPRE_Complex *y)
{
   HYPRE_Int i = item.get_global_linear_id();

   if (i >= n)
   {
      return;
   }

   if (SorA[i])
   {
      y[map[i]-offset] = x[i];
   }
   else
   {
      y[map[i]-offset] += x[i];
   }
}

#endif

