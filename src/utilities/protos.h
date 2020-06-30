/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* amg_linklist.c */
void hypre_dispose_elt ( hypre_LinkList element_ptr );
void hypre_remove_point ( hypre_LinkList *LoL_head_ptr , hypre_LinkList *LoL_tail_ptr , HYPRE_Int measure , HYPRE_Int index , HYPRE_Int *lists , HYPRE_Int *where );
hypre_LinkList hypre_create_elt ( HYPRE_Int Item );
void hypre_enter_on_lists ( hypre_LinkList *LoL_head_ptr , hypre_LinkList *LoL_tail_ptr , HYPRE_Int measure , HYPRE_Int index , HYPRE_Int *lists , HYPRE_Int *where );

/* binsearch.c */
HYPRE_Int hypre_BinarySearch ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int list_length );
HYPRE_Int hypre_BigBinarySearch ( HYPRE_BigInt *list , HYPRE_BigInt value , HYPRE_Int list_length );
HYPRE_Int hypre_BinarySearch2 ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int low , HYPRE_Int high , HYPRE_Int *spot );
HYPRE_Int *hypre_LowerBound( HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int value );
HYPRE_BigInt *hypre_BigLowerBound( HYPRE_BigInt *first, HYPRE_BigInt *last, HYPRE_BigInt value );

/* hypre_complex.c */
#ifdef HYPRE_COMPLEX
HYPRE_Complex hypre_conj( HYPRE_Complex value );
HYPRE_Real    hypre_cabs( HYPRE_Complex value );
HYPRE_Real    hypre_creal( HYPRE_Complex value );
HYPRE_Real    hypre_cimag( HYPRE_Complex value );
#else
#define hypre_conj(value)  value
#define hypre_cabs(value)  fabs(value)
#define hypre_creal(value) value
#define hypre_cimag(value) 0.0
#endif

/* hypre_general.c */
hypre_Handle* hypre_handle();
hypre_Handle* hypre_HandleCreate();
HYPRE_Int hypre_HandleDestroy(hypre_Handle *hypre_handle_);
HYPRE_Int HYPRE_Init();
HYPRE_Int HYPRE_Finalize();
HYPRE_Int hypre_SetDevice(HYPRE_Int use_device, hypre_Handle *hypre_handle_);

/* hypre_qsort.c */
void hypre_swap ( HYPRE_Int *v , HYPRE_Int i , HYPRE_Int j );
void hypre_swap2 ( HYPRE_Int *v , HYPRE_Real *w , HYPRE_Int i , HYPRE_Int j );
void hypre_BigSwap2 ( HYPRE_BigInt *v , HYPRE_Real *w , HYPRE_Int i , HYPRE_Int j );
void hypre_swap2i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int i , HYPRE_Int j );
void hypre_BigSwap2i ( HYPRE_BigInt *v , HYPRE_Int *w , HYPRE_Int i , HYPRE_Int j );
void hypre_swap3i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int i , HYPRE_Int j );
void hypre_swap3_d ( HYPRE_Real *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int i , HYPRE_Int j );
void hypre_swap3_d_perm(HYPRE_Int  *v, HYPRE_Real  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwap4_d ( HYPRE_Real *v , HYPRE_BigInt *w , HYPRE_Int *z , HYPRE_Int *y , HYPRE_Int i , HYPRE_Int j );
void hypre_swap_d ( HYPRE_Real *v , HYPRE_Int i , HYPRE_Int j );
void hypre_qsort0 ( HYPRE_Int *v , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort1 ( HYPRE_Int *v , HYPRE_Real *w , HYPRE_Int left , HYPRE_Int right );
void hypre_BigQsort1 ( HYPRE_BigInt *v , HYPRE_Real *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort2i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int left , HYPRE_Int right );
void hypre_BigQsort2i( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_qsort2 ( HYPRE_Int *v , HYPRE_Real *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort2_abs ( HYPRE_Int *v , HYPRE_Real *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort3i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort3ir ( HYPRE_Int *v , HYPRE_Real *w , HYPRE_Int *z , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort3( HYPRE_Real *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left, HYPRE_Int  right );
void hypre_qsort3_abs ( HYPRE_Real *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int left , HYPRE_Int right );
void hypre_BigQsort4_abs ( HYPRE_Real *v , HYPRE_BigInt *w , HYPRE_Int *z , HYPRE_Int *y , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort_abs ( HYPRE_Real *w , HYPRE_Int left , HYPRE_Int right );
void hypre_BigSwapbi(HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigQsortbi( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigSwapLoc(HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigQsortbLoc( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigSwapb2i(HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigQsortb2i( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigSwap( HYPRE_BigInt *v, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigQsort0( HYPRE_BigInt *v, HYPRE_Int  left, HYPRE_Int  right );
void hypre_topo_sort(const HYPRE_Int *row_ptr, const HYPRE_Int *col_inds, const HYPRE_Complex *data, HYPRE_Int *ordering, HYPRE_Int n);
void hypre_dense_topo_sort(const HYPRE_Complex *L, HYPRE_Int *ordering, HYPRE_Int n, HYPRE_Int is_col_major);

/* qsplit.c */
HYPRE_Int hypre_DoubleQuickSplit ( HYPRE_Real *values , HYPRE_Int *indices , HYPRE_Int list_length , HYPRE_Int NumberKept );

/* random.c */
/* HYPRE_CUDA_GLOBAL */ void hypre_SeedRand ( HYPRE_Int seed );
/* HYPRE_CUDA_GLOBAL */ HYPRE_Int hypre_RandI ( void );
/* HYPRE_CUDA_GLOBAL */ HYPRE_Real hypre_Rand ( void );

/* hypre_prefix_sum.c */
/**
 * Assumed to be called within an omp region.
 * Let x_i be the input of ith thread.
 * The output of ith thread y_i = x_0 + x_1 + ... + x_{i-1}
 * Additionally, sum = x_0 + x_1 + ... + x_{nthreads - 1}
 * Note that always y_0 = 0
 *
 * @param workspace at least with length (nthreads+1)
 *                  workspace[tid] will contain result for tid
 *                  workspace[nthreads] will contain sum
 */
void hypre_prefix_sum(HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int *workspace);
/**
 * This version does prefix sum in pair.
 * Useful when we prefix sum of diag and offd in tandem.
 *
 * @param worksapce at least with length 2*(nthreads+1)
 *                  workspace[2*tid] and workspace[2*tid+1] will contain results for tid
 *                  workspace[3*nthreads] and workspace[3*nthreads + 1] will contain sums
 */
void hypre_prefix_sum_pair(HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2, HYPRE_Int *workspace);
/**
 * @param workspace at least with length 3*(nthreads+1)
 *                  workspace[3*tid:3*tid+3) will contain results for tid
 */
void hypre_prefix_sum_triple(HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2, HYPRE_Int *in_out3, HYPRE_Int *sum3, HYPRE_Int *workspace);

/**
 * n prefix-sums together.
 * workspace[n*tid:n*(tid+1)) will contain results for tid
 * workspace[nthreads*tid:nthreads*(tid+1)) will contain sums
 *
 * @param workspace at least with length n*(nthreads+1)
 */
void hypre_prefix_sum_multiple(HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int n, HYPRE_Int *workspace);

/* hypre_merge_sort.c */
/**
 * Why merge sort?
 * 1) Merge sort can take advantage of eliminating duplicates.
 * 2) Merge sort is more efficiently parallelizable than qsort
 */

/**
 * Out of place merge sort with duplicate elimination
 * @ret number of unique elements
 */
HYPRE_Int hypre_merge_sort_unique(HYPRE_Int *in, HYPRE_Int *out, HYPRE_Int len);
/**
 * Out of place merge sort with duplicate elimination
 *
 * @param out pointer to output can be in or temp
 * @ret number of unique elements
 */
HYPRE_Int hypre_merge_sort_unique2(HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **out);

void hypre_merge_sort(HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **sorted);

void hypre_union2(HYPRE_Int n1, HYPRE_BigInt *arr1, HYPRE_Int n2, HYPRE_BigInt *arr2, HYPRE_Int *n3, HYPRE_BigInt *arr3, HYPRE_Int *map1, HYPRE_Int *map2);

/* hypre_hopscotch_hash.c */

#ifdef HYPRE_USING_OPENMP

/* Check if atomic operations are available to use concurrent hopscotch hash table */
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
#define HYPRE_USING_ATOMIC
//#elif defined _MSC_VER // JSP: haven't tested, so comment out for now
//#define HYPRE_USING_ATOMIC
//#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
// JSP: not many compilers have implemented this, so comment out for now
//#define HYPRE_USING_ATOMIC
//#include <stdatomic.h>
#endif

#endif // HYPRE_USING_OPENMP

#ifdef HYPRE_HOPSCOTCH
#ifdef HYPRE_USING_ATOMIC
// concurrent hopscotch hashing is possible only with atomic supports
#define HYPRE_CONCURRENT_HOPSCOTCH
#endif
#endif

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
typedef struct
{
  HYPRE_Int volatile timestamp;
  omp_lock_t         lock;
} hypre_HopscotchSegment;
#endif

/**
 * The current typical use case of unordered set is putting input sequence
 * with lots of duplication (putting all colidx received from other ranks),
 * followed by one sweep of enumeration.
 * Since the capacity is set to the number of inputs, which is much larger
 * than the number of unique elements, we optimize for initialization and
 * enumeration whose time is proportional to the capacity.
 * For initialization and enumeration, structure of array (SoA) is better
 * for vectorization, cache line utilization, and so on.
 */
typedef struct
{
   HYPRE_Int  volatile              segmentMask;
   HYPRE_Int  volatile              bucketMask;
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   hypre_HopscotchSegment* volatile segments;
#endif
   HYPRE_Int *volatile              key;
   hypre_uint *volatile             hopInfo;
   HYPRE_Int *volatile              hash;
} hypre_UnorderedIntSet;

typedef struct
{
   HYPRE_Int volatile            segmentMask;
   HYPRE_Int volatile            bucketMask;
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   hypre_HopscotchSegment* volatile segments;
#endif
   HYPRE_BigInt *volatile           key;
   hypre_uint *volatile             hopInfo;
   HYPRE_BigInt *volatile           hash;
} hypre_UnorderedBigIntSet;

typedef struct
{
  hypre_uint volatile hopInfo;
  HYPRE_Int  volatile hash;
  HYPRE_Int  volatile key;
  HYPRE_Int  volatile data;
} hypre_HopscotchBucket;

typedef struct
{
  hypre_uint volatile hopInfo;
  HYPRE_BigInt  volatile hash;
  HYPRE_BigInt  volatile key;
  HYPRE_Int  volatile data;
} hypre_BigHopscotchBucket;

/**
 * The current typical use case of unoredered map is putting input sequence
 * with no duplication (inverse map of a bijective mapping) followed by
 * lots of lookups.
 * For lookup, array of structure (AoS) gives better cache line utilization.
 */
typedef struct
{
   HYPRE_Int  volatile              segmentMask;
   HYPRE_Int  volatile              bucketMask;
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   hypre_HopscotchSegment* volatile segments;
#endif
   hypre_HopscotchBucket* volatile table;
} hypre_UnorderedIntMap;

typedef struct
{
   HYPRE_Int  volatile segmentMask;
   HYPRE_Int  volatile bucketMask;
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   hypre_HopscotchSegment* volatile segments;
#endif
   hypre_BigHopscotchBucket* volatile table;
} hypre_UnorderedBigIntMap;

/**
 * Sort array "in" with length len and put result in array "out"
 * "in" will be deallocated unless in == *out
 * inverse_map is an inverse hash table s.t. inverse_map[i] = j iff (*out)[j] = i
 */
void hypre_sort_and_create_inverse_map(
  HYPRE_Int *in, HYPRE_Int len, HYPRE_Int **out, hypre_UnorderedIntMap *inverse_map);

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
void hypre_big_merge_sort(HYPRE_BigInt *in, HYPRE_BigInt *temp, HYPRE_Int len, HYPRE_BigInt **sorted);
void hypre_big_sort_and_create_inverse_map(HYPRE_BigInt *in, HYPRE_Int len, HYPRE_BigInt **out, hypre_UnorderedBigIntMap *inverse_map);
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
HYPRE_Int hypre_SyncCudaComputeStream(hypre_Handle *hypre_handle);
HYPRE_Int hypre_SyncCudaDevice(hypre_Handle *hypre_handle);
HYPRE_Int hypreDevice_DiagScaleVector(HYPRE_Int n, HYPRE_Int *A_i, HYPRE_Complex *A_data, HYPRE_Complex *x, HYPRE_Complex *y);
HYPRE_Int hypreDevice_IVAXPY(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y);
HYPRE_Int hypreDevice_BigIntFilln(HYPRE_BigInt *d_x, size_t n, HYPRE_BigInt v);
#endif

/* hypre_nvtx.c */
void hypre_NvtxPushRangeColor(const char *name, HYPRE_Int cid);
void hypre_NvtxPushRange(const char *name);
void hypre_NvtxPopRange();

