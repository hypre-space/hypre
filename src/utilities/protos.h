/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* amg_linklist.c */
void hypre_dispose_elt ( hypre_LinkList element_ptr );
void hypre_remove_point ( hypre_LinkList *LoL_head_ptr, hypre_LinkList *LoL_tail_ptr,
                          HYPRE_Int measure, HYPRE_Int index, HYPRE_Int *lists, HYPRE_Int *where );
hypre_LinkList hypre_create_elt ( HYPRE_Int Item );
void hypre_enter_on_lists ( hypre_LinkList *LoL_head_ptr, hypre_LinkList *LoL_tail_ptr,
                            HYPRE_Int measure, HYPRE_Int index, HYPRE_Int *lists, HYPRE_Int *where );

/* binsearch.c */
HYPRE_Int hypre_BinarySearch ( HYPRE_Int *list, HYPRE_Int value, HYPRE_Int list_length );
HYPRE_Int hypre_BigBinarySearch ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_Int hypre_BinarySearch2 ( HYPRE_Int *list, HYPRE_Int value, HYPRE_Int low, HYPRE_Int high,
                                HYPRE_Int *spot );
HYPRE_Int *hypre_LowerBound( HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int value );
HYPRE_BigInt *hypre_BigLowerBound( HYPRE_BigInt *first, HYPRE_BigInt *last, HYPRE_BigInt value );

/* log.c */
HYPRE_Int hypre_Log2( HYPRE_Int p );

/* complex.c */
#ifdef HYPRE_COMPLEX
HYPRE_Complex hypre_conj( HYPRE_Complex value );
HYPRE_Real    hypre_cabs( HYPRE_Complex value );
HYPRE_Real    hypre_creal( HYPRE_Complex value );
HYPRE_Real    hypre_cimag( HYPRE_Complex value );
HYPRE_Complex hypre_csqrt( HYPRE_Complex value );
#else
#define hypre_conj(value)  value
#define hypre_cabs(value)  hypre_abs(value)
#define hypre_creal(value) value
#define hypre_cimag(value) 0.0
#define hypre_csqrt(value) hypre_sqrt(value)
#endif

/* state.c */
HYPRE_Int hypre_Initialized( void );
HYPRE_Int hypre_Finalized( void );
HYPRE_Int hypre_SetInitialized( void );
HYPRE_Int hypre_SetFinalized( void );

/* general.c */
hypre_Handle* hypre_handle(void);
hypre_Handle* hypre_HandleCreate(void);
HYPRE_Int hypre_HandleDestroy(hypre_Handle *hypre_handle_);
HYPRE_Int hypre_SetDevice(hypre_int device_id, hypre_Handle *hypre_handle_);
HYPRE_Int hypre_GetDevice(hypre_int *device_id);
HYPRE_Int hypre_GetDeviceCount(hypre_int *device_count);
HYPRE_Int hypre_GetDeviceLastError(void);
HYPRE_Int hypre_UmpireInit(hypre_Handle *hypre_handle_);
HYPRE_Int hypre_UmpireFinalize(hypre_Handle *hypre_handle_);
HYPRE_Int hypre_GetDeviceMaxShmemSize(hypre_int device_id, hypre_int *max_size_ptr,
                                      hypre_int *max_size_optin_ptr);

/* matrix_stats.h */
hypre_MatrixStats* hypre_MatrixStatsCreate( void );
HYPRE_Int hypre_MatrixStatsDestroy( hypre_MatrixStats *stats );
hypre_MatrixStatsArray* hypre_MatrixStatsArrayCreate( HYPRE_Int capacity );
HYPRE_Int hypre_MatrixStatsArrayDestroy( hypre_MatrixStatsArray *stats_array );
HYPRE_Int hypre_MatrixStatsArrayPrint( HYPRE_Int num_hierarchies, HYPRE_Int *num_levels,
                                       HYPRE_Int use_divisors, HYPRE_Int shift,
                                       const char **messages,
                                       hypre_MatrixStatsArray *stats_array );

/* qsort.c */
void hypre_swap ( HYPRE_Int *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_c ( HYPRE_Complex *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2 ( HYPRE_Int *v, HYPRE_Real *w, HYPRE_Int i, HYPRE_Int j );
void hypre_BigSwap2 ( HYPRE_BigInt *v, HYPRE_Real *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2i ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_BigSwap2i ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3i ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3_d ( HYPRE_Real *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3_d_perm(HYPRE_Int  *v, HYPRE_Real  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwap4_d ( HYPRE_Real *v, HYPRE_BigInt *w, HYPRE_Int *z, HYPRE_Int *y, HYPRE_Int i,
                        HYPRE_Int j );
void hypre_swap_d ( HYPRE_Real *v, HYPRE_Int i, HYPRE_Int j );
void hypre_qsort0 ( HYPRE_Int *v, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort1 ( HYPRE_Int *v, HYPRE_Real *w, HYPRE_Int left, HYPRE_Int right );
void hypre_BigQsort1 ( HYPRE_BigInt *v, HYPRE_Real *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2i ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int left, HYPRE_Int right );
void hypre_BigQsort2i( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_qsort2 ( HYPRE_Int *v, HYPRE_Real *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_abs ( HYPRE_Int *v, HYPRE_Real *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3i ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3ir ( HYPRE_Int *v, HYPRE_Real *w, HYPRE_Int *z, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3( HYPRE_Real *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left, HYPRE_Int  right );
void hypre_qsort3_abs ( HYPRE_Real *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int left,
                        HYPRE_Int right );
void hypre_BigQsort4_abs ( HYPRE_Real *v, HYPRE_BigInt *w, HYPRE_Int *z, HYPRE_Int *y,
                           HYPRE_Int left, HYPRE_Int right );
void hypre_qsort_abs ( HYPRE_Real *w, HYPRE_Int left, HYPRE_Int right );
void hypre_BigSwapbi(HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigQsortbi( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigSwapLoc(HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigQsortbLoc( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigSwapb2i(HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigQsortb2i( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left,
                        HYPRE_Int  right );
void hypre_BigSwap( HYPRE_BigInt *v, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigQsort0( HYPRE_BigInt *v, HYPRE_Int  left, HYPRE_Int  right );
void hypre_topo_sort(const HYPRE_Int *row_ptr, const HYPRE_Int *col_inds, const HYPRE_Complex *data,
                     HYPRE_Int *ordering, HYPRE_Int n);
void hypre_dense_topo_sort(const HYPRE_Complex *L, HYPRE_Int *ordering, HYPRE_Int n,
                           HYPRE_Int is_col_major);

/* qsplit.c */
HYPRE_Int hypre_DoubleQuickSplit ( HYPRE_Real *values, HYPRE_Int *indices, HYPRE_Int list_length,
                                   HYPRE_Int NumberKept );

/* random.c */
/* HYPRE_CUDA_GLOBAL */ void hypre_SeedRand ( HYPRE_Int seed );
/* HYPRE_CUDA_GLOBAL */ HYPRE_Int hypre_RandI ( void );
/* HYPRE_CUDA_GLOBAL */ HYPRE_Real hypre_Rand ( void );

/* prefix_sum.c */
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
void hypre_prefix_sum_pair(HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2,
                           HYPRE_Int *workspace);
/**
 * @param workspace at least with length 3*(nthreads+1)
 *                  workspace[3*tid:3*tid+3) will contain results for tid
 */
void hypre_prefix_sum_triple(HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2,
                             HYPRE_Int *sum2, HYPRE_Int *in_out3, HYPRE_Int *sum3, HYPRE_Int *workspace);

/**
 * n prefix-sums together.
 * workspace[n*tid:n*(tid+1)) will contain results for tid
 * workspace[nthreads*tid:nthreads*(tid+1)) will contain sums
 *
 * @param workspace at least with length n*(nthreads+1)
 */
void hypre_prefix_sum_multiple(HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int n,
                               HYPRE_Int *workspace);

/* hopscotch_hash.c */

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

/* merge_sort.c */
/**
 * Why merge sort?
 * 1) Merge sort can take advantage of eliminating duplicates.
 * 2) Merge sort is more efficiently parallelizable than qsort
 */
HYPRE_Int hypre_IntArrayMergeOrdered( hypre_IntArray *array1, hypre_IntArray *array2,
                                      hypre_IntArray *array3 );
void hypre_union2(HYPRE_Int n1, HYPRE_BigInt *arr1, HYPRE_Int n2, HYPRE_BigInt *arr2, HYPRE_Int *n3,
                  HYPRE_BigInt *arr3, HYPRE_Int *map1, HYPRE_Int *map2);
void hypre_merge_sort(HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **sorted);
void hypre_big_merge_sort(HYPRE_BigInt *in, HYPRE_BigInt *temp, HYPRE_Int len,
                          HYPRE_BigInt **sorted);
void hypre_sort_and_create_inverse_map(HYPRE_Int *in, HYPRE_Int len, HYPRE_Int **out,
                                       hypre_UnorderedIntMap *inverse_map);
void hypre_big_sort_and_create_inverse_map(HYPRE_BigInt *in, HYPRE_Int len, HYPRE_BigInt **out,
                                           hypre_UnorderedBigIntMap *inverse_map);

/* device_utils.c */
#if defined(HYPRE_USING_GPU)
HYPRE_Int hypre_SyncComputeStream(hypre_Handle *hypre_handle);
HYPRE_Int hypre_SyncCudaDevice(hypre_Handle *hypre_handle);
HYPRE_Int hypre_ResetCudaDevice(hypre_Handle *hypre_handle);
HYPRE_Int hypreDevice_DiagScaleVector(HYPRE_Int num_vectors, HYPRE_Int num_rows,
                                      HYPRE_Int *A_i, HYPRE_Complex *A_data,
                                      HYPRE_Complex *x, HYPRE_Complex beta,
                                      HYPRE_Complex *y);
HYPRE_Int hypreDevice_DiagScaleVector2(HYPRE_Int num_vectors, HYPRE_Int num_rows,
                                       HYPRE_Complex *diag, HYPRE_Complex *x,
                                       HYPRE_Complex beta, HYPRE_Complex *y,
                                       HYPRE_Complex *z, HYPRE_Int computeY);
HYPRE_Int hypreDevice_ComplexArrayToArrayOfPtrs(HYPRE_Int n, HYPRE_Int m,
                                                HYPRE_Complex *data, HYPRE_Complex **data_aop);
HYPRE_Int hypreDevice_zeqxmydd(HYPRE_Int n, HYPRE_Complex *x, HYPRE_Complex alpha,
                               HYPRE_Complex *y, HYPRE_Complex *z, HYPRE_Complex *d);
HYPRE_Int hypreDevice_IVAXPY(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x, HYPRE_Complex *y);
HYPRE_Int hypreDevice_IVAXPYMarked(HYPRE_Int n, HYPRE_Complex *a, HYPRE_Complex *x,
                                   HYPRE_Complex *y, HYPRE_Int *marker, HYPRE_Int marker_val);
HYPRE_Int hypreDevice_IVAMXPMY(HYPRE_Int m, HYPRE_Int n, HYPRE_Complex *a,
                               HYPRE_Complex *x, HYPRE_Complex *y);
HYPRE_Int hypreDevice_IntFilln(HYPRE_Int *d_x, size_t n, HYPRE_Int v);
HYPRE_Int hypreDevice_BigIntFilln(HYPRE_BigInt *d_x, size_t n, HYPRE_BigInt v);
HYPRE_Int hypreDevice_ComplexFilln(HYPRE_Complex *d_x, size_t n, HYPRE_Complex v);
HYPRE_Int hypreDevice_CharFilln(char *d_x, size_t n, char v);
HYPRE_Int hypreDevice_IntStridedCopy ( HYPRE_Int size, HYPRE_Int stride,
                                       HYPRE_Int *in, HYPRE_Int *out );
HYPRE_Int hypreDevice_ComplexStridedCopy ( HYPRE_Int size, HYPRE_Int stride,
                                           HYPRE_Complex *in, HYPRE_Complex *out );
HYPRE_Int hypreDevice_IntScalen(HYPRE_Int *d_x, size_t n, HYPRE_Int *d_y, HYPRE_Int v);
HYPRE_Int hypreDevice_ComplexScalen(HYPRE_Complex *d_x, size_t n, HYPRE_Complex *d_y,
                                    HYPRE_Complex v);
HYPRE_Int hypreDevice_ComplexAxpyn(HYPRE_Complex *d_x, size_t n, HYPRE_Complex *d_y,
                                   HYPRE_Complex *d_z, HYPRE_Complex a);
HYPRE_Int hypreDevice_ComplexAxpyzn(HYPRE_Int n, HYPRE_Complex *d_x, HYPRE_Complex *d_y,
                                    HYPRE_Complex *d_z, HYPRE_Complex a, HYPRE_Complex b);
HYPRE_Int hypreDevice_IntAxpyn(HYPRE_Int *d_x, size_t n, HYPRE_Int *d_y, HYPRE_Int *d_z,
                               HYPRE_Int a);
HYPRE_Int hypreDevice_BigIntAxpyn(HYPRE_BigInt *d_x, size_t n, HYPRE_BigInt *d_y,
                                  HYPRE_BigInt *d_z, HYPRE_BigInt a);
HYPRE_Int* hypreDevice_CsrRowPtrsToIndices(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr);
HYPRE_Int hypreDevice_CsrRowPtrsToIndices_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr,
                                             HYPRE_Int *d_row_ind);
HYPRE_Int* hypreDevice_CsrRowIndicesToPtrs(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind);
HYPRE_Int hypreDevice_CsrRowIndicesToPtrs_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind,
                                             HYPRE_Int *d_row_ptr);
HYPRE_Int hypreDevice_GetRowNnz(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int *d_diag_ia,
                                HYPRE_Int *d_offd_ia, HYPRE_Int *d_rownnz);
HYPRE_Int hypreDevice_CopyParCSRRows(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int job,
                                     HYPRE_Int has_offd, HYPRE_BigInt first_col,
                                     HYPRE_BigInt *d_col_map_offd_A, HYPRE_Int *d_diag_i,
                                     HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a,
                                     HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j,
                                     HYPRE_Complex *d_offd_a, HYPRE_Int *d_ib,
                                     HYPRE_BigInt *d_jb, HYPRE_Complex *d_ab);
HYPRE_Int hypreDevice_IntegerReduceSum(HYPRE_Int m, HYPRE_Int *d_i);
HYPRE_Complex hypreDevice_ComplexReduceSum(HYPRE_Int m, HYPRE_Complex *d_x);
HYPRE_Int hypreDevice_IntegerInclusiveScan(HYPRE_Int n, HYPRE_Int *d_i);
HYPRE_Int hypreDevice_IntegerExclusiveScan(HYPRE_Int n, HYPRE_Int *d_i);
HYPRE_Int hypre_CudaCompileFlagCheck(void);
#endif

HYPRE_Int hypre_CurandUniform( HYPRE_Int n, HYPRE_Real *urand, HYPRE_Int set_seed,
                               hypre_ulonglongint seed, HYPRE_Int set_offset, hypre_ulonglongint offset);
HYPRE_Int hypre_CurandUniformSingle( HYPRE_Int n, float *urand, HYPRE_Int set_seed,
                                     hypre_ulonglongint seed, HYPRE_Int set_offset, hypre_ulonglongint offset);

HYPRE_Int hypre_ResetDeviceRandGenerator( hypre_ulonglongint seed, hypre_ulonglongint offset );

HYPRE_Int hypre_bind_device_id(HYPRE_Int device_id_in, HYPRE_Int myid,
                               HYPRE_Int nproc, MPI_Comm comm);
HYPRE_Int hypre_bind_device(HYPRE_Int myid, HYPRE_Int nproc, MPI_Comm comm);

/* nvtx.c */
void hypre_GpuProfilingPushRangeColor(const char *name, HYPRE_Int cid);
void hypre_GpuProfilingPushRange(const char *name);
void hypre_GpuProfilingPopRange(void);

/* utilities.c */
HYPRE_Int hypre_multmod(HYPRE_Int a, HYPRE_Int b, HYPRE_Int mod);
void hypre_partition1D(HYPRE_Int n, HYPRE_Int p, HYPRE_Int j, HYPRE_Int *s, HYPRE_Int *e);
char *hypre_strcpy(char *destination, const char *source);
HYPRE_Int hypre_CheckDirExists(const char *path);
HYPRE_Int hypre_CreateDir(const char *path);
HYPRE_Int hypre_CreateNextDirOfSequence(const char *basepath, const char *prefix,
                                        char **fullpath_ptr);

HYPRE_Int hypre_SetSyncCudaCompute(HYPRE_Int action);
HYPRE_Int hypre_RestoreSyncCudaCompute(void);
HYPRE_Int hypre_GetSyncCudaCompute(HYPRE_Int *cuda_compute_stream_sync_ptr);
HYPRE_Int hypre_ForceSyncComputeStream(hypre_Handle *hypre_handle);

/* handle.c */
HYPRE_Int hypre_SetSpTransUseVendor( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpMVUseVendor( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpGemmUseVendor( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpGemmAlgorithm( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmBinned( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateMethod( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateNSamples( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateMultFactor( HYPRE_Real value );
HYPRE_Int hypre_SetSpGemmHashType( char value );
HYPRE_Int hypre_SetUseGpuRand( HYPRE_Int use_gpurand );
HYPRE_Int hypre_SetGaussSeidelMethod( HYPRE_Int gs_method );
HYPRE_Int hypre_SetUserDeviceMalloc(GPUMallocFunc func);
HYPRE_Int hypre_SetUserDeviceMfree(GPUMfreeFunc func);
HYPRE_Int hypre_SetGpuAwareMPI( HYPRE_Int use_gpu_aware_mpi );
HYPRE_Int hypre_GetGpuAwareMPI(void);

/* int_array.c */
hypre_IntArray* hypre_IntArrayCreate( HYPRE_Int size );
HYPRE_Int hypre_IntArrayDestroy( hypre_IntArray *array );
HYPRE_Int hypre_IntArrayInitialize_v2( hypre_IntArray *array,
                                       HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayInitialize( hypre_IntArray *array );
HYPRE_Int hypre_IntArrayCopy( hypre_IntArray *x, hypre_IntArray *y );
hypre_IntArray* hypre_IntArrayCloneDeep_v2( hypre_IntArray *x,
                                            HYPRE_MemoryLocation memory_location );
hypre_IntArray* hypre_IntArrayCloneDeep( hypre_IntArray *x );
HYPRE_Int hypre_IntArrayMigrate( hypre_IntArray *v, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayPrint( MPI_Comm comm, hypre_IntArray *array, const char *filename );
HYPRE_Int hypre_IntArrayRead( MPI_Comm comm, const char *filename, hypre_IntArray **array_ptr );
HYPRE_Int hypre_IntArraySetConstantValuesHost( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_IntArraySetConstantValues( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_IntArraySetInterleavedValues( hypre_IntArray *v, HYPRE_Int cycle );
HYPRE_Int hypre_IntArrayCountHost( hypre_IntArray *v, HYPRE_Int value,
                                   HYPRE_Int *num_values_ptr );
HYPRE_Int hypre_IntArrayCount( hypre_IntArray *v, HYPRE_Int value,
                               HYPRE_Int *num_values_ptr );
HYPRE_Int hypre_IntArrayInverseMapping( hypre_IntArray *v, hypre_IntArray **w_ptr );
HYPRE_Int hypre_IntArrayNegate( hypre_IntArray *v );
HYPRE_Int hypre_IntArraySeparateByValue( HYPRE_Int num_values, HYPRE_Int *values,
                                         HYPRE_Int *sizes, hypre_IntArray *v,
                                         hypre_IntArrayArray **w_ptr );
hypre_IntArrayArray* hypre_IntArrayArrayCreate( HYPRE_Int num_entries, HYPRE_Int *sizes );
HYPRE_Int hypre_IntArrayArrayDestroy( hypre_IntArrayArray *w );
HYPRE_Int hypre_IntArrayArrayInitializeIn( hypre_IntArrayArray *w,
                                           HYPRE_MemoryLocation  memory_location );
HYPRE_Int hypre_IntArrayArrayInitialize( hypre_IntArrayArray *w );
HYPRE_Int hypre_IntArrayArrayMigrate( hypre_IntArrayArray *w,
                                      HYPRE_MemoryLocation memory_location );

/* int_array_device.c */
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
HYPRE_Int hypre_IntArraySetConstantValuesDevice( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_IntArrayCountDevice ( hypre_IntArray *v, HYPRE_Int value,
                                      HYPRE_Int *num_values_ptr );
HYPRE_Int hypre_IntArrayInverseMappingDevice( hypre_IntArray *v, hypre_IntArray *w );
HYPRE_Int hypre_IntArrayNegateDevice( hypre_IntArray *v );
HYPRE_Int hypre_IntArraySetInterleavedValuesDevice( hypre_IntArray *v, HYPRE_Int cycle );
HYPRE_Int hypre_IntArraySeparateByValueDevice( HYPRE_Int num_values, HYPRE_Int *values,
                                               HYPRE_Int *sizes, hypre_IntArray *v,
                                               hypre_IntArrayArray *w );
#endif

/* memory_tracker.c */
#ifdef HYPRE_USING_MEMORY_TRACKER
hypre_MemoryTracker* hypre_memory_tracker(void);
hypre_MemoryTracker * hypre_MemoryTrackerCreate(void);
void hypre_MemoryTrackerDestroy(hypre_MemoryTracker *tracker);
void hypre_MemoryTrackerInsert1(const char *action, void *ptr, size_t nbytes,
                                hypre_MemoryLocation memory_location, const char *filename,
                                const char *function, HYPRE_Int line);
void hypre_MemoryTrackerInsert2(const char *action, void *ptr, void *ptr2, size_t nbytes,
                                hypre_MemoryLocation memory_location,
                                hypre_MemoryLocation memory_location2,
                                const char *filename,
                                const char *function, HYPRE_Int line);
HYPRE_Int hypre_PrintMemoryTracker( size_t *totl_bytes_o, size_t *peak_bytes_o,
                                    size_t *curr_bytes_o, HYPRE_Int do_print, const char *fname );
HYPRE_Int hypre_MemoryTrackerSetPrint(HYPRE_Int do_print);
HYPRE_Int hypre_MemoryTrackerSetFileName(const char *file_name);
#endif

/* magma.c */
#if defined(HYPRE_USING_MAGMA)
HYPRE_Int hypre_MagmaInitialize(void);
HYPRE_Int hypre_MagmaFinalize(void);
#endif
