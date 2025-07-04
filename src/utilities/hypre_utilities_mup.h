
/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * Header file of multiprecision function prototypes.
 * This is needed for mixed-precision algorithm development.
 *****************************************************************************/

#ifndef HYPRE_UTILITIES_MUP_HEADER
#define HYPRE_UTILITIES_MUP_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined (HYPRE_MIXED_PRECISION)

hypre_LinkList hypre_create_elt_flt  ( HYPRE_Int Item );
hypre_LinkList hypre_create_elt_dbl  ( HYPRE_Int Item );
hypre_LinkList hypre_create_elt_long_dbl  ( HYPRE_Int Item );
void hypre_dispose_elt_flt  ( hypre_LinkList element_ptr );
void hypre_dispose_elt_dbl  ( hypre_LinkList element_ptr );
void hypre_dispose_elt_long_dbl  ( hypre_LinkList element_ptr );
void hypre_enter_on_lists_flt  ( hypre_LinkList *LoL_head_ptr, hypre_LinkList *LoL_tail_ptr,
                            HYPRE_Int measure, HYPRE_Int index, HYPRE_Int *lists, HYPRE_Int *where );
void hypre_enter_on_lists_dbl  ( hypre_LinkList *LoL_head_ptr, hypre_LinkList *LoL_tail_ptr,
                            HYPRE_Int measure, HYPRE_Int index, HYPRE_Int *lists, HYPRE_Int *where );
void hypre_enter_on_lists_long_dbl  ( hypre_LinkList *LoL_head_ptr, hypre_LinkList *LoL_tail_ptr,
                            HYPRE_Int measure, HYPRE_Int index, HYPRE_Int *lists, HYPRE_Int *where );
void hypre_remove_point_flt  ( hypre_LinkList *LoL_head_ptr, hypre_LinkList *LoL_tail_ptr,
                          HYPRE_Int measure, HYPRE_Int index, HYPRE_Int *lists, HYPRE_Int *where );
void hypre_remove_point_dbl  ( hypre_LinkList *LoL_head_ptr, hypre_LinkList *LoL_tail_ptr,
                          HYPRE_Int measure, HYPRE_Int index, HYPRE_Int *lists, HYPRE_Int *where );
void hypre_remove_point_long_dbl  ( hypre_LinkList *LoL_head_ptr, hypre_LinkList *LoL_tail_ptr,
                          HYPRE_Int measure, HYPRE_Int index, HYPRE_Int *lists, HYPRE_Int *where );
HYPRE_Int hypre_BigBinarySearch_flt  ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_Int hypre_BigBinarySearch_dbl  ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_Int hypre_BigBinarySearch_long_dbl  ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_BigInt *hypre_BigLowerBound_flt ( HYPRE_BigInt *first, HYPRE_BigInt *last, HYPRE_BigInt value );
HYPRE_BigInt *hypre_BigLowerBound_dbl ( HYPRE_BigInt *first, HYPRE_BigInt *last, HYPRE_BigInt value );
HYPRE_BigInt *hypre_BigLowerBound_long_dbl ( HYPRE_BigInt *first, HYPRE_BigInt *last, HYPRE_BigInt value );
HYPRE_Int hypre_BinarySearch_flt  ( HYPRE_Int *list, HYPRE_Int value, HYPRE_Int list_length );
HYPRE_Int hypre_BinarySearch_dbl  ( HYPRE_Int *list, HYPRE_Int value, HYPRE_Int list_length );
HYPRE_Int hypre_BinarySearch_long_dbl  ( HYPRE_Int *list, HYPRE_Int value, HYPRE_Int list_length );
HYPRE_Int hypre_BinarySearch2_flt  ( HYPRE_Int *list, HYPRE_Int value, HYPRE_Int low, HYPRE_Int high,
                                HYPRE_Int *spot );
HYPRE_Int hypre_BinarySearch2_dbl  ( HYPRE_Int *list, HYPRE_Int value, HYPRE_Int low, HYPRE_Int high,
                                HYPRE_Int *spot );
HYPRE_Int hypre_BinarySearch2_long_dbl  ( HYPRE_Int *list, HYPRE_Int value, HYPRE_Int low, HYPRE_Int high,
                                HYPRE_Int *spot );
HYPRE_Int *hypre_LowerBound_flt ( HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int value );
HYPRE_Int *hypre_LowerBound_dbl ( HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int value );
HYPRE_Int *hypre_LowerBound_long_dbl ( HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int value );
hypre_float    hypre_cabs_flt ( hypre_float value );
hypre_double    hypre_cabs_dbl ( hypre_double value );
hypre_long_double    hypre_cabs_long_dbl ( hypre_long_double value );
hypre_float    hypre_cimag_flt ( hypre_float value );
hypre_double    hypre_cimag_dbl ( hypre_double value );
hypre_long_double    hypre_cimag_long_dbl ( hypre_long_double value );
hypre_float hypre_conj_flt ( hypre_float value );
hypre_double hypre_conj_dbl ( hypre_double value );
hypre_long_double hypre_conj_long_dbl ( hypre_long_double value );
hypre_float    hypre_creal_flt ( hypre_float value );
hypre_double    hypre_creal_dbl ( hypre_double value );
hypre_long_double    hypre_creal_long_dbl ( hypre_long_double value );
hypre_float hypre_csqrt_flt ( hypre_float value );
hypre_double hypre_csqrt_dbl ( hypre_double value );
hypre_long_double hypre_csqrt_long_dbl ( hypre_long_double value );
HYPRE_Int hypre_bind_device_flt (HYPRE_Int myid, HYPRE_Int nproc, MPI_Comm comm);
HYPRE_Int hypre_bind_device_dbl (HYPRE_Int myid, HYPRE_Int nproc, MPI_Comm comm);
HYPRE_Int hypre_bind_device_long_dbl (HYPRE_Int myid, HYPRE_Int nproc, MPI_Comm comm);
void hypre_error_handler_flt (const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg);
void hypre_error_handler_dbl (const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg);
void hypre_error_handler_long_dbl (const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg);
HYPRE_Int hypre_CreateBinaryTree_flt (HYPRE_Int, HYPRE_Int, hypre_BinaryTree*);
HYPRE_Int hypre_CreateBinaryTree_dbl (HYPRE_Int, HYPRE_Int, hypre_BinaryTree*);
HYPRE_Int hypre_CreateBinaryTree_long_dbl (HYPRE_Int, HYPRE_Int, hypre_BinaryTree*);
HYPRE_Int hypre_DataExchangeList_flt (HYPRE_Int num_contacts, HYPRE_Int *contact_proc_list,
                                 void *contact_send_buf, HYPRE_Int *contact_send_buf_starts, HYPRE_Int contact_obj_size,
                                 HYPRE_Int response_obj_size, hypre_DataExchangeResponse *response_obj, HYPRE_Int max_response_size,
                                 HYPRE_Int rnum, MPI_Comm comm, void **p_response_recv_buf, HYPRE_Int **p_response_recv_buf_starts);
HYPRE_Int hypre_DataExchangeList_dbl (HYPRE_Int num_contacts, HYPRE_Int *contact_proc_list,
                                 void *contact_send_buf, HYPRE_Int *contact_send_buf_starts, HYPRE_Int contact_obj_size,
                                 HYPRE_Int response_obj_size, hypre_DataExchangeResponse *response_obj, HYPRE_Int max_response_size,
                                 HYPRE_Int rnum, MPI_Comm comm, void **p_response_recv_buf, HYPRE_Int **p_response_recv_buf_starts);
HYPRE_Int hypre_DataExchangeList_long_dbl (HYPRE_Int num_contacts, HYPRE_Int *contact_proc_list,
                                 void *contact_send_buf, HYPRE_Int *contact_send_buf_starts, HYPRE_Int contact_obj_size,
                                 HYPRE_Int response_obj_size, hypre_DataExchangeResponse *response_obj, HYPRE_Int max_response_size,
                                 HYPRE_Int rnum, MPI_Comm comm, void **p_response_recv_buf, HYPRE_Int **p_response_recv_buf_starts);
HYPRE_Int hypre_DestroyBinaryTree_flt (hypre_BinaryTree*);
HYPRE_Int hypre_DestroyBinaryTree_dbl (hypre_BinaryTree*);
HYPRE_Int hypre_DestroyBinaryTree_long_dbl (hypre_BinaryTree*);
HYPRE_Int hypre_GetDevice_flt (hypre_int *device_id);
HYPRE_Int hypre_GetDevice_dbl (hypre_int *device_id);
HYPRE_Int hypre_GetDevice_long_dbl (hypre_int *device_id);
HYPRE_Int hypre_GetDeviceCount_flt (hypre_int *device_count);
HYPRE_Int hypre_GetDeviceCount_dbl (hypre_int *device_count);
HYPRE_Int hypre_GetDeviceCount_long_dbl (hypre_int *device_count);
HYPRE_Int hypre_GetDeviceLastError_flt (void);
HYPRE_Int hypre_GetDeviceLastError_dbl (void);
HYPRE_Int hypre_GetDeviceLastError_long_dbl (void);

HYPRE_Int hypre_HandleDestroy_flt (hypre_Handle *hypre_handle_);
HYPRE_Int hypre_HandleDestroy_dbl (hypre_Handle *hypre_handle_);
HYPRE_Int hypre_HandleDestroy_long_dbl (hypre_Handle *hypre_handle_);
HYPRE_Int hypre_SetDevice_flt (hypre_int device_id, hypre_Handle *hypre_handle_);
HYPRE_Int hypre_SetDevice_dbl (hypre_int device_id, hypre_Handle *hypre_handle_);
HYPRE_Int hypre_SetDevice_long_dbl (hypre_int device_id, hypre_Handle *hypre_handle_);
HYPRE_Int hypre_SetGaussSeidelMethod_flt ( HYPRE_Int gs_method );
HYPRE_Int hypre_SetGaussSeidelMethod_dbl ( HYPRE_Int gs_method );
HYPRE_Int hypre_SetGaussSeidelMethod_long_dbl ( HYPRE_Int gs_method );
HYPRE_Int hypre_SetSpGemmAlgorithm_flt ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmAlgorithm_dbl ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmAlgorithm_long_dbl ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmBinned_flt ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmBinned_dbl ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmBinned_long_dbl ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateMethod_flt ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateMethod_dbl ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateMethod_long_dbl ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateMultFactor_flt ( hypre_float value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateMultFactor_dbl ( hypre_double value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateMultFactor_long_dbl ( hypre_long_double value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateNSamples_flt ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateNSamples_dbl ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmRownnzEstimateNSamples_long_dbl ( HYPRE_Int value );
HYPRE_Int hypre_SetSpGemmUseVendor_flt ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpGemmUseVendor_dbl ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpGemmUseVendor_long_dbl ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpMVUseVendor_flt ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpMVUseVendor_dbl ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpMVUseVendor_long_dbl ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpTransUseVendor_flt ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpTransUseVendor_dbl ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetSpTransUseVendor_long_dbl ( HYPRE_Int use_vendor );
HYPRE_Int hypre_SetUseGpuRand_flt ( HYPRE_Int use_gpurand );
HYPRE_Int hypre_SetUseGpuRand_dbl ( HYPRE_Int use_gpurand );
HYPRE_Int hypre_SetUseGpuRand_long_dbl ( HYPRE_Int use_gpurand );
HYPRE_Int hypre_SetUserDeviceMalloc_flt (GPUMallocFunc func);
HYPRE_Int hypre_SetUserDeviceMalloc_dbl (GPUMallocFunc func);
HYPRE_Int hypre_SetUserDeviceMalloc_long_dbl (GPUMallocFunc func);
HYPRE_Int hypre_SetUserDeviceMfree_flt (GPUMfreeFunc func);
HYPRE_Int hypre_SetUserDeviceMfree_dbl (GPUMfreeFunc func);
HYPRE_Int hypre_SetUserDeviceMfree_long_dbl (GPUMfreeFunc func);
void hypre_UnorderedBigIntMapCreate_flt ( hypre_UnorderedBigIntMap *m,
                                     HYPRE_Int inCapacity,
                                     HYPRE_Int concurrencyLevel);
void hypre_UnorderedBigIntMapCreate_dbl ( hypre_UnorderedBigIntMap *m,
                                     HYPRE_Int inCapacity,
                                     HYPRE_Int concurrencyLevel);
void hypre_UnorderedBigIntMapCreate_long_dbl ( hypre_UnorderedBigIntMap *m,
                                     HYPRE_Int inCapacity,
                                     HYPRE_Int concurrencyLevel);
void hypre_UnorderedBigIntMapDestroy_flt ( hypre_UnorderedBigIntMap *m );
void hypre_UnorderedBigIntMapDestroy_dbl ( hypre_UnorderedBigIntMap *m );
void hypre_UnorderedBigIntMapDestroy_long_dbl ( hypre_UnorderedBigIntMap *m );
HYPRE_BigInt *hypre_UnorderedBigIntSetCopyToArray_flt ( hypre_UnorderedBigIntSet *s, HYPRE_Int *len );
HYPRE_BigInt *hypre_UnorderedBigIntSetCopyToArray_dbl ( hypre_UnorderedBigIntSet *s, HYPRE_Int *len );
HYPRE_BigInt *hypre_UnorderedBigIntSetCopyToArray_long_dbl ( hypre_UnorderedBigIntSet *s, HYPRE_Int *len );
void hypre_UnorderedBigIntSetCreate_flt ( hypre_UnorderedBigIntSet *s,
                                     HYPRE_Int inCapacity,
                                     HYPRE_Int concurrencyLevel);
void hypre_UnorderedBigIntSetCreate_dbl ( hypre_UnorderedBigIntSet *s,
                                     HYPRE_Int inCapacity,
                                     HYPRE_Int concurrencyLevel);
void hypre_UnorderedBigIntSetCreate_long_dbl ( hypre_UnorderedBigIntSet *s,
                                     HYPRE_Int inCapacity,
                                     HYPRE_Int concurrencyLevel);
void hypre_UnorderedBigIntSetDestroy_flt ( hypre_UnorderedBigIntSet *s );
void hypre_UnorderedBigIntSetDestroy_dbl ( hypre_UnorderedBigIntSet *s );
void hypre_UnorderedBigIntSetDestroy_long_dbl ( hypre_UnorderedBigIntSet *s );
void hypre_UnorderedIntMapCreate_flt ( hypre_UnorderedIntMap *m,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedIntMapCreate_dbl ( hypre_UnorderedIntMap *m,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedIntMapCreate_long_dbl ( hypre_UnorderedIntMap *m,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedIntMapDestroy_flt ( hypre_UnorderedIntMap *m );
void hypre_UnorderedIntMapDestroy_dbl ( hypre_UnorderedIntMap *m );
void hypre_UnorderedIntMapDestroy_long_dbl ( hypre_UnorderedIntMap *m );
HYPRE_Int *hypre_UnorderedIntSetCopyToArray_flt ( hypre_UnorderedIntSet *s, HYPRE_Int *len );
HYPRE_Int *hypre_UnorderedIntSetCopyToArray_dbl ( hypre_UnorderedIntSet *s, HYPRE_Int *len );
HYPRE_Int *hypre_UnorderedIntSetCopyToArray_long_dbl ( hypre_UnorderedIntSet *s, HYPRE_Int *len );
void hypre_UnorderedIntSetCreate_flt ( hypre_UnorderedIntSet *s,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedIntSetCreate_dbl ( hypre_UnorderedIntSet *s,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedIntSetCreate_long_dbl ( hypre_UnorderedIntSet *s,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedIntSetDestroy_flt ( hypre_UnorderedIntSet *s );
void hypre_UnorderedIntSetDestroy_dbl ( hypre_UnorderedIntSet *s );
void hypre_UnorderedIntSetDestroy_long_dbl ( hypre_UnorderedIntSet *s );
hypre_IntArray* hypre_IntArrayCloneDeep_flt ( hypre_IntArray *x );
hypre_IntArray* hypre_IntArrayCloneDeep_dbl ( hypre_IntArray *x );
hypre_IntArray* hypre_IntArrayCloneDeep_long_dbl ( hypre_IntArray *x );
hypre_IntArray* hypre_IntArrayCloneDeep_v2_flt ( hypre_IntArray *x,
                                            HYPRE_MemoryLocation memory_location );
hypre_IntArray* hypre_IntArrayCloneDeep_v2_dbl ( hypre_IntArray *x,
                                            HYPRE_MemoryLocation memory_location );
hypre_IntArray* hypre_IntArrayCloneDeep_v2_long_dbl ( hypre_IntArray *x,
                                            HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayCopy_flt ( hypre_IntArray *x, hypre_IntArray *y );
HYPRE_Int hypre_IntArrayCopy_dbl ( hypre_IntArray *x, hypre_IntArray *y );
HYPRE_Int hypre_IntArrayCopy_long_dbl ( hypre_IntArray *x, hypre_IntArray *y );
HYPRE_Int hypre_IntArrayCount_flt ( hypre_IntArray *v, HYPRE_Int value,
                               HYPRE_Int *num_values_ptr );
HYPRE_Int hypre_IntArrayCount_dbl ( hypre_IntArray *v, HYPRE_Int value,
                               HYPRE_Int *num_values_ptr );
HYPRE_Int hypre_IntArrayCount_long_dbl ( hypre_IntArray *v, HYPRE_Int value,
                               HYPRE_Int *num_values_ptr );
HYPRE_Int hypre_IntArrayCountHost_flt ( hypre_IntArray *v, HYPRE_Int value,
                                   HYPRE_Int *num_values_ptr );
HYPRE_Int hypre_IntArrayCountHost_dbl ( hypre_IntArray *v, HYPRE_Int value,
                                   HYPRE_Int *num_values_ptr );
HYPRE_Int hypre_IntArrayCountHost_long_dbl ( hypre_IntArray *v, HYPRE_Int value,
                                   HYPRE_Int *num_values_ptr );
hypre_IntArray* hypre_IntArrayCreate_flt ( HYPRE_Int size );
hypre_IntArray* hypre_IntArrayCreate_dbl ( HYPRE_Int size );
hypre_IntArray* hypre_IntArrayCreate_long_dbl ( HYPRE_Int size );
HYPRE_Int hypre_IntArrayDestroy_flt ( hypre_IntArray *array );
HYPRE_Int hypre_IntArrayDestroy_dbl ( hypre_IntArray *array );
HYPRE_Int hypre_IntArrayDestroy_long_dbl ( hypre_IntArray *array );
HYPRE_Int hypre_IntArrayInitialize_flt ( hypre_IntArray *array );
HYPRE_Int hypre_IntArrayInitialize_dbl ( hypre_IntArray *array );
HYPRE_Int hypre_IntArrayInitialize_long_dbl ( hypre_IntArray *array );
HYPRE_Int hypre_IntArrayInitialize_v2_flt ( hypre_IntArray *array,
                                       HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayInitialize_v2_dbl ( hypre_IntArray *array,
                                       HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayInitialize_v2_long_dbl ( hypre_IntArray *array,
                                       HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayInverseMapping_flt ( hypre_IntArray *v, hypre_IntArray **w_ptr );
HYPRE_Int hypre_IntArrayInverseMapping_dbl ( hypre_IntArray *v, hypre_IntArray **w_ptr );
HYPRE_Int hypre_IntArrayInverseMapping_long_dbl ( hypre_IntArray *v, hypre_IntArray **w_ptr );
HYPRE_Int hypre_IntArrayMigrate_flt ( hypre_IntArray *v, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayMigrate_dbl ( hypre_IntArray *v, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayMigrate_long_dbl ( hypre_IntArray *v, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_IntArrayNegate_flt ( hypre_IntArray *v );
HYPRE_Int hypre_IntArrayNegate_dbl ( hypre_IntArray *v );
HYPRE_Int hypre_IntArrayNegate_long_dbl ( hypre_IntArray *v );
HYPRE_Int hypre_IntArrayPrint_flt ( MPI_Comm comm, hypre_IntArray *array, const char *filename );
HYPRE_Int hypre_IntArrayPrint_dbl ( MPI_Comm comm, hypre_IntArray *array, const char *filename );
HYPRE_Int hypre_IntArrayPrint_long_dbl ( MPI_Comm comm, hypre_IntArray *array, const char *filename );
HYPRE_Int hypre_IntArrayRead_flt ( MPI_Comm comm, const char *filename, hypre_IntArray **array_ptr );
HYPRE_Int hypre_IntArrayRead_dbl ( MPI_Comm comm, const char *filename, hypre_IntArray **array_ptr );
HYPRE_Int hypre_IntArrayRead_long_dbl ( MPI_Comm comm, const char *filename, hypre_IntArray **array_ptr );
HYPRE_Int hypre_IntArraySetConstantValues_flt ( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_IntArraySetConstantValues_dbl ( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_IntArraySetConstantValues_long_dbl ( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_IntArraySetConstantValuesHost_flt ( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_IntArraySetConstantValuesHost_dbl ( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_IntArraySetConstantValuesHost_long_dbl ( hypre_IntArray *v, HYPRE_Int value );
HYPRE_Int hypre_Log2_flt ( HYPRE_Int p );
HYPRE_Int hypre_Log2_dbl ( HYPRE_Int p );
HYPRE_Int hypre_Log2_long_dbl ( HYPRE_Int p );
/*--------------------------------------------------------------------------
 *Prototypes
 *--------------------------------------------------------------------------*/

/* memory.c */
HYPRE_Int hypre_GetMemoryLocationName(hypre_MemoryLocation memory_location,
                                      char *memory_location_name);
HYPRE_Int hypre_SetCubMemPoolSize_flt ( hypre_uint bin_growth, hypre_uint min_bin, hypre_uint max_bin,
                                   size_t max_cached_bytes );
HYPRE_Int hypre_SetCubMemPoolSize_dbl ( hypre_uint bin_growth, hypre_uint min_bin, hypre_uint max_bin,
                                   size_t max_cached_bytes );
HYPRE_Int hypre_SetCubMemPoolSize_long_dbl ( hypre_uint bin_growth, hypre_uint min_bin, hypre_uint max_bin,
                                   size_t max_cached_bytes );
void hypre_big_merge_sort_flt (HYPRE_BigInt *in, HYPRE_BigInt *temp, HYPRE_Int len,
                          HYPRE_BigInt **sorted);
void hypre_big_merge_sort_dbl (HYPRE_BigInt *in, HYPRE_BigInt *temp, HYPRE_Int len,
                          HYPRE_BigInt **sorted);
void hypre_big_merge_sort_long_dbl (HYPRE_BigInt *in, HYPRE_BigInt *temp, HYPRE_Int len,
                          HYPRE_BigInt **sorted);
void hypre_big_sort_and_create_inverse_map_flt (HYPRE_BigInt *in, HYPRE_Int len, HYPRE_BigInt **out,
                                           hypre_UnorderedBigIntMap *inverse_map);
void hypre_big_sort_and_create_inverse_map_dbl (HYPRE_BigInt *in, HYPRE_Int len, HYPRE_BigInt **out,
                                           hypre_UnorderedBigIntMap *inverse_map);
void hypre_big_sort_and_create_inverse_map_long_dbl (HYPRE_BigInt *in, HYPRE_Int len, HYPRE_BigInt **out,
                                           hypre_UnorderedBigIntMap *inverse_map);
HYPRE_Int hypre_IntArrayMergeOrdered_flt ( hypre_IntArray *array1, hypre_IntArray *array2,
                                      hypre_IntArray *array3 );
HYPRE_Int hypre_IntArrayMergeOrdered_dbl ( hypre_IntArray *array1, hypre_IntArray *array2,
                                      hypre_IntArray *array3 );
HYPRE_Int hypre_IntArrayMergeOrdered_long_dbl ( hypre_IntArray *array1, hypre_IntArray *array2,
                                      hypre_IntArray *array3 );
void hypre_merge_sort_flt (HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **sorted);
void hypre_merge_sort_dbl (HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **sorted);
void hypre_merge_sort_long_dbl (HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **sorted);
void hypre_sort_and_create_inverse_map_flt (HYPRE_Int *in, HYPRE_Int len, HYPRE_Int **out,
                                       hypre_UnorderedIntMap *inverse_map);
void hypre_sort_and_create_inverse_map_dbl (HYPRE_Int *in, HYPRE_Int len, HYPRE_Int **out,
                                       hypre_UnorderedIntMap *inverse_map);
void hypre_sort_and_create_inverse_map_long_dbl (HYPRE_Int *in, HYPRE_Int len, HYPRE_Int **out,
                                       hypre_UnorderedIntMap *inverse_map);
void hypre_union2_flt (HYPRE_Int n1, HYPRE_BigInt *arr1, HYPRE_Int n2, HYPRE_BigInt *arr2, HYPRE_Int *n3,
                  HYPRE_BigInt *arr3, HYPRE_Int *map1, HYPRE_Int *map2);
void hypre_union2_dbl (HYPRE_Int n1, HYPRE_BigInt *arr1, HYPRE_Int n2, HYPRE_BigInt *arr2, HYPRE_Int *n3,
                  HYPRE_BigInt *arr3, HYPRE_Int *map1, HYPRE_Int *map2);
void hypre_union2_long_dbl (HYPRE_Int n1, HYPRE_BigInt *arr1, HYPRE_Int n2, HYPRE_BigInt *arr2, HYPRE_Int *n3,
                  HYPRE_BigInt *arr3, HYPRE_Int *map1, HYPRE_Int *map2);
HYPRE_Int hypre_mm_is_valid_flt (MM_typecode matcode); /* too complex for a macro */
HYPRE_Int hypre_mm_read_banner(FILE *f, MM_typecode *matcode);
HYPRE_Int hypre_mm_is_valid_dbl (MM_typecode matcode); /* too complex for a macro */
HYPRE_Int hypre_mm_read_banner(FILE *f, MM_typecode *matcode);
HYPRE_Int hypre_mm_is_valid_long_dbl (MM_typecode matcode); /* too complex for a macro */
HYPRE_Int hypre_mm_read_banner(FILE *f, MM_typecode *matcode);
HYPRE_Int hypre_mm_read_mtx_crd_size_flt (FILE *f, HYPRE_Int *M, HYPRE_Int *N, HYPRE_Int *nz);
HYPRE_Int hypre_mm_read_mtx_crd_size_dbl (FILE *f, HYPRE_Int *M, HYPRE_Int *N, HYPRE_Int *nz);
HYPRE_Int hypre_mm_read_mtx_crd_size_long_dbl (FILE *f, HYPRE_Int *M, HYPRE_Int *N, HYPRE_Int *nz);
void hypre_GpuProfilingPopRange_flt (void);
void hypre_GpuProfilingPopRange_dbl (void);
void hypre_GpuProfilingPopRange_long_dbl (void);
void hypre_GpuProfilingPushRange_flt (const char *name);
void hypre_GpuProfilingPushRange_dbl (const char *name);
void hypre_GpuProfilingPushRange_long_dbl (const char *name);
void hypre_GpuProfilingPushRangeColor_flt (const char *name, HYPRE_Int cid);
void hypre_GpuProfilingPushRangeColor_dbl (const char *name, HYPRE_Int cid);
void hypre_GpuProfilingPushRangeColor_long_dbl (const char *name, HYPRE_Int cid);
void hypre_prefix_sum_flt (HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int *workspace);
void hypre_prefix_sum_dbl (HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int *workspace);
void hypre_prefix_sum_long_dbl (HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int *workspace);
void hypre_prefix_sum_multiple_flt (HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int n,
                               HYPRE_Int *workspace);
void hypre_prefix_sum_multiple_dbl (HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int n,
                               HYPRE_Int *workspace);
void hypre_prefix_sum_multiple_long_dbl (HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int n,
                               HYPRE_Int *workspace);
void hypre_prefix_sum_pair_flt (HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2,
                           HYPRE_Int *workspace);
void hypre_prefix_sum_pair_dbl (HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2,
                           HYPRE_Int *workspace);
void hypre_prefix_sum_pair_long_dbl (HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2,
                           HYPRE_Int *workspace);
void hypre_prefix_sum_triple_flt (HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2,
                             HYPRE_Int *sum2, HYPRE_Int *in_out3, HYPRE_Int *sum3, HYPRE_Int *workspace);
void hypre_prefix_sum_triple_dbl (HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2,
                             HYPRE_Int *sum2, HYPRE_Int *in_out3, HYPRE_Int *sum3, HYPRE_Int *workspace);
void hypre_prefix_sum_triple_long_dbl (HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2,
                             HYPRE_Int *sum2, HYPRE_Int *in_out3, HYPRE_Int *sum3, HYPRE_Int *workspace);
HYPRE_Int hypre_fprintf_flt ( FILE *stream, const char *format, ... );
HYPRE_Int hypre_fprintf_dbl ( FILE *stream, const char *format, ... );
HYPRE_Int hypre_fprintf_long_dbl ( FILE *stream, const char *format, ... );
HYPRE_Int hypre_fscanf_flt ( FILE *stream, const char *format, ... );
HYPRE_Int hypre_fscanf_dbl ( FILE *stream, const char *format, ... );
HYPRE_Int hypre_fscanf_long_dbl ( FILE *stream, const char *format, ... );
HYPRE_Int hypre_ndigits_flt ( HYPRE_BigInt number );
HYPRE_Int hypre_ndigits_dbl ( HYPRE_BigInt number );
HYPRE_Int hypre_ndigits_long_dbl ( HYPRE_BigInt number );
HYPRE_Int hypre_ParPrintf_flt (MPI_Comm comm, const char *format, ...);
HYPRE_Int hypre_ParPrintf_dbl (MPI_Comm comm, const char *format, ...);
HYPRE_Int hypre_ParPrintf_long_dbl (MPI_Comm comm, const char *format, ...);
HYPRE_Int hypre_printf_flt ( const char *format, ... );
HYPRE_Int hypre_printf_dbl ( const char *format, ... );
HYPRE_Int hypre_printf_long_dbl ( const char *format, ... );
HYPRE_Int hypre_scanf_flt ( const char *format, ... );
HYPRE_Int hypre_scanf_dbl ( const char *format, ... );
HYPRE_Int hypre_scanf_long_dbl ( const char *format, ... );
HYPRE_Int hypre_snprintf_flt ( char *s, size_t size, const char *format, ...);
HYPRE_Int hypre_snprintf_dbl ( char *s, size_t size, const char *format, ...);
HYPRE_Int hypre_snprintf_long_dbl ( char *s, size_t size, const char *format, ...);
HYPRE_Int hypre_sprintf_flt ( char *s, const char *format, ... );
HYPRE_Int hypre_sprintf_dbl ( char *s, const char *format, ... );
HYPRE_Int hypre_sprintf_long_dbl ( char *s, const char *format, ... );
HYPRE_Int hypre_sscanf_flt ( char *s, const char *format, ... );
HYPRE_Int hypre_sscanf_dbl ( char *s, const char *format, ... );
HYPRE_Int hypre_sscanf_long_dbl ( char *s, const char *format, ... );
void hypre_BigQsort0_flt ( HYPRE_BigInt *v, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsort0_dbl ( HYPRE_BigInt *v, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsort0_long_dbl ( HYPRE_BigInt *v, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsort1_flt  ( HYPRE_BigInt *v, hypre_float *w, HYPRE_Int left, HYPRE_Int right );
void hypre_BigQsort1_dbl  ( HYPRE_BigInt *v, hypre_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_BigQsort1_long_dbl  ( HYPRE_BigInt *v, hypre_long_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_BigQsort2i_flt ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsort2i_dbl ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsort2i_long_dbl ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsort4_abs_flt  ( hypre_float *v, HYPRE_BigInt *w, HYPRE_Int *z, HYPRE_Int *y,
                           HYPRE_Int left, HYPRE_Int right );
void hypre_BigQsort4_abs_dbl  ( hypre_double *v, HYPRE_BigInt *w, HYPRE_Int *z, HYPRE_Int *y,
                           HYPRE_Int left, HYPRE_Int right );
void hypre_BigQsort4_abs_long_dbl  ( hypre_long_double *v, HYPRE_BigInt *w, HYPRE_Int *z, HYPRE_Int *y,
                           HYPRE_Int left, HYPRE_Int right );
void hypre_BigQsortb2i_flt ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left,
                        HYPRE_Int  right );
void hypre_BigQsortb2i_dbl ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left,
                        HYPRE_Int  right );
void hypre_BigQsortb2i_long_dbl ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left,
                        HYPRE_Int  right );
void hypre_BigQsortbi_flt ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsortbi_dbl ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsortbi_long_dbl ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsortbLoc_flt ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsortbLoc_dbl ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigQsortbLoc_long_dbl ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int  left, HYPRE_Int  right );
void hypre_BigSwap_flt ( HYPRE_BigInt *v, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwap_dbl ( HYPRE_BigInt *v, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwap_long_dbl ( HYPRE_BigInt *v, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwap2_flt  ( HYPRE_BigInt *v, hypre_float *w, HYPRE_Int i, HYPRE_Int j );
void hypre_BigSwap2_dbl  ( HYPRE_BigInt *v, hypre_double *w, HYPRE_Int i, HYPRE_Int j );
void hypre_BigSwap2_long_dbl  ( HYPRE_BigInt *v, hypre_long_double *w, HYPRE_Int i, HYPRE_Int j );
void hypre_BigSwap2i_flt  ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_BigSwap2i_dbl  ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_BigSwap2i_long_dbl  ( HYPRE_BigInt *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_BigSwap4_d_flt  ( hypre_float *v, HYPRE_BigInt *w, HYPRE_Int *z, HYPRE_Int *y, HYPRE_Int i,
                        HYPRE_Int j );
void hypre_BigSwap4_d_dbl  ( hypre_double *v, HYPRE_BigInt *w, HYPRE_Int *z, HYPRE_Int *y, HYPRE_Int i,
                        HYPRE_Int j );
void hypre_BigSwap4_d_long_dbl  ( hypre_long_double *v, HYPRE_BigInt *w, HYPRE_Int *z, HYPRE_Int *y, HYPRE_Int i,
                        HYPRE_Int j );
void hypre_BigSwapb2i_flt (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwapb2i_dbl (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwapb2i_long_dbl (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwapbi_flt (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwapbi_dbl (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwapbi_long_dbl (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwapLoc_flt (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwapLoc_dbl (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_BigSwapLoc_long_dbl (HYPRE_BigInt  *v, HYPRE_Int  *w, HYPRE_Int  i, HYPRE_Int  j );
void hypre_dense_topo_sort_flt (const hypre_float *L, HYPRE_Int *ordering, HYPRE_Int n,
                           HYPRE_Int is_col_major);
void hypre_dense_topo_sort_dbl (const hypre_double *L, HYPRE_Int *ordering, HYPRE_Int n,
                           HYPRE_Int is_col_major);
void hypre_dense_topo_sort_long_dbl (const hypre_long_double *L, HYPRE_Int *ordering, HYPRE_Int n,
                           HYPRE_Int is_col_major);
void hypre_qsort0_flt  ( HYPRE_Int *v, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort0_dbl  ( HYPRE_Int *v, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort0_long_dbl  ( HYPRE_Int *v, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort1_flt  ( HYPRE_Int *v, hypre_float *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort1_dbl  ( HYPRE_Int *v, hypre_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort1_long_dbl  ( HYPRE_Int *v, hypre_long_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_flt  ( HYPRE_Int *v, hypre_float *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_dbl  ( HYPRE_Int *v, hypre_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_long_dbl  ( HYPRE_Int *v, hypre_long_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_abs_flt  ( HYPRE_Int *v, hypre_float *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_abs_dbl  ( HYPRE_Int *v, hypre_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_abs_long_dbl  ( HYPRE_Int *v, hypre_long_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2i_flt  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2i_dbl  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2i_long_dbl  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3_flt ( hypre_float *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left, HYPRE_Int  right );
void hypre_qsort3_dbl ( hypre_double *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left, HYPRE_Int  right );
void hypre_qsort3_long_dbl ( hypre_long_double *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int  left, HYPRE_Int  right );
void hypre_qsort3_abs_flt  ( hypre_float *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int left,
                        HYPRE_Int right );
void hypre_qsort3_abs_dbl  ( hypre_double *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int left,
                        HYPRE_Int right );
void hypre_qsort3_abs_long_dbl  ( hypre_long_double *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int left,
                        HYPRE_Int right );
void hypre_qsort3i_flt  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3i_dbl  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3i_long_dbl  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3ir_flt  ( HYPRE_Int *v, hypre_float *w, HYPRE_Int *z, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3ir_dbl  ( HYPRE_Int *v, hypre_double *w, HYPRE_Int *z, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort3ir_long_dbl  ( HYPRE_Int *v, hypre_long_double *w, HYPRE_Int *z, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort_abs_flt  ( hypre_float *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort_abs_dbl  ( hypre_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort_abs_long_dbl  ( hypre_long_double *w, HYPRE_Int left, HYPRE_Int right );
void hypre_swap_flt  ( HYPRE_Int *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_dbl  ( HYPRE_Int *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_long_dbl  ( HYPRE_Int *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2_flt  ( HYPRE_Int *v, hypre_float *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2_dbl  ( HYPRE_Int *v, hypre_double *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2_long_dbl  ( HYPRE_Int *v, hypre_long_double *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2i_flt  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2i_dbl  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2i_long_dbl  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3_d_flt  ( hypre_float *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3_d_dbl  ( hypre_double *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3_d_long_dbl  ( hypre_long_double *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3_d_perm_flt (HYPRE_Int  *v, hypre_float  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_swap3_d_perm_dbl (HYPRE_Int  *v, hypre_double  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_swap3_d_perm_long_dbl (HYPRE_Int  *v, hypre_long_double  *w, HYPRE_Int  *z, HYPRE_Int  i, HYPRE_Int  j );
void hypre_swap3i_flt  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3i_dbl  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int i, HYPRE_Int j );
void hypre_swap3i_long_dbl  ( HYPRE_Int *v, HYPRE_Int *w, HYPRE_Int *z, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_c_flt  ( hypre_float *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_c_dbl  ( hypre_double *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_c_long_dbl  ( hypre_long_double *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_d_flt  ( hypre_float *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_d_dbl  ( hypre_double *v, HYPRE_Int i, HYPRE_Int j );
void hypre_swap_d_long_dbl  ( hypre_long_double *v, HYPRE_Int i, HYPRE_Int j );
void hypre_topo_sort_flt (const HYPRE_Int *row_ptr, const HYPRE_Int *col_inds, const hypre_float *data,
                     HYPRE_Int *ordering, HYPRE_Int n);
void hypre_topo_sort_dbl (const HYPRE_Int *row_ptr, const HYPRE_Int *col_inds, const hypre_double *data,
                     HYPRE_Int *ordering, HYPRE_Int n);
void hypre_topo_sort_long_dbl (const HYPRE_Int *row_ptr, const HYPRE_Int *col_inds, const hypre_long_double *data,
                     HYPRE_Int *ordering, HYPRE_Int n);
HYPRE_Int hypre_DoubleQuickSplit_flt  ( hypre_float *values, HYPRE_Int *indices, HYPRE_Int list_length,
                                   HYPRE_Int NumberKept );
HYPRE_Int hypre_DoubleQuickSplit_dbl  ( hypre_double *values, HYPRE_Int *indices, HYPRE_Int list_length,
                                   HYPRE_Int NumberKept );
HYPRE_Int hypre_DoubleQuickSplit_long_dbl  ( hypre_long_double *values, HYPRE_Int *indices, HYPRE_Int list_length,
                                   HYPRE_Int NumberKept );
/* HYPRE_CUDA_GLOBAL_flt  */ hypre_float hypre_Rand_flt ( void );
/* HYPRE_CUDA_GLOBAL_dbl  */ hypre_double hypre_Rand_dbl ( void );
/* HYPRE_CUDA_GLOBAL_long_dbl  */ hypre_long_double hypre_Rand_long_dbl ( void );
/* HYPRE_CUDA_GLOBAL_flt  */ HYPRE_Int hypre_RandI_flt ( void );
/* HYPRE_CUDA_GLOBAL_dbl  */ HYPRE_Int hypre_RandI_dbl ( void );
/* HYPRE_CUDA_GLOBAL_long_dbl  */ HYPRE_Int hypre_RandI_long_dbl ( void );
/* HYPRE_CUDA_GLOBAL_flt  */ void hypre_SeedRand_flt ( HYPRE_Int seed );
/* HYPRE_CUDA_GLOBAL_dbl  */ void hypre_SeedRand_dbl ( HYPRE_Int seed );
/* HYPRE_CUDA_GLOBAL_long_dbl  */ void hypre_SeedRand_long_dbl ( HYPRE_Int seed );
HYPRE_Int HYPRE_Initialize_flt(void);
HYPRE_Int HYPRE_Initialize_dbl(void);
HYPRE_Int HYPRE_Initialize_long_dbl(void);
HYPRE_Int HYPRE_Initialized_flt(void);
HYPRE_Int HYPRE_Initialized_dbl(void);
HYPRE_Int HYPRE_Initialized_long_dbl(void);
HYPRE_Int HYPRE_Finalize_flt(void);
HYPRE_Int HYPRE_Finalize_dbl(void);
HYPRE_Int HYPRE_Finalize_long_dbl(void);
HYPRE_Int HYPRE_Finalized_flt(void);
HYPRE_Int HYPRE_Finalized_dbl(void);
HYPRE_Int HYPRE_Finalized_long_dbl(void);
HYPRE_Int hypre_Finalized_flt ( void );
HYPRE_Int hypre_Finalized_dbl ( void );
HYPRE_Int hypre_Finalized_long_dbl ( void );
HYPRE_Int hypre_Initialized_flt ( void );
HYPRE_Int hypre_Initialized_dbl ( void );
HYPRE_Int hypre_Initialized_long_dbl ( void );
HYPRE_Int hypre_SetFinalized_flt ( void );
HYPRE_Int hypre_SetFinalized_dbl ( void );
HYPRE_Int hypre_SetFinalized_long_dbl ( void );
HYPRE_Int hypre_SetInitialized_flt ( void );
HYPRE_Int hypre_SetInitialized_dbl ( void );
HYPRE_Int hypre_SetInitialized_long_dbl ( void );
void hypre_GetSimpleThreadPartition_flt ( HYPRE_Int *begin, HYPRE_Int *end, HYPRE_Int n );
void hypre_GetSimpleThreadPartition_dbl ( HYPRE_Int *begin, HYPRE_Int *end, HYPRE_Int n );
void hypre_GetSimpleThreadPartition_long_dbl ( HYPRE_Int *begin, HYPRE_Int *end, HYPRE_Int n );
hypre_double time_getCPUSeconds_flt ( void );
hypre_double time_getCPUSeconds_dbl ( void );
hypre_double time_getCPUSeconds_long_dbl ( void );
hypre_double time_getWallclockSeconds_flt ( void );
hypre_double time_getWallclockSeconds_dbl ( void );
hypre_double time_getWallclockSeconds_long_dbl ( void );

#if defined (HYPRE_TIMING)
/* timing.c */
HYPRE_Int hypre_InitializeTiming_flt( const char *name );
HYPRE_Int hypre_InitializeTiming_dbl( const char *name );
HYPRE_Int hypre_InitializeTiming_long_dbl( const char *name );
HYPRE_Int hypre_FinalizeTiming_flt( HYPRE_Int time_index );
HYPRE_Int hypre_FinalizeTiming_dbl( HYPRE_Int time_index );
HYPRE_Int hypre_FinalizeTiming_long_dbl( HYPRE_Int time_index );
HYPRE_Int hypre_FinalizeAllTimings_flt( void );
HYPRE_Int hypre_FinalizeAllTimings_dbl( void );
HYPRE_Int hypre_FinalizeAllTimings_long_dbl( void );
HYPRE_Int hypre_IncFLOPCount_flt( HYPRE_BigInt inc );
HYPRE_Int hypre_IncFLOPCount_dbl( HYPRE_BigInt inc );
HYPRE_Int hypre_IncFLOPCount_long_dbl( HYPRE_BigInt inc );
HYPRE_Int hypre_BeginTiming_flt( HYPRE_Int time_index );
HYPRE_Int hypre_BeginTiming_dbl( HYPRE_Int time_index );
HYPRE_Int hypre_BeginTiming_long_dbl( HYPRE_Int time_index );
HYPRE_Int hypre_EndTiming_flt( HYPRE_Int time_index );
HYPRE_Int hypre_EndTiming_dbl( HYPRE_Int time_index );
HYPRE_Int hypre_EndTiming_long_dbl( HYPRE_Int time_index );
HYPRE_Int hypre_ClearTiming_flt( void );
HYPRE_Int hypre_ClearTiming_dbl( void );
HYPRE_Int hypre_ClearTiming_long_dbl( void );
HYPRE_Int hypre_PrintTiming_flt( const char *heading, MPI_Comm comm );
HYPRE_Int hypre_PrintTiming_dbl( const char *heading, MPI_Comm comm );
HYPRE_Int hypre_PrintTiming_long_dbl( const char *heading, MPI_Comm comm );
HYPRE_Int hypre_GetTiming_flt( const char *heading, hypre_double *wall_time_ptr, MPI_Comm comm );
HYPRE_Int hypre_GetTiming_dbl( const char *heading, hypre_double *wall_time_ptr, MPI_Comm comm );
HYPRE_Int hypre_GetTiming_long_dbl( const char *heading, hypre_double *wall_time_ptr, MPI_Comm comm );
HYPRE_Int hypre_multmod_flt (HYPRE_Int a, HYPRE_Int b, HYPRE_Int mod);
HYPRE_Int hypre_multmod_dbl (HYPRE_Int a, HYPRE_Int b, HYPRE_Int mod);
HYPRE_Int hypre_multmod_long_dbl (HYPRE_Int a, HYPRE_Int b, HYPRE_Int mod);
void hypre_partition1D_flt (HYPRE_Int n, HYPRE_Int p, HYPRE_Int j, HYPRE_Int *s, HYPRE_Int *e);
void hypre_partition1D_dbl (HYPRE_Int n, HYPRE_Int p, HYPRE_Int j, HYPRE_Int *s, HYPRE_Int *e);
void hypre_partition1D_long_dbl (HYPRE_Int n, HYPRE_Int p, HYPRE_Int j, HYPRE_Int *s, HYPRE_Int *e);
char *hypre_strcpy_flt (char *destination, const char *source);
char *hypre_strcpy_dbl (char *destination, const char *source);
char *hypre_strcpy_long_dbl (char *destination, const char *source);
#endif

#endif

#ifdef __cplusplus
}
#endif

#endif
