/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"

#if defined(HYPRE_USING_GPU)

#include "csr_spgemm_device.h"

HYPRE_Int hypreDevice_CSRSpGemmBinnedGetBlockNumDim()
{
   hypre_int multiProcessorCount = 0;
   /* bins 1, 2, ..., num_bins, are effective; 0 is reserved for empty rows */
   const HYPRE_Int num_bins = 10;

   hypre_HandleSpgemmNumBin(hypre_handle()) = num_bins;

#if defined(HYPRE_USING_CUDA)
   cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount,
                          hypre_HandleDevice(hypre_handle()));
#endif

#if defined(HYPRE_USING_HIP)
   hipDeviceGetAttribute(&multiProcessorCount, hipDeviceAttributeMultiprocessorCount,
                         hypre_HandleDevice(hypre_handle()));
#endif

#if defined(HYPRE_USING_SYCL)
   /* WM: todo - is this right? */
   multiProcessorCount = hypre_HandleDevice(
                            hypre_handle())->get_info<sycl::info::device::max_compute_units>();
#endif

   typedef HYPRE_Int arrType[4][HYPRE_SPGEMM_MAX_NBIN + 1];
   arrType &max_nblocks = hypre_HandleSpgemmBlockNumDim(hypre_handle());

   for (HYPRE_Int i = 0; i < num_bins + 1; i++)
   {
      max_nblocks[0][i] = max_nblocks[1][i] = max_nblocks[2][i] = max_nblocks[3][i] = 0;
   }

   /* symbolic */
   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[1], T_GROUP_SIZE[1] >
   (multiProcessorCount, &max_nblocks[0][1], &max_nblocks[2][1]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[2], T_GROUP_SIZE[2] >
   (multiProcessorCount, &max_nblocks[0][2], &max_nblocks[2][2]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[3], T_GROUP_SIZE[3] >
   (multiProcessorCount, &max_nblocks[0][3], &max_nblocks[2][3]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[4], T_GROUP_SIZE[4] >
   (multiProcessorCount, &max_nblocks[0][4], &max_nblocks[2][4]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[5], T_GROUP_SIZE[5] >
   (multiProcessorCount, &max_nblocks[0][5], &max_nblocks[2][5]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[6], T_GROUP_SIZE[6] >
   (multiProcessorCount, &max_nblocks[0][6], &max_nblocks[2][6]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[7], T_GROUP_SIZE[7] >
   (multiProcessorCount, &max_nblocks[0][7], &max_nblocks[2][7]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[8], T_GROUP_SIZE[8] >
   (multiProcessorCount, &max_nblocks[0][8], &max_nblocks[2][8]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[9], T_GROUP_SIZE[9] >
   (multiProcessorCount, &max_nblocks[0][9], &max_nblocks[2][9]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[10], T_GROUP_SIZE[10] >
   (multiProcessorCount, &max_nblocks[0][10], &max_nblocks[2][10]);

   /* numeric */
   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[1], T_GROUP_SIZE[1] >
   (multiProcessorCount, &max_nblocks[1][1], &max_nblocks[3][1]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[2], T_GROUP_SIZE[2] >
   (multiProcessorCount, &max_nblocks[1][2], &max_nblocks[3][2]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[3], T_GROUP_SIZE[3] >
   (multiProcessorCount, &max_nblocks[1][3], &max_nblocks[3][3]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[4], T_GROUP_SIZE[4] >
   (multiProcessorCount, &max_nblocks[1][4], &max_nblocks[3][4]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[5], T_GROUP_SIZE[5] >
   (multiProcessorCount, &max_nblocks[1][5], &max_nblocks[3][5]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[6], T_GROUP_SIZE[6] >
   (multiProcessorCount, &max_nblocks[1][6], &max_nblocks[3][6]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[7], T_GROUP_SIZE[7] >
   (multiProcessorCount, &max_nblocks[1][7], &max_nblocks[3][7]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[8], T_GROUP_SIZE[8] >
   (multiProcessorCount, &max_nblocks[1][8], &max_nblocks[3][8]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[9], T_GROUP_SIZE[9] >
   (multiProcessorCount, &max_nblocks[1][9], &max_nblocks[3][9]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[10], T_GROUP_SIZE[10] >
   (multiProcessorCount, &max_nblocks[1][10], &max_nblocks[3][10]);

   /* highest bin with nonzero num blocks */
   typedef HYPRE_Int arr2Type[2];
   arr2Type &high_bin = hypre_HandleSpgemmHighestBin(hypre_handle());

   for (HYPRE_Int i = num_bins; i >= 0; i--) { if (max_nblocks[0][i] > 0) { high_bin[0] = i; break; } }
   for (HYPRE_Int i = num_bins; i >= 0; i--) { if (max_nblocks[1][i] > 0) { high_bin[1] = i; break; } }

   /* this is just a heuristic; having more blocks (than max active) seems improving performance */
#if defined(HYPRE_USING_CUDA)
   for (HYPRE_Int i = 0; i < num_bins + 1; i++) { max_nblocks[0][i] *= 5; max_nblocks[1][i] *= 5; }
#endif

#if defined(HYPRE_SPGEMM_PRINTF)
   HYPRE_SPGEMM_PRINT("===========================================================================\n");
   HYPRE_SPGEMM_PRINT("SM count %d\n", multiProcessorCount);
   HYPRE_SPGEMM_PRINT("Highest Bin Symbl %d, Numer %d\n",
                      hypre_HandleSpgemmHighestBin(hypre_handle())[0],
                      hypre_HandleSpgemmHighestBin(hypre_handle())[1]);
   HYPRE_SPGEMM_PRINT("---------------------------------------------------------------------------\n");
   HYPRE_SPGEMM_PRINT("Bin:      ");
   for (HYPRE_Int i = 0; i < num_bins + 1; i++) { HYPRE_SPGEMM_PRINT("%5d ", i); } HYPRE_SPGEMM_PRINT("\n");
   HYPRE_SPGEMM_PRINT("---------------------------------------------------------------------------\n");
   HYPRE_SPGEMM_PRINT("Sym-Bdim: ");
   for (HYPRE_Int i = 0; i < num_bins + 1; i++) { HYPRE_SPGEMM_PRINT("%5d ", hypre_HandleSpgemmBlockNumDim(hypre_handle())[2][i]); }
   HYPRE_SPGEMM_PRINT("\n");
   HYPRE_SPGEMM_PRINT("Sym-Gdim: ");
   for (HYPRE_Int i = 0; i < num_bins + 1; i++) { HYPRE_SPGEMM_PRINT("%5d ", hypre_HandleSpgemmBlockNumDim(hypre_handle())[0][i]); }
   HYPRE_SPGEMM_PRINT("\n");
   HYPRE_SPGEMM_PRINT("Num-Bdim: ");
   for (HYPRE_Int i = 0; i < num_bins + 1; i++) { HYPRE_SPGEMM_PRINT("%5d ", hypre_HandleSpgemmBlockNumDim(hypre_handle())[3][i]); }
   HYPRE_SPGEMM_PRINT("\n");
   HYPRE_SPGEMM_PRINT("Num-Gdim: ");
   for (HYPRE_Int i = 0; i < num_bins + 1; i++) { HYPRE_SPGEMM_PRINT("%5d ", hypre_HandleSpgemmBlockNumDim(hypre_handle())[1][i]); }
   HYPRE_SPGEMM_PRINT("\n");
   HYPRE_SPGEMM_PRINT("===========================================================================\n");
#endif

   return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_GPU) */
