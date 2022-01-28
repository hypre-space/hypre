/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "seq_mv.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#include "csr_spgemm_device.h"

HYPRE_Int hypreDevice_CSRSpGemmBinnedGetMaxNumBlocks()
{
   hypre_int multiProcessorCount = 0;
   /* bins 1, 2, ..., num_bins, are effective; 0 is reserved for empty rows */
   const HYPRE_Int num_bins = 10;

   hypre_HandleSpgemmAlgorithmNumBin(hypre_handle()) = num_bins;

#if defined(HYPRE_USING_CUDA)
   cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, hypre_HandleDevice(hypre_handle()));
#endif

#if defined(HYPRE_USING_HIP)
   hipDeviceGetAttribute(&multiProcessorCount, hipDeviceAttributeMultiprocessorCount, hypre_HandleDevice(hypre_handle()));
#endif

   auto max_nblocks = hypre_HandleSpgemmAlgorithmMaxNumBlocks(hypre_handle());

   for (HYPRE_Int i = 0; i < num_bins + 1; i++)
   {
      max_nblocks[0][i] = max_nblocks[1][i] = 0;
   }

   /* symbolic */
   hypre_spgemm_symbolic_max_num_blocks<HYPRE_SPGEMM_SYMBL_HASH_SIZE /  4, HYPRE_SPGEMM_BASE_GROUP_SIZE /  4>
      (multiProcessorCount, &max_nblocks[0][3]);
   hypre_spgemm_symbolic_max_num_blocks<HYPRE_SPGEMM_SYMBL_HASH_SIZE /  2, HYPRE_SPGEMM_BASE_GROUP_SIZE /  2>
      (multiProcessorCount, &max_nblocks[0][4]);
   hypre_spgemm_symbolic_max_num_blocks<HYPRE_SPGEMM_SYMBL_HASH_SIZE,      HYPRE_SPGEMM_BASE_GROUP_SIZE>
      (multiProcessorCount, &max_nblocks[0][5]);
   hypre_spgemm_symbolic_max_num_blocks<HYPRE_SPGEMM_SYMBL_HASH_SIZE *  2, HYPRE_SPGEMM_BASE_GROUP_SIZE *  2>
      (multiProcessorCount, &max_nblocks[0][6]);
   hypre_spgemm_symbolic_max_num_blocks<HYPRE_SPGEMM_SYMBL_HASH_SIZE *  4, HYPRE_SPGEMM_BASE_GROUP_SIZE *  4>
      (multiProcessorCount, &max_nblocks[0][7]);
   hypre_spgemm_symbolic_max_num_blocks<HYPRE_SPGEMM_SYMBL_HASH_SIZE *  8, HYPRE_SPGEMM_BASE_GROUP_SIZE *  8>
      (multiProcessorCount, &max_nblocks[0][8]);
   hypre_spgemm_symbolic_max_num_blocks<HYPRE_SPGEMM_SYMBL_HASH_SIZE * 16, HYPRE_SPGEMM_BASE_GROUP_SIZE * 16>
      (multiProcessorCount, &max_nblocks[0][9]);
   hypre_spgemm_symbolic_max_num_blocks<HYPRE_SPGEMM_SYMBL_HASH_SIZE * 32, HYPRE_SPGEMM_BASE_GROUP_SIZE * 32>
      (multiProcessorCount, &max_nblocks[0][10]);

   /* numeric */
   hypre_spgemm_numerical_max_num_blocks<HYPRE_SPGEMM_NUMER_HASH_SIZE /  4, HYPRE_SPGEMM_BASE_GROUP_SIZE /  4>
      (multiProcessorCount, &max_nblocks[1][3]);
   hypre_spgemm_numerical_max_num_blocks<HYPRE_SPGEMM_NUMER_HASH_SIZE /  2, HYPRE_SPGEMM_BASE_GROUP_SIZE /  2>
      (multiProcessorCount, &max_nblocks[1][4]);
   hypre_spgemm_numerical_max_num_blocks<HYPRE_SPGEMM_NUMER_HASH_SIZE,      HYPRE_SPGEMM_BASE_GROUP_SIZE>
      (multiProcessorCount, &max_nblocks[1][5]);
   hypre_spgemm_numerical_max_num_blocks<HYPRE_SPGEMM_NUMER_HASH_SIZE *  2, HYPRE_SPGEMM_BASE_GROUP_SIZE *  2>
      (multiProcessorCount, &max_nblocks[1][6]);
   hypre_spgemm_numerical_max_num_blocks<HYPRE_SPGEMM_NUMER_HASH_SIZE *  4, HYPRE_SPGEMM_BASE_GROUP_SIZE *  4>
      (multiProcessorCount, &max_nblocks[1][7]);
   hypre_spgemm_numerical_max_num_blocks<HYPRE_SPGEMM_NUMER_HASH_SIZE *  8, HYPRE_SPGEMM_BASE_GROUP_SIZE *  8>
      (multiProcessorCount, &max_nblocks[1][8]);
   hypre_spgemm_numerical_max_num_blocks<HYPRE_SPGEMM_NUMER_HASH_SIZE * 16, HYPRE_SPGEMM_BASE_GROUP_SIZE * 16>
      (multiProcessorCount, &max_nblocks[1][9]);
   hypre_spgemm_numerical_max_num_blocks<HYPRE_SPGEMM_NUMER_HASH_SIZE * 32, HYPRE_SPGEMM_BASE_GROUP_SIZE * 32>
      (multiProcessorCount, &max_nblocks[1][10]);

   /* this is just a heuristic; having more blocks (than max active) seems improving performance */
   for (HYPRE_Int i = 0; i < num_bins + 1; i++) { max_nblocks[0][i] *= 5; max_nblocks[1][i] *= 5; }
   //for (HYPRE_Int i = 0; i < num_bins + 1; i++) { max_nblocks[0][i] = max_nblocks[1][i] = 8192; }

#if defined(HYPRE_SPGEMM_PRINTF)
   printf0("=========================================================================\n");
   printf0("SM count %d\n", multiProcessorCount);
   printf0("Bin: "); for (HYPRE_Int i = 0; i < num_bins + 1; i++) { printf0("%5d ", i); } printf0("\n");
   printf0("Sym: "); for (HYPRE_Int i = 0; i < num_bins + 1; i++) { printf0("%5d ", max_nblocks[0][i]); } printf0("\n");
   printf0("Num: "); for (HYPRE_Int i = 0; i < num_bins + 1; i++) { printf0("%5d ", max_nblocks[1][i]); } printf0("\n");
   printf0("=========================================================================\n");
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

