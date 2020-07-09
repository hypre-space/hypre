/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA)

#if 0
void copy_csr_device_to_host(HYPRE_Int m,  HYPRE_Int nnz,
                             const HYPRE_Int* d_i, const HYPRE_Int* d_j, const HYPRE_Complex* d_v,
                             HYPRE_Int** h_i, HYPRE_Int** h_j, HYPRE_Complex** h_v)
{
         *h_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_HOST);
         *h_j = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(*h_i,   d_i,  HYPRE_Int, m+1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(*h_j,   d_j,  HYPRE_Int, nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
         if(d_v != NULL && h_v != NULL) {
            *h_v = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(*h_v,   d_v,  HYPRE_Complex, nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
         }
}
void assert_csr_equal(HYPRE_Int m,  HYPRE_Int nnz,
                             const HYPRE_Int* h_i1, const HYPRE_Int* h_j1, const HYPRE_Complex* h_v1,
                             const HYPRE_Int* h_i2, const HYPRE_Int* h_j2, const HYPRE_Complex* h_v2) {
   for(int i = 0; i < m+1; i++) {
      assert(h_i1[i] == h_i2[i]);
      if(h_i1[i] != h_i2[i]) {
         printf("FAIL\n");
         exit(0);
      }
   }

   for(int i = 0; i < nnz; i++) {
      assert(h_j1[i] == h_j2[i]);
      if(fabs(h_v1[i] - h_v2[i]) > 1e-14) {
         printf("%f : %f ", h_v1[i], h_v2[i]);
         printf("FAIL\n");
         exit(0);
      }
   }
   printf("Validated %i things\n", nnz);
   }
void print_csr_to_file(HYPRE_Int m, HYPRE_Int nnz, const char* filebase, 
                       const HYPRE_Int* h_i, const HYPRE_Int* h_j, const HYPRE_Complex* h_v) {

   FILE* fp;
   assert(strlen(filebase) < 100);
   char filename[128];
   strcpy(filename, filebase);
   strcat(filename, "_i.csv");
   fp = fopen(filename, "w");
   for(int i = 0; i < m+1; i++) {
      fprintf(fp, "%i\n", h_i[i]);
   }
   fclose(fp);

   strcpy(filename, filebase);
   strcat(filename, "_j.csv");
   fp = fopen(filename, "w");
   for(int i = 0; i < nnz; i++) {
      fprintf(fp, "%i\n", h_j[i]);
   }
   fclose(fp);

   if(h_v != NULL) {
      strcpy(filename, filebase);
      strcat(filename, "_v.csv");
      fp = fopen(filename, "w");
      for(int i = 0; i < nnz; i++) {
         fprintf(fp, "%f\n", h_v[i]);
      }
      fclose(fp);
   }
}
#endif

HYPRE_Int
hypreDevice_CSRSpGemmm(HYPRE_Int   m,        HYPRE_Int   k,        HYPRE_Int       r,       HYPRE_Int n, 
                      HYPRE_Int   nnza,     HYPRE_Int   nnzb,     HYPRE_Int       nnzc,
                      HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_a,
                      HYPRE_Int  *d_ib,     HYPRE_Int  *d_jb,     HYPRE_Complex  *d_b,
                      HYPRE_Int  *d_ic,     HYPRE_Int  *d_jc,     HYPRE_Complex  *d_c,
                      HYPRE_Int **d_id_out, HYPRE_Int **d_jd_out, HYPRE_Complex **d_d_out,
                      HYPRE_Int  *nnzD)
{
   hypre_printf(" Starting tripmat \n");
   for (int z = 0; z < m+1; z++) {
   }
   /* trivial case */
   if (nnza == 0 || nnzb == 0 || nnzc == 0)
   {
      *d_id_out = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      *d_jd_out = hypre_CTAlloc(HYPRE_Int,     0, HYPRE_MEMORY_DEVICE);
      *d_d_out  = hypre_CTAlloc(HYPRE_Complex, 0, HYPRE_MEMORY_DEVICE);
      *nnzD = 0;

      return hypre_error_flag;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM] -= hypre_MPI_Wtime();
#endif

   /* use CUSPARSE */
   if (hypre_HandleSpgemmUseCusparse(hypre_handle()))
   {
      HYPRE_Int *d_ibc, *d_jbc;
      HYPRE_Complex *d_bc;
      HYPRE_Int nnzBC;

      hypreDevice_CSRSpGemmCusparse(k, r, n, nnzb, d_ib, d_jb, d_b, nnzc, d_ic, d_jc, d_c,
                                     &nnzBC, &d_ibc, &d_jbc, &d_bc);

      hypreDevice_CSRSpGemmCusparse(m, k, n, nnza, d_ia, d_ja, d_a, nnzBC, d_ibc, d_jbc, d_bc,
                                     nnzD, d_id_out, d_jd_out, d_d_out);
   }
   else
   {
      HYPRE_Int m2 = hypre_HandleSpgemmNumPasses(hypre_handle()) < 3 ? m : 2*m;
      HYPRE_Int *d_rd = hypre_TAlloc(HYPRE_Int, m2, HYPRE_MEMORY_DEVICE);

      hypreDevice_CSRSpGemmmRownnzEstimate(m, k, r, n, d_ia, d_ja, d_ib, d_jb, d_ic, d_jc, d_rd);

      if (hypre_HandleSpgemmNumPasses(hypre_handle()) < 3)
      {
         hypre_printf("UNSUported\n");
         assert(0);
      }
      else
      {
         HYPRE_Int rownnz_exact;
         /* a binary array to indicate if row nnz counting is failed for a row */
         //HYPRE_Int *d_rf = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
         HYPRE_Int *d_rf = d_rd + m;

         hypreDevice_CSRSpGemmmRownnzUpperbound(m, k, r, n, d_ia, d_ja, d_ib, d_jb, d_ic, d_jc, d_rd, d_rf);

         /* row nnz is exact if no row failed */
         rownnz_exact = hypreDevice_IntegerReduceSum(m, d_rf) == 0;

         hypreDevice_CSRSpGemmmWithRownnzUpperbound(m, k, r, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_rd, rownnz_exact,
                                                   d_id_out, d_jd_out, d_d_out, nnzD);
      }

      hypre_TFree(d_rd, HYPRE_MEMORY_DEVICE);
   }
#ifdef HYPRE_PROFILE
   cudaThreadSynchronize();
   hypre_profile_times[HYPRE_TIMER_ID_SPMM] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}


#endif /* HYPRE_USING_CUDA */
