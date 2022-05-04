/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

/* WM: debug */
HYPRE_Int hypre_ParCSRCommPkgPrint(hypre_ParCSRCommPkg *comm_pkg, const char *filename);
HYPRE_Int hypre_DisplayCSRMatrixRow(hypre_CSRMatrix *A, HYPRE_Int row, const char *name);
HYPRE_Int hypre_DisplayParCSRMatrixRow(hypre_ParCSRMatrix *A, HYPRE_Int row, const char *name);
HYPRE_Int hypre_DisplayInt(HYPRE_Int *array, HYPRE_Int size, HYPRE_Int display_size,
                           const char *name);
HYPRE_Int hypre_DisplayComplex(HYPRE_Complex *array, HYPRE_Int size, HYPRE_Int display_size,
                               const char *name);
HYPRE_Int hypre_DisplayCSRMatrix(hypre_CSRMatrix *A, HYPRE_Int max_display_size, const char *name);
HYPRE_Int hypre_DisplayParCSRMatrix(hypre_ParCSRMatrix *A, HYPRE_Int max_display_size,
                                    const char *name);
HYPRE_Int hypre_CompareCSRMatrix(hypre_CSRMatrix *A, hypre_CSRMatrix *B, const char *nameA,
                                 const char *nameB, HYPRE_BigInt *col_map_offd_A, HYPRE_BigInt *col_map_offd_B);
HYPRE_Int hypre_CompareParCSRMatrix(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B, const char *nameA,
                                    const char *nameB);

HYPRE_Int
hypre_ParCSRCommPkgPrint(hypre_ParCSRCommPkg *comm_pkg, const char *filename)
{
   FILE *file;
   if ((file = fopen(filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   HYPRE_Int i;

   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int *send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   HYPRE_Int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   HYPRE_Int num_send_elmts = send_map_starts[num_sends];
   HYPRE_Int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   HYPRE_Int *recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   HYPRE_Int num_recv_elmts = recv_vec_starts[num_recvs];

   hypre_fprintf(file, "%d %d %d %d\n", num_sends, num_recvs, num_send_elmts, num_recv_elmts);

   for (i = 0; i < num_sends; i++)
   {
      hypre_fprintf(file, "%d ", send_procs[i]);
   }
   hypre_fprintf(file, "\n");

   for (i = 0; i < num_sends + 1; i++)
   {
      hypre_fprintf(file, "%d ", send_map_starts[i]);
   }
   hypre_fprintf(file, "\n");

   for (i = 0; i < num_send_elmts; i++)
   {
      hypre_fprintf(file, "%d ", send_map_elmts[i]);
   }
   hypre_fprintf(file, "\n");

   for (i = 0; i < num_recvs; i++)
   {
      hypre_fprintf(file, "%d ", recv_procs[i]);
   }
   hypre_fprintf(file, "\n");

   for (i = 0; i < num_recvs + 1; i++)
   {
      hypre_fprintf(file, "%d ", recv_vec_starts[i]);
   }
   hypre_fprintf(file, "\n");

   fclose(file);

   return 0;
}

HYPRE_Int
hypre_DisplayCSRMatrixRow(hypre_CSRMatrix *A, HYPRE_Int row, const char *name)
{
   if (row >= hypre_CSRMatrixNumRows(A)) { return 0; }

   HYPRE_Int i;
   hypre_CSRMatrix *mat;
   hypre_MemoryLocation memory_location;

   // Copy to host if necessary
   hypre_GetPointerLocation(hypre_CSRMatrixI(A), &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      mat = hypre_CSRMatrixClone_v2(A, 1, HYPRE_MEMORY_HOST);
   }
   else
   {
      mat = A;
   }

   // Print row
   hypre_printf("%s_diag row %d\nj = ", name, row);
   for (i = hypre_CSRMatrixI(mat)[row]; i < hypre_CSRMatrixI(mat)[row + 1]; i++)
   {
      hypre_printf("%d ", hypre_CSRMatrixJ(mat)[i]);
   }
   hypre_printf("\ndata = ");
   for (i = hypre_CSRMatrixI(mat)[row]; i < hypre_CSRMatrixI(mat)[row + 1]; i++)
   {
      hypre_printf("%.2e ", hypre_CSRMatrixData(mat)[i]);
   }
   hypre_printf("\n");

   // Destroy copy if necessary
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      hypre_CSRMatrixDestroy(mat);
   }

   return 0;
}

HYPRE_Int
hypre_DisplayParCSRMatrixRow(hypre_ParCSRMatrix *A, HYPRE_Int row, const char *name)
{
   char csrName[256];

   strcpy(csrName, name);
   strcat(csrName, "_diag");

   hypre_DisplayCSRMatrixRow(hypre_ParCSRMatrixDiag(A), row, csrName);

   strcpy(csrName, name);
   strcat(csrName, "_offd");

   hypre_DisplayCSRMatrixRow(hypre_ParCSRMatrixOffd(A), row, csrName);

   return 0;
}

HYPRE_Int
hypre_DisplayInt(HYPRE_Int *array, HYPRE_Int size, HYPRE_Int display_size, const char *name)
{

   HYPRE_Int *disp_array;
   hypre_MemoryLocation memory_location;
   hypre_GetPointerLocation(array, &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      disp_array = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(disp_array, array, HYPRE_Int, size, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      disp_array = array;
   }
   hypre_printf("%s = ", name);
   HYPRE_Int i;
   for (i = 0; i < hypre_min(size, display_size); i++)
   {
      hypre_printf("%d ", disp_array[i]);
   }
   hypre_printf("\n");

   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      hypre_TFree(disp_array, HYPRE_MEMORY_HOST);
   }

   return 0;
}

HYPRE_Int
hypre_DisplayComplex(HYPRE_Complex *array, HYPRE_Int size, HYPRE_Int display_size, const char *name)
{

   HYPRE_Complex *disp_array;
   hypre_MemoryLocation memory_location;
   hypre_GetPointerLocation(array, &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      disp_array = hypre_TAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(disp_array, array, HYPRE_Complex, size, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      disp_array = array;
   }
   hypre_printf("%s = ", name);
   HYPRE_Int i;
   for (i = 0; i < hypre_min(size, display_size); i++)
   {
      hypre_printf("%.2e ", disp_array[i]);
   }
   hypre_printf("\n");

   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      hypre_TFree(disp_array, HYPRE_MEMORY_HOST);
   }

   return 0;
}

HYPRE_Int
hypre_DisplayCSRMatrix(hypre_CSRMatrix *A, HYPRE_Int max_display_size, const char *name)
{

   HYPRE_Int i;
   hypre_CSRMatrix *disp_mat;
   hypre_MemoryLocation memory_location;

   // Copy to host if necessary
   hypre_GetPointerLocation(hypre_CSRMatrixI(A), &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      disp_mat = hypre_CSRMatrixClone_v2(A, 1, HYPRE_MEMORY_HOST);
   }
   else
   {
      disp_mat = A;
   }

   // Print info
   hypre_printf("\n");
   hypre_printf("%s: num row, num col, nnz = %d, %d, %d\n", name, hypre_CSRMatrixNumRows(disp_mat),
                hypre_CSRMatrixNumCols(disp_mat), hypre_CSRMatrixNumNonzeros(disp_mat));
   hypre_printf("%s_i = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumRows(disp_mat) + 1); i++)
   {
      hypre_printf("%d ", hypre_CSRMatrixI(disp_mat)[i]);
   }
   hypre_printf("\n");
   hypre_printf("%s_j = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumNonzeros(disp_mat)); i++)
   {
      hypre_printf("%d ", hypre_CSRMatrixJ(disp_mat)[i]);
   }
   hypre_printf("\n");
   hypre_printf("%s_data = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumNonzeros(disp_mat)); i++)
   {
      hypre_printf("%.2e ", hypre_CSRMatrixData(disp_mat)[i]);
   }
   hypre_printf("\n");

   // Destroy host copy if necessary
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      hypre_CSRMatrixDestroy(disp_mat);
   }

   return 0;
}

HYPRE_Int
hypre_DisplayParCSRMatrix(hypre_ParCSRMatrix *A, HYPRE_Int max_display_size, const char *name)
{

   HYPRE_Int i;
   hypre_CSRMatrix *disp_mat;
   hypre_MemoryLocation memory_location;

   // Diag part

   // Copy to host if necessary
   hypre_GetPointerLocation(hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A)), &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      disp_mat = hypre_CSRMatrixClone_v2(hypre_ParCSRMatrixDiag(A), 1, HYPRE_MEMORY_HOST);
   }
   else
   {
      disp_mat = hypre_ParCSRMatrixDiag(A);
   }

   // Print info
   hypre_printf("\n");
   hypre_printf("%s_diag: num row, num col, nnz = %d, %d, %d\n", name,
                hypre_CSRMatrixNumRows(disp_mat), hypre_CSRMatrixNumCols(disp_mat),
                hypre_CSRMatrixNumNonzeros(disp_mat));
   hypre_printf("%s_diag_i = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumRows(disp_mat) + 1); i++)
   {
      hypre_printf("%d ", hypre_CSRMatrixI(disp_mat)[i]);
   }
   hypre_printf("\n");
   hypre_printf("%s_diag_j = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumNonzeros(disp_mat)); i++)
   {
      hypre_printf("%d ", hypre_CSRMatrixJ(disp_mat)[i]);
   }
   hypre_printf("\n");
   hypre_printf("%s_diag_data = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumNonzeros(disp_mat)); i++)
   {
      hypre_printf("%.2e ", hypre_CSRMatrixData(disp_mat)[i]);
   }
   hypre_printf("\n");

   // Destroy host copy if necessary
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      hypre_CSRMatrixDestroy(disp_mat);
   }

   // Offd part

   // Copy to host if necessary
   hypre_GetPointerLocation(hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A)), &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      disp_mat = hypre_CSRMatrixClone_v2(hypre_ParCSRMatrixOffd(A), 1, HYPRE_MEMORY_HOST);
   }
   else
   {
      disp_mat = hypre_ParCSRMatrixOffd(A);
   }

   // Print info
   hypre_printf("%s_offd: num row, num col, nnz = %d, %d, %d\n", name,
                hypre_CSRMatrixNumRows(disp_mat), hypre_CSRMatrixNumCols(disp_mat),
                hypre_CSRMatrixNumNonzeros(disp_mat));
   hypre_printf("%s_offd_i = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumRows(disp_mat) + 1); i++)
   {
      hypre_printf("%d ", hypre_CSRMatrixI(disp_mat)[i]);
   }
   hypre_printf("\n");
   hypre_printf("%s_offd_j = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumNonzeros(disp_mat)); i++)
   {
      hypre_printf("%d ", hypre_CSRMatrixJ(disp_mat)[i]);
   }
   hypre_printf("\n");
   hypre_printf("%s_offd_data = ", name);
   for (i = 0; i < hypre_min(max_display_size, hypre_CSRMatrixNumNonzeros(disp_mat)); i++)
   {
      hypre_printf("%.2e ", hypre_CSRMatrixData(disp_mat)[i]);
   }
   hypre_printf("\n");

   // Destroy host copy if necessary
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      hypre_CSRMatrixDestroy(disp_mat);
   }

   return 0;
}

HYPRE_Int
hypre_CompareCSRMatrix(hypre_CSRMatrix *A, hypre_CSRMatrix *B, const char *nameA, const char *nameB,
                       HYPRE_BigInt *col_map_offd_A, HYPRE_BigInt *col_map_offd_B)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   HYPRE_Int i, j, k, col, found;
   HYPRE_Int equal = 1;
   hypre_CSRMatrix *A_host, *B_host;
   hypre_MemoryLocation memory_location;

   // Get matrices
   hypre_GetPointerLocation(hypre_CSRMatrixI(A), &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      A_host = hypre_CSRMatrixClone_v2(A, 1, HYPRE_MEMORY_HOST);
   }
   else
   {
      A_host = A;
   }
   hypre_GetPointerLocation(hypre_CSRMatrixI(B), &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      B_host = hypre_CSRMatrixClone_v2(B, 1, HYPRE_MEMORY_HOST);
   }
   else
   {
      B_host = B;
   }

   // Compare matrices
   if (hypre_CSRMatrixNumRows(A_host) != hypre_CSRMatrixNumRows(B_host))
   {
      hypre_printf("Rank %d: %s num rows = %d, %s num rows = %d\n", myid,  nameA,
                   hypre_CSRMatrixNumRows(A_host), nameB, hypre_CSRMatrixNumRows(B_host));
      equal = 0;
   }
   if (col_map_offd_A == NULL)
   {
      if (hypre_CSRMatrixNumCols(A_host) != hypre_CSRMatrixNumCols(B_host))
      {
         hypre_printf("Rank %d: %s num cols = %d, %s num cols = %d\n", myid,  nameA,
                      hypre_CSRMatrixNumCols(A_host), nameB, hypre_CSRMatrixNumCols(B_host));
         equal = 0;
      }
   }
   if (hypre_CSRMatrixNumNonzeros(A_host) != hypre_CSRMatrixNumNonzeros(B_host))
   {
      hypre_printf("Rank %d: %s nnz = %d, %snnz = %d\n", myid,  nameA, hypre_CSRMatrixNumNonzeros(A_host),
                   nameB, hypre_CSRMatrixNumNonzeros(B_host));
      equal = 0;
   }
   if (equal)
   {
      for (i = 0; i < hypre_CSRMatrixNumRows(A_host) + 1; i++)
      {
         if (hypre_CSRMatrixI(A_host)[i] != hypre_CSRMatrixI(B_host)[i])
         {
            hypre_printf("Rank %d: %s_i[%d] = %d, %s_i[%d] = %d\n", myid,  nameA, i,
                         hypre_CSRMatrixI(A_host)[i], nameB, i, hypre_CSRMatrixI(B_host)[i]);
            equal = 0;
         }
      }
      for (i = 0; i < hypre_CSRMatrixNumRows(A_host); i++)
      {
         for (j = hypre_CSRMatrixI(A_host)[i]; j < hypre_CSRMatrixI(A_host)[i + 1]; j++)
         {
            col = hypre_CSRMatrixJ(A_host)[j];
            if (col_map_offd_A) { col = col_map_offd_A[col]; }
            found = 0;
            for (k = hypre_CSRMatrixI(B_host)[i]; k < hypre_CSRMatrixI(B_host)[i + 1]; k++)
            {
               HYPRE_Int colB = hypre_CSRMatrixJ(B_host)[k];
               if (col_map_offd_B) { colB = col_map_offd_B[colB]; }
               if (col == colB)
               {
                  found = 1;
                  break;
               }
            }
            if (found)
            {
               HYPRE_Real diff = hypre_cabs(hypre_CSRMatrixData(A_host)[j] - hypre_CSRMatrixData(B_host)[k]);
               HYPRE_Real rel_diff = diff / hypre_cabs(hypre_CSRMatrixData(A_host)[j]);
               if ( rel_diff > 0.01)
                  /* if ( diff > 0.000001 ) */
               {
                  hypre_printf("Rank %d: col = %d, rel_diff = %.2e, diff = %.2e, %s_data = %.2e, %s_data = %.2e\n",
                               myid, col, rel_diff, diff, nameA, hypre_CSRMatrixData(A_host)[j], nameB,
                               hypre_CSRMatrixData(B_host)[k]);
                  equal = 0;
                  break;
               }
            }
            else
            {
               hypre_printf("Rank %d: %s_j = %d not found in %s\n", myid,  nameA, col, nameB);
               equal = 0;
               break;
            }
         }
         if (!equal)
         {
            hypre_printf("row %i does not agree:\n");
            hypre_DisplayCSRMatrixRow(A, i, nameA);
            hypre_DisplayCSRMatrixRow(B, i, nameB);
            break;
         }
      }
   }

   // Destroy copies if necessary
   hypre_GetPointerLocation(hypre_CSRMatrixI(A), &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      hypre_CSRMatrixDestroy(A_host);
   }
   hypre_GetPointerLocation(hypre_CSRMatrixI(B), &memory_location);
   if (memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE))
   {
      hypre_CSRMatrixDestroy(B_host);
   }

   return equal;
}

HYPRE_Int
hypre_CompareParCSRMatrix(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B, const char *nameA,
                          const char *nameB)
{

   HYPRE_Int equal;
   char csrNameA[256];
   char csrNameB[256];

   strcpy(csrNameA, nameA);
   strcat(csrNameA, "_diag");
   strcpy(csrNameB, nameB);
   strcat(csrNameB, "_diag");

   equal = hypre_CompareCSRMatrix(hypre_ParCSRMatrixDiag(A), hypre_ParCSRMatrixDiag(B), csrNameA,
                                  csrNameB, NULL, NULL);

   if (equal)
   {
      strcpy(csrNameA, nameA);
      strcat(csrNameA, "_offd");
      strcpy(csrNameB, nameB);
      strcat(csrNameB, "_offd");

      equal = hypre_CompareCSRMatrix(hypre_ParCSRMatrixOffd(A), hypre_ParCSRMatrixOffd(B), csrNameA,
                                     csrNameB, hypre_ParCSRMatrixColMapOffd(A), hypre_ParCSRMatrixColMapOffd(B));
   }

   return equal;
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGCreateS_rowcount(
#if defined(HYPRE_USING_SYCL)
   sycl::nd_item<1>& item,
#endif
   HYPRE_Int   nr_of_rows,
   HYPRE_Real  max_row_sum,
   HYPRE_Real  strength_threshold,
   HYPRE_Real *A_diag_data,
   HYPRE_Int  *A_diag_i,
   HYPRE_Int  *A_diag_j,
   HYPRE_Real *A_offd_data,
   HYPRE_Int  *A_offd_i,
   HYPRE_Int  *A_offd_j,
   HYPRE_Int  *S_temp_diag_j,
   HYPRE_Int  *S_temp_offd_j,
   HYPRE_Int   num_functions,
   HYPRE_Int  *dof_func,
   HYPRE_Int  *dof_func_offd,
   HYPRE_Int  *jS_diag,
   HYPRE_Int  *jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
             dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_diag_j; weak: -1; diagonal: -2
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_offd_j; weak: -1;
              jS_diag       - row nnz vector for compressed S_diag
              jS_offd       - row nnz vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Real row_scale = 0.0, row_sum = 0.0, row_max = 0.0, row_min = 0.0, diag = 0.0;
   HYPRE_Int row_nnz_diag = 0, row_nnz_offd = 0, diag_pos = -1;

#if defined(HYPRE_USING_SYCL)
   const HYPRE_Int row = hypre_sycl_get_grid_warp_id(item);
#else
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1, 1>();
#endif

   if (row >= nr_of_rows)
   {
      return;
   }

#if defined(HYPRE_USING_SYCL)
   sycl::sub_group SG = item.get_sub_group();
   const HYPRE_Int lane = SG.get_local_linear_id();
#else
   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
#endif
   HYPRE_Int p_diag, q_diag, p_offd, q_offd;

   /* diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row + lane);
   }
#if defined(HYPRE_USING_SYCL)
   SG.barrier();
   q_diag = SG.shuffle(p_diag, 1);
   p_diag = SG.shuffle(p_diag, 0);
#else
   q_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 0);
#endif

   for (HYPRE_Int i = p_diag + lane; i < q_diag; i += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int col = read_only_load(&A_diag_j[i]);

      if ( num_functions == 1 || row == col ||
           read_only_load(&dof_func[row]) == read_only_load(&dof_func[col]) )
      {
         const HYPRE_Real v = read_only_load(&A_diag_data[i]);
         row_sum += v;
         if (row == col)
         {
            diag = v;
            diag_pos = i;
         }
         else
         {
            row_max = hypre_max(row_max, v);
            row_min = hypre_min(row_min, v);
         }
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row + lane);
   }
#if defined(HYPRE_USING_SYCL)
   SG.barrier();
   q_offd = SG.shuffle(p_offd, 1);
   p_offd = SG.shuffle(p_offd, 0);
#else
   q_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 0);
#endif

   for (HYPRE_Int i = p_offd + lane; i < q_offd; i += HYPRE_WARP_SIZE)
   {
      if ( num_functions == 1 ||
           read_only_load(&dof_func[row]) == read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) )
      {
         const HYPRE_Real v = read_only_load(&A_offd_data[i]);
         row_sum += v;
         row_max = hypre_max(row_max, v);
         row_min = hypre_min(row_min, v);
      }
   }

#if defined(HYPRE_USING_SYCL)
   diag = warp_allreduce_sum(diag, SG);
#else
   diag = warp_allreduce_sum(diag);
#endif

   /* sign of diag */
   const HYPRE_Int sdiag = diag > 0.0 ? 1 : -1;

   /* compute scaling factor and row sum */
#if defined(HYPRE_USING_SYCL)
   row_sum = warp_allreduce_sum(row_sum, SG);

   if (diag > 0.0)
   {
      row_scale = warp_allreduce_min(row_min, SG);
   }
   else
   {
      row_scale = warp_allreduce_max(row_max, SG);
   }
#else
   row_sum = warp_allreduce_sum(row_sum);

   if (diag > 0.0)
   {
      row_scale = warp_allreduce_min(row_min);
   }
   else
   {
      row_scale = warp_allreduce_max(row_max);
   }
#endif

   /* compute row of S */
   HYPRE_Int all_weak = max_row_sum < 1.0 && fabs(row_sum) > fabs(diag) * max_row_sum;
   const HYPRE_Real thresh = sdiag * strength_threshold * row_scale;

   for (HYPRE_Int i = p_diag + lane; i < q_diag; i += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int cond = all_weak == 0 && diag_pos != i &&
                             ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                               read_only_load(&dof_func[read_only_load(&A_diag_j[i])]) ) &&
                             sdiag * read_only_load(&A_diag_data[i]) < thresh;
      S_temp_diag_j[i] = cond * (1 + read_only_load(&A_diag_j[i])) - 1;
      row_nnz_diag += cond;
   }

   /* !!! mark diagonal as -2 !!! */
   if (diag_pos >= 0)
   {
      S_temp_diag_j[diag_pos] = -2;
   }

   for (HYPRE_Int i = p_offd + lane; i < q_offd; i += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int cond = all_weak == 0 &&
                             ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                               read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) ) &&
                             sdiag * read_only_load(&A_offd_data[i]) < thresh;
      S_temp_offd_j[i] = cond * (1 + read_only_load(&A_offd_j[i])) - 1;
      row_nnz_offd += cond;
   }

#if defined(HYPRE_USING_SYCL)
   row_nnz_diag = warp_reduce_sum(row_nnz_diag, SG);
   row_nnz_offd = warp_reduce_sum(row_nnz_offd, SG);
#else
   row_nnz_diag = warp_reduce_sum(row_nnz_diag);
   row_nnz_offd = warp_reduce_sum(row_nnz_offd);
#endif

   if (0 == lane)
   {
      jS_diag[row] = row_nnz_diag;
      jS_offd[row] = row_nnz_offd;
   }
}

/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGCreateSabs_rowcount(
#if defined(HYPRE_USING_SYCL)
   sycl::nd_item<1>& item,
#endif
   HYPRE_Int   nr_of_rows,
   HYPRE_Real  max_row_sum,
   HYPRE_Real  strength_threshold,
   HYPRE_Real *A_diag_data,
   HYPRE_Int  *A_diag_i,
   HYPRE_Int  *A_diag_j,
   HYPRE_Real *A_offd_data,
   HYPRE_Int  *A_offd_i,
   HYPRE_Int  *A_offd_j,
   HYPRE_Int  *S_temp_diag_j,
   HYPRE_Int  *S_temp_offd_j,
   HYPRE_Int   num_functions,
   HYPRE_Int  *dof_func,
   HYPRE_Int  *dof_func_offd,
   HYPRE_Int  *jS_diag,
   HYPRE_Int  *jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
             dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_diag_j; weak: -1; diagonal: -2
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_offd_j; weak: -1;
              jS_diag       - row nnz vector for compressed S_diag
              jS_offd       - row nnz vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Real row_scale = 0.0, row_sum = 0.0, diag = 0.0;
   HYPRE_Int row_nnz_diag = 0, row_nnz_offd = 0, diag_pos = -1;

#if defined(HYPRE_USING_SYCL)
   const HYPRE_Int row = hypre_sycl_get_grid_warp_id(item);
#else
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1, 1>();
#endif

   if (row >= nr_of_rows)
   {
      return;
   }

#if defined(HYPRE_USING_SYCL)
   sycl::sub_group SG = item.get_sub_group();
   const HYPRE_Int lane = SG.get_local_linear_id();
#else
   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
#endif
   HYPRE_Int p_diag, q_diag, p_offd, q_offd;

   /* diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row + lane);
   }
#if defined(HYPRE_USING_SYCL)
   SG.barrier();
   q_diag = SG.shuffle(p_diag, 1);
   p_diag = SG.shuffle(p_diag, 0);
#else
   q_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 0);
#endif

   for (HYPRE_Int i = p_diag + lane; i < q_diag; i += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int col = read_only_load(&A_diag_j[i]);

      if ( num_functions == 1 || row == col ||
           read_only_load(&dof_func[row]) == read_only_load(&dof_func[col]) )
      {
         const HYPRE_Real v = hypre_cabs( read_only_load(&A_diag_data[i]) );
         row_sum += v;
         if (row == col)
         {
            diag = v;
            diag_pos = i;
         }
         else
         {
            row_scale = hypre_max(row_scale, v);
         }
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row + lane);
   }
#if defined(HYPRE_USING_SYCL)
   SG.barrier();
   q_offd = SG.shuffle(p_offd, 1);
   p_offd = SG.shuffle(p_offd, 0);
#else
   q_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 0);
#endif

   for (HYPRE_Int i = p_offd + lane; i < q_offd; i += HYPRE_WARP_SIZE)
   {
      if ( num_functions == 1 ||
           read_only_load(&dof_func[row]) == read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) )
      {
         const HYPRE_Real v = hypre_cabs( read_only_load(&A_offd_data[i]) );
         row_sum += v;
         row_scale = hypre_max(row_scale, v);
      }
   }

#if defined(HYPRE_USING_SYCL)
   diag = warp_allreduce_sum(diag, SG);

   /* compute scaling factor and row sum */
   row_sum = warp_allreduce_sum(row_sum, SG);
   row_scale = warp_allreduce_max(row_scale, SG);
#else
   diag = warp_allreduce_sum(diag);

   /* compute scaling factor and row sum */
   row_sum = warp_allreduce_sum(row_sum);
   row_scale = warp_allreduce_max(row_scale);
#endif

   /* compute row of S */
   HYPRE_Int all_weak = max_row_sum < 1.0 && fabs(row_sum) < fabs(diag) * (2.0 - max_row_sum);
   const HYPRE_Real thresh = strength_threshold * row_scale;

   for (HYPRE_Int i = p_diag + lane; i < q_diag; i += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int cond = all_weak == 0 && diag_pos != i &&
                             ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                               read_only_load(&dof_func[read_only_load(&A_diag_j[i])]) ) &&
                             hypre_cabs( read_only_load(&A_diag_data[i]) ) > thresh;
      S_temp_diag_j[i] = cond * (1 + read_only_load(&A_diag_j[i])) - 1;
      row_nnz_diag += cond;
   }

   /* !!! mark diagonal as -2 !!! */
   if (diag_pos >= 0)
   {
      S_temp_diag_j[diag_pos] = -2;
   }

   for (HYPRE_Int i = p_offd + lane; i < q_offd; i += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int cond = all_weak == 0 &&
                             ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                               read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) ) &&
                             hypre_cabs( read_only_load(&A_offd_data[i]) ) > thresh;
      S_temp_offd_j[i] = cond * (1 + read_only_load(&A_offd_j[i])) - 1;
      row_nnz_offd += cond;
   }

#if defined(HYPRE_USING_SYCL)
   row_nnz_diag = warp_reduce_sum(row_nnz_diag, SG);
   row_nnz_offd = warp_reduce_sum(row_nnz_offd, SG);
#else
   row_nnz_diag = warp_reduce_sum(row_nnz_diag);
   row_nnz_offd = warp_reduce_sum(row_nnz_offd);
#endif

   if (0 == lane)
   {
      jS_diag[row] = row_nnz_diag;
      jS_offd[row] = row_nnz_offd;
   }
}

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
HYPRE_Int
hypre_BoomerAMGCreateSDevice(hypre_ParCSRMatrix    *A,
                             HYPRE_Int              abs_soc,
                             HYPRE_Real             strength_threshold,
                             HYPRE_Real             max_row_sum,
                             HYPRE_Int              num_functions,
                             HYPRE_Int             *dof_func,
                             hypre_ParCSRMatrix   **S_ptr)
{
   /* hypre_printf("WM: debug - inside hypre_BoomerAMGCreateSDevice()\n"); */
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] -= hypre_MPI_Wtime();
#endif

   MPI_Comm                 comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix         *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int               *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Real              *A_diag_data     = hypre_CSRMatrixData(A_diag);
   hypre_CSRMatrix         *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int               *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Real              *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int               *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int               *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_BigInt            *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int                num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt             global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int                num_nonzeros_diag;
   HYPRE_Int                num_nonzeros_offd;
   HYPRE_Int                num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   hypre_ParCSRMatrix      *S;
   hypre_CSRMatrix         *S_diag;
   HYPRE_Int               *S_diag_i;
   HYPRE_Int               *S_diag_j, *S_temp_diag_j;
   /* HYPRE_Real           *S_diag_data; */
   hypre_CSRMatrix         *S_offd;
   HYPRE_Int               *S_offd_i = NULL;
   HYPRE_Int               *S_offd_j = NULL, *S_temp_offd_j = NULL;
   /* HYPRE_Real           *S_offd_data; */
   HYPRE_Int                ierr = 0;
   HYPRE_Int               *dof_func_offd_dev = NULL;
   HYPRE_Int                num_sends;

   HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * Default "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * If abs_soc != 0, then use an absolute strength of connection:
    * i depends on j if
    *     abs(aij) > hypre_max (k != i) abs(aik)
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(A_diag);
   num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(A_offd);

   S_diag_i = hypre_TAlloc(HYPRE_Int, num_variables + 1, memory_location);
   S_offd_i = hypre_TAlloc(HYPRE_Int, num_variables + 1, memory_location);
   S_temp_diag_j = hypre_TAlloc(HYPRE_Int, num_nonzeros_diag, HYPRE_MEMORY_DEVICE);
   S_temp_offd_j = hypre_TAlloc(HYPRE_Int, num_nonzeros_offd, HYPRE_MEMORY_DEVICE);

   if (num_functions > 1)
   {
      dof_func_offd_dev = hypre_TAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_DEVICE);
   }

   /*-------------------------------------------------------------------
     * Get the dof_func data for the off-processor columns
     *-------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_functions > 1)
   {
      HYPRE_Int *int_buf_data = hypre_TAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                        num_sends), HYPRE_MEMORY_DEVICE);

      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                        hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                        dof_func,
                        int_buf_data );
#else
      HYPRE_THRUST_CALL( gather,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                         hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                         dof_func,
                         int_buf_data );
#endif

      comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, HYPRE_MEMORY_DEVICE, int_buf_data,
                                                    HYPRE_MEMORY_DEVICE, dof_func_offd_dev);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      hypre_TFree(int_buf_data, HYPRE_MEMORY_DEVICE);
   }

   /* count the row nnz of S */
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_variables, "warp", bDim);

   if (abs_soc)
   {
      HYPRE_GPU_LAUNCH( hypre_BoomerAMGCreateSabs_rowcount, gDim, bDim,
                        num_variables, max_row_sum, strength_threshold,
                        A_diag_data, A_diag_i, A_diag_j,
                        A_offd_data, A_offd_i, A_offd_j,
                        S_temp_diag_j, S_temp_offd_j,
                        num_functions, dof_func, dof_func_offd_dev,
                        S_diag_i, S_offd_i );
   }
   else
   {
      HYPRE_GPU_LAUNCH( hypre_BoomerAMGCreateS_rowcount, gDim, bDim,
                        num_variables, max_row_sum, strength_threshold,
                        A_diag_data, A_diag_i, A_diag_j,
                        A_offd_data, A_offd_i, A_offd_j,
                        S_temp_diag_j, S_temp_offd_j,
                        num_functions, dof_func, dof_func_offd_dev,
                        S_diag_i, S_offd_i );
   }

   hypre_Memset(S_diag_i + num_variables, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
   hypre_Memset(S_offd_i + num_variables, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

   hypreDevice_IntegerExclusiveScan(num_variables + 1, S_diag_i);
   hypreDevice_IntegerExclusiveScan(num_variables + 1, S_offd_i);

   HYPRE_Int *tmp, S_num_nonzeros_diag, S_num_nonzeros_offd;

   hypre_TMemcpy(&S_num_nonzeros_diag, &S_diag_i[num_variables], HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                 memory_location);
   hypre_TMemcpy(&S_num_nonzeros_offd, &S_offd_i[num_variables], HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                 memory_location);

   S_diag_j = hypre_TAlloc(HYPRE_Int, S_num_nonzeros_diag, memory_location);
   S_offd_j = hypre_TAlloc(HYPRE_Int, S_num_nonzeros_offd, memory_location);

#if defined(HYPRE_USING_SYCL)
   tmp = HYPRE_ONEDPL_CALL(std::copy_if, S_temp_diag_j, S_temp_diag_j + num_nonzeros_diag, S_diag_j,
                           is_nonnegative<HYPRE_Int>());
#else
   tmp = HYPRE_THRUST_CALL(copy_if, S_temp_diag_j, S_temp_diag_j + num_nonzeros_diag, S_diag_j,
                           is_nonnegative<HYPRE_Int>());
#endif

   hypre_assert(S_num_nonzeros_diag == tmp - S_diag_j);

#if defined(HYPRE_USING_SYCL)
   tmp = HYPRE_ONEDPL_CALL(std::copy_if, S_temp_offd_j, S_temp_offd_j + num_nonzeros_offd, S_offd_j,
                           is_nonnegative<HYPRE_Int>());
#else
   tmp = HYPRE_THRUST_CALL(copy_if, S_temp_offd_j, S_temp_offd_j + num_nonzeros_offd, S_offd_j,
                           is_nonnegative<HYPRE_Int>());
#endif

   hypre_assert(S_num_nonzeros_offd == tmp - S_offd_j);

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars, row_starts, row_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);

   S_diag = hypre_ParCSRMatrixDiag(S);
   S_offd = hypre_ParCSRMatrixOffd(S);

   hypre_CSRMatrixNumNonzeros(S_diag) = S_num_nonzeros_diag;
   hypre_CSRMatrixNumNonzeros(S_offd) = S_num_nonzeros_offd;
   hypre_CSRMatrixI(S_diag) = S_diag_i;
   hypre_CSRMatrixJ(S_diag) = S_diag_j;
   hypre_CSRMatrixI(S_offd) = S_offd_i;
   hypre_CSRMatrixJ(S_offd) = S_offd_j;
   hypre_CSRMatrixMemoryLocation(S_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(S_offd) = memory_location;

   hypre_ParCSRMatrixCommPkg(S) = NULL;

   hypre_ParCSRMatrixColMapOffd(S) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(S), hypre_ParCSRMatrixColMapOffd(A),
                 HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrixSocDiagJ(S) = S_temp_diag_j;
   hypre_ParCSRMatrixSocOffdJ(S) = S_temp_offd_j;

   *S_ptr = S;
   hypre_ParCSRMatrixPrint(S, "S");

   hypre_TFree(dof_func_offd_dev, HYPRE_MEMORY_DEVICE);
   /*
   hypre_TFree(S_temp_diag_j,     HYPRE_MEMORY_DEVICE);
   hypre_TFree(S_temp_offd_j,     HYPRE_MEMORY_DEVICE);
   */

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] += hypre_MPI_Wtime();
#endif

   /* hypre_printf("WM: debug - finished hypre_BoomerAMGCreateSDevice()\n"); */
   return (ierr);
}


HYPRE_Int
hypre_BoomerAMGMakeSocFromSDevice( hypre_ParCSRMatrix *A,
                                   hypre_ParCSRMatrix *S)
{
   /* hypre_printf("WM: debug - inside hypre_BoomerAMGMakeSocFromSDevice()\n"); */
   if (!hypre_ParCSRMatrixSocDiagJ(S))
   {
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
      HYPRE_Int nnz_diag = hypre_CSRMatrixNumNonzeros(A_diag);
      HYPRE_Int *soc_diag = hypre_TAlloc(HYPRE_Int, nnz_diag, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixIntersectPattern(A_diag, S_diag, soc_diag, 1);
      hypre_ParCSRMatrixSocDiagJ(S) = soc_diag;
   }

   if (!hypre_ParCSRMatrixSocOffdJ(S))
   {
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
      hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
      HYPRE_Int nnz_offd = hypre_CSRMatrixNumNonzeros(A_offd);
      HYPRE_Int *soc_offd = hypre_TAlloc(HYPRE_Int, nnz_offd, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixIntersectPattern(A_offd, S_offd, soc_offd, 0);
      hypre_ParCSRMatrixSocOffdJ(S) = soc_offd;
   }

   /* hypre_printf("WM: debug - finished hypre_BoomerAMGMakeSocFromSDevice()\n"); */
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCorrectCFMarker : corrects CF_marker after aggr. coarsening
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCorrectCFMarkerDevice(hypre_IntArray *CF_marker, hypre_IntArray *new_CF_marker)
{
   /* hypre_printf("WM: debug - inside hypre_BoomerAMGCorrectCFMarkerDevice()\n"); */

   HYPRE_Int n_fine     = hypre_IntArraySize(CF_marker);
   HYPRE_Int n_coarse   = hypre_IntArraySize(new_CF_marker);

   HYPRE_Int *indices   = hypre_CTAlloc(HYPRE_Int, n_coarse, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *CF_C      = hypre_CTAlloc(HYPRE_Int, n_coarse, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   /* save CF_marker values at C points in CF_C and C point indices */
   HYPRE_ONEDPL_CALL( std::copy_if,
                      hypre_IntArrayData(CF_marker),
                      hypre_IntArrayData(CF_marker) + n_fine,
                      CF_C,
                      is_positive<HYPRE_Int>() );
   hypreSycl_copy_if( oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<HYPRE_Int>(n_fine),
                      hypre_IntArrayData(CF_marker),
                      indices,
                      is_positive<HYPRE_Int>() );

   /* replace CF_marker at C points with 1 */
   HYPRE_ONEDPL_CALL( std::replace_if,
                      hypre_IntArrayData(CF_marker),
                      hypre_IntArrayData(CF_marker) + n_fine,
                      is_positive<HYPRE_Int>(),
                      1 );

   /* update with new_CF_marker wherever C point value was initially 1 */
   hypreSycl_scatter_if( hypre_IntArrayData(new_CF_marker),
                         hypre_IntArrayData(new_CF_marker) + n_coarse,
                         indices,
                         CF_C,
                         hypre_IntArrayData(CF_marker),
                         equal<HYPRE_Int>(1) );
#else
   /* save CF_marker values at C points in CF_C and C point indices */
   HYPRE_THRUST_CALL( copy_if,
                      hypre_IntArrayData(CF_marker),
                      hypre_IntArrayData(CF_marker) + n_fine,
                      CF_C,
                      is_positive<HYPRE_Int>() );
   HYPRE_THRUST_CALL( copy_if,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(n_fine),
                      hypre_IntArrayData(CF_marker),
                      indices,
                      is_positive<HYPRE_Int>() );

   /* replace CF_marker at C points with 1 */
   HYPRE_THRUST_CALL( replace_if,
                      hypre_IntArrayData(CF_marker),
                      hypre_IntArrayData(CF_marker) + n_fine,
                      is_positive<HYPRE_Int>(),
                      1 );

   /* update with new_CF_marker wherever C point value was initially 1 */
   HYPRE_THRUST_CALL( scatter_if,
                      hypre_IntArrayData(new_CF_marker),
                      hypre_IntArrayData(new_CF_marker) + n_coarse,
                      indices,
                      CF_C,
                      hypre_IntArrayData(CF_marker),
                      equal<HYPRE_Int>(1) );
#endif

   hypre_TFree(indices, HYPRE_MEMORY_DEVICE);
   hypre_TFree(CF_C, HYPRE_MEMORY_DEVICE);

   /* hypre_printf("WM: debug - finished hypre_BoomerAMGCorrectCFMarkerDevice()\n"); */
   return 0;
}
/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCorrectCFMarker2 : corrects CF_marker after aggr. coarsening,
 * but marks new F-points (previous C-points) as -2
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCorrectCFMarker2Device(hypre_IntArray *CF_marker, hypre_IntArray *new_CF_marker)
{
   /* hypre_printf("WM: debug - inside hypre_BoomerAMGCorrectCFMarker2Device()\n"); */

   HYPRE_Int n_fine     = hypre_IntArraySize(CF_marker);
   HYPRE_Int n_coarse   = hypre_IntArraySize(new_CF_marker);

   HYPRE_Int *indices   = hypre_CTAlloc(HYPRE_Int, n_coarse, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   /* save C point indices */
   hypreSycl_copy_if( oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<HYPRE_Int>(n_fine),
                      hypre_IntArrayData(CF_marker),
                      indices,
                      is_positive<HYPRE_Int>() );

   /* replace CF_marker at C points with 1 */
   HYPRE_ONEDPL_CALL( std::replace_if,
                      hypre_IntArrayData(CF_marker),
                      hypre_IntArrayData(CF_marker) + n_fine,
                      is_positive<HYPRE_Int>(),
                      1 );

   /* update values in CF_marker to -2 wherever new_CF_marker == -1 */
   hypreSycl_transform_if( oneapi::dpl::make_permutation_iterator(hypre_IntArrayData(CF_marker),
                                                                  indices),
                           oneapi::dpl::make_permutation_iterator(hypre_IntArrayData(CF_marker), indices) + n_coarse,
                           hypre_IntArrayData(new_CF_marker),
                           oneapi::dpl::make_permutation_iterator(hypre_IntArrayData(CF_marker), indices),
   [] (const auto & x) { return -2; },
   equal<HYPRE_Int>(-1) );
#else
   /* save C point indices */
   HYPRE_THRUST_CALL( copy_if,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(n_fine),
                      hypre_IntArrayData(CF_marker),
                      indices,
                      is_positive<HYPRE_Int>() );

   /* replace CF_marker at C points with 1 */
   HYPRE_THRUST_CALL( replace_if,
                      hypre_IntArrayData(CF_marker),
                      hypre_IntArrayData(CF_marker) + n_fine,
                      is_positive<HYPRE_Int>(),
                      1 );

   /* update values in CF_marker to -2 wherever new_CF_marker == -1 */
   HYPRE_THRUST_CALL( scatter_if,
                      thrust::make_constant_iterator(-2),
                      thrust::make_constant_iterator(-2) + n_coarse,
                      indices,
                      hypre_IntArrayData(new_CF_marker),
                      hypre_IntArrayData(CF_marker),
                      equal<HYPRE_Int>(-1) );
#endif

   hypre_TFree(indices, HYPRE_MEMORY_DEVICE);

   /* hypre_printf("WM: debug - finished hypre_BoomerAMGCorrectCFMarker2Device()\n"); */
   return 0;
}

#endif /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL) */
