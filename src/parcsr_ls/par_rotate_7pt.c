/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_GenerateLaplacian9pt
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateRotate7pt( MPI_Comm       comm,
                   HYPRE_BigInt   nx,
                   HYPRE_BigInt   ny,
                   HYPRE_Int      P,
                   HYPRE_Int      Q,
                   HYPRE_Int      p,
                   HYPRE_Int      q,
                   HYPRE_Real     alpha,
                   HYPRE_Real     eps )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int    *diag_i;
   HYPRE_Int    *diag_j;
   HYPRE_Real *diag_data;

   HYPRE_Int    *offd_i;
   HYPRE_Int    *offd_j = NULL;
   HYPRE_Real *offd_data = NULL;

   HYPRE_Real *value;
   HYPRE_Real ac, bc, cc, s, c, pi, x;
   HYPRE_BigInt global_part[2];
   HYPRE_BigInt ix, iy;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows;
   HYPRE_BigInt *col_map_offd;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Int row_index;
   HYPRE_Int i;

   HYPRE_Int nx_local, ny_local;
   HYPRE_Int num_cols_offd;
   HYPRE_BigInt grid_size;

   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;

   HYPRE_Int num_procs;
   HYPRE_Int P_busy, Q_busy;

   hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny;

   value = hypre_CTAlloc(HYPRE_Real, 4, HYPRE_MEMORY_HOST);
   pi = 4.0 * hypre_atan(1.0);
   x = pi * alpha / 180.0;
   s = hypre_sin(x);
   c = hypre_cos(x);
   ac = -(c * c + eps * s * s);
   bc = 2.0 * (1.0 - eps) * s * c;
   cc = -(s * s + eps * c * c);
   value[0] = -2 * (2 * ac + bc + 2 * cc);
   value[1] = 2 * ac + bc;
   value[2] = bc + 2 * cc;
   value[3] = -bc;

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);

   local_num_rows = nx_local * ny_local;

   global_part[0] = ny_part[q] * nx + nx_part[p] * ny_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_num_rows;

   diag_i = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);
   offd_i = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);

   P_busy = hypre_min(nx, P);
   Q_busy = hypre_min(ny, Q);

   num_cols_offd = 0;
   if (p) { num_cols_offd += ny_local; }
   if (p < P_busy - 1) { num_cols_offd += ny_local; }
   if (q) { num_cols_offd += nx_local; }
   if (q < Q_busy - 1) { num_cols_offd += nx_local; }
   if (p && q) { num_cols_offd++; }
   if (p && q < Q_busy - 1 ) { num_cols_offd++; }
   if (p < P_busy - 1 && q ) { num_cols_offd++; }
   if (p < P_busy - 1 && q < Q_busy - 1 ) { num_cols_offd++; }

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt,  num_cols_offd, HYPRE_MEMORY_HOST);

   cnt = 0;
   o_cnt = 0;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
   {
      for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
      {
         cnt++;
         o_cnt++;
         diag_i[cnt] = diag_i[cnt - 1];
         offd_i[o_cnt] = offd_i[o_cnt - 1];
         diag_i[cnt]++;
         if (iy > ny_part[q])
         {
            diag_i[cnt]++;
            if (ix > nx_part[p])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (ix)
               {
                  offd_i[o_cnt]++;
               }
            }
         }
         else
         {
            if (iy)
            {
               offd_i[o_cnt]++;
               if (ix > nx_part[p])
               {
                  offd_i[o_cnt]++;
               }
               else if (ix)
               {
                  offd_i[o_cnt]++;
               }
            }
         }
         if (ix > nx_part[p])
         {
            diag_i[cnt]++;
         }
         else
         {
            if (ix)
            {
               offd_i[o_cnt]++;
            }
         }
         if (ix + 1 < nx_part[p + 1])
         {
            diag_i[cnt]++;
         }
         else
         {
            if (ix + 1 < nx)
            {
               offd_i[o_cnt]++;
            }
         }
         if (iy + 1 < ny_part[q + 1])
         {
            diag_i[cnt]++;
            if (ix < nx_part[p + 1] - 1)
            {
               diag_i[cnt]++;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  offd_i[o_cnt]++;
               }
            }
         }
         else
         {
            if (iy + 1 < ny)
            {
               offd_i[o_cnt]++;
               if (ix < nx_part[p + 1] - 1)
               {
                  offd_i[o_cnt]++;
               }
               else if (ix < nx - 1)
               {
                  offd_i[o_cnt]++;
               }
            }
         }
      }
   }

   diag_j    = hypre_CTAlloc(HYPRE_Int,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);
   diag_data = hypre_CTAlloc(HYPRE_Real, diag_i[local_num_rows], HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      big_offd_j = hypre_CTAlloc(HYPRE_BigInt, offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_j     = hypre_CTAlloc(HYPRE_Int,    offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_data  = hypre_CTAlloc(HYPRE_Real,   offd_i[local_num_rows], HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
   {
      for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
      {
         diag_j[cnt] = row_index;
         diag_data[cnt++] = value[0];
         if (iy > ny_part[q])
         {
            if (ix > nx_part[p])
            {
               diag_j[cnt] = row_index - nx_local - 1 ;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = hypre_map2(ix - 1, iy - 1, p - 1, q, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
            diag_j[cnt] = row_index - nx_local;
            diag_data[cnt++] = value[2];
         }
         else
         {
            if (iy)
            {
               if (ix > nx_part[p])
               {
                  big_offd_j[o_cnt] = hypre_map2(ix - 1, iy - 1, p, q - 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
               else if (ix)
               {
                  big_offd_j[o_cnt] = hypre_map2(ix - 1, iy - 1, p - 1, q - 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
               big_offd_j[o_cnt] = hypre_map2(ix, iy - 1, p, q - 1, nx,
                                              nx_part, ny_part);
               offd_data[o_cnt++] = value[2];
            }
         }
         if (ix > nx_part[p])
         {
            diag_j[cnt] = row_index - 1;
            diag_data[cnt++] = value[1];
         }
         else
         {
            if (ix)
            {
               big_offd_j[o_cnt] = hypre_map2(ix - 1, iy, p - 1, q, nx,
                                              nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
            }
         }
         if (ix + 1 < nx_part[p + 1])
         {
            diag_j[cnt] = row_index + 1;
            diag_data[cnt++] = value[1];
         }
         else
         {
            if (ix + 1 < nx)
            {
               big_offd_j[o_cnt] = hypre_map2(ix + 1, iy, p + 1, q, nx,
                                              nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
            }
         }
         if (iy + 1 < ny_part[q + 1])
         {
            diag_j[cnt] = row_index + nx_local;
            diag_data[cnt++] = value[2];
            if (ix < nx_part[p + 1] - 1)
            {
               diag_j[cnt] = row_index + nx_local + 1 ;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = hypre_map2(ix + 1, iy + 1, p + 1, q, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
         }
         else
         {
            if (iy + 1 < ny)
            {
               big_offd_j[o_cnt] = hypre_map2(ix, iy + 1, p, q + 1, nx,
                                              nx_part, ny_part);
               offd_data[o_cnt++] = value[2];
               if (ix < nx_part[p + 1] - 1)
               {
                  big_offd_j[o_cnt] = hypre_map2(ix + 1, iy + 1, p, q + 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
               else if (ix < nx - 1)
               {
                  big_offd_j[o_cnt] = hypre_map2(ix + 1, iy + 1, p + 1, q + 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
         }
         row_index++;
      }
   }

   if (num_procs > 1)
   {
      HYPRE_BigInt *work = hypre_CTAlloc(HYPRE_BigInt, o_cnt, HYPRE_MEMORY_HOST);

      for (i = 0; i < o_cnt; i++)
      {
         work[i] = big_offd_j[i];
      }

      hypre_BigQsort0(work, 0, o_cnt - 1);

      col_map_offd[0] = work[0];
      cnt = 0;
      for (i = 0; i < o_cnt; i++)
      {
         if (work[i] > col_map_offd[cnt])
         {
            cnt++;
            col_map_offd[cnt] = work[i];
         }
      }

      num_cols_offd = cnt + 1;
      for (i = 0; i < o_cnt; i++)
      {
         offd_j[i] = hypre_BigBinarySearch(col_map_offd, big_offd_j[i], num_cols_offd);
      }

      hypre_TFree(work, HYPRE_MEMORY_HOST);
      hypre_TFree(big_offd_j, HYPRE_MEMORY_HOST);
   }

   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, num_cols_offd,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);

   hypre_ParCSRMatrixColMapOffd(A) = col_map_offd;

   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;

   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;
   if (num_cols_offd)
   {
      hypre_CSRMatrixJ(offd) = offd_j;
      hypre_CSRMatrixData(offd) = offd_data;
   }

   hypre_CSRMatrixMemoryLocation(diag) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixMemoryLocation(offd) = HYPRE_MEMORY_HOST;

   hypre_ParCSRMatrixMigrate(A, hypre_HandleMemoryLocation(hypre_handle()));

   hypre_TFree(nx_part, HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part, HYPRE_MEMORY_HOST);
   hypre_TFree(value,   HYPRE_MEMORY_HOST);

   return (HYPRE_ParCSRMatrix) A;
}
