/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_GenerateDifConv
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateDifConv( MPI_Comm comm,
                 HYPRE_BigInt   nx,
                 HYPRE_BigInt   ny,
                 HYPRE_BigInt   nz,
                 HYPRE_Int      P,
                 HYPRE_Int      Q,
                 HYPRE_Int      R,
                 HYPRE_Int      p,
                 HYPRE_Int      q,
                 HYPRE_Int      r,
                 HYPRE_Real  *value )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int  *diag_i;
   HYPRE_Int  *diag_j;
   HYPRE_Real *diag_data;

   HYPRE_Int  *offd_i;
   HYPRE_Int  *offd_j = NULL;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Real *offd_data = NULL;

   HYPRE_BigInt global_part[2];
   HYPRE_BigInt ix, iy, iz;
   HYPRE_Int ip, iq, ir;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows;
   HYPRE_BigInt *col_map_offd = NULL;
   HYPRE_Int row_index;
   HYPRE_Int i, j;

   HYPRE_Int nx_local, ny_local, nz_local;
   HYPRE_Int num_cols_offd;
   HYPRE_BigInt grid_size;

   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;
   HYPRE_BigInt *nz_part;

   HYPRE_Int num_procs;
   HYPRE_Int P_busy, Q_busy, R_busy;

   hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny * nz;

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);
   hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_num_rows = nx_local * ny_local * nz_local;

   ip = p;
   iq = q;
   ir = r;

   global_part[0] = nz_part[ir] * nx * ny + (ny_part[iq] * nx + nx_part[ip] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_num_rows;

   diag_i = hypre_CTAlloc(HYPRE_Int, local_num_rows + 1, HYPRE_MEMORY_HOST);
   offd_i = hypre_CTAlloc(HYPRE_Int, local_num_rows + 1, HYPRE_MEMORY_HOST);

   P_busy = hypre_min(nx, P);
   Q_busy = hypre_min(ny, Q);
   R_busy = hypre_min(nz, R);

   num_cols_offd = 0;
   if (p) { num_cols_offd += ny_local * nz_local; }
   if (p < P_busy - 1) { num_cols_offd += ny_local * nz_local; }
   if (q) { num_cols_offd += nx_local * nz_local; }
   if (q < Q_busy - 1) { num_cols_offd += nx_local * nz_local; }
   if (r) { num_cols_offd += nx_local * ny_local; }
   if (r < R_busy - 1) { num_cols_offd += nx_local * ny_local; }

   if (!local_num_rows) { num_cols_offd = 0; }

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[ir]; iz < nz_part[ir + 1]; iz++)
   {
      for (iy = ny_part[iq];  iy < ny_part[iq + 1]; iy++)
      {
         for (ix = nx_part[ip]; ix < nx_part[ip + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[o_cnt] = offd_i[o_cnt - 1];
            diag_i[cnt]++;
            if (iz > nz_part[ir])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (iz)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iy > ny_part[iq])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (iy)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix > nx_part[ip])
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
            if (ix + 1 < nx_part[ip + 1])
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
            if (iy + 1 < ny_part[iq + 1])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz + 1 < nz_part[ir + 1])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  offd_i[o_cnt]++;
               }
            }
            cnt++;
            o_cnt++;
         }
      }
   }

   diag_j = hypre_CTAlloc(HYPRE_Int,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);
   diag_data = hypre_CTAlloc(HYPRE_Real,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);

   if (offd_i[local_num_rows])
   {
      offd_j = hypre_CTAlloc(HYPRE_Int,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      big_offd_j = hypre_CTAlloc(HYPRE_BigInt, offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_data = hypre_CTAlloc(HYPRE_Real,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[ir]; iz < nz_part[ir + 1]; iz++)
   {
      for (iy = ny_part[iq];  iy < ny_part[iq + 1]; iy++)
      {
         for (ix = nx_part[ip]; ix < nx_part[ip + 1]; ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0];
            if (iz > nz_part[ir])
            {
               diag_j[cnt] = row_index - nx_local * ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (iz)
               {
                  big_offd_j[o_cnt] = hypre_map(ix, iy, iz - 1, ip, iq, ir - 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
            if (iy > ny_part[iq])
            {
               diag_j[cnt] = row_index - nx_local;
               diag_data[cnt++] = value[2];
            }
            else
            {
               if (iy)
               {
                  big_offd_j[o_cnt] = hypre_map(ix, iy - 1, iz, ip, iq - 1, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[2];
               }
            }
            if (ix > nx_part[ip])
            {
               diag_j[cnt] = row_index - 1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = hypre_map(ix - 1, iy, iz, ip - 1, iq, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (ix + 1 < nx_part[ip + 1])
            {
               diag_j[cnt] = row_index + 1;
               diag_data[cnt++] = value[4];
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = hypre_map(ix + 1, iy, iz, ip + 1, iq, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[4];
               }
            }
            if (iy + 1 < ny_part[iq + 1])
            {
               diag_j[cnt] = row_index + nx_local;
               diag_data[cnt++] = value[5];
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_offd_j[o_cnt] = hypre_map(ix, iy + 1, iz, ip, iq + 1, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[5];
               }
            }
            if (iz + 1 < nz_part[ir + 1])
            {
               diag_j[cnt] = row_index + nx_local * ny_local;
               diag_data[cnt++] = value[6];
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_offd_j[o_cnt] = hypre_map(ix, iy, iz + 1, ip, iq, ir + 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[6];
               }
            }
            row_index++;
         }
      }
   }

   if (num_cols_offd)
   {
      col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_cols_offd; i++)
      {
         col_map_offd[i] = big_offd_j[i];
      }

      hypre_BigQsort0(col_map_offd, 0, num_cols_offd - 1);

      for (i = 0; i < num_cols_offd; i++)
         for (j = 0; j < num_cols_offd; j++)
            if (big_offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
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
   hypre_TFree(nz_part, HYPRE_MEMORY_HOST);
   hypre_TFree(big_offd_j, HYPRE_MEMORY_HOST);

   return (HYPRE_ParCSRMatrix) A;
}

