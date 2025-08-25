/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

static inline HYPRE_BigInt
map3(HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_BigInt iz,
     HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
     HYPRE_BigInt nx, HYPRE_BigInt ny,
     HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part, HYPRE_BigInt *nz_part)
{
   HYPRE_Int    nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   HYPRE_Int    ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   HYPRE_Int    nz_local = (HYPRE_Int)(nz_part[r + 1] - nz_part[r]);
   HYPRE_Int    ix_local = (HYPRE_Int)(ix - nx_part[p]);
   HYPRE_Int    iy_local = (HYPRE_Int)(iy - ny_part[q]);
   HYPRE_Int    iz_local = (HYPRE_Int)(iz - nz_part[r]);
   HYPRE_BigInt global_index;

   global_index = nz_part[r] * nx * ny +
                  ny_part[q] * nx * (HYPRE_BigInt)nz_local +
                  nx_part[p] * (HYPRE_BigInt)(ny_local * nz_local);
   global_index += (HYPRE_BigInt)((iz_local * ny_local + iy_local) * nx_local + ix_local);
   return global_index;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

static inline HYPRE_BigInt
map2( HYPRE_BigInt  ix, HYPRE_BigInt  iy,
      HYPRE_Int p, HYPRE_Int q,
      HYPRE_BigInt nx, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part)
{
   HYPRE_Int    nx_local, ny_local;
   HYPRE_Int    ix_local, iy_local;
   HYPRE_BigInt global_index;

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   ix_local = (HYPRE_Int)(ix - nx_part[p]);
   iy_local = (HYPRE_Int)(iy - ny_part[q]);
   global_index = ny_part[q] * nx + nx_part[p] * (HYPRE_BigInt)ny_local;
   global_index += (HYPRE_BigInt)(iy_local * nx_local + ix_local);

   return global_index;
}

/*--------------------------------------------------------------------------
 * 7-pt Laplacian operator
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateLaplacian( MPI_Comm       comm,
                   HYPRE_BigInt   nx,
                   HYPRE_BigInt   ny,
                   HYPRE_BigInt   nz,
                   HYPRE_Int      P,
                   HYPRE_Int      Q,
                   HYPRE_Int      R,
                   HYPRE_Int      ip,
                   HYPRE_Int      iq,
                   HYPRE_Int      ir,
                   HYPRE_Real    *value )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int    *diag_i;
   HYPRE_Int    *diag_j;
   HYPRE_Real *diag_data;

   HYPRE_Int    *offd_i;
   HYPRE_Int    *offd_j = NULL;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Real *offd_data = NULL;

   HYPRE_BigInt global_part[2];
   HYPRE_BigInt ix, iy, iz;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows;
   HYPRE_BigInt *col_map_offd;
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

   nx_local = (HYPRE_Int)(nx_part[ip + 1] - nx_part[ip]);
   ny_local = (HYPRE_Int)(ny_part[iq + 1] - ny_part[iq]);
   nz_local = (HYPRE_Int)(nz_part[ir + 1] - nz_part[ir]);

   local_num_rows = nx_local * ny_local * nz_local;

   global_part[0] = nz_part[ir] * nx * ny + (ny_part[iq] * nx + nx_part[ip] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_num_rows;

   diag_i = hypre_CTAlloc(HYPRE_Int, local_num_rows + 1, HYPRE_MEMORY_HOST);
   offd_i = hypre_CTAlloc(HYPRE_Int, local_num_rows + 1, HYPRE_MEMORY_HOST);

   P_busy = hypre_min(nx, P);
   Q_busy = hypre_min(ny, Q);
   R_busy = hypre_min(nz, R);

   num_cols_offd = 0;
   if (ip) { num_cols_offd += ny_local * nz_local; }
   if (ip < P_busy - 1) { num_cols_offd += ny_local * nz_local; }
   if (iq) { num_cols_offd += nx_local * nz_local; }
   if (iq < Q_busy - 1) { num_cols_offd += nx_local * nz_local; }
   if (ir) { num_cols_offd += nx_local * ny_local; }
   if (ir < R_busy - 1) { num_cols_offd += nx_local * ny_local; }

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);

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
            if (iy > ny_part[iq] )
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
            if (ix > nx_part[ip] )
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
                  big_offd_j[o_cnt] = map3(ix, iy, iz - 1, ip, iq, ir - 1, nx, ny,
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
                  big_offd_j[o_cnt] = map3(ix, iy - 1, iz, ip, iq - 1, ir, nx, ny,
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
                  big_offd_j[o_cnt] = map3(ix - 1, iy, iz, ip - 1, iq, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (ix + 1 < nx_part[ip + 1])
            {
               diag_j[cnt] = row_index + 1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = map3(ix + 1, iy, iz, ip + 1, iq, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (iy + 1 < ny_part[iq + 1])
            {
               diag_j[cnt] = row_index + nx_local;
               diag_data[cnt++] = value[2];
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_offd_j[o_cnt] = map3(ix, iy + 1, iz, ip, iq + 1, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[2];
               }
            }
            if (iz + 1 < nz_part[ir + 1])
            {
               diag_j[cnt] = row_index + nx_local * ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_offd_j[o_cnt] = map3(ix, iy, iz + 1, ip, iq, ir + 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
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

      /*for (i=0; i < offd_i[local_num_rows]; i++)
      {
         offd_j[i] = hypre_BigBinarySearch(col_map_offd,big_offd_j[i],num_cols_offd);
      }*/
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

   hypre_TFree(nx_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(big_offd_j,  HYPRE_MEMORY_HOST);

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 * 7-pt systems laplacian operator
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateSysLaplacian( MPI_Comm comm,
                      HYPRE_BigInt   nx,
                      HYPRE_BigInt   ny,
                      HYPRE_BigInt   nz,
                      HYPRE_Int      P,
                      HYPRE_Int      Q,
                      HYPRE_Int      R,
                      HYPRE_Int      p,
                      HYPRE_Int      q,
                      HYPRE_Int      r,
                      HYPRE_Int      num_fun,
                      HYPRE_Real  *mtrx,
                      HYPRE_Real  *value )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int    *diag_i;
   HYPRE_Int    *diag_j;
   HYPRE_Real *diag_data;

   HYPRE_Int    *offd_i;
   HYPRE_Int    *offd_j = NULL;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Real *offd_data = NULL;

   HYPRE_BigInt global_part[2];
   HYPRE_Int ix, iy, iz;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows;
   HYPRE_BigInt *col_map_offd;
   HYPRE_Int row_index, row, col;
   HYPRE_Int index, diag_index;
   HYPRE_Int i, j;

   HYPRE_Int nx_local, ny_local, nz_local;
   HYPRE_Int num_cols_offd;
   HYPRE_BigInt grid_size;
   HYPRE_Int local_grid_size;
   HYPRE_Int first_j, j_ind;
   HYPRE_BigInt big_first_j, big_num_fun = (HYPRE_BigInt)num_fun;
   HYPRE_Int num_coeffs, num_offd_coeffs;

   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;
   HYPRE_BigInt *nz_part;

   HYPRE_Int num_procs;
   HYPRE_Int P_busy, Q_busy, R_busy;
   HYPRE_Real val;
   HYPRE_Int gp_size;

   hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny * nz;

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);
   hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_grid_size = nx_local * ny_local * nz_local;
   local_num_rows = num_fun * local_grid_size;

   global_part[0] = nz_part[r] * nx * ny + (ny_part[q] * nx + nx_part[p] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_grid_size;
   gp_size = 2;

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
   num_cols_offd *= num_fun;

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);

   cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[cnt] = offd_i[cnt - 1];
            diag_i[cnt] += num_fun;
            if (iz > nz_part[r])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iz)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iy > ny_part[q])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iy)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (ix > nx_part[p])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (ix)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            num_coeffs = diag_i[cnt] - diag_i[cnt - 1];
            num_offd_coeffs = offd_i[cnt] - offd_i[cnt - 1];
            cnt++;
            for (i = 1; i < num_fun; i++)
            {
               diag_i[cnt] = diag_i[cnt - 1] + num_coeffs;
               offd_i[cnt] = offd_i[cnt - 1] + num_offd_coeffs;
               cnt++;
            }
         }
      }
   }

   diag_j    = hypre_CTAlloc(HYPRE_Int,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);
   diag_data = hypre_CTAlloc(HYPRE_Real, diag_i[local_num_rows], HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      offd_j     = hypre_CTAlloc(HYPRE_Int,    offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      big_offd_j = hypre_CTAlloc(HYPRE_BigInt, offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_data  = hypre_CTAlloc(HYPRE_Real,   offd_i[local_num_rows], HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            cnt = diag_i[row_index];;
            o_cnt = offd_i[row_index];;
            num_coeffs = diag_i[row_index + 1] - diag_i[row_index];
            num_offd_coeffs = offd_i[row_index + 1] - offd_i[row_index];
            first_j = row_index;
            for (i = 0; i < num_fun; i++)
            {
               for (j = 0; j < num_fun; j++)
               {
                  j_ind = cnt + i * num_coeffs + j;
                  diag_j[j_ind] = first_j + j;
                  diag_data[j_ind] = value[0] * mtrx[i * num_fun + j];
               }
            }
            cnt += num_fun;
            if (iz > nz_part[r])
            {
               first_j = row_index - nx_local * ny_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[3] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iz)
               {
                  big_first_j = big_num_fun * map3(ix, iy, iz - 1, p, q, r - 1, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[3] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iy > ny_part[q])
            {
               first_j = row_index - nx_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[2] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iy)
               {
                  big_first_j = big_num_fun * map3(ix, iy - 1, iz, p, q - 1, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[2] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (ix > nx_part[p])
            {
               first_j = row_index - num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[1] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (ix)
               {
                  big_first_j = big_num_fun * map3(ix - 1, iy, iz, p - 1, q, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[1] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               first_j = row_index + num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[1] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_first_j = big_num_fun * map3(ix + 1, iy, iz, p + 1, q, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[1] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               first_j = row_index + nx_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[2] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_first_j = big_num_fun * map3(ix, iy + 1, iz, p, q + 1, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[2] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               first_j = row_index + nx_local * ny_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[3] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_first_j = big_num_fun * map3(ix, iy, iz + 1, p, q, r + 1, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[3] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            row_index += num_fun;
         }
      }
   }

   if (num_procs > 1)
   {
      cnt = 0;
      for (i = 0; i < local_num_rows; i += num_fun)
      {
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            col_map_offd[cnt++] = big_offd_j[j];
         }
      }

      hypre_BigQsort0(col_map_offd, 0, num_cols_offd - 1);

      for (i = 0; i < num_fun * num_cols_offd; i++)
         for (j = hypre_min(0, hypre_abs(i - num_fun)); j < num_cols_offd; j++)
            if (big_offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
   }

   for (i = 0; i < gp_size; i++)
   {
      global_part[i] *= big_num_fun;
   }

   for (j = 1; j < num_fun; j++)
   {
      for (i = 0; i < local_grid_size; i++)
      {
         row = i * num_fun + j;
         diag_index = diag_i[row];
         index = diag_index + j;
         val = diag_data[diag_index];
         col = diag_j[diag_index];
         diag_data[diag_index] = diag_data[index];
         diag_j[diag_index] = diag_j[index];
         diag_data[index] = val;
         diag_j[index] = col;
      }
   }

   A = hypre_ParCSRMatrixCreate(comm, big_num_fun * grid_size, big_num_fun * grid_size,
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

   hypre_TFree(nx_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(big_offd_j,  HYPRE_MEMORY_HOST);

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 * Systems laplacian with varying diffusion coefficients in each block
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateSysLaplacianVCoef( MPI_Comm       comm,
                           HYPRE_BigInt   nx,
                           HYPRE_BigInt   ny,
                           HYPRE_BigInt   nz,
                           HYPRE_Int      P,
                           HYPRE_Int      Q,
                           HYPRE_Int      R,
                           HYPRE_Int      p,
                           HYPRE_Int      q,
                           HYPRE_Int      r,
                           HYPRE_Int      num_fun,
                           HYPRE_Real    *mtrx,
                           HYPRE_Real    *value )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int    *diag_i;
   HYPRE_Int    *diag_j;
   HYPRE_Real *diag_data;

   HYPRE_Int    *offd_i;
   HYPRE_Int    *offd_j = NULL;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Real *offd_data = NULL;

   HYPRE_BigInt global_part[2];
   HYPRE_BigInt ix, iy, iz;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows;
   HYPRE_BigInt *col_map_offd;
   HYPRE_Int row_index, row, col;
   HYPRE_Int index, diag_index;
   HYPRE_Int i, j;
   HYPRE_Int gp_size;

   HYPRE_Int nx_local, ny_local, nz_local;
   HYPRE_Int num_cols_offd;
   HYPRE_BigInt grid_size;
   HYPRE_Int local_grid_size;
   HYPRE_Int first_j, j_ind;
   HYPRE_BigInt big_first_j, big_num_fun = (HYPRE_BigInt) num_fun;
   HYPRE_Int num_coeffs, num_offd_coeffs;

   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;
   HYPRE_BigInt *nz_part;

   HYPRE_Int num_procs, P_busy, Q_busy, R_busy;
   HYPRE_Real val;

   /* for indexing in values */
   HYPRE_Int sz = num_fun * num_fun;

   hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny * nz;

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);
   hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_grid_size = nx_local * ny_local * nz_local;
   global_part[0] = nz_part[r] * nx * ny + (ny_part[q] * nx + nx_part[p] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_grid_size;
   gp_size = 2;

   local_num_rows = num_fun * local_grid_size;
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
   num_cols_offd *= num_fun;

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);

   cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[cnt] = offd_i[cnt - 1];
            diag_i[cnt] += num_fun;
            if (iz > nz_part[r])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iz)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iy > ny_part[q])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iy)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (ix > nx_part[p])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (ix)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            num_coeffs = diag_i[cnt] - diag_i[cnt - 1];
            num_offd_coeffs = offd_i[cnt] - offd_i[cnt - 1];
            cnt++;
            for (i = 1; i < num_fun; i++)
            {
               diag_i[cnt] = diag_i[cnt - 1] + num_coeffs;
               offd_i[cnt] = offd_i[cnt - 1] + num_offd_coeffs;
               cnt++;
            }
         }
      }
   }

   diag_j    = hypre_CTAlloc(HYPRE_Int,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);
   diag_data = hypre_CTAlloc(HYPRE_Real, diag_i[local_num_rows], HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      offd_j     = hypre_CTAlloc(HYPRE_Int,    offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      big_offd_j = hypre_CTAlloc(HYPRE_BigInt, offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_data  = hypre_CTAlloc(HYPRE_Real,   offd_i[local_num_rows], HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            cnt = diag_i[row_index];;
            o_cnt = offd_i[row_index];;
            num_coeffs = diag_i[row_index + 1] - diag_i[row_index];
            num_offd_coeffs = offd_i[row_index + 1] - offd_i[row_index];
            first_j = row_index;
            for (i = 0; i < num_fun; i++)
            {
               for (j = 0; j < num_fun; j++)
               {
                  j_ind = cnt + i * num_coeffs + j;
                  diag_j[j_ind] = first_j + j;
                  diag_data[j_ind] = value[0 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
               }
            }
            cnt += num_fun;
            if (iz > nz_part[r])
            {
               first_j = row_index - nx_local * ny_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[3 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iz)
               {
                  big_first_j = big_num_fun * map3(ix, iy, iz - 1, p, q, r - 1, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[3 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iy > ny_part[q])
            {
               first_j = row_index - nx_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[2 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iy)
               {
                  big_first_j = big_num_fun * map3(ix, iy - 1, iz, p, q - 1, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[2 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (ix > nx_part[p])
            {
               first_j = row_index - num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[1 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (ix)
               {
                  big_first_j = big_num_fun * map3(ix - 1, iy, iz, p - 1, q, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[1 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               first_j = row_index + num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[1 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_first_j = big_num_fun * map3(ix + 1, iy, iz, p + 1, q, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[1 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               first_j = row_index + nx_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[2 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_first_j = big_num_fun * map3(ix, iy + 1, iz, p, q + 1, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[2 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               first_j = row_index + nx_local * ny_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[3 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_first_j = big_num_fun * map3(ix, iy, iz + 1, p, q, r + 1, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (HYPRE_BigInt)j;
                        offd_data[j_ind] = value[3 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            row_index += num_fun;
         }
      }
   }

   if (num_procs > 1)
   {
      cnt = 0;
      for (i = 0; i < local_num_rows; i += num_fun)
      {
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            col_map_offd[cnt++] = big_offd_j[j];
         }
      }

      hypre_BigQsort0(col_map_offd, 0, num_cols_offd - 1);

      for (i = 0; i < num_fun * num_cols_offd; i++)
         for (j = hypre_min(0, hypre_abs(i - num_fun)); j < num_cols_offd; j++)
            if (big_offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
   }

   for (i = 0; i < gp_size; i++)
   {
      global_part[i] *= num_fun;
   }

   for (j = 1; j < num_fun; j++)
   {
      for (i = 0; i < local_grid_size; i++)
      {
         row = i * num_fun + j;
         diag_index = diag_i[row];
         index = diag_index + j;
         val = diag_data[diag_index];
         col = diag_j[diag_index];
         diag_data[diag_index] = diag_data[index];
         diag_j[diag_index] = diag_j[index];
         diag_data[index] = val;
         diag_j[index] = col;
      }
   }

   A = hypre_ParCSRMatrixCreate(comm, num_fun * grid_size, num_fun * grid_size,
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

   hypre_TFree(nx_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(big_offd_j,  HYPRE_MEMORY_HOST);

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 * 9-pt Laplacian operator
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateLaplacian9pt( MPI_Comm       comm,
                      HYPRE_BigInt   nx,
                      HYPRE_BigInt   ny,
                      HYPRE_Int      P,
                      HYPRE_Int      Q,
                      HYPRE_Int      p,
                      HYPRE_Int      q,
                      HYPRE_Real    *value )
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

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);

   local_num_rows = nx_local * ny_local;

   global_part[0] = ny_part[q] * nx + nx_part[p] * ny_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_num_rows;

   diag_i = hypre_CTAlloc(HYPRE_Int, local_num_rows + 1, HYPRE_MEMORY_HOST);
   offd_i = hypre_CTAlloc(HYPRE_Int, local_num_rows + 1, HYPRE_MEMORY_HOST);

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

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);

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
               if (ix > nx_part[p])
               {
                  offd_i[o_cnt]++;
               }
               else if (ix)
               {
                  offd_i[o_cnt]++;
               }
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

   diag_j = hypre_CTAlloc(HYPRE_Int,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);
   diag_data = hypre_CTAlloc(HYPRE_Real,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      offd_j = hypre_CTAlloc(HYPRE_Int,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_data = hypre_CTAlloc(HYPRE_Real,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      big_offd_j = hypre_CTAlloc(HYPRE_BigInt,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
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
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = map2(ix - 1, iy - 1, p - 1, q, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            diag_j[cnt] = row_index - nx_local;
            diag_data[cnt++] = value[1];
            if (ix < nx_part[p + 1] - 1)
            {
               diag_j[cnt] = row_index - nx_local + 1 ;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = map2(ix + 1, iy - 1, p + 1, q, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
         }
         else
         {
            if (iy)
            {
               if (ix > nx_part[p])
               {
                  big_offd_j[o_cnt] = map2(ix - 1, iy - 1, p, q - 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               else if (ix)
               {
                  big_offd_j[o_cnt] = map2(ix - 1, iy - 1, p - 1, q - 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               big_offd_j[o_cnt] = map2(ix, iy - 1, p, q - 1, nx,
                                        nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
               if (ix < nx_part[p + 1] - 1)
               {
                  big_offd_j[o_cnt] = map2(ix + 1, iy - 1, p, q - 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               else if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = map2(ix + 1, iy - 1, p + 1, q - 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
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
               big_offd_j[o_cnt] = map2(ix - 1, iy, p - 1, q, nx,
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
               big_offd_j[o_cnt] = map2(ix + 1, iy, p + 1, q, nx,
                                        nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
            }
         }
         if (iy + 1 < ny_part[q + 1])
         {
            if (ix > nx_part[p])
            {
               diag_j[cnt] = row_index + nx_local - 1 ;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = map2(ix - 1, iy + 1, p - 1, q, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            diag_j[cnt] = row_index + nx_local;
            diag_data[cnt++] = value[1];
            if (ix < nx_part[p + 1] - 1)
            {
               diag_j[cnt] = row_index + nx_local + 1 ;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = map2(ix + 1, iy + 1, p + 1, q, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
         }
         else
         {
            if (iy + 1 < ny)
            {
               if (ix > nx_part[p])
               {
                  big_offd_j[o_cnt] = map2(ix - 1, iy + 1, p, q + 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               else if (ix)
               {
                  big_offd_j[o_cnt] = map2(ix - 1, iy + 1, p - 1, q + 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               big_offd_j[o_cnt] = map2(ix, iy + 1, p, q + 1, nx,
                                        nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
               if (ix < nx_part[p + 1] - 1)
               {
                  big_offd_j[o_cnt] = map2(ix + 1, iy + 1, p, q + 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               else if (ix < nx - 1)
               {
                  big_offd_j[o_cnt] = map2(ix + 1, iy + 1, p + 1, q + 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
         }
         row_index++;
      }
   }

   if (num_procs > 1)
   {
      HYPRE_BigInt *tmp = hypre_CTAlloc(HYPRE_BigInt, o_cnt, HYPRE_MEMORY_HOST);

      for (i = 0; i < o_cnt; i++)
      {
         tmp[i] = big_offd_j[i];
      }

      hypre_BigQsort0(tmp, 0, o_cnt - 1);

      col_map_offd[0] = tmp[0];
      cnt = 0;
      for (i = 0; i < o_cnt; i++)
      {
         if (tmp[i] > col_map_offd[cnt])
         {
            cnt++;
            col_map_offd[cnt] = tmp[i];
         }
      }

      for (i = 0; i < o_cnt; i++)
      {
         offd_j[i] = hypre_BigBinarySearch(col_map_offd, big_offd_j[i], num_cols_offd);
      }

      hypre_TFree(big_offd_j, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp, HYPRE_MEMORY_HOST);
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

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 * 27-pt laplacian operator
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateLaplacian27pt(MPI_Comm comm,
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

   HYPRE_Int    *diag_i;
   HYPRE_Int    *diag_j;
   HYPRE_Real *diag_data;

   HYPRE_Int    *offd_i;
   HYPRE_Int    *offd_j = NULL;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Real *offd_data = NULL;

   HYPRE_BigInt global_part[2];
   HYPRE_BigInt ix, iy, iz;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows;
   HYPRE_BigInt *col_map_offd;
   HYPRE_BigInt *work;
   HYPRE_Int row_index;
   HYPRE_Int i;

   HYPRE_Int nx_local, ny_local, nz_local;
   HYPRE_Int num_cols_offd;
   HYPRE_Int nxy;
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

   global_part[0] = nz_part[r] * nx * ny + (ny_part[q] * nx + nx_part[p] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_num_rows;

   diag_i = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);
   offd_i = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);

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
   if (p && q) { num_cols_offd += nz_local; }
   if (p && q < Q_busy - 1 ) { num_cols_offd += nz_local; }
   if (p < P_busy - 1 && q ) { num_cols_offd += nz_local; }
   if (p < P_busy - 1 && q < Q_busy - 1 ) { num_cols_offd += nz_local; }
   if (p && r) { num_cols_offd += ny_local; }
   if (p && r < R_busy - 1 ) { num_cols_offd += ny_local; }
   if (p < P_busy - 1 && r ) { num_cols_offd += ny_local; }
   if (p < P_busy - 1 && r < R_busy - 1 ) { num_cols_offd += ny_local; }
   if (q && r) { num_cols_offd += nx_local; }
   if (q && r < R_busy - 1 ) { num_cols_offd += nx_local; }
   if (q < Q_busy - 1 && r ) { num_cols_offd += nx_local; }
   if (q < Q_busy - 1 && r < R_busy - 1 ) { num_cols_offd += nx_local; }
   if (p && q && r) { num_cols_offd++; }
   if (p && q && r < R_busy - 1) { num_cols_offd++; }
   if (p && q < Q_busy - 1 && r) { num_cols_offd++; }
   if (p && q < Q_busy - 1 && r < R_busy - 1) { num_cols_offd++; }
   if (p < P_busy - 1 && q && r) { num_cols_offd++; }
   if (p < P_busy - 1 && q && r < R_busy - 1 ) { num_cols_offd++; }
   if (p < P_busy - 1 && q < Q_busy - 1 && r ) { num_cols_offd++; }
   if (p < P_busy - 1 && q < Q_busy - 1 && r < R_busy - 1) { num_cols_offd++; }

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);

   cnt = 0;
   o_cnt = 0;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r];  iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            cnt++;
            o_cnt++;
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[o_cnt] = offd_i[o_cnt - 1];
            diag_i[cnt]++;
            if (iz > nz_part[r])
            {
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
                     if (ix > nx_part[p])
                     {
                        offd_i[o_cnt]++;
                     }
                     else if (ix)
                     {
                        offd_i[o_cnt]++;
                     }
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
            else
            {
               if (iz)
               {
                  offd_i[o_cnt]++;
                  if (iy > ny_part[q])
                  {
                     offd_i[o_cnt]++;
                     if (ix > nx_part[p])
                     {
                        offd_i[o_cnt]++;
                     }
                     else
                     {
                        if (ix)
                        {
                           offd_i[o_cnt]++;
                        }
                     }
                     if (ix < nx_part[p + 1] - 1)
                     {
                        offd_i[o_cnt]++;
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
                  if (ix > nx_part[p])
                  {
                     offd_i[o_cnt]++;
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
                     offd_i[o_cnt]++;
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
                     offd_i[o_cnt]++;
                     if (ix > nx_part[p])
                     {
                        offd_i[o_cnt]++;
                     }
                     else
                     {
                        if (ix)
                        {
                           offd_i[o_cnt]++;
                        }
                     }
                     if (ix < nx_part[p + 1] - 1)
                     {
                        offd_i[o_cnt]++;
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
                        if (ix > nx_part[p])
                        {
                           offd_i[o_cnt]++;
                        }
                        else if (ix)
                        {
                           offd_i[o_cnt]++;
                        }
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
                  if (ix > nx_part[p])
                  {
                     offd_i[o_cnt]++;
                  }
                  else if (ix)
                  {
                     offd_i[o_cnt]++;
                  }
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
            if (iz + 1 < nz_part[r + 1])
            {
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
                     if (ix > nx_part[p])
                     {
                        offd_i[o_cnt]++;
                     }
                     else if (ix)
                     {
                        offd_i[o_cnt]++;
                     }
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
            else
            {
               if (iz + 1 < nz)
               {
                  offd_i[o_cnt]++;
                  if (iy > ny_part[q])
                  {
                     offd_i[o_cnt]++;
                     if (ix > nx_part[p])
                     {
                        offd_i[o_cnt]++;
                     }
                     else
                     {
                        if (ix)
                        {
                           offd_i[o_cnt]++;
                        }
                     }
                     if (ix < nx_part[p + 1] - 1)
                     {
                        offd_i[o_cnt]++;
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
                  if (ix > nx_part[p])
                  {
                     offd_i[o_cnt]++;
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
                     offd_i[o_cnt]++;
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
                     offd_i[o_cnt]++;
                     if (ix > nx_part[p])
                     {
                        offd_i[o_cnt]++;
                     }
                     else
                     {
                        if (ix)
                        {
                           offd_i[o_cnt]++;
                        }
                     }
                     if (ix < nx_part[p + 1] - 1)
                     {
                        offd_i[o_cnt]++;
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
                        if (ix > nx_part[p])
                        {
                           offd_i[o_cnt]++;
                        }
                        else if (ix)
                        {
                           offd_i[o_cnt]++;
                        }
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
         }
      }
   }

   diag_j = hypre_CTAlloc(HYPRE_Int,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);
   diag_data = hypre_CTAlloc(HYPRE_Real,  diag_i[local_num_rows], HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      big_offd_j = hypre_CTAlloc(HYPRE_BigInt, offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_j = hypre_CTAlloc(HYPRE_Int,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_data = hypre_CTAlloc(HYPRE_Real,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
   }

   nxy = nx_local * ny_local;
   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[r];  iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0];
            if (iz > nz_part[r])
            {
               if (iy > ny_part[q])
               {
                  if (ix > nx_part[p])
                  {
                     diag_j[cnt] = row_index - nxy - nx_local - 1;
                     diag_data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz - 1, p - 1, q, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
                  diag_j[cnt] = row_index - nxy - nx_local;
                  diag_data[cnt++] = value[1];
                  if (ix < nx_part[p + 1] - 1)
                  {
                     diag_j[cnt] = row_index - nxy - nx_local + 1;
                     diag_data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix + 1 < nx)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz - 1, p + 1, q, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
               }
               else
               {
                  if (iy)
                  {
                     if (ix > nx_part[p])
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz - 1, p, q - 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz - 1, p - 1, q - 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     big_offd_j[o_cnt] = map3(ix, iy - 1, iz - 1, p, q - 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                     if (ix < nx_part[p + 1] - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz - 1, p, q - 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else if (ix < nx - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz - 1, p + 1, q - 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
               }
               if (ix > nx_part[p])
               {
                  diag_j[cnt] = row_index - nxy - 1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix)
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy, iz - 1, p - 1, q, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
               diag_j[cnt] = row_index - nxy;
               diag_data[cnt++] = value[1];
               if (ix + 1 < nx_part[p + 1])
               {
                  diag_j[cnt] = row_index - nxy + 1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix + 1 < nx)
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy, iz - 1, p + 1, q, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
               if (iy + 1 < ny_part[q + 1])
               {
                  if (ix > nx_part[p])
                  {
                     diag_j[cnt] = row_index - nxy + nx_local - 1;
                     diag_data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz - 1, p - 1, q, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
                  diag_j[cnt] = row_index - nxy + nx_local;
                  diag_data[cnt++] = value[1];
                  if (ix < nx_part[p + 1] - 1)
                  {
                     diag_j[cnt] = row_index - nxy + nx_local + 1;
                     diag_data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix + 1 < nx)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz - 1, p + 1, q, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
               }
               else
               {
                  if (iy + 1 < ny)
                  {
                     if (ix > nx_part[p])
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz - 1, p, q + 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz - 1, p - 1, q + 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     big_offd_j[o_cnt] = map3(ix, iy + 1, iz - 1, p, q + 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                     if (ix < nx_part[p + 1] - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz - 1, p, q + 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else if (ix < nx - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz - 1, p + 1, q + 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
               }
            }
            else
            {
               if (iz)
               {
                  if (iy > ny_part[q])
                  {
                     if (ix > nx_part[p])
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz - 1, p, q, r - 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else
                     {
                        if (ix)
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz - 1, p - 1, q, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                     big_offd_j[o_cnt] = map3(ix, iy - 1, iz - 1, p, q, r - 1, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                     if (ix < nx_part[p + 1] - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz - 1, p, q, r - 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else
                     {
                        if (ix + 1 < nx)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz - 1, p + 1, q, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                  }
                  else
                  {
                     if (iy)
                     {
                        if (ix > nx_part[p])
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz - 1, p, q - 1, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        else if (ix)
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz - 1, p - 1, q - 1, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        big_offd_j[o_cnt] = map3(ix, iy - 1, iz - 1, p, q - 1, r - 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                        if (ix < nx_part[p + 1] - 1)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz - 1, p, q - 1, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        else if (ix < nx - 1)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz - 1, p + 1, q - 1, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                  }
                  if (ix > nx_part[p])
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy, iz - 1, p, q, r - 1, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  else
                  {
                     if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy, iz - 1, p - 1, q, r - 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
                  big_offd_j[o_cnt] = map3(ix, iy, iz - 1, p, q, r - 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
                  if (ix + 1 < nx_part[p + 1])
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy, iz - 1, p, q, r - 1, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  else
                  {
                     if (ix + 1 < nx)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy, iz - 1, p + 1, q, r - 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
                  if (iy + 1 < ny_part[q + 1])
                  {
                     if (ix > nx_part[p])
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz - 1, p, q, r - 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else
                     {
                        if (ix)
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz - 1, p - 1, q, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                     big_offd_j[o_cnt] = map3(ix, iy + 1, iz - 1, p, q, r - 1, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                     if (ix < nx_part[p + 1] - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz - 1, p, q, r - 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else
                     {
                        if (ix + 1 < nx)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz - 1, p + 1, q, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                  }
                  else
                  {
                     if (iy + 1 < ny)
                     {
                        if (ix > nx_part[p])
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz - 1, p, q + 1, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        else if (ix)
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz - 1, p - 1, q + 1, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        big_offd_j[o_cnt] = map3(ix, iy + 1, iz - 1, p, q + 1, r - 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                        if (ix < nx_part[p + 1] - 1)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz - 1, p, q + 1, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        else if (ix < nx - 1)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz - 1, p + 1, q + 1, r - 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                  }
               }
            }
            if (iy > ny_part[q])
            {
               if (ix > nx_part[p])
               {
                  diag_j[cnt] = row_index - nx_local - 1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix)
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz, p - 1, q, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
               diag_j[cnt] = row_index - nx_local;
               diag_data[cnt++] = value[1];
               if (ix < nx_part[p + 1] - 1)
               {
                  diag_j[cnt] = row_index - nx_local + 1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix + 1 < nx)
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz, p + 1, q, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
            }
            else
            {
               if (iy)
               {
                  if (ix > nx_part[p])
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz, p, q - 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  else if (ix)
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz, p - 1, q - 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  big_offd_j[o_cnt] = map3(ix, iy - 1, iz, p, q - 1, r, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
                  if (ix < nx_part[p + 1] - 1)
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz, p, q - 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  else if (ix < nx - 1)
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz, p + 1, q - 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
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
                  big_offd_j[o_cnt] = map3(ix - 1, iy, iz, p - 1, q, r, nx, ny,
                                                nx_part, ny_part, nz_part);
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
                  big_offd_j[o_cnt] = map3(ix + 1, iy, iz, p + 1, q, r, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               if (ix > nx_part[p])
               {
                  diag_j[cnt] = row_index + nx_local - 1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix)
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz, p - 1, q, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
               diag_j[cnt] = row_index + nx_local;
               diag_data[cnt++] = value[1];
               if (ix < nx_part[p + 1] - 1)
               {
                  diag_j[cnt] = row_index + nx_local + 1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix + 1 < nx)
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz, p + 1, q, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
            }
            else
            {
               if (iy + 1 < ny)
               {
                  if (ix > nx_part[p])
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz, p, q + 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  else if (ix)
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz, p - 1, q + 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  big_offd_j[o_cnt] = map3(ix, iy + 1, iz, p, q + 1, r, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
                  if (ix < nx_part[p + 1] - 1)
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz, p, q + 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  else if (ix < nx - 1)
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz, p + 1, q + 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               if (iy > ny_part[q])
               {
                  if (ix > nx_part[p])
                  {
                     diag_j[cnt] = row_index + nxy - nx_local - 1;
                     diag_data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz + 1, p - 1, q, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
                  diag_j[cnt] = row_index + nxy - nx_local;
                  diag_data[cnt++] = value[1];
                  if (ix < nx_part[p + 1] - 1)
                  {
                     diag_j[cnt] = row_index + nxy - nx_local + 1;
                     diag_data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix + 1 < nx)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz + 1, p + 1, q, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
               }
               else
               {
                  if (iy)
                  {
                     if (ix > nx_part[p])
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz + 1, p, q - 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz + 1, p - 1, q - 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     big_offd_j[o_cnt] = map3(ix, iy - 1, iz + 1, p, q - 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                     if (ix < nx_part[p + 1] - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz + 1, p, q - 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else if (ix < nx - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz + 1, p + 1, q - 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
               }
               if (ix > nx_part[p])
               {
                  diag_j[cnt] = row_index + nxy - 1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix)
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy, iz + 1, p - 1, q, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
               diag_j[cnt] = row_index + nxy;
               diag_data[cnt++] = value[1];
               if (ix + 1 < nx_part[p + 1])
               {
                  diag_j[cnt] = row_index + nxy + 1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix + 1 < nx)
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy, iz + 1, p + 1, q, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
               }
               if (iy + 1 < ny_part[q + 1])
               {
                  if (ix > nx_part[p])
                  {
                     diag_j[cnt] = row_index + nxy + nx_local - 1;
                     diag_data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz + 1, p - 1, q, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
                  diag_j[cnt] = row_index + nxy + nx_local;
                  diag_data[cnt++] = value[1];
                  if (ix < nx_part[p + 1] - 1)
                  {
                     diag_j[cnt] = row_index + nxy + nx_local + 1;
                     diag_data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix + 1 < nx)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz + 1, p + 1, q, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
               }
               else
               {
                  if (iy + 1 < ny)
                  {
                     if (ix > nx_part[p])
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz + 1, p, q + 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz + 1, p - 1, q + 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     big_offd_j[o_cnt] = map3(ix, iy + 1, iz + 1, p, q + 1, r, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                     if (ix < nx_part[p + 1] - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz + 1, p, q + 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else if (ix < nx - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz + 1, p + 1, q + 1, r, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
               }
            }
            else
            {
               if (iz + 1 < nz)
               {
                  if (iy > ny_part[q])
                  {
                     if (ix > nx_part[p])
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz + 1, p, q, r + 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else
                     {
                        if (ix)
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz + 1, p - 1, q, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                     big_offd_j[o_cnt] = map3(ix, iy - 1, iz + 1, p, q, r + 1, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                     if (ix < nx_part[p + 1] - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz + 1, p, q, r + 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else
                     {
                        if (ix + 1 < nx)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz + 1, p + 1, q, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                  }
                  else
                  {
                     if (iy)
                     {
                        if (ix > nx_part[p])
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz + 1, p, q - 1, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        else if (ix)
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy - 1, iz + 1, p - 1, q - 1, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        big_offd_j[o_cnt] = map3(ix, iy - 1, iz + 1, p, q - 1, r + 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                        if (ix < nx_part[p + 1] - 1)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz + 1, p, q - 1, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        else if (ix < nx - 1)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy - 1, iz + 1, p + 1, q - 1, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                  }
                  if (ix > nx_part[p])
                  {
                     big_offd_j[o_cnt] = map3(ix - 1, iy, iz + 1, p, q, r + 1, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  else
                  {
                     if (ix)
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy, iz + 1, p - 1, q, r + 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
                  big_offd_j[o_cnt] = map3(ix, iy, iz + 1, p, q, r + 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
                  if (ix + 1 < nx_part[p + 1])
                  {
                     big_offd_j[o_cnt] = map3(ix + 1, iy, iz + 1, p, q, r + 1, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                  }
                  else
                  {
                     if (ix + 1 < nx)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy, iz + 1, p + 1, q, r + 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                  }
                  if (iy + 1 < ny_part[q + 1])
                  {
                     if (ix > nx_part[p])
                     {
                        big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz + 1, p, q, r + 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else
                     {
                        if (ix)
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz + 1, p - 1, q, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                     big_offd_j[o_cnt] = map3(ix, iy + 1, iz + 1, p, q, r + 1, nx, ny,
                                                   nx_part, ny_part, nz_part);
                     offd_data[o_cnt++] = value[1];
                     if (ix < nx_part[p + 1] - 1)
                     {
                        big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz + 1, p, q, r + 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                     }
                     else
                     {
                        if (ix + 1 < nx)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz + 1, p + 1, q, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                  }
                  else
                  {
                     if (iy + 1 < ny)
                     {
                        if (ix > nx_part[p])
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz + 1, p, q + 1, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        else if (ix)
                        {
                           big_offd_j[o_cnt] = map3(ix - 1, iy + 1, iz + 1, p - 1, q + 1, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        big_offd_j[o_cnt] = map3(ix, iy + 1, iz + 1, p, q + 1, r + 1, nx, ny,
                                                      nx_part, ny_part, nz_part);
                        offd_data[o_cnt++] = value[1];
                        if (ix < nx_part[p + 1] - 1)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz + 1, p, q + 1, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                        else if (ix < nx - 1)
                        {
                           big_offd_j[o_cnt] = map3(ix + 1, iy + 1, iz + 1, p + 1, q + 1, r + 1, nx, ny,
                                                         nx_part, ny_part, nz_part);
                           offd_data[o_cnt++] = value[1];
                        }
                     }
                  }
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
      work = hypre_CTAlloc(HYPRE_BigInt, o_cnt, HYPRE_MEMORY_HOST);

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

      for (i = 0; i < o_cnt; i++)
      {
         offd_j[i] = hypre_BigBinarySearch(col_map_offd, big_offd_j[i], num_cols_offd);
      }

      hypre_TFree(work, HYPRE_MEMORY_HOST);
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

   hypre_TFree(nx_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part,     HYPRE_MEMORY_HOST);
   hypre_TFree(big_offd_j,  HYPRE_MEMORY_HOST);

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
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
                  big_offd_j[o_cnt] = map2(ix - 1, iy - 1, p - 1, q, nx,
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
                  big_offd_j[o_cnt] = map2(ix - 1, iy - 1, p, q - 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
               else if (ix)
               {
                  big_offd_j[o_cnt] = map2(ix - 1, iy - 1, p - 1, q - 1, nx,
                                           nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
               big_offd_j[o_cnt] = map2(ix, iy - 1, p, q - 1, nx,
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
               big_offd_j[o_cnt] = map2(ix - 1, iy, p - 1, q, nx, nx_part, ny_part);
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
               big_offd_j[o_cnt] = map2(ix + 1, iy, p + 1, q, nx, nx_part, ny_part);
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
                  big_offd_j[o_cnt] = map2(ix + 1, iy + 1, p + 1, q, nx, nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
         }
         else
         {
            if (iy + 1 < ny)
            {
               big_offd_j[o_cnt] = map2(ix, iy + 1, p, q + 1, nx, nx_part, ny_part);
               offd_data[o_cnt++] = value[2];
               if (ix < nx_part[p + 1] - 1)
               {
                  big_offd_j[o_cnt] = map2(ix + 1, iy + 1, p, q + 1, nx, nx_part, ny_part);
                  offd_data[o_cnt++] = value[3];
               }
               else if (ix < nx - 1)
               {
                  big_offd_j[o_cnt] = map2(ix + 1, iy + 1, p + 1, q + 1, nx, nx_part, ny_part);
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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

static inline HYPRE_Real
afun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_Real value;
   /* value = 1.0 + 1000.0*hypre_abs(xx-yy); */
   if ((xx < 0.1 && yy < 0.1 && zz < 0.1)
       || (xx < 0.1 && yy < 0.1 && zz > 0.9)
       || (xx < 0.1 && yy > 0.9 && zz < 0.1)
       || (xx > 0.9 && yy < 0.1 && zz < 0.1)
       || (xx > 0.9 && yy > 0.9 && zz < 0.1)
       || (xx > 0.9 && yy < 0.1 && zz > 0.9)
       || (xx < 0.1 && yy > 0.9 && zz > 0.9)
       || (xx > 0.9 && yy > 0.9 && zz > 0.9))
   {
      value = 0.01;
   }
   else if (xx >= 0.1 && xx <= 0.9
            && yy >= 0.1 && yy <= 0.9
            && zz >= 0.1 && zz <= 0.9)
   {
      value = 1000.0;
   }
   else
   {
      value = 1.0 ;
   }
   /* HYPRE_Real value, pi;
   pi = 4.0 * hypre_atan(1.0);
   value = hypre_cos(pi*xx)*hypre_cos(pi*yy); */
   return value;
}

static inline HYPRE_Real
bfun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_Real value;
   /* value = 1.0 + 1000.0*hypre_abs(xx-yy); */
   if ((xx < 0.1 && yy < 0.1 && zz < 0.1)
       || (xx < 0.1 && yy < 0.1 && zz > 0.9)
       || (xx < 0.1 && yy > 0.9 && zz < 0.1)
       || (xx > 0.9 && yy < 0.1 && zz < 0.1)
       || (xx > 0.9 && yy > 0.9 && zz < 0.1)
       || (xx > 0.9 && yy < 0.1 && zz > 0.9)
       || (xx < 0.1 && yy > 0.9 && zz > 0.9)
       || (xx > 0.9 && yy > 0.9 && zz > 0.9))
   {
      value = 0.01;
   }
   else if (xx >= 0.1 && xx <= 0.9
            && yy >= 0.1 && yy <= 0.9
            && zz >= 0.1 && zz <= 0.9)
   {
      value = 1000.0;
   }
   else
   {
      value = 1.0 ;
   }
   /* HYPRE_Real value, pi;
   pi = 4.0 * hypre_atan(1.0);
   value = 1.0 - 2.0*xx;
   value = hypre_cos(pi*xx)*hypre_cos(pi*yy); */
   /* HYPRE_Real value;
   value = 1.0 + 1000.0 * hypre_abs(xx-yy);
   HYPRE_Real value, x0, y0;
   x0 = hypre_abs(xx - 0.5);
   y0 = hypre_abs(yy - 0.5);
   if (y0 > x0) x0 = y0;
   if (x0 >= 0.125 && x0 <= 0.25)
      value = 1.0;
   else
      value = 1000.0;*/
   return value;
}

static inline HYPRE_Real
cfun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_Real value;
   if ((xx < 0.1 && yy < 0.1 && zz < 0.1)
       || (xx < 0.1 && yy < 0.1 && zz > 0.9)
       || (xx < 0.1 && yy > 0.9 && zz < 0.1)
       || (xx > 0.9 && yy < 0.1 && zz < 0.1)
       || (xx > 0.9 && yy > 0.9 && zz < 0.1)
       || (xx > 0.9 && yy < 0.1 && zz > 0.9)
       || (xx < 0.1 && yy > 0.9 && zz > 0.9)
       || (xx > 0.9 && yy > 0.9 && zz > 0.9))
   {
      value = 0.01;
   }
   else if (xx >= 0.1 && xx <= 0.9
            && yy >= 0.1 && yy <= 0.9
            && zz >= 0.1 && zz <= 0.9)
   {
      value = 1000.0;
   }
   else
   {
      value = 1.0 ;
   }
   /*if (xx <= 0.75 && yy <= 0.75 && zz <= 0.75)
      value = 0.1;
   else if (xx > 0.75 && yy > 0.75 && zz > 0.75)
      value = 100000;
   else
      value = 1.0 ;*/
   return value;
}

static inline HYPRE_Real
dfun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   /*HYPRE_Real pi;
   pi = 4.0 * hypre_atan(1.0);
   value = -hypre_sin(pi*xx)*hypre_cos(pi*yy);*/
   value = 0;
   return value;
}

static inline HYPRE_Real
efun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   /*HYPRE_Real pi;
   pi = 4.0 * hypre_atan(1.0);
   value = hypre_sin(pi*yy)*hypre_cos(pi*xx);*/
   value = 0;
   return value;
}

static inline HYPRE_Real
ffun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   value = 0.0;
   return value;
}

static inline HYPRE_Real
gfun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   value = 0.0;
   return value;
}

static inline HYPRE_Real
rfun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   /* HYPRE_Real value, pi;
   pi = 4.0 * hypre_atan(1.0);
   value = -4.0*pi*pi*hypre_sin(pi*xx)*hypre_sin(pi*yy)*hypre_cos(pi*xx)*hypre_cos(pi*yy); */
   HYPRE_Real value;
   /* value = xx*(1.0-xx)*yy*(1.0-yy); */
   value = 1.0;
   return value;
}

static inline HYPRE_Real
bndfun(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   /*HYPRE_Real pi;
   pi = 4.0 * atan(1.0);
   value = hypre_sin(pi*xx)+hypre_sin(13*pi*xx)+hypre_sin(pi*yy)+hypre_sin(13*pi*yy);*/
   value = 0.0;
   return value;
}

static inline HYPRE_Real
afun_rs(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   value = 1.0;
   return value;
}

static inline HYPRE_Real
bfun_rs(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   value = 1.0;
   return value;
}

static inline HYPRE_Real
cfun_rs(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   value = 1.0;
   return value;
}

static inline HYPRE_Real
dfun_rs(HYPRE_Real rs_example, HYPRE_Real rs_l, HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   if (rs_example == 1)
   {
      value = hypre_sin(rs_l * M_PI / 8.0);
   }
   else if (rs_example == 2)
   {
      value = (2.0 * yy - 1.0) * (1.0 - xx * xx);
   }
   else
   {
      value = 4.0 * xx * (xx - 1.0) * (1.0 - 2.0 * yy);
   }
   return value;
}

static inline HYPRE_Real
efun_rs(HYPRE_Real rs_example, HYPRE_Real rs_l, HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   if (rs_example == 1)
   {
      value = hypre_cos(rs_l * M_PI / 8.0);
   }
   else if (rs_example == 2)
   {
      value = 2.0 * xx * yy * (yy - 1.0);
   }
   else
   {
      value = -4.0 * yy * (yy - 1.0) * (1.0 - 2.0 * xx);
   }
   return value;
}

static inline HYPRE_Real
ffun_rs(HYPRE_Real rs_example, HYPRE_Real rs_l, HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_Real value;
   value = efun_rs(rs_example, rs_l, xx, yy, zz);
   return value;
}

static inline HYPRE_Real
gfun_rs(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   value = 0.0;
   return value;
}

static inline HYPRE_Real
rfun_rs(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   value = 1.0;
   return value;
}

static inline HYPRE_Real
bndfun_rs(HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz)
{
   HYPRE_UNUSED_VAR(xx);
   HYPRE_UNUSED_VAR(yy);
   HYPRE_UNUSED_VAR(zz);

   HYPRE_Real value;
   value = 0.0;
   return value;
}

/*--------------------------------------------------------------------------
 * hypre_GenerateDifConv
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateDifConv( MPI_Comm       comm,
                 HYPRE_BigInt   nx,
                 HYPRE_BigInt   ny,
                 HYPRE_BigInt   nz,
                 HYPRE_Int      P,
                 HYPRE_Int      Q,
                 HYPRE_Int      R,
                 HYPRE_Int      p,
                 HYPRE_Int      q,
                 HYPRE_Int      r,
                 HYPRE_Real    *value )
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
                  big_offd_j[o_cnt] = map3(ix, iy, iz - 1, ip, iq, ir - 1, nx, ny,
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
                  big_offd_j[o_cnt] = map3(ix, iy - 1, iz, ip, iq - 1, ir, nx, ny,
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
                  big_offd_j[o_cnt] = map3(ix - 1, iy, iz, ip - 1, iq, ir, nx, ny,
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
                  big_offd_j[o_cnt] = map3(ix + 1, iy, iz, ip + 1, iq, ir, nx, ny,
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
                  big_offd_j[o_cnt] = map3(ix, iy + 1, iz, ip, iq + 1, ir, nx, ny,
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
                  big_offd_j[o_cnt] = map3(ix, iy, iz + 1, ip, iq, ir + 1, nx, ny,
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

/*--------------------------------------------------------------------------
 * hypre_GenerateVarDifConv
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateVarDifConv( MPI_Comm         comm,
                    HYPRE_BigInt     nx,
                    HYPRE_BigInt     ny,
                    HYPRE_BigInt     nz,
                    HYPRE_Int        P,
                    HYPRE_Int        Q,
                    HYPRE_Int        R,
                    HYPRE_Int        p,
                    HYPRE_Int        q,
                    HYPRE_Int        r,
                    HYPRE_Real       eps,
                    HYPRE_ParVector *rhs_ptr)
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag;
   hypre_CSRMatrix    *offd;
   hypre_ParVector    *par_rhs;
   hypre_Vector       *rhs;
   HYPRE_Real         *rhs_data;

   HYPRE_Int          *diag_i;
   HYPRE_Int          *diag_j;
   HYPRE_Real         *diag_data;

   HYPRE_Int          *offd_i = NULL;
   HYPRE_Int          *offd_j = NULL;
   HYPRE_BigInt       *big_offd_j = NULL;
   HYPRE_Real         *offd_data = NULL;

   HYPRE_BigInt        global_part[2];
   HYPRE_BigInt        ix, iy, iz;
   HYPRE_Int           cnt, o_cnt;
   HYPRE_Int           local_num_rows;
   HYPRE_BigInt       *col_map_offd;
   HYPRE_Int           row_index;
   HYPRE_Int           i, j;

   HYPRE_Int           nx_local, ny_local, nz_local;
   HYPRE_Int           num_cols_offd;
   HYPRE_BigInt        grid_size;

   HYPRE_BigInt       *nx_part;
   HYPRE_BigInt       *ny_part;
   HYPRE_BigInt       *nz_part;

   HYPRE_Int           num_procs;
   HYPRE_Int           P_busy, Q_busy, R_busy;

   HYPRE_Real          hhx, hhy, hhz;
   HYPRE_Real          xx, yy, zz;
   HYPRE_Real          afp, afm, bfp, bfm, cfp, cfm, df, ef, ff, gf;

   hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny * nz;

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);
   hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_num_rows = nx_local * ny_local * nz_local;

   global_part[0] = nz_part[r] * nx * ny + (ny_part[q] * nx + nx_part[p] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_num_rows;

   diag_i   = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);
   offd_i   = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);
   rhs_data = hypre_CTAlloc(HYPRE_Real, local_num_rows,   HYPRE_MEMORY_HOST);

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

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt,  num_cols_offd, HYPRE_MEMORY_HOST);

   hhx = 1.0 / (HYPRE_Real)(nx + 1);
   hhy = 1.0 / (HYPRE_Real)(ny + 1);
   hhz = 1.0 / (HYPRE_Real)(nz + 1);

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[o_cnt] = offd_i[o_cnt - 1];
            diag_i[cnt]++;
            if (iz > nz_part[r])
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
            if (iy > ny_part[q])
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
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz + 1 < nz_part[r + 1])
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
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      zz = (HYPRE_Real)(iz + 1) * hhz;
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         yy = (HYPRE_Real)(iy + 1) * hhy;
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            xx = (HYPRE_Real)(ix + 1) * hhx;
            afp = eps * afun(xx + 0.5 * hhx, yy, zz) / hhx / hhx;
            afm = eps * afun(xx - 0.5 * hhx, yy, zz) / hhx / hhx;
            bfp = eps * bfun(xx, yy + 0.5 * hhy, zz) / hhy / hhy;
            bfm = eps * bfun(xx, yy - 0.5 * hhy, zz) / hhy / hhy;
            cfp = eps * cfun(xx, yy, zz + 0.5 * hhz) / hhz / hhz;
            cfm = eps * cfun(xx, yy, zz - 0.5 * hhz) / hhz / hhz;
            df = dfun(xx, yy, zz) / hhx;
            ef = efun(xx, yy, zz) / hhy;
            ff = ffun(xx, yy, zz) / hhz;
            gf = gfun(xx, yy, zz);
            diag_j[cnt] = row_index;
            diag_data[cnt++] = afp + afm + bfp + bfm + cfp + cfm + gf - df - ef - ff;
            rhs_data[row_index] = rfun(xx, yy, zz);
            if (ix == 0) { rhs_data[row_index] += afm * bndfun(0, yy, zz); }
            if (iy == 0) { rhs_data[row_index] += bfm * bndfun(xx, 0, zz); }
            if (iz == 0) { rhs_data[row_index] += cfm * bndfun(xx, yy, 0); }
            if (ix + 1 == nx) { rhs_data[row_index] += (afp - df) * bndfun(1.0, yy, zz); }
            if (iy + 1 == ny) { rhs_data[row_index] += (bfp - ef) * bndfun(xx, 1.0, zz); }
            if (iz + 1 == nz) { rhs_data[row_index] += (cfp - ff) * bndfun(xx, yy, 1.0); }
            if (iz > nz_part[r])
            {
               diag_j[cnt] = row_index - nx_local * ny_local;
               diag_data[cnt++] = -cfm;
            }
            else
            {
               if (iz)
               {
                  big_offd_j[o_cnt] = map3(ix, iy, iz - 1, p, q, r - 1, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = -cfm;
               }
            }
            if (iy > ny_part[q])
            {
               diag_j[cnt] = row_index - nx_local;
               diag_data[cnt++] = -bfm;
            }
            else
            {
               if (iy)
               {
                  big_offd_j[o_cnt] = map3(ix, iy - 1, iz, p, q - 1, r, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = -bfm;
               }
            }
            if (ix > nx_part[p])
            {
               diag_j[cnt] = row_index - 1;
               diag_data[cnt++] = -afm;
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = map3(ix - 1, iy, iz, p - 1, q, r, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = -afm;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               diag_j[cnt] = row_index + 1;
               diag_data[cnt++] = -afp + df;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = map3(ix + 1, iy, iz, p + 1, q, r, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = -afp + df;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               diag_j[cnt] = row_index + nx_local;
               diag_data[cnt++] = -bfp + ef;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_offd_j[o_cnt] = map3(ix, iy + 1, iz, p, q + 1, r, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = -bfp + ef;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               diag_j[cnt] = row_index + nx_local * ny_local;
               diag_data[cnt++] = -cfp + ff;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_offd_j[o_cnt] = map3(ix, iy, iz + 1, p, q, r + 1, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = -cfp + ff;
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
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
      hypre_TFree(big_offd_j, HYPRE_MEMORY_HOST);
   }

   par_rhs = hypre_ParVectorCreate(comm, grid_size, global_part);
   rhs = hypre_ParVectorLocalVector(par_rhs);
   hypre_VectorData(rhs) = rhs_data;
   hypre_VectorMemoryLocation(rhs) = HYPRE_MEMORY_HOST;

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
   hypre_ParVectorMigrate(par_rhs, hypre_HandleMemoryLocation(hypre_handle()));

   hypre_TFree(nx_part, HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part, HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part, HYPRE_MEMORY_HOST);

   *rhs_ptr = (HYPRE_ParVector) par_rhs;

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 * hypre_GenerateVarDifConv: with the FD discretization and examples
 *                           in Ruge-Stuben's paper ``Algebraic Multigrid''
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix
GenerateRSVarDifConv( MPI_Comm         comm,
                      HYPRE_BigInt     nx,
                      HYPRE_BigInt     ny,
                      HYPRE_BigInt     nz,
                      HYPRE_Int        P,
                      HYPRE_Int        Q,
                      HYPRE_Int        R,
                      HYPRE_Int        p,
                      HYPRE_Int        q,
                      HYPRE_Int        r,
                      HYPRE_Real       eps,
                      HYPRE_ParVector *rhs_ptr,
                      HYPRE_Int        type)
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   hypre_ParVector *par_rhs;
   hypre_Vector *rhs;
   HYPRE_Real *rhs_data;

   HYPRE_Int    *diag_i;
   HYPRE_Int    *diag_j;
   HYPRE_Real *diag_data;

   HYPRE_Int    *offd_i = NULL;
   HYPRE_Int    *offd_j = NULL;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Real   *offd_data = NULL;

   HYPRE_BigInt global_part[2];
   HYPRE_BigInt ix, iy, iz;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows;
   HYPRE_BigInt *col_map_offd;
   HYPRE_Int row_index;
   HYPRE_Int i, j;

   HYPRE_Int nx_local, ny_local, nz_local;
   HYPRE_Int num_cols_offd;
   HYPRE_BigInt grid_size;


   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;
   HYPRE_BigInt *nz_part;

   HYPRE_Int num_procs, my_id;
   HYPRE_Int P_busy, Q_busy, R_busy;

   HYPRE_Real hhx, hhy, hhz;
   HYPRE_Real xx, yy, zz;
   HYPRE_Real afp, afm, bfp, bfm, cfp, cfm, di, ai, mux, ei, bi;
   HYPRE_Real muy, fi, ci, muz, dfm, dfp, efm, efp, ffm, ffp, gi;

   HYPRE_Int rs_example = 1;
   HYPRE_Real rs_l = 3.0;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (type >= 1 && type <= 3)
   {
      rs_example = type;
   }

   grid_size = nx * ny * nz;

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);
   hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_num_rows = nx_local * ny_local * nz_local;

   global_part[0] = nz_part[r] * nx * ny + (ny_part[q] * nx + nx_part[p] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (HYPRE_BigInt)local_num_rows;

   diag_i = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);
   offd_i = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);
   rhs_data = hypre_CTAlloc(HYPRE_Real,  local_num_rows, HYPRE_MEMORY_HOST);

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

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt,  num_cols_offd, HYPRE_MEMORY_HOST);

   hhx = 1.0 / (HYPRE_Real)(nx + 1);
   hhy = 1.0 / (HYPRE_Real)(ny + 1);
   hhz = 1.0 / (HYPRE_Real)(nz + 1);

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[o_cnt] = offd_i[o_cnt - 1];
            diag_i[cnt]++;
            if (iz > nz_part[r])
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
            if (iy > ny_part[q])
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
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz + 1 < nz_part[r + 1])
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

   if (num_procs > 1)
   {
      big_offd_j = hypre_CTAlloc(HYPRE_BigInt,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_j = hypre_CTAlloc(HYPRE_Int,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
      offd_data = hypre_CTAlloc(HYPRE_Real,  offd_i[local_num_rows], HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      zz = (HYPRE_Real)(iz + 1) * hhz;
      for (iy = ny_part[q]; iy < ny_part[q + 1]; iy++)
      {
         yy = (HYPRE_Real)(iy + 1) * hhy;
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            xx = (HYPRE_Real)(ix + 1) * hhx;
            afp = -eps * afun_rs(xx + 0.5 * hhx, yy, zz) / hhx / hhx;
            afm = -eps * afun_rs(xx - 0.5 * hhx, yy, zz) / hhx / hhx;
            bfp = -eps * bfun_rs(xx, yy + 0.5 * hhy, zz) / hhy / hhy;
            bfm = -eps * bfun_rs(xx, yy - 0.5 * hhy, zz) / hhy / hhy;
            cfp = -eps * cfun_rs(xx, yy, zz + 0.5 * hhz) / hhz / hhz;
            cfm = -eps * cfun_rs(xx, yy, zz - 0.5 * hhz) / hhz / hhz;
            /* first order terms */
            /* x-direction */
            di = dfun_rs(rs_example, rs_l, xx, yy, zz);
            ai = afun_rs(xx, yy, zz);
            if (di * hhx > eps * ai)
            {
               mux = eps * ai / (2.0 * di * hhx);
            }
            else if (di * hhx < -eps * ai)
            {
               mux = 1.0 + eps * ai / (2.0 * di * hhx);
            }
            else
            {
               mux = 0.5;
            }
            /* y-direction */
            ei = efun_rs(rs_example, rs_l, xx, yy, zz);
            bi = bfun_rs(xx, yy, zz);
            if (ei * hhy > eps * bi)
            {
               muy = eps * bi / (2.0 * ei * hhy);
            }
            else if (ei * hhy < -eps * bi)
            {
               muy = 1.0 + eps * bi / (2.0 * ei * hhy);
            }
            else
            {
               muy = 0.5;
            }
            /* z-direction */
            fi = ffun_rs(rs_example, rs_l, xx, yy, zz);
            ci = cfun_rs(xx, yy, zz);
            if (fi * hhz > eps * ci)
            {
               muz = eps * ci / (2.0 * fi * hhz);
            }
            else if (fi * hhz < -eps * ci)
            {
               muz = 1.0 + eps * ci / (2.0 * fi * hhz);
            }
            else
            {
               muz = 0.5;
            }

            dfm = di * (mux - 1.0) / hhx;
            dfp = di * mux / hhx;
            efm = ei * (muy - 1.0) / hhy;
            efp = ei * muy / hhy;
            ffm = fi * (muz - 1.0) / hhz;
            ffp = fi * muz / hhz;
            gi = gfun_rs(xx, yy, zz);
            /* stencil: center */
            diag_j[cnt] = row_index;
            diag_data[cnt++] = -(afp + afm + bfp + bfm + cfp + cfm  +
                                 dfp + dfm + efp + efm + ffp + ffm) + gi;
            /* rhs vector */
            rhs_data[row_index] = rfun_rs(xx, yy, zz);
            /* apply boundary conditions */
            if (ix == 0) { rhs_data[row_index] -= (afm + dfm) * bndfun_rs(0, yy, zz); }
            if (iy == 0) { rhs_data[row_index] -= (bfm + efm) * bndfun_rs(xx, 0, zz); }
            if (iz == 0) { rhs_data[row_index] -= (cfm + ffm) * bndfun_rs(xx, yy, 0); }
            if (ix + 1 == nx) { rhs_data[row_index] -= (afp + dfp) * bndfun_rs(1.0, yy, zz); }
            if (iy + 1 == ny) { rhs_data[row_index] -= (bfp + efp) * bndfun_rs(xx, 1.0, zz); }
            if (iz + 1 == nz) { rhs_data[row_index] -= (cfp + ffp) * bndfun_rs(xx, yy, 1.0); }
            /* stencil: z- */
            if (iz > nz_part[r])
            {
               diag_j[cnt] = row_index - nx_local * ny_local;
               diag_data[cnt++] = cfm + ffm;
            }
            else
            {
               if (iz)
               {
                  big_offd_j[o_cnt] = map3(ix, iy, iz - 1, p, q, r - 1, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = cfm + ffm;
               }
            }
            /* stencil: y- */
            if (iy > ny_part[q])
            {
               diag_j[cnt] = row_index - nx_local;
               diag_data[cnt++] = bfm + efm;
            }
            else
            {
               if (iy)
               {
                  big_offd_j[o_cnt] = map3(ix, iy - 1, iz, p, q - 1, r, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = bfm + efm;
               }
            }
            /* stencil: x- */
            if (ix > nx_part[p])
            {
               diag_j[cnt] = row_index - 1;
               diag_data[cnt++] = afm + dfm;
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = map3(ix - 1, iy, iz, p - 1, q, r, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = afm + dfm;
               }
            }
            /* stencil: x+ */
            if (ix + 1 < nx_part[p + 1])
            {
               diag_j[cnt] = row_index + 1;
               diag_data[cnt++] = afp + dfp;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = map3(ix + 1, iy, iz, p + 1, q, r, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = afp + dfp;
               }
            }
            /* stencil: y+ */
            if (iy + 1 < ny_part[q + 1])
            {
               diag_j[cnt] = row_index + nx_local;
               diag_data[cnt++] = bfp + efp;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_offd_j[o_cnt] = map3(ix, iy + 1, iz, p, q + 1, r, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = bfp + efp;
               }
            }
            /* stencil: z+ */
            if (iz + 1 < nz_part[r + 1])
            {
               diag_j[cnt] = row_index + nx_local * ny_local;
               diag_data[cnt++] = cfp + ffp;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_offd_j[o_cnt] = map3(ix, iy, iz + 1, p, q, r + 1, nx, ny,
                                           nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = cfp + ffp;
               }
            }
            /* done with this row */
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
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
      hypre_TFree(big_offd_j, HYPRE_MEMORY_HOST);
   }

   par_rhs = hypre_ParVectorCreate(comm, grid_size, global_part);
   rhs = hypre_ParVectorLocalVector(par_rhs);
   hypre_VectorData(rhs) = rhs_data;

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

   hypre_TFree(nx_part, HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part, HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part, HYPRE_MEMORY_HOST);

   *rhs_ptr = (HYPRE_ParVector) par_rhs;

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

float *
GenerateCoordinates(HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                    HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                    HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                    HYPRE_Int coorddim)
{
   HYPRE_BigInt ix, iy, iz;
   HYPRE_Int cnt;

   HYPRE_Int nx_local, ny_local, nz_local;
   HYPRE_Int local_num_rows;

   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;
   HYPRE_BigInt *nz_part;

   float *coord = NULL;

   if (coorddim < 1 || coorddim > 3)
   {
      return NULL;
   }

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);
   hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_num_rows = nx_local * ny_local * nz_local;
   if (!local_num_rows)
   {
      return NULL;
   }

   coord = hypre_TAlloc(float, coorddim * local_num_rows, HYPRE_MEMORY_HOST);

   cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q]; iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            /* set coordinates BM Oct 17, 2006 */
            if (nx > 1) { coord[cnt++] = ix; }
            if (ny > 1) { coord[cnt++] = iy; }
            if (nz > 1) { coord[cnt++] = iz; }
         }
      }
   }

   hypre_TFree(nx_part, HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part, HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part, HYPRE_MEMORY_HOST);

   return coord;
}
