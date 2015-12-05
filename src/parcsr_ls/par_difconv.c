/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * hypre_GenerateDifConv
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix 
GenerateDifConv( MPI_Comm comm,
                 HYPRE_Int      nx,
                 HYPRE_Int      ny,
                 HYPRE_Int      nz, 
                 HYPRE_Int      P,
                 HYPRE_Int      Q,
                 HYPRE_Int      R,
                 HYPRE_Int      p,
                 HYPRE_Int      q,
                 HYPRE_Int      r,
                 double  *value )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int    *diag_i;
   HYPRE_Int    *diag_j;
   double *diag_data;

   HYPRE_Int    *offd_i;
   HYPRE_Int    *offd_j;
   double *offd_data;

   HYPRE_Int *global_part;
   HYPRE_Int ix, iy, iz;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows; 
   HYPRE_Int *col_map_offd;
   HYPRE_Int row_index;
   HYPRE_Int i,j;

   HYPRE_Int nx_local, ny_local, nz_local;
   HYPRE_Int nx_size, ny_size, nz_size;
   HYPRE_Int num_cols_offd;
   HYPRE_Int grid_size;

   HYPRE_Int *nx_part;
   HYPRE_Int *ny_part;
   HYPRE_Int *nz_part;

   HYPRE_Int num_procs, my_id;
   HYPRE_Int P_busy, Q_busy, R_busy;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   grid_size = nx*ny*nz;

   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);
   hypre_GeneratePartitioning(nz,R,&nz_part);

   global_part = hypre_CTAlloc(HYPRE_Int,P*Q*R+1);

   global_part[0] = 0;
   cnt = 1;
   for (iz = 0; iz < R; iz++)
   {
      nz_size = nz_part[iz+1]-nz_part[iz];
      for (iy = 0; iy < Q; iy++)
      {
         ny_size = ny_part[iy+1]-ny_part[iy];
         for (ix = 0; ix < P; ix++)
         {
            nx_size = nx_part[ix+1] - nx_part[ix];
            global_part[cnt] = global_part[cnt-1];
            global_part[cnt++] += nx_size*ny_size*nz_size;
         }
      }
   }

   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];
   nz_local = nz_part[r+1] - nz_part[r];

   my_id = r*(P*Q) + q*P + p;
   num_procs = P*Q*R;

   local_num_rows = nx_local*ny_local*nz_local;
   diag_i = hypre_CTAlloc(HYPRE_Int, local_num_rows+1);
   offd_i = hypre_CTAlloc(HYPRE_Int, local_num_rows+1);

   P_busy = hypre_min(nx,P);
   Q_busy = hypre_min(ny,Q);
   R_busy = hypre_min(nz,R);

   num_cols_offd = 0;
   if (p) num_cols_offd += ny_local*nz_local;
   if (p < P_busy-1) num_cols_offd += ny_local*nz_local;
   if (q) num_cols_offd += nx_local*nz_local;
   if (q < Q_busy-1) num_cols_offd += nx_local*nz_local;
   if (r) num_cols_offd += nx_local*ny_local;
   if (r < R_busy-1) num_cols_offd += nx_local*ny_local;

   if (!local_num_rows) num_cols_offd = 0;

   col_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd);

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt-1];
            offd_i[o_cnt] = offd_i[o_cnt-1];
            diag_i[cnt]++;
            if (iz > nz_part[r]) 
               diag_i[cnt]++;
            else
            {
               if (iz) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iy > ny_part[q]) 
               diag_i[cnt]++;
            else
            {
               if (iy) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix > nx_part[p]) 
               diag_i[cnt]++;
            else
            {
               if (ix) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (ix+1 < nx_part[p+1]) 
               diag_i[cnt]++;
            else
            {
               if (ix+1 < nx) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (iy+1 < ny_part[q+1]) 
               diag_i[cnt]++;
            else
            {
               if (iy+1 < ny) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz+1 < nz_part[r+1]) 
               diag_i[cnt]++;
            else
            {
               if (iz+1 < nz) 
               {
                  offd_i[o_cnt]++;
               }
            }
            cnt++;
            o_cnt++;
         }
      }
   }

   diag_j = hypre_CTAlloc(HYPRE_Int, diag_i[local_num_rows]);
   diag_data = hypre_CTAlloc(double, diag_i[local_num_rows]);

   if (num_procs > 1)
   {
      offd_j = hypre_CTAlloc(HYPRE_Int, offd_i[local_num_rows]);
      offd_data = hypre_CTAlloc(double, offd_i[local_num_rows]);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0];
            if (iz > nz_part[r]) 
            {
               diag_j[cnt] = row_index-nx_local*ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (iz) 
               {
                  offd_j[o_cnt] = hypre_map(ix,iy,iz-1,p,q,r-1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
            if (iy > ny_part[q]) 
            {
               diag_j[cnt] = row_index-nx_local;
               diag_data[cnt++] = value[2];
            }
            else
            {
               if (iy) 
               {
                  offd_j[o_cnt] = hypre_map(ix,iy-1,iz,p,q-1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[2];
               }
            }
            if (ix > nx_part[p]) 
            {
               diag_j[cnt] = row_index-1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix) 
               {
                  offd_j[o_cnt] = hypre_map(ix-1,iy,iz,p-1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (ix+1 < nx_part[p+1]) 
            {
               diag_j[cnt] = row_index+1;
               diag_data[cnt++] = value[4];
            }
            else
            {
               if (ix+1 < nx) 
               {
                  offd_j[o_cnt] = hypre_map(ix+1,iy,iz,p+1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[4];
               }
            }
            if (iy+1 < ny_part[q+1]) 
            {
               diag_j[cnt] = row_index+nx_local;
               diag_data[cnt++] = value[5];
            }
            else
            {
               if (iy+1 < ny) 
               {
                  offd_j[o_cnt] = hypre_map(ix,iy+1,iz,p,q+1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[5];
               }
            }
            if (iz+1 < nz_part[r+1]) 
            {
               diag_j[cnt] = row_index+nx_local*ny_local;
               diag_data[cnt++] = value[6];
            }
            else
            {
               if (iz+1 < nz) 
               {
                  offd_j[o_cnt] = hypre_map(ix,iy,iz+1,p,q,r+1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[6];
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
      for (i=0; i < num_cols_offd; i++)
         col_map_offd[i] = offd_j[i];
   	
      qsort0(col_map_offd, 0, num_cols_offd-1);

      for (i=0; i < num_cols_offd; i++)
         for (j=0; j < num_cols_offd; j++)
            if (offd_j[i] == col_map_offd[j])
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

   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);

   return (HYPRE_ParCSRMatrix) A;
}

