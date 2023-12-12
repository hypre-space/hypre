/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_GenerateCoordinates
 *--------------------------------------------------------------------------*/

float *
hypre_GenerateCoordinates( MPI_Comm       comm,
                           HYPRE_BigInt   nx,
                           HYPRE_BigInt   ny,
                           HYPRE_BigInt   nz,
                           HYPRE_Int      P,
                           HYPRE_Int      Q,
                           HYPRE_Int      R,
                           HYPRE_Int      p,
                           HYPRE_Int      q,
                           HYPRE_Int      r,
                           HYPRE_Int      coorddim)
{
   HYPRE_UNUSED_VAR(comm);

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

   coord = hypre_CTAlloc(float,  coorddim * local_num_rows, HYPRE_MEMORY_HOST);

   cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            /* set coordinates BM Oct 17, 2006 */
            if (coord)
            {
               if (nx > 1) { coord[cnt++] = ix; }
               if (ny > 1) { coord[cnt++] = iy; }
               if (nz > 1) { coord[cnt++] = iz; }
            }
         }
      }
   }

   hypre_TFree(nx_part, HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part, HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part, HYPRE_MEMORY_HOST);

   return coord;
}
