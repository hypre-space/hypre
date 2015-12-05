/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_GenerateStencilMatrix(int    nx,
                            int    ny,
                            int    nz,
                            char  *infile )
{
   hypre_CSRMatrix *A;

   int     *A_i;
   int     *A_j;
   double  *A_data;

   int      grid_size = nx*ny*nz;

   int      stencil_size;
   typedef  int Index[3];
   Index   *stencil_offsets;
   double  *stencil_values;

   int      ix, iy, iz, i, j, k, s, ss;
   int      I, J, jj;

   FILE    *fp;

   /*---------------------------------------------------
    * read in the stencil (diagonal must be first)
    *---------------------------------------------------*/

   fp = fopen(infile, "r");

   fscanf(fp, "%d\n", &stencil_size);
   stencil_offsets = hypre_CTAlloc(Index,  stencil_size);
   stencil_values  = hypre_CTAlloc(double, stencil_size);

   for (s = 0; s < stencil_size; s++)
   {
      fscanf(fp, "%d", &ss);
      fscanf(fp, "%d%d%d %lf\n",
             &stencil_offsets[ss][0],
             &stencil_offsets[ss][1],
             &stencil_offsets[ss][2],
             &stencil_values[ss]);
      printf("%d %d %d %d %f\n", ss,
             stencil_offsets[ss][0],
             stencil_offsets[ss][1],
             stencil_offsets[ss][2],
             stencil_values[ss]);
   }

   fclose(fp);

   /*---------------------------------------------------
    * set up matrix
    *---------------------------------------------------*/

   A_i    = hypre_CTAlloc(int, grid_size + 1);
   A_j    = hypre_CTAlloc(int, grid_size * stencil_size);
   A_data = hypre_CTAlloc(double, grid_size * stencil_size);

   jj = 0;
   for (iz = 0; iz < nz; iz++)
   {
      for (iy = 0; iy < ny; iy++)
      {
         for (ix = 0; ix < nx; ix++)
         {
            I = ix + iy*nx + iz*ny*nz;

            A_i[I] = jj;

            for (s = 0; s < stencil_size; s++)
            {
               i = ix + stencil_offsets[s][0];
               j = iy + stencil_offsets[s][1];
               k = iz + stencil_offsets[s][2];

               if ((i > -1) && (i < nx) &&
                   (j > -1) && (j < ny) &&
                   (k > -1) && (k < nz))
               {
                  J = i + j*nx + k*ny*nz;
                  A_j[jj]    = J;
                  A_data[jj] = stencil_values[s];

                  jj++;
               }
            }
         }
      }
   }
   A_i[grid_size] = jj;

   A = hypre_CSRMatrixCreate(grid_size, grid_size, A_i[grid_size]);

   hypre_CSRMatrixI(A)    = A_i;
   hypre_CSRMatrixJ(A)    = A_j;
   hypre_CSRMatrixData(A) = A_data;

   return A;
}
