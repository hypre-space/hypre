/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/
 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * hypre_GenerateLaplacian
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_GenerateLaplacian( int      nx,
                         int      ny,
                         int      nz, 
                         double  *value )
{
   hypre_CSRMatrix *A;

   int    *A_i;
   int    *A_j;
   double *A_data;

   int ix, iy, iz;
   int cnt;
   int num_rows; 
   int row_index;

   num_rows = nx * ny *nz;
   A_i = hypre_CTAlloc(int, num_rows+1);

   cnt = 1;
   A_i[0] = 0;
   for (iz = 0; iz < nz; iz++)
   {
      for (iy = 0;  iy < ny; iy++)
      {
         for (ix = 0; ix < nx; ix++)
         {
            A_i[cnt] = A_i[cnt-1];
            A_i[cnt]++;
            if (iz > 0) 
            {
               A_i[cnt]++;
            }
            if (iy > 0) 
            {
               A_i[cnt]++;
            }
            if (ix > 0) 
            {
               A_i[cnt]++;
            }
            if (ix+1 < nx) 
            {
               A_i[cnt]++;
            }
            if (iy+1 < ny) 
            {
               A_i[cnt]++;
            }
            if (iz+1 < nz) 
            {
               A_i[cnt]++;
            }
            cnt++;
         }
      }
   }

   A_j = hypre_CTAlloc(int, A_i[num_rows]);
   A_data = hypre_CTAlloc(double, A_i[num_rows]);

   row_index = 0;
   cnt = 0;
   for (iz = 0; iz < nz; iz++)
   {
      for (iy = 0;  iy < ny; iy++)
      {
         for (ix = 0; ix < nx; ix++)
         {
            A_j[cnt] = row_index;
            A_data[cnt++] = value[0];
            if (iz > 0) 
            {
               A_j[cnt] = row_index-nx*ny;
               A_data[cnt++] = value[3];
            }
            if (iy > 0) 
            {
               A_j[cnt] = row_index-nx;
               A_data[cnt++] = value[2];
            }
            if (ix > 0) 
            {
               A_j[cnt] = row_index-1;
               A_data[cnt++] = value[1];
            }
            if (ix+1 < nx) 
            {
               A_j[cnt] = row_index+1;
               A_data[cnt++] = value[1];
            }
            if (iy+1 < ny) 
            {
               A_j[cnt] = row_index+nx;
               A_data[cnt++] = value[2];
            }
            if (iz+1 < nz) 
            {
               A_j[cnt] = row_index+nx*ny;
               A_data[cnt++] = value[3];
            }
            row_index++;
         }
      }
   }

   /*-----------------------------------------------------------------------
    * Setup matrix structure and return
    *-----------------------------------------------------------------------*/

   A = hypre_CreateCSRMatrix(num_rows, num_rows, A_i[num_rows]);
   hypre_CSRMatrixI(A) = A_i;
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_data;
   hypre_InitializeCSRMatrix(A);

   return A;
}

