/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * hypre_GenerateLaplacian
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_GenerateLaplacian( HYPRE_Int      nx,
                         HYPRE_Int      ny,
                         HYPRE_Int      nz, 
                         HYPRE_Int      P,
                         HYPRE_Int      Q,
                         HYPRE_Int      R,
                         double  *value )
{
   hypre_CSRMatrix *A;

   HYPRE_Int    *A_i;
   HYPRE_Int    *A_j;
   double *A_data;

   HYPRE_Int *global_part;
   HYPRE_Int ix, iy, iz;
   HYPRE_Int p, q, r;
   HYPRE_Int cnt;
   HYPRE_Int num_rows; 
   HYPRE_Int row_index;

   HYPRE_Int nx_size, ny_size, nz_size;

   HYPRE_Int *nx_part;
   HYPRE_Int *ny_part;
   HYPRE_Int *nz_part;

   num_rows = nx*ny*nz;

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

   A_i = hypre_CTAlloc(HYPRE_Int, num_rows+1);

   cnt = 1;
   A_i[0] = 0;
   for (r = 0; r < R; r++)
   {
      for (q = 0; q < Q; q++)
      {
	 for (p = 0; p < P; p++)
	 {
   	    for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   	    {
      	       for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      	       {
         	  for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         	  {
            	     A_i[cnt] = A_i[cnt-1];
            	     A_i[cnt]++;
            	     if (iz > nz_part[r]) 
               		A_i[cnt]++;
            	     else
            	     {
               		if (iz) 
               		{
                  	   A_i[cnt]++;
               		}
            	     }
            	     if (iy > ny_part[q]) 
               		A_i[cnt]++;
            	     else
            	     {
               		if (iy) 
               		{
                  	   A_i[cnt]++;
               		}
            	     }
            	     if (ix > nx_part[p]) 
               		A_i[cnt]++;
            	     else
            	     {
               		if (ix) 
               		{
                  	   A_i[cnt]++; 
               		}
            	     }
            	     if (ix+1 < nx_part[p+1]) 
               		A_i[cnt]++;
            	     else
            	     {
               		if (ix+1 < nx) 
               		{
                  	   A_i[cnt]++; 
               		}
            	     }
            	     if (iy+1 < ny_part[q+1]) 
               		A_i[cnt]++;
            	     else
            	     {
               		if (iy+1 < ny) 
               		{
                  	   A_i[cnt]++;
               		}
            	     }
            	     if (iz+1 < nz_part[r+1]) 
               		A_i[cnt]++;
            	     else
            	     {
               		if (iz+1 < nz) 
               		{
                  	   A_i[cnt]++;
               		}
            	     }
            	     cnt++;
         	  }
      	       }
   	    }
         }
      }
   }

   A_j = hypre_CTAlloc(HYPRE_Int, A_i[num_rows]);
   A_data = hypre_CTAlloc(double, A_i[num_rows]);

   row_index = 0;
   cnt = 0;
   for (r = 0; r < R; r++)
   {
      for (q = 0; q < Q; q++)
      {
         ny_size = ny_part[q+1]-ny_part[q];
	 for (p = 0; p < P; p++)
	 {
            nx_size = nx_part[p+1] - nx_part[p];
   	    for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   	    {
      	       for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      	       {
         	  for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         	  {
            	     A_j[cnt] = row_index;
            	     A_data[cnt++] = value[0];
            	     if (iz > nz_part[r]) 
            	     {
               		A_j[cnt] = row_index-nx_size*ny_size;
               		A_data[cnt++] = value[3];
            	     }
            	     else
            	     {
               	   	if (iz) 
               		{
                  	   A_j[cnt] = map(ix,iy,iz-1,p,q,r-1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  	   A_data[cnt++] = value[3];
               		}
            	     }
            	     if (iy > ny_part[q]) 
            	     {
               		A_j[cnt] = row_index-nx_size;
               		A_data[cnt++] = value[2];
            	     }
            	     else
            	     {
               		if (iy) 
               		{
                  	   A_j[cnt] = map(ix,iy-1,iz,p,q-1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  	   A_data[cnt++] = value[2];
               		}
            	     }
            	     if (ix > nx_part[p]) 
            	     {
               		A_j[cnt] = row_index-1;
               		A_data[cnt++] = value[1];
            	     }
            	     else
            	     {
               		if (ix) 
               		{
                  	   A_j[cnt] = map(ix-1,iy,iz,p-1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  	   A_data[cnt++] = value[1];
               		}
            	     }
            	     if (ix+1 < nx_part[p+1]) 
            	     {
               		A_j[cnt] = row_index+1;
               		A_data[cnt++] = value[1];
            	     }
            	     else
            	     {
               		if (ix+1 < nx) 
               		{
                  	   A_j[cnt] = map(ix+1,iy,iz,p+1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  	   A_data[cnt++] = value[1];
               		}
            	     }
            	     if (iy+1 < ny_part[q+1]) 
            	     {
               		A_j[cnt] = row_index+nx_size;
               		A_data[cnt++] = value[2];
            	     }
            	     else
            	     {
               		if (iy+1 < ny) 
               		{
                     	   A_j[cnt] = map(ix,iy+1,iz,p,q+1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  	   A_data[cnt++] = value[2];
               		}
            	     }
            	     if (iz+1 < nz_part[r+1]) 
            	     {
               		A_j[cnt] = row_index+nx_size*ny_size;
               		A_data[cnt++] = value[3];
            	     }
            	     else
            	     {
               		if (iz+1 < nz) 
               		{
                           A_j[cnt] = map(ix,iy,iz+1,p,q,r+1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  	   A_data[cnt++] = value[3];
               		}
            	     }
            	     row_index++;
         	  }
      	       }
            }
         }
      }
   }

   A = hypre_CSRMatrixCreate(num_rows, num_rows, A_i[num_rows]);

   hypre_CSRMatrixI(A) = A_i;
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_data;

   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);
   hypre_TFree(global_part);

   return A;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
map( HYPRE_Int  ix,
     HYPRE_Int  iy,
     HYPRE_Int  iz,
     HYPRE_Int  p,
     HYPRE_Int  q,
     HYPRE_Int  r,
     HYPRE_Int  P,
     HYPRE_Int  Q,
     HYPRE_Int  R, 
     HYPRE_Int *nx_part,
     HYPRE_Int *ny_part,
     HYPRE_Int *nz_part,
     HYPRE_Int *global_part )
{
   HYPRE_Int nx_local;
   HYPRE_Int ny_local;
   HYPRE_Int ix_local;
   HYPRE_Int iy_local;
   HYPRE_Int iz_local;
   HYPRE_Int global_index;
   HYPRE_Int proc_num;
 
   proc_num = r*P*Q + q*P + p;
   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];
   ix_local = ix - nx_part[p];
   iy_local = iy - ny_part[q];
   iz_local = iz - nz_part[r];
   global_index = global_part[proc_num] 
      + (iz_local*ny_local+iy_local)*nx_local + ix_local;

   return global_index;
}


/*--------------------------------------------------------------------------
 * hypre_GenerateSystemLaplacian
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_GenerateSysLaplacian( HYPRE_Int      nx,
                            HYPRE_Int      ny,
                            HYPRE_Int      nz, 
                            HYPRE_Int      P,
                            HYPRE_Int      Q,
                            HYPRE_Int      R,
                            HYPRE_Int      num_fun,
                            double  *mtrx,
                            double  *value )
{
   hypre_CSRMatrix *A;

   HYPRE_Int    *A_i;
   HYPRE_Int    *A_j;
   double *A_data;

   HYPRE_Int *global_part;
   HYPRE_Int ix, iy, iz;
   HYPRE_Int p, q, r;
   HYPRE_Int cnt;
   HYPRE_Int num_rows, grid_size; 
   HYPRE_Int row_index, row, col;
   HYPRE_Int index; 
   HYPRE_Int i,j;
   HYPRE_Int num_coeffs;
   HYPRE_Int first_j, j_ind; 

   HYPRE_Int nx_size, ny_size, nz_size;


   HYPRE_Int *nx_part;
   HYPRE_Int *ny_part;
   HYPRE_Int *nz_part;

   HYPRE_Int diag_index;
   

   double val;
   

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

   num_rows = grid_size*num_fun;
   
   A_i = hypre_CTAlloc(HYPRE_Int, num_rows+1);

   cnt = 1;
   A_i[0] = 0;
   for (r = 0; r < R; r++)
   {
      for (q = 0; q < Q; q++)
      {
	 for (p = 0; p < P; p++)
         {
            for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
            {
               for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
               {
                  for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
                  {
                     A_i[cnt] = A_i[cnt-1];
                     A_i[cnt] += num_fun;
                     if (iz > nz_part[r]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (iz) 
                        {
                           A_i[cnt] += num_fun;
                        }
                     }
                     if (iy > ny_part[q]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (iy) 
                        {
                           A_i[cnt] += num_fun;
                        }
                     }
                     if (ix > nx_part[p]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (ix) 
                        {
                           A_i[cnt] += num_fun; 
                        }
                     }
                     if (ix+1 < nx_part[p+1]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (ix+1 < nx) 
                        {
                           A_i[cnt] += num_fun; 
                        }
                     }
                     if (iy+1 < ny_part[q+1]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (iy+1 < ny) 
                        {
                           A_i[cnt] += num_fun;
                        }
                     }
                     if (iz+1 < nz_part[r+1]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (iz+1 < nz) 
                        {
                           A_i[cnt] += num_fun;
                        }
                     }

                     
                     num_coeffs = A_i[cnt]-A_i[cnt-1];
                     cnt++;
                     
                     for (i=1; i < num_fun; i++)
                     {
                        A_i[cnt] = A_i[cnt-1]+num_coeffs;
                        cnt++;
                     }
                  }
               }
            }
         }
      }
   }



   A_j = hypre_CTAlloc(HYPRE_Int, A_i[num_rows]);
   A_data = hypre_CTAlloc(double, A_i[num_rows]);

   row_index = 0;

   for (r = 0; r < R; r++)
   {
      for (q = 0; q < Q; q++)
      {
	 for (p = 0; p < P; p++)
	 {

            for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
            {
               for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
               {
                  for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
                  {
                     cnt = A_i[row_index];
                     num_coeffs = A_i[row_index+1]-A_i[row_index];
                     
                     first_j = row_index;
                     for (i=0; i < num_fun; i++)
                     {
                        for (j=0; j < num_fun; j++)
                        {
                           j_ind = cnt+i*num_coeffs+j;
                           A_j[j_ind] = first_j+j;
                           A_data[j_ind] = value[0]*mtrx[i*num_fun+j];
                        }
                     }
                     cnt += num_fun;
                     if (iz > nz_part[r]) 
                     {
                        first_j = row_index-nx_size*ny_size*num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[3]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (iz) 
                        {
                           first_j = num_fun*map(ix,iy,iz-1,p,q,r-1,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[3]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (iy > ny_part[q]) 
                     {
                        first_j = row_index-nx_size*num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[2]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (iy) 
                        {
                           first_j = num_fun*map(ix,iy-1,iz,p,q-1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[2]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (ix > nx_part[p]) 
                     {
                        first_j = row_index-num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[1]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (ix) 
                        {
                           first_j = num_fun*map(ix-1,iy,iz,p-1,q,r,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[1]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (ix+1 < nx_part[p+1]) 
                     {
                        first_j = row_index+num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[1]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (ix+1 < nx) 
                        {
                           first_j = num_fun*map(ix+1,iy,iz,p+1,q,r,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[1]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (iy+1 < ny_part[q+1]) 
                     {
                        first_j = row_index+nx_size*num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[2]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (iy+1 < ny) 
                        {
                           first_j = num_fun*map(ix,iy+1,iz,p,q+1,r,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[2]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (iz+1 < nz_part[r+1]) 
                     {
                        first_j = row_index+nx_size*ny_size*num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[3]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (iz+1 < nz) 
                        {
                           first_j = num_fun*map(ix,iy,iz+1,p,q,r+1,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[3]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     row_index += num_fun;
                  }
               }
            }
            
         }
         
      }
      
   }
   

   for (i=0; i < 2; i++)
      global_part[i] *= num_fun;

   for (j=1; j< num_fun; j++)
   {
      for (i=0; i<grid_size; i++)
      {
	  row = i*num_fun+j;
	  diag_index = A_i[row];
	  index = diag_index+j;
	  val = A_data[diag_index];
	  col = A_j[diag_index];
	  A_data[diag_index] = A_data[index];
	  A_j[diag_index] = A_j[index];
	  A_data[index] = val;
	  A_j[index] = col;
      }
   }

   A = hypre_CSRMatrixCreate(num_rows, num_rows, A_i[num_rows]);

   hypre_CSRMatrixI(A) = A_i;
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_data;


   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);
   hypre_TFree(global_part);

   return A;
}


/*--------------------------------------------------------------------------
 * hypre_GenerateSystemLaplacianVCoef
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_GenerateSysLaplacianVCoef( HYPRE_Int      nx,
                            HYPRE_Int      ny,
                            HYPRE_Int      nz, 
                            HYPRE_Int      P,
                            HYPRE_Int      Q,
                            HYPRE_Int      R,
                            HYPRE_Int      num_fun,
                            double  *mtrx,
                            double  *value )
{
   hypre_CSRMatrix *A;

   HYPRE_Int    *A_i;
   HYPRE_Int    *A_j;
   double *A_data;

   HYPRE_Int *global_part;
   HYPRE_Int ix, iy, iz;
   HYPRE_Int p, q, r;
   HYPRE_Int cnt;
   HYPRE_Int num_rows, grid_size; 
   HYPRE_Int row_index, row, col;
   HYPRE_Int index; 
   HYPRE_Int i,j;
   HYPRE_Int num_coeffs;
   HYPRE_Int first_j, j_ind; 

   HYPRE_Int nx_size, ny_size, nz_size;


   HYPRE_Int *nx_part;
   HYPRE_Int *ny_part;
   HYPRE_Int *nz_part;

   HYPRE_Int diag_index;
   

   double val;
   
   /* for indexing in values */
   HYPRE_Int sz = num_fun*num_fun;
   
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

   num_rows = grid_size*num_fun;
   
   A_i = hypre_CTAlloc(HYPRE_Int, num_rows+1);

   cnt = 1;
   A_i[0] = 0;
   for (r = 0; r < R; r++)
   {
      for (q = 0; q < Q; q++)
      {
	 for (p = 0; p < P; p++)
         {
            for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
            {
               for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
               {
                  for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
                  {
                     A_i[cnt] = A_i[cnt-1];
                     A_i[cnt] += num_fun;
                     if (iz > nz_part[r]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (iz) 
                        {
                           A_i[cnt] += num_fun;
                        }
                     }
                     if (iy > ny_part[q]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (iy) 
                        {
                           A_i[cnt] += num_fun;
                        }
                     }
                     if (ix > nx_part[p]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (ix) 
                        {
                           A_i[cnt] += num_fun; 
                        }
                     }
                     if (ix+1 < nx_part[p+1]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (ix+1 < nx) 
                        {
                           A_i[cnt] += num_fun; 
                        }
                     }
                     if (iy+1 < ny_part[q+1]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (iy+1 < ny) 
                        {
                           A_i[cnt] += num_fun;
                        }
                     }
                     if (iz+1 < nz_part[r+1]) 
                        A_i[cnt] += num_fun;
                     else
                     {
                        if (iz+1 < nz) 
                        {
                           A_i[cnt] += num_fun;
                        }
                     }

                     
                     num_coeffs = A_i[cnt]-A_i[cnt-1];
                     cnt++;
                     
                     for (i=1; i < num_fun; i++)
                     {
                        A_i[cnt] = A_i[cnt-1]+num_coeffs;
                        cnt++;
                     }
                  }
               }
            }
         }
      }
   }



   A_j = hypre_CTAlloc(HYPRE_Int, A_i[num_rows]);
   A_data = hypre_CTAlloc(double, A_i[num_rows]);

   row_index = 0;

   for (r = 0; r < R; r++)
   {
      for (q = 0; q < Q; q++)
      {
	 for (p = 0; p < P; p++)
	 {

            for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
            {
               for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
               {
                  for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
                  {
                     cnt = A_i[row_index];
                     num_coeffs = A_i[row_index+1]-A_i[row_index];
                     
                     first_j = row_index;
                     for (i=0; i < num_fun; i++)
                     {
                        for (j=0; j < num_fun; j++)
                        {
                           j_ind = cnt+i*num_coeffs+j;
                           A_j[j_ind] = first_j+j;
                           A_data[j_ind] = value[0*sz + i*num_fun+j]*mtrx[i*num_fun+j];
                        }
                     }
                     cnt += num_fun;
                     if (iz > nz_part[r]) 
                     {
                        first_j = row_index-nx_size*ny_size*num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[3*sz + i*num_fun+j ]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (iz) 
                        {
                           first_j = num_fun*map(ix,iy,iz-1,p,q,r-1,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[3*sz  + i*num_fun+j ]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (iy > ny_part[q]) 
                     {
                        first_j = row_index-nx_size*num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[2*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (iy) 
                        {
                           first_j = num_fun*map(ix,iy-1,iz,p,q-1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[2*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (ix > nx_part[p]) 
                     {
                        first_j = row_index-num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[1*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (ix) 
                        {
                           first_j = num_fun*map(ix-1,iy,iz,p-1,q,r,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[1*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (ix+1 < nx_part[p+1]) 
                     {
                        first_j = row_index+num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[1*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (ix+1 < nx) 
                        {
                           first_j = num_fun*map(ix+1,iy,iz,p+1,q,r,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[1*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (iy+1 < ny_part[q+1]) 
                     {
                        first_j = row_index+nx_size*num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[2*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (iy+1 < ny) 
                        {
                           first_j = num_fun*map(ix,iy+1,iz,p,q+1,r,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[2*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     if (iz+1 < nz_part[r+1]) 
                     {
                        first_j = row_index+nx_size*ny_size*num_fun;
                        for (i=0; i < num_fun; i++)
                        {
                           for (j=0; j < num_fun; j++)
                           {
                              j_ind = cnt+i*num_coeffs+j;
                              A_j[j_ind] = first_j+j;
                              A_data[j_ind] = value[3*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                           }
                        }
                        cnt += num_fun;
                     }
                     else
                     {
                        if (iz+1 < nz) 
                        {
                           first_j = num_fun*map(ix,iy,iz+1,p,q,r+1,P,Q,R,
                                                       nx_part,ny_part,nz_part,global_part);
                           for (i=0; i < num_fun; i++)
                           {
                              for (j=0; j < num_fun; j++)
                              {
                                 j_ind = cnt+i*num_coeffs+j;
                                 A_j[j_ind] = first_j+j;
                                 A_data[j_ind] = value[3*sz  + i*num_fun+j]*mtrx[i*num_fun+j];
                              }
                           }
                           cnt += num_fun;
                        }
                     }
                     row_index += num_fun;
                  }
               }
            }
            
         }
         
      }
      
   }
   

   for (i=0; i < 2; i++)
      global_part[i] *= num_fun;

   for (j=1; j< num_fun; j++)
   {
      for (i=0; i<grid_size; i++)
      {
	  row = i*num_fun+j;
	  diag_index = A_i[row];
	  index = diag_index+j;
	  val = A_data[diag_index];
	  col = A_j[diag_index];
	  A_data[diag_index] = A_data[index];
	  A_j[diag_index] = A_j[index];
	  A_data[index] = val;
	  A_j[index] = col;
      }
   }

   A = hypre_CSRMatrixCreate(num_rows, num_rows, A_i[num_rows]);

   hypre_CSRMatrixI(A) = A_i;
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_data;


   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);
   hypre_TFree(global_part);

   return A;
}
