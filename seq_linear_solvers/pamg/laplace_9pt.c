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
 * hypre_GenerateLaplacian9pt
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_GenerateLaplacian9pt( int      nx,
                            int      ny,
                            int      P,
                            int      Q,
                            double  *value )
{
   hypre_CSRMatrix *A;

   int *A_i;
   int *A_j;
   double *A_data;

   int *global_part;
   int ix, iy;
   int p, q;
   int cnt;
   int num_rows; 
   int row_index;

   int nx_size, ny_size;

   int *nx_part;
   int *ny_part;

   num_rows = nx*ny;

   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);

   global_part = hypre_CTAlloc(int,P*Q+1);

   global_part[0] = 0;
   cnt = 1;
   for (iy = 0; iy < Q; iy++)
   {
      ny_size = ny_part[iy+1]-ny_part[iy];
      for (ix = 0; ix < P; ix++)
      {
         nx_size = nx_part[ix+1] - nx_part[ix];
         global_part[cnt] = global_part[cnt-1];
         global_part[cnt++] += nx_size*ny_size;
      }
   }

   A_i = hypre_CTAlloc(int,num_rows+1);

   cnt = 0;
   A_i[0] = 0;
   for (q = 0; q < Q; q++)
   {
      for (p=0; p < P; p++)
      {
    	 for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
   	 {
      	    for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
      	    {
               cnt++;
               A_i[cnt] = A_i[cnt-1];
               A_i[cnt]++;
               if (iy > ny_part[q]) 
               {
            	  A_i[cnt]++;
	    	  if (ix > nx_part[p])
	    	  {
	       	     A_i[cnt]++;
	    	  }
	    	  else
	    	  {
	             if (ix) 
		        A_i[cnt]++;
	          }
	          if (ix < nx_part[p+1]-1)
	    	  {
	       	     A_i[cnt]++;
	    	  }
	    	  else
	    	  {
	       	     if (ix+1 < nx) 
		  	A_i[cnt]++;
	    	  }
               }
               else
               {
            	  if (iy) 
            	  {
                     A_i[cnt]++;
	       	     if (ix > nx_part[p])
	       	     {
	          	A_i[cnt]++;
	       	     }
	       	     else if (ix)
	       	     {
	          	A_i[cnt]++;
	       	     }
	             if (ix < nx_part[p+1]-1)
	             {
	          	A_i[cnt]++;
	       	     }
	       	     else if (ix < nx-1)
	       	     {
	          	A_i[cnt]++;
	       	     }
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
               {
            	  A_i[cnt]++;
	    	  if (ix > nx_part[p])
	    	  {
	       	     A_i[cnt]++;
	    	  }
	          else
	    	  {
	             if (ix) 
		  	A_i[cnt]++;
	     	  }
	    	  if (ix < nx_part[p+1]-1)
	    	  {
	       	     A_i[cnt]++;
	    	  }
	    	  else
	    	  {
	             if (ix+1 < nx) 
		  	A_i[cnt]++;
	    	  }
               }
               else
               {
            	  if (iy+1 < ny) 
            	  {
               	     A_i[cnt]++;
	       	     if (ix > nx_part[p])
	       	     {
	          	A_i[cnt]++;
	       	     }
	       	     else if (ix)
	       	     {
	          	A_i[cnt]++;
	             }
	       	     if (ix < nx_part[p+1]-1)
	       	     {
	          	A_i[cnt]++;
	       	     }
	       	     else if (ix < nx-1)
	       	     {
	          	A_i[cnt]++;
	       	     }
                  }
               }
            }
         }
      }
   }

   A_j = hypre_CTAlloc(int, A_i[num_rows]);
   A_data = hypre_CTAlloc(double, A_i[num_rows]);

   row_index = 0;
   cnt = 0;
   for (q=0; q < Q; q++)
   {
      for (p=0; p < P; p++)
      {
	 for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
   	 {
            for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
      	    {
	       nx_size = nx_part[p+1]-nx_part[p];
               A_j[cnt] = row_index;
               A_data[cnt++] = value[0];
               if (iy > ny_part[q]) 
               {
	    	  if (ix > nx_part[p])
	    	  {
	       	     A_j[cnt] = row_index-nx_size-1 ;
                     A_data[cnt++] = value[1];
	          }
	          else
	    	  {
	             if (ix) 
	       	     { 
                  	A_j[cnt] = map2(ix-1,iy-1,p-1,q,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	       	     } 
	          }
                  A_j[cnt] = row_index-nx_size;
            	  A_data[cnt++] = value[1];
	          if (ix < nx_part[p+1]-1)
	    	  {
	             A_j[cnt] = row_index-nx_size+1 ;
                     A_data[cnt++] = value[1];
	          }
	          else
	          {
	             if (ix+1 < nx)
	             { 
		        A_j[cnt] = map2(ix+1,iy-1,p+1,q,P,Q,
                                   nx_part,ny_part,global_part);
                        A_data[cnt++] = value[1];
	             } 
	    	  }
               }
               else
               {
            	  if (iy) 
            	  {
	             if (ix > nx_part[p])
	             {
                  	A_j[cnt] = map2(ix-1,iy-1,p,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	       	     }
	             else if (ix)
	       	     {
                  	A_j[cnt] = map2(ix-1,iy-1,p-1,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	       	     }
               	     A_j[cnt] = map2(ix,iy-1,p,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                     A_data[cnt++] = value[1];
	             if (ix < nx_part[p+1]-1)
	       	     {
                  	A_j[cnt] = map2(ix+1,iy-1,p,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	             }
	             else if (ix+1 < nx)
	             {
                        A_j[cnt] = map2(ix+1,iy-1,p+1,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	             }
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
               	     A_j[cnt] = map2(ix-1,iy,p-1,q,P,Q,
                                   nx_part,ny_part,global_part);
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
               	     A_j[cnt] = map2(ix+1,iy,p+1,q,P,Q,
                                   nx_part,ny_part,global_part);
                     A_data[cnt++] = value[1];
                  }
               }
               if (iy+1 < ny_part[q+1]) 
               {
	    	  if (ix > nx_part[p])
	    	  {
	             A_j[cnt] = row_index+nx_size-1 ;
                     A_data[cnt++] = value[1];
	          }
	          else
	          {
	             if (ix) 
               	     {
                  	A_j[cnt] = map2(ix-1,iy+1,p-1,q,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
               	     }
            	  }
            	  A_j[cnt] = row_index+nx_size;
            	  A_data[cnt++] = value[1];
	    	  if (ix < nx_part[p+1]-1)
	    	  {
	             A_j[cnt] = row_index+nx_size+1 ;
                     A_data[cnt++] = value[1];
	    	  }
	    	  else
	    	  {
	             if (ix+1 < nx)
	       	     { 
                  	A_j[cnt] = map2(ix+1,iy+1,p+1,q,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	       	     } 
	    	  }
               }
               else
               {
                  if (iy+1 < ny) 
            	  {
	             if (ix > nx_part[p])
	             {
                  	A_j[cnt] = map2(ix-1,iy+1,p,q+1,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	             }
	       	     else if (ix)
	       	     {
                  	A_j[cnt] = map2(ix-1,iy+1,p-1,q+1,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	             }
                     A_j[cnt] = map2(ix,iy+1,p,q+1,P,Q,
                                   nx_part,ny_part,global_part);
               	     A_data[cnt++] = value[1];
	             if (ix < nx_part[p+1]-1)
	       	     {
                  	A_j[cnt] = map2(ix+1,iy+1,p,q+1,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	             }
	       	     else if (ix < nx-1)
	       	     {
                  	A_j[cnt] = map2(ix+1,iy+1,p+1,q+1,P,Q,
                                   nx_part,ny_part,global_part);
                  	A_data[cnt++] = value[1];
	             }
                  }
               }
               row_index++;
      	    }
         }
      }
   }

   A = hypre_CreateCSRMatrix(num_rows, num_rows, A_i[num_rows]);

   hypre_CSRMatrixI(A) = A_i;
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_data;

   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(global_part);

   return A;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
map2( int  ix,
      int  iy,
      int  p,
      int  q,
      int  P,
      int  Q,
      int *nx_part,
      int *ny_part,
      int *global_part )
{
   int nx_local;
   int ix_local;
   int iy_local;
   int global_index;
   int proc_num;
 
   proc_num = q*P + p;
   nx_local = nx_part[p+1] - nx_part[p];
   ix_local = ix - nx_part[p];
   iy_local = iy - ny_part[q];
   global_index = global_part[proc_num] 
      + iy_local*nx_local + ix_local;

   return global_index;
}
