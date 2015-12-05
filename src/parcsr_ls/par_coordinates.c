/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.1 $
 ***********************************************************************EHEADER*/



 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * GenerateCoordinates
 *--------------------------------------------------------------------------*/

float *
GenerateCoordinates( MPI_Comm comm,
		     int      nx,
		     int      ny,
		     int      nz, 
		     int      P,
		     int      Q,
		     int      R,
		     int      p,
		     int      q,
		     int      r,
		     int      coorddim)
{
   int ix, iy, iz;
   int cnt;

   int nx_local, ny_local, nz_local;
   int local_num_rows;

   int *nx_part;
   int *ny_part;
   int *nz_part;

   float *coord=NULL;

   if (coorddim<1 || coorddim>3) {
     return NULL;
   }

   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);
   hypre_GeneratePartitioning(nz,R,&nz_part);

   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];
   nz_local = nz_part[r+1] - nz_part[r];

   local_num_rows = nx_local*ny_local*nz_local;
 
   coord = hypre_CTAlloc(float, coorddim*local_num_rows);
     
   cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
     for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
     {
       for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
       {	
	 /* set coordinates BM Oct 17, 2006 */
	 if (coord) {
	   if (nx>1) coord[cnt++] = ix;
	   if (ny>1) coord[cnt++] = iy;
	   if (nz>1) coord[cnt++] = iz;
	 }
       }
     }
   }
   
   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);

   return coord;
}
