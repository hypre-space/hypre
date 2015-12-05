/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * GenerateCoordinates
 *--------------------------------------------------------------------------*/

float *
GenerateCoordinates( MPI_Comm comm,
		     HYPRE_Int      nx,
		     HYPRE_Int      ny,
		     HYPRE_Int      nz, 
		     HYPRE_Int      P,
		     HYPRE_Int      Q,
		     HYPRE_Int      R,
		     HYPRE_Int      p,
		     HYPRE_Int      q,
		     HYPRE_Int      r,
		     HYPRE_Int      coorddim)
{
   HYPRE_Int ix, iy, iz;
   HYPRE_Int cnt;

   HYPRE_Int nx_local, ny_local, nz_local;
   HYPRE_Int local_num_rows;

   HYPRE_Int *nx_part;
   HYPRE_Int *ny_part;
   HYPRE_Int *nz_part;

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
