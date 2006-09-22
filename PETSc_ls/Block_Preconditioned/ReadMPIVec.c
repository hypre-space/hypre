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
 * $Revision$
 ***********************************************************************EHEADER*/



/* Include Petsc linear solver headers */
#include "sles.h"

#include <stdio.h>




int ReadMPIVec( Vec *x, char *file_name )
{
  FILE       *fp;
  int         ierr=0, i, j;
  Scalar      read_Scalar;
   int        size, nprocs, myrows, mb, me, first_row, last_row, ja_indx;
   


   MPI_Comm_rank( MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   fp = fopen(file_name, "r");

   if ( !fp ) {
      return(1);
   }

   /* read in junk line */
   fscanf(fp, "%*[^\n]\n");

   fscanf(fp, "%d", &size);

   myrows = (size-1)/nprocs+1; mb = myrows;
   if( (nprocs-1) == me) {
     /* Last processor gets remainder */
      myrows = size-me*mb;
   }

   first_row = me*mb; last_row = first_row+myrows-1;

   /* Allocate the vector structure */
   ierr = VecCreateMPI( MPI_COMM_WORLD, myrows, size, x );

   /* Go through elements and add them one at a time */
   ja_indx = 0;
   for (j = 0; j < size; j++) {
         fscanf(fp, "%le", &read_Scalar);
         if( (j>=first_row)&&(j<=last_row) ) {
            ierr=VecSetValues( *x, 1, &j, &read_Scalar, INSERT_VALUES );
            CHKERRA(ierr);
         }
   }

   fclose(fp);


   ierr = VecAssemblyBegin(*x); CHKERRA(ierr);
   ierr = VecAssemblyEnd(*x); CHKERRA(ierr);


   return(ierr);
}
