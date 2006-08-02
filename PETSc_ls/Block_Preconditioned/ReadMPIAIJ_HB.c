/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "sles.h"
#include <stdio.h>
#include "headers.h"

int ConHB2MPIAIJ ( Mat *A, char *file_name )
{
  FILE       *fp;
  int         i, j;
  Scalar      *data;
   int        *ia;
   int        *ja;
   int        size;
   

  ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,
         0,PETSC_NULL,0,PETSC_NULL,&(*A) ); CHKERRA(ierr);

   fp = fopen(file_name, "r");
   /* read in junk line */
   fscanf(fp, "%*[^\n]\n");

   fscanf(fp, "%d", &size);

   ia = ctalloc(int, size+1);
   for (j = 0; j < size+1; j++)
      fscanf(fp, "%d", &ia[j]);

   ja = ctalloc(int, ia[size]-1);
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%d", &ja[j]);

   data = ctalloc(sizeof (Scalar), ia[size]-1);
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%le", &data[j]);

   fclose(fp);

  /* put the values in with a series of calls to mat set values */
   for (j = 0; j < size; j++) {
      ierr = MatSetValues( A, 1, &j, ia[j+1]-ia[j], &ja[ia[j]], 
                           &data[ia[j]], INSERT_VALUES );
   }

  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);


  return();
}
