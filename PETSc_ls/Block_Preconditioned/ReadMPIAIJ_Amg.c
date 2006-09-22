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



/* Include headers for problem and solver data structure */
#include "BlockJacobiAmgPcKsp.h"


/* Note that while this routine should really only "know"
about Petsc matrices and the format of the files being read,
we need AMG specific knowledge to preallocate enough space
for AMG's coarse grids, etc., and thus this routine also
"knows" about AMG. When AMG is fixed to remove this 
requirement, this routine can revert back. AC 1-97 */

int ReadMPIAIJ_Amg( Mat *A, char *file_name, int *n )
{
   FILE       *fp;
   int         ierr, i, j;
   Scalar      *data, read_Scalar;
   int        *ia;
   int        *ja;
   int        size, nprocs, myrows, mb, me, first_row, last_row;
   int        dnz, onz, read_int, ja_indx;
   


   MPI_Comm_rank( MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   fp = fopen(file_name, "r");
   /* read in junk line */
   fscanf(fp, "%*[^\n]\n");

   fscanf(fp, "%d", &size);
   *n = size;

   myrows = (size-1)/nprocs+1; mb = myrows;
   if( (nprocs-1) == me) {
     /* Last processor gets remainder */
      myrows = size-(nprocs-1)*mb;
   }

   first_row = me*mb; last_row = first_row+myrows-1;

   dnz = 0; onz = 0;

   ia = ctalloc(int, size+1);
   for (j = 0; j < size+1; j++) {
      fscanf(fp, "%d", &read_int);
      ia[j] = read_int-1;
   }


   ja = ctalloc(int, ia[last_row+1]-ia[first_row]);
   ja_indx = 0;

   for (j = 0; j < size; j++) {
      for (i = ia[j]; i < ia[j+1]; i++) {
         fscanf(fp, "%d", &read_int);
         if( (j>=first_row)&&(j<=last_row) ) {
            ja[ja_indx++] = read_int-1;
	    if( (i>=first_row)&&(i<=last_row) ) {
               dnz+=ia[j+1]-ia[j];
            } else {
               onz+=ia[j+1]-ia[j];
	    }
         }
      }
   }

   data = ctalloc( Scalar, ia[last_row+1]-ia[first_row]);
   ja_indx = 0;

   for (j = 0; j < size; j++) {
      for (i = ia[j]; i < ia[j+1]; i++) {
         fscanf(fp, "%le", &read_Scalar);
         if( (j>=first_row)&&(j<=last_row) ) {
            data[ja_indx++] = read_Scalar;
         }
      }
   }

   fclose(fp);


   /* Allocate the structure for the Petsc matrix */

   /* Turn off INODE option in Petsc. This is needed so that internal
      Petsc storage will be compatible with AMG. */
   ierr = OptionsSetValue( "-mat_aij_no_inode", PETSC_NULL ); CHKERRA(ierr);

   /* Set internal Petsc storage to use index-1 indexing. This is needed so that internal
      Petsc storage will be compatible with AMG. */
   ierr = OptionsSetValue( "-mat_aij_oneindex", PETSC_NULL ); CHKERRA(ierr);

   /* Note that we are tricking Petsc into greatly over-allocating the
      amount of space needed so that we can use the extra as
      appended workspace in AMG. */

   ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,myrows, myrows,
         size,size,NDIMA(dnz)/myrows+1,PETSC_NULL,0,PETSC_NULL,&(*A) ); CHKERRA(ierr);


   /* put the values in by rows with a series of calls to MatSetValues */
   ja_indx=0;
   for (j = first_row; j <= last_row; j++) {
      ierr = MatSetValues( *A, 1, &j, ia[j+1]-ia[j], &ja[ja_indx], 
                           &data[ja_indx], INSERT_VALUES ); CHKERRA(ierr);
      ja_indx += ia[j+1]-ia[j];
   }

  ierr = MatAssemblyBegin(*A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(*A,FINAL_ASSEMBLY); CHKERRA(ierr);


  return( 0 );
}
