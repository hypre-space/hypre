/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
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
