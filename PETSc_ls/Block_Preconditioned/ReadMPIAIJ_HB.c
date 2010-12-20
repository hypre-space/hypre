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




#include "sles.h"
#include <stdio.h>
#include "headers.h"

HYPRE_Int ConHB2MPIAIJ ( Mat *A, char *file_name )
{
  FILE       *fp;
  HYPRE_Int         i, j;
  Scalar      *data;
   HYPRE_Int        *ia;
   HYPRE_Int        *ja;
   HYPRE_Int        size;
   

  ierr = MatCreateMPIAIJ(hypre_MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,
         0,PETSC_NULL,0,PETSC_NULL,&(*A) ); CHKERRA(ierr);

   fp = fopen(file_name, "r");
   /* read in junk line */
   hypre_fscanf(fp, "%*[^\n]\n");

   hypre_fscanf(fp, "%d", &size);

   ia = ctalloc(HYPRE_Int, size+1);
   for (j = 0; j < size+1; j++)
      hypre_fscanf(fp, "%d", &ia[j]);

   ja = ctalloc(HYPRE_Int, ia[size]-1);
   for (j = 0; j < ia[size]-1; j++)
      hypre_fscanf(fp, "%d", &ja[j]);

   data = ctalloc(sizeof (Scalar), ia[size]-1);
   for (j = 0; j < ia[size]-1; j++)
      hypre_fscanf(fp, "%le", &data[j]);

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
