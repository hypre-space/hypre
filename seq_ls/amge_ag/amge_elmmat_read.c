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





/*****************************************************************************
 * reads element_data in the same form as element_dof; i.e., if
 *  element: dof dof dof ... dof, then
 *
 * element_data:    for each element 
 *          dof_1 dof_2  ... dof_p
 *   dof_1  e_11   e_12  ...  e_1p
 *   dof_2  e_21   e_22  ...  e_2p
 *    .
 *    .
 *    .
 *   dof_p  e_p1   e_p2  ...  e_pp
 *
 *  contains: 
 *  ...., e_11, e_12,...,e_1p, e_21, ..., e_2p, ... e_pp;

 ****************************************************************************/


#include "headers.h" 

int hypre_AMGeElmMatRead(double **element_data_pointer, 
			 int *i_element_dof,
			 int *j_element_dof,
			 int num_elements,

			 char *element_matrix_file)

{
  FILE *f;

  FILE *g;


  int ierr = 0;
  int i,j,k, entry;

  double *element_data;
  int num_entries = 0;

  for (i=0; i < num_elements; i++)
    num_entries+= (i_element_dof[i+1]-i_element_dof[i]) *
      (i_element_dof[i+1]-i_element_dof[i]);

  element_data = hypre_CTAlloc(double, num_entries);

  f = fopen(element_matrix_file, "r");
  /* g = fopen("element_matrix_wrote", "w"); */

  entry = 0;
  for (i=0; i< num_elements; i++)
    {
      for (j=i_element_dof[i]; j< i_element_dof[i+1]; j++)
	{
	  for (k=i_element_dof[i]; k< i_element_dof[i+1]; k++)
	    {
	      fscanf(f, "%le", &element_data[entry]);
	      /* fprintf(g, "%e ", element_data[entry]); */
	      entry++;
	    }
	  /* fprintf(g, "\n"); */
	  fscanf(f, "\n");
	}
      
      /* fprintf(g, "\n"); */
      fscanf(f, "\n"); 
    }

  fclose(f);
  /* fclose(g); */

  *element_data_pointer = element_data;

  return ierr;

}
