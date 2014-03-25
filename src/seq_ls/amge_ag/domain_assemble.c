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






#include "headers.h" 

HYPRE_Int hypre_AMGeDomainElementSparseAssemble(HYPRE_Int *i_domain_element,
					  HYPRE_Int *j_domain_element,
					  HYPRE_Int num_domains,

					  HYPRE_Int *i_element_chord,
					  HYPRE_Int *j_element_chord,
					  HYPRE_Real *a_element_chord,

					  HYPRE_Int *i_chord_dof, HYPRE_Int *j_chord_dof,

					  HYPRE_Int **i_domain_chord_pointer, 
					  HYPRE_Int **j_domain_chord_pointer,
					  HYPRE_Real **a_domain_chord_pointer,

					  HYPRE_Int num_elements, HYPRE_Int num_chords,
					  HYPRE_Int num_dofs)

{
  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k,l;
  HYPRE_Int *i_domain_chord, *j_domain_chord;
  HYPRE_Real *a_domain_chord;

  
  ierr = matrix_matrix_product(&i_domain_chord,
			       &j_domain_chord,

			       i_domain_element, j_domain_element,
			       i_element_chord, j_element_chord,
			       num_domains, num_elements, num_chords);

  /* numeric multiplication: --------------------------------------------*/

  a_domain_chord = hypre_CTAlloc(HYPRE_Real, i_domain_chord[num_domains]);


  for (i=0; i < i_domain_chord[num_domains]; i++)
    a_domain_chord[i] = 0.e0;

  for (i=0; i < num_domains; i++)
    for (j=i_domain_element[i]; j < i_domain_element[i+1]; j++)
      for (k=i_element_chord[j_domain_element[j]];
	   k<i_element_chord[j_domain_element[j]+1]; k++)
	{
	  for (l=i_domain_chord[i]; l < i_domain_chord[i+1]; l++)
	    if (j_domain_chord[l] == j_element_chord[k])
	      {
		a_domain_chord[l] += a_element_chord[k];
		break;
	      }
	}


  *i_domain_chord_pointer = i_domain_chord;
  *j_domain_chord_pointer = j_domain_chord;
  *a_domain_chord_pointer = a_domain_chord;

  /*

  hypre_printf("assembled domain sparse matrices: \n");
  for (i=0; i < num_domains; i++)
    {
      if (i_domain_chord[i+1] > i_domain_chord[i])
	{

	  hypre_printf("domain %d: num_nonzero_entries: %d \n", i,
		 i_domain_chord[i+1] - i_domain_chord[i]);

	  for (l=i_domain_chord[i]; l < i_domain_chord[i+1]; l++)
	    {
	      k = j_domain_chord[l];
	      if (j_chord_dof[i_chord_dof[k]] == j_chord_dof[i_chord_dof[k]+1])
		hypre_printf("(%d,%d): %e\n", j_chord_dof[i_chord_dof[k]],
		       j_chord_dof[i_chord_dof[k]+1], a_domain_chord[l]);
	    }
	  hypre_printf("==================================================\n\n");
	}
    }
  

    */


  return ierr;
}
 
