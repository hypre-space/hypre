/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/


#include "headers.h" 

int hypre_AMGeDomainElementSparseAssemble(int *i_domain_element,
					  int *j_domain_element,
					  int num_domains,

					  int *i_element_chord,
					  int *j_element_chord,
					  double *a_element_chord,

					  int *i_chord_dof, int *j_chord_dof,

					  int **i_domain_chord_pointer, 
					  int **j_domain_chord_pointer,
					  double **a_domain_chord_pointer,

					  int num_elements, int num_chords,
					  int num_dofs)

{
  int ierr = 0;
  int i,j,k,l;
  int *i_domain_chord, *j_domain_chord;
  double *a_domain_chord;

  
  ierr = matrix_matrix_product(&i_domain_chord,
			       &j_domain_chord,

			       i_domain_element, j_domain_element,
			       i_element_chord, j_element_chord,
			       num_domains, num_elements, num_chords);

  /* numeric multiplication: --------------------------------------------*/

  a_domain_chord = hypre_CTAlloc(double, i_domain_chord[num_domains]);


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

  printf("assembled domain sparse matrices: \n");
  for (i=0; i < num_domains; i++)
    {
      if (i_domain_chord[i+1] > i_domain_chord[i])
	{

	  printf("domain %d: num_nonzero_entries: %d \n", i,
		 i_domain_chord[i+1] - i_domain_chord[i]);

	  for (l=i_domain_chord[i]; l < i_domain_chord[i+1]; l++)
	    {
	      k = j_domain_chord[l];
	      if (j_chord_dof[i_chord_dof[k]] == j_chord_dof[i_chord_dof[k]+1])
		printf("(%d,%d): %e\n", j_chord_dof[i_chord_dof[k]],
		       j_chord_dof[i_chord_dof[k]+1], a_domain_chord[l]);
	    }
	  printf("==================================================\n\n");
	}
    }
  

    */


  return ierr;
}
 
