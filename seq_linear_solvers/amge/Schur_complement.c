/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

/*****************************************************************************
 * computes subdomain Schur complements as dense matrices stored in:
 *
 *   i_Schur_dof_dof,
 *   a_Schur_dof_dof; for each subdomain i: m^2 entries;
 *                    m = i_subdomain_dof[i+1]- i_subdomain_dof[i];
 * 
 ****************************************************************************/


#include "headers.h" 

int hypre_AMGeSchurComplement(int *i_domain_chord,
			      int *j_domain_chord,
			      double *a_domain_chord,

			      int *i_chord_dof, int *j_chord_dof,

			      int *i_domain_dof,
			      int *j_domain_dof,

			      int *i_subdomain_dof,
			      int *j_subdomain_dof,

			      int **i_Schur_dof_dof_pointer,
			      double **a_Schur_dof_dof_pointer,
			      
			      int num_domains, int num_chords, int num_dofs)

{

  int ierr = 0;
  int i,j;

  int i_loc, j_loc, l_loc, k_loc;
  int chord;

  int *i_Schur_dof_dof, *j_Schur_dof_dof;
  double *a_Schur_dof_dof;



  int *i_local_to_global, *i_global_to_local;
  int *first, *second;

  int first_counter, second_counter;

  double *A, *A_11, *A_22, *X_11;

  int max_num_local_dofs = 0;
  int local_dof_counter;
  int Schur_dof_dof_counter;

  int *i_dof_index;

  /* check if subdomain[i] \subset domain[i]: */
  

  i_dof_index = hypre_CTAlloc(int, num_dofs);

  for (i=0; i < num_domains; i++)
    i_dof_index[i] = -1;
  for (i=0; i < num_domains; i++)
    {
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	i_dof_index[j_domain_dof[j]] = 0;

      for (j=i_subdomain_dof[i]; j < i_subdomain_dof[i+1]; j++)
	if (i_dof_index[j_subdomain_dof[j]] < 0)
	  {
	    printf("subdomain %d contains entry %d not in domain %d\n",
		   i, j_subdomain_dof[j], i);
	    
	    hypre_TFree(i_dof_index);
	    return -1;
	  }

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	i_dof_index[j_domain_dof[j]] = -1;

    }


  for (i=0; i < num_domains; i++)
    if (max_num_local_dofs < i_domain_dof[i+1]- i_domain_dof[i])
      max_num_local_dofs = i_domain_dof[i+1]- i_domain_dof[i];



  i_global_to_local = i_dof_index;
  i_local_to_global = hypre_CTAlloc(int, max_num_local_dofs);
  first = hypre_CTAlloc(int, max_num_local_dofs);
  second = hypre_CTAlloc(int, max_num_local_dofs);

  A    = hypre_CTAlloc(double, max_num_local_dofs*max_num_local_dofs);
  A_11 = hypre_CTAlloc(double, max_num_local_dofs*max_num_local_dofs);
  X_11 = hypre_CTAlloc(double, max_num_local_dofs*max_num_local_dofs);
  A_22 = hypre_CTAlloc(double, max_num_local_dofs*max_num_local_dofs);


  i_Schur_dof_dof = hypre_CTAlloc(int, num_domains+1);

  Schur_dof_dof_counter = 0;
  for (i=0;  i < num_domains; i++)
    Schur_dof_dof_counter+= (i_subdomain_dof[i+1]-i_subdomain_dof[i])*
                            (i_subdomain_dof[i+1]-i_subdomain_dof[i]);

  a_Schur_dof_dof = hypre_CTAlloc(double, Schur_dof_dof_counter);

  Schur_dof_dof_counter = 0;


  for (i=0;  i < num_domains; i++)
    {
      local_dof_counter = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  i_local_to_global[local_dof_counter] = j_domain_dof[j];
	  i_global_to_local[j_domain_dof[j]] = local_dof_counter;
	  local_dof_counter++;
	}

      for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
	for (j_loc = 0; j_loc < local_dof_counter; j_loc++)
	  A[j_loc + i_loc * local_dof_counter] = 0.e0;

      for (j=i_domain_chord[i]; j < i_domain_chord[i+1]; j++)
	{
	  chord = j_domain_chord[j];
	  i_loc = i_global_to_local[j_chord_dof[i_chord_dof[chord]]];
	  j_loc = i_global_to_local[j_chord_dof[i_chord_dof[chord]+1]];
	  A[j_loc + i_loc * local_dof_counter] = a_domain_chord[j];
	}

      /* two--by--two block partitioning: ----------------------- */

      first_counter = 0;
      second_counter = 0;
      for (j=i_subdomain_dof[i]; j < i_subdomain_dof[i+1]; j++)
	{
	  second[second_counter] = i_global_to_local[j_subdomain_dof[j]];
	  i_global_to_local[j_subdomain_dof[j]]+=local_dof_counter;
	  second_counter++;
	}

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  if (i_global_to_local[j_domain_dof[j]] < local_dof_counter)
	    {
	      first[first_counter] = i_global_to_local[j_domain_dof[j]];
	      first_counter++;
	    }
	  else
	    i_global_to_local[j_domain_dof[j]]-=local_dof_counter;
	}

     for (i_loc=0; i_loc < first_counter; i_loc++)
	for (j_loc=0; j_loc < first_counter; j_loc++)
	  A_11[j_loc+first_counter*i_loc] = 
	    A[first[j_loc]+first[i_loc]*local_dof_counter];

      if (first_counter > 0)
	ierr = matinv(X_11, A_11, first_counter);

      /*
      printf("matinv_ierr: %d, first_counter: %d\n", ierr, first_counter);
      */

      if (ierr < 0) 
	{
	  printf("ierr in Schur complement.c: %d\n ", ierr);	 
	  /* ----------------------------------------------------------
	  for (i_loc=0; i_loc < first_counter; i_loc++)
	    {
	      printf("\n ");
	      for (j_loc=0; j_loc < first_counter; j_loc++)
		printf("%e ", A[first[j_loc]+first[i_loc]*local_dof_counter]);

	      printf("\n ");	 
	    }
	    -------------------------------------------------------- */	 
	}


      /* compute Schur complement: --------------------------------- */

      for (i_loc=0; i_loc < second_counter; i_loc++)
	for (j_loc=0; j_loc < second_counter; j_loc++)
	  A_22[j_loc + i_loc * second_counter] = 
	    A[second[j_loc] + second[i_loc] * local_dof_counter];

      for (i_loc=0; i_loc < second_counter; i_loc++)
	for (j_loc=0; j_loc < second_counter; j_loc++)
	  for (l_loc=0; l_loc < first_counter; l_loc++)
	    for (k_loc=0; k_loc < first_counter; k_loc++)
	      A_22[j_loc + i_loc * second_counter] -=	
		A[first[l_loc] + second[i_loc] * local_dof_counter]*
		X_11[k_loc + l_loc * first_counter] *
		A[second[j_loc] + first[k_loc] * local_dof_counter];



      i_Schur_dof_dof[i] = Schur_dof_dof_counter;
      for (i_loc=0; i_loc < second_counter; i_loc++)
	for (j_loc=0; j_loc < second_counter; j_loc++)
	  {
	    a_Schur_dof_dof[Schur_dof_dof_counter] = 
	      A_22[j_loc + i_loc * second_counter];

	    Schur_dof_dof_counter++;
	  }

    }

  i_Schur_dof_dof[num_domains] = Schur_dof_dof_counter;
  
  hypre_TFree(i_global_to_local);
  hypre_TFree(i_local_to_global);
  hypre_TFree(first);
  hypre_TFree(second);

  hypre_TFree(A);
  hypre_TFree(A_22);
  hypre_TFree(X_11);
  hypre_TFree(A_11);



  *i_Schur_dof_dof_pointer = i_Schur_dof_dof;
  *a_Schur_dof_dof_pointer = a_Schur_dof_dof;

  return ierr;

}
