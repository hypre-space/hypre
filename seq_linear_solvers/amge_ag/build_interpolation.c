/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/
/*****************************************************************************
 * element matrices stored in format:
 *  i_element_chord,
 *  j_element_chord,
 *  a_element_chord; 
 *
 * here, chord = (i_dof, j_dof) directed pair of indices
 *  for which A(i_dof, j_dof) \ne 0; A is the global assembled matrix;
 * 
 * also,  as input i_chord_dof, j_chord_dof, num_chords;
 *
 * needed graphs: element_dof, dof_element = (element_dof)^T,
 *                AE_element, AEface_dof, dof_AEface = (AEface_dof)^T;
 *
 * builds:
 *        hypre_CSRMatrix P;
 *        (coarseelement matrices: hypre_CSRMatrix)
 *                 coarseelement_coarsecdof;
 *                 coarseelement_coarsechord;
 *        graph:   dof_coarsedof;
 *        graph:   coarsechord_coarsedof;
 ****************************************************************************/
#include "headers.h" 

int hypre_AMGeAGBuildInterpolation(hypre_CSRMatrix     **P_pointer,

				   int **i_coarseelement_coarsedof_pointer,
				   int **j_coarseelement_coarsedof_pointer,


				   int **i_coarseelement_coarsechord_pointer,
				   int **j_coarseelement_coarsechord_pointer,
				   double **a_coarseelement_coarsechord_pointer,

				   int **i_coarsechord_coarsedof_pointer, 
				   int **j_coarsechord_coarsedof_pointer,

				   int *num_coarsechords_pointer,

				   int **i_dof_coarsedof_pointer,
				   int **j_dof_coarsedof_pointer,

				   int *num_coarsedofs_pointer,

				   int *i_element_dof, int *j_element_dof,
				   int *i_dof_element, int *j_dof_element,

				   int *i_element_chord,
				   int *j_element_chord,
				   double *a_element_chord,


				   int *i_AEface_dof, int *j_AEface_dof,
				   int *i_dof_AEface, int *j_dof_AEface,

				   int *i_AE_element, int *j_AE_element,


				   int *i_chord_dof, int *j_chord_dof,
				 

				   int num_chords,

				   int num_AEs, 
				   int num_AEfaces, 

				   int num_elements, 
				   int num_dofs)

{
  int ierr = 0;
  int i,j,k,l;

  int matz = 1;
  int i_loc, j_loc;

  int chord, i_dof, j_dof, i_coarsedof, j_coarsedof;
  
  

  int *i_AEface_element, *j_AEface_element;
  int *i_AEface_dof_dof, *j_AEface_dof_dof;


  int *i_AE_dof, *j_AE_dof;
  
  int *i_dof_coarsedof, *j_dof_coarsedof;
  int num_coarsedofs;

  int *i_AE_coarsedof, *j_AE_coarsedof;
  int *i_AE_coarsedof_coarsedof;
  double *AE_coarsedof_coarsedof;
  int AE_coarsedof_coarsedof_counter;
  
  
  int *i_domain_element, *j_domain_element;
  int *i_domain_chord, *j_domain_chord;
  double *a_domain_chord;

  int *i_AE_chord, *j_AE_chord;
  double *a_AE_chord;

  int *i_domain_dof, *j_domain_dof;
  int *i_subdomain_dof, *j_subdomain_dof;
  int *i_dof_subdomain, *j_dof_subdomain;
  int *i_Schur_dof_dof;
  double *a_Schur_dof_dof, *P_coeff;

  hypre_CSRMatrix  *P;

  int *i_dof_neighbor_coarsedof, *j_dof_neighbor_coarsedof;
  

  int *i_coarseelement_coarsechord, *j_coarseelement_coarsechord;
  double *a_coarseelement_coarsechord;

  int *i_coarsechord_coarsedof, *j_coarsechord_coarsedof;
  int num_coarsechords;

  int AE_coarsedof_counter;
  
  int num_domains;

  int domain_counter, domain_element_counter, subdomain_dof_counter;

  int *i_local_to_global;
  int *i_global_to_local;
    
  int max_local_dof_counter = 0, local_dof_counter, local_coarsedof_counter;

  double *AE, *QE, *W, *Aux1, *Aux2;
  

  int subdomain_coarsedof_counter, 
    dof_neighbor_coarsedof_counter, coarsedof_counter, dof_coarsedof_counter;

  double  row_sum;


  int *i_dof_index;
  


  /* ------------------------------------------------------------------ */
  /* building interior of domains associated with AEs and AEfaces: ---- */
  /* ------------------------------------------------------------------ */
  ierr = matrix_matrix_product(&i_AE_dof, &j_AE_dof,

			       i_AE_element, j_AE_element,
			       i_element_dof, j_element_dof,

			       num_AEs, num_elements, num_dofs);


  num_domains = num_AEs + num_AEfaces;
  i_subdomain_dof = hypre_CTAlloc(int, num_domains+1);
  

  subdomain_dof_counter = 0;
  domain_counter = 0;
  for (i=0; i < num_AEs; i++)
    {
      i_subdomain_dof[domain_counter] = subdomain_dof_counter;
      domain_counter++;
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	/* dof is not on AEface: -------------------------------------------- */
	if (i_dof_AEface[j_AE_dof[j]+1] == i_dof_AEface[j_AE_dof[j]])
	  subdomain_dof_counter++;
    }

  for (i=0; i < num_AEfaces; i++)
    {
      i_subdomain_dof[domain_counter] = subdomain_dof_counter;
      domain_counter++;
      for (j=i_AEface_dof[i]; j < i_AEface_dof[i+1]; j++)
	/* dof is only on AEface i: ------------------------------------------ */
	if (i_dof_AEface[j_AEface_dof[j]+1] == i_dof_AEface[j_AEface_dof[j]]+1)
	  subdomain_dof_counter++;
    }

  i_subdomain_dof[domain_counter] = subdomain_dof_counter;

  j_subdomain_dof = hypre_CTAlloc(int, subdomain_dof_counter);


  subdomain_dof_counter = 0;
  domain_counter = 0;
  for (i=0; i < num_AEs; i++)
    {
      i_subdomain_dof[domain_counter] = subdomain_dof_counter;
      domain_counter++;
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	/* dof is not on AEface: -------------------------------------------- */
	if (i_dof_AEface[j_AE_dof[j]+1] == i_dof_AEface[j_AE_dof[j]])
	  {
	    j_subdomain_dof[subdomain_dof_counter] = j_AE_dof[j];
	    subdomain_dof_counter++;
	  }
    }

  for (i=0; i < num_AEfaces; i++)
    {
      i_subdomain_dof[domain_counter] = subdomain_dof_counter;
      domain_counter++;
      for (j=i_AEface_dof[i]; j < i_AEface_dof[i+1]; j++)
	/* dof is only on AEface i: ------------------------------------------ */
	if (i_dof_AEface[j_AEface_dof[j]+1] == i_dof_AEface[j_AEface_dof[j]]+1)
	  {
	    j_subdomain_dof[subdomain_dof_counter] = j_AEface_dof[j];
	    subdomain_dof_counter++;
	  }  
    }

  i_subdomain_dof[num_domains] = subdomain_dof_counter;






  ierr = matrix_matrix_product(&i_domain_element, &j_domain_element,

			       i_subdomain_dof, j_subdomain_dof,
			       i_dof_element, j_dof_element,

			       num_domains, num_dofs, num_elements);
  

  printf("-------------- Assembling domain matrices: ----------------------\n");

  /*
  for (i=0; i < num_elements; i++)
    for (j=i_element_chord[i]; j < i_element_chord[i+1]; j++)
	{
	  chord = j_element_chord[j];
	  i_dof = j_chord_dof[i_chord_dof[chord]];
	  j_dof = j_chord_dof[i_chord_dof[chord]+1];

	  if (i_dof == j_dof) printf("diagonal entry %d: %e\n", i_dof,
				     a_element_chord[j]);
	}
	*/

  ierr=hypre_AMGeDomainElementSparseAssemble(i_domain_element, 
					     j_domain_element,
					     num_domains,

					     i_element_chord,
					     j_element_chord,
					     a_element_chord,

					     i_chord_dof, 
					     j_chord_dof,

					     &i_domain_chord,
					     &j_domain_chord,
					     &a_domain_chord,

					     num_elements, 
					     num_chords,
					     num_dofs);

  printf("END assembling domain matrices: --------------------------------\n");


  ierr = matrix_matrix_product(&i_domain_dof, &j_domain_dof,

			       i_domain_element, j_domain_element,
			       i_element_dof, j_element_dof,

			       num_domains, num_elements, num_dofs);

  hypre_TFree(i_domain_element);
  hypre_TFree(j_domain_element);


  /*
  i_dof_index = hypre_CTAlloc(int, num_dofs);
  

  for (i=0; i < num_domains; i++)
    i_dof_index[i] = -1;

  for (i=0; i < num_domains; i++)
    {

      printf("\n domain: %d ====================================\n", i);
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  printf("%d ", j_domain_dof[j]);
	  i_dof_index[j_domain_dof[j]] = 0;
	}
      printf("\n subdomain: %d ====================================\n", i);



      for (j=i_subdomain_dof[i]; j < i_subdomain_dof[i+1]; j++)
	{
	  printf("%d ", j_subdomain_dof[j]);	  
	  if (i_dof_index[j_subdomain_dof[j]] < 0)
	    printf("\nsubdomain %d contains entry %d not in domain %d\n",
		   i, j_subdomain_dof[j], i);
	}
      printf("\n\n");
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	i_dof_index[j_domain_dof[j]] = -1;

    }

  hypre_TFree(i_dof_index);
  */


  printf("----------------------Computing neighborhood Schur complements: ------\n");
  ierr =  hypre_AMGeSchurComplement(i_domain_chord,
				    j_domain_chord,
				    a_domain_chord,

				    i_chord_dof, j_chord_dof,

				    i_domain_dof, j_domain_dof,
				    i_subdomain_dof, j_subdomain_dof,

				    &i_Schur_dof_dof,
				    &a_Schur_dof_dof,
			      
				    num_domains, num_chords, num_dofs);

  printf("END computing neighborhood Schur complements: ---------------------\n");

  hypre_TFree(i_domain_dof);
  hypre_TFree(j_domain_dof);

  hypre_TFree(i_domain_chord);
  hypre_TFree(j_domain_chord);
  hypre_TFree(a_domain_chord);
   


  for (i=0; i < num_domains; i++)
    {
      local_dof_counter = i_subdomain_dof[i+1]-i_subdomain_dof[i];
      if (max_local_dof_counter < local_dof_counter)
	max_local_dof_counter = local_dof_counter;
    }
  

  AE = hypre_CTAlloc(double, max_local_dof_counter*max_local_dof_counter);
  QE = hypre_CTAlloc(double, max_local_dof_counter*max_local_dof_counter);

  W  = hypre_CTAlloc(double, max_local_dof_counter);
  Aux1=hypre_CTAlloc(double, max_local_dof_counter);
  Aux2=hypre_CTAlloc(double, max_local_dof_counter);

  i_local_to_global = hypre_CTAlloc(int, max_local_dof_counter);
  i_global_to_local = hypre_CTAlloc(int, num_dofs);


  i_dof_neighbor_coarsedof = hypre_CTAlloc(int, num_dofs+1);
  for (i=0; i < num_dofs+1; i++)
    i_dof_neighbor_coarsedof[i] = 0;
  
  i_dof_index = hypre_CTAlloc(int, num_dofs);
  for (i=0; i < num_dofs; i++)
    i_dof_index[i] = -1;
  

  dof_neighbor_coarsedof_counter = 0;
  coarsedof_counter = 0;

  for (i=0; i < num_dofs; i++)
    i_global_to_local[i]=-1;
  

  /* select coarsedofs in each subdomain based on first few eigenvalues
     of the subdomain Schur complements --------------------------------------- */
  for (i=0; i < num_domains; i++)
    {
      subdomain_coarsedof_counter = 0;
      
      local_dof_counter = 0;
      for (j=i_subdomain_dof[i]; j < i_subdomain_dof[i+1]; j++)
	{
	  i_local_to_global[local_dof_counter] = j_subdomain_dof[j];
	  i_global_to_local[j_subdomain_dof[j]] = local_dof_counter;
	  local_dof_counter++;
	}

      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	  AE[j_loc + i_loc * local_dof_counter] = 
	    a_Schur_dof_dof[i_Schur_dof_dof[i]
			   + j_loc + i_loc * local_dof_counter];
      
      if (local_dof_counter > 0)
	rs_(&local_dof_counter, &local_dof_counter, AE, W, &matz, QE, 
	    Aux1, Aux2, &ierr);

      printf("end eigenpair computations: ------------------------------\n");
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	 printf("%e ", W[i_loc]);
       
      printf("\n\n");
      
       


      /* label local dofs for coarse if eig[i_loc] <= 0.25 * eig_max; --------- */
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	if (W[i_loc] > 0.25 * W[local_dof_counter-1] 
	    && W[local_dof_counter-1] >0.e0) 
	  break;
	else
	  {
	    subdomain_coarsedof_counter++;
	    coarsedof_counter++;
	    i_dof_index[i_local_to_global[i_loc]] = i_local_to_global[i_loc];
	  }

      /* all subodmain dofs are interpolated from the first 
	 subdomain_coarsedof_counter local coarsedofs; ------------------------- */
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	i_dof_neighbor_coarsedof[i_local_to_global[i_loc]]+=
	  subdomain_coarsedof_counter;

      dof_neighbor_coarsedof_counter += local_dof_counter *
	subdomain_coarsedof_counter;

      /* store eigenvectors: ----------------------------------------- */     
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	  a_Schur_dof_dof[i_Schur_dof_dof[i]
			 + j_loc + i_loc * local_dof_counter]=
	    QE[j_loc + i_loc * local_dof_counter];
 
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	i_global_to_local[i_local_to_global[i_loc]] = -1;
      
    }


  

  ierr = transpose_matrix_create(&i_dof_subdomain,
				 &j_dof_subdomain,

				 i_subdomain_dof,
				 j_subdomain_dof,

				 num_domains, num_dofs);

  for (i=0; i < num_dofs; i++)
    if (i_dof_subdomain[i+1]==i_dof_subdomain[i])
      {
	i_dof_neighbor_coarsedof[i]++;
	dof_neighbor_coarsedof_counter++;
	coarsedof_counter++;
	i_dof_index[i] = i;
      }
    else
      if (i_dof_subdomain[i+1]>1+i_dof_subdomain[i])
	printf("ERROR: dof %d belongs to %d subdomains;\n", i,
	       i_dof_subdomain[i+1]-i_dof_subdomain[i]);

  
  for (i=0; i < num_AEfaces; i++)
    for (j=i_AEface_dof[i]; j < i_AEface_dof[i+1]; j++)
      /* coarsedof if it belongs to > 1 AEface: ------------------------------- */
      if (i_dof_AEface[j_AEface_dof[j]+1] > i_dof_AEface[j_AEface_dof[j]]+1)
	if (i_dof_index[j_AEface_dof[j]] < 0)
	  printf("ERROR: dof %d belonging > 1  AEface but not coarse !\n", 
		 j_AEface_dof[j]);
  

  /* prepare for CSR format: ----------------------------------------- */     
  for (i=0; i < num_dofs; i++)
    i_dof_neighbor_coarsedof[i+1]+=i_dof_neighbor_coarsedof[i];
  
  for (i=num_dofs; i > 0; i--)
    i_dof_neighbor_coarsedof[i]=i_dof_neighbor_coarsedof[i-1];

  i_dof_neighbor_coarsedof[0] = 0;
  

  j_dof_neighbor_coarsedof = hypre_CTAlloc(int, dof_neighbor_coarsedof_counter);
  P_coeff = hypre_CTAlloc(double, dof_neighbor_coarsedof_counter);



  /* identify coarse dofs: --------------------------------------------- */
  num_coarsedofs = coarsedof_counter;
  *num_coarsedofs_pointer = num_coarsedofs;

  i_dof_coarsedof = hypre_CTAlloc(int, num_dofs+1);
  j_dof_coarsedof = hypre_CTAlloc(int, num_coarsedofs);

  printf("====================== num_coarsedofs: %d ======================\n",
	 num_coarsedofs);
  

  dof_neighbor_coarsedof_counter = 0;
  dof_coarsedof_counter = 0;
  for (i=0; i < num_dofs; i++)
    {
      i_dof_coarsedof[i] = dof_coarsedof_counter;
  

      if (i_dof_subdomain[i+1] == i_dof_subdomain[i])
	/* dof i does not belong to a subdomain, hence coarse: ------------ */
	{
	  j_dof_neighbor_coarsedof[dof_neighbor_coarsedof_counter] = i;
	  P_coeff[dof_neighbor_coarsedof_counter] = 1.e0;
	  dof_neighbor_coarsedof_counter++;

	  printf("coarsedof: %d, i_dof_index[%d]: %d\n", 
		 dof_coarsedof_counter, i, i_dof_index[i]);
	  j_dof_coarsedof[dof_coarsedof_counter] = dof_coarsedof_counter;
	  dof_coarsedof_counter++;
	}
      else
	/* dof i belongs to a unique subdomain: ---------------------- */
	{	
	  k=j_dof_subdomain[i_dof_subdomain[i]];
	  
	  local_dof_counter =0;
	  for (l=i_subdomain_dof[k]; l < i_subdomain_dof[k+1]; l++)
	    {
	      i_local_to_global[local_dof_counter] = j_subdomain_dof[l];
		
	      i_global_to_local[j_subdomain_dof[l]] = local_dof_counter;
	      local_dof_counter++;
		
	    }

	  subdomain_coarsedof_counter = i_dof_neighbor_coarsedof[i+1] 
	                              - i_dof_neighbor_coarsedof[i];
		

	  j_loc = i_global_to_local[i];

	    
	  for (i_loc = 0; i_loc < subdomain_coarsedof_counter; i_loc++)
	    {
	      j_dof_neighbor_coarsedof[dof_neighbor_coarsedof_counter] = 
		i_local_to_global[i_loc];
		
	      P_coeff[dof_neighbor_coarsedof_counter] = 
		a_Schur_dof_dof[i_Schur_dof_dof[k]
			       + j_loc + i_loc * local_dof_counter];

	      dof_neighbor_coarsedof_counter++;
	    }

	  /* ------------------------------------------------------------
	     if dof i locally, i.e., j_loc < subdomain_coarsedof_counter,
	     hence coarse: ---------------------------------------------- */
	  if (j_loc < subdomain_coarsedof_counter)
	    {
	      printf("coarsedof: %d, i_dof_index[%d]: %d\n", dof_coarsedof_counter,
		     i, i_dof_index[i]);
	      j_dof_coarsedof[dof_coarsedof_counter] = dof_coarsedof_counter;
	      dof_coarsedof_counter++;
	    }
	  

	  for (l=i_subdomain_dof[k]; l < i_subdomain_dof[k+1]; l++)
	    i_global_to_local[j_subdomain_dof[l]] = -1;
	}
      
    }


  i_dof_coarsedof[num_dofs] = dof_coarsedof_counter;

  *i_dof_coarsedof_pointer = i_dof_coarsedof;
  *j_dof_coarsedof_pointer = j_dof_coarsedof;


  hypre_TFree(i_subdomain_dof);
  hypre_TFree(j_subdomain_dof);

  hypre_TFree(i_dof_subdomain);
  hypre_TFree(j_dof_subdomain);

  hypre_TFree(i_Schur_dof_dof);
  hypre_TFree(a_Schur_dof_dof);

  printf("-------------- Assembling AE matrices: ----------------------\n");
  printf(" num_elements: %d, num_chords: %d, num_dofs: %d. num_AEs: %d\n",
	 num_elements,  num_chords, num_dofs, num_AEs);
  
  ierr=hypre_AMGeDomainElementSparseAssemble(i_AE_element, 
					     j_AE_element,
					     num_AEs,

					     i_element_chord,
					     j_element_chord,
					     a_element_chord,

					     i_chord_dof, 
					     j_chord_dof,

					     &i_AE_chord,
					     &j_AE_chord,
					     &a_AE_chord,

					     num_elements, 
					     num_chords,
					     num_dofs);

  printf("END assembling AE matrices: --------------------------------\n");


  /* change coarsedof numbering from fine to actual coarse: ------------- */
  for (i=0; i < num_dofs; i++)
    for (j=i_dof_neighbor_coarsedof[i]; j<i_dof_neighbor_coarsedof[i+1]; j++)
      j_dof_neighbor_coarsedof[j] = 
	j_dof_coarsedof[i_dof_coarsedof[j_dof_neighbor_coarsedof[j]]];



  AE_coarsedof_counter = 0;
  for (i=0; i < num_AEs; i++)
    for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
      if (i_dof_coarsedof[j_AE_dof[j]+1] > i_dof_coarsedof[j_AE_dof[j]])
	AE_coarsedof_counter++;

  i_AE_coarsedof = hypre_CTAlloc(int, num_AEs+1);
  j_AE_coarsedof = hypre_CTAlloc(int, AE_coarsedof_counter);

  AE_coarsedof_counter = 0;
  for (i=0; i < num_AEs; i++)
    {
      i_AE_coarsedof[i] = AE_coarsedof_counter;
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	if (i_dof_coarsedof[j_AE_dof[j]+1] > i_dof_coarsedof[j_AE_dof[j]])
	  {
	    j_AE_coarsedof[AE_coarsedof_counter] = 
	      j_dof_coarsedof[i_dof_coarsedof[j_AE_dof[j]]];

	    AE_coarsedof_counter++;
	  }
    }

  i_AE_coarsedof[num_AEs] = AE_coarsedof_counter;

  *i_coarseelement_coarsedof_pointer = i_AE_coarsedof;
  *j_coarseelement_coarsedof_pointer = j_AE_coarsedof;
  
  printf("END coarseelement_coarsedof computation: ----------------------------\n");
  

  /* compute coarse element matrices: ----------------------------------------- */


  i_AE_coarsedof_coarsedof = hypre_CTAlloc(int, num_AEs+1);
  
  AE_coarsedof_coarsedof_counter = 0;
  for (i=0; i < num_AEs; i++)
    {
      i_AE_coarsedof_coarsedof[i] = AE_coarsedof_coarsedof_counter;      
      AE_coarsedof_coarsedof_counter+= 
	(i_AE_coarsedof[i+1]-i_AE_coarsedof[i]) *
	(i_AE_coarsedof[i+1]-i_AE_coarsedof[i]);
    }
  
  i_AE_coarsedof_coarsedof[num_AEs] = AE_coarsedof_coarsedof_counter;  

  AE_coarsedof_coarsedof = hypre_CTAlloc(double,
					 AE_coarsedof_coarsedof_counter);
  

  for (i=0; i < num_coarsedofs; i++)
    i_global_to_local[i] = -1;

  for (i=0; i < num_AEs; i++)
    {
      local_coarsedof_counter = 0;
      for (j=i_AE_coarsedof[i]; j < i_AE_coarsedof[i+1]; j++)
	{
	  i_local_to_global[local_coarsedof_counter] = j_AE_coarsedof[j];
	  i_global_to_local[j_AE_coarsedof[j]] = local_coarsedof_counter;
	  local_coarsedof_counter++;
	  
	}

      for (i_loc=0; i_loc < local_coarsedof_counter; i_loc++)
	for (j_loc=0; j_loc < local_coarsedof_counter; j_loc++)
	  AE_coarsedof_coarsedof[i_AE_coarsedof_coarsedof[i]
				+j_loc + local_coarsedof_counter * i_loc] = 0.e0;
      


      /* here we perform RAP locally: -------------------------------- */
      for (j=i_AE_chord[i]; j < i_AE_chord[i+1]; j++)
	{
	  chord = j_AE_chord[j];
	  i_dof = j_chord_dof[i_chord_dof[chord]];
	  j_dof = j_chord_dof[i_chord_dof[chord]+1];

	  /* if (i_dof == j_dof) printf("diagonal entry %d: %e\n", i_dof,
				     a_AE_chord[j]); */
	  
	  for (k=i_dof_neighbor_coarsedof[i_dof];
	       k<i_dof_neighbor_coarsedof[i_dof+1]; k++)
	    {
	      i_coarsedof = j_dof_neighbor_coarsedof[k];
	      i_loc = i_global_to_local[i_coarsedof];
	      if (i_loc < 0) printf("wrong index !!!!!!!!!!!!!!!!!!!!!!!! \n");
	      
	      for (l=i_dof_neighbor_coarsedof[j_dof];
		   l<i_dof_neighbor_coarsedof[j_dof+1]; l++)
		{
		  j_coarsedof = j_dof_neighbor_coarsedof[l]; 
		  j_loc = i_global_to_local[j_coarsedof];
		  if (j_loc < 0) printf("wrong index !!!!!!!!!!!!!!!!!!!!!!!! \n");
		  AE_coarsedof_coarsedof[i_AE_coarsedof_coarsedof[i]
		    +j_loc + local_coarsedof_counter * i_loc]+=
		    P_coeff[k] * a_AE_chord[j] * P_coeff[l];
		}
	    }
	}


      printf("\n\n NEW COARSE ELEMENT MATRIX: ======================\n");
      
      for (i_loc=0; i_loc < local_coarsedof_counter; i_loc++)
	{
	  printf("\n");
	  for (j_loc=0; j_loc < local_coarsedof_counter; j_loc++) 
	    if (j_loc==i_loc) printf("%e ",
		   AE_coarsedof_coarsedof[i_AE_coarsedof_coarsedof[i]
					 +j_loc + local_coarsedof_counter * i_loc]);
      	  printf("\n");

	}


      for (i_loc=0; i_loc < local_coarsedof_counter; i_loc++)
	i_global_to_local[i_local_to_global[i_loc]] = -1;

    }
  
  hypre_TFree(i_AE_coarsedof_coarsedof);

  hypre_TFree(i_AE_chord);
  hypre_TFree(j_AE_chord);
  hypre_TFree(a_AE_chord);
  printf("END coarseelement_matrix computation: -------------------------------\n");

  /*      
  printf("num_AEs: %d, num_coarsedofs: %d\n", num_AEs, num_coarsedofs);
  for (i=0; i < num_AEs; i++)
    {
      printf("coarseelement: %d\n", i);
      
      for (j=i_AE_coarsedof[i]; j < i_AE_coarsedof[i+1]; j++)
	printf("%d ", j_AE_coarsedof[j]);

      printf("\n");
      
    }
    */
  ierr = hypre_AMGeElementMatrixDof(i_AE_coarsedof, j_AE_coarsedof,
				    AE_coarsedof_coarsedof,

				    &i_coarseelement_coarsechord,
				    &j_coarseelement_coarsechord,
				    &a_coarseelement_coarsechord,

				    &i_coarsechord_coarsedof,
				    &j_coarsechord_coarsedof,

				    &num_coarsechords,

				    num_AEs, num_coarsedofs);


  printf("END storing coarseelement_matrices in element_chord format: ---------\n");

  *i_coarseelement_coarsechord_pointer = i_coarseelement_coarsechord;
  *j_coarseelement_coarsechord_pointer = j_coarseelement_coarsechord;
  *a_coarseelement_coarsechord_pointer = a_coarseelement_coarsechord;

  *i_coarsechord_coarsedof_pointer = i_coarsechord_coarsedof;
  *j_coarsechord_coarsedof_pointer = j_coarsechord_coarsedof;
  *num_coarsechords_pointer        = num_coarsechords;
  
  

  P = hypre_CSRMatrixCreate(num_dofs, num_coarsedofs,
			    i_dof_neighbor_coarsedof[num_dofs]);



  hypre_CSRMatrixData(P) = P_coeff;
  hypre_CSRMatrixI(P) = i_dof_neighbor_coarsedof; 
  hypre_CSRMatrixJ(P) = j_dof_neighbor_coarsedof; 

  *P_pointer = P;


  hypre_TFree(i_local_to_global);
  hypre_TFree(i_global_to_local);
  
  hypre_TFree(AE);
  hypre_TFree(QE);
  hypre_TFree(W);
  hypre_TFree(Aux1);
  hypre_TFree(Aux2);
  

  printf("===============================================================\n");
  printf("END Build Interpolation Matrix: ===============================\n");
  printf("===============================================================\n");

  printf("ierr_build_interpolation: %d =================================\n", ierr);
  
  return ierr;
}
				 
/*---------------------------------------------------------------------
 matinv:  X <--  A**(-1) ;  A IS POSITIVE DEFINITE (non--symmetric);
 ---------------------------------------------------------------------*/
      
int matinv(double *x, double *a, int k)
{
  int i,j,l, ierr =0;
  double *b;

  if (k > 0)
    b = hypre_CTAlloc(double, k*k);

  for (l=0; l < k; l++)
    for (j=0; j < k; j++)
      b[j+k*l] = a[j+k*l];

  for (i=0; i < k; i++)
    {
      if (a[i+i*k] <= 1.e-20)
	{
	  ierr = -1; 
	  printf("size: %d, diagonal entry: %d, %e\n", k, i, a[i+i*k]);
	    /*	  
	    printf("matinv: ==========================================\n");


            printf("indefinite singular matrix in *** matinv ***:\n");
            printf("i:%d;  diagonal entry: %e\n", i, a[i+k*i]);


	    for (l=0; l < k; l++)
	      {
		printf("\n");
		for (j=0; j < k; j++)
		  printf("%f ", b[j+k*l]);

		printf("\n");
	      }

	    return ierr;
	    */

	  a[i+i*k] = 0.e0;
	}
         else
            a[i+k*i] = 1.0 / a[i+i*k];

      for (j=1; j < k-i; j++)
	{
	  for (l=1; l < k-i; l++)
	    {
	      a[i+l+k*(i+j)] -= a[i+l+k*i] * a[i+k*i] * a[i+k*(i+j)];
	    }
	}
      
      for (j=1; j < k-i; j++)
	{
	  a[i+j+k*i] = a[i+j+k*i] * a[i+k*i];
	  a[i+k*(i+j)] = a[i+k*(i+j)] * a[i+k*i];
	}
    }

  /* FULL INVERSION: --------------------------------------------*/
  
  if (k > 0)
    x[k*k-1] = a[k*k-1];
  for (i=k-1; i > -1; i--)
    {
      for (j=1; j < k-i; j++)
	{
	  x[i+j+k*i] =0;
	  x[i+k*(i+j)] =0;

	  for (l=1; l< k-i; l++)
	    {
	      x[i+j+k*i] -= x[i+j+k*(i+l)] * a[i+l+k*i];
	      x[i+k*(i+j)] -= a[i+k*(i+l)] * x[i+l+k*(i+j)];
	    }
	}

      x[i+k*i] = a[i+k*i];
      for (j=1; j<k-i; j++)
	{
	  x[i+k*i] -= x[i+k*(i+j)] * a[i+j+k*i];
	}
    }
  /*------------------------------------------------------------------
  for (i=0; i < k; i++) 
    {
      for (j=0; j < k; j++)
	  if (x[j+k*i] != x[i+k*j])
	    printf("\n non_symmetry: %f %f", x[j+k*i], x[i+k*j] );
    }
  printf("\n");

   -----------------------------------------------------------------*/

  if (k > 0) hypre_TFree(b);

  return ierr;
}
