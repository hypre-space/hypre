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

HYPRE_Int hypre_AMGeAGBuildInterpolation(hypre_CSRMatrix     **P_pointer,

				   HYPRE_Int **i_coarseelement_coarsedof_pointer,
				   HYPRE_Int **j_coarseelement_coarsedof_pointer,


				   HYPRE_Int **i_coarseelement_coarsechord_pointer,
				   HYPRE_Int **j_coarseelement_coarsechord_pointer,
				   double **a_coarseelement_coarsechord_pointer,

				   HYPRE_Int **i_coarsechord_coarsedof_pointer, 
				   HYPRE_Int **j_coarsechord_coarsedof_pointer,

				   HYPRE_Int *num_coarsechords_pointer,

				   HYPRE_Int **i_dof_coarsedof_pointer,
				   HYPRE_Int **j_dof_coarsedof_pointer,

				   HYPRE_Int *num_coarsedofs_pointer,

				   HYPRE_Int *i_element_dof, HYPRE_Int *j_element_dof,
				   HYPRE_Int *i_dof_element, HYPRE_Int *j_dof_element,

				   HYPRE_Int *i_element_chord,
				   HYPRE_Int *j_element_chord,
				   double *a_element_chord,


				   HYPRE_Int *i_AEface_dof, HYPRE_Int *j_AEface_dof,
				   HYPRE_Int *i_dof_AEface, HYPRE_Int *j_dof_AEface,

				   HYPRE_Int *i_AE_element, HYPRE_Int *j_AE_element,


				   HYPRE_Int *i_chord_dof, HYPRE_Int *j_chord_dof,

				   /* double tolerance,  */

				   HYPRE_Int num_chords,

				   HYPRE_Int num_AEs, 
				   HYPRE_Int num_AEfaces, 

				   HYPRE_Int num_elements, 
				   HYPRE_Int num_dofs)

{
  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k,l;

  HYPRE_Int matz = 1;
  HYPRE_Int i_loc, j_loc, k_loc, l_loc;

  HYPRE_Int chord, i_dof, j_dof, i_coarsedof, j_coarsedof;
  
  

  HYPRE_Int *i_AEface_element, *j_AEface_element;
  HYPRE_Int *i_AEface_dof_dof, *j_AEface_dof_dof;


  HYPRE_Int *i_AE_dof, *j_AE_dof;
  
  HYPRE_Int *i_dof_coarsedof, *j_dof_coarsedof;
  HYPRE_Int num_coarsedofs;

  HYPRE_Int *i_AE_coarsedof, *j_AE_coarsedof;
  HYPRE_Int *i_AE_coarsedof_coarsedof;
  double *AE_coarsedof_coarsedof;
  HYPRE_Int AE_coarsedof_coarsedof_counter;
  
  
  HYPRE_Int *i_domain_element, *j_domain_element;
  HYPRE_Int *i_domain_chord, *j_domain_chord;
  double *a_domain_chord;

  HYPRE_Int *i_AE_chord, *j_AE_chord;
  double *a_AE_chord;

  HYPRE_Int *i_domain_dof, *j_domain_dof;
  HYPRE_Int *i_subdomain_dof, *j_subdomain_dof;
  HYPRE_Int *i_dof_subdomain, *j_dof_subdomain;
  HYPRE_Int *i_Schur_dof_dof;
  double *a_Schur_dof_dof, *P_coeff;

  hypre_CSRMatrix  *P;

  HYPRE_Int *i_dof_neighbor_coarsedof, *j_dof_neighbor_coarsedof;
  HYPRE_Int *i_dof_neighbor_coarsedof_0;

  HYPRE_Int *i_coarseelement_coarsechord, *j_coarseelement_coarsechord;
  double *a_coarseelement_coarsechord;

  HYPRE_Int *i_coarsechord_coarsedof, *j_coarsechord_coarsedof;
  HYPRE_Int num_coarsechords;

  HYPRE_Int AE_coarsedof_counter;
  
  HYPRE_Int num_domains;

  HYPRE_Int domain_counter, domain_element_counter, subdomain_dof_counter;

  HYPRE_Int *i_local_to_global;
  HYPRE_Int *i_global_to_local;
    
  HYPRE_Int max_local_dof_counter = 0, local_dof_counter, 
    max_local_coarsedof_counter =0, local_coarsedof_counter;

  HYPRE_Int local_boundary_coarsedof_counter, int_dof_counter;

  double *AE, *QE, *W, *Aux1, *Aux2;
  

  HYPRE_Int subdomain_coarsedof_counter, 
    dof_neighbor_coarsedof_counter, coarsedof_counter, dof_coarsedof_counter;

  double  row_sum;


  HYPRE_Int *i_dof_index;
  
  double *P_boundary;

  HYPRE_Int *i_boundary, *i_boundary_to_local, 
    *i_int, *i_interior_to_local;
  double tolerance = 0.001;

  /* ------------------------------------------------------------------ */
  /* building interior of domains associated with AEs and AEfaces: ---- */
  /* ------------------------------------------------------------------ */
  ierr = matrix_matrix_product(&i_AE_dof, &j_AE_dof,

			       i_AE_element, j_AE_element,
			       i_element_dof, j_element_dof,

			       num_AEs, num_elements, num_dofs);


  num_domains = num_AEs + num_AEfaces;
  i_subdomain_dof = hypre_CTAlloc(HYPRE_Int, num_domains+1);
  

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

  j_subdomain_dof = hypre_CTAlloc(HYPRE_Int, subdomain_dof_counter);


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
  

  hypre_printf("-------------- Assembling domain matrices: ----------------------\n");

  /*
  for (i=0; i < num_elements; i++)
    for (j=i_element_chord[i]; j < i_element_chord[i+1]; j++)
	{
	  chord = j_element_chord[j];
	  i_dof = j_chord_dof[i_chord_dof[chord]];
	  j_dof = j_chord_dof[i_chord_dof[chord]+1];

	  if (i_dof == j_dof) hypre_printf("diagonal entry %d: %e\n", i_dof,
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

  hypre_printf("END assembling domain matrices: --------------------------------\n");


  ierr = matrix_matrix_product(&i_domain_dof, &j_domain_dof,

			       i_domain_element, j_domain_element,
			       i_element_dof, j_element_dof,

			       num_domains, num_elements, num_dofs);

  hypre_TFree(i_domain_element);
  hypre_TFree(j_domain_element);


  /*
  i_dof_index = hypre_CTAlloc(HYPRE_Int, num_dofs);
  

  for (i=0; i < num_domains; i++)
    i_dof_index[i] = -1;

  for (i=0; i < num_domains; i++)
    {

      hypre_printf("\n domain: %d ====================================\n", i);
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  hypre_printf("%d ", j_domain_dof[j]);
	  i_dof_index[j_domain_dof[j]] = 0;
	}
      hypre_printf("\n subdomain: %d ====================================\n", i);



      for (j=i_subdomain_dof[i]; j < i_subdomain_dof[i+1]; j++)
	{
	  hypre_printf("%d ", j_subdomain_dof[j]);	  
	  if (i_dof_index[j_subdomain_dof[j]] < 0)
	    hypre_printf("\nsubdomain %d contains entry %d not in domain %d\n",
		   i, j_subdomain_dof[j], i);
	}
      hypre_printf("\n\n");
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	i_dof_index[j_domain_dof[j]] = -1;

    }

  hypre_TFree(i_dof_index);
  */


  hypre_printf("----------------------Computing neighborhood Schur complements: ------\n");
  ierr =  hypre_AMGeSchurComplement(i_domain_chord,
				    j_domain_chord,
				    a_domain_chord,

				    i_chord_dof, j_chord_dof,

				    i_domain_dof, j_domain_dof,
				    i_subdomain_dof, j_subdomain_dof,

				    &i_Schur_dof_dof,
				    &a_Schur_dof_dof,
			      
				    num_domains, num_chords, num_dofs);

  hypre_printf("END computing neighborhood Schur complements: ---------------------\n");

  hypre_TFree(i_domain_dof);
  hypre_TFree(j_domain_dof);

  hypre_TFree(i_domain_chord);
  hypre_TFree(j_domain_chord);
  hypre_TFree(a_domain_chord);
   
  /* ==================================================================
     ============= COMPUTING SCHUR COMPLEMENTS EIGENPAIRS: ============
     ================================================================== */

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

  i_local_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);
  i_global_to_local = hypre_CTAlloc(HYPRE_Int, num_dofs);


  i_dof_neighbor_coarsedof = hypre_CTAlloc(HYPRE_Int, num_dofs+1);
  for (i=0; i < num_dofs+1; i++)
    i_dof_neighbor_coarsedof[i] = 0;
  
  i_dof_index = hypre_CTAlloc(HYPRE_Int, num_dofs);
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

      /*
      hypre_printf("end eigenpair computations: ------------------------------\n");
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	 hypre_printf("%e ", W[i_loc]);
       
      hypre_printf("\n\n");
      
      */       


      /* label local dofs for coarse if eig[i_loc] <= tolerance * eig_max; ------ */
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	if (W[i_loc] > tolerance * W[local_dof_counter-1]
	    && subdomain_coarsedof_counter > 0) 
	  break;
	else 
	  {
	    subdomain_coarsedof_counter++;
	    coarsedof_counter++;
	    i_dof_index[i_local_to_global[i_loc]] = i_local_to_global[i_loc];
	    /*
	    if (subdomain_coarsedof_counter > 5) 
	      break;
	      */
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
	hypre_printf("ERROR: dof %d belongs to %d subdomains;\n", i,
	       i_dof_subdomain[i+1]-i_dof_subdomain[i]);

  
  for (i=0; i < num_AEfaces; i++)
    for (j=i_AEface_dof[i]; j < i_AEface_dof[i+1]; j++)
      /* coarsedof if it belongs to > 1 AEface: ------------------------------- */
      if (i_dof_AEface[j_AEface_dof[j]+1] > i_dof_AEface[j_AEface_dof[j]]+1)
	if (i_dof_index[j_AEface_dof[j]] < 0)
	  hypre_printf("ERROR: dof %d belonging > 1  AEface but not coarse !\n", 
		 j_AEface_dof[j]);
  

  
  /* for each AE wirh non--empty fine interior 
     find its boundary coarse dofs: --------------------- */

  for (i=0; i < num_AEs; i++)
    {
      k = -1;
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	if (i_dof_subdomain[j_AE_dof[j]+1] == i_dof_subdomain[j_AE_dof[j]]+1
	    && j_dof_subdomain[i_dof_subdomain[j_AE_dof[j]]] == i
	    && i_dof_index[j_AE_dof[j]] < 0)
	  {
	    k = 0;
	    break;
	  }

      if (k == 0)
	{
	  local_boundary_coarsedof_counter = 0;
	  for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	    {
	      if (i_dof_subdomain[j_AE_dof[j]+1] == i_dof_subdomain[j_AE_dof[j]]
		  || j_dof_subdomain[i_dof_subdomain[j_AE_dof[j]]] != i)
		if (i_dof_index[j_AE_dof[j]] >= 0)
		  local_boundary_coarsedof_counter++;
	    }
      
	  for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	    if (i_dof_subdomain[j_AE_dof[j]+1] == i_dof_subdomain[j_AE_dof[j]]+1
		&& j_dof_subdomain[i_dof_subdomain[j_AE_dof[j]]] == i)
	      {
		i_dof_neighbor_coarsedof[j_AE_dof[j]]+=
		  local_boundary_coarsedof_counter;
		dof_neighbor_coarsedof_counter +=local_boundary_coarsedof_counter;
	      }
	}
    }


  /* prepare for CSR format: ----------------------------------------- */     
  for (i=0; i < num_dofs; i++)
    i_dof_neighbor_coarsedof[i+1]+=i_dof_neighbor_coarsedof[i];
  
  for (i=num_dofs; i > 0; i--)
    i_dof_neighbor_coarsedof[i]=i_dof_neighbor_coarsedof[i-1];

  i_dof_neighbor_coarsedof[0] = 0;


  /* store i_dof_neighbor_coarsedof[i] in i_dof_neighbor_coarsedof_0[i]; */

  i_dof_neighbor_coarsedof_0 = hypre_CTAlloc(HYPRE_Int, num_dofs+1);    
  for (i=0; i < num_dofs+1; i++)
    i_dof_neighbor_coarsedof_0[i] = i_dof_neighbor_coarsedof[i];

  if (i_dof_neighbor_coarsedof[num_dofs] != dof_neighbor_coarsedof_counter)
    hypre_printf("ERROR: dof_neighbor_coarsedof_counter: %d, %d\n",
	   dof_neighbor_coarsedof_counter, 
	   i_dof_neighbor_coarsedof[num_dofs]);


  j_dof_neighbor_coarsedof = hypre_CTAlloc(HYPRE_Int, dof_neighbor_coarsedof_counter);
  P_coeff = hypre_CTAlloc(double, dof_neighbor_coarsedof_counter);


  /* identify coarse dofs: --------------------------------------------- */
  num_coarsedofs = coarsedof_counter;
  *num_coarsedofs_pointer = num_coarsedofs;

  i_dof_coarsedof = hypre_CTAlloc(HYPRE_Int, num_dofs+1);
  j_dof_coarsedof = hypre_CTAlloc(HYPRE_Int, num_coarsedofs);

  hypre_printf("====================== num_coarsedofs: %d ======================\n",
	 num_coarsedofs);

  hypre_printf("\n\n =========== COMPUTE P_coeff: Ist PHASE ========================\n");

  dof_coarsedof_counter = 0;
  for (i=0; i < num_dofs; i++)
    {
      i_dof_coarsedof[i] = dof_coarsedof_counter;
  

      if (i_dof_subdomain[i+1] == i_dof_subdomain[i])
	/* dof i does not belong to a subdomain, hence coarse: ------------ */
	{
	  j_dof_neighbor_coarsedof[i_dof_neighbor_coarsedof[i]] = i;

	  P_coeff[i_dof_neighbor_coarsedof[i]] = 1.e0;
	  i_dof_neighbor_coarsedof[i]++;

	  /*
	  hypre_printf("coarsedof: %d, i_dof_index[%d]: %d\n", 
		 dof_coarsedof_counter, i, i_dof_index[i]);
		 */
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
	  
	  subdomain_coarsedof_counter = 0;
	  for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
	    if (i_dof_index[i_local_to_global[i_loc]] > -1)
	      subdomain_coarsedof_counter++;


	  j_loc = i_global_to_local[i];

	    
	  for (i_loc = 0; i_loc < subdomain_coarsedof_counter; i_loc++)
	    {
	      j_dof_neighbor_coarsedof[i_dof_neighbor_coarsedof[i]] = 
		i_local_to_global[i_loc];
		
	      P_coeff[i_dof_neighbor_coarsedof[i]] = 
		a_Schur_dof_dof[i_Schur_dof_dof[k]
			       + j_loc + i_loc * local_dof_counter];

	      i_dof_neighbor_coarsedof[i]++;
	    }

	  /* ------------------------------------------------------------
	     if dof i locally, i.e., j_loc < subdomain_coarsedof_counter,
	     hence coarse: ---------------------------------------------- */
	  if (j_loc < subdomain_coarsedof_counter)
	    {
	      /*
	      hypre_printf("coarsedof: %d, i_dof_index[%d]: %d\n", dof_coarsedof_counter,
		     i, i_dof_index[i]); */

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
  hypre_printf("\n\n =========== END Ist PHASE ===================================\n");

  hypre_TFree(i_local_to_global);

  max_local_coarsedof_counter = 0;
  for (i=0; i < num_AEs; i++)
    {
      local_dof_counter = i_AE_dof[i+1]-i_AE_dof[i];
      if (max_local_dof_counter < local_dof_counter)
	max_local_dof_counter = local_dof_counter;
    }

  i_local_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);

  for (i=0; i < num_dofs; i++)
    i_global_to_local[i] = -1;

  hypre_TFree(AE);
  hypre_TFree(QE);
  hypre_TFree(W);
  hypre_TFree(Aux1); 
  hypre_TFree(Aux2);


  hypre_printf("-------------- Assembling AE matrices: ----------------------\n");
  hypre_printf(" num_elements: %d, num_chords: %d, num_dofs: %d. num_AEs: %d\n",
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

  hypre_printf("END assembling AE matrices: --------------------------------\n");


  hypre_printf("\n\n =========== COMPUTE P_coeff: IInd PHASE ========================\n");

  AE = hypre_CTAlloc(double, max_local_dof_counter
		     *max_local_dof_counter);

  Aux1 = hypre_CTAlloc(double, max_local_dof_counter
		     *max_local_dof_counter);

  Aux2 = hypre_CTAlloc(double, max_local_dof_counter
		     *max_local_dof_counter);

  P_boundary = hypre_CTAlloc(double, max_local_dof_counter
		     *max_local_dof_counter);

  i_boundary = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);
  i_boundary_to_local = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);
  i_int = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);
  i_interior_to_local = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);

  for (i=0; i < num_dofs; i++)
    i_global_to_local[i] = -1;

  for (i=0; i < num_AEs; i++)
    {
      local_dof_counter = 0;
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	{
	  i_local_to_global[local_dof_counter] = j_AE_dof[j];
	  i_global_to_local[j_AE_dof[j]] = local_dof_counter;
	  local_dof_counter++;
	}

      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	  AE[j_loc+i_loc * local_dof_counter] = 0.e0;

      for (j=i_AE_chord[i]; j < i_AE_chord[i+1]; j++)
	{
	  chord = j_AE_chord[j];
	  i_loc = i_global_to_local[j_chord_dof[i_chord_dof[chord]]];
	  j_loc = i_global_to_local[j_chord_dof[i_chord_dof[chord]+1]];
	  AE[j_loc+i_loc * local_dof_counter] = a_AE_chord[j];
	}

      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	i_boundary[i_loc] = -1;

      local_boundary_coarsedof_counter = 0;
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	{
	  if (i_dof_subdomain[j_AE_dof[j]+1] == i_dof_subdomain[j_AE_dof[j]]
	      || j_dof_subdomain[i_dof_subdomain[j_AE_dof[j]]] != i)
	    if (i_dof_index[j_AE_dof[j]] >= 0)
	      {
		i_boundary_to_local[local_boundary_coarsedof_counter] = 
		  i_global_to_local[j_AE_dof[j]];
		i_boundary[i_global_to_local[j_AE_dof[j]]] 
		  = local_boundary_coarsedof_counter;
		local_boundary_coarsedof_counter++;
	      }
	}

      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	i_int[i_loc] = -1;

      int_dof_counter = 0;
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	{
	  if (i_dof_subdomain[j_AE_dof[j]+1] > i_dof_subdomain[j_AE_dof[j]]
	      && j_dof_subdomain[i_dof_subdomain[j_AE_dof[j]]] == i)
	    {
	      i_interior_to_local[int_dof_counter] = 
		i_global_to_local[j_AE_dof[j]];
	      i_int[i_global_to_local[j_AE_dof[j]]] 
		= int_dof_counter;
	      int_dof_counter++;
	    }
	}
	

      for (i_loc=0; i_loc < int_dof_counter; i_loc++)
	for (j_loc=0; j_loc < int_dof_counter; j_loc++)
	  Aux1[j_loc+i_loc*int_dof_counter] = 
	    AE[i_interior_to_local[j_loc] + i_interior_to_local[i_loc] *
	      local_dof_counter];

      if (int_dof_counter > 0)
	{
	  ierr = mat_inv(Aux2, Aux1, int_dof_counter); 
	  if (ierr < 0)
	    hypre_printf("ierr_mat_inv: %d\n", ierr);
	}

      /* here we compute: -------------------------------------------------------
	 -[q_+, ..., q_m]*[q_+, ..., q_m]^T * Aux2 * A_{HYPRE_Int, boundary} * P_boundary
	 ----------------------------------------------------------------------- */


      /* build boundary interpolation: --------------------------------- */

      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	{
	  if (i_int[i_global_to_local[j_AE_dof[j]]] < 0)
	    {
	      i_loc = i_global_to_local[j_AE_dof[j]];

	      for (l=i_dof_neighbor_coarsedof_0[j_AE_dof[j]];
		   l<i_dof_neighbor_coarsedof[j_AE_dof[j]]; l++)
		{
		  j_loc = i_global_to_local[j_dof_neighbor_coarsedof[l]];
		  if (j_loc < 0 || i_boundary[j_loc] < 0)
		    hypre_printf("wrong j_dof_neighbor_coarsedof: %d, %d\n",
			   j_loc, i_boundary[j_loc]);

		  P_boundary[i_loc+j_loc*local_dof_counter] = P_coeff[l];
		  /* hypre_printf("boundary interpolation coeff: %e\n", P_coeff[l]); */
		}
	    }
	}
      for (i_loc=0; i_loc < local_boundary_coarsedof_counter; i_loc++)
	for (j_loc=0; j_loc < int_dof_counter; j_loc++)
	  Aux1[j_loc + i_boundary_to_local[i_loc] * int_dof_counter] = 0.e0;
      

      for (l_loc=0; l_loc < local_dof_counter; l_loc++)
	if (i_int[l_loc] < 0)
	  for (i_loc=0; i_loc < local_boundary_coarsedof_counter; i_loc++)
	    for (j_loc=0; j_loc < int_dof_counter; j_loc++)
	      for (k_loc=0; k_loc < int_dof_counter; k_loc++)
		Aux1[j_loc + i_boundary_to_local[i_loc] * int_dof_counter] -= 
		  Aux2[j_loc + k_loc * int_dof_counter] * 
		  AE[i_interior_to_local[k_loc] + l_loc * local_dof_counter] *
		  P_boundary[l_loc + i_boundary_to_local[i_loc] * local_dof_counter];
      

      if (i_subdomain_dof[i+1]-i_subdomain_dof[i] != int_dof_counter)
	hypre_printf("ERROR: AE[%d] has wrong interior # dofs: %d, %d\n",
	       i, i_subdomain_dof[i+1]-i_subdomain_dof[i], int_dof_counter);

      for (j=i_subdomain_dof[i]; j < i_subdomain_dof[i+1]; j++)
	{
	  j_loc=i_global_to_local[j_subdomain_dof[j]];
	  if (i_int[j_loc] < 0)
	    hypre_printf("wrong interior dof: %d, i_int: %d\n", j_loc, i_int[j_loc]);

	  for (l_loc=0; l_loc < local_dof_counter; l_loc++)
	    if (i_boundary[l_loc] >= 0)
	      {
		/*
		for (i_loc= 0; i_loc < int_dof_counter; i_loc++)
		  if (i_dof_index[i_local_to_global[i_interior_to_local[i_loc]]] 
			< 0)
		    {
		      P_coeff[i_dof_neighbor_coarsedof[j_subdomain_dof[j]]] = 0.e0;
		      break;
		    }

		for (i_loc= 0; i_loc < int_dof_counter; i_loc++)
		  {
		    if (i_dof_index[i_local_to_global[i_interior_to_local[i_loc]]] 
			< 0)
		      {
			for (k=i_subdomain_dof[i]; k < i_subdomain_dof[i+1]; k++)
			  {
			    k_loc=i_global_to_local[j_subdomain_dof[k]];
	
			    P_coeff[i_dof_neighbor_coarsedof[j_subdomain_dof[j]]] += 
			      a_Schur_dof_dof[i_Schur_dof_dof[i]
					     + i_int[j_loc] + i_loc 
					     * int_dof_counter] *
			      a_Schur_dof_dof[i_Schur_dof_dof[i]
					     + i_int[k_loc] + i_loc
					     * int_dof_counter] *
			      Aux1[i_int[k_loc] + l_loc * int_dof_counter];
			  }

		      }
		  }
		  */

		for (i_loc= 0; i_loc < int_dof_counter; i_loc++)
		  if (i_dof_index[i_local_to_global[i_interior_to_local[i_loc]]] 
			< 0)
		    {
		      P_coeff[i_dof_neighbor_coarsedof[j_subdomain_dof[j]]]
			= Aux1[i_int[j_loc] + l_loc * int_dof_counter];

		      j_dof_neighbor_coarsedof[i_dof_neighbor_coarsedof
					      [j_subdomain_dof[j]]] = 
			i_local_to_global[l_loc];
		      i_dof_neighbor_coarsedof[j_subdomain_dof[j]]++;
		      break;
		    }
	      }
	}

      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	i_global_to_local[j_AE_dof[j]] = -1;

    }

  hypre_TFree(i_dof_index);
  hypre_TFree(i_subdomain_dof);
  hypre_TFree(j_subdomain_dof);

  hypre_TFree(i_dof_subdomain);
  hypre_TFree(j_dof_subdomain);


  hypre_TFree(i_Schur_dof_dof);
  hypre_TFree(a_Schur_dof_dof);


  hypre_TFree(AE);

  hypre_TFree(Aux1);

  hypre_TFree(Aux2);

  hypre_TFree(P_boundary);

  hypre_TFree(i_boundary);
  hypre_TFree(i_boundary_to_local);
  hypre_TFree(i_int);
  hypre_TFree(i_interior_to_local);




  /* restore i_dof_neighbor_coarsedof: -------------------------------- */
  for (i=num_dofs; i>0; i--)
    if (i_dof_neighbor_coarsedof[i-1]!=i_dof_neighbor_coarsedof_0[i])
      hypre_printf("WRONG i_dof_neighbor_coarsedof[%d] indexing: %d, %d\n",
	     i, i_dof_neighbor_coarsedof[i-1], i_dof_neighbor_coarsedof_0[i]);

  hypre_TFree(i_dof_neighbor_coarsedof);
  i_dof_neighbor_coarsedof = i_dof_neighbor_coarsedof_0;


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

  i_AE_coarsedof = hypre_CTAlloc(HYPRE_Int, num_AEs+1);
  j_AE_coarsedof = hypre_CTAlloc(HYPRE_Int, AE_coarsedof_counter);

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
  
  hypre_printf("END coarseelement_coarsedof computation: ----------------------------\n");
  

  /* compute coarse element matrices: ----------------------------------------- */


  i_AE_coarsedof_coarsedof = hypre_CTAlloc(HYPRE_Int, num_AEs+1);
  
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
  

  hypre_TFree(i_local_to_global);

  max_local_coarsedof_counter = 0;
  for (i=0; i < num_AEs; i++)
    {
      local_coarsedof_counter = i_AE_coarsedof[i+1]-i_AE_coarsedof[i];
      if (max_local_coarsedof_counter < local_coarsedof_counter)
	max_local_coarsedof_counter = local_coarsedof_counter;
    }

  i_local_to_global = hypre_CTAlloc(HYPRE_Int, max_local_coarsedof_counter);

  for (i=0; i < num_coarsedofs; i++)
    i_global_to_local[i] = -1;

  hypre_TFree(AE);
  hypre_TFree(QE);
  hypre_TFree(W);
  hypre_TFree(Aux1);
  hypre_TFree(Aux2);

  AE = hypre_CTAlloc(double, max_local_coarsedof_counter
		     *max_local_coarsedof_counter);
  QE = hypre_CTAlloc(double, max_local_coarsedof_counter
		     *max_local_coarsedof_counter);

  W  = hypre_CTAlloc(double, max_local_coarsedof_counter);
  Aux1=hypre_CTAlloc(double, max_local_coarsedof_counter);
  Aux2=hypre_CTAlloc(double, max_local_coarsedof_counter);

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

	  /* if (i_dof == j_dof) hypre_printf("diagonal entry %d: %e\n", i_dof,
				     a_AE_chord[j]); */
	  
	  for (k=i_dof_neighbor_coarsedof[i_dof];
	       k<i_dof_neighbor_coarsedof[i_dof+1]; k++)
	    {
	      i_coarsedof = j_dof_neighbor_coarsedof[k];
	      i_loc = i_global_to_local[i_coarsedof];
	      if (i_loc < 0) hypre_printf("wrong index !!!!!!!!!!!!!!!!!!!!!!!! \n");
	      
	      for (l=i_dof_neighbor_coarsedof[j_dof];
		   l<i_dof_neighbor_coarsedof[j_dof+1]; l++)
		{
		  j_coarsedof = j_dof_neighbor_coarsedof[l]; 
		  j_loc = i_global_to_local[j_coarsedof];
		  if (j_loc < 0) hypre_printf("wrong index !!!!!!!!!!!!!!!!!!!!!!!! \n");
		  AE_coarsedof_coarsedof[i_AE_coarsedof_coarsedof[i]
		    +j_loc + local_coarsedof_counter * i_loc]+=
		    P_coeff[k] * a_AE_chord[j] * P_coeff[l];
		}
	    }
	}



      for (i_loc=0; i_loc < local_coarsedof_counter; i_loc++)
	for (j_loc=0; j_loc < local_coarsedof_counter; j_loc++)
	  AE[j_loc + i_loc * local_coarsedof_counter] = 
	    AE_coarsedof_coarsedof[i_AE_coarsedof_coarsedof[i]
				  +j_loc + local_coarsedof_counter * i_loc];
      
      if (local_coarsedof_counter > 0)
	rs_(&local_coarsedof_counter, &local_coarsedof_counter, 
	    AE, W, &matz, QE, Aux1, Aux2, &ierr);

      hypre_printf("end coarse element matrix eigenpair computations: ---------------\n");
      for (i_loc=0; i_loc < local_coarsedof_counter; i_loc++)
	 hypre_printf("%e ", W[i_loc]);
       
      hypre_printf("\n\n");
      
       
      /*
      hypre_printf("\n\n NEW COARSE ELEMENT MATRIX: ======================\n");
      
      for (i_loc=0; i_loc < local_coarsedof_counter; i_loc++)
	{
	  hypre_printf("\n");
	  for (j_loc=0; j_loc < local_coarsedof_counter; j_loc++) 
	    if (j_loc==i_loc) hypre_printf("%e ",
		   AE_coarsedof_coarsedof[i_AE_coarsedof_coarsedof[i]
					 +j_loc + local_coarsedof_counter * i_loc]);
      	  hypre_printf("\n");

	}
	*/

      for (i_loc=0; i_loc < local_coarsedof_counter; i_loc++)
	i_global_to_local[i_local_to_global[i_loc]] = -1;

    }
  
  hypre_TFree(i_AE_coarsedof_coarsedof);

  hypre_TFree(i_AE_chord);
  hypre_TFree(j_AE_chord);
  hypre_TFree(a_AE_chord);
  hypre_printf("END coarseelement_matrix computation: -------------------------------\n");

  /*      
  hypre_printf("num_AEs: %d, num_coarsedofs: %d\n", num_AEs, num_coarsedofs);
  for (i=0; i < num_AEs; i++)
    {
      hypre_printf("coarseelement: %d\n", i);
      
      for (j=i_AE_coarsedof[i]; j < i_AE_coarsedof[i+1]; j++)
	hypre_printf("%d ", j_AE_coarsedof[j]);

      hypre_printf("\n");
      
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


  hypre_printf("END storing coarseelement_matrices in element_chord format: ---------\n");

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

  hypre_printf("\n===============================================================\n");
  hypre_printf("END Build Interpolation Matrix: ierr: %d ======================\n", ierr);
  hypre_printf("===============================================================\n");


  return ierr;
}
				 
/*----------------------------------------------------------------------------
 mat_inv:  X <--  A**(-1) ;  A IS POSITIVE DEFINITE (generally non--symmetric);
 -----------------------------------------------------------------------------*/
      
HYPRE_Int mat_inv(double *x, double *a, HYPRE_Int k)
{
  HYPRE_Int i,j,l, ierr =0;
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
	  hypre_printf("size: %d, diagonal entry: %d, %e\n", k, i, a[i+i*k]);
	    /*	  
	    hypre_printf("mat_inv: ==========================================\n");


            hypre_printf("indefinite singular matrix in *** mat_inv ***:\n");
            hypre_printf("i:%d;  diagonal entry: %e\n", i, a[i+k*i]);


	    for (l=0; l < k; l++)
	      {
		hypre_printf("\n");
		for (j=0; j < k; j++)
		  hypre_printf("%f ", b[j+k*l]);

		hypre_printf("\n");
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
	    hypre_printf("\n non_symmetry: %f %f", x[j+k*i], x[i+k*j] );
    }
  hypre_printf("\n");

   -----------------------------------------------------------------*/

  if (k > 0) hypre_TFree(b);

  return ierr;
}
