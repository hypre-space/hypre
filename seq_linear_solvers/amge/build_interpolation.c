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
 * needed graphs: AE_dof, dof_AE = (AE_dof)^T,
 *                dof_neighbor_coarsedof, 
 *                dof_coarsedof;
 * builds:
 *        hypre_CSRMatrix P;
 *        (coarseelement matrices: hypre_CSRMatrix)
 *                 coarseelement_coarsechord;
 *        graph:   coarsechord_coarsedof;
 ****************************************************************************/
#include "headers.h" 

int hypre_AMGeBuildInterpolation(hypre_CSRMatrix     **P_pointer,

				 int **i_coarseelement_coarsechord_pointer,
				 int **j_coarseelement_coarsechord_pointer,
				 double **a_coarseelement_coarsechord_pointer,

				 int **i_coarsechord_coarsedof_pointer, 
				 int **j_coarsechord_coarsedof_pointer,

				 int *num_coarsechords_pointer,

				 int *i_AE_dof, int *j_AE_dof,
				 int *i_dof_AE, int *j_dof_AE,

				 int *i_dof_neighbor_coarsedof,
				 int *j_dof_neighbor_coarsedof,

				 int *i_dof_coarsedof,
				 int *j_dof_coarsedof,

				 int *i_AE_chord,
				 int *j_AE_chord,
				 double *a_AE_chord,

				 int *i_chord_dof, 
				 int *j_chord_dof,

				 int num_chords,

				 int num_AEs, 
				 int num_dofs,
				 int num_coarsedofs)

{
  int ierr = 0;
  int i,j;

  int *i_AE_coarsedof, *j_AE_coarsedof;

  int *i_domain_AE, *j_domain_AE;
  int *i_domain_chord, *j_domain_chord;
  double *a_domain_chord;

  int *i_domain_dof, *j_domain_dof;
  int *i_subdomain_dof, *j_subdomain_dof;
  int *i_Schur_dof_dof;
  double *a_Schur_dof_dof, *P_coeff;
  
  hypre_CSRMatrix  *P;


  int *i_coarseelement_coarsechord, *j_coarseelement_coarsechord;
				 
  double *a_coarseelement_coarsechord;

  int *i_coarsechord_coarsedof, *j_coarsechord_coarsedof;
  int num_coarsechords;

  int num_domains;

  int domain_counter, 
    domain_AE_counter, subdomain_dof_counter, Schur_dof_dof_counter,
    dof_neighbor_coarsedof_counter, AE_coarsedof_counter;

  double diag, row_sum;

  int *i_dof_index; 

  num_domains = num_dofs - num_coarsedofs;

  i_domain_AE = hypre_CTAlloc(int, num_domains+1);
  domain_AE_counter=0;
  for (i=0; i < num_dofs; i++)
    if (i_dof_coarsedof[i+1] == i_dof_coarsedof[i])
      domain_AE_counter+= i_dof_AE[i+1] - i_dof_AE[i];

  j_domain_AE =  hypre_CTAlloc(int, domain_AE_counter);

  domain_AE_counter=0;
  domain_counter= 0;
  for (i=0; i < num_dofs; i++)
    {
      if (i_dof_coarsedof[i+1] == i_dof_coarsedof[i])
	{
	  i_domain_AE[domain_counter] = domain_AE_counter;
	  domain_counter++;

	  for (j=i_dof_AE[i]; j < i_dof_AE[i+1]; j++)
	    {
	      j_domain_AE[domain_AE_counter] = j_dof_AE[j];
	    domain_AE_counter++;
	    }
	}
    }

  i_domain_AE[num_domains] = domain_AE_counter;


  i_subdomain_dof = hypre_CTAlloc(int, num_domains+1);
  subdomain_dof_counter = 0;
  for (i=0; i < num_dofs; i++)
    {
      if (i_dof_coarsedof[i+1] == i_dof_coarsedof[i])
	subdomain_dof_counter+=1+i_dof_neighbor_coarsedof[i+1]-
	  i_dof_neighbor_coarsedof[i];
    }

  j_subdomain_dof = hypre_CTAlloc(int, subdomain_dof_counter);


  subdomain_dof_counter = 0;
  domain_counter = 0;
  for (i=0; i < num_dofs; i++)
    {
      if (i_dof_coarsedof[i+1] == i_dof_coarsedof[i])
	{
	  i_subdomain_dof[domain_counter] = subdomain_dof_counter;
	  domain_counter++;

	  j_subdomain_dof[subdomain_dof_counter]=i;
	  subdomain_dof_counter++;
	  for (j = i_dof_neighbor_coarsedof[i]; 
	       j < i_dof_neighbor_coarsedof[i+1]; j++)
	    {
	      j_subdomain_dof[subdomain_dof_counter]=
		j_dof_neighbor_coarsedof[j];
	      subdomain_dof_counter++;
	    }
	}
    }

  i_subdomain_dof[num_domains] = subdomain_dof_counter;


  printf("---------------------- Assembling neighborhood matrices: ------------\n");
  ierr=hypre_AMGeDomainElementSparseAssemble(i_domain_AE, j_domain_AE,
					     num_domains,

					     i_AE_chord,
					     j_AE_chord,
					     a_AE_chord,

					     i_chord_dof, 
					     j_chord_dof,

					     &i_domain_chord,
					     &j_domain_chord,
					     &a_domain_chord,

					     num_AEs, num_chords,
					     num_dofs);

  printf("END assembling neighborhood matrices: ----------------------------\n");


  ierr = matrix_matrix_product(&i_domain_dof, &j_domain_dof,

			       i_domain_AE, j_domain_AE,
			       i_AE_dof, j_AE_dof,

			       num_domains, num_AEs, num_dofs);

  hypre_TFree(i_domain_AE);
  hypre_TFree(j_domain_AE);


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

  hypre_TFree(i_subdomain_dof);
  hypre_TFree(j_subdomain_dof);

  hypre_TFree(i_domain_chord);
  hypre_TFree(j_domain_chord);
  hypre_TFree(a_domain_chord);



  hypre_TFree(i_Schur_dof_dof);
  
  P_coeff = hypre_CTAlloc(double, i_dof_neighbor_coarsedof[num_dofs]);

  dof_neighbor_coarsedof_counter = 0;
  Schur_dof_dof_counter = 0;
  for (i=0; i < num_dofs; i++)
    {
      if (i_dof_coarsedof[i+1] > i_dof_coarsedof[i])
	{
	  P_coeff[dof_neighbor_coarsedof_counter] = 1.e0;
	  dof_neighbor_coarsedof_counter++;
	}	  
      else
	{
	  row_sum = 0.e0;
	  diag = a_Schur_dof_dof[Schur_dof_dof_counter];
	  Schur_dof_dof_counter++;
	  for (j=i_dof_neighbor_coarsedof[i];
	       j<i_dof_neighbor_coarsedof[i+1]; j++)
	    {
	      P_coeff[dof_neighbor_coarsedof_counter] = 
		-a_Schur_dof_dof[Schur_dof_dof_counter]/ diag;

	      row_sum+=P_coeff[dof_neighbor_coarsedof_counter];
	      Schur_dof_dof_counter++;

	      dof_neighbor_coarsedof_counter++;
	    }

	  /*
	  printf("                      row_sum: %e\n", row_sum);
	  */
	  Schur_dof_dof_counter+=(i_dof_neighbor_coarsedof[i+1]
				  -i_dof_neighbor_coarsedof[i])*
	                         (i_dof_neighbor_coarsedof[i+1]
				  -i_dof_neighbor_coarsedof[i]+1);
	}
    }

  /* compute coarse element matrices: --------------------------- */

  hypre_TFree(a_Schur_dof_dof);


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
	    j_AE_coarsedof[AE_coarsedof_counter] = j_AE_dof[j];
	    AE_coarsedof_counter++;
	  }
    }

  i_AE_coarsedof[num_AEs] = AE_coarsedof_counter;

  /* compute "non--conforming" coarse element matrices: -------------- */

  printf("--------------------- Computing non--conforming coarse element matrices:\n");
  ierr =  hypre_AMGeSchurComplement(i_AE_chord,
				    j_AE_chord,
				    a_AE_chord,

				    i_chord_dof, j_chord_dof,
			      
				    i_AE_dof,
				    j_AE_dof,

				    i_AE_coarsedof,
				    j_AE_coarsedof,

				    &i_Schur_dof_dof,
				    &a_Schur_dof_dof,

				    num_AEs, num_chords, num_dofs);

  printf("END computing non--conforming coarse element matrices: ==========\n");
  hypre_TFree(i_Schur_dof_dof);
  
  AE_coarsedof_counter = 0;
  for (i=0; i < num_AEs; i++)
    for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
      if (i_dof_coarsedof[j_AE_dof[j]+1] > i_dof_coarsedof[j_AE_dof[j]])
	{
	  j_AE_coarsedof[AE_coarsedof_counter]=
	    j_dof_coarsedof[i_dof_coarsedof[j_AE_dof[j]]];
	  AE_coarsedof_counter++;
	}


  ierr = hypre_AMGeElementMatrixDof(i_AE_coarsedof, j_AE_coarsedof,
				    a_Schur_dof_dof,

				    &i_coarseelement_coarsechord,
				    &j_coarseelement_coarsechord,
				    &a_coarseelement_coarsechord,

				    &i_coarsechord_coarsedof,
				    &j_coarsechord_coarsedof,

				    &num_coarsechords,

				    num_AEs, num_coarsedofs);

  hypre_TFree(i_AE_coarsedof);
  hypre_TFree(j_AE_coarsedof);

  *i_coarseelement_coarsechord_pointer = i_coarseelement_coarsechord;
  *j_coarseelement_coarsechord_pointer = j_coarseelement_coarsechord;
  *a_coarseelement_coarsechord_pointer = a_Schur_dof_dof;

  *i_coarsechord_coarsedof_pointer = i_coarsechord_coarsedof;
  *j_coarsechord_coarsedof_pointer = j_coarsechord_coarsedof;
  *num_coarsechords_pointer = num_coarsechords;
  
  
  for (i=0; i < num_dofs; i++)
    for (j=i_dof_neighbor_coarsedof[i]; j<i_dof_neighbor_coarsedof[i+1]; j++)
      j_dof_neighbor_coarsedof[j] = 
	j_dof_coarsedof[i_dof_coarsedof[j_dof_neighbor_coarsedof[j]]];

  P = hypre_CSRMatrixCreate(num_dofs, num_coarsedofs,
			    i_dof_neighbor_coarsedof[num_dofs]);



  hypre_CSRMatrixData(P) = P_coeff;
  hypre_CSRMatrixI(P) = i_dof_neighbor_coarsedof; 
  hypre_CSRMatrixJ(P) = j_dof_neighbor_coarsedof; 

  *P_pointer = P;

  /*
  printf("===============================================================\n");
  printf("END Build Interpolation Matrix: ===============================\n");
  printf("===============================================================\n");
  */

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
	  printf("                        diagonal entry: %e\n", a[i+k*i]);
	    /*	  
	    printf("matinv: ==========================================\n");
	    printf("size: %d, entry: %d, %f\n", k, i, a[i+i*k]);

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

  hypre_TFree(b);

  return ierr;
}
