/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/
/*****************************************************************************
 * AE matrices stored in format:
 *  i_AE_chord,
 *  j_AE_chord,
 *  a_AE_chord; 
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
 *        graph:   coarsechord_coarsedof;
 ****************************************************************************/
#include "headers.h" 

int hypre_AMGeBuildInterpolation(hypre_CSRMatrix     **P_pointer,


				 int *i_AE_dof, int *j_AE_dof,
				 int *i_dof_AE, int *j_dof_AE,

				 int *i_dof_neighbor_coarsedof,
				 int *j_dof_neighbor_coarsedof,

				 int *i_dof_coarsedof,
				 int *j_dof_coarsedof,

				 hypre_CSRMatrix  *Matrix,


				 int *dof_function, 
				 int num_functions,

				 int **coarsedof_function_pointer, 

				 int num_AEs, 
				 int num_dofs,
				 int num_coarsedofs)

{
  int ierr = 0;
  int i,j,k,l, i_dof, j_dof, i_loc, j_loc, k_loc;

  int *i_dof_dof_a, *j_dof_dof_a;
  double *a_dof_dof;


  double *P_coeff;
  
  hypre_CSRMatrix  *P;


  int *coarsedof_function;


  double delta; 


  double *AE, *XE;

  int local_dof_counter, int_dof_counter, boundary_dof_counter;
  int coarsedof_counter;

  int max_local_dof_counter = 0;
  int AE_coarsedof_counter;

  int *i_global_to_local, *i_local_to_global;


  /* ------------------------------------------------------------------ */
  /* first build P for dofs that belong to more than one AE; ---------- */
  /* ------------------------------------------------------------------ */
  /* then, for each AE: 

           (1)   find interior dofs and boundary dofs, 

		 a dof (i_dof) in AE is boundary if it belongs to another AE,
		 i.e., if 

                     i_dof_AE[i_dof+1] - i_dof_AE[i_dof] > 1;

	   (2)   partition AE into blocks:

		     AE_ii  AE_ix  AE_ic   interior dofs
		     AE_xi  AE_xx  AE_xc   boundary fine dofs
		     AE_ci  AE_cx  AE_cc   (boundary) coarse dofs;



            (3)  extract P_boundarydof_coarsedof from P, already computed 
	         based on averaging (element--free interpolation), i.e., 
		 
		 (P_boundarydof_coarsedof)^T = [(P_xc)^T, I];
		

	    (4) compute       

                      X_ic = AE_ix \times  P_xc + A_ic;

	    (4) compute 

	              P_ic = -(AE_ii)^{-1} \times X_ic,
		      (based on super-LU with multiple r.h.s. 
		      -- the columns of X_ic);

	    (5) then P_E (i.e., P restricted to AE) is defined as:

                    P_dof_coarsedof = P_ic, for interior dofs,

		                    = P_xc, for boundary fine dofs;

                                    = I for coarse dofs;


            (6)  compute coarse element matrix:

                 (P_E)^T AE P_E;
		                                                        */
  /* ------------------------------------------------------------------ */

  i_dof_dof_a = hypre_CSRMatrixI(Matrix);
  j_dof_dof_a = hypre_CSRMatrixJ(Matrix);
  a_dof_dof   = hypre_CSRMatrixData(Matrix);

  P_coeff = hypre_CTAlloc(double, i_dof_neighbor_coarsedof[num_dofs]);

  i_global_to_local = hypre_CTAlloc(int, num_dofs); 


  for (i_dof =0; i_dof < num_dofs; i_dof++)
     i_global_to_local[i_dof] = -1;

  for (i_dof =0; i_dof < num_dofs; i_dof++)
    {
      if (i_dof_coarsedof[i_dof+1] > i_dof_coarsedof[i_dof])
	for (j=i_dof_neighbor_coarsedof[i_dof];
	     j<i_dof_neighbor_coarsedof[i_dof+1]; j++)
	  P_coeff[j] = 1.e0;	
      else 
	if (i_dof_AE[i_dof+1] > i_dof_AE[i_dof]+1)
	  {
	    k = dof_function[i_dof];

	    /* define interpolation based on averaging: ----------------- */
	    delta = 0.e0;
	    for (j=i_dof_neighbor_coarsedof[i_dof];
		 j<i_dof_neighbor_coarsedof[i_dof+1]; j++)
	      if (dof_function[j_dof_neighbor_coarsedof[j]] == k)
		delta++;

	    for (j=i_dof_neighbor_coarsedof[i_dof];
		 j<i_dof_neighbor_coarsedof[i_dof+1]; j++)
	      if (dof_function[j_dof_neighbor_coarsedof[j]] == k)
		P_coeff[j] = 1.e0/delta;
	      else
		P_coeff[j] = 0.e0;
	  }
    }

  max_local_dof_counter = 0;
  for (i=0; i < num_AEs; i++)
    {
      local_dof_counter = i_AE_dof[i+1] - i_AE_dof[i];
	if (local_dof_counter > max_local_dof_counter)
	  max_local_dof_counter = local_dof_counter;
    }
	    
  i_local_to_global = hypre_CTAlloc(int, max_local_dof_counter);


  AE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);

  XE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);



  for (i=0; i < num_AEs; i++)
    {
      local_dof_counter = 0;
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	{
	  if (i_dof_AE[j_AE_dof[j]+1] == i_dof_AE[j_AE_dof[j]]+1)
	    {
	      i_local_to_global[local_dof_counter] = j_AE_dof[j];
	      i_global_to_local[j_AE_dof[j]] = local_dof_counter;
	      local_dof_counter++;
	    }
	}

      int_dof_counter = local_dof_counter;

      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	{
	  if (i_dof_AE[j_AE_dof[j]+1] > i_dof_AE[j_AE_dof[j]]+1
	      && i_dof_coarsedof[j_AE_dof[j]+1] == i_dof_coarsedof[j_AE_dof[j]])
	    {
	      i_local_to_global[local_dof_counter] = j_AE_dof[j];
	      i_global_to_local[j_AE_dof[j]] = local_dof_counter;
	      local_dof_counter++;
	    }
	}

      boundary_dof_counter = local_dof_counter - int_dof_counter;

      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	{
	  if (i_dof_AE[j_AE_dof[j]+1] > i_dof_AE[j_AE_dof[j]]+1
	      && i_dof_coarsedof[j_AE_dof[j]+1] >  i_dof_coarsedof[j_AE_dof[j]])
	    {
	      i_local_to_global[local_dof_counter] = j_AE_dof[j];
	      i_global_to_local[j_AE_dof[j]] = local_dof_counter;
	      local_dof_counter++;
	    }
	}

      coarsedof_counter = local_dof_counter 
	                - boundary_dof_counter 
	                - int_dof_counter;


      for (i_loc =0; i_loc < int_dof_counter; i_loc++)
	  for (j_loc =0; j_loc < local_dof_counter; j_loc++)
	    AE[i_loc + int_dof_counter * j_loc] = 0.e0;

      /* ---------------------------------------------------------
      for (l=i_AE_chord[i]; l < i_AE_chord[i+1]; l++)
	{
	  k = j_AE_chord[l];
	  i_dof = j_chord_dof[i_chord_dof[k]];
	  j_dof = j_chord_dof[i_chord_dof[k]+1];

	  i_loc = i_global_to_local[i_dof];
	  j_loc = i_global_to_local[j_dof];

	  if (i_loc < int_dof_counter)
	    AE[i_loc + int_dof_counter * j_loc] = a_AE_chord[l];
	}
	---------------------------------------------------------- */

      for (j = i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	{
	  i_dof = j_AE_dof[j];
	  i_loc = i_global_to_local[i_dof];
	  if (i_loc < int_dof_counter)
	    for (k=i_dof_dof_a[i_dof]; k<i_dof_dof_a[i_dof+1]; k++)
	      {
		j_dof = j_dof_dof_a[k];
		j_loc = i_global_to_local[j_dof];
		AE[i_loc + int_dof_counter * j_loc] = a_dof_dof[k];
	      }
	}

      if (int_dof_counter > 0)
	{
	  ierr = matinv(XE, AE, int_dof_counter);
	  if (ierr == -1)
	    {
	      printf("============= build_interpolation: ===============\n");
	      printf("Indefinite principal submatrix AE_ii: +++++++++\n");
	      printf("==================================================\n");
	      return ierr;
	    }
	}

      for (i_loc = 0; i_loc < int_dof_counter; i_loc++)
	for (k_loc = int_dof_counter; 
	     k_loc < int_dof_counter+boundary_dof_counter; k_loc++)
	  {
	    k = i_local_to_global[k_loc];
	    for (j=i_dof_neighbor_coarsedof[k];
		 j<i_dof_neighbor_coarsedof[k+1]; j++)
	      {
		j_loc = i_global_to_local[j_dof_neighbor_coarsedof[j]];
		AE[i_loc + j_loc * int_dof_counter] +=
		  P_coeff[j] * AE[i_loc + int_dof_counter * k_loc];
	      }
	  }
      
      for (i_loc = 0; i_loc < int_dof_counter; i_loc++)
	{
	  i_dof = i_local_to_global[i_loc];
	  for (j=i_dof_neighbor_coarsedof[i_dof];
	       j<i_dof_neighbor_coarsedof[i_dof+1]; j++)
	    {
	      P_coeff[j] = 0.e0;
	      
	      j_loc = i_global_to_local[j_dof_neighbor_coarsedof[j]];
	      for (k_loc = 0; k_loc < int_dof_counter; k_loc++)
		P_coeff[j] -= XE[i_loc + int_dof_counter * k_loc]
		  * AE[k_loc + int_dof_counter * j_loc];
	    }
	}
      
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	i_global_to_local[j_AE_dof[j]] = -1;

    }

  hypre_TFree(i_global_to_local);  
  hypre_TFree(XE);
  hypre_TFree(AE);
  hypre_TFree(i_local_to_global);


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

  coarsedof_function = hypre_CTAlloc(int, num_coarsedofs);

  for (i=0; i < num_dofs; i++)
    if (i_dof_coarsedof[i+1] > i_dof_coarsedof[i])
      coarsedof_function[j_dof_coarsedof[i_dof_coarsedof[i]]]
	    = dof_function[i];

  *coarsedof_function_pointer = coarsedof_function;

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
