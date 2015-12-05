/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/







#include "headers.h"  



/*--------------------------------------------------------------------------
 * hypre_AMGBuildRBMInterp
 *--------------------------------------------------------------------------*/



HYPRE_Int
hypre_AMGBuildRBMInterp( hypre_CSRMatrix     *A,
                         HYPRE_Int                 *CF_marker,
                         hypre_CSRMatrix     *S,
                         HYPRE_Int                 *dof_func,
                         HYPRE_Int                 num_functions,
                         HYPRE_Int                 **coarse_dof_func_ptr,
                         hypre_CSRMatrix     **P_ptr )
{


  hypre_CSRMatrix    *P; 
  HYPRE_Int                *coarse_dof_func;


  double *Prolong_coeff;
  HYPRE_Int *i_dof_neighbor_coarsedof;
  HYPRE_Int *j_dof_neighbor_coarsedof;



  HYPRE_Int *S_i    = hypre_CSRMatrixI(S);
  HYPRE_Int *S_j    = hypre_CSRMatrixJ(S);



  HYPRE_Int *i_dof_dof = hypre_CSRMatrixI(A);
  HYPRE_Int *j_dof_dof = hypre_CSRMatrixJ(A);
  double *a_dof_dof = hypre_CSRMatrixData(A);


  HYPRE_Int *i_ext_int, *j_ext_int;

  HYPRE_Int ext_int_counter;
                         
  HYPRE_Int *fine_to_coarse;



  HYPRE_Int num_dofs = hypre_CSRMatrixNumRows(A);


  HYPRE_Int ierr = 0;
  HYPRE_Int i, j, k, l_loc, i_loc, j_loc;
  HYPRE_Int i_dof, j_dof;
  HYPRE_Int *i_local_to_global;
  HYPRE_Int *i_global_to_local;


/*  HYPRE_Int i_dof_on_list =-1; */


  HYPRE_Int local_dof_counter, max_local_dof_counter=0; 
  HYPRE_Int fine_node_counter, coarse_node_counter;



  HYPRE_Int dof_neighbor_coarsedof_counter = 0, coarsedof_counter = 0,
    dof_counter = 0;



  HYPRE_Int *i_fine, *i_coarse;


  HYPRE_Int *i_int;

  HYPRE_Int *i_fine_to_global, *i_coarse_to_global;


  double *AE;

/*  double coeff_sum; */

  double *P_ext_int; 

  double diag = 0.e0;
 


  /*-----------------------------------------------------------------------
   *  First Pass: Determine size of Prolong;
   *-----------------------------------------------------------------------*/



  dof_neighbor_coarsedof_counter = 0;
      
  /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/
    
  for (i = 0; i < num_dofs; i++)
    {
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity. 
       *--------------------------------------------------------------------*/


      if (CF_marker[i] >= 0)
        {
          dof_neighbor_coarsedof_counter++;
        }
      
      /*--------------------------------------------------------------------
       *  If i is a f-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/



      else
        {
          for (j = S_i[i]; j < S_i[i+1]; j++)
            {
              i_dof = S_j[j];           
              if (CF_marker[i_dof] >= 0)
                {
                  dof_neighbor_coarsedof_counter++;
                }
            }
        }
    }
  
  /*-----------------------------------------------------------------------
   *  Allocate  arrays.
   *-----------------------------------------------------------------------*/



  i_dof_neighbor_coarsedof = hypre_CTAlloc(HYPRE_Int, num_dofs+1);
  j_dof_neighbor_coarsedof = hypre_CTAlloc(HYPRE_Int, 
                                           dof_neighbor_coarsedof_counter);



  Prolong_coeff = hypre_CTAlloc(double, dof_neighbor_coarsedof_counter);



  dof_neighbor_coarsedof_counter = 0;



  for (i = 0; i < num_dofs; i++)
    {
      i_dof_neighbor_coarsedof[i] = dof_neighbor_coarsedof_counter;
      /*--------------------------------------------------------------------
       *  If i is a c-point, the neighbor is i;
       *--------------------------------------------------------------------*/
      if (CF_marker[i] >= 0)
        {
          j_dof_neighbor_coarsedof[dof_neighbor_coarsedof_counter] = i;
          dof_neighbor_coarsedof_counter++;
        }
      
      /*--------------------------------------------------------------------
       *  If i is a f-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/


      else
        {
          for (j = S_i[i]; j < S_i[i+1]; j++)
            {
              i_dof = S_j[j];           
              if (CF_marker[i_dof] >= 0)
                {
                  j_dof_neighbor_coarsedof[dof_neighbor_coarsedof_counter] 
                    = i_dof;
                  dof_neighbor_coarsedof_counter++;
                }
            }
        }
    }



  i_dof_neighbor_coarsedof[num_dofs] = dof_neighbor_coarsedof_counter;


  i_global_to_local = hypre_CTAlloc(HYPRE_Int, num_dofs); 


  for (i_dof =0; i_dof < num_dofs; i_dof++)
     i_global_to_local[i_dof] = -1;

  for (i_dof =0; i_dof < num_dofs; i_dof++)
    {
      if (CF_marker[i_dof] < 0)
	{
	  local_dof_counter = 0;

	  for (j=i_dof_dof[i_dof]; j < i_dof_dof[i_dof+1]; 
	       j++)
	    {
	      j_dof = j_dof_dof[j];

	      if (i_global_to_local[j_dof] < 0)
		{
		  i_global_to_local[j_dof] = local_dof_counter;
		  local_dof_counter++;
		}

	    }
	

	  if (local_dof_counter > max_local_dof_counter)
	    max_local_dof_counter = local_dof_counter;

	  for (j=i_dof_dof[i_dof]; j < i_dof_dof[i_dof+1]; 
	       j++)
	    {
	      j_dof = j_dof_dof[j];
	      i_global_to_local[j_dof] = -1;
	    }	       
	}

    }


  i_local_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);


  AE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);

  
  i_fine = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);
  i_coarse = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);


  i_fine_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);



  i_coarse_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);


  
  i_int = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);

  P_ext_int = hypre_CTAlloc(double, max_local_dof_counter *
			    max_local_dof_counter);


  /*
  for (i_loc =0; i_loc < max_local_dof_counter; i_loc++)
    for (j_loc =0; j_loc < max_local_dof_counter; j_loc++)
      P_ext_int[j_loc + i_loc * max_local_dof_counter] = 0.e0;

      */

  i_ext_int = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter+1);
  j_ext_int = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter *
			    max_local_dof_counter);
  

  for (l_loc=0; l_loc < max_local_dof_counter; l_loc++)
    i_int[l_loc] = -1;


  for (i_dof =0; i_dof < num_dofs; i_dof++)
    {
      if (CF_marker[i_dof] < 0)
        {
          local_dof_counter = 0;

          for (j=i_dof_dof[i_dof]; j<i_dof_dof[i_dof+1]; j++)
            {
	      j_dof = j_dof_dof[j];

	      if (i_global_to_local[j_dof] < 0)
		{
		  i_local_to_global[local_dof_counter] = j_dof;
		  i_global_to_local[j_dof] = local_dof_counter;
		  local_dof_counter++;
		}
	    }

	  dof_counter = 0;
	  i_int[i_global_to_local[i_dof]]=dof_counter;
	  dof_counter++;

	  for (j = i_dof_neighbor_coarsedof[i_dof]; 
	       j < i_dof_neighbor_coarsedof[i_dof+1]; j++)
	    {
	      j_dof = j_dof_neighbor_coarsedof[j];
	      if (i_int[i_global_to_local[j_dof]] < 0)
		{
		  i_int[i_global_to_local[j_dof]] = dof_counter;
		  dof_counter++;
		}
	    }

	  for (i=0; i < dof_counter; i++)
	    i_coarse_to_global[i] = -1;

          coarse_node_counter = 0;
	  for (j = i_dof_neighbor_coarsedof[i_dof]; 
	       j < i_dof_neighbor_coarsedof[i_dof+1]; j++)
	    {
	      i = i_global_to_local[j_dof_neighbor_coarsedof[j]];

	      i_coarse[coarse_node_counter] = i_int[i];
	      i_coarse_to_global[i_int[i]] = coarse_node_counter; 
	      coarse_node_counter++;
	    }
	  

          fine_node_counter = 0;
          for (i=0; i < local_dof_counter; i++)
	    if (i_int[i] > -1)
	      {
		if (i_coarse_to_global[i_int[i]] < 0)
		  {
		    i_fine[fine_node_counter] = i_int[i];

		    i_fine_to_global[i_int[i]] = fine_node_counter;
		    fine_node_counter++;
		  }
	      }


          /* ============================================================
          hypre_printf("fine nodes: %d;  coarse nodes: %d\n", fine_node_counter,
                 coarse_node_counter);
	   =========================================================== */



          if (fine_node_counter+coarse_node_counter != dof_counter)
            {
              hypre_printf("error in build_Prolong: %d + %d = %d\n",
                     fine_node_counter, coarse_node_counter, 
                     dof_counter);
              return -1;
            }




	  /*
	  hypre_printf("local_dof_counter: %d, dof_counter: %d\n",
		 local_dof_counter, dof_counter); */

	  ext_int_counter = 0;
	  for (i_loc =0; i_loc < local_dof_counter; i_loc++)
	    {
	      i_ext_int[i_loc] = ext_int_counter;

	      if (i_int[i_loc] >=0)
		{
		  P_ext_int[i_loc + i_int[i_loc] * local_dof_counter] = 1.e0;
		  j_ext_int[ext_int_counter] = i_loc;
		  ext_int_counter++;
		}
	      else
		{
		  /* find the neighbors of i_local_to_global[i_loc] */

		  if (num_functions > 1)
		    k = dof_func[i_local_to_global[i_loc]];
		  diag = 0.e0;
		  
		  for (j=i_dof_dof[i_local_to_global[i_loc]];
		       j<i_dof_dof[i_local_to_global[i_loc]+1]; j++)
		    {
		      j_dof = j_dof_dof[j];
		      if (i_global_to_local[j_dof] >= 0)
			if (i_int[i_global_to_local[j_dof]] >= 0)
			  {
			    if (num_functions > 1) 
			      if (dof_func[j_dof] == k) 
				{
				  j_ext_int[ext_int_counter]= 
				    i_global_to_local[j_dof];
				  ext_int_counter++;
				  P_ext_int[i_loc + i_int[i_global_to_local[j_dof]]
					   *local_dof_counter]= fabs(a_dof_dof[j]);

				  diag +=fabs(a_dof_dof[j]);
				}
			    if (num_functions== 1) 
			      {
				j_ext_int[ext_int_counter]= 
				  i_global_to_local[j_dof];
				ext_int_counter++;
				P_ext_int[i_loc + i_int[i_global_to_local[j_dof]]
					 *local_dof_counter]= fabs(a_dof_dof[j]);

				diag +=fabs(a_dof_dof[j]);
			      }
			  }
		    }

		  if (diag > 0.e0)
		    for (j=i_ext_int[i_loc]; j < ext_int_counter; j++)
		      P_ext_int[i_loc + i_int[j_ext_int[j]]*local_dof_counter]
			/=diag;

		}

	    }

	  i_ext_int[local_dof_counter] = ext_int_counter;

	  /* multiply AE times P_ext_int: ================================== */

	  for (j_loc =0; j_loc < dof_counter; j_loc++)
	    AE[i_int[i_global_to_local[i_dof]]+j_loc * dof_counter]= 0.e0;

	  i_loc = i_global_to_local[i_dof];

	  /* for (l_loc =0; l_loc < local_dof_counter; l_loc++) */

	  for (i=i_dof_dof[i_dof]; i < i_dof_dof[i_dof+1]; i++)
	    {
	      l_loc = i_global_to_local[j_dof_dof[i]];
	      for (j=i_ext_int[l_loc]; j < i_ext_int[l_loc+1]; j++)
		{
		  j_loc = j_ext_int[j];
		  AE[i_int[i_loc]+i_int[j_loc] * dof_counter]+=
		    a_dof_dof[i] * 
		    P_ext_int[l_loc + i_int[j_loc] * local_dof_counter];
		}
	    }
	}

      for (i = i_dof_neighbor_coarsedof[i_dof]; 
           i < i_dof_neighbor_coarsedof[i_dof+1]; i++)
        {
          if (CF_marker[i_dof] < 0)
            {
              j_loc= i_coarse_to_global[i_int[i_global_to_local[
			        j_dof_neighbor_coarsedof[i]]]]; 

	      if (AE[i_fine[0]+dof_counter*i_fine[0]] !=0.e0)
		Prolong_coeff[i] = -AE[i_fine[0]+dof_counter *i_coarse[j_loc]]
		  /AE[i_fine[0]+dof_counter*i_fine[0]];
	      else 
		Prolong_coeff[i] = 0.e0;	

	    }
          else 
            Prolong_coeff[i] = 1.e0;
        }

      if (CF_marker[i_dof] < 0)
	{
	  i_int[i_global_to_local[i_dof]]=-1;

	  for (j = i_dof_neighbor_coarsedof[i_dof]; 
	       j < i_dof_neighbor_coarsedof[i_dof+1]; j++)
	    {
	      j_dof = j_dof_neighbor_coarsedof[j];
	      i_int[i_global_to_local[j_dof]] = -1;
	    }
	  

	  for (j=i_dof_dof[i_dof]; j < i_dof_dof[i_dof+1]; 
	       j++)
	    {
	      j_dof = j_dof_dof[j];
	      i_global_to_local[j_dof] = -1;
	    }	       
	}

    }

  /*-----------------------------------------------------------------
  for (i_dof =0; i_dof < num_dofs; i_dof++)
    {
      hypre_printf("\ndof %d: has coefficients:\n", i_dof);
      coeff_sum = 0.0;
      for (i = i_dof_neighbor_coarsedof[i_dof]; 
           i < i_dof_neighbor_coarsedof[i_dof+1]; i++)
        {
          hypre_printf(" %f ", Prolong_coeff[i]);
          coeff_sum=coeff_sum+Prolong_coeff[i];
        }
      hypre_printf("\n coeff_sum: %f \n\n", coeff_sum);
    }
  -----------------------------------------------------------------*/



   fine_to_coarse = i_global_to_local;



   coarsedof_counter = 0;
   for (i=0; i < num_dofs; i++)
     if (CF_marker[i] >=0)
       {
         fine_to_coarse[i] = coarsedof_counter;
         coarsedof_counter++;
       }
     else 
       fine_to_coarse[i] = -1;




   P = hypre_CSRMatrixCreate(num_dofs, coarsedof_counter, 
                             i_dof_neighbor_coarsedof[num_dofs]);



   hypre_CSRMatrixData(P) = Prolong_coeff;
   hypre_CSRMatrixI(P) = i_dof_neighbor_coarsedof; 
   hypre_CSRMatrixJ(P) = j_dof_neighbor_coarsedof; 



   for (i=0; i < num_dofs; i++)
     for (j=i_dof_neighbor_coarsedof[i];
          j<i_dof_neighbor_coarsedof[i+1]; j++)
       hypre_CSRMatrixJ(P)[j] = fine_to_coarse[j_dof_neighbor_coarsedof[j]];



   *P_ptr = P;


   if (num_functions > 1)
     {
       coarse_dof_func = hypre_CTAlloc(HYPRE_Int, coarsedof_counter);

       coarsedof_counter=0;
       for (i=0; i < num_dofs; i++)
	 if (CF_marker[i] >=0)
	   {
	     coarse_dof_func[coarsedof_counter] = dof_func[i];
	     coarsedof_counter++;
	   }

       /* return coarse_dof_func array: ---------------------------------------*/

       *coarse_dof_func_ptr = coarse_dof_func;
     }

  hypre_TFree(i_int);

  hypre_TFree(i_coarse);
  hypre_TFree(i_fine);

  hypre_TFree(i_coarse_to_global);
  hypre_TFree(i_fine_to_global);



  hypre_TFree(AE);


  hypre_TFree(i_ext_int);
  hypre_TFree(j_ext_int);
  hypre_TFree(P_ext_int);



  hypre_TFree(i_global_to_local);
  hypre_TFree(i_local_to_global);



  return ierr;



}
/*---------------------------------------------------------------------
 row_mat_rectmat_prod:    A1[i_row][0:n-1] <---  -A2[i_row][0:m-1]
                                                * A3[0:m-1][0:n-1];
---------------------------------------------------------------------*/
HYPRE_Int row_mat_rectmat_prod(double *a1,
                         double *a2,
                         double *a3,
                         HYPRE_Int i_row, HYPRE_Int m, HYPRE_Int n)
{
  HYPRE_Int i,l, ierr =0;



  for (i=0; i < n; i++)
    {
      a1[i] = 0;
      for (l=0; l < m; l++)
            a1[i] -= a2[i_row+l*m] * a3[l+i*m];
    }



  return ierr;
}
/*---------------------------------------------------------------------
 matinv:  X <--  A**(-1) ;  A IS POSITIVE DEFINITE (non--symmetric);
 ---------------------------------------------------------------------*/
      
HYPRE_Int matinv(double *x, double *a, HYPRE_Int k)
{
  HYPRE_Int i,j,l, ierr =0;



  for (i=0; i < k; i++)
    {
      if (a[i+i*k] <= 0.e0)
        {
	  if (i < k-1)
	    {
	      /*********
	      hypre_printf("indefinite singular matrix in *** matinv ***:\n");
	      hypre_printf("i:%d;  diagonal entry: %e\n", i, a[i+k*i]);
	      */
	      ierr = -1;
	    }

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



  return ierr;
}

