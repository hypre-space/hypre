/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/



#include "headers.h"  



/*--------------------------------------------------------------------------
 * hypre_AMGBuildRBMInterp
 *--------------------------------------------------------------------------*/



int
hypre_AMGBuildRBMInterp( hypre_CSRMatrix     *A,
                         int                 *CF_marker,
                         hypre_CSRMatrix     *S,
                         int                 *dof_func,
                         int                 num_functions,
                         int                 **coarse_dof_func_ptr,
                         hypre_CSRMatrix     **P_ptr )
{


  hypre_CSRMatrix    *P; 
  int                *coarse_dof_func;


  double *Prolong_coeff;
  int *i_dof_neighbor_coarsedof;
  int *j_dof_neighbor_coarsedof;



  int *S_i    = hypre_CSRMatrixI(S);
  int *S_j    = hypre_CSRMatrixJ(S);



  int *i_dof_dof = hypre_CSRMatrixI(A);
  int *j_dof_dof = hypre_CSRMatrixJ(A);
  double *a_dof_dof = hypre_CSRMatrixData(A);


  int *i_ext_int, *j_ext_int;

  int ext_int_counter;
                         
  int *fine_to_coarse;



  int num_dofs = hypre_CSRMatrixNumRows(A);


  int ierr = 0;
  int i, j, k, l_loc, i_loc, j_loc;
  int i_dof, j_dof;
  int *i_local_to_global;
  int *i_global_to_local;


/*  int i_dof_on_list =-1; */


  int local_dof_counter, max_local_dof_counter=0; 
  int fine_node_counter, coarse_node_counter;



  int dof_neighbor_coarsedof_counter = 0, coarsedof_counter = 0,
    dof_counter = 0;



  int *i_fine, *i_coarse;


  int *i_int;

  int *i_fine_to_global, *i_coarse_to_global;


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



  i_dof_neighbor_coarsedof = hypre_CTAlloc(int, num_dofs+1);
  j_dof_neighbor_coarsedof = hypre_CTAlloc(int, 
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


  i_global_to_local = hypre_CTAlloc(int, num_dofs); 


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


  i_local_to_global = hypre_CTAlloc(int, max_local_dof_counter);


  AE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);

  
  i_fine = hypre_CTAlloc(int, max_local_dof_counter);
  i_coarse = hypre_CTAlloc(int, max_local_dof_counter);


  i_fine_to_global = hypre_CTAlloc(int, max_local_dof_counter);



  i_coarse_to_global = hypre_CTAlloc(int, max_local_dof_counter);


  
  i_int = hypre_CTAlloc(int, max_local_dof_counter);

  P_ext_int = hypre_CTAlloc(double, max_local_dof_counter *
			    max_local_dof_counter);


  /*
  for (i_loc =0; i_loc < max_local_dof_counter; i_loc++)
    for (j_loc =0; j_loc < max_local_dof_counter; j_loc++)
      P_ext_int[j_loc + i_loc * max_local_dof_counter] = 0.e0;

      */

  i_ext_int = hypre_CTAlloc(int, max_local_dof_counter+1);
  j_ext_int = hypre_CTAlloc(int, max_local_dof_counter *
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
          printf("fine nodes: %d;  coarse nodes: %d\n", fine_node_counter,
                 coarse_node_counter);
	   =========================================================== */



          if (fine_node_counter+coarse_node_counter != dof_counter)
            {
              printf("error in build_Prolong: %d + %d = %d\n",
                     fine_node_counter, coarse_node_counter, 
                     dof_counter);
              return -1;
            }




	  /*
	  printf("local_dof_counter: %d, dof_counter: %d\n",
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
      printf("\ndof %d: has coefficients:\n", i_dof);
      coeff_sum = 0.0;
      for (i = i_dof_neighbor_coarsedof[i_dof]; 
           i < i_dof_neighbor_coarsedof[i_dof+1]; i++)
        {
          printf(" %f ", Prolong_coeff[i]);
          coeff_sum=coeff_sum+Prolong_coeff[i];
        }
      printf("\n coeff_sum: %f \n\n", coeff_sum);
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
       coarse_dof_func = hypre_CTAlloc(int, coarsedof_counter);

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
int row_mat_rectmat_prod(double *a1,
                         double *a2,
                         double *a3,
                         int i_row, int m, int n)
{
  int i,l, ierr =0;



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
      
int matinv(double *x, double *a, int k)
{
  int i,j,l, ierr =0;



  for (i=0; i < k; i++)
    {
      if (a[i+i*k] <= 0.e0)
        {
	  if (i < k-1)
	    {
	      /*********
	      printf("indefinite singular matrix in *** matinv ***:\n");
	      printf("i:%d;  diagonal entry: %e\n", i, a[i+k*i]);
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

