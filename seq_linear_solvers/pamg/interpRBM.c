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
                         hypre_CSRMatrix     **P_ptr )
{


  hypre_CSRMatrix    *P; 



  double *Prolong_coeff;
  int *i_dof_neighbor_coarsedof;
  int *j_dof_neighbor_coarsedof;



  double *S_data = hypre_CSRMatrixData(S);
  int *S_i    = hypre_CSRMatrixI(S);
  int *S_j    = hypre_CSRMatrixJ(S);



  int *i_dof_dof = hypre_CSRMatrixI(A);
  int *j_dof_dof = hypre_CSRMatrixJ(A);
  double *a_dof_dof = hypre_CSRMatrixData(A);



                         
  int *fine_to_coarse;



  double *RBM[6];
  int num_RBM = 1;



  int num_dofs = hypre_CSRMatrixNumRows(A);



  int ierr, i,j,k,l, l_loc,k_loc, i_loc, j_loc, i_row;
  int i_dof;
  int *i_local_to_global;
  int *i_global_to_local;



  int i_dof_on_list =-1;



  int local_dof_counter, max_local_dof_counter=0; 
  int fine_node_counter, coarse_node_counter;



  int dof_neighbor_coarsedof_counter = 0, coarsedof_counter = 0;



  int *i_fine, *i_coarse;



  int *i_fine_to_global, *i_coarse_to_global;
  double *AE_neighbor_matrix;
  double *G, *G_inv;
  double *AE, *AE_tilde;
  double *AE_f, *AE_fc, *P_coeff, *XE_f;
  double coeff_sum;



  int system_size = 1;





/* Added by VEH */
  int dof_counter;


  for (k=0; k < num_RBM; k++)
    RBM[k] = hypre_CTAlloc(double, num_dofs);




  dof_counter = 0;
  for (i=0; i < num_dofs; i++)
    for (j=0; j <system_size; j++)
      {
        for (k=0; k < num_RBM; k++)
          if (k == j)
            RBM[k][dof_counter] =1.e0;
          else
            RBM[k][dof_counter] =0.e0;



        dof_counter++;
      }



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



  for (i_dof =0; i_dof < num_dofs; i_dof++)
    if (i_dof_dof[i_dof+1]-i_dof_dof[i_dof] > max_local_dof_counter)
       max_local_dof_counter = i_dof_dof[i_dof+1]-i_dof_dof[i_dof];


  i_local_to_global = hypre_CTAlloc(int, max_local_dof_counter);
  i_global_to_local = hypre_CTAlloc(int, num_dofs); 



  G = hypre_CTAlloc(double, num_RBM * num_RBM);
  G_inv = hypre_CTAlloc(double, num_RBM * num_RBM);




  AE_tilde = hypre_CTAlloc(double, max_local_dof_counter *
                           max_local_dof_counter);



  AE = hypre_CTAlloc(double, max_local_dof_counter *
                           max_local_dof_counter);



  AE_neighbor_matrix = hypre_CTAlloc(double, max_local_dof_counter *
                                     max_local_dof_counter);




  i_fine = hypre_CTAlloc(int, max_local_dof_counter);
  i_coarse = hypre_CTAlloc(int, max_local_dof_counter);



  i_fine_to_global = hypre_CTAlloc(int, max_local_dof_counter);



  i_coarse_to_global = hypre_CTAlloc(int, max_local_dof_counter);




  AE_f = hypre_CTAlloc(double, max_local_dof_counter *
                       max_local_dof_counter);



  AE_fc = hypre_CTAlloc(double, max_local_dof_counter *
                        max_local_dof_counter);



  XE_f = hypre_CTAlloc(double, max_local_dof_counter *
                       max_local_dof_counter);



  P_coeff = hypre_CTAlloc(double, max_local_dof_counter *
                          max_local_dof_counter);



  for (i_dof =0; i_dof < num_dofs; i_dof++)
     i_global_to_local[i_dof] = -1;
     
  for (i_dof =0; i_dof < num_dofs; i_dof++)
    {
      if (CF_marker[i_dof] < 0)
        {


          local_dof_counter = 0;
          for (i=i_dof_dof[i_dof]; 
               i<i_dof_dof[i_dof+1]; i++)
            {
              i_local_to_global[local_dof_counter] = j_dof_dof[i];
              i_global_to_local[j_dof_dof[i]] = local_dof_counter;
              local_dof_counter++;
            }



      
          for (l_loc=0; l_loc < local_dof_counter; l_loc++)
            for (k_loc=0; k_loc < local_dof_counter; k_loc++)
              AE[l_loc*local_dof_counter + k_loc] = 0;



          
          for (i=i_dof_dof[i_dof]; i<i_dof_dof[i_dof+1]; i++)
            {
              l_loc = i_global_to_local[j_dof_dof[i]];
              for(j=i_dof_dof[j_dof_dof[i]]; j<i_dof_dof[j_dof_dof[i]+1]; j++)
                {
                  if (i_global_to_local[j_dof_dof[j]] > -1)
                    {
                      k_loc = i_global_to_local[j_dof_dof[j]];
                      AE[l_loc*local_dof_counter+k_loc] 
                        = a_dof_dof[j];
                    }
                }
            }
 
          /* partition A into a two--level block structure: ---------------- */



          coarse_node_counter = 0;
          fine_node_counter = 0;
          for (i=0; i < local_dof_counter; i++)
            {
              if (CF_marker[i_local_to_global[i]] >=0)
                {
                  /* check if it is on the neighbor list: ------------------ */



                  i_dof_on_list = -1;
                  for (j = i_dof_neighbor_coarsedof[i_dof];
                       j < i_dof_neighbor_coarsedof[i_dof+1]; j++)
                    {
                      if (j_dof_neighbor_coarsedof[j] ==i_local_to_global[i])
                        { 
                          i_coarse[coarse_node_counter] = i;



                          i_coarse_to_global[i] = coarse_node_counter; 
                          coarse_node_counter++;
                          i_dof_on_list++;
                          break;
                        }
                    }



                  if (i_dof_on_list == -1)
                    {
                      i_fine[fine_node_counter] = i;



                      i_fine_to_global[i] = fine_node_counter;
                      fine_node_counter++;
                    }
                }



              if (CF_marker[i_local_to_global[i]] < 0)
                {
                  i_fine[fine_node_counter] = i;
                  i_fine_to_global[i] = fine_node_counter; 
                  fine_node_counter++;
                }
            }



          /* =============================================================
          printf("fine nodes: %d;  coarse nodes: %d\n", fine_node_counter,
                 coarse_node_counter);
          =========================================================== */



          if (fine_node_counter+coarse_node_counter != local_dof_counter)
            {
              printf("error in build_Prolong: %d + %d = %d\n",
                     fine_node_counter, coarse_node_counter, 
                     local_dof_counter);
              return -1;
            }



          /* modify principal matrix using RBM (rigid body motions);    */



          for (i=0; i< num_RBM; i++)
            for (j=0; j< num_RBM; j++)
              {
                G[j+i*num_RBM] = 0.e0;
                for (i_loc=0; i_loc < local_dof_counter; i_loc++)
                  G[j+i*num_RBM]+= RBM[j][i_local_to_global[i_loc]]
                    * RBM[i][i_local_to_global[i_loc]];
              }



          ierr = matinv(G_inv, G, num_RBM);



          for (i_loc =0; i_loc < local_dof_counter; i_loc++)
            for (j_loc =0; j_loc < local_dof_counter; j_loc++)
              AE_tilde[j_loc+i_loc*local_dof_counter] = 0.e0;



          for (i_loc =0; i_loc < local_dof_counter; i_loc++)
            AE_tilde[i_loc+i_loc*local_dof_counter] = 1.e0;



          for (i_loc =0; i_loc < local_dof_counter; i_loc++)
            for (j_loc =0; j_loc < local_dof_counter; j_loc++)
              for (i=0; i< num_RBM; i++)
                for (j=0; j< num_RBM; j++)
                  AE_tilde[j_loc+i_loc*local_dof_counter]-=
                    RBM[i][i_local_to_global[i_loc]] *
                    G_inv[j+i*num_RBM] *
                    RBM[j][i_local_to_global[j_loc]];




          for (i_loc =0; i_loc < local_dof_counter; i_loc++)
            for (j_loc =0; j_loc < local_dof_counter; j_loc++)
              AE_neighbor_matrix[j_loc+i_loc*local_dof_counter] = 0.e0;




          for (i_loc =0; i_loc < local_dof_counter; i_loc++)
            for (j_loc =0; j_loc < local_dof_counter; j_loc++)
              for (l_loc =0; l_loc < local_dof_counter; l_loc++)
                for (k_loc =0; k_loc < local_dof_counter; k_loc++)
                  AE_neighbor_matrix[j_loc+local_dof_counter*i_loc] +=
                    AE_tilde[l_loc+i_loc*local_dof_counter] *
                    AE[k_loc+l_loc*local_dof_counter] *
                    AE_tilde[j_loc+k_loc*local_dof_counter];



          
          for (i=0; i< fine_node_counter; i++)
            {
              for (j=0; j< fine_node_counter; j++)
                {
                  AE_f[i+fine_node_counter*j] = 
                    AE_neighbor_matrix[i_fine[i]+local_dof_counter*i_fine[j]];
                }



              for (j=0; j< coarse_node_counter; j++)
                {
                  AE_fc[i+fine_node_counter*j] = 
                    AE_neighbor_matrix[i_fine[i]+local_dof_counter
                                      *i_coarse[j]];
                }
            }




          /* prolongation matrix P  = -(AE_f)^{-1} AE_{fc};  */



          /* invert AE_f: ------------------------------------------------*/
          ierr = matinv(XE_f, AE_f, fine_node_counter);


          /* printf("local matrix inversion: %d\n", fine_node_counter); */
          if (ierr < 0) printf("ierr_matinv: %d\n", ierr);



        }



      if (CF_marker[i_dof] < 0)
        {
          i_row = i_fine_to_global[i_global_to_local[i_dof]];



          ierr = row_mat_rectmat_prod(P_coeff, XE_f, AE_fc, i_row,
                                      fine_node_counter, coarse_node_counter); 
        }
      
      for (i = i_dof_neighbor_coarsedof[i_dof]; 
           i < i_dof_neighbor_coarsedof[i_dof+1]; i++)
        {
          if (CF_marker[i_dof] < 0)
            {
              j_loc= i_coarse_to_global[i_global_to_local[
                                j_dof_neighbor_coarsedof[i]]]; 



              Prolong_coeff[i] = P_coeff[j_loc];



            }
          else 
            Prolong_coeff[i] = 1.e0;
        }




      for (i=i_dof_dof[i_dof]; i<i_dof_dof[i_dof+1]; i++)
          i_global_to_local[j_dof_dof[i]] = -1;



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



  for (k=0; k < num_RBM; k++)
    hypre_TFree(RBM[k]);



  hypre_TFree(i_coarse);
  hypre_TFree(i_fine);



  hypre_TFree(i_coarse_to_global);
  hypre_TFree(i_fine_to_global);



  hypre_TFree(G);
  hypre_TFree(G_inv);



  hypre_TFree(AE_neighbor_matrix);
  hypre_TFree(AE);
  hypre_TFree(AE_tilde);
  hypre_TFree(XE_f);
  hypre_TFree(AE_f);
  hypre_TFree(AE_fc);



  hypre_TFree(P_coeff);
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
      if (a[i+i*k] <= 1.e-20)
        {
            printf("indefinite singular matrix in *** matinv ***:\n");
            printf("i:%d;  diagonal entry: %e\n", i, a[i+k*i]);



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
