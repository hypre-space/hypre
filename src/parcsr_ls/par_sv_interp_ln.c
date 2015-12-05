#include "headers.h"
#include "Common.h"

#define SV_DEBUG 0

/******************************************************************************
  hypre_BoomerAMG_LNExpandInterp

  * This is the LN approach as described in Baker, Kolev and Yang,
    "Improving AMG interpolation operators for linear elasticity problems"

 * so we first refine the current interpolation and then use it to add the
   local variant approach  

 *NOTE: currently assumes that we have either 2 or 3 orig functions
        (we assume that we are adding 1 dof for 2D and 3 dof for 3D)

 *MUST USE NODAL COARSENING! (and so unknowns interlaced)

 *note: changes num_functions and updates dof_array if level = 0

******************************************************************************/

HYPRE_Int hypre_BoomerAMG_LNExpandInterp( hypre_ParCSRMatrix *A,
                                    hypre_ParCSRMatrix **P,
                                    HYPRE_Int *num_cpts_global,
                                    HYPRE_Int *nf, 
                                    HYPRE_Int *dof_func,
                                    HYPRE_Int **coarse_dof_func,
                                    HYPRE_Int *CF_marker, 
                                    HYPRE_Int level,
                                    double *weights,
                                    HYPRE_Int num_smooth_vecs, 
                                    hypre_ParVector **smooth_vecs, 
                                    double abs_trunc, HYPRE_Int q_max,
                                    HYPRE_Int interp_vec_first_level  ) 
{

   HYPRE_Int                i,j, k,kk, pp, jj;

   hypre_ParCSRMatrix *new_P;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
   double          *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);


   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(*P);
   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int             *P_diag_j = hypre_CSRMatrixJ(P_diag);

   HYPRE_Int              num_rows_P = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int              num_cols_P = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int              P_diag_size = P_diag_i[num_rows_P];

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(*P);
   HYPRE_Int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int              P_offd_size = P_offd_i[num_rows_P];

   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int             *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Int              num_cols_P_offd = hypre_CSRMatrixNumCols(P_offd);
   HYPRE_Int             *col_map_offd_P = hypre_ParCSRMatrixColMapOffd(*P);

   HYPRE_Int             *col_starts = hypre_ParCSRMatrixColStarts(*P);
   
   HYPRE_Int             *new_col_map_offd_P = NULL;

   double           orig_row_sum, new_row_sum, gm_row_sum;
   
   HYPRE_Int              orig_diag_start, orig_offd_start, j_offd_pos, j_diag_pos;
   HYPRE_Int              new_nnz_diag, new_nnz_offd;
   HYPRE_Int              fcn_num, p_num_elements, p_num_diag_elements;
   HYPRE_Int              p_num_offd_elements;
   
   
   double           *P_diag_data_new, *P_offd_data_new;
   HYPRE_Int              *P_diag_j_new, *P_diag_i_new, *P_offd_i_new, *P_offd_j_new;

   HYPRE_Int              ncv, ncv_peru;
   
   HYPRE_Int              orig_nf, orig_ncv, new_ncv;

   HYPRE_Int              found, new_col, cur_col;

   HYPRE_Int              num_functions = *nf;

   double          *smooth_vec_offd = NULL;
   double          *smooth_vec_offd_P = NULL;
   double          *offd_vec_data;
   double          *offd_vec_data_P;

   HYPRE_Int              nnz_diag, nnz_offd;

   hypre_ParVector *vector;
   double          *vec_data;

   hypre_ParCSRCommPkg     *comm_pkg_P = hypre_ParCSRMatrixCommPkg(*P);
   hypre_ParCSRCommPkg     *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);

   MPI_Comm        comm;

   HYPRE_Int             coarse_counter, d_sign;
   HYPRE_Int             j_ext_index;

   HYPRE_Int            *fine_to_coarse;
   HYPRE_Int             kk_point, jj_point, jj_point_c, fine_kk, p_point;
   
   double          diagonal, aw, a_ij;

   HYPRE_Int             modify;
   
   HYPRE_Int             add_q;
   HYPRE_Int             num_new_p_diag, num_new_p_offd;
   HYPRE_Int             kk_count;
   
   HYPRE_Int             cur_spot;

   HYPRE_Int             i1;

   HYPRE_Int            *c_dof_func = *coarse_dof_func;

   double          q_val, tmp_d1, tmp_d2;
   double          adj[3], r_extra[3];

   HYPRE_Int            *col_map;
   HYPRE_Int            *coarse_to_fine;
   

   HYPRE_Int            *new_col_starts;
   
   double          af_sum;

   double          theta_2D[] = {.5, .5};
   double          theta_3D[] = {1.0/3.0, 1.0/3.0, 1.0/3.0};

   double          *theta;

   double          sum;

   HYPRE_Int             no_fc;

   hypre_ParCSRCommHandle  *comm_handle;

   HYPRE_Int             use_alt_w, num_f;
   HYPRE_Int             dist_coarse;

   HYPRE_Int             *CF_marker_offd = NULL;
   HYPRE_Int             *dof_func_offd = NULL;
   HYPRE_Int             *fine_to_coarse_offd;

   hypre_CSRMatrix *P_ext;
   double          *P_ext_data;
   HYPRE_Int             *P_ext_i;
   HYPRE_Int             *P_ext_j;
   HYPRE_Int              num_sends_A, index, start;
   HYPRE_Int              myid = 0, num_procs = 1;

   /* truncation */
   HYPRE_Int *is_q = NULL;
   HYPRE_Int q_alloc= 0;
   HYPRE_Int *aux_j = NULL;
   double *aux_data = NULL;
   HYPRE_Int    *is_diag = NULL;
   HYPRE_Int q_count, p_count_diag, p_count_offd;

   HYPRE_Int no_fc_use_gm;
   
   HYPRE_Int num_sends;

   HYPRE_Int loop_q_max;

   HYPRE_Int  *int_buf_data = NULL;
   double *dbl_buf_data = NULL;
   
   HYPRE_Int g_nc;
   

#if SV_DEBUG
   {
      char new_file[80];
      
      hypre_sprintf(new_file,"%s.level.%d","P_orig", level);
      hypre_ParCSRMatrixPrint(*P, new_file); 

      for (i=0; i < num_smooth_vecs; i++)
      {
         hypre_sprintf(new_file,"%s.%d.level.%d","smoothvec", i, level );
         hypre_ParVectorPrint(smooth_vecs[i], new_file); 
      }
   }
#endif
   /* must have a comm pkg */
   if (!comm_pkg_P)
   {
      hypre_MatvecCommPkgCreate ( *P ); 
      comm_pkg_P = hypre_ParCSRMatrixCommPkg(*P);
   }
   comm   = hypre_ParCSRCommPkgComm(comm_pkg_A);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);


#if SV_DEBUG
   {
      char new_file[80];
      
      hypre_CSRMatrix *P_CSR = NULL;
      hypre_Vector *sv = NULL;

      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(*P);

      if (!myid)
      {
         hypre_sprintf(new_file,"%s.level.%d","P_new_orig", level );
         if (P_CSR)
            hypre_CSRMatrixPrint(P_CSR, new_file); 

      }
      
      hypre_CSRMatrixDestroy(P_CSR);

      for (i=0; i < num_smooth_vecs; i++)
      {
         sv = hypre_ParVectorToVectorAll(smooth_vecs[i]);
         
         if (!myid)
         {
            hypre_sprintf(new_file,"%s.%d.level.%d","smoothvec", i, level );
            if (sv)
               hypre_SeqVectorPrint(sv, new_file); 
         }
         
         hypre_SeqVectorDestroy(sv);
         
      }

      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
      if (!myid)
      {
         hypre_sprintf(new_file,"%s.level.%d","A", level );
         if (P_CSR)
            hypre_CSRMatrixPrint(P_CSR, new_file); 
      }
      
      hypre_CSRMatrixDestroy(P_CSR);

   }
#endif

   /*initialize */
   no_fc_use_gm = 1; /* use GM approach when no fine connections */
   modify = 1; /* this indicates to replace P_s
                  based on P_u and P_v */

   ncv = num_cols_P; /* num coarse variables */
   nnz_diag = P_diag_size;
   nnz_offd = P_offd_size;

   /*number of coarse variables for each unknown */
   ncv_peru = ncv/num_functions;
   
   if (level == interp_vec_first_level)
   {
      orig_nf = num_functions;
      orig_ncv = ncv;
   }
   else /* on deeper levels, need to know orig sizes (without new
         * dofs) */
   {
      orig_nf = num_functions - num_smooth_vecs;
      orig_ncv = ncv - ncv_peru*num_smooth_vecs;
   }
  
  /*weights for P_s */
   if (modify)
   {
      if (weights == NULL)
      {
         if (orig_nf == 2)
            theta = theta_2D;
         else
            theta = theta_3D;
      }
      else
      {
         theta = weights;
      }
   }
   
   /* for communication */
   num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg_A,
                                                                     num_sends_A));


   /*-----------------------------------------------------------------------
    *  create and send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 
  
   {
      HYPRE_Int my_first_cpt, tmp_i;
    
  
#ifdef HYPRE_NO_GLOBAL_PARTITION
      my_first_cpt = num_cpts_global[0];
#else
      my_first_cpt = num_cpts_global[myid];
#endif
      
      /* create the fine to coarse and coarse to fine*/
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int, num_rows_P);
      for (i = 0; i < num_rows_P; i++) fine_to_coarse[i] = -1;
      
      coarse_to_fine = hypre_CTAlloc(HYPRE_Int, ncv);

      coarse_counter = 0;
      for (i=0; i < num_rows_P; i++)
      {
         if (CF_marker[i] >= 0)
         {
            fine_to_coarse[i] = coarse_counter;
            coarse_to_fine[coarse_counter] = i;
            coarse_counter++;
         }
      }
      
      /* now from other procs */
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd); 
      
      index = 0;
      for (i = 0; i < num_sends_A; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i+1); j++)
         {
            
            tmp_i = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A,j)];
            int_buf_data[index++] = tmp_i + my_first_cpt; /* makes it global*/
         }
         
      }
      
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg_A, int_buf_data, 
                                                  fine_to_coarse_offd);  
      
      hypre_ParCSRCommHandleDestroy(comm_handle);   
   } /* end fine to coarse {} */
  
   /*-------------------------------------------------------------------
   * Get the CF_marker data for the off-processor columns of A
   *-------------------------------------------------------------------*/
   {

      if (num_cols_A_offd) 
         CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);
      
      if (num_functions > 1 && num_cols_A_offd)
         dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);
      
      index = 0;
      for (i = 0; i < num_sends_A; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i+1); j++)
         {
            int_buf_data[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A,j)];
         }
         
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg_A, int_buf_data, 
                                                  CF_marker_offd);
      
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      if (num_functions > 1)
      {
         index = 0;
         for (i = 0; i < num_sends_A; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
            for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i+1); j++)
            {
               int_buf_data[index++] 
                  = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A,j)];
            }
            
         }
         
         comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg_A, int_buf_data, 
                                                     dof_func_offd);
         
         hypre_ParCSRCommHandleDestroy(comm_handle);   
      }
   
   } /* end cf marker {} */

   /*-------------------------------------------------------------------
    * Get the ghost rows of P
    *-------------------------------------------------------------------*/
  {
      
      HYPRE_Int kc;
      HYPRE_Int col_1 = hypre_ParCSRMatrixFirstColDiag(*P);
      HYPRE_Int col_n = col_1 + hypre_CSRMatrixNumCols(P_diag);

      if (num_procs > 1)
      {
         /* need the rows of P on other processors associated with
            the offd cols of A */
         P_ext      = hypre_ParCSRMatrixExtractBExt(*P,A,1);
         P_ext_i    = hypre_CSRMatrixI(P_ext);
         P_ext_j    = hypre_CSRMatrixJ(P_ext);
         P_ext_data = hypre_CSRMatrixData(P_ext);
      }
      
      index = 0;
      /* now check whether each col is in the diag of offd part of P)*/
      for (i=0; i < num_cols_A_offd; i++)
      {
         for (j=P_ext_i[i]; j < P_ext_i[i+1]; j++)
         {
            k = P_ext_j[j];
            /* is it in the diag ?*/
            if (k >= col_1 && k < col_n)
            {
               P_ext_j[index] = k - col_1;  /* make a local col number */
               P_ext_data[index++] = P_ext_data[j];
            }
            else
            {
               /* off diag entry */
               kc = hypre_BinarySearch(col_map_offd_P, k ,num_cols_P_offd);
               /* now this corresponds to the location in the col_map_offd
                ( so it is a local column number */
               if (kc > -1)
               {
                  P_ext_j[index] = -kc-1; /* make negative */
                  P_ext_data[index++] = P_ext_data[j];
               }
            }
         }
         P_ext_i[i] = index;
      }
      for (i = num_cols_A_offd; i > 0; i--)
         P_ext_i[i] = P_ext_i[i-1];

      if (num_procs > 1) P_ext_i[0] = 0;


   } /* end of ghost rows */

   /*-------------------------------------------------------------------
    * Allocations
    *-------------------------------------------------------------------*/

   /* if level = first_level, we need to fix the col numbering to leave
    * space for the new unknowns */
   col_map = hypre_CTAlloc(HYPRE_Int, ncv);
   
   if (num_smooth_vecs && level == interp_vec_first_level)
   {
      for (i = 0; i < ncv; i++)
      {
         /* map from old col number to new col number (leave spaces
          * for new unknowns to be interleaved */
         col_map[i] = i + (i/num_functions) * num_smooth_vecs;
      }
   }
   else
   {
      for (i = 0; i < ncv; i++)
      {
         /* map from old col number to new col number */
         col_map[i] = i;
      }
   }
   
   /* we will have the same sparsity in Q as in P */
   new_nnz_diag = nnz_diag + nnz_diag*num_smooth_vecs;
   new_nnz_offd = nnz_offd + nnz_offd*num_smooth_vecs;


   /* new number of coarse variables */
   if (level == interp_vec_first_level )
      new_ncv = ncv + ncv_peru*num_smooth_vecs;
   else
      new_ncv = ncv; /* unchanged on level > first_level */


   /* allocations */
   P_diag_j_new = hypre_CTAlloc(HYPRE_Int, new_nnz_diag);
   P_diag_data_new = hypre_CTAlloc (double, new_nnz_diag);
   P_diag_i_new = hypre_CTAlloc(HYPRE_Int, num_rows_P + 1);
   
   P_offd_j_new = hypre_CTAlloc(HYPRE_Int, new_nnz_offd);
   P_offd_data_new = hypre_CTAlloc (double, new_nnz_offd);
   P_offd_i_new = hypre_CTAlloc(HYPRE_Int, num_rows_P + 1);
   
   P_diag_i_new[0] = P_diag_i[0];
   P_offd_i_new[0] = P_offd_i[0];
   
   
   /* doing truncation? if so, need some more allocations*/
   if (q_max > 0 || abs_trunc > 0.0)
   {
      q_count = 0;
      for (i=0; i < num_rows_P; i++)
      {
         p_num_elements = P_diag_i[i+1]-P_diag_i[i];
         p_num_elements += (P_offd_i[i+1]-P_offd_i[i]);
         if (p_num_elements > q_count) q_count = p_num_elements;
      }
      q_alloc =  q_count*(num_smooth_vecs + 1);
      is_q = hypre_CTAlloc (HYPRE_Int, q_alloc );
      aux_data = hypre_CTAlloc(double, q_alloc);
      aux_j = hypre_CTAlloc(HYPRE_Int, q_alloc);
      is_diag = hypre_CTAlloc (HYPRE_Int, q_alloc );
   }

   /*-------------------------------------------------------------------
    * Get smooth vec components for the off-processor columns of A
    *-------------------------------------------------------------------*/
   if (num_procs > 1)
   {
      HYPRE_Int fine_index;
      
      smooth_vec_offd =  hypre_CTAlloc(double, num_cols_A_offd*num_smooth_vecs);
      
      /* for now, do a seperate comm for each smooth vector */
      for (k = 0; k< num_smooth_vecs; k++)
      {
         
         vector = smooth_vecs[k];
         vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
         
         dbl_buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg_A,
                                                                              num_sends_A));
         /* point into smooth_vec_offd */
         offd_vec_data =  smooth_vec_offd + k*num_cols_A_offd;
         
         index = 0;
         for (i = 0; i < num_sends_A; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i+1); j++)
            {   
               
               fine_index = hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A,j);
               
               dbl_buf_data[index++] = vec_data[fine_index];
            }
            
         }
         
         comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg_A, dbl_buf_data, 
                                                     offd_vec_data);
         hypre_ParCSRCommHandleDestroy(comm_handle); 
          
         hypre_TFree(dbl_buf_data);
      } /* end of smooth vecs */
   }/*end num procs > 1 */


    /*-------------------------------------------------------------------
     * Get smooth vec components for the off-processor columns of P 
     *  TO Do: would be less storage to get the offd coarse to fine
     *  instead of this...
     *-------------------------------------------------------------------*/

   if (num_procs > 1)
   {
      HYPRE_Int c_index, fine_index;
      smooth_vec_offd_P =  hypre_CTAlloc(double, num_cols_P_offd*num_smooth_vecs);
      
      /* for now, do a seperate comm for each smooth vector */
      for (k = 0; k< num_smooth_vecs; k++)
      {
         
         vector = smooth_vecs[k];
         vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
         
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg_P);
         dbl_buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg_P,
                                                                              num_sends));
         /* point into smooth_vec_offd_P */
         offd_vec_data_P =  smooth_vec_offd_P + k*num_cols_P_offd;
         
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_P, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_P, i+1); j++)
            {   
               /* we need to do the coarse/fine conversion here */
               c_index = hypre_ParCSRCommPkgSendMapElmt(comm_pkg_P,j);
               fine_index = coarse_to_fine[c_index];      
               dbl_buf_data[index++] = vec_data[fine_index];
            }
            
         }
         
         comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg_P, dbl_buf_data, 
                                                     offd_vec_data_P);
         hypre_ParCSRCommHandleDestroy(comm_handle); 
         
         hypre_TFree(dbl_buf_data);
      }

   }/*end num procs > 1 */

   
   /*-------------------------------------------------------------------
    * Main loop!
    *-------------------------------------------------------------------*/
   

    /******** loop through rows - only operate on rows of original functions******/

   j_diag_pos = 0;
   j_offd_pos = 0;
   orig_diag_start = 0;
   orig_offd_start = 0;
   
   for (i=0; i < num_rows_P; i++)
   {
      orig_row_sum = 0.0;
      new_row_sum = 0.0;
      num_new_p_diag = 0;
      num_new_p_offd = 0;
      
      no_fc = 0;
      
      p_count_diag = 0;/* number of entries of p added */
      p_count_offd = 0;
      q_count = 0; /* number of entries of q added */
      for (j = 0; j< q_alloc; j++)
      {
         is_q[j] = 0;
      }
      
      fcn_num = (HYPRE_Int) fmod(i, num_functions);
      if (fcn_num != dof_func[i]) 
         hypre_printf("WARNING - ROWS incorrectly ordered in hypre_BoomerAMG_LNExpandInterp! myid = %d, row = %d\n", myid, i);
      
      /* number of elements in row of p*/
      p_num_diag_elements = P_diag_i[i+1] - P_diag_i[i];
      p_num_offd_elements = P_offd_i[i+1] - P_offd_i[i];
      
      num_new_p_diag = p_num_diag_elements;
      num_new_p_offd = p_num_offd_elements;
      
      orig_diag_start =  P_diag_i[i];
      orig_offd_start =  P_offd_i[i];
      

      /* if original function dofs? or a new one that we don't want
       * to modify*/ 
      if (fcn_num < orig_nf || modify == 0 )
      {
         
         /* for this row, will we add q entries ? */
         if (fcn_num < orig_nf && num_smooth_vecs)
            add_q = 1;
         else
            add_q = 0;
         
         if (CF_marker[i] >= 0) /* row corres. to coarse point - just copy orig */
         {
            /* diag elements */
            for (j=0; j < p_num_diag_elements; j++)
            {
               P_diag_data_new[j_diag_pos] = P_diag_data[orig_diag_start+j];
               
               new_col = col_map[ P_diag_j[orig_diag_start+j]];
               P_diag_j_new[j_diag_pos] =new_col;;
               
               j_diag_pos++;
               
               p_count_diag++;
            }
            
            /* offd elements */
            p_count_offd = p_count_diag;
            for (j=0; j < p_num_offd_elements; j++)
            {
               P_offd_data_new[j_diag_pos] = P_offd_data[orig_offd_start+j];
               
               /* note that even though we are copying, j
                  needs to go back to regular numbering - will be
                  compressed later when col_map_offd is generated*/
               index = P_offd_j[orig_offd_start+j];

               /* convert to the global col number using col_map_offd */
               index = col_map_offd_P[index];
               
               /*now adjust for the new dofs - since we are offd, can't
                * use col_map[index]*/
               if (num_smooth_vecs && (level == interp_vec_first_level))
               {
                  new_col = index + (index/num_functions) * num_smooth_vecs;
               }
               else /* no adjustment */
               {
                  new_col = index;
               }
               
               P_offd_j_new[j_diag_pos] = new_col;;
               
               j_offd_pos++;
               p_count_offd++;
            }
         }
         else /* row is for fine point  - make new interpolation*/
         {
            /* make orig entries zero and make space for the
              entries of q */

            /* diag entries */
            for (j=0; j < p_num_diag_elements; j++)
            {
               orig_row_sum +=  P_diag_data[orig_diag_start+j];
               P_diag_data_new[j_diag_pos] = 0.0;
               
               new_col = col_map[ P_diag_j[orig_diag_start+j]];
               P_diag_j_new[j_diag_pos] = new_col;;
               
               j_diag_pos++;
               
               if (q_alloc)
                  is_q[p_count_diag] = 0; /* this entry is for orig p*/
               p_count_diag++;
               if (add_q)
               {
                  cur_col = new_col;
                  for (k = 0; k < num_smooth_vecs; k++)
                  {
                     new_col = cur_col + (orig_nf - fcn_num) + k;
                     P_diag_j_new[j_diag_pos]    = new_col;
                     P_diag_data_new[j_diag_pos] = 0.0;
                     j_diag_pos++;
                     
                     if (q_alloc)
                        is_q[p_count_diag] = k+1; /* this entry is for smoothvec k*/
                     
                     num_new_p_diag++;
                     q_count++;
                     p_count_diag++;
                     
                  }
               }
            }
            /* offd */
            p_count_offd = p_count_diag; /* for indexing into is_q*/
            for (j=0; j < p_num_offd_elements; j++)
            {
               orig_row_sum +=  P_offd_data[orig_offd_start+j];
               P_offd_data_new[j_offd_pos] = 0.0;
               
               /* j needs to go back to regular numbering - will be
                  compressed later when col_map_offd is generated*/
               index = P_offd_j[orig_offd_start+j];
               
               /* convert to the global col number using col_map_offd */
               index = col_map_offd_P[index];
               
               /*now adjust for the new dofs - since we are offd, can't
                * use col_map[index]*/
               if (num_smooth_vecs && (level == interp_vec_first_level))
               {
                  new_col = index + (index/num_functions) * num_smooth_vecs;
               }
               else /* no adjustment */
               {
                  new_col = index;
               }
               
               P_offd_j_new[j_offd_pos] = new_col;;
               
               j_offd_pos++;
               
               if (q_alloc)
                  is_q[p_count_offd] = 0; /* this entry is for orig p*/
               
               p_count_offd++;
               if (add_q)
               {
                  cur_col = new_col;
                  for (k = 0; k < num_smooth_vecs; k++)
                  {
                     new_col = cur_col + (orig_nf - fcn_num) + k;
                     P_offd_j_new[j_offd_pos]    = new_col;
                     P_offd_data_new[j_offd_pos] = 0.0;
                     j_offd_pos++;
                     
                     if (q_alloc)
                        is_q[p_count_offd] = k+1; /* this entry is for smoothvec k*/
                     
                     num_new_p_offd++;
                     q_count++;
                     p_count_offd++;
                     
                  }
               }
            }

            /* find r for adjustment (this is r/sum(Af) as in eqn
             * (31) of paper )*/
            for (k = 0; k < num_smooth_vecs; k++)
            {
               r_extra[k] = 0.0;
            }
            if (p_num_diag_elements || p_num_offd_elements)
            {
               for (k = 0; k < num_smooth_vecs; k++)
               {
                  vector = smooth_vecs[k];
                   vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                   
                   for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                   {
                      
                      i1 = A_diag_j[jj];
                      if (dof_func[i1] == fcn_num)
                         r_extra[k] += A_diag_data[jj]*vec_data[i1];
                      
                   }
                   
                   offd_vec_data =  smooth_vec_offd + k*num_cols_A_offd;
                   
                   for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                   {
                      
                      i1 = A_offd_j[jj];
                      if (dof_func_offd[i1] == fcn_num)
                         r_extra[k] += A_offd_data[jj]*offd_vec_data[i1];
                      
                   }
               }
               /*find sum(a_if) */
               af_sum = 0.0;
                
               for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
               { 
                  i1 = A_diag_j[jj];
                  if (dof_func[i1] == fcn_num && CF_marker[i1] < 0)
                     af_sum +=  A_diag_data[jj];
                  
               }
               for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
               { 
                  i1 = A_offd_j[jj];
                  if (dof_func_offd[i1] == fcn_num && CF_marker_offd[i1] < 0)
                     af_sum +=  A_offd_data[jj];
                  
               }

               if (af_sum != 0.0)
               {
                  for (k = 0; k < num_smooth_vecs; k++)
                  {
                     adj[k] = r_extra[k]/af_sum;
                  }
               }
               else /* there are no fine connections */
               {
                  no_fc = 1;
               }
               /* now we will use the adjustment later */


               /* now if we have any coarse connections with no
                  corresponding point in orig p, then these we have to
                  distibute and treat as fine, basically*/
               
               /* diag first */
               for (jj = A_diag_i[i]+ 1; jj < A_diag_i[i+1]; jj++)
               {
                  found = 0;
                  jj_point = A_diag_j[jj]; /* fine index */
                  
                  /* only want like unknowns */
                  if (fcn_num != dof_func[jj_point])
                     continue;
                  
                  /*only look at coarse connections */
                  if (CF_marker[jj_point] < 0) /*fine*/
                     continue;
                  
                  a_ij = A_diag_data[jj];
                  
                  jj_point_c = fine_to_coarse[jj_point];
                  new_col = col_map[jj_point_c];
                  /* is there a P(i,j_c)? */ 
                  
                  for (kk = P_diag_i_new[i]; kk < P_diag_i_new[i] + num_new_p_diag; kk ++)
                  {
                     if (P_diag_j_new[kk] == new_col)
                     {
                        found = 1;
                        break;
                     }
                  }
                  if (!found) /* this will be distributed and treated as an F 
                                 point - so add to the sum) */
                  {
                     af_sum += a_ij;
                     if (af_sum != 0.0)
                     {
                        for (k = 0; k < num_smooth_vecs; k++)
                        {
                           adj[k] = r_extra[k]/af_sum;
                        }
                     }
                  }
               } /* end diag loop */
               /* now offd loop */
               for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
               {
                  found = 0;
                  jj_point = A_offd_j[jj]; /* fine index */
                  
                  /* only want like unknowns */
                  if (fcn_num != dof_func_offd[jj_point])
                     continue;
                  
                  /*only look at coarse connections */
                  if (CF_marker_offd[jj_point] < 0) /*fine*/
                     continue;
                  
                  a_ij = A_offd_data[jj];
                  
                  jj_point_c = fine_to_coarse_offd[jj_point]; /* now global num */
                  /* now need to adjust for new cols */
                  /* TO DO:  VERIFY THIS! */
                  jj_point_c  = jj_point_c + (jj_point_c/num_functions) * num_smooth_vecs;
                  
                  /* is there a P(i,jj_c)? */ 
                  for (kk = P_offd_i_new[i]; kk < P_offd_i_new[i] + num_new_p_offd; kk ++)
                  {
                     
                     index = P_offd_j_new[kk]; /* global number */
                     /* and this index has been adjusted to make room for
                      * new cols */
                     
                     if (index == jj_point_c)
                     {
                        found = 1;
                        break;
                     }
                  }
                  if (!found) /* this will be distributed and treated as an F 
                                 point - so add to the sum) */
                  {
                     af_sum += a_ij;
                     if (af_sum != 0.0)
                     {
                        for (k = 0; k < num_smooth_vecs; k++)
                        {
                           adj[k] = r_extra[k]/af_sum;
                        }
                     }
                  }
               } /* end offd loop */
               
               /* end of checking for coarse connections to treat as fine*/
                

               if (no_fc)/* recheck in case there were weak coarse connections 
                            that will be treated as fine */
               {
                  if (af_sum != 0.0)
                  {
                     no_fc = 0;
                  }
               }
               
               /* Need to use GM for this row? */
               if (no_fc && add_q && no_fc_use_gm)
               {
#if 0                 
                  hypre_printf("Warning - no fine connections to distribute in level = %d, i = %d\n", level, i);
#endif                   
                  /* need to get the row-sum - we will to the GM approach for these
                     rows! (var 6 )*/
                  gm_row_sum = 0.0;
                  
                  for (j=0; j < p_num_diag_elements; j++)
                  {
                     gm_row_sum +=  P_diag_data[orig_diag_start+j];
                  }
                  for (j=0; j < p_num_offd_elements; j++)
                  {
                     gm_row_sum +=  P_offd_data[orig_offd_start+j];
                  }
                  if( (p_num_diag_elements+p_num_offd_elements) && (fabs(gm_row_sum) < 1e-15)) 
                     gm_row_sum = 1.0;
                  
               }
               
            } /* end of looking over elements in this row: 
                 if( p_num_diag_elements || p_num_offd_element)*/
             
            /* get diagonal of A */
            diagonal = A_diag_data[A_diag_i[i]];
            d_sign = 1;
            if (diagonal < 0) d_sign = -1;
            
            /* FIRST LOOP OVER DIAG ELEMENTS */
            /* loop over elements in row i of A (except diagonal)*/
            for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
            {
               jj_point = A_diag_j[jj]; /* fine index */
               
               /* only want like unknowns */
               if (fcn_num != dof_func[jj_point])
                  continue;
               
               dist_coarse = 0;
               a_ij = A_diag_data[jj];
               
               /* don't get rid of these 3/13 */
               /* if (a_ij*d_sign > 0)
                  continue;*/
                
               
               found = 0;
               if (CF_marker[jj_point] >= 0) /*coarse*/
               {
                  jj_point_c = fine_to_coarse[jj_point];
                  
                  new_col = col_map[jj_point_c];
                  
                  /* find P(i,j_c) and put value there (there may not be
                     an entry in P if this coarse connection was not a
                     strong connection */
                  
                  /* we are looping in the diag of this row, so we only
                   * need to look in P_diag */
                  
                  for (kk = P_diag_i_new[i]; kk < P_diag_i_new[i] + num_new_p_diag; kk ++)
                  {
                     if (P_diag_j_new[kk] == new_col)
                     {
                        P_diag_data_new[kk] += a_ij;
                        found = 1;
                        break;
                     }
                  }
                  if (!found)
                  {
                     /*this is a weakly connected c-point - does 
                       not contribute - so no error*/
                     /*( hypre_printf("Error find j_point_c\n");*/
                     /*hypre_printf("dist coarse in i = %d\n", i);*/
                     dist_coarse = 1;
                  }
                }
               else /*fine connection */ 
               {
                  use_alt_w = 0;
                  sum = 0.0;
                  num_f = 0;
                  /*loop over row of orig P for jj_point and get the sum of the
                    connections to c-points of i 
                    ( need to do diag and offd) */
                  
                  /* diag */
                  for (pp = P_diag_i[jj_point]; pp < P_diag_i[jj_point+1]; pp++)
                  {
                     
                     p_point = P_diag_j[pp];/* this is a coarse index */
                     /* is p_point in row i also ? */
                     for (kk = P_diag_i[i]; kk < P_diag_i[i+1]; kk ++)
                     {
                        kk_point = P_diag_j[kk]; /* this is a coarse index */
                        if (p_point == kk_point)
                        {
                           /* add p_jk to sum */
                           sum += P_diag_data[pp];
                           
                           break;
                        }
                     }/* end loop kk over row i */
                  } /* end diag (end loop pp over row jj_point) */
                  /* offd */
                  for (pp = P_offd_i[jj_point]; pp < P_offd_i[jj_point+1]; pp++)
                  {
                     p_point = P_offd_j[pp];/* this is a coarse index */
                     
                     /* is p_point in row i also ? check the offd part*/
                     for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk ++)
                     {
                        kk_point = P_offd_j[kk]; /* this is a coarse index */
                        if (p_point == kk_point)
                        {
                           /* add p_jk to sum */
                           sum += P_offd_data[pp];
                           
                           break;
                        }
                     }/* end loop kk over row i */
                  } /* end offd */
                  
                  if (fabs(sum) < 1e-12)
                  {
                     sum = 1.0;
                     use_alt_w = 1;
                  }
                  
                  if (use_alt_w)
                  {
                     /* distribute a_ij equally among coarse points */
                     aw =  a_ij/( p_num_diag_elements + p_num_offd_elements);
                     kk_count = 0;
                     /* loop through row i of orig p*/
                     /* diag first */
                     for (kk = P_diag_i[i]; kk < P_diag_i[i+1]; kk++)
                     {
                        kk_point = P_diag_j[kk]; /* this is a coarse index */
                        
                        if (add_q)
                        {
                           cur_spot =   P_diag_i_new[i] + kk_count*(num_smooth_vecs+1);
                        }
                        else
                           cur_spot =  P_diag_i_new[i] + kk_count;
                        
                        P_diag_data_new[cur_spot] += aw;
                        
                        /*add q? */
                        if (add_q)
                        {
                           for (k = 0; k < num_smooth_vecs; k++)
                           {
                              /* point to the smooth vector */
                              vector = smooth_vecs[k];
                              vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                              
                              /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                              fine_kk = coarse_to_fine[kk_point];
                              tmp_d1 = vec_data[jj_point] - adj[k];
                              tmp_d2 = vec_data[fine_kk];
                              q_val =  aw*(tmp_d1 - tmp_d2);
                              
                              P_diag_data_new[cur_spot + k + 1]+= q_val;
                           }
                           
                        }
                        kk_count++;
                     } /* did each element of p_diag */
                     /* now do offd */
                     kk_count = 0;
                     for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk++)
                     {
                        kk_point = P_offd_j[kk]; /* this is a coarse index */
                        if (add_q)
                        {
                           cur_spot =   P_offd_i_new[i] + kk_count*(num_smooth_vecs+1);
                        }
                        else
                        {
                           cur_spot =  P_offd_i_new[i] + kk_count;
                        }
                        P_offd_data_new[cur_spot] += aw;
                        /*add q? */
                        if (add_q)
                        {
                           for (k = 0; k < num_smooth_vecs; k++)
                           {
                              /* point to the smooth vector */
                              vector = smooth_vecs[k];
                              vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                              /* alias the offd smooth vector */
                              offd_vec_data_P = smooth_vec_offd_P + k*num_cols_P_offd;
                              
                              /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                              
                              /* jj point is a fine index  from Adiag
                                 .  but kk is a coarse index from P- needs the
                                 coarse offd data */
                              tmp_d1 = vec_data[jj_point] - adj[k];
                              tmp_d2 = offd_vec_data_P[kk_point];
                              
                              q_val =  aw*(tmp_d1 - tmp_d2);
                              P_offd_data_new[cur_spot + k + 1]+= q_val;
                           }
                        }
                        
                        kk_count++;
                     } /* end of offd */

                     continue;
                     /* to go to next jj of A */
   
                  }/* end of alt w */
                   
                    /* Now we need to do the distributing (THIS COULD BE CODED MORE
                       EFFICIENTLY (like classical interp )*/

                  /* loop through row i (diag and off d) of orig p*/
                  /* first the diag part */
                  kk_count = 0;
                  for (kk = P_diag_i[i]; kk < P_diag_i[i+1]; kk++)
                  {
                     
                     kk_point = P_diag_j[kk]; /* this is a coarse index */
                     /* now is there an entry for P(jj_point, kk_point)?  -
                        need to look through row j_point (on -proc since
                        j came from A_diag */
                     
                     found = 0;
                     for (pp = P_diag_i[jj_point]; pp < P_diag_i[jj_point+1]; pp++)
                     {
                        if (P_diag_j[pp] == kk_point)
                        {
                           found = 1;
                           /* a_ij*w_jk */
                           aw =  a_ij*P_diag_data[pp];
                           aw = aw/sum;
                           /* loc in new P */
                           if (add_q)
                           {
                              cur_spot =   P_diag_i_new[i] + kk_count*(num_smooth_vecs+1);
                           }
                           else
                              cur_spot =  P_diag_i_new[i] + kk_count;
                           /* P_diag_data_new[k] += aw; */
                           P_diag_data_new[cur_spot] += aw;
                           
                           /*add q? */
                           if (add_q)
                           {
                              for (k = 0; k < num_smooth_vecs; k++)
                              {
                                 /* point to the smooth vector */
                                 vector = smooth_vecs[k];
                                 vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                                 
                                 /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                                 fine_kk = coarse_to_fine[kk_point];
                                 tmp_d1 = vec_data[jj_point] - adj[k];
                                 tmp_d2 = vec_data[fine_kk];
                                 q_val =  aw*(tmp_d1 - tmp_d2);
                                                                   
                                 P_diag_data_new[cur_spot + k + 1]+= q_val;
                              }
                           }
                           break;
                        }
                     } /* end loop pp over row jj_point */
                         
                     /* if found = 0, do somthing with weight? */
                     kk_count++;
                  } /* end loop kk over row i of Pdiag */
                  /* now do the offd part */
                  kk_count = 0;
                  for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk++)
                  {
                     kk_point = P_offd_j[kk]; /* this is a coarse index */
                     found = 0;
                     for (pp = P_offd_i[jj_point]; pp < P_offd_i[jj_point+1]; pp++)
                     {
                        if (P_offd_j[pp] == kk_point)
                        {
                           found = 1;
                           /* a_ij*w_jk */
                           aw =  a_ij*P_offd_data[pp];
                           aw = aw/sum;
                           
                           /* loc in new P */
                           if (add_q)
                           {
                              cur_spot =   P_offd_i_new[i] + kk_count*(num_smooth_vecs+1);
                           }
                           else
                           {
                              cur_spot =  P_offd_i_new[i] + kk_count;
                           }
                           P_offd_data_new[cur_spot] += aw;
                           if (add_q)
                           {
                              for (k = 0; k < num_smooth_vecs; k++)
                              {
                                 /* point to the smooth vector */
                                 vector = smooth_vecs[k];
                                 vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));

                                 /* alias the offd smooth vector */
                                 offd_vec_data_P = smooth_vec_offd_P + k*num_cols_P_offd;
                                 
                                 /* jj_point is a fine index and kk_point is
                                    a coarse index that is offd */
                                 /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                                 tmp_d1 = vec_data[jj_point] - adj[k]; /* jj point is in diag */
                                 tmp_d2 = offd_vec_data_P[kk_point];
                                 q_val =  aw*(tmp_d1 - tmp_d2);
                                 
                                 P_offd_data_new[cur_spot + k + 1]+= q_val;
                              }
                           }/* end of add_q */
                           break;
                        }
                     }/* end of pp loop */
                     
                     kk_count++;
                  }/* end loop kk over offd part */
                  
               } /* end of if fine connection in row of A*/

               if (dist_coarse)
               {
                  /* coarse not in orig interp (weakly connected) */ 
                  /* distribute a_ij equally among coarse points */
                  aw =  a_ij/(p_num_diag_elements + p_num_offd_elements);
                  kk_count = 0;
                  /* loop through row i of orig p (diag and offd)*/
                  /* diag */
                  for (kk = P_diag_i[i]; kk < P_diag_i[i+1]; kk++)
                  {
                     kk_point = P_diag_j[kk]; /* this is a coarse index */
                     
                     if (add_q)
                     {
                        cur_spot =   P_diag_i_new[i] + kk_count*(num_smooth_vecs+1);
                     }
                     else
                     {
                        cur_spot =  P_diag_i_new[i] + kk_count;
                     }
                     P_diag_data_new[cur_spot] += aw;
                     
                     /*add q? */
                     if (add_q)
                     {
                        for (k = 0; k < num_smooth_vecs; k++)
                        {
                           /* point to the smooth vector */
                           vector = smooth_vecs[k];
                           vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                           
                           /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                           fine_kk = coarse_to_fine[kk_point];
                           tmp_d1 = vec_data[jj_point] - adj[k];
                           tmp_d2 = vec_data[fine_kk];
                           q_val =  aw*(tmp_d1 - tmp_d2);
                           
                            P_diag_data_new[cur_spot + k + 1]+= q_val;
                        }
                        
                     }
                     kk_count++;
                  } /* did each diag element of p */
                  /* now off diag */
                  kk_count = 0;
                  for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk++)
                  {
                     kk_point = P_offd_j[kk]; /* this is a coarse index */
                     if (add_q)
                     {
                        cur_spot =   P_offd_i_new[i] + kk_count*(num_smooth_vecs+1);
                     }
                     else
                     {
                        cur_spot =  P_offd_i_new[i] + kk_count;
                     }
                     P_offd_data_new[cur_spot] += aw;
                     /*add q? */
                     if (add_q)
                     {
                        for (k = 0; k < num_smooth_vecs; k++)
                        {
                            /* point to the smooth vector */
                           vector = smooth_vecs[k];
                           vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));

                           /* alias the offd smooth vector */
                           offd_vec_data_P = smooth_vec_offd_P + k*num_cols_P_offd;
                           
                           /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                           
                           /* jj point is a fine index of Adiag, but
                              kk is a coarse index - needs the coarse
                              offd data */
                           tmp_d1 = vec_data[jj_point] - adj[k];
                           tmp_d2 = offd_vec_data_P[kk_point];
                           
                           q_val =  aw*(tmp_d1 - tmp_d2);
                           P_offd_data_new[cur_spot + k + 1]+= q_val;
                        }
                     }
                     kk_count++;
                  }/* did each off diag element of p */
               }/* end of dist_coarse */
            }/* end loop jj over row i (diag part) of A */
          
         
             /* Still looping over ith row of A - NOW LOOP OVER OFFD! */
            
            for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
            {
               
               jj_point = A_offd_j[jj]; /* fine index */
               
               /* only want like unknowns */

               if (fcn_num != dof_func_offd[jj_point])
                  continue;
               
               dist_coarse = 0;
               a_ij = A_offd_data[jj];
               
               found = 0;
               if (CF_marker_offd[jj_point] >= 0) /*check the offd marker */
               {
                  /*coarse*/
                  jj_point_c = fine_to_coarse_offd[jj_point]; /* now its global!! */

                  /* CHECK THIS - changed on 11/24!! */

                   /* find P(i,j_c) and put value there (there may not be
                      an entry in P if this coarse connection was not a
                      strong connection */
                   
                   /* we are looping in the off diag of this row, so we only
                    * need to look in P_offd  - look in orig P*/
                  for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk ++)
                  {
                     index = P_offd_j[kk]; /* local number */

                     index = col_map_offd_P[index]; /*make a global number
                                                      becuz jj_point_c
                                                      is global  */

                     if (index == jj_point_c)
                     {
                        /* convert jj_point_c (global) to a new col that takes
                         * into account the new unknowns*/
                        if (num_smooth_vecs && (level == interp_vec_first_level))
                        {
                           new_col = jj_point_c + (jj_point_c/num_functions) * num_smooth_vecs;
                        }
                        else /* no adjustment */
                        {
                           new_col = jj_point_c;
                        }

                        /*  now figure out where to add in P_new */
                        for (pp = P_offd_i_new[i]; pp < P_offd_i_new[i] + num_new_p_offd; pp ++)
                        {
                           index =  P_offd_j_new[pp]; /* these are global - haven't done col map yet */
                           if (index == new_col)
                           {
                              P_offd_data_new[pp] += a_ij;
                              found = 1;
                              break; /* from pp loop */
                           }
                        }
                        break; /* from kk loop */
                     }
                  }
                  if (!found)
                  {
                     /*this is a weakly connected c-point - does 
                       not contribute - so no error - but this messes up row sum*/
                     /* we need to distribute this */
                     dist_coarse = 1;
                  }
               }/* end of coarse */
               else /*fine connection */ 
               {
                  use_alt_w = 0;
                  sum = 0.0;
                  num_f = 0;
                  
                  /*loop over row of P for j_point and get the sum of
                    the connections to c-points of i (diag and offd)
                    - now the row for jj_point is on another processor
                    - and jj_point is an index of Aoffd - need to convert
                    it to corresponding index of P */
                  
                  /* j_point is an index of A_off d - so */
                  /* now this is the row in P, but these are stored in P_ext according to offd of A */
                  j_ext_index = jj_point;

                  for (pp = P_ext_i[j_ext_index]; pp < P_ext_i[j_ext_index+1]; pp++)
                  {
                     p_point = P_ext_j[pp];/* this is a coarse index */
                     /* is p_point in row i also ?  check the diag of 
                        offd part*/
                     if (p_point > -1) /* in diag part */
                     {
                        for (kk = P_diag_i[i]; kk < P_diag_i[i+1]; kk++)
                        {
                           kk_point = P_diag_j[kk]; /* this is a coarse index */
                           if (p_point == kk_point)
                           {
                              /* add p_jk to sum */
                              sum += P_ext_data[pp];
                              
                              break;
                           }
                        }/* end loop kk over row i */
                     }
                     else /* in offd diag part */
                     {
                        p_point = -p_point-1;
                        /* p_point is a local col number for P now */
                        for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk ++)
                        {
                           kk_point = P_offd_j[kk]; /* this is a coarse index */
                           if (p_point == kk_point)
                           {
                              /* add p_jk to sum */
                              sum += P_ext_data[pp];
                              
                              break;
                           }
                        }/* end loop k over row i */
                     }/* end if diag or offd */ 
                     
                  }/* end loop over pp for j_ext_index */
                  if (fabs(sum) < 1e-12)
                  {
                     sum = 1.0;
                     use_alt_w = 1;
                  }
                  if (use_alt_w)
                  {
                     /* distribute a_ij equally among coarse points */
                     aw =  a_ij/( p_num_diag_elements + p_num_offd_elements);
                     kk_count = 0;
                     
                     /* loop through row i of orig p*/
                     /* diag first */
                     for (kk = P_diag_i[i]; kk < P_diag_i[i+1]; kk++)
                     {
                        kk_point = P_diag_j[kk]; /* this is a coarse index */
                        
                        if (add_q)
                        {
                           cur_spot =   P_diag_i_new[i] + kk_count*(num_smooth_vecs+1);
                        }
                        else
                           cur_spot =  P_diag_i_new[i] + kk_count;
                        
                        P_diag_data_new[cur_spot] += aw;
                        
                        /*add q? */
                        if (add_q)
                        {
                           for (k = 0; k < num_smooth_vecs; k++)
                           {
                              /* point to the smooth vector */
                              vector = smooth_vecs[k];
                              vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                              offd_vec_data = smooth_vec_offd + k*num_cols_A_offd;

                              /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                              fine_kk = coarse_to_fine[kk_point];  /** kk point is a diag index */
                              tmp_d1 = offd_vec_data[jj_point] - adj[k]; /* jj_point is an offd index */
                              tmp_d2 = vec_data[fine_kk];
                              q_val =  aw*(tmp_d1 - tmp_d2);
                              
                              P_diag_data_new[cur_spot + k + 1]+= q_val;
                           }
                           
                        }
                        kk_count++;
                     } /* did each element of p_diag */
                     /* now do offd */
                     kk_count = 0;
                     for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk++)
                     {
                        kk_point = P_offd_j[kk]; /* this is a coarse index */
                        if (add_q)
                        {
                           cur_spot =   P_offd_i_new[i] + kk_count*(num_smooth_vecs+1);
                        }
                        else
                        {
                           cur_spot =  P_offd_i_new[i] + kk_count;
                        }
                        P_offd_data_new[cur_spot] += aw;
                        /*add q? */
                        if (add_q)
                        {
                           for (k = 0; k < num_smooth_vecs; k++)
                           {
                              /* alias the offd smooth vector */
                              offd_vec_data = smooth_vec_offd + k*num_cols_A_offd;
                              offd_vec_data_P = smooth_vec_offd_P + k*num_cols_P_offd;
                              
                              /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                              
                              /* jj point is a fine index, so that can index into
                                 offd_vec_data.  but kk is a coarse index - needs the
                                 coarse offd data */
                              tmp_d1 = offd_vec_data[jj_point] - adj[k];
                              tmp_d2 = offd_vec_data_P[kk_point];
                              
                              q_val =  aw*(tmp_d1 - tmp_d2);
                              P_offd_data_new[cur_spot + k + 1]+= q_val;
                           }
                        }
                        
                        kk_count++;
                     } /* end of offd */
                      
                     
                     continue;
                     /* to go to next jj of A */
                  }/* end of alt w */
                    
                  /* Now we need to do the distributing */
                  /* loop through row i (diag and off d) of orig p*/
                  /* first the diag part */
                  kk_count = 0;
                  for (kk = P_diag_i[i]; kk < P_diag_i[i+1]; kk++)
                  {
                     kk_point = P_diag_j[kk]; /* this is a coarse index */
                     /* now is there an entry for P(jj_point, kk_point)?  -
                        need to look through row jj_point (now off-proc since
                        jj came from A_offd */
                     found = 0;
                     for (pp = P_ext_i[j_ext_index]; pp < P_ext_i[j_ext_index+1]; pp++)
                     {
                        p_point  = P_ext_j[pp];
                        if (p_point > -1) /* diag part */
                        {
                           if (p_point == kk_point)
                           {
                              found = 1;
                              /* a_ij*w_jk */
                              aw =  a_ij*P_ext_data[pp];
                              aw = aw/sum;
                              /* loc in new P */
                              if (add_q)
                              {
                                 cur_spot =   P_diag_i_new[i] + kk_count*(num_smooth_vecs+1);
                              }
                              else
                                 cur_spot =  P_diag_i_new[i] + kk_count;
                              /* P_diag_data_new[k] += aw; */
                              P_diag_data_new[cur_spot] += aw;
                           
                              /*add q? */
                              if (add_q)
                              {
                                 for (k = 0; k < num_smooth_vecs; k++)
                                 {
                                    /* point to the smooth vector */
                                    vector = smooth_vecs[k];
                                    vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                                    offd_vec_data = smooth_vec_offd + k*num_cols_A_offd;

                                    /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                                    fine_kk = coarse_to_fine[kk_point]; /** kk point is a diag index */
                                    tmp_d1 = offd_vec_data[jj_point] - adj[k];/* jj_point is an offd index */
                                    tmp_d2 = vec_data[fine_kk];
                                    q_val =  aw*(tmp_d1 - tmp_d2);
                                    
                                    P_diag_data_new[cur_spot + k + 1]+= q_val;
                                 }
                                 
                              }/* end addq */
                              break;
                           } /* end point found */
                        } /* in diag part */
                     } /* end loop pp over P_ext_i[jj_point]*/
                     kk_count++; 
                  } /* end loop kk over row i of Pdiag */
                    /* now do the offd part */
                  kk_count = 0;
                  for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk++)
                  {
                     kk_point = P_offd_j[kk]; /* this is a coarse index */
                     found = 0;
                     /* now is there an entry for P(jj_point, kk_point)?  -
                        need to look through row j_point (on offproc since
                        jj came from A_offd */
                     for (pp = P_ext_i[j_ext_index]; pp < P_ext_i[j_ext_index+1]; pp++)
                     {
                        p_point  = P_ext_j[pp];
                        if (p_point < 0) /* in offd part */
                        {
                           p_point = - p_point - 1; /* fix index */
                           if (p_point == kk_point)
                           {
                              found = 1;
                              /* a_ij*w_jk */
                              aw =  a_ij*P_ext_data[pp];
                              aw = aw/sum;
                              
                              /* loc in new P */
                              if (add_q)
                              {
                                 cur_spot =   P_offd_i_new[i] + kk_count*(num_smooth_vecs+1);
                              }
                              else
                              {
                                 cur_spot =  P_offd_i_new[i] + kk_count;
                              }
                              P_offd_data_new[cur_spot] += aw;
                              if (add_q)
                              {
                                 for (k = 0; k < num_smooth_vecs; k++)
                                 {
                                    
                                    /* alias the offd smooth vector */
                                    offd_vec_data = smooth_vec_offd + k*num_cols_A_offd;
                                    offd_vec_data_P = smooth_vec_offd_P + k*num_cols_P_offd;
                                    
                                    /* jj_point is a fine index and kk_point is
                                       a coarse index */
                                    /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                                    tmp_d1 = offd_vec_data[jj_point] - adj[k];
                                    tmp_d2 = offd_vec_data_P[kk_point];
                                    q_val =  aw*(tmp_d1 - tmp_d2);
                                    
                                    P_offd_data_new[cur_spot + k + 1]+= q_val;
                                 }
                              }/* end of add_q */
                              break;
                           } /* end of found */
                        } /* end of in offd */
                     }/* end of pp loop */
                     kk_count++;
                  }/* end loop kk over offd part */
                  
               }/* end of of if fine connection in offd row of A */
                
               if (dist_coarse)
               {
                  /* coarse not in orig interp (weakly connected) */ 
                  /* distribute a_ij equally among coarse points */
                  aw =  a_ij/(p_num_diag_elements + p_num_offd_elements);
                  kk_count = 0;
                  /* loop through row i of orig p (diag and offd)*/
                  /* diag */
                  for (kk = P_diag_i[i]; kk < P_diag_i[i+1]; kk++)
                  {
                     kk_point = P_diag_j[kk]; /* this is a coarse index */
                     
                     if (add_q)
                     {
                        cur_spot =   P_diag_i_new[i] + kk_count*(num_smooth_vecs+1);
                     }
                     else
                     {
                        cur_spot =  P_diag_i_new[i] + kk_count;
                     }
                     P_diag_data_new[cur_spot] += aw;

                      /*add q? */
                     if (add_q)
                     {
                        for (k = 0; k < num_smooth_vecs; k++)
                        {
                           /* point to the smooth vector */
                           vector = smooth_vecs[k];
                           vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                            offd_vec_data = smooth_vec_offd + k*num_cols_A_offd;

                           /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                           fine_kk = coarse_to_fine[kk_point];/** kk point is a diag index */
                           tmp_d1 = offd_vec_data[jj_point] - adj[k];/* jj_point is an offd index */
                           tmp_d2 = vec_data[fine_kk];
                           q_val =  aw*(tmp_d1 - tmp_d2);
                           
                           
                           P_diag_data_new[cur_spot + k + 1]+= q_val;
                        }
                        
                     }
                     kk_count++;
                  } /* did each diag element of p */
                   /* now off-diag */
                  kk_count = 0;
                  for (kk = P_offd_i[i]; kk < P_offd_i[i+1]; kk++)
                  {
                     kk_point = P_offd_j[kk]; /* this is a coarse index */
                     if (add_q)
                     {
                        cur_spot =   P_offd_i_new[i] + kk_count*(num_smooth_vecs+1);
                     }
                     else
                     {
                        cur_spot =  P_offd_i_new[i] + kk_count;
                     }
                     P_offd_data_new[cur_spot] += aw;
                     /*add q? */
                     if (add_q)
                     {
                        for (k = 0; k < num_smooth_vecs; k++)
                        {
                           /* alias the offd smooth vector */
                           offd_vec_data = smooth_vec_offd + k*num_cols_A_offd;
                           offd_vec_data_P = smooth_vec_offd_P + k*num_cols_P_offd;
                           
                           /* q_val = a_ij* w_jk*[s(j) - s(k)] */
                           
                           /* jj point is a fine index, so that can index into
                              offd_vec_data.  but kk is a coarse index - needs the
                              coarse offd data */
                           tmp_d1 = offd_vec_data[jj_point] - adj[k];
                           tmp_d2 = offd_vec_data_P[kk_point];
                           
                           q_val =  aw*(tmp_d1 - tmp_d2);
                           P_offd_data_new[cur_spot + k + 1]+= q_val;
                        }
                     }
                     kk_count++;
                  }/* did each off-diag element of p */
                  
               }/* end of dist_coarse */
               
            } /* end of jj loop over offd of A */

            /* now divide by the diagonal and we are finished with this row!*/
            if (fabs(diagonal) > 0.0)
            {
               for (kk = P_diag_i_new[i] ; kk <  P_diag_i_new[i] + num_new_p_diag; kk++)
               {
                  P_diag_data_new[kk] /= -(diagonal);
                  /* want new_row_sum only to be orig p elements (not q) */
                  new_col =  P_diag_j_new[kk];
                  if (level == interp_vec_first_level)
                     fcn_num = (HYPRE_Int) fmod(new_col, num_functions + num_smooth_vecs);
                  else
                     fcn_num = (HYPRE_Int) fmod(new_col, num_functions);
                  
                  if (fcn_num < orig_nf)
                     new_row_sum +=  P_diag_data_new[kk];
                   
               }
               for (kk = P_offd_i_new[i] ; kk <  P_offd_i_new[i] + num_new_p_offd; kk++)
               {
                  P_offd_data_new[kk] /= -(diagonal);
                  /* want new_row_sum only to be orig p elements (not q) */
                  new_col =  P_offd_j_new[kk];
                  if (level == interp_vec_first_level)
                     fcn_num = (HYPRE_Int) fmod(new_col, num_functions + num_smooth_vecs);
                  else
                     fcn_num = (HYPRE_Int) fmod(new_col, num_functions);
                  
                  if (fcn_num < orig_nf)
                     new_row_sum +=  P_offd_data_new[kk];
                  
               }
            }
            /* if we had no fc, then the Q entries are zero - let's do
             * the GM approach instead for this row*/
            if (no_fc && add_q && no_fc_use_gm)
            {
               HYPRE_Int c_col, num_f;
               double dt, value;
                
               /* DIAG */
               for (kk = P_diag_i_new[i] ; kk <  P_diag_i_new[i] + num_new_p_diag; kk++)
               {
                  new_col =  P_diag_j_new[kk];
                  if (level == interp_vec_first_level)
                     num_f = num_functions + num_smooth_vecs;
                  else
                     num_f = num_functions;
                  
                  fcn_num = (HYPRE_Int) fmod(new_col, num_f);
                  
                  if (fcn_num < orig_nf)
                  {
                     /* get the old col number back to index into vector */
                     if (level == interp_vec_first_level )
                        c_col = new_col - (HYPRE_Int) floor((double) new_col/ (double) num_f);
                     else
                        c_col = new_col;
                     
                     c_col = coarse_to_fine[c_col];
                     
                     for (k = 0; k < num_smooth_vecs; k++)
                     {
                        /* point to the smooth vector */
                        vector = smooth_vecs[k];
                        vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                        dt =  P_diag_data_new[kk];
                        dt = (vec_data[i]/gm_row_sum - vec_data[c_col]);
                        value  = P_diag_data_new[kk]*(vec_data[i]/gm_row_sum - vec_data[c_col]);
                        P_diag_data_new[kk + k + 1] = value;
                     }
                     
                  }
               }
               /* OFFD */
               for (kk = P_offd_i_new[i] ; kk <  P_offd_i_new[i] + num_new_p_offd; kk++)
               {
                  new_col =  P_offd_j_new[kk];
                  if (level == interp_vec_first_level)
                     num_f = num_functions + num_smooth_vecs;
                  else
                     num_f = num_functions;
                  fcn_num = (HYPRE_Int) fmod(new_col, num_f);
                  
                  if (fcn_num < orig_nf)
                  {
                     if (level == interp_vec_first_level )
                        /* get the old col number back to index into vector */
                        c_col = new_col - (HYPRE_Int) floor((double) new_col/ (double) num_f);
                     else
                        c_col = new_col;
                     
                     for (k = 0; k < num_smooth_vecs; k++)
                     {
                        vector = smooth_vecs[k];
                        vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
                        
                        /* alias the offd smooth vector */
                        offd_vec_data_P = smooth_vec_offd_P + k*num_cols_P_offd;
                        
                        dt =  P_offd_data_new[kk];
                        dt = (vec_data[i]/gm_row_sum - offd_vec_data_P[c_col]);
                        value  = P_offd_data_new[kk]*(vec_data[i]/gm_row_sum - offd_vec_data_P[c_col]);
                        P_offd_data_new[kk + k + 1] = value;
                        
                     }
                  }
               }
                
            } /* end no_Fc  - do GM interpolation*/

         } /* end of row of P is fine point - build interp */
      }
      else /* Modify new dofs */
      {
         /* coarse points - just copy */
         if (CF_marker[i] >= 0) /* row corres. to coarse point - just copy orig */
         {
            /* diag */
            for (j=0; j < p_num_diag_elements; j++)
            {
               P_diag_data_new[j_diag_pos] = P_diag_data[orig_diag_start+j];
               
               new_col = col_map[ P_diag_j[orig_diag_start+j]];
               P_diag_j_new[j_diag_pos] =new_col;;
               
               j_diag_pos++;
               p_count_diag++;
            }
            /* offd elements */
            p_count_offd = p_count_diag;
            for (j=0; j < p_num_offd_elements; j++)
            {
               P_offd_data_new[j_offd_pos] = P_offd_data[orig_offd_start+j];
                
               /* note that even though we are copying, j
                  needs to go back to regular numbering - will be
                  compressed later when col_map_offd is generated*/
               index = P_offd_j[orig_offd_start+j];
               
               /* convert to the global col number using col_map_offd */
               index = col_map_offd_P[index];
               
               /*now adjust for the new dofs - since we are offd, can't
                * use col_map[index]*/
               if (num_smooth_vecs && (level == interp_vec_first_level))
               {
                  new_col = index + (index/num_functions) * num_smooth_vecs;
               }
               else /* no adjustment */
               {
                  new_col = index;
               }
               
               P_offd_j_new[j_offd_pos] = new_col;;
               
               j_offd_pos++;
               p_count_offd++;
            }
         }
         else /* row is for fine point - modify exisiting
               *  interpolation corresponding to the new dof - *
               *  for 2D make it (P_u + P_v)/2....I'll use the original
               P values*/
         {
            HYPRE_Int m, m_pos;
            double m_val;
            double tmp;
            
            /* replace each element of P*/
            /* DIAG */
            for (j=0; j < p_num_diag_elements; j++)
            {
               m_val = 0.0;
               
               for (m = 0; m< orig_nf; m++)
               {
                  m_pos = P_diag_i[i - (fcn_num - m)]+ j; /* recall - nodal coarsening */
                  tmp = P_diag_data[m_pos];
                  m_val += theta[m]*P_diag_data[m_pos];
               }
               P_diag_j_new[j_diag_pos] = P_diag_j[orig_diag_start+j];
               P_diag_data_new[j_diag_pos] = m_val;
               j_diag_pos++;
               p_count_diag++;
            }
            /* OFF-DIAG */
            p_count_offd = p_count_diag;
            for (j=0; j < p_num_offd_elements; j++)
            {
               m_val = 0.0;
               for (m = 0; m< orig_nf; m++)
               {
                  m_pos = P_offd_i[i - (fcn_num - m)]+ j; /* recall - nodal coarsening */
                  tmp = P_offd_data[m_pos];
                  m_val += theta[m]*P_offd_data[m_pos];
               }
               index = P_offd_j[orig_offd_start+j];
               /* convert to the global col number using col_map_offd */
               index = col_map_offd_P[index];
               
               P_offd_j_new[j_offd_pos] = index;
               P_offd_data_new[j_offd_pos++] = m_val;
               p_count_offd++;
            }
         } /* end fine */

      }/*end of modify */
      
       /* update i */
      P_diag_i_new[i+1] = P_diag_i_new[i] + num_new_p_diag; 
      P_offd_i_new[i+1] = P_offd_i_new[i] + num_new_p_offd; 
      

      /* adjust p_count_offd to not include diag*/
      p_count_offd = p_count_offd - p_count_diag;
      

      if (p_count_diag != num_new_p_diag)
         hypre_printf("Error diag p_count in hypre_BoomerAMG_LNExpandInterp! myid = %d, row = %d\n", myid, i);
   
      if (p_count_offd != num_new_p_offd)
         hypre_printf("Error offd p_count in hypre_BoomerAMG_LNExpandInterp! myid = %d, row = %d\n", myid, i);
   

       /* NOW TRUNCATE Q ?*/
       if ( add_q && q_count > 0  && (q_max > 0 || abs_trunc > 0.0))
       {
          double value, lost_value, q_dist_value;
          HYPRE_Int q_count_k, num_lost, p_count_tot;
          HYPRE_Int lost_counter_diag, lost_counter_offd, j_counter;
          HYPRE_Int new_num_q, new_j_counter, new_diag_pos, new_offd_pos;
          HYPRE_Int i_qmax, lost_counter_q;
          /* loop through the smooth vectors - we have to do the q
             with each smooth vec separately
             TO DO: re-write to not have this outter loop (like the GM
             interpolation.)  I am not doing this now as we may change
             the LN truncation strategy entirely :)
          */
          for (k = 0; k < num_smooth_vecs; k++)
          {
             q_count_k = 0;
             lost_value = 0.0;
             num_lost = 0;
             i_qmax = 0;
             
             /* first do absolute truncation */
             if (abs_trunc > 0.0)
             {
                /* find out if any will be dropped */
                j_counter = 0;
                /* diag loop */
                for(j =  P_diag_i_new[i]; j <  P_diag_i_new[i] + p_count_diag; j++)
                {
                   if (is_q[j_counter] == (k+1))
                   {
                      q_count_k++;
                      value = fabs(P_diag_data_new[j]);
                      if (value < abs_trunc) 
                      {
                         num_lost ++;
                         lost_value += P_diag_data_new[j];
                      }
                   }
                   j_counter++;
                }
                /* offd loop  - don't reset j_counter*/
                for(j =  P_offd_i_new[i]; j <  P_offd_i_new[i] + p_count_offd; j++)
                {
                   if (is_q[j_counter] == (k+1))
                   {
                      q_count_k++;
                      value = fabs(P_offd_data_new[j]);
                      if (value < abs_trunc) 
                      {
                         num_lost ++;
                         lost_value += P_offd_data_new[j];
                      }
                   }
                   j_counter++;
                }
                /* now drop and adjust values of other entries in Q */
                if (num_lost)
                {
                   if ((q_count_k - num_lost) > 0)
                   {
                      q_dist_value = lost_value/(q_count_k - num_lost);
                   }
                   else
                   {
                      /* originall had this, but this makes it
                       * imposssible to get low complexities */
                      /* i_qmax = 1;
                         num_lost = 0;
                         hypre_printf("Warning: dropping all of Q; level = %d, i = %d, num = %d\n", level, i, num_lost);*/
                   }
                }
                if (num_lost)
                {

                   new_j_counter = 0;
                   lost_counter_diag = 0;
                   q_dist_value = 0.0;

                   /* diag */
                   new_diag_pos =  P_diag_i_new[i];
                   j_counter = 0;
                   for(j =  P_diag_i_new[i]; j < P_diag_i_new[i] + p_count_diag  ; j++)
                   {
                      
                      value = fabs(P_diag_data_new[j]);
                      
                      if ( is_q[j_counter] == (k+1) && (value < abs_trunc) )
                      {
                         /* drop */
                         lost_counter_diag++;
                      }
                      else /* keep */
                      {
                         /* for k, keep this q and add the q_dist (also copy the
                          * orig. p and other q not corres to this
                          * k) */
                         value =  P_diag_data_new[j];
                         if (is_q[j_counter] == (k+1))
                         {
                            value += q_dist_value;
                         }
                         P_diag_data_new[new_diag_pos] = value;
                         P_diag_j_new[new_diag_pos] = P_diag_j_new[j];
                         new_diag_pos++;
                         
                         is_q[new_j_counter] = is_q[j_counter];
                         new_j_counter++;
                         
                      }
                      j_counter++;
                   } /* end loop though j */
             
                   p_count_diag -= lost_counter_diag;
                   j_diag_pos -= lost_counter_diag;

                   /* offd */
                   lost_counter_offd = 0;
                   new_offd_pos =  P_offd_i_new[i];
                   for(j =  P_offd_i_new[i]; j < P_offd_i_new[i] + p_count_offd  ; j++)
                   {
                      value = fabs(P_offd_data_new[j]);
                      
                      if ( is_q[j_counter] == (k+1) && (value < abs_trunc) )
                      {
                         /* drop */
                         lost_counter_offd++;
                      }
                      else /* keep */
                      {
                         /* for k, keep this q and add the q_dist (also copy the
                          * orig. p and other q not corres to this
                          * k) */
                         value =  P_offd_data_new[j];
                         if (is_q[j_counter] == (k+1))
                         {
                            value += q_dist_value;
                         }
                         P_offd_data_new[new_offd_pos] = value;
                         P_offd_j_new[new_offd_pos] = P_offd_j_new[j];
                         new_offd_pos++;
                         
                         is_q[new_j_counter] = is_q[j_counter];
                         new_j_counter++;
                         
                      }
                      j_counter++;
                   } /* end loop though j */

                   p_count_offd -= lost_counter_offd;
                   j_offd_pos -= lost_counter_offd;

                } /* end if num_lost */
             }
             
             /* now max num elements truncation */
             if (i_qmax)
                loop_q_max = 1; /* not used currently */
             else
                loop_q_max = q_max;
             
             if (loop_q_max > 0)
             {
                /* copy all elements for the row and count the q's for
                 * this smoothvec*/
                q_count_k = 0; 
                j_counter = 0;
                for (j = P_diag_i_new[i]; j < P_diag_i_new[i]+ p_count_diag; j++)
                {
                   if (is_q[j_counter] == (k+1))
                      q_count_k++;

                   aux_j[j_counter] = P_diag_j_new[j];
                   aux_data[j_counter] = P_diag_data_new[j];
                   is_diag[j_counter] = 1;
                   j_counter++;
                     
                }
               

                /* offd loop  - don't reset j_counter*/
                for (j = P_offd_i_new[i]; j < P_offd_i_new[i]+ p_count_offd; j++)
                {
                   if (is_q[j_counter] == (k+1))
                      q_count_k++;

                   aux_j[j_counter] = P_offd_j_new[j];
                   aux_data[j_counter] = P_offd_data_new[j];
                   is_diag[j_counter] = 0;
                   j_counter++;
                     
                }
                
                new_num_q = q_count_k;
                num_lost = q_count_k - loop_q_max;

                if (num_lost > 0)
                {

                   p_count_tot = p_count_diag + p_count_offd;

                   /* only keep loop_q_max elements - get rid of smallest */
                   hypre_qsort4_abs(aux_data, aux_j, is_q, is_diag, 0 , p_count_tot -1);
                   
                   lost_value = 0.0;
                   lost_counter_q = 0;
                   lost_counter_diag = 0;
                   lost_counter_offd = 0;

                   j_counter = 0;
                     
                   new_diag_pos =  P_diag_i_new[i];
                   new_offd_pos =  P_offd_i_new[i];

                   new_j_counter = 0;

                   /* have to do diag and offd together because of sorting*/
                   for(j =  0; j < p_count_tot; j++)
                   {
                      
                      if ((is_q[j_counter] == (k+1)) && (lost_counter_q < num_lost))
                      {
                   
                         /*drop*/
                         lost_value += aux_data[j_counter];
                         lost_counter_q++;
                         
                         /* check whether this is diag or offd element */
                         if (is_diag[j])
                         {
                            lost_counter_diag++;
                         }
                         else
                         {
                            lost_counter_offd++;
                         }
                         new_num_q--;
                           
                         /* technically only need to do this the last time */
                         q_dist_value = lost_value/loop_q_max;
                      }
                      else
                      {
                         /* keep and add to the q values (copy q)*/
                         value =  aux_data[j_counter];
                         if ((is_q[j_counter] == (k+1)))
                            value += q_dist_value; 

                         if (is_diag[j])
                         {
                            P_diag_data_new[new_diag_pos] = value;
                            P_diag_j_new[new_diag_pos] = aux_j[j_counter];
                            new_diag_pos++;
                         
                            is_q[new_j_counter] = is_q[j_counter];
                            new_j_counter++;
                         }
                         else
                         {
                            P_offd_data_new[new_offd_pos] = value;
                            P_offd_j_new[new_offd_pos] = aux_j[j];
                            new_offd_pos++;
                            is_q[new_j_counter] = is_q[j];
                            new_j_counter++;
                         }
                      }
                      j_counter++;
                         
                   }/* end element loop */
                   /* adjust p_count and j_pos */
                   p_count_diag -= lost_counter_diag;
                   p_count_offd -= lost_counter_offd;

                   j_diag_pos -= lost_counter_diag;
                   j_offd_pos -= lost_counter_offd;

                   
                } /* end num lost > 0 */

             } /* end loop_q_max > 0  - element truncation */


          }/* end of loop through smoothvecs */
          
          P_diag_i_new[i+1] = P_diag_i_new[i] + p_count_diag;
          P_offd_i_new[i+1] = P_offd_i_new[i] + p_count_offd;

       }/* end of truncation*/

       if (j_diag_pos != P_diag_i_new[i+1])
       {
          hypre_printf("Warning - diag Row Problem in hypre_BoomerAMG_LNExpandInterp! myid = %d, row = %d\n", myid, i);
       }
       if (j_offd_pos != P_offd_i_new[i+1])
       {
          hypre_printf("Warning - off-diag Row Problem in hypre_BoomerAMG_LNExpandInterp! myid = %d, row = %d\n", myid, i);
       }
       
   } 
   /* end of MAIN LOOP i loop through rows of P*/
   /* ***********************************************************/
   
   /* Done looping through rows of P - NOW FINISH THINGS UP! */

   /* if level = first_level , we need to update the number of
          funcs and the dof_func */  
   if (level == interp_vec_first_level)
   {
      HYPRE_Int new_nf;
      
      c_dof_func = hypre_TReAlloc(c_dof_func, HYPRE_Int, new_ncv);
      cur_spot = 0;
      for (i = 0; i < ncv_peru; i++)
      {
         for (k = 0; k< num_functions + num_smooth_vecs; k++)
         {
            c_dof_func[cur_spot++] = k;
         }
      }
      /* return these values */
      new_nf =  num_functions + num_smooth_vecs;
      *nf = new_nf;
      *coarse_dof_func = c_dof_func;
      
      
      /* also we need to update the col starts and global num columns*/
      
      /* assumes that unknowns are together on a procsessor with
       * nodal coarsening  */
#ifdef HYPRE_NO_GLOBAL_PARTITION
      new_col_starts =  hypre_CTAlloc(HYPRE_Int,2);
      new_col_starts[0] = (col_starts[0]/num_functions)*new_nf ;
      new_col_starts[1] = (col_starts[1]/num_functions)*new_nf;
      
      if (myid == (num_procs -1)) g_nc = new_col_starts[1];
      hypre_MPI_Bcast(&g_nc, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
      new_col_starts =  hypre_CTAlloc(HYPRE_Int,num_procs+1);
      for (i = 0; i < (num_procs+1); i++)
      {
         new_col_starts[i] = (col_starts[i]/num_functions)*new_nf ;  
      }
      g_nc = new_col_starts[num_procs];
      
#endif
   }
   else /* not first level */
   {
      /* grab global num cols */
      g_nc = hypre_ParCSRMatrixGlobalNumCols(*P);
      
      /* copy col starts */
#ifdef HYPRE_NO_GLOBAL_PARTITION
      new_col_starts =  hypre_CTAlloc(HYPRE_Int,2);
      new_col_starts[0] = col_starts[0];
      new_col_starts[1] = col_starts[1];
#else
      new_col_starts =  hypre_CTAlloc(HYPRE_Int,num_procs+1);
      for (i = 0; i< (num_procs+1); i++)
      {
         new_col_starts[i] = col_starts[i];  
      }
#endif
    }
   
   /* modify P - now P has more entries and possibly more cols */
   new_P = hypre_ParCSRMatrixCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                    g_nc,
                                    hypre_ParCSRMatrixColStarts(A),
                                    new_col_starts,
                                    0,
                                    P_diag_i_new[num_rows_P],
                                    P_offd_i_new[num_rows_P]);
   
   
   P_diag = hypre_ParCSRMatrixDiag(new_P);
   hypre_CSRMatrixI(P_diag) = P_diag_i_new;
   hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
   hypre_CSRMatrixData(P_diag) = P_diag_data_new;
   hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_i_new[num_rows_P];
   
   P_offd = hypre_ParCSRMatrixOffd(new_P);
   hypre_CSRMatrixData(P_offd) = P_offd_data_new;
   hypre_CSRMatrixI(P_offd) = P_offd_i_new;
   hypre_CSRMatrixJ(P_offd) = P_offd_j_new;

   hypre_ParCSRMatrixOwnsRowStarts(new_P) = 0; 
   hypre_ParCSRMatrixOwnsColStarts(new_P) = 0;  /* we allocate new_col_starts*/
   
   /* If parallel we need to do the col map offd! */
   if (num_procs > 1)
   {
      HYPRE_Int count;
      HYPRE_Int num_cols_P_offd = 0;
      HYPRE_Int P_offd_new_size = P_offd_i_new[num_rows_P];
      
      if (P_offd_new_size)
      {

         HYPRE_Int *j_copy;
         
         /* check this */
         new_col_map_offd_P = hypre_CTAlloc(HYPRE_Int, P_offd_new_size);
         
         /*first copy the j entries (these are GLOBAL numbers) */
         j_copy = hypre_CTAlloc(HYPRE_Int, P_offd_new_size);
         for (i=0; i < P_offd_new_size; i++)
            j_copy[i] = P_offd_j_new[i];
         
         /* now sort them */
         qsort0(j_copy, 0, P_offd_new_size-1);

         /* now copy to col_map offd - but only each col once */
         new_col_map_offd_P[0] = j_copy[0];
         count = 0;
         for (i=0; i < P_offd_new_size; i++)
         {
            if (j_copy[i] > new_col_map_offd_P[count])
            {
               count++;
               new_col_map_offd_P[count] = j_copy[i];
            }
         }
         num_cols_P_offd = count + 1;
         
         /* reset the j entries to be local */
         for (i=0; i < P_offd_new_size; i++)
            P_offd_j_new[i] = hypre_BinarySearch(new_col_map_offd_P,
                                                 P_offd_j_new[i],
                                                 num_cols_P_offd);
         hypre_TFree(j_copy); 
      }

      hypre_ParCSRMatrixColMapOffd(new_P) = new_col_map_offd_P;
      hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
      
   } /* end col map stuff */
   

   /* comm pkg */
   hypre_MatvecCommPkgCreate ( new_P );
   
   /*destroy old */
   hypre_ParCSRMatrixDestroy(*P);
   
   /* RETURN: update P */
   *P = new_P;
   
    
#if SV_DEBUG
   {
      char new_file[80];
      hypre_CSRMatrix *P_CSR;
      
      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(new_P);
      
      if (!myid)
      {
         hypre_sprintf(new_file,"%s.level.%d","P_new_new", level );
         if (P_CSR)
            hypre_CSRMatrixPrint(P_CSR, new_file); 
      }
      
      hypre_CSRMatrixDestroy(P_CSR);
   }
#endif

    /* clean */
   hypre_TFree(coarse_to_fine);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(smooth_vec_offd);
   hypre_TFree(CF_marker_offd);
   hypre_TFree(dof_func_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(col_map);

   hypre_TFree(smooth_vec_offd);
   hypre_TFree(smooth_vec_offd_P);
   
   if (num_procs > 1) hypre_CSRMatrixDestroy(P_ext);

   return hypre_error_flag;
}



