/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/
#include "headers.h"
#include "aux_interp.h"

#define MAX_C_CONNECTIONS 100
#define HAVE_COMMON_C 1

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildStdInterp
 *  Comment: The interpolatory weighting can be changed with the sep_weight
 *           variable. This can enable not separating negative and positive
 *           off diagonals in the weight formula.
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBuildStdInterp(hypre_ParCSRMatrix *A, int *CF_marker,
			      hypre_ParCSRMatrix   *S, int *num_cpts_global,
			      int num_functions, int *dof_func, int debug_flag,
			      double trunc_factor, int max_elmts, 
			      int sep_weight, int *col_offd_S_to_A, 
			      hypre_ParCSRMatrix  **P_ptr)
{
  /* Communication Variables */
  MPI_Comm 	           comm = hypre_ParCSRMatrixComm(A);   
  hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommPkg     *new_comm_pkg;

  int              my_id, num_procs;

  /* Variables to store input variables */
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A); 
  double          *A_diag_data = hypre_CSRMatrixData(A_diag);
  int             *A_diag_i = hypre_CSRMatrixI(A_diag);
  int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
  double          *A_offd_data = hypre_CSRMatrixData(A_offd);
  int             *A_offd_i = hypre_CSRMatrixI(A_offd);
  int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

  int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
  int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  int              n_fine = hypre_CSRMatrixNumRows(A_diag);
  int              col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
  int              local_numrows = hypre_CSRMatrixNumRows(A_diag);
  int              col_n = col_1 + local_numrows;
  int              total_global_cpts, my_first_cpt;

  /* Variables to store strong connection matrix info */
  hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
  int             *S_diag_i = hypre_CSRMatrixI(S_diag);
  int             *S_diag_j = hypre_CSRMatrixJ(S_diag);
   
  hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
  int             *S_offd_i = hypre_CSRMatrixI(S_offd);
  int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

  /* Interpolation matrix P */
  hypre_ParCSRMatrix *P;
  hypre_CSRMatrix    *P_diag;
  hypre_CSRMatrix    *P_offd;   

  double          *P_diag_data;
  int             *P_diag_i, *P_diag_j;
  double          *P_offd_data;
  int             *P_offd_i, *P_offd_j;

  int		  *col_map_offd_P;
  int              P_diag_size; 
  int              P_offd_size;
  int             *P_marker; 
  int             *P_marker_offd;
  int             *P_node_add;  
  int             *CF_marker_offd, *tmp_CF_marker_offd;
  int             *dof_func_offd = NULL;

  /* Full row information for columns of A that are off diag*/
  hypre_CSRMatrix *A_ext;   
  double          *A_ext_data;
  int             *A_ext_i;
  int             *A_ext_j;
  
  int             *fine_to_coarse;
  int             *fine_to_coarse_offd;
  int             *found;

  int              num_cols_P_offd;
  int              newoff, loc_col;
  int              A_ext_rows, full_off_procNodes;
  
  hypre_CSRMatrix *Sop;
  int             *Sop_i;
  int             *Sop_j;
  
  int              Soprows;
  
  /* Variables to keep count of interpolatory points */
  int              jj_counter, jj_counter_offd;
  int              jj_begin_row, jj_end_row;
  int              jj_begin_row_offd, jj_end_row_offd;
  int              coarse_counter, coarse_counter_offd;
    
  /* Interpolation weight variables */
  double          *ahat, *ahat_offd; 
  double           sum_pos, sum_pos_C, sum_neg, sum_neg_C, sum, sum_C;
  double           diagonal, distribute;
  double           alfa, beta;
  int              kmin_offd, kmax_offd, kmin, kmax;
  
  /* Loop variables */
  int              index;
  int              start_indexing = 0;
  int              i, i1, j, j1, jj, kk, k1;

  /* Definitions */
  double           zero = 0.0;
  double           one  = 1.0;
  
  /* Expanded communication for neighbor of neighbors */
  int new_num_recvs, *new_recv_procs, *new_recv_vec_starts;
  int new_num_sends, *new_send_procs, *new_send_map_starts;
  int *new_send_map_elmts;
  
  /* BEGIN */
  MPI_Comm_size(comm, &num_procs);   
  MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   MPI_Bcast(&total_global_cpts, 1, MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(A);
#else
     hypre_MatvecCommPkgCreate(A);
#endif
     comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }
   
   /* Set up off processor information (specifically for neighbors of 
    * neighbors */
   newoff = 0;
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
     /*----------------------------------------------------------------------
      * Get the off processors rows for A and S, associated with columns in 
      * A_offd and S_offd.
      *---------------------------------------------------------------------*/
     A_ext         = hypre_ParCSRMatrixExtractBExt(A,A,1);
     A_ext_i       = hypre_CSRMatrixI(A_ext);
     A_ext_j       = hypre_CSRMatrixJ(A_ext);
     A_ext_data    = hypre_CSRMatrixData(A_ext);
     A_ext_rows    = hypre_CSRMatrixNumRows(A_ext);
     
     Sop           = hypre_ParCSRMatrixExtractBExt(S,S,0);
     Sop_i         = hypre_CSRMatrixI(Sop);
     Sop_j         = hypre_CSRMatrixJ(Sop);
     Soprows       = hypre_CSRMatrixNumRows(Sop);

     /* Find nodes that are neighbors of neighbors, not found in offd */
     newoff = new_offd_nodes(&found, A_ext_rows, A_ext_i, A_ext_j, 
			     num_cols_A_offd, col_map_offd, col_1, col_n, 
			     Sop_i, Sop_j);
     if(newoff >= 0)
       full_off_procNodes = newoff + num_cols_A_offd;
     else
       return(1);

     /* Possibly add new points and new processors to the comm_pkg, all
      * processors need new_comm_pkg */
     new_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);
     hypre_ParCSRCommExtendA(A, newoff, found, &new_num_recvs, 
			     &new_recv_procs, &new_recv_vec_starts, 
			     &new_num_sends, &new_send_procs, 
			     &new_send_map_starts, &new_send_map_elmts, 
			     &P_node_add);
     
     hypre_ParCSRCommPkgComm(new_comm_pkg) = comm;
     hypre_ParCSRCommPkgNumRecvs(new_comm_pkg) = new_num_recvs;
     hypre_ParCSRCommPkgRecvProcs(new_comm_pkg) = new_recv_procs;
     hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg) = new_recv_vec_starts;
     hypre_ParCSRCommPkgNumSends(new_comm_pkg) = new_num_sends;
     hypre_ParCSRCommPkgSendProcs(new_comm_pkg) = new_send_procs;
     hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg) = new_send_map_starts;
     hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg) = new_send_map_elmts;
   
     /* Insert nodes that are added from neighbors of neighbors connections */
     CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
     if (num_functions > 1 && full_off_procNodes > 0)
       dof_func_offd = hypre_CTAlloc(int, full_off_procNodes);
    
     insert_new_nodes(new_comm_pkg, CF_marker, P_node_add, num_cols_A_offd, 
		      full_off_procNodes, num_procs, CF_marker_offd);
     if(num_functions > 1)
       insert_new_nodes(new_comm_pkg, dof_func, P_node_add, num_cols_A_offd, 
	 full_off_procNodes, num_procs, dof_func_offd);
   }
   else
       new_comm_pkg = comm_pkg;

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_i    = hypre_CTAlloc(int, n_fine+1);

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
   fine_to_coarse_offd = hypre_CTAlloc(int, full_off_procNodes);

   P_marker = hypre_CTAlloc(int, n_fine);
   P_marker_offd = hypre_CTAlloc(int, full_off_procNodes);

   tmp_CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);

   initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse, 
		  fine_to_coarse_offd, P_marker, P_marker_offd,
		  tmp_CF_marker_offd);

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;
   coarse_counter_offd = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
     P_diag_i[i] = jj_counter;
     if (num_procs > 1)
       P_offd_i[i] = jj_counter_offd;
     
     if (CF_marker[i] >= 0)
     {
       jj_counter++;
       fine_to_coarse[i] = coarse_counter;
       coarse_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, interpolation is from the C-points that
      *  strongly influence i, or C-points that stronly influence F-points
      *  that strongly influence i.
      *--------------------------------------------------------------------*/
     else
     {
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];           
	 if (CF_marker[i1] >= 0)
	 { /* i1 is a C point */
	   if (P_marker[i1] < P_diag_i[i])
	   {
	     P_marker[i1] = jj_counter;
	     jj_counter++;
	   }
	 }
	 else
	 { /* i1 is a F point, loop through it's strong neighbors */
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] >= 0)
	     {
	       if(P_marker[k1] < P_diag_i[i])
	       {
		 P_marker[k1] = jj_counter;
		 jj_counter++;
	       }
	     } 
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if (CF_marker_offd[k1] >= 0)
	       {
		 if(P_marker_offd[k1] < P_offd_i[i])
		 {
		   tmp_CF_marker_offd[k1] = 1;
		   P_marker_offd[k1] = jj_counter_offd;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       /* Look at off diag strong connections of i */ 
       if (num_procs > 1)
       {
	 for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[S_offd_j[jj]];
	   else
	     i1 = S_offd_j[jj];           
	   if (CF_marker_offd[i1] >= 0)
	   {
	     if(P_marker_offd[i1] < P_offd_i[i])
	     {
	       tmp_CF_marker_offd[i1] = 1;
	       P_marker_offd[i1] = jj_counter_offd;
	       jj_counter_offd++;
	     }
	   }
	   else
	   { /* F point; look at neighbors of i1. Sop contains global col
	      * numbers and entries that could be in S_diag or S_offd or
	      * neither. */
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       k1 = Sop_j[kk];
	       if(k1 >= col_1 && k1 < col_n)
		 { /* In S_diag */
		 loc_col = k1-col_1;
		 if(CF_marker[loc_col] >= 0)
		 {
		   if(P_marker[loc_col] < P_diag_i[i])
		   {
		     P_marker[loc_col] = jj_counter;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       {
		 loc_col = -k1 - 1;
		 if(CF_marker_offd[loc_col] >= 0)
		 {
		   if(P_marker_offd[loc_col] < P_offd_i[i])
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     tmp_CF_marker_offd[loc_col] = 1;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       } 
     }
   }
   
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;
   
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size);

   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size);

   P_diag_i[n_fine] = jj_counter; 
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /* Fine to coarse mapping */
   if(num_procs > 1)
   {
     for (i = 0; i < n_fine; i++)
       fine_to_coarse[i] += my_first_cpt;
     insert_new_nodes(new_comm_pkg, fine_to_coarse, P_node_add, 
		      num_cols_A_offd, full_off_procNodes, num_procs, 
		      fine_to_coarse_offd);
     for (i = 0; i < n_fine; i++)
       fine_to_coarse[i] -= my_first_cpt;
   }

   /* Initialize ahat, which is a modification to a, used in the standard
    * interpolation routine. */
   ahat = hypre_CTAlloc(double, n_fine);
   ahat_offd = hypre_CTAlloc(double, full_off_procNodes);

   for (i = 0; i < n_fine; i++)
   {      
     P_marker[i] = -1;
     ahat[i] = 0;
   }
   for (i = 0; i < full_off_procNodes; i++)
   {      
     P_marker_offd[i] = -1;
     ahat_offd[i] = 0;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
     jj_begin_row = jj_counter;        
     if(num_procs > 1)
       jj_begin_row_offd = jj_counter_offd;

     /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
     
     if (CF_marker[i] >= 0)
     {
       P_diag_j[jj_counter]    = fine_to_coarse[i];
       P_diag_data[jj_counter] = one;
       jj_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
     
     else
     {         
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];   
	 
	 /*--------------------------------------------------------------
	  * If neighbor i1 is a C-point, set column number in P_diag_j
	  * and initialize interpolation weight to zero.
	  *--------------------------------------------------------------*/
	 
	 if (CF_marker[i1] >= 0)
	 {
	   if (P_marker[i1] < jj_begin_row)
	   {
	     P_marker[i1] = jj_counter;
	     P_diag_j[jj_counter]    = i1;
	     P_diag_data[jj_counter] = zero;
	     jj_counter++;
	   }
	 }
	 else 
	 {
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] >= 0)
	     {
	       if(P_marker[k1] < jj_begin_row)
	       {
		 P_marker[k1] = jj_counter;
		 P_diag_j[jj_counter] = k1;
		 P_diag_data[jj_counter] = zero;
		 jj_counter++;
	       }
	     }
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if(CF_marker_offd[k1] >= 0)
	       {
		 if(P_marker_offd[k1] < jj_begin_row_offd)
		 {
		   P_marker_offd[k1] = jj_counter_offd;
		   P_offd_j[jj_counter_offd] = k1;
		   P_offd_data[jj_counter_offd] = zero;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       
       if ( num_procs > 1)
       {
	 for (jj=S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[S_offd_j[jj]];
	   else
	     i1 = S_offd_j[jj];
	   if ( CF_marker_offd[i1] >= 0)
	   {
	     if(P_marker_offd[i1] < jj_begin_row_offd)
	     {
	       P_marker_offd[i1] = jj_counter_offd;
	       P_offd_j[jj_counter_offd]=i1;
	       P_offd_data[jj_counter_offd] = zero;
	       jj_counter_offd++;
	     }
	   }
	   else
	   {
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       k1 = Sop_j[kk];
	       if(k1 >= col_1 && k1 < col_n)
	       {
		 loc_col = k1-col_1;
		 if(CF_marker[loc_col] >= 0)
		 {
		   if(P_marker[loc_col] < jj_begin_row)
		   {		
		     P_marker[loc_col] = jj_counter;
		     P_diag_j[jj_counter] = loc_col;
		     P_diag_data[jj_counter] = zero;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       {
		 loc_col = -k1 - 1;
		 if(CF_marker_offd[loc_col] >= 0)
		 {
		   if(P_marker_offd[loc_col] < jj_begin_row_offd)
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     P_offd_j[jj_counter_offd]=loc_col;
		     P_offd_data[jj_counter_offd] = zero;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       }
       
       jj_end_row = jj_counter;
       jj_end_row_offd = jj_counter_offd;
       
       ahat[i] = A_diag_data[A_diag_i[i]];
       kmin = i; 
       kmax = i;
       kmin_offd = -1;
       kmax_offd = -1;
       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
       { /* i1 is direct neighbor */
	 i1 = A_diag_j[jj];
	 if (P_marker[i1] >= jj_begin_row)
	 {
	   ahat[i1] += A_diag_data[jj];
	   if (i1 < kmin) kmin = i1;
	   if (i1 > kmax) kmax = i1;
	 }
	 else
	 {
	   if(num_functions == 1 || dof_func[i] == dof_func[i1])
	   {
	     distribute = A_diag_data[jj]/A_diag_data[A_diag_i[i1]];
	     for (kk = A_diag_i[i1]+1; kk < A_diag_i[i1+1]; kk++)
	     {
	       k1 = A_diag_j[kk];
	       ahat[k1] -= A_diag_data[kk]*distribute;
	       if (k1 < kmin) kmin = k1;
	       if (k1 > kmax) kmax = k1;
	     }
	     if(num_procs > 1)
	     {
	       for (kk = A_offd_i[i1]; kk < A_offd_i[i1+1]; kk++)
	       {
		 k1 = A_offd_j[kk];
		 if(num_functions == 1 || dof_func[i1] == dof_func[k1])
		 {
		   ahat_offd[k1] -= A_offd_data[kk]*distribute;
		   if(kmin_offd > -1)
		   {
		     if (k1 < kmin_offd) kmin_offd = k1;
		     if (k1 > kmax_offd) kmax_offd = k1;
		   }
		   else
		   {
		     kmin_offd = k1;
		     kmax_offd = k1;
		   }
		 }
	       }
	     }
	   }
	 }
       }
       if(num_procs > 1)
       {
	 for(jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
	 {
	   i1 = A_offd_j[jj];
	   if(P_marker_offd[i1] >= jj_begin_row_offd)
	   {
	     ahat_offd[i1] += A_offd_data[jj];
	     if(kmin_offd > -1)
	     {
	       if (i1 < kmin_offd) kmin_offd = i1;
	       if (i1 > kmax_offd) kmax_offd = i1;
	     }
	     else
	     {
	       kmin_offd = i1;
	       kmax_offd = i1;
	     }
	   }
	   else
	   {
	     if(num_functions == 1 || dof_func[i] == dof_func_offd[i1])
	     {
	       distribute = A_offd_data[jj]/A_ext_data[A_ext_i[i1]];
	       for (kk = A_ext_i[i1]+1; kk < A_ext_i[i1+1]; kk++)
	       {
		 k1 = A_ext_j[kk];
		 if(k1 >= col_1 && k1 < col_n)
		 { /*diag*/
		   loc_col = k1 - col_1;
		   ahat[loc_col] -= A_ext_data[kk]*distribute;
		   if (loc_col < kmin) kmin = loc_col;
		   if (loc_col > kmax) kmax = loc_col;
		 }
		 else
		 {
		   loc_col = -k1 - 1;
		   if(num_functions == 1 || 
		      dof_func[loc_col] == dof_func_offd[i1])
		   {
		     ahat_offd[loc_col] -= A_ext_data[kk]*distribute;
		     if(kmin_offd > -1)
		     {
		       if (loc_col < kmin_offd) kmin_offd = loc_col;
		       if (loc_col > kmax_offd) kmax_offd = loc_col;
		     }
		     else
		     {
		       kmin_offd = loc_col;
		       kmax_offd = loc_col;
		     }
		   }
		 }
	       }
	     }
	   }
	 }
       }

       diagonal = ahat[i];
       ahat[i] = 0;
       sum_pos = 0;
       sum_pos_C = 0;
       sum_neg = 0;
       sum_neg_C = 0;
       sum = 0;
       sum_C = 0;
       if(sep_weight == 1)
       {
	 for (jj=kmin; jj < kmax+1; jj++)
	 {
	   if (ahat[jj] > 0)
	   {
	     sum_pos += ahat[jj];
	     if (P_marker[jj] >= jj_begin_row) sum_pos_C += ahat[jj];
	     else ahat[jj] = 0;
	   }
	   else if (ahat[jj] < 0)
	   {
	     sum_neg += ahat[jj];
	     if (P_marker[jj] >= jj_begin_row) sum_neg_C += ahat[jj];
	     else ahat[jj] = 0;
	   }
	 }
	 if(num_procs > 1 && kmin_offd > -1)
	 {
	   for (jj=kmin_offd; jj < kmax_offd+1; jj++)
	   {
	     if (ahat_offd[jj] > 0)
	     {
	       sum_pos += ahat_offd[jj];
	       if (P_marker_offd[jj] >= jj_begin_row_offd) sum_pos_C += 
							     ahat_offd[jj];
	       else ahat_offd[jj] = 0;
	     }
	     else if (ahat_offd[jj] < 0)
	     {
	       sum_neg += ahat_offd[jj];
	       if (P_marker_offd[jj] >= jj_begin_row_offd) sum_neg_C += 
							     ahat_offd[jj];
	       else ahat_offd[jj] = 0;
	     }
	   }
	 }
	 if (sum_neg_C) alfa = sum_neg/sum_neg_C/diagonal;
	 if (sum_pos_C) beta = sum_pos/sum_pos_C/diagonal;
       
	 /*-----------------------------------------------------------------
	  * Set interpolation weight by dividing by the diagonal.
	  *-----------------------------------------------------------------*/
       
	 for (jj = jj_begin_row; jj < jj_end_row; jj++)
	 {
	   j1 = P_diag_j[jj];
	   if (ahat[j1] > 0)
	     P_diag_data[jj] = -beta*ahat[j1];
	   else 
	     P_diag_data[jj] = -alfa*ahat[j1];
  
	 P_diag_j[jj] = fine_to_coarse[j1];
	 ahat[j1] = 0;
	 }
	 if(num_procs > 1)
	 {
	   for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
	   {
	     j1 = P_offd_j[jj];
	     if (ahat_offd[j1] > 0)
	       P_offd_data[jj] = -beta*ahat_offd[j1];
	     else 
	       P_offd_data[jj] = -alfa*ahat_offd[j1];
	   
	   ahat_offd[j1] = 0;
	   }
	 }
       }
       else
       {
	 for (jj=kmin; jj < kmax+1; jj++)
	 {
	   if (ahat[jj] != 0)
	   {
	     sum += ahat[jj];
	     if (P_marker[jj] >= jj_begin_row) sum_C += ahat[jj];
	     else ahat[jj] = 0;
	   }
	 }
	 if(num_procs > 1 && kmin_offd > -1)
	 {
	   for (jj=kmin_offd; jj < kmax_offd+1; jj++)
	   {
	     if(ahat_offd[jj] != 0)
	     {
	       sum += ahat_offd[jj];
	       if(P_marker_offd[jj] >= jj_begin_row_offd) sum_C += 
							    ahat_offd[jj];
	       else ahat_offd[jj] = 0;
	     }
	   }
	 }
	 if (sum_C) alfa = sum/sum_C/diagonal;
       
	 /*-----------------------------------------------------------------
	  * Set interpolation weight by dividing by the diagonal.
	  *-----------------------------------------------------------------*/
       
	 for (jj = jj_begin_row; jj < jj_end_row; jj++)
	 {
	   j1 = P_diag_j[jj];
	   if (ahat[j1] != 0)
	     P_diag_data[jj] = -alfa*ahat[j1];
  
	   P_diag_j[jj] = fine_to_coarse[j1];
	   ahat[j1] = 0;
	 }
	 if(num_procs > 1)
	 {
	   for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
	   {
	     j1 = P_offd_j[jj];
	     if (ahat_offd[j1] != 0)
	       P_offd_data[jj] = -alfa*ahat_offd[j1];
	     
	     ahat_offd[j1] = 0;
	   }
	 }
       }
     }
   }
   
   P = hypre_ParCSRMatrixCreate(comm,
				hypre_ParCSRMatrixGlobalNumRows(A),
				total_global_cpts,
				hypre_ParCSRMatrixColStarts(A),
				num_cpts_global,
				0,
				P_diag_i[n_fine],
				P_offd_i[n_fine]);
               
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   num_cols_P_offd = 0;
   if(P_offd_size)
   {
     hypre_TFree(P_marker);
     P_marker = hypre_CTAlloc(int, full_off_procNodes);
     
     for (i=0; i < full_off_procNodes; i++)
       P_marker[i] = 0;
     
     num_cols_P_offd = 0;
     for (i=0; i < P_offd_size; i++)
     {
       index = P_offd_j[i];
       if (!P_marker[index])
       {
	 if(tmp_CF_marker_offd[index] >= 0)
	 {
	   num_cols_P_offd++;
	   P_marker[index] = 1;
	 }
       }
     }
     
     col_map_offd_P = hypre_CTAlloc(int, num_cols_P_offd);
     
     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while( P_marker[index] == 0) index++;
       col_map_offd_P[i] = index++;
     }
     for(i = 0; i < P_offd_size; i++)
       P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
					P_offd_j[i],
					num_cols_P_offd);

     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while (P_marker[index] == 0) index++;
       
       col_map_offd_P[i] = fine_to_coarse_offd[index];
       index++;
     }

     /* Sort the col_map_offd_P and P_offd_j correctly */
     for(i = 0; i < num_cols_P_offd; i++)
       P_marker[i] = col_map_offd_P[i];

     /* Check if sort actually changed anything */
     if(ssort(col_map_offd_P,num_cols_P_offd))
     {
       for(i = 0; i < P_offd_size; i++)
	 for(j = 0; j < num_cols_P_offd; j++)
	   if(P_marker[P_offd_j[i]] == col_map_offd_P[j])
	   {
	     P_offd_j[i] = j;
	     j = num_cols_P_offd;
	   }
     }
     hypre_TFree(P_marker); 
   }

   if (num_cols_P_offd)
   { 
     hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
     hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   hypre_MatvecCommPkgCreate(P);
 
   *P_ptr = P;

   /* Deallocate memory */   
   hypre_TFree(fine_to_coarse);
   hypre_TFree(P_marker);
   hypre_TFree(ahat);

   if (num_procs > 1) 
   {
     hypre_CSRMatrixDestroy(Sop);
     hypre_CSRMatrixDestroy(A_ext);
     hypre_TFree(fine_to_coarse_offd);
     hypre_TFree(P_marker_offd);
     hypre_TFree(ahat_offd);
     hypre_TFree(CF_marker_offd);
     hypre_TFree(tmp_CF_marker_offd);
     hypre_TFree(P_node_add);
     if(num_functions > 1)
       hypre_TFree(dof_func_offd);
     hypre_TFree(found);

     if (hypre_ParCSRCommPkgSendProcs(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendProcs(new_comm_pkg));
     if (hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg));
     if (hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg));
     if (hypre_ParCSRCommPkgRecvProcs(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgRecvProcs(new_comm_pkg));
     if (hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg));
     hypre_TFree(new_comm_pkg);
   }
   
   return(0);  
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildExtInterp
 *  Comment: 
 *--------------------------------------------------------------------------*/
int
hypre_BoomerAMGBuildExtInterp(hypre_ParCSRMatrix *A, int *CF_marker,
			      hypre_ParCSRMatrix   *S, int *num_cpts_global,
			      int num_functions, int *dof_func, int debug_flag,
			      double trunc_factor, int max_elmts, 
			      int *col_offd_S_to_A,
			      hypre_ParCSRMatrix  **P_ptr)
{
  /* Communication Variables */
  MPI_Comm 	           comm = hypre_ParCSRMatrixComm(A);   
  hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommPkg     *new_comm_pkg;

  int              my_id, num_procs;

  /* Variables to store input variables */
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A); 
  double          *A_diag_data = hypre_CSRMatrixData(A_diag);
  int             *A_diag_i = hypre_CSRMatrixI(A_diag);
  int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
  double          *A_offd_data = hypre_CSRMatrixData(A_offd);
  int             *A_offd_i = hypre_CSRMatrixI(A_offd);
  int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

  int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
  int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  int              n_fine = hypre_CSRMatrixNumRows(A_diag);
  int              col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
  int              local_numrows = hypre_CSRMatrixNumRows(A_diag);
  int              col_n = col_1 + local_numrows;
  int              total_global_cpts, my_first_cpt;

  /* Variables to store strong connection matrix info */
  hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
  int             *S_diag_i = hypre_CSRMatrixI(S_diag);
  int             *S_diag_j = hypre_CSRMatrixJ(S_diag);
   
  hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
  int             *S_offd_i = hypre_CSRMatrixI(S_offd);
  int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

  /* Interpolation matrix P */
  hypre_ParCSRMatrix *P;
  hypre_CSRMatrix    *P_diag;
  hypre_CSRMatrix    *P_offd;   

  double          *P_diag_data;
  int             *P_diag_i, *P_diag_j;
  double          *P_offd_data;
  int             *P_offd_i, *P_offd_j;

  int		  *col_map_offd_P;
  int              P_diag_size; 
  int              P_offd_size;
  int             *P_marker; 
  int             *P_marker_offd;
  int             *P_node_add;  
  int             *CF_marker_offd, *tmp_CF_marker_offd;
  int             *dof_func_offd = NULL;

  /* Full row information for columns of A that are off diag*/
  hypre_CSRMatrix *A_ext;   
  double          *A_ext_data;
  int             *A_ext_i;
  int             *A_ext_j;
  
  int             *fine_to_coarse;
  int             *fine_to_coarse_offd;
  int             *found;

  int              num_cols_P_offd;
  int              newoff, loc_col;
  int              A_ext_rows, full_off_procNodes;
  
  hypre_CSRMatrix *Sop;
  int             *Sop_i;
  int             *Sop_j;
  
  int              Soprows;
  
  /* Variables to keep count of interpolatory points */
  int              jj_counter, jj_counter_offd;
  int              jj_begin_row, jj_end_row;
  int              jj_begin_row_offd, jj_end_row_offd;
  int              coarse_counter, coarse_counter_offd;
    
  /* Interpolation weight variables */
  double           sum, diagonal, distribute;
  int              strong_f_marker = -2;
 
  /* Loop variables */
  int              index;
  int              start_indexing = 0;
  int              i, i1, i2, j, jj, kk, k1, jj1;

  /* Definitions */
  double           zero = 0.0;
  double           one  = 1.0;
  
  /* Expanded communication for neighbor of neighbors */
  int new_num_recvs, *new_recv_procs, *new_recv_vec_starts;
  int new_num_sends, *new_send_procs, *new_send_map_starts;
  int *new_send_map_elmts;
  
  /* BEGIN */
  MPI_Comm_size(comm, &num_procs);   
  MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   MPI_Bcast(&total_global_cpts, 1, MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(A);
#else
     hypre_MatvecCommPkgCreate(A);
#endif
     comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   /* Set up off processor information (specifically for neighbors of 
    * neighbors */
   newoff = 0;
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
     /*----------------------------------------------------------------------
      * Get the off processors rows for A and S, associated with columns in 
      * A_offd and S_offd.
      *---------------------------------------------------------------------*/
     A_ext         = hypre_ParCSRMatrixExtractBExt(A,A,1);
     A_ext_i       = hypre_CSRMatrixI(A_ext);
     A_ext_j       = hypre_CSRMatrixJ(A_ext);
     A_ext_data    = hypre_CSRMatrixData(A_ext);
     A_ext_rows    = hypre_CSRMatrixNumRows(A_ext);
     
     Sop           = hypre_ParCSRMatrixExtractBExt(S,S,0);
     Sop_i         = hypre_CSRMatrixI(Sop);
     Sop_j         = hypre_CSRMatrixJ(Sop);
     Soprows       = hypre_CSRMatrixNumRows(Sop);

     /* Find nodes that are neighbors of neighbors, not found in offd */
     newoff = new_offd_nodes(&found, A_ext_rows, A_ext_i, A_ext_j, 
			     num_cols_A_offd, col_map_offd, col_1, col_n, 
			     Sop_i, Sop_j);
     if(newoff >= 0)
       full_off_procNodes = newoff + num_cols_A_offd;
     else
       return(1);

     /* Possibly add new points and new processors to the comm_pkg, all
      * processors need new_comm_pkg */
     new_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);
     hypre_ParCSRCommExtendA(A, newoff, found, &new_num_recvs, 
			     &new_recv_procs, &new_recv_vec_starts, 
			     &new_num_sends, &new_send_procs, 
			     &new_send_map_starts, &new_send_map_elmts, 
			     &P_node_add);
     
     hypre_ParCSRCommPkgComm(new_comm_pkg) = comm;
     hypre_ParCSRCommPkgNumRecvs(new_comm_pkg) = new_num_recvs;
     hypre_ParCSRCommPkgRecvProcs(new_comm_pkg) = new_recv_procs;
     hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg) = new_recv_vec_starts;
     hypre_ParCSRCommPkgNumSends(new_comm_pkg) = new_num_sends;
     hypre_ParCSRCommPkgSendProcs(new_comm_pkg) = new_send_procs;
     hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg) = new_send_map_starts;
     hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg) = new_send_map_elmts;
   
     /* Insert nodes that are added from neighbors of neighbors connections */
     CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
     if (num_functions > 1 && full_off_procNodes > 0)
       dof_func_offd = hypre_CTAlloc(int, full_off_procNodes);
    
     insert_new_nodes(new_comm_pkg, CF_marker, P_node_add, num_cols_A_offd, 
		      full_off_procNodes, num_procs, CF_marker_offd);
     if(num_functions > 1)
       insert_new_nodes(new_comm_pkg, dof_func, P_node_add, num_cols_A_offd, 
	 full_off_procNodes, num_procs, dof_func_offd);
   }
   else
     new_comm_pkg = comm_pkg;

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_i    = hypre_CTAlloc(int, n_fine+1);

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
   fine_to_coarse_offd = hypre_CTAlloc(int, full_off_procNodes);

   P_marker = hypre_CTAlloc(int, n_fine);
   P_marker_offd = hypre_CTAlloc(int, full_off_procNodes);

   tmp_CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);

   initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse, 
		  fine_to_coarse_offd, P_marker, P_marker_offd,
		  tmp_CF_marker_offd);

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;
   coarse_counter_offd = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
     P_diag_i[i] = jj_counter;
     if (num_procs > 1)
       P_offd_i[i] = jj_counter_offd;
     
     if (CF_marker[i] >= 0)
     {
       jj_counter++;
       fine_to_coarse[i] = coarse_counter;
       coarse_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, interpolation is from the C-points that
      *  strongly influence i, or C-points that stronly influence F-points
      *  that strongly influence i.
      *--------------------------------------------------------------------*/
     else
     {
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];           
	 if (CF_marker[i1] >= 0)
	 { /* i1 is a C point */
	   if (P_marker[i1] < P_diag_i[i])
	   {
	     P_marker[i1] = jj_counter;
	     jj_counter++;
	   }
	 }
	 else
	 { /* i1 is a F point, loop through it's strong neighbors */
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] >= 0)
	     {
	       if(P_marker[k1] < P_diag_i[i])
	       {
		 P_marker[k1] = jj_counter;
		 jj_counter++;
	       }
	     } 
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if (CF_marker_offd[k1] >= 0)
	       {
		 if(P_marker_offd[k1] < P_offd_i[i])
		 {
		   tmp_CF_marker_offd[k1] = 1;
		   P_marker_offd[k1] = jj_counter_offd;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       /* Look at off diag strong connections of i */ 
       if (num_procs > 1)
       {
	 for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[S_offd_j[jj]];
	   else
	     i1 = S_offd_j[jj];           
	   if (CF_marker_offd[i1] >= 0)
	   {
	     if(P_marker_offd[i1] < P_offd_i[i])
	     {
	       tmp_CF_marker_offd[i1] = 1;
	       P_marker_offd[i1] = jj_counter_offd;
	       jj_counter_offd++;
	     }
	   }
	   else
	   { /* F point; look at neighbors of i1. Sop contains global col
	      * numbers and entries that could be in S_diag or S_offd or
	      * neither. */
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       k1 = Sop_j[kk];
	       if(k1 >= col_1 && k1 < col_n)
	       { /* In S_diag */
		 loc_col = k1-col_1;
		 if(CF_marker[loc_col] >= 0)
		 {
		   if(P_marker[loc_col] < P_diag_i[i])
		   {
		     P_marker[loc_col] = jj_counter;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       {
		 loc_col = -k1 - 1; 
		 if(CF_marker_offd[loc_col] >= 0)
		 {
		   if(P_marker_offd[loc_col] < P_offd_i[i])
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     tmp_CF_marker_offd[loc_col] = 1;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       } 
     }
   }
   
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;
   
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size);

   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size);

   P_diag_i[n_fine] = jj_counter; 
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /* Fine to coarse mapping */
   if(num_procs > 1)
   {
     for (i = 0; i < n_fine; i++)
       fine_to_coarse[i] += my_first_cpt;
     insert_new_nodes(new_comm_pkg, fine_to_coarse, P_node_add, 
		      num_cols_A_offd, full_off_procNodes, num_procs, 
		      fine_to_coarse_offd);
     for (i = 0; i < n_fine; i++)
       fine_to_coarse[i] -= my_first_cpt;
   }

   for (i = 0; i < n_fine; i++)     
     P_marker[i] = -1;
     
   for (i = 0; i < full_off_procNodes; i++)
     P_marker_offd[i] = -1;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
     jj_begin_row = jj_counter;        
     jj_begin_row_offd = jj_counter_offd;

     /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
     
     if (CF_marker[i] >= 0)
     {
       P_diag_j[jj_counter]    = fine_to_coarse[i];
       P_diag_data[jj_counter] = one;
       jj_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
     
     else
     {
       strong_f_marker--;
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];   
	 
	 /*--------------------------------------------------------------
	  * If neighbor i1 is a C-point, set column number in P_diag_j
	  * and initialize interpolation weight to zero.
	  *--------------------------------------------------------------*/
	 
	 if (CF_marker[i1] >= 0)
	 {
	   if (P_marker[i1] < jj_begin_row)
	   {
	     P_marker[i1] = jj_counter;
	     P_diag_j[jj_counter]    = fine_to_coarse[i1];
	     P_diag_data[jj_counter] = zero;
	     jj_counter++;
	   }
	 }
	 else 
	 {
	   P_marker[i1] = strong_f_marker;
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] >= 0)
	     {
	       if(P_marker[k1] < jj_begin_row)
	       {
		 P_marker[k1] = jj_counter;
		 P_diag_j[jj_counter] = fine_to_coarse[k1];
		 P_diag_data[jj_counter] = zero;
		 jj_counter++;
	       }
	     }
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if(CF_marker_offd[k1] >= 0)
	       {
		 if(P_marker_offd[k1] < jj_begin_row_offd)
		 {
		   P_marker_offd[k1] = jj_counter_offd;
		   P_offd_j[jj_counter_offd] = k1;
		   P_offd_data[jj_counter_offd] = zero;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       
       if ( num_procs > 1)
       {
	 for (jj=S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[S_offd_j[jj]];
	   else
	     i1 = S_offd_j[jj];
	   if ( CF_marker_offd[i1] >= 0)
	   {
	     if(P_marker_offd[i1] < jj_begin_row_offd)
	     {
	       P_marker_offd[i1] = jj_counter_offd;
	       P_offd_j[jj_counter_offd] = i1;
	       P_offd_data[jj_counter_offd] = zero;
	       jj_counter_offd++;
	     }
	   }
	   else
	   {
	     P_marker_offd[i1] = strong_f_marker;
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       k1 = Sop_j[kk];
	       /* Find local col number */
	       if(k1 >= col_1 && k1 < col_n)
	       {
		 loc_col = k1-col_1;
		 if(CF_marker[loc_col] >= 0)
		 {
		   if(P_marker[loc_col] < jj_begin_row)
		   {		
		     P_marker[loc_col] = jj_counter;
		     P_diag_j[jj_counter] = fine_to_coarse[loc_col];
		     P_diag_data[jj_counter] = zero;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       { 
		 loc_col = -k1 - 1;
		 if(CF_marker_offd[loc_col] >= 0)
		 {
		   if(P_marker_offd[loc_col] < jj_begin_row_offd)
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     P_offd_j[jj_counter_offd]=loc_col;
		     P_offd_data[jj_counter_offd] = zero;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       }
       
       jj_end_row = jj_counter;
       jj_end_row_offd = jj_counter_offd;
       
       diagonal = A_diag_data[A_diag_i[i]];
       
       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
       { /* i1 is a c-point and strongly influences i, accumulate
	  * a_(i,i1) into interpolation weight */
	 i1 = A_diag_j[jj];
	 if (P_marker[i1] >= jj_begin_row)
	 {
	   P_diag_data[P_marker[i1]] += A_diag_data[jj];
	 }
	 else if(P_marker[i1] == strong_f_marker)
	 {
	   sum = zero;
	   /* Loop over row of A for point i1 and calculate the sum
            * of the connections to c-points that strongly incluence i. */
	   for(jj1 = A_diag_i[i1]+1; jj1 < A_diag_i[i1+1]; jj1++)
	   {
	     i2 = A_diag_j[jj1];
	     if(P_marker[i2] >= jj_begin_row || i2 == i)
	       sum += fabs(A_diag_data[jj1]);
	   }
	   if(num_procs > 1)
	   {
	     for(jj1 = A_offd_i[i1]; jj1< A_offd_i[i1+1]; jj1++)
	     {
	       i2 = A_offd_j[jj1];
	       if(P_marker_offd[i2] >= jj_begin_row_offd)
		 sum += fabs(A_offd_data[jj1]);
	     }
	   }
	   if(sum != 0)
	   {
	     distribute = A_diag_data[jj]/sum;
	     /* Loop over row of A for point i1 and do the distribution */
	     for(jj1 = A_diag_i[i1]+1; jj1 < A_diag_i[i1+1]; jj1++)
	     {
	       i2 = A_diag_j[jj1];
	       if(P_marker[i2] >= jj_begin_row)
		 P_diag_data[P_marker[i2]] += 
		   distribute*fabs(A_diag_data[jj1]);
	       if(i2 == i)
		 diagonal += distribute*fabs(A_diag_data[jj1]);
	     }
	     if(num_procs > 1)
	     {
	       for(jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
	       {
		 i2 = A_offd_j[jj1];
		 if(P_marker_offd[i2] >= jj_begin_row_offd)
		   P_offd_data[P_marker_offd[i2]] +=
		     distribute*fabs(A_offd_data[jj1]);
	       }
	     }
	   }
	   else
	   {
	     diagonal += A_diag_data[jj];
	   }
	 }
	 /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
	  * diagonal */
	 else
	 {
	   if(num_functions == 1 || dof_func[i] == dof_func[i1])
	     diagonal += A_diag_data[jj];
	 }
       }
       if(num_procs > 1)
       {
	 for(jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
	 {
	   i1 = A_offd_j[jj];
	   if(P_marker_offd[i1] >= jj_begin_row_offd)
	     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
	   else if(P_marker_offd[i1] == strong_f_marker)
	   {
	     sum = zero;
	     for(jj1 = A_ext_i[i1]+1; jj1 < A_ext_i[i1+1]; jj1++)
	     {
	       k1 = A_ext_j[jj1];
	       if(k1 >= col_1 && k1 < col_n)
	       { /* diag */
		 loc_col = k1 - col_1;
		 if(P_marker[loc_col] >= jj_begin_row || loc_col == i)
		   sum += fabs(A_ext_data[jj1]);
	       }
	       else
	       { 
		 loc_col = -k1 - 1;
		 if(P_marker_offd[loc_col] >= jj_begin_row_offd)
		   sum += fabs(A_ext_data[jj1]);
	       }
	     }
	     if(sum != 0)
	     {
	       distribute = A_offd_data[jj] / sum;
	       for(jj1 = A_ext_i[i1]+1; jj1 < A_ext_i[i1+1]; jj1++)
	       {
		 k1 = A_ext_j[jj1];
		 if(k1 >= col_1 && k1 < col_n)
		 { /* diag */
		   loc_col = k1 - col_1;
		   if(P_marker[loc_col] >= jj_begin_row)
		     P_diag_data[P_marker[loc_col]] += distribute*
		       fabs(A_ext_data[jj1]);
		   if(loc_col == i)
		     diagonal += distribute*fabs(A_ext_data[jj1]);
		 }
		 else
		 { 
		   loc_col = -k1 - 1;
		   if(P_marker_offd[loc_col] >= jj_begin_row_offd)
		     P_offd_data[P_marker_offd[loc_col]] += distribute*
		       fabs(A_ext_data[jj1]);
		 }
	       }
	     }
	     else
	     {
	       diagonal += A_offd_data[jj];
	     }
	   }
	   else
	   {
	     if(num_functions == 1 || dof_func[i] == dof_func_offd[i1])
	       diagonal += A_offd_data[jj];
	   }
	 }
       }
       for(jj = jj_begin_row; jj < jj_end_row; jj++)
	 P_diag_data[jj] /= -diagonal;
       for(jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
	 P_offd_data[jj] /= -diagonal;
     }
     strong_f_marker--;
   }
   
   P = hypre_ParCSRMatrixCreate(comm,
				hypre_ParCSRMatrixGlobalNumRows(A),
				total_global_cpts,
				hypre_ParCSRMatrixColStarts(A),
				num_cpts_global,
				0,
				P_diag_i[n_fine],
				P_offd_i[n_fine]);
               
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   num_cols_P_offd = 0;
   if(P_offd_size)
   {
     hypre_TFree(P_marker);
     P_marker = hypre_CTAlloc(int, full_off_procNodes);
     
     for (i=0; i < full_off_procNodes; i++)
       P_marker[i] = 0;
     
     num_cols_P_offd = 0;
     for (i=0; i < P_offd_size; i++)
     {
       index = P_offd_j[i];
       if (!P_marker[index])
       {
	 if(tmp_CF_marker_offd[index] >= 0)
	 {
	   num_cols_P_offd++;
	   P_marker[index] = 1;
	 }
       }
     }
     
     col_map_offd_P = hypre_CTAlloc(int, num_cols_P_offd);
     
     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while( P_marker[index] == 0) index++;
       col_map_offd_P[i] = index++;
     }
     for(i = 0; i < P_offd_size; i++)
       P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
					P_offd_j[i],
					num_cols_P_offd);

     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while (P_marker[index] == 0) index++;
       
       col_map_offd_P[i] = fine_to_coarse_offd[index];
       index++;
     }

     /* Sort the col_map_offd_P and P_offd_j correctly */
     for(i = 0; i < num_cols_P_offd; i++)
       P_marker[i] = col_map_offd_P[i];

     /* Check if sort actually changed anything */
     if(ssort(col_map_offd_P,num_cols_P_offd))
     {
       for(i = 0; i < P_offd_size; i++)
	 for(j = 0; j < num_cols_P_offd; j++)
	   if(P_marker[P_offd_j[i]] == col_map_offd_P[j])
	   {
	     P_offd_j[i] = j;
	     j = num_cols_P_offd;
	   }
     }
     hypre_TFree(P_marker); 
   }

   if (num_cols_P_offd)
   { 
     hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
     hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   hypre_MatvecCommPkgCreate(P);
 
   *P_ptr = P;

   /* Deallocate memory */   
   hypre_TFree(fine_to_coarse);
   hypre_TFree(P_marker);
   
   if (num_procs > 1) 
   {
     hypre_CSRMatrixDestroy(Sop);
     hypre_CSRMatrixDestroy(A_ext);
     hypre_TFree(fine_to_coarse_offd);
     hypre_TFree(P_marker_offd);
     hypre_TFree(CF_marker_offd);
     hypre_TFree(tmp_CF_marker_offd);
     if(num_functions > 1)
       hypre_TFree(dof_func_offd);
     hypre_TFree(found);
     hypre_TFree(P_node_add);

     if (hypre_ParCSRCommPkgSendProcs(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendProcs(new_comm_pkg));
     if (hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg));
     if (hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg));
     if (hypre_ParCSRCommPkgRecvProcs(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgRecvProcs(new_comm_pkg));
     if (hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg));
     hypre_TFree(new_comm_pkg);
   }
   
   return(0);  
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildFFInterp
 *  Comment: Only use FF when there is no common c point.
 *--------------------------------------------------------------------------*/
int
hypre_BoomerAMGBuildFFInterp(hypre_ParCSRMatrix *A, int *CF_marker,
			     hypre_ParCSRMatrix   *S, int *num_cpts_global,
			     int num_functions, int *dof_func, int debug_flag,
			     double trunc_factor, int max_elmts, 
			     int *col_offd_S_to_A,
			     hypre_ParCSRMatrix  **P_ptr)
{
  /* Communication Variables */
  MPI_Comm 	           comm = hypre_ParCSRMatrixComm(A);   
  hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommPkg     *new_comm_pkg;

  int              my_id, num_procs;

  /* Variables to store input variables */
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A); 
  double          *A_diag_data = hypre_CSRMatrixData(A_diag);
  int             *A_diag_i = hypre_CSRMatrixI(A_diag);
  int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
  double          *A_offd_data = hypre_CSRMatrixData(A_offd);
  int             *A_offd_i = hypre_CSRMatrixI(A_offd);
  int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

  int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
  int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  int              n_fine = hypre_CSRMatrixNumRows(A_diag);
  int              col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
  int              local_numrows = hypre_CSRMatrixNumRows(A_diag);
  int              col_n = col_1 + local_numrows;
  int              total_global_cpts, my_first_cpt;

  /* Variables to store strong connection matrix info */
  hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
  int             *S_diag_i = hypre_CSRMatrixI(S_diag);
  int             *S_diag_j = hypre_CSRMatrixJ(S_diag);
   
  hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
  int             *S_offd_i = hypre_CSRMatrixI(S_offd);
  int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

  /* Interpolation matrix P */
  hypre_ParCSRMatrix *P;
  hypre_CSRMatrix    *P_diag;
  hypre_CSRMatrix    *P_offd;   

  double          *P_diag_data;
  int             *P_diag_i, *P_diag_j;
  double          *P_offd_data;
  int             *P_offd_i, *P_offd_j;

  int		  *col_map_offd_P;
  int              P_diag_size; 
  int              P_offd_size;
  int             *P_marker; 
  int             *P_marker_offd;
  int             *P_node_add;  
  int             *CF_marker_offd, *tmp_CF_marker_offd;
  int             *dof_func_offd = NULL;
  /*int             **ext_p, **ext_p_offd;*/
  int              ccounter_offd;
  int             *clist_offd;
  int              common_c;

  /* Full row information for columns of A that are off diag*/
  hypre_CSRMatrix *A_ext;   
  double          *A_ext_data;
  int             *A_ext_i;
  int             *A_ext_j;
  
  int             *fine_to_coarse;
  int             *fine_to_coarse_offd;
  int             *found;

  int              num_cols_P_offd;
  int              newoff, loc_col;
  int              A_ext_rows, full_off_procNodes;
  
  hypre_CSRMatrix *Sop;
  int             *Sop_i;
  int             *Sop_j;
  
  int              Soprows;
  
  /* Variables to keep count of interpolatory points */
  int              jj_counter, jj_counter_offd;
  int              jj_begin_row, jj_end_row;
  int              jj_begin_row_offd, jj_end_row_offd;
  int              coarse_counter, coarse_counter_offd;
    
  /* Interpolation weight variables */
  double           sum, diagonal, distribute, sum_direct;
  int              strong_f_marker = -2;
  int              sgn;

  /* Loop variables */
  int              index;
  int              start_indexing = 0;
  int              i, i1, i2, j, jj, kk, k1, jj1;
  int             *clist, ccounter;

  /* Definitions */
  double           zero = 0.0;
  double           one  = 1.0;
  
  /* Expanded communication for neighbor of neighbors */
  int new_num_recvs, *new_recv_procs, *new_recv_vec_starts;
  int new_num_sends, *new_send_procs, *new_send_map_starts;
  int *new_send_map_elmts;
  
  /* BEGIN */
  MPI_Comm_size(comm, &num_procs);   
  MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   MPI_Bcast(&total_global_cpts, 1, MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(A);
#else
     hypre_MatvecCommPkgCreate(A);
#endif
     comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   /* Set up off processor information (specifically for neighbors of 
    * neighbors */
   newoff = 0;
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
     /*----------------------------------------------------------------------
      * Get the off processors rows for A and S, associated with columns in 
      * A_offd and S_offd.
      *---------------------------------------------------------------------*/
     A_ext         = hypre_ParCSRMatrixExtractBExt(A,A,1);
     A_ext_i       = hypre_CSRMatrixI(A_ext);
     A_ext_j       = hypre_CSRMatrixJ(A_ext);
     A_ext_data    = hypre_CSRMatrixData(A_ext);
     A_ext_rows    = hypre_CSRMatrixNumRows(A_ext);
     
     Sop           = hypre_ParCSRMatrixExtractBExt(S,S,0);
     Sop_i         = hypre_CSRMatrixI(Sop);
     Sop_j         = hypre_CSRMatrixJ(Sop);
     Soprows       = hypre_CSRMatrixNumRows(Sop);

     /* Find nodes that are neighbors of neighbors, not found in offd */
     newoff = new_offd_nodes(&found, A_ext_rows, A_ext_i, A_ext_j, 
			     num_cols_A_offd, col_map_offd, col_1, col_n, 
			     Sop_i, Sop_j);
     if(newoff >= 0)
       full_off_procNodes = newoff + num_cols_A_offd;
     else
       return(1);

     /* Possibly add new points and new processors to the comm_pkg, all
      * processors need new_comm_pkg */
     new_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);
     hypre_ParCSRCommExtendA(A, newoff, found, &new_num_recvs, 
			     &new_recv_procs, &new_recv_vec_starts, 
			     &new_num_sends, &new_send_procs, 
			     &new_send_map_starts, &new_send_map_elmts, 
			     &P_node_add);
     
     hypre_ParCSRCommPkgComm(new_comm_pkg) = comm;
     hypre_ParCSRCommPkgNumRecvs(new_comm_pkg) = new_num_recvs;
     hypre_ParCSRCommPkgRecvProcs(new_comm_pkg) = new_recv_procs;
     hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg) = new_recv_vec_starts;
     hypre_ParCSRCommPkgNumSends(new_comm_pkg) = new_num_sends;
     hypre_ParCSRCommPkgSendProcs(new_comm_pkg) = new_send_procs;
     hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg) = new_send_map_starts;
     hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg) = new_send_map_elmts;
     
     /* Insert nodes that are added from neighbors of neighbors connections */
     CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
     if (num_functions > 1 && full_off_procNodes > 0)
       dof_func_offd = hypre_CTAlloc(int, full_off_procNodes);
    
     insert_new_nodes(new_comm_pkg, CF_marker, P_node_add, num_cols_A_offd, 
		      full_off_procNodes, num_procs, CF_marker_offd);
     if(num_functions > 1)
       insert_new_nodes(new_comm_pkg, dof_func, P_node_add, num_cols_A_offd, 
	 full_off_procNodes, num_procs, dof_func_offd);
   }
   else
     new_comm_pkg = comm_pkg;
   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_i    = hypre_CTAlloc(int, n_fine+1);

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
   fine_to_coarse_offd = hypre_CTAlloc(int, full_off_procNodes);

   P_marker = hypre_CTAlloc(int, n_fine);
   P_marker_offd = hypre_CTAlloc(int, full_off_procNodes);

   tmp_CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);

   clist = hypre_CTAlloc(int, MAX_C_CONNECTIONS);
   for(i = 0; i < MAX_C_CONNECTIONS; i++)
     clist[i] = 0;
   if(num_procs > 1)
   {
     clist_offd = hypre_CTAlloc(int, MAX_C_CONNECTIONS);
     for(i = 0; i < MAX_C_CONNECTIONS; i++)
       clist_offd[i] = 0;
   }

   initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse, 
		  fine_to_coarse_offd, P_marker, P_marker_offd,
		  tmp_CF_marker_offd);
   
   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;
   coarse_counter_offd = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
     P_diag_i[i] = jj_counter;
     if (num_procs > 1)
       P_offd_i[i] = jj_counter_offd;
     
     if (CF_marker[i] >= 0)
     {
       jj_counter++;
       fine_to_coarse[i] = coarse_counter;
       coarse_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, interpolation is from the C-points that
      *  strongly influence i, or C-points that stronly influence F-points
      *  that strongly influence i.
      *--------------------------------------------------------------------*/
     else
     {
       /* Initialize ccounter for each f point */
       ccounter = 0;
       ccounter_offd = 0;
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       { /* search through diag to find all c neighbors */
	 i1 = S_diag_j[jj];           
	 if (CF_marker[i1] >= 0)
	 { /* i1 is a C point */
	   if (P_marker[i1] < P_diag_i[i])
	   {
	     clist[ccounter++] = i1;
	     P_marker[i1] = jj_counter;
	     jj_counter++;
	   }
	 }
       }
       qsort0(clist,0,ccounter-1);
       if(num_procs > 1)
       {
	 for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 { /* search through offd to find all c neighbors */
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[S_offd_j[jj]];
	   else
	     i1 = S_offd_j[jj];
	   if(CF_marker_offd[i1] >= 0)
	   { /* i1 is a C point direct neighbor */
	     if(P_marker_offd[i1] < P_offd_i[i])
	     {
	       clist_offd[ccounter_offd++] = i1;
	       tmp_CF_marker_offd[i1] = 1;
	       P_marker_offd[i1] = jj_counter_offd;
	       jj_counter_offd++;
	     }
	   }
	 }
	 qsort0(clist_offd,0,ccounter_offd-1);
       }
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       { /* Search diag to find f neighbors and determine if common c point */
	 i1 = S_diag_j[jj];
	 if (CF_marker[i1] < 0)
	 { /* i1 is a F point, loop through it's strong neighbors */
	   common_c = 0;
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] >= 0)
	     {
	       if(hypre_BinarySearch(clist,k1,ccounter) >= 0)
	       {
		 common_c = 1;
		 kk = S_diag_i[i1+1];
	       }
	     }
	   }
	   if(num_procs > 1 && common_c == 0)
	   { /* no common c point yet, check offd */
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];

	       if (CF_marker_offd[k1] >= 0)
	       { /* k1 is a c point check if it is common */
		 if(hypre_BinarySearch(clist_offd,k1,ccounter_offd) >= 0)
		 {
		   common_c = 1;
		   kk = S_offd_i[i1+1];
		 }
	       }
	     }
	   }
	   if(!common_c)
	   { /* No common c point, extend the interp set */
	     for(kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	     {
	       k1 = S_diag_j[kk];
	       if(CF_marker[k1] >=0)
	       {
		 if(P_marker[k1] < P_diag_i[i])
		 {
		   P_marker[k1] = jj_counter;
		   jj_counter++;
		 }
	       } 
	     }
	     if(num_procs > 1)
	     {
	       for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	       {
		 if(col_offd_S_to_A)
		   k1 = col_offd_S_to_A[S_offd_j[kk]];
		 else
		   k1 = S_offd_j[kk];
		 if (CF_marker_offd[k1] >= 0)
		 {
		   if(P_marker_offd[k1] < P_offd_i[i])
		   {
		     tmp_CF_marker_offd[k1] = 1;
		     P_marker_offd[k1] = jj_counter_offd;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       }
       /* Look at off diag strong connections of i */ 
       if (num_procs > 1)
       {
	 for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 { 
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[S_offd_j[jj]];
	   else
	     i1 = S_offd_j[jj];           
	   if (CF_marker_offd[i1] < 0)
	   { /* F point; look at neighbors of i1. Sop contains global col
	      * numbers and entries that could be in S_diag or S_offd or
	      * neither. */
	     common_c = 0;
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     { /* Check if common c */
	       k1 = Sop_j[kk];
	       if(k1 >= col_1 && k1 < col_n)
	       { /* In S_diag */
		 loc_col = k1-col_1;
		 if(CF_marker[loc_col] >= 0)
		 {
		   if(hypre_BinarySearch(clist,loc_col,ccounter) >= 0)
		   {
		     common_c = 1;
		     kk = Sop_i[i1+1];
		   }
		 }
	       }
	       else
	       { 
		 loc_col = -k1 - 1;
		 if(CF_marker_offd[loc_col] >= 0)
		 {
		   if(hypre_BinarySearch(clist_offd,loc_col,ccounter_offd) >=
		      0)
		   {
		     common_c = 1;
		     kk = Sop_i[i1+1];
		   }
		 }
	       }
	     }
	     if(!common_c)
	     {
	       for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	       { /* Check if common c */
		 k1 = Sop_j[kk];
		 if(k1 >= col_1 && k1 < col_n)
		 { /* In S_diag */
		   loc_col = k1-col_1;
		   if(CF_marker[loc_col] >= 0)
		   {
		     if(P_marker[loc_col] < P_diag_i[i])
		     {
		       P_marker[loc_col] = jj_counter;
		       jj_counter++;
		     }
		   }
		 }
		 else
		 { 
		   loc_col = -k1 - 1;
		   if(CF_marker_offd[loc_col] >= 0)
		   {
		     if(P_marker_offd[loc_col] < P_offd_i[i])
		     {
		       P_marker_offd[loc_col] = jj_counter_offd;
		       tmp_CF_marker_offd[loc_col] = 1;
		       jj_counter_offd++;
		     }
		   }
		 }
	       }
	     }
	   }
	 }
       }
     }
   }
   
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;
   
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size);

   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size);

   P_diag_i[n_fine] = jj_counter; 
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   ccounter = start_indexing;
   ccounter_offd = start_indexing;

   /* Fine to coarse mapping */
   if(num_procs > 1)
   {
     for (i = 0; i < n_fine; i++)
       fine_to_coarse[i] += my_first_cpt;
     insert_new_nodes(new_comm_pkg, fine_to_coarse, P_node_add, 
		      num_cols_A_offd, full_off_procNodes, num_procs, 
		      fine_to_coarse_offd);
     for (i = 0; i < n_fine; i++)
       fine_to_coarse[i] -= my_first_cpt;
   }

   for (i = 0; i < n_fine; i++)     
     P_marker[i] = -1;
     
   for (i = 0; i < full_off_procNodes; i++)
     P_marker_offd[i] = -1;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
     jj_begin_row = jj_counter;        
     if(num_procs > 1)
       jj_begin_row_offd = jj_counter_offd;

     /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
     
     if (CF_marker[i] >= 0)
     {
       P_diag_j[jj_counter]    = fine_to_coarse[i];
       P_diag_data[jj_counter] = one;
       jj_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
     
     else
     {
       ccounter = 0;
       ccounter_offd = 0;
       strong_f_marker--;

       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       { /* Search C points only */
	 i1 = S_diag_j[jj];   
	 
	 /*--------------------------------------------------------------
	  * If neighbor i1 is a C-point, set column number in P_diag_j
	  * and initialize interpolation weight to zero.
	  *--------------------------------------------------------------*/
	 
	 if (CF_marker[i1] >= 0)
	 {
	   if (P_marker[i1] < jj_begin_row)
	   {
	     P_marker[i1] = jj_counter;
	     P_diag_j[jj_counter]    = fine_to_coarse[i1];
	     P_diag_data[jj_counter] = zero;
	     jj_counter++;
	     clist[ccounter++] = i1;
	   }
	 }
       }
       qsort0(clist,0,ccounter-1);
       if ( num_procs > 1)
       {
	 for (jj=S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[S_offd_j[jj]];
	   else
	     i1 = S_offd_j[jj];
	   if ( CF_marker_offd[i1] >= 0)
	   {
	     if(P_marker_offd[i1] < jj_begin_row_offd)
	     {
	       P_marker_offd[i1] = jj_counter_offd;
	       P_offd_j[jj_counter_offd] = i1;
	       P_offd_data[jj_counter_offd] = zero;
	       jj_counter_offd++;
	       clist_offd[ccounter_offd++] = i1;
	     }
	   }
	 }
	 qsort0(clist_offd,0,ccounter_offd-1);
       }

       for(jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       { /* Search through F points */
	 i1 = S_diag_j[jj];
	 if(CF_marker[i1] < 0) 
	 {
	   P_marker[i1] = strong_f_marker;
	   common_c = 0;
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] >= 0)
	     {
	       if(hypre_BinarySearch(clist,k1,ccounter) >= 0)
	       {
		 common_c = 1;
		 kk = S_diag_i[i1+1];
	       }
	     }
	   }
	   if(num_procs > 1 && common_c == 0)
	   { /* no common c point yet, check offd */
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];

	       if (CF_marker_offd[k1] >= 0)
	       { /* k1 is a c point check if it is common */
		 if(hypre_BinarySearch(clist_offd,k1,ccounter_offd) >= 0)
		 {
		   common_c = 1;
		   kk = S_offd_i[i1+1];
		 }
	       }
	     }
	   }
	   if(!common_c)
	   { /* No common c point, extend the interp set */
	     for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	     {
	       k1 = S_diag_j[kk];
	       if (CF_marker[k1] >= 0)
	       {
		 if(P_marker[k1] < jj_begin_row)
		 {
		   P_marker[k1] = jj_counter;
		   P_diag_j[jj_counter] = fine_to_coarse[k1];
		   P_diag_data[jj_counter] = zero;
		   jj_counter++;
		 }
	       }
	     }
	     if(num_procs > 1)
	     {
	       for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	       {
		 if(col_offd_S_to_A)
		   k1 = col_offd_S_to_A[S_offd_j[kk]];
		 else
		   k1 = S_offd_j[kk];
		 if(CF_marker_offd[k1] >= 0)
		 {
		   if(P_marker_offd[k1] < jj_begin_row_offd)
		   {
		     P_marker_offd[k1] = jj_counter_offd;
		     P_offd_j[jj_counter_offd] = k1;
		     P_offd_data[jj_counter_offd] = zero;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       }
       if ( num_procs > 1)
       {
	 for (jj=S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[S_offd_j[jj]];
	   else
	     i1 = S_offd_j[jj];
	   if(CF_marker_offd[i1] < 0)
	   { /* F points that are off proc */
	     P_marker_offd[i1] = strong_f_marker;
	     common_c = 0;
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     { /* Check if common c */
	       k1 = Sop_j[kk];
	       if(k1 >= col_1 && k1 < col_n)
	       { /* In S_diag */
		 loc_col = k1-col_1;
		 if(CF_marker[loc_col] >= 0)
		 {
		   if(hypre_BinarySearch(clist,loc_col,ccounter) >= 0)
		   {
		     common_c = 1;
		     kk = Sop_i[i1+1];
		   }
		 }
	       }
	       else
	       { 
		 loc_col = -k1 - 1;
		 if(CF_marker_offd[loc_col] >= 0)
		 {
		   if(hypre_BinarySearch(clist_offd,loc_col,ccounter_offd) >=
		      0)
		   {
		     common_c = 1;
		     kk = Sop_i[i1+1];
		   }
		 }
	       }
	     }
	     if(!common_c)
	     {
	       for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	       {
		 k1 = Sop_j[kk];
		 /* Find local col number */
		 if(k1 >= col_1 && k1 < col_n)
		 {
		   loc_col = k1-col_1;
		   if(CF_marker[loc_col] >= 0)
		   {
		     if(P_marker[loc_col] < jj_begin_row)
		     {		
		       P_marker[loc_col] = jj_counter;
		       P_diag_j[jj_counter] = fine_to_coarse[loc_col];
		       P_diag_data[jj_counter] = zero;
		       jj_counter++;
		     }
		   }
		 }
		 else
		 { 
		   loc_col = -k1 - 1;
		   if(CF_marker_offd[loc_col] >= 0)
		   {
		     if(P_marker_offd[loc_col] < jj_begin_row_offd)
		     {
		       P_marker_offd[loc_col] = jj_counter_offd;
		       P_offd_j[jj_counter_offd]=loc_col;
		       P_offd_data[jj_counter_offd] = zero;
		       jj_counter_offd++;
		     }
		   }
		 }
	       }
	     }
	   }
	 }
       }

       jj_end_row = jj_counter;
       jj_end_row_offd = jj_counter_offd;
       
       diagonal = A_diag_data[A_diag_i[i]];
       sum_direct = zero;       
       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
       { /* i1 is a c-point and strongly influences i, accumulate
	  * a_(i,i1) into interpolation weight */
	 i1 = A_diag_j[jj];
	 if (P_marker[i1] >= jj_begin_row)
	 {
	   P_diag_data[P_marker[i1]] += A_diag_data[jj];
	 }
	 else if(P_marker[i1] == strong_f_marker)
	 {
	   sum = zero;
	   sgn = 1;
	   if(A_diag_data[A_diag_i[i1]] < 0) sgn = -1;
	   /* Loop over row of A for point i1 and calculate the sum
	    * of the connections to c-points that strongly incluence i. */
	   for(jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
	   {
	     i2 = A_diag_j[jj1];
	     if(P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0) 
	       sum += A_diag_data[jj1];
	   }
	   if(num_procs > 1)
	   {
	     for(jj1 = A_offd_i[i1]; jj1< A_offd_i[i1+1]; jj1++)
	     {
	       i2 = A_offd_j[jj1];
	       if(P_marker_offd[i2] >= jj_begin_row_offd &&
		  (sgn*A_offd_data[jj1]) < 0)
		 sum += A_offd_data[jj1];
	     }
	   }
	   if(sum != 0)
	   {
	     distribute = A_diag_data[jj]/sum;
	     /* Loop over row of A for point i1 and do the distribution */
	     for(jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
	     {
	       i2 = A_diag_j[jj1];
	       if(P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0)
		 P_diag_data[P_marker[i2]] += 
		   distribute*A_diag_data[jj1];
	     }
	     if(num_procs > 1)
	     {
	       for(jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
	       {
		 i2 = A_offd_j[jj1];
		 if(P_marker_offd[i2] >= jj_begin_row_offd &&
		    (sgn*A_offd_data[jj1]) < 0)
		   P_offd_data[P_marker_offd[i2]] +=
		     distribute*A_offd_data[jj1];
	       }
	     }
	   }
	   else
	     diagonal += A_diag_data[jj];
	 }
	 /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
	  * diagonal */
	 else
	 {
	   if(num_functions == 1 || dof_func[i] == dof_func[i1])
	     diagonal += A_diag_data[jj];
	 }
       }
       if(num_procs > 1)
       {
	 for(jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
	 {
	   i1 = A_offd_j[jj];
	   if(P_marker_offd[i1] >= jj_begin_row_offd)
	     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
	   else if(P_marker_offd[i1] == strong_f_marker)
	   {
	     sum = zero;
	     sgn = 1;
	     if(A_ext_data[A_ext_i[i1]] < 0) sgn = -1;
	     for(jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1+1]; jj1++)
	     {
	       k1 = A_ext_j[jj1];
	       if(k1 >= col_1 && k1 < col_n)
	       { /* diag */
		 loc_col = k1 - col_1;
		 if(P_marker[loc_col] >= jj_begin_row && 
		    (sgn*A_ext_data[jj1]) < 0)
		   sum += A_ext_data[jj1];
	       }
	       else
	       { 
		 loc_col = -k1 - 1;
		 if(P_marker_offd[loc_col] >= jj_begin_row_offd &&
		    (sgn*A_ext_data[jj1]) < 0)
		   sum += A_ext_data[jj1];
	       }
	     }
	     if(sum != 0)
	     {
	       distribute = A_offd_data[jj] / sum;
	       for(jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1+1]; jj1++)
	       {
		 k1 = A_ext_j[jj1];
		 if(k1 >= col_1 && k1 < col_n)
		 { /* diag */
		   loc_col = k1 - col_1;
		   if(P_marker[loc_col] >= jj_begin_row && 
		      (sgn*A_ext_data[jj1]) < 0)
		     P_diag_data[P_marker[loc_col]] += distribute*
		       A_ext_data[jj1];
		 }
		 else
		 { 
		   loc_col = -k1 - 1;
		   if(P_marker_offd[loc_col] >= jj_begin_row_offd &&
		      (sgn*A_ext_data[jj1]) < 0)
		     P_offd_data[P_marker_offd[loc_col]] += distribute*
		       A_ext_data[jj1];
		 }
	       }
	     }
	     else
	       diagonal += A_offd_data[jj];
	   }
	   else
	   {
	     if(num_functions == 1 || dof_func[i] == dof_func_offd[i1])
	       diagonal += A_offd_data[jj];
	   }
	 }
       }
       for(jj = jj_begin_row; jj < jj_end_row; jj++)
	 P_diag_data[jj] /= -diagonal;
       for(jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
	 P_offd_data[jj] /= -diagonal;
     }
     strong_f_marker--;
   }
   
   P = hypre_ParCSRMatrixCreate(comm,
				hypre_ParCSRMatrixGlobalNumRows(A),
				total_global_cpts,
				hypre_ParCSRMatrixColStarts(A),
				num_cpts_global,
				0,
				P_diag_i[n_fine],
				P_offd_i[n_fine]);
               
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   num_cols_P_offd = 0;
   if(P_offd_size)
   {
     hypre_TFree(P_marker);
     P_marker = hypre_CTAlloc(int, full_off_procNodes);
     
     for (i=0; i < full_off_procNodes; i++)
       P_marker[i] = 0;
     
     num_cols_P_offd = 0;
     for (i=0; i < P_offd_size; i++)
     {
       index = P_offd_j[i];
       if (!P_marker[index])
       {
	 if(tmp_CF_marker_offd[index] >= 0)
	 {
	   num_cols_P_offd++;
	   P_marker[index] = 1;
	 }
       }
     }
     
     col_map_offd_P = hypre_CTAlloc(int, num_cols_P_offd);
     
     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while( P_marker[index] == 0) index++;
       col_map_offd_P[i] = index++;
     }
     for(i = 0; i < P_offd_size; i++)
       P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
					P_offd_j[i],
					num_cols_P_offd);

     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while (P_marker[index] == 0) index++;
       
       col_map_offd_P[i] = fine_to_coarse_offd[index];
       index++;
     }

     /* Sort the col_map_offd_P and P_offd_j correctly */
     for(i = 0; i < num_cols_P_offd; i++)
       P_marker[i] = col_map_offd_P[i];

     /* Check if sort actually changed anything */
     if(ssort(col_map_offd_P,num_cols_P_offd))
     {
       for(i = 0; i < P_offd_size; i++)
	 for(j = 0; j < num_cols_P_offd; j++)
	   if(P_marker[P_offd_j[i]] == col_map_offd_P[j])
	   {
	     P_offd_j[i] = j;
	     j = num_cols_P_offd;
	   }
     }
   }

   if (num_cols_P_offd)
   { 
     hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
     hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   hypre_MatvecCommPkgCreate(P);
 
   *P_ptr = P;

   /* Deallocate memory */   
   hypre_TFree(fine_to_coarse);
   hypre_TFree(P_marker);
   hypre_TFree(clist);
   
   if (num_procs > 1) 
   {
     hypre_TFree(P_node_add);
     hypre_TFree(clist_offd);
     hypre_CSRMatrixDestroy(Sop);
     hypre_CSRMatrixDestroy(A_ext);
     hypre_TFree(fine_to_coarse_offd);
     hypre_TFree(P_marker_offd);
     hypre_TFree(CF_marker_offd);
     hypre_TFree(tmp_CF_marker_offd);
     if(num_functions > 1)
       hypre_TFree(dof_func_offd);
     hypre_TFree(found);
     
     if (hypre_ParCSRCommPkgSendProcs(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendProcs(new_comm_pkg));
     if (hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg));
     if (hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg));
     if (hypre_ParCSRCommPkgRecvProcs(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgRecvProcs(new_comm_pkg));
     if (hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg))
       hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg));
     hypre_TFree(new_comm_pkg);
   }
   
   return(0);  
}
