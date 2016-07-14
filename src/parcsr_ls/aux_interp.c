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



#include "_hypre_parcsr_ls.h"
#include "aux_interp.h"
#include "hypre_hopscotch_hash.h"

/*---------------------------------------------------------------------------
 * Auxilary routines for the long range interpolation methods.
 *  Implemented: "standard", "extended", "multipass", "FF"
 *--------------------------------------------------------------------------*/
/* AHB 11/06: Modification of the above original - takes two
   communication packages and inserts nodes to position expected for
   OUT_marker
  
   offd nodes from comm_pkg take up first chunk of CF_marker_offd, offd 
   nodes from extend_comm_pkg take up the second chunk 0f CF_marker_offd. */



HYPRE_Int hypre_alt_insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, 
                          hypre_ParCSRCommPkg *extend_comm_pkg,
                          HYPRE_Int *IN_marker, 
                          HYPRE_Int full_off_procNodes,
                          HYPRE_Int *OUT_marker)
{   
  hypre_ParCSRCommHandle  *comm_handle;

  HYPRE_Int i, index, shift;

  HYPRE_Int num_sends, num_recvs;
  
  HYPRE_Int *recv_vec_starts;

  HYPRE_Int e_num_sends;

  HYPRE_Int *int_buf_data;
  HYPRE_Int *e_out_marker;
  

  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  num_recvs =  hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

  e_num_sends = hypre_ParCSRCommPkgNumSends(extend_comm_pkg);


  index = hypre_max(hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                    hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends));

  int_buf_data = hypre_CTAlloc(HYPRE_Int, index);

  /* orig commpkg data*/
  index = 0;
  
  HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
  HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
  for (i = begin; i < end; ++i) {
     int_buf_data[i - begin] =
           IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
					      OUT_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  /* now do the extend commpkg */

  /* first we need to shift our position in the OUT_marker */
  shift = recv_vec_starts[num_recvs];
  e_out_marker = OUT_marker + shift;
  
  index = 0;

  begin = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, 0);
  end = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
  for (i = begin; i < end; ++i) {
     int_buf_data[i - begin] =
           IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg, i)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 11, extend_comm_pkg, int_buf_data, 
					      e_out_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  hypre_TFree(int_buf_data);
    
  return hypre_error_flag;
} 

/* AHB 11/06 : alternate to the extend function below - creates a
 * second comm pkg based on found - this makes it easier to use the
 * global partition*/
HYPRE_Int
hypre_ParCSRFindExtendCommPkg(hypre_ParCSRMatrix *A, HYPRE_Int newoff, HYPRE_Int *found, 
                              hypre_ParCSRCommPkg **extend_comm_pkg)

{
   

   HYPRE_Int			num_sends;
   HYPRE_Int			*send_procs;
   HYPRE_Int			*send_map_starts;
   HYPRE_Int			*send_map_elmts;
 
   HYPRE_Int			num_recvs;
   HYPRE_Int			*recv_procs;
   HYPRE_Int			*recv_vec_starts;

   hypre_ParCSRCommPkg	*new_comm_pkg;

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
  /* use found instead of col_map_offd in A, and newoff instead 
      of num_cols_offd*/

#ifdef HYPRE_NO_GLOBAL_PARTITION

   HYPRE_Int        row_start=0, row_end=0, col_start = 0, col_end = 0;
   HYPRE_Int        global_num_cols;
   hypre_IJAssumedPart   *apart;
   
   hypre_ParCSRMatrixGetLocalRange( A,
                                    &row_start, &row_end ,
                                    &col_start, &col_end );
   

   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A); 

   /* Create the assumed partition */
   if  (hypre_ParCSRMatrixAssumedPartition(A) == NULL)
   {
      hypre_ParCSRMatrixCreateAssumedPartition(A);
   }

   apart = hypre_ParCSRMatrixAssumedPartition(A);
   
   hypre_NewCommPkgCreate_core( comm, found, first_col_diag, 
                                col_start, col_end, 
                                newoff, global_num_cols,
                                &num_recvs, &recv_procs, &recv_vec_starts,
                                &num_sends, &send_procs, &send_map_starts, 
                                &send_map_elmts, apart);

#else   
   HYPRE_Int  *col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_Int	num_cols_diag = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   
   hypre_MatvecCommPkgCreate_core
      (
         comm, found, first_col_diag, col_starts,
         num_cols_diag, newoff,
         first_col_diag, found,
         1,
         &num_recvs, &recv_procs, &recv_vec_starts,
         &num_sends, &send_procs, &send_map_starts,
         &send_map_elmts
         );

#endif

   new_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(new_comm_pkg) = comm;

   hypre_ParCSRCommPkgNumRecvs(new_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(new_comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgNumSends(new_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(new_comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg) = send_map_elmts;



   *extend_comm_pkg = new_comm_pkg;
   

   return hypre_error_flag;
   
}


/* sort for non-ordered arrays */
HYPRE_Int hypre_ssort(HYPRE_Int *data, HYPRE_Int n)
{
  HYPRE_Int i,si;               
  HYPRE_Int change = 0;
  
  if(n > 0)
    for(i = n-1; i > 0; i--){
      si = hypre_index_of_minimum(data,i+1);
      if(i != si)
      {
	hypre_swap_int(data, i, si);
	change = 1;
      }
    }                                                                       
  return change;
}

/* Auxilary function for hypre_ssort */
HYPRE_Int hypre_index_of_minimum(HYPRE_Int *data, HYPRE_Int n)
{
  HYPRE_Int answer;
  HYPRE_Int i;
                                                                               
  answer = 0;
  for(i = 1; i < n; i++)
    if(data[answer] < data[i])
      answer = i;
                                                                               
  return answer;
}
                                                                               
void hypre_swap_int(HYPRE_Int *data, HYPRE_Int a, HYPRE_Int b)
{
  HYPRE_Int temp;
                                                                               
  temp = data[a];
  data[a] = data[b];
  data[b] = temp;

  return;
}

/* Initialize CF_marker_offd, CF_marker, P_marker, P_marker_offd, tmp */
void hypre_initialize_vecs(HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc, HYPRE_Int *offd_ftc, 
		     HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF)
{
  HYPRE_Int i;

  /* Quicker initialization */
  if(offd_n < diag_n)
  {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
    for(i = 0; i < offd_n; i++)
    {
      diag_ftc[i] = -1;
      offd_ftc[i] = -1;
      tmp_CF[i] = -1;
      if(diag_pm != NULL)
      {  diag_pm[i] = -1; }
      if(offd_pm != NULL)
      {  offd_pm[i] = -1;}
    }
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
    for(i = offd_n; i < diag_n; i++)
    { 
      diag_ftc[i] = -1;
      if(diag_pm != NULL)
      {  diag_pm[i] = -1; }
    }
  }
  else
  {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
    for(i = 0; i < diag_n; i++)
    {
      diag_ftc[i] = -1;
      offd_ftc[i] = -1;
      tmp_CF[i] = -1;
      if(diag_pm != NULL)
      {  diag_pm[i] = -1;}
      if(offd_pm != NULL)
      {  offd_pm[i] = -1;}
    }
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
    for(i = diag_n; i < offd_n; i++)
    { 
      offd_ftc[i] = -1;
      tmp_CF[i] = -1;
      if(offd_pm != NULL)
      {  offd_pm[i] = -1;}
    }
  }
  return;
}

/* Find nodes that are offd and are not contained in original offd
 * (neighbors of neighbors) */
static HYPRE_Int hypre_new_offd_nodes(HYPRE_Int **found, HYPRE_Int num_cols_A_offd, HYPRE_Int *A_ext_i, HYPRE_Int *A_ext_j, 
		   HYPRE_Int num_cols_S_offd, HYPRE_Int *col_map_offd, HYPRE_Int col_1, 
		   HYPRE_Int col_n, HYPRE_Int *Sop_i, HYPRE_Int *Sop_j,
		   HYPRE_Int *CF_marker_offd)
{
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif

  HYPRE_Int i, i1, j, kk, k1;
  HYPRE_Int got_loc, loc_col;

  /*HYPRE_Int min;*/
  HYPRE_Int newoff = 0;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  hypre_UnorderedIntMap col_map_offd_inverse;
  hypre_UnorderedIntMapCreate(&col_map_offd_inverse, 2*num_cols_A_offd, 16*hypre_NumThreads());

#pragma omp parallel for HYPRE_SMP_SCHEDULE
  for (i = 0; i < num_cols_A_offd; i++)
  {
     hypre_UnorderedIntMapPutIfAbsent(&col_map_offd_inverse, col_map_offd[i], i);
  }

  /* Find nodes that will be added to the off diag list */ 
  HYPRE_Int size_offP = A_ext_i[num_cols_A_offd];
  hypre_UnorderedIntSet set;
  hypre_UnorderedIntSetCreate(&set, size_offP, 16*hypre_NumThreads());

#pragma omp parallel private(i,j,i1)
  {
#pragma omp for HYPRE_SMP_SCHEDULE
    for (i = 0; i < num_cols_A_offd; i++)
    {
     if (CF_marker_offd[i] < 0)
     {
      for (j = A_ext_i[i]; j < A_ext_i[i+1]; j++)
      {
        i1 = A_ext_j[j];
        if(i1 < col_1 || i1 >= col_n)
        {
          if (!hypre_UnorderedIntSetContains(&set, i1))
          {
            HYPRE_Int k = hypre_UnorderedIntMapGet(&col_map_offd_inverse, i1);
            if (-1 == k)
            {
               hypre_UnorderedIntSetPut(&set, i1);
            }
            else
            {
               A_ext_j[j] = -k - 1;
            }
          }
        }
      }
      for (j = Sop_i[i]; j < Sop_i[i+1]; j++)
      {
        i1 = Sop_j[j];
        if(i1 < col_1 || i1 >= col_n)
        {
          if (!hypre_UnorderedIntSetContains(&set, i1))
          {
            Sop_j[j] = -hypre_UnorderedIntMapGet(&col_map_offd_inverse, i1) - 1;
          }
        }
      }
     } /* CF_marker_offd[i] < 0 */
    } /* for each row */
  } /* omp parallel */

  hypre_UnorderedIntMapDestroy(&col_map_offd_inverse);
  HYPRE_Int *tmp_found = hypre_UnorderedIntSetCopyToArray(&set, &newoff);
  hypre_UnorderedIntSetDestroy(&set);

  /* Put found in monotone increasing order */
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_MERGE] -= hypre_MPI_Wtime();
#endif

  hypre_UnorderedIntMap tmp_found_inverse;
  if (newoff > 0)
  {
    hypre_sort_and_create_inverse_map(tmp_found, newoff, &tmp_found, &tmp_found_inverse);
  }

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_MERGE] += hypre_MPI_Wtime();
#endif

  /* Set column indices for Sop and A_ext such that offd nodes are
   * negatively indexed */
#pragma omp parallel for private(kk,k1,got_loc,loc_col) HYPRE_SMP_SCHEDULE
  for(i = 0; i < num_cols_A_offd; i++)
  {
   if (CF_marker_offd[i] < 0)
   {
     for(kk = Sop_i[i]; kk < Sop_i[i+1]; kk++)
     {
       k1 = Sop_j[kk];
       if(k1 > -1 && (k1 < col_1 || k1 >= col_n))
       { 
         got_loc = hypre_UnorderedIntMapGet(&tmp_found_inverse, k1);
         loc_col = got_loc + num_cols_A_offd;
         Sop_j[kk] = -loc_col - 1;
       }
     }
     for (kk = A_ext_i[i]; kk < A_ext_i[i+1]; kk++)
     {
       k1 = A_ext_j[kk];
       if(k1 > -1 && (k1 < col_1 || k1 >= col_n))
       {
         got_loc = hypre_UnorderedIntMapGet(&tmp_found_inverse, k1);
         loc_col = got_loc + num_cols_A_offd;
         A_ext_j[kk] = -loc_col - 1;
       }
     }
   }
  }
  if (newoff)
  {
    hypre_UnorderedIntMapDestroy(&tmp_found_inverse);
  }
#else /* !HYPRE_CONCURRENT_HOPSCOTCH */
  HYPRE_Int size_offP;

  HYPRE_Int *tmp_found;
  HYPRE_Int min;
  HYPRE_Int ifound;

  size_offP = A_ext_i[num_cols_A_offd]+Sop_i[num_cols_A_offd];
  tmp_found = hypre_CTAlloc(HYPRE_Int, size_offP);

  /* Find nodes that will be added to the off diag list */ 
  for (i = 0; i < num_cols_A_offd; i++)
  {
   if (CF_marker_offd[i] < 0)
   {
    for (j = A_ext_i[i]; j < A_ext_i[i+1]; j++)
    {
      i1 = A_ext_j[j];
      if(i1 < col_1 || i1 >= col_n)
      {
	  ifound = hypre_BinarySearch(col_map_offd,i1,num_cols_A_offd);
	  if(ifound == -1)
	  {
	      tmp_found[newoff]=i1;
	      newoff++;
	  }
	  else
	  {
	      A_ext_j[j] = -ifound-1;
	  }
      }
    }
    for (j = Sop_i[i]; j < Sop_i[i+1]; j++)
    {
      i1 = Sop_j[j];
      if(i1 < col_1 || i1 >= col_n)
      {
	  ifound = hypre_BinarySearch(col_map_offd,i1,num_cols_A_offd);
	  if(ifound == -1)
	  {
	      tmp_found[newoff]=i1;
	      newoff++;
	  }
	  else
	  {
	      Sop_j[j] = -ifound-1;
	  }
      }
    }
   }
  }
  /* Put found in monotone increasing order */
  if (newoff > 0)
  {
     hypre_qsort0(tmp_found,0,newoff-1);
     ifound = tmp_found[0];
     min = 1;
     for (i=1; i < newoff; i++)
     {
       if (tmp_found[i] > ifound)
       {
          ifound = tmp_found[i];
          tmp_found[min++] = ifound;
       }
     }
     newoff = min;
  }

  /* Set column indices for Sop and A_ext such that offd nodes are
   * negatively indexed */
  for(i = 0; i < num_cols_A_offd; i++)
  {
   if (CF_marker_offd[i] < 0)
   {
     for(kk = Sop_i[i]; kk < Sop_i[i+1]; kk++)
     {
       k1 = Sop_j[kk];
       if(k1 > -1 && (k1 < col_1 || k1 >= col_n))
       { 
	 got_loc = hypre_BinarySearch(tmp_found,k1,newoff);
	 if(got_loc > -1)
	   loc_col = got_loc + num_cols_A_offd;
	 Sop_j[kk] = -loc_col - 1;
       }
     }
     for (kk = A_ext_i[i]; kk < A_ext_i[i+1]; kk++)
     {
       k1 = A_ext_j[kk];
       if(k1 > -1 && (k1 < col_1 || k1 >= col_n))
       {
	 got_loc = hypre_BinarySearch(tmp_found,k1,newoff);
	 loc_col = got_loc + num_cols_A_offd;
	 A_ext_j[kk] = -loc_col - 1;
       }
     }
   }
  }
#endif /* !HYPRE_CONCURRENT_HOPSCOTCH */

  *found = tmp_found;

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif
 
  return newoff;
}

HYPRE_Int hypre_exchange_marker(hypre_ParCSRCommPkg *comm_pkg,
                          HYPRE_Int *IN_marker, 
                          HYPRE_Int *OUT_marker)
{   
  HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
  HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
  HYPRE_Int *int_buf_data = hypre_CTAlloc(HYPRE_Int, end);

  HYPRE_Int i;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
  for (i = begin; i < end; ++i) {
     int_buf_data[i - begin] =
           IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
  }
   
  hypre_ParCSRCommHandle *comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
					      OUT_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  hypre_TFree(int_buf_data);
    
  return hypre_error_flag;
} 

HYPRE_Int hypre_exchange_interp_data(
    HYPRE_Int **CF_marker_offd,
    HYPRE_Int **dof_func_offd,
    hypre_CSRMatrix **A_ext,
    HYPRE_Int *full_off_procNodes,
    hypre_CSRMatrix **Sop,
    hypre_ParCSRCommPkg **extend_comm_pkg,
    hypre_ParCSRMatrix *A, 
    HYPRE_Int *CF_marker,
    hypre_ParCSRMatrix *S,
    HYPRE_Int num_functions,
    HYPRE_Int *dof_func,
    HYPRE_Int skip_fine_or_same_sign) // skip_fine_or_same_sign if we want to skip fine points in S and nnz with the same sign as diagonal in A
{
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] -= hypre_MPI_Wtime();
#endif

  hypre_ParCSRCommPkg   *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A); 
  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
  HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
  HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  HYPRE_Int              col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
  HYPRE_Int              local_numrows = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_Int              col_n = col_1 + local_numrows;
  HYPRE_Int             *found = NULL;

  /*----------------------------------------------------------------------
   * Get the off processors rows for A and S, associated with columns in 
   * A_offd and S_offd.
   *---------------------------------------------------------------------*/
  *CF_marker_offd = hypre_TAlloc(HYPRE_Int, num_cols_A_offd);
  hypre_exchange_marker(comm_pkg, CF_marker, *CF_marker_offd);

  hypre_ParCSRCommHandle *comm_handle_a_idx, *comm_handle_a_data;
  *A_ext         = hypre_ParCSRMatrixExtractBExt_Overlap(A,A,1,&comm_handle_a_idx,&comm_handle_a_data,CF_marker,*CF_marker_offd,skip_fine_or_same_sign,skip_fine_or_same_sign);
  HYPRE_Int *A_ext_i        = hypre_CSRMatrixI(*A_ext);
  HYPRE_Int *A_ext_j        = hypre_CSRMatrixJ(*A_ext);
  HYPRE_Int  A_ext_rows     = hypre_CSRMatrixNumRows(*A_ext);

  hypre_ParCSRCommHandle *comm_handle_s_idx;
  *Sop           = hypre_ParCSRMatrixExtractBExt_Overlap(S,A,0,&comm_handle_s_idx,NULL,CF_marker,*CF_marker_offd,skip_fine_or_same_sign,0);
  HYPRE_Int *Sop_i          = hypre_CSRMatrixI(*Sop);
  HYPRE_Int *Sop_j          = hypre_CSRMatrixJ(*Sop);
  HYPRE_Int  Soprows        = hypre_CSRMatrixNumRows(*Sop);

  HYPRE_Int *send_idx = (HYPRE_Int *)comm_handle_s_idx->send_data;
  hypre_ParCSRCommHandleDestroy(comm_handle_s_idx);
  hypre_TFree(send_idx);

  send_idx = (HYPRE_Int *)comm_handle_a_idx->send_data;
  hypre_ParCSRCommHandleDestroy(comm_handle_a_idx);
  hypre_TFree(send_idx);

  /* Find nodes that are neighbors of neighbors, not found in offd */
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] += hypre_MPI_Wtime();
#endif
  HYPRE_Int newoff = hypre_new_offd_nodes(&found, A_ext_rows, A_ext_i, A_ext_j, 
      Soprows, col_map_offd, col_1, col_n, 
      Sop_i, Sop_j, *CF_marker_offd);
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] -= hypre_MPI_Wtime();
#endif
  if(newoff >= 0)
    *full_off_procNodes = newoff + num_cols_A_offd;
  else
  {
    return hypre_error_flag;
  }

  /* Possibly add new points and new processors to the comm_pkg, all
   * processors need new_comm_pkg */

  /* AHB - create a new comm package just for extended info -
     this will work better with the assumed partition*/
  hypre_ParCSRFindExtendCommPkg(A, newoff, found, 
      extend_comm_pkg);

  *CF_marker_offd = hypre_TReAlloc(*CF_marker_offd, HYPRE_Int, *full_off_procNodes);
  hypre_exchange_marker(*extend_comm_pkg, CF_marker, *CF_marker_offd + A_ext_rows);

  if(num_functions > 1)
  {
    if (*full_off_procNodes > 0)
      *dof_func_offd = hypre_CTAlloc(HYPRE_Int, *full_off_procNodes);

    hypre_alt_insert_new_nodes(comm_pkg, *extend_comm_pkg, dof_func, 
        *full_off_procNodes, *dof_func_offd);
  }

  hypre_TFree(found);

  HYPRE_Real *send_data = (HYPRE_Real *)comm_handle_a_data->send_data;
  hypre_ParCSRCommHandleDestroy(comm_handle_a_data);
  hypre_TFree(send_data);

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] += hypre_MPI_Wtime();
#endif

  return hypre_error_flag;
}

void hypre_build_interp_colmap(hypre_ParCSRMatrix *P, HYPRE_Int full_off_procNodes, HYPRE_Int *tmp_CF_marker_offd, HYPRE_Int *fine_to_coarse_offd)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int i, index;

   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(P->diag);

   HYPRE_Int P_offd_size = P->offd->i[n_fine];
   HYPRE_Int *P_offd_j = P->offd->j;
   HYPRE_Int *col_map_offd_P = NULL;

   HYPRE_Int *P_marker = NULL;

   if (full_off_procNodes)
      P_marker = hypre_TAlloc(HYPRE_Int, full_off_procNodes);
   
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i=0; i < full_off_procNodes; i++)
     P_marker[i] = 0;
 
#ifdef HYPRE_CONCURRENT_HOPSCOTCH

   /* These two loops set P_marker[i] to 1 if it appears in P_offd_j and if
    * tmp_CF_marker_offd has i marked. num_cols_P_offd is then set to the
    * total number of times P_marker is set */
#pragma omp parallel for private(i,index) HYPRE_SMP_SCHEDULE
   for (i=0; i < P_offd_size; i++)
   {
     index = P_offd_j[i];
     if(tmp_CF_marker_offd[index] >= 0)
     {  P_marker[index] = 1; }
   }

   HYPRE_Int prefix_sum_workspace[hypre_NumThreads() + 1];
   HYPRE_Int num_cols_P_offd = 0;

#pragma omp parallel private(i)
   {
     HYPRE_Int i_begin, i_end;
     hypre_GetSimpleThreadPartition(&i_begin, &i_end, full_off_procNodes);

     HYPRE_Int local_num_cols_P_offd = 0;
     for (i = i_begin; i < i_end; i++)
     {
       if (P_marker[i] == 1) local_num_cols_P_offd++;
     }

     hypre_prefix_sum(&local_num_cols_P_offd, &num_cols_P_offd, prefix_sum_workspace);

#pragma omp master
     {
       if (num_cols_P_offd)
         col_map_offd_P = hypre_TAlloc(HYPRE_Int, num_cols_P_offd);
     }
#pragma omp barrier

     for (i = i_begin; i < i_end; i++)
     {
       if (P_marker[i] == 1)
       {
          col_map_offd_P[local_num_cols_P_offd++] = fine_to_coarse_offd[i];
       }
     }
   }

   hypre_UnorderedIntMap col_map_offd_P_inverse;
   hypre_sort_and_create_inverse_map(col_map_offd_P, num_cols_P_offd, &col_map_offd_P, &col_map_offd_P_inverse);

   // find old idx -> new idx map
#pragma omp parallel for
   for (i = 0; i < full_off_procNodes; i++)
     P_marker[i] = hypre_UnorderedIntMapGet(&col_map_offd_P_inverse, fine_to_coarse_offd[i]);

   if (num_cols_P_offd)
   {
     hypre_UnorderedIntMapDestroy(&col_map_offd_P_inverse);
   }
#pragma omp parallel for
   for(i = 0; i < P_offd_size; i++)
     P_offd_j[i] = P_marker[P_offd_j[i]];

#else /* HYPRE_CONCURRENT_HOPSCOTCH */
     HYPRE_Int num_cols_P_offd = 0;
     HYPRE_Int j;
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
     
     if (num_cols_P_offd)
	col_map_offd_P = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd);
     
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
     if(hypre_ssort(col_map_offd_P,num_cols_P_offd))
     {
       for(i = 0; i < P_offd_size; i++)
	 for(j = 0; j < num_cols_P_offd; j++)
	   if(P_marker[P_offd_j[i]] == col_map_offd_P[j])
	   {
	     P_offd_j[i] = j;
	     j = num_cols_P_offd;
	   }
     }
#endif /* HYPRE_CONCURRENT_HOPSCOTCH */

   hypre_TFree(P_marker); 

   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P->offd) = num_cols_P_offd;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif
}
