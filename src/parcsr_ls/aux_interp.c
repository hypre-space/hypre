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

/*---------------------------------------------------------------------------
 * Auxilary routines for the long range interpolation methods.
 *  Implemented: "standard", "extended", "multipass", "FF"
 *--------------------------------------------------------------------------*/
/* AHB 11/06: Modification of the above original - takes two
   communication packages and inserts nodes to position expected for
   OUT_marker
  
   offd nodes from comm_pkg take up first chunk of CF_marker_offd, offd 
   nodes from extend_comm_pkg take up the second chunk 0f CF_marker_offd. */



HYPRE_Int alt_insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, 
                          hypre_ParCSRCommPkg *extend_comm_pkg,
                          HYPRE_Int *IN_marker, 
                          HYPRE_Int full_off_procNodes,
                          HYPRE_Int *OUT_marker)
{   
  hypre_ParCSRCommHandle  *comm_handle;

  HYPRE_Int i, j, start, index, shift;

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
  
  for (i = 0; i < num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); 
	 j++)
      int_buf_data[index++] 
	= IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
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

  for (i = 0; i < e_num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, i+1); 
	 j++)
       int_buf_data[index++] 
	= IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg,j)];
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
      si = index_of_minimum(data,i+1);
      if(i != si)
      {
	swap_int(data, i, si);
	change = 1;
      }
    }                                                                       
  return change;
}

/* Auxilary function for hypre_ssort */
HYPRE_Int index_of_minimum(HYPRE_Int *data, HYPRE_Int n)
{
  HYPRE_Int answer;
  HYPRE_Int i;
                                                                               
  answer = 0;
  for(i = 1; i < n; i++)
    if(data[answer] < data[i])
      answer = i;
                                                                               
  return answer;
}
                                                                               
void swap_int(HYPRE_Int *data, HYPRE_Int a, HYPRE_Int b)
{
  HYPRE_Int temp;
                                                                               
  temp = data[a];
  data[a] = data[b];
  data[b] = temp;

  return;
}

/* Initialize CF_marker_offd, CF_marker, P_marker, P_marker_offd, tmp */
void initialize_vecs(HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc, HYPRE_Int *offd_ftc, 
		     HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF)
{
  HYPRE_Int i;

  /* Quicker initialization */
  if(offd_n < diag_n)
  {
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
    for(i = offd_n; i < diag_n; i++)
    { 
      diag_ftc[i] = -1;
      if(diag_pm != NULL)
      {  diag_pm[i] = -1; }
    }
  }
  else
  {
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
HYPRE_Int new_offd_nodes(HYPRE_Int **found, HYPRE_Int num_cols_A_offd, HYPRE_Int *A_ext_i, HYPRE_Int *A_ext_j, 
		   HYPRE_Int num_cols_S_offd, HYPRE_Int *col_map_offd, HYPRE_Int col_1, 
		   HYPRE_Int col_n, HYPRE_Int *Sop_i, HYPRE_Int *Sop_j,
		   HYPRE_Int *CF_marker, hypre_ParCSRCommPkg *comm_pkg)
{
  HYPRE_Int i, i1, ii, j, ifound, kk, k1;
  HYPRE_Int got_loc, loc_col;

  HYPRE_Int min;

  HYPRE_Int size_offP;

  HYPRE_Int *tmp_found;
  HYPRE_Int *CF_marker_offd = NULL;
  HYPRE_Int *int_buf_data;
  HYPRE_Int newoff = 0;
  HYPRE_Int full_off_procNodes = 0;
  hypre_ParCSRCommHandle *comm_handle;
                                                                                                                                         
  CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);
  int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                hypre_ParCSRCommPkgNumSends(comm_pkg)));
  ii = 0;
  for (i=0; i < hypre_ParCSRCommPkgNumSends(comm_pkg); i++)
  {
      for (j=hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
                j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        int_buf_data[ii++]
          = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
  }
  comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,int_buf_data,
        CF_marker_offd);
  hypre_ParCSRCommHandleDestroy(comm_handle);
  hypre_TFree(int_buf_data);

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
     qsort0(tmp_found,0,newoff-1);
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

  full_off_procNodes = newoff + num_cols_A_offd;
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


  hypre_TFree(CF_marker_offd);
  

  *found = tmp_found;
 


  return newoff;
}
