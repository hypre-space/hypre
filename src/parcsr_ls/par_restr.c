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
#include "_hypre_blas.h"
#include "_hypre_lapack.h"

#define AIR_DEBUG 0

HYPRE_Int
hypre_BoomerAMGBuildRestrAIR( hypre_ParCSRMatrix   *A,
                              HYPRE_Int            *CF_marker,
                              hypre_ParCSRMatrix   *S,
                              HYPRE_Int            *num_cpts_global,
                              HYPRE_Int             num_functions,
                              HYPRE_Int            *dof_func,
                              HYPRE_Int             debug_flag,
                              HYPRE_Real            trunc_factor,
                              HYPRE_Int             max_elmts,
                              HYPRE_Int            *col_offd_S_to_A,
                              hypre_ParCSRMatrix  **R_ptr) {
   
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   /* diag part of A */
   hypre_CSRMatrix *A_diag      = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i    = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j    = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix *A_offd      = hypre_ParCSRMatrixOffd(A);   
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i    = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j    = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int       *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);
   /* Strength matrix S */
   /* diag part of S */
   hypre_CSRMatrix *S_diag   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);
   /* off-diag part of S */
   hypre_CSRMatrix *S_offd   = hypre_ParCSRMatrixOffd(S);   
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);
   /* Restriction matrix R */
   hypre_ParCSRMatrix *R;
   /* csr's */
   hypre_CSRMatrix *R_diag;
   hypre_CSRMatrix *R_offd;
   /* arrays */
   HYPRE_Real      *R_diag_data;
   HYPRE_Int       *R_diag_i;
   HYPRE_Int       *R_diag_j;
   HYPRE_Real      *R_offd_data;
   HYPRE_Int       *R_offd_i;
   HYPRE_Int       *R_offd_j;
   HYPRE_Int       *col_map_offd_R;
   /* CF marker off-diag part */
   HYPRE_Int       *CF_marker_offd = NULL;
   /* func type off-diag part */
   HYPRE_Int       *dof_func_offd  = NULL;
   /* ghost rows */
   hypre_CSRMatrix *A_ext      = NULL;
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i    = NULL;
   HYPRE_Int       *A_ext_j    = NULL;
   
   HYPRE_Int        i, j, k, i1, k1, k2, rr, cc, ic, index, start, 
                    local_max_size, local_size, num_cols_offd_R;

   /* LAPACK */
   HYPRE_Real *DAi, *Dbi;
#if AIR_DEBUG
   HYPRE_Real *TMPA, *TMPb, *TMPd;
#endif
   HYPRE_Int *Ipi, lapack_info, ione = 1;
   char charT = 'T';

   HYPRE_Int my_id, num_procs;
   HYPRE_Int total_global_cpts/*, my_first_cpt*/;
   HYPRE_Int nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   HYPRE_Int *marker_diag, *marker_offd;
   HYPRE_Int num_sends, *int_buf_data;
   /* local size, local num of C points */
   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int n_cpts = 0;
   /* my first column range */
   /* XXX is this also right?
   HYPRE_Int col_start = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int col_end   = hypre_ParCSRMatrixLastColDiag(A);
   */
   HYPRE_Int col_start = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int col_end   = col_start + n_fine;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);   
   hypre_MPI_Comm_rank(comm, &my_id);

   /*-------------- global number of C points and my start position */
#ifdef HYPRE_NO_GLOBAL_PARTITION
   /*my_first_cpt = num_cpts_global[0];*/
   if (my_id == (num_procs -1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
   /*my_first_cpt = num_cpts_global[my_id];*/
   total_global_cpts = num_cpts_global[num_procs];
#endif

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   /* CF marker for the off-diag columns */
   if (num_cols_A_offd)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd,HYPRE_MEMORY_HOST);
   }
   /* function type indicator for the off-diag columns */
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd,HYPRE_MEMORY_HOST);
   }
   /* if CommPkg of A is not present, create it */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }
   /* number of sends to do (number of procs) */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* send buffer, of size send_map_starts[num_sends]), 
    * i.e., number of entries to send */
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),HYPRE_MEMORY_HOST);
   /* copy CF markers of elements to send to buffer 
    * RL: why copy them with two for loops? Why not just loop through all in one */
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* loop through all elems to send_proc[i] */
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         /* CF marker of send_map_elemts[j] */
         int_buf_data[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
   /* destroy the handle to finish communication */
   hypre_ParCSRCommHandleDestroy(comm_handle);   
   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, dof_func_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);   
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine the nnz of R and the max local size
    *-----------------------------------------------------------------------*/
   /* nnz in diag and offd parts */
   cnt_diag = 0;
   cnt_offd = 0;
   /* maximum size of local system: will allocate space of this size */
   local_max_size = 0;
   for (i = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }
      /* local number of C-pts */
      n_cpts ++;
      /* If i is a C-point, the restriction is from the F-points that
       * strongly influence i */
      local_size = 0;
      /* loop through the diag part of S */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            cnt_diag ++;
            local_size ++;
         }
      }
      /* if parallel, loop through the offd part */
      if (num_procs > 1)
      {
         /* use this mapping to have offd indices of A */
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            if (CF_marker_offd[i1] < 0)
            {
               cnt_offd ++;
               local_size ++;
            }
         }
      }
      /* keep ths max size */
      local_max_size = hypre_max(local_max_size, local_size);
   }

   /* this is because of the indentity matrix in C part
    * each C-pt has an entry 1.0 */
   cnt_diag += n_cpts;

   nnz_diag = cnt_diag;
   nnz_offd = cnt_offd;

   /*------------- allocate arrays */
   R_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_cpts+1,HYPRE_MEMORY_HOST);
   R_diag_j    = hypre_CTAlloc(HYPRE_Int,  nnz_diag,HYPRE_MEMORY_HOST);
   R_diag_data = hypre_CTAlloc(HYPRE_Real, nnz_diag,HYPRE_MEMORY_HOST);

   /* not in ``if num_procs > 1'', 
    * allocation needed even for empty CSR */
   R_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_cpts+1,HYPRE_MEMORY_HOST);
   R_offd_j    = hypre_CTAlloc(HYPRE_Int,  nnz_offd,HYPRE_MEMORY_HOST);
   R_offd_data = hypre_CTAlloc(HYPRE_Real, nnz_offd,HYPRE_MEMORY_HOST);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;
 
   /*----------------------------------------       .-.
    * Get the GHOST rows of A,                     (o o) boo!
    * i.e., adjacent rows to this proc             | O \
    * whose row indices are in A->col_map_offd      \   \
    *-----------------------------------------       `~~~'  */
   /* external rows of A that are needed for perform A multiplication,
    * the last arg means need data 
    * the number of rows is num_cols_A_offd */
   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_i    = hypre_CSRMatrixI(A_ext);
      A_ext_j    = hypre_CSRMatrixJ(A_ext);
      A_ext_data = hypre_CSRMatrixData(A_ext);
   }

   /* marker array: if this point is i's strong F neighbors
    *             >=  0: yes, and is the local dense id 
    *             == -1: no */
   marker_diag = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      marker_diag[i] = -1;
   }
   marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   for (i = 0; i< num_cols_A_offd; i++)
   {
      marker_offd[i] = -1;
   }

   /* the local matrix and rhs (dense) 
    * column-major as always by BLAS/LAPACK */
   /* matrix */
   DAi = hypre_CTAlloc(HYPRE_Real, local_max_size * local_max_size, HYPRE_MEMORY_HOST);
   /* rhs */
   Dbi = hypre_CTAlloc(HYPRE_Real, local_max_size, HYPRE_MEMORY_HOST);
   /* pivot */
   Ipi = hypre_CTAlloc(HYPRE_Int, local_max_size, HYPRE_MEMORY_HOST);

#if AIR_DEBUG
   /* FOR DEBUG */
   TMPA = hypre_CTAlloc(HYPRE_Real, local_max_size * local_max_size, HYPRE_MEMORY_HOST);
   TMPb = hypre_CTAlloc(HYPRE_Real, local_max_size, HYPRE_MEMORY_HOST);
   TMPd = hypre_CTAlloc(HYPRE_Real, local_max_size, HYPRE_MEMORY_HOST);
#endif

   /*-----------------------------------------------------------------------
    *  Second Pass: Populate R
    *-----------------------------------------------------------------------*/
   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* size of Ai, bi */
      local_size = 0;
      
      /* If i is a C-point, build the restriction, from the F-points that
       * strongly influence i
       * Access S for the first time, mark the points we want */
      /* 1: loop through the diag part of S */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            hypre_assert(marker_diag[i1] == -1);
            /* mark this point */
            marker_diag[i1] = local_size ++;
         }
      }
      /* 2: if parallel, loop through the offd part */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               hypre_assert(marker_offd[i1] == -1);
               /* mark this point */
               marker_offd[i1] = local_size ++;
            }
         }
      }

      /* DEBUG FOR local_size == 0 */
      /*
      if (local_size == 0)
      {
         printf("my_id %d:  ", my_id);
         for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
         {
            i1 = S_diag_j[j];
            printf("%d[d, %d] ", i1, CF_marker[i1]);
         }
         printf("\n");
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            printf("%d[o, %d] ", i1, CF_marker_offd[i1]);
         }

         printf("\n");
         
         exit(0);
      }
      */

      /* Second, copy values to local system: Ai and bi from A */
      /* now we have marked all rows/cols we want. next we extract the entries 
       * we need from these rows and put them in Ai and bi*/

      /* clear DAi and bi */
      memset(DAi, 0, local_size * local_size * sizeof(HYPRE_Real));
      memset(Dbi, 0, local_size * sizeof(HYPRE_Real));

      /* we will populate Ai, bi row-by-row
       * rr is the local dense matrix row counter */
      rr = 0;
      /* 1. diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         /* row i1 */
         i1 = S_diag_j[j];
         /* i1 is an F point */
         if (CF_marker[i1] < 0)
         {
            /* go through row i1 of A: a local row */
            /* diag part of row i1 */
            for (k = A_diag_i[i1]; k < A_diag_i[i1+1]; k++)
            {
               k1 = A_diag_j[k];
               /* if this col is marked with its local dense id */
               if ((cc = marker_diag[k1]) >= 0)
               {
                  hypre_assert(CF_marker[k1] < 0);
                  /* copy the value */
                  /* rr and cc: local dense ids */
                  DAi[rr + cc * local_size] = A_diag_data[k];
               }
            }
            /* if parallel, offd part of row i1 */
            if (num_procs > 1)
            {
               for (k = A_offd_i[i1]; k < A_offd_i[i1+1]; k++)
               {
                  k1 = A_offd_j[k];
                  /* if this col is marked with its local dense id */
                  if ((cc = marker_offd[k1]) >= 0)
                  {
                     hypre_assert(CF_marker_offd[k1] < 0);
                     /* copy the value */
                     /* rr and cc: local dense ids */
                     DAi[rr + cc * local_size] = A_offd_data[k];
                  }
               }
            }
            /* done with row i1 */
            rr++;
         }
      } /* for (j=...), diag part of row i done */

      /* 2. if parallel, offd part of row i. The corresponding rows are
       *    in matrix A_ext */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            /* row i1: use this mapping to have offd indices of A */
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            /* if this is an F point */
            if (CF_marker_offd[i1] < 0)
            {
               /* loop through row i1 of A_ext, a global CSR matrix */
               for (k = A_ext_i[i1]; k < A_ext_i[i1+1]; k++)
               {
                  /* k1 is a global index! */
                  k1 = A_ext_j[k];
                  if (k1 >= col_start && k1 < col_end)
                  {
                     /* k1 is in the diag part, adjust to local index */
                     k1 -= col_start;
                     /* if this col is marked with its local dense id*/
                     if ((cc = marker_diag[k1]) >= 0)
                     {
                        hypre_assert(CF_marker[k1] < 0);
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        DAi[rr + cc * local_size] = A_ext_data[k];
                     }
                  }
                  else
                  {
                     /* k1 is in the offd part
                      * search k1 in A->col_map_offd */
                     k2 = hypre_BinarySearch(col_map_offd_A, k1, num_cols_A_offd);
                     /* if found, k2 is the position of column id k1 in col_map_offd */
                     if (k2 > -1)
                     {
                        /* if this col is marked with its local dense id */
                        if ((cc = marker_offd[k2]) >= 0)
                        {
                           hypre_assert(CF_marker_offd[k2] < 0);
                           /* copy the value */
                           /* rr and cc: local dense ids */
                           DAi[rr + cc * local_size] = A_ext_data[k];
                        }
                     }
                  }
               }
               /* done with row i1 */
               rr++;
            }
         }
      }

      hypre_assert(rr == local_size);

      /* assemble rhs bi: entries from row i of A */
      rr = 0;
      /* diag part */
      for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
      {
         i1 = A_diag_j[j];
         if ((cc = marker_diag[i1]) >= 0)
         {
            /* this should be true but not very important
             * what does it say is that eqn order == unknown order
             * this is true, since order in A is preserved in S */
            hypre_assert(rr == cc);
            /* Note the sign change */
            Dbi[cc] = -A_diag_data[j];
            rr++;
         }
      }
      /* if parallel, offd part */
      if (num_procs > 1)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
         {
            i1 = A_offd_j[j];
            if ((cc = marker_offd[i1]) >= 0)
            {
               /* this should be true but not very important
                * what does it say is that eqn order == unknown order
                * this is true, since order in A is preserved in S */
               hypre_assert(rr == cc);
               /* Note the sign change */
               Dbi[cc] = -A_offd_data[j];
               rr++;
            }
         }
      }
      hypre_assert(rr == local_size);

      if (local_size > 0)
      {
         /* we have Ai and bi build
          * solve the linear system by LAPACK : LU factorization */
#if AIR_DEBUG
         memcpy(TMPA, DAi, local_size*local_size*sizeof(HYPRE_Real));
         memcpy(TMPb, Dbi, local_size*sizeof(HYPRE_Real)); 
#endif

         hypre_dgetrf(&local_size, &local_size, DAi, &local_size, Ipi,
               &lapack_info);

         hypre_assert(lapack_info == 0);

         if (lapack_info == 0)
         {
            /* solve A_i^T x_i = b_i,
             * solution is saved in b_i on return */
            hypre_dgetrs(&charT, &local_size, &ione, DAi, &local_size,
                  Ipi, Dbi, &local_size, &lapack_info);
            hypre_assert(lapack_info == 0);
         }

#if AIR_DEBUG
         HYPRE_Int one = 1;
         HYPRE_Real alp = 1.0, bet = 0.0;
         hypre_dgemv(&charT, &local_size, &local_size, &alp, TMPA, &local_size, Dbi, 
               &one, &bet, TMPd, &one);
         alp = -1.0;
         hypre_daxpy(&local_size, &alp, TMPb, &one, TMPd, &one);
         HYPRE_Real err = hypre_dnrm2(&local_size, TMPd, &one);
         if (err > 1e-8)
         {
            hypre_printf("local res norm %e\n", err);
         }
#endif
      }
      
      /* now we are ready to fill this row of R */
      /* diag part */
      rr = 0;
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            hypre_assert(marker_diag[i1] == rr);
            /* col idx: use i1, local idx  */
            R_diag_j[cnt_diag] = i1;
            /* copy the value */
            R_diag_data[cnt_diag++] = Dbi[rr++];
         }
      }

      /* don't forget the identity to this row */
      /* global col idx of this entry is ``col_start + i''; */
      R_diag_j[cnt_diag] = i;
      R_diag_data[cnt_diag++] = 1.0;
      
      /* row ptr of the next row */
      R_diag_i[ic+1] = cnt_diag;

      /* offd part */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               hypre_assert(marker_offd[i1] == rr);
               /* col idx: use the local col id of A_offd,
                * and you will see why later (very soon!) */
               R_offd_j[cnt_offd] = i1;
               /* copy the value */
               R_offd_data[cnt_offd++] = Dbi[rr++];
            }
         }
      }
      /* row ptr of the next row */
      R_offd_i[ic+1] = cnt_offd;

      /* we must have copied all entries */
      hypre_assert(rr == local_size);
      
      /* reset markers */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            hypre_assert(marker_diag[i1] >= 0);
            marker_diag[i1] = -1;
         }
      }
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               hypre_assert(marker_offd[i1] >= 0);
               marker_offd[i1] = -1;
            }
         }
      }

      /* next C-pt */
      ic++;
   } /* outermost loop, for (i=0,...), for each C-pt find restriction */

   hypre_assert(ic == n_cpts)
   hypre_assert(cnt_diag == nnz_diag)
   hypre_assert(cnt_offd == nnz_offd)
   
   /* num of cols in the offd part of R */
   num_cols_offd_R = 0;
   /* to this point, marker_offd should be all -1 */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      if (marker_offd[i1] == -1)
      {
         num_cols_offd_R++;
         marker_offd[i1] = 1;
      }
   }

   /* col_map_offd_R: the col indices of the offd of R
    * we first keep them be the offd-idx of A */
   col_map_offd_R = hypre_CTAlloc(HYPRE_Int, num_cols_offd_R, HYPRE_MEMORY_HOST);
   for (i = 0, i1 = 0; i < num_cols_A_offd; i++)
   {
      if (marker_offd[i] == 1)
      {
         col_map_offd_R[i1++] = i;
      }
   }
   hypre_assert(i1 == num_cols_offd_R);

   /* now, adjust R_offd_j to local idx w.r.t col_map_offd_R
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      k1 = hypre_BinarySearch(col_map_offd_R, i1, num_cols_offd_R);
      /* search must succeed */
      hypre_assert(k1 >= 0 && k1 < num_cols_offd_R);
      R_offd_j[i] = k1;
   }

   /* change col_map_offd_R to global ids */
   for (i = 0; i < num_cols_offd_R; i++)
   {
      col_map_offd_R[i] = col_map_offd_A[col_map_offd_R[i]];
   }

   /* Now, we should have everything of Parcsr matrix R */
   R = hypre_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                hypre_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                hypre_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrixData(R_diag) = R_diag_data;
   hypre_CSRMatrixI(R_diag)    = R_diag_i;
   hypre_CSRMatrixJ(R_diag)    = R_diag_j;

   R_offd = hypre_ParCSRMatrixOffd(R);
   hypre_CSRMatrixData(R_offd) = R_offd_data;
   hypre_CSRMatrixI(R_offd)    = R_offd_i;
   hypre_CSRMatrixJ(R_offd)    = R_offd_j;
   /* R does not own ColStarts, since A does */
   hypre_ParCSRMatrixOwnsColStarts(R) = 0;
   
   hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   hypre_MatvecCommPkgCreate(R);

   *R_ptr = R;

   /* free workspace */
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(marker_diag, HYPRE_MEMORY_HOST);
   hypre_TFree(marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(DAi, HYPRE_MEMORY_HOST);
   hypre_TFree(Dbi, HYPRE_MEMORY_HOST);
#if AIR_DEBUG
   hypre_TFree(TMPA, HYPRE_MEMORY_HOST);
   hypre_TFree(TMPb, HYPRE_MEMORY_HOST);
   hypre_TFree(TMPd, HYPRE_MEMORY_HOST);
#endif
   hypre_TFree(Ipi, HYPRE_MEMORY_HOST);
   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(A_ext);
   }

   return 0;
}

