/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.15 $
 ***********************************************************************EHEADER*/




#include "headers.h"

void RowsWithColumn_original
( HYPRE_Int * rowmin, HYPRE_Int * rowmax, HYPRE_Int column, hypre_ParCSRMatrix * A )
/* Finds rows of A which have a nonzero at the given (global) column number.
   Sets rowmin to the minimum (local) row number of such rows, and rowmax
   to the max.  If there are no such rows, will return rowmax<0<=rowmin */
{
   hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int * mat_i, * mat_j;
   HYPRE_Int i, j, num_rows;
   HYPRE_Int firstColDiag;
   HYPRE_Int * colMapOffd;

   mat_i = hypre_CSRMatrixI(diag);
   mat_j = hypre_CSRMatrixJ(diag);
   num_rows = hypre_CSRMatrixNumRows(diag);
   firstColDiag = hypre_ParCSRMatrixFirstColDiag(A);
   *rowmin = num_rows;
   *rowmax = -1;

   for ( i=0; i<num_rows; ++i ) {
      /* global number: row = i + firstRowIndex;*/
      for ( j=mat_i[i]; j<mat_i[i+1]; ++j ) {
         if ( mat_j[j]+firstColDiag==column ) {
            /* row i (local row number) has column mat_j[j] (local column number) */
            *rowmin = i<*rowmin ? i : *rowmin;
            *rowmax = i>*rowmax ? i : *rowmax;
            break;
         }
      }
   }
   mat_i = hypre_CSRMatrixI(offd);
   mat_j = hypre_CSRMatrixJ(offd);
   num_rows = hypre_CSRMatrixNumRows(offd);
   colMapOffd = hypre_ParCSRMatrixColMapOffd(A);
   for ( i=0; i<num_rows; ++i ) {
      /* global number: row = i + firstRowIndex;*/
      for ( j=mat_i[i]; j<mat_i[i+1]; ++j ) {
         if ( colMapOffd[ mat_j[j] ]==column ) {
            /* row i (local row number) has column mat_j[j] (local column number) */
            *rowmin = i<*rowmin ? i : *rowmin;
            *rowmax = i>*rowmax ? i : *rowmax;
            break;
         }
      }
   }

/*      global col no.:  mat_j[j]+hypre_ParCSRMatrixFirstColDiag(A)
                      or hypre_ParCSRMatrixColMapOffd(A)[ mat_j[j] ]
        global row no.: i + hypre_ParCSRMatrixFirstRowIndex(A)
*/

}

void RowsWithColumn
( HYPRE_Int * rowmin, HYPRE_Int * rowmax, HYPRE_Int column,
  HYPRE_Int num_rows_diag, HYPRE_Int firstColDiag, HYPRE_Int * colMapOffd,
  HYPRE_Int * mat_i_diag, HYPRE_Int * mat_j_diag, HYPRE_Int * mat_i_offd, HYPRE_Int * mat_j_offd )
/* Finds rows of A which have a nonzero at the given (global) column number.
   Sets rowmin to the minimum (local) row number of such rows, and rowmax
   to the max.  If there are no such rows, will return rowmax<0<=rowmin
   The matrix A, normally a hypre_ParCSRMatrix or hypre_ParCSRBooleanMatrix,
   is specified by:
 num_rows_diag, (number of rows in diag, assumed to be same in offd)
 firstColDiag, colMapOffd (to map CSR-type matrix columns to ParCSR-type columns
 mat_i_diag, mat_j_diag: indices in the hypre_CSRMatrix or hypre_CSRBooleanMatrix for
   diag block of A
 mat_i_offd, mat_j_offd: indices in the hypre_CSRMatrix or hypre_CSRBooleanMatrix for
   offd block of A
 */
{
   HYPRE_Int i, j;

   *rowmin = num_rows_diag;
   *rowmax = -1;

   for ( i=0; i<num_rows_diag; ++i ) {
      /* global number: row = i + firstRowIndex;*/
      for ( j=mat_i_diag[i]; j<mat_i_diag[i+1]; ++j ) {
         if ( mat_j_diag[j]+firstColDiag==column ) {
            /* row i (local row number) has column mat_j[j] (local column number) */
            *rowmin = i<*rowmin ? i : *rowmin;
            *rowmax = i>*rowmax ? i : *rowmax;
            break;
         }
      }
   }
   for ( i=0; i<num_rows_diag; ++i ) {
      /* global number: row = i + firstRowIndex;*/
      for ( j=mat_i_offd[i]; j<mat_i_offd[i+1]; ++j ) {
         if ( colMapOffd[ mat_j_offd[j] ]==column ) {
            /* row i (local row number) has column mat_j[j] (local column number) */
            *rowmin = i<*rowmin ? i : *rowmin;
            *rowmax = i>*rowmax ? i : *rowmax;
            break;
         }
      }
   }

/*      global col no.:  mat_j[j]+hypre_ParCSRMatrixFirstColDiag(A)
                      or hypre_ParCSRMatrixColMapOffd(A)[ mat_j[j] ]
        global row no.: i + hypre_ParCSRMatrixFirstRowIndex(A)
*/

}


/* hypre_MatTCommPkgCreate_core does all the communications and computations for
       hypre_MatTCommPkgCreate ( hypre_ParCSRMatrix *A)
 and   hypre_BoolMatTCommPkgCreate ( hypre_ParCSRBooleanMatrix *A)
 To support both data types, it has hardly any data structures other than HYPRE_Int*.

*/

void
hypre_MatTCommPkgCreate_core (

/* input args: */
   MPI_Comm comm, HYPRE_Int * col_map_offd, HYPRE_Int first_col_diag, HYPRE_Int * col_starts,
   HYPRE_Int num_rows_diag, HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd, HYPRE_Int * row_starts,
   HYPRE_Int firstColDiag, HYPRE_Int * colMapOffd,
   HYPRE_Int * mat_i_diag, HYPRE_Int * mat_j_diag, HYPRE_Int * mat_i_offd, HYPRE_Int * mat_j_offd,

   HYPRE_Int data,  /* = 1 for a matrix with floating-point data, =0 for Boolean matrix */

/* pointers to output args: */
   HYPRE_Int * p_num_recvs, HYPRE_Int ** p_recv_procs, HYPRE_Int ** p_recv_vec_starts,
   HYPRE_Int * p_num_sends, HYPRE_Int ** p_send_procs, HYPRE_Int ** p_send_map_starts,
   HYPRE_Int ** p_send_map_elmts

   )
{
   HYPRE_Int			num_sends;
   HYPRE_Int			*send_procs;
   HYPRE_Int			*send_map_starts;
   HYPRE_Int			*send_map_elmts;
   HYPRE_Int			num_recvs;
   HYPRE_Int			*recv_procs;
   HYPRE_Int			*recv_vec_starts;
   HYPRE_Int	i, j, j2, k, ir, rowmin, rowmax;
   HYPRE_Int	*tmp, *recv_buf, *displs, *info, *send_buf, *all_num_sends3;
   HYPRE_Int	num_procs, my_id, num_elmts;
   HYPRE_Int	local_info, index, index2;
   HYPRE_Int pmatch, col, kc, p;
   HYPRE_Int * recv_sz_buf;
   HYPRE_Int * row_marker;
   
   hypre_MPI_Comm_size(comm, &num_procs);  
   hypre_MPI_Comm_rank(comm, &my_id);

   info = hypre_CTAlloc(HYPRE_Int, num_procs);

/* ----------------------------------------------------------------------
 * determine which processors to receive from (set proc_mark) and num_recvs,
 * at the end of the loop proc_mark[i] contains the number of elements to be
 * received from Proc. i
 *
 * - For A*b or A*B: for each off-diagonal column i of A, you want to identify
 * the processor which has the corresponding element i of b (row i of B)
 * (columns in the local diagonal block, just multiply local rows of B).
 * You do it by finding the processor which has that column of A in its
 * _diagonal_ block - assuming b or B is distributed the same, which I believe
 * is evenly among processors, by row.  There is a unique solution because
 * the diag/offd blocking is defined by which processor owns which rows of A.
 *
 * - For A*A^T: A^T is not distributed by rows as B or any 'normal' matrix is.
 * For each off-diagonal row,column k,i element of A, you want to identify
 * the processors which have the corresponding row,column i,j elements of A^T
 * i.e., row,column j,i elements of A (all i,j,k for which these entries are
 * nonzero, row k of A lives on this processor, and row j of A lives on
 * a different processor).  So, given a column i in the local A-offd or A-diag,
 * we want to find all the processors which have column i, in diag or offd
 * blocks.  Unlike the A*B case, I don't think you can eliminate looking at
 * any class of blocks.
 * ---------------------------------------------------------------------*/

/* The algorithm for A*B was:
   For each of my columns i (in offd block), use known information on data
   distribution of columns in _diagonal_ blocks to find the processor p which
   owns row i. (Note that for i in diag block, I own the row, nothing to do.)
   Count up such i's for each processor in proc_mark.  Construct a data
   structure, recv_buf, made by appending a structure tmp from each processor.
   The data structure tmp looks like (p, no. of i's, i1, i2,...) (p=0,...) .
   There are two communication steps: gather size information (local_info) from
   all processors (into info), then gather the data (tmp) from all processors
   (into recv_buf).  Then you go through recv_buf.  For each (sink) processor p
   you search for for the appearance of my (source) processor number
   (most of recv_buf pertains to other processors and is ignored).
   When you find the appropriate section, pull out the i's, count them and
   save them, in send_map_elmts, and save p in send_procs and index information
   in send_map_starts.
*/
/* The algorithm for A*A^T:
   [ Originally I had planned to figure out approximately which processors
   had the information (for A*B it could be done exactly) to save on
   communication.  But even for A*B where the data owner is known, all data is
   sent to all processors, so that's not worth worrying about on the first cut.
   One consequence is that proc_mark is not needed.]
   Construct a data structure, recv_buf, made by appending a structure tmp for
   each processor.  It simply consists of (no. of i's, i1, i2,...) where i is
   the global number of a column in the offd block.  There are still two
   communication steps: gather size information (local_info) from all processors
   (into info), then gather the data (tmp) from all processors (into recv_buf).
   Then you go through recv_buf.  For each (sink) processor p you go through
   all its column numbers in recv_buf.  Check each one for whether you have
   data in that column.  If so, put in in send_map_elmts, p in send_procs,
   and update the index information in send_map_starts.  Note that these
   arrays don't mean quite the same thing as for A*B.
*/

   num_recvs=num_procs-1;
   local_info = num_procs + num_cols_offd + num_cols_diag;
			
   hypre_MPI_Allgather(&local_info, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, comm); 

/* ----------------------------------------------------------------------
 * generate information to be send: tmp contains for each recv_proc:
 * {deleted: id of recv_procs}, number of elements to be received for this processor,
 * indices of elements (in this order)
 * ---------------------------------------------------------------------*/

   displs = hypre_CTAlloc(HYPRE_Int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
	displs[i] = displs[i-1]+info[i-1]; 
   recv_buf = hypre_CTAlloc(HYPRE_Int, displs[num_procs]); 
   tmp = hypre_CTAlloc(HYPRE_Int, local_info);

   j = 0;
   for (i=0; i < num_procs; i++) {
      j2 = j++;
      tmp[j2] = 0;
      for (k=0; k < num_cols_offd; k++)
         if (col_map_offd[k] >= col_starts[i] && 
             col_map_offd[k] < col_starts[i+1]) {
            tmp[j++] = col_map_offd[k];
            ++(tmp[j2]);
         };
      for (k=0; k < num_cols_diag; k++)
         if ( k+first_col_diag >= col_starts[i] && 
              k+first_col_diag < col_starts[i+1] ) {
            tmp[j++] = k + first_col_diag;
            ++(tmp[j2]);
         }
   }

   hypre_MPI_Allgatherv(tmp,local_info,HYPRE_MPI_INT,recv_buf,info,displs,HYPRE_MPI_INT,comm);
	

/* ----------------------------------------------------------------------
 * determine send_procs and actual elements to be send (in send_map_elmts)
 * and send_map_starts whose i-th entry points to the beginning of the 
 * elements to be send to proc. i
 * ---------------------------------------------------------------------*/
/* Meanings of arrays being set here, more verbosely stated:
   send_procs: processors p to send to
   send_map_starts: for each p, gives range of indices in send_map_elmts; 
   send_map_elmts:  Each element is a send_map_elmts[i], with i in a range given
     by send_map_starts[p..p+1], for some p. This element is is the global
     column number for a column in the offd block of p which is to be multiplied
     by data from this processor.
     For A*B, send_map_elmts[i] is therefore a row of B belonging to this
     processor, to be sent to p.  For A*A^T, send_map_elmts[i] is a row of A
     belonging to this processor, to be sent to p; this row was selected
     because it has a nonzero on a _column_ needed by p.
*/
   num_sends = num_procs;   /* may turn out to be less, but we can't know yet */
   num_elmts = (num_procs-1)*num_rows_diag;
   /* ... a crude upper bound; should try to do better even if more comm required */
   send_procs = hypre_CTAlloc(HYPRE_Int, num_sends);
   send_map_starts = hypre_CTAlloc(HYPRE_Int, num_sends+1);
   send_map_elmts = hypre_CTAlloc(HYPRE_Int, num_elmts);
   row_marker = hypre_CTAlloc(HYPRE_Int,num_rows_diag);
 
   index = 0;
   index2 = 0; 
   send_map_starts[0] = 0;
   for (i=0; i < num_procs; i++) {
      send_map_starts[index+1] = send_map_starts[index];
      j = displs[i];
      pmatch = 0;
      for ( ir=0; ir<num_rows_diag; ++ir ) row_marker[ir] = 0;
      while ( j < displs[i+1])
      {
         num_elmts = recv_buf[j++];  /* no. of columns proc. i wants */
         for ( k=0; k<num_elmts; k++ ) {
            col = recv_buf[j++]; /* a global column no. at proc. i */
            for ( kc=0; kc<num_cols_offd; kc++ ) {
               if ( col_map_offd[kc]==col && i!=my_id ) {
                  /* this processor has the same column as proc. i (but is different) */
                  pmatch = 1;
                  send_procs[index] = i;
                  /* this would be right if we could send columns, but we can't ...
                     offset = first_col_diag;
                     ++send_map_starts[index+1];
                     send_map_elmts[index2++] = col - offset; */
                  /* Plan to send all of my rows which use this column... */
                  RowsWithColumn( &rowmin, &rowmax, col,
                                  num_rows_diag, 
                                  firstColDiag, colMapOffd,
                                  mat_i_diag, mat_j_diag, mat_i_offd, mat_j_offd
                     );
                  for ( ir=rowmin; ir<=rowmax; ++ir ) {
                     if ( row_marker[ir]==0 ) {
                        row_marker[ir] = 1;
                        ++send_map_starts[index+1];
                        send_map_elmts[index2++] = ir;
                     }
                  }
               }
            }
/* alternative way of doing the following for-loop:
            for ( kc=0; kc<num_cols_diag; kc++ ) {
               if ( kc+first_col_diag==col && i!=my_id ) {
               / * this processor has the same column as proc. i (but is different) * /
                  pmatch = 1;
/ * this would be right if we could send columns, but we can't ... >>> * /
                  send_procs[index] = i;
                  ++send_map_starts[index+1];
                  send_map_elmts[index2++] = col - offset;
                  / * Plan to send all of my rows which use this column... * /
                  / * NOT DONE * /
               }
            }
*/
            for ( kc=row_starts[my_id]; kc<row_starts[my_id+1]; kc++ ) {
               if ( kc==col && i!=my_id ) {
                  /* this processor has the same column as proc. i (but is different) */
                  pmatch = 1;
                  send_procs[index] = i;
/* this would be right if we could send columns, but we can't ... >>> 
                  ++send_map_starts[index+1];
                  send_map_elmts[index2++] = col - offset;*/
                  /* Plan to send all of my rows which use this column... */
                  RowsWithColumn( &rowmin, &rowmax, col,
                                  num_rows_diag, 
                                  firstColDiag, colMapOffd,
                                  mat_i_diag, mat_j_diag, mat_i_offd, mat_j_offd
                     );
                  for ( ir=rowmin; ir<=rowmax; ++ir ) {
                     if ( row_marker[ir]==0 ) {
                        row_marker[ir] = 1;
                        ++send_map_starts[index+1];
                        send_map_elmts[index2++] = ir;
                     }
                  }
               }
            }
         }
      }
      if ( pmatch ) index++;
   }
   num_sends = index;  /* no. of proc. rows will be sent to */

/* Compute receive arrays recv_procs, recv_vec_starts ... */
   recv_procs = hypre_CTAlloc(HYPRE_Int, num_recvs);
   recv_vec_starts = hypre_CTAlloc(HYPRE_Int, num_recvs+1);
   j2 = 0;
   for (i=0; i < num_procs; i++) {
      if ( i!=my_id ) { recv_procs[j2] = i; j2++; };
   };

/* Compute recv_vec_starts.
   The real job is, for each processor p, to figure out how many rows
   p will send to me (me=this processor).  I now know how many (and which)
   rows I will send to each p.  Indeed, if send_procs[index]=p, then the
   number is send_map_starts[index+1]-send_map_starts[index].
   More communication is needed.
   options:
   hypre_MPI_Allgather of communication sizes. <--- my choice, for now
     good: simple   bad: send num_procs*num_sends data, only need num_procs
                    but: not that much data compared to previous communication
   hypre_MPI_Allgatherv of communication sizes, only for pairs of procs. that communicate
     good: less data than above   bad: need extra commun. step to get recvcounts
   hypre_MPI_ISend,hypre_MPI_IRecv of each size, separately between each pair of procs.
     good: no excess data sent   bad: lots of little messages
                                 but: Allgather might be done the same under the hood
     may be much slower than Allgather or may be a bit faster depending on
     implementations
*/
   send_buf = hypre_CTAlloc( HYPRE_Int, 3*num_sends );
   all_num_sends3 = hypre_CTAlloc( HYPRE_Int, num_procs );

   /* scatter-gather num_sends, to set up the size for the main comm. step */
   i = 3*num_sends;
   hypre_MPI_Allgather( &i, 1, HYPRE_MPI_INT, all_num_sends3, 1, HYPRE_MPI_INT, comm );
   displs[0] = 0;
   for ( p=0; p<num_procs; ++p ) {
      displs[p+1] = displs[p] + all_num_sends3[p];
   };
   recv_sz_buf = hypre_CTAlloc( HYPRE_Int, displs[num_procs] );
   
   /* scatter-gather size of row info to send, and proc. to send to */
   index = 0;
   for ( i=0; i<num_sends; ++i ) {
      send_buf[index++] = send_procs[i];   /* processor to send to */
      send_buf[index++] = my_id;
      send_buf[index++] = send_map_starts[i+1] - send_map_starts[i];
      /* ... sizes of info to send */
   };

   hypre_MPI_Allgatherv( send_buf, 3*num_sends, HYPRE_MPI_INT,
                   recv_sz_buf, all_num_sends3, displs, HYPRE_MPI_INT, comm);

   recv_vec_starts[0] = 0;
   j2 = 0;  j = 0;
   for ( i=0; i<displs[num_procs]; i=i+3 ) {
      j = i;
      if ( recv_sz_buf[j++]==my_id ) {
         recv_procs[j2] = recv_sz_buf[j++];
         recv_vec_starts[j2+1] = recv_vec_starts[j2] + recv_sz_buf[j++];
         j2++;
      }
   }
   num_recvs = j2;

#if 0
   hypre_printf("num_procs=%i send_map_starts (%i):",num_procs,num_sends+1);
   for( i=0; i<=num_sends; ++i ) hypre_printf(" %i", send_map_starts[i] );
   hypre_printf("  send_procs (%i):",num_sends);
   for( i=0; i<num_sends; ++i ) hypre_printf(" %i", send_procs[i] );
   hypre_printf("\n");
   hypre_printf("my_id=%i num_sends=%i send_buf[0,1,2]=%i %i %i",
          my_id, num_sends, send_buf[0], send_buf[1], send_buf[2] );
   hypre_printf(" all_num_sends3[0,1]=%i %i\n", all_num_sends3[0], all_num_sends3[1] );
   hypre_printf("my_id=%i rcv_sz_buf (%i):", my_id, displs[num_procs] );
   for( i=0; i<displs[num_procs]; ++i ) hypre_printf(" %i", recv_sz_buf[i] );
   hypre_printf("\n");
   hypre_printf("my_id=%i recv_vec_starts (%i):",my_id,num_recvs+1);
   for( i=0; i<=num_recvs; ++i ) hypre_printf(" %i", recv_vec_starts[i] );
   hypre_printf("  recv_procs (%i):",num_recvs);
   for( i=0; i<num_recvs; ++i ) hypre_printf(" %i", recv_procs[i] );
   hypre_printf("\n");
   hypre_printf("my_id=%i num_recvs=%i recv_sz_buf[0,1,2]=%i %i %i\n",
          my_id, num_recvs, recv_sz_buf[0], recv_sz_buf[1], recv_sz_buf[2] );
#endif

   hypre_TFree(send_buf);
   hypre_TFree(all_num_sends3);
   hypre_TFree(tmp);
   hypre_TFree(recv_buf);
   hypre_TFree(displs);
   hypre_TFree(info);
   hypre_TFree(recv_sz_buf);
   hypre_TFree(row_marker);


   /* finish up with the hand-coded call-by-reference... */
   *p_num_recvs = num_recvs;
   *p_recv_procs = recv_procs;
   *p_recv_vec_starts = recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_procs;
   *p_send_map_starts = send_map_starts;
   *p_send_map_elmts = send_map_elmts;

}

/* ----------------------------------------------------------------------
 * hypre_MatTCommPkgCreate
 * generates a special comm_pkg for A - for use in multiplying by its
 * transpose, A * A^T
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d 
 * ---------------------------------------------------------------------*/

HYPRE_Int
hypre_MatTCommPkgCreate ( hypre_ParCSRMatrix *A)
{
   hypre_ParCSRCommPkg	*comm_pkg;
   
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
/*   hypre_MPI_Datatype         *recv_mpi_types;
   hypre_MPI_Datatype         *send_mpi_types;
*/
   HYPRE_Int			num_sends;
   HYPRE_Int			*send_procs;
   HYPRE_Int			*send_map_starts;
   HYPRE_Int			*send_map_elmts;
   HYPRE_Int			num_recvs;
   HYPRE_Int			*recv_procs;
   HYPRE_Int			*recv_vec_starts;
   
   HYPRE_Int  *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int  first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int  *col_starts = hypre_ParCSRMatrixColStarts(A);

   HYPRE_Int	ierr = 0;
   HYPRE_Int	num_rows_diag = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int	num_cols_diag = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int	num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   HYPRE_Int * row_starts = hypre_ParCSRMatrixRowStarts(A);

   hypre_MatTCommPkgCreate_core (
      comm, col_map_offd, first_col_diag, col_starts,
      num_rows_diag, num_cols_diag, num_cols_offd, row_starts,
                                  hypre_ParCSRMatrixFirstColDiag(A),
                                  hypre_ParCSRMatrixColMapOffd(A),
                                  hypre_CSRMatrixI( hypre_ParCSRMatrixDiag(A) ),
                                  hypre_CSRMatrixJ( hypre_ParCSRMatrixDiag(A) ),
                                  hypre_CSRMatrixI( hypre_ParCSRMatrixOffd(A) ),
      hypre_CSRMatrixJ( hypre_ParCSRMatrixOffd(A) ),
      1,
      &num_recvs, &recv_procs, &recv_vec_starts,
      &num_sends, &send_procs, &send_map_starts,
      &send_map_elmts
      );

   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;

   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts;

   hypre_ParCSRMatrixCommPkgT(A) = comm_pkg;

   return ierr;
}
