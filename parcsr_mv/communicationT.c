#include "headers.h"

void RowsWithColumn
( int * rowmin, int * rowmax, int column, hypre_ParCSRMatrix * A )
/* Finds rows of A which have a nonzero at the given (global) column number.
   Sets rowmin to the minimum (local) row number of such rows, and rowmax
   to the max.  If there are no such rows, will return rowmax<0<=rowmin */
{
   hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(A);
   int * mat_i, * mat_j;
   int i, j, num_rows;
   int firstColDiag;
   int * colMapOffd;

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

/* ----------------------------------------------------------------------
 * hypre_MatTCommPkgCreate
 * generates a special comm_pkg for A - for use in multiplying by its
 * transpose, A * A^T
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d 
 * ---------------------------------------------------------------------*/

int
hypre_MatTCommPkgCreate ( hypre_ParCSRMatrix *A)
{
   hypre_ParCSRCommPkg	*comm_pkg;
   
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
/*   MPI_Datatype         *recv_mpi_types;
   MPI_Datatype         *send_mpi_types;
*/
   int			num_sends;
   int			*send_procs;
   int			*send_map_starts;
   int			*send_map_elmts;
   int			num_recvs;
   int			*recv_procs;
   int			*recv_vec_starts;
   
   int  *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   int  first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
   int  *col_starts = hypre_ParCSRMatrixColStarts(A);

   int	i, j, j2, k, ir, rowmin, rowmax;
   int	*tmp, *recv_buf, *displs, *info, *send_buf, *all_num_sends2;
   int	num_procs, my_id, num_elmts;
   int	local_info, index, index2;
   int	ierr = 0;
   int pmatch, col, kc, p;
   int * recv_sz_buf;
   int * row_marker;
   int	num_rows_diag = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   int	num_cols_diag = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   int	num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));

   MPI_Comm_size(comm, &num_procs);  
   MPI_Comm_rank(comm, &my_id);

   info = hypre_CTAlloc(int, num_procs);

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
			
   MPI_Allgather(&local_info, 1, MPI_INT, info, 1, MPI_INT, comm); 

/* ----------------------------------------------------------------------
 * generate information to be send: tmp contains for each recv_proc:
 * {deleted: id of recv_procs}, number of elements to be received for this processor,
 * indices of elements (in this order)
 * ---------------------------------------------------------------------*/

   displs = hypre_CTAlloc(int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
	displs[i] = displs[i-1]+info[i-1]; 
   recv_buf = hypre_CTAlloc(int, displs[num_procs]); 
   tmp = hypre_CTAlloc(int, local_info);

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

   MPI_Allgatherv(tmp,local_info,MPI_INT,recv_buf,info,displs,MPI_INT,comm);
	

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
   send_procs = hypre_CTAlloc(int, num_sends);
   send_map_starts = hypre_CTAlloc(int, num_sends+1);
   send_map_elmts = hypre_CTAlloc(int, num_elmts);
   row_marker = hypre_CTAlloc(int,num_rows_diag);
 
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
                  RowsWithColumn( &rowmin, &rowmax, col, A );
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
            for ( kc=hypre_ParCSRMatrixRowStarts(A)[my_id];
                  kc<hypre_ParCSRMatrixRowStarts(A)[my_id+1]; kc++ ) {
               if ( kc==col && i!=my_id ) {
                  /* this processor has the same column as proc. i (but is different) */
                  pmatch = 1;
                  send_procs[index] = i;
/* this would be right if we could send columns, but we can't ... >>> 
                  ++send_map_starts[index+1];
                  send_map_elmts[index2++] = col - offset;*/
                  /* Plan to send all of my rows which use this column... */
                  RowsWithColumn( &rowmin, &rowmax, col, A );
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
   recv_procs = hypre_CTAlloc(int, num_recvs);
   recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);
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
   MPI_Allgather of communication sizes. <--- my choice, for now
     good: simple   bad: send num_procs*num_sends data, only need num_procs
                    but: not that much data compared to previous communication
   MPI_Allgatherv of communication sizes, only for pairs of procs. that communicate
     good: less data than above   bad: need extra commun. step to get recvcounts
   MPI_ISend,MPI_IRecv of each size, separately between each pair of procs.
     good: no excess data sent   bad: lots of little messages
                                 but: Allgather might be done the same under the hood
     may be much slower than Allgather or may be a bit faster depending on
     implementations
*/
   send_buf = hypre_CTAlloc( int, 2*num_sends );
   all_num_sends2 = hypre_CTAlloc( int, num_procs );

   /* scatter-gather num_sends, to set up the size for the main comm. step */
   i = 2*num_sends;
   MPI_Allgather( &i, 1, MPI_INT, all_num_sends2, 1, MPI_INT, comm );
   displs[0] = 0;
   for ( p=0; p<num_procs; ++p ) {
      displs[p+1] = displs[p] + all_num_sends2[p];
   };
   recv_sz_buf = hypre_CTAlloc( int, displs[num_procs] );
   
   /* scatter-gather size of row info to send, and proc. to send to */
   index = 0;
   for ( i=0; i<num_sends; ++i ) {
      send_buf[index++] = send_procs[i];   /* processor to send to */
      send_buf[index++] = send_map_starts[i+1] - send_map_starts[i];
      /* ... sizes of info to send */
   };
#if 0
      printf("num_procs=%i send_map_starts (%i):",num_procs,num_sends+1);
      for( i=0; i<=num_sends; ++i ) printf(" %i", send_map_starts[i] );
      printf("\n");
      printf("my_id=%i num_sends=%i send_buf[0,1]=%i %i",
             my_id, num_sends, send_buf[0], send_buf[1] );
      printf(" all_num_sends2[0,1]=%i %i\n", all_num_sends2[0], all_num_sends2[1] );
#endif
   MPI_Allgatherv( send_buf, 2*num_sends, MPI_INT,
                   recv_sz_buf, all_num_sends2, displs, MPI_INT, comm);
   recv_vec_starts[0] = 0;
   j2 = 0;  j = 0;
   for ( p=0; p<num_procs; ++p ) {
      if ( recv_sz_buf[j++]==my_id ) {
         recv_procs[j2] = p;
         recv_vec_starts[j2+1] = recv_vec_starts[j2] + recv_sz_buf[j++];
         j2++;
      }
      else {
         j++;
      };
   };

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

   hypre_TFree(send_buf);
   hypre_TFree(all_num_sends2);
   hypre_TFree(tmp);
   hypre_TFree(recv_buf);
   hypre_TFree(displs);
   hypre_TFree(info);
   hypre_TFree(recv_sz_buf);
   hypre_TFree(row_marker);

   return ierr;
}
