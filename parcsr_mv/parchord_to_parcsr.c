/* ----------------------------------------------------------------------- */
/*                                                                         */
/*                     ParCSRMatrix to ParChordMatrix                      */
/*                                 and                                     */
/*                     ParCSRMatrix to ParChordMatrix:                     */
/*                                                                         */
/* ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#include "headers.h"

void hypre_ParChordMatrix_RowStarts(
   hypre_ParChordMatrix *Ac, MPI_Comm comm,
   int ** row_starts, int * global_num_cols )
   /* This function computes the ParCSRMatrix-style row_starts from a chord matrix.
      It requires the the idofs of the chord matrix be partitioned among
      processors, so their numbering is monotonic with the processor number;
      see below.

      The algorithm: each proc. p knows its min & max global row & col numbers.
      Mins are first_index_rdof[p], first_index_idof[p]
      ***IF*** these were in proper order (see below),
      first_index_rdof[p] is row_starts[p].
      Add num_rdofs-1 to get the max, i.e. add num_rdofs
      to get row_starts[p+1] (IF the processors are ordered thus).
      Compute these, then broadcast to the other processors to form
      row_starts.
      (We also could get global_num_rows by an AllReduce num_idofs.)
      We get global_num_cols by taking the min and max over processors of
      the min and max col no.s on each processor.

      If the chord matrix is not ordered so the above will work, then we
      would need to to completely move matrices around sometimes, a very expensive
      operation.
      The problem is that the chord matrix format makes no assumptions about
      processor order, but the ParCSR format assumes that
      p<q => (local row numbers of p) < (local row numbers of q)
      Maybe instead I could change the global numbering scheme as part of this
      conversion.
      A closely related ordering-type problem to watch for: row_starts must be
      a partition for a ParCSRMatrix.  In a ChordMatrix, the struct itself
      makes no guarantees, but Panayot said, in essence, that row_starts will
      be a partition.
      col_starts should be NULL; later we shall let the Create function compute one.
   */
{
   int * fis_idof = hypre_ParChordMatrixFirstindexIdof(Ac);
   int * fis_rdof = hypre_ParChordMatrixFirstindexRdof(Ac);
   int my_id, num_procs;
   int num_idofs = hypre_ParChordMatrixNumIdofs(Ac);
   int num_rdofs = hypre_ParChordMatrixNumRdofs(Ac);
   int min_rdof, max_rdof, global_min_rdof, global_max_rdof;
   int p, lens[2], lastlens[2];
   MPI_Status *status;
   MPI_Request *request;

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);
   request = hypre_CTAlloc(MPI_Request, 1 );
   status = hypre_CTAlloc(MPI_Status, 1 );

   min_rdof = fis_rdof[my_id];
   max_rdof = min_rdof + num_rdofs;
   lens[0] = num_idofs;
   lens[1] = num_rdofs;

   /* row_starts (except last value */
   *row_starts = hypre_CTAlloc( int, num_procs+1 );
   for ( p=0; p<num_procs; ++p ) {
      (*row_starts)[p] = fis_idof[p];
   }

   /* check that ordering and partitioning of rows is as expected
      (much is missing, and even then not perfect)... */
   if ( my_id<num_procs-1 )
      MPI_Isend( lens, 2, MPI_INT, my_id+1, 0, comm, request );
   if ( my_id>0 )
      MPI_Recv( lastlens, 2, MPI_INT, my_id-1, 0, comm, status );
   if ( my_id<num_procs-1 )
	MPI_Waitall( 1, request, status);
   if ( my_id>0 )
      assert( (*row_starts)[my_id] == (*row_starts)[my_id-1] + lastlens[0] );
   hypre_TFree( request );
   hypre_TFree( status );
      
   /* Get the upper bound for all the rows */
   MPI_Bcast( lens, 2, MPI_INT, num_procs-1, comm );
   (*row_starts)[num_procs] = (*row_starts)[num_procs-1] + lens[0];

   /* Global number of columns */
/*   MPI_Allreduce( &num_rdofs, global_num_cols, 1, MPI_INT, MPI_SUM, comm );*/
   MPI_Allreduce( &min_rdof, &global_min_rdof, 1, MPI_INT, MPI_MIN, comm );
   MPI_Allreduce( &max_rdof, &global_max_rdof, 1, MPI_INT, MPI_MAX, comm );
   *global_num_cols = global_max_rdof - global_min_rdof;
}

int
hypre_ParChordMatrixToParCSRMatrix(
   hypre_ParChordMatrix *Ac,
   MPI_Comm comm,
   hypre_ParCSRMatrix **pAp )
{
   /* Some parts of this function are copied from hypre_CSRMatrixToParCSRMatrix. */

   hypre_ParCSRMatrix *Ap;
   int *row_starts, *col_starts;
   int global_num_rows, global_num_cols, my_id, num_procs;
   int num_cols_offd, num_nonzeros_diag, num_nonzeros_offd;
   int          *local_num_rows;
/* not computed   int          *local_num_nonzeros; */
   int          num_nonzeros, first_col_diag, last_col_diag;
   int i,ic,ij,ir,ilocal,p,r,r_p,r_global,r_local, jlen;
   int *a_i, *a_j, *ilen;
   int **rdofs, **ps;
   double data;
   double *a_data;
   double **datas;
   hypre_CSRMatrix *local_A;

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);

   hypre_ParChordMatrix_RowStarts
      ( Ac, comm, &row_starts, &global_num_cols );
   /* ... this function works correctly only under some assumptions;
      see the function definition for details */
   global_num_rows = row_starts[num_procs] - row_starts[0];

   col_starts = NULL;
   /* The offd and diag blocks aren't defined until we have both row
      and column partitions... */
   num_cols_offd = 0;
   num_nonzeros_diag = 0;
   num_nonzeros_offd = 0;

   Ap  = hypre_ParCSRMatrixCreate( comm, global_num_rows, global_num_cols,
                          row_starts, col_starts,
                          num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
   *pAp = Ap;

   row_starts = hypre_ParCSRMatrixRowStarts(Ap);
   col_starts = hypre_ParCSRMatrixColStarts(Ap);

   local_num_rows = hypre_CTAlloc(int, num_procs);
   for (i=0; i < num_procs; i++)
         local_num_rows[i] = row_starts[i+1] - row_starts[i];

   num_nonzeros = 0;
   for ( p=0; p<hypre_ParChordMatrixNumInprocessors(Ac); ++p ) {
      num_nonzeros += hypre_ParChordMatrixNumInchords(Ac)[p];
   };

   local_A = hypre_CSRMatrixCreate( local_num_rows[my_id], global_num_cols,
                                    num_nonzeros );

   /* Compute local CSRMatrix-like i,j arrays for this processor. */

   ps = hypre_CTAlloc( int*, hypre_ParChordMatrixNumIdofs(Ac) );
   rdofs = hypre_CTAlloc( int*, hypre_ParChordMatrixNumIdofs(Ac) );
   datas = hypre_CTAlloc( double*, hypre_ParChordMatrixNumIdofs(Ac) );
   ilen  = hypre_CTAlloc( int, hypre_ParChordMatrixNumIdofs(Ac) );
   jlen = 0;
   for ( i=0; i<hypre_ParChordMatrixNumIdofs(Ac); ++i ) {
      ilen[i] = 0;
      ps[i] = hypre_CTAlloc( int, hypre_ParChordMatrixNumRdofs(Ac) );
      rdofs[i] = hypre_CTAlloc( int, hypre_ParChordMatrixNumRdofs(Ac) );
      datas[i] = hypre_CTAlloc( double, hypre_ParChordMatrixNumRdofs(Ac) );
      /* ... rdofs[i], datas[i] will generally, not always, be much too big */
   }
   for ( p=0; p<hypre_ParChordMatrixNumInprocessors(Ac); ++p ) {
      for ( ic=0; ic<hypre_ParChordMatrixNumInchords(Ac)[p]; ++ic ) {
         ilocal = hypre_ParChordMatrixInchordIdof(Ac)[p][ic];
         r = hypre_ParChordMatrixInchordRdof(Ac)[p][ic];
         data = hypre_ParChordMatrixInchordData(Ac)[p][ic];
         ps[ilocal][ ilen[ilocal] ] = p;
         rdofs[ilocal][ ilen[ilocal] ] = r;
         datas[ilocal][ ilen[ilocal] ] = data;
         ++ilen[ilocal];
         ++jlen;
      }
   };

   a_i = hypre_CTAlloc( int, hypre_ParChordMatrixNumIdofs(Ac)+1 );
   a_j = hypre_CTAlloc( int, jlen );
   a_data = hypre_CTAlloc( double, jlen );
   a_i[0] = 0;
   for ( ilocal=0; ilocal<hypre_ParChordMatrixNumIdofs(Ac); ++ilocal ) {
      a_i[ilocal+1] = a_i[ilocal] + ilen[ilocal];
      ir = 0;
      for ( ij=a_i[ilocal]; ij<a_i[ilocal+1]; ++ij ) {
         p = ps[ilocal][ir];
         r_p = rdofs[ilocal][ir];  /* local in proc. p */
         r_global = r_p + hypre_ParChordMatrixFirstindexRdof(Ac)[p];
         r_local = r_global - hypre_ParChordMatrixFirstindexRdof(Ac)[my_id];
         a_j[ij] = r_local;
         a_data[ij] = datas[ilocal][ir];
         ir++;
      };
   };

   for ( i=0; i<hypre_ParChordMatrixNumIdofs(Ac); ++i ) {
      hypre_TFree( ps[i] );
      hypre_TFree( rdofs[i] );
      hypre_TFree( datas[i] );
   };
   hypre_TFree( ps );
   hypre_TFree( rdofs );
   hypre_TFree( datas );
   hypre_TFree( ilen );

   first_col_diag = col_starts[my_id];
   last_col_diag = col_starts[my_id+1]-1;

   hypre_CSRMatrixData(local_A) = a_data;
   hypre_CSRMatrixI(local_A) = a_i;
   hypre_CSRMatrixJ(local_A) = a_j;
   hypre_CSRMatrixOwnsData(local_A) = 0;

   GenerateDiagAndOffd(local_A, Ap, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {
      hypre_TFree(a_data);
      /* ... the data has been copied into different diag & offd arrays of Ap */
      hypre_TFree(a_j);
      hypre_TFree(a_i);
      hypre_CSRMatrixData(local_A) = NULL;
      hypre_CSRMatrixI(local_A) = NULL;
      hypre_CSRMatrixJ(local_A) = NULL; 
   }      
   hypre_CSRMatrixDestroy(local_A);
   hypre_TFree(local_num_rows);
/*   hypre_TFree(csr_matrix_datatypes);*/
}

int
hypre_ParCSRMatrixToParChordMatrix(
   hypre_ParCSRMatrix *Ap,
   MPI_Comm comm,
   hypre_ParChordMatrix **pAc )
{
   int * row_starts = hypre_ParCSRMatrixRowStarts(Ap);
   int * col_starts = hypre_ParCSRMatrixColStarts(Ap);
   hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(Ap);
   hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(Ap);
   int * offd_j = hypre_CSRMatrixJ(offd);
   int * diag_j = hypre_CSRMatrixJ(diag);
   int * col_map_offd = hypre_ParCSRMatrixColMapOffd(Ap);
   int first_col_diag = hypre_ParCSRMatrixFirstColDiag(Ap);

   hypre_ParChordMatrix * Ac;
   hypre_NumbersNode * rdofs, * offd_cols_me;
   hypre_NumbersNode ** offd_cols;
   int ** offd_col_array;
   int * len_offd_col_array, * offd_col_array_me;
   int len_offd_col_array_me;
   int num_idofs, num_rdofs, j_local, j_global, row_global;
   int i, j, jj, p, pto, q, qto, my_id, my_q, row, ireq, jjd;
   int num_inprocessors, num_toprocessors, num_procs, len_num_rdofs_toprocessor;
   int *inprocessor, *toprocessor, *pcr, *qcr, *num_inchords, *chord, *chordto;
   int *inproc, *toproc, *num_rdofs_toprocessor;
   int **inchord_idof, **inchord_rdof, **rdof_toprocessor;
   double **inchord_data;
   double data;
   int *first_index_idof, *first_index_rdof;
   MPI_Request * request;
   MPI_Status * status;

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);
   num_idofs = row_starts[my_id+1] - row_starts[my_id];
   num_rdofs = col_starts[my_id+1] - col_starts[my_id];

   hypre_ParChordMatrixCreate( pAc, comm, num_idofs, num_rdofs );
   Ac = *pAc;

/* The following block sets Inprocessor:
   On each proc. my_id, we find the columns in the offd and diag blocks
   (global no.s).  The columns are rdofs (contrary to what I wrote in
   ChordMatrix.txt).
   For each such col/rdof r, find the proc. p which owns row/idof r.
   We set the temporary array pcr[p]=1 for such p.
   An MPI all-to-all will exchange such arrays so my_id's array qcr has
   qcr[q]=1 iff, on proc. q, pcr[my_id]=1.  In other words, qcr[q]=1 if
   my_id owns a row/idof i which is the same as a col/rdof owned by q.
   Collect all such q's into in the array Inprocessor.
   While constructing pcr, we also construct pj such that for any index jj
   into offd_j,offd_data, pj[jj] is the processor which owns jj as a row (idof)
   (the number jj is local to this processor).
   */
   pcr = hypre_CTAlloc( int, num_procs );
   qcr = hypre_CTAlloc( int, num_procs );
   for ( p=0; p<num_procs; ++p ) pcr[p]=0;
   for ( jj=0; jj<hypre_CSRMatrixNumNonzeros(offd); ++jj ) {
      j_local = offd_j[jj];
      j_global =  col_map_offd[j_local];
      for ( p=0; p<num_procs; ++p ) {
         if ( j_global >= row_starts[p] && j_global<row_starts[p+1] ) {
            pcr[p]=1;
/* not used yet...            pj[jj] = p;*/
            break;
         }
      }
   }
   jjd = jj;

   /* pcr[my_id] = 1; ...for square matrices (with nonzero diag block)
      this one line  would do the job of the following nested loop.
      For non-square matrices, the data distribution is too arbitrary. */
   for ( jj=0; jj<hypre_CSRMatrixNumNonzeros(diag); ++jj ) {
      j_local = diag_j[jj];
      j_global = j_local + first_col_diag;
      for ( p=0; p<num_procs; ++p ) {
         if ( j_global >= row_starts[p] && j_global<row_starts[p+1] ) {
            pcr[p]=1;
/* not used yet...            pj[jj+jjd] = p;*/
            break;
         }
      }
   }


   /* Now pcr[p]=1 iff my_id owns a col/rdof r which proc. p owns as a row/idof */
   MPI_Alltoall( pcr, 1, MPI_INT, qcr, 1, MPI_INT, comm );
   /* Now qcr[q]==1 if my_id owns a row/idof i which is a col/rdof of proc. q
    The array of such q's is the array Inprocessor. */

   num_inprocessors = 0;
   for ( q=0; q<num_procs; ++q ) if ( qcr[q]==1 ) ++num_inprocessors;
   inprocessor = hypre_CTAlloc( int, num_inprocessors );
   p = 0;
   for ( q=0; q<num_procs; ++q ) if ( qcr[q]==1 ) inprocessor[ p++ ] = q;
   num_toprocessors = 0;
   for ( q=0; q<num_procs; ++q ) if ( pcr[q]==1 ) ++num_toprocessors;
   toprocessor = hypre_CTAlloc( int, num_toprocessors );
   p = 0;
   for ( q=0; q<num_procs; ++q ) if ( pcr[q]==1 ) toprocessor[ p++ ] = q;

   hypre_ParChordMatrixNumInprocessors(Ac) = num_inprocessors;
   hypre_ParChordMatrixInprocessor(Ac) = inprocessor;
   hypre_ParChordMatrixNumToprocessors(Ac) = num_toprocessors;
   hypre_ParChordMatrixToprocessor(Ac) = toprocessor;
   hypre_TFree( qcr );

   /* FirstIndexIdof[p] is the global index of proc. p's row 0 */
   /* FirstIndexRdof[p] is the global index of proc. p's col 0 */
   /* Fir FirstIndexIdof, we copy the array row_starts rather than its pointers,
      because the chord matrix will think it's free to delete FirstIndexIdof */
   /* col_starts[p] contains the global index of the first column
      in the diag block of p.  But for first_index_rdof we want the global
      index of the first column in p (whether that's in the diag or offd block).
      So it's more involved than row/idof: we also check the offd block, and
      have to do a gather to get first_index_rdof for every proc. on every proc. */
   first_index_idof = hypre_CTAlloc( int, num_procs+1 );
   first_index_rdof = hypre_CTAlloc( int, num_procs+1 );
   for ( p=0; p<=num_procs; ++p ) {
      first_index_idof[p] = row_starts[p];
      first_index_rdof[p] = col_starts[p];
   };
   if ( hypre_CSRMatrixNumRows(offd) > 0  && hypre_CSRMatrixNumCols(offd) > 0 )
      first_index_rdof[my_id] =
         col_starts[my_id]<col_map_offd[0] ? col_starts[my_id] : col_map_offd[0];
   MPI_Allgather( &first_index_rdof[my_id], 1, MPI_INT,
                  first_index_rdof, 1, MPI_INT, comm );

   /* Set num_inchords: num_inchords[p] is no. chords on my_id connected to p.
      Set each chord (idof,jdof,data).
      We go through each matrix element in the diag block, find what processor
      owns its column no. as a row, then update num_inchords[p], inchord_idof[p],
      inchord_rdof[p], inchord_data[p].
   */

   inchord_idof = hypre_CTAlloc( int*, num_inprocessors );
   inchord_rdof = hypre_CTAlloc( int*, num_inprocessors );
   inchord_data = hypre_CTAlloc( double*, num_inprocessors );
   num_inchords = hypre_CTAlloc( int, num_inprocessors );
   chord = hypre_CTAlloc( int, num_inprocessors );
   chordto = hypre_CTAlloc( int, num_toprocessors );
   num_rdofs = 0;
   for ( q=0; q<num_inprocessors; ++q ) num_inchords[q] = 0;
   my_q = -1;
   for ( q=0; q<num_inprocessors; ++q ) if ( inprocessor[q]==my_id ) my_q = q;
   assert( my_q>=0 );

   /* diag block: first count chords (from my_id to my_id),
      then set them from diag block's CSR data structure */
   num_idofs = hypre_CSRMatrixNumRows(diag);
   rdofs = hypre_NumbersNewNode();
   for ( row=0; row<hypre_CSRMatrixNumRows(diag); ++row ) {
      for ( i=hypre_CSRMatrixI(diag)[row]; i<hypre_CSRMatrixI(diag)[row+1]; ++i ) {
         j_local = hypre_CSRMatrixJ(diag)[i];
         hypre_NumbersEnter( rdofs, j_local );
         ++num_inchords[my_q];
      }
   };
   num_rdofs = hypre_NumbersNEntered( rdofs );
   inchord_idof[my_q] = hypre_CTAlloc( int, num_inchords[my_q] );
   inchord_rdof[my_q] = hypre_CTAlloc( int, num_inchords[my_q] );
   inchord_data[my_q] = hypre_CTAlloc( double, num_inchords[my_q] );
   chord[0] = 0;
   for ( row=0; row<hypre_CSRMatrixNumRows(diag); ++row ) {
      for ( i=hypre_CSRMatrixI(diag)[row]; i<hypre_CSRMatrixI(diag)[row+1]; ++i ) {
         j_local = hypre_CSRMatrixJ(diag)[i];
         data = hypre_CSRMatrixData(diag)[i];
         inchord_idof[my_q][chord[0]] = row;
         /* Here We need to convert from j_local - a column local to
            the diag of this proc., to a j which is local only to this
            processor - a column (rdof) numbering scheme to be shared by the
            diag and offd blocks...  */
         j_global = j_local + hypre_ParCSRMatrixColStarts(Ap)[my_q];
         j = j_global - first_index_rdof[my_q];
         /* This j is local to the processor q - but for diag & offd combined */
         inchord_rdof[my_q][chord[0]] = j;
         inchord_data[my_q][chord[0]] = data;
         assert( chord[0] < num_inchords[my_q] );
         ++chord[0];
      }
   };
   hypre_NumbersDeleteNode(rdofs);


   /* offd block: */

   /* >>> offd_cols_me duplicates rdofs */
   offd_cols_me = hypre_NumbersNewNode();
   for ( row=0; row<hypre_CSRMatrixNumRows(offd); ++row ) {
      for ( i=hypre_CSRMatrixI(offd)[row]; i<hypre_CSRMatrixI(offd)[row+1]; ++i ) {
         j_local = hypre_CSRMatrixJ(offd)[i];
         j_global =  col_map_offd[j_local];
         hypre_NumbersEnter( offd_cols_me, j_global );
      }
   }
   offd_col_array = hypre_CTAlloc( int*, num_inprocessors );
   len_offd_col_array = hypre_CTAlloc( int, num_inprocessors );
   offd_col_array_me = hypre_NumbersArray( offd_cols_me );
   len_offd_col_array_me = hypre_NumbersNEntered( offd_cols_me );
   request = hypre_CTAlloc(MPI_Request, 2*num_procs );
   ireq = 0;
   for ( q=0; q<num_inprocessors; ++q )
      MPI_Irecv( &len_offd_col_array[q], 1, MPI_INT,
                 inprocessor[q], 0, comm, &request[ireq++] );
   for ( p=0; p<num_procs; ++p ) if ( pcr[p]==1 ) {
      MPI_Isend( &len_offd_col_array_me, 1, MPI_INT, p, 0, comm, &request[ireq++] );
   }
   status = hypre_CTAlloc(MPI_Status, ireq );
   MPI_Waitall( ireq, request, status );
   hypre_TFree(status);
   ireq = 0;
   for ( q=0; q<num_inprocessors; ++q )
      offd_col_array[q] = hypre_CTAlloc( int, len_offd_col_array[q] );
   for ( q=0; q<num_inprocessors; ++q )
      MPI_Irecv( offd_col_array[q], len_offd_col_array[q], MPI_INT,
                 inprocessor[q], 0, comm, &request[ireq++] );
   for ( p=0; p<num_procs; ++p ) if ( pcr[p]==1 ) {
      MPI_Isend( offd_col_array_me, len_offd_col_array_me,
                 MPI_INT, p, 0, comm, &request[ireq++] );
   }
   status = hypre_CTAlloc(MPI_Status, ireq );
   MPI_Waitall( ireq, request, status );
   hypre_TFree(request);
   hypre_TFree(status);
   offd_cols = hypre_CTAlloc( hypre_NumbersNode *, num_inprocessors );
   for ( q=0; q<num_inprocessors; ++q ) {
      offd_cols[q] = hypre_NumbersNewNode();
      for ( i=0; i<len_offd_col_array[q]; ++i )
         hypre_NumbersEnter( offd_cols[q], offd_col_array[q][i] );
   }

   len_num_rdofs_toprocessor = 1 + hypre_CSRMatrixI(offd)
      [hypre_CSRMatrixNumRows(offd)];
   inproc = hypre_CTAlloc( int, len_num_rdofs_toprocessor );
   toproc = hypre_CTAlloc( int, len_num_rdofs_toprocessor );
   num_rdofs_toprocessor = hypre_CTAlloc( int, len_num_rdofs_toprocessor );
   for ( qto=0; qto<len_num_rdofs_toprocessor; ++qto ) {
      inproc[qto] = -1;
      toproc[qto] = -1;
      num_rdofs_toprocessor[qto] = 0;
   };
   rdofs = hypre_NumbersNewNode();
   for ( row=0; row<hypre_CSRMatrixNumRows(offd); ++row ) {
      for ( i=hypre_CSRMatrixI(offd)[row]; i<hypre_CSRMatrixI(offd)[row+1]; ++i ) {
         j_local = hypre_CSRMatrixJ(offd)[i];
         j_global =  col_map_offd[j_local];
         hypre_NumbersEnter( rdofs, j_local );
         
         /* TO DO: find faster ways to do the two processor lookups below.
         /* Find a processor p (local index q) from the inprocessor list,
            which owns the column(rdof) whichis the same as this processor's
            row(idof) row. Update num_inchords for p.
            Save q as inproc[i] for quick recall later.  It represents
            an inprocessor (not unique) connected to a chord i.
         */
         inproc[i] = -1;
         for ( q=0; q<num_inprocessors; ++q ) if (q!=my_q) {
            p = inprocessor[q];
            if ( hypre_NumbersQuery( offd_cols[q],
                                     row+hypre_ParCSRMatrixFirstRowIndex(Ap) )
                 == 1 ) {
               /* row is one of the offd columns of p */
               ++num_inchords[q];
               inproc[i] = q;
               break;
            }
         }
         if ( inproc[i]<0 ) {
            /* For square matrices, we would have found the column in some
               other processor's offd.  But for non-square matrices it could
               exist only in some other processor's diag...*/
            /* Note that all data in a diag block is stored.  We don't check
               whether the value of a data entry is zero. */
            for ( q=0; q<num_inprocessors; ++q ) if (q!=my_q) {
               p = inprocessor[q];
               row_global = row+hypre_ParCSRMatrixFirstRowIndex(Ap);
               if ( row_global>=col_starts[p] &&
                    row_global< col_starts[p+1] ) {
                  /* row is one of the diag columns of p */
                  ++num_inchords[q];
                  inproc[i] = q;
                  break;
               }
            }  
         }
         assert( inproc[i]>=0 );

         /* Find the processor pto (local index qto) from the toprocessor list,
            which owns the row(idof) which is the  same as this processor's
            column(rdof) j_global. Update num_rdofs_toprocessor for pto.
            Save pto as toproc[i] for quick recall later. It represents
            the toprocessor connected to a chord i. */
         for ( qto=0; qto<num_toprocessors; ++qto ) {
            pto = toprocessor[qto];
            if ( j_global >= row_starts[pto] && j_global<row_starts[pto+1] ) {
               assert( qto < len_num_rdofs_toprocessor );
               ++num_rdofs_toprocessor[qto];
               /* ... an overestimate, as if two chords share an rdof, that
                  rdof will be counted twice in num_rdofs_toprocessor.
                  It can be fixed up later.*/
               toproc[i] = qto;
               break;
            }
         }
      }
   };
   num_rdofs += hypre_NumbersNEntered(rdofs);
   hypre_NumbersDeleteNode(rdofs);

   for ( q=0; q<num_inprocessors; ++q ) if (q!=my_q) {
      inchord_idof[q] = hypre_CTAlloc( int, num_inchords[q] );
      inchord_rdof[q] = hypre_CTAlloc( int, num_inchords[q] );
      inchord_data[q] = hypre_CTAlloc( double, num_inchords[q] );
      chord[q] = 0;
   };
   for ( q=0; q<num_inprocessors; ++q ) if (q!=my_q) {
      for ( i=0; i<num_inchords[q]; ++i ) {
         inchord_idof[q][i] = -1;
      }
   };
   rdof_toprocessor = hypre_CTAlloc( int*, num_toprocessors );
   for ( qto=0; qto<num_toprocessors; ++qto )  /*if (qto!=my_q)*/ {
      assert( qto < len_num_rdofs_toprocessor );
      rdof_toprocessor[qto] = hypre_CTAlloc( int, num_rdofs_toprocessor[qto] );
      chordto[qto] = 0;
   };
   for ( row=0; row<hypre_CSRMatrixNumRows(offd); ++row ) {
      for ( i=hypre_CSRMatrixI(offd)[row]; i<hypre_CSRMatrixI(offd)[row+1]; ++i ) {
         j_local = hypre_CSRMatrixJ(offd)[i];
         j_global =  col_map_offd[j_local];
         data = hypre_CSRMatrixData(offd)[i];
         qto = toproc[i];
         q = inproc[i];
         assert( q!=my_q );
         assert( chord[q] < num_inchords[q] );
         inchord_idof[q][chord[q]] = row;
         j = j_global - first_index_rdof[q];
         /* This j is local to the processor q - but for diag & offd combined */
         inchord_rdof[q][chord[q]] = j;
         inchord_data[q][chord[q]] = data;
         /* Note that although inchord_* is organized according to the
            inprocessors, the rdof has the local number of a toprocessor -
            the only thing which makes sense and fits with what I've been
            told about chord matrices. */
         assert( chord[q] < num_inchords[q] );
         ++chord[q];
         if ( qto>=0 ) {
            /* There is an rdof processor for this chord */
            rdof_toprocessor[qto][chordto[qto]] = j;
            ++chordto[qto];
         }
      }
   };
   /* fix up overestimate of num_rdofs_toprocessor.  We're not going to
      bother to fix the excessive size which has been allocated to
      rdof_toprocessor... */
   for ( qto=0; qto<num_toprocessors; ++qto )  /*if (qto!=my_q)*/ {
      num_rdofs_toprocessor[qto] = chordto[qto] - 1;
   }
   hypre_NumbersDeleteNode( offd_cols_me );
   for ( q=0; q<num_inprocessors; ++q )
      hypre_NumbersDeleteNode( offd_cols[q]);
   hypre_TFree( offd_cols );
   for ( q=0; q<num_inprocessors; ++q )
      hypre_TFree( offd_col_array[q] );
   hypre_TFree( offd_col_array );
   hypre_TFree( len_offd_col_array );
   hypre_TFree( chordto );
   hypre_TFree( inproc );
   hypre_TFree( toproc );
   hypre_TFree( chord );
   hypre_TFree( pcr );


   hypre_ParChordMatrixFirstindexIdof(Ac) = first_index_idof;
   hypre_ParChordMatrixFirstindexRdof(Ac) = first_index_rdof;

   hypre_ParChordMatrixNumInchords(Ac) = num_inchords;
   hypre_ParChordMatrixInchordIdof(Ac) = inchord_idof;
   hypre_ParChordMatrixInchordRdof(Ac) = inchord_rdof;
   hypre_ParChordMatrixInchordData(Ac) = inchord_data;
   hypre_ParChordMatrixNumIdofs(Ac) = num_idofs;
   hypre_ParChordMatrixNumRdofs(Ac) = num_rdofs;
   hypre_ParChordMatrixNumRdofsToprocessor(Ac) = num_rdofs_toprocessor;
   hypre_ParChordMatrixRdofToprocessor(Ac) = rdof_toprocessor;
      

/* >>> to set...

   hypre_ParChordMatrixNumIdofsInprocessor(Ac)  (low priority - not used);
   hypre_ParChordMatrixIdofInprocessor(Ac)  (low priority - not used);
*/

   }


