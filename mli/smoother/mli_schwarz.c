/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include "parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_smoother.h"

extern int MLI_Smoother_Apply_Schwarz(void *smoother_obj, MLI_Vector *f, 
                                      MLI_Vector *u);
extern int MLI_Smoother_ComposedOverlappedMatrix(hypre_ParCSRMatrix *A,
                 int *off_nrows, int **off_row_lengths, int **off_cols,
                 double **off_vals);

/******************************************************************************
 * Schwarz relaxation scheme 
 *****************************************************************************/

typedef struct MLI_Smoother_Schwarz_Struct
{
   MLI_Matrix *Amat;
   int        nblocks, *block_lengths, **block_indices;
   double     **block_inverses;
} 
MLI_Smoother_Schwarz;

/*--------------------------------------------------------------------------
 * MLI_Smoother_Destroy_Schwarz
 *--------------------------------------------------------------------------*/

void MLI_Smoother_Destroy_Schwarz(void *smoother_obj)
{
   int                  i;
   MLI_Smoother_Schwarz *schwarz_smoother;

   schwarz_smoother = (MLI_Smoother_Schwarz *) smoother_obj;
   if ( schwarz_smoother != NULL ) 
   {
      if ( schwarz_smoother->Amat != NULL ) 
         MLI_Matrix_Destroy( schwarz_smoother->Amat );
      if (schwarz_smoother->block_lengths != NULL) 
         hypre_TFree(schwarz_smoother->block_lengths);
      if (schwarz_smoother->block_indices != NULL) 
      {
         for ( i = 0; i < schwarz_smoother->nblocks; i++ )
            if (schwarz_smoother->block_indices[i] != NULL) 
               hypre_TFree(schwarz_smoother->block_lengths[i]);
         hypre_TFree(schwarz_smoother->block_lengths);
      }
      if (schwarz_smoother->block_inverses != NULL) 
      {
         for ( i = 0; i < schwarz_smoother->nblocks; i++ )
            if (schwarz_smoother->block_inverses[i] != NULL) 
               hypre_TFree(schwarz_smoother->block_inverses[i]);
         hypre_TFree(schwarz_smoother->block_inverses);
      }
      hypre_TFree( schwarz_smoother );
   }
   return;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Setup_Schwarz
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Setup_Schwarz(void *smoother_obj, MLI_Matrix *Amat)
{
   int                  n, *partition, mypid, num_procs, start_row, end_row;
   int                  row, row_length, *col_indices, local_nrows;
   int                  ext_nrows, num_cols_offd;
   int                  off_nrows, *off_row_lengths, *off_cols;
   double               *off_vals, *col_values;
   MPI_Comm             comm;
   MLI_Smoother_Schwarz *schwarz_smoother;
   MLI_Smoother         *generic_smoother;
   hypre_ParCSRMatrix   *A;
   hypre_CSRMatrix      *A_diag, *A_offd;
   int                  *A_diag_i, *A_diag_j, *A_offd_i, *A_offd_j;
   double               *A_diag_data, *A_offd_data;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   A = Amat->matrix;
   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&num_procs);  
   
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   start_row = partition[mypid];
   end_row   = partition[mypid+1] - 1;
   local_nrows = end_row - start_row + 1;

   /*-----------------------------------------------------------------
    * fetch the extended (to other processors) portion of the matrix 
    *-----------------------------------------------------------------*/

   if ( num_procs > 1 )
      MLI_Smoother_ComposedOverlappedMatrix(A, &off_nrows, &off_row_lengths, 
                                            &off_cols, &off_vals);

   /*-----------------------------------------------------------------
    * construct a ParaSails matrix
    *-----------------------------------------------------------------*/

   A_diag        = hypre_ParCSRMatrixDiag(A);
   n             = hypre_CSRMatrixNumRows(A_diag);
   A_diag_i      = hypre_CSRMatrixI(A_diag);
   A_diag_j      = hypre_CSRMatrixJ(A_diag);
   A_diag_data   = hypre_CSRMatrixData(A_diag);
   A_offd        = hypre_ParCSRMatrixOffd(A);
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   A_offd_i      = hypre_CSRMatrixI(A_offd);
   A_offd_j      = hypre_CSRMatrixJ(A_offd);
   A_offd_data   = hypre_CSRMatrixData(A_offd);
   for (row = start_row; row <= end_row; row++)
   {
      hypre_ParCSRMatrixGetRow(A, row, &row_length, &col_indices, &col_values);
      hypre_ParCSRMatrixRestoreRow(A,row,&row_length,&col_indices,&col_values);
   }

   /*-----------------------------------------------------------------
    * construct a Schwarz smoother object
    *-----------------------------------------------------------------*/

   schwarz_smoother = hypre_CTAlloc( MLI_Smoother_Schwarz, 1 );
   if ( schwarz_smoother == NULL ) { return 1; }
   schwarz_smoother->Amat = Amat;

   /*-----------------------------------------------------------------
    * clean up and return object and function
    *-----------------------------------------------------------------*/

   generic_smoother               = (MLI_Smoother *) smoother_obj;
   generic_smoother->object       = (void *) schwarz_smoother;
   generic_smoother->apply_func   = MLI_Smoother_Apply_Schwarz;
   generic_smoother->destroy_func = MLI_Smoother_Destroy_Schwarz;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_Schwarz
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_Schwarz(void *smoother_obj, MLI_Vector *f_in, 
                               MLI_Vector *u_in)
{
   hypre_ParCSRMatrix   *A;
   hypre_CSRMatrix      *A_diag;
   hypre_ParVector      *Vtemp, *f, *u;
   hypre_Vector         *u_local, *Vtemp_local;
   double               *u_data, *Vtemp_data;
   int                  i, n, relax_error = 0, global_size;
   int                  num_procs, *partition1, *partition2;
   int                  parasails_factorized;
   double               *tmp_data;
   MPI_Comm             comm;
   MLI_Smoother_Schwarz *smoother;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   smoother      = (MLI_Smoother_Schwarz *) smoother_obj;
   A             = smoother->Amat->matrix;
   comm          = hypre_ParCSRMatrixComm(A);
   A_diag        = hypre_ParCSRMatrixDiag(A);
   n             = hypre_CSRMatrixNumRows(A_diag);
   u             = u_in->vector;
   u_local       = hypre_ParVectorLocalVector(u);
   u_data        = hypre_VectorData(u_local);

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   f           = f_in->vector;
   global_size = hypre_ParVectorGlobalSize(f);
   partition1  = hypre_ParVectorPartitioning(f);
   partition2  = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partition2[i] = partition1[i];
   Vtemp = hypre_ParVectorCreate(comm, global_size, partition2);
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * perform smoothing
    *-----------------------------------------------------------------*/

   hypre_ParVectorCopy(f, Vtemp);
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   tmp_data = hypre_CTAlloc( double, n );

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   hypre_TFree( tmp_data );

   return(relax_error); 
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_ComposedOverlappedMatrix
 *--------------------------------------------------------------------------*/

int MLI_Smoother_ComposedOverlappedMatrix(hypre_ParCSRMatrix *A,
                 int *off_nrows, int **off_row_lengths, int **off_cols,
                 double **off_vals)
{
   MPI_Comm    comm;
   MPI_Request *requests;
   MPI_Status  *status;
   int         i, j, k, mypid, num_procs, *partition, start_row, end_row;
   int         local_nrows, ext_nrows, num_sends, *send_procs, num_recvs;
   int         *recv_procs, *recv_starts, proc, offset, length, req_num; 
   int         total_send_nnz, total_recv_nnz, index, base, total_sends;
   int         total_recvs, row_num, row_length, *col_ind, *send_starts;
   int         limit, *isend_buf, *cols, cur_nnz; 
   double      *dsend_buf, *vals, *col_val;
   hypre_ParCSRCommPkg *comm_pkg;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters (off_offset)
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&num_procs);  
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   start_row = partition[mypid];
   end_row   = partition[mypid+1] - 1;
   local_nrows = end_row - start_row + 1;

   /*-----------------------------------------------------------------
    * fetch matrix communication information (off_nrows)
    *-----------------------------------------------------------------*/

   ext_nrows = local_nrows;
   if ( num_procs > 1 )
   {
      comm_pkg    = hypre_ParCSRMatrixCommPkg(A);
      num_sends   = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_procs  = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      num_recvs   = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs  = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      for ( i = 0; i < num_recvs; i++ ) 
         ext_nrows += ( recv_starts[i+1] - recv_starts[i] );
      requests = hypre_CTAlloc( MPI_Request, num_recvs+num_sends );
      total_sends  = send_starts[num_sends];
      total_recvs  = recv_starts[num_recvs];
      (*off_nrows) = total_recvs;
   }
   else num_recvs = num_sends = (*off_nrows) = total_recvs = total_sends = 0;

   /*-----------------------------------------------------------------
    * construct off_row_lengths 
    *-----------------------------------------------------------------*/

   req_num = 0;
   for (i = 0; i < num_recvs; i++)
   {
      proc   = recv_procs[i];
      offset = recv_starts[i];
      length = recv_starts[i+1] - offset;
      MPI_Irecv(off_row_lengths[offset], length, MPI_INT, proc, 0, comm, 
                &requests[req_num++]);
   }
   if ( total_sends > 0 ) isend_buf = hypre_CTAlloc( int, total_sends );
   index = total_send_nnz = 0;
   for (i = 0; i < num_sends; i++)
   {
      proc   = send_procs[i];
      offset = send_starts[i];
      limit  = send_starts[i+1];
      length = limit - offset;
      for (j = offset; j < limit; j++)
      {
         row_num = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         hypre_ParCSRMatrixGetRow(A,row_num,&row_length,&col_ind,NULL);
         isend_buf[index++] = row_length;
         total_send_nnz += row_length;
         hypre_ParCSRMatrixRestoreRow(A,row_num,&row_length,&col_ind,NULL);
      }
      MPI_Isend(&isend_buf[offset], length, MPI_INT, proc, 0, comm, 
                &requests[req_num++]);
   }
   status = hypre_CTAlloc(MPI_Status, req_num);
   MPI_Waitall( req_num, requests, status );
   hypre_TFree( status );
   if ( total_sends > 0 ) hypre_TFree( isend_buf );

   /*-----------------------------------------------------------------
    * construct off_cols 
    *-----------------------------------------------------------------*/

   total_recv_nnz = 0;
   for (i = 0; i < total_recvs; i++) total_recv_nnz += (*off_row_lengths)[i];
   if ( total_recv_nnz > 0 )
   {
      cols = hypre_CTAlloc( int, total_recv_nnz );
      vals = hypre_CTAlloc( double, total_recv_nnz );
   }
   req_num = total_recv_nnz = 0;
   for (i = 0; i < num_recvs; i++)
   {
      proc    = recv_procs[i];
      offset  = recv_starts[i];
      length  = recv_starts[i+1] - offset;
      cur_nnz = 0;
      for (j = 0; j < length; j++) cur_nnz += (*off_row_lengths)[offset+j];
      MPI_Irecv(&cols[total_recv_nnz], cur_nnz, MPI_INT, proc, 0, comm, 
                &requests[req_num++]);
      total_recv_nnz += cur_nnz;
   }
   if ( total_send_nnz > 0 )
   {
      isend_buf = hypre_CTAlloc( int, total_send_nnz );
   }
   index = total_send_nnz = 0;
   for (i = 0; i < num_sends; i++)
   {
      proc   = send_procs[i];
      offset = send_starts[i];
      limit  = send_starts[i+1];
      length = limit - offset;
      base   = total_send_nnz;
      for (j = offset; j < limit; j++)
      {
         row_num = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         hypre_ParCSRMatrixGetRow(A,row_num,&row_length,&col_ind,NULL);
         for (k = 0; k < row_length; k++) 
            isend_buf[total_send_nnz++] = col_ind[k];
         hypre_ParCSRMatrixRestoreRow(A,row_num,&row_length,&col_ind,NULL);
      }
      length = total_send_nnz - base;
      MPI_Isend(&isend_buf[base], length, MPI_INT, proc, 0, comm, 
                &requests[req_num++]);
   }
   status = hypre_CTAlloc(MPI_Status, req_num);
   MPI_Waitall( req_num, requests, status );
   hypre_TFree( status );
   if ( total_send_nnz > 0 ) hypre_TFree( isend_buf );

   /*-----------------------------------------------------------------
    * construct off_vals 
    *-----------------------------------------------------------------*/

   req_num = total_recv_nnz = 0;
   for (i = 0; i < num_recvs; i++)
   {
      proc    = recv_procs[i];
      offset  = recv_starts[i];
      length  = recv_starts[i+1] - offset;
      cur_nnz = 0;
      for (j = 0; j < length; j++) cur_nnz += (*off_row_lengths)[offset+j];
      MPI_Irecv(&vals[total_recv_nnz], cur_nnz, MPI_DOUBLE, proc, 0, comm, 
                &requests[req_num++]);
      total_recv_nnz += cur_nnz;
   }
   if ( total_send_nnz > 0 )
   {
      dsend_buf = hypre_CTAlloc( double, total_send_nnz );
   }
   index = total_send_nnz = 0;
   for (i = 0; i < num_sends; i++)
   {
      proc   = send_procs[i];
      offset = send_starts[i];
      limit  = send_starts[i+1];
      length = limit - offset;
      base   = total_send_nnz;
      for (j = offset; j < limit; j++)
      {
         row_num = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         hypre_ParCSRMatrixGetRow(A,row_num,&row_length,NULL,&col_val);
         for (k = 0; k < row_length; k++) 
            dsend_buf[total_send_nnz++] = col_val[k];
         hypre_ParCSRMatrixRestoreRow(A,row_num,&row_length,NULL,&col_val);
      }
      length = total_send_nnz - base;
      MPI_Isend(&dsend_buf[base], length, MPI_DOUBLE, proc, 0, comm, 
                &requests[req_num++]);
   }
   status = hypre_CTAlloc(MPI_Status, req_num);
   MPI_Waitall( req_num, requests, status );
   hypre_TFree( status );
   if ( total_send_nnz > 0 ) hypre_TFree( dsend_buf );

   if ( num_procs > 1 ) hypre_TFree( requests );

   if ( total_recv_nnz > 0 )
   {
      hypre_TFree( cols );
      hypre_TFree( vals );
   }
   return 0;
}

