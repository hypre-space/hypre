/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_DDILUT interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_MHMatrix.h"

#ifdef HAVE_ML

#include "ml_struct.h"
#include "ml_aggregate.h"

#endif

#include "HYPRE_MHMatrix.h"
#include "HYPRE_FEI.h"

extern int HYPRE_LSI_MLConstructMHMatrix(HYPRE_ParCSRMatrix,MH_Matrix *,
                                     MPI_Comm, int *, MH_Context *);
extern int HYPRE_LSI_DDIlutGetRowLengths(MH_Matrix *,int *, int **,MPI_Comm);
extern int HYPRE_LSI_DDIlutGetOffProcRows(MH_Matrix *Amat, int leng, int *,
                 int Noffset, int *map, int *map2, int **int_buf,
                 double **dble_buf, MPI_Comm mpi_comm);
extern int HYPRE_LSI_DDIlutDecompose(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
                 int total_recv_leng, int *recv_lengths, int *ext_ja,
                 double *ext_aa, int *map, int *map2, int Noffset);
extern int HYPRE_LSI_DDIlutDecompose2(HYPRE_LSI_DDIlut *ilut_ptr,
                 MH_Matrix *Amat,int total_recv_leng, int *recv_lengths,
                 int *ext_ja, double *ext_aa, int *map, int *map2, int Noffset);
extern void HYPRE_LSI_qsort1a(int *, int *, int, int);
extern void hypre_qsort0(int *, int, int);
extern int  HYPRE_LSI_SplitDSort(double*,int,int*,int);
extern int  MH_ExchBdry(double *, void *);
extern int  MH_ExchBdryBack(double *, void *, int *, double **, int **);
extern int  MH_GetRow(void *, int, int *, int, int *, double *, int *);
extern int  HYPRE_LSI_Cuthill(int, int *, int *, double *, int *, int *);
extern int  HYPRE_LSI_Search(int *, int, int);

#define habs(x) ((x) > 0 ? (x) : -(x))

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutCreate - Return a DDIlut preconditioner object "solver".
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_DDIlut *ilut_ptr;

   ilut_ptr = hypre_TAlloc(HYPRE_LSI_DDIlut, 1, HYPRE_MEMORY_HOST);

   if (ilut_ptr == NULL) return 1;

   ilut_ptr->comm          = comm;
   ilut_ptr->mh_mat        = NULL;
   ilut_ptr->fillin        = 0.0;
   ilut_ptr->thresh        = 0.0; /* defaults */
   ilut_ptr->mat_ia        = NULL;
   ilut_ptr->mat_ja        = NULL;
   ilut_ptr->mat_aa        = NULL;
   ilut_ptr->outputLevel   = 0;
   ilut_ptr->overlap       = 0;
   ilut_ptr->order_array   = NULL;
   ilut_ptr->reorder_array = NULL;
   ilut_ptr->reorder       = 0;

   *solver = (HYPRE_Solver) ilut_ptr;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutDestroy - Destroy a DDIlut object.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutDestroy( HYPRE_Solver solver )
{
   int              i;
   HYPRE_LSI_DDIlut *ilut_ptr;

   ilut_ptr = (HYPRE_LSI_DDIlut *) solver;
   hypre_TFree(ilut_ptr->mat_ia, HYPRE_MEMORY_HOST);
   hypre_TFree(ilut_ptr->mat_ja, HYPRE_MEMORY_HOST);
   hypre_TFree(ilut_ptr->mat_aa, HYPRE_MEMORY_HOST);
   if ( ilut_ptr->mh_mat != NULL )
   {
      hypre_TFree(ilut_ptr->mh_mat->sendProc, HYPRE_MEMORY_HOST);
      hypre_TFree(ilut_ptr->mh_mat->sendLeng, HYPRE_MEMORY_HOST);
      hypre_TFree(ilut_ptr->mh_mat->recvProc, HYPRE_MEMORY_HOST);
      hypre_TFree(ilut_ptr->mh_mat->recvLeng, HYPRE_MEMORY_HOST);
      for ( i = 0; i < ilut_ptr->mh_mat->sendProcCnt; i++ )
         hypre_TFree(ilut_ptr->mh_mat->sendList[i], HYPRE_MEMORY_HOST);
      hypre_TFree(ilut_ptr->mh_mat->sendList, HYPRE_MEMORY_HOST);
      hypre_TFree(ilut_ptr->mh_mat, HYPRE_MEMORY_HOST);
   }
   ilut_ptr->mh_mat = NULL;
   hypre_TFree(ilut_ptr->order_array, HYPRE_MEMORY_HOST);
   hypre_TFree(ilut_ptr->reorder_array, HYPRE_MEMORY_HOST);
   hypre_TFree(ilut_ptr, HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutSetFillin - Set the fill-in parameter.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSetFillin(HYPRE_Solver solver, double fillin)
{
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;

   ilut_ptr->fillin = fillin;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutSetDropTolerance - Set the threshold for dropping
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSetDropTolerance(HYPRE_Solver solver, double thresh)
{
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;

   ilut_ptr->thresh = thresh;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutSetOverlap - turn on overlap
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSetOverlap(HYPRE_Solver solver)
{
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;

   ilut_ptr->overlap = 1;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutSetReorder - turn on reordering
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSetReorder(HYPRE_Solver solver)
{
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;

   ilut_ptr->reorder = 1;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutSetOutputLevel - Set debug level
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSetOutputLevel(HYPRE_Solver solver, int level)
{
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;

   ilut_ptr->outputLevel = level;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutSolve - Solve function for DDILUT.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int              i, j, *idiag, Nrows, extNrows, *mat_ia, *mat_ja;
   int              column, *order_list, *reorder_list, order_flag;
   double           *rhs, *soln, *dbuffer, ddata, *mat_aa;
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;
   MH_Context       *context;
   MPI_Comm         mpi_comm;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   Nrows        = ilut_ptr->Nrows;
   extNrows     = ilut_ptr->extNrows;
   mat_ia       = ilut_ptr->mat_ia;
   mat_ja       = ilut_ptr->mat_ja;
   mat_aa       = ilut_ptr->mat_aa;
   order_list   = ilut_ptr->order_array;
   reorder_list = ilut_ptr->reorder_array;
   order_flag   = ilut_ptr->reorder;

   dbuffer = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   idiag   = hypre_TAlloc(int, extNrows , HYPRE_MEMORY_HOST);
   for ( i = 0; i < Nrows; i++ ) dbuffer[i] = rhs[i];

   HYPRE_ParCSRMatrixGetComm(A, &mpi_comm);
   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->Amat = ilut_ptr->mh_mat;
   context->comm = mpi_comm;

   if ( extNrows > Nrows ) MH_ExchBdry(dbuffer, context);
   if ( order_flag )
      for ( i = 0; i < Nrows; i++ ) dbuffer[i] = rhs[order_list[i]];
   else
      for ( i = 0; i < Nrows; i++ ) dbuffer[i] = rhs[i];

   for ( i = 0; i < extNrows; i++ )
   {
      ddata = 0.0;
      for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
      {
         column = mat_ja[j];
         if ( column == i ) { idiag[i] = j; break;}
         ddata += mat_aa[j] * dbuffer[column];
      }
      dbuffer[i] -= ddata;
   }
   for ( i = extNrows-1; i >= 0; i-- )
   {
      ddata = 0.0;
      for ( j = idiag[i]+1; j < mat_ia[i+1]; j++ )
      {
         column = mat_ja[j];
         ddata += mat_aa[j] * dbuffer[column];
      }
      dbuffer[i] -= ddata;
      dbuffer[i] /= mat_aa[idiag[i]];
   }
   if ( order_flag )
      for ( i = 0; i < Nrows; i++ ) soln[i] = dbuffer[reorder_list[i]];
   else
      for ( i = 0; i < Nrows; i++ ) soln[i] = dbuffer[i];
   hypre_TFree(dbuffer, HYPRE_MEMORY_HOST);
   hypre_TFree(idiag, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutSetup - Set up function for LSI_DDIlut.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_csr,
                          HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int              i, j, offset, total_recv_leng, *recv_lengths=NULL;
   int              *int_buf=NULL, mypid, nprocs, *parray;
   int              *map=NULL, *map2=NULL, *row_partition=NULL,*parray2;
   double           *dble_buf=NULL;
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;
   MH_Context       *context=NULL;
   MH_Matrix        *mh_mat=NULL;
   MPI_Comm         mpi_comm;

   /* ---------------------------------------------------------------- */
   /* get the row information in my processors                         */
   /* ---------------------------------------------------------------- */

   HYPRE_ParCSRMatrixGetComm(A_csr, &mpi_comm);
   MPI_Comm_rank(mpi_comm, &mypid);
   MPI_Comm_size(mpi_comm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &row_partition);

   /* ---------------------------------------------------------------- */
   /* convert the incoming CSR matrix into a MH matrix                 */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->comm = mpi_comm;
   context->globalEqns = row_partition[nprocs];
   context->partition = hypre_TAlloc(int, (nprocs+1), HYPRE_MEMORY_HOST);
   for (i=0; i<=nprocs; i++) context->partition[i] = row_partition[i];
   hypre_TFree( row_partition , HYPRE_MEMORY_HOST);
   mh_mat = hypre_TAlloc( MH_Matrix, 1, HYPRE_MEMORY_HOST);
   context->Amat = mh_mat;
   HYPRE_LSI_MLConstructMHMatrix(A_csr,mh_mat,mpi_comm,
                                 context->partition,context);

   /* ---------------------------------------------------------------- */
   /* compose the enlarged overlapped local matrix                     */
   /* ---------------------------------------------------------------- */

   if ( ilut_ptr->overlap != 0 )
   {
      HYPRE_LSI_DDIlutComposeOverlappedMatrix(mh_mat, &total_recv_leng,
                 &recv_lengths, &int_buf, &dble_buf, &map, &map2,&offset,
                 mpi_comm);
   }
   else
   {
      total_recv_leng = 0;
      recv_lengths = NULL;
      int_buf = NULL;
      dble_buf = NULL;
      map = NULL;
      map2 = NULL;
      parray  = hypre_TAlloc(int, nprocs , HYPRE_MEMORY_HOST);
      parray2 = hypre_TAlloc(int, nprocs , HYPRE_MEMORY_HOST);
      for ( i = 0; i < nprocs; i++ ) parray2[i] = 0;
      parray2[mypid] = mh_mat->Nrows;
      MPI_Allreduce(parray2,parray,nprocs,MPI_INT,MPI_SUM,mpi_comm);
      offset = 0;
      for (i = 0; i < mypid; i++) offset += parray[i];
      hypre_TFree(parray, HYPRE_MEMORY_HOST);
      hypre_TFree(parray2, HYPRE_MEMORY_HOST);
   }

   /* ---------------------------------------------------------------- */
   /* perform ILUT decomposition on local matrix                       */
   /* ---------------------------------------------------------------- */

   if ( ilut_ptr->mat_ia == NULL )
      HYPRE_LSI_DDIlutDecompose(ilut_ptr,mh_mat,total_recv_leng,recv_lengths,
                                int_buf, dble_buf, map,map2, offset);
   else
   {
      HYPRE_LSI_DDIlutDecompose2(ilut_ptr,mh_mat,total_recv_leng,recv_lengths,
                                 int_buf, dble_buf, map,map2, offset);
      if ( mypid == 0 && ilut_ptr->outputLevel >= 1 )
         printf("DDILUT : preconditioner pattern reused.\n");
   }
   if ( mypid == 0 && ilut_ptr->outputLevel > 2 )
   {
      for ( i = 0; i < ilut_ptr->extNrows; i++ )
         for ( j = ilut_ptr->mat_ia[i]; j < ilut_ptr->mat_ia[i+1]; j++ )
            printf("LA(%d,%d) = %e;\n", i+1, ilut_ptr->mat_ja[j]+1,
                   ilut_ptr->mat_aa[j]);
   }

   ilut_ptr->mh_mat = mh_mat;
   hypre_TFree(mh_mat->rowptr, HYPRE_MEMORY_HOST);
   hypre_TFree(mh_mat->colnum, HYPRE_MEMORY_HOST);
   hypre_TFree(mh_mat->values, HYPRE_MEMORY_HOST);
   hypre_TFree(map, HYPRE_MEMORY_HOST);
   hypre_TFree(map2, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(dble_buf , HYPRE_MEMORY_HOST);
   hypre_TFree(recv_lengths, HYPRE_MEMORY_HOST);
   hypre_TFree(context->partition, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);
   return 0;
}

/*****************************************************************************/
/* subroutines used for constructing overlapped matrix                       */
/*****************************************************************************/

int HYPRE_LSI_DDIlutGetRowLengths(MH_Matrix *Amat, int *leng, int **recv_leng,
                                  MPI_Comm mpi_comm)
{
   int         i, j, m, mypid, index, *temp_list, allocated_space, length;
   int         nRecv, *recvProc, *recvLeng, *cols, total_recv, mtype, msgtype;
   int         nSend, *sendProc, *sendLeng, **sendList, proc_id, offset;
   double      *vals;
   MPI_Request *Request;
   MPI_Status  status;
   MH_Context  *context;

   /* ---------------------------------------------------------------- */
   /* fetch communication information                                  */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(mpi_comm, &mypid);
   nRecv    = Amat->recvProcCnt;
   nSend    = Amat->sendProcCnt;
   recvProc = Amat->recvProc;
   recvLeng = Amat->recvLeng;
   sendProc = Amat->sendProc;
   sendLeng = Amat->sendLeng;
   sendList = Amat->sendList;
   total_recv = 0;
   for ( i = 0; i < nRecv; i++ ) total_recv += recvLeng[i];

   (*leng) = total_recv;
   if ( nRecv <= 0 ) (*recv_leng) = NULL;

   MPI_Barrier(mpi_comm);

   /* ---------------------------------------------------------------- */
   /* post receives for all messages                                   */
   /* ---------------------------------------------------------------- */

   (*recv_leng) = hypre_TAlloc(int, total_recv , HYPRE_MEMORY_HOST);
   if (nRecv > 0) Request = hypre_TAlloc(MPI_Request, nRecv, HYPRE_MEMORY_HOST);
   offset = 0;
   mtype = 2001;
   for (i = 0; i < nRecv; i++)
   {
      proc_id = recvProc[i];
      msgtype = mtype;
      length  = recvLeng[i];
      MPI_Irecv((void *) &((*recv_leng)[offset]), length, MPI_INT, proc_id,
               msgtype, mpi_comm, &Request[i]);
      offset += length;
   }

   /* ---------------------------------------------------------------- */
   /* write out all messages                                           */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->Amat = Amat;
   allocated_space = 100;
   cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
   vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
   for (i = 0; i < nSend; i++)
   {
      proc_id   = sendProc[i];
      length    = sendLeng[i];
      temp_list = hypre_TAlloc(int, sendLeng[i] , HYPRE_MEMORY_HOST);
      for (j = 0; j < length; j++)
      {
         index = sendList[i][j];
         while (MH_GetRow(context,1,&index,allocated_space,cols,vals,&m)==0)
         {
            hypre_TFree(cols, HYPRE_MEMORY_HOST);
            hypre_TFree(vals, HYPRE_MEMORY_HOST);
            allocated_space += 200 + 1;
            cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
            vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
         }
         temp_list[j] = m;
      }
      msgtype = mtype;
      MPI_Send((void*)temp_list,length,MPI_INT,proc_id,msgtype,mpi_comm);
      hypre_TFree(temp_list, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* wait for messages                                                */
   /* ---------------------------------------------------------------- */

   for ( i = 0; i < nRecv; i++ )
   {
      MPI_Wait( &Request[i], &status );
   }

   if (nRecv > 0)
      hypre_TFree(Request, HYPRE_MEMORY_HOST);
   return 0;
}

/*****************************************************************************/
/* needed for overlapped smoothers                                           */
/*****************************************************************************/

int HYPRE_LSI_DDIlutGetOffProcRows(MH_Matrix *Amat, int leng, int *recv_leng,
                           int Noffset, int *map, int *map2, int **int_buf,
                           double **dble_buf, MPI_Comm mpi_comm)
{
   int         i, j, k, m, length, offset, allocated_space, proc_id;
   int         nRecv, nSend, *recvProc, *sendProc, total_recv, mtype, msgtype;
   int         *sendLeng, *recvLeng, **sendList, *cols, *isend_buf, Nrows;
   int         nnz, nnz_offset, index, mypid;
   double      *vals, *send_buf;
   MPI_Request *request;
   MPI_Status  status;
   MH_Context  *context;

   /* ---------------------------------------------------------------- */
   /* fetch communication information                                  */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(mpi_comm, &mypid);
   Nrows    = Amat->Nrows;
   nRecv    = Amat->recvProcCnt;
   nSend    = Amat->sendProcCnt;
   recvProc = Amat->recvProc;
   recvLeng = Amat->recvLeng;
   sendProc = Amat->sendProc;
   sendLeng = Amat->sendLeng;
   sendList = Amat->sendList;
   if ( nRecv <= 0 ) { (*int_buf) = NULL; (*dble_buf) = NULL;}
   total_recv = 0;
   for ( i = 0; i < leng; i++ ) total_recv += recv_leng[i];

   /* ---------------------------------------------------------------- */
   /* allocate buffer space                                            */
   /* ---------------------------------------------------------------- */

   if ( nRecv > 0 )
        request     = hypre_TAlloc(MPI_Request , nRecv, HYPRE_MEMORY_HOST);
   else request = NULL;

   if ( total_recv > 0 )
   {
      (*int_buf)  = hypre_TAlloc(int, total_recv , HYPRE_MEMORY_HOST);
      (*dble_buf) = hypre_TAlloc(double, total_recv , HYPRE_MEMORY_HOST);
   }

   /* ---------------------------------------------------------------- */
   /* post receives for all messages                                   */
   /* ---------------------------------------------------------------- */

   offset     = 0;
   mtype      = 2002;
   nnz_offset = 0;
   for (i = 0; i < nRecv; i++)
   {
      proc_id = recvProc[i];
      msgtype = mtype;
      length  = recvLeng[i];
      nnz = 0;
      for (j = 0; j < length; j++)  nnz += recv_leng[offset+j];

      MPI_Irecv((void *) &((*dble_buf)[nnz_offset]), nnz, MPI_DOUBLE,
               proc_id, msgtype, mpi_comm, request+i);
      offset += length;
      nnz_offset += nnz;
   }

   /* ---------------------------------------------------------------- */
   /* send rows to other processors                                    */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->Amat = Amat;
   mtype = 2002;
   allocated_space = 100;
   cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
   vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
   for (i = 0; i < nSend; i++)
   {
      proc_id   = sendProc[i];
      length    = sendLeng[i];
      nnz       = 0;
      for (j = 0; j < length; j++)
      {
         index = sendList[i][j];
         while (MH_GetRow(context,1,&index,allocated_space,cols,vals,&m)==0)
         {
            hypre_TFree(cols, HYPRE_MEMORY_HOST);
            hypre_TFree(vals, HYPRE_MEMORY_HOST);
            allocated_space += 200 + 1;
            cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
            vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
         }
         nnz += m;
      }
      if ( nnz > 0 ) send_buf = hypre_TAlloc(double,  nnz , HYPRE_MEMORY_HOST);
      offset = 0;
      for (j = 0; j < length; j++)
      {
         index = sendList[i][j];
         MH_GetRow(context,1,&index,allocated_space,cols,vals,&m);
         for (k = 0; k < m; k++) send_buf[offset+k] = vals[k];
         offset += m;
      }
      msgtype = mtype;
      MPI_Send((void*) send_buf, nnz, MPI_DOUBLE, proc_id, msgtype,
                       mpi_comm);
      if ( nnz > 0 )
         hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* wait for all messages                                            */
   /* ---------------------------------------------------------------- */

   for (i = 0; i < nRecv; i++)
   {
      MPI_Wait(request+i, &status);
   }

   /* ----------------------------------------------------------- */
   /* post receives for all messages                              */
   /* ----------------------------------------------------------- */

   mtype  = 2003;
   offset = 0;
   nnz_offset = 0;
   for (i = 0; i < nRecv; i++)
   {
      proc_id = recvProc[i];
      msgtype = mtype;
      length  = recvLeng[i];
      nnz = 0;
      for (j = 0; j < length; j++)  nnz += recv_leng[offset+j];
      MPI_Irecv((void *) &((*int_buf)[nnz_offset]), nnz, MPI_INT,
                   proc_id, msgtype, mpi_comm, request+i);
      offset += length;
      nnz_offset += nnz;
   }

   /* ---------------------------------------------------------------- */
   /* send rows to other processors                                    */
   /* ---------------------------------------------------------------- */

   mtype = 2003;
   cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
   vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
   for (i = 0; i < nSend; i++)
   {
      proc_id   = sendProc[i];
      length    = sendLeng[i];
      nnz       = 0;
      for (j = 0; j < length; j++)
      {
         index = sendList[i][j];
         MH_GetRow(context,1,&index,allocated_space,cols,vals,&m);
         nnz += m;
      }
      if ( nnz > 0 ) isend_buf = hypre_TAlloc(int,  nnz , HYPRE_MEMORY_HOST);
      offset = 0;
      for (j = 0; j < length; j++)
      {
         index = sendList[i][j];
         MH_GetRow(context,1,&index,allocated_space,cols,vals,&m);
         for (k = 0; k < m; k++)
         {
            if ( cols[k] < Nrows ) isend_buf[offset+k] = cols[k] + Noffset;
            else                   isend_buf[offset+k] = map[cols[k]-Nrows];
         }
         offset += m;
      }
      msgtype = mtype;
      MPI_Send((void*) isend_buf, nnz, MPI_INT, proc_id, msgtype,
                       mpi_comm);
      if ( nnz > 0 )
         hypre_TFree(isend_buf, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);

   /* ----------------------------------------------------------- */
   /* post receives for all messages                              */
   /* ----------------------------------------------------------- */

   for (i = 0; i < nRecv; i++)
   {
      MPI_Wait(request+i, &status);
   }

   hypre_TFree(request, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);
   return 0;
}

/*****************************************************************************/
/* construct an enlarged overlapped local matrix                             */
/*****************************************************************************/

int HYPRE_LSI_DDIlutComposeOverlappedMatrix(MH_Matrix *mh_mat,
              int *total_recv_leng, int **recv_lengths, int **int_buf,
              double **dble_buf, int **sindex_array, int **sindex_array2,
              int *offset, MPI_Comm mpi_comm)
{
   int        i, nprocs, mypid, Nrows, *proc_array, *proc_array2;
   int        extNrows, NrowsOffset, *index_array, *index_array2;
   int        nRecv, *recvLeng;
   double     *dble_array;
   MH_Context *context;

   /* ---------------------------------------------------------------- */
   /* fetch communication information                                  */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(mpi_comm, &mypid);
   MPI_Comm_size(mpi_comm, &nprocs);

   /* ---------------------------------------------------------------- */
   /* fetch matrix information                                         */
   /* ---------------------------------------------------------------- */

   nRecv    = mh_mat->recvProcCnt;
   recvLeng = mh_mat->recvLeng;
   Nrows    = mh_mat->Nrows;

   /* ---------------------------------------------------------------- */
   /* compute the enlarged matrix size                                 */
   /* ---------------------------------------------------------------- */

   (*total_recv_leng) = 0;
   for ( i = 0; i < nRecv; i++ ) (*total_recv_leng) += recvLeng[i];
   extNrows = Nrows + (*total_recv_leng);

   /* ---------------------------------------------------------------- */
   /* compose NrowsOffset and processor offsets proc_array             */
   /* ---------------------------------------------------------------- */

   proc_array  = hypre_TAlloc(int, nprocs , HYPRE_MEMORY_HOST);
   proc_array2 = hypre_TAlloc(int, nprocs , HYPRE_MEMORY_HOST);
   for ( i = 0; i < nprocs; i++ ) proc_array2[i] = 0;
   proc_array2[mypid] = Nrows;
   MPI_Allreduce(proc_array2,proc_array,nprocs,MPI_INT,MPI_SUM,mpi_comm);
   NrowsOffset = 0;
   for (i = 0; i < mypid; i++) NrowsOffset += proc_array[i];
   for (i = 1; i < nprocs; i++) proc_array[i] += proc_array[i-1];
   hypre_TFree(proc_array2, HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* compose the column index map (index_array,index_array2)          */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->comm = mpi_comm;
   context->Amat = mh_mat;
   dble_array  = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   for (i = Nrows; i < extNrows; i++) dble_array[i] = 0.0;
   for (i = 0; i < Nrows; i++) dble_array[i] = 1.0 * ( i + NrowsOffset );
   MH_ExchBdry(dble_array, context);
   if ( extNrows-Nrows > 0 )
      index_array = hypre_TAlloc(int, (extNrows-Nrows) , HYPRE_MEMORY_HOST);
   else
      index_array = NULL;
   for (i = Nrows; i < extNrows; i++) index_array[i-Nrows] = dble_array[i];
   if ( extNrows-Nrows > 0 )
      index_array2  = hypre_TAlloc(int, (extNrows-Nrows) , HYPRE_MEMORY_HOST);
   else
      index_array2 = NULL;
   for (i = 0; i < extNrows-Nrows; i++) index_array2[i] = i;
   hypre_TFree(dble_array, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* send the lengths of each row to remote processor                 */
   /* at the end, additional row information should be given           */
   /* in total_recv_leng, recv_lengths, int_buf, dble_buf              */
   /* ---------------------------------------------------------------- */

   HYPRE_LSI_DDIlutGetRowLengths(mh_mat,total_recv_leng,recv_lengths,mpi_comm);
   HYPRE_LSI_DDIlutGetOffProcRows(mh_mat, *total_recv_leng, *recv_lengths,
              NrowsOffset,index_array,index_array2,int_buf, dble_buf,mpi_comm);

   hypre_TFree(proc_array, HYPRE_MEMORY_HOST);
   HYPRE_LSI_qsort1a(index_array, index_array2, 0, extNrows-Nrows-1);
   (*sindex_array) = index_array;
   (*sindex_array2) = index_array2;
   (*offset) = NrowsOffset;
   return 0;
}

/*****************************************************************************/
/* function for doing ILUT decomposition                                     */
/* ( based on ILU(0) + ILUT based on magnitude)                              */
/*****************************************************************************/

int HYPRE_LSI_DDIlutDecompose(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
           int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa,
           int *map, int *map2, int Noffset)
{
   int          *mat_ia, *mat_ja, i, m, allocated_space, *cols, mypid;
   int          index, first, Lcount, Ucount, j, k, total_nnz;
   int          sortcnt, colIndex, offset, nnz_count, Nrows, extNrows;
   int          *track_array, track_leng, num_small_pivot, printstep, nnz_row;
   int          *sortcols, *Amat_ia, *Amat_ja, *order_list, *reorder_list;
   int          max_nnz_row, touch_cnt=0, order_flag;
   double       *vals, ddata, *mat_aa, *diagonal, *rowNorms, *Norm2;
   double       *dble_buf, fillin, tau, rel_tau, *sortvals, *Amat_aa;
   MH_Context   *context;

   /* ---------------------------------------------------------------- */
   /* fetch ILUT parameters                                            */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(ilut_ptr->comm, &mypid);
   fillin   = ilut_ptr->fillin;
   tau      = ilut_ptr->thresh;
   Nrows    = Amat->Nrows;
   extNrows = Nrows + total_recv_leng;
   ilut_ptr->Nrows = Nrows;
   ilut_ptr->extNrows = extNrows;
   order_flag = ilut_ptr->reorder;

   /* ---------------------------------------------------------------- */
   /* allocate temporary storage space                                 */
   /* ---------------------------------------------------------------- */

   allocated_space = extNrows;
   cols     = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
   vals     = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
   sortcols = hypre_TAlloc(int, extNrows , HYPRE_MEMORY_HOST);
   sortvals = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   dble_buf = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   diagonal = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   rowNorms = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* compute the storage requirement for the ILU matrix               */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->Amat = Amat;
   total_nnz     = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      rowNorms[i] = 0.0;
      while (MH_GetRow(context,1,&i,allocated_space,cols,vals,&m)==0)
      {
         hypre_TFree(vals, HYPRE_MEMORY_HOST);
         hypre_TFree(cols, HYPRE_MEMORY_HOST);
         allocated_space += 200 + 1;
         cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
         vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
      }
      total_nnz += m;
      for ( j = 0; j < m; j++ ) rowNorms[i] += habs(vals[j]);
      rowNorms[i] /= extNrows;
   }
   hypre_TFree(vals, HYPRE_MEMORY_HOST);
   hypre_TFree(cols, HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* permute the matrix                                               */
   /* ---------------------------------------------------------------- */

   Amat_ia      = hypre_TAlloc(int,  (Nrows+1) , HYPRE_MEMORY_HOST);
   Amat_ja      = hypre_TAlloc(int,  total_nnz , HYPRE_MEMORY_HOST);
   Amat_aa      = hypre_TAlloc(double,  total_nnz , HYPRE_MEMORY_HOST);
   total_nnz    = 0;
   Amat_ia[0]   = total_nnz;
   for ( i = 0; i < Nrows; i++ )
   {
      MH_GetRow(context,1,&i,allocated_space,&Amat_ja[total_nnz],
                &Amat_aa[total_nnz],&m);
      total_nnz    += m;
      Amat_ia[i+1] = total_nnz;
   }

   if ( order_flag )
   {
      order_list   = hypre_TAlloc(int,  Nrows , HYPRE_MEMORY_HOST);
      reorder_list = hypre_TAlloc(int,  Nrows , HYPRE_MEMORY_HOST);
      for ( i = 0; i < Nrows; i++ ) order_list[i] = reorder_list[i] = i;
      HYPRE_LSI_Cuthill(Nrows,Amat_ia,Amat_ja,Amat_aa,order_list,reorder_list);
      ilut_ptr->order_array = order_list;
      ilut_ptr->reorder_array = reorder_list;
      Norm2 = hypre_TAlloc(double, Nrows , HYPRE_MEMORY_HOST);
      for ( i = 0; i < Nrows; i++ ) Norm2[i] = rowNorms[order_list[i]];
      hypre_TFree(rowNorms, HYPRE_MEMORY_HOST);
      rowNorms = Norm2;
   }
   /*
   for ( i = 0; i < Nrows; i++ )
      for ( j = Amat_ia[i]; j < Amat_ia[i+1]; j++ )
         printf("%10d %10d %25.16e\n", i+1, Amat_ja[j]+1, Amat_aa[j]);
   */

   /* ---------------------------------------------------------------- */
   /* allocate space                                                   */
   /* ---------------------------------------------------------------- */

   for ( i = 0; i < total_recv_leng; i++ ) total_nnz += recv_lengths[i];
   total_nnz = (int) ((double) total_nnz * (fillin + 1.0));
   ilut_ptr->mat_ia = hypre_TAlloc(int,  (extNrows + 1 ) , HYPRE_MEMORY_HOST);
   ilut_ptr->mat_ja = hypre_TAlloc(int,  total_nnz , HYPRE_MEMORY_HOST);
   ilut_ptr->mat_aa = hypre_TAlloc(double,  total_nnz , HYPRE_MEMORY_HOST);
   mat_ia = ilut_ptr->mat_ia;
   mat_ja = ilut_ptr->mat_ja;
   mat_aa = ilut_ptr->mat_aa;

   offset = 0;
   max_nnz_row = 0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      rowNorms[i+Nrows] = 0.0;
      nnz_row = 0;
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         index = ext_ja[j];
         if ( index >= Noffset && index < Noffset+Nrows )
            ext_ja[j] = index - Noffset;
         else
         {
            m = HYPRE_LSI_Search(map, index, extNrows-Nrows);
            if ( m >= 0 ) ext_ja[j] = map2[m] + Nrows;
            else          ext_ja[j] = -1;
         }
         if ( ext_ja[j] != -1 )
         {
            rowNorms[i+Nrows] += habs(ext_aa[j]);
            nnz_row++;
         }
      }
      if ( nnz_row > max_nnz_row ) max_nnz_row = nnz_row;
      rowNorms[i+Nrows] /= extNrows;
      offset += recv_lengths[i];
   }

   /* ---------------------------------------------------------------- */
   /* process the first Nrows                                          */
   /* ---------------------------------------------------------------- */

   num_small_pivot = 0;
   nnz_count = 0;
   mat_ia[0] = 0;
   track_array = hypre_TAlloc(int,  extNrows , HYPRE_MEMORY_HOST);
   for ( i = 0; i < extNrows; i++ ) dble_buf[i] = 0.0;

   printstep = extNrows /  10;

   for ( i = 0; i < Nrows; i++ )
   {
      if ( i % printstep == 0 && ilut_ptr->outputLevel > 0 )
         printf("%4d : 0DDILUT Processing row %d(%d)\n",mypid,i,extNrows);

      track_leng = 0;
      cols = &(Amat_ja[Amat_ia[i]]);
      vals = &(Amat_aa[Amat_ia[i]]);
      m    = Amat_ia[i+1] - Amat_ia[i];

      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] < extNrows )
         {
            dble_buf[cols[j]] = vals[j];
            track_array[track_leng++] = cols[j];
         }
      }
      Lcount = Ucount = first = 0;
      first  = extNrows;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( dble_buf[index] != 0 )
         {
            if ( index < i ) Lcount++;
            else if ( index > i ) Ucount++;
            else if ( index == i ) diagonal[i] = dble_buf[index];
            if ( index < first ) first = index;
         }
      }
      Lcount  = Lcount * fillin;
      Ucount  = Ucount * fillin;
      rel_tau = tau * rowNorms[i];
      for ( j = first; j < i; j++ )
      {
         if ( habs(dble_buf[j]) > rel_tau )
         {
            ddata = dble_buf[j] / diagonal[j];
touch_cnt++;
            for ( k = mat_ia[j]; k < mat_ia[j+1]; k++ )
            {
               colIndex = mat_ja[k];
               if ( colIndex > j )
               {
                  if ( dble_buf[colIndex] != 0.0 )
                     dble_buf[colIndex] -= (ddata * mat_aa[k]);
                  else
                  {
                     dble_buf[colIndex] = - (ddata * mat_aa[k]);
                     if ( dble_buf[colIndex] != 0.0 )
                        track_array[track_leng++] = colIndex;
                  }
               }
            }
            dble_buf[j] = ddata;
         }
         else dble_buf[j] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] < extNrows )
         {
            vals[j] = dble_buf[cols[j]];
            if ( cols[j] != i ) dble_buf[cols[j]] = 0.0;
         }
      }
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i )
         {
            if ( dble_buf[index] < -rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = - dble_buf[index] * rowNorms[index];
            }
            else if ( dble_buf[index] > rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = dble_buf[index] * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Lcount )
      {
         HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Lcount);
         for ( j = Lcount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] < i && vals[j] != 0.0 )
         {
            mat_aa[nnz_count] = vals[j];
            mat_ja[nnz_count++] = cols[j];
         }
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      diagonal[i] = dble_buf[i];
      if ( habs(diagonal[i]) < 1.0e-16 )
      {
         diagonal[i] = 1.0E-6;
         num_small_pivot++;
      }
      mat_aa[nnz_count] = diagonal[i];
      mat_ja[nnz_count++] = i;
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i )
         {
            if ( dble_buf[index] < -rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = - dble_buf[index] * rowNorms[index];
            }
            else if ( dble_buf[index] > rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = dble_buf[index] * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Ucount )
      {
         HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Ucount);
         for ( j = Ucount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] > i && vals[j] != 0.0 )
         {
            mat_aa[nnz_count] = vals[j];
            mat_ja[nnz_count++] = cols[j];
         }
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      dble_buf[i] = 0.0;
      mat_ia[i+1] = nnz_count;
   }
   hypre_TFree(Amat_ia, HYPRE_MEMORY_HOST);
   hypre_TFree(Amat_ja, HYPRE_MEMORY_HOST);
   hypre_TFree(Amat_aa, HYPRE_MEMORY_HOST);
   printf("touch_cnt = %d\n", touch_cnt);

   /* ---------------------------------------------------------------- */
   /* process the off-processor rows                                   */
   /* ---------------------------------------------------------------- */

   offset = 0;
   cols = hypre_TAlloc(int,  max_nnz_row , HYPRE_MEMORY_HOST);
   vals = hypre_TAlloc(double,  max_nnz_row , HYPRE_MEMORY_HOST);
   for ( i = 0; i < extNrows; i++ ) dble_buf[i] = 0.0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      if ( (i+Nrows) % printstep == 0 && ilut_ptr->outputLevel > 0 )
         printf("%4d : *DDILUT Processing row %d(%d)\n",mypid,i+Nrows,extNrows);

      track_leng = m = 0;
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         if ( ext_ja[j] != -1 )
         {
            if (order_flag && ext_ja[j] < Nrows) index = reorder_list[ext_ja[j]];
            else                                 index = ext_ja[j];
            dble_buf[index] = ext_aa[j];
            track_array[track_leng++] = index;
            cols[m] = index;
            vals[m++] = ext_aa[j];
         }
      }
      Lcount = Ucount = 0;
      first  = extNrows;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( dble_buf[index] != 0 )
         {
            if ( index < i+Nrows ) Lcount++;
            else if ( index > i+Nrows ) Ucount++;
            else if ( i+Nrows == index ) diagonal[i+Nrows] = dble_buf[index];
            if ( index < first ) first = index;
         }
      }
      Lcount  = Lcount * fillin;
      Ucount  = Ucount * fillin;
      rel_tau = tau * rowNorms[i+Nrows];
      for ( j = first; j < i+Nrows; j++ )
      {
         if ( habs(dble_buf[j]) > rel_tau )
         {
            ddata = dble_buf[j] / diagonal[j];
            for ( k = mat_ia[j]; k < mat_ia[j+1]; k++ )
            {
               colIndex = mat_ja[k];
               if ( colIndex > j )
               {
                  if ( dble_buf[colIndex] != 0.0 )
                     dble_buf[colIndex] -= (ddata * mat_aa[k]);
                  else
                  {
                     dble_buf[colIndex] = - (ddata * mat_aa[k]);
                     if ( dble_buf[colIndex] != 0.0 )
                        track_array[track_leng++] = colIndex;
                  }
               }
            }
            dble_buf[j] = ddata;
         }
         else dble_buf[j] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] < extNrows )
         {
            vals[j] = dble_buf[cols[j]];
            if ( cols[j] != i+Nrows ) dble_buf[cols[j]] = 0.0;
         }
      }
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i+Nrows )
         {
            if ( dble_buf[index] < -rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = - dble_buf[index]*rowNorms[index];
            }
            else if ( dble_buf[index] > rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = dble_buf[index] * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Lcount )
      {
         HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Lcount);
         for ( j = Lcount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] < i+Nrows && vals[j] != 0.0 )
         {
            mat_aa[nnz_count] = vals[j];
            mat_ja[nnz_count++] = cols[j];
         }
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i+Nrows && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      diagonal[i+Nrows] = dble_buf[i+Nrows];
      if ( habs(diagonal[i+Nrows]) < 1.0e-16 )
      {
         diagonal[i+Nrows] = 1.0E-6;
         num_small_pivot++;
      }
      mat_aa[nnz_count] = diagonal[i+Nrows];
      mat_ja[nnz_count++] = i+Nrows;
      dble_buf[i+Nrows] = 0.0;
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i+Nrows )
         {
            if ( dble_buf[index] < -rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = - dble_buf[index] * rowNorms[index];
            }
            else if ( dble_buf[index] > rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = dble_buf[index] * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Ucount )
      {
         HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Ucount);
         for ( j = Ucount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] > i+Nrows && cols[j] < extNrows && vals[j] != 0.0 )
         {
            mat_aa[nnz_count] = vals[j];
            mat_ja[nnz_count++] = cols[j];
         }
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i+Nrows && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      mat_ia[i+Nrows+1] = nnz_count;
      offset += recv_lengths[i];
   }
   if ( nnz_count > total_nnz )
      printf("WARNING in ILUTDecomp : memory bound passed.\n");
   if ( ilut_ptr->outputLevel > 0 )
   {
      printf("%4d :  DDILUT number of nonzeros     = %d\n",mypid,nnz_count);
      printf("%4d :  DDILUT number of small pivots = %d\n",mypid,num_small_pivot);
   }

   /* ---------------------------------------------------------- */
   /* deallocate temporary storage space                         */
   /* ---------------------------------------------------------- */

   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);
   hypre_TFree(sortcols, HYPRE_MEMORY_HOST);
   hypre_TFree(sortvals, HYPRE_MEMORY_HOST);
   hypre_TFree(dble_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal, HYPRE_MEMORY_HOST);
   hypre_TFree(rowNorms, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);
   hypre_TFree(track_array, HYPRE_MEMORY_HOST);

   return 0;
}

/*****************************************************************************/
/* function for doing ILUT decomposition                                     */
/* (attempted for pattern reuse, but not done yet)                           */
/*****************************************************************************/

int HYPRE_LSI_DDIlutDecompose2(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
           int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa,
           int *map, int *map2, int Noffset)
{
   int          *mat_ia, *mat_ja, i, m, allocated_space, *cols, mypid;
   int          index, first, Lcount, Ucount, ncnt, j, k, total_nnz;
   int          sortcnt, colIndex, offset, nnz_count, Nrows, extNrows;
   int          *track_array, track_leng, num_small_pivot, printstep, ndisc;
   int          *sortcols;
   double       *vals, ddata, *mat_aa, *diagonal, *rowNorms;
   double       *dble_buf, fillin, tau, rel_tau, *sortvals, absval;
   MH_Context   *context;

   /* ---------------------------------------------------------------- */
   /* fetch ILUT parameters                                            */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(ilut_ptr->comm, &mypid);
   fillin   = ilut_ptr->fillin;
   tau      = ilut_ptr->thresh;
   Nrows    = Amat->Nrows;
   extNrows = Nrows + total_recv_leng;
   ilut_ptr->Nrows = Nrows;
   ilut_ptr->extNrows = extNrows;

   /* ---------------------------------------------------------------- */
   /* allocate temporary storage space                                 */
   /* ---------------------------------------------------------------- */

   allocated_space = extNrows;
   cols     = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
   vals     = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
   sortcols = hypre_TAlloc(int, extNrows , HYPRE_MEMORY_HOST);
   sortvals = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   dble_buf = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   diagonal = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   rowNorms = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* compute the storage requirement for the ILU matrix               */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->Amat = Amat;
   for ( i = 0; i < Nrows; i++ )
   {
      rowNorms[i] = 0.0;
      while (MH_GetRow(context,1,&i,allocated_space,cols,vals,&m)==0)
      {
         hypre_TFree(vals, HYPRE_MEMORY_HOST);
         hypre_TFree(cols, HYPRE_MEMORY_HOST);
         allocated_space += 200 + 1;
         cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
         vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
      }
      for ( j = 0; j < m; j++ ) rowNorms[i] += habs(vals[j]);
      rowNorms[i] /= extNrows;
   }
   total_nnz = 0;
   for ( i = 0; i < total_recv_leng; i++ ) total_nnz += recv_lengths[i];
   total_nnz = (int) ((double) total_nnz*(fillin+1.0)) + ilut_ptr->mat_ia[Nrows];
   mat_ia = ilut_ptr->mat_ia;
   mat_ja = ilut_ptr->mat_ja;
   mat_aa = ilut_ptr->mat_aa;
   ilut_ptr->mat_ia = hypre_TAlloc(int,  (extNrows + 1 ) , HYPRE_MEMORY_HOST);
   ilut_ptr->mat_ja = hypre_TAlloc(int,  total_nnz , HYPRE_MEMORY_HOST);
   ilut_ptr->mat_aa = hypre_TAlloc(double,  total_nnz , HYPRE_MEMORY_HOST);

   ncnt = 0;
   ilut_ptr->mat_ia[0] = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
         if ( mat_ja[j] >= 0 && mat_ja[j] < extNrows )
            ilut_ptr->mat_ja[ncnt++] = mat_ja[j];
      ilut_ptr->mat_ia[i+1] = ncnt;
   }
   hypre_TFree(mat_ia, HYPRE_MEMORY_HOST);
   hypre_TFree(mat_ja, HYPRE_MEMORY_HOST);
   hypre_TFree(mat_aa, HYPRE_MEMORY_HOST);
   mat_ia = ilut_ptr->mat_ia;
   mat_ja = ilut_ptr->mat_ja;
   mat_aa = ilut_ptr->mat_aa;

   /* ---------------------------------------------------------------- */
   /* process the first Nrows                                          */
   /* ---------------------------------------------------------------- */

   num_small_pivot = 0;
   mat_ia[0] = 0;
   track_array = hypre_TAlloc(int,  extNrows , HYPRE_MEMORY_HOST);
   for ( i = 0; i < extNrows; i++ ) dble_buf[i] = 0.0;

   printstep = extNrows /  10;

   ndisc = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      if ( i % printstep == 0 && ilut_ptr->outputLevel > 0 )
         printf("%4d : 1DDILUT Processing row %d(%d,%d)\n",mypid,i,extNrows,Nrows);

      MH_GetRow(context,1,&i,allocated_space,cols,vals,&m);

      /* ------------------------------------------------------------- */
      /* load the row into buffer                                      */
      /* ------------------------------------------------------------- */

      track_leng = 0;
      first      = extNrows;
      for ( j = 0; j < m; j++ )
      {
         index = cols[j];
         if ( index < extNrows )
         {
            dble_buf[index] = vals[j];
            track_array[track_leng++] = index;
         }
         if ( index < first ) first = index;
      }
      for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
      {
         index = mat_ja[j];
         if ( dble_buf[index] == 0.0 ) track_array[track_leng++] = index;
         if ( index < first ) first = index;
      }
if ( (mat_ia[i+1]-mat_ia[i]) != track_leng) ndisc++;

      /* ------------------------------------------------------------- */
      /* perform factorization                                         */
      /* ------------------------------------------------------------- */

      rel_tau = tau * rowNorms[i];
      for ( j = first; j < i; j++ )
      {
         if ( habs(dble_buf[j]) > rel_tau )
         {
            ddata = dble_buf[j] / diagonal[j];
            for ( k = mat_ia[j]; k < mat_ia[j+1]; k++ )
            {
               colIndex = mat_ja[k];
               if ( colIndex > j )
               {
                  if ( dble_buf[colIndex] != 0.0 )
                     dble_buf[colIndex] -= (ddata * mat_aa[k]);
                  else
                  {
                     dble_buf[colIndex] = - (ddata * mat_aa[k]);
                     if ( dble_buf[colIndex] != 0.0 )
                        track_array[track_leng++] = colIndex;
                  }
               }
            }
            dble_buf[j] = ddata;
         }
         else dble_buf[j] = 0.0;
      }

      diagonal[i] = dble_buf[i];
      if ( habs(diagonal[i]) < 1.0e-16 )
      {
         diagonal[i] = dble_buf[i] = 1.0E-6;
         num_small_pivot++;
      }
      for (j = mat_ia[i]; j < mat_ia[i+1]; j++) mat_aa[j] = dble_buf[mat_ja[j]];
      for ( j = 0; j < track_leng; j++ ) dble_buf[track_array[j]] = 0.0;
   }
   nnz_count = mat_ia[Nrows];
   ncnt = 0;
   k = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      for ( j = k; j < mat_ia[i+1]; j++ )
      {
         if ( mat_aa[j] != 0.0 )
         {
            mat_ja[ncnt] = mat_ja[j];
            mat_aa[ncnt++] = mat_aa[j];
         }
      }
      k = mat_ia[i+1];
      mat_ia[i+1] = ncnt;
   }
   if ( ilut_ptr->outputLevel > 0 )
   {
      printf("%4d :  DDILUT after Nrows - nnz = %d %d\n", mypid, nnz_count, ncnt);
      printf("%4d :  DDILUT number of small pivots = %d\n",mypid,num_small_pivot);
      printf("%4d :  DDILUT number of pattern mismatch = %d\n",mypid,ndisc);
   }
   nnz_count = ncnt;

   /* ---------------------------------------------------------------- */
   /* preparation for processing the off-processor rows                */
   /* ---------------------------------------------------------------- */

   offset = 0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      rowNorms[i+Nrows] = 0.0;
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         index = ext_ja[j];
         if ( index >= Noffset && index < Noffset+Nrows )
            ext_ja[j] = index - Noffset;
         else
         {
            m = HYPRE_LSI_Search(map, index, extNrows-Nrows);
            if ( m >= 0 ) ext_ja[j] = map2[m] + Nrows;
            else          ext_ja[j] = -1;
            if ( ext_ja[j] >= extNrows ) ext_ja[j] = -1;
         }
         if ( ext_ja[j] != -1 ) rowNorms[i+Nrows] += habs(ext_aa[j]);
      }
      rowNorms[i+Nrows] /= extNrows;
      offset += recv_lengths[i];
   }

   /* ---------------------------------------------------------------- */
   /* process the off-processor rows                                   */
   /* ---------------------------------------------------------------- */

   offset = 0;
   for ( i = 0; i < extNrows; i++ ) dble_buf[i] = 0.0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      if ( (i+Nrows) % printstep == 0 && ilut_ptr->outputLevel > 0 )
         printf("%4d : *DDILUT Processing row %d(%d)\n",mypid,i+Nrows,extNrows);

      track_leng = m = 0;
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         index = ext_ja[j];
         if ( index != -1 )
         {
            cols[m] = index;
            vals[m++] = ext_aa[j];
            track_array[track_leng++] = index;
            dble_buf[index] = ext_aa[j];
         }
      }
      Lcount = Ucount = 0;
      first  = extNrows;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( dble_buf[index] != 0 )
         {
            if ( index < i+Nrows ) Lcount++;
            else if ( index > i+Nrows ) Ucount++;
            else if ( i+Nrows == index ) diagonal[i+Nrows] = dble_buf[index];
            if ( index < first ) first = index;
         }
      }
      Lcount  = Lcount * fillin;
      Ucount  = Ucount * fillin;
      rel_tau = tau * rowNorms[i+Nrows];
      for ( j = first; j < i+Nrows; j++ )
      {
         if ( habs(dble_buf[j]) > rel_tau )
         {
            ddata = dble_buf[j] / diagonal[j];
            for ( k = mat_ia[j]; k < mat_ia[j+1]; k++ )
            {
               colIndex = mat_ja[k];
               if ( colIndex > j )
               {
                  if ( dble_buf[colIndex] != 0.0 )
                     dble_buf[colIndex] -= (ddata * mat_aa[k]);
                  else
                  {
                     dble_buf[colIndex] = - (ddata * mat_aa[k]);
                     if ( dble_buf[colIndex] != 0.0 )
                        track_array[track_leng++] = colIndex;
                  }
               }
            }
            dble_buf[j] = ddata;
         }
         else dble_buf[j] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         index = cols[j];
         vals[j] = dble_buf[index];
         if ( index != i+Nrows ) dble_buf[index] = 0.0;
      }
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i+Nrows )
         {
            absval = habs( dble_buf[index] );
            if ( absval > rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = absval * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Lcount )
      {
         HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Lcount);
         for ( j = Lcount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] < i+Nrows && vals[j] != 0.0 )
         {
            mat_aa[nnz_count] = vals[j];
            mat_ja[nnz_count++] = cols[j];
         }
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i+Nrows && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      diagonal[i+Nrows] = dble_buf[i+Nrows];
      if ( habs(diagonal[i+Nrows]) < 1.0e-16 )
      {
         diagonal[i+Nrows] = 1.0E-6;
         num_small_pivot++;
      }
      mat_aa[nnz_count] = diagonal[i+Nrows];
      mat_ja[nnz_count++] = i+Nrows;
      dble_buf[i+Nrows] = 0.0;
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i+Nrows )
         {
            absval = habs( dble_buf[index] );
            if ( absval > rel_tau )
            {
               sortcols[sortcnt] = index;
               sortvals[sortcnt++] = absval * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Ucount )
      {
         HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Ucount);
         for ( j = Ucount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
      }
      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] > i+Nrows && vals[j] != 0.0 )
         {
            mat_aa[nnz_count] = vals[j];
            mat_ja[nnz_count++] = cols[j];
         }
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i+Nrows && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      mat_ia[i+Nrows+1] = nnz_count;
      offset += recv_lengths[i];
   }

   if ( nnz_count > total_nnz )
      printf("WARNING in ILUTDecomp : memory bound passed.\n");
   if ( ilut_ptr->outputLevel > 0 )
   {
      printf("%4d :  DDILUT number of nonzeros     = %d\n",mypid,nnz_count);
      printf("%4d :  DDILUT number of small pivots = %d\n",mypid,num_small_pivot);
   }

   /* ---------------------------------------------------------- */
   /* deallocate temporary storage space                         */
   /* ---------------------------------------------------------- */

   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);
   hypre_TFree(sortcols, HYPRE_MEMORY_HOST);
   hypre_TFree(sortvals, HYPRE_MEMORY_HOST);
   hypre_TFree(dble_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal, HYPRE_MEMORY_HOST);
   hypre_TFree(rowNorms, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);
   hypre_TFree(track_array, HYPRE_MEMORY_HOST);

   return 0;
}

/*****************************************************************************/
/* function for doing ILUT decomposition                                     */
/* (purely based on magnitude)                                               */
/*****************************************************************************/

int HYPRE_LSI_DDIlutDecompose3(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
           int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa,
           int *map, int *map2, int Noffset)
{
   int          *mat_ia, *mat_ja, i, m, allocated_space, *cols, mypid;
   int          index, first, Lcount, Ucount, j, k, total_nnz;
   int          sortcnt, colIndex, offset, nnz_count, Nrows, extNrows;
   int          *track_array, track_leng, num_small_pivot;
   double       *vals, ddata, *mat_aa, *diagonal, *rowNorms;
   double       *dble_buf, fillin, tau, rel_tau;
   MH_Context   *context;

   /* ---------------------------------------------------------------- */
   /* fetch ILUT parameters                                            */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(ilut_ptr->comm, &mypid);
   fillin   = ilut_ptr->fillin;
   tau      = ilut_ptr->thresh;
   Nrows    = Amat->Nrows;
   extNrows = Nrows + total_recv_leng;
   ilut_ptr->Nrows = Nrows;
   ilut_ptr->extNrows = extNrows;

   /* ---------------------------------------------------------------- */
   /* allocate temporary storage space                                 */
   /* ---------------------------------------------------------------- */

   allocated_space = extNrows;
   cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
   vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
   dble_buf = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   diagonal = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   rowNorms = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* compute the storage requirement for the ILU matrix               */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->Amat = Amat;
   total_nnz     = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      rowNorms[i] = 0.0;
      while (MH_GetRow(context,1,&i,allocated_space,cols,vals,&m)==0)
      {
         hypre_TFree(vals, HYPRE_MEMORY_HOST);
         hypre_TFree(cols, HYPRE_MEMORY_HOST);
         allocated_space += 200 + 1;
         cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
         vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
      }
      total_nnz += m;
      for ( j = 0; j < m; j++ ) rowNorms[i] += habs(vals[j]);
      rowNorms[i] /= extNrows;
   }
   for ( i = 0; i < total_recv_leng; i++ ) total_nnz += recv_lengths[i];
   total_nnz = (int) ((double) total_nnz * (fillin + 1.0));
   ilut_ptr->mat_ia = hypre_TAlloc(int,  (extNrows + 1 ) , HYPRE_MEMORY_HOST);
   ilut_ptr->mat_ja = hypre_TAlloc(int,  total_nnz , HYPRE_MEMORY_HOST);
   ilut_ptr->mat_aa = hypre_TAlloc(double,  total_nnz , HYPRE_MEMORY_HOST);
   mat_ia = ilut_ptr->mat_ia;
   mat_ja = ilut_ptr->mat_ja;
   mat_aa = ilut_ptr->mat_aa;

   offset = 0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      rowNorms[i+Nrows] = 0.0;
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         index = ext_ja[j];
         if ( index >= Noffset && index < Noffset+Nrows )
            ext_ja[j] = index - Noffset;
         else
         {
            m = HYPRE_LSI_Search(map, index, extNrows-Nrows);
            if ( m >= 0 ) ext_ja[j] = map2[m] + Nrows;
            else          ext_ja[j] = -1;
         }
         if ( ext_ja[j] != -1 ) rowNorms[i+Nrows] += habs(ext_aa[j]);
      }
      rowNorms[i+Nrows] /= extNrows;
      offset += recv_lengths[i];
   }

   /* ---------------------------------------------------------------- */
   /* process the first Nrows                                          */
   /* ---------------------------------------------------------------- */

   num_small_pivot = 0;
   nnz_count = 0;
   mat_ia[0] = 0;
   track_array = hypre_TAlloc(int,  extNrows , HYPRE_MEMORY_HOST);
   for ( i = 0; i < extNrows; i++ ) dble_buf[i] = 0.0;

   for ( i = 0; i < Nrows; i++ )
   {
      if ( i % 1000 == 0 && ilut_ptr->outputLevel > 0 )
         printf("%4d : 2DDILUT Processing row %d(%d)\n",mypid,i,extNrows);

      track_leng = 0;
      MH_GetRow(context,1,&i,allocated_space,cols,vals,&m);
      if ( m < 0 )
         printf("IlutDecompose WARNING(1): row nnz = %d\n",m);

      for ( j = 0; j < m; j++ )
      {
         if ( cols[j] < extNrows )
         {
            dble_buf[cols[j]] = vals[j];
            track_array[track_leng++] = cols[j];
         }
      }
      Lcount = Ucount = first = 0;
      first  = extNrows;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( dble_buf[index] != 0 )
         {
            if ( index < i ) Lcount++;
            else if ( index > i ) Ucount++;
            else if ( index == i ) diagonal[i] = dble_buf[index];
            if ( index < first ) first = index;
         }
      }
      Lcount  = Lcount * (fillin + 1);
      Ucount  = Ucount * (fillin + 1);
      rel_tau = tau * rowNorms[i];
      for ( j = first; j < i; j++ )
      {
         if ( habs(dble_buf[j]) > rel_tau )
         {
            ddata = dble_buf[j] / diagonal[j];
            for ( k = mat_ia[j]; k < mat_ia[j+1]; k++ )
            {
               colIndex = mat_ja[k];
               if ( colIndex > j )
               {
                  if ( dble_buf[colIndex] != 0.0 )
                     dble_buf[colIndex] -= (ddata * mat_aa[k]);
                  else
                  {
                     dble_buf[colIndex] = - (ddata * mat_aa[k]);
                     track_array[track_leng++] = colIndex;
                  }
               }
            }
            dble_buf[j] = ddata;
         }
         else dble_buf[j] = 0.0;
      }

      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i )
         {
            if ( dble_buf[index] < -rel_tau )
            {
               cols[sortcnt] = index;
               vals[sortcnt++] = - dble_buf[index] * rowNorms[index];
            }
            else if ( dble_buf[index] > rel_tau )
            {
               cols[sortcnt] = index;
               vals[sortcnt++] = dble_buf[index] * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }

      if ( sortcnt > Lcount ) HYPRE_LSI_SplitDSort(vals,sortcnt,cols,Lcount);
      if ( sortcnt > Lcount )
      {
         for ( j = Lcount; j < sortcnt; j++ ) dble_buf[cols[j]] = 0.0;
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      diagonal[i] = dble_buf[i];
      if ( habs(diagonal[i]) < 1.0e-16 )
      {
         diagonal[i] = 1.0E-6;
         num_small_pivot++;
      }
      mat_aa[nnz_count] = diagonal[i];
      mat_ja[nnz_count++] = i;
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i )
         {
            if ( dble_buf[index] < -rel_tau )
            {
               cols[sortcnt] = index;
               vals[sortcnt++] = - dble_buf[index] * rowNorms[index];
            }
            else if ( dble_buf[index] > rel_tau )
            {
               cols[sortcnt] = index;
               vals[sortcnt++] = dble_buf[index] * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Ucount ) HYPRE_LSI_SplitDSort(vals,sortcnt,cols,Ucount);
      if ( sortcnt > Ucount )
      {
         for ( j = Ucount; j < sortcnt; j++ ) dble_buf[cols[j]] = 0.0;
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      dble_buf[i] = 0.0;
      mat_ia[i+1] = nnz_count;
   }

   /* ---------------------------------------------------------------- */
   /* process the off-processor rows                                   */
   /* ---------------------------------------------------------------- */

   offset = 0;
   for ( i = 0; i < extNrows; i++ ) dble_buf[i] = 0.0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      if ( (i+Nrows) % 1000 == 0 && ilut_ptr->outputLevel > 0 )
         printf("%4d : *DDILUT Processing row %d(%d)\n",mypid,i+Nrows,extNrows);

      track_leng = 0;
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         if ( ext_ja[j] != -1 )
         {
            dble_buf[ext_ja[j]] = ext_aa[j];
            track_array[track_leng++] = ext_ja[j];
         }
      }
      Lcount = Ucount = 0;
      first  = extNrows;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( dble_buf[index] != 0 )
         {
            if ( index < i+Nrows ) Lcount++;
            else if ( index > i+Nrows ) Ucount++;
            else if ( i+Nrows == index ) diagonal[i+Nrows] = dble_buf[index];
            if ( index < first ) first = index;
         }
      }
      Lcount  = Lcount * (fillin + 1);
      Ucount  = Ucount * (fillin + 1);
      rel_tau = tau * rowNorms[i+Nrows];
      for ( j = first; j < i+Nrows; j++ )
      {
         if ( habs(dble_buf[j]) > rel_tau )
         {
            ddata = dble_buf[j] / diagonal[j];
            for ( k = mat_ia[j]; k < mat_ia[j+1]; k++ )
            {
               colIndex = mat_ja[k];
               if ( colIndex > j )
               {
                  if ( dble_buf[colIndex] != 0.0 )
                     dble_buf[colIndex] -= (ddata * mat_aa[k]);
                  else
                  {
                     dble_buf[colIndex] = - (ddata * mat_aa[k]);
                     track_array[track_leng++] = colIndex;
                  }
               }
            }
            dble_buf[j] = ddata;
         }
         else dble_buf[j] = 0.0;
      }
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i+Nrows )
         {
            if ( dble_buf[index] < -rel_tau )
            {
               cols[sortcnt] = index;
               vals[sortcnt++] = - dble_buf[index]*rowNorms[index];
            }
            else if ( dble_buf[index] > rel_tau )
            {
               cols[sortcnt] = index;
               vals[sortcnt++] = dble_buf[index] * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Lcount ) HYPRE_LSI_SplitDSort(vals,sortcnt,cols,Lcount);
      if ( sortcnt > Lcount )
      {
         for ( j = Lcount; j < sortcnt; j++ ) dble_buf[cols[j]] = 0.0;
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i+Nrows && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      diagonal[i+Nrows] = dble_buf[i+Nrows];
      if ( habs(diagonal[i+Nrows]) < 1.0e-16 )
      {
         diagonal[i+Nrows] = 1.0E-6;
         num_small_pivot++;
      }
      mat_aa[nnz_count] = diagonal[i+Nrows];
      mat_ja[nnz_count++] = i+Nrows;
      dble_buf[i+Nrows] = 0.0;
      sortcnt = 0;
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i+Nrows )
         {
            if ( dble_buf[index] < -rel_tau )
            {
               cols[sortcnt] = index;
               vals[sortcnt++] = - dble_buf[index] * rowNorms[index];
            }
            else if ( dble_buf[index] > rel_tau )
            {
               cols[sortcnt] = index;
               vals[sortcnt++] = dble_buf[index] * rowNorms[index];
            }
            else dble_buf[index] = 0.0;
         }
      }
      if ( sortcnt > Ucount ) HYPRE_LSI_SplitDSort(vals,sortcnt,cols,Ucount);
      if ( sortcnt > Ucount )
      {
         for ( j = Ucount; j < sortcnt; j++ ) dble_buf[cols[j]] = 0.0;
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index > i+Nrows && dble_buf[index] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[index];
            mat_ja[nnz_count++] = index;
            dble_buf[index] = 0.0;
         }
      }
      mat_ia[i+Nrows+1] = nnz_count;
      offset += recv_lengths[i];
   }
   if ( nnz_count > total_nnz )
      printf("WARNING in ILUTDecomp : memory bound passed.\n");
   if ( ilut_ptr->outputLevel > 0 )
   {
      printf("%4d :  DDILUT number of nonzeros     = %d\n",mypid,nnz_count);
      printf("%4d :  DDILUT number of small pivots = %d\n",mypid,num_small_pivot);
   }

   /* ---------------------------------------------------------- */
   /* deallocate temporary storage space                         */
   /* ---------------------------------------------------------- */

   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);
   hypre_TFree(dble_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal, HYPRE_MEMORY_HOST);
   hypre_TFree(rowNorms, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);
   hypre_TFree(track_array, HYPRE_MEMORY_HOST);

   return 0;
}

/*****************************************************************************/
/* function for doing ILUT decomposition                                     */
/* (This version is based on ILU(1).  It converges less well as the original */
/*  ILUT based on ILU(0) and magnitude.  Its setup time is not faster either)*/
/*****************************************************************************/

int HYPRE_LSI_DDIlutDecomposeNew(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
           int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa,
           int *map, int *map2, int Noffset)
{
   int          *mat_ia, *mat_ja, i, m, allocated_space, *cols, mypid;
   int          index, first, ncnt, j, k, total_nnz;
   int          colIndex, offset, nnz_count, Nrows, extNrows;
   int          *track_array, track_leng, num_small_pivot, printstep;
   int          *mat_ia2, *mat_ja2, *iarray;
   double       *vals, ddata, *mat_aa, *diagonal, *rowNorms;
   double       *dble_buf, tau, rel_tau, *mat_aa2;
   MH_Context   *context;

   /* ---------------------------------------------------------------- */
   /* fetch ILUT parameters                                            */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(ilut_ptr->comm, &mypid);
   tau      = ilut_ptr->thresh;
   Nrows    = Amat->Nrows;
   extNrows = Nrows + total_recv_leng;
   ilut_ptr->Nrows = Nrows;
   ilut_ptr->extNrows = extNrows;

   /* ---------------------------------------------------------------- */
   /* allocate temporary storage space                                 */
   /* ---------------------------------------------------------------- */

   allocated_space = extNrows;
   cols     = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
   vals     = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
   dble_buf = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   diagonal = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
   rowNorms = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* compute the storage requirement for the ILU matrix               */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->Amat = Amat;
   total_nnz     = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      rowNorms[i] = 0.0;
      while (MH_GetRow(context,1,&i,allocated_space,cols,vals,&m)==0)
      {
         hypre_TFree(vals, HYPRE_MEMORY_HOST);
         hypre_TFree(cols, HYPRE_MEMORY_HOST);
         allocated_space += 200 + 1;
         cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
         vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
      }
      total_nnz += m;
      for ( j = 0; j < m; j++ ) rowNorms[i] += habs(vals[j]);
      rowNorms[i] /= extNrows;
   }
   for ( i = 0; i < total_recv_leng; i++ ) total_nnz += recv_lengths[i];
   mat_ia = hypre_TAlloc(int,  (extNrows + 1 ) , HYPRE_MEMORY_HOST);
   mat_ja = hypre_TAlloc(int,  total_nnz , HYPRE_MEMORY_HOST);
   mat_aa = hypre_TAlloc(double,  total_nnz , HYPRE_MEMORY_HOST);
   total_nnz = total_nnz * 7;
   mat_ia2 = hypre_TAlloc(int,  (extNrows + 1 ) , HYPRE_MEMORY_HOST);
   mat_ja2 = hypre_TAlloc(int,  total_nnz , HYPRE_MEMORY_HOST);
   ncnt = 0;
   mat_ia[0] = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      MH_GetRow(context,1,&i,allocated_space,cols,vals,&m);
      for ( j = 0; j < m; j++ )
      {
         if ( vals[j] != 0.0 )
         {
            mat_ja[ncnt] = cols[j];
            mat_aa[ncnt++] = vals[j];
         }
      }
      mat_ia[i+1] = ncnt;
   }
   offset = 0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      rowNorms[i+Nrows] = 0.0;
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         index = ext_ja[j];
         if ( index >= Noffset && index < Noffset+Nrows )
            ext_ja[j] = index - Noffset;
         else
         {
            m = HYPRE_LSI_Search(map, index, extNrows-Nrows);
            if ( m >= 0 ) ext_ja[j] = map2[m] + Nrows;
            else          ext_ja[j] = -1;
         }
         if ( ext_ja[j] != -1 && ext_aa[j] != 0.0 )
         {
            rowNorms[i+Nrows] += habs(ext_aa[j]);
            mat_ja[ncnt] = ext_ja[j];
            mat_aa[ncnt++] = ext_aa[j];
         }
      }
      rowNorms[i+Nrows] /= extNrows;
      offset += recv_lengths[i];
      mat_ia[Nrows+i+1] = ncnt;
   }

   /* ---------------------------------------------------------------- */
   /* process the pattern                                              */
   /* ---------------------------------------------------------------- */

   ncnt = 0;
   mat_ia2[0] = 0;
   printstep = extNrows /  10;
   for ( i = 0; i < extNrows; i++ )
   {
      if ( ( i % printstep == 0 ) && ilut_ptr->outputLevel > 0 )
         printf("%4d :  DDILUT Processing pattern row = %d (%d)\n",mypid,i,extNrows);
      k = mat_ia[i+1] - mat_ia[i];
      for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
      {
         index = mat_ja[j];
         k += ( mat_ia[index+1] - mat_ia[index] );
      }
      if ( (k+ncnt) > total_nnz )
      {
         iarray = mat_ja2;
         total_nnz += (extNrows - i ) * k;
         mat_ja2 = hypre_TAlloc(int,  total_nnz , HYPRE_MEMORY_HOST);
         for ( j = 0; j < ncnt; j++ ) mat_ja2[j] = iarray[j];
         hypre_TFree(iarray, HYPRE_MEMORY_HOST);
      }
      for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
      {
         index = mat_ja[j];
         mat_ja2[ncnt++] = index;
         for (k = mat_ia[index]; k < mat_ia[index+1]; k++)
            mat_ja2[ncnt++] = mat_ja[k];
      }
      hypre_qsort0(mat_ja2, mat_ia2[i], ncnt-1);
      k = mat_ia2[i] + 1;
      for ( j = mat_ia2[i]+1; j < ncnt; j++ )
      {
         if ( mat_ja2[j] != mat_ja2[k-1] ) mat_ja2[k++] = mat_ja2[j];
      }
      mat_ia2[i+1] = k;
      ncnt = k;
   }
   for ( i = 0; i < ncnt; i++ )
      if ( mat_ja2[i] < 0 || mat_ja2[i] >= extNrows )
         printf("%4d :  DDILUT ERROR  ja %d = %d \n",mypid,i,mat_ja2[i]);

   mat_aa2 = hypre_TAlloc(double,  ncnt , HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* process the rows                                                 */
   /* ---------------------------------------------------------------- */

   num_small_pivot = 0;
   track_array = hypre_TAlloc(int,  extNrows , HYPRE_MEMORY_HOST);
   for ( i = 0; i < extNrows; i++ ) dble_buf[i] = 0.0;

   for ( i = 0; i < extNrows; i++ )
   {
      if ( i % printstep == 0 && ilut_ptr->outputLevel > 0 )
         printf("%4d : $DDILUT Processing row %d(%d,%d)\n",mypid,i,extNrows,Nrows);

      /* ------------------------------------------------------------- */
      /* load the row into buffer                                      */
      /* ------------------------------------------------------------- */

      track_leng = 0;
      first      = extNrows;
      for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
      {
         index = mat_ja[j];
         dble_buf[index] = mat_aa[j];
         track_array[track_leng++] = index;
         if ( index < first ) first = index;
      }

      /* ------------------------------------------------------------- */
      /* perform factorization                                         */
      /* ------------------------------------------------------------- */

      rel_tau = tau * rowNorms[i];
      for ( j = first; j < i; j++ )
      {
         if ( habs(dble_buf[j]) > rel_tau )
         {
            ddata = dble_buf[j] / diagonal[j];
            for ( k = mat_ia2[j]; k < mat_ia2[j+1]; k++ )
            {
               colIndex = mat_ja2[k];
               if ( colIndex > j && mat_aa2[k] != 0.0 )
               {
                  if ( dble_buf[colIndex] != 0.0 )
                     dble_buf[colIndex] -= (ddata * mat_aa2[k]);
                  else
                  {
                     dble_buf[colIndex] = - (ddata * mat_aa2[k]);
                     if ( dble_buf[colIndex] != 0.0 )
                        track_array[track_leng++] = colIndex;
                  }
               }
            }
            dble_buf[j] = ddata;
         }
         else dble_buf[j] = 0.0;
      }
      diagonal[i] = dble_buf[i];
      if ( habs(diagonal[i]) < 1.0e-16 )
      {
         diagonal[i] = dble_buf[i] = 1.0E-6;
         num_small_pivot++;
      }
      for (j = mat_ia2[i]; j < mat_ia2[i+1]; j++)
         mat_aa2[j] = dble_buf[mat_ja2[j]];
      for ( j = 0; j < track_leng; j++ ) dble_buf[track_array[j]] = 0.0;
   }
   nnz_count = mat_ia2[extNrows];

   if ( ilut_ptr->outputLevel > 0 )
   {
      printf("%4d :  DDILUT number of nonzeros     = %d\n",mypid,nnz_count);
      printf("%4d :  DDILUT number of small pivots = %d\n",mypid,num_small_pivot);
   }

   /* ---------------------------------------------------------- */
   /* deallocate temporary storage space                         */
   /* ---------------------------------------------------------- */

   ilut_ptr->mat_ia = mat_ia2;
   ilut_ptr->mat_ja = mat_ja2;
   ilut_ptr->mat_aa = mat_aa2;
   hypre_TFree(mat_ia, HYPRE_MEMORY_HOST);
   hypre_TFree(mat_ja, HYPRE_MEMORY_HOST);
   hypre_TFree(mat_aa, HYPRE_MEMORY_HOST);
   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);
   hypre_TFree(dble_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal, HYPRE_MEMORY_HOST);
   hypre_TFree(rowNorms, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);
   hypre_TFree(track_array, HYPRE_MEMORY_HOST);

   return 0;
}


