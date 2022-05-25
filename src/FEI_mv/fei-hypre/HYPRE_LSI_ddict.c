/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_DDICT interface
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
typedef struct HYPRE_LSI_DDICT_Struct
{
   MPI_Comm  comm;
   MH_Matrix *mh_mat;
   double    thresh;
   double    fillin;
   int       Nrows;
   int       extNrows;
   int       *mat_ja;
   double    *mat_aa;
   int       outputLevel;
}
HYPRE_LSI_DDICT;

extern int  HYPRE_LSI_MLConstructMHMatrix(HYPRE_ParCSRMatrix,MH_Matrix *,
                                     MPI_Comm, int *, MH_Context *);
extern int  HYPRE_LSI_DDICTComposeOverlappedMatrix(MH_Matrix *, int *,
                 int **recv_lengths, int **int_buf, double **dble_buf,
                 int **sindex_array, int **sindex_array2, int *offset);
extern int  HYPRE_LSI_DDICTGetRowLengths(MH_Matrix *Amat, int *leng, int **);
extern int  HYPRE_LIS_DDICTGetOffProcRows(MH_Matrix *Amat, int leng, int *,
                 int Noffset, int *map, int *map2, int **int_buf,
                 double **dble_buf);
extern int  HYPRE_LSI_DDICTDecompose(HYPRE_LSI_DDICT *ict_ptr,MH_Matrix *Amat,
                 int total_recv_leng, int *recv_lengths, int *ext_ja,
                 double *ext_aa, int *map, int *map2, int Noffset);
extern void HYPRE_LSI_qsort1a(int *, int *, int, int);
extern int  HYPRE_LSI_SplitDSort(double *,int,int*,int);
extern int  HYPRE_LSI_Search(int *, int, int);

extern int  HYPRE_LSI_DDICTFactorize(HYPRE_LSI_DDICT *ict_ptr, double *mat_aa,
                 int *mat_ja, int *mat_ia, double *rowNorms);

extern int  MH_ExchBdry(double *, void *);
extern int  MH_ExchBdryBack(double *, void *, int *, double **, int **);
extern int  MH_GetRow(void *, int, int *, int, int *, double *, int *);

#define habs(x) ((x) > 0 ? (x) : -(x))

/*****************************************************************************/
/* HYPRE_LSI_DDICTCreate - Return a DDICT preconditioner object "solver".    */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_DDICT *ict_ptr;

   ict_ptr = hypre_TAlloc(HYPRE_LSI_DDICT, 1, HYPRE_MEMORY_HOST);

   if (ict_ptr == NULL) return 1;

   ict_ptr->comm        = comm;
   ict_ptr->mh_mat      = NULL;
   ict_ptr->fillin      = 0.0;
   ict_ptr->thresh      = 0.0; /* defaults */
   ict_ptr->mat_ja      = NULL;
   ict_ptr->mat_aa      = NULL;
   ict_ptr->outputLevel = 0;

   *solver = (HYPRE_Solver) ict_ptr;

   return 0;
}

/*****************************************************************************/
/* HYPRE_LSI_DDICTDestroy - Destroy a DDICT object.                          */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTDestroy( HYPRE_Solver solver )
{
   int              i;
   HYPRE_LSI_DDICT *ict_ptr;

   ict_ptr = (HYPRE_LSI_DDICT *) solver;
   hypre_TFree(ict_ptr->mat_ja, HYPRE_MEMORY_HOST);
   hypre_TFree(ict_ptr->mat_aa, HYPRE_MEMORY_HOST);
   if ( ict_ptr->mh_mat != NULL )
   {
      hypre_TFree(ict_ptr->mh_mat->sendProc, HYPRE_MEMORY_HOST);
      hypre_TFree(ict_ptr->mh_mat->sendLeng, HYPRE_MEMORY_HOST);
      hypre_TFree(ict_ptr->mh_mat->recvProc, HYPRE_MEMORY_HOST);
      hypre_TFree(ict_ptr->mh_mat->recvLeng, HYPRE_MEMORY_HOST);
      for ( i = 0; i < ict_ptr->mh_mat->sendProcCnt; i++ )
         hypre_TFree(ict_ptr->mh_mat->sendList[i], HYPRE_MEMORY_HOST);
      hypre_TFree(ict_ptr->mh_mat->sendList, HYPRE_MEMORY_HOST);
      hypre_TFree(ict_ptr, HYPRE_MEMORY_HOST);
   }
   ict_ptr->mh_mat = NULL;
   hypre_TFree(ict_ptr, HYPRE_MEMORY_HOST);

   return 0;
}

/*****************************************************************************/
/* HYPRE_LSI_DDICTSetFillin - Set the fill-in parameter.                     */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTSetFillin(HYPRE_Solver solver, double fillin)
{
   HYPRE_LSI_DDICT *ict_ptr = (HYPRE_LSI_DDICT *) solver;

   ict_ptr->fillin = fillin;

   return 0;
}

/*****************************************************************************/
/* HYPRE_LSI_DDICTSetDropTolerance - Set the threshold for dropping          */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTSetDropTolerance(HYPRE_Solver solver, double thresh)
{
   HYPRE_LSI_DDICT *ict_ptr = (HYPRE_LSI_DDICT *) solver;

   ict_ptr->thresh = thresh;

   return 0;
}

/*****************************************************************************/
/* HYPRE_LSI_DDICTSetOutputLevel - Set debug level                           */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTSetOutputLevel(HYPRE_Solver solver, int level)
{
   HYPRE_LSI_DDICT *ict_ptr = (HYPRE_LSI_DDICT *) solver;

   ict_ptr->outputLevel = level;

   return 0;
}

/*****************************************************************************/
/* HYPRE_LSI_DDICTSolve - Solve function for DDICT.                          */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int             i, j, Nrows, extNrows, *mat_ja, *ibuf, length;
   double          *rhs, *soln, *dbuf, *mat_aa, *dbuf2, dtmp;
   HYPRE_LSI_DDICT *ict_ptr = (HYPRE_LSI_DDICT *) solver;
   MH_Context      *context;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   Nrows    = ict_ptr->Nrows;
   extNrows = ict_ptr->extNrows;
   mat_ja   = ict_ptr->mat_ja;
   mat_aa   = ict_ptr->mat_aa;

   if ( extNrows > 0 )
   {
      dbuf  = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
      dbuf2 = hypre_TAlloc(double, extNrows , HYPRE_MEMORY_HOST);
      for ( i = 0; i < Nrows; i++ ) dbuf[i] = rhs[i];
   }
   else dbuf = dbuf2 = NULL;

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->Amat = ict_ptr->mh_mat;
   context->comm = MPI_COMM_WORLD;

   MH_ExchBdry(dbuf, context);

   for ( i = 0; i < extNrows; i++ )
   {
      dtmp = dbuf[i];
      for ( j = mat_ja[i]; j < mat_ja[i+1]; j++ )
         dtmp -= ( mat_aa[j] * dbuf2[mat_ja[j]] );
      dbuf2[i] = dtmp * mat_aa[i];
   }
   for ( i = extNrows-1; i >= 0; i-- )
   {
      dbuf2[i] *= mat_aa[i];
      dtmp = dbuf2[i];
      for ( j = mat_ja[i]; j < mat_ja[i+1]; j++ )
         dbuf2[mat_ja[j]] -= ( dtmp * mat_aa[j] );
   }
   hypre_TFree(dbuf, HYPRE_MEMORY_HOST);

   for ( i = 0; i < Nrows; i++ ) soln[i] = dbuf2[i];

   MH_ExchBdryBack(dbuf2, context, &length, &dbuf, &ibuf);

   for ( i = 0; i < length; i++ ) soln[ibuf[i]] = soln[ibuf[i]] + dbuf[i];

   hypre_TFree(ibuf, HYPRE_MEMORY_HOST);
   hypre_TFree(dbuf, HYPRE_MEMORY_HOST);
   hypre_TFree(dbuf2, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);

   return 0;
}

/*****************************************************************************/
/* HYPRE_LSI_DDICTSetup - Set up function for LSI_DDICT.                     */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_csr,
                          HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int             i, j, offset, total_recv_leng, *recv_lengths=NULL;
   int             *int_buf=NULL, mypid, nprocs, overlap_flag=1,*parray;
   int             *map=NULL, *map2=NULL, *row_partition=NULL,*parray2;
   double          *dble_buf=NULL;
   HYPRE_LSI_DDICT *ict_ptr = (HYPRE_LSI_DDICT *) solver;
   MH_Context      *context=NULL;
   MH_Matrix       *mh_mat=NULL;

   /* ---------------------------------------------------------------- */
   /* get the row information in my processors                         */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &row_partition);

   /* ---------------------------------------------------------------- */
   /* convert the incoming CSR matrix into a MH matrix                 */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->comm = MPI_COMM_WORLD;
   context->globalEqns = row_partition[nprocs];
   context->partition = hypre_TAlloc(int, (nprocs+1), HYPRE_MEMORY_HOST);
   for (i=0; i<=nprocs; i++) context->partition[i] = row_partition[i];
   hypre_TFree( row_partition , HYPRE_MEMORY_HOST);
   mh_mat = hypre_TAlloc( MH_Matrix, 1, HYPRE_MEMORY_HOST);
   context->Amat = mh_mat;
   HYPRE_LSI_MLConstructMHMatrix(A_csr,mh_mat,MPI_COMM_WORLD,
                                 context->partition,context);

   /* ---------------------------------------------------------------- */
   /* compose the enlarged overlapped local matrix                     */
   /* ---------------------------------------------------------------- */

   if ( overlap_flag )
   {
      HYPRE_LSI_DDICTComposeOverlappedMatrix(mh_mat, &total_recv_leng,
                 &recv_lengths, &int_buf, &dble_buf, &map, &map2,&offset);
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
      MPI_Allreduce(parray2,parray,nprocs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      offset = 0;
      for (i = 0; i < mypid; i++) offset += parray[i];
      hypre_TFree(parray, HYPRE_MEMORY_HOST);
      hypre_TFree(parray2, HYPRE_MEMORY_HOST);
   }

   /* ---------------------------------------------------------------- */
   /* perform ICT decomposition on local matrix                        */
   /* ---------------------------------------------------------------- */

   HYPRE_LSI_DDICTDecompose(ict_ptr,mh_mat,total_recv_leng,recv_lengths,
                             int_buf, dble_buf, map,map2, offset);

   if ( mypid == 0 && ict_ptr->outputLevel > 2 )
   {
      for ( i = 0; i < ict_ptr->extNrows; i++ )
         for ( j = ict_ptr->mat_ja[i]; j < ict_ptr->mat_ja[i+1]; j++ )
            printf("LA(%d,%d) = %e;\n", i+1, ict_ptr->mat_ja[j]+1,
                   ict_ptr->mat_aa[j]);
   }
   ict_ptr->mh_mat = mh_mat;
   hypre_TFree(recv_lengths, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(dble_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(map, HYPRE_MEMORY_HOST);
   hypre_TFree(map2, HYPRE_MEMORY_HOST);
   hypre_TFree(context->partition, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);
   return 0;
}

/*****************************************************************************/
/* subroutines used for constructing overlapped matrix                       */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTGetRowLengths(MH_Matrix *Amat, int *leng, int **recv_leng)
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

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
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

   MPI_Barrier(MPI_COMM_WORLD);

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
               msgtype, MPI_COMM_WORLD, &Request[i]);
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
      MPI_Send((void*)temp_list,length,MPI_INT,proc_id,msgtype,MPI_COMM_WORLD);
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
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTGetOffProcRows(MH_Matrix *Amat, int leng, int *recv_leng,
                           int Noffset, int *map, int *map2, int **int_buf,
                           double **dble_buf)
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

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
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
               proc_id, msgtype, MPI_COMM_WORLD, request+i);
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
                       MPI_COMM_WORLD);
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
                   proc_id, msgtype, MPI_COMM_WORLD, request+i);
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
                       MPI_COMM_WORLD);
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
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTComposeOverlappedMatrix(MH_Matrix *mh_mat,
              int *total_recv_leng, int **recv_lengths, int **int_buf,
              double **dble_buf, int **sindex_array, int **sindex_array2,
              int *offset)
{
   int        i, nprocs, mypid, Nrows, *proc_array, *proc_array2;
   int        extNrows, NrowsOffset, *index_array, *index_array2;
   int        nRecv, *recvLeng;
   double     *dble_array;
   MH_Context *context;

   /* ---------------------------------------------------------------- */
   /* fetch communication information                                  */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
   MPI_Allreduce(proc_array2,proc_array,nprocs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   NrowsOffset = 0;
   for (i = 0; i < mypid; i++) NrowsOffset += proc_array[i];
   for (i = 1; i < nprocs; i++) proc_array[i] += proc_array[i-1];
   hypre_TFree(proc_array2, HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* compose the column index map (index_array,index_array2)          */
   /* ---------------------------------------------------------------- */

   context = hypre_TAlloc(MH_Context, 1, HYPRE_MEMORY_HOST);
   context->comm = MPI_COMM_WORLD;
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

   HYPRE_LSI_DDICTGetRowLengths(mh_mat, total_recv_leng, recv_lengths);
   HYPRE_LSI_DDICTGetOffProcRows(mh_mat, *total_recv_leng, *recv_lengths,
              NrowsOffset,index_array,index_array2,int_buf, dble_buf);

   hypre_TFree(proc_array, HYPRE_MEMORY_HOST);
   HYPRE_LSI_qsort1a(index_array, index_array2, 0, extNrows-Nrows-1);
   (*sindex_array) = index_array;
   (*sindex_array2) = index_array2;
   (*offset) = NrowsOffset;
   return 0;
}

/*****************************************************************************/
/* function for doing ICT decomposition                                      */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTDecompose(HYPRE_LSI_DDICT *ict_ptr,MH_Matrix *Amat,
           int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa,
           int *map, int *map2, int Noffset)
{
   int          i, j, row_leng, *mat_ia, *mat_ja, allocated_space, *cols, mypid;
   int          index, ind2, total_nnz, offset, Nrows, extNrows;
   double       *vals, *mat_aa, *rowNorms, tau, rel_tau;
   MH_Context   *context;

   /* ---------------------------------------------------------------- */
   /* fetch ICT parameters                                             */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(ict_ptr->comm, &mypid);
   tau      = ict_ptr->thresh;
   Nrows    = Amat->Nrows;
   extNrows = Nrows + total_recv_leng;
   ict_ptr->Nrows = Nrows;
   ict_ptr->extNrows = extNrows;

   /* ---------------------------------------------------------------- */
   /* allocate temporary storage space                                 */
   /* ---------------------------------------------------------------- */

   allocated_space = extNrows;
   cols     = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
   vals     = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
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
      while (MH_GetRow(context,1,&i,allocated_space,cols,vals,&row_leng)==0)
      {
         hypre_TFree(vals, HYPRE_MEMORY_HOST);
         hypre_TFree(cols, HYPRE_MEMORY_HOST);
         allocated_space += 200 + 1;
         cols = hypre_TAlloc(int, allocated_space , HYPRE_MEMORY_HOST);
         vals = hypre_TAlloc(double, allocated_space , HYPRE_MEMORY_HOST);
      }
      total_nnz += row_leng;
      for ( j = 0; j < row_leng; j++ ) rowNorms[i] += habs(vals[j]);
      rowNorms[i] /= extNrows;
rowNorms[i] = 1.0;
   }
   for ( i = 0; i < total_recv_leng; i++ ) total_nnz += recv_lengths[i];
   mat_ia = hypre_TAlloc(int,  (extNrows + 1 ) , HYPRE_MEMORY_HOST);
   mat_ja = hypre_TAlloc(int,  total_nnz , HYPRE_MEMORY_HOST);
   mat_aa = hypre_TAlloc(double,  total_nnz , HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* construct the orginal matrix in CSR format                       */
   /* ---------------------------------------------------------------- */

   total_nnz = 0;
   mat_ia[0] = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      rel_tau   = tau * rowNorms[i];
      MH_GetRow(context,1,&i,allocated_space,cols,vals,&row_leng);
      for ( j = 0; j < row_leng; j++ )
      {
         if ( cols[j] <= i && habs(vals[j]) > rel_tau )
         {
            mat_aa[total_nnz] = vals[j];
            mat_ja[total_nnz++] = cols[j];
         }
      }
      mat_ia[i+1] = total_nnz;
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
            ind2 = HYPRE_LSI_Search(map, index, extNrows-Nrows);
            if ( ind2 >= 0 ) ext_ja[j] = map2[ind2] + Nrows;
            else             ext_ja[j] = -1;
         }
         if ( ext_ja[j] != -1 ) rowNorms[i+Nrows] += habs(ext_aa[j]);
      }
      rowNorms[i+Nrows] /= extNrows;
rowNorms[i+Nrows] = 1.0;
      rel_tau = tau * rowNorms[i+Nrows];
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         if (ext_ja[j] != -1 && ext_ja[j] <= Nrows+i && habs(ext_aa[j]) > rel_tau)
         {
            mat_aa[total_nnz] = ext_aa[j];
            mat_ja[total_nnz++] = ext_ja[j];
         }
      }
      offset += recv_lengths[i];
      mat_ia[Nrows+i+1] = total_nnz;
   }

   /* ---------------------------------------------------------------- */
   /* clean up a little                                                */
   /* ---------------------------------------------------------------- */

   hypre_TFree(Amat->rowptr, HYPRE_MEMORY_HOST);
   hypre_TFree(Amat->colnum, HYPRE_MEMORY_HOST);
   hypre_TFree(Amat->values, HYPRE_MEMORY_HOST);
   hypre_TFree(context, HYPRE_MEMORY_HOST);
   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* call ICT factorization                                           */
   /* ---------------------------------------------------------------- */

   HYPRE_LSI_DDICTFactorize(ict_ptr, mat_aa, mat_ja, mat_ia, rowNorms);

   hypre_TFree(mat_aa , HYPRE_MEMORY_HOST);
   hypre_TFree(mat_ia , HYPRE_MEMORY_HOST);
   hypre_TFree(mat_ja , HYPRE_MEMORY_HOST);
   hypre_TFree(rowNorms, HYPRE_MEMORY_HOST);

   if ( ict_ptr->outputLevel > 0 )
   {
      total_nnz = ict_ptr->mat_ja[extNrows];
      printf("%d : DDICT number of nonzeros     = %d\n",mypid,total_nnz);
   }

   return 0;
}

/*****************************************************************************/
/* function for doing ICT factorization                                      */
/*---------------------------------------------------------------------------*/

int HYPRE_LSI_DDICTFactorize(HYPRE_LSI_DDICT *ict_ptr, double *mat_aa,
                 int *mat_ja, int *mat_ia, double *rowNorms)
{
   int    i, j, row_leng, first, row_beg, row_endp1, track_leng, *track_array;
   int    k, mypid, nnz_count, num_small_pivot,  printstep, extNrows;
   int    *msr_iptr, *msc_jptr, *msc_jend, rowMax, Lcount, sortcnt, *sortcols;
   int    totalFill, colIndex, index;
   double fillin, tau, rel_tau, *dble_buf, *msr_aptr, *msc_aptr, absval;
   double *sortvals, ddata;

   /* ---------------------------------------------------------------- */
   /* fetch ICT parameters                                             */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(ict_ptr->comm, &mypid);
   tau       = ict_ptr->thresh;
   fillin    = ict_ptr->fillin;
   extNrows  = ict_ptr->extNrows;
   rowMax    = 0;
   for ( i = 0; i < extNrows; i++ )
   {
      row_leng = mat_ia[i+1] - mat_ia[i];
      if ( row_leng > rowMax ) rowMax = row_leng;
   }
   totalFill = rowMax * (fillin + 1) * extNrows;

   /* ---------------------------------------------------------------- */
   /* allocate permanent and temporary storage                         */
   /* ---------------------------------------------------------------- */

   track_array = hypre_TAlloc(int,  extNrows , HYPRE_MEMORY_HOST);
   sortcols    = hypre_TAlloc(int,  extNrows , HYPRE_MEMORY_HOST);
   sortvals    = hypre_TAlloc(double,  extNrows , HYPRE_MEMORY_HOST);
   dble_buf    = hypre_TAlloc(double,  extNrows , HYPRE_MEMORY_HOST);
   msr_iptr    = hypre_TAlloc(int,  (totalFill+extNrows+1) , HYPRE_MEMORY_HOST);
   msc_jptr    = hypre_TAlloc(int,  (totalFill+extNrows+1) , HYPRE_MEMORY_HOST);
   msc_jend    = hypre_TAlloc(int,  (extNrows + 1 ) , HYPRE_MEMORY_HOST);
   msr_aptr    = hypre_TAlloc(double,  (totalFill+extNrows) , HYPRE_MEMORY_HOST);
   msc_aptr    = hypre_TAlloc(double,  (totalFill+extNrows) , HYPRE_MEMORY_HOST);
   msc_jptr[0] = msc_jend[0] = extNrows + 1;
   for ( i = 1; i <= extNrows; i++ )
   {
      msc_jptr[i] = msc_jptr[i-1] + rowMax * (fillin + 1);
      msc_jend[i] = msc_jptr[i];
   }
   for ( i = 0; i < extNrows; i++ ) dble_buf[i] = 0.0;
   printstep = extNrows /  10;

   /* ---------------------------------------------------------------- */
   /* process the rows                                                 */
   /* ---------------------------------------------------------------- */

   num_small_pivot = 0;
   nnz_count       = extNrows + 1;
   msr_iptr[0]     = nnz_count;

   for ( i = 0; i < extNrows; i++ )
   {
      if ( i % printstep == 0 && ict_ptr->outputLevel > 0 )
         printf("%4d : DDICT Processing row %6d (%6d)\n",mypid,i,extNrows);

      /* ------------------------------------------------------------- */
      /* get the row information                                       */
      /* ------------------------------------------------------------- */

      track_leng = 0;
      row_beg    = mat_ia[i];
      row_endp1  = mat_ia[i+1];
      row_leng   = row_endp1 - row_beg;
      first      = i;
      rel_tau    = tau * rowNorms[i];

      /* ------------------------------------------------------------- */
      /* load the row into dble_buf                                    */
      /* ------------------------------------------------------------- */

      for ( j = row_beg; j < row_endp1; j++ )
      {
         colIndex = mat_ja[j];
         if ( colIndex > i ) printf("WARNING (A)\n");
         dble_buf[colIndex] = mat_aa[j];
         track_array[track_leng++] = colIndex;
         if ( colIndex < first ) first = colIndex;
      }
      Lcount = row_leng * fillin;

      /* ------------------------------------------------------------- */
      /* reduce the row                                                */
      /* ------------------------------------------------------------- */

      for ( j = first; j < i; j++ )
      {
         if ( habs(dble_buf[j]) > rel_tau )
         {
            ddata = dble_buf[j] * msr_aptr[j];

            for ( k = msc_jptr[j]; k < msc_jend[j]; k++ )
            {
               colIndex = msc_jptr[k];
               if ( colIndex > j && colIndex < i )
               {
                  if ( dble_buf[colIndex] != 0.0 )
                     dble_buf[colIndex] -= (ddata * msc_aptr[k]);
                  else
                  {
                     dble_buf[colIndex] = - (ddata * msc_aptr[k]);
                     track_array[track_leng++] = colIndex;
                  }
               }
            }
            dble_buf[j] = ddata;
         }
         else dble_buf[j] = 0.0;
      }

      /* ------------------------------------------------------------- */
      /* sort the new nonzeros                                         */
      /* ------------------------------------------------------------- */

      sortcnt = 0;
      if ( track_leng > extNrows ) printf("WARNING (B)\n");
      for ( j = row_leng; j < track_leng; j++ )
      {
         index = track_array[j];
         absval = habs(dble_buf[index]);
         if ( absval > rel_tau )
         {
            sortcols[sortcnt] = index;
            sortvals[sortcnt++] = absval * rowNorms[index];
         }
         else dble_buf[index] = 0.0;
      }
      if ( sortcnt > Lcount )
      {
         HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Lcount);
         for ( j = Lcount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
         for ( j = 0; j < row_leng; j++ )
         {
            index = track_array[j];
            if ( index != i )
            {
               ddata = dble_buf[i] - (dble_buf[index] * dble_buf[index]);
               if ( ddata > 1.0E-10 ) dble_buf[i] = ddata;
               else
               {
                  printf("%d : DDICT negative pivot  (%d,%d,%d)\n", mypid,
                         i, j, extNrows);
                  num_small_pivot++;
                  for ( k = j; k < row_leng; k++ )
                  {
                     index = track_array[k];
                     dble_buf[index] = 0.0;
                  }
                  Lcount = 0;
                  break;
               }
            }
         }
         for ( j = 0; j < Lcount; j++ )
         {
            index = sortcols[j];
            ddata = dble_buf[i] - (dble_buf[index] * dble_buf[index]);
            if ( ddata > 1.0E-10 ) dble_buf[i] = ddata;
            else
            {
               printf("%d : (2) DDICT negative pivot  (%d,%d,%d)\n", mypid,
                      i, j, extNrows);
               num_small_pivot++;
               for ( k = j; k < Lcount; k++ )
               {
                  index = sortcols[k];
                  dble_buf[index] = 0.0;
               }
               Lcount = j;
               break;
            }
         }
      }
      else
      {
         for ( j = 0; j < row_leng; j++ )
         {
            index = track_array[j];
            if ( index != i )
            {
               ddata = dble_buf[i] - (dble_buf[index] * dble_buf[index]);
               if ( ddata > 1.0E-10 ) dble_buf[i] = ddata;
               else
               {
                  printf("%d : DDICT negative pivot  (%d,%d,%d)\n", mypid,
                         i, j, extNrows);
                  num_small_pivot++;
                  for ( k = j; k < row_leng; k++ )
                  {
                     index = track_array[k];
                     dble_buf[index] = 0.0;
                  }
                  sortcnt = 0;
                  break;
               }
            }
         }
         for ( j = 0; j < sortcnt; j++ )
         {
            index = sortcols[j];
            ddata = dble_buf[i] - (dble_buf[index] * dble_buf[index]);
            if ( ddata > 1.0E-10 ) dble_buf[i] = ddata;
            else
            {
               printf("%d : (2) DDICT negative pivot  (%d,%d,%d)\n", mypid,
                      i, j, extNrows);
               num_small_pivot++;
               for ( k = j; k < sortcnt; k++ )
               {
                  index = sortcols[k];
                  dble_buf[index] = 0.0;
               }
               sortcnt = j;
               break;
            }
         }
      }
      if ( dble_buf[i] > 0 )
      {
         if ( dble_buf[i] < 1.0E-10 )
         {
            num_small_pivot++;
            msc_aptr[i] = msr_aptr[i] = 1.0E5;
         }
         else msc_aptr[i] = msr_aptr[i] = 1.0 / sqrt( dble_buf[i] );
         dble_buf[i] = 0.0;
      }
      else
      {
         printf("%4d : ERROR in DDICT - negative or zero pivot.\n", mypid);
         printf("                       L(%4d,%4d) = %e\n", i, i, dble_buf[i]);
         msc_aptr[i] = msr_aptr[i] = 1.0 / sqrt( - dble_buf[i] );
         dble_buf[i] = 0.0;
      }
      for ( j = 0; j < track_leng; j++ )
      {
         index = track_array[j];
         if ( index < i && dble_buf[index] != 0.0 )
         {
            msr_aptr[nnz_count] = dble_buf[index];
            msr_iptr[nnz_count++] = index;
            colIndex = msc_jend[index]++;
            msc_aptr[colIndex] = dble_buf[index];
            msc_jptr[colIndex] = i;
            dble_buf[index] = 0.0;
         }
      }
      msr_iptr[i+1] = nnz_count;
   }

   if ( nnz_count > totalFill+extNrows )
      printf("%4d : DDICT WARNING : buffer overflow (%d,%d)\n",mypid,nnz_count,
              totalFill+extNrows);
   if ( ict_ptr->outputLevel > 0 )
   {
      printf("%4d : DDICT number of nonzeros     = %d\n",mypid,nnz_count);
      printf("%4d : DDICT number of small pivots = %d\n",mypid,num_small_pivot);
   }

   /* ---------------------------------------------------------- */
   /* deallocate temporary storage space                         */
   /* ---------------------------------------------------------- */

   hypre_TFree(track_array, HYPRE_MEMORY_HOST);
   hypre_TFree(sortcols, HYPRE_MEMORY_HOST);
   hypre_TFree(sortvals, HYPRE_MEMORY_HOST);
   hypre_TFree(dble_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(msc_jptr, HYPRE_MEMORY_HOST);
   hypre_TFree(msc_jend, HYPRE_MEMORY_HOST);
   hypre_TFree(msc_aptr, HYPRE_MEMORY_HOST);

   ict_ptr->mat_ja = msr_iptr;
   ict_ptr->mat_aa = msr_aptr;
   return 0;
}

