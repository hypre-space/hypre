/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_DDILUT interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/utilities.h"
#include "HYPRE.h"
#include "IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "HYPRE_MHMatrix.h"

#ifdef MLPACK

#include "ml_struct.h"
#include "ml_aggregate.h"

#endif

#include "HYPRE_MHMatrix.h"

typedef struct HYPRE_LSI_DDIlut_Struct
{
   MPI_Comm  comm;
   MH_Matrix *mh_mat;
   double    thresh;
   double    fillin;
   int       Nrows;
   int       extNrows;
   int       *mat_ia;
   int       *mat_ja;
   double    *mat_aa;
   int       outputLevel;
}
HYPRE_LSI_DDIlut;

extern HYPRE_ParCSRMLConstructMHMatrix(HYPRE_ParCSRMatrix,MH_Matrix *,
                                       MPI_Comm, int *, MH_Context *);
extern int HYPRE_LSI_DDIlutComposeOverlappedMatrix(MH_Matrix *, int *, 
                 int **recv_lengths, int **int_buf, double **dble_buf, 
                 int **sindex_array, int **sindex_array2, int *offset);
extern int HYPRE_LSI_DDIlutGetRowLengths(MH_Matrix *Amat, int *leng, int **);
extern int HYPRE_LIS_DDIlutGetOffProcRows(MH_Matrix *Amat, int leng, int *,
                 int Noffset, int *map, int *map2, int **int_buf,
                 double **dble_buf);
extern int HYPRE_LSI_DDIlutDecompose(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
                 int total_recv_leng, int *recv_lengths, int *ext_ja, 
                 double *ext_aa, int *map, int *map2, int Noffset);
extern void HYPRE_LSI_Sort(int *, int, int *, double *);
extern int  HYPRE_LSI_SplitDSort(double*,int,int*,int);

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutCreate - Return a DDIlut preconditioner object "solver".  
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_DDIlut *ilut_ptr;
   
   ilut_ptr = (HYPRE_LSI_DDIlut *) malloc(sizeof(HYPRE_LSI_DDIlut));

   if (ilut_ptr == NULL) return 1;

   ilut_ptr->comm        = comm;
   ilut_ptr->mh_mat      = NULL;
   ilut_ptr->fillin      = 0.0;
   ilut_ptr->thresh      = 0.0; /* defaults */
   ilut_ptr->mat_ia      = NULL;
   ilut_ptr->mat_ja      = NULL;
   ilut_ptr->mat_aa      = NULL;
   ilut_ptr->outputLevel = 0;

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
   if ( ilut_ptr->mat_ia != NULL ) free(ilut_ptr->mat_ia);
   if ( ilut_ptr->mat_ja != NULL ) free(ilut_ptr->mat_ja);
   if ( ilut_ptr->mat_aa != NULL ) free(ilut_ptr->mat_aa);
   ilut_ptr->mat_ia = NULL;
   ilut_ptr->mat_ja = NULL;
   ilut_ptr->mat_aa = NULL;
   ilut_ptr->mh_mat = NULL;
   if ( ilut_ptr->mh_mat != NULL ) 
   {
      if (ilut_ptr->mh_mat->sendProc != NULL) free(ilut_ptr->mh_mat->sendProc);
      if (ilut_ptr->mh_mat->sendLeng != NULL) free(ilut_ptr->mh_mat->sendLeng);
      if (ilut_ptr->mh_mat->recvProc != NULL) free(ilut_ptr->mh_mat->recvProc);
      if (ilut_ptr->mh_mat->recvLeng != NULL) free(ilut_ptr->mh_mat->recvLeng);
      for ( i = 0; i < ilut_ptr->mh_mat->sendProcCnt; i++ )
         if (ilut_ptr->mh_mat->sendList[i] != NULL) 
            free(ilut_ptr->mh_mat->sendList[i]);
      if (ilut_ptr->mh_mat->sendList != NULL) free(ilut_ptr->mh_mat->sendList);
   }  
   free(ilut_ptr);

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
 * HYPRE_LSI_DDIlutSetOutputLevel - Set debug level 
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSetOutputLevel(HYPRE_Solver solver, int level)
{
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;

   ilut_ptr->outputLevel = level;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSolve - Solve function for ParaSails.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int               i, j, ierr = 0, *idiag, Nrows, extNrows, *mat_ia, *mat_ja;
   int               column;
   double            *rhs, *soln, *dbuffer, ddata, *mat_aa;
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;
   MH_Context        *context;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   Nrows    = ilut_ptr->Nrows;
   extNrows = ilut_ptr->extNrows;
   mat_ia   = ilut_ptr->mat_ia;
   mat_ja   = ilut_ptr->mat_ja;
   mat_aa   = ilut_ptr->mat_aa;

   dbuffer = (double *) malloc(extNrows * sizeof(double));
   idiag   = (int *)    malloc(extNrows * sizeof(int));
   for ( i = 0; i < Nrows; i++ ) dbuffer[i] = rhs[i];

   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->Amat = ilut_ptr->mh_mat;
   context->comm = MPI_COMM_WORLD;

   MH_ExchBdry(dbuffer, context);

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
   for ( i = 0; i < Nrows; i++ ) soln[i] = dbuffer[i];
   free(dbuffer);
   free(idiag);
   free(context);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_DDIlutSetup - Set up function for LSI_DDIlut.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DDIlutSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_csr,
                          HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int              i, j, offset, total_recv_leng, *recv_lengths=NULL;
   int              *int_buf=NULL, mypid, nprocs, overlap_flag=1,*parray;
   int              *map=NULL, *map2=NULL, *row_partition=NULL,*parray2;
   double           *dble_buf=NULL;
   HYPRE_LSI_DDIlut *ilut_ptr = (HYPRE_LSI_DDIlut *) solver;
   MH_Context       *context=NULL;
   MH_Matrix        *mh_mat=NULL;

   /* ---------------------------------------------------------------- */
   /* get the row information in my processors                         */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &row_partition);

   /* ---------------------------------------------------------------- */
   /* convert the incoming CSR matrix into a MH matrix                 */
   /* ---------------------------------------------------------------- */

   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->comm = MPI_COMM_WORLD;
   context->globalEqns = row_partition[nprocs];
   context->partition = (int *) malloc(sizeof(int)*(nprocs+1));
   for (i=0; i<=nprocs; i++) context->partition[i] = row_partition[i];
   hypre_TFree( row_partition );
   mh_mat = ( MH_Matrix * ) malloc( sizeof( MH_Matrix) );
   context->Amat = mh_mat;
   HYPRE_ParCSRMLConstructMHMatrix(A_csr,mh_mat,MPI_COMM_WORLD,
                                   context->partition,context); 

   /* ---------------------------------------------------------------- */
   /* compose the enlarged overlapped local matrix                     */
   /* ---------------------------------------------------------------- */
   
   if ( overlap_flag )
   {
      HYPRE_LSI_DDIlutComposeOverlappedMatrix(mh_mat, &total_recv_leng, 
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
      parray  = (int *) malloc(nprocs * sizeof(int) );
      parray2 = (int *) malloc(nprocs * sizeof(int) );
      for ( i = 0; i < nprocs; i++ ) parray2[i] = 0;
      parray2[mypid] = mh_mat->Nrows;
      MPI_Allreduce(parray2,parray,nprocs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      offset = 0;
      for (i = 0; i < mypid; i++) offset += parray[i];
      free(parray);
      free(parray2);
   }

   /* ---------------------------------------------------------------- */
   /* perform ILUT decomposition on local matrix                       */
   /* ---------------------------------------------------------------- */

   HYPRE_LSI_DDIlutDecompose(ilut_ptr,mh_mat,total_recv_leng,recv_lengths,
                             int_buf, dble_buf, map,map2, offset);

   /*
   if ( mypid == 0 )
   {
      for ( i = 0; i < ilut_ptr->extNrows; i++ )
         for ( j = ilut_ptr->mat_ia[i]; j < ilut_ptr->mat_ia[i+1]; j++ )
            printf("LA(%d,%d) = %e;\n", i+1, ilut_ptr->mat_ja[j]+1,
                   ilut_ptr->mat_aa[j]);
   }
   */

   ilut_ptr->mh_mat = mh_mat;
   if ( mh_mat->rowptr != NULL ) free (mh_mat->rowptr);
   if ( mh_mat->colnum != NULL ) free (mh_mat->colnum);
   if ( mh_mat->values != NULL ) free (mh_mat->values);
   mh_mat->rowptr = NULL;
   mh_mat->colnum = NULL;
   mh_mat->values = NULL;
   if ( map  != NULL ) free(map);
   if ( map2 != NULL ) free(map2);
   if ( int_buf != NULL ) free(int_buf);
   if ( dble_buf != NULL ) free(dble_buf);
   if ( recv_lengths != NULL ) free(recv_lengths);
   free( context->partition );
   free( context );
}

/*****************************************************************************/
/* subroutines used for constructing overlapped matrix                       */
/*****************************************************************************/

int HYPRE_LSI_DDIlutGetRowLengths(MH_Matrix *Amat, int *leng, int **recv_leng)
{
   int         i, j, m, index, *temp_list, allocated_space, length;
   int         nRecv, *recvProc, *recvLeng, *cols, total_recv, mtype, msgtype;
   int         nSend, *sendProc, *sendLeng, **sendList, proc_id, offset;
   double      *vals;
   MPI_Status  status;
   MH_Context  *context;

   /* ---------------------------------------------------------------- */
   /* fetch communication information                                  */
   /* ---------------------------------------------------------------- */

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
   MPI_Barrier(MPI_COMM_WORLD);

   /* ---------------------------------------------------------------- */
   /* write out all messages                                           */
   /* ---------------------------------------------------------------- */

   mtype = 2001;
   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->Amat = Amat;
   allocated_space = 100;
   cols = (int *) malloc(allocated_space * sizeof(int));
   vals = (double *) malloc(allocated_space * sizeof(double));
   for (i = 0; i < nSend; i++)
   {
      proc_id   = sendProc[i];
      length    = sendLeng[i];
      temp_list = (int*) malloc(sendLeng[i] * sizeof(int));
      for (j = 0; j < length; j++)
      {
         index = sendList[i][j];
         while (MH_GetRow(context,1,&index,allocated_space,cols,vals,&m)==0)
         {
            free(cols); free(vals);
            allocated_space += 200 + 1;
            cols = (int *) malloc(allocated_space * sizeof(int));
            vals = (double *) malloc(allocated_space * sizeof(double));
         } 
         temp_list[j] = m;
      }
      msgtype = mtype;
      MPI_Send((void*)temp_list,length,MPI_INT,proc_id,msgtype,MPI_COMM_WORLD);
      free( temp_list );
   }
   free(cols);
   free(vals);
   free(context);

   /* ---------------------------------------------------------------- */
   /* receives for all messages                                        */
   /* ---------------------------------------------------------------- */

   (*recv_leng) = (int  *)        malloc(total_recv * sizeof(int));
   offset = 0;
   for (i = 0; i < nRecv; i++)
   {
      proc_id = recvProc[i];
      msgtype = mtype;
      length  = recvLeng[i];
      MPI_Recv((void *) &((*recv_leng)[offset]), length, MPI_INT, proc_id,
               msgtype, MPI_COMM_WORLD, &status);
      offset += length;
   }

   return 0;
}

/*****************************************************************************/
/* needed for overlapped smoothers                                           */
/*****************************************************************************/

int HYPRE_LSI_DDIlutGetOffProcRows(MH_Matrix *Amat, int leng, int *recv_leng,
                           int Noffset, int *map, int *map2, int **int_buf,
                           double **dble_buf)
{
   int         i, j, k, m, *temp_list, length, offset, allocated_space, proc_id;
   int         nRecv, nSend, *recvProc, *sendProc, total_recv, mtype, msgtype;
   int         *sendLeng, *recvLeng, **sendList, *cols, *isend_buf, Nrows;
   int         nnz, nnz_offset, index;
   double      *vals, *send_buf;
   MPI_Request *request;
   MPI_Status  status;
   MH_Context  *context;

   /* ---------------------------------------------------------------- */
   /* fetch communication information                                  */
   /* ---------------------------------------------------------------- */

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
        request     = (MPI_Request *) malloc(nRecv*sizeof(MPI_Request ));
   else request = NULL;

   if ( total_recv > 0 )
   {
      (*int_buf)  = (int  *)    malloc(total_recv * sizeof(int));
      (*dble_buf) = (double  *) malloc(total_recv * sizeof(double));
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

   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->Amat = Amat;
   mtype = 2002;
   allocated_space = 100;
   cols = (int *) malloc(allocated_space * sizeof(int));
   vals = (double *) malloc(allocated_space * sizeof(double));
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
            free(cols); free(vals);
            allocated_space += 200 + 1;
            cols = (int *) malloc(allocated_space * sizeof(int));
            vals = (double *) malloc(allocated_space * sizeof(double));
         } 
         nnz += m;
      }
      if ( nnz > 0 ) send_buf = (double *) malloc( nnz * sizeof(double));
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
      if ( nnz > 0 ) free( send_buf );
   }
   free(cols);
   free(vals);

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
   cols = (int *) malloc(allocated_space * sizeof(int));
   vals = (double *) malloc(allocated_space * sizeof(double));
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
      if ( nnz > 0 ) isend_buf = (int *) malloc( nnz * sizeof(int));
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
      if ( nnz > 0 ) free( isend_buf );
   }
   free(cols);
   free(vals);

   /* ----------------------------------------------------------- */
   /* post receives for all messages                              */
   /* ----------------------------------------------------------- */

   for (i = 0; i < nRecv; i++)
   {
      MPI_Wait(request+i, &status);
   }

   free(request);
   free(context);
   return 0;
}

/*****************************************************************************/
/* construct an enlarged overlapped local matrix                             */
/*****************************************************************************/

int HYPRE_LSI_DDIlutComposeOverlappedMatrix(MH_Matrix *mh_mat, 
              int *total_recv_leng, int **recv_lengths, int **int_buf, 
              double **dble_buf, int **sindex_array, int **sindex_array2, 
              int *offset)
{
   int        i, j, nprocs, mypid, Nrows, *proc_array, *proc_array2;
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

   proc_array  = (int *) malloc(nprocs * sizeof(int) );
   proc_array2 = (int *) malloc(nprocs * sizeof(int) );
   for ( i = 0; i < nprocs; i++ ) proc_array2[i] = 0;
   proc_array2[mypid] = Nrows;
   MPI_Allreduce(proc_array2,proc_array,nprocs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   NrowsOffset = 0;
   for (i = 0; i < mypid; i++) NrowsOffset += proc_array[i];
   for (i = 1; i < nprocs; i++) proc_array[i] += proc_array[i-1];
   free(proc_array2);

   /* ---------------------------------------------------------------- */
   /* compose the column index map (index_array,index_array2)          */
   /* ---------------------------------------------------------------- */

   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->comm = MPI_COMM_WORLD;
   context->Amat = mh_mat;
   dble_array  = (double *) malloc(extNrows *sizeof(double));
   for (i = Nrows; i < extNrows; i++) dble_array[i] = 0.0;
   for (i = 0; i < Nrows; i++) dble_array[i] = 1.0 * ( i + NrowsOffset );
   MH_ExchBdry(dble_array, context);
   if ( extNrows-Nrows > 0 )
      index_array = ( int *) malloc((extNrows-Nrows) * sizeof(int));
   else 
      index_array = NULL;
   for (i = Nrows; i < extNrows; i++) index_array[i-Nrows] = dble_array[i];
   if ( extNrows-Nrows > 0 )
      index_array2  = (int *) malloc((extNrows-Nrows) *sizeof(int));
   else 
      index_array2 = NULL;
   for (i = 0; i < extNrows-Nrows; i++) index_array2[i] = i;
   free( dble_array );
   free(context);

   /* ---------------------------------------------------------------- */
   /* send the lengths of each row to remote processor                 */
   /* at the end, additional row information should be given           */
   /* in total_recv_leng, recv_lengths, int_buf, dble_buf              */
   /* ---------------------------------------------------------------- */

   HYPRE_LSI_DDIlutGetRowLengths(mh_mat, total_recv_leng, recv_lengths);
   HYPRE_LSI_DDIlutGetOffProcRows(mh_mat, *total_recv_leng, *recv_lengths, 
              NrowsOffset,index_array,index_array2,int_buf, dble_buf);

   free(proc_array);
   HYPRE_LSI_Sort(index_array, extNrows-Nrows, index_array2, NULL);
   (*sindex_array) = index_array;
   (*sindex_array2) = index_array2;
   (*offset) = NrowsOffset;
   return 0;
}

/*****************************************************************************/
/* function for doing ILUT decomposition                                     */
/*****************************************************************************/

int HYPRE_LSI_DDIlutDecompose(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
           int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa, 
           int *map, int *map2, int Noffset)
{
   int          *mat_ia, *mat_ja, i, m, allocated_space, *cols, mypid;
   int          index, first, Lcount, Ucount, ncnt, j, k, total_nnz;
   int          sortcnt, colIndex, offset, nnz, nnz_count, Nrows, extNrows;
   double       *vals, ddata, thresh, *mat_aa, *diagonal, *rowNorms;
   double       *dble_buf, fillin;
   MH_Context   *context;

   /* ---------------------------------------------------------------- */
   /* fetch ILUT parameters                                            */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(ilut_ptr->comm, &mypid);
   fillin   = ilut_ptr->fillin;
   thresh   = ilut_ptr->thresh;
   Nrows    = Amat->Nrows;
   extNrows = Nrows + total_recv_leng;
   ilut_ptr->Nrows = Nrows;
   ilut_ptr->extNrows = extNrows;

   /* ---------------------------------------------------------------- */
   /* allocate temporary storage space                                 */
   /* ---------------------------------------------------------------- */

   allocated_space = extNrows;
   cols = (int *)    malloc(allocated_space * sizeof(int));
   vals = (double *) malloc(allocated_space * sizeof(double));
   dble_buf = (double *) malloc(extNrows * sizeof(double));
   diagonal = (double *) malloc(extNrows * sizeof(double));
   rowNorms = (double *) malloc(extNrows * sizeof(double));

   /* ---------------------------------------------------------------- */
   /* compute the storage requirement for the ILU matrix               */
   /* ---------------------------------------------------------------- */

   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->Amat = Amat;
   total_nnz     = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      rowNorms[i] = 0.0;
      while (MH_GetRow(context,1,&i,allocated_space,cols,vals,&m)==0)
      {
         free(vals); free(cols);
         allocated_space += 200 + 1;
         cols = (int *) malloc(allocated_space * sizeof(int));
         vals = (double *) malloc(allocated_space * sizeof(double));
      }
      total_nnz += m;
      for ( j = 0; j < m; j++ ) rowNorms[i] += abs(vals[j]);
      rowNorms[i] /= extNrows;
   }
   for ( i = 0; i < total_recv_leng; i++ ) total_nnz += recv_lengths[i];
   total_nnz = (int) ((double) total_nnz * (fillin + 1.0));
   ilut_ptr->mat_ia = (int *) malloc( (extNrows + 1 ) * sizeof(int));
   ilut_ptr->mat_ja = (int *) malloc( total_nnz * sizeof(int));
   ilut_ptr->mat_aa = (double *) malloc( total_nnz * sizeof(double));
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
         if ( ext_ja[j] != -1 ) rowNorms[i+Nrows] += abs(ext_aa[j]);
      }
      rowNorms[i+Nrows] /= extNrows;
      offset += recv_lengths[i];
   }

   /* ---------------------------------------------------------------- */
   /* process the first Nrows                                          */
   /* ---------------------------------------------------------------- */

   nnz_count = 0;
   nnz       = 0;
   mat_ia[0] = 0;
   for ( i = 0; i < Nrows; i++ )
   {
      /*if ( i % 1000 == 0 && ilut_ptr->outputLevel > 0 ) 
         printf("%4d : Processing row %d(%d)\n",mypid,i,extNrows);
      */
      for ( j = 0; j < extNrows; j++ ) dble_buf[j] = 0.0;
      MH_GetRow(context,1,&i,allocated_space,cols,vals,&m);
      if ( m < 0 )
         printf("IlutDecompose WARNING(1): row nnz = %d\n",m);

      for ( j = 0; j < m; j++ ) 
      {
         if ( cols[j] < extNrows ) dble_buf[cols[j]] = vals[j];
         else if ( extNrows > Nrows ) 
         {
            printf("IlutDecompose WARNING(2) : index = %d(%d)\n",cols[j],extNrows);
         }
      }
      Lcount = Ucount = first = 0;
      first  = -1;
      for ( j = 0; j < extNrows; j++ )
      {
         if ( dble_buf[j] != 0 ) 
         {
            if ( j < i ) Lcount++;
            else if ( j > i ) Ucount++;
            else if ( j == i ) diagonal[i] = dble_buf[j];
            if ( first == -1 ) first = j;
         }
      }
      Lcount = Lcount * (fillin + 1);
      Ucount = Ucount * (fillin + 1);
      for ( j = first; j < i; j++ )
      {
         ddata = dble_buf[j] / diagonal[j];
         if ( ddata != 0.0 )
         {
            for ( k = mat_ia[j]; k < mat_ia[j+1]; k++ )
            {
               colIndex = mat_ja[k];
               if ( colIndex > j ) dble_buf[colIndex] -= (ddata * mat_aa[k]);
            }
            dble_buf[j] = ddata;
         }

      }

      sortcnt = 0;
      for ( j = 0; j < i; j++ )
      {
         if ( dble_buf[j] < 0 )
         {
            cols[sortcnt] = j;
            vals[sortcnt++] = - dble_buf[j] * rowNorms[j];
         }
         else if ( dble_buf[j] > 0 )
         {
            cols[sortcnt] = j;
            vals[sortcnt++] = dble_buf[j] * rowNorms[j];
         }
      }

      if ( sortcnt > Lcount ) HYPRE_LSI_SplitDSort(vals,sortcnt,cols,Lcount);
      ddata = thresh * rowNorms[i];
      if ( sortcnt > Lcount )
      {
         for ( j = 0; j < Lcount; j++ )
            if ( abs(vals[j]) < ddata ) dble_buf[cols[j]] = 0.0;
         for ( j = Lcount; j < sortcnt; j++ ) dble_buf[cols[j]] = 0.0;
      }
      for ( j = 0; j < i; j++ )
      {
         if ( dble_buf[j] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[j];
            mat_ja[nnz_count++] = j;
         }
      }
      diagonal[i] = dble_buf[i];
      if ( diagonal[i] == 0.0 ) diagonal[i] = 1.0E-6;
      mat_aa[nnz_count] = diagonal[i];
      mat_ja[nnz_count++] = i;
      sortcnt = 0;
      for ( j = i+1; j < extNrows; j++ )
      {
         if ( dble_buf[j] < 0 )
         {
            cols[sortcnt] = j;
            vals[sortcnt++] = - dble_buf[j] * rowNorms[j];
         }
         else if ( dble_buf[j] > 0 )
         {
            cols[sortcnt] = j;
            vals[sortcnt++] = dble_buf[j] * rowNorms[j];
         }
      }
      if ( sortcnt > Ucount ) HYPRE_LSI_SplitDSort(vals,sortcnt,cols,Ucount);
      ddata = thresh * rowNorms[i];
      if ( sortcnt > Ucount )
      {
         for ( j = 0; j < Ucount; j++ )
            if ( abs(vals[j]) < ddata ) dble_buf[cols[j]] = 0.0;
         for ( j = Ucount; j < sortcnt; j++ ) dble_buf[cols[j]] = 0.0;
      }
      for ( j = i+1; j < extNrows; j++ )
      {
         if ( dble_buf[j] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[j];
            mat_ja[nnz_count++] = j;
         }
      }
      mat_ia[i+1] = nnz_count;
   }

   /* ---------------------------------------------------------------- */
   /* process the off-processor rows                                   */
   /* ---------------------------------------------------------------- */

   offset = 0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      /*
      if ( (i+Nrows) % 1000 == 0 && ilut_ptr->outputLevel > 0 ) 
         printf("%4d : Processing row %d(%d)\n",mypid,i+Nrows,extNrows);
      */
      for ( j = 0; j < extNrows; j++ ) dble_buf[j] = 0.0;
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         if ( ext_ja[j] != -1 ) dble_buf[ext_ja[j]] = ext_aa[j];
      }
      Lcount = Ucount = 0;
      first  = -1;
      for ( j = 0; j < extNrows; j++ )
      {
         if ( dble_buf[j] != 0 && j < i+Nrows ) Lcount++;
         else if ( dble_buf[j] != 0 && j > i+Nrows ) Ucount++;
         else if ( i+Nrows == j ) diagonal[i+Nrows] = dble_buf[j];
         if ( dble_buf[j] != 0.0 && first == -1 ) first = j;
      }
      Lcount = Lcount * (fillin + 1);
      Ucount = Ucount * (fillin + 1);
      for ( j = first; j < i+Nrows; j++ )
      {
         ddata = dble_buf[j] / diagonal[j];
         if ( ddata != 0.0 )
         {
            for ( k = mat_ia[j]; k < mat_ia[j+1]; k++ )
            {
               colIndex = mat_ja[k];
               if ( colIndex > j ) dble_buf[colIndex] -= ddata * mat_aa[k];
            }
            dble_buf[j] = ddata;
         }
      }
      sortcnt = 0;
      for ( j = 0; j < i+Nrows; j++ )
      {
         if ( dble_buf[j] < 0 )
         {
            cols[sortcnt] = j;
            vals[sortcnt++] = - dble_buf[j]*rowNorms[j];
         }
         else if ( dble_buf[j] > 0 )
         {
            cols[sortcnt] = j;
            vals[sortcnt++] = dble_buf[j] * rowNorms[j];
         }
      }
      if ( sortcnt > Lcount ) HYPRE_LSI_SplitDSort(vals,sortcnt,cols,Lcount);
      ddata = thresh * rowNorms[i+Nrows];
      if ( sortcnt > Lcount )
      {
         for ( j = 0; j < Lcount; j++ )
            if ( abs(vals[j]) < ddata ) dble_buf[cols[j]] = 0.0;
         for ( j = Lcount; j < sortcnt; j++ ) dble_buf[cols[j]] = 0.0;
      }
      for ( j = 0; j < i+Nrows; j++ )
      {
         if ( dble_buf[j] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[j];
            mat_ja[nnz_count++] = j;
         }
      }
      diagonal[i+Nrows] = dble_buf[i+Nrows];
      if ( diagonal[i+Nrows] == 0.0 ) diagonal[i+Nrows] = 1.0E-6;
      mat_aa[nnz_count] = diagonal[i+Nrows];
      mat_ja[nnz_count++] = i+Nrows;
      sortcnt = 0;
      for ( j = i+Nrows+1; j < extNrows; j++ )
      {
         if ( dble_buf[j] < 0 )
         {
            cols[sortcnt] = j;
            vals[sortcnt++] = - dble_buf[j]*rowNorms[j];
         }
         else if ( dble_buf[j] > 0 )
         {
            cols[sortcnt] = j;
            vals[sortcnt++] = dble_buf[j] * rowNorms[j];
         }
      }
      if ( sortcnt > Ucount ) HYPRE_LSI_SplitDSort(vals,sortcnt,cols,Ucount);
      ddata = thresh * rowNorms[i+Nrows];
      if ( sortcnt > Ucount )
      {
         for ( j = 0; j < Ucount; j++ )
            if ( abs(vals[j]) < ddata ) dble_buf[cols[j]] = 0.0;
         for ( j = Ucount; j < sortcnt; j++ ) dble_buf[cols[j]] = 0.0;
      }
      for ( j = i+Nrows+1; j < extNrows; j++ )
      {
         if ( dble_buf[j] != 0.0 )
         {
            mat_aa[nnz_count] = dble_buf[j];
            mat_ja[nnz_count++] = j;
         }
      }
      mat_ia[i+Nrows+1] = nnz_count;
      offset += recv_lengths[i];
   }
   if ( nnz_count > total_nnz )
      printf("WARNING in ILUTDecomp : memory bound passed.\n");

   /* ---------------------------------------------------------- */
   /* deallocate temporary storage space                         */
   /* ---------------------------------------------------------- */

   free(cols);
   free(vals);
   free(dble_buf);
   free(diagonal);
   free(rowNorms);
   free(context);

   return 0;
}

