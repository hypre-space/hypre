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

#include "HYPRE_MHMatrix.h"

typedef struct
{
    MPI_Comm  comm;
    MH_Matrix *mh_mat;
    double    thresh;
    int       fillin;
}
DD_Ilut;

/*--------------------------------------------------------------------------
 * HYPRE_DDIlutCreate - Return a DDIlut preconditioner object "solver".  
 *--------------------------------------------------------------------------*/

int HYPRE_DDIlutCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   DDIlut *ilut_ptr;
   
   ilut_ptr = (DD_Ilut *) malloc(sizeof(DDIlut));

   if (ilut_ptr == NULL) return 1;

   ilut_ptr->comm    = comm;
   ilut_ptr->mh_mat  = NULL;
   ilut_ptr->fillin  = 0;
   ilut_ptr->thresh  = 0.0; /* defaults */

   *solver = (HYPRE_Solver) ilut_ptr;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_DDIlutDestroy - Destroy a DDIlut object.
 *--------------------------------------------------------------------------*/

int HYPRE_DDIlutDestroy( HYPRE_Solver solver )
{
   DD_Ilut *ilut_ptr;

   ilut_ptr = (DD_Ilut *) solver;
   /*if ( ilut_ptr->mh_mat != NULL ) free(ilut_ptr->mh_mat); */

   free(ilut_ptr);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_DDIlutSetup - Set up function for DDIlut.
 *--------------------------------------------------------------------------*/

int HYPRE_DDIlutSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int     i, j, ierr = 0, nprocs, rowStart, rowEnd, mypid, rowSize, *colInd;
   int     localNRows, localExtNRows, nSend, nRecv, *sendList, *sendLeng;
   int     *recvLeng, *recvProc, *sendProc, **tempSend, **tempRecv;
   int     msgid, length, procNum, *int_buf, *in_int_buf;
   double  *colVal, *dble_buf, *in_dble_buf;
   DD_Ilut *ilut_ptr = (DD_Ilut *) solver;
   MPI_Status status;
   MH_Context *context;
   MH_Matrix  *mh_mat;

   /* ----------------------------------------------------------- */
   /* get the row information in my processors                    */
   /* ----------------------------------------------------------- */

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &rowPartition);
   rowStart = rowPartition[mypid];
   rowEnd   = rowPartition[mypid+1] - 1;

   /* ----------------------------------------------------------- */
   /* convert the incoming CSR matrix into a MH matrix            */
   /* ----------------------------------------------------------- */

   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->comm = MPIlink->comm;
   context->globalEqns = rowPartition[nprocs];
   context->partition = (int *) malloc(sizeof(int)*(nprocs+1));
   for (i=0; i<=nprocs; i++) context->partition[i] = rowPartition[i];
   hypre_TFree( row_partition );
   mh_mat = ( MH_Matrix * ) malloc( sizeof( MH_Matrix) );
   context->Amat = mh_mat;
   HYPRE_ParCSRMLConstructMHMatrix(A_csr,mh_mat,MPI_COMM_WORLD,
                                   context->partition,context); 

   /* ----------------------------------------------------------- */
   /* extract information from mh_matrix                          */
   /* ----------------------------------------------------------- */
   
   nSend = mh_mat->sendProcCnt;
   nRecv = mh_mat->recvProcCnt;
   sendLeng = mh_mat->sendLeng;
   recvLeng = mh_mat->recvLeng;
   sendProc = mh_mat->sendProc;
   recvProc = mh_mat->recvProc;
   sendList = mh_mat->sendList;
   mat_ia = mh_mat->rowptr;
   mat_ja = mh_mat->colnum;
   mat_aa = mh_mat->values;

   /* ----------------------------------------------------------- */
   /* allocate buffers                                            */
   /* ----------------------------------------------------------- */

   if ( nSend > 0 ) 
   {
      tempSend = (int ** ) malloc( nSend * sizeof (int*) );
      for ( j = 0; j < nSend; j++ ) 
         tempSend[i] = (int *) malloc( sendLeng[i] * sizeof (int) );
   }
   if ( nRecv > 0 ) 
   {
      tempRecv = (int ** ) malloc( nRecv * sizeof (int*) );
      for ( j = 0; j < nRecv; j++ ) 
         tempSend[i] = (int *) malloc( recvLeng[i] * sizeof (int) );
   }

   /* ----------------------------------------------------------- */
   /* communicate length information for additional rows          */
   /* ----------------------------------------------------------- */

   msgid = 539;
   for ( i = 0; i < nSend; i++ ) 
   {
      for ( j = 0; j < sendLeng[i]; j++ ) 
      {
         rowNum = sendList[i][j] + context->partition[mypid];
         HYPRE_ParCSRMatrixGetRow(A_csr,rowNum,&rowSize,&colInd,&colVal);
         tempSend[i][j] = rowSize;
         HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
      }
      length = sendLeng[i];
      procNum = sendProc[i];
      MPI_Send((void*)&tempSend[i],length,MPI_INT,procNum,msgid,MPI_COMM_WORLD);
   }

   for ( i = 0; i < nRecv; i++ ) 
   {
      length = recvLeng[i];
      procNum = recvProc[i];
      MPI_Recv((void*) &tempRecv[i],length,MPI_INT,procNum,msgid,
               MPI_COMM_WROLD,&status);
   }

   /* ----------------------------------------------------------- */
   /* communicate the additional row information                  */
   /* ----------------------------------------------------------- */

   localExtNRows = localNRows;
   for ( i = 0; i < nRecv; i++ ) localExtNRows += recvLeng[i]; 
   new_nnz = mat_a[localNRows]; 
   for ( i = 0; i < nRecv; i++ ) 
      for ( j = 0; j < recvLeng[i]; j++ ) 
         new_nnz += tempRecv[i][j]; 
   new_mat_ia = (int *)    malloc( (localExtNRows+1) * sizeof(int) ); 
   new_mat_ja = (int *)    malloc( new_nnz * sizeof(int) ); 
   new_mat_aa = (double *) malloc( new_nnz * sizeof(double) ); 
   for ( i = 0; i <= localNRows; i++ ) new_mat_ia[i] = mat_ia[i];
   for ( i = 0; i < mat_ia[localNRows]; i++ )
   {
      new_mat_ja[i] = mat_ja[i];
      new_mat_aa[i] = mat_aa[i];
   }
   free( mat_ia );
   free( mat_ja );
   free( mat_aa );
   msgid = 2539;
   for ( i = 0; i < nSend; i++ ) 
   {
      length = 0;
      for ( j = 0; j < sendLeng[i]; j++ ) length += tempSend[i][j]; 
      int_buf  = new int[length];
      dble_buf = new double[length];
      ncnt = 0;
      for ( j = 0; j < sendLeng[i]; j++ ) 
      {
         rowNum = sendList[i][j] + context->partition[mypid];
         HYPRE_ParCSRMatrixGetRow(A_csr,rowNum,&rowSize,&colInd,&colVal);
         for ( k = 0; k < rowSize; k++ ) 
         {
            int_buf[ncnt] = colInd[j];
            dble_buf[ncnt++] = colVal[j];
         }
         HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
      }
      procNum = sendProc[i];
      MPI_Send((void*)int_buf,length,MPI_INT,procNum,msgid,MPI_COMM_WORLD);
      MPI_Send((void*)dble_buf,length,MPI_DOUBLE,procNum,msgid+1,
               MPI_COMM_WORLD);
      free( int_buf );
      free( dble_buf );
   }
   new_nnz = mat_ia[localNRows];
   localExtNRows = localNRows;
   for ( i = 0; i < nRecv; i++ ) 
   {
      length = 0;
      for ( j = 0; j < recvLeng[i]; j++ ) length += tempRecv[i][j]; 
      in_int_buf  = (int*)    malloc( length * sizeof(int) );
      in_dble_buf = (double*) malloc( length * sizeof(double) );
      procNum = recvProc[i];
      MPI_Recv((void*) in_int_buf,length,MPI_INT,procNum,msgid,
               MPI_COMM_WROLD,&status);
      MPI_Recv((void*) in_dble_buf,length,MPI_DOUBLE,procNum,msgid+1,
               MPI_COMM_WORLD,&status);
      for ( j = 0; j < length; j++ )
      {
         new_mat_ja[new_nnz] = in_int_buf[j];
         new_mat_aa[new_nnz++] = in_dble_buf[j];
      }   
      for ( j = 0; j < recvLeng[i]; j++ ) 
      {
         localExtNRows++;
         mat_ia[localExtNRows] = tempRecv[i][j]; 
      }
      free( in_int_buf );
      free( in_dble_buf );
   }

   /* ----------------------------------------------------------- */
   /* reinitialize the MH_matrix structure                        */
   /* ----------------------------------------------------------- */

   mh_mat->Nrows  = localExtNRows;
   mh_mat->rowptr = new_mat_ia;
   mh_mat->colnum = new_mat_ja;
   mh_mat->values = new_mat_aa;
   free( context->partition );
   context->partition = NULL;

   /* ----------------------------------------------------------- */
   /* calling the ILUT decomposition routine                      */
   /* ----------------------------------------------------------- */

   HYPRE_DDILUTDecompose( context );

   /* ----------------------------------------------------------- */
   /* clean up                                                    */
   /* ----------------------------------------------------------- */

   if ( nSend > 0 ) 
   {
      for ( j = 0; j < nSend; j++ ) free( tempSend[i] ); 
      free( tempSend );
   }
   if ( nRecv > 0 ) 
   {
      for ( j = 0; j < nRecv; j++ ) free( tempRecv[i] ); 
      free( tempRecv );
   }
   ilut_ptr->comm   = MPI_COMM_WORLD; 
   ilut_ptr->mh_mat = mh_mat; 
   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSolve - Solve function for ParaSails.
 *--------------------------------------------------------------------------*/

int HYPRE_DDIlutSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int ierr = 0;
   double *rhs, *soln;
   Secret *secret = (Secret *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_DDIlutSetFillin - Set the fill-in parameter.
 *--------------------------------------------------------------------------*/

int
HYPRE_DDIlutSetFillin(HYPRE_Solver solver, double fillin)
{
   DD_Ilut *ilut_ptr = (DD_Ilut *) solver;

   ilut_ptr->fillin = fillin;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_DDIlutSetThreshold - Set the threshold for dropping
 *--------------------------------------------------------------------------*/

int HYPRE_DDIlutSetThreshold(HYPRE_Solver solver, double thresh)
{
   DD_Ilut *ilut_ptr = (DD_Ilut *) solver;

   ilut_ptr->thresh = thresh;

   return 0;
}

