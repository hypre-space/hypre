/*bhEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include "parcsr_mv/parcsr_mv.h"
#include "base/mli_defs.h"
#include "solver/mli_solver_bsgs.h"
#include "util/mli_utils.h"

/******************************************************************************
 * BSGS relaxation scheme 
 *****************************************************************************/

/******************************************************************************
 * constructor
 *--------------------------------------------------------------------------*/

MLI_Solver_BSGS::MLI_Solver_BSGS() : MLI_Solver(MLI_SOLVER_BSGS_ID)
{
   Amat_             = NULL;
   nBlocks_          = 0;
   blockLengths_     = NULL;
   blockIndices_     = NULL;
   blockInverses_    = NULL;
   relaxWeights_     = NULL;
   nSweeps_          = 1;
   zeroInitialGuess_ = 0;
   useOverlap_       = 1;
   offNRows_         = 0;
   offRowIndices_    = NULL;
   offRowLengths_    = NULL;
   offCols_          = NULL;
   offVals_          = NULL;
}

/******************************************************************************
 * destructor
 *--------------------------------------------------------------------------*/

MLI_Solver_BSGS::~MLI_Solver_BSGS()
{
   if (relaxWeights_ != NULL) delete [] relaxWeights_;
   if (blockLengths_ != NULL) delete [] blockLengths_;
   if (blockIndices_ != NULL) 
   {
      for ( int i = 0; i < nBlocks_; i++ )
         if (blockIndices_[i] != NULL) delete [] blockIndices_[i];
      delete [] blockIndices_;
      if (blockInverses_ != NULL) 
      {
         for ( int i = 0; i < nBlocks_; i++ )
         {
            for ( int j = 0; j < nBlocks_; j++ )
               if (blockInverses_[i][j] != NULL) delete blockInverses_[i][j];
            delete blockInverses_[i];
         }
         delete [] blockInverses_;
      }
   }
   if ( offRowIndices_ != NULL ) delete [] offRowIndices_;
   if ( offRowLengths_ != NULL ) delete [] offRowLengths_;
   if ( offCols_       != NULL ) delete [] offCols_;
   if ( offVals_       != NULL ) delete [] offVals_;
}

/******************************************************************************
 * setup 
 *--------------------------------------------------------------------------*/

int MLI_Solver_BSGS::setup(MLI_Matrix *Amat_in)
{
   int                nprocs;
   MPI_Comm           comm;
   hypre_ParCSRMatrix *A;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   Amat_ = Amat_in;
   A     = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm  = hypre_ParCSRMatrixComm(A);
   MPI_Comm_size(comm,&nprocs);  
   
   /*-----------------------------------------------------------------
    * fetch the extended (to other processors) portion of the matrix 
    *-----------------------------------------------------------------*/

   if ( nprocs > 1 && useOverlap_ != 0 ) composeOverlappedMatrix();

   /*-----------------------------------------------------------------
    * construct the extended matrix
    *-----------------------------------------------------------------*/

   buildBlocks();
   adjustOffColIndices();

   return 0;
}

/******************************************************************************
 * solve function
 *---------------------------------------------------------------------------*/

int MLI_Solver_BSGS::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   int     ip, nRecvs, *recvProcs, *recvStarts, nRecvBefore;
   int     blockStartRow, ib, is, js, blockSize, blockEndRow, blkLeng;
   int     localNRows, iStart, iEnd, irow, jcol, colIndex, index, mypid;
   int     nSends, numColsOffd, start, relaxError=0, maxBlkLeng;
   int     nprocs, *partition, startRow, endRow, offOffset, *tmpJ;
   int     *ADiagI, *ADiagJ, *AOffdI, *AOffdJ, offIRow, totalOffNNZ;
   double  *ADiagA, *AOffdA, *uData, *fData, *tmpA, ddiag, *fExtData;
   double  relaxWeight, *vBufData, *vExtData, res, *blkAX, *blkX;
   MPI_Comm               comm;
   hypre_ParCSRMatrix     *A;
   hypre_CSRMatrix        *ADiag, *AOffd;
   hypre_ParCSRCommPkg    *commPkg;
   hypre_ParCSRCommHandle *commHandle;
   hypre_ParVector        *f, *u;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A           = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm        = hypre_ParCSRMatrixComm(A);
   commPkg     = hypre_ParCSRMatrixCommPkg(A);
   ADiag       = hypre_ParCSRMatrixDiag(A);
   localNRows  = hypre_CSRMatrixNumRows(ADiag);
   ADiagI      = hypre_CSRMatrixI(ADiag);
   ADiagJ      = hypre_CSRMatrixJ(ADiag);
   ADiagA      = hypre_CSRMatrixData(ADiag);
   AOffd       = hypre_ParCSRMatrixOffd(A);
   numColsOffd = hypre_CSRMatrixNumCols(AOffd);
   AOffdI      = hypre_CSRMatrixI(AOffd);
   AOffdJ      = hypre_CSRMatrixJ(AOffd);
   AOffdA      = hypre_CSRMatrixData(AOffd);
   u           = (hypre_ParVector *) u_in->getVector();
   uData       = hypre_VectorData(hypre_ParVectorLocalVector(u));
   f           = (hypre_ParVector *) f_in->getVector();
   fData       = hypre_VectorData(hypre_ParVectorLocalVector(f));
   partition   = hypre_ParVectorPartitioning(f);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  
   startRow    = partition[mypid];
   endRow      = partition[mypid+1] - 1;
   free( partition );
   nRecvBefore = 0;
   totalOffNNZ = 0;
   if ( nprocs > 1 )
   {
      nRecvs      = hypre_ParCSRCommPkgNumRecvs(commPkg);
      recvProcs   = hypre_ParCSRCommPkgRecvProcs(commPkg);
      recvStarts  = hypre_ParCSRCommPkgRecvVecStarts(commPkg);
      if ( useOverlap_ )
      {
         for ( ip = 0; ip < nRecvs; ip++ )
            if ( recvProcs[ip] > mypid ) break;
         nRecvBefore = recvStarts[ip];
         offNRows_   = recvStarts[nRecvs];
         totalOffNNZ = 0;
         for ( ip = 0; ip < offNRows_; ip++ )
            totalOffNNZ += offRowLengths_[ip];
      } 
   }
   blockSize = ( localNRows + offNRows_ ) / nBlocks_;
printf("%d : nRecvBefore = %d\n", mypid, nRecvBefore);

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = hypre_ParCSRCommPkgNumSends(commPkg);
      vBufData = new double[hypre_ParCSRCommPkgSendMapStart(commPkg,nSends)];
      vExtData = new double[numColsOffd];
      fExtData = new double[numColsOffd];

      if (numColsOffd)
      {
         AOffdJ = hypre_CSRMatrixJ(AOffd);
         AOffdA = hypre_CSRMatrixData(AOffd);
      }
   }

   /*--------------------------------------------------------------------
    * communicate right hand side
    *--------------------------------------------------------------------*/

   if (nprocs > 1 && useOverlap_)
   {
      index = 0;
      for (is = 0; is < nSends; is++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg, is);
         for (js=start;js<hypre_ParCSRCommPkgSendMapStart(commPkg,is+1);js++)
            vBufData[index++]
                      = fData[hypre_ParCSRCommPkgSendMapElmt(commPkg,js)];
      }
      commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                fExtData);
      hypre_ParCSRCommHandleDestroy(commHandle);
      commHandle = NULL;
   }

printf("%d : %d %d \n", mypid, startRow, endRow);
   /*-----------------------------------------------------------------
    * perform block SGS sweeps
    *-----------------------------------------------------------------*/
 
   maxBlkLeng = 1;
   if ( blockLengths_ != NULL ) 
   {
      for ( ib = 0; ib < nBlocks_; ib++ )
         if (blockLengths_[ib] > maxBlkLeng) maxBlkLeng = blockLengths_[ib];
   }
   blkX  = new double[maxBlkLeng];
   blkAX = new double[maxBlkLeng];

   for( is = 0; is < nSweeps_; is++ )
   {
      if ( relaxWeights_ != NULL ) relaxWeight = relaxWeights_[is];
      else                         relaxWeight = 1.0;

      /*-----------------------------------------------------------------
       * communicate data on processor boundaries
       *-----------------------------------------------------------------*/

      if (nprocs > 1)
      {
         if ( ! zeroInitialGuess_)
         {
            index = 0;
            for (is = 0; is < nSends; is++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(commPkg, is);
               for (js=start;js<hypre_ParCSRCommPkgSendMapStart(commPkg,is+1);
                    js++)
                  vBufData[index++]
                      = uData[hypre_ParCSRCommPkgSendMapElmt(commPkg,js)];
            }
            commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                      vExtData);
            hypre_ParCSRCommHandleDestroy(commHandle);
            commHandle = NULL;
         }
      }

printf("%d : %d %d \n", mypid, startRow, endRow);
      /*-----------------------------------------------------------------
       * process each block forward
       *-----------------------------------------------------------------*/

      offOffset = 0;
      offIRow   = 0;
      for ( ib = 0; ib < nBlocks_; ib++ )
      {
         if ( blockLengths_ != NULL ) blkLeng = blockLengths_[ib];
         else                         blkLeng = 1;
         blockStartRow = ib * blockSize + startRow - nRecvBefore;
         blockEndRow   = blockStartRow + blkLeng - 1;

         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            index  = irow - startRow;
            if ( irow < startRow )
            {
               iStart = 0;
               iEnd   = offRowLengths_[offIRow];
               tmpA   = &(offVals_[offOffset]);
               tmpJ   = &(offCols_[offOffset]);
               res    = fExtData[offIRow++];
               ddiag  = 1.0 / tmpA[0];
               for (jcol = iStart; jcol < iEnd; jcol++)
               {
                  colIndex = *tmpJ++;
                  if ( colIndex >= localNRows )   
                     res -= *tmpA++ * vExtData[colIndex-localNRows];
                  else if ( colIndex >= 0 )   
                     res -= *tmpA++ * uData[colIndex];
                  else tmpA++;
               }
               offOffset += iEnd;
               blkX[irow-blockStartRow] = res;
            }
            else if ( irow <= endRow )
            {
               iStart = ADiagI[index];
               iEnd   = ADiagI[index+1];
               tmpJ   = &(ADiagJ[iStart]);
               tmpA   = &(ADiagA[iStart]);
               res    = fData[index];
               for (jcol = iStart; jcol < iEnd; jcol++)
                  res -= *tmpA++ * uData[*tmpJ++];
               if ( ! zeroInitialGuess_)
               {
                  iStart = AOffdI[index];
                  iEnd   = AOffdI[index+1];
                  tmpJ   = &(AOffdJ[iStart]);
                  tmpA   = &(AOffdA[iStart]);
                  for (jcol = iStart; jcol < iEnd; jcol++)
                     res -= *tmpA++ * vExtData[*tmpJ++];
               }
               blkX[irow-blockStartRow] = res;
               ddiag = 1.0 / ADiagA[ADiagI[index]];
            } 
            else if ( irow > endRow )
            {
               iStart = 0;
               iEnd   = offRowLengths_[offIRow];
               tmpA   = &(offVals_[offOffset]);
               tmpJ   = &(offCols_[offOffset]);
               res    = fExtData[offIRow++];
               ddiag  = 1.0 / tmpA[0];
               for (jcol = iStart; jcol < iEnd; jcol++)
               {
                  colIndex = *tmpJ++;
                  if ( colIndex >= localNRows )   
                     res -= *tmpA++ * vExtData[colIndex-localNRows];
                  else if ( colIndex >= 0 )   
                     res -= *tmpA++ * uData[colIndex];
                  else tmpA++;
               }
               offOffset += iEnd;
               blkX[irow-blockStartRow] = res;
            }
         }

         if ( blkLeng == 1 ) blkAX[0] = blkX[0] * ddiag;
         else 
            MLI_Utils_Matvec(blockInverses_[ib], blkLeng, blkX, blkAX);

         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            if ( irow < startRow )
            {
               vExtData[offIRow-blockSize+irow-blockStartRow] += 
                               relaxWeight * blkAX[irow-blockStartRow];
            }
            else if ( irow <= endRow )
            { 
               uData[irow-startRow] += relaxWeight * blkAX[irow-blockStartRow];
            }
            else 
            {
               vExtData[offIRow-blockSize+irow-blockStartRow] += 
                               relaxWeight * blkAX[irow-blockStartRow];
            }
         }
      }

      /*-----------------------------------------------------------------
       * process each block backward
       *-----------------------------------------------------------------*/

      offOffset = offIRow = 0;
      if ( offRowLengths_ != NULL ) 
      {
         offOffset = totalOffNNZ - offRowLengths_[offNRows_-1];
         offIRow   = offNRows_ - 1;
      }
      for ( ib = nBlocks_-1; ib >= 0; ib-- )
      {
         if ( blockLengths_ != NULL ) blkLeng = blockLengths_[ib];
         else                         blkLeng = 1;
         blockStartRow = ib * blockSize + startRow - nRecvBefore;
         blockEndRow   = blockStartRow + blkLeng - 1;

         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            index  = irow - startRow;
            if ( irow < startRow )
            {
               iStart = 0;
               iEnd   = offRowLengths_[offIRow];
               tmpA   = &(offVals_[offOffset]);
               tmpJ   = &(offCols_[offOffset]);
               res    = fExtData[offIRow--];
               ddiag  = 1.0 / tmpA[0];
               for (jcol = iStart; jcol < iEnd; jcol++)
               {
                  colIndex = *tmpJ++;
                  if ( colIndex >= localNRows )   
                     res -= *tmpA++ * vExtData[colIndex-localNRows];
                  else if ( colIndex >= 0 )   
                     res -= *tmpA++ * uData[colIndex];
                  else tmpA++;
               }
               offOffset -= iEnd;
               blkX[irow-blockStartRow] = res;
            }
            else if ( irow <= endRow )
            {
               iStart = ADiagI[index];
               iEnd   = ADiagI[index+1];
               res    = fData[index];
               tmpJ   = &(ADiagJ[iStart]);
               tmpA   = &(ADiagA[iStart]);
               for (jcol = iStart; jcol < iEnd; jcol++)
                  res -= *tmpA++ * uData[*tmpJ++];
               if ( ! zeroInitialGuess_ )
               {
                  iStart = AOffdI[index];
                  iEnd   = AOffdI[index+1];
                  tmpJ   = &(AOffdJ[iStart]);
                  tmpA   = &(AOffdA[iStart]);
                  for (jcol = iStart; jcol < iEnd; jcol++)
                     res -= *tmpA++ * vExtData[*tmpJ++];
               }
               blkX[irow-blockStartRow] = res;
               ddiag = 1.0 / ADiagA[ADiagI[index]];
            } 
            else if ( irow > endRow )
            {
               iStart = 0;
               iEnd   = offRowLengths_[offIRow];
               tmpA   = &(offVals_[offOffset]);
               tmpJ   = &(offCols_[offOffset]);
               res    = fExtData[offIRow--];
               ddiag  = 1.0 / tmpA[0];
               for (jcol = iStart; jcol < iEnd; jcol++)
               {
                  colIndex = *tmpJ++;
                  if ( colIndex >= localNRows )   
                     res -= *tmpA++ * vExtData[colIndex-localNRows];
                  else if ( colIndex >= 0 )   
                     res -= *tmpA++ * uData[colIndex];
                  else tmpA++;
               }
               offOffset -= iEnd;
               blkX[irow-blockStartRow] = res;
            }
         }

         if ( blkLeng == 1 ) blkAX[0] = blkX[0] * ddiag;
         else 
            MLI_Utils_Matvec(blockInverses_[ib], blkLeng, blkX, blkAX);

         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            if ( irow < startRow )
            {
               vExtData[offIRow-blockSize+irow-blockStartRow] += 
                               relaxWeight * blkAX[irow-blockStartRow];
            }
            else if ( irow <= endRow )
            { 
               uData[irow-startRow] += relaxWeight * blkAX[irow-blockStartRow];
            }
            else 
            {
               vExtData[offIRow-blockSize+irow-blockStartRow] += 
                               relaxWeight * blkAX[irow-blockStartRow];
            }
         }
      }
      zeroInitialGuess_ = 0;
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      delete [] vExtData;
      delete [] vBufData;
      delete [] fExtData;
   }
   delete [] blkAX;
   delete [] blkX;
   return(relaxError); 
}

/******************************************************************************
 * set parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_BSGS::setParams(char *param_string, int argc, char **argv)
{
   int    i;
   double *weights;
   char   param1[200];

   if ( !strcasecmp(param_string, "nblocks") )
   {
      sscanf(param_string, "%s %d", param1, &nBlocks_);
      if ( nBlocks_ < 1 ) nBlocks_ = -1;
      return 0;
   }
   else if ( !strcasecmp(param_string, "numSweeps") )
   {
      sscanf(param_string, "%s %d", param1, &nSweeps_);
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param_string, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("Solver_BSGS::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nSweeps_ = *(int*)   argv[0];
      if ( argc == 2 ) weights  = (double*) argv[1];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = NULL;
      if ( weights != NULL )
      {
         relaxWeights_ = new double[nSweeps_];
         for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = weights[i];
      }
   }
   else if ( !strcasecmp(param_string, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
      return 0;
   }
   return 1;
}

/******************************************************************************
 * compose overlapped matrix
 *--------------------------------------------------------------------------*/

int MLI_Solver_BSGS::composeOverlappedMatrix()
{
   hypre_ParCSRMatrix *A;
   MPI_Comm    comm;
   MPI_Request *requests;
   MPI_Status  *status;
   int         i, j, k, mypid, nprocs, *partition, startRow, endRow;
   int         localNRows, extNRows, nSends, *sendProcs, nRecvs;
   int         *recvProcs, *recvStarts, proc, offset, length, reqNum; 
   int         totalSendNnz, totalRecvNnz, index, base, totalSends;
   int         totalRecvs, rowNum, rowSize, *colInd, *sendStarts;
   int         limit, *iSendBuf, curNnz, *recvIndices; 
   double      *dSendBuf, *colVal;
   hypre_ParCSRCommPkg *commPkg;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters 
    *-----------------------------------------------------------------*/

   A = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   startRow   = partition[mypid];
   endRow     = partition[mypid+1] - 1;
   localNRows = endRow - startRow + 1;
   free( partition );

   /*-----------------------------------------------------------------
    * fetch matrix communication information (offNRows_)
    *-----------------------------------------------------------------*/

   extNRows = localNRows;
   if ( nprocs > 1 && useOverlap_ )
   {
      commPkg    = hypre_ParCSRMatrixCommPkg(A);
      nSends     = hypre_ParCSRCommPkgNumSends(commPkg);
      sendProcs  = hypre_ParCSRCommPkgSendProcs(commPkg);
      sendStarts = hypre_ParCSRCommPkgSendMapStarts(commPkg);
      nRecvs     = hypre_ParCSRCommPkgNumRecvs(commPkg);
      recvProcs  = hypre_ParCSRCommPkgRecvProcs(commPkg);
      recvStarts = hypre_ParCSRCommPkgRecvVecStarts(commPkg);
      for ( i = 0; i < nRecvs; i++ ) 
         extNRows += ( recvStarts[i+1] - recvStarts[i] );
      requests = new MPI_Request[nRecvs+nSends];
      totalSends  = sendStarts[nSends];
      totalRecvs  = recvStarts[nRecvs];
      if ( totalRecvs > 0 ) offRowLengths_ = new int[totalRecvs];
      else                  offRowLengths_ = NULL;
      recvIndices = hypre_ParCSRMatrixColMapOffd(A);
      if ( totalRecvs > 0 ) offRowIndices_ = new int[totalRecvs];
      else                  offRowIndices_ = NULL;
      for ( i = 0; i < totalRecvs; i++ ) 
         offRowIndices_[i] = recvIndices[i];
      offNRows_ = totalRecvs;
   }
   else nRecvs = nSends = offNRows_ = totalRecvs = totalSends = 0;

   /*-----------------------------------------------------------------
    * construct offRowLengths 
    *-----------------------------------------------------------------*/

   reqNum = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      MPI_Irecv(&(offRowLengths_[offset]), length, MPI_INT, proc, 
                17304, comm, &(requests[reqNum++]));
   }
   if ( totalSends > 0 ) iSendBuf = new int[totalSends];

   index = totalSendNnz = 0;
   for (i = 0; i < nSends; i++)
   {
      proc   = sendProcs[i];
      offset = sendStarts[i];
      limit  = sendStarts[i+1];
      length = limit - offset;
      for (j = offset; j < limit; j++)
      {
         rowNum = hypre_ParCSRCommPkgSendMapElmt(commPkg,j) + startRow;
         hypre_ParCSRMatrixGetRow(A,rowNum,&rowSize,&colInd,NULL);
         iSendBuf[index++] = rowSize;
         totalSendNnz += rowSize;
         hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowSize,&colInd,NULL);
      }
      MPI_Isend(&(iSendBuf[offset]), length, MPI_INT, proc, 17304, comm, 
                &(requests[reqNum++]));
   }
   status = new MPI_Status[reqNum];
   MPI_Waitall( reqNum, requests, status );
   delete [] status;
   if ( totalSends > 0 ) delete [] iSendBuf;

   /*-----------------------------------------------------------------
    * construct offCols 
    *-----------------------------------------------------------------*/

   totalRecvNnz = 0;
   for (i = 0; i < totalRecvs; i++) totalRecvNnz += offRowLengths_[i];
   if ( totalRecvNnz > 0 )
   {
      offCols_ = new int[totalRecvNnz];
      offVals_ = new double[totalRecvNnz];
   }
   reqNum = totalRecvNnz = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      curNnz = 0;
      for (j = 0; j < length; j++) curNnz += offRowLengths_[offset+j];
      MPI_Irecv(&(offCols_[totalRecvNnz]), curNnz, MPI_INT, proc, 17305, 
                comm, &(requests[reqNum++]));
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) iSendBuf = new int[totalSendNnz];

   index = totalSendNnz = 0;
   for (i = 0; i < nSends; i++)
   {
      proc   = sendProcs[i];
      offset = sendStarts[i];
      limit  = sendStarts[i+1];
      length = limit - offset;
      base   = totalSendNnz;
      for (j = offset; j < limit; j++)
      {
         rowNum = hypre_ParCSRCommPkgSendMapElmt(commPkg,j) + startRow;
         hypre_ParCSRMatrixGetRow(A,rowNum,&rowSize,&colInd,NULL);
         for (k = 0; k < rowSize; k++) 
            iSendBuf[totalSendNnz++] = colInd[k];
         hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowSize,&colInd,NULL);
      }
      length = totalSendNnz - base;
      MPI_Isend(&(iSendBuf[base]), length, MPI_INT, proc, 17305, comm, 
                &(requests[reqNum++]));
   }
   status = new MPI_Status[reqNum];
   if ( reqNum > 0 ) MPI_Waitall( reqNum, requests, status );
   delete [] status;
   if ( totalSendNnz > 0 ) delete [] iSendBuf;

   /*-----------------------------------------------------------------
    * construct offVals 
    *-----------------------------------------------------------------*/

   reqNum = totalRecvNnz = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc    = recvProcs[i];
      offset  = recvStarts[i];
      length  = recvStarts[i+1] - offset;
      curNnz = 0;
      for (j = 0; j < length; j++) curNnz += offRowLengths_[offset+j];
      MPI_Irecv(&(offVals_[totalRecvNnz]), curNnz, MPI_DOUBLE, proc, 
                17306, comm, &(requests[reqNum++]));
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) dSendBuf = new double[totalSendNnz];

   index = totalSendNnz = 0;
   for (i = 0; i < nSends; i++)
   {
      proc   = sendProcs[i];
      offset = sendStarts[i];
      limit  = sendStarts[i+1];
      length = limit - offset;
      base   = totalSendNnz;
      for (j = offset; j < limit; j++)
      {
         rowNum = hypre_ParCSRCommPkgSendMapElmt(commPkg,j) + startRow;
         hypre_ParCSRMatrixGetRow(A,rowNum,&rowSize,NULL,&colVal);
         for (k = 0; k < rowSize; k++) 
            dSendBuf[totalSendNnz++] = colVal[k];
         hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowSize,NULL,&colVal);
      }
      length = totalSendNnz - base;
      MPI_Isend(&(dSendBuf[base]), length, MPI_DOUBLE, proc, 17306, comm, 
                &(requests[reqNum++]));
   }
   status = new MPI_Status[reqNum];
   if ( reqNum > 0 ) MPI_Waitall( reqNum, requests, status );
   delete [] status;
   if ( totalSendNnz > 0 ) delete [] dSendBuf;

   if ( nprocs > 1 && useOverlap_ ) delete [] requests;
   return 0;
}

/******************************************************************************
 * build the blocks 
 *--------------------------------------------------------------------------*/

int MLI_Solver_BSGS::buildBlocks()
{
   int         ib, ii, ip, mypid, nprocs, *partition, startRow, endRow;
   int         localNRows, nRecvs;
   int         *recvProcs, *recvStarts, nRecvBefore=0; 
   int         blockSize, rowOffset, nnzOffset, blockStartRow, blockEndRow;
   int         irow, jcol, colIndex, rowSize, *colInd;
   int         maxBlkSize, blkLeng;
   double      *colVal, **tempBlock, **tempBlockInverse;
   MPI_Comm    comm;
   hypre_ParCSRCommPkg *commPkg;
   hypre_ParCSRMatrix  *A = (hypre_ParCSRMatrix *) Amat_->getMatrix();

   /*-----------------------------------------------------------------
    * clean up first 
    *-----------------------------------------------------------------*/

   if ( nBlocks_ > 0 ) 
   {
      if ( blockIndices_ != NULL )
         for ( ib = 0; ib < nBlocks_; ib++ ) delete [] blockIndices_[ib];
      delete [] blockIndices_;
      blockIndices_ = NULL;
      if ( blockInverses_ != NULL )
      {
         for ( ib = 0; ib < nBlocks_; ib++ ) 
         {
            for ( ii = 0; ii < nBlocks_; ii++ ) 
               delete [] blockInverses_[ib][ii];
            delete [] blockInverses_[ib];
         }
         delete [] blockInverses_;
      }
      blockInverses_ = NULL;
      if (blockLengths_ != NULL) delete [] blockLengths_;
      blockLengths_ = NULL;
   }

   /*-----------------------------------------------------------------
    * fetch matrix information 
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   startRow   = partition[mypid];
   endRow     = partition[mypid+1] - 1;
   localNRows = endRow - startRow + 1;
   if ( nprocs > 1 && useOverlap_ )
   {
      commPkg     = hypre_ParCSRMatrixCommPkg(A);
      nRecvs      = hypre_ParCSRCommPkgNumRecvs(commPkg);
      recvProcs   = hypre_ParCSRCommPkgRecvProcs(commPkg);
      recvStarts  = hypre_ParCSRCommPkgRecvVecStarts(commPkg);
      nRecvBefore = 0;
      for ( ip = 0; ip < nRecvs; ip++ )
         if ( recvProcs[ip] > mypid ) break;
      nRecvBefore = recvStarts[ip];
   }
   else nRecvs = 0;

   /*-----------------------------------------------------------------
    * compute block information (blockLengths_, blockIndices_ not used
    * for now assuming contiguity)
    *-----------------------------------------------------------------*/

   if ( nBlocks_ <= 0 ) nBlocks_ = ( localNRows + offNRows_ );
   if ( nBlocks_ >= (localNRows+offNRows_) ) 
   {
      nBlocks_ = localNRows + offNRows_;
      return 0;
   }
   else blockSize = (localNRows + offNRows_) / nBlocks_;
   blockLengths_ = new int[nBlocks_];
   for ( ib = 0; ib < nBlocks_; ib++ ) blockLengths_[ib] = blockSize;
   blockLengths_[nBlocks_-1] = localNRows+offNRows_-blockSize*(nBlocks_-1);

   /*-----------------------------------------------------------------
    * build inverses (blockInverses_)
    *-----------------------------------------------------------------*/

   maxBlkSize = 0;
   for ( ib = 0; ib < nBlocks_; ib++ ) 
      if ( blockLengths_[ib] > maxBlkSize ) maxBlkSize = blockLengths_[ib];
   tempBlock = new double*[maxBlkSize];
   for ( ib = 0; ib < maxBlkSize; ib++ ) 
      tempBlock[ib] = new double[maxBlkSize];
   blockInverses_ = new double**[nBlocks_];
   for ( ib = 0; ib < nBlocks_; ib++ ) 
   {
      blockInverses_[ib] = new double*[blockLengths_[ib]];
      for ( ii = 0; ii < blockLengths_[ib]; ii++ ) 
         blockInverses_[ib][ii] = new double[blockLengths_[ib]];
   }
   rowOffset = 0;
   nnzOffset = 0;
   for ( ib = 0; ib < nBlocks_; ib++ )
   {
      blkLeng       = blockLengths_[ib];
      blockStartRow = ib * blockSize + startRow - nRecvBefore;
      blockEndRow   = blockStartRow + blkLeng - 1;
      for ( irow = 0; irow < blkLeng; irow++ )
         for ( jcol = 0; jcol < blkLeng; jcol++ )
            tempBlock[irow][jcol] = 0.0;
      for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
      {
         if ( irow < startRow )
         {
            rowSize = offRowLengths_[rowOffset];
            colInd = &(offCols_[nnzOffset]);
            colVal = &(offVals_[nnzOffset]);
            for ( jcol = 0; jcol < rowSize; jcol++ )
            {
               colIndex = colInd[jcol];
               if ((colIndex >= blockStartRow) && (colIndex <= blockEndRow))
                  tempBlock[irow-blockStartRow][colIndex-blockStartRow] = 
                           colVal[jcol];
            }
            rowOffset++;
            nnzOffset += rowSize;
         }
         else if ( irow <= endRow )
         {
            hypre_ParCSRMatrixGetRow(A, irow, &rowSize, &colInd, &colVal);
            for ( jcol = 0; jcol < rowSize; jcol++ )
            {
               colIndex = colInd[jcol];
               if ((colIndex >= blockStartRow) && (colIndex <= blockEndRow))
                  tempBlock[irow-blockStartRow][colIndex-blockStartRow] = 
                           colVal[jcol];
            }
            hypre_ParCSRMatrixRestoreRow(A,irow,&rowSize,&colInd,&colVal);
         }
         else
         {
            rowSize = offRowLengths_[rowOffset];
            colInd = &(offCols_[nnzOffset]);
            colVal = &(offVals_[nnzOffset]);
            for ( jcol = 0; jcol < rowSize; jcol++ )
            {
               colIndex = colInd[jcol];
               if ((colIndex >= blockStartRow) && (colIndex <= blockEndRow))
                  tempBlock[irow-blockStartRow][colIndex-blockStartRow] = 
                           colVal[jcol];
            }
            rowOffset++;
            nnzOffset += rowSize;
         }
      }
#if 0
      for ( irow = 0; irow < blkLeng; irow++ )
         for ( jcol = 0; jcol < blkLeng; jcol++ )
            printf("%d : (%6d,%6d) = %e\n",ib,irow,jcol,tempBlock[irow][jcol]);
#endif
      MLI_Utils_MatrixInverse( tempBlock, blkLeng, &tempBlockInverse ); 
      for ( irow = 0; irow < blkLeng; irow++ )
      {
         for ( jcol = 0; jcol < blkLeng; jcol++ )
            blockInverses_[ib][irow][jcol] = tempBlockInverse[irow][jcol];
         free( tempBlockInverse[irow] );
      }
      free( tempBlockInverse );
   }
   for ( ib = 0; ib < maxBlkSize; ib++ ) delete [] tempBlock[ib];
   delete [] tempBlock;
   return 0;
}

/******************************************************************************
 * adjust the off processor incoming matrix
 *--------------------------------------------------------------------------*/

int MLI_Solver_BSGS::adjustOffColIndices()
{
   int                mypid, *partition, startRow, endRow;
   int                offset, index, colIndex, irow, jcol;
   hypre_ParCSRMatrix *A;
   MPI_Comm           comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters 
    *-----------------------------------------------------------------*/

   A = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   startRow   = partition[mypid];
   endRow     = partition[mypid+1] - 1;
   free( partition );

   /*-----------------------------------------------------------------
    * convert column indices
    *-----------------------------------------------------------------*/

   offset = 0;
   for ( irow = 0; irow < offNRows_; irow++ )
   {
      for ( jcol = 0; jcol < offRowLengths_[irow]; jcol++ )
      {
         colIndex = offCols_[offset];
         if ( colIndex >= startRow && colIndex <= endRow )
            offCols_[offset] = colIndex - startRow;
         else
         {
            index = MLI_Utils_BinarySearch(colIndex,offRowIndices_,offNRows_);
            if ( index >= 0 ) offCols_[offset] = index;
            else              offCols_[offset] = -1;
         }
         offset++;
      }
   }
   return 0;
}

