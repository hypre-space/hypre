/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include "mli_solver_bsgs.h"
#ifdef HAVE_ESSL
#include <essl.h>
#endif

#define switchSize 200

/******************************************************************************
 * BSGS relaxation scheme 
 *****************************************************************************/

/******************************************************************************
 * constructor
 *--------------------------------------------------------------------------*/

MLI_Solver_BSGS::MLI_Solver_BSGS(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   nSweeps_          = 1;
   relaxWeights_     = NULL;
   zeroInitialGuess_ = 0;
   useOverlap_       = 0;
   blockSize_        = 512;
   nBlocks_          = 0;
   blockLengths_     = NULL;
   blockSolvers_     = NULL;
   maxBlkLeng_       = 0;
   offNRows_         = 0;
   offRowIndices_    = NULL;
   offRowLengths_    = NULL;
   offCols_          = NULL;
   offVals_          = NULL;
   myColor_          = 0;
   numColors_        = 1;
   scheme_           = 1;
#ifdef HAVE_ESSL
   esslMatrices_     = NULL;
#endif
}

/******************************************************************************
 * destructor
 *--------------------------------------------------------------------------*/

MLI_Solver_BSGS::~MLI_Solver_BSGS()
{
   cleanBlocks();
   if (relaxWeights_ != NULL) delete [] relaxWeights_;
}

/******************************************************************************
 * setup 
 *--------------------------------------------------------------------------*/

int MLI_Solver_BSGS::setup(MLI_Matrix *Amat_in)
{
   hypre_ParCSRMatrix *A;
   MPI_Comm           comm;

   Amat_ = Amat_in;

   /*-----------------------------------------------------------------
    * set up coloring scheme 
    *-----------------------------------------------------------------*/

   if      ( scheme_ == 0 ) doProcColoring();
   else if ( scheme_ == 1 )
   {
      myColor_   = 0;
      numColors_ = 1;
   }
   else
   {
      A    = (hypre_ParCSRMatrix *) Amat_->getMatrix();
      comm = hypre_ParCSRMatrixComm(A);
      MPI_Comm_size(comm, &numColors_);
      MPI_Comm_rank(comm, &myColor_);
   }

   /*-----------------------------------------------------------------
    * clean up first
    *-----------------------------------------------------------------*/

   cleanBlocks();

   /*-----------------------------------------------------------------
    * fetch the extended (to other processors) portion of the matrix 
    *-----------------------------------------------------------------*/

   composeOverlappedMatrix();
   adjustOffColIndices();

   /*-----------------------------------------------------------------
    * construct the extended matrix
    *-----------------------------------------------------------------*/

   buildBlocks();

   return 0;
}

/******************************************************************************
 * solve function
 *---------------------------------------------------------------------------*/

int MLI_Solver_BSGS::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   int     iP, jP, iC, nRecvs, *recvProcs, *recvStarts, nRecvBefore;
   int     blockStartRow, iB, iS, blockEndRow, blkLeng;
   int     localNRows, iStart, iEnd, irow, jcol, colIndex, index, mypid;
   int     nSends, numColsOffd, start, relaxError=0, length;
   int     nprocs, *partition, startRow, endRow, offOffset, *tmpJ;
   int     *ADiagI, *ADiagJ, *AOffdI, *AOffdJ, offIRow, totalOffNNZ;
   double  *ADiagA, *AOffdA, *uData, *fData, *tmpA, *fExtData=NULL;
   double  relaxWeight, *vBufData=NULL, *vExtData=NULL, res;
   double  *dbleX=NULL, *dbleB=NULL;
   char    vecName[30];
   MPI_Comm               comm;
   hypre_ParCSRMatrix     *A;
   hypre_CSRMatrix        *ADiag, *AOffd;
   hypre_ParCSRCommPkg    *commPkg;
   hypre_ParCSRCommHandle *commHandle;
   hypre_ParVector        *f, *u;
   hypre_Vector           *sluB=NULL, *sluX=NULL;
   MLI_Vector             *mliX=NULL, *mliB=NULL;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
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
   nRecvBefore = 0;
   totalOffNNZ = 0;
   if ( nprocs > 1 )
   {
      nRecvs      = hypre_ParCSRCommPkgNumRecvs(commPkg);
      recvProcs   = hypre_ParCSRCommPkgRecvProcs(commPkg);
      recvStarts  = hypre_ParCSRCommPkgRecvVecStarts(commPkg);
      if ( useOverlap_ )
      {
         for ( iP = 0; iP < nRecvs; iP++ )
            if ( recvProcs[iP] > mypid ) break;
         nRecvBefore = recvStarts[iP];
         offNRows_   = recvStarts[nRecvs];
         for ( iP = 0; iP < offNRows_; iP++ )
            totalOffNNZ += offRowLengths_[iP];
      } 
   }

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = hypre_ParCSRCommPkgNumSends(commPkg);
      length = hypre_ParCSRCommPkgSendMapStart(commPkg,nSends);
      if ( length > 0 ) vBufData = new double[length];
      if ( numColsOffd > 0 )
      {
         vExtData = new double[numColsOffd];
         fExtData = new double[numColsOffd];
      }
      for ( irow = 0; irow < numColsOffd; irow++ ) vExtData[irow] = 0.0;
   }

   /*--------------------------------------------------------------------
    * communicate right hand side
    *--------------------------------------------------------------------*/

   if (nprocs > 1 && useOverlap_)
   {
      index = 0;
      for (iP = 0; iP < nSends; iP++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg, iP);
         for (jP=start;jP<hypre_ParCSRCommPkgSendMapStart(commPkg,iP+1);jP++)
         {
            vBufData[index++]
                      = 0.5 * fData[hypre_ParCSRCommPkgSendMapElmt(commPkg,jP)];
            fData[hypre_ParCSRCommPkgSendMapElmt(commPkg,jP)] *= 0.5;
         }
      }
      commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                fExtData);
      hypre_ParCSRCommHandleDestroy(commHandle);
      commHandle = NULL;
   }

   if ( maxBlkLeng_ > 0 )
   {
      dbleB = new double[maxBlkLeng_];
      dbleX = new double[maxBlkLeng_];
   }
#ifdef HAVE_ESSL
   if ( blockSize_ > switchSize )
   {
#endif
      if ( maxBlkLeng_ > 0 )
      {
         sluB  = hypre_SeqVectorCreate( maxBlkLeng_ );
         sluX  = hypre_SeqVectorCreate( maxBlkLeng_ );
      }
      hypre_VectorData(sluB) = dbleB;
      hypre_VectorData(sluX) = dbleX;
#ifdef HAVE_ESSL
   }
#endif

   /*-----------------------------------------------------------------
    * perform block SGS sweeps
    *-----------------------------------------------------------------*/
 
   for ( iS = 0; iS < nSweeps_; iS++ )
   {
      if ( relaxWeights_ != NULL ) relaxWeight = relaxWeights_[iS];
      else                         relaxWeight = 1.0;
      if ( relaxWeight <= 0.0 ) relaxWeight = 1.0;

      for ( iC = 0; iC < numColors_; iC++ )
      {
         if (nprocs > 1)
         {
            if ( ! zeroInitialGuess_)
            {
               index = 0;
               for (iP = 0; iP < nSends; iP++)
               {
                  start = hypre_ParCSRCommPkgSendMapStart(commPkg, iP);
                  for (jP=start;
                       jP<hypre_ParCSRCommPkgSendMapStart(commPkg,iP+1); jP++)
                     vBufData[index++]
                         = uData[hypre_ParCSRCommPkgSendMapElmt(commPkg,jP)];
               }
               commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                         vExtData);
               hypre_ParCSRCommHandleDestroy(commHandle);
               commHandle = NULL;
            }
         }

         /*--------------------------------------------------------------
          * process each block forward
          *--------------------------------------------------------------*/

         if ( iC == myColor_ )
         {
            offOffset = offIRow = 0;
            if ( offRowLengths_ != NULL ) 
            {
               offOffset = 0;
               offIRow--;
            }
            for ( iB = 0; iB < nBlocks_; iB++ )
            {
               blkLeng = blockLengths_[iB];
               blockStartRow = iB * blockSize_ + startRow - nRecvBefore;
               blockEndRow   = blockStartRow + blkLeng - 1;

               for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
               {
                  index  = irow - startRow;
                  if ( irow >= startRow && irow <= endRow )
                  {
                     iStart = ADiagI[index];
                     iEnd   = ADiagI[index+1];
                     tmpJ   = &(ADiagJ[iStart]);
                     tmpA   = &(ADiagA[iStart]);
                     res    = fData[index];
                     for (jcol = iStart; jcol < iEnd; jcol++)
                     {
                        res -= *tmpA * uData[*tmpJ];
                        tmpA++; tmpJ++;
                     }
                     if ( ! zeroInitialGuess_)
                     {
                        iStart = AOffdI[index];
                        iEnd   = AOffdI[index+1];
                        tmpJ   = &(AOffdJ[iStart]);
                        tmpA   = &(AOffdA[iStart]);
                        for (jcol = iStart; jcol < iEnd; jcol++)
                        {
                           res -= (*tmpA) * vExtData[(*tmpJ)];
                           tmpA++; tmpJ++;
                        }
                     }
                     dbleB[irow-blockStartRow] = res;
                  }
                  else
                  {
                     offIRow++;
                     iStart = 0;
                     iEnd   = offRowLengths_[offIRow];
                     tmpA   = &(offVals_[offOffset]);
                     tmpJ   = &(offCols_[offOffset]);
                     res    = fExtData[offIRow];
                     for (jcol = iStart; jcol < iEnd; jcol++)
                     {
                        colIndex = *tmpJ;
                        if ( colIndex >= localNRows )   
                           res -= (*tmpA) * vExtData[colIndex-localNRows];
                        else if ( colIndex >= 0 )   
                           res -= (*tmpA) * uData[colIndex];
                        tmpA++; 
                        tmpJ++;
                     }
                     offOffset += iEnd;
                     dbleB[irow-blockStartRow] = res;
                  }
               }
#ifdef HAVE_ESSL
	   if ( blockSize_ > switchSize )
	   {
#endif
               hypre_VectorSize(sluB) = blkLeng;
               hypre_VectorSize(sluX) = blkLeng;
               strcpy( vecName, "HYPRE_Vector" );
               mliB = new MLI_Vector((void*) sluB, vecName, NULL);
               mliX = new MLI_Vector((void*) sluX, vecName, NULL);

               blockSolvers_[iB]->solve( mliB, mliX );

               delete mliB;
               delete mliX;
#ifdef HAVE_ESSL
            }
            else
            {
               for (irow = 0; irow < blkLeng; irow++) dbleX[irow] = dbleB[irow];
	       dpps(esslMatrices_[iB], blkLeng, dbleX, 1);
            }
#endif

               for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
               {
                  if ( irow < startRow )
                     vExtData[offIRow-blockSize_+irow-blockStartRow+1] += 
                               relaxWeight * dbleX[irow-blockStartRow];
                  else if ( irow <= endRow )
                     uData[irow-startRow] += relaxWeight * 
                                             dbleX[irow-blockStartRow];
                  else 
                     vExtData[offIRow-blockSize_+irow-blockStartRow+1] += 
                               relaxWeight * dbleX[irow-blockStartRow];
               }
            }
         }
         zeroInitialGuess_ = 0;
      }

      /*-----------------------------------------------------------------
       * process each block backward
       *-----------------------------------------------------------------*/

      for ( iC = 0; iC < numColors_; iC++ )
      {
         if ( numColors_ > 1 && nprocs > 1 )
         {
            if ( ! zeroInitialGuess_)
            {
               index = 0;
               for (iP = 0; iP < nSends; iP++)
               {
                  start = hypre_ParCSRCommPkgSendMapStart(commPkg, iP);
                  for (jP=start;
                       jP<hypre_ParCSRCommPkgSendMapStart(commPkg,iP+1); jP++)
                     vBufData[index++]
                         = uData[hypre_ParCSRCommPkgSendMapElmt(commPkg,jP)];
               }
               commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                         vExtData);
               hypre_ParCSRCommHandleDestroy(commHandle);
               commHandle = NULL;
            }
         }
         if ( iC == myColor_ )
         {
            offOffset = totalOffNNZ;
            offIRow   = offNRows_;
            for ( iB = nBlocks_-1; iB >= 0; iB-- )
            {
               blkLeng = blockLengths_[iB];
               blockStartRow = iB * blockSize_ + startRow - nRecvBefore;
               blockEndRow   = blockStartRow + blkLeng - 1;

               for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
               {
                  index  = irow - startRow;
                  if ( irow >= startRow && irow <= endRow )
                  {
                     iStart = ADiagI[index];
                     iEnd   = ADiagI[index+1];
                     tmpJ   = &(ADiagJ[iStart]);
                     tmpA   = &(ADiagA[iStart]);
                     res    = fData[index];
                     for (jcol = iStart; jcol < iEnd; jcol++)
                     {
                        res -= (*tmpA) * uData[*tmpJ];
                        tmpA++; tmpJ++;
                     }
                     if ( ! zeroInitialGuess_ )
                     {
                        iStart = AOffdI[index];
                        iEnd   = AOffdI[index+1];
                        tmpJ   = &(AOffdJ[iStart]);
                        tmpA   = &(AOffdA[iStart]);
                        for (jcol = iStart; jcol < iEnd; jcol++)
                        {
                           res -= (*tmpA) * vExtData[*tmpJ];
                           tmpA++; tmpJ++;
                        }
                     }
                     dbleB[irow-blockStartRow] = res;
                  } 
                  else
                  {
                     offIRow--;
                     iStart    = 0;
                     iEnd      = offRowLengths_[offIRow];
                     offOffset -= iEnd;
                     tmpA      = &(offVals_[offOffset]);
                     tmpJ      = &(offCols_[offOffset]);
                     res       = fExtData[offIRow];
                     for (jcol = iStart; jcol < iEnd; jcol++)
                     {
                        colIndex = *tmpJ;
                        if ( colIndex >= localNRows )   
                           res -= (*tmpA) * vExtData[colIndex-localNRows];
                        else if ( colIndex >= 0 )   
                           res -= (*tmpA) * uData[colIndex];
                        tmpA++; tmpJ++;
                     }
                     dbleB[irow-blockStartRow] = res;
                  }
               }

#ifdef HAVE_ESSL
            if ( blockSize_ > switchSize )
            {
#endif
               hypre_VectorSize(sluB) = blkLeng;
               hypre_VectorSize(sluX) = blkLeng;
               strcpy( vecName, "HYPRE_Vector" );
               mliB = new MLI_Vector((void*) sluB, vecName, NULL);
               mliX = new MLI_Vector((void*) sluX, vecName, NULL);

               blockSolvers_[iB]->solve( mliB, mliX );

               delete mliB;
               delete mliX;
#ifdef HAVE_ESSL
            }
            else
            {
               for (irow = 0; irow < blkLeng; irow++) dbleX[irow] = dbleB[irow];
	       dpps(esslMatrices_[iB], blkLeng, dbleX, 1);
            }
#endif

               for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
               {
                  if ( irow < startRow )
                     vExtData[offIRow+irow-blockStartRow] += 
                               relaxWeight * dbleX[irow-blockStartRow];
                  else if ( irow <= endRow )
                     uData[irow-startRow] += relaxWeight *
                                             dbleX[irow-blockStartRow];
                  else 
                     vExtData[offIRow+irow-blockStartRow] += 
                               relaxWeight * dbleX[irow-blockStartRow];
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------
    * symmetrize
    *-----------------------------------------------------------------*/

   if (nprocs > 1 && useOverlap_)
   {
      commHandle = hypre_ParCSRCommHandleCreate(2,commPkg,vExtData,
                                                vBufData);
      hypre_ParCSRCommHandleDestroy(commHandle);
      commHandle = NULL;
      index = 0;
      for (iP = 0; iP < nSends; iP++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg, iP);
         for (jP=start;jP<hypre_ParCSRCommPkgSendMapStart(commPkg,iP+1);jP++)
         {
            iS = hypre_ParCSRCommPkgSendMapElmt(commPkg,jP); 
            uData[iS] = (uData[iS] + vBufData[index++]) * 0.5; 
            fData[iS] *= 2.0;
         }
      }
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if ( vExtData != NULL ) delete [] vExtData;
   if ( vBufData != NULL ) delete [] vBufData;
   if ( fExtData != NULL ) delete [] fExtData;
#ifdef HAVE_ESSL
   if ( blockSize_ > switchSize )
   {
#endif
      if ( sluX != NULL ) hypre_SeqVectorDestroy( sluX );
      if ( sluB != NULL ) hypre_SeqVectorDestroy( sluB );
#ifdef HAVE_ESSL
   }
#endif

   return(relaxError); 
}

/******************************************************************************
 * set parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_BSGS::setParams(char *paramString, int argc, char **argv)
{
   int    i;
   double *weights=NULL;
   char   param1[200], param2[200];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "blockSize") )
   {
      sscanf(paramString, "%s %d", param1, &blockSize_);
      if ( blockSize_ < 10 ) blockSize_ = 10;
      return 0;
   }
   else if ( !strcmp(param1, "numSweeps") )
   {
      sscanf(paramString, "%s %d", param1, &nSweeps_);
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
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
   else if ( !strcmp(param1, "setScheme") )
   {
      sscanf(paramString, "%s %s", param1, param2);
      if      ( !strcmp(param2, "multicolor") ) scheme_ = 0;
      else if ( !strcmp(param2, "parallel") )   scheme_ = 1;
      else if ( !strcmp(param2, "sequential") ) scheme_ = 2;
      return 0;
   }
   else if ( !strcmp(param1, "zeroInitialGuess") )
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
   if ( ! useOverlap_ || nprocs <= 1 ) return 0;
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
   int      iB, iP, mypid, nprocs, *partition, startRow, endRow;
   int      localNRows, nRecvs, *recvProcs, *recvStarts, nRecvBefore=0; 
   int      offRowOffset, offRowNnz, blockStartRow, blockEndRow;
   int      irow, jcol, colIndex, rowSize, *colInd, localRow, localNnz;
   int      blkLeng, *csrIA, *csrJA;
   double   *colVal, *csrAA;
   char     sName[20];
   MPI_Comm comm;
   hypre_ParCSRCommPkg *commPkg;
   hypre_ParCSRMatrix  *A = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   hypre_CSRMatrix     *seqA;
   MLI_Matrix          *mliMat;
   MLI_Function        *funcPtr;
#ifdef HAVE_ESSL
   int	    index, offset, rowIndex;
   double   *esslMatrix;
#endif

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
   free( partition );
   if ( blockSize_ == 1 ) 
   {
      nBlocks_ = localNRows;
      blockLengths_ = new int[nBlocks_];
      for ( iB = 0; iB < nBlocks_; iB++ ) blockLengths_[iB] = 1;
      maxBlkLeng_ = 1;
      return 0;
   }
   if ( nprocs > 1 && useOverlap_ )
   {
      commPkg     = hypre_ParCSRMatrixCommPkg(A);
      nRecvs      = hypre_ParCSRCommPkgNumRecvs(commPkg);
      recvProcs   = hypre_ParCSRCommPkgRecvProcs(commPkg);
      recvStarts  = hypre_ParCSRCommPkgRecvVecStarts(commPkg);
      nRecvBefore = 0;
      for ( iP = 0; iP < nRecvs; iP++ )
         if ( recvProcs[iP] > mypid ) break;
      nRecvBefore = recvStarts[iP];
   }
   else nRecvs = 0;

   /*-----------------------------------------------------------------
    * compute block information (blockLengths_) 
    *-----------------------------------------------------------------*/

   nBlocks_ = ( localNRows + blockSize_ - 1 + offNRows_ ) / blockSize_;
   if ( nBlocks_ == 0 ) nBlocks_ = 1;
   blockLengths_ = new int[nBlocks_];
   for ( iB = 0; iB < nBlocks_; iB++ ) blockLengths_[iB] = blockSize_;
   blockLengths_[nBlocks_-1] = localNRows+offNRows_-blockSize_*(nBlocks_-1);
   maxBlkLeng_ = 0;
   for ( iB = 0; iB < nBlocks_; iB++ ) 
      maxBlkLeng_ = ( blockLengths_[iB] > maxBlkLeng_ ) ? 
                       blockLengths_[iB] : maxBlkLeng_;

   /*-----------------------------------------------------------------
    * construct block matrices inverses
    *-----------------------------------------------------------------*/

#ifdef HAVE_ESSL
   if ( blockSize_ > switchSize )
   {
#endif
   strcpy( sName, "SeqSuperLU" );
   blockSolvers_ = new MLI_Solver_SeqSuperLU*[nBlocks_];
   for ( iB = 0; iB < nBlocks_; iB++ ) 
      blockSolvers_[iB] = new MLI_Solver_SeqSuperLU(sName);
   funcPtr = hypre_TAlloc(MLI_Function, 1, HYPRE_MEMORY_HOST);
#ifdef HAVE_ESSL
   }
   else
   {
      esslMatrices_ = new double*[nBlocks_];
      for ( iB = 0; iB < nBlocks_; iB++ ) esslMatrices_[iB] = NULL;
   }
#endif

   offRowOffset = offRowNnz = 0;

   for ( iB = 0; iB < nBlocks_; iB++ )
   {
      blkLeng       = blockLengths_[iB];
      blockStartRow = iB * blockSize_ + startRow - nRecvBefore;
      blockEndRow   = blockStartRow + blkLeng - 1;
      localNnz      = 0;
#ifdef HAVE_ESSL
	if ( blockSize_ > switchSize )
      {
#endif
      for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
      {
         if ( irow >= startRow && irow <= endRow )
         {
            hypre_ParCSRMatrixGetRow(A, irow, &rowSize, &colInd, &colVal);
            localNnz += rowSize;
            hypre_ParCSRMatrixRestoreRow(A, irow, &rowSize, &colInd, &colVal);
         }
         else localNnz += offRowLengths_[offRowOffset+irow-blockStartRow];
      }
      seqA = hypre_CSRMatrixCreate( blkLeng, blkLeng, localNnz );
      csrIA = new int[blkLeng+1];
      csrJA = new int[localNnz];
      csrAA = new double[localNnz];
      localRow = 0;
      localNnz = 0;
      csrIA[0] = localNnz;

      for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
      {
         if ( irow >= startRow && irow <= endRow )
         {
            hypre_ParCSRMatrixGetRow(A, irow, &rowSize, &colInd, &colVal);
            for ( jcol = 0; jcol < rowSize; jcol++ )
            {
               colIndex = colInd[jcol];
               if ((colIndex >= blockStartRow) && (colIndex <= blockEndRow))
               {
                  csrJA[localNnz] = colIndex - blockStartRow; 
                  csrAA[localNnz++] = colVal[jcol];
               }
            }
            hypre_ParCSRMatrixRestoreRow(A, irow, &rowSize, &colInd, &colVal);
         }
         else
         {
            rowSize = offRowLengths_[offRowOffset];
            colInd = &(offCols_[offRowNnz]);
            colVal = &(offVals_[offRowNnz]);
            for ( jcol = 0; jcol < rowSize; jcol++ )
            {
               colIndex = colInd[jcol];
               if ((colIndex >= blockStartRow) && (colIndex <= blockEndRow))
               {
                  csrJA[localNnz] = colIndex - blockStartRow; 
                  csrAA[localNnz++] = colVal[jcol];
               }
            }
            offRowOffset++;
            offRowNnz += rowSize;
         }
         localRow++;
         csrIA[localRow] = localNnz;
      }
      hypre_CSRMatrixI(seqA)    = csrIA;
      hypre_CSRMatrixJ(seqA)    = csrJA;
      hypre_CSRMatrixData(seqA) = csrAA;
      MLI_Utils_HypreCSRMatrixGetDestroyFunc(funcPtr);
      strcpy( sName, "HYPRE_CSR" );
      mliMat = new MLI_Matrix((void*) seqA, sName, funcPtr);
      blockSolvers_[iB]->setup( mliMat );
      delete mliMat;
#ifdef HAVE_ESSL
      }
      else
      {
         esslMatrices_[iB] = new double[blkLeng * (blkLeng+1)/2];
         esslMatrix = esslMatrices_[iB];
         bzero((char *) esslMatrices_[iB],blkLeng*(blkLeng+1)/2*sizeof(double));
         offset = 0;
         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            rowIndex = irow - blockStartRow;
            if ( irow >= startRow && irow <= endRow )
            {
               hypre_ParCSRMatrixGetRow(A, irow, &rowSize, &colInd, &colVal);
               for ( jcol = 0; jcol < rowSize; jcol++ )
               {
                  colIndex = colInd[jcol] - blockStartRow;
                  if ((colIndex >= rowIndex) && (colIndex <= blkLeng))
                  {
                     index = colIndex - rowIndex;
                     esslMatrix[offset+index] = colVal[jcol];
                  }
               }
               hypre_ParCSRMatrixRestoreRow(A,irow,&rowSize,&colInd,&colVal);
            }
            else
            {
               rowSize = offRowLengths_[offRowOffset];
               colInd = &(offCols_[offRowNnz]);
               colVal = &(offVals_[offRowNnz]);
               for ( jcol = 0; jcol < rowSize; jcol++ )
               {
                  colIndex = colInd[jcol] - blockStartRow;
                  if ((colIndex >= rowIndex) && (colIndex <= blkLeng))
                  {
                     index = colIndex - rowIndex;
                     esslMatrix[offset+index] = colVal[jcol];
                  }
               }
               offRowOffset++;
               offRowNnz += rowSize;
            }
            offset += blkLeng - irow + blockStartRow;
         }
         dppf(esslMatrix, blkLeng, 1);
      }
#endif
   }
#ifdef HAVE_ESSL
   if ( blockSize_ > switchSize )
   {
#endif
   free( funcPtr );
#ifdef HAVE_ESSL
   }
#endif
   return 0;
}

/******************************************************************************
 * adjust the off processor incoming matrix
 *--------------------------------------------------------------------------*/

int MLI_Solver_BSGS::adjustOffColIndices()
{
   int                mypid, *partition, startRow, endRow, localNRows;
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
   localNRows = endRow - startRow + 1;
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
            if ( index >= 0 ) offCols_[offset] = localNRows + index;
            else              offCols_[offset] = -1;
         }
         offset++;
      }
   }
   return 0;
}

/******************************************************************************
 * clean blocks
 *--------------------------------------------------------------------------*/

int MLI_Solver_BSGS::cleanBlocks()
{
   int iB;

   if ( blockSolvers_ != NULL ) 
   {
      for ( iB = 0; iB < nBlocks_; iB++ ) delete blockSolvers_[iB];
      delete blockSolvers_;
   }
   if ( blockLengths_  != NULL ) delete [] blockLengths_;
   if ( offRowIndices_ != NULL ) delete [] offRowIndices_;
   if ( offRowLengths_ != NULL ) delete [] offRowLengths_;
   if ( offCols_       != NULL ) delete [] offCols_;
   if ( offVals_       != NULL ) delete [] offVals_;
   nBlocks_       = 0; 
   blockLengths_  = NULL;
   blockSolvers_  = NULL;
   offNRows_      = 0;
   offRowIndices_ = NULL;
   offRowLengths_ = NULL;
   offCols_       = NULL;
   offVals_       = NULL;
   return 0;
}

/******************************************************************************
 * color processors
 *---------------------------------------------------------------------------*/

int MLI_Solver_BSGS::doProcColoring()
{
   int                 nSends, *sendProcs, mypid, nprocs, *commGraphI;
   int                 *commGraphJ, *recvCounts, i, j, *colors, *colorsAux;
   int                 pIndex, pColor;
   MPI_Comm            comm;
   hypre_ParCSRMatrix  *A;
   hypre_ParCSRCommPkg *commPkg;

   A       = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm    = hypre_ParCSRMatrixComm(A);
   commPkg = hypre_ParCSRMatrixCommPkg(A);
   if ( commPkg == NULL )
   {
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A);
      commPkg = hypre_ParCSRMatrixCommPkg(A);
   }
   nSends    = hypre_ParCSRCommPkgNumSends(commPkg);
   sendProcs = hypre_ParCSRCommPkgSendProcs(commPkg);

   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);

   commGraphI = new int[nprocs+1]; 
   recvCounts = new int[nprocs];
   MPI_Allgather(&nSends, 1, MPI_INT, recvCounts, 1, MPI_INT, comm);
   commGraphI[0] = 0;
   for ( i = 1; i <= nprocs; i++ )
      commGraphI[i] = commGraphI[i-1] + recvCounts[i-1];
   commGraphJ = new int[commGraphI[nprocs]]; 
   MPI_Allgatherv(sendProcs, nSends, MPI_INT, commGraphJ,
                  recvCounts, commGraphI, MPI_INT, comm);
   delete [] recvCounts;

#if 0
   if ( mypid == 0 )
   {
      for ( i = 0; i < nprocs; i++ )
         for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
            printf("Graph(%d,%d)\n", i, commGraphJ[j]);
   }
#endif

   colors = new int[nprocs];
   colorsAux = new int[nprocs];
   for ( i = 0; i < nprocs; i++ ) colors[i] = colorsAux[i] = -1;
   for ( i = 0; i < nprocs; i++ )
   {
      for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
      {
         pIndex = commGraphJ[j];
         pColor = colors[pIndex];
         if ( pColor >= 0 ) colorsAux[pColor] = 1;
      }
      for ( j = 0; j < nprocs; j++ ) 
         if ( colorsAux[j] < 0 ) break;
      colors[i] = j;
      for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
      {
         pIndex = commGraphJ[j];
         pColor = colors[pIndex];
         if ( pColor >= 0 ) colorsAux[pColor] = -1;
      }
   }
   delete [] colorsAux;
   myColor_ = colors[mypid];
   numColors_ = 0;
   for ( j = 0; j < nprocs; j++ ) 
      if ( colors[j]+1 > numColors_ ) numColors_ = colors[j]+1;
   delete [] colors;
   return 0;
}

