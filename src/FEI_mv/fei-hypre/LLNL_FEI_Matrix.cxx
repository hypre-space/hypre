/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




/**************************************************************************
  Module:  LLNL_FEI_Matrix.cpp
  Author:  Charles Tong
  Purpose: custom implementation of the FEI/Matrix
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "LLNL_FEI_Matrix.h"

/**************************************************************************
 Constructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_Matrix::LLNL_FEI_Matrix( MPI_Comm comm )
{
   mpiComm_ = comm;
   MPI_Comm_rank(comm, &mypid_);
   outputLevel_ = 0;

   localNRows_       = 0;
   nConstraints_     = 0;
   extNRows_         = 0;
   constrEqns_       = NULL;
   globalEqnOffsets_ = NULL;
   globalCROffsets_  = NULL;
   extColMap_        = NULL;

   /* -----------------------------------------------------------------
    * matrix and vector information
    * ----------------------------------------------------------------*/

   diagIA_     = NULL;
   diagJA_     = NULL;
   diagAA_     = NULL;
   offdIA_     = NULL;
   offdJA_     = NULL;
   offdAA_     = NULL;
   diagonal_   = NULL;

   /* ----------------------------------------------------------------*
    * communication information
    * ----------------------------------------------------------------*/

   nRecvs_          = 0;
   recvLengs_       = NULL;
   recvProcs_       = NULL;
   recvProcIndices_ = NULL;
   dRecvBufs_       = NULL;
   dExtBufs_        = NULL;

   nSends_          = 0;
   sendLengs_       = NULL;
   sendProcs_       = NULL;
   sendProcIndices_ = NULL;
   dSendBufs_       = NULL;
   mpiRequests_     = NULL;

   /* -----------------------------------------------------------------
    * others
    * ----------------------------------------------------------------*/

   FLAG_PrintMatrix_   = 0;
   FLAG_MatrixOverlap_ = 1;
}

/**************************************************************************
 destructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_Matrix::~LLNL_FEI_Matrix()
{
   double zero=0.0;
   resetMatrix(zero);
}

/**************************************************************************
 parameters function
 -------------------------------------------------------------------------*/
int LLNL_FEI_Matrix::parameters(int numParams, char **paramString)
{
   int  i;
   char param[256], param1[256];

   for ( i = 0; i < numParams; i++ )
   {
      sscanf(paramString[i],"%s", param1);
      if ( !strcmp(param1, "outputLevel") )
      {
         sscanf(paramString[i],"%s %d", param1, &outputLevel_);
         if ( outputLevel_ < 0 ) outputLevel_ = 0;
      }
      else if ( !strcmp(param1, "setDebug") )
      {
         sscanf(paramString[i],"%s %s", param1, param);
         if ( !strcmp(param, "printMatrix") ) FLAG_PrintMatrix_ = 1;
      }
      else if ( !strcmp(param1, "matrixNoOverlap") )
      {
         FLAG_MatrixOverlap_ = 0;
      }
   }
   return 0;
}

/**************************************************************************
  reset matrix function
 -------------------------------------------------------------------------*/
int LLNL_FEI_Matrix::resetMatrix(double s)
{
   (void) s;

   localNRows_   = 0;
   nConstraints_ = 0;
   extNRows_     = 0;
   if ( constrEqns_       != NULL ) delete [] constrEqns_;
   if ( globalEqnOffsets_ != NULL ) delete [] globalEqnOffsets_;
   if ( globalCROffsets_  != NULL ) delete [] globalCROffsets_;
   if ( extColMap_        != NULL ) delete [] extColMap_;

   if ( diagIA_   != NULL ) delete [] diagIA_;
   if ( diagJA_   != NULL ) delete [] diagJA_;
   if ( diagAA_   != NULL ) delete [] diagAA_;
   if ( offdIA_   != NULL ) delete [] offdIA_;
   if ( offdJA_   != NULL ) delete [] offdJA_;
   if ( offdAA_   != NULL ) delete [] offdAA_;
   if ( diagonal_ != NULL ) delete [] diagonal_;

   if ( recvLengs_       != NULL ) delete [] recvLengs_;
   if ( recvProcs_       != NULL ) delete [] recvProcs_;
   if ( recvProcIndices_ != NULL ) delete [] recvProcIndices_;
   if ( dRecvBufs_       != NULL ) delete [] dRecvBufs_;
   if ( dExtBufs_        != NULL ) delete [] dExtBufs_;
   if ( sendLengs_       != NULL ) delete [] sendLengs_;
   if ( sendProcs_       != NULL ) delete [] sendProcs_;
   if ( sendProcIndices_ != NULL ) delete [] sendProcIndices_;
   if ( dSendBufs_       != NULL ) delete [] dSendBufs_;
   if ( mpiRequests_     != NULL ) delete [] mpiRequests_; 
   localNRows_       = 0;
   nConstraints_     = 0;
   extNRows_         = 0;
   constrEqns_       = NULL;
   globalEqnOffsets_ = NULL;
   globalCROffsets_  = NULL;
   extColMap_        = NULL;
   diagIA_           = NULL;
   diagJA_           = NULL;
   diagAA_           = NULL;
   offdIA_           = NULL;
   offdJA_           = NULL;
   offdAA_           = NULL;
   diagonal_         = NULL;
   nRecvs_           = 0;
   recvLengs_        = NULL;
   recvProcs_        = NULL;
   recvProcIndices_  = NULL;
   dRecvBufs_        = NULL;
   dExtBufs_         = NULL;
   nSends_           = 0;
   sendLengs_        = NULL;
   sendProcs_        = NULL;
   sendProcIndices_  = NULL;
   dSendBufs_        = NULL;
   mpiRequests_      = NULL;
   return 0;
}

/**************************************************************************
 set element and node information
 -------------------------------------------------------------------------*/
int LLNL_FEI_Matrix::setMatrix(int nRows, int *diagIA, int *diagJA, 
                    double *diagAA, int extNRows, int *colMap, int *offdIA, 
                    int *offdJA, double *offdAA, double *diagonal,
                    int *eqnOffsets, int *crOffsets)
{
   double zero=0.0;

   resetMatrix(zero);
   localNRows_ = nRows;
   diagIA_     = diagIA;
   diagJA_     = diagJA;
   diagAA_     = diagAA;
   extNRows_   = extNRows;
   extColMap_  = colMap;
   offdIA_     = offdIA;
   offdJA_     = offdJA;
   offdAA_     = offdAA;
   diagonal_   = diagonal;
   globalEqnOffsets_ = eqnOffsets;
   globalCROffsets_  = crOffsets;
   return 0;
}

/**************************************************************************
 set communication pattern
 -------------------------------------------------------------------------*/
int LLNL_FEI_Matrix::setCommPattern(int nRecvs, int *recvLengs, int *recvProcs,
                      int *recvProcIndices, int nSends, int *sendLengs,
                      int *sendProcs, int *sendProcIndices)
{
   int iP, nSize;

   if (recvLengs_       != NULL) delete [] recvLengs_;
   if (recvProcs_       != NULL) delete [] recvProcs_;
   if (recvProcIndices_ != NULL) delete [] recvProcIndices_;
   if (dRecvBufs_       != NULL) delete [] dRecvBufs_;
   if (dExtBufs_        != NULL) delete [] dExtBufs_;
   if (sendLengs_       != NULL) delete [] sendLengs_;
   if (sendProcs_       != NULL) delete [] sendProcs_;
   if (sendProcIndices_ != NULL) delete [] sendProcIndices_;
   if (dSendBufs_       != NULL) delete [] dSendBufs_;
   if (mpiRequests_     != NULL) delete [] mpiRequests_;
   nRecvs_ = nRecvs;
   recvProcs_ = recvProcs;
   recvLengs_ = recvLengs;
   recvProcIndices_ = recvProcIndices;
   nSends_ = nSends;
   sendProcs_ = sendProcs;
   sendLengs_ = sendLengs;
   sendProcIndices_ = sendProcIndices;
   dRecvBufs_ = NULL;
   dSendBufs_ = NULL;
   dExtBufs_  = NULL;
   mpiRequests_ = NULL;
   if (nRecvs_ > 0)
   {
      nSize = 0;
      for (iP = 0; iP < nRecvs_; iP++) nSize += recvLengs_[iP];
      dRecvBufs_ = new double[nSize];
      dExtBufs_  = new double[nSize];
   }
   if (nSends_ > 0)
   {
      nSize = 0;
      for (iP = 0; iP < nSends_; iP++) nSize += sendLengs_[iP];
      dSendBufs_ = new double[nSize];
   }
   if (nRecvs_+nSends_ > 0) mpiRequests_ = new MPI_Request[nRecvs_+nSends_];
   return 0;
}

/**************************************************************************
 set complete pattern
 -------------------------------------------------------------------------*/
int LLNL_FEI_Matrix::setComplete()
{
   if ( FLAG_MatrixOverlap_ == 0 ) exchangeSubMatrices();
   if ( FLAG_PrintMatrix_   == 1 ) printMatrix();
   return 0;
}

/**************************************************************************
 set constraints 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Matrix::setConstraints(int nConstr, int *constrEqns)
{
   if (constrEqns_ != NULL) delete [] constrEqns_;
   nConstraints_ = nConstr;
   constrEqns_ = constrEqns;
   return 0;
}

/**************************************************************************
 form residual norm
 -------------------------------------------------------------------------*/
int LLNL_FEI_Matrix::residualNorm(int whichNorm, double *solnVec, 
                              double *rhsVec, double* norms)
{
   int    totalNRows, irow;
   double *rVec, rnorm, dtemp;

   (*norms) = 0.0;
   if (whichNorm < 0 || whichNorm > 2) return(-1);

   totalNRows = localNRows_ + extNRows_;
   rVec       = new double[totalNRows];
   matvec( solnVec, rVec ); 
   for ( irow = 0; irow < localNRows_; irow++ ) 
      rVec[irow] = rhsVec[irow] - rVec[irow];

   switch(whichNorm) 
   {
      case 0:
           rnorm = 0.0;
           for ( irow = 0; irow < localNRows_; irow++ ) 
           {
              dtemp = fabs( rVec[irow] );
              if ( dtemp > rnorm ) rnorm = dtemp;
           }
           MPI_Allreduce(&rnorm, &dtemp, 1, MPI_DOUBLE, MPI_MAX, mpiComm_);
           (*norms) = dtemp;
           break;
      case 1:
           rnorm = 0.0;
           for ( irow = 0; irow < localNRows_; irow++ ) 
              rnorm += fabs( rVec[irow] );
           MPI_Allreduce(&rnorm, &dtemp, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
           (*norms) = dtemp;
           break;
      case 2:
           rnorm = 0.0;
           for ( irow = 0; irow < localNRows_; irow++ ) 
              rnorm += rVec[irow] * rVec[irow];
           MPI_Allreduce(&rnorm, &dtemp, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
           (*norms) = sqrt(dtemp);
           break;
   }
   delete [] rVec;
   return 0;
}

/**************************************************************************
 matrix vector multiply
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::matvec(double *xvec, double *yvec)
{
   int    iD, iD2, matDim;
   double ddata;

   if ( FLAG_MatrixOverlap_ == 1 ) matDim = localNRows_ + extNRows_;
   else                            matDim = localNRows_;

   /* -----------------------------------------------------------------
    * exchange vector information between processors
    * -----------------------------------------------------------------*/

   scatterDData( xvec );

   /* -----------------------------------------------------------------
    * in case global stiffness matrix has been composed, use it
    * -----------------------------------------------------------------*/

   for ( iD = 0; iD < matDim; iD++ ) 
   {
      ddata = 0.0;
      for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ ) 
         ddata += diagAA_[iD2] * xvec[diagJA_[iD2]];
      yvec[iD] = ddata;
   }

   /* -----------------------------------------------------------------
    * in case global stiffness matrix has been composed, use it
    * -----------------------------------------------------------------*/

   if ( offdIA_ != NULL )
   {
      for ( iD = 0; iD < matDim; iD++ ) 
      {
         ddata = 0.0;
         for ( iD2 = offdIA_[iD]; iD2 < offdIA_[iD+1]; iD2++ ) 
           ddata += offdAA_[iD2] * dExtBufs_[offdJA_[iD2]-localNRows_];
         yvec[iD] += ddata;
      }
   }

   /* -----------------------------------------------------------------
    * exchange vector information between processors
    * -----------------------------------------------------------------*/

   if ( FLAG_MatrixOverlap_ == 1 ) gatherAddDData( yvec );
}

/**************************************************************************
 exchange extended vectors between processors
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::scatterDData( double *dvec )
{
   int        iD, iP, ind1, offset;
   MPI_Status status;

   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      MPI_Irecv( &dRecvBufs_[offset], recvLengs_[iP], MPI_DOUBLE,
                 recvProcs_[iP], 40343, mpiComm_, &mpiRequests_[iP]);
      offset += recvLengs_[iP];
   }
   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      for ( iD = 0; iD < sendLengs_[iP]; iD++ )
      {
         ind1 = sendProcIndices_[offset+iD];
         dSendBufs_[offset+iD] = dvec[ind1]; 
      }
      MPI_Send( &dSendBufs_[offset], sendLengs_[iP], MPI_DOUBLE,
                sendProcs_[iP], 40343, mpiComm_);
      offset += sendLengs_[iP];
   }
   for ( iP = 0; iP < nRecvs_; iP++ ) MPI_Wait( &mpiRequests_[iP], &status );

   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      for ( iD = 0; iD < recvLengs_[iP]; iD++ )
      {
         ind1 = recvProcIndices_[offset+iD] - localNRows_;
         dExtBufs_[ind1] = dRecvBufs_[offset+iD]; 
      }
      offset += recvLengs_[iP];
   }
}

/**************************************************************************
 exchange data between processors 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::gatherAddDData( double *dvec )
{
   int        iD, iP, ind1, offset;
   MPI_Status status;

   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      MPI_Irecv( &dSendBufs_[offset], sendLengs_[iP], MPI_DOUBLE,
                 sendProcs_[iP], 40342, mpiComm_, &mpiRequests_[iP]);
      offset += sendLengs_[iP];
   }
   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      for ( iD = 0; iD < recvLengs_[iP]; iD++ )
      {
         ind1 = recvProcIndices_[offset+iD];
         dRecvBufs_[offset+iD] = dvec[ind1]; 
      }
      MPI_Send( &dRecvBufs_[offset], recvLengs_[iP], MPI_DOUBLE,
                recvProcs_[iP], 40342, mpiComm_);
      offset += recvLengs_[iP];
   }
   for ( iP = 0; iP < nSends_; iP++ ) MPI_Wait( &mpiRequests_[iP], &status );

   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      for ( iD = 0; iD < sendLengs_[iP]; iD++ )
      {
         ind1 = sendProcIndices_[offset+iD];
         dvec[ind1] += dSendBufs_[offset+iD]; 
      }
      offset += sendLengs_[iP];
   }
}

/**************************************************************************
 print matrix and right hand side vector to a file
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::printMatrix()
{
   int  iD, iD2, offset, iEnd, totalNNZ, colInd;
   int  rowInd, matDim;
   char filename[20];
   FILE *fp;

   sprintf(filename, "mat.%d", mypid_);
   fp       = fopen(filename, "w");
   if ( FLAG_MatrixOverlap_ == 1 ) matDim = localNRows_ + extNRows_;
   else                            matDim = localNRows_;
   totalNNZ = diagIA_[matDim];
   if ( offdIA_ != NULL ) totalNNZ += offdIA_[matDim];
   fprintf(fp, "%6d  %7d \n", matDim, totalNNZ);

   offset = globalEqnOffsets_[mypid_];
   for ( iD = 0; iD < localNRows_; iD++ )
   {
      for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ )
         if ( diagJA_[iD2] == iD )
            fprintf(fp,"%6d  %6d  %25.16e \n", iD+offset+1, 
                    diagJA_[iD2]+offset+1, diagAA_[iD2]);
      for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ )
         if ( diagJA_[iD2] != iD )
            fprintf(fp,"%6d  %6d  %25.16e \n", iD+offset+1, 
                    diagJA_[iD2]+offset+1, diagAA_[iD2]);
      if ( offdIA_ != NULL )
      {
         for ( iD2 = offdIA_[iD]; iD2 < offdIA_[iD+1]; iD2++ )
         {
            colInd = extColMap_[offdJA_[iD2]-localNRows_] + 1;
            fprintf(fp,"%6d  %6d  %25.16e \n",iD+offset+1,colInd,offdAA_[iD2]);
         }
      }
   }
   if ( FLAG_MatrixOverlap_ == 1 ) 
   {
      iEnd = localNRows_ + extNRows_;
      for ( iD = localNRows_; iD < iEnd; iD++ )
      {
         for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ )
         {
            if ( diagJA_[iD2] == iD )
            {
               rowInd    = extColMap_[iD-localNRows_] + 1;
               colInd    = diagJA_[iD2] + offset + 1;
               fprintf(fp,"%6d  %6d  %25.16e \n",rowInd,colInd,diagAA_[iD2]);
            }
         }
         for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ )
         {
            if ( diagJA_[iD2] != iD )
            {
               rowInd    = extColMap_[iD-localNRows_] + 1;
               colInd    = diagJA_[iD2] + offset + 1;
               fprintf(fp,"%6d  %6d  %25.16e \n",rowInd,colInd,diagAA_[iD2]);
            }
         }
         if ( offdIA_ != NULL )
         {
            for ( iD2 = offdIA_[iD]; iD2 < offdIA_[iD+1]; iD2++ )
            {
               rowInd = extColMap_[iD-localNRows_] + 1;
               colInd = extColMap_[offdJA_[iD2]-localNRows_] + 1;
               fprintf(fp,"%6d  %6d  %25.16e \n",rowInd,colInd,offdAA_[iD2]);
            }
         }
      }
   }
   fclose(fp);
}

/**************************************************************************
 perform local matrix matrix multiplication
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::matMult( int ANRows, int ANCols, int *AIA, int *AJA, 
             double *AAA, int BNRows, int BNCols, int *BIA, int *BJA, 
             double *BAA, int *DNRows, int *DNCols, int **DIA, int **DJA, 
             double **DAA)
{
   (void) ANCols;
   (void) BNRows;
   int    CNRows, CNCols, CNnz, *CReg, ia, ib, ia2, colIndA, colIndB, iTemp; 
   int    *CIA, *CJA, offset;
   double dTempA, dTempB, *CAA;

   /* -----------------------------------------------------------------------
    * matrix matrix multiply - first compute sizes of each row in C
    * ----------------------------------------------------------------------*/

   CNRows = ANRows;
   CNCols = BNCols;
   CNnz   = 0;
   CReg   = new int[CNRows];
   for ( ib = 0; ib < CNRows; ib++ ) CReg[ib] = -1;
   for ( ia = 0; ia < ANRows; ia++ )
   {
      for ( ia2 = AIA[ia]; ia2 < AIA[ia+1]; ia2++ )
      {
         colIndA = AJA[ia2];
         for ( ib = BIA[colIndA]; ib < BIA[colIndA+1]; ib++ )
         {
            colIndB = BJA[ib];
            if ( CReg[colIndB] != ia )
            {
               CReg[colIndB] = ia;
               CNnz++;
            }
         }
      }
   }

   /* -----------------------------------------------------------------------
    * matrix matrix multiply - perform the actual multiplication
    * ----------------------------------------------------------------------*/

   CIA  = new int[CNRows+1];
   CJA  = new int[CNnz];
   CAA  = new double[CNnz];
   CNnz = 0;
   for ( ib = 0; ib < CNRows; ib++ ) CReg[ib] = -1;

   CIA[0] = 0;
   for ( ia = 0; ia < ANRows; ia++ )
   {
      iTemp = CNnz;
      for ( ia2 = AIA[ia]; ia2 < AIA[ia+1]; ia2++ )
      {
         colIndA = AJA[ia2];
         dTempA  = AAA[ia2];
         for ( ib = BIA[colIndA]; ib < BIA[colIndA+1]; ib++ )
         {
            colIndB = BJA[ib];
            dTempB  = BAA[ib];
            offset  = CReg[colIndB];
            if ( offset < iTemp )
            {
               CReg[colIndB] = CNnz;
               CJA[CNnz] = colIndB;
               CAA[CNnz++] = dTempA * dTempB;
            }
            else CAA[offset] += dTempA * dTempB;
         }
      }
      CIA[ia+1] = CNnz;
   }
   if ( CNRows > 0 ) delete [] CReg;
   (*DNRows) = CNRows;
   (*DNCols) = CNCols;
   (*DIA)    = CIA;
   (*DJA)    = CJA;
   (*DAA)    = CAA;
}

/**************************************************************************
 exchange the off-diagonal matrices between processors
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::exchangeSubMatrices()
{
   int         **sendRowLengs, **recvRowLengs, currRow, iP, iN, rowInd;
   int         *sendMatLengs, *recvMatLengs, **iSendBufs, **iRecvBufs;
   int         count, index, iD, iD2, rowStart, rowEndp1, offset;
   int         *diagRowLengs, *offdRowLengs, *TdiagIA, *TdiagJA;
   int         *ToffdIA, *ToffdJA, *iSortArray1, *iSortArray2, *iShortList;
   int         newDiagNNZ, newOffdNNZ, nRecvs, nSends, *recvLengs, *recvProcs;
   int         *sendLengs, *sendProcs, nprocs, *recvProcIndices, eqnOffset;
   int         *sendProcIndices, *procLengs, *procLengs2, totalRecvs, leng;
   double      **dSendBufs, **dRecvBufs, *TdiagAA, *ToffdAA;
   MPI_Request *requests;
   MPI_Status  status;

   MPI_Comm_size( mpiComm_, &nprocs );
   if ( nprocs == 1 ) return;
   if ( outputLevel_ > 2 )
      printf("%4d : exchangeSubMatrices begins... \n", mypid_);

   /* -----------------------------------------------------------------
    * construct sendRowLengs and recvRowLengs
    * -----------------------------------------------------------------*/

   nRecvs = nSends_;
   recvLengs = sendLengs_;
   recvProcs = sendProcs_;
   recvProcIndices = sendProcIndices_;

   nSends = nRecvs_;
   sendLengs = recvLengs_;
   sendProcs = recvProcs_;
   sendProcIndices = recvProcIndices_;

   if ( nSends > 0 ) sendRowLengs = new int*[nSends];
   if ( nRecvs > 0 ) recvRowLengs = new int*[nRecvs];
   if ( nRecvs > 0 ) requests = new MPI_Request[nRecvs];
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      recvRowLengs[iP] = new int[recvLengs[iP]];
      MPI_Irecv(recvRowLengs[iP], recvLengs[iP], MPI_INT, recvProcs[iP],
                2476, mpiComm_, &requests[iP]);
   }
   offset = 0;
   for ( iP = 0; iP < nSends; iP++ )
   {
      sendRowLengs[iP] = new int[sendLengs[iP]];
      for ( iN = 0; iN < sendLengs[iP]; iN++ )
      {
         currRow = sendProcIndices[offset+iN];
         sendRowLengs[iP][iN] = (diagIA_[currRow+1] - diagIA_[currRow]);
         if ( offdIA_ != NULL )
            sendRowLengs[iP][iN] += (offdIA_[currRow+1]-offdIA_[currRow]);
      }
      MPI_Send( sendRowLengs[iP], sendLengs[iP], MPI_INT, sendProcs[iP],
                2476, mpiComm_);
      offset += sendLengs[iP];
   }
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait( &requests[iP], &status );
   if ( nRecvs > 0 ) delete [] requests;

   /* -----------------------------------------------------------------
    * construct sendMatLengs and recvMatLengs
    * -----------------------------------------------------------------*/

   if ( nSends > 0 ) sendMatLengs = new int[nSends];
   if ( nRecvs > 0 ) recvMatLengs = new int[nRecvs];

   for ( iP = 0; iP < nSends; iP++ )
   {
      sendMatLengs[iP] = 0;
      for ( iN = 0; iN < sendLengs[iP]; iN++ )
         sendMatLengs[iP] += sendRowLengs[iP][iN];
   }
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      recvMatLengs[iP] = 0;
      for ( iN = 0; iN < recvLengs[iP]; iN++ )
         recvMatLengs[iP] += recvRowLengs[iP][iN];
   }

   /* -----------------------------------------------------------------
    * construct and fill the communication buffers for sending matrix rows
    * -----------------------------------------------------------------*/

   if ( nRecvs > 0 ) 
   {
      dRecvBufs = new double*[nRecvs];
      iRecvBufs = new int*[nRecvs];
   }
   if ( nSends > 0 ) 
   {
      dSendBufs = new double*[nSends];
      iSendBufs = new int*[nSends];
   }
   eqnOffset = globalEqnOffsets_[mypid_];
   offset = 0;
   for ( iP = 0; iP < nSends; iP++ )
   {
      iSendBufs[iP] = new int[sendMatLengs[iP]];
      dSendBufs[iP] = new double[sendMatLengs[iP]];
      count = 0;
      for ( iN = 0; iN < sendLengs[iP]; iN++ )
      {
         currRow  = sendProcIndices[offset+iN];
         for ( iD = diagIA_[currRow]; iD < diagIA_[currRow+1]; iD++ )
         {
            iSendBufs[iP][count] = diagJA_[iD] + eqnOffset;
            dSendBufs[iP][count++] = diagAA_[iD];
         }
         if ( offdIA_ != NULL )
         {
            for ( iD = offdIA_[currRow]; iD < offdIA_[currRow+1]; iD++ )
            {
               index = extColMap_[offdJA_[iD]-localNRows_]; 
               iSendBufs[iP][count] = index;
               dSendBufs[iP][count++] = offdAA_[iD];
            }
         }
      }
      offset += sendLengs[iP];
   }

   /* -----------------------------------------------------------------
    * send the matrix rows
    * -----------------------------------------------------------------*/

   if ( nRecvs > 0 ) requests = new MPI_Request[nRecvs];
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      iRecvBufs[iP] = new int[recvMatLengs[iP]];
      MPI_Irecv(iRecvBufs[iP], recvMatLengs[iP], MPI_INT, recvProcs[iP],
                2477, mpiComm_, &requests[iP]);
   }
   for ( iP = 0; iP < nSends; iP++ )
      MPI_Send( iSendBufs[iP], sendMatLengs[iP], MPI_INT, sendProcs[iP],
                2477, mpiComm_);
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait( &requests[iP], &status );

   for ( iP = 0; iP < nRecvs; iP++ )
   {
      dRecvBufs[iP] = new double[recvMatLengs[iP]];
      MPI_Irecv(dRecvBufs[iP], recvMatLengs[iP], MPI_DOUBLE, recvProcs[iP],
                2478, mpiComm_, &requests[iP]);
   }
   for ( iP = 0; iP < nSends; iP++ )
      MPI_Send( dSendBufs[iP], sendMatLengs[iP], MPI_DOUBLE, sendProcs[iP],
                2478, mpiComm_);
   for ( iP = 0; iP < nRecvs; iP++ ) MPI_Wait( &requests[iP], &status );

   if ( nRecvs > 0 ) delete [] requests;
   if ( nSends > 0 )  
   {
      for ( iP = 0; iP < nSends; iP++ ) delete [] sendRowLengs[iP];
      delete [] sendRowLengs;
      delete [] sendMatLengs;
      for ( iP = 0; iP < nSends; iP++ ) delete [] iSendBufs[iP];
      for ( iP = 0; iP < nSends; iP++ ) delete [] dSendBufs[iP];
      delete [] iSendBufs;
      delete [] dSendBufs;
   }

   /* -----------------------------------------------------------------
    * (now all information are in iRecvBufs and dRecvBufs)
    * -----------------------------------------------------------------*/

   rowStart = eqnOffset;
   rowEndp1 = rowStart + localNRows_;
   diagRowLengs = new int[localNRows_];
   for ( iD = 0; iD < localNRows_; iD++ ) 
      diagRowLengs[iD] = diagIA_[iD+1] - diagIA_[iD];
   offdRowLengs = new int[localNRows_];
   if ( offdIA_ != NULL )
   {
      for ( iD = 0; iD < localNRows_; iD++ ) 
         offdRowLengs[iD] = offdIA_[iD+1] - offdIA_[iD];
   }
   else
   {
      for ( iD = 0; iD < localNRows_; iD++ ) offdRowLengs[iD] = 0;
   }
   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ ) 
   {
      count = 0;
      for ( iN = 0; iN < sendLengs_[iP]; iN++ ) 
      {
         rowInd = sendProcIndices_[offset+iN];
         for ( iD = 0; iD < recvRowLengs[iP][iN]; iD++ ) 
         {
            index = iRecvBufs[iP][count++];
            if ( index >= rowStart && index < rowEndp1 )
                 diagRowLengs[rowInd]++;
            else offdRowLengs[rowInd]++;
         }
      }
      offset += sendLengs_[iP];
   }
   newDiagNNZ = newOffdNNZ = 0;
   for ( iD = 0; iD < localNRows_; iD++ ) newDiagNNZ += diagRowLengs[iD];
   for ( iD = 0; iD < localNRows_; iD++ ) newOffdNNZ += offdRowLengs[iD];

   /* -----------------------------------------------------------------
    * -----------------------------------------------------------------*/

   TdiagIA = new int[localNRows_+1];
   TdiagJA = new int[newDiagNNZ];
   TdiagAA = new double[newDiagNNZ];
   TdiagIA[0] = 0;
   for ( iD = 1; iD <= localNRows_; iD++ ) 
      TdiagIA[iD] = TdiagIA[iD-1] + diagRowLengs[iD-1];
   for ( iD = 0; iD < localNRows_; iD++ ) 
   {
      index = TdiagIA[iD];
      for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ ) 
      {
         TdiagJA[index] = diagJA_[iD2];
         TdiagAA[index] = diagAA_[iD2];
         index++;
      }
      TdiagIA[iD] = index;
   }
   delete [] diagIA_;
   delete [] diagJA_;
   delete [] diagAA_;
   if ( newOffdNNZ > 0 )
   {
      ToffdIA = new int[localNRows_+1];
      ToffdJA = new int[newOffdNNZ];
      ToffdAA = new double[newOffdNNZ];
      ToffdIA[0] = 0;
      for ( iD = 1; iD <= localNRows_; iD++ ) 
         ToffdIA[iD] = ToffdIA[iD-1] + offdRowLengs[iD-1];
      if ( offdIA_ != NULL )
      {
         for ( iD = 0; iD < localNRows_; iD++ ) 
         {
            index = ToffdIA[iD];
            for ( iD2 = offdIA_[iD]; iD2 < offdIA_[iD+1]; iD2++ ) 
            {
               count = extColMap_[offdJA_[iD2]-localNRows_]; 
               ToffdJA[index] = count;
               ToffdAA[index] = offdAA_[iD2];
               index++;
            }
            ToffdIA[iD] = index;
         }
         delete [] offdIA_;
         delete [] offdJA_;
         delete [] offdAA_;
         offdIA_ = NULL;
         offdJA_ = NULL;
         offdAA_ = NULL;
      }
   }
   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ ) 
   {
      count = 0;
      for ( iN = 0; iN < sendLengs_[iP]; iN++ ) 
      {
         rowInd = sendProcIndices_[offset+iN];
         for ( iD = 0; iD < recvRowLengs[iP][iN]; iD++ ) 
         {
            index = iRecvBufs[iP][count];
            if ( index >= rowStart && index < rowEndp1 )
            {
               TdiagJA[TdiagIA[rowInd]] = index - rowStart;
               TdiagAA[TdiagIA[rowInd]++] = dRecvBufs[iP][count];
            }
            else
            {
               ToffdJA[ToffdIA[rowInd]] = index;
               ToffdAA[ToffdIA[rowInd]++] = dRecvBufs[iP][count];
            }
            count++;
         }
      }
      offset += sendLengs_[iP];
   }
   if (nRecvs > 0) 
   {
      for ( iP = 0; iP < nRecvs; iP++ ) delete [] iRecvBufs[iP];
      for ( iP = 0; iP < nRecvs; iP++ ) delete [] dRecvBufs[iP];
      for ( iP = 0; iP < nRecvs; iP++ ) delete [] recvRowLengs[iP];
      delete [] iRecvBufs;
      delete [] dRecvBufs;
      delete [] recvRowLengs;
      delete [] recvMatLengs;
   }

   /* -----------------------------------------------------------------
    * (now all data in Tdiag and Toffd and diagRowLengs and offdRowLengs)
    * sort the diagonal block and construct new diag block
    * -----------------------------------------------------------------*/

   TdiagIA[0] = 0;
   for ( iD = 1; iD <= localNRows_; iD++ ) 
      TdiagIA[iD] = TdiagIA[iD-1] + diagRowLengs[iD-1];
   for ( iD = 0; iD < localNRows_; iD++ ) 
   {
      index = TdiagIA[iD];
      leng  = diagRowLengs[iD];
      IntSort2a(&(TdiagJA[index]),&(TdiagAA[index]),0,leng-1);
      count = index;
      for ( iN = index+1; iN < index+leng; iN++ ) 
      {
         if ( TdiagJA[iN] != TdiagJA[count] )
         {
            count++;
            TdiagJA[count] = TdiagJA[iN];
            TdiagAA[count] = TdiagAA[iN];
         }
         else TdiagAA[count] += TdiagAA[iN];
      }
      if ( leng > 0 ) diagRowLengs[iD] = count - index + 1;
      else            diagRowLengs[iD] = 0;
   }
   newDiagNNZ = 0;
   for ( iD = 0; iD < localNRows_; iD++ ) newDiagNNZ += diagRowLengs[iD];
   diagIA_ = new int[localNRows_+1];
   diagJA_ = new int[newDiagNNZ];
   diagAA_ = new double[newDiagNNZ];
   newDiagNNZ = 0;
   diagIA_[0] = newDiagNNZ;
   for ( iD = 0; iD < localNRows_; iD++ ) 
   {
      index = TdiagIA[iD];
      leng  = diagRowLengs[iD];
      for ( iN = index; iN < index+leng; iN++ ) 
      {
         diagJA_[newDiagNNZ] = TdiagJA[iN];
         diagAA_[newDiagNNZ++] = TdiagAA[iN];
      }
      diagIA_[iD+1] = newDiagNNZ;
   }
   delete [] TdiagIA;
   delete [] TdiagJA;
   delete [] TdiagAA;
   delete [] diagRowLengs;

   /* -----------------------------------------------------------------
    * sort the off-diagonal block
    * -----------------------------------------------------------------*/

   if ( newOffdNNZ > 0 )
   {
      ToffdIA[0] = 0;
      for ( iD = 1; iD <= localNRows_; iD++ ) 
         ToffdIA[iD] = ToffdIA[iD-1] + offdRowLengs[iD-1];
      newOffdNNZ = 0;
      for ( iD = 0; iD < localNRows_; iD++ ) 
      {
         index = ToffdIA[iD];
         leng  = offdRowLengs[iD];
         IntSort2a(&(ToffdJA[index]),&(ToffdAA[index]),0,leng-1);
         count = index;
         for ( iN = index+1; iN < index+leng; iN++ ) 
         {
            if ( ToffdJA[iN] != ToffdJA[count] )
            {
               count++;
               ToffdJA[count] = ToffdJA[iN];
               ToffdAA[count] = ToffdAA[iN];
            }
            else ToffdAA[count] += ToffdAA[iN];
         }
         if ( leng > 0 ) offdRowLengs[iD] = count - index + 1;
         else            offdRowLengs[iD] = 0;
         for ( iN = 0; iN < offdRowLengs[iD]; iN++ ) 
         {
            ToffdJA[newOffdNNZ] = ToffdJA[index+iN];
            ToffdAA[newOffdNNZ++] = ToffdAA[index+iN];
         }
      }
   }
   
   /* -----------------------------------------------------------------
    * sort the off-diagonal block to find distinct indices and construct
    * new receive information
    * -----------------------------------------------------------------*/

   nRecvs = 0;
   recvProcs = recvLengs = NULL;
   recvProcIndices = NULL;
   if ( newOffdNNZ > 0 )
   {
      /* sort all the off-diagonal indices */

      iSortArray1 = new int[newOffdNNZ];
      for ( iD = 0; iD < newOffdNNZ; iD++ ) iSortArray1[iD] = ToffdJA[iD]; 
      iSortArray2 = new int[newOffdNNZ];
      for ( iD = 0; iD < newOffdNNZ; iD++ ) iSortArray2[iD] = iD; 
      IntSort2(iSortArray1, iSortArray2, 0, newOffdNNZ-1);

      /* put the short list in iShortList and the offset in iSortArray1 */

      totalRecvs = 0;
      index = iSortArray1[0];
      for ( iD = 1; iD < newOffdNNZ; iD++ ) 
      {
         if ( iSortArray1[iD] != index ) 
         {
            totalRecvs++; 
            index = iSortArray1[iD];
         }
      }
      totalRecvs++;
      iShortList = new int[totalRecvs];
      totalRecvs = 0;
      index = iSortArray1[0];
      iShortList[0] = iSortArray1[0];
      iSortArray1[0] = totalRecvs;
      for ( iD = 1; iD < newOffdNNZ; iD++ ) 
      {
         if ( iSortArray1[iD] != index ) 
         {
            totalRecvs++; 
            index = iSortArray1[iD]; 
            iShortList[totalRecvs] = index;
         }
         iSortArray1[iD] = totalRecvs;
      }
      totalRecvs++;
      if ( extColMap_ != NULL ) delete [] extColMap_;
      extColMap_ = iShortList;
      extNRows_ = totalRecvs;

      /* convert the indices in ToffdJA */

      for ( iD = 0; iD < newOffdNNZ; iD++ ) 
         ToffdJA[iSortArray2[iD]] = iSortArray1[iD] + localNRows_;

      /* compress the Toffd matrix */

      ToffdIA[0] = 0;
      for ( iD = 1; iD <= localNRows_; iD++ ) 
         ToffdIA[iD] = ToffdIA[iD-1] + offdRowLengs[iD-1];

      offdIA_ = ToffdIA;
      offdJA_ = new int[newOffdNNZ];
      offdAA_ = new double[newOffdNNZ];
      newOffdNNZ = 0;
      for ( iD = 0; iD < localNRows_; iD++ ) 
      {
         index = ToffdIA[iD];
         leng  = offdRowLengs[iD];
         for ( iN = index; iN < index+leng; iN++ ) 
         {
            offdJA_[newOffdNNZ] = ToffdJA[iN];
            offdAA_[newOffdNNZ++] = ToffdAA[iN];
         }
      }
      delete [] ToffdJA;
      delete [] ToffdAA;

      /* construct nRecvs, recvLengs and recvProcs */

      procLengs = new int[nprocs+1];
      for ( iP = 0; iP < nprocs; iP++ ) procLengs[iP] = 0; 
      for ( iP = 0; iP <= nprocs; iP++ ) 
      {
         index = globalEqnOffsets_[iP];
         iD2 = BinarySearch2(iShortList,0,totalRecvs,index);
         if ( iD2 == -1 ) iD2 = 0;
         else if ( iD2 == -totalRecvs+1 ) iD2 = - iD2 + 1;
         else if ( iD2 < 0 ) iD2 = - iD2;
         procLengs[iP] = iD2;
      }
      nRecvs = 0;
      for ( iP = 0; iP < nprocs; iP++ ) 
         if ( procLengs[iP] != procLengs[iP+1] ) nRecvs++; 
      if ( nRecvs > 0 )
      {
         recvProcs = new int[nRecvs];
         recvLengs = new int[nRecvs];
      }
      else
      {
         recvProcs = NULL;
         recvLengs = NULL;
      }
      nRecvs = 0;
      for ( iP = 0; iP < nprocs; iP++ ) 
         if ( procLengs[iP] != procLengs[iP+1] ) 
         {
            recvLengs[nRecvs] = procLengs[iP+1] - procLengs[iP]; 
            recvProcs[nRecvs++] = iP; 
         }
      delete [] iSortArray1;
      delete [] iSortArray2;
      delete [] procLengs;
      if ( nRecvs > 0 )
      {
         count = 0;
         for ( iP = 0; iP < nRecvs; iP++ ) count += recvLengs[iP];
         recvProcIndices = new int[count];
         for ( iN = 0; iN < count; iN++ )
            recvProcIndices[iN] = iN + localNRows_;
      }
   }
   delete [] offdRowLengs;

   /* -----------------------------------------------------------------
    * diagnostics 
    * -----------------------------------------------------------------*/

#if 0
{
   char fname[20];
   sprintf(fname,"extMap.%d",mypid_);
   FILE *fp = fopen(fname, "w");
   for ( iD = 0; iD < extNRows_; iD++ ) 
      fprintf(fp,"%10d %10d\n",iD,extColMap_[iD]); 
   for ( iP = 0; iP < nRecvs; iP++ )
      fprintf(fp,"recv proc = %10d, length = %10d\n",recvProcs[iP],
              recvLengs[iP]);
   fclose(fp);
}
#endif

   /* -----------------------------------------------------------------
    * construct send information
    * -----------------------------------------------------------------*/

   procLengs = new int[nprocs];
   for ( iP = 0; iP < nprocs; iP++ ) procLengs[iP] = 0;
   for ( iP = 0; iP < nRecvs; iP++ ) procLengs[recvProcs[iP]] = 1;
   procLengs2 = new int[nprocs];
   MPI_Allreduce(procLengs,procLengs2,nprocs,MPI_INT,MPI_SUM,mpiComm_);
   nSends = procLengs2[mypid_];
   delete [] procLengs;
   delete [] procLengs2;
   sendProcs = sendLengs = NULL;
   sendProcIndices = NULL;
   if ( nSends > 0 )
   {
      sendProcs = new int[nSends];
      sendLengs = new int[nSends];
      requests  = new MPI_Request[nSends];
   }
   for ( iP = 0; iP < nSends; iP++ )
      MPI_Irecv(&(sendLengs[iP]),1,MPI_INT,MPI_ANY_SOURCE,12233,mpiComm_, 
                &requests[iP]);
   for ( iP = 0; iP < nRecvs; iP++ )
      MPI_Send(&(recvLengs[iP]),1,MPI_INT,recvProcs[iP],12233,mpiComm_);
   for ( iP = 0; iP < nSends; iP++ ) 
   {
      MPI_Wait( &requests[iP], &status );
      sendProcs[iP] = status.MPI_SOURCE;
   }
   if ( nSends > 0 ) 
   {
      count = 0;
      for ( iP = 0; iP < nSends; iP++ ) count += sendLengs[iP];
      sendProcIndices = new int[count];
   }
   count = 0;
   for ( iP = 0; iP < nSends; iP++ )
   {
      MPI_Irecv(&sendProcIndices[count],sendLengs[iP],MPI_INT,sendProcs[iP],
                12234,mpiComm_, &requests[iP]);
      count += sendLengs[iP];
   }
   index = 0;
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      iShortList = &(extColMap_[index]);
      leng = recvLengs[iP];
      index += leng;
      MPI_Send(iShortList,leng,MPI_INT,recvProcs[iP],12234,mpiComm_);
   }
   for ( iP = 0; iP < nSends; iP++ ) MPI_Wait( &requests[iP], &status );

   if ( nSends > 0 ) delete [] requests;
   nRecvs_ = nRecvs;
   if ( recvLengs_ != NULL ) delete [] recvLengs_;
   if ( recvProcs_ != NULL ) delete [] recvProcs_;
   if ( recvProcIndices_ != NULL ) delete [] recvProcIndices_;
   recvLengs_ = recvLengs;
   recvProcs_ = recvProcs;
   recvProcIndices_ = recvProcIndices;
   nSends_ = nSends;
   if ( sendLengs_ != NULL ) delete [] sendLengs_;
   if ( sendProcs_ != NULL ) delete [] sendProcs_;
   if ( sendProcIndices_ != NULL ) delete [] sendProcIndices_;
   sendLengs_ = sendLengs;
   sendProcs_ = sendProcs;
   sendProcIndices_ = sendProcIndices;
   count = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      for ( iN = 0; iN < sendLengs_[iP]; iN++ )
      {
         if ( sendProcIndices[count+iN] < eqnOffset || 
              sendProcIndices[count+iN] >= eqnOffset+localNRows_ )
            printf("%4d : exchangeSubMatrices ERROR : sendIndex %d (%d,%d).\n", 
                   mypid_, sendProcIndices[count+iN], eqnOffset, 
                   eqnOffset+localNRows_);
         else
            sendProcIndices[count+iN] -= eqnOffset;
      }
      count += sendLengs_[iP];
   }

   if ( dRecvBufs_ != NULL ) delete [] dRecvBufs_;
   if ( dExtBufs_  != NULL ) delete [] dExtBufs_;
   if ( nRecvs_ > 0 )
   {
      count = 0;
      for (iP = 0; iP < nRecvs_; iP++) count += recvLengs_[iP];
      dRecvBufs_ = new double[count];
      dExtBufs_  = new double[count];
   }
   if ( dSendBufs_ != NULL ) delete [] dSendBufs_;
   if ( nSends_ > 0 )
   {
      count = 0;
      for (iP = 0; iP < nSends_; iP++) count += sendLengs_[iP];
      dSendBufs_ = new double[count];
   }
   if ( mpiRequests_ != NULL ) delete [] mpiRequests_;
   if (nRecvs_+nSends_ > 0) mpiRequests_ = new MPI_Request[nRecvs_+nSends_];

   /* -----------------------------------------------------------------
    * diagnostics 
    * -----------------------------------------------------------------*/

#if 0
{
   char fname[20];
   sprintf(fname,"commInfo.%d",mypid_);
   FILE *fp = fopen(fname, "w");
   count = 0;
   for ( iP = 0; iP < nRecvs_; iP++ ) 
   {
      fprintf(fp,"recv from %10d = %10d\n",recvProcs_[iP],recvLengs_[iP]);
      for ( iD = 0; iD < recvLengs_[iP]; iD++ )
      {
         fprintf(fp,"recv ind %10d = %10d\n",count,recvProcIndices_[count]);
         count++;
      }
   }
   count = 0;
   for ( iP = 0; iP < nSends_; iP++ ) 
   {
      fprintf(fp,"send  to  %10d = %10d\n",sendProcs_[iP],sendLengs_[iP]);
      for ( iD = 0; iD < sendLengs_[iP]; iD++ )
      {
         fprintf(fp,"send ind %10d = %10d\n",count,sendProcIndices_[count]);
         count++;
      }
   }
   fclose(fp);
}
#endif
   if ( outputLevel_ > 2 )
      printf("%4d : exchangeSubMatrices ends. \n", mypid_);
}

/************************************************************************
 * Function  : BinarySearch2
 * Purpose   : The algorithm was taken from Numerical Recipes in C, 
 *             Second Edition.
 ************************************************************************/
int LLNL_FEI_Matrix::BinarySearch2(int *map, int start, int mapSize, int num)
{
   int k, khi, klo ;

   if (map == NULL) return -1 ;
   
   klo = start ;
   khi = start + mapSize;
   k = ((khi+klo) >> 1) + 1 ;

   while (khi-klo > 1) {
      k = (khi+klo) >> 1 ;
      if (map[k] == num) return k ;
      else if (map[k] > num) khi = k ;
      else klo = k ;
   }
   if (map[khi] == num) return khi;
   if (map[klo] == num) return klo;
   else return -(klo+1) ;
}

/**************************************************************************
 sort an integer array
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::IntSort(int *ilist, int left, int right)
{
   int i, last, mid, itemp;

   if (left >= right) return;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
      }
   }
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   IntSort(ilist, left, last-1);
   IntSort(ilist, last+1, right);
}

/**************************************************************************
 sort an integer array and an auxiliary array
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::IntSort2(int *ilist, int *ilist2, int left, int right)
{
   int i, last, mid, itemp;

   if (left >= right) return;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   itemp        = ilist2[left];
   ilist2[left] = ilist2[mid];
   ilist2[mid]  = itemp;
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
         itemp        = ilist2[last];
         ilist2[last] = ilist2[i];
         ilist2[i]    = itemp;
      }
   }
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   itemp        = ilist2[left];
   ilist2[left] = ilist2[last];
   ilist2[last] = itemp;
   IntSort2(ilist, ilist2, left, last-1);
   IntSort2(ilist, ilist2, last+1, right);
}

/**************************************************************************
 sort an integer array with an auxiliary double array
 -------------------------------------------------------------------------*/
void LLNL_FEI_Matrix::IntSort2a(int *ilist,double *dlist,int left,int right)
{
   int    mid, i, itemp, last, end2, isort, *ilist2, *ilist3;
   double dtemp, *dlist2, *dlist3;

   if (left >= right) return;
   mid         = (left + right) / 2;
   itemp       = ilist[left];
   ilist[left] = ilist[mid];
   ilist[mid]  = itemp;
   dtemp       = dlist[left];
   dlist[left] = dlist[mid];
   dlist[mid]  = dtemp;
   last        = left;
   isort       = ilist[left];
   ilist2      = &(ilist[last]);
   dlist2      = &(dlist[last]);
   ilist3      = &(ilist[left+1]);
   dlist3      = &(dlist[left+1]);
   end2        = right + 1;
   for (i = left+1; i < end2; i++)
   {
      if ( *ilist3 < isort )
      {
         last++;
         ilist2++; dlist2++;
         itemp   = *ilist2;
         *ilist2 = *ilist3;
         *ilist3 = itemp;
         dtemp   = *dlist2;
         *dlist2 = *dlist3;
         *dlist3 = dtemp;
      }
      ilist3++; dlist3++;
   }
   itemp       = ilist[left];
   ilist[left] = ilist[last];
   ilist[last] = itemp;
   dtemp       = dlist[left];
   dlist[left] = dlist[last];
   dlist[last] = dtemp;
   IntSort2a(ilist, dlist, left, last-1);
   IntSort2a(ilist, dlist, last+1, right);
}

