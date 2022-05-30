/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include <stdio.h>
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "mli_matrix.h"
#include "mli_utils.h"

extern "C" {
   void hypre_qsort0(int *, int, int);
}

/***************************************************************************
 * compute triple matrix product function 
 *--------------------------------------------------------------------------*/

int MLI_Matrix_ComputePtAP(MLI_Matrix *Pmat, MLI_Matrix *Amat, 
                           MLI_Matrix **RAPmat_out)
{
   int          ierr;
   char         paramString[200];
   void         *Pmat2, *Amat2, *RAPmat2;
   MLI_Matrix   *RAPmat;
   MLI_Function *funcPtr;

   if ( strcmp(Pmat->getName(),"HYPRE_ParCSR") || 
        strcmp(Amat->getName(),"HYPRE_ParCSR") )
   {
      printf("MLI_Matrix_computePtAP ERROR - matrix has invalid type.\n");
      exit(1);
   }
   Pmat2 = (void *) Pmat->getMatrix();
   Amat2 = (void *) Amat->getMatrix();
   ierr = MLI_Utils_HypreMatrixComputeRAP(Pmat2,Amat2,&RAPmat2);
   if ( ierr ) printf("ERROR in MLI_Matrix_ComputePtAP\n");
   sprintf(paramString, "HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   RAPmat = new MLI_Matrix(RAPmat2,paramString,funcPtr);
   delete funcPtr;
   (*RAPmat_out) = RAPmat;
   return 0;
}

/***************************************************************************
 * compute triple matrix product function 
 *--------------------------------------------------------------------------*/

int MLI_Matrix_FormJacobi(MLI_Matrix *Amat, double alpha, MLI_Matrix **Jmat)
{
   int          ierr;
   char         paramString[200];
   void         *A, *J;
   MLI_Function *funcPtr;
   
   if ( strcmp(Amat->getName(),"HYPRE_ParCSR") ) 
   {
      printf("MLI_Matrix_FormJacobi ERROR - matrix has invalid type.\n");
      exit(1);
   }
   A = (void *) Amat->getMatrix();;
   ierr = MLI_Utils_HypreMatrixFormJacobi(A, alpha, &J);
   if ( ierr ) printf("ERROR in MLI_Matrix_FormJacobi\n");
   sprintf(paramString, "HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   (*Jmat) = new MLI_Matrix(J,paramString,funcPtr);
   delete funcPtr;
   return ierr;
}

/***************************************************************************
 * compress matrix by block size > 1
 *--------------------------------------------------------------------------*/

int MLI_Matrix_Compress(MLI_Matrix *Amat, int blksize, MLI_Matrix **Amat2)
{
   int          ierr;
   char         paramString[200];
   void         *A, *A2;
   MLI_Function *funcPtr;
   
   if ( strcmp(Amat->getName(),"HYPRE_ParCSR") ) 
   {
      printf("MLI_Matrix_Compress ERROR - matrix has invalid type.\n");
      exit(1);
   }
   if ( blksize <= 1 )
   {
      printf("MLI_Matrix_Compress WARNING - blksize <= 1.\n");
      (*Amat2) = NULL;
      return 1;
   }
   A = (void *) Amat->getMatrix();;
   ierr = MLI_Utils_HypreMatrixCompress(A, blksize, &A2);
   if ( ierr ) printf("ERROR in MLI_Matrix_Compress\n");
   sprintf(paramString, "HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   (*Amat2) = new MLI_Matrix(A2,paramString,funcPtr);
   delete funcPtr;
   return ierr;
}

/***************************************************************************
 * get submatrix given row indices
 *--------------------------------------------------------------------------*/

int MLI_Matrix_GetSubMatrix(MLI_Matrix *A_in, int nRows, int *rowIndices,
                            int *newNRows, double **newAA)
{
   int        mypid, nprocs, *partition, startRow, endRow;
   int        i, j, myNRows, irow, rowInd, rowLeng, *cols, *myRowIndices;
   double     *AA, *vals;
   hypre_ParCSRMatrix *A;
   MPI_Comm           comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters (off_offset)
    *-----------------------------------------------------------------*/

   A = (hypre_ParCSRMatrix *) A_in;
   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm, &mypid);  
   MPI_Comm_size(comm, &nprocs);  
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   startRow = partition[mypid];
   endRow   = partition[mypid+1] - 1;
   free( partition );

   myNRows = 0;
   for ( irow = 0; irow < nRows; irow++ )
   {
      rowInd = rowIndices[irow];
      if ( rowInd >= startRow && rowInd < endRow )
      {
         hypre_ParCSRMatrixGetRow(A,rowInd,&rowLeng,&cols,NULL);
         myNRows += rowLeng;
         hypre_ParCSRMatrixRestoreRow(A,rowInd,&rowLeng,&cols,NULL);
      }
   }

   myRowIndices = new int[myNRows]; 
   myNRows = 0;
   for ( irow = 0; irow < nRows; irow++ )
   {
      rowInd = rowIndices[irow];
      if ( rowInd >= startRow && rowInd < endRow )
      {
         hypre_ParCSRMatrixGetRow(A,rowInd,&rowLeng,&cols,NULL);
         for ( i = 0; i < rowLeng; i++ )
            myRowIndices[myNRows++] = cols[i];
         hypre_ParCSRMatrixRestoreRow(A,rowInd,&rowLeng,&cols,NULL);
      }
   }

   hypre_qsort0(myRowIndices, 0, myNRows-1);
   j = 1;
   for ( i = 1; i < myNRows; i++ )
      if ( myRowIndices[i] != myRowIndices[j-1] ) 
         myRowIndices[j++] = myRowIndices[i]; 
   myNRows = j;

   AA = new double[myNRows*myNRows];
   for ( irow = 0; irow < myNRows*myNRows; irow++ ) AA[i] = 0.0;

   for ( irow = 0; irow < myNRows; irow++ )
   {
      rowInd = myRowIndices[irow];
      if ( rowInd >= startRow && rowInd < endRow )
      {
         hypre_ParCSRMatrixGetRow(A,rowInd,&rowLeng,&cols,&vals);
         for ( i = 0; i < rowLeng; i++ )
            AA[(cols[i]-startRow)*myNRows+irow] = vals[i]; 
         hypre_ParCSRMatrixRestoreRow(A,rowInd,&rowLeng,&cols,&vals);
      }
   }

   (*newAA) = AA;
   (*newNRows) = myNRows;
   return 0;
}

/***************************************************************************
 * get submatrix given row indices
 *--------------------------------------------------------------------------*/

int MLI_Matrix_GetOverlappedMatrix(MLI_Matrix *mli_mat, int *offNRows, 
                 int **offRowLengths, int **offCols, double **offVals)
{
   int         i, j, k, mypid, nprocs, *partition, startRow;
   int         nSends, *sendProcs, nRecvs;
   int         *recvProcs, *recvStarts, proc, offset, length, reqNum; 
   int         totalSendNnz, totalRecvNnz, index, base, totalSends;
   int         totalRecvs, rowNum, rowLength, *colInd, *sendStarts;
   int         limit, *isendBuf, *cols, curNnz, *rowIndices; 
   double      *dsendBuf, *vals, *colVal;
   hypre_ParCSRMatrix  *A;
   MPI_Comm            comm;
   MPI_Request         *requests;
   MPI_Status          *status;
   hypre_ParCSRCommPkg *commPkg;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters (off_offset)
    *-----------------------------------------------------------------*/

   A    = (hypre_ParCSRMatrix *) mli_mat->getMatrix();
   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  
   if ( nprocs == 1 )
   {
      (*offNRows) = 0;
      (*offRowLengths) = NULL;
      (*offCols) = NULL;
      (*offVals) = NULL;
      return 0;
   }
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   startRow   = partition[mypid];
   hypre_TFree( partition , HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * fetch matrix communication information (off_nrows)
    *-----------------------------------------------------------------*/

   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A);
   commPkg    = hypre_ParCSRMatrixCommPkg(A);
   nSends     = hypre_ParCSRCommPkgNumSends(commPkg);
   sendProcs  = hypre_ParCSRCommPkgSendProcs(commPkg);
   sendStarts = hypre_ParCSRCommPkgSendMapStarts(commPkg);
   nRecvs     = hypre_ParCSRCommPkgNumRecvs(commPkg);
   recvProcs  = hypre_ParCSRCommPkgRecvProcs(commPkg);
   recvStarts = hypre_ParCSRCommPkgRecvVecStarts(commPkg);
   requests = hypre_CTAlloc( MPI_Request, nRecvs+nSends , HYPRE_MEMORY_HOST);
   totalSends  = sendStarts[nSends];
   totalRecvs  = recvStarts[nRecvs];
   (*offNRows) = totalRecvs;

   /*-----------------------------------------------------------------
    * construct offRowLengths 
    *-----------------------------------------------------------------*/

   if ( totalRecvs > 0 ) (*offRowLengths) = new int[totalRecvs];
   else                  (*offRowLengths) = NULL;
   reqNum = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      MPI_Irecv(&((*offRowLengths)[offset]),length,MPI_INT,proc,13278,comm, 
                &requests[reqNum++]);
   }
   if ( totalSends > 0 ) isendBuf = hypre_CTAlloc( int, totalSends , HYPRE_MEMORY_HOST);
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
         hypre_ParCSRMatrixGetRow(A,rowNum,&rowLength,&colInd,NULL);
         isendBuf[index++] = rowLength;
         totalSendNnz += rowLength;
         hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowLength,&colInd,NULL);
      }
      MPI_Isend(&isendBuf[offset], length, MPI_INT, proc, 13278, comm, 
                &requests[reqNum++]);
   }
   status = hypre_CTAlloc(MPI_Status, reqNum, HYPRE_MEMORY_HOST);
   MPI_Waitall( reqNum, requests, status );
   hypre_TFree( status , HYPRE_MEMORY_HOST);
   if ( totalSends > 0 ) hypre_TFree( isendBuf , HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * construct row indices 
    *-----------------------------------------------------------------*/

   if ( totalRecvs > 0 ) rowIndices = new int[totalRecvs];
   else                  rowIndices = NULL;
   reqNum = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      MPI_Irecv(&(rowIndices[offset]), length, MPI_INT, proc, 13279, comm, 
                &requests[reqNum++]);
   }
   if ( totalSends > 0 ) isendBuf = hypre_CTAlloc( int, totalSends , HYPRE_MEMORY_HOST);
   index = 0;
   for (i = 0; i < nSends; i++)
   {
      proc   = sendProcs[i];
      offset = sendStarts[i];
      limit  = sendStarts[i+1];
      length = limit - offset;
      for (j = offset; j < limit; j++)
      {
         rowNum = hypre_ParCSRCommPkgSendMapElmt(commPkg,j) + startRow;
         isendBuf[index++] = rowNum;
      }
      MPI_Isend(&isendBuf[offset], length, MPI_INT, proc, 13279, comm, 
                &requests[reqNum++]);
   }
   status = hypre_CTAlloc(MPI_Status, reqNum, HYPRE_MEMORY_HOST);
   MPI_Waitall( reqNum, requests, status );
   hypre_TFree( status , HYPRE_MEMORY_HOST);
   if ( totalSends > 0 ) hypre_TFree( isendBuf , HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * construct offCols 
    *-----------------------------------------------------------------*/

   totalRecvNnz = 0;
   for (i = 0; i < totalRecvs; i++) totalRecvNnz += (*offRowLengths)[i];
   if ( totalRecvNnz > 0 )
   {
      cols = new int[totalRecvNnz];
      vals = new double[totalRecvNnz];
   }
   reqNum = totalRecvNnz = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc    = recvProcs[i];
      offset  = recvStarts[i];
      length  = recvStarts[i+1] - offset;
      curNnz = 0;
      for (j = 0; j < length; j++) curNnz += (*offRowLengths)[offset+j];
      MPI_Irecv(&cols[totalRecvNnz], curNnz, MPI_INT, proc, 13280, comm, 
                &requests[reqNum++]);
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) isendBuf = hypre_CTAlloc( int, totalSendNnz , HYPRE_MEMORY_HOST);
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
         hypre_ParCSRMatrixGetRow(A,rowNum,&rowLength,&colInd,NULL);
         for (k = 0; k < rowLength; k++) 
            isendBuf[totalSendNnz++] = colInd[k];
         hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowLength,&colInd,NULL);
      }
      length = totalSendNnz - base;
      MPI_Isend(&isendBuf[base], length, MPI_INT, proc, 13280, comm, 
                &requests[reqNum++]);
   }
   status = hypre_CTAlloc(MPI_Status, reqNum, HYPRE_MEMORY_HOST);
   MPI_Waitall( reqNum, requests, status );
   hypre_TFree( status , HYPRE_MEMORY_HOST);
   if ( totalSendNnz > 0 ) hypre_TFree( isendBuf , HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * construct offVals 
    *-----------------------------------------------------------------*/

   reqNum = totalRecvNnz = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      curNnz = 0;
      for (j = 0; j < length; j++) curNnz += (*offRowLengths)[offset+j];
      MPI_Irecv(&vals[totalRecvNnz], curNnz, MPI_DOUBLE, proc, 13281, comm, 
                &requests[reqNum++]);
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) dsendBuf = hypre_CTAlloc( double, totalSendNnz , HYPRE_MEMORY_HOST);
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
         hypre_ParCSRMatrixGetRow(A,rowNum,&rowLength,NULL,&colVal);
         for (k = 0; k < rowLength; k++) 
            dsendBuf[totalSendNnz++] = colVal[k];
         hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowLength,NULL,&colVal);
      }
      length = totalSendNnz - base;
      MPI_Isend(&dsendBuf[base], length, MPI_DOUBLE, proc, 13281, comm, 
                &requests[reqNum++]);
   }
   status = hypre_CTAlloc(MPI_Status, reqNum, HYPRE_MEMORY_HOST);
   MPI_Waitall( reqNum, requests, status );
   hypre_TFree( status , HYPRE_MEMORY_HOST);
   if ( totalSendNnz > 0 ) hypre_TFree( dsendBuf , HYPRE_MEMORY_HOST);

   if ( nSends+nRecvs > 0 ) hypre_TFree( requests , HYPRE_MEMORY_HOST);

   (*offCols) = cols;
   (*offVals) = vals;
   return 0;
}

/***************************************************************************
 * perform matrix transpose (modified from parcsr_mv function by putting
 * diagonal entries at the beginning of the row)
 *--------------------------------------------------------------------------*/

void MLI_Matrix_Transpose(MLI_Matrix *Amat, MLI_Matrix **AmatT)
{
   int                one=1, ia, ia2, ib, iTemp, *ATDiagI, *ATDiagJ;
   int                localNRows;
   double             dTemp, *ATDiagA;
   char               paramString[30];
   hypre_CSRMatrix    *ATDiag;
   hypre_ParCSRMatrix *hypreA, *hypreAT;
   MLI_Matrix         *mli_AmatT;
   MLI_Function       *funcPtr;

   hypreA = (hypre_ParCSRMatrix *) Amat->getMatrix();
   hypre_ParCSRMatrixTranspose( hypreA, &hypreAT, one );
   ATDiag = hypre_ParCSRMatrixDiag(hypreAT);
   localNRows = hypre_CSRMatrixNumRows(ATDiag);
   ATDiagI = hypre_CSRMatrixI(ATDiag);
   ATDiagJ = hypre_CSRMatrixJ(ATDiag);
   ATDiagA = hypre_CSRMatrixData(ATDiag);

   /* -----------------------------------------------------------------------
    * move the diagonal entry to the beginning of the row
    * ----------------------------------------------------------------------*/

   for ( ia = 0; ia < localNRows; ia++ ) 
   {
      iTemp = -1;
      for ( ia2 = ATDiagI[ia]; ia2 < ATDiagI[ia+1]; ia2++ ) 
      {
         if ( ATDiagJ[ia2] == ia ) 
         {
            iTemp = ATDiagJ[ia2];
            dTemp = ATDiagA[ia2];
            break;
         }
      }
      if ( iTemp >= 0 )
      {
         for ( ib = ia2; ib > ATDiagI[ia]; ib-- ) 
         {
            ATDiagJ[ib] = ATDiagJ[ib-1];
            ATDiagA[ib] = ATDiagA[ib-1];
         }
         ATDiagJ[ATDiagI[ia]] = iTemp;
         ATDiagA[ATDiagI[ia]] = dTemp;
      }  
   }  

   /* -----------------------------------------------------------------------
    * construct MLI_Matrix
    * ----------------------------------------------------------------------*/

   sprintf( paramString, "HYPRE_ParCSRMatrix" );
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr); 
   mli_AmatT = new MLI_Matrix((void*) hypreAT, paramString, funcPtr);
   delete funcPtr;

   *AmatT = mli_AmatT;
}
