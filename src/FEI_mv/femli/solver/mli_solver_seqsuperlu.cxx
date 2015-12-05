/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.15 $
 ***********************************************************************EHEADER*/

/* **************************************************************************** 
 * -- SuperLU routine (version 1.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center, 
 * and Lawrence Berkeley National Lab.
 * ************************************************************************* */

#ifdef MLI_SUPERLU

#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "mli_solver_seqsuperlu.h"
#include "HYPRE.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "IJ_mv/_hypre_IJ_mv.h"

/* ****************************************************************************
 * constructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_SeqSuperLU::MLI_Solver_SeqSuperLU(char *name) : MLI_Solver(name)
{
   permRs_       = NULL;
   permCs_       = NULL;
   mliAmat_      = NULL;
   factorized_   = 0;
   localNRows_   = 0;
   nSubProblems_ = 1;
   subProblemRowSizes_   = NULL;
   subProblemRowIndices_ = NULL;
   numColors_ = 1;
   myColors_  = new int[numColors_];
   myColors_[0] = 0;

   // for domain decomposition with coarse overlaps
   nSends_ = 0;
   nRecvs_ = 0;
   sendProcs_ = NULL;
   recvProcs_ = NULL;
   sendLengs_ = NULL;
   recvLengs_ = NULL;
   PSmat_ = NULL;
   AComm_ = 0;
   PSvec_ = NULL;
}

/* ****************************************************************************
 * destructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_SeqSuperLU::~MLI_Solver_SeqSuperLU()
{
   int iP;

   for ( iP = 0; iP < nSubProblems_; iP++ )
   {
      if ( permRs_[iP] != NULL ) 
      {
         Destroy_SuperNode_Matrix(&(superLU_Lmats[iP]));
         Destroy_CompCol_Matrix(&(superLU_Umats[iP]));
      }
   }
   if ( permRs_ != NULL ) 
   {
      for ( iP = 0; iP < nSubProblems_; iP++ )
         if ( permRs_[iP] != NULL ) delete [] permRs_[iP];
      delete [] permRs_;
   }
   if ( permCs_ != NULL ) 
   {
      for ( iP = 0; iP < nSubProblems_; iP++ )
         if ( permCs_[iP] != NULL ) delete [] permCs_[iP];
      delete [] permCs_;
   }
   if ( subProblemRowSizes_ != NULL ) delete [] subProblemRowSizes_;
   if ( subProblemRowIndices_ != NULL ) 
   {
      for (iP = 0; iP < nSubProblems_; iP++)
         if (subProblemRowIndices_[iP] != NULL) 
            delete [] subProblemRowIndices_[iP];
      delete [] subProblemRowIndices_;
   }
   if ( myColors_  != NULL ) delete [] myColors_;
   if ( sendProcs_ != NULL ) delete [] sendProcs_;
   if ( recvProcs_ != NULL ) delete [] recvProcs_;
   if ( sendLengs_ != NULL ) delete [] sendLengs_;
   if ( recvLengs_ != NULL ) delete [] recvLengs_;
   if ( PSmat_     != NULL ) delete PSmat_;
   if ( PSvec_     != NULL ) delete PSvec_;
}

/* ****************************************************************************
 * setup 
 * --------------------------------------------------------------------------*/

int MLI_Solver_SeqSuperLU::setup( MLI_Matrix *Amat )
{
   int      nrows, iP, startRow, nnz, *csrIA, *csrJA, *cscJA, *cscIA;
   int      irow, icol, *rowArray, *countArray, colNum, index, nSubRows;
   int      *etree, permcSpec, lwork, panelSize, relax, info, rowCnt;
   double   *csrAA, *cscAA, diagPivotThresh, dropTol;
   hypre_ParCSRMatrix  *hypreA;
   hypre_CSRMatrix     *ADiag;
   SuperMatrix         AC, superLU_Amat;
   superlu_options_t   slu_options;
   SuperLUStat_t       slu_stat;

   /* ---------------------------------------------------------------
    * fetch matrix
    * -------------------------------------------------------------*/

   if ( nSubProblems_ > 100 )
   {
      printf("MLI_Solver_SeqSuperLU::setup ERROR - over 100 subproblems.\n");
      exit(1);
   }
   mliAmat_ = Amat;
   if ( !strcmp( mliAmat_->getName(), "HYPRE_ParCSR" ) )
   {
      hypreA = (hypre_ParCSRMatrix *) mliAmat_->getMatrix();
      ADiag = hypre_ParCSRMatrixDiag(hypreA);
   }
   else if ( !strcmp( mliAmat_->getName(), "HYPRE_CSR" ) )
   {
      ADiag = (hypre_CSRMatrix *) mliAmat_->getMatrix();
   }
   else
   {
      printf("MLI_Solver_SeqSuperLU::setup ERROR - invalid format(%s).\n",
             mliAmat_->getName());
      exit(1);
   }

   /* ---------------------------------------------------------------
    * fetch matrix
    * -------------------------------------------------------------*/
 
   csrAA = hypre_CSRMatrixData(ADiag);
   csrIA = hypre_CSRMatrixI(ADiag);
   csrJA = hypre_CSRMatrixJ(ADiag);
   nrows = hypre_CSRMatrixNumRows(ADiag);
   nnz   = hypre_CSRMatrixNumNonzeros(ADiag);
   startRow = 0;
   localNRows_ = nrows;

   /* ---------------------------------------------------------------
    * set up coloring and then take out overlap subdomains 
    * -------------------------------------------------------------*/

   if ( numColors_ > 1 ) setupBlockColoring();
#if 0
   if ( nSubProblems_ > 0 )
   {
      int *domainNumArray = new int[nrows];
      for (iP = nSubProblems_-1; iP >= 0; iP--)
      {
         for (irow = 0; irow < subProblemRowSizes_[iP]; irow++)
            domainNumArray[subProblemRowIndices_[iP][irow]] = iP;
         delete [] subProblemRowIndices_[iP];
      }
      delete [] subProblemRowSizes_;
      delete [] subProblemRowIndices_;
      subProblemRowSizes_ = new int[nSubProblems_];
      subProblemRowIndices_ = new int*[nSubProblems_];
      for (iP = 0; iP < nSubProblems_; iP++) subProblemRowSizes_[iP] = 0;
      for (irow = 0; irow < nrows; irow++)
         subProblemRowSizes_[domainNumArray[irow]]++;
      for (iP = 0; iP < nSubProblems_; iP++) 
         subProblemRowIndices_[iP] = new int[subProblemRowSizes_[iP]];
      for (iP = 0; iP < nSubProblems_; iP++) subProblemRowSizes_[iP] = 0;
      for (irow = 0; irow < nrows; irow++)
      {
         index = domainNumArray[irow];
         subProblemRowIndices_[index][subProblemRowSizes_[index]++] = irow;
      } 
      delete [] domainNumArray;
   } 
#endif

   /* ---------------------------------------------------------------
    * allocate space
    * -------------------------------------------------------------*/
 
   permRs_ = new int*[nSubProblems_];
   permCs_ = new int*[nSubProblems_];

   /* ---------------------------------------------------------------
    * conversion from CSR to CSC 
    * -------------------------------------------------------------*/

   countArray = new int[nrows];
   if ( subProblemRowIndices_ != NULL ) rowArray = new int[nrows];

#if 0
FILE *fp = fopen("matfile","w");
for (irow = 0; irow < nrows; irow++)
for (icol = csrIA[irow]; icol < csrIA[irow+1]; icol++)
fprintf(fp,"%8d %8d %25.16e\n",irow,csrJA[icol],csrAA[icol]);
fclose(fp);
#endif

   for ( iP = 0; iP < nSubProblems_; iP++ )
   {
      if ( subProblemRowIndices_ != NULL )
      {
         nSubRows = subProblemRowSizes_[iP]; 
         for (irow = 0; irow < nrows; irow++) rowArray[irow] = -1;
         rowCnt = 0;
         for (irow = 0; irow < nSubRows; irow++) 
         {
            index = subProblemRowIndices_[iP][irow] - startRow;
            if (index >= 0 && index < nrows) rowArray[index] = rowCnt++;
         }
         for ( irow = 0; irow < nSubRows; irow++ ) countArray[irow] = 0;
         rowCnt = 0;
         for ( irow = 0; irow < nrows; irow++ ) 
         {
            if ( rowArray[irow] >= 0 )
            {
               for ( icol = csrIA[irow]; icol < csrIA[irow+1]; icol++ ) 
               {
                  index = csrJA[icol];
                  if (rowArray[index] >= 0) countArray[rowArray[index]]++;
               }
            }
         }
         nnz = 0;
         for ( irow = 0; irow < nSubRows; irow++ ) nnz += countArray[irow];
         cscJA = (int *)    malloc( (nSubRows+1) * sizeof(int) );
         cscIA = (int *)    malloc( nnz * sizeof(int) );
         cscAA = (double *) malloc( nnz * sizeof(double) );
         cscJA[0] = 0;
         nnz = 0;
         for ( icol = 1; icol <= nSubRows; icol++ ) 
         {
            nnz += countArray[icol-1]; 
            cscJA[icol] = nnz;
         }
         for ( irow = 0; irow < nrows; irow++ )
         {
            if ( rowArray[irow] >= 0 )
            {
               for ( icol = csrIA[irow]; icol < csrIA[irow+1]; icol++ ) 
               {
                  colNum = rowArray[csrJA[icol]];
                  if ( colNum >= 0) 
                  {
                     index  = cscJA[colNum]++;
                     cscIA[index] = rowArray[irow];
                     cscAA[index] = csrAA[icol];
                  }
               }
            }
         }
         cscJA[0] = 0;
         nnz = 0;
         for ( icol = 1; icol <= nSubRows; icol++ ) 
         {
            nnz += countArray[icol-1]; 
            cscJA[icol] = nnz;
         }
         dCreate_CompCol_Matrix(&superLU_Amat, nSubRows, nSubRows, 
                  cscJA[nSubRows], cscAA, cscIA, cscJA, SLU_NC, SLU_D, SLU_GE);
         etree   = new int[nSubRows];
         permCs_[iP]  = new int[nSubRows];
         permRs_[iP]  = new int[nSubRows];
         permcSpec = 0;
         get_perm_c(permcSpec, &superLU_Amat, permCs_[iP]);
         slu_options.Fact = DOFACT;
         slu_options.SymmetricMode = NO;
         sp_preorder(&slu_options, &superLU_Amat, permCs_[iP], etree, &AC);
         diagPivotThresh = 1.0;
         dropTol = 0.0;
         panelSize = sp_ienv(1);
         relax = sp_ienv(2);
         StatInit(&slu_stat);
         lwork = 0;
         dgstrf(&slu_options, &AC, dropTol, relax, panelSize,
                etree,NULL,lwork,permCs_[iP],permRs_[iP],
                &(superLU_Lmats[iP]),&(superLU_Umats[iP]),&slu_stat,&info);
         Destroy_CompCol_Permuted(&AC);
         Destroy_CompCol_Matrix(&superLU_Amat);
         delete [] etree;
         StatFree(&slu_stat);
      }
      else
      {
         for ( irow = 0; irow < nrows; irow++ ) countArray[irow] = 0;
         for ( irow = 0; irow < nrows; irow++ ) 
         {
            for ( icol = csrIA[irow]; icol < csrIA[irow+1]; icol++ ) 
            {
               if (csrJA[icol] < 0 || csrJA[icol] >= nrows)
                  printf("SeqSuperLU ERROR : colNum = %d\n", colNum);
               countArray[csrJA[icol]]++;
            }
         }
         cscJA = (int *)    malloc( (nrows+1) * sizeof(int) );
         cscAA = (double *) malloc( nnz * sizeof(double) );
         cscIA = (int *)    malloc( nnz * sizeof(int) );
         cscJA[0] = 0;
         nnz = 0;
         for ( icol = 1; icol <= nrows; icol++ ) 
         {
            nnz += countArray[icol-1]; 
            cscJA[icol] = nnz;
         }
         for ( irow = 0; irow < nrows; irow++ )
         {
            for ( icol = csrIA[irow]; icol < csrIA[irow+1]; icol++ ) 
            {
               colNum = csrJA[icol];
               if (colNum < 0 || colNum >= nrows)
               {
                  printf("ERROR : irow, icol, colNum = %d, %d, %d\n", irow, 
                      icol, colNum);
                  exit(1);
               }
               index  = cscJA[colNum]++;
               if (index < 0 || index >= nnz)
                  printf("ERROR : index = %d %d %d\n", index, colNum, irow);
               cscIA[index] = irow;
               cscAA[index] = csrAA[icol];
            }
         }
         cscJA[0] = 0;
         nnz = 0;
         for ( icol = 1; icol <= nrows; icol++ ) 
         {
            nnz += countArray[icol-1]; 
            cscJA[icol] = nnz;
         }
         dCreate_CompCol_Matrix(&superLU_Amat, nrows, nrows, cscJA[nrows], 
                                cscAA, cscIA, cscJA, SLU_NC, SLU_D, SLU_GE);
         etree = new int[nrows];
         permCs_[iP]  = new int[nrows];
         permRs_[iP]  = new int[nrows];
         permcSpec = 0;
         get_perm_c(permcSpec, &superLU_Amat, permCs_[iP]);
         slu_options.Fact = DOFACT;
         slu_options.SymmetricMode = NO;
         sp_preorder(&slu_options, &superLU_Amat, permCs_[iP], etree, &AC);
         diagPivotThresh = 1.0;
         dropTol = 0.0;
         panelSize = sp_ienv(1);
         relax = sp_ienv(2);
         StatInit(&slu_stat);
         lwork = 0;
         dgstrf(&slu_options, &AC, dropTol, relax, panelSize,
                etree,NULL,lwork,permRs_[iP],permCs_[iP],&(superLU_Lmats[iP]),
                &(superLU_Umats[iP]),&slu_stat,&info);
         Destroy_CompCol_Permuted(&AC);
         Destroy_CompCol_Matrix(&superLU_Amat);
         delete [] etree;
         StatFree(&slu_stat);
      }
   }
   factorized_ = 1;
   delete [] countArray;
   if ( subProblemRowIndices_ != NULL ) delete [] rowArray;
   return 0;
}

/* ****************************************************************************
 * This subroutine calls the SuperLU subroutine to perform LU 
 * backward substitution 
 * --------------------------------------------------------------------------*/

int MLI_Solver_SeqSuperLU::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int    iP, iC, irow, nrows, info, nSubRows, extNCols, nprocs, endp1;
   int    jP, jcol, index, nSends, start, rowInd, *AOffdI, *AOffdJ;
   int    *ADiagI, *ADiagJ, rlength, offset, length;
   double *uData, *fData, *subUData, *sBuffer, *rBuffer, res, *AOffdA;
   double *ADiagA, *f2Data, *u2Data, one=1.0, zero=0.0;
   MPI_Comm    comm;
   SuperMatrix B;
   trans_t     trans;
   SuperLUStat_t   slu_stat;
   hypre_ParVector *f, *u, *f2;
   hypre_CSRMatrix *ADiag, *AOffd;
   hypre_ParCSRMatrix  *A, *P;
   hypre_ParCSRCommPkg *commPkg;
   hypre_ParCSRCommHandle *commHandle;
   MPI_Request *mpiRequests;
   MPI_Status  mpiStatus;

   /* -------------------------------------------------------------
    * check that the factorization has been called
    * -----------------------------------------------------------*/

   if ( ! factorized_ )
   {
      printf("MLI_Solver_SeqSuperLU::Solve ERROR - not factorized yet.\n");
      exit(1);
   }

   /* -------------------------------------------------------------
    * fetch matrix and vector parameters
    * -----------------------------------------------------------*/

   A       = (hypre_ParCSRMatrix *) mliAmat_->getMatrix();
   comm    = hypre_ParCSRMatrixComm(A);
   commPkg = hypre_ParCSRMatrixCommPkg(A);
   if ( commPkg == NULL )
   {
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A);
      commPkg = hypre_ParCSRMatrixCommPkg(A);
   }
   MPI_Comm_size(comm, &nprocs);
   ADiag    = hypre_ParCSRMatrixDiag(A);
   ADiagI   = hypre_CSRMatrixI(ADiag);
   ADiagJ   = hypre_CSRMatrixJ(ADiag);
   ADiagA   = hypre_CSRMatrixData(ADiag);
   AOffd    = hypre_ParCSRMatrixOffd(A);
   AOffdI   = hypre_CSRMatrixI(AOffd);
   AOffdJ   = hypre_CSRMatrixJ(AOffd);
   AOffdA   = hypre_CSRMatrixData(AOffd);
   extNCols = hypre_CSRMatrixNumCols(AOffd);

   nrows  = localNRows_;
   u      = (hypre_ParVector *) uIn->getVector();
   uData  = hypre_VectorData(hypre_ParVectorLocalVector(u));
   f      = (hypre_ParVector *) fIn->getVector();
   fData  = hypre_VectorData(hypre_ParVectorLocalVector(f));

   /* -------------------------------------------------------------
    * allocate communication buffers if overlap but not DD
    * -----------------------------------------------------------*/

   rBuffer = sBuffer = NULL;
   if ( PSmat_ != NULL )
   { 
      rlength = 0;
      for ( iP = 0; iP < nRecvs_; iP++ ) rlength += recvLengs_[iP]; 
      f2      = (hypre_ParVector *) PSvec_->getVector();
      f2Data  = hypre_VectorData(hypre_ParVectorLocalVector(f2));
      u2Data  = new double[localNRows_];
      if ( nRecvs_ > 0 ) mpiRequests = new MPI_Request[nRecvs_];
   }
   else
   {
      if (nprocs > 1)
      {
         nSends = hypre_ParCSRCommPkgNumSends(commPkg);
         if ( nSends > 0 )
         {
            length = hypre_ParCSRCommPkgSendMapStart(commPkg,nSends);
            sBuffer = new double[length];
         }
         else sBuffer = NULL;
         if ( extNCols > 0 ) rBuffer = new double[extNCols];
      }
   }

   /* -------------------------------------------------------------
    * collect global vector and create a SuperLU dense matrix
    * solve the problem
    * clean up 
    * -----------------------------------------------------------*/

   if ( nSubProblems_ == 1 )
   {
      if ( PSmat_ != NULL )
      {
         P = (hypre_ParCSRMatrix *) PSmat_->getMatrix();
         hypre_ParCSRMatrixMatvecT(one, P, f, zero, f2); 
         offset = nrows - rlength;
         for ( iP = 0; iP < nRecvs_; iP++ )
         {
            MPI_Irecv(&u2Data[offset],recvLengs_[iP],MPI_DOUBLE,
                      recvProcs_[iP], 45716, AComm_, &(mpiRequests[iP]));
            offset += recvLengs_[iP];
         }
         length = nrows - rlength;
         for ( iP = 0; iP < nSends_; iP++ )
            MPI_Send(f2Data,sendLengs_[iP],MPI_DOUBLE,sendProcs_[iP],45716,
                     AComm_);
         for ( iP = 0; iP < nRecvs_; iP++ )
            MPI_Wait( &(mpiRequests[iP]), &mpiStatus );
         if ( nRecvs_ > 0 ) delete [] mpiRequests;
         length = nrows - rlength;
         for ( irow = 0; irow < length; irow++ ) u2Data[irow] = fData[irow];
         dCreate_Dense_Matrix(&B, nrows, 1, u2Data, nrows, SLU_DN, SLU_D, SLU_GE);
         StatInit(&slu_stat);
         trans = NOTRANS;
         dgstrs (trans, &(superLU_Lmats[0]), &(superLU_Umats[0]), permCs_[0], 
                 permRs_[0], &B, &slu_stat, &info);
         Destroy_SuperMatrix_Store(&B);
         for ( irow = 0; irow < length; irow++ ) uData[irow] = u2Data[irow];
         //delete [] u2Data;
         StatFree(&slu_stat);
         return info;
      }
      else
      {
         for ( irow = 0; irow < nrows; irow++ ) uData[irow] = fData[irow];
         dCreate_Dense_Matrix(&B, nrows, 1, uData, nrows, SLU_DN, SLU_D, SLU_GE);
         trans = NOTRANS;
         StatInit(&slu_stat);
         dgstrs (trans, &(superLU_Lmats[0]), &(superLU_Umats[0]), permCs_[0], 
                 permRs_[0], &B, &slu_stat, &info);
         Destroy_SuperMatrix_Store(&B);
         StatFree(&slu_stat);
         return info;
      }
   }

   /* -------------------------------------------------------------
    * if more than 1 subProblems
    * -----------------------------------------------------------*/

   subUData = new double[nrows];
   for ( iC = 0; iC < numColors_; iC++ )
   {
      if (nprocs > 1 && iC > 0)
      {
         index = 0;
         for (iP = 0; iP < nSends; iP++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(commPkg, iP);
            endp1 = hypre_ParCSRCommPkgSendMapStart(commPkg, iP+1);
            for (jP = start; jP < endp1; jP++)
               sBuffer[index++]
                      = uData[hypre_ParCSRCommPkgSendMapElmt(commPkg,jP)];
         }
         commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,sBuffer,rBuffer);
         hypre_ParCSRCommHandleDestroy(commHandle);
         commHandle = NULL;
      }
      for ( iP = 0; iP < nSubProblems_; iP++ ) 
      {
         if ( iC == myColors_[iP] )
         {
            for (irow = 0; irow < subProblemRowSizes_[iP]; irow++)
            {
               rowInd = subProblemRowIndices_[iP][irow];
               res    = fData[rowInd];
               for (jcol = ADiagI[rowInd]; jcol < ADiagI[rowInd+1]; jcol++)
                  res -= ADiagA[jcol] * uData[ADiagJ[jcol]];
               for (jcol = AOffdI[rowInd]; jcol < AOffdI[rowInd+1]; jcol++)
                  res -= AOffdA[jcol] * rBuffer[AOffdJ[jcol]];
               subUData[irow] = res;
            }
            nSubRows = subProblemRowSizes_[iP];
            dCreate_Dense_Matrix(&B,nSubRows,1,subUData,nSubRows,SLU_DN,SLU_D,SLU_GE);
            trans = NOTRANS;
            dgstrs(trans,&(superLU_Lmats[iP]),&(superLU_Umats[iP]),
                   permCs_[iP],permRs_[iP],&B,&slu_stat,&info);
            Destroy_SuperMatrix_Store(&B);
            for ( irow = 0; irow < nSubRows; irow++ ) 
               uData[subProblemRowIndices_[iP][irow]] += subUData[irow];
         }
      }
   }
   if (sBuffer != NULL) delete [] sBuffer;
   if (rBuffer != NULL) delete [] rBuffer;
   return info;
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_SeqSuperLU::setParams(char *paramString, int argc, char **argv)
{
   int    i, j, *iArray, **iArray2;
   char   param1[100];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "setSubProblems") )
   {
      if ( argc != 3 ) 
      {
         printf("MLI_Solver_SeqSuperLU::setParams ERROR : needs 3 arg.\n");
         return 1;
      }
      if (subProblemRowSizes_ != NULL) delete [] subProblemRowSizes_;
      subProblemRowSizes_ = NULL; 
      if (subProblemRowIndices_ != NULL) 
      {
         for (i = 0; i < nSubProblems_; i++) 
            if (subProblemRowIndices_[i] != NULL)
               delete [] subProblemRowIndices_[i];
         subProblemRowIndices_ = NULL; 
      }
      nSubProblems_ = *(int *) argv[0];
      if (nSubProblems_ <= 0) nSubProblems_ = 1;
      if (nSubProblems_ > 1)
      {
         iArray = (int *) argv[1];
         subProblemRowSizes_ = new int[nSubProblems_];; 
         for (i = 0; i < nSubProblems_; i++) subProblemRowSizes_[i] = iArray[i];
         iArray2 = (int **) argv[2];
         subProblemRowIndices_ = new int*[nSubProblems_];; 
         for (i = 0; i < nSubProblems_; i++) 
         {
            subProblemRowIndices_[i] = new int[subProblemRowSizes_[i]]; 
            for (j = 0; j < subProblemRowSizes_[i]; j++) 
               subProblemRowIndices_[i][j] = iArray2[i][j];
         }
      }
   }
   else if ( !strcmp(param1, "setPmat") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_SeqSuperLU::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      HYPRE_IJVector auxVec;
      PSmat_ = (MLI_Matrix *) argv[0];
      hypre_ParCSRMatrix *hypreAux;
      hypre_ParCSRMatrix *ps = (hypre_ParCSRMatrix *) PSmat_->getMatrix();
      int nCols = hypre_ParCSRMatrixNumCols(ps);
      int start = hypre_ParCSRMatrixFirstColDiag(ps);
      MPI_Comm vComm = hypre_ParCSRMatrixComm(ps);
      HYPRE_IJVectorCreate(vComm, start, start+nCols-1, &auxVec);
      HYPRE_IJVectorSetObjectType(auxVec, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(auxVec);
      HYPRE_IJVectorAssemble(auxVec);
      HYPRE_IJVectorGetObject(auxVec, (void **) &hypreAux);
      HYPRE_IJVectorSetObjectType(auxVec, -1);
      HYPRE_IJVectorDestroy(auxVec);
      strcpy( paramString, "HYPRE_ParVector" );
      MLI_Function *funcPtr = new MLI_Function();
      MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
      PSvec_ = new MLI_Vector( (void*) hypreAux, paramString, funcPtr );
      delete funcPtr;
   }
   else if ( !strcmp(param1, "setCommData") )
   {
      if ( argc != 7 ) 
      {
         printf("MLI_Solver_SeqSuperLU::setParams ERROR : needs 7 arg.\n");
         return 1;
      }
      nRecvs_ = *(int *) argv[0];
      if ( nRecvs_ > 0 ) 
      {
         recvProcs_ = new int[nRecvs_];
         recvLengs_ = new int[nRecvs_];
         iArray =  (int *) argv[1];
         for ( i = 0; i < nRecvs_; i++ ) recvProcs_[i] = iArray[i];
         iArray =  (int *) argv[2];
         for ( i = 0; i < nRecvs_; i++ ) recvLengs_[i] = iArray[i];
      }
      nSends_ = *(int *) argv[3];
      if ( nSends_ > 0 ) 
      {
         sendProcs_ = new int[nSends_];
         sendLengs_ = new int[nSends_];
         iArray =  (int *) argv[4];
         for ( i = 0; i < nSends_; i++ ) sendProcs_[i] = iArray[i];
         iArray =  (int *) argv[5];
         for ( i = 0; i < nSends_; i++ ) sendLengs_[i] = iArray[i];
      }
      AComm_ = *(MPI_Comm *) argv[6];
   }
   else
   {   
      printf("MLI_Solver_SeqSuperLU::setParams - parameter not recognized.\n");
      printf("                 Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * multicoloring 
 *---------------------------------------------------------------------------*/

int MLI_Solver_SeqSuperLU::setupBlockColoring()
{
   int                 i, j, k, nSends, mypid, nprocs, myRowOffset, nEntries;
   int                 *procNRows, gNRows, *globalGI, *globalGJ; 
   int                 *localGI, *localGJ, *offsets, globalOffset, gRowCnt; 
   int                 searchIndex, searchStatus;
   MPI_Comm            comm;
   hypre_ParCSRMatrix     *A;
   hypre_ParCSRCommPkg    *commPkg;
   hypre_ParCSRCommHandle *commHandle;
   hypre_CSRMatrix        *AOffd;

   /*---------------------------------------------------------------*/
   /* fetch matrix                                                  */
   /*---------------------------------------------------------------*/

   A       = (hypre_ParCSRMatrix *) mliAmat_->getMatrix();
   comm    = hypre_ParCSRMatrixComm(A);
   commPkg = hypre_ParCSRMatrixCommPkg(A);
   if ( commPkg == NULL )
   {
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A);
      commPkg = hypre_ParCSRMatrixCommPkg(A);
   }
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);

   /*---------------------------------------------------------------*/
   /* construct local graph ==> (nSubProblems_, GDiagI, GDiagJ)     */
   /*---------------------------------------------------------------*/

   int *sortIndices;
   int *graphMatrix = new int[nSubProblems_*nSubProblems_];
   for (i = 0; i < nSubProblems_*nSubProblems_; i++) graphMatrix[i] = 0;
   for (i = 0; i < nSubProblems_; i++) 
   {
      for (j = i+1; j < nSubProblems_; j++) 
      {
         nEntries = subProblemRowSizes_[i] + subProblemRowSizes_[j];
         sortIndices = new int[nEntries];
         nEntries = subProblemRowSizes_[i];
         for (k = 0; k < subProblemRowSizes_[i]; k++) 
            sortIndices[k] = subProblemRowIndices_[i][k];
         for (k = 0; k < subProblemRowSizes_[j]; k++) 
            sortIndices[nEntries+k] = subProblemRowIndices_[j][k];
         nEntries += subProblemRowSizes_[j];
         MLI_Utils_IntQSort2(sortIndices,NULL,0,nEntries-1);
         for (k = 1; k < nEntries; k++) 
         {
            if (sortIndices[k] == sortIndices[k-1]) 
            {
               graphMatrix[i*nSubProblems_+j] = 1;
               graphMatrix[j*nSubProblems_+i] = 1;
               break;
            }
         }
         delete [] sortIndices;
      }
   }
   nEntries = 0;
   for (i = 0; i < nSubProblems_*nSubProblems_; i++) 
      if (graphMatrix[i] != 0) nEntries++;
   int *GDiagI = new int[nSubProblems_+1];
   int *GDiagJ = new int[nEntries];
   nEntries = 0;
   GDiagI[0] = nEntries;
   for (i = 0; i < nSubProblems_; i++) 
   {
      for (j = 0; j < nSubProblems_; j++) 
         if (graphMatrix[i*nSubProblems_+j] == 1) GDiagJ[nEntries++] = j;
      GDiagI[i+1] = nEntries;
   }
   delete [] graphMatrix;

   /*---------------------------------------------------------------*/
   /* compute processor number of rows and my row offset            */
   /* (myRowOffset, proNRows)                                       */
   /*---------------------------------------------------------------*/

   procNRows = new int[nprocs];
   MPI_Allgather(&nSubProblems_,1,MPI_INT,procNRows,1,MPI_INT,comm);
   gNRows = 0;
   for (i = 0; i < nprocs; i++) gNRows += procNRows[i];
   myRowOffset = 0;
   for (i = 0; i < mypid; i++) myRowOffset += procNRows[i];
   for (i = 0; i < GDiagI[nSubProblems_]; i++) GDiagJ[i] += myRowOffset;

   /*---------------------------------------------------------------*/
   /* construct local off-diagonal graph                            */
   /*---------------------------------------------------------------*/

   int    extNCols, mapStart, mapEnd, mapIndex, *AOffdI, *AOffdJ;
   int    localNRows;
   double *sBuffer=NULL, *rBuffer=NULL;

   localNRows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   AOffd    = hypre_ParCSRMatrixOffd(A);
   AOffdI   = hypre_CSRMatrixI(AOffd);
   AOffdJ   = hypre_CSRMatrixJ(AOffd);
   extNCols = hypre_CSRMatrixNumCols(AOffd);
   nSends   = hypre_ParCSRCommPkgNumSends(commPkg);
   if (extNCols > 0) rBuffer = new double[extNCols];
   if (nSends > 0)
      sBuffer = new double[hypre_ParCSRCommPkgSendMapStart(commPkg,nSends)];
   mapIndex  = 0;
   for (i = 0; i < nSends; i++)
   {
      mapStart = hypre_ParCSRCommPkgSendMapStart(commPkg, i);
      mapEnd   = hypre_ParCSRCommPkgSendMapStart(commPkg, i+1);
      for (j=mapStart; j<mapEnd; j++)
      {
         searchIndex = hypre_ParCSRCommPkgSendMapElmt(commPkg,j);
         for (k = 0; k < nSubProblems_; k++)
         {
            searchStatus = MLI_Utils_BinarySearch(searchIndex,
                             subProblemRowIndices_[k],subProblemRowSizes_[k]);
            if (searchStatus >= 0)
            {
               sBuffer[mapIndex++] = (double) (k + myRowOffset);
               break;
            }
         }
      }
   }
   if ( nSends > 0 || extNCols > 0 )
   {
      commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,sBuffer,rBuffer);
      hypre_ParCSRCommHandleDestroy(commHandle);
      commHandle = NULL;
   }
   if ( extNCols > 0 )
   {
      int indexI, indexJ;
      int *GOffdCnt = new int[nSubProblems_];
      int *GOffdJ = new int[nSubProblems_*extNCols];
      for (i = 0; i < nSubProblems_; i++) GOffdCnt[i] = 0;
      for (i = 0; i < nSubProblems_*extNCols; i++) GOffdJ[i] = -1;
      for (i = 0; i < localNRows; i++)
      {
         if ( AOffdI[i+1] > AOffdI[i] )
         {
            for (k = 0; k < nSubProblems_; k++)
            {
               indexI = MLI_Utils_BinarySearch(i,subProblemRowIndices_[k],
                                               subProblemRowSizes_[k]);
               if (indexI >= 0) break;
            }
            for (j = AOffdI[i]; j < AOffdJ[i+1]; j++)
            {
               indexJ = (int) rBuffer[i];
               GOffdJ[extNCols*k+AOffdJ[j]] = indexJ;
            }
         }
      }
      int totalNNZ = GDiagI[nSubProblems_];
      for (i = 0; i < nSubProblems_; i++) totalNNZ += GOffdCnt[i];
      localGI = new int[nSubProblems_+1];
      localGJ = new int[totalNNZ];
      totalNNZ = 0;
      localGI[0] = totalNNZ;
      for (i = 0; i < nSubProblems_; i++) 
      {
         for (j = GDiagI[i]; j < GDiagI[i+1]; j++) 
            localGJ[totalNNZ++] = GDiagJ[j];
         for (j = 0; j < extNCols; j++) 
            if (GOffdJ[i*extNCols+j] >= 0) 
               localGJ[totalNNZ++] = GOffdJ[i*extNCols+j];
         localGI[i+1] = totalNNZ;
      } 
      delete [] GDiagI;
      delete [] GDiagJ;
      delete [] GOffdCnt;
      delete [] GOffdJ;
   }
   else
   {
      localGI = GDiagI;
      localGJ = GDiagJ;
   }
   if (sBuffer != NULL) delete [] sBuffer;
   if (rBuffer != NULL) delete [] rBuffer;
   
   /*---------------------------------------------------------------*/
   /* form global graph (gNRows, globalGI, globalGJ)                */
   /*---------------------------------------------------------------*/

   globalGI = new int[gNRows+1];
   offsets  = new int[nprocs+1];
   offsets[0] = 0;
   for (i = 1; i <= nprocs; i++)
      offsets[i] = offsets[i-1] + procNRows[i-1];
   MPI_Allgatherv(&localGI[1], nSubProblems_, MPI_INT, &globalGI[1],
                  procNRows, offsets, MPI_INT, comm);
   delete [] offsets;
   globalOffset = 0; 
   gRowCnt = 1;
   globalGI[0] = globalOffset;
   for (i = 0; i < nprocs; i++)
   {
      for (j = 0; j < procNRows[i]; j++)
      {
         globalGI[gRowCnt] = globalOffset + globalGI[gRowCnt]; 
         gRowCnt++;
      }
      globalOffset += globalGI[gRowCnt-1];
   }
   globalGJ = new int[globalOffset];
   int *recvCnts = new int[nprocs+1];
   globalOffset = 0;
   for (i = 0; i < nprocs; i++)
   {
      gRowCnt = globalOffset;
      globalOffset = globalGI[gRowCnt+procNRows[i]];
      recvCnts[i] = globalOffset - gRowCnt;
   }
   offsets = new int[nprocs+1];
   offsets[0] = 0;
   for (i = 1; i <= nprocs; i++)
      offsets[i] = offsets[i-1] + recvCnts[i-1];
   nEntries = localGI[nSubProblems_];
   MPI_Allgatherv(localGJ, nEntries, MPI_INT, globalGJ, recvCnts, 
                  offsets, MPI_INT, comm);
   delete [] offsets;
   delete [] recvCnts;
   delete [] localGI;
   delete [] localGJ;

#if 0
   if ( mypid == 0 )
   {
      for ( i = 0; i < gNRows; i++ )
         for ( j = globalGI[i]; j < globalGI[i+1]; j++ )
            printf("Graph(%d,%d)\n", i, globalGJ[j]);
   }
#endif

   /*---------------------------------------------------------------*/
   /* start coloring                                                */
   /*---------------------------------------------------------------*/

   int *colors = new int[gNRows];
   int *colorsAux = new int[gNRows];
   int gIndex, gColor;

   for ( i = 0; i < gNRows; i++ ) colors[i] = colorsAux[i] = -1;
   for ( i = 0; i < gNRows; i++ )
   {
      for ( j = globalGI[i]; j < globalGI[i+1]; j++ )
      {
         gIndex = globalGJ[j];
         gColor = colors[gIndex];
         if ( gColor >= 0 ) colorsAux[gColor] = 1;
      }
      for ( j = 0; j < gNRows; j++ ) 
         if ( colorsAux[j] < 0 ) break;
      colors[i] = j;
      for ( j = globalGI[i]; j < globalGI[i+1]; j++ )
      {
         gIndex = globalGJ[j];
         gColor = colors[gIndex];
         if ( gColor >= 0 ) colorsAux[gColor] = -1;
      }
   }
   delete [] colorsAux;
   myColors_ = new int[nSubProblems_];
   for ( j = myRowOffset; j < myRowOffset+nSubProblems_; j++ ) 
      myColors_[j-myRowOffset] = colors[j]; 
   numColors_ = 0;
   for ( j = 0; j < gNRows; j++ ) 
      if ( colors[j]+1 > numColors_ ) numColors_ = colors[j]+1;
   delete [] colors;
   if ( mypid == 0 )
      printf("\tMLI_Solver_SeqSuperLU : number of colors = %d\n",numColors_);
   return 0;
}

#endif

