/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

// *********************************************************************
// This file is customized to use HYPRE matrix format
// *********************************************************************

// *********************************************************************
// system includes
// ---------------------------------------------------------------------

#include <string.h>

#define MABS(x) ((x) > 0 ? (x) : (-(x)))

// *********************************************************************
// HYPRE includes external to MLI
// ---------------------------------------------------------------------

#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"

#define mabs(x) ((x) > 0 ? x : -(x))

// *********************************************************************
// local defines
// ---------------------------------------------------------------------

#define MLI_METHOD_AMGSA_READY       -1
#define MLI_METHOD_AMGSA_SELECTED    -2
#define MLI_METHOD_AMGSA_PENDING     -3
#define MLI_METHOD_AMGSA_NOTSELECTED -4
#define MLI_METHOD_AMGSA_SELECTED2   -5

// *********************************************************************
// local MLI includes
// ---------------------------------------------------------------------

#include "mli_method_amgsa.h"
#include "mli_utils.h"
#include "mli_sfei.h"
#include "mli_fedata_utils.h"
#include "mli_matrix.h"
#include "mli_matrix_misc.h"
#include "mli_solver.h"

// *********************************************************************
// functions external to MLI
// ---------------------------------------------------------------------

extern "C"
{
   /* ARPACK function to compute eigenvalues/eigenvectors */

   void dnstev_(int *n, int *nev, char *which, double *sigmar,
                double *sigmai, int *colptr, int *rowind, double *nzvals,
                double *dr, double *di, double *z, int *ldz, int *info,
                double *tol);
}

/***********************************************************************
 * COMPUTE SUBDOMAIN-BASED NULL SPACES USING EIGENDECOMPOSITION
 ***********************************************************************
 * compute initial null spaces (for the subdomain only) using FEData
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupSFEIBasedNullSpaces(MLI *mli)
{
   int          k, iN, iD, iR, level, mypid, nElems, elemNNodes;
   int          iE, iN2, **elemNodeLists, *elemNodeList1D, totalNNodes;
   int          *partition, localStartRow, localNRows, *newElemNodeList;
   int          eMatDim, newNNodes, *elemNodeList, count, *orderArray;
   int          csrNrows, *csrIA, *csrJA, offset, rowSize, startCol;
   int          rowInd, colInd, colOffset, rowLeng, start, nSubdomains;
   double       **elemMatrices, *elemMat, *csrAA, dAccum;
   double       *eigenR, *eigenI, *eigenV;
   char         which[20], filename[100];;
   FILE         *fp;
   MPI_Comm     comm;
   MLI_SFEI     *sfei;
   MLI_Matrix   *mliAmat;
   hypre_ParCSRMatrix *hypreA;
#ifdef MLI_ARPACK
   int          info;
   double       sigmaR, sigmaI;
#endif

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces begins.\n");
#endif

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if (mli == NULL)
   {
      printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces ERROR");
      printf(" - no mli.\n");
      exit(1);
   }
   level = 0;
   sfei = mli->getSFEI(level);
   if (sfei == NULL)
   {
      printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces ERROR");
      printf(" - no sfei.\n");
      exit(1);
   }
   nSubdomains = sfei->getNumElemBlocks();
   if (nSubdomains <= 0) return 0;

   /* --------------------------------------------------------------- */
   /* fetch communicator matrix information                           */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank(comm, &mypid);
   mliAmat = mli->getSystemMatrix(level);
   hypreA  = (hypre_ParCSRMatrix *) mliAmat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,
                                        &partition);
   localStartRow = partition[mypid];
   localNRows    = partition[mypid+1] - localStartRow;
   free(partition);

   /* --------------------------------------------------------------- */
   /* fetch communicator matrix information                           */
   /* --------------------------------------------------------------- */

   startCol = 0;
   if (nullspaceVec_ != NULL)
   {
      for (k = 0; k < nullspaceDim_; k++)
      {
         dAccum = 0.0;
         for (iR = 0; iR < nullspaceLen_; iR++)
            dAccum += mabs(nullspaceVec_[iR+k*nullspaceLen_]);
         if (dAccum == 0.0) {startCol = k; break;}
      }
      if (k == nullspaceDim_) startCol = nullspaceDim_;
      if ( startCol == nullspaceDim_ ) return 0;
   }

   /* --------------------------------------------------------------- */
   /* initialize null space vector and aggregation label              */
   /* --------------------------------------------------------------- */

   //if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   if (nullspaceVec_ != NULL) hypre_assert( nullspaceLen_ == localNRows );
   if (nullspaceVec_ == NULL)
   {
      nullspaceLen_ = localNRows;
      nullspaceVec_ = new double[localNRows*nullspaceDim_];
   }
   if (saLabels_ == NULL)
   {
      saLabels_ = new int*[maxLevels_];
      for (k = 0; k < maxLevels_; k++) saLabels_[k] = NULL;
   }
   if (saLabels_[0] != NULL) delete [] saLabels_[0];
   saLabels_[0] = new int[localNRows];
   for (k = 0; k < localNRows; k++) saLabels_[0][k] = -1;

   /* --------------------------------------------------------------- */
   /* fetch SFEI information (nElems,elemIDs,elemNNodes,elemNodeLists)*/
   /* --------------------------------------------------------------- */

   if ((printToFile_ & 8) != 0 && nSubdomains == 1)
   {
      sprintf(filename,"elemNodeList.%d", mypid);
      fp = fopen(filename,"w");
   }
   else fp = NULL;
   for (iD = 0; iD < nSubdomains; iD++)
   {
      nElems = sfei->getBlockNumElems(iD);
      if (fp != NULL) fprintf(fp, "%d\n", nElems);
      elemNNodes  = sfei->getBlockElemNEqns(iD);
      elemNodeLists = sfei->getBlockElemEqnLists(iD);
      elemMatrices  = sfei->getBlockElemStiffness(iD);
      totalNNodes = nElems * elemNNodes;
      elemNodeList1D = new int[totalNNodes];
      count = 0;
      for (iE = 0; iE < nElems; iE++)
         for (iN = 0; iN < elemNNodes; iN++)
            elemNodeList1D[count++] = elemNodeLists[iE][iN];

      /* ------------------------------------------------------ */
      /* find the number of nodes in local subdomain (including */
      /* external nodes)                                        */
      /* ------------------------------------------------------ */

      orderArray = new int[totalNNodes];
      newElemNodeList = new int[totalNNodes];
      for ( iN = 0; iN < totalNNodes; iN++ )
      {
         orderArray[iN] = iN;
         newElemNodeList[iN] = elemNodeList1D[iN];
      }
      MLI_Utils_IntQSort2(newElemNodeList,orderArray,0,totalNNodes-1);
      elemNodeList1D[orderArray[0]] = 0;
      newNNodes = 0;
      for (iN = 1; iN < totalNNodes; iN++)
      {
         if (newElemNodeList[iN] == newElemNodeList[newNNodes])
            elemNodeList1D[orderArray[iN]] = newNNodes;
         else
         {
            newNNodes++;
            elemNodeList1D[orderArray[iN]] = newNNodes;
            newElemNodeList[newNNodes] = newElemNodeList[iN];
         }
      }
      if (totalNNodes > 0) newNNodes++;
      delete [] orderArray;
      delete [] newElemNodeList;

      /* -------------------------------------------------------- */
      /* allocate and initialize subdomain matrix                 */
      /* -------------------------------------------------------- */

      eMatDim  = elemNNodes;
      rowSize  = elemNNodes * 16;
      csrNrows = newNNodes;
      csrIA    = new int[csrNrows+1];
      csrJA    = new int[csrNrows*rowSize];
      hypre_assert( csrJA != NULL );
      csrAA    = new double[csrNrows*rowSize];
      hypre_assert(csrAA != NULL);
      for (iR = 0; iR < csrNrows; iR++) csrIA[iR] = iR * rowSize;

      /* -------------------------------------------------------- */
      /* construct CSR matrix (with holes)                        */
      /* -------------------------------------------------------- */

      for (iE = 0; iE < nElems; iE++)
      {
         elemMat = elemMatrices[iE];
         elemNodeList = elemNodeLists[iE];
         for (iN = 0; iN < elemNNodes; iN++)
         {
            colInd = elemNodeList1D[iN+iE*elemNNodes];
            colOffset = eMatDim * iN;
            for (iN2 = 0; iN2 < elemNNodes; iN2++)
            {
               if (elemMat[colOffset+iN2] != 0.0)
               {
                  rowInd = elemNodeList1D[iN2+iE*elemNNodes];
                  offset = csrIA[rowInd]++;
                  csrJA[offset] = colInd;
                  csrAA[offset] = elemMat[iN2+colOffset];
               }
            }
         }
      }

      /* -------------------------------------------------------- */
      /* compress the CSR matrix                                  */
      /* -------------------------------------------------------- */

      offset = 0;
      for (iR = 0; iR < csrNrows; iR++)
      {
         if (csrIA[iR] > rowSize * (iR+1))
         {
            printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces ");
            printf("ERROR : rowSize too large (increase it). \n");
            printf("   => allowed = %d, actual = %d\n",rowSize,
                   csrIA[iR]-rowSize*iR);
            exit(1);
         }
         rowLeng = csrIA[iR] - iR * rowSize;
         csrIA[iR] = offset;
         start = iR * rowSize;

         MLI_Utils_IntQSort2a(&(csrJA[start]),&(csrAA[start]),0,rowLeng-1);
         count = start;
         for (k = start+1; k < start+rowLeng; k++)
         {
            if (csrJA[k] == csrJA[count]) csrAA[count] += csrAA[k];
            else
            {
               count++;
               csrJA[count] = csrJA[k];
               csrAA[count] = csrAA[k];
            }
         }
         if (rowLeng > 0) count = count - start + 1;
         else               count = 0;
         for (k = 0; k < count; k++)
         {
            csrJA[offset+k] = csrJA[start+k];
            csrAA[offset+k] = csrAA[start+k];
         }
         offset += count;
      }
      csrIA[csrNrows] = offset;

      /* -------------------------------------------------------- */
      /* change from base-0 to base-1 indexing for Fortran call   */
      /* -------------------------------------------------------- */

      for (iR = 0; iR < csrIA[csrNrows]; iR++) csrJA[iR]++;
      for (iR = 0; iR <= csrNrows; iR++) csrIA[iR]++;

      /* -------------------------------------------------------- */
      /* compute near null spaces                                 */
      /* -------------------------------------------------------- */

      strcpy(which, "Shift");
      eigenR = new double[nullspaceDim_+1];
      eigenI = new double[nullspaceDim_+1];
      eigenV = new double[csrNrows*(nullspaceDim_+1)];
      hypre_assert((long) eigenV);

#ifdef MLI_ARPACK
      sigmaR = 1.0e-6;
      sigmaI = 0.0e0;
      dnstev_(&csrNrows, &nullspaceDim_, which, &sigmaR, &sigmaI,
           csrIA, csrJA, csrAA, eigenR, eigenI, eigenV, &csrNrows, &info,
           &arpackTol_);
      if (outputLevel_ > 2)
      {
         printf("%5d : Subdomain %3d (%3d) (size=%d) : \n",mypid,iD,
                nSubdomains,csrNrows);
         for (k = 0; k < nullspaceDim_; k++)
         printf("\t%5d : ARPACK eigenvalues %2d = %16.8e %16.8e\n", mypid,
                k, eigenR[k], eigenI[k]);
      }
#else
      printf("MLI_Method_AMGSA::FATAL ERROR : ARPACK not installed.\n");
      exit(1);
#endif

//    strcpy( which, "destroy" );
#ifdef MLI_ARPACK
//    dnstev_(&csrNrows, &nullspaceDim_, which, &sigmaR, &sigmaI,
//            csrIA, csrJA, csrAA, eigenR, eigenI, eigenV, &csrNrows, &info,
//            &arpackTol_);
#else
      printf("MLI_Method_AMGSA::FATAL ERROR : ARPACK not installed.\n");
      exit(1);
#endif

      delete [] eigenR;
      delete [] eigenI;
      delete [] csrIA;
      delete [] csrJA;
      delete [] csrAA;

      /* -------------------------------------------------------- */
      /* load the null space vectors                              */
      /* -------------------------------------------------------- */

      if (nullspaceLen_ == 0) nullspaceLen_ = localNRows;
      if (nullspaceVec_ == NULL)
         nullspaceVec_ = new double[nullspaceLen_ * nullspaceDim_];
      for (iE = 0; iE < nElems; iE++)
      {
         elemNodeList = elemNodeLists[iE];
         for (iN = 0; iN < elemNNodes; iN++)
         {
            rowInd = elemNodeList[iN] - localStartRow;
            if (fp != NULL) fprintf(fp,"%7d ", rowInd+1);
            if (rowInd >= 0 && rowInd < localNRows)
            {
               saLabels_[0][rowInd] = iD;
               colInd = elemNodeList1D[iE*elemNNodes+iN];
               for (k = startCol; k < nullspaceDim_; k++)
                  nullspaceVec_[rowInd+k*nullspaceLen_] =
                        eigenV[colInd+k*csrNrows];
            }
         }
         if (fp != NULL) fprintf(fp,"\n");
      }
      delete [] elemNodeList1D;

      /* -------------------------------------------------------- */
      /* clean up                                                 */
      /* -------------------------------------------------------- */

      delete [] eigenV;
   }
   if (fp != NULL) fclose(fp);

   if ((printToFile_ & 4) != 0)
   {
      sprintf(filename, "eVec.%d", mypid);
      fp = fopen(filename, "w");
      fprintf(fp," %d %d\n", nullspaceLen_, nullspaceDim_);
      for ( iN = 0; iN < nullspaceLen_; iN++ )
      {
         for ( k = 0; k < nullspaceDim_; k++ )
            fprintf(fp,"%17.9e ",nullspaceVec_[nullspaceLen_*k+iN]);
         fprintf(fp,"\n");
      }
      fclose(fp);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces ends.\n");
#endif

   return 0;
}

/***********************************************************************
 * SET UP AGGREGATES BASED ON FEI SUBDOMAIN INFORMATION
 ***********************************************************************
 * set up domain decomposition method by having each subdomain with
 * the same aggregate number 0
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupSFEIBasedAggregates(MLI *mli)
{
   int                iR, iD, level, mypid, *partition, localNRows, *aggrMap;
   int                nSubdomains, nElems, elemNNodes, **elemNodeLists;
   int                iE, iN, localStartRow, nprocs, index, count, *aggrMap2;
   MPI_Comm           comm;
   MLI_Matrix         *mliAmat;
   hypre_ParCSRMatrix *hypreA;
   MLI_SFEI           *sfei;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA:setupSFEIBasedAggregates begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if (mli == NULL)
   {
      printf("MLI_Method_AMGSA::setupSFEIBasedAggregates ERROR");
      printf(" - no mli.\n");
      exit(1);
   }
   level = 0;
   sfei = mli->getSFEI(level);
   if (sfei == NULL)
   {
      printf("MLI_Method_AMGSA::setupSFEIBasedAggregates ERROR");
      printf(" - no sfei.\n");
      exit(1);
   }
   sfei->freeStiffnessMatrices();
   nSubdomains = sfei->getNumElemBlocks();
   if (nSubdomains <= 0) return 0;

   /* --------------------------------------------------------------- */
   /* fetch communicator and matrix information                       */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   mliAmat = mli->getSystemMatrix(level);
   hypreA  = (hypre_ParCSRMatrix *) mliAmat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,
                                        &partition);
   localStartRow = partition[mypid];
   localNRows = partition[mypid+1] - localStartRow;
   free(partition);

   /* --------------------------------------------------------------- */
   /* create and fill the aggrMap array                               */
   /* --------------------------------------------------------------- */

   aggrMap  = new int[localNRows];
   aggrMap2 = new int[localNRows];
   for (iR = 0; iR < localNRows; iR++) aggrMap[iR] = -1;
   if (saDataAux_ != NULL)
   {
      index = saDataAux_[0][0] + 1;
      for (iD = 0; iD < index; iD++) delete [] saDataAux_[iD];
      delete [] saDataAux_;
   }
   saDataAux_ = new int*[nSubdomains+1];
   saDataAux_[0] = new int[nSubdomains+1];
   for (iD = 1; iD < nSubdomains+1; iD++) saDataAux_[iD] = NULL;
   saDataAux_[0][0] = nSubdomains;

   for (iD = 0; iD < nSubdomains; iD++)
   {
      for ( iR = 0; iR < localNRows; iR++ ) aggrMap2[iR] = -1;
      nElems = sfei->getBlockNumElems(iD);
      elemNNodes  = sfei->getBlockElemNEqns(iD);
      elemNodeLists = sfei->getBlockElemEqnLists(iD);
      for (iE = 0; iE < nElems; iE++)
      {
         for (iN = 0; iN < elemNNodes; iN++)
         {
            index = elemNodeLists[iE][iN] - localStartRow;
            if (index >= 0 && index < localNRows && aggrMap[index] < 0)
               aggrMap[index] = iD;
            if (index >= 0 && index < localNRows)
               aggrMap2[index] = iD;
         }
      }
      count = 0;
      for (iR = 0; iR < localNRows; iR++) if (aggrMap2[iR] >= 0) count++;
      saDataAux_[0][iD+1] = count;
      saDataAux_[iD+1] = new int[count];
      count = 0;
      for (iR = 0; iR < localNRows; iR++)
         if (aggrMap2[iR] >= 0) saDataAux_[iD+1][count++] = iR;
   }
#if 0
   /* force non-overlapped aggregates */
   for ( iD = 0; iD < nSubdomains; iD++ )
   {
      count = 0;
      for (iR = 0; iR < localNRows; iR++) if (aggrMap[iR] == iD) count++;
      saDataAux_[0][iD+1] = count;
      if (saDataAux_[iD+1] != NULL) delete [] saDataAux_[iD+1];
      saDataAux_[iD+1] = new int[count];
      count = 0;
      for (iR = 0; iR < localNRows; iR++)
         if (aggrMap[iR] == iD) saDataAux_[iD+1][count++] = iR;
   }
#endif
   delete [] aggrMap2;
   saData_[0]     = aggrMap;
   saCounts_[0]   = nSubdomains;
   numLevels_     = 2;
   minCoarseSize_ = nprocs;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupSFEIBasedAggregates ends.\n");
#endif

   return 0;
}

/***********************************************************************
 * set up domain decomposition method by extending the local problem
 * (based on Bank-Lu-Tong-Vassilevski method but with aggregation)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupExtendedDomainDecomp(MLI *mli)
{
   MLI_Function *funcPtr;

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if (mli == NULL)
   {
      printf("MLI_Method_AMGSA::setupExtendedDomainDecomp ERROR");
      printf(" - no mli.\n");
      exit(1);
   }

   /* *************************************************************** */
   /* creating the local expanded matrix                              */
   /* --------------------------------------------------------------- */

   /* --------------------------------------------------------------- */
   /* fetch communicator and fine matrix information                  */
   /* --------------------------------------------------------------- */

   MPI_Comm           comm;
   int                mypid, nprocs, level, *partition, ANRows;
   MLI_Matrix         *mli_Amat;
   hypre_ParCSRMatrix *hypreA;

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );

#ifdef MLI_DEBUG_DETAILED
   printf("%d : MLI_Method_AMGSA::setupExtendedDomainDecomp begins...\n",mypid);
#endif

   level = 0;
   mli_Amat = mli->getSystemMatrix( level );
   hypreA  = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,
                                        &partition);
   ANRows = partition[mypid+1] - partition[mypid];
   free( partition );

   /* --------------------------------------------------------------- */
   /* first save the nodeDofs and null space information              */
   /* (since genP replaces the null vectors with new ones)            */
   /* --------------------------------------------------------------- */

   int    nodeDofs, iD, iD2;
   double *nullVecs=NULL;

   nodeDofs = currNodeDofs_;
   nullVecs = new double[nullspaceDim_*ANRows];
   if (nullspaceVec_ != NULL)
   {
      for (iD = 0; iD < nullspaceDim_*ANRows; iD++)
         nullVecs[iD] = nullspaceVec_[iD];
   }
   else
   {
      for (iD = 0; iD < nullspaceDim_; iD++)
         for (iD2 = 0; iD2 < ANRows; iD2++)
            if (MABS((iD - iD2)) % nullspaceDim_ == 0)
                 nullVecs[iD*ANRows+iD2] = 1.0;
            else nullVecs[iD*ANRows+iD2] = 0.0;
   }

   /* --------------------------------------------------------------- */
   /* create the first coarse grid matrix (the off processor block    */
   /* will be the 2,2 block of my expanded matrix)                    */
   /* Note: genP_DD coarsens less on processor boundaries.            */
   /* --------------------------------------------------------------- */

   int        *ACPartition, ACNRows, ACStart, *aggrInfo, *bdryData;
   MLI_Matrix *mli_Pmat, *mli_cAmat;
   hypre_ParCSRMatrix  *hypreAc, *hypreP;
   hypre_ParCSRCommPkg *commPkg;

   genP_DD(mli_Amat, &mli_Pmat, &aggrInfo, &bdryData);
   delete [] aggrInfo;
   hypreP = (hypre_ParCSRMatrix *) mli_Pmat->getMatrix();
   commPkg = hypre_ParCSRMatrixCommPkg(hypreP);
   if (commPkg == NULL) hypre_MatvecCommPkgCreate(hypreP);

   MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
   hypreAc = (hypre_ParCSRMatrix *) mli_cAmat->getMatrix();

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreAc,
                                        &ACPartition);
   ACStart = ACPartition[mypid];
   ACNRows = ACPartition[mypid+1] - ACStart;

   /* --------------------------------------------------------------- */
   /* fetch communication information for the coarse matrix           */
   /* --------------------------------------------------------------- */

   int   nRecvs, *recvProcs, nSends, *sendProcs;

   commPkg = hypre_ParCSRMatrixCommPkg(hypreAc);
   if ( commPkg == NULL )
   {
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreAc);
      commPkg = hypre_ParCSRMatrixCommPkg(hypreAc);
   }
   nRecvs    = hypre_ParCSRCommPkgNumRecvs(commPkg);
   recvProcs = hypre_ParCSRCommPkgRecvProcs(commPkg);
   nSends    = hypre_ParCSRCommPkgNumSends(commPkg);
   sendProcs = hypre_ParCSRCommPkgSendProcs(commPkg);

   /* --------------------------------------------------------------- */
   /* calculate the size of my local expanded off matrix (this is     */
   /* needed to create the permutation matrix) - the local size of    */
   /* the expanded off matrix should be the total number of rows of   */
   /* my recv neighbors                                               */
   /* --------------------------------------------------------------- */

   int  iP, PENCols, proc, *PEPartition, PECStart, *recvLengs;

   PENCols = 0;
   if (nRecvs > 0) recvLengs = new int[nRecvs];
   for (iP = 0; iP < nRecvs; iP++)
   {
      proc = recvProcs[iP];
      PENCols += (ACPartition[proc+1] - ACPartition[proc]);
      recvLengs[iP] = ACPartition[proc+1] - ACPartition[proc];
   }
   PEPartition = new int[nprocs+1];
   MPI_Allgather(&PENCols,1,MPI_INT,&(PEPartition[1]),1,MPI_INT,comm);
   PEPartition[0] = 0;
   for (iP = 2; iP <= nprocs; iP++)
      PEPartition[iP] += PEPartition[iP-1];
   PECStart = PEPartition[mypid];

   /* --------------------------------------------------------------- */
   /* since PE(i,j) = 1 means putting external row i into local row   */
   /* j, and since I may not own row i of PE, I need to tell my       */
   /* neighbor processor who owns row i the value of j.               */
   /* ==> procOffsets                                                 */
   /* --------------------------------------------------------------- */

   int         *procOffsets, offset;
   MPI_Request *mpiRequests;
   MPI_Status  mpiStatus;

   if (nSends > 0)
   {
      mpiRequests = new MPI_Request[nSends];
      procOffsets = new int[nSends];
   }
   for (iP = 0; iP < nSends; iP++)
      MPI_Irecv(&procOffsets[iP],1,MPI_INT,sendProcs[iP],13582,comm,
                &(mpiRequests[iP]));
   offset = 0;
   for (iP = 0; iP < nRecvs; iP++)
   {
      MPI_Send(&offset, 1, MPI_INT, recvProcs[iP], 13582, comm);
      offset += (ACPartition[recvProcs[iP]+1] - ACPartition[recvProcs[iP]]);
   }
   for (iP = 0; iP < nSends; iP++)
      MPI_Wait(&(mpiRequests[iP]), &mpiStatus);
   if (nSends > 0) delete [] mpiRequests;

   /* --------------------------------------------------------------- */
   /* create the permutation matrix to gather expanded coarse matrix  */
   /* PE has same number of rows as Ac but it has many more columns   */
   /* --------------------------------------------------------------- */

   int            ierr, *rowSizes, *colInds, rowIndex;
   double         *colVals;
   char           paramString[50];
   MLI_Matrix     *mli_PE;
   HYPRE_IJMatrix IJ_PE;
   hypre_ParCSRMatrix *hyprePE;

   ierr  = HYPRE_IJMatrixCreate(comm,ACStart,ACStart+ACNRows-1,
                                PECStart,PECStart+PENCols-1,&IJ_PE);
   ierr += HYPRE_IJMatrixSetObjectType(IJ_PE, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if (ACNRows > 0) rowSizes = new int[ACNRows];
   for (iD = 0; iD < ACNRows; iD++) rowSizes[iD] = nSends;
   ierr  = HYPRE_IJMatrixSetRowSizes(IJ_PE, rowSizes);
   ierr += HYPRE_IJMatrixInitialize(IJ_PE);
   hypre_assert(!ierr);
   if (ACNRows > 0) delete [] rowSizes;
   if (nSends > 0)
   {
      colInds = new int[nSends];
      colVals = new double[nSends];
      for (iP = 0; iP < nSends; iP++) colVals[iP] = 1.0;
   }
   for (iD = 0; iD < ACNRows; iD++)
   {
      rowIndex = iD + ACStart;
      for (iP = 0; iP < nSends; iP++)
         colInds[iP] = procOffsets[iP] + PEPartition[sendProcs[iP]] + iD;
      HYPRE_IJMatrixSetValues(IJ_PE, 1, &nSends, (const int *) &rowIndex,
                (const int *) colInds, (const double *) colVals);
   }
   if (nSends > 0)
   {
      delete [] colInds;
      delete [] colVals;
      delete [] procOffsets;
   }
   HYPRE_IJMatrixAssemble(IJ_PE);
   HYPRE_IJMatrixGetObject(IJ_PE, (void **) &hyprePE);
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR");
   mli_PE = new MLI_Matrix( (void *) hyprePE, paramString, funcPtr);
   delete funcPtr;
   commPkg = hypre_ParCSRMatrixCommPkg(hyprePE);
   if (commPkg == NULL)
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hyprePE);

   /* --------------------------------------------------------------- */
   /* form the expanded coarse matrix (that is, the (2,2) block of my */
   /* final matrix). Also, form A * P (in order to form the (1,2) and */
   /* and (2,1) block                                                 */
   /* --------------------------------------------------------------- */

   MLI_Matrix         *mli_AExt;
   hypre_ParCSRMatrix *hypreAExt, *hypreAP;

   MLI_Matrix_ComputePtAP(mli_PE, mli_cAmat, &mli_AExt);
   hypreAExt = (hypre_ParCSRMatrix *) mli_AExt->getMatrix();
   hypreP    = (hypre_ParCSRMatrix *) mli_Pmat->getMatrix();
   hypreAP   = hypre_ParMatmul(hypreA, hypreP);

   /* --------------------------------------------------------------- */
   /* coarsen one more time for non-neighbor fine aggregates          */
   /* then create [A_c   Aco Poo Po2                                  */
   /*              trans Po2^T Aoo Po2]                               */
   /* --------------------------------------------------------------- */

#if 0
   // generate Po2
   MLI_Matrix *mli_Pmat2;
   // set up one aggregate per processor
//### need to set bdryData for my immediate neighbors to 0
   genP_Selective(mli_AExt, &mli_Pmat2, PENCols, bdryData);
   delete [] bdryData;

   // compute Aco Poo Po2
   hypre_ParCSRMatrix *hypreP2, *hypreP3, *hypreAP2;
   hypreP2  = (hypre_ParCSRMatrix *) mli_Pmat2->getMatrix();
   hypreAP2 = hypre_ParMatmul(hypreAP, hypreP2);
   hypreP3  = hypre_ParMatmul(hypreP, hypreP2);

   // compute Po2^T Aoo Po2
   MLI_Matrix *mli_AExt2;
   MLI_Matrix_ComputePtAP(mli_Pmat2, mli_AExt, &mli_AExt2);

   // adjust pointers
   hypre_ParCSRMatrixDestroy(hypreAP);
   hypreAP = hypreAP2;
   delete mli_Pmat;
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR");
   mli_Pmat = new MLI_Matrix(hypreP3, paramString, funcPtr);
   delete funcPtr;
   delete mli_Pmat2;
   delete mli_AExt;
   mli_AExt = mli_AExt2;
   hypreAExt = (hypre_ParCSRMatrix *) mli_AExt->getMatrix();
   hypreP    = (hypre_ParCSRMatrix *) mli_Pmat->getMatrix();
#endif

   /* --------------------------------------------------------------- */
   /* concatenate the local and the (1,2) and (2,2) block             */
   /* --------------------------------------------------------------- */

   int    *AdiagI, *AdiagJ, *APoffdI, *APoffdJ, *AEdiagI, *AEdiagJ, index;
   int    *APcmap, newNrows, *newIA, *newJA, newNnz, *auxIA, iR, iC;
   double *AdiagA, *APoffdA, *AEdiagA, *newAA;
   hypre_CSRMatrix *Adiag, *APoffd, *AEdiag;

   Adiag   = hypre_ParCSRMatrixDiag(hypreA);
   AdiagI  = hypre_CSRMatrixI(Adiag);
   AdiagJ  = hypre_CSRMatrixJ(Adiag);
   AdiagA  = hypre_CSRMatrixData(Adiag);
   APoffd  = hypre_ParCSRMatrixOffd(hypreAP);
   APoffdI = hypre_CSRMatrixI(APoffd);
   APoffdJ = hypre_CSRMatrixJ(APoffd);
   APoffdA = hypre_CSRMatrixData(APoffd);
   APcmap  = hypre_ParCSRMatrixColMapOffd(hypreAP);
   AEdiag  = hypre_ParCSRMatrixDiag(hypreAExt);
   AEdiagI = hypre_CSRMatrixI(AEdiag);
   AEdiagJ = hypre_CSRMatrixJ(AEdiag);
   AEdiagA = hypre_CSRMatrixData(AEdiag);
   newNrows = ANRows + PENCols;
   newIA    = new int[newNrows+1];
   newNnz   = AdiagI[ANRows] + AEdiagI[PENCols] + 2 * APoffdI[ANRows];
   newJA    = new int[newNnz];
   newAA    = new double[newNnz];
   auxIA    = new int[PENCols];
   newNnz   = 0;
   newIA[0] = newNnz;
   for ( iR = 0; iR < PENCols; iR++ ) auxIA[iR] = 0;

   // (1,1) and (1,2) blocks
   for ( iR = 0; iR < ANRows; iR++ )
   {
      for ( iC = AdiagI[iR]; iC < AdiagI[iR+1]; iC++ )
      {
         newJA[newNnz] = AdiagJ[iC];
         newAA[newNnz++] = AdiagA[iC];
      }
      for ( iC = APoffdI[iR]; iC < APoffdI[iR+1]; iC++ )
      {
         index = APcmap[APoffdJ[iC]];
         offset = ANRows;
         for ( iP = 0; iP < nRecvs; iP++ )
         {
            if ( index < ACPartition[recvProcs[iP]+1] )
            {
               index = index - ACPartition[recvProcs[iP]] + offset;
               break;
            }
            offset += (ACPartition[recvProcs[iP]+1] -
                       ACPartition[recvProcs[iP]]);
         }
         newJA[newNnz] = index;
         newAA[newNnz++] = APoffdA[iC];
         APoffdJ[iC] = index;
         auxIA[index-ANRows]++;
      }
      newIA[iR+1] = newNnz;
   }

   // (2,2) block
   for ( iR = ANRows; iR < ANRows+PENCols; iR++ )
   {
      newNnz += auxIA[iR-ANRows];
      for ( iC = AEdiagI[iR-ANRows]; iC < AEdiagI[iR-ANRows+1]; iC++ )
      {
         newJA[newNnz] = AEdiagJ[iC] + ANRows;
         newAA[newNnz++] = AEdiagA[iC];
      }
      newIA[iR+1] = newNnz;
   }

   // (2,1) block
   for ( iR = 0; iR < PENCols; iR++ ) auxIA[iR] = 0;
   for ( iR = 0; iR < ANRows; iR++ )
   {
      for ( iC = APoffdI[iR]; iC < APoffdI[iR+1]; iC++ )
      {
         index = APoffdJ[iC];
         offset = newIA[index] + auxIA[index-ANRows];
         newJA[offset] = iR;
         newAA[offset] = APoffdA[iC];
         auxIA[index-ANRows]++;
      }
   }

   /* --------------------------------------------------------------- */

   int                iZero=0, *newRowSizes;
   MPI_Comm           newMPIComm;
   HYPRE_IJMatrix     IJnewA;
   hypre_ParCSRMatrix *hypreNewA;
   MLI_Matrix         *mli_newA;

   MPI_Comm_split(comm, mypid, iZero, &newMPIComm);
   ierr  = HYPRE_IJMatrixCreate(newMPIComm,iZero,newNrows-1,iZero,
                                newNrows-1,&IJnewA);
   ierr += HYPRE_IJMatrixSetObjectType(IJnewA, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if ( newNrows > 0 ) newRowSizes = new int[newNrows];
   for ( iD = 0; iD < newNrows; iD++ )
      newRowSizes[iD] = newIA[iD+1] - newIA[iD];
   ierr  = HYPRE_IJMatrixSetRowSizes(IJnewA, newRowSizes);
   ierr += HYPRE_IJMatrixInitialize(IJnewA);
   hypre_assert(!ierr);
   for ( iD = 0; iD < newNrows; iD++ )
   {
      offset = newIA[iD];
      HYPRE_IJMatrixSetValues(IJnewA, 1, &newRowSizes[iD], (const int *) &iD,
               (const int *) &newJA[offset], (const double *) &newAA[offset]);
   }
   if ( newNrows > 0 ) delete [] newRowSizes;
   delete [] newIA;
   delete [] newJA;
   delete [] newAA;
   HYPRE_IJMatrixAssemble(IJnewA);
   HYPRE_IJMatrixGetObject(IJnewA, (void **) &hypreNewA);
   sprintf(paramString, "HYPRE_ParCSR" );
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   mli_newA = new MLI_Matrix( (void *) hypreNewA, paramString, funcPtr);

   /* --------------------------------------------------------------- */
   /* communicate null space vectors                                  */
   /* --------------------------------------------------------------- */

   int    rLength, sLength;
   double *tmpNullVecs, *newNullVecs;

   if ( PENCols > 0 ) tmpNullVecs = new double[PENCols*nullspaceDim_];
   if ( nRecvs > 0 ) mpiRequests = new MPI_Request[nRecvs];

   offset = 0;
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      rLength = ACPartition[recvProcs[iP]+1] - ACPartition[recvProcs[iP]];
      rLength *= nullspaceDim_;
      MPI_Irecv(&tmpNullVecs[offset],rLength,MPI_DOUBLE,recvProcs[iP],14581,
                comm,&(mpiRequests[iP]));
      offset += rLength;
   }
   for ( iP = 0; iP < nSends; iP++ )
   {
      sLength = ACNRows * nullspaceDim_;
      MPI_Send(nullVecs, sLength, MPI_DOUBLE, sendProcs[iP], 14581, comm);
   }
   for ( iP = 0; iP < nRecvs; iP++ )
      MPI_Wait( &(mpiRequests[iP]), &mpiStatus );
   if ( nRecvs > 0 ) delete [] mpiRequests;

   newNullVecs = new double[newNrows*nullspaceDim_];
   for ( iD = 0; iD < nullspaceDim_; iD++ )
      for ( iD2 = 0; iD2 < ANRows; iD2++ )
         newNullVecs[iD*newNrows+iD2] = nullVecs[iD*ANRows+iD2];
   offset = 0;
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      rLength = ACPartition[recvProcs[iP]+1] - ACPartition[recvProcs[iP]];
      for ( iD = 0; iD < nullspaceDim_; iD++ )
         for ( iD2 = 0; iD2 < rLength; iD2++ )
            newNullVecs[iD*newNrows+iD2+offset] =
               tmpNullVecs[offset+iD*rLength+iD2];
      rLength *= nullspaceDim_;
      offset += rLength;
   }
   if ( PENCols > 0 ) delete [] tmpNullVecs;

   /* --------------------------------------------------------------- */
   /* create new overlapped DD smoother using sequential SuperLU      */
   /* --------------------------------------------------------------- */

   int        *sendLengs;
   char       *targv[7];
   MLI_Solver *smootherPtr;
   MLI_Matrix *mli_PSmat;

   //sprintf( paramString, "SeqSuperLU" );
   if (!strcmp(preSmoother_, "CGMLI")) sprintf(paramString, "CGMLI");
   else                                sprintf(paramString, "CGAMG");
   smootherPtr = MLI_Solver_CreateFromName(paramString);
   sprintf( paramString, "numSweeps 10000" );
   smootherPtr->setParams(paramString, 0, NULL);
   sprintf( paramString, "tolerance 1.0e-6" );
   smootherPtr->setParams(paramString, 0, NULL);

   // restate these lines if aggregation is used for subdomain solve
   // delete [] newNullVecs;
   //sprintf( paramString, "setNullSpace" );
   //targv[0] = (char *) &nullspaceDim_;
   //targv[1] = (char *) newNullVecs;
   //smootherPtr->setParams(paramString, 2, targv);

   // send PSmat and communication information to smoother
   if ( nSends > 0 ) sendLengs = new int[nSends];
   for ( iP = 0; iP < nSends; iP++ ) sendLengs[iP] = ACNRows;
   sprintf( paramString, "setPmat" );
   mli_PSmat = mli_Pmat;
   targv[0] = (char *) mli_PSmat;
   smootherPtr->setParams(paramString, 1, targv);
   sprintf( paramString, "setCommData" );
   targv[0] = (char *) &nRecvs;
   targv[1] = (char *) recvProcs;
   targv[2] = (char *) recvLengs;
   targv[3] = (char *) &nSends;
   targv[4] = (char *) sendProcs;
   targv[5] = (char *) sendLengs;
   targv[6] = (char *) &comm;
   smootherPtr->setParams(paramString, 7, targv);
   if ( nSends > 0 ) delete [] sendLengs;

   smootherPtr->setup(mli_newA);
   mli->setSmoother( level, MLI_SMOOTHER_PRE, smootherPtr );

   /* --------------------------------------------------------------- */
   /* create prolongation and coarse grid operators                   */
   /* --------------------------------------------------------------- */

   MLI_Solver *csolvePtr;
   MLI_Matrix *mli_Rmat;

   delete mli_cAmat;

   // set up one aggregate per processor
   saCounts_[0] = 1;
   if (saData_[0] != NULL) delete [] saData_[0];
   saData_[0] = new int[ANRows];
   for ( iD = 0; iD < ANRows; iD++ ) saData_[0][iD] = 0;

   // restore nullspace changed by the last genP
   currNodeDofs_ = nodeDofs;
   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   nullspaceVec_ = new double[nullspaceDim_*ANRows];
   for ( iD = 0; iD < nullspaceDim_*ANRows; iD++)
      nullspaceVec_[iD] = nullVecs[iD];

   // create prolongation and coarse grid operators
   genP(mli_Amat, &mli_Pmat, saCounts_[0], saData_[0]);
   MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
   mli->setSystemMatrix(level+1, mli_cAmat);
   mli->setProlongation(level+1, mli_Pmat);
   sprintf(paramString, "HYPRE_ParCSRT");
   mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), paramString, NULL);
   mli->setRestriction(level, mli_Rmat);

   /* --------------------------------------------------------------- */
   /* setup coarse solver                                             */
   /* --------------------------------------------------------------- */

   strcpy( paramString, "SuperLU" );
   csolvePtr = MLI_Solver_CreateFromName( paramString );
   csolvePtr->setup(mli_cAmat);
   mli->setCoarseSolve(csolvePtr);

   /* --------------------------------------------------------------- */
   /* clean up                                                        */
   /* --------------------------------------------------------------- */

   free( ACPartition );
   delete [] PEPartition;
   delete [] auxIA;
   HYPRE_IJMatrixDestroy(IJ_PE);
   delete mli_AExt;
   delete [] nullVecs;
   delete mli_PE;

#ifdef MLI_DEBUG_DETAILED
   printf("%d : MLI_Method_AMGSA::setupExtendedDomainDecomp ends.\n",mypid);
#endif

   level = 2;
   return (level);
}

// ************************************************************************
// Purpose : Given Amat, perform preferential coarsening (small aggregates
//           near processor boundaries and create the corresponding Pmat
// (called by setupExtendedDomainDecomp)
// ------------------------------------------------------------------------

double MLI_Method_AMGSA::genP_DD(MLI_Matrix *mli_Amat,MLI_Matrix **PmatOut,
                                 int **eqn2aggrOut, int **bdryDataOut)
{
   int    mypid, nprocs, *partition, AStartRow, AEndRow, ALocalNRows;
   int    blkSize, naggr, *node2aggr, ierr, PLocalNCols, PStartCol;
   int    PLocalNRows, PStartRow, *eqn2aggr, irow, jcol, ig, *bdryData;
   int    *PCols, maxAggSize, *aggCntArray, index, **aggIndArray;
   int    aggSize, nzcnt, *rowLengths, rowNum, *colInd;
   double **PVecs, *newNull, *qArray, *rArray, *colVal;
   char   paramString[200];
   HYPRE_IJMatrix      IJPmat;
   hypre_ParCSRMatrix  *Amat, *A2mat, *Pmat;
   MLI_Matrix          *mli_A2mat=NULL, *mli_Pmat;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg *commPkg;
   MLI_Function        *funcPtr;

   /*-----------------------------------------------------------------
    * fetch matrix and machine information
    *-----------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);

   /*-----------------------------------------------------------------
    * fetch other matrix information
    *-----------------------------------------------------------------*/

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   AStartRow = partition[mypid];
   AEndRow   = partition[mypid+1] - 1;
   ALocalNRows = AEndRow - AStartRow + 1;
   free( partition );

   /*-----------------------------------------------------------------
    * reduce Amat based on the block size information (if nodeDofs_ > 1)
    *-----------------------------------------------------------------*/

   blkSize = currNodeDofs_;
   if (blkSize > 1) MLI_Matrix_Compress(mli_Amat, blkSize, &mli_A2mat);
   else             mli_A2mat = mli_Amat;
   A2mat = (hypre_ParCSRMatrix *) mli_A2mat->getMatrix();

   /*-----------------------------------------------------------------
    * modify minimum aggregate size, if needed
    *-----------------------------------------------------------------*/

   minAggrSize_ = nullspaceDim_ / currNodeDofs_;
   if ( minAggrSize_ <= 1 ) minAggrSize_ = 2;

   /*-----------------------------------------------------------------
    * perform coarsening (small aggregates on processor boundaries)
    * 10/2005 : add bdryData for secondary aggregation
    *-----------------------------------------------------------------*/

   coarsenGraded(A2mat, &naggr, &node2aggr, &bdryData);
   if (blkSize > 1 && mli_A2mat != NULL) delete mli_A2mat;
   if (blkSize > 1)
   {
      (*bdryDataOut) = new int[ALocalNRows];
      for (irow = 0; irow < ALocalNRows; irow++)
         (*bdryDataOut)[irow] = bdryData[irow/blkSize];
      delete [] bdryData;
   }
   else (*bdryDataOut) = bdryData;

   /*-----------------------------------------------------------------
    * fetch the coarse grid information and instantiate P
    *-----------------------------------------------------------------*/

   PLocalNCols = naggr * nullspaceDim_;
   MLI_Utils_GenPartition(comm, PLocalNCols, &partition);
   PStartCol = partition[mypid];
   free( partition );
   PLocalNRows = ALocalNRows;
   PStartRow   = AStartRow;
   ierr = HYPRE_IJMatrixCreate(comm,PStartRow,PStartRow+PLocalNRows-1,
                          PStartCol,PStartCol+PLocalNCols-1,&IJPmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJPmat, HYPRE_PARCSR);
   hypre_assert(!ierr);

   /*-----------------------------------------------------------------
    * expand the aggregation information if block size > 1 ==> eqn2aggr
    *-----------------------------------------------------------------*/

   if ( blkSize > 1 )
   {
      eqn2aggr = new int[ALocalNRows];
      for ( irow = 0; irow < ALocalNRows; irow++ )
         eqn2aggr[irow] = node2aggr[irow/blkSize];
      delete [] node2aggr;
   }
   else eqn2aggr = node2aggr;

   /*-----------------------------------------------------------------
    * create a compact form for the null space vectors
    * (get ready to perform QR on them)
    *-----------------------------------------------------------------*/

   PVecs = new double*[nullspaceDim_];
   PCols = new int[PLocalNRows];
   for (irow = 0; irow < nullspaceDim_; irow++)
      PVecs[irow] = new double[PLocalNRows];
   for ( irow = 0; irow < PLocalNRows; irow++ )
   {
      if ( eqn2aggr[irow] >= 0 )
           PCols[irow] = PStartCol + eqn2aggr[irow] * nullspaceDim_;
      else PCols[irow] = PStartCol + (-eqn2aggr[irow]-1) * nullspaceDim_;
      if ( nullspaceVec_ != NULL )
      {
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
         PVecs[jcol][irow] = nullspaceVec_[jcol*PLocalNRows+irow];
      }
      else
      {
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
         {
            if (irow % nullspaceDim_ == jcol) PVecs[jcol][irow] = 1.0;
            else                              PVecs[jcol][irow] = 0.0;
         }
      }
   }

   /*-----------------------------------------------------------------
    * perform QR for null space
    *-----------------------------------------------------------------*/

   newNull = NULL;
   if ( PLocalNRows > 0 )
   {
      /* ------ count the size of each aggregate ------ */

      aggCntArray = new int[naggr];
      for ( ig = 0; ig < naggr; ig++ ) aggCntArray[ig] = 0;
      for ( irow = 0; irow < PLocalNRows; irow++ )
         if ( eqn2aggr[irow] >= 0 ) aggCntArray[eqn2aggr[irow]]++;
         else                       aggCntArray[(-eqn2aggr[irow]-1)]++;
      maxAggSize = 0;
      for ( ig = 0; ig < naggr; ig++ )
         if (aggCntArray[ig] > maxAggSize) maxAggSize = aggCntArray[ig];

      /* ------ register which equation is in which aggregate ------ */

      aggIndArray = new int*[naggr];
      for ( ig = 0; ig < naggr; ig++ )
      {
         aggIndArray[ig] = new int[aggCntArray[ig]];
         aggCntArray[ig] = 0;
      }
      for ( irow = 0; irow < PLocalNRows; irow++ )
      {
         index = eqn2aggr[irow];
         if ( index >= 0 )
            aggIndArray[index][aggCntArray[index]++] = irow;
         else
            aggIndArray[-index-1][aggCntArray[-index-1]++] = irow;
      }

      /* ------ allocate storage for QR factorization ------ */

      qArray  = new double[maxAggSize * nullspaceDim_];
      rArray  = new double[nullspaceDim_ * nullspaceDim_];
      newNull = new double[naggr*nullspaceDim_*nullspaceDim_];

      /* ------ perform QR on each aggregate ------ */

      for ( ig = 0; ig < naggr; ig++ )
      {
         aggSize = aggCntArray[ig];

         if ( aggSize < nullspaceDim_ )
         {
            printf("Aggregation ERROR : underdetermined system in QR.\n");
            printf("            error on Proc %d\n", mypid);
            printf("            error on aggr %d (%d)\n", ig, naggr);
            printf("            aggr size is %d\n", aggSize);
            exit(1);
         }

         /* ------ put data into the temporary array ------ */

         for ( jcol = 0; jcol < aggSize; jcol++ )
         {
            for ( irow = 0; irow < nullspaceDim_; irow++ )
               qArray[aggSize*irow+jcol] = PVecs[irow][aggIndArray[ig][jcol]];
         }

         /* ------ call QR function ------ */

/*
         if ( currLevel_ < (numLevels_-1) )
         {
            info = MLI_Utils_QR(qArray, rArray, aggSize, nullspaceDim_);
            if (info != 0)
            {
               printf("%4d : Aggregation WARNING : QR returns non-zero for\n",
                      mypid);
               printf("  aggregate %d, size = %d, info = %d\n",ig,aggSize,info);
            }
         }
         else
         {
            for ( irow = 0; irow < nullspaceDim_; irow++ )
            {
               dtemp = 0.0;
               for ( jcol = 0; jcol < aggSize; jcol++ )
                  dtemp += qArray[aggSize*irow+jcol]*qArray[aggSize*irow+jcol];
               dtemp = 1.0 / sqrt(dtemp);
               for ( jcol = 0; jcol < aggSize; jcol++ )
                  qArray[aggSize*irow+jcol] *= dtemp;
            }
         }
*/

         /* ------ after QR, put the R into the next null space ------ */

/*
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
            for ( irow = 0; irow < nullspaceDim_; irow++ )
               newNull[ig*nullspaceDim_+jcol+irow*naggr*nullspaceDim_] =
                         rArray[jcol+nullspaceDim_*irow];
*/
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
            for ( irow = 0; irow < nullspaceDim_; irow++ )
               if ( irow == jcol )
                  newNull[ig*nullspaceDim_+jcol+irow*naggr*nullspaceDim_] = 1.0;
               else
                  newNull[ig*nullspaceDim_+jcol+irow*naggr*nullspaceDim_] = 0.0;

         /* ------ put the P to PVecs ------ */

         for ( jcol = 0; jcol < aggSize; jcol++ )
         {
            for ( irow = 0; irow < nullspaceDim_; irow++ )
            {
               index = aggIndArray[ig][jcol];
               PVecs[irow][index] = qArray[ irow*aggSize + jcol ];
            }
         }
      }
      for ( ig = 0; ig < naggr; ig++ ) delete [] aggIndArray[ig];
      delete [] aggIndArray;
      delete [] aggCntArray;
      delete [] qArray;
      delete [] rArray;
   }
   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   nullspaceVec_ = newNull;

   /*-----------------------------------------------------------------
    * initialize Pmat
    *-----------------------------------------------------------------*/

   rowLengths = new int[PLocalNRows];
   for ( irow = 0; irow < PLocalNRows; irow++ )
      rowLengths[irow] = nullspaceDim_;
   ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJPmat);
   hypre_assert(!ierr);
   delete [] rowLengths;

   /*--------------------------------------------------------------------
    * load and assemble Pmat
    *--------------------------------------------------------------------*/

   colInd = new int[nullspaceDim_];
   colVal = new double[nullspaceDim_];
   for ( irow = 0; irow < PLocalNRows; irow++ )
   {
      if ( PCols[irow] >= 0 )
      {
         nzcnt = 0;
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
         {
            if ( PVecs[jcol][irow] != 0.0 )
            {
               colInd[nzcnt] = PCols[irow] + jcol;
               colVal[nzcnt++] = PVecs[jcol][irow];
            }
         }
         rowNum = PStartRow + irow;
         HYPRE_IJMatrixSetValues(IJPmat, 1, &nzcnt,
                             (const int *) &rowNum, (const int *) colInd,
                             (const double *) colVal);
      }
   }
   ierr = HYPRE_IJMatrixAssemble(IJPmat);
   hypre_assert( !ierr );
   HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Pmat);
   commPkg = hypre_ParCSRMatrixCommPkg(Amat);
   if (!commPkg) hypre_MatvecCommPkgCreate(Amat);
   HYPRE_IJMatrixSetObjectType(IJPmat, -1);
   HYPRE_IJMatrixDestroy( IJPmat );
   delete [] colInd;
   delete [] colVal;

   /*-----------------------------------------------------------------
    * clean up
    *-----------------------------------------------------------------*/

   if ( PCols != NULL ) delete [] PCols;
   if ( PVecs != NULL )
   {
      for (irow = 0; irow < nullspaceDim_; irow++)
         if ( PVecs[irow] != NULL ) delete [] PVecs[irow];
      delete [] PVecs;
   }
   (*eqn2aggrOut) = eqn2aggr;

   /*-----------------------------------------------------------------
    * set up and return Pmat
    *-----------------------------------------------------------------*/

   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   mli_Pmat = new MLI_Matrix( Pmat, paramString, funcPtr );
   (*PmatOut) = mli_Pmat;
   delete funcPtr;
   return 0.0;
}

// *********************************************************************
// graded coarsening scheme (Given a graph, aggregate on the local subgraph
// but give smaller aggregate near processor boundaries)
// (called by setupExtendedDomainDecomp/genP_DD)
// ---------------------------------------------------------------------

int MLI_Method_AMGSA::coarsenGraded(hypre_ParCSRMatrix *hypreG,
                         int *mliAggrLeng, int **mliAggrArray, int **bdryData)
{
   MPI_Comm  comm;
   int       mypid, nprocs, *partition, startRow, endRow, maxInd;
   int       localNRows, naggr=0, *node2aggr, *aggrSizes, nUndone;
   int       irow, jcol, colNum, rowLeng, *cols, globalNRows;
   int       *nodeStat, selectFlag, nSelected=0, nNotSelected=0, count;
   int       ibuf[2], itmp[2], *bdrySet, localMinSize, index, maxCount;
   int       *GDiagI, *GDiagJ, *GOffdI;
   double    maxVal, *vals, *GDiagA;
   hypre_CSRMatrix *GDiag, *GOffd;
#ifdef MLI_DEBUG_DETAILED
   int       rowNum;
#endif

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(hypreG);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreG,
                                        &partition);
   startRow = partition[mypid];
   endRow   = partition[mypid+1] - 1;
   free( partition );
   localNRows = endRow - startRow + 1;
   MPI_Allreduce(&localNRows, &globalNRows, 1, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(G) : total nodes to aggregate = %d\n",
             globalNRows);
   }
   GDiag  = hypre_ParCSRMatrixDiag(hypreG);
   GDiagI = hypre_CSRMatrixI(GDiag);
   GDiagJ = hypre_CSRMatrixJ(GDiag);
   GDiagA = hypre_CSRMatrixData(GDiag);
   GOffd  = hypre_ParCSRMatrixOffd(hypreG);
   GOffdI = hypre_CSRMatrixI(GOffd);

   /*-----------------------------------------------------------------
    * allocate status arrays
    *-----------------------------------------------------------------*/

   if (localNRows > 0)
   {
      node2aggr = new int[localNRows];
      aggrSizes = new int[localNRows];
      nodeStat  = new int[localNRows];
      bdrySet   = new int[localNRows];
      for ( irow = 0; irow < localNRows; irow++ )
      {
         aggrSizes[irow] = 0;
         node2aggr[irow] = -1;
         nodeStat[irow]  = MLI_METHOD_AMGSA_READY;
         bdrySet[irow]   = 0;
      }
   }
   else node2aggr = aggrSizes = nodeStat = bdrySet = NULL;

   /*-----------------------------------------------------------------
    * search for zero rows and rows near the processor boundaries
    *-----------------------------------------------------------------*/

   for ( irow = 0; irow < localNRows; irow++ )
   {
      rowLeng = GDiagI[irow+1] - GDiagI[irow];
      if (rowLeng <= 0)
      {
         nodeStat[irow] = MLI_METHOD_AMGSA_NOTSELECTED;
         nNotSelected++;
      }
      if (GOffdI != NULL && (GOffdI[irow+1] - GOffdI[irow]) > 0)
         bdrySet[irow] = 1;
   }

   /*-----------------------------------------------------------------
    * Phase 0 : form aggregates near the boundary (aggregates should be
    *           as small as possible, but has to satisfy min size.
    *           The algorithm gives preference to nodes on boundaries.)
    *-----------------------------------------------------------------*/

   localMinSize = nullspaceDim_ / currNodeDofs_ * 2;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY && bdrySet[irow] == 1 )
      {
         nSelected++;
         node2aggr[irow]  = - naggr - 1;
         aggrSizes[naggr] = 1;
         nodeStat[irow]  = MLI_METHOD_AMGSA_SELECTED2;
         if (localMinSize > 1)
         {
            rowLeng = GDiagI[irow+1] - GDiagI[irow];
            cols = &(GDiagJ[GDiagI[irow]]);
            jcol = 0;
            while ( aggrSizes[naggr] < localMinSize && jcol < rowLeng )
            {
               index = cols[jcol];
               if ( index != irow && (bdrySet[index] != 1) &&
                    nodeStat[irow] == MLI_METHOD_AMGSA_READY )
               {
                  node2aggr[index] = naggr;
                  aggrSizes[naggr]++;
                  nodeStat[index]  = MLI_METHOD_AMGSA_SELECTED2;
               }
               jcol++;
            }
         }
         naggr++;
      }
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if (outputLevel_ > 1) MPI_Allreduce(itmp, ibuf, 2, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(G) P0 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(G) P0 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   // search for seed node with largest number of neighbors

   maxInd   = -1;
   maxCount = -1;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
      {
         count = 0;
         rowLeng = GDiagI[irow+1] - GDiagI[irow];
         cols = &(GDiagJ[GDiagI[irow]]);
         for ( jcol = 0; jcol < rowLeng; jcol++ )
         {
            index = cols[jcol];
            if ( nodeStat[index] == MLI_METHOD_AMGSA_READY ) count++;
         }
         if ( count > maxCount )
         {
            maxCount = count;
            maxInd = irow;
         }
      }
   }

   /*-----------------------------------------------------------------
    * Phase 1 : form aggregates
    *-----------------------------------------------------------------*/

   for ( irow = 0; irow < localNRows; irow++ )
   {
      if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
      {
         rowLeng = GDiagI[irow+1] - GDiagI[irow];
         cols = &(GDiagJ[GDiagI[irow]]);
         selectFlag = 1;
         count      = 1;
         for ( jcol = 0; jcol < rowLeng; jcol++ )
         {
            colNum = cols[jcol];
            if ( nodeStat[colNum] != MLI_METHOD_AMGSA_READY )
            {
               selectFlag = 0;
               break;
            }
            else count++;
         }
         if ( selectFlag == 1 && count >= minAggrSize_ )
         {
            aggrSizes[naggr] = 0;
            for ( jcol = 0; jcol < rowLeng; jcol++ )
            {
               colNum = cols[jcol];
               node2aggr[colNum] = naggr;
               nodeStat[colNum] = MLI_METHOD_AMGSA_SELECTED;
               aggrSizes[naggr]++;
               nSelected++;
            }
            naggr++;
         }
      }
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if (outputLevel_ > 1) MPI_Allreduce(itmp, ibuf, 2, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(G) P1 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(G) P1 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 2 : put the rest into one of the existing aggregates
    *-----------------------------------------------------------------*/

   if ( (nSelected+nNotSelected) < localNRows )
   {
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            rowLeng = GDiagI[irow+1] - GDiagI[irow];
            cols    = &(GDiagJ[GDiagI[irow]]);
            vals    = &(GDiagA[GDiagI[irow]]);
            maxInd  = -1;
            maxVal  = 0.0;
            for ( jcol = 0; jcol < rowLeng; jcol++ )
            {
               colNum = cols[jcol];
               if ( nodeStat[colNum] == MLI_METHOD_AMGSA_SELECTED )
               {
                  if (vals[jcol] > maxVal)
                  {
                     maxInd = colNum;
                     maxVal = vals[jcol];
                  }
               }
            }
            if ( maxInd != -1 )
            {
               node2aggr[irow] = node2aggr[maxInd];
               nodeStat[irow] = MLI_METHOD_AMGSA_PENDING;
               aggrSizes[node2aggr[maxInd]]++;
            }
         }
      }
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_PENDING )
         {
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nSelected++;
         }
      }
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if (outputLevel_ > 1) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(G) P2 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(G) P2 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 3 : form aggregates for all other rows
    *-----------------------------------------------------------------*/

   if ( (nSelected+nNotSelected) < localNRows )
   {
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            rowLeng = GDiagI[irow+1] - GDiagI[irow];
            cols    = &(GDiagJ[GDiagI[irow]]);
            count = 1;
            for ( jcol = 0; jcol < rowLeng; jcol++ )
            {
               colNum = cols[jcol];
               if ( nodeStat[colNum] == MLI_METHOD_AMGSA_READY ) count++;
            }
            if ( count > 1 && count >= minAggrSize_ )
            {
               aggrSizes[naggr] = 0;
               for ( jcol = 0; jcol < rowLeng; jcol++ )
               {
                  colNum = cols[jcol];
                  if ( nodeStat[colNum] == MLI_METHOD_AMGSA_READY )
                  {
                     nodeStat[colNum] = MLI_METHOD_AMGSA_SELECTED;
                     node2aggr[colNum] = naggr;
                     aggrSizes[naggr]++;
                     nSelected++;
                  }
               }
               naggr++;
            }
         }
      }
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if (outputLevel_ > 1) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(G) P3 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(G) P3 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 4 : finally put all lone rows into some neighbor aggregate
    *-----------------------------------------------------------------*/

   if ( (nSelected+nNotSelected) < localNRows )
   {
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            rowLeng = GDiagI[irow+1] - GDiagI[irow];
            cols    = &(GDiagJ[GDiagI[irow]]);
            for ( jcol = 0; jcol < rowLeng; jcol++ )
            {
               colNum = cols[jcol];
               if ( nodeStat[colNum] == MLI_METHOD_AMGSA_SELECTED )
               {
                  node2aggr[irow] = node2aggr[colNum];
                  nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
                  aggrSizes[node2aggr[colNum]]++;
                  nSelected++;
                  break;
               }
            }
         }
      }
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if ( outputLevel_ > 1 ) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(G) P4 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(G) P4 : no. nodes aggregated  = %d\n",ibuf[1]);
   }
   nUndone = localNRows - nSelected - nNotSelected;
//if ( nUndone > 0 )
   if ( nUndone > localNRows )
   {
      count = nUndone / minAggrSize_;
      if ( count == 0 ) count = 1;
      count += naggr;
      irow = jcol = 0;
      while ( nUndone > 0 )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            node2aggr[irow] = naggr;
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nUndone--;
            nSelected++;
            jcol++;
            if ( jcol >= minAggrSize_ && naggr < count-1 )
            {
               jcol = 0;
               naggr++;
            }
         }
         irow++;
      }
      naggr = count;
   }
   itmp[0] = naggr;
   itmp[1] = nSelected;
   if ( outputLevel_ > 1 ) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(G) P5 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(G) P5 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * diagnostics
    *-----------------------------------------------------------------*/

   if ( (nSelected+nNotSelected) < localNRows )
   {
#ifdef MLI_DEBUG_DETAILED
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            rowNum = startRow + irow;
            printf("%5d : unaggregated node = %8d\n", mypid, rowNum);
            hypre_ParCSRMatrixGetRow(hypreG,rowNum,&rowLeng,&cols,NULL);
            for ( jcol = 0; jcol < rowLeng; jcol++ )
            {
               colNum = cols[jcol];
               printf("ERROR : neighbor of unselected node %9d = %9d\n",
                     rowNum, colNum);
            }
            hypre_ParCSRMatrixRestoreRow(hypreG,rowNum,&rowLeng,&cols,NULL);
         }
      }
#else
      printf("%5d : ERROR - not all nodes aggregated.\n", mypid);
      exit(1);
#endif
   }

   /*-----------------------------------------------------------------
    * clean up and initialize the output arrays
    *-----------------------------------------------------------------*/

   if (localNRows > 0) delete [] aggrSizes;
   if (localNRows > 0) delete [] nodeStat;
   if (localNRows == 1 && naggr == 0)
   {
      node2aggr[0] = 0;
      naggr = 1;
   }
   (*bdryData)     = bdrySet;
   (*mliAggrArray) = node2aggr;
   (*mliAggrLeng)  = naggr;
   return 0;
}

// ************************************************************************
// Purpose : Given Amat, perform preferential coarsening (no coarsening
//           when the bdry flag = 1
// (called by setupExtendedDomainDecomp)
// ------------------------------------------------------------------------

double MLI_Method_AMGSA::genP_Selective(MLI_Matrix *mli_Amat,
                  MLI_Matrix **PmatOut, int ALen, int *bdryData)
{
   int    mypid, nprocs, *partition, AStartRow, AEndRow, ALocalNRows;
   int    blkSize, naggr, *node2aggr, ierr, PLocalNCols, PStartCol;
   int    PLocalNRows, PStartRow, *eqn2aggr, irow, jcol, ig;
   int    *PCols, maxAggSize, *aggCntArray, index, **aggIndArray;
   int    aggSize, nzcnt, *rowLengths, rowNum, *colInd, *compressBdryData;
   double **PVecs, *newNull, *qArray, *rArray, *colVal;
   char   paramString[200];
   HYPRE_IJMatrix      IJPmat;
   hypre_ParCSRMatrix  *Amat, *A2mat, *Pmat;
   MLI_Matrix          *mli_A2mat=NULL, *mli_Pmat;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg *commPkg;
   MLI_Function        *funcPtr;

   /*-----------------------------------------------------------------
    * fetch matrix information
    *-----------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   AStartRow = partition[mypid];
   AEndRow   = partition[mypid+1] - 1;
   ALocalNRows = AEndRow - AStartRow + 1;
   free(partition);

   /*-----------------------------------------------------------------
    * reduce Amat based on the block size information (if nodeDofs_ > 1)
    *-----------------------------------------------------------------*/

   blkSize = currNodeDofs_;
   if (blkSize > 1) MLI_Matrix_Compress(mli_Amat, blkSize, &mli_A2mat);
   else             mli_A2mat = mli_Amat;
   A2mat = (hypre_ParCSRMatrix *) mli_A2mat->getMatrix();

   /*-----------------------------------------------------------------
    * modify minimum aggregate size, if needed
    *-----------------------------------------------------------------*/

   minAggrSize_ = nullspaceDim_ / currNodeDofs_;
   if (minAggrSize_ <= 1) minAggrSize_ = 2;

   /*-----------------------------------------------------------------
    * perform coarsening (no aggregation on processor boundaries)
    *-----------------------------------------------------------------*/

   if (blkSize > 1)
   {
      compressBdryData = new int[ALocalNRows/blkSize];
      for (irow = 0; irow < ALocalNRows; irow+=blkSize)
         compressBdryData[irow/blkSize] = bdryData[irow];
   }
   else compressBdryData = bdryData;

   coarsenSelective(A2mat, &naggr, &node2aggr, bdryData);
   if (blkSize > 1 && mli_A2mat != NULL) delete mli_A2mat;
   if (blkSize > 1) delete [] compressBdryData;

   /*-----------------------------------------------------------------
    * fetch the coarse grid information and instantiate P
    *-----------------------------------------------------------------*/

   PLocalNCols = naggr * nullspaceDim_;
   MLI_Utils_GenPartition(comm, PLocalNCols, &partition);
   PStartCol = partition[mypid];
   free( partition );
   PLocalNRows = ALocalNRows;
   PStartRow   = AStartRow;
   ierr = HYPRE_IJMatrixCreate(comm,PStartRow,PStartRow+PLocalNRows-1,
                          PStartCol,PStartCol+PLocalNCols-1,&IJPmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJPmat, HYPRE_PARCSR);
   hypre_assert(!ierr);

   /*-----------------------------------------------------------------
    * expand the aggregation information if block size > 1 ==> eqn2aggr
    *-----------------------------------------------------------------*/

   if ( blkSize > 1 )
   {
      eqn2aggr = new int[ALocalNRows];
      for ( irow = 0; irow < ALocalNRows; irow++ )
         eqn2aggr[irow] = node2aggr[irow/blkSize];
      delete [] node2aggr;
   }
   else eqn2aggr = node2aggr;

   /*-----------------------------------------------------------------
    * create a compact form for the null space vectors
    * (get ready to perform QR on them)
    *-----------------------------------------------------------------*/

   PVecs = new double*[nullspaceDim_];
   PCols = new int[PLocalNRows];
   for (irow = 0; irow < nullspaceDim_; irow++)
      PVecs[irow] = new double[PLocalNRows];
   for ( irow = 0; irow < PLocalNRows; irow++ )
   {
      if ( eqn2aggr[irow] >= 0 )
           PCols[irow] = PStartCol + eqn2aggr[irow] * nullspaceDim_;
      else PCols[irow] = PStartCol + (-eqn2aggr[irow]-1) * nullspaceDim_;
      if ( nullspaceVec_ != NULL )
      {
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
         PVecs[jcol][irow] = nullspaceVec_[jcol*PLocalNRows+irow];
      }
      else
      {
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
         {
            if (irow % nullspaceDim_ == jcol) PVecs[jcol][irow] = 1.0;
            else                              PVecs[jcol][irow] = 0.0;
         }
      }
   }

   /*-----------------------------------------------------------------
    * perform QR for null space
    *-----------------------------------------------------------------*/

   newNull = NULL;
   if ( PLocalNRows > 0 )
   {
      /* ------ count the size of each aggregate ------ */

      aggCntArray = new int[naggr];
      for ( ig = 0; ig < naggr; ig++ ) aggCntArray[ig] = 0;
      for ( irow = 0; irow < PLocalNRows; irow++ )
         if ( eqn2aggr[irow] >= 0 ) aggCntArray[eqn2aggr[irow]]++;
         else                       aggCntArray[(-eqn2aggr[irow]-1)]++;
      maxAggSize = 0;
      for ( ig = 0; ig < naggr; ig++ )
         if (aggCntArray[ig] > maxAggSize) maxAggSize = aggCntArray[ig];

      /* ------ register which equation is in which aggregate ------ */

      aggIndArray = new int*[naggr];
      for ( ig = 0; ig < naggr; ig++ )
      {
         aggIndArray[ig] = new int[aggCntArray[ig]];
         aggCntArray[ig] = 0;
      }
      for ( irow = 0; irow < PLocalNRows; irow++ )
      {
         index = eqn2aggr[irow];
         if ( index >= 0 )
            aggIndArray[index][aggCntArray[index]++] = irow;
         else
            aggIndArray[-index-1][aggCntArray[-index-1]++] = irow;
      }

      /* ------ allocate storage for QR factorization ------ */

      qArray  = new double[maxAggSize * nullspaceDim_];
      rArray  = new double[nullspaceDim_ * nullspaceDim_];
      newNull = new double[naggr*nullspaceDim_*nullspaceDim_];

      /* ------ perform QR on each aggregate ------ */

      for ( ig = 0; ig < naggr; ig++ )
      {
         aggSize = aggCntArray[ig];

         if ( aggSize < nullspaceDim_ )
         {
            printf("Aggregation ERROR : underdetermined system in QR.\n");
            printf("            error on Proc %d\n", mypid);
            printf("            error on aggr %d (%d)\n", ig, naggr);
            printf("            aggr size is %d\n", aggSize);
            exit(1);
         }

         /* ------ put data into the temporary array ------ */

         for ( jcol = 0; jcol < aggSize; jcol++ )
         {
            for ( irow = 0; irow < nullspaceDim_; irow++ )
               qArray[aggSize*irow+jcol] = PVecs[irow][aggIndArray[ig][jcol]];
         }

         /* ------ after QR, put the R into the next null space ------ */

         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
            for ( irow = 0; irow < nullspaceDim_; irow++ )
               if ( irow == jcol )
                  newNull[ig*nullspaceDim_+jcol+irow*naggr*nullspaceDim_]=1.0;
               else
                  newNull[ig*nullspaceDim_+jcol+irow*naggr*nullspaceDim_]=0.0;

         /* ------ put the P to PVecs ------ */

         for ( jcol = 0; jcol < aggSize; jcol++ )
         {
            for ( irow = 0; irow < nullspaceDim_; irow++ )
            {
               index = aggIndArray[ig][jcol];
               PVecs[irow][index] = qArray[ irow*aggSize + jcol ];
            }
         }
      }
      for ( ig = 0; ig < naggr; ig++ ) delete [] aggIndArray[ig];
      delete [] aggIndArray;
      delete [] aggCntArray;
      delete [] qArray;
      delete [] rArray;
   }
   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   nullspaceVec_ = newNull;

   /*-----------------------------------------------------------------
    * initialize Pmat
    *-----------------------------------------------------------------*/

   rowLengths = new int[PLocalNRows];
   for ( irow = 0; irow < PLocalNRows; irow++ )
      rowLengths[irow] = nullspaceDim_;
   ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJPmat);
   hypre_assert(!ierr);
   delete [] rowLengths;

   /*--------------------------------------------------------------------
    * load and assemble Pmat
    *--------------------------------------------------------------------*/

   colInd = new int[nullspaceDim_];
   colVal = new double[nullspaceDim_];
   for ( irow = 0; irow < PLocalNRows; irow++ )
   {
      if ( PCols[irow] >= 0 )
      {
         nzcnt = 0;
         for ( jcol = 0; jcol < nullspaceDim_; jcol++ )
         {
            if ( PVecs[jcol][irow] != 0.0 )
            {
               colInd[nzcnt] = PCols[irow] + jcol;
               colVal[nzcnt++] = PVecs[jcol][irow];
            }
         }
         rowNum = PStartRow + irow;
         HYPRE_IJMatrixSetValues(IJPmat, 1, &nzcnt,
                             (const int *) &rowNum, (const int *) colInd,
                             (const double *) colVal);
      }
   }
   ierr = HYPRE_IJMatrixAssemble(IJPmat);
   hypre_assert( !ierr );
   HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Pmat);
   commPkg = hypre_ParCSRMatrixCommPkg(Amat);
   if (!commPkg) hypre_MatvecCommPkgCreate(Amat);
   HYPRE_IJMatrixSetObjectType(IJPmat, -1);
   HYPRE_IJMatrixDestroy( IJPmat );
   delete [] colInd;
   delete [] colVal;

   /*-----------------------------------------------------------------
    * clean up
    *-----------------------------------------------------------------*/

   if (PCols != NULL) delete [] PCols;
   if (PVecs != NULL)
   {
      for (irow = 0; irow < nullspaceDim_; irow++)
         if (PVecs[irow] != NULL) delete [] PVecs[irow];
      delete [] PVecs;
   }
   delete [] eqn2aggr;

   /*-----------------------------------------------------------------
    * set up and return Pmat
    *-----------------------------------------------------------------*/

   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   mli_Pmat = new MLI_Matrix( Pmat, paramString, funcPtr );
   (*PmatOut) = mli_Pmat;
   delete funcPtr;
   return 0.0;
}

// *********************************************************************
// selective coarsening scheme (Given a graph, aggregate on the local
// subgraph but no aggregation near processor boundaries)
// (called by setupExtendedDomainDecomp/genP_Selective)
// ---------------------------------------------------------------------

int MLI_Method_AMGSA::coarsenSelective(hypre_ParCSRMatrix *hypreG,
                         int *naggrOut, int **aggrInfoOut, int *bdryData)
{
   MPI_Comm  comm;
   int       mypid, nprocs, *partition, startRow, endRow, maxInd;
   int       localNRows, naggr=0, *node2aggr, *aggrSizes, nUndone;
   int       irow, jcol, colNum, rowLeng, *cols, globalNRows;
   int       *nodeStat, selectFlag, nSelected=0, nNotSelected=0, count;
   int       *GDiagI, *GDiagJ;
   double    maxVal, *vals, *GDiagA;
   hypre_CSRMatrix *GDiag;
#ifdef MLI_DEBUG_DETAILED
   int       rowNum;
#endif

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(hypreG);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreG,
                                        &partition);
   startRow = partition[mypid];
   endRow   = partition[mypid+1] - 1;
   free( partition );
   localNRows = endRow - startRow + 1;
   MPI_Allreduce(&localNRows, &globalNRows, 1, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && outputLevel_ > 1 )
   {
      printf("\t*** Aggregation(G) : total nodes to aggregate = %d\n",
             globalNRows);
   }
   GDiag  = hypre_ParCSRMatrixDiag(hypreG);
   GDiagI = hypre_CSRMatrixI(GDiag);
   GDiagJ = hypre_CSRMatrixJ(GDiag);
   GDiagA = hypre_CSRMatrixData(GDiag);

   /*-----------------------------------------------------------------
    * allocate status arrays
    *-----------------------------------------------------------------*/

   if (localNRows > 0)
   {
      node2aggr = new int[localNRows];
      aggrSizes = new int[localNRows];
      nodeStat  = new int[localNRows];
      for ( irow = 0; irow < localNRows; irow++ )
      {
         if (bdryData[irow] == 1)
         {
            nodeStat[irow]  = MLI_METHOD_AMGSA_SELECTED;
            node2aggr[irow] = naggr++;
            aggrSizes[irow] = 1;
            nSelected++;
         }
         else
         {
            nodeStat[irow]  = MLI_METHOD_AMGSA_READY;
            node2aggr[irow] = -1;
            aggrSizes[irow] = 0;
         }
      }
   }
   else node2aggr = aggrSizes = nodeStat = NULL;

   /*-----------------------------------------------------------------
    * search for zero rows and rows near the processor boundaries
    *-----------------------------------------------------------------*/

   for ( irow = 0; irow < localNRows; irow++ )
   {
      rowLeng = GDiagI[irow+1] - GDiagI[irow];
      if (rowLeng <= 0)
      {
         nodeStat[irow] = MLI_METHOD_AMGSA_NOTSELECTED;
         nNotSelected++;
      }
   }

   /*-----------------------------------------------------------------
    * Phase 1 : form aggregates
    *-----------------------------------------------------------------*/

   for (irow = 0; irow < localNRows; irow++)
   {
      if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
      {
         rowLeng = GDiagI[irow+1] - GDiagI[irow];
         cols = &(GDiagJ[GDiagI[irow]]);
         selectFlag = 1;
         count      = 1;
         for ( jcol = 0; jcol < rowLeng; jcol++ )
         {
            colNum = cols[jcol];
            if (nodeStat[colNum] != MLI_METHOD_AMGSA_READY)
            {
               selectFlag = 0;
               break;
            }
            else count++;
         }
         if (selectFlag == 1 && count >= minAggrSize_)
         {
            aggrSizes[naggr] = 0;
            for ( jcol = 0; jcol < rowLeng; jcol++ )
            {
               colNum = cols[jcol];
               node2aggr[colNum] = naggr;
               nodeStat[colNum] = MLI_METHOD_AMGSA_SELECTED;
               aggrSizes[naggr]++;
               nSelected++;
            }
            naggr++;
         }
      }
   }

   /*-----------------------------------------------------------------
    * Phase 2 : put the rest into one of the existing aggregates
    *-----------------------------------------------------------------*/

   if ((nSelected+nNotSelected) < localNRows)
   {
      for (irow = 0; irow < localNRows; irow++)
      {
         if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
         {
            rowLeng = GDiagI[irow+1] - GDiagI[irow];
            cols    = &(GDiagJ[GDiagI[irow]]);
            vals    = &(GDiagA[GDiagI[irow]]);
            maxInd  = -1;
            maxVal  = 0.0;
            for (jcol = 0; jcol < rowLeng; jcol++)
            {
               colNum = cols[jcol];
               if (nodeStat[colNum] == MLI_METHOD_AMGSA_SELECTED)
               {
                  if (vals[jcol] > maxVal)
                  {
                     maxInd = colNum;
                     maxVal = vals[jcol];
                  }
               }
            }
            if (maxInd != -1)
            {
               node2aggr[irow] = node2aggr[maxInd];
               nodeStat[irow] = MLI_METHOD_AMGSA_PENDING;
               aggrSizes[node2aggr[maxInd]]++;
            }
         }
      }
      for (irow = 0; irow < localNRows; irow++)
      {
         if (nodeStat[irow] == MLI_METHOD_AMGSA_PENDING)
         {
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nSelected++;
         }
      }
   }

   /*-----------------------------------------------------------------
    * Phase 3 : form aggregates for all other rows
    *-----------------------------------------------------------------*/

   if ((nSelected+nNotSelected) < localNRows)
   {
      for (irow = 0; irow < localNRows; irow++)
      {
         if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
         {
            rowLeng = GDiagI[irow+1] - GDiagI[irow];
            cols    = &(GDiagJ[GDiagI[irow]]);
            count = 1;
            for (jcol = 0; jcol < rowLeng; jcol++)
            {
               colNum = cols[jcol];
               if (nodeStat[colNum] == MLI_METHOD_AMGSA_READY) count++;
            }
            if (count > 1 && count >= minAggrSize_)
            {
               aggrSizes[naggr] = 0;
               for (jcol = 0; jcol < rowLeng; jcol++)
               {
                  colNum = cols[jcol];
                  if (nodeStat[colNum] == MLI_METHOD_AMGSA_READY)
                  {
                     nodeStat[colNum] = MLI_METHOD_AMGSA_SELECTED;
                     node2aggr[colNum] = naggr;
                     aggrSizes[naggr]++;
                     nSelected++;
                  }
               }
               naggr++;
            }
         }
      }
   }

   /*-----------------------------------------------------------------
    * Phase 4 : finally put all lone rows into some neighbor aggregate
    *-----------------------------------------------------------------*/

   if ((nSelected+nNotSelected) < localNRows)
   {
      for (irow = 0; irow < localNRows; irow++)
      {
         if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
         {
            rowLeng = GDiagI[irow+1] - GDiagI[irow];
            cols    = &(GDiagJ[GDiagI[irow]]);
            for (jcol = 0; jcol < rowLeng; jcol++)
            {
               colNum = cols[jcol];
               if (nodeStat[colNum] == MLI_METHOD_AMGSA_SELECTED)
               {
                  node2aggr[irow] = node2aggr[colNum];
                  nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
                  aggrSizes[node2aggr[colNum]]++;
                  nSelected++;
                  break;
               }
            }
         }
      }
   }
   nUndone = localNRows - nSelected - nNotSelected;
//if ( nUndone > 0 )
   if ( nUndone > localNRows )
   {
      count = nUndone / minAggrSize_;
      if ( count == 0 ) count = 1;
      count += naggr;
      irow = jcol = 0;
      while ( nUndone > 0 )
      {
         if ( nodeStat[irow] == MLI_METHOD_AMGSA_READY )
         {
            node2aggr[irow] = naggr;
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nUndone--;
            nSelected++;
            jcol++;
            if ( jcol >= minAggrSize_ && naggr < count-1 )
            {
               jcol = 0;
               naggr++;
            }
         }
         irow++;
      }
      naggr = count;
   }

   /*-----------------------------------------------------------------
    * diagnostics
    *-----------------------------------------------------------------*/

   if ((nSelected+nNotSelected) < localNRows)
   {
#ifdef MLI_DEBUG_DETAILED
      for (irow = 0; irow < localNRows; irow++)
      {
         if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
         {
            rowNum = startRow + irow;
            printf("%5d : unaggregated node = %8d\n", mypid, rowNum);
            hypre_ParCSRMatrixGetRow(hypreG,rowNum,&rowLeng,&cols,NULL);
            for ( jcol = 0; jcol < rowLeng; jcol++ )
            {
               colNum = cols[jcol];
               printf("ERROR : neighbor of unselected node %9d = %9d\n",
                     rowNum, colNum);
            }
            hypre_ParCSRMatrixRestoreRow(hypreG,rowNum,&rowLeng,&cols,NULL);
         }
      }
#else
      printf("%5d : ERROR - not all nodes aggregated.\n", mypid);
      exit(1);
#endif
   }

   /*-----------------------------------------------------------------
    * clean up and initialize the output arrays
    *-----------------------------------------------------------------*/

   if (localNRows > 0) delete [] aggrSizes;
   if (localNRows > 0) delete [] nodeStat;
   if (localNRows == 1 && naggr == 0)
   {
      node2aggr[0] = 0;
      naggr = 1;
   }
   (*aggrInfoOut) = node2aggr;
   (*naggrOut)  = naggr;
   return 0;
}

//**********************************************************************
// set up domain decomposition method by extending the local problem
// (A simplified version of setupExtendedDomainDecomp using inefficient
// method - just for testing only)
// ---------------------------------------------------------------------

int MLI_Method_AMGSA::setupExtendedDomainDecomp2(MLI *mli)
{
   MLI_Function *funcPtr;
   int     nRecvs, *recvProcs, nSends, *sendProcs, ierr, *rowSizes;
   int     *colInds, rowIndex, iP;
   int     *recvLengs, mypid, nprocs, level, *Apartition, ANRows;
   int     nodeDofs, iD, iD2, AStart, AExtNRows;
   double  *nullVecs=NULL, *colVals;
   char    paramString[50];
   MPI_Comm            comm;
   MLI_Matrix          *mli_Amat;
   hypre_ParCSRMatrix  *hypreA;
   hypre_ParCSRCommPkg *commPkg;

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if (mli == NULL)
   {
      printf("MLI_Method_AMGSA::setupExtendedDomainDecomp2 ERROR");
      printf(" - no mli.\n");
      exit(1);
   }

   /* --------------------------------------------------------------- */
   /* fetch communicator and fine matrix information                  */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);

#ifdef MLI_DEBUG_DETAILED
   printf("%d : AMGSA::setupExtendedDomainDecomp2 begins...\n",mypid);
#endif

   level = 0;
   mli_Amat = mli->getSystemMatrix(level);
   hypreA  = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,
                                        &Apartition);
   AStart = Apartition[mypid];
   ANRows = Apartition[mypid+1] - AStart;

   /* --------------------------------------------------------------- */
   /* first save the nodeDofs and null space information since it     */
   /* will be destroyed soon and needs to be recovered later          */
   /* --------------------------------------------------------------- */

   nodeDofs = currNodeDofs_;
   nullVecs = new double[nullspaceDim_*ANRows];
   if (nullspaceVec_ != NULL)
   {
      for (iD = 0; iD < nullspaceDim_*ANRows; iD++)
         nullVecs[iD] = nullspaceVec_[iD];
   }
   else
   {
      for (iD = 0; iD < nullspaceDim_; iD++)
         for (iD2 = 0; iD2 < ANRows; iD2++)
            if (MABS((iD - iD2)) % nullspaceDim_ == 0)
                 nullVecs[iD*ANRows+iD2] = 1.0;
            else nullVecs[iD*ANRows+iD2] = 0.0;
   }

   /* --------------------------------------------------------------- */
   /* get my neighbor processor information                           */
   /* --------------------------------------------------------------- */

   commPkg = hypre_ParCSRMatrixCommPkg(hypreA);
   if (commPkg == NULL)
   {
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreA);
      commPkg = hypre_ParCSRMatrixCommPkg(hypreA);
   }
   nRecvs    = hypre_ParCSRCommPkgNumRecvs(commPkg);
   recvProcs = hypre_ParCSRCommPkgRecvProcs(commPkg);
   nSends    = hypre_ParCSRCommPkgNumSends(commPkg);
   sendProcs = hypre_ParCSRCommPkgSendProcs(commPkg);
   if (nRecvs > 0) recvLengs = new int[nRecvs];

   /* --------------------------------------------------------------- */
   /* calculate the size of my expanded matrix AExt which is the sum  */
   /* of the rows of my neighbors (and mine also)                     */
   /* --------------------------------------------------------------- */

   AExtNRows = 0;
   for (iP = 0; iP < nRecvs; iP++)
   {
      recvLengs[iP] = Apartition[recvProcs[iP]+1] - Apartition[recvProcs[iP]];
      AExtNRows += recvLengs[iP];
   }

   /* --------------------------------------------------------------- */
   /* communicate processor offsets for AExt (needed to create        */
   /* partition)                                                      */
   /* --------------------------------------------------------------- */

   int *AExtpartition = new int[nprocs+1];
   MPI_Allgather(&AExtNRows, 1, MPI_INT, &AExtpartition[1], 1, MPI_INT, comm);
   AExtpartition[0] = 0;
   for (iP = 1; iP < nprocs; iP++)
      AExtpartition[iP+1] = AExtpartition[iP+1] + AExtpartition[iP];

   /* --------------------------------------------------------------- */
   /* create QExt (permutation matrix for getting local AExt)         */
   /* --------------------------------------------------------------- */

   int QExtCStart = AExtpartition[mypid];
   int QExtNCols = AExtpartition[mypid+1] - QExtCStart;
   int QExtNRows = ANRows;
   int QExtRStart = AStart;
   HYPRE_IJMatrix IJ_QExt;

   ierr  = HYPRE_IJMatrixCreate(comm,QExtRStart,QExtRStart+QExtNRows-1,
                                QExtCStart,QExtCStart+QExtNCols-1,&IJ_QExt);
   ierr += HYPRE_IJMatrixSetObjectType(IJ_QExt, HYPRE_PARCSR);
   hypre_assert(!ierr);
   rowSizes = new int[QExtNRows];
   for (iD = 0; iD < ANRows; iD++) rowSizes[iD] = 2 * nSends;
   ierr  = HYPRE_IJMatrixSetRowSizes(IJ_QExt, rowSizes);
   ierr += HYPRE_IJMatrixInitialize(IJ_QExt);
   hypre_assert(!ierr);
   delete [] rowSizes;

   /* --------------------------------------------------------------- */
   /* when creating QExt, need to set up the QExtOffsets ==>          */
   /* since QExt(i,j) = 1 means putting local col i into external col */
   /* j, and since I may not own row i of PE, I need to tell my       */
   /* neighbor processor who owns row i the value of j.               */
   /* --------------------------------------------------------------- */

   int         *QExtOffsets, offset, pindex;
   MPI_Request *mpiRequests;
   MPI_Status  mpiStatus;

   if (nSends > 0)
   {
      mpiRequests = new MPI_Request[nSends];
      QExtOffsets = new int[nSends];
   }
   for (iP = 0; iP < nSends; iP++)
      MPI_Irecv(&QExtOffsets[iP],1,MPI_INT,sendProcs[iP],434243,comm,
                &(mpiRequests[iP]));
   offset = 0;
   // tell neighbor processors that their cols will become my offset
   // cols (+ my processor offset)
   for (iP = 0; iP < nRecvs; iP++)
   {
      MPI_Send(&offset, 1, MPI_INT, recvProcs[iP], 434243, comm);
      offset += (Apartition[recvProcs[iP]+1] - Apartition[recvProcs[iP]]);
   }
   for ( iP = 0; iP < nSends; iP++ )
      MPI_Wait( &(mpiRequests[iP]), &mpiStatus );
   if (nSends > 0) delete [] mpiRequests;

   /* --------------------------------------------------------------- */
   /* create QExt                                                     */
   /* --------------------------------------------------------------- */

   hypre_ParCSRMatrix *hypreQExt;
   MLI_Matrix         *mli_QExt;

   if (nSends > 0)
   {
      colInds = new int[nSends+1];
      colVals = new double[nSends+1];
      for (iP = 0; iP <= nSends; iP++) colVals[iP] = 1.0;
   }
   for (iD = 0; iD < QExtNRows; iD++)
   {
      rowIndex = QExtRStart + iD;
      colInds[0] = QExtRStart + iD;
      for (iP = 0; iP < nSends; iP++)
      {
         pindex = sendProcs[iP];
         colInds[iP] = AExtpartition[pindex] + QExtOffsets[iP] + iD;
      }
      HYPRE_IJMatrixSetValues(IJ_QExt, 1, &nSends, (const int *) &rowIndex,
                (const int *) colInds, (const double *) colVals);
   }
   if (nSends > 0)
   {
      delete [] colInds;
      delete [] colVals;
      delete [] QExtOffsets;
   }
   HYPRE_IJMatrixAssemble(IJ_QExt);
   HYPRE_IJMatrixGetObject(IJ_QExt, (void **) &hypreQExt);
   sprintf(paramString, "HYPRE_ParCSR");
   mli_QExt = new MLI_Matrix( (void *) hypreQExt, paramString, NULL);
   commPkg = hypre_ParCSRMatrixCommPkg(hypreQExt);
   if (commPkg == NULL)
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreQExt);

   /* --------------------------------------------------------------- */
   /* form the expanded fine matrix                                   */
   /* --------------------------------------------------------------- */

   MLI_Matrix         *mli_AExt;
   //hypre_ParCSRMatrix *hypreAExt;

   MLI_Matrix_ComputePtAP(mli_QExt, mli_Amat, &mli_AExt);
   //hypreAExt = (hypre_ParCSRMatrix *) mli_AExt->getMatrix();
   delete mli_QExt;
   HYPRE_IJMatrixDestroy(IJ_QExt);

   /* --------------------------------------------------------------- */
   /* communicate null space information for graded coarsening        */
   /* --------------------------------------------------------------- */

   int    rLength, sLength;
   double *tmpNullVecs;

   if (QExtNCols > 0) tmpNullVecs = new double[QExtNCols*nullspaceDim_];
   if (nRecvs > 0) mpiRequests = new MPI_Request[nRecvs];

   offset = 0;
   for (iP = 0; iP < nRecvs; iP++)
   {
      rLength = AExtpartition[recvProcs[iP]+1] - AExtpartition[recvProcs[iP]];
      rLength *= nullspaceDim_;
      MPI_Irecv(&tmpNullVecs[offset],rLength,MPI_DOUBLE,recvProcs[iP],14581,
                comm,&(mpiRequests[iP]));
      offset += rLength;
   }
   for (iP = 0; iP < nSends; iP++)
   {
      sLength = ANRows * nullspaceDim_;
      MPI_Send(nullVecs, sLength, MPI_DOUBLE, sendProcs[iP], 14581, comm);
   }
   for (iP = 0; iP < nRecvs; iP++)
      MPI_Wait(&(mpiRequests[iP]), &mpiStatus);
   if (nRecvs > 0) delete [] mpiRequests;

   if (nullspaceVec_ != NULL) delete [] nullspaceVec_;
   nullspaceVec_ = new double[QExtNCols*nullspaceDim_];
   for (iD = 0; iD < nullspaceDim_; iD++)
      for (iD2 = 0; iD2 < ANRows; iD2++)
         nullspaceVec_[iD*QExtNCols+iD2] = nullVecs[iD*ANRows+iD2];
   offset = ANRows;
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      rLength = AExtpartition[recvProcs[iP]+1] - AExtpartition[recvProcs[iP]];
      for (iD = 0; iD < nullspaceDim_; iD++)
         for (iD2 = 0; iD2 < rLength; iD2++)
            nullspaceVec_[iD*QExtNCols+iD2+offset] =
               tmpNullVecs[offset+iD*rLength+iD2];
      rLength *= nullspaceDim_;
      offset += rLength;
   }
   if (QExtNCols > 0) delete [] tmpNullVecs;
   delete [] AExtpartition;

   /* --------------------------------------------------------------- */
   /* graded coarsening on the expanded fine matrix                   */
   /* (this P will be needed later to collect neighbor unknowns       */
   /*  before smoothing - its transpose)                              */
   /* --------------------------------------------------------------- */

   MLI_Matrix *mli_PExt;
   genP_AExt(mli_AExt, &mli_PExt, ANRows);

   /* --------------------------------------------------------------- */
   /* create the local domain decomposition matrix                    */
   /* reduced so that the local subdomain matrix is smaller - will be */
   /* used as a smoother                                              */
   /* --------------------------------------------------------------- */

   MLI_Matrix *mli_ACExt;
   MLI_Matrix_ComputePtAP(mli_PExt, mli_AExt, &mli_ACExt);

   /* --------------------------------------------------------------- */
   /* need to convert the ACExt into local matrix (since the matrix   */
   /* is local)                                                       */
   /* --------------------------------------------------------------- */

   int      iZero=0, ACExtNRows, *newRowSizes, *ACExtI, *ACExtJ;
   double   *ACExtD;
   MPI_Comm newMPIComm;
   hypre_ParCSRMatrix *hypreACExt, *hypreNewA;
   hypre_CSRMatrix    *csrACExt;
   HYPRE_IJMatrix     IJnewA;

   hypreACExt = (hypre_ParCSRMatrix *) mli_AExt->getMatrix();
   ACExtNRows = hypre_ParCSRMatrixNumRows(hypreACExt);

   MPI_Comm_split(comm, mypid, iZero, &newMPIComm);
   ierr  = HYPRE_IJMatrixCreate(newMPIComm,iZero,ACExtNRows-1,iZero,
                                ACExtNRows-1, &IJnewA);
   ierr += HYPRE_IJMatrixSetObjectType(IJnewA, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if (ACExtNRows > 0) newRowSizes = new int[ACExtNRows];
   csrACExt = hypre_ParCSRMatrixDiag(hypreACExt);
   ACExtI = hypre_CSRMatrixI(csrACExt);
   ACExtJ = hypre_CSRMatrixJ(csrACExt);
   ACExtD = hypre_CSRMatrixData(csrACExt);
   for (iD = 0; iD < ACExtNRows; iD++)
      newRowSizes[iD] = ACExtI[iD+1] - ACExtI[iD];
   ierr  = HYPRE_IJMatrixSetRowSizes(IJnewA, newRowSizes);
   ierr += HYPRE_IJMatrixInitialize(IJnewA);
   hypre_assert(!ierr);
   for (iD = 0; iD < ACExtNRows; iD++)
   {
      offset = ACExtI[iD];
      HYPRE_IJMatrixSetValues(IJnewA, 1, &newRowSizes[iD], (const int *) &iD,
               (const int *) &ACExtJ[offset], (const double *) &ACExtD[offset]);
   }
   if (ACExtNRows > 0) delete [] newRowSizes;
   HYPRE_IJMatrixAssemble(IJnewA);
   HYPRE_IJMatrixGetObject(IJnewA, (void **) &hypreNewA);
   sprintf(paramString, "HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   delete mli_ACExt;
   mli_ACExt = new MLI_Matrix( (void *) hypreNewA, paramString, funcPtr);
   delete mli_AExt;
   HYPRE_IJMatrixSetObjectType(IJnewA, -1);
   HYPRE_IJMatrixDestroy(IJnewA);

   /* --------------------------------------------------------------- */
   /* create new overlapped DD smoother using sequential SuperLU      */
   /* --------------------------------------------------------------- */

   int        *sendLengs;
   char       *targv[7];
   MLI_Solver *smootherPtr;
   MLI_Matrix *mli_PSmat;

   if (!strcmp(preSmoother_, "CGMLI")) sprintf(paramString, "CGMLI");
   else                                sprintf(paramString, "CGAMG");
   smootherPtr = MLI_Solver_CreateFromName(paramString);
   sprintf(paramString, "numSweeps 10000");
   smootherPtr->setParams(paramString, 0, NULL);
   sprintf(paramString, "tolerance 1.0e-6");
   smootherPtr->setParams(paramString, 0, NULL);

   // send PSmat and communication information to smoother
   if (nSends > 0) sendLengs = new int[nSends];
   for (iP = 0; iP < nSends; iP++) sendLengs[iP] = ANRows;
   sprintf(paramString, "setPmat");
   mli_PSmat = mli_PExt;
   targv[0] = (char *) mli_PSmat;
   smootherPtr->setParams(paramString, 1, targv);
   sprintf(paramString, "setCommData");
   targv[0] = (char *) &nRecvs;
   targv[1] = (char *) recvProcs;
   targv[2] = (char *) recvLengs;
   targv[3] = (char *) &nSends;
   targv[4] = (char *) sendProcs;
   targv[5] = (char *) sendLengs;
   targv[6] = (char *) &comm;
   smootherPtr->setParams(paramString, 7, targv);
   if (nSends > 0) delete [] sendLengs;
   if (nRecvs > 0) delete [] recvLengs;

   smootherPtr->setup(mli_ACExt);
   mli->setSmoother(level, MLI_SMOOTHER_PRE, smootherPtr);

   /* --------------------------------------------------------------- */
   /* create prolongation and coarse grid operators                   */
   /* --------------------------------------------------------------- */

   MLI_Solver *csolvePtr;
   MLI_Matrix *mli_Pmat, *mli_Rmat, *mli_cAmat;

   // set up one aggregate per processor
   saCounts_[0] = 1;
   if (saData_[0] != NULL) delete [] saData_[0];
   saData_[0] = new int[ANRows];
   for (iD = 0; iD < ANRows; iD++) saData_[0][iD] = 0;

   // restore nullspace changed by the last genP
   currNodeDofs_ = nodeDofs;
   if (nullspaceVec_ != NULL) delete [] nullspaceVec_;
   nullspaceVec_ = new double[nullspaceDim_*ANRows];
   for (iD = 0; iD < nullspaceDim_*ANRows; iD++)
      nullspaceVec_[iD] = nullVecs[iD];
   delete [] nullVecs;

   // create prolongation and coarse grid operators
   genP(mli_Amat, &mli_Pmat, saCounts_[0], saData_[0]);
   MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
   mli->setSystemMatrix(level+1, mli_cAmat);
   mli->setProlongation(level+1, mli_Pmat);
   sprintf(paramString, "HYPRE_ParCSRT");
   mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), paramString, NULL);
   mli->setRestriction(level, mli_Rmat);

   /* --------------------------------------------------------------- */
   /* setup coarse solver                                             */
   /* --------------------------------------------------------------- */

   strcpy( paramString, "SuperLU" );
   csolvePtr = MLI_Solver_CreateFromName( paramString );
   csolvePtr->setup(mli_cAmat);
   mli->setCoarseSolve(csolvePtr);

   /* --------------------------------------------------------------- */
   /* clean up                                                        */
   /* --------------------------------------------------------------- */

   free(Apartition);

#ifdef MLI_DEBUG_DETAILED
   printf("%d : MLI_Method_AMGSA::setupExtendedDomainDecomp2 ends.\n",mypid);
#endif

   level = 2;
   return (level);
}

// ************************************************************************
// Purpose : Given Amat, perform preferential coarsening
// (setupExtendedDomainDecomp2)
// ------------------------------------------------------------------------

double MLI_Method_AMGSA::genP_AExt(MLI_Matrix *mli_Amat,MLI_Matrix **PmatOut,
                                   int inANRows)
{
   int    mypid, nprocs, *partition, AStartRow, AEndRow, ALocalNRows;
   int    blkSize, naggr, *node2aggr, ierr, PLocalNCols, PStartCol;
   int    PLocalNRows, PStartRow, *eqn2aggr, irow, jcol, ig;
   int    *PCols, maxAggSize, *aggCntArray, index, **aggIndArray;
   int    aggSize, nzcnt, *rowLengths, rowNum, *colInd;
   double **PVecs, *newNull, *qArray, *rArray, *colVal;
   char   paramString[200];
   HYPRE_IJMatrix      IJPmat;
   hypre_ParCSRMatrix  *Amat, *A2mat, *Pmat;
   MLI_Matrix          *mli_A2mat=NULL, *mli_Pmat;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg *commPkg;
   MLI_Function        *funcPtr;

   /*-----------------------------------------------------------------
    * fetch matrix and machine information
    *-----------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);

   /*-----------------------------------------------------------------
    * fetch other matrix information
    *-----------------------------------------------------------------*/

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   AStartRow = partition[mypid];
   AEndRow   = partition[mypid+1] - 1;
   ALocalNRows = AEndRow - AStartRow + 1;
   free(partition);

   /*-----------------------------------------------------------------
    * reduce Amat based on the block size information (if nodeDofs_ > 1)
    *-----------------------------------------------------------------*/

   blkSize = currNodeDofs_;
   if (blkSize > 1) MLI_Matrix_Compress(mli_Amat, blkSize, &mli_A2mat);
   else             mli_A2mat = mli_Amat;
   A2mat = (hypre_ParCSRMatrix *) mli_A2mat->getMatrix();

   /*-----------------------------------------------------------------
    * modify minimum aggregate size, if needed
    *-----------------------------------------------------------------*/

   minAggrSize_ = nullspaceDim_ / currNodeDofs_;
   if (minAggrSize_ <= 1) minAggrSize_ = 2;

   /*-----------------------------------------------------------------
    * perform coarsening
    *-----------------------------------------------------------------*/

   coarsenAExt(A2mat, &naggr, &node2aggr, inANRows);
   if (blkSize > 1 && mli_A2mat != NULL) delete mli_A2mat;

   /*-----------------------------------------------------------------
    * fetch the coarse grid information and instantiate P
    *-----------------------------------------------------------------*/

   PLocalNCols = naggr * nullspaceDim_;
   MLI_Utils_GenPartition(comm, PLocalNCols, &partition);
   PStartCol = partition[mypid];
   free(partition);
   PLocalNRows = ALocalNRows;
   PStartRow   = AStartRow;
   ierr = HYPRE_IJMatrixCreate(comm,PStartRow,PStartRow+PLocalNRows-1,
                          PStartCol,PStartCol+PLocalNCols-1,&IJPmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJPmat, HYPRE_PARCSR);
   hypre_assert(!ierr);

   /*-----------------------------------------------------------------
    * expand the aggregation information if block size > 1 ==> eqn2aggr
    *-----------------------------------------------------------------*/

   if (blkSize > 1)
   {
      eqn2aggr = new int[ALocalNRows];
      for (irow = 0; irow < ALocalNRows; irow++)
         eqn2aggr[irow] = node2aggr[irow/blkSize];
      delete [] node2aggr;
   }
   else eqn2aggr = node2aggr;

   /*-----------------------------------------------------------------
    * create a compact form for the null space vectors
    * (get ready to perform QR on them)
    *-----------------------------------------------------------------*/

   PVecs = new double*[nullspaceDim_];
   PCols = new int[PLocalNRows];
   for (irow = 0; irow < nullspaceDim_; irow++)
      PVecs[irow] = new double[PLocalNRows];
   for (irow = 0; irow < PLocalNRows; irow++)
   {
      if (eqn2aggr[irow] >= 0)
           PCols[irow] = PStartCol + eqn2aggr[irow] * nullspaceDim_;
      else PCols[irow] = PStartCol + (-eqn2aggr[irow]-1) * nullspaceDim_;
      if (nullspaceVec_ != NULL)
      {
         for (jcol = 0; jcol < nullspaceDim_; jcol++)
            PVecs[jcol][irow] = nullspaceVec_[jcol*PLocalNRows+irow];
      }
      else
      {
         for (jcol = 0; jcol < nullspaceDim_; jcol++)
         {
            if (irow % nullspaceDim_ == jcol) PVecs[jcol][irow] = 1.0;
            else                              PVecs[jcol][irow] = 0.0;
         }
      }
   }

   /*-----------------------------------------------------------------
    * perform QR for null space
    *-----------------------------------------------------------------*/

   newNull = NULL;
   if (PLocalNRows > 0)
   {
      /* ------ count the size of each aggregate ------ */

      aggCntArray = new int[naggr];
      for (ig = 0; ig < naggr; ig++) aggCntArray[ig] = 0;
      for (irow = 0; irow < PLocalNRows; irow++)
         if (eqn2aggr[irow] >= 0) aggCntArray[eqn2aggr[irow]]++;
         else                     aggCntArray[(-eqn2aggr[irow]-1)]++;
      maxAggSize = 0;
      for (ig = 0; ig < naggr; ig++)
         if (aggCntArray[ig] > maxAggSize) maxAggSize = aggCntArray[ig];

      /* ------ register which equation is in which aggregate ------ */

      aggIndArray = new int*[naggr];
      for (ig = 0; ig < naggr; ig++)
      {
         aggIndArray[ig] = new int[aggCntArray[ig]];
         aggCntArray[ig] = 0;
      }
      for (irow = 0; irow < PLocalNRows; irow++)
      {
         index = eqn2aggr[irow];
         if (index >= 0)
            aggIndArray[index][aggCntArray[index]++] = irow;
         else
            aggIndArray[-index-1][aggCntArray[-index-1]++] = irow;
      }

      /* ------ allocate storage for QR factorization ------ */

      qArray  = new double[maxAggSize * nullspaceDim_];
      rArray  = new double[nullspaceDim_ * nullspaceDim_];
      newNull = new double[naggr*nullspaceDim_*nullspaceDim_];

      /* ------ perform QR on each aggregate ------ */

      for (ig = 0; ig < naggr; ig++)
      {
         aggSize = aggCntArray[ig];

         if (aggSize < nullspaceDim_)
         {
            printf("Aggregation ERROR : underdetermined system in QR.\n");
            printf("            error on Proc %d\n", mypid);
            printf("            error on aggr %d (%d)\n", ig, naggr);
            printf("            aggr size is %d\n", aggSize);
            exit(1);
         }

         /* ------ put data into the temporary array ------ */

         for (jcol = 0; jcol < aggSize; jcol++)
         {
            for (irow = 0; irow < nullspaceDim_; irow++)
               qArray[aggSize*irow+jcol] = PVecs[irow][aggIndArray[ig][jcol]];
         }

         /* ------ after QR, put the R into the next null space ------ */

         for (jcol = 0; jcol < nullspaceDim_; jcol++)
            for (irow = 0; irow < nullspaceDim_; irow++)
               if (irow == jcol)
                  newNull[ig*nullspaceDim_+jcol+irow*naggr*nullspaceDim_] = 1.0;
               else
                  newNull[ig*nullspaceDim_+jcol+irow*naggr*nullspaceDim_] = 0.0;

         /* ------ put the P to PVecs ------ */

         for (jcol = 0; jcol < aggSize; jcol++)
         {
            for (irow = 0; irow < nullspaceDim_; irow++)
            {
               index = aggIndArray[ig][jcol];
               PVecs[irow][index] = qArray[ irow*aggSize + jcol ];
            }
         }
      }
      for (ig = 0; ig < naggr; ig++) delete [] aggIndArray[ig];
      delete [] aggIndArray;
      delete [] aggCntArray;
      delete [] qArray;
      delete [] rArray;
   }
   if (nullspaceVec_ != NULL) delete [] nullspaceVec_;
   nullspaceVec_ = newNull;

   /*-----------------------------------------------------------------
    * initialize Pmat
    *-----------------------------------------------------------------*/

   rowLengths = new int[PLocalNRows];
   for (irow = 0; irow < PLocalNRows; irow++)
      rowLengths[irow] = nullspaceDim_;
   ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, rowLengths);
   ierr = HYPRE_IJMatrixInitialize(IJPmat);
   hypre_assert(!ierr);
   delete [] rowLengths;

   /*--------------------------------------------------------------------
    * load and assemble Pmat
    *--------------------------------------------------------------------*/

   colInd = new int[nullspaceDim_];
   colVal = new double[nullspaceDim_];
   for (irow = 0; irow < PLocalNRows; irow++)
   {
      if (PCols[irow] >= 0)
      {
         nzcnt = 0;
         for (jcol = 0; jcol < nullspaceDim_; jcol++)
         {
            if (PVecs[jcol][irow] != 0.0)
            {
               colInd[nzcnt] = PCols[irow] + jcol;
               colVal[nzcnt++] = PVecs[jcol][irow];
            }
         }
         rowNum = PStartRow + irow;
         HYPRE_IJMatrixSetValues(IJPmat, 1, &nzcnt,
                             (const int *) &rowNum, (const int *) colInd,
                             (const double *) colVal);
      }
   }
   ierr = HYPRE_IJMatrixAssemble(IJPmat);
   hypre_assert(!ierr);
   HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Pmat);
   commPkg = hypre_ParCSRMatrixCommPkg(Amat);
   if (!commPkg) hypre_MatvecCommPkgCreate(Amat);
   HYPRE_IJMatrixSetObjectType(IJPmat, -1);
   HYPRE_IJMatrixDestroy(IJPmat);
   delete [] colInd;
   delete [] colVal;

   /*-----------------------------------------------------------------
    * clean up
    *-----------------------------------------------------------------*/

   if (PCols != NULL) delete [] PCols;
   if (PVecs != NULL)
   {
      for (irow = 0; irow < nullspaceDim_; irow++)
         if (PVecs[irow] != NULL) delete [] PVecs[irow];
      delete [] PVecs;
   }

   /*-----------------------------------------------------------------
    * set up and return Pmat
    *-----------------------------------------------------------------*/

   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   mli_Pmat = new MLI_Matrix( Pmat, paramString, funcPtr );
   (*PmatOut) = mli_Pmat;
   delete funcPtr;
   return 0.0;
}

// *********************************************************************
// graded coarsening scheme (Given a graph, aggregate on the local subgraph
// but give smaller aggregate near processor boundaries)
// (called by setupExtendedDomainDecomp2/genP_AExt)
// ---------------------------------------------------------------------

int MLI_Method_AMGSA::coarsenAExt(hypre_ParCSRMatrix *hypreG,
               int *mliAggrLeng, int **mliAggrArray, int inANRows)
{
   MPI_Comm  comm;
   int       mypid, nprocs, *partition, startRow, endRow, maxInd;
   int       localNRows, naggr=0, *node2aggr, *aggrSizes;
   int       irow, jcol, rowLeng, globalNRows, index;
   int       *nodeStat, selectFlag, nSelected=0, nNotSelected=0, count;
   int       *GDiagI, *GDiagJ;
   double    maxVal, *GDiagA;
   hypre_CSRMatrix *GDiag;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(hypreG);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreG,
                                        &partition);
   startRow = partition[mypid];
   endRow   = partition[mypid+1] - 1;
   free(partition);
   localNRows = endRow - startRow + 1;
   MPI_Allreduce(&localNRows, &globalNRows, 1, MPI_INT, MPI_SUM, comm);
   if (mypid == 0 && outputLevel_ > 1)
   {
      printf("\t*** Aggregation(E) : total nodes to aggregate = %d\n",
             globalNRows);
   }
   GDiag  = hypre_ParCSRMatrixDiag(hypreG);
   GDiagI = hypre_CSRMatrixI(GDiag);
   GDiagJ = hypre_CSRMatrixJ(GDiag);
   GDiagA = hypre_CSRMatrixData(GDiag);

   /*-----------------------------------------------------------------
    * allocate status arrays
    *-----------------------------------------------------------------*/

   if (localNRows > 0)
   {
      node2aggr = new int[localNRows];
      aggrSizes = new int[localNRows];
      nodeStat  = new int[localNRows];
      for (irow = 0; irow < inANRows; irow++)
      {
         aggrSizes[irow] = 1;
         node2aggr[irow] = -1;
         nodeStat[irow]  = MLI_METHOD_AMGSA_SELECTED;
      }
      for (irow = inANRows; irow < localNRows; irow++)
      {
         aggrSizes[irow] = 0;
         node2aggr[irow] = -1;
         nodeStat[irow]  = MLI_METHOD_AMGSA_READY;
      }
      nSelected = inANRows;
   }
   else node2aggr = aggrSizes = nodeStat = NULL;

   /*-----------------------------------------------------------------
    * search for zero rows and rows near the processor boundaries
    *-----------------------------------------------------------------*/

   for (irow = inANRows; irow < localNRows; irow++)
   {
      rowLeng = GDiagI[irow+1] - GDiagI[irow];
      if (rowLeng <= 0)
      {
         nodeStat[irow] = MLI_METHOD_AMGSA_NOTSELECTED;
         nNotSelected++;
      }
   }

   /*-----------------------------------------------------------------
    * Phase 0 : 1 node per aggregate for the immediate neighbors
    *-----------------------------------------------------------------*/

   for (irow = 0; irow < inANRows; irow++)
   {
      for (jcol = GDiagI[irow]; jcol < GDiagI[irow+1]; jcol++)
      {
         index = GDiagJ[jcol];
         if (index >= inANRows && nodeStat[index]==MLI_METHOD_AMGSA_READY)
         {
            node2aggr[index]  = naggr;
            nodeStat[index] = MLI_METHOD_AMGSA_SELECTED;
            aggrSizes[naggr] = 1;
            naggr++;
            nSelected++;
         }
      }
   }

   /*-----------------------------------------------------------------
    * Phase 1 : small aggregates for the next level
    *-----------------------------------------------------------------*/

   for (irow = inANRows; irow < localNRows; irow++)
   {
      if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
      {
         selectFlag = 0;
         for (jcol = GDiagI[irow]; jcol < GDiagI[irow+1]; jcol++)
         {
            index = GDiagJ[jcol];
            if (nodeStat[index] == MLI_METHOD_AMGSA_SELECTED)
            {
               selectFlag = 1;
               break;
            }
         }
         if (selectFlag == 1)
         {
            nSelected++;
            node2aggr[irow]  = naggr;
            aggrSizes[naggr] = 1;
            nodeStat[irow]  = MLI_METHOD_AMGSA_SELECTED2;
            for (jcol = GDiagI[irow]; jcol < GDiagI[irow+1]; jcol++)
            {
               index = GDiagJ[jcol];
               if (nodeStat[irow]==MLI_METHOD_AMGSA_READY)
               {
                  nSelected++;
                  node2aggr[index] = naggr;
                  aggrSizes[naggr]++;
                  nodeStat[index]  = MLI_METHOD_AMGSA_SELECTED2;
               }
            }
            naggr++;
         }
      }
   }
   for (irow = inANRows; irow < localNRows; irow++)
      if (nodeStat[index] == MLI_METHOD_AMGSA_SELECTED2)
         nodeStat[index] = MLI_METHOD_AMGSA_SELECTED;

   /*-----------------------------------------------------------------
    * Phase 1 : form aggregates
    *-----------------------------------------------------------------*/

   for (irow = inANRows; irow < localNRows; irow++)
   {
      if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
      {
         selectFlag = 1;
         for (jcol = GDiagI[irow]; jcol < GDiagI[irow+1]; jcol++)
         {
            index = GDiagJ[jcol];
            if (nodeStat[index] != MLI_METHOD_AMGSA_READY)
            {
               selectFlag = 0;
               break;
            }
         }
         if (selectFlag == 1)
         {
            aggrSizes[naggr] = 1;
            node2aggr[irow] = naggr;
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nSelected++;
            for (jcol = GDiagI[irow]; jcol < GDiagI[irow+1]; jcol++)
            {
               index = GDiagJ[jcol];
               node2aggr[index] = naggr;
               nodeStat[index] = MLI_METHOD_AMGSA_SELECTED;
               aggrSizes[naggr]++;
               nSelected++;
            }
            naggr++;
         }
      }
   }

   /*-----------------------------------------------------------------
    * Phase 2 : put the rest into one of the existing aggregates
    *-----------------------------------------------------------------*/

   if ((nSelected+nNotSelected) < localNRows)
   {
      for (irow = inANRows; irow < localNRows; irow++)
      {
         if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
         {
            maxInd  = -1;
            maxVal  = 0.0;
            for (jcol = GDiagI[irow]; jcol < GDiagI[irow+1]; jcol++)
            {
               index = GDiagJ[jcol];
               if (nodeStat[index] == MLI_METHOD_AMGSA_SELECTED)
               {
                  if (GDiagA[jcol] > maxVal)
                  {
                     maxInd = jcol;
                     maxVal = GDiagA[jcol];
                  }
               }
            }
            if (maxInd != -1)
            {
               node2aggr[irow] = node2aggr[maxInd];
               nodeStat[irow] = MLI_METHOD_AMGSA_PENDING;
               aggrSizes[node2aggr[maxInd]]++;
            }
         }
      }
      for (irow = inANRows; irow < localNRows; irow++)
      {
         if (nodeStat[irow] == MLI_METHOD_AMGSA_PENDING)
         {
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nSelected++;
         }
      }
   }

   /*-----------------------------------------------------------------
    * Phase 4 : finally put all lone rows into some neighbor aggregate
    *-----------------------------------------------------------------*/

   int nUndone = localNRows - nSelected - nNotSelected;
   if (nUndone > 0)
   {
      count = nUndone / minAggrSize_;
      if (count == 0) count = 1;
      count += naggr;
      irow = jcol = 0;
      while (nUndone > 0)
      {
         if (nodeStat[irow] == MLI_METHOD_AMGSA_READY)
         {
            node2aggr[irow] = naggr;
            nodeStat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nUndone--;
            nSelected++;
            jcol++;
            if (jcol >= minAggrSize_ && naggr < count-1)
            {
               jcol = 0;
               naggr++;
            }
         }
         irow++;
      }
      naggr = count;
   }

   /*-----------------------------------------------------------------
    * clean up and initialize the output arrays
    *-----------------------------------------------------------------*/

   if (localNRows > 0) delete [] aggrSizes;
   if (localNRows > 0) delete [] nodeStat;
   if (localNRows == 1 && naggr == 0)
   {
      node2aggr[0] = 0;
      naggr = 1;
   }
   (*mliAggrArray) = node2aggr;
   (*mliAggrLeng)  = naggr;
   return 0;
}

