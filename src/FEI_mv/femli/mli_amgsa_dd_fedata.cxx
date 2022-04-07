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

// *********************************************************************
// HYPRE includes external to MLI
// ---------------------------------------------------------------------

#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"

// *********************************************************************
// local MLI includes
// ---------------------------------------------------------------------

#include "mli_method_amgsa.h"
#include "mli_utils.h"
#include "mli_mapper.h"
#include "mli_fedata.h"
#include "mli_fedata_utils.h"
#include "mli_matrix.h"
 
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
 ***********************************************************************
 * compute initial null spaces (for the subdomain only) using FEData
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupFEDataBasedNullSpaces( MLI *mli ) 
{
   int          i, j, k, jj, k1, level, mypid, nElems, nodeNumFields;
   int          nodeFieldID, elemNNodes, **elemNodeLists, *elemNodeList1D;
   int          blockSize, *nodeEqnList, eMatDim, *partition, localStartRow;
   int          localNRows, rowInd1, rowInd2, colInd1, colInd2;
   int          colOffset1, colOffset2, elemID, *elemIDs, rowSize;
   int          csrNrows, *csrIA, *csrJA, totalNNodes, sInd1, sInd2, offset;
   int          *newNodeEqnList, *newElemNodeList, *orderArray, newNNodes;
   double       *elemMat, *csrAA;
   double       *eigenR, *eigenI, *eigenV;
   char         which[20];
   char         *targv[1];
   MPI_Comm     comm;
   MLI_FEData   *fedata;
   MLI_Mapper   *nodeEqnMap;
   MLI_Matrix   *mliAmat;
   hypre_ParCSRMatrix *hypreA;
#ifdef MLI_ARPACK
   int          info;
   double       sigmaR, sigmaI;
#endif

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupFEDataBasedNullSpaces begins.\n");
#endif

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if ( mli == NULL )
   {
      printf("MLI_Method_AMGSA::setupFEDataBasedNullSpaces ERROR");
      printf(" - no mli.\n");
      exit(1);
   }
   level = 0;
   fedata = mli->getFEData(level);
   if ( fedata == NULL )
   {
      printf("MLI_Method_AMGSA::setupFEDataBasedNullSpaces ERROR");
      printf(" - no fedata.\n");
      exit(1);
   }

   /* --------------------------------------------------------------- */
   /* fetch communicator matrix information                           */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   mliAmat = mli->getSystemMatrix( level );
   hypreA  = (hypre_ParCSRMatrix *) mliAmat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   localStartRow = partition[mypid];
   localNRows    = partition[mypid+1] - localStartRow;
   free( partition );

   /* --------------------------------------------------------------- */
   /* fetch FEData information (nElems, elemIDs, elemNNodes,          */
   /*   elemNodeLists, blockSize)                                     */
   /* --------------------------------------------------------------- */

   fedata->getNodeNumFields(nodeNumFields);
   if ( nodeNumFields != 1 ) 
   {
      printf("MLI_Method_AMGSA::setupFEDataBasedNullSpaces - ");
      printf("nodeNumFields != 1.\n");
      return 1;
   }
   fedata->getNumElements( nElems );
   if ( nElems <= 0 ) return 0;
   elemIDs = new int[nElems];
   fedata->getElemBlockGlobalIDs(nElems, elemIDs);
   fedata->getElemNumNodes(elemNNodes);
   totalNNodes = nElems * elemNNodes;
   elemNodeList1D = new int[totalNNodes];
   elemNodeLists = new int*[nElems];
   for ( i = 0; i < nElems; i++ )
      elemNodeLists[i] = &(elemNodeList1D[i*elemNNodes]);
   fedata->getElemBlockNodeLists(nElems, elemNNodes, elemNodeLists);
   fedata->getNodeFieldIDs(nodeNumFields, &nodeFieldID);
   fedata->getFieldSize(nodeFieldID, blockSize);

   /* --------------------------------------------------------------- */
   /* find the number of nodes in local subdomain (including external */
   /* nodes) and assign a local equation number to them               */
   /* output : newElemNodeList, newNodeEqnList, newNNodes             */
   /* --------------------------------------------------------------- */

   newNodeEqnList  = new int[totalNNodes];
   newElemNodeList = new int[totalNNodes];
   orderArray      = new int[totalNNodes];
   for (i = 0; i < totalNNodes; i++) 
   {
      orderArray[i] = i;
      newElemNodeList[i] = elemNodeList1D[i];
   }
   MLI_Utils_IntQSort2(newElemNodeList, orderArray, 0, totalNNodes-1);
   newNNodes = 1;
   newNodeEqnList[orderArray[0]] = (newNNodes - 1) * blockSize;
   for ( i = 1; i < totalNNodes; i++ )
   {
      if (newElemNodeList[i] != newElemNodeList[newNNodes-1]) 
         newElemNodeList[newNNodes++] = newElemNodeList[i];
      newNodeEqnList[orderArray[i]] = (newNNodes - 1) * blockSize;
   }
   delete [] newElemNodeList;

   /* --------------------------------------------------------------- */
   /* allocate and initialize subdomain matrix                        */
   /* --------------------------------------------------------------- */

   eMatDim  = elemNNodes * blockSize;
   elemMat  = new double[eMatDim*eMatDim];
   rowSize  = elemNNodes * blockSize * 8;
   csrNrows = newNNodes * blockSize;
   csrIA    = new int[csrNrows+1];
   csrJA    = new int[csrNrows*rowSize];
   hypre_assert( ((long) csrJA) );
   csrAA    = new double[csrNrows*rowSize];
   hypre_assert( ((long) csrAA) );
   csrIA[0] = 0;
   for ( i = 1; i < csrNrows; i++ ) csrIA[i] = csrIA[i-1] + rowSize;

   /* --------------------------------------------------------------- */
   /* construct CSR matrix (with holes)                               */
   /* --------------------------------------------------------------- */

   strcpy( which, "destroyElemMatrix" );
   targv[0] = (char *) &elemID;
   for ( i = 0; i < nElems; i++ )
   {
      elemID = elemIDs[i];
      fedata->getElemMatrix(elemID, eMatDim, elemMat); 
      fedata->impSpecificRequests(which, 1, targv); 
      for ( j = 0; j < elemNNodes; j++ )
      {
         colInd1 = newNodeEqnList[i*elemNNodes+j];
         colInd2 = j * blockSize;
         for ( jj = 0; jj < blockSize; jj++ )
         {
            colOffset1 = colInd1 + jj;
            colOffset2 = eMatDim * ( colInd2 + jj );
            for ( k = 0; k < elemNNodes; k++ )
            {
               rowInd1 = newNodeEqnList[i*elemNNodes+k];
               rowInd2 = k * blockSize;
               for ( k1 = 0; k1 < blockSize; k1++ )
               {
                  if ( elemMat[rowInd2+k1+colOffset2] != 0.0 )
                  {
                     offset = csrIA[rowInd1+k1]++;
                     csrJA[offset] = colOffset1;
                     csrAA[offset] = elemMat[rowInd2+k1+colOffset2];
                  }
               }
            }
         }
      }
   }
   delete [] elemMat;

   /* --------------------------------------------------------------- */
   /* compress the CSR matrix                                         */
   /* --------------------------------------------------------------- */

   jj = 0;
   csrIA[csrNrows] = 0;
   for ( i = 0; i <= csrNrows; i++ )
   {
      if ( csrIA[i] > rowSize * (i+1) )
      {
         printf("MLI_Method_AMGSA::setupFEDataBasedNullSpaces ");
         printf("ERROR : rowSize too large (increase it). \n");
         printf("   => allowed = %d, actual = %d\n",rowSize, 
                csrIA[i]-rowSize*i);
         exit(1);
      }
      if ( i < csrNrows )
      {
         k1 = csrIA[i] - i * rowSize;
         sInd1 = i * rowSize;
         sInd2 = i * rowSize + k1 - 1;
         MLI_Utils_IntQSort2a(&(csrJA[sInd1]),&(csrAA[sInd1]),0,sInd2-sInd1);
         k1 = sInd1;
         for ( j = sInd1+1; j <= sInd2; j++ )
         {
            if ( csrJA[j] == csrJA[k1] ) csrAA[k1] += csrAA[j]; 
            else
            {
               k1++;
               csrJA[k1] = csrJA[j];
               csrAA[k1] = csrAA[j];
            }
         }
         if ( sInd2 >= sInd1 ) k1 = k1 - sInd1 + 1;
         else                  k1 = 0;
         sInd2 = sInd1 + k1;
         k1 = sInd1;
         for ( j = sInd1; j < sInd2; j++ )
         {
            csrJA[k1] = csrJA[j];
            csrAA[k1++] = csrAA[j];
         }
         k1 = k1 - sInd1;
         for ( j = jj; j < jj+k1; j++ )
         {
            csrJA[j] = csrJA[sInd1+j-jj];
            csrAA[j] = csrAA[sInd1+j-jj];
         }
      }
      csrIA[i] = jj;
      jj += k1;
   }

#if 0
   FILE *fp;
   fp = fopen("SMat", "w");
   for ( i = 0; i < csrNrows; i++ )
      for ( j = csrIA[i]; j < csrIA[i+1]; j++ )
         fprintf(fp, "%5d %5d %25.16e \n", i+1, csrJA[j]+1, csrAA[j]);
   fclose( fp );
#endif

   /* --------------------------------------------------------------- */
   /* change from base-0 to base-1 indexing for Fortran call          */
   /* --------------------------------------------------------------- */

   for ( i = 0; i < csrIA[csrNrows]; i++ ) csrJA[i]++;
   for ( i = 0; i <= csrNrows; i++ ) csrIA[i]++;

   /* --------------------------------------------------------------- */
   /* compute near null spaces                                        */
   /* --------------------------------------------------------------- */

   strcpy( which, "shift" );
   eigenR = new double[nullspaceDim_+1];
   eigenI = new double[nullspaceDim_+1];
   eigenV = new double[csrNrows*(nullspaceDim_+1)];
   hypre_assert((long) eigenV);

#ifdef MLI_ARPACK
   sigmaR = 1.0e-5;
   sigmaI = 0.0e-1;
   dnstev_(&csrNrows, &nullspaceDim_, which, &sigmaR, &sigmaI, 
           csrIA, csrJA, csrAA, eigenR, eigenI, eigenV, &csrNrows, &info,
           &arpackTol_);
#else
   printf("MLI_Method_AMGSA::FATAL ERROR : ARPACK not installed.\n");
   exit(1);
#endif

   if ( useSAMGDDFlag_ ) ARPACKSuperLUExists_ = 1; 
   else 
   {
      strcpy( which, "destroy" );
#ifdef MLI_ARPACK
      dnstev_(&csrNrows, &nullspaceDim_, which, &sigmaR, &sigmaI, 
              csrIA, csrJA, csrAA, eigenR, eigenI, eigenV, &csrNrows, &info,
              &arpackTol_);
#else
   printf("MLI_Method_AMGSA::FATAL ERROR : ARPACK not installed.\n");
   exit(1);
#endif
   }
#if 1
   for ( i = 0; i < nullspaceDim_; i++ )
      printf("%5d : eigenvalue %5d = %e\n", mypid, i, eigenR[i]);
#endif

   delete [] eigenR;
   delete [] eigenI;
   delete [] csrIA;
   delete [] csrJA;
   delete [] csrAA;

   /* --------------------------------------------------------------- */
   /* get node to equation map                                        */
   /* --------------------------------------------------------------- */

   nodeEqnMap = mli->getNodeEqnMap(level);
   if ( nodeEqnMap != NULL ) 
   {
      nodeEqnList = new int[totalNNodes];
      nodeEqnMap->getMap(totalNNodes,elemNodeList1D,nodeEqnList);
   }
   else
   {
      nodeEqnList = new int[totalNNodes];
      for ( i = 0; i < nElems; i++ )
         for ( j = 0; j < elemNNodes; j++ )
            nodeEqnList[i*elemNNodes+j] = elemNodeLists[i][j] * blockSize;
   }

   /* --------------------------------------------------------------- */
   /* load the null space vectors                                     */
   /* --------------------------------------------------------------- */

   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   nullspaceLen_ = localNRows;
   nullspaceVec_ = new double[nullspaceLen_*nullspaceDim_];

   for ( i = 0; i < totalNNodes; i++ )
   {
      jj = orderArray[i]; 
      k1 = elemNodeList1D[jj];   /* get node number */
      rowInd1 = nodeEqnList[jj]; /* get equation number for the node */
      rowInd1 -= localStartRow;
      if ( rowInd1 >= 0 && rowInd1 < localNRows )
      {
         rowInd2 = newNodeEqnList[jj];
         for ( j = 0; j < blockSize; j++ )
            for ( k = 0; k < nullspaceDim_; k++ )
               nullspaceVec_[rowInd1+j+k*nullspaceLen_] = 
                  eigenV[rowInd2+j+k*csrNrows];
      }
   } 

#if 0
   FILE *fp;
   char param_string[80];
   sprintf(param_string, "ANull.%d", mypid);
   fp = fopen( param_string, "w" );
   for ( i = 0; i < localNRows; i++ ) 
   {
      for ( j = 0; j < nullspaceDim_; j++ ) 
         fprintf(fp, "%25.16e ", nullspaceVec_[localNRows*j+i]);
      fprintf(fp, "\n");
   }
   fclose(fp);
   MPI_Barrier(comm);
   exit(1);
#endif

   /* --------------------------------------------------------------- */
   /* clean up                                                        */
   /* --------------------------------------------------------------- */

   delete [] orderArray;
   delete [] newNodeEqnList;
   delete [] eigenV;
   delete [] elemNodeList1D;
   delete [] elemNodeLists;
   delete [] elemIDs;
   delete [] nodeEqnList;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupFEDataBasedNullSpaces ends.\n");
#endif

   return 0;
}

/***********************************************************************
 * set up domain decomposition method by having each subdomain with
 * the same aggregate number 0
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupFEDataBasedAggregates( MLI *mli ) 
{
   int                i, level, mypid, *partition, localNRows, *aggrMap;
   int                nprocs;
   MPI_Comm           comm;
   MLI_Matrix         *mliAmat;
   hypre_ParCSRMatrix *hypreA;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupFEDataBasedAggregates begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* fetch communicator matrix information                           */
   /* --------------------------------------------------------------- */

   level = 0;
   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   mliAmat = mli->getSystemMatrix( level );
   hypreA  = (hypre_ParCSRMatrix *) mliAmat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   localNRows = partition[mypid+1] - partition[mypid];
   free( partition );

   aggrMap = new int[localNRows];
   for ( i = 0; i < localNRows; i++ ) aggrMap[i] = 0;
   saData_[0]     = aggrMap;
   saCounts_[0]   = 1;
   numLevels_     = 2;
   minCoarseSize_ = nprocs;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupFEDataBasedAggregates ends.\n");
#endif

   return 0;
}

/***********************************************************************
 * set up domain decomposition on each subdomain as smoother
 * using the SuperLU factors already generated here.
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupFEDataBasedSuperLUSmoother(MLI *mli, int level) 
{
   int         mypid, nprocs, localStartRow, localEndRow;
   int         *partition, nodeNumFields, nElems, *elemIDs, elemNNodes;
   int         totalNNodes, **elemNodeLists, *elemNodeList1D;
   int         nodeFieldID, blockSize, *newNodeEqnList, *newElemNodeList;
   int         *orderArray, newNNodes, *nodeEqnList;
   int         *iTempArray, iE, iN, iP, jN, index;
   int         *procArray, globalEqnNum, nSendMap, *sendMap;
   int         nRecvs, *recvLengs, *recvProcs, **iRecvBufs;
   int         nSends, *sendLengs, *sendProcs, **iSendBufs;
   MPI_Comm    comm;
   MPI_Request *requests;
   MPI_Status  *statuses;
   MLI_Mapper  *nodeEqnMap;
   MLI_FEData  *fedata;
   MLI_Matrix  *mliAmat;
   hypre_ParCSRMatrix *hypreA;

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if ( mli == NULL )
   {
      printf("MLI_Method_AMGSA::setupFEDataBasedSuperLUSmoother ERROR - ");
      printf("no mli\n");
      exit(1);
   }
   fedata = mli->getFEData(level);
   if ( fedata == NULL )
   {
      printf("MLI_Method_AMGSA::setupFEDataBasedSuperLUSmoother ERROR - ");
      printf("no fedata\n");
      exit(1);
   }

   /* --------------------------------------------------------------- */
   /* fetch communicator matrix information                           */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   mliAmat = mli->getSystemMatrix( level );
   hypreA  = (hypre_ParCSRMatrix *) mliAmat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   localStartRow = partition[mypid];
   localEndRow   = partition[mypid+1] - 1;
   free( partition );

   /* --------------------------------------------------------------- */
   /* fetch FEData information (nElems, elemIDs, elemNNodes,          */
   /*   elemNodeLists, blockSize)                                     */
   /* --------------------------------------------------------------- */

   fedata->getNodeNumFields(nodeNumFields);
   if ( nodeNumFields != 1 ) 
   {
      printf("MLI_Method_AMGSA::setupFEDataBasedSuperLUSmoother - ");
      printf("nodeNumFields!=1.\n");
      return 1;
   }
   fedata->getNumElements( nElems );
   if ( nElems <= 0 ) return 0;
   elemIDs = new int[nElems];
   fedata->getElemBlockGlobalIDs(nElems, elemIDs);
   fedata->getElemNumNodes(elemNNodes);
   totalNNodes = nElems * elemNNodes;
   elemNodeList1D = new int[totalNNodes];
   elemNodeLists = new int*[nElems];
   for ( iE = 0; iE < nElems; iE++ )
      elemNodeLists[iE] = &(elemNodeList1D[iE*elemNNodes]);
   fedata->getElemBlockNodeLists(nElems, elemNNodes, elemNodeLists);
   fedata->getNodeFieldIDs(nodeNumFields, &nodeFieldID);
   fedata->getFieldSize(nodeFieldID, blockSize);

   /* --------------------------------------------------------------- */
   /* find the number of nodes in local subdomain (including external */
   /* nodes) and assign a local equation number to them               */
   /* output : newElemNodeList, newNodeEqnList, newNNodes             */
   /* --------------------------------------------------------------- */

   newNodeEqnList  = new int[totalNNodes];
   newElemNodeList = new int[totalNNodes];
   orderArray      = new int[totalNNodes];
   for ( iN = 0; iN < totalNNodes; iN++ ) 
   {
      orderArray[iN] = iN;
      newElemNodeList[iN] = elemNodeList1D[iN];
   }
   MLI_Utils_IntQSort2(newElemNodeList, orderArray, 0, totalNNodes-1);
   newNNodes = 1;
   newNodeEqnList[orderArray[0]] = (newNNodes - 1) * blockSize;
   for ( iN = 1; iN < totalNNodes; iN++ )
   {
      if (newElemNodeList[iN] != newElemNodeList[newNNodes-1]) 
         newElemNodeList[newNNodes++] = newElemNodeList[iN];
      newNodeEqnList[orderArray[iN]] = (newNNodes - 1) * blockSize;
   }
   delete [] newElemNodeList;

   /* --------------------------------------------------------------- */
   /* get node to equation map                                        */
   /* --------------------------------------------------------------- */

   nodeEqnMap = mli->getNodeEqnMap(level);
   if ( nodeEqnMap != NULL ) 
   {
      nodeEqnList = new int[totalNNodes];
      nodeEqnMap->getMap(totalNNodes,elemNodeList1D,nodeEqnList);
   }
   else
   {
      nodeEqnList = new int[totalNNodes];
      for ( iE = 0; iE < nElems; iE++ )
         for ( jN = 0; jN < elemNNodes; jN++ )
            nodeEqnList[iE*elemNNodes+jN] = elemNodeLists[iE][jN] * blockSize;
   }

   /* --------------------------------------------------------------- */
   /* form incoming A-ordered equation to local-ordered equation map  */
   /* local     node-equation map in newNodeEqnList (for SuperLU)     */
   /* A-ordered node-equation map in nodeEqnList                      */
   /* - first sort and compress the nodeEqnList and newNodeEqnList    */
   /* - then see how many variables are from external processors      */
   /* - construct communication pattern                               */
   /* --------------------------------------------------------------- */

   /* --- construct unique equation list for A --- */

   MLI_Utils_IntQSort2(nodeEqnList, newNodeEqnList, 0, totalNNodes-1);
   newNNodes = 1;
   for ( iN = 1; iN < totalNNodes; iN++ )
   {
      if (nodeEqnList[iN] != nodeEqnList[newNNodes-1]) 
      {
         newNodeEqnList[newNNodes] = newNodeEqnList[iN];
         nodeEqnList[newNNodes++] = nodeEqnList[iN];
      }
   }

   /* --- construct receive processor list --- */

   iTempArray = new int[nprocs];
   for ( iP = 0; iP < nprocs; iP++ ) iTempArray[iP] = 0;
   for ( iN = 0; iN < newNNodes; iN++ )
   {
      if (nodeEqnList[iN] < localStartRow || nodeEqnList[iN] >= localEndRow)
      {
         for ( iP = 0; iP < nprocs; iP++ )
            if ( nodeEqnList[iN] < partition[iP] ) break;
         iTempArray[iP-1]++;
      }
   }
   nRecvs = 0;
   for ( iP = 0; iP < nprocs; iP++ ) if ( iTempArray[iP] > 0 ) nRecvs++;
   if ( nRecvs > 0 )
   {
      recvLengs = new int[nRecvs];
      recvProcs = new int[nRecvs];
      iSendBufs = new int*[nRecvs];
      nRecvs = 0;
      for ( iP = 0; iP < nprocs; iP++ ) 
      {
         if ( iTempArray[iP] > 0 ) 
         {
            recvProcs[nRecvs] = iP;
            recvLengs[nRecvs++] = iTempArray[iP];
         }
      }
   }
   delete [] iTempArray;

   /* --- construct send processor list --- */

   procArray = new int[nprocs];
   for ( iP = 0; iP < nprocs; iP++ ) procArray[iP] = 0;
   for ( iP = 0; iP < nRecvs; iP++ ) procArray[recvProcs[iP]] = 1;
   iTempArray = procArray;
   procArray = new int[nprocs];
   MPI_Allreduce(iTempArray, procArray, nprocs, MPI_INT, MPI_SUM, comm);
   nSends = procArray[mypid];
   delete [] procArray;
   delete [] iTempArray;
   if ( nSends > 0 )
   {
      sendLengs = new int[nSends];
      sendProcs = new int[nSends];
      iRecvBufs = new int*[nSends];
      requests  = new MPI_Request[nSends];
      statuses  = new MPI_Status[nSends];
   }

   /* --- get send lengths and sort send processor list --- */

   for ( iP = 0; iP < nSends; iP++ )
      MPI_Irecv(&(sendLengs[iP]), 1, MPI_INT, MPI_ANY_SOURCE, 57421, comm,
                &(requests[iP]));
   for ( iP = 0; iP < nRecvs; iP++ )
      MPI_Send(&(recvLengs[iP]), 1, MPI_INT, recvProcs[iP], 57421, comm);

   for ( iP = 0; iP < nSends; iP++ )
   {
      MPI_Wait( &(requests[iP]), &(statuses[iP]) );
      sendProcs[iP] = statuses[iP].MPI_SOURCE;
   }
   MLI_Utils_IntQSort2(sendProcs, sendLengs, 0, nSends-1);

   /* --- send the receive equation list to the send processors --- */

   for ( iP = 0; iP < nSends; iP++ )
   {
      iRecvBufs[iP] = new int[sendLengs[iP]];
      MPI_Irecv(iRecvBufs[iP], sendLengs[iP], MPI_INT, sendProcs[iP], 37290, 
                comm, &(requests[iP]));
   }
   for ( iP = 0; iP < nRecvs; iP++ ) 
   {
      iSendBufs[iP] = new int[recvLengs[iP]];
      recvLengs[iP] = 0;
   }
   for ( iN = 0; iN < newNNodes; iN++ )
   {
      if (nodeEqnList[iN] < localStartRow || nodeEqnList[iN] >= localEndRow)
      {
         for ( iP = 0; iP < nprocs; iP++ )
            if ( nodeEqnList[iN] < partition[iP] ) break;
         index = MLI_Utils_BinarySearch( iP-1, recvProcs, nRecvs);
         iSendBufs[index][recvLengs[index]++] = nodeEqnList[iN];
      }
   }
   for ( iP = 0; iP < nRecvs; iP++ )
   {
      MPI_Send(iSendBufs[iP], recvLengs[iP], MPI_INT, recvProcs[iP], 37290, 
               comm);
      delete [] iSendBufs[iP];
   }
   if ( nRecvs > 0 ) delete [] iSendBufs;
   MPI_Waitall( nSends, requests, statuses );

   /* --- set up the send Map (which elements to send) --- */

   nSendMap = 0;
   for ( iP = 0; iP < nSends; iP++ ) nSendMap += sendLengs[iP];
   sendMap = new int[nSendMap];
   nSendMap = 0;
   for ( iP = 0; iP < nSends; iP++ )
   {
      for ( jN = 0; jN < sendLengs[iP]; jN++ )
      {
         globalEqnNum = iRecvBufs[iP][jN];
         index = MLI_Utils_BinarySearch(globalEqnNum,nodeEqnList,newNNodes);
         sendMap[nSendMap++] = index;
      }
      delete [] iRecvBufs[iP];
   }
   if ( nSends > 0 ) 
   {
      delete [] iRecvBufs;
      delete [] requests;
      delete [] statuses;
   }

   /* --- store the communication information --- */

   ddObj_ = new MLI_AMGSA_DD[1];
   ddObj_->sendMap    = sendMap;
   ddObj_->nSendMap   = nSendMap;
   ddObj_->nSends     = nSends;
   ddObj_->nRecvs     = nRecvs;
   ddObj_->sendProcs  = sendProcs;
   ddObj_->recvProcs  = recvProcs;
   ddObj_->sendLengs  = sendLengs;
   ddObj_->recvLengs  = recvLengs;
   ddObj_->NNodes     = newNNodes;
   ddObj_->dofPerNode = blockSize;
   ddObj_->ANodeEqnList = nodeEqnList;
   ddObj_->SNodeEqnList = newNodeEqnList;
   return 1;
}

