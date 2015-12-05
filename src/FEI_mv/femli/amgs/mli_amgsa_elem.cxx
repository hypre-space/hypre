/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/





// *********************************************************************
// This file is customized to use HYPRE matrix format
// *********************************************************************

// *********************************************************************
// local includes
// ---------------------------------------------------------------------

#include <string.h>
#include <assert.h>

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "seq_mv/seq_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#include "amgs/mli_method_amgsa.h"
#include "util/mli_utils.h"
#include "mapper/mli_mapper.h"
#include "fedata/mli_fedata.h"
#include "fedata/mli_fedata_utils.h"
#include "matrix/mli_matrix.h"
 
extern "C"
{
void mli_computespectrum_(int *,int *,double *, double *, int *, double *,
                          double *, double *, int *);
}

/***********************************************************************
 * set up initial data using FEData
 * (nullspaceVec, saData_, saCounts_)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupUsingFEData( MLI *mli ) 
{
   int          i, j, k, level, mypid, nprocs, nElems, nodeNumFields;
   int          nodeFieldID, elemNNodes, **elemNodeLists, *elemNodeList1D;
   int          blockSize=3, nNodes, *nodeEqnList, *sortArray, elemIndex;
   int          eqnIndex, *aggrMap, currMacroNumber, elemStart, elemCount;
   int          *macroNumbers, eMatDim, ierr, matz=1, *partition, aggrSize;
   int          localStartRow, localNRows, nMacros, *macroSizes, nAggr;
   int          j1, k1, *macroNodeEqnList, nodeIndex, eqnInd1, eqnInd2;
   int          eqnNumber, colOffset1, colOffset2, *nodeNodeMap, elemID;
   int          macroMatDim, *macroNodeList, macroNumNodes, *elemIDs;
   double       *evalues, *evectors, *dAux1, *dAux2;
   double       *elemMat, *elemMats;
   char         paramString[100];
   MPI_Comm     comm;
   MLI_FEData   *fedata;
   MLI_Mapper   *nodeEqnMap;
   MLI_Function *funcPtr;
   MLI_Matrix   *mliAmat, *mliENMat, *mliNEMat, *mliEEMat;
   hypre_ParCSRMatrix *hypreA, *hypreEE, *hypreEN, *hypreNE;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupUsingFEData begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if ( mli == NULL )
   {
      printf("MLI_Method_AMGSA::setupUsingFEData ERROR - no mli.");
      exit(1);
   }
   level = 0;
   fedata = mli->getFEData(level);
   if ( fedata == NULL )
   {
      printf("MLI_Method_AMGSA::setupUsingFEData ERROR - no fedata.\n");
      exit(1);
   }
   nodeEqnMap = mli->getNodeEqnMap(level);

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
   localNRows    = partition[mypid+1] - localStartRow;
   free( partition );

   /* --------------------------------------------------------------- */
   /* fetch FEData information                                        */
   /* --------------------------------------------------------------- */

   fedata->getNodeNumFields(nodeNumFields);
   if ( nodeNumFields != 1 ) return 1;
   fedata->getNumElements( nElems );
   elemIDs = new int[nElems];
   fedata->getElemBlockGlobalIDs(nElems, elemIDs);
   fedata->getElemNumNodes(elemNNodes);
   elemNodeList1D = new int[nElems*elemNNodes];
   elemNodeLists = new int*[nElems];
   for ( i = 0; i < nElems; i++ )
      elemNodeLists[i] = &(elemNodeList1D[i*elemNNodes]);
   fedata->getElemBlockNodeLists(nElems, elemNNodes, elemNodeLists);
   fedata->getNodeFieldIDs(nodeNumFields, &nodeFieldID);
   fedata->getFieldSize(nodeFieldID, blockSize);
   fedata->getNumNodes(nNodes);

   /* --------------------------------------------------------------- */
   /* construct element-element matrix                                */
   /* --------------------------------------------------------------- */

   MLI_FEDataConstructElemNodeMatrix(comm, fedata, &mliENMat);
   hypreEN = (hypre_ParCSRMatrix *) mliENMat->getMatrix();
   MLI_FEDataConstructNodeElemMatrix(comm, fedata, &mliNEMat);
   hypreNE = (hypre_ParCSRMatrix *) mliNEMat->getMatrix();
   hypreEE = (hypre_ParCSRMatrix *) 
              hypre_ParMatmul( (hypre_ParCSRMatrix *) hypreEN,
                               (hypre_ParCSRMatrix *) hypreNE);

   /* --------------------------------------------------------------- */
   /* perform element agglomeration                                   */
   /* --------------------------------------------------------------- */

   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   mliEEMat = new MLI_Matrix( (void *) hypreEE, paramString, funcPtr );
   MLI_FEDataAgglomerateElemsLocalOld(mliEEMat, &macroNumbers);
   delete mliENMat;
   delete mliNEMat;
   delete mliEEMat;

   /* --------------------------------------------------------------- */
   /* form aggregates                                                 */
   /* --------------------------------------------------------------- */

   /* --- get the node to equation map --- */

   nodeEqnList = NULL;
   if ( nodeEqnMap != NULL ) 
   {
      if ( nElems > 0 ) 
      {
         nodeEqnList = new int[nElems*elemNNodes];
         nodeEqnMap->getMap(nElems*elemNNodes,elemNodeList1D,nodeEqnList);
      }
      else nodeEqnList = NULL;
   }

   /* --- sort the element to macroelement map --- */

   if ( nElems > 0 ) sortArray = new int[nElems];
   else              sortArray = NULL;
   for ( i = 0; i < nElems; i++ ) sortArray[i] = i; 
   MLI_Utils_IntQSort2(macroNumbers, sortArray, 0, nElems-1);
   if ( nElems > 0 ) nMacros = macroNumbers[nElems-1] + 1;
   else              nMacros = 0;

   /* --- get the node to equation map --- */

   if ( localNRows > 0 ) aggrMap = new int[localNRows];
   else                  aggrMap = NULL;
   for ( i = 0; i < localNRows; i++ ) aggrMap[i] = -1; 

   /* --- map equation to aggregates --- */

   currMacroNumber = -1;
   for ( i = 0; i < nElems; i++ ) 
   {
      if ( macroNumbers[i] != currMacroNumber )
         currMacroNumber = macroNumbers[i];
       
      elemIndex = sortArray[i];
      for ( j = 0; j < elemNNodes; j++ ) 
      {
         eqnIndex = nodeEqnList[elemIndex*elemNNodes+j] - localStartRow;
         /* option between how aggregates are chosen in view of overlap 
            if (eqnIndex>=0 && eqnIndex<localNRows && aggrMap[eqnIndex]<0)
         */
         if ( eqnIndex >= 0 && eqnIndex < localNRows )
            for ( k = 0; k < blockSize; k++ ) 
               aggrMap[eqnIndex+k] = currMacroNumber;
      }
   }

   /* --- analyze aggregate sizes (merge small aggregates) --- */
 
   if ( nElems > 0 ) macroSizes = new int[nElems];
   else              macroSizes = NULL;
   for ( i = 0; i < nMacros; i++ ) macroSizes[i] = 0; 
   for ( i = 0; i < localNRows; i++ ) macroSizes[aggrMap[i]]++;
   
   /* --- compute null spaces --- */

   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   nullspaceLen_ = localNRows;
   nullspaceVec_ = new double[nullspaceLen_*nullspaceDim_];
   eMatDim  = elemNNodes * blockSize;
   elemMat  = new double[eMatDim*eMatDim];
   evectors = NULL;
   elemMats = NULL;

   elemStart = 0;
   while ( elemStart < nElems )
   {
      currMacroNumber = macroNumbers[elemStart];
      if ( outputLevel_ >= 1 && currMacroNumber % 200 == 0 )
         printf("Computing null spaces of aggregate %d (%d)\n", 
                currMacroNumber, nMacros);
      elemCount = elemStart + 1;
      while (macroNumbers[elemCount] == currMacroNumber && elemCount < nElems) 
         elemCount++;
      macroNumNodes = ( elemCount - elemStart ) * elemNNodes;
      macroNodeList = new int[macroNumNodes];
      for ( i = elemStart; i < elemCount; i++ )
      {
         elemIndex = sortArray[i];
         for ( j = 0; j < elemNNodes; j++ )
            macroNodeList[(i-elemStart)*elemNNodes+j] = 
               elemNodeLists[elemIndex][j];
      }
      MLI_Utils_IntQSort2(macroNodeList, NULL, 0, macroNumNodes-1);
      k = 1;
      for ( i = 1; i < macroNumNodes; i++ )
         if ( macroNodeList[i] != macroNodeList[k-1] ) 
            macroNodeList[k++] = macroNodeList[i]; 
      macroNumNodes = k;

      macroNodeEqnList = new int[macroNumNodes];
      nodeEqnMap->getMap(macroNumNodes,macroNodeList,macroNodeEqnList);

      aggrSize = 0;
      for ( j = 0; j < macroNumNodes; j++ )
      {
         eqnNumber = macroNodeEqnList[j] - localStartRow;
         if ( eqnNumber >= 0 && eqnNumber < localNRows &&
              aggrMap[eqnNumber] == macroNumbers[elemStart] )
            aggrSize += blockSize;
      } 
      if ( aggrSize == 0 ) continue;

      macroMatDim = macroNumNodes * blockSize;
      evectors = new double[macroMatDim*macroMatDim];
      evalues  = new double[macroMatDim];
      dAux1    = new double[macroMatDim];
      dAux2    = new double[macroMatDim];
      elemMats = new double[macroMatDim*macroMatDim];
      nodeNodeMap = new int[elemNNodes];
      for ( i = 0; i < macroMatDim*macroMatDim; i++ ) elemMats[i] = 0.0;
      for ( i = elemStart; i < elemCount; i++ )
      {
         elemIndex = sortArray[i];
         elemID = elemIDs[elemIndex];
         fedata->getElemMatrix(elemID, eMatDim, elemMat); 
         for ( j = 0; j < elemNNodes; j++ )
         {
            nodeIndex = elemNodeLists[elemIndex][j];
            nodeNodeMap[j] = 
               MLI_Utils_BinarySearch(nodeIndex, macroNodeList, macroNumNodes);
         }
         for ( j = 0; j < elemNNodes; j++ )
         {
            eqnInd1 = nodeNodeMap[j] * blockSize;
            for ( j1 = 0; j1 < blockSize; j1++ )
            {
               colOffset1 = macroMatDim * ( eqnInd1 + j1 );
               colOffset2 = eMatDim * ( j * blockSize + j1 );
               for ( k = 0; k < elemNNodes; k++ )
               {
                  eqnInd2 = nodeNodeMap[k] * blockSize;
                  for ( k1 = 0; k1 < blockSize; k1++ )
                     elemMats[eqnInd2+k1+colOffset1] +=
                          elemMat[k*blockSize+k1+colOffset2];
               }
            }
         }
      }
      mli_computespectrum_(&macroMatDim, &macroMatDim, elemMats, evalues, 
                           &matz, evectors, dAux1, dAux2, &ierr);

      for ( i = 0; i < nullspaceDim_; i++ )
      {
         for ( j = 0; j < macroNumNodes; j++ )
         {
            eqnNumber = macroNodeEqnList[j] - localStartRow;
            if ( eqnNumber >= 0 && eqnNumber < localNRows &&
                 aggrMap[eqnNumber] == macroNumbers[elemStart] )
            {
               for ( k = 0; k < blockSize; k++ )
                  nullspaceVec_[eqnNumber+k+i*nullspaceLen_] =
                     evectors[j*blockSize+k+i*macroMatDim];
            }
         }
      } 
      delete [] macroNodeEqnList;
      delete [] nodeNodeMap;
      delete [] macroNodeList;
      delete [] evectors;
      delete [] elemMats;
      delete [] evalues;
      delete [] dAux1;
      delete [] dAux2;
      elemStart = elemCount;
   }

   /* --------------------------------------------------------------- */
   /* massage aggregate numbers and store them                        */
   /* --------------------------------------------------------------- */

   for ( i = 0; i < nMacros; i++ ) macroNumbers[i] = 0; 
   for ( i = 0; i < localNRows; i++ ) macroNumbers[aggrMap[i]]++;
   for ( i = 0; i < nMacros; i++ ) 
   {
      if ( macroNumbers[i] > 0 ) macroNumbers[i] = 1;
      else                       macroNumbers[i] = -1;
   }
   nAggr = 0;
   for ( i = 0; i < nMacros; i++ ) 
      if ( macroNumbers[i] > 0 ) macroNumbers[i] = nAggr++;
   for ( i = 0; i < localNRows; i++ ) 
      aggrMap[i] = macroNumbers[aggrMap[i]];

printf("setupUsingFEData : no aggregate\n");
/*
   saCounts_[0] = nAggr;
   saData_[0]   = aggrMap;
*/
   
   /* --------------------------------------------------------------- */
   /* clean up                                                        */
   /* --------------------------------------------------------------- */

   if ( eMatDim > 0 ) delete [] elemMat;
   if ( nElems  > 0 ) delete [] elemNodeList1D;
   if ( nElems  > 0 ) delete [] elemNodeLists;
   if ( nElems  > 0 ) delete [] elemIDs;
   if ( nElems  > 0 ) delete [] sortArray;
   if ( nElems  > 0 ) delete [] macroSizes;
   if ( nElems  > 0 ) free( macroNumbers );
   if ( nodeEqnList != NULL ) delete [] nodeEqnList;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupUsingFEData ends.\n");
#endif

   return 0;
}

