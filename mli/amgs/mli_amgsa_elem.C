/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

// *********************************************************************
// This file is customized to use HYPRE matrix format
// *********************************************************************

// *********************************************************************
// local includes
// ---------------------------------------------------------------------

#include <string.h>
#include <iostream.h>
#include <assert.h>

#include "HYPRE.h"
#include "utilities/utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "seq_mv/seq_mv.h"
#include "parcsr_mv/parcsr_mv.h"

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
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupUsingFEData( MLI *mli ) 
{
   int          i, j, k, level, mypid, nprocs, nElems, nodeNumFields;
   int          nodeFieldID, elemNNodes, **elemNodeLists, *elemNodeList1D;
   int          blockSize=3, nNodes, *nodeEqnList, *sortArray, elemIndex;
   int          eqnIndex, *aggrMap, currMacroNumber, elemStart, elemCount;
   int          *macroNumbers, eMatDim, ierr, matz=1, *partition;
   int          localStartRow, localNRows, nMacros, *macroSizes, macroMax;
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
   cout << " MLI_Method_AMGSA::setupUsingFEData begins..." << endl;
   cout.flush();
#endif

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if ( mli == NULL )
   {
      cout << " MLI_Method_AMGSA::setupUsingFEData ERROR - no mli." << endl;
      exit(1);
   }
   level = 0;
   fedata = mli->getFEData(level);
   if ( fedata == NULL )
   {
      cout << " MLI_Method_AMGSA::setupUsingFEData ERROR - no fedata." << endl;
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
   hypreA   = (hypre_ParCSRMatrix *) mliAmat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   localStartRow = partition[mypid];
   localNRows    = partition[mypid+1] - localStartRow;
printf("localNRows = %d\n", localNRows);

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
printf("blockSize = %d\n", blockSize);
   fedata->getNumNodes(nNodes);
printf("nNodes = %d\n", nNodes);

   /* --------------------------------------------------------------- */
   /* construct element-element matrix                                */
   /* --------------------------------------------------------------- */

printf("get EN \n");
   MLI_FEDataConstructElemNodeMatrix(comm, fedata, &mliENMat);
   hypreEN = (hypre_ParCSRMatrix *) mliENMat->getMatrix();
printf("get NE \n");
   MLI_FEDataConstructNodeElemMatrix(comm, fedata, &mliNEMat);
   hypreNE = (hypre_ParCSRMatrix *) mliNEMat->getMatrix();
printf("get EE \n");
   hypreEE = (hypre_ParCSRMatrix *) 
              hypre_ParMatmul( (hypre_ParCSRMatrix *) hypreEN,
                               (hypre_ParCSRMatrix *) hypreNE);
printf("get EE done\n");

   /* --------------------------------------------------------------- */
   /* perform element agglomeration                                   */
   /* --------------------------------------------------------------- */

   funcPtr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   mliEEMat = new MLI_Matrix( (void *) hypreEE, paramString, funcPtr );
printf("agglomerate \n");
   MLI_FEDataAgglomerateElemsLocal(mliEEMat, &macroNumbers);
printf("agglomerate done\n");
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
      nodeEqnList = new int[nElems*elemNNodes];
      nodeEqnMap->getMap(nElems*elemNNodes, elemNodeList1D, nodeEqnList);
   }

   /* --- sort the element to macroelement map --- */

   sortArray = new int[nElems];
   for ( i = 0; i < nElems; i++ ) sortArray[i] = i; 
   MLI_Utils_IntQSort2(macroNumbers, sortArray, 0, nElems-1);
   nMacros = macroNumbers[nElems-1] + 1;

   /* --- get the node to equation map --- */

   aggrMap = new int[localNRows];
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
         eqnIndex = nodeEqnList[elemIndex*elemNNodes+j];
         if ( aggrMap[eqnIndex] < 0 )
            for ( k = 0; k < blockSize; k++ ) 
               aggrMap[eqnIndex+k] = currMacroNumber;
      }
   }
   sa_data[0] = aggrMap;
   sa_counts[0] = nMacros;
   macroSizes = new int[nMacros];
   for ( i = 0; i < nMacros; i++ ) macroSizes[i] = 0;
   for ( i = 0; i < localNRows; i++ ) macroSizes[aggrMap[i]]++;
   macroMax = 0;
   for ( i = 0; i < nMacros; i++ ) 
      if ( macroSizes[i] > macroMax ) macroMax = macroSizes[i];
   
   /* --- compute null spaces --- */

   if ( nullspace_vec != NULL ) delete [] nullspace_vec;
   nullspace_len = localNRows;
   nullspace_vec = new double[nullspace_len*nullspace_dim];
   evalues  = new double[macroMax];
   dAux1    = new double[macroMax];
   dAux2    = new double[macroMax];
   eMatDim  = elemNNodes * blockSize;
   elemMat  = new double[eMatDim];
   evectors = NULL;
   elemMats = NULL;

   currMacroNumber = macroNumbers[0];
   elemStart = 0;
   while ( elemStart < nElems )
   {
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
      macroMatDim = macroNumNodes * blockSize;
      evectors = new double[macroMatDim*macroMatDim];
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
               MLI_Utils_BinarySearch(nodeIndex, macroNodeList, macroNumNodes );
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
      macroNodeEqnList = new int[macroNumNodes];
      nodeEqnMap->getMap(macroNumNodes, macroNodeList, macroNodeEqnList);
      for ( i = 0; i < nullspace_dim; i++ )
      {
         for ( j = 0; j < macroNumNodes; j++ )
         {
            eqnNumber = macroNodeEqnList[j];
            for ( k = 0; k < blockSize; k++ )
               nullspace_vec[eqnNumber+k+i*nullspace_len] =
                  evectors[j*blockSize+k+i*macroMatDim];
         }
      } 
      delete [] macroNodeEqnList;
      delete [] evectors;
      delete [] elemMats;
      delete [] nodeNodeMap;
      delete [] macroNodeList;
      elemStart = elemCount;
   }

   /* --------------------------------------------------------------- */
   /* clean up                                                        */
   /* --------------------------------------------------------------- */

   delete [] evalues;
   delete [] elemMat;
   delete [] dAux1;
   delete [] dAux2;
   delete [] elemNodeList1D;
   delete [] elemNodeLists;
   delete [] elemIDs;
   delete [] sortArray;
   if ( nodeEqnList != NULL ) delete [] nodeEqnList;
   return 0;
}

