/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.8 $
 ***********************************************************************EHEADER*/





/**************************************************************************
 **************************************************************************
 * MLI_FEData Class functions
 **************************************************************************
 **************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#ifdef WIN32
#define strcmp _stricmp
#endif

#include "utilities/_hypre_utilities.h"
#include "fedata/mli_fedata.h"
#include "util/mli_utils.h"

/**************************************************************************
 * constructor 
 *-----------------------------------------------------------------------*/

MLI_FEData::MLI_FEData(MPI_Comm mpi_comm)
{
   mpiComm_          = mpi_comm;
   outputLevel_      = 0;
   spaceDimension_   = -1;
   orderOfPDE_       = -1;
   orderOfFE_        = -1;
   numElemBlocks_    = 0;
   elemBlockList_    = NULL;
   numFields_        = 0;
   fieldIDs_         = NULL;
   fieldSizes_       = NULL;
   currentElemBlock_ = -1;
   USR_computeShapeFuncInterpolant = NULL;
   USR_getElemMatrix               = NULL;
}

//*************************************************************************
// destructor 
//-------------------------------------------------------------------------

MLI_FEData::~MLI_FEData()
{
   for ( int i = 0; i < numElemBlocks_; i++ ) deleteElemBlock(i);
   delete [] elemBlockList_;
   if ( fieldIDs_   != NULL ) delete [] fieldIDs_;
   if ( fieldSizes_ != NULL ) delete [] fieldSizes_;
}

//*************************************************************************
// set diagnostics output level
//-------------------------------------------------------------------------

int MLI_FEData::setOutputLevel(int level)
{
   if ( level < 0 ) 
   {
      printf("setOutputLevel ERROR : level should be >= 0.\n");
      return 0;
   }
   outputLevel_ = level;
   return 1;
}

//*************************************************************************
// dimension of the physical problem (2D, 3D, etc.) 
//-------------------------------------------------------------------------

int MLI_FEData::setSpaceDimension(int dimension)
{
   if ( dimension <= 0 || dimension > 4 )
   {
      printf("setSpaceDimension ERROR : dimension should be > 0 and <= 4.\n");
      exit(1);
   }
   if (outputLevel_ >= 1) printf("setSpaceDimension = %d\n", dimension);
   spaceDimension_ = dimension;
   return 1;
}

//*************************************************************************
// order of the partial differential equation 
//-------------------------------------------------------------------------

int MLI_FEData::setOrderOfPDE(int pdeOrder)
{
   if ( pdeOrder <= 0 || pdeOrder > 4 )
   {
      printf("setOrderOfPDE ERROR : PDE order should be > 0 and <= 4.\n");
      exit(1);
   }
   if (outputLevel_ >= 1) printf("setOrderOfPDE = %d\n", pdeOrder);
   orderOfPDE_ = pdeOrder;
   return 1;
}

//*************************************************************************
// order of the finite element discretization
//-------------------------------------------------------------------------

int MLI_FEData::setOrderOfFE(int feOrder)
{
   if ( feOrder <= 0 || feOrder > 4 )
   {
      printf("setOrderOfFE ERROR : order should be > 0 and <= 4.\n");
      exit(1);
   }
   if (outputLevel_ >= 1) printf("setOrderOfFE = %d\n", feOrder);
   orderOfFE_ = feOrder;
   return 1;
}

//*************************************************************************
// set current element block ID (not implemented yet for blockID > 0)
//-------------------------------------------------------------------------

int MLI_FEData::setCurrentElemBlockID(int blockID)
{
   if ( blockID != 0 )
   {
      printf("setCurrentElemBlockID ERROR : blockID other than 0 invalid.\n");
      exit(1);
   }
   if ( outputLevel_ >= 1 ) printf("setCurrentElemBlockID = %d\n", blockID);
   currentElemBlock_ = blockID;
   return 1;
}

//*************************************************************************
// initialize field information
//-------------------------------------------------------------------------

int MLI_FEData::initFields(int nFields, const int *fieldSizes,
                           const int *fieldIDs)
{
   int  i, mypid;

   if ( nFields <= 0 || nFields > 10 )
   {
      printf("initFields ERROR : nFields invalid.\n");
      exit(1);
   }
   MPI_Comm_rank(mpiComm_, &mypid);
   if ( outputLevel_ >= 1 && mypid == 0 )
   {
      printf("\tinitFields : number of fields = %d\n", nFields);
      for ( i = 0; i < nFields; i++ )
         printf("\t  fieldID and size = %d %d\n",fieldIDs[i],fieldSizes[i]);
   }
   numFields_ = nFields;
   if ( fieldSizes_ != NULL ) delete [] fieldSizes_;
   fieldSizes_ = new int[nFields];
   for ( i = 0; i < nFields; i++ ) fieldSizes_[i] = fieldSizes[i];
   if ( fieldIDs_ != NULL ) delete [] fieldIDs_;
   fieldIDs_ = new int[nFields];
   for ( i = 0; i < nFields; i++ ) fieldIDs_[i] = fieldIDs[i];
   return 1;
}

//*************************************************************************
// initialize the element block information
//-------------------------------------------------------------------------

int MLI_FEData::initElemBlock(int nElems, int nNodesPerElem,
                     int nodeNumFields, const int *nodeFieldIDs,
                     int elemNumFields, const int *elemFieldIDs)
{
   int           i;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   if ( nElems <= 0 )
   {
      printf("initElemBlock ERROR : nElems <= 0.\n");
      exit(1);
   }
   if ( elemNumFields < 0 )
   {
      printf("initElemBlock ERROR : elemNumFields < 0.\n");
      exit(1);
   }
   if ( nodeNumFields < 0 )
   {
      printf("initElemBlock ERROR : nodeNumFields < 0.\n");
      exit(1);
   }
   if (outputLevel_ >= 1) 
   {
      printf("initElemBlock : nElems = %d\n", nElems);
      printf("initElemBlock : node nFields = %d\n", nodeNumFields);
      printf("initElemBlock : elem nFields = %d\n", elemNumFields);
   }

   // -------------------------------------------------------------
   // --- clean up previous element setups
   // -------------------------------------------------------------

   if ( currentElemBlock_ >= 0 && currentElemBlock_ < numElemBlocks_ && 
        elemBlockList_[currentElemBlock_] != NULL ) 
   {
      deleteElemBlock(currentElemBlock_);
      createElemBlock(currentElemBlock_);
   }
   else if ( currentElemBlock_ >= 0 && currentElemBlock_ < numElemBlocks_ )
      createElemBlock(currentElemBlock_);
   else
      createElemBlock(++currentElemBlock_);

   currBlock = elemBlockList_[currentElemBlock_];

   // -------------------------------------------------------------
   // --- allocate space for element IDs and node lists
   // -------------------------------------------------------------

   currBlock->numLocalElems_ = nElems;
   currBlock->elemGlobalIDs_ = new int[nElems];
   for ( i = 0; i < nElems; i++ ) currBlock->elemGlobalIDs_[i] = -1;
   currBlock->elemNodeIDList_ = new int*[nElems];
   for ( i = 0; i < nElems; i++ ) currBlock->elemNodeIDList_[i] = NULL;

   // -------------------------------------------------------------
   // --- store number of nodes per element information
   // -------------------------------------------------------------

   if ( nNodesPerElem <= 0 || nNodesPerElem > 200 )
   {
      printf("initElemBlock ERROR : nNodesPerElem <= 0 or > 200.\n");
      exit(1);
   }
   currBlock->elemNumNodes_ = nNodesPerElem;

   // -------------------------------------------------------------
   // --- store node level data
   // -------------------------------------------------------------

   currBlock->nodeNumFields_ = nodeNumFields;
   currBlock->nodeFieldIDs_  = new int[nodeNumFields];
   for ( i = 0; i < nodeNumFields; i++ ) 
      currBlock->nodeFieldIDs_[i] = nodeFieldIDs[i]; 

   // -------------------------------------------------------------
   // --- store element level data
   // -------------------------------------------------------------

   currBlock->elemNumFields_ = elemNumFields;
   if ( elemNumFields > 0 )
   {
      currBlock->elemFieldIDs_  = new int[elemNumFields];
      for ( i = 0; i < elemNumFields; i++ ) 
         currBlock->elemFieldIDs_[i] = elemFieldIDs[i]; 
   }
   return 1;
}

//*************************************************************************
// initialize the element connectivities
//-------------------------------------------------------------------------

int MLI_FEData::initElemBlockNodeLists(int nElems, 
                        const int *eGlobalIDs, int nNodesPerElem,
                        const int* const *nGlobalIDLists,
                        int spaceDim, const double* const *coord)
{
   int           i, j, length, *intArray;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("initElemBlockNodeLists ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( nNodesPerElem != currBlock->elemNumNodes_ )
   {
      printf("initElemBlockNodeLists ERROR : nNodesPerElem invalid.\n");
      exit(1);
   }
   if ( spaceDimension_ != spaceDim && coord != NULL )
   {
      printf("initElemBlockNodeLists ERROR : spaceDim invalid.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("initElemBlockNodeLists Diagnostics: segFault test.\n");
   double ddata;
   for (i = 0; i < nElems; i++) 
   {
      index  = eGlobalIDs[i];
      for (j = 0; j < nNodesPerElem; j++) 
         length = nGlobalIDLists[i][j];
      if ( coord != NULL )
      {
         for (j = 0; j < nNodesPerElem*spaceDim; j++) 
         ddata = coord[i][j];
      }
   }
   printf("initElemBlockNodeLists Diagnostics : passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate storage and load for storing element global IDs
   // -------------------------------------------------------------

   if ( currBlock->elemGlobalIDs_ == NULL )
   {
      printf("initElemBlockNodeLists ERROR : have not called initElemBlock.");
      exit(1);
   }
   for (i = 0; i < nElems; i++) currBlock->elemGlobalIDs_[i] = eGlobalIDs[i];

   // -------------------------------------------------------------
   // --- allocate storage and load for element node connectivities 
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ ) 
   {
      currBlock->elemNodeIDList_[i] = new int[nNodesPerElem];
      intArray = currBlock->elemNodeIDList_[i];
      for ( j = 0; j < nNodesPerElem; j++ ) 
         intArray[j] = nGlobalIDLists[i][j]; 
   }   
   if ( coord == NULL ) return 1;

   // -------------------------------------------------------------
   // --- temporarily store away nodal coordinates
   // -------------------------------------------------------------

   length = nNodesPerElem * spaceDimension_ * nElems;
   currBlock->nodeCoordinates_ =  new double[length];
   length = nNodesPerElem * spaceDimension_;
   for ( i = 0; i < nElems; i++ ) 
   {
      for ( j = 0; j < length; j++ ) 
         currBlock->nodeCoordinates_[i*length+j] = coord[i][j];
   }
   return 1;
}

//*************************************************************************
// initialize the element connectivities
//-------------------------------------------------------------------------

int MLI_FEData::initElemNodeList( int eGlobalID, int nNodesPerElem,
                                  const int* nGlobalIDs, int spaceDim, 
                                  const double *coord)
{
   int           i, j, length, index, *intArray, nElems;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nNodesPerElem != currBlock->elemNumNodes_ )
   {
      printf("initElemNodeList ERROR : nNodesPerElem invalid.\n");
      exit(1);
   }
   if ( spaceDimension_ != spaceDim && coord != NULL )
   {
      printf("initElemNodeList ERROR : spaceDim invalid.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("initElemNodeList Diagnostics: segFault test.\n");
   double ddata;
   for (i = 0; i < nNodesPerElem; i++) index = nGlobalIDs[i];
   if ( coord != NULL )
      for (i = 0; i < nNodesPerElem*spaceDim; i++) ddata = coord[i];
   printf("initElemNodeList Diagnostics : passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate storage and load for storing element global IDs
   // -------------------------------------------------------------

   if ( currBlock->elemGlobalIDs_ == NULL )
   {
      printf("initElemNodeList ERROR : have not called initElemBlock.");
      exit(1);
   }
   index = currBlock->elemOffset_++;
   currBlock->elemGlobalIDs_[index] = eGlobalID;

   // -------------------------------------------------------------
   // --- allocate storage and load for element node connectivities 
   // -------------------------------------------------------------

   currBlock->elemNodeIDList_[index] = new int[nNodesPerElem];
   intArray = currBlock->elemNodeIDList_[index];
   for ( j = 0; j < nNodesPerElem; j++ ) intArray[j] = nGlobalIDs[j]; 
   if ( coord == NULL ) return 1;

   // -------------------------------------------------------------
   // --- temporarily store away nodal coordinates
   // -------------------------------------------------------------

   nElems = currBlock->numLocalElems_;
   length = nNodesPerElem * spaceDimension_ * nElems;
   if ( currBlock->nodeCoordinates_ == NULL )
      currBlock->nodeCoordinates_ =  new double[length];
   length = nNodesPerElem * spaceDimension_;
   for ( i = 0; i < length; i++ ) 
      currBlock->nodeCoordinates_[index*length+i] = coord[i];
   return 1;
}

//*************************************************************************
// initialize shared node list 
//-------------------------------------------------------------------------

int MLI_FEData::initSharedNodes(int nNodes, const int *nGlobalIDs, 
                    const int *numProcs, const int * const *procLists)
{
   int i, j, length, index, index2, *nodeIDs, *auxArray;
   int *sharedNodeIDs, *sharedNodeNProcs, **sharedNodeProc, nSharedNodes;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   if ( nNodes < 0 )
   {
      printf("initSharedNodes ERROR : nNodes < 0.\n");
      exit(1);
   }
   if ( nNodes == 0 ) return 0;
   currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->sharedNodeIDs_ != NULL )
      printf("initSharedNodes WARNING : already initialized (1) ?\n");
   if ( currBlock->sharedNodeNProcs_ != NULL )
      printf("initSharedNodes WARNING : already initialized (2) ?\n");
   if ( currBlock->sharedNodeProc_ != NULL )
      printf("initSharedNodes WARNING : already initialized (3) ?\n");

#ifdef MLI_DEBUG_DETAILED
   printf("initSharedNodes Diagnostics: segFault test.\n");
   for (i = 0; i < nNodes; i++) 
   {
      index  = nGlobalIDs[i];
      length = numProcs[i];
      for (j = 0; j < length; j++) 
         index = procLists[i][j];
   }
   printf("initSharedNodes Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- sort and copy the shared node list
   // -------------------------------------------------------------

   nodeIDs  = new int[nNodes];
   auxArray = new int[nNodes];
   for (i = 0; i < nNodes; i++) nodeIDs[i]  = nGlobalIDs[i];
   for (i = 0; i < nNodes; i++) auxArray[i] = i;
   MLI_Utils_IntQSort2(nodeIDs, auxArray, 0, nNodes-1);
   nSharedNodes = 1;
   for (i = 1; i < nNodes; i++) 
      if ( nodeIDs[i] != nodeIDs[nSharedNodes-1] ) nSharedNodes++;
   sharedNodeIDs    = new int[nSharedNodes];
   sharedNodeNProcs = new int[nSharedNodes];
   sharedNodeProc   = new int*[nSharedNodes];
   nSharedNodes = 1;
   sharedNodeIDs[0] = nodeIDs[0];
   for (i = 1; i < nNodes; i++) 
      if ( nodeIDs[i] != sharedNodeIDs[nSharedNodes-1] )
         sharedNodeIDs[nSharedNodes++] = nodeIDs[i];
   for ( i = 0; i < nSharedNodes; i++ ) sharedNodeNProcs[i] = 0;
   for ( i = 0; i < nNodes; i++ )
   {
      index  = MLI_Utils_BinarySearch(nodeIDs[i],sharedNodeIDs,
                                      nSharedNodes);
      index2 = auxArray[i];
      sharedNodeNProcs[index] += numProcs[index2];
   }
   for ( i = 0; i < nSharedNodes; i++ )
   {
      sharedNodeProc[i] = new int[sharedNodeNProcs[i]];
      sharedNodeNProcs[i] = 0;
   }
   for ( i = 0; i < nNodes; i++ )
   {
      index  = MLI_Utils_BinarySearch(nodeIDs[i],sharedNodeIDs,
                                      nSharedNodes);
      index2 = auxArray[i];
      for ( j = 0; j < numProcs[index2]; j++ )
         sharedNodeProc[index][sharedNodeNProcs[index]++] = 
                       procLists[index2][j];
   }
   delete [] nodeIDs;
   delete [] auxArray;
   for ( i = 0; i < nSharedNodes; i++ )
   {
      MLI_Utils_IntQSort2(sharedNodeProc[i],NULL,0,sharedNodeNProcs[i]-1);
      length = 1;       
      for ( j = 1; j < sharedNodeNProcs[i]; j++ )
         if ( sharedNodeProc[i][j] != sharedNodeProc[i][length-1] )
            sharedNodeProc[i][length++] = sharedNodeProc[i][j];
      sharedNodeNProcs[i] = length;
   }
   currBlock->numSharedNodes_   = nSharedNodes;
   currBlock->sharedNodeIDs_    = sharedNodeIDs;
   currBlock->sharedNodeNProcs_ = sharedNodeNProcs;
   currBlock->sharedNodeProc_   = sharedNodeProc;

   return 1;
}

//*************************************************************************
// initialize element face lists 
//-------------------------------------------------------------------------

int MLI_FEData::initElemBlockFaceLists(int nElems, int nFaces,
                        const int* const *fGlobalIDLists)
{
   int           i, j, index, *elemFaceList;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("initElemBlockFaceLists ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( nFaces <= 0 || nFaces > 100 )
   {
      printf("initElemBlockFaceLists ERROR : nFaces invalid.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("initElemBlockFaceLists Diagnostics: segFault test.\n");
   for (i = 0; i < nElems; i++) 
      for (j = 0; j < nFaces; j++) index = fGlobalIDLists[i][j];
   printf("initElemBlockFaceLists Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate storage space
   // -------------------------------------------------------------

   if ( currBlock->elemFaceIDList_ == NULL )
   {
      currBlock->elemFaceIDList_ = new int*[nElems];
      currBlock->elemNumFaces_   = nFaces; 
      for (i = 0; i < nElems; i++) 
         currBlock->elemFaceIDList_[i] = new int[nFaces];
   }

   // -------------------------------------------------------------
   // --- load face information
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      index        = currBlock->elemGlobalIDAux_[i];
      elemFaceList = currBlock->elemFaceIDList_[i];
      for ( j = 0; j < nFaces; j++ ) 
         elemFaceList[j] = fGlobalIDLists[index][j];
   }
   return 1;
} 

//*************************************************************************
// initialize face node list
//-------------------------------------------------------------------------

int MLI_FEData::initFaceBlockNodeLists(int nFaces, const int *fGlobalIDs,
                              int nNodes, const int * const *nGlobalIDLists)
{
   int           i, j, index, *faceArray, **faceNodeList;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->elemFaceIDList_ == NULL )
   {
      printf("initFaceBlockNodeLists ERROR : elem-face not initialized.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("initFaceBlockNodeLists Diagnostics: segFault test.\n");
   for (i = 0; i < nFaces; i++) 
   {
      index  = fGlobalIDs[i];
      for (j = 0; j < nNodes; j++) 
         index = nGlobalIDLists[i][j];
   }
   printf("initFaceBlockNodeLists Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- find out how many distinct faces from elemFaceIDList
   // -------------------------------------------------------------

   currBlock->numLocalFaces_    = nFaces;
   currBlock->faceNumNodes_     = nNodes;
   currBlock->numExternalFaces_ = 0;
   currBlock->faceGlobalIDs_    = new int[nFaces];
   currBlock->faceNodeIDList_   = new int*[nFaces]; 
   faceArray = new int[nFaces];
   for ( i = 0; i < nFaces; i++ )
   {
      currBlock->faceGlobalIDs_[i]  = fGlobalIDs[i]; 
      currBlock->faceNodeIDList_[i] = NULL;
      faceArray[i]                  = i;
   } 
   MLI_Utils_IntQSort2(currBlock->faceGlobalIDs_, faceArray, 0, nFaces-1);

   // -------------------------------------------------------------
   // --- load the face Node list 
   // -------------------------------------------------------------

   faceNodeList = currBlock->faceNodeIDList_;
   for ( i = 0; i < nFaces; i++ )
   {
      index = faceArray[faceArray[i]];
      faceNodeList[index] = new int[nNodes];
      for ( j = 0; j < nNodes; j++ ) 
         faceNodeList[i][j] = nGlobalIDLists[index][j];
   }
   delete [] faceArray;
   return 1;
} 

//*************************************************************************
// initialize shared face list 
// (*** need to take into consideration of repeated face numbers in the
//      face list - for pairs, just as already been done in initSharedNodes)
//-------------------------------------------------------------------------

int MLI_FEData::initSharedFaces(int nFaces, const int *fGlobalIDs,
                       const int *numProcs, const int* const *procLists)
{
   int           i, j, index, *intArray;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   if ( nFaces <= 0 )
   {
      printf("initSharedFaces ERROR : nFaces <= 0.\n");
      exit(1);
   }
   currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->sharedFaceIDs_ != NULL )
      printf("initSharedFaces WARNING : already initialized (1) ?\n");
   if ( currBlock->sharedFaceNProcs_ != NULL )
      printf("initSharedFaces WARNING : already initialized (2) ?\n");
   if ( currBlock->sharedFaceProc_ != NULL )
      printf("initSharedFaces WARNING : already initialized (3) ?\n");

#ifdef MLI_DEBUG_DETAILED
   printf("initSharedFaces Diagnostics: segFault test.\n");
   for (i = 0; i < nFaces; i++) 
   {
      index  = fGlobalIDs[i];
      length = numProcs[i];
      for (j = 0; j < length; j++) index = procList[i][j];
   }
   printf("initSharedFaces Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate space for the incoming data 
   // -------------------------------------------------------------

   currBlock->numSharedFaces_   = nFaces;
   currBlock->sharedFaceIDs_    = new int[nFaces];
   currBlock->sharedFaceNProcs_ = new int[nFaces];
   currBlock->sharedFaceProc_   = new int*[nFaces];

   // -------------------------------------------------------------
   // --- load shared face information 
   // -------------------------------------------------------------

   intArray = new int[nFaces];
   for (i = 0; i < nFaces; i++) currBlock->sharedFaceIDs_[i] = fGlobalIDs[i];
   for (i = 0; i < nFaces; i++) intArray[i] = i;
   MLI_Utils_IntQSort2(currBlock->sharedFaceIDs_, intArray, 0, nFaces-1);

   for ( i = 0; i < nFaces; i++ )
   {
      index = intArray[i];
      if ( numProcs[index] <= 0 )
      {
         printf("initSharedFaces ERROR : numProcs not valid.\n");
         exit(1);
      }
      currBlock->sharedFaceNProcs_[i] = numProcs[index];
      currBlock->sharedFaceProc_[i]   = new int[numProcs[index]];
      for ( j = 0; j < numProcs[index]; j++ )
         currBlock->sharedFaceProc_[i][j] = procLists[index][j];
      MLI_Utils_IntQSort2(currBlock->sharedFaceProc_[i], NULL, 0, 
                          numProcs[index]-1);
   } 
   delete [] intArray;
   return 1;
}

//*************************************************************************
// initialization complete
//-------------------------------------------------------------------------

int MLI_FEData::initComplete()
{
   int           i, j, k, nElems, *elemList, totalNodes, *nodeArray, counter;
   int           index, temp_cnt, numSharedNodes, nExtNodes, elemNumNodes;
   int           *sharedNodeIDs, *sharedNodeNProcs, **sharedNodeProc;
   int           **elemNodeList, searchInd, elemNumFaces, numSharedFaces;
   int           *sharedFaceIDs, *sharedFaceNProcs, **sharedFaceProc, numProcs;
   int           mypid, totalFaces, nExtFaces, *faceArray, *procArray;
   int           **elemFaceList, *procArray2, *ownerP, *sndrcvReg, nProcs;
   int           nRecv, nSend, *recvProcs, *sendProcs, *recvLengs, *sendLengs;
   int           nNodes, pnum, **sendBuf, **recvBuf, *iauxArray, index2; 
   int           *intArray, **intArray2, nNodesPerElem, length, *nodeArrayAux;
   double        *dtemp_array, *nodeCoords; 
   MPI_Request   *request;
   MPI_Status    status;
   MLI_ElemBlock *currBlock;

   currBlock = elemBlockList_[currentElemBlock_];

   // -------------------------------------------------------------
   // --- check all element connectivities have been loaded
   // -------------------------------------------------------------

   nElems = currBlock->numLocalElems_;
   assert( nElems > 0 );
   elemList = currBlock->elemGlobalIDs_;
   if ( elemList == NULL )
   {
      printf("initComplete ERROR : initElemBlockNodeLists not called.\n");
      exit(1);
   }
   for ( i = 0; i < nElems; i++ )
   {
      if ( elemList[i] < 0 )
      {
         printf("initComplete ERROR : negative element ID.\n");
         exit(1);
      }
   }
   for ( i = 0; i < nElems; i++ )
   {
      for ( j = 0; j < currBlock->elemNumNodes_; j++ )
      {
         if ( currBlock->elemNodeIDList_[i][j] < 0 )
         {
            printf("initComplete ERROR : negative node ID.\n");
            exit(1);
         }
      }
   }

   // -------------------------------------------------------------
   // --- sort elemGlobalIDs in increasing order and shuffle
   // -------------------------------------------------------------

   currBlock->elemGlobalIDAux_ = new int[nElems];
   for ( i = 0; i < nElems; i++ ) currBlock->elemGlobalIDAux_[i] = i;
   MLI_Utils_IntQSort2(currBlock->elemGlobalIDs_,
                       currBlock->elemGlobalIDAux_, 0, nElems-1);

   // -------------------------------------------------------------
   // --- error checking (for duplicate element IDs)
   // -------------------------------------------------------------

   for ( i = 1; i < nElems; i++ ) 
   { 
      assert( currBlock->elemGlobalIDs_[i] >= 0 );
      if ( currBlock->elemGlobalIDs_[i] == currBlock->elemGlobalIDs_[i-1] )
      {
         printf("initComplete ERROR : duplicate elemIDs.\n");
         exit(1);
      }
   }

   // -------------------------------------------------------------
   // --- allocate storage and load for element node connectivities 
   // -------------------------------------------------------------

   nNodesPerElem = currBlock->elemNumNodes_;
   intArray2 = new int*[nElems];
   for ( i = 0; i < nElems; i++ ) intArray2[i] = new int[nNodesPerElem];
   for ( i = 0; i < nElems; i++ ) 
   {
      index = currBlock->elemGlobalIDAux_[i];
      intArray = currBlock->elemNodeIDList_[index];
      for ( j = 0; j < nNodesPerElem; j++ ) intArray2[i][j] = intArray[j]; 
   }   
   for ( i = 0; i < nElems; i++ ) delete [] currBlock->elemNodeIDList_[i];
   delete [] currBlock->elemNodeIDList_;
   currBlock->elemNodeIDList_ = intArray2;
   length = nNodesPerElem * spaceDimension_;
   if ( currBlock->nodeCoordinates_ != NULL )
   {
      nodeCoords = new double[length];
      for ( i = 0; i < nElems; i++ ) 
      {
         for ( j = 0; j < length; j++ ) 
         {
            index = currBlock->elemGlobalIDAux_[i];
            nodeCoords[i*length+j] =
               currBlock->nodeCoordinates_[index*length+j];
         }
      }
      delete [] currBlock->nodeCoordinates_;
      currBlock->nodeCoordinates_ = nodeCoords;
   }

   // -------------------------------------------------------------
   // --- compute element and nodal degrees of freedom
   // -------------------------------------------------------------

   currBlock->elemDOF_ = 0;
   for ( i = 0; i < currBlock->elemNumFields_; i++ )
      currBlock->elemNumFields_ += fieldSizes_[currBlock->elemFieldIDs_[i]];

   currBlock->nodeDOF_ = 0;
   for ( i = 0; i < currBlock->nodeNumFields_; i++ )
      currBlock->nodeDOF_ += fieldSizes_[currBlock->nodeFieldIDs_[i]];

   // -------------------------------------------------------------
   // --- obtain an ordered array of distinct node IDs
   // -------------------------------------------------------------

   elemNumNodes = currBlock->elemNumNodes_;
   elemNodeList = currBlock->elemNodeIDList_;
   temp_cnt     = nElems * elemNumNodes;
   nodeArray    = new int[temp_cnt];
   nodeArrayAux = new int[temp_cnt];
   totalNodes   = 0;
   for ( i = 0; i < nElems; i++ )
   {
      for ( j = 0; j < elemNumNodes; j++ )
         nodeArray[totalNodes++] = elemNodeList[i][j];
   }
   MLI_Utils_IntQSort2(nodeArray, NULL, 0, temp_cnt-1);
   totalNodes = 1;
   for ( i = 1; i < temp_cnt; i++ )
      if ( nodeArray[i] != nodeArray[totalNodes-1] )
         nodeArray[totalNodes++] = nodeArray[i];
   for ( i = 0; i < totalNodes; i++ ) nodeArrayAux[i] = nodeArray[i];

   // -------------------------------------------------------------
   // --- search for external nodes
   // -------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   numSharedNodes   = currBlock->numSharedNodes_;
   sharedNodeIDs    = currBlock->sharedNodeIDs_;
   sharedNodeNProcs = currBlock->sharedNodeNProcs_;
   sharedNodeProc   = currBlock->sharedNodeProc_;

   nExtNodes = 0;
   for ( i = 0; i < numSharedNodes; i++ )
   {
      for ( j = 0; j < sharedNodeNProcs[i]; j++ )
      {
         if ( sharedNodeProc[i][j] < mypid ) 
         {
            nExtNodes++;
            index = MLI_Utils_BinarySearch( sharedNodeIDs[i], nodeArray, 
                                            totalNodes);
            if ( index < 0 ) 
            {
               printf("initComplete ERROR : shared node not in elements.\n");
               printf("         %d\n", sharedNodeIDs[i]);
               for ( k = 0; k < totalNodes; k++ )
                  printf(" nodeArray = %d\n", nodeArray[k]);
               exit(1);
            }
            if ( nodeArrayAux[index] >= 0 )
               nodeArrayAux[index] = - nodeArrayAux[index] - 1;
            break;
         }
      }
   }

   // -------------------------------------------------------------
   // --- initialize the external nodes apart from internal
   // -------------------------------------------------------------

   nExtNodes = 0;
   for (i = 0; i < totalNodes; i++) if ( nodeArrayAux[i] < 0 ) nExtNodes++;
   currBlock->numExternalNodes_ = nExtNodes;
   currBlock->numLocalNodes_    = totalNodes - nExtNodes;
   currBlock->nodeGlobalIDs_    = new int[totalNodes];
   temp_cnt = 0;
   for (i = 0; i < totalNodes; i++) 
   {
      if ( nodeArrayAux[i] >= 0 ) 
         currBlock->nodeGlobalIDs_[temp_cnt++] = nodeArray[i];
   }
   for (i = 0; i < totalNodes; i++) 
   {
      if ( nodeArrayAux[i] < 0 ) 
         currBlock->nodeGlobalIDs_[temp_cnt++] = nodeArray[i];
   }
   delete [] nodeArray;
   delete [] nodeArrayAux;

   // -------------------------------------------------------------
   // --- create an aux array for holding mapped external node IDs
   // -------------------------------------------------------------

   MPI_Comm_size( mpiComm_, &nProcs );

   ownerP    = NULL;
   iauxArray = NULL;
   sndrcvReg = NULL;
   if ( nExtNodes > 0 ) ownerP = new int[nExtNodes];
   if ( nExtNodes > 0 ) iauxArray = new int[nExtNodes];
   if ( numSharedNodes > 0 ) sndrcvReg = new int[numSharedNodes];

   nNodes = currBlock->numLocalNodes_;
   for ( i = 0; i < numSharedNodes; i++ )
   {
      index = searchNode( sharedNodeIDs[i] ) - nNodes;
      if ( index >= nExtNodes )
      {
         printf("FEData initComplete ERROR : ext node ID construction.\n");
         exit(1);
      }
      if ( index >= 0 )
      {
         sndrcvReg[i] = 1; // recv
         pnum  = mypid;
         for ( j = 0; j < sharedNodeNProcs[i]; j++ )
            if ( sharedNodeProc[i][j] < pnum ) pnum = sharedNodeProc[i][j];
         ownerP[index] = pnum;
         iauxArray[index] = pnum;
      }
      else sndrcvReg[i] = 0; // send
   }

   nRecv     = 0;
   recvProcs = NULL;
   recvLengs = NULL;
   recvBuf   = NULL;

   MLI_Utils_IntQSort2( iauxArray, NULL, 0, nExtNodes-1);
   if ( nExtNodes > 0 ) nRecv = 1;
   for ( i = 1; i < nExtNodes; i++ )
      if (iauxArray[i] != iauxArray[nRecv-1]) 
         iauxArray[nRecv++] = iauxArray[i];
   if ( nRecv > 0 )
   {
      recvProcs = new int[nRecv];
      for ( i = 0; i < nRecv; i++ ) recvProcs[i] = iauxArray[i];
      recvLengs = new int[nRecv];
      for ( i = 0; i < nRecv; i++ ) recvLengs[i] = 0;
      for ( i = 0; i < nExtNodes; i++ ) 
      {
         index = MLI_Utils_BinarySearch( ownerP[i], recvProcs, nRecv );
         recvLengs[index]++;
      }
      recvBuf = new int*[nRecv];
      for ( i = 0; i < nRecv; i++ ) recvBuf[i] = new int[recvLengs[i]];
   }
   if ( nExtNodes > 0 ) delete [] iauxArray;

   counter = 0;
   for ( i = 0; i < numSharedNodes; i++ ) 
      if ( sndrcvReg[i] == 0 ) counter += sharedNodeNProcs[i];
   if ( counter > 0 ) iauxArray = new int[counter];
   counter = 0;
   for ( i = 0; i < numSharedNodes; i++ ) 
   {
      if ( sndrcvReg[i] == 0 ) 
      {
         for ( j = 0; j < sharedNodeNProcs[i]; j++ ) 
            if ( sharedNodeProc[i][j] != mypid )
               iauxArray[counter++] = sharedNodeProc[i][j];
      }
   }
   nSend     = 0;
   sendProcs = NULL;
   sendLengs = NULL;
   sendBuf   = NULL;
   if ( counter > 0 )
   {
      MLI_Utils_IntQSort2( iauxArray, NULL, 0, counter-1);
      nSend = 1;
      for ( i = 1; i < counter; i++ )
         if (iauxArray[i] != iauxArray[nSend-1]) 
            iauxArray[nSend++] = iauxArray[i];
      sendProcs = new int[nSend];
      for ( i = 0; i < nSend; i++ ) sendProcs[i] = iauxArray[i];
      sendLengs = new int[nSend];
      for ( i = 0; i < nSend; i++ ) sendLengs[i] = 0;
      for ( i = 0; i < numSharedNodes; i++ ) 
      {
         if ( sndrcvReg[i] == 0 ) 
         {
            for ( j = 0; j < sharedNodeNProcs[i]; j++ )
            {
               if ( sharedNodeProc[i][j] != mypid ) 
               {
                  index = sharedNodeProc[i][j];
                  index = MLI_Utils_BinarySearch( index, sendProcs, nSend );
                  sendLengs[index]++;
               }        
            }        
         }        
      }
      sendBuf = new int*[nSend];
      for ( i = 0; i < nSend; i++ ) sendBuf[i] = new int[sendLengs[i]];
      for ( i = 0; i < nSend; i++ ) sendLengs[i] = 0;
      for ( i = 0; i < numSharedNodes; i++ ) 
      {
         if ( sndrcvReg[i] == 0 ) 
         {
            for ( j = 0; j < sharedNodeNProcs[i]; j++ )
            {
               if ( sharedNodeProc[i][j] != mypid ) 
               {
                  index = sharedNodeProc[i][j];
                  index = MLI_Utils_BinarySearch( index, sendProcs, nSend );
                  index2 = searchNode( sharedNodeIDs[i] );
                  sendBuf[index][sendLengs[index]++] = 
                     currBlock->nodeOffset_ + index2;
               }        
            }        
         }        
      }        
   }
   if ( counter > 0 ) delete [] iauxArray;

   if ( nRecv > 0 ) request = new MPI_Request[nRecv];
   for ( i = 0; i < nRecv; i++ )
      MPI_Irecv( recvBuf[i], recvLengs[i], MPI_INT, 
                 recvProcs[i], 183, mpiComm_, &request[i]);
   for ( i = 0; i < nSend; i++ )
      MPI_Send( sendBuf[i], sendLengs[i], MPI_INT, 
                sendProcs[i], 183, mpiComm_);
   for ( i = 0; i < nRecv; i++ ) MPI_Wait( &request[i], &status );

   if ( nExtNodes > 0 ) currBlock->nodeExtNewGlobalIDs_ = new int[nExtNodes];
   for ( i = 0; i < nRecv; i++ ) recvLengs[i] = 0;
   for ( i = 0; i < nExtNodes; i++ ) 
   {
      index = MLI_Utils_BinarySearch( ownerP[i], recvProcs, nRecv );
      j = recvBuf[index][recvLengs[index]++];
      currBlock->nodeExtNewGlobalIDs_[i] = j;
   }
   if ( nExtNodes > 0 ) delete [] ownerP;
   if ( numSharedNodes > 0 ) delete [] sndrcvReg;
   if ( nRecv > 0 ) delete [] recvLengs;
   if ( nRecv > 0 ) delete [] recvProcs;
   for ( i = 0; i < nRecv; i++ ) delete [] recvBuf[i];
   if ( nRecv > 0 ) delete [] recvBuf;
   if ( nSend > 0 ) delete [] sendLengs;
   if ( nSend > 0 ) delete [] sendProcs;
   for ( i = 0; i < nSend; i++ ) delete [] sendBuf[i];
   if ( nSend > 0 ) delete [] sendBuf;
   if ( nRecv > 0 ) delete [] request;

   // -------------------------------------------------------------
   // --- now that the node list if finalized, shuffle the coordinates
   // -------------------------------------------------------------

   if ( currBlock->nodeCoordinates_ != NULL )
   {
      nodeArray   = currBlock->nodeGlobalIDs_;
      dtemp_array = currBlock->nodeCoordinates_;
      nodeCoords  = new double[totalNodes * spaceDimension_];

      for ( i = 0; i < nElems; i++ )
      {
         for ( j = 0; j < currBlock->elemNumNodes_; j++ )
         {
            index     = currBlock->elemNodeIDList_[i][j];
            searchInd = MLI_Utils_BinarySearch(index, nodeArray, 
                                               totalNodes-nExtNodes);
            if ( searchInd < 0 )
               searchInd = MLI_Utils_BinarySearch(index, 
                               &(nodeArray[totalNodes-nExtNodes]), 
                               nExtNodes) + totalNodes - nExtNodes;
            for ( k = 0; k < spaceDimension_; k++ )
               nodeCoords[searchInd*spaceDimension_+k] = 
                  dtemp_array[(i*elemNumNodes+j)*spaceDimension_+k];
         }
      }
      delete [] dtemp_array;
      currBlock->nodeCoordinates_ = nodeCoords;
   }

   // -------------------------------------------------------------
   // --- check for correctness in loading IDs
   // -------------------------------------------------------------

   if ( currBlock->elemFaceIDList_ != NULL )
   {
      elemNumFaces = currBlock->elemNumFaces_;
      elemFaceList = currBlock->elemFaceIDList_;
      temp_cnt     = nElems * elemNumFaces;
      faceArray    = new int[temp_cnt];
      totalFaces   = 0;
      for ( i = 0; i < nElems; i++ )
      {
         for ( j = 0; j < elemNumFaces; j++ )
            faceArray[totalFaces++] = elemFaceList[i][j];
      }
      MLI_Utils_IntQSort2(faceArray, NULL, 0, temp_cnt-1);
      totalFaces = 1;
      for ( i = 1; i < temp_cnt; i++ )
         if ( faceArray[i] != faceArray[i-1] )
            faceArray[totalFaces++] = faceArray[i];

      if ( totalFaces != currBlock->numLocalFaces_ && 
           currBlock->faceGlobalIDs_ == NULL )
      {
         printf("initComplete WARNING : face IDs not initialized.\n");
      }
      else if ( totalFaces != currBlock->numLocalFaces_ && 
                currBlock->faceGlobalIDs_ != NULL )
      {
         printf("initComplete ERROR : numbers of face do not match.\n");
         exit(1);
      }
      else 
      {
         delete [] currBlock->faceGlobalIDs_;
         currBlock->faceGlobalIDs_ = NULL;
      }
   }

   // -------------------------------------------------------------
   // --- search for external faces
   // -------------------------------------------------------------

   if ( currBlock->elemFaceIDList_ != NULL && currBlock->numSharedFaces_ > 0 )
   {
      numSharedFaces   = currBlock->numSharedFaces_;
      sharedFaceIDs    = currBlock->sharedFaceIDs_;
      sharedFaceNProcs = currBlock->sharedFaceNProcs_;
      sharedFaceProc   = currBlock->sharedFaceProc_;

      nExtFaces = 0;
      for ( i = 0; i < numSharedFaces; i++ )
      {
         for ( j = 0; j < sharedFaceNProcs[i]; j++ )
         {
            if ( sharedFaceProc[i][j] < mypid ) 
            {
               nExtFaces++;
               index = MLI_Utils_BinarySearch( sharedFaceIDs[i], faceArray, 
                                               totalFaces);
               if ( index < 0 ) 
               {
                  printf("initComplete ERROR : shared node not in elements.\n");
                  exit(1);
               }
               faceArray[index] = - faceArray[index];
               break;
            }
         }
      }

      currBlock->numExternalFaces_ = nExtFaces;
      currBlock->numLocalFaces_    = totalFaces - nExtFaces;
      currBlock->faceGlobalIDs_    = new int[totalFaces];
      temp_cnt = 0;
      for (i = 0; i < totalFaces; i++) 
      {
         if ( faceArray[i] >= 0 ) 
            currBlock->faceGlobalIDs_[temp_cnt++] = faceArray[i];
      }
      for (i = 0; i < totalFaces; i++) 
      {
         if ( faceArray[i] < 0 ) 
            currBlock->faceGlobalIDs_[temp_cnt++] = - faceArray[i];
      }
      delete [] faceArray;
   }

   // -------------------------------------------------------------
   // --- get element, node and face offsets 
   // -------------------------------------------------------------

   MPI_Comm_size( mpiComm_, &numProcs ); 
   procArray  = new int[numProcs];
   procArray2 = new int[numProcs];
   for ( i = 0; i < numProcs; i++ ) procArray2[i] = 0;
   procArray2[mypid] = currBlock->numLocalElems_;
   MPI_Allreduce(procArray2, procArray, numProcs, MPI_INT, MPI_SUM, mpiComm_);
   currBlock->elemOffset_ = 0;
   for ( i = 0; i < mypid; i++ ) currBlock->elemOffset_ += procArray[i];
   procArray2[mypid] = currBlock->numLocalNodes_ - currBlock->numExternalNodes_;
   MPI_Allreduce(procArray2, procArray, numProcs, MPI_INT, MPI_SUM, mpiComm_);
   currBlock->nodeOffset_ = 0;
   for ( i = 0; i < mypid; i++ ) currBlock->nodeOffset_ += procArray[i];
   procArray2[mypid] = currBlock->numLocalFaces_ - currBlock->numExternalFaces_;
   MPI_Allreduce(procArray2, procArray, numProcs, MPI_INT, MPI_SUM, mpiComm_);
   currBlock->faceOffset_ = 0;
   for ( i = 0; i < mypid; i++ ) currBlock->faceOffset_ += procArray[i];
   delete [] procArray;
   delete [] procArray2;

   // -------------------------------------------------------------
   // --- initialization complete
   // -------------------------------------------------------------

   currBlock->initComplete_ = 1;
   return 1;
}

//*************************************************************************
// load all element matrices at once
//-------------------------------------------------------------------------

int MLI_FEData::loadElemBlockMatrices(int nElems, int sMatDim,
                             const double* const *stiffMat)
{
   int           i, j, length, index;
   double        *row_darray;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("loadElemBlockMatrices ERROR : nElems mismatch.\n");
      exit(1);
   }
   if ( ! currBlock->initComplete_ ) 
   {
      printf("loadElemBlockMatrices ERROR : initialization not completed.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("loadElemBlockMatrices Diagnostics: segFault test.\n");
   double ddata;
   for (i = 0; i < nElems; i++) 
   {
      for (j = 0; j < sMatDim*sMatDim; j++) ddata = stiffMat[i][j];
   }
   printf("loadElemBlockMatrices Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate storage and load element stiffness matrices
   // -------------------------------------------------------------

   if ( sMatDim <= 0 || sMatDim > 200 )
   {
      printf("loadElemBlockMatrices ERROR : sMatDim invalid.\n");
      exit(1);
   }
   currBlock->elemStiffDim_ = sMatDim;
   currBlock->elemStiffMat_ = new double*[nElems];
   for ( i = 0; i < nElems; i++ ) 
   {
      length = sMatDim * sMatDim;
      currBlock->elemStiffMat_[i] = new double[length];
      index = currBlock->elemGlobalIDAux_[i];
      row_darray = currBlock->elemStiffMat_[i];
      for ( j = 0; j < length; j++ ) row_darray[j] = stiffMat[index][j]; 
   }
   return 1;
}

//*************************************************************************
// load element nullspace for all elements 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemBlockNullSpaces(int nElems, const int *nNSpace,
                    int sMatDim, const double* const *nSpace)
{
   int           i, j, index, length;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   (void) sMatDim; 

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("loadElemBlockNullSpaces ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( ! currBlock->initComplete_ ) 
   {
      printf("loadElemBlockNullSpaces ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->elemNullSpace_ == NULL || currBlock->elemNumNS_ == NULL )
   {
      currBlock->elemNullSpace_ = new double*[nElems];
      currBlock->elemNumNS_     = new int[nElems];
      for ( i = 0; i < nElems; i++ )
      {
         currBlock->elemNullSpace_[i] = NULL;
         currBlock->elemNumNS_[i]     = 0;
      }
   }
#ifdef MLI_DEBUG_DETAILED
   printf("loadElemBlockNullSpaces Diagnostics: segFault test.\n");
   double ddata;
   for (i = 0; i < nElems; i++) 
   {
      length = nNSpace[i];
      for (j = 0; j < sMatDim*length; j++) ddata = nSpace[i][j];
   }
   printf("loadElemBlockNullSpaces Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- load null space information
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      index = currBlock->elemGlobalIDAux_[i];
      currBlock->elemNumNS_[i] = nNSpace[index];
      length = currBlock->elemStiffDim_ * nNSpace[index];
      currBlock->elemNullSpace_[i] = new double[length];
      for ( j = 0; j < length; j++ )
         currBlock->elemNullSpace_[i][j] = nSpace[index][j];
   }
   return 1;
}

//*************************************************************************
// load element volumes for all elements 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemBlockVolumes(int nElems, const double *elemVols)
{
   int           i, index;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("loadElemBlockVolumes ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( ! currBlock->initComplete_ ) 
   {
      printf("loadElemBlockVolumes ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->elemVolume_ == NULL )
      currBlock->elemVolume_ = new double[nElems];

#ifdef MLI_DEBUG_DETAILED
   printf("loadElemBlockVolumes Diagnostics: segFault test.\n");
   double ddata;
   for (i = 0; i < nElems; i++) ddata = elemVols[i];
   printf("loadElemBlockVolumes Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- load element volume information
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      index = currBlock->elemGlobalIDAux_[i];
      currBlock->elemVolume_[i] = elemVols[index];
   }
   return 1;
}

//*************************************************************************
// load element material for all elements 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemBlockMaterials(int nElems, const int *elemMats)
{
   int           i, index;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("loadElemBlockMaterials ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( ! currBlock->initComplete_ ) 
   {
      printf("loadElemBlockMaterials ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->elemMaterial_ == NULL )
      currBlock->elemMaterial_ = new int[nElems];

#ifdef MLI_DEBUG_DETAILED
   printf("loadElemBlockMaterials Diagnostics: segFault test.\n");
   double ddata;
   for (i = 0; i < nElems; i++) ddata = elemVols[i];
   printf("loadElemBlockMaterials Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- load element material space information
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      index = currBlock->elemGlobalIDAux_[i];
      currBlock->elemMaterial_[i] = elemMats[index];
   }
   return 1;
}

//*************************************************************************
// load element parent IDs for all elements 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemBlockParentIDs(int nElems, const int *elemPIDs)
{
   int           i, index;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("loadElemBlockParentIDs ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( ! currBlock->initComplete_ ) 
   {
      printf("loadElemBlockParentIDs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->elemParentIDs_ == NULL )
      currBlock->elemParentIDs_ = new int[nElems];

#ifdef MLI_DEBUG_DETAILED
   printf("loadElemBlockParentIDs Diagnostics: segFault test.\n");
   for (i = 0; i < nElems; i++) index = elemPIDs[i];
   printf("loadElemBlockParentIDs Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- load element material space information
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      index = currBlock->elemGlobalIDAux_[i];
      currBlock->elemParentIDs_[i] = elemPIDs[index];
   }
   return 1;
}

//*************************************************************************
// load element load 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemBlockLoads(int nElems, int loadDim,
                        const double* const *elemLoads)
{
   int           i, j, index;
   double        *dble_array;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("loadElemBlockLoads ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( loadDim != currBlock->elemStiffDim_ )
   {
      printf("loadElemBlockLoads ERROR : loadDim invalid.\n");
      exit(1);
   }
   if ( ! currBlock->initComplete_ ) 
   {
      printf("loadElemBlockLoads ERROR : initialization not complete.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   double ddata;
   printf("loadElemBlockLoads Diagnostics: segFault test.\n");
   for (i = 0; i < nElems; i++) 
      for (j = 0; j < loadDim; j++) ddata = elemLoads[i][j];
   printf("loadElemBlockLoads Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate storage space
   // -------------------------------------------------------------

   if ( currBlock->elemLoads_ == NULL )
   {
      currBlock->elemLoads_ = new double*[nElems];
      for ( i = 0; i < nElems; i++ )
         currBlock->elemLoads_[i] = new double[loadDim];
   }

   // -------------------------------------------------------------
   // --- load face information
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      index = currBlock->elemGlobalIDAux_[i];
      dble_array = currBlock->elemLoads_[i];
      for ( j = 0; j < loadDim; j++ ) dble_array[j] = elemLoads[index][j];
   }
   return 1;
}

//*************************************************************************
// load element solution 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemBlockSolutions(int nElems, int solDim,
                                       const double* const *elemSols)
{
   int           i, j, index;
   double        *dble_array;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems != currBlock->numLocalElems_ )
   {
      printf("loadElemBlockSolutions ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( solDim != currBlock->elemStiffDim_ )
   {
      printf("loadElemBlockSolutions ERROR : solDim invalid.");
      exit(1);
   }
   if ( ! currBlock->initComplete_ ) 
   {
      printf("loadElemBlockSolutions ERROR : initialization not complete.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("loadElemBlockSolutions Diagnostics: segFault test.\n");
   double ddata;
   for (i = 0; i < nElems; i++) 
      for (j = 0; j < loadDim; j++) ddata = elemSols[i][j];
   printf("loadElemBlockSolutions Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate storage space
   // -------------------------------------------------------------

   if ( currBlock->elemSol_ == NULL )
   {
      currBlock->elemSol_ = new double*[nElems];
      for ( i = 0; i < nElems; i++ )
         currBlock->elemSol_[i] = new double[solDim];
   }

   // -------------------------------------------------------------
   // --- load face information
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      index = currBlock->elemGlobalIDAux_[i];
      dble_array = currBlock->elemSol_[i];
      for ( j = 0; j < solDim; j++ ) dble_array[j] = elemSols[index][j];
   }
   return 1;
}

//*************************************************************************
// load element boundary conditions 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemBCs(int nElems, const int *eGlobalIDs, 
                            int elemDOF, const char * const *BCFlags, 
                            const double *const *BCVals)

{
   int           i, j, elemDOFCheck;
   double        *bcData;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- initial checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nElems <= 0 )
   {
      printf("loadElemBCs ERROR : nElems <= 0.\n");
      exit(1);
   }
   elemDOFCheck = 0;
   for ( i = 0; i < currBlock->elemNumFields_; i++ )
      elemDOFCheck += fieldSizes_[currBlock->elemFieldIDs_[i]];
   if ( elemDOFCheck != elemDOF )
   {
      printf("loadElemBCs ERROR : element DOF not valid.\n");
      exit(1);
   }
   if ( ! currBlock->initComplete_ ) 
   {
      printf("loadElemBCs ERROR : initialization not complete.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("loadElemBCs Diagnostics: segFault test.\n");
   char   cdata;
   double ddata;
   for (i = 0; i < nElems; i++) 
   {
      j = eGlobalIDs[i];
      for (j = 0; j < elemDOF; j++) cdata = BCFlags[i][j];
      for (j = 0; j < elemDOF; j++) ddata = BCVals[i][j];
   }
   printf("loadElemBCs Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate storage space
   // -------------------------------------------------------------

   if ( currBlock->elemNumBCs_ == 0 )
   {
      currBlock->elemNumBCs_ = nElems;
      currBlock->elemBCIDList_   = new int[nElems];
      currBlock->elemBCFlagList_ = new char*[nElems];
      currBlock->elemBCValues_   = new double*[nElems];
      for ( i = 0; i < nElems; i++ )
      {
         currBlock->elemBCFlagList_[i] = new char[elemDOF];
         currBlock->elemBCValues_[i]   = new double[elemDOF];
      }
   }

   // -------------------------------------------------------------
   // --- load boundary information
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      currBlock->elemBCIDList_[i] = eGlobalIDs[i];
      bcData = currBlock->elemBCValues_[i];
      for ( j = 0; j < elemDOF; j++ ) 
      {
         bcData[j] = BCVals[i][j];
         currBlock->elemBCFlagList_[i][j] = BCFlags[i][j];
      }
   }
   return 1;
}

//*************************************************************************
// load element node list and stiffness matrix 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemMatrix(int eGlobalID, int eMatDim, 
                               const double *elemMat)
{
   int           i, j, index;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
#ifdef MLI_DEBUG_DETAILED
   if ( ! currBlock->intComplete_ ) 
   {
      printf("loadElemMatrix ERROR : initialization not complete.\n");
      exit(1);
   }
   if (currBlock->elemStiffMat_ != NULL && eMatDim != currBlock->elemStiffDim_) 
   {
      printf("loadElemMatrix ERROR : dimension mismatch.\n");
      exit(1);
   }
   if ( nNodesPerElem <= 0 )
   {
      printf("loadElemMatrix ERROR : NodesPerElem <= 0.");
      exit(1);
   }
#endif

   // -------------------------------------------------------------
   // --- allocate memory for stiffness matrix
   // -------------------------------------------------------------

   if ( currBlock->elemStiffMat_ == NULL )
   {
      currBlock->elemStiffMat_ = new double*[currBlock->numLocalElems_];
      for ( i = 0; i < currBlock->numLocalElems_; i++ )
         currBlock->elemStiffMat_[i] = NULL;
      currBlock->elemStiffDim_ = eMatDim;
   }
  
   // -------------------------------------------------------------
   // --- search for the data holder
   // -------------------------------------------------------------

   index = searchElement( eGlobalID );
#ifdef MLI_DEBUG_DETAILED
   if ( index < 0 )
   {
      printf("loadElemMatrix ERROR : invalid elementID %d\n", eGlobalID);
      exit(1);
   }
   if ( elemStiff_[index] != NULL )
   {
      printf("loadElemMatrix ERROR : element loaded before.\n");
      exit(1);
   }
#endif

   // -------------------------------------------------------------
   // --- search for the data holder
   // -------------------------------------------------------------

   currBlock->elemStiffMat_[index] = new double[eMatDim*eMatDim];
   for ( j = 0; j < eMatDim*eMatDim; j++ ) 
      currBlock->elemStiffMat_[index][j] = elemMat[j];
    
   return 1;
}

//*************************************************************************
// load element nullspace 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemNullSpace(int eGlobalID, int numNS, int eMatDim,
                                  const double *nSpace)
{
   int           i, nElems, index;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];

#ifdef MLI_DEBUG_DETAILED
   if ( ! currBlock->intComplete_ ) 
   {
      printf("loadElemNullSpace ERROR : initialization not complete.\n");
      exit(1);
   }
#endif

   // -------------------------------------------------------------
   // --- allocate storage
   // -------------------------------------------------------------

   nElems = currBlock->numLocalElems_;
   if ( currBlock->elemNullSpace_ == NULL || currBlock->elemNumNS_ == NULL )
   {
      currBlock->elemNullSpace_ = new double*[nElems];
      currBlock->elemNumNS_     = new int[nElems];
      for ( i = 0; i < nElems; i++ )
      {
         currBlock->elemNullSpace_[i] = NULL;
         currBlock->elemNumNS_[i]     = 0;
      }
   }

   // -------------------------------------------------------------
   // --- search for the data holder
   // -------------------------------------------------------------

   index = searchElement( eGlobalID );
#ifdef MLI_DEBUG_DETAILED
   if ( index < 0 )
   {
      printf("loadElemNullSpace ERROR : invalid elementID %d\n",eGlobalID);
      exit(1);
   }
#endif

   index = searchElement( eGlobalID );
#ifdef MLI_DEBUG_DETAILED
   if ( index < 0 )
   {
      printf("loadElemNullSpace ERROR : invalid element %d\n",elemGlobalID);
      exit(1);
   }
   if ( currBlock->elemNullSpace_[index] != NULL )
   {
      printf("loadElemNullSpace ERROR : NullSpace already initialized.\n");
      exit(1);
   }
#endif
   currBlock->elemNumNS_[index] = numNS;
   currBlock->elemNullSpace_[index] = new double[numNS*eMatDim];
   for ( i = 0; i < numNS*eMatDim; i++ )
      currBlock->elemNullSpace_[index][i] = nSpace[i];
   return 1;
}

//*************************************************************************
// load element load (right hand side) 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemLoad(int eGlobalID, int eMatDim,
                             const double *elemLoad)
{
   int           i, nElems, index;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];

#ifdef MLI_DEBUG_DETAILED
   if ( ! currBlock->intComplete_ ) 
   {
      printf("loadElemLoad ERROR : initialization not complete.\n");
      exit(1);
   }
#endif

   // -------------------------------------------------------------
   // --- allocate storage
   // -------------------------------------------------------------

   nElems = currBlock->numLocalElems_;
   if ( currBlock->elemLoads_ == NULL )
   {
      currBlock->elemLoads_ = new double*[nElems];
      for ( i = 0; i < nElems; i++ ) currBlock->elemLoads_[i] = NULL;
   }

   // -------------------------------------------------------------
   // --- search for the data holder
   // -------------------------------------------------------------

   index = searchElement( eGlobalID );
#ifdef MLI_DEBUG_DETAILED
   if ( index < 0 )
   {
      printf("loadElemLoad ERROR : invalid elementID %d\n", eGlobalID);
      exit(1);
   }
#endif

   // -------------------------------------------------------------
   // --- load data
   // -------------------------------------------------------------

#ifdef MLI_DEBUG_DETAILED
   if ( currBlock->elemLoads_[index] != NULL )
   {
      printf("loadElemLoad ERROR : element load already initialized.\n");
      exit(1);
   }
#endif
   currBlock->elemLoads_[index] = new double[eMatDim];
   for ( i = 0; i < eMatDim; i++ )
      currBlock->elemLoads_[index][i] = elemLoad[i];
   return 1;
}

//*************************************************************************
// load element solution 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemSolution(int eGlobalID, int eMatDim,
                                 const double *elemSol)
{
   int           i, nElems, index;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];

#ifdef MLI_DEBUG_DETAILED
   if ( ! currBlock->intComplete_ ) 
   {
      printf("loadElemSolution ERROR : initialization not complete.\n");
      exit(1);
   }
#endif

   // -------------------------------------------------------------
   // --- allocate storage
   // -------------------------------------------------------------

   nElems = currBlock->numLocalElems_;
   if ( currBlock->elemSol_ == NULL )
   {
      currBlock->elemSol_ = new double*[nElems];
      for ( i = 0; i < nElems; i++ ) currBlock->elemSol_[i] = NULL;
   }

   // -------------------------------------------------------------
   // --- search for the data holder
   // -------------------------------------------------------------

   index = searchElement( eGlobalID );
#ifdef MLI_DEBUG_DETAILED
   if ( index < 0 )
   {
      printf("loadElemSolution ERROR : invalid elementID %d\n",eGlobalID);
      exit(1);
   }
#endif

   // -------------------------------------------------------------
   // --- load data
   // -------------------------------------------------------------

#ifdef MLI_DEBUG_DETAILED
   if ( currBlock->elemSol_[index] != NULL )
   {
      printf("loadElemSolution ERROR : element load already initialized.\n");
      exit(1);
   }
#endif
   currBlock->elemSol_[index] = new double[eMatDim];
   for ( i = 0; i < eMatDim; i++ )
      currBlock->elemSol_[index][i] = elemSol[i];
   return 1;
}

//*************************************************************************
// set node boundary condition 
//-------------------------------------------------------------------------

int MLI_FEData::loadNodeBCs(int nNodes, const int *nodeIDs, int nodeDOF,
                            const char * const *BCFlags,
                            const double * const *BCVals)
{
   int           i, j, nodeDOFCheck;
   double        *bcData;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( nNodes <= 0 )
   {
      printf("loadNodeBCs ERROR : nNodes <= 0.\n");
      exit(1);
   }
   nodeDOFCheck = 0;
   for ( i = 0; i < currBlock->nodeNumFields_; i++ )
      nodeDOFCheck += fieldSizes_[currBlock->nodeFieldIDs_[i]];
   if ( nodeDOFCheck != nodeDOF )
   {
      printf("loadNodeBCs ERROR : node DOF not valid.\n");
      exit(1);
   }
   if ( ! currBlock->initComplete_ )
   {
      printf("loadNodeBCs ERROR : initialization not complete.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("loadNodeBCs Diagnostics: segFault test.\n");
   char   cdata;
   double ddata;
   for (i = 0; i < nNodes; i++) 
   {
      j = nodeIDs[i];
      for (j = 0; j < nodeDOF; j++) cdata = BCFlags[i][j];
      for (j = 0; j < nodeDOF; j++) ddata = BCVals[i][j];
   }
   printf("loadNodeBCs Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- allocate storage space
   // -------------------------------------------------------------

   if ( currBlock->nodeNumBCs_ == 0 )
   {
      currBlock->nodeNumBCs_      = nNodes;
      currBlock->nodeBCIDList_    = new int[nNodes];
      currBlock->nodeBCFlagList_  = new char*[nNodes];
      currBlock->nodeBCValues_    = new double*[nNodes];
      for ( i = 0; i < nNodes; i++ )
      {
         currBlock->nodeBCFlagList_[i] = new char[nodeDOF];
         currBlock->nodeBCValues_[i]   = new double[nodeDOF];
      }
   }

   // -------------------------------------------------------------
   // --- load boundary information
   // -------------------------------------------------------------

   for ( i = 0; i < nNodes; i++ )
   {
      currBlock->nodeBCIDList_[i] = nodeIDs[i];
      bcData = currBlock->nodeBCValues_[i];
      for ( j = 0; j < nodeDOF; j++ )
      {
         bcData[j] = BCVals[i][j];
         currBlock->nodeBCFlagList_[i][j] = BCFlags[i][j];
      }
   }
   return 1;
}

//*************************************************************************
// get dimension of physical problem
//-------------------------------------------------------------------------

int MLI_FEData::getSpaceDimension(int& numDim)
{
   numDim = spaceDimension_;
   return 1;
}

//*************************************************************************
// get order of PDE 
//-------------------------------------------------------------------------

int MLI_FEData::getOrderOfPDE(int& order)
{
   order = orderOfPDE_;
   return 1;
}

//*************************************************************************
// get order of FE 
//-------------------------------------------------------------------------

int MLI_FEData::getOrderOfFE(int& order)
{
   order = orderOfFE_;
   return 1;
}

//*************************************************************************
// get field size  
//-------------------------------------------------------------------------

int MLI_FEData::getFieldSize(int fieldID, int& fieldSize)
{
   fieldSize = 0;
   for ( int i = 0; i < numFields_; i++ )
      if ( fieldIDs_[i] == fieldID ) fieldSize = fieldSizes_[i];
   if ( fieldSize > 0 ) return 1;
   else                 return 0;
}

//*************************************************************************
// get number of local elements 
//-------------------------------------------------------------------------

int MLI_FEData::getNumElements(int& nelems)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   nelems = currBlock->numLocalElems_;
   return 1;
}

//*************************************************************************
// get element's number of fields
//-------------------------------------------------------------------------

int MLI_FEData::getElemNumFields(int& numFields)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   numFields = currBlock->elemNumFields_;
   return 1;
}

//*************************************************************************
// get element's field IDs
//-------------------------------------------------------------------------

int MLI_FEData::getElemFieldIDs(int numFields, int *fieldIDs)
{
#ifdef MLI_DEBUG_DETAILED
   printf("getElemFieldIDs Diagnostics: segFault test.\n");
   for (int i = 0; i < numFields; i++) fieldIDs[i] = 0;
   printf("getElemFieldIDs Diagnostics: passed the segFault test.\n");
#endif

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   for ( int j = 0; j < numFields; j++ )
      fieldIDs[j] = currBlock->elemFieldIDs_[j];
   return 1;
}

//*************************************************************************
// get an element globalID 
//-------------------------------------------------------------------------

int MLI_FEData::getElemGlobalID(int localID, int &globalID)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
#ifdef MLI_DEBUG_DETAILED
   if ( ! currBlock->initComplete_ )
   {
      printf("getElemGlobalID ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ < localID )
   {
      printf("getElemGlobalID ERROR : invalid local ID.\n");
      exit(1);
   }
#endif
   globalID = currBlock->elemGlobalIDs_[localID];
   return 1;
}

//*************************************************************************
// get all element globalIDs 
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockGlobalIDs(int nElems, int *eGlobalIDs)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getElemGlobalID ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockGlobalIDs ERROR : nElems mismatch.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("getElemBlockGlobalIDs Diagnostics: segFault test.\n");
   for (int i = 0; i < nElems; i++) eGlobalIDs[i] = 0;
   printf("getElemBlockGlobalIDs Diagnostics: passed the segFault test.\n");
#endif

   for ( int j = 0; j < nElems; j++ ) 
      eGlobalIDs[j] = currBlock->elemGlobalIDs_[j];
   return 1;
}

//*************************************************************************
// get element number of nodes
//-------------------------------------------------------------------------

int MLI_FEData::getElemNumNodes(int& nNodes)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   nNodes = currBlock->elemNumNodes_;
   return 1;
}

//*************************************************************************
// get element block nodelists 
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockNodeLists(int nElems, int nNodes, int **nodeList)
{
   int i, j;
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemBlockNodeLists ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockNodeLists ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( currBlock->elemNumNodes_ != nNodes )
   {
      printf("getElemBlockNodeLists ERROR : elemNumNodes do not match.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("getElemBlockNodeLists Diagnostics: segFault test.\n");
   for (i = 0; i < nElems; i++) 
      for (j = 0; j < nNodes; j++) nodeList[i][j] = 0;
   printf("getElemBlockNodeLists Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- get nodelists
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      for ( j = 0; j < nNodes; j++ )
         nodeList[i][j] = currBlock->elemNodeIDList_[i][j];
   }
   return 1;
}

//*************************************************************************
// get element matrices' dimension 
//-------------------------------------------------------------------------

int MLI_FEData::getElemMatrixDim(int& matDim)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   matDim = currBlock->elemStiffDim_;
   return 1;
}

//*************************************************************************
// get all element stiffness matrices 
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockMatrices(int nElems,int eMatDim,double **elemMat)
{
   int    i, j;
   double *outMat, *stiffMat;

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemBlockMatrices ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockMatrices ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( currBlock->elemStiffDim_ != eMatDim )
   {
      printf("getElemBlockMatrices ERROR : matrix dimension do not match.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("getElemBlockMatrices Diagnostics: segFault test.\n");
   for (i = 0; i < nElems; i++) 
      for (j = 0; j < eMatDim*eMatDim; j++) elemMat[i][j] = 0.0;
   printf("getElemBlockMatrices Diagnostics: passed the segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- get element matrices
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ )
   {
      if ( currBlock->elemStiffMat_[i] == NULL )
      {
         printf("getElemBlockMatrices ERROR : elemMat not initialized.\n");
         exit(1);
      }
      outMat   = elemMat[i];
      stiffMat = currBlock->elemStiffMat_[i];
      for ( j = 0; j < eMatDim*eMatDim; j++ ) outMat[j] = stiffMat[j];
   }
   return 1;
}

//*************************************************************************
// get all element nullspace sizes
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockNullSpaceSizes(int nElems, int *dimNS)
{
   int i;
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemBlockNullSpaceSizes ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockNullSpaceSizes ERROR : nElems do not match.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("getElemBlockNullSpaceSizes Diagnostics: segFault test.\n");
   for (i = 0; i < nElems; i++) dimNS[i] = 0; 
   printf("getElemBlockNullSpaceSizes Diagnostics: passed segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- load nullspace sizes
   // -------------------------------------------------------------

   if ( currBlock->elemNumNS_ == NULL )
      for ( i = 0; i < nElems; i++ ) dimNS[i] = 0;
   else
      for ( i = 0; i < nElems; i++ ) dimNS[i] = currBlock->elemNumNS_[i];

   return 1;
}

//*************************************************************************
// get all element nullspaces 
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockNullSpaces(int nElems, const int *dimNS, 
                                       int eMatDim, double **nullSpaces)
{
   int i,j;
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemBlockNullSpaces ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockNullSpaces ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( currBlock->elemStiffDim_ == eMatDim )
   {
      printf("getElemBlockNullSpaces ERROR : eMatDim do not match.\n");
      exit(1);
   }
   if ( currBlock->elemNumNS_ == NULL )
   {
      printf("getElemBlockNullSpaces ERROR : no null space information.\n");
      exit(1);
   }

#ifdef MLI_DEBUG_DETAILED
   printf("getElemBlockNullSpaces Diagnostics: segFault test.\n");
   for (i = 0; i < nElems; i++) 
      for (j = 0; j < dimNS[i]*eMatDim; j++) nullSpaces[i][j] = 0.0; 
   printf("getElemBlockNullSpaces Diagnostics: passed segFault test.\n");
#endif

   // -------------------------------------------------------------
   // --- load nullspace sizes
   // -------------------------------------------------------------

   for ( i = 0; i < nElems; i++ ) 
   {
      if ( dimNS[i] != currBlock->elemNumNS_[i] )
      {
         printf("getElemBlockNullSpaces ERROR : dimension do not match.\n");
         exit(1);
      }
      for ( j = 0; j < eMatDim*dimNS[i]; j++ ) 
         nullSpaces[i][j] = currBlock->elemNullSpace_[i][j];
   }
   return 1;
}

//*************************************************************************
// get all element volumes 
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockVolumes(int nElems, double *elemVols)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemBlockVolumes ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockVolumes ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( currBlock->elemVolume_ == NULL )
   {
      printf("getElemBlockVolumes ERROR : no volumes available.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load element volumes 
   // -------------------------------------------------------------

   for ( int i = 0; i < nElems; i++ ) elemVols[i] = currBlock->elemVolume_[i];

   return 1;
}

//*************************************************************************
// get all element materials 
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockMaterials(int nElems, int *elemMats)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemBlockMaterials ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockMaterials ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( currBlock->elemMaterial_ == NULL )
   {
      printf("getElemBlockMaterials ERROR : no material available.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load element materials 
   // -------------------------------------------------------------

   for (int i = 0; i < nElems; i++) elemMats[i] = currBlock->elemMaterial_[i];

   return 1;
}

//*************************************************************************
// get all element parent IDs 
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockParentIDs(int nElems, int *parentIDs)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemBlockParentIDs ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockParentIDs ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( currBlock->elemParentIDs_ == NULL )
   {
      printf("getElemBlockParentIDs ERROR : no parent ID available.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load element parent IDs 
   // -------------------------------------------------------------

   for (int i = 0; i < nElems; i++) parentIDs[i] = currBlock->elemParentIDs_[i];

   return 1;
}

//*************************************************************************
// get element number of faces
//-------------------------------------------------------------------------

int MLI_FEData::getElemNumFaces(int& nFaces)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   nFaces = currBlock->elemNumFaces_;
   return 1;
}

//*************************************************************************
// get all element face lists
//-------------------------------------------------------------------------

int MLI_FEData::getElemBlockFaceLists(int nElems, int nFaces, int **faceList)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemBlockFaceLists ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->numLocalElems_ != nElems )
   {
      printf("getElemBlockFaceLists ERROR : nElems do not match.\n");
      exit(1);
   }
   if ( currBlock->elemNumFaces_ != nFaces )
   {
      printf("getElemBlockFaceLists ERROR : elemNumFaces do not match.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get face lists
   // -------------------------------------------------------------

   for ( int i = 0; i < nElems; i++ )
   {
      for ( int j = 0; j < nFaces; j++ )
         faceList[i][j] = currBlock->elemFaceIDList_[i][j];
   }
   return 1;
}

//*************************************************************************
// get element node list given an element global ID
//-------------------------------------------------------------------------

int MLI_FEData::getElemNodeList(int eGlobalID, int nNodes, int *nodeList)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemNodeList ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->elemNumNodes_ != nNodes )
   {
      printf("getElemNodeList ERROR : elemNumNodes do not match.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get node list
   // -------------------------------------------------------------

   int index = searchElement(eGlobalID);
   if ( index < 0 )
   {
      printf("getElemNodeList ERROR : element not found.\n");
      exit(1);
   }
   for ( int i = 0; i < nNodes; i++ )
      nodeList[i] = currBlock->elemNodeIDList_[index][i];
   return 1;
}

//*************************************************************************
// get an element matrix 
//-------------------------------------------------------------------------

int MLI_FEData::getElemMatrix(int eGlobalID, int eMatDim, double *elemMat)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemMatrix ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->elemStiffDim_ != eMatDim )
   {
      printf("getElemMatrix ERROR : matrix dimension do not match.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load element matrix
   // -------------------------------------------------------------

   int index = searchElement(eGlobalID);
   if ( index < 0 )
   {
      printf("getElemMatrix ERROR : element not found.\n");
      exit(1);
   }
   if ( currBlock->elemStiffMat_[index] == NULL )
   {
      printf("getElemBlockMatrix ERROR : elemMat not initialized.\n");
      exit(1);
   }
   double *stiffMat = currBlock->elemStiffMat_[index];
   for ( int i = 0; i < eMatDim*eMatDim; i++ ) elemMat[i] = stiffMat[i];

   return 1;
}

//*************************************************************************
// get an element nullspace size
//-------------------------------------------------------------------------

int MLI_FEData::getElemNullSpaceSize(int eGlobalID, int &dimNS)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemNullSpaceSize ERROR : not initialized.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load nullspace size
   // -------------------------------------------------------------

   int index = searchElement(eGlobalID);
   if ( index < 0 )
   {
      printf("getElemNullSpaceSize ERROR : element not found.\n");
      exit(1);
   }
   if ( currBlock->elemNumNS_ == NULL ) dimNS = 0;
   else                                 dimNS = currBlock->elemNumNS_[index];

   return 1;
}

//*************************************************************************
// get an element nullspace 
//-------------------------------------------------------------------------

int MLI_FEData::getElemNullSpace(int eGlobalID, int dimNS, int eMatDim, 
                                 double *nullSpaces)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemNullSpace ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->elemStiffDim_ == eMatDim )
   {
      printf("getElemNullSpace ERROR : eMatDim do not match.\n");
      exit(1);
   }
   if ( currBlock->elemNumNS_ == NULL )
   {
      printf("getElemNullSpace ERROR : no null space information.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load nullspace sizes
   // -------------------------------------------------------------

   int index = searchElement(eGlobalID);
   if ( index < 0 )
   {
      printf("getElemNullSpace ERROR : element not found.\n");
      exit(1);
   }
   for ( int i = 0; i < eMatDim*dimNS; i++ ) 
      nullSpaces[i] = currBlock->elemNullSpace_[index][i];
   return 1;
}

//*************************************************************************
// get an element volume 
//-------------------------------------------------------------------------

int MLI_FEData::getElemVolume(int eGlobalID, double &elemVol)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemVolume ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->elemVolume_ == NULL )
   {
      printf("getElemVolumes ERROR : no volumes available.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load element volumes 
   // -------------------------------------------------------------

   int index = searchElement(eGlobalID);
   if ( index < 0 )
   {
      printf("getElemVolume ERROR : element not found.\n");
      exit(1);
   }
   elemVol = currBlock->elemVolume_[index];

   return 1;
}

//*************************************************************************
// get an element material 
//-------------------------------------------------------------------------

int MLI_FEData::getElemMaterial(int eGlobalID, int &elemMat)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemMaterial ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->elemMaterial_ == NULL )
   {
      printf("getElemMaterial ERROR : no material available.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load element material 
   // -------------------------------------------------------------

   int index = searchElement(eGlobalID);
   if ( index < 0 )
   {
      printf("getElemMaterial ERROR : element not found.\n");
      exit(1);
   }
   elemMat = currBlock->elemMaterial_[index];

   return 1;
}

//*************************************************************************
// get all element parent IDs 
//-------------------------------------------------------------------------

int MLI_FEData::getElemParentID(int eGlobalID, int &parentID)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemParentID ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->elemParentIDs_ == NULL )
   {
      printf("getElemParentID ERROR : no parent ID available.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load element parent IDs 
   // -------------------------------------------------------------

   int index = searchElement(eGlobalID);
   if ( index < 0 )
   {
      printf("getElemParentId ERROR : element not found.\n");
      exit(1);
   }
   parentID = currBlock->elemParentIDs_[index];

   return 1;
}

//*************************************************************************
// get an element's face list
//-------------------------------------------------------------------------

int MLI_FEData::getElemFaceList(int eGlobalID, int nFaces, int *faceList)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( currBlock->initComplete_ != 1 )
   {
      printf("getElemFaceList ERROR : not initialized.\n");
      exit(1);
   }
   if ( currBlock->elemNumFaces_ != nFaces )
   {
      printf("getElemFaceList ERROR : elemNumFaces do not match.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get face list
   // -------------------------------------------------------------

   int index = searchElement(eGlobalID);
   if ( index < 0 )
   {
      printf("getElemFaceList ERROR : element not found.\n");
      exit(1);
   }
   for ( int i = 0; i < nFaces; i++ )
      faceList[i] = currBlock->elemFaceIDList_[index][i];
   return 1;
}

//*************************************************************************
// get number of boundary elements
//-------------------------------------------------------------------------

int MLI_FEData::getNumBCElems(int& nElems)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   nElems = currBlock->elemNumBCs_;
   return 1;
}

//*************************************************************************
// get number of boundary elements
//-------------------------------------------------------------------------

int MLI_FEData::getElemBCs(int nElems, int *eGlobalIDs, int eDOFs, 
                           char **fieldFlag, double **BCVals)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getElemBCs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->elemNumBCs_ != nElems )
   {
      printf("getElemBCs ERROR : nElems mismatch.\n");
      exit(1);
   }
   if ( eDOFs != currBlock->elemDOF_ )
   {
      printf("getElemBCs ERROR : element DOF mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load information
   // -------------------------------------------------------------

   for ( int i = 0; i < nElems; i++ )
   {
      eGlobalIDs[i] = currBlock->elemBCIDList_[i];
      for ( int j = 0; j < eDOFs; j++ )
      {
         fieldFlag[i][j] = currBlock->elemBCFlagList_[i][j];
         BCVals[i][j] = currBlock->elemBCValues_[i][j];
      }
   }
   return 1;
}

//*************************************************************************
// get number of total nodes (local + external) 
//-------------------------------------------------------------------------

int MLI_FEData::getNumNodes(int& nNodes)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   nNodes = currBlock->numLocalNodes_ + currBlock->numExternalNodes_;
   return 1;
}

//*************************************************************************
// get all node globalIDs 
//-------------------------------------------------------------------------

int MLI_FEData::getNodeBlockGlobalIDs(int nNodes, int *nGlobalIDs)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getNodeBlockGlobalIDs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( (currBlock->numLocalNodes_+currBlock->numExternalNodes_) != nNodes )
   {
      printf("getNodeBlockGlobalIDs ERROR : nNodes mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get nodal global IDs
   // -------------------------------------------------------------

   for (int i = 0; i < nNodes; i++) 
      nGlobalIDs[i] = currBlock->nodeGlobalIDs_[i];
   return 1;
}

//*************************************************************************
// get node's number of fields
//-------------------------------------------------------------------------

int MLI_FEData::getNodeNumFields(int &numFields)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   numFields = currBlock->nodeNumFields_;
   return 1;
}

//*************************************************************************
// get node's field IDs
//-------------------------------------------------------------------------

int MLI_FEData::getNodeFieldIDs(int numFields, int *fieldIDs)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   for ( int i = 0; i < numFields; i++ )
      fieldIDs[i] = currBlock->nodeFieldIDs_[i];
   return 1;
}

//*************************************************************************
// get all node coordinates 
//-------------------------------------------------------------------------

int MLI_FEData::getNodeBlockCoordinates(int nNodes, int spaceDim,
                                        double *coordinates)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getNodeBlockCoordinates ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( (currBlock->numLocalNodes_+currBlock->numExternalNodes_) != nNodes )
   {
      printf("getNodeBlockCoordinates ERROR : nNodes mismatch.\n");
      exit(1);
   }
   if ( spaceDimension_ != spaceDim )
   {
      printf("getNodeBlockCoordinates ERROR : space dimension mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get nodal coordinates
   // -------------------------------------------------------------

   for (int i = 0; i < nNodes*spaceDim; i++) 
      coordinates[i] = currBlock->nodeCoordinates_[i];
   return 1;
}

//*************************************************************************
// get number of boundary nodes
//-------------------------------------------------------------------------

int MLI_FEData::getNumBCNodes(int& nNodes)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   nNodes = currBlock->nodeNumBCs_;
   return 1;
}

//*************************************************************************
// get number of boundary nodes
//-------------------------------------------------------------------------

int MLI_FEData::getNodeBCs(int nNodes, int *nGlobalIDs, int nDOFs, 
                           char **fieldFlag, double **BCVals)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getNodeBCs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->nodeNumBCs_ != nNodes )
   {
      printf("getNodeBCs ERROR : nNodes mismatch.\n");
      exit(1);
   }
   if ( nDOFs != currBlock->nodeDOF_ )
   {
      printf("getNodeBCs ERROR : nodal DOF mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- load information
   // -------------------------------------------------------------

   for ( int i = 0; i < nNodes; i++ )
   {
      nGlobalIDs[i] = currBlock->nodeBCIDList_[i];
      for ( int j = 0; j < nDOFs; j++ )
      {
         fieldFlag[i][j] = currBlock->nodeBCFlagList_[i][j];
         BCVals[i][j] = currBlock->nodeBCValues_[i][j];
      }
   }
   return 1;
}

//*************************************************************************
// get number of shared nodes
//-------------------------------------------------------------------------

int MLI_FEData::getNumSharedNodes(int& nNodes)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   nNodes = currBlock->numSharedNodes_;
   return 1;
}

//*************************************************************************
// get shared nodes number of processor information
//-------------------------------------------------------------------------

int MLI_FEData::getSharedNodeNumProcs(int nNodes, int *nGlobalIDs,
                                      int *numProcs)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getSharedNodeNumProcs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->numSharedNodes_ != nNodes )
   {
      printf("getSharedNodeNumProcs ERROR : nNodes mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get information 
   // -------------------------------------------------------------

   for ( int i = 0; i < nNodes; i++ )
   {
      nGlobalIDs[i] = currBlock->sharedNodeIDs_[i];
      numProcs[i]   = currBlock->sharedNodeNProcs_[i];
   }
   return 1;
}

//*************************************************************************
// get shared nodes processor lists
//-------------------------------------------------------------------------

int MLI_FEData::getSharedNodeProcs(int nNodes, int *numProcs,
                                      int **procLists)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getSharedNodeProcs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->numSharedNodes_ != nNodes )
   {
      printf("getSharedNodeProcs ERROR : nNodes mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get information 
   // -------------------------------------------------------------

   for ( int i = 0; i < nNodes; i++ )
   {
      if ( numProcs[i] != currBlock->sharedNodeNProcs_[i] )
      {
         printf("NumSharedNodeProcs ERROR : numProcs mismatch.\n");
         exit(1);
      }
      for ( int j = 0; j < numProcs[i]; j++ )
         procLists[i][j] = currBlock->sharedNodeProc_[i][j];
   }
   return 1;
}

//*************************************************************************
// get number of faces 
//-------------------------------------------------------------------------

int MLI_FEData::getNumFaces(int &nFaces)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getNumFaces ERROR : initialization not complete.\n");
      exit(1);
   }
   nFaces = currBlock->numLocalFaces_ + currBlock->numExternalFaces_;
   return 1;
}

//*************************************************************************
// get all face globalIDs 
//-------------------------------------------------------------------------

int MLI_FEData::getFaceBlockGlobalIDs(int nFaces, int *fGlobalIDs)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getFaceBlockGlobalIDs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( (currBlock->numLocalFaces_+currBlock->numExternalFaces_) != nFaces )
   {
      printf("getFaceBlockGlobalIDs ERROR : nFaces mismatch.\n");
      exit(1);
   }
   for ( int i = 0; i < nFaces; i++ ) 
      fGlobalIDs[i] = currBlock->faceGlobalIDs_[i];
   return 1;
}

//*************************************************************************
// get number of shared faces 
//-------------------------------------------------------------------------

int MLI_FEData::getNumSharedFaces(int &nFaces)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getNumSharedFaces ERROR : initialization not complete.\n");
      exit(1);
   }
   nFaces = currBlock->numSharedFaces_;
   return 1;
}

//*************************************************************************
// get shared faces number of processor information
//-------------------------------------------------------------------------

int MLI_FEData::getSharedFaceNumProcs(int nFaces, int *fGlobalIDs,
                                      int *numProcs)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getSharedFaceNumProcs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->numSharedFaces_ != nFaces )
   {
      printf("getSharedFaceNumProcs ERROR : nFaces mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get information 
   // -------------------------------------------------------------

   for ( int i = 0; i < nFaces; i++ )
   {
      fGlobalIDs[i] = currBlock->sharedFaceIDs_[i];
      numProcs[i]   = currBlock->sharedFaceNProcs_[i];
   }
   return 1;
}

//*************************************************************************
// get shared face processor lists
//-------------------------------------------------------------------------

int MLI_FEData::getSharedFaceProcs(int nFaces, int *numProcs,
                                   int **procLists)
{
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getSharedFaceProcs ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->numSharedFaces_ != nFaces )
   {
      printf("getSharedFaceProcs ERROR : nFaces mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- get information 
   // -------------------------------------------------------------

   for ( int i = 0; i < nFaces; i++ )
   {
      if ( numProcs[i] != currBlock->sharedFaceNProcs_[i] )
      {
         printf("NumSharedFaceProcs ERROR : numProcs mismatch.\n");
         exit(1);
      }
      for ( int j = 0; j < numProcs[i]; j++ )
         procLists[i][j] = currBlock->sharedFaceProc_[i][j];
   }
   return 1;
}

//*************************************************************************
// get number of nodes on a face
//-------------------------------------------------------------------------

int MLI_FEData::getFaceNumNodes(int &nNodes)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("getFaceNumNodes ERROR : initialization not complete.\n");
      exit(1);
   }
   nNodes = currBlock->faceNumNodes_;
   return 1;
}

//*************************************************************************
// get block face node list 
//-------------------------------------------------------------------------

int MLI_FEData::getFaceBlockNodeLists(int nFaces, int nNodesPerFace,
                                      int **nGlobalIDLists)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   if ( ! currBlock->initComplete_ )
   {
      printf("getFaceBlockNodeLists ERROR : initialization not complete.\n");
      exit(1);
   }
   if ((currBlock->numLocalFaces_+currBlock->numExternalFaces_) != nFaces)
   {
      printf("getFaceBlockNodeLists ERROR : number of faces mismatch.\n");
      exit(1);
   }
   if ( currBlock->faceNumNodes_ != nNodesPerFace )
   {
      printf("getFaceBlockNodeLists ERROR : face numNodes mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- search face and get face node list
   // -------------------------------------------------------------

   for ( int i = 0; i < nFaces; i++ )
      for ( int j = 0; j < nNodesPerFace; j++ )
         nGlobalIDLists[i][j] = currBlock->faceNodeIDList_[i][j];
   
   return 1;
}

//*************************************************************************
// get face node list 
//-------------------------------------------------------------------------

int MLI_FEData::getFaceNodeList(int fGlobalID, int nNodes, int *nodeList)
{
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   if ( ! currBlock->initComplete_ )
   {
      printf("getFaceNodeList ERROR : initialization not complete.\n");
      exit(1);
   }
   if ( currBlock->faceNumNodes_ != nNodes )
   {
      printf("getFaceNodeList ERROR : face numNodes mismatch.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- search face and get face node list
   // -------------------------------------------------------------

   int index = searchFace( fGlobalID );
   if ( index < 0 )
   {
      printf("getFaceNodeList ERROR : face ID not found.\n");
      exit(1);
   }
   for ( int i = 0; i < nNodes; i++ )
      nodeList[i] = currBlock->faceNodeIDList_[index][i];
   
   return 1;
}

//*************************************************************************
// load in the function to calculate shape function interpolant 
//-------------------------------------------------------------------------

int MLI_FEData::loadFunc_computeShapeFuncInterpolant(void *object, int (*func)
                (void*,int elemID,int nNodes,const double *coord,double *coef))
{
   USR_FEGridObj_ = object;
   USR_computeShapeFuncInterpolant = func;
   return 1;
}

//*************************************************************************
// get shape function interpolant 
//-------------------------------------------------------------------------

int MLI_FEData::getShapeFuncInterpolant(int elemID, int nNodes, 
                   const double *coord, double *coef)
{
   USR_computeShapeFuncInterpolant(USR_FEGridObj_, elemID, nNodes,
                                   coord,coef);
   return 1;
}

//*************************************************************************
// implementation specific requests
//-------------------------------------------------------------------------

int MLI_FEData::impSpecificRequests(char *data_key, int argc, char **argv)
{
   int           mypid, nprocs;
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];
  
   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   if ( ! currBlock->initComplete_ )
   {
      printf("impSpecificRequests ERROR : call initComplete first.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- output help menu 
   // -------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid);
   MPI_Comm_size( mpiComm_, &nprocs);

   if ( ! strcmp("help",data_key) )
   {
      printf("impSpecifRequests : Available requests are \n");
      printf("    getElemOffset : get element processor offset \n");
      printf("                  argc    - >= 1.\n");
      printf("                  argv[0] - (int *) of length 1.\n");
      printf("    getNodeOffset : get node processor offset \n");
      printf("                  argc    - >= 1.\n");
      printf("                  argv[0] - (int *) of length 1.\n");
      printf("    getFaceOffset : get face processor offset \n");
      printf("                  argc    - >= 1.\n");
      printf("                  argv[0] - (int *) of length 1.\n");
      printf("    getNumExtNodes : get number of external nodes \n");
      printf("                  argc    - >= 1.\n");
      printf("                  argv[0] - (int *) of length 1.\n");
      printf("    getNumExtFaces : get number of external faces \n");
      printf("                  argc    - >= 1.\n");
      printf("                  argv[0] - (int *) of length 1.\n");
      printf("    getExtNodeNewGlobalIDs : get  external nodes' mapped IDs\n");
      printf("                  argc    - >= 1.\n");
      printf("                  argv[0] - (int *) of length nNnodesExt.\n");
      printf("    getExtFaceNewGlobalIDs : get  external faces' mapped IDs\n");
      printf("                  argc    - >= 1.\n");
      printf("                  argv[0] - (int *) of length nNnodesExt.\n");
      return 1;
   }

   // -------------------------------------------------------------
   // --- process requests
   // -------------------------------------------------------------

   // --- get element processor offset

   if ( ! strcmp("getElemOffset",data_key) )
   {
      if ( argc < 1 ) 
      {
         printf("implSpecificRequests ERROR : getElemOffset - argc < 1.\n");
         exit(1);
      } 
      int *offset = (int *) argv[0];
      (*offset) = currBlock->elemOffset_;
      return 1;
   }

   // --- get node processor offset

   else if ( ! strcmp("getNodeOffset", data_key) )
   {
      if ( argc < 1 ) 
      {
         printf("impSpecificRequests ERROR : getNodeOffset - argc < 1.\n");
         exit(1);
      } 
      int *offset = (int *) argv[0];
      (*offset) = currBlock->nodeOffset_;
      return 1;
   }

   // --- get face processor offset

   else if ( ! strcmp("getFaceOffset", data_key) )
   {
      if ( argc < 1 ) 
      {
         printf("impSpecificRequests ERROR : getFaceOffset - argc < 1.\n");
         exit(1);
      } 
      int *offset = (int *) argv[0];
      (*offset) = currBlock->faceOffset_;
      return 1;
   }

   // --- get number of external nodes (to my processor)

   else if ( ! strcmp("getNumExtNodes", data_key) )
   {
      if ( argc < 1 ) 
      {
         printf("impSpecificRequests ERROR : getNumExtNodes - argc < 1.\n");
         exit(1);
      } 
      int *nNodesExt = (int *) argv[0];
      (*nNodesExt) = currBlock->numExternalNodes_;
      return 1;
   }

   // --- get number of external faces (to my processor)

   else if ( ! strcmp("getNumExtFaces", data_key) )
   {
      if ( argc < 1 ) 
      {
         printf("impSpecificRequests ERROR : getNumExtFaces - argc < 1.\n");
         exit(1);
      } 
      int *nFacesExt = (int *) argv[0];
      (*nFacesExt) = currBlock->numExternalFaces_;
      return 1;
   }

   // --- get the mapped globalIDs of external nodes

   else if ( ! strcmp("getExtNodeNewGlobalIDs", data_key) )
   {
      if ( argc < 1 ) 
      {
         printf("impSpecificRequests ERROR : getExtNodeNewGlobalIDs-argc<1\n");
         exit(1);
      } 
      int *newGlobalIDs = (int *) argv[0];
      for ( int i = 0; i < currBlock->numExternalNodes_; i++ )
         newGlobalIDs[i] = currBlock->nodeExtNewGlobalIDs_[i];
      return 1;
   }

   // --- get the mapped globalIDs of external faces

   else if ( ! strcmp("getExtFaceNewGlobalIDs", data_key) )
   {
      if ( argc < 1 ) 
      {
         printf("impSpecificRequests ERROR : getExtFaceNewGlobalIDs-argc<1\n");
         exit(1);
      } 
      int *newGlobalIDs = (int *) argv[0];
      for ( int j = 0; j < currBlock->numExternalFaces_; j++ )
         newGlobalIDs[j] = currBlock->faceExtNewGlobalIDs_[j];
      return 1;
   }

   // --- get the mapped globalIDs of external faces

   else if ( ! strcmp("destroyElemMatrix", data_key) )
   {
      int elemNum = *(int *) argv[0];
      int index = searchElement(elemNum);
      if ( index < 0 )
      {
         printf("impSpecificRequests ERROR : getElemMatrix not found.\n");
         exit(1);
      }
      if ( currBlock->elemStiffMat_[index] != NULL )
      {
         delete [] currBlock->elemStiffMat_[index];
         currBlock->elemStiffMat_[index] = NULL;
      }
   }

   // --- create node element matrix (given local nodeExt element matrix)

   else if ( ! strcmp("updateNodeElemMatrix",data_key) )
   {
      int         *ncols = (int *) argv[0], **cols = (int **) argv[1];
      int         nNodes = currBlock->numLocalNodes_;
      int         nNodesExt = currBlock->numExternalNodes_;
      int         *nodeList = currBlock->nodeGlobalIDs_;
      int         *sharedNodeList = currBlock->sharedNodeIDs_;
      int         numSharedNodes = currBlock->numSharedNodes_;
      int         *sharedNodeNProcs = currBlock->sharedNodeNProcs_;
      int         **sharedNodeProc = currBlock->sharedNodeProc_;
      int         i, j, index, pnum, pSrc, *iBuf, msgID, nodeGID, ncnt;
      int         *columns, *procList, *procTemp, nSends, *sendLengs;
      int         *sendProcs, **sendBufs, nRecvs, *recvProcs, *recvLengs;
      int         **recvBufs, *owner, length;
      MPI_Request *request;
      MPI_Status  status;

      // get the owners for the external nodes

      MPI_Barrier(mpiComm_);

      if ( nNodesExt > 0 ) owner = new int[nNodesExt];
      else                 owner = NULL;

      for ( i = 0; i < numSharedNodes; i++ )
      {
         index = searchNode( sharedNodeList[i] ) - nNodes;
         if ( index >= 0 )
         {
            pnum  = mypid;
            for ( j = 0; j < sharedNodeNProcs[i]; j++ )
               if ( sharedNodeProc[i][j] < pnum )
                  pnum = sharedNodeProc[i][j];
            owner[index] = pnum;
         }
      }

      // find out how many distinct processor numbers and fill the 
      // send buffer

      if ( nNodesExt > 0 ) procList = new int[mypid];
      else                 procList = NULL;
      for ( i = 0; i < nNodesExt; i++ ) procList[i] = 0;
      for ( i = 0; i < nNodesExt; i++ ) 
         procList[owner[index]] += ncols[i+nNodes] + 2;
      nSends = 0;
      for ( i = 0; i < mypid; i++ ) if ( procList[i] > 0 ) nSends++;
      sendLengs = NULL;
      sendProcs = NULL;
      sendBufs  = NULL;
      if ( nSends > 0 ) 
      {
         sendLengs = new int[nSends];
         sendProcs = new int[nSends];
         sendBufs  = new int*[nSends];
         nSends = 0;
         for ( i = 0; i < mypid; i++ ) 
         {
            if ( procList[i] > 0 ) 
            {
               sendLengs[nSends] = procList[i];
               sendProcs[nSends] = i;
               sendBufs[i] = new int[sendLengs[nSends]];
               sendLengs[nSends] = 0;
               nSends++;
            }
         }
         nSends = 0;
         for ( i = 0; i < mypid; i++ ) 
            if ( procList[i] > 0 ) procList[i] = nSends++; 
         for ( i = 0; i < nNodesExt; i++ ) owner[i] = procList[owner[i]];
         for ( i = 0; i < nNodesExt; i++ ) 
         {
            sendBufs[owner[i]][sendLengs[owner[i]]++] = nodeList[i+nNodes];
            sendBufs[owner[i]][sendLengs[owner[i]]++] = ncols[i+nNodes];
            for ( j = 0; j < ncols[i+nNodes]; j++ ) 
               sendBufs[owner[i]][sendLengs[owner[i]]++] = cols[i+nNodes][j];
         }
      }

      // let the receiver knows about its intent to send

      procList = new int[nprocs];
      procTemp = new int[nprocs];
      for ( i = 0; i < nprocs; i++ ) procTemp[i] = 0;
      for ( i = 0; i < nSends; i++ ) procTemp[sendProcs[i]] = 1;
      MPI_Allreduce(procTemp,procList,nprocs,MPI_INT,MPI_SUM,mpiComm_);
      nRecvs = procList[mypid];
      delete [] procList;
      delete [] procTemp;
      recvLengs = NULL;
      recvProcs = NULL;
      recvBufs  = NULL;
      request   = NULL;
      if ( nRecvs > 0 )
      {
         request = new MPI_Request[nRecvs];
         recvLengs = new int[nRecvs];
         pSrc = MPI_ANY_SOURCE;
         msgID = 33420;
         for ( i = 0; i < nRecvs; i++ ) 
            MPI_Irecv(&recvLengs[i],1,MPI_INT,pSrc,msgID,mpiComm_,&request[i]);
      }
      if ( nSends > 0 )
      {
         msgID = 33420;
         for ( i = 0; i < nSends; i++ ) 
            MPI_Send(&sendLengs[i],1,MPI_INT,sendProcs[i],msgID,mpiComm_);
      }
      if ( nRecvs > 0 )
      {
         recvProcs = new int[nRecvs];
         recvBufs  = new int*[nRecvs];
         for ( i = 0; i < nRecvs; i++ ) 
         {
            MPI_Wait( &request[i], &status );
            recvProcs[i] = status.MPI_SOURCE;
            recvBufs[i]  = new int[recvLengs[i]];
         }
      }
      
      // now send/receive the external information

      if ( nRecvs > 0 )
      {
         msgID = 33421;
         for ( i = 0; i < nRecvs; i++ ) 
         {
            pSrc   = recvProcs[i];
            length = recvLengs[i];
            iBuf   = recvBufs[i];
            MPI_Irecv(iBuf,length,MPI_INT,pSrc,msgID,mpiComm_,&request[i]);
         }
      }
      if ( nSends > 0 )
      {
         msgID = 33421;
         for ( i = 0; i < nSends; i++ ) 
         {
            pSrc   = sendProcs[i];
            length = sendLengs[i];
            iBuf   = sendBufs[i];
            MPI_Send(iBuf, length, MPI_INT, pSrc, msgID, mpiComm_);
         }
      }
      if ( nRecvs > 0 )
      {
         for ( i = 0; i < nRecvs; i++ ) MPI_Wait( &request[i], &status );
      }
      
      // owners of shared nodes receive data

      for( i = 0; i < nRecvs; i++ )
      {
         ncnt = 0;
         while ( ncnt < recvLengs[i] )
         {
            nodeGID = recvBufs[i][ncnt++];
            length  = recvBufs[i][ncnt++];
            iBuf    = recvBufs[i];
            index   = MLI_Utils_BinarySearch(nodeGID, nodeList, nNodes);
            if ( index < 0 )
            {
               printf("updateNodeElemMatrix ERROR : in communication.\n");
               exit(1);
            }
            columns = new int[ncols[index]+length];
            for ( j = 0; j < ncols[index]; j++ ) columns[j] = cols[index][j];
            for ( j = 0; j < length; j++ ) 
               columns[ncols[index]++] = iBuf[j+ncnt];
            ncnt += length;
            delete [] cols[index];
            cols[index] = columns;
         }
      }
      delete [] procList;
      delete [] owner;
      for ( i = 0; i < nSends; i++ ) delete [] sendBufs[i];
      delete [] sendBufs;
      delete [] sendLengs;
      delete [] sendProcs;
      for ( i = 0; i < nRecvs; i++ ) delete [] recvBufs[i];
      delete [] recvBufs;
      delete [] recvLengs;
      delete [] recvProcs;
      delete [] request;
      return 1;
   }

   // --- create face element matrix (given local faceExt element matrix)

   else if ( !strcmp("updatefaceElemMatrix",data_key) )
   {
      MPI_Barrier(mpiComm_);

      int i, j, index, n, pnum, Buf[100];
      int *ncols = (int *) argv[0], **cols = (int **) argv[1];
      int *ind = new int[currBlock->numSharedFaces_];
      int *columns, l, k;
      MPI_Request request;
      MPI_Status  Status;
      
      // get the owners for the external faces

      int nFaces = currBlock->numLocalFaces_;
      int nFacesExt = currBlock->numExternalFaces_;
      int *faceList = currBlock->faceGlobalIDs_;
      int *sharedFaceList = currBlock->sharedFaceIDs_;
      int numSharedFaces = currBlock->numSharedFaces_;
      int *sharedFaceNProcs = currBlock->sharedFaceNProcs_;
      int **sharedFaceProc = currBlock->sharedFaceProc_;
      int *owner = new int [nFacesExt];
      for ( i = 0; i < numSharedFaces; i++ )
      {
         index = searchFace( sharedFaceList[i] ) - nFaces;
         if ( index >= 0 )
         {
            pnum  = mypid;
            for ( j = 0; j < sharedFaceNProcs[i]; j++ )
               if ( currBlock->sharedFaceProc_[i][j] < pnum )
                  pnum = currBlock->sharedFaceProc_[i][j];
            owner[index] = pnum;
         }
      }
      
      // external faces send with which elements are connected

      for ( i = 0; i < nFacesExt; i++ )
         MPI_Isend(cols[i+nFaces], ncols[i+nFaces], MPI_INT, 
                   owner[i], faceList[i+nFaces], mpiComm_, &request);
      
      // owners of shared faces receive data

      for ( i = 0; i < numSharedFaces; i++ )
      {
         ind[i] = MLI_Utils_BinarySearch(sharedFaceList[i], faceList, nFaces);
	  
         // the shared face is owned by this subdomain

         if (ind[i] >= 0)
	 {
	    for ( j = 0; j < sharedFaceNProcs[i]; j++ )
               if ( sharedFaceProc[i][j] != mypid )
	       {
                  MPI_Recv( Buf, 100, MPI_INT, MPI_ANY_SOURCE,
                            MPI_ANY_TAG, mpiComm_, &Status);
                  MPI_Get_count( &Status, MPI_INT, &n);
                  k = MLI_Utils_BinarySearch(Status.MPI_TAG,faceList,nFaces);
                  columns = new int[ncols[k]+n];
		  
                  for( l = 0; l < ncols[k]; l++ ) columns[l] = cols[k][l];
                  for( l = 0; l < n; l++ ) columns[ncols[k]++] = Buf[l];
		    
                  delete [] cols[k];
                  cols[k] = columns;
               }
	 }
      }
      delete [] ind;
      delete [] owner;
      return 1;
   }
   return 0;
}

//*************************************************************************
// read grid information from files
//-------------------------------------------------------------------------

int MLI_FEData::readFromFile(char *infile)
{
   int    i, j, k, nNodes, nodeDOF, index, nNodesPerElem;
   int    numFields, *fieldIDs=NULL, *fieldSizes=NULL, nElems_check;
   int    index2, nElems, *elemIDs=NULL, eMatDim;
   int    nodeNumFields, *nodeFieldIDs=NULL;
   int    elemNumFields, *elemFieldIDs=NULL;
   int    *nodeIDs=NULL, **IDLists=NULL, *numProcs=NULL, **procLists=NULL;
   int    spaceDim, *nodeIDAux=NULL, mypid;
   char   **nodeBCFlags=NULL;
   double *nodeCoords=NULL, **newCoords=NULL, **elemMat, **nodeBCVals;
   char   filename[80], inputString[256];;
   FILE   *fp;

   // -------------------------------------------------------------
   // --- read element node connectivity information
   // Format : space dimension
   //          number of fields
   //          fieldID and field size ...
   //          number of elements
   //          number of nodes per element
   //          number of element fields
   //          element field IDs... 
   //          number of nodal fields
   //          nodal field IDs... 
   //          element global IDs (nElems of them)
   //          element node list (nElems*nNodesPerElem of them)
   // -------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   sprintf( filename, "%s.elemConn.%d", infile, mypid );
   fp = fopen( filename, "r" );
   if ( fp == NULL )
   {
      printf("readFromFile ERROR : file elemConn does not exist.\n");
      exit(1);
   }
   fgets(inputString, 100, fp);
   while ( inputString[0] == '#' ) fgets(inputString, 100, fp);
   sscanf(inputString, "%d", &spaceDimension_);
   fscanf(fp, "%d", &numFields);
   fieldIDs = new int[numFields];
   fieldSizes = new int[numFields];
   for ( i = 0; i < numFields; i++ )
      fscanf(fp, "%d %d", &(fieldIDs[i]), &(fieldSizes[i]));

   fscanf(fp, "%d", &nElems);
   fscanf(fp, "%d", &nNodesPerElem);

   fscanf(fp, "%d", &elemNumFields);
   if ( elemNumFields > 0 ) elemFieldIDs = new int[elemNumFields];
   for (i = 0; i < elemNumFields; i++) fscanf(fp, "%d", &elemFieldIDs[i]);

   fscanf(fp, "%d", &nodeNumFields);
   if ( nodeNumFields > 0 ) nodeFieldIDs = new int[nodeNumFields];
   for (i = 0; i < nodeNumFields; i++) fscanf(fp, "%d", &nodeFieldIDs[i]);

   elemIDs = new int[nElems];
   for (i = 0; i < nElems; i++) fscanf(fp, "%d", &(elemIDs[i]));

   IDLists = new int*[nElems];
   for (i = 0; i < nElems; i++) IDLists[i] = new int[nNodesPerElem];
   for (i = 0; i < nElems; i++) 
   {
      for (j = 0; j < nNodesPerElem; j++) fscanf(fp, "%d", &(IDLists[i][j]));
   }
   fclose(fp);

   // -------------------------------------------------------------
   // --- read coordinate file, if present 
   // Format : number of nodes
   //          space dimension
   //          node global ID x y z ...
   // -------------------------------------------------------------

   sprintf( filename, "%s.nodeCoord.%d", infile, mypid );
   fp = fopen( filename, "r" );
   if ( fp != NULL )
   {
      fgets(inputString, 100, fp);
      while ( inputString[0] == '#' ) fgets(inputString, 100, fp);
      sscanf(inputString, "%d", &nNodes);
      fscanf(fp, "%d", &spaceDim);
      nodeIDs    = new int[nNodes];
      nodeCoords = new double[nNodes * spaceDim];
      for (i = 0; i < nNodes; i++) 
      {
         fscanf(fp, "%d", &(nodeIDs[i]));
         for (j = 0; j < spaceDim; j++) 
            fscanf(fp, "%lg", &(nodeCoords[i*spaceDim+j]));
      }
      fclose(fp);

      nodeIDAux = new int[nNodes];
      for (i = 0; i < nNodes; i++) nodeIDAux[i] = i; 
      newCoords = new double*[nElems];
      for (i = 0; i < nElems; i++) 
         newCoords[i] = new double[nNodesPerElem*spaceDim]; 

      MLI_Utils_IntQSort2(nodeIDs, nodeIDAux, 0, nNodes-1);
      for (i = 0; i < nElems; i++) 
      {
         for (j = 0; j < nNodesPerElem; j++) 
         {
            index = MLI_Utils_BinarySearch(IDLists[i][j], nodeIDs, nNodes);
            if ( index < 0 )
            {
               printf("readFromFile ERROR : element node ID not found.\n");
               exit(1);
            }
            for (k = 0; k < spaceDim; k++) 
            {
               index2 = nodeIDAux[index];
               newCoords[i][j*spaceDim+k] = nodeCoords[index2*spaceDim+k];
            }
         }
      }
   }

   // -------------------------------------------------------------
   // --- initialize the element block
   // -------------------------------------------------------------

   initFields(numFields, fieldSizes, fieldIDs);
   initElemBlock(nElems, nNodesPerElem, nodeNumFields, nodeFieldIDs,
                 elemNumFields, elemFieldIDs);
   initElemBlockNodeLists(nElems, elemIDs, nNodesPerElem,
                          IDLists, spaceDim, newCoords);

   // -------------------------------------------------------------
   // --- clean up
   // -------------------------------------------------------------

   if ( fieldIDs != NULL ) delete [] fieldIDs;
   if ( fieldSizes != NULL ) delete [] fieldSizes;
   if ( newCoords != NULL )
   {
      for (i = 0; i < nElems; i++) delete [] newCoords[i];
      delete [] newCoords;
   }
   if ( nodeCoords != NULL ) delete [] nodeCoords;
   if ( IDLists != NULL )
   {
      for (i = 0; i < nElems; i++) delete [] IDLists[i];
      delete [] IDLists;
   }
   if ( elemIDs      != NULL ) delete [] elemIDs;
   if ( nodeIDs      != NULL ) delete [] nodeIDs;
   if ( nodeIDAux    != NULL ) delete [] nodeIDAux;
   if ( elemFieldIDs != NULL ) delete [] elemFieldIDs;
   if ( nodeFieldIDs != NULL ) delete [] nodeFieldIDs;
   
   // -------------------------------------------------------------
   // --- read and shared nodes information
   // -------------------------------------------------------------

   sprintf( filename, "%s.nodeShared.%d", infile, mypid );
   fp = fopen( filename, "r" );
   if ( fp != NULL )
   {
      fgets(inputString, 100, fp);
      while ( inputString[0] == '#' ) fgets(inputString, 100, fp);
      sscanf(inputString, "%d", &nNodes);
      nodeIDs   = new int[nNodes];
      numProcs  = new int[nNodes];
      procLists = new int*[nNodes];
      for ( i = 0; i < nNodes; i++ ) 
      {
         fscanf(fp, "%d %d", &(nodeIDs[i]), &(numProcs[i]));
         procLists[i] = new int[numProcs[i]];
         for ( j = 0; j < numProcs[i]; j++ ) 
            fscanf(fp, "%d", &(procLists[i][j]));
      }
      initSharedNodes(nNodes, nodeIDs, numProcs, procLists);
      delete [] nodeIDs;
      delete [] numProcs;
      for ( i = 0; i < nNodes; i++ ) delete [] procLists[i];
      delete [] procLists;
   }
   initComplete();

   // -------------------------------------------------------------
   // --- read and load element stiffness matrices
   // -------------------------------------------------------------

   sprintf( filename, "%s.elemMatrix.%d", infile, mypid );
   fp = fopen( filename, "r" );
   if ( fp == NULL )
   {
      printf("readFromFile ERROR : file elemMatrix does not exist.\n");
      exit(1);
   }
   fgets(inputString, 100, fp);
   while ( inputString[0] == '#' ) fgets(inputString, 100, fp);
   sscanf(inputString, "%d", &nElems_check);
   if ( nElems_check != nElems )
   {
      printf("readFromFile ERROR : elemMat dimension do not match.\n");
      exit(1);
   }
   fscanf(fp, "%d", &eMatDim);
   elemMat = new double*[nElems];
   for ( i = 0; i < nElems; i++ ) elemMat[i] = new double[eMatDim*eMatDim];

   for ( i = 0; i < nElems; i++ )
   {
      for ( j = 0; j < eMatDim; j++ )
         for ( k = 0; k < eMatDim; k++ )
            fscanf(fp, "%lg", &(elemMat[i][k*eMatDim+j]));
   }
   fclose(fp);

   loadElemBlockMatrices(nElems, eMatDim, elemMat);
   for ( i = 0; i < nElems; i++ ) delete [] elemMat[i];
   delete [] elemMat;

   // -------------------------------------------------------------
   // --- read and load node boundary information
   // -------------------------------------------------------------

   sprintf( filename, "%s.nodeBC.%d", infile, mypid );
   fp = fopen( filename, "r" );
   if ( fp != NULL )
   {
      fgets(inputString, 100, fp);
      while ( inputString[0] == '#' ) fgets(inputString, 100, fp);
      sscanf(inputString, "%d %d", &nNodes, &nodeDOF);
      nodeIDs = new int[nNodes];
      nodeBCFlags = new char*[nNodes];
      nodeBCVals  = new double*[nNodes];
      for ( i = 0; i < nNodes; i++ ) nodeBCFlags[i] = new char[nodeDOF];
      for ( i = 0; i < nNodes; i++ ) 
      {
         nodeBCVals[i] = new double[nodeDOF];
         for ( j = 0; j < nodeDOF; j++ ) nodeBCVals[i][j] = 0.0; 
      }
      for ( i = 0; i < nNodes; i++ ) 
      {
         fscanf(fp, "%d", &(nodeIDs[i]));
         for ( j = 0; j < nodeDOF; j++ ) 
         {
            fscanf(fp, "%d", &k);
            if ( k > 0 )
            {
               nodeBCFlags[i][j] = 'Y';
               fscanf(fp, "%lg", &(nodeBCVals[i][j])); 
            }
            else nodeBCFlags[i][j] = 'N';
         }
      }
      fclose(fp);
      loadNodeBCs(nNodes, nodeIDs, nodeDOF, nodeBCFlags, nodeBCVals);
      if ( nodeIDs != NULL ) delete [] nodeIDs;
      for ( i = 0; i < nNodes; i++ ) delete [] nodeBCFlags[i];
      if ( nodeBCFlags != NULL ) delete [] nodeBCFlags;
      for ( i = 0; i < nNodes; i++ ) delete [] nodeBCVals[i];
      if ( nodeBCVals  != NULL ) delete [] nodeBCVals;
   }

   return 1;
}

//*************************************************************************
// write grid information to files
//-------------------------------------------------------------------------

int MLI_FEData::writeToFile(char *infile)
{
   int           i, j, k, nElems, nNodes, nodeDOF, length, mypid;
   char          filename[80];
   FILE          *fp;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   currBlock = elemBlockList_[currentElemBlock_];
   if ( ! currBlock->initComplete_ )
   {
      printf("writeToFile ERROR : initialization not complete.\n");
      exit(1);
   }

   // -------------------------------------------------------------
   // --- write element node connectivity information
   // Format : space dimension
   //          number of fields
   //          fieldID and field size ...
   //          number of elements
   //          number of nodes per element
   //          number of element fields
   //          element field IDs... 
   //          number of nodal fields
   //          nodal field IDs... 
   //          element global IDs (nElems of them)
   //          element node list (nElems*nNodesPerElem of them)
   // -------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   sprintf( filename, "%s.elemConn.%d", infile, mypid );
   fp = fopen( filename, "w" );
   if ( fp == NULL )
   {
      printf("writeToFile ERROR : cannot write to elemConn file.\n");
      exit(1);
   }
   fprintf(fp, "# Data format \n");
   fprintf(fp, "# A. space dimension \n");
   fprintf(fp, "# B. number of fields \n");
   fprintf(fp, "# C. fieldIDs fieldSizes \n");
   fprintf(fp, "# D. number of elements \n");
   fprintf(fp, "# E. number of nodes per element \n");
   fprintf(fp, "# F. number of element fields\n");
   fprintf(fp, "# G. element field IDs\n");
   fprintf(fp, "# H. number of nodal fields\n");
   fprintf(fp, "# I. nodal field IDs\n");
   fprintf(fp, "# J. element globalIDs \n");
   fprintf(fp, "# K. element node lists \n");
   fprintf(fp, "#\n");

   fprintf(fp, "%12d\n", spaceDimension_);
   fprintf(fp, "%12d\n", numFields_);
   for ( i = 0; i < numFields_; i++ )
      fprintf(fp, "%12d %12d\n", fieldIDs_[i], fieldSizes_[i]);
   
   nElems = currBlock->numLocalElems_;
   fprintf(fp, "%12d\n", nElems);
   fprintf(fp, "%12d\n", currBlock->elemNumNodes_);
   fprintf(fp, "%12d\n", currBlock->elemNumFields_);
   for (i = 0; i < currBlock->elemNumFields_; i++) 
      fprintf(fp, "%12d\n", currBlock->elemFieldIDs_[i]);
   fprintf(fp, "%12d\n", currBlock->nodeNumFields_);
   for (i = 0; i < currBlock->nodeNumFields_; i++) 
      fprintf(fp, "%12d\n", currBlock->nodeFieldIDs_[i]);

   fprintf(fp, "\n");
   for (i = 0; i < nElems; i++) 
      fprintf(fp, "%12d\n", currBlock->elemGlobalIDs_[i]);
   fprintf(fp,"\n");

   for (i = 0; i < nElems; i++) 
   {
      for ( j = 0; j < currBlock->elemNumNodes_; j++ ) 
         fprintf(fp, "%d ", currBlock->elemNodeIDList_[i][j]);
      fprintf(fp,"\n");
   } 
   fclose(fp);

   // -------------------------------------------------------------
   // --- write coordinate file, if needed 
   // Format : number of nodes
   //          space dimension
   //          node global ID x y z ...
   // -------------------------------------------------------------

   if ( currBlock->nodeCoordinates_ != NULL )
   {
      sprintf( filename, "%s.nodeCoord.%d", infile, mypid );
      fp = fopen( filename, "w" );
      if ( fp == NULL )
      {
         printf("writeToFile ERROR : cannot write to nodeCoord file.\n");
         exit(1);
      }
      fprintf(fp, "# Data format \n");
      fprintf(fp, "# A. number of nodes \n");
      fprintf(fp, "# B. space dimension \n");
      fprintf(fp, "# C. node ID  xcoord ycoord zcoord\n");
      fprintf(fp, "#\n");

      nNodes = currBlock->numLocalNodes_ + currBlock->numExternalNodes_;
      fprintf(fp, "%12d\n", nNodes);
      fprintf(fp, "%12d\n", spaceDimension_);
      for ( i = 0; i < nNodes; i++ ) 
      {
         fprintf(fp, "%12d", currBlock->nodeGlobalIDs_[i]);
         for (j = 0; j < spaceDimension_; j++) 
            fprintf(fp, "%20.12e",
                    currBlock->nodeCoordinates_[i*spaceDimension_+j]);
         fprintf(fp,"\n");
      }
      fclose(fp);
   }

   // -------------------------------------------------------------
   // --- write and shared nodes information
   // Format : number of shared nodes
   //          shared Node ID  number of processors processor list
   // -------------------------------------------------------------

   nNodes = currBlock->numSharedNodes_;
   if ( nNodes > 0 )
   {
      sprintf( filename, "%s.nodeShared.%d", infile, mypid );
      fp = fopen( filename, "w" );
      if ( fp == NULL )
      {
         printf("writeToFile ERROR : cannot write to nodeShared file.\n");
         exit(1);
      }
      fprintf(fp, "# Data format \n");
      fprintf(fp, "# A. number of shared nodes \n");
      fprintf(fp, "# B. shared node ID, nprocs, processor list \n");
      fprintf(fp, "#\n");

      fprintf(fp, "%d\n", nNodes);
      for ( i = 0; i < nNodes; i++ ) 
      {
         fprintf(fp, "%12d %12d\n", currBlock->sharedNodeIDs_[i], 
                currBlock->sharedNodeNProcs_[i]);
         for ( j = 0; j < currBlock->sharedNodeNProcs_[i]; j++ ) 
            fprintf(fp, "%12d\n", currBlock->sharedNodeProc_[i][j]);
      }
      fclose(fp);
   }

   // -------------------------------------------------------------
   // --- write element stiffness matrices
   // -------------------------------------------------------------

   length = currBlock->elemStiffDim_;

   sprintf( filename, "%s.elemMatrix.%d", infile, mypid );
   fp = fopen( filename, "w" );
   if ( fp == NULL )
   {
      printf("writeToFile ERROR : cannot write to elemMatrix file.\n");
      exit(1);
   }
   fprintf(fp, "# Data format \n");
   fprintf(fp, "# A. number of Elements \n");
   fprintf(fp, "# B. dimension of element matrix \n");
   fprintf(fp, "# C. element matrices \n");
   fprintf(fp, "#\n");

   fprintf(fp, "%d\n", nElems);
   fprintf(fp, "%d\n\n", length);
   for ( i = 0; i < nElems; i++ )
   {
      for ( j = 0; j < length; j++ )
      {
         for ( k = 0; k < length; k++ )
            fprintf(fp, "%25.16e ", currBlock->elemStiffMat_[i][k*length+j]);
         fprintf(fp, "\n");
      }
      fprintf(fp, "\n");
   }
   fclose(fp);

   // -------------------------------------------------------------
   // --- write node boundary information
   // -------------------------------------------------------------

   nNodes = currBlock->nodeNumBCs_;
   if ( nNodes > 0 )
   {
      sprintf( filename, "%s.nodeBC.%d", infile, mypid );
      fp = fopen( filename, "w" );
      if ( fp == NULL )
      {
         printf("writeToFile ERROR : cannot write to nodeBC file.\n");
         exit(1);
      }
      nodeDOF = currBlock->nodeDOF_;
      fprintf(fp, "# Data format \n");
      fprintf(fp, "# A. number of boundary nodes \n");
      fprintf(fp, "# B. nodal degree of freedom \n");
      fprintf(fp, "# C. node ID   (1 or -1)  value (if 1) \n\n");
      fprintf(fp, "#\n");

      fprintf(fp, "%d\n", nNodes );
      fprintf(fp, "%d\n", nodeDOF );
      for ( i = 0; i < nNodes; i++ ) 
      {
         for ( j = 0; j < nodeDOF; j++ ) 
         {
            if ( currBlock->nodeBCFlagList_[i][j] == 'Y' )
               fprintf(fp, "%12d  1  %25.16e\n", currBlock->nodeBCIDList_[i],
                    currBlock->nodeBCValues_[i][j]);
            else
               fprintf(fp, "%12d -1\n", currBlock->nodeBCIDList_[i]);
         }
      }
      fclose(fp);
   }
   return 1;
}

/**************************************************************************
 * constructor for the elemBlock 
 *-----------------------------------------------------------------------*/

int MLI_FEData::createElemBlock(int blockID)
{
   int           i;
   MLI_ElemBlock **tempBlocks, *currBlock;

   // -------------------------------------------------------------
   // --- check that element block ID cannot be arbitrary
   // -------------------------------------------------------------

   if ( blockID > numElemBlocks_ )
   {
      printf("createElemBlock : block ID %d invalid.\n", blockID);
      exit(1);
   }

   // -------------------------------------------------------------
   // --- if a new elemBlock is requested
   // -------------------------------------------------------------

   if ( blockID == numElemBlocks_ )
   {
      tempBlocks = elemBlockList_;
      numElemBlocks_++;
      elemBlockList_ = new MLI_ElemBlock*[numElemBlocks_];
      for (i = 0; i < numElemBlocks_-1; i++) elemBlockList_[i] = tempBlocks[i];
      elemBlockList_[numElemBlocks_-1] = new MLI_ElemBlock();
      delete [] tempBlocks;
   }

   // -------------------------------------------------------------
   // --- initialize variables in the new element block
   // -------------------------------------------------------------

   currBlock = elemBlockList_[blockID];
   currBlock->numLocalElems_       = 0;
   currBlock->elemGlobalIDs_       = NULL;
   currBlock->elemGlobalIDAux_     = NULL;
   currBlock->elemNumFields_       = 0;
   currBlock->elemFieldIDs_        = NULL;
   currBlock->elemDOF_             = 0;
   currBlock->elemNumNodes_        = 0;
   currBlock->elemNodeIDList_      = NULL;
   currBlock->elemStiffDim_        = 0;
   currBlock->elemStiffMat_        = NULL;
   currBlock->elemNumNS_           = NULL;
   currBlock->elemNullSpace_       = NULL;
   currBlock->elemVolume_          = NULL;
   currBlock->elemMaterial_        = NULL;
   currBlock->elemParentIDs_       = NULL;
   currBlock->elemLoads_           = NULL;
   currBlock->elemSol_             = NULL;
   currBlock->elemNumFaces_        = 0;
   currBlock->elemFaceIDList_      = NULL;
   currBlock->elemNumBCs_          = 0;
   currBlock->elemBCIDList_        = NULL;
   currBlock->elemBCFlagList_      = NULL;
   currBlock->elemBCValues_        = NULL;
   currBlock->elemOffset_          = 0;

   currBlock->numLocalNodes_       = 0;
   currBlock->numExternalNodes_    = 0;
   currBlock->nodeGlobalIDs_       = NULL;
   currBlock->nodeNumFields_       = 0;
   currBlock->nodeFieldIDs_        = NULL;
   currBlock->nodeDOF_             = 0;
   currBlock->nodeCoordinates_     = NULL;
   currBlock->nodeNumBCs_          = 0;
   currBlock->nodeBCIDList_        = NULL;
   currBlock->nodeBCFlagList_      = NULL;
   currBlock->nodeBCValues_        = NULL;
   currBlock->numSharedNodes_      = 0;
   currBlock->sharedNodeIDs_       = NULL;
   currBlock->sharedNodeNProcs_    = NULL;
   currBlock->sharedNodeProc_      = NULL;
   currBlock->nodeExtNewGlobalIDs_ = NULL;
   currBlock->nodeOffset_          = 0;

   currBlock->numLocalFaces_       = 0;
   currBlock->numExternalFaces_    = 0;
   currBlock->faceGlobalIDs_       = NULL;
   currBlock->faceNumNodes_        = 0;
   currBlock->faceNodeIDList_      = NULL;
   currBlock->numSharedFaces_      = 0;
   currBlock->sharedFaceIDs_       = NULL;
   currBlock->sharedFaceNProcs_    = NULL;
   currBlock->sharedFaceProc_      = NULL;
   currBlock->faceExtNewGlobalIDs_ = NULL;
   currBlock->faceOffset_          = 0;

   currBlock->initComplete_        = 0;
   return 0;
}

/**************************************************************************
 * destructor for the elemBlock 
 *-----------------------------------------------------------------------*/

int MLI_FEData::deleteElemBlock(int blockID)
{
   int           i;
   MLI_ElemBlock *currBlock;

   // -------------------------------------------------------------
   // --- error checking
   // -------------------------------------------------------------

   if ( blockID >= numElemBlocks_ || blockID < 0 )
   {
      printf("deleteElemBlock : block ID %d invalid.\n", blockID);
      exit(1);
   }
   if ( elemBlockList_[blockID] == NULL )
   {
      printf("deleteElemBlock : block %d NULL.\n", blockID);
      exit(1);
   }

   // -------------------------------------------------------------
   // --- initialize variables in the new element block
   // -------------------------------------------------------------

   currBlock = elemBlockList_[blockID];
   if (currBlock->elemGlobalIDs_ != NULL) delete [] currBlock->elemGlobalIDs_;
   if (currBlock->elemGlobalIDAux_ != NULL) 
      delete [] currBlock->elemGlobalIDAux_;
   if (currBlock->elemFieldIDs_ != NULL) delete [] currBlock->elemFieldIDs_;
   if (currBlock->elemNodeIDList_ != NULL) 
   {
      for ( i = 0; i < currBlock->numLocalElems_; i++ )
         delete [] currBlock->elemNodeIDList_[i];
      delete [] currBlock->elemNodeIDList_;
   }
   if (currBlock->elemStiffMat_ != NULL) 
   {
      for ( i = 0; i < currBlock->numLocalElems_; i++ )
         delete [] currBlock->elemStiffMat_[i];
      delete [] currBlock->elemStiffMat_;
   }
   if (currBlock->elemNumNS_ != NULL) delete [] currBlock->elemNumNS_;
   if (currBlock->elemNullSpace_ != NULL) delete [] currBlock->elemNullSpace_;
   if (currBlock->elemVolume_ != NULL) delete [] currBlock->elemVolume_;
   if (currBlock->elemMaterial_ != NULL) delete [] currBlock->elemMaterial_;
   if (currBlock->elemParentIDs_ != NULL) delete [] currBlock->elemParentIDs_;
   if (currBlock->elemLoads_ != NULL) 
   {
      for ( i = 0; i < currBlock->numLocalElems_; i++ )
         delete [] currBlock->elemLoads_[i];
      delete [] currBlock->elemLoads_;
   }
   if (currBlock->elemSol_ != NULL) 
   {
      for ( i = 0; i < currBlock->numLocalElems_; i++ )
         delete [] currBlock->elemSol_[i];
      delete [] currBlock->elemSol_;
   }
   if (currBlock->elemFaceIDList_ != NULL) 
   {
      for ( i = 0; i < currBlock->numLocalElems_; i++ )
         delete [] currBlock->elemFaceIDList_[i];
      delete [] currBlock->elemFaceIDList_;
   }
   if (currBlock->elemBCIDList_ != NULL) delete [] currBlock->elemBCIDList_;
   if (currBlock->elemBCFlagList_ != NULL) 
   {
      for ( i = 0; i < currBlock->numLocalElems_; i++ )
         delete [] currBlock->elemBCFlagList_[i];
      delete [] currBlock->elemBCFlagList_;
      for ( i = 0; i < currBlock->numLocalElems_; i++ )
         delete [] currBlock->elemBCValues_[i];
      delete [] currBlock->elemBCValues_;
   }
   currBlock->elemNumFields_    = 0;
   currBlock->elemDOF_          = 0;
   currBlock->elemNumNodes_     = 0;
   currBlock->elemStiffDim_     = 0;
   currBlock->numLocalElems_    = 0;
   currBlock->elemNumFaces_     = 0;
   currBlock->elemNumBCs_       = 0;
   currBlock->elemOffset_       = 0;

   if (currBlock->nodeGlobalIDs_ != NULL) delete [] currBlock->nodeGlobalIDs_;
   if (currBlock->nodeFieldIDs_ != NULL) delete [] currBlock->nodeFieldIDs_;
   if (currBlock->nodeCoordinates_ != NULL) 
      delete [] currBlock->nodeCoordinates_;
   if (currBlock->nodeBCIDList_ != NULL) delete [] currBlock->nodeBCIDList_;
   if (currBlock->nodeBCFlagList_ != NULL)
   {
      for ( i = 0; i < currBlock->nodeNumBCs_; i++ )
         delete [] currBlock->nodeBCFlagList_[i];
      delete [] currBlock->nodeBCFlagList_;
      for ( i = 0; i < currBlock->nodeNumBCs_; i++ )
         delete [] currBlock->nodeBCValues_[i];
      delete [] currBlock->nodeBCValues_;
   }
   if (currBlock->sharedNodeIDs_ != NULL) delete [] currBlock->sharedNodeIDs_;
   if (currBlock->sharedNodeNProcs_ != NULL) 
      delete [] currBlock->sharedNodeNProcs_;
   if (currBlock->sharedNodeProc_ != NULL) 
   {
      for ( i = 0; i < currBlock->numSharedNodes_; i++ )
         delete [] currBlock->sharedNodeProc_[i];
      delete [] currBlock->sharedNodeProc_;
   }
   if ( currBlock->nodeExtNewGlobalIDs_ != NULL )
      delete [] currBlock->nodeExtNewGlobalIDs_;
   currBlock->numLocalNodes_    = 0;
   currBlock->numExternalNodes_ = 0;
   currBlock->nodeNumFields_    = 0;
   currBlock->nodeDOF_          = 0;
   currBlock->nodeNumBCs_       = 0;
   currBlock->numSharedNodes_   = 0;
   currBlock->nodeOffset_       = 0;

   if (currBlock->faceGlobalIDs_ != NULL) delete [] currBlock->faceGlobalIDs_;
   if (currBlock->faceNodeIDList_ != NULL) 
   {
      int nFaces = currBlock->numLocalFaces_ + currBlock->numExternalFaces_;
      for ( i = 0; i < nFaces; i++ )
         delete [] currBlock->faceNodeIDList_[i];
      delete [] currBlock->faceNodeIDList_;
   }
   if (currBlock->sharedFaceIDs_ != NULL) delete [] currBlock->sharedFaceIDs_;
   if (currBlock->sharedFaceNProcs_ != NULL) 
      delete [] currBlock->sharedFaceNProcs_;
   if (currBlock->sharedFaceProc_ != NULL) 
   {
      for ( i = 0; i < currBlock->numSharedFaces_; i++ )
         delete [] currBlock->sharedFaceProc_[i];
      delete [] currBlock->sharedFaceProc_;
   }
   if ( currBlock->faceExtNewGlobalIDs_ != NULL )
      delete [] currBlock->faceExtNewGlobalIDs_;
   currBlock->numLocalFaces_    = 0;
   currBlock->numExternalFaces_ = 0;
   currBlock->faceNumNodes_     = 0;
   currBlock->numSharedFaces_   = 0;
   currBlock->faceOffset_       = 0;

   currBlock->initComplete_     = 0;
   return 0;
}

/**************************************************************************
 * search element ID in an ordered array
 *-----------------------------------------------------------------------*/

int MLI_FEData::searchElement(int key)
{
   int           index;
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];

   index = MLI_Utils_BinarySearch(key, currBlock->elemGlobalIDs_, 
                                  currBlock->numLocalElems_);
   return index;
}

/**************************************************************************
 * search node ID in an ordered array
 *-----------------------------------------------------------------------*/

int MLI_FEData::searchNode(int key)
{
   int           index;
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];

   index = MLI_Utils_BinarySearch(key, currBlock->nodeGlobalIDs_, 
                                  currBlock->numLocalNodes_);
   if ( index < 0 )
   {
      index = MLI_Utils_BinarySearch(key,
                  &(currBlock->nodeGlobalIDs_[currBlock->numLocalNodes_]),
                  currBlock->numExternalNodes_);
      if ( index >= 0 ) index += currBlock->numLocalNodes_;
   }
   return index;
}

/**************************************************************************
 * search face ID in an ordered array
 *-----------------------------------------------------------------------*/

int MLI_FEData::searchFace(int key)
{
   int           index;
   MLI_ElemBlock *currBlock = elemBlockList_[currentElemBlock_];

   index = MLI_Utils_BinarySearch(key, currBlock->faceGlobalIDs_, 
                                  currBlock->numLocalFaces_);
   if ( index < 0 )
   {
      index = MLI_Utils_BinarySearch(key,
                  &(currBlock->faceGlobalIDs_[currBlock->numLocalFaces_]),
                  currBlock->numExternalFaces_);
      if ( index >= 0 ) index += currBlock->numLocalFaces_;
   }
   return index;
}

