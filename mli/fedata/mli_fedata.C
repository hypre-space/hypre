/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/**************************************************************************
 **************************************************************************
 * MLI_FEData Class functions
 **************************************************************************
 **************************************************************************/

#include <iostream.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "mli_fedata.h"

/**************************************************************************
 * constructor 
 *-----------------------------------------------------------------------*/

MLI_FEData::MLI_FEData(int my_id)
{
   mypid_           = my_id;
   spaceDimension_  = -1;
   orderOfPDE_      = -1;

   numLocalElems_   = 0;
   elemNodeLeng_    = NULL;
   elemNodeList_    = NULL;
   elemGlobalID_    = NULL;
   elemStiff_       = NULL;
   elemNullSpace_   = NULL;
   elemNullLeng_    = NULL;
   elemVolume_      = NULL;
   elemMaterial_    = NULL;
   elemEdgeLeng_    = NULL;
   elemEdgeList_    = NULL;
   elemFaceLeng_    = NULL;
   elemFaceList_    = NULL;
   elemParentID_    = NULL;

   numLocalEdges_   = 0;
   edgeGlobalID_    = NULL;
   edgeNodeList_    = NULL;

   numLocalFaces_   = 0;
   numExternalFaces_= 0;
   faceGlobalID_    = NULL;
   externalFaces_   = NULL;
   faceDOF_         = NULL;
   faceBCLengMax_   = 0;
   numFaceBCs_      = 0;
   faceBCList_      = NULL;
   faceBCDofList_   = NULL;
   faceBCValues_    = NULL;
   numSharedFaces_  = 0;
   sharedFaceLeng_  = NULL;
   sharedFaceProc_  = NULL;
   faceNodeLeng_    = NULL;
   faceNodeList_    = NULL;

   numLocalNodes_   = 0;
   numExternalNodes_= 0;
   nodeGlobalID_    = NULL;
   externalNodes_   = NULL;
   nodeDOF_         = NULL;
   nodeBCLengMax_   = 0;
   numNodeBCs_      = 0;
   nodeBCList_      = NULL;
   nodeBCDofList_   = NULL;
   nodeBCValues_    = NULL;
   nodeCoordinates_ = NULL;
   numSharedNodes_  = 0;
   sharedNodeLeng_  = NULL;
   sharedNodeProc_  = NULL;

   processNodeFlag_ = 0;
   processElemFlag_ = 0;
   processEdgeFlag_ = 0;
   processFaceFlag_ = 0;
   outputLevel_     = 0;

   USR_computeShapeFuncInterpolant = NULL;
}

//*************************************************************************
// destructor 
//-------------------------------------------------------------------------

MLI_FEData::~MLI_FEData()
{
   cleanElemInfo();
   cleanNodeInfo();
   cleanEdgeInfo();
   cleanFaceInfo();
}

//*************************************************************************
// set diagnostics output level
//-------------------------------------------------------------------------

int MLI_FEData::setOutputLevel(int level)
{
   if ( level < 0 )
   {
      printf("MLI_FEData ERROR : setOutputLevel - level not valid.\n");
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
   if ( dimension <= 0 || dimension > 4)
   {
      printf("MLI_FEData ERROR : setSpaceDimension - dim not valid.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("MLI_FEData :setSpaceDimension = %d\n", dimension);
   spaceDimension_ = dimension;
   return 1;
}

//*************************************************************************
// order of the partial differential equation 
//-------------------------------------------------------------------------

int MLI_FEData::setOrderOfPDE(int pdeOrder)
{
   if ( pdeOrder <= 0 || pdeOrder > 4)
   {
      printf("MLI_FEData ERROR : setOrderOfPDE - order not valid.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("MLI_FEData :setOrderOfPDE = %d\n", pdeOrder);
   orderOfPDE_ = pdeOrder;
   return 1;
}

//*************************************************************************
// set number of elements 
//-------------------------------------------------------------------------

int MLI_FEData::beginInitElemSet(int nElems, int *gid)
{
   int i;

   if ( nElems <= 0 )
   {
      printf("MLI_FEData ERROR : beginInitElemSet - nElems <= 0.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("MLI_FEData :beginInitElemSet = %d\n", nElems);

   cleanElemInfo();

   numLocalElems_ = nElems;

   elemStiff_ = new double**[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) elemStiff_[i] = NULL;

   elemNodeLeng_  = new int[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) elemNodeLeng_[i] = 0;

   elemNodeList_ = new int*[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) elemNodeList_[i] = NULL;

   elemGlobalID_ = new int[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) elemGlobalID_[i] = gid[i];

   elemNullLeng_ = new int[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) elemNullLeng_[i] = 0;

   elemNullSpace_ = new double**[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) elemNullSpace_[i] = NULL;

   elemVolume_ = new double[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) elemVolume_[i] = 0.0;

   elemMaterial_ = new int[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) elemMaterial_[i] = 0;

   intSort2(elemGlobalID_, NULL, 0, numLocalElems_-1);
   for ( i = 1; i < numLocalElems_; i++ ) 
   { 
      if ( elemGlobalID_[i] == elemGlobalID_[i-1] )
         printf("MLI_FEData ERROR : beginInitElemSet - duplicate elemIDs.\n");
   }
   return 1;
}

//*************************************************************************
// to be called after all element information has been loaded
//-------------------------------------------------------------------------

int MLI_FEData::endInitElemSet()
{
   int i;

   if ( numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : endInitElemSet - nElems <= 0.\n");
      return 0;
   }
   for ( i = 0; i < numLocalElems_; i++ )
   {
     if ( elemNodeLeng_[i] <= 0 || elemNodeList_[i] == NULL ||
	  elemStiff_[i] == NULL )
     {
       printf("MLI_FEData ERROR : endInitElemSet - elem not initialized.\n");
       return 0;
     }
   }
   processElemFlag_ = 1;

   return 1;
}

//*************************************************************************
// load element node list and stiffness matrix 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemSet(int elemID,int nNodesPerElem,int *nodeList,
                            int sdim, double **sMat)
{
   int  i, j, index;

   if ( outputLevel_ >= 1 )
      printf("MLI_FEData :loadElemSet, element ID = %d\n", elemID);

   if ( numLocalElems_ <= 0 ) 
   {
      printf("MLI_FEData ERROR : loadElemSet - numElems <= 0.\n");
      return 0;
   }
   else if ( elemNodeList_ == NULL )
   {
      printf("MLI_FEData ERROR : loadElemSet - not initialized (1).\n");
      return 0;
   }
   if ( elemNodeLeng_ == NULL )
   {
      printf("MLI_FEData ERROR : loadElemSet - not initialized (2).\n");
      return 0;
   }
   if ( elemStiff_ == NULL )
   {
      printf("MLI_FEData ERROR : loadElemSet - not initialized (3).\n");
      return 0;
   }
   if ( nNodesPerElem <= 0 )
   {
      printf("MLI_FEData ERROR : loadElemSet - NodesPerElem <= 0.\n");
      return 0;
   }

   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : loadElemSet - element not local.\n");
      return 0;
   }
   if ( elemStiff_[index] != NULL )
   {
      printf("MLI_FEData ERROR : loadElemSet - element loaded before.\n");
      return 0;
   }

   elemNodeLeng_[index]  = nNodesPerElem;
   elemNodeList_[index] = new int[nNodesPerElem];
   for ( i = 0; i < nNodesPerElem; i++ ) 
      elemNodeList_[index][i] = nodeList[i];
   elemStiff_[index] = new double*[sdim];
   for ( i = 0; i < sdim; i++ ) 
      elemStiff_[index][i] = new double[sdim];
   for ( i = 0; i < sdim; i++ ) 
      for ( j = 0; j < sdim; j++ ) 
         elemStiff_[index][i][j] = sMat[i][j];
    
   return 1;
}

//*************************************************************************
// load element nullspace 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemNullSpace(int elemID,int nSize,double **nSpace)
{
   int  i, j, length, index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("MLI_FEData ERROR : loadElemNullSpace - numElems <= 0.\n");
      return 0;
   }
   else if ( elemNullSpace_ == NULL || elemNullLeng_ == NULL )
   {
      printf("MLI_FEData ERROR : loadElemNullSpace - not initialized (1).\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : loadElemNullSpace - element not local.\n");
      return 0;
   }
   elemNullLeng_[index] = nSize;
   length = elemNodeLeng_[index];
   elemNullSpace_[index] = new double*[nSize];
   for ( i = 0; i < nSize; i++ )
      elemNullSpace_[index][i] = new double[length];
   for ( i = 0; i < nSize; i++ ) 
      for ( j = 0; j < length; j++ ) 
         elemNullSpace_[index][i][j] = nSpace[i][j];
   return 1;
}

//*************************************************************************
// load element volume 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemVolume(int elemID, double volume)
{
   int index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("MLI_FEData ERROR : loadElemVolume - numElems <= 0.\n");
      return 0;
   }
   else if ( elemVolume_ == NULL )
   {
      printf("MLI_FEData ERROR : loadElemVolume - not initialized.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : loadElemVolume - element not local.\n");
      return 0;
   }
   elemVolume_[index] = volume;
   return 1;
}

//*************************************************************************
// load element material 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemMaterial(int elemID, int material)
{
   int index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("MLI_FEData ERROR : loadElemMaterial - numElems <= 0.\n");
      return 0;
   }
   else if ( elemMaterial_ == NULL )
   {
      printf("MLI_FEData ERROR : loadElemMaterial - not initialized.\n");
      return 0;
   }
   index   = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : loadElemMaterial - element not local.\n");
      return 0;
   }
   elemMaterial_[index] = material;
   return 1;
}

//*************************************************************************
// load element edge list 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemEdgeList(int elemID, int nEdges, int *edgeList)
{
   int i, index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("MLI_FEData ERROR : loadElemEdgeList - numElems <= 0.\n");
      return 0;
   }
   else if ( elemEdgeList_ == NULL )
   {
      elemEdgeList_ = new int*[numLocalElems_];
      elemEdgeLeng_ = new int[numLocalElems_];
      for ( i = 0; i < numLocalElems_; i++ ) 
      {
         elemEdgeList_[i] = NULL;
         elemEdgeLeng_[i] = 0;
      }
   }
   index   = searchElement( elemID );
   elemEdgeLeng_[index] = nEdges;
   elemEdgeList_[index] = new int[nEdges];
   for ( i = 0; i < nEdges; i++ ) 
      elemEdgeList_[index][i] = edgeList[i];
   return 1;
} 


//*************************************************************************
// load element face list 
//-------------------------------------------------------------------------

int MLI_FEData::loadElemFaceList(int elemID, int nFaces, int *faceList)
{
   int i, index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("MLI_FEData ERROR : loadElemFaceList - numElems <= 0.\n");
      return 0;
   }
   else if ( elemFaceList_ == NULL )
   {
      elemFaceList_ = new int*[numLocalElems_];
      elemFaceLeng_ = new int[numLocalElems_];
      for ( i = 0; i < numLocalElems_; i++ ) 
      {
         elemFaceList_[i] = NULL;
         elemFaceLeng_[i] = 0;
      }
   }
   index   = searchElement( elemID );
   elemFaceLeng_[index] = nFaces;
   elemFaceList_[index] = new int[nFaces];
   for ( i = 0; i < nFaces; i++ ) 
      elemFaceList_[index][i] = faceList[i];
   return 1;
} 


//*************************************************************************
// load element's parent element in the coarse grid
//-------------------------------------------------------------------------

int MLI_FEData::loadElemParentID(int elemID, int parentID)
{
   int i, index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("MLI_FEData ERROR : loadElemParentID - numElems <= 0.\n");
      return 0;
   }
   else if ( elemParentID_ == NULL )
   {
      elemParentID_ = new int[numLocalElems_];
      for ( i = 0; i < numLocalElems_; i++ ) elemParentID_[i] = 0;
   }
   index = searchElement( elemID );
   elemParentID_[index] = parentID;
   return 1;
} 

//*************************************************************************
// begin initializing nodal information 
//-------------------------------------------------------------------------

int MLI_FEData::beginInitNodeSet()
{
   int    i, j, index, totalNodes, count, count2, *node_array;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : beginInitNodeSet - elemSet not done.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("MLI_FEData :beginInitNodeSet\n");

   // TTT
   // cleanNodeInfo();

   totalNodes = 0;
   for ( i = 0; i < numLocalElems_; i++ ) totalNodes += elemNodeLeng_[i];
   node_array = new int[totalNodes];
   count = 0;
   for ( i = 0; i < numLocalElems_; i++ ) 
   {
      for ( j = 0; j < elemNodeLeng_[i]; j++ ) 
         node_array[count++] = elemNodeList_[i][j];
   }
   intSort2(node_array, NULL, 0, count-1);
   count2 = 1;
   for ( i = 1; i < count; i++ ) 
      if ( node_array[i] != node_array[i-1] ) 
         node_array[count2++] = node_array[i];

   numLocalNodes_ = count2;
   numExternalNodes_ = 0;

   nodeGlobalID_ = new int[count2];
   for ( i = 0; i < count2; i++ ) nodeGlobalID_[i] = node_array[i]; 

   nodeDOF_ = new int[numLocalNodes_];
   for ( i = 0; i < numLocalNodes_; i++ ) nodeDOF_[i] = 0;

   nodeCoordinates_ = new double[numLocalNodes_ * spaceDimension_];
   
   delete [] node_array;
   return 1;
}

//*************************************************************************
// terminate initializing nodal information 
//-------------------------------------------------------------------------

int MLI_FEData::endInitNodeSet()
{
   int    i, j, nnodesLocal, nnodesExt, *extNodeFlag, *newNodeList;
   int    count, *iarray;
   double *darray;
   char   param_string[100];

   nnodesExt   = 0;
   extNodeFlag = new int[numLocalNodes_];

   for ( i = 0; i < numLocalNodes_; i++ ) extNodeFlag[i] = 0;

   for ( i = 0; i < numSharedNodes_; i++ )
      for ( j = 0; j < sharedNodeLeng_[i]; j++ )
         if ( sharedNodeProc_[i][j] < mypid_ ) 
	 {
	   nnodesExt++; 
	   extNodeFlag[searchNode(sharedNodeID_[i])] = - 1;
	   break;
	 }

   numExternalNodes_ = nnodesExt;
   nnodesLocal       = numLocalNodes_;
   numLocalNodes_   -= numExternalNodes_;
   newNodeList       = new int[nnodesLocal];

   count = 0;
   for ( i = 0; i < nnodesLocal; i++ )
   {
      if ( extNodeFlag[i] == 0 ) 
      {
         newNodeList[count] = nodeGlobalID_[i];
         extNodeFlag[i] = count++;
      }
   }
   for ( i = 0; i < nnodesLocal; i++ )
   {
      if ( extNodeFlag[i] == - 1 ) 
      {
         newNodeList[count] = nodeGlobalID_[i];
         extNodeFlag[i] = count++;
      }
   } 

   delete [] nodeGlobalID_;
   nodeGlobalID_ = newNodeList;

   if ( nodeDOF_ != NULL )
   {
      iarray = nodeDOF_;
      nodeDOF_ = new int[nnodesLocal];
      for ( i = 0; i < nnodesLocal; i++ ) 
         nodeDOF_[extNodeFlag[i]] = iarray[i];
      delete [] iarray;
   }

   if ( nodeCoordinates_ != NULL )
   {
      darray = nodeCoordinates_;
      nodeCoordinates_ = new double[nnodesLocal*spaceDimension_];

      for ( i = 0; i < nnodesLocal; i++ ) 
         for ( j = 0; j < spaceDimension_; j++ ) 
            nodeCoordinates_[extNodeFlag[i]*spaceDimension_+j] =
	      darray[i*spaceDimension_+j];
      /* TTT
      for ( i = 0; i < nnodesLocal; i++ ) 
         for ( j = 0; j < spaceDimension_; j++ ) 
            nodeCoordinates_[i*spaceDimension_+j] = 
               darray[extNodeFlag[i]*spaceDimension_+j];
      */
      delete [] darray;
   }

   delete [] extNodeFlag;
 
   // fix the indices for the external nodes
   externalNodes_ = new int[numExternalNodes_];

   MPI_Request request;
   MPI_Status Status;
   int *ind = new int[numSharedNodes_], external_index;

   strcpy( param_string, "node_offset" );
   getSpecificData(param_string, &node_off);
   MPI_Barrier(MPI_COMM_WORLD);

   // send the indices of the owned shared nodes

   for(i=0; i<numSharedNodes_; i++)
   {
      ind[i] = search(sharedNodeID_[i], nodeGlobalID_, numLocalNodes_);
       
      // the shared node is owned by this subdomain

      if (ind[i] >= 0)
      {
         ind[i] += node_off;
         for ( j = 0; j < sharedNodeLeng_[i]; j++ )
            if (sharedNodeProc_[i][j] != mypid_)
               MPI_Isend(&ind[i], 1, MPI_INT, sharedNodeProc_[i][j], 
                         sharedNodeID_[i], MPI_COMM_WORLD, &request);
      }
   }

   // receive the indices of the external nodes

   for(i=0; i<numExternalNodes_; i++)
   {
      MPI_Recv( &external_index, 1, MPI_INT, MPI_ANY_SOURCE,
	        MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
      externalNodes_[ search ( Status.MPI_TAG, 
				nodeGlobalID_ + numLocalNodes_, 
				numExternalNodes_ ) ] = external_index;
   }
   MPI_Barrier(MPI_COMM_WORLD);

   delete [] ind;
   return 1;
}   

//*************************************************************************
// set node degree of freedom 
//-------------------------------------------------------------------------

int MLI_FEData::loadNodeDOF(int nodeID, int dof)
{
   int i, index;

   if ( numLocalNodes_ == 0 )
   {
      printf("MLI_FEData ERROR : loadNodeDOF - beginInitNodeSet ?\n");
      return 0;
   }
   else if ( nodeDOF_ == NULL )
   {
      printf("MLI_FEData ERROR : loadNodeDOF - nodeDOF not initialized.\n");
      return 0;
   }
   if ( nodeID == - 1 )
   {
      for ( i = 0; i < numLocalNodes_+numExternalNodes_; i++ ) 
         nodeDOF_[i] = dof;
   }
   else
   {
      index = searchNode( nodeID );
      if ( index < 0 )
      {
         printf("MLI_FEData ERROR : loadNodeDOF - node not local.\n");
         return 0;
      }
      nodeDOF_[index] = dof;
   }
   return 1;
}

//*************************************************************************
// load node coordinates 
//-------------------------------------------------------------------------

int MLI_FEData::loadNodeCoordinate(int nodeID, double *coord)
{
   int i, index;

   if ( numLocalNodes_ == 0 )
   {
      printf("MLI_FEData ERROR : loadNodeCoordinate - beginInitNodeSet ?\n");
      return 0;
   }
   else if ( nodeCoordinates_ == NULL )
   {
      printf("MLI_FEData ERROR : loadNodeCoordinate - not initialized.\n");
      return 0;
   }

   index = searchNode( nodeID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : loadNodeCoordinate - node not local.\n");
      return 0;
   }
   for ( i = 0; i < spaceDimension_; i++ )
      nodeCoordinates_[index*spaceDimension_+i] = coord[i];
   return 1;
}

//*************************************************************************
// set node boundary condition 
//-------------------------------------------------------------------------

int MLI_FEData::loadNodeEssBCs(int nnodes, int *nodeIDs, int *dofList, 
                               double *val)
{
   int    i, node, index, *iarray, *iarray2;
   double *darray;

   if ( numLocalNodes_ == 0 )
   {
      printf("MLI_FEData ERROR : loadNodeEssBCs - beginInitNodeSet ?\n");
      return 0;
   }

   if ( nnodes <= 0 )
   {
      printf("MLI_FEData ERROR : loadNodeEssBCs - invalid input.\n");
      return 0;
   }

   //-------------------------------------------------------------------
   // adjusting variable arrays
   //-------------------------------------------------------------------

   if ( numNodeBCs_+nnodes >= nodeBCLengMax_ )
   {
      iarray  = nodeBCList_;
      iarray2 = nodeBCDofList_;
      darray  = nodeBCValues_;
      nodeBCLengMax_ = nodeBCLengMax_ + 5 * nnodes + 1;
      nodeBCList_    = new int[nodeBCLengMax_];
      nodeBCDofList_ = new int[nodeBCLengMax_];
      nodeBCValues_  = new double[nodeBCLengMax_];
      for (i = 0; i < numNodeBCs_; i++) 
      {
         nodeBCList_[i] = iarray[i];
         nodeBCDofList_[i] = iarray2[i];
         nodeBCValues_[i] = darray[i];
      }
   }

   //-------------------------------------------------------------------
   // put new informaiton into the variable arrays
   //-------------------------------------------------------------------

   for ( i = 0; i < nnodes; i++ )
   {
      node = nodeIDs[i];
      index = searchNode( node );
      if ( index < 0 )
      {
         printf("MLI_FEData ERROR : loadNodeBC - node not local.\n");
         return 0;
      }
      nodeBCList_[numNodeBCs_]     = index;
      nodeBCDofList_[numNodeBCs_]  = dofList[i];
      nodeBCValues_[numNodeBCs_++] = val[i];
   }
   return 1;
}

//*************************************************************************
// load shared node list 
//-------------------------------------------------------------------------

int MLI_FEData::loadSharedNodes(int nnodes, int *nodeList, int *procLeng, 
                                int **nodeProc)
{
   int i, j;
   
   sharedNodeLeng_ = new int[nnodes];
   sharedNodeID_   = new int[nnodes];
   sharedNodeProc_ = new int*[nnodes];

   for(i=0; i<nnodes; i++)
     sharedNodeProc_[i] = NULL;

   if ( numLocalNodes_ == 0 )
   {
      printf("MLI_FEData ERROR : loadNodeEssBCs - beginInitNodeSet ?\n");
      return 0;
   }

   for ( i = 0; i < nnodes; i++ )
   {
      if ( procLeng[i] <= 0 )
      {
         printf("MLI_FEData ERROR : loadSharedNodes - procLeng not valid.\n");
         return 0;
      }

      sharedNodeID_[i]   = nodeList[i];
      sharedNodeLeng_[i] = procLeng[i];
      sharedNodeProc_[i] = new int[procLeng[i]];
      for ( j = 0; j < procLeng[i]; j++ )
         sharedNodeProc_[i][j] =  nodeProc[i][j];
      numSharedNodes_++;
   } 
   return 1;
}

//*************************************************************************
// begin initializing edge set
//-------------------------------------------------------------------------

int MLI_FEData::beginInitEdgeSet()
{
   int    i, j, index, totalEdges, count, count2, *edge_array;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : beginInitEdgeSet - elemSet not done.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("MLI_FEData :beginInitEdgeSet\n");

   totalEdges = 0;
   for ( i = 0; i < numLocalElems_; i++ ) totalEdges += elemEdgeLeng_[i];
   edge_array = new int[totalEdges];
   count = 0;
   for ( i = 0; i < numLocalElems_; i++ ) 
   {
      for ( j = 0; j < elemEdgeLeng_[i]; j++ ) 
         edge_array[count++] = elemEdgeList_[i][j];
   }
   intSort2(edge_array, NULL, 0, count-1);
   count2 = 1;
   for ( i = 1; i < count; i++ ) 
      if ( edge_array[i] != edge_array[i-1] ) 
         edge_array[count2++] = edge_array[i];

   numLocalEdges_ = count2;
   edgeGlobalID_ = new int[count2];
   for ( i = 0; i < count2; i++ ) edgeGlobalID_[i] = edge_array[i]; 

   edgeNodeList_ = new int*[numLocalEdges_]; 
   for ( i = 0; i < numLocalEdges_; i++ ) edgeNodeList_[i] = NULL;
   return 1;
}

//*************************************************************************
// begin initializing face set
//-------------------------------------------------------------------------

int MLI_FEData::beginInitFaceSet()
{
   int i, j, index, totalFaces, count, count2, *face_array;
   
   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : beginInitFaceSet - elemSet not done.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("MLI_FEData :beginInitFaceSet\n");

   // TTT
   // cleanFaceInfo();

   totalFaces = 0;
   for ( i = 0; i < numLocalElems_; i++ ) 
     totalFaces += elemFaceLeng_[i];
  
   face_array = new int[totalFaces];
   count = 0;
   for ( i = 0; i < numLocalElems_; i++ ) 
   {
      for ( j = 0; j < elemFaceLeng_[i]; j++ ) 
         face_array[count++] = elemFaceList_[i][j];
   }
   intSort2(face_array, NULL, 0, count-1);
   count2 = 1;
   for ( i = 1; i < count; i++ ) 
      if ( face_array[i] != face_array[i-1] ) 
         face_array[count2++] = face_array[i];

   numLocalFaces_ = count2;
   numExternalFaces_ = 0;

   faceGlobalID_ = new int[count2];
   for ( i = 0; i < count2; i++ ) faceGlobalID_[i] = face_array[i]; 

   faceDOF_ = new int[numLocalFaces_];
   for ( i = 0; i < numLocalFaces_; i++ ) faceDOF_[i] = 0;

   faceNodeList_ = new int*[numLocalFaces_]; 
   for ( i = 0; i < numLocalFaces_; i++ ) faceNodeList_[i] = NULL;
   
   faceNodeLeng_ = new int[numLocalFaces_];

   delete [] face_array;
   return 1;
}

//*************************************************************************
// terminate initializing edge information 
//-------------------------------------------------------------------------

int MLI_FEData::endInitEdgeSet()
{
   int i;

   for ( i = 0; i < numLocalEdges_; i++ )
   {
      if ( edgeNodeList_[i] == NULL )
      {
         printf("MLI_FEData ERROR : endInitEdgeSet - not complete.\n");
         return 0;
      }
   }
   processEdgeFlag_ = 1;
   return 1;
}

//*************************************************************************
// terminate initializing face information 
//-------------------------------------------------------------------------

int MLI_FEData::endInitFaceSet()
{
   int    i, j, nfacesLocal, nfacesExt, *extFaceFlag, *newFaceList;
   int    count, *iarray, **darray;
   char   param_string[100];

   nfacesExt   = 0;
   extFaceFlag = new int[numLocalFaces_];

   for ( i = 0; i < numLocalFaces_; i++ ) extFaceFlag[i] = 0;

   for ( i = 0; i < numSharedFaces_; i++ )
      for ( j = 0; j < sharedFaceLeng_[i]; j++ )
         if ( sharedFaceProc_[i][j] < mypid_ ) 
	 {
	    nfacesExt++; 
	    extFaceFlag[searchFace(sharedFaceID_[i])] = - 1;
	    break;
	 }

   numExternalFaces_ = nfacesExt;
   nfacesLocal       = numLocalFaces_;
   numLocalFaces_   -= numExternalFaces_;
   newFaceList       = new int[nfacesLocal];

   count = 0;
   for ( i = 0; i < nfacesLocal; i++ )
   {
      if ( extFaceFlag[i] == 0 ) 
      {
         newFaceList[count] = faceGlobalID_[i];
         extFaceFlag[i] = count++;
      }
   }
   for ( i = 0; i < nfacesLocal; i++ )
   {
      if ( extFaceFlag[i] == - 1 ) 
      {
	/*
	  cout << " " << count << " ";
	*/
         newFaceList[count] = faceGlobalID_[i];
         extFaceFlag[i] = count++;
      }
   } 
   delete [] faceGlobalID_;
   faceGlobalID_ = newFaceList;

   if ( faceDOF_ != NULL )
   {
      iarray = faceDOF_;
      faceDOF_ = new int[nfacesLocal];
      for ( i = 0; i < nfacesLocal; i++ ) 
         faceDOF_[extFaceFlag[i]] = iarray[i];
      delete [] iarray;
   }

   if ( faceNodeList_ != NULL )
   {
      darray = faceNodeList_;
      faceNodeList_ = new int*[nfacesLocal]; 
      for ( i = 0; i < nfacesLocal; i++ )
         faceNodeList_[extFaceFlag[i]] = darray[i];
      delete [] darray;
   }
   delete [] extFaceFlag;
   
   // fix the indices for the external faces

   externalFaces_ = new int[numExternalFaces_];

   MPI_Request request;
   MPI_Status  Status;
   int         *ind = new int[numSharedFaces_], external_index;

   strcpy( param_string, "face_offset" );
   getSpecificData(param_string, &face_off);
   MPI_Barrier(MPI_COMM_WORLD);

   // send the indices of the owned shared faces

   for(i=0; i<numSharedFaces_; i++)
   {
      ind[i] = search(sharedFaceID_[i], faceGlobalID_, numLocalFaces_);
       
      // the shared face is owned by this subdomain

      if (ind[i] >= 0)
      {
         ind[i] += face_off;
         for ( j = 0; j < sharedFaceLeng_[i]; j++ )
            if (sharedFaceProc_[i][j] != mypid_)
               MPI_Isend(&ind[i], 1, MPI_INT, sharedFaceProc_[i][j], 
                         sharedFaceID_[i], MPI_COMM_WORLD, &request);
      }
   }

   // receive the indices of the external faces

   for ( i = 0; i < numExternalFaces_; i++ )
   {
      MPI_Recv( &external_index, 1, MPI_INT, MPI_ANY_SOURCE,
	        MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
      externalFaces_[ search ( Status.MPI_TAG, 
				faceGlobalID_ + numLocalFaces_, 
				numExternalFaces_ ) ] = external_index;
   }
   MPI_Barrier(MPI_COMM_WORLD);
   
   delete [] ind;
   return 1;
}

//*************************************************************************
// set face degree of freedom 
//-------------------------------------------------------------------------

int MLI_FEData::loadFaceDOF(int faceID, int dof)
{
   int i, index;

   if ( numLocalFaces_ == 0 )
   {
      printf("MLI_FEData ERROR : loadFaceDOF - beginInitFaceSet ?\n");
      return 0;
   }
   else if ( faceDOF_ == NULL )
   {
      printf("MLI_FEData ERROR : loadFaceDOF - faceDOF not initialized.\n");
      return 0;
   }
   if ( faceID == - 1 )
   {
      for ( i = 0; i < numLocalFaces_+numExternalFaces_; i++ ) 
         faceDOF_[i] = dof;
   }
   else
   {
      index = searchFace( faceID );
      if ( index < 0 )
      {
         printf("MLI_FEData ERROR : loadFaceDOF - face not local.\n");
         return 0;
      }
      faceDOF_[index] = dof;
   }
   return 1;
}

//*************************************************************************
// load edge node list
//-------------------------------------------------------------------------

int MLI_FEData::loadEdgeNodeList(int edgeID, int *nodeList)
{
   int    i, index, totalEdges, count, count2, *edge_array;

   index = searchEdge( edgeID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : loadEdgeNodeList - edge not local.\n");
      return 0;
   }
   if ( edgeNodeList_ == NULL )
   {
      printf("MLI_FEData ERROR : loadEdgeNodeList - list not initialized.\n");
      return 0;
   }
   edgeNodeList_[index] = new int[2];
   edgeNodeList_[index][0] = nodeList[0];
   edgeNodeList_[index][1] = nodeList[1];
   return 1;
} 

//*************************************************************************
// load face node list
//-------------------------------------------------------------------------

int MLI_FEData::loadFaceNodeList(int faceID, int nNodes, int *nodeList)
{
   int i, index;

   if ( numLocalFaces_ <= 0 ) 
   {
      printf("MLI_FEData ERROR : loadFaceNodeList - numFaces <= 0.\n");
      return 0;
   }
   else if ( faceNodeList_ == NULL )
   {
      faceNodeList_ = new int*[numLocalFaces_];
      faceNodeLeng_ = new int[numLocalFaces_];
      for ( i = 0; i < numLocalFaces_; i++ ) 
      {
         faceNodeList_[i] = NULL;
         faceNodeLeng_[i] = 0;
      }
   }
   index   = searchFace( faceID );
   faceNodeLeng_[index] = nNodes;
   faceNodeList_[index] = new int[nNodes];
   for ( i = 0; i < nNodes; i++ ) 
      faceNodeList_[index][i] = nodeList[i];
   return 1;
} 

//*************************************************************************
// load shared face list 
//-------------------------------------------------------------------------

int MLI_FEData::loadSharedFaces ( int nfaces, int *faceList, int *procLeng,
				  int **faceProc )
{
   int i, j;

   sharedFaceLeng_ = new int[nfaces];
   sharedFaceID_   = new int[nfaces];
   sharedFaceProc_ = new int*[nfaces];

   for(i=0; i<nfaces; i++)
     sharedFaceProc_[i] = NULL;

   if ( numLocalFaces_ == 0 )
   {
      printf("MLI_FEData ERROR : loadFaceEssBCs - beginInitFaceSet ?\n");
      return 0;
   }

   for ( i = 0; i < nfaces; i++ )
   {
      if ( procLeng[i] <= 0 )
      {
         printf("MLI_FEData ERROR : loadSharedFaces - procLeng not valid.\n");
         return 0;
      }

      sharedFaceID_[i]    = faceList[i];
      sharedFaceLeng_[i] = procLeng[i];
      sharedFaceProc_[i] = new int[procLeng[i]];
      for ( j = 0; j < procLeng[i]; j++ )
         sharedFaceProc_[i][j] =  faceProc[i][j];
      numSharedFaces_++;
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
// get number of local elements 
//-------------------------------------------------------------------------

int MLI_FEData::getNumElements(int& nelems)
{
   nelems = numLocalElems_;
   return 1;
}

//*************************************************************************
// get all element globalIDs 
//-------------------------------------------------------------------------

int MLI_FEData::getElemIDs(int *gid)
{
   int i;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemIDs - elemSet not done.\n");
      return 0;
   }
   if ( elemGlobalID_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemIDs - info not available.\n");
      return 0;
   }
   for ( i = 0; i < numLocalElems_; i++ ) gid[i] = elemGlobalID_[i];
   return 1;
}

//*************************************************************************
// get all face globalIDs 
//-------------------------------------------------------------------------

int MLI_FEData::getFaceIDs ( int *gid )
{
  int i;
  
  if ( processFaceFlag_ != 1 )
  {
     printf("MLI_FEData ERROR : getFaceIDs - faceSet not done.\n");
     return 0;
  }
  if ( faceGlobalID_ == NULL || numLocalFaces_ <= 0 )
  {
     printf("MLI_FEData ERROR : getFaceIDs - info not available.\n");
     return 0;
  }
  for ( i = 0; i < numLocalFaces_; i++ ) gid[i] = faceGlobalID_[i];
  return 1;
}

//*************************************************************************
// get element nodelist size 
//-------------------------------------------------------------------------

int MLI_FEData::getElemNumNodes(int elemID, int& nnodes)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemNumNodes - elemSet not done.\n");
      return 0;
   }
   if ( elemNodeLeng_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemNumNodes - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemNumNodes - element not local.\n");
      return 0;
   }
   nnodes = elemNodeLeng_[index];
   return 1;
}

//*************************************************************************
// get face nodelist size 
//-------------------------------------------------------------------------

int MLI_FEData::getFaceNumNodes(int faceID, int &nnodes)
{
   int index;
   if ( processFaceFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getFaceNumNodes - faceSet not done.\n");
      return 0;
   }
   if ( faceNodeLeng_ == NULL || numLocalFaces_ <= 0 )
   {
      printf("MLI_FEData ERROR : getFaceNumNodes - info not available.\n");
      return 0;
   }
   index = searchFace( faceID );
   if ( index < 0 )
     return 0;
   
   nnodes = faceNodeLeng_[index];
   return 1;
}

//*************************************************************************
// get element nodelist 
//-------------------------------------------------------------------------

int MLI_FEData::getElemNodeList(int elemID, int *nodeList)
{
   int i, index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemNodeList - elemSet not done.\n");
      return 0;
   }
   else if ( elemNodeList_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemNodeList - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemNodeList - element not local.\n");
      return 0;
   }

   for ( i = 0; i < elemNodeLeng_[index]; i++ )
     nodeList[i] =  elemNodeList_[index][i];

   return 1;
}

//*************************************************************************
// get global node index 
//-------------------------------------------------------------------------

int MLI_FEData::getNodeGlobalID(int node_index, int &global_node_index)
{
   global_node_index = search ( node_index, nodeGlobalID_, numLocalNodes_ );
   if (global_node_index < 0)
   {
      global_node_index = search( node_index, nodeGlobalID_ + numLocalNodes_,
                                  numExternalNodes_);
     global_node_index = externalNodes_[global_node_index];
   }
   else global_node_index += node_off;

   return 1;
}

//*************************************************************************
// get element stiffness matrix 
//-------------------------------------------------------------------------

int MLI_FEData::getElemStiffMat(int elemID, double **ematrix)
{
   int i, j, index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemStiffMat - elemSet not done.\n");
      return 0;
   }
   else if ( elemStiff_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemStiffmat - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemStiffMat - element not local.\n");
      return 0;
   }
   for ( i = 0; i < elemNodeLeng_[index]; i++ )
      for ( j = 0; j < elemNodeLeng_[index]; j++ )
         ematrix[i][j] = elemStiff_[index][i][j];
   return 1;
}

//*************************************************************************
// get element null space size
//-------------------------------------------------------------------------

int MLI_FEData::getElemNullSpaceSize(int elemID, int& size)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemNullSpaceSize - elemSet not done.\n");
      return 0;
   }
   else if ( elemNullLeng_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemNullSpaceSize - not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemNullSpaceSize - element not local.\n");
      return 0;
   }
   size = elemNullLeng_[index];
   return 1;
}

//*************************************************************************
// get element null space 
//-------------------------------------------------------------------------

int MLI_FEData::getElemNullSpace(int elemID, double **nullSpace)
{
   int i, j, index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemNullSpace - elemSet not done.\n");
      return 0;
   }
   else if ( elemNullSpace_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemNullSpace - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemNullSpace - element not local.\n");
      return 0;
   }
   if ( elemNullSpace_[index] == NULL )
   {
      printf("MLI_FEData ERROR : getElemNullSpace - info not available.\n");
      return 0;
   }
   for ( i = 0; i < elemNullLeng_[index]; i++ )
      for ( j = 0; j < elemNodeLeng_[index]; j++ )
         nullSpace[i][j] = elemNullSpace_[index][i][j];
   return 1;
}

//*************************************************************************
// get element volume 
//-------------------------------------------------------------------------

int MLI_FEData::getElemVolume(int elemID, double& volume)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemVolume - elemSet not done.\n");
      return 0;
   }
   else if ( elemVolume_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemVolume - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemVolume - element not local.\n");
      return 0;
   }
   volume = elemVolume_[index];
   return 1;
}

//*************************************************************************
// get element material 
//-------------------------------------------------------------------------

int MLI_FEData::getElemMaterial(int elemID, int& material)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemMaterial - elemSet not done.\n");
      return 0;
   }
   else if ( elemMaterial_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemMaterial - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemMaterial - element not local.\n");
      return 0;
   }
   material = elemMaterial_[index];
   return 1;
}

//*************************************************************************
// get element number of edges 
//-------------------------------------------------------------------------

int MLI_FEData::getElemNumEdges(int elemID, int& numEdges)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemNumEdges - elemSet not done.\n");
      return 0;
   }
   else if ( elemEdgeLeng_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemNumEdges - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemNumEdges - element not local.\n");
      return 0;
   }
   numEdges = elemEdgeLeng_[index];
   return 1;
}

//*************************************************************************
// get element edge list 
//-------------------------------------------------------------------------

int MLI_FEData::getElemEdgeList(int elemID, int *edgeList)
{
   int i, index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemEdgeList - elemSet not done.\n");
      return 0;
   }
   else if ( elemEdgeList_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemEdgeList - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemEdgeList - element not local.\n");
      return 0;
   }
   for ( i = 0; i < elemEdgeLeng_[index]; i++ )
      edgeList[i] = elemEdgeList_[index][i];
   return 1;
}

//*************************************************************************
// get element number of faces 
//-------------------------------------------------------------------------

int MLI_FEData::getElemNumFaces(int elemID, int& numFaces)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemNumFaces - elemSet not done.\n");
      return 0;
   }
   else if ( elemFaceLeng_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemNumFaces - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemNumFaces - element not local.\n");
      return 0;
   }
   numFaces = elemFaceLeng_[index];
   return 1;
}

//*************************************************************************
// get element face list 
//-------------------------------------------------------------------------

int MLI_FEData::getElemFaceList(int elemID, int *faceList)
{
   int i, index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemFaceList - elemSet not done.\n");
      return 0;
   }
   else if ( elemFaceList_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemFaceList - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemFaceList - element not local.\n");
      return 0;
   }
   
   for ( i = 0; i < elemFaceLeng_[index]; i++ )
     faceList[i] = elemFaceList_[index][i];

   return 1;
}

//*************************************************************************
// get global face ID
//-------------------------------------------------------------------------

int MLI_FEData::getFaceGlobalID(int face_index, int &global_face_index)
{
   global_face_index = search ( face_index, faceGlobalID_, numLocalFaces_ );
   if ( global_face_index < 0 ) 
   {
      global_face_index = search(face_index, faceGlobalID_ + numLocalFaces_,
                                 numExternalFaces_);
      global_face_index = externalFaces_[global_face_index];
   }
   else global_face_index += face_off;
  
   return 1;
}

//*************************************************************************
// get element's parent ID 
//-------------------------------------------------------------------------

int MLI_FEData::getElemParentID(int elemID, int& parentID)
{
   int i, index;

   if ( processElemFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getElemParentID - elemSet not done.\n");
      return 0;
   }
   else if ( elemParentID_ == NULL || numLocalElems_ <= 0 )
   {
      printf("MLI_FEData ERROR : getElemParentID - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getElemParentID - element not local.\n");
      return 0;
   }
   parentID = elemParentID_[index];
   return 1;
}

//*************************************************************************
// get number of local nodes 
//-------------------------------------------------------------------------

int MLI_FEData::getNumLocalNodes(int& node)
{
   int status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNumLocalNodes - not initialized.\n");
      return 0;
   }
   node = numLocalNodes_;
   return 1;
}

//*************************************************************************
// get number of external nodes 
//-------------------------------------------------------------------------

int MLI_FEData::getNumExternalNodes(int& node)
{
   int  status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNumExternalNodes - not initialized.\n");
      return 0;
   }
   node = numExternalNodes_;
   return 1;
}

//*************************************************************************
// get node local ID 
//-------------------------------------------------------------------------

int MLI_FEData::getNodeLocalID(int nodeID, int &localID)
{
   int index, status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNodeLocalID - not initialized.\n");
      return 0;
   }
   index = searchNode( nodeID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getNodeLocalID - node not local.\n");
      return 0;
   }
   localID = index;
   return 1;
}

//*************************************************************************
// get node degree of freedom 
//-------------------------------------------------------------------------

int MLI_FEData::getNodeDOF(int nodeID, int& dof)
{
   int i, index, status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNodeDOF - not initialized.\n");
      return 0;
   }
   index = searchNode( nodeID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getNodeDOF - node not local.\n");
      return 0;
   }
   dof = nodeDOF_[index];
   return 1;
}

//*************************************************************************
// get node coordinates 
//-------------------------------------------------------------------------

int MLI_FEData::getNodeCoordinate(int nodeID, double *coord)
{
   int  i, index, status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNodeCoordinate - not initialized.\n");
      return 0;
   }
   index = searchNode( nodeID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getNodeCoordinate - node not local.\n");
      return 0;
   }
   for ( i = 0; i < spaceDimension_; i++ )
      coord[i] = nodeCoordinates_[spaceDimension_*index+i];
   return 1;
}

//*************************************************************************
// get node degree of freedom 
//-------------------------------------------------------------------------

int MLI_FEData::getNodeEssBCs(int& numBCs, int **nodeList, int **dofList, 
                              double **val)
{
   int  status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNodeEssBCs - not initialized.\n");
      return 0;
   }
   numBCs = numNodeBCs_;
   (*nodeList) = nodeBCList_;
   (*dofList)  = nodeBCDofList_;
   (*val)      = nodeBCValues_;
   return 1;
}

//*************************************************************************
// get number of shared nodes 
//-------------------------------------------------------------------------

int MLI_FEData::getNumSharedNodes(int& nnodes)
{
   int  status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNumSharedNodes - not initialized.\n");
      return 0;
   }
   nnodes = numSharedNodes_;
   return 1;
}

//*************************************************************************
// get shared nodes list 
//-------------------------------------------------------------------------

int MLI_FEData::getSharedNodeInfo(int nNodes, int *nodeList, int *procLeng)
{
   int  i, count, status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
     printf("MLI_FEData ERROR : getSharedNodesInfo - not initialized.\n");
      return 0;
   }
   if ( nNodes != numSharedNodes_ )
   {
      printf("MLI_FEData ERROR : getSharedNodesInfo - wrong no. of nodes.\n");
      return 0;
   }
   count = 0;
   for ( i = 0; i < numLocalNodes_; i++ )
   {
      if ( sharedNodeProc_[i] != NULL )
      {
         nodeList[count] = nodeGlobalID_[i];
         procLeng[count] = sharedNodeLeng_[i];
         count++;
      } 
   }
   return 1;
}

//*************************************************************************
// get shared nodes 
//-------------------------------------------------------------------------

int MLI_FEData::getSharedNodeProc(int nodeID, int *procList)
{
   int  i, index, status = 1;

   if ( processNodeFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getSharedNodeProc - not initialized.\n");
      return 0;
   }
   index = searchNode( nodeID );
   if ( index < 0 || sharedNodeProc_[index] == NULL )
   {
      printf("MLI_FEData ERROR : getSharedNodeProc - nodeID invalid.\n");
      return 0;
   }
   for ( i = 0; i < sharedNodeLeng_[index]; i++ )
      procList[i] = sharedNodeProc_[index][i];
   return 1;
}
    
//*************************************************************************
// get edge node list 
//-------------------------------------------------------------------------

int MLI_FEData::getEdgeNodeList(int edgeID, int *nodeList)
{
   int index;

   if ( processEdgeFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getEdgeNodeList - edgeSet not done.\n");
      return 0;
   }
   else if ( edgeNodeList_ == NULL || numLocalEdges_ <= 0 )
   {
      printf("MLI_FEData ERROR : getEdgeNodeList - info not available.\n");
      return 0;
   }
   index = searchEdge( edgeID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getEdgeNodeList - edge not local.\n");
      return 0;
   }
   nodeList[0] = edgeNodeList_[index][0];
   nodeList[1] = edgeNodeList_[index][1];
   return 1;
}
//*************************************************************************
// get face node list 
//-------------------------------------------------------------------------

int MLI_FEData::getFaceNodeList(int faceID, int *nodeList)
{
   int i, index;

   if ( processFaceFlag_ != 1 )
   {
      printf("MLI_FEData ERROR : getFaceNodeList - faceSet not done.\n");
      return 0;
   }
   else if ( faceNodeList_ == NULL || numLocalFaces_ <= 0 )
   {
      printf("MLI_FEData ERROR : getFaceNodeList - info not available.\n");
      return 0;
   }
   index = searchFace( faceID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getFaceNodeList - face not local.\n");
      return 0;
   }

   for ( i = 0; i < faceNodeLeng_[index]; i++ )
     nodeList[i] = faceNodeList_[index][i];
   
   return 1;
}

//*************************************************************************
// get the number of Local faces
//-------------------------------------------------------------------------

int MLI_FEData::getNumLocalFaces(int& nfaces)
{
  int status = 1;
  
  if ( processFaceFlag_ == 0 ) status = processElemInfo();
  if ( status == 0 )
    {
      printf("MLI_FEData ERROR : getNumLocalFaces - not initialized.\n");
      return 0;
    }

  nfaces = numLocalFaces_;
  return 1;
}

//*************************************************************************
// get number of external faces
//-------------------------------------------------------------------------

int MLI_FEData::getNumExternalFaces( int & faces )
{
   int  status = 1;

   if ( processFaceFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNumExternalFaces - not initialized.\n");
      return 0;
   }
   faces = numExternalFaces_;
   return 1;
}

//*************************************************************************
// get face local ID 
//-------------------------------------------------------------------------

int MLI_FEData::getFaceLocalID(int faceID, int& localID)
{
   int index, status = 1;

   if ( processFaceFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getFaceLocalID - not initialized.\n");
      return 0;
   }
   index = searchFace( faceID );
   if ( index < 0 )
   {
      // printf("MLI_FEData ERROR : getFaceLocalID - face not local.\n");
      return 0;
   }
   localID = index;
   return 1;  
}

//*************************************************************************
// get face degree of freedom 
//-------------------------------------------------------------------------

int MLI_FEData::getFaceDOF(int faceID, int& dof)
{
   int i, index, status = 1;

   if ( processFaceFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getFaceDOF - not initialized.\n");
      return 0;
   }
   index = searchFace( faceID );
   if ( index < 0 )
   {
      printf("MLI_FEData ERROR : getFaceDOF - face not local.\n");
      return 0;
   }
   dof = faceDOF_[index];
   return 1;
}

//*************************************************************************
// get number of shared faces 
//-------------------------------------------------------------------------

int MLI_FEData::getNumSharedFaces(int& nfaces)
{
   int  status = 1;

   if ( processFaceFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getNumSharedFaces - not initialized.\n");
      return 0;
   }
   nfaces = numSharedFaces_;
   return 1;
}

//*************************************************************************
// get shared faces list 
//-------------------------------------------------------------------------

int MLI_FEData::getSharedFaceInfo(int nFaces, int *faceList, int *procLeng)
{
   int  i, count, status = 1;

   if ( processFaceFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getSharedFacesInfo - not initialized.\n");
      return 0;
   }
   if ( nFaces != numSharedFaces_ )
   {
      printf("MLI_FEData ERROR : getSharedFacesInfo - wrong no. of faces.\n");
      return 0;
   }
   count = 0;
   for ( i = 0; i < numLocalFaces_; i++ )
   {
      if ( sharedFaceProc_[i] != NULL )
      {
         faceList[count] = faceGlobalID_[i];
         procLeng[count] = sharedFaceLeng_[i];
         count++;
      } 
   }
   return 1;
}

//*************************************************************************
// get shared faces 
//-------------------------------------------------------------------------

int MLI_FEData::getSharedFaceProc(int faceID, int *procList)
{
   int  i, index, status = 1;

   if ( processFaceFlag_ == 0 ) status = processElemInfo();
   if ( status == 0 )
   {
      printf("MLI_FEData ERROR : getSharedFaceProc - not initialized.\n");
      return 0;
   }
   index = searchFace( faceID );
   if ( index < 0 || sharedFaceProc_[index] == NULL )
   {
      printf("MLI_FEData ERROR : getSharedFaceProc - faceID invalid.\n");
      return 0;
   }
   for ( i = 0; i < sharedFaceLeng_[index]; i++ )
      procList[i] = sharedFaceProc_[index][i];
   return 1;
}

//*************************************************************************
// get shape function interpolant 
//-------------------------------------------------------------------------

int MLI_FEData::getShapeFuncInterpolant(int element, int nn, double *coord,
                                        double *coef)
{
   (void) element;
   (void) nn;
   (void) coord;
   (void) coef;
   return 0;
}

//*************************************************************************
// get specific (specified by data_key) information in data
//-------------------------------------------------------------------------

int MLI_FEData::getSpecificData(char *data_key, void *data)
{
   int        numprocs, offset, flag=1;
   MPI_Status Status;

   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  
   if ( strcmp("elem_offset",data_key)==0 )
   {
      MPI_Barrier(MPI_COMM_WORLD);
      *(int *)data = 0;
      if (mypid_ != 0)
         MPI_Recv( data, 1, MPI_INT, MPI_ANY_SOURCE,
                   MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
     
      if (mypid_ + 1 < numprocs)
      {
         offset = numLocalElems_ + *(int *)data;
         MPI_Send(&offset, 1, MPI_INT, mypid_+1, flag, MPI_COMM_WORLD);
      }
      return 1;
   }
   else if (strcmp("node_offset", data_key)==0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      *(int *)data = 0;
      if (mypid_ != 0)
         MPI_Recv( data, 1, MPI_INT, MPI_ANY_SOURCE,
                   MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
    
      if (mypid_ + 1 < numprocs)
      {
         offset = numLocalNodes_ + *(int *)data;
         MPI_Send(&offset, 1, MPI_INT, mypid_+1, flag, MPI_COMM_WORLD);
      }
      return 1;
   }
   else if (strcmp("edge_offset", data_key)==0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      *(int *)data = 0;
      if (mypid_ != 0)
         MPI_Recv( data, 1, MPI_INT, MPI_ANY_SOURCE,
                   MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
    
      if (mypid_ + 1 < numprocs)
      {
         offset = numLocalEdges_ + *(int *)data;
         MPI_Send(&offset, 1, MPI_INT, mypid_+1, flag, MPI_COMM_WORLD);
      }
      return 1;
   }
   else if (strcmp("face_offset", data_key)==0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      *(int *)data = 0;
      if (mypid_ != 0)
         MPI_Recv( data, 1, MPI_INT, MPI_ANY_SOURCE,
                   MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
     
      if (mypid_ + 1 < numprocs)
      {
         offset = numLocalFaces_ + *(int *)data;
         MPI_Send(&offset, 1, MPI_INT, mypid_+1, flag, MPI_COMM_WORLD);
      }
      return 1;
   }

   // return the owners for the external nodes

   else if (strcmp("external_node_owners", data_key)==0)
   {
      int i, j, ind, *owners = (int *)data;
      MPI_Request request;

      MPI_Barrier(MPI_COMM_WORLD);
      for ( i = 0; i < numSharedNodes_; i++ )
      {
         ind = search(sharedNodeID_[i], nodeGlobalID_, numLocalNodes_);
        
         // the shared node is owned by this subdomain
         if (ind >= 0)
 	    for ( j = 0; j < sharedNodeLeng_[i]; j++ )
 	       if (sharedNodeProc_[i][j] != mypid_)
                  MPI_Isend(sharedNodeID_+i, 1, MPI_INT, sharedNodeProc_[i][j], 
                            mypid_, MPI_COMM_WORLD, &request);
	  
      }
      for(i=0; i<numExternalNodes_; i++)
      {
         MPI_Recv( &j, 1, MPI_INT, MPI_ANY_SOURCE,
                   MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
         owners[ search ( j, nodeGlobalID_ + numLocalNodes_, 
                          numExternalNodes_ ) ] = Status.MPI_TAG; 
      }
      MPI_Barrier(MPI_COMM_WORLD);
    
      return 1;
   }
   return 0;
}

//*************************************************************************
// get specific (specified by data_key) information in data
//-------------------------------------------------------------------------

int MLI_FEData::getSpecificData(char *data_key, void *data1, void *data2)
{
   int         numprocs, i, j;
   MPI_Request request;
   MPI_Status  Status;
   char        param_string[100];

   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  
   // Until this point table node_element has only local elements;
   // here the elements for owned shared nodes are updated  

   if (strcmp("update_node_elements",data_key)==0)
   {
      MPI_Barrier(MPI_COMM_WORLD);

      int Buf[100];
      int *ncols = (int *)data1, **cols = (int **)data2, n;
      int *ind = new int[numSharedNodes_];
      int *columns, l, k, *p;
      
      // get the owners for the external nodes

      int *owner = new int [numExternalNodes_];
      strcpy( param_string, "external_node_owners" );
      getSpecificData(param_string, owner);
      
      // external nodes send with which elements are connected

      for ( i = 0; i < numExternalNodes_; i++ )
         MPI_Isend(cols[i+numLocalNodes_], ncols[i+numLocalNodes_], MPI_INT, 
		   owner[i], nodeGlobalID_[i+numLocalNodes_], 
		   MPI_COMM_WORLD, &request);
      
      // owners of shared nodes receive data

      for(i=0; i<numSharedNodes_; i++)
      {
         ind[i] = search(sharedNodeID_[i], nodeGlobalID_, numLocalNodes_);
	  
         // the shared node is owned by this subdomain

         if (ind[i] >= 0)
         {
            for ( j = 0; j < sharedNodeLeng_[i]; j++ )
               if (sharedNodeProc_[i][j] != mypid_)
               {
                  MPI_Recv( Buf, 100, MPI_INT, MPI_ANY_SOURCE,
                            MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
                  MPI_Get_count( &Status, MPI_INT, &n);
                  k = search( Status.MPI_TAG,nodeGlobalID_,numLocalNodes_);
                  columns = new int[ncols[k]+n];
                  for ( l = 0; l < ncols[k]; l++ ) columns[l] = cols[k][l];
                  for ( l = 0; l < n; l++ ) columns[ncols[k]++] = Buf[l];
                  delete [] cols[k];
                  cols[k] = columns;
               }
         }
      }
      delete [] ind;
      delete [] owner;
      
      return 1;
   }

   // The same as above but for the faces

   else if (strcmp("update_face_elements",data_key)==0)
   {
      MPI_Barrier(MPI_COMM_WORLD);

      int Buf[100];
      int *ncols = (int *)data1, **cols = (int **)data2, n;
      int *ind = new int[numSharedFaces_];
      int *columns, l, k, *p;
      
      // get the owners for the external faces

      int *owner = new int [numExternalFaces_];
      strcpy( param_string, "external_face_owners" );
      getSpecificData( param_string, owner);
      
      // external faces send with which elements are connected

      for ( i = 0; i < numExternalFaces_; i++ )
         MPI_Isend(cols[i+numLocalFaces_], ncols[i+numLocalFaces_], MPI_INT, 
		   owner[i], faceGlobalID_[i+numLocalFaces_], 
		   MPI_COMM_WORLD, &request);
      
      // owners of shared faces receive data

      for ( i = 0; i < numSharedFaces_; i++ )
      {
         ind[i] = search(sharedFaceID_[i], faceGlobalID_, numLocalFaces_);
	  
	 // the shared face is owned by this subdomain

	 if (ind[i] >= 0)
	 {
	    for ( j = 0; j < sharedFaceLeng_[i]; j++ )
               if (sharedFaceProc_[i][j] != mypid_)
	       {
                  MPI_Recv( Buf, 100, MPI_INT, MPI_ANY_SOURCE,
                            MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
                  MPI_Get_count( &Status, MPI_INT, &n);
                  k = search( Status.MPI_TAG,faceGlobalID_,numLocalFaces_);
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
// load in the function to calculate shape function interpolant 
//-------------------------------------------------------------------------

int MLI_FEData::loadFunc_computeShapeFuncInterpolant(int (*func)
                (void*,int elem,int nnodes,double *coord,double *coef))
{
   (void) func;
   return 0;
}

//*************************************************************************
// cleanup the element storage 
//-------------------------------------------------------------------------

int MLI_FEData::cleanElemInfo()
{
   int i, j;

   if ( elemStiff_ != NULL ) 
   {
      for ( i = 0; i < numLocalElems_; i++ ) 
      {
         if ( elemStiff_[i] != NULL )
         {
            for ( j = 0; j < elemNodeLeng_[i]; j++ ) 
               if ( elemStiff_[i][j] != NULL ) delete elemStiff_[i][j];
            delete elemStiff_[i];
         }
      }
      delete elemStiff_;
   }
   elemStiff_ = NULL;

   if ( elemNodeLeng_ != NULL ) delete [] elemNodeLeng_;
   elemNodeLeng_ = NULL;

   if ( elemNodeList_ != NULL ) 
   {
      for ( i = 0; i < numLocalElems_; i++ ) 
         if ( elemNodeList_[i] != NULL ) delete elemNodeList_[i];
      delete [] elemNodeList_;
   }
   elemNodeList_ = NULL;

   if ( elemGlobalID_ != NULL ) delete [] elemGlobalID_;
   elemGlobalID_ = NULL;

   if ( elemNullSpace_ != NULL ) 
   {
      for ( i = 0; i < numLocalElems_; i++ ) 
      {
         if ( elemNullSpace_[i] != NULL )
         {
            for ( j = 0; j < elemNullLeng_[i]; j++ ) 
               if ( elemNullSpace_[i][j] != NULL )
                  delete [] elemNullSpace_[i][j];
            delete [] elemNullSpace_[i];
         }
      }
      delete [] elemNullSpace_;
   }
   elemNullSpace_   = NULL;

   if ( elemNullLeng_ != NULL ) delete [] elemNullLeng_;
   elemNullLeng_ = NULL;

   if ( elemVolume_ != NULL ) delete [] elemVolume_;
   elemVolume_ = NULL;

   if ( elemMaterial_ != NULL ) delete [] elemMaterial_;
   elemMaterial_ = NULL;

   if ( elemEdgeLeng_ != NULL ) delete [] elemEdgeLeng_;
   elemEdgeLeng_ = NULL;

   if ( elemEdgeList_ != NULL ) 
   {
      for ( i = 0; i < numLocalElems_; i++ ) 
         if ( elemEdgeList_[i] != NULL ) delete [] elemEdgeList_[i];
      delete [] elemEdgeList_;
   }
   elemEdgeList_ = NULL;

   if ( elemFaceLeng_ != NULL ) delete [] elemFaceLeng_;
   elemFaceLeng_ = NULL;

   if ( elemFaceList_ != NULL ) 
   {
      for ( i = 0; i < numLocalElems_; i++ ) 
         if ( elemFaceList_[i] != NULL ) delete [] elemFaceList_[i];
      delete [] elemFaceList_;
   }
   elemFaceList_ = NULL;

   if ( elemParentID_ != NULL ) delete [] elemParentID_;
   elemParentID_ = NULL;

   numLocalElems_ = 0;
   processElemFlag_ = 0;

   return 1;
}

//*************************************************************************
// cleanup the node storage 
//-------------------------------------------------------------------------

int MLI_FEData::cleanNodeInfo()
{
   int i;

   numNodeBCs_ = 0;

   if ( nodeBCList_ != NULL ) delete [] nodeBCList_;
   nodeBCList_ = NULL;

   if ( nodeBCValues_ != NULL ) delete [] nodeBCValues_; 
   nodeBCValues_ = NULL;

   if ( nodeBCDofList_ != NULL ) delete [] nodeBCDofList_; 
   nodeBCDofList_ = NULL;

   if ( nodeCoordinates_ != NULL ) delete [] nodeCoordinates_;
   nodeCoordinates_ = NULL;

   if ( sharedNodeProc_  != NULL ) 
   {
      for ( i = 0; i < numSharedNodes_; i++ ) 
         if ( sharedNodeProc_[i] != NULL ) delete sharedNodeProc_[i];
      delete [] sharedNodeProc_;
   }
   sharedNodeProc_ = NULL;

   if ( sharedNodeLeng_  != NULL ) delete [] sharedNodeLeng_;
   sharedNodeLeng_ = NULL;

   if ( sharedNodeID_ != NULL ) delete [] sharedNodeID_;
   sharedNodeID_ = NULL;

   if ( nodeDOF_ != NULL ) delete [] nodeDOF_;
   nodeDOF_ = NULL;
   
   if ( nodeGlobalID_  != NULL ) delete [] nodeGlobalID_;
   nodeGlobalID_ = NULL;

   if ( externalNodes_ != NULL ) delete [] externalNodes_;
   externalNodes_ = NULL;
   
   numLocalNodes_ = 0;
   numExternalNodes_ = 0;
   nodeBCLengMax_    = 0;
   processNodeFlag_ = 0;
   return 1;
}

//*************************************************************************
// cleanup the edge storage 
//-------------------------------------------------------------------------

int MLI_FEData::cleanEdgeInfo()
{
   int i;

   if ( edgeGlobalID_ != NULL ) delete [] edgeGlobalID_;
   edgeGlobalID_ = NULL;

   if ( edgeNodeList_  != NULL ) 
   {
      for ( i = 0; i < numLocalEdges_; i++ ) 
         if ( edgeNodeList_[i] != NULL ) delete edgeNodeList_[i];
      delete [] edgeNodeList_;
   }
   edgeNodeList_ = NULL;
   processEdgeFlag_ = 0;
   numLocalEdges_ = 0;
   return 1;
}

//*************************************************************************
// cleanup the face storage 
//-------------------------------------------------------------------------

int MLI_FEData::cleanFaceInfo()
{
   int i;

   numFaceBCs_ = 0;

   if ( faceBCList_ != NULL ) delete [] faceBCList_;
   faceBCList_ = NULL;

   if ( faceBCValues_ != NULL ) delete [] faceBCValues_; 
   faceBCValues_ = NULL;

   if ( faceBCDofList_ != NULL ) delete [] faceBCDofList_; 
   faceBCDofList_ = NULL;

   if ( sharedFaceProc_  != NULL ) 
   {
      for ( i = 0; i < numSharedFaces_; i++ ) 
         if ( sharedFaceProc_[i] != NULL ) delete sharedFaceProc_[i];
      delete [] sharedFaceProc_;
   }
   sharedFaceProc_ = NULL;

   if ( sharedFaceLeng_  != NULL ) delete [] sharedFaceLeng_;
   sharedFaceLeng_ = NULL;

   if ( sharedFaceID_ != NULL ) delete [] sharedFaceID_;
   sharedFaceID_ = NULL;

   if ( faceDOF_ != NULL ) delete [] faceDOF_;
   faceDOF_ = NULL;
   
   if ( faceGlobalID_  != NULL ) delete [] faceGlobalID_;
   faceGlobalID_ = NULL;
   
   numLocalFaces_    = 0;
   numExternalFaces_ = 0;
   faceBCLengMax_    = 0;
   processFaceFlag_  = 0;

   if ( faceGlobalID_ != NULL ) delete [] faceGlobalID_;
   faceGlobalID_ = NULL;

   if ( faceNodeList_ != NULL ) 
   {
      for ( i = 0; i < numLocalFaces_; i++ ) 
         if ( faceNodeList_[i] != NULL ) delete faceNodeList_[i];
      delete [] faceNodeList_;
   }
   if ( faceNodeLeng_ != NULL)
     delete [] faceNodeLeng_;

   if ( externalFaces_ != NULL ) delete [] externalFaces_;
   externalFaces_ = NULL;

   faceNodeList_ = NULL;
   faceNodeLeng_ = NULL;

   return 1;
}

//*************************************************************************
// clean up everything 
//-------------------------------------------------------------------------

int MLI_FEData::cleanAll()
{
   cleanElemInfo();
   cleanNodeInfo();
   cleanEdgeInfo();
   cleanFaceInfo();
   spaceDimension_ = -1;
   orderOfPDE_     = -1;
   outputLevel_    = 0;
   USR_computeShapeFuncInterpolant = NULL;
   return 1;
}

//*************************************************************************
// set up element stuff 
//-------------------------------------------------------------------------

int MLI_FEData::processElemInfo()
{
  processNodeFlag_ = processFaceFlag_ = 1;
  return 1;
/*
   int i, index, *intarray;

   int      *new_elemNodeLeng_  = elemNodeLeng_;
   int     **new_elemNodeList_  = elemNodeList_;
   int      *new_elemGlobalID_  = elemGlobalID_;
   double ***new_elemStiff_     = elemStiff_;
   double ***new_elemNullSpace_ = elemNullSpace_;
   int      *new_elemNullLeng_  = elemNullLeng_;
   double   *new_elemVolume_    = elemVolume_;
   int      *new_elemMaterial_  = elemMaterial_;
   int      *new_elemEdgeLeng_  = elemEdgeLeng_;
   int     **new_elemEdgeList_  = elemEdgeList_;
   int      *new_elemFaceLeng_  = elemFaceLeng_;
   int     **new_elemFaceList_  = elemFaceList_; 
   int      *new_elemParentID_  = elemParentID_;

   intarray = new int[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) intarray[i] = i;
   intSort2(elemGlobalID_, intarray, 0, numLocalElems_-1);
  
   for ( i = 1; i < numLocalElems_; i++ ) 
   { 
      if ( elemGlobalID_[i] == elemGlobalID_[i-1] )
         printf("MLI_FEData ERROR : processNodeInfo - duplicate elemIDs.\n");
   }

   for ( i = 0; i < numLocalElems_; i++ ) 
   {
     index = intarray[i];
      
     if (elemNodeLeng_ != NULL)
       elemNodeLeng_[index]  = new_elemNodeLeng_[i];
     if (elemNodeList_ != NULL)
       elemNodeList_[index]  = new_elemNodeList_[i];
     if (elemGlobalID_ != NULL)
       elemGlobalID_[index]  = new_elemGlobalID_[i];
     if (elemStiff_ != NULL)
       elemStiff_[index]     = new_elemStiff_[i];
     if (elemNullSpace_ != NULL)
       elemNullSpace_[index] = new_elemNullSpace_[i];
     if (elemNullLeng_ != NULL)
       elemNullLeng_[index]  = new_elemNullLeng_[i];
     if (elemVolume_ != NULL)
       elemVolume_[index]    = new_elemVolume_[i];
     if (elemMaterial_ != NULL)
       elemMaterial_[index]  = new_elemMaterial_[i];
     if (elemEdgeLeng_ != NULL)
       elemEdgeLeng_[index]  = new_elemEdgeLeng_[i];
     if (elemEdgeList_ != NULL)
       elemEdgeList_[index]  = new_elemEdgeList_[i];
     if (elemFaceLeng_ != NULL)
       elemFaceLeng_[index]  = new_elemFaceLeng_[i];
     if (elemFaceList_ != NULL)
       elemFaceList_[index]  = new_elemFaceList_[i]; 
     if (elemParentID_ != NULL)
       elemParentID_[index]  = new_elemParentID_[i];
   }
   delete [] intarray;
   processNodeFlag_ = processFaceFlag_ = 1;

   return 1;
 */
   int i, index, *intarray;

   int      *new_elemNodeLeng_  = new int [numLocalElems_];
   int     **new_elemNodeList_  = new int*[numLocalElems_];
   int      *new_elemGlobalID_  = new int [numLocalElems_];
   double ***new_elemStiff_     = new double **[numLocalElems_];
   double ***new_elemNullSpace_ = new double **[numLocalElems_];
   int      *new_elemNullLeng_  = new int [numLocalElems_];
   double   *new_elemVolume_    = new double[numLocalElems_];
   int      *new_elemMaterial_  = new int [numLocalElems_];
   int      *new_elemEdgeLeng_  = new int [numLocalElems_];
   int     **new_elemEdgeList_  = new int*[numLocalElems_];
   int      *new_elemFaceLeng_  = new int [numLocalElems_];
   int     **new_elemFaceList_  = new int*[numLocalElems_];
   int      *new_elemParentID_  = new int [numLocalElems_];

   intarray = new int[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) intarray[i] = i;
   intSort2(elemGlobalID_, intarray, 0, numLocalElems_-1);
  
   for ( i = 1; i < numLocalElems_; i++ ) 
   { 
      if ( elemGlobalID_[i] == elemGlobalID_[i-1] )
         printf("MLI_FEData ERROR : processNodeInfo - duplicate elemIDs.\n");
   }

   for ( i = 0; i < numLocalElems_; i++ ) 
   {
     index = intarray[i];
      
     if (elemNodeLeng_ != NULL)
       new_elemNodeLeng_[i]  = elemNodeLeng_[index];
     if (elemNodeList_ != NULL)
       new_elemNodeList_[i]  = new_elemNodeList_[index];
     if (elemGlobalID_ != NULL)
       new_elemGlobalID_[i]  = new_elemGlobalID_[index];
     if (elemStiff_ != NULL)
       new_elemStiff_[i]     = new_elemStiff_[index];
     if (elemNullSpace_ != NULL)
       new_elemNullSpace_[i] = new_elemNullSpace_[index];
     if (elemNullLeng_ != NULL)
       new_elemNullLeng_[i]  = new_elemNullLeng_[index];
     if (elemVolume_ != NULL)
       new_elemVolume_[i]    = new_elemVolume_[index];
     if (elemMaterial_ != NULL)
       new_elemMaterial_[i]  = new_elemMaterial_[index];
     if (elemEdgeLeng_ != NULL)
       new_elemEdgeLeng_[i]  = new_elemEdgeLeng_[index];
     if (elemEdgeList_ != NULL)
       new_elemEdgeList_[i]  = new_elemEdgeList_[index];
     if (elemFaceLeng_ != NULL)
       new_elemFaceLeng_[i]  = new_elemFaceLeng_[index];
     if (elemFaceList_ != NULL)
       new_elemFaceList_[i]  = new_elemFaceList_[index]; 
     if (elemParentID_ != NULL)
       new_elemParentID_[i]  = new_elemParentID_[index];
   }
   delete [] intarray;
   processNodeFlag_ = processFaceFlag_ = 1;

   return 1;
}

//-------------------------------------------------------------------------

int MLI_FEData::search(int key, int *GlobalID_, int size)
{
   int  nfirst, nlast, nmid, found, index;

   if (size <= 0) return -1;
   nfirst = 0;
   nlast  = size - 1;
   if (key > GlobalID_[nlast])  return -(nlast+1);
   if (key < GlobalID_[nfirst]) return -(nfirst+1);
   found = 0;
   while ((found == 0) && ((nlast-nfirst)>1)) 
   {
      nmid = (nfirst + nlast) / 2;
      if (key == GlobalID_[nmid])     {index  = nmid; found = 1;}
      else if (key > GlobalID_[nmid])  nfirst = nmid;
      else                             nlast  = nmid;
   }
   if (found == 1)                    return index;
   else if (key == GlobalID_[nfirst]) return nfirst;
   else if (key == GlobalID_[nlast])  return nlast;
   else                               return -(nfirst+1);
}

//-------------------------------------------------------------------------

int MLI_FEData::searchElement(int key)
{
   int  nfirst, nlast, nmid, found, index;

   if (numLocalElems_ <= 0) return -1;
   nfirst = 0;
   nlast  = numLocalElems_ - 1;
   if (key > elemGlobalID_[nlast])  return -(nlast+1);
   if (key < elemGlobalID_[nfirst]) return -(nfirst+1);
   found = 0;
   while ((found == 0) && ((nlast-nfirst)>1)) 
   {
      nmid = (nfirst + nlast) / 2;
      if (key == elemGlobalID_[nmid])     {index  = nmid; found = 1;}
      else if (key > elemGlobalID_[nmid])  nfirst = nmid;
      else                                  nlast  = nmid;
   }
   if (found == 1)                         return index;
   else if (key == elemGlobalID_[nfirst]) return nfirst;
   else if (key == elemGlobalID_[nlast])  return nlast;
   else                                    return -(nfirst+1);
}

//-------------------------------------------------------------------------

int MLI_FEData::searchNode(int key)
{
  int index = search(key, nodeGlobalID_, numLocalNodes_) ;
  if (index < 0)
    index = search(key, nodeGlobalID_ + numLocalNodes_, 
		   numExternalNodes_) + numLocalNodes_;
  return index;

  /*
   int  nfirst, nlast, nmid, found, index, nnodes;

   nnodes = numLocalNodes_ + numExternalNodes_;
   if ( nnodes <= 0 ) return -1;
   nfirst = 0;
   nlast  = nnodes - 1;
   if ( key > nodeGlobalID_[nlast] )  return -(nlast+1);
   if ( key < nodeGlobalID_[nfirst] ) return -(nfirst+1);
   found = 0;
   while ((found == 0) && ((nlast-nfirst)>1)) 
   {
      nmid = (nfirst + nlast) / 2;
      if (key == nodeGlobalID_[nmid])     {index  = nmid; found = 1;}
      else if (key > nodeGlobalID_[nmid])  nfirst = nmid;
      else                                  nlast  = nmid;
   }
   if (found == 1)                         return index;
   else if (key == nodeGlobalID_[nfirst]) return nfirst;
   else if (key == nodeGlobalID_[nlast])  return nlast;
   else                                    return -(nfirst+1);
  */
}

//-------------------------------------------------------------------------

int MLI_FEData::searchEdge(int key)
{
   int  nfirst, nlast, nmid, found, index, nEdges;

   nEdges = numLocalEdges_;
   if ( nEdges <= 0 ) return -1;
   nfirst = 0;
   nlast  = nEdges - 1;
   if ( key > edgeGlobalID_[nlast] )  return -(nlast+1);
   if ( key < edgeGlobalID_[nfirst] ) return -(nfirst+1);
   found = 0;
   while ((found == 0) && ((nlast-nfirst)>1)) 
   {
      nmid = (nfirst + nlast) / 2;
      if (key == edgeGlobalID_[nmid])     {index  = nmid; found = 1;}
      else if (key > edgeGlobalID_[nmid])  nfirst = nmid;
      else                                  nlast  = nmid;
   }
   if (found == 1)                         return index;
   else if (key == edgeGlobalID_[nfirst]) return nfirst;
   else if (key == edgeGlobalID_[nlast])  return nlast;
   else                                    return -(nfirst+1);
}

//-------------------------------------------------------------------------

int MLI_FEData::searchFace(int key)
{
  int index = search(key, faceGlobalID_, numLocalFaces_) ;
  if (index < 0)
    index = search(key, faceGlobalID_ + numLocalFaces_, 
		   numExternalFaces_) + numLocalFaces_;
  return index;
  /*
   int  nfirst, nlast, nmid, found, index, nFaces;

   nFaces = numLocalFaces_;
   if ( nFaces <= 0 ) return -1;
   nfirst = 0;
   nlast  = nFaces - 1;
   if ( key > faceGlobalID_[nlast] )  return -(nlast+1);
   if ( key < faceGlobalID_[nfirst] ) return -(nfirst+1);
   found = 0;
   while ((found == 0) && ((nlast-nfirst)>1)) 
   {
      nmid = (nfirst + nlast) / 2;
      if (key == faceGlobalID_[nmid])     {index  = nmid; found = 1;}
      else if (key > faceGlobalID_[nmid])  nfirst = nmid;
      else                                  nlast  = nmid;
   }
   if (found == 1)                         return index;
   else if (key == faceGlobalID_[nfirst]) return nfirst;
   else if (key == faceGlobalID_[nlast])  return nlast;
   else                                    return -(nfirst+1);
  */
}

//-------------------------------------------------------------------------

int MLI_FEData::intSort2(int *ilist, int *ilist2, int left, int right)
{
   int i, last, mid, itemp;

   if (left >= right) return 0;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   if ( ilist2 != NULL )
   {
      itemp        = ilist2[left];
      ilist2[left] = ilist2[mid];
      ilist2[mid]  = itemp;
   }
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
         if ( ilist2 != NULL )
         {
            itemp        = ilist2[last];
            ilist2[last] = ilist2[i];
            ilist2[i]    = itemp;
         } 
      } 
   } 
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   if ( ilist2 != NULL )
   {
      itemp        = ilist2[left];
      ilist2[left] = ilist2[last];
      ilist2[last] = itemp;
   }
   intSort2(ilist, ilist2, left, last-1);
   intSort2(ilist, ilist2, last+1, right);
   return 0;
}

//*************************************************************************
// write grid information to files
//-------------------------------------------------------------------------

int MLI_FEData::writeToFile()
{
   int  i, j, k, length;
   FILE *fp;

   if ( processElemFlag_ == 0 || processNodeFlag_ == 0 )
   {
     printf("MLI_FEData ERROR : writeToFile - not initialized completely.\n");
     return 0;
   }

   fp = fopen("element_chord", "w");

   for ( i = 0; i < numLocalElems_; i++ )
   {
      length = 0;
      for ( j = 0; j < elemNodeLeng_[i]; j++ ) 
         length += nodeDOF_[elemNodeList_[i][j]];
      for ( j = 0; j < length; j++ )
      {
         for ( k = 0; k < length; k++ )
            fprintf(fp, "%13.6e ", elemStiff_[i][j][k]);
         fprintf(fp, "\n");
      }
   }
   fprintf(fp, "\n");
   fclose(fp);

   fp = fopen("element_node", "w");
   
   fprintf(fp, "%d %d\n", numLocalElems_, numLocalNodes_);
   for (i = 0; i < numLocalElems_; i++) 
   {
      for (j = 0; j < elemNodeLeng_[i]; j++) 
         fprintf(fp, "%d ", elemNodeList_[i][j]+1);
      fprintf(fp,"\n");
   } 

   fclose(fp);

   fp = fopen("node_on_bdry", "w");

   for (i = 0; i < numNodeBCs_; i++) 
   {
      fprintf(fp, "%d\n", nodeBCList_[i]);
   }
   fclose(fp);

   return 1;
}

//============================================================================

int MLI_FEData::Print(ostream &out)
{
   int i, j;

   // output the nodes

   out << "node_coord" << endl
       << "local "     << numLocalNodes_ << " offset " << node_off << endl;
  
   for ( i = 0; i < numLocalNodes_; i++ )
   {
      out << "globalID "  << nodeGlobalID_[i] 
 	  << "  localID " << i+node_off << "   ";
      for (j = 0; j < spaceDimension_; j++ ) 
         out << nodeCoordinates_[i*spaceDimension_+j] << "  "; 
      out << endl;
   }

   out << "external " << numExternalNodes_ << endl;
  
   for ( i = 0; i < numExternalNodes_; i++ )
   {
      out << "globalID " << nodeGlobalID_[i+numLocalNodes_] 
	  << "localID "  << externalNodes_[i] << "  ";
      for ( j = 0; j < spaceDimension_; j++ ) 
         out << nodeCoordinates_[(i+numLocalNodes_)*spaceDimension_+j] << "  "; 
      out << endl;
   }

   // output the shared nodes

   out << "shared_nodes " << endl 
       << "local " << numSharedNodes_ << endl;
   for ( i = 0; i < numSharedNodes_; i++ )
   {
      out << "globalID " << sharedNodeID_[i] << "  in_processors  ";
      for ( j = 0; j < sharedNodeLeng_[i]; j++ )
         out << sharedNodeProc_[i][j] << " ";
      out << endl;
   }
   out << endl;

   // output the elements

   out << "element_globalIDnode" << endl
       << "local " << numLocalElems_ << endl;

   for ( i = 0; i < numLocalElems_; i++ )
   {
      for(j=0; j<elemNodeLeng_[i]; j++)
         out << elemNodeList_[i][j] << "  ";
      out << endl;
   }

   // output element_face

   out << "element_globalIDface" << endl
       << "local " << numLocalElems_ << endl;

   for ( i = 0; i < numLocalElems_; i++ )
   {
      for ( j = 0; j < elemFaceLeng_[i]; j++ )
         out << elemFaceList_[i][j] << "  ";
      out << endl;
   }
   out << endl;

   // output face_node

   out << "face_globalIDnode"  << endl
       << "local "     << numLocalFaces_ << " offset " << face_off << endl;
  
   for ( i = 0; i < numLocalFaces_; i++ )
   {
      out << "globalID "  << faceGlobalID_[i] 
	  << "  localID " << i+face_off << "   ";
      for ( j = 0; j < faceNodeLeng_[i]; j++ ) 
         out << faceNodeList_[i][j] << "  "; 
      out << endl;
   }
  
   out << "globalIDexternal " << numExternalFaces_ << endl;
  
   for(i=0; i<numExternalFaces_; i++){
      out << "globalID " << faceGlobalID_[i+numLocalFaces_] 
  	  << " localID "  << externalFaces_[i] << "  ";

      for(j=0; j<faceNodeLeng_[i+numLocalFaces_]; j++ ) 
         out << faceNodeList_[i+numLocalFaces_][j] << "  "; 
      out << endl;
   }  

   // output the shared faces

   out << "shared_faces " << endl 
       << "local " << numSharedFaces_ << endl;
   for ( i = 0; i < numSharedFaces_; i++ )
   {
      out << "globalID " << sharedFaceID_[i] << "  in_processors  ";
      for ( j = 0; j < sharedFaceLeng_[i]; j++ )
         out << sharedFaceProc_[i][j] << " ";
      out << endl;
   }
   return 1;
}

