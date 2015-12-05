/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/**************************************************************************
 **************************************************************************
 * FEGridInfo Class functions
 **************************************************************************
 **************************************************************************/

#include <stdio.h>
#include "fegridinfo.h"

/**************************************************************************
 * constructor 
 *-----------------------------------------------------------------------*/

FEGridInfo::FEGridInfo(int my_id)
{
   mypid_            = my_id;
   spaceDimension_   = -1;
   orderOfPDE_       = -1;
   numLocalElems_    = 0;
   elemNodeLeng_     = NULL;
   elemNodeList_     = NULL;
   elemGlobalID_     = NULL;
   elemStiff_        = NULL;
   elemNullSpace_    = NULL;
   elemNullLeng_     = NULL;
   elemVolume_       = NULL;
   elemMaterial_     = NULL;
   elemEdgeLeng_     = NULL;
   elemEdgeList_     = NULL;
   elemParentID_     = NULL;
   numLocalEdges_    = 0;
   edgeGlobalID_     = NULL;
   edgeNodeList_     = NULL;
   numLocalNodes_    = 0;
   numExternalNodes_ = 0;
   nodeGlobalID_     = NULL;
   nodeDOF_          = NULL;
   nodeBCLengMax_    = 0;
   numNodeBCs_       = 0;
   nodeBCList_       = NULL;
   nodeBCDofList_    = NULL;
   nodeBCValues_     = NULL;
   nodeCoordinates_  = NULL;
   numSharedNodes_   = 0;
   sharedNodeLeng_   = NULL;
   sharedNodeProc_   = NULL;
   processElemFlag_  = 0;
   processNodeFlag_  = 0;
   processEdgeFlag_  = 0;
   outputLevel_      = 0;
   USR_computeShapeFuncInterpolant = NULL;
}

//*************************************************************************
// destructor 
//-------------------------------------------------------------------------

FEGridInfo::~FEGridInfo()
{
   cleanElemInfo();
   cleanNodeInfo();
   cleanEdgeInfo();
}

//*************************************************************************
// set diagnostics output level
//-------------------------------------------------------------------------

int FEGridInfo::setOutputLevel(int level)
{
   if ( level < 0 )
   {
      printf("FEGridInfo ERROR : setOutputLevel - level not valid.\n");
      return 0;
   }
   outputLevel_ = level;
   return 0;
}

//*************************************************************************
// dimension of the physical problem (2D, 3D, etc.) 
//-------------------------------------------------------------------------

int FEGridInfo::setSpaceDimension(int dimension)
{
   if ( dimension <= 0 || dimension > 4)
   {
      printf("FEGridInfo ERROR : setSpaceDimension - dim not valid.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("FEGridInfo :setSpaceDimension = %d\n", dimension);
   spaceDimension_ = dimension;
   return 1;
}

//*************************************************************************
// order of the partial differential equation 
//-------------------------------------------------------------------------

int FEGridInfo::setOrderOfPDE(int pdeOrder)
{
   if ( pdeOrder <= 0 || pdeOrder > 4)
   {
      printf("FEGridInfo ERROR : setOrderOfPDE - order not valid.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("FEGridInfo :setOrderOfPDE = %d\n", pdeOrder);
   orderOfPDE_ = pdeOrder;
   return 1;
}

//*************************************************************************
// set number of elements 
//-------------------------------------------------------------------------

int FEGridInfo::beginInitElemSet(int nElems, int *gid)
{
   int i;

   if ( nElems <= 0 )
   {
      printf("FEGridInfo ERROR : beginInitElemSet - nElems <= 0.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("FEGridInfo :beginInitElemSet = %d\n", nElems);

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

   intSort2(numLocalElems_, elemGlobalID_, NULL);
   for ( i = 1; i < numLocalElems_; i++ ) 
   { 
      if ( elemGlobalID_[i] == elemGlobalID_[i-1] )
         printf("FEGridInfo ERROR : beginInitElemSet - duplicate elemIDs.\n");
   }
   return 1;
}

//*************************************************************************
// to be called after all element information has been loaded
//-------------------------------------------------------------------------

int FEGridInfo::endInitElemSet()
{
   int i;

   if ( numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : endInitElemSet - nElems <= 0.\n");
      return 0;
   }
   for ( i = 0; i < numLocalElems_; i++ )
   {
      if ( elemNodeLeng_[i] <= 0 || elemNodeList_[i] == NULL ||
           elemStiff_[i] == NULL )
      {
         printf("FEGridInfo ERROR : endInitElemSet - elem not initialized.\n");
         return 0;
      }
   }
   processElemFlag_ = 1;

   return 1;
}

//*************************************************************************
// load element node list and stiffness matrix 
//-------------------------------------------------------------------------

int FEGridInfo::loadElemSet(int elemID,int nNodesPerElem,int *nodeList,
                            int sdim, double **sMat)
{
   int  i, j, index;

   if ( outputLevel_ >= 1 )
      printf("FEGridInfo :loadElemSet, element ID = %d\n", elemID);

   if ( numLocalElems_ <= 0 ) 
   {
      printf("FEGridInfo ERROR : loadElemSet - numElems <= 0.\n");
      return 0;
   }
   else if ( elemNodeList_ == NULL )
   {
      printf("FEGridInfo ERROR : loadElemSet - not initialized (1).\n");
      return 0;
   }
   if ( elemNodeLeng_ == NULL )
   {
      printf("FEGridInfo ERROR : loadElemSet - not initialized (2).\n");
      return 0;
   }
   if ( elemStiff_ == NULL )
   {
      printf("FEGridInfo ERROR : loadElemSet - not initialized (3).\n");
      return 0;
   }
   if ( nNodesPerElem <= 0 )
   {
      printf("FEGridInfo ERROR : loadElemSet - NodesPerElem <= 0.\n");
      return 0;
   }

   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : loadElemSet - element not local.\n");
      return 0;
   }
   if ( elemStiff_[index] != NULL )
   {
      printf("FEGridInfo ERROR : loadElemSet - element loaded before.\n");
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

int FEGridInfo::loadElemNullSpace(int elemID,int nSize,double **nSpace)
{
   int  i, j, length, index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("FEGridInfo ERROR : loadElemNullSpace - numElems <= 0.\n");
      return 0;
   }
   else if ( elemNullSpace_ == NULL || elemNullLeng_ == NULL )
   {
      printf("FEGridInfo ERROR : loadElemNullSpace - not initialized (1).\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : loadElemNullSpace - element not local.\n");
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

int FEGridInfo::loadElemVolume(int elemID, double volume)
{
   int index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("FEGridInfo ERROR : loadElemVolume - numElems <= 0.\n");
      return 0;
   }
   else if ( elemVolume_ == NULL )
   {
      printf("FEGridInfo ERROR : loadElemVolume - not initialized.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : loadElemVolume - element not local.\n");
      return 0;
   }
   elemVolume_[index] = volume;
   return 1;
}

//*************************************************************************
// load element material 
//-------------------------------------------------------------------------

int FEGridInfo::loadElemMaterial(int elemID, int material)
{
   int index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("FEGridInfo ERROR : loadElemMaterial - numElems <= 0.\n");
      return 0;
   }
   else if ( elemMaterial_ == NULL )
   {
      printf("FEGridInfo ERROR : loadElemMaterial - not initialized.\n");
      return 0;
   }
   index   = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : loadElemMaterial - element not local.\n");
      return 0;
   }
   elemMaterial_[index] = material;
   return 1;
}

//*************************************************************************
// load element edge list 
//-------------------------------------------------------------------------

int FEGridInfo::loadElemEdgeList(int elemID, int nEdges, int *edgeList)
{
   int i, index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("FEGridInfo ERROR : loadElemEdgeList - numElems <= 0.\n");
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
// load element's parent element in the coarse grid
//-------------------------------------------------------------------------

int FEGridInfo::loadElemParentID(int elemID, int parentID)
{
   int i, index;

   if ( numLocalElems_ <= 0 ) 
   {
      printf("FEGridInfo ERROR : loadElemParentID - numElems <= 0.\n");
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

int FEGridInfo::beginInitNodeSet()
{
   int    i, j, index, totalNodes, count, count2, *node_array;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : beginInitNodeSet - elemSet not done.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("FEGridInfo :beginInitNodeSet\n");

   cleanNodeInfo();

   totalNodes = 0;
   for ( i = 0; i < numLocalElems_; i++ ) totalNodes += elemNodeLeng_[i];
   node_array = new int[totalNodes];
   count = 0;
   for ( i = 0; i < numLocalElems_; i++ ) 
   {
      for ( j = 0; j < elemNodeLeng_[i]; j++ ) 
         node_array[count++] = elemNodeList_[i][j];
   }
   intSort2(count, node_array, NULL);
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
   sharedNodeLeng_ = new int[numLocalNodes_]; 
   for ( i = 0; i < numLocalNodes_; i++ ) sharedNodeLeng_[i] = 0;
   sharedNodeProc_ = new int*[numLocalNodes_]; 
   for ( i = 0; i < numLocalNodes_; i++ ) sharedNodeProc_[i] = NULL;
   return 1;
}

//*************************************************************************
// terminate initializing nodal information 
//-------------------------------------------------------------------------

int FEGridInfo::endInitNodeSet()
{
   int    i, j, nnodesLocal, nnodesExt, *extNodeFlag, *newNodeList;
   int    count, *iarray;
   double *darray;

   nnodesExt = 0;
   extNodeFlag = new int[numLocalNodes_];
   for ( i = 0; i < numLocalNodes_; i++ )
   {
      extNodeFlag[i] = 0;
      if ( sharedNodeProc_[i] != NULL )
      {
         for ( j = 0; j < sharedNodeLeng_[i]; j++ )
            if ( sharedNodeProc_[i][j] < mypid_ ) 
            {
               nnodesExt++; 
               extNodeFlag[i] = - 1;
               break;
            }
      }
   }
   numExternalNodes_ = nnodesExt;
   nnodesLocal = numLocalNodes_;
   numLocalNodes_ -= numExternalNodes_;
   newNodeList = new int[nnodesLocal];
   count = 0;
   for ( i = 0; i < numLocalNodes_; i++ )
   {
      if ( extNodeFlag[i] == 0 ) 
      {
         newNodeList[count] = nodeGlobalID_[i];
         extNodeFlag[i] = count++;
      }
   }
   for ( i = 0; i < numLocalNodes_; i++ )
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
         nodeDOF_[i] = iarray[extNodeFlag[i]];
      delete [] iarray;
   }
   if ( nodeCoordinates_ != NULL )
   {
      darray = nodeCoordinates_;
      nodeCoordinates_ = new double[nnodesLocal*spaceDimension_];
      for ( i = 0; i < nnodesLocal; i++ ) 
         for ( j = 0; j < spaceDimension_; j++ ) 
            nodeCoordinates_[i*spaceDimension_+j] = 
               darray[extNodeFlag[i]*spaceDimension_+j];
      delete [] darray;
   }
   delete [] extNodeFlag;
   return 1;
}   

//*************************************************************************
// set node degree of freedom 
//-------------------------------------------------------------------------

int FEGridInfo::loadNodeDOF(int nodeID, int dof)
{
   int i, index;

   if ( numLocalNodes_ == 0 )
   {
      printf("FEGridInfo ERROR : loadNodeDOF - beginInitNodeSet ?\n");
      return 0;
   }
   else if ( nodeDOF_ == NULL )
   {
      printf("FEGridInfo ERROR : loadNodeDOF - nodeDOF not initialized.\n");
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
         printf("FEGridInfo ERROR : loadNodeDOF - node not local.\n");
         return 0;
      }
      nodeDOF_[index] = dof;
   }
   return 1;
}

//*************************************************************************
// load node coordinates 
//-------------------------------------------------------------------------

int FEGridInfo::loadNodeCoordinate(int nodeID, double *coord)
{
   int i, index;

   if ( numLocalNodes_ == 0 )
   {
      printf("FEGridInfo ERROR : loadNodeCoordinate - beginInitNodeSet ?\n");
      return 0;
   }
   else if ( nodeCoordinates_ == NULL )
   {
      printf("FEGridInfo ERROR : loadNodeCoordinate - not initialized.\n");
      return 0;
   }

   index = searchNode( nodeID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : loadNodeCoordinate - node not local.\n");
      return 0;
   }
   for ( i = 0; i < spaceDimension_; i++ )
      nodeCoordinates_[index*spaceDimension_+i] = coord[i];
   return 1;
}

//*************************************************************************
// set node boundary condition 
//-------------------------------------------------------------------------

int FEGridInfo::loadNodeEssBCs(int nnodes, int *nodeIDs, int *dofList, 
                               double *val)
{
   int    i, node, index, *iarray, *iarray2;
   double *darray;

   if ( numLocalNodes_ == 0 )
   {
      printf("FEGridInfo ERROR : loadNodeEssBCs - beginInitNodeSet ?\n");
      return 0;
   }
   //else if ( nodeBCList_ == NULL || nodeBCValues_ == NULL )
   //{
   //   printf("FEGridInfo ERROR : loadNodeEssBCs - nodeBC not initialized.\n");
   //   return 0;
   //}
   if ( nnodes <= 0 )
   {
      printf("FEGridInfo ERROR : loadNodeEssBCs - invalid input.\n");
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
         printf("FEGridInfo ERROR : loadNodeBC - node not local.\n");
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

int FEGridInfo::loadSharedNodes(int nnodes, int *nodeList, int *procLeng, 
                                int **nodeProc)
{
   int i, j, node, index;

   if ( numLocalNodes_ == 0 )
   {
      printf("FEGridInfo ERROR : loadNodeEssBCs - beginInitNodeSet ?\n");
      return 0;
   }
   else if ( sharedNodeLeng_ == NULL )
   {
      printf("FEGridInfo ERROR : loadSharedNodes - not initialized.\n");
      return 0;
   }
   for ( i = 0; i < nnodes; i++ )
   {
      node = nodeList[i];
      index = searchNode( node );
      if ( index < 0 )
      {
         printf("FEGridInfo ERROR : loadSharedNodes - node not local.\n");
         return 0;
      }
      if ( sharedNodeProc_[index] != NULL )
      {
         printf("FEGridInfo ERROR : loadSharedNodes - already initialized.\n");
         return 0;
      }
      if ( procLeng[i] <= 0 )
      {
         printf("FEGridInfo ERROR : loadSharedNodes - procLeng not valid.\n");
         return 0;
      }

      sharedNodeLeng_[index] = procLeng[i];
      sharedNodeProc_[index] = new int[procLeng[i]];
      for ( j = 0; j < procLeng[i]; j++ )
         sharedNodeProc_[index][j] =  nodeProc[i][j];
      numSharedNodes_++;
   } 
   return 1;
}

//*************************************************************************
// begin initializing edge set
//-------------------------------------------------------------------------

int FEGridInfo::beginInitEdgeSet()
{
   int    i, j, index, totalEdges, count, count2, *edge_array;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : beginInitEdgeSet - elemSet not done.\n");
      return 0;
   }
   if ( outputLevel_ >= 1 )
      printf("FEGridInfo :beginInitEdgeSet\n");

   totalEdges = 0;
   for ( i = 0; i < numLocalElems_; i++ ) totalEdges += elemEdgeLeng_[i];
   edge_array = new int[totalEdges];
   count = 0;
   for ( i = 0; i < numLocalElems_; i++ ) 
   {
      for ( j = 0; j < elemEdgeLeng_[i]; j++ ) 
         edge_array[count++] = elemEdgeList_[i][j];
   }
   intSort2(count, edge_array, NULL);
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
// terminate initializing edge information 
//-------------------------------------------------------------------------

int FEGridInfo::endInitEdgeSet()
{
   int i;

   for ( i = 0; i < numLocalEdges_; i++ )
   {
      if ( edgeNodeList_[i] == NULL )
      {
         printf("FEGridInfo ERROR : endInitEdgeSet - not complete.\n");
         return 0;
      }
   }
   processEdgeFlag_ = 1;
   return 1;
}

//*************************************************************************
// load edge node list
//-------------------------------------------------------------------------

int FEGridInfo::loadEdgeNodeList(int edgeID, int *nodeList)
{
   int    i, index, totalEdges, count, count2, *edge_array;

   index = searchEdge( edgeID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : loadEdgeNodeList - edge not local.\n");
      return 0;
   }
   if ( edgeNodeList_ == NULL )
   {
      printf("FEGridInfo ERROR : loadEdgeNodeList - list not initialized.\n");
      return 0;
   }
   edgeNodeList_[index] = new int[2];
   edgeNodeList_[index][0] = nodeList[0];
   edgeNodeList_[index][1] = nodeList[1];
   return 1;
} 

//*************************************************************************
// get dimension of physical problem
//-------------------------------------------------------------------------

int FEGridInfo::getSpaceDimension(int& numDim)
{
   numDim = spaceDimension_;
   return 1;
}

//*************************************************************************
// get order of PDE 
//-------------------------------------------------------------------------

int FEGridInfo::getOrderOfPDE(int& order)
{
   order = orderOfPDE_;
   return 1;
}

//*************************************************************************
// get number of local elements 
//-------------------------------------------------------------------------

int FEGridInfo::getNumElements(int& nelems)
{
   nelems = numLocalElems_;
   return 1;
}

//*************************************************************************
// get all element globalIDs 
//-------------------------------------------------------------------------

int FEGridInfo::getElemIDs(int *gid)
{
   int i;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemIDs - elemSet not done.\n");
      return 0;
   }
   if ( elemGlobalID_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemIDs - info not available.\n");
      return 0;
   }
   for ( i = 0; i < numLocalElems_; i++ ) gid[i] = elemGlobalID_[i];
   return 1;
}

//*************************************************************************
// get element nodelist size 
//-------------------------------------------------------------------------

int FEGridInfo::getElemNumNodes(int elemID, int& nnodes)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemNumNodes - elemSet not done.\n");
      return 0;
   }
   if ( elemNodeLeng_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemNumNodes - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemNumNodes - element not local.\n");
      return 0;
   }
   nnodes = elemNodeLeng_[index];
   return 1;
}

//*************************************************************************
// get element nodelist 
//-------------------------------------------------------------------------

int FEGridInfo::getElemNodeList(int elemID, int *nodeList)
{
   int i, index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemNodeList - elemSet not done.\n");
      return 0;
   }
   else if ( elemNodeList_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemNodeList - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemNodeList - element not local.\n");
      return 0;
   }
   for ( i = 0; i < elemNodeLeng_[index]; i++ )
      nodeList[i] = elemNodeList_[index][i];
   return 1;
}

//*************************************************************************
// get element stiffness matrix 
//-------------------------------------------------------------------------

int FEGridInfo::getElemStiffMat(int elemID, double **ematrix)
{
   int i, j, index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemStiffMat - elemSet not done.\n");
      return 0;
   }
   else if ( elemStiff_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemStiffmat - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemStiffMat - element not local.\n");
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

int FEGridInfo::getElemNullSpaceSize(int elemID, int& size)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemNullSpaceSize - elemSet not done.\n");
      return 0;
   }
   else if ( elemNullLeng_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemNullSpaceSize - not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemNullSpaceSize - element not local.\n");
      return 0;
   }
   size = elemNullLeng_[index];
   return 1;
}

//*************************************************************************
// get element null space 
//-------------------------------------------------------------------------

int FEGridInfo::getElemNullSpace(int elemID, double **nullSpace)
{
   int i, j, index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemNullSpace - elemSet not done.\n");
      return 0;
   }
   else if ( elemNullSpace_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemNullSpace - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemNullSpace - element not local.\n");
      return 0;
   }
   if ( elemNullSpace_[index] == NULL )
   {
      printf("FEGridInfo ERROR : getElemNullSpace - info not available.\n");
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

int FEGridInfo::getElemVolume(int elemID, double& volume)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemVolume - elemSet not done.\n");
      return 0;
   }
   else if ( elemVolume_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemVolume - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemVolume - element not local.\n");
      return 0;
   }
   volume = elemVolume_[index];
   return 1;
}

//*************************************************************************
// get element material 
//-------------------------------------------------------------------------

int FEGridInfo::getElemMaterial(int elemID, int& material)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemMaterial - elemSet not done.\n");
      return 0;
   }
   else if ( elemMaterial_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemMaterial - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemMaterial - element not local.\n");
      return 0;
   }
   material = elemMaterial_[index];
   return 1;
}

//*************************************************************************
// get element number of edges 
//-------------------------------------------------------------------------

int FEGridInfo::getElemNumEdges(int elemID, int& numEdges)
{
   int index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemNumEdges - elemSet not done.\n");
      return 0;
   }
   else if ( elemEdgeLeng_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemNumEdges - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemNumEdges - element not local.\n");
      return 0;
   }
   numEdges = elemEdgeLeng_[index];
   return 1;
}

//*************************************************************************
// get element edge list 
//-------------------------------------------------------------------------

int FEGridInfo::getElemEdgeList(int elemID, int *edgeList)
{
   int i, index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemEdgeList - elemSet not done.\n");
      return 0;
   }
   else if ( elemEdgeList_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemEdgeList - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemEdgeList - element not local.\n");
      return 0;
   }
   for ( i = 0; i < elemEdgeLeng_[index]; i++ )
      edgeList[i] = elemEdgeList_[index][i];
   return 1;
}

//*************************************************************************
// get element's parent ID 
//-------------------------------------------------------------------------

int FEGridInfo::getElemParentID(int elemID, int& parentID)
{
   int i, index;

   if ( processElemFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getElemParentID - elemSet not done.\n");
      return 0;
   }
   else if ( elemParentID_ == NULL || numLocalElems_ <= 0 )
   {
      printf("FEGridInfo ERROR : getElemParentID - info not available.\n");
      return 0;
   }
   index = searchElement( elemID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getElemParentID - element not local.\n");
      return 0;
   }
   parentID = elemParentID_[index];
   return 1;
}

//*************************************************************************
// get number of local nodes 
//-------------------------------------------------------------------------

int FEGridInfo::getNumLocalNodes(int& node)
{
   int status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getNumLocalNodes - not initialized.\n");
      return 0;
   }
   node = numLocalNodes_;
   return 1;
}

//*************************************************************************
// get number of external nodes 
//-------------------------------------------------------------------------

int FEGridInfo::getNumExternalNodes(int& node)
{
   int  status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getNumExternalNodes - not initialized.\n");
      return 0;
   }
   node = numExternalNodes_;
   return 1;
}

//*************************************************************************
// get node local ID 
//-------------------------------------------------------------------------

int FEGridInfo::getNodeLocalID(int nodeID, int &localID)
{
   int index, status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getNodeLocalID - not initialized.\n");
      return 0;
   }
   index = searchNode( nodeID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getNodeLocalID - node not local.\n");
      return 0;
   }
   localID = index;
   return 1;
}

//*************************************************************************
// get node degree of freedom 
//-------------------------------------------------------------------------

int FEGridInfo::getNodeDOF(int nodeID, int& dof)
{
   int i, index, status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getNodeDOF - not initialized.\n");
      return 0;
   }
   index = searchNode( nodeID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getNodeDOF - node not local.\n");
      return 0;
   }
   dof = nodeDOF_[index];
   return 1;
}

//*************************************************************************
// get node coordinates 
//-------------------------------------------------------------------------

int FEGridInfo::getNodeCoordinate(int nodeID, double *coord)
{
   int  i, index, status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getNodeCoordinate - not initialized.\n");
      return 0;
   }
   index = searchNode( nodeID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getNodeCoordinate - node not local.\n");
      return 0;
   }
   for ( i = 0; i < spaceDimension_; i++ )
      coord[i] = nodeCoordinates_[spaceDimension_*index+i];
   return 1;
}

//*************************************************************************
// get node degree of freedom 
//-------------------------------------------------------------------------

int FEGridInfo::getNodeEssBCs(int& numBCs, int **nodeList, int **dofList, 
                              double **val)
{
   int  status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getNodeEssBCs - not initialized.\n");
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

int FEGridInfo::getNumSharedNodes(int& nnodes)
{
   int  status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getNumSharedNodes - not initialized.\n");
      return 0;
   }
   nnodes = numSharedNodes_;
   return 1;
}

//*************************************************************************
// get shared nodes list 
//-------------------------------------------------------------------------

int FEGridInfo::getSharedNodeInfo(int nNodes, int *nodeList, int *procLeng)
{
   int  i, count, status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getSharedNodesInfo - not initialized.\n");
      return 0;
   }
   if ( nNodes != numSharedNodes_ )
   {
      printf("FEGridInfo ERROR : getSharedNodesInfo - wrong no. of nodes.\n");
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

int FEGridInfo::getSharedNodeProc(int nodeID, int *procList)
{
   int  i, index, status;

   if ( processNodeFlag_ == 0 ) status = processNodeInfo();
   if ( status == 0 )
   {
      printf("FEGridInfo ERROR : getSharedNodeProc - not initialized.\n");
      return 0;
   }
   index = searchNode( nodeID );
   if ( index < 0 || sharedNodeProc_[index] == NULL )
   {
      printf("FEGridInfo ERROR : getSharedNodeProc - nodeID invalid.\n");
      return 0;
   }
   for ( i = 0; i < sharedNodeLeng_[index]; i++ )
      procList[i] = sharedNodeProc_[index][i];
   return 1;
}
    
//*************************************************************************
// get edge node list 
//-------------------------------------------------------------------------

int FEGridInfo::getEdgeNodeList(int edgeID, int *nodeList)
{
   int index;

   if ( processEdgeFlag_ != 1 )
   {
      printf("FEGridInfo ERROR : getEdgeNodeList - edgeSet not done.\n");
      return 0;
   }
   else if ( edgeNodeList_ == NULL || numLocalEdges_ <= 0 )
   {
      printf("FEGridInfo ERROR : getEdgeNodeList - info not available.\n");
      return 0;
   }
   index = searchEdge( edgeID );
   if ( index < 0 )
   {
      printf("FEGridInfo ERROR : getEdgeNodeList - edge not local.\n");
      return 0;
   }
   nodeList[0] = edgeNodeList_[index][0];
   nodeList[1] = edgeNodeList_[index][1];
   return 1;
}

//*************************************************************************
// get shape function interpolant 
//-------------------------------------------------------------------------

int FEGridInfo::getShapeFuncInterpolant(int element, int nn, double *coord,
                                        double *coef)
{
   (void) element;
   (void) nn;
   (void) coord;
   (void) coef;
   return 0;
}

//*************************************************************************
// load in the function to calculate shape function interpolant 
//-------------------------------------------------------------------------

int FEGridInfo::loadFunc_computeShapeFuncInterpolant(int (*func)
                (void*,int elem,int nnodes,double *coord,double *coef))
{
   (void) func;
   return 0;
}

//*************************************************************************
// cleanup the element storage 
//-------------------------------------------------------------------------

int FEGridInfo::cleanElemInfo()
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

   if ( elemParentID_ != NULL ) delete [] elemParentID_;
   elemParentID_ = NULL;

   numLocalElems_ = 0;
   processElemFlag_ = 0;

   return 1;
}

//*************************************************************************
// cleanup the node storage 
//-------------------------------------------------------------------------

int FEGridInfo::cleanNodeInfo()
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

   if ( nodeDOF_ != NULL ) delete [] nodeDOF_;
   nodeDOF_ = NULL;
   
   if ( nodeGlobalID_  != NULL ) delete [] nodeGlobalID_;
   nodeGlobalID_ = NULL;
   
   numLocalNodes_ = 0;
   numExternalNodes_ = 0;
   nodeBCLengMax_    = 0;
   processNodeFlag_ = 0;
   return 1;
}

//*************************************************************************
// cleanup the edge storage 
//-------------------------------------------------------------------------

int FEGridInfo::cleanEdgeInfo()
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
// clean up everything 
//-------------------------------------------------------------------------

int FEGridInfo::cleanAll()
{
   cleanElemInfo();
   cleanNodeInfo();
   cleanEdgeInfo();
   spaceDimension_ = -1;
   orderOfPDE_     = -1;
   outputLevel_    = 0;
   USR_computeShapeFuncInterpolant = NULL;
   return 1;
}

//*************************************************************************
// set up node stuff 
//-------------------------------------------------------------------------

int FEGridInfo::processNodeInfo()
{
   int    i, index, *intarray, *iarray, **iiarray, *iarray2, *iarray3;
   double *darray, ***dddarray, ***dddarray2;

   intarray = new int[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) intarray[i] = i;
   intSort2(numLocalElems_, elemGlobalID_, intarray);

   intarray = new int[numLocalElems_];
   for ( i = 0; i < numLocalElems_; i++ ) intarray[i] = i;
   intSort2(numLocalElems_, elemGlobalID_, intarray);
   for ( i = 1; i < numLocalElems_; i++ ) 
   { 
      if ( elemGlobalID_[i] == elemGlobalID_[i-1] )
         printf("FEGridInfo ERROR : processNodeInfo - duplicate elemIDs.\n");
   }
   dddarray  = elemStiff_;
   dddarray2 = elemNullSpace_;
   iiarray   = elemNodeList_;
   iarray    = elemNodeLeng_;
   darray    = elemVolume_;
   iarray2   = elemMaterial_;
   iarray3   = elemNullLeng_;
   for ( i = 0; i < numLocalElems_; i++ ) 
   {
      index = intarray[i];
      elemStiff_[index]     = dddarray[i];
      elemNullSpace_[index] = dddarray2[i];
      elemNodeList_[index]  = iiarray[i];
      elemNodeLeng_[index]  = iarray[i];
      elemVolume_[index]    = darray[i];
      elemMaterial_[index]  = iarray2[i];
      elemNullLeng_[index]  = iarray3[i];
   } 
   delete [] intarray;
   processNodeFlag_ = 1;
   return 1;
}

//*************************************************************************
// search and sort stuff (borrowed from Aztec and ML)
//-------------------------------------------------------------------------

int FEGridInfo::searchElement(int key)
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

int FEGridInfo::searchNode(int key)
{
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
}

//-------------------------------------------------------------------------

int FEGridInfo::searchEdge(int key)
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

int FEGridInfo::intSort2(int N, int list[], int list2[])
{
   int    l, r, RR, K, j, i, flag;
   int    RR2;

   if (N <= 1) return 0;

   l   = N / 2 + 1;
   r   = N - 1;
   l   = l - 1;
   RR  = list[l - 1];
   K   = list[l - 1];

   if (list2 != NULL) {
      RR2 = list2[l - 1];
      while (r != 0) {
         j = l;
         flag = 1;

         while (flag == 1) {
            i = j;
            j = j + j;

            if (j > r + 1)
               flag = 0;
            else {
               if (j < r + 1)
                  if (list[j] > list[j - 1]) j = j + 1;

               if (list[j - 1] > K) {
                  list[ i - 1] = list[ j - 1];
                  list2[i - 1] = list2[j - 1];
               }
               else {
                  flag = 0;
               }
            }
         }
         list[ i - 1] = RR;
         list2[i - 1] = RR2;
         if (l == 1) {
            RR  = list [r];
            RR2 = list2[r];
            K = list[r];
            list[r ] = list[0];
            list2[r] = list2[0];
            r = r - 1;
         }
         else {
            l   = l - 1;
            RR  = list[ l - 1];
            RR2 = list2[l - 1];
            K   = list[l - 1];
         }
      }
      list[ 0] = RR;
      list2[0] = RR2;
   }
   else {
      while (r != 0) {
         j = l;
         flag = 1;
         while (flag == 1) {
            i = j;
            j = j + j;
            if (j > r + 1)
               flag = 0;
            else {
               if (j < r + 1)
                  if (list[j] > list[j - 1]) j = j + 1;
               if (list[j - 1] > K) {
                  list[ i - 1] = list[ j - 1];
               }
               else {
                  flag = 0;
               }
            }
         }
         list[ i - 1] = RR;
         if (l == 1) {
            RR  = list [r];
            K = list[r];
            list[r ] = list[0];
            r = r - 1;
         }
         else {
            l   = l - 1;
            RR  = list[ l - 1];
            K   = list[l - 1];
         }
      }
      list[ 0] = RR;
   }
   return 1;
}

//*************************************************************************
// write grid information to files
//-------------------------------------------------------------------------

int FEGridInfo::writeToFile()
{
   int  i, j, k, length;
   FILE *fp;

   if ( processElemFlag_ == 0 || processNodeFlag_ == 0 )
   {
      printf("FEGridInfo ERROR : writeToFile - not initialized completely.\n");
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

