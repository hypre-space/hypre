/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLIFEDATAH__
#define __MLIFEDATAH__

/****************************************************************************/ 
/* data structures for Finite element grid information                      */
/*--------------------------------------------------------------------------*/

class MLI_FEData 
{
   int    mypid_;
   int    spaceDimension_;
   int    orderOfPDE_;

   int    numLocalElems_;
   int    *elemNodeLeng_;
   int    **elemNodeList_;
   int    *elemGlobalID_;
   double ***elemStiff_;
   double ***elemNullSpace_;
   int    *elemNullLeng_;
   double *elemVolume_;
   int    *elemMaterial_;
   int    *elemEdgeLeng_;
   int    **elemEdgeList_;
   int    *elemFaceLeng_;
   int    **elemFaceList_;
   int    *elemParentID_;

   int    numLocalEdges_;
   int    *edgeGlobalID_;
   int    **edgeNodeList_;

   int    face_off;
   int    numLocalFaces_;
   int    numExternalFaces_;
   int    *faceGlobalID_;
   int    *externalFaces_;
   int    *faceDOF_;
   int    faceBCLengMax_;
   int    numFaceBCs_;
   int    *faceBCList_;
   int    *faceBCDofList_;
   double *faceBCValues_;
   int    numSharedFaces_;
   int    *sharedFaceID_;
   int    *sharedFaceLeng_;
   int    **sharedFaceProc_;
   int    *faceNodeLeng_;
   int    **faceNodeList_;

   int    node_off;
   int    numLocalNodes_;
   int    numExternalNodes_;
   int    *nodeGlobalID_;
   int    *externalNodes_;
   int    *nodeDOF_;
   int    nodeBCLengMax_;
   int    numNodeBCs_;
   int    *nodeBCList_;
   int    *nodeBCDofList_;
   double *nodeBCValues_;
   double *nodeCoordinates_;
   int    numSharedNodes_;
   int    *sharedNodeID_;
   int    *sharedNodeLeng_;
   int    **sharedNodeProc_;

   int    processNodeFlag_;
   int    processElemFlag_;
   int    processEdgeFlag_;
   int    processFaceFlag_;
   int    outputLevel_;

   int    (*USR_computeShapeFuncInterpolant)(void*, int elem, int nn, 
                                    double *coord, double *coef);

public :

   MLI_FEData(int my_id);

   ~MLI_FEData();

   int cleanAll();

   // -----------------------------------------------------------
   // load general information
   // -----------------------------------------------------------

   int setOutputLevel(int level);

   int setSpaceDimension(int numDim);

   int setOrderOfPDE(int pdeOrder);

   // -----------------------------------------------------------
   // load element information
   // -----------------------------------------------------------

   int beginInitElemSet(int nelems, int *gid);

   int endInitElemSet();

   int loadElemSet(int elemID, int nNodesPerElem, int *nodeList,
                   int sdim, double **sMat);

   int loadElemNullSpace(int elemID, int nSize, double **nSpace);

   int loadElemVolume(int elemID, double volume);

   int loadElemMaterial(int elemID, int material);

   int loadElemEdgeList(int elemID, int nEdges, int *edgeList);

   int loadElemFaceList(int elemID, int nFacess, int *faceList);

   int loadElemParentID(int elemID, int parentID);

   // -----------------------------------------------------------
   // load node information
   // -----------------------------------------------------------

   int beginInitNodeSet();

   int endInitNodeSet();

   int loadNodeDOF(int nodeID, int dof);

   int loadNodeCoordinate(int nodeID, double *coord);

   int loadNodeEssBCs(int nnodes, int *nodeList, int *dofList, 
                      double *val);

   int loadSharedNodes(int nnodes, int *nodeList, int *procLeng,
                       int **nodeProc);

   // -----------------------------------------------------------
   // load edge information
   // -----------------------------------------------------------

   int beginInitEdgeSet();

   int endInitEdgeSet();

   int loadEdgeNodeList(int edgeID, int *nodeList);

   // -----------------------------------------------------------
   // load face information
   // -----------------------------------------------------------

   int beginInitFaceSet();

   int endInitFaceSet();

   int loadFaceDOF(int faceID, int dof);

   int loadFaceNodeList(int faceID, int nNodes, int *nodeList);

   int loadSharedFaces ( int nfaces, int *faceList, int *procLeng,
			 int **faceProc);

   // -----------------------------------------------------------
   // get general information
   // -----------------------------------------------------------

   int getSpaceDimension(int& numDim);

   int getOrderOfPDE(int& order);

   // -----------------------------------------------------------
   // get element information
   // Everywhere below elemID denotes global element ID.
   // -----------------------------------------------------------

   // Return the number of the local elements in nelems.
   int getNumElements(int& nelems);

   // Return the element global IDs in gid (of size getNumElements).
   int getElemIDs(int *gid);

   // Return the number of nodes for element with given global ID.
   int getElemNumNodes(int elemID, int& nnodes);

   // Return the nodes (in global index) for element with given global ID.
   int getElemNodeList(int elemID, int *nodeList);

   // Return the element stiffness matrix.
   int getElemStiffMat(int elemID, double **ematrix);

   // Return the null space size for element with given global ID. 
   int getElemNullSpaceSize(int elemID, int &size);

   // Return the null space for element with given global ID. 
   int getElemNullSpace(int elemID, double **nullSpace);

   // Return the volume  
   int getElemVolume(int elemID, double& volume);

   // Return the material.
   int getElemMaterial(int elemID, int& material);

   // Return the number of edges.
   int getElemNumEdges(int elemID, int& nEdges);

   // Return the edge list.
   int getElemEdgeList(int elemID, int *edgeList);

   // Return the number of faces.
   int getElemNumFaces(int elemID, int& nFaces);

   // Return the face list.
   int getElemFaceList(int elemID, int *faceList);

   // Return the parent ID.
   int getElemParentID(int elemID, int& parentID);

   // -----------------------------------------------------------
   // get node information
   // Everywhere below nodeID denotes global node ID.
   // -----------------------------------------------------------

   // Return the number of local nodes ("owned" by "this" processor).
   int getNumLocalNodes(int& nnodes);

   // Return the number of external nodes.
   int getNumExternalNodes(int& nnodes);

   /* Return the local ID. Return error if the node is not local (even
      if the node is among the external ones).                            */
   int getNodeLocalID(int nodeID, int& localID);

   // Return the "GLOBAL" (see the top) node ID.
   int getNodeGlobalID(int nodeID, int &globalID);

   // This assumes there is only one degree of freedom per node.! 
   int getNodeDOF(int nodeID, int& dof);

   // Return the coordinates in coord (should be of dimension spaceDimension_)
   int getNodeCoordinate(int nodeID, double *coord);

   // Return the essential BCs (giving pointers to the data in FEGridInfo).
   int getNodeEssBCs(int& numBCs, int **nodeList, int **dofList, 
                     double **val);

   // Return the number of shared nodes.
   int getNumSharedNodes(int& nnodes);
    
   /* Return shared node list and for every of the nodes with how many
      processors is shared (the data is copied).                          */
   int getSharedNodeInfo(int nnodes, int *nodeList, int *procLeng);

   // Return the processors that share the given node.
   int getSharedNodeProc(int nodeID, int *procList);

   // -----------------------------------------------------------
   // get edge information
   // -----------------------------------------------------------

   int getEdgeNodeList(int edgeID, int *nodeList);

   // -----------------------------------------------------------
   // get face information
   // -----------------------------------------------------------

   int getNumLocalFaces(int& nfaces);

   int getNumExternalFaces(int& nfaces);

   int getFaceLocalID(int faceID, int& localID);

   int getFaceGlobalID(int faceID, int &globalID);

   int getFaceDOF(int faceID, int& dof);

   int getNumSharedFaces(int& nfaces);

   // The data is copied.
   int getSharedFaceInfo(int nfaces, int *faceList, int *procLeng);

   // The data is copied.
   int getSharedFaceProc(int faceID, int *procList);

   // The data is copied.
   int getFaceNodeList(int faceID, int *nodeList);

   // The data is copied.
   int getFaceIDs ( int *gid );

   int getFaceNumNodes(int faceID, int &nnodes);

   // -----------------------------------------------------------
   // shape function information
   // -----------------------------------------------------------

   int getShapeFuncInterpolant(int element, int nn, double *coord,
                               double *coef);

   /* This function is used to get specific data from the FEGridInfo
      object. For now is used to get element, face, edge and node
      offsets. Also it may be used to return the owners of the
      external nodes (see the implementation).                            */
   int getSpecificData(char *data_key, void *data);

   /* Similar to the above but two pointers to data are used. For now is 
      used to update tables of connectivity information - to do the 
      communication between the processors (see the implementation).      */
   int getSpecificData(char *data_key, void *data1, void *data2);

   int loadFunc_computeShapeFuncInterpolant(int (*func)
            (void *,int elem,int nnodes,double *coord,double *coef));

   int writeToFile();

   int Print(ostream &out);

   /* For a given global node ID return the local node ID (the function
      searches also among the external nodes.                             */
   int searchNode(int key);

   /* For a given global face ID return the local face ID (the function
      searches also among the external faces.                             */
   int searchFace(int key);

private :

   int cleanElemInfo();
   int cleanNodeInfo();
   int cleanEdgeInfo();
   int cleanFaceInfo();
   int processElemInfo();
   int search(int, int *, int);
   int searchElement(int);
   int searchEdge(int);   
   int intSort2(int *list, int *list2, int left, int right);
};

#endif

