/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __FEGRIDINFO__
#define __FEGRIDINFO__

/****************************************************************************/ 
/* data structures for Finite element grid information                      */
/*--------------------------------------------------------------------------*/

class FEGridInfo 
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
   int    *elemParentID_;
   int    numLocalEdges_;
   int    *edgeGlobalID_;
   int    **edgeNodeList_;
   int    numLocalNodes_;
   int    numExternalNodes_;
   int    *nodeGlobalID_;
   int    *nodeDOF_;
   int    nodeBCLengMax_;
   int    numNodeBCs_;
   int    *nodeBCList_;
   int    *nodeBCDofList_;
   double *nodeBCValues_;
   double *nodeCoordinates_;
   int    numSharedNodes_;
   int    *sharedNodeLeng_;
   int    **sharedNodeProc_;
   int    processNodeFlag_;
   int    processElemFlag_;
   int    processEdgeFlag_;
   int    outputLevel_;

   int    (*USR_computeShapeFuncInterpolant)(void*, int elem, int nn, 
                                    double *coord, double *coef);

public :

   FEGridInfo(int my_id);

   ~FEGridInfo();

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
   // get general information
   // -----------------------------------------------------------

   int getSpaceDimension(int& numDim);

   int getOrderOfPDE(int& order);

   // -----------------------------------------------------------
   // get element information
   // -----------------------------------------------------------

   int getNumElements(int& nelems);

   int getElemIDs(int *gid);

   int getElemNumNodes(int elemID, int& nnodes);

   int getElemNodeList(int elemID, int *nodeList);

   int getElemStiffMat(int elemID, double **ematrix);

   int getElemNullSpaceSize(int elemID, int &size);

   int getElemNullSpace(int elemID, double **nullSpace);

   int getElemVolume(int elemID, double& volume);

   int getElemMaterial(int elemID, int& material);

   int getElemNumEdges(int elemID, int& nEdges);

   int getElemEdgeList(int elemID, int *edgeList);

   int getElemParentID(int elemID, int& parentID);

   // -----------------------------------------------------------
   // get node information
   // -----------------------------------------------------------

   int getNumLocalNodes(int& nnodes);

   int getNumExternalNodes(int& nnodes);

   int getNodeLocalID(int nodeID, int& localID);

   int getNodeDOF(int nodeID, int& dof);

   int getNodeCoordinate(int nodeID, double *coord);

   int getNodeEssBCs(int& numBCs, int **nodeList, int **dofList, 
                     double **val);

   int getNumSharedNodes(int& nnodes);
    
   int getSharedNodeInfo(int nnodes, int *nodeList, int *procLeng);

   int getSharedNodeProc(int nodeID, int *procList);
    
   // -----------------------------------------------------------
   // get edge information
   // -----------------------------------------------------------

   int getEdgeNodeList(int edgeID, int *nodeList);

   // -----------------------------------------------------------
   // shape function information
   // -----------------------------------------------------------

   int getShapeFuncInterpolant(int element, int nn, double *coord,
                               double *coef);

   int loadFunc_computeShapeFuncInterpolant(int (*func)
            (void *,int elem,int nnodes,double *coord,double *coef));

   int writeToFile();

private :

   int cleanElemInfo();
   int cleanNodeInfo();
   int cleanEdgeInfo();
   int processNodeInfo();
   int searchNode(int);
   int searchElement(int);
   int searchEdge(int);
   int intSort2(int nlist, int *list, int *list2);
};

#endif

