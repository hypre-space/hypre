/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLIFEDATA_H__
#define __MLIFEDATA_H__

#include "_hypre_utilities.h"
#include "mli_febase.h"

/****************************************************************************/ 
/* data structures for Finite element grid information                      */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/* first, definition of an element block (all elements in an element block  */
/* have the same number of nodes and same number of fields, etc.)           */
/*--------------------------------------------------------------------------*/

typedef struct MLI_ElemBlock_Struct
{
   int    numLocalElems_;        /* number of elements in this block */
   int    *elemGlobalIDs_;       /* element global IDs in this block */
   int    *elemGlobalIDAux_;     /* for conversion to local IDs */
   int    elemNumNodes_;         /* number of nodes per elements */
   int    **elemNodeIDList_;     /* element node list (global IDs) */
   int    elemNumFields_;        /* number of element fields */
   int    *elemFieldIDs_;        /* a list of element field IDs */
   int    elemDOF_;              /* element degree of freedom */
   int    elemStiffDim_;         /* element stiffness matrix dimension */
   double **elemStiffMat_;       /* element stiffness matrices */
   int    *elemNumNS_;           /* element number of nullspace vectors */
   double **elemNullSpace_;      /* element null space vectors */
   double *elemVolume_;          /* element volumes */
   int    *elemMaterial_;        /* element materials */
   int    *elemParentIDs_;       /* element parentIDs */
   double **elemLoads_;
   double **elemSol_;
   int    elemNumFaces_;         /* number of faces in an element */
   int    **elemFaceIDList_;     /* element face global ID lists */
   int    elemNumBCs_;
   int    *elemBCIDList_;
   char   **elemBCFlagList_;
   double **elemBCValues_;
   int    elemOffset_;

   int    numLocalNodes_;        /* number of internal nodes */
   int    numExternalNodes_;     /* number of external nodes */
   int    *nodeGlobalIDs_;       /* a list of node global IDs */
   int    nodeNumFields_;        /* number of node fields */
   int    *nodeFieldIDs_;        /* a list of node field IDs */
   int    nodeDOF_;              /* nodal degree of freedom */
   double *nodeCoordinates_;     /* a list of nodal coordinates */
   int    nodeNumBCs_;           /* number of node BCs */
   int    *nodeBCIDList_;        /* a list of BC node global IDs */
   char   **nodeBCFlagList_;     /* a list of node BC flags */
   double **nodeBCValues_;       /* node BCs */
   int    numSharedNodes_;       /* number of shared nodes */
   int    *sharedNodeIDs_;       /* shared node global IDs */
   int    *sharedNodeNProcs_;    /* number of processors each node is shared*/
   int    **sharedNodeProc_;     /* processor IDs for shared nodes */
   int    *nodeExtNewGlobalIDs_; /* processor IDs for shared nodes */
   int    nodeOffset_;           /* node processor offset */

   int    numLocalFaces_;        /* number of local faces */
   int    numExternalFaces_;     /* number of external faces (local element)*/
   int    *faceGlobalIDs_;       /* a list of face global IDs */
   int    faceNumNodes_;         /* number of nodes in a face */
   int    **faceNodeIDList_;     /* face node list */
   int    numSharedFaces_;       /* number of shared faces */
   int    *sharedFaceIDs_;       /* shared face IDs */
   int    *sharedFaceNProcs_;    /* number of processors each face is shared*/
   int    **sharedFaceProc_;     /* processor IDs of shared faces */
   int    *faceExtNewGlobalIDs_; /* processor IDs for shared nodes */
   int    faceOffset_;           /* face global offsets */

   int    initComplete_;
}
MLI_ElemBlock;

class MLI_FEData : public MLI_FEBase
{
   MPI_Comm      mpiComm_;
   int           outputLevel_;
   int           spaceDimension_;
   int           orderOfPDE_;
   int           orderOfFE_;
   int           numElemBlocks_;
   MLI_ElemBlock **elemBlockList_;
   int           currentElemBlock_;
   int           numFields_;
   int           *fieldIDs_;
   int           *fieldSizes_;

   // The private fields below appear to be unused
   /*
   int           elemsAssembled_;
   int           nodesAssembled_;
   int           facesAssembled_;

   void          *USR_FEMatrixObj_;
   */

   void          *USR_FEGridObj_;
   int           (*USR_computeShapeFuncInterpolant)(void*, int eGlobalID, 
                       int nNodes, const double *coord, double *coef);
   int           (*USR_getElemMatrix)(void*, int eGlobalID, int sMatDim,
                       double *stiffMat);

public :

   MLI_FEData(MPI_Comm comm);

   ~MLI_FEData();

   // -------------------------------------------------------------------------
   // load general information
   // -------------------------------------------------------------------------

   int setOutputLevel(int level);

   int setSpaceDimension(int numDim);

   int setOrderOfPDE(int pdeOrder);

   int setOrderOfFE(int feOrder);

   // =========================================================================
   // initialization functions
   // =========================================================================

   int setCurrentElemBlockID(int blockID);

   int initFields(int numFields, const int *fieldSizes, const int *fieldIDs);

   int initElemBlock(int nElems, int nNodesPerElem,
                     int nodeNumFields, const int *nodeFieldIDs,
                     int elemNumFields, const int *elemFieldIDs);

   int initElemBlockNodeLists(int nElems, const int *eGlobalIDs,
                              int nNodesPerElem,
                              const int* const *nGlobalIDLists,
                              int spaceDim, const double* const *coord);

   int initElemNodeList(int eGlobalIDs,int nNodesPerElem,const int *nGlobalIDs,
                        int spaceDim, const double *coord);

   int initSharedNodes(int nNodes, const int *nGlobalIDs, const int *numProcs,
                       const int * const *procLists);

   int initElemBlockFaceLists(int nElems, int nFaces,
                              const int* const *fGlobalIDLists); 

   int initFaceBlockNodeLists(int nFaces, const int *fGlobalIDs,
                              int nNodes, const int * const *nGlobalIDLists);

   int initSharedFaces(int nFaces, const int *fGlobalIDs,
                       const int *numProcs, const int* const *procLists);

   int initComplete();

   // =========================================================================
   // load element information
   // =========================================================================

   // -------------------------------------------------------------------------
   // collective loading of element data
   // -------------------------------------------------------------------------

   int loadElemBlockMatrices(int nElems, int sMatDim,
                             const double* const *stiffMat);

   int loadElemBlockNullSpaces(int nElems, const int *nNSpace,
                               int sMatDim, const double* const *nSpace);

   int loadElemBlockVolumes(int nElems, const double *elemVols);

   int loadElemBlockMaterials(int nElems, const int *elemMaterial);

   int loadElemBlockParentIDs(int nElems, const int *pGlobalIDs);

   int loadElemBlockLoads(int nElems, int loadDim, 
                          const double* const *elemLoads);

   int loadElemBlockSolutions(int nElems, int solDim, 
                              const double* const *elemSols);

   int loadElemBCs(int nElems, const int *eGlobalIDs, int elemDOF, 
                   const char* const *BCFlags, const double *const *bcVals);

   // -------------------------------------------------------------------------
   // These functions allows elements to be loaded individually.
   // -------------------------------------------------------------------------

   int loadElemMatrix(int eGlobalID, int sMatDim, const double *stiffMat);

   int loadElemNullSpace(int eGlobalID, int nNSpace, int sMatDim, 
                         const double *nSpace);

   int loadElemLoad(int eGlobalID, int sMatDim, const double *elemLoad);

   int loadElemSolution(int eGlobalID, int sMatDim, const double *elemSol);

   //int loadFunc_getElemMatrix(void *object, int (*func)(void *,int eGlobalID,
   //                int sMatDim,double *stiffMat));

   // =========================================================================
   // load node boundary conditions
   // =========================================================================

   int loadNodeBCs(int nNodes, const int *nGlobalIDs, int nodeDOF, 
                   const char *const *BCFlags, const double* const *bcVals);

   // =========================================================================
   // get general information
   // =========================================================================

   int getSpaceDimension(int& numDim);

   int getOrderOfPDE(int& pdeOrder);

   int getOrderOfFE(int& feOrder);

   int getFieldSize(int fieldID, int &fieldSize);

   // =========================================================================
   // get element information
   // =========================================================================

   int getNumElements(int& nElems);

   int getElemNumFields(int& numFields);

   int getElemFieldIDs(int numFields, int *fieldIDs);

   int getElemGlobalID(int eLocalID, int &eGlobalID);

   int getElemBlockGlobalIDs(int nElems, int *eGlobalIDs);

   int getElemNumNodes(int& nNodes); 

   int getElemBlockNodeLists(int nElems,int nNodes,int **nGlobalIDLists);

   int getElemMatrixDim(int &sMatDim);

   int getElemBlockMatrices(int nElems, int sMatDim, double **elemMat);

   int getElemBlockNullSpaceSizes(int nElems, int *dimsNS);

   int getElemBlockNullSpaces(int nElems, const int *dimsNS, int sMatDim, 
                              double **nullSpaces);

   int getElemBlockVolumes(int nElems, double *elemVols);

   int getElemBlockMaterials(int nElems, int *elemVols);

   int getElemBlockParentIDs(int nElems, int *pGlobalIDs);

   int getElemNumFaces(int& nFaces);

   int getElemBlockFaceLists(int nElems, int nFaces, int **fGlobalIDLists);

   int getElemNodeList(int eGlobalID, int nNodes, int *nGlobalIDs);

   int getElemMatrix(int eGlobalID, int sMatDim, double *elemMat);

   int getElemNullSpaceSize(int eGlobalID, int &dimNS); 

   int getElemNullSpace(int eGlobalID, int dimNS, int sMatDim, double *nSpace);

   int getElemVolume(int eGlobalID, double& elemVol);

   int getElemMaterial(int eGlobalID, int& elemMat);

   int getElemParentID(int eGlobalID, int& pGlobalID); 

   int getElemFaceList(int eGlobalID, int nFaces, int *fGlobalIDs); 

   int getNumBCElems(int& nElems);

   int getElemBCs(int nElems, int *eGlobalIDs, int eDOFs, char **fieldFlag, 
                  double **BCVals);

   // =========================================================================
   // get node information
   // =========================================================================

   int getNumNodes(int& nNodes);

   int getNodeBlockGlobalIDs(int nNodes, int *nGlobalIDs);

   int getNodeNumFields(int &numFields); 

   int getNodeFieldIDs(int numFields, int *fieldIDs); 

   int getNodeBlockCoordinates(int nNodes, int spaceDim, double *coordinates);

   int getNumBCNodes(int& nNodes);

   int getNodeBCs(int nNodes, int *nGlobalIDs, int nDOFs, char **fieldFlag, 
                  double **BCVals);

   int getNumSharedNodes(int& nNodes); 
    
   int getSharedNodeNumProcs(int nNodes, int *nGlobalIDs, int *numProcs);

   int getSharedNodeProcs(int nNodes, int *numProcs, int **procList);

   // -------------------------------------------------------------------------
   // get face information
   // -------------------------------------------------------------------------

   int getNumFaces(int& nfaces);

   int getFaceBlockGlobalIDs(int nFaces, int *fGlobalIDs);

   int getNumSharedFaces(int& nFaces); 

   int getSharedFaceNumProcs(int nFaces, int *fGlobalIDs, int *numProcs);

   int getSharedFaceProcs(int nFaces, int *numProcs, int **procList);

   int getFaceNumNodes(int &nNodes); 

   int getFaceBlockNodeLists(int nFaces, int nNodesPerFace,
                             int **nGlobalIDLists);

   int getFaceNodeList(int fGlobalID, int nNodes, int *nGlobalIDs);

   // -------------------------------------------------------------------------
   // shape function information
   // -------------------------------------------------------------------------

   int loadFunc_computeShapeFuncInterpolant(void *object, int (*func) (void *,
                   int eGlobalID,int nNodes,const double *coord, double *coef)); 

   int getShapeFuncInterpolant(int eGlobalID, int nNodes, const double *coord, 
                               double *coef); 

   // -------------------------------------------------------------------------
   // other functions
   // -------------------------------------------------------------------------

   int impSpecificRequests(char *param_string, int argc, char **argv);

   int readFromFile(char *filename);

   int writeToFile(char *filename);

   // -------------------------------------------------------------------------
   // internal functions
   // -------------------------------------------------------------------------

   int createElemBlock(int blockID);
   int deleteElemBlock(int blockID);
   int searchElement(int);
   int searchNode(int key);
   int searchFace(int key);
};

#endif

