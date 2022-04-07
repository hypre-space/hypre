/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLIFEBASEH__
#define __MLIFEBASEH__

/****************************************************************************/ 
/* class definition for abstract finite element structure                   */
/* (The design of this class attempts to follow the FEI 2.0 (Sandia) as     */
/*  closely as possible.  Functions related to matrix and solver            */
/*  construction and utilization are removed.  Additional finite element    */
/*  information (e.g. nodal coordinates, face information) have been added) */
/*--------------------------------------------------------------------------*/

class MLI_FEBase 
{
public :

   MLI_FEBase();

   virtual ~MLI_FEBase();

   // =========================================================================
   // load general information
   // =========================================================================

   virtual int setOutputLevel(int level);

   virtual int setSpaceDimension(int numDim);

   virtual int setOrderOfPDE(int pdeOrder);

   virtual int setOrderOfFE(int feOrder);

   // =========================================================================
   // initialization functions 
   // =========================================================================

   virtual int setCurrentElemBlockID(int blockID);

   virtual int initFields(int numFields, const int *fieldSizes, 
                          const int *fieldIDs); 

   virtual int initElemBlock(int nElems, int nNodesPerElem, 
                             int nodeNumFields, const int *nodeFieldIDs,
                             int elemNumFields, const int *elemFieldIDs);

   virtual int initElemNodeList(int elemID,int elemNNodes,const int *nodeIDs,
                        int spaceDim, const double *coord);

   virtual int initElemBlockNodeLists(int nElems, const int *eGlobalIDs, 
                       int nNodesPerElem, const int* const *nGlobalIDLists,
                       int spaceDim, const double* const *coord);

   virtual int initSharedNodes(int nNodes, const int *nGlobalIDs, 
                   const int *numProcs, const int * const *procLists);

   virtual int initElemBlockFaceLists(int nElems, int nFaces, 
                                      const int* const *fGlobalIDLists);

   virtual int initFaceBlockNodeLists(int nFaces, const int *fGlobalIDs, 
                       int nNodes, const int * const *nGlobalIDLists);

   virtual int initSharedFaces(int nFaces, const int *fGlobalIDs, 
                       const int *numProcs, const int* const *procLists);

   virtual int initComplete();

   // =========================================================================
   // load element information
   // =========================================================================

   virtual int loadElemBlockMatrices(int nElems, int sMatDim, 
                                     const double* const *stiffMat);

   virtual int loadElemBlockNullSpaces(int nElems, const int *nNSpace, 
                            int sMatDim, const double* const *nSpace); 

   virtual int loadElemBlockVolumes(int nElems, const double *elemVols); 

   virtual int loadElemBlockMaterials(int nElems, const int *elemMaterial);

   virtual int loadElemBlockParentIDs(int nElems, const int *pGlobalIDs);

   virtual int loadElemBlockLoads(int nElems, int loadDim, 
                                  const double* const *elemLoads);

   virtual int loadElemBlockSolutions(int nElems, int solDim,
                                     const double* const *elemSols);

   virtual int loadElemBCs(int nElems, const int *eGlobalIDs, 
                           int elemDOF, const char *const *BCFlags, 
                           const double *const *bcVals);

   virtual int loadElemMatrix(int eGlobalID, int sMatDim, 
                              const double *stiffMat);

   virtual int loadElemNullSpace(int eGlobalID, int nNSpace, int sMatDim, 
                                 const double *nSpace);

   virtual int loadElemLoad(int eGlobalID, int sMatDim, 
                            const double *elemLoad);

   virtual int loadElemSolution(int eGlobalID, int sMatDim, 
                                const double *elemSol);

   virtual int loadFunc_getElemMatrix(void *object, 
                int (*func)(void*,int eGlobalID,int sMatDim,double *stiffMat));

   // =========================================================================
   // load node boundary conditions and share nodes
   // =========================================================================

   virtual int loadNodeBCs(int nNodes, const int *nGlobalIDs, 
                           int nodeDOF, const char *const *BCFlags, 
                           const double * const *bcVals);

   // =========================================================================
   // get general information
   // =========================================================================

   virtual int getSpaceDimension(int& numDim);

   virtual int getOrderOfPDE(int& pdeOrder);

   virtual int getOrderOfFE(int& feOrder);

   virtual int getFieldSize(int fieldID, int &fieldSize);

   // =========================================================================
   // get element information
   // =========================================================================

   virtual int getNumElements(int& nElems);

   virtual int getElemNumFields(int& numFields);

   virtual int getElemFieldIDs(int numFields, int *fieldIDs);

   virtual int getElemBlockGlobalIDs(int nElems, int *eGlobalIDs);

   virtual int getElemNumNodes(int& nNodes);

   virtual int getElemBlockNodeLists(int nElems, int nNodes, 
                                    int **nGlobalIDLists);

   virtual int getElemMatrixDim(int &sMatDim);

   virtual int getElemBlockMatrices(int nElems, int sMatDim, double **elemMat);

   virtual int getElemBlockNullSpaceSizes(int nElems, int *dimsNS);

   virtual int getElemBlockNullSpaces(int nElems, const int *dimsNS, 
                                      int sMatDim, double **nullSpaces);

   virtual int getElemBlockVolumes(int nElems, double *elemVols);

   virtual int getElemBlockMaterials(int nElems, int *elemMats);

   virtual int getElemBlockParentIDs(int nElems, int *pGlobalIDs);

   virtual int getElemNumFaces(int& nFaces);

   virtual int getElemBlockFaceLists(int nElems, int nFaces, 
                                     int **fGlobalIDLists);

   virtual int getElemNodeList(int eGlobalID, int nNodes, int *nGlobalIDs);

   virtual int getElemMatrix(int eGlobalID, int sMatDim, double *elemMat);

   virtual int getElemNullSpaceSize(int eGlobalID, int &dimNS);

   virtual int getElemNullSpace(int eGlobalID, int dimNS, 
                                int sMatDim, double *nSpace);

   virtual int getElemVolume(int eGlobalID, double& elemVol);

   virtual int getElemMaterial(int eGlobalID, int& elemMaterial);

   virtual int getElemParentID(int eGlobalID, int& pGlobalID);

   virtual int getElemFaceList(int eGlobalID, int nFaces, int *fGlobalIDs);

   virtual int getNumBCElems(int& nElems);

   virtual int getElemBCs(int nElems, int *eGlobalIDs, int eDOFs, 
                          char **BCFlags, double **BCVals);

   // =========================================================================
   // get node information
   // =========================================================================

   virtual int getNumNodes(int& nNodes); 

   virtual int getNodeBlockGlobalIDs(int nNodes, int *nGlobalIDs);

   virtual int getNodeNumFields(int &numFields);

   virtual int getNodeFieldIDs(int numFields, int *fieldIDs);

   virtual int getNodeBlockCoordinates(int nNodes, int spaceDim,
                                       double *coordinates);

   virtual int getNumBCNodes(int& nNodes);

   virtual int getNodeBCs(int nNodes, int *nGlobalIDs, 
                          int nDOFs, char **BCFlags, double **BCVals);

   virtual int getNumSharedNodes(int& nNodes);
    
   virtual int getSharedNodeNumProcs(int nNodes, int *nGlobalIDs, 
                                     int *numProcs);

   virtual int getSharedNodeProcs(int nNodes, int *numProcs, 
                                  int **procList);

   // =========================================================================
   // get face information
   // =========================================================================

   virtual int getNumFaces(int& nFaces);

   virtual int getFaceBlockGlobalIDs(int nFaces, int *fGlobalIDs);

   virtual int getNumSharedFaces(int& nFaces);

   virtual int getSharedFaceNumProcs(int nFaces, int *fGlobalIDs,  
                                     int *numProcs);

   virtual int getSharedFaceProcs(int nFaces, int *numProcs, int **procList);

   virtual int getFaceNumNodes(int &nNodes);

   virtual int getFaceBlockNodeLists(int nFaces, int nNodesPerFace,
                                     int **nGlobalIDLists);

   virtual int getFaceNodeList(int fGlobalID, int nNodes, int *nGlobalIDs);

   // -------------------------------------------------------------------------
   // shape function information
   // -------------------------------------------------------------------------

   virtual int loadFunc_computeShapeFuncInterpolant(void *object, 
                   int (*func) (void *,int elemID,int nNodes,const double *coor,
                   double *coef));

   virtual int getShapeFuncInterpolant(int eGlobalID, int nNodes, 
                                       const double *coord, double *coef);

   // -------------------------------------------------------------------------
   // other functions
   // -------------------------------------------------------------------------

   virtual int impSpecificRequests(char *paramString, int argc, char **argv);

   virtual int readFromFile(char *filename);

   virtual int writeToFile(char *filename);

};

#endif

