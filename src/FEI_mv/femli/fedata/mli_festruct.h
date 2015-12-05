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





#ifndef __MLIFESTRUCTH__
#define __MLIFESTRUCTH__

/****************************************************************************/ 
/* class definition for abstract finite element structure                   */
/* (The design of this class attempts to follow the FEI 2.0 (Sandia) as     */
/*  closely as possible.  Functions related to matrix and solver            */
/*  construction and utilization are removed.  Additional finite element    */
/*  information (e.g. nodal coordinates, face information) have been added) */
/*--------------------------------------------------------------------------*/

class MLI_FEStruct 
{
public :

   MLI_FEStruct() {}

   virtual ~MLI_FEStruct() {}

   // =========================================================================
   // load general information
   // =========================================================================

   /** set diagnostics level 
       @param level - diagnostics level (0 or higher) 
   */
   virtual int setOutputLevel(int level) {(void) level; return -1;}

   /** set space dimension 
       @param numDim - number of space dimension (3 for 3D)
   */
   virtual int setSpaceDimension(int numDim) {(void) numDim; return -1;}

   /** set order of PDE 
       @param pdeOrder - order of PDE (e.g. 2 for second order)
   */
   virtual int setOrderOfPDE(int pdeOrder) {(void) pdeOrder; return -1;}

   /** set order of finite element discretization 
       @param feOrder - order of FE discretization e.g. 2 for second order)
   */
   virtual int setOrderOfFE(int feOrder) {(void) feOrder; return -1;}

   // =========================================================================
   // initialization functions 
   // =========================================================================

   /** set the current element block to load 
       @param blockID - choose blockID as current element block number 
                        (if not called, assume blockID=0)
   */
   virtual int setCurrentElemBlockID(int blockID) {(void) blockID; return -1;}

   /** declare all field variables used in this context
       @param numFields  - total number of fields
       @param fieldSizes - degree of freedom for each fieldID declared
       @param fieldIDs   - a list of distinct field IDs
   */
   virtual int initFields(int numFields, 
                          const int *fieldSizes, 
                          const int *fieldIDs) = 0;

   /** initialize element block information 
       @param nElems        - no. of elements in this block
       @param nNodesPerElem - no of nodes in each element in this block
       @param nodeNumFields - number of fields per node
       @param nodeFieldIDs  - node field IDs
       @param elemNumFields - number of element fields
       @param elemFieldIDs  - element field IDs
   */
   virtual int initElemBlock(int nElems, 
                             int nNodesPerElem, 
                             int nodeNumFields, 
                             const int *nodeFieldIDs,
                             int elemNumFields, 
                             const int *elemFieldIDs) = 0;

   /** initialize element connectivities 
       @param nElems - no. of elements in this block
       @param eGlobalIDs - a list of global element IDs
       @param nNodesPerElem - no of nodes in each element in this block
       @param nGlobalIDLists - lists of node IDs for each element
       @param spaceDim - space dimension (e.g. 3 for 3D)
       @param coord - nodal coordinates (can be NULL)
   */
   virtual int initElemBlockNodeLists(int nElems, 
                                      const int *eGlobalIDs, 
                                      int nNodesPerElem, 
                                      const int* const *nGlobalIDLists,
                                      int spaceDim, 
                                      const double* const *coord) = 0;

   /** initialize all shared nodes and which processors have them 
       @param nNodes - number of shared nodes 
       @param nGlobalIDs - IDs of shared nodes 
       @param numProcs[i] - number of processors node i shares with
       @param procLists[i] - a list of processors to share node i
   */
   virtual int initSharedNodes(int nNodes, 
                               int *nGlobalIDs, 
                               const int *numProcs, 
                               const int * const *procLists) 
                               {(void) nNodes; (void) nGlobalIDs;
                                (void) numProcs; (void) procLists;
                                return -1;}

   /** initialize the element face lists of all elements in local processors 
       in the same order as the globalIDs list given above
       @param nElems     - has to match nElems in loadElemBlock
       @param nFaces     - number of faces for each element
       @param fGlobalIDs - lists of face global IDs for each element
   */
   virtual int initElemBlockFaceLists(int nElems, 
                                      int nFaces, 
                                      const int* const *fGlobalIDLists) 
                                      {(void) nElems; (void) nFaces;
                                       (void) fGlobalIDLists; return -1;}

   /** initialize node lists of all local faces 
       @param nFaces            - number of faces in local processor
       @param fGlobalIDs        - a list of face global IDs
       @param nNodes            - number of nodes each face has
       @param nGlobalIDLists[i] - node list of face fGlobalIDs[i]
   */
   virtual int initFaceBlockNodeLists(int nFaces, 
                                      const int *fGlobalIDs, 
                                      int nNodes, 
                                      const int * const *nGlobalIDLists) 
                                      {(void) nFaces; (void) fGlobalIDs;
                                       (void) nNodes; (void) nGlobalIDLists;
                                       return -1;}

   /** initialize a list of faces that are shared with other processors 
       @param nFaces - number of shared faces 
       @param fGlobalIDs - IDs of shared faces 
       @param numProcs[i] - number of processors face i shares with
       @param procLists[i] - a list of processors to share face i
   */
   virtual int initSharedFaces(int nFaces, 
                               const int *fGlobalIDs, 
                               const int *numProcs, 
                               const int* const *procLists) 
                               {(void) nFaces; (void) fGlobalIDs;
                                (void) numProcs; (void) procLists;
                                return -1;}

   /** initialization complete
   */
   virtual int initComplete() = 0;

   // =========================================================================
   // load element information
   // =========================================================================

   // -------------------------------------------------------------------------
   // Collective loading : may mean excessive memory allocation
   // -------------------------------------------------------------------------

   /** Load connectivity and stiffness matrices of all elements in local
       processor (collective loading for one element block) 
       @params nElems - no. of elements in this block
       @params sMatDim - dimension of element matrices (for checking only)
       @params stiffMat - element matrices 
   */
   virtual int loadElemBlockMatrices(int nElems, 
                                     int sMatDim, 
                                     const double* const *stiffMat) 
                                     {(void) nElems; (void) sMatDim;
                                      (void) stiffMat; return -1;}

   /** Load null spaces of all elements in local processors in the same
       order as the globalIDs list given above
       @param nElems  - has to match nElems in loadElemBlock
       @param nNSpace - number of null space for each elements (it is an
                        array as it may be different for different elements)
       @param sMatDim - number of unknowns for each element (for checking)
       @param nSpace  - the null space information (column major)
   */
   virtual int loadElemBlockNullSpaces(int nElems, 
                                       const int *nNSpace, 
                                       int sMatDim, 
                                       const double* const *nSpace) 
                                       {(void) nElems; (void) nNSpace;
                                        (void) sMatDim; (void) nSpace;
                                        return -1;}

   /** Load volumes of all elements in local processors in the same
       order as the globalIDs list given above
       @param nElems   - has to match nElems in loadElemBlock
       @param elemVols - volume for each element in the list
   */
   virtual int loadElemBlockVolumes(int nElems, 
                                    const double *elemVols)  
                                    {(void) nElems; (void) elemVols;
                                     return -1;}

   /** Load material type of all elements in local processors in the same
       order as the globalIDs list given above
       @param nElems        - has to match nElems in loadElemBlock
       @param elemMaterials - material types for each element
   */
   virtual int loadElemBlockMaterials(int nElems, 
                                      const int *elemMaterial)  
                                      {(void) nElems; (void) elemMaterial;
                                       return -1;}

   /** Load the element parent IDs of all elements in local processors in the 
       same order as the globalIDs list given above
       @param nElems     - has to match nElems in loadElemBlock
       @param pGlobalIDs - parent element global IDs for each element
   */
   virtual int loadElemBlockParentIDs(int nElems, 
                                      const int *pGlobalIDs) 
                                      {(void) nElems; (void) pGlobalIDs;
                                       return -1;}

   /** Load the element loads (rhs) of all elements in local processors in the 
       same order as the globalIDs list given above
       @param nElems    - has to match nElems in loadElemBlock
       @param loadDim   - length of each load vector (for checking only)
       @param elemLoads - right hand sides for each element
   */
   virtual int loadElemBlockLoads(int nElems, 
                                  int loadDim, 
                                  const double* const *elemLoads) 
                                  {(void) nElems; (void) loadDim;
                                   (void) elemLoads; return -1;}

   /** Load the element initial guess of all elements in local processors in 
       the same order as the globalIDs list given above
       @param nElems   - has to match nElems in loadElemBlock
       @param loadDim  - length of each solution vector (for checking)
       @param elemSols - solution for each element
   */
   virtual int loadElemBlockSolutions(int nElems, 
                                     int solDim,
                                     const double* const *elemSols) 
                                     {(void) nElems; (void) solDim;
                                      (void) elemSols; return -1;}

   /** Load element boundary conditions 
       @param nElems     - number of elements having boundary conditions
       @param eGlobalIDs - element global IDs
       @param elemDOF    - element degree of freedom
       @param BCFlags    - flag those DOFs that are BCs ('Y' for BCs)
       @param bcVals     - boundary condition values (nElems values)
   */
   virtual int loadElemBCs(int nElems, 
                           const int *eGlobalIDs, 
                           int elemDOF, 
                           const char *const *BCFlags, 
                           const double *const *bcVals) 
                           {(void) nElems; (void) eGlobalIDs;
                            (void) elemDOF; (void) BCFlags;
                            (void) bcVals; return -1;}

   // -------------------------------------------------------------------------
   // These functions allows elements to be loaded individually.
   // This may be more memory efficient, but not CPU efficient (many searches)
   // -------------------------------------------------------------------------

   /** Load element matrix 
       @param eGlobalID     - element global ID 
       @param nNodesPerElem - number of nodes per element (for checking)
       @param nGlobalIDs    - a list of nodes for the element
       @param sMatDim       - dimension of the element matrix (for checking)
       @param stiffMat      - element stiffness matrix (column major)
   */
   virtual int loadElemMatrix(int eGlobalID, 
                              int nNodesPerElem, 
                              const int *nGlobalIDs, 
                              int sMatDim, 
                              const double *stiffMat) 
                              {(void) eGlobalID; (void) nNodesPerElem; 
                               (void) nGlobalIDs; (void) sMatDim; 
                               (void) stiffMat; return -1;}

   /** Load element null space 
       @param eGlobalID - element global ID 
       @param nNSpace   - number of null space vectors in the element
       @param sMatDim   - length of each null space (for checking)
       @param nSpace    - null space stored in column major manner
   */
   virtual int loadElemNullSpace(int eGlobalID, 
                                 int nNSpace, 
                                 int sMatDim, 
                                 const double *nSpace) 
                                 {(void) eGlobalID; (void) nNSpace; 
                                  (void) sMatDim; (void) nSpace; 
                                  return -1;}

   /** load element load 
       @param eGlobalID - element global ID 
       @param sMatDim   - dimension of the load vector (for checking)
       @param elemLoad  - the element load vector
   */
   virtual int loadElemLoad(int eGlobalID, 
                            int sMatDim, 
                            const double *elemLoad)
                            {(void) eGlobalID; (void) sMatDim;
                             (void) elemLoad; return -1;}

   /** load element solution (initial guess) 
       @param eGlobalID - element global ID 
       @param sMatDim - dimension of the solution vector
       @param elemSol - the element solution vector
   */
   virtual int loadElemSolution(int eGlobalID, 
                                int sMatDim, 
                                const double *elemSol)
                                {(void) eGlobalID; (void) sMatDim;
                                 (void) elemSol; return -1;}

   /** load the function pointer to fetch element stiffness matrices 
       @param object - object to pass back to function to get element matrices
       @param func   - pointer to the function for getting element matrices
   */
   virtual int loadFunc_getElemMatrix(void *object, 
                   int (*func) (int eGlobalID,int sMatDim,double *stiffMat)) 
                   {(void) object; (void) func; return -1;}

   // =========================================================================
   // load node boundary conditions and share nodes
   // =========================================================================

   /** load all node essential boundary conditions 
       @param nNodes - number of BC nodes 
       @param nGlobalIDs - IDs of boundary nodes 
       @param nodeDOF - total DOF in a node (for checking only)
       @param BCFlags - flag those DOFs that are BCs ('Y' - BCs)
       @param bcVals - boundary condition values (nNodes values)
   */
   virtual int loadNodeBCs(int nNodes, 
                           const int *nGlobalIDs, 
                           int nodeDOF, 
                           const char *const *BCFlags, 
                           const double * const *bcVals) 
                           {(void) nNodes; (void) nGlobalIDs; 
                            (void) nodeDOF; (void) BCFlags;
                            (void) bcVals; return -1;}

   // =========================================================================
   // get general information
   // =========================================================================

   /** Return the space dimension (2D or 3D) 
       @param (output) numDim - number of space dimension (3 for 3D)
   */
   virtual int getSpaceDimension(int& numDim) {(void) numDim; return -1;}

   /** Return the order of the PDE 
       @param (output) pdeOrder - order of PDE (2 for 2nd order)
   */
   virtual int getOrderOfPDE(int& pdeOrder) {(void) pdeOrder; return -1;}

   /** Return the order of the FE discretization 
       @param (output) feOrder - order of finite element discretization
   */
   virtual int getOrderOfFE(int& feOrder) {(void) feOrder; return -1;}

   /** Return the number of unknowns for a given field 
       @param (input)  fieldID   - field ID
       @param (output) fieldSize - degree of freedom for the field
   */
   virtual int getFieldSize(int fieldID, 
                            int &fieldSize) 
                            {(void) fieldID; (void) fieldSize; return -1;}

   // =========================================================================
   // get element information
   // =========================================================================

   /** Return the number of the local elements in nElems 
       @param (output) nElems - return the number of element in this block 
   */
   virtual int getNumElements(int& nElems) 
                              {(void) nElems; return -1;}

   /** Return the number of fields in each element in this block
       @param (output) numFields - number of element fields in this block 
   */
   virtual int getElemNumFields(int& numFields) 
                                {(void) numFields; return -1;}

   /** Return the field IDs for each element in this block
       @param (input)  numFields - number of element fields (checking)
       @param (output) fieldIDs  - field IDs
   */
   virtual int getElemFieldIDs(int numFields,
                               int *fieldIDs)
                               {(void) numFields; (void) fieldIDs; 
                                return -1;}

   // -------------------------------------------------------------------------
   // collective gets
   // -------------------------------------------------------------------------

   /** Return all element global IDs (of size from getNumElements) 
       @param (input)  nElems     - number of elements (from getNumElements)
       @param (output) eGlobalIDs - has the element GlobalIDs upon return
                      (eGlobalIDs should be pre-allocated with length nElems)
   */
   virtual int getElemBlockGlobalIDs(int nElems,
                                     int *eGlobalIDs) 
                                     {(void) nElems; (void) eGlobalIDs;
                                      return -1;}

   /** Return the number of nodes for the elements in this block 
       @param (output) nNodes - number of nodes for each element upon return
   */
   virtual int getElemNumNodes(int& nNodes) 
                               {(void) nNodes; return -1;}

   /** Return the globalIDs of nodes for elements with global IDs 
       given in the call to getElemGlobalIDs (same order) 
       @param (input)  nElems         - no. of elements in this block (checking)
       @param (input)  nNodes         - no. of nodes for each element (checking)
       @param (ouptut) nGlobalIDLists - node lists for each element 
   */
   virtual int getElemBlockNodeLists(int nElems, 
                                    int nNodes, 
                                    int **nGlobalIDLists) 
                                    {(void) nElems; (void) nNodes; 
                                     (void) nGlobalIDLists; return -1;}

   /** Return dimension of element matrix in this block
       @param (output) sMatDim - dimension of the element matrix 
   */
   virtual int getElemMatrixDim(int &sMatDim) 
                                {(void) sMatDim; return -1;}

   /** Return element matrices for all elements in the block with global IDs
       given in the call to getElemGlobalIDs (same order) 
       @param (input)  nElems - number of elements in this block (checking)
       @param (input)  sMatDim - dimension of element matrices (checking)
       @param (ouptut) elemMat - a list of element matrices
   */
   virtual int getElemBlockMatrices(int nElems, 
                                    int sMatDim, 
                                    double **elemMat) 
                                    {(void) nElems; (void) sMatDim; 
                                     (void) elemMat; return -1;}

   /** Return the element null space sizes for elements with global IDs 
       given in the call to getElemGlobalIDs (same order) 
       @param (input)  nElems - number of elements in this block (checking)
       @param (output) dimsNS - dimensions of the null spaces for all elements
   */
   virtual int getElemBlockNullSpaceSizes(int nElems, 
                                          int *dimsNS) 
                                          {(void) nElems; (void) dimsNS;
                                           return -1;}

   /** Return the element null space for elements with global IDs 
       given in the call to getElemGlobalIDs (same order) 
       @param (input)  nElems        - no. of elements in this block (checking)
       @param (input)  dimsNS[i]     - no. of null space vectors for element i
       @param (input)  sMatDim       - dimension of each null space vector
       @param (ouptut) nullSpaces[i] - null space vectors for element i
   */
   virtual int getElemBlockNullSpaces(int nElems, 
                                      const int *dimsNS, 
                                      int sMatDim, 
                                      double **nullSpaces) 
                                      {(void) nElems; (void) dimsNS;
                                       (void) sMatDim; (void) nullSpaces;
                                       return -1;}

   /** Return element volumes for elements given in the call to 
       getElemGlobalIDs (in the same order) 
       @param (input)  nElems   - number of elements in this block (checking)
       @param (output) elemVols - element volumes
   */  
   virtual int getElemBlockVolumes(int nElems, 
                                   double *elemVols) 
                                   {(void) nElems; (void) elemVols;
                                    return -1;}

   /** Return element materials for elements given in the call to 
       getElemGlobalIDs (in the same order) 
       @param (input)  nElems   - number of elements in this block (checking)
       @param (output) elemMats - element materials
   */  
   virtual int getElemBlockMaterials(int nElems, 
                                     int *elemMats) 
                                     {(void) nElems; (void) elemMats;
                                      return -1;}

   /** Return the parent IDs of elements (global IDs) in the same order as
       the list obtained from a call to getElemGlobalIDs 
       @param (input)  nElems      - no. of elements in this block (checking)
       @param (output) pGlobalIDts - list of parent elements' global IDs
   */
   virtual int getElemBlockParentIDs(int nElems, 
                                     int *pGlobalIDs) 
                                     {(void) nElems; (void) pGlobalIDs;
                                      return -1;}

   /** Return the number of faces of each element in this block
       @param (output) nFaces - number of faces for elements in this block
   */
   virtual int getElemNumFaces(int& nFaces) 
                               {(void) nFaces; return -1;}

   /** Return the face lists of the elements (global IDs) in the same
       order as the IDs obtained by a call to the getElemGlobalIDs 
       @param (input)  nElems         - no. of elements in this block (checking)
       @param (input)  nFaces         - no. of faces for all elements (checking)
       @param (output) fGlobalIDLists - lists of face global IDs
   */
   virtual int getElemBlockFaceLists(int nElems, 
                                     int nFaces, 
                                     int **fGlobalIDLists) 
                                     {(void) nElems; (void) nFaces;
                                      (void) fGlobalIDLists; return -1;}

   // -------------------------------------------------------------------------
   // individual (element) gets
   // -------------------------------------------------------------------------

   /** Return the globalIDs of nodes for the element given element's global ID 
       @param (input)  eGlobalID  - global ID of element 
       @param (input)  nNodes     - number of nodes in this element (checking)
       @param (output) nGlobalIDs - a list of node global IDs for this element
   */
   virtual int getElemNodeList(int eGlobalID, 
                               int nNodes, 
                               int *nGlobalIDs) 
                               {(void) eGlobalID; (void) nNodes;
                                (void) nGlobalIDs; return -1;}

   /** Return an element stiffness matrix 
       @param (input)  eGlobalID - global ID of element 
       @param (input)  sMatDim   - dimension of the element matrix (checking)
       @param (output) elemMat   - element matrix in column major ordering
   */
   virtual int getElemMatrix(int eGlobalID, 
                             int sMatDim, 
                             double *elemMat)
                             {(void) eGlobalID; (void) sMatDim;
                              (void) elemMat; return -1;}

   /** Return the element's null space size 
       @param (input)  eGlobalID - element global ID 
       @param (output) dimNS     - dimension of the null space
   */ 
   virtual int getElemNullSpaceSize(int eGlobalID, 
                                    int &dimNS) 
                                    {(void) eGlobalID; (void) dimNS;
                                     return -1;}

   /** Return the element's null space 
       @param (input)  eGlobalID - element global ID
       @param (input)  dimNS     - number of null space vectors
       @param (input)  sMatDim   - length of each null space vector
       @param (output) nSpace    - the null space vectors
   */
   virtual int getElemNullSpace(int eGlobalID, 
                                int dimNS, 
                                int sMatDim, 
                                double *nSpace) 
                                {(void) eGlobalID; (void) dimNS; 
                                 (void) sMatDim; (void) nSpace;
                                 return -1;}

   /** Return element volume 
       @param (input)  eGlobalID - element global ID
       @param (output) elemVol   - element volume
   */  
   virtual int getElemVolume(int eGlobalID, 
                             double& elemVol) 
                             {(void) eGlobalID; (void) elemVol; return -1;}

   /** Return element material 
       @param (input)  eGlobalID    - element global ID
       @param (output) elemMaterial - element material
   */  
   virtual int getElemMaterial(int eGlobalID, 
                             int& elemMaterial) 
                             {(void) eGlobalID; (void) elemMaterial; 
                              return -1;}

   /** Return the parent ID of a given element 
       @param (input)  eGlobalID  - element global ID
       @param (output) pGlobalIDs - parent element's global IDs
   */
   virtual int getElemParentID(int eGlobalID, 
                               int& pGlobalID) 
                               {(void) eGlobalID; (void) pGlobalID; 
                                return -1;}

   /** Return the face list of a given element 
       @param (input)  eGlobalID  - element global ID
       @param (input)  nFaces     - number of faces for this element (checking) 
       @param (output) fGlobalIDs - face global IDs
   */
   virtual int getElemFaceList(int eGlobalID, 
                               int nFaces, 
                               int *fGlobalIDs) 
                               {(void) eGlobalID; (void) nFaces;
                                (void) fGlobalIDs; return -1;}

   /** Return the number of elements having essential BCs 
       @param (output) nElems - number of boundary elements
   */
   virtual int getNumBCElems(int& nElems)
                             {(void) nElems; return -1;}

   /** Return the essential BCs 
       @param (input)  nElems     - number of boundary elements
       @param (output) eGlobalIDs - a list of boundary elements' global IDs 
       @param (output) eDOFs      - element DOFs (sum of all field DOFs)
                                    for checking purposes
       @param (output) BCFlags[i][j] = 'Y' if element i has a BC at the j-th
                       position (if node i has field 0 and 1 and field 0
                       has rank 1 and field 1 has rank 3, then if field 1
                       has a BC at its second position, then 
                       BCFlags[i][2] = 'Y')
       @param (output) BCVals - boundary values
   */
   virtual int getElemBCs(int nElems, 
                          int *eGlobalIDs, 
                          int eDOFs, 
                          char **BCFlags, 
                          double **BCVals) 
                          {(void) nElems; (void) eGlobalIDs;
                           (void) eDOFs; (void) BCFlags;
                           (void) BCVals; return -1;}

   // =========================================================================
   // get node information
   // =========================================================================

   /** Return the number of local and external nodes 
       @param (output) nNodes - number of nodes for this processor
   */
   virtual int getNumNodes(int& nNodes) 
                           {(void) nNodes; return -1;}

   /** Return the global node IDs corresponding to local IDs from 0 to nNodes-1 
       with nNodes is the parameter returned from getNumNodes
       @param (input)  nNodes     - total no. of nodes (checking)
       @param (output) nGlobalIDs - node global IDs
   */
   virtual int getNodeBlockGlobalIDs(int nNodes, 
                                     int *nGlobalIDs) 
                                     {(void) nNodes; (void) nGlobalIDs; 
                                      return -1;}

   /** Return the number of fields for each node in the block 
       @param (output) numFields - number of fields for each node
   */
   virtual int getNodeNumFields(int &numFields) 
                                {(void) numFields; return -1;}

   /** Return the field ID for each node in the block 
       @param (input)  numFields - number of fields for each node
       @param (output) fieldIDs - a list of field IDs 
   */
   virtual int getNodeFieldIDs(int numFields, 
                               int *fieldIDs) 
                               {(void) numFields; (void) fieldIDs; 
                                return -1;}

   /** Return the coordinates of the nodes in the block (in the order of
       the node IDs from the call to getNodeBlockGlobalIDs)
       @param (input)  nNodes      - number of local/external nodes
       @param (input)  spaceDim    - spatial dimension (2 for 2D)
       @param (output) coordinates - coordinates
   */
   virtual int getNodeBlockCoordinates(int nNodes, 
                                       int spaceDim,
                                       double *coordinates) 
                                       {(void) nNodes; (void) spaceDim;
                                        (void) coordinates; return -1;}

   /** Return the number of nodes having essential BCs 
       @param (output) nNodes - number of boundary nodes
   */
   virtual int getNumBCNodes(int& nNodes)
                             {(void) nNodes; return -1;}

   /** Return the essential BCs 
       @param (input)  nNodes     - number of boundary nodes
       @param (output) nGlobalIDs - a list of boundary nodes' global IDs 
       @param (output) nDOFs      - nodal DOFs (sum of all field DOFs)
                                    for checking purposes
       @param (output) BCFlags[i][j] = 'Y' if node i has a BC at the j-th
                       position (if node i has field 0 and 1 and field 0
                       has rank 1 and field 1 has rank 3, then if field 1
                       has a BC at its second position, then 
                       BCFlags[i][2] = 'Y')
       @param (output) BCVals - boundary values
   */
   virtual int getNodeBCs(int nNodes, 
                          int *nGlobalIDs, 
                          int nDOFs, 
                          char **BCFlags, 
                          double **BCVals) 
                          {(void) nNodes; (void) nGlobalIDs; 
                           (void) nDOFs; (void) BCFlags;
                           (void) BCVals; return -1;}

   /** Return the number of shared nodes 
       @param (output) nNodes - number of shared nodes
   */
   virtual int getNumSharedNodes(int& nNodes) 
                                 {(void) nNodes; return -1;}
    
   /** Return shared node list and for every of the nodes with how many
       processors is shared (the data is copied)
       @param (input)  nNodes      - number of shared nodes
       @param (output) nGlobalIDs  - node global IDs
       @param (output) numProcs[i] - number of processors sharing nGlobalIDs[i]
   */
   virtual int getSharedNodeNumProcs(int nNodes, 
                                     int *nGlobalIDs, 
                                     int *numProcs)
                                     {(void) nNodes; (void) nGlobalIDs; 
                                      (void) numProcs; return -1;}

   /** Return the processors that share the given node 
       @param (input)  nNodes - number of shared nodes
       @param (input)  numProcs[i] - number of processors sharing nGlobalIDs[i] 
                                     from call to getSharedNodeNumProcs
       @param (input)  procList[i] - processor IDs sharing nGlobalIDs[i]
   */
   virtual int getSharedNodeProcs(int nNodes, 
                                  int *numProcs, 
                                  int **procList) 
                                  {(void) nNodes; (void) numProcs;
                                   (void) procList; return -1;}

   // =========================================================================
   // get face information
   // =========================================================================

   /** Return the number of faces in my processor 
       @param (output) nFaces - number of faces
   */
   virtual int getNumFaces(int& nFaces) 
                           {(void) nFaces; return -1;}

   /** Return the global IDs of all internal and external faces
       @param (input)  nFaces     - number of external faces
       @param (output) fGlobalIDs - a list of all faces' global IDs
   */
   virtual int getFaceBlockGlobalIDs(int nFaces, 
                                     int *fGlobalIDs) 
                                     {(void) nFaces; (void) fGlobalIDs;
                                      return -1;}

   /** Return number of faces shared with other processors 
       @param (output) nFaces - number of shared faces
   */
   virtual int getNumSharedFaces(int& nFaces) 
                                 {(void) nFaces; return -1;}

   /** Return shared face information 
       @param (output) nFaces - number of shared faces
       @param (output) fGlobalIDs - face global IDs
       @param (output) numProcs[i] - number of processors sharing fGlobalIDs[i]
   */
   virtual int getSharedFaceNumProcs(int nFaces, 
                                     int *fGlobalIDs, int *numProcs) 
                                     {(void) nFaces; (void) fGlobalIDs; 
                                      (void) numProcs; return -1;}

   /** Return shared face information (processor) 
       @param (input)  nFaces      - number of shared faces
       @param (input)  numProcs[i] - number of processors sharing fGlobalIDs[i] 
                                     from call to getSharedFaceNumProcs
       @param (input)  procList[i] - processor IDs sharing fGlobalIDs[i]
   */
   virtual int getSharedFaceProcs(int nFaces, 
                                  int *numProcs, 
                                  int **procList) 
                                  {(void) nFaces; (void) numProcs;
                                   (void) procList; return -1;}

   /** Return the number of nodes in a face
       @param (output) nNodes - number of nodes in a face
   */
   virtual int getFaceNumNodes(int &nNodes) 
                               {(void) nNodes; return -1;}

   /** Return the number of nodes in a face
       @param (input)  nFaces         - toal number of faces in this block
       @param (input)  nNodesPerFace  - number of nodes in a face
       @param (output) nGlobalIDLists - lists of nodes for faces
   */
   virtual int getFaceBlockNodeLists(int nFaces,
                                     int nNodesPerFace,
                                     int **nGlobalIDLists) 
                                     {(void) nFaces; (void) nNodesPerFace;
                                      (void) nGlobalIDLists; return -1;}

   /** Return the node list of a face given a global face ID 
       @param (input)  fGlobalID  - face global ID 
       @param (input)  nNodes     - number of nodes in a face (for checking)
       @param (output) nGlobalIDs - node global IDs 
   */
   virtual int getFaceNodeList(int fGlobalID, 
                               int nNodes, 
                               int *nGlobalIDs) 
                               {(void) fGlobalID; (void) nNodes;
                                (void) nGlobalIDs; return -1;}

   // -------------------------------------------------------------------------
   // shape function information
   // -------------------------------------------------------------------------

   /** load the function pointer to fetch shape function interpolant 
       @param object - data to be passed back to the called function
       @param object - data object 
       @param func   - function pointer
   */
   virtual int loadFunc_computeShapeFuncInterpolant(void *object, 
                   int (*func) (void *,int elemID,int nNodes,const double *coor,
                   double *coef)) 
                   {(void) object; (void) func; return -1;}

   /** call fetch shape function interpolant 
       @param (input)  eGlobalID - element global ID
       @param (input)  nNodes    - number of nodes in this element
       @param (input)  coord     - coordinate information about this node
       @param (output) coef      - coefficients returned 
   */
   virtual int getShapeFuncInterpolant(int eGlobalID, 
                                       int nNodes, 
                                       const double *coord, 
                                       double *coef) 
                                       {(void) eGlobalID; (void) nNodes;
                                        (void) coord; (void) coef; return -1;}

   // -------------------------------------------------------------------------
   // other functions
   // -------------------------------------------------------------------------

   /** This function is used to get/set implementation-specific data 
       of a derived FEBase object. 
       @param (input)        paramString - command string
       @param (input)        argc         - dimension of argv
       @param (input/output) argv         - data
       returns the number of arguments returned, if appropriate.
   */
   virtual int impSpecificRequests(char *paramString, 
                                   int argc, 
                                   char **argv) 
                                   {(void) paramString; (void) argc;
                                    (void) argv; return -1;}

   /** read the element data from a file
       @param filename - a string storing name of the output file
   */
   virtual int readFromFile(char *filename) 
                            {(void) filename; return -1;}

   /** write the element data to a file 
       @param filename - a string storing name of the output file
   */
   virtual int writeToFile(char *filename) 
                          {(void) filename; return -1;}

};

#endif

