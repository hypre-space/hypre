#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream.h>

/* #### added by Charles to ensure HYPRE compatibility ### */
#include "utilities.h"
/*#include <fei-isis.h>*/
/* ########################################################### */

// include matrix-vector files from ISIS++

#include "other/basicTypes.h"
#include "RealArray.h"
#include "IntArray.h"
#include "GlobalIDArray.h"
#include "CommInfo.h"
#include "Map.h"
#include "Vector.h"
#include "Matrix.h"

// include the Hypre package header here

#include "HYPRE_IJ_mv.h"
//#include "parcsr_matrix_vector.h"
//#include "csr_matrix.h"
//#include "par_csr_matrix.h"

#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"

// includes needed in order to be able to include BASE_SLE.h

#include "other/basicTypes.h"
#include "fei.h"
#include "src/CommBufferDouble.h"
#include "src/CommBufferInt.h"
#include "src/NodePackets.h"
#include "src/BCRecord.h"
#include "src/FieldRecord.h"
#include "src/BlockRecord.h"
#include "src/MultConstRecord.h"
#include "src/PenConstRecord.h"
#include "src/SimpleList.h"
#include "src/NodeRecord.h"
#include "src/SharedNodeRecord.h"
#include "src/SharedNodeBuffer.h"
#include "src/ExternNodeRecord.h"
#include "src/SLE_utils.h"
#include "BASE_SLE.h"

#include "HYPRE_SLE.h"

static HYPRE_SLE **linearSystems;

static int number_of_systems;

/*============================================================================*/
/* utility function. This is not in the O-O interface. Must be called before
   any of the other functions to instantiate the linear systems objects. */
extern "C" void numLinearSystems(int numSystems, 
                                 MPI_Comm FEI_COMM_WORLD, 
                                 int masterRank){

    linearSystems = new HYPRE_SLE*[numSystems];
    for (int i = 0; i < numSystems; i++) {
        linearSystems[i] = new HYPRE_SLE(FEI_COMM_WORLD, masterRank);
    }
    number_of_systems = numSystems;

    return;
}

/*============================================================================*/
/* utility function, not in the O-O interface. Must be called at the end, so
   the linear systems can be destroyed (to avoid memory leaks).  */
extern "C" void destroyAllLinearSystems() {

    for(int i=0; i<number_of_systems; i++)
        delete linearSystems[i];

    delete [] linearSystems;

    return;
}

/*============================================================================*/
/* per-solve-step initialization */
extern "C" int initSolveStep(int sysHandle, 
                             int numElemBlocks, 
                             int solveType)
{

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].initSolveStep(numElemBlocks, 
                                                  solveType);
    }
    else {
        cout << "ERROR initSolveStep: sysHandle " << sysHandle <<
        		" is out of range " << number_of_systems << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* identify all the solution fields present in the analysis */
extern "C" int initFields(int sysHandle, 
                          int numFields, 
                          int *cardFields, 
                          int *fieldIDs)
{

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].initFields(numFields, 
                                               cardFields,
                                               fieldIDs);
    }
    else {
        cout << "ERROR initFields: sysHandle " << sysHandle <<
        		" is out of range " << number_of_systems << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* begin blocked-element initialization step */
extern "C" int beginInitElemBlock(int sysHandle, 
                                  GlobalID elemBlockID,
                                  int numNodesPerElement, 
                                  int *numElemFields,
                                  int **elemFieldIDs,
                                  int interleaveStrategy,
                                  int lumpingStrategy,
                                  int numElemDOF, 
                                  int numElemSets,
                                  int numElemTotal) {
 
    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].beginInitElemBlock(elemBlockID,
                                                       numNodesPerElement, 
                                                       numElemFields, 
                                                       elemFieldIDs,
                                                       interleaveStrategy, 
                                                       lumpingStrategy, 
                                                       numElemDOF, 
                                                       numElemSets, 
                                                       numElemTotal);
    }
    else {
        cout << "ERROR beginInitElemBlock: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* initialize element sets that make up the blocks */
extern "C" int initElemSet(int sysHandle, 
                           int numElems,
                           GlobalID *elemIDs, 
                           GlobalID **elemConn) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].initElemSet(numElems, 
                                                elemIDs, 
                                                elemConn);
    }
    else {
        cout << "ERROR initElemSet: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* end blocked-element initialization */
extern "C" int endInitElemBlock(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].endInitElemBlock();
    }
    else {
        cout << "ERROR endInitElemBlock: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* begin collective node set initialization step */
extern "C" int beginInitNodeSets(int sysHandle, 
                                 int numSharedNodeSets, 
                                 int numExtNodeSets) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].beginInitNodeSets(numSharedNodeSets, 
                                                      numExtNodeSets);
    }
    else {
        cout << "ERROR beginInitNodeSets: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* initialize nodal sets for shared nodes */
extern "C" int initSharedNodeSet(int sysHandle, 
                                 GlobalID *sharedNodeIDs,
                                 int lenSharedNodeIDs, 
                                 int **sharedProcIDs,
                                 int *lenSharedProcIDs) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].initSharedNodeSet(sharedNodeIDs,
                                                      lenSharedNodeIDs, 
                                                      sharedProcIDs, 
                                                      lenSharedProcIDs);
    }
    else {
        cout << "ERROR initSharedNodeSet: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* initialize nodal sets for external (off-processor) communcation */
extern "C" int initExtNodeSet(int sysHandle, 
                              GlobalID *extNodeIDs,
                              int lenExtNodeIDs, 
                              int **extProcIDs, 
                              int *lenExtProcIDs) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].initExtNodeSet(extNodeIDs,
                                                   lenExtNodeIDs, 
                                                   extProcIDs, 
                                                   lenExtProcIDs);
    }
    else {
        cout << "ERROR initExtNodeSet: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* end node set initialization */
extern "C" int endInitNodeSets(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].endInitNodeSets();
    }
    else {
        cout << "ERROR endInitNodeSets: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* begin constraint relation set initialization step */
extern "C" int beginInitCREqns(int sysHandle, 
                               int numCRMultSets,
                               int numCRPenSets) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].beginInitCREqns(numCRMultSets, 
                                                    numCRPenSets);
    }
    else {
        cout << "ERROR beginInitCREqns: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* constraint relation initialization - lagrange multiplier formulation */
extern "C" int initCRMult(int sysHandle, 
                          GlobalID **CRNodeTable,  
                          int *CRFieldList,
		                  int numMultCRs, 
		                  int lenCRNodeList,
		                  int* CRMultID) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].initCRMult(CRNodeTable,
                                               CRFieldList, 
                                               numMultCRs, 
                                               lenCRNodeList,
                                               *CRMultID);
    }
    else {
        cout << "ERROR initCRMult: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* constraint relation initialization - penalty function formulation */
extern "C" int initCRPen(int sysHandle, 
                         GlobalID **CRNodeTable, 
                         int *CRFieldList,
                         int numPenCRs, 
                         int lenCRNodeList, 
                         int* CRPenID) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].initCRPen(CRNodeTable,
                                              CRFieldList, 
                                              numPenCRs, 
                                              lenCRNodeList, 
                                              *CRPenID);
    }
    else {
        cout << "ERROR initCRPen: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* end constraint relation list initialization */
extern "C" int endInitCREqns(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].endInitCREqns();
    }
    else {
        cout << "ERROR endInitCREqns: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* indicate that overall initialization sequence is complete */
extern "C" int initComplete(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].initComplete();
    }
    else {
        cout << "ERROR initComplete: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* set a value (usually zeros) througout the linear system.....*/
extern "C" int resetSystem(int sysHandle, 
                           double s) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].resetSystem(s);
    }
    else {
        cout << "ERROR resetSystem: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* begin node-set data load step */
extern "C" int beginLoadNodeSets(int sysHandle, 
                                 int numBCNodeSets) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].beginLoadNodeSets(numBCNodeSets);
    }
    else {
        cout << "ERROR beginLoadNodeSets: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* boundary condition data load step */
extern "C" int loadBCSet(int sysHandle, 
                         GlobalID *BCNodeSet,
                         int lenBCNodeSet,
                         int BCFieldID,
                         double **alphaBCDataTable,
                         double **betaBCDataTable,
                         double **gammaBCDataTable) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].loadBCSet(BCNodeSet, 
                                              lenBCNodeSet,
                                              BCFieldID,
                                              alphaBCDataTable, 
                                              betaBCDataTable, 
                                              gammaBCDataTable);
    }
    else {
        cout << "ERROR loadBCSet: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* end node-set data load step */
extern "C" int endLoadNodeSets(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].endLoadNodeSets();
    }
    else {
        cout << "ERROR endLoadNodeSets: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* begin blocked-element data loading step */
extern "C" int beginLoadElemBlock(int sysHandle, 
                                  GlobalID elemBlockID,
                                  int numElemSets,
                                  int numElemTotal) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].beginLoadElemBlock(elemBlockID,
                                                       numElemSets,
                                                       numElemTotal);
    }
    else {
        cout << "ERROR beginLoadElemBlock: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* elemSet-based stiffness/rhs data loading step */
extern "C" int loadElemSet(int sysHandle, 
                           int elemSetID, 
                           int numElems, 
                           GlobalID *elemIDs,  
                           GlobalID **elemConn,
                           double ***elemStiffness,
                           double **elemLoad,
                           int elemFormat) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].loadElemSet(elemSetID, 
                                                numElems, 
                                                elemIDs,  
                                                elemConn,
                                                elemStiffness,
                                                elemLoad,
                                                elemFormat);
    }
    else {
        cout << "ERROR loadElemSet: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* end blocked-element data loading step */
extern "C" int endLoadElemBlock(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].endLoadElemBlock();
    }
    else {
        cout << "ERROR endLoadElemBlock: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* begin constraint relation data load step */
extern "C" int beginLoadCREqns(int sysHandle, 
                               int numCRMultSets,
                               int numCRPenSets) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].beginLoadCREqns(numCRMultSets, 
                                                    numCRPenSets);
    }
    else {
        cout << "ERROR beginLoadCREqns: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* lagrange-multiplier constraint relation load step */
extern "C" int loadCRMult(int sysHandle, 
                          int CRMultID, 
                          int numMultCRs,
                          GlobalID **CRNodeTable,  
                          int *CRFieldList,
                          double **CRWeightTable,
                          double *CRValueList,
                          int lenCRNodeList) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].loadCRMult(CRMultID, 
                                               numMultCRs,
                                               CRNodeTable,  
                                               CRFieldList,
                                               CRWeightTable, 
                                               CRValueList,
                                               lenCRNodeList);
    }
    else {
        cout << "ERROR loadCRMult: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* penalty formulation constraint relation load step */
extern "C" int loadCRPen(int sysHandle, 
                         int CRPenID,
                         int numPenCRs, 
                         GlobalID **CRNodeTable,
                         int *CRFieldList,
                         double **CRWeightTable,  
                         double *CRValueList,
                         double *penValues,
                         int lenCRNodeList) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].loadCRPen(CRPenID, 
                                              numPenCRs,
                                              CRNodeTable,  
                                              CRFieldList,
                                              CRWeightTable, 
                                              CRValueList, 
                                              penValues,
                                              lenCRNodeList);
    }
    else {
        cout << "ERROR loadCRPen: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* end constraint relation data load step */
extern "C" int endLoadCREqns(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].endLoadCREqns();
    }
    else {
        cout << "ERROR endLoadCREqns: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* indicate that overall data loading sequence is complete */
extern "C" int loadComplete(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].loadComplete();
    }
    else {
        cout << "ERROR loadComplete: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* set parameters associated with solver choice, etc. */
extern "C" void parameters(int sysHandle, 
                           int numParams, 
                           char **paramStrings) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].parameters(numParams, 
                                               paramStrings);
    }
    else {
        cout << "ERROR parameters: sysHandle out of range." << endl;
        abort();
    }
    return;
}
             
/*============================================================================*/
/* start iterative solution */
extern "C" int iterateToSolve(int sysHandle) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].iterateToSolve();
    }
    else {
        cout << "ERROR iterateToSolve: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* return nodal-based solution to FE analysis on a block-by-block basis */
extern "C" int getBlockNodeSolution(int sysHandle, 
                                    GlobalID elemBlockID,
                                    GlobalID *nodeIDList, 
                                    int* lenNodeIDList, 
                                    int *offset,
                                    double *results) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].getBlockNodeSolution(elemBlockID,
                                                         nodeIDList, 
                                                         *lenNodeIDList, 
                                                         offset, 
                                                         results);
    }
    else {
        cout << "ERROR getBlockNodeSolution: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* return nodal-based solution to FE analysis on a block-by-block basis */
extern "C" int getBlockFieldNodeSolution(int sysHandle, 
                                         GlobalID elemBlockID,
                                         int fieldID,
                                         GlobalID *nodeIDList, 
                                         int* lenNodeIDList, 
                                         int *offset,
                                         double *results) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].getBlockFieldNodeSolution(elemBlockID,
                                                              fieldID, 
                                                              nodeIDList, 
                                                              *lenNodeIDList, 
                                                              offset, 
                                                              results);
    }
    else {
        cout << "ERROR getBlockNodeSolution: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* return Lagrange solution to FE analysis on a whole-processor basis */
extern "C" int getBlockElemSolution(int sysHandle, 
                                    GlobalID elemBlockID,
                                    GlobalID *elemIDList, 
                                    int* lenElemIDList, 
                                    int *offset,
                                    double *results, 
                                    int* numElemDOF) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].getBlockElemSolution(elemBlockID,
                                                         elemIDList, 
                                                         *lenElemIDList, 
                                                         offset, results, 
                                                         *numElemDOF);
    }
    else {
        cout << "ERROR getBlockElemSolution: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* return Lagrange solution to FE analysis on a whole-processor basis */
extern "C" int getCRMultSolution(int sysHandle, 
                                 int* numCRMultSets, 
                                 int *CRMultIDs,
                                 int *offset, 
                                 double *results) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].getCRMultSolution(*numCRMultSets,
                                                      CRMultIDs, 
                                                      offset, 
                                                      results);
    }
    else {
        cout << "ERROR getCRMultSolution: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* return Lagrange solution to FE analysis on a constraint-set basis */
extern "C" int getCRMultParam(int sysHandle, 
                              int CRMultID,
                              int numMultCRs, 
                              double *multValues) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].getCRMultParam(CRMultID,
                                                   numMultCRs, 
                                                   multValues);
    }
    else {
        cout << "ERROR getCRMultParam: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}
 
/*============================================================================*/
/* return sizes associated with Lagrange solution to FE analysis */
extern "C" int getCRMultSizes(int sysHandle, 
                              int* numCRMultIDs, 
                              int* lenResults) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].getCRMultSizes(*numCRMultIDs, 
                                                   *lenResults);
    }
    else {
        cout << "ERROR getCRMultSizes: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* form some data structures needed to return element solution parameters */
extern "C" int getBlockElemIDList(int sysHandle, 
                                  GlobalID elemBlockID, 
	                              GlobalID *elemIDList, 
	                              int *lenElemIDList) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].getBlockElemIDList(elemBlockID,
                                                       elemIDList, 
                                                       *lenElemIDList);
    }
    else {
        cout << "ERROR getBlockElemIDList: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/* form some data structures needed to return nodal solution parameters */
extern "C" int getBlockNodeIDList(int sysHandle, 
                                  GlobalID elemBlockID, 
	                              GlobalID *nodeIDList, 
	                              int *lenNodeIDList) {

    if (sysHandle < number_of_systems) {
        (*linearSystems)[sysHandle].getBlockNodeIDList(elemBlockID,
                                                       nodeIDList, 
                                                       *lenNodeIDList);
    }
    else {
        cout << "ERROR getBlockNodeIDList: sysHandle out of range." << endl;
        abort();
    }
    return(0);
}

/*============================================================================*/
/*  return the number of solution parameters at a given node */
extern "C" int getNumSolnParams(int sysHandle, 
                                GlobalID globalNodeID) {

    int numSolnParams = -1;
    if (sysHandle < number_of_systems) {
        numSolnParams = (*linearSystems)[sysHandle].getNumSolnParams(globalNodeID);
    }
    else {
        cout << "ERROR getNumSolnParams: sysHandle out of range." << endl;
        abort();
    }
    return(numSolnParams);
}

/*============================================================================*/
/*  return the number of stored element blocks */
extern "C" int getNumElemBlocks(int sysHandle) {

    int numElemBlocks = -1;
    if (sysHandle < number_of_systems) {
        numElemBlocks = (*linearSystems)[sysHandle].getNumElemBlocks();
    }
    else {
        cout << "ERROR getNumElemBlocks: sysHandle out of range." << endl;
        abort();
    }
    return(numElemBlocks);
}

/*============================================================================*/
/*  return the number of active nodes in a given element block */
extern "C" int getNumBlockActNodes(int sysHandle, 
                                   int blockID) {

    int numBlockActNodes = -1;
    if (sysHandle < number_of_systems) {
        numBlockActNodes = (*linearSystems)[sysHandle].getNumBlockActNodes(blockID);
    }
    else {
        cout << "ERROR getNumBlockActNodes: sysHandle out of range." << endl;
        abort();
    }
    return(numBlockActNodes);
}

/*============================================================================*/
/*  return the number of active equations in a given element block */
extern "C" int getNumBlockActEqns(int sysHandle, 
                                  int blockID) {

    int numBlockActEqns = -1;
    if (sysHandle < number_of_systems) {
        numBlockActEqns = (*linearSystems)[sysHandle].getNumBlockActEqns(blockID);
    }
    else {
        cout << "ERROR getNumBlockActEqns: sysHandle out of range." << endl;
        abort();
    }
    return(numBlockActEqns);
}
  
/*============================================================================*/
/*  return the number of nodes associated with elements of a
    given block ID */
extern "C" int getNumNodesPerElement(int sysHandle, 
                                     GlobalID blockID) {

    int numNodesPerElement = -1;
    if (sysHandle < number_of_systems) {
        numNodesPerElement = (*linearSystems)[sysHandle].getNumNodesPerElement(blockID);
    }
    else {
        cout << "ERROR getNumNodesPerElement: sysHandle out of range." << endl;
        abort();
    }
    return(numNodesPerElement);
}

/*============================================================================*/
/*  return the number of eqns associated with elements of a
    given block ID */
extern "C" int getNumEqnsPerElement(int sysHandle, 
                                    GlobalID blockID) {

    int numEqnsPerElement = -1;
    if (sysHandle < number_of_systems) {
        numEqnsPerElement = (*linearSystems)[sysHandle].getNumEqnsPerElement(blockID);
    }
    else {
        cout << "ERROR getNumEqnsPerElement: sysHandle out of range." << endl;
        abort();
    }
    return(numEqnsPerElement);
}
 
/*============================================================================*/
/*  return the number of elements in a given element block */
extern "C" int getNumBlockElements(int sysHandle, GlobalID blockID) {

    int numBlockElements = -1;
    if (sysHandle < number_of_systems) {
        numBlockElements = 
                  (*linearSystems)[sysHandle].getNumBlockElements(blockID);
    }
    else {
        cout << "ERROR getNumBlockElements: sysHandle out of range." << endl;
        abort();
    }
    return(numBlockElements);
}

/*============================================================================*/
/*  return the number of element equations in a given element block */
extern "C" int getNumBlockElemEqns(int sysHandle, GlobalID blockID) {

    int numBlockElemEqns = -1;
    if (sysHandle < number_of_systems) {
        numBlockElemEqns = 
                  (*linearSystems)[sysHandle].getNumBlockElemEqns(blockID);
    }
    else {
        cout << "ERROR getNumBlockElemEqns: sysHandle out of range." << endl;
        abort();
    }
    return(numBlockElemEqns);
}

