#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream.h>

#include <other/basicTypes.h>
#include <../isis-mv/GlobalIDArray.h>
#include <../isis-mv/RealArray.h>
#include <../isis-mv/IntArray.h>
#include <fei-hypre.h>

static FEI** linearSystems;
static int number_of_systems;
static HYPRE_LinSysCore *LSCore_=NULL;

/*============================================================================*/
/* utility function. This is not in the O-O interface. Must be called before
   any of the other functions to instantiate the linear systems objects. */
extern "C" void FEI_create(int numSystems, 
                                 MPI_Comm comm, 
                                 int masterRank){

    linearSystems = new FEI*[numSystems];
    if ( LSCore_ == NULL ) LSCore_ = new HYPRE_LinSysCore(comm);

    for (int i = 0; i < numSystems; i++) {
        linearSystems[i] = HYPRE_Builder::FEIBuilder(LSCore_,comm,masterRank);
    }
    number_of_systems = numSystems;

    return;
}

/*============================================================================*/
/* utility function, not in the O-O interface. Must be called at the end, so
   the linear systems can be destroyed (to avoid memory leaks).  */
extern "C" void FEI_destroy() {

    for(int i=0; i<number_of_systems; i++)
        delete linearSystems[i];

    delete [] linearSystems;
    LSCore_ = NULL;
    return;
}

/*============================================================================*/
/* per-solve-step initialization */
extern "C" int FEI_initSolveStep(int sysHandle, 
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
extern "C" int FEI_initFields(int sysHandle, 
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
extern "C" int FEI_beginInitElemBlock(int sysHandle, 
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
extern "C" int FEI_initElemSet(int sysHandle, 
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
extern "C" int FEI_endInitElemBlock(int sysHandle) {

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
extern "C" int FEI_beginInitNodeSets(int sysHandle, 
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
extern "C" int FEI_initSharedNodeSet(int sysHandle, 
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
extern "C" int FEI_initExtNodeSet(int sysHandle, 
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
extern "C" int FEI_endInitNodeSets(int sysHandle) {

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
extern "C" int FEI_beginInitCREqns(int sysHandle, 
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
extern "C" int FEI_initCRMult(int sysHandle, 
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
extern "C" int FEI_initCRPen(int sysHandle, 
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
extern "C" int FEI_endInitCREqns(int sysHandle) {

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
extern "C" int FEI_initComplete(int sysHandle) {

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
extern "C" int FEI_resetSystem(int sysHandle, 
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
extern "C" int FEI_beginLoadNodeSets(int sysHandle, 
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
extern "C" int FEI_loadBCSet(int sysHandle, 
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
extern "C" int FEI_endLoadNodeSets(int sysHandle) {

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
extern "C" int FEI_beginLoadElemBlock(int sysHandle, 
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
extern "C" int FEI_loadElemSet(int sysHandle, 
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
extern "C" int FEI_endLoadElemBlock(int sysHandle) {

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
extern "C" int FEI_beginLoadCREqns(int sysHandle, 
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
extern "C" int FEI_loadCRMult(int sysHandle, 
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
extern "C" int FEI_loadCRPen(int sysHandle, 
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
extern "C" int FEI_endLoadCREqns(int sysHandle) {

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
extern "C" int FEI_loadComplete(int sysHandle) {

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
extern "C" void FEI_parameters(int sysHandle, 
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
extern "C" int FEI_iterateToSolve(int sysHandle, int* status) {

    int err;
    if (sysHandle < number_of_systems) {
        err = (*linearSystems)[sysHandle].iterateToSolve(*status);
    }
    else {
        cout << "ERROR iterateToSolve: sysHandle out of range." << endl;
        abort();
    }
    return(err);
}

/*============================================================================*/
/* return nodal-based solution to FE analysis on a block-by-block basis */
extern "C" int FEI_getBlockNodeSolution(int sysHandle, 
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
extern "C" int FEI_getBlockFieldNodeSolution(int sysHandle, 
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
extern "C" int FEI_getBlockElemSolution(int sysHandle, 
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
extern "C" int FEI_getCRMultSolution(int sysHandle, 
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
extern "C" int FEI_getCRMultParam(int sysHandle, 
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
extern "C" int FEI_getCRMultSizes(int sysHandle, 
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
extern "C" int FEI_getBlockElemIDList(int sysHandle, 
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
extern "C" int FEI_getBlockNodeIDList(int sysHandle, 
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
extern "C" int FEI_getNumSolnParams(int sysHandle, 
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
extern "C" int FEI_getNumElemBlocks(int sysHandle) {

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
extern "C" int FEI_getNumBlockActNodes(int sysHandle, 
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
extern "C" int FEI_getNumBlockActEqns(int sysHandle, 
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
extern "C" int FEI_getNumNodesPerElement(int sysHandle, 
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
extern "C" int FEI_getNumEqnsPerElement(int sysHandle, 
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
extern "C" int FEI_getNumBlockElements(int sysHandle, GlobalID blockID) {

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
extern "C" int FEI_getNumBlockElemEqns(int sysHandle, GlobalID blockID) {

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

/*============================================================================*/
/*============================================================================*/
/*============================================================================*/
/* new functions                                                              */
/*============================================================================*/

/*============================================================================*/
/* load constraint numbers                                                    */
/*============================================================================*/
extern "C" int FEI_loadConstraintNumbers(int sysHandle, int leng, int *numbers)
{
    if (sysHandle == 0) LSCore_->loadConstraintNumbers(leng,numbers);
    else {
       cout << "ERROR loadConstraintNumbers: sysHandle should be 0." << endl;
       abort();
    }
    return(0);
}

/*============================================================================*/
/* load constraint numbers                                                    */
/*============================================================================*/
extern "C" int FEI_buildReducedSystem(int sysHandle)
{
    if (sysHandle == 0) LSCore_->buildReducedSystem();
    else {
       cout << "ERROR buildReducedSystem: sysHandle should be 0." << endl;
       abort();
    }
    return(0);
}

