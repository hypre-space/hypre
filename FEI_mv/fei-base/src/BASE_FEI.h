#ifndef __BASE_FEI_H
#define __BASE_FEI_H

//required headers:
// #ifdef FEI_SER
// #include <mpiuni/mpi.h>
// #else
// #include <mpi.h>
// #endif
//
// #include "other/basicTypes.h"
//
// #include "src/BCRecord.h"
// #include "src/FieldRecord.h"
// #include "src/BlockDescriptor.h"
// #include "src/MultConstRecord.h"
// #include "src/PenConstRecord.h"
// #include "src/NodeDescriptor.h"
// #include "src/NodeCommMgr.h"
// #include "src/EqnCommMgr.h"
//
// #include "src/Data.h"
// #include "src/LinearSystemCore.h"
 
class BASE_FEI {

  public:
    //Constructor
    BASE_FEI(MPI_Comm comm, LinearSystemCore* linSysCore, int masterRank=0);

    //Destructor
    virtual ~BASE_FEI();

// Structural initialization sequence.............................

    int setRHSID(int index) {linSysCore_->setRHSID(index); return(0);};

    // per-solve-step initialization
    int initSolveStep(int numElemBlocks, 
                      int solveType);    

    // per-solve-step initialization
    int initSolveStep(int numElemBlocks, 
                      int solveType,
                      int numMatrices,
                      int* matrixIDs,
                      int* numRHSs,
                      int** rhsIDs);    

    // identify all the solution fields present in the analysis
    int initFields(int numFields, 
                   const int *cardFields, 
                   const int *fieldIDs);

    // begin blocked-element initialization step..................
    int beginInitElemBlock(GlobalID elemBlockID,  
                           int numNodesPerElement, 
                           const int *numElemFields,
                           const int *const *elemFieldIDs,
                           int interleaveStrategy,
                           int lumpingStrategy,
                           int numElemDOF, 
                           int numElemSets,
                           int numElemTotal);                                    

    // initialize element sets that make up the blocks
    int initElemSet(int numElems, 
                    const GlobalID *elemIDs, 
                    const GlobalID *const *elemConn); 

    // end blocked-element initialization
    int endInitElemBlock();


    // begin collective node set initialization step........................
    int beginInitNodeSets(int numSharedNodeSets, 
                          int numExtNodeSets);

    // initialize nodal sets for shared nodes
    int initSharedNodeSet(const GlobalID *sharedNodeIDs,  
                          int lenSharedNodeIDs, 
                          const int *const *sharedProcIDs, 
                          const int *lenSharedProcIDs);

    // initialize nodal sets for external (off-proc) communication
    int initExtNodeSet(const GlobalID *extNodeIDs,  
                       int lenExtNodeIDs, 
                       const int *const *extProcIDs, 
                       const int *lenExtProcIDs);

    // end node set initialization
    int endInitNodeSets();


    // begin constraint relation set initialization step.........
    int beginInitCREqns(int numCRMultSets, 
                        int numCRPenSets);

    // constraint relation initialization
    // - lagrange multiplier formulation
    int initCRMult(const GlobalID *const *CRNodeTable,
                   const int *CRFieldList,
                   int numMultCRs, 
                   int lenCRNodeList,
                   int& CRMultID); 

    // constraint relation initialization
    // - penalty function formulation
    int initCRPen(const GlobalID *const *CRNodeTable, 
                  const int *CRFieldList,
                  int numPenCRs, 
                  int lenCRNodeList,
                  int& CRPenID); 

    // end constraint relation list initialization
    int endInitCREqns();

    // indicate that overall initialization sequence is complete
    int initComplete();

// FE data load sequence..........................................

    // set a value (usually zeros) throughout the linear system
    int resetSystem(double s);

    // begin node-set data load step.............................
    int beginLoadNodeSets(int numBCNodeSets);

    // boundary condition data load step
    int loadBCSet(const GlobalID *BCNodeSet,  
                  int lenBCNodeSet,  
                  int BCFieldID,
                  const double *const *alphaBCDataTable,  
                  const double *const *betaBCDataTable,  
                  const double *const *gammaBCDataTable);

    // end node-set data load step
    int endLoadNodeSets();


    // begin blocked-element data loading step....................
    int beginLoadElemBlock(GlobalID elemBlockID,
                           int numElemSets,
                           int numElemTotal);
  
    // elemSet-based stiffness/rhs data loading step
    int loadElemSet(int elemSetID, 
                    int numElems, 
                    const GlobalID *elemIDs,  
                    const GlobalID *const *elemConn,
                    const double *const *const *elemStiffness,
                    const double *const *elemLoad,
                    int elemFormat);

    // elemSet-based stiffness/rhs data loading step
    int loadElemSetMatrix(int elemSetID, 
                          int numElems, 
                          const GlobalID *elemIDs,  
                          const GlobalID *const *elemConn,
                          const double *const *const *elemStiffness,
                          int elemFormat);

    // elemSet-based stiffness/rhs data loading step
    int loadElemSetRHS(int elemSetID, 
                       int numElems, 
                       const GlobalID *elemIDs,  
                       const GlobalID *const *elemConn,
                       const double *const *elemLoad);

    // end blocked-element data loading step
    int endLoadElemBlock();


    // begin constraint relation data load step...................
    int beginLoadCREqns(int numCRMultSets, 
                        int numCRPenSets);                                

    // lagrange-multiplier constraint relation load step
    int loadCRMult(int CRMultID, 
                   int numMultCRs,
                   const GlobalID *const *CRNodeTable, 
                   const int *CRFieldList,
                   const double *const *CRWeightTable,
                   const double *CRValueList,
                   int lenCRNodeList);

    // penalty formulation constraint relation load step
    int loadCRPen(int CRPenID, 
                  int numPenCRs, 
                  const GlobalID *const *CRNodeTable,
                  const int *CRFieldList,
                  const double *const *CRWeightTable,
                  const double *CRValueList,
                  const double *penValues,
                  int lenCRNodeList);

    // end constraint relation data load step
    int endLoadCREqns();

    // indicate that overall data loading sequence is complete
    int loadComplete();

// Equation solution services.....................................

    // set scalars by which to multiply matrices.
    int setMatScalars(int* IDs,
                      double* scalars,
                      int numScalars);

    // set scalars by which to multiply RHS vectors.
    int setRHSScalars(int* IDs,
                      double* scalars,
                      int numScalars);

    // set parameters associated with solver choice, etc.
    void parameters(int numParams, char **paramStrings);

    // start iterative solution
    int iterateToSolve(int& status);

    // query function iterations performed.
    int iterations() const {return(iterations_);};

// Solution return services.......................................
 
    // return all nodal solution params on a block-by-block basis 
    int getBlockNodeSolution(GlobalID elemBlockID,  
                             GlobalID *nodeIDList, 
                             int &lenNodeIDList, 
                             int *offset,  
                             double *results);
 
    // return nodal solution for one field on a block-by-block basis 
    int getBlockFieldNodeSolution(GlobalID elemBlockID,
                                  int fieldID,
                                  GlobalID *nodeIDList, 
                                  int& lenNodeIDList, 
                                  int *offset,
                                  double *results);
         
    // return element solution params on a block-by-block basis 
    int getBlockElemSolution(GlobalID elemBlockID,  
                             GlobalID *elemIDList,
                             int& lenElemIDList, 
                             int *offset,  
                             double *results, 
                             int& numElemDOF);
                                      
    // return Lagrange solution to FE analysis on a constraint-set basis 
    int getCRMultParam(int CRMultID, 
                       int numMultCRs,
                       double *multValues);

    // return Lagrange solution to FE analysis on a whole-processor basis 
    int getCRMultSolution(int& numCRMultSets, 
                          int *CRMultIDs,  
                          int *offset, 
                          double *results);

 
// associated "puts" paralleling the solution return services.
// 
// the int sizing parameters are passed for error-checking purposes, so
// that the interface implementation can tell if the passed estimate
// vectors make sense -before- an attempt is made to utilize them as
// initial guesses by unpacking them into the solver's native solution
// vector format (these parameters include lenNodeIDList, lenElemIDList,
// numElemDOF, and numMultCRs -- all other passed params are either 
// vectors or block/constraint-set IDs)

    // put nodal-based solution guess on a block-by-block basis 
    int putBlockNodeSolution(GlobalID elemBlockID, 
                             const GlobalID *nodeIDList, 
                             int lenNodeIDList, 
                             const int *offset, 
                             const double *estimates);

    // put nodal-based guess for one field on a block-by-block basis 
    int putBlockFieldNodeSolution(GlobalID elemBlockID, 
                                  int fieldID, 
                                  const GlobalID *nodeIDList, 
                                  int lenNodeIDList, 
                                  const int *offset,
                                  const double *estimates);
         
    // put element-based solution guess on a block-by-block basis
    int putBlockElemSolution(GlobalID elemBlockID,  
                             const GlobalID *elemIDList, 
                             int lenElemIDList, 
                             const int *offset,  
                             const double *estimates, 
                             int numElemDOF);
  
    // put Lagrange solution to FE analysis on a constraint-set basis 
    int putCRMultParam(int CRMultID, 
                       int numMultCRs,
                       const double *multEstimates);


// utility functions that aid in integrating the FEI calls..............

// support methods for the "gets" and "puts" of the soln services.

    // return info associated with Lagrange multiplier solution
    int getCRMultSizes(int& numCRMultIDs, 
                       int& lenResults);

    // return info associated with blocked nodal solution
    int getBlockNodeIDList(GlobalID elemBlockID,
                           GlobalID *nodeIDList, 
                           int& lenNodeIDList);
                               
    // return info associated with blocked element solution
    int getBlockElemIDList(GlobalID elemBlockID, 
                           GlobalID *elemIDList, 
                           int& lenElemIDList);
 
// miscellaneous self-explanatory "read-only" utility functions............ 
 
    int getNumSolnParams(GlobalID globalNodeID) const;

    //  return the number of stored element blocks
    int getNumElemBlocks() const;

    //  return the number of active nodes in a given element block
    int getNumBlockActNodes(GlobalID blockID) const;

    //  return the number of active equations in a given element block
    int getNumBlockActEqns(GlobalID blockID) const;

    //  return the number of nodes associated with elements of a
    //  given block ID
    int getNumNodesPerElement(GlobalID blockID) const;
    
    //  return the number of equations (including element eqns)
    //  associated with elements of a given block ID
    int getNumEqnsPerElement(GlobalID blockID) const;

    //  return the number of elements associated with this blockID
    int getNumBlockElements(GlobalID blockID) const;

    //  return the number of elements eqns for elems w/ this blockID
    int getNumBlockElemEqns(GlobalID blockID) const;

//===== a couple of public non-FEI functions... ================================
//These are intended to be used by an 'outer-layer' class like FEI_ISIS.
//
  public:
    void unpackSolution();

    int getNodalScatterIndices(GlobalID* nodeIDs, int numNodes,
                               int *scatterIndices);

    void getLinearSystemCore(LinearSystemCore*& linSysCore) {
        linSysCore = linSysCore_;
    }

    void getProblemStructure(ProblemStructure*& probStruct) {
        probStruct = problemStructure_;
    }

    void setProblemStructure(ProblemStructure* probStruct);

    void setNumRHSVectors(int numRHSs, int* rhsIDs);

//==============================================================================
//private functions for internal implementation of BASE_FEI.
//==============================================================================
  private:

    void flagNodeAsActive(GlobalID nodeID);

    void setActiveNodeOwnerProcs();
    int countActiveNodeEqns();
    int calcTotalNumElemDOF();
    int setActiveNodeEqnInfo();
    void setElemDOFEqnInfo(int numNodalEqns);

    void getGlobalEqnInfo(int numLocalEqns, int& numGlobalEqns,
                          int& localStart, int& localEnd);

    void storeElementScatterIndices(int* indices, int numIndices);
    void storeMatrixPosition(int localRow, int col);

    void storeNodalColumnIndices(int eqn, NodeDescriptor& node, int fieldID);
    void storeNodalRowIndices(NodeDescriptor& node, int fieldID, int eqn);
    void storeNodalColumnCoefs(int eqn, NodeDescriptor& node, int fieldID,
                               double* coefs);
    void storeNodalRowCoefs(NodeDescriptor& node, int fieldID,
                               double* coefs, int eqn);

    void storeNodalSendIndex(NodeDescriptor& node, int fieldID, int col);
    void storeNodalSendEqn(NodeDescriptor& node, int fieldID, int col,
                           double* coefs);
    void storeNodalSendIndices(NodeDescriptor& iNode, int iField,
                               NodeDescriptor& jNode, int jField);

    void storePenNodeSendData(NodeDescriptor& iNode, int iField, double* iCoefs,
                              NodeDescriptor& jNode, int jField, double* jCoefs,
                              double penValue, double CRValue);

    void storeLocalNodeIndices(NodeDescriptor& iNode, int iField,
                               NodeDescriptor& jNode, int jField);

    void storePenNodeData(NodeDescriptor& iNode, int iField, double* iCoefs,
                          NodeDescriptor& jNode, int jField, double* jCoefs,
                          double penValue, double CRValue);

    void allocateWorksetChecks(int numElemBlocks);

    void setMatrixStructure();

    void handleMultCRStructure();
    void handlePenCRStructure();

    NodeDescriptor& findNodeDescriptor(GlobalID nodeID) const;

    void exchangeRemoteEquations();

    void copyStiffness(const double* const* elemStiff, int numRows,
                       int elemFormat, double** localStiffness);

    void getBCEqns(IntArray& essEqns, RealArray& essAlpha, RealArray& essGamma,
                   IntArray& otherEqns, RealArray& otherAlpha,
                   RealArray& otherBeta, RealArray& otherGamma);

    void implementAllBCs();

    void packSharedStiffness(const int* remoteEqnOffsets,
                             const int* remoteProcs,
                             int numRemoteEqns,
                             int* scatterIndices,
                             const double* const* stiffness,
                             int numIndices);

    void packSharedLoad(const int* remoteEqnOffsets,
                        const int* remoteProcs,
                        int numRemoteEqns,
                        int* scatterIndices,
                        const double* load);

    void assembleStiffnessAndLoad(int numNodes, int* scatterIndices, 
                                  const double* const* stiff,
                                  const double* load);

    void assembleStiffness(int numNodes, int* scatterIndices, 
                           const double* const* stiff);

    void assembleLoad(int numNodes, int* scatterIndices, 
                      const double* load);

    void debugOutput(char* mesg);

//==============================================================================
//private BASE_FEI variables
//==============================================================================
  private:

    LinearSystemCore* linSysCore_;

    int internalFei_;

    int solveCounter_;
    void setDebugOutput(char* path, char* name);
    char* debugFileName_;
    char* debugPath_;
    int debugOutput_;
    FILE *debugFile_;

    int numParams_;
    char **paramStrings_;

    int localStartRow_, localEndRow_, numLocalEqns_, numGlobalEqns_;

    int iterations_;
    int numRHSs_;
    int currentRHS_;
    int* rhsIDs_;

    int outputLevel_;

// wall clock time stuff
    double baseTime_, wTime_, sTime_;

    int storeNumCRMultRecords; // number of Lagrange constraint records

// lists of Lagrange constraints
    MultConstRecord *ceqn_MultConstraints; 

//    communications layer

    MPI_Comm comm_;

    int masterRank_;
    int localRank_;
    int numProcs_;

//-------------------------------------------------------------------
//  control parameters (can be considered local to each processor)
//-------------------------------------------------------------------

//  non-volatile control data

    int storeNumProcActNodes;  // number of active nodes on this processor
    int storeNumProcActEqns;   // number of equations arising from active nodes
    int storeBCNodeSets;       // number of bc node sets
    int storeSharedNodeSets;   // number of shared node sets
    int storeExtNodeSets;      // number of external node sets
    int storeNumCRPenRecords;  // number of penalty constraint records

    int *numWorksets;             // number of worksets for each block
    int *numWorksetsStored;       // number of worksets stored for each block
    int *numMatrixWorksetsStored; // number of worksets stored for each block
    int *numRHSWorksetsStored;    // number of worksets stored for each block
    int *nextElemIndex;

//  some data for consistency checking (some of these not yet utilized)

    int checkElemBlocksLoaded;
    int checkNumElemBlocks;    // verify number of blocks in this problem
    int checkNumProcActNodes;  // verify number of active nodes on this proc
    int checkNumProcActEqns;   // verify number of equations
                               // arising from active nodes
    int checkBCNodeSets;       // verify number of bc node sets
    int checkSharedNodeSets;   // verify number of shared node sets
    int checkExtNodeSets;      // verify number of external node sets
    int checkNumCRMultRecords; // verify number of Lagrange constraint relations
    int checkNumCRPenRecords;  // verify number of penalty constraint relations
    int doneEndInitElemData;   //non-zero indicates that doEndInitElemData
                               //has been called

//  volatile control data

    GlobalID currentElemBlockID;         // current block ID counter
    int currentElemBlockIndex;           // current block index
    int currentWorkSetID;                // current workset ID counter
    BlockDescriptor* currentBlock;

    ProblemStructure* problemStructure_;
    bool problemStructureAllocated_;
    bool problemStructureSet_;
    bool matrixAllocated_;

    BCManager* bcManager_; //Boundary condition manager

// lists of Penalty constraints

    PenConstRecord *ceqn_PenConstraints;     

    int currentSharedNodeSet;
    int currentExtNodeSet;
};

#endif

