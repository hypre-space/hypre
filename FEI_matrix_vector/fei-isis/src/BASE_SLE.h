#ifndef __BASE_SLE_H
#define __BASE_SLE_H

// #include "fei.h"
// #include "FieldRecord.h"
// #include "BlockRecord.h"
// #include "MultConstRecord.h"
// #include "PenConstRecord.h"
// #include "NodeRecord.h"
// #include "SharedNodeRecord.h"
// #include "SharedNodeBuffer.h"
// #include "ExternNodeRecord.h"

// #include "Map.h"
// #include "RowMatrix.h"
// #include "Vector.h"
// #include "IterativeSolver.h"
// #include "Preconditioner.h"

#ifndef BASE_SLE_ser

// #include "mpi.h"
// #include "NodePackets.h"

#endif

class BASE_SLE : public FEI {

  public:
    //Constructor
    BASE_SLE(MPI_Comm PASSED_COMM_WORLD, int masterRank=0);

    //Destructor
    virtual ~BASE_SLE();

// Structural initialization sequence.............................

    virtual int setMatrixID(int index) {(void)index; return(0);};

    virtual int setRHSID(int index) {(void)index; return(0);};

    // per-solve-step initialization
    virtual int initSolveStep(int numElemBlocks, 
                              int solveType);    

    // per-solve-step initialization
    virtual int initSolveStep(int numElemBlocks, 
                              int solveType,
                              int numMatrices,
                              int* matrixIDs,
                              int* numRHSs,
                              int** rhsIDs);    

    // identify all the solution fields present in the analysis
    virtual int initFields(int numFields, 
                           const int *cardFields, 
                           const int *fieldIDs);

    // begin blocked-element initialization step..................
    virtual int beginInitElemBlock(GlobalID elemBlockID,  
                                   int numNodesPerElement, 
                                   const int *numElemFields,
                                   const int *const *elemFieldIDs,
                                   int interleaveStrategy,
                                   int lumpingStrategy,
                                   int numElemDOF, 
                                   int numElemSets,
                                   int numElemTotal);                                    

    // initialize element sets that make up the blocks
    virtual int initElemSet(int numElems, 
                            const GlobalID *elemIDs, 
                            const GlobalID *const *elemConn); 

    // end blocked-element initialization
    virtual int endInitElemBlock();


    // begin collective node set initialization step........................
    virtual int beginInitNodeSets(int numSharedNodeSets, 
                                  int numExtNodeSets);

    // initialize nodal sets for shared nodes
    virtual int initSharedNodeSet(const GlobalID *sharedNodeIDs,  
                                  int lenSharedNodeIDs, 
                                  const int *const *sharedProcIDs, 
                                  const int *lenSharedProcIDs);

    // initialize nodal sets for external (off-proc) communication
    virtual int initExtNodeSet(const GlobalID *extNodeIDs,  
                               int lenExtNodeIDs, 
                               const int *const *extProcIDs, 
                               const int *lenExtProcIDs);

    // end node set initialization
    virtual int endInitNodeSets();


    // begin constraint relation set initialization step.........
    virtual int beginInitCREqns(int numCRMultSets, 
                                int numCRPenSets);

    // constraint relation initialization
    // - lagrange multiplier formulation
    virtual int initCRMult(const GlobalID *const *CRNodeTable,
                           const int *CRFieldList,
                           int numMultCRs, 
                           int lenCRNodeList,
                           int& CRMultID); 

    // constraint relation initialization
    // - penalty function formulation
    virtual int initCRPen(const GlobalID *const *CRNodeTable, 
                          const int *CRFieldList,
                          int numPenCRs, 
                          int lenCRNodeList,
                          int& CRPenID); 

    // end constraint relation list initialization
    virtual int endInitCREqns();

    // indicate that overall initialization sequence is complete
    virtual int initComplete();

// FE data load sequence..........................................

    // set a value (usually zeros) throughout the linear system
    virtual int resetSystem(double s);

    // begin node-set data load step.............................
    virtual int beginLoadNodeSets(int numBCNodeSets);

    // boundary condition data load step
    virtual int loadBCSet(const GlobalID *BCNodeSet,  
                          int lenBCNodeSet,  
                          int BCFieldID,
                          const double *const *alphaBCDataTable,  
                          const double *const *betaBCDataTable,  
                          const double *const *gammaBCDataTable);

    // end node-set data load step
    virtual int endLoadNodeSets();


    // begin blocked-element data loading step....................
    virtual int beginLoadElemBlock(GlobalID elemBlockID,
                                   int numElemSets,
                                   int numElemTotal);
  
    // elemSet-based stiffness/rhs data loading step
    virtual int loadElemSet(int elemSetID, 
                            int numElems, 
                            const GlobalID *elemIDs,  
                            const GlobalID *const *elemConn,
                            const double *const *const *elemStiffness,
                            const double *const *elemLoad,
                            int elemFormat);

    // elemSet-based stiffness/rhs data loading step
    virtual int loadElemSetMatrix(int elemSetID, 
                            int numElems, 
                            const GlobalID *elemIDs,  
                            const GlobalID *const *elemConn,
                            const double *const *const *elemStiffness,
                            int elemFormat);

    // elemSet-based stiffness/rhs data loading step
    virtual int loadElemSetRHS(int elemSetID, 
                            int numElems, 
                            const GlobalID *elemIDs,  
                            const GlobalID *const *elemConn,
                            const double *const *elemLoad);

    // element-wise transfer operator loading.
    virtual int loadElemSetTransfers(int elemSetID,
                                     int numElems,
                                     GlobalID** coarseNodeLists,
                                     GlobalID** fineNodeLists,
                                     int fineNodesPerCoarseElem,
                                     double*** elemProlong,
                                     double*** elemRestrict);

    // end blocked-element data loading step
    virtual int endLoadElemBlock();


    // begin constraint relation data load step...................
    virtual int beginLoadCREqns(int numCRMultSets, 
                                int numCRPenSets);                                

    // lagrange-multiplier constraint relation load step
    virtual int loadCRMult(int CRMultID, 
                           int numMultCRs,
                           const GlobalID *const *CRNodeTable, 
                           const int *CRFieldList,
                           const double *const *CRWeightTable,
                           const double *CRValueList,
                           int lenCRNodeList);

    // penalty formulation constraint relation load step
    virtual int loadCRPen(int CRPenID, 
                          int numPenCRs, 
                          const GlobalID *const *CRNodeTable,
                          const int *CRFieldList,
                          const double *const *CRWeightTable,
                          const double *CRValueList,
                          const double *penValues,
                          int lenCRNodeList);

    // end constraint relation data load step
    virtual int endLoadCREqns();

    // indicate that overall data loading sequence is complete
    virtual int loadComplete();

// Equation solution services.....................................

    // set scalars by which to multiply matrices.
    virtual int setMatScalars(int* IDs,
                              double* scalars,
                              int numScalars);

    // set scalars by which to multiply RHS vectors.
    virtual int setRHSScalars(int* IDs,
                              double* scalars,
                              int numScalars);

    // set parameters associated with solver choice, etc.
    virtual void parameters(int numParams, char **paramStrings);

    // start iterative solution
    virtual int iterateToSolve();

    // query function iterations performed.
    int iterations() const {return(iterations_);};

// Solution return services.......................................
 
    // return all nodal solution params on a block-by-block basis 
    virtual int getBlockNodeSolution(GlobalID elemBlockID,  
                                     GlobalID *nodeIDList, 
                                     int &lenNodeIDList, 
                                     int *offset,  
                                     double *results);
 
    // return nodal solution for one field on a block-by-block basis 
    virtual int getBlockFieldNodeSolution(GlobalID elemBlockID,
                                          int fieldID,
                                          GlobalID *nodeIDList, 
                                          int& lenNodeIDList, 
                                          int *offset,
                                          double *results);
         
    // return element solution params on a block-by-block basis 
    virtual int getBlockElemSolution(GlobalID elemBlockID,  
                                     GlobalID *elemIDList,
                                     int& lenElemIDList, 
                                     int *offset,  
                                     double *results, 
                                     int& numElemDOF);
                                      
    // return Lagrange solution to FE analysis on a constraint-set basis 
    virtual int getCRMultParam(int CRMultID, 
                               int numMultCRs,
                               double *multValues);

    // return Lagrange solution to FE analysis on a whole-processor basis 
    virtual int getCRMultSolution(int& numCRMultSets, 
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
    virtual int putBlockNodeSolution(GlobalID elemBlockID, 
                                     const GlobalID *nodeIDList, 
                                     int lenNodeIDList, 
                                     const int *offset, 
                                     const double *estimates);

    // put nodal-based guess for one field on a block-by-block basis 
    virtual int putBlockFieldNodeSolution(GlobalID elemBlockID, 
                                          int fieldID, 
                                          const GlobalID *nodeIDList, 
                                          int lenNodeIDList, 
                                          const int *offset,
                                          const double *estimates);
         
    // put element-based solution guess on a block-by-block basis
    virtual int putBlockElemSolution(GlobalID elemBlockID,  
                                     const GlobalID *elemIDList, 
                                     int lenElemIDList, 
                                     const int *offset,  
                                     const double *estimates, 
                                     int numElemDOF);
  
    // put Lagrange solution to FE analysis on a constraint-set basis 
    virtual int putCRMultParam(int CRMultID, 
                               int numMultCRs,
                               const double *multEstimates);


// utility functions that aid in integrating the FEI calls..............

// support methods for the "gets" and "puts" of the soln services.

    // return info associated with Lagrange multiplier solution
    virtual int getCRMultSizes(int& numCRMultIDs, 
                               int& lenResults);

    // return info associated with blocked nodal solution
    virtual int getBlockNodeIDList(GlobalID elemBlockID,
                                   GlobalID *nodeIDList, 
                                   int& lenNodeIDList);
                               
    // return info associated with blocked element solution
    virtual int getBlockElemIDList(GlobalID elemBlockID, 
                                   GlobalID *elemIDList, 
                                   int& lenElemIDList);
 
// miscellaneous self-explanatory "read-only" utility functions............ 
 
    virtual int getNumSolnParams(GlobalID globalNodeID) const;

    //  return the number of stored element blocks
    virtual int getNumElemBlocks() const;

    //  return the number of active nodes in a given element block
    virtual int getNumBlockActNodes(GlobalID blockID) const;

    //  return the number of active equations in a given element block
    virtual int getNumBlockActEqns(GlobalID blockID) const;

    //  return the number of nodes associated with elements of a
    //  given block ID
    virtual int getNumNodesPerElement(GlobalID blockID) const;
    
    //  return the number of equations (including element eqns)
    //  associated with elements of a given block ID
    virtual int getNumEqnsPerElement(GlobalID blockID) const;

    //  return the number of elements associated with this blockID
    virtual int getNumBlockElements(GlobalID blockID) const;

    //  return the number of elements eqns for elems w/ this blockID
    virtual int getNumBlockElemEqns(GlobalID blockID) const;

//===== a couple of public non-FEI functions... ================================
//These are intended to be used by an 'outer-layer' class like FEI_ISIS.
//
  public:
    void unpackSolution();

    int getNodalScatterIndices(GlobalID* nodeIDs, int numNodes,
                               int *scatterIndices);

    virtual void getSolnVectorPtr(Vector** vec) = 0;

    virtual void getMatrixPtr(Matrix** mat) = 0;
    virtual void getRHSVectorPtr(Vector** vec) = 0;
    virtual void setRHSIndex(int index) = 0;

  private:

//==============================================================================
//below are the private functions for
//internal implementation of the base FEI.
//==============================================================================

    //a function to do internal work at end of elem init stage
    int doEndInitElemData();

    //a function to do initializing communications work for the shared node
    //structures
    void doSharedNodeInitComm();
    void sharedNodeLaunchInitRecvs();

    void getMaxNodesAndDOF(int &nodes, int &DOF);

    //a function to do internal work at end of multCR init stage
    int doCRInit();

    int GlobalToLocalNode(GlobalID globalNodeID) const;
    void SetActive(GlobalID nodeNum, int nodeDOF);
    int getNumActNodes() const;
    int getActNodeList(NodeRecord *actNodeIDs);
    int getNumActNodalEqns() const;
    void populateExtNodeTables();

    void setupMPITypes();

    void exchangeSharedMatrixData();
    void exchangeSharedRHSData(int rhsIndex);

    int getNodeDOF(GlobalID nodeID);
    int getNodeEqnNumber(GlobalID nodeID);
    int getGlobalActEqnNumber(GlobalID myNode, int nodeIndex);
    int getGlobalActEqnDOF(GlobalID myNode);

    int getFieldRosterIndex(int fieldID);
    int getFieldCardinality(int fieldID);
    void buildNodalFieldLists();
    void calculateLocalNodeIndices(const GlobalID* const elemConn,
                                   int numNodes, int* localIndices);

    void copyStiffness(const double* const* elemStiff, int numRows,
                       int elemFormat, double** localStiffness);

    void implementAllBCs();

    void packSharedStiffnessAndLoad(const int* conn, int numNodes,
                                    int* localNodeIndices,
                                    int* scatterIndices, int numIndices,
                                    const double* const* stiffness,
                                    const double* load);

    int formElemScatterList(int blockIndex, 
                            const GlobalID *elemConn, 
                            int* localNodeIndices,
                            int *scatterIndices);

    void assembleStiffnessAndLoad(int numNodes, int* scatterIndices, 
                                  const double* const* stiff,
                                  const double* load);

    void assembleStiffness(int numNodes, int* scatterIndices, 
                           const double* const* stiff);

    void assembleLoad(int numNodes, int* scatterIndices, 
                      const double* load);

    //this function determines which processor is the "owner"
    int ownerProc(int* procs, int numProcs);

//==============================================================================

  protected:
    void appendParamStrings(int numStrings, char **strings);

    int getParam(const char *flag, int numParams,
                     char **paramStrings, char *param);

//==============================================================================
  protected:
//
//The following functions are associated with the specific tasks of
//using the underlying solver sub-library -- filling data structures,
//selecting and launching solver-methods, etc.
//
//These functions must be implemented by the concrete class that
//inherits from BASE_SLE.
//
    virtual void initLinearAlgebraCore() = 0;
    virtual void deleteLinearAlgebraCore() = 0;

    virtual void createLinearAlgebraCore(int globalNumEqns, int localStartRow,
                                 int localEndRow, int localStartCol,
                                 int localEndCol) = 0;

    virtual void resetMatrixAndVector(double s) = 0;
    virtual void matrixConfigure(IntArray* sysRowLengths) = 0;

    virtual void sumIntoRHSVector(int num, 
                          const int* indices, 
                          const double* values) = 0;

    virtual void putIntoSolnVector(int num, 
                           const int* indices,
                           const double* values) = 0;

    virtual double accessSolnVector(int equation) = 0;

    virtual void sumIntoSystemMatrix(int row,
                             int numValues,
                             const double* values,
                             const int* scatterIndices) = 0;

    virtual void enforceEssentialBC(int* globalEqn, double* alpha,
                                    double* gamma, int len) = 0;

    virtual void enforceOtherBC(int* globalEqn, double* alpha, double* beta,
                                double* gamma, int len) = 0;

    virtual void matrixLoadComplete() = 0;
    virtual void selectSolver(char *name) = 0;
    virtual void selectPreconditioner(char *name) = 0;
    virtual void launchSolver(int* solveStatus) = 0;

//==============================================================================
  private:
    int internalFei_;

  protected:
    int solveCounter_;
    void setDebugOutput(char* path, char* name);
    char* debugFileName_;
    char* debugPath_;
    int debugOutput_;
    FILE *debugFile_;

  private:
    bool loadRecvsLaunched_;
    NodeControlPacket **shNodeInfo_;
    MPI_Request *shRequests_, *shScatRequests_, *shCoefRequests_;
    int *shProc_, *shNodesFromProc_;
    int numShProcs_, shSize_, shCoeffSize_;
    int **shScatterIndices_;
    double **shCoeff_;
    SharedNodeBuffer *sharedBuffI_, *sharedBuffL_;
    CommBufferDouble* shBuffLoadD_;
    CommBufferInt*    shBuffLoadI_;
    CommBufferDouble* shBuffRHSLoadD_;
    bool shBuffLoadAllocated_;
    bool shBuffRHSAllocated_;

//==============================================================================
// Following are some 'protected' variables. Take great care when
// changing these, as they are directly inherited and used by the
// concrete classes that extend this class (i.e., ISIS_SLE and
// Aztec_SLE).
//
  protected:
    int numParams_;
    char **paramStrings_;

    int localStartRow_, localEndRow_;
    IntArray* sysMatIndices;

    bool rowScale_, colScale_;

    int iterations_;
    int numRHSs_;
    int currentRHS_;

    char *solverName_;
    char *precondName_;
    int outputLevel_;

// wall clock time stuff
    double baseTime_, wTime_, sTime_;

    int storeNumProcEqns;      // number of equations on this processor
    int storeNumCRMultRecords; // number of Lagrange constraint records

// lists of Lagrange constraints
    MultConstRecord *ceqn_MultConstraints; 

//    communications layer

    MPI_Comm FEI_COMM_WORLD;

    int masterRank_;
    int localRank_;
    int numProcs_;

  private:
    int packet_tag1, packet_tag2, indices_tag, coeff_tag;
    int length_tag, field_tag, id_tag;
    int extSendPacketTag, extRecvPacketTag;

    int wtBlock[WTPACK_SIZE]; //WTPACK_SIZE is defined in NodePackets.h

    MPI_Datatype MPI_GLOBALID, MPI_NodeWtPacket;
    MPI_Datatype MPI_NodePacket;
    MPI_Aint base, wtDisp[WTPACK_SIZE], wtBase;

  private:

//-------------------------------------------------------------------
//  control parameters (can be considered local to each processor)
//-------------------------------------------------------------------

//  non-volatile control data

    int storeNumFields;        // number of solution fields in this problem
    int storeNumElemBlocks;    // number of blocks in this problem
    int storeSolvType;         // type of solution process to invoke
    int storeNumProcActNodes;  // number of active nodes on this processor
    int storeNumProcActEqns;   // number of equations arising from active nodes
    int storeBCNodeSets;       // number of bc node sets
    int storeSharedNodeSets;   // number of shared node sets
    int storeExtNodeSets;      // number of external node sets
    int storeNumCRPenRecords;  // number of penalty constraint records

    int *storeNumWorksets;     // number of worksets in each block

//  some data for consistency checking (some of these not yet utilized)

    int checkElemBlocksLoaded;
    int checkNumElemBlocks;    // verify number of blocks in this problem
    int checkSolvType;         // verify type of solution process
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

    GlobalIDArray GID_blockIDList; //for managing blockID numbers that
                                    //are arbitrarily large. (i.e., can't
                                    //be used for direct array indices)

//  volatile control data

    GlobalID currentElemBlockID;         // current block ID counter
    int currentElemBlockIndex;           // current block blockRoster index
    int currentWorkSetID;                // current workset ID counter


//$kdm begin new block storage formats for v1.0

    BlockRecord *blockRoster;
    FieldRecord *fieldRoster;
    bool fieldRosterAllocated_;
    
//$kdm end new block storage formats for v1.0


// lists of Penalty constraints

    PenConstRecord *ceqn_PenConstraints;     


///////////  global node data bank lists /////////////

// (local) list of active nodes

    NodeRecord *gnod_LocalNodes;

// add some stuff here to manage global-to-local data lookups...

//    we really ought to derive an "active node" object with appropriate fields
//    like globalNodeID, numDOF, and localEqnID, and then write appropriate
//    comparison functions so that we can keep all the fields coherent as we 
//    sort them...

//    this makes for a bit more computational effort, but it's all going to 
//    disappear anyway (I hope!)

    GlobalIDArray GID_ActNodeList;    // sorted active nodes 
    IntArray *IA_NodDOFList;   // list of soln cardinalities
    IntArray *IA_localElems; //number of elems each node is in
     
//  here's the data for identifying our shared nodes

    SharedNodeRecord *sharedNodes_;
    int currentSharedNodeSet;

//  similar structures for caching external node data...

    ExternNodeRecord *externalNodes_;

    int currentExtNodeSet;

};

#endif

