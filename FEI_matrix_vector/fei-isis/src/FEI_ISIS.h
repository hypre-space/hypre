#ifndef __FEI_ISIS_H
#define __FEI_ISIS_H

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

// #include "mpi.h"
// #include "NodePackets.h"

class FEI_ISIS : public FEI {

  public:
    //Constructor
    FEI_ISIS(MPI_Comm PASSED_COMM_WORLD, int masterRank=0);

    //Destructor
    virtual ~FEI_ISIS();

// Structural initialization sequence.............................

    //direct data to a particular internal matrix
    int beginInitPhase(int index);

    //direct data to a specific internal data structure
    //i.e., set the current matrix 'context'.
    int setMatrixID(int index);

    //direct data to a specific internal data structure
    //i.e., set the current RHS 'context'.
    int setRHSID(int index);

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

    //direct data to a particular internal matrix
    int beginLoadPhase(int index);

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

    // elemSet-based stiffness matrix data loading step
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

    // element-wise transfer operator loading.
    int loadElemSetTransfers(int elemSetID,
                             int numElems,
                             GlobalID** coarseNodeLists,
                             GlobalID** fineNodeLists,
                             int fineNodesPerCoarseElem,
                             double*** elemProlong,
                             double*** elemRestrict);

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
    int iterateToSolve();

    // query function iterations performed.
    int iterations() const {
        return(feiInternal_[index_soln_fei_]->iterations());
    };

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

  private:

//==============================================================================
//below are the private functions and variable declarations for
//this implementation of the FEI.
//==============================================================================

    void allocateFeiInternals();
    void allocateFeiInternals(int numMatrices, int* matrixIDs,
                              int* numRHSs, int** rhsIDs);

    ISIS_SLE** feiInternal_;
    int numFeiInternal_;
    bool feiInternalAllocated_;

    int* feiIDs_;
    int* numRHSIDs_;
    int** rhsIDs_;

    bool IDsAllocated_;

    double* matScalars_;
    bool matScalarsSet_;
    double** rhsScalars_;
    bool rhsScalarsSet_;

    int index_soln_fei_;
    int index_current_fei_;
    int index_current_rhs_;
    int index_current_rhs_row_;

    int solveType_;

    void buildLinearSystem();
    void aggregateSystem();
    bool aggregateSystemFormed_;

    DCRS_Matrix* A_solve_;
    Vector* b_solve_;

    Op_Matrix* A_op_;
    Vector* b_op_;

    MPI_Comm FEI_COMM_WORLD;
    int masterRank_;
    int localRank_;
    int numProcs_;

    void messageAbort(char* msg);
    void notAllocatedAbort(char* name);
    void needParametersAbort(char* name);
    void badParametersAbort(char* name);

  private:
    int outputLevel_;

    void setDebugOutput(char* path, char* name);
    char* debugPath_;
    char* debugFileName_;
    int solveCounter_;
    int debugOutput_;
    FILE *debugFile_;

    double baseTime_, wTime_, sTime_;

    void appendParamStrings(int numStrings, char **strings);
    int getParam(const char *flag, int numParams,
                     char **paramStrings, char *param);

    int numParams_;
    char** paramStrings_;
};

#endif

