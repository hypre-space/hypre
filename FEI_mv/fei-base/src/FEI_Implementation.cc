#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef FEI_SER
#include <mpiuni/mpi.h>
#else
#include <mpi.h>
#endif

#include "other/basicTypes.h"
#include "fei.h"

#include <../isis-mv/RealArray.h>
#include <../isis-mv/IntArray.h>
#include <../isis-mv/GlobalIDArray.h>

#include "src/BCRecord.h"
#include "src/BCManager.h"
#include "src/FieldRecord.h"
#include "src/BlockDescriptor.h"
#include "src/MultConstRecord.h"
#include "src/PenConstRecord.h"
#include "src/NodeDescriptor.h"
#include "src/NodeCommMgr.h"
#include "src/ProcEqns.h"
#include "src/EqnBuffer.h"
#include "src/EqnCommMgr.h"
#include "src/ProblemStructure.h"
#include "src/SLE_utils.h"
#include "src/Utils.h"

#include "src/Data.h"
#include "src/LinearSystemCore.h"

#include "src/BASE_FEI.h"

#include "src/FEI_Implementation.h"

//------------------------------------------------------------------------------
FEI_Implementation::FEI_Implementation(LinearSystemCore* linSysCore,
                                       MPI_Comm comm, int masterRank)
 : FEI(),
   constructorLinSysCore_(linSysCore),
   linSysCore_(NULL),
   fei_(NULL),
   numInternalFEIs_(0),
   internalFEIsAllocated_(false),
   initSolveStepCalled_(false),
   initPhaseIsComplete_(false),
   aggregateSystemFormed_(false),
   soln_fei_matrix_(NULL),
   soln_fei_vector_(NULL),
   comm_(comm),
   masterRank_(masterRank),
   outputLevel_(0),
   debugOutput_(0)
{
//  start the wall clock time recording

    baseTime_ = MPI_Wtime();
    wTime_ = 0.0;
    sTime_ = 0.0;

//  initialize MPI communications info

    MPI_Comm_rank(comm_, &localRank_);
    MPI_Comm_size(comm_, &numProcs_);

    solveCounter_ = 1;
    debugPath_ = NULL;
    debugFileName_ = NULL;

    numParams_ = 0;
    paramStrings_ = NULL;

    index_soln_fei_ = 0;
    index_current_fei_ = 0;
    index_current_rhs_ = 0;

    IDsAllocated_ = false;
    numRHSIDs_ = NULL;
    rhsIDs_ = NULL;

    matScalars_ = NULL;
    matScalarsSet_ = false;
    rhsScalars_ = NULL;
    rhsScalarsSet_ = false;

    solveType_ = -1;

//  and the time spent in the constructor is...

    wTime_  = MPI_Wtime() - baseTime_;

    return;
}

//------------------------------------------------------------------------------
FEI_Implementation::~FEI_Implementation() {
//
//  Destructor function. Free allocated memory, etc.
//

    int i;

    if (soln_fei_matrix_) {
       linSysCore_[0]->destroyMatrixData(*soln_fei_matrix_);
       delete soln_fei_matrix_;
       soln_fei_matrix_ = NULL;
    }

    if (soln_fei_vector_) {
       linSysCore_[0]->destroyVectorData(*soln_fei_vector_);
       delete soln_fei_vector_;
       soln_fei_vector_ = NULL;
    }

    if (internalFEIsAllocated_) {
        for(i=0; i<numInternalFEIs_; i++){
            delete linSysCore_[i];
            delete fei_[i];
        }
        delete [] linSysCore_;
        delete [] fei_;
    }

    for(i=0; i<numInternalFEIs_; i++){
        delete [] rhsIDs_[i];
        delete [] rhsScalars_[i];
    }
    delete [] rhsIDs_;
    delete [] rhsScalars_;
    delete [] numRHSIDs_;

    internalFEIsAllocated_ = false;
    numInternalFEIs_ = 0;
    delete [] feiIDs_;

    delete [] matScalars_;

    for(i=0; i<numParams_; i++) delete [] paramStrings_[i];
    delete [] paramStrings_;

    if (debugOutput_) {
        delete [] debugPath_;
        delete [] debugFileName_;
        fclose(debugFile_);
    }

    return;
}


//------------------------------------------------------------------------------
int FEI_Implementation::setMatrixID(int matID){

   debugOut("setMatrixID");

   index_current_fei_ = -1;

   if (!initPhaseIsComplete_) {
      cerr << "FEI_Implementation::setMatrixID: WARNING, setting the matrix "
         << "ID before the init phase is complete, will produce undefined"
         << " behavior. A single init phase automatically applies to all "
         << "matrices being assembled." << endl;
   }

   for(int i=0; i<numInternalFEIs_; i++){
      if (feiIDs_[i] == matID) index_current_fei_ = i;
   }

   if (debugOutput_) {
      fprintf(debugFile_,"--- ID: %d, ind: %d\n",
              matID, index_current_fei_);
      fflush(debugFile_);
   }

   //if matID wasn't found, return non-zero (error)
   if (index_current_fei_ == -1) {
      cerr << "FEI_Implementation::setMatrixID: ERROR, invalid matrix ID "
           << "supplied" << endl;
      debugOut("leaving setMatrixID, ERROR");
      return(1);
   }

   debugOut("leaving setMatrixID");

   return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::setRHSID(int rhsID){

    debugOut("setRHSID");

    bool found = false;

    for(int j=0; j<numInternalFEIs_; j++){
        int index = search_index(rhsID, rhsIDs_[j], numRHSIDs_[j]);
        if (index >= 0) {
            index_current_rhs_row_ = j;
            index_current_rhs_ = index;
            found = true;
            break;
        }
    }

    if (debugOutput_) {
        fprintf(debugFile_,"--- ID: %d, row: %d, ind: %d\n",
                rhsID, index_current_rhs_row_, index_current_rhs_);
        fflush(debugFile_);
    }

    if (!found) {
        cerr << "FEI_Implementation::setRHSID: ERROR, invalid RHS ID supplied"
             << endl;
        debugOut("leaving setRHSID, ERROR");
        return(1);
    }

    debugOut("leaving setRHSID");

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::initSolveStep(int numElemBlocks, int solveType) {
//
//  tasks: allocate baseline data structures
//

    baseTime_ = MPI_Wtime();

    debugOut("initSolveStep");

    if (debugOutput_) {
        fprintf(debugFile_,"--- numElemBlocks: %d, solveType: %d\n",
                numElemBlocks, solveType);
        fflush(debugFile_);
    }
    
    if (initSolveStepCalled_) {
        cerr << "FEI_Implementation::initSolveStep: ERROR, initSolveStep has "
             << "already been called. Aborting." << endl;
        abort();
    }
    else initSolveStepCalled_ = true;

    if (solveType_ == -1) solveType_ = solveType;

    if (solveType_ == 0) {
        //0 means standard Ax=b solution
        if (numInternalFEIs_ <= 0) numInternalFEIs_ = 1;
    }
    else if (solveType_ == 1) {
        //1 means we'll be doing an eigen-solution
    }
    else if (solveType_ == 2) {
        //2 means we're solving a linear-combination of separately
        //assembled matrices and rhs vectors
    }
    else if (solveType_ == 3) {
        //3 means we're solving a product of separately assembled
        //matrices -- i.e., (C^T*M*C)x = rhs
    }
    else if (solveType_ == 4) {
        //4 means we'll be doing a multi-level solution
    }

    allocateInternalFEIs();

    ProblemStructure* probStruc = NULL;

    for(int i=0; i<numInternalFEIs_; i++){
        if (debugOutput_) {
            fprintf(debugFile_,"-- fei[%d]->setNumRHSVectors %d\n",
                    i, numRHSIDs_[i]);
            fprintf(debugFile_,"-- fei[%d]->initSolveStep\n",i);
            fflush(debugFile_);
        }
        if (numRHSIDs_[i] == 0) {
            int dummyID = -1;
            fei_[i]->setNumRHSVectors(1, &dummyID);
        }
        else {
            fei_[i]->setNumRHSVectors(numRHSIDs_[i], rhsIDs_[i]);
        }

        if (i==index_soln_fei_) {
           fei_[i]->initSolveStep(numElemBlocks, solveType);
           fei_[i]->getProblemStructure(probStruc);
        }
        else fei_[i]->setProblemStructure(probStruc);
    }

    debugOut("leaving initSolveStep");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::initSolveStep(int numElemBlocks, int solveType,
                            int numMatrices, int* matrixIDs,
                            int* numRHSs, int** rhsIDs) {
//
    if (numMatrices <= 0) {
        if (solveType == 0) numMatrices = 1;
        else badParametersAbort("FEI_Implementation::initSolveStep");
    }

    allocateInternalFEIs(numMatrices, matrixIDs, numRHSs, rhsIDs);

    initSolveStep(numElemBlocks, solveType);

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::initFields(int numFields,
                         const int *cardFields,
                         const int *fieldIDs) {
//
//  tasks: identify all the solution fields present in the analysis
//

    baseTime_ = MPI_Wtime();

    debugOut("initFields", index_soln_fei_);

    if (internalFEIsAllocated_){
       fei_[index_soln_fei_]->initFields(numFields, cardFields, fieldIDs);
    }
    else {
        notAllocatedAbort("FEI_Implementation::initFields");
    }

    debugOut("leaving initFields");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 

//------------------------------------------------------------------------------
int FEI_Implementation::beginInitElemBlock(GlobalID elemBlockID,
                                 int numNodesPerElement,
                                 const int *numElemFields,
                                 const int *const *elemFieldIDs,
                                 int interleaveStrategy,
                                 int lumpingStrategy,
                                 int numElemDOF,
                                 int numElemSets,
                                 int numElemTotal) {
//
//  tasks: store defining parameters for the blocks that will
//         be utilized in subsequent calls.
//

    baseTime_ = MPI_Wtime();

    debugOut("beginInitElemBlock", index_soln_fei_);

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->beginInitElemBlock(elemBlockID,
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
        notAllocatedAbort("FEI_Implementation::beginInitElemBlock");
    }

    debugOut("leaving beginInitElemBlock");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_Implementation::initElemSet(int numElems, 
                          const GlobalID *elemIDs, 
                         const GlobalID *const *elemConn) {
//
//  tasks: convert element data from block-based to globalElemBank basis
//
//         store element connectivities for use in determining sparsity
//         pattern (reuse the space required by these stored parameters
//         to simplify assembly later on?).
//

    baseTime_ = MPI_Wtime();

    debugOut("initElemSet", index_soln_fei_);

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->initElemSet(numElems, 
                                                     elemIDs,
                                                     elemConn);
    }
    else {
        notAllocatedAbort("FEI_Implementation::initElemSet");
    }

    debugOut("leaving initElemSet");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::endInitElemBlock() {
//
//  tasks: check to insure consistency of data
//

    baseTime_ = MPI_Wtime();

    debugOut("endInitElemBlock", index_soln_fei_);

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->endInitElemBlock();
    }
    else {
        notAllocatedAbort("FEI_Implementation::endInitElemBlock");
    }

    debugOut("leaving endInitElemBlock");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_Implementation::beginInitNodeSets(int numSharedNodeSets, 
                                int numExtNodeSets) {
//
//  tasks: simply set the number of shared node sets and external node sets
//         that are to be identified.
//

    baseTime_ = MPI_Wtime();

    debugOut("beginInitNodeSets", index_soln_fei_);

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->beginInitNodeSets(numSharedNodeSets,
                                                           numExtNodeSets);
    }
    else {
        notAllocatedAbort("FEI_Implementation::beginInitNodeSets");
    }

    debugOut("leaving beginInitNodeSets");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int FEI_Implementation::initSharedNodeSet(const GlobalID *sharedNodeIDs,  
                               int lenSharedNodeIDs, 
                                const int *const *sharedProcIDs, 
                               const int *lenSharedProcIDs) {
//
//  In this function we simply accumulate the incoming data into internal arrays
//  in the shareNodes_ object.
//

    baseTime_ = MPI_Wtime();

    debugOut("initSharedNodeSet", index_soln_fei_);

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->initSharedNodeSet(sharedNodeIDs,
                                              lenSharedNodeIDs,
                                              sharedProcIDs,
                                              lenSharedProcIDs);
    }
    else {
        notAllocatedAbort("FEI_Implementation::initSharedNodeSet");
    }
 
    debugOut("leaving initSharedNodeSet");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::initExtNodeSet(const GlobalID *extNodeIDs,
                             int lenExtNodeIDs, 
                             const int *const *extProcIDs,
                             const int *lenExtProcIDs) {

    baseTime_ = MPI_Wtime();

    debugOut("initExtNodeSet", index_soln_fei_);

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->initExtNodeSet(extNodeIDs,
                                           lenExtNodeIDs,
                                           extProcIDs,
                                           lenExtProcIDs);
    }
    else {
        notAllocatedAbort("FEI_Implementation::initExtNodeSet");
    }

    debugOut("leaving initExtNodeSet");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::endInitNodeSets() {
//
//  tasks: check to insure consistency of data (e.g., number of
//         passed lists equals number given in initSolveStep).
//

    baseTime_ = MPI_Wtime();

    debugOut("leaving endInitNodeSets", index_soln_fei_);

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->endInitNodeSets();
    }
    else {
        notAllocatedAbort("FEI_Implementation::endInitNodeSets");
    }
    
    debugOut("leaving endInitNodeSets");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::beginInitCREqns(int numCRMultRecords, 
                              int numCRPenRecords) {
//
//  tasks: allocate baseline data for the constraint relations
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"beginInitCREqns\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->beginInitCREqns(numCRMultRecords,
                                            numCRPenRecords);
    }
    else {
        notAllocatedAbort("FEI_Implementation::beginInitCREqns");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::initCRMult(const GlobalID *const *CRMultNodeTable,
                         const int *CRFieldList,
                         int numMultCRs, 
                         int lenCRNodeList,
                         int& CRMultID) {
//
//  tasks: store Lagrange constraint data into internal structures
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"initCRMult\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->initCRMult(CRMultNodeTable,
                                       CRFieldList,
                                       numMultCRs,
                                       lenCRNodeList,
                                       CRMultID);
    }
    else {
        notAllocatedAbort("FEI_Implementation::initCRMult");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::initCRPen(const GlobalID *const *CRPenNodeTable, 
                        const int *CRFieldList,
                        int numPenCRs, 
                        int lenCRNodeList,
                        int& CRPenID) {
//
//  tasks: store penalty constraint data into internal structures
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"initCRPen\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->initCRPen(CRPenNodeTable,
                                      CRFieldList,
                                      numPenCRs,
                                      lenCRNodeList,
                                      CRPenID);
    }
    else {
        notAllocatedAbort("FEI_Implementation::initCRPen");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::endInitCREqns() {
//
//  tasks: check consistency of constraint equation data.
//

    if (debugOutput_) {
        fprintf(debugFile_,"endInitCREqns\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->endInitCREqns();
    }
    else {
        notAllocatedAbort("FEI_Implementation::endInitCREqns");
    }

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_Implementation::initComplete() {
//
//  tasks: determine final sparsity pattern for use in allocating memory
//         for sparse matrix storage in preparation for assembling
//         element and constraint data.
//
//         allocate storage for upcoming assembly of element terms
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"initComplete\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_soln_fei_]->initComplete();
    }
    else {
        notAllocatedAbort("FEI_Implementation::initComplete");
    }

    initPhaseIsComplete_ = true;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::resetSystem(double s) {
//
//  This function may only be called after the initialization phase is
//  complete. It requires that the system matrix and rhs vector have already
//  been created.
//  It then puts the value s throughout both the matrix and the vector.
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"resetSystem\n");
    }

    if (internalFEIsAllocated_){
        fei_[index_current_fei_]->resetSystem(s);
    }
    else {
        notAllocatedAbort("FEI_Implementation::resetSystem");
    }
 
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int FEI_Implementation::beginLoadNodeSets(int numBCNodeSets) {
//
//  tasks: start the loading of nodal loading information
//

    if (debugOutput_) {
        fprintf(debugFile_,"beginLoadNodeSets\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        int index = index_current_fei_;
        if (solveType_ == 2) index = index_soln_fei_;

        fei_[index]->beginLoadNodeSets(numBCNodeSets);
    }
    else {
        notAllocatedAbort("FEI_Implementation::beginLoadNodeSets");
    }

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_Implementation::loadBCSet(const GlobalID *BCNodeSet,  
                      int lenBCNodeSet,  
                      int BCFieldID,
                      const double *const *alphaBCDataTable,  
                      const double *const *betaBCDataTable,  
                      const double *const *gammaBCDataTable) {
//
//  tasks: load boundary condition information for a given nodal data set
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"loadBCSet\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        int index = index_current_fei_;
        if (solveType_ == 2) index = index_soln_fei_;

        fei_[index]->loadBCSet(BCNodeSet,
                                      lenBCNodeSet,
                                      BCFieldID,
                                      alphaBCDataTable,
                                      betaBCDataTable,
                                      gammaBCDataTable);
    }
    else {
        notAllocatedAbort("FEI_Implementation::loadBCSet");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::endLoadNodeSets() {
//
//  tasks: complete the loading of nodal loading information
//

    debugOut("endLoadNodeSets");

    if (internalFEIsAllocated_){
        int index = index_current_fei_;
        if (solveType_ == 2) index = index_soln_fei_;

        fei_[index]->endLoadNodeSets();
    }
    else {
        notAllocatedAbort("FEI_Implementation::endLoadNodeSets");
    }

    debugOut("leaving endLoadNodeSets");

    return(0);
}

 
//------------------------------------------------------------------------------
int FEI_Implementation::beginLoadElemBlock(GlobalID elemBlockID,
                                 int numElemSets,
                                 int numElemTotal) {
//
//  tasks: begin blocked-element data loading phase
//

    baseTime_ = MPI_Wtime();

    debugOut("beginLoadElemBlock", index_current_fei_);

    if (internalFEIsAllocated_){
        fei_[index_current_fei_]->beginLoadElemBlock(elemBlockID,
                                               numElemSets,
                                               numElemTotal);
    }
    else {
        notAllocatedAbort("FEI_Implementation::beginLoadElemBlock");
    }

    debugOut("leaving beginLoadElemBlock");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_Implementation::loadElemSet(int elemSetID, 
                          int numElems, 
                          const GlobalID *elemIDs,  
                          const GlobalID *const *elemConn,
                          const double *const *const *elemStiffness,
                          const double *const *elemLoad,
                          int elemFormat) {
//
//  tasks: pass, manipulate, and assemble the element stiffness matrices and
//         load vectors for a given workset
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSet, -> fei[%d], rhs %d\n",
                index_current_fei_, index_current_rhs_);
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){

        int rhsContext = rhsIDs_[index_current_fei_][index_current_rhs_];
        fei_[index_current_fei_]->setRHSID(rhsContext);

        fei_[index_current_fei_]->loadElemSet(elemSetID,
                                        numElems,
                                        elemIDs,
                                        elemConn,
                                        elemStiffness,
                                        elemLoad,
                                        elemFormat);
    }
    else {
        notAllocatedAbort("FEI_Implementation::loadElemSet");
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadElemSet\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::loadElemSetMatrix(int elemSetID,
                                int numElems,
                                const GlobalID *elemIDs,
                                const GlobalID *const *elemConn,
                                const double *const *const *elemStiffness,
                                int elemFormat) {
//
//  tasks: pass, manipulate, and assemble the element stiffness matrices and
//         load vectors for a given workset
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSetMatrix, -> fei[%d]\n",
                index_current_fei_);
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){

        fei_[index_current_fei_]->loadElemSetMatrix(elemSetID,
                                        numElems,
                                        elemIDs,
                                        elemConn,
                                        elemStiffness,
                                        elemFormat);
    }
    else {
        notAllocatedAbort("FEI_Implementation::loadElemSetMatrix");
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadElemSetMatrix\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::loadElemSetRHS(int elemSetID,
                             int numElems,
                             const GlobalID *elemIDs,
                             const GlobalID *const *elemConn,
                             const double *const *elemLoad) {
//
//  tasks: pass, manipulate, and assemble the element
//         load vectors for a given workset
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSetRHS, -> fei[%d], rhs %d\n",
                index_current_rhs_row_, index_current_rhs_);
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){

        int rhsContext = rhsIDs_[index_current_rhs_row_][index_current_rhs_];
        fei_[index_current_rhs_row_]->setRHSID(rhsContext);

        fei_[index_current_rhs_row_]->loadElemSetRHS(elemSetID,
                                                numElems,
                                                elemIDs,
                                                elemConn,
                                                elemLoad);
    }
    else {
        notAllocatedAbort("FEI_Implementation::loadElemSetRHS");
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadElemSetRHS\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
// element-wise transfer operator loading.
int FEI_Implementation::loadElemSetTransfers(int elemSetID,
                                   int numElems,
                                   GlobalID** coarseNodeLists,
                                   GlobalID** fineNodeLists,
                                   int fineNodesPerCoarseElem,
                                   double*** elemProlong,
                                   double*** elemRestrict){

    //these void casts simply prevent compiler warnings about
    //"declared but never referenced" variables.
    (void)elemSetID;
    (void)numElems;
    (void)coarseNodeLists;
    (void)fineNodeLists;
    (void)fineNodesPerCoarseElem;
    (void)elemProlong;
    (void)elemRestrict;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::endLoadElemBlock() {
//
//  tasks: end blocked-element data loading step
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"endLoadElemBlock\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_current_fei_]->endLoadElemBlock();
    }
    else {
        notAllocatedAbort("FEI_Implementation::endLoadElemBlock");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_Implementation::beginLoadCREqns(int numCRMultSets, 
                              int numCRPenSets) {
//
//  tasks: initiate constraint condition data loading step
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"beginLoadCREqns\n");
        fflush(debugFile_);
    }
        
    if (internalFEIsAllocated_){
        fei_[index_current_fei_]->beginLoadCREqns(numCRMultSets,
                                            numCRPenSets);
    }
    else {
        notAllocatedAbort("FEI_Implementation::beginLoadCREqns");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

         
//------------------------------------------------------------------------------
int FEI_Implementation::loadCRMult(int CRMultID, 
                         int numMultCRs,
                         const GlobalID *const *CRNodeTable, 
                         const int *CRFieldList,
                         const double *const *CRWeightTable,
                         const double *CRValueList,
                         int lenCRNodeList) {
//
//  tasks: load step for Lagrange multiplier constraint condition sets
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"loadCRMult\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_current_fei_]->loadCRMult(CRMultID,
                                       numMultCRs,
                                       CRNodeTable,
                                       CRFieldList,
                                       CRWeightTable,
                                       CRValueList,
                                       lenCRNodeList);
    }
    else {
        notAllocatedAbort("FEI_Implementation::loadCRMult");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
//
//  tasks: perform penalty constraint relation data loading step
//
int FEI_Implementation::loadCRPen(int CRPenID, 
                        int numPenCRs, 
                        const GlobalID *const *CRNodeTable,
                        const int *CRFieldList,
                        const double *const *CRWeightTable,
                        const double *CRValueList,
                        const double *penValues,
                        int lenCRNodeList) {

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"loadCRPen\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_current_fei_]->loadCRPen(CRPenID,
                                      numPenCRs,
                                      CRNodeTable,
                                      CRFieldList,
                                      CRWeightTable,
                                      CRValueList,
                                      penValues,
                                      lenCRNodeList);
    }
    else {
        notAllocatedAbort("FEI_Implementation::loadCRPen");
    }

    
    wTime_ += MPI_Wtime() - baseTime_;
    
    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::endLoadCREqns() {

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"endLoadCREqns\n");
        fflush(debugFile_);
    }

    if (internalFEIsAllocated_){
        fei_[index_current_fei_]->endLoadCREqns();
    }
    else {
        notAllocatedAbort("FEI_Implementation::endLoadCREqns");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::loadComplete() {

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"loadComplete, fei[%d]\n", index_current_fei_);
        fflush(debugFile_);
    }
    
    if (internalFEIsAllocated_){
        fei_[index_current_fei_]->loadComplete();
    }
    else {
        notAllocatedAbort("FEI_Implementation::loadComplete");
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadComplete\n");
        fflush(debugFile_);
    }
 
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int FEI_Implementation::setMatScalars(int* IDs, double* scalars, int numScalars){

    int index;

    for(int i=0; i<numScalars; i++){
        index = search_index(IDs[i], feiIDs_, numInternalFEIs_);
        if (index>=0) {
            matScalars_[index] = scalars[i];
        }
        else {
            cerr << "FEI_Implementation::setMatScalars: ERROR, invalid ID supplied"
                 << endl;
            return(1);
        }
    }

    matScalarsSet_ = true;
    if (rhsScalarsSet_) {
        for(int j=0; j<numInternalFEIs_; j++)
            linSysCore_[j]->matrixLoadComplete();

        aggregateSystem();

        matScalarsSet_ = false;
        rhsScalarsSet_ = false;
    }

    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::setRHSScalars(int* IDs, double* scalars, int numScalars){

    int index;
    bool found;

    for(int i=0; i<numScalars; i++){
        found = false;

        for(int j=0; j<numInternalFEIs_; j++){
            index = search_index(IDs[i], rhsIDs_[j], numRHSIDs_[j]);
            if (index>=0) {
                rhsScalars_[j][index] = scalars[i];
                found = true;
                break;
            }
        }

        if (!found) {
            cerr << "FEI_Implementation::setRHSScalars: ERROR, invalid RHS ID supplied"
                 << endl;
            return(1);
        }
    }

    rhsScalarsSet_ = true;
    if (matScalarsSet_) {
        for(int j=0; j<numInternalFEIs_; j++)
            linSysCore_[j]->matrixLoadComplete();

        aggregateSystem();

        matScalarsSet_ = false;
        rhsScalarsSet_ = false;
    }

    return(0);
}

//------------------------------------------------------------------------------
void FEI_Implementation::parameters(int numParams, char **paramStrings) {
//
// this function takes parameters and passes them to the internal
// fei objects.
//

    baseTime_ = MPI_Wtime();

    debugOut("parameters");

    if (internalFEIsAllocated_){
        for(int i=0; i<numInternalFEIs_; i++){
            fei_[i]->parameters(numParams, paramStrings);
        }
    }

    if (numParams == 0 || paramStrings == NULL) {
        debugOut("--- no parameters");
    }
    else {
        // take a copy of these parameters, for later use.
        Utils::appendToCharArrayList(paramStrings_, numParams_,
                                     paramStrings, numParams);

        char param[64];
        if (Utils::getParam("numMatrices", numParams, paramStrings, param) == 1)
            sscanf(param,"%d",&numInternalFEIs_);

        if ( Utils::getParam("outputLevel",numParams,paramStrings,param) == 1){
            sscanf(param,"%d", &outputLevel_);
        }

        if ( Utils::getParam("debugOutput",numParams,paramStrings,param) == 1){
            setDebugOutput(param,"FEI_Impl_debug");
        }

        if (debugOutput_) {
           fprintf(debugFile_,"--- numParams %d\n",numParams);
           for(int i=0; i<numParams; i++){
               fprintf(debugFile_,"----- paramStrings[%d]: %s\n",i,
                       paramStrings[i]);
           }
        }
    }

    wTime_ += MPI_Wtime() - baseTime_;

    debugOut("leaving parameters");
 
    return;
}

//------------------------------------------------------------------------------
void FEI_Implementation::setDebugOutput(char* path, char* name){
//
//This function turns on debug output, and opens a file to put it in.
//
    if (debugOutput_) {
        fclose(debugFile_);
        debugFile_ = NULL;
    }

    int pathLength = strlen(path);
    if (path != debugPath_) {
        delete [] debugPath_;
        debugPath_ = new char[pathLength + 1];
        strcpy(debugPath_, path);
    }

    int nameLength = strlen(name);
    if (name != debugFileName_) {
        delete [] debugFileName_;
        debugFileName_ = new char[nameLength + 1];
        strcpy(debugFileName_,name);
    }

    char* dbFileName = new char[pathLength + nameLength + 24];

    sprintf(dbFileName,"%s/%s.slv%d.%d.%d", path,name,solveCounter_,numProcs_,
            localRank_);

    debugOutput_ = 1;
    debugFile_ = fopen(dbFileName,"w");
    if (!debugFile_){
        cerr << "couldn't open debug output file: " << dbFileName << endl;
        debugOutput_ = 0;
    }

    delete [] dbFileName;
}

//------------------------------------------------------------------------------
int FEI_Implementation::iterateToSolve(int& status) {

   baseTime_ = MPI_Wtime();

   debugOut("iterateToSolve", index_soln_fei_);
 
   buildLinearSystem();

   wTime_ += MPI_Wtime() - baseTime_;

   sTime_ = MPI_Wtime();

   int err = fei_[index_soln_fei_]->iterateToSolve(status);

   sTime_ = MPI_Wtime() - sTime_;

   if (solveType_ == 2) {
      aggregateSystemFormed_ = false;
   }

   debugOut("leaving iterateToSolve");
 
   return(err);
}
             
//------------------------------------------------------------------------------
int FEI_Implementation::getBlockNodeSolution(GlobalID elemBlockID,  
                                   GlobalID *nodeIDList, 
                                   int &lenNodeIDList, 
                                   int *offset,  
                                   double *results) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"getBlockNodeSolution, -> fei[%d]\n",
                index_soln_fei_);
        fflush(debugFile_);
    }

    fei_[index_soln_fei_]->getBlockNodeSolution(elemBlockID,
                                                       nodeIDList,
                                                       lenNodeIDList,
                                                       offset,
                                                       results);
    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::getBlockFieldNodeSolution(GlobalID elemBlockID,
                                        int fieldID,
                                        GlobalID *nodeIDList, 
                                        int& lenNodeIDList, 
                                        int *offset,
                                        double *results) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"getBlockFieldNodeSolution\n");
        fflush(debugFile_);
    }

    fei_[index_soln_fei_]->getBlockFieldNodeSolution(elemBlockID,
                                                            fieldID,
                                                            nodeIDList,
                                                            lenNodeIDList,
                                                            offset,
                                                            results);
    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::putBlockNodeSolution(GlobalID elemBlockID,
                                   const GlobalID *nodeIDList, 
                                   int lenNodeIDList, 
                                   const int *offset,
                                   const double *estimates) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"putBlockNodeSolution\n");
    }

    fei_[index_soln_fei_]->putBlockNodeSolution(elemBlockID,
                                                       nodeIDList,
                                                       lenNodeIDList,
                                                       offset,
                                                       estimates);
    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::putBlockFieldNodeSolution(GlobalID elemBlockID, 
                                        int fieldID, 
                                        const GlobalID *nodeIDList, 
                                        int lenNodeIDList, 
                                        const int *offset,
                                        const double *estimates) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"putBlockFieldNodeSolution\n");
        fflush(debugFile_);
    }

    fei_[index_soln_fei_]->putBlockFieldNodeSolution(elemBlockID,
                                                            fieldID,
                                                            nodeIDList,
                                                            lenNodeIDList,
                                                            offset,
                                                            estimates);
    return(0);
}

//------------------------------------------------------------------------------
int FEI_Implementation::getBlockElemSolution(GlobalID elemBlockID,  
                                   GlobalID *elemIDList,
                                   int& lenElemIDList, 
                                   int *offset,  
                                   double *results, 
                                   int& numElemDOF) {
//
//  return the elemental solution parameters associated with a 
//  particular block of elements
//
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getElemBlockSolution\n");
        fflush(debugFile_);
    }

    return(
    fei_[index_soln_fei_]->getBlockElemSolution(elemBlockID,
                                                       elemIDList,
                                                       lenElemIDList,
                                                       offset,
                                                       results,
                                                       numElemDOF)
    );
} 
      
//------------------------------------------------------------------------------
int FEI_Implementation::putBlockElemSolution(GlobalID elemBlockID,
                                   const GlobalID *elemIDList, 
                                   int lenElemIDList, 
                                   const int *offset, 
                                   const double *estimates, 
                                   int numElemDOF) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"trace: putElemBlockSolution\n");
        fflush(debugFile_);
    }

    return(
    fei_[index_soln_fei_]->putBlockElemSolution(elemBlockID,
                                                       elemIDList,
                                                       lenElemIDList,
                                                       offset,
                                                       estimates,
                                                       numElemDOF)
    );
}

//------------------------------------------------------------------------------
int FEI_Implementation::getCRMultSizes(int& numCRMultIDs, int& lenResults) {
//
//  This function returns the dimensions of the lists that get filled by
//  the getCRMultSolution function. In that function, *CRMultIDs and
//  *offset are both of length numCRMultIDs, and *results is of length
//  lenResults.
//

    return(
    fei_[index_soln_fei_]->getCRMultSizes(numCRMultIDs,
                                                 lenResults)

    );
}

//------------------------------------------------------------------------------
int FEI_Implementation::getCRMultSolution(int& numCRMultSets, 
                                int *CRMultIDs,  
                                int *offset, 
                                double *results) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getCRMultSolution\n");
        fflush(debugFile_);
    }

    return(
    fei_[index_soln_fei_]->getCRMultSolution(numCRMultSets,
                                                    CRMultIDs,
                                                    offset,
                                                    results)
    );
} 

//------------------------------------------------------------------------------
int FEI_Implementation::getCRMultParam(int CRMultID, 
                             int numMultCRs,
                             double *multValues) {

    if (debugOutput_) {
        fprintf(debugFile_,"trace: getCRMultParam\n");
        fflush(debugFile_);
    }

    return(
    fei_[index_soln_fei_]->getCRMultParam(CRMultID,
                                                 numMultCRs,
                                                 multValues)
    );
}


//------------------------------------------------------------------------------
int FEI_Implementation::putCRMultParam(int CRMultID, 
                             int numMultCRs,
                             const double *multEstimates) {
//
//  this method is just the inverse of getCRMultParam(), so...
//
    if (debugOutput_) {
        fprintf(debugFile_,"trace: putCRMultParam\n");
        fflush(debugFile_);
    }

    return(
    fei_[index_soln_fei_]->putCRMultParam(CRMultID,
                                                 numMultCRs,
                                                 multEstimates)
    );
}


//-----------------------------------------------------------------------------
//  some utility functions to aid in using the "put" functions for passing
//  an initial guess to the solver
//-----------------------------------------------------------------------------

//------------------------------------------------------------------------------
int FEI_Implementation::getBlockElemIDList(GlobalID elemBlockID,
                                 GlobalID *elemIDList, 
                                 int& lenElemIDList) {
//
//  return the list of element IDs for a given block... the length parameter
//  lenElemIDList should be used to check memory allocation for the calling
//  method, as the calling method should have gotten a copy of this param 
//  from a call to getNumBlockElements before allocating memory for elemIDList
//
        
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getBlockElemIDList\n");
        fflush(debugFile_);
    }

    return(
    fei_[index_soln_fei_]->getBlockElemIDList(elemBlockID,
                                                     elemIDList,
                                                     lenElemIDList)
    );
}

//------------------------------------------------------------------------------
int FEI_Implementation::getBlockNodeIDList(GlobalID elemBlockID,
                                 GlobalID *nodeIDList, 
                                 int& lenNodeIDList) {
//
//  similar comments as for getBlockElemIDList(), except for returning the
//  active node list
//

    if (debugOutput_) {
        fprintf(debugFile_,"trace: getBlockNodeIDList\n");
        fflush(debugFile_);
    }

    return(
    fei_[index_soln_fei_]->getBlockNodeIDList(elemBlockID,
                                                     nodeIDList,
                                                     lenNodeIDList)
    );
}

//------------------------------------------------------------------------------
int FEI_Implementation::getNumNodesPerElement(GlobalID blockID) const {
//
//  return the number of nodes associated with elements of a given block ID
//

    return(
    fei_[index_soln_fei_]->getNumNodesPerElement(blockID)
    );
}
 
 
//------------------------------------------------------------------------------
int FEI_Implementation::getNumEqnsPerElement(GlobalID blockID) const {
//
//  return the number of eqns associated with elements of a given block ID
//

    return(
    fei_[index_soln_fei_]->getNumEqnsPerElement(blockID)
    );
}


//------------------------------------------------------------------------------
int FEI_Implementation::getNumSolnParams(GlobalID iGlobal) const {
//
//  return the number of solution parameters at a given node
//

    return(
    fei_[index_soln_fei_]->getNumSolnParams(iGlobal)
    );
}
 
 
//------------------------------------------------------------------------------
int FEI_Implementation::getNumElemBlocks() const {
//
//  return the number of element blocks
//

    return(
    fei_[index_soln_fei_]->getNumElemBlocks()
    );
}

//------------------------------------------------------------------------------
int FEI_Implementation::getNumBlockActNodes(GlobalID blockID) const {
//
//  return the number of active nodes associated with a given element block ID
//

    return(
    fei_[index_soln_fei_]->getNumBlockActNodes(blockID)
    );
}


//------------------------------------------------------------------------------
int FEI_Implementation::getNumBlockActEqns(GlobalID blockID) const {
//
// return the number of active equations associated with a given element
// block ID
//

    return(
    fei_[index_soln_fei_]->getNumBlockActEqns(blockID)
    );
}

//------------------------------------------------------------------------------
int FEI_Implementation::getNumBlockElements(GlobalID blockID) const {
//
//  return the number of elements associated with a given elem blockID
//

    return(
    fei_[index_soln_fei_]->getNumBlockElements(blockID)
    );
}


//------------------------------------------------------------------------------
int FEI_Implementation::getNumBlockElemEqns(GlobalID blockID) const {
//
//  return the number of elem equations associated with a given elem blockID
//

    return(
    fei_[index_soln_fei_]->getNumBlockElemEqns(blockID)
    );
}

//------------------------------------------------------------------------------
void FEI_Implementation::buildLinearSystem(){
//
//At the point when this function is called, all matrix assembly has
//already taken place, with the data having been directed into the
//appropriate BASE_FEI instance in the fei_ list. Now it's
//time to get pointers and build a matrix A and vectors x and b
//to give to a solver. (If we're just doing a standard single Ax=b
//system solve, then there's nothing to do here.)
//
   debugOut("   buildLinearSystem");

   if (solveType_ == 2){
      //solveType_ == 2 means this is a linear-combination solve --
      //i.e., we're solving an aggregate system which is the sum of
      //several individual matrices and rhs's.

      if (!aggregateSystemFormed_) {
         cerr << "FEI_Implementation: WARNING: solveType_==2, but aggregate system"
              << " hasn't been formed before solve requested." << endl;
         aggregateSystem();
      }
   }
   debugOut("   leaving buildLinearSystem");
}

//------------------------------------------------------------------------------
void FEI_Implementation::aggregateSystem() { 

   debugOut("   aggregateSystem");

   if (soln_fei_matrix_ == NULL) {
      soln_fei_matrix_ = new Data();
      linSysCore_[index_soln_fei_]->copyOutMatrix(1.0, *soln_fei_matrix_);
   }

   if (soln_fei_vector_ == NULL) {
      soln_fei_vector_ = new Data();

      linSysCore_[index_soln_fei_]->setRHSID(rhsIDs_[index_soln_fei_][0]);
      linSysCore_[index_soln_fei_]->copyOutRHSVector(1.0, *soln_fei_vector_);
   }

   Data tmp;
   Data tmpv;

   for(int i=0; i<numInternalFEIs_; i++){

      if (i == index_soln_fei_) {
         tmp.setTypeName(soln_fei_matrix_->getTypeName());
         tmp.setDataPtr(soln_fei_matrix_->getDataPtr());
         linSysCore_[index_soln_fei_]->copyInMatrix(matScalars_[i], tmp);
      }
      else {
         linSysCore_[i]->getMatrixPtr(tmp);
         linSysCore_[index_soln_fei_]->sumInMatrix(matScalars_[i], tmp);
      }

      for(int j=0; j<numRHSIDs_[i]; j++){
         if ((i == index_soln_fei_) && (j == 0)) {
            tmpv.setTypeName(soln_fei_vector_->getTypeName());
            tmpv.setDataPtr(soln_fei_vector_->getDataPtr());
         }
         else {
            linSysCore_[i]->setRHSID(rhsIDs_[i][j]);
            linSysCore_[i]->getRHSVectorPtr(tmpv);
         }

         if (i == index_soln_fei_) {
            linSysCore_[index_soln_fei_]->
                         copyInRHSVector(rhsScalars_[i][j], tmpv);
         }
         else {
            linSysCore_[index_soln_fei_]->
                         sumInRHSVector(rhsScalars_[i][j], tmpv);
         }
      }
   }

   aggregateSystemFormed_ = true;

   debugOut("   leaving aggregateSystem");
}

//==============================================================================
void FEI_Implementation::allocateInternalFEIs(int numMatrices, int* matrixIDs,
                                    int* numRHSs, int** rhsIDs){
//
    numInternalFEIs_ = numMatrices;

    feiIDs_ = new int[numMatrices];

    numRHSIDs_ = new int[numMatrices];
    rhsIDs_ = new int*[numMatrices];

    matScalars_ = new double[numMatrices];
    rhsScalars_ = new double*[numMatrices];

    for(int i=0; i<numInternalFEIs_; i++){
       feiIDs_[i] = matrixIDs[i];
       matScalars_[i] = 1.0;

       numRHSIDs_[i] = numRHSs[i];
       rhsIDs_[i] = new int[numRHSIDs_[i]];
       rhsScalars_[i] = new double[numRHSIDs_[i]];

       for(int j=0; j<numRHSIDs_[i]; j++){
          rhsIDs_[i][j] = rhsIDs[i][j];
          rhsScalars_[i][j] = 1.0;
       }
    }

    IDsAllocated_ = true;

    allocateInternalFEIs();
}

//==============================================================================
void FEI_Implementation::allocateInternalFEIs(){
//
//This is a private FEI_Implementation function, to be called from within initSolveStep.
//Assumes that numInternalFEIs_ has already been set.
//

   if (internalFEIsAllocated_) return;

   if (numInternalFEIs_ > 0) {
      index_soln_fei_ = 0;
      index_current_fei_ = 0;
      fei_ = new BASE_FEI*[numInternalFEIs_];
      linSysCore_ = new LinearSystemCore*[numInternalFEIs_];

      if (!IDsAllocated_) {
         //if IDsAllocated_ is false, then initSolveStep was called without
         //the matrixIDs and rhsIDs arguments. So we're going to assume that
         //there is only 1 rhs per matrix, and IDs for matrices and rhs's
         //will just be 0-based indices.
         //

         feiIDs_ = new int[numInternalFEIs_];
         numRHSIDs_ = new int[numInternalFEIs_];
         rhsIDs_ = new int*[numInternalFEIs_];

         matScalars_ = new double[numInternalFEIs_];
         rhsScalars_ = new double*[numInternalFEIs_];

         for(int i=0; i<numInternalFEIs_; i++){
            feiIDs_[i] = i;
            matScalars_[i] = 1.0;

            numRHSIDs_[i] = 1;
            rhsIDs_[i] = new int[1];
            rhsIDs_[i][0] = i;

            rhsScalars_[i] = new double[1];
            rhsScalars_[i][0] = 1.0;
         }

         IDsAllocated_ = true;
      }

      linSysCore_[0] = constructorLinSysCore_;

      char *param = new char[32];
      for(int i=0; i<numInternalFEIs_; i++){
         if (i != 0) linSysCore_[i] = constructorLinSysCore_->clone();

         fei_[i] = new BASE_FEI(comm_, linSysCore_[i], masterRank_);

         sprintf(param, "internalFei %d", i);
         fei_[i]->parameters(1, &param);

         fei_[i]->parameters(numParams_, paramStrings_);
      }

      delete [] param;

      internalFEIsAllocated_ = true;
   }
   else {
      needParametersAbort("FEI_Implementation::allocateInternalFEIs");
   }
}

//==============================================================================
void FEI_Implementation::debugOut(char* msg) {
   if (debugOutput_) {
      fprintf(debugFile_,"%s\n", msg);
      fflush(debugFile_);
   }
}

//==============================================================================
void FEI_Implementation::debugOut(char* msg, int whichFEI) {
   if (debugOutput_) {
      fprintf(debugFile_,"%s, -> fei[%d]\n", msg, whichFEI);
      fflush(debugFile_);
   }
}

//==============================================================================
void FEI_Implementation::messageAbort(char* msg){

    cerr << "FEI_Implementation: ERROR " << msg << " Aborting." << endl;
    abort();
}

//==============================================================================
void FEI_Implementation::notAllocatedAbort(char* name){

    cerr << name
         << endl << "ERROR, internal data structures not allocated."
         << endl << "'initSolveStep' and/or 'parameters' must be called"
         << endl << "up front to identify solveType and number of matrices"
         << endl << "to be assembled." << endl;
    abort();
}

//==============================================================================
void FEI_Implementation::needParametersAbort(char* name){

   cerr << name
       << endl << "FEI_Implementation: ERROR, numMatrices has not been specified."
       << endl << "FEI_Implementation: 'parameters' must be called up front with"
       << endl << "FEI_Implementation: the string 'numMatrices n' to specify that"
       << endl << "FEI_Implementation: n matrices will be assembled." << endl;
   abort();
}

//==============================================================================
void FEI_Implementation::badParametersAbort(char* name){

   cerr << name
        << endl << "FEI_Implementation: ERROR, inconsistent 'solveType' and"
        << endl << "FEI_Implementation: 'numMatrices' parameters specified."
        << endl << "FEI_Implementation: Aborting."
        << endl;
   abort();
}

