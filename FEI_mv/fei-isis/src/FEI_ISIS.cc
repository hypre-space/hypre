#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef FEI_SER
#include "mpiuni/mpi.h"
#else
#include <mpi.h>
#endif

#include <fei-isis.h>
#include "FEI_ISIS.h"

//CASC#include "pc/SAILS_PC.h"
#ifdef HYPRE
#include "pc/PILUT_PC.h"
#endif

//------------------------------------------------------------------------------
FEI_ISIS::FEI_ISIS(MPI_Comm PASSED_COMM_WORLD, int masterRank) : 
    FEI() {

//  start the wall clock time recording

    baseTime_ = MPI_Wtime();
    wTime_ = 0.0;
    sTime_ = 0.0;

//  initialize MPI communications info

    masterRank_ = masterRank;
    FEI_COMM_WORLD = PASSED_COMM_WORLD;
    MPI_Comm_rank(FEI_COMM_WORLD, &localRank_);
    MPI_Comm_size(FEI_COMM_WORLD, &numProcs_);

    outputLevel_ = 0;

    debugOutput_ = 0; //no debug output by default.
    solveCounter_ = 1;
    debugPath_ = NULL;
    debugFileName_ = NULL;

    numParams_ = 0;
    paramStrings_ = NULL;

    feiInternal_ = NULL;
    numFeiInternal_ = 0;
    feiIDs_ = NULL;
    feiInternalAllocated_ = false;
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

    aggregateSystemFormed_ = false;
    soln_fei_matrix_ptr_ = NULL;
    soln_fei_vector_ptr_ = NULL;
    A_solve_ = NULL;
    b_solve_ = NULL;

//  and the time spent in the constructor is...

    wTime_  = MPI_Wtime() - baseTime_;

    return;
}

//------------------------------------------------------------------------------
FEI_ISIS::~FEI_ISIS() {
//
//  Destructor function. Free allocated memory, etc.
//

    int i;

    for(i=0; i<numParams_; i++) delete [] paramStrings_[i];
    delete [] paramStrings_;

    if (feiInternalAllocated_) {
        for(i=0; i<numFeiInternal_; i++){
            delete feiInternal_[i];
        }
        delete [] feiInternal_;
    }

    if ((A_solve_) && (solveType_ == 2)) delete A_solve_;
    if ((b_solve_) && (solveType_ == 2)) delete b_solve_;

    for(i=0; i<numFeiInternal_; i++){
        delete [] rhsIDs_[i];
        delete [] rhsScalars_[i];
    }
    delete [] rhsIDs_;
    delete [] rhsScalars_;
    delete [] numRHSIDs_;

    feiInternalAllocated_ = false;
    numFeiInternal_ = 0;
    delete [] feiIDs_;

    delete [] matScalars_;

    if (debugOutput_) {
        delete [] debugPath_;
        delete [] debugFileName_;
        fclose(debugFile_);
    }

    return;
}


//------------------------------------------------------------------------------
int FEI_ISIS::setMatrixID(int matID){
    index_current_fei_ = -1;

    for(int i=0; i<numFeiInternal_; i++){
        if (feiIDs_[i] == matID) index_current_fei_ = i;
    }

    if (debugOutput_) {
        fprintf(debugFile_,"setMatrixID, ID: %d, ind: %d\n",
                matID, index_current_fei_);
        fflush(debugFile_);
    }

    //if matID wasn't found, return non-zero (error)
    if (index_current_fei_ == -1) {
        cerr << "FEI_ISIS::setMatrixID: ERROR, invalid matrix ID supplied"
             << endl;
        return(1);
    }

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::setRHSID(int rhsID){

    bool found = false;

    for(int j=0; j<numFeiInternal_; j++){
        int index = search_index(rhsID, rhsIDs_[j], numRHSIDs_[j]);
        if (index >= 0) {
            index_current_rhs_row_ = j;
            index_current_rhs_ = index;
            found = true;
            break;
        }
    }

    if (debugOutput_) {
        fprintf(debugFile_,"setRHSID, ID: %d, row: %d, ind: %d\n",
                rhsID, index_current_rhs_row_, index_current_rhs_);
        fflush(debugFile_);
    }

    if (!found) {
        cerr << "FEI_ISIS::setRHSID: ERROR, invalid RHS ID supplied"
             << endl;
        return(1);
    }

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::initSolveStep(int numElemBlocks, int solveType) {
//
//  tasks: allocate baseline data structures
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"initSolveStep: numElemBlocks: %d, solveType: %d\n",
                numElemBlocks, solveType);
        fflush(debugFile_);
    }
    
    if (solveType_ == -1) solveType_ = solveType;

    if (solveType_ == 0) {
        //0 means standard Ax=b solution
        if (numFeiInternal_ <= 0) numFeiInternal_ = 1;
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

    allocateFeiInternals();

    for(int i=0; i<numFeiInternal_; i++){
        if (debugOutput_) {
            fprintf(debugFile_,"-- fei[%d]->setNumRHSVectors %d\n",
                    i, numRHSIDs_[i]);
            fprintf(debugFile_,"-- fei[%d]->initSolveStep\n",i);
            fflush(debugFile_);
        }
        if (numRHSIDs_[i] == 0) {
            int dummyID = -1;
            feiInternal_[i]->setNumRHSVectors(1, &dummyID);
        }
        else {
            feiInternal_[i]->setNumRHSVectors(numRHSIDs_[i], rhsIDs_[i]);
        }
        feiInternal_[i]->initSolveStep(numElemBlocks, solveType);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::initSolveStep(int numElemBlocks, int solveType,
                            int numMatrices, int* matrixIDs,
                            int* numRHSs, int** rhsIDs) {
//
    if (numMatrices <= 0) {
        if (solveType == 0) numMatrices = 1;
        else badParametersAbort("FEI_ISIS::initSolveStep");
    }

    allocateFeiInternals(numMatrices, matrixIDs, numRHSs, rhsIDs);

    initSolveStep(numElemBlocks, solveType);

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::initFields(int numFields,
                         const int *cardFields,
                         const int *fieldIDs) {
//
//  tasks: identify all the solution fields present in the analysis
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"initFields\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        for(int i=0; i<numFeiInternal_; i++){
            if (debugOutput_) {
                fprintf(debugFile_,"fei[%d]->initFields\n",i);
                fflush(debugFile_);
            }
            feiInternal_[i]->initFields(numFields, cardFields, fieldIDs);
        }
    }
    else {
        notAllocatedAbort("FEI_ISIS::initFields");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 

//------------------------------------------------------------------------------
int FEI_ISIS::beginInitElemBlock(GlobalID elemBlockID,
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

    if (debugOutput_) {
        fprintf(debugFile_,"beginInitElemBlock, -> fei[%d]\n",
                index_current_fei_);
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->beginInitElemBlock(elemBlockID,
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
        notAllocatedAbort("FEI_ISIS::beginInitElemBlock");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_ISIS::initElemSet(int numElems, 
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

    if (debugOutput_) {
        fprintf(debugFile_,"initElemSet, numElems: %d\n", numElems);
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->initElemSet(numElems, 
                                                     elemIDs,
                                                     elemConn);
    }
    else {
        notAllocatedAbort("FEI_ISIS::initElemSet");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::endInitElemBlock() {
//
//  tasks: check to insure consistency of data
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"endInitElemBlock\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->endInitElemBlock();
    }
    else {
        notAllocatedAbort("FEI_ISIS::endInitElemBlock");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_ISIS::beginInitNodeSets(int numSharedNodeSets, 
                                int numExtNodeSets) {
//
//  tasks: allocate baseline data structures for various node lists
//         in preparation for conversion of passed node lists to
//         globalNodeBank format.
//
//         perform initialization work for storing nodal data (e.g.,
//         constructing an active node list for this processor)
//
//         (these tasks were formerly done in endInitElemBlocks, but
//         the FEM005.h design rearranged the calling sequence to
//         replace endInitElemBlocks with repeated calls to
//         endInitElemBlock, hence we can't use endInitElemBlock as
//         an "end of all element block initialization" call anymore!)
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"beginInitNodeSets\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->beginInitNodeSets(numSharedNodeSets,
                                                           numExtNodeSets);
    }
    else {
        notAllocatedAbort("FEI_ISIS::beginInitNodeSets");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int FEI_ISIS::initSharedNodeSet(const GlobalID *sharedNodeIDs,  
                               int lenSharedNodeIDs, 
                                const int *const *sharedProcIDs, 
                               const int *lenSharedProcIDs) {
//
//  In this function we simply accumulate the incoming data into internal arrays
//  in the shareNodes_ object.
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"initSharedNodeSet\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->initSharedNodeSet(sharedNodeIDs,
                                              lenSharedNodeIDs,
                                              sharedProcIDs,
                                              lenSharedProcIDs);
    }
    else {
        notAllocatedAbort("FEI_ISIS::initSharedNodeSet");
    }
 
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
//
// store the input parameters in the externalNodes_ object...
//
int FEI_ISIS::initExtNodeSet(const GlobalID *extNodeIDs,
                             int lenExtNodeIDs, 
                             const int *const *extProcIDs,
                             const int *lenExtProcIDs) {

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"initExtNodeSet\n");
        fprintf(debugFile_,"--- getting %d nodes.\n",lenExtNodeIDs);
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->initExtNodeSet(extNodeIDs,
                                           lenExtNodeIDs,
                                           extProcIDs,
                                           lenExtProcIDs);
    }
    else {
        notAllocatedAbort("FEI_ISIS::initExtNodeSet");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::endInitNodeSets() {
//
//  tasks: check to insure consistency of data (e.g., number of
//         passed lists equals number given in initSolveStep).
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"endInitNodeSets\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->endInitNodeSets();
    }
    else {
        notAllocatedAbort("FEI_ISIS::endInitNodeSets");
    }
    
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::beginInitCREqns(int numCRMultRecords, 
                              int numCRPenRecords) {
//
//  tasks: allocate baseline data for the constraint relations
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"beginInitCREqns\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->beginInitCREqns(numCRMultRecords,
                                            numCRPenRecords);
    }
    else {
        notAllocatedAbort("FEI_ISIS::beginInitCREqns");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::initCRMult(const GlobalID *const *CRMultNodeTable,
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

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->initCRMult(CRMultNodeTable,
                                       CRFieldList,
                                       numMultCRs,
                                       lenCRNodeList,
                                       CRMultID);
    }
    else {
        notAllocatedAbort("FEI_ISIS::initCRMult");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::initCRPen(const GlobalID *const *CRPenNodeTable, 
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

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->initCRPen(CRPenNodeTable,
                                      CRFieldList,
                                      numPenCRs,
                                      lenCRNodeList,
                                      CRPenID);
    }
    else {
        notAllocatedAbort("FEI_ISIS::initCRPen");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::endInitCREqns() {
//
//  tasks: check consistency of constraint equation data.
//

    if (debugOutput_) {
        fprintf(debugFile_,"endInitCREqns\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->endInitCREqns();
    }
    else {
        notAllocatedAbort("FEI_ISIS::endInitCREqns");
    }

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_ISIS::initComplete() {
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

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->initComplete();
    }
    else {
        notAllocatedAbort("FEI_ISIS::initComplete");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::resetSystem(double s) {
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

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->resetSystem(s);
    }
    else {
        notAllocatedAbort("FEI_ISIS::resetSystem");
    }
 
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int FEI_ISIS::beginLoadNodeSets(int numBCNodeSets) {
//
//  tasks: start the loading of nodal loading information
//

    if (debugOutput_) {
        fprintf(debugFile_,"beginLoadNodeSets\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        int index = index_current_fei_;
        if (solveType_ == 2) index = index_soln_fei_;

        feiInternal_[index]->beginLoadNodeSets(numBCNodeSets);
    }
    else {
        notAllocatedAbort("FEI_ISIS::beginLoadNodeSets");
    }

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_ISIS::loadBCSet(const GlobalID *BCNodeSet,  
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

    if (feiInternalAllocated_){
        int index = index_current_fei_;
        if (solveType_ == 2) index = index_soln_fei_;

        feiInternal_[index]->loadBCSet(BCNodeSet,
                                      lenBCNodeSet,
                                      BCFieldID,
                                      alphaBCDataTable,
                                      betaBCDataTable,
                                      gammaBCDataTable);
    }
    else {
        notAllocatedAbort("FEI_ISIS::loadBCSet");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::endLoadNodeSets() {
//
//  tasks: complete the loading of nodal loading information
//

    if (debugOutput_) {
        fprintf(debugFile_,"endLoadNodeSets\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        int index = index_current_fei_;
        if (solveType_ == 2) index = index_soln_fei_;

        feiInternal_[index]->endLoadNodeSets();
    }
    else {
        notAllocatedAbort("FEI_ISIS::endLoadNodeSets");
    }

    return(0);
}

 
//------------------------------------------------------------------------------
int FEI_ISIS::beginLoadElemBlock(GlobalID elemBlockID,
                                 int numElemSets,
                                 int numElemTotal) {
//
//  tasks: begin blocked-element data loading phase
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"beginLoadElemBlock\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->beginLoadElemBlock(elemBlockID,
                                               numElemSets,
                                               numElemTotal);
    }
    else {
        notAllocatedAbort("FEI_ISIS::beginLoadElemBlock");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_ISIS::loadElemSet(int elemSetID, 
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

    if (feiInternalAllocated_){

        int rhsContext = rhsIDs_[index_current_fei_][index_current_rhs_];
        feiInternal_[index_current_fei_]->setRHSID(rhsContext);

        feiInternal_[index_current_fei_]->loadElemSet(elemSetID,
                                        numElems,
                                        elemIDs,
                                        elemConn,
                                        elemStiffness,
                                        elemLoad,
                                        elemFormat);
    }
    else {
        notAllocatedAbort("FEI_ISIS::loadElemSet");
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadElemSet\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::loadElemSetMatrix(int elemSetID,
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

    if (feiInternalAllocated_){

        feiInternal_[index_current_fei_]->loadElemSetMatrix(elemSetID,
                                        numElems,
                                        elemIDs,
                                        elemConn,
                                        elemStiffness,
                                        elemFormat);
    }
    else {
        notAllocatedAbort("FEI_ISIS::loadElemSetMatrix");
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadElemSetMatrix\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::loadElemSetRHS(int elemSetID,
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

    if (feiInternalAllocated_){

        int rhsContext = rhsIDs_[index_current_rhs_row_][index_current_rhs_];
        feiInternal_[index_current_rhs_row_]->setRHSID(rhsContext);

        feiInternal_[index_current_rhs_row_]->loadElemSetRHS(elemSetID,
                                                numElems,
                                                elemIDs,
                                                elemConn,
                                                elemLoad);
    }
    else {
        notAllocatedAbort("FEI_ISIS::loadElemSetRHS");
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
int FEI_ISIS::loadElemSetTransfers(int elemSetID,
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
int FEI_ISIS::endLoadElemBlock() {
//
//  tasks: end blocked-element data loading step
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"endLoadElemBlock\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->endLoadElemBlock();
    }
    else {
        notAllocatedAbort("FEI_ISIS::endLoadElemBlock");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int FEI_ISIS::beginLoadCREqns(int numCRMultSets, 
                              int numCRPenSets) {
//
//  tasks: initiate constraint condition data loading step
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"beginLoadCREqns\n");
        fflush(debugFile_);
    }
        
    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->beginLoadCREqns(numCRMultSets,
                                            numCRPenSets);
    }
    else {
        notAllocatedAbort("FEI_ISIS::beginLoadCREqns");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

         
//------------------------------------------------------------------------------
int FEI_ISIS::loadCRMult(int CRMultID, 
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

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->loadCRMult(CRMultID,
                                       numMultCRs,
                                       CRNodeTable,
                                       CRFieldList,
                                       CRWeightTable,
                                       CRValueList,
                                       lenCRNodeList);
    }
    else {
        notAllocatedAbort("FEI_ISIS::loadCRMult");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
//
//  tasks: perform penalty constraint relation data loading step
//
int FEI_ISIS::loadCRPen(int CRPenID, 
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

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->loadCRPen(CRPenID,
                                      numPenCRs,
                                      CRNodeTable,
                                      CRFieldList,
                                      CRWeightTable,
                                      CRValueList,
                                      penValues,
                                      lenCRNodeList);
    }
    else {
        notAllocatedAbort("FEI_ISIS::loadCRPen");
    }

    
    wTime_ += MPI_Wtime() - baseTime_;
    
    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::endLoadCREqns() {

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"endLoadCREqns\n");
        fflush(debugFile_);
    }

    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->endLoadCREqns();
    }
    else {
        notAllocatedAbort("FEI_ISIS::endLoadCREqns");
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::loadComplete() {

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"loadComplete, fei[%d]\n", index_current_fei_);
        fflush(debugFile_);
    }
    
    if (feiInternalAllocated_){
        feiInternal_[index_current_fei_]->loadComplete();
    }
    else {
        notAllocatedAbort("FEI_ISIS::loadComplete");
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadComplete\n");
        fflush(debugFile_);
    }
 
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int FEI_ISIS::getParam(const char *flag, int numParams,
                       char **paramStrings, char *param){
//
//  This is a private function. Used internally by FEI_ISIS only.
//  paramStrings is a collection of string pairs - each string in
//  paramStrings consists of two strings separated by a space.
//  This function looks through the strings in paramStrings, looking
//  for one that contains flag in the first string. The second string
//  is then returned in param.
//  Assumes that param is allocated by the calling code.
//

    int i;
    char temp[64];

    if (flag == 0 || paramStrings == 0)
        return(0); // flag or paramStrings is the NULL pointer

    for (i = 0; i<numParams; i++) {
        if (paramStrings[i] != 0)  { // check for NULL pointer
            if (strncmp(flag,paramStrings[i],strlen(flag)) == 0) {
                // flag found
                sscanf(paramStrings[i],"%s %s",temp,param);
                return(1);
            }
        }
    }
    return(0);  // flag was not found in paramStrings 
}

//------------------------------------------------------------------------------
void FEI_ISIS::appendParamStrings(int numStrings, char **strings){

    if (numParams_ == 0) {
        paramStrings_ = new char*[numStrings];

        for(int i=0; i<numStrings; i++){
            paramStrings_[i] = new char[strlen(strings[i])+1];

            strcpy(paramStrings_[i], strings[i]);
            paramStrings_[i][strlen(strings[i])] = '\0';
        }

        numParams_ = numStrings;
    }
    else {
        char **newStrTable = new char*[numParams_ + numStrings];
        int i;

        //first, copy the pre-existing string pointers into the
        //new table.
        for(i=0; i<numParams_; i++){
            newStrTable[i] = paramStrings_[i];
        }

        //now copy in the new strings
        for(i=numParams_; i<numParams_+numStrings; i++){
            newStrTable[i] = new char[strlen(strings[i-numParams_])+1];

            strcpy(newStrTable[i], strings[i-numParams_]);
            newStrTable[i][strlen(strings[i-numParams_])] = '\0';
        }

        //now delete the old table and set the pointer to the new one.
        delete [] paramStrings_;

        paramStrings_ = newStrTable;
        numParams_ += numStrings;
    }
}

//------------------------------------------------------------------------------
int FEI_ISIS::setMatScalars(int* IDs, double* scalars, int numScalars){

    int index, i;

    for(i=0; i<numScalars; i++){
        index = search_index(IDs[i], feiIDs_, numFeiInternal_);
        if (index>=0) {
            matScalars_[index] = scalars[i];
        }
        else {
            cerr << "FEI_ISIS::setMatScalars: ERROR, invalid ID supplied"
                 << endl;
            return(1);
        }
    }

    matScalarsSet_ = true;
    if (rhsScalarsSet_) {
        aggregateSystem();

        matScalarsSet_ = false;
        rhsScalarsSet_ = false;
    }

    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::setRHSScalars(int* IDs, double* scalars, int numScalars){

    int index, i;
    bool found;

    for(i=0; i<numScalars; i++){
        found = false;

        for(int j=0; j<numFeiInternal_; j++){
            index = search_index(IDs[i], rhsIDs_[j], numRHSIDs_[j]);
            if (index>=0) {
                rhsScalars_[j][index] = scalars[i];
                found = true;
                break;
            }
        }

        if (!found) {
            cerr << "FEI_ISIS::setRHSScalars: ERROR, invalid RHS ID supplied"
                 << endl;
            return(1);
        }
    }

    rhsScalarsSet_ = true;
    if (matScalarsSet_) {
        aggregateSystem();

        matScalarsSet_ = false;
        rhsScalarsSet_ = false;
    }

    return(0);
}

//------------------------------------------------------------------------------
void FEI_ISIS::parameters(int numParams, char **paramStrings) {
//
// this function takes parameters and passes them to the internal
// fei objects.
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"parameters\n");
        fflush(debugFile_);
    }
 
    if (feiInternalAllocated_){
        for(int i=0; i<numFeiInternal_; i++){
            feiInternal_[i]->parameters(numParams, paramStrings);
        }
    }

    if (numParams == 0 || paramStrings == NULL) {
        if (debugOutput_) {
            fprintf(debugFile_,"FEI_ISIS::parameters: no parameters.\n");
        }
    }
    else {
        // take a copy of these parameters, for later use.
        appendParamStrings(numParams, paramStrings);

        char param[64];
        if ( getParam("numMatrices",numParams,paramStrings,param) == 1)
            sscanf(param,"%d",&numFeiInternal_);

        if ( getParam("outputLevel",numParams,paramStrings,param) == 1){
            sscanf(param,"%d", &outputLevel_);
        }

        if ( getParam("debugOutput",numParams,paramStrings,param) == 1){
            setDebugOutput(param,"FEI_ISIS_debug");
        }

        if (debugOutput_) {
           fprintf(debugFile_,"FEI_ISIS::parameters: numParams %d\n",numParams);
           for(int i=0; i<numParams; i++){
               fprintf(debugFile_,"---paramStrings[%d]: %s\n",i,
                       paramStrings[i]);
           }
        }
    }

    wTime_ += MPI_Wtime() - baseTime_;

    if (debugOutput_) {
        fprintf(debugFile_,"leaving parameters function\n");
        fflush(debugFile_);
    }
 
    return;
}

//------------------------------------------------------------------------------
void FEI_ISIS::setDebugOutput(char* path, char* name){
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

    sprintf(dbFileName,"%s/%s.%d.%d", path,name,solveCounter_,localRank_);

    debugOutput_ = 1;
    debugFile_ = fopen(dbFileName,"w");
    if (!debugFile_){
        cerr << "couldn't open debug output file: " << dbFileName << endl;
        debugOutput_ = 0;
    }

    delete [] dbFileName;
}

//------------------------------------------------------------------------------
int FEI_ISIS::iterateToSolve() {

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"in iterateToSolve...\n");
        fflush(debugFile_);
    }
 
    buildLinearSystem();

    wTime_ += MPI_Wtime() - baseTime_;

    sTime_ = MPI_Wtime();

    int solveStatus = feiInternal_[index_soln_fei_]->iterateToSolve();

    sTime_ = MPI_Wtime() - sTime_;

    if (debugOutput_) {
        char name[64];
        Vector* x_tmp = NULL;
        feiInternal_[index_soln_fei_]->getSolnVectorPtr(&x_tmp);
        sprintf(name,"x_FEI_ISIS.txt.slv%d.np%d",solveCounter_, numProcs_);
        x_tmp->writeToFile(name);
    }

    if (solveType_ == 2) {
        delete A_solve_;
        A_solve_ = NULL;
        delete b_solve_;
        b_solve_ = NULL;
        aggregateSystemFormed_ = false;
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving iterateToSolve\n");
        fflush(debugFile_);
    }
 
    return(solveStatus);
}
             
//------------------------------------------------------------------------------
int FEI_ISIS::getBlockNodeSolution(GlobalID elemBlockID,  
                                   GlobalID *nodeIDList, 
                                   int &lenNodeIDList, 
                                   int *offset,  
                                   double *results) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"getBlockNodeSolution, -> fei[%d]\n",
                index_soln_fei_);
        fflush(debugFile_);
    }

    feiInternal_[index_soln_fei_]->getBlockNodeSolution(elemBlockID,
                                                       nodeIDList,
                                                       lenNodeIDList,
                                                       offset,
                                                       results);
    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::getBlockFieldNodeSolution(GlobalID elemBlockID,
                                        int fieldID,
                                        GlobalID *nodeIDList, 
                                        int& lenNodeIDList, 
                                        int *offset,
                                        double *results) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"getBlockFieldNodeSolution\n");
        fflush(debugFile_);
    }

    feiInternal_[index_soln_fei_]->getBlockFieldNodeSolution(elemBlockID,
                                                            fieldID,
                                                            nodeIDList,
                                                            lenNodeIDList,
                                                            offset,
                                                            results);
    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::putBlockNodeSolution(GlobalID elemBlockID,
                                   const GlobalID *nodeIDList, 
                                   int lenNodeIDList, 
                                   const int *offset,
                                   const double *estimates) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"putBlockNodeSolution\n");
    }

    feiInternal_[index_soln_fei_]->putBlockNodeSolution(elemBlockID,
                                                       nodeIDList,
                                                       lenNodeIDList,
                                                       offset,
                                                       estimates);
    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::putBlockFieldNodeSolution(GlobalID elemBlockID, 
                                        int fieldID, 
                                        const GlobalID *nodeIDList, 
                                        int lenNodeIDList, 
                                        const int *offset,
                                        const double *estimates) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"putBlockFieldNodeSolution\n");
        fflush(debugFile_);
    }

    feiInternal_[index_soln_fei_]->putBlockFieldNodeSolution(elemBlockID,
                                                            fieldID,
                                                            nodeIDList,
                                                            lenNodeIDList,
                                                            offset,
                                                            estimates);
    return(0);
}

//------------------------------------------------------------------------------
int FEI_ISIS::getBlockElemSolution(GlobalID elemBlockID,  
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
    feiInternal_[index_soln_fei_]->getBlockElemSolution(elemBlockID,
                                                       elemIDList,
                                                       lenElemIDList,
                                                       offset,
                                                       results,
                                                       numElemDOF)
    );
} 
      
//------------------------------------------------------------------------------
int FEI_ISIS::putBlockElemSolution(GlobalID elemBlockID,
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
    feiInternal_[index_soln_fei_]->putBlockElemSolution(elemBlockID,
                                                       elemIDList,
                                                       lenElemIDList,
                                                       offset,
                                                       estimates,
                                                       numElemDOF)
    );
}

//------------------------------------------------------------------------------
int FEI_ISIS::getCRMultSizes(int& numCRMultIDs, int& lenResults) {
//
//  This function returns the dimensions of the lists that get filled by
//  the getCRMultSolution function. In that function, *CRMultIDs and
//  *offset are both of length numCRMultIDs, and *results is of length
//  lenResults.
//

    return(
    feiInternal_[index_soln_fei_]->getCRMultSizes(numCRMultIDs,
                                                 lenResults)

    );
}

//------------------------------------------------------------------------------
int FEI_ISIS::getCRMultSolution(int& numCRMultSets, 
                                int *CRMultIDs,  
                                int *offset, 
                                double *results) {
        
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getCRMultSolution\n");
        fflush(debugFile_);
    }

    return(
    feiInternal_[index_soln_fei_]->getCRMultSolution(numCRMultSets,
                                                    CRMultIDs,
                                                    offset,
                                                    results)
    );
} 

//------------------------------------------------------------------------------
int FEI_ISIS::getCRMultParam(int CRMultID, 
                             int numMultCRs,
                             double *multValues) {

    if (debugOutput_) {
        fprintf(debugFile_,"trace: getCRMultParam\n");
        fflush(debugFile_);
    }

    return(
    feiInternal_[index_soln_fei_]->getCRMultParam(CRMultID,
                                                 numMultCRs,
                                                 multValues)
    );
}


//------------------------------------------------------------------------------
int FEI_ISIS::putCRMultParam(int CRMultID, 
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
    feiInternal_[index_soln_fei_]->putCRMultParam(CRMultID,
                                                 numMultCRs,
                                                 multEstimates)
    );
}


//-----------------------------------------------------------------------------
//  some utility functions to aid in using the "put" functions for passing
//  an initial guess to the solver
//-----------------------------------------------------------------------------

//------------------------------------------------------------------------------
int FEI_ISIS::getBlockElemIDList(GlobalID elemBlockID,
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
    feiInternal_[index_soln_fei_]->getBlockElemIDList(elemBlockID,
                                                     elemIDList,
                                                     lenElemIDList)
    );
}

//------------------------------------------------------------------------------
int FEI_ISIS::getBlockNodeIDList(GlobalID elemBlockID,
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
    feiInternal_[index_soln_fei_]->getBlockNodeIDList(elemBlockID,
                                                     nodeIDList,
                                                     lenNodeIDList)
    );
}

//------------------------------------------------------------------------------
int FEI_ISIS::getNumNodesPerElement(GlobalID blockID) const {
//
//  return the number of nodes associated with elements of a given block ID
//

    return(
    feiInternal_[index_soln_fei_]->getNumNodesPerElement(blockID)
    );
}
 
 
//------------------------------------------------------------------------------
int FEI_ISIS::getNumEqnsPerElement(GlobalID blockID) const {
//
//  return the number of eqns associated with elements of a given block ID
//

    return(
    feiInternal_[index_soln_fei_]->getNumEqnsPerElement(blockID)
    );
}


//------------------------------------------------------------------------------
int FEI_ISIS::getNumSolnParams(GlobalID iGlobal) const {
//
//  return the number of solution parameters at a given node
//

    return(
    feiInternal_[index_soln_fei_]->getNumSolnParams(iGlobal)
    );
}
 
 
//------------------------------------------------------------------------------
int FEI_ISIS::getNumElemBlocks() const {
//
//  return the number of element blocks
//

    return(
    feiInternal_[index_soln_fei_]->getNumElemBlocks()
    );
}

//------------------------------------------------------------------------------
int FEI_ISIS::getNumBlockActNodes(GlobalID blockID) const {
//
//  return the number of active nodes associated with a given element block ID
//

    return(
    feiInternal_[index_soln_fei_]->getNumBlockActNodes(blockID)
    );
}


//------------------------------------------------------------------------------
int FEI_ISIS::getNumBlockActEqns(GlobalID blockID) const {
//
// return the number of active equations associated with a given element
// block ID
//

    return(
    feiInternal_[index_soln_fei_]->getNumBlockActEqns(blockID)
    );
}

//------------------------------------------------------------------------------
int FEI_ISIS::getNumBlockElements(GlobalID blockID) const {
//
//  return the number of elements associated with a given elem blockID
//

    return(
    feiInternal_[index_soln_fei_]->getNumBlockElements(blockID)
    );
}


//------------------------------------------------------------------------------
int FEI_ISIS::getNumBlockElemEqns(GlobalID blockID) const {
//
//  return the number of elem equations associated with a given elem blockID
//

    return(
    feiInternal_[index_soln_fei_]->getNumBlockElemEqns(blockID)
    );
}

//------------------------------------------------------------------------------
void FEI_ISIS::buildLinearSystem(){
//
//At the point when this function is called, all matrix assembly has
//already taken place, with the data having been directed into the
//appropriate ISIS_SLE instance in the feiInternal_ list. Now it's
//time to get pointers and build a matrix A and vectors x and b
//to give to a solver.
//

    if (solveType_ == 0) {
        //solveType_ == 0 means this is just a standard, single, Ax=b solve.

        if (debugOutput_) {
            char name[64];

            sprintf(name,"A_FEI_ISIS.mtx.slv%d.np%d",
                    solveCounter_, numProcs_);

            feiInternal_[index_soln_fei_]->getISISMatrixPtr(&A_solve_);
//            A_solve_->writeToFile(name);

            sprintf(name,"b_FEI_ISIS.txt.slv%d.np%d",
                    solveCounter_, numProcs_);

            feiInternal_[index_soln_fei_]->getRHSVectorPtr(&b_solve_);
            b_solve_->writeToFile(name);
        }
    }

    if (solveType_ == 2){
        //solveType_ == 2 means this is a linear-combination solve --
        //i.e., we're solving an aggregate system which is the sum of
        //several individual matrices and rhs's.

        if (!aggregateSystemFormed_) {
            cerr << "FEI_ISIS: WARNING: solveType_==2, but aggregate system"
                  << " hasn't been formed before solve requested." << endl;
            aggregateSystem();
        }
    }
}

//------------------------------------------------------------------------------
void FEI_ISIS::aggregateSystem() { 

    Vector* tmpv = NULL;
    feiInternal_[index_soln_fei_]->getSolnVectorPtr(&tmpv);

    if (soln_fei_matrix_ptr_ == NULL) {
        feiInternal_[index_soln_fei_]->getISISMatrixPtr(&soln_fei_matrix_ptr_);
    }
    const Map& map = soln_fei_matrix_ptr_->getMap();

    if (soln_fei_vector_ptr_ == NULL) {
        feiInternal_[index_soln_fei_]->setRHSID(rhsIDs_[index_soln_fei_][0]);
        feiInternal_[index_soln_fei_]->getRHSVectorPtr(&soln_fei_vector_ptr_);
    }

    if (A_solve_) delete A_solve_;
    A_solve_ = new DCRS_Matrix(map);

    if (b_solve_) delete b_solve_;
    b_solve_ = tmpv->newVector();
    b_solve_->put(0.0);

    DCRS_Matrix* tmp = NULL;

    for(int i=0; i<numFeiInternal_; i++){

        if (i != index_soln_fei_)
            feiInternal_[i]->getISISMatrixPtr(&tmp);
        else
            tmp = soln_fei_matrix_ptr_;

        if (debugOutput_) {
            char name[64];
            sprintf(name,"A_FEI_ISIS.mtx.slv%d.np%d.mat%d",
                    solveCounter_, numProcs_, i);
//            tmp->writeToFile(name);
        }

        if (i==0) {
            A_solve_->copyScaledMatrix(matScalars_[i], *tmp);
        }
        else {
            A_solve_->addScaledMatrix(matScalars_[i], *tmp);
        }

        for(int j=0; j<numRHSIDs_[i]; j++){
            if ((i == index_soln_fei_) && (j == 0)) {
                tmpv = soln_fei_vector_ptr_;
            }
            else {
                feiInternal_[i]->setRHSID(rhsIDs_[i][j]);
                feiInternal_[i]->getRHSVectorPtr(&tmpv);
            }

            if (debugOutput_) {
                char name[64];
                sprintf(name,"b_FEI_ISIS.txt.slv%d.np%d.mat%d.rhs%d",
                        solveCounter_, numProcs_, i, j);
                tmpv->writeToFile(name);
            }

            b_solve_->addVec(rhsScalars_[i][j], *tmpv);
        }
    }

    aggregateSystemFormed_ = true;

    if (debugOutput_) {
        char name[64];
        sprintf(name,"A_FEI_ISIS.mtx.slv%d.np%d.agg",
                solveCounter_, numProcs_);
//        A_solve_->writeToFile(name);
    }

    feiInternal_[index_soln_fei_]->setMatrixPtr(A_solve_);
    feiInternal_[index_soln_fei_]->setRHSPtr(b_solve_);
}

//==============================================================================
void FEI_ISIS::allocateFeiInternals(int numMatrices, int* matrixIDs,
                                    int* numRHSs, int** rhsIDs){
//
    numFeiInternal_ = numMatrices;

    feiIDs_ = new int[numMatrices];

    numRHSIDs_ = new int[numMatrices];
    rhsIDs_ = new int*[numMatrices];

    matScalars_ = new double[numMatrices];
    rhsScalars_ = new double*[numMatrices];

    for(int i=0; i<numFeiInternal_; i++){
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

    allocateFeiInternals();
}

//==============================================================================
void FEI_ISIS::allocateFeiInternals(){
//
//This is a private FEI_ISIS function, to be called from within initSolveStep.
//Assumes that numFeiInternal_ has already been set.
//

    if (feiInternalAllocated_) return;

    if (numFeiInternal_ > 0) {
        index_soln_fei_ = 0;
        index_current_fei_ = 0;
        feiInternal_ = new ISIS_SLE*[numFeiInternal_];

        if (!IDsAllocated_) {
            //if IDsAllocated_ is false, then initSolveStep was called without
            //the matrixIDs and rhsIDs arguments. So we're going to assume that
            //there is only 1 rhs per matrix, and IDs for matrices and rhs's
            //will just be 0-based indices.
            //

            feiIDs_ = new int[numFeiInternal_];
            numRHSIDs_ = new int[numFeiInternal_];
            rhsIDs_ = new int*[numFeiInternal_];

            matScalars_ = new double[numFeiInternal_];
            rhsScalars_ = new double*[numFeiInternal_];

            for(int i=0; i<numFeiInternal_; i++){
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

        char *param = new char[32];
        for(int i=0; i<numFeiInternal_; i++){
            feiInternal_[i] = new ISIS_SLE(FEI_COMM_WORLD, masterRank_);

            sprintf(param, "internalFei %d", i);
            feiInternal_[i]->parameters(1, &param);

            feiInternal_[i]->parameters(numParams_, paramStrings_);
        }

        delete [] param;

        feiInternalAllocated_ = true;
    }
    else {
        needParametersAbort("FEI_ISIS::allocateFeiInternals");
    }
}

//==============================================================================
void FEI_ISIS::messageAbort(char* msg){

    cerr << "FEI_ISIS: ERROR " << msg << " Aborting." << endl;
    abort();
}

//==============================================================================
void FEI_ISIS::notAllocatedAbort(char* name){

    cerr << name
         << endl << "ERROR, internal data structures not allocated."
         << endl << "'initSolveStep' and/or 'parameters' must be called"
         << endl << "up front to identify solveType and number of matrices"
         << endl << "to be assembled." << endl;
    abort();
}

//==============================================================================
void FEI_ISIS::needParametersAbort(char* name){

    cerr << name
         << endl << "FEI_ISIS: ERROR, numMatrices has not been specified."
         << endl << "FEI_ISIS: 'parameters' must be called up front with"
         << endl << "FEI_ISIS: the string 'numMatrices n' to specify that"
         << endl << "FEI_ISIS: n matrices will be assembled." << endl;
    abort();
}

//==============================================================================
void FEI_ISIS::badParametersAbort(char* name){

    cerr << name
         << endl << "FEI_ISIS: ERROR, inconsistent 'solveType' and"
         << endl << "FEI_ISIS: 'numMatrices' parameters specified."
         << endl << "FEI_ISIS: Aborting."
         << endl;
    abort();
}

