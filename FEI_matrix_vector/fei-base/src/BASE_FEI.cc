#include <math.h>
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

#include "../isis-mv/RealArray.h"
#include "../isis-mv/IntArray.h"
#include "../isis-mv/GlobalIDArray.h"

#include "src/BCRecord.h"
#include "src/BCManager.h"
#include "src/FieldRecord.h"
#include "src/NodeDescriptor.h"
#include "src/NodeCommMgr.h"
#include "src/ProcEqns.h"
#include "src/EqnBuffer.h"
#include "src/EqnCommMgr.h"
#include "src/BlockDescriptor.h"
#include "src/ProblemStructure.h"
#include "src/MultConstRecord.h"
#include "src/PenConstRecord.h"
#include "src/SLE_utils.h"
#include "src/Utils.h"

#include "src/Data.h"
#include "src/LinearSystemCore.h"

#include "src/BASE_FEI.h"

//------------------------------------------------------------------------------
BASE_FEI::BASE_FEI(MPI_Comm comm, LinearSystemCore* linSysCore, int masterRank)
 : linSysCore_(linSysCore),
   internalFei_(0),
   solveCounter_(1),
   debugFileName_(NULL),
   debugPath_(NULL),
   debugOutput_(0),
   debugFile_(NULL),
   numParams_(0),
   paramStrings_(NULL),
   localStartRow_(0),             //
   localEndRow_(0),               //Initialize all private variables here,
   numLocalEqns_(0),              //in the order that they are declared.
   iterations_(0),                //
   numRHSs_(0),
   currentRHS_(0),
   rhsIDs_(NULL),
   outputLevel_(0),
   wTime_(0.0),
   sTime_(0.0),
   storeNumCRMultRecords(0),
   ceqn_MultConstraints(NULL),
   comm_(comm),
   masterRank_(masterRank),
   localRank_(-1),
   numProcs_(0),
   numWorksets(NULL),
   numWorksetsStored(NULL),
   numMatrixWorksetsStored(NULL),
   numRHSWorksetsStored(NULL),
   nextElemIndex(NULL),
   currentBlock(NULL),
   problemStructure_(NULL),
   problemStructureAllocated_(false),
   problemStructureSet_(false),
   matrixAllocated_(false),
   bcManager_(NULL)
{

//  start the wall clock time recording

    baseTime_ = MPI_Wtime();

//  initialize a couple of MPI things

    MPI_Comm_rank(comm_, &localRank_);
    MPI_Comm_size(comm_, &numProcs_);

    numRHSs_ = 1;
    rhsIDs_ = new int[numRHSs_];
    rhsIDs_[0] = 0;

//  initialize some state variables

    storeNumProcActNodes = 0;  // number of active nodes on this processor
    storeNumProcActEqns = 0;   // number of equations arising from active nodes
    storeBCNodeSets = 0;       // number of bc node sets
    storeSharedNodeSets = 0;   // number of shared node sets
    storeExtNodeSets = 0;      // number of external node sets
    storeNumCRPenRecords = 0;  // number of penalty constraint records

//  some data for consistency checking (some of these not yet utilized)

    checkElemBlocksLoaded = 0;
    checkNumElemBlocks = 0;    // verify number of blocks in this problem
    checkNumProcActNodes = 0;  // verify number of active nodes on this proc
    checkNumProcActEqns = 0;   // verify number of equations
                               // arising from active nodes
    checkBCNodeSets = 0;       // verify number of bc node sets
    checkSharedNodeSets = 0;   // verify number of shared node sets
    checkExtNodeSets = 0;      // verify number of external node sets
    checkNumCRMultRecords = 0; // verify number of Lagrange constraint records
    checkNumCRPenRecords = 0;  // verify number of penalty constraint records
    doneEndInitElemData = 0;

    bcManager_ = new BCManager();

//  and the time spent in the constructor is...

    wTime_  = MPI_Wtime() - baseTime_;

    return;
}

//------------------------------------------------------------------------------
BASE_FEI::~BASE_FEI() {
//
//  Destructor function. Free allocated memory, etc.
//

    int i;

    delete [] rhsIDs_;
    numRHSs_ = 0;

    if (problemStructureAllocated_) delete problemStructure_;

    if (storeNumCRMultRecords > 0) 
        delete [] ceqn_MultConstraints;
    if (storeNumCRPenRecords > 0) 
        delete [] ceqn_PenConstraints;

    delete [] numWorksets;
    delete [] numWorksetsStored;
    delete [] numMatrixWorksetsStored;
    delete [] numRHSWorksetsStored;
    delete [] nextElemIndex;

    for(i=0; i<numParams_; i++) delete [] paramStrings_[i];
    delete [] paramStrings_;

    if (debugOutput_) {
        delete [] debugPath_;
        delete [] debugFileName_;
        fclose(debugFile_);
    }

    delete bcManager_;
}

//------------------------------------------------------------------------------
int BASE_FEI::initSolveStep(int numElemBlocks, int solveType) {
//
//  tasks: allocate baseline data structures for elements and nodes
//

    baseTime_ = MPI_Wtime();

    (void)solveType;

    debugOutput("initSolveStep");

    if (debugOutput_) {
       fprintf(debugFile_, "   numElemBlocks: %d, solveType: %d\n",
               numElemBlocks, solveType);
       fflush(debugFile_);
    }

    if (!problemStructureAllocated_) {
        problemStructure_ = new ProblemStructure(localRank_);
        problemStructureAllocated_ = true;
    }

    problemStructure_->setNumBlocks(numElemBlocks);

//  next, allocate some consistency-checking arrays

    allocateWorksetChecks(numElemBlocks);

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//==============================================================================
void BASE_FEI::allocateWorksetChecks(int numElemBlocks) {

    numWorksets = new int[numElemBlocks];
    numWorksetsStored = new int[numElemBlocks];
    numMatrixWorksetsStored = new int[numElemBlocks];
    numRHSWorksetsStored = new int[numElemBlocks];
    nextElemIndex = new int[numElemBlocks];

    for(int i=0; i<numElemBlocks; i++) {
       numWorksets[i] = 0;
       numWorksetsStored[i] = 0;
       numMatrixWorksetsStored[i] = 0;
       numRHSWorksetsStored[i] = 0;
       nextElemIndex[i] = 0;
    }
}

//------------------------------------------------------------------------------
int BASE_FEI::initSolveStep(int numElemBlocks, int solveType,
                            int numMatrices, int* matrixIDs,
                            int* numRHSs, int** rhsIDs) {

    (void)numMatrices;
    (void)matrixIDs;
    (void)numRHSs;
    (void)rhsIDs;

    return(
    initSolveStep(numElemBlocks, solveType)
    );
}

//------------------------------------------------------------------------------
int BASE_FEI::initFields(int numFields, 
                         const int *cardFields, 
                         const int *fieldIDs) {
//
//  tasks: identify all the solution fields present in the analysis
//

    baseTime_ = MPI_Wtime();

    debugOutput("initFields");

    if (debugOutput_) {
       fprintf(debugFile_, "   numFields: %d\n", numFields);
       for(int i=0; i<numFields; i++) {
          fprintf(debugFile_,
                  "      cardFields[%d] (fieldSize): %d, fieldIDs[%d], %d\n",
                  i, cardFields[i], i, fieldIDs[i]);
       }
       fflush(debugFile_);
    }

    assert (numFields > 0);

    if (!problemStructureAllocated_) {
        problemStructure_ = new ProblemStructure(localRank_);
        problemStructureAllocated_ = true;
    }

    problemStructure_->setNumFields(numFields);

    FieldRecord* fieldRosterPtr = problemStructure_->getFieldRosterPtr();

    for (int i = 0; i < numFields; i++) {
        fieldRosterPtr[i].setFieldID(fieldIDs[i]);
        fieldRosterPtr[i].setNumFieldParams(cardFields[i]);
//$kdm debug        fieldRosterPtr[i].dumpToScreen();
    }
    
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//==============================================================================
int BASE_FEI::beginInitElemBlock(GlobalID elemBlockID,  
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

    (void)lumpingStrategy;

    debugOutput("beginInitElemBlock");

    if (debugOutput_) {
       fprintf(debugFile_, "   elemBlockID: %d\n", (int)elemBlockID);
       fprintf(debugFile_, "    numNodesPerElement: %d\n", numNodesPerElement);
       for(int i=0; i<numNodesPerElement; i++) {
          fprintf(debugFile_, "      fieldsPerNode[%d]: %d, fieldIDs: ",
                  i, numElemFields[i]);
          for(int j=0; j<numElemFields[i]; j++) {
             fprintf(debugFile_, "%d ", elemFieldIDs[i][j]);
          }
          fprintf(debugFile_, "\n");
       }
       fprintf(debugFile_, "   interleaveStrategy: %d\n", interleaveStrategy);
       fprintf(debugFile_, "   lumpingStrategy: %d\n", lumpingStrategy);
       fprintf(debugFile_, "   numElemDOF: %d\n", numElemDOF);
       fprintf(debugFile_, "   numElemSets: %d\n", numElemSets);
       fprintf(debugFile_, "   numElemTotal: %d\n", numElemTotal);
       fflush(debugFile_);
    }

    int j;

    problemStructure_->addBlockID(elemBlockID);
    currentBlock = &(problemStructure_->getBlockDescriptor(elemBlockID));

    currentBlock->setGlobalBlockID(elemBlockID);
    currentBlock->setNumNodesPerElement(numNodesPerElement);
    currentBlock->setNumElements(numElemTotal);
    currentBlock->setNumElemDOFPerElement(numElemDOF);
    currentBlock->setInterleaveStrategy(interleaveStrategy);
    assert (interleaveStrategy == 0);   // only support NODE_CENTRIC for now...

    int *fieldsPerNodePtr = currentBlock->fieldsPerNodePtr();

    FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

//  construct the list of nodal solution cardinalities for this block

    int myNodalEqns = 0;
    int fieldIndex;
    int countDOF;
    for (j = 0; j < numNodesPerElement; j++) {

        countDOF = 0;
        for(int k = 0; k < numElemFields[j]; k++) {
            fieldIndex = problemStructure_->
                             getFieldRosterIndex(elemFieldIDs[j][k]);
            assert (fieldIndex >= 0);
            countDOF += fieldRoster[fieldIndex].getNumFieldParams();
        }

        fieldsPerNodePtr[j] = numElemFields[j];
        myNodalEqns += countDOF;
    }

    currentBlock->setNumEqnsPerElement(myNodalEqns + numElemDOF);

//  cache a copy of the element fields array for later use...

    currentBlock->allocateFieldIDsTable();
    int **fieldIDsTablePtr = currentBlock->fieldIDsTablePtr();

    for (j = 0; j < numNodesPerElement; j++) {
        for(int k = 0; k < numElemFields[j]; k++) {
            fieldIDsTablePtr[j][k] = elemFieldIDs[j][k];
        }
    }
    
//  create data structures for storage of element ID and topology info

    if (numElemTotal > 0)
       problemStructure_->allocateConnectivityTable(elemBlockID);

    if (numElemDOF > 0) {
       currentBlock->setNumElemDOFPerElement(numElemDOF);
    }

    currentElemBlockIndex = problemStructure_->getBlockIndex(elemBlockID);

    numWorksets[currentElemBlockIndex] = numElemSets;

    checkNumElemBlocks++;
    currentElemBlockID = elemBlockID;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_FEI::initElemSet(int numElems, 
                          const GlobalID *elemIDs, 
                          const GlobalID *const *elemConn) {
//
//  task: store element connectivities for later use in determining sparsity
//         pattern (should reuse the space required by these stored parameters
//         to simplify assembly later on?).
//

    baseTime_ = MPI_Wtime();

    debugOutput("initElemSet");

    int numNodes = currentBlock->getNumNodesPerElement();

    if (debugOutput_) {
        fprintf(debugFile_,"   numElems: %d    (currentBlockID: %d)\n",
                numElems, currentElemBlockID);
        for(int i=0; i<numElems; i++) {
           fprintf(debugFile_, "      elemID[%d]: %d\n", i, (int)elemIDs[i]);
           fprintf(debugFile_, "         nodes: ");
           for(int j=0; j<numNodes; j++) {
              fprintf(debugFile_, "%d ", (int)elemConn[i][j]);
           }
           fprintf(debugFile_, "\n");
        }
        fflush(debugFile_);
    }

    numWorksetsStored[currentElemBlockIndex]++;
    
    if (numElems <= 0) return(0); //zero-length workset, do nothing.

//
//  the "start" variable gives us the offset to the next open index in the
//  connectivity table for this block.
//
    int start = nextElemIndex[currentElemBlockIndex];

    ConnectivityTable& connTable =
                problemStructure_->getConnectivityTable(currentElemBlockID);

    GlobalID *elemIDList = connTable.elemIDs;
    GlobalID **conn = connTable.connectivities;

    int elemIndex;
    for (int i = 0; i < numElems; i++) {
        elemIndex = start + i;
        assert (elemIndex < connTable.numElems); // debug - don't overrun list
        elemIDList[elemIndex] = elemIDs[i];
        for (int j = 0; j < numNodes; j++) {
            conn[elemIndex][j] = elemConn[i][j];

            flagNodeAsActive(elemConn[i][j]);
        }
    }

//  now cache an index for use by the next element set

    nextElemIndex[currentElemBlockIndex] = elemIndex+1;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::endInitElemBlock() {
//
//  tasks: check to insure consistency of data
//

    baseTime_ = MPI_Wtime();

    debugOutput("endInitElemBlock");

    assert (numWorksetsStored[currentElemBlockIndex] ==
            numWorksets[currentElemBlockIndex]);
 
    if (debugOutput_) {
        fprintf(debugFile_,"elemBlock %d, numWorksetsStored %d\n",
                (int)currentElemBlockID,
                numWorksetsStored[currentElemBlockIndex]);
    }
    
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_FEI::beginInitNodeSets(int numSharedNodeSets, 
                                int numExtNodeSets) {
//
//  tasks: simply set the number of shared node sets and external node sets
//         that are to be identified.
//

    baseTime_ = MPI_Wtime();

    debugOutput("beginInitNodeSets");

    if (debugOutput_) {
        fprintf(debugFile_,"   numSharedNodeSets: %d, numExtNodeSets: %d\n",
                numSharedNodeSets, numExtNodeSets);
        fflush(debugFile_);
    }

//  store the node set parameters

    storeSharedNodeSets = numSharedNodeSets;
    currentSharedNodeSet = 0;
 
    storeExtNodeSets = numExtNodeSets;
    currentExtNodeSet = 0;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::initSharedNodeSet(const GlobalID *sharedNodeIDs,  
                                int lenSharedNodeIDs, 
                                const int *const *sharedProcIDs, 
                                const int *lenSharedProcIDs) {
//
//  In this function we simply accumulate the incoming data into the
//  ProblemStructure::NodeCommMgr object.
//

    baseTime_ = MPI_Wtime();

    debugOutput("initSharedNodeSet");
 
    if (debugOutput_) {
        fprintf(debugFile_,"   numSharedNodes: %d\n", lenSharedNodeIDs);
        for(int i=0; i<lenSharedNodeIDs; i++){
            fprintf(debugFile_, "   sharedNode[%d]: %d procs: ",i,
                    (int)sharedNodeIDs[i]);
            for(int j=0; j<lenSharedProcIDs[i]; j++) {
               fprintf(debugFile_, "%d ", sharedProcIDs[i][j]);
            }
            fprintf(debugFile_,"\n");
        }
        fflush(debugFile_);
    }

    NodeCommMgr& commMgr = problemStructure_->getNodeCommMgr();

    bool externNodes = false;

    commMgr.addCommNodes(externNodes, sharedNodeIDs, lenSharedNodeIDs,
                         sharedProcIDs, lenSharedProcIDs);

    currentSharedNodeSet++;
    checkSharedNodeSets++;
 
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::initExtNodeSet(const GlobalID *extNodeIDs,
                             int lenExtNodeIDs, 
                             const int *const *extProcIDs,
                             const int *lenExtProcIDs) {
//
// store the input parameters in the ProblemStructure::NodeCommMgr object...
//

    baseTime_ = MPI_Wtime();

    debugOutput("initExtNodeSet");
 
    if (debugOutput_) {
        fprintf(debugFile_,"   numExtNodes: %d\n", lenExtNodeIDs);
        for(int i=0; i<lenExtNodeIDs; i++){
            fprintf(debugFile_, "   extNode[%d]: %d procs: ",i,
                    (int)extNodeIDs[i]);
            for(int j=0; j<lenExtProcIDs[i]; j++) {
               fprintf(debugFile_, "%d ", extProcIDs[i][j]);
            }
            fprintf(debugFile_,"\n");
        }
        fflush(debugFile_);
    }

    NodeCommMgr& commMgr = problemStructure_->getNodeCommMgr();

    bool externNodes = true;

    commMgr.addCommNodes(externNodes, extNodeIDs, lenExtNodeIDs,
                         extProcIDs, lenExtProcIDs);

    currentExtNodeSet++;
    checkExtNodeSets++;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::endInitNodeSets() {
//
//  tasks: check to insure consistency of data (e.g., number of
//         passed lists equals number given in initSolveStep).
//

    baseTime_ = MPI_Wtime();

    debugOutput("endInitNodeSets");

    assert (checkBCNodeSets == storeBCNodeSets);
    assert (checkSharedNodeSets == storeSharedNodeSets);
    assert (checkExtNodeSets == storeExtNodeSets);
    
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::beginInitCREqns(int numCRMultRecords, 
                              int numCRPenRecords) {
//
//  tasks: allocate baseline data for the constraint relations
//

    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"beginInitCREqns, numCRMultRecords: %d",
                numCRMultRecords);
        fprintf(debugFile_,", numCRPenRecords: %d\n",
                numCRPenRecords);
        fflush(debugFile_);
    }

    storeNumCRMultRecords = numCRMultRecords;
    storeNumCRPenRecords = numCRPenRecords;

    if (storeNumCRMultRecords > 0)
        ceqn_MultConstraints = new MultConstRecord [numCRMultRecords];
    
    if (storeNumCRPenRecords > 0)
        ceqn_PenConstraints = new PenConstRecord [numCRPenRecords];

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::initCRMult(const GlobalID *const *CRMultNodeTable,
                         const int *CRFieldList,
                         int numMultCRs, 
                         int lenCRNodeList,
                         int& CRMultID) {
//
//  tasks: convert Lagrange constraint data into constraintEqnBank
//         format, and update localEqnBank data appropriately (i.e.,
//         add another local constraint equations, determine sparsity
//         patterns for the new constraint relation)
//

    baseTime_ = MPI_Wtime();

    debugOutput("initCRMult");

    if (debugOutput_) {
       fprintf(debugFile_, "   numMultCRs: %d\n", numMultCRs);
       fprintf(debugFile_, "   node table:\n");
       int i;
       for(i=0; i<numMultCRs; i++) {
          fprintf(debugFile_, "      ");
          for(int j=0; j<lenCRNodeList; j++) {
             fprintf(debugFile_, "%d ", (int)CRMultNodeTable[i][j]);
          }
          fprintf(debugFile_, "\n");
       }
       fprintf(debugFile_, "   field list:\n      ");
       for(i=0; i<lenCRNodeList; i++)
          fprintf(debugFile_, "%d ", CRFieldList[i]);
       fprintf(debugFile_, "\n");
       fflush(debugFile_);
    }

    int i, j, k;

    k = checkNumCRMultRecords;
    CRMultID = k;
    ceqn_MultConstraints[k].setCRMultID(checkNumCRMultRecords);
    ceqn_MultConstraints[k].setLenCRNodeList(lenCRNodeList);
    ceqn_MultConstraints[k].setNumMultCRs(numMultCRs);
    ceqn_MultConstraints[k].allocateCRFieldList(lenCRNodeList);
    ceqn_MultConstraints[k].allocateCRNodeTable(numMultCRs, lenCRNodeList);
    
    int rowCheck = 0;
    int colCheck = 0;
    GlobalID **CRNodeTablePtr =
        ceqn_MultConstraints[k].pointerToCRNodeTable(rowCheck, colCheck);
    assert (rowCheck == numMultCRs);
    assert (colCheck == lenCRNodeList);
    for (i = 0; i < numMultCRs; i++) {
        for (j = 0; j < lenCRNodeList; j++) {
            CRNodeTablePtr[i][j] = CRMultNodeTable[i][j];
        }
    }

    int *CRFieldListPtr =
        ceqn_MultConstraints[k].pointerToCRFieldList(rowCheck);
    assert (rowCheck == lenCRNodeList);
    for (i = 0; i < lenCRNodeList; i++) {
        CRFieldListPtr[i] = CRFieldList[i];
    }

//    ceqn_MultConstraints[k].dumpToScreen();   //$kdm debugging

    checkNumCRMultRecords++;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
//
//  tasks: convert penalty constraint data into constraintEqnBank
//         format, and update localEqnBank data appropriately (i.e.,
//         don't add new local equations, but update sparsity patterns
//         arising from previous nodal terms to account for penalty
//         energy contributions).
//
int BASE_FEI::initCRPen(const GlobalID *const *CRPenNodeTable, 
                        const int *CRFieldList,
                        int numPenCRs, 
                        int lenCRNodeList,
                        int& CRPenID) {

    baseTime_ = MPI_Wtime();

    debugOutput("initCRPen");

    int i, j, k;
    int myNumPenCRs, myLenCRNodeList;

    k = checkNumCRPenRecords;
    CRPenID = k;
    ceqn_PenConstraints[k].setCRPenID(checkNumCRPenRecords);
    ceqn_PenConstraints[k].setLenCRNodeList(lenCRNodeList);
    ceqn_PenConstraints[k].setNumPenCRs(numPenCRs);
    ceqn_PenConstraints[k].allocateCRFieldList(lenCRNodeList);
    ceqn_PenConstraints[k].allocateCRNodeTable(numPenCRs,lenCRNodeList);
    
    GlobalID **CRNodeTablePtr = ceqn_PenConstraints[k].
        pointerToCRNodeTable(myNumPenCRs, myLenCRNodeList);
    assert(numPenCRs == myNumPenCRs);
    assert(lenCRNodeList == myLenCRNodeList);

    for (i = 0; i < myNumPenCRs; i++) {
        for (j = 0; j < myLenCRNodeList; j++) {
            CRNodeTablePtr[i][j] = CRPenNodeTable[i][j];
        }
    }

    int *CRFieldListPtr =
        ceqn_PenConstraints[k].pointerToCRFieldList(myLenCRNodeList);
    assert (myLenCRNodeList == lenCRNodeList);
    for (i = 0; i < lenCRNodeList; i++) {
        CRFieldListPtr[i] = CRFieldList[i];
    }

    checkNumCRPenRecords++;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
//
//  tasks: check consistency of constraint equation data.
//
int BASE_FEI::endInitCREqns() {

    debugOutput("endInitCREqns");

    assert (checkNumCRMultRecords == storeNumCRMultRecords);
    assert (checkNumCRPenRecords == storeNumCRPenRecords);

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_FEI::initComplete() {
//
//  tasks: determine final sparsity pattern for use in allocating memory
//         for sparse matrix storage in preparation for assembling
//         element and constraint data.
//
//         allocate storage for upcoming assembly of element terms
//

   baseTime_ = MPI_Wtime();

   debugOutput("initComplete");

   //first, have the ProblemStructure class marshall all of the nodes we
   //received in element connectivities into an active node list, and set
   //the fields and blocks associated with those nodes.

   problemStructure_->initializeActiveNodes();

   //now, call initComplete() on the nodeCommMgr, so that it can
   //allocate a list of NodeDescriptor pointers, into which we'll be
   //be putting useful information like the fieldIDs and eqn numbers for
   //each of those nodes. 

   problemStructure_->getNodeCommMgr().initComplete();

   setActiveNodeOwnerProcs();

   // ok, now the active equation calculations -- we need to count how many
   // local equations there are, then obtain the global equation info, then
   // set equation numbers on all of our active nodes.

   numLocalEqns_ = countActiveNodeEqns();

   //now, lets add the element-DOF to the numLocalEqns_ count...

   int totalNumElemDOF = calcTotalNumElemDOF();
   numLocalEqns_ += totalNumElemDOF;

   //  add equations for any constraint relations here

   int i;
   for (i = 0; i < storeNumCRMultRecords; i++) {
      numLocalEqns_ += ceqn_MultConstraints[i].getNumMultCRs();
   }

   // now, obtain the global equation info, such as how many equations there
   // are globally, and what the local starting and ending row-numbers are.

   getGlobalEqnInfo(numLocalEqns_,
                    numGlobalEqns_, localStartRow_, localEndRow_);

   //now, we can run through the list of active nodes and assign equation
   //numbers to them. Global equation numbers, by the way.

   int numActNodeEqns = setActiveNodeEqnInfo();

   //now we need to set equation numbers for the element-DOF's, and for the
   //lagrange-multipler constraints.
   //(exactly where to put these element DOF is an interesting issue... here, 
   //just toss them at the end of the nodal active eqns, which may or may not
   //be such a great choice.)

   setElemDOFEqnInfo(numActNodeEqns);

   int eqnNumber = localStartRow_  + numActNodeEqns + totalNumElemDOF;

   for(i=0; i<storeNumCRMultRecords; i++) {
      ceqn_MultConstraints[i].setEqnNumber(eqnNumber);
      eqnNumber += ceqn_MultConstraints[i].getNumMultCRs();
   }

   //--------------------------------------------------------------------------
   //  ----- end active equation calculations -----

   //  ok; now we know which equation number is associated with each
   //  local node, each element-DOF, and each lagrange multiplier constraint.
   //
   //  next, we need to have the node comm manager get the field IDs and 
   //  equation numbers for all of the nodes that we know about but don't
   //  own. i.e., the remotely-owned shared and external nodes.
   //

   problemStructure_->getNodeCommMgr().exchangeEqnInfo(comm_);

   //so now the node comm mgr knows the field IDs and eqn numbers for all of
   //the nodes that we care about -- both shared and external. We're going to
   //be packing the connectivities for remotely-owned shared nodes into the
   //eqn comm mgr later, as we go through all of the element connectivities
   //constructing the sparse matrix structure. At this point, let's have the
   //problemStructure_ object identify to the eqn comm mgr, which equations we
   //can expect other processors to be packing up to send to us.

   //(first we need to set the number of RHSs in the eqn comm manager)
   problemStructure_->getEqnCommMgr().setNumRHSs(numRHSs_);
   problemStructure_->initializeEqnCommMgr();

   //now lets inform the problemStructure_ class what the global eqn info is.
   //This will allow it to allocate space for the row length arrays (called
   //sysMatIndices_). This is an array of IntArrays, each one containing the
   //column indices for a row of
   //the matrix. The number of IntArrays is equal to the number
   //of equations local to this processor.

   problemStructure_->setEqnInfo(numGlobalEqns_, numLocalEqns_,
                                 localStartRow_, localEndRow_);
   
   IntArray* sysMatIndices = problemStructure_->getSystemMatrixIndices();
 
   if (debugOutput_) {
      fprintf(debugFile_,"numLocalEqns_: %d\n", numLocalEqns_);
   }

   //now let's tell the underlying linear algebra core to start building
   //its internal stuff.

   linSysCore_->createMatricesAndVectors(numGlobalEqns_, localStartRow_,
                                       numLocalEqns_);

   //now, we'll do our local equation profile calculations (i.e., determine
   //how long each row is). use the element scatter arrays to determine 
   //the sparsity pattern

   BlockDescriptor* blockRoster = problemStructure_->getBlockDescriptorsPtr();
 
   int bLimit = problemStructure_->getNumBlocks();
   for (int bIndex = 0; bIndex < bLimit; bIndex++) {
      int numBlockNodes = blockRoster[bIndex].getNumNodesPerElement();
      int numBlockElems = blockRoster[bIndex].getNumElements();
      int numBlockEqns = blockRoster[bIndex].getNumEqnsPerElement();

      int* scatterIndices = new int[numBlockEqns];

      int* remoteEqnOffsets = new int[numBlockEqns];
      int* remoteProcs = new int[numBlockEqns];
      int numRemoteEqns;

      //loop over all the elements, determining the elemental (both from nodes 
      //and from element DOF) contributions to the sparse matrix structure

      for(int elemIndex = 0; elemIndex < numBlockElems; elemIndex++) {

         problemStructure_->getScatterIndices_index(bIndex, elemIndex,
                                                    scatterIndices,
                                                    remoteEqnOffsets,
                                                    remoteProcs,
                                                    numRemoteEqns);

         //now, the eqns that are remotely owned (i.e., those from
         //shared nodes) need to be packed up so they can be shipped off
         //to the owning processor(s) later.

         for (i=0; i<numRemoteEqns; i++) {

            int index = remoteEqnOffsets[i];
            int eqn = scatterIndices[index];
            problemStructure_->
               getEqnCommMgr().addSendIndices(eqn, remoteProcs[i],
                                              scatterIndices, numBlockEqns);
         }

         //now, figure out where the scatterIndices go in the matrix 
         //structure.

         storeElementScatterIndices(scatterIndices, numBlockEqns);
      }

      delete [] scatterIndices;
      delete [] remoteEqnOffsets;
      delete [] remoteProcs;
   }

   // next, handle the matrix structure imposed by the constraint relations...
   //

   handleMultCRStructure();

   // we also need to accomodate penalty constraints, so let's process them
   // now...

   handlePenCRStructure();

   //we've now processed all of the local data that produces matrix structure.
   //so now let's have the equation comm manager exchange indices with the
   //other processors so we can account for remote contributions to our
   //matrix structure.

   EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();
   eqnCommMgr.exchangeIndices(comm_);

   //so now the remote contributions should be available, let's get them out
   //of the eqn comm mgr and put them into our local matrix structure.

   int numRecvEqns = eqnCommMgr.getNumRecvEqns();
   int* recvEqnLengths = eqnCommMgr.recvEqnLengthsPtr();
   int* recvEqnNumbers = eqnCommMgr.recvEqnNumbersPtr();
   int** recvIndices = eqnCommMgr.recvIndicesPtr();

   for(i=0; i<numRecvEqns; i++) {
      int eqn = recvEqnNumbers[i];
      if ((localStartRow_ > eqn) || (localEndRow_ < eqn)) {
         cerr << "BASE_FEI::initComplete: ERROR, recvEqn " << eqn
              << " out of range. (localStartRow_: " << localStartRow_
              << ", localEndRow_: " << localEndRow_ << ", localRank_: "
              << localRank_ << ")" << endl;
         abort();
      }

      int localRow = eqn - localStartRow_;

      for(int j=0; j<recvEqnLengths[i]; j++) {
         storeMatrixPosition(localRow, recvIndices[i][j]);
      }
   }

   //
   // all profile calculations are done, i.e., we now know the structure
   // of the sparse matrix, so we can configure (allocate) it.
   //

    IntArray rowLens(numLocalEqns_);
    int** indices = new int*[numLocalEqns_];

    for(int jj=0; jj<numLocalEqns_; jj++) {
        indices[jj] = &((sysMatIndices[jj])[0]);
        rowLens[jj] = sysMatIndices[jj].size();
    }

    linSysCore_->allocateMatrix(indices, &(rowLens[0]));

    delete [] indices;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//==============================================================================
void BASE_FEI::setProblemStructure(ProblemStructure* probStruct) {
   problemStructure_ = probStruct;
   problemStructureSet_ = true;
}

//==============================================================================
void BASE_FEI::setMatrixStructure() {

   problemStructure_->getEqnInfo(numGlobalEqns_, numLocalEqns_,
                                 localStartRow_, localEndRow_);

   linSysCore_->createMatricesAndVectors(numGlobalEqns_, localStartRow_,
                                       numLocalEqns_);

   IntArray* sysMatIndices = problemStructure_->getSystemMatrixIndices();

   IntArray rowLens(numLocalEqns_);
   int** indices = new int*[numLocalEqns_];

   for(int jj=0; jj<numLocalEqns_; jj++) {
      indices[jj] = &((sysMatIndices[jj])[0]);
      rowLens[jj] = sysMatIndices[jj].size();
   }

   linSysCore_->allocateMatrix(indices, &(rowLens[0]));

   matrixAllocated_ = true;

   delete [] indices;
}

//==============================================================================
void BASE_FEI::storeElementScatterIndices(int* indices, int numIndices) {
//
//This function takes a list of equation numbers, as input, and for each
//one, if it is a local equation number, goes to that row of the sysMatIndices
//structure and stores the whole list of equation numbers as column indices
//in that row of the matrix structure.
//
   for(int i=0; i<numIndices; i++) {
      int row = indices[i];

      if ((localStartRow_ <= row) && (row <= localEndRow_)) {
         int localRow = row - localStartRow_;

         for(int j=0; j<numIndices; j++) {
            int col = indices[j];

            storeMatrixPosition(localRow, col);
         }
      }
   }
}

//==============================================================================
void BASE_FEI::storeMatrixPosition(int localRow, int col) {
//
//This function inserts the column index 'col' in the system matrix structure
//in row 'localRow', if it isn't already there.
//
//This is a private BASE_FEI function. We may safely assume that this
//function will only be called with a valid 'localRow' -- i.e., it is
//0-based, and within the range [0..numLocalEqns_]. 'col' is a global,
//1-based equation number.
//
   IntArray& thisRow = problemStructure_->getSystemMatrixIndices()[localRow];

   if (thisRow.size() > 0) {

      int insertPoint = 0;
      int index = Utils::sortedIntListFind(col, &(thisRow[0]), thisRow.size(),
                                           &insertPoint);

      if (index < 0) thisRow.insert(insertPoint, col);
   }
   else {
      thisRow.append(col);
   }
}

//==============================================================================
void BASE_FEI::storeNodalColumnIndices(int eqn, NodeDescriptor& node,
                                       int fieldID) {
//
//This function stores the equation numbers associated with 'node' at
//'fieldID' as column indices in row 'eqn' of the system matrix structure.
//
   if ((localStartRow_ > eqn) || (eqn > localEndRow_)) return;

   int localRow = eqn - localStartRow_;

   int colNumber = node.getFieldEqnNumber(fieldID);

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();
   int index = problemStructure_->getFieldRosterIndex(fieldID);
   int numParams = fieldRoster[index].getNumFieldParams();

   for(int j=0; j<numParams; j++) {
      storeMatrixPosition(localRow, colNumber+j);
   }
}

//==============================================================================
void BASE_FEI::storeNodalRowIndices(NodeDescriptor& node,
                                    int fieldID, int eqn) {
//
//This function stores column 'eqn' in equation numbers associated with
//'node' at 'fieldID' in the system matrix structure. 
//
   int eqnNumber = node.getFieldEqnNumber(fieldID);

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();
   int index = problemStructure_->getFieldRosterIndex(fieldID);
   int numParams = fieldRoster[index].getNumFieldParams();

   for(int j=0; j<numParams; j++) {
      int localRow = eqnNumber + j - localStartRow_;

      storeMatrixPosition(localRow, eqn);
   }
}

//==============================================================================
void BASE_FEI::storeNodalColumnCoefs(int eqn, NodeDescriptor& node,
                                     int fieldID, double* coefs) {
//
//This function stores the coeficients for 'node' at 'fieldID' at the correct
//column indices in row 'eqn' of the system matrix.
//

   int eqnNumber = node.getFieldEqnNumber(fieldID);

   if ((localStartRow_ > eqn) || (eqn > localEndRow_)) return;

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();
   int index = problemStructure_->getFieldRosterIndex(fieldID);
   int numParams = fieldRoster[index].getNumFieldParams();

   int* cols = new int[numParams];

   for(int j=0; j<numParams; j++) {
      cols[j] = eqnNumber + j;
   }

   linSysCore_->sumIntoSystemMatrix(eqn, numParams, coefs, cols);

   delete [] cols;
}

//==============================================================================
void BASE_FEI::storeNodalRowCoefs(NodeDescriptor& node,
                                     int fieldID, double* coefs, int eqn) {
//
//This function stores coeficients in the equations for 'node', 'fieldID' at
//column index 'eqn' of the system matrix.
//
   int eqnNumber = node.getFieldEqnNumber(fieldID);

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();
   int index = problemStructure_->getFieldRosterIndex(fieldID);
   int numParams = fieldRoster[index].getNumFieldParams();

   for(int j=0; j<numParams; j++) {
      int row = eqnNumber + j;

      linSysCore_->sumIntoSystemMatrix(row, 1, &coefs[j], &eqn);
   }
}

//==============================================================================
NodeDescriptor& BASE_FEI::findNodeDescriptor(GlobalID nodeID) const {
//
//This function returns a NodeDescriptor reference if nodeID is a comm node
//or if it's an active node.
//
   NodeCommMgr& ncm = problemStructure_->getNodeCommMgr();
   int index = ncm.getCommNodeIndex(nodeID);

   if (index >= 0) {
      return(ncm.getCommNode(index));
   }
   else {
      index = problemStructure_->getActiveNodeIndex(nodeID);

      if (index < 0) {
         cerr << "BASE_FEI::findNodeDescriptor: ERROR, nodeID " << (int)nodeID
              << " is neither an active node nor a comm node." << endl;
         abort();
      }

      return(problemStructure_->getActiveNodesPtr()[index]);
   }
}

//==============================================================================
void BASE_FEI::handleMultCRStructure() {
//
//Private BASE_FEI function, to be called from initComplete.
//
//Records the system matrix structure arising from Lagrange Multiplier
//Constraint Relations.
//
// since at initialization all we are being passed is the
// general form of the constraint equations, we can't check to see if any
// of the weight terms are zeros at this stage of the game.  Hence, we
// have to reserve space for all the nodal weight vectors, even though
// they might turn out to be zeros during the load step....

   NodeCommMgr& nodeCommMgr = problemStructure_->getNodeCommMgr();

   for(int k = 0; k < storeNumCRMultRecords; k++) {
      int numMultCRs = ceqn_MultConstraints[k].getNumMultCRs();
      int lenList = ceqn_MultConstraints[k].getLenCRNodeList();

      int ntRow, ntCol;
      GlobalID **CRNodeTablePtr =
            ceqn_MultConstraints[k].pointerToCRNodeTable(ntRow, ntCol);
      int* CRFieldList = ceqn_MultConstraints[k].pointerToCRFieldList(ntCol);

      for(int i=0; i<numMultCRs; i++) {
         int crEqn = ceqn_MultConstraints[k].getEqnNumber() + i;

         for(int j=0; j<lenList; j++) {
            GlobalID nodeID = CRNodeTablePtr[i][j];
            int fieldID = CRFieldList[j];

            NodeDescriptor& node = findNodeDescriptor(nodeID);

            //first, store the column indices associated with this node, in
            //the constraint's equation. i.e., position (crEqn, node)

            storeNodalColumnIndices(crEqn, node, fieldID);

            //now we need to store the transpose of the above contribution,
            //i.e., position(s) (node, crEqn)

            if (node.getOwnerProc() == localRank_) {
               //if we own this node, we will simply store its equation
               //numbers in local rows (equations) of the matrix.

               storeNodalRowIndices(node, fieldID, crEqn);
            }
            else {
               //if we don't own it, then we need to register with the
               //eqn comm mgr that we'll be sending contributions to
               //column crEqn of the remote equations associated with this
               //node.

               storeNodalSendIndex(node, fieldID, crEqn);
            }
         }
      }
   }
}

//==============================================================================
void BASE_FEI::handlePenCRStructure() {
//
//This function oversees the putting in of any matrix structure that results
//from Penalty constraint relations.
//
// note that penalty constraints don't generate new equations
// (unlike Lagrange constraints), but they do add terms to the system
// stiffness matrix that couple all the nodes that contribute to the
// penalty constraint.  In addition, each submatrix is defined by the pair
// of nodes that creates its contribution, hence a submatrix can be defined
// in terms of two weight vectors (of length p and q) instead of the
// generally larger product matrix (of size pq)

// the additional terms take the form of little submatrices that look a lot 
// like an element stiffness and load matrix, where the nodes in the
// constraint list take on the role of the nodes associated with an
// element, and the individual matrix terms arise from the outer products
// of the constraint nodal weight vectors 

// rather than get elegant and treat this task as such an elemental energy
// term, we'll use some brute force to construct these penalty contributions
// on the fly, primarily to simplify -reading- this thing, so that the 
// derivations in the annotated implementation document are more readily
// followed...

   for(int k = 0; k < storeNumCRPenRecords; k++) {
      int numPenCRs = ceqn_PenConstraints[k].getNumPenCRs();
      int lenList = ceqn_PenConstraints[k].getLenCRNodeList();
      int ntRow, ntCol;
      GlobalID **CRNodeTablePtr = 
         ceqn_PenConstraints[k].pointerToCRNodeTable(ntRow, ntCol);
      assert (ntRow == numPenCRs);
      assert (ntCol == lenList);

      int* CRFieldList = ceqn_PenConstraints[k].pointerToCRFieldList(ntCol);

      // each constraint equation generates a set of nodal energy terms, so
      // we have to process a matrix of nodes for each constraint

      for(int p = 0; p < numPenCRs; p++) {
         for(int i = 0; i < lenList; i++) {
            GlobalID iNodeID = CRNodeTablePtr[p][i];
            int iField = CRFieldList[i];

            NodeDescriptor& iNode = findNodeDescriptor(iNodeID);

            for(int j = 0; j < lenList; j++) {
               GlobalID jNodeID = CRNodeTablePtr[p][j];
               int jField = CRFieldList[j];

               NodeDescriptor& jNode = findNodeDescriptor(jNodeID);

               if (iNode.getOwnerProc() == localRank_) {
                  //if iNode is local, we'll put equations into the local 
                  //matrix structure.

                  storeLocalNodeIndices(iNode, iField, jNode, jField);
               }
               else {
                  //if iNode is remotely owned, we'll be registering equations
                  //to send to its owning processor.

                  storeNodalSendIndices(iNode, iField, jNode, jField);
               }
            }
         }   //   end i loop
      }   //   end p loop
   }   //   end k loop
}
 
//==============================================================================
void BASE_FEI::storeNodalSendIndex(NodeDescriptor& node, int fieldID, int col) {
//
//This is a private BASE_FEI function. We can safely assume that it will only
//be called with a node that is not locally owned.
//
//This function tells the eqn comm mgr that we'll be sending contributions
//to column 'col' for the equations associated with 'fieldID', on 'node', on
//node's owning processor.
//
   EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();

   int proc = node.getOwnerProc();

   int eqnNumber = node.getFieldEqnNumber(fieldID);

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int index = problemStructure_->getFieldRosterIndex(fieldID);
   int numEqns = fieldRoster[index].getNumFieldParams();

   for(int i=0; i<numEqns; i++) {
      eqnCommMgr.addSendIndices(eqnNumber+i, proc, &col, 1);
   }
}

//==============================================================================
void BASE_FEI::storeNodalSendEqn(NodeDescriptor& node, int fieldID, int col,
                                 double* coefs) {
//
//This is a private BASE_FEI function. We can safely assume that it will only
//be called with a node that is not locally owned.
//
//This function tells the eqn comm mgr that we'll be sending contributions
//to column 'col' for the equations associated with 'fieldID', on 'node', on
//node's owning processor.
//
   EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();

   int proc = node.getOwnerProc();

   int eqnNumber = node.getFieldEqnNumber(fieldID);

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int index = problemStructure_->getFieldRosterIndex(fieldID);
   int numEqns = fieldRoster[index].getNumFieldParams();

   for(int i=0; i<numEqns; i++) {
      eqnCommMgr.addSendEqn(eqnNumber+i, proc, &coefs[i], &col, 1);
   }
}

//==============================================================================
void BASE_FEI::storeNodalSendIndices(NodeDescriptor& iNode, int iField,
                                  NodeDescriptor& jNode, int jField){
//
//This function will register with the eqn comm mgr the equations associated
//with iNode, field 'iField' having column indices that are the equations
//associated with jNode, field 'jField', to be sent to the owner of iNode.
//
   EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();
   int proc = iNode.getOwnerProc();

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int iEqn = iNode.getFieldEqnNumber(iField);
   int jEqn = jNode.getFieldEqnNumber(jField);

   int index = problemStructure_->getFieldRosterIndex(iField);
   int iNumParams = fieldRoster[index].getNumFieldParams();

   index = problemStructure_->getFieldRosterIndex(jField);
   int jNumParams = fieldRoster[index].getNumFieldParams();

   for(int i=0; i<iNumParams; i++) {
      int eqn = iEqn + i;

      for(int j=0; j<jNumParams; j++) {
         int col = jEqn + j;

         eqnCommMgr.addSendIndices(eqn, proc, &col, 1);
      }
   }
}

//==============================================================================
void BASE_FEI::storePenNodeSendData(NodeDescriptor& iNode, int iField,
                                    double* iCoefs,
                                    NodeDescriptor& jNode, int jField,
                                    double* jCoefs,
                                    double penValue, double CRValue){
//
//This function will register with the eqn comm mgr the equations associated
//with iNode, field 'iField' having column indices that are the equations
//associated with jNode, field 'jField', to be sent to the owner of iNode.
//
   EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();
   int proc = iNode.getOwnerProc();

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int iEqn = iNode.getFieldEqnNumber(iField);
   int jEqn = jNode.getFieldEqnNumber(jField);

   int index = problemStructure_->getFieldRosterIndex(iField);
   int iNumParams = fieldRoster[index].getNumFieldParams();

   index = problemStructure_->getFieldRosterIndex(jField);
   int jNumParams = fieldRoster[index].getNumFieldParams();

   double* coefs = new double[jNumParams];
   int* cols = new int[jNumParams];

   for(int i=0; i<iNumParams; i++) {
      for(int j=0; j<jNumParams; j++) {
         cols[j] = jEqn + j;
         coefs[j] = penValue*iCoefs[i]*jCoefs[j];
      }

      int row = iEqn + i;
      eqnCommMgr.addSendEqn(row, proc, coefs, cols, jNumParams);

      double rhsValue = penValue*iCoefs[i]*CRValue;
      eqnCommMgr.addSendRHS(row, proc, currentRHS_, rhsValue);
   }

   delete [] coefs;
   delete [] cols;
}

//==============================================================================
void BASE_FEI::storeLocalNodeIndices(NodeDescriptor& iNode, int iField,
                                     NodeDescriptor& jNode, int jField){
//
//This function will add to the local matrix structure the equations associated
//with iNode at iField, having column indices that are the equations associated
//with jNode at jField.
//
   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int iEqn = iNode.getFieldEqnNumber(iField);
   int jEqn = jNode.getFieldEqnNumber(jField);

   int index = problemStructure_->getFieldRosterIndex(iField);
   int iNumParams = fieldRoster[index].getNumFieldParams();

   index = problemStructure_->getFieldRosterIndex(jField);
   int jNumParams = fieldRoster[index].getNumFieldParams();

   for(int i=0; i<iNumParams; i++) {
      int localRow = iEqn + i - localStartRow_;

      for(int j=0; j<jNumParams; j++) {
         int col = jEqn + j;

         storeMatrixPosition(localRow, col);
      }
   }
}

//==============================================================================
void BASE_FEI::storePenNodeData(NodeDescriptor& iNode, int iField,
                                double* iCoefs,
                                NodeDescriptor& jNode, int jField,
                                double* jCoefs,
                                double penValue, double CRValue){
//
//This function will add to the local matrix the penalty constraint equations
//associated with iNode at iField, having column indices that are the
//equations associated with jNode at jField.
//Also, add the penalty contribution to the RHS vector.
//
   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int iEqn = iNode.getFieldEqnNumber(iField);
   int jEqn = jNode.getFieldEqnNumber(jField);

   int index = problemStructure_->getFieldRosterIndex(iField);
   int iNumParams = fieldRoster[index].getNumFieldParams();

   index = problemStructure_->getFieldRosterIndex(jField);
   int jNumParams = fieldRoster[index].getNumFieldParams();

   double* coefs = new double[jNumParams];
   int* cols = new int[jNumParams];

   for(int i=0; i<iNumParams; i++) {
      for(int j=0; j<jNumParams; j++) {
         cols[j] = jEqn + j;
         coefs[j] = penValue*iCoefs[i]*jCoefs[j];
      }

      int row = iEqn + i;
      linSysCore_->sumIntoSystemMatrix(row, jNumParams, coefs, cols);

      double rhsValue = penValue*iCoefs[i]*CRValue;
      linSysCore_->sumIntoRHSVector(1, &rhsValue, &row);
   }

   delete [] coefs;
   delete [] cols;
}

//==============================================================================
int BASE_FEI::countActiveNodeEqns() {
//
//Private BASE_FEI function, to be called from initComplete, after
//setActiveNodeOwnerProcs() has been called.
//
//Counts and returns the number of equations associated with locally owned
//active nodes.
//
   int numActiveNodes = problemStructure_->getNumActiveNodes();
   NodeDescriptor* actNodes = problemStructure_->getActiveNodesPtr();
   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int numEqns = 0;

   for(int i=0; i<numActiveNodes; i++) {
      if (actNodes[i].getOwnerProc() == localRank_) {
         int numFields = actNodes[i].getNumFields();
         int* fieldIDList = actNodes[i].getFieldIDList();

         for(int j=0; j<numFields; j++) {
            int index = problemStructure_->getFieldRosterIndex(fieldIDList[j]);
            int numParams = fieldRoster[index].getNumFieldParams();

            numEqns += numParams;
         }
      }
   }

   return(numEqns);
}

//==============================================================================
void BASE_FEI::setActiveNodeOwnerProcs() {
//
//Private BASE_FEI function, to be called from initComplete, after
//nodeCommMgr->initComplete() has been called.
//
//This function sets the ownerProc() field on each of the active nodes.
//

   int numActiveNodes = problemStructure_->getNumActiveNodes();
   NodeDescriptor* actNodes = problemStructure_->getActiveNodesPtr();

   NodeCommMgr& nodeCommMgr = problemStructure_->getNodeCommMgr();

   for(int node=0; node<numActiveNodes; node++) {
      GlobalID nodeID = actNodes[node].getGlobalNodeID();

      int lowestProc = nodeCommMgr.getLowestProcNumber(nodeID);

      bool isExtern = nodeCommMgr.isExternal(nodeID);

      if ((lowestProc >= 0) && (lowestProc < localRank_) && (!isExtern)) {

         actNodes[node].setOwnerProc(lowestProc);
      }
      else {
         //... else lowestProc < 0 (it's not a comm node, it's purely local)
         //    or lowestProc >= localRank_ (which means we own it)

         actNodes[node].setOwnerProc(localRank_);
      }
   }
}

//==============================================================================
int BASE_FEI::setActiveNodeEqnInfo() {
//
//Private BASE_FEI function, to be called in initComplete, after
//'getGlobalEqnInfo()'.
//
//At this point we know which fields are associated with each active node,
//and also how many equations there are globally, so we can go through and
//set equation numbers for all active nodes.
//
//We will also feed all of our local nodes into the node comm manager so it
//can record the ones it cares about.
//
//While we're here, we'll also inform each block how many equations are
//associated with it.
//
//The return value will be the number of equations arising from active nodes.
//
   int numNodes = problemStructure_->getNumActiveNodes();
   NodeDescriptor* actNodes = problemStructure_->getActiveNodesPtr();

   if (debugOutput_) {
      fprintf(debugFile_, "   number of active nodes: %d\n", numNodes);
      fflush(debugFile_);
   }

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   NodeCommMgr& nodeCommMgr = problemStructure_->getNodeCommMgr();

   int numBlocks = problemStructure_->getNumBlocks();
   BlockDescriptor* blockRoster = problemStructure_->getBlockDescriptorsPtr();

   int* eqnsPerBlock = new int[numBlocks];
   for(int ii=0; ii<numBlocks; ii++) eqnsPerBlock[ii] = 0;

   int numEqns = 0;

   FILE* e2nFile = NULL;

   if (debugOutput_) {
      char* e2nFileName;
      int len = strlen(debugPath_) + 64;
      e2nFileName = new char[len];

      sprintf(e2nFileName, "%s/BASE_FEI.eqn2node.%d.%d",debugPath_,
              numProcs_, localRank_);

      e2nFile = fopen(e2nFileName, "w");

      delete [] e2nFileName;
   }

   for(int i=0; i<numNodes; i++) {
      int numFields = actNodes[i].getNumFields();
      int* fieldIDList = actNodes[i].getFieldIDList();

      int numNodalDOF = 0;
      int eqnNumber;

      for(int j=0; j<numFields; j++) {
         int index = problemStructure_->getFieldRosterIndex(fieldIDList[j]);
         int numFieldParams = fieldRoster[index].getNumFieldParams();
         numNodalDOF += numFieldParams;

         //
         //here we're going to update our count of equations per block
         //
         int numBlks = actNodes[i].getNumBlocks();
         GlobalID* blkList = actNodes[i].getBlockList();
         for(int k=0; k<numBlks; k++) {
            int bIndex = problemStructure_->getBlockIndex(blkList[k]);
            if (blockRoster[bIndex].containsField(fieldIDList[j])) {
               eqnsPerBlock[bIndex] += numFieldParams;
            }
         }

         if (actNodes[i].getOwnerProc() == localRank_) {
            eqnNumber = localStartRow_ + numEqns;

            if (e2nFile != NULL) {
               for(int l=0; l<numFieldParams; l++) {
                  fprintf(e2nFile, "%d %d\n", eqnNumber+l,
                          (int)actNodes[i].getGlobalNodeID());
               }
            }

            numEqns += numFieldParams;

            actNodes[i].setFieldEqnNumber(fieldIDList[j], eqnNumber);
         }
      }

      actNodes[i].setNumNodalDOF(numNodalDOF);

      if (actNodes[i].getOwnerProc() == localRank_) {
         nodeCommMgr.addLocalNode(actNodes[i]);
      }
   }

   if (e2nFile != NULL) {
      fclose(e2nFile);
      e2nFile = NULL;
   }

   //eqnsPerBlock now holds the number of nodal equations for each block.
   //let's add to that the number of element-DOF for each block, then inform
   //each block how many total equations it has.
   for(int j=0; j<numBlocks; j++) {
      int numElemDOF = blockRoster[j].getNumElemDOFPerElement();
      int numElems = blockRoster[j].getNumElements();

      eqnsPerBlock[j] += numElemDOF*numElems;

      blockRoster[j].setTotalNumEqns(eqnsPerBlock[j]);
   }

   delete [] eqnsPerBlock;

   return(numEqns);
}

//==============================================================================
int BASE_FEI::calcTotalNumElemDOF() {

   BlockDescriptor* blockRoster = problemStructure_->getBlockDescriptorsPtr();

   int totalNumElemDOF = 0;

   for(int i = 0; i < problemStructure_->getNumBlocks(); i++) {
      int numElemDOF = blockRoster[i].getNumElemDOFPerElement();
      if (numElemDOF > 0) {
         int numElems = blockRoster[i].getNumElements();

         totalNumElemDOF += numElems*numElemDOF;
      }
   }

   return(totalNumElemDOF);
}

//==============================================================================
void BASE_FEI::setElemDOFEqnInfo(int numNodalEqns) {
//
//Private BASE_FEI function, to be called from initComplete, after
//setActiveNodeEqnInfo() has been called.
//
//Sets global equation numbers for all element-DOF.
//
   BlockDescriptor* blockRoster = problemStructure_->getBlockDescriptorsPtr();

   int eqnNumber = localStartRow_ + numNodalEqns;

   for(int i = 0; i < problemStructure_->getNumBlocks(); i++) {
      int numElemDOF = blockRoster[i].getNumElemDOFPerElement();

      if (numElemDOF > 0) {
         int numElems = blockRoster[i].getNumElements();
         int* elemDOFEqnsPtr = blockRoster[i].elemDOFEqnNumbersPtr();

         for(int j=0; j<numElems; j++) {
            elemDOFEqnsPtr[j] = eqnNumber;
            eqnNumber += numElemDOF;
         }
      }
   }
}

//==============================================================================
void BASE_FEI::getGlobalEqnInfo(int numLocalEqns, int& numGlobalEqns, 
                                int& localStart, int& localEnd) {
//
//This function does the inter-process communication necessary to obtain,
//on each processor, the global number of equations, and the local starting
//and ending equation numbers.
//

   //first, get each processor's local number of equations on the master proc.

   int* globalNumProcEqns = new int[numProcs_];

   if (numProcs_ > 1) {
      MPI_Gather(&numLocalEqns, 1, MPI_INT, globalNumProcEqns, 1, MPI_INT,
                 masterRank_, comm_);
   }
   else {
      globalNumProcEqns[0] = numLocalEqns;
   }

   //compute offsets for all processors (starting index for local equations)

   int* globalStartRow = new int[numProcs_];
   globalStartRow[0] = 1;    // we're going to start rows & cols at 1 (global)

   if (localRank_ == masterRank_) {
      for(int i=1;i<numProcs_;i++) {
         globalStartRow[i] = globalStartRow[i-1] + globalNumProcEqns[i-1];
      }
   }

   //now, scatter vector of startRows to all processors from the master node.

   if (numProcs_ == 1){
      localStart = globalStartRow[0];
   }
   else {
      MPI_Scatter(globalStartRow, 1, MPI_INT, &localStart, 1, MPI_INT,
                  masterRank_, comm_);
   }

   delete [] globalStartRow;

   //now each processor can compute its local end row (last local equation).

   localEnd = localStart + numLocalEqns - 1;

   //compute global number of equations.

   numGlobalEqns = 0;

   if (localRank_ == masterRank_) {
      numGlobalEqns = globalNumProcEqns[0];

      for(int i=1; i<numProcs_; i++) {
          numGlobalEqns += globalNumProcEqns[i];
      }
   }

   delete [] globalNumProcEqns;

   if (numProcs_ > 1) {
      MPI_Bcast(&numGlobalEqns, 1, MPI_INT, masterRank_, comm_);
   }
}

//------------------------------------------------------------------------------
int BASE_FEI::resetSystem(double s) {
//
//  This function may only be called after the initialization phase is
//  complete. It requires that the system matrix and rhs vector have already
//  been created.
//  It then puts the value s throughout both the matrix and the vector.
//

    baseTime_ = MPI_Wtime();
 
    debugOutput("resetSystem");

    linSysCore_->resetMatrixAndVector(s);

    //clear away any boundary condition data.
    bcManager_->clearAllBCs();

    //clear away the values in the eqnCommMgr.
    
    EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();

    eqnCommMgr.resetCoefs();

//    if (debugOutput_ && (solveCounter_>1)){
//        //if debug output is 'on' and we've already completed at least one
//        //solve, reset the debug output file.
//
//        setDebugOutput(debugPath_, debugFileName_);
//    }
 
    debugOutput("leaving resetSystem");

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int BASE_FEI::beginLoadNodeSets(int numBCNodeSets) {
//
//  tasks: start the loading of nodal loading information
//
    debugOutput("beginLoadNodeSets");

    if (debugOutput_) {
       fprintf(debugFile_, "   numBCNodeSets: %d\n", numBCNodeSets);
       fflush(debugFile_);
    }

    (void)numBCNodeSets; // this prevents "unused variable" warnings

    if ((!problemStructureSet_) && (!problemStructureAllocated_)) {
        cerr << "BASE_FEI::beginLoadNodeSets: ERROR, problem structure"
            << " class has not been set. Aborting." << endl;
        abort();
    }

    if ((problemStructureSet_) && (!matrixAllocated_)) setMatrixStructure();

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_FEI::loadBCSet(const GlobalID *BCNodeSet,  
                        int lenBCNodeSet,  
                        int BCFieldID,
                        const double *const *alphaBCDataTable,  
                        const double *const *betaBCDataTable,  
                        const double *const *gammaBCDataTable) {
//
//  task: load boundary condition information for a given nodal data set
//

   baseTime_ = MPI_Wtime();

   debugOutput("loadBCSet");

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int index = problemStructure_->getFieldRosterIndex(BCFieldID);
   int size = fieldRoster[index].getNumFieldParams();

   if (debugOutput_) {
      fprintf(debugFile_, "   numBCNodes: %d\n", lenBCNodeSet);
      for(int i=0; i<lenBCNodeSet; i++) {
         fprintf(debugFile_,
                 "      nodeID[%d]: %d, fieldID: %d, fieldSize: %d\n",
                 i, (int)BCNodeSet[i], BCFieldID, size);
         int j;
         fprintf(debugFile_, "         alpha: ");
         for(j=0; j<size; j++)
            fprintf(debugFile_, "%le ", alphaBCDataTable[i][j]);
         fprintf(debugFile_,"\n");
         fprintf(debugFile_, "         beta: ");
         for(j=0; j<size; j++)
            fprintf(debugFile_, "%le ", betaBCDataTable[i][j]);
         fprintf(debugFile_,"\n");
         fprintf(debugFile_, "         gamma: ");
         for(j=0; j<size; j++)
            fprintf(debugFile_, "%le ", gammaBCDataTable[i][j]);
         fprintf(debugFile_,"\n");
      }
   }

   //simply loop over the BC nodes, adding the bc data arrays to
   //the bcManager class.

   for(int i = 0; i < lenBCNodeSet; i++) {
  
      bcManager_->addBCRecord(BCNodeSet[i],
                              BCFieldID, size,
                              alphaBCDataTable[i],
                              betaBCDataTable[i],
                              gammaBCDataTable[i]);
   }

//  the tables of boundary condition params have been stored, so we're
//  all done here...

   wTime_ += MPI_Wtime() - baseTime_;

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::endLoadNodeSets() {
//
//  tasks: complete the loading of nodal information
//

   debugOutput("endLoadNodeSets");

   //
   //have the boundary condition manager consolidate the BC specifications
   //that were loaded during the loadBCSet calls.
   //

   bcManager_->consolidateBCs();

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::beginLoadElemBlock(GlobalID elemBlockID,
                                 int numElemSets,
                                 int numElemTotal) {
//
//  tasks: begin blocked-element data loading step
//
//  store the current element block ID, as we need it to find our way within
//  the various element worksets that will be passed.  In addition, set a
//  current workset counter so that we can keep track of where we are in the
//  workset lists, too...
//

    baseTime_ = MPI_Wtime();

    debugOutput("beginLoadElemBlock");

    if ((!problemStructureSet_) && (!problemStructureAllocated_)) {
        cerr << "BASE_FEI::beginLoadElemBlock: ERROR, problem structure"
            << " class has not been set. Aborting." << endl;
        abort();
    }

    if ((problemStructureSet_) && (!matrixAllocated_)) setMatrixStructure();

    if (problemStructureSet_)
       allocateWorksetChecks(problemStructure_->getNumBlocks());

    currentElemBlockID = elemBlockID;
    currentWorkSetID = 0;

    int nb = problemStructure_->getNumBlocks();

    currentElemBlockIndex = problemStructure_->getBlockIndex(elemBlockID);
    currentBlock = &(problemStructure_->getBlockDescriptor(elemBlockID));

//  reset starting index for element list

    nextElemIndex[currentElemBlockIndex] = 0;

    numWorksets[currentElemBlockIndex] = numElemSets;
    numMatrixWorksetsStored[currentElemBlockIndex] = 0;
    numRHSWorksetsStored[currentElemBlockIndex] = 0;

//  insure that the number of elements present in the init step is the same
//  as that we're about to get in the load step

    assert (numElemTotal == currentBlock->getNumElements());

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_FEI::loadElemSet(int elemSetID, 
                          int numElems, 
                          const GlobalID *elemIDs,  
                          const GlobalID *const *elemConn,
                          const double *const *const *elemStiffness,
                          const double *const *elemLoad,
                          int elemFormat) {
//
//  task: assemble the element stiffness matrices and
//         load vectors for a given workset
//

   baseTime_ = MPI_Wtime();

   (void)elemConn;

   if (debugOutput_) {
      fprintf(debugFile_,"loadElemSet, numElems: %d\n", numElems);
      fprintf(debugFile_,"currentElemBlockID: %d\n",(int)currentElemBlockID);
      fflush(debugFile_);
   }

   //the assumption is that loadElemSet must be called inside a
   //beginLoadElemBlock/endLoadElemBlock context, so we don't need to re-fetch
   //the BlockDescriptor or the currentElemBlockIndex that we obtained in the
   //last call to beginLoadElemBlock...

   currentWorkSetID = elemSetID;

   numMatrixWorksetsStored[currentElemBlockIndex]++;
   numRHSWorksetsStored[currentElemBlockIndex]++;
 
   if (numElems <= 0) return(0); //zero-length workset, do nothing.

   int i, k;

   //  we're gonna need some control parameters that didn't come in through this
   //  function's argument list.

   int numElemRows = 0;    //  number of rows per element array in this block

   numElemRows = currentBlock->getNumEqnsPerElement();

//    if (debugOutput_) {
//        fprintf(debugFile_,"numElemRows: %d\n", numElemRows);
//        fflush(debugFile_);
//    }

//    if (debugOutput_) {
//        fprintf(debugFile_,"loadElemSet, numElems: %d\n", numElems);
//        for(i=0; i<numElems; i++) {
//           fprintf(debugFile_,"coeffs for elemID %d\n",(int)elemIDs[i]);
//           for(int j=0; j<numElemRows; j++) {
//              fprintf(debugFile_,"   ");
//              for(k=0; k<numElemRows; k++) {
//                 fprintf(debugFile_, "%e ",elemStiffness[i][j][k]);
//              }
//              fprintf(debugFile_, "\n");
//           }
//        }
//        fflush(debugFile_);
//    }

   // scatter indices into the system matrix
   IntArray scatterIndices(numElemRows);
   IntArray remoteEqnOffsets(numElemRows);
   IntArray remoteProcs(numElemRows);

   // local copy of (single) element stiffness
   double **ek = NULL;

   ek = new double* [numElemRows];
   for (i = 0; i < numElemRows; i++) {
      ek[i] = new double[numElemRows];
   }

   //now we'll loop through the elements in this workset, assembling
   //the stiffnesses and loads into the global sparse system stiffness
   //matrix.

   for (k = 0; k < numElems; k++) {

      //we'll make a local dense copy of the element stiffness array
      //if the stiffness array was passed in using one of the "weird"
      //element formats.
      if (elemFormat != 0) {
         copyStiffness(elemStiffness[k], numElemRows, elemFormat, ek);
      }

      //our element-assembly logic assumes implicitly that all nodal solution 
      //parameters are grouped contiguously, to simplify matrix allocation and
      //other low-level tasks.  If this isn't the case (e.g., for the 
      //FIELD_CENTRIC data scheme, the only alternative interleave strategy),
      //then we need to reorder the rows and columns of the element stiffness
      //and load in order to facilitate the nodal-equation-oriented assembly ops
      //below...

      if (currentBlock->getInterleaveStrategy() != 0) {
         continue;  // remapping of interleaving not yet supported...
                    // but when it is, the remapping function goes here!
      }

      //now let's obtain the scatter indices for assembling the equations
      //into their appropriate places in the system stiffness and load
      //structures...

      int numRemoteEqns = 0;
      problemStructure_->getScatterIndices_ID(currentElemBlockID, elemIDs[k],
                                              &scatterIndices[0],
                                              &remoteEqnOffsets[0],
                                              &remoteProcs[0],
                                              numRemoteEqns);
      if (elemFormat != 0) {
         packSharedStiffness(&remoteEqnOffsets[0], &remoteProcs[0],
                             numRemoteEqns, &scatterIndices[0],
                             ek, numElemRows);
      }
      else {
         packSharedStiffness(&remoteEqnOffsets[0], &remoteProcs[0],
                             numRemoteEqns, &scatterIndices[0],
                             elemStiffness[k], numElemRows);
      }

      packSharedLoad(&remoteEqnOffsets[0], &remoteProcs[0],
                     numRemoteEqns, &scatterIndices[0], elemLoad[k]);

      // assembly operation

      if (elemFormat != 0) {
         assembleStiffnessAndLoad(numElemRows, &scatterIndices[0],
                                  ek, elemLoad[k]);
      }
      else {
         assembleStiffnessAndLoad(numElemRows, &scatterIndices[0],
                                  elemStiffness[k], elemLoad[k]);
      }

   }//end of the 'for k' loop over numElems
 
//  all done

   for (i = 0; i < numElemRows; i++) {
      delete [] ek[i];
   }
   delete [] ek;

   debugOutput("leaving loadElemSet");

   wTime_ += MPI_Wtime() - baseTime_;

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::loadElemSetMatrix(int elemSetID, 
                          int numElems, 
                          const GlobalID *elemIDs,  
                          const GlobalID *const *elemConn,
                          const double *const *const *elemStiffness,
                          int elemFormat) {
//
//  task: assemble the element stiffness matrices for a given workset
//

   baseTime_ = MPI_Wtime();

   (void)elemConn;

   //  number of rows per element array in this block
   int numElemRows = currentBlock->getNumEqnsPerElement();

   debugOutput("loadElemSetMatrix");

   if (debugOutput_) {
      fprintf(debugFile_,"   numElems: %d  (currentBlockID: %d)\n",
              numElems, currentElemBlockID);
      for(int i=0; i<numElems; i++) {
         fprintf(debugFile_, "   elemID[%d]: %d\n", i, (int)elemIDs[i]);
         fprintf(debugFile_, "   stiffness:\n");
         for(int j=0; j<numElemRows; j++) {
            fprintf(debugFile_, "   ");
            for(int k=0; k<numElemRows; k++) {
               fprintf(debugFile_, "%f ", elemStiffness[i][j][k]);
            }
            fprintf(debugFile_, "\n");
         }
      }
      fflush(debugFile_);
   }

   currentWorkSetID = elemSetID;

   numMatrixWorksetsStored[currentElemBlockIndex]++;

   if (numElems <= 0) return(0); //zero-length workset, do nothing.

   int i, k;

   // scatter indices into the system matrix
   IntArray scatterIndices(numElemRows);
   IntArray remoteEqnOffsets(numElemRows);
   IntArray remoteProcs(numElemRows);

   // local copy of (single) element stiffness
   double **ek = NULL;

   ek = new double* [numElemRows];
   for (i = 0; i < numElemRows; i++) {
      ek[i] = new double[numElemRows];
   }

   //now we'll loop through the elements in this workset, assembling
   //the stiffnesses into the global sparse system stiffness
   //matrix.

   for (k = 0; k < numElems; k++) {

      //we'll make a local dense copy of the element stiffness array
      //if the stiffness array was passed in using one of the "weird"
      //element formats.
      if (elemFormat != 0) {
         copyStiffness(elemStiffness[k], numElemRows, elemFormat, ek);
      }

      //our element-assembly logic assumes implicitly that all nodal
      //solution parameters are grouped contiguously, to simplify matrix
      //allocation and other low-level tasks.  If this isn't the case (e.g.,
      //for the FIELD_CENTRIC data scheme, the only alternative interleave
      //strategy supported), then we need to reorder the rows and columns
      //of the element stiffness and load in order to facilitate the
      //nodal-equation-oriented assembly ops below...

      if (currentBlock->getInterleaveStrategy() != 0) {
         continue;  // remapping of interleaving not yet supported...
                    // but when it is, the remapping function goes here!
      }

      //  now let's obtain the scatter indices for assembling the equations
      //  into their appropriate places in the system stiffness
      //  structure...

      int numRemoteEqns = 0;
      problemStructure_->getScatterIndices_ID(currentElemBlockID, elemIDs[k],
                                              &scatterIndices[0],
                                              &remoteEqnOffsets[0],
                                              &remoteProcs[0],
                                              numRemoteEqns);

      if (elemFormat != 0) {
         packSharedStiffness(&remoteEqnOffsets[0], &remoteProcs[0],
                             numRemoteEqns, &scatterIndices[0],
                             ek, numElemRows);
      }
      else {
         packSharedStiffness(&remoteEqnOffsets[0], &remoteProcs[0],
                             numRemoteEqns, &scatterIndices[0],
                             elemStiffness[k], numElemRows);
      }

      // assembly operation

      if (elemFormat != 0) {
         assembleStiffness(numElemRows, &scatterIndices[0], ek);
      }
      else {
         assembleStiffness(numElemRows, &scatterIndices[0],
                           elemStiffness[k]);
      }

   }//end of the 'for k' loop over numElems

//  all done

   for (i = 0; i < numElemRows; i++) {
      delete [] ek[i];
   }
   delete [] ek;

   wTime_ += MPI_Wtime() - baseTime_;

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::loadElemSetRHS(int elemSetID, 
                          int numElems, 
                          const GlobalID *elemIDs,  
                          const GlobalID *const *elemConn,
                          const double *const *elemLoad) {
//
//  task: assemble the element stiffness matrices for a given workset
//

   baseTime_ = MPI_Wtime();

   (void)elemConn;

   if (debugOutput_) {
      fprintf(debugFile_,"loadElemSetRHS, numElems: %d\n", numElems);
      fprintf(debugFile_,"currentElemBlockID: %d\n",(int)currentElemBlockID);
      fflush(debugFile_);
   }

   currentWorkSetID = elemSetID;

   numRHSWorksetsStored[currentElemBlockIndex]++;

   if (numElems <= 0) return(0); //zero-length workset, do nothing.

   //  number of rows per element array in this block
   int numElemRows = currentBlock->getNumEqnsPerElement();

   if (debugOutput_) {
      fprintf(debugFile_,"numElemRows: %d\n", numElemRows);
      fflush(debugFile_);
   }

   // scatter indices into the system matrix
   IntArray scatterIndices(numElemRows);
   IntArray remoteEqnOffsets(numElemRows);
   IntArray remoteProcs(numElemRows);

   //now we'll loop through the elements in this workset, assembling
   //the stiffnesses into the global sparse system stiffness
   //matrix.

   for (int k = 0; k < numElems; k++) {

      //our element-assembly logic assumes implicitly that all nodal
      //solution parameters are grouped contiguously, to simplify matrix
      //allocation and other low-level tasks.  If this isn't the case
      //(e.g., for the FIELD_CENTRIC data scheme, the only alternative
      //interleave strategy supported), then we need to reorder the rows
      //and columns of the element stiffness and load in order to facilitate
      //the nodal-equation-oriented assembly ops below...

      if (currentBlock->getInterleaveStrategy() != 0) {
         continue;  // remapping of interleaving not yet supported...
                    // but when it is, the remapping function goes here!
      }

      //now let's obtain the scatter indices for assembling the equations
      //into their appropriate places in the system rhs vector...

      int numRemoteEqns = 0;
      problemStructure_->getScatterIndices_ID(currentElemBlockID, elemIDs[k],
                                              &scatterIndices[0],
                                              &remoteEqnOffsets[0],
                                              &remoteProcs[0],
                                              numRemoteEqns);

      packSharedLoad(&remoteEqnOffsets[0], &remoteProcs[0],
                        numRemoteEqns, &scatterIndices[0], elemLoad[k]);

      // assembly operation

      assembleLoad(numElemRows, &scatterIndices[0], elemLoad[k]);

   }//end of the 'for k' loop over numElems

//  all done

   wTime_ += MPI_Wtime() - baseTime_;

   return(0);
}

//------------------------------------------------------------------------------
void BASE_FEI::copyStiffness(const double* const* elemStiff,
                             int numRows, int elemFormat,
                             double** localStiffness){
//
//Unpack the element stiffness array in elemStiff into a full dense
//local copy 'localStiffness'.
//
    int i, j;
    const double* elStiff_k_i = NULL;

    switch (elemFormat) {

        case 0:                 // row-contiguous dense storage
            for (i = 0; i < numRows; i++) {
                elStiff_k_i = elemStiff[i];
                for (j = 0; j < numRows; j++) {
                    localStiffness[i][j] = elStiff_k_i[j];
                }
            }
            break;

        case 1:                 // row-contiguous packed upper-symmetric
            for (i = 0; i < numRows; i++) {
                elStiff_k_i = elemStiff[i];
                int jcol = 0;
                for (j = i; j < numRows; j++) {
                    localStiffness[i][j] = elStiff_k_i[jcol];
                    localStiffness[j][i] = localStiffness[i][j];
                    ++jcol;
                }
            }
            break;

        case 2:                 // row-contiguous packed lower-symmetric
            for (i = 0; i < numRows; i++) {
                elStiff_k_i = elemStiff[i];
                int jcol = 0;
                for (j = 0; j <= i; j++) {
                    localStiffness[i][j] = elStiff_k_i[jcol];
                    localStiffness[j][i] = localStiffness[i][j];
                    ++jcol;
                }
            }
            break;

         case 3:                 // column-contiguous dense storage
            for (i = 0; i < numRows; i++) {
                elStiff_k_i = elemStiff[i];
                for (j = 0; j < numRows; j++) {
                    localStiffness[j][i] = elStiff_k_i[j];
                }
            }
            break;

        case 4:                 // column-contiguous packed upper-symmetric
            for (i = 0; i < numRows; i++) {
                elStiff_k_i = elemStiff[i];
                int jcol = 0;
                for (j = 0; j <= i; j++) {
                    localStiffness[i][j] = elStiff_k_i[jcol];
                    localStiffness[j][i] = localStiffness[i][j];
                    ++jcol;
                }
            }
            break;

        case 5:                 // column-contiguous packed lower-symmetric
            for (i = 0; i < numRows; i++) {
                elStiff_k_i = elemStiff[i];
                int jcol = 0;
                for (j = i; j < numRows; j++) {
                    localStiffness[i][j] = elStiff_k_i[jcol];
                    localStiffness[j][i] = localStiffness[i][j];
                    ++jcol;
                }
            }
            break;

        default:
                    // need a warning here
            break;
    }
}

//------------------------------------------------------------------------------
void BASE_FEI::getBCEqns(IntArray& essEqns, RealArray& essAlpha,
                   RealArray& essGamma,
                   IntArray& otherEqns, RealArray& otherAlpha,
                   RealArray& otherBeta, RealArray& otherGamma) {

   double eps = 1.e-18;

   int numBCNodes = bcManager_->getNumBCNodes();

   GlobalID* BCNodeIDs = bcManager_->getBCNodeIDsPtr();
   int* fieldsPerNode = bcManager_->fieldsPerNodePtr();

   int** fieldIDs = bcManager_->bcFieldIDsPtr();
   int** fieldSizes = bcManager_->bcFieldSizesPtr();

   double*** alpha = bcManager_->alphaPtr();
   double*** beta = bcManager_->betaPtr();
   double*** gamma = bcManager_->gammaPtr();

   NodeDescriptor* nodes = problemStructure_->getActiveNodesPtr();

   for(int i=0; i<numBCNodes; i++){
      NodeDescriptor& node = findNodeDescriptor(BCNodeIDs[i]);

      for(int j=0; j<fieldsPerNode[i]; j++) {
         int eqn = node.getFieldEqnNumber(fieldIDs[i][j]);

         double* myAlpha = alpha[i][j];
         double* myBeta = beta[i][j];
         double* myGamma = gamma[i][j];

         for(int k = 0; k < fieldSizes[i][j]; k++) {

            //compute this equation index for manipulating the matrix and rhs.

            int thisEqn = eqn + k;

            double thisAlpha = myAlpha[k];
            double thisBeta = myBeta[k];
            double thisGamma = myGamma[k];

            //is it an essential bc in the current solution component direction?

            if ((fabs(thisAlpha) > eps) && (fabs(thisBeta) <= eps)) {

               int ind = -1, insert = 0;
               if (essEqns.size() > 0) {
                  ind = Utils::sortedIntListFind(thisEqn, &essEqns[0],
                                                   essEqns.size(), &insert);
                  if (ind < 0) {
                     essEqns.insert(insert, thisEqn);
                     essAlpha.insert(insert, thisAlpha);
                     essGamma.insert(insert, thisGamma);
                  }
                  else {
                     essEqns.insert(ind, thisEqn);
                     essAlpha.insert(ind, thisAlpha);
                     essGamma.insert(ind, thisGamma);
                  }
               }
               else {
                  //if essEqns.size()==0, just append the stuff.
                  essEqns.append(thisEqn);
                  essAlpha.append(thisAlpha);
                  essGamma.append(thisGamma);
               }
            }
            else {
               //if we have a natural or mixed b.c. (beta != 0) then we're
               //gonna add terms to the diagonal and the rhs vector and hope
               //for the best...

               if (fabs(thisBeta) > eps) {

                  int ind = -1, insert = 0;
                  if (otherEqns.size() > 0) {
                     ind = find_index(thisEqn, &otherEqns[0],
                                        otherEqns.size(), &insert);
                     if (ind < 0) {
                        otherEqns.insert(insert, thisEqn);
                        otherAlpha.insert(insert, thisAlpha);
                        otherBeta.insert(insert, thisBeta);
                        otherGamma.insert(insert, thisGamma);
                     }
                     else {
                        otherEqns.insert(ind, thisEqn);
                        otherAlpha.insert(ind, thisAlpha);
                        otherBeta.insert(ind, thisBeta);
                        otherGamma.insert(ind, thisGamma);
                     }
                  }
                  else {
                     //if otherEqns.size()==0, just append the stuff.
                     otherEqns.append(thisEqn);
                     otherAlpha.append(thisAlpha);
                     otherBeta.append(thisBeta);
                     otherGamma.append(thisGamma);
                  }
               }
            }
         }//for(k<fieldSize)loop
      }
   }
}

//------------------------------------------------------------------------------
void BASE_FEI::implementAllBCs() {
//
// This function will handle the modifications to the stiffness and load
// necessary to enforce nodal boundary conditions.
//
   debugOutput("implementAllBCs");

   int numBCNodes = bcManager_->getNumBCNodes();

   GlobalID* BCNodeIDs = bcManager_->getBCNodeIDsPtr();
   int* fieldsPerNode = bcManager_->fieldsPerNodePtr();

   int** fieldIDs = bcManager_->bcFieldIDsPtr();
   int** fieldSizes = bcManager_->bcFieldSizesPtr();

   double*** alpha = bcManager_->alphaPtr(); 
   double*** beta = bcManager_->betaPtr(); 
   double*** gamma = bcManager_->gammaPtr(); 

   IntArray essEqns;
   RealArray essAlpha;
   RealArray essGamma;

   IntArray otherEqns;
   RealArray otherAlpha;
   RealArray otherBeta;
   RealArray otherGamma;

   getBCEqns(essEqns, essAlpha, essGamma,
             otherEqns, otherAlpha, otherBeta, otherGamma);

   if (essEqns.size() > 0)
      linSysCore_->enforceEssentialBC(&essEqns[0], &essAlpha[0], &essGamma[0],
                                      essEqns.size());

   if (otherEqns.size() > 0)
      linSysCore_->enforceOtherBC(&otherEqns[0], &otherAlpha[0],
                                  &otherBeta[0], &otherGamma[0],
                                  otherEqns.size());

   debugOutput("leaving implementAllBCs");
   return;
}

//------------------------------------------------------------------------------
int BASE_FEI::endLoadElemBlock() {
//
//  tasks: end blocked-element data loading step
//

    baseTime_ = MPI_Wtime();

    debugOutput("endLoadElemBlock");

    checkElemBlocksLoaded++;

    assert(numMatrixWorksetsStored[currentElemBlockIndex] ==
           numWorksets[currentElemBlockIndex]);

//    assert(numRHSWorksetsStored[currentElemBlockIndex] ==
//           numWorksets[currentElemBlockIndex]);

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
void BASE_FEI::exchangeRemoteEquations(){
//
// This function is where processors send local contributions to remote
// equations to the owners of those equations, and receive remote
// contributions to local equations.
//
   debugOutput("exchangeRemoteEquations");

   EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();

   eqnCommMgr.exchangeEqns(comm_);

   //so now the remote contributions should be available, let's get them out
   //of the eqn comm mgr and put them into our local matrix structure.

   int numRecvEqns = eqnCommMgr.getNumRecvEqns();
   int* recvEqnLengths = eqnCommMgr.recvEqnLengthsPtr();
   int* recvEqnNumbers = eqnCommMgr.recvEqnNumbersPtr();
   int** recvIndices = eqnCommMgr.recvIndicesPtr();
   double** recvCoefs = eqnCommMgr.recvCoefsPtr();
   double** recvRHSs = eqnCommMgr.recvRHSsPtr();

   for(int i=0; i<numRecvEqns; i++) {
      int eqn = recvEqnNumbers[i];
      if ((localStartRow_ > eqn) || (localEndRow_ < eqn)) {
         cerr << "BASE_FEI::initComplete: ERROR, recvEqn " << eqn
              << " out of range. (localStartRow_: " << localStartRow_
              << ", localEndRow_: " << localEndRow_ << ", localRank_: "
              << localRank_ << ")" << endl;
         abort();
      }

      //sum the equation into the matrix,
      linSysCore_->sumIntoSystemMatrix(eqn, recvEqnLengths[i],
                                       recvCoefs[i], recvIndices[i]);

      //and finally the RHS contributions.
      for(int j=0; j<numRHSs_; j++) {
         linSysCore_->sumIntoRHSVector(1, &(recvRHSs[i][j]), &eqn);
      }
   }

   //now we need to make sure that the right thing happens for essential
   //boundary conditions that get applied to nodes on elements that touch
   //a processor boundary. For an essential boundary condition, the row and
   //column of the corresponding equation must be diagonalized. If there is
   //a processor boundary on any side of the element that contains the node,
   //then there are column contributions to the matrix on the other processor.
   //That other processor must be notified and told to make the adjustments
   //necessary to enforce the boundary condition.

   IntArray essEqns;
   RealArray essAlpha;
   RealArray essGamma;
   IntArray otherEqns;
   RealArray otherAlpha;
   RealArray otherBeta;
   RealArray otherGamma;

   getBCEqns(essEqns, essAlpha, essGamma, otherEqns, otherAlpha,
             otherBeta, otherGamma);

   eqnCommMgr.exchangeEssBCs(&essEqns[0], essEqns.size(),
                             &essAlpha[0], &essGamma[0], comm_);

   int numRemoteEssBCEqns = eqnCommMgr.getNumEssBCEqns();

   if (numRemoteEssBCEqns > 0) {
      int* remEssBCEqns = eqnCommMgr.essBCEqnsPtr();
      int** remEssBCIndices = eqnCommMgr.essBCIndicesPtr();
      int* remEssBCIndicesLengths = eqnCommMgr.essBCEqnLengthsPtr();
      double** remEssBCCoefs = eqnCommMgr.essBCCoefsPtr();

      linSysCore_->enforceRemoteEssBCs(numRemoteEssBCEqns, remEssBCEqns,
                                       remEssBCIndices,
                                       remEssBCIndicesLengths, remEssBCCoefs);
   }

   debugOutput("leaving exchangeRemoteEquations");

   return;
}
 
//------------------------------------------------------------------------------
int BASE_FEI::beginLoadCREqns(int numCRMultSets, 
                              int numCRPenSets) {
//
//  tasks: initiate constraint condition data loading step
//
//

    baseTime_ = MPI_Wtime();

    debugOutput("beginLoadCREqns");
        
    assert (numCRMultSets == storeNumCRMultRecords);
    assert (numCRPenSets == storeNumCRPenRecords);

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::loadCRMult(int CRMultID, 
                         int numMultCRs,
                         const GlobalID *const *CRNodeTable, 
                         const int *CRFieldList,
                         const double *const *CRWeightTable,
                         const double *CRValueList,
                         int lenCRNodeList) {
//
//  tasks: load Lagrange multiplier constraint relation sets
//
//   Question: do we really need to pass CRNodeTable again?  Here, I'm going
//            to ignore it for now (i.e., not store it, but just check it), 
//            as it got passed during the initialization phase, so all we'll 
//            do here is check for errors...
//

   baseTime_ = MPI_Wtime();

   debugOutput("loadCRMult");

    if (debugOutput_) {
       fprintf(debugFile_, "   CRMultID: %d, numMultCRs: %d\n",
               CRMultID, numMultCRs);
       fprintf(debugFile_, "   node table:\n");
       int i;
       for(i=0; i<numMultCRs; i++) {
          fprintf(debugFile_, "      ");
          for(int j=0; j<lenCRNodeList; j++) {
             fprintf(debugFile_, "%d ", (int)CRNodeTable[i][j]);
          }
          fprintf(debugFile_, "\n");
       }
       fprintf(debugFile_, "   field list:\n      ");
       for(i=0; i<lenCRNodeList; i++)
          fprintf(debugFile_, "%d ", CRFieldList[i]);
       fprintf(debugFile_, "\n");
       fflush(debugFile_);
    }
   int i;

   int lenList = ceqn_MultConstraints[CRMultID].getLenCRNodeList();

//  perform some temporary tests (for now, assuming that returned
//  ID's are simply the array indices of the constraint records)

   assert (lenList > 0);
   assert (CRMultID == ceqn_MultConstraints[CRMultID].getCRMultID());
   assert (numMultCRs == ceqn_MultConstraints[CRMultID].getNumMultCRs());

//  recall the data stored earlier in the call to initCRMult() and insure
//  that the passed data (here, the node table) agrees with the
//  initialization data
    
   int ntRowCheck = 0;
   int ntColCheck = 0;
   GlobalID **CRNodeTablePtr = ceqn_MultConstraints[CRMultID].
                                  pointerToCRNodeTable(ntRowCheck, ntColCheck);
   assert (ntRowCheck == numMultCRs);
   assert (ntColCheck == lenList);

   for (i = 0; i < numMultCRs; i++) {
      for(int j = 0; j < lenList; j++) {
         assert(CRNodeTablePtr[i][j] == CRNodeTable[i][j]);
      }
   }

   int fieldRowCheck = 0;
   int *CRFieldListPtr =
        ceqn_MultConstraints[CRMultID].pointerToCRFieldList(fieldRowCheck);
   assert (fieldRowCheck == lenList);
   for (i = 0; i < lenCRNodeList; i++) {
      assert(CRFieldListPtr[i] == CRFieldList[i]);
   }
                                  
//  squirrel away the new data (weight table and constant value list) using 
//  the same methods used in InitCRMult()
//
//  using the first row of the matrix to get some generic info (namely, 
//  numSolnParams) probably isn't the most robust thing to do, but this info
//  is generic (i.e., each column of the node table has to contain nodes with
//  the same solution cardinality, or else the weight table doesn't conform 
//  for the multiplications inherent in the constraint statement), we don't
//  at this stage have a particular index in mind, and (this is the biggie!)
//  we don't have any guarantees that there is more than one row, so...

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int *tableRowLengths = new int [lenList];
   for (i = 0; i < lenList; i++) {
      int index = problemStructure_->getFieldRosterIndex(CRFieldList[i]);
      int numSolnParams = fieldRoster[index].getNumFieldParams();
      tableRowLengths[i] = numSolnParams;
   }
   ceqn_MultConstraints[CRMultID].
                allocateCRNodeWeights(lenList, tableRowLengths);
   int wtRowCheck = 0;
   int *wtColCheck = NULL;
   double **CRWeightTablePtr = ceqn_MultConstraints[CRMultID].
                               pointerToCRNodeWeights(wtRowCheck, wtColCheck);
   assert (wtRowCheck == lenList);
   for (i = 0; i < lenList; i++) {
       assert (wtColCheck[i] == tableRowLengths[i]);
   }

   for (i = 0; i < lenList; i++) {
      for (int j = 0; j < tableRowLengths[i]; j++) {
         CRWeightTablePtr[i][j] = CRWeightTable[i][j];
      }
   }
    
   delete [] tableRowLengths;
   
   ceqn_MultConstraints[CRMultID].allocateCRConstValues(numMultCRs);
   int cvRowCheck = 0;
   double *CRValueListPtr =
            ceqn_MultConstraints[CRMultID].pointerToCRConstValues(cvRowCheck);
   assert (cvRowCheck == numMultCRs);
   for (i = 0; i < numMultCRs; i++) {
      CRValueListPtr[i] = CRValueList[i];
   }

//    ceqn_MultConstraints[CRMultID].dumpToScreen();   //$kdm debugging

//  next, perform assembly of the various terms into the system arrays
//  (this is a good candidate for a separate function...)

//  note use of CRMultID as a position index here... ultimately, we need
//  to set aside a space in the appropriate ceqn_MultConstraints[] object
//  for caching this offset into the local storage for the system stiffness
//  and load arrays
    
   for(i=0; i<numMultCRs; i++) {
      int irow = ceqn_MultConstraints[CRMultID].getEqnNumber() + i;

      for (int j = 0; j < lenList; j++) {
         int myFieldID = CRFieldList[j];

         NodeDescriptor& node = findNodeDescriptor(CRNodeTablePtr[i][j]);

         //first, store the column coeficients for equation irow, the
         //constraint's equation.

         storeNodalColumnCoefs(irow, node, myFieldID, CRWeightTablePtr[j]);

         linSysCore_->sumIntoRHSVector(1, &(CRValueListPtr[i]), &irow);

         //next, store store the transpose of the above. i.e., column irow,
         //in equations associated with 'node' at 'myFieldID'.

         if (node.getOwnerProc() == localRank_) {

            storeNodalRowCoefs(node, myFieldID, CRWeightTablePtr[j], irow);
         }
         else {

            storeNodalSendEqn(node, myFieldID, irow, CRWeightTablePtr[j]);
         }
      }
   }
    
   wTime_ += MPI_Wtime() - baseTime_;

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::loadCRPen(int CRPenID, 
                        int numPenCRs, 
                        const GlobalID *const *CRNodeTable,
                        const int *CRFieldList,
                        const double *const *CRWeightTable,
                        const double *CRValueList,
                        const double *penValues,
                        int lenCRNodeList) {
//
//  tasks: perform penalty constraint relation data loading step
//

   baseTime_ = MPI_Wtime();

   debugOutput("loadCRPen");

   int i;

   int lenList = ceqn_PenConstraints[CRPenID].getLenCRNodeList();

//  perform some temporary tests (for now, assuming that returned ID's are
//  simply the array indices of the constraint records)

   assert (lenList == lenCRNodeList);
   assert (numPenCRs == ceqn_PenConstraints[CRPenID].getNumPenCRs());

//  recall the data stored earlier in the call to initCRPen() and insure 
//  that the passed data (here, the node table) agrees with the 
//  initialization data
    
   int ntRowCheck = 0;
   int ntColCheck = 0;
   GlobalID **CRNodeTablePtr = ceqn_PenConstraints[CRPenID].
                                 pointerToCRNodeTable(ntRowCheck, ntColCheck);
   assert (ntRowCheck == numPenCRs);
   assert (ntColCheck == lenList);
                                  
   for (i = 0; i < numPenCRs; i++) {
      for (int j = 0; j < lenList; j++) {
         assert(CRNodeTablePtr[i][j] == CRNodeTable[i][j]);
      }
   }

//  squirrel away the new data (weight table and constant value list) 
//  using the same methods used in InitCRPen()

   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   int *tableRowLengths = new int [lenList];
   for (i = 0; i < lenList; i++) {
      int index = problemStructure_->getFieldRosterIndex(CRFieldList[i]);
      int numSolnParams = fieldRoster[index].getNumFieldParams();
      tableRowLengths[i] = numSolnParams;
   }
   ceqn_PenConstraints[CRPenID].allocateCRNodeWeights(lenList, tableRowLengths);
   int wtRowCheck = 0;
   int *wtColCheck;
   double **CRWeightTablePtr = ceqn_PenConstraints[CRPenID].
                                pointerToCRNodeWeights(wtRowCheck, wtColCheck);
   assert (wtRowCheck == lenList);
   for (i = 0; i < lenList; i++) {
      assert (wtColCheck[i] == tableRowLengths[i]);
   }
   for (i = 0; i < lenList; i++) {
      for (int j = 0; j < tableRowLengths[i]; j++) {
         CRWeightTablePtr[i][j] = CRWeightTable[i][j];
      }
   }

   delete [] tableRowLengths;

   ceqn_PenConstraints[CRPenID].allocateCRConstValues(numPenCRs);
   int cvRowCheck = 0;
   double *CRValueListPtr = ceqn_PenConstraints[CRPenID].
                                 pointerToCRConstValues(cvRowCheck);
   assert (cvRowCheck == numPenCRs);
   for (i = 0; i < numPenCRs; i++) {
      CRValueListPtr[i] = CRValueList[i];
   }

//  use some brute force to construct these penalty contributions on the 
//  fly, primarily to simplify -reading- this thing, so that the derivations
//  in the annotated implementation document are more readily followed...

   for(int p = 0; p < numPenCRs; p++) {
      for (i = 0; i < lenList; i++) {
         GlobalID iNodeID = CRNodeTablePtr[p][i];
         int iField = CRFieldList[i];

         NodeDescriptor& iNode = findNodeDescriptor(iNodeID);

         for (int j = 0; j < lenList; j++) {
            GlobalID jNodeID = CRNodeTablePtr[p][j];
            int jField = CRFieldList[j];

            NodeDescriptor& jNode = findNodeDescriptor(jNodeID);
            
            if (iNode.getOwnerProc() == localRank_) {

               storePenNodeData(iNode, iField, CRWeightTablePtr[i],
                                jNode, jField, CRWeightTablePtr[j],
                                penValues[p], CRValueListPtr[p]);
            }
            else {
               storePenNodeSendData(iNode, iField, CRWeightTablePtr[i],
                                    jNode, jField, CRWeightTablePtr[j],
                                    penValues[p], CRValueListPtr[p]);
            }
         }
      }
   }

   wTime_ += MPI_Wtime() - baseTime_;

   return(0);
}

//==============================================================================
int BASE_FEI::endLoadCREqns() {

   baseTime_ = MPI_Wtime();

   debugOutput("endLoadCREqns");

   wTime_ += MPI_Wtime() - baseTime_;

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::loadComplete() {

   baseTime_ = MPI_Wtime();
 
// all blocks have been loaded, so let's
// have all processors exchange remote equation data

   debugOutput("loadComplete, calling exchangeRemoteEquations");

   exchangeRemoteEquations();

   debugOutput("leaving loadComplete");
 
   wTime_ += MPI_Wtime() - baseTime_;
 
   return(0);
}

//------------------------------------------------------------------------------
void BASE_FEI::parameters(int numParams, char **paramStrings) {
//
// this function takes parameters for setting internal things like solver
// and preconditioner choice, etc.
//
   baseTime_ = MPI_Wtime();
 
   debugOutput("parameters");
 
   if (numParams == 0 || paramStrings == NULL) {
      debugOutput("--- no parameters.");
   }
   else {
      // take a copy of these parameters, for later use.
      Utils::appendToCharArrayList(paramStrings_, numParams_,
                                   paramStrings, numParams);

      // pass the parameter strings through to the linear algebra core
      linSysCore_->parameters(numParams, paramStrings);

      char param[256];

      if ( Utils::getParam("outputLevel",numParams,paramStrings,param) == 1){
         sscanf(param,"%d", &outputLevel_);
      }

      if ( Utils::getParam("internalFei",numParams,paramStrings,param) == 1){
         sscanf(param,"%d", &internalFei_);
      }

      if ( Utils::getParam("debugOutput",numParams,paramStrings,param) == 1){
         char *name = new char[32];
         sprintf(name, "BASE_FEI%d_debug.%d.%d",
                 internalFei_, numProcs_, localRank_);
         setDebugOutput(param, name);
         delete [] name;
      }

      if (debugOutput_) {
         fprintf(debugFile_,"--- numParams %d\n",numParams);
         for(int i=0; i<numParams; i++){
            fprintf(debugFile_,"------ paramStrings[%d]: %s\n",i,
                    paramStrings[i]);
         }
      }
   }

   debugOutput("leaving parameters function");
 
   wTime_ += MPI_Wtime() - baseTime_;
 
   return;
}

//------------------------------------------------------------------------------
void BASE_FEI::setDebugOutput(char* path, char* name){
//
//This function turns on debug output, and opens a file to put it in.
//
   if (debugOutput_) {
      fprintf(debugFile_,"setDebugOutput closing this file.");
      fflush(debugFile_);
      fclose(debugFile_);
      debugFile_ = NULL;
   }

   int pathLength = strlen(path);
   if (path != debugPath_) {
      delete [] debugPath_;
      debugPath_ = new char[pathLength + 1];
      sprintf(debugPath_, path);
   }

   int nameLength = strlen(name);
   if (name != debugFileName_) {
      delete [] debugFileName_;
      debugFileName_ = new char[nameLength + 1];
      sprintf(debugFileName_,name);
   }

   char* dbFileName = new char[pathLength + nameLength + 3];

   sprintf(dbFileName,"%s/%s", path, name);

   debugOutput_ = 1;
   debugFile_ = fopen(dbFileName,"w");

   if (!debugFile_){
      cerr << "couldn't open debug output file: " << dbFileName << endl;
      debugOutput_ = 0;
   }

   delete [] dbFileName;
}

//------------------------------------------------------------------------------
int BASE_FEI::iterateToSolve(int& status) {

   baseTime_ = MPI_Wtime();

   debugOutput("iterateToSolve");

// now the matrix can do its internal gyrations in preparation for
// parallel matrix-vector products.

   debugOutput("   calling matrixLoadComplete");

//#### Tong : changed here 
   implementAllBCs();
   linSysCore_->matrixLoadComplete();

// now we will implement the boundary conditions. This can be done after
// the matrix's 'loadComplete' because we're not altering the structure,
// only the coefficient values.
//


   debugOutput("in iterateToSolve, calling launchSolver...");
 
   wTime_ += MPI_Wtime() - baseTime_;

   sTime_ = MPI_Wtime();

   linSysCore_->launchSolver(status, iterations_);

   sTime_ = MPI_Wtime() - sTime_;

   debugOutput("... back from solver");
 
   if ((localRank_ == masterRank_) && (outputLevel_ > 0)){
      if (status == 1) {
         printf("solve successful, time: %f, iterations: %d, FEI time: %f\n",
                sTime_, iterations_, wTime_);
      }
      else {
         printf("solve UNSUCCESSFUL, time: %f, iterations: %d, FEI time: %f\n",
                sTime_, iterations_, wTime_);
      }
   }

   //now unpack the contents of the solution vector into the nodal data
   //structures.
   unpackSolution();

   solveCounter_++;

   debugOutput("leaving iterateToSolve");

   if (status != 1) return(1);
   else return(0);
}

//==============================================================================
void BASE_FEI::setNumRHSVectors(int numRHSs, int* rhsIDs){

    if (numRHSs < 0) {
        cerr << "BASE_FEI::setNumRHSVectors: ERROR, numRHSs < 0." << endl;
    }

   linSysCore_->setNumRHSVectors(numRHSs, rhsIDs);

    numRHSs_ = numRHSs;

    if (rhsIDs_ != NULL) delete [] rhsIDs_;

    rhsIDs_ = new int[numRHSs_];
    for(int i=0; i<numRHSs_; i++) rhsIDs_[i] = rhsIDs[i];
}

//==============================================================================
void BASE_FEI::unpackSolution() {
//
//This function should be called after the iterative solver has returned,
//and we know that there is a solution in the underlying vector.
//This function ensures that any shared solution values are available on
//the sharing processors.
//
   baseTime_ = MPI_Wtime();

   if (debugOutput_) {
      fprintf(debugFile_, "entering unpackSolution, outputLevel: %d\n",
              outputLevel_);
      fflush(debugFile_);
   }

   //what we need to do is as follows.
   //The eqn comm mgr has a list of what it calls 'recv eqns'. These are
   //equations that we own, for which we received contributions from other
   //processors. The solution values corresponding to these equations need
   //to be made available to those remote contributing processors.

   EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();

   int numRecvEqns = eqnCommMgr.getNumRecvEqns();
   int* recvEqnNumbers = eqnCommMgr.recvEqnNumbersPtr();

   for(int i=0; i<numRecvEqns; i++) {
      int eqn = recvEqnNumbers[i];

      if ((localStartRow_ > eqn) || (localEndRow_ < eqn)) {
         cerr << "BASE_FEI::unpackSolution: ERROR, 'recv' eqn (" << eqn
              << ") out of local range." << endl;
         abort();
      }

      double solnValue;

      //I should figure out a more efficient way of retrieving these
      //solution values and registering them with the eqn comm mgr.

      linSysCore_->getSolnEntry(eqn, solnValue);

      eqnCommMgr.addSolnValues(&eqn, &solnValue, 1);
   }

   eqnCommMgr.exchangeSoln(comm_);

   debugOutput("leaving unpackSolution");
 
   wTime_ = MPI_Wtime() - baseTime_;
}
             
//------------------------------------------------------------------------------
int BASE_FEI::getBlockNodeSolution(GlobalID elemBlockID,  
                                   GlobalID *nodeIDList, 
                                   int &lenNodeIDList, 
                                   int *offset,  
                                   double *results) {
        
   debugOutput("getBlockNodeSolution");

   lenNodeIDList = 0;

   int numActiveNodes = problemStructure_->getNumActiveNodes();

   if (numActiveNodes <= 0) return(0);

   int numSolnParams = 0;

   NodeDescriptor* nodes = problemStructure_->getActiveNodesPtr();
   NodeCommMgr& nodeCommMgr = problemStructure_->getNodeCommMgr();
   BlockDescriptor& block = 
        problemStructure_->getBlockDescriptor(elemBlockID);
   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   EqnCommMgr& eqnCommMgr = problemStructure_->getEqnCommMgr();

   int numRemoteEqns = eqnCommMgr.getNumSendEqns();
   int* remoteEqnNumbers = eqnCommMgr.sendEqnNumbersPtr();
   double* remoteSoln = eqnCommMgr.sendEqnSolnPtr();

   //traverse the node list, checking for nodes associated with this block
   //when an associated node is found, add it and its 'answers' to the list...

   for(int i=0; i<numActiveNodes; i++) {

      if (nodes[i].getBlockIndex(elemBlockID) >= 0) {
         GlobalID nodeID = nodes[i].getGlobalNodeID();
         nodeIDList[lenNodeIDList] = nodeID;
         offset[lenNodeIDList++] = numSolnParams;

         if (nodes[i].getOwnerProc() == localRank_) {

            int numFields = nodes[i].getNumFields();
            int* fieldIDs = nodes[i].getFieldIDList();
            int* fieldEqnNumbers = nodes[i].getFieldEqnNumbers();

            for(int j=0; j<numFields; j++) {
               if (block.containsField(fieldIDs[j])) {
                  int ind = problemStructure_->
                                 getFieldRosterIndex(fieldIDs[j]);
                  int size = fieldRoster[ind].getNumFieldParams();

                  double answer;
                  for(int k=0; k<size; k++) {
                     linSysCore_->getSolnEntry(fieldEqnNumbers[j]+k, answer);
                     results[numSolnParams++] = answer;
                  }
               }
            }//for(j<numFields)loop
         }
         else {
            //the node is a remotely-owned shared node.

            int index = nodeCommMgr.getCommNodeIndex(nodeID);
            NodeDescriptor& node = nodeCommMgr.getCommNode(index);

            int numFields = node.getNumFields();
            int* fieldIDs = node.getFieldIDList();
            int* fieldEqnNumbers = node.getFieldEqnNumbers();

            for(int j=0; j<numFields; j++) {
               if (block.containsField(fieldIDs[j])) {
                  int ind = problemStructure_->
                                 getFieldRosterIndex(fieldIDs[j]);
                  int size = fieldRoster[ind].getNumFieldParams();

                  int tmp;
                  for(int k=0; k<size; k++) {
                     int sInd = Utils::sortedIntListFind(fieldEqnNumbers[j]+k,
                                                         remoteEqnNumbers,
                                                         numRemoteEqns, &tmp);

                     if (sInd < 0) {
                        cerr << "BASE_FEI::getBlockNodeSolution: ERROR, remote"
                             << " eqn " << fieldEqnNumbers+k << " not found."
                             << endl;
                        abort();
                     }

                     results[numSolnParams++] = remoteSoln[sInd];
                  }
               }
            }//for(j<numFields)loop
         }
      }
   }

   offset[lenNodeIDList] = numSolnParams;

   return(0);
}

//------------------------------------------------------------------------------
//
int BASE_FEI::getBlockFieldNodeSolution(GlobalID elemBlockID,
                                        int fieldID,
                                        GlobalID *nodeIDList, 
                                        int& lenNodeIDList, 
                                        int *offset,
                                        double *results) {
        
    debugOutput("getBlockFieldNodeSolution");

    //these if's prevent "declared and never referenced" warnings.
    if (elemBlockID);
    if (fieldID);
    if (nodeIDList);
    if (lenNodeIDList);
    if (offset);
    if (results);

//  stub method, needs v1.0 implementation as restriction of the method
//  getBlockNodeSolution() to a given fieldID

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::putBlockNodeSolution(GlobalID elemBlockID,
                                   const GlobalID *nodeIDList, 
                                   int lenNodeIDList, 
                                   const int *offset,
                                   const double *estimates) {
        
   debugOutput("putBlockNodeSolution");

   (void)offset;

   int nodeOffset = 0;

   int numActiveNodes = problemStructure_->getNumActiveNodes();

   if (numActiveNodes <= 0) return(0);

   int numSolnParams = 0;

   NodeDescriptor* nodes = problemStructure_->getActiveNodesPtr();
   BlockDescriptor& block =
        problemStructure_->getBlockDescriptor(elemBlockID);
   FieldRecord* fieldRoster = problemStructure_->getFieldRosterPtr();

   //traverse the node list, checking for nodes associated with this block
   //when an associated node is found, add it and its 'answers' to the list...

   for(int i=0; i<lenNodeIDList; i++) {

      int index = problemStructure_->
                    getActiveNodeIndex(nodeIDList[nodeOffset++]);
   
      if (nodes[index].getBlockIndex(elemBlockID) >= 0) {

         if (nodes[index].getOwnerProc() == localRank_) {

            int numFields = nodes[i].getNumFields();
            int* fieldIDs = nodes[i].getFieldIDList();
            int* fieldEqnNumbers = nodes[i].getFieldEqnNumbers();

            for(int j=0; j<numFields; j++) {
               if (block.containsField(fieldIDs[j])) {
                  int ind = problemStructure_->
                                 getFieldRosterIndex(fieldIDs[j]);
                  int size = fieldRoster[ind].getNumFieldParams();

                  for(int k=0; k<size; k++) {
                     int eqn = fieldEqnNumbers[j]+k;
                     linSysCore_->putInitialGuess(&eqn,
                                                  &estimates[numSolnParams++],
                                                  1);
                  }
               }
            }//for(j<numFields)loop
         }
      }
   }

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::putBlockFieldNodeSolution(GlobalID elemBlockID, 
                                        int fieldID, 
                                        const GlobalID *nodeIDList, 
                                        int lenNodeIDList, 
                                        const int *offset,
                                        const double *estimates) {
        
    debugOutput("putBlockFieldNodeSolution");

    //these if's prevent "declared and never referenced" warnings.
    if (elemBlockID);
    if (fieldID);
    if (nodeIDList);
    if (lenNodeIDList);
    if (offset);
    if (estimates);

//  stub method, needs v1.0 implementation as restriction of the method
//  putBlockNodeSolution() to a given fieldID

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::getBlockElemSolution(GlobalID elemBlockID,  
                                   GlobalID *elemIDList,
                                   int& lenElemIDList, 
                                   int *offset,  
                                   double *results, 
                                   int& numElemDOF) {
//
//  return the elemental solution parameters associated with a 
//  particular block of elements
//
   debugOutput("getElemBlockSolution");

   BlockDescriptor& block = problemStructure_->getBlockDescriptor(elemBlockID);

   GlobalID* elemIDs = problemStructure_->
                          getConnectivityTable(elemBlockID).elemIDs;

   int numElems = block.getNumElements();
   int DOFPerElement = block.getNumElemDOFPerElement();
   int* elemDOFEqnNumbers = block.elemDOFEqnNumbersPtr();
   double answer;

   if (DOFPerElement <= 0) return(0);

   lenElemIDList = 0;
   numElemDOF = 0;

   for(int i=0; i<numElems; i++) {
      elemIDList[lenElemIDList] = elemIDs[i];
      offset[lenElemIDList++] = numElemDOF;

      for(int j=0; j<DOFPerElement; j++) {
         int eqn = elemDOFEqnNumbers[i] + j;

         linSysCore_->getSolnEntry(eqn, answer);

         results[numElemDOF++] = answer;
      }
   }

   offset[lenElemIDList] = numElemDOF;

   return(0);
} 
      
//------------------------------------------------------------------------------
int BASE_FEI::putBlockElemSolution(GlobalID elemBlockID,
                                   const GlobalID *elemIDList, 
                                   int lenElemIDList, 
                                   const int *offset, 
                                   const double *estimates, 
                                   int numElemDOF) {
        
   debugOutput("putElemBlockSolution");

   (void)offset;

   BlockDescriptor& block = problemStructure_->getBlockDescriptor(elemBlockID);

   GlobalID* elemIDs = problemStructure_->
                          getConnectivityTable(elemBlockID).elemIDs;

   int numElems = block.getNumElements();
   int DOFPerElement = block.getNumElemDOFPerElement();
   int* elemDOFEqnNumbers = block.elemDOFEqnNumbersPtr();

   if (DOFPerElement <= 0) return(0);

   int elemCount = 0;
   int nedof = 0;
   for(int i=0; i<numElems; i++) {
      assert(elemIDList[elemCount] == elemIDs[i]);
      elemCount++;

      for(int j=0; j<DOFPerElement; j++) {
         int eqn = elemDOFEqnNumbers[i] + j;
         double soln = estimates[nedof++];

         linSysCore_->putInitialGuess(&eqn, &soln, 1);
      }
   }

//  perform some obvious checks to guard against major blunders

   assert (nedof == numElemDOF);
   assert (elemCount == lenElemIDList);

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::getCRMultSizes(int& numCRMultIDs, int& lenResults) {
//
//  This function returns the dimensions of the lists that get filled by
//  the getCRMultSolution function. In that function, *CRMultIDs and
//  *offset are both of length numCRMultIDs, and *results is of length
//  lenResults.
//

    numCRMultIDs = storeNumCRMultRecords;
    lenResults = 0;
    for (int i=0; i<storeNumCRMultRecords; i++) {
        lenResults += ceqn_MultConstraints[i].getNumMultCRs();
    }

    return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::getCRMultSolution(int& numCRMultSets, 
                                int *CRMultIDs,  
                                int *offset, 
                                double *results) {
        
   debugOutput("getCRMultSolution");

   numCRMultSets = storeNumCRMultRecords;

   int resultsOffset = 0;

   for(int i=0; i<storeNumCRMultRecords; i++) {
      CRMultIDs[i] = ceqn_MultConstraints[i].getCRMultID();
      int numMultCRs = ceqn_MultConstraints[i].getNumMultCRs();
      int eqnNumber = ceqn_MultConstraints[i].getEqnNumber();

      offset[i] = resultsOffset;

      for(int j=0; j<numMultCRs; j++) {
         linSysCore_->getSolnEntry(eqnNumber+j, results[resultsOffset++]);
      }
   }

   debugOutput("leaving getCRMultSolution");

   return(0);
} 
  
//------------------------------------------------------------------------------
int BASE_FEI::getCRMultParam(int CRMultID, 
                             int numMultCRs,
                             double *multValues) {

   debugOutput("getCRMultParam");

   assert(numMultCRs ==  ceqn_MultConstraints[CRMultID].getNumMultCRs());

   int eqnNumber = ceqn_MultConstraints[CRMultID].getEqnNumber();
   for(int j = 0; j < numMultCRs; j++) {
      linSysCore_->getSolnEntry(eqnNumber+j, multValues[j]);
   }

   return(0);
}

//------------------------------------------------------------------------------
int BASE_FEI::putCRMultParam(int CRMultID, 
                             int numMultCRs,
                             const double *multEstimates) {
//
//  this method is just the inverse of getCRMultParam(), so...
//
        
   debugOutput("putCRMultParam");

//  note that we're assuming that CRMultID is the index (this is how it's
//  assigned elsewhere, so if this changes, then that change must propagate
//  here, to something more akin to how we're handling inherited ID's like
//  the blockID passed parameters)
  
   assert(numMultCRs == ceqn_MultConstraints[CRMultID].getNumMultCRs());

   int eqnNumber = ceqn_MultConstraints[CRMultID].getEqnNumber();
   for(int j = 0; j < numMultCRs; j++) {
      int eqn = eqnNumber+j;
      linSysCore_->putInitialGuess(&eqn, &(multEstimates[j]), 1);
   }

   return(0);
}

//-----------------------------------------------------------------------------
//  some utility functions to aid in using the "put" functions for passing
//  an initial guess to the solver
//-----------------------------------------------------------------------------

//------------------------------------------------------------------------------
int BASE_FEI::getBlockElemIDList(GlobalID elemBlockID,
                                 GlobalID *elemIDList, 
                                 int& lenElemIDList) {
//
//  return the list of element IDs for a given block... the length parameter
//  lenElemIDList should be used to check memory allocation for the calling
//  method, as the calling method should have gotten a copy of this param 
//  from a call to getNumBlockElements before allocating memory for elemIDList
//
        
   debugOutput("getBlockElemIDList");

   ConnectivityTable& connTable = problemStructure_->
                          getConnectivityTable(elemBlockID);

   GlobalID* elemIDs = connTable.elemIDs;
   lenElemIDList = connTable.numElems;

   for(int i=0; i<lenElemIDList; i++) {
      elemIDList[i] = elemIDs[i];
   }

   return(0);
}

//------------------------------------------------------------------------------
//
//  similar comments as for getBlockElemIDList(), except for returning the
//  active node list
//
int BASE_FEI::getBlockNodeIDList(GlobalID elemBlockID,
                                 GlobalID *nodeIDList, 
                                 int& lenNodeIDList) {

   debugOutput("getBlockNodeIDList");

   int numActiveNodes = problemStructure_->getNumActiveNodes();
   NodeDescriptor* nodes = problemStructure_->getActiveNodesPtr();

   BlockDescriptor& block = problemStructure_->getBlockDescriptor(elemBlockID);

   lenNodeIDList = 0;

   for(int i=0; i<numActiveNodes; i++) {
      if (nodes[i].getBlockIndex(elemBlockID) >= 0)
         nodeIDList[lenNodeIDList++] = nodes[i].getGlobalNodeID();
   }

   return(0);
}

//------------------------------------------------------------------------------
//
//  return the number of nodes associated with elements of a given block ID
//
int BASE_FEI::getNumNodesPerElement(GlobalID blockID) const {

   BlockDescriptor& block = problemStructure_->getBlockDescriptor(blockID);

   return(block.getNumNodesPerElement());
}
 
//------------------------------------------------------------------------------
int BASE_FEI::getNumEqnsPerElement(GlobalID blockID) const {
//
//  return the number of eqns associated with elements of a given block ID
//

   BlockDescriptor& block = problemStructure_->getBlockDescriptor(blockID);

   return(block.getNumEqnsPerElement());
}

//------------------------------------------------------------------------------
int BASE_FEI::getNumSolnParams(GlobalID iGlobal) const {
//
//  return the number of solution parameters at a given node
//
   NodeDescriptor& node = findNodeDescriptor(iGlobal);

   return(node.getNumNodalDOF());
}
 
//==============================================================================
void BASE_FEI::flagNodeAsActive(GlobalID nodeID) {
//
//Flag node 'nodeID' as active, by adding it to the list of active nodeIDs.
//

   GlobalIDArray& nodeList = problemStructure_->getActiveNodeIDList();

   // see if the node is already in the active list

   if (nodeList.size()>0) {
      int index = -1;
      int found = Utils::sortedGlobalIDListFind(nodeID,&nodeList[0],
                                                nodeList.size(), &index);

      // if we didn't find it, then insert it into the list

      if (found<0) {
         nodeList.insert(index, nodeID);
      }
   }
   else nodeList.append(nodeID);
}

//------------------------------------------------------------------------------
int BASE_FEI::getNumElemBlocks() const {
//
//  return the number of stored element blocks
//

    return(problemStructure_->getNumBlocks());
}

//------------------------------------------------------------------------------
int BASE_FEI::getNumBlockActNodes(GlobalID blockID) const {
//
//  return the number of active nodes associated with a given element block ID
//

   BlockDescriptor& block = problemStructure_->getBlockDescriptor(blockID);

   return(block.getNumActiveNodes());
}

//------------------------------------------------------------------------------
int BASE_FEI::getNumBlockActEqns(GlobalID blockID) const {
//
// return the number of active equations associated with a given element
// block ID
//
   BlockDescriptor& block = problemStructure_->getBlockDescriptor(blockID);

   return(block.getTotalNumEqns());
}

//------------------------------------------------------------------------------
int BASE_FEI::getNumBlockElements(GlobalID blockID) const {
//
//  return the number of elements associated with a given elem blockID
//

   BlockDescriptor& block = problemStructure_->getBlockDescriptor(blockID);
   
   return(block.getNumElements());
}

//------------------------------------------------------------------------------
int BASE_FEI::getNumBlockElemEqns(GlobalID blockID) const {
//
//  return the number of elem equations associated with a given blockID
//

   BlockDescriptor& block = problemStructure_->getBlockDescriptor(blockID);

   int numElementDOF = block.getNumElemDOFPerElement();
   int numBlockElemEqns = numElementDOF*block.getNumElements();

   return(numBlockElemEqns);
}

//------------------------------------------------------------------------------
void BASE_FEI::packSharedStiffness(const int* remoteEqnOffsets,
                                   const int* remoteProcs,
                                   int numRemoteEqns,
                                   int* scatterIndices,
                                   const double* const* stiffness,
                                   int numIndices) {
//
// This function will pack the appropriate coefficients and indices into
// the eqn comm mgr to be sent to other processors later.
//

   for (int i=0; i<numRemoteEqns; i++) {

      int index = remoteEqnOffsets[i];
      int eqn = scatterIndices[index];
      problemStructure_->
               getEqnCommMgr().addSendEqn(eqn, remoteProcs[i],
                                          stiffness[index],
                                          scatterIndices, numIndices);
   }
}

//------------------------------------------------------------------------------
void BASE_FEI::packSharedLoad(const int* remoteEqnOffsets,
                              const int* remoteProcs,
                              int numRemoteEqns,
                              int* scatterIndices,
                              const double* load) {
//
// This function will pack the appropriate coefficients and indices into
// the eqn comm mgr to be sent to other processors later.
//

   for (int i=0; i<numRemoteEqns; i++) {

      int index = remoteEqnOffsets[i];
      int eqn = scatterIndices[index];
      problemStructure_->
               getEqnCommMgr().addSendRHS(eqn, remoteProcs[i],
                                          currentRHS_, load[index]);
   }
}

//------------------------------------------------------------------------------
void BASE_FEI::assembleStiffnessAndLoad(int numRows, int* scatterIndices, 
                                  const double* const* stiff,
                                  const double* load) {
//
//This function hands the element data off to the routine that finally
//sticks it into the matrix and RHS vector.
//
   for(int i = 0; i < numRows; i++) {
      int ib = scatterIndices[i];
      if ((localStartRow_ <= ib) && (ib <= localEndRow_)) {
         linSysCore_->sumIntoRHSVector(1, &(load[i]), &ib);
         linSysCore_->sumIntoSystemMatrix(ib, numRows, &(stiff[i][0]),
                                          &scatterIndices[0]);
      }
   }
}

//------------------------------------------------------------------------------
void BASE_FEI::assembleStiffness(int numRows, int* scatterIndices,
                                  const double* const* stiff) {
//
//This function hands the element data off to the routine that finally
//sticks it into the matrix.
//
   for(int i = 0; i < numRows; i++) {
      int ib = scatterIndices[i];
      if ((localStartRow_ <= ib) && (ib <= localEndRow_)) {
         linSysCore_->sumIntoSystemMatrix(ib, numRows, &(stiff[i][0]),
                                          &scatterIndices[0]);
      }
   }
}

//------------------------------------------------------------------------------
void BASE_FEI::assembleLoad(int numRows, int* scatterIndices,
                                  const double* load) {
//
//This function hands the data off to the routine that finally
//sticks it into the RHS vector.
//
   for(int i = 0; i < numRows; i++) {
      int ib = scatterIndices[i];
      if ((localStartRow_ <= ib) && (ib <= localEndRow_)) {
         linSysCore_->sumIntoRHSVector(1, &(load[i]), &ib );
      }
   }
}

//==============================================================================
void BASE_FEI::debugOutput(char* mesg) {
   if (debugOutput_) {
      fprintf(debugFile_,"%s\n",mesg);
      fflush(debugFile_);
   }
}
