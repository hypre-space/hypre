#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef SER
#include <mpiuni/mpi.h>
#else
#include <mpi.h>
#endif

#include "other/basicTypes.h"

#include "mv/RealArray.h"
#include "mv/IntArray.h"
#include "mv/GlobalIDArray.h"

#include "mv/CommInfo.h"
#include "mv/Map.h"
#include "mv/Vector.h"
#include "mv/Matrix.h"

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

#include "src/BASE_SLE.h"

//CASC#include "pc/SAILS_PC.h"
#ifdef HYPRE
#include "pc/PILUT_PC.h"
#endif

//------------------------------------------------------------------------------
BASE_SLE::BASE_SLE(MPI_Comm PASSED_COMM_WORLD, int masterRank) : 
    FEI() {

    blockRoster = NULL; // added by edmond 9-24-99

//  start the wall clock time recording

    baseTime_ = MPI_Wtime();
    wTime_ = 0.0;
    sTime_ = 0.0;

//  initialize MPI communications info

    masterRank_ = masterRank;
    FEI_COMM_WORLD = PASSED_COMM_WORLD;
    MPI_Comm_rank(FEI_COMM_WORLD, &localRank_);
    MPI_Comm_size(FEI_COMM_WORLD, &numProcs_);

//  we'll need some tag numbers for distinguishing between MPI messages. 
//  These are just arbitrarily chosen numbers. (NOTE: since we're not
//  'MPI_Comm_duping' the communicator, we are susceptible to conflicts
//  with messages in the calling scope. This is to be fixed. The reason
//  we're not dup'ing, is observed scalability problems with dup on the
//  ASCI red machine.

    packet_tag1 = 9191;
    packet_tag2 = 9192;
    indices_tag = 9193;
    coeff_tag   = 9194;
    length_tag  = 9195;
    field_tag   = 9196;
    id_tag      = 9197;
    extSendPacketTag = 9295;
    extRecvPacketTag = 9296;

    outputLevel_ = 0;

    debugOutput_ = 0; //no debug output by default.
    solveCounter_ = 1;
    internalFei_ = 0;
    debugPath_ = NULL;
    debugFileName_ = NULL;

//  initialize some state variables

    storeNumElemBlocks = 0;    // number of blocks in this problem
    storeSolvType = 0;         // type of solution process to invoke
    storeNumProcActNodes = 0;  // number of active nodes on this processor
    storeNumProcActEqns = 0;   // number of equations arising from active nodes
    storeBCNodeSets = 0;       // number of bc node sets
    storeSharedNodeSets = 0;   // number of shared node sets
    storeExtNodeSets = 0;      // number of external node sets
    storeNumCRMultRecords = 0; // number of Lagrange constraint records
    storeNumCRPenRecords = 0;  // number of penalty constraint records
    storeNumProcEqns = 0;      // number of equations on this processor

//  some data for consistency checking (some of these not yet utilized)

    checkElemBlocksLoaded = 0;
    checkNumElemBlocks = 0;    // verify number of blocks in this problem
    checkSolvType = 0;         // verify type of solution process
    checkNumProcActNodes = 0;  // verify number of active nodes on this proc
    checkNumProcActEqns = 0;   // verify number of equations
                               // arising from active nodes
    checkBCNodeSets = 0;       // verify number of bc node sets
    checkSharedNodeSets = 0;   // verify number of shared node sets
    checkExtNodeSets = 0;      // verify number of external node sets
    checkNumCRMultRecords = 0; // verify number of Lagrange constraint records
    checkNumCRPenRecords = 0;  // verify number of penalty constraint records
    doneEndInitElemData = 0;

    numParams_ = 0;
    paramStrings_ = NULL;
    solverName_ = new char[64];
    strcpy(solverName_, "garbage");

    precondName_ = new char[64];
    strcpy(precondName_, "garbage");

    externalNodes_ = new ExternNodeRecord;

//  set up some special MPI send/recv types

    setupMPITypes();

    gnod_LocalNodes = NULL;

    fieldRoster = NULL;
    fieldRosterAllocated_ = false;

    sysMatIndices = NULL;

    sharedNodes_ = new SharedNodeRecord;

    loadRecvsLaunched_ = false;
    shNodeInfo_ = NULL;
    shRequests_ = NULL;
    shScatRequests_ = NULL;
    shCoefRequests_ = NULL;
    shProc_ = NULL;
    shNodesFromProc_ = NULL;
    numShProcs_ = 0;
    shSize_ = 0;
    shCoeffSize_ = 0;
    shScatterIndices_ = NULL;
    shCoeff_ = NULL;
    sharedBuffI_ = NULL;
    sharedBuffL_ = NULL;
 
    shBuffLoadD_ = NULL;
    shBuffLoadI_ = NULL;
    shBuffRHSLoadD_ = NULL;
    shBuffLoadAllocated_ = false;
    shBuffRHSAllocated_ = false;

    IA_NodDOFList = new IntArray();
    IA_localElems = new IntArray();

//  and the time spent in the constructor is...

    wTime_  = MPI_Wtime() - baseTime_;

    return;
}

//------------------------------------------------------------------------------
BASE_SLE::~BASE_SLE() {
//
//  Destructor function. Free allocated memory, etc.
//

    int i;

    //delete the linear algebra stuff -- matrix pointers, etc.
    //deleteLinearAlgebraCore();
    //a derived class will have the responsibility for doing this.

    delete [] gnod_LocalNodes;

    delete [] blockRoster;
    if (fieldRosterAllocated_) delete [] fieldRoster;
    fieldRosterAllocated_ = false;

    if (storeNumCRMultRecords > 0) 
        delete [] ceqn_MultConstraints;
    if (storeNumCRPenRecords > 0) 
        delete [] ceqn_PenConstraints;

    delete [] storeNumWorksets;

    delete [] sysMatIndices;

    for(i=0; i<numParams_; i++) delete [] paramStrings_[i];
    delete [] paramStrings_;

    delete [] solverName_;
    delete [] precondName_;

    delete sharedNodes_;

    if (shRequests_ != NULL) delete [] shRequests_;
    if (shProc_ != NULL) delete [] shProc_;
    if (shNodesFromProc_ != NULL) delete [] shNodesFromProc_;
    if (shScatRequests_ != NULL) delete [] shScatRequests_;
    if (shCoefRequests_ != NULL) delete [] shCoefRequests_;

    delete externalNodes_;

    delete IA_NodDOFList;
    delete IA_localElems;

    if (debugOutput_) {
        delete [] debugPath_;
        delete [] debugFileName_;
        fclose(debugFile_);
    }

    return;
}

//------------------------------------------------------------------------------
void BASE_SLE::setupMPITypes() {
//
//  This function sets up some special MPI types to use in communications.
//  At the top of this file are declarations for MPI_GLOBALID and
//  MPI_NodePacket, etc., to be available for all of the BASE_SLE
//  implementation.
//

    int ierror = MPI_Type_contiguous(sizeof(GlobalID)/sizeof(int),MPI_INT,
                                 &MPI_GLOBALID);
    if (ierror)
        printf("after MPI_Type_contiguous, ierror = %d\n",ierror);
    MPI_Type_commit(&MPI_GLOBALID);
    
    MPI_Aint disp[8];
    int block[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    (void)block;
    MPI_Datatype type[8] = {MPI_GLOBALID, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                            MPI_INT, MPI_INT, MPI_INT};
    (void)type;
    NodeControlPacket samplePacket;

//  calculate the displacements - i.e., the distance in memory to each
//  element of the NodeControlPacket structure

    MPI_Address(&samplePacket, disp);
    MPI_Address(&samplePacket.numEqns, disp+1);
    MPI_Address(&samplePacket.eqnNumber, disp+2);
    MPI_Address(&samplePacket.numElems, disp+3);
    MPI_Address(&samplePacket.numIndices, disp+4);
    MPI_Address(&samplePacket.numPenCRs, disp+5);
    MPI_Address(&samplePacket.numMultCRs, disp+6);
    MPI_Address(&samplePacket.numFields, disp+7);

//  make these into relative addresses

    base = disp[0];
    for (int i = 0; i < 8; i++) {
        disp[i] -= base;
    }

//  now put it into MPI

    MPI_Type_struct( 8, block, disp, type, &MPI_NodePacket);
    MPI_Type_commit( &MPI_NodePacket);

    if (debugOutput_) {
            fprintf(debugFile_,"sizeof(NodeControlPacket): %d,",
                sizeof(NodeControlPacket));
            fprintf(debugFile_," sizeof(NodeWeightPacket): %d\n",
                sizeof(NodeWeightPacket));
            fflush(debugFile_);
    }

//  intialize the node-weight packet, following the same steps used
//  for nodePackets...

    int iwt;
    for (iwt = 0; iwt < WTPACK_SIZE; iwt++) {
        wtBlock[iwt] = 1;
    }

    MPI_Datatype wtType[WTPACK_SIZE];;
    wtType[0] = MPI_GLOBALID;
    wtType[1] = MPI_INT;
    for (iwt = 2; iwt < WTPACK_SIZE; iwt++) {
        wtType[iwt] = MPI_DOUBLE;
    }

    NodeWeightPacket sampleWtPacket;
    MPI_Address(&sampleWtPacket, wtDisp);
    MPI_Address(&sampleWtPacket.sysEqnID, wtDisp + 1);
    MPI_Address(&sampleWtPacket.numSolnParams, wtDisp + 2);
    for (iwt = 0; iwt < MAX_SOLN_CARD; iwt++) {
        MPI_Address(&sampleWtPacket.weights[iwt], wtDisp + iwt + 3);
    }

    wtBase = wtDisp[0];
    for (iwt = 0; iwt < WTPACK_SIZE; iwt++) {
        wtDisp[iwt] -= wtBase;
    }

    MPI_Type_struct(WTPACK_SIZE, wtBlock, wtDisp, wtType, &MPI_NodeWtPacket);
    MPI_Type_commit(&MPI_NodeWtPacket);

    return;
}

//------------------------------------------------------------------------------
int BASE_SLE::initSolveStep(int numElemBlocks, int solveType) {
//
//  tasks: allocate baseline data structures for elements and nodes
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: initSolveStep\n");
        fflush(debugFile_);
    }

//  first, store the number of element blocks and node sets

    storeNumElemBlocks = numElemBlocks;
    storeSolvType = solveType;

//  next, allocate the requisite initial data structures

    storeNumWorksets = new int[numElemBlocks];

    blockRoster = new BlockRecord[numElemBlocks];

    //initialize the linear algebra stuff -- matrix pointers
    //to null, etc.
    initLinearAlgebraCore();

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::initSolveStep(int numElemBlocks, int solveType,
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
int BASE_SLE::initFields(int numFields, 
                         const int *cardFields, 
                         const int *fieldIDs) {
//
//  tasks: identify all the solution fields present in the analysis
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: initFields\n");
        fflush(debugFile_);
    }

    assert (numFields > 0);
    if (!fieldRosterAllocated_) fieldRoster = new FieldRecord [numFields];
    fieldRosterAllocated_ = true;

    for (int i = 0; i < numFields; i++) {
        fieldRoster[i].setFieldID(fieldIDs[i]);
        fieldRoster[i].setNumFieldParams(cardFields[i]);
//$kdm debug        fieldRoster[i].dumpToScreen();
    }
    storeNumFields = numFields;
    
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_SLE::beginInitElemBlock(GlobalID elemBlockID,  
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

    if (debugOutput_) {
        fprintf(debugFile_,"trace: beginInitElemBlock\n");
        fflush(debugFile_);
    }

    int i, j, k;

    if (GID_blockIDList.size()==0) {
        GID_blockIDList.append(elemBlockID);
        i = 0;
    }
    else {
        int found = search_ID_index(elemBlockID,&GID_blockIDList[0],
                                  GID_blockIDList.size());
        if (found<0) {
            GID_blockIDList.append(elemBlockID);
            found = search_ID_index(elemBlockID,&GID_blockIDList[0],
                                  GID_blockIDList.size());
        }
        i = found;
    }

 
    if (debugOutput_) {
        fprintf(debugFile_,"elemBlock %d, numElemSets %d\n",i,numElemSets);
    }

    int myRowTest, *myColTest;

    blockRoster[i].setBlockID(elemBlockID);
    blockRoster[i].setNumNodesPerElement(numNodesPerElement);
    blockRoster[i].setNumElemDOF(numElemDOF);
    blockRoster[i].setNumElemSets(numElemSets);
    blockRoster[i].setInitNumElemSets(numElemSets);
    blockRoster[i].setInitNumElemTotal(numElemTotal);
    blockRoster[i].setNumElemTotal(numElemTotal);
    blockRoster[i].setInterleaveStrategy(interleaveStrategy);
    assert (interleaveStrategy == 0);   // only support NODE_CENTRIC for now...

    blockRoster[i].allocateNumNodalDOF(numNodesPerElement);
    int *numNodeDOFPtr = blockRoster[i].pointerToNumNodalDOF(myRowTest);
    assert (myRowTest == numNodesPerElement);
    
    blockRoster[i].allocateNumElemFields(numNodesPerElement);
    int *numElemFieldsPtr = blockRoster[i].pointerToNumElemFields(myRowTest);
    assert (myRowTest == numNodesPerElement);

//  construct the list of nodal solution cardinalities for this block

    int myNodalEqns = 0;
    int fieldIndex;
    int countDOF;
    for (j = 0; j < numNodesPerElement; j++) {
        countDOF = 0;
        for (k = 0; k < numElemFields[j]; k++) {
            fieldIndex = getFieldRosterIndex(elemFieldIDs[j][k]);
            assert (fieldIndex >= 0);
            countDOF += fieldRoster[fieldIndex].getNumFieldParams();
        }
        numElemFieldsPtr[j] = numElemFields[j];
        numNodeDOFPtr[j] = countDOF;
        myNodalEqns += countDOF;
    }


    blockRoster[i].setNumEqnsPerElement(myNodalEqns + numElemDOF);

//  cache a copy of the element fields array for later use...

    blockRoster[i].allocateElemFieldIDs(numNodesPerElement, numElemFieldsPtr);
    int **numElemFieldIDsPtr = 
                blockRoster[i].pointerToElemFieldIDs(myRowTest, myColTest);
    assert (myRowTest == numNodesPerElement);
    for (j = 0; j < numNodesPerElement; j++) {
        assert (myColTest[j] == numElemFields[j]);
    }
    for (j = 0; j < numNodesPerElement; j++) {
        for (k = 0; k < numElemFields[j]; k++) {
            numElemFieldIDsPtr[j][k] = elemFieldIDs[j][k];
        }
    }
    
//  create data structures for storage of element ID and topology info

    blockRoster[i].setNextElemIndex(0);
    blockRoster[i].allocateElemIDs(numElemTotal);
    blockRoster[i].allocateElemConn(numElemTotal, numNodesPerElement);
    if (numElemDOF > 0) {
       blockRoster[i].allocateElemSoln(numElemTotal, numElemDOF);
       blockRoster[i].allocateLocalEqnElemDOF(numElemTotal);
    }

    storeNumWorksets[i] = 0;

    checkNumElemBlocks++;
    currentElemBlockID = blockRoster[i].getBlockID();
    currentElemBlockIndex = i;
    assert(currentElemBlockID == elemBlockID);

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_SLE::initElemSet(int numElems, 
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
        fprintf(debugFile_,"trace: initElemSet, numElems: %d", numElems);
        fprintf(debugFile_,", currentBlockID: %d\n",currentElemBlockID);
        fflush(debugFile_);
    }

    int bIndex = search_ID_index(currentElemBlockID,&GID_blockIDList[0],
                               GID_blockIDList.size());
    assert(bIndex >= 0);
    
    int myRows, myCols;
    int myNumNodes = blockRoster[bIndex].getNumNodesPerElement();

//  find where we should start storing this element set in the flattened
//  lists of element data that reside in the BlockRecord.
//
//  the "start" variable should give us the offset to the next open index,
//  assuming none of my usual "off-by-one" errors on such fronts... *sigh*

    int start = blockRoster[bIndex].getNextElemIndex();
    GlobalID *myElemIDList = blockRoster[bIndex].pointerToElemIDs(myRows);
    GlobalID **myElemConn = blockRoster[bIndex].pointerToElemConn(myRows, myCols);
    assert (myCols == myNumNodes);
    
    int myElemIndex;
    for (int i = 0; i < numElems; i++) {
        myElemIndex = start + i;
        assert (myElemIndex < myRows);  // debug - insure we don't overrun list
        myElemIDList[myElemIndex] = elemIDs[i];
        for (int j = 0; j < myNumNodes; j++) {
            myElemConn[myElemIndex][j] = elemConn[i][j];
        }        
    }

//  now cache the current element index for use by the next element set

    blockRoster[bIndex].setNextElemIndex(myElemIndex+1);

    storeNumWorksets[bIndex]++;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::endInitElemBlock() {
//
//  tasks: check to insure consistency of data
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: endInitElemBlock\n");
        fflush(debugFile_);
    }

    int bIndex = search_ID_index(currentElemBlockID,&GID_blockIDList[0],
                               GID_blockIDList.size());
    assert (storeNumWorksets[bIndex] == blockRoster[bIndex].getNumElemSets());
  
 
    if (debugOutput_) {
        fprintf(debugFile_,"elemBlock %d, numWorksets %d\n",bIndex,
            storeNumWorksets[bIndex]);
    }
    
//  check for end of element data initialization to perform overhead tasks 

    if (checkNumElemBlocks == storeNumElemBlocks) {
        int errStat = doEndInitElemData();
        if (errStat) cerr << "ERROR in BASE_SLE::doEndInitElemData()." << endl;
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_SLE::beginInitNodeSets(int numSharedNodeSets, 
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
        fprintf(debugFile_,"beginInitNodeSets, numSharedNodeSets: %d",
                numSharedNodeSets);
        fprintf(debugFile_,", numExtNodeSets: %d",
                numExtNodeSets);
        fflush(debugFile_);
    }

//  perform overhead tasks on end of element data initialization first

    if (doneEndInitElemData == 0) {
        int errStat = doEndInitElemData();
        if (errStat){
            cout << "ERROR from doEndInitElemData" << endl;
            exit(0);
        }
    }

//  then store the node set parameters

 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, numSharedNodeSets: %d\n",localRank_,
            numSharedNodeSets);
    }
    storeSharedNodeSets = numSharedNodeSets;
    currentSharedNodeSet = 0;
 
 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, numExtNodeSets: %d\n",localRank_,
            numExtNodeSets);
    }
    storeExtNodeSets = numExtNodeSets;
    currentExtNodeSet = 0;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int BASE_SLE::initSharedNodeSet(const GlobalID *sharedNodeIDs,  
                                int lenSharedNodeIDs, 
                                const int *const *sharedProcIDs, 
                                const int *lenSharedProcIDs) {
//
//  In this function we simply accumulate the incoming data into internal arrays
//  in the shareNodes_ object.
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"initSharedNodeSet: lenSharedNodeIDs: %d\n",
                lenSharedNodeIDs);
        for(int i=0; i<lenSharedNodeIDs; i++){
            fprintf(debugFile_, "-- sharedNode[%d]: %d\n",i,
                    (int)sharedNodeIDs[i]);
        }
        fflush(debugFile_);
    }

    sharedNodes_->sharedNodes(sharedNodeIDs, lenSharedNodeIDs,
                              sharedProcIDs, lenSharedProcIDs,
                              localRank_);

    currentSharedNodeSet++;
    checkSharedNodeSets++;
 
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int BASE_SLE::initExtNodeSet(const GlobalID *extNodeIDs,
                             int lenExtNodeIDs, 
                             const int *const *extProcIDs,
                             const int *lenExtProcIDs) {
//
// store the input parameters in the externalNodes_ object...
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: initExtNodeSet\n");
        fprintf(debugFile_,"--- getting %d nodes.\n",lenExtNodeIDs);
        fflush(debugFile_);
    }

    externalNodes_->externNodes((GlobalID*)extNodeIDs, lenExtNodeIDs,
                                (int**)extProcIDs, (int*)lenExtProcIDs);

    currentExtNodeSet++;
    checkExtNodeSets++;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::endInitNodeSets() {
//
//  tasks: check to insure consistency of data (e.g., number of
//         passed lists equals number given in initSolveStep).
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: endInitNodeSets\n");
        fflush(debugFile_);
    }

    assert (checkBCNodeSets == storeBCNodeSets);
    assert (checkSharedNodeSets == storeSharedNodeSets);
    assert (checkExtNodeSets == storeExtNodeSets);
    
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::beginInitCREqns(int numCRMultRecords, 
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

    if (doneEndInitElemData == 0) {
        int errStat = doEndInitElemData();
        if (errStat){
            cout << "ERROR from doEndInitElemData" << endl;
            exit(0);
        }
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
int BASE_SLE::initCRMult(const GlobalID *const *CRMultNodeTable,
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

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: initCRMult\n");
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
    ceqn_MultConstraints[k].allocateCRIsLocalTable(numMultCRs, lenCRNodeList);
    
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
int BASE_SLE::initCRPen(const GlobalID *const *CRPenNodeTable, 
                        const int *CRFieldList,
                        int numPenCRs, 
                        int lenCRNodeList,
                        int& CRPenID) {

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: initCRPen\n");
        fflush(debugFile_);
    }

    int i, j, k;
    int myNumPenCRs, myLenCRNodeList;

    k = checkNumCRPenRecords;
    CRPenID = k;
    ceqn_PenConstraints[k].setCRPenID(checkNumCRPenRecords);
    ceqn_PenConstraints[k].setLenCRNodeList(lenCRNodeList);
    ceqn_PenConstraints[k].setNumPenCRs(numPenCRs);
    ceqn_PenConstraints[k].allocateCRFieldList(lenCRNodeList);
    ceqn_PenConstraints[k].allocateCRNodeTable(numPenCRs,lenCRNodeList);
    ceqn_PenConstraints[k].allocateCRIsLocalTable(numPenCRs, lenCRNodeList);
    
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
int BASE_SLE::endInitCREqns() {

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: endInitCREqns\n");
        fflush(debugFile_);
    }

    assert (checkNumCRMultRecords == storeNumCRMultRecords);
    assert (checkNumCRPenRecords == storeNumCRPenRecords);

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_SLE::initComplete() {
//
//  tasks: determine final sparsity pattern for use in allocating memory
//         for sparse matrix storage in preparation for assembling
//         element and constraint data.
//
//         allocate storage for upcoming assembly of element terms
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: initComplete\n");
        fflush(debugFile_);
    }

    int i, j, k, m;

//  first, call sharedNodes_->initComplete(), so that the shared node object
//  can go through and set the owning processor for each of the shared
//  nodes, make lists of local-shared and remote-shared nodes, etc.

    sharedNodes_->initComplete();

    //set an ownerProc variable on each node record, so we don't have to
    //call sharedNodes_->isShared a bunch of times during the assembly.
    //we're trading a little memory for a lot of time.

    int numActiveNodes = getNumActNodes();
    for(i=0; i<numActiveNodes; i++){
        GlobalID nodeID = gnod_LocalNodes[i].getGlobalNodeID();
        int proc = sharedNodes_->isShared(nodeID);
        if (proc >= 0) gnod_LocalNodes[i].ownerProc(proc);
        else gnod_LocalNodes[i].ownerProc(localRank_);
    }

//  now, determine the total number of active equations on this processor
//  need to count active nodal contributions, element equations, and all
//  the Lagrange multiplier constraint equations, too...
    
//  nodal solution data first...

    storeNumProcActEqns = getNumActNodalEqns();
    storeNumProcEqns = storeNumProcActEqns;

//  check the element blocks to find which have element DOF to include in
//  the storeNumProcEqns count - if any block has element DOF, then compute the
//  offset to each element's local DOF for purposes of allocating matrix memory
//
//  (exactly where to put these element DOF is an interesting issue... here, 
//  just toss them at the end of the nodal active eqns, which may or may not be
//  such a great choice.)

    for (i = 0; i < storeNumElemBlocks; i++) {
        int numElementDOF = blockRoster[i].getNumElemDOF();
        if (numElementDOF > 0) {
            int numElems;
            int *myElemDOFPtr = blockRoster[i].pointerToLocalEqnElemDOF(numElems);
            for (j = 0; j < numElems; j++) {
                myElemDOFPtr[j] = storeNumProcEqns;
                storeNumProcEqns += j*numElementDOF;
            }
        }
    }

//  add equations for any constraint relations here

    for (i = 0; i < storeNumCRMultRecords; i++) {
        ceqn_MultConstraints[i].setLocalEqnID(storeNumProcEqns);
        storeNumProcEqns += ceqn_MultConstraints[i].getNumMultCRs();
    }

//  ----- end active equation calculations -----

//  now we know how many equations there are on this processor so we can
//  allocate space for the row length arrays. This is going to be an
//  array of IntArrays, each one containing the column indices for
//  a row of the matrix. The number of IntArrays is equal to the number
//  of equations local to this processor.

    sysMatIndices = new IntArray[storeNumProcEqns];
 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, storeNumProcEqns: %d\n" ,localRank_,
            storeNumProcEqns);
    }

//  set global equation offsets for each processor. compute and store 
//  startRow and endRow terms.

//  first, get each processor's local number of equations on the master proc.

    int* globalNumProcEqns = new int[numProcs_];

    MPI_Gather(&storeNumProcEqns, 1, MPI_INT, globalNumProcEqns, 1, MPI_INT,
               masterRank_, FEI_COMM_WORLD);

//  compute offsets for all processors (starting index for local equations)

    int* globalStartRow = new int[numProcs_];
    globalStartRow[0] = 1;    // isis++ starts rows & cols at 1 (global)

    if (localRank_ == masterRank_) {
        for (i=1;i<numProcs_;i++) {
            globalStartRow[i] = globalStartRow[i-1] + globalNumProcEqns[i-1];
        }
    }

//  now, scatter vector of offsets to all processors from the master node.

    if (numProcs_ == 1){
        localStartRow_ = globalStartRow[0];
    }
    else {
        MPI_Scatter(globalStartRow, 1, MPI_INT, &localStartRow_, 1, MPI_INT,
                    masterRank_, FEI_COMM_WORLD);
    }

    delete [] globalStartRow;

//  now each processor can compute its localEndRow.

    localEndRow_ = localStartRow_ + storeNumProcEqns - 1;

//  compute global number of equations.

    int globalNumEqns = 0;

    if (localRank_ == masterRank_) {
        globalNumEqns = globalNumProcEqns[0];

        for (i=1; i<numProcs_; i++) {
            globalNumEqns += globalNumProcEqns[i];
        }
    }

    delete [] globalNumProcEqns;

    MPI_Bcast(&globalNumEqns, 1, MPI_INT, masterRank_, FEI_COMM_WORLD);

//  use block-row decomposition.

    int localStartCol = 1;
    int localEndCol = globalNumEqns;

// now we can allocate the pointers to the core pieces of the
// linear algebra library, create the 'Map', etc.

    createLinearAlgebraCore(globalNumEqns, localStartRow_, localEndRow_,
                            localStartCol, localEndCol);

//  do the shared node initialization communication stuff

    doSharedNodeInitComm();

//  we're going to need to receive and deal with information from other
//  processors (the sends which are executed below) which share nodes that
//  we own. We'll be needing the scatter indices for columns that are on
//  the other processors.

    sharedNodeLaunchInitRecvs();
 
 
    if (debugOutput_) {
        fprintf(debugFile_,"shSize_: %d, shCoeffSize_: %d\n",
          shSize_, shCoeffSize_);
        fflush(debugFile_);
    }

//  now, we'll do our local equation profile calculations (i.e., determine
//  how long each row is).
//  evaluate the element scatter arrays for use in determining the 
//  sparsity pattern
//  loop over all the elements, determining the elemental (both from nodes 
//  and from element DOF) contributions to the sparse matrix structure

    int ne, nn, proc, myLocalNode;

    NodeControlPacket shNodePacket;
    sharedBuffI_ = new SharedNodeBuffer(shSize_, shCoeffSize_);
 
    int bLimit = storeNumElemBlocks;
    for (int bIndex = 0; bIndex < bLimit; bIndex++) {
        int numBlockNodes = blockRoster[bIndex].getNumNodesPerElement();
        int numBlockElems = blockRoster[bIndex].getNumElemTotal();
        int numBlockEqns = blockRoster[bIndex].getNumEqnsPerElement();

        int *scatterIndices = new int[numBlockEqns];
        GlobalID *remShNodes = new GlobalID[numBlockNodes];
        int *remShNodeProc = new int[numBlockNodes];

//        GlobalID *elemIDList = blockRoster[bIndex].pointerToElemIDs(ne);
//        assert (ne == numBlockElems);
        GlobalID **connTable = blockRoster[bIndex].pointerToElemConn(ne, nn);
        assert (ne == numBlockElems);
        assert (nn == numBlockNodes);

        for (int elemIndex = 0; elemIndex < numBlockElems; elemIndex++) {

            int indexCounter = 0;
            int numRemShNodes = 0;

            for (int nodeIndex = 0; nodeIndex < numBlockNodes; nodeIndex++) {
                GlobalID myNode = connTable[elemIndex][nodeIndex];
                myLocalNode = GlobalToLocalNode(myNode);
                proc = gnod_LocalNodes[myLocalNode].ownerProc();

                if (proc == localRank_) {
                    int localEqnNumber = gnod_LocalNodes[myLocalNode].
                                                   getLocalEqnID();
                    int nDOF = gnod_LocalNodes[myLocalNode].getNumNodalDOF();
                    for(int ii=0; ii<nDOF; ii++){
                        scatterIndices[indexCounter++] = localEqnNumber + ii +
                                                              localStartRow_;
                    }
                }
                else {
                    //this node is shared but we don't own it, so the
                    // column index here will be the equation number we
                    // received from the node's owner

                    int eqnNumber = sharedNodes_->equationNumber(myNode);
                    for(int ii=0; ii<sharedNodes_->numEquations(myNode); ii++){
                        scatterIndices[indexCounter++] = eqnNumber + ii;
                    }

                    //record this remote shared node's ID, since we'll need
                    //to send its info to the owning processor.
                    remShNodes[numRemShNodes] = myNode;
                    remShNodeProc[numRemShNodes++] = proc;
                }

            }

            //now the elementDOF equations for this element...
            //Kim, check me on this logic.

//            int numElementDOF = blockRoster[bIndex].getNumElemDOF();
//            int numElems;
//            int *elemDOFEqns = blockRoster[bIndex].
//                     pointerToLocalEqnElemDOF(numElems);
//
//            int thisEqn = elemDOFEqns[elemIndex];
//
//            if (numElementDOF > 0) {
//
//                for (i = 0; i < numElementDOF; i++) {
//                    scatterIndices[indexCounter++] = thisEqn + i +
//                                                          localStartRow_;
//                }
//            }

            int mLimit = indexCounter;

            //now, for the nodes that are shared and remotely owned,
            //we need to pack up the indices to be shipped off to the
            //node's owner later.

            for (i=0; i<numRemShNodes; i++) {

                // for each remote shared node, we want to send
                // the number of nodal DOF and the length of the
                // scatterIndices array to the owning proc

                shNodePacket.nodeID = remShNodes[i];
                shNodePacket.numEqns = sharedNodes_->
                    numEquations(remShNodes[i]);
                shNodePacket.numIndices = mLimit;

                sharedBuffI_->addNodeControlPacket(shNodePacket,
                                                   remShNodeProc[i]);
                sharedBuffI_->addIndices(scatterIndices, mLimit,
                                                   remShNodeProc[i]);
            }

            for (int mrow = 0; mrow < mLimit; mrow++) {
                int sIndMrow = scatterIndices[mrow];
                if ((localStartRow_ <= sIndMrow)&& (sIndMrow <= localEndRow_)) {
                    int irow = sIndMrow - localStartRow_;
                    for (int mcol = 0; mcol < mLimit; mcol++) {
                        int jcol = scatterIndices[mcol];

                        int found = -1, insertPoint = 0;
                        IntArray& rowLenIrow = sysMatIndices[irow];
                        if (rowLenIrow.size()>0) {
                            found = find_index(jcol,
                                               &(rowLenIrow[0]),
                                               rowLenIrow.size(),
                                               &insertPoint);
                        }
                        if (found<0) {
                            rowLenIrow.insert(insertPoint, jcol);
                        }
                    }
                }
            }
        }
        delete [] scatterIndices;
        delete [] remShNodes;
        delete [] remShNodeProc;
    }


 
    if (debugOutput_) {
        fprintf(debugFile_,"arriving at barrier\n");
        fflush(debugFile_);
    }

    MPI_Barrier(FEI_COMM_WORLD);

    //now lets get the shared node stuff back out of sharedBuffI_ and send it
    //to the other processors.

    int numDestProcs;
    int *destProcs = sharedBuffI_->packetDestProcsPtr(numDestProcs);

 
    if (debugOutput_) {
        fprintf(debugFile_,"in initComplete, numDestProcs: %d\n",
         numDestProcs);
        fprintf(debugFile_,"destProcs: ");
    for(i=0; i<numDestProcs; i++){
            fprintf(debugFile_,"%d ", destProcs[i]);
    }
        fprintf(debugFile_,"\n");
        fflush(debugFile_);
    }

    for(i=0; i<numDestProcs; i++){
 
    if (debugOutput_) {
        fprintf(debugFile_,"sending %d packets to proc %d\n",
           sharedBuffI_->numPacketUnits(destProcs[i]), destProcs[i]);
        fflush(debugFile_);
    }
        MPI_Send(sharedBuffI_->packetPtr(destProcs[i]),
                  sharedBuffI_->numPacketUnits(destProcs[i]),
                  MPI_NodePacket, destProcs[i], packet_tag1+localRank_,
                  FEI_COMM_WORLD);
 
    if (debugOutput_) {
        fprintf(debugFile_,"indicesUnitLength: %d, numIndexUnits: %d\n",
        sharedBuffI_->indicesUnitLength(), sharedBuffI_->numIndexUnits(destProcs[i]))
;
    int *inds = sharedBuffI_->indicesPtr(destProcs[i]);
    for(j=0; j<(sharedBuffI_->numIndexUnits(destProcs[i]) *
                sharedBuffI_->indicesUnitLength()); j++){
      if (inds[j]<0)
            fprintf(debugFile_,"inds[%d]: %d, destProcs[i]: %d\n", j, inds[j],
           destProcs[i]);
    }
        fprintf(debugFile_,"\n");
        fflush(debugFile_);
    }
        MPI_Send(sharedBuffI_->indicesPtr(destProcs[i]),
                  sharedBuffI_->numIndexUnits(destProcs[i]) *
                  sharedBuffI_->indicesUnitLength(),
                  MPI_INT, destProcs[i], indices_tag+localRank_,
                  FEI_COMM_WORLD);
    }

    //destroy the SharedNodeBuffer
    delete sharedBuffI_;
    sharedBuffI_ = NULL;

    // now, lets receive and deal with the sends (above) which were executed
    // on other processors. We'll be needing the scatter indices for columns
    // that are on the other processors.

    int n, index;
    MPI_Status status;

    for (i=0; i<numShProcs_; i++) {
        MPI_Waitany(numShProcs_, shRequests_, &index, &status);

        MPI_Wait(&shScatRequests_[index], &status);

 
        if (debugOutput_) {
            fprintf(debugFile_,
                "shProc_[%d]: %d, shSize_: %d, shNodesFromProc_[%d]: %d\n",
                index, shProc_[index], shSize_, index, shNodesFromProc_[index]);
        }

        // now we have the scatter indices, so let's put them into our matrix
        // structure
        for(n=0; n<shNodesFromProc_[index]; n++){
            myLocalNode = GlobalToLocalNode(shNodeInfo_[index][n].nodeID);
            int localRow = gnod_LocalNodes[myLocalNode].getLocalEqnID();
            for (j=0; j<shNodeInfo_[index][n].numEqns; j++) {
                int irow = localRow + j;
                for (int mcol = 0; mcol < shNodeInfo_[index][n].numIndices;
                                                                 mcol++) {
                    int jcol = shScatterIndices_[index][n*shSize_ + mcol];
 
                    int found = -1, insertPoint = 0;
                    if (sysMatIndices[irow].size()>0)
                        found = find_index(jcol,&((sysMatIndices[irow])[0]),
                                           sysMatIndices[irow].size(),
                                           &insertPoint);
                    if (found<0) {
                        sysMatIndices[irow].insert(insertPoint, jcol);
                    }
                }
            }
        }
        delete [] shNodeInfo_[index];
        delete [] shScatterIndices_[index];
    }

    delete [] shNodeInfo_;
    shNodeInfo_ = NULL;
    delete [] shRequests_;
    shRequests_ = NULL;
    delete [] shScatRequests_;
    shScatRequests_ = NULL;

    delete [] shScatterIndices_;
    shScatterIndices_ = NULL;

 
    if (debugOutput_) {
        fprintf(debugFile_,"done with the initComplete shared node stuff...\n");
        fflush(debugFile_);
    }
    MPI_Barrier(FEI_COMM_WORLD);

    int thisFlag = 0;
    MPI_Status thisStatus;
    thisStatus.MPI_SOURCE = -1;

    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, FEI_COMM_WORLD,
               &thisFlag, &thisStatus);

 
    if (debugOutput_) {
        fprintf(debugFile_,"Iprobe thisFlag: %d, moving on\n",thisFlag);
        fflush(debugFile_);
    }
 
    // next, handle the matrix structure imposed by the constraint relations...
    // this version is a touch kludgy (namely, one entry at a time).
    // using a collection of column indices (i.e., scatterIndices) is a good
    // idea for the row-like portions of the matrix, namely the rows
    // corresponding to the constraint equations, but it's not so simple for
    // the column-like portions of the matrix, since we're using a row-oriented
    // data structure.
    //
    // FURTHERMORE, since at initialization all we are being passed is the
    // general form of the constraint equations, we can't check to see if any
    // of the weight terms are zeros at this stage of the game.  Hence, we
    // have to reserve space for all the nodal weight vectors, even though
    // they might turn out to be zeros during the load step.... 
    
    // finally, note that we're using CRMultID here as a zero-based offset for
    // the equation set, and that we'll ultimately need to store this offset and
    // look it up using a CRMultID-to-local_constraint_storage scheme...
    
    // let's start by performing some initializations to help in subsequent
    // calcs. the doCRInit function loads up the constraint records and the
    // external node record with the information that we'll be needing. i.e.,
    // any necessary communications, such as exchanging equation numbers etc.,
    // is done by doCRInit.

    int errorCode = doCRInit();
    if (errorCode) cerr << "ERROR in BASE_SLE::doCRInit()." << endl;

 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, back from doCRInit\n",localRank_);
        fflush(debugFile_);
    }

    MPI_Barrier(FEI_COMM_WORLD);

    // now it's time to allocate storage in the sparse matrix for the constraint
    // equations, but this task is complicated by the fact that the constraints 
    // will commonly involve nodes that aren't locally available. Thus we'll
    // make use of the external node records constructed in doCRInit().

    // the first step in the allocation task is to process the rows
    // corresponding to the constraints, since this step involves local
    // equations.
        
    int icOffset;
    int ntRow, ntCol;

    // now process the rows created for the individual constraint records -
    // during this process, we can store some node-weight data to simplify
    // handling column allocation off-processor for external nodes

    for (k = 0; k < storeNumCRMultRecords; k++) {
        int numMultCRs = ceqn_MultConstraints[k].getNumMultCRs();
        icOffset = ceqn_MultConstraints[k].getLocalEqnID();
        int lenList = ceqn_MultConstraints[k].getLenCRNodeList();
        GlobalID **CRNodeTablePtr = 
            ceqn_MultConstraints[k].pointerToCRNodeTable(ntRow, ntCol);
        bool** CRIsLocalTablePtr = 
            ceqn_MultConstraints[k].pointerToCRIsLocalTable(ntRow, ntCol);
        for (i = 0; i < numMultCRs; i++) {
            for (j = 0; j < lenList; j++) {
                int myLocalEqn;
                int numSolnParams;
          
                if (CRIsLocalTablePtr[i][j] == true) {
                  // first, consider the case where the node to be
                  // processed is local. Since it's local, all needed info
                  // can be obtained from the gnod_LocalNodes data structure.

                  myLocalNode = GlobalToLocalNode(CRNodeTablePtr[i][j]);
                  myLocalEqn = gnod_LocalNodes[myLocalNode].getLocalEqnID();
                  numSolnParams = gnod_LocalNodes[myLocalNode].getNumNodalDOF();
 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d processing MultCR, node %d, Eqn %d, DOF %d\n",
            localRank_,(int)CRNodeTablePtr[i][j],myLocalEqn,numSolnParams);
    }
                  int ir = icOffset + i;
                  for (m = 0; m < numSolnParams; m++) {
                      int jc = myLocalEqn + m;

 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, MultCR producing matrix position %d %d\n",
             localRank_,ir+localStartRow_,jc+localStartRow_);
    }
                      int found = -1, insertPoint = 0;
                      if (sysMatIndices[ir].size()>0) 
                          found = find_index(jc+localStartRow_,
                                             &((sysMatIndices[ir])[0]),
                                             sysMatIndices[ir].size(),
                                             &insertPoint);
                      if (found<0) {
                          sysMatIndices[ir].insert(insertPoint,
                                                   jc+localStartRow_);
                      }
                      found = -1;
                      insertPoint = 0;
                      if (sysMatIndices[jc].size()>0) 
                          found = find_index(ir+localStartRow_,
                                             &((sysMatIndices[jc])[0]),
                                             sysMatIndices[jc].size(),
                                             &insertPoint);
                      if (found<0) {
                          sysMatIndices[jc].insert(insertPoint,
                                                   ir+localStartRow_);
                      }
                  }
                }
                else {
                    // ...else it ain't local. This means we have to get some
                    // information from the external node records.
            
                    int myExtNodeID = CRNodeTablePtr[i][j];
                    int myEqnNumber = externalNodes_->globalEqn(myExtNodeID);
                    numSolnParams = externalNodes_->numSolnParams(myExtNodeID);

 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d doing MultCR, extnode %d, Eqn %d, DOF %d\n",
            localRank_,(int)CRNodeTablePtr[i][j],myEqnNumber,numSolnParams);
    }
                    // we will only handle the local-row-oriented storage at
                    // this stage so do this work and postpone the
                    // off-processor column stuff until later...
                        
                    int ir = icOffset + i;
                    for (m = 0; m < numSolnParams; m++) {
                        int jc = myEqnNumber + m;

 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, MultCR producing matrix position %d %d\n",
             localRank_,ir+localStartRow_,jc+localStartRow_);
    }
                        int found = -1, insertPoint = 0;
                        if (sysMatIndices[ir].size()>0) 
                            found = find_index(jc, &((sysMatIndices[ir])[0]),
                                               sysMatIndices[ir].size(),
                                               &insertPoint);
                        if (found<0) {
                            sysMatIndices[ir].insert(insertPoint, jc);
                        }
                    }
                }
            }
        }
    }
    
    // we also need to accomodate penalty constraints, so let's process them
    // now... note that penalty constraints don't generate new equations
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

    int p;
    for (k = 0; k < storeNumCRPenRecords; k++) {
        int numPenCRs = ceqn_PenConstraints[k].getNumPenCRs();
        int lenList = ceqn_PenConstraints[k].getLenCRNodeList();
        GlobalID **CRNodeTablePtr = 
           ceqn_PenConstraints[k].pointerToCRNodeTable(ntRow, ntCol);
        assert (ntRow == numPenCRs);
        assert (ntCol == lenList);
        bool** CRIsLocalTablePtr = 
            ceqn_PenConstraints[k].pointerToCRIsLocalTable(ntRow, ntCol);

        // each constraint equation generates a set of nodal energy terms, so
        // we have to process a matrix of nodes for each constraint

        for (p = 0; p < numPenCRs; p++) {
            for (i = 0; i < lenList; i++) {
                int ipNode = CRNodeTablePtr[p][i];

                // if i is local, then we can handle the allocation locally

                if (CRIsLocalTablePtr[p][i] == true) {
                    int iLocalNode = GlobalToLocalNode(ipNode);
                    int iLocalEqn = gnod_LocalNodes[iLocalNode].getLocalEqnID();
                    int iRows = gnod_LocalNodes[iLocalNode].getNumNodalDOF();
                    for (j = 0; j < lenList; j++) {
                        int jpNode = CRNodeTablePtr[p][j];
                       if (CRIsLocalTablePtr[p][j] == true) {
                            int jLocalNode = GlobalToLocalNode(jpNode);
                           int jLocalEqn = gnod_LocalNodes[jLocalNode].
                                                                getLocalEqnID();
                           int jCols = gnod_LocalNodes[jLocalNode].
                                                                getNumNodalDOF();

                           for (int irIndex = 0; irIndex < iRows; irIndex++) {
                                int ir = iLocalEqn + irIndex;
                               for (int jcIndex=0; jcIndex<jCols; jcIndex++) {
                                   int jc = jLocalEqn + jcIndex;

                                   int found = -1, insertPoint = 0;
                                   if (sysMatIndices[ir].size()>0) 
                                       found = find_index(jc+localStartRow_,
                                                   &((sysMatIndices[ir])[0]),
                                                   sysMatIndices[ir].size(),
                                                   &insertPoint);
                                   if (found<0) {
                                           sysMatIndices[ir].insert(insertPoint,
                                                            jc+localStartRow_);
                                   }
                               }
                           }
                       }
                       else {   
                           //j not local, so look it up in external node data...
                       
                           int jRemoteEqn = externalNodes_->globalEqn(jpNode);
                           int jCols = externalNodes_->numSolnParams(jpNode);

                           for (int irIndex = 0; irIndex < iRows; irIndex++) {
                               int ir = iLocalEqn + irIndex;
                               for (int jcIndex=0; jcIndex<jCols; jcIndex++) {
                                   int jc = jRemoteEqn + jcIndex;

                                      int found = -1, insertPoint = 0;
                                   if (sysMatIndices[ir].size()>0) 
                                       found = find_index(jc,
                                                   &((sysMatIndices[ir])[0]),
                                                   sysMatIndices[ir].size(),
                                                   &insertPoint);
                                   if (found<0) {
                                           sysMatIndices[ir].insert(insertPoint,
                                                                    jc);
                                   }
                               }
                           }
                        }
                    }   //   end j loop
                }   //   end i local test

                // for i not local, we can't assemble the penalty terms on this
                // processor, so we'll defer these remote cases until later
                // on... (see below)

            }   //   end i loop
        }   //   end p loop
    }   //   end k loop
    
    // finally, allocate space for the terms that arise from external nodes
    // which reside locally but appear in remote penalty constraints.

    int numRemotePenCRs, jj;
    int* lenRemotePenNodes;
    numRemotePenCRs = externalNodes_->remoteNumPenCRs();
    GlobalID** remotePenNodes = externalNodes_->
                           remotePenNodeTable(&lenRemotePenNodes);
    int** remotePenEqnIDs = externalNodes_->
                           remotePenEqnIDTable(&lenRemotePenNodes);
    int** remotePenNumDOFs = externalNodes_->
                           remotePenNumDOFTable(&lenRemotePenNodes);

    for (i = 0; i < numRemotePenCRs; i++) {
        for (j = 0; j < lenRemotePenNodes[i]; j++) {
            GlobalID myNodeID = remotePenNodes[i][j];
            proc = externalNodes_->ownerProc(myNodeID);

            if (proc == localRank_) {
                myLocalNode = GlobalToLocalNode(myNodeID);
                assert (myLocalNode >= 0);
                int myBaseEqnID = remotePenEqnIDs[i][j];
                int numSolnParams = remotePenNumDOFs[i][j];

                //jj will loop along (across) a row in the remotePenNodes table
                for(jj=0; jj<lenRemotePenNodes[i]; jj++){

                    //k will loop over the DOF at node myNodeID
                    for(k=0; k<numSolnParams; k++){
                        int jDOF = remotePenNumDOFs[i][jj];
                        int jEqnNum = remotePenEqnIDs[i][jj];

                        //jjj loops over the DOF at node remotePenNodes[i][jj]
                        for(int jjj=0; jjj<jDOF; jjj++){
                            int myRow = myBaseEqnID + k - localStartRow_;
                            int myCol = jEqnNum + jjj;

                            //see if this position exists in the matrix,
                            //and create it if it doesn't.
                            int found = -1, insertPoint = 0;
                            if (sysMatIndices[myRow].size()>0)  {
                                found = find_index(myCol,
                                            &((sysMatIndices[myRow])[0]),
                                            sysMatIndices[myRow].size(),
                                            &insertPoint);
                            }
                            if (found<0) {
                                sysMatIndices[myRow].insert(insertPoint, myCol);
                            }

                        }//end of jjj loop
                    }//end of k loop
                }//end of jj loop
            }//end of 'if proc == localRank_'
        }//end of j loop
    }//end of i loop
 
    int numSendProcs;
    int* sendProcs = externalNodes_->sendProcListPtr(numSendProcs);
    int* lenLocalNodeIDs;
    GlobalID** localNodeIDsPtr = externalNodes_->
                                         localNodeIDsPtr(&lenLocalNodeIDs);

    for (i = 0; i < numSendProcs; i++) {

        //  Lagrange multiplier constraints

        for (j = 0; j < lenLocalNodeIDs[i]; j++) {
            GlobalID myNodeID = localNodeIDsPtr[i][j];
            myLocalNode = GlobalToLocalNode(myNodeID);
            assert (myLocalNode >= 0);
            int numSolnParams = gnod_LocalNodes[myLocalNode].getNumNodalDOF();
            int myBaseEqnID = gnod_LocalNodes[myLocalNode].getLocalEqnID();
 
            // reminder: numMultCRs is the number of Lagrange constraints that
            // this node appears in on the other processor. i.e., the number of
            // columns in which we need to make modifications. 

            int found, myNumMultCRs, insertPoint = 0;
            int *myColumnIDs = externalNodes_->globalCREqn(myNodeID,
                                                           myNumMultCRs);
            // handle the list of Lagrange multiplier constraints associated
            // with this node.
            for (jj=0; jj < myNumMultCRs; jj++) {
                for (k = 0; k < numSolnParams; k++) {
                    int myRow = myBaseEqnID + k;

                    found = -1;
                    insertPoint = 0;
                    if (sysMatIndices[myRow].size()>0)  {
                        found = find_index(myColumnIDs[jj],
                                           &((sysMatIndices[myRow])[0]),
                                           sysMatIndices[myRow].size(),
                                           &insertPoint);
                    }
                    if (found<0) {
                        sysMatIndices[myRow].insert(insertPoint, myColumnIDs[jj]);
                    }
                }
            }
        }
    }

    //
    // all profile calculations are done, i.e., we now know the structure
    // of the sparse matrix, so we can now configure (allocate) it.
    //

    matrixConfigure(sysMatIndices);

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

 
//------------------------------------------------------------------------------
int BASE_SLE::resetSystem(double s) {
//
//  This function may only be called after the initialization phase is
//  complete. It requires that the system matrix and rhs vector have already
//  been created.
//  It then puts the value s throughout both the matrix and the vector.
//

    baseTime_ = MPI_Wtime();
 
    if (debugOutput_) {
        fprintf(debugFile_,"resetSystem\n");
        fflush(debugFile_);
    }

    resetMatrixAndVector(s);

    //now clear away the  nodal boundary condition data.
    for(int i=0; i<storeNumProcActNodes; i++){
        gnod_LocalNodes[i].removeBCData();
    }
 
    if (debugOutput_ && (solveCounter_>1)){
        //if debug output is 'on' and we've already completed at least one
        //solve, reset the debug output file.

        setDebugOutput(debugPath_, debugFileName_);
    }
 
    if (debugOutput_) {
        fprintf(debugFile_,"leaving resetSystem\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
int BASE_SLE::beginLoadNodeSets(int numBCNodeSets) {
//
//  tasks: start the loading of nodal loading information
//
    if (debugOutput_) {
        fprintf(debugFile_,"trace: beginLoadNodeSets\n");
        fflush(debugFile_);
    }

    (void)numBCNodeSets; // this prevents "unused variable" warnings

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_SLE::loadBCSet(const GlobalID *BCNodeSet,  
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
        fprintf(debugFile_,"trace: loadBCSet\n");
        fflush(debugFile_);
    }

    int i, iLocal;
    GlobalID iGlobal;

//  for now, put the bc data arrays in the nodal parameter storage
//  ultimately, it may be better just to store the raw lists and
//  to provide lookups to make it easier to handle implementing these
//  conditions (for example, during element assembly)

    for (i = 0; i < lenBCNodeSet; i++) {
        iGlobal = BCNodeSet[i];
        iLocal = GlobalToLocalNode(iGlobal);
        gnod_LocalNodes[iLocal].setBCDataFlag();

        int BCFieldSize = getFieldCardinality(BCFieldID);
        gnod_LocalNodes[iLocal].addBCRecord(BCFieldID,
                                            BCFieldSize, 
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
int BASE_SLE::endLoadNodeSets() {
//
//  tasks: complete the loading of nodal information
//

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: endLoadNodeSets\n");
        fflush(debugFile_);
    }

//  here's a natural place to check for any conflicting BCs and take any
//  corrective actions that may be possible

//  first, let's convert the linked-list representation of the accumulated
//  BC data for each node into an array-based storage, as we're gonna need
//  something more efficient while we check BC data during element assembly
//
//  in particular, to avoid having to do a lot of field offset calculations
//  during the element assembly process, we'll create a list of field offsets
//  that will reside in each BC record, so it's easy to locate these BC's 
//  within the element stiffness and load arrays during element assembly 

    int i, j, k, myNumBCs, newNumBCs;
    int myFieldID, myFieldSize, myFieldOffset, myTest;
    double *alpha, *beta, *gamma, *newAlpha, *newBeta, *newGamma;
    BCRecord *myBCRecordPtr;

    for (i = 0; i < storeNumProcActNodes; i++){
        BCRecord** BCList = gnod_LocalNodes[i].pointerToBCTempList(myNumBCs);
        if (myNumBCs > 0) {   // there's BC data at this node, so...

            //we're only going to end up with 1 BCRecord for each fieldID.
            //so first, lets construct a list of the fieldIDs for which a
            //boundary condition has been specified, and make sure that each
            //of those fieldIDs only appears in the list once.

            IntArray fieldList;
            int insertPoint;

            //form a list of the fieldIDs
            for(j=0; j<myNumBCs; j++){
                int found = -1;
                insertPoint = 0;
                myFieldID = BCList[j]->getFieldID();

                if (fieldList.size() > 0){
                    found = find_index(myFieldID,
                                       &fieldList[0],
                                       fieldList.size(),
                                       &insertPoint);
                }
                if (found < 0){
                    fieldList.insert(insertPoint, myFieldID);
                }
            }

            //ok, the length of the list we just created is the number of 
            //boundary condition records that we'll have at this node.
            newNumBCs = fieldList.size();

            gnod_LocalNodes[i].allocateBCRecords(newNumBCs);
            myBCRecordPtr = gnod_LocalNodes[i].pointerToBCRecords(myTest);

            //now, lets set the fieldID's and fieldSizes for these BCRecords,
            //and allocate the alpha's, beta's and gamma's.
            for(j=0; j<newNumBCs; j++){
                myBCRecordPtr[j].setFieldID(fieldList[j]);
                myFieldOffset = gnod_LocalNodes[i].getFieldOffset(fieldList[j]);
                assert (myFieldOffset >= 0);  // insure this field exists here
                myBCRecordPtr[j].setFieldOffset(myFieldOffset);
                myFieldSize = getFieldCardinality(fieldList[j]);
                myBCRecordPtr[j].setFieldSize(myFieldSize);

                myBCRecordPtr[j].allocateAlpha();
                myBCRecordPtr[j].allocateBeta();
                myBCRecordPtr[j].allocateGamma();
            }

            //ok. now lets loop through the old (not necessarily unique) BC
            //list, and for each one, we will either accumulate its components
            //into one of the new BCRecords if it's a natural or mixed BC, or
            //simply 'put' it (overwriting any previous) if it's essential.
 
            for (j = 0; j < myNumBCs; j++) {
                myFieldID = BCList[j]->getFieldID();
                myFieldSize = BCList[j]->getFieldSize();

                alpha = BCList[j]->pointerToAlpha(myFieldSize);
                beta = BCList[j]->pointerToBeta(myFieldSize);
                gamma = BCList[j]->pointerToGamma(myFieldSize);

                //we know that this next call to find_index
                //will find an index, because we're re-running the list we
                //used above to construct fieldList.
                int index = find_index(myFieldID, &fieldList[0],
                                       fieldList.size(), &insertPoint);

                newAlpha = myBCRecordPtr[index].pointerToAlpha(myTest);
                assert (myTest == myFieldSize);  // more paranoia 
                newBeta = myBCRecordPtr[index].pointerToBeta(myTest);
                assert (myTest == myFieldSize);  // more paranoia 
                newGamma = myBCRecordPtr[index].pointerToGamma(myTest);
                assert (myTest == myFieldSize);  // more paranoia 

//  here's the proper place to test for essential BC's (for example) if we
//  want to set flags for processing these more efficiently within the
//  assembly loops in loadElemData()

                for (k = 0; k < myFieldSize; k++) {
                    if ((alpha[k] != 0.0) && (beta[k] == 0.0)) {
                        //it's essential, so 'put' it in
                        newAlpha[k] = alpha[k];
                        newBeta[k] = beta[k];
                        newGamma[k] = gamma[k];
                    }
                    else {
                        //it's natural or mixed, so sum it in, but not if
                        //an essential BC has already been put in.
                        if (!((newAlpha[k] != 0.0) && (newBeta[k] == 0.0))){
                            newAlpha[k] += alpha[k];
                            newBeta[k] += beta[k];
                            newGamma[k] += gamma[k];
                        }
                    }
                }
            }
            
//  all done with the original BC list, so we can get rid of it here...

            gnod_LocalNodes[i].destroyTempBCList();
        }
    }

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::beginLoadElemBlock(GlobalID elemBlockID,
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

 
    if (debugOutput_) {
        fprintf(debugFile_,"beginLoadElemBlock\n");
        fflush(debugFile_);

    }

    if (!shBuffLoadAllocated_) {
        shBuffLoadD_ = new CommBufferDouble();
        shBuffLoadI_ = new CommBufferInt();
        shBuffLoadAllocated_ = true;
    }

    if (!shBuffRHSAllocated_) {
        if (debugOutput_) {
            fprintf(debugFile_,"  allocating shBuffRHSLoadD_, numRHSs_: %d\n",
                   numRHSs_);
        }

        shBuffRHSLoadD_ = new CommBufferDouble[numRHSs_];
        shBuffRHSAllocated_ = true;
    }

    currentElemBlockID = elemBlockID;
    currentWorkSetID = 0;

    int i = search_ID_index(elemBlockID, &GID_blockIDList[0],
                            GID_blockIDList.size());
    assert(i >= 0);
    currentElemBlockIndex = i;

//  reset starting index for element list

    blockRoster[i].setNextElemIndex(0);

//  cache the data passed for the load step

    blockRoster[i].setNumElemSets(numElemSets);
    blockRoster[i].setLoadNumElemSets(numElemSets);
    blockRoster[i].setLoadNumElemTotal(numElemTotal);
    blockRoster[i].setNumElemTotal(numElemTotal);

//  insure that the number of elements present in the init step is the same
//  as that we're about to get in the load step

    assert (numElemTotal == blockRoster[i].getInitNumElemTotal());

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
int BASE_SLE::loadElemSet(int elemSetID, 
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

    //this prevents "declared and never referenced" warnings.
    (void)elemIDs;

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSet, numElems: %d\n", numElems);
        fprintf(debugFile_,"currentElemBlockID: %d\n",(int)currentElemBlockID);
        fflush(debugFile_);
    }

    currentWorkSetID = elemSetID;
 
    if (numElems <= 0) return(0); //zero-length workset, do nothing.

    int i, k;

//  we're gonna need some control parameters that didn't come in through this
//  function's argument list.

    int numNodesPerElem = 0;//  number of nodes per element in this block
    int numElemRows = 0;    //  number of rows per element array in this block

    currentElemBlockIndex
               = search_ID_index(currentElemBlockID,&GID_blockIDList[0],
                              GID_blockIDList.size());

    assert(currentElemBlockIndex >= 0);

    numElemRows = blockRoster[currentElemBlockIndex].getNumEqnsPerElement();

    if (debugOutput_) {
        fprintf(debugFile_,"numElemRows: %d\n", numElemRows);
        fflush(debugFile_);
    }

    int* numNodalDOFPtr = blockRoster[currentElemBlockIndex].
                             pointerToNumNodalDOF(numNodesPerElem);

    //  dense local index of each nodal eqn
    IntArray nodeEqnIndices(numNodesPerElem);

    nodeEqnIndices[0] = 0;
    for (i = 1; i < numNodesPerElem; i++) {
        nodeEqnIndices[i] = nodeEqnIndices[i-1] + numNodalDOFPtr[i-1];
    }

    IntArray localNodeIndices(numNodesPerElem);

    // scatter indices into the system matrix
    IntArray scatterIndices(numElemRows);

    // local copy of (single) element stiffness and load
    double **ek = NULL, *ef = NULL;

    ef = new double[numElemRows];
    ek = new double* [numElemRows];
    for (i = 0; i < numElemRows; i++) {
        ek[i] = new double[numElemRows];
    }

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSet starting numElems %d loop...\n",
            numElems);
        fflush(debugFile_);
    }

    //now we'll loop through the elements in this workset, assembling
    //the stiffnesses and loads into the global sparse system stiffness
    //matrix.

    for (k = 0; k < numElems; k++) {

        calculateLocalNodeIndices(elemConn[k], numNodesPerElem,
                                  &localNodeIndices[0]);


        //now we'll make a local dense copy of the element stiffness array
        //if the stiffness array was passed in using one of the "weird"
        //element formats.
        if (elemFormat != 0) {
            copyStiffness(elemStiffness[k], numElemRows, elemFormat, ek);

            //also, take a copy of the load vector.
            for(i=0; i<numElemRows; i++){
                ef[i] = elemLoad[k][i];
            }
        }

//  our element-assembly logic assumes implicitly that all nodal solution 
//  parameters are grouped contiguously, to simplify matrix allocation and other
//  low-level tasks.  If this isn't the case (e.g., for the FIELD_CENTRIC data
//  scheme, the only alternative interleave strategy supported), then 
//  we need to reorder the rows and columns of the element stiffness and load
//  in order to facilitate the nodal-equation-oriented assembly ops below...

        if (blockRoster[currentElemBlockIndex].getInterleaveStrategy() != 0) {
            continue;  // remapping of interleaving not yet supported...
                       // but when it is, the remapping function goes here!
        }

        //  now let's obtain the scatter indices for assembling the equations
        //  into their appropriate places in the system stiffness and load
        //  structures...

        int numIndices = formElemScatterList(currentElemBlockIndex, elemConn[k],
                                             &localNodeIndices[0],
                                             &scatterIndices[0]);
//  these scatter indices are coming out of formElemScatterList as global
//  equation numbers. i.e., they have already had the appropriate offset added
//  to them, for the parallel case.

        if (elemFormat != 0) {
            packSharedStiffnessAndLoad(elemConn[k], numNodesPerElem,
                                       &localNodeIndices[0],
                                       &scatterIndices[0], numIndices,
                                       ek, ef);
        }
        else {
            packSharedStiffnessAndLoad(elemConn[k], numNodesPerElem,
                                       &localNodeIndices[0],
                                       &scatterIndices[0], numIndices,
                                       elemStiffness[k], elemLoad[k]);
        }

        // assembly operation

        if (elemFormat != 0) {
            assembleStiffnessAndLoad(numElemRows, &scatterIndices[0],
                                     ek, ef);
        }
        else {
            assembleStiffnessAndLoad(numElemRows, &scatterIndices[0],
                                     elemStiffness[k], elemLoad[k]);
        }

    }//end of the 'for k' loop over numElems
 
//  update the current workset ID counter to keep our place in the data 
//  stream

    currentWorkSetID++;

//  all done

    delete [] ef;
    for (i = 0; i < numElemRows; i++) {
        delete [] ek[i];
    }
    delete [] ek;

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadElemSet\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::loadElemSetMatrix(int elemSetID, 
                          int numElems, 
                          const GlobalID *elemIDs,  
                          const GlobalID *const *elemConn,
                          const double *const *const *elemStiffness,
                          int elemFormat) {
//
//  task: assemble the element stiffness matrices for a given workset
//

    baseTime_ = MPI_Wtime();

    //this prevents "declared and never referenced" warnings.
    (void)elemIDs;

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSetMatrix, numElems: %d\n", numElems);
        fprintf(debugFile_,"currentElemBlockID: %d\n",(int)currentElemBlockID);
        fflush(debugFile_);
    }

    currentWorkSetID = elemSetID;

    if (numElems <= 0) return(0); //zero-length workset, do nothing.

    int i, k;

//  we're gonna need some control parameters that didn't come in through this
//  function's argument list.

    int numNodesPerElem = 0;//  number of nodes per element in this block
    int numElemRows = 0;    //  number of rows per element array in this block

    currentElemBlockIndex
               = search_ID_index(currentElemBlockID,&GID_blockIDList[0],
                              GID_blockIDList.size());

    assert(currentElemBlockIndex >= 0);

    numElemRows = blockRoster[currentElemBlockIndex].getNumEqnsPerElement();

    if (debugOutput_) {
        fprintf(debugFile_,"numElemRows: %d\n", numElemRows);
        fflush(debugFile_);
    }

    int* numNodalDOFPtr = blockRoster[currentElemBlockIndex].
                             pointerToNumNodalDOF(numNodesPerElem);

    //  dense local index of each nodal eqn
    IntArray nodeEqnIndices(numNodesPerElem);

    nodeEqnIndices[0] = 0;
    for (i = 1; i < numNodesPerElem; i++) {
        nodeEqnIndices[i] = nodeEqnIndices[i-1] + numNodalDOFPtr[i-1];
    }

    IntArray localNodeIndices(numNodesPerElem);

    // scatter indices into the system matrix
    IntArray scatterIndices(numElemRows);

    // local copy of (single) element stiffness
    //(we're also creating a dummy load vector (ef) to pass to the
    //'packSharedStiffnessAndLoad' function. This is just a lazy-ass
    //band-aid solution until I write some more general Shared-info-passing
    //code.
    double **ek = NULL, *ef;

    ef = new double[numElemRows];
    ek = new double* [numElemRows];
    for (i = 0; i < numElemRows; i++) {
        ek[i] = new double[numElemRows];
        ef[i] = 0.0;
    }

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSetMatrix starting numElems %d loop...\n",
            numElems);
        fflush(debugFile_);
    }

    //now we'll loop through the elements in this workset, assembling
    //the stiffnesses into the global sparse system stiffness
    //matrix.

    for (k = 0; k < numElems; k++) {

        calculateLocalNodeIndices(elemConn[k], numNodesPerElem,
                                  &localNodeIndices[0]);


        //now we'll make a local dense copy of the element stiffness array
        //if the stiffness array was passed in using one of the "weird"
        //element formats.
        if (elemFormat != 0) {
            copyStiffness(elemStiffness[k], numElemRows, elemFormat, ek);
        }

//  our element-assembly logic assumes implicitly that all nodal solution
//  parameters are grouped contiguously, to simplify matrix allocation and other
//  low-level tasks.  If this isn't the case (e.g., for the FIELD_CENTRIC data
//  scheme, the only alternative interleave strategy supported), then
//  we need to reorder the rows and columns of the element stiffness and load
//  in order to facilitate the nodal-equation-oriented assembly ops below...

        if (blockRoster[currentElemBlockIndex].getInterleaveStrategy() != 0) {
            continue;  // remapping of interleaving not yet supported...
                       // but when it is, the remapping function goes here!
        }

        //  now let's obtain the scatter indices for assembling the equations
        //  into their appropriate places in the system stiffness
        //  structure...

        int numIndices = formElemScatterList(currentElemBlockIndex, elemConn[k],
                                             &localNodeIndices[0],
                                             &scatterIndices[0]);
//  these scatter indices are coming out of formElemScatterList as global
//  equation numbers. i.e., they have already had the appropriate offset added
//  to them, for the parallel case.

        if (elemFormat != 0) {
            packSharedStiffnessAndLoad(elemConn[k], numNodesPerElem,
                                       &localNodeIndices[0],
                                       &scatterIndices[0], numIndices,
                                       ek, ef);
        }
        else {
            packSharedStiffnessAndLoad(elemConn[k], numNodesPerElem,
                                       &localNodeIndices[0],
                                       &scatterIndices[0], numIndices,
                                       elemStiffness[k], ef);
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

//  update the current workset ID counter to keep our place in the data
//  stream

    currentWorkSetID++;

//  all done

    delete [] ef;
    for (i = 0; i < numElemRows; i++) {
        delete [] ek[i];
    }
    delete [] ek;

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadElemSetMatrix\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::loadElemSetRHS(int elemSetID, 
                          int numElems, 
                          const GlobalID *elemIDs,  
                          const GlobalID *const *elemConn,
                          const double *const *elemLoad) {
//
//  task: assemble the element stiffness matrices for a given workset
//

    baseTime_ = MPI_Wtime();

    //this prevents "declared and never referenced" warnings.
    (void)elemIDs;

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSetRHS, numElems: %d\n", numElems);
        fprintf(debugFile_,"currentElemBlockID: %d\n",(int)currentElemBlockID);
        fflush(debugFile_);
    }

    currentWorkSetID = elemSetID;

    if (numElems <= 0) return(0); //zero-length workset, do nothing.

    int i, k;

//  we're gonna need some control parameters that didn't come in through this
//  function's argument list.

    int numNodesPerElem = 0;//  number of nodes per element in this block
    int numElemRows = 0;    //  number of rows per element array in this block

    currentElemBlockIndex
               = search_ID_index(currentElemBlockID,&GID_blockIDList[0],
                              GID_blockIDList.size());

    assert(currentElemBlockIndex >= 0);

    numNodesPerElem = blockRoster[currentElemBlockIndex].
                                    getNumNodesPerElement();
    numElemRows = blockRoster[currentElemBlockIndex].getNumEqnsPerElement();

    if (debugOutput_) {
        fprintf(debugFile_,"numNodesPerElem: %d\n", numNodesPerElem);
        fprintf(debugFile_,"numElemRows: %d\n", numElemRows);
        fflush(debugFile_);
    }

    int* numNodalDOFPtr = blockRoster[currentElemBlockIndex].
                             pointerToNumNodalDOF(numNodesPerElem);

    //  dense local index of each nodal eqn
    IntArray nodeEqnIndices(numNodesPerElem);

    nodeEqnIndices[0] = 0;
    for (i = 1; i < numNodesPerElem; i++) {
        nodeEqnIndices[i] = nodeEqnIndices[i-1] + numNodalDOFPtr[i-1];
    }

    IntArray localNodeIndices(numNodesPerElem);

    // scatter indices into the system matrix
    IntArray scatterIndices(numElemRows);

    //(we're creating a dummy stiffness array (ek) to pass to the
    //'packSharedStiffnessAndLoad' function. This is just a lazy-ass
    //band-aid solution until I write some more general Shared-info-passing
    //code.
    double **ek = NULL;

    ek = new double* [numElemRows];
    for (i = 0; i < numElemRows; i++) {
        ek[i] = new double[numElemRows];
        for(k=0; k<numElemRows; k++)ek[i][k] = 0.0;
    }

    if (debugOutput_) {
        fprintf(debugFile_,"loadElemSetRHS starting numElems %d loop...\n",
            numElems);
        fflush(debugFile_);
    }

    //now we'll loop through the elements in this workset, assembling
    //the stiffnesses into the global sparse system stiffness
    //matrix.

    for (k = 0; k < numElems; k++) {

        calculateLocalNodeIndices(elemConn[k], numNodesPerElem,
                                  &localNodeIndices[0]);


//  our element-assembly logic assumes implicitly that all nodal solution
//  parameters are grouped contiguously, to simplify matrix allocation and other
//  low-level tasks.  If this isn't the case (e.g., for the FIELD_CENTRIC data
//  scheme, the only alternative interleave strategy supported), then
//  we need to reorder the rows and columns of the element stiffness and load
//  in order to facilitate the nodal-equation-oriented assembly ops below...

        if (blockRoster[currentElemBlockIndex].getInterleaveStrategy() != 0) {
            continue;  // remapping of interleaving not yet supported...
                       // but when it is, the remapping function goes here!
        }

        //  now let's obtain the scatter indices for assembling the equations
        //  into their appropriate places in the system rhs vector...

        int numIndices = formElemScatterList(currentElemBlockIndex, elemConn[k],
                                             &localNodeIndices[0],
                                             &scatterIndices[0]);
//  these scatter indices are coming out of formElemScatterList as global
//  equation numbers. i.e., they have already had the appropriate offset added
//  to them, for the parallel case.

        if (debugOutput_) {
            fprintf(debugFile_,"-- calling packSharedSiffnessAndLoad\n");
            fflush(debugFile_);
        }

        packSharedStiffnessAndLoad(elemConn[k], numNodesPerElem,
                                   &localNodeIndices[0],
                                   &scatterIndices[0], numIndices,
                                   ek, elemLoad[k]);

        // assembly operation

        assembleLoad(numElemRows, &scatterIndices[0], elemLoad[k]);

    }//end of the 'for k' loop over numElems

//  update the current workset ID counter to keep our place in the data
//  stream

    currentWorkSetID++;

//  all done

    for (i = 0; i < numElemRows; i++) {
        delete [] ek[i];
    }
    delete [] ek;

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadElemSetRHS\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::loadElemSetTransfers(int elemSetID, 
                                     int numElems,
                                     GlobalID** coarseNodeLists,
                                     GlobalID** fineNodeLists,
                                     int fineNodesPerCoarseElem,
                                     double*** elemProlong,
                                     double*** elemRestrict){
//
// This function is not implemented by BASE_SLE. It is not part of
// the 'single-assembly' code that BASE_SLE contains. This function is
// intended to be implemented by an 'outer shell' FEI-implementation,
// which will be managing the assembly of multiple different matrices for
// multi-level solutions.
//
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
void BASE_SLE::calculateLocalNodeIndices(const GlobalID* const elemConn,
                                         int numNodes, int* localIndices){
    for(int i=0; i<numNodes; i++){
        localIndices[i] = GlobalToLocalNode(elemConn[i]);
    }
}

//------------------------------------------------------------------------------
void BASE_SLE::copyStiffness(const double* const* elemStiff,
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
void BASE_SLE::implementAllBCs() {
//
// This function will handle the modifications to the stiffness and load
// necessary to enforce nodal boundary conditions.
//
    int numLocalNodes = getNumActNodes();

    IntArray essEqns;
    RealArray essAlpha;
    RealArray essGamma;

    IntArray otherEqns;
    RealArray otherAlpha;
    RealArray otherBeta;
    RealArray otherGamma;

    for(int node=0; node<numLocalNodes; node++){
        if (gnod_LocalNodes[node].hasBCDataFlag()) {
            GlobalID nodeID = gnod_LocalNodes[node].getGlobalNodeID();
            int      eqn    = getGlobalActEqnNumber(nodeID, node);

            int myNumBCs = 0;
            BCRecord* myBCPtr =
                     gnod_LocalNodes[node].pointerToBCRecords(myNumBCs);
            for (int jbc = 0; jbc < myNumBCs; jbc++) {

//  should this field offset be block-based?  Methinks different blocks may
//  cause problems for the following nodal-offset approach...  Hmmm???

                int myFieldOffset = myBCPtr[jbc].getFieldOffset();
                int myFieldSize;
                double* myAlpha = myBCPtr[jbc].pointerToAlpha(myFieldSize);
                double* myBeta = myBCPtr[jbc].pointerToBeta(myFieldSize);
                double* myGamma = myBCPtr[jbc].pointerToGamma(myFieldSize);

                for(int j = 0; j < myFieldSize; j++) {

//  compute this equation index for manipulating the matrix and rhs.

                    int thisEqn = eqn + myFieldOffset + j;

                    double alpha = myAlpha[j];
                    double beta = myBeta[j];
                    double gamma = myGamma[j];

//  is it an essential bc in the current solution component direction?

                    if ((alpha != 0.0) && (beta == 0.0)) {

                        int found = -1, insert = 0;
                        if (essEqns.size() > 0) {
                            found = find_index(thisEqn, &essEqns[0],
                                               essEqns.size(), &insert);
                            if (found < 0) {
                                essEqns.insert(insert, thisEqn);
                                essAlpha.insert(insert, alpha);
                                essGamma.insert(insert, gamma);
                            }
                            else {
                                essEqns.insert(found, thisEqn);
                                essAlpha.insert(found, alpha);
                                essGamma.insert(found, gamma);
                            }
                        }
                        else {
                            //if essEqns.size()==0, just append the stuff.
                            essEqns.append(thisEqn);
                            essAlpha.append(alpha);
                            essGamma.append(gamma);
                        }

                    }
                    else {

//  if we have a natural or mixed b.c. (beta != 0) then we add terms
//  to the diagonal and the rhs vector and hope for the best...

                        if (beta != 0.0) {

                            int found = -1, insert = 0;
                            if (otherEqns.size() > 0) {
                                found = find_index(thisEqn, &otherEqns[0],
                                                   otherEqns.size(), &insert);
                                if (found < 0) {
                                    otherEqns.insert(insert, thisEqn);
                                    otherAlpha.insert(insert, alpha);
                                    otherBeta.insert(insert, beta);
                                    otherGamma.insert(insert, gamma);
                                }
                                else {
                                    otherEqns.insert(found, thisEqn);
                                    otherAlpha.insert(found, alpha);
                                    otherBeta.insert(found, beta);
                                    otherGamma.insert(found, gamma);
                                }
                            }
                            else {
                                //if otherEqns.size()==0, just append the stuff.
                                otherEqns.append(thisEqn);
                                otherAlpha.append(alpha);
                                otherBeta.append(beta);
                                otherGamma.append(gamma);
                            }

                        }
                        else {
                            cout << "\n\ninconsistent BC specification, node :"
                                 << (int)nodeID << endl;
                        }
                    }
                }
            }
        }
    }

    if (essEqns.size() > 0)
        enforceEssentialBC(&essEqns[0], &essAlpha[0], &essGamma[0],
                           essEqns.size());

    if (otherEqns.size() > 0)
        enforceOtherBC(&otherEqns[0], &otherAlpha[0], &otherBeta[0],
                       &otherGamma[0], otherEqns.size());

    return;
}

//------------------------------------------------------------------------------
int BASE_SLE::endLoadElemBlock() {
//
//  tasks: end blocked-element data loading step
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"endLoadElemBlock\n");
        fflush(debugFile_);
    }

    checkElemBlocksLoaded++;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}
 
//------------------------------------------------------------------------------
//
// This function gets the maximum (across all blocks) nodes-per-element
// and the maximum number of nodal DOF.
//
void BASE_SLE::getMaxNodesAndDOF(int &nodes, int &DOF){

    int i, j;

    nodes = 0;
    DOF = 0;

    for(i=0; i<storeNumElemBlocks; i++){
        int tmp;
        int bNodes = blockRoster[i].getNumNodesPerElement();
        if (bNodes > nodes) nodes = bNodes;

        int *numNodeDOFPtr = blockRoster[i].pointerToNumNodalDOF(tmp);

        for(j=0; j<tmp; j++){
            if (numNodeDOFPtr[j] > DOF) DOF = numNodeDOFPtr[j];
        }
    }

    return;
}
 
//------------------------------------------------------------------------------
//
void BASE_SLE::sharedNodeLaunchInitRecvs(){

    int i, j, k, totalRemoteElems;

    int nodesPerElem, nodeDOF;
    getMaxNodesAndDOF(nodesPerElem, nodeDOF);
    shSize_ = nodesPerElem*nodeDOF;
    shCoeffSize_ = 0;

    //first, how many remote elements do our shared nodes appear in?
    totalRemoteElems = sharedNodes_->totalRemoteElems();

    if (totalRemoteElems <= 0){
        if (debugOutput_) {
            fprintf(debugFile_,"totalRemoteElems: %d, returning\n",
                    totalRemoteElems);
            fflush(debugFile_);
        }
        return;
    }

    if (shRequests_ != NULL) delete [] shRequests_;
    if (shProc_ != NULL) delete [] shProc_;
    if (shNodesFromProc_ != NULL) delete [] shNodesFromProc_;
    if (shScatRequests_ != NULL) delete [] shScatRequests_;

    int *tmpProcs = new int[numProcs_];
    for(i=0; i<numProcs_; i++){
        tmpProcs[i] = 0;
    }

    int numLocalShared;
    GlobalID *localSharedNodes = sharedNodes_->
                         pointerToLocalShared(numLocalShared);

    //first, figure out how many processors we'll be recv'ing from,
    //and how much stuff we'll be getting from each of them.
    int counter = 0;
    for(i=0; i<numLocalShared; i++){
        int numProcs;
        int *procIDs = sharedNodes_->procIDs(localSharedNodes[i], numProcs);

        for(j=0; j<numProcs; j++){
            int numRemoteElems = sharedNodes_->remoteElems(localSharedNodes[i],
                                                           procIDs[j]);
            for(k=0; k<numRemoteElems; k++){
                counter++;
                if (counter > totalRemoteElems)
                    cout << "ERROR, counter > totalRemoteElems"
                         << endl << flush;

                tmpProcs[procIDs[j]]++;
            }
        }
    }

    if (counter != totalRemoteElems)
        cout << "ERROR in SNLIR, counter != totalRemoteElems" << endl << flush;

    //now, we need to collapse the tmpProcs_ list down into the shProc_
    // list, which will be of length numShProcs_. So first calculate the
    //value of numShProcs_.
    numShProcs_ = 0;
    for(i=0; i<numProcs_; i++){
        if (tmpProcs[i] >0) numShProcs_++;
    }

    //so now we can allocate shProc_, the list which will tell us which
    //processors we'll be receiving stuff from, and shNodesFromProc_,
    //which will tell how many node-messages to expect from each processor.
    shProc_ = new int[numShProcs_];
    shNodesFromProc_ = new int[numShProcs_];

    counter = 0;
    for(i=0; i<numProcs_; i++){
        if (tmpProcs[i] > 0){
            shProc_[counter] = i;
            shNodesFromProc_[counter] = tmpProcs[i];
            counter++;
        }
    }

    delete [] tmpProcs;

 
    if (debugOutput_) {
        fprintf(debugFile_,"SNLIR: numShProcs_: %d\n", numShProcs_);
    for(i=0; i<numShProcs_; i++){
            fprintf(debugFile_,"SNLIR: shProc_[%d]: %d, shNodesFromProc_[%d]: %d\n",
                i, shProc_[i], i, shNodesFromProc_[i]);
    }
        fflush(debugFile_);
    }

    shNodeInfo_ = new NodeControlPacket*[numShProcs_];
    shRequests_ = new MPI_Request[numShProcs_];
    shScatRequests_ = new MPI_Request[numShProcs_];
    shScatterIndices_ = new int*[numShProcs_];

    for(i=0; i<numShProcs_; i++){
        shNodeInfo_[i] = new NodeControlPacket[shNodesFromProc_[i]];
        shScatterIndices_[i] = new int[shNodesFromProc_[i]*shSize_];
        for(j=0; j<shNodesFromProc_[i]*shSize_; j++){
            shScatterIndices_[i][j] = -1;
        }
    }

    //finally, launch the async receives.
    for(i=0; i<numShProcs_; i++){
        MPI_Irecv(shNodeInfo_[i],shNodesFromProc_[i], MPI_NodePacket,
                  shProc_[i], packet_tag1+shProc_[i], FEI_COMM_WORLD,
                  &shRequests_[i]);

        MPI_Irecv(shScatterIndices_[i], shNodesFromProc_[i]*shSize_, MPI_INT,
                  shProc_[i], indices_tag+shProc_[i], FEI_COMM_WORLD,
                  &shScatRequests_[i]);
    }

 
    if (debugOutput_) {
        fprintf(debugFile_,"leaving SNLIR\n");
        fflush(debugFile_);
    }

    return;
}

//------------------------------------------------------------------------------
void BASE_SLE::exchangeSharedMatrixData(){
//
// This function is where processors exchange stiffness data
// associated with shared nodes.
//
// shared nodes that we own:
//     - recv rows of stiffnesses and scatter indices
//       from sharing processors
// shared nodes that we don't own:
//     - send rows of stiffnesses scatter indices to
//       the owning processor
//
// For nodes that we own, we'll recv a whole bunch of stiffness coefs,
// along with scatter indices that are the connectivity info for the other
// processor's element.
//
    if (debugOutput_) {
        fprintf(debugFile_, "exchangeSharedMatrixData\n");
    }

    int numSharingProcs;
    int* sharingProcs = sharedNodes_->pointerToSharingProcs(numSharingProcs);

    if (debugOutput_) {
        fprintf(debugFile_, "-- sharingProcs: ");
        for(int i=0; i<numSharingProcs; i++) {
            fprintf(debugFile_,"%d ",sharingProcs[i]);
        }
        fprintf(debugFile_, "\n");
    }

    //the issue of mpi tags is gonna bite us sometime, but for now we're just
    //going to (lazily) use some arbitrary numbers.
    int infoTag = 11;
    int eqnTag = 13, coefTag = 17, indicesTag = 19, offsetsTag = 23;

    int numLengthParams = 2;

    int** recvInfo = NULL;
    MPI_Request* reqs = NULL;

    //now launch Irecvs for info about how many shared-node coefficients
    //and indices we'll be recving.
    if (numSharingProcs>0) {
        recvInfo = new int*[numSharingProcs];

        reqs = new MPI_Request[numSharingProcs];

        for(int i=0; i<numSharingProcs; i++) {
            recvInfo[i] = new int[numLengthParams]; 

            MPI_Irecv(recvInfo[i], numLengthParams, MPI_INT, sharingProcs[i], 
                      infoTag*(sharingProcs[i]+1), FEI_COMM_WORLD, &(reqs[i]));
        }
    }

    shBuffLoadD_->buildLists(); //buffer of stiffness coefficients
    shBuffLoadI_->buildLists(); //buffer of stiffness scatter indices

    MPI_Barrier(FEI_COMM_WORLD);

    int numRemoteOwnerProcs;
    int* remoteOwnerProcs = shBuffLoadD_->pointerToProcs(numRemoteOwnerProcs);

    if (debugOutput_) {
        fprintf(debugFile_,"remoteOwnerProcs: ");
        for(int ii=0; ii<numRemoteOwnerProcs; ii++)
            fprintf(debugFile_,"%d ",remoteOwnerProcs[ii]);
        fprintf(debugFile_,"\n");
        fflush(debugFile_);
    }

    //we're going to be sending 4 arrays to each remote owner processor with
    //whom we share nodes. Those 4 arrays only have 2 distinct lengths, which
    //will be described in the sendInfo array. sendInfo will be of size
    //numRemoteOwnerProcs X 2. For each remote owner, we'll need the following
    //lengths:
    //sendInfo[p][0]: numEqns ( = numOffsets-1)
    //sendInfo[p][1]: numCoefs ( = numIndices)
    //

    int** sendInfo = NULL;

    //here are the array declarations
    int** sendEqns;
    double** sendCoefs;
    int** sendIndices;
    int** sendOffsets;

    if (numRemoteOwnerProcs>0) {

        sendInfo = new int*[numRemoteOwnerProcs];

        sendEqns = new int*[numRemoteOwnerProcs];
        sendCoefs = new double*[numRemoteOwnerProcs];
        sendIndices = new int*[numRemoteOwnerProcs];
        sendOffsets = new int*[numRemoteOwnerProcs];

        for(int i=0; i<numRemoteOwnerProcs; i++) {
            sendInfo[i] = new int[numLengthParams];
            int proc = remoteOwnerProcs[i];

            int temp;
            sendEqns[i] = shBuffLoadD_->
                               pointerToEqnNumbers(sendInfo[i][0], proc);

            sendCoefs[i] = shBuffLoadD_->
                                         pointerToDoubles(sendInfo[i][1], proc);

            sendIndices[i] = shBuffLoadI_->pointerToInts(temp, proc);
            for(int ii=0; ii<temp; ii++) {
                if (sendIndices[i][ii] <= 0) {
                    cout << "********** sendIndices["<<i<<"]["<<ii<<"]: "
                        << sendIndices[i][ii] << ", from " << localRank_ << endl;
                }
            }

            sendOffsets[i] = shBuffLoadD_->pointerToOffsets(temp, proc);

            if (debugOutput_) {
                fprintf(debugFile_, "----to proc %d, sendInfo: ", proc);
                for(int j=0; j<numLengthParams; j++)
                    fprintf(debugFile_, "%d ",sendInfo[i][j]);
                fprintf(debugFile_,"\n");
            }

            MPI_Send(sendInfo[i], numLengthParams, MPI_INT, proc,
                     infoTag*(localRank_+1), FEI_COMM_WORLD);
        }
    }
    int i, j, index;
    MPI_Status status;

    //now lets complete the Irecvs for the recvInfo lists.
    for(i=0; i<numSharingProcs; i++) {
        MPI_Waitany(numSharingProcs, reqs, &index, &status);
    }

    if (numSharingProcs > 0) delete [] reqs;

    if (debugOutput_) {
        for(i=0; i<numSharingProcs; i++) {
            fprintf(debugFile_, "from proc %d, recvInfo: ", sharingProcs[i]);
            for(j=0; j<numLengthParams; j++)
                fprintf(debugFile_,"%d ", recvInfo[i][j]);
            fprintf(debugFile_,"\n");
        }
        fflush(debugFile_);
    }

    //here are the recv array declarations, and we'll make a list of
    //MPI_Requests for each.
    int** recvEqns;
    double** recvCoefs;
    int** recvIndices;
    int** recvOffsets;

    MPI_Request** recvReqs;

    //so now lets do the allocations and launch the Irecvs for the incoming
    //shared node stuff.
    if (numSharingProcs > 0) {

        recvEqns = new int*[numSharingProcs];
        recvCoefs = new double*[numSharingProcs];
        recvIndices = new int*[numSharingProcs];
        recvOffsets = new int*[numSharingProcs];

        recvReqs = new MPI_Request*[4];

        for(i=0; i<4; i++)
            recvReqs[i] = new MPI_Request[numSharingProcs];

        for(i=0; i<numSharingProcs; i++) {

            recvEqns[i] = new int[recvInfo[i][0]];
            recvCoefs[i] = new double[recvInfo[i][1]];
            recvIndices[i] = new int[recvInfo[i][1]];
            recvOffsets[i] = new int[recvInfo[i][0]+1];

            if (debugOutput_) {
                fprintf(debugFile_, "irecv from proc %d, recvEqns %d\n",
                        sharingProcs[i], recvInfo[i][0]);
            }

            MPI_Irecv(recvEqns[i], recvInfo[i][0], MPI_INT, sharingProcs[i],
                      eqnTag*(sharingProcs[i]+1), FEI_COMM_WORLD,
                      &(recvReqs[0][i]));

            if (debugOutput_) {
                fprintf(debugFile_, "irecv from proc %d, recvCoefs %d\n",
                        sharingProcs[i], recvInfo[i][1]);
            }

            MPI_Irecv(recvCoefs[i], recvInfo[i][1], MPI_DOUBLE, sharingProcs[i],
                      coefTag*(sharingProcs[i]+1), FEI_COMM_WORLD,
                      &(recvReqs[1][i]));

            if (debugOutput_) {
                fprintf(debugFile_, "irecv from proc %d, recvIndices %d\n",
                        sharingProcs[i], recvInfo[i][1]);
            }

            MPI_Irecv(recvIndices[i], recvInfo[i][1], MPI_INT, sharingProcs[i],
                      indicesTag*(sharingProcs[i]+1), FEI_COMM_WORLD,
                      &(recvReqs[2][i]));

            if (debugOutput_) {
                fprintf(debugFile_, "irecv from proc %d, recvOffsets %d\n",
                        sharingProcs[i], recvInfo[i][0]+1);
            }

            MPI_Irecv(recvOffsets[i], recvInfo[i][0]+1, MPI_INT,
                      sharingProcs[i], offsetsTag*(sharingProcs[i]+1),
                      FEI_COMM_WORLD, &(recvReqs[3][i]));
        }
    }

    MPI_Barrier(FEI_COMM_WORLD);

    //now we want to do the sends that the above Irecvs are waiting for.
    for(i=0; i<numRemoteOwnerProcs; i++) {

        if (debugOutput_) {
            fprintf(debugFile_, "send to proc %d, sendEqns %d\n",
                    remoteOwnerProcs[i], sendInfo[i][0]);
        }

        MPI_Send(sendEqns[i], sendInfo[i][0], MPI_INT, remoteOwnerProcs[i],
                 eqnTag*(localRank_+1), FEI_COMM_WORLD);

        if (debugOutput_) {
            fprintf(debugFile_, "send to proc %d, sendCoefs %d\n",
                    remoteOwnerProcs[i], sendInfo[i][1]);
        }

        MPI_Send(sendCoefs[i], sendInfo[i][1], MPI_DOUBLE, remoteOwnerProcs[i],
                 coefTag*(localRank_+1), FEI_COMM_WORLD);

        if (debugOutput_) {
            fprintf(debugFile_, "send to proc %d, sendIndices %d\n",
                    remoteOwnerProcs[i], sendInfo[i][1]);
        }

        MPI_Send(sendIndices[i], sendInfo[i][1], MPI_INT, remoteOwnerProcs[i],
                 indicesTag*(localRank_+1), FEI_COMM_WORLD);

        if (debugOutput_) {
            fprintf(debugFile_, "send to proc %d, sendOffsets %d\n",
                    remoteOwnerProcs[i], sendInfo[i][0]+1);
        }

        MPI_Send(sendOffsets[i], sendInfo[i][0]+1, MPI_INT, remoteOwnerProcs[i],
                 offsetsTag*(localRank_+1), FEI_COMM_WORLD);
    }    

    //and finally, we're ready to receive and deal with the shared node info
    //from the sharing (non-owner) processors.

    if (debugOutput_) {
        fprintf(debugFile_,"-- numSharingProcs: %d\n",numSharingProcs);
        fflush(debugFile_);
    }

    for(i=0; i<numSharingProcs; i++) {

        //wait for any of the eqn arrays to arrive...
        MPI_Waitany(numSharingProcs, recvReqs[0], &index, &status);

        if (debugOutput_) {
            fprintf(debugFile_,"completed MPI_Waitany, from proc: %d\n",
                    status.MPI_SOURCE);
            fflush(debugFile_);
        }

        //now, complete the recvs for the rest of the stuff from the
        //processor that sent the above eqn array.
        for(int iii=1; iii<4; iii++) {
            MPI_Wait(&recvReqs[iii][index], &status);

            if (debugOutput_) {
                fprintf(debugFile_,"completed MPI_Wait, iii: %d, index: %d\n",
                        iii, index);
                fflush(debugFile_);
            }
        }

        //now, stick the stuff into the matrix.

        //loop over the number of eqns that arrived...
        for(j=0; j<recvInfo[index][0]; j++) {
            int numCoefs = recvOffsets[index][j+1]-recvOffsets[index][j];
            int offset = recvOffsets[index][j];

            if (debugOutput_) {
                fprintf(debugFile_,"recvd eqn %d from proc %d, len %d\n",
                        recvEqns[index][j], sharingProcs[index], numCoefs);
                fflush(debugFile_);
            }

            for(int jj=0; jj<numCoefs; jj++) {
                if (recvIndices[index][offset+jj] <= 0) {
                    cerr << "proc " << localRank_ << ", recvd colInd "
                    << recvIndices[index][offset+jj] << " for eqn "
                    << recvEqns[index][j] << " from proc " <<sharingProcs[index]
                     << endl;
                    abort();
                }
            }

            if (debugOutput_) {
                fprintf(debugFile_," sumInShared, row %d, indices: ",
                        recvEqns[index][j]);
                for(int ii=0; ii<numCoefs; ii++) {
                    fprintf(debugFile_,"%d ",recvIndices[index][offset+ii]);
                }
                fprintf(debugFile_,"\n");
                fflush(debugFile_);
            }

            sumIntoSystemMatrix(recvEqns[index][j], numCoefs,
                                &(recvCoefs[index][offset]),
                                &(recvIndices[index][offset]));
        }
    }
 
    if (numSharingProcs > 0) {
        int ii;
        for(ii=0; ii<4; ii++) delete [] recvReqs[ii];
        delete [] recvReqs;

        for(ii=0; ii<numSharingProcs; ii++)
            delete [] recvInfo[ii];
        delete [] recvInfo;

        for(ii=0; ii<numSharingProcs; ii++) {
            delete [] recvEqns[ii];
            delete [] recvCoefs[ii];
            delete [] recvIndices[ii];
            delete [] recvOffsets[ii];
        }
 
        delete [] recvEqns;
        delete [] recvCoefs;
        delete [] recvIndices;
        delete [] recvOffsets;
    }

    if (numRemoteOwnerProcs > 0) {
        for(int ii=0; ii<numRemoteOwnerProcs; ii++)
            delete [] sendInfo[ii];
        delete [] sendInfo;

        delete [] sendEqns;
        delete [] sendCoefs;
        delete [] sendIndices;
        delete [] sendOffsets;
    }

    delete shBuffLoadD_;
    delete shBuffLoadI_;
    shBuffLoadAllocated_ = false;
 
    if (debugOutput_) {
        fprintf(debugFile_,"leaving exchangeSharedMatrixData\n");
        fflush(debugFile_);
    }

    return;
}
 
//------------------------------------------------------------------------------
void BASE_SLE::exchangeSharedRHSData(int rhsIndex){
//
// This function is where processors exchange RHS load data
// associated with shared nodes.
//
// shared nodes that we own:
//     - recv RHS coefficients from sharing processors
// shared nodes that we don't own:
//     - send RHS coefficients to the owning processor
//
    if (debugOutput_) {
        fprintf(debugFile_, "entering exchangeSharedRHSData\n");
    }

    int numSharingProcs;
    int* sharingProcs = sharedNodes_->pointerToSharingProcs(numSharingProcs);

    if (debugOutput_) {
        fprintf(debugFile_, "-- numSharingProcs: %d\n",numSharingProcs);
        fprintf(debugFile_, "-- sharingProcs: ");
        for(int i=0; i<numSharingProcs; i++) {
            fprintf(debugFile_,"%d",sharingProcs[i]);
        }
        fprintf(debugFile_, "\n");
    }

    //the issue of mpi tags is gonna bite us sometime, but for now we're just
    //going to (lazily) use some arbitrary numbers.
    int infoTag = 11;
    int rhsEqnTag = 3, rhsCoefTag = 5, rhsOffsetsTag = 7;

    int numLengthParams = 2;

    int** recvInfo = NULL;
    MPI_Request* reqs = NULL;

    //now launch Irecvs for info about how many shared-node coefficients
    //we'll be recving.
    if (numSharingProcs>0) {
        recvInfo = new int*[numSharingProcs];

        reqs = new MPI_Request[numSharingProcs];

        for(int i=0; i<numSharingProcs; i++) {
            recvInfo[i] = new int[numLengthParams];

            MPI_Irecv(recvInfo[i], numLengthParams, MPI_INT, sharingProcs[i], 
                      infoTag*(sharingProcs[i]+1), FEI_COMM_WORLD, &(reqs[i]));
        }
    }

    shBuffRHSLoadD_[rhsIndex].buildLists(); //buffer of rhs load coefficients.

    MPI_Barrier(FEI_COMM_WORLD);

    int numRemoteOwnerProcs;
    int* remoteOwnerProcs = shBuffRHSLoadD_[rhsIndex].
                                pointerToProcs(numRemoteOwnerProcs);

    if (debugOutput_) {
        fprintf(debugFile_,"numRemoteOwnerProcs: %d\n",numRemoteOwnerProcs);
        fprintf(debugFile_,"remoteOwnerProcs: ");
        for(int ii=0; ii<numRemoteOwnerProcs; ii++)
            fprintf(debugFile_,"%d ",remoteOwnerProcs[ii]);
        fprintf(debugFile_,"\n");
        fflush(debugFile_);
    }

    //we're going to be sending 3 arrays to each remote owner processor with
    //whom we share nodes. Those 3 arrays only have 2 distinct lengths, which
    //will be described in the sendInfo array. sendInfo will be of size
    //numRemoteOwnerProcs X 2. For each remote owner, we'll need the following
    //lengths:
    //sendInfo[p][0]: numRHSEqns ( = numRHSOffsets-1)
    //sendInfo[p][1]: numRHSCoefs
    //

    int** sendInfo = NULL;

    //here are the array declarations
    int** sendRHSEqns;
    double** sendRHSCoefs;
    int** sendRHSOffsets;

    if (numRemoteOwnerProcs>0) {

        sendInfo = new int*[numRemoteOwnerProcs];

        sendRHSEqns = new int*[numRemoteOwnerProcs];
        sendRHSCoefs = new double*[numRemoteOwnerProcs];
        sendRHSOffsets = new int*[numRemoteOwnerProcs];

        for(int i=0; i<numRemoteOwnerProcs; i++) {
            sendInfo[i] = new int[numLengthParams];
            int proc = remoteOwnerProcs[i];

            int temp;

            sendRHSEqns[i] = shBuffRHSLoadD_[rhsIndex].
                                      pointerToEqnNumbers(sendInfo[i][0], proc);

            sendRHSCoefs[i] = shBuffRHSLoadD_[rhsIndex].
                                        pointerToDoubles(sendInfo[i][1], proc);

            sendRHSOffsets[i] = shBuffRHSLoadD_[rhsIndex].
                                            pointerToOffsets(temp, proc);

            if (debugOutput_) {
                fprintf(debugFile_, "----to proc %d, sendInfo: ", proc);
                for(int j=0; j<numLengthParams; j++)
                    fprintf(debugFile_, "%d ",sendInfo[i][j]);
                fprintf(debugFile_,"\n");
            }

            MPI_Send(sendInfo[i], numLengthParams, MPI_INT, proc,
                     infoTag*(localRank_+1), FEI_COMM_WORLD);
        }
    }
    int i, j, index;
    MPI_Status status;

    //now lets complete the Irecvs for the recvInfo lists.
    for(i=0; i<numSharingProcs; i++) {
        MPI_Waitany(numSharingProcs, reqs, &index, &status);
    }

    if (numSharingProcs > 0) delete [] reqs;

    if (debugOutput_) {
        for(i=0; i<numSharingProcs; i++) {
            fprintf(debugFile_, "from proc %d, recvInfo: ", sharingProcs[i]);
            for(j=0; j<numLengthParams; j++)
                fprintf(debugFile_,"%d ", recvInfo[i][j]);
            fprintf(debugFile_,"\n");
        }
        fflush(debugFile_);
    }

    //here are the recv array declarations, and we'll make a list of
    //MPI_Requests for each.
    int** recvRHSEqns;
    double** recvRHSCoefs;
    int** recvRHSOffsets;

    MPI_Request** recvReqs;

    //so now lets do the allocations and launch the Irecvs for the incoming
    //shared node stuff.
    if (numSharingProcs > 0) {

        recvRHSEqns = new int*[numSharingProcs];
        recvRHSCoefs = new double*[numSharingProcs];
        recvRHSOffsets = new int*[numSharingProcs];

        recvReqs = new MPI_Request*[3];

        for(i=0; i<3; i++)
            recvReqs[i] = new MPI_Request[numSharingProcs];

        for(i=0; i<numSharingProcs; i++) {

            recvRHSEqns[i] = new int[recvInfo[i][0]];
            recvRHSCoefs[i] = new double[recvInfo[i][1]];
            recvRHSOffsets[i] = new int[recvInfo[i][0]+1];

            if (debugOutput_) {
                fprintf(debugFile_, "irecv from proc %d, recvRHSEqns %d\n",
                        sharingProcs[i], recvInfo[i][0]);
            }

            MPI_Irecv(recvRHSEqns[i], recvInfo[i][0], MPI_INT, sharingProcs[i],
                      rhsEqnTag*(sharingProcs[i]+1), FEI_COMM_WORLD,
                      &(recvReqs[0][i]));

            if (debugOutput_) {
                fprintf(debugFile_, "irecv from proc %d, recvRHSCoefs %d\n",
                        sharingProcs[i], recvInfo[i][1]);
            }

            MPI_Irecv(recvRHSCoefs[i], recvInfo[i][1], MPI_DOUBLE,
                      sharingProcs[i], rhsCoefTag*(sharingProcs[i]+1),
                      FEI_COMM_WORLD, &(recvReqs[1][i]));

            if (debugOutput_) {
                fprintf(debugFile_, "irecv from proc %d, recvRHSOffsets %d\n",
                        sharingProcs[i], recvInfo[i][0]+1);
            }

            MPI_Irecv(recvRHSOffsets[i], recvInfo[i][0]+1, MPI_INT,
                      sharingProcs[i], rhsOffsetsTag*(sharingProcs[i]+1),
                      FEI_COMM_WORLD, &(recvReqs[2][i]));
        }
    }

    MPI_Barrier(FEI_COMM_WORLD);

    //now we'll do the sends that the above Irecvs are waiting for.
    for(i=0; i<numRemoteOwnerProcs; i++) {

        if (debugOutput_) {
            fprintf(debugFile_, "send to proc %d, sendRHSEqns %d\n",
                    remoteOwnerProcs[i], sendInfo[i][0]);
        }

        MPI_Send(sendRHSEqns[i], sendInfo[i][0], MPI_INT, remoteOwnerProcs[i],
                 rhsEqnTag*(localRank_+1), FEI_COMM_WORLD);

        if (debugOutput_) {
            fprintf(debugFile_, "send to proc %d, sendRHSCoefs %d\n",
                    remoteOwnerProcs[i], sendInfo[i][1]);
        }

        MPI_Send(sendRHSCoefs[i], sendInfo[i][1], MPI_DOUBLE,
                 remoteOwnerProcs[i], rhsCoefTag*(localRank_+1),
                 FEI_COMM_WORLD);

        if (debugOutput_) {
            fprintf(debugFile_, "send to proc %d, sendRHSOffsets %d\n",
                    remoteOwnerProcs[i], sendInfo[i][0]+1);
        }

        MPI_Send(sendRHSOffsets[i], sendInfo[i][0]+1, MPI_INT,
                 remoteOwnerProcs[i], rhsOffsetsTag*(localRank_+1),
                 FEI_COMM_WORLD);
    }

    //and finally, we're ready to receive and deal with the shared node info
    //from the sharing (non-owner) processors.

    for(i=0; i<numSharingProcs; i++) {

        //wait for any of the rhs eqn arrays to arrive...
        MPI_Waitany(numSharingProcs, recvReqs[0], &index, &status);

        if (debugOutput_) {
            fprintf(debugFile_,"completed MPI_Waitany, index: %d\n",index);
            fflush(debugFile_);
        }

        //now, complete the recvs for the rest of the stuff from the
        //processor that sent the above eqn array.
        for(int ii=1; ii<3; ii++) {
            MPI_Wait(&recvReqs[ii][index], &status);

            if (debugOutput_) {
                fprintf(debugFile_,"completed MPI_Wait, ii: %d, index: %d\n",
                        ii, index);
                fflush(debugFile_);
            }
        }

        //loop over the number of rhs eqns that arrived...
        for(j=0; j<recvInfo[index][0]; j++) {
            int numCoefs = recvRHSOffsets[index][j+1]-recvRHSOffsets[index][j];
            for(int jj=0; jj<numCoefs; jj++) {
                int offset = recvRHSOffsets[index][j+jj];

                sumIntoRHSVector(1, &recvRHSEqns[index][j],
                                 &(recvRHSCoefs[index][offset]));
            }
        }
    }
 
    if (numSharingProcs > 0) {
        int ii;
        for(ii=0; ii<3; ii++) delete [] recvReqs[ii];
        delete [] recvReqs;

        for(ii=0; ii<numSharingProcs; ii++)
            delete [] recvInfo[ii];
        delete [] recvInfo;

        for(ii=0; ii<numSharingProcs; ii++) {
            delete [] recvRHSEqns[ii];
            delete [] recvRHSCoefs[ii];
            delete [] recvRHSOffsets[ii];
        }
 
        delete [] recvRHSEqns;
        delete [] recvRHSCoefs;
        delete [] recvRHSOffsets;
    }

    if (numRemoteOwnerProcs > 0) {
        for(int ii=0; ii<numRemoteOwnerProcs; ii++)
            delete [] sendInfo[ii];
        delete [] sendInfo;

        delete [] sendRHSEqns;
        delete [] sendRHSCoefs;
        delete [] sendRHSOffsets;
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving exchangeSharedRHSData\n");
        fflush(debugFile_);
    }

    return;
}

//------------------------------------------------------------------------------
int BASE_SLE::beginLoadCREqns(int numCRMultSets, 
                              int numCRPenSets) {
//
//  tasks: initiate constraint condition data loading step
//
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: beginLoadCREqns\n");
        fflush(debugFile_);
    }
        
    assert (numCRMultSets == storeNumCRMultRecords);
    assert (numCRPenSets == storeNumCRPenRecords);

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::loadCRMult(int CRMultID, 
                         int numMultCRs,
                         const GlobalID *const *CRNodeTable, 
                         const int *CRFieldList,
                         const double *const *CRWeightTable,
                         const double *CRValueList,
                         int lenCRNodeList) {
//
//  tasks: load step for Lagrange multiplier constraint condition sets
//
//   Question: do we really need to pass CRNodeTable again?  Here, I'm going
//            to ignore it for now (i.e., not store it, but just check it), 
//            as it got passed during the initialization phase, so all we'll 
//            do here is check for errors...
//

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: loadCRMult\n");
        fflush(debugFile_);
    }

    int i, j, k, m, lenList, numSolnParams;
    GlobalID sampleNodeNumber;

    k = CRMultID;
    lenList = ceqn_MultConstraints[k].getLenCRNodeList();

//  perform some temporary tests (for now, assuming that returned
//  ID's are simply the array indices of the constraint records)

    assert (lenList > 0);
    assert (k == ceqn_MultConstraints[k].getCRMultID());
    assert (numMultCRs == ceqn_MultConstraints[k].getNumMultCRs());

//  recall the data stored earlier in the call to initCRMult() and insure
//  that the passed data (here, the node table) agrees with the
//  initialization data
    
    int ntRowCheck = 0;
    int ntColCheck = 0;
    GlobalID **CRNodeTablePtr = 
        ceqn_MultConstraints[k].pointerToCRNodeTable(ntRowCheck, ntColCheck);
    assert (ntRowCheck == numMultCRs);
    assert (ntColCheck == lenList);
    bool **CRIsLocalPtr = 
        ceqn_MultConstraints[k].pointerToCRIsLocalTable(ntRowCheck, ntColCheck);

    for (i = 0; i < numMultCRs; i++) {
        for (j = 0; j < lenList; j++) {
            assert(CRNodeTablePtr[i][j] == CRNodeTable[i][j]);
        }
    }

    int fieldRowCheck = 0;
    int *CRFieldListPtr =
        ceqn_MultConstraints[k].pointerToCRFieldList(fieldRowCheck);
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

    int *tableRowLengths = new int [lenList];
    for (i = 0; i < lenList; i++) {
        if ((CRIsLocalPtr[0][i]) == false) { 
            numSolnParams = externalNodes_->numSolnParams(CRNodeTablePtr[0][i]);
        }
        else {
            sampleNodeNumber = CRNodeTablePtr[0][i];
           numSolnParams = getNumSolnParams(sampleNodeNumber);
        }
       tableRowLengths[i] = numSolnParams;
    }
    ceqn_MultConstraints[k].allocateCRNodeWeights(lenList, tableRowLengths);
    int wtRowCheck = 0;
    int *wtColCheck = NULL;
    double **CRWeightTablePtr = 
        ceqn_MultConstraints[k].pointerToCRNodeWeights(wtRowCheck, wtColCheck);
    assert (wtRowCheck == lenList);
    for (i = 0; i < lenList; i++) {
        assert (wtColCheck[i] == tableRowLengths[i]);
    }

    for (i = 0; i < lenList; i++) {
        for (j = 0; j < tableRowLengths[i]; j++) {
            CRWeightTablePtr[i][j] = CRWeightTable[i][j];
        }
    }
    
    delete [] tableRowLengths;
   
    ceqn_MultConstraints[k].allocateCRConstValues(numMultCRs);
    int cvRowCheck = 0;
    double *CRValueListPtr =
            ceqn_MultConstraints[k].pointerToCRConstValues(cvRowCheck);
    assert (cvRowCheck == numMultCRs);
    for (i = 0; i < numMultCRs; i++) {
        CRValueListPtr[i] = CRValueList[i];
    }

//    ceqn_MultConstraints[k].dumpToScreen();   //$kdm debugging

//  next, perform assembly of the various terms into the system arrays
//  (this is a good candidate for a separate function...)

//  note use of CRMultID as a position index here... ultimately, we need
//  to set aside a space in the appropriate ceqn_MultConstraints[] object
//  for caching this offset into the local storage for the system stiffness
//  and load arrays
    
//  first, determine how many columns each constraint can contribute to
//  brute-force kludge connection to ISIS++'s sum-into-row function here...
//  by doing the assembly on an entry-by-entry basis (which is reasonable
//  for the column-like portions of the stiffness created by the Lagrange
//  multiplier, but it's inefficient for the row-like portions, which should
//  ultimately get a row-wise scatter operation implemented).

    int irow, jcol, ib, one = 1;
    int multScatter[1];
    double values[1];

    for (i = 0; i < numMultCRs; i++) {
        irow = ceqn_MultConstraints[k].getLocalEqnID() + i + localStartRow_;
        for (j = 0; j < lenList; j++) {
            int myFieldID = CRFieldList[j];

//  here, perform the summations on the local nodes separately from those 
//  on remote nodes.  In the latter case, we can easily only access the 
//  row-oriented terms arising from the Lagrange constraints, so we'll 
//  postpone the column-oriented accesses until later.

//  local nodes first (this is probably the most common, and it's certainly
//  the easiest)

            if (CRIsLocalPtr[i][j] == true) {
                int myLocalNode = GlobalToLocalNode(CRNodeTablePtr[i][j]);
                int myLocalEqn = gnod_LocalNodes[myLocalNode].getLocalEqnID();
                numSolnParams = gnod_LocalNodes[myLocalNode].getNumNodalDOF();
                ib = irow; 
                sumIntoRHSVector(1, &ib, &(CRValueListPtr[i]));

//$kdm v1.0 begin work needed for constraint handling---------------------
//  here, we need to initialize the values[] array to zeros, and then pack
//  the weights into only those locations corresponding to the given field
//  in this local-node case, we can alternatively just allocation sufficient
//  space in the values[] array for the given field, and perform the 
//  sum-into-row operation only for these nonzero values
//$kdm v1.0 end work comments

                int myFieldOffset = 
                    gnod_LocalNodes[myLocalNode].getFieldOffset(myFieldID);
                assert(myFieldOffset >= 0);
                int myFieldCardinality = 
                    getFieldCardinality(myFieldID);
                assert(myFieldCardinality > 0);
                    
                for (m = myFieldOffset; m < myFieldCardinality; m++) {
                    jcol = myLocalEqn + m + localStartRow_;
                    values[0] = CRWeightTablePtr[j][m];
                    multScatter[0] = jcol;
                    sumIntoSystemMatrix(irow, one, values, multScatter);

                    multScatter[0] = irow;
                    sumIntoSystemMatrix(jcol, one, values, multScatter);
                }
            }
            else {

//  non-local nodes (called "external", but we'll be putting the shared 
//  but non-local nodes into the same mix eventually...

                int myExtNodeID = CRNodeTablePtr[i][j];
               int myEqnNumber = externalNodes_->globalEqn(myExtNodeID);
                numSolnParams = externalNodes_->numSolnParams(myExtNodeID);

//  perform the operations on the local rows here...            

                ib = irow; 
                sumIntoRHSVector(1, &ib, &(CRValueListPtr[i]));

                int myFieldOffset = 
                    externalNodes_->getFieldOffset(myExtNodeID, myFieldID);
                assert(myFieldOffset >= 0);
                int myFieldCardinality = getFieldCardinality(myFieldID);
                assert(myFieldCardinality > 0);
                    
                for (m = myFieldOffset; m < myFieldCardinality; m++) {
                   jcol = myEqnNumber + m;
                   values[0] = CRWeightTablePtr[j][m];
                   multScatter[0] = jcol;
 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, loadCRMult: calling sumIntoSystemMatrix, row=%d, col=%d\n",localRank_, ib, jcol);
    }
                    sumIntoSystemMatrix(ib, one, values, multScatter);
                }

//  and while we're here, let's send the weight packet off to the 
//  processor that will be needing it.

//  first, declare and load up a weight packet

                NodeWeightPacket wtPacket;
                wtPacket.nodeID = myExtNodeID;
                wtPacket.sysEqnID = irow + i;

//  initialize the packet to all zeros so we can insert the field-based
//  weight information

                for (m = 0; m < numSolnParams; m++) {
                    wtPacket.weights[m] = 0.0;
                }

//  now insert the field-based weights into the larger node-based packet

                for (m = myFieldOffset; m < myFieldCardinality; m++) {
                    wtPacket.weights[m] = CRWeightTablePtr[j][m];
                }
                
 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, loadCRMult: sending packet for node %d to proc %d\n",
         localRank_,(int)myExtNodeID,externalNodes_->ownerProc(myExtNodeID));
    }

//  now, send it.

                MPI_Send(&wtPacket, 1, MPI_NodeWtPacket,
                         externalNodes_->ownerProc(myExtNodeID),
                         extRecvPacketTag, FEI_COMM_WORLD);
            }
        }
    }
    
    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}


//------------------------------------------------------------------------------
//
//  tasks: perform penalty constraint relation data loading step
//
int BASE_SLE::loadCRPen(int CRPenID, 
                        int numPenCRs, 
                        const GlobalID *const *CRNodeTable,
                        const int *CRFieldList,
                        const double *const *CRWeightTable,
                        const double *CRValueList,
                        const double *penValues,
                        int lenCRNodeList) {

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: loadCRPen\n");
        fflush(debugFile_);
    }

    //this 'if' prevents "declared and never referenced" warnings.
    if (CRFieldList);

    int i, ii, jj, j, k, lenList, numSolnParams;

    k = CRPenID;
    lenList = ceqn_PenConstraints[k].getLenCRNodeList();

//  perform some temporary tests (for now, assuming that returned ID's are
//  simply the array indices of the constraint records)

    assert (lenList == lenCRNodeList);
    assert (k == ceqn_PenConstraints[k].getCRPenID());
    assert (numPenCRs == ceqn_PenConstraints[k].getNumPenCRs());

//  recall the data stored earlier in the call to initCRPen() and insure 
//  that the passed data (here, the node table) agrees with the 
//  initialization data
    
    int ntRowCheck = 0;
    int ntColCheck = 0;
    GlobalID **CRNodeTablePtr = 
        ceqn_PenConstraints[k].pointerToCRNodeTable(ntRowCheck, ntColCheck);
    assert (ntRowCheck == numPenCRs);
    assert (ntColCheck == lenList);
    bool **CRIsLocalPtr = 
      ceqn_PenConstraints[k].pointerToCRIsLocalTable(ntRowCheck, ntColCheck);
                                  
    for (i = 0; i < numPenCRs; i++) {
        for (j = 0; j < lenList; j++) {
            assert(CRNodeTablePtr[i][j] == CRNodeTable[i][j]);
        }
    }

//  squirrel away the new data (weight table and constant value list) 
//  using the same methods used in InitCRPen()

    int *tableRowLengths = new int [lenList];
    for (i = 0; i < lenList; i++) {
        if (CRIsLocalPtr[0][i] == false) { 
            numSolnParams = getNodeDOF(CRNodeTablePtr[0][i]);
        }
        else {
            numSolnParams = getNumSolnParams(CRNodeTablePtr[0][i]);
        }
        tableRowLengths[i] = numSolnParams;
    }
    ceqn_PenConstraints[k].allocateCRNodeWeights(lenList, tableRowLengths);
    int wtRowCheck = 0;
    int *wtColCheck;
    double **CRWeightTablePtr = 
        ceqn_PenConstraints[k].pointerToCRNodeWeights(wtRowCheck, wtColCheck);
    assert (wtRowCheck == lenList);
    for (i = 0; i < lenList; i++) {
        assert (wtColCheck[i] == tableRowLengths[i]);
    }
    for (i = 0; i < lenList; i++) {
        for (j = 0; j < tableRowLengths[i]; j++) {
          CRWeightTablePtr[i][j] = CRWeightTable[i][j];
        }
    }

    delete [] tableRowLengths;

    ceqn_PenConstraints[k].allocateCRConstValues(numPenCRs);
    int cvRowCheck = 0;
    double *CRValueListPtr =
                 ceqn_PenConstraints[k].pointerToCRConstValues(cvRowCheck);
    assert (cvRowCheck == numPenCRs);
    for (i = 0; i < numPenCRs; i++) {
        CRValueListPtr[i] = CRValueList[i];
    }

//  lets get a penID that's unique across the set of penalty constraints
//  on this processor. For this kth constraint set, the penID will be the
//  number of constraint relations in constraint sets 0..k-1.

    int penID = 0;
    for(i=0; i<k; i++){
        penID += ceqn_PenConstraints[i].getNumPenCRs();
    }

    int p, proc;
    
//  now we're going to look through these constraint relations, and send
//  weights etc., to any processors that own nodes in these constraints.

    for(p=0; p<numPenCRs; p++){
    
//  we only want to send the weight stuff to each relevant processor ONCE. 
//  Each time we send to a processor, we'll put that proc in a list, so we
//  can make sure we don't send it again. 

        int *procList = NULL;
        int lenProcList = 0;

        for(i=0; i<lenList; i++){
            if ((proc = externalNodes_->ownerProc(CRNodeTablePtr[p][i])) >=0){
                if (proc != localRank_) {
                
//  this is an external node that we don't own, which means that its owner
//  will need the weights associated with this constraint relation.

                    if (!inList(procList, lenProcList, proc)) {
                    
//  so we haven't already sent this CR to this proc. record this proc in 
//  the list, and then proceed to send stuff to it.

                        appendIntList(&procList, &lenProcList, proc);


//  We'll send two messages.  The first will just be the penID

                        MPI_Send(&penID,1,MPI_INT,proc,indices_tag,
                                 FEI_COMM_WORLD);
 
    if (debugOutput_) {
        fprintf(debugFile_,"%d in loadCRPen, sending penID %d to proc %d\n",
           localRank_, penID, proc);
    }

//  The second message will be a bunch of doubles which includes the weight
//  table, value list, and penValues.  The receiving processor will know 
//  the dimensions required to unpack this stuff.  First, we'll pack this 
//  stuff into a 1D array.  The number of items in the array is:
//        weight table: lenList*numSolnParams
//        value list:   1
//        penValues:    1
//
//  !!!!!! NOTE, numSolnParams could be different for each node in the 
//  constraint relation, so the above dimensions need to be generalized.
                    
                        double *array = new double[lenList*(numSolnParams+1) +
                                                   numPenCRs];
                        int ind = 0;
                        for(jj=0; jj<numSolnParams; jj++){
                            for(ii=0; ii<lenList; ii++){
                                array[ind++] = CRWeightTablePtr[ii][jj];
                            }
                        }

                        array[ind++] = CRValueListPtr[p];

                        array[ind++] = penValues[p];
 
                        MPI_Send(array,ind,MPI_DOUBLE,proc,coeff_tag,
                                 FEI_COMM_WORLD);

                        delete [] array;
                    }
                }
            }
        }
        penID++;
    }

//  use some brute force to construct these penalty contributions on the 
//  fly, primarily to simplify -reading- this thing, so that the derivations
//  in the annotated implementation document are more readily followed...

    int one = 1;
    int multScatter[1];
    double values[1];
    double myWeight, myPenFac, iFac, jFac;

    for (p = 0; p < numPenCRs; p++) {
        myPenFac = penValues[p];
        for (i = 0; i < lenList; i++) {
            int ipNode = CRNodeTablePtr[p][i];
            int iptest = externalNodes_->ownerProc(ipNode);
            for (j = 0; j < lenList; j++) {
                int jpNode = CRNodeTablePtr[p][j];
                int jptest = externalNodes_->ownerProc(jpNode);

                if ((iptest < 0) && (jptest < 0)) {
                
//  case 1: both nodes i and j are local, so we can accumulate these terms
//  into the sparse matrix with no communications

                    int iLocalNode = GlobalToLocalNode(ipNode);
                    int iLocalEqn = gnod_LocalNodes[iLocalNode].getLocalEqnID();
                    int iRows = gnod_LocalNodes[iLocalNode].getNumNodalDOF();
                    int jLocalNode = GlobalToLocalNode(jpNode);
                    int jLocalEqn = gnod_LocalNodes[jLocalNode].getLocalEqnID();
                    int jCols = gnod_LocalNodes[jLocalNode].getNumNodalDOF();

                    for (int irIndex = 0; irIndex < iRows; irIndex++) {
                        int ir = iLocalEqn + irIndex + localStartRow_;
                        iFac = CRWeightTablePtr[i][irIndex];
                        double val = myPenFac*iFac*CRValueListPtr[p];
                        sumIntoRHSVector(1, &ir, &val);
                        for (int jcIndex = 0; jcIndex < jCols; jcIndex++) {
                            int jc = jLocalEqn + jcIndex + localStartRow_;
                            jFac = CRWeightTablePtr[j][jcIndex];
                            myWeight = myPenFac*iFac*jFac;
                            values[0] = myWeight;
                            multScatter[0] = jc;
 
    if (debugOutput_) {
        fprintf(debugFile_,"%d loadCRPen case 1, summing into row %d, col %d\n",
           localRank_, ir, jc);
        fprintf(debugFile_,"%d loadCRPen, myPenFac %e, iFac %e, jFac %e\n",
           localRank_, myPenFac, iFac, jFac);
    }
                            sumIntoSystemMatrix(ir, one, values, multScatter);
                        }
                    }
                }
                else if ((iptest < 0) && (jptest >= 0)) {
                
//  case 2: node i is local, and node j is remote, so we have to get the 
//  system equation IDs for node j out of externalNodes_, and then we can 
//  add the values for the submatrices corresponding to (i,i) and (i,j), 
//  into the global system matrix A

                    int iLocalNode = GlobalToLocalNode(ipNode);
                    int iLocalEqn = gnod_LocalNodes[iLocalNode].getLocalEqnID();
                    int iRows = gnod_LocalNodes[iLocalNode].getNumNodalDOF();
                    
                    int jRemoteEqn = externalNodes_->globalEqn(jpNode);
                    int jCols = externalNodes_->numSolnParams(jpNode);
                    
                    for (int irIndex = 0; irIndex < iRows; irIndex++) {
                        int ir = iLocalEqn + irIndex + localStartRow_;
                        iFac = CRWeightTablePtr[i][irIndex];
                        double val = myPenFac*iFac*CRValueListPtr[i];
                        sumIntoRHSVector(1, &ir, &val);
                        for (int jcIndex = 0; jcIndex < jCols; jcIndex++) {
                            int jc = jRemoteEqn + jcIndex;
                            jFac = CRWeightTablePtr[j][jcIndex];
                            myWeight = myPenFac*iFac*jFac;
                            values[0] = myWeight;
                            multScatter[0] = jc;
 
    if (debugOutput_) {
        fprintf(debugFile_,"%d loadCRPen case 2, summing into row %d, col %d\n",
           localRank_, ir, jc);
        fprintf(debugFile_,"%d loadCRPen, myPenFac %e, iFac %e, jFac %e\n",
           localRank_, myPenFac, iFac, jFac);
    }
                            sumIntoSystemMatrix(ir , one, values, multScatter);
                        }
                    }
                }

//  case 3: node i is remote, and node j is local, so the space for the 
//  submatrices corresponding to (i,i) and (i,j) will be located on a 
//  different processor, which means we need to send the weight vector 
//  and penalty value to that processor. This was already done, in the 
//  code above.

            }
        }
    }
    
    wTime_ += MPI_Wtime() - baseTime_;
    
    return(0);
}

//------------------------------------------------------------------------------
//
int BASE_SLE::endLoadCREqns() {

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: endLoadCREqns\n");
        fflush(debugFile_);
    }

//  here, all the constraint relations have been loaded, so we can go ahead
//  and fill in the column-oriented components of the matrix, namely the ones
//  that require communications between processors

    int i, ii, j, jj, k;

//  now post the receives required to initialize storage in the sparse
//  matrix structure by catching the remote node weight packets. We're
//  going to first have to loop through our send nodes and figure out how
//  many packets to expect...

    int numSendProcs;
    int*sendProcs = externalNodes_->sendProcListPtr(numSendProcs);

    int* lenLocalNodeIDs;
    GlobalID** localNodeIDsPtr = externalNodes_->
                                          localNodeIDsPtr(&lenLocalNodeIDs);

    int numPackets = 0;
    for (i=0; i<numSendProcs; i++) {
        for (j=0; j<lenLocalNodeIDs[i]; j++) {
                numPackets +=externalNodes_->
                                  getNumMultCRs(localNodeIDsPtr[i][j]);
        }
    }

 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, recv'ing %d MultCR packets\n",localRank_,
            numPackets);
    }

//  now we know how many packets to expect, so lets declare some packets 
//  to receive into, and post the receives for 'em. We're doing this 
//  asynchronously, because we don't know which order they'll arrive in.

    NodeWeightPacket* wtPackets = new NodeWeightPacket[numPackets];
    MPI_Status status;
    MPI_Request* wtRequests = new MPI_Request[numPackets];

    for (i = 0; i < numPackets; i++) {
        MPI_Irecv(&(wtPackets[i]),1,MPI_NodeWtPacket, MPI_ANY_SOURCE,
                  extRecvPacketTag, FEI_COMM_WORLD, &wtRequests[i]);
    }

    int index;
    for (i = 0; i < numPackets; i++) {
        MPI_Waitany(numPackets, wtRequests, &index, &status);
    }

//  ok, so now we know they're all here.  so we can proceed to stick the 
//  stuff into the matrix.
    
    int multScatter[1];
    double values[1];
    int one = 1;

//  Lagrange multiplier constraints

    for (j = 0; j < numPackets; j++) {
        GlobalID myNodeID = wtPackets[j].nodeID;
        int myLocalNode = GlobalToLocalNode(myNodeID);
        assert (myLocalNode >= 0);

        int myColumnID = wtPackets[j].sysEqnID;
        int myBaseEqnID = gnod_LocalNodes[myLocalNode].getLocalEqnID();       
            
        int numSolnParams = gnod_LocalNodes[myLocalNode].getNumNodalDOF();

        for (k = 0; k < numSolnParams; k++) {
            int myRow = myBaseEqnID + k + localStartRow_;
            values[0] = wtPackets[j].weights[k];
           multScatter[0] = myColumnID;

 
    if (debugOutput_) {
        fprintf(debugFile_,"endLoadCREqns, node %d, calling sumIntoSystemMatrix, row=%d, col=%d\n",
           (int)myNodeID,myRow,myColumnID);
    }
           sumIntoSystemMatrix(myRow, one, values, multScatter);
        }
    }

    delete [] wtPackets;
    delete [] wtRequests;

//  now we need to catch and deal with the penalty constraint stuff, if
//  there is any.

    numPackets = externalNodes_->remoteNumPenCRs();
    MPI_Request* requests = new MPI_Request[numPackets];
    int proc, *penIDs = new int[numPackets];

    for(i=0; i<numPackets; i++){
        MPI_Irecv(&penIDs[i],1,MPI_INT, MPI_ANY_SOURCE,
                  indices_tag, FEI_COMM_WORLD, &requests[i]);
    }

//  now get the stuff we'll need out of the externalNodes_ record.

    int *lenRemotePenNodes;
    GlobalID **remotePenNodes = externalNodes_->
                                     remotePenNodeTable(&lenRemotePenNodes);
    int **remotePenEqnIDs = externalNodes_->
                                     remotePenEqnIDTable(&lenRemotePenNodes);
    int **remotePenDOFs = externalNodes_->
                                     remotePenNumDOFTable(&lenRemotePenNodes);

 
    if (debugOutput_) {
        fprintf(debugFile_,"%d in endLoadCREqns, incoming PenCRs: %d\n",
        localRank_, numPackets);
    }

    int CRIndex;
    
//  now catch and deal with the penalty stuff.

    for(i=0; i<numPackets; i++){
        MPI_Waitany(numPackets, requests, &index, &status);
        proc = status.MPI_SOURCE;
        CRIndex = externalNodes_->getRemotePenIDIndex(penIDs[index], proc);

 
    if (debugOutput_) {
        fprintf(debugFile_,"%d, recv'd penID %d from proc %d\n",localRank_,
       penIDs[index], proc);
    }
        int size = externalNodes_->remotePenArraySize(penIDs[index], proc);
        double *array = new double[size];

        MPI_Recv(array, size, MPI_DOUBLE, proc, coeff_tag, FEI_COMM_WORLD,
                 &status);

//  now we've got the stuff, so lets put it in the matrix and rhs vector.

        double CRValue = array[size-2];
        double penValue = array[size-1];

        int lenList = lenRemotePenNodes[CRIndex];

//  j will loop over the nodes in this penalty constraint.

        for(j=0; j<lenList; j++){

//  some of the nodes in this penalty constraint may not be local OR 
//  external nodes. That's fine, we'll do nothing for nodes that we don't 
//  own.

            if (externalNodes_->ownerProc(remotePenNodes[CRIndex][j]) ==
                localRank_){
                
//  we own this node, so loop across the 'row' of penNodes and form the 
//  contributions in this node's equation.

//  "this node's equation" consists of DOF equations, so ii will loop over 
//  the DOF at this node.

                for(ii=0; ii<remotePenDOFs[CRIndex][j]; ii++){
                    int row = remotePenEqnIDs[CRIndex][j] + ii;
                    double rowFac = array[ii*lenList + j];

//  sum the contribution into the rhs vector

                    double val = penValue*rowFac*CRValue;
                    sumIntoRHSVector(1, &row, &val);

//  the penalty constraint will make contributions at 'lenList' different 
//  places in this row of the matrix.  k will loop over the nodes at which
//  contributions will be made.

                    for(k=0; k<lenList; k++){

//  at each node, there are of course DOF number of contributions, jj will
//  loop over the DOF at this node.

                        for(jj=0; jj<remotePenDOFs[CRIndex][k]; jj++){
                            int col = remotePenEqnIDs[CRIndex][k] + jj;
                            double colFac = array[jj*lenList + k];
                            double thisWeight = penValue*rowFac*colFac;

 
    if (debugOutput_) {
        fprintf(debugFile_,"%d endLoadCREqn, summing into row %d, col %d\n",
           localRank_, row, col);
        fprintf(debugFile_,"%d loadCRPen, penValue %e, rowFac %e, colFac %e\n",
           localRank_, penValue, rowFac, colFac);
    }

// now sum the contribution into the matrix.

                            sumIntoSystemMatrix(row, one, &thisWeight, &col);
                        }
                    }  // end of k loop
                }
            }  // end of 'if we own it' block
        }
    }  // end of 'for i over numPackets' loop

    delete [] requests;
    delete [] penIDs;

    wTime_ += MPI_Wtime() - baseTime_;

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::loadComplete() {

    baseTime_ = MPI_Wtime();
 
//  all blocks have been loaded, so let's
//  have all processors exchange data associated with shared nodes.

    if (debugOutput_) {
        fprintf(debugFile_,"loadComplete, calling exchangeSharedMatrixData\n");
        fflush(debugFile_);
    }

    exchangeSharedMatrixData();
    MPI_Barrier(FEI_COMM_WORLD);
    for(int i=0; i<numRHSs_; i++) {
        if (debugOutput_) {
            fprintf(debugFile_,"   calling exchangeSharedRHSData(%d)\n",i);
            fflush(debugFile_);
        }

        setRHSIndex(i);
        exchangeSharedRHSData(i);
    }

    delete [] shBuffRHSLoadD_;
    shBuffRHSAllocated_ = false;

    if (debugOutput_) {
        fprintf(debugFile_,"leaving loadComplete\n");
        fflush(debugFile_);
    }
 
    wTime_ += MPI_Wtime() - baseTime_;
 
    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::getParam(const char *flag, int numParams,
                       char **paramStrings, char *param){
//
//  This is a private function. Used internally by BASE_SLE only.
//  paramStrings is a collection of string pairs - each string in
//  paramStrings consists of two strings separated by a space.
//  This function looks through the strings in paramStrings, looking
//  for one that contains flag in the first string. The second string
//  is then returned in param.
//  Assumes that param is allocated by the calling code.
//

    int i;
    char temp[256];

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
void BASE_SLE::appendParamStrings(int numStrings, char **strings){

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
int BASE_SLE::setMatScalars(int* IDs, double* scalars, int numScalars){
    (void)IDs;
    (void)scalars;
    (void)numScalars;
    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::setRHSScalars(int* IDs, double* scalars, int numScalars){
    (void)IDs;
    (void)scalars;
    (void)numScalars;
    return(0);
}

//------------------------------------------------------------------------------
void BASE_SLE::parameters(int numParams, char **paramStrings) {
//
// this function takes parameters for setting internal things like solver
// and preconditioner choice, etc.
//
    baseTime_ = MPI_Wtime();
 
    if (debugOutput_) {
        fprintf(debugFile_,"BASE_SLE: parameters\n");
        fflush(debugFile_);
    }
 
    if (numParams == 0 || paramStrings == NULL) {
 
        if (debugOutput_) {
            fprintf(debugFile_,"--- no parameters.\n");
        }
    }
    else {
        // take a copy of these parameters, for later use.
        appendParamStrings(numParams, paramStrings);

        char param[256];

        if ( getParam("outputLevel",numParams,paramStrings,param) == 1){
            sscanf(param,"%d", &outputLevel_);
        }

        if ( getParam("internalFei",numParams,paramStrings,param) == 1){
            sscanf(param,"%d", &internalFei_);
        }

//For now, BASE_SLE won't open its own debug output file. This will be 
//handled by the derived class (ISIS_SLE or Aztec_SLE).
//
//        if ( getParam("debugOutput",numParams,paramStrings,param) == 1){
//            char *name = new char[32];
//            sprintf(name, "BASE_SLE%d_debug",internalFei_);
//            setDebugOutput(param, name);
//            delete [] name;
//        }

        if (debugOutput_) {
           fprintf(debugFile_,"--- numParams %d\n",numParams);
           for(int i=0; i<numParams; i++){
               fprintf(debugFile_,"------ paramStrings[%d]: %s\n",i,
                       paramStrings[i]);
           }
        }
    }

    if (debugOutput_) {
        fprintf(debugFile_,"leaving parameters function\n");
        fflush(debugFile_);
    }
 
    wTime_ += MPI_Wtime() - baseTime_;
 
    return;
}

//------------------------------------------------------------------------------
void BASE_SLE::setDebugOutput(char* path, char* name){
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
int BASE_SLE::iterateToSolve() {

    baseTime_ = MPI_Wtime();

 
    if (debugOutput_) {
        fprintf(debugFile_,"iterateToSolve\n");
        fflush(debugFile_);
    }

// now the matrix can do its internal gyrations in preparation for
// parallel matrix-vector products.

    if (debugOutput_) {
        fprintf(debugFile_,"   calling matrixLoadComplete\n");
        fflush(debugFile_);
    }
    
    matrixLoadComplete();

// now we will implement the boundary conditions. This can be done after
// the matrix's 'loadComplete' because we're not altering the structure,
// only the coefficient values.
//
    implementAllBCs();

    selectSolver(solverName_);
    selectPreconditioner(precondName_);

    if (debugOutput_) {
        fprintf(debugFile_,"in iterateToSolve, calling launchSolver...\n");
        fflush(debugFile_);
    }
 
    wTime_ += MPI_Wtime() - baseTime_;

    sTime_ = MPI_Wtime();

    int solveStatus = 0;
    launchSolver(&solveStatus);

    sTime_ = MPI_Wtime() - sTime_;

 
    if (debugOutput_) {
        fprintf(debugFile_,"... back from solver\n");
        fflush(debugFile_);
    }
 
    if ((localRank_ == masterRank_) && (outputLevel_ > 0)){
        if (solveStatus == 1) {
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

    if (debugOutput_) {
        fprintf(debugFile_,"leaving iterateToSolve\n");
        fflush(debugFile_);
    }

    return(solveStatus);
}

//------------------------------------------------------------------------------
void BASE_SLE::unpackSolution() {
//
//This function should be called after the iterative solver has returned,
//and we know that there is a solution in the underlying vector.
//This function then 'unpacks' the vector into our nodal data structures.
//
    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_, "entering unpackSolution, outputLevel: %d\n",
                outputLevel_);
        fflush(debugFile_);
    }

    int i, j, k, m;
    int numSolnParams, numMultParams;
    double *nodeSolnPtr, **elemSolnPtr, *multSolnPtr;

//  have a solution, so let's store it where it belongs, namely in the nodal
//  data structures, or in elemental DOF data structures, or the Lagrange 
//  multiplier storage

//  first, declare a shared node buffer to load up with stuff to exchange 
//  between processors.

    int indSize = 0, maxDOF, tmp;
    getMaxNodesAndDOF(tmp, maxDOF);
    SharedNodeBuffer *shBuff = new SharedNodeBuffer(indSize, maxDOF);

//  also declare a shared node buffer to assemble info on what we'll
//  be receiving

    SharedNodeBuffer *shRecv = new SharedNodeBuffer(0,0);

    for (i = 0; i < storeNumProcActNodes; i++) {
        gnod_LocalNodes[i].allocateSolnList();
        nodeSolnPtr = gnod_LocalNodes[i].pointerToSoln(numSolnParams);
        GlobalID thisNode = gnod_LocalNodes[i].getGlobalNodeID();

        int proc = sharedNodes_->isShared(thisNode);

        if (proc<0) {
        
//  the node is not shared, so store its solution values as normal

            if (debugOutput_) {
                fprintf(debugFile_,"--- node %d not shared.\n",thisNode);
                fflush(debugFile_);
            }

            for (j = 0; j < numSolnParams; j++) {
                k = localStartRow_ + gnod_LocalNodes[i].getLocalEqnID();
                nodeSolnPtr[j] = accessSolnVector(k + j);
            }
        }
        else if (proc == localRank_) {

//  the node is shared, and we own it so we need to store its solution
//  values AND send them to the sharing processors

            if (debugOutput_) {
                fprintf(debugFile_,"--- node %d shared but owned.\n",thisNode);
                fflush(debugFile_);
            }

            for (j = 0; j < numSolnParams; j++) {
                k = localStartRow_ + gnod_LocalNodes[i].getLocalEqnID();
                nodeSolnPtr[j] = accessSolnVector(k + j);
            }

            NodeControlPacket shNode;
            int numShare;
            int *shProcIDs = sharedNodes_->pointerToProcIDs(thisNode, numShare);

//  loop over the procs that share this node

            for (j=0; j<numShare; j++) {
                if (shProcIDs[j] != localRank_) {

//  first send a node packet telling the node ID and numEqns

                    shNode.nodeID = thisNode;
                    shNode.numEqns = numSolnParams;

                    if (debugOutput_) {
                        fprintf(debugFile_,"---sending to proc %d, nodeID %d, numSolnParams %d\n",
                                shProcIDs[j], (int)thisNode, numSolnParams);
                        fflush(debugFile_);
                    }

                    shBuff->addNodeControlPacket(shNode, shProcIDs[j]);

                    shBuff->addCoeffs(nodeSolnPtr, numSolnParams, shProcIDs[j]);
                }
            }
        }
        else {
        
//  the node is shared and we DON'T own it, so we'll be getting solution 
//  values from its owning processor

            if (debugOutput_) {
                fprintf(debugFile_,"--- node %d shared, owned by %d.\n",
                        thisNode, proc);
                fflush(debugFile_);
            }

            NodeControlPacket shNode;
            shNode.nodeID = thisNode;
            shNode.numEqns = numSolnParams;

            shRecv->addNodeControlPacket(shNode, proc);

            if (debugOutput_) {
                fprintf(debugFile_,"---recving from proc %d, nodeID %d, numSolnParams %d\n",
                        proc, (int)thisNode, numSolnParams);
                fflush(debugFile_);
            }
        }
    }

    if (debugOutput_) {
        fprintf(debugFile_, "--- done looping over nodes\n");
        fflush(debugFile_);
    }

//  now launch the Irecvs.

    int numRecvProcs;
    int *recvProcs = shRecv->packetDestProcsPtr(numRecvProcs);

    if (debugOutput_) {
        fprintf(debugFile_, "--- numRecvProcs: %d\n",numRecvProcs);
        for(int ii=0; ii<numRecvProcs; ii++){
            fprintf(debugFile_, "--- recvProcs[%d]: %d\n",ii,recvProcs[ii]);
        }
        fflush(debugFile_);
    }

    MPI_Request *shReq = NULL, *shReqS = NULL;
    NodeControlPacket **recvPackets = NULL;
    double **solnPtr = NULL;

    if (numRecvProcs > 0){
        shReq = new MPI_Request[numRecvProcs];
        shReqS = new MPI_Request[numRecvProcs];
        recvPackets = new NodeControlPacket*[numRecvProcs];
        solnPtr = new double*[numRecvProcs];

        for(i=0; i<numRecvProcs; i++){
            recvPackets[i] = new
                        NodeControlPacket[shRecv->numPacketUnits(recvProcs[i])];

            MPI_Irecv(recvPackets[i], shRecv->numPacketUnits(recvProcs[i]),
                      MPI_NodePacket, recvProcs[i], packet_tag1+recvProcs[i],
                      FEI_COMM_WORLD, &shReq[i]);

            solnPtr[i] = new double[shRecv->numPacketUnits(recvProcs[i]) *
                                 maxDOF];

            MPI_Irecv(solnPtr[i], shRecv->numPacketUnits(recvProcs[i]) * maxDOF,
                      MPI_DOUBLE, recvProcs[i], coeff_tag+recvProcs[i],
                      FEI_COMM_WORLD, &shReqS[i]);
        }
    }

//  now get the shared node stuff back out of the shBuff object and send
//  it off to the other processors.

    int numDestProcs;
    int *destProcs = shBuff->packetDestProcsPtr(numDestProcs);

    if (debugOutput_) {
        fprintf(debugFile_, "--- numDestProcs: %d\n",numDestProcs);
        for(int ii=0; ii<numDestProcs; ii++){
            fprintf(debugFile_, "--- destProcs[%d]: %d\n",ii,destProcs[ii]);
        }
        fflush(debugFile_);
    }

    for(i=0; i<numDestProcs; i++){
        MPI_Send(shBuff->packetPtr(destProcs[i]),
                 shBuff->numPacketUnits(destProcs[i]), MPI_NodePacket,
                 destProcs[i], packet_tag1+localRank_, FEI_COMM_WORLD);
        MPI_Send(shBuff->coeffPtr(destProcs[i]),
                 shBuff->numCoeffUnits(destProcs[i]) * maxDOF,
                 MPI_DOUBLE, destProcs[i], coeff_tag+localRank_,
                 FEI_COMM_WORLD);
    }

    if (debugOutput_) {
        fprintf(debugFile_," --- done launching shared sends and recvs\n");
        fflush(debugFile_);
    }

//  now, we need to finish the Irecvs and store the solution values for
//  nodes that we share but don't own

    int index;
    MPI_Status status;

    for (i=0; i<numRecvProcs; i++) {
        MPI_Waitany(numRecvProcs, shReq, &index, &status);

        MPI_Wait(&shReqS[index], &status);

        int offset = 0;
        for(j=0; j<shRecv->numPacketUnits(recvProcs[index]); j++){

            int iLocal = GlobalToLocalNode(recvPackets[index][j].nodeID);
            double *soln = gnod_LocalNodes[iLocal].pointerToSoln(numSolnParams);
            assert(numSolnParams == recvPackets[index][j].numEqns);

            for(int ii=0; ii<numSolnParams; ii++){
                soln[ii] = solnPtr[index][offset+ii];
            }

            offset += maxDOF;
        }
        delete [] solnPtr[index];
        delete [] recvPackets[index];
    }

    if (debugOutput_) {
        fprintf(debugFile_," --- done catching shared sends\n");
        fflush(debugFile_);
    }

    if (numRecvProcs > 0){
        delete [] solnPtr;
        delete [] shReq;
        delete [] shReqS;
        delete [] recvPackets;
    }
    delete shRecv;
    delete shBuff;


//  get the element DOF and put into the returned solution vector - note that
//  this is a local operation by definition, as elements aren't shared or
//  external

    int ne, nedof, myOffset, elemOffset, countElems;
    int *myElemDOFPtr;
    for (i = 0; i < storeNumElemBlocks; i++) {
        int numElementDOF = blockRoster[i].getNumElemDOF();
        if (numElementDOF > 0) {
            myElemDOFPtr = blockRoster[i].pointerToLocalEqnElemDOF(countElems);
            assert (countElems == blockRoster[i].getNumElemTotal());
            elemSolnPtr = blockRoster[i].pointerToElemSoln(ne, nedof);
            assert (countElems == ne);
            assert (numElementDOF == nedof);
            for (j = 0; j < countElems; j++) {
                elemOffset = myElemDOFPtr[j];
                for (m = 0; m < numElementDOF; m++) {
                    myOffset = elemOffset + m + localStartRow_;
                    assert (myOffset < storeNumProcEqns);
                    elemSolnPtr[j][m] = accessSolnVector(myOffset);
                }
            }
        }
    }

//  get the Lagrange multiplier constraint solution parameter data

    int nmp;
    for (i = 0; i < storeNumCRMultRecords; i++) {
        numMultParams = ceqn_MultConstraints[i].getNumMultCRs();
        ceqn_MultConstraints[i].allocateMultipliers(numMultParams);
        multSolnPtr = ceqn_MultConstraints[i].pointerToMultipliers(nmp);
        assert (nmp == numMultParams);
        for (j = 0; j < numMultParams; j++) {
            k = ceqn_MultConstraints[i].getLocalEqnID() + localStartRow_;
            multSolnPtr[j] = accessSolnVector(k + j);
        }
    }

//  write out all the nodal data to the screen, including the solution params
//  then write out the multiplier solution data, too...

 
    if (outputLevel_ > 1) {
        for (i = 0; i < storeNumProcActNodes; i++) {
            cout << "node " << i << "\n";
            gnod_LocalNodes[i].dumpToScreen();
        }
        cout << "\n\n";
    
        for (i = 0; i < storeNumCRMultRecords; i++) {
            cout << "constraint " << i << "\n";
            ceqn_MultConstraints[i].dumpToScreen();
        }
    }
          
    solveCounter_++;

    if (debugOutput_) {
        fprintf(debugFile_,"leaving unpackSolution\n");
        fflush(debugFile_);
    }
 
    wTime_ = MPI_Wtime() - baseTime_;
 
}
             
//------------------------------------------------------------------------------
int BASE_SLE::getNodalScatterIndices(GlobalID* nodeIDs, int numNodes,
                                     int* scatterIndices) {
//
// This function takes as input the array nodeIDs, which contains numNodes
// global node IDs, and returns equation indices in the array scatterIndices,
// which is assumed to be allocated by the calling code, and is of length
// getNumEqnsPerElement(<appropriate-block-number>).
//
    int offset = 0;
    for(int i=0; i<numNodes; i++){
        int iLocal = GlobalToLocalNode(nodeIDs[i]);
        int proc = gnod_LocalNodes[iLocal].ownerProc();

        if ((proc<0) || (proc == localRank_)) {
            //either this node is not shared, or
            //it is shared but we own it.

            if (iLocal < 0) {
                cout << "ERROR in getNodalScatterIndices: nodeID " <<
                        (int)nodeIDs[i] << " not found " << endl;
                abort();
            }

            int nodeDOF = gnod_LocalNodes[iLocal].getNumNodalDOF();
            int equation = gnod_LocalNodes[iLocal].getLocalEqnID() +
                                  localStartRow_;

            for(int m=0; m<nodeDOF; m++){
                scatterIndices[offset++] = equation + m;
            }
        }
        else {
            //this is a shared node that we don't own. so we'll have to
            //look up its info in the sharedNodes_ record.
            int nodeDOF = sharedNodes_->numEquations(nodeIDs[i]);
            int equation = sharedNodes_->equationNumber(nodeIDs[i]);

            for(int m=0; m<nodeDOF; m++){
                scatterIndices[offset++] = equation + m;
            }
        }
    }

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::getBlockNodeSolution(GlobalID elemBlockID,  
                                   GlobalID *nodeIDList, 
                                   int &lenNodeIDList, 
                                   int *offset,  
                                   double *results) {
        
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getBlockNodeSolution\n");
        fflush(debugFile_);
    }

    int i, j, k;
    GlobalID blkID;
    int actNodeListSize, numBlocks, numSolnParams;
    int checkNodes = 0;
    int checkEqns = 0;
    int myOffset = 0;
    double *mySolnParams;
   
    int bIndex = search_ID_index(elemBlockID,&GID_blockIDList[0],
                               GID_blockIDList.size());
    assert(bIndex >= 0);
    actNodeListSize = blockRoster[bIndex].getNumActiveNodes();
    assert (actNodeListSize > 0);   // may want to return in this case...
    lenNodeIDList = actNodeListSize;
   
//  traverse the node list, checking for nodes associated with this block
//  when an associated node is found, add it to the list...

    for (i = 0; i < storeNumProcActNodes; i++) {
        int *blockListPtr = gnod_LocalNodes[i].pointerToBlockList(numBlocks);
        for (j = 0; j < numBlocks; j++) {
            blkID = blockListPtr[j];
           if (blkID == elemBlockID) {      // found a match, so get the data...
               mySolnParams = gnod_LocalNodes[i].pointerToSoln(numSolnParams);
               for (k = 0; k < numSolnParams; k++) {
                   results[myOffset + k] = mySolnParams[k];
                    checkEqns++;
               }
               nodeIDList[checkNodes] = gnod_LocalNodes[i].getGlobalNodeID();
               offset[checkNodes] = myOffset;
                checkNodes++;
                myOffset += numSolnParams;
            }
        }
    }

//  perform some verification checks to insure that data isn't corrupted

    int foundEqns = blockRoster[bIndex].getNumActiveEqns();
    assert (checkEqns == foundEqns);
    assert (checkNodes == actNodeListSize);

    return(0);
}

//------------------------------------------------------------------------------
//
int BASE_SLE::getBlockFieldNodeSolution(GlobalID elemBlockID,
                                        int fieldID,
                                        GlobalID *nodeIDList, 
                                        int& lenNodeIDList, 
                                        int *offset,
                                        double *results) {
        
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getBlockFieldNodeSolution\n");
        fflush(debugFile_);
    }

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
int BASE_SLE::putBlockNodeSolution(GlobalID elemBlockID,
                                   const GlobalID *nodeIDList, 
                                   int lenNodeIDList, 
                                   const int *offset,
                                   const double *estimates) {
        
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: putBlockNodeSolution\n");
    }

    int i, j, k, iLocal;
    int actNodeListSize, numSolnParams;
    GlobalID myActiveNode;
    int myOffset, myLocalOffset;

//  perform some error checking on the passed parameters to insure that
//  the FE client's view of the data agrees with our internal ideas...
   
    int bIndex = search_ID_index(elemBlockID,&GID_blockIDList[0],
                                 GID_blockIDList.size());
    assert(bIndex >= 0);
    actNodeListSize = blockRoster[bIndex].getNumActiveNodes();
    assert (lenNodeIDList == actNodeListSize);

//  translate the FE client's nodal-based lookups into an equation-based
//  approach in order to fill the appropriate entries in the solution vector

    for (i = 0; i < lenNodeIDList; i++) {
        myActiveNode = nodeIDList[i];
        iLocal = GlobalToLocalNode(myActiveNode);

//  if the node is locally-owned, then we can set its initial value, else
//  it's shared (in which case iLocal < 0), in which case we have no business
//  trying to set its value, as that ought to get done on the processor
//  owning the node.  Note that this assumes that the FE client will pass
//  initial estimates for each copy of any shared nodes so that the one
//  arising on the owning processor will "take" the initial estimate

        if (iLocal >= 0) {
            numSolnParams = gnod_LocalNodes[iLocal].getNumNodalDOF();
            myLocalOffset = gnod_LocalNodes[iLocal].getLocalEqnID();
            myOffset = offset[i];
            for (j = 0; j < numSolnParams; j++) {
                k = myLocalOffset + j + localStartRow_;
                putIntoSolnVector(1, &k, &(estimates[myOffset+j]) );
            }
        }
    }

//  we may eventually want to incorporate the new getNodeEqnNumber() method
//  in the logic above, to localize the lookups more consistently

    return(0);
}

//------------------------------------------------------------------------------
int BASE_SLE::putBlockFieldNodeSolution(GlobalID elemBlockID, 
                                        int fieldID, 
                                        const GlobalID *nodeIDList, 
                                        int lenNodeIDList, 
                                        const int *offset,
                                        const double *estimates) {
        
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: putBlockFieldNodeSolution\n");
        fflush(debugFile_);
    }

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
//
int BASE_SLE::getBlockElemSolution(GlobalID elemBlockID,  
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

    int i, j, k;
    int ne, nedof, eqnCount, elemCount;
    int numElements, numElementDOF;
    GlobalID *elemIDPtr;
    double **elemSolnPtr;

//  initialize
   
    i = search_ID_index(elemBlockID,&GID_blockIDList[0], GID_blockIDList.size());
    assert(i >= 0);
    
//  figure out the length of the various lists for this block

    numElements = blockRoster[i].getNumElemTotal();
    numElementDOF = blockRoster[i].getNumElemDOF();
    
//  go get the element solution data for this block

    eqnCount = 0;
    elemCount = 0;
    elemIDPtr = blockRoster[i].pointerToElemIDs(ne);
    assert (ne == numElements);
    elemSolnPtr = blockRoster[i].pointerToElemSoln(ne, nedof);
    assert (ne == numElements);
    assert (nedof == numElementDOF);
    
    for (j = 0; j < numElements; j++) {
        elemIDList[j] = elemIDPtr[j];
        offset[j] = eqnCount;
        for (k = 0; k < numElementDOF; k++) {
            results[eqnCount+k] = elemSolnPtr[j][k];
        }
        eqnCount += numElementDOF;
        ++elemCount;
    }

    lenElemIDList = elemCount;
    assert (lenElemIDList == numElements);
    numElemDOF = numElementDOF;
    
    return(0);
} 
      

//------------------------------------------------------------------------------
//
int BASE_SLE::putBlockElemSolution(GlobalID elemBlockID,
                                   const GlobalID *elemIDList, 
                                   int lenElemIDList, 
                                   const int *offset, 
                                   const double *estimates, 
                                   int numElemDOF) {
        
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: putElemBlockSolution\n");
        fflush(debugFile_);
    }

    if (numElemDOF); //prevents "declared and never referenced" warning.

    int i, k, m;
    int ne, elemCount, numElements, nedof, numElementDOF, myOffset;
    GlobalID *elemIDPtr;
    double **elemSolnPtr;

//  find the element block data first...
   
    i = search_ID_index(elemBlockID,&GID_blockIDList[0], GID_blockIDList.size());
    
//  next, calculate the table sizes for this block's element solution data

    assert(i >= 0);
    numElements = getNumBlockElements(elemBlockID);
    numElementDOF = blockRoster[i].getNumElemDOF();

//  no point in wasting time if there are no elemental solution parameters

    if (numElementDOF <= 0) {
        return(-1);
    }
    
//  finally, loop over all the rows and columns of the element block table,
//  while setting the element solution parameters to the passed values.

    elemCount = 0;
    elemIDPtr = blockRoster[i].pointerToElemIDs(ne);
    assert (ne == numElements);
    elemSolnPtr = blockRoster[i].pointerToElemSoln(ne, nedof);
    assert (ne == numElements);
    assert (nedof == numElementDOF);

    for (k = 0; k < numElements; k++) {

//  check to insure that the elements are in the correct order!

            assert (elemIDList[elemCount] == elemIDPtr[k]);
            myOffset = offset[elemCount];
            for (m = 0; m < numElementDOF; m++) {
                elemSolnPtr[k][m] = estimates[myOffset+m];
            }
            ++elemCount;
    }

//  perform some obvious checks to guard against major blunders

    assert (nedof == numElementDOF);
    assert (elemCount == lenElemIDList);

    return(0);
}


//------------------------------------------------------------------------------
int BASE_SLE::getCRMultSizes(int& numCRMultIDs, int& lenResults) {
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
int BASE_SLE::getCRMultSolution(int& numCRMultSets, 
                                int *CRMultIDs,  
                                int *offset, 
                                double *results) {
        
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getCRMultSolution\n");
        fflush(debugFile_);
    }

    int i, j, multLength, CRMultOffset,resultsOffset;
    double *CRMultPtr;

    numCRMultSets = storeNumCRMultRecords;

    CRMultOffset = 0;
    resultsOffset = 0;

    for (i=0; i<storeNumCRMultRecords; i++) {
        CRMultIDs[CRMultOffset] = ceqn_MultConstraints[i].getCRMultID();
        offset[CRMultOffset] = resultsOffset;

        CRMultPtr = ceqn_MultConstraints[i].pointerToMultipliers(multLength);
        for (j=0; j<multLength; j++) {
            results[resultsOffset] = CRMultPtr[j];
            resultsOffset++;
        }
        CRMultOffset++;
    }
    
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: leaving getCRMultSolution\n");
        fflush(debugFile_);
    }

    return(0);
} 

  
//------------------------------------------------------------------------------
int BASE_SLE::getCRMultParam(int CRMultID, 
                             int numMultCRs,
                             double *multValues) {

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getCRMultParam\n");
        fflush(debugFile_);
    }

    int j, nmp;      
    int i = CRMultID;   // NOTE: we're assuming that CRMultID is the index!
   
    int numMultParams = ceqn_MultConstraints[i].getNumMultCRs();
    double *myValues = ceqn_MultConstraints[i].pointerToMultipliers(nmp);
    for (j = 0; j < numMultParams; j++) {
        multValues[j] = myValues[j];
    }

//  the usual paranoid + pathological debugging stuff follows...
   
    assert (numMultParams == numMultCRs);
    assert (nmp == numMultParams);

    return(0);
}


//------------------------------------------------------------------------------
//
//  this method is just the inverse of getCRMultParam(), so...

int BASE_SLE::putCRMultParam(int CRMultID, 
                             int numMultCRs,
                             const double *multEstimates) {
        
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: putCRMultParam\n");
        fflush(debugFile_);
    }

    int j, nmp;

//  note that we're assuming (a) that CRMultID is the index (this is how it's
//  assigned elsewhere, so if this changes, then that changes must propagate
//  here, to something more akin to how we're handling inherited ID's like
//  the blockID passed parameters), and (b) that the Lagrange multipliers for
//  this constraint set have already been allocated...
  
    int i = CRMultID;
   
    int numMultParams = ceqn_MultConstraints[i].getNumMultCRs();
    double *myValues = ceqn_MultConstraints[i].pointerToMultipliers(nmp);
    for (j = 0; j < numMultParams; j++) {
        myValues[j] = multEstimates[j];
    }

//  the usual paranoid + pathological debugging stuff follows, in order to
//  check some of the underlying assumptions for calling this method...
   
    assert (numMultParams == numMultCRs);
    assert (nmp == numMultParams);

    return(0);
}


//-----------------------------------------------------------------------------
//  some utility functions to aid in using the "put" functions for passing
//  an initial guess to the solver
//-----------------------------------------------------------------------------

//------------------------------------------------------------------------------
//
//  return the list of element IDs for a given block... the length parameter
//  lenElemIDList should be used to check memory allocation for the calling
//  method, as the calling method should have gotten a copy of this param 
//  from a call to getNumBlockElements before allocating memory for elemIDList
//
int BASE_SLE::getBlockElemIDList(GlobalID elemBlockID,
                                 GlobalID *elemIDList, 
                                 int& lenElemIDList) {
        
 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getBlockElemIDList\n");
        fflush(debugFile_);
    }

    int i, k;
    int ne, elemCount, numElements;
    GlobalID *elemIDPtr;

//  find the element block index first...
   
    i = search_ID_index(elemBlockID, &GID_blockIDList[0],
                        GID_blockIDList.size());
    assert(i >= 0);
    
//  next, calculate the table sizes for this block's element solution data

    numElements = getNumBlockElements(elemBlockID);
    elemIDPtr = blockRoster[i].pointerToElemIDs(ne);
    assert (ne == numElements);

//  now just copy the element ID list...

    elemCount = 0;
    for (k = 0; k < numElements; k++) {
        elemIDList[elemCount] = elemIDPtr[k];
        ++elemCount;
    }
    lenElemIDList = elemCount;

    return(0);
}

//------------------------------------------------------------------------------
//
//  similar comments as for getBlockElemIDList(), except for returning the
//  active node list
//
int BASE_SLE::getBlockNodeIDList(GlobalID elemBlockID,
                                 GlobalID *nodeIDList, 
                                 int& lenNodeIDList) {

 
    if (debugOutput_) {
        fprintf(debugFile_,"trace: getBlockNodeIDList\n");
        fflush(debugFile_);
    }

    int i, j;
    GlobalID blkID;
    int actNodeListSize, numBlocks;
    int checkNodes = 0;
    
    int bIndex = search_ID_index(elemBlockID,&GID_blockIDList[0],
                               GID_blockIDList.size());
    assert(bIndex >= 0);
    actNodeListSize = blockRoster[bIndex].getNumActiveNodes();
    assert (actNodeListSize > 0);   // may want to return in this case...
    lenNodeIDList = actNodeListSize;

//  traverse the node list, checking for nodes associated with this block
//  when an associated node is found, add it to the list...

    for (i = 0; i < storeNumProcActNodes; i++) {
        int *blockListPtr = gnod_LocalNodes[i].pointerToBlockList(numBlocks);
        for (j = 0; j < numBlocks; j++) {
            blkID = blockListPtr[j];
           if (blkID == elemBlockID) {      // found a match, so get the node
               nodeIDList[checkNodes] = gnod_LocalNodes[i].getGlobalNodeID();
                checkNodes++;
            }
        }
    }
    
    return(0);
}


//------------------------------------------------------------------------------
int BASE_SLE::doEndInitElemData() {
//
// ***************************************************************
// *************BASE_SLE private function ************************
// ***************************************************************
//  tasks: do the work required upon completion of the element block
//         initialization stage...
//
//         construct active node lists for this processor and
//         for each block.
//
//         allocate memory for globalNodeBank data structures.
//
//  because the endInitElemBlock() function no longer indicates the
//  end of ALL element initialization data (as it did in earlier
//  versions of the interface), we can't perform these element-based
//  calculations in the endInitElemBlock() routine any more.  Hence,
//  they go here now...
//
//

    int i, j, k;

    // ought to do some initial consistency checking here (someday...)

    // construct the active node list for this processor

    int bIndex, bLimit;
    int elemLimit;
    int numNodes, myLocalNode, numNodeDOF, myTest;
    GlobalID myNode;
    int ne, nn;

    // increment a variable to indicate that this function has been called
 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d, incrementing doneEndInitElemData\n",
            localRank_);
    }
    doneEndInitElemData++;

    // loop over all the elements, collecting associated node numbers (this
    // is an obvious candidate for a separate function...)

    bLimit = storeNumElemBlocks;
    for (bIndex = 0; bIndex < bLimit; bIndex++) {
        numNodes = blockRoster[bIndex].getNumNodesPerElement();
        elemLimit = blockRoster[bIndex].getNumElemTotal();
       int *numNodeDOFPtr = blockRoster[bIndex].pointerToNumNodalDOF(myTest);
       assert (myTest == numNodes);
        GlobalID **connTable = blockRoster[bIndex].pointerToElemConn(ne, nn);
        assert (ne == elemLimit);
        assert (nn == numNodes);
        
        for (k = 0; k < elemLimit; k++) {
            for (i = 0; i < numNodes; i++) {
                myNode = connTable[k][i];
                numNodeDOF = numNodeDOFPtr[i];
                SetActive(myNode, numNodeDOF);
            }
        }
    }

    // determine the total number of active nodes on this processor, and
    // create the active node list

    storeNumProcActNodes = getNumActNodes();
    assert (storeNumProcActNodes > 0);
    gnod_LocalNodes = new NodeRecord[storeNumProcActNodes];

    int testNumProcActNodes = getActNodeList(gnod_LocalNodes);
    assert (testNumProcActNodes == storeNumProcActNodes);

    // determine the number of nodal equations arising from the active nodes

    storeNumProcActEqns = getNumActNodalEqns();

    // ----construct the active node list for each block-----

    // this task is just like the "construct active node list for this
    // processor" task above, but this time with consideration for activity
    // defined on a block-by-block basis... hence any similarities are not a
    // coincidence!
    
    // first, construct lists of blocks for each node (nodeBlockLists[i] is the
    // list of all blocks associated with node i)... this approach is taken
    // because it -seems- easier to handle insertion and sorts when there
    // aren't many entries in a list, and each node is generally associated
    // with only one (or sometimes two, three, or four...) block, hence the
    // insertion/sorting costs should be negligible here, compared to
    // constructing each active node list directly...

    GlobalIDArray *nodeBlockLists = new GlobalIDArray[storeNumProcActNodes];

    // create vector over active processor nodes of each associated block

    bLimit = storeNumElemBlocks;
    for (bIndex = 0; bIndex < bLimit; bIndex++) {
        GlobalID thisBlockID = GID_blockIDList[bIndex];
        numNodes = blockRoster[bIndex].getNumNodesPerElement();
        elemLimit = blockRoster[bIndex].getNumElemTotal();
        GlobalID **connTable = blockRoster[bIndex].pointerToElemConn(ne, nn);
        assert (ne == elemLimit);
        assert (nn == numNodes);
        
        for (k = 0; k < elemLimit; k++) {
            for (i = 0; i < numNodes; i++) {
                myNode = connTable[k][i];
                myLocalNode = GlobalToLocalNode(myNode);
                int found = -1;
                if (nodeBlockLists[myLocalNode].size()>0)
                    found = find_ID_index(thisBlockID,
                                          &((nodeBlockLists[myLocalNode])[0]),
                                          nodeBlockLists[myLocalNode].size());
                if (found<0) {
                    GID_insert_orderedID(thisBlockID,
                                        &(nodeBlockLists[myLocalNode]));
                }
            }
        }
    }
    
    // store list lengths in each block data structure (here, increment the
    // block's internal count of the number of nodes associated with the block
    // structure) also, create private vectors for each node (list of
    // associated blocks), and fill these private nodal vectors with data

    int *blockListPtr;
    int bTest, numNodEqns;

    for (i = 0; i < storeNumProcActNodes; i++) {
        int numAssocBlocks = nodeBlockLists[i].size();
        gnod_LocalNodes[i].allocateBlockList(numAssocBlocks);
        numNodEqns = gnod_LocalNodes[i].getNumNodalDOF();
        blockListPtr = gnod_LocalNodes[i].pointerToBlockList(bTest);
        assert (bTest == numAssocBlocks);
        for (j = 0; j < numAssocBlocks; j++) {
            k = search_ID_index((nodeBlockLists[i])[j],&GID_blockIDList[0],
                              GID_blockIDList.size());
            blockRoster[k].incrementNumActiveNodes(1);
            blockRoster[k].incrementNumActiveEqns(numNodEqns);
           blockListPtr[j] = (nodeBlockLists[i])[j];
       }
    }

    // armed with these new data structures (namely, the number of nodes 
    // associated with each block, stored in the block objects, and the 
    // list of blocks associated with each node, stored in the node objects), 
    // the solution return task consists merely of allocating solution vectors
    // for each block (whose length is retrieved from the block objects), and
    // filling them with data by parsing the nodal objects in order to resolve
    // block return associations

    delete [] nodeBlockLists;

//  perform the computations required to create the field lists for each
//  active node

    buildNodalFieldLists();

    return(0);
}

//------------------------------------------------------------------------------
//
//  find the index into the array of local nodes for node globalNodeID
//
int BASE_SLE::GlobalToLocalNode(GlobalID globalNodeID) const {

    int iLocal = find_ID_index(globalNodeID,&GID_ActNodeList[0],
                               GID_ActNodeList.size());

    return(iLocal);
}


//------------------------------------------------------------------------------
//
//  return the number of nodes associated with elements of a given block ID
//
int BASE_SLE::getNumNodesPerElement(GlobalID blockID) const {

    assert (blockID >= 0);

    int bIndex = search_ID_index(blockID,&GID_blockIDList[0], 
                                 GID_blockIDList.size());
    assert(bIndex >= 0);

    return(blockRoster[bIndex].getNumNodesPerElement());
}
 
 
//------------------------------------------------------------------------------
//
//  return the number of eqns associated with elements of a given block ID
//
int BASE_SLE::getNumEqnsPerElement(GlobalID blockID) const {

    assert (blockID >= 0);

    int bIndex = search_ID_index(blockID,&GID_blockIDList[0],
                            GID_blockIDList.size());
    assert(bIndex >= 0);

    return(blockRoster[bIndex].getNumEqnsPerElement());
}


//------------------------------------------------------------------------------
//
//  return the number of solution parameters at a given node
//
int BASE_SLE::getNumSolnParams(GlobalID iGlobal) const {

    int iLocal, numSolnParams;

    iLocal = GlobalToLocalNode(iGlobal);
    numSolnParams = gnod_LocalNodes[iLocal].getNumNodalDOF();

    return(numSolnParams);
}
 
//------------------------------------------------------------------------------
void BASE_SLE::SetActive(GlobalID nodeNum, int nodeDOF) {
//
//  set the node with global ID nodeNum to be active, by flagging it with a
//  nonzero constant value... in this case, if we let the flag take on the value
//  of the number of equations per node, we can substantially simplify the
//  subsequent tasks of (a) determining the number of equations that arise from
//  the active nodes, and (b) setting up some globalNodeBank data structures
//

    // see if the node is already in the active list
    int found = -1;
    if (GID_ActNodeList.size()>0) 
        found = find_ID_index(nodeNum,&GID_ActNodeList[0],
                              GID_ActNodeList.size());

    // if we didn't find it, then insert it into the list

    if (found<0) {
        int index = GID_insert_orderedID(nodeNum, &GID_ActNodeList); 
        IA_NodDOFList->insert(index, nodeDOF);
        IA_localElems->insert(index, 1);
    }
    else {

        // if we did find it, check for differing nodal solution cardinalities,
        // and if found, print a message and use the larger value

       int numDOF = (*IA_NodDOFList)[found];
        (*IA_localElems)[found]++;
       if (nodeDOF != numDOF) {
           cout << "\n\n alert: different solution cardinalities found, node:"
               << (int)nodeNum << "\n";
           if (nodeDOF > numDOF) {
               (*IA_NodDOFList)[found] = nodeDOF;
               cout << "  used new solution cardinality data for node\n";
           }
           else {
               cout << "  used old solution cardinality data for node\n";
           }
       }
    }

    return;
}


//------------------------------------------------------------------------------
//
//  determine the number of active nodes
//
int BASE_SLE::getNumActNodes() const {

    int count = GID_ActNodeList.size();

    return(count);
}


//------------------------------------------------------------------------------
int BASE_SLE::getActNodeList(NodeRecord *actNodeIDs) {
//
//  create a list of all active node records on this processor, and update some
//  some appropriate nodal data fields in the process (this routine may need to
//  get broken up into separate pieces, eventually)
//
//  presently, GID_ActNodeList[] stores the active nodes in a sorted list, but
//  there are parallel arrays for storing nodal DOF and for storing active node
//  numbers in the order they appear in the problem
//

    int i, numDOF;
    int countEqns = 0;
    int iLimit = GID_ActNodeList.size();
    for (i = 0; i < iLimit; i++) {
       numDOF = (*IA_NodDOFList)[i];
        actNodeIDs[i].setGlobalNodeID(GID_ActNodeList[i]);
        actNodeIDs[i].setNumNodalDOF(numDOF);
        actNodeIDs[i].setLocalEqnID(countEqns);
        actNodeIDs[i].allocateProcList(0);
//        actNodeIDs[i].dumpToScreen();   //$kdm debug only
       countEqns += numDOF;
    }

    // update the verification counter for the number of nodal equations and do
    // some minimal checking for active node list length

    checkNumProcActEqns = countEqns;

    return(iLimit);
}


//------------------------------------------------------------------------------
//
//  determine the number of active equations arising from nodal contributions
//
int BASE_SLE::getNumActNodalEqns() const {

    int countEqns = 0;
    int iLimit = GID_ActNodeList.size();
    int proc;
    GlobalID thisID;

    for (int i = 0; i < iLimit; i++) {
        thisID = gnod_LocalNodes[i].getGlobalNodeID();

        proc = sharedNodes_->isShared(thisID);

        if ((proc<0) || (proc == localRank_)) {

            //either node is not shared, or it is shared and we own it

            gnod_LocalNodes[i].setLocalEqnID(countEqns);
            countEqns += gnod_LocalNodes[i].getNumNodalDOF();
        }
    }

    return(countEqns);
}


//------------------------------------------------------------------------------
//
//  return the number of stored element blocks
//
int BASE_SLE::getNumElemBlocks() const {

    return(storeNumElemBlocks);
}


 
//------------------------------------------------------------------------------
//
//  return the number of active nodes associated with a given element block ID
//
int BASE_SLE::getNumBlockActNodes(GlobalID blockID) const {

    assert (blockID >= 0);
    
    int bIndex = search_ID_index(blockID, &GID_blockIDList[0],
                               GID_blockIDList.size());   
    int numNodes = blockRoster[bIndex].getNumActiveNodes();

    return(numNodes);
}


//------------------------------------------------------------------------------
//
// return the number of active equations associated with a given element
// block ID
//
int BASE_SLE::getNumBlockActEqns(GlobalID blockID) const {

    assert (blockID >= 0);

    int bIndex = search_ID_index(blockID, 
                               &GID_blockIDList[0],GID_blockIDList.size());   
    int numEqns = blockRoster[bIndex].getNumActiveEqns();

    return(numEqns);
}

//------------------------------------------------------------------------------
//
//  return the number of elements associated with a given elem blockID
//
int BASE_SLE::getNumBlockElements(GlobalID blockID) const {

    assert (blockID >= 0);

    int i = search_ID_index(blockID, &GID_blockIDList[0],
                            GID_blockIDList.size());   
    assert(i >= 0);
    return(blockRoster[i].getNumElemTotal());
}


//------------------------------------------------------------------------------
//
//  return the number of elem equations associated with a given elem blockID
//
int BASE_SLE::getNumBlockElemEqns(GlobalID blockID) const {

    assert (blockID >= 0);

    int i = search_ID_index(blockID, &GID_blockIDList[0],
                            GID_blockIDList.size());   
    assert(i >= 0);
    int numElementDOF = blockRoster[i].getNumElemDOF();
    int numBlockElemEqns = numElementDOF*getNumBlockElements(blockID);

    return(numBlockElemEqns);
}

//------------------------------------------------------------------------------
void BASE_SLE::doSharedNodeInitComm() {
//
//  loop over all nodes, checking to see if they're shared.
//  Then afterwards, we'll be able to send a message to the owner of each shared
//  node telling them how many sets of scatter indices to expect from us.
//

    int i, k, m, kLimit;
    int proc;

    int numNodes = GID_ActNodeList.size();
    for(i=0; i<numNodes; i++){
        proc = sharedNodes_->isShared(GID_ActNodeList[i]);

        if ((proc>=0) && (proc != localRank_)) {

            // it is shared, but we don't own it. here we'll
            // set the number of elements it is associated
            // with on this processor

            sharedNodes_->localElems(GID_ActNodeList[i], (*IA_localElems)[i]);
        }
    }

    // so now, for each shared node that we don't own, send a packet telling
    // the owning processor how many elements it appears in here.
    // also, for each shared node that we do own, send a packet telling its
    // equation number and DOF to the processors that share it.

    NodeControlPacket shNodePacket;
    SharedNodeBuffer *sharedBuff1 = new SharedNodeBuffer(0,0);
    SharedNodeBuffer *sharedBuff2 = new SharedNodeBuffer(0,0);
 
    int source, myLocalNode, numNDOF;
    MPI_Status status;

    // get the list of local shared nodes.
    int numLocalShared;
    GlobalID *localShared = sharedNodes_->pointerToLocalShared(numLocalShared);

 
    if (debugOutput_) {
        fprintf(debugFile_,"numLocalShared: %d\n", numLocalShared);
        fflush(debugFile_);
    }

    // for each shared node that we own (localShared), we'll need to set its
    // equation number, and the number of DOF (numEquations), and send this
    // information to the other processors that share it.
    // we'll aggregate the packets to be sent into SharedNodeBuffer
    // objects, where they will be organized into lists according to the
    // destination processor.

    for (i=0; i<numLocalShared; i++) {

        myLocalNode = GlobalToLocalNode(localShared[i]);
        m = gnod_LocalNodes[myLocalNode].getLocalEqnID() + localStartRow_;
        numNDOF = gnod_LocalNodes[myLocalNode].getNumNodalDOF();

        // first, set the equation number and # DOF in the
        // shared node record
        sharedNodes_->equationNumber(localShared[i], m);
        sharedNodes_->numEquations(localShared[i], numNDOF);

        // we'll use shNodePacket to send stuff to the other procs,
        // along with the nodeID
        shNodePacket.nodeID = localShared[i];
        shNodePacket.eqnNumber = m;
        shNodePacket.numEqns = numNDOF;

        int* sharedProcIDPtr = sharedNodes_->pointerToProcIDs(localShared[i],
                                                              kLimit);

        // now, put it in the buffers
        for (k=0; k<kLimit; k++) {
            if (sharedProcIDPtr[k] != localRank_) {
                int dest = sharedProcIDPtr[k];
                sharedBuff1->addNodeControlPacket(shNodePacket, dest);
            }
        }
    }

    // get the list of remote shared nodes.
    int numRemoteShared;
    GlobalID *remoteShared =
        sharedNodes_->pointerToRemoteShared(numRemoteShared);

 
    if (debugOutput_) {
        fprintf(debugFile_,"numRemoteShared: %d\n", numRemoteShared);
        fflush(debugFile_);
    }

    // now for each remotely-owned shared node, do the same thing.

    for (i=0; i<numRemoteShared; i++) {
        proc = sharedNodes_->ownerProc(remoteShared[i]);

        // now put the numElems info in the buffers
        shNodePacket.nodeID = remoteShared[i];
        shNodePacket.numElems = sharedNodes_->localElems(remoteShared[i]);

        sharedBuff2->addNodeControlPacket(shNodePacket, proc);
    }

    //now get the shared node stuff back out of sharedBuff1 and launch
    //the Irecvs.

    int numDestProcs1;
    int *destProcs1 = sharedBuff1->packetDestProcsPtr(numDestProcs1);

 
    if (debugOutput_) {
        fprintf(debugFile_,"dsnic: numDestProcs1: %d\n",numDestProcs1);
    for(i=0; i<numDestProcs1; i++){
            fprintf(debugFile_,"dsnic: destProcs1[%d]: %d, numPacketUnits: %d\n",
                i,destProcs1[i],sharedBuff1->numPacketUnits(destProcs1[i]));
    }
        fflush(debugFile_);
    }

    NodeControlPacket **recvPackets1 = NULL;
    MPI_Request *requests1 = NULL;

    if (numDestProcs1 > 0) {
        requests1 = new MPI_Request[numDestProcs1];
        recvPackets1 = new NodeControlPacket*[numDestProcs1];
        for(i=0; i<numDestProcs1; i++){
            recvPackets1[i] = new NodeControlPacket[sharedBuff1->
                                                numPacketUnits(destProcs1[i])];

            MPI_Irecv(recvPackets1[i],
                      sharedBuff1->numPacketUnits(destProcs1[i]),
                      MPI_NodePacket, destProcs1[i], packet_tag2+destProcs1[i],
                      FEI_COMM_WORLD, &requests1[i]);
        }
    }

    //now get the other shared node info out of sharedBuff2 and launch
    //their Irecvs.

    int numDestProcs2;
    int *destProcs2 = sharedBuff2->packetDestProcsPtr(numDestProcs2);

 
    if (debugOutput_) {
        fprintf(debugFile_,"dsnic: numDestProcs2: %d\n",numDestProcs2);
    for(i=0; i<numDestProcs2; i++){
            fprintf(debugFile_,"dsnic: destProcs2[%d]: %d, numPacketUnits: %d\n",
                i,destProcs2[i],sharedBuff2->numPacketUnits(destProcs2[i]));
    }
        fflush(debugFile_);
    }

    NodeControlPacket **recvPackets2 = NULL;
    MPI_Request *requests2 = NULL;

    if (numDestProcs2 > 0) {
        requests2 = new MPI_Request[numDestProcs2];
        recvPackets2 = new NodeControlPacket*[numDestProcs2];
        for(i=0; i<numDestProcs2; i++){
            recvPackets2[i] = new NodeControlPacket[sharedBuff2->
                                                numPacketUnits(destProcs2[i])];

            MPI_Irecv(recvPackets2[i],
                      sharedBuff2->numPacketUnits(destProcs2[i]),
                      MPI_NodePacket, destProcs2[i], packet_tag1+destProcs2[i],
                      FEI_COMM_WORLD, &requests2[i]);
        }
    }

    MPI_Barrier(FEI_COMM_WORLD);

    for(i=0; i<numDestProcs1; i++){
        MPI_Send(sharedBuff1->packetPtr(destProcs1[i]),
                  sharedBuff1->numPacketUnits(destProcs1[i]),
                  MPI_NodePacket, destProcs1[i], packet_tag1+localRank_,
                  FEI_COMM_WORLD);
    }

    for(i=0; i<numDestProcs2; i++){
        MPI_Send(sharedBuff2->packetPtr(destProcs2[i]),
                  sharedBuff2->numPacketUnits(destProcs2[i]),
                  MPI_NodePacket, destProcs2[i], packet_tag2+localRank_,
                  FEI_COMM_WORLD);
    }

    // now, for the nodes we don't own, complete the recv's and
    // store the equation numbers for each one.

    int index, j;
    for (i=0; i<numDestProcs2; i++) {
        MPI_Waitany(numDestProcs2,requests2,&index,&status);

        source = status.MPI_SOURCE;
        for(j=0; j<sharedBuff2->numPacketUnits(source); j++){
            GlobalID thisID = recvPackets2[index][j].nodeID;
            int eqnNumber = recvPackets2[index][j].eqnNumber;
            int numEqns = recvPackets2[index][j].numEqns;

            sharedNodes_->equationNumber(thisID,eqnNumber);
            sharedNodes_->numEquations(thisID,numEqns);
        }
    }

    // now, for the nodes we own, complete the other recv's
    // and store the numElems data for each one

    for (i=0; i<numDestProcs1; i++) {
        MPI_Waitany(numDestProcs1,requests1,&index,&status);

        source = status.MPI_SOURCE;
        for(j=0; j<sharedBuff1->numPacketUnits(source); j++){
            GlobalID thisID = recvPackets1[index][j].nodeID;
            int numElems = recvPackets1[index][j].numElems;

            sharedNodes_->remoteElems(thisID,source, numElems);
        }
    }

    for(i=0; i<numDestProcs1; i++){
        delete [] recvPackets1[i];
    }

    for(i=0; i<numDestProcs2; i++){
        delete [] recvPackets2[i];
    }

    if (numDestProcs1>0){
        delete [] recvPackets1;
        delete [] requests1;
    }
    delete sharedBuff1;

    if (numDestProcs2>0){
        delete [] recvPackets2;
        delete [] requests2;
    }
    delete sharedBuff2;

 
    if (debugOutput_) {
        fprintf(debugFile_,"leaving doSharedNodeInitComm\n");
        fflush(debugFile_);
    }

    return;
}

//------------------------------------------------------------------------------
int BASE_SLE::doCRInit() {
//
//  here we're going to put relevant information into the external node
//  object, and perform the communication that is necessary to exchange
//  the information that will be needed from externalNodes_ later on.
//
//  for each remotely owned node, we will need to know things like:
//     - how many constraint relations does the node appear in?
//     - which processor owns it?
//     - how many solution parameters (DOF) are there at that node?
//     - what global equation number is associated with the constraint
//       relation that this node appears in? (will need to send this to
//       the owning processor)
//     - what global equation number is associated with this node on the
//       processor that owns it?
//

    if (debugOutput_) {
        fprintf(debugFile_,"doCRInit: storeNumCRMultRecords: %d",
                storeNumCRMultRecords);
        fprintf(debugFile_,", storeNumCRPenRecords: %d",
                storeNumCRPenRecords);
        fprintf(debugFile_,", storeExtNodeSets: %d\n",
                storeExtNodeSets);
        fflush(debugFile_);
    }

    if (storeExtNodeSets == 0) return(0);

    int i, j, k;
    int ntRow, ntCol, numMultCRs, numPenCRs;
    int proc, eqnNum;
    GlobalID myGlobalID;
    GlobalID **CRNodeTablePtr;

    //first, handle the Lagrange multiplier constraint records...
    //we're going to go through the constraint records and determine which
    //nodes are remote. We will let the constraint know which nodes are remote,
    //and we'll let the external node object know the equation number associated
    //with the constraint relation that the node appears in.

    for (k = 0; k < storeNumCRMultRecords; k++) {
        numMultCRs = ceqn_MultConstraints[k].getNumMultCRs();
        int lenList = ceqn_MultConstraints[k].getLenCRNodeList();
        CRNodeTablePtr = ceqn_MultConstraints[k].pointerToCRNodeTable(ntRow, 
                                                                      ntCol);
        eqnNum = ceqn_MultConstraints[k].getLocalEqnID() + localStartRow_;
        for (i = 0; i < numMultCRs; i++) {
            for (j = 0; j < lenList; j++) {
                myGlobalID = CRNodeTablePtr[i][j];

                // is this node one of the external nodes?
                if (externalNodes_->isExternNode(myGlobalID) == true) {
                    // so we'll inform the constraint

                    ceqn_MultConstraints[k].remoteNode(myGlobalID);
                    externalNodes_->globalCREqn(myGlobalID, &eqnNum, 1);
                }

                // is this node a shared node that we don't own?
                proc = sharedNodes_->isShared(myGlobalID);
                if ((proc >= 0) && (proc != localRank_)) {
                    // This is a shared node that we don't own, so
                    // we'll inform the constraint that it's remote.
                    // (not sure yet what else needs done to handle this
                    // case.)***************
                    cout << "BASE_SLE::doCRInit: non-local shared node" << endl;

                    ceqn_MultConstraints[k].remoteNode(myGlobalID);
                }
            }
            eqnNum++;
        }
    }

    // next, process the penalty constraint records...
    
    int CRPenID = 0;
    if (storeNumCRPenRecords > 0)
        CRPenID = ceqn_PenConstraints[0].getCRPenID();
    for (k = 0; k < storeNumCRPenRecords; k++) {
        numPenCRs = ceqn_PenConstraints[k].getNumPenCRs();
        int lenList = ceqn_PenConstraints[k].getLenCRNodeList();
       CRNodeTablePtr = ceqn_PenConstraints[k].pointerToCRNodeTable(ntRow,
                                                                     ntCol);
        for (i = 0; i < numPenCRs; i++) {
            for (j = 0; j < lenList; j++) {
                myGlobalID = CRNodeTablePtr[i][j];

                // is this node one of the external nodes?
                if (externalNodes_->isExternNode(myGlobalID) == true) {
                    // so we'll inform the constraint. We will also inform
                    // the external node record that this is a penalty term,
                    // since some different processing is required for these
                    // nodes. Thirdly, we'll give the list of all nodes in
                    // this constraint to the external node record, because
                    // this list will need to be sent to one or more remote
                    // processors.

                    ceqn_PenConstraints[k].remoteNode(myGlobalID);

                    externalNodes_->penaltyTerm(myGlobalID);

                    externalNodes_->localPenNodeList(CRPenID, CRNodeTablePtr[i],
                                                lenList);
                }

                // is this node a shared node that we don't own?
                proc = sharedNodes_->isShared(myGlobalID);
                if ((proc >= 0) && (proc != localRank_)) {
                    // this is a shared node that we don't own, so
                    // we'll inform the constraint that it's remote
                    // (not sure yet what else needs done to handle this
                    // case.)***************
                    cout << "BASE_SLE::doCRInit: non-local shared node" << endl;

                    ceqn_PenConstraints[k].remoteNode(myGlobalID);
                    externalNodes_->penaltyTerm(myGlobalID);
                }
            }
            CRPenID++;
        }
    }

    //now lets get a list of all the external nodes, and explicitly check
    //to see which ones are local, and inform externalNodes_ of this.
    //this may not be necessary, and may be expensive, but for the short
    //term it is simple and easy.

    int numExtNodes;
    int iLocal;
    
    GlobalID *extNodes = externalNodes_->externNodeList(numExtNodes);
    for (i = 0; i < numExtNodes; i++){
        iLocal = GlobalToLocalNode(extNodes[i]);
        if (iLocal >= 0) {
            externalNodes_->ownerProc(extNodes[i], localRank_);

//  while we're in the neighborhood, this is a good time to cache the field
//  data into the external nodes container, as these field IDs and offsets
//  will be needed during the communications passes below

//-- begin new (and perhaps unnecessary? - we need to check!) code block
            int *myFieldIDs;
            int *myOffsets;
            int myNumFields;

            myFieldIDs = gnod_LocalNodes[iLocal].pointerToFieldIDList(myNumFields);
            myOffsets = gnod_LocalNodes[iLocal].pointerToFieldOffsetList(myNumFields);
            assert (myFieldIDs != NULL);
            assert (myOffsets != NULL);

            externalNodes_->setNumFields(extNodes[i], myNumFields);
            externalNodes_->setFieldIDList(extNodes[i], myFieldIDs, myNumFields);
            externalNodes_->setFieldOffsetList(extNodes[i], myOffsets, myNumFields);
//$            cout << "Init: numFields : " << myNumFields
//$                 << "      nodeID : " << extNodes[i]
//$                 << "      procID : " << localRank_ << endl << flush;
//-- end new (and perhaps unnecessary?) code block
        }
    }

    // now, the external nodes record can gather up some internal information
    // such as lists of processors to send and receive with, etc.

    MPI_Barrier(FEI_COMM_WORLD);
    externalNodes_->initComplete();

    // now: it's time to exchange some data between processors.
    // For our remote nodes, (the remote nodes in local constraint relations)
    // we'll need to send the numMultCRs and globalCREqn information TO the
    // owning processors and receive the DOF and globalEqn information FROM
    // the owning processors. This is also where we'll inform the external
    // node record about which processor owns the node.

    // For our local nodes, (the nodes we own which appear in constraint 
    // relations on other processors) we'll need to send the DOF and globalEqn
    // information and receive the numMultCRs and globalCREqn information.
    // i.e., this is the mirror of what gets done for the remote nodes.

    MPI_Status status;

    // now let's get the lists of stuff we'll need (to do the inter-processor
    // communication) from the external node records.

    //"send procs" are processors to which we send info about our local nodes.
    int numSendProcs;
    int* sendProcs = externalNodes_->sendProcListPtr(numSendProcs);

    //"recv procs" are processors from which we recv info about remote nodes.
    int numRecvProcs;
    int* recvProcs = externalNodes_->recvProcListPtr(numRecvProcs);

    int* lenLocalNodeIDs = NULL;
    GlobalID** localNodeIDsPtr = externalNodes_->
                                      localNodeIDsPtr(&lenLocalNodeIDs);

 
    if (debugOutput_) {
        fprintf(debugFile_,"%d numSendProcs: %d\n",localRank_,numSendProcs);
    for (i=0; i<numSendProcs; i++) {
            fprintf(debugFile_,"%d lenLocalNodeIDs[%d]: %d\n",localRank_,i,
                lenLocalNodeIDs[i]);
        for (j=0; j<lenLocalNodeIDs[i]; j++) {
                fprintf(debugFile_,"%d localNodeIDs[%d][%d]: %d\n",localRank_,i,j,
                    (int)localNodeIDsPtr[i][j]);
        }
    }
    }

    int* lenRemoteNodeIDs = NULL;
    GlobalID** remoteNodeIDsPtr = externalNodes_->
                                          remoteNodeIDsPtr(&lenRemoteNodeIDs);

 
    if (debugOutput_) {
        fprintf(debugFile_,"%d numRecvProcs: %d\n",localRank_,numRecvProcs);
    for (i=0; i<numRecvProcs; i++) {
            fprintf(debugFile_,"%d lenRemoteNodeIDs[%d]: %d\n",localRank_,i,
                lenRemoteNodeIDs[i]);
        for (j=0; j<lenRemoteNodeIDs[i]; j++) {
                fprintf(debugFile_,"%d remoteNodeIDs[%d][%d]: %d\n",localRank_,
                    i,j,(int)remoteNodeIDsPtr[i][j]);
        }
    }
    }

    int maxSendLength = 0;
    for (i=0; i<numSendProcs; i++) {
        if (lenLocalNodeIDs[i] > maxSendLength) 
            maxSendLength = lenLocalNodeIDs[i];
    }

    NodeControlPacket* remoteNodes = NULL;
    NodeControlPacket* localNodes = NULL;

 
    if (debugOutput_) {
        fprintf(debugFile_,"%d new'ing localNodes, length %d\n",localRank_,
            maxSendLength);
    }
    if (maxSendLength > 0) localNodes = new NodeControlPacket[maxSendLength];

    int maxRecvLength = 0;
    for(i=0; i<numRecvProcs; i++) {
        if (lenRemoteNodeIDs[i] > maxRecvLength) 
            maxRecvLength = lenRemoteNodeIDs[i];
    }
 
    if (debugOutput_) {
        fprintf(debugFile_,"%d new'ing remoteNodes, length %d\n",localRank_,
            maxRecvLength);
    }
    if (maxRecvLength > 0) remoteNodes = new NodeControlPacket[maxRecvLength];

//---- begin handshaking communications to determine scope of communications

//  first, let's send the node packet info for our local nodes.
//  do this for all local nodes, regardless of whether they are associated
//  with Lagrange or penalty constraints, or whatever...

    for (i = 0; i < numSendProcs; i++) {
    
//  first, load up the array of node packets to send to processor sendProcs[i]

        for (j = 0; j < lenLocalNodeIDs[i]; j++) {
            localNodes[j].nodeID = localNodeIDsPtr[i][j];
            localNodes[j].numEqns = getNumSolnParams(localNodeIDsPtr[i][j]);
            int nodeIndex = GlobalToLocalNode(localNodeIDsPtr[i][j]);
            localNodes[j].eqnNumber = gnod_LocalNodes[nodeIndex].getLocalEqnID() +
                localStartRow_;
            localNodes[j].numFields = gnod_LocalNodes[nodeIndex].getNumNodalFields();
//$            cout << "Send: numFields : " << localNodes[j].numFields
//$                 << "      nodeID : " << localNodes[j].nodeID
//$                 << "      procID : " << localRank_ << endl << flush;
        }

//  now, send the array of node packets

        MPI_Send(localNodes, lenLocalNodeIDs[i], MPI_NodePacket, sendProcs[i],
                 extSendPacketTag, FEI_COMM_WORLD);
    }

//  now let's receive the node packet info for our remote nodes.
//  (all of them, regardless of what kind of constraint they're in)

    for (i = 0; i < numRecvProcs; i++) {

        MPI_Recv(remoteNodes, lenRemoteNodeIDs[i], MPI_NodePacket, recvProcs[i],
                 extSendPacketTag, FEI_COMM_WORLD, &status);

//  check to make sure that we got the right number of node packets

        int numPackets;
        MPI_Get_count(&status, MPI_NodePacket, &numPackets);
        if (numPackets != lenRemoteNodeIDs[i]) {
            printf("%d BASE_SLE::doCRInit: ERROR, wrong number of node packets. expected %d, got %d\n",localRank_, lenRemoteNodeIDs[i], numPackets);
            exit(0);
        }

//  now store the info we received

        for (j = 0; j < lenRemoteNodeIDs[i]; j++) {
            externalNodes_->numSolnParams(remoteNodes[j].nodeID, 
                                          remoteNodes[j].numEqns);
            externalNodes_->globalEqn(remoteNodes[j].nodeID, 
                                      remoteNodes[j].eqnNumber);
            externalNodes_->ownerProc(remoteNodes[j].nodeID, recvProcs[i]);
            externalNodes_->setNumFields(remoteNodes[j].nodeID,     //$kdm v1.0 addition 
                                         remoteNodes[j].numFields); //$kdm v1.0 addition
//$            cout << "Recv: numFields : " << remoteNodes[j].numFields
//$                 << "      nodeID : " << remoteNodes[j].nodeID
//$                 << "      procID : " << localRank_ << endl << flush;
        }
    }

//---- end handshaking communications to determine scope of communications


//---- begin communications to handle v1.0 multiple-field data arrays

//  first, send the local data to all processors that need it...

    for (i = 0; i < numSendProcs; i++) {
        for (j = 0; j < lenLocalNodeIDs[i]; j++) {
            GlobalID myNodeID;
            int myNumFields;
            int *myFieldIDList;
            int *myOffsetList;

//  get the nodal field data

            myNumFields = localNodes[j].numFields;
            myNodeID = localNodes[j].nodeID;
//$            cout << "ListSend: numFields : " << myNumFields
//$                 << "      nodeID : " << myNodeID
//$                 << "      procID : " << localRank_ << endl << flush;

            myFieldIDList = externalNodes_->getFieldIDList(myNodeID, myNumFields);
            myOffsetList = externalNodes_->getFieldOffsetList(myNodeID, myNumFields);

//  ship the fieldID and field offset lists

            assert (myNumFields > 0);         // these asserts can go eventually
            assert (myFieldIDList != NULL);
            assert (myOffsetList != NULL);
            MPI_Send(myFieldIDList, myNumFields, MPI_INT, sendProcs[i],
                     id_tag, FEI_COMM_WORLD);
            MPI_Send(myOffsetList, myNumFields, MPI_INT, sendProcs[i],
                     field_tag, FEI_COMM_WORLD);
        }
    }

//  next, recv the remote data from all processors that sent stuff...

    for (i = 0; i < numRecvProcs; i++) {
        for (j = 0; j < lenRemoteNodeIDs[i]; j++) {

//  receive the fieldID and field offset lists

            GlobalID myNodeID;
            int myNumFields;
            int *myFieldIDList;
            int *myOffsetList;

            myNumFields = remoteNodes[j].numFields;
            myNodeID = remoteNodes[j].nodeID;
//$            cout << "ListRecv: numFields : " << myNumFields
//$                 << "      nodeID : " << myNodeID
//$                 << "      procID : " << localRank_ << endl << flush;

            myFieldIDList = new int [myNumFields];
            myOffsetList = new int [myNumFields];
            MPI_Recv(myFieldIDList, myNumFields, MPI_INT, recvProcs[i],
                     id_tag, FEI_COMM_WORLD, &status);
            MPI_Recv(myOffsetList, myNumFields, MPI_INT, recvProcs[i],
                     field_tag, FEI_COMM_WORLD, &status);

            externalNodes_->setNumFields(myNodeID, myNumFields);
            externalNodes_->setFieldIDList(myNodeID, myFieldIDList, myNumFields);
            externalNodes_->setFieldOffsetList(myNodeID, myOffsetList, myNumFields);

            delete [] myFieldIDList;
            delete [] myOffsetList;
        }
    }


//---- end communications to handle v1.0 multiple-field data arrays


//---- begin exchange of the numMultCRs and globalCREqn constraint data

    for (i = 0; i < numRecvProcs; i++) {

//  first, create and load up an array of node packets to send to
//  processor recvProcs[i]

        for (j = 0; j < lenRemoteNodeIDs[i]; j++) {
            GlobalID tempID = remoteNodeIDsPtr[i][j];
            remoteNodes[j].nodeID = tempID;
            remoteNodes[j].numMultCRs = externalNodes_->getNumMultCRs(tempID);
            remoteNodes[j].numPenCRs = externalNodes_->getNumPenCRs(tempID);
            remoteNodes[j].numFields = externalNodes_->getNumFields(tempID);   //$kdm v1.0
        }

//  now, send the array of node packets

        MPI_Send(remoteNodes, lenRemoteNodeIDs[i], MPI_NodePacket, recvProcs[i],
                 extRecvPacketTag, FEI_COMM_WORLD);

//  now we need to send, to the same processor, the list of globalCREqn's
//  for each of the nodes in the array of node packets we just sent

        for (j = 0; j < lenRemoteNodeIDs[i]; j++) {
            if (remoteNodes[j].numMultCRs > 0) {
                int numMCR;
                int* globalCREqn;
                globalCREqn = externalNodes_->globalCREqn(remoteNodeIDsPtr[i][j],
                                                          numMCR);

 
    if (debugOutput_) {
                    fprintf(debugFile_,"%d sending CREqns of length %d to %d\n",
                        localRank_, numMCR, recvProcs[i]);
    }

                MPI_Send(globalCREqn, numMCR, MPI_INT, recvProcs[i],
                         indices_tag, FEI_COMM_WORLD);
            }
        }
    }   //end of i<numRecvProcs loop
 
//  now, let's receive the numMultCRs and globalCREqn info for our
//  local nodes.

    for (i = 0; i < numSendProcs; i++) {

        MPI_Recv(localNodes, lenLocalNodeIDs[i], MPI_NodePacket, sendProcs[i],
                 extRecvPacketTag, FEI_COMM_WORLD, &status);

//  let's check to make sure that we got the right number of node packets

        int numPackets;
        MPI_Get_count(&status, MPI_NodePacket, &numPackets);
        if (numPackets != lenLocalNodeIDs[i]) {
            cout << localRank_ << " BASE_SLE::doCRInit: ERROR, wrong number of "
                 << "node packets. expected " << lenRemoteNodeIDs[i] << ", got "
                 << numPackets << endl;
            exit(0);
        }

//  now let's store the info we got, as well as recv'ing (and storing)
//  the CREqn numbers, and penNodeLists.

        for (j = 0; j < lenLocalNodeIDs[i]; j++) {
            if (localNodes[j].numMultCRs > 0) {

//  here, let's receive the list of globalCREqn's for this node

                int* CREqns =(int*)malloc(sizeof(int)*localNodes[j].numMultCRs);

 
    if (debugOutput_) {
        fprintf(debugFile_,"%d recv'ing CREqns of length %d from %d\n",
                localRank_, localNodes[j].numMultCRs, sendProcs[i]);
        fflush(debugFile_);
    }

                MPI_Recv(CREqns, localNodes[j].numMultCRs, MPI_INT,sendProcs[i],
                         indices_tag, FEI_COMM_WORLD, &status);

//  let's check to make sure that we got the right number of eqn numbers.

                int numCR;
                MPI_Get_count(&status, MPI_INT, &numCR);
                if (numCR != localNodes[j].numMultCRs) {
                    cout << localRank_ << " BASE_SLE::doCRInit: ERROR, wrong "
                         << "number of CREqns. expected "
                         << localNodes[j].numMultCRs
                         << ", got " << numCR << endl;
                    exit(0);
                }

//  ok, so let's store them

                externalNodes_->globalCREqn(localNodes[j].nodeID, CREqns,
                                            localNodes[j].numMultCRs);

                free(CREqns);
            }//end of else block for 'if (localNodes[j].numPenCRs > 0)'
        }
    }

//---- end exchange of the numMultCRs and globalCREqn constraint data


    if (maxSendLength > 0) delete [] localNodes;
    if (maxRecvLength > 0) delete [] remoteNodes;

    //call the function that fills in the eqnID lists and numDOF lists for
    //the penalty constraint tables in externalNodes_.
    populateExtNodeTables();

    //now we have to figure out how many remote penalty constraint relations
    //we need to know about. i.e., how many sets of lists of nodes, eqnIDs and
    //DOFs we'll be recieving.
    //we'll calculate this by filling an array of size numProcs_ with position
    //i containing the number of OUTGOING penCRs to processor i, and then,
    //using MPI_Allreduce, do a global sum and the result will contain the
    //number of incoming penCRs for this processor in position localRank_.

    int *incomingPenCRs = new int[numProcs_];
    int *outgoingPenCRs = new int[numProcs_];
    for(i=0; i<numProcs_; i++){
        outgoingPenCRs[i] = 0;
        if (i != localRank_) {
            outgoingPenCRs[i] = externalNodes_->numOutgoingPenCRs(i);
 
            if (debugOutput_) {
                fprintf(debugFile_,"%d: %d PenCRs going to proc %d\n",
                        localRank_, outgoingPenCRs[i], i);
            }
        }
    }

    MPI_Allreduce(outgoingPenCRs, incomingPenCRs, numProcs_, MPI_INT,
                  MPI_SUM, FEI_COMM_WORLD);

    int numIncoming = incomingPenCRs[localRank_];
 
    if (debugOutput_) {
        fprintf(debugFile_,"%d numIncoming PenCRs: %d\n",
                localRank_,numIncoming);
    }

    delete [] outgoingPenCRs;
    delete [] incomingPenCRs;

    //call the function in externalNodes_ which figures out where the
    //penalty stuff will be sent to.
    externalNodes_->calcPenProcTable();

    //so now we're gonna launch the sends for all of the outgoing penalty stuff.
    //first, get the penalty stuff out of the externalNodes_ record.

    int numLocPenCRs, *lenPenNodes;
    numLocPenCRs = externalNodes_->localNumPenCRs();
    GlobalID **penNodeTable = externalNodes_->localPenNodeTable(&lenPenNodes);
    int **penEqnIDTable = externalNodes_->localPenEqnIDTable(&lenPenNodes);
    int **penNumDOFTable = externalNodes_->localPenNumDOFTable(&lenPenNodes);
    int *penIDList = externalNodes_->localPenIDList(numLocPenCRs);
    int *lenPenProcTable;
    int **penProcTable = externalNodes_->localPenProcTable(&lenPenProcTable);

    int *len_and_id = new int[2];

    //now for each local penalty constraint, send a list of nodes, a list
    //of eqnID's and a list of DOF's to remote processors who have nodes that
    //appear in the constraint.
    for(i=0; i<numLocPenCRs; i++){
        for(j=0; j<lenPenProcTable[i]; j++){
            if (penProcTable[i][j] != localRank_){
                //first, send the length and penID for the soon-to-follow lists.
                len_and_id[0] = lenPenNodes[i];
                len_and_id[1] = penIDList[i];

                MPI_Send(len_and_id, 2, MPI_INT, penProcTable[i][j], 
                         length_tag, FEI_COMM_WORLD);
                MPI_Send(penNodeTable[i], lenPenNodes[i], MPI_GLOBALID,
                         penProcTable[i][j], indices_tag, FEI_COMM_WORLD);
                MPI_Send(penEqnIDTable[i], lenPenNodes[i], MPI_INT,
                         penProcTable[i][j], indices_tag, FEI_COMM_WORLD);

                MPI_Send(penNumDOFTable[i], lenPenNodes[i], MPI_INT,
                         penProcTable[i][j], indices_tag, FEI_COMM_WORLD);
            }
        }
    }
    
    //and now we're finally ready to catch the incoming penalty stuff.
    for(i=0; i<numIncoming; i++){
        int length, source, remPenID;
        MPI_Recv(len_and_id, 2, MPI_INT, MPI_ANY_SOURCE, length_tag,
                 FEI_COMM_WORLD, &status);

        source = status.MPI_SOURCE;
        length = len_and_id[0];
        remPenID = len_and_id[1];
        GlobalID *nodeIDs = new GlobalID[length];
        int *penStuff = new int[length];

        MPI_Recv(nodeIDs, length, MPI_GLOBALID, source, indices_tag,
                 FEI_COMM_WORLD, &status);

        externalNodes_->remotePenNodeList(remPenID, source, nodeIDs, length);

        MPI_Recv(penStuff, length, MPI_INT, source, indices_tag,
                 FEI_COMM_WORLD, &status);

        externalNodes_->remotePenEqnIDList(remPenID, source, penStuff, length);

        MPI_Recv(penStuff, length, MPI_INT, source, indices_tag,
                 FEI_COMM_WORLD, &status);

        externalNodes_->remotePenNumDOFList(remPenID, source, penStuff, length);
    }

    delete [] len_and_id;

    MPI_Barrier(FEI_COMM_WORLD);
 
    if (debugOutput_) {
        fprintf(debugFile_,"proc %d leaving doCRInit\n",localRank_);
        fflush(debugFile_);
    }
    return(0);
}

//------------------------------------------------------------------------------
void BASE_SLE::populateExtNodeTables(){
//
//  This function populates the penEqnID table and the penNumDOF table.
//
//  This is purely a utility function for doCRInit, must be called AFTER
//  the penNodes table has been populated.
//

    int i, j, *lenPenNodes;
    int numPenCRs = externalNodes_->localNumPenCRs();
    if (numPenCRs > 0) {
        GlobalID **penNodes = externalNodes_->localPenNodeTable(&lenPenNodes);
        int *penIDs = externalNodes_->localPenIDList(numPenCRs);
        int **penEqnIDs = new int*[numPenCRs];
        int **penNumDOFs = new int*[numPenCRs];
        for(i=0; i<numPenCRs; i++){
            penEqnIDs[i] = new int[lenPenNodes[i]];
            penNumDOFs[i] = new int[lenPenNodes[i]];
        }

        for(i=0; i<numPenCRs; i++){

            for(j=0; j<lenPenNodes[i]; j++){
                //for each node in the penalty constraint, we have to get
                //its equation number and its solution cardinality.

                penEqnIDs[i][j] = getNodeEqnNumber(penNodes[i][j]);
                penNumDOFs[i][j] = getNodeDOF(penNodes[i][j]);
            }

            //now we can stick this row of the tables into the external node
            //record, and then destroy our copy.
            externalNodes_->localPenEqnIDList(penIDs[i], penEqnIDs[i],
                                         lenPenNodes[i]);
            externalNodes_->localPenNumDOFList(penIDs[i], penNumDOFs[i],
                                         lenPenNodes[i]);
            delete [] penEqnIDs[i];
            delete [] penNumDOFs[i];
        }

        delete [] penEqnIDs;
        delete [] penNumDOFs;
    }

    return;
}

//------------------------------------------------------------------------------
//
//  This function takes a list of processors, and returns the one which
//  is designated as the "owner". Currently, using the rule that the
//  lowest numbered processor is the owner.
//
int BASE_SLE::ownerProc(int* procs, int numProcs) {

    int i, owner;

    if (numProcs <= 0) {
        cout << "BASE_SLE::ownerProc: ERROR: numProcs <= 0" << endl;
        exit(0);
    }

    owner = procs[0];
    for (i=1; i<numProcs; i++) {
        if (procs[i] < owner) owner = procs[i];
    }

    return(owner);
}


//------------------------------------------------------------------------------
//
//  return the index of the field associated with a given fieldID
//

int BASE_SLE::getFieldRosterIndex(int fieldID) {

    assert (fieldID >= 0);       //$kdm - is this really necessary?  

//  here, we're assuming that there are few fields at each node (i.e.,
//  typically, a VERY small integer, like "one"!), so there's no point
//  in doing anything more clever than just a simple linear search

    for (int i = 0; i < storeNumFields; i++) {
        if (fieldID == fieldRoster[i].getFieldID()) {
            return(i);
        }
    }
    return(-1);
}


//------------------------------------------------------------------------------
//
//  return the solution cardinality of the field associated with a given fieldID
//

int BASE_SLE::getFieldCardinality(int fieldID) {

    assert (fieldID >= 0);       //$kdm - is this really necessary?  

//  here, we're assuming that there are few fields at each node (i.e.,
//  typically, a VERY small integer, like "one"!), so there's no point
//  in doing anything more clever than just a simple linear search

    for (int i = 0; i < storeNumFields; i++) {
        if (fieldID == fieldRoster[i].getFieldID()) {
            return(fieldRoster[i].getNumFieldParams());
        }
    }
    return(-1);
}


//------------------------------------------------------------------------------
//
//  construct the nodal field lists for all active nodes
//

void BASE_SLE::buildNodalFieldLists() {

    IntArray *nodeFieldLists = new IntArray[storeNumProcActNodes];
    int myRows, myCols;
    int i, j, k, m;

//  process all the blocks, refinding all the active nodes and computing
//  the number of fields present at each active node

    int blockLimit = storeNumElemBlocks;
    for (i = 0; i < blockLimit; i++) {
        int elemLimit = blockRoster[i].getInitNumElemTotal();
        int nodeLimit = blockRoster[i].getNumNodesPerElement();

        GlobalID **elemConn = 
              blockRoster[i].pointerToElemConn(myRows, myCols);
        assert (myRows == elemLimit);
        assert (myCols == nodeLimit);
              
        int *listNumFields = NULL;
        int **fieldIDTable = 
              blockRoster[i].pointerToElemFieldIDs(myRows, listNumFields);
        assert (myRows == nodeLimit);

//  find the number of fields present at each node
      
        for (j = 0; j < elemLimit; j++) {
            for (k = 0; k < nodeLimit; k++) {
                GlobalID myNode = elemConn[j][k];
                int myLocalNode = GlobalToLocalNode(myNode);
                for (m = 0; m < listNumFields[k]; m++) {
                    int thisFieldID = fieldIDTable[k][m];
                    int found = -1;
                    if (nodeFieldLists[myLocalNode].size() > 0)
                        found = find_ID_index(thisFieldID,
                                          &((nodeFieldLists[myLocalNode])[0]),
                                          nodeFieldLists[myLocalNode].size());
                    if (found<0) {
                        IA_insert_ordered(thisFieldID,
                                            &(nodeFieldLists[myLocalNode]));
                    }
                } 
            }
        }
    }

//  allocate the field lists now that we know how many fields are found
//  at each node, and then fill them up with fieldIDs and field offsets

    for (i = 0; i < storeNumProcActNodes; i++) {
        int numFields = nodeFieldLists[i].size();
        gnod_LocalNodes[i].setNumNodalFields(numFields);
        gnod_LocalNodes[i].allocateFieldLists(numFields);
        int *myIDList = gnod_LocalNodes[i].pointerToFieldIDList(myRows);
        assert (myRows == numFields);
        int *myOffsetList = gnod_LocalNodes[i].pointerToFieldOffsetList(myRows);
        assert (myRows == numFields);
        
        int localOffset = 0;
        for (j = 0; j < numFields; j++) {
            myIDList[j] = (nodeFieldLists[i])[j];
            myOffsetList[j] = localOffset;
            localOffset += getFieldCardinality(myIDList[j]);
        }
        assert (localOffset == gnod_LocalNodes[i].getNumNodalDOF());
    }

    delete [] nodeFieldLists;

    return;
}


//------------------------------------------------------------------------------
//
//  this function finds the DOF (number-of-solution-parameters) for node
//  nodeID, regardless of whether nodeID is an external node, a shared node,
//  or a node in the active node list.
//
//  it returns -1 if the node isn't in any of those sets.
//
int BASE_SLE::getNodeDOF(GlobalID nodeID){

    int index = GlobalToLocalNode(nodeID);
    if (index >= 0) {
        return(gnod_LocalNodes[index].getNumNodalDOF());
    }
    
    if (sharedNodes_->isShared(nodeID) >= 0) {
        return(sharedNodes_->numEquations(nodeID));
    }
    
    if (externalNodes_->isExternNode(nodeID) == true) {
        return(externalNodes_->numSolnParams(nodeID));
    }
    
    return(-1);
}

//------------------------------------------------------------------------------
//
//  This function finds the equation number for node nodeID,
//  regardless of whether nodeID is an external node, a shared node,
//  or a node in the active node list.
//
//  Returns a global equation number. i.e., adds localStartRow_ if
//  necessary. This means that this function should not be called
//  before localStartRow_ has been calculated, which happens during
//  the early part of BASE_SLE::initComplete().
//  returns -1 if the node isn't in any of those sets.
//
int BASE_SLE::getNodeEqnNumber(GlobalID nodeID) {

    int index = GlobalToLocalNode(nodeID);
    if (index >= 0) {
//$kdm dbg cout << "A - node, eqnum : " << (int) nodeID << "   "
//$kdm dbg      << gnod_LocalNodes[index].getLocalEqnID()+localStartRow_ << endl;
        return(gnod_LocalNodes[index].getLocalEqnID()+localStartRow_);
    }

    if (sharedNodes_->isShared(nodeID) >= 0) {
//$kdm dbg cout << "B - node, eqnum : " << (int) nodeID << "   "
//$kdm dbg      << sharedNodes_->equationNumber(nodeID) << endl;
        return(sharedNodes_->equationNumber(nodeID));
    }

    if (externalNodes_->isExternNode(nodeID) == true) {
//$kdm dbg cout << "C - node, eqnum : " << (int) nodeID << "   " 
//$kdm dbg      << externalNodes_->globalEqn(nodeID) << endl;
        return(externalNodes_->globalEqn(nodeID));
    }

    return(-1);
}


//------------------------------------------------------------------------------
//
//  return the total number of DOF for an active node (i.e., a node that is
//  either local to this processor, or shared and owned by another processor)
//
//  NOTE: this routine is presumed to be used for ACTIVE nodes, as in those
//  utilized for element assembly operations.  At present, external nodes
//  aren't supported, so be forewarned...
//
int BASE_SLE::getGlobalActEqnDOF(GlobalID myNode) {

    int procNum = sharedNodes_->isShared(myNode);
    
//  it's locally owned (shared and owned, or not shared at all...)

    if ((procNum < 0) || (procNum == localRank_)) {
        int myLocalNode = GlobalToLocalNode(myNode);
        int myLocalNumDOF = gnod_LocalNodes[myLocalNode].getNumNodalDOF();
        return(myLocalNumDOF);
    }

//  it's shared but not owned locally, so we need to look elsewhere

    else {
        int theirRemoteNumDOF = sharedNodes_->numEquations(myNode);
        return(theirRemoteNumDOF);
    }
}


//------------------------------------------------------------------------------
int BASE_SLE::getGlobalActEqnNumber(GlobalID myNode, int nodeIndex) {
//
//  return the global equation number for an active node (i.e., a node that is
//  either local to this processor, or shared and owned by another processor)
//
//  NOTE: this routine is presumed to be used for ACTIVE nodes, as in those
//  utilized for element assembly operations.  At present, external nodes
//  aren't supported, so be forewarned...
//

    int procNum = gnod_LocalNodes[nodeIndex].ownerProc();
    
//  it's locally owned (shared and owned, or not shared at all...)

    if (procNum == localRank_) {
        int myLocalEqnID = gnod_LocalNodes[nodeIndex].getLocalEqnID();
        return(myLocalEqnID + localStartRow_);
    }

//  it's shared but not owned locally, so we need to look elsewhere

    else {
        int theirRemoteEqnID = sharedNodes_->equationNumber(myNode);
        return(theirRemoteEqnID);
    }
}


//------------------------------------------------------------------------------
void BASE_SLE::packSharedStiffnessAndLoad(const int* conn, int numNodes,
                                          int* localNodeIndices,
                                          int* scatterIndices, int numIndices,
                                          const double* const* stiffness,
                                          const double* load) {
//
// This function will pack the appropriate coefficients and indices into
// CommBuffer(s) to be sent to other processors later.
//
    (void)conn;

    int nodeRow = 0; // this will be the row corresponding to the current
                     // node, in the stiffness matrix
    for(int i = 0; i < numNodes; i++) {
        int iLocal = localNodeIndices[i];
        int proc = gnod_LocalNodes[iLocal].ownerProc();

        if ((proc>=0) && (proc != localRank_)) {

//  this is a shared node that we don't own, so we'll store some stuff in
//  CommBuffer object to be sent to other processors later.
//  We'll send the scatter indices and the coefficients.

            GlobalID thisNode = gnod_LocalNodes[iLocal].getGlobalNodeID();
            int eqnNumber = sharedNodes_->equationNumber(thisNode);
            int numEqns = sharedNodes_->numEquations(thisNode);

            for(int j=0; j<numEqns; j++) {
                shBuffLoadD_->addDoubles(eqnNumber+j,
                                         stiffness[nodeRow+j],
                                         numIndices, proc);
                shBuffLoadI_->addInts(eqnNumber+j,
                                      scatterIndices,
                                      numIndices, proc);

                if (debugOutput_) {
                    fprintf(debugFile_,"   packShared... currentRHS_: %d\n",
                            currentRHS_);
                    fflush(debugFile_);
                }

                shBuffRHSLoadD_[currentRHS_].addDoubles(eqnNumber+j,
                                            &load[nodeRow+j], 1, proc);
            }

            nodeRow += numEqns;
        }
        else {
            //..else it's a shared node that we DO own.

            nodeRow += gnod_LocalNodes[iLocal].getNumNodalDOF();
        }
    }
}

//------------------------------------------------------------------------------
int BASE_SLE::formElemScatterList(int blockIndex, const GlobalID* elemConn,
                                  int* localNodeIndices, 
                                  int *scatterIndices) {
//
//  utility function to convert an element's connectivity list into a list
//  of element scatter indices.  it is (of course) assumed that the list
//  of scatter indices was allocated elsewhere, and has the right length,
//  namely the sum of all the appropriate nodal solution cardinalities
//

//  first, get the terms arising from nodal contributions

    int numNodes, *nodalDOFPtr;
    nodalDOFPtr = blockRoster[blockIndex].pointerToNumNodalDOF(numNodes);
    int localCount = 0;
    for (int i = 0; i < numNodes; i++) {
        GlobalID myNode = elemConn[i];
        int iStart = getGlobalActEqnNumber(myNode, localNodeIndices[i]);
        int jLimit = nodalDOFPtr[i];
        for (int j = 0; j < jLimit; j++) {
            scatterIndices[localCount] = iStart + j;
            localCount++;
        }
    }

//  then, add any contributions arising from elemental DOF
//
//  temporarily removed for testing purposes, as the new blockRoster 
//  storage format for v1.0 necessitates rethinking the strict rules 
//  about how the elemental DOF need to be stored in the sparse matrix

    int numElementDOF = blockRoster[blockIndex].getNumElemDOF();
    assert (numElementDOF == 0);  //$kdm remove & fix when elemDOF added...

    return(localCount);
}

//------------------------------------------------------------------------------
void BASE_SLE::assembleStiffnessAndLoad(int numRows, int* scatterIndices, 
                                  const double* const* stiff,
                                  const double* load) {
//
//This function hands the element data off to the routine that finally
//sticks it into the matrix and RHS vector.
//
    for(int i = 0; i < numRows; i++) {
        int ib = scatterIndices[i];
        if ((localStartRow_ <= ib) && (ib <= localEndRow_)) {
            sumIntoRHSVector(1, &ib, &(load[i]) );
            sumIntoSystemMatrix(ib, numRows, &(stiff[i][0]),
                                &scatterIndices[0]);
        }
    }
}

//------------------------------------------------------------------------------
void BASE_SLE::assembleStiffness(int numRows, int* scatterIndices,
                                  const double* const* stiff) {
//
//This function hands the element data off to the routine that finally
//sticks it into the matrix.
//
    for(int i = 0; i < numRows; i++) {
        int ib = scatterIndices[i];
        if ((localStartRow_ <= ib) && (ib <= localEndRow_)) {
            sumIntoSystemMatrix(ib, numRows, &(stiff[i][0]),
                                &scatterIndices[0]);
        }
    }
}

//------------------------------------------------------------------------------
void BASE_SLE::assembleLoad(int numRows, int* scatterIndices,
                                  const double* load) {
//
//This function hands the data off to the routine that finally
//sticks it into the RHS vector.
//
    for(int i = 0; i < numRows; i++) {
        int ib = scatterIndices[i];
        if ((localStartRow_ <= ib) && (ib <= localEndRow_)) {
            sumIntoRHSVector(1, &ib, &(load[i]) );
        }
    }
}

