//
//--distExtBeamDriver.cc--
//
//  sample finite-element interface driver program
//
//  solve a simple (3 unknowns/per node, transverse+axial displacement and
//  slope) beam analysis using the ISIS++ Finite-Element interface.  Here,
//  the parallelism is supported by subdividing the uniform beam into 
//  blocks, with one block per processor.  Other than this scalability
//  convention, this is pretty much just another simple beam analysis... 
//
//
//  this version supports the interface header FEI.h, version 1.0, and is 
//  based on a simple 1D beam analysis FE code designed to exercise various
//  features in the prototype FE interface implementation
//
//  this driver tests enforcement of penalty constraints at the 
//  interface nodes, to test external node logic added for v1.0
//
//
//  k.d. mish 8/26/97, stolen from aw/kdm's seqFEIdriver.cc code
//  k.d. mish 9/1/97,  modified for distributed-memory MPI implementation
//  k.d. mish 10/3/97, added elastic foundation terms to simplify quality
//                     assurance testing and to permit certain unusual
//                     code conditions (like not specifying any b.c.'s)
//  k.d. mish 6/17/98  add FEI v1.0a support and general cleanup
//  k.d. mish 10/22/98 add multiphysics support from the final v1.0 spec
//
//

// #define DRV_MULTIPHYSICS  // if defined, we're testing multiple fields
// #define dbgTrace          // limit amount of stuff dumped to stdout

#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <fei-isis.h>

#include "FE_Node.h"
#include "FE_Elem.h"

//  strategy for dividing beam up among blocks (== processors)

void block_strategy(int numElems, int numBlocks, int *numLocalElems);


//  distributed-memory parallel initialization routine

void pgm_init(int argc, char **argv, int &num_procs, int &my_rank,
    int &master_rank);


//  bailout routine for fatal errors

void pgm_abort(int local_rank);


//-----------------------------------------------------------------------
//  main starts here...
//

int main(int argc, char **argv) {

    int i, j, k, m;
    int localRank, masterRank, numProcs, ierror;


//  parallel program initialization

    pgm_init(argc, argv, numProcs, localRank, masterRank);

#ifdef dbgTrace
    cout << "numProcs: " << numProcs << ", localRank: " << localRank
         << ", masterRank: " << masterRank << endl;
#endif

//  set things up so we can send and receive GlobalID stuff

    MPI_Datatype MPI_GLOBALID;
    ierror = MPI_Type_contiguous(sizeof(GlobalID)/sizeof(int),MPI_INT,
                                 &MPI_GLOBALID);
    if (ierror) cout << "MPI_Type_contiguous ierror: " << ierror << endl;
    MPI_Type_commit(&MPI_GLOBALID);

//  jump out if there isn't a filename argument

    if ((localRank==masterRank) && (argc < 2)) {
       cout << "argc: " << argc << endl;
       for(i=0; i<argc; i++)
           cout << "argv[" << i << "]: " << argv[i] << endl;
       cout << "usage: " << argv[0] << " <input_file>" << endl;
       exit(0);
    }
    else if (localRank==masterRank)
        cout << "Assuming argv[1] (" << argv[1] << ") is the input file."
              << endl;


//  parameters starting with prefix "my" are working copies for generating
//  data from the beam program...

//  declare the FEInterface Object first 
    
    ISIS_SLE linearSystem(MPI_COMM_WORLD, 0);

    char* param = new char[128];
    sprintf(param, "debugOutput .");
    linearSystem.parameters(1, &param);
    sprintf(param, "outputLevel 2");
    linearSystem.parameters(1, &param);
    delete [] param;

//  declarations for basic beam discretization data

    int myNumElements;                 // number of elements in global beam

    FE_Elem *myLocalElements = NULL;   // list of element objects (distributed)
    FE_Node *myLocalNodes = NULL;      // list of node objects (distributed)
    
    int *elemsPerBlock = NULL;         // number of elements for a given block
    GlobalID *startElem = NULL;        // first element number for a given block
    GlobalID *endElem = NULL;          // last element number for a given block
    int *nodesPerBlock = NULL;         // number of nodes for a given block
    GlobalID *startNode = NULL;        // first node number for a given block
    GlobalID *endNode = NULL;          // last node number for a given block

//  declarations for initSolveStep()

    int myNumElemBlocks;
    int mySolvType = 0;                // solve Ax=b linear system

//  declarations for initFields()

#ifdef DRV_MULTIPHYSICS
    int myNumFields = 2;               // 2 fields: x,y-displacements, z-rotations
    int myCardFields[2] = {2, 1};      // cards are thus 2 and 1 
    int myFieldIDs[2] = {5, 10};       // arbitrary field identifiers
#else
    int myNumFields = 1;               // 1 field: displacements and rotations
    int myCardFields[1] = {3};         // card is now 3  ( = 2 + 1 ... duh!)
    int myFieldIDs[1] = {5};           // arbitrary field ID
#endif

//  declarations for beginInitElemBlock()

    GlobalID myElemBlockID;            // block ID for each group of elems
    int myNumNodesPerElem = 2;         // the usual 2-node Hermitian beam elements
    int myStrategy = 0;                // take default field/node interleave
    int myNumElemDOF = 0;              // no elemental DOF (-yet-)
    int myNumElemSets = 1;             // keep things simple for now
    int myNumElemTotal;                // total number of elems in block

#ifdef DRV_MULTIPHYSICS
    int myNumElemFields[2] = {2, 2};   // both fields at each node
    int **myElemFieldIDs;              // table of field identifiers
    myElemFieldIDs = new int* [2];     // 2 nodes to identify fields for
    myElemFieldIDs[0] = new int [2];   // 2 fields at each node
    myElemFieldIDs[1] = new int [2];
    myElemFieldIDs[0][0] = 5;
    myElemFieldIDs[0][1] = 10;
    myElemFieldIDs[1][0] = 5;
    myElemFieldIDs[1][1] = 10;
#else
    int myNumElemFields[2] = {1, 1};   // both fields at each node
    int **myElemFieldIDs;              // table of field identifiers
    myElemFieldIDs = new int* [2];     // 2 nodes to identify fields for
    myElemFieldIDs[0] = new int [1];   // 1 field at each node
    myElemFieldIDs[1] = new int [1];
    myElemFieldIDs[0][0] = 5;
    myElemFieldIDs[1][0] = 5;
#endif

//  declarations for initElemSet()

    int myNumElems;
    GlobalID *myElemIDs = NULL;
    GlobalID **myElemConn = NULL;

//  declarations for beginInitNodeSets()

    int myNumSharedNodeSets = 0;
    int myNumExtSendNodeSets = 0;      // will calc on the fly
    int myNumExtRecvNodeSets = 0;      // both send & recv external node sets

//  declarations for initExtNodeSet()

    GlobalID *myExtSendNodes = NULL;
    int myLenExtSendNodeSet = 0;
    int **myExtSendProcIDs = NULL;
    int *myLenExtSendProcIDs = NULL;

    GlobalID *myExtRecvNodes = NULL;
    int myLenExtRecvNodeSet = 0;
    int **myExtRecvProcIDs = NULL;
    int *myLenExtRecvProcIDs = NULL;

//  declarations for beginInitCREqns()

    int myNumCRMultSets = 0;    // if we want to use these, generate them
    int myNumCRPenSets = 0;     // on the fly (e.g., to test external nodes)

//  declarations for initCRMult() and initCRPen()

    int myLenCRList = 0;
    int myNumCRs = 0;
    GlobalID **myCRNodeTable = NULL;
    int *myCRPenFieldIDList = NULL;
    int *myCRPenIDList = NULL;

//  declarations for loadBCSet()

    GlobalID myBCNodeSet[1] = {0};
    double **myAlphaTable = NULL;
    double **myBetaTable = NULL;
    double **myGammaTable = NULL;

#ifdef DRV_MULTIPHYSICS
    int myNumBCNodeSets = 2;           // 2 sets of BC data, consisting of...
    int myLenBCNodeSet[2] = {1, 1};    // 1 ess BC at node 0 for each field
    int myBCFieldID[2] = {5, 10};      // fieldID for each BC
    int myNumFieldParams[2] = {2, 1};  // num of DOF for each field
#else
    int myNumBCNodeSets = 1;           // 1 sets of BC data, consisting of...
    int myLenBCNodeSet[1] = {1};       // 1 ess BC at node 0 for this field
    int myBCFieldID[1] = {5};          // fieldID for this single BC
    int myNumFieldParams[1] = {3};     // num of DOF for this field
#endif

//  declarations for loadElemSet()

    double **myElemLoads = NULL;
    double ***myElemStiffnesses = NULL;
    int myElemFormat;
    int myElemRows;
    
//  declarations for loadCRMult() and loadCRPen()

    double **myCRWeightTable = NULL;
    double *myCRValueList = NULL;
    double *myPenValues = NULL;
    
    
//  declarations for local use

    int myErrorCode;

    int s_tag1 = 19191;
    MPI_Status status;

    GlobalID localStartElem;
    GlobalID localEndElem;
    GlobalID localStartNode;
    GlobalID localEndNode;
    int localElemsPerBlock;
    int localNodesPerBlock;
    int numProcessors;

    FILE *infd;


//  get down to business...................................

    if (localRank == masterRank){
        infd = fopen(argv[1], "r");
        if (!infd) {
            cout << "ERROR opening input file " << argv[1] << endl;
            exit(0);
        }
    }

//----------------------------------------------------------------
//  initialization process section
//----------------------------------------------------------------

//  read the overall beam discretization initialization data (here,
//  just the number of elements to divide the beam into...)

    if (localRank == masterRank) {
        fscanf(infd, "%d", &myNumElements);     // how many elements to use
        myNumElemBlocks = numProcs;

        assert (myNumElements > 0);
        assert (myNumElemBlocks > 0);

//  create global structures to help map blocks to processors

        elemsPerBlock = new int[myNumElemBlocks];
        startElem = new GlobalID[myNumElemBlocks];
        endElem = new GlobalID[myNumElemBlocks];

        nodesPerBlock = new int[myNumElemBlocks];
        startNode = new GlobalID[myNumElemBlocks];
        endNode = new GlobalID[myNumElemBlocks];

//  map the elements to the blocks (== processors) more-or-less evenly
//  and populate the load-balancing data structures

        block_strategy(myNumElements, myNumElemBlocks, elemsPerBlock);

//  form the appropriate block (== processor) data structures - this step
//  is done once and for all on the master CPU, and the results of this
//  global calculation are sent to the slave CPU's...

        int sumElem = 0;
        for (i = 0; i < myNumElemBlocks; i++) {
            startElem[i] = sumElem;
            endElem[i] = startElem[i] + elemsPerBlock[i] - 1;
            startNode[i] = startElem[i] + i;
            endNode[i] = startNode[i] + elemsPerBlock[i];
            nodesPerBlock[i] = elemsPerBlock[i] + 1;
            sumElem += elemsPerBlock[i];

#ifdef dbgTrace
            cout << " i : " << i << endl;
            cout << "    elemsPerBlock(i) : " << elemsPerBlock[i] << endl;
            cout << "        startElem(i) : " << startElem[i] << endl;
            cout << "          endElem(i) : " << endElem[i] << endl ;
            cout << "    nodesPerBlock(i) : " << nodesPerBlock[i] << endl;
            cout << "        startNode(i) : " << startNode[i] << endl;
            cout << "          endNode(i) : " << endNode[i] << endl << endl ;
            cout << endl << endl ;
#endif
        }

//  ship all this overhead data to the various processors and then
//  the real parallel computation can get done...

        for (i = 1; i < myNumElemBlocks; i++) {
            MPI_Send(&myNumElements, 1, MPI_INT, i, s_tag1,
                     MPI_COMM_WORLD);        
            MPI_Send(&myNumElemBlocks, 1, MPI_INT, i, s_tag1,
                     MPI_COMM_WORLD);        
            MPI_Send(&startElem[i], 1, MPI_GLOBALID, i, s_tag1,
                     MPI_COMM_WORLD);
            MPI_Send(&endElem[i], 1, MPI_GLOBALID, i, s_tag1,
                     MPI_COMM_WORLD);
            MPI_Send(&elemsPerBlock[i], 1, MPI_INT, i, s_tag1,
                     MPI_COMM_WORLD);        
            MPI_Send(&startNode[i], 1, MPI_GLOBALID, i, s_tag1,
                     MPI_COMM_WORLD);
            MPI_Send(&endNode[i], 1, MPI_GLOBALID, i, s_tag1,
                     MPI_COMM_WORLD);
            MPI_Send(&nodesPerBlock[i], 1, MPI_INT, i, s_tag1,
                     MPI_COMM_WORLD);        
        }
 
//  create masterRank local copies of partitioning parameters

        numProcessors = myNumElemBlocks;
        localStartElem = startElem[0];
        localEndElem = endElem[0];
        localStartNode = startNode[0];
        localEndNode = endNode[0];
        localElemsPerBlock = elemsPerBlock[0];
        localNodesPerBlock = nodesPerBlock[0];
       
        delete [] startElem;
        delete [] endElem;
        delete [] nodesPerBlock;
        delete [] startNode;
        delete [] endNode;

    }       // end the "if (localRank == masterRank)" block

    else {

//  receive the data sent by the master CPU...

        MPI_Recv(&myNumElements, 1, MPI_INT, masterRank, s_tag1,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&numProcessors, 1, MPI_INT, masterRank, s_tag1,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&localStartElem, 1, MPI_GLOBALID, masterRank, s_tag1,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&localEndElem, 1, MPI_GLOBALID, masterRank, s_tag1,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&localElemsPerBlock, 1, MPI_INT, masterRank, s_tag1,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&localStartNode, 1, MPI_GLOBALID, masterRank, s_tag1,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&localEndNode, 1, MPI_GLOBALID, masterRank, s_tag1,
                 MPI_COMM_WORLD, &status);
        MPI_Recv(&localNodesPerBlock, 1, MPI_INT, masterRank, s_tag1,
                 MPI_COMM_WORLD, &status);
    }    


//---------------------------------------------------------------------
//  begin SPMD computation, since all the global overhead has been done
//---------------------------------------------------------------------

//  redefine the element and node sizes for local storage instead of
//  the global storage required for initialization
    
//  create local list of element and node objects

    myLocalElements = new FE_Elem[localElemsPerBlock];
    myLocalNodes = new FE_Node[localNodesPerBlock];

//  load up the element and node objects with appropriate data for this
//  particular problem (namely, a uniform 1D beam with a fixed left end
//  and a uniform transverse load... here, the parameters EI, etc., have
//  been chosen to make the tip displacement/rotation work out to be nice
//  whole numbers)

    double EI = 100.0;
    double EA = 200.0;
    double qLoad = 12.0;
    double pLoad = 4.0;
    double fndSpring = 0.0;
    double elemLength = 10.0/myNumElements;

    GlobalID jGlobal;
    GlobalID myElemNodeList[2];
    for (j = 0; j < localElemsPerBlock; j++) {
        jGlobal = localStartElem + (GlobalID)j;
        myLocalElements[j].globalElemID(jGlobal);
        myLocalElements[j].numElemRows(6);
        myLocalElements[j].numElemNodes(2);
        myLocalElements[j].elemLength(elemLength);
        myLocalElements[j].bendStiff(EI);
        myLocalElements[j].foundStiff(fndSpring);
        myLocalElements[j].axialStiff(EA);
        myLocalElements[j].distLoad(qLoad);
        myLocalElements[j].axialLoad(pLoad);
        myLocalElements[j].allocateNodeList();

        myElemNodeList[0] = localStartNode + (GlobalID)j;
        myElemNodeList[1] = localStartNode + (GlobalID)(j + 1);
        myLocalElements[j].storeNodeList(2, myElemNodeList);

#ifdef dbgTrace
        myLocalElements[j].dumpToScreen();
#endif

    }

    for (j = 0; j < localNodesPerBlock; j++) {
        jGlobal = localStartNode + (GlobalID)j;
        myLocalNodes[j].globalNodeID(jGlobal);
        myLocalNodes[j].numNodalDOF(3);
        myLocalNodes[j].nodePosition((localStartNode+j)*elemLength);
        myLocalNodes[j].allocateSolnList();

#ifdef dbgTrace
        myLocalNodes[j].dumpToScreen();
#endif

    }


//  FE interface call ----------------------------------------------

    myNumElemBlocks = 1;   // temporary pll driver assumption, 1 per proc!!
    
    myErrorCode = linearSystem.initSolveStep(myNumElemBlocks, 
                                             mySolvType); 
    if (myErrorCode) cout << "error in linearSystem.initSolveStep" << endl;

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.initFields(myNumFields, 
                                          myCardFields,
                                          myFieldIDs); 
    if (myErrorCode) cout << "error in linearSystem.initFields" << endl;

//  pass the block control parameter data for a beam

    myElemBlockID = (GlobalID) localRank;
    myNumElems = localElemsPerBlock;
    myNumElemTotal = myNumElems;
    assert (myNumElems > 0);

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.beginInitElemBlock(myElemBlockID, 
                                                  myNumNodesPerElem,
                                                  myNumElemFields, 
                                                  myElemFieldIDs,
                                                  myStrategy,
                                                  myStrategy,
                                                  myNumElemDOF, 
                                                  myNumElemSets,
                                                  myNumElemTotal);
    if (myErrorCode) cout << "error in linearSystem.beginInitElemBlock" << endl;

#ifdef dbgTrace
    cout << "args to beginInitElemBlock:" << endl;
    cout << "  ElemBlockID          = " << myElemBlockID << endl;
    cout << "  NumNodesPerElem      = " << myNumNodesPerElem << endl;
    cout << "  myNumElemFields[0]   = " << myNumElemFields[0] << endl;
    cout << "  myElemFieldIDs[0][0] = " << myElemFieldIDs[0][0] << endl;
    cout << "  myStrategy           = " << myStrategy << endl;
    cout << "  NumElemDOF           = " << myNumElemDOF << endl;
    cout << "  myNumElemSets        = " << myNumElemSets << endl;
    cout << "  myNumElemTotal       = " << myNumElemTotal << endl << endl;
#endif

//  for now, to keep things simple, just use one workset per block...

    int ntest; 
    for (i = 0; i < myNumElemSets; ++i) {
	    myElemIDs = new GlobalID[myNumElems];
	    myElemConn = new GlobalID* [myNumElems];
	
	    for (k = 0; k < myNumElems; k++) {
	        myElemIDs[k] = localStartElem + (GlobalID)k;
	        myElemConn[k] = new GlobalID[myNumNodesPerElem];
	        myLocalElements[k].returnNodeList(ntest,myElemConn[k]);
	    }
	
//  FE interface call ----------------------------------------------
	
	    myErrorCode = linearSystem.initElemSet(myNumElems, 
	                                           myElemIDs, 
	                                           myElemConn);
	    if (myErrorCode) cout << "error in linearSystem.initElemSet" << endl;
	
	
	    delete [] myElemIDs;
    }
    
//
//  we need the connectivity later, so don't get rid of it yet...
//

//  close the element workset initialization phase

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.endInitElemBlock();
    if (myErrorCode) cout << "error in linearSystem.endInitElemBlock" << endl;

//  (recall that there is only one block per processor here, by definition)


//  pass the node list control initialization data first

//  FE interface call ----------------------------------------------

    if (numProcessors == 1) {         
        myNumExtSendNodeSets = 0;
        myNumExtRecvNodeSets = 0;
    }
    else if (localRank == 0) {
        myNumExtSendNodeSets = 0;
        myNumExtRecvNodeSets = 1;
    }
    else if (localRank == (numProcessors - 1)) {
        myNumExtSendNodeSets = 1;
        myNumExtRecvNodeSets = 0;
    }
    else {
        myNumExtSendNodeSets = 1;
        myNumExtRecvNodeSets = 1;
    }
    int myNumExtNodeSets = myNumExtSendNodeSets + myNumExtRecvNodeSets;

    myErrorCode = linearSystem.beginInitNodeSets(myNumSharedNodeSets,
                                                 myNumExtNodeSets);
                                                 
    if (myErrorCode) cout << "error in linearSystem.beginInitNodeSets" << endl;


//  pass the external send/recv node data

    for (i = 0; i < myNumExtSendNodeSets; i++) {
        myLenExtSendNodeSet = 1;
        myExtSendNodes = new GlobalID[myLenExtSendNodeSet];
        myLenExtSendProcIDs = new int[myLenExtSendNodeSet];
        myExtSendProcIDs = new int* [myLenExtSendNodeSet];
#ifdef dbgTrace
        cout << "procID : " << localRank << endl;
#endif
        for (j = 0; j < myLenExtSendNodeSet; j++) {
            myExtSendNodes[j] = localStartNode;
            myLenExtSendProcIDs[j] = 1;
            myExtSendProcIDs[j] = new int[myLenExtSendProcIDs[j]];
#ifdef dbgTrace
            cout << "    myExtSendNodes[j]      : " << (int)myExtSendNodes[j] << endl;
            cout << "    myLenExtSendProcIDs[j] : " << myLenExtSendProcIDs[j] << endl;
            cout << "    myExtSendProcIDs       : ";
#endif
            for (k = 0; k < myLenExtSendProcIDs[j]; k++) {
                myExtSendProcIDs[j][k] = localRank - 1;
#ifdef dbgTrace
                cout << myExtSendProcIDs[j][k] << "  ";
#endif
            }
#ifdef dbgTrace
            cout << endl;
#endif
        }

//  FE interface call ----------------------------------------------

	    myErrorCode = linearSystem.initExtNodeSet(myExtSendNodes,
                                	              myLenExtSendNodeSet,
                                	              myExtSendProcIDs,
                                	              myLenExtSendProcIDs);
	    if (myErrorCode) 
                cout << "error in linearSystem.initExtNodeSet" << endl;


	    delete [] myExtSendNodes;
	    for (j = 0; j < myLenExtSendNodeSet; j++) {
                delete [] myExtSendProcIDs[j];
	    }
	    delete [] myExtSendProcIDs;
    }


    for (i = 0; i < myNumExtRecvNodeSets; i++) {
        myLenExtRecvNodeSet = 1;
        myExtRecvNodes = new GlobalID[myLenExtRecvNodeSet];
        myLenExtRecvProcIDs = new int[myLenExtRecvNodeSet];
        myExtRecvProcIDs = new int* [myLenExtRecvNodeSet];

        for (j = 0; j < myLenExtRecvNodeSet; j++) {
            myExtRecvNodes[j] = localEndNode;
            myExtRecvNodes[j]++;
            myLenExtRecvProcIDs[j] = 1;
            myExtRecvProcIDs[j] = new int[myLenExtRecvProcIDs[j]];
            for (k = 0; k < myLenExtRecvProcIDs[j]; k++) {
                myExtRecvProcIDs[j][k] = localRank + 1;
            }
        }

//  FE interface call ----------------------------------------------

	    myErrorCode = linearSystem.initExtNodeSet(myExtRecvNodes,
                                	              myLenExtRecvNodeSet,
                                	              myExtRecvProcIDs,
                                	              myLenExtRecvProcIDs);
	    if (myErrorCode) 
                cout << "error in linearSystem.initExtNodeSet" << endl;

	    delete [] myExtRecvNodes;
	    for (j = 0; j < myLenExtRecvNodeSet; j++) {
            delete [] myExtRecvProcIDs[j];
	    }
	    delete [] myExtRecvProcIDs;
        delete [] myLenExtRecvProcIDs;
    }

//  close the node list initialization phase

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.endInitNodeSets();
    if (myErrorCode) cout << "error in linearSystem.endInitNodeSets" << endl;


//  read and store the constraint relation control data

//  apply 3 constraints at the right end of each block, in this case, to tie
//  the displacement and slope at the right end of the block to the same
//  solution parameters at the left end of the next block...
//
//  obviously, there won't be any such constraints at the right end of the
//  last block, hence the test on localRank and (numProcessors - 1)

    if (localRank < (numProcessors - 1)) {  //$kdm debug here!
        myNumCRPenSets = 3;  // get 3 at each block interface
#ifdef dbgTrace
        cout << "processor " << localRank << "    myNumCRPenSets    " 
             << myNumCRPenSets << endl;
#endif
    } 
    else {
        myNumCRPenSets = 0;  // no interface at right end
    }

//  in this driver, there are no Lagrange constraint sets, so...

    myNumCRMultSets = 0;

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.beginInitCREqns(myNumCRMultSets, 
                                               myNumCRPenSets); 
    if (myErrorCode) cout << "error in linearSystem.beginInitCREqns" << endl;

//  generate the constraint relation initialization data on the fly...

    if (myNumCRPenSets > 0) {
        myNumCRs = 1;          // 1 eqn (disp or slope) in each constraint set
        myLenCRList = 2;       // 2 nodes (one per block) in each constraint
        myCRPenIDList = new int[myNumCRPenSets];
        myCRPenFieldIDList = new int [myLenCRList];
        int countCR = 0;
        for (i = 0; i < myNumFields; i++) {
            myCRPenFieldIDList[0] = myFieldIDs[i];   // here, tie the same 
            myCRPenFieldIDList[1] = myFieldIDs[i];   // field at each end...
            for (j = 0; j < myCardFields[i]; j++) {
                myCRNodeTable = new GlobalID* [myNumCRs];
                for (k = 0; k < myNumCRs; k++) {
                    myCRNodeTable[k] = new GlobalID[myLenCRList];
                    myCRNodeTable[k][0] = localEndNode;
                    myCRNodeTable[k][1] = localEndNode;
                    myCRNodeTable[k][1]++;
#ifdef dbgTrace
                    cout << "localRank = " << localRank << endl;
                    cout << "myLenCRList = " << myLenCRList << endl;
                    cout << "nodes for constrSet " << j << ", constraint " 
                         << k << ", block " << i << endl;
                    cout << "     " << (int)myCRNodeTable[k][0] << "  " 
                                    << (int)myCRNodeTable[k][1] << endl;
#endif
                }

//  FE interface call ----------------------------------------------

                myErrorCode = linearSystem.initCRPen(myCRNodeTable,
                                                     myCRPenFieldIDList, 
                                                     myNumCRs, 
                                                     myLenCRList,
                                                     myCRPenIDList[countCR]);
                if (myErrorCode) cout << "error in linearSystem.initCRPen" << endl;

                ++countCR;

                for (m = 0; m < myNumCRs; m++) {
                    delete [] myCRNodeTable[m];
                }
                delete [] myCRNodeTable;
            }
        }
    }


//  close the constraint relation initialization phase
 
//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.endInitCREqns();
    if (myErrorCode) cout << "error in linearSystem.endInitCREqns" << endl;


//  close the finite-element data initialization phase

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "processor " << localRank << " ready to call initComplete" << endl;
    cout << flush;
    MPI_Barrier(MPI_COMM_WORLD);

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.initComplete();
    if (myErrorCode) cout << "error in linearSystem.initComplete" << endl;

    cout << "processor " << localRank << " back from initComplete" << endl;
    cout << flush;
    MPI_Barrier(MPI_COMM_WORLD);



//----------------------------------------------------------------
//  load process section
//----------------------------------------------------------------

//  begin node-set data load step

//  load the essential boundary condition tabular data vales - the essential
//  bc at the left support (x = 0) only applies to the master CPU...

//  FE interface call ----------------------------------------------


    if (localRank != masterRank) {
        myNumBCNodeSets = 0;
    }
    else {

        myErrorCode = linearSystem.beginLoadNodeSets(myNumBCNodeSets);
        if (myErrorCode) cout << "error in linearSystem.beginLoadNodeSets" << endl;

//  generate the essential BC's for all fields at node zero

        for (k = 0; k < myNumFields; k++) {
            myAlphaTable = new double* [myLenBCNodeSet[k]];
            myBetaTable = new double* [myLenBCNodeSet[k]];
            myGammaTable = new double* [myLenBCNodeSet[k]];
            for (j = 0; j < myLenBCNodeSet[k]; j++) {
                myAlphaTable[j] = new double[myNumFieldParams[k]];
                myBetaTable[j] = new double[myNumFieldParams[k]];
                myGammaTable[j] = new double[myNumFieldParams[k]];
            }
            for (i = 0; i < myLenBCNodeSet[k]; i++) {
                for (j = 0; j < myNumFieldParams[k]; j++) {
                    myAlphaTable[i][j] = 1.0;
                    myBetaTable[i][j] = 0.0;
                    myGammaTable[i][j] = 0.0;
                }
            }
    
//  FE interface call ----------------------------------------------

            myErrorCode = linearSystem.loadBCSet(myBCNodeSet, 
                                                 myLenBCNodeSet[k],
                                                 myBCFieldID[k],
                                                 myAlphaTable, 
                                                 myBetaTable, 
                                                 myGammaTable);
            if (myErrorCode) cout << "error in linearSystem.loadBCSet" << endl;

            for (j = 0; j < myLenBCNodeSet[k]; j++) {
                delete [] myAlphaTable[j];
                delete [] myBetaTable[j];
                delete [] myGammaTable[j];
            }
            delete [] myAlphaTable;
            delete [] myBetaTable;
            delete [] myGammaTable;
        }
    }

//  end node-set data load step

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.endLoadNodeSets();
    if (myErrorCode) cout << "error in linearSystem.endLoadNodeSets" << endl;

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "processor " << localRank << ", finished endLoadNodeSets" << endl;
    cout << flush;
    MPI_Barrier(MPI_COMM_WORLD);


//  begin blocked-element (i.e., processor-based) data loading step

    myElemBlockID = (GlobalID) localRank;
    myNumElems = localElemsPerBlock;
    myNumElemTotal = myNumElems;
    assert (myNumElems > 0);

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.beginLoadElemBlock(myElemBlockID,
                                                  myNumElemSets,
                                                  myNumElemTotal);
    if (myErrorCode) cout << "error in linearSystem.beginLoadElemBlock" << endl;

//  assemble the element matrix data assuming one workset per block
//  (note that in a production code, we'd probably be allocating memory
//  for the connectivity here... in this simple example with one workset
//  per block and one block per processor, we can just reuse the array
//  that we created back in the initialization step

    for (i = 0; i < myNumElemSets; ++i) {
	    myNumElems = localElemsPerBlock;
	    assert (myNumElems > 0);
	    myElemIDs = new GlobalID[myNumElems];
	
	    int myElemIndex; 
	    for (j = 0; j < myNumElems; j++) {
	        myElemIndex = localStartElem + j;
			myElemIDs[j] = (GlobalID) myElemIndex;
	        myLocalElements[j].returnNodeList(ntest,myElemConn[j]);
	    }
	
	    myElemFormat = 0;       // not worrying about format -yet- !!!
	    myElemLoads = new double* [myNumElems];
	    myElemStiffnesses = new double** [myNumElems];
	
//  evaluate the element load vector and stiffness matrices for this workset
	
	    int neRows;
	    myElemRows = 6;
#ifdef dbgTrace
	    cout << "proc " << localRank << ", myNumElems: " << myNumElems << endl << flush;
#endif
	    for (j = 0; j < myNumElems; j++) {
	        myElemLoads[j] = new double[myElemRows];
	        myElemStiffnesses[j] = new double* [myElemRows];
	        for (m = 0; m < myElemRows; m++) {
	            myElemStiffnesses[j][m] = new double[myElemRows];
	        }
	        myLocalElements[j].evaluateLoad(neRows,myElemLoads[j]);
	        assert(neRows == myElemRows);
	        myLocalElements[j].evaluateStiffness(neRows,myElemStiffnesses[j]);
	        assert(neRows == myElemRows);
	    }
	
//  FE interface call ----------------------------------------------
	
	    int myElemSetID = 0;
	
	    myErrorCode = linearSystem.loadElemSet(myElemSetID, 
	                                           myNumElems, 
	                                           myElemIDs, 
	                                           myElemConn,
	                                           myElemStiffnesses, 
	                                           myElemLoads, 
	                                           myElemFormat);
	    if (myErrorCode) cout << "error in linearSystem.loadElemSet" << endl;
	
	
	    for (j = 0; j < myNumElems; j++) {
	        delete [] myElemConn[j];
	        delete [] myElemLoads[j];
	        for (m = 0; m < myElemRows; m++) {
	            delete [] myElemStiffnesses[j][m];
	        }
	        delete [] myElemStiffnesses[j];
	    }
	    delete [] myElemStiffnesses;
	
	    delete [] myElemIDs;
	    delete [] myElemConn;
	    delete [] myElemLoads;
    }

//  close the element workset data load phase

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "processor " << localRank << ", finished loadElemSet" << endl;
    cout << flush;
    MPI_Barrier(MPI_COMM_WORLD);

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.endLoadElemBlock();
    if (myErrorCode) cout << "error in linearSystem.endLoadElemBlock" << endl;


//  begin constraint relation data load step

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.beginLoadCREqns(myNumCRMultSets, 
                                               myNumCRPenSets);
    if (myErrorCode) cout << "error in linearSystem.beginLoadCREqns" << endl;

//  Penalty constraint relation load step 
//  (see the initialization step for some of the data definitions below)

    if (myNumCRPenSets > 0) {
        int countCR = 0;
        double factor;
        for (i = 0; i < myNumFields; i++) {
            myCRPenFieldIDList[0] = myFieldIDs[i];   // here, tie the same 
            myCRPenFieldIDList[1] = myFieldIDs[i];   // field at each end...
            for (j = 0; j < myCardFields[i]; j++) {
                myCRNodeTable = new GlobalID* [myNumCRs];
                for (k = 0; k < myNumCRs; k++) {
                    myCRNodeTable[k] = new GlobalID[myLenCRList];
                    myCRNodeTable[k][0] = localEndNode;
                    myCRNodeTable[k][1] = localEndNode;
                    myCRNodeTable[k][1]++;
#ifdef dbgTrace
                    cout << "nodes for constrSet " << countCR  
                         << ", constraint " << k << ", block " << i << endl;
                    cout << "     " << (int)myCRNodeTable[k][0] << "  " << 
                                       (int)myCRNodeTable[k][1] << endl;
#endif
                }
    
                myCRWeightTable = new double* [myLenCRList];
                myNumFieldParams[j] = myCardFields[i];
                for (k = 0; k < myLenCRList; k++) {
                    myCRWeightTable[k] = new double[myNumFieldParams[j]];

//  we want to populate the various field-based weight tables with zeros
//  in columns not participating in the constraint, and +1 or -1 in those
//  participating in the constraint, so here's some screwy code to do this...

                    if (k == 0) 
                        factor = 1.0;
                    else
                        factor = -1.0;
                    
                    for (m = 0; m < myCardFields[i]; m++) {
                        myCRWeightTable[k][m] = 0.0;
                    }
                    myCRWeightTable[k][j] = factor;
                }
            
                myCRValueList = new double[myNumCRs];
           	    myPenValues = new double[myNumCRs];
                for (k = 0; k < myNumCRs; k++) {
                    myCRValueList[k] = 0.0;
                    myPenValues[k] = 1.0e10;
                }

#ifdef dbgTrace
                cout << "proc " << localRank << ", myLenCRList: " << myLenCRList
                     << ", myNumFieldParams: " << myNumFieldParams[j] << endl;
#endif

//  FE interface call ----------------------------------------------

                myErrorCode = linearSystem.loadCRPen(myCRPenIDList[j],
                                                     myNumCRs, 
                                                     myCRNodeTable, 
                                                     myCRPenFieldIDList, 
                                                     myCRWeightTable, 
                                                     myCRValueList,
                                                     myPenValues,
                                                     myLenCRList);
                if (myErrorCode) cout << "error in linearSystem.loadCRPen" 
                                      << endl;

                ++countCR;

                delete [] myPenValues;
                delete [] myCRValueList;
            
                for (m = 0; m < myNumCRs; m++) {
                    delete [] myCRNodeTable[m];
                }
                delete [] myCRNodeTable;
            
                for (m = 0; m < myLenCRList; m++) {
                    delete [] myCRWeightTable[m];
                }
                delete [] myCRWeightTable;
            }
        }
    }

//  end constraint relation data load step 

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.endLoadCREqns();
    if (myErrorCode) cout << "error in linearSystem.endLoadCREqns" << endl;

    if (myNumCRPenSets > 0) {
        delete [] myCRPenIDList;
    }
 
//  end of overall data loading sequence

//  FE interface call ----------------------------------------------

    myErrorCode = linearSystem.loadComplete();
    if (myErrorCode) cout << "error in linearSystem.loadComplete" << endl;

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "processor " << localRank << " finished loadComplete" << endl <<
    flush;
    MPI_Barrier(MPI_COMM_WORLD);


//  all done

    if (localRank == masterRank) fclose(infd);
    

//----------------------------------------------------------------
//  test the "put" functions for passing an initial guess to the
//  solver module
//----------------------------------------------------------------

    double *putSolnValues;
    int *putSolnOffsets;
    GlobalID *putNodeList;
    int putLenList;

	myElemBlockID = (GlobalID) localRank;
    int putNumBlkActNodes = linearSystem.getNumBlockActNodes(myElemBlockID);
    int putNumBlkActEqns = linearSystem.getNumBlockActEqns(myElemBlockID);
    putNodeList = new GlobalID[putNumBlkActNodes];
    putSolnOffsets = new int[putNumBlkActNodes];
    putSolnValues = new double[putNumBlkActEqns];

//	get the active node list for this block of elements

    linearSystem.getBlockNodeIDList(myElemBlockID, putNodeList, putLenList);

//	initialize the active node list solution parameters

	int offsetCounter = 0;
    for (j = 0; j < putLenList; j++) {
        jGlobal = putNodeList[j];
        int kLimit = linearSystem.getNumSolnParams(jGlobal);
        assert (kLimit == 3);	// paranoia just for this beam problem
        for (k = 0; k < kLimit; k++) {
        	putSolnValues[offsetCounter+k] = jGlobal;  // dummy guess
        }
        putSolnOffsets[j] = offsetCounter;
        offsetCounter += kLimit;
    }
    assert (offsetCounter == putNumBlkActEqns);  // a general paranoia

    linearSystem.putBlockNodeSolution(myElemBlockID, putNodeList,
                                      putLenList, putSolnOffsets, 
                                      putSolnValues);

    delete [] putNodeList;
    delete [] putSolnOffsets;
    delete [] putSolnValues;
	

//----------------------------------------------------------------
//  solution process section
//----------------------------------------------------------------

    int ii, numParams = 6;
    char **paramStrings = new char*[numParams];
    for(ii=0; ii<numParams; ii++) paramStrings[ii] = new char[64];

    strcpy(paramStrings[0],"solver qmr");
    strcpy(paramStrings[1],"preconditioner diagonal");
    strcpy(paramStrings[2],"maxIterations 500");
    strcpy(paramStrings[3],"tolerance 1.e-10");
    strcpy(paramStrings[4],"rowScale false");
    strcpy(paramStrings[5],"colScale false");

    linearSystem.parameters(numParams, paramStrings);

    for(ii=0; ii<numParams; ii++) delete [] paramStrings[ii];
    delete [] paramStrings;

//  FE interface call ----------------------------------------------

    linearSystem.iterateToSolve();


//----------------------------------------------------------------
//  solution return process section
//----------------------------------------------------------------

//  check the solution return functions...

    double *mySolnValues;
    int *mySolnOffsets;
    GlobalID *myNodeList;
    int myLenList;
    
    j = localRank;

    int myNumBlkActNodes = linearSystem.getNumBlockActNodes(j);
    int myNumBlkActEqns = linearSystem.getNumBlockActEqns(j);
    myNodeList = new GlobalID[myNumBlkActNodes];
    mySolnOffsets = new int[myNumBlkActNodes];
    mySolnValues = new double[myNumBlkActEqns];

//  FE interface call ----------------------------------------------

    linearSystem.getBlockNodeSolution(j, myNodeList, myLenList,
                                      mySolnOffsets, mySolnValues);

#ifdef dbgTrace
    cout << "myNumBlkActNodes = " << myNumBlkActNodes << "\n";
    cout << "myLenList = " << myLenList << "\n";
    cout << "myNumBlkActEqns = " << myNumBlkActEqns << "\n";
#endif

    assert (myNumBlkActNodes == myLenList);

    delete [] myNodeList;
    delete [] mySolnOffsets;
    delete [] mySolnValues;

//  clean up everything...
    
    delete [] myLocalElements;
    delete [] myLocalNodes;
    delete [] elemsPerBlock;


    cout << "Calling MPI_Finalize..." << endl;
    MPI_Finalize();

    cout << "...now exiting." << endl;
    return(1);
}


//================================================================================
//
//  strategy for blocking a 1D mesh composed of numElems elements
//  into p blocks, such that the number of elements per block does
//  not vary by more than one
//

void block_strategy(int numElems, int numBlocks, int *numLocalElems)
{
    int i, numTry, limit;

    assert (numElems > 0);
    assert (numBlocks > 0);

    numTry = numElems/numBlocks;
    limit = numElems % numBlocks;
    
    for (i = 0; i < numBlocks; i++) {
        numLocalElems[i] = numTry;
    }
    
    for (i = 0; i < limit; i++) {
        ++numLocalElems[i];
    }
    
    return;
}


//================================================================================
//
//  init all the MPI efforts
//

void pgm_init(int argc, char **argv, int &num_procs, int &my_rank,
    int &master_rank) {

//  Parallel program initialization.

//  Perform MPI initialization.

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

//  Let everybody know who the master proc is.

    master_rank = 0;
    MPI_Bcast(&master_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return;
}


//================================================================================
//
//  abort as cleanly as we can, given some error condition
//
void pgm_abort(int local_rank) {
 
    cout << endl << endl << "pgm_abort: ...local rank = " 
         << local_rank << endl;
    cout << "pgm_abort: ...now calling MPI_Finalize..." << endl;
    MPI_Finalize();
    cout << "pgm_abort: ...now exiting." << endl << endl;
    exit(1);
}
 
