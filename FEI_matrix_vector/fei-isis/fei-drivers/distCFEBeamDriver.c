/*
//
//--distCFEBeamDriver.c--
//
//  sample C-callable finite-element interface (FEI) driver program
//
//  solve a simple (2 unknowns/per node, namely transverse displacement and
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
//
//  k.d. mish 8/26/97, stolen from aw/kdm's seqFEIdriver.cc code
//  k.d. mish 9/1/97,  modified for distributed-memory MPI implementation
//  k.d. mish 6/17/97  add FEI v1.0-interim support and general cleanup
//  k.d. mish 1/6/99   add FEI v1.0-final support
//
//
//
*/

/*
#define dbgTrace         // limit amount of stuff dumped to stdout
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "cfei-isis.h"

#include "CFE_Node.h"
#include "CFE_Elem.h"


/* 
//  strategy for dividing beam up among blocks (== processors)
*/

void block_strategy(int numElems, int numBlocks, int *numLocalElems);


/* 
//  distributed-memory parallel initialization routine
*/

void pgm_init(int argc, char **argv, int *num_procs, int *my_rank,
    int *master_rank);


/* 
//-----------------------------------------------------------------------
//  main starts here...
//
*/

int main(int argc, char **argv) {

    int localRank, masterRank, numProcs, ierror;


/* 
//  parameters starting with prefix "my" are working copies for generating
//  data from the beam program...
*/

/* 
//  declarations for basic beam discretization data
*/

    int myNumElements;              /* // number of elements in global beam */
    int myNumNodes;                 /* // number of nodes in global beam */
    int myElemCount;                /* // simple element counter */
    int myNodeCount;                /* // simple node counter */

    static FE_Elem *myLocalElements = NULL;   /* // list of element objects (distributed) */
    static FE_Node *myLocalNodes = NULL;      /* // list of node objects (distributed) */
    
    int *elemsPerBlock = NULL;         /* // number of elements for a given block */
    GlobalID *startElem = NULL;        /* // first element number for a given block */
    GlobalID *endElem = NULL;          /* // last element number for a given block */
    int *nodesPerBlock = NULL;         /* // number of nodes for a given block */
    GlobalID *startNode = NULL;        /* // first node number for a given block */
    GlobalID *endNode = NULL;          /* // last node number for a given block */
    int numSharedNodes;                /* // number of shared nodes for each block */
    GlobalID *listSharedNodes = NULL;  /* // list of shared nodes for each block */
    int **listSharedProcs = NULL;      /* // table of procs for each shared node */

/* 
//  declarations for initSolveStep()
*/

    int myNumElemBlocks;
    int mySolvType = 0;                /* solve Ax=b linear system */

/* 
//  declarations for initFields()
*/
    int myNumFields = 1;               /* 1 field: displacement and rotations */
    int myCardFields[1] = {2};         /* card is now 2  ( = 1 + 1 ... duh!) */
    int myFieldIDs[1] = {5};           /* arbitrary field ID */

/* 
//  declarations for beginInitElemBlock()
*/

    GlobalID myElemBlockID;            /* block ID for each group of elems */
    int myNumNodesPerElem = 2;         /* the usual 2-node Hermitian beam elements */
    int myStrategy = 0;                /* take default field/node interleave */
    int myNumElemDOF = 0;              /* no elemental DOF (-yet-) */
    int myNumElemSets = 1;             /* keep things simple for now */
    int myNumElemTotal;                /* total number of elems in block */

    int myNumElemFields[2] = {1, 1};   /* both fields at each node */
    int **myElemFieldIDs;              /* table of field identifiers */

/* 
//  declarations for initElemSet()
*/

    int myNumElems;
    GlobalID *myElemIDs = NULL;
    GlobalID **myElemConn = NULL;

/* 
//  declarations for beginInitNodeSets()
*/

    int myNumSharedNodeSets = 0;
    int myNumExtNodeSets = 0;          /* no external nodes in this driver */

/* 
//  declarations for initSharedNodeSet()
*/

    GlobalID *mySharedNodes = NULL;
    int myLenSharedNodeSet;
    int **mySharedProcIDs = NULL;
    int *myLenSharedProcIDs;

/* 
//  declarations for beginInitCREqns()    [currently unused]
*/

    int myNumCRMultSets = 0;           /* no constraint relations in this code */
    int myNumCRPenSets = 0;

/* 
//  declarations for initCRMult() and initCRPen()    [currently unused]
*/

    int myLenCRList = 0;
    int myNumCRs = 0;
    GlobalID **myCRNodeTable = NULL;
    int *myCRMultIDList = NULL;
    int *myCRPenIDList = NULL;

/* 
//  declarations for loadBCSet()
*/

    int myNumBCNodeSets = 1;         /* one set of bc data, consisting of... */
    int myLenBCNodeSet = 1;          /* just the one essential bc at node 0 */
    int myNumSolnParams = 2;         /* with two solution params for a beam */
    GlobalID myBCNodeSet[1] = {0};
    double **myAlphaTable = NULL;
    double **myBetaTable = NULL;
    double **myGammaTable = NULL;
    int myBCFieldID = 5;             /* fieldID for this single BC */
    
/* 
//  declarations for loadElementData()
*/

    double **myElemLoads = NULL;
    double ***myElemStiffnesses = NULL;
    int myElemFormat;
    int myElemRows;
    int myElemSetID = 0;
    
/* 
//  declarations for loadCRMult() and loadCRPen()   [currently unused]
*/

    double **myCRWeightTable = NULL;
    double *myCRValueList = NULL;
    double *myPenValues = NULL;
    int myPenType;
        
/* 
//  declarations for parameters() call
*/

    int ii, numParams = 6;
    char **paramStrings = malloc(numParams*sizeof(char*));
       
/* 
//  declarations for getBlockNodeSolution() call
*/

    double *mySolnValues;
    int *mySolnOffsets;
    GlobalID *myNodeList;
    int myLenList;
    int myNumBlkActNodes;
    int myNumBlkActEqns;

/* 
//  declarations for local use
*/

    int myErrorCode;
    int i, j, k, m, n;

    int s_tag1 = 19191, s_tag2 = 29292;
    MPI_Status status; 

    GlobalID localStartElem;
    GlobalID localEndElem;
    GlobalID localStartNode;
    GlobalID localEndNode;
    int localElemsPerBlock;
    int localNodesPerBlock;
    int numProcessors;
	int mySysNum;
    int sumElem;
    int neRows;

    FILE *infd;

    double EI;
    double qLoad;
    double elemLength;
    GlobalID jGlobal;

    MPI_Datatype MPI_GLOBALID;
    
    FE_Elem *myElemPtr;
    FE_Node *myNodePtr;


/* 
//  parallel program initialization
*/

    pgm_init(argc, argv, &numProcs, &localRank, &masterRank);

    
/* 
//  set things up so we can send and receive GlobalID stuff
*/

    ierror = MPI_Type_contiguous(sizeof(GlobalID)/sizeof(int), MPI_INT,
                                 &MPI_GLOBALID);
    MPI_Type_commit(&MPI_GLOBALID);


/* 
//  jump out if there isn't a filename argument
*/

    if ((localRank==masterRank) && (argc < 2)) {
       printf("fatal error: incorrect filename argument -- exiting...\n\n");
       exit(0);
    }


    myAlphaTable = malloc(myLenBCNodeSet*sizeof(double*));
    myBetaTable = malloc(myLenBCNodeSet*sizeof(double*));
    myGammaTable = malloc(myLenBCNodeSet*sizeof(double*));
    for (j = 0; j < myLenBCNodeSet; j++) {
        myAlphaTable[j] = malloc(myNumSolnParams*sizeof(double));
        myBetaTable[j] = malloc(myNumSolnParams*sizeof(double));
        myGammaTable[j] = malloc(myNumSolnParams*sizeof(double));
    }
    myAlphaTable[0][0] = 1.0;
    myAlphaTable[0][1] = 1.0;
    myBetaTable[0][0] = 0.0;
    myBetaTable[0][1] = 0.0;
    myGammaTable[0][0] = 0.0;
    myGammaTable[0][1] = 0.0;
    

/* 
//  get down to business...................................
*/

    if (localRank == masterRank){
        infd = fopen(argv[1], "r");
        if (!infd) {
            printf("ERROR opening input file -- exiting...\n");
            exit(0);
        }
    }


/* 
//----------------------------------------------------------------
//  initialization process section
//----------------------------------------------------------------
*/


/* 
//  initialization call for the procedural driver, to specify the
//  number of linear systems that will be solved
*/

	mySysNum = 0;
	numLinearSystems(mySysNum+1, MPI_COMM_WORLD, 0);

/* 
//  read the overall beam discretization initialization data (here,
//  just the number of elements to divide the beam into...)
*/

    if (localRank == masterRank) {
        fscanf(infd, "%d", &myNumElements);     /* how many elements to use */
        myNumElemBlocks = numProcs;
        myNumNodes = myNumElements + 1;

        assert (myNumElements > 0);
        assert (myNumElemBlocks > 0);

/* 
//  create global structures to help map blocks to processors
*/

        elemsPerBlock = malloc(myNumElemBlocks*sizeof(int));
        startElem = malloc(myNumElemBlocks*sizeof(GlobalID));
        endElem = malloc(myNumElemBlocks*sizeof(GlobalID));

        nodesPerBlock = malloc(myNumElemBlocks*sizeof(int));
        startNode = malloc(myNumElemBlocks*sizeof(GlobalID));
        endNode = malloc(myNumElemBlocks*sizeof(GlobalID));

/*
//  map the elements to the blocks (== processors) more-or-less evenly
//  and populate the load-balancing data structures
*/

        block_strategy(myNumElements, myNumElemBlocks, elemsPerBlock);

/*
//  form the appropriate block (== processor) data structures - this step
//  is done once and for all on the master CPU, and the results of this
//  global calculation are sent to the slave CPU's...
*/

        sumElem = 0;
        for (i = 0; i < myNumElemBlocks; i++) {
            startElem[i] = sumElem;
            endElem[i] = startElem[i] + elemsPerBlock[i] - (GlobalID)1;
            startNode[i] = startElem[i];
            endNode[i] = endElem[i] + (GlobalID)1;
            nodesPerBlock[i] = elemsPerBlock[i] + 1;
            sumElem += elemsPerBlock[i];
        }

/* 
//  ship all this overhead data to the various processors and then
//  the real parallel computation can get done...
*/

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
 
/* 
//  create masterRank local copies of partitioning parameters
*/

        numProcessors = myNumElemBlocks;
        localStartElem = startElem[0];
        localEndElem = endElem[0];
        localStartNode = startNode[0];
        localEndNode = endNode[0];
        localElemsPerBlock = elemsPerBlock[0];
        localNodesPerBlock = nodesPerBlock[0];
       
    }       /* end the "if (localRank == masterRank)" block */

    else {

/* 
//  receive the data sent by the master CPU...
*/

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


/* 
//---------------------------------------------------------------------
//  begin SPMD computation, since all the global overhead has been done
//---------------------------------------------------------------------
*/

/* 
//  redefine the element and node sizes for local storage instead of
//  the global storage required for initialization
*/
    
/* 
//  create local list of element and node objects
*/

    myLocalElements = malloc(localElemsPerBlock*sizeof(FE_Elem));
    myLocalNodes = malloc(localNodesPerBlock*sizeof(FE_Node));

/* 
//  load up the element and node objects with appropriate data for this
//  particular problem (namely, a uniform 1D beam with a fixed left end
//  and a uniform transverse load... here, the parameters EI, etc., have
//  been chosen to make the tip displacement/rotation work out to be nice
//  whole numbers)
*/

    EI = 100.0;
    qLoad = 12.0;
    elemLength = 10.0/myNumElements;

    myElemCount = 0;
    myNodeCount = 0;
    for (j = 0; j < localElemsPerBlock; j++) {
        jGlobal = localStartElem + j;
        myElemPtr = &myLocalElements[j];
        cfn_FE_Elem_FE_Elem(myElemPtr);
        cfn_FE_Elem_putGlobalElemID(myElemPtr, jGlobal);
        cfn_FE_Elem_putNumElemRows(myElemPtr, 4);
        cfn_FE_Elem_putNumElemNodes(myElemPtr, 2);
        cfn_FE_Elem_putElemLength(myElemPtr, elemLength);
        cfn_FE_Elem_putBendStiff(myElemPtr, EI);
        cfn_FE_Elem_putFoundStiff(myElemPtr, 0.0);
        cfn_FE_Elem_putDistLoad(myElemPtr, qLoad);
        cfn_FE_Elem_allocateNodeList(myElemPtr);
        cfn_FE_Elem_allocateElemForces(myElemPtr);
        cfn_FE_Elem_allocateLoad(myElemPtr);
        cfn_FE_Elem_allocateStiffness(myElemPtr);

#ifdef dbgTrace
        cfn_FE_Elem_dumpToScreen(myElemPtr);
#endif

    }

    for (j = 0; j < localNodesPerBlock; j++) {
        jGlobal = localStartNode + j;
        myNodePtr = &myLocalNodes[j];
        cfn_FE_Node_putGlobalNodeID(myNodePtr, jGlobal);
        cfn_FE_Node_putNumNodalDOF(myNodePtr, 2);
        cfn_FE_Node_putNodePosition(myNodePtr, jGlobal*elemLength);
        cfn_FE_Node_allocateSolnList(myNodePtr);

#ifdef dbgTrace
        cfn_FE_Node_dumpToScreen(myNodePtr);
#endif

    }
    
/* 
//  construct some saved shared node tables to simplify passing shared 
//  node data to the interface later on... each internal block has 2 shared
//  nodes to deal with, while the end blocks only have 1.  Each shared node
//  is shared between two processors, either on the left, right, or both.
*/
 
/* 
//  first, if this is a serial problem, then there aren't any shared nodes
*/

    if (numProcessors == 1) {         
        numSharedNodes = 0;
    }
    else if (localRank == 0) {
        numSharedNodes = 1;
        listSharedNodes = malloc(numSharedNodes*sizeof(GlobalID));
        listSharedProcs = malloc(numSharedNodes*sizeof(int*));
        listSharedProcs[0] = malloc(2*sizeof(int));
        listSharedNodes[0] = localEndNode;
        listSharedProcs[0][0] = localRank;
        listSharedProcs[0][1] = localRank + 1;
    }
    else if (localRank == (numProcessors - 1)) {
        numSharedNodes = 1;
        listSharedNodes = malloc(numSharedNodes*sizeof(GlobalID));
        listSharedProcs = malloc(numSharedNodes*sizeof(int*));
        listSharedProcs[0] = malloc(2*sizeof(int));
        listSharedNodes[0] = localStartNode;
        listSharedProcs[0][0] = localRank - 1;
        listSharedProcs[0][1] = localRank;
    }
    else {
        numSharedNodes = 2;
        listSharedNodes = malloc(numSharedNodes*sizeof(GlobalID));
        listSharedProcs = malloc(numSharedNodes*sizeof(int*));
        listSharedProcs[0] = malloc(2*sizeof(int));
        listSharedProcs[1] = malloc(2*sizeof(int));
        listSharedNodes[0] = localStartNode;
        listSharedNodes[1] = localEndNode;
        listSharedProcs[0][0] = localRank - 1;
        listSharedProcs[0][1] = localRank;
        listSharedProcs[1][0] = localRank;
        listSharedProcs[1][1] = localRank + 1;
    }


/* 
//  FE interface call ----------------------------------------------
*/
    myNumElemBlocks = 1;   /* // temporary pll driver assumption, 1 per proc!! */
    
    myErrorCode = initSolveStep(mySysNum, 
                                myNumElemBlocks, 
                                mySolvType); 
    if (myErrorCode) printf("error in initSolveStep\n");

/* 
//  FE interface call ----------------------------------------------
*/
    myErrorCode = initFields(mySysNum, 
                             myNumFields, 
                             myCardFields,
                             myFieldIDs); 
    if (myErrorCode) printf("error in linearSystem.initFields\n");



/* 
//  pass the block control parameter data for a beam
*/

    myElemBlockID = (GlobalID) localRank;
    myNumElems = localElemsPerBlock;
    myNumElemTotal = myNumElems;
    assert (myNumElems > 0);

/* 
//  FE interface call ----------------------------------------------
*/

    myElemFieldIDs = malloc(2*sizeof(int));      /* 2 nodes to identify fields for */
    myElemFieldIDs[0] = malloc(1*sizeof(int));   /* 1 field at each node */
    myElemFieldIDs[1] = malloc(1*sizeof(int));
    myElemFieldIDs[0][0] = 5;
    myElemFieldIDs[1][0] = 5;

    myErrorCode = beginInitElemBlock(mySysNum, 
                                     myElemBlockID, 
                                     myNumNodesPerElem,
                                     myNumElemFields,
                                     myElemFieldIDs,
                                     myStrategy,
                                     myStrategy,
                                     myNumElemDOF, 
                                     myNumElemSets,
                                     myNumElemTotal);
    if (myErrorCode) printf("error in beginInitElemBlock\n");

/* 
//  for now, to keep things simple, just use one workset per block...
*/

    myNumElems = localElemsPerBlock;
    assert (myNumElems > 0);
    myElemIDs = malloc(myNumElems*sizeof(GlobalID));
    myElemConn = malloc(myNumElems*sizeof(GlobalID*));

    for (k = 0; k < myNumElems; k++) {
        myElemIDs[k] = (GlobalID) (localStartElem + k);
        myElemConn[k] = malloc(myNumNodesPerElem*sizeof(GlobalID));
        myElemConn[k][0] = myElemIDs[k];
        myElemConn[k][1] = myElemIDs[k] + 1;
    }

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = initElemSet(mySysNum, myNumElems, myElemIDs, 
                                           myElemConn);
    if (myErrorCode) printf("error in initElemSet\n");


    free(myElemIDs);
    for (k = 0; k < myNumElems; k++) {
       free(myElemConn[k]);
    }
    free(myElemConn);


/* 
//  close the element workset initialization phase
*/

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = endInitElemBlock(mySysNum);
    if (myErrorCode) printf("error in endInitElemBlock\n");

/* 
//  (recall that there is only one block per processor here, by definition)
*/


/* 
//  pass the node list control initialization data first
*/

/* 
//  FE interface call ----------------------------------------------
*/
    
    if (numProcessors == 1) {         
        myNumSharedNodeSets = 0;
    }
    else {
        myNumSharedNodeSets = 1;
    }
    myErrorCode = beginInitNodeSets(mySysNum, 
                                    myNumSharedNodeSets,
                                    myNumExtNodeSets);
    if (myErrorCode) printf("error in beginInitNodeSets\n");


/* 
//  pass the shared node data (each shared node is shared between two CPU's)
*/

    for (i = 0; i < myNumSharedNodeSets; i++) {
        myLenSharedNodeSet = numSharedNodes;
        mySharedNodes = malloc(myLenSharedNodeSet*sizeof(GlobalID));
        myLenSharedProcIDs = malloc(myLenSharedNodeSet*sizeof(int));
        mySharedProcIDs = malloc(myLenSharedNodeSet*sizeof(int*));

        printf("procID : %d\n", localRank);
        for (j = 0; j < myLenSharedNodeSet; j++) {
            mySharedNodes[j] = listSharedNodes[j];
            myLenSharedProcIDs[j] = 2;
            mySharedProcIDs[j] = malloc(myLenSharedProcIDs[j]*sizeof(int));
            printf("    mySharedNodes[j]      : %d\n", (int)mySharedNodes[j]);
            printf("    myLenSharedProcIDs[j] : %d\n", myLenSharedProcIDs[j]);
            printf("    mySharedProcIDs       : ");
            for (k = 0; k < myLenSharedProcIDs[j]; k++) {
                mySharedProcIDs[j][k] = listSharedProcs[j][k];
                printf("%d   ", mySharedProcIDs[j][k]);
            }
        printf("\n\n");
        }

/* 
//  FE interface call ----------------------------------------------
*/

		myErrorCode = initSharedNodeSet(mySysNum, mySharedNodes,
                                	             myLenSharedNodeSet,
                                	             mySharedProcIDs,
                                	             myLenSharedProcIDs);
		if (myErrorCode) printf("error in initSharedNodeSet\n");

		free(mySharedNodes);
		for (j = 0; j < myLenSharedNodeSet; j++) {
            	free(mySharedProcIDs[j]);
		}
		free(mySharedProcIDs);
    }


/* 
//  close the node list initialization phase
*/

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = endInitNodeSets(mySysNum);
    if (myErrorCode) printf("error in endInitNodeSets\n");


/* 
//  read and store the constraint relation control data
*/

    myNumCRMultSets = 0;
    myNumCRPenSets = 0;

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = beginInitCREqns(mySysNum, myNumCRMultSets, myNumCRPenSets); 
    if (myErrorCode) printf("error in beginInitCREqns\n");

/* 
//  close the constraint relation initialization phase
*/
 
/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = endInitCREqns(mySysNum);
    if (myErrorCode) printf("error in endInitCREqns\n");


/* 
//  close the finite-element data initialization phase
*/


    MPI_Barrier(MPI_COMM_WORLD); 
	printf("processor %d ready to call initComplete\n", localRank); 
    MPI_Barrier(MPI_COMM_WORLD);

/* 
//  FE interface call ----------------------------------------------
*/


    myErrorCode = initComplete(mySysNum);
    if (myErrorCode) printf("error in initComplete\n");
    

    printf("processor %d back from initComplete\n", localRank);
    MPI_Barrier(MPI_COMM_WORLD);



/* 
//----------------------------------------------------------------
//  load process section
//----------------------------------------------------------------
*/

/* 
//  begin node-set data load step
*/

/* 
//  load the essential boundary condition tabular data vales - the essential
//  bc at the left support (x = 0) only applies to the master CPU...
*/

/* 
//  FE interface call ----------------------------------------------
*/

    if (localRank != masterRank) {
        myNumBCNodeSets = 0;
    }
    
    myErrorCode = beginLoadNodeSets(mySysNum, myNumBCNodeSets);
    if (myErrorCode) printf("error in beginLoadNodeSets\n");

/* 
//  FE interface call ----------------------------------------------
*/

    if (localRank == masterRank) {
        myErrorCode = loadBCSet(mySysNum, 
                                myBCNodeSet, 
                                myLenBCNodeSet,
                                myBCFieldID,
                                myAlphaTable, 
                                myBetaTable, 
                                myGammaTable);
        if (myErrorCode) printf("error in loadBCSet\n");
    }


/* 
//  end node-set data load step
*/

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = endLoadNodeSets(mySysNum);
    if (myErrorCode) printf("error in endLoadNodeSets\n");


/* 
//  begin blocked-element (i.e., processor-based) data loading step
*/

    myElemBlockID = (GlobalID) localRank;
    myNumElems = localElemsPerBlock;
    myNumElemTotal = myNumElems;
    assert (myNumElems > 0);

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = beginLoadElemBlock(mySysNum, 
                                     myElemBlockID,
                                     myNumElemSets,
                                     myNumElemTotal);
    if (myErrorCode) printf("error in beginLoadElemBlock\n");

/* 
//  assemble the element matrix data assuming one workset per block
*/

    myNumElems = localElemsPerBlock;
    assert (myNumElems > 0);
    myElemIDs = malloc(myNumElems*sizeof(GlobalID));
    myElemConn = malloc(myNumElems*sizeof(GlobalID*));

    for (j = 0; j < myNumElems; j++) {
	    myElemIDs[j] = (GlobalID) (localStartElem + j);
        myElemConn[j] = malloc(myNumNodesPerElem*sizeof(GlobalID));
	    myElemConn[j][0] = myElemIDs[j];
	    myElemConn[j][1] = myElemIDs[j] + 1;
    }

    myElemFormat = 0;       /*  // not worrying about format -yet- !!!  */
    myElemLoads = malloc(myNumElems*sizeof(double*));
    myElemStiffnesses = malloc(myNumElems*sizeof(double**));

/* 
//  evaluate the element load vector and stiffness matrices for this workset
*/

    myElemRows = 4;
    for (j = 0; j < myNumElems; j++) {
        myElemPtr = &myLocalElements[j];
        myElemLoads[j] = malloc(myElemRows*sizeof(double));
        myElemStiffnesses[j] = malloc(myElemRows*sizeof(double*));
        for (m = 0; m < myElemRows; m++) {
            myElemStiffnesses[j][m] = malloc(myElemRows*sizeof(double));
        }
        myElemLoads[j] = cfn_FE_Elem_evaluateLoad(myElemPtr, &neRows);
        assert(neRows == myElemRows);
        myElemStiffnesses[j] = cfn_FE_Elem_evaluateStiffness(myElemPtr, &neRows);
        assert(neRows == myElemRows);
    }

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = loadElemSet(mySysNum, 
                              myElemSetID, 
                              myNumElems, 
                              myElemIDs, 
                              myElemConn,
                              myElemStiffnesses, 
                              myElemLoads, 
                              myElemFormat);
    if (myErrorCode) printf("error in loadElementData\n");


#ifdef dbgTrace
	printf("\nelement dump after completion of load step\n");
    for (k = 0; k < localElemsPerBlock; k++) {
        cfn_FE_Elem_dumpToScreen(&myLocalElements[k]);
    }

	printf("\nnodal dump after completion of load step\n");
    for (k = 0; k < localNodesPerBlock; k++) {
		cfn_FE_Node_dumpToScreen(&myLocalNodes[k]);
    }
#endif


    for (k = 0; k < myNumElems; k++) {
        free(myElemConn[k]);
        free(myElemLoads[k]);
        for (j = 0; j < myElemRows; j++) {
            free(myElemStiffnesses[k][j]);
        }
        free(myElemStiffnesses[k]);
    }
    free(myElemStiffnesses);

    free(myElemIDs);
    free(myElemConn);
    free(myElemLoads); 

/* 
//  close the element workset data load phase
*/


/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = endLoadElemBlock(mySysNum);
    if (myErrorCode) printf("error in endLoadElemBlock\n");


/* 
//  begin constraint relation data load step
*/

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = beginLoadCREqns(mySysNum, myNumCRMultSets, myNumCRPenSets);
    if (myErrorCode) printf("error in beginLoadCREqns\n");

 
/* 
// end constraint relation data load step 
*/

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = endLoadCREqns(mySysNum);
    if (myErrorCode) printf("error in endLoadCREqns\n");
		
    if (myNumCRMultSets > 0) {
        free(myCRMultIDList);
    }

    if (myNumCRPenSets > 0) {
        free(myCRPenIDList);
    }
 
         
/* 
// end of overall data loading sequence
*/

/* 
//  FE interface call ----------------------------------------------
*/

    myErrorCode = loadComplete(mySysNum);
    if (myErrorCode) printf("error in loadComplete\n");

    MPI_Barrier(MPI_COMM_WORLD);
    printf("processor %d finished loadComplete\n", localRank);
    MPI_Barrier(MPI_COMM_WORLD);


/* 
//  all done with the input file at this point
*/

    if (localRank == masterRank) fclose(infd);


/* 
//----------------------------------------------------------------
//  solution process section
//----------------------------------------------------------------
*/

    for(ii=0; ii<numParams; ii++) paramStrings[ii] = malloc(64*sizeof(char));

    strcpy(paramStrings[0], "solver qmr");
    strcpy(paramStrings[1], "preconditioner diagonal");
    strcpy(paramStrings[2], "maxIterations 50000");
    strcpy(paramStrings[3], "tolerance 1.e-10");
    strcpy(paramStrings[4], "rowScale false");
    strcpy(paramStrings[5], "colScale false");


    parameters(mySysNum, numParams, paramStrings);

    for (ii = 0; ii < numParams; ii++) 
        free(paramStrings[ii]);
    free(paramStrings);


/* 
//  FE interface call ----------------------------------------------
*/

    iterateToSolve(mySysNum);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("processor %d finished iterateToSolve\n", localRank);
    MPI_Barrier(MPI_COMM_WORLD);

/* 
//----------------------------------------------------------------
//  solution return process section
//----------------------------------------------------------------
*/

/* 
//  check the solution return functions...
*/
    
    j = localRank;

    myNumBlkActNodes = getNumBlockActNodes(mySysNum, j);
    myNumBlkActEqns = getNumBlockActEqns(mySysNum, j);
    myNodeList = malloc(myNumBlkActNodes*sizeof(GlobalID));
    mySolnOffsets = malloc(myNumBlkActNodes*sizeof(int));
    mySolnValues = malloc(myNumBlkActEqns*sizeof(double));

/* 
//  FE interface call ----------------------------------------------
*/
    getBlockNodeSolution(mySysNum, j, myNodeList, &myLenList,
                         mySolnOffsets, mySolnValues);

    assert (myNumBlkActNodes == myLenList);

	printf("\nSolution Vector\n");
    for (k = 0; k < myLenList; k++) {
    	j = mySolnOffsets[k];
		printf(" %d   %d   %d    %f\n", k, j, (int)myNodeList[k], mySolnValues[j]);
    }



    free(myNodeList);
    free(mySolnOffsets);
    free(mySolnValues);

/* 
//  clean up everything...
*/
    
    free(myLocalElements);
    free(myLocalNodes);
    free(elemsPerBlock);


    printf("Calling MPI_Finalize...\n");
    MPI_Finalize();

    printf("...now exiting...\n");
    return(1);
}


/* 
//================================================================================
//
//  strategy for blocking a 1D mesh composed of numElems elements
//  into p blocks, such that the number of elements per block does
//  not vary by more than one
//
*/

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


/**=========================================================================**/
void pgm_init(int argc, char **argv, int *num_procs, int *my_rank,
    int *master_rank) {

/*//  Parallel program initialization.
*/
/*//  Perform MPI initialization.
*/
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, num_procs);

/*//  Let everybody know who the master proc is.
*/
    *master_rank = 0;
    MPI_Bcast(master_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return;
}

/* 
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
*/

