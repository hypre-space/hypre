//
//This is a class that will exercise FEI implementations.
//

#include <stdlib.h>
#include <math.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "fei.h"

#include "Poisson_Elem.h"

#include "PoissonData.h"

static int int_sqrt(int x) {
//a not-safe function for taking the sqrt of a square int.
    return((int)ceil(sqrt((double)x)));
}

//==============================================================================
PoissonData::PoissonData(int L, int W, int DOF, 
                       int numProcs, int localProc, int outputLevel) {
//
//PoissonData constructor.
//
//Arguments:
//
// L:                global bar length (number-of-elements)
// W:                bar width
// DOF:              degrees-of-freedom per node.
// numProcs:         number of processors participating in this FEI test.
// localProc:        local processor number.
// fei:              the FEI instance to be exercised.
//

    L_ = L;
    W_ = W;
    DOF_ = DOF;
    
    startElement_ = 0;
    numLocalElements_ = 0;

    numProcs_ = numProcs;
    localProc_ = localProc;
    outputLevel_ = outputLevel;

    check1();

    fieldArraysAllocated_ = false;
    elemsAllocated_ = false;
    elemIDsAllocated_ = false;
    elemConnAllocated_ = false;
    elemStiffAllocated_ = false;
    elemLoadAllocated_ = false;
    sharedNodeIDsAllocated_ = false;
    sharedNodeProcsAllocated_ = false;

    BCNodeIDsAllocated_ = false;
    BCArraysAllocated_ = false;

    fei_has_been_set = false;

    numFields_ = NULL;
    fieldIDs_ = NULL;

    elemIDs_ = NULL;
    elemConn_ = NULL;
    elemStiff_ = NULL;
    elemLoad_ = NULL;

    sharedNodeIDs_ = NULL;
    numSharedNodes_ = 0;
    sharedNodeProcs_ = NULL;
    numSharedNodeProcs_ = NULL;

    BCNodeIDs_ = NULL;
    numBCNodes_ = 0;
    alpha_ = NULL;
    beta_ = NULL;
    gamma_ = NULL;

    calculateDistribution();

    initializeElements();

    numElemBlocks_ = 1;
    elemBlockID_ = (GlobalID)localProc_;
    elemSetID_ = 0;
    elemFormat_ = 0;

    nodesPerElement_ = 4;
    fieldsPerNode_ = 1;

    initializeFieldStuff();

}

//==============================================================================
PoissonData::~PoissonData() {
//
//Destructor -- delete any allocated memory.
//

    deleteFieldArrays();

    if (elemsAllocated_) delete [] elems_;
    if (elemIDsAllocated_) delete [] elemIDs_;
    if (elemConnAllocated_) delete [] elemConn_;
    if (elemStiffAllocated_) delete [] elemStiff_;
    if (elemLoadAllocated_) delete [] elemLoad_;

    deleteSharedNodeArrays();

    deleteBCArrays();
}

//==============================================================================
void PoissonData::print_storage_summary() {

    if (localProc_ != 0) return;

    int tmp1 = 4*DOF_;
    int elemDStorage = tmp1*(tmp1+1);
    int elemIStorage = 11;
    double totalMbytes = (numLocalElements_ * elemDStorage * sizeof(double) +
                      numLocalElements_ * elemIStorage * sizeof(int))/1.e+6;

    cout << "PoissonData storage summary: " << endl
         << "    Elements per processor: " << numLocalElements_ << endl
         << "    Storage per element: " << elemDStorage << " doubles, "
                                        << elemIStorage << " ints" << endl;
    cout << " (connectivity, stiffness and load arrays, misc variables)"
         << endl << endl;
    cout << "    Total Mbytes required per processor: " << totalMbytes << endl;

    cout <<endl<< "Note: this does not include the memory that gets allocated"
         <<endl<< "within the FEI object. FEI storage (including the underlying"
         <<endl<< "sparse matrix, etc.) should roughly equal the storage in"
         <<endl<< "this PoissonData object."
         <<endl;

    cout << "========================================================" << endl;
    cout << endl;
}

//==============================================================================
void PoissonData::set_FEI_instance(FEI& fei) {
    fei_ = &fei;
    fei_has_been_set = true;
}

//==============================================================================
void PoissonData::do_init_phase() {

    if (!fei_has_been_set) messageAbort("FEI instance hasn't been set.");

    int err = 0;

    if (outputLevel_>0) cout << "initFields" << endl;
    err = fei_->initFields(numFields_[0], &fieldSize_, &fieldIDs_[0][0]);
    if (err) messageAbort("error in fei_->initFields.");

    if (outputLevel_>0) cout << "beginInitElemBlock" << endl;
    err = fei_->beginInitElemBlock((GlobalID)localProc_, //blockID
                             nodesPerElement_, //nodes-per-element
                             numFields_, //fields-per-node
                             fieldIDs_, //element fieldIDs
                             0, //interleaving strategy (only 0 is valid)
                             0, //lumping strategy (only 0 is valid)
                             0, //number of element-DOF
                             1, //number of element-sets
                             numLocalElements_);
    if (err) messageAbort("error in fei_->beginInitElemBlock.");

    if (outputLevel_>0) cout << "   initElemSet" << endl;
    err = fei_->initElemSet(numLocalElements_, elemIDs_, elemConn_);
    if (err) messageAbort("error in fei_->initElemSet.");

    if (outputLevel_>0) cout << "endInitElemBlock" << endl;
    err = fei_->endInitElemBlock();
    if (err) messageAbort("error in fei_->endInitElemBlock.");

    numSharedNodeSets_ = 4;
    if (outputLevel_>0) cout << "beginInitNodeSets" << endl;
    err = fei_->beginInitNodeSets(numSharedNodeSets_,
                            0); //no external node sets
    if (err) messageAbort("error in fei_->beginInitNodeSets.");

    allocateSharedNodeArrays();

    int numSharedLeft = 0;
    getLeftSharedNodes(numSharedLeft);

    if (outputLevel_>1)
        printSharedNodes("left", numSharedLeft, sharedNodeIDs_,
                     sharedNodeProcs_, numSharedNodeProcs_);

    if (outputLevel_>0) cout << "   initSharedNodeSet" << endl;
    err = fei_->initSharedNodeSet(sharedNodeIDs_, numSharedLeft,
                                  sharedNodeProcs_, numSharedNodeProcs_);
    if (err) messageAbort("error in fei_->initSharedNodeSet.");

    int numSharedRight = 0;
    getRightSharedNodes(numSharedRight);

    if (outputLevel_>1)
        printSharedNodes("right", numSharedRight, sharedNodeIDs_,
                     sharedNodeProcs_, numSharedNodeProcs_);

    if (outputLevel_>0) cout << "   initSharedNodeSet" << endl;
    err = fei_->initSharedNodeSet(sharedNodeIDs_, numSharedRight,
                                  sharedNodeProcs_, numSharedNodeProcs_);
    if (err) messageAbort("error in fei_->initSharedNodeSet.");

    int numSharedBottom = 0;
    getBottomSharedNodes(numSharedBottom);

    if (outputLevel_>1)
        printSharedNodes("bottom", numSharedBottom, sharedNodeIDs_,
                     sharedNodeProcs_, numSharedNodeProcs_);

    if (outputLevel_>0) cout << "   initSharedNodeSet" << endl;
    err = fei_->initSharedNodeSet(sharedNodeIDs_, numSharedBottom,
                                  sharedNodeProcs_, numSharedNodeProcs_);
    if (err) messageAbort("error in fei_->initSharedNodeSet.");

    int numSharedTop = 0;
    getTopSharedNodes(numSharedTop);

    if (outputLevel_>1)
        printSharedNodes("top", numSharedTop, sharedNodeIDs_,
                     sharedNodeProcs_, numSharedNodeProcs_);

    if (outputLevel_>0) cout << "   initSharedNodeSet" << endl;
    err = fei_->initSharedNodeSet(sharedNodeIDs_, numSharedTop,
                                  sharedNodeProcs_, numSharedNodeProcs_);
    if (err) messageAbort("error in fei_->initSharedNodeSet.");

    deleteSharedNodeArrays();

    if (outputLevel_>0) cout << "endInitNodeSets" << endl;
    err = fei_->endInitNodeSets();
    if (err) messageAbort("error in fei_->endInitNodeSets.");
    if (outputLevel_>0) cout << "initComplete" << endl;
    err = fei_->initComplete();
    if (err) messageAbort("error in fei_->initComplete.");
}

//==============================================================================
void PoissonData::do_load_phase() {

    if (!fei_has_been_set) messageAbort("FEI instance hasn't been set.");

    int err = 0;

    numBCNodeSets_ = 1;

    if (outputLevel_>0) cout << "beginLoadNodeSets" << endl;
    err = fei_->beginLoadNodeSets(numBCNodeSets_);
    if (err) messageAbort("error in fei_->beginLoadNodeSets.");

    deleteBCArrays();

    numBCNodes_ = 0;
    calculateBCs();
    if (outputLevel_>1) printBCs();

    if (outputLevel_>0) cout << "   loadBCSet" << endl;
    err = fei_->loadBCSet(BCNodeIDs_, numBCNodes_, fieldIDs_[0][0],
                          alpha_, beta_, gamma_);
    if (err) messageAbort("error in fei_->loadBCSet.");

    deleteBCArrays();

    if (outputLevel_>0) cout << "endLoadNodeSets" << endl;
    err = fei_->endLoadNodeSets();
    if (err) messageAbort("error in fei_->endLoadNodeSets.");

    if (outputLevel_>0) cout << "beginLoadElemBlock" << endl;
    err = fei_->beginLoadElemBlock(elemBlockID_,
                                   1, //number of element sets
                                   numLocalElements_);
    if (err) messageAbort("error in fei_->beginLoadElemBlock.");

    if (outputLevel_>0) cout << "   loadElemSet" << endl;
    err = fei_->loadElemSet(elemSetID_, numLocalElements_,
                            elemIDs_, elemConn_,
                            elemStiff_, elemLoad_, elemFormat_);
    if (err) messageAbort("error in fei_->loadElemSet.");

    if (outputLevel_>0) cout << "endLoadElemBlock" << endl<<endl;
    err = fei_->endLoadElemBlock();
    if (err) messageAbort("error in fei_->endLoadElemBlock.");

    if (outputLevel_>0) cout << "loadComplete" << endl<<endl;
    err = fei_->loadComplete();
    if (err) messageAbort("error in fei_->loadComplete.");
}

//==============================================================================
void PoissonData::do_load_BC_phase() {

    if (!fei_has_been_set) messageAbort("FEI instance hasn't been set.");

    int err = 0;

    numBCNodeSets_ = 1;

    if (outputLevel_>0) cout << "beginLoadNodeSets" << endl;
    err = fei_->beginLoadNodeSets(numBCNodeSets_);
    if (err) messageAbort("error in fei_->beginLoadNodeSets.");

    deleteBCArrays();

    numBCNodes_ = 0;
    calculateBCs();
    if (outputLevel_>1) printBCs();

    if (outputLevel_>0) cout << "   loadBCSet" << endl;
    err = fei_->loadBCSet(BCNodeIDs_, numBCNodes_, fieldIDs_[0][0],
                          alpha_, beta_, gamma_);
    if (err) messageAbort("error in fei_->loadBCSet.");

    deleteBCArrays();

    if (outputLevel_>0) cout << "endLoadNodeSets" << endl;
    err = fei_->endLoadNodeSets();
    if (err) messageAbort("error in fei_->endLoadNodeSets.");
}

//==============================================================================
void PoissonData::do_begin_load_block() {
    
    if (!fei_has_been_set) messageAbort("FEI instance hasn't been set.");
    
    int err = 0;
    if (outputLevel_>0) cout << "beginLoadElemBlock" << endl;
    err = fei_->beginLoadElemBlock(elemBlockID_,
                                   1, //number of element sets
                                   numLocalElements_);
    if (err) messageAbort("error in fei_->beginLoadElemBlock.");
}

//==============================================================================
void PoissonData::do_load_stiffness() {

    if (!fei_has_been_set) messageAbort("FEI instance hasn't been set.");

    int err = 0;
    if (outputLevel_>0) cout << "   loadElemSetMatrix" << endl;
    err = fei_->loadElemSetMatrix(elemSetID_, numLocalElements_,
                                  elemIDs_, elemConn_,
                                  elemStiff_, elemFormat_);
    if (err) messageAbort("error in fei_->loadElemSetMatrix.");
}

//==============================================================================
void PoissonData::do_load_rhs() {
 
    if (!fei_has_been_set) messageAbort("FEI instance hasn't been set.");

    int err = 0;
    if (outputLevel_>0) cout << "   loadElemSetRHS" << endl;
    err = fei_->loadElemSetRHS(elemSetID_, numLocalElements_,
                               elemIDs_, elemConn_,
                               elemLoad_);
    if (err) messageAbort("error in fei_->loadElemSetRHS.");
}

//==============================================================================
void PoissonData::do_end_load_block() {
    
    if (!fei_has_been_set) messageAbort("FEI instance hasn't been set.");
    
    int err = 0;
    if (outputLevel_>0) cout << "endLoadElemBlock" << endl;
    err = fei_->endLoadElemBlock();
    if (err) messageAbort("error in fei_->endLoadElemBlock.");
}

//==============================================================================
void PoissonData::do_loadComplete() {

    if (!fei_has_been_set) messageAbort("FEI instance hasn't been set.");

    int err = 0;
    if (outputLevel_>0) cout << "loadComplete" << endl<<endl;
    err = fei_->loadComplete();
    if (err) messageAbort("error in fei_->loadComplete.");
}

//==============================================================================
void PoissonData::check1() {
//
//Private function to be called from the constructor, simply makes sure that
//the constructor's input arguments were reasonable.
//
//If they aren't, a message is printed on standard err, and abort() is called.
//
    if (L_ <= 0)                 messageAbort("bar length L <= 0.");
    if (W_ <= 0)                 messageAbort("bar width W <= 0.");
    if (L_ != W_)                messageAbort("bar width must equal length.");
    if (DOF_ != 1)               messageAbort("nodal DOF must equal 1.");
    if (numProcs_ <= 0)          messageAbort("numProcs <= 0.");
    if (L_%int_sqrt(numProcs_)) 
        messageAbort("L must be an integer multiple of sqrt(numProcs).");
    if (localProc_ < 0)          messageAbort("localProc < 0.");
    if (localProc_ >= numProcs_) messageAbort("localProc >= numProcs.");
    if (outputLevel_ < 0)        messageAbort("outputLevel < 0.");
}

//==============================================================================
void PoissonData::calculateDistribution() {
//
//Calculate which elements this processor owns. The element domain is a
//square, and we can assume that sqrt(numProcs_) divides evenly into
//L_. We're working with a (logically) 2D processor arrangement.
//Furthermore, the logical processor layout is such that processor 0 is at
//the bottom left corner of a 2D grid, and a side of the grid is of length
//sqrt(numProcs_). The element domain is numbered such that element 1 is at
//the bottom left corner of the square, and element numbers increase from
//left to right. i.e., element 1 is in position (1,1), element L is in
//position (1,L), element L+1 is in position (2,1).
//
//Use 1-based numbering for the elements and the x- and y- coordinates in
//the element grid, but use 0-based numbering for processor IDs and the
//coordinates in the processor grid.
//
    numLocalElements_ = (L_*W_)/numProcs_;

    elemIDs_ = new GlobalID[numLocalElements_];
    if (!elemIDs_) messageAbort("ERROR allocating elemIDs_.");
    elemIDsAllocated_ = true;

    //0-based x-coordinate of this processor in the 2D processor grid.
    procX_ = localProc_%int_sqrt(numProcs_);

    //0-based maximum processor x-coordinate.
    maxProcX_ = int_sqrt(numProcs_) - 1;

    //0-based y-coordinate of this processor in the 2D processor grid.
    procY_ = localProc_/int_sqrt(numProcs_);

    //0-based maximum processor y-coordinate.
    maxProcY_ = int_sqrt(numProcs_) - 1;

    int sqrtElems = int_sqrt(numLocalElements_);
    int sqrtProcs = int_sqrt(numProcs_);

    //1-based first-element-on-this-processor
    startElement_ = 1 + procY_*sqrtProcs*numLocalElements_ + procX_*sqrtElems;

    if (outputLevel_>1) {
        cout << localProc_ << ", calcDist.: numLocalElements: " 
             << numLocalElements_ << ", startElement: " << startElement_ 
             << endl;
        cout << localProc_ << ", procX: " << procX_ << ", procY_: " << procY_
             << ", maxProcX: " << maxProcX_ << ", maxProcY: " << maxProcY_
             << endl;
    }

    int offset = 0;
    for(int i=0; i<sqrtElems; i++) {
        for(int j=0; j<sqrtElems; j++) {
            elemIDs_[offset] = (GlobalID)(startElement_ + i*L_ + j);
            offset++;
        }
    }
}

//==============================================================================
void PoissonData::messageAbort(char* message) {
    cerr << endl << "PoissonData: " << message 
         << endl << "  Aborting." << endl;
    abort();
}

//==============================================================================
void PoissonData::initializeElements() {
//
//This function allocates the array of local elements, and initializes
//their contents.
//
    elems_ = new Poisson_Elem[numLocalElements_];
    if (!elems_) messageAbort("initializeElements: allocation error.");
    elemsAllocated_ = true;

    elemConn_ = new int*[numLocalElements_];
    if (!elemConn_) messageAbort("initializeElements: allocation error.");
    elemConnAllocated_ = true;

    elemStiff_ = new double**[numLocalElements_];
    if (!elemStiff_) messageAbort("initializeElements: allocation error.");
    elemStiffAllocated_ = true;
    elemLoad_ = new double*[numLocalElements_];
    if (!elemLoad_) messageAbort("initializeElements: allocation error.");
    elemLoadAllocated_ = true;

    for(int i=0; i<numLocalElements_; i++){
        elems_[i].setElemID(elemIDs_[i]);
        elems_[i].setElemLength(1.0/L_);
        elems_[i].setTotalLength(1.0);

        int err = 0;
        err = elems_[i].allocateInternals(DOF_);
        if (err) messageAbort("Allocation error in element.");

        //now get a pointer to this element's connectivity array and
        //calculate that connectivity (in place).
        int size = 0;
        elemConn_[i] = elems_[i].getElemConnPtr(size);
        if (size == 0) messageAbort("initializeElements: bad conn ptr.");

        calculateConnectivity(elemConn_[i], size, elemIDs_[i]);

        elems_[i].calculateCoords();

        if (outputLevel_>1) {
            double* x = elems_[i].getNodalX(size);
            double* y = elems_[i].getNodalY(size);
            cout << localProc_ << ", elemID " << elemIDs_[i] << ", nodes: ";
            for(int j=0; j<size; j++) {
                cout << elemConn_[i][j] << " ";
                cout << "("<<x[j]<<","<<y[j]<<") ";
            }
            cout << endl;
        }

        elems_[i].calculateStiffness();
        elemStiff_[i] = elems_[i].getElemStiff(size);
        if (size==0) messageAbort("initializeElements: bad stiff ptr.");

        elems_[i].calculateLoad();
        elemLoad_[i] = elems_[i].getElemLoad(size);
        if (size==0) messageAbort("initializeElements: bad load ptr.");
    }
}

//==============================================================================
void PoissonData::calculateConnectivity(GlobalID* conn, int size,
                                        GlobalID elemID) {
//
//Calculate a single element's connectivity array -- the list of nodes
//that it 'contains'.
//
//Note that we're assuming the element is a 2D square.
//
    //elemX will be the global 'x-coordinate' of this element in the square. The
    //'origin' is the lower-left corner of the bar, which is element 1,
    //and it is in position 1,1.
    int elemX = (int)elemID%L_;
    if (elemX == 0) elemX = L_;

    //elemY will be the global (1-based) 'y-coordinate'.
    int elemY = ((int)elemID - elemX)/L_ + 1;

    //These are the four nodes for this element.
    GlobalID lowerLeft = elemID + (GlobalID)(elemY-1);
    GlobalID lowerRight = lowerLeft + (GlobalID)1;
    GlobalID upperRight = lowerRight + (GlobalID)(L_+1);
    GlobalID upperLeft = upperRight - (GlobalID)1;

    //now fill the connectivity array. We'll always fill the connectivity
    //array with the lower left node first, and then proceed counter-clockwise.
    conn[0] = lowerLeft;
    conn[1] = lowerRight;
    conn[2] = upperRight;
    conn[3] = upperLeft;
}

//==============================================================================
void PoissonData::initializeFieldStuff() {
//
//Set up the field-descriptor variables that will be passed
//to the FEI's initFields function, beginInitElemBlock function, etc.
//
//Note we're assuming 1 field, and 4-node elements.
//
    fieldSize_ = DOF_;
    numFields_ = new int[nodesPerElement_];
    fieldIDs_ = new int*[nodesPerElement_];
    for(int i=0; i<nodesPerElement_; i++) {
        numFields_[i] = fieldsPerNode_;
        fieldIDs_[i] = new int[fieldsPerNode_];
        for(int j=0; j<fieldsPerNode_; j++) {
            fieldIDs_[i][j] = fieldsPerNode_;
        }
    }
    fieldArraysAllocated_ = true;
}

//==============================================================================
void PoissonData::deleteFieldArrays() {

    if (fieldArraysAllocated_) {

        for(int i=0; i<nodesPerElement_; i++) {
            delete [] fieldIDs_[i];
        }

        delete [] fieldIDs_;
        delete [] numFields_;
    }
    fieldArraysAllocated_ = false;
}

//==============================================================================
void PoissonData::allocateSharedNodeArrays() {

    numSharedNodes_ = int_sqrt(numLocalElements_);
    procsPerNode_ = 4;

    sharedNodeIDs_ = new GlobalID[numSharedNodes_];
    sharedNodeIDsAllocated_ = true;
    sharedNodeProcs_ = new int*[numSharedNodes_];
    sharedNodeProcsAllocated_ = true;
    numSharedNodeProcs_ = new int[numSharedNodes_];
    for(int i=0; i<numSharedNodes_; i++){
        sharedNodeIDs_[i] = (GlobalID)0;
        sharedNodeProcs_[i] = new int[procsPerNode_];
        numSharedNodeProcs_[i] = 0;
        for(int j=0; j<procsPerNode_; j++){
            sharedNodeProcs_[i][j] = -1;
        }
    }
}

//==============================================================================
void PoissonData::deleteSharedNodeArrays() {

    if (sharedNodeIDsAllocated_) delete [] sharedNodeIDs_;
    sharedNodeIDsAllocated_ = false;

    if (sharedNodeProcsAllocated_) {
        for(int i=0; i<numSharedNodes_; i++) delete [] sharedNodeProcs_[i];
        delete [] sharedNodeProcs_;
        delete [] numSharedNodeProcs_;
    }
    sharedNodeProcsAllocated_ = false;
}

//==============================================================================
void PoissonData::getLeftSharedNodes(int& numShared) {
//
//This function decides whether any of the nodes along the left edge,
//including the top node but not the bottom node, are shared. It also
//decides which processors the nodes are shared with.
//

    if (numProcs_ == 1) {
        numShared = 0;
        return;
    }

    if (procX_ == 0) {
        //if this proc is on the left edge of the square...

        if (procY_ < maxProcY_) {
            //if this proc is not the top left proc...

            numShared = 1;

            int topLeftElemIndex = numLocalElements_ -
                               int_sqrt(numLocalElements_);

            int size = 0;
            GlobalID* nodes = elems_[topLeftElemIndex].getElemConnPtr(size);
            if (size==0) messageAbort("getLeftSharedNodes: can't get connPtr");

            sharedNodeIDs_[0] = nodes[3]; //elem's top left node is node 3
            numSharedNodeProcs_[0] = 2;
            sharedNodeProcs_[0][0] = localProc_;
            sharedNodeProcs_[0][1] = localProc_ + int_sqrt(numProcs_);

            return;
        }
        else {
            //else this proc is the top left proc...
            numShared = 0;
        }
    }
    else {
        //else this proc is not on the left edge of the square...

        numShared = numSharedNodes_;
        int lowerLeftElemIndex = 0;

        int sqrtElems = int_sqrt(numLocalElements_);

        int shOffset = 0;
        for(int i=0; i<sqrtElems; i++){
            //stride up the left edge of the local elements...
            int size=0;
            GlobalID* nodes = elems_[lowerLeftElemIndex+i*sqrtElems].
                                                 getElemConnPtr(size);
            if (size==0) messageAbort("getLeftSharedNodes: bad conn ptr.");

            //now put in the top left node
            sharedNodeIDs_[shOffset] = nodes[3];
            sharedNodeProcs_[shOffset][0] = localProc_-1;
            sharedNodeProcs_[shOffset][1] = localProc_;
            numSharedNodeProcs_[shOffset++] = 2;
        }

        if (procY_ < maxProcY_) {
            //if this proc isn't on the top edge, the upper left node (the
            //last one we put into the shared node list) is shared by 4 procs.
            shOffset--;
            numSharedNodeProcs_[shOffset] = 4;
            sharedNodeProcs_[shOffset][2] = localProc_ + int_sqrt(numProcs_);
            sharedNodeProcs_[shOffset][3] = sharedNodeProcs_[shOffset][2] - 1;
        }
    }
}

//==============================================================================
void PoissonData::getRightSharedNodes(int& numShared) {
//
//This function decides whether any of the nodes along the right edge,
//including the bottom node but not the top node, are shared. It also
//decides which processors the nodes are shared with.
//

    if (numProcs_ == 1) {
        numShared = 0;
        return;
    }

    if (procX_ == maxProcX_) {
        //if this proc is on the right edge of the square...

        if (procY_ > 0) {
            //if this proc is not the bottom right proc...

            numShared = 1;

            int lowerRightElemIndex = int_sqrt(numLocalElements_) - 1;

            int size = 0;
            GlobalID* nodes = elems_[lowerRightElemIndex].getElemConnPtr(size);
            if (size==0) messageAbort("getRightSharedNodes: can't get connPtr");

            sharedNodeIDs_[0] = nodes[1]; //elem's bottom right node is node 1
            numSharedNodeProcs_[0] = 2;
            sharedNodeProcs_[0][0] = localProc_;
            sharedNodeProcs_[0][1] = localProc_ - int_sqrt(numProcs_);

            return;
        }
        else {
            //else this proc is the bottom right proc...
            numShared = 0;
        }
    }
    else {
        //else this proc is not on the right edge of the square...

        numShared = numSharedNodes_;
        int upperRightElemIndex = numLocalElements_ - 1;

        int sqrtElems = int_sqrt(numLocalElements_);

        int shOffset = 0;
        for(int i=0; i<sqrtElems; i++){
            //stride down the right edge of the local elements...
            int size=0;
            GlobalID* nodes = elems_[upperRightElemIndex-i*sqrtElems].
                                                  getElemConnPtr(size);
            if (size==0) messageAbort("getRightSharedNodes: bad conn ptr.");

            //now put in the lower right node
            sharedNodeIDs_[shOffset] = nodes[1];
            sharedNodeProcs_[shOffset][0] = localProc_+1;
            sharedNodeProcs_[shOffset][1] = localProc_;
            numSharedNodeProcs_[shOffset++] = 2;
        }

        if (procY_ > 0) {
            //if this proc isn't on the bottom edge, the lower right node (the
            //last one we put into the shared node list) is shared by 4 procs.
            shOffset--;
            numSharedNodeProcs_[shOffset] = 4;
            sharedNodeProcs_[shOffset][2] = localProc_ - int_sqrt(numProcs_);
            sharedNodeProcs_[shOffset][3] = sharedNodeProcs_[shOffset][2] + 1;
        }
    }
}

//==============================================================================
void PoissonData::getTopSharedNodes(int& numShared) {
//
//This function decides whether any of the nodes along the top edge,
//including the right node but not the left node, are shared. It also
//decides which processors the nodes are shared with.
//

    if (numProcs_ == 1) {
        numShared = 0;
        return;
    }

    if (procY_ == maxProcY_) {
        //if this proc is on the top edge of the square...

        if (procX_ < maxProcX_) {
            //if this proc is not the top right proc...

            numShared = 1;

            int topRightElemIndex = numLocalElements_ - 1;

            int size = 0;
            GlobalID* nodes = elems_[topRightElemIndex].getElemConnPtr(size);
            if (size==0) messageAbort("getTopSharedNodes: can't get connPtr");

            sharedNodeIDs_[0] = nodes[2]; //elem's top right node is node 2
            numSharedNodeProcs_[0] = 2;
            sharedNodeProcs_[0][0] = localProc_;
            sharedNodeProcs_[0][1] = localProc_ + 1;

            return;
        }
        else {
            //else this proc is the top right proc...
            numShared = 0;
        }
    }
    else {
        //else this proc is not on the top edge of the square...

        numShared = numSharedNodes_;
        int topLeftElemIndex = numLocalElements_ - int_sqrt(numLocalElements_);

        int sqrtElems = int_sqrt(numLocalElements_);

        int shOffset = 0;
        for(int i=0; i<sqrtElems; i++){
            //stride across the top edge of the local elements...
            int size=0;
            GlobalID* nodes = elems_[topLeftElemIndex+i].getElemConnPtr(size);
            if (size==0) messageAbort("getTopSharedNodes: bad conn ptr.");

            //now put in the upper right node
            sharedNodeIDs_[shOffset] = nodes[2];
            sharedNodeProcs_[shOffset][0] = localProc_+int_sqrt(numProcs_);
            sharedNodeProcs_[shOffset][1] = localProc_;
            numSharedNodeProcs_[shOffset++] = 2;
        }
        if (procX_ < maxProcX_) {
            //if this proc isn't on the right edge, the top right node (the
            //last one we put into the shared node list) is shared by 4 procs.
            shOffset--;
            numSharedNodeProcs_[shOffset] = 4;
            sharedNodeProcs_[shOffset][2] = localProc_ + 1;
            sharedNodeProcs_[shOffset][3] = sharedNodeProcs_[shOffset][0] + 1;
        }
    }
}

//==============================================================================
void PoissonData::getBottomSharedNodes(int& numShared) {
//
//This function decides whether any of the nodes along the bottom edge,
//including the left node but not the right node, are shared. It also
//decides which processors the nodes are shared with.
//

    if (numProcs_ == 1) {
        numShared = 0;
        return;
    }

    if (procY_ == 0) {
        //if this proc is on the bottom edge of the square...

        if (procX_ > 0) {
            //if this proc is not the bottom left proc...

            numShared = 1;

            int lowerLeftElemIndex = 0;

            int size = 0;
            GlobalID* nodes = elems_[lowerLeftElemIndex].getElemConnPtr(size);
            if (size==0) messageAbort("getBottomSharedNodes: cant get connPtr");

            sharedNodeIDs_[0] = nodes[0]; //elem's bottom left node is node 0
            numSharedNodeProcs_[0] = 2;
            sharedNodeProcs_[0][0] = localProc_;
            sharedNodeProcs_[0][1] = localProc_ - 1;

            return;
        }
        else {
            //else this proc is the top right proc...
            numShared = 0;
        }
    }
    else {
        //else this proc is not on the bottom edge of the square...

        numShared = numSharedNodes_;
        int lowerRightElemIndex = int_sqrt(numLocalElements_) - 1;

        int sqrtElems = int_sqrt(numLocalElements_);

        int shOffset = 0;
        for(int i=0; i<sqrtElems; i++){
            //stride across the bottom edge of the local elements, from 
            //right to left...
            int size=0;
            GlobalID* nodes = elems_[lowerRightElemIndex-i].
                                              getElemConnPtr(size);
            if (size==0) messageAbort("getBottomSharedNodes: bad conn ptr.");

            //now put in the lower left node
            sharedNodeIDs_[shOffset] = nodes[0];
            sharedNodeProcs_[shOffset][0] = localProc_ - int_sqrt(numProcs_);
            sharedNodeProcs_[shOffset][1] = localProc_;
            numSharedNodeProcs_[shOffset++] = 2;
        }
        if (procX_ > 0) {
            //if this proc isn't on the left edge, the lower left node (the
            //last one we put into the shared node list) is shared by 4 procs.
            shOffset--;
            numSharedNodeProcs_[shOffset] = 4;
            sharedNodeProcs_[shOffset][2] = localProc_ - 1;
            sharedNodeProcs_[shOffset][3] = sharedNodeProcs_[shOffset][0] - 1;
        }
    }
}

//==============================================================================
void PoissonData::printSharedNodes(char* str, int numShared, GlobalID* nodeIDs,
                                   int** shareProcs, int* numShareProcs) {

    for(int i=0; i<numShared; i++) {
        cout << localProc_ << ", " << str << " node: " << (int) nodeIDs[i];
        cout << ", procs: ";
        for(int j=0; j<numShareProcs[i]; j++) {
            cout << shareProcs[i][j] << " ";
        }
        cout << endl;
    }
}

//==============================================================================
void PoissonData::calculateBCs() {
//
//This function figures out which nodes lie on the boundary. The ones that
//do are added to the BC set, along with appropriate alpha/beta/gamma values.
//
    for(int i=0; i<numLocalElements_; i++) {
        int size=0;
        GlobalID* nodeIDs = elems_[i].getElemConnPtr(size);
        if (size==0) messageAbort("calculateBCs: bad conn ptr.");

        double* xcoord = elems_[i].getNodalX(size);
        double* ycoord = elems_[i].getNodalY(size);

        //now loop over the nodes and see if any are on a boundary.
        for(int j=0; j<size; j++) {
            if ((xcoord[j] == 0.0) || (xcoord[j] == 1.0) ||
                (ycoord[j] == 0.0) || (ycoord[j] == 1.0)) {

                addBCNode(nodeIDs[j], xcoord[j], ycoord[j]);
            }
        }
    }
}

//==============================================================================
void PoissonData::addBCNode(GlobalID nodeID, double x, double y){

    if (appendBCNodeID(nodeID) == 1) {
        BCNodeIDsAllocated_ = true;

        appendBCRow(alpha_, 1.0);
        appendBCRow(beta_, 0.0);

        double gammaValue = pow(x, 2.0) + pow(y, 2.0);

        appendBCRow(gamma_, gammaValue);

        BCArraysAllocated_ = true;

        numBCNodes_++;
    }
}

//==============================================================================
int PoissonData::appendBCNodeID(GlobalID nodeID) {
//
//Returns 1 if nodeID is not already in BCNodeIDs_, 0 if it is.
//nodeID is only appended to BCNodeIDs_ if it isn't already in there.

    for(int j=0; j<numBCNodes_; j++)
        if (BCNodeIDs_[j] == nodeID) return(0);

    GlobalID* newNodeList = new GlobalID[numBCNodes_+1];
    for(int i=0; i<numBCNodes_; i++) {
        newNodeList[i] = BCNodeIDs_[i];
    }
    newNodeList[numBCNodes_] = nodeID;

    delete [] BCNodeIDs_;
    BCNodeIDs_ = newNodeList;

    return(1);
}

//==============================================================================
void PoissonData::appendBCRow(double**& valTable, double value) {

    double** newTable = new double*[numBCNodes_+1];

    for(int i=0; i<numBCNodes_; i++) {
        newTable[i] = new double[fieldSize_];

        for(int j=0; j<fieldSize_; j++) {
            newTable[i][j] = valTable[i][j];
        }

        delete [] valTable[i];
    }

    newTable[numBCNodes_] = new double[fieldSize_];
    for(int j=0; j<fieldSize_; j++) {
        newTable[numBCNodes_][j] = value;
    }

    delete [] valTable;
    valTable = newTable;
}

//==============================================================================
void PoissonData::printBCs() {
    for(int i=0; i<numBCNodes_; i++) {
        cout << localProc_ << ", BC node: " << (int)BCNodeIDs_[i]
             << ", gamma: " << gamma_[i][0] << endl;
    }
}

//==============================================================================
void PoissonData::deleteBCArrays() {

    if (BCNodeIDsAllocated_) {
        delete [] BCNodeIDs_;
        BCNodeIDs_ = NULL;
        BCNodeIDsAllocated_ = false;
    }

    if (BCArraysAllocated_) {
        for(int i=0; i<numBCNodes_; i++) {
            delete [] alpha_[i];
            delete [] beta_[i];
            delete [] gamma_[i];
        }
        delete [] alpha_;
        alpha_ = NULL;
        delete [] beta_;
        beta_ = NULL;
        delete [] gamma_;
        gamma_ = NULL;
    }
    BCArraysAllocated_ = false;
}

