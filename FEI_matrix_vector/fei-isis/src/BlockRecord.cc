#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "other/basicTypes.h"
#include "src/BlockRecord.h"

//------------------------------------------------------------------------------
//
//  BlockRecord is a container class for aggregating element block information
//  Normally, there will be an array of these containers, one for each
//  element block found in the FE analysis.
//
//  There is a fundamental assumption embedded in some of the methods that are
//  associated with this class, namely that the number of blocks found on a
//  given processor is small enough so that some brute-force logic (e.g.,
//  doing a linear search on a list of blocks) will not constitute a serious
//  computational burden.  If this assumption is not warranted, some of the
//  methods defined here may need some reorganization for performance gains.
//
//  kdm Sept 14, 1998
//
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//  blockRecord constructor/destructor
//------------------------------------------------------------------------------

BlockRecord::BlockRecord() {

    blockID_ = 0; 
    numNodesPerElement_ = 0; 
    interleaveStrategy_ = 0; 
    numElemDOF_ = 0; 
    numElemSets_ = 0; 
    numElemTotal_ = 0; 

    numElemFields_ = NULL;
    elemFieldIDs_ = NULL;
    elemIDs_ = NULL;
    elemConn_ = NULL;
    elemSoln_ = NULL;
    localEqnElemSoln_ = NULL;
    numNodalDOF_ = NULL;

    numEqnsPerElement_ = 0;
    numInitElemSets_ = 0;
    numLoadElemSets_ = 0;
    numInitElemTotal_ = 0;
    numLoadElemTotal_ = 0;
    numActiveNodes_ = 0; 
    numActiveEqns_ = 0;
    nextElemIndex_ = 0;
    
    return;
}


BlockRecord::~BlockRecord() {

    int i;

    if (elemFieldIDs_ != NULL) {
        for (i = 0; i < numNodesPerElement_; i++){
            delete [] elemFieldIDs_[i];
        }
        delete [] elemFieldIDs_;
    }
    
    if (numElemFields_ != NULL) {
        delete [] numElemFields_;
    }
    
    if (numNodalDOF_ != NULL) {
        delete [] numNodalDOF_;
    }

    if (localEqnElemSoln_ != NULL) {
        delete [] localEqnElemSoln_;
    }
    
    if (elemConn_ != NULL) {
        for (i = 0; i < numElemTotal_; i++){
            delete [] elemConn_[i];
        }
        delete [] elemConn_;
    }

    if (elemSoln_ != NULL) {
        for (i = 0; i < numElemTotal_; i++){
            delete [] elemSoln_[i];
        }
        delete [] elemSoln_;
    }
    
    if (elemIDs_ != NULL) {
        delete [] elemIDs_;
    }
}


//------------------------------------------------------------------------------
//  member functions to set/get block scalar internals
//------------------------------------------------------------------------------

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the block identifier
//
GlobalID BlockRecord::getBlockID() {
	return (blockID_);
}
void BlockRecord::setBlockID(GlobalID blockID) {
    blockID_ = blockID;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the number of nodes per generic element in this block
//
int BlockRecord::getNumNodesPerElement() {
	return (numNodesPerElement_);
}
void BlockRecord::setNumNodesPerElement(int numNodesPerElement) {
    numNodesPerElement_ = numNodesPerElement;
    return;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the element matrix row-interleave strategy (node-centric
//  or field-centric)
//
int BlockRecord::getInterleaveStrategy() {
	return (interleaveStrategy_);
}
void BlockRecord::setInterleaveStrategy(int interleaveStrategy) {
    interleaveStrategy_ = interleaveStrategy;
    return;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the number of elemental (non-nodal) solution parameters
//
int BlockRecord::getNumElemDOF() {
    return (numElemDOF_);
}
void BlockRecord::setNumElemDOF(int numElemDOF) {
    numElemDOF_ = numElemDOF;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the current number of element sets in this block
//  (may vary between init and load steps)
//
int BlockRecord::getNumElemSets() {
    return (numElemSets_);
}
void BlockRecord::setNumElemSets(int numElemSets) {
    numElemSets_ = numElemSets;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the total number of elements in this block
//  (should NOT vary between init and load steps)
//
int BlockRecord::getNumElemTotal() {
    return (numElemTotal_);
}
void BlockRecord::setNumElemTotal(int numElemTotal) {
    numElemTotal_ = numElemTotal;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the total number of eqns per element, including elemDOF
//  (this gives the size of the element arrays)
//
int BlockRecord::getNumEqnsPerElement() {
    return (numEqnsPerElement_);
}
void BlockRecord::setNumEqnsPerElement(int numEqnsPerElement) {
    numEqnsPerElement_ = numEqnsPerElement;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the number of element sets for the initialization step
//
int BlockRecord::getInitNumElemSets() {
    return (numInitElemSets_);
}
void BlockRecord::setInitNumElemSets(int numInitElemSets) {
    numInitElemSets_ = numInitElemSets;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the number of element sets for the load step
//
int BlockRecord::getLoadNumElemSets() {
    return (numLoadElemSets_);
}
void BlockRecord::setLoadNumElemSets(int numLoadElemSets) {
    numLoadElemSets_ = numLoadElemSets;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the total number of elements for the initialization step
//  (currently, this should be the same as for the load step)
//
int BlockRecord::getInitNumElemTotal() {
    return (numInitElemTotal_);
}
void BlockRecord::setInitNumElemTotal(int numInitElemTotal) {
    numInitElemTotal_ = numInitElemTotal;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set the total number of elements for the load step
//  (currently, this should be the same as for the initialization step)
//
int BlockRecord::getLoadNumElemTotal() {
    return (numLoadElemTotal_);
}
void BlockRecord::setLoadNumElemTotal(int numLoadElemTotal) {
    numLoadElemTotal_ = numLoadElemTotal;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return (or increment) the number of active nodes for this block
//  (here, the "set" is replaced by an "increment", because we must
//  determine this important size parameter by iterating over all
//  the elements in the block, incrementing the active node list
//  whenever we find a new node
//
int BlockRecord::getNumActiveNodes() {
    return numActiveNodes_;
}
void BlockRecord::incrementNumActiveNodes(int numNewNodes) {
    numActiveNodes_ += numNewNodes;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return (or increment) the number of active eqns for this block
//  (see note for get/increment of active nodes)
//
int BlockRecord::getNumActiveEqns() {
    return numActiveEqns_;
}
void BlockRecord::incrementNumActiveEqns(int numNewEqns) {
    numActiveEqns_ += numNewEqns;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  get/set internal counter for finding the next element ID
//  (used to avoid searches when handling the same ordering of
//  elements being passed between the init and load step)
//
int BlockRecord::getNextElemIndex() {
    return nextElemIndex_;
}
void BlockRecord::setNextElemIndex(int nextIndex) {
    nextElemIndex_ = nextIndex;
}


//------------------------------------------------------------------------------
//  member functions to set/get block list and table internals
//------------------------------------------------------------------------------

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return/allocate list of the number of fields defined at each
//  node in the generic element that represents this block.  
//
//  For a single-physics problem, this would be all ones...
//
int *BlockRecord::pointerToNumElemFields(int& lenNumElemFields) {
    lenNumElemFields = numNodesPerElement_;
    return(numElemFields_);
}
void BlockRecord::allocateNumElemFields(int lenNumElemFields) {

    int i;
    assert (lenNumElemFields == numNodesPerElement_);
    numElemFields_ = new int [numNodesPerElement_];
    for (i = 0; i < numNodesPerElement_; i++) {
        numElemFields_[i] = 0;
    }
    return;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return/allocate a table of the element field identifiers for each
//  generic element.  Here, each row corresponds the various nodes in the
//  element, and the number of columns in each row is given by the number
//  of solution fields defined at that node
//
//  for a single-physics problem, this table has only one column, and that
//  column is populated by the single field ID for the single soln field
//
int **BlockRecord::pointerToElemFieldIDs(int& numIDRows, int* &numIDCols) {

    numIDRows = numNodesPerElement_;
    numIDCols = numElemFields_;

    return(elemFieldIDs_);
} 
void BlockRecord::allocateElemFieldIDs(int numIDRows, int *numIDCols) {

    int i, j;
    assert (numIDRows == numNodesPerElement_);
    elemFieldIDs_ = new int* [numNodesPerElement_];

    for (i = 0; i < numNodesPerElement_; i++) {
        assert (numIDCols[i] == numElemFields_[i]);
        elemFieldIDs_[i] = new int[numElemFields_[i]];
        for (j = 0; j < numElemFields_[i]; ++j) {
            elemFieldIDs_[i][j] = 0;
        }
    }
    
    return;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return/allocate the list of element IDs for each element in this
//  block.  These elements are supposed to be passed in via the same
//  order in both the init and load steps
//
GlobalID *BlockRecord::pointerToElemIDs(int& lenElemIDs) {
    lenElemIDs = numElemTotal_;
    return(elemIDs_);
}
void BlockRecord::allocateElemIDs(int numElems) {

    int i;
    assert (numElems == numElemTotal_);
    elemIDs_ = new GlobalID[numElemTotal_];
    for (i = 0; i < numElemTotal_; i++) {
        elemIDs_[i] = 0;
    }
    return;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return/allocate the table of element connectivities for each element
//  in the block.  The rows represent the list of associated nodes for a
//  given element in the block
//
GlobalID **BlockRecord::pointerToElemConn(int& numElems, 
                                          int& numNodesPerElement) {

    numElems = numElemTotal_;
    numNodesPerElement = numNodesPerElement_;

    return(elemConn_);
}
void BlockRecord::allocateElemConn(int numElems, int numNodesPerElement) {

    int i, j;
    assert (numElems == numElemTotal_);
    assert (numNodesPerElement == numNodesPerElement_);

    elemConn_ = new GlobalID* [numElemTotal_];
    for(i = 0; i < numElemTotal_; i++) {
        elemConn_[i] = new GlobalID[numNodesPerElement_];
        for (j = 0; j < numNodesPerElement_; ++j) {
            elemConn_[i][j] = 0;
        }
    }

    return;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return/allocate a list of solution cardinalities for elements in 
//  this block.  This list represents only the solution fields used for
//  this block, so for a multiphysics problem, different blocks may have
//  different views of the solution cardinality at a node lying on the
//  boundary between those blocks
//
int *BlockRecord::pointerToNumNodalDOF(int& lenNumNodalDOF) {
    lenNumNodalDOF = numNodesPerElement_;
    return(numNodalDOF_);
}
void BlockRecord::allocateNumNodalDOF(int lenNumNodalDOF) {

    int i;
    assert (lenNumNodalDOF == numNodesPerElement_);
    numNodalDOF_ = new int[numNodesPerElement_];
    for (i = 0; i < numNodesPerElement_; i++) {
        numNodalDOF_[i] = 0;
    }
    return;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return/allocate a list of solution cardinalities for elements in 
//  this block.  This list represents only the solution fields used for
//  this block, so for a multiphysics problem, different blocks may have
//  different views of the solution cardinality at a node lying on the
//  boundary between those blocks
//
int *BlockRecord::pointerToLocalEqnElemDOF(int& numElems) {
    numElems = numElemTotal_;
    return(localEqnElemSoln_);
}
void BlockRecord::allocateLocalEqnElemDOF(int numElems) {

    int i;
    assert (numElems == numElemTotal_);
    localEqnElemSoln_ = new int[numElemTotal_];
    for (i = 0; i < numElemTotal_; i++) {
        localEqnElemSoln_[i] = 0;
    }
    return;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  return/allocate the table of element solution parameters, where each
//  row represents the list of element DOF corresponding to a given element
//
//  since these parameters are by definition local to a processor (as elements
//  aren't shared among processors), it may be advantageous to store these
//  element solution parameters here ONLY if there are few element DOF.  If
//  there are lot of elemental DOF for a given block, it may be a better idea
//  simply to look up these parameters in the local solution vector on the
//  fly instead of putting them in a (potentially large) table.
//
double **BlockRecord::pointerToElemSoln(int& numElems, int& numElemDOF) {

    numElems = numElemTotal_;
    numElemDOF = numElemDOF_;

    return(elemSoln_);
}
void BlockRecord::allocateElemSoln(int numElems, int numElemDOF) {

    int i, j;
    assert (numElems == numElemTotal_);
    assert (numElemDOF == numElemDOF_);

    elemSoln_ = new double* [numElemTotal_];
    for(i = 0; i < numElemTotal_; i++) {
        elemSoln_[i] = new double[numElemDOF_];
        for (j = 0; j < numElemDOF_; ++j) {
            elemSoln_[i][j] = 0.0;
        }
    }

    return;
}


//------------------------------------------------------------------------------
//  find a given elemID in this block's list of element identifiers
//
//  (normally used during the load step to find stuff passed during
//   the initialization step, for example)
//
int BlockRecord::findElemIndex(GlobalID findThisElemID) {

    if (elemIDs_ != NULL) {  // paranoia never hurts...

//  check to see if cached index is correct, which it is supposed to
//  be, since the elements are normally passed into the FEI 
//  implementation using the same ordering for both init and load...

        if (elemIDs_[nextElemIndex_] == findThisElemID) {
            return(nextElemIndex_);
        }

//  if we hit here, then something is amiss, so let's do a painful
//  linear search for now - if this gets hit a lot, then we probably
//  ought to add a key to the list of elemIDs so we can keep them
//  implicitly sorted to do some better "find" scheme...

        for (int i = 0; i < numElemTotal_; i++) {
            if (elemIDs_[i] == findThisElemID) {
                return(i);
            }
        }
    }
    return(-1);  // if all else fails...
    
}

//------------------------------------------------------------------------------
//  utility member functions 
//------------------------------------------------------------------------------

void BlockRecord::dumpToScreen() {

    int i, j;

    cout << " blockID_            = " << blockID_ << endl;
    cout << " numNodesPerElement_ = " << numNodesPerElement_ << endl;
    cout << " interleaveStrategy_ = " << interleaveStrategy_ << endl;
    cout << " numElemDOF_         = " << numElemDOF_ << endl;
    cout << " numElemSets_        = " << numElemSets_ << endl;
    cout << " numElemTotal_       = " << numElemTotal_ << endl;
    cout << endl;
    
    cout << " numEqnsPerElement_  = " << numEqnsPerElement_ << endl;
    cout << " numInitElemSets_    = " << numInitElemSets_ << endl;
    cout << " numLoadElemSets_    = " << numLoadElemSets_ << endl;
    cout << " numInitElemTotal_   = " << numInitElemTotal_ << endl;
    cout << " numLoadElemTotal_   = " << numLoadElemTotal_ << endl;
    cout << " numActiveNodes_     = " << numActiveNodes_ << endl;
    cout << " numActiveEqns_      = " << numActiveEqns_ << endl;
    cout << endl;
    
    cout << " the numNodalDOF_ list of length : " << numNodesPerElement_;
    cout << endl;
    if (numNodalDOF_ != NULL) {
        for (i = 0; i < numNodesPerElement_; ++i) {
            cout << "   " << numNodalDOF_[i];
        }
        cout << endl << endl;
    }
    
    cout << " the numElemFields list of length : " << numNodesPerElement_;
    cout << endl;
    if (numElemFields_ != NULL) {
        for (i = 0; i < numNodesPerElement_; ++i) {
            cout << "   " << numElemFields_[i];
        }
        cout << endl << endl;
    }
    
    if ((numElemFields_ != NULL) && (elemFieldIDs_ != NULL)) {
        cout << " the " << numNodesPerElement_ << " fieldID lists: ";
        cout << endl;
        for (i = 0; i < numNodesPerElement_; ++i) {
            for (j = 0; j < numElemFields_[i]; ++j) {
    	        cout << "   " << elemFieldIDs_[i][j];
    	    }
            cout << endl;
        }
        cout << endl;
    }
    
    if ((elemConn_ != NULL) && (elemIDs_ != NULL)) {
        cout << " the " << numElemTotal_ << " element IDs and their "
             << numNodesPerElement_ << " associated nodes ";
        cout << endl;
        for (i = 0; i < numElemTotal_; ++i) {
            cout << "  i = " << i << "  : " << (int) elemIDs_[i] << "  ";
            for (j = 0; j < numNodesPerElement_; ++j) {
    	        cout << "   " << (int) elemConn_[i][j];
    	    }
            cout << endl;
        }
        cout << endl;
    }
    
    return;
}
