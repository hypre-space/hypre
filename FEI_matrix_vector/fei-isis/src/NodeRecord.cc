#include <iostream.h>
#include <assert.h>
#include <stdio.h>

#include "other/basicTypes.h"
#include "src/BCRecord.h"
#include "src/SimpleList.h"

#include "src/NodeRecord.h"

//==========================================================================
NodeRecord::NodeRecord() {

    globalNodeID_ = -99;
    numNodalDOF_ = 0;
    numNodalFields_ = 0;
    localEqnID_ = 0;
    numProcs_ = 0;
    procList_ = NULL;
    numBlocks_ = 0;
    blockList_ = NULL;
    numFields_ = 0; 
    fieldIDList_ = NULL;
    fieldOffsetList_ = NULL;
    nodSoln_ = NULL;
    hasBCData_ = false;
    hasBCDone_ = false;

    BCTempList_ = NULL;
    numTempBCs_ = 0;

    BCRecordList_ = NULL;
    numBCRecords_ = 0;

    return;
}


//==========================================================================
NodeRecord::~NodeRecord() {

    if (procList_ != NULL)
        delete [] procList_;

    if (blockList_ != NULL)
        delete [] blockList_;

    if (fieldIDList_ != NULL)
        delete [] fieldIDList_;

    if (fieldOffsetList_ != NULL)
        delete [] fieldOffsetList_;

    if (nodSoln_ != NULL)
        delete [] nodSoln_;

    if (numBCRecords_ > 0)
        delete [] BCRecordList_;

    if (numTempBCs_ > 0)
        delete [] BCTempList_;

    return;
}

//==========================================================================
int *NodeRecord::pointerToProcList(int& nProcs) {
    nProcs = numProcs_;
    return(procList_);
}

//==========================================================================
void NodeRecord::allocateProcList(int nProcs) {

    if (nProcs==0) return;

    int i;
    numProcs_ = nProcs;
    procList_ = new int[numProcs_];

    for (i = 0; i < numProcs_; i++)
        procList_[i] = 0;

    return;
}

//==========================================================================
int *NodeRecord::pointerToBlockList(int& nBlocks) {
    nBlocks = numBlocks_;
    return(blockList_);
}

//==========================================================================
void NodeRecord::allocateBlockList(int nBlocks) {

    if (nBlocks==0) return;

    int i;
    numBlocks_ = nBlocks;
    blockList_ = new int[numBlocks_];
    for (i = 0; i < numBlocks_; i++)
        blockList_[i] = 0;
    return;
}

//==========================================================================
int *NodeRecord::pointerToFieldIDList(int& nFields) {
    nFields = numFields_;
    return(fieldIDList_);
}

//==========================================================================
int *NodeRecord::pointerToFieldOffsetList(int& nFields) {
    nFields = numFields_;
    return(fieldOffsetList_);
}

//==========================================================================
void NodeRecord::allocateFieldLists(int nFields) {

    if (nFields==0) return;

    int i;
    numFields_ = nFields;
    fieldIDList_ = new int[numFields_];
    fieldOffsetList_ = new int[numFields_];
    for (i = 0; i < numFields_; i++) {
        fieldIDList_[i] = 0;
        fieldOffsetList_[i] = 0;
    }

    return;
}
 
//==========================================================================
double *NodeRecord::pointerToSoln(int& numDOF) {
    numDOF = numNodalDOF_;
    return(nodSoln_);
}

//==========================================================================
void NodeRecord::allocateSolnList() {
    int i;

    if (nodSoln_ != NULL) delete [] nodSoln_;

    nodSoln_ = new double[numNodalDOF_];
    for (i = 0; i < numNodalDOF_; i++)
        nodSoln_[i] = 0.0;
    return;
}

//==========================================================================
void NodeRecord::dumpToScreen() {
    cout << "  globalNodeID_ = " << (int)globalNodeID_ << endl;
    cout << "  localEqnID_ = " << localEqnID_ << endl;
    cout << "  numBlocks_ = " << numBlocks_ << endl;
    cout << "  numProcs_ = " << numProcs_ << endl;
    cout << "  numTempBCs_ = " << numTempBCs_<< endl;
    cout << "  hasBCData = " << hasBCData_ << endl;
    cout << "  hasBCDone = " << hasBCDone_ << endl;
    cout << "  numNodalDOF_ = " << numNodalDOF_ << endl << endl;
    if (nodSoln_ != NULL) {
        cout << "  the " << numNodalDOF_ << " nodal soln parameters: ";
        for (int i = 0; i < numNodalDOF_; ++i) {
            cout << "   " << nodSoln_[i];
        }
    }
    cout << endl << endl;
    return;
}

//==========================================================================
BCRecord *NodeRecord::pointerToBCRecords(int& lenBCList) {
    lenBCList = numBCRecords_;
    return(BCRecordList_);
}

//==========================================================================
void NodeRecord::allocateBCRecords(int numBCRecords) {

    if (numBCRecords==0) return;

    numBCRecords_ = numBCRecords;
    BCRecordList_ = new BCRecord [numBCRecords];
    
    return;
}

//==========================================================================
void NodeRecord::removeBCData(){
    if (numBCRecords_>0){
        delete [] BCRecordList_;
        BCRecordList_ = NULL;
        numBCRecords_ = 0;

        clearBCDoneFlag();
        clearBCDataFlag();
    }
}

//==========================================================================
void NodeRecord::addBCRecord(int fieldID,
                             int fieldSize, 
                             const double *alpha, 
                             const double *beta, 
                             const double *gamma) {

    BCRecord *BCRecordPtr;
    double *myBCParams;
    int i, test;

    BCRecordPtr = new BCRecord;
    BCRecordPtr->setFieldID(fieldID);
    BCRecordPtr->setFieldSize(fieldSize);
    
    BCRecordPtr->allocateAlpha();
    myBCParams = BCRecordPtr->pointerToAlpha(test);
    assert (test == fieldSize);   // paranoia - remove when testing done
    for (i = 0; i < fieldSize; i++) {
        myBCParams[i] = alpha[i];
    }

    BCRecordPtr->allocateBeta();
    myBCParams = BCRecordPtr->pointerToBeta(test);
    assert (test == fieldSize);   // paranoia - remove when testing done
    for (i = 0; i < fieldSize; i++) {
        myBCParams[i] = beta[i];
    }

    BCRecordPtr->allocateGamma();
    myBCParams = BCRecordPtr->pointerToGamma(test);
    assert (test == fieldSize);   // paranoia - remove when testing done
    for (i = 0; i < fieldSize; i++) {
        myBCParams[i] = gamma[i];
    }

    //allocate a longer list of BCRecord pointers.
    BCRecord** newList = new BCRecord*[numTempBCs_+1];

    //now copy the existing pointers into the new list.
    for(i=0; i<numTempBCs_; i++){
        newList[i] = BCTempList_[i];
    }

    //now add the new one to the end.
    newList[numTempBCs_] = BCRecordPtr;

    //now delete the old list and reset the pointer to point to
    //the new one.
    if (numTempBCs_>0) delete [] BCTempList_;
    BCTempList_ = newList;

    //and finally, increment the list length.
    numTempBCs_++;

    return;
}

//==========================================================================
BCRecord** NodeRecord::pointerToBCTempList(int& numTempBCs){
    numTempBCs = numTempBCs_;
    return(BCTempList_);
}

//==========================================================================
void NodeRecord::destroyTempBCList() {
//
//  since this list is merely a temporary storage for collecting BC data,
//  we can delete it before the NodeRecord destructor gets called, and
//  we SHOULD do so in endLoadNodeSets()
//

    for(int i=0; i<numTempBCs_; i++){
        delete BCTempList_[i];
    }

    delete [] BCTempList_;
    numTempBCs_ = 0;

    return;
}

//==========================================================================
int NodeRecord::getFieldOffset(int myFieldID) {

    for (int i = 0; i < numFields_; i++) {
        if (myFieldID == fieldIDList_[i]) {
            return(fieldOffsetList_[i]);
        }
    }
    return(-1);
}

