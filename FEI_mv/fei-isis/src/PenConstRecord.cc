#include <iostream.h>
#include <stdio.h>
#include <assert.h>

#include "other/basicTypes.h"
#include "src/PenConstRecord.h"

//======================================================================
PenConstRecord::PenConstRecord() {

    CRNodeTable_ = NULL;
    CRNodeWeights_ = NULL;
    allocatedCRNodeWeights_ = false;
    numColsNodeWeights_ = NULL;
    CRIsLocalTable_ = NULL;
    allocatedCRIsLocalTable_ = false;
    CRConstValue_ = NULL;
    CRFieldList_ = NULL;

    lenCRNodeList_ = 0;
    numRowsNodeTable_ = 0;
    numColsNodeTable_ = 0; 
    numRowsNodeWeights_ = 0; 
    lenCRConstValue_ = 0;
    lenCRFieldList_ = 0;
    
    return;
}


//======================================================================
PenConstRecord::~PenConstRecord() {
    
    int i;
    
    if (CRNodeTable_ != NULL) {
        for (i = 0; i < numRowsNodeTable_; i++) {
            delete [] CRNodeTable_[i];
        }
        delete [] CRNodeTable_;
    }

    if (CRIsLocalTable_ != NULL) {
        for (i = 0; i < numRowsNodeTable_; i++) {
            delete [] CRIsLocalTable_[i];
        }
        delete [] CRIsLocalTable_;
    }

    if (CRNodeWeights_ != NULL) {
        for (i = 0; i < numRowsNodeWeights_; i++) {
            delete [] CRNodeWeights_[i];
        }
        delete [] CRNodeWeights_;
    }

    if (numColsNodeWeights_ != NULL) {
        delete [] numColsNodeWeights_;
    }

    if (numColsNodeWeights_ != NULL) {
        delete [] numColsNodeWeights_;
    }
    
    if (CRFieldList_ != NULL) {
        delete [] CRFieldList_;
    }

    return;
}


//======================================================================
GlobalID **PenConstRecord::pointerToCRNodeTable(int& numRows, 
                                                int& numCols) {

    numRows = numRowsNodeTable_;
    numCols = numColsNodeTable_;
    return(CRNodeTable_);
} 


//======================================================================
void PenConstRecord::allocateCRNodeTable(int numRows, 
                                         int numCols) {
    int i;

    numRowsNodeTable_ = numRows;
    numColsNodeTable_ = numCols;
    CRNodeTable_ = new GlobalID* [numRows];

    for (i = 0; i < numRows; i++)
        CRNodeTable_[i] = new GlobalID[numCols];

    return;
}


//======================================================================
void PenConstRecord::allocateCRIsLocalTable(int numRows, 
                                            int numCols) {
    int i;

//  this table ought to be a shadow of the actual node table, hence we
//  can just check to make sure we're getting the right number of args
//  passed here instead of setting our internal params to the passed args

    assert (numRowsNodeTable_ == numRows);
    assert (numColsNodeTable_ == numCols);
 
    CRIsLocalTable_ = new bool* [numRows];

    for (i = 0; i < numRows; i++) {
        CRIsLocalTable_[i] = new bool[numCols];
        for(int j=0; j<numCols; j++) CRIsLocalTable_[i][j] = true;
    }

    allocatedCRIsLocalTable_ = true;

    return;
}


//======================================================================
void PenConstRecord::remoteNode(GlobalID nodeID) {

    int i, j;

    for (i = 0; i < numRowsNodeTable_; i++) {
        for (j = 0; j < numColsNodeTable_; j++) {
            if (CRNodeTable_[i][j] == nodeID)
                CRIsLocalTable_[i][j] = false;
        }
    }

    return;
} 


//======================================================================
bool **PenConstRecord::pointerToCRIsLocalTable(int& numRows, 
                                               int& numCols) {

    numRows = numRowsNodeTable_;
    numCols = numColsNodeTable_;
    return(CRIsLocalTable_);
} 


//======================================================================
double **PenConstRecord::pointerToCRNodeWeights(int& numRows, 
                                                int* &numCols) {

    numRows = numRowsNodeWeights_;
    numCols = numColsNodeWeights_;

    return(CRNodeWeights_);
} 


//======================================================================
void PenConstRecord::allocateCRNodeWeights(int numRows, 
                                           int *numCols) {

    int i;
    
    numRowsNodeWeights_ = numRows;
    for (i = 0; i < numRowsNodeWeights_; i++) {
        if (allocatedCRNodeWeights_) delete [] CRNodeWeights_[i];
    }
    delete [] CRNodeWeights_;

    if (numColsNodeWeights_ != NULL) delete [] numColsNodeWeights_;
 
    numColsNodeWeights_ = new int[numRows];

    for (i = 0; i < numRows; i++) {
        numColsNodeWeights_[i] = numCols[i];
    }

    CRNodeWeights_ = new double* [numRows];

    for (i = 0; i < numRows; i++)
        CRNodeWeights_[i] = new double[numCols[i]];

    allocatedCRNodeWeights_ = true;

    return;
}
 

//======================================================================
double *PenConstRecord::pointerToCRConstValues(int& length) {

    length = lenCRConstValue_;
    return(CRConstValue_);
}


//======================================================================
void PenConstRecord::allocateCRConstValues(int length) {

    int i;

    lenCRConstValue_ = length;
    CRConstValue_ = new double[lenCRConstValue_];
    for (i = 0; i < lenCRConstValue_; i++)
        CRConstValue_[i] = 0.0;
    return;
}
 
 
//======================================================================
int *PenConstRecord::pointerToCRFieldList(int& length) {

    length = lenCRFieldList_;
    return(CRFieldList_);
}


//======================================================================
void PenConstRecord::allocateCRFieldList(int length) {

    int i;
    lenCRFieldList_ = length;

    if (CRFieldList_ != NULL) delete [] CRFieldList_;

    CRFieldList_ = new int[lenCRFieldList_];
    for (i = 0; i < lenCRFieldList_; i++)
        CRFieldList_[i] = 0;
    return;
}
 
 


//======================================================================
void PenConstRecord::dumpToScreen() {
    cout << " CRPenID_           = " << CRPenID_ << "\n";
    cout << " numPenCRs_         = " << numPenCRs_ << "\n";
    cout << " lenCRNodeList_      = " << lenCRNodeList_ << "\n";
    cout << " numRowsNodeTable_   = " << numRowsNodeTable_ << "\n";
    cout << " numColsNodeTable_   = " << numColsNodeTable_ << "\n";
    cout << " numRowsNodeWeights_ = " << numRowsNodeWeights_ << "\n";
    cout << "\n\n";
    return;
}
