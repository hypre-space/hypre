#include <assert.h>
#include <stdlib.h>

#include "other/basicTypes.h"
#include "src/MultConstRecord.h"

//======================================================================
MultConstRecord::MultConstRecord() {

    CRMultID_ = 0; 
    lenCRNodeList_ = 0;
    numMultCRs_ = 0;
    localEqnID_ = 0;  

    CRNodeTable_ = NULL;
    CRIsLocalTable_ = NULL; 
    allocatedCRIsLocalTable_ = false;
    numRowsNodeTable_ = 0;
    numColsNodeTable_ = 0; 

    allocatedCRNodeWeights_ = false;
    CRNodeWeights_ = NULL;
    numRowsNodeWeights_ = 0; 
    numColsNodeWeights_ = NULL;

    CRConstValue_ = NULL;
    lenCRConstValue_ = 0;

    CRFieldList_ = NULL;
    lenCRFieldList_ = 0;

    MultValues_ = NULL;
    lenMultValues_ = 0;
}

//======================================================================
MultConstRecord::~MultConstRecord() {

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
        
    if (CRConstValue_ != NULL) {
        delete [] CRConstValue_;
    }

    if (CRFieldList_ != NULL) {
        delete [] CRFieldList_;
    }

    if (MultValues_ != NULL) {
        delete [] MultValues_;
    }
    
    return;
}


//======================================================================
GlobalID **MultConstRecord::pointerToCRNodeTable(int& numRows, 
                                                 int& numCols) {

    numRows = numRowsNodeTable_;
    numCols = numColsNodeTable_;
    return(CRNodeTable_);
} 


//======================================================================
void MultConstRecord::allocateCRNodeTable(int numRows, 
                                          int numCols) {
    int i;
    numRowsNodeTable_ = numRows;
    numColsNodeTable_ = numCols;
    CRNodeTable_ = new GlobalID* [numRows];

    for (i = 0; i < numRowsNodeTable_; i++)
        CRNodeTable_[i] = new GlobalID[numCols];

    return;
}


//======================================================================
bool **MultConstRecord::pointerToCRIsLocalTable(int& numRows, 
                                                int& numCols) {

    numRows = numRowsNodeTable_;
    numCols = numColsNodeTable_;
    return(CRIsLocalTable_);
} 


//======================================================================
void MultConstRecord::allocateCRIsLocalTable(int numRows, 
                                             int numCols) {
    int i;
    
//  this table ought to be a shadow of the actual node table, hence we
//  can just check to make sure we're getting the right number of args
//  passed here instead of setting our internal params to the passed args

    assert (numRowsNodeTable_ == numRows);
    assert (numColsNodeTable_ == numCols);
    
    CRIsLocalTable_ = new bool* [numRows];

    for (i = 0; i < numRowsNodeTable_; i++){
        CRIsLocalTable_[i] = new bool [numCols];
        for(int j=0; j<numCols; j++) CRIsLocalTable_[i][j] = true;
    }

    allocatedCRIsLocalTable_ = true;

    return;
}


//======================================================================
double **MultConstRecord::pointerToCRNodeWeights(int& numRows, 
                                                 int* &numCols) {

    numRows = numRowsNodeWeights_;
    numCols = numColsNodeWeights_;

    return(CRNodeWeights_);
} 


//======================================================================
void MultConstRecord::allocateCRNodeWeights(int numRows, 
                                            int *numCols) {

    int i;
    numRowsNodeWeights_ = numRows;

    for(i=0; i<numRowsNodeWeights_; i++){
        if (allocatedCRNodeWeights_) delete [] CRNodeWeights_[i];
    }
    delete [] CRNodeWeights_;

    if (numColsNodeWeights_ != NULL) delete [] numColsNodeWeights_;

    numColsNodeWeights_ = new int[numRows];

    for (i = 0; i < numRowsNodeWeights_; i++){
        numColsNodeWeights_[i] = numCols[i];
    }

    CRNodeWeights_ = new double* [numRows];

    for (i = 0; i < numRowsNodeWeights_; i++)
        CRNodeWeights_[i] = new double[numCols[i]];

    allocatedCRNodeWeights_ = true;

    return;
}

 
//======================================================================
double *MultConstRecord::pointerToCRConstValues(int& length) {

    length = lenCRConstValue_;
    return(CRConstValue_);
}


//======================================================================
void MultConstRecord::allocateCRConstValues(int length) {

    int i;
    lenCRConstValue_ = length;

    if (CRConstValue_ != NULL) delete [] CRConstValue_;

    CRConstValue_ = new double[lenCRConstValue_];
    for (i = 0; i < lenCRConstValue_; i++)
        CRConstValue_[i] = 0.0;
    return;
}

 
//======================================================================
int *MultConstRecord::pointerToCRFieldList(int& length) {

    length = lenCRFieldList_;
    return(CRFieldList_);
}


//======================================================================
void MultConstRecord::allocateCRFieldList(int length) {

    int i;
    lenCRFieldList_ = length;

    if (CRFieldList_ != NULL) delete [] CRFieldList_;

    CRFieldList_ = new int[lenCRFieldList_];
    for (i = 0; i < lenCRFieldList_; i++)
        CRFieldList_[i] = 0;
    return;
}

 
//======================================================================
double *MultConstRecord::pointerToMultipliers(int& multLength) {

    multLength = lenMultValues_;
    return(MultValues_);
}


//======================================================================
void MultConstRecord::allocateMultipliers(int multLength) {

    int i;
    lenMultValues_ = multLength;

    if (MultValues_ != NULL) delete [] MultValues_;

    MultValues_ = new double[lenMultValues_];
    for (i = 0; i < lenMultValues_; i++)
        MultValues_[i] = 0.0;
    return;
}


#include <iostream.h>
//======================================================================
void MultConstRecord::dumpToScreen() {
    cout << " CRMultID_           = " << CRMultID_ << "\n";
    cout << " numMultCRs_         = " << numMultCRs_ << "\n";
    cout << " localEqnID_         = " << localEqnID_ << "\n";
    cout << " lenCRNodeList_      = " << lenCRNodeList_ << "\n";
    cout << " lenCRFieldList_     = " << lenCRFieldList_ << "\n";
    cout << " numRowsNodeTable_   = " << numRowsNodeTable_ << "\n";
    cout << " numColsNodeTable_   = " << numColsNodeTable_ << "\n";
    cout << " numRowsNodeWeights_ = " << numRowsNodeWeights_ << "\n";
    cout << " the " << lenCRFieldList_ << " fieldIDs:\n";
	for (int i = 0; i < lenCRFieldList_; ++i) {
    	cout << "   " << CRFieldList_[i];
    }
    cout << "\n\n";
    return;
}


//======================================================================
void MultConstRecord::remoteNode(GlobalID nodeID) {

    int i, j;

    for(i=0; i<numRowsNodeTable_; i++) {
        for(j=0; j<numColsNodeTable_; j++) {
            if (CRNodeTable_[i][j] == nodeID)
                CRIsLocalTable_[i][j] = false;
        }
    }
    
    return;
}

