#include <string.h>
#include <assert.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "FE_Elem.h"

//  Element class for a simple-minded 1D beam program
//
//  k.d. mish, august 13, 1997
//
//  kludged for purposes of testing constraint relations kdm 9/29/97
//  (modified line indicated with trailing comments of the form "//kdm"
//
//  this program is used for testing the ISIS++ finite-element interface
//  module, as a 1D beam is (a) easy to partition among an arbitrary
//  arrangement of parallel processors, (b) easy to check answers for (since
//  various closed-form solutions exist), and (c) possesses sufficiently
//  interesting data (e.g., multiple nodal solution parameters) so that one
//  can perform reasonable code coverage checks...


//-----------------------------------------------------------------------
FE_Elem::FE_Elem() {

    globalElemID_ = 0;   
    numElemNodes_ = 2;
    numElemRows_ = 6;

    nodeList_ = NULL;
    elemLoad_ = NULL;
    elemStiff_ = NULL;

    distLoad_ = 0.0;
    axialLoad_ = 0.0;
    bendStiff_ = 0.0;
    axialStiff_ = 0.0;
    foundStiff_ = 0.0;
    elemLength_ = 0.0;

}


//-----------------------------------------------------------------------
FE_Elem::~FE_Elem() {
    
    delete [] nodeList_;

}

//-----------------------------------------------------------------------

void FE_Elem::allocateNodeList() {

    int i;
    
    nodeList_ = new GlobalID[numElemNodes_];
    for (i = 0; i < numElemNodes_; i++)
        nodeList_[i] = 0;
    return;
}


//-----------------------------------------------------------------------

void FE_Elem::storeNodeList(int neNodes, GlobalID *elemNodes) {

//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  nodes before doing anything inconsistent

    assert (numElemNodes_ == neNodes);

    assert (numElemNodes_ == 2);   // here, two nodes
    
    nodeList_[0] = elemNodes[0];
    nodeList_[1] = elemNodes[1];

    return;
}


//-----------------------------------------------------------------------

void FE_Elem::returnNodeList(int& neNodes, GlobalID *listPtr) {

//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  nodes before doing anything inconsistent

    assert (numElemNodes_ == 2);   // here, two nodes
    
    int i;
    for(i=0; i<numElemNodes_; i++){
        listPtr[i] = nodeList_[i];
    }

    neNodes = numElemNodes_;
    return;
}


//-----------------------------------------------------------------------

void FE_Elem::evaluateLoad(int& neRows, double* loadPtr) {

//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  rows before doing anything inconsistent
//
//  eventually, one could put a big case statement here (or an appropriate
//  class hierarchy), but this is just a driver to test ISIS++, so what 
//  the heck...
//
//  storage here is axial and transverse displacement, then rotation

    assert (numElemRows_ == 6);   // two nodes, each with three unknowns...
    neRows = numElemRows_;
    
    elemLoad_ = loadPtr;

    elemLoad_[0] = 0.5*elemLength_*axialLoad_;
    elemLoad_[3] = elemLoad_[0];
    elemLoad_[1] = 0.5*elemLength_*distLoad_;
    elemLoad_[4] = elemLoad_[1];
    elemLoad_[2] = elemLoad_[1]*elemLength_/6.0;
    elemLoad_[5] = -elemLoad_[2];

    return;
}


//-----------------------------------------------------------------------

void FE_Elem::evaluateStiffness(int& neRows, double **stiffPtr) {

//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  rows before doing anything inconsistent
//
//  storage here is axial and transverse displacement, then rotation

    assert (numElemRows_ == 6);   // two nodes, each with three unknowns...
    neRows = numElemRows_;
    
    double bend_term, found_term, axial_term;
    int i, j;
    
    bend_term = 2.0*bendStiff_/(elemLength_*elemLength_*elemLength_);
    found_term = elemLength_*foundStiff_/3.0;
    axial_term = axialStiff_/elemLength_;

    elemStiff_ = stiffPtr;

//  zero out matrix just to be safe about its initialization

    for (i = 0; i < numElemRows_; i++) {
        for (j = 0; j < numElemRows_; j++) {
            elemStiff_[i][j] = 0.0;
        }
    }

//  evaluate the upper-triangle of the symmetric stiffness matrix

    elemStiff_[0][0] = axial_term;
    elemStiff_[3][3] = axial_term;
    elemStiff_[0][3] = -axial_term;
    
    elemStiff_[1][1] = 6.0*bend_term;
    elemStiff_[1][2] = 3.0*bend_term*elemLength_;
    elemStiff_[1][4] = -elemStiff_[1][1];
    elemStiff_[1][5] = elemStiff_[1][2];

    elemStiff_[2][2] = 2.0*bend_term*elemLength_*elemLength_;
    elemStiff_[2][4] = -elemStiff_[1][2];
    elemStiff_[2][5] = 0.5*elemStiff_[2][2];

    elemStiff_[4][4] = elemStiff_[1][1];
    elemStiff_[4][5] = elemStiff_[2][4];

    elemStiff_[5][5] = elemStiff_[2][2];

//  fill in the lower-triangle of the symmetric stiffness matrix
    
    for (i = 1; i < numElemRows_; i++) {
        for (j = 0; j < i; j++) {
            elemStiff_[i][j] = elemStiff_[j][i];
        }
    }

//  add the foundation stiffness terms, which (if nonzero) have the
//  not-undesirable effect of making the element stiffness matrices 
//  nonsingular (which comes in handy for testing purposes...)

    elemStiff_[1][1] += found_term;
    elemStiff_[4][4] += found_term;
    elemStiff_[1][4] += 0.5*found_term;
    elemStiff_[4][1] += 0.5*found_term;
    
    return;
}


//-----------------------------------------------------------------------

void FE_Elem::dumpToScreen() {

    int i;
    
    cout << " globalElemID_ = " << (int)globalElemID_ << endl;
    cout << " numElemRows_  = " << numElemRows_ << endl;
    cout << " distLoad_     = " << distLoad_ << endl;
    cout << " axialLoad_    = " << axialLoad_ << endl;
    cout << " bendStiff_    = " << bendStiff_ << endl;
    cout << " axialStiff_   = " << axialStiff_ << endl;
    cout << " foundStiff_   = " << foundStiff_ << endl;
    cout << " elemLength_   = " << elemLength_ << endl;
    cout << " the " << numElemNodes_ << " nodes: ";
    for (i = 0; i < numElemNodes_; ++i) {
    	cout << "   " << (int)nodeList_[i];
    }
    cout << endl;

    return;
}
