#include <stdio.h>
#include <iostream.h>
#include <assert.h>

#include "other/basicTypes.h"
#include "FE_Node.h"

//  Node class for a simple-minded 1D beam program
//
//  k.d. mish, august 13, 1997
//
//  this program is used for testing the ISIS++ finite-element interface
//  module, as a 1D beam is (a) easy to partition among an arbitrary
//  arrangement of parallel processors, (b) easy to check answers for (since
//  various closed-form solutions exist), and (c) possesses sufficiently
//  interesting data (e.g., multiple nodal solution parameters) so that one
//  can perform reasonable code coverage checks...


//-----------------------------------------------------------------------

FE_Node::FE_Node() {

    globalNodeID_ = 0;
    numNodalDOF_ = 3;
    nodePosition_ = 0.0;
    
    nodSoln_ = NULL;
     
}


//-----------------------------------------------------------------------

FE_Node::~FE_Node() {

    if (nodSoln_ != NULL)
        delete [] nodSoln_;

}


//-----------------------------------------------------------------------

double *FE_Node::pointerToSoln(int& numDOF) {
    numDOF = numNodalDOF_;
    return(nodSoln_);
}


//-----------------------------------------------------------------------

void FE_Node::allocateSolnList() {
    int i;
    nodSoln_ = new double[numNodalDOF_];
    for (i = 0; i < numNodalDOF_; i++)
        nodSoln_[i] = 0.0;
    return;
}



//-----------------------------------------------------------------------

void FE_Node::dumpToScreen() {
    printf(" globalNodeID_ = %d\n",(int)globalNodeID_);
    cout << " numNodalDOF_ = " << numNodalDOF_ << endl;
    cout << " nodePosition_ = " << nodePosition_ << endl;
    cout << " the " << numNodalDOF_ << " nodal solution parameters: ";
	for (int i = 0; i < numNodalDOF_; ++i) {
    	cout << "   " << nodSoln_[i];
    }
    cout << endl << endl;
    return;
}
