#include "CFE_Node.h"
#include <stdlib.h>
#include <assert.h>

/*
//  Node class for a simple-minded 1D beam program
//
//  k.d. mish, august 13, 1997
//  degeneration to ANSI-C struct by k.d. mish, jan, 1998
//
//  this program is used for testing the ISIS++ finite-element interface
//  module, as a 1D beam is (a) easy to partition among an arbitrary
//  arrangement of parallel processors, (b) easy to check answers for (since
//  various closed-form solutions exist), and (c) possesses sufficiently
//  interesting data (e.g., multiple nodal solution parameters) so that one
//  can perform reasonable code coverage checks...
*/

/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Node_FE_Node(FE_Node *myNode) {

    (*myNode).numNodalDOF_ = 2;
    (*myNode).nodSoln_ = NULL; 
}


/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Node_destroyFE_Node(FE_Node *myNode) {

    free ((*myNode).nodSoln_); 
}


GlobalID cfn_FE_Node_getGlobalNodeID(FE_Node *myNode) {
	return (*myNode).globalNodeID_;
}

void cfn_FE_Node_putGlobalNodeID(FE_Node *myNode, GlobalID gNID) {
	(*myNode).globalNodeID_ = gNID;
}

int cfn_FE_Node_getNumNodalDOF(FE_Node *myNode) {
	return (*myNode).numNodalDOF_;
}

void cfn_FE_Node_putNumNodalDOF(FE_Node *myNode, int gNDOF) {
	(*myNode).numNodalDOF_ = gNDOF;
}

double cfn_FE_Node_getNodePosition(FE_Node *myNode) {
	return (*myNode).nodePosition_;
}

void cfn_FE_Node_putNodePosition(FE_Node *myNode, double gPosition) {
	(*myNode).nodePosition_ = gPosition;
}


/*
//-----------------------------------------------------------------------
*/
double *cfn_FE_Node_pointerToSoln(FE_Node *myNode, int *numDOF) {
    *numDOF = (*myNode).numNodalDOF_;
    return((*myNode).nodSoln_);
}


/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Node_allocateSolnList(FE_Node *myNode) {
    int i;
	FE_Node nodePtr;
	nodePtr = *myNode;

    (*myNode).nodSoln_ = malloc(nodePtr.numNodalDOF_*sizeof(double));
    for (i = 0; i < nodePtr.numNodalDOF_; i++)
        (*myNode).nodSoln_[i] = 0.0;
    return;
}



#include <stdio.h>

/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Node_dumpToScreen(FE_Node *myNode) {
	int i;
	FE_Node nodePtr;
	nodePtr = *myNode;

    printf(" globalNodeID_ = %d\n",(int)nodePtr.globalNodeID_);
    printf(" numNodalDOF_ =  %d\n", nodePtr.numNodalDOF_);
    printf(" nodePosition_ = %f\n", nodePtr.nodePosition_);
    printf(" the %d nodal solution parameters:  ", nodePtr.numNodalDOF_);
	for (i = 0; i < nodePtr.numNodalDOF_; ++i) {
    	printf("   %f", nodePtr.nodSoln_[i]);
    }
    printf("\n\n");
    return;
}
