#ifndef __CFE_Node_H
#define __CFE_Node_H

#include <stdio.h>
#include <assert.h>

#include "other/basicTypes.h"

/*
//  Node class for a simple-minded 1D beam program
//
//  original C++ coding by k.d. mish, august 13, 1997
//  degeneration to ANSI-C struct by k.d. mish, jan, 1998
//
//  this program is used for testing the ISIS++ finite-element interface
//  module, as a 1D beam is (a) easy to partition among an arbitrary
//  arrangement of parallel processors, (b) easy to check answers for (since
//  various closed-form solutions exist), and (c) possesses sufficiently
//  interesting data (e.g., multiple nodal solution parameters) so that one
//  can perform reasonable code coverage checks...

//  ANSI-C versions obtained by converting private data to structures, 
//  and explicitly declaring all the public functions (this works here
//  because of the simple decomposition of data and function)
*/

struct FE_Node {
    GlobalID globalNodeID_;
    int numNodalDOF_;
    double *nodSoln_;
    double nodePosition_;
};

typedef struct FE_Node FE_Node;

void cfn_FE_Node_FE_Node(FE_Node *myNode);
void cfn_FE_Node_destroyFE_Node(FE_Node *myNode);

GlobalID cfn_FE_Node_getGlobalNodeID(FE_Node *myNode);
void cfn_FE_Node_putGlobalNodeID(FE_Node *myNode, GlobalID gNID);

int cfn_FE_Node_getNumNodalDOF(FE_Node *myNode);
void cfn_FE_Node_putNumNodalDOF(FE_Node *myNode, int gNDOF);

double cfn_FE_Node_getNodePosition(FE_Node *myNode);
void cfn_FE_Node_putNodePosition(FE_Node *myNode, double gPosition);

double *cfn_FE_Node_pointerToSoln(FE_Node *myNode, int *numDOF);
void cfn_FE_Node_allocateSolnList(FE_Node *myNode);

void cfn_FE_Node_dumpToScreen(FE_Node *myNode);
 
#endif

