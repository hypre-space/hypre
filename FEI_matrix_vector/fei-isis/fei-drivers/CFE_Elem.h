#ifndef __CFE_Elem_H
#define __CFE_Elem_H

#include <stdio.h>
#include <assert.h>

#include "other/basicTypes.h"

/*
//  Element class for a simple 1D beam on elastic foundation program
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
*/

struct FE_Elem {
    GlobalID globalElemID_;
    int numElemNodes_;
    int numElemRows_;
    int numForceDOF_;
    
    GlobalID *nodeList_;
    double *elemLoad_;
    double **elemStiff_;

    double *elemForce_;
    
    double elemLength_;
    double distLoad_;
    double bendStiff_;
    double foundStiff_;
};

typedef struct FE_Elem FE_Elem;

void cfn_FE_Elem_FE_Elem(FE_Elem *myElem);
void cfn_FE_Elem_destroyFE_Elem(FE_Elem *myElem);

GlobalID cfn_FE_Elem_getGlobalElemID(FE_Elem *myElem);
void cfn_FE_Elem_putGlobalElemID(FE_Elem *myElem, GlobalID gNID);

int cfn_FE_Elem_getNumElemRows(FE_Elem *myElem);
void cfn_FE_Elem_putNumElemRows(FE_Elem *myElem, int gNERows);

int cfn_FE_Elem_getNumElemNodes(FE_Elem *myElem);
void cfn_FE_Elem_putNumElemNodes(FE_Elem *myElem, int gNodes);

int cfn_FE_Elem_getNumForceDOF(FE_Elem *myElem);
void cfn_FE_Elem_putNumForceDOF(FE_Elem *myElem, int gForceDOF);

double cfn_FE_Elem_getDistLoad(FE_Elem *myElem);
void cfn_FE_Elem_putDistLoad(FE_Elem *myElem, double gLoad);

double cfn_FE_Elem_getBendStiff(FE_Elem *myElem);
void cfn_FE_Elem_putBendStiff(FE_Elem *myElem, double gStiff);

double cfn_FE_Elem_getFoundStiff(FE_Elem *myElem);
void cfn_FE_Elem_putFoundStiff(FE_Elem *myElem, double gStiff);

double cfn_FE_Elem_getElemLength(FE_Elem *myElem);
void cfn_FE_Elem_putElemLength(FE_Elem *myElem, double gLength);

void cfn_FE_Elem_allocateNodeList(FE_Elem *myElem);
void cfn_FE_Elem_storeNodeList(FE_Elem *myElem, int gNumNodes, GlobalID *elemNodes);
GlobalID *cfn_FE_Elem_returnNodeList(FE_Elem *myElem, int *gNumNodes);

void cfn_FE_Elem_allocateElemForces(FE_Elem *myElem);
double *cfn_FE_Elem_evaluateElemForces(FE_Elem *myElem, int *gForceDOF);

void cfn_FE_Elem_allocateLoad(FE_Elem *myElem);
double *cfn_FE_Elem_evaluateLoad(FE_Elem *myElem, int *neRows);

void cfn_FE_Elem_allocateStiffness(FE_Elem *myElem);
double **cfn_FE_Elem_evaluateStiffness(FE_Elem *myElem, int *neRows);

void cfn_FE_Elem_dumpToScreen(FE_Elem *myElem);


#endif

