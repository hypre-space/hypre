#include "CFE_Elem.h"
#include <stdlib.h>
#include <assert.h>

/*
//  Element class for a simple-minded 1D beam program
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
void cfn_FE_Elem_FE_Elem(FE_Elem *myElem) {

	(*myElem).numForceDOF_ = 2;
	(*myElem).numElemNodes_ = 2;
	(*myElem).numElemRows_ = 4;
	(*myElem).nodeList_ = NULL;
	(*myElem).elemLoad_ = NULL;
	(*myElem).elemStiff_ = NULL;
	(*myElem).elemForce_ = NULL;
	(*myElem).distLoad_ = 0.0;
	(*myElem).foundStiff_ = 0.0;
	(*myElem).bendStiff_ = 1.0;
	(*myElem).elemLength_ = 1.0;
}


/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Elem_destroyFE_Elem(FE_Elem *myElem) {
    
    free ((*myElem).elemForce_);

/*
//  the element load and stiffness are deleted in the main driver 
//  program under the current implementation, so we better not get 
//  rid of them again here!
*/

}


GlobalID cfn_FE_Elem_getGlobalElemID(FE_Elem *myElem)
	{return (*myElem).globalElemID_;}
	
void cfn_FE_Elem_putGlobalElemID(FE_Elem *myElem, GlobalID gNID)
	{(*myElem).globalElemID_ = gNID;}


int cfn_FE_Elem_getNumElemRows(FE_Elem *myElem)
	{return (*myElem).numElemRows_;}
	
void cfn_FE_Elem_putNumElemRows(FE_Elem *myElem, int gNERows)
	{(*myElem).numElemRows_ = gNERows;}


int cfn_FE_Elem_getNumElemNodes(FE_Elem *myElem)
	{return (*myElem).numElemNodes_;}
	
void cfn_FE_Elem_putNumElemNodes(FE_Elem *myElem, int gNodes)
	{(*myElem).numElemNodes_ = gNodes;}


int cfn_FE_Elem_getNumForceDOF(FE_Elem *myElem)
	{return (*myElem).numForceDOF_;}
	
void cfn_FE_Elem_putNumForceDOF(FE_Elem *myElem, int gForceDOF)
	{(*myElem).numForceDOF_ = gForceDOF;}


double cfn_FE_Elem_getDistLoad(FE_Elem *myElem)
	{return (*myElem).distLoad_;}

void cfn_FE_Elem_putDistLoad(FE_Elem *myElem, double gLoad)
	{(*myElem).distLoad_ = gLoad;}


double cfn_FE_Elem_getBendStiff(FE_Elem *myElem)
	{return (*myElem).bendStiff_;}

void cfn_FE_Elem_putBendStiff(FE_Elem *myElem, double gStiff)
	{(*myElem).bendStiff_ = gStiff;}


double cfn_FE_Elem_getFoundStiff(FE_Elem *myElem)
	{return (*myElem).foundStiff_;}

void cfn_FE_Elem_putFoundStiff(FE_Elem *myElem, double gStiff)
	{(*myElem).foundStiff_ = gStiff;}


double cfn_FE_Elem_getElemLength(FE_Elem *myElem)
	{return (*myElem).elemLength_;}

void cfn_FE_Elem_putElemLength(FE_Elem *myElem, double gLength)
	{(*myElem).elemLength_ = gLength;}


/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Elem_allocateNodeList(FE_Elem *myElem) {

    int i;
	FE_Elem elemPtr;
	elemPtr = *myElem;
    
    (*myElem).nodeList_ = malloc((*myElem).numElemNodes_*sizeof(GlobalID));
    for (i = 0; i < (*myElem).numElemNodes_; i++) {
        (*myElem).nodeList_[i] = 0;
    }
    return;
}


/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Elem_storeNodeList(FE_Elem *myElem, int neNodes, GlobalID *elemNodes) {

/*
//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  nodes before doing anything inconsistent
*/

	FE_Elem elemPtr;
	elemPtr = *myElem;
    
    assert (elemPtr.numElemNodes_ == neNodes);

    assert (elemPtr.numElemNodes_ == 2);   /*// two nodes per element here... */
    
    elemPtr.nodeList_[0] = elemNodes[0];
    elemPtr.nodeList_[1] = elemNodes[1];

    return;
}


/*
//-----------------------------------------------------------------------
*/
GlobalID *cfn_FE_Elem_returnNodeList(FE_Elem *myElem, int *neNodes) {

/*
//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  nodes before doing anything inconsistent
*/

	FE_Elem elemPtr;
	elemPtr = *myElem;
    
    assert (elemPtr.numElemNodes_ == 2);   /*// two nodes per element here... */
    
    *neNodes = elemPtr.numElemNodes_;
    return(elemPtr.nodeList_);
}



/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Elem_allocateLoad(FE_Elem *myElem) {

    int i;
	FE_Elem elemPtr;
	elemPtr = *myElem;
        
    (*myElem).elemLoad_ = malloc(elemPtr.numElemRows_*sizeof(double));
    for (i = 0; i < elemPtr.numElemRows_; i++)
        (*myElem).elemLoad_[i] = 0.0;
    return;
}


/*
//-----------------------------------------------------------------------
*/
double *cfn_FE_Elem_evaluateLoad(FE_Elem *myElem, int *neRows) {

	FE_Elem elemPtr;
	elemPtr = *myElem;
    
/*
//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  rows before doing anything inconsistent
//
//  eventually, one could put a big case statement here (or an appropriate
//  class hierarchy), but this is just a driver to test ISIS++, so what 
//  the heck...
*/

    assert (elemPtr.numElemRows_ == 4);   /*// two nodes, each with two unknowns*/
    *neRows = elemPtr.numElemRows_;
    
    (*myElem).elemLoad_[0] = 0.5*elemPtr.elemLength_*elemPtr.distLoad_;
    (*myElem).elemLoad_[2] = elemPtr.elemLoad_[0];
    (*myElem).elemLoad_[1] = elemPtr.elemLoad_[0]*elemPtr.elemLength_/6.0;
    (*myElem).elemLoad_[3] = -elemPtr.elemLoad_[1];

    return((*myElem).elemLoad_);
}



/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Elem_allocateStiffness(FE_Elem *myElem) {

    int i, j;
	FE_Elem elemPtr;
	elemPtr = *myElem;
    
    (*myElem).elemStiff_ = malloc(elemPtr.numElemRows_*sizeof(double*));
    for (i = 0; i < elemPtr.numElemRows_; i++) {
        (*myElem).elemStiff_[i] = malloc(elemPtr.numElemRows_*sizeof(double));
        for (j = 0; j < elemPtr.numElemRows_; j++) {
            (*myElem).elemStiff_[i][j] = 0.0;
        }
    }
    return;
}


/*
//-----------------------------------------------------------------------
*/
double **cfn_FE_Elem_evaluateStiffness(FE_Elem *myElem, int *neRows) {

/*
//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  rows before doing anything inconsistent
*/

    double bend_term, found_term;
    int i, j;
	FE_Elem elemPtr;
	elemPtr = *myElem;
    
    assert (elemPtr.numElemRows_ == 4);   /*// two nodes, each with two unknowns*/
    *neRows = elemPtr.numElemRows_;
    
    bend_term = 2.0*elemPtr.bendStiff_/
    			(elemPtr.elemLength_*elemPtr.elemLength_*elemPtr.elemLength_);
    found_term = elemPtr.elemLength_*elemPtr.foundStiff_/3.0;

    for (i = 0; i < elemPtr.numElemRows_; i++) {
        for (j = 0; j < elemPtr.numElemRows_; j++) {
            (*myElem).elemStiff_[i][j] = bend_term;
        }
    }

/*
//  evaluate the upper-triangle of the symmetric stiffness matrix
*/
    
    (*myElem).elemStiff_[0][0] *= 6.0;
    (*myElem).elemStiff_[0][1] *= 3.0*elemPtr.elemLength_;
    (*myElem).elemStiff_[0][2] = -elemPtr.elemStiff_[0][0];
    (*myElem).elemStiff_[0][3] = elemPtr.elemStiff_[0][1];

    (*myElem).elemStiff_[1][1] *= 2.0*elemPtr.elemLength_*elemPtr.elemLength_;
    (*myElem).elemStiff_[1][2] = -elemPtr.elemStiff_[0][1];
    (*myElem).elemStiff_[1][3] = 0.5*elemPtr.elemStiff_[1][1];

    (*myElem).elemStiff_[2][2] = elemPtr.elemStiff_[0][0];
    (*myElem).elemStiff_[2][3] = elemPtr.elemStiff_[1][2];

    (*myElem).elemStiff_[3][3] = elemPtr.elemStiff_[1][1];

/*
//  fill in the lower-triangle of the symmetric stiffness matrix
*/
    
    for (i = 1; i < elemPtr.numElemRows_; i++) {
        for (j = 0; j < i; j++) {
            (*myElem).elemStiff_[i][j] = (*myElem).elemStiff_[j][i];
        }
    }

/*
//  add the foundation stiffness terms, which (if nonzero) have the
//  not-undesirable effect of making the element stiffness matrices 
//  nonsingular (which comes in handy for testing purposes...)
*/

    (*myElem).elemStiff_[0][0] += found_term;
    (*myElem).elemStiff_[2][2] += found_term;
    (*myElem).elemStiff_[0][2] += 0.5*found_term;
    (*myElem).elemStiff_[2][0] += 0.5*found_term;
    

    return((*myElem).elemStiff_);
}



/*
//-----------------------------------------------------------------------
*/
double *cfn_FE_Elem_evaluateElemForces(FE_Elem *myElem, int *gForceDOF) {

/*
//  we haven't implemented any elements except the simple 2-node Hermite
//  bending element, so check to insure that we've got the right number of
//  rows before doing anything inconsistent
*/

    double dispLeft, slopeLeft, dispRight, slopeRight;
    double bendFactor, shearFactor;
	FE_Elem elemPtr;
	elemPtr = *myElem;
    
    assert (elemPtr.numElemNodes_ == 2);   /*// two nodes, each with two unknowns*/
    assert (elemPtr.numElemRows_ == 4);
    
    *gForceDOF = elemPtr.numForceDOF_;
    
/*
//  evaluate the element forces (moment, shear) at the element center, 
//  as it's simple and reasonably accurate there (and we aren't writing
//  a production beam program, just making sure everything works!)

//  need to evaluate the nodal solution params here... 
//  how best to do this?

//  dispLeft = 
//  slopeLeft = 
//  dispLeft = 
//  slopeRight = 

//  bending moment resultant
*/

    bendFactor = elemPtr.bendStiff_/elemPtr.elemLength_;
    (*myElem).elemForce_[0] = bendFactor*(slopeRight - slopeLeft);

/*
//  transverse shear force resultant
*/

    bendFactor = 6.0*elemPtr.bendStiff_/(elemPtr.elemLength_*elemPtr.elemLength_);
    shearFactor = 2.0*bendFactor/elemPtr.elemLength_;
    (*myElem).elemForce_[1] = bendFactor*(slopeRight + slopeLeft) +
                           shearFactor*(dispLeft - dispLeft);

    return((*myElem).elemForce_);
}


/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Elem_allocateElemForces(FE_Elem *myElem) {

    int i;
	FE_Elem elemPtr;
	elemPtr = *myElem;
    
    (*myElem).elemForce_ = malloc(elemPtr.numForceDOF_*sizeof(double));
    for (i = 0; i < elemPtr.numForceDOF_; i++)
        (*myElem).elemForce_[i] = 0.0;
    return;
}


/*
//-----------------------------------------------------------------------
*/
void cfn_FE_Elem_dumpToScreen(FE_Elem *myElem) {

    int i;
	FE_Elem elemPtr;
	elemPtr = *myElem;
    
    printf(" globalElemID_ = %d\n", (int)elemPtr.globalElemID_);
    printf(" numElemRows_  = %d\n", elemPtr.numElemRows_);
    printf(" numForceDOF_  = %d\n", elemPtr.numForceDOF_);
    printf(" distLoad_     = %f\n", elemPtr.distLoad_ );
    printf(" bendStiff_    = %f\n", elemPtr.bendStiff_);
    printf(" foundStiff_   = %f\n", elemPtr.foundStiff_);
    printf(" elemLength_   = %f\n", elemPtr.elemLength_);
    printf(" the %d nodes:  ", elemPtr.numElemNodes_);
    for (i = 0; i < elemPtr.numElemNodes_; ++i) {
    	printf("   %d", (int)elemPtr.nodeList_[i]);
    }
    printf("\n");
    printf(" the %d load vector terms:  ", elemPtr.numElemRows_);
    for (i = 0; i < elemPtr.numElemRows_; ++i) {
    	printf("   %f", elemPtr.elemLoad_[i]);
    }
    printf("\n\n");
    return;
}
