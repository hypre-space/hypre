#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
//#include "mv/RealArray.h"
//#include "mv/IntArray.h"
//#include "mv/GlobalIDArray.h"
#include "isis-mv/RealArray.h"
#include "isis-mv/IntArray.h"
#include "isis-mv/GlobalIDArray.h"

#include "src/SLE_utils.h"
#include "src/NodePackets.h"
#include "src/CommBufferDouble.h"

//==============================================================================
CommBufferDouble::CommBufferDouble() {

    //set tables to null and numDestProcs_ to 0.
    initializeTables();

    addEqnNumbers_ = new IntArray();
    addDoubles_ = NULL;
    addDestProcs_ = new IntArray();
}

//==============================================================================
CommBufferDouble::~CommBufferDouble() {

    destroyTables();

    for(int i=0; i<addDestProcs_->size(); i++) {
        delete addDoubles_[i];
    }
    delete addEqnNumbers_;
    delete addDestProcs_;
}

//==============================================================================
void CommBufferDouble::initializeTables() {
//
//This function possibly needs a better name. It doesn't
//create or allocate the tables, it simply initializes them
//to null and resets numDestProcs_ (which is the variable
//that keeps track of how many rows each table has).
//
    eqnNumbers_ = NULL;
    doubles_ = NULL;
    offsets_ = NULL;
    destProcs_ = NULL;
    numDestProcs_ = 0;
}

//==============================================================================
void CommBufferDouble::destroyTables() {
//
//Destroys 'em if they've been created.
//
    if (numDestProcs_ > 0) {
        delete [] eqnNumbers_;
        delete [] doubles_;
        delete [] offsets_;
        delete destProcs_;
        numDestProcs_ = 0;
    }
}

//==============================================================================
void CommBufferDouble::addDoubles(int eqnNum, const double* coefs, int numCoefs,
                             int destProc) {

    //How many sets of coefs have previously been added?
    int previousSize = addEqnNumbers_->size();

    //add in the new equation number.
    addEqnNumbers_->append(eqnNum);

    //create a new bigger list of RealArrays
    RealArray** temp = new RealArray*[previousSize + 1];

    //copy the previously added RealArray pointers into the new list
    for(int i=0; i<previousSize; i++) {
        temp[i] = addDoubles_[i];
    }

    //copy the new coefficients into a RealArray, in the new temp list
    temp[previousSize] = new RealArray(numCoefs);
    for(int j=0; j<numCoefs; j++) {
        (*(temp[previousSize]))[j] = coefs[j];
    }

    //reset the pointer to the new memory, after discarding the old
    //pointer list.
    delete [] addDoubles_;
    addDoubles_ = temp;

    //finally, add the new destination processor to the list of procs.
    addDestProcs_->append(destProc);
}

//==============================================================================
void CommBufferDouble::buildLists() {

    //set tables to null and numDestProcs_ to 0.
    destroyTables();
    initializeTables();

    //allocate and fill the list of destination processors.
    buildProcList();

    //allocate and fill the eqnNumbers_, doubles_, and offsets_ tables.
    buildTables();
}

//==============================================================================
void CommBufferDouble::buildProcList() {

    int size = addDestProcs_->size();

    if (size==0) {
        return;
    }

    destProcs_ = new IntArray();
    numDestProcs_ = 0;

    destProcs_->append((*addDestProcs_)[0]);
    numDestProcs_++;

    for(int i=1; i<size; i++) {
        int found = -1, insert = 0;
        int proc = (*addDestProcs_)[i];

        found = find_index(proc, &((*destProcs_)[0]),
                           destProcs_->size(), &insert);

        if (found<0) {
            destProcs_->insert(insert, proc);
            numDestProcs_++;
        }
    }
}

//==============================================================================
void CommBufferDouble::buildTables() {
//
//This function assumes that the destProcs_ list has already been built.
//
    int numRows = numDestProcs_;

    if (numDestProcs_ <= 0) return;

    //now we're allocating lists of arrays -- not arrays of size numRows!

    eqnNumbers_ = new IntArray[numRows];
    doubles_ = new RealArray[numRows];
    offsets_ = new IntArray[numRows];

    int addSize = addEqnNumbers_->size();

    for(int i=0; i<addSize; i++) {
        int proc = (*addDestProcs_)[i];

        int insert = 0;
        int procIndex = find_index(proc, &((*destProcs_)[0]),
                               destProcs_->size(), &insert);

        if (procIndex<0) {
            cerr << "CommBufferDouble::buildTables: ERROR, proc not found."
                 << " Aborting." << endl;
            abort();
        }

        eqnNumbers_[procIndex].append((*addEqnNumbers_)[i]);

        int thisOffset = doubles_[procIndex].size();
        offsets_[procIndex].append(thisOffset);

        //now for all elements j of the ith addDoubles_ array, append them
        //to the doubles_[procIndex] array.

        for(int j=0; j<(addDoubles_[i])->size(); j++) {
            doubles_[procIndex].append( (*(addDoubles_[i]))[j] );
        }
    }

    //now append the final size of the doubles arrays to the offsets arrays
    //so that for each eqnNumber i, numDoubles[i] = offsets_[i+1]-offsets_[i]
    //
    for(int j=0; j<numRows; j++) {
        offsets_[j].append(doubles_[j].size());
    }
}

//==============================================================================
int* CommBufferDouble::pointerToEqnNumbers(int& numEqnNumbers, int destProc) {

    if (numDestProcs_ <= 0) {
        numEqnNumbers = 0;
        return(NULL);
    }
    else {
        int index = -1, insert = 0;
        index = find_index(destProc, &((*destProcs_)[0]), destProcs_->size(),
                       &insert); 
        if (index<0) {
            cerr << "CommBufferDouble::pointerEqnNumbers: index<0 aborting."
                 << endl;
            abort();
        }

        numEqnNumbers = eqnNumbers_[index].size();
        return(&((eqnNumbers_[index])[0]));
    }
}

//==============================================================================
double* CommBufferDouble::pointerToDoubles(int& numDoubles, int destProc) {

    if (numDestProcs_ <= 0) {
        numDoubles = 0;
        return(NULL); 
    }   
    else {
        int index = -1, insert = 0;
        index = find_index(destProc, &((*destProcs_)[0]), destProcs_->size(),
                       &insert); 
        if (index<0) {
            cerr << "CommBufferDouble::pointerToDoubles: index<0 aborting."
                 << endl;
            abort();
        }

        numDoubles = doubles_[index].size();
        return(&((doubles_[index])[0]));
    }
}

//==============================================================================
int* CommBufferDouble::pointerToOffsets(int& numOffsets, int destProc) {

    if (numDestProcs_ <= 0) {
        numOffsets = 0;
        return(NULL); 
    }   
    else {
        int index = -1, insert = 0;
        index = find_index(destProc, &((*destProcs_)[0]), destProcs_->size(),
                       &insert); 
        if (index<0) {
            cerr << "CommBufferDouble::pointerToOffsets: index<0 aborting."
                 << endl;
            abort();
        }

        numOffsets = offsets_[index].size();
        return(&((offsets_[index])[0]));
    }

}

//==============================================================================
int* CommBufferDouble::pointerToProcs(int& numProcs) {
    numProcs = numDestProcs_;

    if (numProcs <= 0) return(NULL);

    return(&((*destProcs_)[0]));
}

