#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "other/basicTypes.h"
#include "src/FieldRecord.h"

//------------------------------------------------------------------------------
//
//  Field is a really simple utility class for storing some information 
//  about solution fields (collections of scalar solution parameters that
//  define the multiphysics results at the nodes).
//
//  This barely rates a class definition, but I have a sneaking suspicion 
//  that this abstraction is gonna get more important with time, so we 
//  might as well give it some room from the get-go, eh?
//
//  kdm Oct 18, 1998
//
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//  fieldRecord constructor/destructor
//------------------------------------------------------------------------------

FieldRecord::FieldRecord() {

    fieldID_ = 0; 
    numFieldParams_ = 0;
    
    return;
}


FieldRecord::~FieldRecord() {

    return;  // not much going on here, eh?
}


//------------------------------------------------------------------------------
//  member functions to set/get field scalar internals
//------------------------------------------------------------------------------

int FieldRecord::getFieldID() {
	return (fieldID_);
}
void FieldRecord::setFieldID(int fieldID) {
    fieldID_ = fieldID;
}



int FieldRecord::getNumFieldParams() {
	return (numFieldParams_);
}
void FieldRecord::setNumFieldParams(int numFieldParams) {
    numFieldParams_ = numFieldParams;
}


//------------------------------------------------------------------------------
//  utility member functions 
//------------------------------------------------------------------------------

void FieldRecord::dumpToScreen() {

    cout << " fieldID_         = " << fieldID_ << "\n";
    cout << " numFieldParams_  = " << numFieldParams_ << "\n\n";
    
    return;
}
