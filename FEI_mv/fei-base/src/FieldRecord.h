#ifndef __FieldRecord_H
#define __FieldRecord_H

// #include "basicTypes.h"

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

class FieldRecord {

    public:

        FieldRecord();
        ~FieldRecord();
   
//------------------------------------------------------------------------------
//  member functions to set/get field internals

        int getFieldID();
        void setFieldID(int fieldID);

        int getNumFieldParams();
        void setNumFieldParams(int numFieldParams);

        void dumpToScreen();
        
    private:

//------------------------------------------------------------------------------
//  cached internals from initFields() method

        int fieldID_;           // ID for this field
        int numFieldParams_;    // number of solution parameters for this field
};

#endif

