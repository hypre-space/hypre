#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "src/Utils.h"

#include "src/BlockDescriptor.h"

//====Constructor===============================================================
BlockDescriptor::BlockDescriptor()
 : blockID_((GlobalID)-1),
   numNodesPerElement_(0),
   numFieldsPerNode_(NULL),
   nodalFieldIDs_(NULL),
   fieldIDsAllocated_(false),
   interleaveStrategy_(0),
   lumpingStrategy_(0),
   numElements_(0),
   numElemDOFPerElement_(0),
   elemDOFEqnNumbers_(NULL),
   numEqnsPerElement_(0),
   numActiveNodes_(0),
   totalNumEqns_(0)
{
   //There's nothing else for this constructor to do.
}

//====Destructor================================================================
BlockDescriptor::~BlockDescriptor() {

   destroyFieldArrays();

   delete [] elemDOFEqnNumbers_;
   elemDOFEqnNumbers_ = NULL;
   numElemDOFPerElement_ = 0;
}

//==============================================================================
void BlockDescriptor::destroyFieldArrays() {
   if (numNodesPerElement_ == 0) return;

   for(int i=0; i<numNodesPerElement_; i++) {
      delete [] nodalFieldIDs_[i];
   }

   delete [] nodalFieldIDs_;
   nodalFieldIDs_ = NULL;
   delete [] numFieldsPerNode_;
   numFieldsPerNode_ = NULL;
   numNodesPerElement_ = 0;
}

//==============================================================================
void BlockDescriptor::setNumNodesPerElement(int numNodes) {

   if (numNodes <= 0) {
      cerr << "BlockDescriptor::setNumNodesPerElement: ERROR, numNodes <= 0."
           << endl;
      return;
   }

   destroyFieldArrays();

   numNodesPerElement_ = numNodes;

   numFieldsPerNode_ = new int[numNodesPerElement_];

   for(int i=0; i<numNodesPerElement_; i++) {
      numFieldsPerNode_[i] = 0;
   }
}

//==============================================================================
int* BlockDescriptor::fieldsPerNodePtr() {

   return(numFieldsPerNode_);
}

//==============================================================================
void BlockDescriptor::allocateFieldIDsTable() {

   if (numNodesPerElement_ == 0) {
      cerr << "BlockDescriptor::allocateFieldIDsTable: ERROR, must call "
           << "setNumNodesPerElement, and set num-fields-per-node first."
           << endl;
      return;
   }

   nodalFieldIDs_ = new int*[numNodesPerElement_];
   bool rowsAllZeroLength = true;

   for(int i=0; i<numNodesPerElement_; i++) {
      if (numFieldsPerNode_[i] > 0) {
         nodalFieldIDs_[i] = new int[numFieldsPerNode_[i]];
         rowsAllZeroLength = false;
      }
      else nodalFieldIDs_[i] = NULL;
   }

   if (rowsAllZeroLength) {
      cerr << "BlockDescriptor::allocateFieldIDsTable: ERROR, all rows of"
           << " fieldIDs table have zero length. Set fieldsPerNode entries"
           << " first." << endl;
      return;
   }

   fieldIDsAllocated_ = true;
}

//==============================================================================
int** BlockDescriptor::fieldIDsTablePtr() {

   if (!fieldIDsAllocated_) {
      cerr << "BlockDescriptor::fieldIDsTablePtr: WARNING, fieldIDs not"
           << " allocated. Returning NULL." << endl;
      return(NULL);
   }
   
   return(nodalFieldIDs_);
}

//==============================================================================
bool BlockDescriptor::containsField(int fieldID) {
//
//This function will mostly be called by the BASE_FEI function for
//getting solutions to return to the user.
//
//For cases where each of the nodes in an element have the same fields,
//this function will be quite fast.
//
//It will be slow for cases where there are quite a few nodes per element
//and the different nodes have different solution fields. (i.e., the search
//below has to step through most of the fieldIDs table before finding the
//fieldID in question.
//
//In general though, this function won't be called if the fieldID isn't
//associated with ANY node in this block, because the calling code can first
//query the node to find out if IT is associated with this block. And if the
//node is associated with this block, then the node's fields usually will be
//also, unless the node lies on a block boundary and 'fieldID' is only in
//the other block.
//
   for(int i=0; i<numNodesPerElement_; i++) {
      for(int j=0; j<numFieldsPerNode_[i]; j++) {
         if (nodalFieldIDs_[i][j] == fieldID) return(true);
      }
   }

   return(false);
}

//==============================================================================
void BlockDescriptor::setNumElemDOFPerElement(int numDOF) {

   numElemDOFPerElement_ = numDOF;

//   if (numElements_ <= 0) {
//      cerr << "BlockDescriptor::setNumElemDOFPerElement: WARNING, "
//           << "numElements_ <= 0." << endl;
//      return;
//   }

   elemDOFEqnNumbers_ = new int[numElements_];

   for(int i=0; i<numElements_; i++) {
      elemDOFEqnNumbers_[i] = 0;
   }
}

//==============================================================================
int* BlockDescriptor::elemDOFEqnNumbersPtr() {

   return(elemDOFEqnNumbers_);
}

