#ifndef _BlockDescriptor_h_
#define _BlockDescriptor_h_

//==============================================================================
//
//The BlockDescriptor class holds the information that the FEI implementation
//needs to know about a block in the finite element problem:
//
//   Global block identifier
//   number of nodes per element in this block
//   list containing the number-of-fields-per-node
//   table containing the field identifiers for those nodal fields
//   interleaveStrategy
//   lumpingStrategy
//   number of elements in this block
//   number of element-DOF per element in this block
//   the first element-DOF equation number for each block
//   total number of equations associated with each element
//
//A block is a collection of homogeneous elements -- which means that all
//elements in a block have the same topology: number of nodes, same solution
//fields at those nodes, etc.
//
//Usage notes:
//
//   There is only one way to set each length/size parameter in this class,
//   and only one way to retrieve each length/size parameter. e.g., you can't
//   set the number of nodes to be one thing, then pass a different value to
//   a function that allocates the fieldsPerNode list. When you set the
//   number of nodes, the fieldsPerNode list is allocated right then.
//   When you retrieve the fieldsPerNode list, you must obtain its length by
//   separately calling the getNumNodesPerElement() function. Similar schemes
//   are in place for the fieldIDsTable, etc.
//
//There are some order-dependencies for these member functions.
//
// - setNumNodesPerElement should be called before fieldsPerNodePtr() is
//   called, because fieldsPerNodePtr returns a list of length numNodesPerElem.
//
// - then, in turn, fieldsPerNodePtr() should be called, and the entries of
//   that list should be set, before allocateFieldIDsTable is called, because
//   the row-lengths of the fieldIDsTable are given by the entries in the
//   fieldsPerNode list.
//
// - naturally allocateFieldIDsTable should be called before fieldIDsTablePtr()
//   is called.
//
// - setNumElemDOFPerElement should be called before elemDOFEqnNumbersPtr(),
//   because elemDOFEqnNumbersPtr returns a list of length numElemDOFPerElem.
//
//In general, out-of-order function calls will be flagged with a message to
//stderr.
//==============================================================================

//requires:
//#include "other/basicTypes.h"

class BlockDescriptor {
 public:
   BlockDescriptor();
   virtual ~BlockDescriptor();

   GlobalID getGlobalBlockID() {return(blockID_);};
   void setGlobalBlockID(GlobalID blockID) {blockID_ = blockID;};

   int getNumNodesPerElement() {return(numNodesPerElement_);};
   void setNumNodesPerElement(int numNodes);

   int* fieldsPerNodePtr();  //length of this list = getNumNodesPerElement()

   void allocateFieldIDsTable();

   int** fieldIDsTablePtr(); //number-of-rows is getNumNodesPerElement()
                             //The row-lengths are given by
                             //the fieldsPerNodePtr() list.

   bool containsField(int fieldID);

   int getInterleaveStrategy() const {return(interleaveStrategy_);};
   void setInterleaveStrategy(int strat) {interleaveStrategy_ = strat;};
   
   int getLumpingStrategy() const {return(lumpingStrategy_);};
   void setLumpingStrategy(int strat) {lumpingStrategy_ = strat;};

   int getNumElements() {return(numElements_);};
   void setNumElements(int numElems) {numElements_ = numElems;};
   
   int getNumElemDOFPerElement() {return(numElemDOFPerElement_);};
   void setNumElemDOFPerElement(int numDOF);

   int* elemDOFEqnNumbersPtr();

   int getNumEqnsPerElement() {return(numEqnsPerElement_);};
   void setNumEqnsPerElement(int numEqns) {numEqnsPerElement_ = numEqns;};

   int getNumActiveNodes() {return(numActiveNodes_);};
   void setNumActiveNodes(int num) {numActiveNodes_ = num;};

   int getTotalNumEqns() {return(totalNumEqns_);};
   void setTotalNumEqns(int numEqns) {totalNumEqns_ = numEqns;};

 private:
   void destroyFieldArrays();

   GlobalID blockID_;
   int numNodesPerElement_;

   int* numFieldsPerNode_; //list: length = numNodesPerElement_

   int** nodalFieldIDs_;   //table: number-of-rows = numNodesPerElement_
                           //       length-of-row[i] = numFieldsPerNode_[i]
   bool fieldIDsAllocated_;

   int interleaveStrategy_;
   int lumpingStrategy_;

   int numElements_;
   int numElemDOFPerElement_; //number of elem-dof in each element (all
                              //elems in block have the same number)

   int* elemDOFEqnNumbers_; //list -- length = numElements_
                            //holds eqn number of each element's first elemDOF

   int numEqnsPerElement_;

   int numActiveNodes_;
   int totalNumEqns_;
};

#endif

