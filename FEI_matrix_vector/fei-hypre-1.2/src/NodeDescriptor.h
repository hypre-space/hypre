#ifndef _NodeDescriptor_h_
#define _NodeDescriptor_h_

//==============================================================================
//
//The NodeDescriptor class holds the information that the FEI implementation
//needs to know about the nodes in the finite element problem:
//
//   Global node identifier
//   number of nodal degrees-of-freedom
//   list of associated field identifiers, with their (global) equation numbers
//   which processor is this node's owner
//   list of (local) blocks that contain this node
//
//==============================================================================

//requires:
//#include "src/basicTypes.h"

class NodeDescriptor {
 public:
   NodeDescriptor();
   NodeDescriptor(const NodeDescriptor& source);
   virtual ~NodeDescriptor();

   GlobalID getGlobalNodeID() const {return(nodeID_);};
   void setGlobalNodeID(GlobalID node) {nodeID_ = node;};

   int getNumNodalDOF() const {return(numNodalDOF_);};
   void setNumNodalDOF(int dof) {numNodalDOF_ = dof;};

   void addField(int fieldID);
   void setFieldEqnNumber(int fieldID, int eqn);
   int getFieldEqnNumber(int fieldID);
   int getFieldIndex(int fieldID);

   int getNumFields() const {return(numFields_);};
   int* getFieldIDList() const {return(fieldIDList_);};
   int* getFieldEqnNumbers() const {return(fieldEqnNumbers_);};

   int getOwnerProc() const {return(ownerProc_);};
   void setOwnerProc(int proc) {ownerProc_ = proc;};

   void addBlock(GlobalID blk);
   int getNumBlocks() const {return(numBlocks_);};
   GlobalID* getBlockList() const {return(blockList_);};
   int getBlockIndex(GlobalID blk);

 private:
   void allocFieldLists();
   void allocBlockList();

   GlobalID nodeID_;

   int numNodalDOF_;      //total number of nodal degrees-of-freedom

   int numFields_;        //nodal solution variables are called fields
   int* fieldIDList_;     //list of field identifiers
   int* fieldEqnNumbers_; //list of starting (global) equation numbers.
                          //fields can consist of more than one scalar (and
                          //have more than one associated equation), this
                          //is the first equation number

   int ownerProc_;        //processor that owns the equations for this node

   int numBlocks_;
   GlobalID* blockList_;       //blocks that contain this node
};

#endif

