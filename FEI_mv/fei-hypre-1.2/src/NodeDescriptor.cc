#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "src/Utils.h"

#include "src/NodeDescriptor.h"

//======Constructor=============================================================
NodeDescriptor::NodeDescriptor()
 : nodeID_((GlobalID)-1),
   numNodalDOF_(0),
   numFields_(0),
   fieldIDList_(NULL),
   fieldEqnNumbers_(NULL),
   ownerProc_(-1),
   numBlocks_(0),
   blockList_(NULL)
{
   //There's nothing else for this constructor to do, apart from the
   //above initializations.
}

//=====Copy-Constructor=========================================================
NodeDescriptor::NodeDescriptor(const NodeDescriptor& source)
 : nodeID_(source.nodeID_),
   numNodalDOF_(source.numNodalDOF_),
   numFields_(source.numFields_),
   fieldIDList_(NULL),
   fieldEqnNumbers_(NULL),
   ownerProc_(source.ownerProc_),
   numBlocks_(source.numBlocks_),
   blockList_(NULL)
{
   if (numFields_ > 0) {
      fieldIDList_ = new int[numFields_];
      fieldEqnNumbers_ = new int[numFields_];

      for(int i=0; i<numFields_; i++) {
         fieldIDList_[i] = source.fieldIDList_[i];
         fieldEqnNumbers_[i] = source.fieldEqnNumbers_[i];
      }
   }

   if (numBlocks_ > 0) {
      blockList_ = new GlobalID[numBlocks_];

      for(int j=0; j<numBlocks_; j++) {
         blockList_[j] = source.blockList_[j];
      }
   }
}

//======Destructor==============================================================
NodeDescriptor::~NodeDescriptor() {

   delete [] fieldIDList_;
   fieldIDList_ = NULL;
   delete [] fieldEqnNumbers_;
   fieldEqnNumbers_ = NULL;
   numFields_ = 0;
   delete [] blockList_;
   blockList_ = NULL;
   numBlocks_ = 0;
}

//==============================================================================
void NodeDescriptor::addField(int fieldID) {
//
//Add a field identifier to this node, ONLY if that field identifier
//is not already present.
//
//If fieldID is added, lengthen the corresponding list for equation numbers.
//
   int tmpLen = numFields_;

   int index = Utils::sortedIntListInsert(fieldID, fieldIDList_, numFields_);

   if (numFields_ > tmpLen) {
      //
      //if the length of fieldIDList_ changed, we need to lengthen the 
      //fieldEqnNumbers_ list.
      //fieldEqnNumbers_ will have an empty position 'index', which we'll set to
      //-99 for now. The calling code (BASE_FEI) will set the fieldEqnNumber for
      //this fieldID using setFieldEqnNumber(...).
      //

      int* newList = new int[numFields_];

      for(int i=0; i<index; i++) {
         newList[i] = fieldEqnNumbers_[i];
      }

      newList[index] = -99;

      for(int j=index; j<tmpLen; j++) {
         newList[j+1] = fieldEqnNumbers_[j];
      }

      delete [] fieldEqnNumbers_;
      fieldEqnNumbers_ = newList;
   }
}

//==============================================================================
void NodeDescriptor::setFieldEqnNumber(int fieldID, int eqn) {
//
//Set the equation number corresponding to fieldID. fieldID must
//already have been added to this node using the addField function.
//If it was already added, then the fieldEqnNumbers_ list was lengthened
//appropriately, with an empty spot left for this eqn number.
//
   int tmp;
   int index = Utils::sortedIntListFind(fieldID, fieldIDList_, numFields_,
                                        &tmp);

   if (index < 0) {
      cerr << "NodeDescriptor::setFieldEqnNumber: ERROR, fieldID " << fieldID
           << " not found." << endl;
      return;
   }

   fieldEqnNumbers_[index] = eqn;
}

//==============================================================================
int NodeDescriptor::getFieldEqnNumber(int fieldID) {

   int tmp;
   int index = Utils::sortedIntListFind(fieldID, fieldIDList_, numFields_,
                                        &tmp);

   if (index < 0) {
      cerr << "NodeDescriptor::getFieldEqnNumber: ERROR, fieldID " << fieldID
           << " not found." << endl;
      abort();
   }

   return(fieldEqnNumbers_[index]);
}

//==============================================================================
int NodeDescriptor::getFieldIndex(int fieldID) {
   int tmp;
   int index = Utils::sortedIntListFind(fieldID, fieldIDList_, numFields_,
                                        &tmp);

   return(index);
}

//==============================================================================
void NodeDescriptor::addBlock(GlobalID blk) {
//
//Insert 'blk' in this node's list of blocks, if it isn't already there.
//
   int index = Utils::sortedGlobalIDListInsert(blk, blockList_, numBlocks_);

   //we don't care what the returned index value (the position at which
   //it was inserted, if it was inserted) is -- do we?
   (void)index;
}

//==============================================================================
int NodeDescriptor::getBlockIndex(GlobalID blk) {
   int tmp;
   int index = Utils::sortedIntListFind(blk, blockList_, numBlocks_, &tmp);

   return(index);
}

