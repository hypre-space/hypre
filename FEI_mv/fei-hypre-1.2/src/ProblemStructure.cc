#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "src/BlockDescriptor.h"
#include "src/NodeDescriptor.h"
#include "src/FieldRecord.h"

/* changes to conform to HYPRE */
/*#ifdef FEI_SER */
/*#include "mpiuni/mpi.h" */
/*#else */
/*#include <mpi.h> */
/*#endif */
#include "../../utilities/utilities.h"

#include <../isis-mv/IntArray.h>
#include <../isis-mv/GlobalIDArray.h>
#include "src/NodeCommMgr.h"
#include "src/ProcEqns.h"
#include "src/EqnBuffer.h"
#include "src/EqnCommMgr.h"

#include "src/Utils.h"
#include "src/SLE_utils.h"

#include "src/ProblemStructure.h"

//=====Constructor==============================================================
ProblemStructure::ProblemStructure(int localProc)
 : localProc_(localProc),
   numFields_(0),
   fieldRoster_(NULL),
   numBlocks_(0),
   GID_blockIDList(),
   blockRoster_(NULL),
   connTables_(NULL),
   activeNodes_(NULL),
   GID_ActNodeList(),
   nodeCommMgr_(localProc),
   eqnCommMgr_(localProc),
   numGlobalEqns_(0),
   numLocalEqns_(0),
   localStartRow_(0),
   localEndRow_(0),
   sysMatIndices_(NULL)
{
}

//=====Destructor===============================================================
ProblemStructure::~ProblemStructure() {

   delete [] fieldRoster_;
   fieldRoster_ = NULL;
   numFields_ = 0;

   destroyBlockRoster();
   destroyConnectivityTables();
   destroyNodeLists();

   numBlocks_ = 0;

   delete [] sysMatIndices_;
   sysMatIndices_ = NULL;
}

//==============================================================================
void ProblemStructure::destroyBlockRoster() {

   delete [] blockRoster_;
   blockRoster_ = NULL;
}

//==============================================================================
void ProblemStructure::destroyConnectivityTables() {

   delete [] connTables_;
   connTables_ = NULL;
}

//==============================================================================
void ProblemStructure::destroyNodeLists() {

   delete [] activeNodes_;
   activeNodes_ = NULL;
   numActiveNodes_ = 0;
}

//==============================================================================
void ProblemStructure::setNumFields(int fields) {
   if (fields <= 0) {
      cerr << "ProblemStructure::setNumFields: ERROR, fields <= 0." << endl;
      return;
   }

   if (numFields_ > 0) delete [] fieldRoster_;

   numFields_ = fields;

   fieldRoster_ = new FieldRecord[numFields_];
}

//==============================================================================
FieldRecord* ProblemStructure::getFieldRosterPtr() {
   if (numFields_ == 0) {
      cerr << "ProblemStructure::getFieldRosterPtr: WARNING, numFields_ == 0."
           << endl;
      return(NULL);
   }

   return(fieldRoster_);
}

//==============================================================================
int ProblemStructure::getFieldRosterIndex(int fieldID) {

   for (int i = 0; i<getNumFields(); i++) {
      if (fieldID == fieldRoster_[i].getFieldID()) {
         return(i);
      }
   }
   return(-1);
}

//==============================================================================
void ProblemStructure::setNumBlocks(int blks) {
   if (blks <= 0) {
      cerr << "ProblemStructure::setNumBlocks: ERROR, blks <= 0." << endl;
      return;
   }

   numBlocks_ = blks;

   blockRoster_ = new BlockDescriptor[numBlocks_];

   connTables_ = new ConnectivityTable[numBlocks_];
}

//==============================================================================
void ProblemStructure::addBlockID(GlobalID blockID) {
//
//Append a blockID to our (unsorted) list of blockIDs if it isn't
//already present.
//
   if (GID_blockIDList.size()==0) {
      GID_blockIDList.append(blockID);
   }
   else {
      int found = search_ID_index(blockID, &GID_blockIDList[0],
                                  GID_blockIDList.size());
      if (found<0) {
         GID_blockIDList.append(blockID);
      }
   }
}

//==============================================================================
int ProblemStructure::getBlockIndex(GlobalID blockID) {

   int index = search_ID_index(blockID, &GID_blockIDList[0],
                               GID_blockIDList.size());

   if (index < 0) {
      cerr << "ProblemStructure::getBlockIndex: ERROR, blockID "
           << (int)blockID << " not found. Aborting." << endl;
      abort();
   }

   return(index);
}

//==============================================================================
BlockDescriptor* ProblemStructure::getBlockDescriptorsPtr() {
   if (numBlocks_ == 0) {
      cerr << "ProblemStructure::getBlockDescriptorsPtr: ERROR, numBlocks_ == 0"
           << endl;
      return(NULL);
   }

   return(blockRoster_);
}

//==============================================================================
BlockDescriptor& ProblemStructure::getBlockDescriptor(GlobalID blockID) {

   int index = search_ID_index(blockID, &GID_blockIDList[0],
                               GID_blockIDList.size());

   if (index < 0) {
      cerr << "ProblemStructure::getBlockDescriptor: ERROR, blockID "
           << (int)blockID << " not found. Aborting." << endl;
      abort();
   }

   return(blockRoster_[index]);
}

//==============================================================================
void ProblemStructure::allocateConnectivityTable(GlobalID blockID) {

   int index = search_ID_index(blockID, &GID_blockIDList[0],
                               GID_blockIDList.size());

   if (index < 0) {
      cerr << "ProblemStructure::allocateConnectivityTable: ERROR, blockID "
           << (int)blockID << " not found. Aborting." << endl;
      abort();
   }

   connTables_[index].numElems = blockRoster_[index].getNumElements();
   connTables_[index].numNodesPerElem = 
                         blockRoster_[index].getNumNodesPerElement();

   int numRows = connTables_[index].numElems;
   int numCols = connTables_[index].numNodesPerElem;

   if ((numRows <= 0) || (numCols <= 0)) {
      cerr << "ProblemStructure::allocateConnectivityTable: ERROR, either "
           << "numElems or numNodesPerElem not yet set for blockID "
           << (int)blockID << ". Aborting." << endl;
      abort();
   }

   connTables_[index].elemIDs = new GlobalID[numRows];
   connTables_[index].connectivities = new GlobalID*[numRows];

   for(int i=0; i<numRows; i++) {
      connTables_[index].connectivities[i] = new GlobalID[numCols];
   }
}

//==============================================================================
ConnectivityTable& ProblemStructure::getConnectivityTable(GlobalID blockID) {

   int index = search_ID_index(blockID, &GID_blockIDList[0],
                               GID_blockIDList.size());
                               
   if (index < 0) {
      cerr << "ProblemStructure::getConnectivityTable: ERROR, blockID "
           << (int)blockID << " not found. Aborting." << endl;
      abort();
   }  

   return(connTables_[index]);
}

//==============================================================================
void ProblemStructure::initializeActiveNodes() {
//
//Allocate the list of NodeDescriptors (we have the list of Global nodeIDs
//int GID_ActNodeList).
//
   numActiveNodes_ = GID_ActNodeList.size();

   activeNodes_ = new NodeDescriptor[numActiveNodes_];

   int i;
   for(i=0; i<numActiveNodes_; i++) {
      activeNodes_[i].setGlobalNodeID(GID_ActNodeList[i]);
   }

//
//Now run through the connectivity tables, and set the nodal field and block
//info.
//

   for(int bIndex=0; bIndex<numBlocks_; bIndex++) {

      GlobalID blockID = blockRoster_[bIndex].getGlobalBlockID();
      ConnectivityTable& conn = connTables_[bIndex];

      GlobalID** elemConn = conn.connectivities;

      int numElems = conn.numElems;
      int nodesPerElem = conn.numNodesPerElem;

      int* fieldsPerNodePtr = blockRoster_[bIndex].fieldsPerNodePtr();
      int** fieldIDsTablePtr = blockRoster_[bIndex].fieldIDsTablePtr();

      //
      //Now we want to go through the connectivity table, and for each node,
      //add its fields and this block to the correct NodeDescriptor in the
      //actNodes list.
      //(Note that the addField and addBlock functions only add the input if
      //it isn't already present in that NodeDescriptor.)
      //

      for(int elem=0; elem<numElems; elem++) {
         for(int node=0; node<nodesPerElem; node++) {
            int index = getActiveNodeIndex(elemConn[elem][node]);

            for(i=0; i<fieldsPerNodePtr[node]; i++) {
               activeNodes_[index].addField(fieldIDsTablePtr[node][i]);
            }

            activeNodes_[index].addBlock(blockID);
         }
      }
   }

   //while we're here, let's count how many active nodes there are for
   //each block.

   int* nodesPerBlock = new int[numBlocks_];
   int j;
   for(j=0; j<numBlocks_; j++) nodesPerBlock[j] = 0;

   for(j=0; j<numActiveNodes_; j++) {
      int numBlks = activeNodes_[j].getNumBlocks();
      GlobalID* blkList = activeNodes_[j].getBlockList();

      for(int jj=0; jj<numBlks; jj++) {
         nodesPerBlock[getBlockIndex(blkList[jj])]++;
      }
   }

   for(j=0; j<numBlocks_; j++)
      blockRoster_[j].setNumActiveNodes(nodesPerBlock[j]);

   delete [] nodesPerBlock;
}

//==============================================================================
NodeDescriptor* ProblemStructure::getActiveNodesPtr() {
   if (numActiveNodes_ == 0) {
      cerr << "ProblemStructure::getActiveNodesPtr: WARNING, numActiveNodes "
           << "== 0." << endl;
      return(NULL);
   }

   return(activeNodes_);
}

//==============================================================================
GlobalIDArray& ProblemStructure::getActiveNodeIDList() {
   return(GID_ActNodeList);
}

//==============================================================================
int ProblemStructure::getActiveNodeIndex(GlobalID nodeID) {

   int insertPoint = -1;

   return(Utils::sortedGlobalIDListFind(nodeID, &(GID_ActNodeList[0]),
                                 GID_ActNodeList.size(), &insertPoint));
}

//==============================================================================
void ProblemStructure::initializeEqnCommMgr() {
//
//This function will take information from the nodeCommMgr and use it to tell
//the eqnCommMgr which equations we can expect other processors to contribute
//to.
//
//This function can not be called until after all local comm nodes have been
//identified to the nodeCommMgr.
//
   int numLocalCommNodes = nodeCommMgr_.getNumLocalNodes();
   GlobalID* localNodeIDs = nodeCommMgr_.getLocalNodeIDs();

   for(int i=0; i<numLocalCommNodes; i++) {
      int index = nodeCommMgr_.getCommNodeIndex(localNodeIDs[i]);
      NodeDescriptor& node = nodeCommMgr_.getCommNode(index);

      int numFields = node.getNumFields();
      int* fieldIDsPtr = node.getFieldIDList();
      int* eqnNumbers = node.getFieldEqnNumbers();

      int numProcs = nodeCommMgr_.getNumProcsPerCommNode(index);
      int* proc = nodeCommMgr_.getCommNodeProcs(index);

      for(int p=0; p<numProcs; p++) {

         if (proc[p] == localProc_) continue;

         for(int j=0; j<numFields; j++) {
            int fIndex = getFieldRosterIndex(fieldIDsPtr[j]);
            int numEqns = fieldRoster_[fIndex].getNumFieldParams();

            int eqn;
            for(eqn=0; eqn<numEqns; eqn++) {
               eqnCommMgr_.addRecvEqn(eqnNumbers[j]+eqn, proc[p]);
            }
         }
      }
   }
}

//==============================================================================
void ProblemStructure::getEqnInfo(int& numGlobalEqns, int& numLocalEqns,
                   int& localStartRow, int& localEndRow){

   numGlobalEqns = numGlobalEqns_;
   numLocalEqns = numLocalEqns_;
   localStartRow = localStartRow_;
   localEndRow = localEndRow_;
}

//==============================================================================
void ProblemStructure::setEqnInfo(int numGlobalEqns, int numLocalEqns,
                   int localStartRow, int localEndRow){

   numGlobalEqns_ = numGlobalEqns;
   numLocalEqns_ = numLocalEqns;
   localStartRow_ = localStartRow;
   localEndRow_ = localEndRow;

   sysMatIndices_ = new IntArray[numLocalEqns_];
}

//==============================================================================
void ProblemStructure::getScatterIndices_ID(GlobalID blockID, GlobalID elemID, 
                                            int* scatterIndices,
                                            int* remoteEqnOffsets,
                                            int* remoteProcs,
                                            int& numRemoteEqns) {

   int index = search_ID_index(blockID, &GID_blockIDList[0],
                               GID_blockIDList.size());

   if (index < 0) {
      cerr << "ProblemStructure::getScatterIndices_ID: ERROR, blockID "
           << (int)blockID << " not found. Aborting." << endl;
      abort();
   }

   int elemIndex = search_ID_index(elemID, connTables_[index].elemIDs,
                                   connTables_[index].numElems);

   if (elemIndex < 0) {
      cerr << "ProblemStructure::getScatterIndices_ID: ERROR, elemID "
           << (int)elemID << " not found. Aborting." << endl;
      abort();
   }

   getScatterIndices_index(index, elemIndex, scatterIndices,
                           remoteEqnOffsets, remoteProcs, numRemoteEqns);
}

//==============================================================================
void ProblemStructure::getScatterIndices_index(int blockIndex, int elemIndex,
                                               int* scatterIndices,
                                               int* remoteEqnOffsets,
                                               int* remoteProcs,
                                               int& numRemoteEqns) {
//
//On input, the arrays scatterIndices, remoteEqnOffsets and remoteProcs are
//assumed to all be allocated by the calling code, and be of length the
//number of equations per element.
//
   int numNodes = connTables_[blockIndex].numNodesPerElem;
   GlobalID* nodeIDs = connTables_[blockIndex].connectivities[elemIndex];

   int offset = 0;
   numRemoteEqns = 0;
   for(int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
      int index = getActiveNodeIndex(nodeIDs[nodeIndex]);

      int proc = activeNodes_[index].getOwnerProc();

      if (proc == localProc_) {
         NodeDescriptor& node = activeNodes_[index];
         int numFields = node.getNumFields();
         int* fieldIDsPtr = node.getFieldIDList();
         int* eqnNumbers = node.getFieldEqnNumbers();

         for(int j=0; j<numFields; j++) {
            int fInd = getFieldRosterIndex(fieldIDsPtr[j]);
            int numEqns = fieldRoster_[fInd].getNumFieldParams();

            for(int jj=0; jj<numEqns; jj++) {
               scatterIndices[offset++] = eqnNumbers[j]+jj;
            }
         }
      }
      else {
         //this node is a remotely-owned comm node, so we need to get its
         //NodeDescriptor from the node comm manager.

         index = nodeCommMgr_.getCommNodeIndex(nodeIDs[nodeIndex]);

         if (index < 0) {
            cerr << "ProblemStructure::getScatterIndices_index: ERROR, nodeID "
                 << (int)nodeIDs[nodeIndex] << " not found." << endl;
            abort();
         }

         NodeDescriptor& node = nodeCommMgr_.getCommNode(index);
         int numFields = node.getNumFields();
         int* fieldIDsPtr = node.getFieldIDList();
         int* eqnNumbers = node.getFieldEqnNumbers();

         for(int j=0; j<numFields; j++) {
            int fInd = getFieldRosterIndex(fieldIDsPtr[j]);
            int numEqns = fieldRoster_[fInd].getNumFieldParams();

            for(int jj=0; jj<numEqns; jj++) {
               scatterIndices[offset++] = eqnNumbers[j]+jj;
               remoteEqnOffsets[numRemoteEqns] = offset-1;
               remoteProcs[numRemoteEqns++] = proc;
            }
         }
      }
   }

   //now the element-DOF.
   int numElemDOF = blockRoster_[blockIndex].getNumElemDOFPerElement();
   int* elemDOFEqnsPtr = blockRoster_[blockIndex].elemDOFEqnNumbersPtr();

   for(int i=0; i<numElemDOF; i++) {
      scatterIndices[offset++] = elemDOFEqnsPtr[elemIndex] + i;
   }
}

