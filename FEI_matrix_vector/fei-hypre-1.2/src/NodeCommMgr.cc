#include <stdlib.h>
#include <iostream.h>

#ifdef FEI_SER
#include "mpiuni/mpi.h"
#else
#include <mpi.h>
#endif

#include "other/basicTypes.h"

#include <../isis-mv/GlobalIDArray.h>

#include "src/NodeDescriptor.h"
#include "src/NodeCommMgr.h"
#include "src/Utils.h"

//======Constructor=============================================================
NodeCommMgr::NodeCommMgr(int localProc)
 : localProc_(localProc),
   numLocalNodes_(0),
   localNodeIDs(),
   commNodes_(NULL),
   numCommNodes_(0),
   commNodeIDs(),
   extNodeIDs(),
   procsPerNode_(NULL),
   procs_(NULL),
   initCompleteCalled_(false)
{
}

//=====Destructor===============================================================
NodeCommMgr::~NodeCommMgr() {

   for(int i=0; i<numCommNodes_; i++) {
      delete commNodes_[i];
      delete [] procs_[i];
   }

   delete [] commNodes_;
   commNodes_ = NULL;
   delete [] procs_;
   procs_ = NULL;
   delete [] procsPerNode_;
   procsPerNode_ = NULL;

   numCommNodes_ = 0;
   numLocalNodes_ = 0;
}

//==============================================================================
void NodeCommMgr::addLocalNode(const NodeDescriptor& node) {
//
//This function shouldn't be called until after NodeCommMgr::initComplete
//has been called.
//
//NodeCommMgr is being informed that 'node' is present in the local
//active node list, and is owned by the local processor.
//This means that either:
// 1. it's an external node that is locally owned (i.e., it appears in
//    another proc's constraint relation), or
// 2. it is a shared node, or
// 3. this node isn't even a comm node (it's a purely local node), in which
//    case we'll do nothing.
//
   if (!initCompleteCalled_) {
      cerr << "NodeCommMgr::addLocalNode: ERROR, don't use this function until"
           << " after you've called NodeCommMgr::initComplete." << endl;
   }

   GlobalID nodeID = node.getGlobalNodeID();

   int commIndex = getCommNodeIndex(nodeID);

   //if this node isn't a commNode, then simply return.
   if (commIndex < 0) return;

   //
   //ok, we've established that 'node' is a
   //comm node, so let's install a NodeDescriptor for it.
   //
   commNodes_[commIndex] = new NodeDescriptor(node);

   //let's also put nodeID in the localNodeIDs list if it isn't already there.

   int index = -1, insertPoint = -1;
   if (localNodeIDs.size() == 0) localNodeIDs.append(nodeID);
   else {
      index = Utils::sortedGlobalIDListFind(nodeID, &(localNodeIDs[0]),
                                     localNodeIDs.size(), &insertPoint);

      if (index < 0) {
         localNodeIDs.insert(insertPoint, node.getGlobalNodeID());
      }
   }
   numLocalNodes_ = localNodeIDs.size();

   //there's a possibility that the local proc-number was not one of those
   //passed in when this node was identified as a commNode. (The FEI's
   //initSharedNodeSet function doesn't mandate that the local processor be
   //included in the list of sharing processors. So let's insert
   //localProc_ in procs_[commIndex] if it isn't already there.

   index = Utils::sortedIntListInsert(localProc_, procs_[commIndex],
                                            procsPerNode_[commIndex]);
}

//==============================================================================
void NodeCommMgr::exchangeEqnInfo(MPI_Comm comm) {
//
//This function will perform the communication necessary to:
//
//   1. For each locally owned node, send to each remote proc that
//      is associated with that node: the node's fieldID(s), those
//      field-size(s), and the first global equation numbers for those
//      fields.
//   2. For each remotely owned node, receive the above information from
//      the owners of these nodes.
//
//This is a collective function. All procs must enter it before any can
//exit it.
//

   //first, set things up so we can send and receive GlobalID stuff

   MPI_Datatype MPI_GLOBAL_ID;
   int ierror = MPI_Type_contiguous(sizeof(GlobalID)/sizeof(int),MPI_INT,
                                &MPI_GLOBAL_ID);
   if (ierror) cout << "MPI_Type_contiguous ierror:" << ierror << endl;
   MPI_Type_commit(&MPI_GLOBAL_ID);

   int i;
   int numProcs;
   MPI_Comm_size(comm, &numProcs);

   int* localNodesPerProc = new int[numProcs];
   int* remoteNodesPerProc = new int[numProcs];
   int* maxNumFields = new int[numProcs];

   for(i=0; i<numProcs; i++) {
      localNodesPerProc[i] = 0;
      remoteNodesPerProc[i] = 0;
      maxNumFields[i] = 0;
   }

   //first, figure out how many locally-owned nodes each remote processor is
   //associated with, and how many remotely-owned nodes we'll be recv'ing info
   //about (from those nodes' owners). Also record the maximum number of fields
   //associated with any local node.
   for(i=0; i<commNodeIDs.size(); i++) {
      if (commNodes_[i] == NULL) {
         //this is a remotely-owned node, so we'll be recving info from its
         //owner, which is the lowest numbered processor in the procs_[i] list,
         //or the second lowest if localProc_ (us) is the lowest.
         //NOTE!!!!!! This could bite us if an external node is also a
         //shared node. i.e., the owner-proc may not be the lowest not-us proc.
         //FIX THIS.

         int proc;

         if (procs_[i][0] != localProc_) proc = procs_[i][0];
         else proc = procs_[i][1];

         remoteNodesPerProc[proc]++;
      }
      else {
         //this is a locally-owned node

         int numFields = commNodes_[i]->getNumFields();

         if (maxNumFields[localProc_] < numFields) {
            maxNumFields[localProc_] = numFields;
         }

         for(int j=0; j<procsPerNode_[i]; j++) {
            int proc = procs_[i][j];

            if (proc != localProc_) localNodesPerProc[proc]++;
         }
      }
   }

   //let's create condensed lists of procs from which we'll be recving, and
   //procs to which we'll be sending.
   int numSendProcs = 0;
   int* sendProcs = NULL;
   createProcList(localNodesPerProc, numProcs, sendProcs, numSendProcs);

   int numRecvProcs = 0;
   int* recvProcs = NULL;
   createProcList(remoteNodesPerProc, numProcs, recvProcs, numRecvProcs);

   //now each proc will find out a max. number of fields to expect per node.
   int* recvMaxFields = new int[numProcs];

   MPI_Allreduce(maxNumFields, recvMaxFields, numProcs,
                 MPI_INT, MPI_SUM, comm);

   int maxFields = 0;
   for(i=0; i<numProcs; i++) {
      if (maxFields < recvMaxFields[i]) maxFields = recvMaxFields[i];
   }

   delete [] recvMaxFields;
   recvMaxFields = NULL;

   //now, we can allocate lists to recv into and launch the irecv's.
   //from each processor, we'll recv a list of length:
   //            num-nodes*(2+ maxFields*2)

   int len = 2 + maxFields*2;

   GlobalID** recvNodeIDs = NULL;
   int** recvData = NULL;
   MPI_Request* recvNodeReqs = NULL;
   MPI_Request* recvDataReqs = NULL;

   if (numRecvProcs > 0) {
      recvNodeIDs = new GlobalID*[numRecvProcs];
      recvData = new int*[numRecvProcs];
      recvNodeReqs = new MPI_Request[numRecvProcs];
      recvDataReqs = new MPI_Request[numRecvProcs];
   }

   int nodeTag = 199901;
   int dataTag = 199902;

   for(i=0; i<numRecvProcs; i++) {
      int numNodes = remoteNodesPerProc[recvProcs[i]];

      recvNodeIDs[i] = new GlobalID[numNodes];
      recvData[i] = new int[numNodes*len];

      MPI_Irecv(recvNodeIDs[i], numNodes, MPI_GLOBAL_ID,
                recvProcs[i], nodeTag, comm, &recvNodeReqs[i]);
      MPI_Irecv(recvData[i], numNodes*len, MPI_INT,
                recvProcs[i], dataTag, comm, &recvDataReqs[i]);
   }

   //next, send all outgoing messages.

   MPI_Barrier(comm);

   for(i=0; i<numSendProcs; i++) {
      int numNodes = localNodesPerProc[sendProcs[i]];

      GlobalID* sendNodeIDs = new GlobalID[numNodes];
      int* sendData = new int[numNodes*len];

      packSendNodesAndData(sendNodeIDs, sendData, sendProcs[i],
                           numNodes, numNodes*len);

      MPI_Send(sendNodeIDs, numNodes, MPI_GLOBAL_ID, sendProcs[i],
               nodeTag, comm);
      MPI_Send(sendData, numNodes*len, MPI_INT, sendProcs[i],
               dataTag, comm);

      delete [] sendNodeIDs;
      delete [] sendData;
   }

   //now, lets allocate the rest of the NodeDescriptors (the ones that aren't
   //local, into which we'll be putting the information we receive).
   for(i=0; i<commNodeIDs.size(); i++) {
      if (commNodes_[i] == NULL) {
         commNodes_[i] = new NodeDescriptor();
         commNodes_[i]->setGlobalNodeID(commNodeIDs[i]);
      }
   }

   //finally, complete the irecvs and put away the node field info.
   for(i=0; i<numRecvProcs; i++) {
      MPI_Status status;
      int index;

      MPI_Waitany(numRecvProcs, recvNodeReqs, &index, &status);
      MPI_Wait(&recvDataReqs[index], &status);

      int offset = 0;
      int numNodes = remoteNodesPerProc[recvProcs[index]];

      for(int j=0; j<numNodes; j++) {
         int nIndex = getCommNodeIndex(recvNodeIDs[index][j]);
         commNodes_[nIndex]->setOwnerProc(recvProcs[index]);

         int numFields = recvData[index][offset++];
         commNodes_[nIndex]->setNumNodalDOF(recvData[index][offset++]);

         for(int fld=0; fld<numFields; fld++) {
            int fieldID = recvData[index][offset++];
            int eqnNum = recvData[index][offset++];

            commNodes_[nIndex]->addField(fieldID);
            commNodes_[nIndex]->setFieldEqnNumber(fieldID, eqnNum);
         }
      }
   }

   delete [] localNodesPerProc;
   delete [] maxNumFields;
   delete [] remoteNodesPerProc;
   delete [] recvProcs;
   delete [] sendProcs;

   for(i=0; i<numRecvProcs; i++) {
      delete [] recvNodeIDs[i];
      delete [] recvData[i];
   }

   delete [] recvNodeIDs;
   delete [] recvData;
   delete [] recvNodeReqs;
   delete [] recvDataReqs;
}

//==============================================================================
void NodeCommMgr::packSendNodesAndData(GlobalID* sendNodeIDs, int* sendData,
                                       int proc, int numNodes, int len) {
//
//This function packs up the nodeIDs, as well as the list containing, for
//each node, the following:
//   numFields
//     'numFields' pairs of (fieldID,eqnNumber)
//
//Incoming parameter len is numNodes * (2 + maxFields*2), where maxFields is
//the maximum number of fields associated with any node.
//sendNodeIDs is of length numNodes, and
//sendData is of length numNodes*len.
//
   int nodeCounter = 0;
   int offset = 0;

   for(int i=0; i<commNodeIDs.size(); i++) {
      if (commNodes_[i] != NULL) {

         //is this local node associated with processor 'proc'?

         int tmp = -1;
         int index = Utils::sortedIntListFind(proc, procs_[i],
                                              procsPerNode_[i], &tmp);

         //if so...
         if (index >= 0) {

            if (nodeCounter >= numNodes) {
               cerr << "NodeCommMgr::packSendNodesAndData: ERROR,"
                    << " nodeCounter >= numNodes." << endl;
            }

            sendNodeIDs[nodeCounter++] = commNodes_[i]->getGlobalNodeID();

            int numFields = commNodes_[i]->getNumFields();
            int* fieldIDsPtr = commNodes_[i]->getFieldIDList();
            int* fieldEqnNums = commNodes_[i]->getFieldEqnNumbers();

            sendData[offset++] = numFields;
            sendData[offset++] = commNodes_[i]->getNumNodalDOF();

            for(int j=0; j<numFields; j++) {
               sendData[offset++] = fieldIDsPtr[j];

               if (offset >= len) {
                  cerr << "NodeCommMgr::packSendNodesAndData: ERROR,"
                       << " offset >= len." << endl;
               }

               sendData[offset++] = fieldEqnNums[j];
            }
         }
      }
   }
}

//==============================================================================
void NodeCommMgr::createProcList(int* itemsPerProc, int numProcs,
                                 int*& commProcs, int& numCommProcs) {
//
//This function looks through the itemsPerProc list and counts how many
//positions in this list are greater than 0. Then it creates a list of
//the indices of those positions. i.e., itemsPerProc is a list of how many
//items are to be sent to or recvd from each proc. When itemsPerProc is
//greater than 0, that proc is put in the commProcs list.
//
   int i;
   numCommProcs = 0;

   for(i=0; i<numProcs; i++) {
      if (itemsPerProc[i] > 0) numCommProcs++;
   }

   commProcs = new int[numCommProcs];

   int offset = 0;

   for(i=0; i<numProcs; i++) {
      if (itemsPerProc[i] > 0) commProcs[offset++] = i;
   }
}

//==============================================================================
int NodeCommMgr::getCommNodeIndex(GlobalID nodeID) {

   int insertPoint = -1;
   return( Utils::sortedGlobalIDListFind(nodeID,
                                 &(commNodeIDs[0]), numCommNodes_,
                                 &insertPoint)
         );
}

//==============================================================================
bool NodeCommMgr::isExternal(GlobalID nodeID) {
   if (extNodeIDs.size() == 0) return(false);

   int insertPoint = -1;
   int index = Utils::sortedGlobalIDListFind(nodeID, &(extNodeIDs[0]),
                                             extNodeIDs.size(), &insertPoint);
   if (index >= 0) return(true);
   else return(false); 
}

//==============================================================================
NodeDescriptor& NodeCommMgr::getCommNode(int index) {

   if ((index < 0) || (index >= numCommNodes_) || (!initCompleteCalled_)) {
      cerr << "NodeCommMgr::getCommNode: ERROR, index out of range, or "
           << "NodeCommMgr::initComplete hasn't been called yet." << endl;
      abort();
   }

   return(*(commNodes_[index]));
}

//==============================================================================
void NodeCommMgr::addCommNodes(bool externNodes,
                               const GlobalID* nodeIDs, int numNodes, 
                               const int* const* procs, const int* numProcs) {

   if (numNodes <= 0) return;

   int start;

   if (externNodes) {
      if (extNodeIDs.size() == 0) {
         extNodeIDs.append(nodeIDs[0]);
         start = 1;
      }
      else start = 0;

      for(int i=start; i<numNodes; i++) {
         int insert;
         int index = Utils::sortedGlobalIDListFind(nodeIDs[i], &(extNodeIDs[0]),
                                           extNodeIDs.size(), &insert);

         if (index < 0) {
            extNodeIDs.insert(insert, nodeIDs[i]);
         }
      }
   }

   if (numCommNodes_ == 0) {
      commNodeIDs.append(nodeIDs[0]);

      storeCommNodeProcs(0, procs[0], numProcs[0]);

      start = 1;
   }
   else start = 0;

   for(int i=start; i<numNodes; i++) {
      int insert;
      int index = Utils::sortedGlobalIDListFind(nodeIDs[i], &(commNodeIDs[0]),
                                           commNodeIDs.size(), &insert);

      if (index < 0) {
         commNodeIDs.insert(insert, nodeIDs[i]);
         storeCommNodeProcs(insert, procs[i], numProcs[i]);
      }
      else {
         storeCommNodeProcs(index, procs[i], numProcs[i]);
      }
   }
}

//==============================================================================
void NodeCommMgr::initComplete() {
//
//This function is called when initialization is complete (i.e., when
//all commNodes have been added.
//
//We can now allocate the list of NodeDescriptor pointers into which the
//commNode info will be put. We'll be putting in fully populated 
//NodeDescriptors for local nodes, and receiving the field/eqn info for
//the nodes that aren't local.
//
   commNodes_ = new NodeDescriptor*[numCommNodes_];

   for(int i=0; i<numCommNodes_; i++) {
      commNodes_[i] = NULL;
   }

   initCompleteCalled_ = true;
}

//==============================================================================
int NodeCommMgr::getLowestProcNumber(GlobalID nodeID) {
//
//return the lowest processor number that's associated with node 'nodeID'.
//If nodeID isn't even a comm node, return -1.
//
   int index = getCommNodeIndex(nodeID);

   if (index < 0) return(-1);
   else return(procs_[index][0]);
}

//==============================================================================
void NodeCommMgr::storeCommNodeProcs(int index,
                                     const int* procs, int numProcs) {
//
//Private NodeCommMgr function.
//
//This function stores the procs associated with the commNodeID at position
//index. If numCommNodes_ != commNodeIDs.size(), then we know that a new node
//has been inserted, so we must add a new row to the procs_ table.
//Otherwise, we insert each of these incoming proc-numbers into the 'index'th
//row of the procs_ table, only if not already present.
//
   int tmp, start;

   if (numCommNodes_ != commNodeIDs.size()) {
      //a new node has been inserted, so this proc list will be a new row
      //in the procs_ table.

      //start the new row and insert it in the procs_ table...
      int* newRow = new int[1];
      newRow[0] = procs[0];

      tmp = numCommNodes_;

      Utils::intTableInsertRow(newRow, index, procs_, numCommNodes_);

      //now set the procsPerNode_ length counter.
      Utils::intListInsert(1, index, procsPerNode_, tmp);

      start = 1;
   }
   else start = 0;

   //now add the procs to procs_[index].
   for(int i=start; i<numProcs; i++) {
      tmp = Utils::sortedIntListInsert(procs[i], procs_[index],
                                     procsPerNode_[index]);
   }
}

