#ifndef _NodeCommMgr_h_
#define _NodeCommMgr_h_

//
//The NodeCommMgr (Node communication manager) class is responsible for
//keeping track of all nodes that require communication. This includes
//both shared and external nodes.
//
//All shared and external nodes are put in to this class through the
//addCommNodes function. When they've all been put in, initComplete is
//called, which runs through the list of those nodes and separates out
//the ones that are local (owned by this processor). After that, the
//NodeDescriptors for the local nodes can be put in using the
//addLocalNode function. At this stage, those local NodeDescriptors
//should be already initialized, containing the fields associated with
//the node, as well as the global equation numbers associated with
//those fields.
//
//The communication that must then happen is this:
//
//  For all nodes that we own, we must send their field-IDs and
//  (global) equation numbers to the remote processors that either
//  share the node or know about it as an external node.
//
//  For all nodes that we don't own, we need to receive the field-IDs
//  and equation numbers.
//

//required:
//
//#include <mpi.h>
//#include "other/basicTypes.h"
//#include "src/NodeDescriptor.h"

class NodeCommMgr {
 public:
   NodeCommMgr(int localProc);
   virtual ~NodeCommMgr();

   int getNumLocalNodes() {return(numLocalNodes_);};
   GlobalID* getLocalNodeIDs() {return(&(localNodeIDs[0]));};

   void addCommNodes(bool externNodes, const GlobalID* nodeIDs, int numNodes,
                     const int* const* procs, const int* numProcs);

   void initComplete();

   int getLowestProcNumber(GlobalID nodeID);

   void addLocalNode(const NodeDescriptor& node);

   void exchangeEqnInfo(MPI_Comm comm);

   int getCommNodeIndex(GlobalID nodeID);

   bool isExternal(GlobalID nodeID);

   NodeDescriptor& getCommNode(int index);

   int getNumProcsPerCommNode(int index) {return(procsPerNode_[index]);};

   int* getCommNodeProcs(int index) {return(procs_[index]);};

 private:
   void storeCommNodeProcs(int index, const int* procs, int numProcs);

   void packSendNodesAndData(GlobalID* sendNodeIDs, int* sendData, int proc,
                             int numNodes, int len);
   void createProcList(int* itemsPerProc, int numProcs, int*& commProcs,
                       int& numCommProcs);

   int numLocalNodes_;
   GlobalIDArray localNodeIDs;

   NodeDescriptor** commNodes_;

   int numCommNodes_;         //number of nodes requiring communication
   GlobalIDArray commNodeIDs; //global node identifiers

   GlobalIDArray extNodeIDs; //nodes that are external nodes

   int* procsPerNode_;    //how many processors are associated with each node
   int** procs_;          //which procs are associated with each node

   int localProc_;

   bool initCompleteCalled_;
};

#endif

