#ifndef _ProblemStructure_h_
#define _ProblemStructure_h_

//==============================================================================
//
//The ProblemStructure class is a container for the data that makes up
//the structure of a finite element problem.
//It contains:
//  the list of FieldRecords,
//  the list of BlockDescriptors,
//  the lists of NodeDescriptors
//    (for active nodes as well as nodes that are external and thus require
//    inter-process communication),
//  and the element-connectivity tables (one for each block).
//
//Note that the list of active nodes includes shared nodes, which also
//require inter-process communication.
//
//==============================================================================

//requires:
//#include "other/basicTypes.h"
//#include "src/BlockDescriptor.h"
//#include "src/NodeDescriptor.h"
//#include <mv/IntArray.h>

//Declare a class to hold a connectivity table. This class is so simple
//it's almost a struct.

class ConnectivityTable {
 public:
   ConnectivityTable() : elemIDs(NULL), connectivities(NULL),
                         numElems(0), numNodesPerElem(0) {};

   virtual ~ConnectivityTable() {
      delete [] elemIDs;
      for(int i=0; i<numElems; i++) delete [] connectivities[i];
      delete [] connectivities;
      numElems = 0;
   };

   GlobalID* elemIDs;
   GlobalID** connectivities;
   int numElems;
   int numNodesPerElem;
};

//Now declare the ProblemStructure class.

class ProblemStructure {
 public:
   ProblemStructure(int localProc);
   virtual ~ProblemStructure();

   int getNumFields() {return(numFields_);};
   void setNumFields(int fields);

   FieldRecord* getFieldRosterPtr();
   int getFieldRosterIndex(int fieldID);

   int getNumBlocks() {return(numBlocks_);};
   void setNumBlocks(int blks);

   void addBlockID(GlobalID blockID);

   int getBlockIndex(GlobalID blockID);

   BlockDescriptor* getBlockDescriptorsPtr();

   BlockDescriptor& getBlockDescriptor(GlobalID blockID);

   void allocateConnectivityTable(GlobalID blockID);
   void destroyConnectivityTables();

   ConnectivityTable& getConnectivityTable(GlobalID blockID);

   int getNumActiveNodes() {return(numActiveNodes_);};

   void initializeActiveNodes();

   NodeDescriptor* getActiveNodesPtr();

   GlobalIDArray& getActiveNodeIDList();

   int getActiveNodeIndex(GlobalID nodeID);

   NodeCommMgr& getNodeCommMgr() {return(nodeCommMgr_);};

   EqnCommMgr& getEqnCommMgr() {return(eqnCommMgr_);};

   void initializeEqnCommMgr();

   void getEqnInfo(int& numGlobalEqns, int& numLocalEqns,
                   int& localStartRow, int& localEndRow);

   void setEqnInfo(int numGlobalEqns, int numLocalEqns,
                   int localStartRow, int localEndRow);

   IntArray* getSystemMatrixIndices() {return(sysMatIndices_);};

   void getScatterIndices_ID(GlobalID blockID, GlobalID elemID,
                             int* scatterIndices, int* remoteEqnOffsets,
                             int* remoteProcs, int& numRemoteEqns);

   void getScatterIndices_index(int blockIndex, int elemIndex,
                             int* scatterIndices, int* remoteEqnOffsets,
                             int* remoteProcs, int& numRemoteEqns);

 private:
   void destroyBlockRoster();
   void destroyNodeLists();

   int localProc_;

   int numFields_;

   FieldRecord* fieldRoster_;

   int numBlocks_;

   GlobalIDArray GID_blockIDList;

   BlockDescriptor* blockRoster_;
   
   ConnectivityTable* connTables_;

   NodeDescriptor* activeNodes_;
   int numActiveNodes_;

   GlobalIDArray GID_ActNodeList;

   NodeCommMgr nodeCommMgr_;

   EqnCommMgr eqnCommMgr_;

   int numGlobalEqns_;
   int numLocalEqns_;
   int localStartRow_;
   int localEndRow_;

   IntArray* sysMatIndices_;
};

#endif

