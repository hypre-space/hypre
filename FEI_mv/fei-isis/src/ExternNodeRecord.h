#ifndef __ExternNodeRecord_H
#define __ExternNodeRecord_H

//  
//  ExternNodeRecord is a container class which holds information about
//  external nodes.
//  
//  Two kinds of external nodes come in through the FEI functions: local nodes
//  and remote nodes. Local nodes are nodes which this processor owns but which
//  other processors need information about; i.e., nodes which appear in 
//  remote constraints. Remote nodes are nodes which this processor does not own,
//  but needs information about; i.e., nodes which appear in local constraints. 
//  Both kinds of nodes are dealt with by this class.
//  
//  The ExtNode struct defined below is a compact container for holding
//  information about nodes which appear in a constraint record on one
//  processor but which are "owned" by another processor. There will be an
//  array of ExtNode structs, corresponding to the external nodes that we
//  are informed about through the FEI functions.
//  
//  If a node appears in more than one constraint relation, then that is
//  recorded in the "numMultCRs" or "numPenCRs" variable in ExtNode. These
//  variables track the number of Lagrange and Penalty constraints,
//  respectively. If new constraint types are added (that require
//  communications) then appropriate new fields should be created to track
//  these types.
//  

struct ExtNode {

    GlobalID nodeID;     // This node's global ID number
    int ownerProc;       // This node's owner
    int* procIDs;        // Processors that information about this node will
                         // be sent to and/or received from
    int numProcs;        // length of the procIDs list
    int numSolnParams;   // Number of solution parameters at this node
    int globalEqn;       // Global equation number of this node.
    int numMultCRs;      // Number of Lagrange constraints that this node
                         // appears in
    int numPenCRs;       // Number of penalty constraints that this node
                         // appears in

    int* globalCREqn;    // Global equation numbers of the CRs that this
                         // node is in.	this list of global equation numbers
                         // is allocated only when this node is associated
                         // with Lagrange constraints, and has length
                         // numMultCRs in that case


    bool penaltyTerm;    // indicates whether this node appears in a penalty
                         // constraint. penalty term nodes require 
                         // different processing.

    int numFields;       // number of fieldIDs for this node
    int* fieldIDs;       // list of fieldIDs for this node
    int* fieldOffsets;   // list of field offsets for this node

};
 
class ExternNodeRecord {
  public:
    ExternNodeRecord();
    virtual ~ExternNodeRecord();

    //function for initializing the external nodes

    void externNodes(GlobalID *nodeIDs, int numNodes,
                     int **procs, int *lenProcs);

    //the initComplete function will be called to build the lists of
    //processors to be communicated with, etc., after the nodes and their
    //appropriate information has been loaded in.
    void initComplete();

    int* sendProcListPtr(int& lenSendProcs);
    int* recvProcListPtr(int& lenRecvProcs);

    GlobalID* externNodeList(int& lenExtNodeList);

    GlobalID** localNodeIDsPtr(int** lenLocNodeIDs);
    GlobalID** remoteNodeIDsPtr(int** lenRemoteNodeIDs);

    //functions for manipulating the remote node information.
    //see the .cc file for descriptions of these functions.

    bool isExternNode(GlobalID nodeID);
    bool isRemoteNode(GlobalID nodeID);

    int getNumMultCRs(GlobalID nodeID);
    void setNumMultCRs(GlobalID nodeID, int numMultCRs);

    int getNumPenCRs(GlobalID nodeID);
    void setNumPenCRs(GlobalID nodeID, int numPenCRs);

    int ownerProc(GlobalID nodeID);
    void ownerProc(GlobalID nodeID, int owner);

    int* procIDs(GlobalID nodeID, int& numprocs);
    void procIDs(ExtNode *node, int* procs, int numprocs);
    
    int numSolnParams(GlobalID nodeID);
    void numSolnParams(GlobalID nodeID, int dof);

    int globalEqn(GlobalID nodeID);
    void globalEqn(GlobalID nodeID, int eqnNum);

    int* globalCREqn(GlobalID nodeID, int& numMultCRs);
    void globalCREqn(GlobalID nodeID, int* eqnNum, int numMultCRs);

    bool isPenaltyTerm(GlobalID nodeID); //this function queries
    void penaltyTerm(GlobalID nodeID);  //this function sets the
                                                    //value.

//  begin field-oriented additions for v1.0

    int getNumFields(GlobalID nodeID);
    void setNumFields(GlobalID nodeID, int numFields);

    int* getFieldIDList(GlobalID nodeID, int& numFields);
    void setFieldIDList(GlobalID nodeID, int* fieldIDList, int numFields);

    int* getFieldOffsetList(GlobalID nodeID, int& numFields);
    void setFieldOffsetList(GlobalID nodeID, int* fieldOffsets, int numFields);

    int getFieldOffset(GlobalID nodeID, int fieldID);

//  end field-oriented additions for v1.0

    //the following functions are for managing the tables of penalty
    //constraint information for nodes that appear in penalty constraints.
    //the functions which return void are for putting information into
    //the tables, and the other functions are for getting information out.

    void localPenNodeList(int penID, GlobalID *nodeList, int lenList);
    void localPenEqnIDList(int penID, int *eqnIDList, int lenList);
    void localPenNumDOFList(int penID, int *DOFList, int lenList);

    GlobalID** localPenNodeTable(int** lenPenNodes);
    int** localPenEqnIDTable(int** lenPenNodes);
    int** localPenNumDOFTable(int** lenLocPenNodes);
    int* localPenIDList(int& lclNumPenCRs);
    int** localPenProcTable(int **lenProcTable);
    int localNumPenCRs() {return(localNumPenCRs_);};

    void calcPenProcTable();
    int numOutgoingPenCRs(int proc);

    void remotePenNodeList(int penID, int proc, GlobalID *nodeList,
                           int lenList);
    void remotePenEqnIDList(int penID, int proc, int *eqnIDList, int lenList);
    void remotePenNumDOFList(int penID, int proc, int *DOFList, int lenList);
    int remotePenArraySize(int penID, int proc);

    GlobalID** remotePenNodeTable(int** lenRemPenNodes);
    int** remotePenEqnIDTable(int** lenRemPenNodes);
    int** remotePenNumDOFTable(int** lenRemPenNodes);
    int* remotePenIDList(int& remNumPenCRs);
    int& remoteNumPenCRs() {return(remoteNumPenCRs_);};
    int getRemotePenIDIndex(int penID, int proc);

  private:

    ExtNode* newExtNode();
    void deleteExtNode(ExtNode *node);

    int allExtNodeIndex(GlobalID nodeID);
    int localNodeIndex(GlobalID nodeID);
    int remoteNodeIndex(GlobalID nodeID);

    int getRecvProcIndex(int proc);
    int getSendProcIndex(int proc);
    int getLocPenIDIndex(int penID);
 
    //following are pointers which will hold the main data structures of
    //this class. These are 1D lists of pointer-to-ExtNode structs.

    ExtNode** allExtNodes_; //this is the initial array of "all external nodes"
    int numAllExtNodes_;

    ExtNode** remoteNodes_; //external nodes which reside remotely.
    int numRemoteNodes_;

    ExtNode** localNodes_; //external nodes which reside locally.
    int numLocalNodes_;

    GlobalID* externNodeList_;

          //Below are some lists and tables which duplicate a lot
          //of the information in the remoteNodes_ structs. These
          //lists and tables collect the information into groups that
          //are better suited to sending to/receiving from other
          //processors, similarly to the way Kim was doing it in the
          //main ISIS_SLE.cc file. There is some memory duplication
          //here, which should be reduced. Right now I'm going after
          //encapsulation, removing lists and tables from ISIS_SLE in
          //order to improve readability.

    int* recvProcList_;     //this is a list of the processors that
                            //own nodes which appear in CR's on this proc
    int lenRecvProcList_;

    GlobalID** remoteNodeIDs_; //this array has first dimension of size
                             //lenRecvProcList_ and ragged second dimension,
                             //depending on how many of our remote nodes are
                             //owned by each processor in
    int* lenRecv_;           //recvProcList_. These ragged array sizes are
                             //kept in lenRecv_.

    int* sendProcList_;     //this is a list of the processors that have
                            //CR's that contain nodes that we own
    int lenSendProcList_;

    GlobalID** localNodeIDs_;  //these two arrays have first dimension of size
     
    int** DOFs_;              //lenSendProcList_, and second dimension of
                              //ragged size, depending on how many nodes are
                              //$kdm - is this array used anywhere?
                              
    int* lenSend_;            //in CRs on each processor in sendProcList_.
                              //These ragged array sizes are kept in lenSend_.

    int remoteNumPenCRs_;
    int localNumPenCRs_;      //How many penalty constraints, in total, the
                              //external nodes on this processor are involved
                              //in.
    
                //the following arrays are only allocated if this external
                //node record contains nodes which appear in penalty
                //constraints.
    GlobalID **locPenNodes_; // first dimension localNumPenCRs_,
                         // second dim. lenLocPenNodes_[i]
    int *lenLocPenNodes_;   // length of rows of penNodes array = # of nodes in
                         // each penalty constraint
    int **locPenEqnIDs_;     //same dimensions as locPenNodes
    int **locPenNumDOF_;     //same dimensions as locPenNodes
    int *locPenID_;    //list of IDs for distinguishing between the different
                    //penalty constraints
    int **locPenProcs_; //procs to which local penalty information will need
                        //to be sent.
    int *lenLocPenProcs_; //lengths of the rows of locPenProcs_;

    GlobalID **remPenNodes_; // first dimension incomingNumPenCRs_,
                         // second dim. lenRemPenNodes_[i]
    int *lenRemPenNodes_; // lengths of rows of remPenNodes array = # of nodes
                          // in each penalty constraint
    int **remPenEqnIDs_;     //same dimensions as remPenNodes
    int **remPenNumDOF_;     //same dimensions as remPenNodes
    int *remPenID_;    //list of IDs for distinguishing between the different
                    //penalty constraints
    int *remPenProcs_; //processors that own remote penalty constraints.
};

#endif
