#ifndef __SharedNodeRecord_H
#define __SharedNodeRecord_H

// #include "basicTypes.h"

/*
   SharedNodeRecord is a container class which holds and manages
   information about shared nodes.

   Shared nodes are nodes which reside (in the finite element sense) on
   other processors besides this one. However, in terms of the global
   matrix system to be solved, the equation(s) associated with shared
   nodes will only reside on one processor. Thus, we need to determine
   "ownership" and exchange information between sharing processors
   relating to the node's global equation number, etc.
*/

struct SharedNode {
    GlobalID nodeID; //this node's ID
    int ownerProc;     //this node's owner
    int numProcs;      //number of sharing procs, including owner
    int numEquations; //number of DOF for this node
    int* procIDs; //list of sharing procs
    int* equationLengths; //number of scatter indices coming from each proc
    int equationNumber;  //global equation number on owning proc
    int localElems;      //elements on this proc which contain this node
    int *remoteElems;    //number of elems on sharing procs that contain
                          //this node
    int numLocalElems;  //number of elems on this proc that contain this node
};

class SharedNodeRecord {
  public:
    SharedNodeRecord();
    ~SharedNodeRecord();

    //function for putting in shared nodes along with lists of sharing procs.

    void sharedNodes(const GlobalID *sharedNodeIDs, int lenSharedNodeIDs,
                     const int *const *sharedProcIDs,
                     const int *lenSharedProcIDs,
                     int localProc);

    void initComplete();

    //functions for setting and retrieving shared node information. see
    //.cc file for descriptions of these functions.

    int numMesgsLocalNodes();
    int numMesgsRemoteNodes();

    GlobalID *pointerToLocalShared(int& numLocalShared);
    GlobalID *pointerToRemoteShared(int& numLocalShared);

    int ownerProc(GlobalID nodeID);

    int* procIDs(GlobalID nodeID, int& numprocs);
    void procIDs(GlobalID nodeID, const int* procs, int numprocs);
 
    int isShared(GlobalID nodeID);

    int numEquations(GlobalID nodeID);
    void numEquations(GlobalID nodeID, int numEqns);

    int* pointerToProcIDs(GlobalID nodeID, int& numProcs);

    int remoteElems(GlobalID nodeID, int procID) const;
    void remoteElems(GlobalID nodeID, int procID, int numRElems);

    int totalRemoteElems();

    int equationNumber(GlobalID nodeID);
    void equationNumber(GlobalID nodeID, int eqnNum);

    int localElems(GlobalID nodeID) const;
    void localElems(GlobalID nodeID, int numElems);

    int* pointerToSharingProcs(int& numProcs);

    void printem();
  private:
    void appendIntList(int **list, int* lenList, int newItem);
    void appendGlobalIDList(GlobalID **list, int* lenList, GlobalID newItem);
 
    int sharedNodeIndex(GlobalID nodeID) const;
    int chooseOwner(int *list, int lenList);
 
    void buildSharingProcList();

    SharedNode* sharedNodes_;
    int numSharedNodes_;

    int localProc_;
    bool initCompleted_;

    GlobalID *localShared_;
    int numLocalShared_;
    GlobalID *remoteShared_;
    int numRemoteShared_;

    int* sharingProcs_;
    int numSharingProcs_;
};

#endif
