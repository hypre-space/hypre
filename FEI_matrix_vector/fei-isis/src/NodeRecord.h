#ifndef __NodeRecord_H
#define __NodeRecord_H

// #include "basicTypes.h"
// #include "BCRecord.h"

//  NodeRecord class stores raw data for each node - note that since a
//  given node may be associated with more than one block, some block-style
//  data (such as the number of solution parameters at each node) is more
//  naturally defined at the level of the node

class NodeRecord {

  public:
    NodeRecord();
    ~NodeRecord();

    GlobalID getGlobalNodeID() const {return globalNodeID_;};
    void setGlobalNodeID(GlobalID gNID) {globalNodeID_ = gNID;};

    int getNumNodalDOF() const {return numNodalDOF_;};
    void setNumNodalDOF(int gNDOF) {numNodalDOF_ = gNDOF;};

    int getLocalEqnID() const {return localEqnID_;};
    void setLocalEqnID(int eqnID) {localEqnID_ = eqnID;};

    int ownerProc() {return ownerProc_;};
    void ownerProc(int ownProc) {ownerProc_ = ownProc;};

    int getNumNodalFields() const {return numNodalFields_;};
    void setNumNodalFields(int gNumNodalFields) 
        {numNodalFields_ = gNumNodalFields;};

    int *pointerToProcList(int& nProcs);
    void allocateProcList(int nProcs);

    int *pointerToBlockList(int& nBlocks);
    void allocateBlockList(int nBlocks);

    int *pointerToFieldIDList(int& nFields);
    int *pointerToFieldOffsetList(int& nFields);
    void allocateFieldLists(int nFields);

    double *pointerToSoln(int& numDOF);
    void allocateSolnList();

//indicate whether a given node has b.c. data associated with it, and if it 
//does, whether that data has been implemented into the system matrices yet

    bool hasBCDataFlag() const {return hasBCData_;};
    void setBCDataFlag() {hasBCData_ = true;};
    void clearBCDataFlag() {hasBCData_ = false;};

    bool hasBCDoneFlag() const {return hasBCDone_;};
    void setBCDoneFlag() {hasBCDone_ = true;};
    void clearBCDoneFlag() {hasBCDone_ = false;};

//  new functions to store multiple-record and/or multiple-field BCs

    int getNumTempBCs() const {return numTempBCs_;};
    int getNumBCs() const {return numBCRecords_;};
    void removeBCData();
    
    int getFieldOffset(int myFieldID);

    BCRecord *pointerToBCRecords(int& lenBCList);
    void allocateBCRecords(int lenBCList);

    BCRecord** pointerToBCTempList(int& numTempBCs);

    void addBCRecord(int fieldID, 
                     int fieldSize,
                     const double *alpha, 
                     const double *beta, 
                     const double *gamma);

    void destroyTempBCList();


//  debug output stuff...

    void dumpToScreen();


  private:
    GlobalID globalNodeID_; // global ID number for this node
    int numNodalDOF_;       // number of soln params for this node
    int numNodalFields_;    // number of fields present for this node
    int localEqnID_;        // index into local eqns for first nodal soln param

    int numProcs_;     // number of processors sharing this node
    int ownerProc_;    // processor that owns this node
    int *procList_;    // list of processors associated with this node

    int numBlocks_;         // number of blocks sharing this node
    int *blockList_;        // list of blocks sharing this node

    int numFields_;         // number of fields present at this node
    int *fieldIDList_;      // list of IDs for fields present at this node
    int *fieldOffsetList_;  // list of offsets for fields present at this node

//  the following nodal parameters don't have to be stored for each node, and
//  will probably end up packed into lists, with appropriate list indices stored
//  in the node record fields.  For now, let's put all the data here, and worry
//  about getting it more efficiently stored later...

    BCRecord *BCRecordList_;
    int numBCRecords_;

//  intermediate storage for BC data    
    BCRecord** BCTempList_;
    int numTempBCs_;            // number of individual bc's at this node

    double *nodSoln_;       // solution parameters for this node
    
    bool hasBCData_;        // is there b.c. data for this node?
    bool hasBCDone_;        // has the b.c. data for this node been implemented?
};
 
#endif

