#ifndef __FE_Node_H
#define __FE_Node_H

//  Node class for a simple-minded 1D beam program
//
//  k.d. mish, august 13, 1997
//
//  this program is used for testing the ISIS++ finite-element interface
//  module, as a 1D beam is (a) easy to partition among an arbitrary
//  arrangement of parallel processors, (b) easy to check answers for (since
//  various closed-form solutions exist), and (c) possesses sufficiently
//  interesting data (e.g., multiple nodal solution parameters) so that one
//  can perform reasonable code coverage checks...

class FE_Node {

  public:
    FE_Node();
    ~FE_Node();

    GlobalID globalNodeID() const {return globalNodeID_;};
    void globalNodeID(GlobalID gNID) {globalNodeID_ = gNID;};

    int numNodalDOF() const {return numNodalDOF_;};
    void numNodalDOF(int gNDOF) {numNodalDOF_ = gNDOF;};

    double nodePosition() const {return nodePosition_;};
    void nodePosition(double gPosition) {nodePosition_ = gPosition;};

    double *pointerToSoln(int& numDOF);
    void allocateSolnList();

//$ temporary output for debugging...

    void dumpToScreen();


  private:
    GlobalID globalNodeID_;   // global ID number for this node
    int numNodalDOF_;         // number of soln params for this node
    double nodePosition_;     // x-coordinate for this node

    double *nodSoln_;         // solution parameters for this node
};
 
#endif

