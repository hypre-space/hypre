#ifndef __NodePackets_h
#define __NodePackets_h

// declare a couple of struct types to use for passing information about
// nodes back and forth between processors

struct NodePacket {
    GlobalID nodeID;
    int numEquations;
    int numStiffnessDoubles;
    int numStiffnessInts;
    int numLoadDoubles;
};

struct NodeControlPacket {
    GlobalID nodeID;
    int numEqns;    //number of equations
    int eqnNumber;  //equation number on the processor that owns this node
    int numElems;   //number of elements this node appears in
    int numIndices; //number of scatter indices
    int numPenCRs;  //number of penalty constraints
    int numMultCRs; //number of multiplier constraints
    int numFields;  //number of solution fields
};
 
#define MAX_SOLN_CARD 9                 // maximum number of unknowns per node
#define WTPACK_SIZE  MAX_SOLN_CARD+3    // size of packet
 
struct NodeWeightPacket {
    GlobalID nodeID;
    int sysEqnID;
    int numSolnParams;
    double weights[MAX_SOLN_CARD];
};
 
#endif
