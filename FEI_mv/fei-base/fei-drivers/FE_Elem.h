#ifndef __FE_Elem_H
#define __FE_Elem_H

//  Element class for a simple 1D beam on elastic foundation program
//
//  k.d. mish, august 13, 1997
//
//  this program is used for testing the ISIS++ finite-element interface
//  module, as a 1D beam is (a) easy to partition among an arbitrary
//  arrangement of parallel processors, (b) easy to check answers for (since
//  various closed-form solutions exist), and (c) possesses sufficiently
//  interesting data (e.g., multiple nodal solution parameters) so that one
//  can perform reasonable code coverage checks...

class FE_Elem {

  public:
    FE_Elem();
    ~FE_Elem();


    GlobalID globalElemID() const {return globalElemID_;};
    void globalElemID(GlobalID gNID) {globalElemID_ = gNID;};

    int numElemRows() const {return numElemRows_;};
    void numElemRows(int gNERows) {numElemRows_ = gNERows;};

    int numElemNodes() const {return numElemNodes_;};
    void numElemNodes(int gNodes) {numElemNodes_ = gNodes;};

    double distLoad() const {return distLoad_;};
    void distLoad(double gLoad) {distLoad_ = gLoad;};

    double axialLoad() const {return axialLoad_;};
    void axialLoad(double gLoad) {axialLoad_ = gLoad;};

    double bendStiff() const {return bendStiff_;};
    void bendStiff(double gStiff) {bendStiff_ = gStiff;};

    double axialStiff() const {return axialStiff_;};
    void axialStiff(double gStiff) {axialStiff_ = gStiff;};

    double foundStiff() const {return foundStiff_;};
    void foundStiff(double gStiff) {foundStiff_ = gStiff;};

    double elemLength() const {return elemLength_;};
    void elemLength(double gLength) {elemLength_ = gLength;};


    void allocateNodeList();
    void storeNodeList(int gNumNodes, GlobalID *elemNodes);
    void returnNodeList(int& gNumNodes, GlobalID* listPtr);

    void allocateElemForces();
    double *evaluateElemForces(int& gForceDOF);

    void allocateLoad();
    void evaluateLoad(int& neRows, double *loadPtr);

    void allocateStiffness();
    void evaluateStiffness(int& neRows, double **stiffPtr);

//$ temporary output for debugging...

    void dumpToScreen();


  private:
    GlobalID globalElemID_;   // global ID number for this element
    int numElemNodes_;        // number of nodes associated with this element
    int numElemRows_;         // number of rows in the element matrices
    
    GlobalID *nodeList_;      // list of nodes associated with this element
    double *elemLoad_;        // load vector ptr for this element
    double **elemStiff_;      // stiffness matrix ptr for this element

    double distLoad_;         // uniform transerve load on this element
    double axialLoad_;        // uniform axial load on this element
    double bendStiff_;        // bending stiffness (EI) for this element
    double axialStiff_;       // membrane stiffness (EA) for this element
    double foundStiff_;       // elastic foundation stiffness for this element
    double elemLength_;       // length of this 1D element
};
 
#endif

