#ifndef __PenConstRecord_H
#define __PenConstRecord_H

//  PenConstRecord class stores raw data for each penalty formulation
//  constraint equation, including the nodes defining the constraint and
//  the various weights that contribute to the nonzero matrix terms

class PenConstRecord {
  public:
    PenConstRecord();
    ~PenConstRecord();

    int getCRPenID() {return CRPenID_;};
    void setCRPenID(int icPenID) {CRPenID_ = icPenID;};

    int getLenCRNodeList() {return lenCRNodeList_;};
    void setLenCRNodeList(int lenCRList) {lenCRNodeList_ = lenCRList;};

    int getNumPenCRs() {return numPenCRs_;};
    void setNumPenCRs(int numPenCRs) {numPenCRs_ = numPenCRs;};

    GlobalID **pointerToCRNodeTable(int& numRows, int& numCols);
	void allocateCRNodeTable(int numRows, int numCols);

    bool **pointerToCRIsLocalTable(int& numRows, int& numCols);
    void allocateCRIsLocalTable(int numRows, int numCols);

    double **pointerToCRNodeWeights(int& numRows, int* &numCols);
	void allocateCRNodeWeights(int numRows, int *numCols);

    double *pointerToCRConstValues(int& length);
	void allocateCRConstValues(int length);

    int *pointerToCRFieldList(int& length);
    void allocateCRFieldList(int length);

    double getValCRPenID() {return valCRPenID_;};
    void setValCRPenID(double CRPenVal) {valCRPenID_ = CRPenVal;};

//$	temporary output for debugging...

	void dumpToScreen();
    void remoteNode(GlobalID nodeID);

  private:

    int CRPenID_;             // returned identifier for constraint
    double valCRPenID_;       // penalty number for this constraint
    int lenCRNodeList_;       // length of CRNodeList
    int numPenCRs_;           // number of constraints in this record
       
    GlobalID **CRNodeTable_;   // list of nodes associated with constraint
    bool **CRIsLocalTable_;    // table of flags for locality of nodes
    bool allocatedCRIsLocalTable_;
    int numRowsNodeTable_;     // number of rows in CRNodeTable_
    int numColsNodeTable_;     // number of columns in CRNodeTable_
    
    bool allocatedCRNodeWeights_;
    double **CRNodeWeights_;   // node-based weights for constraint
    int numRowsNodeWeights_;   // number of rows in CRNodeWeights_
    int *numColsNodeWeights_;  // array of rowlengths in CRNodeWeights_
    
    double *CRConstValue_;     // constant values for constraint enforcement
    int lenCRConstValue_;      // constant values list length (redundant)

    int *CRFieldList_;         // list of FieldIDs for constraint block
    int lenCRFieldList_;       // fieldIDs list length (redundant)

};
 
#endif

