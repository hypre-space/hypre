#ifndef __MultConstRecord_H
#define __MultConstRecord_H

//  MultConstRecord class stores raw data for each Lagrange-multiplier
//  constraint equation, including the nodes defining the constraint, the
//  multipliers included in the constraint, and the various weights that
//  contribute to the nonzero matrix terms

class MultConstRecord {
  public:
    MultConstRecord();
    virtual ~MultConstRecord();

    int getCRMultID() {return CRMultID_;};
    void setCRMultID(int icMultID) {CRMultID_ = icMultID;};

    int getEqnNumber() const {return(eqnNumber_);};
    void setEqnNumber(int eqn) {eqnNumber_ = eqn;};

    int getLenCRNodeList() {return lenCRNodeList_;};
    void setLenCRNodeList(int lenCRList) {lenCRNodeList_ = lenCRList;};

    int getNumMultCRs() {return numMultCRs_;};
    void setNumMultCRs(int numMultCRs) {numMultCRs_ = numMultCRs;};

    GlobalID **pointerToCRNodeTable(int& numRows, int& numCols);
	void allocateCRNodeTable(int numRows, int numCols);

    double **pointerToCRNodeWeights(int& numRows, int* &numCols);
    void allocateCRNodeWeights(int numRows, int *numCols);

    double *pointerToCRConstValues(int& length);
    void allocateCRConstValues(int length);

    int *pointerToCRFieldList(int& length);
    void allocateCRFieldList(int length);

    double *pointerToMultipliers(int& multLength);
	void allocateMultipliers(int multLength);

//$	temporary output for debugging...

	void dumpToScreen();

  private:
  
    int CRMultID_;             // returned identifier for Lagrange multiplier
    int lenCRNodeList_;        // length of CRNodeList
    int numMultCRs_;           // number of constraints in this record
    int eqnNumber_;           // index into soln vector for first multiplier
       
    GlobalID **CRNodeTable_;   // table of nodes associated with constraint
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
    
    double *MultValues_;       // list of Lagrange multiplier computed values
    int lenMultValues_;        // length of MultValues_ list (redundant)

};

#endif

