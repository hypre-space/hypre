#ifndef _EqnBuffer_h_
#define _EqnBuffer_h_

class EqnBuffer {
 public:
   EqnBuffer();
   ~EqnBuffer();

   int getNumEqns() {return(numEqns_);};
   int* eqnNumbersPtr() {return(eqnNumbers_);};
   int** indicesPtr() {return(indices_);};
   double** coefsPtr() {return(coefs_);};
   int* lengthsPtr() {return(eqnLengths_);};

   int getNumRHSs() {return(numRHSs_);};
   void setNumRHSs(int n);
   void addRHS(int eqnNumber, int rhsIndex, double value);
   int* rhsIndicesPtr() {return(rhsIndices_);};
   double** rhsCoefsPtr() {return(rhsCoefs_);};

   int getEqnIndex(int eqn);

   int isInIndices(int eqn);

   void addEqn(int eqnNumber, const double* coefs, const int* indices,
               int len, bool accumulate);

   void resetCoefs();

   void addIndices(int eqnNumber, const int* indices, int len);

 private:
   void deleteMemory();
   void internalAddEqn(int index, const double* coefs,
                       const int* indices, int len, bool accumulate);

   int numEqns_;
   int* eqnNumbers_;
   int** indices_;
   double** coefs_;
   int* eqnLengths_;
   int numRHSs_;
   int* rhsIndices_;
   double** rhsCoefs_;
   bool setNumRHSsCalled_;
   bool rhsCoefsAllocated_;
};

#endif

