#ifndef _ProcEqns_h_
#define _ProcEqns_h_

class ProcEqns {
 public:
   ProcEqns();
   ~ProcEqns();

   int getNumProcs() {return(numProcs_);};
   int* procsPtr() {return(procs_);};
   int* eqnsPerProcPtr() {return(eqnsPerProc_);};
   int** procEqnNumbersPtr() {return(procEqnNumbers_);};
   int** procEqnLengthsPtr() {return(procEqnLengths_);};

   void addEqn(int eqnNumber, int proc);
   void addEqn(int eqnNumber, int eqnLength, int proc);

   void setProcEqnLength(int index1, int index2, int eqnLength);

   void setProcEqnLengths(int* eqnNumbers, int* eqnLengths, int len);

 private:
   void deleteMemory();
   void internalAddEqn(int eqnNumber, int eqnLength, int proc);

   int numProcs_;
   int* procs_;
   int* eqnsPerProc_;
   int** procEqnNumbers_;
   int** procEqnLengths_;
};

#endif

