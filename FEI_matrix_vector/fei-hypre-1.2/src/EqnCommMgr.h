#ifndef _EqnCommMgr_h_
#define _EqnCommMgr_h_

//
//The EqnCommMgr (Equation communication manager) class is responsible
//for keeping track of equations that require communications. There
//are two types of equations in this class:
//
// 1. Equations that are owned by the local processor, but which remote
//    processors contribute to (e.g., because they share some of our
//    active nodes, or they own a node that appears in a constraint that
//    this processor owns).
//    These will be called 'recv equations' -- we're receiving contributions
//    to our section of the system matrix.
//
// 2. Equations that are remotely owned, but that the local processor
//    contributes to (the mirror of case 1.).
//    These will be called 'send equations' -- we're sending contributions
//    to remote sections of the system matrix.
//
//The send/recv naming thing breaks down a little bit, because we do the
//reverse when we exchange solution values. The 'send equations', or remotely
//owned equations, represent equations for which we must receive the solution
//from the owning processor. Likewise, the recv equations (locally owned)
//represent equations for which we must send the solution to the sharing
//processors. Oh well, I couldn't think of a better pair of names though...
//
//Usage Notes:
//
// 1. The first function called after construction should be setNumRHSs.
// 2. You can't call exchangeIndices until after setNumRHSs has been called.
// 3. After exchangeIndices is called, you can't call addRecvEqn anymore.
// 4. You can't call exchangeEqns until after exchangeIndices has been called.
// 5. You can't call addSolnValues until after exchangeIndices has been
//    called.
//
//In general, usage should proceed like this:
//
//in BASE_FEI::initComplete:
//       setNumRHSs
//       addRecvEqn  (a bunch of times, probably)
//       addSendIndices  (also a bunch of times)
//
//       exchangeIndices
//
//       getNumRecvEqns
//       recvEqnNumbersPtr
//       recvIndicesPtr
//
//in BASE_FEI::loadElemSet/loadElemSetMatrix/loadElemSetRHS
//       addSendEqn and/or addSendRHS
//
//in BASE_FEI::loadComplete
//       exchangeEqns
//       getNumRecvEqns
//       recvEqnNumbersPtr
//       recvIndicesPtr
//       recvCoefsPtr
//       recvRHSsPtr
//
//in BASE_FEI::unpackSolution
//       getNumRecvEqns
//       recvEqnNumbersPtr
//       addSolnValues
//       exchangeSoln
//
//in BASE_FEI various getSoln functions
//       getNumSendEqns
//       sendEqnNumbersPtr
//       sendEqnSolnPtr
//

class EqnCommMgr {
 public:
   EqnCommMgr(int localProc);
   virtual ~EqnCommMgr();

   int getNumRecvProcs() {return(recvProcEqns_->getNumProcs());};
   int* recvProcsPtr() {return(recvProcEqns_->procsPtr());};

   int getNumSendProcs() {return(sendProcEqns_->getNumProcs());};
   int* sendProcsPtr() {return(sendProcEqns_->procsPtr());};

   void addRecvEqn(int eqnNumber, int srcProc);

   void addSolnValues(int* eqnNumbers, double* values, int num);

   void exchangeIndices(MPI_Comm comm);
   void exchangeEqns(MPI_Comm comm);
   void exchangeSoln(MPI_Comm comm);

   int getNumRecvEqns() {return(recvEqns_->getNumEqns());};

   int* recvEqnNumbersPtr() {return(recvEqns_->eqnNumbersPtr());};
   int* recvEqnLengthsPtr() {return(recvEqns_->lengthsPtr());};
   int** recvIndicesPtr() {return(recvEqns_->indicesPtr());};
   double** recvCoefsPtr() {return(recvEqns_->coefsPtr());};
   double** recvRHSsPtr() {return(recvEqns_->rhsCoefsPtr());};

   void addSendEqn(int eqnNumber, int destProc, const double* coefs,
                   const int* indices, int num);

   void setNumRHSs(int numRHSs);

   void addSendRHS(int eqnNumber, int destProc, int rhsIndex, double value);

   void addSendIndices(int eqnNumber, int destProc, int* indices, int num);

   int getNumSendEqns() {return(sendEqns_->getNumEqns());};

   int* sendEqnNumbersPtr() {return(sendEqns_->eqnNumbersPtr());};

   double* sendEqnSolnPtr() {return(sendEqnSoln_);};

   void resetCoefs();

   void exchangeEssBCs(int* essEqns, int numEssEqns, double* essAlpha,
                       double* essGamma, MPI_Comm comm);

   int getNumEssBCEqns() {return(essBCEqns_->getNumEqns());};
   int* essBCEqnsPtr() {return(essBCEqns_->eqnNumbersPtr());};
   int** essBCIndicesPtr() {return(essBCEqns_->indicesPtr());};
   int* essBCEqnLengthsPtr() {return(essBCEqns_->lengthsPtr());};
   double** essBCCoefsPtr() {return(essBCEqns_->coefsPtr());};

 private:
   void deleteEssBCs();
   int getSendProcNumber(int eqn);
   int getNumRecvEqns(ProcEqns* sendProcEqns, MPI_Comm comm);
   void exchangeEqnBuffers(MPI_Comm comm, ProcEqns* sendProcEqns,
                           EqnBuffer* sendEqns, ProcEqns* recvProcEqns,
                           EqnBuffer* recvEqns, bool accumulate);

   int localProc_;

   ProcEqns* recvProcEqns_;

   bool setNumRHSsCalled_;      //whether or not the setNumRHSs function has
                                //been called yet.
   int numRHSs_;

   bool exchangeIndicesCalled_; //whether or not the exchangeIndices function
                                //has been called yet.

   EqnBuffer* recvEqns_;

   double* solnValues_;       //solution values we'll need to return to the
                              //processors that contribute to our equations

   ProcEqns* sendProcEqns_;

   int** sendProcEqnLocations_; //each send proc eqn's location in the single
                                //sendEqnNumbers_ list.

   EqnBuffer* sendEqns_;

   double* sendEqnSoln_;  //the solution values for the send equations. i.e.,
                          //we'll recv these solution values for the equations
                          //that we contributed to (sent) for other processors.

   EqnBuffer* essBCEqns_;
};

#endif

