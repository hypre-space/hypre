#include <stdlib.h>
#include <iostream.h>

#ifdef FEI_SER
#include "mpiuni/mpi.h"
#else
#include <mpi.h>
#endif

#include "other/basicTypes.h"

#include "src/Utils.h"
#include "src/ProcEqns.h"
#include "src/EqnBuffer.h"
#include "src/EqnCommMgr.h"

//=====Constructor==============================================================
EqnCommMgr::EqnCommMgr(int localProc)
 : localProc_(localProc),
   recvProcEqns_(NULL),
   setNumRHSsCalled_(false),
   numRHSs_(0),
   exchangeIndicesCalled_(false),
   recvEqns_(NULL),
   solnValues_(NULL),
   sendProcEqns_(NULL),
   sendProcEqnLocations_(NULL),
   sendEqns_(NULL),
   sendEqnSoln_(NULL),
   essBCEqns_(NULL)
{
   recvEqns_ = new EqnBuffer();
   sendEqns_ = new EqnBuffer();
   recvProcEqns_ = new ProcEqns();
   sendProcEqns_ = new ProcEqns();

   essBCEqns_ = new EqnBuffer();
}

//=====Destructor===============================================================
EqnCommMgr::~EqnCommMgr() {

   delete [] sendEqnSoln_;
   sendEqnSoln_ = NULL;

   delete [] solnValues_;
   solnValues_ = NULL;

   int numSendProcs = sendProcEqns_->getNumProcs();

   for(int i=0; i<numSendProcs; i++) {
      delete [] sendProcEqnLocations_[i];
   }

   delete [] sendProcEqnLocations_;
   sendProcEqnLocations_ = NULL;

   delete recvEqns_;
   delete sendEqns_;
   delete recvProcEqns_;
   delete sendProcEqns_;

   delete essBCEqns_;
}

//==============================================================================
void EqnCommMgr::addRecvEqn(int eqnNumber, int srcProc) {
//
//This function adds the eqnNumber, srcProc pair to the recvProcsEqns_
//object, which does the right thing as far as only putting it in lists that
//it isn't already in, preserving order, etc. 
//
   if (srcProc == localProc_) {
      cerr << "EqnCommMgr::addRecvEqn: ERROR, srcProc == localProc_, which is"
           << " a recipe for a deadlock." << endl;
      abort();

   }

   recvProcEqns_->addEqn(eqnNumber, srcProc);
}

//==============================================================================
void EqnCommMgr::addSolnValues(int* eqnNumbers, double* values, int num) {

   if (!exchangeIndicesCalled_) {
      cerr << "EqnCommMgr::addSolnValues: ERROR, you may not call this until"
              " after exchangeIndices has been called." << endl;
      abort();
   }

   int numRecvEqns = recvEqns_->getNumEqns();
   int* recvEqnNumbers = recvEqns_->eqnNumbersPtr();

   for(int i=0; i<num; i++) {
      int tmp = -1;
      int index = Utils::sortedIntListFind(eqnNumbers[i], recvEqnNumbers,
                                           numRecvEqns, &tmp);

      if (index < 0) continue;

      solnValues_[index] = values[i];
   }
}

//==============================================================================
void EqnCommMgr::exchangeIndices(MPI_Comm comm) {
//
//This function performs the communication necessary to exchange remote
//contributions to the matrix structure (indices only, not coefficients)
//among all participating processors.
//
   int i;

   int numRecvProcs = recvProcEqns_->getNumProcs();
   int numSendProcs = sendProcEqns_->getNumProcs();

   if ((numRecvProcs == 0) && (numSendProcs == 0)) return;

   if (!setNumRHSsCalled_) {
      cerr << "EqnCommMgr::exchangeIndices: ERROR, don't call this until after"
           << " EqnCommMgr::setNumRHSs." << endl;
      abort();
   }

   sendEqns_->setNumRHSs(numRHSs_);

   int localProc;
   MPI_Comm_rank(comm, &localProc);

   int lenTag = 199903;
   int indTag = 199904;

   int numSendEqns = sendEqns_->getNumEqns();
   int* sendEqnNumbers = sendEqns_->eqnNumbersPtr();
   int* sendEqnLengths = sendEqns_->lengthsPtr();
   int** sendIndices = sendEqns_->indicesPtr();

   int* sendProcs = sendProcEqns_->procsPtr();
   int* eqnsPerSendProc = sendProcEqns_->eqnsPerProcPtr();
   int** sendProcEqnNumbers = sendProcEqns_->procEqnNumbersPtr();

   //we'll need to get the lengths of the incoming index lists, so let's create
   //the lists into which we'll recv those lengths and indices.

   int* recvProcs = recvProcEqns_->procsPtr();
   int** recvProcEqnNumbers = recvProcEqns_->procEqnNumbersPtr();
   int* eqnsPerRecvProc = recvProcEqns_->eqnsPerProcPtr();

   int** tmpRecvLengths = NULL;
   int** recvProcEqnIndices = NULL;

   if (numRecvProcs > 0) {
      tmpRecvLengths = new int*[numRecvProcs];
      recvProcEqnIndices = new int*[numRecvProcs];
   }

   for(i=0; i<numRecvProcs; i++) {
      tmpRecvLengths[i] = new int[eqnsPerRecvProc[i]];
      for(int j=0; j<eqnsPerRecvProc[i]; j++)
         tmpRecvLengths[i][j] = 0;
   }

   MPI_Request* lenRequests = NULL;
   MPI_Request* indRequests = NULL;

   if (numRecvProcs > 0) {
      lenRequests = new MPI_Request[numRecvProcs];
      indRequests = new MPI_Request[numRecvProcs];
   }

   for(i=0; i<numRecvProcs; i++) {
      MPI_Irecv(tmpRecvLengths[i], eqnsPerRecvProc[i], MPI_INT,
                recvProcs[i], lenTag, comm, &lenRequests[i]);
   }

   sendProcEqns_->setProcEqnLengths(sendEqnNumbers, sendEqnLengths,
                                    numSendEqns);

   int** sendProcLengths = sendProcEqns_->procEqnLengthsPtr();
   if (numSendProcs > 0) {

      //let's construct lists containing, for each send proc, the location
      //of each of its equations in the sendEqnNumbers list.
      sendProcEqnLocations_ = new int*[numSendProcs];
   }

   for(i=0; i<numSendProcs; i++) {
      sendProcEqnLocations_[i] = new int[eqnsPerSendProc[i]];

      //find the location of each outgoing send eqn...
      for(int j=0; j<eqnsPerSendProc[i]; j++) {
         int tmp = -1;
         int index = Utils::sortedIntListFind(sendProcEqnNumbers[i][j],
                                              sendEqnNumbers, numSendEqns,
                                              &tmp);

         sendProcEqnLocations_[i][j] = index;
      }

      MPI_Send(sendProcLengths[i], eqnsPerSendProc[i], MPI_INT,
               sendProcs[i], lenTag, comm);
   }

   //while we're here, let's allocate the array into which we will (later)
   //recv the soln values for the remote equations we've contributed to.
   sendEqnSoln_ = new double[numSendEqns];

   //now, let's complete the above recvs and allocate the space for the 
   //incoming equations.
   for(i=0; i<numRecvProcs; i++) {
      MPI_Status status;
      int index;
      MPI_Waitany(numRecvProcs, lenRequests, &index, &status);

      int totalLength = 0;
      for(int j=0; j<eqnsPerRecvProc[index]; j++) {
         totalLength += tmpRecvLengths[index][j];
         recvProcEqns_->setProcEqnLength(index, j, tmpRecvLengths[index][j]);
      }

      delete [] tmpRecvLengths[index];

      recvProcEqnIndices[index] = new int[totalLength];

      //let's go ahead and launch the recvs for the indices now.
      MPI_Irecv(recvProcEqnIndices[index], totalLength, MPI_INT,
                recvProcs[index], indTag, comm, &indRequests[index]);
   }

   delete [] tmpRecvLengths;

   //ok, now we need to build the lists of outgoing indices and send those.
   for(i=0; i<numSendProcs; i++) {
      int totalLength = 0;
      int j;
      for(j=0; j<eqnsPerSendProc[i]; j++)
         totalLength += sendProcLengths[i][j];

      int* indices = new int[totalLength];
      int offset = 0;

      for(j=0; j<eqnsPerSendProc[i]; j++) {
         int eqnLoc = sendProcEqnLocations_[i][j];

         for(int k=0; k<sendProcLengths[i][j]; k++) {
            indices[offset++] = sendIndices[eqnLoc][k];
         }
      }

      MPI_Send(indices, totalLength, MPI_INT, sendProcs[i], indTag, comm);
 
      delete [] indices;
   }

   int** recvProcEqnLengths = recvProcEqns_->procEqnLengthsPtr();

   //and finally, we're ready to complete the irecvs for the indices and put
   //them away.
   for(i=0; i<numRecvProcs; i++) {
      MPI_Status status;
      int index;
      MPI_Waitany(numRecvProcs, indRequests, &index, &status);

      int offset = 0;
      for(int j=0; j<eqnsPerRecvProc[index]; j++) {
         int eqn = recvProcEqnNumbers[index][j];
         int* indices = &(recvProcEqnIndices[index][offset]);
         int len = recvProcEqnLengths[index][j];

         recvEqns_->addIndices(eqn, indices, len);

         offset += len;
      }

      delete [] recvProcEqnIndices[index];
   }

   //allocate the solnValue_ list, which is of size numRecvEqns.
   int numRecvEqns = recvEqns_->getNumEqns();
   solnValues_ = new double[numRecvEqns];

   delete [] recvProcEqnIndices;
   delete [] lenRequests;
   delete [] indRequests;

   exchangeIndicesCalled_ = true;
}

//==============================================================================
void EqnCommMgr::exchangeEqns(MPI_Comm comm) {
//
//This function performs the communication necessary to exchange remote
//equations (both indices and coefficients) among all participating processors.
//

   exchangeEqnBuffers(comm, sendProcEqns_, sendEqns_,
                      recvProcEqns_, recvEqns_);
}

//==============================================================================
void EqnCommMgr::exchangeEqnBuffers(MPI_Comm comm, ProcEqns* sendProcEqns,
                              EqnBuffer* sendEqns, ProcEqns* recvProcEqns,
                              EqnBuffer* recvEqns) {
//
//This function performs the communication necessary to exchange remote
//equations (both indices and coefficients) among all participating processors.
//
   int i, indTag = 19991130, coefTag = 19991131;

   int numRecvProcs = recvProcEqns->getNumProcs();
   int numSendProcs = sendProcEqns->getNumProcs();

   if ((numRecvProcs == 0) && (numSendProcs == 0)) return;

   if (!setNumRHSsCalled_ || !exchangeIndicesCalled_) {
      cerr << "EqnCommMgr::exchangeEqns: ERROR, don't call this until after"
           << " setNumRHSs and exchangeIndices." << endl;
      abort();
   }

   //a lot of information that we need was gathered during exchangeIndices,
   //such as recvProcEqnLengths, sendProcEqnLengths, etc.

   MPI_Request* indRequests = NULL;
   MPI_Request* coefRequests = NULL;
   int** recvProcEqnIndices = NULL;
   double** recvProcEqnCoefs = NULL;

   if (numRecvProcs > 0) {
      indRequests = new MPI_Request[numRecvProcs];
      coefRequests = new MPI_Request[numRecvProcs];
      recvProcEqnIndices = new int*[numRecvProcs];
      recvProcEqnCoefs = new double*[numRecvProcs];
   }

   int numRHSs = sendEqns->getNumRHSs();

   //now, let's allocate the space for the incoming equations.
   //each row of recvProcEqnIndices will be of length
   //sum-of-recvProcEqnLengths[i], and each row of recvProcEqnCoefs will be
   //of length sum-of-recvProcEqnLengths[i] + numRHSs*eqnsPerRecvProc[i].

   int* recvProcs = recvProcEqns->procsPtr();
   int* eqnsPerRecvProc = recvProcEqns->eqnsPerProcPtr();
   int** recvProcEqnNumbers = recvProcEqns->procEqnNumbersPtr();
   int** recvProcEqnLengths = recvProcEqns->procEqnLengthsPtr();

   for(i=0; i<numRecvProcs; i++) {
      int totalLength = 0;

      for(int j=0; j<eqnsPerRecvProc[i]; j++) {
         totalLength += recvProcEqnLengths[i][j];
      }

      recvProcEqnIndices[i] = new int[totalLength];

      int coefLength = totalLength + numRHSs*eqnsPerRecvProc[i];

      recvProcEqnCoefs[i] = new double[coefLength];

      //let's go ahead and launch the recvs for the indices and coefs now.
      MPI_Irecv(recvProcEqnIndices[i], totalLength, MPI_INT,
                recvProcs[i], indTag, comm, &indRequests[i]);

      MPI_Irecv(recvProcEqnCoefs[i], coefLength, MPI_DOUBLE,
                recvProcs[i], coefTag, comm, &coefRequests[i]);
   }

   //ok, now we need to build the lists of outgoing indices and coefs, and
   //send those.
   int* sendProcs = sendProcEqns->procsPtr();
   int* eqnsPerSendProc = sendProcEqns->eqnsPerProcPtr();
   int** sendProcEqnNumbers = sendProcEqns->procEqnNumbersPtr();
   int** sendProcEqnLengths = sendProcEqns->procEqnLengthsPtr();

   int numSendEqns = sendEqns->getNumEqns();
   int* sendEqnNumbers = sendEqns->eqnNumbersPtr();
   int** sendIndices = sendEqns->indicesPtr();
   double** sendCoefs = sendEqns->coefsPtr();
   int* sendEqnLengths = sendEqns->lengthsPtr();
   double** sendRHS = sendEqns->rhsCoefsPtr();

   for(i=0; i<numSendProcs; i++) {
      int totalLength = 0;
      int j;
      for(j=0; j<eqnsPerSendProc[i]; j++)
         totalLength += sendProcEqnLengths[i][j];

      int* indices = new int[totalLength];
      int coefLength = totalLength + numRHSs*eqnsPerSendProc[i];

      double* coefs = new double[coefLength];

      int offset = 0;

      //first pack up the coefs and indices
      for(j=0; j<eqnsPerSendProc[i]; j++) {
         int eqnLoc = sendEqns->getEqnIndex(sendProcEqnNumbers[i][j]);

         for(int k=0; k<sendProcEqnLengths[i][j]; k++) {
            indices[offset] = sendIndices[eqnLoc][k];
            coefs[offset++] = sendCoefs[eqnLoc][k];
         }
      }

      //now append the RHS coefs to the end of the coefs array
      for(j=0; j<eqnsPerSendProc[i]; j++) {
         int eqnLoc = sendEqns->getEqnIndex(sendProcEqnNumbers[i][j]);

         for(int k=0; k<numRHSs; k++) {
            coefs[offset++] = sendRHS[eqnLoc][k];
         }
      }

      MPI_Send(indices, totalLength, MPI_INT, sendProcs[i], indTag, comm);
      MPI_Send(coefs, coefLength, MPI_DOUBLE, sendProcs[i], coefTag, comm);

      delete [] indices;
      delete [] coefs;
   }

   //and finally, we're ready to complete the irecvs for the indices and coefs,
   //and put them away.
   for(i=0; i<numRecvProcs; i++) {
      MPI_Status status;
      int index;
      MPI_Waitany(numRecvProcs, indRequests, &index, &status);
      MPI_Wait(&coefRequests[index], &status);

      int j, offset = 0;
      for(j=0; j<eqnsPerRecvProc[index]; j++) {
         int eqn = recvProcEqnNumbers[index][j];
         int* indices = &(recvProcEqnIndices[index][offset]);
         double* coefs = &(recvProcEqnCoefs[index][offset]);
         int len = recvProcEqnLengths[index][j];

         recvEqns->addEqn(eqn, coefs, indices, len);

         offset += len;
      }

      delete [] recvProcEqnIndices[index];
   }

   //now unpack the RHS entries

   recvEqns->setNumRHSs(numRHSs);

   for(i=0; i<numRecvProcs; i++) {
      int j, offset = 0;
      for(j=0; j<eqnsPerRecvProc[i]; j++) {
         offset += recvProcEqnLengths[i][j];
      }

      for(j=0; j<eqnsPerRecvProc[i]; j++) {
         int eqn = recvProcEqnNumbers[i][j];

         for(int k=0; k<numRHSs; k++) {
            recvEqns->addRHS(eqn, k, recvProcEqnCoefs[i][offset++]);
         }
      }

      delete [] recvProcEqnCoefs[i];
   }

   delete [] recvProcEqnIndices;
   delete [] recvProcEqnCoefs;
   delete [] indRequests;
   delete [] coefRequests;
}

//==============================================================================
void EqnCommMgr::exchangeSoln(MPI_Comm comm) {

   int solnTag = 199906;

   MPI_Request* solnRequests = NULL;
   double** solnBuffer = NULL;

   int numSendProcs = sendProcEqns_->getNumProcs();
   int* sendProcs = sendProcEqns_->procsPtr();
   int* eqnsPerSendProc = sendProcEqns_->eqnsPerProcPtr();

   if (numSendProcs > 0) {
      solnRequests = new MPI_Request[numSendProcs];
      solnBuffer = new double*[numSendProcs];
   }

   int i;
   //let's launch the recv's for the incoming solution values.
   for(i=0; i<numSendProcs; i++) {
      solnBuffer[i] = new double[eqnsPerSendProc[i]];

      MPI_Irecv(solnBuffer[i], eqnsPerSendProc[i], MPI_DOUBLE, sendProcs[i],
                solnTag, comm, &solnRequests[i]);
   }

   int numRecvProcs = recvProcEqns_->getNumProcs();
   int* recvProcs = recvProcEqns_->procsPtr();
   int* eqnsPerRecvProc = recvProcEqns_->eqnsPerProcPtr();
   int** recvProcEqnNumbers = recvProcEqns_->procEqnNumbersPtr();

   //now let's send the outgoing solutions.
   for(i=0; i<numRecvProcs; i++) {
      double* solnBuff = new double[eqnsPerRecvProc[i]];

      for(int j=0; j<eqnsPerRecvProc[i]; j++) {
         int index = recvEqns_->getEqnIndex(recvProcEqnNumbers[i][j]);
         solnBuff[j] = solnValues_[index];
      }

      MPI_Send(solnBuff, eqnsPerRecvProc[i], MPI_DOUBLE, recvProcs[i],
               solnTag, comm);

      delete [] solnBuff;
   }

   //ok, complete the above recvs and store the soln values.
   for(i=0; i<numSendProcs; i++) {
      int index;
      MPI_Status status;
      MPI_Waitany(numSendProcs, solnRequests, &index, &status);

      for(int j=0; j<eqnsPerSendProc[index]; j++) {
         int ind = sendProcEqnLocations_[index][j];

         sendEqnSoln_[ind] = solnBuffer[index][j];
      }

      delete [] solnBuffer[index];
   }

   delete [] solnRequests;
   delete [] solnBuffer;
}

//==============================================================================
void EqnCommMgr::addSendEqn(int eqnNumber, int destProc,
                            const double* coefs, const int* indices, int num) {
   (void)destProc;
   sendEqns_->addEqn(eqnNumber, coefs, indices, num);
}

//==============================================================================
void EqnCommMgr::setNumRHSs(int numRHSs) {
   if (numRHSs <= 0) {
      cerr << "EqnCommMgr::setNumRHSs: ERROR, numRHSs <= 0." << endl;
      abort();
   }

   setNumRHSsCalled_ = true;
   numRHSs_ = numRHSs;

   sendEqns_->setNumRHSs(numRHSs);
}

//==============================================================================
void EqnCommMgr::addSendRHS(int eqnNumber, int destProc, int rhsIndex,
                            double value) {

   (void)destProc;

   sendEqns_->addRHS(eqnNumber, rhsIndex, value);
}

//==============================================================================
void EqnCommMgr::addSendIndices(int eqnNumber, int destProc,
                                int* indices, int num) {

   sendEqns_->addIndices(eqnNumber, indices, num);

   sendProcEqns_->addEqn(eqnNumber, destProc);
}

//==============================================================================
void EqnCommMgr::exchangeEssBCs(int* essEqns, int numEssEqns,double* essAlpha,
                                double* essGamma, MPI_Comm comm) {
   delete essBCEqns_;
   essBCEqns_ = new EqnBuffer();

   EqnBuffer* sendEssEqns = new EqnBuffer();
   ProcEqns* essSendProcEqns = new ProcEqns();

   int** _sendIndices = sendEqns_->indicesPtr();
   int* _sendEqnNumbers = sendEqns_->eqnNumbersPtr();
   int* _sendEqnLengths = sendEqns_->lengthsPtr();

   //check to see if any of the essEqns are in the sendIndices_ table.
   //the ones that are, will need to be sent to other processors.

   int i;
   for(i=0; i<numEssEqns; i++) {
      int index = sendEqns_->isInIndices(essEqns[i]);

      if (index >= 0) {
         int proc = getSendProcNumber(_sendEqnNumbers[index]);

         double coef = essGamma[i]/essAlpha[i];

         sendEssEqns->addEqn(_sendEqnNumbers[index], &coef, &(essEqns[i]), 1);

         essSendProcEqns->addEqn(_sendEqnNumbers[index], proc);
      }
   }

   int numSendEqns = sendEssEqns->getNumEqns();
   int numRecvEqns = getNumRecvEqns(essSendProcEqns, comm);

   if ((numRecvEqns == 0) && (numSendEqns == 0)) {
      delete sendEssEqns;
      delete essSendProcEqns;

      return;
   }

   int** recvEqnLengths = NULL;
   MPI_Request* lenRequests = NULL;
   int lenTag = 19991129;

   if (numRecvEqns > 0) {
      recvEqnLengths = new int*[numRecvEqns];
      lenRequests = new MPI_Request[numRecvEqns];
   }

   for(i=0; i<numRecvEqns; i++) {
      recvEqnLengths[i] = new int[2];
      MPI_Irecv(recvEqnLengths[i], 2, MPI_INT, MPI_ANY_SOURCE, lenTag,
                comm, &lenRequests[i]);
   }

   //now send the lengths...

   int numSendProcs = essSendProcEqns->getNumProcs();
   int* sendProcs = essSendProcEqns->procsPtr();
   int** sendProcEqnNumbers = essSendProcEqns->procEqnNumbersPtr();
   int* eqnsPerSendProc = essSendProcEqns->eqnsPerProcPtr();

   int* eqnNumbers = sendEssEqns->eqnNumbersPtr();
   int* eqnLengths = sendEssEqns->lengthsPtr();

   essSendProcEqns->setProcEqnLengths(eqnNumbers, eqnLengths, numSendEqns);

   int** sendProcEqnLengths = essSendProcEqns->procEqnLengthsPtr();

   for(i=0; i<numSendProcs; i++) {
      int proc = sendProcs[i];
      for(int j=0; j<eqnsPerSendProc[i]; j++) {
         int* msg = new int[2];
         msg[0] = sendProcEqnNumbers[i][j];
         msg[1] = sendProcEqnLengths[i][j];

         MPI_Send(msg, 2, MPI_INT, proc, lenTag, comm);
      }
   }

   ProcEqns* essRecvProcEqns = new ProcEqns();

   //now catch the Irecv's with the equation lengths.
   for(i=0; i<numRecvEqns; i++) {
      MPI_Status status;
      int index;
      MPI_Waitany(numRecvEqns, lenRequests, &index, &status);
      int proc = status.MPI_SOURCE;

      int eqn = recvEqnLengths[index][0];
      int len = recvEqnLengths[index][1];
      essRecvProcEqns->addEqn(eqn, len, proc);

      delete [] recvEqnLengths[index];
   }

   delete [] recvEqnLengths;

   delete [] lenRequests;

   exchangeEqnBuffers(comm, essSendProcEqns, sendEssEqns,
                      essRecvProcEqns, essBCEqns_);

   delete sendEssEqns;
   delete essSendProcEqns;
   delete essRecvProcEqns;
}

//==============================================================================
int EqnCommMgr::getNumRecvEqns(ProcEqns* sendProcEqns, MPI_Comm comm) {

   int numProcs = 0;
   MPI_Comm_size(comm, &numProcs);

   int* recvEqnsPerProc = new int[numProcs];
   int* eqnsPerProc = new int[numProcs];
   int i;
   for(i=0; i<numProcs; i++) {
      recvEqnsPerProc[i] = 0;
      eqnsPerProc[i] = 0;
   }

   int numSendProcs = sendProcEqns->getNumProcs();
   int* sendProcs = sendProcEqns->procsPtr();
   int* eqnsPerSendProc = sendProcEqns->eqnsPerProcPtr();

   for(i=0; i<numSendProcs; i++) {
      eqnsPerProc[sendProcs[i]] = eqnsPerSendProc[i];
   }

   MPI_Allreduce(eqnsPerProc, recvEqnsPerProc, numProcs, MPI_INT, MPI_SUM,
                 comm);

   int returnValue = recvEqnsPerProc[localProc_];

   delete [] recvEqnsPerProc;
   delete [] eqnsPerProc;

   return(returnValue);
}

//==============================================================================
int EqnCommMgr::getSendProcNumber(int eqn) {

   int numSendProcs = sendProcEqns_->getNumProcs();
   int* sendProcs = sendProcEqns_->procsPtr();
   int** sendProcEqnNumbers = sendProcEqns_->procEqnNumbersPtr();
   int* eqnsPerSendProc = sendProcEqns_->eqnsPerProcPtr();

   for(int i=0; i<numSendProcs; i++) {
      int ins;
      int index = Utils::sortedIntListFind(eqn, sendProcEqnNumbers[i],
                                           eqnsPerSendProc[i], &ins);

      if (index >= 0) return(sendProcs[i]);
   }

   return(-1);
}

