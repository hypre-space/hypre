#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "src/Utils.h"
#include "src/ProcEqns.h"

//==============================================================================
ProcEqns::ProcEqns()
 : numProcs_(0),
   procs_(NULL),
   eqnsPerProc_(NULL),
   procEqnNumbers_(NULL),
   procEqnLengths_(NULL)
{
}

//==============================================================================
ProcEqns::~ProcEqns() {
   deleteMemory();
}

//==============================================================================
void ProcEqns::deleteMemory() {

   for(int i=0; i<numProcs_; i++) {
      delete [] procEqnNumbers_[i];
      delete [] procEqnLengths_[i];
   }

   delete [] procEqnNumbers_;
   delete [] procEqnLengths_;

   delete [] procs_;
   delete [] eqnsPerProc_;
   numProcs_ = 0;
}

//==============================================================================
void ProcEqns::addEqn(int eqnNumber, int proc) {

   internalAddEqn(eqnNumber, 0, proc);
}

//==============================================================================
void ProcEqns::addEqn(int eqnNumber, int eqnLength, int proc) {
   internalAddEqn(eqnNumber, eqnLength, proc);
}

//==============================================================================
void ProcEqns::internalAddEqn(int eqnNumber, int eqnLength, int proc) {
//
//This function adds proc to the recvProcs_ list if it isn't already
//present, and adds eqnNumber to the correct row of the procEqnNumbers_
//table if eqnNumber isn't already in there.
//

   //is proc already in our list of procs?
   //if not, add it.
   int tmp = numProcs_;
   int index = Utils::sortedIntListInsert(proc, procs_, numProcs_);

   if (numProcs_ > tmp) {
      //if numProcs_ > tmp, then proc was NOT already present, so
      //we need to insert new rows in the tables procEqnNumbers_
      //and procEqnLengths.

      int* newRecvEqn = new int[1];
      newRecvEqn[0] = eqnNumber;

      //index is the position at which proc was inserted in procs_.
      Utils::intTableInsertRow(newRecvEqn, index, procEqnNumbers_, tmp);
      tmp--;

      int* newLenRow = new int[1];
      newLenRow[0] = eqnLength;
      Utils::intTableInsertRow(newLenRow, index, procEqnLengths_, tmp);
      tmp--;

      Utils::intListInsert(1, index, eqnsPerProc_, tmp);
   }
   else {
      //is eqnNumber already in our list of eqns for proc?
      //if not, add it.

      int tmpLen = eqnsPerProc_[index];
      int ind1 = Utils::sortedIntListInsert(eqnNumber, procEqnNumbers_[index],
                                       eqnsPerProc_[index]);

      if (tmpLen < eqnsPerProc_[index]) {
         Utils::intListInsert(eqnLength, ind1, procEqnLengths_[index], tmpLen);
      }
      else {
         procEqnLengths_[index][ind1] = eqnLength;
      }
   }
}

//==============================================================================
void ProcEqns::setProcEqnLength(int index1, int index2, int eqnLength) {

   procEqnLengths_[index1][index2] = eqnLength;
}

//==============================================================================
void ProcEqns::setProcEqnLengths(int* eqnNumbers, int* eqnLengths, int len) {

   if ((len == 0) && (numProcs_ > 0)) {
      cerr << "ProcEqns::setProcEqnLengths: ERROR, len == 0 but numProcs_ > 0."
         << endl;
      abort();
   }

   if (len == 0) return;

   for(int jj=0; jj<numProcs_; jj++) delete [] procEqnLengths_[jj];
   delete [] procEqnLengths_;

   procEqnLengths_ = new int*[numProcs_];

   for(int i=0; i<numProcs_; i++) {
      procEqnLengths_[i] = new int[eqnsPerProc_[i]];

      for(int j=0; j<eqnsPerProc_[i]; j++) {
         int ins = -1;
         int index = Utils::sortedIntListFind(procEqnNumbers_[i][j],
                                              eqnNumbers, len, &ins);

         if (index < 0) {
            cerr << "ProcEqns::setProcEqnLengths: ERROR, "
                 << procEqnNumbers_[i][j] << " not found." << endl;
            abort();
         }

         procEqnLengths_[i][j] = eqnLengths[index];
      }
   }
}

