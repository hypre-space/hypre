#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "src/Utils.h"
#include "src/EqnBuffer.h"

//==============================================================================
EqnBuffer::EqnBuffer()
 : numEqns_(0),
   eqnNumbers_(NULL),
   indices_(NULL),
   coefs_(NULL),
   eqnLengths_(NULL),
   numRHSs_(0),
   rhsIndices_(NULL),
   rhsCoefs_(NULL),
   setNumRHSsCalled_(false),
   rhsCoefsAllocated_(false)
{
}

//==============================================================================
EqnBuffer::~EqnBuffer() {
   deleteMemory();
}

//==============================================================================
void EqnBuffer::deleteMemory() {
   for(int i=0; i<numEqns_; i++) {
      delete [] indices_[i];
      delete [] coefs_[i];
      if (numRHSs_ > 0) delete [] rhsCoefs_[i];
   }

   delete [] indices_;
   delete [] coefs_;
   delete [] rhsCoefs_;
   delete [] rhsIndices_;
   delete [] eqnLengths_;
   delete [] eqnNumbers_;
   numRHSs_ = 0;
   numEqns_ = 0;
}

//==============================================================================
int EqnBuffer::getEqnIndex(int eqn) {

   int insertPoint = -1;
   return(
      Utils::sortedIntListFind(eqn, eqnNumbers_,
                                        numEqns_, &insertPoint)
   );

}

//==============================================================================
void EqnBuffer::setNumRHSs(int n) {

   setNumRHSsCalled_ = true;

   if (n <= 0) return;

   numRHSs_ = n;

   if (rhsIndices_ != NULL) delete [] rhsIndices_;

   rhsIndices_ = new int[numRHSs_];

   if (numEqns_ <= 0) return;

   if (rhsCoefsAllocated_) {
      for(int i=0; i<numEqns_; i++) delete [] rhsCoefs_[i];
      delete [] rhsCoefs_;
   }

   rhsCoefs_ = new double*[numEqns_];

   for(int i=0; i<numEqns_; i++) {
      rhsCoefs_[i] = new double[numRHSs_];

      for(int j=0; j<numRHSs_; j++) rhsCoefs_[i][j] = 0.0;
   }

   rhsCoefsAllocated_ = true;
}

//==============================================================================
void EqnBuffer::addRHS(int eqnNumber, int rhsIndex, double value) {
   int tmp;
   int index = Utils::sortedIntListFind(eqnNumber, eqnNumbers_,
                                        numEqns_, &tmp);

   if (index < 0) {
      cerr << "EqnBuffer::addRHS: ERROR, eqnNumber " << eqnNumber
           << " not found in send eqns." << endl;
      abort();
   }

   rhsCoefs_[index][rhsIndex] += value;
}

//==============================================================================
int EqnBuffer::isInIndices(int eqn) {
//
//This function checks the indices_ table to see if 'eqn' is present.
//If it is, the appropriate row index into the table is returned.
//-1 is return otherwise.
//
   for(int i=0; i<numEqns_; i++) {
      int ins;
      int index = Utils::sortedIntListFind(eqn, indices_[i], eqnLengths_[i],
                                           &ins);
      if (index >= 0) return(i);
   }

   return(-1);
}

//==============================================================================
void EqnBuffer::addEqn(int eqnNumber, const double* coefs, const int* indices,
                       int len, bool accumulate) {
   int insertPoint = -1;
   int index = Utils::sortedIntListFind(eqnNumber, eqnNumbers_,
                                        numEqns_, &insertPoint);

   //is eqnNumber present in our list of equation numbers?

   if (index >= 0) {
      //if so, insert the index/coef pairs, or add them to ones already in
      //place, as appropriate.

      internalAddEqn(index, coefs, indices, len, accumulate);
   }
   else {
      //if eqnNumber was not already present, add this equation number to
      //corresponding rows (row 'insertPoint') in coefs_ and indices_.

      int tmp = numEqns_;
      Utils::intListInsert(eqnNumber, insertPoint, eqnNumbers_,
                           numEqns_);

      double* newCoefRow = new double[1];
      int* newIndexRow = new int[1];
      newCoefRow[0] = coefs[0];
      newIndexRow[0] = indices[0];

      Utils::doubleTableInsertRow(newCoefRow, insertPoint, coefs_, tmp);
      tmp--;
      Utils::intTableInsertRow(newIndexRow, insertPoint, indices_, tmp);
      tmp--;

      Utils::intListInsert(1, insertPoint, eqnLengths_, tmp);

      internalAddEqn(insertPoint, &(coefs[1]), &(indices[1]), len-1,
                     accumulate);
   }
}

//==============================================================================
void EqnBuffer::internalAddEqn(int index, const double* coefs,
                               const int* indices, int len, bool accumulate) {
//
//Private EqnBuffer function. We can safely assume that this function is only
//called if indices_ and coefs_ already contain an 'index'th row.
//

   double*& coefRow = coefs_[index];
   int*& indexRow = indices_[index];

   for(int i=0; i<len; i++) {
      int tmp = eqnLengths_[index];
      int position = Utils::sortedIntListInsert(indices[i], indexRow,
                                                eqnLengths_[index]);

      if (eqnLengths_[index] > tmp) {
         //indices[i] wasn't already there, so just insert this coef.

         Utils::doubleListInsert(coefs[i], position, coefRow, tmp);
      }
      else {
         //indices[i] was already there so sum in this coef.

         if (accumulate) coefRow[position] += coefs[i];
         else coefRow[position] = coefs[i];
      }
   }
}

//==============================================================================
void EqnBuffer::resetCoefs() {
   for(int i=0; i<numEqns_; i++) {
      for(int j=0; j<eqnLengths_[i]; j++) {
         coefs_[i][j] = 0.0;
      }
      for(int k=0; k<numRHSs_; k++) {
         rhsCoefs_[i][k] = 0.0;
      }
   }
}

//==============================================================================
void EqnBuffer::addIndices(int eqnNumber, const int* indices, int len) {

   int insertPoint = -1;
   int index = Utils::sortedIntListFind(eqnNumber, eqnNumbers_,
                                        numEqns_, &insertPoint);

   //(we're adding a row to coefs_ as well, even though there are no
   //incoming coefs at this point).

   double* dummyCoefs = new double[len];
   for(int i=0; i<len; i++) {
      dummyCoefs[i] = 0.0;
   }

   //is eqnNumber already present?

   if (index >= 0) {
      //if so, insert the indices, or add them to ones already in
      //place, as appropriate.

      bool accumulate = true;
      internalAddEqn(index, dummyCoefs, indices, len, accumulate);
   }
   else {
      //if eqnNumber was not already present, insert this equation number at 
      //appropriate rows (row 'insertPoint') in indices_ and coefs_.

      int tmp = numEqns_;
      Utils::intListInsert(eqnNumber, insertPoint, eqnNumbers_,
                           numEqns_);

      int* newIndexRow = new int[1];
      newIndexRow[0] = indices[0];

      Utils::intTableInsertRow(newIndexRow, insertPoint, indices_, tmp);
      tmp--;

      double* newCoefRow = new double[1];
      newCoefRow[0] = 0.0;

      Utils::doubleTableInsertRow(newCoefRow, insertPoint, coefs_, tmp);
      tmp--;

      Utils::intListInsert(1, insertPoint, eqnLengths_, tmp);

      bool accumulate = true;
      internalAddEqn(insertPoint, &(dummyCoefs[1]), &(indices[1]),
                     len-1, accumulate);
   }

   delete [] dummyCoefs;
}

