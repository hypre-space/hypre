#include <stdlib.h>
#include <iostream.h>
#include <assert.h>
#include "other/basicTypes.h"

#include "src/BCRecord.h"

//==========================================================================
BCRecord::BCRecord()
 : nodeID_((GlobalID)-1),
   fieldID_(0),
   fieldSize_(0),
   alpha_(NULL),
   beta_(NULL),
   gamma_(NULL)
{
}

//==========================================================================
BCRecord::~BCRecord() {

   if (fieldSize_ > 0){
      delete [] alpha_;
      delete [] beta_;
      delete [] gamma_;
      fieldSize_ = 0;
   }
}

//==========================================================================
void BCRecord::setAlpha(const double* alpha) {

   setDoubleList(alpha_, alpha, fieldSize_);
}
 
//==========================================================================
void BCRecord::setBeta(const double* beta) {

   setDoubleList(beta_, beta, fieldSize_);
}
 
//==========================================================================
void BCRecord::setGamma(const double* gamma) {

   setDoubleList(gamma_, gamma, fieldSize_);
}
 
//==============================================================================
void BCRecord::setDoubleList(double*& list, const double* input, int len) {
   if (list != NULL) delete [] list;

   if (len <= 0) {
      cerr << "BCRecord::setDoubleList: ERROR, fieldSize not set yet." << endl;
      abort();
   }

   list = new double[len];

   for(int i=0; i<len; i++) list[i] = input[i];
}

