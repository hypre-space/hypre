#ifndef __BCRecord_H
#define __BCRecord_H

//requires:
//#include <iostream.h>
//#include <assert.h>
//#include "src/basicTypes.h"

//  BCRecord is a boundary condition specification for one node.

class BCRecord {
 public:
   BCRecord();
   virtual ~BCRecord();

   GlobalID getNodeID() {return(nodeID_);};
   void setNodeID(GlobalID nID) {nodeID_ = nID;};
 
   int getFieldID() {return(fieldID_);};
   void setFieldID(int fID) {fieldID_ = fID;};
   int getFieldSize() {return(fieldSize_);};
   void setFieldSize(int fSize) {fieldSize_ = fSize;};

   void setAlpha(const double* alpha);         // copy in the soln coefficients
   double *pointerToAlpha() {return(alpha_);};
   void setBeta(const double* beta);           // copy in the dual coefficients
   double *pointerToBeta() {return(beta_);};
   void setGamma(const double* gamma);         // copy in the rhs coefficients
   double *pointerToGamma() {return(gamma_);};

 private:   
   void setDoubleList(double*& list, const double* input, int len);

   GlobalID nodeID_;

   int fieldID_;                       // cached field ID
   int fieldSize_;                     // cached field cardinality
   double *alpha_;                     // cached soln coefficients
   double *beta_;                      // cached dual coefficients
   double *gamma_;                     // cached rhs terms
};
 
#endif
