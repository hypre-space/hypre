#ifndef __BCRecord_H
#define __BCRecord_H

//requires:
//#include <stdio.h>
//#include "basicTypes.h"

//  BCRecord is one item in a linked list of boundary-condition information
//  it's implemented using a linked list because we don't have any a priori
//  information about how long this list will become, courtesy of the 
//  interaction of v1.0 extensions to the FEI spec (permitting multiple 
//  fields per node, which makes it possible to specify multiple independent
//  BC's at each node), and ALE3D needs to be able to process multiple
//  natural BC's at each node for any given field.
//
//  in this case, the data structure needs to be simple but extensible, 
//  and since we don't know how many of these records will be encountered
//  (if any!), this looks like a good way to go...

class BCRecord
{
  public:
      BCRecord();
      ~BCRecord();

      int getFieldID();                         // return field ID
      void setFieldID(int fieldID);             // set the field ID
      int getFieldSize();                       // return field cardinality
      void setFieldSize(int fieldSize);         // set the field cardinality
      int getFieldOffset();                     // return field offset
      void setFieldOffset(int fieldOffset);     // set the field offset
      void allocateAlpha();                     // create the soln coefficients
      double *pointerToAlpha(int& fieldSize);   // return the soln coefficients
      void allocateBeta();                      // create the dual coefficients
      double *pointerToBeta(int& fieldSize);    // return the dual coefficients
      void allocateGamma();                     // create the rhs coefficients
      double *pointerToGamma(int& fieldSize);   // return the rhs values
      
      void dumpToScreen();                 // debug output

  private:   
      int myFieldID;                       // cached field ID
      int myFieldSize;                     // cached field cardinality
      int myFieldOffset;                   // cached field solution offset
      double *myAlpha;                     // cached soln coefficients
      double *myBeta;                      // cached dual coefficients
      double *myGamma;                     // cached rhs terms
};
 
#endif
