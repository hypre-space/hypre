
#ifndef _LSC_h_
#define _LSC_h_

//
//Files that need to be included before the compiler
//reaches this header:
//
//#include "src/Data.h"
//#include <mpi.h>

class LSC : public LinearSystemCore {
 public:
   LSC(){};
   virtual ~LSC() {};

   //for cloning a LSC instance.
   virtual LinearSystemCore* clone() = 0;

   //int parameters:
   //for setting generic argc/argv style parameters.

   virtual int parameters(int numParams, char** params) = 0;

   virtual int setLookup(Lookup& lookup);

   virtual int setConnectivities(GlobalID elemBlock,
                                  int numElements,
                                  int numNodesPerElem,
                                  const GlobalID* elemIDs,
                                  const int* const* connNodes) ;

   virtual int setStiffnessMatrices(GlobalID elemBlock,
                                     int numElems,
                                     const GlobalID* elemIDs,
                                     const double *const *const *stiff,
                                     int numEqnsPerElem,
                                     const int *const * eqnIndices);

   virtual int setLoadVectors(GlobalID elemBlock,
                               int numElems,
                               const GlobalID* elemIDs,
                               const double *const * load,
                               int numEqnsPerElem,
                               const int *const * eqnIndices);

   virtual int setMultCREqns(int multCRSetID,
                              int numCRs, int numNodesPerCR,
                              int** nodeNumbers, int** eqnNumbers,
                              int* fieldIDs,
                              int* multiplierEqnNumbers);

   virtual int setPenCREqns(int penCRSetID,
                              int numCRs, int numNodesPerCR,
                              int** nodeNumbers, int** eqnNumbers,
                              int* fieldIDs);

   //for providing nodal data associated with a particular fieldID.
   //nodeNumbers is a list of length numNodes.
   //offsets is a list of length numNodes+1.
   //data contains the incoming data. data for the ith node lies in the
   //locations data[offsets[i]] ... data[offsets[i+1] -1 ]

   virtual int putNodalFieldData(int fieldID, int fieldSize,
                                  int* nodeNumbers, int numNodes,
                                  const double* data);

 private:
   int LSCmessageAbort(const char* name);
};

#endif

