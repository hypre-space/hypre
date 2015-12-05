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

   //void parameters:
   //for setting generic argc/argv style parameters.

   virtual void parameters(int numParams, char** params) = 0;

   virtual void setLookup(Lookup& lookup);

   virtual void setGlobalOffsets(int len, int* nodeOffsets, int* eqnOffsets,
                                 int* blkEqnOffsets);

   virtual void setConnectivities(GlobalID elemBlock,
                                  int numElements,
                                  int numNodesPerElem,
                                  const GlobalID* elemIDs,
                                  const int* const* connNodes) ;

   virtual void setStiffnessMatrices(GlobalID elemBlock,
                                     int numElems,
                                     const GlobalID* elemIDs,
                                     const double *const *const *stiff,
                                     int numEqnsPerElem,
                                     const int *const * eqnIndices);

   virtual void setLoadVectors(GlobalID elemBlock,
                               int numElems,
                               const GlobalID* elemIDs,
                               const double *const * load,
                               int numEqnsPerElem,
                               const int *const * eqnIndices);

   virtual void setMatrixStructure(int** ptColIndices,
                                   int* ptRowLengths,
                                   int** blkColIndices,
                                   int* blkRowLengths,
                                   int* ptRowsPerBlkRow);

   virtual void setMultCREqns(int multCRSetID,
                              int numCRs, int numNodesPerCR,
                              int** nodeNumbers, int** eqnNumbers,
                              int* fieldIDs,
                              int* multiplierEqnNumbers);

   virtual void setPenCREqns(int penCRSetID,
                              int numCRs, int numNodesPerCR,
                              int** nodeNumbers, int** eqnNumbers,
                              int* fieldIDs);

   virtual void sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                                    int numPtCols, const int* ptCols,
                                    int numBlkRows, const int* blkRows,
                                    int numBlkCols, const int* blkCols,
                                    const double* const* values);

   virtual void sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                                    int numPtCols, const int* ptCols,
                                    const double* const* values);

   virtual void sumIntoRHSVector(int num, const double* values,
                                 const int* indices) = 0;

   virtual void matrixLoadComplete() = 0;

   //for providing nodal data associated with a particular fieldID.
   //nodeNumbers is a list of length numNodes.
   //offsets is a list of length numNodes+1.
   //data contains the incoming data. data for the ith node lies in the
   //locations data[offsets[i]] ... data[offsets[i+1] -1 ]

   virtual void putNodalFieldData(int fieldID, int fieldSize,
                                  int* nodeNumbers, int numNodes,
                                  const double* data);

   
   virtual void resetMatrixAndVector(double s) = 0;
   virtual void resetMatrix(double s) = 0;
   virtual void resetRHSVector(double s) = 0;

   //functions for enforcing boundary conditions.
   virtual void enforceEssentialBC(int* globalEqn, double* alpha,
                                   double* gamma, int len);

   virtual void enforceRemoteEssBCs(int numEqns, int* globalEqns,
                                          int** colIndices, int* colIndLen,
                                          double** coefs);

   virtual void enforceOtherBC(int* globalEqn, double* alpha,
                               double* beta, double* gamma, int len);

   //functions for getting/setting matrix or rhs vector(s), or pointers to them.

   virtual void getMatrixPtr(Data& data) = 0;

   virtual void copyInMatrix(double scalar, const Data& data) = 0;
   virtual void copyOutMatrix(double scalar, Data& data) = 0;
   virtual void sumInMatrix(double scalar, const Data& data) = 0;

   virtual void getRHSVectorPtr(Data& data) = 0;

   virtual void copyInRHSVector(double scalar, const Data& data) = 0;
   virtual void copyOutRHSVector(double scalar, Data& data) = 0;
   virtual void sumInRHSVector(double scalar, const Data& data) = 0;

   //Utility functions for destroying matrix or vector in a Data container.
   //The caller (owner of 'data') can't destroy the matrix because they don't
   //know what type it is and can't get to its destructor.

   virtual void destroyMatrixData(Data& data) = 0;
   virtual void destroyVectorData(Data& data) = 0;

   //functions for managing multiple rhs vectors
   virtual void setNumRHSVectors(int numRHSs, const int* rhsIDs) = 0;
   virtual void setRHSID(int rhsID) = 0;

   //functions for getting/setting the contents of the solution vector.

   virtual void putInitialGuess(const int* eqnNumbers, const double* values,
                                int len) = 0;

   virtual void getSolution(double* answers, int len) = 0;

   virtual void getSolnEntry(int eqnNumber, double& answer) = 0;

   virtual void formResidual(double* values, int len) = 0;

   //function for launching the linear solver
   virtual void launchSolver(int& solveStatus, int& iterations) = 0;

   virtual void writeSystem(const char* name) = 0;

 private:
   void LSCmessageAbort(const char* name);
};

#endif

