#ifndef _ISIS_LinSysCore_h_
#define _ISIS_LinSysCore_h_

//
//This is the ISIS++ implementation of LinearSystemCore.
//

//Files that need to be included before the compiler
//reaches this header:
//
//#include "src/Data.h"
//#include <mpi.h>
//#include "src/LinearSystemCore.h"
//#include <isis-mpi.h>

class ISIS_LinSysCore: public LinearSystemCore {
 public:
   ISIS_LinSysCore(MPI_Comm comm);
   virtual ~ISIS_LinSysCore();

   //for creating another one, without knowing the run-time type
   //of 'this' one.
   LinearSystemCore* clone();

   //void parameters:
   //for setting generic argc/argv style parameters.

   void parameters(int numParams, char** params);

   //void createMatricesVectors:
   //provide info for initial creation of matrix/vector data,
   //Equation numbers are 1-based, and local sets of equation numbers
   //are contiguous.

   void createMatricesAndVectors(int numGlobalEqns, 
                              int firstLocalEqn,
                              int numLocalEqns);

   //void allocateMatrix:
   //provide enough info to allocate the matrix -- i.e., define
   //the structure.

   void allocateMatrix(int** colIndices, int* rowLengths);

   //void resetMatrixAndVector:
   //don't destroy the structure of the matrix, but set the value 's'
   //throughout the matrix and vectors.

   void resetMatrixAndVector(double s);

   //void sumIntoSystemMatrix:
   //this is the primary assembly function. The coefficients 'values'
   //are to be accumumlated into (added to any values already in place)
   //global (1-based) equation 'row' of the matrix.

   void sumIntoSystemMatrix(int row, int numValues,
                            const double* values,
                            const int* scatterIndices);

   //void sumIntoRHSVector:
   //this is the rhs vector equivalent to sumIntoSystemMatrix above.

   void sumIntoRHSVector(int num,
                          const double* values,
                          const int* indices);

   //void matrixLoadComplete:
   //do any internal synchronization/communication.

   void matrixLoadComplete();
   
   //functions for enforcing boundary conditions.
   void enforceEssentialBC(int* globalEqn,
                           double* alpha,
                           double* gamma, int len);

   void enforceRemoteEssBCs(int numEqns, int* globalEqns,
                                          int** colIndices, int* colIndLen,
                                          double** coefs);

   void enforceOtherBC(int* globalEqn, double* alpha,
                       double* beta, double* gamma,
                       int len);

   //functions for getting/setting matrix or vector pointers.

   //getMatrixPtr:
   //obtain a pointer to the 'A' matrix. This should be considered a
   //constant pointer -- i.e., this class remains responsible for the
   //matrix (e.g., de-allocation upon destruction). 
   void getMatrixPtr(Data& data);

   //copyInMatrix:
   //replaces the internal matrix with a copy of the input argument, scaled
   //by the coefficient 'scalar'.

   void copyInMatrix(double scalar, const Data& data);

   //copyOutMatrix:
   //passes out a copy of the internal matrix, scaled by the coefficient
   //'scalar'.

   void copyOutMatrix(double scalar, Data& data);

   //sumInMatrix:
   //accumulate (sum) a copy of the input argument into the internal
   //matrix, scaling the input by the coefficient 'scalar'.

   void sumInMatrix(double scalar, const Data& data);

   //get/setRHSVectorPtr:
   //the same semantics apply here as for the matrixPtr functions above.

   void getRHSVectorPtr(Data& data);

   //copyInRHSVector/copyOutRHSVector/sumInRHSVector:
   //the same semantics apply here as for the matrix functions above.

   void copyInRHSVector(double scalar, const Data& data);
   void copyOutRHSVector(double scalar, Data& data);
   void sumInRHSVector(double scalar, const Data& data);

   //destroyMatrixData/destroyVectorData:
   //Utility function for destroying the matrix (or vector) in Data

   void destroyMatrixData(Data& data);
   void destroyVectorData(Data& data);

   //functions for managing multiple rhs vectors
   void setNumRHSVectors(int numRHSs, const int* rhsIDs);

   //void setRHSID:
   //set the 'current' rhs context, assuming multiple rhs vectors.
   void setRHSID(int rhsID);

   //void putInitialGuess:
   //function for setting (a subset of) the initial-guess
   //solution values (i.e., in the 'x' vector).

   void putInitialGuess(const int* eqnNumbers, const double* values,
                        int len);

   //function for getting all of the answers ('x' vector).
   void getSolution(int* eqnNumbers, double* answers, int len);

   //function for getting the (single) entry at equation
   //number 'eqnNumber'.
   void getSolnEntry(int eqnNumber, double& answer);

   //function for launching the linear solver
   void launchSolver(int& solveStatus, int& iterations);

 private:        //functions

   //functions for selecting solver/preconditioner
   void selectSolver(char* name);
   void selectPreconditioner(char* name);

   void setDebugOutput(char* path, char* name);

   void debugOutput(char* mesg) const;

   void messageAbort(char* msg) const;

 private:            //variables

   MPI_Comm comm_;

   int numProcs_;
   int thisProc_;
   int masterProc_;

   CommInfo *commInfo_;
   Map *map_;
   DCRS_Matrix *A_;
   DCRS_Matrix *A_ptr_;
   Dist_Vector *x_, **b_;
   Dist_Vector *b_ptr_;
   bool matricesVectorsCreated_;

   int* rhsIDs_;
   int numRHSs_;

   int currentRHS_;

   int localStartRow_;
   int numLocalRows_;
   int localEndRow_;

   IterativeSolver *pSolver_;
   char* solverName_;
   bool solverAllocated_;
   Preconditioner *pPrecond_;
   char* precondName_;
   bool precondAllocated_;

   bool rowScale_;
   bool colScale_;
   int solveCounter_;

   int outputLevel_;
   int numParams_;
   char** paramStrings_;

   int debugOutput_;
   int debugFileCounter_;
   char* debugPath_;
   char* debugFileName_;
   FILE* debugFile_;
};

#endif

