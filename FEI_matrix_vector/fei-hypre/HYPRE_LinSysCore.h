#ifndef _HYPRE_LinSysCore_h_
#define _HYPRE_LinSysCore_h_

//
//This is the HYPRE implementation of LinearSystemCore.
//

#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream.h>
#include <math.h>
#include "HYPRE.h"
#include "IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"

// local enumerations

enum HYsolverID {HYPCG,HYGMRES,HYSUPERLU,HYSUPERLUX,HYY12M};
enum HYpreconID {HYDIAGONAL,HYPILUT,HYPARASAILS,HYBOOMERAMG,HYNONE,HYML};

class HYPRE_LinSysCore: public LinearSystemCore {
 public:
   HYPRE_LinSysCore(MPI_Comm comm);
   virtual ~HYPRE_LinSysCore();

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

   void  writeSystem(char *);

   void  loadConstraintNumbers(int, int*);
   void  buildReducedSystem();
   void  buildSchurSystem();

 private:        //functions

   //functions for selecting solver/preconditioner
   void selectSolver(char* name);
   void selectPreconditioner(char* name);

   // not implemented in HYPRE
   //void setDebugOutput(char* path, char* name);
   //void debugOutput(char* mesg) const;
   //void messageAbort(char* msg) const;

   void  solveUsingSuperLU(int&);
   void  solveUsingSuperLUX(int&);
   void  solveUsingY12M(int&);
   int   getMatrixCSR(int nrows, int nnz, int *ia, int *ja, double *val);
   int   HYFEI_BinarySearch(int*, int, int);
   void  HYFEI_Get_IJAMatrixFromFile(double **val, int **ia, int **ja, 
                  int *N, double **rhs, char *matfile, char *rhsfile);

 private:            //variables

   MPI_Comm        comm_;

   int             numProcs_;
   int             mypid_;
   int             HYOutputLevel_;

   HYPRE_IJMatrix  HYA_;
   HYPRE_IJVector  HYb_;
   HYPRE_IJVector  *HYbs_;
   HYPRE_IJVector  HYx_;
   HYPRE_IJVector  HYr_;

   HYPRE_IJMatrix  reducedA_;
   HYPRE_IJVector  reducedB_;
   HYPRE_IJVector  reducedX_;
   HYPRE_IJVector  reducedR_;
   HYPRE_IJMatrix  HYA21_;
   int             reducedAStartRow_;
   HYPRE_IJMatrix  HYinvA22_;

   HYPRE_IJMatrix  currA_;
   HYPRE_IJVector  currB_;
   HYPRE_IJVector  currX_;
   HYPRE_IJVector  currR_;

   int             matrixVectorsCreated_;
   int             systemAssembled_;
   int             systemReduced_;
   int             *selectedList_;
   int             *selectedListAux_;

   int             numGlobalRows_;
   int             localStartRow_;
   int             localEndRow_;
   int             *rowLengths_;
   int             **colIndices_;
   double          **colValues_;

   int             nConstraints_;
   int             *constrList_;

   int             *rhsIDs_;
   int             numRHSs_;
   int             currentRHS_;

   char            *HYSolverName_;
   HYPRE_Solver    HYSolver_;
   HYsolverID      HYSolverID_;

   char            *HYPreconName_;
   HYPRE_Solver    HYPrecon_;
   HYpreconID      HYPreconID_;
   int             HYPreconReuse_;

   int             maxIterations_;
   int             finalResNorm_;
   double          tolerance_;
   int             normAbsRel_;

   int             gmresDim_;
   int             amgCoarsenType_;
   int             amgNumSweeps_[4];
   int             amgRelaxType_[4];
   double          amgRelaxWeight_[25];
   double          amgStrongThreshold_;
   int             pilutRowSize_;
   double          pilutDropTol_;
   int             pilutMaxNnzPerRow_;
   int             parasailsNlevels_;
   double          parasailsThreshold_;
   double          parasailsFilter_;
   int             mlNumPreSweeps_;
   int             mlNumPostSweeps_;
   int             mlPresmootherType_;
   int             mlPostsmootherType_;
   double          mlRelaxWeight_;
   double          mlStrongThreshold_;
   int             superluOrdering_;
   char            superluScale_[1];

friend void fei_hypre_test(int argc, char *argv[]);
friend void fei_hypre_dd(int argc, char *argv[]);

};

#endif

