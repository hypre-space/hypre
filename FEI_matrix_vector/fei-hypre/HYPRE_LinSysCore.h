// *************************************************************************
// This is the HYPRE implementation of LinearSystemCore.
// *************************************************************************

#ifndef _HYPRE_LinSysCore_h_
#define _HYPRE_LinSysCore_h_

// *************************************************************************
// system libraries used
// -------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// *************************************************************************
// HYPRE libraries used
// -------------------------------------------------------------------------

//#include "HYPRE.h"
//#include "../../IJ_matrix_vector/HYPRE_IJ_mv.h"
//#include "../../parcsr_matrix_vector/HYPRE_parcsr_mv.h"
//#include "../../parcsr_linear_solvers/HYPRE_parcsr_ls.h"

// *************************************************************************
// local enumerations and defines
// -------------------------------------------------------------------------

enum HYsolverID {HYPCG,HYGMRES,HYSUPERLU,HYSUPERLUX,HYY12M};
enum HYpreconID {HYDIAGONAL,HYPILUT,HYPARASAILS,HYBOOMERAMG,HYNONE,HYML,
                 HYDDILUT};

#define HYFEI_HIGHMASK      2147483647-255
#define HYFEI_SPECIALMASK          255
#define HYFEI_SLIDEREDUCE1         256
#define HYFEI_SLIDEREDUCE2         512
#define HYFEI_SLIDEREDUCE3        1024
#define HYFEI_PRINTMAT            2048
#define HYFEI_PRINTREDMAT         4096
#define HYFEI_PRINTSOL            8192
#define HYFEI_DDILUT             16384

// *************************************************************************
// class definition
// -------------------------------------------------------------------------

class HYPRE_LinSysCore

#ifndef NOFEI
           : public LinearSystemCore 
#endif
{
 public:
   HYPRE_LinSysCore(MPI_Comm comm);
   virtual ~HYPRE_LinSysCore();

   // ----------------------------------------------------------------------
   // for creating another one, w/o knowing the run-time type of 'this' one.
   // ----------------------------------------------------------------------

#ifndef NOFEI
   LinearSystemCore* clone();
#endif

   // ----------------------------------------------------------------------
   //void parameters: for setting generic argc/argv style parameters.
   // ----------------------------------------------------------------------

   void parameters(int numParams, char** params);

   // ----------------------------------------------------------------------
   // void createMatricesVectors: provide info for initial creation of 
   //      matrix/vector data, Equation numbers are 1-based, and local sets 
   //      of equation numbers are contiguous.
   // ----------------------------------------------------------------------

   void createMatricesAndVectors(int numGlobalEqns, int firstLocalEqn,
                                 int numLocalEqns);

   // ----------------------------------------------------------------------
   // void allocateMatrix: provide enough info to allocate the matrix -- 
   //                      i.e., define the structure.
   // ----------------------------------------------------------------------

   void allocateMatrix(int** colIndices, int* rowLengths);

   // ----------------------------------------------------------------------
   // void resetMatrixAndVector: don't destroy the structure of the matrix, 
   //      but set the value 's' throughout the matrix and vectors.
   // ----------------------------------------------------------------------

   void resetMatrixAndVector(double s);

   // ----------------------------------------------------------------------
   // void sumIntoSystemMatrix:
   // this is the primary assembly function. The coefficients 'values'
   // are to be accumumlated into (added to any values already in place)
   // global (1-based) equation 'row' of the matrix.
   // ----------------------------------------------------------------------

   void sumIntoSystemMatrix(int row, int numValues, const double* values,
                            const int* scatterIndices);

   // ----------------------------------------------------------------------
   // void sumIntoRHSVector:
   // this is the rhs vector equivalent to sumIntoSystemMatrix above.
   // ----------------------------------------------------------------------

   void sumIntoRHSVector(int num, const double* values, const int* indices);

   // ----------------------------------------------------------------------
   // void matrixLoadComplete:
   // do any internal synchronization/communication.
   // ----------------------------------------------------------------------

   void matrixLoadComplete();
   
   // ----------------------------------------------------------------------
   // functions for enforcing boundary conditions.
   // ----------------------------------------------------------------------

   void enforceEssentialBC(int* globalEqn,double* alpha,double* gamma,int len);

   void enforceRemoteEssBCs(int numEqns, int* globalEqns, int** colIndices, 
                            int* colIndLen, double** coefs);

   void enforceOtherBC(int* globalEqn, double* alpha, double* beta, 
                       double* gamma, int len);

   // ----------------------------------------------------------------------
   //functions for getting/setting matrix or vector pointers.
   // ----------------------------------------------------------------------

   // ----------------------------------------------------------------------
   // getMatrixPtr:
   // obtain a pointer to the 'A' matrix. This should be considered a
   // constant pointer -- i.e., this class remains responsible for the
   // matrix (e.g., de-allocation upon destruction). 
   // ----------------------------------------------------------------------

#ifndef NOFEI
   void getMatrixPtr(Data& data);
#endif

   // ----------------------------------------------------------------------
   // copyInMatrix:
   // replaces the internal matrix with a copy of the input argument, scaled
   // by the coefficient 'scalar'.
   // ----------------------------------------------------------------------

#ifndef NOFEI
   void copyInMatrix(double scalar, const Data& data);
#endif

   // ----------------------------------------------------------------------
   // copyOutMatrix:
   // passes out a copy of the internal matrix, scaled by the coefficient
   // 'scalar'.
   // ----------------------------------------------------------------------

#ifndef NOFEI
   void copyOutMatrix(double scalar, Data& data);
#endif

   // ----------------------------------------------------------------------
   // sumInMatrix:
   // accumulate (sum) a copy of the input argument into the internal
   // matrix, scaling the input by the coefficient 'scalar'.
   // ----------------------------------------------------------------------

#ifndef NOFEI
   void sumInMatrix(double scalar, const Data& data);
#endif 

   // ----------------------------------------------------------------------
   // get/setRHSVectorPtr:
   // the same semantics apply here as for the matrixPtr functions above.
   // ----------------------------------------------------------------------

#ifndef NOFEI
   void getRHSVectorPtr(Data& data);
#endif 

   // ----------------------------------------------------------------------
   // copyInRHSVector/copyOutRHSVector/sumInRHSVector:
   // the same semantics apply here as for the matrix functions above.
   // ----------------------------------------------------------------------

#ifndef NOFEI
   void copyInRHSVector(double scalar, const Data& data);
   void copyOutRHSVector(double scalar, Data& data);
   void sumInRHSVector(double scalar, const Data& data);
#endif 

   // ----------------------------------------------------------------------
   // destroyMatrixData/destroyVectorData:
   // Utility function for destroying the matrix (or vector) in Data
   // ----------------------------------------------------------------------

#ifndef NOFEI
   void destroyMatrixData(Data& data);
   void destroyVectorData(Data& data);
#endif 

   // ----------------------------------------------------------------------
   // functions for managing multiple rhs vectors
   // ----------------------------------------------------------------------

   void setNumRHSVectors(int numRHSs, const int* rhsIDs);

   // ----------------------------------------------------------------------
   // void setRHSID:
   // set the 'current' rhs context, assuming multiple rhs vectors.
   // ----------------------------------------------------------------------

   void setRHSID(int rhsID);

   // ----------------------------------------------------------------------
   // void putInitialGuess:
   // function for setting (a subset of) the initial-guess
   // solution values (i.e., in the 'x' vector).
   // ----------------------------------------------------------------------

   void putInitialGuess(const int* eqnNumbers, const double* values,int len);

   // ----------------------------------------------------------------------
   // function for getting all of the answers ('x' vector).
   // ----------------------------------------------------------------------

   void getSolution(int* eqnNumbers, double* answers, int len);

   // ----------------------------------------------------------------------
   //function for getting the (single) entry at equation number 'eqnNumber'.
   // ----------------------------------------------------------------------

   void getSolnEntry(int eqnNumber, double& answer);

   // ----------------------------------------------------------------------
   // fetch the residual vector (FEI 1.4.x compatible)
   // ----------------------------------------------------------------------

   void formResidual(int* eqnNumbers, double* values, int len);

   // ----------------------------------------------------------------------
   //function for launching the linear solver
   // ----------------------------------------------------------------------

   void launchSolver(int& solveStatus, int& iterations);

   // ----------------------------------------------------------------------
   // other functions
   // ----------------------------------------------------------------------

#ifdef FEI_V13
   void  writeSystem(char *);
#else
   void  writeSystem(const char *);
#endif

   // ----------------------------------------------------------------------
   // HYPRE-specific public functions
   // ----------------------------------------------------------------------

   void  loadConstraintNumbers(int length, int *list);
   void  buildSlideReducedSystem();
   void  buildSlideReducedSystem2();
   void  buildSlideReducedSoln();
   void  buildSlideReducedSoln2();
   void  buildSchurReducedSystem();
   void  buildSchurReducedSoln();
   void  getVersion(char**);
   void  createMapFromSoln();

 private:        //functions

   // ----------------------------------------------------------------------
   //functions for selecting solver/preconditioner
   // ----------------------------------------------------------------------

   void selectSolver(char* name);
   void selectPreconditioner(char* name);

   // ----------------------------------------------------------------------
   // not implemented in HYPRE
   //void setDebugOutput(char* path, char* name);
   //void debugOutput(char* mesg) const;
   //void messageAbort(char* msg) const;
   // ----------------------------------------------------------------------

   // ----------------------------------------------------------------------
   // HYPRE specific private functions
   // ----------------------------------------------------------------------

   void  solveUsingSuperLU(int&);
   void  solveUsingSuperLUX(int&);
   void  solveUsingY12M(int&);

 private:        //variables

   // ----------------------------------------------------------------------
   // parallel communication information and output levels
   // ----------------------------------------------------------------------

   MPI_Comm        comm_;               // MPI communicator
   int             numProcs_;           // number of processors
   int             mypid_;              // my processor ID
   int             HYOutputLevel_;      // control print information

   // ----------------------------------------------------------------------
   // matrix and vectors
   // ----------------------------------------------------------------------

   HYPRE_IJMatrix  HYA_;                // the system matrix
   HYPRE_IJVector  HYb_;                // the current RHS 
   HYPRE_IJVector  *HYbs_;              // an array of RHSs
   HYPRE_IJVector  HYx_;                // the solution vector
   HYPRE_IJVector  HYr_;                // temporary vector for residual
   int             numGlobalRows_;
   int             localStartRow_;
   int             localEndRow_;
   int             *rowLengths_;
   int             **colIndices_;
   double          **colValues_;

   // ----------------------------------------------------------------------
   // matrix and vectors for reduction
   // ----------------------------------------------------------------------

   HYPRE_IJMatrix  reducedA_;           // matrix for reduction
   HYPRE_IJVector  reducedB_;           // RHS vector for reduction
   HYPRE_IJVector  reducedX_;           // solution vector for reduction
   HYPRE_IJVector  reducedR_;           // temporary vector for reduction
   HYPRE_IJMatrix  HYA21_;              // (2,1) block in reduction
   int             A21NRows_;           // number of rows in (2,1) block
   int             reducedAStartRow_;   // Nrows in reduced system
   HYPRE_IJMatrix  HYinvA22_;           // inv(A22) in slide reduction

   // ----------------------------------------------------------------------
   // pointers to current matrix and vectors for solver
   // ----------------------------------------------------------------------

   HYPRE_IJMatrix  currA_;
   HYPRE_IJVector  currB_;
   HYPRE_IJVector  currX_;
   HYPRE_IJVector  currR_;
   int             currentRHS_;
   int             *rhsIDs_;
   int             numRHSs_;

   // ----------------------------------------------------------------------
   // various flags
   // ----------------------------------------------------------------------

   int             matrixVectorsCreated_;
   int             systemAssembled_;
   int             slideReduction_;
   int             schurReduction_;

   // ----------------------------------------------------------------------
   // variables for slide reduction
   // ----------------------------------------------------------------------

   int             *selectedList_;
   int             *selectedListAux_;
   int             nConstraints_;
   int             *constrList_;

   // ----------------------------------------------------------------------
   // variables for the selected solver and preconditioner
   // ----------------------------------------------------------------------

   char            *HYSolverName_;
   HYPRE_Solver    HYSolver_;
   HYsolverID      HYSolverID_;
   int             gmresDim_;
   int             maxIterations_;
   int             finalResNorm_;
   double          tolerance_;
   int             normAbsRel_;

   char            *HYPreconName_;
   HYPRE_Solver    HYPrecon_;
   HYpreconID      HYPreconID_;
   int             HYPreconReuse_;

   // ----------------------------------------------------------------------
   // preconditioner specific variables 
   // ----------------------------------------------------------------------

   int             amgCoarsenType_;
   int             amgNumSweeps_[4];
   int             amgRelaxType_[4];
   double          amgRelaxWeight_[25];
   double          amgStrongThreshold_;
   int             pilutRowSize_;
   double          pilutDropTol_;
   int             pilutMaxNnzPerRow_;
   int             parasailsSym_;
   double          parasailsThreshold_;
   int             parasailsNlevels_;
   double          parasailsFilter_;
   double          parasailsLoadbal_;
   int             parasailsReuse_;
   int             mlNumPreSweeps_;
   int             mlNumPostSweeps_;
   int             mlPresmootherType_;
   int             mlPostsmootherType_;
   double          mlRelaxWeight_;
   double          mlStrongThreshold_;
   int             superluOrdering_;
   char            superluScale_[1];
   double          ddilutFillin_;
   double          ddilutDropTol_;

   // ----------------------------------------------------------------------
   // map 
   // ----------------------------------------------------------------------

   int             *node2EqnMap;

   // ----------------------------------------------------------------------
   // temporary functions for testing purposes
   // ----------------------------------------------------------------------

   friend void fei_hypre_test(int argc, char *argv[]);
   friend void fei_hypre_domaindecomposition(int argc, char *argv[]);

};

#endif

