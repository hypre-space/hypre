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

#ifndef NOFEI
#if defined(FEI_V14) || defined(FEI_V13)
class Lookup
{
   int bogus;
};
#endif
#endif

#ifdef NOFEI
#define GlobalID int
class Lookup
{
   int bogus;
};
#endif

#define HYPRE_FEI_Version() "FEI/HYPRE 1.5.0"

// *************************************************************************
// local enumerations and defines
// -------------------------------------------------------------------------

enum HYsolverID {HYPCG,HYGMRES,HYCGSTAB,HYCGSTABL,HYTFQMR,HYBICGS,HYAMG,
                 HYSUPERLU,HYSUPERLUX,HYY12M,HYAMGE};
enum HYpreconID {HYDIAGONAL,HYPILUT,HYPARASAILS,HYBOOMERAMG,HYML,HYDDILUT,
                 HYPOLY,HYDDICT,HYSCHWARZ};

#define HYFEI_HIGHMASK      2147483647-255
#define HYFEI_SPECIALMASK              255
#define HYFEI_SLIDEREDUCE1             256
#define HYFEI_SLIDEREDUCE2             512
#define HYFEI_SLIDEREDUCE3            1024
#define HYFEI_PRINTMAT                2048
#define HYFEI_PRINTREDMAT             4096
#define HYFEI_PRINTSOL                8192
#define HYFEI_DDILUT                 16384
#define HYFEI_SCHURREDUCE1           32768
#define HYFEI_SCHURREDUCE2           65536
#define HYFEI_SCHURREDUCE3          131072
#define HYFEI_PRINTFEINFO           262144

// *************************************************************************
// class definition
// -------------------------------------------------------------------------

class HYPRE_LinSysCore
#ifndef NOFEI
#if defined(FEI_V14) || defined(FEI_V13) 
           : public LinearSystemCore 
#else
           : public LSC 
#endif
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

   // ======================================================================
   // new functions in FEI 1.5 (not implemented here)
   // ======================================================================

   void setLookup(Lookup& lookup);

   void setConnectivities(GlobalID elemBlock, int numElements,
           int numNodesPerElem, const GlobalID* elemIDs,
           const int* const* connNodes) ;

   void setStiffnessMatrices(GlobalID elemBlock, int numElems,
           const GlobalID* elemIDs, const double *const *const *stiff,
           int numEqnsPerElem, const int *const * eqnIndices);

   void setLoadVectors(GlobalID elemBlock, int numElems,
           const GlobalID* elemIDs, const double *const * load,
           int numEqnsPerElem, const int *const * eqnIndices);

   void setMultCREqns(int multCRSetID, int numCRs, int numNodesPerCR,
           int** nodeNumbers, int** eqnNumbers, int* fieldIDs,
           int* multiplierEqnNumbers);

   void setPenCREqns(int penCRSetID, int numCRs, int numNodesPerCR,
           int** nodeNumbers, int** eqnNumbers, int* fieldIDs);

   // ======================================================================
   // createMatricesAndVectors replaced by setGlobalOffsets in 1.5
   // ======================================================================
   // void createMatricesVectors: provide info for initial creation of 
   //      matrix/vector data, Equation numbers are 1-based, and local sets 
   //      of equation numbers are contiguous.
   // ----------------------------------------------------------------------

   void createMatricesAndVectors(int numGlobalEqns, int firstLocalEqn,
                                 int numLocalEqns);

   void setGlobalOffsets(int len, int* nodeOffsets, int* eqnOffsets,
           int* blkEqnOffsets);

   // ======================================================================
   // allocateMatrix replaced by setMatrixStructure in 1.5
   // ======================================================================
   // void allocateMatrix: provide enough info to allocate the matrix -- 
   //                      i.e., define the structure.
   // ----------------------------------------------------------------------

   void allocateMatrix(int** colIndices, int* rowLengths);

   void setMatrixStructure(int** ptColIndices, int* ptRowLengths,
           int** blkColIndices, int* blkRowLengths, int* ptRowsPerBlkRow);

   // ----------------------------------------------------------------------
   // void resetMatrixAndVector: don't destroy the structure of the matrix, 
   //      but set the value 's' throughout the matrix and vectors.
   // ----------------------------------------------------------------------

   void resetMatrixAndVector(double s);

   // ======================================================================
   // 2 new functions in 1.5
   // ======================================================================

   void resetMatrix(double s);
   void resetRHSVector(double s);

   // ======================================================================
   // new function in 1.5 to deal with block matrix
   // ======================================================================
   // void sumIntoSystemMatrix:
   // this is the primary assembly function. The coefficients 'values'
   // are to be accumumlated into (added to any values already in place)
   // global (0-based) equation 'row' of the matrix.
   // ----------------------------------------------------------------------

   void sumIntoSystemMatrix(int numPtRows, const int* ptRows,
           int numPtCols, const int* ptCols, int numBlkRows, 
           const int* blkRows, int numBlkCols, const int* blkCols,
           const double* const* values);

   // ======================================================================
   // syntax of this functioin has been changed in 1.5
   // ======================================================================
   // void sumIntoSystemMatrix:
   // this is the primary assembly function. The coefficients 'values'
   // are to be accumumlated into (added to any values already in place)
   // global (1-based) [old - 0-based] equation 'row' of the matrix.
   // ----------------------------------------------------------------------

   void sumIntoSystemMatrix(int row, int numValues, const double* values,
                            const int* scatterIndices);

   void sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                            int numPtCols, const int* ptCols,
                            const double* const* values);

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
   
   // ======================================================================
   // new function in 1.5 
   // ======================================================================

   void putNodalFieldData(int fieldID, int fieldSize, int* nodeNumbers,
                          int numNodes, const double* data);

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

   // ======================================================================
   // syntax of this functioin has been changed in 1.5
   // ======================================================================
   // function for getting all of the answers ('x' vector).
   // ----------------------------------------------------------------------

   void getSolution(int* eqnNumbers, double* answers, int len);

   void getSolution(double* answers, int len);

   // ----------------------------------------------------------------------
   //function for getting the (single) entry at equation number 'eqnNumber'.
   // ----------------------------------------------------------------------

   void getSolnEntry(int eqnNumber, double& answer);

   // ======================================================================
   // syntax of this functioin has been changed in 1.5
   // ======================================================================
   // fetch the residual vector (FEI 1.4.x compatible)
   // ----------------------------------------------------------------------

   void formResidual(int* eqnNumbers, double* values, int len);

   void formResidual(double* values, int len);

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

   void   loadConstraintNumbers(int length, int *list);
   void   buildSlideReducedSystem();
   void   buildSlideReducedSystem2();
   double buildSlideReducedSoln();
   double buildSlideReducedSoln2();
   void   buildSchurReducedSystem();
   void   buildSchurReducedRHS();
   double buildSchurReducedSoln();
   void   computeMinResProjection(HYPRE_ParCSRMatrix A_csr, HYPRE_ParVector x, 
                                  HYPRE_ParVector b, double& nrm1, double& nrm2);
   void   addToProjectionSpace(HYPRE_IJVector x, HYPRE_IJVector b);
   char  *getVersion();
   void   beginCreateMapFromSoln();
   void   endCreateMapFromSoln();
   void   putIntoMappedMatrix(int row, int numValues, const double* values,
                              const int* scatterIndices);
   void   getFEGridObject(void **object) { (*object) = fegrid; }

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

   void  solveUsingBoomeramg(int&);
   void  solveUsingSuperLU(int&);
   void  solveUsingSuperLUX(int&);
   void  solveUsingY12M(int&);
   void  solveUsingAMGe(int&);

 private:        //variables

   // ----------------------------------------------------------------------
   // parallel communication information and output levels
   // ----------------------------------------------------------------------

   MPI_Comm        comm_;               // MPI communicator
   int             numProcs_;           // number of processors
   int             mypid_;              // my processor ID
   int             HYOutputLevel_;      // control print information

   // ----------------------------------------------------------------------
   // for storing information about how to load matrix directly
   // ----------------------------------------------------------------------

   int             mapFromSolnFlag_;
   int             mapFromSolnLeng_;
   int             mapFromSolnLengMax_;
   int             *mapFromSolnList_;
   int             *mapFromSolnList2_;

   // ----------------------------------------------------------------------
   // matrix and vectors
   // ----------------------------------------------------------------------

   HYPRE_IJMatrix  HYA_;                // the system matrix
   HYPRE_IJVector  HYb_;                // the current RHS 
   HYPRE_IJVector  *HYbs_;              // an array of RHSs
   HYPRE_IJVector  HYx_;                // the solution vector
   HYPRE_IJVector  HYr_;                // temporary vector for residual
   HYPRE_IJVector  *HYpxs_;             // an array of previous solutions
   HYPRE_IJVector  *HYpbs_;             // an array of previous rhs
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
   HYPRE_IJMatrix  HYA12_;              // (1,2) block in reduction
   int             A21NRows_;           // number of rows in (2,1) block
   int             A21NCols_;           // number of cols in (2,1) block
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
   int             schurReductionCreated_;
   int             minResProjection_;
   int             projectSize_;
   int             projectCurrSize_;

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
   int             amgMeasureType_;
   int             amgNumSweeps_[4];
   int             amgRelaxType_[4];
   double          amgRelaxWeight_[25];
   double          amgStrongThreshold_;
   int             pilutFillin_;
   double          pilutDropTol_;
   int             pilutMaxNnzPerRow_;
   int             parasailsSym_;
   double          parasailsThreshold_;
   int             parasailsNlevels_;
   double          parasailsFilter_;
   double          parasailsLoadbal_;
   int             parasailsReuse_;
   int             mlMethod_;
   int             mlNumPreSweeps_;
   int             mlNumPostSweeps_;
   int             mlPresmootherType_;
   int             mlPostsmootherType_;
   double          mlRelaxWeight_;
   double          mlStrongThreshold_;
   int             mlCoarseSolver_;
   int             superluOrdering_;
   char            superluScale_[1];
   double          ddilutFillin_;
   double          ddilutDropTol_;
   double          ddictFillin_;
   double          ddictDropTol_;
   double          schwarzFillin_;
   int             schwarzNblocks_;
   int             schwarzBlksize_;
   int             polyOrder_;

   // ----------------------------------------------------------------------
   // map and others 
   // ----------------------------------------------------------------------

   void            *fegrid;
   Lookup          *lookup_;
   int             haveLookup_;
   double          **projectionMatrix_; 

   // ----------------------------------------------------------------------
   // temporary functions for testing purposes
   // ----------------------------------------------------------------------

   friend void fei_hypre_test(int argc, char *argv[]);
   friend void fei_hypre_domaindecomposition(int argc, char *argv[]);

};

#endif

