#ifndef _LinearSystemCore_h_
#define _LinearSystemCore_h_

//
//When creating a specific FEI implementation, i.e., a version that
//supports a specific underlying linear solver library, the main
//task that must be performed is the implementation of a class that
//derives from this class, LinearSystemCore.
//
//This is the class that holds and manipulates all solver-library-specific
//stuff, such as matrices/vectors, solvers/preconditioners, etc. An
//instance of this class is owned and used by the class that implements
//the public FEI spec. i.e., when element contributions, etc., are received
//from the finite-element application, the data is ultimately passed to this
//class for assembly into the sparse matrix and associated vectors. This class
//will also be asked to launch any underlying solver, and finally to
//return the solution.
//
//See the file LinearSystemCore.README for some general descriptions, etc.
//

//Files that should to be included before the compiler
//reaches this header:
//
//#include "src/Data.h"        //for the declaration of the 'Data' class.
//#include "base/basicTypes.h" //for the definition of the 'GlobalID' type.
//

class Lookup;

class LinearSystemCore {
 public:
   LinearSystemCore(){};
   virtual ~LinearSystemCore() {};


   //clone
   //for cloning a LinearSystemCore instance.
   virtual LinearSystemCore* clone() = 0;


   //parameters
   //for setting generic argc/argv style parameters.

   virtual void parameters(int numParams, char** params) = 0;


   //setLookup
   //for providing an object to use for looking up an equation, given a
   //node or node/field pair, and vice-versa, etc.

   virtual void setLookup(Lookup& lookup) = 0;


   //setGlobalOffsets
   //for providing three lists, of length numProcs+1. Those lists are:
   //nodeOffsets - first local nodeNumber for each processor
   //eqnOffsets - first local equation-number for each processor
   //blkEqnOffsets - first local block-equation-number for each processor.
   //
   //These numbers are all 0-based.
   //
   //len is numProcs+1
   //
   //From this information, LinearSystemCore implementations may trivially
   //obtain local and global number-of-nodes, number-of-eqns, and
   //number-of-block-eqns.

   virtual void setGlobalOffsets(int len, int* nodeOffsets,
                                 int* eqnOffsets, int* blkEqnOffsets) = 0;


   //setConnectivities
   //for providing element connectivity lists -- lists of nodes
   //connected to each element, on an elem-block-by-elem-block basis.
                                 
   virtual void setConnectivities(GlobalID elemBlock,
                                  int numElements,
                                  int numNodesPerElem,
                                  const GlobalID* elemIDs,
                                  const int* const* connNodes) = 0;


   //setStiffnessMatrices
   //for providing un-modified, direct-from-the-application, element-wise
   //stiffness matrices

   virtual void setStiffnessMatrices(GlobalID elemBlock,
                                     int numElems,
                                     const GlobalID* elemIDs,
                                     const double *const *const *stiff,
                                     int numEqnsPerElem,
                                     const int *const * eqnIndices) = 0;


   //setLoadVectors
   //for providing un-modified, direct-from-the-application, element-wise
   //load vectors

   virtual void setLoadVectors(GlobalID elemBlock,
                               int numElems,
                               const GlobalID* elemIDs,
                               const double *const * load,
                               int numEqnsPerElem,
                               const int *const * eqnIndices) = 0;


   //setMatrixStructure
   //for providing the information necessary to allocate the matrix/vectors

   virtual void setMatrixStructure(int** ptColIndices,
                                   int* ptRrowLengths,
                                   int** blkColIndices,
                                   int* blkRowLengths,
                                   int* ptRowsPerBlkRow) = 0;


   //setMultCREqns/setPenCREqns
   //identify which nodes and equations are associated with constraint-equations
   //
   //these functions each provide info for a 'constraint-relation-set'. There
   //may be many CR-sets. Each CR-set contains 'numCRs' constraint-relations,
   //and each constraint-relation applies to 'numNodesPerCR' nodes. Thus,
   //'nodeNumbers' and 'eqnNumbers' are 2-D arrays with 'numCRs' rows, and
   //'numNodesPerCR' columns. The 'nodeNumbers' table contains the nodes that
   //are involved in each constraint, and the 'eqnNumbers' table contains the
   //constrained global equation-number at each node (actually it contains the
   //first equation-number of the constrained field on the node; that field may
   //have more than one equation).
   //The 'fieldIDs' list is of length 'numNodesPerCR', and contains the
   //constrained field at each node. All constraints in a constraint-set are
   //homogeneous on the constrained field at each node. By this I mean, each
   //constraint in the set constrains fieldIds[j] at nodeNumbers[i][j].
   //
   //Finally, the argument 'multiplierEqnNumbers' is a list of length 'numCRs',
   //and contains the equation number for each of the lagrange multipliers.

   virtual void setMultCREqns(int multCRSetID,
                              int numCRs, int numNodesPerCR,
                              int** nodeNumbers, int** eqnNumbers,
                              int* fieldIDs,
                              int* multiplierEqnNumbers) = 0;

   virtual void setPenCREqns(int penCRSetID,
                              int numCRs, int numNodesPerCR,
                              int** nodeNumbers, int** eqnNumbers,
                              int* fieldIDs) = 0;


   //sumIntoSystemMatrix, provides point-entry data, as well as
   //block-entry data. This is the primary assembly function, used for
   //providing the local portions of element contributions.

   virtual void sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                                    int numPtCols, const int* ptCols,
                                    int numBlkRows, const int* blkRows,
                                    int numBlkCols, const int* blkCols,
                                    const double* const* values) = 0;

   //sumIntoSystemMatrix, purely point-entry version
   //for accumulating coefficient data into the matrix,
   //This will be called when a matrix contribution fills only part of a
   //block-equation. e.g., when a penalty constraint is being applied to a
   //single solution field on a node that has several solution fields.
   //(A block-equation contains all solution field equations at a node.)
   //

   virtual void sumIntoSystemMatrix(int numPtRows, const int* ptRows,
                                    int numPtCols, const int* ptCols,
                                    const double* const* values) = 0;

   //sumIntoRHSVector
   //for accumulating coefficients into the rhs vector

   virtual void sumIntoRHSVector(int num, const double* values,
                                 const int* indices) = 0;

   //matrixLoadComplete
   //for signalling the linsyscore object that data-loading is finished.

   virtual void matrixLoadComplete() = 0;

   //putNodalFieldData
   //for providing nodal data associated with a particular fieldID.
   //nodeNumbers is a list of length numNodes.
   //offsets is a list of length numNodes+1.
   //data contains the incoming data. data for the ith node lies in the
   //locations data[offsets[i]] ... data[offsets[i+1] -1 ]
   //
   //incoming data may include non-local nodes. These should simply be
   //skipped by the linsyscore object.

   virtual void putNodalFieldData(int fieldID, int fieldSize,
                                  int* nodeNumbers, int numNodes,
                                  const double* data) = 0;


   //resetMatrixAndVector
   //for setting the scalar 's' throughout the matrix and/or rhs vector.

   virtual void resetMatrixAndVector(double s) = 0;
   virtual void resetMatrix(double s) = 0;
   virtual void resetRHSVector(double s) = 0;

   //functions for enforcing boundary conditions.
   virtual void enforceEssentialBC(int* globalEqn, double* alpha,
                                   double* gamma, int len) = 0;

   virtual void enforceRemoteEssBCs(int numEqns, int* globalEqns,
                                          int** colIndices, int* colIndLen,
                                          double** coefs) = 0;

   virtual void enforceOtherBC(int* globalEqn, double* alpha,
                               double* beta, double* gamma, int len) = 0;

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
};

#endif

