#ifndef _FEI_Implementation_h_
#define _FEI_Implementation_h_

class ESI_Broker;
class Data;
class LinearSystemCore;
class SNL_FEI_Structure;
class Filter;
class CommUtils;

#include "base/FEI_SNL.h"
#include "base/fei_mpi.h"
#include "base/feiArray.h"

/**
This is the (C++) user's point of interaction with the FEI implementation. The
user will declare an instance of this class in their code, and call the public
FEI functions on that instance. The functions implemented by this class are
those in the abstract FEI declaration, plus possibly others. i.e., the functions
provided by this class are a superset of those in the FEI specification.
<p>
This class takes, as a constructor argument, an ESI_Broker implementation which 
may be either
a "genuine" factory for ESI interface instances, or it may be a shell containing
only an instance of a LinearSystemCore implementation or a FiniteElementData
implementation. These are the abstract interfaces through which solver libraries
may be coupled to this FEI implementation.<p>
As of August 2001, the following solver implementations of these interfaces
exist:<p>
<ul>
<li>LinearSystemCore:
   <ul>
   <li>Aztec
   <li>HYPRE
   <li>ISIS++
   <li>PETSc
   <li>Prometheus
   <li>SPOOLES
   </ul>
<li>ESI_Broker (actual ESI implementations):
   <ul>
   <li>ISIS_ESI
   <li>Trilinos
   </ul>
<li>FiniteElementData:
   <ul>
   <li>FETI
   </ul>
</ul>

 */

class FEI_Implementation : public FEI_SNL {

 public:
  /**  constructor.
      @param broker an instance of an ESI_Broker. (LSC_Broker can be used here,
      it is a wrapper for the old LinearSystemCore.)
      @param comm MPI_Comm communicator
      @param masterRank The "master" mpi rank. Defaults to 0 if not supplied.
      This is not an important parameter, simply determining which processor
      will produce screen output if the parameter "outputLevel" is set to a
      value greater than 0 via a call to the parameters function.
  */
   FEI_Implementation(ESI_Broker* broker, MPI_Comm comm,
                      int masterRank=0);

   /** Destructor. */
   virtual ~FEI_Implementation();

//public FEI functions:

   // set misc. argc/argv style parameters for solver choice, etc.
   int parameters(int numParams, char **paramStrings);

//Structural initialization functions.............................

   int setIDLists(int numMatrices,
                  const int* matrixIDs,
                  int numRHSs,
                  const int* rhsIDs);

   int setSolveType(int solveType);

   // identify all the solution fields present in the analysis
   int initFields(int numFields, 
                  const int *fieldSizes, 
                  const int *fieldIDs);

   int initElemBlock(GlobalID elemBlockID,
                     int numElements,
                     int numNodesPerElement,
                     const int* numFieldsPerNode,
                     const int* const* nodalFieldIDs,
                     int numElemDOFPerElement,
                     const int* elemDOFFieldIDs,
                     int interleaveStrategy);

   int initElem(GlobalID elemBlockID,
                GlobalID elemID,
                const GlobalID* elemConn);

   int initSlaveVariable(GlobalID slaveNodeID, 
			 int slaveFieldID,
			 int offsetIntoSlaveField,
			 int numMasterNodes,
			 const GlobalID* masterNodeIDs,
			 const int* masterFieldIDs,
			 const double* weights,
			 double rhsValue);

   int deleteMultCRs();

   // identify sets of shared nodes
   int initSharedNodes(int numSharedNodes,
                       const GlobalID *sharedNodeIDs,  
                       const int* numProcsPerNode, 
                       const int *const *sharingProcIDs);

   int initCRMult(int numCRNodes,
                  const GlobalID* CRNodes,
                  const int *CRFields,
                  int& CRID); 

   int initCRPen(int numCRNodes,
                 const GlobalID* CRNodes, 
                 const int *CRFields,
                 int& CRID); 

   int initCoefAccessPattern(int patternID,
                             int numRowIDs,
                             const int* numFieldsPerRow,
                             const int* const* rowFieldIDs,
                             int numColIDsPerRow,
                             const int* numFieldsPerCol,
                             const int* const* colFieldIDs,
                             int interleaveStrategy);

   int initCoefAccess(int patternID,
		      const int* rowIDTypes,
                      const GlobalID* rowIDs,
		      const int* colIDTypes,
                      const GlobalID* colIDs);

   int initSubstructure(int substructureID,
                        int numIDs,
			const int* IDTypes,
                        const GlobalID* IDs);

   // indicate that overall initialization sequence is complete
   int initComplete();

// FEI data loading sequence..........................................

   //direct data to a specific internal data structure
   //i.e., set the current matrix 'context'.
   int setCurrentMatrix(int matID);

   //direct data to a specific internal data structure
   //i.e., set the current RHS 'context'.
   int setCurrentRHS(int rhsID);

   // set a value (usually zeros) throughout the linear system
   int resetSystem(double s=0.0);

   // set a value (usually zeros) throughout the matrix or rhs-vector
   // separately
   int resetMatrix(double s=0.0);
   int resetRHSVector(double s=0.0);
   int resetInitialGuess(double s=0.0);

    int loadNodeBCs(int numNodes,
                    const GlobalID *nodeIDs,  
                    int fieldID,
                    const double *const *alpha,  
                    const double *const *beta,  
                    const double *const *gamma);

    int loadElemBCs(int numElems,
                    const GlobalID* elemIDs,  
                    int fieldID,
                    const double *const *alpha,  
                    const double *const *beta,  
                    const double *const *gamma);

   int sumInElem(GlobalID elemBlockID,
                 GlobalID elemID,
                 const GlobalID* elemConn,
                 const double* const* elemStiffness,
                 const double* elemLoad,
                 int elemFormat);

   int sumInElemMatrix(GlobalID elemBlockID,
                       GlobalID elemID,
                       const GlobalID* elemConn,
                       const double* const* elemStiffness,
                       int elemFormat);

   int sumInElemRHS(GlobalID elemBlockID,
                    GlobalID elemID,
                    const GlobalID* elemConn,
                    const double* elemLoad);

   //element-wise transfer operator
   int loadElemTransfer(GlobalID elemBlockID,
                        GlobalID elemID,
                        const GlobalID* coarseNodeList,
                        int fineNodesPerCoarseElem,
                        const GlobalID* fineNodeList,
                        const double* const* elemProlong,
                        const double* const* elemRestrict);

   int loadCRMult(int CRID,
                  int numCRNodes,
                  const GlobalID* CRNodes,
                  const int* CRFields,
                  const double* CRWeights,
                  double CRValue);

   int loadCRPen(int CRID,
                 int numCRNodes,
                 const GlobalID* CRNodes,
                 const int* CRFields,
                 const double* CRWeights,
                 double CRValue,
                 double penValue);

   int sumIntoMatrix(int patternID,
		     const int* rowIDTypes,
                     const GlobalID* rowIDs,
		     const int* colIDTypes,
                     const GlobalID* colIDs,
                     const double* const* matrixEntries);

   int sumIntoRHS(int patternID,
		  const int* IDTypes,
                  const GlobalID* IDs,
                  const double* rhsEntries);

   int putIntoMatrix(int patternID,
		     const int* rowIDTypes,
                     const GlobalID* rowIDs,
		     const int* colIDTypes,
                     const GlobalID* colIDs,
                     const double* const* matrixEntries);

   int putIntoRHS(int patternID,
		  const int* IDTypes,
                  const GlobalID* IDs,
                  const double* rhsEntries);

   int getFromMatrix(int patternID,
		     const int* rowIDTypes,
                     const GlobalID* rowIDs,
		     const int* colIDTypes,
                     const GlobalID* colIDs,
                     double** matrixEntries);

   int getFromRHS(int patternID,
		  const int* IDTypes,
                  const GlobalID* IDs,
                  double* rhsEntries);

// Equation solution services.....................................

    // set scalar coefficients for forming aggregate (linear-combination)
    // system of matrices.

   int setMatScalars(int numScalars,
                     const int* IDs, 
                     const double* scalars);

    // set scalar coefficients for aggregating RHS vectors.
    
   int setRHSScalars(int numScalars,
                     const int* IDs,
                     const double* scalars);

   //indicate that the matrix/vectors can be finalized now. e.g., boundary-
   //conditions enforced, etc., etc.

   int loadComplete();

   //get residual norms
   int residualNorm(int whichNorm,
                    int numFields,
                    int* fieldIDs,
                    double* norms);

    // start iterative solution
   int solve(int& status);

    // query iterations performed.

   int iterations(int& itersTaken) const;

   int version(char*& versionString);

   // query for some accumulated timing information. Collective function.
   int cumulative_MPI_Wtimes(double& initTime,
                             double& loadTime,
                             double& solveTime,
                             double& solnReturnTime,
                             int timingMode);

   // query the amount of memory currently allocated within this implementation
   int allocatedSize(int& bytes);

// Solution return services.......................................
 
    // return all nodal solution params on a block-by-block basis 
 
    int getBlockNodeSolution(GlobalID elemBlockID,  
                             int numNodes, 
                             const GlobalID *nodeIDs, 
                             int *offsets,
                             double *results);
 
    // return nodal solution for one field on a block-by-block basis 
    int getBlockFieldNodeSolution(GlobalID elemBlockID,
                                  int fieldID,
                                  int numNodes, 
                                  const GlobalID *nodeIDs, 
                                  double *results);
         
    // return element solution params on a block-by-block basis 
    int getBlockElemSolution(GlobalID elemBlockID,  
                             int numElems, 
                             const GlobalID *elemIDs,
                             int& numElemDOFPerElement,
                             double *results);

   int getNumCRMultipliers(int& numMultCRs);
   int getCRMultIDList(int numMultCRs, int* multIDs);

   // get Lagrange Multipliers
   int getCRMultipliers(int numCRs,
                        const int* CRIDs,
                        double *multipliers);

   int getSubstructureSize( int substructureID,
			    int& numIDs );

   int getSubstructureIDList(int substructureID,
			     int numNodes,
			     int* IDTypes,
			     GlobalID* IDs );

   int getSubstructureFieldSolution(int substructureID,
				    int fieldID,
				    int numIDs,
				    const int* IDTypes,
				    const GlobalID *IDs,
				    double *results);

   int putSubstructureFieldSolution(int substructureID,
				    int fieldID,
				    int numIDs,
				    const int* IDTypes,
				    const GlobalID *IDs,
				    const double *estimates);

   int putSubstructureFieldData(int substructureID,
				int fieldID,
				int numNodes,
				const int* IDTypes,
				const GlobalID *nodeIDs,
				const double *data);   

 
// associated "puts" paralleling the solution return services.
// 
// the int sizing parameters are passed for error-checking purposes, so
// that the interface implementation can tell if the passed estimate
// vectors make sense -before- an attempt is made to utilize them as
// initial guesses by unpacking them into the solver's native solution
// vector format (these parameters include lenNodeIDList, lenElemIDList,
// numElemDOF, and numMultCRs -- all other passed params are either 
// vectors or block/constraint-set IDs)

    // put nodal-based solution guess on a block-by-block basis 
    int putBlockNodeSolution(GlobalID elemBlockID, 
                             int numNodes, 
                             const GlobalID *nodeIDs, 
                             const int *offsets,
                             const double *estimates);

    // put nodal-based guess for one field on a block-by-block basis 
    int putBlockFieldNodeSolution(GlobalID elemBlockID, 
                                  int fieldID, 
                                  int numNodes, 
                                  const GlobalID *nodeIDs, 
                                  const double *estimates);
         
    // put element-based solution guess on a block-by-block basis
    int putBlockElemSolution(GlobalID elemBlockID,  
                             int numElems, 
                             const GlobalID *elemIDs, 
                             int dofPerElem,
                             const double *estimates);

    // put Lagrange solution to FE analysis on a constraint-set basis 
    int putCRMultipliers(int numMultCRs, 
                         const int* CRIDs,
                         const double* multEstimates);

// utility functions that aid in integrating the FEI calls..............

// support methods for the "gets" and "puts" of the soln services.


    // return info associated with blocked nodal solution
    int getBlockNodeIDList(GlobalID elemBlockID,
                           int numNodes,
                           GlobalID *nodeIDs);

    // return info associated with blocked element solution
   int getBlockElemIDList(GlobalID elemBlockID, 
                          int numElems, 
                          GlobalID* elemIDs);
 
// miscellaneous self-explanatory "read-only" query functions............ 
 
    int getNumSolnParams(GlobalID nodeID, int& numSolnParams) const;

    int getNumElemBlocks(int& numElemBlocks) const;

    //  return the number of active nodes in a given element block
    int getNumBlockActNodes(GlobalID blockID, int& numNodes) const;

    //  return the number of active equations in a given element block
    int getNumBlockActEqns(GlobalID blockID, int& numEqns) const;

    //  return the number of nodes associated with elements of a
    //  given block ID
    int getNumNodesPerElement(GlobalID blockID, int& nodesPerElem) const;
    
    //  return the number of equations (including element eqns)
    //  associated with elements of a given block ID
    int getNumEqnsPerElement(GlobalID blockID, int& numEqns) const;

    //  return the number of elements associated with this blockID
    int getNumBlockElements(GlobalID blockID, int& numElems) const;

    //  return the number of elements eqns for elems w/ this blockID
    int getNumBlockElemDOF(GlobalID blockID, int& DOFPerElem) const;


    // return the parameters that have been set so far. The caller should
    // NOT delete the paramStrings pointer.
    int getParameters(int& numParams, char**& paramStrings);

    //And now a couple of non-FEI query functions that Sandia applications
    //need to augment the matrix-access functions. I (Alan Williams) will
    //argue to have these included in the FEI 2.1 specification update.

    //Query the size of a field. This info is supplied to the FEI (initFields)
    //by the application, but may not be easily obtainable on the app side at
    //all times. Thus, it would be nice if the FEI could answer this query.
    int getFieldSize(int fieldID, int& numScalars);

    /**Since the ultimate intent for matrix-access is to bypass the FEI and go
     straight to the underlying ESI data objects, we need a translation
     function to map between the IDs that the FEI deals in, and equation
     numbers that ESI linear algebra objects deal in.
     @param ID Identifier of either a node or an element.
     @param idType Can take either of the values FEI_NODE or FEI_ELEMENT.
     @param fieldID Identifies a particular field at this [node||element].
     @param numEqns Output. Number of equations associated with this
     node/field (or element/field) pair.
     @param eqnNumbers Caller-allocated array. On exit, this is filled with the
     above-described equation-numbers. They are global 0-based numbers.
    */
    int getEqnNumbers(GlobalID ID,
		      int idType, 
		      int fieldID,
		      int& numEqns,
		      int* eqnNumbers);

    /**Get the solution data for a particular field, on an arbitrary set of
       nodes.
       @param fieldID Input. field identifier for which solution data is being
       requested.
       @param numNodes Input. Length of the nodeIDs list.
       @param nodeIDs Input. List specifying the nodes on which solution
       data is being requested.
       @param results Allocated by caller, but contents are output.
       Solution data for the i-th node/element starts in position i*fieldSize,
       where fieldSize is the number of scalar components that make up 
       'fieldID'.
       @return error-code 0 if successful
    */
    int getNodalFieldSolution(int fieldID,
			      int numNodes,
			      const GlobalID* nodeIDs,
			      double* results);

    int getNumLocalNodes(int& numNodes);

    int getLocalNodeIDList(int& numNodes,
			   GlobalID* nodeIDs,
			   int lenNodeIDs);

    int putNodalFieldData(int fieldID,
			  int numNodes,
			  const GlobalID* nodeIDs,
			  const double* nodeData);

  //============================================================================
  private: //functions

    void deleteIDs();
    void deleteRHSScalars();

    int allocateInternalFEIs();

    void debugOut(const char* msg);
    void debugOut(const char* msg, int whichFEI);

    void buildLinearSystem();
    int aggregateSystem();

    void messageAbort(const char* msg);
    void notAllocatedAbort(const char* name);
    void needParametersAbort(const char* name);
    void badParametersAbort(const char* name);

    void setDebugOutput(const char* path, const char* name);

  //============================================================================
  private: //member variables

    ESI_Broker* broker_;
    LinearSystemCore* linSysCore_;
    feiArray<LinearSystemCore*> lscArray_;
    bool haveESI_;
    bool haveLinSysCore_;
    bool haveFEData_;
    SNL_FEI_Structure* problemStructure_;
    Filter** filter_;

    CommUtils* commUtils_;

    int numInternalFEIs_;
    bool internalFEIsAllocated_;

    feiArray<int> feiIDs_;
    feiArray<int> matrixIDs_;
    feiArray<int> numRHSIDs_;
    int** rhsIDs_;

    bool IDsAllocated_;

    double* matScalars_;
    bool matScalarsSet_;
    double** rhsScalars_;
    bool rhsScalarsSet_;

    int index_soln_filter_;
    int index_current_filter_;
    int index_current_rhs_row_;

    int solveType_;

    bool setSolveTypeCalled_;
    bool initPhaseIsComplete_;

    bool aggregateSystemFormed_;
    int newMatrixDataLoaded_;

    Data *soln_fei_matrix_;
    Data *soln_fei_vector_;

    MPI_Comm comm_;

    int masterRank_;
    int localRank_;
    int numProcs_;

    int outputLevel_;

    char* debugPath_;
    char* debugFileName_;
    int solveCounter_;
    int debugOutput_;
    ostream* dbgOStreamPtr_;
    bool dbgFileOpened_;
    ofstream* dbgFStreamPtr_;

    double initTime_, loadTime_, solveTime_, solnReturnTime_;

    int numParams_;
    char** paramStrings_;
};

#endif

