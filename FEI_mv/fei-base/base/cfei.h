#ifndef __cfei_H
#define __cfei_H

/*------------------------------------------------------------------------------
   This is the header for the prototypes of the procedural ("C") version
   of the finite element interface.

   NOTE: ALL functions return an error code which is 0 if successful,
         non-zero if un-successful.

   Noteworthy special case: the iterateToSolve function may return non-zero
   if the solver failed to converge. This is, of course, a non-fatal 
   situation, and the caller should then check the 'status' argument for
   possible further information (solver-specific/solver-dependent).
------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
   First, we define a "Linear System Core" struct. This is the beast that
   provides all solver-library-specific functionality like sumIntoMatrix,
   launchSolver, etc., etc. The pointer 'lsc_' needs to hold an instance
   of an object which implements the C++ interface defined in
   ../base/LinearSystemCore.h. Naturally, an implementation-specific 
   function will be required to create one of these. 

   e.g., ISIS_LinSysCore_create(LinSysCore** lsc,
                                MPI_Comm comm);

   This function would be found in ../support-ISIS/cfei_isis.h, in the case
   of an ISIS++ FEI implementation. Each other FEI implementation will also
   need an equivalent function.
------------------------------------------------------------------------------*/

struct LinSysCore_struct {
   void* lsc_;
};
typedef struct LinSysCore_struct LinSysCore;

/*------------------------------------------------------------------------------
   Next, define an opaque CFEI thingy which will be an FEI context, and will
   be the first argument to all of the C FEI functions which follow in this
   header.
------------------------------------------------------------------------------*/

struct CFEI_struct {
   void* cfei_;
};
typedef struct CFEI_struct CFEI;

/*------------------------------------------------------------------------------
   And now, the function prototypes...
------------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C" {
#endif

/*
   Initialization function. Creates an FEI instance, wrapped in a CFEI pointer.
*/
int FEI_create(CFEI** cfei,
               LinSysCore* lsc,
               MPI_Comm FEI_COMM_WORLD, 
               int masterRank);

/* A function to destroy allocated memory. */
int FEI_destroy(CFEI** cfei);

/* A function to destroy those LinSysCore things. */
int LinSysCore_destroy(LinSysCore** lsc);

/* Structural initialization sequence.............................*/

/* per-solve-step initialization */
int FEI_initSolveStep(CFEI* cfei, 
                      int numElemBlocks, 
                      int solveType);

/* identify all the solution fields present in the analysis.......*/
int FEI_initFields(CFEI* cfei, 
                   int numFields, 
                   int *cardFields, 
                   int *fieldIDs); 

/* begin blocked-element initialization step..................*/
int FEI_beginInitElemBlock(CFEI* cfei, 
                           GlobalID elemBlockID, 
                           int numNodesPerElement, 
                           int *numElemFields,
                           int **elemFieldIDs,
                           int interleaveStrategy,
                           int lumpingStrategy,
                           int numElemDOF, 
                           int numElemSets,
                           int numElemTotal);

/* initialize element sets that make up the blocks */
int FEI_initElemSet(CFEI* cfei, 
                    int numElems,
                    GlobalID *elemIDs, 
                    GlobalID **elemConn);

/* end blocked-element initialization */
int FEI_endInitElemBlock(CFEI* cfei);

/* begin collective node set initialization step........................*/
int FEI_beginInitNodeSets(CFEI* cfei, 
                          int numSharedNodeSets, 
                          int numExtNodeSets);

/* initialize nodal sets for shared nodes */
int FEI_initSharedNodeSet(CFEI* cfei, 
                          GlobalID *sharedNodeIDs,
                          int lenSharedNodeIDs, 
                          int **sharedProcIDs,
                          int *lenSharedProcIDs);

/* initialize nodal sets for external (off-processor) communcation */
int FEI_initExtNodeSet(CFEI* cfei, 
                       GlobalID *extNodeIDs,
                       int lenExtNodeIDs, 
                       int **extProcIDs, 
                       int *lenExtProcIDs);

/* end node set initialization */
int FEI_endInitNodeSets(CFEI* cfei);


/* begin constraint relation set initialization step.........*/
int FEI_beginInitCREqns(CFEI* cfei, 
                        int numCRMultSets,
                        int numCRPenSets);

/* constraint relation initialization - lagrange multiplier formulation */
int FEI_initCRMult(CFEI* cfei,
                   GlobalID **CRNodeTable,
                   int *CRFieldList,
                   int numMultCRs,
                   int lenCRNodeList,
                   int* CRMultID);

/* constraint relation initialization - penalty function formulation */
int FEI_initCRPen(CFEI* cfei, 
                  GlobalID **CRNodeTable, 
                  int *CRFieldList,
                  int numPenCRs, 
                  int lenCRNodeList, 
                  int* CRPenID); 

/* end constraint relation list initialization */
int FEI_endInitCREqns(CFEI* cfei);

/* indicate that overall initialization sequence is complete */
int FEI_initComplete(CFEI* cfei);

/* FE data load sequence..........................................*/

/* set a value (usually zeros) througout the linear system.....*/
int FEI_resetSystem(CFEI* cfei, double s);

/* begin node-set data load step.............................*/
int FEI_beginLoadNodeSets(CFEI* cfei, int numBCNodeSets);

/* boundary condition data load step */
int FEI_loadBCSet(CFEI* cfei, 
                  GlobalID *BCNodeSet,
                  int lenBCNodeSet,
                  int BCFieldID,
                  double **alphaBCDataTable,
                  double **betaBCDataTable,
                  double **gammaBCDataTable);

/* end node-set data load step */
int FEI_endLoadNodeSets(CFEI* cfei);

/* begin blocked-element data loading step....................*/
int FEI_beginLoadElemBlock(CFEI* cfei, 
                           GlobalID elemBlockID,
                           int numElemSets,
                           int numElemTotal);
  
/* elemSet-based stiffness/rhs data loading step */
int FEI_loadElemSet(CFEI* cfei, 
                    int elemSetID, 
                    int numElems, 
                    GlobalID *elemIDs,  
                    GlobalID **elemConn,
                    double ***elemStiffness,
                    double **elemLoad,
                    int elemFormat);

/* end blocked-element data loading step*/
int FEI_endLoadElemBlock(CFEI* cfei);


/* begin constraint relation data load step...................*/
int FEI_beginLoadCREqns(CFEI* cfei, 
                        int numCRMultSets,
                        int numCRPenSets);

/* lagrange-multiplier constraint relation load step */
int FEI_loadCRMult(CFEI* cfei, 
                   int CRMultID, 
                   int numMultCRs,
                   GlobalID **CRNodeTable,  
                   int *CRFieldList,
                   double **CRWeightTable,
                   double *CRValueList,
                   int lenCRNodeList);

/* penalty formulation constraint relation load step */
int FEI_loadCRPen(CFEI* cfei, 
                  int CRPenID,
                  int numPenCRs, 
                  GlobalID **CRNodeTable,
                  int *CRFieldList,
                  double **CRWeightTable,  
                  double *CRValueList,
                  double *penValues,
                  int lenCRNodeList);

/* end constraint relation data load step */
int FEI_endLoadCREqns(CFEI* cfei);

/* indicate that overall data loading sequence is complete */
int FEI_loadComplete(CFEI* cfei);

/* Equation solution services..................................... */

/* set parameters associated with solver choice, etc. */
int FEI_parameters(CFEI* cfei, 
                   int numParams, 
                   char **paramStrings);

/* start iterative solution */
int FEI_iterateToSolve(CFEI* cfei, int* status);

/* query how many iterations it took to solve. */
int FEI_iterations(CFEI* cfei, int* iterations);

/* Solution return services....................................... */
 
/* return nodal-based solution to FE analysis on a block-by-block basis */
int FEI_getBlockNodeSolution(CFEI* cfei, 
                             GlobalID elemBlockID,
                             GlobalID *nodeIDList, 
                             int* lenNodeIDList, 
                             int *offset,
                             double *answers);

/* return field-based solution to FE analysis on a block-by-block basis */
int FEI_getBlockFieldNodeSolution(CFEI* cfei, 
                                  GlobalID elemBlockID,
                                  int fieldID,
                                  GlobalID *nodeIDList, 
                                  int* lenNodeIDList, 
                                  int *offset,
                                  double *results);

/* return element-based solution to FE analysis on a block-by-block basis */
int FEI_getBlockElemSolution(CFEI* cfei, 
                             GlobalID elemBlockID,
                             GlobalID *elemIDList, 
                             int* lenElemIDList, 
                             int *offset,
                             double *results, 
                             int* numElemDOF);

/* return Lagrange solution to FE analysis on a whole-processor basis */
int FEI_getCRMultSolution(CFEI* cfei, 
                          int* numCRMultSets, 
                          int *CRMultIDs,
                          int *offset, 
                          double *results);

/* return Lagrange solution to FE analysis on a constraint-set basis */
int FEI_getCRMultParam(CFEI* cfei, 
                       int CRMultID, 
                       int numMultCRs,
                       double *multValues); 

/* put nodal-based solution to FE analysis on a block-by-block basis */
int FEI_putBlockNodeSolution(CFEI* cfei, 
                             GlobalID elemBlockID,  
                             GlobalID *nodeIDList, 
                             int lenNodeIDList, 
                             const int *offset,  
                             const double *estimates);

/* put field-based solution to FE analysis on a block-by-block basis */
int FEI_putBlockFieldNodeSolution(CFEI* cfei, 
                                  GlobalID elemBlockID,  
                                  int fieldID, 
                                  GlobalID *nodeIDList, 
                                  int lenNodeIDList, 
                                  int *offset,  
                                  double *estimates);
         
/*  put element-based solution to FE analysis on a block-by-block basis */ 
int FEI_putElemBlockSolution(CFEI* cfei, 
                             GlobalID elemBlockID,  
                             GlobalID *elemIDList, 
                             int lenElemIDList, 
                             int *offset,  
                             double *estimates, 
                             int numElemDOF);
  
/*  put Lagrange solution to FE analysis on a constraint-set basis */
int FEI_putCRMultParam(CFEI* cfei, 
                       int CRMultID, 
                       int numMultCRs,
                       double *multEstimates);
 
/*  some utility functions to aid in using the "put" functions for passing */
/*  an initial guess to the solver */

/* return sizes associated with Lagrange solution to FE analysis */
int FEI_getCRMultSizes(CFEI* cfei, 
                       int* numCRMultIDs, 
                       int* lenAnswers);
 
/* form some data structures needed to return element solution parameters */
int FEI_getBlockElemIDList(CFEI* cfei, 
                           GlobalID elemBlockID, 
                           GlobalID *elemIDList, 
                           int *lenElemIDList);

/* form some data structures needed to return nodal solution parameters */
int FEI_getBlockNodeIDList(CFEI* cfei, 
                           GlobalID elemBlockID,
                           GlobalID *nodeIDList, 
                           int *lenNodeIDList);

/*  return the number of solution parameters at a given node */
int FEI_getNumSolnParams(CFEI* cfei, GlobalID globalNodeID, int* numSolnParams);

/*  return the number of stored element blocks */
int FEI_getNumElemBlocks(CFEI* cfei, int* numElemBlocks);

/*  return the number of active nodes in a given element block */
int FEI_getNumBlockActNodes(CFEI* cfei, GlobalID blockID,
                            int* numBlockActNodes);

/*  return the number of active equations in a given element block */
int FEI_getNumBlockActEqns(CFEI* cfei, GlobalID blockID, int* numBlockActEqns);

/*  return the number of nodes associated with elements of a
    given block ID */
int FEI_getNumNodesPerElement(CFEI* cfei, GlobalID blockID,
                              int* numNodesPerElement);

/*  return the number of eqns associated with elements of a
    given block ID */
int FEI_getNumEqnsPerElement(CFEI* cfei, GlobalID blockID,
                             int* numEqnsPerElement);
 
/*  return the number of elements in a given element block */
int FEI_getNumBlockElements(CFEI* cfei, GlobalID blockID,
                            int* numBlockElements);

/*  return the number of element equations in a given element block */
int FEI_getNumBlockElemEqns(CFEI* cfei, GlobalID blockID,
                            int* numBlockElemEqns);

#ifdef __cplusplus
}
#endif

#endif

