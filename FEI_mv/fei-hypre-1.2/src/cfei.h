#ifndef __cfei_H
#define __cfei_H

/*
   This is the header for the prototypes of the procedural version
   of the finite element interface.
*/

/* Initialization function. Specifies how many linear systems will
   be being worked with. */
void FEI_create(int numSystems, 
                MPI_Comm FEI_COMM_WORLD, 
                int masterRank);

/* Like a destructor function, gets rid of allocated memory. */
void FEI_destroy(void);

/* Structural initialization sequence.............................*/

/* per-solve-step initialization */
int FEI_initSolveStep(int sysHandle, 
                  int numElemBlocks, 
                  int solveType);

/* identify all the solution fields present in the analysis.......*/
int FEI_initFields(int sysHandle, 
               int numFields, 
               int *cardFields, 
               int *fieldIDs); 

/* begin blocked-element initialization step..................*/
int FEI_beginInitElemBlock(int sysHandle, 
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
int FEI_initElemSet(int sysHandle, 
                int numElems,
                GlobalID *elemIDs, 
                GlobalID **elemConn);

/* end blocked-element initialization */
int FEI_endInitElemBlock(int sysHandle);

/* begin collective node set initialization step........................*/
int FEI_beginInitNodeSets(int sysHandle, 
                      int numSharedNodeSets, 
                      int numExtNodeSets);

/* initialize nodal sets for shared nodes */
int FEI_initSharedNodeSet(int sysHandle, 
                      GlobalID *sharedNodeIDs,
                      int lenSharedNodeIDs, 
                      int **sharedProcIDs,
                      int *lenSharedProcIDs);

/* initialize nodal sets for external (off-processor) communcation */
int FEI_initExtNodeSet(int sysHandle, 
                   GlobalID *extNodeIDs,
                   int lenExtNodeIDs, 
                   int **extProcIDs, 
                   int *lenExtProcIDs);

/* end node set initialization */
int FEI_endInitNodeSets(int sysHandle);


/* begin constraint relation set initialization step.........*/
int FEI_beginInitCREqns(int sysHandle, 
                    int numCRMultSets,
                    int numCRPenSets);

/* constraint relation initialization - lagrange multiplier formulation */
int FEI_initCRMult(int sysHandle, 
               GlobalID **CRNodeTable,  
               int *CRFieldList,
		       int numMultCRs, 
		       int lenCRNodeList,
		       int* CRMultID); 

/* constraint relation initialization - penalty function formulation */
int FEI_initCRPen(int sysHandle, 
              GlobalID **CRNodeTable, 
              int *CRFieldList,
              int numPenCRs, 
              int lenCRNodeList, 
              int* CRPenID); 

/* end constraint relation list initialization */
int FEI_endInitCREqns(int sysHandle);

/* indicate that overall initialization sequence is complete */
int FEI_initComplete(int sysHandle);

/* FE data load sequence..........................................*/

/* set a value (usually zeros) througout the linear system.....*/
int FEI_resetSystem(int sysHandle, 
                double s);

/* begin node-set data load step.............................*/
int FEI_beginLoadNodeSets(int sysHandle, 
                      int numBCNodeSets);

/* boundary condition data load step */
int FEI_loadBCSet(int sysHandle, 
              GlobalID *BCNodeSet,
              int lenBCNodeSet,
              int BCFieldID,
              double **alphaBCDataTable,
              double **betaBCDataTable,
              double **gammaBCDataTable);

/* end node-set data load step */
int FEI_endLoadNodeSets(int sysHandle);

/* begin blocked-element data loading step....................*/
int FEI_beginLoadElemBlock(int sysHandle, 
                       GlobalID elemBlockID,
                       int numElemSets,
                       int numElemTotal);
  
/* elemSet-based stiffness/rhs data loading step */
int FEI_loadElemSet(int sysHandle, 
                int elemSetID, 
                int numElems, 
                GlobalID *elemIDs,  
                GlobalID **elemConn,
                double ***elemStiffness,
                double **elemLoad,
                int elemFormat);

/* end blocked-element data loading step*/
int FEI_endLoadElemBlock(int sysHandle);


/* begin constraint relation data load step...................*/
int FEI_beginLoadCREqns(int sysHandle, 
                    int numCRMultSets,
                    int numCRPenSets);

/* lagrange-multiplier constraint relation load step */
int FEI_loadCRMult(int sysHandle, 
               int CRMultID, 
               int numMultCRs,
               GlobalID **CRNodeTable,  
               int *CRFieldList,
               double **CRWeightTable,
               double *CRValueList,
               int lenCRNodeList);

/* penalty formulation constraint relation load step */
int FEI_loadCRPen(int sysHandle, 
              int CRPenID,
              int numPenCRs, 
              GlobalID **CRNodeTable,
              int *CRFieldList,
              double **CRWeightTable,  
              double *CRValueList,
              double *penValues,
              int lenCRNodeList);

/* end constraint relation data load step */
int FEI_endLoadCREqns(int sysHandle);

/* indicate that overall data loading sequence is complete */
int FEI_loadComplete(int sysHandle);

/* Equation solution services..................................... */

/* set parameters associated with solver choice, etc. */
void FEI_parameters(int sysHandle, 
                int numParams, 
                char **paramStrings);

/* start iterative solution */
int FEI_iterateToSolve(int sysHandle, int* status);

/* Solution return services....................................... */
 
/* return nodal-based solution to FE analysis on a block-by-block basis */
int FEI_getBlockNodeSolution(int sysHandle, 
                         GlobalID elemBlockID,
                         GlobalID *nodeIDList, 
                         int* lenNodeIDList, 
                         int *offset,
                         double *answers);

/* return field-based solution to FE analysis on a block-by-block basis */
int FEI_getBlockFieldNodeSolution(int sysHandle, 
                              GlobalID elemBlockID,
                              int fieldID,
                              GlobalID *nodeIDList, 
                              int* lenNodeIDList, 
                              int *offset,
                              double *results);

/* return element-based solution to FE analysis on a block-by-block basis */
int FEI_getBlockElemSolution(int sysHandle, 
                         GlobalID elemBlockID,
                         GlobalID *elemIDList, 
                         int* lenElemIDList, 
                         int *offset,
                         double *results, 
                         int* numElemDOF);

/* return Lagrange solution to FE analysis on a whole-processor basis */
int FEI_getCRMultSolution(int sysHandle, 
                      int* numCRMultSets, 
                      int *CRMultIDs,
                      int *offset, 
                      double *results);

/* return Lagrange solution to FE analysis on a constraint-set basis */
int FEI_getCRMultParam(int sysHandle, 
                   int CRMultID, 
                   int numMultCRs,
                   double *multValues); 

/* put nodal-based solution to FE analysis on a block-by-block basis */
int FEI_putBlockNodeSolution(int sysHandle, 
                         GlobalID elemBlockID,  
                         GlobalID *nodeIDList, 
                         int lenNodeIDList, 
                         const int *offset,  
                         const double *estimates);

/* put field-based solution to FE analysis on a block-by-block basis */
int FEI_putBlockFieldNodeSolution(int sysHandle, 
                              GlobalID elemBlockID,  
                              int fieldID, 
                              GlobalID *nodeIDList, 
                              int lenNodeIDList, 
                              int *offset,  
                              double *estimates);
         
/*  put element-based solution to FE analysis on a block-by-block basis */ 
int FEI_putElemBlockSolution(int sysHandle, 
                         GlobalID elemBlockID,  
                         GlobalID *elemIDList, 
                         int lenElemIDList, 
                         int *offset,  
                         double *estimates, 
                         int numElemDOF);
  
/*  put Lagrange solution to FE analysis on a constraint-set basis */
int FEI_putCRMultParam(int sysHandle, 
                   int CRMultID, 
                   int numMultCRs,
                   double *multEstimates);
 
/*  some utility functions to aid in using the "put" functions for passing */
/*  an initial guess to the solver */

/* return sizes associated with Lagrange solution to FE analysis */
int FEI_getCRMultSizes(int sysHandle, 
                   int* numCRMultIDs, 
                   int* lenAnswers);
 
/* form some data structures needed to return element solution parameters */
int FEI_getBlockElemIDList(int sysHandle, 
                       GlobalID elemBlockID, 
	                   GlobalID *elemIDList, 
	                   int *lenElemIDList);

/* form some data structures needed to return nodal solution parameters */
int FEI_getBlockNodeIDList(int sysHandle, 
                       GlobalID elemBlockID,
                       GlobalID *nodeIDList, 
                       int *lenNodeIDList);

/*  return the number of solution parameters at a given node */
int FEI_getNumSolnParams(int sysHandle, 
                     GlobalID globalNodeID);

/*  return the number of stored element blocks */
int FEI_getNumElemBlocks(int sysHandle);

/*  return the number of active nodes in a given element block */
int FEI_getNumBlockActNodes(int sysHandle, 
                        GlobalID blockID);

/*  return the number of active equations in a given element block */
int FEI_getNumBlockActEqns(int sysHandle, 
                       GlobalID blockID);

/*  return the number of nodes associated with elements of a
    given block ID */
int FEI_getNumNodesPerElement(int sysHandle, 
                          GlobalID blockID);

/*  return the number of eqns associated with elements of a
    given block ID */
int FEI_getNumEqnsPerElement(int sysHandle, 
                         GlobalID blockID);
 
/*  return the number of elements in a given element block */
int FEI_getNumBlockElements(int sysHandle, 
                        GlobalID blockID);

/*  return the number of element equations in a given element block */
int FEI_getNumBlockElemEqns(int sysHandle, 
                        GlobalID blockID);

/*  load the constraint numbers to the LinSysCore */
int FEI_loadConstraintNumbers(int sysHandle, int leng, int *list);

/*  build reduced system */
int FEI_buildReducedSystem(int sysHandle);

#endif
