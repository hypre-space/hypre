#ifndef __fei_proc_H
#define __fei_proc_H

/*
   This is the header for the prototypes of the procedural version
   of the finite element interface.
*/

/* Initialization function. Specifies how many linear systems will
   be being worked with. */
void numLinearSystems(int numSystems, 
                      MPI_Comm FEI_COMM_WORLD, 
                      int masterRank);

/* Like a destructor function, gets rid of allocated memory. */
void destroyAllLinearSystems(void);

/* Structural initialization sequence.............................*/

/* per-solve-step initialization */
int initSolveStep(int sysHandle, 
                  int numElemBlocks, 
                  int solveType);

/* identify all the solution fields present in the analysis.......*/
int initFields(int sysHandle, 
               int numFields, 
               int *cardFields, 
               int *fieldIDs); 

/* begin blocked-element initialization step..................*/
int beginInitElemBlock(int sysHandle, 
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
int initElemSet(int sysHandle, 
                int numElems,
                GlobalID *elemIDs, 
                GlobalID **elemConn);

/* end blocked-element initialization */
int endInitElemBlock(int sysHandle);

/* begin collective node set initialization step........................*/
int beginInitNodeSets(int sysHandle, 
                      int numSharedNodeSets, 
                      int numExtNodeSets);

/* initialize nodal sets for shared nodes */
int initSharedNodeSet(int sysHandle, 
                      GlobalID *sharedNodeIDs,
                      int lenSharedNodeIDs, 
                      int **sharedProcIDs,
                      int *lenSharedProcIDs);

/* initialize nodal sets for external (off-processor) communcation */
int initExtNodeSet(int sysHandle, 
                   GlobalID *extNodeIDs,
                   int lenExtNodeIDs, 
                   int **extProcIDs, 
                   int *lenExtProcIDs);

/* end node set initialization */
int endInitNodeSets(int sysHandle);


/* begin constraint relation set initialization step.........*/
int beginInitCREqns(int sysHandle, 
                    int numCRMultSets,
                    int numCRPenSets);

/* constraint relation initialization - lagrange multiplier formulation */
int initCRMult(int sysHandle, 
               GlobalID **CRNodeTable,  
               int *CRFieldList,
		       int numMultCRs, 
		       int lenCRNodeList,
		       int* CRMultID); 

/* constraint relation initialization - penalty function formulation */
int initCRPen(int sysHandle, 
              GlobalID **CRNodeTable, 
              int *CRFieldList,
              int numPenCRs, 
              int lenCRNodeList, 
              int* CRPenID); 

/* end constraint relation list initialization */
int endInitCREqns(int sysHandle);

/* indicate that overall initialization sequence is complete */
int initComplete(int sysHandle);

/* FE data load sequence..........................................*/

/* set a value (usually zeros) througout the linear system.....*/
int resetSystem(int sysHandle, 
                double s);

/* begin node-set data load step.............................*/
int beginLoadNodeSets(int sysHandle, 
                      int numBCNodeSets);

/* boundary condition data load step */
int loadBCSet(int sysHandle, 
              GlobalID *BCNodeSet,
              int lenBCNodeSet,
              int BCFieldID,
              double **alphaBCDataTable,
              double **betaBCDataTable,
              double **gammaBCDataTable);

/* end node-set data load step */
int endLoadNodeSets(int sysHandle);

/* begin blocked-element data loading step....................*/
int beginLoadElemBlock(int sysHandle, 
                       GlobalID elemBlockID,
                       int numElemSets,
                       int numElemTotal);
  
/* elemSet-based stiffness/rhs data loading step */
int loadElemSet(int sysHandle, 
                int elemSetID, 
                int numElems, 
                GlobalID *elemIDs,  
                GlobalID **elemConn,
                double ***elemStiffness,
                double **elemLoad,
                int elemFormat);

/* end blocked-element data loading step*/
int endLoadElemBlock(int sysHandle);


/* begin constraint relation data load step...................*/
int beginLoadCREqns(int sysHandle, 
                    int numCRMultSets,
                    int numCRPenSets);

/* lagrange-multiplier constraint relation load step */
int loadCRMult(int sysHandle, 
               int CRMultID, 
               int numMultCRs,
               GlobalID **CRNodeTable,  
               int *CRFieldList,
               double **CRWeightTable,
               double *CRValueList,
               int lenCRNodeList);

/* penalty formulation constraint relation load step */
int loadCRPen(int sysHandle, 
              int CRPenID,
              int numPenCRs, 
              GlobalID **CRNodeTable,
              int *CRFieldList,
              double **CRWeightTable,  
              double *CRValueList,
              double *penValues,
              int lenCRNodeList);

/* end constraint relation data load step */
int endLoadCREqns(int sysHandle);

/* indicate that overall data loading sequence is complete */
int loadComplete(int sysHandle);

/* Equation solution services..................................... */

/* set parameters associated with solver choice, etc. */
void parameters(int sysHandle, 
                int numParams, 
                char **paramStrings);

/* start iterative solution */
int iterateToSolve(int sysHandle);

/* Solution return services....................................... */
 
/* return nodal-based solution to FE analysis on a block-by-block basis */
int getBlockNodeSolution(int sysHandle, 
                         GlobalID elemBlockID,
                         GlobalID *nodeIDList, 
                         int* lenNodeIDList, 
                         int *offset,
                         double *answers);

/* return field-based solution to FE analysis on a block-by-block basis */
int getBlockFieldNodeSolution(int sysHandle, 
                              GlobalID elemBlockID,
                              int fieldID,
                              GlobalID *nodeIDList, 
                              int* lenNodeIDList, 
                              int *offset,
                              double *results);

/* return element-based solution to FE analysis on a block-by-block basis */
int getBlockElemSolution(int sysHandle, 
                         GlobalID elemBlockID,
                         GlobalID *elemIDList, 
                         int* lenElemIDList, 
                         int *offset,
                         double *results, 
                         int* numElemDOF);

/* return Lagrange solution to FE analysis on a whole-processor basis */
int getCRMultSolution(int sysHandle, 
                      int* numCRMultSets, 
                      int *CRMultIDs,
                      int *offset, 
                      double *results);

/* return Lagrange solution to FE analysis on a constraint-set basis */
int getCRMultParam(int sysHandle, 
                   int CRMultID, 
                   int numMultCRs,
                   double *multValues); 

/* put nodal-based solution to FE analysis on a block-by-block basis */
int putBlockNodeSolution(int sysHandle, 
                         GlobalID elemBlockID,  
                         GlobalID *nodeIDList, 
                         int lenNodeIDList, 
                         const int *offset,  
                         const double *estimates);

/* put field-based solution to FE analysis on a block-by-block basis */
int putBlockFieldNodeSolution(int sysHandle, 
                              GlobalID elemBlockID,  
                              int fieldID, 
                              GlobalID *nodeIDList, 
                              int lenNodeIDList, 
                              int *offset,  
                              double *estimates);
         
/*  put element-based solution to FE analysis on a block-by-block basis */ 
int putElemBlockSolution(int sysHandle, 
                         GlobalID elemBlockID,  
                         GlobalID *elemIDList, 
                         int lenElemIDList, 
                         int *offset,  
                         double *estimates, 
                         int numElemDOF);
  
/*  put Lagrange solution to FE analysis on a constraint-set basis */
int putCRMultParam(int sysHandle, 
                   int CRMultID, 
                   int numMultCRs,
                   double *multEstimates);
 
/*  some utility functions to aid in using the "put" functions for passing */
/*  an initial guess to the solver */

/* return sizes associated with Lagrange solution to FE analysis */
int getCRMultSizes(int sysHandle, 
                   int* numCRMultIDs, 
                   int* lenAnswers);
 
/* form some data structures needed to return element solution parameters */
int getBlockElemIDList(int sysHandle, 
                       GlobalID elemBlockID, 
	                   GlobalID *elemIDList, 
	                   int *lenElemIDList);

/* form some data structures needed to return nodal solution parameters */
int getBlockNodeIDList(int sysHandle, 
                       GlobalID elemBlockID,
                       GlobalID *nodeIDList, 
                       int *lenNodeIDList);

/*  return the number of solution parameters at a given node */
int getNumSolnParams(int sysHandle, 
                     GlobalID globalNodeID);

/*  return the number of stored element blocks */
int getNumElemBlocks(int sysHandle);

/*  return the number of active nodes in a given element block */
int getNumBlockActNodes(int sysHandle, 
                        GlobalID blockID);

/*  return the number of active equations in a given element block */
int getNumBlockActEqns(int sysHandle, 
                       GlobalID blockID);

/*  return the number of nodes associated with elements of a
    given block ID */
int getNumNodesPerElement(int sysHandle, 
                          GlobalID blockID);

/*  return the number of eqns associated with elements of a
    given block ID */
int getNumEqnsPerElement(int sysHandle, 
                         GlobalID blockID);
 
/*  return the number of elements in a given element block */
int getNumBlockElements(int sysHandle, 
                        GlobalID blockID);

/*  return the number of element equations in a given element block */
int getNumBlockElemEqns(int sysHandle, 
                        GlobalID blockID);

#endif
