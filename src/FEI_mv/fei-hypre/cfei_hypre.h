#ifndef _cfei_hypre_h_
#define _cfei_hypre_h_

/*
   This header defines the prototype for the HYPRE-specific function that
   creates the LinSysCore struct pointer, which is used by FEI_create.
*/

#ifdef __cplusplus
extern "C" {
#endif

int HYPRE_LinSysCore_create(LinSysCore** lsc, MPI_Comm comm);

int HYPRE_LinSysCore_destroy(LinSysCore** lsc);

int HYPRE_LSC_BeginMappedMatrixLoad(LinSysCore* lsc);

int HYPRE_LSC_EndMappedMatrixLoad(LinSysCore* lsc);

int HYPRE_LSC_MappedMatrixLoad(LinSysCore* lsc, int row, int col, double val);

char *HYPRE_LSC_GetVersion(LinSysCore* lsc);

int HYPRE_LSC_GetFEGridObject(LinSysCore* lsc, void **object);

int HYPRE_FEGrid_beginInitElemSet(void *grid, int nElems, int *gid);

int HYPRE_FEGrid_endInitElemSet(void *grid);

int HYPRE_FEGrid_loadElemSet(void *grid, int elemID, int nNodes,
                             int *nList, int sDim, double **sMat);

int HYPRE_FEGrid_beginInitNodeSet(void *grid);

int HYPRE_FEGrid_endInitNodeSet(void *grid);

int HYPRE_FEGrid_loadNodeDOF(void *grid, int nodeID, int dof);

int HYPRE_FEGrid_loadNodeEssBCs(void *grid, int nNodes, int *nList,
                                int *dofList, double *val);

int HYPRE_FEGrid_loadSharedNodes(void *grid, int nNodes, int *nList,
                                 int *procLeng, int **nodeProc);

int HYPRE_parameters(LinSysCore* lsc, int numParams, char **params);

int HYPRE_setGlobalOffsets(LinSysCore* lsc, int leng, int* nodeOffsets,
                           int* eqnOffsets, int* blkEqnOffsets);

int HYPRE_setMatrixStructure(LinSysCore *lsc, int** ptColIndices,
                     int* ptRowLengths, int** blkColIndices, int* blkRowLengths,
                     int* ptRowsPerBlkRow);

int HYPRE_resetMatrixAndVector(LinSysCore *lsc, double val);

int HYPRE_resetMatrix(LinSysCore *lsc, double val);

int HYPRE_resetRHSVector(LinSysCore *lsc, double val);

int HYPRE_sumIntoSystemMatrix(LinSysCore *lsc, int numPtRows,
                     const int* ptRows, int numPtCols, const int* ptCols,
                     int numBlkRows, const int* blkRows, int numBlkCols,
                     const int* blkCols, const double* const* values);

int HYPRE_sumIntoRHSVector(LinSysCore *lsc, int num, const double* values, 
                           const int* indices);

int HYPRE_matrixLoadComplete(LinSysCore *lsc);

int HYPRE_enforceEssentialBC(LinSysCore *lsc, int* globalEqn, double* alpha, 
                             double* gamma, int leng);

int HYPRE_enforceRemoteEssBCs(LinSysCore *lsc,int numEqns,int* globalEqns,
                             int** colIndices, int* colIndLen, double** coefs);

int HYPRE_enforceOtherBC(LinSysCore *lsc, int* globalEqn, double* alpha, 
                         double* beta, double* gamma, int leng);

int HYPRE_putInitialGuess(LinSysCore *lsc, const int* eqnNumbers,
                          const double* values, int leng);

int HYPRE_getSolution(LinSysCore *lsc, double *answers, int leng);

int HYPRE_getSolnEntry(LinSysCore *lsc, int eqnNumber, double *answer);

int HYPRE_formResidual(LinSysCore *lsc, double *values, int leng);

int HYPRE_launchSolver(LinSysCore *lsc, int *solveStatus, int *iter);

#ifdef __cplusplus
}
#endif

#endif

