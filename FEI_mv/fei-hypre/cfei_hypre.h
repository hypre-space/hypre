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

int HYPRE_LSC_BeginMappedMatrixLoad(LinSysCore* lsc);

int HYPRE_LSC_EndMappedMatrixLoad(LinSysCore* lsc);

int HYPRE_LSC_MappedMatrixLoad(LinSysCore* lsc, int row, int col, double val);

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

#ifdef __cplusplus
}
#endif

#endif

