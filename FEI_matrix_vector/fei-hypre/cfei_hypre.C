#include <string.h>
#include <stdlib.h>

#include "utilities/utilities.h"


#include "basicTypes.h"
#include "Data.h"
#if defined(FEI_V13)
#include "LinearSystemCore.1.3.h"
#elseif defined(FEI_V14)
#include "LinearSystemCore.1.4.h"
#else
#include "LinearSystemCore.h"
#include "LSC.h"
#endif
#include "cfei.h"

#include "cfei_hypre.h"
#include "HYPRE.h"
#include "../../IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "../../parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "../../parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "HYPRE_LinSysCore.h"
#include "fegridinfo.h"

/******************************************************************************/
/* Create function for a HYPRE_LinSysCore object.                             */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LinSysCore_create(LinSysCore** lsc, MPI_Comm comm) 
{
   HYPRE_LinSysCore* linSys = new HYPRE_LinSysCore(comm);

   if (linSys == NULL) return(1);

   *lsc = new LinSysCore;

   if (*lsc == NULL) return(1);

   (*lsc)->lsc_ = (void*)linSys;

   return(0);
}

/******************************************************************************/
/* Destroy function, to de-allocate a HYPRE_LinSysCore object.                */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LinSysCore_destroy(LinSysCore** lsc) 
{
   if (*lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)((*lsc)->lsc_);

   if (linSys == NULL) return(1);

   delete linSys;

   delete *lsc;
   *lsc = NULL;

   return(0);
}

/******************************************************************************/
/* function for loading to the matrix directly but with FEI equation mappings */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_BeginMappedMatrixLoad(LinSysCore* lsc) 
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->beginCreateMapFromSoln();

   return(0);
}

/******************************************************************************/
/* function for loading to the matrix directly but with FEI equation mappings */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_EndMappedMatrixLoad(LinSysCore* lsc) 
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->endCreateMapFromSoln();

   return(0);
}

/******************************************************************************/
/* function for loading to the matrix directly but with FEI equation mappings */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_MappedMatrixLoad(LinSysCore* lsc, int row,
                int col, double val) 
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->putIntoMappedMatrix(row, 1, &val, &col);

   return(0);
}

/******************************************************************************/
/******************************************************************************/
/* functions for interfacing to a finite element object                       */
/******************************************************************************/

/******************************************************************************/
/* get the finite element grid object                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_GetFEGridObject(LinSysCore* lsc, void **object)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->getFEGridObject(object);

   return(0);
}

/******************************************************************************/
/* begin initializing the element set                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEGrid_beginInitElemSet(void *grid, int nElems, int *gid)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->beginInitElemSet(nElems, gid);

   return(0);
}

/******************************************************************************/
/* terminate initializing the element set                                     */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEGrid_endInitElemSet(void *grid)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->endInitElemSet();

   return(0);
}

/******************************************************************************/
/* load element connectivity information                                      */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEGrid_loadElemSet(void *grid, int elemID, int nNodes,
                            int *nList, int sDim, double **sMat)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->loadElemSet(elemID, nNodes, nList, sDim, sMat);

   return(0);
}

/******************************************************************************/
/* begin loading the node information                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEGrid_beginInitNodeSet(void *grid)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->beginInitNodeSet();

   return(0);
}

/******************************************************************************/
/* terminate loading the node information                                     */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEGrid_endInitNodeSet(void *grid)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->endInitNodeSet();

   return(0);
}

/******************************************************************************/
/* set the node degree of freedom                                             */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEGrid_loadNodeDOF(void *grid, int nodeID, int dof)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->loadNodeDOF(nodeID, dof);

   return(0);
}

/******************************************************************************/
/* load node essential boundary conditions                                    */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEGrid_loadNodeEssBCs(void *grid, int nNodes, int *nList,
                            int *dofList, double *val)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->loadNodeEssBCs(nNodes, nList, dofList, val);

   return(0);
}

/******************************************************************************/
/* load shared nodes                                                          */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEGrid_loadSharedNodes(void *grid, int nNodes, int *nList,
                            int *procLeng, int **nodeProc)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->loadSharedNodes(nNodes, nList, procLeng, nodeProc);

   return(0);
}

/******************************************************************************/
/******************************************************************************/
/* Wrapper functions for the HYPRE LSI                                        */
/******************************************************************************/

/******************************************************************************/
/* the parameter function (to set parameter values)                           */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_parameters(LinSysCore* lsc, int numParams, char **params) 
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->parameters(numParams, params);

   return(0);
}

/******************************************************************************/
/* This function sets up the equation offset on each processor                */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_setGlobalOffsets(LinSysCore* lsc, int leng, int* nodeOffsets, 
                                      int* eqnOffsets, int* blkEqnOffsets)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->setGlobalOffsets(leng, nodeOffsets, eqnOffsets, blkEqnOffsets);

   return(0);
}

/******************************************************************************/
/* set up the matrix sparsity pattern                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_setMatrixStructure(LinSysCore *lsc, int** ptColIndices, 
                     int* ptRowLengths, int** blkColIndices, int* blkRowLengths, 
                     int* ptRowsPerBlkRow)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->setMatrixStructure(ptColIndices, ptRowLengths, blkColIndices, 
                              blkRowLengths, ptRowsPerBlkRow);

   return(0);
}

/******************************************************************************/
/* reset the matrix but keep the sparsity pattern                             */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_resetMatrixAndVector(LinSysCore *lsc, double val)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->resetMatrixAndVector(val);

   return(0);
}

/******************************************************************************/
/* reset the matrix but keep the sparsity pattern                             */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_resetMatrix(LinSysCore *lsc, double val)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->resetMatrix(val);

   return(0);
}

/******************************************************************************/
/* reset the right hand side vector                                           */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_resetRHSVector(LinSysCore *lsc, double val)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->resetRHSVector(val);

   return(0);
}

/******************************************************************************/
/* load the matrix                                                            */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_sumIntoSystemMatrix(LinSysCore *lsc, int numPtRows, 
                     const int* ptRows, int numPtCols, const int* ptCols, 
                     int numBlkRows, const int* blkRows, int numBlkCols, 
                     const int* blkCols, const double* const* values)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->sumIntoSystemMatrix(numPtRows, ptRows, numPtCols, ptCols, numBlkRows, 
                               blkRows, numBlkCols, blkCols, values);

   return(0);
}

/******************************************************************************/
/* load the right hand side vector                                            */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_sumIntoRHSVector(LinSysCore *lsc, int num, const double* 
                                      values, const int* indices)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->sumIntoRHSVector(num, values, indices);

   return(0);
}

/******************************************************************************/
/* matrix loading completed                                                   */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_matrixLoadComplete(LinSysCore *lsc)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->matrixLoadComplete();

   return(0);
}

/******************************************************************************/
/* enforce essential boundary condition                                       */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_enforceEssentialBC(LinSysCore *lsc, int* globalEqn,
                     double* alpha, double* gamma, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->enforceEssentialBC(globalEqn, alpha, gamma, leng);

   return(0);
}

/******************************************************************************/
/* enforce essential boundary condition (cross processor boundary)            */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_enforceRemoteEssBCs(LinSysCore *lsc,int numEqns,int* globalEqns,
                             int** colIndices, int* colIndLen, double** coefs)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->enforceRemoteEssBCs(numEqns, globalEqns, colIndices, colIndLen, coefs);

   return(0);
}

/******************************************************************************/
/* enforce natural boundary condition                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_enforceOtherBC(LinSysCore *lsc, int* globalEqn, 
                     double* alpha, double* beta, double* gamma, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->enforceOtherBC(globalEqn, alpha, beta, gamma, leng);

   return(0);
}

/******************************************************************************/
/* put initial guess into HYPRE                                               */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_putInitialGuess(LinSysCore *lsc, const int* eqnNumbers, 
                                     const double* values, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->putInitialGuess(eqnNumbers, values, leng);

   return(0);
}

/******************************************************************************/
/* get the whole solution vector                                              */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_getSolution(LinSysCore *lsc, double *answers, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->getSolution(answers, leng);

   return(0);
}

/******************************************************************************/
/* get a solution entry                                                       */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_getSolnEntry(LinSysCore *lsc, int eqnNumber, double *answer)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->getSolnEntry(eqnNumber, (*answer));

   return(0);
}

/******************************************************************************/
/* form and fetch the residual vector                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_formResidual(LinSysCore *lsc, double *values, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->formResidual(values, leng);

   return(0);
}

/******************************************************************************/
/* start iterating                                                            */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_launchSolver(LinSysCore *lsc, int *solveStatus, int *iter)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->launchSolver(*solveStatus, *iter);

   return(0);
}

