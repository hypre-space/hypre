#include <string.h>
#include <stdlib.h>

#include "utilities/utilities.h"


#include "base/basicTypes.h"
#include "base/Data.h"
#if defined(FEI_V13) || defined(FEI_V14)
#include "base1.4/LinearSystemCore.h"
#else
#include "base/LinearSystemCore.h"
#include "base/LSC.h"
#endif
#include "base/cfei.h"

#include "cfei_hypre.h"
#include "HYPRE.h"
#include "../../IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "../../parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "../../parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "HYPRE_LinSysCore.h"
#include "fegridinfo.h"

/*============================================================================*/
/* Create function for a ISIS_LinSysCore object.
*/
extern "C" int HYPRE_LinSysCore_create(LinSysCore** lsc, 
                                      MPI_Comm comm) {

   HYPRE_LinSysCore* linSys = new HYPRE_LinSysCore(comm);

   if (linSys == NULL) return(1);

   *lsc = new LinSysCore;

   if (*lsc == NULL) return(1);

   (*lsc)->lsc_ = (void*)linSys;

   return(0);
}

/*============================================================================*/
/* Destroy function, to de-allocate a ISIS_LinSysCore object.
*/
extern "C" int HYPRE_LinSysCore_destroy(LinSysCore** lsc) {

   if (*lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)((*lsc)->lsc_);

   if (linSys == NULL) return(1);

   delete linSys;

   delete *lsc;
   *lsc = NULL;

   return(0);
}
/*============================================================================*/
/*new function */
extern "C" int HYPRE_LSC_BeginMappedMatrixLoad(LinSysCore* lsc) {

   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->beginCreateMapFromSoln();

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_LSC_EndMappedMatrixLoad(LinSysCore* lsc) {

   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->endCreateMapFromSoln();

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_LSC_MappedMatrixLoad(LinSysCore* lsc, int row,
                int col, double val) 
{

   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->putIntoMappedMatrix(row, 1, &val, &col);

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_LSC_GetFEGridObject(LinSysCore* lsc, void **object)
{

   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->getFEGridObject(object);

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_FEGrid_beginInitElemSet(void *grid, int nElems, int *gid)
{

   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->beginInitElemSet(nElems, gid);

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_FEGrid_endInitElemSet(void *grid)
{

   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->endInitElemSet();

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_FEGrid_loadElemSet(void *grid, int elemID, int nNodes,
                            int *nList, int sDim, double **sMat)
{

   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->loadElemSet(elemID, nNodes, nList, sDim, sMat);

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_FEGrid_beginInitNodeSet(void *grid)
{

   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->beginInitNodeSet();

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_FEGrid_endInitNodeSet(void *grid)
{

   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->endInitNodeSet();

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_FEGrid_loadNodeDOF(void *grid, int nodeID, int dof)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->loadNodeDOF(nodeID, dof);

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_FEGrid_loadNodeEssBCs(void *grid, int nNodes, int *nList,
                            int *dofList, double *val)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->loadNodeEssBCs(nNodes, nList, dofList, val);

   return(0);
}

/*============================================================================*/
/*new function */

extern "C" int HYPRE_FEGrid_loadSharedNodes(void *grid, int nNodes, int *nList,
                            int *procLeng, int **nodeProc)
{
   if (grid == NULL) return(1);

   FEGridInfo* fegrid = (FEGridInfo*) grid;

   fegrid->loadSharedNodes(nNodes, nList, procLeng, nodeProc);

   return(0);
}

