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

