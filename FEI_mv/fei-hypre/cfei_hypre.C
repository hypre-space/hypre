#include <stdio.h>
#include <iostream.h>
#include <stdlib.h>

#include "utilities/utilities.h"


#include "base/basicTypes.h"
#include "base/Data.h"
#include "base/LinearSystemCore.h"

#include "base/cfei.h"
#include "cfei_hypre.h"

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

