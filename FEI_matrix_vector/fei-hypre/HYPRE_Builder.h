#ifndef _HYPRE_Builder_h_
#define _HYPRE_Builder_h_

#include "base/Data.h"

#include "utilities/utilities.h"

#include "base/LinearSystemCore.h"
#include "HYPRE_LinSysCore.h"

#include "base/FEI_Implementation.h"

class HYPRE_Builder {
 public:
   static FEI* FEIBuilder(MPI_Comm comm, int masterProc) {
      HYPRE_LinSysCore* linSysCore = new HYPRE_LinSysCore(comm);

      return(new FEI_Implementation(linSysCore, comm, masterProc));
   }
};

#endif

