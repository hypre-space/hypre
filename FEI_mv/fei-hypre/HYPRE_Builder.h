#ifndef _HYPRE_Builder_h_
#define _HYPRE_Builder_h_

#include "Data.h"

#include "utilities/utilities.h"

#include "LinearSystemCore.h"
#include "HYPRE.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../parcsr_mv/HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_LinSysCore.h"

#include "FEI_Implementation.h"

class HYPRE_Builder {
 public:
   static FEI* FEIBuilder(MPI_Comm comm, int masterProc) {
      HYPRE_LinSysCore* linSysCore = new HYPRE_LinSysCore(comm);

      return(new FEI_Implementation(linSysCore, comm, masterProc));
   }
};

#endif

