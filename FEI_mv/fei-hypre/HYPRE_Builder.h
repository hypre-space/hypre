#ifndef _HYPRE_Builder_h_
#define _HYPRE_Builder_h_

#include "Data.h"

#include "utilities/utilities.h"

#include "LinearSystemCore.h"
#include "HYPRE.h"
#include "../../IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "../../parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "../../parcsr_linear_solvers/HYPRE_parcsr_ls.h"
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

