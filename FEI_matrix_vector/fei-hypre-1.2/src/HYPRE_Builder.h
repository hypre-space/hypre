#ifndef _HYPRE_Builder_h_
#define _IHYPREBuilder_h_

#include "src/Data.h"

#include "utilities/utilities.h"

#include "src/LinearSystemCore.h"
#include "src/HYPRE_LinSysCore.h"

#include "src/FEI_Implementation.h"

class HYPRE_Builder {
 public:
   static FEI* FEIBuilder(HYPRE_LinSysCore *lsCore, MPI_Comm comm, 
                          int masterProc) 
   {
      return(new FEI_Implementation(lsCore, comm, masterProc));
   };
};

#endif

