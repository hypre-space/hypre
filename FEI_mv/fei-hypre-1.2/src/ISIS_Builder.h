#ifndef _ISIS_Builder_h_
#define _ISIS_Builder_h_

#include "src/Data.h"

#ifdef FEI_SER
#include <mpiuni/mpi.h>
#else
#include <mpi.h>
#endif

#include "src/LinearSystemCore.h"
#include "src/ISIS_LinSysCore.h"

#include "src/FEI_Implementation.h"

class ISIS_Builder {
 public:
   static FEI* FEIBuilder(MPI_Comm comm, int masterProc) {
      ISIS_LinSysCore* linSysCore = new ISIS_LinSysCore(comm);

      return(new FEI_Implementation(linSysCore, comm, masterProc));
   };
};

#endif

