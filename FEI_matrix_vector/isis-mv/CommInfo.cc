/*
          This file is part of Sandia National Laboratories
          copyrighted software.  You are legally liable for any
          unauthorized use of this software.

          NOTICE:  The United States Government has granted for
          itself and others acting on its behalf a paid-up,
          nonexclusive, irrevocable worldwide license in this
          data to reproduce, prepare derivative works, and
          perform publicly and display publicly.  Beginning five
          (5) years after June 5, 1997, the United States
          Government is granted for itself and others acting on
          its behalf a paid-up, nonexclusive, irrevocable
          worldwide license in this data to reproduce, prepare
          derivative works, distribute copies to the public,
          perform publicly and display publicly, and to permit
          others to do so.

          NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED
          STATES DEPARTMENT OF ENERGY, NOR SANDIA CORPORATION,
          NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS
          OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
          RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
          USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT, OR
          PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
          INFRINGE PRIVATELY OWNED RIGHTS.
*/

#undef NDEBUG 			// make sure asserts are enabled
#include <assert.h>
#include <stdlib.h>
#include <iostream.h>

//#include <mpi.h> Taken out to be HYPRE-compatible
#include "utilities/utilities.h"

#include "CommInfo.h"

//==============================================================================
CommInfo::CommInfo(int masterRank, MPI_Comm COMM_WORLD) : 
    masterRank_(masterRank), COMM_WORLD_(COMM_WORLD)
{
// CommInfo::CommInfo -- construct the parallel communications object.

    MPI_Comm_rank(COMM_WORLD_, &localRank_);
    MPI_Comm_size(COMM_WORLD_, &numProcessors_);

    if (localRank_ < 0) {
        cerr << "CommInfo::CommInfo: ERROR, localRank_ < 0. Aborting." << endl;
        abort();
    }

    if (numProcessors_ <= 0) {
        cerr << "CommInfo::CommInfo: ERROR, numProcessors_ <= 0. Aborting."
             << endl;
        abort();
    }
}

//==============================================================================
CommInfo::CommInfo(const CommInfo& commInfo) {
//
//Copy constructor.
//
    masterRank_ = commInfo.masterRank();
    localRank_ = commInfo.localRank();
    numProcessors_ = commInfo.numProcessors();
    COMM_WORLD_ = commInfo.getCommunicator();
}

