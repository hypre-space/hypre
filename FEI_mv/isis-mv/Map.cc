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

#undef NDEBUG             // make sure asserts are enabled
#include <assert.h>
#include <iostream.h>
#include <stdlib.h>

#include "other/basicTypes.h" // needed for definition of bool

#include <mpi.h>
#include "CommInfo.h"
#include "Map.h"

/**=========================================================================**/
Map::Map(int globalSize) : globalSize_(globalSize), localStart_(1),
                        localEnd_(globalSize), localSize_(globalSize) {
//
//Serial constructor. This constructs a Map to describe a single-processor
//problem of size globalSize. All necessary internal variables will be
//set so that the user of this map class may still call all member functions
//and obtain meaningful results.
//

    if (globalSize <= 0) {
        cerr << "Map::Map: ERROR, globalSize <= 0." << " Aborting." << endl;
        abort();
    }

    serial_ = true;
    sCommInfo_ = new CommInfo(0, 0);

    globalStart_ = new int[1];
    globalEnd_ = new int[1];
    globalStart_[0] = localStart_;
    globalEnd_[0] = localEnd_;
}

/**=========================================================================**/
Map::Map(int globalSize, int localStart, int localEnd,
         const CommInfo& commInfo) : 
    globalSize_(globalSize),
    localStart_(localStart),
    localEnd_(localEnd),
    localSize_(localEnd - localStart + 1) {
//
//Parallel constructor. This constructs a Map to describe a problem-dimension
//of global size 'globalSize', of which indices localStart..localEnd lie on
//this processor.
//The CommInfo argument contains the MPI_Communicator and associated info.
//

    if (globalSize <= 0) {
        cerr << "Map::Map: ERROR, globalSize <= 0." << " Aborting." << endl;
        abort();
    }

    if ((localStart <= 0) || (localStart > localEnd)) {
        cerr << "Map::Map: ERROR, localStart <= 0 or localStart > localEnd."
             << " Aborting." << endl;
        abort();
    }

    if (localEnd > globalSize) {
        cerr << "Map::Map: ERROR, localEnd > globalSize."
             << " Aborting." << endl;
        abort();
    }

    int numProcs = commInfo.numProcessors();
    globalStart_ = new int[numProcs];
    globalEnd_ = new int[numProcs];

    commInfo_ = &commInfo;
    sCommInfo_ = NULL;
    serial_ = false;

    MPI_Comm thisComm = commInfo.getCommunicator();

    MPI_Allgather(&localStart_, 1, MPI_INT, globalStart_, 1, MPI_INT,
                  thisComm);
    MPI_Allgather(&localEnd_, 1, MPI_INT, globalEnd_, 1, MPI_INT,
                  thisComm);

    return;
}

/**=========================================================================**/
Map::Map (const Map& map) :
    globalSize_(map.globalSize()), 
    localStart_(map.localStart()), 
    localEnd_(map.localEnd()),
    localSize_(map.localEnd() - map.localStart() + 1)
{
    
// Map::Map -- copy constructor.

    commInfo_ = (CommInfo*)&map.getCommInfo();

    int numProcs = commInfo_->numProcessors();
    globalStart_ = new int[numProcs];
    globalEnd_ = new int[numProcs];
    sCommInfo_ = NULL;
    serial_ = false;

    MPI_Comm thisComm = commInfo_->getCommunicator();

    MPI_Allgather(&localStart_, 1, MPI_INT, globalStart_, 1, MPI_INT,
                thisComm);
    MPI_Allgather(&localEnd_, 1, MPI_INT, globalEnd_, 1, MPI_INT,
                thisComm);

   return;
}

/**=========================================================================**/
Map::~Map(){

    if (serial_) delete sCommInfo_;
    delete [] globalStart_;
    delete [] globalEnd_;

    return;
}
 
/**=========================================================================**/
const CommInfo& Map::getCommInfo() const {

    if (serial_) {
        return(*sCommInfo_);
    }

    return(*commInfo_);
}
 
