#ifndef __CommInfo_H
#define __CommInfo_H

//==============================================================================
//
//The CommInfo class is a simple container class for holding an
//MPI_Comm object, and also 'caches' the values of localRank
//and masterRank, and numProcessors.
//
//==============================================================================

// requires:
//#include <assert.h>
//#include <stdlib.h>
//#include <iostream.h>
//#include <mpi.h>

class CommInfo  {
    
  public:
    CommInfo(int masterRank, MPI_Comm COMM_WORLD);
    CommInfo(const CommInfo& commInfo); //copy constructor.
    virtual ~CommInfo () {};

    // Functions for getting values.
    int masterRank() const {return masterRank_;};
    int localRank() const {return localRank_;};
    int numProcessors() const {return numProcessors_;};
    MPI_Comm getCommunicator() const {return(COMM_WORLD_);};

  private:
    // Parallel communications info.
    int masterRank_;            // rank for the master processor
    int localRank_;             // rank for this processor
    int numProcessors_;         // total number of processors
    MPI_Comm COMM_WORLD_;
};

#endif
