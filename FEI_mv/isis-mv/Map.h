#ifndef __Map_H
#define __Map_H

//==============================================================================
//
//The Map class is a data layout descriptor -- it describes the global
//size of (one dimension of) a problem, and also identifies which portions
//of the problem lie on which processors.
//
//Data distributions are assumed to consist of contiguous 'chunks'.
//i.e., the section of a vector (or 'stripe' of a matrix) that is local,
//is assumed to have a contiguous set of indices. Thus a processor's
//portion of a problem may be described by a 'localStart' and 'localEnd'
//index pair.
//
//==============================================================================

//requires:
//#include <assert.h>
//#include <iostream.h>
//#include <stdlib.h>
//#include "other/basicTypes.h"
//#include <mpi.h>
//#include "mv/CommInfo.h"

class Map  {
    
  public:
    Map(int globalSize); //simple serial constructor.
    Map(int globalSize, int localStart, int localEnd,
        const CommInfo& commInfo);
    Map(const Map& map);        // copy constructor    
    virtual ~Map(void);

    // Functions for getting values.
    int globalSize(void) const {return globalSize_;};
    const CommInfo& getCommInfo() const;
 
    int localStart(void) const {return localStart_;};
    int localEnd(void) const {return localEnd_;};
    int localSize(void) const {return localSize_;};

    //globalStart and globalEnd return integer arrays of size number-of-procs,
    //which contain each processor's localStart and localEnd, respectively.
    int* globalStart(void) const {return globalStart_;};
    int* globalEnd(void) const {return globalEnd_;};

  private:

    int globalSize_;       // total size of the global vector or
                           // matrix-dimension

    int localStart_;       // lowest local index
    int localEnd_;         // highest local index
    int localSize_;        // number of local indices
    int* globalStart_;     // arrays of length numProcessors which hold
    int* globalEnd_;       // each proc's start- and end- indices.

    bool serial_;
    const CommInfo *commInfo_;  // communciations info data object
    CommInfo *sCommInfo_;

};

#endif
