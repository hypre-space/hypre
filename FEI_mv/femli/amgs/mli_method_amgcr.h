/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the Compatible relaxation-based multilevel method
 *
 *****************************************************************************/

#ifndef __MLIMETHODAMGCRH__
#define __MLIMETHODAMGCRH__

#include "utilities/utilities.h"
#include "parcsr_mv/parcsr_mv.h"
#include "base/mli.h"
#include "matrix/mli_matrix.h"
#include "amgs/mli_method.h"

/* ***********************************************************************
 * definition of the classical Ruge Stuben AMG data structure
 * ----------------------------------------------------------------------*/

class MLI_Method_AMGCR : public MLI_Method
{
   int      maxLevels_;
   int      numLevels_;
   int      currLevel_;
   int      outputLevel_;
   int      findMIS_;
   int      numTrials_;
   int      numVectors_;
   int      minCoarseSize_;
   double   cutThreshold_;
   double   targetMu_;
   char     smoother_[20];
   int      smootherNum_;
   double   *smootherWgts_;
   char     coarseSolver_[20];
   int      coarseSolverNum_;
   double   *coarseSolverWgts_;
   double   RAPTime_;
   double   totalTime_;
   char     paramFile_[50];
   int      PDegree_;

public :

   MLI_Method_AMGCR(MPI_Comm comm);
   ~MLI_Method_AMGCR();
   int    setup( MLI *mli );
   int    setParams(char *name, int argc, char *argv[]);

   int    setOutputLevel(int outputLevel);
   int    setNumLevels(int nlevels);
   int    selectIndepSet(MLI_Matrix *, int **indepSet);
   MLI_Matrix *performCR(MLI_Matrix *, int *indepSet, MLI_Matrix **);
   MLI_Matrix *createPmat(int *indepSet,MLI_Matrix *,MLI_Matrix *,MLI_Matrix *);
   MLI_Matrix *createRmat(int *indepSet, MLI_Matrix *, MLI_Matrix *);
   int    print();
   int    printStatistics(MLI *mli);
};

#endif

