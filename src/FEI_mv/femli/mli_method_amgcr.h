/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the Compatible relaxation-based multilevel method
 *
 *****************************************************************************/

#ifndef __MLIMETHODAMGCRH__
#define __MLIMETHODAMGCRH__

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "mli.h"
#include "mli_matrix.h"
#include "mli_method.h"

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

