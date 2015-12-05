/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header info for the Compatible relaxation-based multilevel method
 *
 *****************************************************************************/

#ifndef __MLIMETHODAMGCRH__
#define __MLIMETHODAMGCRH__

#include "utilities/_hypre_utilities.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
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

