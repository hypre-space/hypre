/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.16 $
 ***********************************************************************EHEADER*/





// **************************************************************************
// This is the class that handles slide surface reduction
// **************************************************************************

#ifndef __HYPRE_SLIDEREDUCTION__
#define __HYPRE_SLIDEREDUCTION__

// **************************************************************************
// system libraries used
// --------------------------------------------------------------------------

#include "utilities/_hypre_utilities.h"
#include "IJ_mv/_hypre_IJ_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

// **************************************************************************
// class definition
// --------------------------------------------------------------------------

class HYPRE_SlideReduction
{
   MPI_Comm       mpiComm_;
   HYPRE_IJMatrix Amat_;
   HYPRE_IJMatrix A21mat_;
   HYPRE_IJMatrix invA22mat_;
   HYPRE_IJMatrix reducedAmat_;
   HYPRE_IJVector reducedBvec_;
   HYPRE_IJVector reducedXvec_;
   HYPRE_IJVector reducedRvec_;
   int            outputLevel_;
   int            *procNConstr_;
   int            *slaveEqnList_;
   int            *slaveEqnListAux_;
   int            *gSlaveEqnList_;
   int            *gSlaveEqnListAux_;
   int            *constrBlkInfo_;
   int            *constrBlkSizes_;
   int            *eqnStatuses_;
   double         blockMinNorm_;
   HYPRE_ParCSRMatrix hypreRAP_;
   double         truncTol_;
   double         *ADiagISqrts_;
   int            scaleMatrixFlag_;
   int            useSimpleScheme_;

 public:

   HYPRE_SlideReduction(MPI_Comm);
   virtual ~HYPRE_SlideReduction();
   int    setOutputLevel(int level);
   int    setUseSimpleScheme();
   int    setTruncationThreshold(double trunc);
   int    setScaleMatrix();
   int    setBlockMinNorm(double norm);

   int    getMatrixNumRows(); 
   double *getMatrixDiagonal();
   int    getReducedMatrix(HYPRE_IJMatrix *mat); 
   int    getReducedRHSVector(HYPRE_IJVector *rhs);
   int    getReducedSolnVector(HYPRE_IJVector *sol);
   int    getReducedAuxVector(HYPRE_IJVector *auxV);
   int    getProcConstraintMap(int **map);
   int    getSlaveEqnList(int **slist);
   int    getPerturbationMatrix(HYPRE_ParCSRMatrix *matrix);
   int    setup(HYPRE_IJMatrix , HYPRE_IJVector , HYPRE_IJVector );
   int    buildReducedSolnVector(HYPRE_IJVector x, HYPRE_IJVector b);
   int    buildModifiedSolnVector(HYPRE_IJVector x);

 private:

   int    findConstraints();
   int    findSlaveEqns1();
   int    findSlaveEqnsBlock(int blkSize);
   int    composeGlobalList();
   int    buildSubMatrices();
   int    buildModifiedRHSVector(HYPRE_IJVector, HYPRE_IJVector);
   int    buildReducedMatrix();
   int    buildReducedRHSVector(HYPRE_IJVector);
   int    buildA21Mat();
   int    buildInvA22Mat();
   int    scaleMatrixVector();
   double matrixCondEst(int, int, int *, int);

   int    findSlaveEqns2(int **couplings);
   int    buildReducedMatrix2();
};

#endif

