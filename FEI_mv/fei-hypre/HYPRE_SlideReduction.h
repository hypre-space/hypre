/*BHEADER*********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 ********************************************************************EHEADER*/

// **************************************************************************
// This is the class that handles slide surface reduction
// **************************************************************************

#ifndef __HYPRE_SLIDEREDUCTION__
#define __HYPRE_SLIDEREDUCTION__

// **************************************************************************
// system libraries used
// --------------------------------------------------------------------------

#include "utilities/utilities.h"
#include "IJ_mv/IJ_mv.h"
#include "parcsr_mv/parcsr_mv.h"

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

 public:

   HYPRE_SlideReduction(MPI_Comm);
   virtual ~HYPRE_SlideReduction();
   int    setOutputLevel(int level) {outputLevel_ = level; return 0;}
   int    setBlockMinNorm(double norm) {blockMinNorm_ = norm; return 0;}
   int    setup(HYPRE_IJMatrix , HYPRE_IJVector , HYPRE_IJVector );
   int    buildReducedMatrix();
   int    buildReducedRHSVector(HYPRE_IJVector);
   int    buildReducedSolnVector(HYPRE_IJVector x, HYPRE_IJVector b);
   int    getReducedMatrix(HYPRE_IJMatrix *mat) 
                       { (*mat) = reducedAmat_; return 0; }
   int    getReducedRHSVector(HYPRE_IJVector *rhs) 
                       { (*rhs) = reducedBvec_; return 0; }
   int    getReducedSolnVector(HYPRE_IJVector *sol) 
                       { (*sol) = reducedXvec_; return 0; }
   int    getReducedAuxVector(HYPRE_IJVector *auxV ) 
                       { (*auxV) = reducedRvec_; return 0; }
   int    getProcConstraintMap(int **map) 
                       { (*map) = procNConstr_; return 0; }
   int    getSlaveEqnList(int **slist) 
                       { (*slist) = slaveEqnList_; return 0; }
   int    getPerturbationMatrix(HYPRE_ParCSRMatrix *matrix) 
                       { (*matrix) = hypreRAP_; hypreRAP_ = NULL; return 0; }

 private:
   int    findConstraints();
   int    findSlaveEqns1();
   int    findSlaveEqnsBlock(int blkSize);
   int    composeGlobalList();
   int    buildA21Mat();
   int    buildInvA22Mat();

   int    findSlaveEqns2(int **couplings);
   int    buildReducedMatrix2();
   double matrixCondEst(int, int, int *, int);
};

#endif

