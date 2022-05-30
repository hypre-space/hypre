/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

// *************************************************************************
// This is the HYPRE implementation of block preconditioners
// *************************************************************************

#ifndef _HYPRE_INCFLOW_BLOCKPRECOND_
#define _HYPRE_INCFLOW_BLOCKPRECOND_

// *************************************************************************
// system libraries used
// -------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

// *************************************************************************
// local defines
// -------------------------------------------------------------------------

#define HYPRE_INCFLOW_BDIAG  1
#define HYPRE_INCFLOW_SDIAG  2
#define HYPRE_INCFLOW_BTRI   3
#define HYPRE_INCFLOW_BLU    4


// *************************************************************************
// FEI include files
// -------------------------------------------------------------------------

#include "HYPRE_FEI_includes.h"

// *************************************************************************
// C-wrapper for the FEI Lookup class
// -------------------------------------------------------------------------

typedef struct HYPRE_Lookup_Struct
{
   void *object;
}
HYPRE_Lookup;

// *************************************************************************
// Solver-Preconditioner parameter data structure
// -------------------------------------------------------------------------

typedef struct HYPRE_LSI_BLOCKP_PARAMS_Struct
{
   int            SolverID_;      // solver ID
   int            PrecondID_;     // preconditioner ID
   double         Tol_;           // tolerance for Krylov solver
   int            MaxIter_;       // max iterations for Krylov solver
   int            PSNLevels_;     // Nlevels for ParaSails
   double         PSThresh_;      // threshold for ParaSails
   double         PSFilter_;      // filter for ParaSails
   double         AMGThresh_;     // threshold for BoomerAMG
   int            AMGRelaxType_;  // smoother for BoomerAMG
   int            AMGNSweeps_;    // no. of relaxations for BoomerAMG
   int            AMGSystemSize_; // system size for BoomerAMG
   int            PilutFillin_;   // Fillin for Pilut
   double         PilutDropTol_;  // drop tolerance for Pilut
   int            EuclidNLevels_; // nlevels for Euclid
   double         EuclidThresh_;  // threshold for Euclid
   double         DDIlutFillin_;  // fill-in for DDIlut
   double         DDIlutDropTol_; // drop tolerance for DDIlut
   double         MLThresh_;      // threshold for SA AMG
   int            MLNSweeps_;     // no. of relaxations for SA AMG
   double         MLIThresh_;     // threshold for MLI's SA AMG
   int            MLIRelaxType_;  // smoother for MLI's SA AMG
   int            MLINSweeps_;    // no. of relaxations for MLI's SA AMG
   double         MLIPweight_;    // prolongation smoother weight
   int            MLINodeDOF_;    // nodal degree of freedom
   int            MLINullDim_;    // null space dimension
}
HYPRE_LSI_BLOCKP_PARAMS;

// *************************************************************************
// class definition
// -------------------------------------------------------------------------

class HYPRE_LSI_BlockP
{
   HYPRE_ParCSRMatrix Amat_;         // incoming system matrix
   HYPRE_IJMatrix A11mat_;           // velocity matrix
   HYPRE_IJMatrix A12mat_;           // gradient (divergence) matrix
   HYPRE_IJMatrix A22mat_;           // pressure Poisson
   HYPRE_IJVector F1vec_;            // rhs for velocity
   HYPRE_IJVector F2vec_;            // rhs for pressure
   HYPRE_IJVector X1vec_;            // solution for velocity
   HYPRE_IJVector X2vec_;            // solution for pressure
   HYPRE_IJVector X1aux_;            // auxiliary vector for velocity
   int            *APartition_;      // processor partition of matrix A
   int            P22Size_;          // number of pressure variables
   int            P22GSize_;         // global number of pressure variables
   int            *P22LocalInds_;    // pressure local row indices (global)
   int            *P22GlobalInds_;   // pressure off-processor row indices
   int            *P22Offsets_;      // processor partiton of matrix A22
   int            block1FieldID_;    // identifier for (1,1) block
   int            block2FieldID_;    // identifier for (2,2) block
   int            assembled_;        // set up complete flag
   int            outputLevel_;      // for diagnostics
   int            lumpedMassScheme_; // use diagonal or approximate inverse
   int            lumpedMassNlevels_;// if approx inverse, nlevels
   double         lumpedMassThresh_; // if approx inverse, threshold
   int            lumpedMassLength_; // length of M_v and M_p
   double         *lumpedMassDiag_;  // M_v and M_p lumped
   int            scheme_;           // which preconditioning ?
   int            printFlag_;        // for diagnostics
   HYPRE_Solver   A11Solver_;        // solver for velocity matrix
   HYPRE_Solver   A11Precond_;       // preconditioner for velocity matrix
   HYPRE_Solver   A22Solver_;        // solver for pressure Poisson
   HYPRE_Solver   A22Precond_;       // preconditioner for pressure Poisson
   HYPRE_LSI_BLOCKP_PARAMS A11Params_;
   HYPRE_LSI_BLOCKP_PARAMS A22Params_;
   Lookup         *lookup_;          // FEI lookup object

 public:

   HYPRE_LSI_BlockP();
   virtual ~HYPRE_LSI_BlockP();
   int     setLumpedMasses( int length, double *Mdiag );
   int     setParams(char *param);
   int     setLookup( Lookup *lookup );
   int     setup(HYPRE_ParCSRMatrix Amat);
   int     solve( HYPRE_ParVector fvec, HYPRE_ParVector xvec );
   int     print();

 private:
   int     destroySolverPrecond();
   int     computeBlockInfo();
   int     buildBlocks();
   int     setupPrecon(HYPRE_Solver *precon, HYPRE_IJMatrix Amat,
                       HYPRE_LSI_BLOCKP_PARAMS);
   int     setupSolver(HYPRE_Solver *solver, HYPRE_IJMatrix Amat,
                       HYPRE_IJVector f, HYPRE_IJVector x,
                       HYPRE_Solver precon, HYPRE_LSI_BLOCKP_PARAMS);
   int     solveBDSolve (HYPRE_IJVector x1, HYPRE_IJVector x2,
                         HYPRE_IJVector f1, HYPRE_IJVector f2 );
   int     solveBTSolve (HYPRE_IJVector x1, HYPRE_IJVector x2,
                         HYPRE_IJVector f1, HYPRE_IJVector f2 );
   int     solveBLUSolve(HYPRE_IJVector x1, HYPRE_IJVector x2,
                         HYPRE_IJVector f1, HYPRE_IJVector f2 );
};

#endif

