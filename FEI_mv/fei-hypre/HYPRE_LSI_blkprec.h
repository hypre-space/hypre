/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

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
#include <assert.h>
#include <math.h>
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_ls/parcsr_ls.h"
#include "parcsr_mv/parcsr_mv.h"

// *************************************************************************
// local defines 
// -------------------------------------------------------------------------

#define HYPRE_INCFLOW_BDIAG  1
#define HYPRE_INCFLOW_BTRI   2
#define HYPRE_INCFLOW_BAI    3

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
   int            assembled_;        // set up complete flag
   int            outputLevel_;      // for diagnostics
   int            lumpedMassLength_; // length of M_v and M_p
   double         *lumpedMassDiag_;  // M_v and M_p lumped
   int            scheme_;           // which preconditioning ?
   HYPRE_Solver   A11Solver_;        // solver for velocity matrix
   HYPRE_Solver   A11Precond_;       // preconditioner for velocity matrix
   HYPRE_Solver   A22Solver_;        // solver for pressure Poisson 
   HYPRE_Solver   A22Precond_;       // preconditioner for pressure Poisson

 public:

   HYPRE_LSI_BlockP();
   virtual ~HYPRE_LSI_BlockP();
   int     setSchemeBlockDiagonal()   {scheme_ = HYPRE_INCFLOW_BDIAG; return 0;}
   int     setSchemeBlockTriangular() {scheme_ = HYPRE_INCFLOW_BTRI;  return 0;}
   int     setSchemeBlockInverse()    {scheme_ = HYPRE_INCFLOW_BAI;   return 0;}
   int     setLumpedMasses( int length, double *Mdiag );
   int     setup(HYPRE_ParCSRMatrix Amat);
   int     solve( HYPRE_ParVector xvec, HYPRE_ParVector fvec );

 private:
   int     computeBlockInfo();
   int     buildBlocks();
   int     solveBSolve (HYPRE_IJVector x1, HYPRE_IJVector x2,
                        HYPRE_IJVector f1, HYPRE_IJVector f2 );
   int     solveBISolve(HYPRE_IJVector x1, HYPRE_IJVector x2,
                        HYPRE_IJVector f1, HYPRE_IJVector f2 );
};

#endif

