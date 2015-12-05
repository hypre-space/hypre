/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/





// *************************************************************************
// This is the HYPRE implementation of Schur reduction 
// *************************************************************************

#ifndef __HYPRE_LSI_SCHURH__
#define __HYPRE_LSI_SCHURH__

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

// *************************************************************************
// local defines 
// -------------------------------------------------------------------------

#include "HYPRE_FEI_includes.h"

// *************************************************************************
// class definition
// -------------------------------------------------------------------------

class HYPRE_LSI_Schur
{
   HYPRE_IJMatrix A11mat_;           // mass matrix (should be diagonal)
   HYPRE_IJMatrix A12mat_;           // gradient (divergence) matrix
   HYPRE_IJMatrix A22mat_;           // stabilization matrix 
   HYPRE_IJVector F1vec_;            // rhs for block(1,1)
   HYPRE_IJMatrix Smat_;             // Schur complement matrix 
   HYPRE_IJVector Svec_;             // reduced RHS 
   int            *APartition_;      // processor partition of matrix A
   int            P22Size_;          // number of pressure variables
   int            P22GSize_;         // global number of pressure variables
   int            *P22LocalInds_;    // pressure local row indices (global)
   int            *P22GlobalInds_;   // pressure off-processor row indices
   int            *P22Offsets_;      // processor partiton of matrix A22
   int            assembled_;        // set up complete flag
   int            outputLevel_;      // for diagnostics
   Lookup         *lookup_;          // FEI lookup object
   MPI_Comm       mpiComm_;

 public:

   HYPRE_LSI_Schur();
   virtual ~HYPRE_LSI_Schur();
   int     setLookup( Lookup *lookup );
   int     setup(HYPRE_IJMatrix Amat, 
                 HYPRE_IJVector sol,   HYPRE_IJVector rhs,
                 HYPRE_IJMatrix *redA, HYPRE_IJVector *rsol, 
                 HYPRE_IJVector *rrhs,  HYPRE_IJVector *rres);
   int     computeRHS(HYPRE_IJVector rhs,  HYPRE_IJVector *rrhs);
   int     computeSol(HYPRE_IJVector rsol, HYPRE_IJVector sol);
   int     print();

 private:
   int     computeBlockInfo();
   int     buildBlocks(HYPRE_IJMatrix Amat);
};

#endif

