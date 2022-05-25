/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the top level MLI data structure
 *
 *****************************************************************************/

#ifndef __MLIH__
#define __MLIH__

#define MLI_VCYCLE 0
#define MLI_WCYCLE 1

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include "_hypre_utilities.h"

#include "mli_defs.h"
#include "mli_solver.h"
#include "mli_method.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_fedata.h"
#include "mli_sfei.h"
#include "mli_mapper.h"
#include "mli_oneLevel.h"

class MLI_OneLevel;
class MLI_Method;

/*--------------------------------------------------------------------------
 * MLI data structure declaration
 *--------------------------------------------------------------------------*/

class MLI
{
   MPI_Comm       mpiComm_;         /* MPI communicator                   */
   int            maxLevels_;       /* maximum number of levels           */
   int            numLevels_;       /* number of levels requested by user */
   int            coarsestLevel_;   /* indicate the coarsest level number */
   int            outputLevel_;     /* for diagnostics                    */
   double         tolerance_;       /* for convergence check              */
   int            maxIterations_;   /* termination criterion              */
   int            currIter_;        /* current iteration (within ML)      */
   MLI_OneLevel   **oneLevels_;     /* store information for each level   */
   MLI_Solver     *coarseSolver_;   /* temporarily store the coarse solver*/
   MLI_Method     *methodPtr_;      /* data object for a given method     */
   int            assembled_;       /* indicate MG hierarchy is assembled */
   double         solveTime_;
   double         buildTime_;

public :

   MLI( MPI_Comm mpiComm);
   ~MLI();
   int  setOutputLevel( int level )    { outputLevel_ = level; return 0;}
   int  setTolerance( double tol )     { tolerance_ = tol; return 0;}
   int  setMaxIterations( int iter )   { maxIterations_ = iter; return 0;}
   int  setNumLevels(int levels)       { numLevels_ = levels; return 0;}
   int  setSystemMatrix( int level, MLI_Matrix *Amat );
   int  setRestriction(  int level, MLI_Matrix *Rmat );
   int  setProlongation( int level, MLI_Matrix *Pmat );
   int  setSmoother( int level , int prePost, MLI_Solver *solver );
   int  setFEData( int level, MLI_FEData *fedata, MLI_Mapper *map );
   int  setSFEI( int level, MLI_SFEI *sfei );
   int  setCoarseSolve( MLI_Solver *solver );
   int  setCyclesAtLevel(int level, int cycles);
   int  setMethod( MLI_Method *method_data );
   int  setup();
   int  cycle( MLI_Vector *sol, MLI_Vector *rhs );
   int  solve( MLI_Vector *sol, MLI_Vector *rhs );
   int  print();
   int  printTiming();

   MLI_OneLevel *getOneLevelObject( int level );
   MLI_Matrix   *getSystemMatrix( int level );
   MLI_Matrix   *getProlongation( int level );
   MLI_Matrix   *getRestriction( int level );
   MLI_Solver   *getSmoother( int level, int pre_post );
   MLI_FEData   *getFEData( int level );
   MLI_SFEI     *getSFEI( int level );
   MLI_Mapper   *getNodeEqnMap( int level );
   int          resetSystemMatrix( int level );
   int          getNumLevels()         { return numLevels_; }
   MLI_Method   *getMethod()           { return methodPtr_; }
};

#endif

