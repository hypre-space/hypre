/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

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

#include <mpi.h>
#include "utilities/utilities.h"

#include "mli_defs.h"
#include "mli_oneLevel.h"
#include "../solver/mli_solver.h"
#include "../amgs/mli_method.h"
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "../fedata/mli_fedata.h"

class MLI_OneLevel;
class MLI_Method;

/*--------------------------------------------------------------------------
 * MLI data structure declaration
 *--------------------------------------------------------------------------*/

class MLI
{
   MPI_Comm       mpi_comm;         /* MPI communicator                   */
   int            max_levels;       /* maximum number of levels           */
   int            num_levels;       /* number of levels requested by user */
   int            coarsest_level;   /* indicate the coarsest level number */
   int            output_level;     /* for diagnostics                    */
   double         tolerance;        /* for convergence check              */
   int            max_iterations;   /* termination criterion              */
   int            curr_iter;        /* current iteration (within ML)      */
   int            method;           /* which multilevel method to use     */
   MLI_OneLevel   **one_levels;     /* store information for each level   */
   MLI_Solver     *coarse_solver;   /* temporarily store the coarse solver*/
   MLI_Method     *method_data;     /* data object for a given method     */
   int            assembled;        /* indicate MG hierarchy is assembled */
   double         solve_time;
   double         build_time;

public :

   MLI( MPI_Comm mpi_comm);
   ~MLI();
   int  setOutputLevel( int level )    { output_level = level; return 0;}
   int  setTolerance( double tol )     { tolerance = tol; return 0;}
   int  setMaxIterations( int iter )   { max_iterations = iter; return 0;}
   int  setNumLevels(int levels)       { num_levels = levels; return 0;}
   int  setSystemMatrix( int level, MLI_Matrix *Amat );
   int  setRestriction(  int level, MLI_Matrix *Rmat );
   int  setProlongation( int level, MLI_Matrix *Pmat );
   int  setSmoother( int level , int pre_post, MLI_Solver *solver );
   int  setFEData( int level, MLI_FEData *fedata );
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
   int          resetSystemMatrix( int level );
   int          resetMethod();
   MLI_Method   *getMethod()           { return method_data; }
};

#endif

