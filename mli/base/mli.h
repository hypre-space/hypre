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

#include "utilities.h"

/*--------------------------------------------------------------------------
 * type definition 
 *--------------------------------------------------------------------------*/

typedef struct MLI_Struct MLI;

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include "mli_onelevel.h"
#include "../solver/mli_solver.h"
#include "../amg/mli_method.h"
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"

/*--------------------------------------------------------------------------
 * MLI_Timing data structure declaration
 *--------------------------------------------------------------------------*/

typedef struct 
{
   double solve_time;
   double build_time;
}
ML_Timing;

/*--------------------------------------------------------------------------
 * MLI data structure declaration
 *--------------------------------------------------------------------------*/

typedef struct MLI_Struct
{
   int            assembled;        /* indicate MG hierarchy is assembled */
   int            method;           /* which multilevel method to use     */
   int            max_levels;       /* maximum number of levels           */
   int            coarsest_level;   /* indicate the coarsest level number */
   int            debug_level;      /* for diagnostics                    */
   double         tolerance;        /* for convergence check              */
   int            max_iterations;   /* termination criterion              */
   MLI_OneLevel   *one_levels;      /* store information for each level   */
   MLI_Timing     timing;           /* for storing timing information     */
   MPI_Comm       mpi_comm;         /* MPI communicator                   */
   MLI_Solver     *coarse_solver;   /* temporarily store the coarse solver*/
   MLI_Method     *method_data;     /* data object for a given method     */
};

/*--------------------------------------------------------------------------
 * functions for MLI 
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern MLI *MLI_Create( MPI_Comm mpi_comm, int nlevels );
extern int MLI_Destroy( MLI *mli );
extern int MLI_SetDebugLevel( MLI *mli, int debug_level );
extern int MLI_SetTolerance( MLI *mli, double tolerance );
extern int MLI_SetMaxIterations( MLI *mli, int iterations );
extern int MLI_PrintTiming( MLI *mli );
extern int MLI_SetAmat( MLI *mli, int level, MLI_Matrix *Amat);
extern int MLI_SetRmat( MLI *mli, int level, MLI_Matrix *Rmat);
extern int MLI_SetPmat( MLI *mli, int level, MLI_Matrix *Pmat);
extern int MLI_SetSmoother( MLI *mli, int level , int pre_post, 
                            MLI_Solver *solver );
extern int MLI_SetCoarseSolver( MLI *mli, MLI_Solver *solver );
extern int MLI_SetMethod( MLI *mli, int method, MLI_Method *method_data);
extern int MLI_SetCycleType( MLI *mli, int cycle_type );
extern int MLI_Setup( MLI *mli );
extern int MLI_Iterate( MLI *mli, MLI_Vector *sol, MLI_Vector *rhs );

#ifdef __cplusplus
}
#endif

#endif

