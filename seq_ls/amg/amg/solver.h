/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Solver data structures
 *
 *****************************************************************************/

#ifndef _SOLVER_HEADER
#define _SOLVER_HEADER


/*--------------------------------------------------------------------------
 * Data
 *--------------------------------------------------------------------------*/

typedef void Data;

/*--------------------------------------------------------------------------
 * Solver
 *--------------------------------------------------------------------------*/

typedef struct
{
   int          type;

   double       stop_tolerance;

   Data        *pcg_data;

   Data        *wjacobi_data;

   Data        *amgs01_data;
   
} Solver;

/*--------------------------------------------------------------------------
 * Accessor functions for the Solver structure
 *--------------------------------------------------------------------------*/

#define SolverType(solver)           ((solver) -> type)

#define SolverStopTolerance(solver)  ((solver) -> stop_tolerance)

#define SolverPCGData(solver)        ((solver) -> pcg_data)

#define SolverWJacobiData(solver)    ((solver) -> wjacobi_data)

#define SolverAMGS01Data(solver)     ((solver) -> amgs01_data)


#endif
