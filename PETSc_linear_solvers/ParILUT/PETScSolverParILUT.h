/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the hypre_StructSolver structures
 *
 *****************************************************************************/

#ifndef hypre_PETScSolverParILUT_HEADER
#define hypre_PETScSolverParILUT_HEADER

#include <../../HYPRE_config.h>

#include "../../utilities/general.h"
#include "../../utilities/utilities.h"
#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

/* Include Petsc linear solver headers */
#include "sles.h"

#include "../../HYPRE.h"

/* type definitions from this directory */
#include "./HYPRE_PETScSolverParILUT_types.h"

/* type and prototype declarations for object used in this implementation */
#include "../../PETScMat_linear_solvers/pilut/HYPRE_PETScMatPilutSolver_types.h"
#include "../../PETScMat_linear_solvers/pilut/HYPRE_PETScMatPilutSolver_protos.h"

/* type definition for member SlesOwner in ParILUTData structure */
#define ParILUTLibrary 47
#define ParILUTUser    98

/*--------------------------------------------------------------------------
 * hypre_PETScSolverParILUT
 *--------------------------------------------------------------------------*/

typedef struct
{

  MPI_Comm        comm;

  /* Linear solver structure from Petsc */
  SLES            Sles;
  int             SlesOwner; /* Keeps track of whether library or user allocated
                                SLES for freeing purposes */

  /* Petsc Matrix that defines the system to be solved */
  Mat             SystemMatrix;

  /* Petsc Matrix from which to build the preconditioner */
  Mat             PreconditionerMatrix;

  /* Preconditioner is Parallel ILUT through the HYPRE_PETScMatPilutSolver */
  HYPRE_PETScMatPilutSolver PETScMatPilutSolver;

  /* Diagnostic information */
  int             number_of_iterations;

} hypre_PETScSolverParILUT;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_PETScSolverParILUT
 *--------------------------------------------------------------------------*/

#define hypre_PETScSolverParILUTComm(parilut_data)        ((parilut_data) -> comm)
#define hypre_PETScSolverParILUTSles(parilut_data)        ((parilut_data) -> Sles)
#define hypre_PETScSolverParILUTSlesOwner(parilut_data)       ((parilut_data) -> SlesOwner)
#define hypre_PETScSolverParILUTPreconditionerMatrix(parilut_data)\
                                         ((parilut_data) -> PreconditionerMatrix)
#define hypre_PETScSolverParILUTSystemMatrix(parilut_data)\
                                         ((parilut_data) -> SystemMatrix)
#define hypre_PETScSolverParILUTPETScMatPilutSolver(parilut_data)\
                                         ((parilut_data) -> PETScMatPilutSolver)
#define hypre_PETScSolverParILUTNumIts(parilut_data)\
                                         ((parilut_data) -> number_of_iterations)

/* Include internal prototypes */
#include "./hypre_protos.h"
#include "./internal_protos.h"


#endif
