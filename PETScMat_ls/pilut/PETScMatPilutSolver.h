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

#ifndef hypre_PETSC_MAT_PILUT_SOLVER_HEADER
#define hypre_PETSC_MAT_PILUT_SOLVER_HEADER

#include "../../includes/general.h"
#include "../../utilities/memory.h"
#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include "mpi.h"

/* Include Petsc matrix and vector headers */
#include "mat.h"
#include "vec.h"

#include "HYPRE.h"

/*--------------------------------------------------------------------------
 * hypre_PETScMatPilutSolver
 *--------------------------------------------------------------------------*/

typedef struct
{

  MPI_Comm        comm;

  /* Petsc Matrix that defines the system to be solved */
  Mat             Matrix;

  /* This solver is a wrapper for DistributedMatrixPilutSolver; */
  HYPRE_DistributedMatrixPilutSolver DistributedSolver;

} hypre_PETScMatPilutSolver;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_PETScMatPilutSolver
 *--------------------------------------------------------------------------*/

#define hypre_PETScMatPilutSolverComm(parilut_data)        ((parilut_data) -> comm)
#define hypre_PETScMatPilutSolverMatrix(parilut_data)\
                                         ((parilut_data) -> Matrix)
#define hypre_PETScMatPilutSolverDistributedSolver(parilut_data)\
                                         ((parilut_data) -> DistributedSolver)

/* Include internal prototypes */
#include "./hypre_protos.h"
#include "./internal_protos.h"


#endif
