
#include <HYPRE_config.h>

#include "HYPRE_CI_ls.h"

#ifndef hypre_CI_LS_HEADER
#define hypre_CI_LS_HEADER

#include "utilities.h"
#include "CI_struct_mv.h"
#include "HYPRE.h"

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
 * Header info for the hypre_StructInterfaceSolver structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_INTERFACE_SOLVER_HEADER
#define hypre_STRUCT_INTERFACE_SOLVER_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolver:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;

   HYPRE_StructInterfaceMatrix matrix;
   HYPRE_StructInterfaceVector soln;
   HYPRE_StructInterfaceVector rhs;

   int           solver_type;
   void     	*data; /* Must be cast to some available solver type */

} hypre_StructInterfaceSolver;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructInterfaceSolver
 *--------------------------------------------------------------------------*/

#define hypre_StructInterfaceSolverContext(solver)      ((solver) -> context)
#define hypre_StructInterfaceSolverStructGrid(solver)         ((solver) -> grid)
#define hypre_StructInterfaceSolverStructStencil(solver)      ((solver) -> stencil)

#define hypre_StructInterfaceSolverMatrix(solver)       ((solver) -> matrix)
#define hypre_StructInterfaceSolverSoln(solver)         ((solver) -> soln)
#define hypre_StructInterfaceSolverRhs(solver)          ((solver) -> rhs)

#define hypre_StructInterfaceSolverSolverType(solver)   ((solver) -> solver_type)
#define hypre_StructInterfaceSolverData(solver)         ((solver) -> data)


#endif

#endif
