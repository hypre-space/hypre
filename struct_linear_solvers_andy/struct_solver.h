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

#ifndef hypre_STRUCT_SOLVER_HEADER
#define hypre_STRUCT_SOLVER_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructSolver:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;

   HYPRE_StructMatrix matrix;
   HYPRE_StructVector soln;
   HYPRE_StructVector rhs;

   int           solver_type;
   void     	*data; /* Must be cast to some available solver type */

} hypre_StructSolver;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructSolver
 *--------------------------------------------------------------------------*/

#define hypre_StructSolverContext(solver)      ((solver) -> context)
#define hypre_StructSolverStructGrid(solver)         ((solver) -> grid)
#define hypre_StructSolverStructStencil(solver)      ((solver) -> stencil)

#define hypre_StructSolverMatrix(solver)       ((solver) -> matrix)
#define hypre_StructSolverSoln(solver)         ((solver) -> soln)
#define hypre_StructSolverRhs(solver)          ((solver) -> rhs)

#define hypre_StructSolverSolverType(solver)   ((solver) -> solver_type)
#define hypre_StructSolverData(solver)         ((solver) -> data)


#endif
