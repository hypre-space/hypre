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
 * Header info for the zzz_StructSolver structures
 *
 *****************************************************************************/

#ifndef zzz_STENCIL_SOLVER_HEADER
#define zzz_STENCIL_SOLVER_HEADER


/*--------------------------------------------------------------------------
 * zzz_StructSolver:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   zzz_StructGrid     *grid;
   zzz_StructStencil  *stencil;

   zzz_StructMatrix *matrix;
   zzz_StructVector *soln;
   zzz_StructVector *rhs;

   int           solver_type;
   void     	*data;

} zzz_StructSolver;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_StructSolver
 *--------------------------------------------------------------------------*/

#define zzz_StructSolverContext(solver)      ((solver) -> context)
#define zzz_StructSolverStructGrid(solver)         ((solver) -> grid)
#define zzz_StructSolverStructStencil(solver)      ((solver) -> stencil)

#define zzz_StructSolverMatrix(solver)       ((solver) -> matrix)
#define zzz_StructSolverSoln(solver)         ((solver) -> soln)
#define zzz_StructSolverRhs(solver)          ((solver) -> rhs)

#define zzz_StructSolverSolverType(solver)   ((solver) -> solver_type)
#define zzz_StructSolverData(solver)         ((solver) -> data)


#endif
