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
 * Header file for ZZZ library
 *
 *****************************************************************************/

#ifndef ZZZ_HEADER
#define ZZZ_HEADER


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void *ZZZ_StructStencil;
typedef void *ZZZ_StructGrid;
typedef void *ZZZ_StructMatrix;
typedef void *ZZZ_StructVector;
typedef void *ZZZ_StructSolver;

/*--------------------------------------------------------------------------
 * Constants
 *--------------------------------------------------------------------------*/

#define ZZZ_PETSC_MATRIX 1
#define ZZZ_PETSC_VECTOR 33
#define ZZZ_PETSC_SOLVER 22

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif

ZZZ_StructGrid ZZZ_NewStructGrid P((int dim ));
void ZZZ_FreeStructGrid P((ZZZ_StructGrid grid ));
void ZZZ_SetStructGridExtents P((ZZZ_StructGrid grid , int *ilower , int *iupper ));
void ZZZ_AssembleStructGrid P((ZZZ_StructGrid grid ));
ZZZ_StructMatrix ZZZ_NewStructMatrix P((MPI_Comm context , ZZZ_StructGrid grid , ZZZ_StructStencil stencil ));
int ZZZ_FreeStructMatrix P((ZZZ_StructMatrix struct_matrix ));
int ZZZ_SetStructMatrixCoeffs P((ZZZ_StructMatrix matrix , int *grid_index , double *coeffs ));
int ZZZ_AssembleStructMatrix P((ZZZ_StructMatrix matrix ));
int ZZZ_PrintStructMatrix P((ZZZ_StructMatrix matrix ));
int ZZZ_SetStructMatrixStorageType P((ZZZ_StructMatrix struct_matrix , int type ));
ZZZ_StructSolver *ZZZ_NewStructSolver P((MPI_Comm context , ZZZ_StructGrid *grid , ZZZ_StructStencil *stencil ));
int ZZZ_FreeStructSolver P((ZZZ_StructSolver *struct_solver ));
int ZZZ_StructSolverSetType P((ZZZ_StructSolver *solver , int type ));
int ZZZ_StructSolverSetup P((ZZZ_StructSolver *solver , ZZZ_StructMatrix *matrix , ZZZ_StructVector *soln , ZZZ_StructVector *rhs ));
int ZZZ_StructSolverSolve P((ZZZ_StructSolver *solver ));
ZZZ_StructStencil ZZZ_NewStructStencil P((int dim , int size ));
void ZZZ_SetStructStencilElement P((ZZZ_StructStencil stencil , int element_index , int *offset ));
void ZZZ_FreeStructStencil P((ZZZ_StructStencil stencil ));
ZZZ_StructVector ZZZ_NewStructVector P((MPI_Comm context , ZZZ_StructGrid grid , ZZZ_StructStencil stencil ));
int ZZZ_FreeStructVector P((ZZZ_StructVector struct_vector ));
int ZZZ_SetStructVectorCoeffs P((ZZZ_StructVector vector , int *grid_index , double *coeffs ));
int ZZZ_SetStructVector P((ZZZ_StructVector vector , double *val ));
int ZZZ_AssembleStructVector P((ZZZ_StructVector vector ));
int ZZZ_SetStructVectorStorageType P((ZZZ_StructVector struct_vector , int type ));
int ZZZ_PrintStructVector P((ZZZ_StructVector vector ));

#undef P

#endif
