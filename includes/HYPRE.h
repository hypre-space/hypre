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
 * Header file for HYPRE library
 *
 *****************************************************************************/

#ifndef HYPRE_HEADER
#define HYPRE_HEADER


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void *HYPRE_StructStencil;
typedef void *HYPRE_StructGrid;

#include "../struct_matrix_vector_andy/HYPRE_types.h"
#include "../struct_linear_solvers_andy/HYPRE_types.h"
#include "../distributed_matrix/HYPRE_types.h"
#include "../distributed_linear_solvers/pilut/HYPRE_types.h"
#include "../PETScMat_linear_solvers/pilut/HYPRE_types.h"
#include "../PETSc_linear_solvers/ParILUT/HYPRE_types.h"

/*--------------------------------------------------------------------------
 * Constants
 *--------------------------------------------------------------------------*/

#define HYPRE_PETSC_MATRIX 12
#define HYPRE_PETSC_VECTOR 33

#define HYPRE_PETSC_MAT_PARILUT_SOLVER 22

#define HYPRE_PARILUT      872

#define HYPRE_UNITIALIZED -47

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#include "../struct_matrix_vector_andy/HYPRE_protos.h"
#include "../struct_linear_solvers_andy/HYPRE_protos.h"
#include "../distributed_matrix/HYPRE_protos.h"
#include "../distributed_linear_solvers/pilut/HYPRE_protos.h"
#include "../PETScMat_linear_solvers/pilut/HYPRE_protos.h"
#include "../PETSc_linear_solvers/ParILUT/HYPRE_protos.h"

#include "../matrix_matrix/HYPRE_protos.h"

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif

HYPRE_StructGrid HYPRE_NewStructGrid P((int dim ));
void HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
void HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
void HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));
HYPRE_StructStencil HYPRE_NewStructStencil P((int dim , int size ));
void HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

#undef P

#endif
