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

#include "CI_struct_matrix_vector/HYPRE_types.h"
#include "CI_struct_linear_solvers/HYPRE_types.h"
#include "distributed_matrix/HYPRE_types.h"
#include "distributed_linear_solvers/pilut/HYPRE_types.h"
#include "PETScMat_linear_solvers/pilut/HYPRE_types.h"
#include "PETSc_linear_solvers/ParILUT/HYPRE_types.h"

/*--------------------------------------------------------------------------
 * Constants
 *--------------------------------------------------------------------------*/

#define HYPRE_ISIS_MATRIX 11

#define HYPRE_PETSC_MATRIX 12
#define HYPRE_PETSC_VECTOR 33

#define HYPRE_PETSC_MAT_PARILUT_SOLVER 22

#define HYPRE_PARILUT      872

#define HYPRE_UNITIALIZED -47

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#include "CI_struct_matrix_vector/HYPRE_protos.h"
#include "CI_struct_linear_solvers/HYPRE_protos.h"
#include "distributed_matrix/HYPRE_protos.h"
#include "distributed_linear_solvers/pilut/HYPRE_protos.h"
#include "PETScMat_linear_solvers/pilut/HYPRE_protos.h"
#include "PETSc_linear_solvers/ParILUT/HYPRE_protos.h"

#include "matrix_matrix/HYPRE_protos.h"

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif

#undef P

#endif
