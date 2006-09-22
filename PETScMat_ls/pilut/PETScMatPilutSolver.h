/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Header info for the hypre_StructSolver structures
 *
 *****************************************************************************/

#ifndef hypre_PETSC_MAT_PILUT_SOLVER_HEADER
#define hypre_PETSC_MAT_PILUT_SOLVER_HEADER

#include <HYPRE_config.h>

#include "general.h"
#include "utilities.h"
#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

/* Include Petsc matrix and vector headers */
#include "mat.h"
#include "vec.h"

#include "HYPRE.h"

/* types of objects declared in this directory */
#include "HYPRE_PETScMatPilutSolver_types.h"

/* type declaration for matrix object used by the implementation in this directory */
#include "HYPRE_distributed_matrix_types.h"

/* type declaration and prototypes for solvers used in this implementation */
#include "HYPRE_DistributedMatrixPilutSolver_types.h"
#include "HYPRE_DistributedMatrixPilutSolver_protos.h"

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
