/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "HYPRE_distributed_matrix_types.h"

#ifdef PETSC_AVAILABLE
/* HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
HYPRE_Int HYPRE_ConvertPETScMatrixToDistributedMatrix (Mat PETSc_matrix,
                                                       HYPRE_DistributedMatrix *DistributedMatrix );
#endif

/* HYPRE_ConvertParCSRMatrixToDistributedMatrix.c */
HYPRE_Int HYPRE_ConvertParCSRMatrixToDistributedMatrix (HYPRE_ParCSRMatrix parcsr_matrix,
                                                        HYPRE_DistributedMatrix *DistributedMatrix );

