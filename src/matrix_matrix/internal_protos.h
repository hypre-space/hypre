/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef HYPRE_MIXED_PRECISION
#include "matrix_matrix_mup_func.h"
#endif

/* HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
HYPRE_Int HYPRE_ConvertPETScMatrixToDistributedMatrix (Mat PETSc_matrix,
                                                       HYPRE_DistributedMatrix *DistributedMatrix );
