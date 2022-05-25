/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre_ParaSails.h header file.
 *
 *****************************************************************************/

#include "HYPRE_distributed_matrix_protos.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"

typedef void *hypre_ParaSails;

HYPRE_Int hypre_ParaSailsCreate(MPI_Comm comm, hypre_ParaSails *obj);
HYPRE_Int hypre_ParaSailsDestroy(hypre_ParaSails ps);
HYPRE_Int hypre_ParaSailsSetup(hypre_ParaSails obj,
  HYPRE_DistributedMatrix distmat, HYPRE_Int sym, HYPRE_Real thresh, HYPRE_Int nlevels,
  HYPRE_Real filter, HYPRE_Real loadbal, HYPRE_Int logging);
HYPRE_Int hypre_ParaSailsSetupPattern(hypre_ParaSails obj,
  HYPRE_DistributedMatrix distmat, HYPRE_Int sym, HYPRE_Real thresh, HYPRE_Int nlevels, 
  HYPRE_Int logging);
HYPRE_Int hypre_ParaSailsSetupValues(hypre_ParaSails obj,
  HYPRE_DistributedMatrix distmat, HYPRE_Real filter, HYPRE_Real loadbal, 
  HYPRE_Int logging);
HYPRE_Int hypre_ParaSailsApply(hypre_ParaSails ps, HYPRE_Real *u, HYPRE_Real *v);
HYPRE_Int hypre_ParaSailsApplyTrans(hypre_ParaSails ps, HYPRE_Real *u, HYPRE_Real *v);
HYPRE_Int hypre_ParaSailsBuildIJMatrix(hypre_ParaSails obj, HYPRE_IJMatrix *pij_A);
