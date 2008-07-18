/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * hypre_ParaSails.h header file.
 *
 *****************************************************************************/

#include "HYPRE_distributed_matrix_protos.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"

typedef void *hypre_ParaSails;

int hypre_ParaSailsCreate(MPI_Comm comm, hypre_ParaSails *obj);
int hypre_ParaSailsDestroy(hypre_ParaSails ps);
int hypre_ParaSailsSetup(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels,
  double filter, double loadbal, int logging);
int hypre_ParaSailsSetupPattern(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels, 
  int logging);
int hypre_ParaSailsSetupValues(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, double filter, double loadbal, 
  int logging);
int hypre_ParaSailsApply(hypre_ParaSails ps, double *u, double *v);
int hypre_ParaSailsApplyTrans(hypre_ParaSails ps, double *u, double *v);
int hypre_ParaSailsBuildIJMatrix(hypre_ParaSails obj, HYPRE_IJMatrix *pij_A);

