/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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

