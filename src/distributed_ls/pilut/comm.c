/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * comm.c
 *
 * This function provides a communication function interface to
 * T3D's pvm
 *
 * 7/8
 * - MPI and verified
 * 7/11
 * - removed shmem validation
 */

#include "HYPRE_config.h"
#include <stdlib.h>
/* #include <unistd.h> */
#include <time.h>

#include "DistributedMatrixPilutSolver.h"

/*************************************************************************
* High level collective routines
**************************************************************************/

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
HYPRE_Int hypre_GlobalSEMax(HYPRE_Int value, MPI_Comm hypre_MPI_Context )
{
  HYPRE_Int max;
  hypre_MPI_Allreduce( &value, &max, 1, HYPRE_MPI_INT, hypre_MPI_MAX, hypre_MPI_Context );

  return max;
}


/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
HYPRE_Int hypre_GlobalSEMin(HYPRE_Int value, MPI_Comm hypre_MPI_Context)
{
  HYPRE_Int min;
  hypre_MPI_Allreduce( &value, &min, 1, HYPRE_MPI_INT, hypre_MPI_MIN, hypre_MPI_Context );

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
HYPRE_Int hypre_GlobalSESum(HYPRE_Int value, MPI_Comm hypre_MPI_Context)
{
  HYPRE_Int sum;

  hypre_MPI_Allreduce( &value, &sum, 1, HYPRE_MPI_INT, hypre_MPI_SUM, hypre_MPI_Context );

  return sum;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
HYPRE_Real hypre_GlobalSEMaxDouble(HYPRE_Real value, MPI_Comm hypre_MPI_Context)
{
  HYPRE_Real max;
  hypre_MPI_Allreduce( &value, &max, 1, hypre_MPI_REAL, hypre_MPI_MAX, hypre_MPI_Context );

  return max;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
HYPRE_Real hypre_GlobalSEMinDouble(HYPRE_Real value, MPI_Comm hypre_MPI_Context)
{
  HYPRE_Real min;
  hypre_MPI_Allreduce( &value, &min, 1, hypre_MPI_REAL, hypre_MPI_MIN, hypre_MPI_Context );

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
HYPRE_Real hypre_GlobalSESumDouble(HYPRE_Real value, MPI_Comm hypre_MPI_Context)
{
  HYPRE_Real sum;
  hypre_MPI_Allreduce( &value, &sum, 1, hypre_MPI_REAL, hypre_MPI_SUM, hypre_MPI_Context );

  return sum;
}
