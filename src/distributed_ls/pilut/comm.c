/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




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
double hypre_GlobalSEMaxDouble(double value, MPI_Comm hypre_MPI_Context)
{
  double max;
  hypre_MPI_Allreduce( &value, &max, 1, hypre_MPI_DOUBLE, hypre_MPI_MAX, hypre_MPI_Context );

  return max;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double hypre_GlobalSEMinDouble(double value, MPI_Comm hypre_MPI_Context)
{
  double min;
  hypre_MPI_Allreduce( &value, &min, 1, hypre_MPI_DOUBLE, hypre_MPI_MIN, hypre_MPI_Context );

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double hypre_GlobalSESumDouble(double value, MPI_Comm hypre_MPI_Context)
{
  double sum;
  hypre_MPI_Allreduce( &value, &sum, 1, hypre_MPI_DOUBLE, hypre_MPI_SUM, hypre_MPI_Context );

  return sum;
}
