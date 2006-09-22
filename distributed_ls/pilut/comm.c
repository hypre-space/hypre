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
#if HAVE_UNISTD_H
#  include <unistd.h>
#endif /* HAVE_UNISTD_H */
#include <time.h>

#include "DistributedMatrixPilutSolver.h"

/*************************************************************************
* High level collective routines
**************************************************************************/

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int hypre_GlobalSEMax(int value, MPI_Comm MPI_Context )
{
  int max;
  MPI_Allreduce( &value, &max, 1, MPI_INT, MPI_MAX, MPI_Context );

  return max;
}


/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int hypre_GlobalSEMin(int value, MPI_Comm MPI_Context)
{
  int min;
  MPI_Allreduce( &value, &min, 1, MPI_INT, MPI_MIN, MPI_Context );

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int hypre_GlobalSESum(int value, MPI_Comm MPI_Context)
{
  int sum;

  MPI_Allreduce( &value, &sum, 1, MPI_INT, MPI_SUM, MPI_Context );

  return sum;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double hypre_GlobalSEMaxDouble(double value, MPI_Comm MPI_Context)
{
  double max;
  MPI_Allreduce( &value, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_Context );

  return max;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double hypre_GlobalSEMinDouble(double value, MPI_Comm MPI_Context)
{
  double min;
  MPI_Allreduce( &value, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_Context );

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double hypre_GlobalSESumDouble(double value, MPI_Comm MPI_Context)
{
  double sum;
  MPI_Allreduce( &value, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_Context );

  return sum;
}
