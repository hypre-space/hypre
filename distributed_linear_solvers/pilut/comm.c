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

#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "./DistributedMatrixPilutSolver.h"


/*************************************************************************
* High level collective routines
**************************************************************************/

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int GlobalSEMax(int value, MPI_Comm MPI_Context )
{
  int max;
  MPI_Allreduce( &value, &max, 1, MPI_INT, MPI_MAX, MPI_Context );

  return max;
}


/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int GlobalSEMin(int value, MPI_Comm MPI_Context)
{
  int min;
  MPI_Allreduce( &value, &min, 1, MPI_INT, MPI_MIN, MPI_Context );

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int GlobalSESum(int value, MPI_Comm MPI_Context)
{
  int sum;

  MPI_Allreduce( &value, &sum, 1, MPI_INT, MPI_SUM, MPI_Context );

  return sum;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double GlobalSEMaxDouble(double value, MPI_Comm MPI_Context)
{
  double max;
  MPI_Allreduce( &value, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_Context );

  return max;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double GlobalSEMinDouble(double value, MPI_Comm MPI_Context)
{
  double min;
  MPI_Allreduce( &value, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_Context );

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double GlobalSESumDouble(double value, MPI_Comm MPI_Context)
{
  double sum;
  MPI_Allreduce( &value, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_Context );

  return sum;
}
