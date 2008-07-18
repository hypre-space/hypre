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






#include "headers.h"  

/*****************************************************************************
 *
 * solves:P^T( LD) U Px = rhs; using change of variables;
 *     P^{-1} = ILUdof_to_dof from ILUdof to original dofordering;
 *     LD = (L + D^{-1}): lower triangular + diagonal^{-1} part;
 *      U: unit upper triangular part;
 ****************************************************************************/



int hypre_ILUsolve(double *x,

		   int *i_ILUdof_to_dof,
		   
		   int *i_ILUdof_ILUdof,
		   int *j_ILUdof_ILUdof,
		   double *LD_data,

		   int *i_ILUdof_ILUdof_t,
		   int *j_ILUdof_ILUdof_t,
		   double *U_data,

		   double *rhs,

		   int num_dofs)

{
  int ierr = 0;

  int i,j;

  int i_dof;


  /* initiate: -----------------------------------------------*/
  for (i=0; i < num_dofs; i++)
    x[i] = rhs[i];

  /* forward loop: -------------------------------------------*/

  for (i=0; i < num_dofs; i++)
    {
      i_dof = i_ILUdof_to_dof[i];

      for (j=i_ILUdof_ILUdof[i]+1; j < i_ILUdof_ILUdof[i+1]; j++)
	x[i_dof] -= LD_data[j] * x[i_ILUdof_to_dof[j_ILUdof_ILUdof[j]]];
    }

  for (i=0; i < num_dofs; i++)
    x[i_ILUdof_to_dof[i]] *= LD_data[i_ILUdof_ILUdof[i]];

  /* backward loop: -----------------------------------------*/

  for (i = num_dofs-1; i > -1; i--)
    {
      i_dof = i_ILUdof_to_dof[i];

      for (j=i_ILUdof_ILUdof_t[i]; j < i_ILUdof_ILUdof_t[i+1]; j++)
	x[i_dof] -= U_data[j] * x[i_ILUdof_to_dof[j_ILUdof_ILUdof_t[j]]];
    }


  return ierr;

}
int hypre_LDsolve(double *x,

		   int *i_ILUdof_to_dof,
		   
		   int *i_ILUdof_ILUdof,
		   int *j_ILUdof_ILUdof,
		   double *LD_data,

		   double *rhs,

		   int num_dofs)

{
  int ierr = 0;

  int i,j;

  int i_dof;


  /* initiate: -----------------------------------------------*/
  for (i=0; i < num_dofs; i++)
    x[i] = rhs[i];

  /* forward loop: -------------------------------------------*/

  for (i=0; i < num_dofs; i++)
    {
      i_dof = i_ILUdof_to_dof[i];

      for (j=i_ILUdof_ILUdof[i]+1; j < i_ILUdof_ILUdof[i+1]; j++)
	x[i_dof] -= LD_data[j] * x[i_ILUdof_to_dof[j_ILUdof_ILUdof[j]]];
    }

  for (i=0; i < num_dofs; i++)
    x[i_ILUdof_to_dof[i]] *= LD_data[i_ILUdof_ILUdof[i]];

  return ierr;

}
int hypre_Dsolve(double *x,

		 int *i_ILUdof_to_dof,
		   
		 int *i_ILUdof_ILUdof,
		 int *j_ILUdof_ILUdof,
		 double *LD_data,

		 double *rhs,

		 int num_dofs)

{
  int ierr = 0;

  int i,j;

  int i_dof;


  for (i=0; i < num_dofs; i++)
    x[i_ILUdof_to_dof[i]] = LD_data[i_ILUdof_ILUdof[i]] *
      rhs[i_ILUdof_to_dof[i]];

  return ierr;

}
int hypre_Usolve(double *x,

		 int *i_ILUdof_to_dof,
		 int *i_ILUdof_ILUdof_t,
		 int *j_ILUdof_ILUdof_t,
		 double *U_data,

		 double *rhs,

		 int num_dofs)

{
  int ierr = 0;

  int i,j;

  int i_dof;


  /* initiate: -----------------------------------------------*/
  for (i=0; i < num_dofs; i++)
    x[i] = rhs[i];

  /* backward loop: -----------------------------------------*/

  for (i = num_dofs-1; i > -1; i--)
    {
      i_dof = i_ILUdof_to_dof[i];

      for (j=i_ILUdof_ILUdof_t[i]; j < i_ILUdof_ILUdof_t[i+1]; j++)
	x[i_dof] -= U_data[j] * x[i_ILUdof_to_dof[j_ILUdof_ILUdof_t[j]]];
    }


  return ierr;

}
