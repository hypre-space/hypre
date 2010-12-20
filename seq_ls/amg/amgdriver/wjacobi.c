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
 * Weighted Jacobi functions
 *
 *****************************************************************************/

#include "headers.h"
#include "wjacobi.h"


/*--------------------------------------------------------------------------
 * WJacobi
 *--------------------------------------------------------------------------*/

HYPRE_Int    	 WJacobi(x, b, tol, data)
hypre_Vector 	*x;
hypre_Vector 	*b;
double 	 tol;
void    *data;
{
   WJacobiData    *wjacobi_data = data;

   double          weight   = WJacobiDataWeight(wjacobi_data);
   HYPRE_Int         	   max_iter = WJacobiDataMaxIter(wjacobi_data);

   hypre_Matrix         *A        = WJacobiDataA(wjacobi_data);
   hypre_Vector      	  *t        = WJacobiDataT(wjacobi_data);

   double         *a  = hypre_MatrixData(A);
   HYPRE_Int            *ia = hypre_MatrixIA(A);
   HYPRE_Int            *ja = hypre_MatrixJA(A);
   HYPRE_Int             n  = hypre_MatrixSize(A);
	          
   double         *xp = hypre_VectorData(x);
   double         *bp = hypre_VectorData(b);
   double         *tp = hypre_VectorData(t);
	          
   HYPRE_Int             i, j, jj;
   HYPRE_Int             iter = 0;


   /*-----------------------------------------------------------------------
    * Start WJacobi
    *-----------------------------------------------------------------------*/

   while ((iter+1) <= max_iter)
   {
      iter++;

      hypre_CopyVector(b, t);

      for (i = 0; i < n; i++)
      {
	 for (jj = ia[i]; jj < ia[i+1]-1; jj++)
	 {
	    j = ja[jj]-1;
	    tp[i] -= a[jj] * xp[j];
	 }
	 tp[i] /= a[ia[i]-1];
      }

      if (weight != 1.0)
      {
	 hypre_ScaleVector((1.0 - weight), x);
	 hypre_Axpy(weight, t, x);
      }
      else
      {
	 hypre_CopyVector(t, x);
      }
   }
}

/*--------------------------------------------------------------------------
 * WJacobiSetup
 *--------------------------------------------------------------------------*/

void      WJacobiSetup(A, data)
hypre_Matrix   *A;
void     *data;
{
   WJacobiData  *wjacobi_data = data;

   double   *darray;
   HYPRE_Int       size;


   WJacobiDataA(wjacobi_data) = A;

   size = hypre_MatrixSize(A);
   darray = hypre_CTAlloc(double, hypre_NDIMU(size));
   WJacobiDataT(wjacobi_data) = hypre_NewVector(darray, size);
}

/*--------------------------------------------------------------------------
 * NewWJacobiData
 *--------------------------------------------------------------------------*/

void     *NewWJacobiData(problem, solver, log_file_name)
Problem  *problem;
Solver   *solver;
char     *log_file_name;
{
   WJacobiData  *wjacobi_data;

   wjacobi_data = hypre_CTAlloc(WJacobiData, 1);

   WJacobiDataWeight(wjacobi_data)      = SolverWJacobiWeight(solver);
   WJacobiDataMaxIter(wjacobi_data)     = SolverWJacobiMaxIter(solver);

   WJacobiDataLogFileName(wjacobi_data) = log_file_name;

   return (void *)wjacobi_data;
}

/*--------------------------------------------------------------------------
 * FreeWJacobiData
 *--------------------------------------------------------------------------*/

void   FreeWJacobiData(data)
void  *data;
{
   WJacobiData  *wjacobi_data = data;


   if (wjacobi_data)
   {
      hypre_FreeVector(WJacobiDataT(wjacobi_data));
      hypre_TFree(wjacobi_data);
   }
}

