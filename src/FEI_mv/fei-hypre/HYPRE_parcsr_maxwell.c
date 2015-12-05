/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_CotreeData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   double   tol;
   hypre_ParCSRMatrix *Aee;
   hypre_ParCSRMatrix *Att;
   hypre_ParCSRMatrix *Atc;
   hypre_ParCSRMatrix *Act;
   hypre_ParCSRMatrix *Acc;
   hypre_ParCSRMatrix *Gen;
   hypre_ParCSRMatrix *Gc;
   hypre_ParCSRMatrix *Gt;
   hypre_ParCSRMatrix *Gtinv;
   hypre_ParVector    *w;
} hypre_CotreeData;

/******************************************************************************
 *
 * HYPRE_ParCSRCotree interface
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCotreeCreate
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRCotreeCreate(MPI_Comm comm, HYPRE_Solver *solver)
{
   hypre_CotreeData *cotree_data;
   void             *void_data;
 
   cotree_data = hypre_CTAlloc(hypre_CotreeData, 1);
   void_data = (void *) cotree_data;
   *solver = (HYPRE_Solver) void_data;
 
   (cotree_data -> Aee)                = NULL;
   (cotree_data -> Acc)                = NULL;
   (cotree_data -> Act)                = NULL;
   (cotree_data -> Atc)                = NULL;
   (cotree_data -> Att)                = NULL;
   (cotree_data -> Gen)                = NULL;
   (cotree_data -> Gc)                 = NULL;
   (cotree_data -> Gt)                 = NULL;
   (cotree_data -> Gtinv)              = NULL;
   (cotree_data -> tol)                = 1.0e-06;
   (cotree_data -> max_iter)           = 1000;
   (cotree_data -> w)                  = NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCotreeDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRCotreeDestroy(HYPRE_Solver solver)
{
   void             *cotree_vdata = (void *) solver;
   hypre_CotreeData *cotree_data = (hypre_CotreeData *) cotree_vdata;
 
   if (cotree_data)
   {
      hypre_TFree(cotree_data);
      if ((cotree_data->w) != NULL)
      {
         hypre_ParVectorDestroy(cotree_data->w);
         cotree_data->w = NULL;
      }
      if ((cotree_data->Acc) != NULL)
      {
         hypre_ParCSRMatrixDestroy(cotree_data->Acc);
         cotree_data->Acc = NULL;
      }
      if ((cotree_data->Act) != NULL)
      {
         hypre_ParCSRMatrixDestroy(cotree_data->Act);
         cotree_data->Act = NULL;
      }
      if ((cotree_data->Atc) != NULL)
      {
         hypre_ParCSRMatrixDestroy(cotree_data->Atc);
         cotree_data->Atc = NULL;
      }
      if ((cotree_data->Att) != NULL)
      {
         hypre_ParCSRMatrixDestroy(cotree_data->Att);
         cotree_data->Att = NULL;
      }
      if ((cotree_data->Gc) != NULL)
      {
         hypre_ParCSRMatrixDestroy(cotree_data->Gc);
         cotree_data->Gc = NULL;
      }
      if ((cotree_data->Gt) != NULL)
      {
         hypre_ParCSRMatrixDestroy(cotree_data->Gt);
         cotree_data->Gt = NULL;
      }
      if ((cotree_data->Gtinv) != NULL)
      {
         hypre_ParCSRMatrixDestroy(cotree_data->Gtinv);
         cotree_data->Gtinv = NULL;
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCotreeSetup
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRCotreeSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b, HYPRE_ParVector x)
{
   int           *partition, *new_partition, nprocs, *tindices, ii;
   void *vsolver = (void *) solver;
/*
   void *vA      = (void *) A;
   void *vb      = (void *) b;
   void *vx      = (void *) x;
*/
   hypre_CotreeData   *cotree_data = (hypre_CotreeData *) vsolver;
   hypre_ParCSRMatrix **submatrices;
   hypre_ParVector    *new_vector;
   MPI_Comm           comm;

   cotree_data->Aee = (hypre_ParCSRMatrix *) A;
   hypre_ParCSRMatrixGenSpanningTree(cotree_data->Gen, &tindices, 1);
   submatrices = (hypre_ParCSRMatrix **) malloc(sizeof(hypre_ParCSRMatrix *));
   hypre_ParCSRMatrixExtractSubmatrices(cotree_data->Aee, tindices,
                                        &submatrices);
   cotree_data->Att = submatrices[0];
   cotree_data->Atc = submatrices[1];
   cotree_data->Act = submatrices[2];
   cotree_data->Acc = submatrices[3];

   hypre_ParCSRMatrixExtractRowSubmatrices(cotree_data->Gen, tindices,
                                           &submatrices);
   cotree_data->Gt = submatrices[0];
   cotree_data->Gc = submatrices[1];
   free(submatrices);

   comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) A);
   MPI_Comm_size(comm, &nprocs);
   partition = hypre_ParVectorPartitioning((hypre_ParVector *) b);
   new_partition = (int *) malloc((nprocs+1) * sizeof(int));
   for (ii = 0; ii <= nprocs; ii++) new_partition[ii] = partition[ii];
/*   partition = hypre_ParVectorPartitioning((hypre_ParVector *) b);  */
   new_vector = hypre_ParVectorCreate(hypre_ParVectorComm((hypre_ParVector *)b),
		   (int) hypre_ParVectorGlobalSize((hypre_ParVector *) b),	
                   new_partition);
   hypre_ParVectorInitialize(new_vector);
   cotree_data->w = new_vector;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCotreeSolve
 * (1) Given initial E and f, compute residual R
 * (2) Use GMRES to solve for cotree system given Rc with preconditioner
 *     (a) (I + FF^t) solve
 *     (b) preconditioned \hat{Acc} solve
 *     (c) (I + FF^t) solve
 * (3) update E
 *--------------------------------------------------------------------------
 * (I + FF^t) x = y   where F = G_c G_t^{-1}
 * (1) w2 = G_c^t y
 * (2) Poisson solve A z1 = w2
 * (3) z2 = y - F G_t z1
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRCotreeSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b, HYPRE_ParVector x)
{
   void *cotree_vdata = (void *) solver;
   hypre_CotreeData *cotree_data  = cotree_vdata;
   cotree_data->w = NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCotreeSetTol
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRCotreeSetTol(HYPRE_Solver solver, double tol)
{
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCotreeSetMaxIter
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRCotreeSetMaxIter(HYPRE_Solver solver, int max_iter)
{
   return 0;
}

