/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * BlockTridiag functions
 *
 *****************************************************************************/

#include <assert.h>
#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/_hypre_IJ_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "block_tridiag.h"

/*--------------------------------------------------------------------------
 * hypre_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

void *hypre_BlockTridiagCreate()
{
   hypre_BlockTridiagData *b_data;
   b_data = hypre_CTAlloc(hypre_BlockTridiagData, 1);
   b_data->threshold = 0.0;
   b_data->num_sweeps = 1;
   b_data->relax_type = 6;
   b_data->print_level = 0;
   b_data->index_set1 = NULL;
   b_data->index_set2 = NULL;
   b_data->F1 = NULL;
   b_data->F2 = NULL;
   b_data->U1 = NULL;
   b_data->U2 = NULL;
   b_data->A11 = NULL;
   b_data->A21 = NULL;
   b_data->A22 = NULL;
   b_data->precon1 = NULL;
   b_data->precon2 = NULL;
   return (void *) b_data;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BlockTridiagDestroy(void *data)
{
   hypre_BlockTridiagData *b_data = (hypre_BlockTridiagData *) data;

   if (b_data->F1)
   {
      hypre_ParVectorDestroy(b_data->F1);
      b_data->F1 = NULL;
   }
   if (b_data->F2)
   {
      hypre_ParVectorDestroy(b_data->F2);
      b_data->F2 = NULL;
   }
   if (b_data->U1)
   {
      hypre_ParVectorDestroy(b_data->U1);
      b_data->U1 = NULL;
   }
   if (b_data->U2)
   {
      hypre_ParVectorDestroy(b_data->U2);
      b_data->U2 = NULL;
   }
   if (b_data->index_set1)
   {
      hypre_TFree(b_data->index_set1);
      b_data->index_set1 = NULL;
   }
   if (b_data->index_set2)
   {
      hypre_TFree(b_data->index_set2);
      b_data->index_set2 = NULL;
   }
   if (b_data->A11)
   {
      hypre_ParCSRMatrixDestroy(b_data->A11);
      b_data->A11 = NULL;
   }
   if (b_data->A21)
   {
      hypre_ParCSRMatrixDestroy(b_data->A21);
      b_data->A21 = NULL;
   }
   if (b_data->A22)
   {
      hypre_ParCSRMatrixDestroy(b_data->A22);
      b_data->A22 = NULL;
   }
   if (b_data->precon1)
   {
      HYPRE_BoomerAMGDestroy(b_data->precon1);
      b_data->precon1 = NULL;
   }
   if (b_data->precon2)
   {
      HYPRE_BoomerAMGDestroy(b_data->precon2);
      b_data->precon2 = NULL;
   }
   hypre_TFree(b_data);
   return (0);
}

/*--------------------------------------------------------------------------
 * Routines to setup the preconditioner
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BlockTridiagSetup(void *data, hypre_ParCSRMatrix *A,
                            hypre_ParVector *b, hypre_ParVector *x) 
{
   HYPRE_Int                i, j, *index_set1, print_level, nsweeps, relax_type;
   HYPRE_Int                nrows, nrows1, nrows2, start1, start2, *index_set2;
   HYPRE_Int                count, ierr;
   double             threshold;
   hypre_ParCSRMatrix **submatrices;
   HYPRE_Solver       precon1;
   HYPRE_Solver       precon2;
   HYPRE_IJVector     ij_u1, ij_u2, ij_f1, ij_f2;
   hypre_ParVector    *vector;
   MPI_Comm           comm;
   hypre_BlockTridiagData *b_data = (hypre_BlockTridiagData *) data;

   HYPRE_ParCSRMatrixGetComm((HYPRE_ParCSRMatrix) A, &comm);
   index_set1 = b_data->index_set1;
   nrows1 = index_set1[0];
   nrows  = hypre_ParCSRMatrixNumRows(A);
   nrows2 = nrows - nrows1;
   b_data->index_set2 = hypre_CTAlloc(HYPRE_Int, nrows2+1);
   index_set2 = b_data->index_set2;
   index_set2[0] = nrows2;
   count = 1;
   for (i = 0; i < index_set1[1]; i++) index_set2[count++] = i;
   for (i = 1; i < nrows1; i++) 
      for (j = index_set1[i]+1; j < index_set1[i+1]; j++) 
         index_set2[count++] = j;
   for (i = index_set1[nrows1]+1; i < nrows; i++) index_set2[count++] = i;

   submatrices = hypre_CTAlloc(hypre_ParCSRMatrix *, 4);
   hypre_ParCSRMatrixExtractSubmatrices(A, index_set1, &submatrices);

   nrows1 = hypre_ParCSRMatrixNumRows(submatrices[0]);
   nrows2 = hypre_ParCSRMatrixNumRows(submatrices[3]); 
   start1 = hypre_ParCSRMatrixFirstRowIndex(submatrices[0]);
   start2 = hypre_ParCSRMatrixFirstRowIndex(submatrices[3]);
   HYPRE_IJVectorCreate(comm, start1, start1+nrows1-1, &ij_u1);
   HYPRE_IJVectorSetObjectType(ij_u1, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(ij_u1);
   ierr += HYPRE_IJVectorAssemble(ij_u1);
   hypre_assert(!ierr);
   HYPRE_IJVectorCreate(comm, start1, start1+nrows1-1, &ij_f1);
   HYPRE_IJVectorSetObjectType(ij_f1, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(ij_f1);
   ierr += HYPRE_IJVectorAssemble(ij_f1);
   hypre_assert(!ierr);
   HYPRE_IJVectorCreate(comm, start2, start2+nrows2-1, &ij_u2);
   HYPRE_IJVectorSetObjectType(ij_u2, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(ij_u2);
   ierr += HYPRE_IJVectorAssemble(ij_u2);
   hypre_assert(!ierr);
   HYPRE_IJVectorCreate(comm, start2, start2+nrows1-1, &ij_f2);
   HYPRE_IJVectorSetObjectType(ij_f2, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(ij_f2);
   ierr += HYPRE_IJVectorAssemble(ij_f2);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(ij_f1, (void **) &vector);
   b_data->F1 = vector;
   HYPRE_IJVectorGetObject(ij_u1, (void **) &vector);
   b_data->U1 = vector;
   HYPRE_IJVectorGetObject(ij_f2, (void **) &vector);
   b_data->F2 = vector;
   HYPRE_IJVectorGetObject(ij_u2, (void **) &vector);
   b_data->U2 = vector;

   print_level = b_data->print_level;
   threshold   = b_data->threshold;
   nsweeps     = b_data->num_sweeps;
   relax_type  = b_data->relax_type;
   threshold = b_data->threshold;
   HYPRE_BoomerAMGCreate(&precon1);
   HYPRE_BoomerAMGSetMaxIter(precon1, 1);
   HYPRE_BoomerAMGSetCycleType(precon1, 1);
   HYPRE_BoomerAMGSetPrintLevel(precon1, print_level);
   HYPRE_BoomerAMGSetMaxLevels(precon1, 25);
   HYPRE_BoomerAMGSetMeasureType(precon1, 0);
   HYPRE_BoomerAMGSetCoarsenType(precon1, 0);
   HYPRE_BoomerAMGSetStrongThreshold(precon1, threshold);
   HYPRE_BoomerAMGSetNumFunctions(precon1, 1);
   HYPRE_BoomerAMGSetNumSweeps(precon1, nsweeps);
   HYPRE_BoomerAMGSetRelaxType(precon1, relax_type);
   hypre_BoomerAMGSetup(precon1, submatrices[0], b_data->U1, b_data->F1);

   HYPRE_BoomerAMGCreate(&precon2);
   HYPRE_BoomerAMGSetMaxIter(precon2, 1);
   HYPRE_BoomerAMGSetCycleType(precon2, 1);
   HYPRE_BoomerAMGSetPrintLevel(precon2, print_level);
   HYPRE_BoomerAMGSetMaxLevels(precon2, 25);
   HYPRE_BoomerAMGSetMeasureType(precon2, 0);
   HYPRE_BoomerAMGSetCoarsenType(precon2, 0);
   HYPRE_BoomerAMGSetMeasureType(precon2, 1);
   HYPRE_BoomerAMGSetStrongThreshold(precon2, threshold);
   HYPRE_BoomerAMGSetNumFunctions(precon2, 1);
   HYPRE_BoomerAMGSetNumSweeps(precon2, nsweeps);
   HYPRE_BoomerAMGSetRelaxType(precon2, relax_type);
   hypre_BoomerAMGSetup(precon2, submatrices[3], NULL, NULL);

   b_data->precon1 = precon1;
   b_data->precon2 = precon2;

   b_data->A11 = submatrices[0];
   hypre_ParCSRMatrixDestroy(submatrices[1]);
   b_data->A21 = submatrices[2];
   b_data->A22 = submatrices[3];

   hypre_TFree(submatrices);
   return (0);
}

/*--------------------------------------------------------------------------
 * Routines to solve the preconditioner
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BlockTridiagSolve(void *data, hypre_ParCSRMatrix *A,
                            hypre_ParVector *b, hypre_ParVector *x) 
{
   HYPRE_Int                i, ind, nrows1, nrows2, *index_set1, *index_set2;
   double             *ffv, *uuv, *f1v, *f2v, *u1v, *u2v;
   HYPRE_ParCSRMatrix A21, A11, A22;
   hypre_ParVector    *F1, *U1, *F2, *U2;
   HYPRE_Solver       precon1, precon2;
   hypre_BlockTridiagData *b_data = (hypre_BlockTridiagData *) data;
 
   index_set1 = b_data->index_set1;
   index_set2 = b_data->index_set2;
   nrows1  = index_set1[0];
   nrows2  = index_set2[0];
   precon1 = b_data->precon1;
   precon2 = b_data->precon2;
   A11 = (HYPRE_ParCSRMatrix) b_data->A11;
   A22 = (HYPRE_ParCSRMatrix) b_data->A22;
   A21 = (HYPRE_ParCSRMatrix) b_data->A21;
   F1  = b_data->F1;
   U1  = b_data->U1;
   F2  = b_data->F2;
   U2  = b_data->U2;
   ffv = hypre_VectorData(hypre_ParVectorLocalVector(b));
   uuv = hypre_VectorData(hypre_ParVectorLocalVector(x));
   f1v = hypre_VectorData(hypre_ParVectorLocalVector(F1));
   u1v = hypre_VectorData(hypre_ParVectorLocalVector(U1));
   f2v = hypre_VectorData(hypre_ParVectorLocalVector(F2));
   u2v = hypre_VectorData(hypre_ParVectorLocalVector(U2));
   for (i = 0; i < nrows1; i++)
   {
      ind = index_set1[i+1];
      f1v[i] = ffv[ind];
      u1v[i] = 0.0;
   }
   HYPRE_BoomerAMGSolve(precon1, A11, (HYPRE_ParVector) F1, 
                        (HYPRE_ParVector) U1);
   for (i = 0; i < nrows2; i++)
   {
      ind = index_set2[i+1];
      f2v[i] = ffv[ind];
      u2v[i] = 0.0;
   }
   HYPRE_ParCSRMatrixMatvec(-1.0,A21,(HYPRE_ParVector) U1,1.0,
                            (HYPRE_ParVector) F2);
   HYPRE_BoomerAMGSolve(precon2, A22, (HYPRE_ParVector) F2, 
                        (HYPRE_ParVector) U2);
   for (i = 0; i < nrows1; i++)
   {
      ind = index_set1[i+1];
      uuv[ind] = u1v[i];
   }
   for (i = 0; i < nrows2; i++)
   {
      ind = index_set2[i+1];
      uuv[ind] = u2v[i];
   }
   return (0);
}

/*--------------------------------------------------------------------------
 * Routines to set the index set for block 1
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BlockTridiagSetIndexSet(void *data, HYPRE_Int n, HYPRE_Int *inds)
{
   HYPRE_Int i, ierr=0, *indices;
   hypre_BlockTridiagData *b_data = (hypre_BlockTridiagData *) data;

   if (n <= 0 || inds == NULL) ierr = 1;
   b_data->index_set1 = hypre_CTAlloc(HYPRE_Int, n+1);
   indices = b_data->index_set1;
   indices[0] = n;
   for (i = 0; i < n; i++) indices[i+1] = inds[i];
   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the strength threshold for AMG
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BlockTridiagSetAMGStrengthThreshold(void *data, double thresh)
{
   hypre_BlockTridiagData *b_data = data;
   b_data->threshold = thresh;
   return (0);
}

/*--------------------------------------------------------------------------
 * Routines to set the number of relaxation sweeps for AMG
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BlockTridiagSetAMGNumSweeps(void *data, HYPRE_Int nsweeps)
{
   hypre_BlockTridiagData *b_data = data;
   b_data->num_sweeps = nsweeps;
   return (0);
}

/*--------------------------------------------------------------------------
 * Routines to set the relaxation method for AMG
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BlockTridiagSetAMGRelaxType(void *data, HYPRE_Int relax_type)
{
   hypre_BlockTridiagData *b_data = data;
   b_data->relax_type = relax_type;
   return (0);
}

/*--------------------------------------------------------------------------
 * Routines to set the print level
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BlockTridiagSetPrintLevel(void *data, HYPRE_Int print_level)
{
   hypre_BlockTridiagData *b_data = (hypre_BlockTridiagData *) data;
   b_data->print_level = print_level;
   return (0);
}

