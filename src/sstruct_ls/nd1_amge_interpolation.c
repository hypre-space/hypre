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





#include "_hypre_IJ_mv.h"
#include "headers.h"

#include "nd1_amge_interpolation.h"

/*
  Assume that we are given a fine and coarse topology and the
  coarse degrees of freedom (DOFs) have been chosen. Assume also,
  that the global interpolation matrix dof_DOF has a prescribed
  nonzero pattern. Then, the fine degrees of freedom can be split
  into 4 groups (here "i" stands for "interior"):

  NODEidof - dofs which are interpolated only from the DOF
             in one coarse vertex
  EDGEidof - dofs which are interpolated only from the DOFs
             in one coarse edge
  FACEidof - dofs which are interpolated only from the DOFs
             in one coarse face
  ELEMidof - dofs which are interpolated only from the DOFs
             in one coarse element

  The interpolation operator dof_DOF can be build in 4 steps, by
  consequently filling-in the rows corresponding to the above groups.
  The code below uses harmonic extension to extend the interpolation
  from one group to the next.
*/
HYPRE_Int hypre_ND1AMGeInterpolation (hypre_ParCSRMatrix       * Aee,
                                hypre_ParCSRMatrix       * ELEM_idof,
                                hypre_ParCSRMatrix       * FACE_idof,
                                hypre_ParCSRMatrix       * EDGE_idof,
                                hypre_ParCSRMatrix       * ELEM_FACE,
                                hypre_ParCSRMatrix       * ELEM_EDGE,
                                HYPRE_Int                  num_OffProcRows,
                                hypre_MaxwellOffProcRow ** OffProcRows,
                                hypre_IJMatrix           * IJ_dof_DOF)
{
   HYPRE_Int ierr = 0;

   HYPRE_Int  i, j, k;
   HYPRE_Int *offproc_rnums, *swap;

   hypre_ParCSRMatrix * dof_DOF = hypre_IJMatrixObject(IJ_dof_DOF);
   hypre_ParCSRMatrix * ELEM_DOF = ELEM_EDGE;
   hypre_ParCSRMatrix * ELEM_FACEidof;
   hypre_ParCSRMatrix * ELEM_EDGEidof;
   hypre_CSRMatrix *A, *P;
   HYPRE_Int numELEM = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(ELEM_EDGE));

   HYPRE_Int getrow_ierr;
   HYPRE_Int three_dimensional_problem;

   MPI_Comm comm= hypre_ParCSRMatrixComm(Aee);
   HYPRE_Int      myproc;

   hypre_MPI_Comm_rank(comm, &myproc);

#if 0
   hypre_IJMatrix * ij_dof_DOF = hypre_CTAlloc(hypre_IJMatrix, 1);
   /* Convert dof_DOF to IJ matrix, so we can use AddToValues */
   hypre_IJMatrixComm(ij_dof_DOF) = hypre_ParCSRMatrixComm(dof_DOF);
   hypre_IJMatrixRowPartitioning(ij_dof_DOF) =
      hypre_ParCSRMatrixRowStarts(dof_DOF);
   hypre_IJMatrixColPartitioning(ij_dof_DOF) =
      hypre_ParCSRMatrixColStarts(dof_DOF);
   hypre_IJMatrixObject(ij_dof_DOF) = dof_DOF;
   hypre_IJMatrixAssembleFlag(ij_dof_DOF) = 1;
#endif

  /* sort the offproc rows to get quicker comparison for later */
   if (num_OffProcRows)
   {
      offproc_rnums= hypre_TAlloc(HYPRE_Int, num_OffProcRows);
      swap         = hypre_TAlloc(HYPRE_Int, num_OffProcRows);
      for (i= 0; i< num_OffProcRows; i++)
      {
         offproc_rnums[i]=(OffProcRows[i] -> row);
         swap[i]         = i;
      }
   }

   if (num_OffProcRows > 1)
   {
      hypre_qsort2i(offproc_rnums, swap, 0, num_OffProcRows-1);
   }

   if (FACE_idof == EDGE_idof)
      three_dimensional_problem = 0;
   else
      three_dimensional_problem = 1;

   /* ELEM_FACEidof = ELEM_FACE x FACE_idof */
   if (three_dimensional_problem)
      ELEM_FACEidof = hypre_ParMatmul(ELEM_FACE, FACE_idof);

   /* ELEM_EDGEidof = ELEM_EDGE x EDGE_idof */
   ELEM_EDGEidof = hypre_ParMatmul(ELEM_EDGE, EDGE_idof);

   /* Loop over local coarse elements */
   k = hypre_ParCSRMatrixFirstRowIndex(ELEM_EDGE);
   for (i = 0; i < numELEM; i++, k++)
   {
      HYPRE_Int size1, size2;
      HYPRE_Int *col_ind0, *col_ind1, *col_ind2;

      HYPRE_Int num_DOF, *DOF0, *DOF;
      HYPRE_Int num_idof, *idof0, *idof;
      HYPRE_Int num_bdof, *bdof;

      double *boolean_data;

      /* Determine the coarse DOFs */
      hypre_ParCSRMatrixGetRow (ELEM_DOF, k, &num_DOF, &DOF0, &boolean_data);
      DOF= hypre_TAlloc(HYPRE_Int, num_DOF);
      for (j= 0; j< num_DOF; j++)
      {
         DOF[j]= DOF0[j];
      }
      hypre_ParCSRMatrixRestoreRow (ELEM_DOF, k, &num_DOF, &DOF0, &boolean_data);

      qsort0(DOF,0,num_DOF-1);

      /* Find the fine dofs interior for the current coarse element */
      hypre_ParCSRMatrixGetRow (ELEM_idof, k, &num_idof, &idof0, &boolean_data);
      idof= hypre_TAlloc(HYPRE_Int, num_idof);
      for (j= 0; j< num_idof; j++)
      {
         idof[j]= idof0[j];
      }
      hypre_ParCSRMatrixRestoreRow (ELEM_idof, k, &num_idof, &idof0, &boolean_data);

      /* Sort the interior dofs according to their global number */
      qsort0(idof,0,num_idof-1);

      /* Find the fine dofs on the boundary of the current coarse element */
      if (three_dimensional_problem)
      {
         hypre_ParCSRMatrixGetRow (ELEM_FACEidof, k, &size1, &col_ind0, &boolean_data);
         col_ind1= hypre_TAlloc(HYPRE_Int, size1);
         for (j= 0; j< size1; j++)
         {
            col_ind1[j]= col_ind0[j];
         }
         hypre_ParCSRMatrixRestoreRow (ELEM_FACEidof, k, &size1, &col_ind0, &boolean_data);
      }
      else
         size1 = 0;

      hypre_ParCSRMatrixGetRow (ELEM_EDGEidof, k, &size2, &col_ind0, &boolean_data);
      col_ind2= hypre_TAlloc(HYPRE_Int, size2);
      for (j= 0; j< size2; j++)
      {
         col_ind2[j]= col_ind0[j];
      }
      hypre_ParCSRMatrixRestoreRow (ELEM_EDGEidof, k, &size2, &col_ind0, &boolean_data);

      /* Merge and sort the boundary dofs according to their global number */
      num_bdof = size1 + size2;
      bdof = hypre_CTAlloc(HYPRE_Int, num_bdof);
      if (three_dimensional_problem)
         memcpy(bdof, col_ind1, size1*sizeof(HYPRE_Int));
      memcpy(bdof+size1, col_ind2, size2*sizeof(HYPRE_Int));

      qsort0(bdof,0,num_bdof-1);

      /* A = extract_rows(Aee, idof) */
      A = hypre_CSRMatrixCreate (num_idof, num_idof + num_bdof,
                                 num_idof * (num_idof + num_bdof));
      hypre_CSRMatrixInitialize(A);
      {
         HYPRE_Int *I = hypre_CSRMatrixI(A);
         HYPRE_Int *J = hypre_CSRMatrixJ(A);
         double *data = hypre_CSRMatrixData(A);

         HYPRE_Int *tmp_J;
         double *tmp_data;

         I[0] = 0;
         for (j = 0; j < num_idof; j++)
         {
            getrow_ierr= hypre_ParCSRMatrixGetRow (Aee, idof[j], &I[j+1], &tmp_J, &tmp_data);
            if (getrow_ierr <0)
               hypre_printf("getrow Aee off proc[%d] = \n",myproc);
            memcpy(J, tmp_J, I[j+1]*sizeof(HYPRE_Int));
            memcpy(data, tmp_data, I[j+1]*sizeof(double));
            J+= I[j+1];
            data+= I[j+1];
            hypre_ParCSRMatrixRestoreRow (Aee, idof[j], &I[j+1], &tmp_J, &tmp_data);
            I[j+1] += I[j];
         }
      }

      /* P = extract_rows(dof_DOF, idof+bdof) */
      P = hypre_CSRMatrixCreate (num_idof + num_bdof, num_DOF,
                                 (num_idof + num_bdof) * num_DOF);
      hypre_CSRMatrixInitialize(P);
      {
         HYPRE_Int *I = hypre_CSRMatrixI(P);
         HYPRE_Int *J = hypre_CSRMatrixJ(P);
         double *data = hypre_CSRMatrixData(P);
         HYPRE_Int     m;

         HYPRE_Int *tmp_J;
         double *tmp_data;
     
         I[0] = 0;
         for (j = 0; j < num_idof; j++)
         {
            getrow_ierr= hypre_ParCSRMatrixGetRow (dof_DOF, idof[j], &I[j+1], &tmp_J, &tmp_data);
            if (getrow_ierr >= 0)
            {
               memcpy(J, tmp_J, I[j+1]*sizeof(HYPRE_Int));
               memcpy(data, tmp_data, I[j+1]*sizeof(double));
               J+= I[j+1];
               data+= I[j+1];
               hypre_ParCSRMatrixRestoreRow (dof_DOF, idof[j], &I[j+1], &tmp_J, &tmp_data);
               I[j+1] += I[j];
            }
            else    /* row offproc */
            {
               hypre_ParCSRMatrixRestoreRow (dof_DOF, idof[j], &I[j+1], &tmp_J, &tmp_data);
              /* search for OffProcRows */
               m= 0;
               while (m < num_OffProcRows)
               {
                  if (offproc_rnums[m] == idof[j])
                  { 
                     break;
                  }
                  else
                  {
                     m++;
                  }
               }
               I[j+1]= (OffProcRows[swap[m]] -> ncols);
               tmp_J = (OffProcRows[swap[m]] -> cols);
               tmp_data= (OffProcRows[swap[m]] -> data);
               memcpy(J, tmp_J, I[j+1]*sizeof(HYPRE_Int));
               memcpy(data, tmp_data, I[j+1]*sizeof(double));
               J+= I[j+1];
               data+= I[j+1];
               I[j+1] += I[j];
            }

         }
         for ( ; j < num_idof + num_bdof; j++)
         {
            getrow_ierr= hypre_ParCSRMatrixGetRow (dof_DOF, bdof[j-num_idof], &I[j+1], &tmp_J, &tmp_data);
            if (getrow_ierr >= 0)
            {
               memcpy(J, tmp_J, I[j+1]*sizeof(HYPRE_Int));
               memcpy(data, tmp_data, I[j+1]*sizeof(double));
               J+= I[j+1];
               data+= I[j+1];
               hypre_ParCSRMatrixRestoreRow (dof_DOF, bdof[j-num_idof], &I[j+1], &tmp_J, &tmp_data);
               I[j+1] += I[j];
            }
            else    /* row offproc */
            {
               hypre_ParCSRMatrixRestoreRow (dof_DOF, bdof[j-num_idof], &I[j+1], &tmp_J, &tmp_data);
              /* search for OffProcRows */
               m= 0;
               while (m < num_OffProcRows)
               {
                  if (offproc_rnums[m] == bdof[j-num_idof])
                  {
                     break;
                  }
                  else
                  {
                     m++;
                  }
               }
               if (m>= num_OffProcRows)hypre_printf("here the mistake\n");
               I[j+1]= (OffProcRows[swap[m]] -> ncols);
               tmp_J = (OffProcRows[swap[m]] -> cols);
               tmp_data= (OffProcRows[swap[m]] -> data);
               memcpy(J, tmp_J, I[j+1]*sizeof(HYPRE_Int));
               memcpy(data, tmp_data, I[j+1]*sizeof(double));
               J+= I[j+1];
               data+= I[j+1];
               I[j+1] += I[j];
            }
         }
      }

      /* Pi = Aii^{-1} Aib Pb */
      hypre_HarmonicExtension (A, P, num_DOF, DOF,
                               num_idof, idof, num_bdof, bdof);

      /* Insert Pi in dof_DOF */
      {
         HYPRE_Int * ncols = hypre_CTAlloc(HYPRE_Int, num_idof);

         for (j = 0; j < num_idof; j++)
            ncols[j] = num_DOF;

         hypre_IJMatrixAddToValuesParCSR (IJ_dof_DOF,
                                          num_idof, ncols, idof,
                                          hypre_CSRMatrixJ(P),
                                          hypre_CSRMatrixData(P));

         hypre_TFree(ncols);
      }

      hypre_TFree(DOF);
      hypre_TFree(idof);
      if (three_dimensional_problem)
      {
         hypre_TFree(col_ind1);
      }
      hypre_TFree(col_ind2);
      hypre_TFree(bdof);

      hypre_CSRMatrixDestroy(A);
      hypre_CSRMatrixDestroy(P);
   }

#if 0
   hypre_TFree(ij_dof_DOF);
#endif

   if (three_dimensional_problem)
      hypre_ParCSRMatrixDestroy(ELEM_FACEidof);
   hypre_ParCSRMatrixDestroy(ELEM_EDGEidof);

   if (num_OffProcRows)
   {
      hypre_TFree(offproc_rnums);
      hypre_TFree(swap);
   }

   return ierr;
}




HYPRE_Int hypre_HarmonicExtension (hypre_CSRMatrix *A,
                             hypre_CSRMatrix *P,
                             HYPRE_Int num_DOF, HYPRE_Int *DOF,
                             HYPRE_Int num_idof, HYPRE_Int *idof,
                             HYPRE_Int num_bdof, HYPRE_Int *bdof)
{
   HYPRE_Int ierr = 0;

   HYPRE_Int i, j, k, l, m;
   double factor;

   HYPRE_Int *IA = hypre_CSRMatrixI(A);
   HYPRE_Int *JA = hypre_CSRMatrixJ(A);
   double *dataA = hypre_CSRMatrixData(A);

   HYPRE_Int *IP = hypre_CSRMatrixI(P);
   HYPRE_Int *JP = hypre_CSRMatrixJ(P);
   double *dataP = hypre_CSRMatrixData(P);

   double * Aii = hypre_CTAlloc(double, num_idof*num_idof);
   double * Pi = hypre_CTAlloc(double, num_idof*num_DOF);

   /* Loop over the rows of A */
   for (i = 0; i < num_idof; i++)
      for (j = IA[i]; j < IA[i+1]; j++)
      {
         /* Global to local*/
         k = hypre_BinarySearch(idof,JA[j], num_idof);
         /* If a column is a bdof, compute its participation in Pi = Aib x Pb */
         if (k == -1)
         {
            k = hypre_BinarySearch(bdof,JA[j], num_bdof);
            if (k > -1)
            {
               for (l = IP[k+num_idof]; l < IP[k+num_idof+1]; l++)
               {
                  m = hypre_BinarySearch(DOF,JP[l], num_DOF);
                  if (m > -1)
                  {
                     m+=i*num_DOF;
                    /* Pi[i*num_DOF+m] += dataA[j] * dataP[l];*/
                     Pi[m] += dataA[j] * dataP[l];
                  }
               }
            }
         }
         /* If a column is an idof, put it in Aii */
         else
            Aii[i*num_idof+k] = dataA[j];
      }

   /* Perform Gaussian elimination in [Aii, Pi] */
   for (j = 0; j < num_idof-1; j++)
      if (Aii[j*num_idof+j] != 0.0)
         for (i = j+1; i < num_idof; i++)
            if (Aii[i*num_idof+j] != 0.0)
            {
               factor = Aii[i*num_idof+j]/Aii[j*num_idof+j];
               for (m = j+1; m < num_idof; m++)
                  Aii[i*num_idof+m] -= factor * Aii[j*num_idof+m];
               for (m = 0; m < num_DOF; m++)
                  Pi[i*num_DOF+m] -= factor * Pi[j*num_DOF+m];
            }

   /* Back Substitution */
   for (i = num_idof-1; i >= 0; i--)
   {
      for (j = i+1; j < num_idof; j++)
         if (Aii[i*num_idof+j] != 0.0)
            for (m = 0; m < num_DOF; m++)
               Pi[i*num_DOF+m] -= Aii[i*num_idof+j] * Pi[j*num_DOF+m];

      for (m = 0; m < num_DOF; m++)
         Pi[i*num_DOF+m] /= Aii[i*num_idof+i];
   }

   /* Put -Pi back in P. We assume that each idof depends on _all_ DOFs */
   for (i = 0; i < num_idof; i++, JP += num_DOF, dataP += num_DOF)
      for (j = 0; j < num_DOF; j++)
      {
         JP[j]    = DOF[j];
         dataP[j] = -Pi[i*num_DOF+j];
      }

   hypre_TFree(Aii);
   hypre_TFree(Pi);

   return ierr;
}
