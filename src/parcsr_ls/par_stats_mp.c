/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "HYPRE_parcsr_ls_mp.h"
#include "hypre_parcsr_ls_mp.h"
#include "hypre_parcsr_ls_mup.h"
#include "hypre_parcsr_mv_mup.h"
#include "hypre_utilities_mup.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * hypre_MPAMGSetupStats
 *
 * Routine for getting matrix statistics from setup
 *
 * AHB - using block norm 6 (sum of all elements) instead of 1 (frobenius)
 *--------------------------------------------------------------------*/
#ifdef HYPRE_MIXED_PRECISION

HYPRE_Int
hypre_MPAMGSetupStats_mp( void               *amg_vdata,
                              hypre_ParCSRMatrix *A )
{
   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);

   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **P_array;

   hypre_CSRMatrix *A_diag, *A_diag_clone;
   void            *A_diag_data;
   HYPRE_Int       *A_diag_i;

   hypre_CSRMatrix *A_offd, *A_offd_clone;
   void            *A_offd_data;
   HYPRE_Int       *A_offd_i;

   hypre_CSRMatrix *P_diag, *P_diag_clone;
   void            *P_diag_data;
   HYPRE_Int       *P_diag_i;

   hypre_CSRMatrix *P_offd, *P_offd_clone;
   void            *P_offd_data;
   HYPRE_Int       *P_offd_i;

   HYPRE_Int      numrows;

   HYPRE_BigInt  *row_starts;

   HYPRE_Int      num_levels;
   HYPRE_Int      coarsen_type;
   HYPRE_Int      interp_type;
   HYPRE_Int      restri_type;
   HYPRE_Int      agg_interp_type;
   HYPRE_Int      measure_type;
   HYPRE_Int      agg_num_levels;
   hypre_double   global_nonzeros;

   hypre_double  *send_buff;
   hypre_double  *gather_buff;

   /* Local variables */

   HYPRE_Int       level;
   HYPRE_Int       j;
   HYPRE_BigInt    fine_size;

   HYPRE_Int       min_entries;
   HYPRE_Int       max_entries;

   HYPRE_Int       num_procs, my_id;
   HYPRE_Int       num_threads;


   hypre_double    min_rowsum;
   hypre_double    max_rowsum;
   hypre_double    sparse;


   HYPRE_Int       i;
   HYPRE_Int       ndigits[4];

   HYPRE_BigInt    coarse_size;
   HYPRE_Int       entries;

   hypre_double    avg_entries;
   hypre_double    rowsum;

   hypre_double    min_weight;
   hypre_double    max_weight;

   HYPRE_Int       global_min_e;
   HYPRE_Int       global_max_e;

   hypre_double    global_min_rsum;
   hypre_double    global_max_rsum;
   hypre_double    global_min_wt;
   hypre_double    global_max_wt;

   hypre_double  *num_mem;
   hypre_double  *num_coeffs;
   hypre_double  *num_variables;
   hypre_double   total_variables;
   hypre_double   operat_cmplxty;
   hypre_double   grid_cmplxty = 0;
   hypre_double   memory_cmplxty = 0;

   /* amg solve params */
   HYPRE_Int      max_iter;
   HYPRE_Int      cycle_type;
   HYPRE_Int      fcycle;
   HYPRE_Int     *num_grid_sweeps;
   HYPRE_Int     *grid_relax_type;
   HYPRE_Int      relax_order;
   HYPRE_Int    **grid_relax_points;
   hypre_double   tol;

   HYPRE_Int block_mode;
   HYPRE_Int block_size = 1;
   HYPRE_Int bnnz = 1;

   hypre_double tmp_norm;


   HYPRE_Int one = 1;
   HYPRE_Int minus_one = -1;
   HYPRE_Int zero = 0;
   hypre_double add_rlx_wt;
   HYPRE_Int six = 6;

   HYPRE_Precision precision;
   HYPRE_Precision  *precision_array;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   num_threads = hypre_NumThreads();

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);
   interp_type = hypre_ParAMGDataInterpType(amg_data);
   restri_type = hypre_ParAMGDataRestriction(amg_data); /* RL */
   agg_interp_type = hypre_ParAMGDataAggInterpType(amg_data);
   measure_type = hypre_ParAMGDataMeasureType(amg_data);
   agg_num_levels = hypre_ParAMGDataAggNumLevels(amg_data);
   precision_array = hypre_ParAMGDataPrecisionArray(amg_data);

   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/

   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_iter   = hypre_ParAMGDataMaxIter(amg_data);
   cycle_type = hypre_ParAMGDataCycleType(amg_data);
   fcycle     = hypre_ParAMGDataFCycle(amg_data);
   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order = hypre_ParAMGDataRelaxOrder(amg_data);
   tol = hypre_ParAMGDataTol(amg_data);

   send_buff = hypre_CAlloc_dbl((size_t)(six), (size_t)sizeof(hypre_double), HYPRE_MEMORY_HOST);
   gather_buff = hypre_CAlloc_dbl((size_t)(six), (size_t)sizeof(hypre_double), HYPRE_MEMORY_HOST);

   if (my_id == 0)
   {
      hypre_printf_dbl("\n\n Num MPI tasks = %d\n\n", num_procs);
      hypre_printf_dbl(" Num OpenMP threads = %d\n\n", num_threads);
      hypre_printf_dbl("\nMPAMG SETUP PARAMETERS:\n\n");
      hypre_printf_dbl(" Max levels = %d\n", hypre_ParAMGDataMaxLevels(amg_data));
      hypre_printf_dbl(" Num levels = %d\n\n", num_levels);
      hypre_printf_dbl(" Strength Threshold = %f\n",
                   hypre_ParAMGDataStrongThreshold(amg_data));
      hypre_printf_dbl(" Interpolation Truncation Factor = %f\n",
                   hypre_ParAMGDataTruncFactor(amg_data));
      hypre_printf_dbl(" Maximum Row Sum Threshold for Dependency Weakening = %f\n\n",
                   hypre_ParAMGDataMaxRowSum(amg_data));

      if (coarsen_type == 0)
      {
         hypre_printf_dbl(" Coarsening Type = Cleary-Luby-Jones-Plassman\n");
      }
      else if (hypre_abs(coarsen_type) == 1)
      {
         hypre_printf_dbl(" Coarsening Type = Ruge\n");
      }
      else if (hypre_abs(coarsen_type) == 2)
      {
         hypre_printf_dbl(" Coarsening Type = Ruge2B\n");
      }
      else if (hypre_abs(coarsen_type) == 3)
      {
         hypre_printf_dbl(" Coarsening Type = Ruge3\n");
      }
      else if (hypre_abs(coarsen_type) == 4)
      {
         hypre_printf_dbl(" Coarsening Type = Ruge 3c \n");
      }
      else if (hypre_abs(coarsen_type) == 5)
      {
         hypre_printf_dbl(" Coarsening Type = Ruge relax special points \n");
      }
      else if (hypre_abs(coarsen_type) == 6)
      {
         hypre_printf_dbl(" Coarsening Type = Falgout-CLJP \n");
      }
      else if (hypre_abs(coarsen_type) == 8)
      {
         hypre_printf_dbl(" Coarsening Type = PMIS \n");
      }
      else if (hypre_abs(coarsen_type) == 10)
      {
         hypre_printf_dbl(" Coarsening Type = HMIS \n");
      }
      else if (hypre_abs(coarsen_type) == 11)
      {
         hypre_printf_dbl(" Coarsening Type = Ruge 1st pass only \n");
      }
      else if (hypre_abs(coarsen_type) == 9)
      {
         hypre_printf_dbl(" Coarsening Type = PMIS fixed random \n");
      }
      else if (hypre_abs(coarsen_type) == 7)
      {
         hypre_printf_dbl(" Coarsening Type = CLJP, fixed random \n");
      }

      if (agg_num_levels > 0)
      {
         hypre_printf_dbl("\n No. of levels of aggressive coarsening: %d\n\n", agg_num_levels);
         if (agg_interp_type == 4)
         {
            hypre_printf_dbl(" Interpolation on agg. levels= multipass interpolation\n");
         }
         else if (agg_interp_type == 1)
         {
            hypre_printf_dbl(" Interpolation on agg. levels = 2-stage extended+i interpolation \n");
         }
         else if (agg_interp_type == 2)
         {
            hypre_printf_dbl(" Interpolation on agg. levels = 2-stage std interpolation \n");
         }
         else if (agg_interp_type == 3)
         {
            hypre_printf_dbl(" Interpolation on agg. levels = 2-stage extended interpolation \n");
         }
      }


      if (coarsen_type)
         hypre_printf_dbl(" measures are determined %s\n\n",
                      (measure_type ? "globally" : "locally"));

      hypre_printf_dbl( "\n No global partition option chosen.\n\n");

      if (interp_type == 0)
      {
         hypre_printf_dbl(" Interpolation = modified classical interpolation\n");
      }
      else if (interp_type == 1)
      {
         hypre_printf_dbl(" Interpolation = LS interpolation \n");
      }
      else if (interp_type == 2)
      {
         hypre_printf_dbl(" Interpolation = modified classical interpolation for hyperbolic PDEs\n");
      }
      else if (interp_type == 3)
      {
         hypre_printf_dbl(" Interpolation = direct interpolation with separation of weights\n");
      }
      else if (interp_type == 4)
      {
         hypre_printf_dbl(" Interpolation = multipass interpolation\n");
      }
      else if (interp_type == 5)
      {
         hypre_printf_dbl(" Interpolation = multipass interpolation with separation of weights\n");
      }
      else if (interp_type == 6)
      {
         hypre_printf_dbl(" Interpolation = extended+i interpolation\n");
      }
      else if (interp_type == 7)
      {
         hypre_printf_dbl(" Interpolation = extended+i interpolation (if no common C point)\n");
      }
      else if (interp_type == 12)
      {
         hypre_printf_dbl(" Interpolation = F-F interpolation\n");
      }
      else if (interp_type == 13)
      {
         hypre_printf_dbl(" Interpolation = F-F1 interpolation\n");
      }
      else if (interp_type == 14)
      {
         hypre_printf_dbl(" Interpolation = extended interpolation\n");
      }
      else if (interp_type == 15)
      {
         hypre_printf_dbl(" Interpolation = direct interpolation with separation of weights\n");
      }
      else if (interp_type == 16)
      {
         hypre_printf_dbl(" Interpolation = extended interpolation with MMs\n");
      }
      else if (interp_type == 17)
      {
         hypre_printf_dbl(" Interpolation = extended+i interpolation with MMs\n");
      }
      else if (interp_type == 8)
      {
         hypre_printf_dbl(" Interpolation = standard interpolation\n");
      }
      else if (interp_type == 9)
      {
         hypre_printf_dbl(" Interpolation = standard interpolation with separation of weights\n");
      }
      else if (interp_type == 100)
      {
         hypre_printf_dbl(" Interpolation = one-point interpolation \n");
      }

      if (restri_type == 1)
      {
         hypre_printf_dbl(" Restriction = local approximate ideal restriction (AIR-1)\n");
      }
      else if (restri_type == 2)
      {
         hypre_printf_dbl(" Restriction = local approximate ideal restriction (AIR-2)\n");
      }
      else if (restri_type == 15)
      {
         hypre_printf_dbl(" Restriction = local approximate ideal restriction (AIR-1.5)\n");
      }
      else if (restri_type >= 3)
      {
         hypre_printf_dbl(" Restriction = local approximate ideal restriction (Neumann AIR-%d)\n",
                      restri_type - 3);
      }

      hypre_printf_dbl( "\nOperator Matrix Information:\n\n");
   }

   ndigits[0] = hypre_ndigits_dbl(hypre_ParCSRMatrixGlobalNumRows(A_array[0]));
   ndigits[1] = hypre_ndigits_dbl(hypre_ParCSRMatrixNumNonzeros(A_array[0]));
   ndigits[0] = hypre_max(7, ndigits[0]);
   ndigits[1] = hypre_max(8, ndigits[1]);
   ndigits[2] = 4;
   for (level = 0; level < num_levels; level++)
   {
      fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
      global_nonzeros = (hypre_double) hypre_ParCSRMatrixNumNonzeros(A_array[level]);
      ndigits[2] = hypre_max(hypre_ndigits_dbl((HYPRE_BigInt) global_nonzeros / fine_size ), ndigits[2]);
   }
   ndigits[2] = ndigits[2] + 2;
   ndigits[3] = ndigits[0] + ndigits[1] + ndigits[2];

   if (my_id == 0)
   {
      hypre_printf_dbl("%*s", (ndigits[0] + 13), "nonzero");
      hypre_printf_dbl("%*s", (ndigits[1] + 15), "entries/row");
      hypre_printf_dbl("%18s\n", "row sums");
      hypre_printf_dbl("%s %*s ", "lev", ndigits[0], "rows");
      hypre_printf_dbl("%*s", ndigits[1], "entries");
      hypre_printf_dbl("%7s %5s %4s", "sparse", "min", "max");
      hypre_printf_dbl("%*s %8s %11s\n", (ndigits[2] + 2), "avg", "min", "max");
      for (i = 0; i < (49 + ndigits[3]); i++) { hypre_printf_dbl("%s", "="); }
      hypre_printf_dbl("\n");
   }

   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/

   num_coeffs = hypre_CAlloc_dbl((size_t) (num_levels), (size_t)sizeof(hypre_double), HYPRE_MEMORY_HOST);
   num_mem = hypre_CAlloc_dbl((size_t) (num_levels), (size_t)sizeof(hypre_double), HYPRE_MEMORY_HOST);

   num_variables = hypre_CAlloc_dbl((size_t) (num_levels), (size_t)sizeof(hypre_double), HYPRE_MEMORY_HOST);

   for (level = 0; level < num_levels; level++)
   {
      A_diag = hypre_ParCSRMatrixDiag(A_array[level]);
      A_diag_data = hypre_CSRMatrixData(A_diag);
      A_diag_i = hypre_CSRMatrixI(A_diag);

      A_offd = hypre_ParCSRMatrixOffd(A_array[level]);
      A_offd_data = hypre_CSRMatrixData(A_offd);
      A_offd_i = hypre_CSRMatrixI(A_offd);

      row_starts = hypre_ParCSRMatrixRowStarts(A_array[level]);

      fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);

      precision = precision_array[level];

      if (precision == HYPRE_REAL_DOUBLE)
	      hypre_ParCSRMatrixSetDNumNonzeros_dbl(A_array[level]);
      else if (precision == HYPRE_REAL_SINGLE)
	      hypre_ParCSRMatrixSetDNumNonzeros_flt(A_array[level]);
      else if (precision == HYPRE_REAL_LONGDOUBLE)
	      hypre_ParCSRMatrixSetDNumNonzeros_long_dbl(A_array[level]);

      global_nonzeros = (hypre_double) hypre_ParCSRMatrixDNumNonzeros(A_array[level]);
      num_coeffs[level] = global_nonzeros;
      num_mem[level] += global_nonzeros;
      num_variables[level] = (hypre_double)fine_size;

      sparse = global_nonzeros / ((hypre_double) fine_size * (hypre_double) fine_size);

      min_entries = 0;
      max_entries = 0;
      min_rowsum = 0.0;
      max_rowsum = 0.0;

      if (hypre_CSRMatrixNumRows(A_diag))
      {
        if (precision == HYPRE_REAL_DOUBLE)
	{
         min_entries = (A_diag_i[1] - A_diag_i[0]) + (A_offd_i[1] - A_offd_i[0]);
         for (j = A_diag_i[0]; j < A_diag_i[1]; j++)
         {
            min_rowsum += ((hypre_double *)A_diag_data)[j];
         }
         for (j = A_offd_i[0]; j < A_offd_i[1]; j++)
         {
            min_rowsum += ((hypre_double *) A_offd_data)[j];
         }

         max_rowsum = min_rowsum;

         for (j = 0; j < hypre_CSRMatrixNumRows(A_diag); j++)
         {
            entries = (A_diag_i[j + 1] - A_diag_i[j]) + (A_offd_i[j + 1] - A_offd_i[j]);
            min_entries = hypre_min(entries, min_entries);
            max_entries = hypre_max(entries, max_entries);

            rowsum = 0.0;
            for (i = A_diag_i[j]; i < A_diag_i[j + 1]; i++)
            {
               rowsum += ((hypre_double *) A_diag_data)[i];
            }

            for (i = A_offd_i[j]; i < A_offd_i[j + 1]; i++)
            {
               rowsum += ((hypre_double *) A_offd_data)[i];
            }

            min_rowsum = hypre_min(rowsum, min_rowsum);
            max_rowsum = hypre_max(rowsum, max_rowsum);
         }
        }
	else if (precision == HYPRE_REAL_SINGLE)
	{
         min_entries = (A_diag_i[1] - A_diag_i[0]) + (A_offd_i[1] - A_offd_i[0]);
         for (j = A_diag_i[0]; j < A_diag_i[1]; j++)
         {
            min_rowsum += (hypre_double) ((hypre_float *) A_diag_data)[j];
         }
         for (j = A_offd_i[0]; j < A_offd_i[1]; j++)
         {
            min_rowsum += (hypre_double) ((hypre_float *) A_offd_data)[j];
         }

         max_rowsum = min_rowsum;

         for (j = 0; j < hypre_CSRMatrixNumRows(A_diag); j++)
         {
            entries = (A_diag_i[j + 1] - A_diag_i[j]) + (A_offd_i[j + 1] - A_offd_i[j]);
            min_entries = hypre_min(entries, min_entries);
            max_entries = hypre_max(entries, max_entries);

            rowsum = 0.0;
            for (i = A_diag_i[j]; i < A_diag_i[j + 1]; i++)
            {
               rowsum += (hypre_double) ((hypre_float *) A_diag_data)[i];
            }

            for (i = A_offd_i[j]; i < A_offd_i[j + 1]; i++)
            {
               rowsum += (hypre_double) ((hypre_float *)A_offd_data)[i];
            }

            min_rowsum = hypre_min(rowsum, min_rowsum);
            max_rowsum = hypre_max(rowsum, max_rowsum);
         }
        }
	else if (precision == HYPRE_REAL_LONGDOUBLE)
	{
         min_entries = (A_diag_i[1] - A_diag_i[0]) + (A_offd_i[1] - A_offd_i[0]);
         for (j = A_diag_i[0]; j < A_diag_i[1]; j++)
         {
            min_rowsum += (hypre_double) ((hypre_long_double *) A_diag_data)[j];
         }
         for (j = A_offd_i[0]; j < A_offd_i[1]; j++)
         {
            min_rowsum += (hypre_double) ((hypre_long_double *) A_offd_data)[j];
         }

         max_rowsum = min_rowsum;

         for (j = 0; j < hypre_CSRMatrixNumRows(A_diag); j++)
         {
            entries = (A_diag_i[j + 1] - A_diag_i[j]) + (A_offd_i[j + 1] - A_offd_i[j]);
            min_entries = hypre_min(entries, min_entries);
            max_entries = hypre_max(entries, max_entries);

            rowsum = 0.0;
            for (i = A_diag_i[j]; i < A_diag_i[j + 1]; i++)
            {
               rowsum += (hypre_double) ((hypre_long_double *) A_diag_data)[i];
            }

            for (i = A_offd_i[j]; i < A_offd_i[j + 1]; i++)
            {
               rowsum += (hypre_double) ((hypre_long_double *)A_offd_data)[i];
            }

            min_rowsum = hypre_min(rowsum, min_rowsum);
            max_rowsum = hypre_max(rowsum, max_rowsum);
         }
        }
        avg_entries = global_nonzeros / ((hypre_double) fine_size);
      }

      numrows = (HYPRE_Int)(row_starts[1] - row_starts[0]);
      if (!numrows) /* if we don't have any rows, then don't have this count toward
                       min row sum or min num entries */
      {
         min_entries = 1000000;
         min_rowsum =  1.0e7;
      }

      send_buff[0] = - min_entries;
      send_buff[1] = max_entries;
      send_buff[2] = - min_rowsum;
      send_buff[3] = max_rowsum;

      MPI_Reduce(send_buff, gather_buff, 4, MPI_DOUBLE, MPI_MAX, 0, comm);

      if (my_id == 0)
      {
         global_min_e = - (HYPRE_Int)gather_buff[0];
         global_max_e = (HYPRE_Int)gather_buff[1];
         global_min_rsum = - gather_buff[2];
         global_max_rsum = gather_buff[3];

         hypre_printf_dbl("%3d %*b %*.0f  %0.3f  %4d %4d",
                      level, ndigits[0], fine_size, ndigits[1], global_nonzeros,
                      sparse, global_min_e, global_max_e);
         hypre_printf_dbl("  %*.1f  %10.3e  %10.3e\n", ndigits[2], avg_entries,
                      global_min_rsum, global_max_rsum);
      }
   }

   ndigits[0] = 5;
   if ((num_levels - 1))
   {
      ndigits[0] = hypre_max(hypre_ndigits_dbl(hypre_ParCSRMatrixGlobalNumRows(P_array[0])), ndigits[0]);
   }

   if (my_id == 0)
   {
      hypre_printf_dbl( "\n\nInterpolation Matrix Information:\n");

      hypre_printf_dbl("%*s ", (2 * ndigits[0] + 21), "entries/row");
      hypre_printf_dbl("%10s %10s %19s\n", "min", "max", "row sums");
      hypre_printf_dbl("lev %*s x %-*s min  max  avgW", ndigits[0], "rows", ndigits[0], "cols");
      hypre_printf_dbl("%11s %11s %9s %11s\n", "weight", "weight", "min", "max");
      for (i = 0; i < (70 + 2 * ndigits[0]); i++) { hypre_printf_dbl("%s", "="); }
      hypre_printf_dbl("\n");
   }

   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/

   for (level = 0; level < num_levels - 1; level++)
   {
      P_diag = hypre_ParCSRMatrixDiag(P_array[level]);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);

      P_offd = hypre_ParCSRMatrixOffd(P_array[level]);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);

      row_starts = hypre_ParCSRMatrixRowStarts(P_array[level]);

      fine_size = hypre_ParCSRMatrixGlobalNumRows(P_array[level]);
      coarse_size = hypre_ParCSRMatrixGlobalNumCols(P_array[level]);
      hypre_ParCSRMatrixSetDNumNonzeros_dbl(P_array[level]);
      global_nonzeros = (hypre_double) hypre_ParCSRMatrixDNumNonzeros(P_array[level]);
      num_mem[level] += global_nonzeros;

      min_weight = 1.0;
      max_weight = 0.0;
      max_rowsum = 0.0;
      min_rowsum = 0.0;
      min_entries = 0;
      max_entries = 0;

      precision = precision_array[level];

      if (hypre_CSRMatrixNumRows(P_diag))
      {
       if (precision == HYPRE_REAL_DOUBLE)
       {
         if (P_diag_data) { min_weight = ((hypre_double *) P_diag_data)[0]; }
         for (j = P_diag_i[0]; j < P_diag_i[1]; j++)
         {
            min_weight = hypre_min(min_weight, ((hypre_double *) P_diag_data)[j]);
            if (((hypre_double *) P_diag_data)[j] != 1.0)
            {
               max_weight = hypre_max(max_weight, ((hypre_double *) P_diag_data)[j]);
            }
            min_rowsum += ((hypre_double *) P_diag_data)[j];
         }
         for (j = P_offd_i[0]; j < P_offd_i[1]; j++)
         {
            min_weight = hypre_min(min_weight, ((hypre_double *) P_offd_data)[j]);
            if (((hypre_double *) P_offd_data)[j] != 1.0)
            {
               max_weight = hypre_max(max_weight, ((hypre_double *) P_offd_data)[j]);
            }
            min_rowsum += ((hypre_double *) P_offd_data)[j];
         }

         max_rowsum = min_rowsum;

         min_entries = (P_diag_i[1] - P_diag_i[0]) + (P_offd_i[1] - P_offd_i[0]);
         max_entries = 0;

         for (j = 0; j < hypre_CSRMatrixNumRows(P_diag); j++)
         {
            entries = (P_diag_i[j + 1] - P_diag_i[j]) + (P_offd_i[j + 1] - P_offd_i[j]);
            min_entries = hypre_min(entries, min_entries);
            max_entries = hypre_max(entries, max_entries);

            rowsum = 0.0;
            for (i = P_diag_i[j]; i < P_diag_i[j + 1]; i++)
            {
               min_weight = hypre_min(min_weight, ((hypre_double *) P_diag_data)[i]);
               if (((hypre_double *) P_diag_data)[i] != 1.0)
               {
                  max_weight = hypre_max(max_weight, ((hypre_double *) P_diag_data)[i]);
               }
               rowsum += ((hypre_double *) P_diag_data)[i];
            }

            for (i = P_offd_i[j]; i < P_offd_i[j + 1]; i++)
            {
               min_weight = hypre_min(min_weight, ((hypre_double *) P_offd_data)[i]);
               if (((hypre_double *) P_offd_data)[i] != 1.0)
               {
                  max_weight = hypre_max(max_weight, ((hypre_double *) P_offd_data)[i]);
               }
               rowsum += ((hypre_double *) P_offd_data)[i];
            }

            min_rowsum = hypre_min(rowsum, min_rowsum);
            max_rowsum = hypre_max(rowsum, max_rowsum);
         }
       }
       else if (precision == HYPRE_REAL_SINGLE)
       {
         if (P_diag_data) { min_weight = (hypre_double) ((hypre_float *)P_diag_data)[0]; }
         for (j = P_diag_i[0]; j < P_diag_i[1]; j++)
         {
            min_weight = hypre_min(min_weight, (hypre_double) ((hypre_float *)P_diag_data)[j]);
            if ((hypre_double) ((hypre_float *)P_diag_data)[j] != 1.0)
            {
               max_weight = hypre_max(max_weight, (hypre_double) ((hypre_float *) P_diag_data)[j]);
            }
            min_rowsum += (hypre_double) ((hypre_float *)P_diag_data)[j];
         }
         for (j = P_offd_i[0]; j < P_offd_i[1]; j++)
         {
            min_weight = hypre_min(min_weight, (hypre_double) ((hypre_float *)P_offd_data)[j]);
            if ((hypre_double)((hypre_float *)P_offd_data)[j] != 1.0)
            {
               max_weight = hypre_max(max_weight, (hypre_double) ((hypre_float *)P_offd_data)[j]);
            }
            min_rowsum += (hypre_double) ((hypre_float *)P_offd_data)[j];
         }

         max_rowsum = min_rowsum;

         min_entries = (P_diag_i[1] - P_diag_i[0]) + (P_offd_i[1] - P_offd_i[0]);
         max_entries = 0;

         for (j = 0; j < hypre_CSRMatrixNumRows(P_diag); j++)
         {
            entries = (P_diag_i[j + 1] - P_diag_i[j]) + (P_offd_i[j + 1] - P_offd_i[j]);
            min_entries = hypre_min(entries, min_entries);
            max_entries = hypre_max(entries, max_entries);

            rowsum = 0.0;
            for (i = P_diag_i[j]; i < P_diag_i[j + 1]; i++)
            {
               min_weight = hypre_min(min_weight, (hypre_double)((hypre_float *)P_diag_data)[i]);
               if ((hypre_double)((hypre_float *)P_diag_data)[i] != 1.0)
               {
                  max_weight = hypre_max(max_weight, (hypre_double) ((hypre_float *)P_diag_data)[i]);
               }
               rowsum += (hypre_double) ((hypre_float *)P_diag_data)[i];
            }

            for (i = P_offd_i[j]; i < P_offd_i[j + 1]; i++)
            {
               min_weight = hypre_min(min_weight, (hypre_double) ((hypre_float *)P_offd_data)[i]);
               if ((hypre_double)((hypre_float *)P_offd_data)[i] != 1.0)
               {
                  max_weight = hypre_max(max_weight, (hypre_double)((hypre_float *)P_offd_data)[i]);
               }
               rowsum += (hypre_double)((hypre_float *)P_offd_data)[i];
            }

            min_rowsum = hypre_min(rowsum, min_rowsum);
            max_rowsum = hypre_max(rowsum, max_rowsum);
         }
       }
       else if (precision == HYPRE_REAL_LONGDOUBLE)
       {
         if (P_diag_data) { min_weight = (hypre_double) ((hypre_long_double *)P_diag_data)[0]; }
         for (j = P_diag_i[0]; j < P_diag_i[1]; j++)
         {
            min_weight = hypre_min(min_weight, (hypre_double) ((hypre_long_double *)P_diag_data)[j]);
            if ((hypre_double) ((hypre_long_double *)P_diag_data)[j] != 1.0)
            {
               max_weight = hypre_max(max_weight, (hypre_double) ((hypre_long_double *) P_diag_data)[j]);
            }
            min_rowsum += (hypre_double) ((hypre_long_double *)P_diag_data)[j];
         }
         for (j = P_offd_i[0]; j < P_offd_i[1]; j++)
         {
            min_weight = hypre_min(min_weight, (hypre_double) ((hypre_long_double *)P_offd_data)[j]);
            if ((hypre_double)((hypre_long_double *)P_offd_data)[j] != 1.0)
            {
               max_weight = hypre_max(max_weight, (hypre_double) ((hypre_long_double *)P_offd_data)[j]);
            }
            min_rowsum += (hypre_double) ((hypre_long_double *)P_offd_data)[j];
         }

         max_rowsum = min_rowsum;

         min_entries = (P_diag_i[1] - P_diag_i[0]) + (P_offd_i[1] - P_offd_i[0]);
         max_entries = 0;

         for (j = 0; j < hypre_CSRMatrixNumRows(P_diag); j++)
         {
            entries = (P_diag_i[j + 1] - P_diag_i[j]) + (P_offd_i[j + 1] - P_offd_i[j]);
            min_entries = hypre_min(entries, min_entries);
            max_entries = hypre_max(entries, max_entries);

            rowsum = 0.0;
            for (i = P_diag_i[j]; i < P_diag_i[j + 1]; i++)
            {
               min_weight = hypre_min(min_weight, (hypre_double)((hypre_long_double *)P_diag_data)[i]);
               if ((hypre_double)((hypre_long_double *)P_diag_data)[i] != 1.0)
               {
                  max_weight = hypre_max(max_weight, (hypre_double) ((hypre_long_double *)P_diag_data)[i]);
               }
               rowsum += (hypre_double) ((hypre_long_double *)P_diag_data)[i];
            }

            for (i = P_offd_i[j]; i < P_offd_i[j + 1]; i++)
            {
               min_weight = hypre_min(min_weight, (hypre_double) ((hypre_long_double *)P_offd_data)[i]);
               if ((hypre_double)((hypre_long_double *)P_offd_data)[i] != 1.0)
               {
                  max_weight = hypre_max(max_weight, (hypre_double)((hypre_long_double *)P_offd_data)[i]);
               }
               rowsum += (hypre_double)((hypre_long_double *)P_offd_data)[i];
            }

            min_rowsum = hypre_min(rowsum, min_rowsum);
            max_rowsum = hypre_max(rowsum, max_rowsum);
         }
       }
      }
      avg_entries = ((HYPRE_Real) (global_nonzeros - coarse_size)) / ((HYPRE_Real) (
                                                                         fine_size - coarse_size));

      numrows = row_starts[1] - row_starts[0];
      if (!numrows) /* if we don't have any rows, then don't have this count toward
                       min row sum or min num entries */
      {
         min_entries = 1000000;
         min_rowsum =  1.0e7;
         min_weight = 1.0e7;
      }

      send_buff[0] = - (hypre_double) min_entries;
      send_buff[1] = (hypre_double) max_entries;
      send_buff[2] = - min_rowsum;
      send_buff[3] = max_rowsum;
      send_buff[4] = - min_weight;
      send_buff[5] = max_weight;

      MPI_Reduce(send_buff, gather_buff, 6, MPI_DOUBLE, MPI_MAX, 0, comm);

      if (my_id == 0)
      {
         global_min_e = - (HYPRE_Int)gather_buff[0];
         global_max_e = (HYPRE_Int)gather_buff[1];
         global_min_rsum = -gather_buff[2];
         global_max_rsum = gather_buff[3];
         global_min_wt = -gather_buff[4];
         global_max_wt = gather_buff[5];

         hypre_printf_dbl("%3d %*b x %-*b %3d  %3d",
                      level, ndigits[0], fine_size, ndigits[0], coarse_size,
                      global_min_e, global_max_e);
         hypre_printf_dbl("  %4.1f  %10.3e  %10.3e  %10.3e  %10.3e\n",
                      avg_entries, global_min_wt, global_max_wt,
                      global_min_rsum, global_max_rsum);
      }
   }

   total_variables = 0.0;
   operat_cmplxty = 0.0;
   for (j = 0; j < hypre_ParAMGDataNumLevels(amg_data); j++)
   {
      memory_cmplxty  += num_mem[j] / num_coeffs[0];
      operat_cmplxty  += num_coeffs[j] / num_coeffs[0];
      total_variables += num_variables[j];
   }
   if (num_variables[0] != 0)
   {
      grid_cmplxty = total_variables / num_variables[0];
   }

   if (my_id == 0 )
   {
      hypre_printf_dbl("\n\n");
      hypre_printf_dbl("     Complexity:   grid = %f\n", grid_cmplxty);
      hypre_printf_dbl("               operator = %f\n", operat_cmplxty);
      hypre_printf_dbl("                 memory = %f\n", memory_cmplxty);
      hypre_printf_dbl("\n\n");
   }

   if (my_id == 0)
   {
      hypre_printf_dbl("\n\nMPAMG SOLVER PARAMETERS:\n\n");
      hypre_printf_dbl( "  Maximum number of cycles:         %d \n", max_iter);
      hypre_printf_dbl( "  Stopping Tolerance:               %e \n", tol);
      if (fcycle)
      {
         hypre_printf_dbl( "  Full Multigrid. Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      }
      else
      {
         hypre_printf_dbl( "  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      }

      hypre_printf_dbl( "  Relaxation Parameters:\n");
      hypre_printf_dbl( "   Visiting Grid:                     down   up  coarse\n");
      hypre_printf_dbl( "            Number of sweeps:         %4d   %2d  %4d \n",
                       num_grid_sweeps[1],
                       num_grid_sweeps[2], num_grid_sweeps[3]);
      hypre_printf_dbl( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d \n",
                       grid_relax_type[1],
                       grid_relax_type[2], grid_relax_type[3]);
      hypre_printf_dbl( "   Point types, partial sweeps (1=C, -1=F):\n");
      if (grid_relax_points && grid_relax_type[1] != 8)
      {
         hypre_printf_dbl( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf_dbl("  %2d", grid_relax_points[1][j]);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf_dbl("  %2d", grid_relax_points[2][j]);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf_dbl("  %2d", grid_relax_points[3][j]);
         }
         hypre_printf_dbl( "\n");
      }
      else if (relax_order == 1 && grid_relax_type[1] != 8)
      {
         hypre_printf_dbl( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf_dbl("  %2d  %2d", one, minus_one);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf_dbl("  %2d  %2d", minus_one, one);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf_dbl("  %2d", zero);
         }
         hypre_printf_dbl( "\n");
      }
      else
      {
         hypre_printf_dbl( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf_dbl("  %2d", zero);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf_dbl("  %2d", zero);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf_dbl("  %2d", zero);
         }
         hypre_printf_dbl( "\n");
      }
      hypre_printf_dbl( "\n");
   }

   hypre_Free_dbl(num_coeffs, HYPRE_MEMORY_HOST);
   hypre_Free_dbl(num_mem, HYPRE_MEMORY_HOST);
   hypre_Free_dbl(num_variables, HYPRE_MEMORY_HOST);
   hypre_Free_dbl(send_buff, HYPRE_MEMORY_HOST);
   hypre_Free_dbl(gather_buff, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*---------------------------------------------------------------
 * hypre_MPAMGWriteSolverParams
 *---------------------------------------------------------------*/

HYPRE_Int
hypre_MPAMGWriteSolverParams_mp(void* data)
{
   hypre_ParAMGData  *amg_data = (hypre_ParAMGData*) data;

   /* amg solve params */
   HYPRE_Int          num_levels;
   HYPRE_Int          max_iter;
   HYPRE_Int          cycle_type;
   HYPRE_Int          fcycle;
   HYPRE_Int         *num_grid_sweeps;
   HYPRE_Int         *grid_relax_type;
   HYPRE_Int        **grid_relax_points;
   HYPRE_Int          relax_order;
   hypre_double       tol;

   /* amg output params */
   HYPRE_Int          amg_print_level;

   HYPRE_Int          j;
   HYPRE_Int          one = 1;
   HYPRE_Int          minus_one = -1;
   HYPRE_Int          zero = 0;

   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/

   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_iter   = hypre_ParAMGDataMaxIter(amg_data);
   cycle_type = hypre_ParAMGDataCycleType(amg_data);
   fcycle     = hypre_ParAMGDataFCycle(amg_data);
   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order = hypre_ParAMGDataRelaxOrder(amg_data);
   tol = hypre_ParAMGDataTol(amg_data);

   amg_print_level = hypre_ParAMGDataPrintLevel(amg_data);

   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/

   if (amg_print_level == 1 || amg_print_level == 3)
   {
      hypre_printf_dbl("\n\nMPAMG SOLVER PARAMETERS:\n\n");
      hypre_printf_dbl( "  Maximum number of cycles:         %d \n", max_iter);
      hypre_printf_dbl( "  Stopping Tolerance:               %e \n", tol);
      if (fcycle)
      {
         hypre_printf_dbl( "  Full Multigrid. Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      }
      else
      {
         hypre_printf_dbl( "  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      }
      hypre_printf_dbl( "  Relaxation Parameters:\n");
      hypre_printf_dbl( "   Visiting Grid:                     down   up  coarse\n");
      hypre_printf_dbl( "            Number of sweeps:         %4d   %2d  %4d \n",
                    num_grid_sweeps[1],
                    num_grid_sweeps[2], num_grid_sweeps[3]);
      hypre_printf_dbl( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d \n",
                    grid_relax_type[1],
                    grid_relax_type[2], grid_relax_type[3]);
      hypre_printf_dbl( "   Point types, partial sweeps (1=C, -1=F):\n");
      if (grid_relax_points)
      {
         hypre_printf_dbl( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf_dbl("  %2d", grid_relax_points[1][j]);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf_dbl("  %2d", grid_relax_points[2][j]);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf_dbl("  %2d", grid_relax_points[3][j]);
         }
         hypre_printf_dbl( "\n\n");
      }
      else if (relax_order == 1)
      {
         hypre_printf_dbl( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf_dbl("  %2d  %2d", one, minus_one);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf_dbl("  %2d  %2d", minus_one, one);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf_dbl("  %2d", zero);
         }
         hypre_printf_dbl( "\n\n");
      }
      else
      {
         hypre_printf_dbl( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf_dbl("  %2d", zero);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf_dbl("  %2d", zero);
         }
         hypre_printf_dbl( "\n");
         hypre_printf_dbl( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf_dbl("  %2d", zero);
         }
         hypre_printf_dbl( "\n\n");
      }

      hypre_printf_dbl( " Output flag (print_level): %d \n", amg_print_level);
   }

   return hypre_error_flag;
}

#endif
