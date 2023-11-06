/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * hypre_BoomerAMGSetupStats
 *
 * Routine for getting matrix statistics from setup
 *
 * AHB - using block norm 6 (sum of all elements) instead of 1 (frobenius)
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetupStats( void               *amg_vdata,
                           hypre_ParCSRMatrix *A )
{
   hypre_GpuProfilingPushRange("AMGSetupStats");

   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);

   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **P_array;

   hypre_ParCSRBlockMatrix **A_block_array;
   hypre_ParCSRBlockMatrix **P_block_array;

   hypre_CSRMatrix *A_diag, *A_diag_clone;
   HYPRE_Real      *A_diag_data;
   HYPRE_Int       *A_diag_i;

   hypre_CSRBlockMatrix *A_block_diag;

   hypre_CSRMatrix *A_offd, *A_offd_clone;
   HYPRE_Real      *A_offd_data;
   HYPRE_Int       *A_offd_i;

   hypre_CSRBlockMatrix *A_block_offd;

   hypre_CSRMatrix *P_diag, *P_diag_clone;
   HYPRE_Real      *P_diag_data;
   HYPRE_Int       *P_diag_i;

   hypre_CSRBlockMatrix *P_block_diag;

   hypre_CSRMatrix *P_offd, *P_offd_clone;
   HYPRE_Real      *P_offd_data;
   HYPRE_Int       *P_offd_i;

   hypre_CSRBlockMatrix *P_block_offd;


   HYPRE_Int      numrows;

   HYPRE_BigInt  *row_starts;


   HYPRE_Int      num_levels;
   HYPRE_Int      coarsen_type;
   HYPRE_Int      interp_type;
   HYPRE_Int      restri_type;
   HYPRE_Int      agg_interp_type;
   HYPRE_Int      measure_type;
   HYPRE_Int      agg_num_levels;
   HYPRE_Real   global_nonzeros;

   HYPRE_Real  *send_buff;
   HYPRE_Real  *gather_buff;

   /* Local variables */

   HYPRE_Int       level;
   HYPRE_Int       j;
   HYPRE_BigInt    fine_size;

   HYPRE_Int       min_entries;
   HYPRE_Int       max_entries;

   HYPRE_Int       num_procs, my_id;
   HYPRE_Int       num_threads;


   HYPRE_Real    min_rowsum;
   HYPRE_Real    max_rowsum;
   HYPRE_Real    sparse;


   HYPRE_Int       i;
   HYPRE_Int       ndigits[4];

   HYPRE_BigInt    coarse_size;
   HYPRE_Int       entries;

   HYPRE_Real    avg_entries;
   HYPRE_Real    rowsum;

   HYPRE_Real    min_weight;
   HYPRE_Real    max_weight;

   HYPRE_Int     global_min_e;
   HYPRE_Int     global_max_e;

   HYPRE_Real    global_min_rsum;
   HYPRE_Real    global_max_rsum;
   HYPRE_Real    global_min_wt;
   HYPRE_Real    global_max_wt;

   HYPRE_Real  *num_mem;
   HYPRE_Real  *num_coeffs;
   HYPRE_Real  *num_variables;
   HYPRE_Real   total_variables;
   HYPRE_Real   operat_cmplxty;
   HYPRE_Real   grid_cmplxty = 0;
   HYPRE_Real   memory_cmplxty = 0;

   /* amg solve params */
   HYPRE_Int      max_iter;
   HYPRE_Int      cycle_type;
   HYPRE_Int      fcycle;
   HYPRE_Int     *num_grid_sweeps;
   HYPRE_Int     *grid_relax_type;
   HYPRE_Int      relax_order;
   HYPRE_Int    **grid_relax_points;
   HYPRE_Real  *relax_weight;
   HYPRE_Real  *omega;
   HYPRE_Real   tol;

   HYPRE_Int block_mode;
   HYPRE_Int block_size = 1;
   HYPRE_Int bnnz = 1;

   HYPRE_Real tmp_norm;


   HYPRE_Int one = 1;
   HYPRE_Int minus_one = -1;
   HYPRE_Int zero = 0;
   HYPRE_Int smooth_type;
   HYPRE_Int smooth_num_levels;
   HYPRE_Int additive;
   HYPRE_Int mult_additive;
   HYPRE_Int simple;
   HYPRE_Int add_end;
   HYPRE_Int add_rlx;
   HYPRE_Real add_rlx_wt;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = hypre_NumThreads();

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);
   interp_type = hypre_ParAMGDataInterpType(amg_data);
   restri_type = hypre_ParAMGDataRestriction(amg_data); /* RL */
   agg_interp_type = hypre_ParAMGDataAggInterpType(amg_data);
   measure_type = hypre_ParAMGDataMeasureType(amg_data);
   smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   agg_num_levels = hypre_ParAMGDataAggNumLevels(amg_data);
   additive = hypre_ParAMGDataAdditive(amg_data);
   mult_additive = hypre_ParAMGDataMultAdditive(amg_data);
   simple = hypre_ParAMGDataSimple(amg_data);
   add_end = hypre_ParAMGDataAddLastLvl(amg_data);
   add_rlx = hypre_ParAMGDataAddRelaxType(amg_data);
   add_rlx_wt = hypre_ParAMGDataAddRelaxWt(amg_data);

   A_block_array = hypre_ParAMGDataABlockArray(amg_data);
   P_block_array = hypre_ParAMGDataPBlockArray(amg_data);

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
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
   relax_order = hypre_ParAMGDataRelaxOrder(amg_data);
   omega = hypre_ParAMGDataOmega(amg_data);
   tol = hypre_ParAMGDataTol(amg_data);

   block_mode = hypre_ParAMGDataBlockMode(amg_data);

   send_buff     = hypre_CTAlloc(HYPRE_Real,  6, HYPRE_MEMORY_HOST);
   gather_buff = hypre_CTAlloc(HYPRE_Real, 6, HYPRE_MEMORY_HOST);

   if (my_id == 0)
   {
      hypre_printf("\n\n Num MPI tasks = %d\n\n", num_procs);
      hypre_printf(" Num OpenMP threads = %d\n\n", num_threads);
      hypre_printf("\nBoomerAMG SETUP PARAMETERS:\n\n");
      hypre_printf(" Max levels = %d\n", hypre_ParAMGDataMaxLevels(amg_data));
      hypre_printf(" Num levels = %d\n\n", num_levels);
      hypre_printf(" Strength Threshold = %f\n",
                   hypre_ParAMGDataStrongThreshold(amg_data));
      hypre_printf(" Interpolation Truncation Factor = %f\n",
                   hypre_ParAMGDataTruncFactor(amg_data));
      hypre_printf(" Maximum Row Sum Threshold for Dependency Weakening = %f\n\n",
                   hypre_ParAMGDataMaxRowSum(amg_data));

      if (coarsen_type == 0)
      {
         hypre_printf(" Coarsening Type = Cleary-Luby-Jones-Plassman\n");
      }
      else if (hypre_abs(coarsen_type) == 1)
      {
         hypre_printf(" Coarsening Type = Ruge\n");
      }
      else if (hypre_abs(coarsen_type) == 2)
      {
         hypre_printf(" Coarsening Type = Ruge2B\n");
      }
      else if (hypre_abs(coarsen_type) == 3)
      {
         hypre_printf(" Coarsening Type = Ruge3\n");
      }
      else if (hypre_abs(coarsen_type) == 4)
      {
         hypre_printf(" Coarsening Type = Ruge 3c \n");
      }
      else if (hypre_abs(coarsen_type) == 5)
      {
         hypre_printf(" Coarsening Type = Ruge relax special points \n");
      }
      else if (hypre_abs(coarsen_type) == 6)
      {
         hypre_printf(" Coarsening Type = Falgout-CLJP \n");
      }
      else if (hypre_abs(coarsen_type) == 8)
      {
         hypre_printf(" Coarsening Type = PMIS \n");
      }
      else if (hypre_abs(coarsen_type) == 10)
      {
         hypre_printf(" Coarsening Type = HMIS \n");
      }
      else if (hypre_abs(coarsen_type) == 11)
      {
         hypre_printf(" Coarsening Type = Ruge 1st pass only \n");
      }
      else if (hypre_abs(coarsen_type) == 9)
      {
         hypre_printf(" Coarsening Type = PMIS fixed random \n");
      }
      else if (hypre_abs(coarsen_type) == 7)
      {
         hypre_printf(" Coarsening Type = CLJP, fixed random \n");
      }
      else if (hypre_abs(coarsen_type) == 21) /* BM Aug 29, 2006 */
      {
         hypre_printf(" Coarsening Type = CGC \n");
      }
      else if (hypre_abs(coarsen_type) == 22) /* BM Aug 29, 2006 */
      {
         hypre_printf(" Coarsening Type = CGC-E \n");
      }
      /*if (coarsen_type > 0)
        {
        hypre_printf(" Hybrid Coarsening (switch to CLJP when coarsening slows)\n");
        }*/

      if (agg_num_levels > 0)
      {
         hypre_printf("\n No. of levels of aggressive coarsening: %d\n\n", agg_num_levels);
         if (agg_interp_type == 4)
         {
            hypre_printf(" Interpolation on agg. levels= multipass interpolation\n");
         }
         else if (agg_interp_type == 1)
         {
            hypre_printf(" Interpolation on agg. levels = 2-stage extended+i interpolation \n");
         }
         else if (agg_interp_type == 2)
         {
            hypre_printf(" Interpolation on agg. levels = 2-stage std interpolation \n");
         }
         else if (agg_interp_type == 3)
         {
            hypre_printf(" Interpolation on agg. levels = 2-stage extended interpolation \n");
         }
      }


      if (coarsen_type)
         hypre_printf(" measures are determined %s\n\n",
                      (measure_type ? "globally" : "locally"));

      hypre_printf( "\n No global partition option chosen.\n\n");

      if (interp_type == 0)
      {
         hypre_printf(" Interpolation = modified classical interpolation\n");
      }
      else if (interp_type == 1)
      {
         hypre_printf(" Interpolation = LS interpolation \n");
      }
      else if (interp_type == 2)
      {
         hypre_printf(" Interpolation = modified classical interpolation for hyperbolic PDEs\n");
      }
      else if (interp_type == 3)
      {
         hypre_printf(" Interpolation = direct interpolation with separation of weights\n");
      }
      else if (interp_type == 4)
      {
         hypre_printf(" Interpolation = multipass interpolation\n");
      }
      else if (interp_type == 5)
      {
         hypre_printf(" Interpolation = multipass interpolation with separation of weights\n");
      }
      else if (interp_type == 6)
      {
         hypre_printf(" Interpolation = extended+i interpolation\n");
      }
      else if (interp_type == 7)
      {
         hypre_printf(" Interpolation = extended+i interpolation (if no common C point)\n");
      }
      else if (interp_type == 12)
      {
         hypre_printf(" Interpolation = F-F interpolation\n");
      }
      else if (interp_type == 13)
      {
         hypre_printf(" Interpolation = F-F1 interpolation\n");
      }
      else if (interp_type == 14)
      {
         hypre_printf(" Interpolation = extended interpolation\n");
      }
      else if (interp_type == 15)
      {
         hypre_printf(" Interpolation = direct interpolation with separation of weights\n");
      }
      else if (interp_type == 16)
      {
         hypre_printf(" Interpolation = extended interpolation with MMs\n");
      }
      else if (interp_type == 17)
      {
         hypre_printf(" Interpolation = extended+i interpolation with MMs\n");
      }
      else if (interp_type == 8)
      {
         hypre_printf(" Interpolation = standard interpolation\n");
      }
      else if (interp_type == 9)
      {
         hypre_printf(" Interpolation = standard interpolation with separation of weights\n");
      }
      else if (interp_type == 10)
      {
         hypre_printf(" Interpolation = block classical interpolation for nodal systems AMG\n");
      }
      else if (interp_type == 11)
      {
         hypre_printf(" Interpolation = block classical interpolation with diagonal blocks\n");
         hypre_printf("                 for nodal systems AMG\n");
      }
      else if (interp_type == 24)
      {
         hypre_printf(" Interpolation = block direct interpolation \n");
         hypre_printf("                 for nodal systems AMG\n");
      }
      else if (interp_type == 100)
      {
         hypre_printf(" Interpolation = one-point interpolation \n");
      }

      if (restri_type == 1)
      {
         hypre_printf(" Restriction = local approximate ideal restriction (AIR-1)\n");
      }
      else if (restri_type == 2)
      {
         hypre_printf(" Restriction = local approximate ideal restriction (AIR-2)\n");
      }
      else if (restri_type == 15)
      {
         hypre_printf(" Restriction = local approximate ideal restriction (AIR-1.5)\n");
      }
      else if (restri_type >= 3)
      {
         hypre_printf(" Restriction = local approximate ideal restriction (Neumann AIR-%d)\n",
                      restri_type - 3);
      }

      if (block_mode)
      {
         hypre_printf( "\nBlock Operator Matrix Information:\n");
         hypre_printf( "(Row sums and weights use sum of all elements in the block -keeping signs)\n\n");
      }
      else
      {
         hypre_printf( "\nOperator Matrix Information:\n\n");
      }
   }

   if (block_mode)
   {
      ndigits[0] = hypre_ndigits(hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[0]));
      ndigits[1] = hypre_ndigits(hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[0]));
   }
   else
   {
      ndigits[0] = hypre_ndigits(hypre_ParCSRMatrixGlobalNumRows(A_array[0]));
      ndigits[1] = hypre_ndigits(hypre_ParCSRMatrixNumNonzeros(A_array[0]));
   }
   ndigits[0] = hypre_max(7, ndigits[0]);
   ndigits[1] = hypre_max(8, ndigits[1]);
   ndigits[2] = 4;
   for (level = 0; level < num_levels; level++)
   {

      if (block_mode)
      {
         fine_size = hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[level]);
         global_nonzeros = hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[level]);
         ndigits[2] = hypre_max(hypre_ndigits((HYPRE_BigInt) global_nonzeros / fine_size ), ndigits[2]);
      }
      else
      {
         fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
         global_nonzeros = hypre_ParCSRMatrixNumNonzeros(A_array[level]);
         ndigits[2] = hypre_max(hypre_ndigits((HYPRE_BigInt) global_nonzeros / fine_size ), ndigits[2]);
      }

   }
   ndigits[2] = ndigits[2] + 2;
   ndigits[3] = ndigits[0] + ndigits[1] + ndigits[2];

   if (my_id == 0)
   {
      hypre_printf("%*s", (ndigits[0] + 13), "nonzero");
      hypre_printf("%*s", (ndigits[1] + 15), "entries/row");
      hypre_printf("%18s\n", "row sums");
      hypre_printf("%s %*s ", "lev", ndigits[0], "rows");
      hypre_printf("%*s", ndigits[1], "entries");
      hypre_printf("%7s %5s %4s", "sparse", "min", "max");
      hypre_printf("%*s %8s %11s\n", (ndigits[2] + 2), "avg", "min", "max");
      for (i = 0; i < (49 + ndigits[3]); i++) { hypre_printf("%s", "="); }
      hypre_printf("\n");
   }

   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/

   num_coeffs = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);
   num_mem = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);

   num_variables = hypre_CTAlloc(HYPRE_Real, num_levels, HYPRE_MEMORY_HOST);

   for (level = 0; level < num_levels; level++)
   {

      if (block_mode)
      {
         A_block_diag = hypre_ParCSRBlockMatrixDiag(A_block_array[level]);
         A_diag_data = hypre_CSRBlockMatrixData(A_block_diag);
         A_diag_i = hypre_CSRBlockMatrixI(A_block_diag);

         A_block_offd = hypre_ParCSRBlockMatrixOffd(A_block_array[level]);
         A_offd_data = hypre_CSRMatrixData(A_block_offd);
         A_offd_i = hypre_CSRMatrixI(A_block_offd);

         block_size =  hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]);
         bnnz = block_size * block_size;

         row_starts = hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]);

         fine_size = hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[level]);
         global_nonzeros = hypre_ParCSRBlockMatrixDNumNonzeros(A_block_array[level]);
         num_coeffs[level] = global_nonzeros;
         num_mem[level] = global_nonzeros;
         num_variables[level] = (HYPRE_Real) fine_size;

         sparse = global_nonzeros / ((HYPRE_Real) fine_size * (HYPRE_Real) fine_size);

         min_entries = 0;
         max_entries = 0;
         min_rowsum = 0.0;
         max_rowsum = 0.0;

         if (hypre_CSRBlockMatrixNumRows(A_block_diag))
         {
            min_entries = (A_diag_i[1] - A_diag_i[0]) + (A_offd_i[1] - A_offd_i[0]);
            for (j = A_diag_i[0]; j < A_diag_i[1]; j++)
            {
               hypre_CSRBlockMatrixBlockNorm(6, &A_diag_data[j * bnnz], &tmp_norm, block_size);
               min_rowsum += tmp_norm;
            }

            for (j = A_offd_i[0]; j < A_offd_i[1]; j++)
            {
               hypre_CSRBlockMatrixBlockNorm(6, &A_offd_data[j * bnnz], &tmp_norm, block_size);
               min_rowsum += tmp_norm;
            }

            max_rowsum = min_rowsum;

            for (j = 0; j < hypre_CSRBlockMatrixNumRows(A_block_diag); j++)
            {
               entries = (A_diag_i[j + 1] - A_diag_i[j]) + (A_offd_i[j + 1] - A_offd_i[j]);
               min_entries = hypre_min(entries, min_entries);
               max_entries = hypre_max(entries, max_entries);

               rowsum = 0.0;
               for (i = A_diag_i[j]; i < A_diag_i[j + 1]; i++)
               {
                  hypre_CSRBlockMatrixBlockNorm(6, &A_diag_data[i * bnnz], &tmp_norm, block_size);
                  rowsum += tmp_norm;
               }
               for (i = A_offd_i[j]; i < A_offd_i[j + 1]; i++)
               {
                  hypre_CSRBlockMatrixBlockNorm(6, &A_offd_data[i * bnnz], &tmp_norm, block_size);
                  rowsum += tmp_norm;
               }
               min_rowsum = hypre_min(rowsum, min_rowsum);
               max_rowsum = hypre_max(rowsum, max_rowsum);
            }
         }
         avg_entries = global_nonzeros / ((HYPRE_Real) fine_size);
      }
      else
      {
         A_diag = hypre_ParCSRMatrixDiag(A_array[level]);
         if ( hypre_GetActualMemLocation(hypre_CSRMatrixMemoryLocation(A_diag)) != hypre_MEMORY_HOST )
         {
            A_diag_clone = hypre_CSRMatrixClone_v2(A_diag, 1, HYPRE_MEMORY_HOST);
         }
         else
         {
            A_diag_clone = A_diag;
         }
         A_diag_data = hypre_CSRMatrixData(A_diag_clone);
         A_diag_i = hypre_CSRMatrixI(A_diag_clone);

         A_offd = hypre_ParCSRMatrixOffd(A_array[level]);
         if ( hypre_GetActualMemLocation(hypre_CSRMatrixMemoryLocation(A_offd)) != hypre_MEMORY_HOST )
         {
            A_offd_clone = hypre_CSRMatrixClone_v2(A_offd, 1, HYPRE_MEMORY_HOST);
         }
         else
         {
            A_offd_clone = A_offd;
         }
         A_offd_data = hypre_CSRMatrixData(A_offd_clone);
         A_offd_i = hypre_CSRMatrixI(A_offd_clone);

         row_starts = hypre_ParCSRMatrixRowStarts(A_array[level]);

         fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
         global_nonzeros = hypre_ParCSRMatrixDNumNonzeros(A_array[level]);
         num_coeffs[level] = global_nonzeros;
         if (level == 0)
         {
            num_mem[level] += global_nonzeros;
         }
         if (level == 0 && (additive == 0 || mult_additive == 0) )
         {
            num_mem[level] += global_nonzeros;
         }
         if (level > 0)
         {
            if (simple > level || simple == -1)
            {
               num_mem[level] += global_nonzeros;
            }
         }
         num_variables[level] = (HYPRE_Real) fine_size;

         sparse = global_nonzeros / ((HYPRE_Real) fine_size * (HYPRE_Real) fine_size);

         min_entries = 0;
         max_entries = 0;
         min_rowsum = 0.0;
         max_rowsum = 0.0;

         if (hypre_CSRMatrixNumRows(A_diag))
         {
            min_entries = (A_diag_i[1] - A_diag_i[0]) + (A_offd_i[1] - A_offd_i[0]);
            for (j = A_diag_i[0]; j < A_diag_i[1]; j++)
            {
               min_rowsum += A_diag_data[j];
            }
            for (j = A_offd_i[0]; j < A_offd_i[1]; j++)
            {
               min_rowsum += A_offd_data[j];
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
                  rowsum += A_diag_data[i];
               }

               for (i = A_offd_i[j]; i < A_offd_i[j + 1]; i++)
               {
                  rowsum += A_offd_data[i];
               }

               min_rowsum = hypre_min(rowsum, min_rowsum);
               max_rowsum = hypre_max(rowsum, max_rowsum);
            }
         }
         avg_entries = global_nonzeros / ((HYPRE_Real) fine_size);

         if (A_diag_clone != A_diag)
         {
            hypre_CSRMatrixDestroy(A_diag_clone);
         }

         if (A_offd_clone != A_offd)
         {
            hypre_CSRMatrixDestroy(A_offd_clone);
         }
      }

      numrows = (HYPRE_Int)(row_starts[1] - row_starts[0]);
      if (!numrows) /* if we don't have any rows, then don't have this count toward
                       min row sum or min num entries */
      {
         min_entries = 1000000;
         min_rowsum =  1.0e7;
      }

      send_buff[0] = - (HYPRE_Real) min_entries;
      send_buff[1] = (HYPRE_Real) max_entries;
      send_buff[2] = - min_rowsum;
      send_buff[3] = max_rowsum;

      hypre_MPI_Reduce(send_buff, gather_buff, 4, HYPRE_MPI_REAL, hypre_MPI_MAX, 0, comm);

      if (my_id == 0)
      {
         global_min_e = - (HYPRE_Int)gather_buff[0];
         global_max_e = (HYPRE_Int)gather_buff[1];
         global_min_rsum = - gather_buff[2];
         global_max_rsum = gather_buff[3];

         hypre_printf("%3d %*b %*.0f  %0.3f  %4d %4d",
                      level, ndigits[0], fine_size, ndigits[1], global_nonzeros,
                      sparse, global_min_e, global_max_e);
         hypre_printf("  %*.1f  %10.3e  %10.3e\n", ndigits[2], avg_entries,
                      global_min_rsum, global_max_rsum);
      }
   }

   ndigits[0] = 5;
   if ((num_levels - 1))
   {
      if (block_mode)
      {
         ndigits[0] = hypre_max(hypre_ndigits(hypre_ParCSRBlockMatrixGlobalNumRows(P_block_array[0])),
                                ndigits[0]);
      }
      else
      {
         ndigits[0] = hypre_max(hypre_ndigits(hypre_ParCSRMatrixGlobalNumRows(P_array[0])), ndigits[0]);
      }
   }

   if (my_id == 0)
   {
      if (block_mode)
      {
         hypre_printf( "\n\nBlock Interpolation Matrix Information:\n\n");
         hypre_printf( "(Row sums and weights use sum of all elements in the block - keeping signs)\n\n");
      }
      else
      {
         hypre_printf( "\n\nInterpolation Matrix Information:\n");

      }

      hypre_printf("%*s ", (2 * ndigits[0] + 21), "entries/row");
      hypre_printf("%10s %10s %19s\n", "min", "max", "row sums");
      hypre_printf("lev %*s x %-*s min  max  avgW", ndigits[0], "rows", ndigits[0], "cols");
      hypre_printf("%11s %11s %9s %11s\n", "weight", "weight", "min", "max");
      for (i = 0; i < (70 + 2 * ndigits[0]); i++) { hypre_printf("%s", "="); }
      hypre_printf("\n");
   }

   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/

   for (level = 0; level < num_levels - 1; level++)
   {

      if (block_mode)
      {
         P_block_diag = hypre_ParCSRBlockMatrixDiag(P_block_array[level]);
         P_diag_data = hypre_CSRBlockMatrixData(P_block_diag);
         P_diag_i = hypre_CSRBlockMatrixI(P_block_diag);

         P_block_offd = hypre_ParCSRBlockMatrixOffd(P_block_array[level]);
         P_offd_data = hypre_CSRBlockMatrixData(P_block_offd);
         P_offd_i = hypre_CSRBlockMatrixI(P_block_offd);

         row_starts = hypre_ParCSRBlockMatrixRowStarts(P_block_array[level]);

         fine_size = hypre_ParCSRBlockMatrixGlobalNumRows(P_block_array[level]);
         coarse_size = hypre_ParCSRBlockMatrixGlobalNumCols(P_block_array[level]);
         global_nonzeros = hypre_ParCSRBlockMatrixDNumNonzeros(P_block_array[level]);
         num_mem[level] += global_nonzeros;

         min_weight = 1.0;
         max_weight = 0.0;
         max_rowsum = 0.0;
         min_rowsum = 0.0;
         min_entries = 0;
         max_entries = 0;

         if (hypre_CSRBlockMatrixNumRows(P_block_diag))
         {
            if (hypre_CSRBlockMatrixNumCols(P_block_diag))
            {
               hypre_CSRBlockMatrixBlockNorm(6, &P_diag_data[0], &tmp_norm, block_size);
               min_weight = tmp_norm;
            }


            for (j = P_diag_i[0]; j < P_diag_i[1]; j++)
            {
               hypre_CSRBlockMatrixBlockNorm(6, &P_diag_data[j * bnnz], &tmp_norm, block_size);
               min_weight = hypre_min(min_weight, tmp_norm);

               if (tmp_norm != 1.0)
               {
                  max_weight = hypre_max(max_weight, tmp_norm);
               }

               min_rowsum += tmp_norm;


            }
            for (j = P_offd_i[0]; j < P_offd_i[1]; j++)
            {
               hypre_CSRBlockMatrixBlockNorm(6, &P_offd_data[j * bnnz], &tmp_norm, block_size);
               min_weight = hypre_min(min_weight, tmp_norm);

               if (tmp_norm != 1.0)
               {
                  max_weight = hypre_max(max_weight, tmp_norm);
               }

               min_rowsum += tmp_norm;
            }

            max_rowsum = min_rowsum;

            min_entries = (P_diag_i[1] - P_diag_i[0]) + (P_offd_i[1] - P_offd_i[0]);
            max_entries = 0;

            for (j = 0; j < hypre_CSRBlockMatrixNumRows(P_block_diag); j++)
            {
               entries = (P_diag_i[j + 1] - P_diag_i[j]) + (P_offd_i[j + 1] - P_offd_i[j]);
               min_entries = hypre_min(entries, min_entries);
               max_entries = hypre_max(entries, max_entries);

               rowsum = 0.0;
               for (i = P_diag_i[j]; i < P_diag_i[j + 1]; i++)
               {
                  hypre_CSRBlockMatrixBlockNorm(6, &P_diag_data[i * bnnz], &tmp_norm, block_size);
                  min_weight = hypre_min(min_weight, tmp_norm);

                  if (tmp_norm != 1.0)
                  {
                     max_weight = hypre_max(max_weight, tmp_norm);
                  }

                  rowsum += tmp_norm;
               }

               for (i = P_offd_i[j]; i < P_offd_i[j + 1]; i++)
               {
                  hypre_CSRBlockMatrixBlockNorm(6, &P_offd_data[i * bnnz], &tmp_norm, block_size);
                  min_weight = hypre_min(min_weight, tmp_norm);

                  if (tmp_norm != 1.0)
                  {
                     max_weight = hypre_max(max_weight, P_offd_data[i]);
                  }

                  rowsum += tmp_norm;
               }

               min_rowsum = hypre_min(rowsum, min_rowsum);
               max_rowsum = hypre_max(rowsum, max_rowsum);
            }


         }
         avg_entries = ((HYPRE_Real) (global_nonzeros - coarse_size)) / ((HYPRE_Real) (
                                                                            fine_size - coarse_size));
      }
      else
      {
         P_diag = hypre_ParCSRMatrixDiag(P_array[level]);
         if ( hypre_GetActualMemLocation(hypre_CSRMatrixMemoryLocation(P_diag)) != hypre_MEMORY_HOST )
         {
            P_diag_clone = hypre_CSRMatrixClone_v2(P_diag, 1, HYPRE_MEMORY_HOST);
         }
         else
         {
            P_diag_clone = P_diag;
         }
         P_diag_data = hypre_CSRMatrixData(P_diag_clone);
         P_diag_i = hypre_CSRMatrixI(P_diag_clone);

         P_offd = hypre_ParCSRMatrixOffd(P_array[level]);
         if ( hypre_GetActualMemLocation(hypre_CSRMatrixMemoryLocation(P_offd)) != hypre_MEMORY_HOST )
         {
            P_offd_clone = hypre_CSRMatrixClone_v2(P_offd, 1, HYPRE_MEMORY_HOST);
         }
         else
         {
            P_offd_clone = P_offd;
         }
         P_offd_data = hypre_CSRMatrixData(P_offd_clone);
         P_offd_i = hypre_CSRMatrixI(P_offd_clone);

         row_starts = hypre_ParCSRMatrixRowStarts(P_array[level]);

         fine_size = hypre_ParCSRMatrixGlobalNumRows(P_array[level]);
         coarse_size = hypre_ParCSRMatrixGlobalNumCols(P_array[level]);
         hypre_ParCSRMatrixSetDNumNonzeros(P_array[level]);
         global_nonzeros = hypre_ParCSRMatrixDNumNonzeros(P_array[level]);
         num_mem[level] += (HYPRE_Real) global_nonzeros;

         min_weight = 1.0;
         max_weight = 0.0;
         max_rowsum = 0.0;
         min_rowsum = 0.0;
         min_entries = 0;
         max_entries = 0;

         if (hypre_CSRMatrixNumRows(P_diag))
         {
            if (P_diag_data) { min_weight = P_diag_data[0]; }
            for (j = P_diag_i[0]; j < P_diag_i[1]; j++)
            {
               min_weight = hypre_min(min_weight, P_diag_data[j]);
               if (P_diag_data[j] != 1.0)
               {
                  max_weight = hypre_max(max_weight, P_diag_data[j]);
               }
               min_rowsum += P_diag_data[j];
            }
            for (j = P_offd_i[0]; j < P_offd_i[1]; j++)
            {
               min_weight = hypre_min(min_weight, P_offd_data[j]);
               if (P_offd_data[j] != 1.0)
               {
                  max_weight = hypre_max(max_weight, P_offd_data[j]);
               }
               min_rowsum += P_offd_data[j];
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
                  min_weight = hypre_min(min_weight, P_diag_data[i]);
                  if (P_diag_data[i] != 1.0)
                  {
                     max_weight = hypre_max(max_weight, P_diag_data[i]);
                  }
                  rowsum += P_diag_data[i];
               }

               for (i = P_offd_i[j]; i < P_offd_i[j + 1]; i++)
               {
                  min_weight = hypre_min(min_weight, P_offd_data[i]);
                  if (P_offd_data[i] != 1.0)
                  {
                     max_weight = hypre_max(max_weight, P_offd_data[i]);
                  }
                  rowsum += P_offd_data[i];
               }

               min_rowsum = hypre_min(rowsum, min_rowsum);
               max_rowsum = hypre_max(rowsum, max_rowsum);
            }

         }
         avg_entries = ((HYPRE_Real) (global_nonzeros - coarse_size)) / ((HYPRE_Real) (
                                                                            fine_size - coarse_size));

         if (P_diag_clone != P_diag)
         {
            hypre_CSRMatrixDestroy(P_diag_clone);
         }

         if (P_offd_clone != P_offd)
         {
            hypre_CSRMatrixDestroy(P_offd_clone);
         }
      }

      numrows = row_starts[1] - row_starts[0];
      if (!numrows) /* if we don't have any rows, then don't have this count toward
                       min row sum or min num entries */
      {
         min_entries = 1000000;
         min_rowsum =  1.0e7;
         min_weight = 1.0e7;
      }

      send_buff[0] = - (HYPRE_Real) min_entries;
      send_buff[1] = (HYPRE_Real) max_entries;
      send_buff[2] = - min_rowsum;
      send_buff[3] = max_rowsum;
      send_buff[4] = - min_weight;
      send_buff[5] = max_weight;

      hypre_MPI_Reduce(send_buff, gather_buff, 6, HYPRE_MPI_REAL, hypre_MPI_MAX, 0, comm);

      if (my_id == 0)
      {
         global_min_e = - (HYPRE_Int)gather_buff[0];
         global_max_e = (HYPRE_Int)gather_buff[1];
         global_min_rsum = -gather_buff[2];
         global_max_rsum = gather_buff[3];
         global_min_wt = -gather_buff[4];
         global_max_wt = gather_buff[5];

         hypre_printf("%3d %*b x %-*b %3d  %3d",
                      level, ndigits[0], fine_size, ndigits[0], coarse_size,
                      global_min_e, global_max_e);
         hypre_printf("  %4.1f  %10.3e  %10.3e  %10.3e  %10.3e\n",
                      avg_entries, global_min_wt, global_max_wt,
                      global_min_rsum, global_max_rsum);
      }
   }

   total_variables = 0;
   operat_cmplxty = 0;
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
      hypre_printf("\n\n");
      hypre_printf("     Complexity:   grid = %f\n", grid_cmplxty);
      hypre_printf("               operator = %f\n", operat_cmplxty);
      hypre_printf("                 memory = %f\n", memory_cmplxty);
      hypre_printf("\n\n");
   }

   if (my_id == 0)
   {
      hypre_printf("\n\nBoomerAMG SOLVER PARAMETERS:\n\n");
      hypre_printf( "  Maximum number of cycles:         %d \n", max_iter);
      hypre_printf( "  Stopping Tolerance:               %e \n", tol);
      if (fcycle)
      {
         hypre_printf( "  Full Multigrid. Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      }
      else
      {
         hypre_printf( "  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      }

      if (additive == 0 || mult_additive == 0 || simple == 0)
      {
         HYPRE_Int add_lvl = (add_end == -1) ? num_levels - 1 : add_end;

         if (additive > -1)
         {
            hypre_printf( "  Additive V-cycle 1st level %d last level %d: \n", additive, add_lvl);
         }
         if (mult_additive > -1)
         {
            hypre_printf( "  Mult-Additive V-cycle 1st level %d last level %d: \n", mult_additive, add_lvl);
         }
         if (simple > -1)
         {
            hypre_printf( "  Simplified Mult-Additive V-cycle 1st level %d: last level %d \n", simple,
                          add_lvl);
         }
         hypre_printf( "  Relaxation Parameters:\n");
         if (add_lvl == num_levels - 1)
         {
            hypre_printf( "   Visiting Grid:                     down   up  coarse\n");
            hypre_printf( "            Number of sweeps:         %4d   %2d  %4d \n",
                          num_grid_sweeps[1],
                          num_grid_sweeps[1], (2 * num_grid_sweeps[1]));
            hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:    %2d   %2d   %2d \n", add_rlx, add_rlx,
                          add_rlx);
         }
         else
         {
            hypre_printf( "   Visiting Grid:                     down   up\n");
            hypre_printf( "            Number of sweeps:         %4d   %2d\n",
                          num_grid_sweeps[1], num_grid_sweeps[1]);
            hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:    %2d   %2d\n", add_rlx, add_rlx);
         }
         if (add_lvl < num_levels - 1)
         {
            hypre_printf( " \n");
            hypre_printf( "Multiplicative portion: \n");
            hypre_printf( "   Visiting Grid:                     down   up  coarse\n");
            hypre_printf( "            Number of sweeps:         %4d   %2d  %4d\n",
                          num_grid_sweeps[1], num_grid_sweeps[2], num_grid_sweeps[3]);
            hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d\n",
                          grid_relax_type[1], grid_relax_type[2], grid_relax_type[3]);
         }
         if (add_rlx == 0)
         {
            hypre_printf( "   Relaxation Weight:   %e \n", add_rlx_wt);
         }
         {
            hypre_printf( "   Point types, partial sweeps (1=C, -1=F):\n");
            hypre_printf( "                  Pre-CG relaxation (down):");
         }
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         {
            hypre_printf( "\n");
            hypre_printf( "                   Post-CG relaxation (up):");
         }
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         {
            hypre_printf( "\n");
            hypre_printf( "                             Coarsest grid:");
         }
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         {
            hypre_printf( "\n");
         }
      }
      else if (additive > 0 || mult_additive > 0 || simple > 0)
      {
         HYPRE_Int add_lvl = (add_end == -1) ? (num_levels - 1) : add_end;

         hypre_printf( "  Relaxation Parameters:\n");
         if (add_lvl < num_levels - 1)
         {
            hypre_printf( "   Visiting Grid:                     down   up  coarse\n");
            hypre_printf( "            Number of sweeps:         %4d   %2d  %4d\n",
                          num_grid_sweeps[1], num_grid_sweeps[2], num_grid_sweeps[3]);
            hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d\n",
                          grid_relax_type[1], grid_relax_type[2], grid_relax_type[3]);
         }
         else
         {
            hypre_printf( "   Visiting Grid:                     down   up  \n");
            hypre_printf( "            Number of sweeps:         %4d   %2d  \n",
                          num_grid_sweeps[1], num_grid_sweeps[2]);
            hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  \n",
                          grid_relax_type[1], grid_relax_type[2]);
         }
         hypre_printf( "   Point types, partial sweeps (1=C, -1=F):\n");
         if (grid_relax_points && grid_relax_type[1] != 8)
         {
            hypre_printf( "                  Pre-CG relaxation (down):");
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_printf("  %2d", grid_relax_points[1][j]);
            }
            hypre_printf( "\n");
            hypre_printf( "                   Post-CG relaxation (up):");
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               hypre_printf("  %2d", grid_relax_points[2][j]);
            }
            hypre_printf( "\n");
         }
         else if (relax_order == 1 && grid_relax_type[1] != 8)
         {
            hypre_printf( "                  Pre-CG relaxation (down):");
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_printf("  %2d  %2d", one, minus_one);
            }
            hypre_printf( "\n");
            hypre_printf( "                   Post-CG relaxation (up):");
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               hypre_printf("  %2d  %2d", minus_one, one);
            }
            hypre_printf( "\n");
         }
         else
         {
            hypre_printf( "                  Pre-CG relaxation (down):");
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_printf("  %2d", zero);
            }
            hypre_printf( "\n");
            hypre_printf( "                   Post-CG relaxation (up):");
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               hypre_printf("  %2d", zero);
            }
            hypre_printf( "\n");
         }
         {
            hypre_printf( "\n\n");
         }
         if (additive > -1)
         {
            hypre_printf( "  Additive V-cycle 1st level %d last level %d:  \n", additive, add_lvl);
         }
         if (mult_additive > -1)
         {
            hypre_printf( "  Mult-Additive V-cycle 1st level %d last level %d: \n", mult_additive, add_lvl);
         }
         if (simple > -1)
         {
            hypre_printf( "  Simplified Mult-Additive V-cycle 1st level %d: last level %d  \n", simple,
                          add_lvl);
         }
         {
            hypre_printf( "  Relaxation Parameters:\n");
         }
         if (add_lvl == num_levels - 1)
         {
            hypre_printf( "   Visiting Grid:                     down   up  coarse\n");
            hypre_printf( "            Number of sweeps:         %4d   %2d  %4d \n",
                          num_grid_sweeps[1],
                          num_grid_sweeps[1], (2 * num_grid_sweeps[1]));
            hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:    %2d   %2d   %2d \n", add_rlx, add_rlx,
                          add_rlx);
         }
         else
         {
            hypre_printf( "   Visiting Grid:                     down   up\n");
            hypre_printf( "            Number of sweeps:         %4d   %2d\n",
                          num_grid_sweeps[1], num_grid_sweeps[1]);
            hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:    %2d   %2d\n", add_rlx, add_rlx);
         }
         if (add_rlx == 0)
         {
            hypre_printf( "   Relaxation Weight:   %e \n", add_rlx_wt);
         }
         {
            hypre_printf( "   Point types, partial sweeps (1=C, -1=F):\n");
            hypre_printf( "                  Pre-CG relaxation (down):");
         }
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         {
            hypre_printf( "\n");
            hypre_printf( "                   Post-CG relaxation (up):");
         }
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         {
            hypre_printf( "\n");
            hypre_printf( "                             Coarsest grid:");
         }
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         {
            hypre_printf( "\n");
         }
      }
      else
      {
         hypre_printf( "  Relaxation Parameters:\n");
         hypre_printf( "   Visiting Grid:                     down   up  coarse\n");
         hypre_printf( "            Number of sweeps:         %4d   %2d  %4d \n",
                       num_grid_sweeps[1],
                       num_grid_sweeps[2], num_grid_sweeps[3]);
         hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d \n",
                       grid_relax_type[1],
                       grid_relax_type[2], grid_relax_type[3]);
         hypre_printf( "   Point types, partial sweeps (1=C, -1=F):\n");
         if (grid_relax_points && grid_relax_type[1] != 8)
         {
            hypre_printf( "                  Pre-CG relaxation (down):");
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_printf("  %2d", grid_relax_points[1][j]);
            }
            hypre_printf( "\n");
            hypre_printf( "                   Post-CG relaxation (up):");
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               hypre_printf("  %2d", grid_relax_points[2][j]);
            }
            hypre_printf( "\n");
            hypre_printf( "                             Coarsest grid:");
            for (j = 0; j < num_grid_sweeps[3]; j++)
            {
               hypre_printf("  %2d", grid_relax_points[3][j]);
            }
            {
               hypre_printf( "\n");
            }
         }
         else if (relax_order == 1 && grid_relax_type[1] != 8)
         {
            hypre_printf( "                  Pre-CG relaxation (down):");
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_printf("  %2d  %2d", one, minus_one);
            }
            hypre_printf( "\n");
            hypre_printf( "                   Post-CG relaxation (up):");
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               hypre_printf("  %2d  %2d", minus_one, one);
            }
            hypre_printf( "\n");
            hypre_printf( "                             Coarsest grid:");
            for (j = 0; j < num_grid_sweeps[3]; j++)
            {
               hypre_printf("  %2d", zero);
            }
            {
               hypre_printf( "\n");
            }
         }
         else
         {
            hypre_printf( "                  Pre-CG relaxation (down):");
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_printf("  %2d", zero);
            }
            hypre_printf( "\n");
            hypre_printf( "                   Post-CG relaxation (up):");
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               hypre_printf("  %2d", zero);
            }
            hypre_printf( "\n");
            hypre_printf( "                             Coarsest grid:");
            for (j = 0; j < num_grid_sweeps[3]; j++)
            {
               hypre_printf("  %2d", zero);
            }
            {
               hypre_printf( "\n");
            }
         }
      }
#if defined(HYPRE_USING_MAGMA)
      if (grid_relax_type[3] ==  98 || grid_relax_type[3] ==  99 ||
          grid_relax_type[3] == 198 || grid_relax_type[3] == 199)
      {
         hypre_printf( "   Using MAGMA's LU factorization on coarse level\n");
      }
#endif
      {
         hypre_printf( "\n");
      }

      if (smooth_type == 6)
      {
         for (j = 0; j < smooth_num_levels; j++)
         {
            hypre_printf( " Schwarz Relaxation Weight %f level %d\n",
                          hypre_ParAMGDataSchwarzRlxWeight(amg_data), j);
         }
      }
      if (smooth_type == 7)
      {
         for (j = 0; j < smooth_num_levels; j++)
         {
            hypre_printf( " Pilut smoother level %d\n", j);
         }
      }
      if (smooth_type == 8)
      {
         for (j = 0; j < smooth_num_levels; j++)
         {
            hypre_printf( " ParaSails smoother level %d\n", j);
         }
      }
      if (smooth_type == 9)
      {
         for (j = 0; j < smooth_num_levels; j++)
         {
            hypre_printf( " Euclid smoother level %d\n", j);
         }
      }
      for (j = 0; j < num_levels; j++)
      {
         if (relax_weight[j] != 1)
         {
            hypre_printf( " Relaxation Weight %f level %d\n", relax_weight[j], j);
         }
      }
      for (j = 0; j < num_levels; j++)
      {
         if (omega[j] != 1)
         {
            hypre_printf( " Outer relaxation weight %f level %d\n", omega[j], j);
         }
      }
   }

   hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);
   hypre_TFree(num_mem, HYPRE_MEMORY_HOST);
   hypre_TFree(num_variables, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buff, HYPRE_MEMORY_HOST);
   hypre_TFree(gather_buff, HYPRE_MEMORY_HOST);

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*---------------------------------------------------------------
 * hypre_BoomerAMGWriteSolverParams
 *---------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGWriteSolverParams(void* data)
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
   HYPRE_Real        *relax_weight;
   HYPRE_Real        *omega;
   HYPRE_Real         tol;
   HYPRE_Int          smooth_type;
   HYPRE_Int          smooth_num_levels;

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
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
   omega = hypre_ParAMGDataOmega(amg_data);
   smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   tol = hypre_ParAMGDataTol(amg_data);

   amg_print_level = hypre_ParAMGDataPrintLevel(amg_data);

   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/

   if (amg_print_level == 1 || amg_print_level == 3)
   {
      hypre_printf("\n\nBoomerAMG SOLVER PARAMETERS:\n\n");
      hypre_printf( "  Maximum number of cycles:         %d \n", max_iter);
      hypre_printf( "  Stopping Tolerance:               %e \n", tol);
      if (fcycle)
      {
         hypre_printf( "  Full Multigrid. Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      }
      else
      {
         hypre_printf( "  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      }
      hypre_printf( "  Relaxation Parameters:\n");
      hypre_printf( "   Visiting Grid:                     down   up  coarse\n");
      hypre_printf( "            Number of sweeps:         %4d   %2d  %4d \n",
                    num_grid_sweeps[1],
                    num_grid_sweeps[2], num_grid_sweeps[3]);
      hypre_printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d \n",
                    grid_relax_type[1],
                    grid_relax_type[2], grid_relax_type[3]);
      hypre_printf( "   Point types, partial sweeps (1=C, -1=F):\n");
      if (grid_relax_points)
      {
         hypre_printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf("  %2d", grid_relax_points[1][j]);
         }
         hypre_printf( "\n");
         hypre_printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf("  %2d", grid_relax_points[2][j]);
         }
         hypre_printf( "\n");
         hypre_printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf("  %2d", grid_relax_points[3][j]);
         }
         hypre_printf( "\n\n");
      }
      else if (relax_order == 1)
      {
         hypre_printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf("  %2d  %2d", one, minus_one);
         }
         hypre_printf( "\n");
         hypre_printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf("  %2d  %2d", minus_one, one);
         }
         hypre_printf( "\n");
         hypre_printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         hypre_printf( "\n\n");
      }
      else
      {
         hypre_printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         hypre_printf( "\n");
         hypre_printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         hypre_printf( "\n");
         hypre_printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
         {
            hypre_printf("  %2d", zero);
         }
         hypre_printf( "\n\n");
      }

      if (smooth_type == 6)
      {
         for (j = 0; j < smooth_num_levels; j++)
         {
            hypre_printf( " Schwarz Relaxation Weight %f level %d\n",
                          hypre_ParAMGDataSchwarzRlxWeight(amg_data), j);
         }
      }
      for (j = 0; j < num_levels; j++)
      {
         if (relax_weight[j] != 1)
         {
            hypre_printf( " Relaxation Weight %f level %d\n", relax_weight[j], j);
         }
      }
      for (j = 0; j < num_levels; j++)
      {
         if (omega[j] != 1)
         {
            hypre_printf( " Outer relaxation weight %f level %d\n", omega[j], j);
         }
      }

      hypre_printf( " Output flag (print_level): %d \n", amg_print_level);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_BoomerAMGGetProlongationName
 *--------------------------------------------------------------------*/

const char*
hypre_BoomerAMGGetProlongationName(hypre_ParAMGData *amg_data)
{
   switch (hypre_ParAMGDataInterpType(amg_data))
   {
      case 0:
         return "modified classical";

      case 1:
         return "LS";

      case 2:
         return "modified classical for hyperbolic PDEs";

      case 3:
         return "direct with separation of weights";

      case 4:
         return "multipass";

      case 5:
         return "multipass with separation of weights";

      case 6:
         return "extended+i";

      case 7:
         return "extended+i (if no common C-point)";

      case 8:
         return "standard";

      case 9:
         return "standard with separation of weights";

      case 10:
         return "block classical for nodal systems";

      case 11:
         return "block classical with diagonal blocks for nodal systems";

      case 12:
         return "F-F";

      case 13:
         return "F-F1";

      case 14:
         return "extended";

      case 15:
         return "direct with separation of weights";

      case 16:
         return "MM-extended";

      case 17:
         return "MM-extended+i";

      case 18:
         return "MM-extended+e";

      case 24:
         return "block direct for nodal systems";

      case 100:
         return "one-point";

      default:
         return "Unknown";
   }
}

/*--------------------------------------------------------------------
 * hypre_BoomerAMGGetAggProlongationName
 *--------------------------------------------------------------------*/

const char*
hypre_BoomerAMGGetAggProlongationName(hypre_ParAMGData *amg_data)
{
   if (hypre_ParAMGDataAggNumLevels(amg_data))
   {
      switch (hypre_ParAMGDataAggInterpType(amg_data))
      {
         case 1:
            return "2-stage extended+i";

         case 2:
            return "2-stage standard";

         case 3:
            return "2-stage extended";

         case 4:
            return "multipass";

         default:
            return "Unknown";
      }
   }
   else
   {
      return "";
   }
}

/*--------------------------------------------------------------------
 * hypre_BoomerAMGGetCoarseningName
 *--------------------------------------------------------------------*/

const char*
hypre_BoomerAMGGetCoarseningName(hypre_ParAMGData *amg_data)
{
   switch (hypre_ParAMGDataCoarsenType(amg_data))
   {
      case 0:
         return "Cleary-Luby-Jones-Plassman";

      case 1:
         return "Ruge";

      case 2:
         return "Ruge-2B";

      case 3:
         return "Ruge-3";

      case 4:
         return "Ruge-3c";

      case 5:
         return "Ruge relax special points";

      case 6:
         return "Falgout-CLJP";

      case 7:
         return "CLJP, fixed random";

      case 8:
         return "PMIS";

      case 9:
         return "PMIS, fixed random";

      case 10:
         return "HMIS";

      case 11:
         return "Ruge 1st pass only";

      case 21:
         return "CGC";

      case 22:
         return "CGC-E";

      default:
         return "Unknown";
   }
}

/*--------------------------------------------------------------------
 * hypre_BoomerAMGGetCoarseningName
 *--------------------------------------------------------------------*/

const char*
hypre_BoomerAMGGetCycleName(hypre_ParAMGData *amg_data)
{
   static char name[10];

   switch (hypre_ParAMGDataCycleType(amg_data))
   {
      case 1:
         hypre_sprintf(name, "V(%d,%d)",
                       hypre_ParAMGDataNumGridSweeps(amg_data)[0],
                       hypre_ParAMGDataNumGridSweeps(amg_data)[1]);
         break;

      case 2:
         hypre_sprintf(name, "W(%d,%d)",
                       hypre_ParAMGDataNumGridSweeps(amg_data)[0],
                       hypre_ParAMGDataNumGridSweeps(amg_data)[1]);
         break;

      default:
         return "Unknown";
   }

   return name;
}

/*--------------------------------------------------------------------
 * hypre_BoomerAMGPrintGeneralInfo
 *
 * Prints to stdout info about BoomerAMG parameters.
 * The input parameter "shift" refers to the number of whitespaces
 * added to the beginning of each line.
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGPrintGeneralInfo(hypre_ParAMGData *amg_data,
                                HYPRE_Int         shift)
{
   HYPRE_PRINT_SHIFTED_PARAM(shift,
                             "Solver Type = BoomerAMG\n");

   HYPRE_PRINT_SHIFTED_PARAM(shift,
                             "Strength Threshold = %f\n",
                             hypre_ParAMGDataStrongThreshold(amg_data));

   HYPRE_PRINT_SHIFTED_PARAM(shift,
                             "Interpolation Truncation Factor = %f\n",
                             hypre_ParAMGDataTruncFactor(amg_data));

   HYPRE_PRINT_SHIFTED_PARAM(shift,
                             "Maximum Row Sum Threshold for Dependency Weakening = %f\n",
                             hypre_ParAMGDataMaxRowSum(amg_data));

   HYPRE_PRINT_SHIFTED_PARAM(shift,
                             "Number of functions = %d\n",
                             hypre_ParAMGDataNumFunctions(amg_data));

   HYPRE_PRINT_SHIFTED_PARAM(shift,
                             "Coarsening type = %s\n",
                             hypre_BoomerAMGGetCoarseningName(amg_data));

   HYPRE_PRINT_SHIFTED_PARAM(shift,
                             "Prolongation type = %s\n",
                             hypre_BoomerAMGGetProlongationName(amg_data));

   HYPRE_PRINT_SHIFTED_PARAM(shift,
                             "Cycle type = %s\n",
                             hypre_BoomerAMGGetCycleName(amg_data));
   hypre_printf("\n");

   return hypre_error_flag;
}
