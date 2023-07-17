/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_mgr.h" /* TODO (VPM): include this into _hypre_parcsr_ls.h */
#include "par_ilu.h" /* TODO (VPM): include this into _hypre_parcsr_ls.h */

/*--------------------------------------------------------------------
 * hypre_MGRGetGlobalRelaxName
 *--------------------------------------------------------------------*/

char*
hypre_MGRGetGlobalRelaxName(hypre_ParMGRData  *mgr_data,
                            HYPRE_Int          level )
{
   HYPRE_Int    smoother_type = hypre_ParMGRDataLevelSmoothTypeI(mgr_data, level);

   switch (smoother_type)
   {
      case -1:
         return "--";

      case 0:
         return "Blk-Jacobi";

      case 1:
         return "Blk-GS";

      case 2:
         return "GS";

      case 3:
         return "Forward hGS";

      case 4:
         return "Backward hGS";

      case 5:
         return "Chaotic hGS";

      case 6:
         return "hSGS";

      case 7:
         return "Jacobi";

      case 8:
         return "Euclid ILU";

      case 13:
         return "Forward L1-hGS";

      case 14:
         return "Backward L1-hGS";

      case 16:
         /* TODO (VPM): Move this to hypre_ILUGetName */
         HYPRE_Solver      smoother     = hypre_ParMGRDataLevelSmootherI(mgr_data, level);
         hypre_ParILUData *ilu_smoother = (hypre_ParILUData*) smoother;
         HYPRE_Int         ilu_type = hypre_ParILUDataIluType(ilu_smoother);
         HYPRE_Int         ilu_fill = hypre_ParILUDataLfil(ilu_smoother);

         switch (ilu_type)
         {
            case 0:
               return (ilu_fill == 0) ? "BJ-ILU0" : "BJ-ILUK";

            case 1:
               return "BJ-ILUT";

            case 10:
               return (ilu_fill == 0) ? "GMRES-ILU0" : "GMRES-ILUK";

            case 11:
               return "GMRES-ILUT";

            case 20:
               return (ilu_fill == 0) ? "NSH-ILU0" : "NSH-ILUK";

            case 21:
               return "NSH-ILUT";

            case 30:
               return (ilu_fill == 0) ? "RAS-ILU0" : "RAS-ILUK";

            case 31:
               return "RAS-ILUT";

            case 40:
               return (ilu_fill == 0) ? "ddPQ-GMRES-ILU0" : "ddPQ-GMRES-ILUK";

            case 41:
               return "ddPQ-GMRES-ILUT";

            case 50:
               return "RAP-modILU0";

            default:
               return "Unknown";
         }

      default:
         return "Unknown";
   }
}

/*--------------------------------------------------------------------
 * hypre_MGRGetFRelaxName
 *--------------------------------------------------------------------*/

char*
hypre_MGRGetFRelaxName(hypre_ParMGRData  *mgr_data,
                       HYPRE_Int          level )
{
   HYPRE_Int  F_relax_type = hypre_ParMGRDataFRelaxTypeI(mgr_data, level);
   HYPRE_Int  relax_type = hypre_ParMGRDataRelaxType(mgr_data);

   if (F_relax_type == 0)
   {
      switch (relax_type)
      {
         case 0: case 7:
            if (hypre_ParMGRDataInterpTypeI(mgr_data, level) == 12)
            {
               return "Blk-Jacobi";
            }
            else
            {
               return "Jacobi";
            }

         case 1:
            return "GS 1";

         case 2:
            return "GS 2";

         case 3:
            return "Forward hGS";

         case 4:
            return "Backward hGS";

         case 5:
            return "Chaotic hGS";

         case 6:
            return "hSGS";

         case 8:
            return "L1-hSGS";

         case 13:
            return "Forward L1-hGS";

         case 14:
            return "Backward L1-hGS";

         case 16:
            return "Chebyshev";

         default:
            return "Unknown";
      }
   }
   else if (F_relax_type == 1)
   {
      return "Default AMG";
   }
   else if (F_relax_type == 2)
   {
      return "User AMG";
   }
   else if (F_relax_type == 9)
   {
      return "GaussElim";
   }
   else if (F_relax_type == 19)
   {
      return "LU";
   }
   else if (F_relax_type == 199)
   {
      return "Dense Inv";
   }
   else
   {
      return "Unknown";
   }
}

/*--------------------------------------------------------------------
 * hypre_MGRGetProlongationName
 *--------------------------------------------------------------------*/

char*
hypre_MGRGetProlongationName(hypre_ParMGRData  *mgr_data,
                             HYPRE_Int          level )
{
   switch (hypre_ParMGRDataInterpTypeI(mgr_data, level))
   {
      case 0:
         return "Injection";

      case 1:
         return "L1-Jac Inv";

      case 2:
         return "Diag Inv";

      case 4:
         return "Approx Inv";

      case 5:
         return "MM-ext";

      case 6:
         return "MM-ext+i";

      case 7:
         return "MM-ext+e";

      case 12:
         return "Blk-Diag Inv";

      default:
         return "Classical";
   }
}

/*--------------------------------------------------------------------
 * hypre_MGRGetRestrictionName
 *--------------------------------------------------------------------*/

char*
hypre_MGRGetRestrictionName(hypre_ParMGRData  *mgr_data,
                            HYPRE_Int          level )
{
   switch (hypre_ParMGRDataRestrictTypeI(mgr_data, level))
   {
      case 0:
         return "Injection";

      case 1:
         return "L1-Jac Inv";

      case 2:
         return "Diag Inv";

      case 3:
         return "Approx Inv";

      case 12:
         return "Blk-Diag Inv";

      case 13:
         return "CPR-like";

      default:
         return "Classical";
   }
}

/*--------------------------------------------------------------------
 * hypre_MGRGetCoarseGridName
 *--------------------------------------------------------------------*/

char*
hypre_MGRGetCoarseGridName(hypre_ParMGRData  *mgr_data,
                           HYPRE_Int          level )
{
   switch (hypre_ParMGRDataCoarseGridMethodI(mgr_data, level))
   {
      case 0:
         return "RAP";

      case 1:
         return "NG-BlkDiag";

      case 2:
         return "NG-CPR-Diag";

      case 3:
         return "NG-CPR-BlkDiag";

      case 4:
         return "NG-ApproxInv";

      default:
         return "Unknown";
   }
}

/*--------------------------------------------------------------------
 * hypre_BoomerAMGGetProlongationName
 *
 * TODO (VPM): move this to par_stats.c
 *--------------------------------------------------------------------*/

char*
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
 *
 * TODO (VPM): move this to par_stats.c
 *--------------------------------------------------------------------*/

char*
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
 *
 * TODO (VPM): move this to par_stats.c
 *--------------------------------------------------------------------*/

char*
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
 *
 * TODO (VPM): move this to par_stats.c
 *--------------------------------------------------------------------*/

char*
hypre_BoomerAMGGetCycleName(hypre_ParAMGData *amg_data)
{
   char *name = hypre_CTAlloc(char, 10, HYPRE_MEMORY_HOST);

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
 * TODO (VPM): move this to par_stats.c
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGPrintGeneralInfo(hypre_ParAMGData *amg_data,
                                HYPRE_Int         shift)
{
   HYPRE_PRINT_INDENT(shift);
   hypre_printf("Solver Type = BoomerAMG\n");

   HYPRE_PRINT_INDENT(shift);
   hypre_printf("Strength Threshold = %f\n",
                hypre_ParAMGDataStrongThreshold(amg_data));

   HYPRE_PRINT_INDENT(shift);
   hypre_printf("Interpolation Truncation Factor = %f\n",
                hypre_ParAMGDataTruncFactor(amg_data));

   HYPRE_PRINT_INDENT(shift);
   hypre_printf("Maximum Row Sum Threshold for Dependency Weakening = %f\n",
                hypre_ParAMGDataMaxRowSum(amg_data));

   HYPRE_PRINT_INDENT(shift);
   hypre_printf("Number of functions = %d\n",
                hypre_ParAMGDataNumFunctions(amg_data));

   HYPRE_PRINT_INDENT(shift);
   hypre_printf("Coarsening type = %s\n",
                hypre_BoomerAMGGetCoarseningName(amg_data));

   HYPRE_PRINT_INDENT(shift);
   hypre_printf("Prolongation type = %s\n",
                hypre_BoomerAMGGetProlongationName(amg_data));

   HYPRE_PRINT_INDENT(shift);
   hypre_printf("Cycle type = %s\n",
                hypre_BoomerAMGGetCycleName(amg_data));
   hypre_printf("\n");

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_MGRSetupStats
 *
 * TODO (VPM):
 *      1) Add total number of GPUs or number of ranks using 1 GPU?
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRSetupStats(void *mgr_vdata)
{
   hypre_ParMGRData          *mgr_data   = (hypre_ParMGRData*) mgr_vdata;

   /* MGR data */
   hypre_ParCSRMatrix        *A_finest       = hypre_ParMGRDataA(mgr_data, 0);
   hypre_ParCSRMatrix        *A_coarsest     = hypre_ParMGRDataRAP(mgr_data);
   HYPRE_Int                  num_levels_mgr = hypre_ParMGRDataNumCoarseLevels(mgr_data);
   HYPRE_Solver               coarse_solver  = hypre_ParMGRDataCoarseGridSolver(mgr_data);
   HYPRE_Solver             **A_FF_solver    = hypre_ParMGRDataAFFsolver(mgr_data);
   HYPRE_Int                 *Frelax_type    = hypre_ParMGRDataFRelaxType(mgr_data);

   /* Matrix variables */
   MPI_Comm                   comm           = hypre_ParCSRMatrixComm(A_finest);

   /* Local variables */
   hypre_ParAMGData          *amg_solver = NULL;
   hypre_ParAMGData          *coarse_amg_solver = NULL;
   hypre_ParCSRMatrix       **A_array;
   hypre_ParCSRMatrix       **P_array;
   hypre_ParCSRMatrix       **RT_array;
   hypre_MatrixStatsArray    *stats_array;

   HYPRE_Real                *gridcomp;
   HYPRE_Real                *opcomp;
   HYPRE_Real                *memcomp;

   HYPRE_Int                  coarsest_mgr_level;
   HYPRE_Int                  num_levels_total;
   HYPRE_Int                  num_levels[2];
   HYPRE_Int                  max_levels;
   HYPRE_Int                 *num_sublevels_amg;
   HYPRE_Int                  num_procs, num_threads;
   HYPRE_Int                  i, k, my_id;
   HYPRE_Int                  divisors[1];

   /*-------------------------------------------------
    *  Initialize and allocate data
    *-------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = hypre_NumThreads();

   coarsest_mgr_level = num_levels_mgr;
   num_sublevels_amg  = hypre_CTAlloc(HYPRE_Int, num_levels_mgr + 1, HYPRE_MEMORY_HOST);

   /* Check MGR's coarse level solver */
   if ((void*) hypre_ParMGRDataCoarseGridSolverSetup(mgr_data) == (void*) HYPRE_BoomerAMGSetup ||
       (void*) hypre_ParMGRDataCoarseGridSolverSetup(mgr_data) == (void*) hypre_BoomerAMGSetup)
   {
      coarse_amg_solver = (hypre_ParAMGData *) coarse_solver;
      num_sublevels_amg[coarsest_mgr_level] = hypre_ParAMGDataNumLevels(coarse_amg_solver);
   }
#ifdef HYPRE_USING_DSUPERLU
   else if ((void*) hypre_ParMGRDataCoarseGridSolverSetup(mgr_data) ==
            (void*) hypre_MGRDirectSolverSetup)
   {
      /* TODO (VPM): Set SuperLU solver specifics */
      num_sublevels_amg[coarsest_mgr_level] = 0;
   }
#endif
   num_levels_total = num_levels_mgr + num_sublevels_amg[coarsest_mgr_level];

   /* Compute number of AMG sublevels at each MGR level */
   max_levels = num_levels_total;
   for (i = 0; i < num_levels_mgr; i++)
   {
      if (Frelax_type[i] == 2)
      {
         amg_solver = (hypre_ParAMGData *) A_FF_solver[i];
         num_sublevels_amg[i] = hypre_ParAMGDataNumLevels(amg_solver);

         max_levels = hypre_max(max_levels, num_sublevels_amg[i]);
      }
   }

   /* Create array of statistics */
   stats_array = hypre_MatrixStatsArrayCreate(max_levels + 1);

   /*-------------------------------------------------
    *  Print general info
    *-------------------------------------------------*/

   if (my_id == 0)
   {
      hypre_printf("\n\n");
      hypre_printf(" Num MPI tasks = %d\n",  num_procs);
      hypre_printf(" Num OpenMP threads = %d\n\n", num_threads);
      hypre_printf("\n");
      hypre_printf("MGR SETUP PARAMETERS:\n\n");
      hypre_printf("    MGR num levels = %d\n", num_levels_mgr + 1);
      hypre_printf("    AMG num levels = %d\n", num_sublevels_amg[coarsest_mgr_level] - 1);
      hypre_printf("  Total num levels = %d\n\n", num_levels_total);

      divisors[0] = 84;
      //hypre_printf("\nMGR level options:\n\n");
      hypre_printf("%18s %14s %16s\n", "Global", "Fine", "Coarse");
      hypre_printf("%3s %14s %14s %16s %16s %16s\n", "lev",
                   "relaxation", "relaxation", "grid method",
                   "Interpolation", "Restriction");
      HYPRE_PRINT_TOP_DIVISOR(1, divisors);
      for (i = 0; i < num_levels_mgr; i++)
      {
         hypre_printf("%3d %14s %14s %16s %16s %16s\n",
                      i,
                      hypre_MGRGetGlobalRelaxName(mgr_data, i),
                      hypre_MGRGetFRelaxName(mgr_data, i),
                      hypre_MGRGetCoarseGridName(mgr_data, i),
                      hypre_MGRGetProlongationName(mgr_data, i),
                      hypre_MGRGetRestrictionName(mgr_data, i));
      }
      hypre_printf("%3d %14s %14s %16s %16s %16s\n",
                   i,
                   hypre_MGRGetGlobalRelaxName(mgr_data, i),
                   hypre_MGRGetFRelaxName(mgr_data, i),
                   "--", "--", "--");
      hypre_printf("\n\n");
   }

   /*-------------------------------------------------
    *  Print MGR hierarchy info
    *-------------------------------------------------*/

   /* Set pointer to level matrices */
   A_array = hypre_TAlloc(hypre_ParCSRMatrix *, max_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_levels_mgr; i++)
   {
      A_array[i] = hypre_ParMGRDataA(mgr_data, i);
   }
   A_array[num_levels_mgr] = A_coarsest;

   for (i = 1; i < num_sublevels_amg[coarsest_mgr_level]; i++)
   {
      A_array[num_levels_mgr + i] = hypre_ParAMGDataAArray(coarse_amg_solver)[i];
   }

   /* Compute statistics data structure */
   hypre_ParCSRMatrixStatsArrayCompute(num_levels_total, A_array, stats_array);

   if (my_id == 0)
   {
      char *msg[] = { "Full Operator Matrix Hierarchy Information:\n\n",
                      "MGR's coarsest level",
                      "\t( MGR )",
                      "\t( AMG )" };

      num_levels[0] = num_levels_mgr;
      num_levels[1] = num_sublevels_amg[coarsest_mgr_level];

      hypre_MatrixStatsArrayPrint(2, num_levels, 1, 0, msg, stats_array);
   }

   /*-------------------------------------------------
    *  Print MGR level input data
    *-------------------------------------------------*/

   /* Set pointer to level matrices */
   P_array  = hypre_TAlloc(hypre_ParCSRMatrix *, max_levels, HYPRE_MEMORY_HOST);
   RT_array = hypre_TAlloc(hypre_ParCSRMatrix *, max_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_levels_mgr; i++)
   {
      P_array[i] = hypre_ParMGRDataP(mgr_data, i);
   }

   for (i = 0; i < num_sublevels_amg[coarsest_mgr_level]; i++)
   {
      P_array[num_levels_mgr + i] = hypre_ParAMGDataPArray(coarse_amg_solver)[i];
   }

   /* Compute statistics data structure */
   hypre_ParCSRMatrixStatsArrayCompute(num_levels_total - 1, P_array, stats_array);

   if (my_id == 0)
   {
      char *msg[] = { "Full Prolongation Matrix Hierarchy Information:\n\n",
                      "MGR's coarsest level",
                      "\t( MGR )",
                      "\t( AMG )" };

      num_levels[0] = num_levels_mgr;
      num_levels[1] = num_sublevels_amg[coarsest_mgr_level] - 1;
      hypre_MatrixStatsArrayPrint(2, num_levels, 1, 0, msg, stats_array);
   }

   /*-------------------------------------------------
    *  Print MGR F-relaxation info
    *-------------------------------------------------*/

   for (i = 0; i < num_levels_mgr; i++)
   {
      if (num_sublevels_amg[i] > 0)
      {
         if (my_id == 0)
         {
            hypre_printf("At MGR level %d --> F-relaxation solver parameters\n\n", i);
         }
         amg_solver = (hypre_ParAMGData *) A_FF_solver[i];

         /* General AMG info */
         if (my_id == 0)
         {
            hypre_BoomerAMGPrintGeneralInfo(amg_solver, 3);
         }

         /* Gather A matrices */
         for (k = 0; k < num_sublevels_amg[i]; k++)
         {
            A_array[k] = hypre_ParAMGDataAArray(amg_solver)[k];
         }

         /* Compute statistics */
         hypre_ParCSRMatrixStatsArrayCompute(num_sublevels_amg[i], A_array, stats_array);

         /* Print A matrices info */
         if (my_id == 0)
         {
            char *msg[] = {"Operator Matrix Hierarchy Information:\n\n"};
            num_levels[0] = num_sublevels_amg[i];
            hypre_MatrixStatsArrayPrint(1, num_levels, 1, 3, msg, stats_array);
         }

         /* Gather P matrices */
         for (k = 0; k < num_sublevels_amg[i] - 1; k++)
         {
            P_array[k] = hypre_ParAMGDataPArray(amg_solver)[k];
         }

         /* Compute statistics */
         hypre_ParCSRMatrixStatsArrayCompute(num_sublevels_amg[i] - 1, P_array, stats_array);

         /* Print P matrices info */
         if (my_id == 0)
         {
            char *msg[] = {"Prolongation Matrix Hierarchy Information:\n\n"};
            num_levels[0] = num_sublevels_amg[i] - 1;
            hypre_MatrixStatsArrayPrint(1, num_levels, 1, 3, msg, stats_array);
         }
      }
   }

   /*-------------------------------------------------
    *  Print MGR coarsest level solver info
    *-------------------------------------------------*/

   if (my_id == 0)
   {
      if (coarse_amg_solver && num_sublevels_amg[coarsest_mgr_level] > 1)
      {
         hypre_printf("At MGR level %d (Coarsest) --> Solver parameters:\n\n", num_levels_mgr);
         hypre_BoomerAMGPrintGeneralInfo(coarse_amg_solver, 3);
      }
   }

   /*-------------------------------------------------
    *  Print MGR complexities
    *-------------------------------------------------*/

   /* Allocate memory for complexities arrays */
   gridcomp = hypre_CTAlloc(HYPRE_Real, num_levels_mgr + 2, HYPRE_MEMORY_HOST);
   opcomp   = hypre_CTAlloc(HYPRE_Real, num_levels_mgr + 2, HYPRE_MEMORY_HOST);
   memcomp  = hypre_CTAlloc(HYPRE_Real, num_levels_mgr + 2, HYPRE_MEMORY_HOST);

   /* Compute complexities at each MGR level */
   for (i = 0; i < num_levels_mgr + 1; i++)
   {
      if (num_sublevels_amg[i] > 0)
      {
         if (i < num_levels_mgr)
         {
            amg_solver = (hypre_ParAMGData *) A_FF_solver[i];
         }
         else
         {
            amg_solver = coarse_amg_solver;
         }

         for (k = 0; k < num_sublevels_amg[i]; k++)
         {
            A_array[k]  = hypre_ParAMGDataAArray(amg_solver)[k];
            P_array[k]  = (k < (num_sublevels_amg[i] - 1)) ?
                          hypre_ParAMGDataPArray(amg_solver)[k] : NULL;
            RT_array[k] = (k < (num_sublevels_amg[i] - 1)) ?
                          hypre_ParAMGDataRArray(amg_solver)[k] : NULL;

            gridcomp[i] += (HYPRE_Real) hypre_ParCSRMatrixGlobalNumRows(A_array[k]);
            opcomp[i]   += (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_array[k]);
            if (k < (num_sublevels_amg[i] - 1))
            {
               memcomp[i] += (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_array[k]) +
                             (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P_array[k]) +
                             (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(RT_array[k]);
            }
         }
         gridcomp[num_levels_mgr + 1] += gridcomp[i];
         opcomp[num_levels_mgr + 1]   += opcomp[i];
         memcomp[num_levels_mgr + 1]  += memcomp[i];

         gridcomp[i] /= (HYPRE_Real) hypre_ParCSRMatrixGlobalNumRows(A_array[0]);
         opcomp[i]   /= (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_array[0]);
         memcomp[i]  /= (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_array[0]);
      }
      else
      {
         /* TODO (VPM): Assume single-level for now, extend to ILU later */
         gridcomp[i] = 1.0;
         opcomp[i]   = 1.0;
         memcomp[i]  = 1.0;

         gridcomp[num_levels_mgr] += gridcomp[i];
         opcomp[num_levels_mgr]   += opcomp[i];
         memcomp[num_levels_mgr]  += memcomp[i];
      }
   }
   gridcomp[num_levels_mgr + 1] /= (HYPRE_Real) hypre_ParCSRMatrixGlobalNumRows(A_finest);
   opcomp[num_levels_mgr + 1]   /= (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_finest);
   memcomp[num_levels_mgr + 1]  /= (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_finest);

   /* Print complexities */
   if (my_id == 0)
   {
      divisors[0] = 37;
      hypre_printf("MGR complexities:\n\n");
      hypre_printf("%4s ",  "lev");
      hypre_printf("%10s ", "grid");
      hypre_printf("%10s ", "operator");
      hypre_printf("%10s ", "memory");
      hypre_printf("\n");
      HYPRE_PRINT_TOP_DIVISOR(1, divisors);

      for (i = 0; i < num_levels_mgr + 1; i++)
      {
         hypre_printf("%4d ",    i);
         hypre_printf("%10.3f ", gridcomp[i]);
         hypre_printf("%10.3f ", opcomp[i]);
         hypre_printf("%10.3f ", memcomp[i]);
         hypre_printf("\n");
      }
      HYPRE_PRINT_MID_DIVISOR(1, divisors, "")
      hypre_printf("%4s ", "All");
      hypre_printf("%10.3f ", gridcomp[i]);
      hypre_printf("%10.3f ", opcomp[i]);
      hypre_printf("%10.3f ", memcomp[i]);
      hypre_printf("\n");
      HYPRE_PRINT_MID_DIVISOR(1, divisors, "")
      hypre_printf("\n\n");
   }

   /*-------------------------------------------------
    *  Free memory
    *-------------------------------------------------*/

   hypre_MatrixStatsArrayDestroy(stats_array);
   hypre_TFree(A_array, HYPRE_MEMORY_HOST);
   hypre_TFree(P_array, HYPRE_MEMORY_HOST);
   hypre_TFree(RT_array, HYPRE_MEMORY_HOST);
   hypre_TFree(num_sublevels_amg, HYPRE_MEMORY_HOST);
   hypre_TFree(gridcomp, HYPRE_MEMORY_HOST);
   hypre_TFree(opcomp, HYPRE_MEMORY_HOST);
   hypre_TFree(memcomp, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
