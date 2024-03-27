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

const char*
hypre_MGRGetGlobalRelaxName(hypre_ParMGRData  *mgr_data,
                            HYPRE_Int          level )
{
   HYPRE_Int    smoother_type = hypre_ParMGRDataLevelSmoothTypeI(mgr_data, level);

   if ((mgr_data -> level_smooth_iters)[level] < 1)
   {
      return "--";
   }

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
      {
         hypre_ParILUData *ilu_smoother = (hypre_ParILUData*)
                                          hypre_ParMGRDataLevelSmootherI(mgr_data, level);
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
      }

      default:
         return "Unknown";
   }
}

/*--------------------------------------------------------------------
 * hypre_MGRGetFRelaxName
 *--------------------------------------------------------------------*/

const char*
hypre_MGRGetFRelaxName(hypre_ParMGRData  *mgr_data,
                       HYPRE_Int          level )
{
   HYPRE_Int  F_relax_type = hypre_ParMGRDataFRelaxTypeI(mgr_data, level);

   if ((mgr_data -> num_relax_sweeps)[level] < 1)
   {
      return "--";
   }

   switch (F_relax_type)
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
         return "Default AMG";

      case 2:
         return "User AMG";

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

      case 9:
         return "GaussElim";

      case 13:
         return "Forward L1-hGS";

      case 14:
         return "Backward L1-hGS";

      case 16:
         return "Chebyshev";

      case 19:
         return "LU";

      case 99:
         return "LU piv";

      case 199:
         return "Dense Inv";

      default:
         return "Unknown";
   }
}

/*--------------------------------------------------------------------
 * hypre_MGRGetProlongationName
 *--------------------------------------------------------------------*/

const char*
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

const char*
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

      case 14:
         return "Blk-ColLumped";

      default:
         return "Classical";
   }
}

/*--------------------------------------------------------------------
 * hypre_MGRGetCoarseGridName
 *--------------------------------------------------------------------*/

const char*
hypre_MGRGetCoarseGridName(hypre_ParMGRData  *mgr_data,
                           HYPRE_Int          level )
{
   switch (hypre_ParMGRDataCoarseGridMethodI(mgr_data, level))
   {
      case 0:
         return "Glk-RAP";

      case 1:
         return "NG-BlkDiag";

      case 2:
         return "NG-CPR-Diag";

      case 3:
         return "NG-CPR-BlkDiag";

      case 4:
         return "NG-ApproxInv";

      case 5:
         return "Glk-RAI";

      default:
         return "Unknown";
   }
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
   HYPRE_Int                  print_level     = (mgr_data -> print_level);
   hypre_ParCSRMatrix        *A_finest        = hypre_ParMGRDataA(mgr_data, 0);
   hypre_ParCSRMatrix        *A_coarsest      = hypre_ParMGRDataRAP(mgr_data);
   HYPRE_Int                  num_levels_mgr  = hypre_ParMGRDataNumCoarseLevels(mgr_data);
   HYPRE_Solver               coarse_solver   = hypre_ParMGRDataCoarseGridSolver(mgr_data);
   HYPRE_Solver             **A_FF_solver     = hypre_ParMGRDataAFFsolver(mgr_data);
   HYPRE_Int                 *Frelax_type     = hypre_ParMGRDataFRelaxType(mgr_data);

   /* Finest matrix variables */
   MPI_Comm                   comm            = hypre_ParCSRMatrixComm(A_finest);
   HYPRE_MemoryLocation       memory_location = hypre_ParCSRMatrixMemoryLocation(A_finest);
   HYPRE_ExecutionPolicy      exec            = hypre_GetExecPolicy1(memory_location);

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
   HYPRE_Int                  i, k, myid;
   HYPRE_Int                  divisors[1];

   /* Print statistics only if first print_level bit is set */
   if (!(print_level & HYPRE_MGR_PRINT_INFO_SETUP))
   {
      return hypre_error_flag;
   }

   /*-------------------------------------------------
    *  Initialize and allocate data
    *-------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);
   num_threads = hypre_NumThreads();

   coarsest_mgr_level = num_levels_mgr;
   num_sublevels_amg  = hypre_CTAlloc(HYPRE_Int, num_levels_mgr + 1, HYPRE_MEMORY_HOST);

   /* Check MGR's coarse level solver */
   if ((HYPRE_PtrToParSolverFcn) hypre_ParMGRDataCoarseGridSolverSetup(mgr_data) ==
       HYPRE_BoomerAMGSetup)
   {
      coarse_amg_solver = (hypre_ParAMGData *) coarse_solver;
      num_sublevels_amg[coarsest_mgr_level] = hypre_ParAMGDataNumLevels(coarse_amg_solver);
   }
#ifdef HYPRE_USING_DSUPERLU
   else if ((HYPRE_PtrToParSolverFcn) hypre_ParMGRDataCoarseGridSolverSetup(mgr_data) ==
            (HYPRE_PtrToParSolverFcn) hypre_MGRDirectSolverSetup)
   {
      /* TODO (VPM): Set SuperLU solver specifics */
      num_sublevels_amg[coarsest_mgr_level] = 0;
   }
#endif
   else
   {
      hypre_TFree(num_sublevels_amg, HYPRE_MEMORY_HOST);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown coarsest level solver for MGR!\n");
      return hypre_error_flag;
   }
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

   if (!myid)
   {
      hypre_printf("\n\n");
      hypre_printf(" Num MPI tasks = %d\n",  num_procs);
      hypre_printf(" Num OpenMP threads = %d\n", num_threads);
      hypre_printf(" Execution policy = %s\n\n", HYPRE_GetExecutionPolicyName(exec));
      hypre_printf("\n");
      hypre_printf("MGR SETUP PARAMETERS:\n\n");
      hypre_printf("        MGR num levels = %d\n", num_levels_mgr);
      hypre_printf(" coarse AMG num levels = %d\n", num_sublevels_amg[coarsest_mgr_level]);
      hypre_printf("      Total num levels = %d\n\n", num_levels_total);

      divisors[0] = 84;
      //hypre_printf("\nMGR level options:\n\n");
      hypre_printf("%18s %14s %16s\n", "Global", "Fine", "Coarse");
      hypre_printf("%3s %14s %14s %16s %16s %16s\n", "lev",
                   "relaxation", "relaxation", "grid method",
                   "Prolongation", "Restriction");
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
      hypre_printf("\n\n");
   }

   /*-------------------------------------------------
    *  Print MGR hierarchy info
    *-------------------------------------------------*/

   /* Set pointers to level matrices */
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

   if (!myid)
   {
      const char *msg[] = { "Full Operator Matrix Hierarchy Information:\n\n",
                            "MGR's coarsest level",
                            "\t( MGR )",
                            "\t( AMG )"
                          };

      num_levels[0] = num_levels_mgr - 1;
      num_levels[1] = num_sublevels_amg[coarsest_mgr_level] + 1;
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

   if (!myid)
   {
      const char *msg[] = { "Full Prolongation Matrix Hierarchy Information:\n\n",
                            "MGR's coarsest level",
                            "\t( MGR )",
                            "\t( AMG )"
                          };

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
         if (!myid)
         {
            hypre_printf("At MGR level %d --> F-relaxation solver parameters\n\n", i);
         }
         amg_solver = (hypre_ParAMGData *) A_FF_solver[i];

         /* General AMG info */
         if (!myid)
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
         if (!myid)
         {
            const char *msg[] = {"Operator Matrix Hierarchy Information:\n\n"};
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
         if (!myid)
         {
            const char *msg[] = {"Prolongation Matrix Hierarchy Information:\n\n"};
            num_levels[0] = num_sublevels_amg[i] - 1;
            hypre_MatrixStatsArrayPrint(1, num_levels, 1, 3, msg, stats_array);
         }
      }
   }

   /*-------------------------------------------------
    *  Print MGR coarsest level solver info
    *-------------------------------------------------*/

   if (!myid && coarse_amg_solver && num_sublevels_amg[coarsest_mgr_level] > 1)
   {
      hypre_printf("At coarsest MGR level --> Solver parameters:\n\n");
      hypre_BoomerAMGPrintGeneralInfo(coarse_amg_solver, 3);
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
         opcomp[num_levels_mgr + 1]   += opcomp[i] /
                                         hypre_ParCSRMatrixNumNonzeros(A_finest);
         memcomp[num_levels_mgr + 1]  += memcomp[i] /
                                         hypre_ParCSRMatrixNumNonzeros(A_finest);

         gridcomp[i] /= (HYPRE_Real) hypre_ParCSRMatrixGlobalNumRows(A_array[0]);
         opcomp[i]   /= hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
         memcomp[i]  /= hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
      }
      else
      {
         /* TODO (VPM): Assume single-level for now, extend to ILU later */
         gridcomp[i] = 1.0;
         opcomp[i]   = 1.0; /* TODO (VPM): adjust according to F/G-relaxation choices */
         memcomp[i]  = 1.0;

         A_array[i] = hypre_ParMGRDataA(mgr_data, i);
         gridcomp[num_levels_mgr + 1] += hypre_ParCSRMatrixGlobalNumRows(A_array[i]);
         opcomp[num_levels_mgr + 1]   += hypre_ParCSRMatrixDNumNonzeros(A_array[i]) /
                                         hypre_ParCSRMatrixNumNonzeros(A_finest);
         memcomp[num_levels_mgr + 1]  += hypre_ParCSRMatrixDNumNonzeros(A_array[i]) /
                                         hypre_ParCSRMatrixNumNonzeros(A_finest);
      }
   }
   gridcomp[num_levels_mgr + 1] /= (HYPRE_Real) hypre_ParCSRMatrixGlobalNumRows(A_finest);

   /* Print complexities */
   if (!myid)
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
