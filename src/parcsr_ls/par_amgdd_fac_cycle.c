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


#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"   

#define DEBUG_FAC 0
#define DUMP_INTERMEDIATE_TEST_SOLNS 0
#define DEBUGGING_MESSAGES 0

HYPRE_Int
FAC_Cycle(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int first_iteration);

HYPRE_Int 
FAC_FCycle(void *amg_vdata, HYPRE_Int first_iteration);

HYPRE_Int
FAC_Cycle_timed(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int time_part);

HYPRE_Int 
FAC_FCycle_timed(void *amg_vdata, HYPRE_Int time_part);

HYPRE_Int
FAC_Interpolate( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c );

HYPRE_Int
FAC_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int first_iteration );

HYPRE_Int
FAC_CFL1Jacobi( hypre_ParAMGData *amg_data, hypre_ParCompGrid *compGrid, HYPRE_Int relax_set );

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle( void *amg_vdata, HYPRE_Int first_iteration )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int cycle_type = hypre_ParAMGDataFACCycleType(amg_data);

   if (cycle_type == 1 || cycle_type == 2) FAC_Cycle(amg_vdata, hypre_ParAMGDataAMGDDStartLevel(amg_data), cycle_type, first_iteration);
   else if (cycle_type == 3) FAC_FCycle(amg_vdata, first_iteration);
   else
   {
      if (myid == 0) hypre_printf("Error: unknown cycle type\n");
   }

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle_timed( void *amg_vdata, HYPRE_Int time_part )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int cycle_type = hypre_ParAMGDataFACCycleType(amg_data);

   if (cycle_type == 1 || cycle_type == 2) FAC_Cycle_timed(amg_vdata, hypre_ParAMGDataAMGDDStartLevel(amg_data), cycle_type, time_part);
   else if (cycle_type == 3) FAC_FCycle_timed(amg_vdata, time_part);
   else
   {
      if (myid == 0) hypre_printf("Error: unknown cycle type\n");
   }

   return 0;
}

HYPRE_Int FAC_Cycle(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int first_iteration)
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   char filename[256];

   HYPRE_Int i;

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // Relax on the real nodes
   #if DEBUGGING_MESSAGES
   printf("Rank %d, relax on level %d\n", myid, level);
   #endif
   for (i = 0; i < numRelax[1]; i++) (*hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data))( (HYPRE_Solver) amg_vdata, compGrid[level], 1 );

   #if DUMP_INTERMEDIATE_TEST_SOLNS
   sprintf(filename, "outputs/actual/u%d_level%d_relax1", myid, level);
   hypre_SeqVectorPrint(hypre_ParCompGridUNew(compGrid[level]), filename);
   if (level == 0)
   {
     sprintf(filename, "outputs/actual/f%d_level%d", myid, level);
     hypre_SeqVectorPrint(hypre_ParCompGridFNew(compGrid[level]), filename);
   }
   #endif

   // Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
   if (num_levels > 1)
   {
      #if DEBUGGING_MESSAGES
      printf("Rank %d, restrict on level %d\n", myid, level);
      #endif
      FAC_Restrict( compGrid[level], compGrid[level+1], first_iteration );
      hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridSNew(compGrid[level]), 0.0 );
      hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridTNew(compGrid[level]), 0.0 );

      #if DUMP_INTERMEDIATE_TEST_SOLNS
      sprintf(filename, "outputs/actual/f%d_level%d", myid, level+1);
      hypre_SeqVectorPrint(hypre_ParCompGridFNew(compGrid[level+1]), filename);
      #endif

      //  Either solve on the coarse level or recurse
      if (level+1 == num_levels-1)
      {
         #if DEBUGGING_MESSAGES
         printf("Rank %d, coarse solve on level %d\n", myid, num_levels-1);
         #endif
         for (i = 0; i < numRelax[3]; i++) (*hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data))( (HYPRE_Solver) amg_vdata, compGrid[num_levels-1], 3 );

         #if DUMP_INTERMEDIATE_TEST_SOLNS
         sprintf(filename, "outputs/actual/u%d_level%d_relax2", myid, num_levels-1);
         hypre_SeqVectorPrint(hypre_ParCompGridUNew(compGrid[num_levels-1]), filename);
         #endif

      }
      else for (i = 0; i < cycle_type; i++)
      {
         #if DEBUGGING_MESSAGES
         printf("Rank %d, recurse on level %d\n", myid, level);
         #endif
         FAC_Cycle(amg_vdata, level+1, cycle_type, first_iteration);
         first_iteration = 0;
      }

      // Interpolate up and relax
      #if DEBUGGING_MESSAGES
      printf("Rank %d, interpolate on level %d\n", myid, level);
      #endif
      FAC_Interpolate( compGrid[level], compGrid[level+1] );
   }

   #if DUMP_INTERMEDIATE_TEST_SOLNS
   sprintf(filename, "outputs/actual/u%d_level%d_project", myid, level);
   hypre_SeqVectorPrint(hypre_ParCompGridUNew(compGrid[level]), filename);
   #endif

   #if DEBUGGING_MESSAGES
   printf("Rank %d, relax on level %d\n", myid, level);
   #endif
   for (i = 0; i < numRelax[2]; i++) (*hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data))( (HYPRE_Solver) amg_vdata, compGrid[level], 2 );

   #if DUMP_INTERMEDIATE_TEST_SOLNS
   sprintf(filename, "outputs/actual/u%d_level%d_relax2", myid, level);
   hypre_SeqVectorPrint(hypre_ParCompGridUNew(compGrid[level]), filename);
   #endif

   return 0;
}

HYPRE_Int FAC_FCycle(void *amg_vdata, HYPRE_Int first_iteration)
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   HYPRE_Int level, i;

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // ... work down to coarsest ... 
   if (!first_iteration)
   {
      for (level = hypre_ParAMGDataAMGDDStartLevel(amg_data); level < num_levels - 1; level++)
      {
         FAC_Restrict( compGrid[level], compGrid[level+1], 0 );
         hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridSNew(compGrid[level]), 0.0 );
         hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridTNew(compGrid[level]), 0.0 );
      }
   }

   //  ... solve on coarsest level ...
   for (i = 0; i < numRelax[3]; i++) (*hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data))( (HYPRE_Solver) amg_vdata, compGrid[num_levels-1], 3 );

   // ... and work back up to the finest
   for (level = num_levels - 2; level > hypre_ParAMGDataAMGDDStartLevel(amg_data)-1; level--)
   {
      // Interpolate up and relax
      FAC_Interpolate( compGrid[level], compGrid[level+1] );

      // V-cycle
      FAC_Cycle( amg_vdata, level, 1, 0 );
   }

   return 0;
}

HYPRE_Int FAC_Cycle_timed(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int time_part)
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i; // loop variables

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // Relax on the real nodes
   if (time_part == 1) for (i = 0; i < numRelax[1]; i++) (*hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data))( (HYPRE_Solver) amg_vdata, compGrid[level], 1 );

   // Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
   if (time_part == 2)
   {
      FAC_Restrict( compGrid[level], compGrid[level+1], 1 );
      hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridSNew(compGrid[level]), 0.0 );
      hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridTNew(compGrid[level]), 0.0 );
   }

   //  Either solve on the coarse level or recurse
   if (level+1 == num_levels-1) for (i = 0; i < numRelax[3]; i++) (*hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data))( (HYPRE_Solver) amg_vdata, compGrid[num_levels-1], 3 );
   else for (i = 0; i < cycle_type; i++) FAC_Cycle_timed(amg_vdata, level+1, cycle_type, time_part);

   // Interpolate up and relax
   if (time_part == 3) FAC_Interpolate( compGrid[level], compGrid[level+1] );

   if (time_part == 1) for (i = 0; i < numRelax[2]; i++) (*hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data))( (HYPRE_Solver) amg_vdata, compGrid[level], 2 );

   return 0;
}

HYPRE_Int FAC_FCycle_timed(void *amg_vdata, HYPRE_Int time_part)
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   HYPRE_Int level, i;

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // ... work down to coarsest ... 
   for (level = hypre_ParAMGDataAMGDDStartLevel(amg_data); level < num_levels - 1; level++)
   {
      if (time_part == 2)
      {
         FAC_Restrict( compGrid[level], compGrid[level+1], 0 );
         hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridSNew(compGrid[level]), 0.0 );
         hypre_ParCompGridVectorSetConstantValues( hypre_ParCompGridTNew(compGrid[level]), 0.0 );
      }
   }

   //  ... solve on coarsest level ...
   if (time_part == 1) for (i = 0; i < numRelax[3]; i++) (*hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data))( (HYPRE_Solver) amg_vdata, compGrid[num_levels-1], 3 );

   // ... and work back up to the finest
   for (level = num_levels - 2; level > hypre_ParAMGDataAMGDDStartLevel(amg_data)-1; level--)
   {
      // Interpolate up and relax
      if (time_part == 3) FAC_Interpolate( compGrid[level], compGrid[level+1] );

      // V-cycle
      FAC_Cycle_timed( amg_vdata, level, 1, time_part );
   }

   return 0;
}

HYPRE_Int
FAC_Interpolate( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c )
{
   hypre_ParCompGridMatvec(1.0, hypre_ParCompGridPNew(compGrid_f), hypre_ParCompGridUNew(compGrid_c), 1.0, hypre_ParCompGridUNew(compGrid_f));
   return 0;
}

HYPRE_Int
FAC_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int first_iteration )
{
   // Recalculate residual on coarse grid
   if (!first_iteration) hypre_ParCompGridMatvec(-1.0, hypre_ParCompGridANew(compGrid_c), hypre_ParCompGridUNew(compGrid_c), 1.0, hypre_ParCompGridFNew(compGrid_c));

   // Get update: s_l <- A_lt_l + s_l 
   hypre_ParCompGridMatvec(1.0, hypre_ParCompGridANew(compGrid_f), hypre_ParCompGridTNew(compGrid_f), 1.0, hypre_ParCompGridSNew(compGrid_f));

   // If we need to preserve the updates on the next level !!! Do we need this if statement? 
   if (hypre_ParCompGridSNew(compGrid_c))
   {
      hypre_ParCompGridMatvec(1.0, hypre_ParCompGridRNew(compGrid_f), hypre_ParCompGridSNew(compGrid_f), 0.0, hypre_ParCompGridSNew(compGrid_c));

      // Subtract restricted update from recalculated residual: f_{l+1} <- f_{l+1} - s_{l+1}
      hypre_ParCompGridVectorAxpy(-1.0, hypre_ParCompGridSNew(compGrid_c), hypre_ParCompGridFNew(compGrid_c));
   }
   else
   {
      // Restrict and subtract update from recalculated residual: f_{l+1} <- f_{l+1} - P_l^Ts_l
      hypre_ParCompGridMatvec(-1.0, hypre_ParCompGridRNew(compGrid_f), hypre_ParCompGridSNew(compGrid_f), 1.0, hypre_ParCompGridFNew(compGrid_c));
   }

   // Zero out initial guess on coarse grid
   hypre_ParCompGridVectorSetConstantValues(hypre_ParCompGridUNew(compGrid_c), 0.0);

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_Jacobi( HYPRE_Solver amg_vdata, hypre_ParCompGrid *compGrid, HYPRE_Int cycle_param )
{
   HYPRE_Int i,j; 
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Real relax_weight = hypre_ParAMGDataRelaxWeight(amg_data)[0];

   // !!! Optimization: don't have to do the matrix vector multiplication, etc. over the ghost dofs

   // Calculate l1_norms if necessary (right now, I'm just using this vector for the diagonal of A and doing straight ahead Jacobi)
   if (!hypre_ParCompGridL1Norms(compGrid))
   {
      HYPRE_Int total_real_nodes = hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedRealNodes(compGrid);
      hypre_ParCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, total_real_nodes, HYPRE_MEMORY_SHARED);
      hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid));
      for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
         {
            // hypre_ParCompGridL1Norms(compGrid)[i] += fabs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i) hypre_ParCompGridL1Norms(compGrid)[i] = hypre_CSRMatrixData(diag)[j];
         }
      }
      diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid));
      for (i = 0; i < hypre_ParCompGridNumNonOwnedRealNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
         {
            // hypre_ParCompGridL1Norms(compGrid)[i + hypre_ParCompGridNumOwnedNodes(compGrid)] += fabs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i) hypre_ParCompGridL1Norms(compGrid)[i + hypre_ParCompGridNumOwnedNodes(compGrid)] = hypre_CSRMatrixData(diag)[j];
         }
      }
   }

   // Allocate temporary vector if necessary
   if (!hypre_ParCompGridTempNew(compGrid))
   {
      hypre_ParCompGridTempNew(compGrid) = hypre_ParCompGridVectorCreate();
      hypre_ParCompGridVectorInitialize(hypre_ParCompGridTempNew(compGrid), hypre_ParCompGridNumOwnedNodes(compGrid), hypre_ParCompGridNumNonOwnedNodes(compGrid));
   }

   hypre_ParCompGridVectorCopy(hypre_ParCompGridFNew(compGrid),hypre_ParCompGridTempNew(compGrid));

   hypre_ParCompGridMatvec(-relax_weight, hypre_ParCompGridANew(compGrid), hypre_ParCompGridUNew(compGrid), relax_weight, hypre_ParCompGridTempNew(compGrid));

   #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   VecScale(hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridUNew(compGrid))),
            hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTempNew(compGrid))),
            hypre_ParCompGridL1Norms(compGrid),
            hypre_ParCompGridNumOwnedNodes(compGrid),
            HYPRE_STREAM(4));
   VecScale(hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridUNew(compGrid))),
            hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTempNew(compGrid))),
            &(hypre_ParCompGridL1Norms(compGrid)[ hypre_ParCompGridNumOwnedNodes(compGrid) ]),
            hypre_ParCompGridNumNonOwnedRealNodes(compGrid),
            HYPRE_STREAM(4));
   if (hypre_ParCompGridTNew(compGrid))
   {
      VecScale(hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTNew(compGrid))),
               hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTempNew(compGrid))),
               hypre_ParCompGridL1Norms(compGrid),
               hypre_ParCompGridNumOwnedNodes(compGrid),
               HYPRE_STREAM(4));
      VecScale(hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTNew(compGrid))),
               hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTempNew(compGrid))),
               &(hypre_ParCompGridL1Norms(compGrid)[ hypre_ParCompGridNumOwnedNodes(compGrid) ]),
               hypre_ParCompGridNumNonOwnedRealNodes(compGrid),
               HYPRE_STREAM(4));
   }
   if (hypre_ParCompGridQNew(compGrid))
   {
      VecScale(hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridQNew(compGrid))),
               hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTempNew(compGrid))),
               hypre_ParCompGridL1Norms(compGrid),
               hypre_ParCompGridNumOwnedNodes(compGrid),
               HYPRE_STREAM(4));
      VecScale(hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridQNew(compGrid))),
               hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTempNew(compGrid))),
               &(hypre_ParCompGridL1Norms(compGrid)[ hypre_ParCompGridNumOwnedNodes(compGrid) ]),
               hypre_ParCompGridNumNonOwnedRealNodes(compGrid),
               HYPRE_STREAM(4));
   }
   #else
   for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
      hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridUNew(compGrid)))[i] += hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTempNew(compGrid)))[i] / hypre_ParCompGridL1Norms(compGrid)[i];
   for (i = 0; i < hypre_ParCompGridNumNonOwnedRealNodes(compGrid); i++)
      hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridUNew(compGrid)))[i] += hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTempNew(compGrid)))[i] / hypre_ParCompGridL1Norms(compGrid)[i + hypre_ParCompGridNumOwnedNodes(compGrid)];
   if (hypre_ParCompGridTNew(compGrid))
   {
      for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
         hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTNew(compGrid)))[i] += hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTempNew(compGrid)))[i] / hypre_ParCompGridL1Norms(compGrid)[i];
      for (i = 0; i < hypre_ParCompGridNumNonOwnedRealNodes(compGrid); i++)
         hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTNew(compGrid)))[i] += hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTempNew(compGrid)))[i] / hypre_ParCompGridL1Norms(compGrid)[i + hypre_ParCompGridNumOwnedNodes(compGrid)];
   }
   if (hypre_ParCompGridQNew(compGrid))
   {
      for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
         hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridQNew(compGrid)))[i] += hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTempNew(compGrid)))[i] / hypre_ParCompGridL1Norms(compGrid)[i];
      for (i = 0; i < hypre_ParCompGridNumNonOwnedRealNodes(compGrid); i++)
         hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridQNew(compGrid)))[i] += hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTempNew(compGrid)))[i] / hypre_ParCompGridL1Norms(compGrid)[i + hypre_ParCompGridNumOwnedNodes(compGrid)];
   }
   #endif

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_GaussSeidel( HYPRE_Solver amg_vdata, hypre_ParCompGrid *compGrid, HYPRE_Int cycle_param )
{
   HYPRE_Int               i, j; // loop variables
   HYPRE_Complex           diagonal; // placeholder for the diagonal of A
   HYPRE_Complex           u_before;
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;

   // Get all the info
   HYPRE_Complex *u_owned_data = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridUNew(compGrid)));
   HYPRE_Complex *u_nonowned_data = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridUNew(compGrid)));
   HYPRE_Complex *f_owned_data = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridFNew(compGrid)));
   HYPRE_Complex *f_nonowned_data = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridFNew(compGrid)));
   HYPRE_Complex *t_owned_data = NULL;
   HYPRE_Complex *t_nonowned_data = NULL;
   HYPRE_Complex *q_owned_data = NULL;
   HYPRE_Complex *q_nonowned_data = NULL;
   if (hypre_ParCompGridTNew(compGrid))
   {
      t_owned_data = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTNew(compGrid)));
      t_nonowned_data = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTNew(compGrid)));
   }
   
   if (hypre_ParCompGridQNew(compGrid))
   {
      q_owned_data = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridQNew(compGrid)));
      q_nonowned_data = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridQNew(compGrid)));
   }
   hypre_CSRMatrix *owned_diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid));
   hypre_CSRMatrix *owned_offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid));
   hypre_CSRMatrix *nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid));
   hypre_CSRMatrix *nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid));

   // Do Gauss-Seidel relaxation on the owned nodes
   for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
   {
      u_before = u_owned_data[i];

      // Initialize u as RHS
      u_owned_data[i] = f_owned_data[i];
      diagonal = 0.0;

      // Loop over diag entries
      for (j = hypre_CSRMatrixI(owned_diag)[i]; j < hypre_CSRMatrixI(owned_diag)[i+1]; j++)
      {
         if (hypre_CSRMatrixJ(owned_diag)[j] == i) diagonal = hypre_CSRMatrixData(owned_diag)[j];
         else u_owned_data[i] -= hypre_CSRMatrixData(owned_diag)[j] * u_owned_data[ hypre_CSRMatrixJ(owned_diag)[j] ];
      }
      // Loop over offd entries
      for (j = hypre_CSRMatrixI(owned_offd)[i]; j < hypre_CSRMatrixI(owned_offd)[i+1]; j++)
      {
         u_owned_data[i] -= hypre_CSRMatrixData(owned_offd)[j] * u_nonowned_data[ hypre_CSRMatrixJ(owned_offd)[j] ];
      }
      // Divide by diagonal
      if (diagonal == 0.0) printf("Tried to divide by zero diagonal in Gauss-Seidel!\n");
      u_owned_data[i] /= diagonal;

      if (hypre_ParCompGridTNew(compGrid)) t_owned_data[i] += u_owned_data[i] - u_before;
      if (hypre_ParCompGridQNew(compGrid)) q_owned_data[i] += u_owned_data[i] - u_before;
   }

   // Do Gauss-Seidel relaxation on the nonowned nodes
   for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid); i++)
   {
      u_before = u_nonowned_data[i];

      // Initialize u as RHS
      u_nonowned_data[i] = f_nonowned_data[i];
      diagonal = 0.0;

      // Loop over diag entries
      for (j = hypre_CSRMatrixI(nonowned_diag)[i]; j < hypre_CSRMatrixI(nonowned_diag)[i+1]; j++)
      {
         if (hypre_CSRMatrixJ(nonowned_diag)[j] == i) diagonal = hypre_CSRMatrixData(nonowned_diag)[j];
         else u_nonowned_data[i] -= hypre_CSRMatrixData(nonowned_diag)[j] * u_nonowned_data[ hypre_CSRMatrixJ(nonowned_diag)[j] ];
      }
      // Loop over offd entries
      for (j = hypre_CSRMatrixI(nonowned_offd)[i]; j < hypre_CSRMatrixI(nonowned_offd)[i+1]; j++)
      {
         u_nonowned_data[i] -= hypre_CSRMatrixData(nonowned_offd)[j] * u_owned_data[ hypre_CSRMatrixJ(nonowned_offd)[j] ];
      }
      // Divide by diagonal
      if (diagonal == 0.0) printf("Tried to divide by zero diagonal in Gauss-Seidel!\n");
      u_nonowned_data[i] /= diagonal;

      if (hypre_ParCompGridTNew(compGrid)) t_nonowned_data[i] += u_nonowned_data[i] - u_before;
      if (hypre_ParCompGridQNew(compGrid)) q_nonowned_data[i] += u_nonowned_data[i] - u_before;
   }

   return 0;
}

HYPRE_Int hypre_BoomerAMGDD_FAC_OrderedGaussSeidel( HYPRE_Solver amg_vdata, hypre_ParCompGrid *compGrid, HYPRE_Int cycle_param  )
{
   // HYPRE_Int               unordered_i, i, j; // loop variables
   // HYPRE_Complex           diag; // placeholder for the diagonal of A
   // HYPRE_Complex           u_before;
   // hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;

   // if (!hypre_ParCompGridRelaxOrdering(compGrid)) 
   // {
   //    hypre_ParCompGridRelaxOrdering(compGrid) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST);
   //    hypre_topo_sort(hypre_ParCompGridANewRowPtr(compGrid), hypre_ParCompGridANewColInd(compGrid), hypre_ParCompGridANewData(compGrid), hypre_ParCompGridRelaxOrdering(compGrid), hypre_ParCompGridNumNodes(compGrid));
   // }

   // // Do Gauss-Seidel relaxation on the real nodes (ordered)
   // for (unordered_i = 0; unordered_i < hypre_ParCompGridNumNodes(compGrid); unordered_i++)
   // {
   //    i = hypre_ParCompGridRelaxOrdering(compGrid)[unordered_i];
      
   //    if (i < hypre_ParCompGridNumRealNodes(compGrid))
   //    {
   //       u_before = hypre_VectorData(hypre_ParCompGridUNew(compGrid))[i];

   //       // Initialize u as RHS
   //       hypre_VectorData(hypre_ParCompGridUNew(compGrid))[i] = hypre_VectorData(hypre_ParCompGridFNew(compGrid))[i];
   //       diag = 0.0;

   //       // Loop over entries in A
   //       for (j = hypre_ParCompGridANewRowPtr(compGrid)[i]; j < hypre_ParCompGridANewRowPtr(compGrid)[i+1]; j++)
   //       {
   //          #if DEBUG_FAC
   //          if (hypre_ParCompGridANewColInd(compGrid)[j] < 0) printf("Real node doesn't have its full stencil in A! row %d, entry %d\n",i,j);
   //          #endif
   //          // If this is the diagonal, store for later division
   //          if (hypre_ParCompGridANewColInd(compGrid)[j] == i) diag = hypre_ParCompGridANewData(compGrid)[j];
   //          // Else, subtract off A_ij*u_j
   //          else
   //          {
   //             hypre_VectorData(hypre_ParCompGridUNew(compGrid))[i] -= hypre_ParCompGridANewData(compGrid)[j] * hypre_VectorData(hypre_ParCompGridUNew(compGrid))[ hypre_ParCompGridANewColInd(compGrid)[j] ];
   //          }
   //       }

   //       // Divide by diagonal
   //       if (diag == 0.0) printf("Tried to divide by zero diagonal!\n");
   //       hypre_VectorData(hypre_ParCompGridUNew(compGrid))[i] /= diag;

   //       if (hypre_ParCompGridTNew(compGrid)) hypre_VectorData(hypre_ParCompGridTNew(compGrid))[i] += hypre_VectorData(hypre_ParCompGridUNew(compGrid))[i] - u_before;
   //       if (hypre_ParCompGridQ(compGrid)) hypre_VectorData(hypre_ParCompGridQ(compGrid))[i] += hypre_VectorData(hypre_ParCompGridUNew(compGrid))[i] - u_before;
   //    }
   // }

   // return 0;
}

HYPRE_Int 
hypre_BoomerAMGDD_FAC_Cheby( HYPRE_Solver amg_vdata, hypre_ParCompGrid *compGrid, HYPRE_Int cycle_param )
{
   // // !!! NOTE: is this correct??? If I'm doing a bunch of matvecs that include the ghost dofs, is that right?
   // // I think this is fine for now because I don't store the rows associated with ghost dofs, so their values shouldn't change at all, but this may change in a later version.

   // HYPRE_Int   myid;
   // hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );


   // HYPRE_Int i,j;
   // hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;

   // HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid);
   // HYPRE_Int num_real_nodes = hypre_ParCompGridNumRealNodes(compGrid);

   // hypre_CSRMatrix *A = hypre_ParCompGridANewReal(compGrid);
   // hypre_Vector *u = hypre_ParCompGridUNew(compGrid);
   // hypre_Vector *t = hypre_ParCompGridTNew(compGrid);
   // hypre_Vector *q = hypre_ParCompGridQ(compGrid);
   // hypre_Vector *f = hypre_ParCompGridFNew(compGrid);

   // HYPRE_Real    *coefs = hypre_ParCompGridChebyCoeffs(compGrid);
   // // HYPRE_Real    *coefs = hypre_ParAMGDataChebyCoefs(amg_data)[level];
   // HYPRE_Int     scale = hypre_ParAMGDataChebyScale(amg_data);
   // HYPRE_Int     order = hypre_ParAMGDataChebyOrder(amg_data);

   // HYPRE_Int cheby_order;

   // if (order > 4)
   //    order = 4;
   // if (order < 1)
   //    order = 1;

   // cheby_order = order -1;

   // // Get temporary/auxiliary vectors
   // if (!hypre_ParCompGridTemp(compGrid))
   // {
   //    hypre_ParCompGridTemp(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
   //    hypre_SeqVectorInitialize(hypre_ParCompGridTemp(compGrid));
   // }
   // if (!hypre_ParCompGridTemp2(compGrid))
   // {
   //    hypre_ParCompGridTemp2(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
   //    hypre_SeqVectorInitialize(hypre_ParCompGridTemp2(compGrid));
   // }
   // if (!hypre_ParCompGridTemp3(compGrid))
   // {
   //    hypre_ParCompGridTemp3(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
   //    hypre_SeqVectorInitialize(hypre_ParCompGridTemp3(compGrid));
   // }
   // hypre_Vector *r = hypre_ParCompGridTemp(compGrid); // length = num real 
   // hypre_VectorSize(r) = num_real_nodes;
   // hypre_Vector *u_update = hypre_ParCompGridTemp2(compGrid); // length = num nodes
   // hypre_Vector *v = hypre_ParCompGridTemp3(compGrid); // length = num real
   // hypre_VectorSize(v) = num_real_nodes;

   // // hypre_SeqVectorCopy(u, u_update);

   // if (!scale)
   // {
   //    /* get residual: r = f - A*u */
   //    hypre_SeqVectorCopy(f, r);
   //    hypre_CSRMatrixMatvec(-1.0, A, u, 1.0, r);

   //    hypre_SeqVectorCopy(r, u_update);
   //    hypre_SeqVectorScale(coefs[cheby_order], u_update);

   //    for (i = cheby_order - 1; i >= 0; i--) 
   //    {
   //       hypre_CSRMatrixMatvec(1.0, A, u_update, 0.0, v);

   //       hypre_SeqVectorAxpy(coefs[i], r, v);
   //       hypre_SeqVectorCopy(v, u_update);
   //    }
   // }
   // else /* scaling! */
   // {      
   //  /* get scaled residual: r = D^(-1/2)f -
   //     * D^(-1/2)A*u */
   //    hypre_SeqVectorCopy(f, r); 
   //    hypre_CSRMatrixMatvec(-1.0, A, u, 1.0, r);
   //    #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   //    VecComponentwiseScale(hypre_VectorData(r), hypre_ParCompGridL1Norms(compGrid), num_real_nodes, HYPRE_STREAM(4));
   //    #else
   //    for (j = 0; j < num_real_nodes; j++) hypre_VectorData(r)[j] *= hypre_ParCompGridL1Norms(compGrid)[j];
   //    #endif

   //    /* save original u, then start 
   //       the iteration by multiplying r by the cheby coef.*/
   //    hypre_SeqVectorCopy(r, u_update);
   //    hypre_SeqVectorScale(coefs[cheby_order], u_update);

   //    /* now do the other coefficients */   
   //    for (i = cheby_order - 1; i >= 0; i--) 
   //    {
   //       /* v = D^(-1/2)AD^(-1/2)u */
   //       #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   //       VecComponentwiseScale(hypre_VectorData(u_update), hypre_ParCompGridL1Norms(compGrid), num_real_nodes, HYPRE_STREAM(4));
   //       #else
   //       for (j = 0; j < num_real_nodes; j++) hypre_VectorData(u_update)[j] *= hypre_ParCompGridL1Norms(compGrid)[j];
   //       #endif
   //       hypre_CSRMatrixMatvec(1.0, A, u_update, 0.0, v);
   //       #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   //       VecComponentwiseScale(hypre_VectorData(v), hypre_ParCompGridL1Norms(compGrid), num_real_nodes, HYPRE_STREAM(4));
   //       #else
   //       for (j = 0; j < num_real_nodes; j++) hypre_VectorData(v)[j] *= hypre_ParCompGridL1Norms(compGrid)[j];
   //       #endif

   //       /* u_new = coef*r + v*/
   //       hypre_SeqVectorAxpy(coefs[i], r, v);
   //       hypre_SeqVectorCopy(v, u_update);         
   //    } /* end of cheby_order loop */

   //    /* now we have to scale u_data before adding it to u_orig*/
   //    #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   //    VecComponentwiseScale(hypre_VectorData(u_update), hypre_ParCompGridL1Norms(compGrid), num_real_nodes, HYPRE_STREAM(4));
   //    #else
   //    for (j = 0; j < num_real_nodes; j++) hypre_VectorData(u_update)[j] *= hypre_ParCompGridL1Norms(compGrid)[j];
   //    #endif

   // }/* end of scaling code */


   // // Update only over real dofs by adjusting size of vectors
   // hypre_VectorSize(u) = num_real_nodes;
   // if (t) hypre_VectorSize(t) = num_real_nodes;
   // hypre_VectorSize(u_update) = num_real_nodes;

   // if (t) hypre_SeqVectorAxpy(1.0, u_update, t);
   // if (q) hypre_SeqVectorAxpy(1.0, u_update, q);
   // hypre_SeqVectorAxpy(1.0, u_update, u);
  
   // hypre_VectorSize(u) = num_nodes;
   // if (t) hypre_VectorSize(t) = num_nodes;
   // hypre_VectorSize(u_update) = num_nodes;

   // hypre_VectorSize(r) = num_nodes;
   // hypre_VectorSize(v) = num_real_nodes;

   // return hypre_error_flag;
}


HYPRE_Int
hypre_BoomerAMGDD_FAC_CFL1Jacobi( HYPRE_Solver amg_vdata, hypre_ParCompGrid *compGrid, HYPRE_Int cycle_param )
{
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   if (cycle_param == 1)
   {
      FAC_CFL1Jacobi(amg_data, compGrid, 1); 
      FAC_CFL1Jacobi(amg_data, compGrid, 0);
   }
   else if (cycle_param == 2)
   {
      FAC_CFL1Jacobi(amg_data, compGrid, 0);
      FAC_CFL1Jacobi(amg_data, compGrid, 1);
   }
   else FAC_CFL1Jacobi(amg_data, compGrid, 0);

   return 0;
}


HYPRE_Int
FAC_CFL1Jacobi( hypre_ParAMGData *amg_data, hypre_ParCompGrid *compGrid, HYPRE_Int relax_set )
{
   HYPRE_Int            i, j;

   HYPRE_Real relax_weight = hypre_ParAMGDataRelaxWeight(amg_data)[0];

#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)

   // Get cusparse handle and setup bsr matrix
   static cusparseHandle_t handle;
   static cusparseMatDescr_t descr;
   static HYPRE_Int FirstCall=1;

   if (FirstCall)
   {
      handle=getCusparseHandle();

      cusparseStatus_t status= cusparseCreateMatDescr(&descr);
      if (status != CUSPARSE_STATUS_SUCCESS) {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: Matrix descriptor initialization failed\n");
         return hypre_error_flag;
      }

      cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

      FirstCall=0;
   }
   if (!hypre_ParCompGridTempNew(compGrid))
   {
      hypre_ParCompGridTempNew(compGrid) = hypre_ParCompGridVectorCreate();
      hypre_ParCompGridVectorInitialize(hypre_ParCompGridTempNew(compGrid), hypre_ParCompGridNumOwnedNodes(compGrid), hypre_ParCompGridNumNonOwnedNodes(compGrid));
   }
   hypre_ParCompGridVectorCopy(hypre_ParCompGridFNew(compGrid), hypre_ParCompGridTempNew(compGrid));
   double alpha = -relax_weight;
   double beta = relax_weight;

   HYPRE_Complex *owned_u = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridUNew(compGrid)));
   HYPRE_Complex *nonowned_u = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridUNew(compGrid)));
   HYPRE_Complex *owned_tmp = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTempNew(compGrid)));
   HYPRE_Complex *nonowned_tmp = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTempNew(compGrid)));

   if (relax_set)
   {
      hypre_CSRMatrix *mat = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid));
      beta = relax_weight;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_ParCompGridNumOwnedCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_ParCompGridOwnedCMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                owned_u,
                &beta,
                owned_tmp);

      mat = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid));
      beta = 1.0;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_ParCompGridNumOwnedCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_ParCompGridOwnedCMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                nonowned_u,
                &beta,
                owned_tmp);

      mat = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid));
      beta = relax_weight;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_ParCompGridNonOwnedCMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                nonowned_u,
                &beta,
                nonowned_tmp);

      mat = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid));
      beta = 1.0;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_ParCompGridNonOwnedCMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                owned_u,
                &beta,
                nonowned_tmp);

      hypre_CheckErrorDevice(cudaPeekAtLastError());
      hypre_CheckErrorDevice(cudaDeviceSynchronize());

      VecScaleMasked(owned_u,owned_tmp,hypre_ParCompGridL1Norms(compGrid),hypre_ParCompGridOwnedCMask(compGrid),hypre_ParCompGridNumOwnedCPoints(compGrid),HYPRE_STREAM(4));
      VecScaleMasked(nonowned_u,nonowned_tmp,&(hypre_ParCompGridL1Norms(compGrid)[hypre_ParCompGridNumOwnedNodes(compGrid)]),hypre_ParCompGridNonOwnedCMask(compGrid),hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),HYPRE_STREAM(4));
      
      if (hypre_ParCompGridTNew(compGrid))
      {
         HYPRE_Complex *owned_t = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTNew(compGrid)));
         HYPRE_Complex *nonowned_t = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTNew(compGrid)));
         VecScaleMasked(owned_t,owned_tmp,hypre_ParCompGridL1Norms(compGrid),hypre_ParCompGridOwnedCMask(compGrid),hypre_ParCompGridNumOwnedCPoints(compGrid),HYPRE_STREAM(4));
         VecScaleMasked(nonowned_t,nonowned_tmp,&(hypre_ParCompGridL1Norms(compGrid)[hypre_ParCompGridNumOwnedNodes(compGrid)]),hypre_ParCompGridNonOwnedCMask(compGrid),hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),HYPRE_STREAM(4));
      }
      if (hypre_ParCompGridQNew(compGrid))
      {
         HYPRE_Complex *owned_t = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridQNew(compGrid)));
         HYPRE_Complex *nonowned_t = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridQNew(compGrid)));
         VecScaleMasked(owned_t,owned_tmp,hypre_ParCompGridL1Norms(compGrid),hypre_ParCompGridOwnedCMask(compGrid),hypre_ParCompGridNumOwnedCPoints(compGrid),HYPRE_STREAM(4));
         VecScaleMasked(nonowned_t,nonowned_tmp,&(hypre_ParCompGridL1Norms(compGrid)[hypre_ParCompGridNumOwnedNodes(compGrid)]),hypre_ParCompGridNonOwnedCMask(compGrid),hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),HYPRE_STREAM(4));
      }
      hypre_CheckErrorDevice(cudaPeekAtLastError());
      hypre_CheckErrorDevice(cudaDeviceSynchronize());
   }
   else
   {
      hypre_CSRMatrix *mat = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid));
      beta = relax_weight;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_ParCompGridNumOwnedNodes(compGrid) - hypre_ParCompGridNumOwnedCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_ParCompGridOwnedFMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                owned_u,
                &beta,
                owned_tmp);

      mat = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid));
      beta = 1.0;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_ParCompGridNumOwnedNodes(compGrid) - hypre_ParCompGridNumOwnedCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_ParCompGridOwnedFMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                nonowned_u,
                &beta,
                owned_tmp);

      mat = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid));
      beta = relax_weight;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_ParCompGridNumNonOwnedRealNodes(compGrid) - hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_ParCompGridNonOwnedFMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                nonowned_u,
                &beta,
                nonowned_tmp);

      mat = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid));
      beta = 1.0;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_ParCompGridNumNonOwnedRealNodes(compGrid) - hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_ParCompGridNonOwnedFMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                owned_u,
                &beta,
                nonowned_tmp);

      hypre_CheckErrorDevice(cudaPeekAtLastError());
      hypre_CheckErrorDevice(cudaDeviceSynchronize());

      VecScaleMasked(owned_u,owned_tmp,hypre_ParCompGridL1Norms(compGrid),hypre_ParCompGridOwnedFMask(compGrid),hypre_ParCompGridNumOwnedNodes(compGrid) - hypre_ParCompGridNumOwnedCPoints(compGrid),HYPRE_STREAM(4));
      VecScaleMasked(nonowned_u,nonowned_tmp,&(hypre_ParCompGridL1Norms(compGrid)[hypre_ParCompGridNumOwnedNodes(compGrid)]),hypre_ParCompGridNonOwnedFMask(compGrid),hypre_ParCompGridNumNonOwnedRealNodes(compGrid) - hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),HYPRE_STREAM(4));
      
      if (hypre_ParCompGridTNew(compGrid))
      {
         HYPRE_Complex *owned_t = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTNew(compGrid)));
         HYPRE_Complex *nonowned_t = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTNew(compGrid)));
         VecScaleMasked(owned_t,owned_tmp,hypre_ParCompGridL1Norms(compGrid),hypre_ParCompGridOwnedFMask(compGrid),hypre_ParCompGridNumOwnedNodes(compGrid) - hypre_ParCompGridNumOwnedCPoints(compGrid),HYPRE_STREAM(4));
         VecScaleMasked(nonowned_t,nonowned_tmp,&(hypre_ParCompGridL1Norms(compGrid)[hypre_ParCompGridNumOwnedNodes(compGrid)]),hypre_ParCompGridNonOwnedFMask(compGrid),hypre_ParCompGridNumNonOwnedRealNodes(compGrid) - hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),HYPRE_STREAM(4));
      }
      if (hypre_ParCompGridQNew(compGrid))
      {
         HYPRE_Complex *owned_q = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridQNew(compGrid)));
         HYPRE_Complex *nonowned_q = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridQNew(compGrid)));
         VecScaleMasked(owned_q,owned_tmp,hypre_ParCompGridL1Norms(compGrid),hypre_ParCompGridOwnedFMask(compGrid),hypre_ParCompGridNumOwnedNodes(compGrid) - hypre_ParCompGridNumOwnedCPoints(compGrid),HYPRE_STREAM(4));
         VecScaleMasked(nonowned_q,nonowned_tmp,&(hypre_ParCompGridL1Norms(compGrid)[hypre_ParCompGridNumOwnedNodes(compGrid)]),hypre_ParCompGridNonOwnedFMask(compGrid),hypre_ParCompGridNumNonOwnedRealNodes(compGrid) - hypre_ParCompGridNumNonOwnedRealCPoints(compGrid),HYPRE_STREAM(4));
      }
      hypre_CheckErrorDevice(cudaPeekAtLastError());
      hypre_CheckErrorDevice(cudaDeviceSynchronize());
   }

#else

   hypre_CSRMatrix *owned_diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid));
   hypre_CSRMatrix *owned_offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid));
   hypre_CSRMatrix *nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid));
   hypre_CSRMatrix *nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid));

   HYPRE_Complex *owned_u = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridUNew(compGrid)));
   HYPRE_Complex *nonowned_u = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridUNew(compGrid)));

   HYPRE_Complex *owned_f = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridFNew(compGrid)));
   HYPRE_Complex *nonowned_f = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridFNew(compGrid)));

   HYPRE_Complex *owned_tmp = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTempNew(compGrid)));
   HYPRE_Complex *nonowned_tmp = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTempNew(compGrid)));

   HYPRE_Complex *owned_t = NULL;
   HYPRE_Complex *nonowned_t = NULL;
   if (hypre_ParCompGridTNew(compGrid))
   {
      owned_t = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridTNew(compGrid)));
      nonowned_t = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTNew(compGrid)));
   }

   HYPRE_Complex *owned_q = NULL;
   HYPRE_Complex *nonowned_q = NULL;
   if (hypre_ParCompGridQNew(compGrid))
   {
      owned_q = hypre_VectorData(hypre_ParCompGridVectorOwned(hypre_ParCompGridQNew(compGrid)));
      nonowned_q = hypre_VectorData(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridQNew(compGrid)));
   }

   HYPRE_Real     *l1_norms = hypre_ParCompGridL1Norms(compGrid);
   HYPRE_Int      *cf_marker = hypre_ParCompGridCFMarkerArray(compGrid);

   HYPRE_Real    res;

   /*-----------------------------------------------------------------
   * Copy current approximation into temporary vector.
   *-----------------------------------------------------------------*/

   #ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
   {
      owned_tmp[i] = owned_u[i];
   }
   #ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid); i++)
   {
      nonowned_tmp[i] = nonowned_u[i];
   }

   /*-----------------------------------------------------------------
   * Relax only C or F points as determined by relax_points.
   *-----------------------------------------------------------------*/

   #ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,res) HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
   {
      if (cf_marker[i] == relax_set)
      {
         res = owned_f[i];
         for (j = hypre_CSRMatrixI(owned_diag)[i]; j < hypre_CSRMatrixI(owned_diag)[i+1]; j++)
         {
            res -= hypre_CSRMatrixData(owned_diag)[j] * owned_tmp[ hypre_CSRMatrixJ(owned_diag)[j] ];
         }
         for (j = hypre_CSRMatrixI(owned_offd)[i]; j < hypre_CSRMatrixI(owned_offd)[i+1]; j++)
         {
            res -= hypre_CSRMatrixData(owned_offd)[j] * nonowned_tmp[ hypre_CSRMatrixJ(owned_offd)[j] ];
         }
         owned_u[i] += (relax_weight * res)/l1_norms[i];
         if (owned_t) owned_t[i] += owned_u[i] - owned_tmp[i];
         if (owned_q) owned_q[i] += owned_u[i] - owned_tmp[i];
      }
   }
   for (i = 0; i < hypre_ParCompGridNumNonOwnedRealNodes(compGrid); i++)
   {
      if (cf_marker[i + hypre_ParCompGridNumOwnedNodes(compGrid)] == relax_set)
      {
         res = nonowned_f[i];
         for (j = hypre_CSRMatrixI(nonowned_diag)[i]; j < hypre_CSRMatrixI(nonowned_diag)[i+1]; j++)
         {
            res -= hypre_CSRMatrixData(nonowned_diag)[j] * nonowned_tmp[ hypre_CSRMatrixJ(nonowned_diag)[j] ];
         }
         for (j = hypre_CSRMatrixI(nonowned_offd)[i]; j < hypre_CSRMatrixI(nonowned_offd)[i+1]; j++)
         {
            res -= hypre_CSRMatrixData(nonowned_offd)[j] * owned_tmp[ hypre_CSRMatrixJ(nonowned_offd)[j] ];
         }
         nonowned_u[i] += (relax_weight * res)/l1_norms[i + hypre_ParCompGridNumOwnedNodes(compGrid)];
         if (nonowned_t) nonowned_t[i] += nonowned_u[i] - nonowned_tmp[i];
         if (nonowned_q) nonowned_q[i] += nonowned_u[i] - nonowned_tmp[i];
      }
   }

#endif

   return 0;
}
