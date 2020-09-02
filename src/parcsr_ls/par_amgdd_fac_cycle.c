/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

HYPRE_Int
hypre_BoomerAMGDD_FAC( void *amgdd_vdata, HYPRE_Int first_iteration )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   hypre_ParAMGDDData   *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   HYPRE_Int cycle_type = hypre_ParAMGDDDataFACCycleType(amgdd_data);

   if (cycle_type == 1 || cycle_type == 2) hypre_BoomerAMGDD_FAC_Cycle(amgdd_vdata, hypre_ParAMGDDDataStartLevel(amgdd_data), cycle_type, first_iteration);
   else if (cycle_type == 3) hypre_BoomerAMGDD_FAC_FCycle(amgdd_vdata, first_iteration);
   else
   {
      if (myid == 0) hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING: unknown AMG-DD FAC cycle type. Defaulting to 1 (V-cycle).\n");
      hypre_ParAMGDDDataFACCycleType(amgdd_data) = 1;
      hypre_BoomerAMGDD_FAC_Cycle(amgdd_vdata, hypre_ParAMGDDDataStartLevel(amgdd_data), 1, first_iteration);
   }

   return 0;
}

HYPRE_Int hypre_BoomerAMGDD_FAC_Cycle(void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int first_iteration)
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i;

   // Get the AMG structure
   hypre_ParAMGDDData   *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_ParAMGData   *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

   // Get the composite grid
   hypre_AMGDDCompGrid          **compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data);

   // Relax on the real nodes
   hypre_BoomerAMGDD_FAC_Relax(amgdd_vdata, level, 1);

   // Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
   if (num_levels > 1)
   {
      hypre_BoomerAMGDD_FAC_Restrict( compGrid[level], compGrid[level+1], first_iteration );
      hypre_AMGDDCompGridVectorSetConstantValues( hypre_AMGDDCompGridS(compGrid[level]), 0.0 );
      hypre_AMGDDCompGridVectorSetConstantValues( hypre_AMGDDCompGridT(compGrid[level]), 0.0 );

      //  Either solve on the coarse level or recurse
      if (level+1 == num_levels-1)
      {
         hypre_BoomerAMGDD_FAC_Relax(amgdd_vdata, num_levels-1, 3);
      }
      else for (i = 0; i < cycle_type; i++)
      {
         hypre_BoomerAMGDD_FAC_Cycle(amgdd_vdata, level+1, cycle_type, first_iteration);
         first_iteration = 0;
      }

      // Interpolate up and relax
      hypre_BoomerAMGDD_FAC_Interpolate( compGrid[level], compGrid[level+1] );
   }

   hypre_BoomerAMGDD_FAC_Relax(amgdd_vdata, level, 2);

   return 0;
}

HYPRE_Int hypre_BoomerAMGDD_FAC_FCycle(void *amgdd_vdata, HYPRE_Int first_iteration)
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   HYPRE_Int level;

   // Get the AMG structure
   hypre_ParAMGDDData   *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_ParAMGData   *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

   // Get the composite grid
   hypre_AMGDDCompGrid          **compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data);

   // ... work down to coarsest ...
   if (!first_iteration)
   {
      for (level = hypre_ParAMGDDDataStartLevel(amgdd_data); level < num_levels - 1; level++)
      {
         hypre_BoomerAMGDD_FAC_Restrict( compGrid[level], compGrid[level+1], 0 );
         hypre_AMGDDCompGridVectorSetConstantValues( hypre_AMGDDCompGridS(compGrid[level]), 0.0 );
         hypre_AMGDDCompGridVectorSetConstantValues( hypre_AMGDDCompGridT(compGrid[level]), 0.0 );
      }
   }

   //  ... solve on coarsest level ...
   hypre_BoomerAMGDD_FAC_Relax(amgdd_vdata, num_levels-1, 3);

   // ... and work back up to the finest
   for (level = num_levels - 2; level > -1; level--)
   {
      // Interpolate up and relax
      hypre_BoomerAMGDD_FAC_Interpolate( compGrid[level], compGrid[level+1] );

      // V-cycle
      hypre_BoomerAMGDD_FAC_Cycle( amgdd_vdata, level, 1, 0 );
   }

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_Interpolate( hypre_AMGDDCompGrid *compGrid_f, hypre_AMGDDCompGrid *compGrid_c )
{
   hypre_AMGDDCompGridMatvec(1.0, hypre_AMGDDCompGridP(compGrid_f), hypre_AMGDDCompGridU(compGrid_c), 1.0, hypre_AMGDDCompGridU(compGrid_f));
   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_Restrict( hypre_AMGDDCompGrid *compGrid_f, hypre_AMGDDCompGrid *compGrid_c, HYPRE_Int first_iteration )
{
   // Recalculate residual on coarse grid
   if (!first_iteration) hypre_AMGDDCompGridMatvec(-1.0, hypre_AMGDDCompGridA(compGrid_c), hypre_AMGDDCompGridU(compGrid_c), 1.0, hypre_AMGDDCompGridF(compGrid_c));

   // Get update: s_l <- A_lt_l + s_l
   hypre_AMGDDCompGridMatvec(1.0, hypre_AMGDDCompGridA(compGrid_f), hypre_AMGDDCompGridT(compGrid_f), 1.0, hypre_AMGDDCompGridS(compGrid_f));

   // If we need to preserve the updates on the next level
   if (hypre_AMGDDCompGridS(compGrid_c))
   {
      hypre_AMGDDCompGridMatvec(1.0, hypre_AMGDDCompGridR(compGrid_f), hypre_AMGDDCompGridS(compGrid_f), 0.0, hypre_AMGDDCompGridS(compGrid_c));

      // Subtract restricted update from recalculated residual: f_{l+1} <- f_{l+1} - s_{l+1}
      hypre_AMGDDCompGridVectorAxpy(-1.0, hypre_AMGDDCompGridS(compGrid_c), hypre_AMGDDCompGridF(compGrid_c));
   }
   else
   {
      // Restrict and subtract update from recalculated residual: f_{l+1} <- f_{l+1} - P_l^Ts_l
      hypre_AMGDDCompGridMatvec(-1.0, hypre_AMGDDCompGridR(compGrid_f), hypre_AMGDDCompGridS(compGrid_f), 1.0, hypre_AMGDDCompGridF(compGrid_c));
   }

   // Zero out initial guess on coarse grid
   hypre_AMGDDCompGridVectorSetConstantValues(hypre_AMGDDCompGridU(compGrid_c), 0.0);

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_Relax( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param )
{
   hypre_ParAMGDDData *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   HYPRE_Int numRelax = hypre_ParAMGDDDataFACNumRelax(amgdd_data);
   HYPRE_Int i;

   if (hypre_AMGDDCompGridT(compGrid) || hypre_AMGDDCompGridQ(compGrid))
   {
      hypre_AMGDDCompGridVectorCopy(hypre_AMGDDCompGridU(compGrid), hypre_AMGDDCompGridTemp(compGrid));
      hypre_AMGDDCompGridVectorScale(-1.0, hypre_AMGDDCompGridTemp(compGrid));
   }

    for (i = 0; i < numRelax; i++)
    {
       (*hypre_ParAMGDDDataUserFACRelaxation(amgdd_data))( amgdd_vdata, level, cycle_param );
    }

   if (hypre_AMGDDCompGridT(compGrid) || hypre_AMGDDCompGridQ(compGrid))
   {
      hypre_AMGDDCompGridVectorAxpy(1.0, hypre_AMGDDCompGridU(compGrid), hypre_AMGDDCompGridTemp(compGrid));
      if (hypre_AMGDDCompGridT(compGrid))
      {
         hypre_AMGDDCompGridVectorAxpy(1.0, hypre_AMGDDCompGridTemp(compGrid), hypre_AMGDDCompGridT(compGrid));
      }
      if (hypre_AMGDDCompGridQ(compGrid))
      {
         hypre_AMGDDCompGridVectorAxpy(1.0, hypre_AMGDDCompGridTemp(compGrid), hypre_AMGDDCompGridQ(compGrid));
      }
   }

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_Jacobi( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param )
{
#if defined(HYPRE_USING_CUDA)
   return hypre_BoomerAMGDD_FAC_Jacobi_device(amgdd_vdata, level);
#endif

   HYPRE_Int i,j;

   hypre_ParAMGDDData *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   hypre_AMGDDCompGridMatrix *A = hypre_AMGDDCompGridA(compGrid);
   hypre_AMGDDCompGridVector *f = hypre_AMGDDCompGridF(compGrid);
   hypre_AMGDDCompGridVector *u = hypre_AMGDDCompGridU(compGrid);
   HYPRE_Real relax_weight = hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);

   // Calculate l1_norms if necessary (right now, I'm just using this vector for the diagonal of A and doing straight ahead Jacobi)
   if (!hypre_AMGDDCompGridL1Norms(compGrid))
   {
      HYPRE_Int total_real_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid) + hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);
      hypre_AMGDDCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, total_real_nodes, hypre_AMGDDCompGridMemoryLocation(compGrid));
      hypre_CSRMatrix *diag = hypre_AMGDDCompGridMatrixOwnedDiag(A);
      for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
         {
            // hypre_AMGDDCompGridL1Norms(compGrid)[i] += fabs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i) hypre_AMGDDCompGridL1Norms(compGrid)[i] = hypre_CSRMatrixData(diag)[j];
         }
      }
      diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
      for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
         {
            // hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += fabs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i) hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] = hypre_CSRMatrixData(diag)[j];
         }
      }
   }

   // Allocate temporary vector if necessary
   if (!hypre_AMGDDCompGridTemp2(compGrid))
   {
      hypre_AMGDDCompGridTemp2(compGrid) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp2(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }

   hypre_AMGDDCompGridVectorCopy(f, hypre_AMGDDCompGridTemp2(compGrid));

   hypre_AMGDDCompGridMatvec(-relax_weight, A, u, relax_weight, hypre_AMGDDCompGridTemp2(compGrid));

   for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
      hypre_VectorData(hypre_AMGDDCompGridVectorOwned(u))[i] += hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridTemp2(compGrid)))[i] / hypre_AMGDDCompGridL1Norms(compGrid)[i];
   for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
      hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(u))[i] += hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridTemp2(compGrid)))[i] / hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)];

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_GaussSeidel( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param )
{
   HYPRE_Int               i, j; // loop variables
   HYPRE_Complex           diagonal; // placeholder for the diagonal of A

   hypre_ParAMGDDData *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   hypre_AMGDDCompGridMatrix *A = hypre_AMGDDCompGridA(compGrid);
   hypre_AMGDDCompGridVector *f = hypre_AMGDDCompGridF(compGrid);
   hypre_AMGDDCompGridVector *u = hypre_AMGDDCompGridU(compGrid);

   // Get all the info
   HYPRE_Complex *u_owned_data = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(u));
   HYPRE_Complex *u_nonowned_data = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(u));
   HYPRE_Complex *f_owned_data = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(f));
   HYPRE_Complex *f_nonowned_data = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(f));
   hypre_CSRMatrix *owned_diag = hypre_AMGDDCompGridMatrixOwnedDiag(A);
   hypre_CSRMatrix *owned_offd = hypre_AMGDDCompGridMatrixOwnedOffd(A);
   hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   hypre_CSRMatrix *nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   // Do Gauss-Seidel relaxation on the owned nodes
   for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
   {
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
      if (diagonal == 0.0) hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING: Divide by zero diagonal in hypre_BoomerAMGDD_FAC_GaussSeidel().\n");
      u_owned_data[i] /= diagonal;
   }

   // Do Gauss-Seidel relaxation on the nonowned nodes
   for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
   {
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
      if (diagonal == 0.0) hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING: Divide by zero diagonal in hypre_BoomerAMGDD_FAC_GaussSeidel().\n");
      u_nonowned_data[i] /= diagonal;
   }

   return 0;
}

HYPRE_Int hypre_BoomerAMGDD_FAC_OrderedGaussSeidel( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param )
{
   HYPRE_Int               unordered_i, i, j; // loop variables
   HYPRE_Complex           diagonal; // placeholder for the diagonal of A

   hypre_ParAMGDDData *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   hypre_AMGDDCompGridMatrix *A = hypre_AMGDDCompGridA(compGrid);
   hypre_AMGDDCompGridVector *f = hypre_AMGDDCompGridF(compGrid);
   hypre_AMGDDCompGridVector *u = hypre_AMGDDCompGridU(compGrid);

   if (!hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid))
   {
      hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid) = hypre_CTAlloc(HYPRE_Int, hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
      hypre_topo_sort(hypre_CSRMatrixI(hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid))), hypre_CSRMatrixJ(hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid))), hypre_CSRMatrixData(hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid))), hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid));
   }
   if (!hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid))
   {
      hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid) = hypre_CTAlloc(HYPRE_Int, hypre_AMGDDCompGridNumNonOwnedNodes(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
      hypre_topo_sort(hypre_CSRMatrixI(hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid))), hypre_CSRMatrixJ(hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid))), hypre_CSRMatrixData(hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid))), hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid), hypre_AMGDDCompGridNumNonOwnedNodes(compGrid));
   }

   // Get all the info
   HYPRE_Complex *u_owned_data = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(u));
   HYPRE_Complex *u_nonowned_data = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(u));
   HYPRE_Complex *f_owned_data = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(f));
   HYPRE_Complex *f_nonowned_data = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(f));
   hypre_CSRMatrix *owned_diag = hypre_AMGDDCompGridMatrixOwnedDiag(A);
   hypre_CSRMatrix *owned_offd = hypre_AMGDDCompGridMatrixOwnedOffd(A);
   hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   hypre_CSRMatrix *nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   // Do Gauss-Seidel relaxation on the nonowned real nodes
   for (unordered_i = 0; unordered_i < hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); unordered_i++)
   {
      i = hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid)[unordered_i];

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
      if (diagonal == 0.0) hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING: Divide by zero diagonal in hypre_BoomerAMGDD_FAC_OrderedGaussSeidel().\n");
      u_nonowned_data[i] /= diagonal;
   }

   // Do Gauss-Seidel relaxation on the owned nodes
   for (unordered_i = 0; unordered_i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); unordered_i++)
   {
      i = hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid)[unordered_i];

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
      if (diagonal == 0.0) hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING: Divide by zero diagonal in hypre_BoomerAMGDD_FAC_OrderedGaussSeidel().\n");
      u_owned_data[i] /= diagonal;
   }


   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_CFL1Jacobi( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param )
{
#if defined(HYPRE_USING_CUDA)
   if (cycle_param == 1)
   {
      hypre_BoomerAMGDD_FAC_CFL1Jacobi_device(amgdd_vdata, level, 1);
      hypre_BoomerAMGDD_FAC_CFL1Jacobi_device(amgdd_vdata, level, 0);
   }
   else if (cycle_param == 2)
   {
      hypre_BoomerAMGDD_FAC_CFL1Jacobi_device(amgdd_vdata, level, 0);
      hypre_BoomerAMGDD_FAC_CFL1Jacobi_device(amgdd_vdata, level, 1);
   }
   else hypre_BoomerAMGDD_FAC_CFL1Jacobi_device(amgdd_vdata, level, 0);
#else
   if (cycle_param == 1)
   {
      hypre_BoomerAMGDD_FAC_CFL1Jacobi_cpu(amgdd_vdata, level, 1);
      hypre_BoomerAMGDD_FAC_CFL1Jacobi_cpu(amgdd_vdata, level, 0);
   }
   else if (cycle_param == 2)
   {
      hypre_BoomerAMGDD_FAC_CFL1Jacobi_cpu(amgdd_vdata, level, 0);
      hypre_BoomerAMGDD_FAC_CFL1Jacobi_cpu(amgdd_vdata, level, 1);
   }
   else hypre_BoomerAMGDD_FAC_CFL1Jacobi_cpu(amgdd_vdata, level, 0);
#endif

   return 0;
}


HYPRE_Int
hypre_BoomerAMGDD_FAC_CFL1Jacobi_cpu( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int relax_set )
{
   HYPRE_Int            i, j;

   hypre_ParAMGDDData *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];

   HYPRE_Real relax_weight = hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);

   hypre_CSRMatrix *owned_diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid));
   hypre_CSRMatrix *owned_offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid));
   hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid));
   hypre_CSRMatrix *nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid));

   HYPRE_Complex *owned_u = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridU(compGrid)));
   HYPRE_Complex *nonowned_u = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridU(compGrid)));

   HYPRE_Complex *owned_f = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridF(compGrid)));
   HYPRE_Complex *nonowned_f = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridF(compGrid)));

   if (!hypre_AMGDDCompGridTemp2(compGrid))
   {
      hypre_AMGDDCompGridTemp2(compGrid) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp2(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }

   HYPRE_Complex *owned_tmp = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridTemp2(compGrid)));
   HYPRE_Complex *nonowned_tmp = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridTemp2(compGrid)));

   HYPRE_Real     *l1_norms = hypre_AMGDDCompGridL1Norms(compGrid);
   HYPRE_Int      *cf_marker = hypre_AMGDDCompGridCFMarkerArray(compGrid);

   HYPRE_Real    res;

   /*-----------------------------------------------------------------
   * Copy current approximation into temporary vector.
   *-----------------------------------------------------------------*/

   #ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
   {
      owned_tmp[i] = owned_u[i];
   }
   #ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedNodes(compGrid); i++)
   {
      nonowned_tmp[i] = nonowned_u[i];
   }

   /*-----------------------------------------------------------------
   * Relax only C or F points as determined by relax_points.
   *-----------------------------------------------------------------*/

   #ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,res) HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
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
      }
   }
   for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
   {
      if (cf_marker[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] == relax_set)
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
         nonowned_u[i] += (relax_weight * res)/l1_norms[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)];
      }
   }

   return 0;
}






