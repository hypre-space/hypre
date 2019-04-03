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

HYPRE_Int
FAC_Cycle(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int first_iteration);

HYPRE_Int 
FAC_FCycle(void *amg_vdata, HYPRE_Int first_iteration);

HYPRE_Int
FAC_Cycle_timed(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int time_part);

HYPRE_Int 
FAC_FCycle_timed(void *amg_vdata, HYPRE_Int time_part);

HYPRE_Int
FAC_Project( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c );

HYPRE_Int
FAC_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int first_iteration );

HYPRE_Int
FAC_Simple_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int level );

HYPRE_Int
FAC_Relax( hypre_ParCompGrid *compGrid, HYPRE_Int type );

HYPRE_Int
FAC_Jacobi( hypre_ParCompGrid *compGrid );

HYPRE_Int
FAC_GaussSeidel( hypre_ParCompGrid *compGrid );

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle( void *amg_vdata, HYPRE_Int first_iteration )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Int cycle_type = hypre_ParAMGDataFACCycleType(amg_data);

   if (cycle_type == 1 || cycle_type == 2) FAC_Cycle(amg_vdata, 0, cycle_type, first_iteration);
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

   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Int cycle_type = hypre_ParAMGDataFACCycleType(amg_data);

   if (cycle_type == 1 || cycle_type == 2) FAC_Cycle_timed(amg_vdata, 0, cycle_type, time_part);
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

   HYPRE_Int i, j; // loop variables

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);


   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // Relax on the real nodes
   for (i = 0; i < numRelax[1]; i++) FAC_Relax( compGrid[level], relax_type );

   // Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
   if (level < transition_level) FAC_Restrict( compGrid[level], compGrid[level+1], first_iteration );
   else FAC_Simple_Restrict( compGrid[level], compGrid[level+1], level );

   if (hypre_ParCompGridS(compGrid[level])) for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) hypre_ParCompGridS(compGrid[level])[i] = 0.0;
   if (hypre_ParCompGridT(compGrid[level])) for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) hypre_ParCompGridT(compGrid[level])[i] = 0.0;

   //  Either solve on the coarse level or recurse
   if (level+1 == num_levels-1) for (i = 0; i < 20; i++) FAC_Relax( compGrid[num_levels-1], relax_type );
   else for (i = 0; i < cycle_type; i++)
   {
      FAC_Cycle(amg_vdata, level+1, cycle_type, first_iteration);
      first_iteration = 0;
   }

   // Project up and relax
   FAC_Project( compGrid[level], compGrid[level+1] );

   for (i = 0; i < numRelax[2]; i++) FAC_Relax( compGrid[level], relax_type );

   return 0;
}

HYPRE_Int FAC_FCycle(void *amg_vdata, HYPRE_Int first_iteration)
{
   char filename[256];
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   HYPRE_Int level, i, j; // loop variables

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // ... work down to coarsest ... Note: proper restricted values already stored on and above transition level
   if (!first_iteration)
   {
      for (level = 0; level < num_levels - 1; level++)
      {
         // Restrict down from the transition level
         if (level < transition_level) FAC_Restrict( compGrid[level], compGrid[level+1], 0 );
         else FAC_Simple_Restrict( compGrid[level], compGrid[level+1], level );

         if (hypre_ParCompGridS(compGrid[level])) for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) hypre_ParCompGridS(compGrid[level])[i] = 0.0;
         if (hypre_ParCompGridT(compGrid[level])) for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) hypre_ParCompGridT(compGrid[level])[i] = 0.0;

      }
   }

   //  ... solve on coarsest level ...
   for (i = 0; i < 20; i++) FAC_Relax( compGrid[num_levels-1], relax_type );

   // ... and work back up to the finest
   for (level = num_levels - 2; level > -1; level--)
   {
      // Project up and relax
      FAC_Project( compGrid[level], compGrid[level+1] );

      // V-cycle
      FAC_Cycle( amg_vdata, level, 1, 0 );
   }

   return 0;
}

HYPRE_Int FAC_Cycle_timed(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int time_part)
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i, j; // loop variables

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // Relax on the real nodes
   if (time_part == 1) for (i = 0; i < numRelax[1]; i++) FAC_Relax( compGrid[level], relax_type );

   // Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
   if (time_part == 2)
   { 
      if (level < transition_level) FAC_Restrict( compGrid[level], compGrid[level+1], 1 );
      else FAC_Simple_Restrict( compGrid[level], compGrid[level+1], level );
   }

   if (hypre_ParCompGridS(compGrid[level])) for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) hypre_ParCompGridS(compGrid[level])[i] = 0.0;
   if (hypre_ParCompGridT(compGrid[level])) for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) hypre_ParCompGridT(compGrid[level])[i] = 0.0;

   //  Either solve on the coarse level or recurse
   if (level+1 == num_levels-1) for (i = 0; i < 20; i++) FAC_Relax( compGrid[num_levels-1], relax_type );
   else for (i = 0; i < cycle_type; i++) FAC_Cycle_timed(amg_vdata, level+1, cycle_type, time_part);

   // Project up and relax
   if (time_part == 3) FAC_Project( compGrid[level], compGrid[level+1] );

   if (time_part == 1) for (i = 0; i < numRelax[2]; i++) FAC_Relax( compGrid[level], relax_type );

   return 0;
}

HYPRE_Int FAC_FCycle_timed(void *amg_vdata, HYPRE_Int time_part)
{
   char filename[256];
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   HYPRE_Int level, i, j; // loop variables

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // ... work down to coarsest ... Note: proper restricted values already stored on and above transition level
   for (level = 0; level < num_levels - 1; level++)
   {
      // Restrict down from the transition level
      if (time_part == 2)
      {
         if (level < transition_level) FAC_Restrict( compGrid[level], compGrid[level+1], 0 );
         else FAC_Simple_Restrict( compGrid[level], compGrid[level+1], level );
      }

      if (hypre_ParCompGridS(compGrid[level])) for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) hypre_ParCompGridS(compGrid[level])[i] = 0.0;
      if (hypre_ParCompGridT(compGrid[level])) for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++) hypre_ParCompGridT(compGrid[level])[i] = 0.0;

   }

   //  ... solve on coarsest level ...
   if (time_part == 1) for (i = 0; i < 20; i++) FAC_Relax( compGrid[num_levels-1], relax_type );

   // ... and work back up to the finest
   for (level = num_levels - 2; level > -1; level--)
   {
      // Project up and relax
      if (time_part == 3) FAC_Project( compGrid[level], compGrid[level+1] );

      // V-cycle
      FAC_Cycle_timed( amg_vdata, level, 1, time_part );
   }

   return 0;
}

HYPRE_Int
FAC_Project( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c )
{
	HYPRE_Int 					i, j; // loop variables

	// Loop over nodes on the fine grid
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
	{
		// Loop over entries in row of P
		for (j = hypre_ParCompGridPRowPtr(compGrid_f)[i]; j < hypre_ParCompGridPRowPtr(compGrid_f)[i+1]; j++)
		{
         // Update fine grid solution with coarse FAC_projection
			if (hypre_ParCompGridPColInd(compGrid_f)[j] >= 0)
         {
            hypre_ParCompGridU(compGrid_f)[i] += hypre_ParCompGridPData(compGrid_f)[j] * hypre_ParCompGridU(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ];
         }
      }
	}
	return 0;
}

HYPRE_Int
FAC_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int first_iteration )
{
   char filename[256];
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int               i, j, k; // loop variables

   // Recalculate residual on coarse grid
   if (!first_iteration)
   {
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++)
      {
         for (j = hypre_ParCompGridARowPtr(compGrid_c)[i]; j < hypre_ParCompGridARowPtr(compGrid_c)[i+1]; j++)
         {
            if (hypre_ParCompGridAColInd(compGrid_c)[j] >= 0) 
            {
               hypre_ParCompGridF(compGrid_c)[i] -= hypre_ParCompGridAData(compGrid_c)[j] * hypre_ParCompGridU(compGrid_c)[ hypre_ParCompGridAColInd(compGrid_c)[j] ];   
            }
         }
      }
   }

   // Get update: s_l <- A_lt_l + s_l (NOTE: I'm assuming here that A is symmetric and computing s_l <- A_l^Tt_l + s_l)
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
   {
      for (j = hypre_ParCompGridARowPtr(compGrid_f)[i]; j < hypre_ParCompGridARowPtr(compGrid_f)[i+1]; j++)
      {
         hypre_ParCompGridS(compGrid_f)[ hypre_ParCompGridAColInd(compGrid_f)[j] ] += hypre_ParCompGridAData(compGrid_f)[j] * hypre_ParCompGridT(compGrid_f)[i];
      }
   }

   // If we need to preserve the updates on the next level
   if (hypre_ParCompGridS(compGrid_c)) 
   {
      // Zero out coarse grid update: s_{l+1} <- 0
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++) hypre_ParCompGridS(compGrid_c)[i] = 0.0;

      // Restrict update: s_{l+1} <- P_l^Ts_l
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
      {
         for (j = hypre_ParCompGridPRowPtr(compGrid_f)[i]; j < hypre_ParCompGridPRowPtr(compGrid_f)[i+1]; j++)
         {
            if (hypre_ParCompGridPColInd(compGrid_f)[j] >= 0)
            {
               hypre_ParCompGridS(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ] += hypre_ParCompGridPData(compGrid_f)[j] * hypre_ParCompGridS(compGrid_f)[i];
            }
         }
      }

      // Subtract restricted update from recalculated residual: f_{l+1} <- f_{l+1} - s_{l+1}
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++)
      {
         hypre_ParCompGridF(compGrid_c)[i] -= hypre_ParCompGridS(compGrid_c)[i];
      }
   }
   else
   {
      // Restrict and subtract update from recalculated residual: f_{l+1} <- f_{l+1} - P_l^Ts_l
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
      {
         for (j = hypre_ParCompGridPRowPtr(compGrid_f)[i]; j < hypre_ParCompGridPRowPtr(compGrid_f)[i+1]; j++)
         {
            if (hypre_ParCompGridPColInd(compGrid_f)[j] >= 0)
            {
               hypre_ParCompGridF(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ] -= hypre_ParCompGridPData(compGrid_f)[j] * hypre_ParCompGridS(compGrid_f)[i];
            }
         }
      }
   }

   // Zero out initial guess on coarse grid
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++) hypre_ParCompGridU(compGrid_c)[i] = 0.0;

   return 0;
}

HYPRE_Int
FAC_Simple_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int level )
{
   char filename[256];
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int               i, j, k; // loop variables

   // Zero out coarse grid right hand side where we will FAC_restrict from fine grid
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++) hypre_ParCompGridF(compGrid_c)[i] = 0.0;

   // Calculate fine grid residuals and restrict
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
   {
      // Initialize res to RHS
      HYPRE_Complex res = hypre_ParCompGridF(compGrid_f)[i];

      // Loop over entries in A
      for (j = hypre_ParCompGridARowPtr(compGrid_f)[i]; j < hypre_ParCompGridARowPtr(compGrid_f)[i+1]; j++)
         res -= hypre_ParCompGridAData(compGrid_f)[j] * hypre_ParCompGridU(compGrid_f)[ hypre_ParCompGridAColInd(compGrid_f)[j] ];

      for (j = hypre_ParCompGridPRowPtr(compGrid_f)[i]; j < hypre_ParCompGridPRowPtr(compGrid_f)[i+1]; j++)
      {
         #if DEBUG_FAC
         if (hypre_ParCompGridPColInd(compGrid_f)[j] < 0) printf("Rank %d, P has -1 col index when simple restricting\n", myid);
         else if (hypre_ParCompGridPColInd(compGrid_f)[j] >= hypre_ParCompGridNumNodes(compGrid_c)) printf("Rank %d, P col index out of bounds when simple restricting\n", myid);
         #endif
         hypre_ParCompGridF(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ] += res*hypre_ParCompGridPData(compGrid_f)[j];
      }
   }
   
   // Zero out initial guess on coarse grid
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++) hypre_ParCompGridU(compGrid_c)[i] = 0.0;

   return 0;
}

HYPRE_Int
FAC_Relax( hypre_ParCompGrid *compGrid, HYPRE_Int type )
{
   if (type == 0) FAC_Jacobi(compGrid);
   else if (type == 1) FAC_GaussSeidel(compGrid);
	return 0;
}

HYPRE_Int
FAC_Jacobi( hypre_ParCompGrid *compGrid )
{
   HYPRE_Int               i, j; // loop variables
   HYPRE_Int               is_real;
   HYPRE_Complex           diag; // placeholder for the diagonal of A
   HYPRE_Complex           u_before;


   // Temporary vector to calculate Jacobi sweep
   if (!hypre_ParCompGridTemp(compGrid)) hypre_ParCompGridTemp(compGrid) = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST);

   // Do Jacobi relaxation on the real nodes
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridARowPtr(compGrid)[i+1] - hypre_ParCompGridARowPtr(compGrid)[i] > 0)
      {
         // Initialize u as RHS
         hypre_ParCompGridTemp(compGrid)[i] = hypre_ParCompGridF(compGrid)[i];
         diag = 0.0;

         // Loop over entries in A
         for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
         {
            #if DEBUG_FAC
            if (hypre_ParCompGridAColInd(compGrid)[j] < 0) printf("Real node doesn't have its full stencil in A! row %d, entry %d\n",i,j);
            #endif
            // If this is the diagonal, store for later division
            if (hypre_ParCompGridAColInd(compGrid)[j] == i) diag = hypre_ParCompGridAData(compGrid)[j];
            // Else, subtract off A_ij*u_j
            else
            {
               hypre_ParCompGridTemp(compGrid)[i] -= hypre_ParCompGridAData(compGrid)[j] * hypre_ParCompGridU(compGrid)[ hypre_ParCompGridAColInd(compGrid)[j] ];
            }
         }

         // Divide by diagonal
         if (diag == 0.0) printf("Tried to divide by zero diagonal!\n");
         hypre_ParCompGridTemp(compGrid)[i] /= diag;
      }
   }

   // Copy over relaxed vector
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridARowPtr(compGrid)[i+1] - hypre_ParCompGridARowPtr(compGrid)[i] > 0)
      {
         u_before = hypre_ParCompGridU(compGrid)[i];

         hypre_ParCompGridU(compGrid)[i] = hypre_ParCompGridTemp(compGrid)[i];

         // if (hypre_ParCompGridT(compGrid)) hypre_ParCompGridT(compGrid)[i] = hypre_ParCompGridT(compGrid)[i] - u_before;
         if (hypre_ParCompGridT(compGrid)) hypre_ParCompGridT(compGrid)[i] += hypre_ParCompGridTemp(compGrid)[i] - u_before;
      }
   }

   return 0;
}

HYPRE_Int
FAC_GaussSeidel( hypre_ParCompGrid *compGrid )
{
   HYPRE_Int               i, j; // loop variables
   HYPRE_Int               is_real;
   HYPRE_Complex           diag; // placeholder for the diagonal of A
   HYPRE_Complex           u_before;

   // Do Gauss-Seidel relaxation on the real nodes
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridARowPtr(compGrid)[i+1] - hypre_ParCompGridARowPtr(compGrid)[i] > 0)
      {
         u_before = hypre_ParCompGridU(compGrid)[i];

         // Initialize u as RHS
         hypre_ParCompGridU(compGrid)[i] = hypre_ParCompGridF(compGrid)[i];
         diag = 0.0;

         // Loop over entries in A
         for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
         {
            #if DEBUG_FAC
            if (hypre_ParCompGridAColInd(compGrid)[j] < 0) printf("Real node doesn't have its full stencil in A! row %d, entry %d\n",i,j);
            #endif
            // If this is the diagonal, store for later division
            if (hypre_ParCompGridAColInd(compGrid)[j] == i) diag = hypre_ParCompGridAData(compGrid)[j];
            // Else, subtract off A_ij*u_j
            else
            {
               hypre_ParCompGridU(compGrid)[i] -= hypre_ParCompGridAData(compGrid)[j] * hypre_ParCompGridU(compGrid)[ hypre_ParCompGridAColInd(compGrid)[j] ];
            }
         }

         // Divide by diagonal
         if (diag == 0.0) printf("Tried to divide by zero diagonal!\n");
         hypre_ParCompGridU(compGrid)[i] /= diag;

         if (hypre_ParCompGridT(compGrid)) hypre_ParCompGridT(compGrid)[i] += hypre_ParCompGridU(compGrid)[i] - u_before;
      }
   }

   return 0;
}
