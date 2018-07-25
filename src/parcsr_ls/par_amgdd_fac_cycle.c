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
FAC_Project( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c );

HYPRE_Int
FAC_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int level );

HYPRE_Int
FAC_Relax( hypre_ParCompGrid *compGrid, HYPRE_Int type );

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle( void *amg_vdata )
{

   char filename[256];
	HYPRE_Int   myid, num_procs;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
	hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

	HYPRE_Int level, i, j; // loop variables
	HYPRE_Int numCoarseRelax = 20; // number of relaxations used to solve the coarse grid
   HYPRE_Int relax_type = 0;

	// Get the AMG structure
  	hypre_ParAMGData   *amg_data = amg_vdata;
  	HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

	// Get the composite grid
  	hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

  	// Do FAC V-cycle 

   for (level = 0; level < num_levels; level++)
   {
      sprintf(filename, "outputs/comp_global_indices%d_level%d", myid, level);
      hypre_ParCompGridGlobalIndicesDump(compGrid[level], filename);
   }

	// ... work down to coarsest ...
	for (level = 0; level < num_levels - 1; level++)
	{
		// Relax on the real nodes
      if (level <= 1) FAC_Relax( compGrid[level], relax_type );
      sprintf(filename, "outputs/comp_u%d_level%d_relax1", myid, level);
      hypre_ParCompGridUDump(compGrid[level],filename);
		// Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
		FAC_Restrict( compGrid[level], compGrid[level+1], level );
	}

	//  ... solve on coarsest level ...
	// for (i = 0; i < numCoarseRelax; i++) FAC_Relax( compGrid[num_levels-1], relax_type );
   sprintf(filename, "outputs/comp_u%d_level%d_relax2", myid, num_levels-1);
   hypre_ParCompGridUDump(compGrid[num_levels-1],filename);

	// ... and work back up to the finest
	for (level = num_levels - 2; level > -1; level--)
	{
		// Project up and relax
		FAC_Project( compGrid[level], compGrid[level+1] );
      sprintf(filename, "outputs/comp_u%d_level%d_project", myid, level);
      hypre_ParCompGridUDump(compGrid[level],filename);
		if (level <= 1) FAC_Relax( compGrid[level], relax_type );
      sprintf(filename, "outputs/comp_u%d_level%d_relax2", myid, level);
      hypre_ParCompGridUDump(compGrid[level],filename);
	}

	return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle_timed( void *amg_vdata, HYPRE_Int time_part )
{

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   HYPRE_Int level, i, j; // loop variables
   HYPRE_Int numCoarseRelax = 20; // number of relaxations used to solve the coarse grid
   HYPRE_Int relax_type = 0;

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // Do FAC V-cycle 

   // ... work down to coarsest ...
   for (level = 0; level < num_levels - 1; level++)
   {
      // Relax on the real nodes
      if (time_part == 1) FAC_Relax( compGrid[level], relax_type );
      // FAC_Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
      if (time_part == 2) FAC_Restrict( compGrid[level], compGrid[level+1], level );
   }

   //  ... solve on coarsest level ...
   if (time_part == 1) for (i = 0; i < numCoarseRelax; i++) FAC_Relax( compGrid[num_levels-1], relax_type );

   // ... and work back up to the finest
   for (level = num_levels - 2; level > -1; level--)
   {
      // FAC_Project up and relax
      if (time_part == 3) FAC_Project( compGrid[level], compGrid[level+1] );
      if (time_part == 1) FAC_Relax( compGrid[level], relax_type );
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
			#if DEBUG_FAC
			if (hypre_ParCompGridPColInd(compGrid_f)[j] < 0) printf("A point doesn't have its full interpolation stencil! P row %d, entry %d is < 0\n",i,j);
			#endif
         // Update fine grid solution with coarse FAC_projection
			hypre_ParCompGridU(compGrid_f)[i] += hypre_ParCompGridPData(compGrid_f)[j] * hypre_ParCompGridU(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ];
		}
	}
	return 0;
}

HYPRE_Int
FAC_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int level )
{
   char filename[256];
	HYPRE_Int   myid;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

	HYPRE_Int 					i, j, k; // loop variables

	// Zero out coarse grid right hand side where we will FAC_restrict from fine grid
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++)
	{
		if (hypre_ParCompGridCoarseResidualMarker(compGrid_c)[i] == 2) hypre_ParCompGridF(compGrid_c)[i] = 0.0;
	}



   sprintf(filename, "outputs/comp_r%d_level%d", myid, level);
   FILE *file;
   file = fopen(filename, "w");

	// Calculate fine grid residuals and FAC_restrict where appropriate
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
	{
		// Initialize res to RHS
		HYPRE_Complex res = hypre_ParCompGridF(compGrid_f)[i];
		HYPRE_Int do_FAC_restrict = 1;

		// Loop over entries in A
		for (j = hypre_ParCompGridARowPtr(compGrid_f)[i]; j < hypre_ParCompGridARowPtr(compGrid_f)[i+1]; j++)
		{
			// If -1 index encountered, mark the coarse grid connections to this node (don't want to FAC_restrict to these)
			if ( hypre_ParCompGridAColInd(compGrid_f)[j] == -1 )
			{
				do_FAC_restrict = 0;
				break;
			}
			// Otherwise just subtract off A_ij * u_j
			else res -= hypre_ParCompGridAData(compGrid_f)[j] * hypre_ParCompGridU(compGrid_f)[ hypre_ParCompGridAColInd(compGrid_f)[j] ];
			if (hypre_ParCompGridAColInd(compGrid_f)[j] >= hypre_ParCompGridNumNodes(compGrid_f)) printf("Rank %d, index %d is out of bounds, num_nodes_f = %d\n", myid, hypre_ParCompGridAColInd(compGrid_f)[j], hypre_ParCompGridNumNodes(compGrid_f));
		}
		if (do_FAC_restrict)
		{
			for (j = hypre_ParCompGridPRowPtr(compGrid_f)[i]; j < hypre_ParCompGridPRowPtr(compGrid_f)[i+1]; j++)
			{
				if (hypre_ParCompGridCoarseResidualMarker(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ] == 2)
					hypre_ParCompGridF(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ] += res*hypre_ParCompGridPData(compGrid_f)[j];
			}
		}
      fprintf(file, "%e\n", res);
	}
	
	// Set residual on coarse grid where there was no (or incorrect) FAC_restriction from fine grid
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++)
	{
		if (hypre_ParCompGridCoarseResidualMarker(compGrid_c)[i] != 2)
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

   sprintf(filename, "outputs/comp_f%d_level%d", myid, level+1);
   hypre_ParCompGridFDump(compGrid_c,filename);

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
   HYPRE_Int               is_ghost;
   HYPRE_Complex           diag; // placeholder for the diagonal of A

   // Temporary vector to calculate Jacobi sweep
   HYPRE_Complex           *u_temp = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_HOST);

   // Do Gauss-Seidel relaxation on the real nodes
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridGhostMarker(compGrid)) is_ghost = hypre_ParCompGridGhostMarker(compGrid)[i];
      else is_ghost = 0;
      if (!is_ghost)
      {
         // Initialize u as RHS
         u_temp[i] = hypre_ParCompGridF(compGrid)[i];
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
               u_temp[i] -= hypre_ParCompGridAData(compGrid)[j] * hypre_ParCompGridU(compGrid)[ hypre_ParCompGridAColInd(compGrid)[j] ];
            }
         }

         // Divide by diagonal
         if (diag == 0.0) printf("Tried to divide by zero diagonal!\n");
         u_temp[i] /= diag;
      }
   }

   // Copy over relaxed vector
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridGhostMarker(compGrid)) is_ghost = hypre_ParCompGridGhostMarker(compGrid)[i];
      else is_ghost = 0;
      if (!is_ghost)
      {
         hypre_ParCompGridU(compGrid)[i] = u_temp[i];
      }
   }
   hypre_TFree(u_temp, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int
FAC_GaussSeidel( hypre_ParCompGrid *compGrid )
{
   HYPRE_Int               i, j; // loop variables
   HYPRE_Int               is_ghost;
   HYPRE_Complex           diag; // placeholder for the diagonal of A

   // Do Gauss-Seidel relaxation on the real nodes
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridGhostMarker(compGrid)) is_ghost = hypre_ParCompGridGhostMarker(compGrid)[i];
      else is_ghost = 0;
      if (!is_ghost)
      {
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
      }
   }

   return 0;
}