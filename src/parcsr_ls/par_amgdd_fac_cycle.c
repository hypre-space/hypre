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

// #define DEBUG_COMP_GRID 1 // if true, prints out what is stored in the comp grids for each processor to a file at different points in the iteration

HYPRE_Int
Project( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c );

HYPRE_Int
Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c );

HYPRE_Int
Relax( hypre_ParCompGrid *compGrid );

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle( void *amg_vdata )
{

	HYPRE_Int   myid, num_procs;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
	hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

	HYPRE_Int level, i, j; // loop variables
	HYPRE_Int numCoarseRelax = 20; // number of relaxations used to solve the coarse grid

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
		Relax( compGrid[level] );
		// Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
		Restrict( compGrid[level], compGrid[level+1] );
	}

	//  ... solve on coarsest level ...
	for (i = 0; i < numCoarseRelax; i++) Relax( compGrid[num_levels-1] );

	// ... and work back up to the finest
	for (level = num_levels - 2; level > -1; level--)
	{
		// Project up and relax
		Project( compGrid[level], compGrid[level+1] );
		Relax( compGrid[level] );
	}

	return 0;
}

HYPRE_Int
Project( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c )
{
	HYPRE_Int 					i, j; // loop variables

	// Loop over nodes on the fine grid
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
	{
		// Loop over entries in row of P
		for (j = hypre_ParCompGridPRowPtr(compGrid_f)[i]; j < hypre_ParCompGridPRowPtr(compGrid_f)[i+1]; j++)
		{
			// Debugging: make sure everyone has full interpolation stencil
			if (hypre_ParCompGridPColInd(compGrid_f)[j] < 0) printf("A point doesn't have its full interpolation stencil! P row %d, entry %d is < 0\n",i,j);
			// Update fine grid solution with coarse projection
			hypre_ParCompGridU(compGrid_f)[i] += hypre_ParCompGridPData(compGrid_f)[j] * hypre_ParCompGridU(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ];
		}
	}
	return 0;
}

HYPRE_Int
Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c )
{

	HYPRE_Int   myid;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

	HYPRE_Int 					i, j, k; // loop variables

	// Zero out coarse grid right hand side where we will restrict from fine grid
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++)
	{
		if (hypre_ParCompGridCoarseResidualMarker(compGrid_c)[i] == 2) hypre_ParCompGridF(compGrid_c)[i] = 0.0;
	}

	// Calculate fine grid residuals and restrict where appropriate
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
	{
		// Initialize res to RHS
		HYPRE_Complex res = hypre_ParCompGridF(compGrid_f)[i];
		HYPRE_Int do_restrict = 1;

		// Loop over entries in A
		for (j = hypre_ParCompGridARowPtr(compGrid_f)[i]; j < hypre_ParCompGridARowPtr(compGrid_f)[i+1]; j++)
		{
			// If -1 index encountered, mark the coarse grid connections to this node (don't want to restrict to these)
			if ( hypre_ParCompGridAColInd(compGrid_f)[j] == -1 )
			{
				do_restrict = 0;
				break;
			}
			// Otherwise just subtract off A_ij * u_j
			else res -= hypre_ParCompGridAData(compGrid_f)[j] * hypre_ParCompGridU(compGrid_f)[ hypre_ParCompGridAColInd(compGrid_f)[j] ];
			if (hypre_ParCompGridAColInd(compGrid_f)[j] >= hypre_ParCompGridNumNodes(compGrid_f)) printf("Rank %d, index %d is out of bounds, num_nodes_f = %d\n", myid, hypre_ParCompGridAColInd(compGrid_f)[j], hypre_ParCompGridNumNodes(compGrid_f));
		}
		if (do_restrict)
		{
			for (j = hypre_ParCompGridPRowPtr(compGrid_f)[i]; j < hypre_ParCompGridPRowPtr(compGrid_f)[i+1]; j++)
			{
				if (hypre_ParCompGridCoarseResidualMarker(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ] == 2)
					hypre_ParCompGridF(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ] += res*hypre_ParCompGridPData(compGrid_f)[j];
			}
		}
	}
	
	// Set residual on coarse grid where there was no (or incorrect) restriction from fine grid
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

	// Zero out initial guess on coarse grid
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_c); i++) hypre_ParCompGridU(compGrid_c)[i] = 0.0;

	return 0;
}

HYPRE_Int
Relax( hypre_ParCompGrid *compGrid )
{
	HYPRE_Int 					i, j; // loop variables
   HYPRE_Int               is_ghost;
	HYPRE_Complex 				diag; // placeholder for the diagonal of A

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
				// Debugging: make sure we have the full neighborhood for all real nodes
				if (hypre_ParCompGridAColInd(compGrid)[j] < 0) printf("Real node doesn't have its full stencil in A! row %d, entry %d\n",i,j);
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