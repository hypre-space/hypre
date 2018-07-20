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
	for (i = 0; i < num_levels - 1; i++)
	{
		// Relax on the real nodes
		Relax( compGrid[level] );
		// Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
		Restrict( compGrid[level], compGrid[level+1] );
	}

	//  ... solve on coarsest level ...
	for (i = 0; i < numCoarseRelax; i++) Relax( compGrid[level] );

	// ... and work back up to the finest
	for (i = num_levels - 2; i > -1; i--)
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
	hypre_ParCompMatrixRow 		*row; // row of matrix P

	// Loop over nodes on the fine grid
	for (i = 0; i < hypre_ParCompGridNumNodes(compGrid_f); i++)
	{
		// Loop over entries in row of P
		for (j = hypre_ParCompGridPRowPtr(compGrid_f)[i]; j < hypre_ParCompGridPRowPtr(compGrid_f)[i+1]; j++)
		{
			// Debugging: make sure everyone has full interpolation stencil
			if (hypre_ParCompGridPColInd(compGrid)[j] < 0) printf("A point doesn't have its full interpolation stencil! P row %d, entry %d is < 0\n",i,j);
			// Update fine grid solution with coarse projection
			hypre_ParCompGridU(compGrid_f)[i] += hypre_ParCompGridPData(compGrid)[j] * hypre_ParCompGridU(compGrid_c)[ hypre_ParCompGridPColInd(compGrid_f)[j] ];
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
	hypre_ParCompMatrixRow 		*row; // variable to store required matrix rows
	HYPRE_Complex 				*res, *restrict_res;
	HYPRE_Int 					*coarse_res_marker; 

	// Allocate space for the calculated residual, temporary restricted residual, and the restriction marker
	res = hypre_CTAlloc(HYPRE_Complex, num_nodes_f, HYPRE_MEMORY_HOST);
	restrict_res = hypre_CTAlloc(HYPRE_Complex, num_nodes_c, HYPRE_MEMORY_HOST);
	coarse_res_marker = hypre_CTAlloc(HYPRE_Int, num_nodes_c, HYPRE_MEMORY_HOST); // mark the coarse dofs as we restrict (or don't) to make sure they are all updated appropriately: 0 = nothing has happened yet, 1 = has incomplete residual info, 2 = restricted to from fine grid


	// Calculate fine grid residuals and restrict
	for (i = 0; i < num_nodes_f; i++)
	{
		// Get row of A
		row = A_rows_f[i];

		// Initialize res to RHS
		res[i] = f_f[i];

		// Loop over entries in A
		for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
		{
			// If -1 index encountered, mark the coarse grid connections to this node (don't want to restrict to these)
			if ( hypre_ParCompMatrixRowLocalIndices(row)[j] == -1 )
			{
				for (k = 0; k < hypre_ParCompMatrixRowSize(P_rows[i]); k++)
				{
					coarse_res_marker[ hypre_ParCompMatrixRowLocalIndices(P_rows[i])[k] ] = 1; // Mark coarse dofs that we don't want to restrict to from fine grid
				}
				break;
			}
			// Otherwise just subtract off A_ij * u_j
			else res[i] -= hypre_ParCompMatrixRowData(row)[j] * u_f[ hypre_ParCompMatrixRowLocalIndices(row)[j] ];
			if (hypre_ParCompMatrixRowLocalIndices(row)[j] >= num_nodes_f) printf("Rank %d, index %d is out of bounds, num_nodes_f = %d\n", myid, hypre_ParCompMatrixRowLocalIndices(row)[j], num_nodes_f);
		}
	}
	
	// Restrict where we have complete residual information
	for (i = 0; i < num_nodes_f; i++)
	{
		// Get row of P associated with node i
		row = P_rows[i];

		// Loop over entries in P
		for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
		{
			// Add contribution to restricted residual where appropriate
			if (coarse_res_marker[ hypre_ParCompMatrixRowLocalIndices(row)[j] ] != 1) 
			{
				restrict_res[ hypre_ParCompMatrixRowLocalIndices(row)[j] ] += hypre_ParCompMatrixRowData(row)[j] * res[i];
				coarse_res_marker[ hypre_ParCompMatrixRowLocalIndices(row)[j] ] = 2; // Mark coarse dofs that successfully recieve their value from restriction from the fine grid
			}
		}
	}

	// Set residual on coarse grid where there was no restriction from fine grid
	for (i = 0; i < num_nodes_c; i++)
	{
		if (coarse_res_marker[i] != 2)
		{
			restrict_res[i] = f_c[i];
			// Loop over row of coarse grid operator 
			row = A_rows_c[i];
			for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
			{
            if (hypre_ParCompMatrixRowLocalIndices(row)[j] >= 0) restrict_res[i] -= hypre_ParCompMatrixRowData(row)[j] * u_c[ hypre_ParCompMatrixRowLocalIndices(row)[j] ];
			}
		}
	}

	// Now restrict_res should hold all appropriate restricted residaul values, so copy into f_c
	for (i = 0; i < num_nodes_c; i++) f_c[i] = restrict_res[i];

	// Zero out initial guess on coarse grid
	for (i = 0; i < num_nodes_c; i++) u_c[i] = 0;

	// Cleanup memory
	hypre_TFree(res, HYPRE_MEMORY_HOST);
	hypre_TFree(restrict_res, HYPRE_MEMORY_HOST);
	hypre_TFree(coarse_res_marker, HYPRE_MEMORY_HOST);

	return 0;
}

HYPRE_Int
Relax( hypre_ParCompGrid *compGrid )
{
	HYPRE_Int 					i, j; // loop variables
   HYPRE_Int               is_ghost;
	hypre_ParCompMatrixRow 		*row; // variable to store required matrix rows
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