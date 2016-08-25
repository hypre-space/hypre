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

#define DEBUG_COMP_GRID 0 // if true, prints out what is stored in the comp grids for each processor to a file at different points in the iteration

HYPRE_Int
Project( hypre_ParCompMatrixRow **P_rows, HYPRE_Complex *u_f, HYPRE_Complex *u_c, HYPRE_Int num_fine );

HYPRE_Int
Restrict( hypre_ParCompMatrixRow **A_rows_f, hypre_ParCompMatrixRow **A_rows_c, hypre_ParCompMatrixRow **P_rows, HYPRE_Complex *u_f, HYPRE_Complex *u_c, HYPRE_Complex *f_f, 
			HYPRE_Complex *f_c, HYPRE_Int num_nodes_f, HYPRE_Int num_real_nodes_f, HYPRE_Int num_nodes_c, HYPRE_Int num_owned_nodes_c );

HYPRE_Int
Relax( hypre_ParCompMatrixRow **A_rows, HYPRE_Complex *u, HYPRE_Complex *f, HYPRE_Int num_real_nodes );

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle( void *amg_vdata )
{
	#if DEBUG_COMP_GRID
	HYPRE_Int   myid, num_procs;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
	hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
	#endif

	HYPRE_Int level, i; // loop variables
	HYPRE_Int numCoarseRelax = 20; // number of relaxations used to solve the coarse grid

	// Get the AMG structure
  	hypre_ParAMGData   *amg_data = amg_vdata;
  	HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

	// Get the composite grid
   	hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   	// Get operators and nodes from comp grid on each level
	HYPRE_Int        *num_nodes = hypre_CTAlloc(HYPRE_Int, num_levels );
	HYPRE_Int		 *num_real_nodes = hypre_CTAlloc(HYPRE_Int, num_levels );

	HYPRE_Complex     **u = hypre_CTAlloc(HYPRE_Complex*, num_levels );
	HYPRE_Complex     **f = hypre_CTAlloc(HYPRE_Complex*, num_levels );

	HYPRE_Int        **global_indices = hypre_CTAlloc(HYPRE_Int*, num_levels );
	HYPRE_Int        **coarse_global_indices = hypre_CTAlloc(HYPRE_Int*, num_levels );
	HYPRE_Int        **coarse_local_indices = hypre_CTAlloc(HYPRE_Int*, num_levels ); 

	hypre_ParCompMatrixRow 	***A_rows = hypre_CTAlloc(hypre_ParCompMatrixRow**, num_levels );
	hypre_ParCompMatrixRow  ***P_rows = hypre_CTAlloc(hypre_ParCompMatrixRow**, num_levels );

	for (level = 0; level < num_levels; level++)
	{
		num_nodes[level] = hypre_ParCompGridNumNodes(compGrid[level]);
		num_real_nodes[level] = hypre_ParCompGridNumRealNodes(compGrid[level]);

		u[level] = hypre_ParCompGridU(compGrid[level]);
		f[level] = hypre_ParCompGridF(compGrid[level]);

		global_indices[level] = hypre_ParCompGridGlobalIndices(compGrid[level]);
		coarse_global_indices[level] = hypre_ParCompGridCoarseGlobalIndices(compGrid[level]);
		coarse_local_indices[level] = hypre_ParCompGridCoarseLocalIndices(compGrid[level]); 

		A_rows[level] = hypre_ParCompGridARows(compGrid[level]);
		P_rows[level] = hypre_ParCompGridPRows(compGrid[level]);
	}



	// // Debug: just relax on fine grid
	// for (i = 0; i < 100; i++) Relax( A_rows[0], u[0], f[0], num_real_nodes[0] );




   	// Do FAC V-cycle

	#if DEBUG_COMP_GRID
	char filename[256];
	for (level = 0; level < num_levels; level++)
	{
		hypre_sprintf(filename, "../../../scratch/CompGrids/before_Proc%dLevel%d.txt", myid, level);
		hypre_ParCompGridDebugPrint( compGrid[level], filename );
	}
	#endif


	// ... work down to coarsest ...
	for (i = 0; i < num_levels - 1; i++)
	{
		// Relax on the real nodes
		Relax( A_rows[i], u[i], f[i], num_real_nodes[i] );
		// Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
		Restrict( A_rows[i], A_rows[i+1], P_rows[i], u[i], u[i+1], f[i], f[i+1], num_nodes[i], num_real_nodes[i], num_nodes[i+1], hypre_ParCompGridNumOwnedNodes(compGrid[i+1]) );
	}

	//  ... solve on coarsest level ...
	// hypre_printf("Level %d: solve\n", i);
	for (i = 0; i < numCoarseRelax; i++) Relax( A_rows[num_levels-1], u[num_levels-1], f[num_levels-1], num_real_nodes[num_levels-1] );


	#if DEBUG_COMP_GRID
	for (level = 0; level < num_levels; level++)
	{
		hypre_sprintf(filename, "../../../scratch/CompGrids/post_relax_Proc%dLevel%d.txt", myid, level);
		hypre_ParCompGridDebugPrint( compGrid[level], filename );
	}
	#endif


	// ... and work back up to the finest
	for (i = num_levels - 2; i > -1; i--)
	{
		// Project up and relax
		Project(  P_rows[i], u[i], u[i+1], num_nodes[i] );
		Relax( A_rows[i], u[i], f[i], num_real_nodes[i] );
	}


	#if DEBUG_COMP_GRID
	for (level = 0; level < num_levels; level++)
	{
		hypre_sprintf(filename, "../../../scratch/CompGrids/after_Proc%dLevel%d.txt", myid, level);
		hypre_ParCompGridDebugPrint( compGrid[level], filename );
	}
	#endif

	// Cleanup memory
	hypre_TFree(num_nodes);
	hypre_TFree(num_real_nodes);

	hypre_TFree(u);
	hypre_TFree(f);

	hypre_TFree(global_indices);
	hypre_TFree(coarse_global_indices);
	hypre_TFree(coarse_local_indices ); 

	hypre_TFree(A_rows);
	hypre_TFree(P_rows);

	return 0;
}

HYPRE_Int
Project( hypre_ParCompMatrixRow **P_rows, HYPRE_Complex *u_f, HYPRE_Complex *u_c, HYPRE_Int num_fine )
{
	HYPRE_Int 					i, j; // loop variables
	hypre_ParCompMatrixRow 		*row; // row of matrix P

	// Loop over nodes on the fine grid
	for (i = 0; i < num_fine; i++)
	{
		// Get row of P associated with node i
		row = P_rows[i];

		// Loop over entries in row of P
		for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
		{
			// Debugging: make sure everyone has full interpolation stencil
			if (hypre_ParCompMatrixRowLocalIndices(row)[j] < 0) printf("A point doesn't have its full interpolation stencil! P row %d, entry %d is < 0\n",i,j);
			// Update fine grid solution with coarse projection
			u_f[i] += hypre_ParCompMatrixRowData(row)[j] * u_c[ hypre_ParCompMatrixRowLocalIndices(row)[j] ];
		}
	}
	return 0;
}

HYPRE_Int
Restrict( hypre_ParCompMatrixRow **A_rows_f, hypre_ParCompMatrixRow **A_rows_c, hypre_ParCompMatrixRow **P_rows, HYPRE_Complex *u_f, HYPRE_Complex *u_c, HYPRE_Complex *f_f, 
			HYPRE_Complex *f_c, HYPRE_Int num_nodes_f, HYPRE_Int num_real_nodes_f, HYPRE_Int num_nodes_c, HYPRE_Int num_owned_nodes_c )
{
	HYPRE_Int 					i, j; // loop variables
	hypre_ParCompMatrixRow 		*row; // variable to store required matrix rows
	HYPRE_Complex 				*res, *restrict_res;
	HYPRE_Int 					*restrict_marker; // markers to denote which C-points are restricted to from the fine grid

	// Allocate space for the calculated residual, temporary restricted residual, and the restriction marker
	res = hypre_CTAlloc(HYPRE_Complex, num_nodes_f);
	restrict_res = hypre_CTAlloc(HYPRE_Complex, num_nodes_c);
	restrict_marker = hypre_CTAlloc(HYPRE_Int, num_nodes_c);


	// Calculate fine grid residuals and restrict
	int no_residual_counter = 0;
	for (i = 0; i < num_nodes_f; i++)
	{
		// Get row of A
		row = A_rows_f[i];

		// Initialize res to RHS
		res[i] = f_f[i];

		// Loop over entries in A
		for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
		{
			// If -1 index encountered, disregard skip this computation (don't need a residual here)
			if ( hypre_ParCompMatrixRowLocalIndices(row)[j] == -1 )
			{
				no_residual_counter++;
				break;
			}
			// Otherwise just subtract off A_ij * u_j
			else res[i] -= hypre_ParCompMatrixRowData(row)[j] * u_f[ hypre_ParCompMatrixRowLocalIndices(row)[j] ];
		}
	}
	if (no_residual_counter > (0.5)*(num_nodes_f - num_real_nodes_f)) hypre_printf("Num nodes where no residual calculated / num ghost nodes = %f\n", ((float) no_residual_counter) / ((float) num_nodes_f - num_real_nodes_f));

	// Restrict from real nodes
	for (i = 0; i < num_real_nodes_f; i++)
	{
		// Get row of P associated with node i
		row = P_rows[i];

		// Loop over entries in P
		for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
		{
			// Add contribution to restricted residual and mark this as a node that should be restricted to from ghost nodes
			restrict_res[ hypre_ParCompMatrixRowLocalIndices(row)[j] ] += hypre_ParCompMatrixRowData(row)[j] * res[i];
			restrict_marker[ hypre_ParCompMatrixRowLocalIndices(row)[j] ] = 1;
			// Debugging: make sure local indices in P are appropriate (i.e. they point to something on this procs coarse grid and aren't -1)
			if ( (hypre_ParCompMatrixRowLocalIndices(row)[j] > num_nodes_c - 1) || (hypre_ParCompMatrixRowLocalIndices(row)[j] < 0) ) printf("Real rows of P: local index = %d, i = %d, j = %d, num_nodes = %d\n", hypre_ParCompMatrixRowLocalIndices(row)[j], i, j, num_nodes_c);
		}
	}

	// Restrict from ghost nodes where needed
	for (i = num_real_nodes_f; i < num_nodes_f; i++)
	{
		// Get row of P associated with node i
		row = P_rows[i];

		// Loop over entries in P
		for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
		{
			// If the coarse point was restricted to from a real node, add contribution to restricted residual
			if (restrict_marker[ hypre_ParCompMatrixRowLocalIndices(row)[j] ] == 1) restrict_res[ hypre_ParCompMatrixRowLocalIndices(row)[j] ] += hypre_ParCompMatrixRowData(row)[j] * res[i];
			if ( (hypre_ParCompMatrixRowLocalIndices(row)[j] > num_nodes_c - 1) || (hypre_ParCompMatrixRowLocalIndices(row)[j] < 0) ) printf("Ghost rows of P: local index = %d, i = %d, j = %d, num_nodes = %d\n", hypre_ParCompMatrixRowLocalIndices(row)[j], i, j, num_nodes_c);
		}
	}

	// Set residual on coarse grid where there was no restriction from fine grid
	int restriction_counter = 0;
	for (i = 0; i < num_nodes_c; i++)
	{
		if (!restrict_marker[i])
		{
			restrict_res[i] = f_c[i];
			// Loop over row of coarse grid operator 
			row = A_rows_c[i];
			for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
			{
				if ( hypre_ParCompMatrixRowLocalIndices(row)[j] == -1 ) break;
				else restrict_res[i] -= hypre_ParCompMatrixRowData(row)[j] * u_c[ hypre_ParCompMatrixRowLocalIndices(row)[j] ];
			}
		}
		// Debugging: count up how many coarse grid nodes are restricted to from the fine grid
		else restriction_counter++;
	}

	// Debugging: compare how many nodes are restricted to vs. num owned nodes
	if (num_owned_nodes_c > restriction_counter) printf("Num owned nodes = %d, num nodes restricted to = %d\n", num_owned_nodes_c, restriction_counter );

	// Now restrict_res should hold all appropriate restricted residaul values, so copy into f_c
	for (i = 0; i < num_nodes_c; i++) f_c[i] = restrict_res[i];

	// Zero out initial guess on coarse grid
	for (i = 0; i < num_nodes_c; i++) u_c[i] = 0;

	// Cleanup memory
	hypre_TFree(res);
	hypre_TFree(restrict_res);
	hypre_TFree(restrict_marker);

	return 0;
}

HYPRE_Int
Relax( hypre_ParCompMatrixRow **A_rows, HYPRE_Complex *u, HYPRE_Complex *f, HYPRE_Int num_real_nodes )
{
	HYPRE_Int 					i, j; // loop variables
	hypre_ParCompMatrixRow 		*row; // variable to store required matrix rows
	HYPRE_Complex 				diag; // placeholder for the diagonal of A


	// Do Gauss-Seidel relaxation on the real nodes
	for (i = 0; i < num_real_nodes; i++)
	{
		// Get row of A
		row = A_rows[i];

		// Initialize u as RHS
		u[i] = f[i];

		// Loop over entries in A
		for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
		{
			// Debugging: make sure we have the full neighborhood for all real nodes
			if (hypre_ParCompMatrixRowLocalIndices(row)[j] < 0) printf("Real node doesn't have its full stencil in A! row %d, entry %d\n",i,j);
			// If this is the diagonal, store for later division
			if (hypre_ParCompMatrixRowLocalIndices(row)[j] == i) diag = hypre_ParCompMatrixRowData(row)[j];
			// Else, subtract off A_ij*u_j
			else
			{
				u[i] -= hypre_ParCompMatrixRowData(row)[j] * u[ hypre_ParCompMatrixRowLocalIndices(row)[j] ];
			}
		}

		// Divide by diagonal
		u[i] /= diag;
	}
	return 0;
}