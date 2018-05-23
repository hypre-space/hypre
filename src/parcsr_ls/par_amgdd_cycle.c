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

HYPRE_Int
AddSolution( void *amg_vdata );

HYPRE_Int
ZeroInitialGuess( void *amg_vdata );

HYPRE_Int
hypre_BoomerAMGDD_Cycle( void *amg_vdata, HYPRE_Int num_comp_cycles, HYPRE_Int plot_iteration )
{
	HYPRE_Int   myid;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

	HYPRE_Int i,j,k,level;
	hypre_ParAMGData	*amg_data = amg_vdata;
	hypre_ParCompGrid 	**compGrid = hypre_ParAMGDataCompGrid(amg_data);
  	HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

	// Form residual and do residual communication
	hypre_BoomerAMGDDResidualCommunication( amg_vdata );

	// Set zero initial guess for all comp grids on all levels
	ZeroInitialGuess( amg_vdata );
	
	// Debugging: show norm u
	// HYPRE_Complex 		prev_norm_u = 0.0, norm_u = 0.0;
	// hypre_ParAMGData	*amg_data = amg_vdata;
 //   	hypre_ParCompGrid 	**compGrid = hypre_ParAMGDataCompGrid(amg_data);
 //   	HYPRE_Complex 		*u_comp = hypre_ParCompGridU(compGrid[0]);
 //   	HYPRE_Int 			num_nodes = hypre_ParCompGridNumNodes(compGrid[0]);
 //   	HYPRE_Int 			num_owned_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[0]);
	// for (j = 0; j < num_owned_nodes; j++) prev_norm_u += u_comp[j]*u_comp[j];
	// prev_norm_u = sqrt(prev_norm_u);


	// Debugging: look at residuals on each level
	HYPRE_Complex **residual_norm = hypre_CTAlloc(HYPRE_Complex*, num_comp_cycles+1, HYPRE_MEMORY_HOST);

	// Do the cycles
	for (i = 0; i < num_comp_cycles; i++)
	{

		if (i == 0 && plot_iteration >= 0)
		{
			hypre_ParAMGData	*amg_data = amg_vdata;
  			HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
			hypre_ParCompGrid 	**compGrid = hypre_ParAMGDataCompGrid(amg_data);
			char filename[256];
			for (level = 0; level < num_levels; level++)
			{
				// hypre_sprintf(filename, "/p/lscratchd/wbm/CompGrids/before%d_Proc%dLevel%d.txt", plot_iteration, myid, level);
				// hypre_sprintf(filename, "outputs/CompGrids/before%d_Proc%dLevel%d.txt", plot_iteration, myid, level);
				// hypre_ParCompGridPrintSolnRHS( compGrid[level], filename );
			}
		}

		// Debugging: store previous u
		// HYPRE_Complex *u = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridNumOwnedNodes(compGrid[0]), HYPRE_MEMORY_HOST);
		// for (j = 0; j < hypre_ParCompGridNumOwnedNodes(compGrid[0]); j++) u[j] = hypre_ParCompGridU(compGrid[0])[j];



		// Debugging: measure residuals on each level
		residual_norm[i] = hypre_CTAlloc(HYPRE_Complex, num_levels, HYPRE_MEMORY_HOST);
		for (level = 0; level < num_levels; level++)
		{
			for (j = 0; j < hypre_ParCompGridNumRealNodes(compGrid[level]); j++)
			{
				hypre_ParCompMatrixRow *row = hypre_ParCompGridARows(compGrid[level])[j];
				HYPRE_Complex residual = hypre_ParCompGridF(compGrid[level])[j];
				for (k = 0; k < hypre_ParCompMatrixRowSize(row); k++) residual -= hypre_ParCompMatrixRowData(row)[k] * hypre_ParCompGridU(compGrid[level])[ hypre_ParCompMatrixRowLocalIndices(row)[k] ];
				residual_norm[i][level] += residual*residual;
			}
			residual_norm[i][level] = sqrt(residual_norm[i][level]);
			// if (i > 0 && level < num_levels-1) if (residual_norm[i-1][level] < residual_norm[i][level]) printf("Rank %d: residual grew on level %d after FAC cycle %d on AMG-DD iteration %d!\n", myid, level, i, plot_iteration );
		}



		hypre_BoomerAMGDD_FAC_Cycle( amg_vdata );



		// Debugging: measure change in u
		// HYPRE_Complex change = 0.0, u_norm = 1.0;
		// for (j = 0; j < hypre_ParCompGridNumOwnedNodes(compGrid[0]); j++) change += (u[j] - hypre_ParCompGridU(compGrid[0])[j])*(u[j] - hypre_ParCompGridU(compGrid[0])[j]);
		// if (i == 1) for (j = 0; j < hypre_ParCompGridNumOwnedNodes(compGrid[0]); j++) u_norm += u[j]*u[j];
		// printf("Rank %d: fac cycle %d, normalized change in fine grid u = %e\n", myid, i, sqrt(change)/sqrt(u_norm));

		if (i == num_comp_cycles-1 && plot_iteration >= 0)
		{
			char filename[256];
			for (level = 0; level < num_levels; level++)
			{
				// hypre_sprintf(filename, "/p/lscratchd/wbm/CompGrids/after%d_Proc%dLevel%d.txt", plot_iteration, myid, level);
				// hypre_sprintf(filename, "outputs/CompGrids/after%d_Proc%dLevel%d.txt", plot_iteration, myid, level);
				// hypre_ParCompGridPrintSolnRHS( compGrid[level], filename );
			}
		}

		// if (plot_iteration >= 0)
		// {
		// 	char filename[256];
		// 	for (level = 0; level < num_levels; level++)
		// 	{
		// 		// hypre_sprintf(filename, "/p/lscratchd/wbm/CompGrids/after%d_Proc%dLevel%d.txt", plot_iteration, myid, level);
		// 		hypre_sprintf(filename, "outputs/CompGrids/after%d_Proc%dLevel%d.txt", i, myid, level);
		// 		hypre_ParCompGridPrintSolnRHS( compGrid[level], filename );
		// 	}
		// }

		// Debugging: show norm u after FAC cycle
		// norm_u = 0.0;
		// for (j = 0; j < num_owned_nodes; j++) norm_u += u_comp[j]*u_comp[j];
		// norm_u = sqrt(norm_u);
		// if ( (norm_u / prev_norm_u) >= 0.99 ) printf("Rank %d: ||u_%d||/||u_%d|| = %f\n", myid, i+1, i, norm_u / prev_norm_u );
	}

	// Debugging: measure residuals on each level
	// residual_norm[num_comp_cycles] = hypre_CTAlloc(HYPRE_Complex, num_levels, HYPRE_MEMORY_HOST);
	// for (level = 0; level < num_levels; level++)
	// {
	// 	for (j = 0; j < hypre_ParCompGridNumRealNodes(compGrid[level]); j++)
	// 	{
	// 		hypre_ParCompMatrixRow *row = hypre_ParCompGridARows(compGrid[level])[j];
	// 		HYPRE_Complex residual = hypre_ParCompGridF(compGrid[level])[j];
	// 		for (k = 0; k < hypre_ParCompMatrixRowSize(row); k++) residual -= hypre_ParCompMatrixRowData(row)[k] * hypre_ParCompGridU(compGrid[level])[ hypre_ParCompMatrixRowLocalIndices(row)[k] ];
	// 		residual_norm[num_comp_cycles][level] += residual*residual;
	// 	}
	// 	residual_norm[num_comp_cycles][level] = sqrt(residual_norm[num_comp_cycles][level]);
	// }
	// char filename[255];
	// sprintf(filename,"outputs/CompGrids/CompGridResiduals_rank%d_iteration%d.txt", myid, plot_iteration);
	// FILE *file;
	// file = fopen(filename,"w");
	// for (i = 0; i < num_levels; i++)
	// {
	// 	for (j = 0; j < num_comp_cycles; j++) fprintf(file, "%e ", residual_norm[j][i] );
	// 	fprintf(file, "%e\n", residual_norm[num_comp_cycles][i]);  
	// }
	// fclose(file);

	// Update fine grid solution
	AddSolution( amg_vdata );

	return 0;
}

HYPRE_Int
AddSolution( void *amg_vdata )
{
	hypre_ParAMGData	*amg_data = amg_vdata;
   	HYPRE_Complex 		*u = hypre_VectorData( hypre_ParVectorLocalVector( hypre_ParAMGDataUArray(amg_data)[0] ) );
   	hypre_ParCompGrid 	**compGrid = hypre_ParAMGDataCompGrid(amg_data);
   	HYPRE_Complex 		*u_comp = hypre_ParCompGridU(compGrid[0]);
   	HYPRE_Int 			num_owned_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[0]);
   	HYPRE_Int 			i;

   	for (i = 0; i < num_owned_nodes; i++) u[i] += u_comp[i];

   	return 0;
}

HYPRE_Int
ZeroInitialGuess( void *amg_vdata )
{
	HYPRE_Int   myid;
	hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

	hypre_ParAMGData	*amg_data = amg_vdata;
   	hypre_ParCompGrid 	**compGrid = hypre_ParAMGDataCompGrid(amg_data);
   	HYPRE_Int 			num_nodes;
   	HYPRE_Int 			num_real_nodes;
   	HYPRE_Int 			i, level;
   	HYPRE_Int 			num_levels = hypre_ParAMGDataNumLevels(amg_data);

   	for (level = 0; level < num_levels; level++)
   	{
   		num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
   		num_real_nodes = hypre_ParCompGridNumRealNodes(compGrid[level]);

   		for (i = 0; i < num_nodes; i++) hypre_ParCompGridU(compGrid[level])[i] = 0.0;

   		// Debugging: try random initial guess and zero rhs to debug FAC cycle
		// hypre_SeedRand(myid+1);
  //  		if (level == 0) for (i = 0; i < num_real_nodes; i++) hypre_ParCompGridU(compGrid[level])[i] = hypre_Rand();
  //  		for (i = 0; i < num_nodes; i++) hypre_ParCompGridF(compGrid[level])[i] = 0.0;

   		// Debugging: try zeroing out residuals away from fine grid patch
   		// if (level != 0) for (i = 0; i < num_nodes; i++) hypre_ParCompGridF(compGrid[level])[i] = 0.0;

   	}

   	return 0;
}