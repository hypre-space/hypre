/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_Schwarz_DATA_HEADER
#define hypre_Schwarz_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_SchwarzData
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int      variant;
   HYPRE_Int      domain_type;
   HYPRE_Int      overlap;
   HYPRE_Int      num_functions;
   HYPRE_Int      use_nonsymm;
   HYPRE_Real   relax_weight;

   hypre_CSRMatrix *domain_structure;
   hypre_CSRMatrix *A_boundary;
   hypre_ParVector *Vtemp;
   HYPRE_Real  *scale;
   HYPRE_Int     *dof_func;
   HYPRE_Int     *pivots;



} hypre_SchwarzData;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_SchwarzData structure
 *--------------------------------------------------------------------------*/

#define hypre_SchwarzDataVariant(schwarz_data) ((schwarz_data)->variant)
#define hypre_SchwarzDataDomainType(schwarz_data) ((schwarz_data)->domain_type)
#define hypre_SchwarzDataOverlap(schwarz_data) ((schwarz_data)->overlap)
#define hypre_SchwarzDataNumFunctions(schwarz_data) \
((schwarz_data)->num_functions)
#define hypre_SchwarzDataUseNonSymm(schwarz_data) \
((schwarz_data)->use_nonsymm)
#define hypre_SchwarzDataRelaxWeight(schwarz_data) \
((schwarz_data)->relax_weight)
#define hypre_SchwarzDataDomainStructure(schwarz_data) \
((schwarz_data)->domain_structure)
#define hypre_SchwarzDataABoundary(schwarz_data) ((schwarz_data)->A_boundary)
#define hypre_SchwarzDataVtemp(schwarz_data) ((schwarz_data)->Vtemp)
#define hypre_SchwarzDataScale(schwarz_data) ((schwarz_data)->scale)
#define hypre_SchwarzDataDofFunc(schwarz_data) ((schwarz_data)->dof_func)
#define hypre_SchwarzDataPivots(schwarz_data) ((schwarz_data)->pivots)

#endif



