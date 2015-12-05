/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/





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
   double   relax_weight;

   hypre_CSRMatrix *domain_structure;
   hypre_CSRMatrix *A_boundary;
   hypre_ParVector *Vtemp;
   double  *scale;
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



