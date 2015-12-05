/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header info for the FAC solver
 *
 *****************************************************************************/
/*--------------------------------------------------------------------------
 * hypre_FACData:
 *--------------------------------------------------------------------------*/

#ifndef hypre_FAC_HEADER
#define hypre_FAC_HEADER

typedef struct
{
   MPI_Comm               comm;
  
   int                   *plevels;
   hypre_Index           *prefinements;

   int                    max_levels;
   int                   *level_to_part;
   int                   *part_to_level;
   hypre_Index           *refine_factors;       /* refine_factors[level] */

   hypre_SStructGrid    **grid_level;
   hypre_SStructGraph   **graph_level;

   hypre_SStructMatrix   *A_rap;
   hypre_SStructMatrix  **A_level;
   hypre_SStructVector  **b_level;
   hypre_SStructVector  **x_level;
   hypre_SStructVector  **r_level;
   hypre_SStructVector  **e_level;
   hypre_SStructPVector **tx_level;
   hypre_SStructVector   *tx;


   void                 **matvec_data_level;
   void                 **pmatvec_data_level;
   void                  *matvec_data;
   void                 **relax_data_level;
   void                 **restrict_data_level;
   void                 **interp_data_level;

   int                    csolver_type;
   HYPRE_SStructSolver    csolver;
   HYPRE_SStructSolver    cprecond;

   double                 tol;
   int                    max_cycles;
   int                    zero_guess;
   int                    relax_type;
   double                 jacobi_weight;  /* weighted jacobi weight */
   int                    usr_jacobi_weight; /* indicator flag for user weight */

   int                    num_pre_smooth;
   int                    num_post_smooth;

   /* log info (always logged) */
   int                    num_iterations;
   int                    time_index;
   int                    rel_change;
   int                    logging;
   double                *norms;
   double                *rel_norms;


} hypre_FACData;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_FACData
 *--------------------------------------------------------------------------*/

#define hypre_FACDataMaxLevels(fac_data)\
((fac_data) -> max_levels)
#define hypre_FACDataLevelToPart(fac_data)\
((fac_data) -> level_to_part)
#define hypre_FACDataPartToLevel(fac_data)\
((fac_data) -> part_to_level)
#define hypre_FACDataRefineFactors(fac_data)\
((fac_data) -> refine_factors)
#define hypre_FACDataRefineFactorsLevel(fac_data, level)\
((fac_data) -> refinements[level])


#endif
