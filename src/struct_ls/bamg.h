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




/******************************************************************************
 *
 * Header info for the BAMG solver
 *
 *****************************************************************************/

#ifndef hypre_BAMG_HEADER
#define hypre_BAMG_HEADER

#include <assert.h>


#define DEBUG_BAMG 1

#define bamg_dbgmsg( format, args... ) \
{ \
  if ( DEBUG_BAMG ) hypre_printf( "DEBUG_BAMG: " format, ## args ); fflush(stdout); \
}


/*--------------------------------------------------------------------------
 * hypre_BAMGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;
                      
   HYPRE_Real            tol;
   HYPRE_Int             max_iter;
   HYPRE_Int             rel_change;
   HYPRE_Int             zero_guess;
   HYPRE_Int             max_levels;  /* max_level <= 0 means no limit */
                      
   HYPRE_Int             relax_type;     /* type of relaxation to use */
   HYPRE_Real            jacobi_weight;  /* weighted jacobi weight */
   HYPRE_Int             usr_jacobi_weight; /* indicator flag for user weight */

   HYPRE_Int             num_pre_relax;  /* number of pre relaxation sweeps */
   HYPRE_Int             num_post_relax; /* number of post relaxation sweeps */
   HYPRE_Int             skip_relax;     /* flag to allow skipping relaxation */
   HYPRE_Real            relax_weight;

   HYPRE_Int             num_levels;
                      
   HYPRE_Int            *cdir_l;  /* coarsening directions */
   HYPRE_Int            *active_l;  /* flags to relax on level l*/

   hypre_StructGrid    **grid_l;
   hypre_StructGrid    **P_grid_l;
                    
   HYPRE_Real           *data;
   hypre_StructMatrix  **A_l;
   hypre_StructMatrix  **P_l;
   hypre_StructMatrix  **RT_l;
   hypre_StructVector  **b_l;
   hypre_StructVector  **x_l;

   /* temp vectors */
   hypre_StructVector  **tx_l;
   hypre_StructVector  **r_l;
   hypre_StructVector  **e_l;

   void                **relax_data_l;
   void                **matvec_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   /* log info (always logged) */
   HYPRE_Int             num_iterations;
   HYPRE_Int             time_index;

   HYPRE_Int             print_level;
   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int             logging;
   HYPRE_Real           *norms;
   HYPRE_Real           *rel_norms;

} hypre_BAMGData;


#define hypre_index2rank(indexRAP, rank, dim) \
{ \
  HYPRE_Int d, s; \
  rank = 0; \
  for ( d = 0, s = 1; d < dim; d++, s*=3 ) \
    rank += (hypre_IndexD(indexRAP,d)+1) * s; \
}

#define hypre_rank2index(rank, indexRAP, dim) \
{ \
  HYPRE_Int d, s; \
  for ( d = 0, s = 1; d < dim; d++, s*=3 ) \
    hypre_IndexD(indexRAP,d) = ( rank % (3*s) ) / s - 1; \
}

#endif
