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

#ifndef hypre_SYS_BAMG_HEADER
#define hypre_SYS_BAMG_HEADER


#define DEBUG_SYSBAMG 1

#define DEBUG_SYSBAMG_PFMG 0

#define sysbamg_dbgmsg( format, args... ) \
{ \
  if ( DEBUG_SYSBAMG ) hypre_printf( "DEBUG_SYSBAMG (%s:%d): " format, __FILE__, __LINE__, ## args ); fflush(stdout); \
}


/*--------------------------------------------------------------------------
 * hypre_SysBAMGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm               comm;

   HYPRE_Real             tol;
   HYPRE_Int              max_iter;
   HYPRE_Int              rel_change;
   HYPRE_Int              zero_guess;
   HYPRE_Int              max_levels;  /* max_level <= 0 means no limit */

   HYPRE_Int              relax_type;     /* type of relaxation to use */
   HYPRE_Real             jacobi_weight;  /* weighted jacobi weight */
   HYPRE_Int              usr_jacobi_weight; /* indicator flag for user weight */

   HYPRE_Int              num_pre_relax;  /* number of pre relaxation sweeps */
   HYPRE_Int              num_post_relax; /* number of post relaxation sweeps */
   HYPRE_Int              skip_relax;     /* flag to allow skipping relaxation */
   HYPRE_Real             dxyz[3];     /* parameters used to determine cdir */

   HYPRE_Int              num_levels;

   HYPRE_Int*             cdir_l;  /* coarsening directions */
   HYPRE_Int*             active_l;  /* flags to relax on level l*/

   hypre_SStructPGrid**   PGrid_l;
   hypre_SStructPGrid**   P_PGrid_l;

   HYPRE_Real*            data;
   hypre_SStructPMatrix** A_l;
   hypre_SStructPMatrix** P_l;
   hypre_SStructPMatrix** RT_l;
   hypre_SStructPVector** b_l;
   hypre_SStructPVector** x_l;

   HYPRE_Int              num_refine;         // number of SVD/Re-setup cycles to perform
   HYPRE_Int              num_rtv;            // num random test vectors
   HYPRE_Int              num_stv;            // num singular test vectors (or eigenvectors)
   HYPRE_Int              num_pre_relax_tv;   // num relaxation iterations before coarsening tv's
   HYPRE_Int              num_post_relax_tv;  // num relaxation iterations after prolongating tv's
   HYPRE_Int              symmetric;          // true iff matrix is symmetric

   /* temp vectors */
   hypre_SStructPVector** tx_l;
   hypre_SStructPVector** r_l;
   hypre_SStructPVector** e_l;

   void**                 relax_data_l;
   void**                 matvec_data_l;
   void**                 restrict_data_l;
   void**                 interp_data_l;

   /* log info (always logged) */
   HYPRE_Int              num_iterations;
   HYPRE_Int              time_index;
   HYPRE_Int              print_level;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int              logging;
   HYPRE_Real*            norms;
   HYPRE_Real*            rel_norms;

} hypre_SysBAMGData;

#endif
