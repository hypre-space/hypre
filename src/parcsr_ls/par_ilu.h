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

#ifndef hypre_ParILU_DATA_HEADER
#define hypre_ParILU_DATA_HEADER
/*--------------------------------------------------------------------------
 * hypre_ParILUData
 *--------------------------------------------------------------------------*/
typedef struct
{
  //general data
  HYPRE_Int global_solver;
  hypre_ParCSRMatrix *matA;
  hypre_ParCSRMatrix *matL;
  HYPRE_Real	     *matD;
  hypre_ParCSRMatrix *matU;
  HYPRE_Real	     droptol;
  HYPRE_Real	     lfil;
  HYPRE_Int	     maxRowNnz;
  HYPRE_Int *CF_marker_array;
  hypre_ParVector    *F;
  hypre_ParVector    *U;
  hypre_ParVector    *residual;
  HYPRE_Real    *rel_res_norms;  
  HYPRE_Int  	num_iterations;
  HYPRE_Real   **l1_norms;
  HYPRE_Real	final_rel_residual_norm;
  HYPRE_Real	tol;
  HYPRE_Int	logging;
  HYPRE_Int	print_level;
  HYPRE_Int	max_iter;
  
  HYPRE_Int	ilu_type;

  /* temp vectors for solve phase */
  hypre_ParVector   *Utemp;
  hypre_ParVector   *Ftemp;
  
} hypre_ParILUData;


#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define SMALLREAL 1e-20

#endif
