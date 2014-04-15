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

#include "_hypre_struct_ls.h"
#include "bamg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_BAMGSetup( void               *bamg_vdata,
    hypre_StructMatrix *A,
    hypre_StructVector *b,
    hypre_StructVector *x        )
{
  hypre_BAMGData       *bamg_data = bamg_vdata;

  MPI_Comm              comm = (bamg_data -> comm);

  HYPRE_Int             relax_type =       (bamg_data -> relax_type);
  HYPRE_Int             usr_jacobi_weight= (bamg_data -> usr_jacobi_weight);
  HYPRE_Real            jacobi_weight    = (bamg_data -> jacobi_weight);
  HYPRE_Int             skip_relax =       (bamg_data -> skip_relax);

  HYPRE_Int             max_iter;
  HYPRE_Int             max_levels;

  HYPRE_Int             num_levels;

  hypre_Index           cindex;
  hypre_Index           findex;
  hypre_Index           stride;

  HYPRE_Int            *cdir_l;
  HYPRE_Int            *active_l;
  hypre_StructGrid    **grid_l;
  hypre_StructGrid    **P_grid_l;

  HYPRE_Real           *data;
  HYPRE_Int             data_size = 0;

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

  hypre_StructGrid     *grid;
  HYPRE_Int             ndim;

  hypre_Box            *cbox;

  HYPRE_Int             cdir, periodic, cmaxsize;
  HYPRE_Int             d, l;

  HYPRE_Int             b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
  HYPRE_Int             x_num_ghost[]  = {1, 1, 1, 1, 1, 1};

#if DEBUG_BAMG
  char                  filename[255];
#endif


  /*-----------------------------------------------------
   * Set up coarse grids - full coarsening (i.e., 1 / 2^d )
   *-----------------------------------------------------*/

  bamg_dbgmsg("Set up coarse grids\n");

  grid  = hypre_StructMatrixGrid(A);
  ndim  = hypre_StructGridNDim(grid);

  /* Compute a new max_levels value based on the grid */
  cbox = hypre_BoxDuplicate(hypre_StructGridBoundingBox(grid));
  max_levels = 6;
  (bamg_data -> max_levels) = max_levels;

  grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels);
  hypre_StructGridRef(grid, &grid_l[0]);

  P_grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels);
  P_grid_l[0] = NULL;

  cdir_l = hypre_TAlloc(HYPRE_Int, max_levels);
  active_l = hypre_TAlloc(HYPRE_Int, max_levels);

  for (l = 0; ; l++)
  {
    cdir = l % ndim;

    if (cdir != -1)
    {
      /* don't coarsen if a periodic direction and not divisible by 2 */
      periodic = hypre_IndexD(hypre_StructGridPeriodic(grid_l[l]), cdir);
      if ((periodic) && (periodic % 2))
      {
        cdir = -1;
      }

      /* don't coarsen if we've reached max_levels */
      if (l == (max_levels - 1))
      {
        cdir = -1;
      }
    }

    /* stop coarsening */
    if (cdir == -1)
    {
      active_l[l] = 1; /* forces relaxation on coarsest grid */
      cmaxsize = 0;
      for (d = 0; d < ndim; d++)
      {
        cmaxsize = hypre_max(cmaxsize, hypre_BoxSizeD(cbox, d));
      }
      break;
    }

    cdir_l[l] = cdir;

    active_l[l] = 0;

    /* set cindex, findex, and stride */
    hypre_SetIndex(cindex, 0);
    hypre_SetIndex(findex, 0); hypre_IndexD(findex,cdir)=1;
    hypre_SetIndex(stride, 1); hypre_IndexD(stride,cdir)=2;

    /* coarsen cbox*/
    hypre_ProjectBox(cbox, cindex, stride);
    hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride,
        hypre_BoxIMin(cbox));
    hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride,
        hypre_BoxIMax(cbox));

    /* build the interpolation grid */
    hypre_StructCoarsen(grid_l[l], findex, stride, 0, &P_grid_l[l+1]);

    /* build the coarse grid */
    hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l+1]);
  }
  num_levels = l + 1;

  /* free up some things */
  hypre_BoxDestroy(cbox);

  /* set all levels active if skip_relax = 0 */
  if (!skip_relax)
  {
    for (l = 0; l < num_levels; l++)
    {
      active_l[l] = 1;
    }
  }

  (bamg_data -> num_levels)   = num_levels;
  (bamg_data -> cdir_l)       = cdir_l;
  (bamg_data -> grid_l)       = grid_l;
  (bamg_data -> P_grid_l)     = P_grid_l;

  /*-----------------------------------------------------
   * Set up matrix and vector structures
   *-----------------------------------------------------*/

  A_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels);
  P_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1);
  RT_l = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1);
  b_l  = hypre_TAlloc(hypre_StructVector *, num_levels);
  x_l  = hypre_TAlloc(hypre_StructVector *, num_levels);
  tx_l = hypre_TAlloc(hypre_StructVector *, num_levels);
  r_l  = tx_l;
  e_l  = tx_l;

  A_l[0] = hypre_StructMatrixRef(A);
  b_l[0] = hypre_StructVectorRef(b);
  x_l[0] = hypre_StructVectorRef(x);

  tx_l[0] = hypre_StructVectorCreate(comm, grid_l[0]);
  hypre_StructVectorSetNumGhost(tx_l[0], x_num_ghost);
  hypre_StructVectorInitializeShell(tx_l[0]);
  data_size += hypre_StructVectorDataSize(tx_l[0]);

  for (l = 0; l < (num_levels - 1); l++)
  {
    cdir = cdir_l[l];

    bamg_dbgmsg("CreateInterpOp l=%d cdir=%d\n", l, cdir);

    P_l[l]  = hypre_BAMGCreateInterpOp(A_l[l], P_grid_l[l+1], cdir);
    hypre_StructMatrixInitializeShell(P_l[l]);
    data_size += hypre_StructMatrixDataSize(P_l[l]);

    if (hypre_StructMatrixSymmetric(A))
    {
      RT_l[l] = P_l[l];
    }
    else
    {
      RT_l[l] = P_l[l];
#if 0
      /* Allow RT != P for non symmetric case */
      /* NOTE: Need to create a non-pruned grid for this to work */
      RT_l[l]   = hypre_BAMGCreateRestrictOp(A_l[l], grid_l[l+1], cdir);
      hypre_StructMatrixInitializeShell(RT_l[l]);
      data_size += hypre_StructMatrixDataSize(RT_l[l]);
#endif
    }

    bamg_dbgmsg("CreateRAPOp\n");

    A_l[l+1] = hypre_BAMGCreateRAPOp(RT_l[l], A_l[l], P_l[l],
        grid_l[l+1], cdir);
    hypre_StructMatrixInitializeShell(A_l[l+1]);
    data_size += hypre_StructMatrixDataSize(A_l[l+1]);

    b_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
    hypre_StructVectorSetNumGhost(b_l[l+1], b_num_ghost);
    hypre_StructVectorInitializeShell(b_l[l+1]);
    data_size += hypre_StructVectorDataSize(b_l[l+1]);

    x_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
    hypre_StructVectorSetNumGhost(x_l[l+1], x_num_ghost);
    hypre_StructVectorInitializeShell(x_l[l+1]);
    data_size += hypre_StructVectorDataSize(x_l[l+1]);

    tx_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
    hypre_StructVectorSetNumGhost(tx_l[l+1], x_num_ghost);
    hypre_StructVectorInitializeShell(tx_l[l+1]);
  }

  data = hypre_SharedCTAlloc(HYPRE_Real, data_size);
  (bamg_data -> data) = data;

  hypre_StructVectorInitializeData(tx_l[0], data);
  hypre_StructVectorAssemble(tx_l[0]);
  data += hypre_StructVectorDataSize(tx_l[0]);

  for (l = 0; l < (num_levels - 1); l++)
  {
    hypre_StructMatrixInitializeData(P_l[l], data);
    data += hypre_StructMatrixDataSize(P_l[l]);

#if 0
    /* Allow R != PT for non symmetric case */
    if (!hypre_StructMatrixSymmetric(A))
    {
      hypre_StructMatrixInitializeData(RT_l[l], data);
      data += hypre_StructMatrixDataSize(RT_l[l]);
    }
#endif

    hypre_StructMatrixInitializeData(A_l[l+1], data);
    data += hypre_StructMatrixDataSize(A_l[l+1]);

    hypre_StructVectorInitializeData(b_l[l+1], data);
    hypre_StructVectorAssemble(b_l[l+1]);
    data += hypre_StructVectorDataSize(b_l[l+1]);

    hypre_StructVectorInitializeData(x_l[l+1], data);
    hypre_StructVectorAssemble(x_l[l+1]);
    data += hypre_StructVectorDataSize(x_l[l+1]);

    hypre_StructVectorInitializeData(tx_l[l+1],
        hypre_StructVectorData(tx_l[0]));
    hypre_StructVectorAssemble(tx_l[l+1]);
  }

  (bamg_data -> A_l)  = A_l;
  (bamg_data -> P_l)  = P_l;
  (bamg_data -> RT_l) = RT_l;
  (bamg_data -> b_l)  = b_l;
  (bamg_data -> x_l)  = x_l;
  (bamg_data -> tx_l) = tx_l;
  (bamg_data -> r_l)  = r_l;
  (bamg_data -> e_l)  = e_l;

  /*-----------------------------------------------------
   * Set up multigrid operators and call setup routines
   *-----------------------------------------------------*/

  bamg_dbgmsg("Set up multigrid operators ...\n");

  relax_data_l    = hypre_TAlloc(void *, num_levels);
  matvec_data_l   = hypre_TAlloc(void *, num_levels);
  restrict_data_l = hypre_TAlloc(void *, num_levels);
  interp_data_l   = hypre_TAlloc(void *, num_levels);

  for (l = 0; l < (num_levels - 1); l++)
  {
    cdir = cdir_l[l];

    hypre_SetIndex(cindex, 0);
    hypre_SetIndex(findex, 0); hypre_IndexD(findex,cdir)=1;
    hypre_SetIndex(stride, 1); hypre_IndexD(stride,cdir)=2;

    bamg_dbgmsg("SetupInterpOp l=%d cdir=%d\n", l, cdir);

    /* set up interpolation operator */
    hypre_BAMGSetupInterpOp(A_l[l], cdir, findex, stride, P_l[l]);

    /* set up the restriction operator */
#if 0
    /* Allow R != PT for non symmetric case */
    if (!hypre_StructMatrixSymmetric(A))
      hypre_BAMGSetupRestrictOp(A_l[l], tx_l[l],
          cdir, cindex, stride, RT_l[l]);
#endif

    bamg_dbgmsg("SetupRAPOp\n");

    /* set up the coarse grid operator */
    hypre_BAMGSetupRAPOp(RT_l[l], A_l[l], P_l[l],
        cdir, cindex, stride, A_l[l+1]);

    /* set up the interpolation routine */
    interp_data_l[l] = hypre_SemiInterpCreate();
    hypre_SemiInterpSetup(interp_data_l[l], P_l[l], 0, x_l[l+1], e_l[l],
        cindex, findex, stride);

    /* set up the restriction routine */
    restrict_data_l[l] = hypre_SemiRestrictCreate();
    hypre_SemiRestrictSetup(restrict_data_l[l], RT_l[l], 1, r_l[l], b_l[l+1],
        cindex, findex, stride);
  }

  /*-----------------------------------------------------
   * Check for zero diagonal on coarsest grid, occurs with
   * singular problems like full Neumann or full periodic.
   * Note that a processor with zero diagonal will set
   * active_l =0, other processors will not. This is OK
   * as we only want to avoid the division by zero on the
   * one processor which owns the single coarse grid
   * point.
   *-----------------------------------------------------*/

  if ( hypre_ZeroDiagonal(A_l[l]))
  {
    bamg_dbgmsg("ZeroDiagonal ...\n");
    active_l[l] = 0;
  }

  /* set up fine grid relaxation */
  bamg_dbgmsg("Set Jacobi weights et al.\n");

  relax_data_l[0] = hypre_BAMGRelaxCreate(comm);
  hypre_BAMGRelaxSetTol(relax_data_l[0], 0.0);
    
  hypre_BAMGRelaxSetJacobiWeight(relax_data_l[0], jacobi_weight);
  hypre_BAMGRelaxSetType(relax_data_l[0], relax_type);
  hypre_BAMGRelaxSetTempVec(relax_data_l[0], tx_l[0]);
  hypre_BAMGRelaxSetup(relax_data_l[0], A_l[0], b_l[0], x_l[0]);

  if (num_levels > 1)
  {
    for (l = 1; l < num_levels; l++)
    {
      /* set relaxation parameters */
      if (active_l[l])
      {
        relax_data_l[l] = hypre_BAMGRelaxCreate(comm);
        hypre_BAMGRelaxSetTol(relax_data_l[l], 0.0);
        hypre_BAMGRelaxSetJacobiWeight(relax_data_l[l], jacobi_weight);
        hypre_BAMGRelaxSetType(relax_data_l[l], relax_type);
        hypre_BAMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      }
    }

    /* change coarsest grid relaxation parameters */
    l = num_levels - 1;
    if (active_l[l])
    {
      HYPRE_Int maxwork, maxiter;
      hypre_BAMGRelaxSetType(relax_data_l[l], 0);
      /* do no more work on the coarsest grid than the cost of a V-cycle
       * (estimating roughly 4 communications per V-cycle level) */
      maxwork = 4*num_levels;
      /* do sweeps proportional to the coarsest grid size */
      maxiter = hypre_min(maxwork, cmaxsize);
#if 0
      hypre_printf("maxwork = %d, cmaxsize = %d, maxiter = %d\n",
          maxwork, cmaxsize, maxiter);
#endif
      hypre_BAMGRelaxSetMaxIter(relax_data_l[l], maxiter);
    }

    /* call relax setup */
    for (l = 1; l < num_levels; l++)
    {
      if (active_l[l])
      {
        hypre_BAMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
      }
    }
  }

  for (l = 0; l < num_levels; l++)
  {
    /* set up the residual routine */
    matvec_data_l[l] = hypre_StructMatvecCreate();
    hypre_StructMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
  }

  (bamg_data -> active_l)        = active_l;
  (bamg_data -> relax_data_l)    = relax_data_l;
  (bamg_data -> matvec_data_l)   = matvec_data_l;
  (bamg_data -> restrict_data_l) = restrict_data_l;
  (bamg_data -> interp_data_l)   = interp_data_l;

  /*-----------------------------------------------------
   * Allocate space for log info
   *-----------------------------------------------------*/

  if ((bamg_data -> logging) > 0)
  {
    max_iter = (bamg_data -> max_iter);
    (bamg_data -> norms)     = hypre_TAlloc(HYPRE_Real, max_iter);
    (bamg_data -> rel_norms) = hypre_TAlloc(HYPRE_Real, max_iter);
  }

#if DEBUG_BAMG
  for (l = 0; l < (num_levels - 1); l++)
  {
    hypre_sprintf(filename, "zout_A.%02d", l);
    hypre_StructMatrixPrint(filename, A_l[l], 0);
    hypre_sprintf(filename, "zout_P.%02d", l);
    hypre_StructMatrixPrint(filename, P_l[l], 0);
  }
  hypre_sprintf(filename, "zout_A.%02d", l);
  hypre_StructMatrixPrint(filename, A_l[l], 0);
#endif

  bamg_dbgmsg("BAMGSetup finished.\n");

  return hypre_error_flag;
}

