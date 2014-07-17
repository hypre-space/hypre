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

#include "_hypre_sstruct_ls.h"
#include "sys_bamg.h"

#define hypre_BAMGSetCIndex(cdir, cindex)       \
{                                            \
  hypre_SetIndex3(cindex, 0, 0, 0);          \
  hypre_IndexD(cindex, cdir) = 0;           \
}

#define hypre_BAMGSetFIndex(cdir, findex)       \
{                                            \
  hypre_SetIndex3(findex, 0, 0, 0);          \
  hypre_IndexD(findex, cdir) = 1;           \
}

#define hypre_BAMGSetStride(cdir, stride)       \
{                                            \
  hypre_SetIndex3(stride, 1, 1, 1);          \
  hypre_IndexD(stride, cdir) = 2;           \
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetup( void                 *sys_bamg_vdata,
    hypre_SStructMatrix  *A_in,
    hypre_SStructVector  *b_in,
    hypre_SStructVector  *x_in        )
{
  hypre_SysBAMGData      *sys_bamg_data = sys_bamg_vdata;

  MPI_Comm                comm = (sys_bamg_data -> comm);

  hypre_SStructPMatrix   *A;
  hypre_SStructPVector   *b;
  hypre_SStructPVector   *x;

  HYPRE_Int               relax_type = (sys_bamg_data -> relax_type);
  HYPRE_Int               usr_jacobi_weight= (sys_bamg_data -> usr_jacobi_weight);
  HYPRE_Real              jacobi_weight    = (sys_bamg_data -> jacobi_weight);
  HYPRE_Int               skip_relax = (sys_bamg_data -> skip_relax);
  HYPRE_Real             *dxyz       = (sys_bamg_data -> dxyz);

  HYPRE_Int               num_tv1 = (sys_bamg_data -> num_tv1);
  HYPRE_Int               num_tv2 = (sys_bamg_data -> num_tv2);
  HYPRE_Int               num_tv = num_tv1 + num_tv2;
  HYPRE_Int               num_tv_relax = (sys_bamg_data -> num_tv_relax);
  void                   *tv_relax;

  HYPRE_Int               max_iter;
  HYPRE_Int               max_levels;

  HYPRE_Int               num_levels;

  hypre_Index             cindex;
  hypre_Index             findex;
  hypre_Index             stride;

  hypre_Index             coarsen;

  HYPRE_Int              *cdir_l;
  HYPRE_Int              *active_l;
  hypre_SStructPGrid    **grid_l;
  hypre_SStructPGrid    **P_grid_l;

  hypre_SStructPMatrix  **A_l;
  hypre_SStructPMatrix  **P_l;
  hypre_SStructPMatrix  **RT_l;
  hypre_SStructPVector  **b_l;
  hypre_SStructPVector  **x_l;

  /* temp vectors */
  hypre_SStructPVector  **tx_l;
  hypre_SStructPVector  **r_l;
  hypre_SStructPVector  **e_l;

  void                  **relax_data_l;
  void                  **matvec_data_l;
  void                  **restrict_data_l;
  void                  **interp_data_l;

  hypre_SStructPGrid     *grid;
  hypre_StructGrid       *sgrid;
  HYPRE_Int               dim;
  HYPRE_Int               full_periodic;

  hypre_Box              *cbox;

  HYPRE_Real             *relax_weights;
  HYPRE_Real             *mean, *deviation;
  HYPRE_Real              alpha, beta;
  HYPRE_Int               dxyz_flag;

  HYPRE_Real              min_dxyz;
  HYPRE_Int               cdir, periodic, cmaxsize;
  HYPRE_Int               d, l;
  HYPRE_Int               i, j, k;

  HYPRE_Real**            sys_dxyz;

  HYPRE_Int               nvars;

  hypre_SStructPVector ***tv;     // tv[l][k] == k'th test vector on level l

#if DEBUG_SYSBAMG
  char                    filename[255];
#endif

  /*-----------------------------------------------------
   * Refs to A,x,b (the PMatrix & PVectors within
   * the input SStructMatrix & SStructVectors)
   *-----------------------------------------------------*/
  hypre_SStructPMatrixRef(hypre_SStructMatrixPMatrix(A_in, 0), &A);
  hypre_SStructPVectorRef(hypre_SStructVectorPVector(b_in, 0), &b);
  hypre_SStructPVectorRef(hypre_SStructVectorPVector(x_in, 0), &x);

  /*--------------------------------------------------------
   * Allocate arrays for mesh sizes for each diagonal block
   *--------------------------------------------------------*/
  nvars    = hypre_SStructPMatrixNVars(A);
  sys_dxyz = hypre_TAlloc(HYPRE_Real *, nvars);
  for ( i = 0; i < nvars; i++)
  {
    sys_dxyz[i] = hypre_TAlloc(HYPRE_Real, 3);
  }

  /*-----------------------------------------------------
   * Set up coarse grids
   *-----------------------------------------------------*/

  grid  = hypre_SStructPMatrixPGrid(A);
  sgrid = hypre_SStructPGridSGrid(grid, 0);
  dim   = hypre_StructGridNDim(sgrid);

  /* Compute a new max_levels value based on the grid */
  cbox = hypre_BoxDuplicate(hypre_StructGridBoundingBox(sgrid));
  max_levels =
    hypre_Log2(hypre_BoxSizeD(cbox, 0)) + 2 +
    hypre_Log2(hypre_BoxSizeD(cbox, 1)) + 2 +
    hypre_Log2(hypre_BoxSizeD(cbox, 2)) + 2;
  if ((sys_bamg_data -> max_levels) > 0)
  {
    max_levels = hypre_min(max_levels, (sys_bamg_data -> max_levels));
  }
  (sys_bamg_data -> max_levels) = max_levels;

  /* compute dxyz */
  dxyz_flag= 0;
  if ((dxyz[0] == 0) || (dxyz[1] == 0) || (dxyz[2] == 0))
  {
    mean = hypre_CTAlloc(HYPRE_Real, 3);
    deviation = hypre_CTAlloc(HYPRE_Real, 3);

    dxyz_flag = 0;
    for (i = 0; i < nvars; i++)
    {
      hypre_PFMGComputeDxyz(hypre_SStructPMatrixSMatrix(A,i,i), sys_dxyz[i],
          mean, deviation);

      /* signal flag if any of the flag has a large (square) coeff. of
       * variation */
      if (!dxyz_flag)
      {
        for (d = 0; d < dim; d++)
        {
          deviation[d] -= mean[d]*mean[d];
          /* square of coeff. of variation */
          if (deviation[d]/(mean[d]*mean[d]) > .1)
          {
            dxyz_flag = 1;
            break;
          }
        }
      }

      for (d = 0; d < 3; d++)
      {
        dxyz[d] += sys_dxyz[i][d];
      }
    }
    hypre_TFree(mean);
    hypre_TFree(deviation);
  }

  grid_l = hypre_TAlloc(hypre_SStructPGrid *, max_levels);
  grid_l[0] = grid;
  P_grid_l = hypre_TAlloc(hypre_SStructPGrid *, max_levels);
  P_grid_l[0] = NULL;
  cdir_l = hypre_TAlloc(HYPRE_Int, max_levels);
  active_l = hypre_TAlloc(HYPRE_Int, max_levels);
  relax_weights = hypre_CTAlloc(HYPRE_Real, max_levels);
  hypre_SetIndex3(coarsen, 1, 1, 1); /* forces relaxation on finest grid */
  for (l = 0; ; l++)
  {
    /* determine cdir */
    min_dxyz = dxyz[0] + dxyz[1] + dxyz[2] + 1;
    cdir = -1;
    alpha = 0.0;
    for (d = 0; d < dim; d++)
    {
      if ((hypre_BoxIMaxD(cbox, d) > hypre_BoxIMinD(cbox, d)) &&
          (dxyz[d] < min_dxyz))
      {
        min_dxyz = dxyz[d];
        cdir = d;
      }
      alpha += 1.0/(dxyz[d]*dxyz[d]);
    }
    relax_weights[l] = 2.0/3.0;

    /* If it's possible to coarsen, change relax_weights */
    beta = 0.0;
    if (cdir != -1)
    {
      if (dxyz_flag)
      {
        relax_weights[l] = 2.0/3.0;
      }

      else
      {
        for (d = 0; d < dim; d++)
        {
          if (d != cdir)
          {
            beta += 1.0/(dxyz[d]*dxyz[d]);
          }
        }
        if (beta == alpha)
        {
          alpha = 0.0;
        }
        else
        {
          alpha = beta/alpha;
        }

        /* determine level Jacobi weights */
        if (dim > 1)
        {
          relax_weights[l] = 2.0/(3.0 - alpha);
        }
        else
        {
          relax_weights[l] = 2.0/3.0; /* always 2/3 for 1-d */
        }
      }
    }

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
      for (d = 0; d < dim; d++)
      {
        cmaxsize = hypre_max(cmaxsize, hypre_BoxSizeD(cbox, d));
      }

      break;
    }

    cdir_l[l] = cdir;

    if (hypre_IndexD(coarsen, cdir) != 0)
    {
      /* coarsened previously in this direction, relax level l */
      active_l[l] = 1;
      hypre_SetIndex3(coarsen, 0, 0, 0);
      hypre_IndexD(coarsen, cdir) = 1;
    }
    else
    {
      active_l[l] = 0;
      hypre_IndexD(coarsen, cdir) = 1;
    }

    /* set cindex, findex, and stride */
    hypre_BAMGSetCIndex(cdir, cindex);
    hypre_BAMGSetFIndex(cdir, findex);
    hypre_BAMGSetStride(cdir, stride);

    /* update dxyz and coarsen cbox*/
    dxyz[cdir] *= 2;
    hypre_ProjectBox(cbox, cindex, stride);
    hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride,
        hypre_BoxIMin(cbox));
    hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride,
        hypre_BoxIMax(cbox));

    /* build the interpolation grid */
    hypre_SysBAMGCoarsen(grid_l[l], findex, stride, 0, &P_grid_l[l+1]);

    /* build the coarse grid */
    hypre_SysBAMGCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l+1]);
  }
  num_levels = l + 1;

  /*-----------------------------------------------------
   * For fully periodic problems, the coarsest grid
   * problem (a single node) can have zero diagonal
   * blocks. This causes problems with the gselim
   * routine (which doesn't do pivoting). We avoid
   * this by skipping relaxation.
   *-----------------------------------------------------*/

  full_periodic = 1;
  for (d = 0; d < dim; d++)
  {
    full_periodic *= hypre_IndexD(hypre_SStructPGridPeriodic(grid),d);
  }
  if( full_periodic != 0)
  {
    hypre_SStructPGridDestroy(grid_l[num_levels-1]);
    hypre_SStructPGridDestroy(P_grid_l[num_levels-1]);
    num_levels -= 1;
  }

  /* free up some things */
  hypre_BoxDestroy(cbox);
  for ( i = 0; i < nvars; i++)
  {
    hypre_TFree(sys_dxyz[i]);
  }
  hypre_TFree(sys_dxyz);


  /* set all levels active if skip_relax = 0 */
  if (!skip_relax)
  {
    for (l = 0; l < num_levels; l++)
    {
      active_l[l] = 1;
    }
  }

  (sys_bamg_data -> num_levels) = num_levels;
  (sys_bamg_data -> cdir_l)     = cdir_l;
  (sys_bamg_data -> active_l)   = active_l;
  (sys_bamg_data -> grid_l)     = grid_l;
  (sys_bamg_data -> P_grid_l)   = P_grid_l;

  /*-----------------------------------------------------
   * Set up matrix and vector structures
   *-----------------------------------------------------*/

  A_l  = hypre_TAlloc(hypre_SStructPMatrix *, num_levels);
  P_l  = hypre_TAlloc(hypre_SStructPMatrix *, num_levels - 1);
  RT_l = hypre_TAlloc(hypre_SStructPMatrix *, num_levels - 1);
  b_l  = hypre_TAlloc(hypre_SStructPVector *, num_levels);
  x_l  = hypre_TAlloc(hypre_SStructPVector *, num_levels);
  tx_l = hypre_TAlloc(hypre_SStructPVector *, num_levels);
  r_l  = tx_l;
  e_l  = tx_l;

  hypre_SStructPMatrixRef(A, &A_l[0]);
  hypre_SStructPVectorRef(b, &b_l[0]);
  hypre_SStructPVectorRef(x, &x_l[0]);

  hypre_SStructPVectorCreate(comm, grid_l[0], &tx_l[0]);
  hypre_SStructPVectorInitialize(tx_l[0]);

  for (l = 0; l < (num_levels - 1); l++)
  {
    cdir = cdir_l[l];

    P_l[l]  = hypre_SysBAMGCreateInterpOp(A_l[l], P_grid_l[l+1], cdir);
    hypre_SStructPMatrixInitialize(P_l[l]);

    RT_l[l] = P_l[l];

    A_l[l+1] = hypre_SysBAMGCreateRAPOp(RT_l[l], A_l[l], P_l[l],
        grid_l[l+1], cdir);
    hypre_SStructPMatrixInitialize(A_l[l+1]);

    hypre_SStructPVectorCreate(comm, grid_l[l+1], &b_l[l+1]);
    hypre_SStructPVectorInitialize(b_l[l+1]);

    hypre_SStructPVectorCreate(comm, grid_l[l+1], &x_l[l+1]);
    hypre_SStructPVectorInitialize(x_l[l+1]);

    hypre_SStructPVectorCreate(comm, grid_l[l+1], &tx_l[l+1]);
    hypre_SStructPVectorInitialize(tx_l[l+1]);
  }

  hypre_SStructPVectorAssemble(tx_l[0]);
  for (l = 0; l < (num_levels - 1); l++)
  {
    hypre_SStructPVectorAssemble(b_l[l+1]);
    hypre_SStructPVectorAssemble(x_l[l+1]);
    hypre_SStructPVectorAssemble(tx_l[l+1]);
  }

  (sys_bamg_data -> A_l)  = A_l;
  (sys_bamg_data -> P_l)  = P_l;
  (sys_bamg_data -> RT_l) = RT_l;
  (sys_bamg_data -> b_l)  = b_l;
  (sys_bamg_data -> x_l)  = x_l;
  (sys_bamg_data -> tx_l) = tx_l;
  (sys_bamg_data -> r_l)  = r_l;
  (sys_bamg_data -> e_l)  = e_l;

  /*-----------------------------------------------------
   * Set up multigrid operators and call setup routines
   *-----------------------------------------------------*/

  sysbamg_dbgmsg("Set up multigrid operators ...\n");

  relax_data_l    = hypre_TAlloc(void *, num_levels);
  matvec_data_l   = hypre_TAlloc(void *, num_levels);
  restrict_data_l = hypre_TAlloc(void *, num_levels);
  interp_data_l   = hypre_TAlloc(void *, num_levels);

  // set up the test vectors (initial + singular)

  sysbamg_dbgmsg("num_tv = %d = %d + %d\n", num_tv, num_tv1, num_tv2);
  tv = hypre_TAlloc(hypre_SStructPVector**, num_levels);
  for ( l=0; l<num_levels; l++ )
  {
    tv[l] = hypre_TAlloc(hypre_SStructPVector*, num_tv);
    for ( k = 0; k < num_tv; k++ )
    {
      hypre_SStructPVectorCreate(comm, grid_l[l], &tv[l][k]);
      hypre_SStructPVectorInitialize(tv[l][k]);
      hypre_SStructPVectorAssemble(tv[l][k]);
      
      if ( l == 0 && k < num_tv1 )
        hypre_SStructPVectorSetRandomValues(tv[l][k], (HYPRE_Int)time(0)+k);
    }
  }

  for (l = 0; l < (num_levels - 1); l++)
  {
    cdir = cdir_l[l];

    hypre_BAMGSetCIndex(cdir, cindex);
    hypre_BAMGSetFIndex(cdir, findex);
    hypre_BAMGSetStride(cdir, stride);

    // XXX smooth the test vectors XXX
    // 1) set up the rhs for smoothing, zero for now
    hypre_SStructPVector *rhs;
    hypre_SStructPVectorCreate(comm,grid_l[l],&rhs);
    hypre_SStructPVectorInitialize(rhs);
    hypre_SStructPVectorAssemble(rhs);
    hypre_SStructPVectorSetConstantValues(rhs,0.0);
    // 2) set up the relax struct
    tv_relax = hypre_SysBAMGRelaxCreate(comm);
    hypre_SysBAMGRelaxSetTol(tv_relax, 0.0);
    if (usr_jacobi_weight)
      hypre_SysBAMGRelaxSetJacobiWeight(tv_relax, jacobi_weight);
    else
      hypre_SysBAMGRelaxSetJacobiWeight(tv_relax, relax_weights[l]);
    hypre_SysBAMGRelaxSetType(tv_relax, relax_type);
    hypre_SysBAMGRelaxSetTempVec(tv_relax, tx_l[l]);
    hypre_SysBAMGRelaxSetPreRelax(tv_relax);
    hypre_SysBAMGRelaxSetMaxIter(tv_relax, num_tv_relax);
    hypre_SysBAMGRelaxSetZeroGuess(tv_relax, 0);
    hypre_SysBAMGRelaxSetup(tv_relax, A_l[l], rhs, x_l[l]);
    // 3) smooth
    for ( k = 0; k < num_tv1; k++ )
      hypre_SysBAMGRelax( tv_relax, A_l[l], rhs, tv[l][k] );
    // 4) destroy the relax struct
    hypre_SysBAMGRelaxDestroy( tv_relax );
    // 5) destroy the rhs
    hypre_SStructPVectorDestroy( rhs );

#if DEBUG_SYSBAMG
  for ( k = 0; k < num_tv1; k++ ) {
    hypre_sprintf(filename, "sysbamg_tv_l=%d,k=%d.dat", l, k);
    hypre_SStructPVectorPrint(filename, tv[l][k], 0);
  }
#endif

    /* set up interpolation operator */
    hypre_SysBAMGSetupInterpOp(A_l[l], cdir, findex, stride, P_l[l], num_tv1, tv[l]);

    /* set up the coarse grid operator */
    hypre_SysBAMGSetupRAPOp(RT_l[l], A_l[l], P_l[l],
        cdir, cindex, stride, A_l[l+1]);

    /* set up the interpolation routine */
    hypre_SysSemiInterpCreate(&interp_data_l[l]);
    hypre_SysSemiInterpSetup(interp_data_l[l], P_l[l], 0, x_l[l+1], e_l[l],
        cindex, findex, stride);

    /* set up the restriction routine */
    hypre_SysSemiRestrictCreate(&restrict_data_l[l]);
    hypre_SysSemiRestrictSetup(restrict_data_l[l], RT_l[l], 1, r_l[l], b_l[l+1],
        cindex, findex, stride);
    
    // restrict the tv[l] to tv[l+1] (NB: don't need tv's on the coarsest grid)
    if ( l < num_levels-2 )
    {
      for ( k = 0; k < num_tv1; k++ )
        hypre_SysSemiRestrict(restrict_data_l[l], RT_l[l], tv[l][k], tv[l+1][k]);
    }
  }

  /* set up fine grid relaxation */
  relax_data_l[0] = hypre_SysBAMGRelaxCreate(comm);
  hypre_SysBAMGRelaxSetTol(relax_data_l[0], 0.0);
  if (usr_jacobi_weight)
  {
    hypre_SysBAMGRelaxSetJacobiWeight(relax_data_l[0], jacobi_weight);
  }
  else
  {
    hypre_SysBAMGRelaxSetJacobiWeight(relax_data_l[0], relax_weights[0]);
  }
  hypre_SysBAMGRelaxSetType(relax_data_l[0], relax_type);
  hypre_SysBAMGRelaxSetTempVec(relax_data_l[0], tx_l[0]);
  hypre_SysBAMGRelaxSetup(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
  if (num_levels > 1)
  {
    for (l = 1; l < num_levels; l++)
    {
      /* set relaxation parameters */
      relax_data_l[l] = hypre_SysBAMGRelaxCreate(comm);
      hypre_SysBAMGRelaxSetTol(relax_data_l[l], 0.0);
      if (usr_jacobi_weight)
      {
        hypre_SysBAMGRelaxSetJacobiWeight(relax_data_l[l], jacobi_weight);
      }
      else
      {
        hypre_SysBAMGRelaxSetJacobiWeight(relax_data_l[l], relax_weights[l]);
      }
      hypre_SysBAMGRelaxSetType(relax_data_l[l], relax_type);
      hypre_SysBAMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
    }

    /* change coarsest grid relaxation parameters */
    l = num_levels - 1;
    {
      HYPRE_Int maxwork, maxiter;
      hypre_SysBAMGRelaxSetType(relax_data_l[l], 0);
      /* do no more work on the coarsest grid than the cost of a V-cycle
       * (estimating roughly 4 communications per V-cycle level) */
      maxwork = 4*num_levels;
      /* do sweeps proportional to the coarsest grid size */
      maxiter = hypre_min(maxwork, cmaxsize);
#if 0
      hypre_printf("maxwork = %d, cmaxsize = %d, maxiter = %d\n",
          maxwork, cmaxsize, maxiter);
#endif
      hypre_SysBAMGRelaxSetMaxIter(relax_data_l[l], maxiter);
    }

    /* call relax setup */
    for (l = 1; l < num_levels; l++)
    {
      hypre_SysBAMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
    }
  }
  hypre_TFree(relax_weights);

  for (l = 0; l < num_levels; l++)
  {
    /* set up the residual routine */
    hypre_SStructPMatvecCreate(&matvec_data_l[l]);
    hypre_SStructPMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
  }

  (sys_bamg_data -> relax_data_l)    = relax_data_l;
  (sys_bamg_data -> matvec_data_l)   = matvec_data_l;
  (sys_bamg_data -> restrict_data_l) = restrict_data_l;
  (sys_bamg_data -> interp_data_l)   = interp_data_l;

  /*-----------------------------------------------------
   * Allocate space for log info
   *-----------------------------------------------------*/

  if ((sys_bamg_data -> logging) > 0)
  {
    max_iter = (sys_bamg_data -> max_iter);
    (sys_bamg_data -> norms)     = hypre_TAlloc(HYPRE_Real, max_iter);
    (sys_bamg_data -> rel_norms) = hypre_TAlloc(HYPRE_Real, max_iter);
  }

#if DEBUG_SYSBAMG
  for (l = 0; l < (num_levels - 1); l++)
  {
    hypre_sprintf(filename, "sysbamg_A.%02d", l);
    hypre_SStructPMatrixPrint(filename, A_l[l], 0);
    hypre_sprintf(filename, "sysbamg_P.%02d", l);
    hypre_SStructPMatrixPrint(filename, P_l[l], 0);
  }
  hypre_sprintf(filename, "sysbamg_A.%02d", l);
  hypre_SStructPMatrixPrint(filename, A_l[l], 0);
#endif

  /*-----------------------------------------------------
   * Destroy Refs to A,x,b (the PMatrix & PVectors within
   * the input SStructMatrix & SStructVectors).
   *-----------------------------------------------------*/
  hypre_SStructPMatrixDestroy(A);
  hypre_SStructPVectorDestroy(x);
  hypre_SStructPVectorDestroy(b);

  for ( l = 0; l < num_levels; l++ )
  {
    for ( k = 0; k < num_tv; k++ )
    {
      hypre_SStructPVectorDestroy(tv[l][k]);
    }
    hypre_TFree(tv[l]);
  }
  hypre_TFree(tv);

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGCoarsen( hypre_SStructPGrid  *fgrid,
    hypre_Index          index,
    hypre_Index          stride,
    HYPRE_Int            prune,
    hypre_SStructPGrid **cgrid_ptr )
{
  hypre_SStructPGrid   *cgrid;

  hypre_StructGrid     *sfgrid;
  hypre_StructGrid     *scgrid;

  MPI_Comm               comm;
  HYPRE_Int              ndim;
  HYPRE_Int              nvars;
  hypre_SStructVariable *vartypes;
  hypre_SStructVariable *new_vartypes;
  HYPRE_Int              i;
  HYPRE_Int              t;

  /*-----------------------------------------
   * Copy information from fine grid
   *-----------------------------------------*/

  comm      = hypre_SStructPGridComm(fgrid);
  ndim      = hypre_SStructPGridNDim(fgrid);
  nvars     = hypre_SStructPGridNVars(fgrid);
  vartypes  = hypre_SStructPGridVarTypes(fgrid);

  cgrid = hypre_TAlloc(hypre_SStructPGrid, 1);

  hypre_SStructPGridComm(cgrid)     = comm;
  hypre_SStructPGridNDim(cgrid)     = ndim;
  hypre_SStructPGridNVars(cgrid)    = nvars;
  new_vartypes = hypre_TAlloc(hypre_SStructVariable, nvars);
  for (i = 0; i < nvars; i++)
  {
    new_vartypes[i] = vartypes[i];
  }
  hypre_SStructPGridVarTypes(cgrid) = new_vartypes;

  for (t = 0; t < 8; t++)
  {
    hypre_SStructPGridVTSGrid(cgrid, t)     = NULL;
    hypre_SStructPGridVTIBoxArray(cgrid, t) = NULL;
  }

  /*-----------------------------------------
   * Set the coarse sgrid
   *-----------------------------------------*/

  sfgrid = hypre_SStructPGridCellSGrid(fgrid);
  hypre_StructCoarsen(sfgrid, index, stride, prune, &scgrid);

  hypre_CopyIndex(hypre_StructGridPeriodic(scgrid),
      hypre_SStructPGridPeriodic(cgrid));

  hypre_SStructPGridSetCellSGrid(cgrid, scgrid);

  hypre_SStructPGridPNeighbors(cgrid) = hypre_BoxArrayCreate(0, ndim);
  hypre_SStructPGridPNborOffsets(cgrid) = NULL;

  hypre_SStructPGridLocalSize(cgrid)  = 0;
  hypre_SStructPGridGlobalSize(cgrid) = 0;
  hypre_SStructPGridGhlocalSize(cgrid)= 0;

  hypre_SStructPGridAssemble(cgrid);

  *cgrid_ptr = cgrid;

  return hypre_error_flag;
}

