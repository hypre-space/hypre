/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.27 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Structured matrix-transpose-vector multiply routine
 *  NB: Hermitian conjugate if HYPRE_COMPLEX !
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/* this currently cannot be greater than 7 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 1

#define LINE printf("%s %d\n", __FILE__, __LINE__);

/*--------------------------------------------------------------------------
 * hypre_StructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
  hypre_StructMatrix  *A;
  hypre_StructVector  *x;
  hypre_ComputePkg    *compute_pkg;

} hypre_StructMatvecData;

/*--------------------------------------------------------------------------
 * hypre_StructMatvecTSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecTSetup( void               *matvec_vdata,
    hypre_StructMatrix *A,
    hypre_StructVector *x            )
{
  hypre_StructMatvecData  *matvec_data = matvec_vdata;

  hypre_StructGrid        *grid;
  hypre_StructStencil     *stencil;
  hypre_ComputeInfo       *compute_info;
  hypre_ComputePkg        *compute_pkg;

  /*----------------------------------------------------------
   * Set up the compute package
   *----------------------------------------------------------*/

  grid    = hypre_StructMatrixGrid(A);
  stencil = hypre_StructMatrixStencil(A);

  HYPRE_Int reverse = 1; // reverse > 0 ==> send ("push") ghost cells, rather than recv

  hypre_CreateComputeInfo(grid, stencil, &compute_info);
  hypre_ComputePkgCreate2(compute_info, hypre_StructVectorDataSpace(x), 1,
      reverse, grid, &compute_pkg);

  /*----------------------------------------------------------
   * Set up the matvec data structure
   *----------------------------------------------------------*/

  (matvec_data -> A)           = hypre_StructMatrixRef(A);
  (matvec_data -> x)           = hypre_StructVectorRef(x);
  (matvec_data -> compute_pkg) = compute_pkg;

  return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_StructMatvecCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructMatvecTCompute( 
    void               *matvec_vdata,
    HYPRE_Complex       alpha,
    hypre_StructMatrix *A,
    hypre_StructVector *x,
    HYPRE_Complex       beta,
    hypre_StructVector *y )
{
  hypre_StructMatvecData  *matvec_data = matvec_vdata;

  hypre_ComputePkg        *compute_pkg;

  hypre_CommHandle        *comm_handle;

  hypre_BoxArrayArray     *compute_box_aa;
  hypre_Box               *y_data_box;

  HYPRE_Int                yi;

  HYPRE_Complex           *xp;
  HYPRE_Complex           *yp;

  hypre_BoxArray          *boxes;
  hypre_Box               *box;
  hypre_Index              loop_size;
  hypre_IndexRef           start;
  hypre_IndexRef           stride;

  HYPRE_Int                constant_coefficient;

  HYPRE_Complex            temp;
  HYPRE_Int                compute_i, i;

  /*-----------------------------------------------------------------------
   * Initialize some things
   *-----------------------------------------------------------------------*/

  constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

  if (constant_coefficient) {
    hypre_error_w_msg(HYPRE_ERROR_GENERIC, "constant_coefficient must be 1 for MatvecT.");
    return hypre_error_flag;
  }

  if (constant_coefficient) hypre_StructVectorClearBoundGhostValues(x, 0);

  compute_pkg = (matvec_data -> compute_pkg);

  stride = hypre_ComputePkgStride(compute_pkg);

  /*-----------------------------------------------------------------------
   * Do (alpha == 0.0) computation
   *-----------------------------------------------------------------------*/

  if (alpha == 0.0)
  {
    boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
    hypre_ForBoxI(i, boxes)
    {
      box   = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_BoxGetSize(box, loop_size);

      hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
          y_data_box, start, stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop1For(yi)
      {
        yp[yi] *= beta;
      }
      hypre_BoxLoop1End(yi);
    }

    return hypre_error_flag;
  }

  /*-----------------------------------------------------------------------
   * Do (alpha != 0.0) computation
   *
   * For Tranpose(A), read local x and write local and non-local y.
   * So zero y ghost cells,
   *    compute y Dept region
   *    start communicating y in reverse ("push"), "+=" mode
   *    compute y Indt region
   *    finish communicating y
   *
   *-----------------------------------------------------------------------*/

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch(compute_i)
    {
      case 0:
        {
          /*--------------------------------------------------------------
           * initialize y = (beta/alpha)*y normally (and mult. by alpha at end)
           *                 beta*y for constant coeff (and mult only Ax by alpha)
           *--------------------------------------------------------------*/
           
          if ( constant_coefficient==1 )
            temp = beta;
          else
            temp = beta / alpha;

          if (temp != 1.0)
          {
            boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
            hypre_ForBoxI(i, boxes)
            {
              box   = hypre_BoxArrayBox(boxes, i);
              start = hypre_BoxIMin(box);

              y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
              yp = hypre_StructVectorBoxData(y, i);

              if (temp == 0.0)
              {
                hypre_BoxGetSize(box, loop_size);

                hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                    y_data_box, start, stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
                hypre_BoxLoop1For(yi)
                {
                  yp[yi] = 0.0;
                }
                hypre_BoxLoop1End(yi);
              }
              else
              {
                hypre_BoxGetSize(box, loop_size);

                hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                    y_data_box, start, stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
                hypre_BoxLoop1For(yi)
                {
                  yp[yi] *= temp;
                }
                hypre_BoxLoop1End(yi);
              }
            }
          }

          // Init boundary ghost cells to zero (every time)
          hypre_StructVectorClearGhostValues(y);

          // compute ghost and ghost-connected y cells
          compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
        }
        break;

      case 1:
        {
          // start to push/append y ghost cells (computed for compute_i = 0)
          yp = hypre_StructVectorData(y);
          HYPRE_Int action = 1;
          hypre_InitializeIndtComputations2(compute_pkg, yp, &comm_handle, action);

          // y regions not connected to ghost cells
          compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
        }
        break;
    }

    /*--------------------------------------------------------------------
     * y += Transpose(A)*x in regions defined in 'compute_box_aa'
     *--------------------------------------------------------------------*/

    switch( constant_coefficient )
    {
      case 0:
        hypre_StructMatvecTCC0( alpha, A, x, y, compute_box_aa, stride );
        break;
    }
  }

  // Finalize communication - y push/append
  hypre_FinalizeIndtComputations(comm_handle);

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecTCC0
 * core of struct matvec computation, for the case constant_coefficient==0
 * (all coefficients are variable)
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructMatvecTCC0(
    HYPRE_Complex        alpha,
    hypre_StructMatrix  *A,
    hypre_StructVector  *x,
    hypre_StructVector  *y,
    hypre_BoxArrayArray *compute_box_aa,
    hypre_IndexRef       stride )
{
  HYPRE_Int i, j, si;
  HYPRE_Complex           *Ap0;
  HYPRE_Complex           *Ap1;
  HYPRE_Complex           *Ap2;
  HYPRE_Complex           *Ap3;
  HYPRE_Complex           *Ap4;
  HYPRE_Complex           *Ap5;
  HYPRE_Complex           *Ap6;
  HYPRE_Int                xoff0;
  HYPRE_Int                xoff1;
  HYPRE_Int                xoff2;
  HYPRE_Int                xoff3;
  HYPRE_Int                xoff4;
  HYPRE_Int                xoff5;
  HYPRE_Int                xoff6;
  HYPRE_Int                Ai;
  HYPRE_Int                xi;
  hypre_BoxArray          *compute_box_a;
  hypre_Box               *compute_box;

  hypre_Box               *A_data_box;
  hypre_Box               *x_data_box;
  hypre_StructStencil     *stencil;
  hypre_Index             *stencil_shape;
  HYPRE_Int                stencil_size;

  hypre_Box               *y_data_box;
  HYPRE_Complex           *xp;
  HYPRE_Complex           *yp;
  HYPRE_Int                depth;
  hypre_Index              loop_size;
  hypre_IndexRef           start;
  HYPRE_Int                yi;
  HYPRE_Int                ndim;

  HYPRE_Int                rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // XXX Not correct in general

  stencil       = hypre_StructMatrixStencil(A);
  stencil_shape = hypre_StructStencilShape(stencil);
  stencil_size  = hypre_StructStencilSize(stencil);
  ndim          = hypre_StructVectorNDim(x);

  hypre_ForBoxArrayI(i, compute_box_aa)
  {
    compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

    A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
    x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
    y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

    xp = hypre_StructVectorBoxData(x, i);
    yp = hypre_StructVectorBoxData(y, i);

    hypre_ForBoxI(j, compute_box_a)
    {
      compute_box = hypre_BoxArrayBox(compute_box_a, j);

      hypre_BoxGetSize(compute_box, loop_size);
      start  = hypre_BoxIMin(compute_box);

      /* unroll up to depth MAX_DEPTH */
      for (si = 0; si < stencil_size; si += MAX_DEPTH)
      {
        depth = hypre_min(MAX_DEPTH, (stencil_size-si));
        switch(depth)
        {
          case 1:
            Ap0 = hypre_StructMatrixBoxData(A, i, si+0);

            xoff0 = hypre_BoxOffsetDistance(x_data_box,
                stencil_shape[si+0]);

            hypre_BoxLoop3Begin(ndim, loop_size,
                A_data_box, start, stride, Ai,
                x_data_box, start, stride, xi,
                y_data_box, start, stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi,xi,Ai) HYPRE_SMP_SCHEDULE
#endif
            hypre_BoxLoop3For(Ai, xi, yi)
            {
              // transpose here by swapping x and y
              HYPRE_Int y_index = xi + xoff0;
              HYPRE_Int x_index = yi;
              yp[y_index] += hypre_conj( Ap0[Ai] ) * xp[x_index];
            }
            hypre_BoxLoop3End(Ai, xi, yi);

            break;
        }
      }

      if (alpha != 1.0)
      {
        hypre_BoxLoop1Begin(ndim, loop_size,
            y_data_box, start, stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
        hypre_BoxLoop1For(yi)
        {
          yp[yi] *= alpha;
        }
        hypre_BoxLoop1End(yi);
      }
    }
  }

  return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_StructMatvecT
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructMatvecT(
    HYPRE_Complex       alpha,
    hypre_StructMatrix *A,
    hypre_StructVector *x,
    HYPRE_Complex       beta,
    hypre_StructVector *y )
{
  void *matvec_data;

  matvec_data = hypre_StructMatvecCreate();
  hypre_StructMatvecTSetup(matvec_data, A, x);
  hypre_StructMatvecTCompute(matvec_data, alpha, A, x, beta, y);
  hypre_StructMatvecDestroy(matvec_data);

  return hypre_error_flag;
}

