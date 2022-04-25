/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/* this currently cannot be greater than 7 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   HYPRE_Real              tol;       /* tolerance, set =0 for no convergence testing */
   HYPRE_Real              rresnorm;  /* relative residual norm, computed only if tol>0.0 */
   HYPRE_Int               max_iter;
   HYPRE_Int               rel_change;         /* not yet used */
   HYPRE_Int               zero_guess;
   HYPRE_Real              weight;

   HYPRE_Int               num_pointsets;
   HYPRE_Int              *pointset_sizes;
   HYPRE_Int              *pointset_ranks;
   hypre_Index            *pointset_strides;
   hypre_Index           **pointset_indices;

   hypre_StructMatrix     *A;
   hypre_StructVector     *b;
   hypre_StructVector     *x;
   hypre_StructVector     *t;

   HYPRE_Int               diag_rank;

   hypre_ComputePkg      **compute_pkgs;

   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;
   HYPRE_BigInt            flops;

} hypre_PointRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_PointRelaxCreate( MPI_Comm  comm )
{
   hypre_PointRelaxData *relax_data;

   hypre_Index           stride;
   hypre_Index           indices[1];

   relax_data = hypre_CTAlloc(hypre_PointRelaxData,  1, HYPRE_MEMORY_HOST);

   (relax_data -> comm)       = comm;
   (relax_data -> time_index) = hypre_InitializeTiming("PointRelax");

   /* set defaults */
   (relax_data -> tol)              = 0.0;  /* tol=0 means no convergence testing */
   (relax_data -> rresnorm)         = 0.0;
   (relax_data -> max_iter)         = 1000;
   (relax_data -> rel_change)       = 0;
   (relax_data -> zero_guess)       = 0;
   (relax_data -> weight)           = 1.0;
   (relax_data -> num_pointsets)    = 0;
   (relax_data -> pointset_sizes)   = NULL;
   (relax_data -> pointset_ranks)   = NULL;
   (relax_data -> pointset_strides) = NULL;
   (relax_data -> pointset_indices) = NULL;
   (relax_data -> A)                = NULL;
   (relax_data -> b)                = NULL;
   (relax_data -> x)                = NULL;
   (relax_data -> t)                = NULL;
   (relax_data -> compute_pkgs)     = NULL;

   hypre_SetIndex3(stride, 1, 1, 1);
   hypre_SetIndex3(indices[0], 0, 0, 0);
   hypre_PointRelaxSetNumPointsets((void *) relax_data, 1);
   hypre_PointRelaxSetPointset((void *) relax_data, 0, 1, stride, indices);

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxDestroy( void *relax_vdata )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;
   HYPRE_Int             i;

   if (relax_data)
   {
      for (i = 0; i < (relax_data -> num_pointsets); i++)
      {
         hypre_TFree(relax_data -> pointset_indices[i], HYPRE_MEMORY_HOST);
      }
      if (relax_data -> compute_pkgs)
      {
         for (i = 0; i < (relax_data -> num_pointsets); i++)
         {
            hypre_ComputePkgDestroy(relax_data -> compute_pkgs[i]);
         }
      }
      hypre_TFree(relax_data -> pointset_sizes, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> pointset_ranks, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> pointset_strides, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> pointset_indices, HYPRE_MEMORY_HOST);
      hypre_StructMatrixDestroy(relax_data -> A);
      hypre_StructVectorDestroy(relax_data -> b);
      hypre_StructVectorDestroy(relax_data -> x);
      hypre_StructVectorDestroy(relax_data -> t);
      hypre_TFree(relax_data -> compute_pkgs, HYPRE_MEMORY_HOST);

      hypre_FinalizeTiming(relax_data -> time_index);
      hypre_TFree(relax_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetup( void               *relax_vdata,
                       hypre_StructMatrix *A,
                       hypre_StructVector *b,
                       hypre_StructVector *x           )
{
   hypre_PointRelaxData  *relax_data = (hypre_PointRelaxData *)relax_vdata;

   HYPRE_Int              num_pointsets    = (relax_data -> num_pointsets);
   HYPRE_Int             *pointset_sizes   = (relax_data -> pointset_sizes);
   hypre_Index           *pointset_strides = (relax_data -> pointset_strides);
   hypre_Index          **pointset_indices = (relax_data -> pointset_indices);
   HYPRE_Int              ndim = hypre_StructMatrixNDim(A);
   hypre_StructVector    *t;
   HYPRE_Int              diag_rank;
   hypre_ComputeInfo     *compute_info;
   hypre_ComputePkg     **compute_pkgs;

   hypre_Index            diag_index;
   hypre_IndexRef         stride;
   hypre_IndexRef         index;

   hypre_StructGrid      *grid;
   hypre_StructStencil   *stencil;

   hypre_BoxArrayArray   *orig_indt_boxes;
   hypre_BoxArrayArray   *orig_dept_boxes;
   hypre_BoxArrayArray   *box_aa;
   hypre_BoxArray        *box_a;
   hypre_Box             *box;
   HYPRE_Int              box_aa_size;
   HYPRE_Int              box_a_size;
   hypre_BoxArrayArray   *new_box_aa;
   hypre_BoxArray        *new_box_a;
   hypre_Box             *new_box;

   HYPRE_Real             scale;
   HYPRE_Int              frac;

   HYPRE_Int              i, j, k, p, m, compute_i;

   /*----------------------------------------------------------
    * Set up the temp vector
    *----------------------------------------------------------*/

   if ((relax_data -> t) == NULL)
   {
      t = hypre_StructVectorCreate(hypre_StructVectorComm(b),
                                   hypre_StructVectorGrid(b));
      hypre_StructVectorSetNumGhost(t, hypre_StructVectorNumGhost(b));
      hypre_StructVectorInitialize(t);
      hypre_StructVectorAssemble(t);
      (relax_data -> t) = t;
   }

   /*----------------------------------------------------------
    * Find the matrix diagonal
    *----------------------------------------------------------*/

   grid    = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);

   hypre_SetIndex3(diag_index, 0, 0, 0);
   diag_rank = hypre_StructStencilElementRank(stencil, diag_index);

   /*----------------------------------------------------------
    * Set up the compute packages
    *----------------------------------------------------------*/

   compute_pkgs = hypre_CTAlloc(hypre_ComputePkg *,  num_pointsets, HYPRE_MEMORY_HOST);

   for (p = 0; p < num_pointsets; p++)
   {
      hypre_CreateComputeInfo(grid, stencil, &compute_info);
      orig_indt_boxes = hypre_ComputeInfoIndtBoxes(compute_info);
      orig_dept_boxes = hypre_ComputeInfoDeptBoxes(compute_info);

      stride = pointset_strides[p];

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
               box_aa = orig_indt_boxes;
               break;

            case 1:
               box_aa = orig_dept_boxes;
               break;
         }
         box_aa_size = hypre_BoxArrayArraySize(box_aa);
         new_box_aa = hypre_BoxArrayArrayCreate(box_aa_size, ndim);

         for (i = 0; i < box_aa_size; i++)
         {
            box_a = hypre_BoxArrayArrayBoxArray(box_aa, i);
            box_a_size = hypre_BoxArraySize(box_a);
            new_box_a = hypre_BoxArrayArrayBoxArray(new_box_aa, i);
            hypre_BoxArraySetSize(new_box_a, box_a_size * pointset_sizes[p]);

            k = 0;
            for (m = 0; m < pointset_sizes[p]; m++)
            {
               index  = pointset_indices[p][m];

               for (j = 0; j < box_a_size; j++)
               {
                  box = hypre_BoxArrayBox(box_a, j);
                  new_box = hypre_BoxArrayBox(new_box_a, k);

                  hypre_CopyBox(box, new_box);
                  hypre_ProjectBox(new_box, index, stride);

                  k++;
               }
            }
         }

         switch (compute_i)
         {
            case 0:
               hypre_ComputeInfoIndtBoxes(compute_info) = new_box_aa;
               break;

            case 1:
               hypre_ComputeInfoDeptBoxes(compute_info) = new_box_aa;
               break;
         }
      }

      hypre_CopyIndex(stride, hypre_ComputeInfoStride(compute_info));

      hypre_ComputePkgCreate(compute_info, hypre_StructVectorDataSpace(x), 1,
                             grid, &compute_pkgs[p]);

      hypre_BoxArrayArrayDestroy(orig_indt_boxes);
      hypre_BoxArrayArrayDestroy(orig_dept_boxes);
   }

   /*----------------------------------------------------------
    * Set up the relax data structure
    *----------------------------------------------------------*/

   (relax_data -> A) = hypre_StructMatrixRef(A);
   (relax_data -> x) = hypre_StructVectorRef(x);
   (relax_data -> b) = hypre_StructVectorRef(b);
   (relax_data -> diag_rank)    = diag_rank;
   (relax_data -> compute_pkgs) = compute_pkgs;

   /*-----------------------------------------------------
    * Compute flops
    *-----------------------------------------------------*/

   scale = 0.0;
   for (p = 0; p < num_pointsets; p++)
   {
      stride = pointset_strides[p];
      frac   = hypre_IndexX(stride);
      frac  *= hypre_IndexY(stride);
      frac  *= hypre_IndexZ(stride);
      scale += (pointset_sizes[p] / frac);
   }
   (relax_data -> flops) = (HYPRE_BigInt)scale * (hypre_StructMatrixGlobalSize(A) +
                                                  hypre_StructVectorGlobalSize(x));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelax( void               *relax_vdata,
                  hypre_StructMatrix *A,
                  hypre_StructVector *b,
                  hypre_StructVector *x           )
{
   hypre_PointRelaxData  *relax_data = (hypre_PointRelaxData *)relax_vdata;

   HYPRE_Int              max_iter         = (relax_data -> max_iter);
   HYPRE_Int              zero_guess       = (relax_data -> zero_guess);
   HYPRE_Real             weight           = (relax_data -> weight);
   HYPRE_Int              num_pointsets    = (relax_data -> num_pointsets);
   HYPRE_Int             *pointset_ranks   = (relax_data -> pointset_ranks);
   hypre_Index           *pointset_strides = (relax_data -> pointset_strides);
   hypre_StructVector    *t                = (relax_data -> t);
   HYPRE_Int              diag_rank        = (relax_data -> diag_rank);
   hypre_ComputePkg     **compute_pkgs     = (relax_data -> compute_pkgs);
   HYPRE_Real             tol              = (relax_data -> tol);
   HYPRE_Real             tol2             = tol * tol;

   hypre_ComputePkg      *compute_pkg;
   hypre_CommHandle      *comm_handle;

   hypre_BoxArrayArray   *compute_box_aa;
   hypre_BoxArray        *compute_box_a;
   hypre_Box             *compute_box;

   hypre_Box             *A_data_box;
   hypre_Box             *b_data_box;
   hypre_Box             *x_data_box;
   hypre_Box             *t_data_box;

   HYPRE_Real            *Ap;
   HYPRE_Real            AAp0;
   HYPRE_Real            *bp;
   HYPRE_Real            *xp;
   HYPRE_Real            *tp;
   void                  *matvec_data = NULL;

   HYPRE_Int              Ai;

   hypre_IndexRef         stride;
   hypre_IndexRef         start;
   hypre_Index            loop_size;

   HYPRE_Int              constant_coefficient;

   HYPRE_Int              iter, p, compute_i, i, j;
   HYPRE_Int              pointset;

   HYPRE_Real             bsumsq, rsumsq;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/

   hypre_BeginTiming(relax_data -> time_index);

   hypre_StructMatrixDestroy(relax_data -> A);
   hypre_StructVectorDestroy(relax_data -> b);
   hypre_StructVectorDestroy(relax_data -> x);
   (relax_data -> A) = hypre_StructMatrixRef(A);
   (relax_data -> x) = hypre_StructVectorRef(x);
   (relax_data -> b) = hypre_StructVectorRef(b);

   (relax_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
      }

      hypre_EndTiming(relax_data -> time_index);
      return hypre_error_flag;
   }

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if (constant_coefficient) { hypre_StructVectorClearBoundGhostValues(x, 0); }

   rsumsq = 0.0;
   if ( tol > 0.0 )
   {
      bsumsq = hypre_StructInnerProd( b, b );
   }

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/

   p    = 0;
   iter = 0;
   if ( tol > 0.0)
   {
      matvec_data = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup( matvec_data, A, x );
   }

   if (zero_guess)
   {
      if ( p == 0 ) { rsumsq = 0.0; }
      if (num_pointsets > 1)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
      }
      pointset = pointset_ranks[p];
      compute_pkg = compute_pkgs[pointset];
      stride = pointset_strides[pointset];

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_data_box =
               hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
            b_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), i);
            x_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);

            Ap = hypre_StructMatrixBoxData(A, i, diag_rank);
            bp = hypre_StructVectorBoxData(b, i);
            xp = hypre_StructVectorBoxData(x, i);

            hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = hypre_BoxArrayBox(compute_box_a, j);

               start  = hypre_BoxIMin(compute_box);
               hypre_BoxGetStrideSize(compute_box, stride, loop_size);

               /* all matrix coefficients are constant */
               if ( constant_coefficient == 1 )
               {
                  Ai = hypre_CCBoxIndexRank( A_data_box, start );
                  AAp0 = 1 / Ap[Ai];
#define DEVICE_VAR is_device_ptr(xp,bp)
                  hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                                      b_data_box, start, stride, bi,
                                      x_data_box, start, stride, xi);
                  {
                     xp[xi] = bp[bi] * AAp0;
                  }
                  hypre_BoxLoop2End(bi, xi);
#undef DEVICE_VAR
               }
               /* constant_coefficent 0 (variable) or 2 (variable diagonal
                  only) are the same for the diagonal */
               else
               {
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                  hypre_BoxLoop3Begin(hypre_StructVectorNDim(x), loop_size,
                                      A_data_box, start, stride, Ai,
                                      b_data_box, start, stride, bi,
                                      x_data_box, start, stride, xi);
                  {
                     xp[xi] = bp[bi] / Ap[Ai];
                  }
                  hypre_BoxLoop3End(Ai, bi, xi);
#undef DEVICE_VAR
               }
            }
         }
      }

      if (weight != 1.0)
      {
         hypre_StructScale(weight, x);
      }

      p    = (p + 1) % num_pointsets;
      iter = iter + (p == 0);

      if ( tol > 0.0 && p == 0 )
         /* ... p==0 here means we've finished going through all the pointsets,
            i.e. this iteration is complete.
            tol>0.0 means to do a convergence test, using tol.
            The test is simply ||r||/||b||<tol, where r=residual, b=r.h.s., unweighted L2 norm */
      {
         hypre_StructCopy( b, t ); /* t = b */
         hypre_StructMatvecCompute( matvec_data,
                                    -1.0, A, x, 1.0, t );  /* t = - A x + t = - A x + b */
         rsumsq = hypre_StructInnerProd( t, t ); /* <t,t> */
         if ( rsumsq / bsumsq < tol2 ) { max_iter = iter; } /* converged; reset max_iter to prevent more iterations */
      }
   }

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/

   while (iter < max_iter)
   {
      if ( p == 0 ) { rsumsq = 0.0; }
      pointset = pointset_ranks[p];
      compute_pkg = compute_pkgs[pointset];
      stride = pointset_strides[pointset];

      /*hypre_StructCopy(x, t); ... not needed as long as the copy at the end of the loop
        is restricted to the current pointset (hypre_relax_copy, hypre_relax_wtx */

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x);
               hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
               compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_data_box =
               hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
            b_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), i);
            x_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
            t_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(t), i);

            bp = hypre_StructVectorBoxData(b, i);
            xp = hypre_StructVectorBoxData(x, i);
            tp = hypre_StructVectorBoxData(t, i);

            hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = hypre_BoxArrayBox(compute_box_a, j);

               if ( constant_coefficient == 1 || constant_coefficient == 2 )
               {
                  hypre_PointRelax_core12(
                     relax_vdata, A, constant_coefficient,
                     compute_box, bp, xp, tp, i,
                     A_data_box, b_data_box, x_data_box, t_data_box,
                     stride
                  );
               }

               else
               {
                  hypre_PointRelax_core0(
                     relax_vdata, A, constant_coefficient,
                     compute_box, bp, xp, tp, i,
                     A_data_box, b_data_box, x_data_box, t_data_box,
                     stride
                  );
               }

               Ap = hypre_StructMatrixBoxData(A, i, diag_rank);

               if ( constant_coefficient == 0 || constant_coefficient == 2 )
                  /* divide by the variable diagonal */
               {
                  start  = hypre_BoxIMin(compute_box);
                  hypre_BoxGetStrideSize(compute_box, stride, loop_size);
#define DEVICE_VAR is_device_ptr(tp,Ap)
                  hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                                      A_data_box, start, stride, Ai,
                                      t_data_box, start, stride, ti);
                  {
                     tp[ti] /= Ap[Ai];
                  }
                  hypre_BoxLoop2End(Ai, ti);
#undef DEVICE_VAR
               }
            }
         }
      }


      if (weight != 1.0)
      {
         /*        hypre_StructScale((1.0 - weight), x);
                   hypre_StructAxpy(weight, t, x);*/
         hypre_relax_wtx( relax_data, pointset, t, x ); /* x=w*t+(1-w)*x on pointset */
      }
      else
      {
         hypre_relax_copy( relax_data, pointset, t, x ); /* x=t on pointset */
         /* hypre_StructCopy(t, x);*/
      }

      p    = (p + 1) % num_pointsets;
      iter = iter + (p == 0);

      if ( tol > 0.0 && p == 0 )
         /* ... p==0 here means we've finished going through all the pointsets,
            i.e. this iteration is complete.
            tol>0.0 means to do a convergence test, using tol.
            The test is simply ||r||/||b||<tol, where r=residual, b=r.h.s., unweighted L2 norm */
      {
         hypre_StructCopy( b, t ); /* t = b */
         hypre_StructMatvecCompute( matvec_data,
                                    -1.0, A, x, 1.0, t );  /* t = - A x + t = - A x + b */
         rsumsq = hypre_StructInnerProd( t, t ); /* <t,t> */
         if ( rsumsq / bsumsq < tol2 ) { break; }
      }
   }

   if ( tol > 0.0 )
   {
      hypre_StructMatvecDestroy( matvec_data );
   }

   if ( tol > 0.0 ) { (relax_data -> rresnorm) = sqrt( rsumsq / bsumsq ); }
   (relax_data -> num_iterations) = iter;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(relax_data -> flops);
   hypre_EndTiming(relax_data -> time_index);

   return hypre_error_flag;
}

/* for constant_coefficient==0, all coefficients may vary ...*/
HYPRE_Int
hypre_PointRelax_core0( void               *relax_vdata,
                        hypre_StructMatrix *A,
                        HYPRE_Int           constant_coefficient,
                        hypre_Box          *compute_box,
                        HYPRE_Real         *bp,
                        HYPRE_Real         *xp,
                        HYPRE_Real         *tp,
                        HYPRE_Int           boxarray_id,
                        hypre_Box          *A_data_box,
                        hypre_Box          *b_data_box,
                        hypre_Box          *x_data_box,
                        hypre_Box          *t_data_box,
                        hypre_IndexRef      stride
                      )
{
   hypre_PointRelaxData  *relax_data = (hypre_PointRelaxData *)relax_vdata;

   HYPRE_Real            *Ap0;
   HYPRE_Real            *Ap1;
   HYPRE_Real            *Ap2;
   HYPRE_Real            *Ap3;
   HYPRE_Real            *Ap4;
   HYPRE_Real            *Ap5;
   HYPRE_Real            *Ap6;

   HYPRE_Int              xoff0;
   HYPRE_Int              xoff1;
   HYPRE_Int              xoff2;
   HYPRE_Int              xoff3;
   HYPRE_Int              xoff4;
   HYPRE_Int              xoff5;
   HYPRE_Int              xoff6;

   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;

   HYPRE_Int              diag_rank        = (relax_data -> diag_rank);
   hypre_IndexRef         start;
   hypre_Index            loop_size;
   HYPRE_Int              si, sk, ssi[MAX_DEPTH], depth, k;

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   start  = hypre_BoxIMin(compute_box);
   hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(tp,bp)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       b_data_box, start, stride, bi,
                       t_data_box, start, stride, ti);
   {
      tp[ti] = bp[bi];
   }
   hypre_BoxLoop2End(bi, ti);
#undef DEVICE_VAR

   /* unroll up to depth MAX_DEPTH */
   for (si = 0; si < stencil_size; si += MAX_DEPTH)
   {
      depth = hypre_min(MAX_DEPTH, (stencil_size - si));

      for (k = 0, sk = si; k < depth; sk++)
      {
         if (sk == diag_rank)
         {
            depth--;
         }
         else
         {
            ssi[k] = sk;
            k++;
         }
      }

      switch (depth)
      {
         case 7:
            Ap6 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[6]);
            xoff6 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[6]]);

         case 6:
            Ap5 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[5]);
            xoff5 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[5]]);

         case 5:
            Ap4 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[4]);
            xoff4 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[4]]);

         case 4:
            Ap3 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[3]);
            xoff3 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[3]]);

         case 3:
            Ap2 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[2]);
            xoff2 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[2]]);

         case 2:
            Ap1 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[1]);
            xoff1 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[1]]);

         case 1:
            Ap0 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[0]);
            xoff0 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[0]]);

         case 0:

            break;
      }

      switch (depth)
      {
         case 7:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,xp)
            hypre_BoxLoop3Begin(hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4] +
                  Ap5[Ai] * xp[xi + xoff5] +
                  Ap6[Ai] * xp[xi + xoff6];
            }
            hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 6:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,xp)
            hypre_BoxLoop3Begin(hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4] +
                  Ap5[Ai] * xp[xi + xoff5];
            }
            hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 5:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,Ap3,Ap4,xp)
            hypre_BoxLoop3Begin(hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4];
            }
            hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 4:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,Ap3,xp)
            hypre_BoxLoop3Begin(hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3];
            }
            hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 3:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,xp)
            hypre_BoxLoop3Begin(hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2];
            }
            hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 2:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,xp)
            hypre_BoxLoop3Begin(hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1];
            }
            hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 1:
#define DEVICE_VAR is_device_ptr(tp,Ap0,xp)
            hypre_BoxLoop3Begin(hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0];
            }
            hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 0:
            break;
      }
   }

   return hypre_error_flag;
}


/* for constant_coefficient==1 or 2, all offdiagonal coefficients constant over space ...*/
HYPRE_Int
hypre_PointRelax_core12( void               *relax_vdata,
                         hypre_StructMatrix *A,
                         HYPRE_Int           constant_coefficient,
                         hypre_Box          *compute_box,
                         HYPRE_Real         *bp,
                         HYPRE_Real         *xp,
                         HYPRE_Real         *tp,
                         HYPRE_Int           boxarray_id,
                         hypre_Box          *A_data_box,
                         hypre_Box          *b_data_box,
                         hypre_Box          *x_data_box,
                         hypre_Box          *t_data_box,
                         hypre_IndexRef      stride
                       )
{
   hypre_PointRelaxData  *relax_data = (hypre_PointRelaxData *)relax_vdata;

   HYPRE_Real            *Apd;
   HYPRE_Real            *Ap0;
   HYPRE_Real            *Ap1;
   HYPRE_Real            *Ap2;
   HYPRE_Real            *Ap3;
   HYPRE_Real            *Ap4;
   HYPRE_Real            *Ap5;
   HYPRE_Real            *Ap6;
   HYPRE_Real            AAp0;
   HYPRE_Real            AAp1;
   HYPRE_Real            AAp2;
   HYPRE_Real            AAp3;
   HYPRE_Real            AAp4;
   HYPRE_Real            AAp5;
   HYPRE_Real            AAp6;
   HYPRE_Real            AApd;

   HYPRE_Int              xoff0;
   HYPRE_Int              xoff1;
   HYPRE_Int              xoff2;
   HYPRE_Int              xoff3;
   HYPRE_Int              xoff4;
   HYPRE_Int              xoff5;
   HYPRE_Int              xoff6;

   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;

   HYPRE_Int              diag_rank        = (relax_data -> diag_rank);
   hypre_IndexRef         start;
   hypre_Index            loop_size;
   HYPRE_Int              si, sk, ssi[MAX_DEPTH], depth, k;
   HYPRE_Int              Ai;

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   start  = hypre_BoxIMin(compute_box);
   hypre_BoxGetStrideSize(compute_box, stride, loop_size);

   /* The standard (variable coefficient) algorithm initializes
      tp=bp.  Do it here, but for constant diagonal, also
      divide by the diagonal (and set up AApd for other
      division-equivalents.
      For a variable diagonal, this diagonal division is done
      at the end of the computation. */
   Ai = hypre_CCBoxIndexRank( A_data_box, start );

#define DEVICE_VAR is_device_ptr(tp,bp)
   if ( constant_coefficient == 1 ) /* constant diagonal */
   {
      Apd = hypre_StructMatrixBoxData(A, boxarray_id, diag_rank);
      AApd = 1 / Apd[Ai];

      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          b_data_box, start, stride, bi,
                          t_data_box, start, stride, ti);
      {
         tp[ti] = AApd * bp[bi];
      }
      hypre_BoxLoop2End(bi, ti);
   }
   else /* constant_coefficient==2, variable diagonal */
   {
      AApd = 1;
      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          b_data_box, start, stride, bi,
                          t_data_box, start, stride, ti);
      {
         tp[ti] = bp[bi];
      }
      hypre_BoxLoop2End(bi, ti);
   }
#undef DEVICE_VAR

   /* unroll up to depth MAX_DEPTH */
   for (si = 0; si < stencil_size; si += MAX_DEPTH)
   {
      depth = hypre_min(MAX_DEPTH, (stencil_size - si));

      for (k = 0, sk = si; k < depth; sk++)
      {
         if (sk == diag_rank)
         {
            depth--;
         }
         else
         {
            ssi[k] = sk;
            k++;
         }
      }

      switch (depth)
      {
         case 7:
            Ap6 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[6]);
            xoff6 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[6]]);

         case 6:
            Ap5 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[5]);
            xoff5 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[5]]);

         case 5:
            Ap4 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[4]);
            xoff4 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[4]]);

         case 4:
            Ap3 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[3]);
            xoff3 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[3]]);

         case 3:
            Ap2 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[2]);
            xoff2 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[2]]);

         case 2:
            Ap1 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[1]);
            xoff1 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[1]]);

         case 1:
            Ap0 = hypre_StructMatrixBoxData(A, boxarray_id, ssi[0]);
            xoff0 = hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[0]]);

         case 0:

            break;
      }

#define DEVICE_VAR is_device_ptr(tp,xp)
      switch (depth)
      {
         case 7:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            AAp3 = Ap3[Ai] * AApd;
            AAp4 = Ap4[Ai] * AApd;
            AAp5 = Ap5[Ai] * AApd;
            AAp6 = Ap6[Ai] * AApd;
            hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2] +
                  AAp3 * xp[xi + xoff3] +
                  AAp4 * xp[xi + xoff4] +
                  AAp5 * xp[xi + xoff5] +
                  AAp6 * xp[xi + xoff6];
            }
            hypre_BoxLoop2End(xi, ti);
            break;

         case 6:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            AAp3 = Ap3[Ai] * AApd;
            AAp4 = Ap4[Ai] * AApd;
            AAp5 = Ap5[Ai] * AApd;
            hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2] +
                  AAp3 * xp[xi + xoff3] +
                  AAp4 * xp[xi + xoff4] +
                  AAp5 * xp[xi + xoff5];
            }
            hypre_BoxLoop2End(xi, ti);
            break;

         case 5:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            AAp3 = Ap3[Ai] * AApd;
            AAp4 = Ap4[Ai] * AApd;
            hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2] +
                  AAp3 * xp[xi + xoff3] +
                  AAp4 * xp[xi + xoff4];
            }
            hypre_BoxLoop2End(xi, ti);
            break;

         case 4:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            AAp3 = Ap3[Ai] * AApd;
            hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2] +
                  AAp3 * xp[xi + xoff3];
            }
            hypre_BoxLoop2End(xi, ti);
            break;

         case 3:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2];
            }
            hypre_BoxLoop2End(xi, ti);
            break;

         case 2:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1];
            }
            hypre_BoxLoop2End(xi, ti);
            break;

         case 1:
            AAp0 = Ap0[Ai] * AApd;
            hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0];
            }
            hypre_BoxLoop2End(xi, ti);
            break;

         case 0:
            break;
      }
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetTol( void   *relax_vdata,
                        HYPRE_Real  tol         )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   (relax_data -> tol) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxGetTol( void   *relax_vdata,
                        HYPRE_Real *tol         )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   *tol = (relax_data -> tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetMaxIter( void *relax_vdata,
                            HYPRE_Int   max_iter    )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   (relax_data -> max_iter) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxGetMaxIter( void *relax_vdata,
                            HYPRE_Int * max_iter    )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   *max_iter = (relax_data -> max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetZeroGuess( void *relax_vdata,
                              HYPRE_Int   zero_guess  )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   (relax_data -> zero_guess) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxGetZeroGuess( void *relax_vdata,
                              HYPRE_Int * zero_guess  )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   *zero_guess = (relax_data -> zero_guess);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxGetNumIterations( void *relax_vdata,
                                  HYPRE_Int * num_iterations  )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   *num_iterations = (relax_data -> num_iterations);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetWeight( void    *relax_vdata,
                           HYPRE_Real   weight      )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   (relax_data -> weight) = weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetNumPointsets( void *relax_vdata,
                                 HYPRE_Int   num_pointsets )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;
   HYPRE_Int             i;

   /* free up old pointset memory */
   for (i = 0; i < (relax_data -> num_pointsets); i++)
   {
      hypre_TFree(relax_data -> pointset_indices[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(relax_data -> pointset_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> pointset_ranks, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> pointset_strides, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> pointset_indices, HYPRE_MEMORY_HOST);

   /* alloc new pointset memory */
   (relax_data -> num_pointsets)    = num_pointsets;
   (relax_data -> pointset_sizes)   = hypre_TAlloc(HYPRE_Int,  num_pointsets, HYPRE_MEMORY_HOST);
   (relax_data -> pointset_ranks)   = hypre_TAlloc(HYPRE_Int,  num_pointsets, HYPRE_MEMORY_HOST);
   (relax_data -> pointset_strides) = hypre_TAlloc(hypre_Index,  num_pointsets, HYPRE_MEMORY_HOST);
   (relax_data -> pointset_indices) = hypre_TAlloc(hypre_Index *,
                                                   num_pointsets, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_pointsets; i++)
   {
      (relax_data -> pointset_sizes[i]) = 0;
      (relax_data -> pointset_ranks[i]) = i;
      (relax_data -> pointset_indices[i]) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetPointset( void        *relax_vdata,
                             HYPRE_Int    pointset,
                             HYPRE_Int    pointset_size,
                             hypre_Index  pointset_stride,
                             hypre_Index *pointset_indices )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;
   HYPRE_Int             i;

   /* free up old pointset memory */
   hypre_TFree(relax_data -> pointset_indices[pointset], HYPRE_MEMORY_HOST);

   /* alloc new pointset memory */
   (relax_data -> pointset_indices[pointset]) =
      hypre_TAlloc(hypre_Index,  pointset_size, HYPRE_MEMORY_HOST);

   (relax_data -> pointset_sizes[pointset]) = pointset_size;
   hypre_CopyIndex(pointset_stride,
                   (relax_data -> pointset_strides[pointset]));
   for (i = 0; i < pointset_size; i++)
   {
      hypre_CopyIndex(pointset_indices[i],
                      (relax_data -> pointset_indices[pointset][i]));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetPointsetRank( void *relax_vdata,
                                 HYPRE_Int   pointset,
                                 HYPRE_Int   pointset_rank )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   (relax_data -> pointset_ranks[pointset]) = pointset_rank;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetTempVec( void               *relax_vdata,
                            hypre_StructVector *t           )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   hypre_StructVectorDestroy(relax_data -> t);
   (relax_data -> t) = hypre_StructVectorRef(t);

   return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_PointRelaxGetFinalRelativeResidualNorm( void * relax_vdata, HYPRE_Real * norm )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   *norm = relax_data -> rresnorm;
   return 0;
}

/*--------------------------------------------------------------------------
 * Special vector operation for use in hypre_PointRelax -
 * convex combination of vectors on specified pointsets.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_relax_wtx( void *relax_vdata, HYPRE_Int pointset,
                           hypre_StructVector *t, hypre_StructVector *x )
/* Sets x to a convex combination of x and t,  x = weight * t + (1-weight) * x,
   but only in the specified pointset */
{
   hypre_PointRelaxData  *relax_data = (hypre_PointRelaxData *)relax_vdata;
   HYPRE_Real             weight           = (relax_data -> weight);
   hypre_Index           *pointset_strides = (relax_data -> pointset_strides);
   hypre_ComputePkg     **compute_pkgs     = (relax_data -> compute_pkgs);
   hypre_ComputePkg      *compute_pkg;

   hypre_IndexRef         stride;
   hypre_IndexRef         start;
   hypre_Index            loop_size;

   HYPRE_Real weightc = 1 - weight;
   HYPRE_Real *xp, *tp;
   HYPRE_Int compute_i, i, j;

   hypre_BoxArrayArray   *compute_box_aa;
   hypre_BoxArray        *compute_box_a;
   hypre_Box             *compute_box;
   hypre_Box             *x_data_box;
   hypre_Box             *t_data_box;

   compute_pkg = compute_pkgs[pointset];
   stride = pointset_strides[pointset];

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
         }
         break;

         case 1:
         {
            compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      hypre_ForBoxArrayI(i, compute_box_aa)
      {
         compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

         x_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         t_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(t), i);

         xp = hypre_StructVectorBoxData(x, i);
         tp = hypre_StructVectorBoxData(t, i);

         hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = hypre_BoxArrayBox(compute_box_a, j);

            start  = hypre_BoxIMin(compute_box);
            hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,tp)
            hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               xp[xi] = weight * tp[ti] + weightc * xp[xi];
            }
            hypre_BoxLoop2End(xi, ti);
#undef DEVICE_VAR
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Special vector operation for use in hypre_PointRelax -
 * vector copy on specified pointsets.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_relax_copy( void *relax_vdata, HYPRE_Int pointset,
                            hypre_StructVector *t, hypre_StructVector *x )
/* Sets x to t, x=t, but only in the specified pointset. */
{
   hypre_PointRelaxData  *relax_data = (hypre_PointRelaxData *)relax_vdata;
   hypre_Index           *pointset_strides = (relax_data -> pointset_strides);
   hypre_ComputePkg     **compute_pkgs     = (relax_data -> compute_pkgs);
   hypre_ComputePkg      *compute_pkg;

   hypre_IndexRef         stride;
   hypre_IndexRef         start;
   hypre_Index            loop_size;

   HYPRE_Real *xp, *tp;
   HYPRE_Int compute_i, i, j;

   hypre_BoxArrayArray   *compute_box_aa;
   hypre_BoxArray        *compute_box_a;
   hypre_Box             *compute_box;
   hypre_Box             *x_data_box;
   hypre_Box             *t_data_box;

   compute_pkg = compute_pkgs[pointset];
   stride = pointset_strides[pointset];

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
         }
         break;

         case 1:
         {
            compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      hypre_ForBoxArrayI(i, compute_box_aa)
      {
         compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

         x_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         t_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(t), i);

         xp = hypre_StructVectorBoxData(x, i);
         tp = hypre_StructVectorBoxData(t, i);

         hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = hypre_BoxArrayBox(compute_box_a, j);

            start  = hypre_BoxIMin(compute_box);
            hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,tp)
            hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               xp[xi] = tp[ti];
            }
            hypre_BoxLoop2End(xi, ti);
#undef DEVICE_VAR
         }
      }
   }

   return hypre_error_flag;
}
