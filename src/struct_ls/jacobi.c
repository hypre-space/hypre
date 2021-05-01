/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   HYPRE_Real              tol;        /* tolerance, set =0 for no convergence testing */
   HYPRE_Real              rresnorm;   /* relative residual norm, computed only if tol>0.0 */
   HYPRE_Int               max_iter;
   HYPRE_Int               rel_change; /* not yet used */
   HYPRE_Int               zero_guess;
   HYPRE_Real              weight;

   hypre_StructMatrix     *A;
   hypre_StructVector     *b;
   hypre_StructVector     *x;
   hypre_StructVector     *r;

   void                   *matvec_data;

   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;
   HYPRE_BigInt            flops;

} hypre_StructJacobiData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_StructJacobiCreate( MPI_Comm comm )
{
   hypre_StructJacobiData *jacobi_data;

   jacobi_data = hypre_CTAlloc(hypre_StructJacobiData, 1, HYPRE_MEMORY_HOST);

   (jacobi_data -> comm)       = comm;
   (jacobi_data -> time_index) = hypre_InitializeTiming("Jacobi");

   /* set defaults */
   (jacobi_data -> tol)        = 0.0;  /* tol=0 means no convergence testing */
   (jacobi_data -> rresnorm)   = 0.0;
   (jacobi_data -> max_iter)   = 1000;
   (jacobi_data -> rel_change) = 0;
   (jacobi_data -> zero_guess) = 0;
   (jacobi_data -> weight)     = 1.0;
   (jacobi_data -> A)          = NULL;
   (jacobi_data -> b)          = NULL;
   (jacobi_data -> x)          = NULL;
   (jacobi_data -> r)          = NULL;

   return (void *) jacobi_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiDestroy( void *jacobi_vdata )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   if (jacobi_data)
   {
      hypre_StructMatvecDestroy(jacobi_data -> matvec_data);
      hypre_StructMatrixDestroy(jacobi_data -> A);
      hypre_StructVectorDestroy(jacobi_data -> b);
      hypre_StructVectorDestroy(jacobi_data -> x);
      hypre_StructVectorDestroy(jacobi_data -> r);

      hypre_FinalizeTiming(jacobi_data -> time_index);
      hypre_TFree(jacobi_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiSetup( void               *jacobi_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *b,
                         hypre_StructVector *x )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;
   hypre_StructVector     *r           = (jacobi_data -> r);

   /*----------------------------------------------------------
    * Set up the residual vector
    *----------------------------------------------------------*/

   if (r == NULL)
   {
      r = hypre_StructVectorClone(b);
      (jacobi_data -> r) = r;
   }

   /*----------------------------------------------------------
    * Set up matvec
    *----------------------------------------------------------*/

   (jacobi_data -> matvec_data) = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup((jacobi_data -> matvec_data), A, x);

   /*----------------------------------------------------------
    * Set up the relax data structure
    *----------------------------------------------------------*/

   (jacobi_data -> A) = hypre_StructMatrixRef(A);
   (jacobi_data -> x) = hypre_StructVectorRef(x);
   (jacobi_data -> b) = hypre_StructVectorRef(b);

   /*-----------------------------------------------------
    * Compute flops
    *-----------------------------------------------------*/

   (jacobi_data -> flops) =
      (HYPRE_BigInt)(hypre_StructMatrixGlobalSize(A) + hypre_StructVectorGlobalSize(x));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiSolve( void               *jacobi_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *b,
                         hypre_StructVector *x )
{
   hypre_StructJacobiData *jacobi_data      = (hypre_StructJacobiData *) jacobi_vdata;

   void                   *matvec_data      = (jacobi_data -> matvec_data);
   HYPRE_Int               max_iter         = (jacobi_data -> max_iter);
   HYPRE_Int               zero_guess       = (jacobi_data -> zero_guess);
   HYPRE_Real              weight           = (jacobi_data -> weight);
   hypre_StructVector     *r                = (jacobi_data -> r);
   HYPRE_Real              tol              = (jacobi_data -> tol);
   HYPRE_Real              tol2             = tol*tol;
   HYPRE_Int               ndim             = hypre_StructMatrixNDim(A);

   hypre_BoxArray         *boxes;
   hypre_Box              *compute_box;
   hypre_Box              *A_data_box, *x_data_box, *r_data_box;

   HYPRE_Real             *Ap, *xp, *rp;

   hypre_IndexRef          dstart;
   hypre_Index             dstride;
   hypre_Index             loop_size;

   HYPRE_Int               iter, i, stencil_diag;
   HYPRE_Real              bsumsq, rsumsq;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/

   hypre_BeginTiming(jacobi_data -> time_index);

   hypre_StructMatrixDestroy(jacobi_data -> A);
   hypre_StructVectorDestroy(jacobi_data -> b);
   hypre_StructVectorDestroy(jacobi_data -> x);
   (jacobi_data -> A) = hypre_StructMatrixRef(A);
   (jacobi_data -> x) = hypre_StructVectorRef(x);
   (jacobi_data -> b) = hypre_StructVectorRef(b);

   (jacobi_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
      }

      hypre_EndTiming(jacobi_data -> time_index);
      return hypre_error_flag;
   }

//   hypre_StructVectorClearBoundGhostValues(x, 0); /* RDF: Shouldn't need this */

   rsumsq = 0.0;
   if (tol > 0.0)
   {
      bsumsq = hypre_StructInnerProd( b, b );
   }

   stencil_diag = hypre_StructStencilDiagEntry(hypre_StructMatrixStencil(A));
   compute_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(dstride, 1);

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/

   iter = 0;

   if (zero_guess)
   {
      /* Compute x <-- weight D^-1 b */

      boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(x));
      hypre_ForBoxI(i, boxes)
      {
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), compute_box);
         hypre_StructVectorMapDataBox(x, compute_box);
         hypre_BoxGetSize(compute_box, loop_size);
         dstart = hypre_BoxIMin(compute_box);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         r_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), i); /* r = b */

         Ap = hypre_StructMatrixBoxData(A, i, stencil_diag);
         xp = hypre_StructVectorBoxData(x, i);
         rp = hypre_StructVectorBoxData(b, i); /* r = b */

         if (hypre_StructMatrixConstEntry(A, stencil_diag))
         {
            /* Constant coefficient case */
            HYPRE_Real scale = weight / Ap[0];
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, dstart, dstride, xi,
                                r_data_box, dstart, dstride, ri);
            {
               xp[xi] = scale * rp[ri];
            }
            hypre_BoxLoop2End(xi, ri);
         }
         else
         {
            /* Variable coefficient case */
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, dstart, dstride, Ai,
                                x_data_box, dstart, dstride, xi,
                                r_data_box, dstart, dstride, ri);
            {
               xp[xi] = weight * rp[ri] / Ap[Ai];
            }
            hypre_BoxLoop3End(Ai, xi, ri);
         }
      }

      iter = iter + 1;

      if ( tol > 0.0 )
      {
         /* Compute residual and check convergence */
         hypre_StructCopy(b, r);
         hypre_StructMatvecCompute(matvec_data, -1.0, A, x, 1.0, r);
         rsumsq = hypre_StructInnerProd(r, r);
         if ( (rsumsq/bsumsq) < tol2 )
         {
            max_iter = iter; /* converged; reset max_iter to prevent more iterations */
         }
      }
   }

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/

   while (iter < max_iter)
   {
      /* Compute residual */

      hypre_StructCopy(b, r);
      /* Matvec is optimized for residual computations: alpha = -1, beta = 1 */
      hypre_StructMatvecCompute(matvec_data, -1.0, A, x, 1.0, r);

      /* Compute x <-- x + weight D^-1 r */

      boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(x));
      hypre_ForBoxI(i, boxes)
      {
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), compute_box);
         hypre_StructVectorMapDataBox(x, compute_box);
         hypre_BoxGetSize(compute_box, loop_size);
         dstart = hypre_BoxIMin(compute_box);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         r_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(r), i);

         Ap = hypre_StructMatrixBoxData(A, i, stencil_diag);
         xp = hypre_StructVectorBoxData(x, i);
         rp = hypre_StructVectorBoxData(r, i);

         if (hypre_StructMatrixConstEntry(A, stencil_diag))
         {
            /* Constant coefficient case */
            HYPRE_Real scale = weight / Ap[0];
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, dstart, dstride, xi,
                                r_data_box, dstart, dstride, ri);
            {
               xp[xi] += scale * rp[ri];
            }
            hypre_BoxLoop2End(xi, ri);
         }
         else
         {
            /* Variable coefficient case */
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, dstart, dstride, Ai,
                                x_data_box, dstart, dstride, xi,
                                r_data_box, dstart, dstride, ri);
            {
               xp[xi] += weight * rp[ri] / Ap[Ai];
            }
            hypre_BoxLoop3End(Ai, xi, ri);
         }
      }

      iter = iter + 1;

      if ( tol > 0.0 )
      {
         /* Compute residual and check convergence */
         hypre_StructCopy(b, r);
         hypre_StructMatvecCompute(matvec_data, -1.0, A, x, 1.0, r);
         rsumsq = hypre_StructInnerProd(r, r);
         if ( (rsumsq/bsumsq) < tol2 )
         {
            break;
         }
      }
   }

   if ( tol > 0.0 )
   {
      (jacobi_data -> rresnorm) = sqrt(rsumsq/bsumsq);
   }
   (jacobi_data -> num_iterations) = iter;

   hypre_BoxDestroy(compute_box);

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(jacobi_data -> flops);
   hypre_EndTiming(jacobi_data -> time_index);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiSetTol( void       *jacobi_vdata,
                          HYPRE_Real  tol )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   (jacobi_data -> tol) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiGetTol( void       *jacobi_vdata,
                          HYPRE_Real *tol )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   *tol = (jacobi_data -> tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiSetMaxIter( void      *jacobi_vdata,
                              HYPRE_Int  max_iter )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   (jacobi_data -> max_iter) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructJacobiGetMaxIter( void      *jacobi_vdata,
                              HYPRE_Int *max_iter )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   *max_iter = (jacobi_data -> max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiSetZeroGuess( void      *jacobi_vdata,
                                HYPRE_Int  zero_guess )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   (jacobi_data -> zero_guess) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiGetZeroGuess( void      *jacobi_vdata,
                                HYPRE_Int *zero_guess )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   *zero_guess = (jacobi_data -> zero_guess);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructJacobiGetNumIterations( void      *jacobi_vdata,
                                    HYPRE_Int *num_iterations )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   *num_iterations = (jacobi_data -> num_iterations);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiSetWeight( void       *jacobi_vdata,
                             HYPRE_Real  weight )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   (jacobi_data -> weight) = weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructJacobiSetTempVec( void               *jacobi_vdata,
                              hypre_StructVector *r )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   hypre_StructVectorDestroy(jacobi_data -> r);
   (jacobi_data -> r) = hypre_StructVectorRef(r);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructJacobiGetFinalRelativeResidualNorm( void       *jacobi_vdata,
                                                          HYPRE_Real *norm )
{
   hypre_StructJacobiData *jacobi_data = (hypre_StructJacobiData *)jacobi_vdata;

   *norm = jacobi_data -> rresnorm;

   return hypre_error_flag;
}
