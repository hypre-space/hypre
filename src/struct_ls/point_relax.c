/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/* RDF NOTE: This is just Jacobi now, and should be renamed as such.  It uses
 * Matvec and relies on optimizations done there. */

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

   hypre_StructMatrix     *A;
   hypre_StructVector     *b;
   hypre_StructVector     *x;
   hypre_StructVector     *r;

   void                   *matvec_data;

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

   relax_data = hypre_CTAlloc(hypre_PointRelaxData, 1, HYPRE_MEMORY_HOST);

   (relax_data -> comm)       = comm;
   (relax_data -> time_index) = hypre_InitializeTiming("PointRelax");

   /* set defaults */
   (relax_data -> tol)              = 0.0;  /* tol=0 means no convergence testing */
   (relax_data -> rresnorm)         = 0.0;
   (relax_data -> max_iter)         = 1000;
   (relax_data -> rel_change)       = 0;
   (relax_data -> zero_guess)       = 0;
   (relax_data -> weight)           = 1.0;
   (relax_data -> A)                = NULL;
   (relax_data -> b)                = NULL;
   (relax_data -> x)                = NULL;
   (relax_data -> r)                = NULL;

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxDestroy( void *relax_vdata )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   if (relax_data)
   {
      hypre_StructMatvecDestroy(relax_data -> matvec_data);
      hypre_StructMatrixDestroy(relax_data -> A);
      hypre_StructVectorDestroy(relax_data -> b);
      hypre_StructVectorDestroy(relax_data -> x);
      hypre_StructVectorDestroy(relax_data -> r);

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
   hypre_StructVector    *r          = (relax_data -> r);

   /*----------------------------------------------------------
    * Set up the residual vector
    *----------------------------------------------------------*/

   if (r == NULL)
   {
      r = hypre_StructVectorClone(b);
      (relax_data -> r) = r;
   }

   /*----------------------------------------------------------
    * Set up matvec
    *----------------------------------------------------------*/

   (relax_data -> matvec_data) = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup((relax_data -> matvec_data), A, x);

   /*----------------------------------------------------------
    * Set up the relax data structure
    *----------------------------------------------------------*/

   (relax_data -> A) = hypre_StructMatrixRef(A);
   (relax_data -> x) = hypre_StructVectorRef(x);
   (relax_data -> b) = hypre_StructVectorRef(b);

   /*-----------------------------------------------------
    * Compute flops
    *-----------------------------------------------------*/

   (relax_data -> flops) =
      (HYPRE_BigInt)(hypre_StructMatrixGlobalSize(A) + hypre_StructVectorGlobalSize(x));

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
   hypre_PointRelaxData  *relax_data       = (hypre_PointRelaxData *) relax_vdata;

   void                  *matvec_data      = (relax_data -> matvec_data);
   HYPRE_Int              max_iter         = (relax_data -> max_iter);
   HYPRE_Int              zero_guess       = (relax_data -> zero_guess);
   HYPRE_Real             weight           = (relax_data -> weight);
   hypre_StructVector    *r                = (relax_data -> r);
   HYPRE_Real             tol              = (relax_data -> tol);
   HYPRE_Real             tol2             = tol*tol;
   HYPRE_Int              ndim             = hypre_StructMatrixNDim(A);

   hypre_BoxArray        *boxes;
   hypre_Box             *compute_box;
   hypre_Box             *A_data_box, *x_data_box, *r_data_box;

   HYPRE_Real            *Ap, *xp, *rp;

   hypre_IndexRef         dstart;
   hypre_Index            dstride;
   hypre_Index            loop_size;

   HYPRE_Int              iter, i, stencil_diag, bsumsq, rsumsq;

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
            HYPRE_Real  scale = weight / Ap[0];
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
            HYPRE_Real  scale = weight / Ap[0];
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
      (relax_data -> rresnorm) = sqrt(rsumsq/bsumsq);
   }
   (relax_data -> num_iterations) = iter;

   hypre_BoxDestroy(compute_box);

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(relax_data -> flops);
   hypre_EndTiming(relax_data -> time_index);

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
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetPointsetRank( void *relax_vdata,
                                 HYPRE_Int   pointset,
                                 HYPRE_Int   pointset_rank )
{
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PointRelaxSetTempVec( void               *relax_vdata,
                            hypre_StructVector *r           )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   hypre_StructVectorDestroy(relax_data -> r);
   (relax_data -> r) = hypre_StructVectorRef(r);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_PointRelaxGetFinalRelativeResidualNorm( void * relax_vdata,
                                                        HYPRE_Real * norm )
{
   hypre_PointRelaxData *relax_data = (hypre_PointRelaxData *)relax_vdata;

   *norm = relax_data -> rresnorm;

   return hypre_error_flag;
}
