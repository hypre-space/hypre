/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * FlexGmres - Preconditioned flexible GMRES algorithm using the
 * ParaSails preconditioner.
 *
 *****************************************************************************/

#include "math.h"
#include "Common.h"
#include "Matrix.h"
#include "ParaSails.h"
#include "_hypre_blas.h"

static HYPRE_Real InnerProd(HYPRE_Int n, HYPRE_Real *x, HYPRE_Real *y, MPI_Comm comm)
{
    HYPRE_Real local_result, result;

    HYPRE_Int one = 1;
    local_result = hypre_ddot(&n, x, &one, y, &one);

    hypre_MPI_Allreduce(&local_result, &result, 1, hypre_MPI_REAL, hypre_MPI_SUM, comm);

    return result;
}

static void CopyVector(HYPRE_Int n, HYPRE_Real *x, HYPRE_Real *y)
{
    HYPRE_Int one = 1;
    hypre_dcopy(&n, x, &one, y, &one);
}

static void ScaleVector(HYPRE_Int n, HYPRE_Real alpha, HYPRE_Real *x)
{
    HYPRE_Int one = 1;
    hypre_dscal(&n, &alpha, x, &one);
}

static void Axpy(HYPRE_Int n, HYPRE_Real alpha, HYPRE_Real *x, HYPRE_Real *y)
{
    HYPRE_Int one = 1;
    hypre_daxpy(&n, &alpha, x, &one, y, &one);
}

/* simulate 2-D arrays at the cost of some arithmetic */
#define V(i) (&V[(i)*n])
#define W(i) (&W[(i)*n])
#define H(i,j) (H[(j)*m1+(i)])

static void
GeneratePlaneRotation(HYPRE_Real dx, HYPRE_Real dy, HYPRE_Real *cs, HYPRE_Real *sn)
{
  if (dy == 0.0) {
    *cs = 1.0;
    *sn = 0.0;
  } else if (ABS(dy) > ABS(dx)) {
    HYPRE_Real temp = dx / dy;
    *sn = 1.0 / hypre_sqrt( 1.0 + temp*temp );
    *cs = temp * *sn;
  } else {
    HYPRE_Real temp = dy / dx;
    *cs = 1.0 / hypre_sqrt( 1.0 + temp*temp );
    *sn = temp * *cs;
  }
}

static void ApplyPlaneRotation(HYPRE_Real *dx, HYPRE_Real *dy, HYPRE_Real cs, HYPRE_Real sn)
{
  HYPRE_Real temp  =  cs * *dx + sn * *dy;
  *dy = -sn * *dx + cs * *dy;
  *dx = temp;
}

void FGMRES_ParaSails(Matrix *mat, ParaSails *ps, HYPRE_Real *b, HYPRE_Real *x,
  HYPRE_Int dim, HYPRE_Real tol, HYPRE_Int max_iter)
{
    HYPRE_Int mype;
    HYPRE_Int iter;
    HYPRE_Real rel_resid;

    HYPRE_Real *H  = hypre_TAlloc(HYPRE_Real, dim*(dim+1) , HYPRE_MEMORY_HOST);

    /* local problem size */
    HYPRE_Int n = mat->end_row - mat->beg_row + 1;

    HYPRE_Int m1 = dim+1; /* used inside H macro */
    HYPRE_Int i, j, k;
    HYPRE_Real beta, resid0;

    HYPRE_Real *s  = hypre_TAlloc(HYPRE_Real, (dim+1) , HYPRE_MEMORY_HOST);
    HYPRE_Real *cs = hypre_TAlloc(HYPRE_Real, dim , HYPRE_MEMORY_HOST);
    HYPRE_Real *sn = hypre_TAlloc(HYPRE_Real, dim , HYPRE_MEMORY_HOST);

    HYPRE_Real *V  = hypre_TAlloc(HYPRE_Real, n*(dim+1) , HYPRE_MEMORY_HOST);
    HYPRE_Real *W  = hypre_TAlloc(HYPRE_Real, n*dim , HYPRE_MEMORY_HOST);

    MPI_Comm comm = mat->comm;
    hypre_MPI_Comm_rank(comm, &mype);

    iter = 0;
    do
    {
        /* compute initial residual and its norm */
        MatrixMatvec(mat, x, V(0));                      /* V(0) = A*x        */
        Axpy(n, -1.0, b, V(0));                          /* V(0) = V(0) - b   */
        beta = hypre_sqrt(InnerProd(n, V(0), V(0), comm));     /* beta = norm(V(0)) */
        ScaleVector(n, -1.0/beta, V(0));                 /* V(0) = -V(0)/beta */

        /* save very first residual norm */
        if (iter == 0)
            resid0 = beta;

        for (i = 1; i < dim+1; i++)
            s[i] = 0.0;
        s[0] = beta;

        i = -1;
        do
        {
            i++;
            iter++;

            if (ps != NULL)
                ParaSailsApply(ps, V(i), W(i));
            else
                CopyVector(n, V(i), W(i));

            MatrixMatvec(mat, W(i), V(i+1));

            for (k = 0; k <= i; k++)
            {
                H(k, i) = InnerProd(n, V(i+1), V(k), comm);
                /* V(i+1) -= H(k, i) * V(k); */
                Axpy(n, -H(k,i), V(k), V(i+1));
            }

            H(i+1, i) = hypre_sqrt(InnerProd(n, V(i+1), V(i+1), comm));
            /* V(i+1) = V(i+1) / H(i+1, i) */
            ScaleVector(n, 1.0 / H(i+1, i), V(i+1));

            for (k = 0; k < i; k++)
                ApplyPlaneRotation(&H(k,i), &H(k+1,i), cs[k], sn[k]);

            GeneratePlaneRotation(H(i,i), H(i+1,i), &cs[i], &sn[i]);
            ApplyPlaneRotation(&H(i,i), &H(i+1,i), cs[i], sn[i]);
            ApplyPlaneRotation(&s[i], &s[i+1], cs[i], sn[i]);

            rel_resid = ABS(s[i+1]) / resid0;
#ifdef PARASAILS_CG_PRINT
            if (mype == 0 && iter % 10 == 0)
               hypre_printf("Iter (%d): rel. resid. norm: %e\n", iter, rel_resid);
#endif
            if (rel_resid <= tol)
                break;
        }
        while (i+1 < dim && iter+1 <= max_iter);

        /* solve upper triangular system in place */
        for (j = i; j >= 0; j--)
        {
            s[j] /= H(j,j);
            for (k = j-1; k >= 0; k--)
                s[k] -= H(k,j) * s[j];
        }

        /* update the solution */
        for (j = 0; j <= i; j++)
        {
            /* x = x + s[j] * W(j) */
            Axpy(n, s[j], W(j), x);
        }
    }
    while (rel_resid > tol && iter+1 <= max_iter);

    /* compute exact residual norm reduction */
    MatrixMatvec(mat, x, V(0));                         /* V(0) = A*x        */
    Axpy(n, -1.0, b, V(0));                             /* V(0) = V(0) - b   */
    beta = hypre_sqrt(InnerProd(n, V(0), V(0), comm));        /* beta = norm(V(0)) */
    rel_resid = beta / resid0;

    if (mype == 0)
        hypre_printf("Iter (%d): computed rrn    : %e\n", iter, rel_resid);

    hypre_TFree(H, HYPRE_MEMORY_HOST);
    hypre_TFree(s, HYPRE_MEMORY_HOST);
    hypre_TFree(cs, HYPRE_MEMORY_HOST);
    hypre_TFree(sn, HYPRE_MEMORY_HOST);
    hypre_TFree(V, HYPRE_MEMORY_HOST);
    hypre_TFree(W, HYPRE_MEMORY_HOST);
}

