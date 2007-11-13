/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




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

double hypre_F90_NAME_BLAS(ddot, DDOT)(int *, double *, int *, double *, int *);
int hypre_F90_NAME_BLAS(dcopy, DCOPY)(int *, double *, int *, double *, int *);
int hypre_F90_NAME_BLAS(dscal, DSCAL)(int *, double *, double *, int *);
int hypre_F90_NAME_BLAS(daxpy, DAXPY)(int *, double *, double *, int *, double *, int *);

static double InnerProd(int n, double *x, double *y, MPI_Comm comm)
{
    double local_result, result;

    int one = 1;
    local_result = hypre_F90_NAME_BLAS(ddot, DDOT)(&n, x, &one, y, &one);

    MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, comm);

    return result;
}

static void CopyVector(int n, double *x, double *y)
{
    int one = 1;
    hypre_F90_NAME_BLAS(dcopy, DCOPY)(&n, x, &one, y, &one);
}

static void ScaleVector(int n, double alpha, double *x)
{
    int one = 1;
    hypre_F90_NAME_BLAS(dscal, DSCAL)(&n, &alpha, x, &one);
}

static void Axpy(int n, double alpha, double *x, double *y)
{
    int one = 1;
    hypre_F90_NAME_BLAS(daxpy, DAXPY)(&n, &alpha, x, &one, y, &one);
}

/* simulate 2-D arrays at the cost of some arithmetic */
#define V(i) (&V[(i)*n])
#define W(i) (&W[(i)*n])
#define H(i,j) (H[(j)*m1+(i)])

static void
GeneratePlaneRotation(double dx, double dy, double *cs, double *sn)
{
  if (dy == 0.0) {
    *cs = 1.0;
    *sn = 0.0;
  } else if (ABS(dy) > ABS(dx)) {
    double temp = dx / dy;
    *sn = 1.0 / sqrt( 1.0 + temp*temp );
    *cs = temp * *sn;
  } else {
    double temp = dy / dx;
    *cs = 1.0 / sqrt( 1.0 + temp*temp );
    *sn = temp * *cs;
  }
}

static void ApplyPlaneRotation(double *dx, double *dy, double cs, double sn)
{
  double temp  =  cs * *dx + sn * *dy;
  *dy = -sn * *dx + cs * *dy;
  *dx = temp;
}

void FGMRES_ParaSails(Matrix *mat, ParaSails *ps, double *b, double *x,
  int dim, double tol, int max_iter)
{
    int mype;
    int iter;
    double rel_resid;

    double *H  = (double *) malloc(dim*(dim+1) * sizeof(double));

    /* local problem size */
    int n = mat->end_row - mat->beg_row + 1;

    int m1 = dim+1; /* used inside H macro */
    int i, j, k;
    double beta, resid0;

    double *s  = (double *) malloc((dim+1) * sizeof(double));
    double *cs = (double *) malloc(dim * sizeof(double));
    double *sn = (double *) malloc(dim * sizeof(double));

    double *V  = (double *) malloc(n*(dim+1) * sizeof(double));
    double *W  = (double *) malloc(n*dim * sizeof(double));

    MPI_Comm comm = mat->comm;
    MPI_Comm_rank(comm, &mype);

    iter = 0;
    do
    {
        /* compute initial residual and its norm */
        MatrixMatvec(mat, x, V(0));                      /* V(0) = A*x        */
        Axpy(n, -1.0, b, V(0));                          /* V(0) = V(0) - b   */
        beta = sqrt(InnerProd(n, V(0), V(0), comm));     /* beta = norm(V(0)) */
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

            H(i+1, i) = sqrt(InnerProd(n, V(i+1), V(i+1), comm));
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
               printf("Iter (%d): rel. resid. norm: %e\n", iter, rel_resid);
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
    beta = sqrt(InnerProd(n, V(0), V(0), comm));        /* beta = norm(V(0)) */
    rel_resid = beta / resid0;

    if (mype == 0)
        printf("Iter (%d): computed rrn    : %e\n", iter, rel_resid);

    free(H);
    free(s);
    free(cs);
    free(sn);
    free(V);
    free(W);
}
