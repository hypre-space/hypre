/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Chebyshev setup and solve
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "float.h"
#include "_hypre_lapack.h"
#include "_hypre_blas.h"

/* GMRES_DEVELOPMENT will be removed by the time development is finished/merged*/

#define GMRES_DEVELOPMENT 0
#define GMRES_DEBUG 0

#if GMRES_DEVELOPMENT
void PrintVector(hypre_ParVector* x) {

      HYPRE_Real* tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
      HYPRE_Int n= hypre_VectorSize(hypre_ParVectorLocalVector(x));
      for(int k = 0; k < n; k++) {
        printf("%f\n", tmp_data[k]);
      }
}

void PrintMatrix(int m, int n, double* x) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      printf("%f ", x[i+j*m]);
    }
    printf("\n");
  }
}
#endif

/******************************************************************************

Chebyshev relaxation

 
Can specify order 1-4 (this is the order of the resid polynomial)- here we
explicitly code the coefficients (instead of
iteratively determining)


variant 0: standard chebyshev
this is rlx 11 if scale = 0, and 16 if scale == 1

variant 1: modified cheby: T(t)* f(t) where f(t) = (1-b/t)
this is rlx 15 if scale = 0, and 17 if scale == 1

ratio indicates the percentage of the whole spectrum to use (so .5
means half, and .1 means 10percent)


*******************************************************************************/

/**
 * @brief Uses Arnoldi MGS to generate H,Q such that Q_m^TAQ_m=H_m, 
 * Algorithm 6.2 in Saad */
HYPRE_Int hypre_ParCSRConstructArnoldi(hypre_ParCSRMatrix *A,
                                      hypre_ParVector *b,
                                      HYPRE_Int d,
                                      hypre_ParVector** Q,
                                      HYPRE_Real** H
                                      )
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int i = 0;
   HYPRE_Int j = 0;
   HYPRE_Int k = 0;
   HYPRE_Int ierr;



   HYPRE_Real norm;

   /* Normalize input b */
   HYPRE_ParVectorInnerProd(b,b,&norm);
   norm = 1./sqrt(norm);
   ierr = HYPRE_ParVectorScale(norm, b);

   hypre_ParVectorCopy(b, Q[0]); 
   PrintVector(Q[0]);
   while(i < d)
   {
     i++;
     PrintVector(Q[i-1]);
     hypre_ParCSRMatrixMatvec(1, A, Q[i-1], 0, Q[i]);
     PrintVector(Q[i]);
     for(j = 0; j < i; j++)
     {
       HYPRE_ParVectorInnerProd(Q[j],Q[i],&H[j][i-1]);
       HYPRE_ParVectorAxpy(-H[j][i-1], Q[j], Q[i]);
     }
     HYPRE_ParVectorInnerProd(Q[i],Q[i],&norm);
     H[i][i-1] = norm;
     if (norm != 0.0)
     {
       norm = 1.0 / norm;
       ierr = HYPRE_ParVectorScale(norm, Q[i]);

      PrintVector(Q[i]);
     }
   }

   return hypre_error_flag;
}

/*
 * @brief Used to verify the correctness of the arnoldi implementation
 * Assumes nrank == 1 
 */
HYPRE_Int hypre_ParCSRVerifyArnoldi(hypre_ParCSRMatrix *A,
                                      hypre_ParVector *b,
                                      HYPRE_Int d,
                                      hypre_ParVector** Q,
                                      HYPRE_Real** H
    ) {
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int i = 0;
   HYPRE_Int j = 0;
   HYPRE_Int k = 0;
   HYPRE_Int ierr;
  HYPRE_Real* Hm = hypre_CTAlloc(HYPRE_Real, (d)*(d+1), HYPRE_MEMORY_HOST);
  HYPRE_Real* Qm = hypre_CTAlloc(HYPRE_Real, (d+1)*(num_rows), HYPRE_MEMORY_HOST);
  for(k=0; k < d+1; k++) {
    hypre_Memcpy(&Qm[k*num_rows], hypre_VectorData(hypre_ParVectorLocalVector(Q[k])), sizeof(HYPRE_Real) * num_rows, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
  }


  HYPRE_Real* AQ = hypre_CTAlloc(HYPRE_Real, (d)*(num_rows), HYPRE_MEMORY_HOST);

  hypre_ParVector* Qtmp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
      hypre_ParCSRMatrixGlobalNumRows(A),
      hypre_ParCSRMatrixRowStarts(A)
      );
  hypre_ParVectorInitialize(Qtmp);
  hypre_ParVectorSetPartitioningOwner(Qtmp,0);
  for(k=0; k < d+1; k++) {
    hypre_Memcpy((void*) &Qm[k*num_rows], hypre_VectorData(hypre_ParVectorLocalVector(Q[k])), sizeof(HYPRE_Real) * num_rows, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
  }

  for(k=0; k < d; k++) {
    hypre_ParCSRMatrixMatvec(1, A, Q[k], 0, (void*) Qtmp);
    hypre_Memcpy((void*)&AQ[k*num_rows], hypre_VectorData(hypre_ParVectorLocalVector(Qtmp)), sizeof(HYPRE_Real) * num_rows, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

  }
  for(k=0; k < d+1; k++) {
    for(j=0; j < d; j++) {
      Hm[k+j*(d+1)] = H[k][j];
    }
  }

#if GMRES_DEVELOPMENT
  printf("Hm\n");
  PrintMatrix(d+1, d, Hm);

  printf("Qm\n");
  PrintMatrix(num_rows, d+1, Qm);
#endif

  HYPRE_Int dpo = d+1;
  HYPRE_Real* QH = hypre_CTAlloc(HYPRE_Real, (d)*(num_rows), HYPRE_MEMORY_HOST);
  HYPRE_Real done = 1.0;
  HYPRE_Real dzero = 0.0;
  hypre_dgemm("N","N", &num_rows, &d, &dpo, &done, Qm, &num_rows, Hm, &dpo, &dzero, QH, &num_rows);

#if GMRES_DEVELOPMENT
  printf("QH - AQ: %f\n", QH[0] - AQ[0]);
  PrintMatrix(num_rows, d, AQ);
  printf("\n\n\n");

  PrintMatrix(num_rows, d, QH);
  printf("\n\n\n");
#endif

#if GMRES_DEBUG
  HYPRE_Int errs = 0;
  for(k=0; k < num_rows;k++) {
    for(j=0; j < d;j++) {
      printf("%e ",fabs(QH[j*num_rows + k]-AQ[j*num_rows + k]));
      if(fabs(QH[j*num_rows + k]-AQ[j*num_rows + k]) > 1e-13) {
        errs++;
      }
    }
    printf("\n");
  }
  assert(fabs(QH[0] - AQ[0]) < 1e-13);
  assert(errs == 0);
#endif
  hypre_ParVectorDestroy(Qtmp);

  hypre_Free(Hm, HYPRE_MEMORY_HOST);
  hypre_Free(Qm, HYPRE_MEMORY_HOST);
  hypre_Free(QH, HYPRE_MEMORY_HOST);
  hypre_Free(AQ, HYPRE_MEMORY_HOST);
  return 0;
}



HYPRE_Int hypre_ParCSRRelax_GMRES_Setup(hypre_ParCSRMatrix *A, /* matrix to relax with */
    HYPRE_Int degree,
    HYPRE_Real **coefs_ptr)
{
  assert(A != NULL);
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
  HYPRE_Real      *A_diag_data  = hypre_CSRMatrixData(A_diag);
  HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);

  /* H will be global, Q local */
  HYPRE_Real** H = hypre_CTAlloc(HYPRE_Real*, degree+1, HYPRE_MEMORY_HOST);

  /* Alloc on same memory as A */
  hypre_ParVector** Q = hypre_CTAlloc(hypre_ParVector*, degree+1, HYPRE_MEMORY_HOST);
  for(int i = 0; i < degree+1; i++) {
    Q[i] = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
        hypre_ParCSRMatrixGlobalNumRows(A),
        hypre_ParCSRMatrixRowStarts(A)
        );
    hypre_ParVectorInitialize(Q[i]);
    hypre_ParVectorSetPartitioningOwner(Q[i],0);
  }
  for(int i = 0; i < degree+1; i++) {
    H[i] = hypre_CTAlloc(HYPRE_Real, degree, HYPRE_MEMORY_HOST);
  }

  hypre_ParVector* b =   
    hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
        hypre_ParCSRMatrixGlobalNumRows(A),
        hypre_ParCSRMatrixRowStarts(A));
  hypre_ParVectorInitialize(b);
  hypre_ParVectorSetPartitioningOwner(b,0);
  HYPRE_ParVectorSetRandomValues(b, 22775);

  hypre_ParCSRConstructArnoldi(A, b, degree, Q, H);
  hypre_ParCSRVerifyArnoldi(A,b,degree,Q,H);

  hypre_assert(false && "Not yet implemented");
#if GMRES_DEVELOPMENT

  HYPRE_Real* Hcpy = hypre_CTAlloc(HYPRE_Real,  degree*degree, HYPRE_MEMORY_HOST);

  hypre_Memcpy(Hcpy, H, sizeof(HYPRE_Real)*degree*degree, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

  HYPRE_Real* f = hypre_CTAlloc(HYPRE_Real,  degree, HYPRE_MEMORY_HOST);
  HYPRE_Real* ed = hypre_CTAlloc(HYPRE_Real,  degree, HYPRE_MEMORY_HOST);

  ed[degree-1] = 1.0;
  f[degree-1] = 1.0;

  /* H is COL MAJOR, but we want to solve with H^T */
  HYPRE_Int ione = 1;

  HYPRE_Int* piv = hypre_CTAlloc(HYPRE_Int,  degree, HYPRE_MEMORY_HOST);

  /* H is hessenberg, so a more intelligent solver could be used. However, 
   * degree will be small, so just using a general solver. */

  hypre_dgesvd(LAPACK_ROW_MAJOR,  &degree, &ione, H, &degree, piv, f, &degree);

  /* H <- H(d+1,d) + f * e' */
  hypre_dger(&degree, &degree, H[(degree+1)*degree], f, e);


  //hypre_dggev(); <- coefs
  //reorder





#endif



  HYPRE_Real* coefs = hypre_CTAlloc(HYPRE_Real,  degree+1, HYPRE_MEMORY_HOST);
  *coefs_ptr = coefs;

  HYPRE_Free(e, HYPRE_MEMORY_HOST);
  HYPRE_Free(piv, HYPRE_MEMORY_HOST);

  //Construct arnoldi
  //

  return hypre_error_flag;
}
