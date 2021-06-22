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
#include "cblas.h"
//
#include "lapack.h"

/* GMRES_DEVELOPMENT will be removed by the time development is finished/merged*/

#define GMRES_DEVELOPMENT 1
#define GMRES_DEBUG 0

#if GMRES_DEVELOPMENT
void PrintVector(hypre_ParVector* x) {

      HYPRE_Real* tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
      HYPRE_Int n= hypre_VectorSize(hypre_ParVectorLocalVector(x));
      for(int k = 0; k < n; k++) {
        printf("%f\n", tmp_data[k]);
      }
}

void Printvector(int n, double* x) {
  for(int i = 0; i < n; i++) {
      printf("%f\n", x[i]);
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


void find_next_lambda(int neigs, double* eig_real, double* eig_imag, double* out_real, double* out_imag, int d)
{
  double maxval = 0;
  double maxind = 0;
  double res_real = 0;
  double res_imag = 0;
  for(int i = 0; i < neigs; i++) {
    double pd = 0;
    for(int j = 0; j <= d; j++) {
#if GMRES_DEBUG
      printf("%f %f | %f %f \n", eig_real[i], eig_imag[j], out_real[j], out_imag[j]);
#endif
      //printf("%f %f\n", pow(eig_real[i]-out_real[j],2), pow(eig_imag[i]-out_imag[j],2));
      //printf("%f\n", sqrt(pow(eig_real[i]-out_real[j],2)+ pow(eig_imag[i]-out_imag[j],2)));
      //printf("%f\n", log(sqrt(pow(eig_real[i]-out_real[j],2)+ pow(eig_imag[i]-out_imag[j],2))));
      double diff = sqrt(pow(eig_real[i]-out_real[j],2) + pow(eig_imag[i]-out_imag[j],2));
#if GMRES_DEBUG
      printf("i: %i, j: %i, diff: %f\n", i, j, diff);
#endif
      pd = pd + log(diff);
    }
#if GMRES_DEBUG
    printf("i: %i pd: %f, maxval: %f\n", i, pd, maxval);
#endif
    if((pd > maxval) && (eig_imag[i] >= 0)) {
      maxind = i;
      maxval = pd;
      res_real = eig_real[i];
      res_imag = eig_imag[i];
    }
  }
  out_real[d+1] = res_real;
  out_imag[d+1] = res_imag;
}


void leja_ordereing(int neigs, double* eig_real, double* eig_imag, double* out_real, double* out_imag)
{
  double maxabs = 0;
  int ind_maxabs = -1;
  for(int i = 0; i < neigs; i++) {
    double absval = sqrt(eig_real[i]*eig_real[i] + eig_imag[i]*eig_imag[i]);
    if((absval > maxabs) && (eig_imag[i] >= 0)) {
      ind_maxabs = i;
      maxabs = absval;
    }
  }
  out_real[0] = eig_real[ind_maxabs];
  out_imag[0] = eig_imag[ind_maxabs];

  for(int i = 0; i < neigs-1; i++) {
    if(out_imag[i] > 0) {
#if GMRES_DEBUG
      printf("Forcing next %f %f \n", out_real[i], out_imag[i]);
#endif
      out_real[i+1] = out_real[i];
      out_imag[i+1] = -out_imag[i];
    }
    else
    {
#if GMRES_DEBUG
      printf("Finding next %f %f | i: %i \n", out_real[i], out_imag[i], i);
#endif
     find_next_lambda(neigs, eig_real, eig_imag, out_real, out_imag, i);
#if GMRES_DEBUG
      printf("Found next %f %f \n", out_real[i+1], out_imag[i+1]);
#endif
    }
  }

}

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
#if GMRES_DEBUG
   PrintVector(Q[0]);
#endif
   while(i < d)
   {
     i++;
#if GMRES_DEBUG
     PrintVector(Q[i-1]);
#endif
     hypre_ParCSRMatrixMatvec(1, A, Q[i-1], 0, Q[i]);
#if GMRES_DEBUG
     PrintVector(Q[i]);
#endif
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

#if GMRES_DEBUG
      PrintVector(Q[i]);
#endif
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

#if GMRES_DEBUG
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

#if GMRES_DEBUG
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
    HYPRE_Real **coefs_real_ptr,
    HYPRE_Real **coefs_imag_ptr
    )
{
  assert(A != NULL);
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
  HYPRE_Real      *A_diag_data  = hypre_CSRMatrixData(A_diag);
  HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);

  HYPRE_Int ierr;

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
  assert(sizeof(HYPRE_Real) == sizeof(double));

  hypre_ParVector* b =   
    hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
        hypre_ParCSRMatrixGlobalNumRows(A),
        hypre_ParCSRMatrixRowStarts(A));
  hypre_ParVectorInitialize(b);
  hypre_ParVectorSetPartitioningOwner(b,0);
  HYPRE_ParVectorSetRandomValues(b, 22775);

  hypre_ParCSRConstructArnoldi(A, b, degree, Q, H);
  hypre_ParCSRVerifyArnoldi(A,b,degree,Q,H);

#if !GMRES_DEVELOPMENT
  hypre_assert(0 && "Not yet implemented");
#endif
#if GMRES_DEVELOPMENT

  HYPRE_Real* Hm = hypre_CTAlloc(HYPRE_Real, (degree)*(degree), HYPRE_MEMORY_HOST);
  HYPRE_Int k = 0;
  HYPRE_Int j = 0;
  for(k=0; k < degree; k++) {
    for(j=0; j < degree; j++) {
      Hm[k+j*(degree)] = H[k][j];
    }
  }
#if GMRES_DEBUG
  PrintMatrix(degree, degree, Hm);
  printf("%f\n",H[degree][degree-1]);
#endif

  HYPRE_Real* Htrans = hypre_CTAlloc(HYPRE_Real,  degree*degree, HYPRE_MEMORY_HOST);

  //hypre_Memcpy(Hcpy, Hm, sizeof(HYPRE_Real)*degree*degree, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
  /* Transpose and copy */
  cblas_domatcopy(CblasColMajor, CblasTrans, degree, degree, 1.0f, Hm, degree, Htrans, degree);

  HYPRE_Real* f = hypre_CTAlloc(HYPRE_Real,  degree, HYPRE_MEMORY_HOST);
  HYPRE_Real* ed = hypre_CTAlloc(HYPRE_Real,  degree, HYPRE_MEMORY_HOST);

  ed[degree-1] = 1.0;
  f[degree-1] = 1.0;

  /* H is COL MAJOR, but we want to solve with H^T */
  HYPRE_Int ione = 1;
  HYPRE_Int inone = -1;

  HYPRE_Int* piv = hypre_CTAlloc(HYPRE_Int,  degree, HYPRE_MEMORY_HOST);

  /* H is hessenberg, so a more intelligent solver could be used. However, 
   * degree will be small, so just using a general solver. */

  /* f := H(1:d,1:d) \ e_{degree} */
  /* H is actually n+1xn, but we now will only use the top nxn block, so we let LDA = degree */
  hypre_dgetrf(&degree, &degree, Htrans, &degree, piv, &ierr);
  hypre_dgetrs("N", &degree, &ione, Htrans, &degree, piv, f, &degree, &ierr);
#if GMRES_DEBUG
  printf("F: \n");
  Printvector(degree, f);
#endif
  //hypre_dgesv(&degree, &ione, Hcpy, &degree, piv, f, &degree, &ierr);
  if(ierr != 0) { printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__); }

  /* H <- H(d+1,d) + f * e' */
  /* H is actually n+1xn, but we now will only use the top nxn block, so we let LDA = degree */
  ierr = hypre_dger(&degree, &degree, &H[degree][degree-1], f, &ione, ed, &ione, Hm, &degree);
  if(ierr != 0) { printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__); }

#if GMRES_DEBUG
  printf("Blah: \n");
  PrintMatrix(degree, degree, Hm);
#endif


  HYPRE_Real* harmonics_real = hypre_CTAlloc(HYPRE_Real,  degree, HYPRE_MEMORY_HOST);
  HYPRE_Real* harmonics_imag = hypre_CTAlloc(HYPRE_Real,  degree, HYPRE_MEMORY_HOST);

  HYPRE_Int lwork;
  HYPRE_Real wkopt;
#if GMRES_DEBUG
  PrintMatrix(degree, degree, Hm);
#endif

  /* Hm should still be upper hessenberg */
  for(int i = 0; i < degree; i++) {
    for(j = 0; j < degree; j++) {
      if (j-1 > i) {
        assert(Hm[i*degree+j] == 0);
      }
    }
  }

  char cN = 'N';
  char cE = 'E';

  /*hypre_dhseqr(&cE, &cN, &degree, 1, &degree, Hm, &degree, harmonics_real, harmonics_imag, NULL, &ione, &wkopt, &inone, &ierr);
  if(ierr != 0) { printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__); }
  lwork = (HYPRE_Int) wkopt;
  HYPRE_Real* work = hypre_CTAlloc(HYPRE_Real,  lwork, HYPRE_MEMORY_HOST);
  hypre_dhseqr(&cE, &cN, &degree, 1, &degree, Hm, &degree, harmonics_real, harmonics_imag, NULL, &ione, work, &lwork, &ierr);
  if(ierr != 0) { printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__); }*/

  LAPACK_dgeev(&cN, &cN,&degree,Hm, &degree, harmonics_real, harmonics_imag, NULL, &ione, NULL, &ione, &wkopt,  &inone, &ierr);

  lwork = (HYPRE_Int) wkopt;
  HYPRE_Real* work = hypre_CTAlloc(HYPRE_Real,  lwork, HYPRE_MEMORY_HOST);

  LAPACK_dgeev(&cN, &cN,&degree,Hm, &degree, harmonics_real, harmonics_imag, NULL, &ione, NULL, &ione, work,  &lwork, &ierr);
  if(ierr != 0) { printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__); }

#if GMRES_DEBUG
  for(int i=0; i < degree; i++) {
    printf("%.18f, %.18f\n", harmonics_real[i], harmonics_imag[i]);
  }
#endif

  HYPRE_Real* ordered_real = hypre_CTAlloc(HYPRE_Real,  degree, HYPRE_MEMORY_HOST);
  HYPRE_Real* ordered_imag = hypre_CTAlloc(HYPRE_Real,  degree, HYPRE_MEMORY_HOST);

  leja_ordereing(degree, harmonics_real, harmonics_imag, ordered_real, ordered_imag);
#if GMRES_DEBUG
  printf("Leja\n");

  for(int i=0; i < degree; i++) {
    printf("%.16f, %.16f\n", ordered_real[i], ordered_imag[i]);
  }
#endif

#endif



  *coefs_real_ptr = ordered_real;
  *coefs_imag_ptr = ordered_imag;

  hypre_Free(ed, HYPRE_MEMORY_HOST);
  hypre_Free(piv, HYPRE_MEMORY_HOST);

  //Construct arnoldi
  //

  return hypre_error_flag;
}

HYPRE_Int hypre_ParCSRRelax_GMRES_Solve(hypre_ParCSRMatrix *A, /* matrix to relax with */
                            hypre_ParVector *f,    /* right-hand side */
                            HYPRE_Real *coefs_real,
                            HYPRE_Real *coefs_imag,
                            HYPRE_Int order,            /* polynomial order */
                            hypre_ParVector *prod,
                            hypre_ParVector *p,
                            hypre_ParVector *tmp)

{
   // Topy prod <- x
   HYPRE_ParVectorCopy(f, prod);
   int i = 0;
   while (i < order)
   {
      if (coefs_imag[i] == 0)
      {
         HYPRE_Real alpha = 1 / coefs_real[i];
         hypre_ParVectorAxpy(alpha, prod, p);
         /* prod <- prod - alpha*mv(prod) */
         hypre_ParCSRMatrixMatvec(-alpha, A, prod, 1.0, prod);
         i++;
      }
      else
      {
         HYPRE_Real a = coefs_real[i];
         HYPRE_Real b = coefs_imag[i];

         hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, prod, 2 * a, prod, tmp);
         HYPRE_Real alpha = 1 / (a * a + b * b);

         hypre_ParVectorAxpy(alpha, tmp, p);
         if (i < order - 2)
         {
            hypre_ParCSRMatrixMatvec(-alpha, A, tmp, 1.0, prod);
         }
         i += 2;
      }
   }
   if (coefs_imag[order-1] == 0)
   {
      hypre_ParVectorAxpy(1.0 / coefs_real[order-1], prod, p);
   }
   return hypre_error_flag;
}
