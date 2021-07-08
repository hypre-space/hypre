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

#include "_hypre_blas.h"
#include "_hypre_lapack.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "float.h"
//#include "cblas.h"
//
//#include "lapack.h"
//#include "mkl.h"

/* GMRES_DEVELOPMENT will be removed by the time development is finished/merged*/

#define GMRES_DEVELOPMENT 1
#define GMRES_DEBUG 1

#if GMRES_DEVELOPMENT
void
fPrintCSRMatrixCOO(FILE* fd, hypre_CSRMatrix *A)
{
  HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
  HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);
  HYPRE_Int* I = hypre_CSRMatrixI(A);
  HYPRE_Int* J = hypre_CSRMatrixJ(A);
  HYPRE_Real* values = hypre_CSRMatrixData(A);
  HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(A);
  fprintf(fd, "%i %i %i %i\n", 0, num_rows, 0, num_cols);
  for(HYPRE_Int i = 0; i < num_rows; i++) {
    printf("%i ", I[i]);
  }
  printf("\n");
  for(HYPRE_Int i = 0; i < nnz; i++) {
    printf("%i ", J[i]);
  }
  printf("\n");
  for(HYPRE_Int i = 0; i < nnz; i++) {
    printf("%f ", values[i]);
  }
  printf("\n");

  for(HYPRE_Int i = 0; i <num_rows; i++) {
    HYPRE_Int num_nnz_in_row = I[i+1]-I[i];
      for(HYPRE_Int k = 0; k <num_nnz_in_row; k++) {
          fprintf(fd, "%i %i %f\n", i+1,J[I[i]+k]+1, values[I[i]+k]);
      }
    }
}

void
fPrintCSRMatrixCSR(FILE* fd, hypre_CSRMatrix *A)
{
  HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
  HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);
  HYPRE_Int* I = hypre_CSRMatrixI(A);
  HYPRE_Int* J = hypre_CSRMatrixJ(A);
  HYPRE_Real* values = hypre_CSRMatrixData(A);
  HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(A);
  fprintf(fd, "%i\n", num_rows);
  for(HYPRE_Int i = 0; i < num_rows+1; i++) {
    fprintf(fd, "%i ", I[i]+1);
  }
  fprintf(fd,"\n");
  for(HYPRE_Int i = 0; i < nnz; i++) {
    fprintf(fd,"%i ", J[i]+1);
  }
  fprintf(fd,"\n");
  for(HYPRE_Int i = 0; i < nnz; i++) {
    fprintf(fd,"%f ", values[i]);
  }
  printf("\n");
}

void
fPrintCSRMatrixDense(FILE* fd, hypre_CSRMatrix *A)
{
  HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
  HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);
  HYPRE_Int* I = hypre_CSRMatrixI(A);
  HYPRE_Int* J = hypre_CSRMatrixJ(A);
  HYPRE_Real* values = hypre_CSRMatrixData(A);
  HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(A);
  for(HYPRE_Int i = 0; i < num_rows; i++) {
    printf("%i ", I[i]);
  }
  printf("\n");
  for(HYPRE_Int i = 0; i < nnz; i++) {
    printf("%i ", J[i]);
  }
  printf("\n");
  for(HYPRE_Int i = 0; i < nnz; i++) {
    printf("%f ", values[i]);
  }
  printf("\n");
  for(HYPRE_Int i = 0; i <num_rows; i++) {
    HYPRE_Int num_nnz_in_row = I[i+1]-I[i];

    for(HYPRE_Int j = 0; j < num_cols; j++) {
      HYPRE_Int found = 0;

      for(HYPRE_Int k = 0; k <num_nnz_in_row; k++) {
        if(j == J[I[i]+k]) {
          fprintf(fd, "%.2e", (values[I[i] + k]));
          found = 1;
          break;
        }
      }

      if(!found) {
        fprintf(fd, "%.2e",0.0);
      }

      if(j < num_cols-1)
      {
        fprintf(fd, ",");
      }
    }
    fprintf(fd, "\n");
  }
}

void
PrintVector(hypre_ParVector *x)
{

   HYPRE_Real *tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   HYPRE_Int   n        = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   HYPRE_Int k;
   for (k = 0; k < n; k++)
   {
      printf("%.15f\n", tmp_data[k]);
   }
}

void
Printvector(HYPRE_Int n, HYPRE_Real *x)
{
   HYPRE_Int i;
   for (i = 0; i < n; i++)
   {
      printf("%f\n", x[i]);
   }
}

void
PrintMatrix(HYPRE_Int m, HYPRE_Int n, HYPRE_Real *x)
{
   HYPRE_Int i,j;
   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
      {
         printf("%.15f ", x[i + j * m]);
      }
      printf("\n");
   }
}
#endif

HYPRE_Int
find_next_lambda(HYPRE_Int   neigs,
                 HYPRE_Real *eig_real,
                 HYPRE_Real *eig_imag,
                 HYPRE_Real *out_real,
                 HYPRE_Real *out_imag,
                 HYPRE_Int   d)
{
   HYPRE_Real maxval   = -DBL_MAX;
   HYPRE_Real res_real = -1;
   HYPRE_Real res_imag = -1;
   HYPRE_Int i,j;
   for (i = 0; i < neigs; i++)
   {
      HYPRE_Real pd = 0;

      for (j = 0; j <= d; j++)
      {
         HYPRE_Real diff = sqrt(pow(eig_real[i] - out_real[j], 2) + pow(eig_imag[i] - out_imag[j], 2));
         pd              = pd + log(diff);
      }

      if ((pd > maxval) && (eig_imag[i] >= 0))
      {
         maxval   = pd;
         res_real = eig_real[i];
         res_imag = eig_imag[i];
      }
   }
   out_real[d + 1] = res_real;
   out_imag[d + 1] = res_imag;

   return hypre_error_flag;
}

HYPRE_Int
leja_ordering(HYPRE_Int neigs, HYPRE_Real *eig_real, HYPRE_Real *eig_imag, HYPRE_Real *out_real, HYPRE_Real *out_imag)
{
   HYPRE_Real maxabs     = 0;
   HYPRE_Int  ind_maxabs = -1;
   HYPRE_Int i;
   for (i = 0; i < neigs; i++)
   {
      HYPRE_Real absval = sqrt(eig_real[i] * eig_real[i] + eig_imag[i] * eig_imag[i]);
      if ((absval > maxabs) && (eig_imag[i] >= 0))
      {
         ind_maxabs = i;
         maxabs     = absval;
      }
   }
   out_real[0] = eig_real[ind_maxabs];
   out_imag[0] = eig_imag[ind_maxabs];

   for (i = 0; i < neigs - 1; i++)
   {
      if (out_imag[i] > 0)
      {
         out_real[i + 1] = out_real[i];
         out_imag[i + 1] = -out_imag[i];
      }
      else
      {
         find_next_lambda(neigs, eig_real, eig_imag, out_real, out_imag, i);
      }
   }

   return hypre_error_flag;
}

/**
 * @brief Uses Arnoldi MGS to generate H,Q such that Q_m^TAQ_m=H_m,
 * Algorithm 6.2 in Saad
 *
 * @param[in] A Input matrix used for matvecs
 * @param[in] b Vector used for Krylov subspace
 * @param[in] d Size of subspace
 * @param[in,out] Q Resulting parallel vectors, must be preallocated
 * @param[in,out] H Resulting vectors of Hessenberg Matrix, must be preallocated (duplicated on each rank)
 * */
HYPRE_Int
hypre_ParCSRConstructArnoldi(
    hypre_ParCSRMatrix *A, hypre_ParVector *b, HYPRE_Int d, hypre_ParVector **Q, HYPRE_Real **H)
{
   HYPRE_Int  i = 0;
   HYPRE_Int  j = 0;
   HYPRE_Real norm;

   /* Copy b to Q[0] */
   hypre_ParVectorCopy(b, Q[0]);

   /* Normalize input vector */
   HYPRE_ParVectorInnerProd(Q[0], Q[0], &norm);
   norm = 1. / sqrt(norm);
   HYPRE_ParVectorScale(norm, Q[0]);

   while (i < d)
   {
      i++;
      hypre_ParCSRMatrixMatvec(1, A, Q[i - 1], 0, Q[i]);
      for (j = 0; j < i; j++)
      {
         HYPRE_ParVectorInnerProd(Q[j], Q[i], &H[j][i - 1]);
         HYPRE_ParVectorAxpy(-H[j][i - 1], Q[j], Q[i]);
      }
      HYPRE_ParVectorInnerProd(Q[i], Q[i], &norm);
      norm        = sqrt(norm);
      H[i][i - 1] = norm;
      if (norm != 0.0)
      {
         norm = 1.0 / norm;
         HYPRE_ParVectorScale(norm, Q[i]);
      }
   }

   return hypre_error_flag;
}

/*
 * @brief Used to verify the correctness of the arnoldi implementation
 * Assumes serial (all data on single process)
 * Assumes A is on host
 */
HYPRE_Int
hypre_ParCSRVerifyArnoldi(hypre_ParCSRMatrix *A, hypre_ParVector *b, HYPRE_Int d, hypre_ParVector **Q, HYPRE_Real **H)
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int j        = 0;
   HYPRE_Int k        = 0;


   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   fprintf(stderr, "MEMORY LOCATION: %i / %i\n", memory_location, HYPRE_MEMORY_HOST);
   assert(memory_location == HYPRE_MEMORY_HOST);

   /* Allocate space for linear arrays of Hm and Qm */

   HYPRE_Real *Hm = hypre_CTAlloc(HYPRE_Real, (d) * (d + 1), HYPRE_MEMORY_HOST);
   HYPRE_Real *Qm = hypre_CTAlloc(HYPRE_Real, (d + 1) * (num_rows), HYPRE_MEMORY_HOST);

   /* Fill Qm */
   for (k = 0; k < d + 1; k++)
   {
      hypre_Memcpy(&Qm[k * num_rows],
                   hypre_VectorData(hypre_ParVectorLocalVector(Q[k])),
                   sizeof(HYPRE_Real) * num_rows,
                   HYPRE_MEMORY_HOST,
                   HYPRE_MEMORY_HOST);
   }

   /* Allocate space for AQ */
   HYPRE_Real *AQ = hypre_CTAlloc(HYPRE_Real, (d) * (num_rows), HYPRE_MEMORY_HOST);

   /* Allocate space for the output of AQ Matvecs */
   hypre_ParVector *Qtmp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Qtmp);
   hypre_ParVectorSetPartitioningOwner(Qtmp, 0);

   for (k = 0; k < d + 1; k++)
   {
      hypre_Memcpy((void *)&Qm[k * num_rows],
                   hypre_VectorData(hypre_ParVectorLocalVector(Q[k])),
                   sizeof(HYPRE_Real) * num_rows,
                   HYPRE_MEMORY_HOST,
                   HYPRE_MEMORY_HOST);
   }

   for (k = 0; k < d; k++)
   {
      hypre_ParCSRMatrixMatvec(1, A, Q[k], 0, (void *)Qtmp);
      hypre_Memcpy((void *)&AQ[k * num_rows],
                   hypre_VectorData(hypre_ParVectorLocalVector(Qtmp)),
                   sizeof(HYPRE_Real) * num_rows,
                   HYPRE_MEMORY_HOST,
                   HYPRE_MEMORY_HOST);
   }

   for (k = 0; k < d + 1; k++)
   {
      for (j = 0; j < d; j++)
      {
         Hm[k + j * (d + 1)] = H[k][j];
      }
   }

#if GMRES_DEBUG
   printf("Hm\n");
   PrintMatrix(d + 1, d, Hm);
#endif

   HYPRE_Int   dpo   = d + 1;
   HYPRE_Real *QH    = hypre_CTAlloc(HYPRE_Real, (d) * (num_rows), HYPRE_MEMORY_HOST);
   HYPRE_Real  done  = 1.0;
   HYPRE_Real  dzero = 0.0;

#if GMRES_DEBUG
   printf("Num rows: %i\n", num_rows);
   printf("H: %p ", Hm);
#endif

   hypre_dgemm("N", "N", &num_rows, &d, &dpo, &done, Qm, &num_rows, Hm, &dpo, &dzero, QH, &num_rows);

#if GMRES_DEBUG
   printf("QH - AQ: %f\n", QH[0] - AQ[0]);
   PrintMatrix(num_rows, d, AQ);
   printf("\n\n\n");

   PrintMatrix(num_rows, d, QH);
   printf("\n\n\n");
#endif

#if GMRES_DEBUG
   HYPRE_Int errs = 0;
   for (k = 0; k < num_rows; k++)
   {
      for (j = 0; j < d; j++)
      {
         printf("%e ", fabs(QH[j * num_rows + k] - AQ[j * num_rows + k]));
         if (fabs(QH[j * num_rows + k] - AQ[j * num_rows + k]) > 1e-13)
         {
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

/**
 * @brief Calculates the coefficients used for a GMRES Polynomial of matrix A
 *
 * @param[in] A Matrix for which to construct the polynomial
 * @param[in] degree Degree of polynomial to construct
 * @param[out] coefs_real_ptr Real part of the coefficients
 * @param[out] coefs_imag_ptr Imaginary part of the coefficients
 *
 * @todo Do we want to allow an input vector?
   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
 *
 * */
HYPRE_Int
hypre_ParCSRRelax_GMRES_Setup(hypre_ParCSRMatrix *A, /* matrix to relax with */
                              HYPRE_Int           degree,
                              hypre_ParVector    *b,
                              HYPRE_Real        **coefs_real_ptr,
                              HYPRE_Real        **coefs_imag_ptr)
{
   assert(A != NULL);
   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   //printf("Num rows: %i cols: %i\n",num_rows,num_cols);
   assert(num_rows == num_cols);
   assert(degree != 2); //Currently there is a bug in the degree = 2 case
#if 0
   char buffer[256];
   sprintf(buffer, "mat.%i.csv", num_rows);
   fPrintCSRMatrixDense(stdout, hypre_ParCSRMatrixDiag(A));
   FILE* fd = fopen(buffer,"w");
   fPrintCSRMatrixDense(fd, hypre_ParCSRMatrixDiag(A));
   fclose(fd);

   sprintf(buffer, "mat.%i.coo", num_rows);
   fd = fopen(buffer,"w");
   fPrintCSRMatrixCOO(fd, hypre_ParCSRMatrixDiag(A));
   fclose(fd);

   sprintf(buffer, "mat.%i.csr", num_rows);
   fd = fopen(buffer,"w");
   fPrintCSRMatrixCSR(fd, hypre_ParCSRMatrixDiag(A));
   fclose(fd);
#endif

   HYPRE_Int ierr;
   HYPRE_Int i;

   /* H will be the same on every rank, Q is split between all ranks */
   HYPRE_Real **H = hypre_CTAlloc(HYPRE_Real *, degree + 1, HYPRE_MEMORY_HOST);

   /* Alloc on same memory as A */
   hypre_ParVector **Q = hypre_CTAlloc(hypre_ParVector *, degree + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i < degree + 1; i++)
   {
      Q[i] = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(Q[i]);
      hypre_ParVectorSetPartitioningOwner(Q[i], 0);
   }

   for (i = 0; i < degree + 1; i++)
   {
      H[i] = hypre_CTAlloc(HYPRE_Real, degree, HYPRE_MEMORY_HOST);
   }
   assert(sizeof(HYPRE_Real) == sizeof(HYPRE_Real));

   if(b == NULL) 
   {
     b = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                              hypre_ParCSRMatrixGlobalNumRows(A),
                                              hypre_ParCSRMatrixRowStarts(A));
    hypre_ParVectorInitialize(b);
    hypre_ParVectorSetPartitioningOwner(b, 0);
    HYPRE_ParVectorSetRandomValues(b, 1);
   }

   /* Construct the Arnoldi Basis (we only use H) */
   hypre_ParCSRConstructArnoldi(A, b, degree, Q, H);

#if GMRES_DEBUG
#if !HYPRE_USING_GPU
   //hypre_ParCSRVerifyArnoldi(A, b, degree, Q, H);
#endif
#endif

   /* Take the row major 2D H and construct a column major linear Hm */
   HYPRE_Real *Hm = hypre_CTAlloc(HYPRE_Real, (degree) * (degree), HYPRE_MEMORY_HOST);
   HYPRE_Int   k  = 0;
   HYPRE_Int   j  = 0;

   for (k = 0; k < degree; k++)
   {
      for (j = 0; j < degree; j++)
      {
         Hm[k + j * (degree)] = H[k][j];
      }
   }

   HYPRE_Real *Htrans = hypre_CTAlloc(HYPRE_Real, degree * degree, HYPRE_MEMORY_HOST);

   /* Construct a column major Htranspose */
   /* Transpose and copy */
   // cblas_domatcopy(CblasColMajor, CblasTrans, degree, degree, 1.0f, Hm, degree, Htrans, degree);
   for (k = 0; k < degree; k++)
   {
      for (j = 0; j < degree; j++)
      {
         Htrans[k * degree + j] = H[k][j];
      }
   }

   HYPRE_Real *f  = hypre_CTAlloc(HYPRE_Real, degree, HYPRE_MEMORY_HOST);
   HYPRE_Real *ed = hypre_CTAlloc(HYPRE_Real, degree, HYPRE_MEMORY_HOST);

   /* ed = I(:,d), f := H(1:d,1:d \ ed */
   ed[degree - 1] = 1.0;
   f[degree - 1]  = 1.0;

   /* H is COL MAJOR, but we want to solve with H^T */
   HYPRE_Int ione  = 1;
   HYPRE_Int inone = -1;

   HYPRE_Int *piv = hypre_CTAlloc(HYPRE_Int, degree, HYPRE_MEMORY_HOST);

   /* H is hessenberg, so a more intelligent solver could be used. However,
    * degree will be small, so just using a general solver. */

   /* f := H(1:d,1:d) \ e_{degree} */
   hypre_dgetrf(&degree, &degree, Htrans, &degree, piv, &ierr);
   hypre_dgetrs("N", &degree, &ione, Htrans, &degree, piv, f, &degree, &ierr);

#if GMRES_DEBUG
   printf("F: \n");
   Printvector(degree, f);
#endif

   // hypre_dgesv(&degree, &ione, Hcpy, &degree, piv, f, &degree, &ierr);
   if (ierr != 0)
   {
      printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__);
   }

   /* Hm <- H(d+1,d) + f * e' */
   ierr = hypre_dger(&degree, &degree, &H[degree][degree - 1], f, &ione, ed, &ione, Hm, &degree);
   if (ierr != 0)
   {
      printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__);
   }

   /* Now we want to find the harmonic eigenvalues to use as coefficients */

#if GMRES_DEBUG
   printf("Blah: \n");
   PrintMatrix(degree, degree, Hm);
#endif

   HYPRE_Real *harmonics_real = hypre_CTAlloc(HYPRE_Real, degree, HYPRE_MEMORY_HOST);
   HYPRE_Real *harmonics_imag = hypre_CTAlloc(HYPRE_Real, degree, HYPRE_MEMORY_HOST);

   HYPRE_Int  lwork;
   HYPRE_Real wkopt;

#if GMRES_DEBUG
   PrintMatrix(degree, degree, Hm);

   /* Hm should still be upper hessenberg */
   for (i = 0; i < degree; i++)
   {
      for (j = 0; j < degree; j++)
      {
         if (j - 1 > i)
         {
            assert(Hm[i * degree + j] == 0);
         }
      }
   }
#endif

   char cN = 'N';
   char cE = 'E';

   /* Calculate the eigenvalues of Hm, which is an upper Hessenberg matrix */
   hypre_dhseqr(&cE,
                &cN,
                &degree,
                &ione,
                &degree,
                Hm,
                &degree,
                harmonics_real,
                harmonics_imag,
                NULL,
                &ione,
                &wkopt,
                &inone,
                &ierr);

   if (ierr != 0)
   {
      hypre_printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__);
      hypre_assert(ierr);
   }

   lwork            = (HYPRE_Int)wkopt;
   HYPRE_Real *work = hypre_CTAlloc(HYPRE_Real, lwork, HYPRE_MEMORY_HOST);
   hypre_dhseqr(&cE,
                &cN,
                &degree,
                &ione,
                &degree,
                Hm,
                &degree,
                harmonics_real,
                harmonics_imag,
                NULL,
                &ione,
                work,
                &lwork,
                &ierr);
   if (ierr != 0)
   {
      hypre_printf("Ierr: %i, line: %i, file: %s\n", ierr, __LINE__, __FILE__);
      hypre_assert(ierr);
   }

#if GMRES_DEBUG
   for (i = 0; i < degree; i++)
   {
      printf("%.18f, %.18f\n", harmonics_real[i], harmonics_imag[i]);
   }
#endif

   /* Now that the have the eigenvalues, reorder them according to Leja ordering */

   HYPRE_Real *ordered_real = hypre_CTAlloc(HYPRE_Real, degree, HYPRE_MEMORY_HOST);
   HYPRE_Real *ordered_imag = hypre_CTAlloc(HYPRE_Real, degree, HYPRE_MEMORY_HOST);

   leja_ordering(degree, harmonics_real, harmonics_imag, ordered_real, ordered_imag);

#if GMRES_DEBUG
   printf("Leja Ordered\n");

   for (i = 0; i < degree; i++)
   {
      printf("%.16f, %.16f\n", ordered_real[i], ordered_imag[i]);
   }

   for (HYPRE_Int i = 0; i < degree; i++) {
      HYPRE_Int found_relevant = 0; 
      for (HYPRE_Int j = 0; j < degree; j++) {
         if(harmonics_real[i] == ordered_real[j] && harmonics_imag[i] == ordered_imag[j]) {
           found_relevant = 1;
         }
      }
      if(!( found_relevant)) {
        hypre_printf("COULD NOT FIND :i | %f %f \n", i, harmonics_real[i], harmonics_imag[i]);
        exit(-1);
      }
   }
#endif

   *coefs_real_ptr = ordered_real;
   *coefs_imag_ptr = ordered_imag;

   hypre_Free(ed, HYPRE_MEMORY_HOST);
   hypre_Free(piv, HYPRE_MEMORY_HOST);

   hypre_Free(harmonics_real, HYPRE_MEMORY_HOST);
   hypre_Free(harmonics_imag, HYPRE_MEMORY_HOST);


   return hypre_error_flag;
}

/* p = p(A)*x */
HYPRE_Int
apply_GMRES_poly(hypre_ParCSRMatrix *A,
                 HYPRE_Real         *coefs_real,
                 HYPRE_Real         *coefs_imag,
                 HYPRE_Int           order, /* polynomial order */
                 hypre_ParVector    *x,
                 hypre_ParVector    *tmp,
                 hypre_ParVector    *prod,
                 hypre_ParVector    *p)
{
   HYPRE_ParVectorCopy(x, prod);
#if GMRES_DEBUG
   printf("Prod\n");
   PrintVector(prod);
#endif

   HYPRE_Int i = 0;
   while (i < order)
   {
      if (coefs_imag[i] == 0)
      {
         HYPRE_Real alpha = 1 / coefs_real[i];
#if GMRES_DEBUG
         printf("Alpha: :%f\n", alpha);
#endif
         hypre_ParVectorAxpy(alpha, prod, p);
#if GMRES_DEBUG
         printf("P\n");
         PrintVector(p);
#endif
         /* prod <- prod - alpha*mv(prod) */
         hypre_ParCSRMatrixMatvec(-alpha, A, prod, 1.0, prod);
#if GMRES_DEBUG
         printf("Prod\n");
         PrintVector(prod);
#endif
         i++;
      }
      else
      {
         HYPRE_Real a = coefs_real[i];
         HYPRE_Real b = coefs_imag[i];
#if GMRES_DEBUG
         printf("a b: :%f %f\n", a, b);
#endif

         hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, prod, 2 * a, prod, tmp);
#if GMRES_DEBUG
         printf("tmp\n");
         PrintVector(tmp);
#endif

         HYPRE_Real alpha = 1 / (a * a + b * b);
#if GMRES_DEBUG
         printf("alpha: :%f \n", alpha);
#endif

         hypre_ParVectorAxpy(alpha, tmp, p);
#if GMRES_DEBUG
         printf("p\n");
         PrintVector(p);
#endif

         if (i <= order - 2)
         {
#if GMRES_DEBUG
         printf("theta: :%f \n", alpha);
         printf("pre prod:\n");
         PrintVector(prod);
         printf("tmp");
         PrintVector(tmp);
#endif
            hypre_ParCSRMatrixMatvec(-alpha, A, tmp, 1.0, prod);
#if GMRES_DEBUG
         printf("prod:\n");
         PrintVector(prod);
#endif
         }
         i += 2;
      }
   }
   if (coefs_imag[order - 1] == 0)
   {
#if GMRES_DEBUG
      printf("Extra\n");
      printf("%f \n", 1.0 / coefs_real[order - 1]);
      printf("Prod\n");
      PrintVector(prod);
#endif
      hypre_ParVectorAxpy(1.0 / coefs_real[order - 1], prod, p);

#if GMRES_DEBUG
      printf("p\n");
      PrintVector(p);
#endif
   }
   return hypre_error_flag;
}

HYPRE_Int
hypre_ParCSRRelax_GMRES_Solve(hypre_ParCSRMatrix *A, /* matrix to relax with */
                              hypre_ParVector    *f, /* right-hand side */
                              HYPRE_Real         *coefs_real,
                              HYPRE_Real         *coefs_imag,
                              HYPRE_Int           order, /* polynomial order */
                              hypre_ParVector    *u,
                              hypre_ParVector    *tmp,
                              hypre_ParVector    *r,
                              hypre_ParVector    *prod,
                              hypre_ParVector    *p)

{

   hypre_ParVectorSetConstantValues(p, 0);

   // Topy prod <- x
   hypre_ParVectorCopy(f, r);
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

   apply_GMRES_poly(A, coefs_real, coefs_imag, order, r, tmp, prod, p);

   hypre_ParVectorAxpy(1.0, p, u);
   return hypre_error_flag;
}
