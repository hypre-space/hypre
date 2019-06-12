#include <omp.h>
#include "spmv.h"

/*----------------------------------------------------*/
void spmv_csr_cpu(struct csr_t *csr, REAL *x, REAL *y) {
/*------------- CPU CSR SpMV kernel */
  double t1, t2;
  t1 = wall_timer();
  for (int ii=0; ii<REPEAT; ii++) {
    //memset(y, 0, csr->n*sizeof(REAL));
    #pragma omp parallel for schedule(static)
    for (int i=0; i<csr->nrows; i++) {
      REAL r = 0.0;
      for (int j=csr->ia[i]; j<csr->ia[i+1]; j++)
        r += csr->a[j]*x[csr->ja[j]];
      y[i] = r;
    }
  }
  t2 = wall_timer() - t1;
/*--------------------------------------------------*/
  printf("\n=== [CPU] CSR Kernel ===\n");
  //printf("  number of threads %d\n", omp_get_num_threads());
  printf("  %.2f ms, %.2f GFLOPS \n",
  t2*1e3, 2*(csr->ia[csr->nrows])/t2/1e9*REPEAT);
}

/*-----------------------------------------*/
int main (int argc, char **argv) {
/*-----------------------------------------*
 *           y = A * x
 *     CPU CSR        kernel
 *     GPU CSR        kernel
 *-----------------------------------------*/
  int i, n, nx=32, ny=32, nz=32, npts=7, flg=0, mm=1;
  REAL *x, *y0, *y;
  double e2, e3;
  struct coo_t coo;
  struct csr_t csr;
  hypre_CSRMatrix hypre_csr;

  char fname[2048];
/*-----------------------------------------*/
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg) {
    printf("Usage: ./testL.ex -nx [int] -ny [int] -nz [int] -npts [int] -mat fname -mm [int]\n");
    return 0;
  }
  srand (SEED);
/*------------ Init GPU */
  cuda_init(argc, argv);
/*------------ output header */
  print_header();
/*------------ cmd-line input */
  findarg("nx", INT, &nx, argc, argv);
  findarg("ny", INT, &ny, argc, argv);
  findarg("nz", INT, &nz, argc, argv);
  findarg("npts", INT, &npts, argc, argv);
  flg = findarg("mat", STR, fname, argc, argv);
  findarg("mm", INT, &mm, argc, argv);
/*---------- Read from Martrix Market file */
  if (flg == 1) {
    read_coo_MM(fname, mm, 0, &coo);
  } else {
    lapgen(nx, ny, nz, &coo, npts);
  }
/*-----------------------------------------*/
  n = coo.nrows;
  x  = (REAL *) malloc(n*sizeof(REAL));
  y0 = (REAL *) malloc(n*sizeof(REAL));
  y  = (REAL *) malloc(n*sizeof(REAL));
/*---------- randomly init x */
  for (i=0; i<n; i++)
    x[i] = rand() / (RAND_MAX + 1.0);
/*---------- convert COO to CSR */
  coo_to_csr(0, &coo, &csr);
/*---------- CPU SpMV CSR kernel */
  spmv_csr_cpu(&csr, x, y0);
/*---------- GPU SpMV CSR-vector kernel */
  spmv_csr_vector(&csr, x, y);
  e2 = error_norm(y0, y, n);
  fprintf(stdout, "err norm %.2e\n", e2);
/*---------- GPU SpMV CUSPARSE CSR kernel */
  spmv_cusparse_csr(&csr, x, y);
  e3 = error_norm(y0, y, n);
  fprintf(stdout, "err norm %.2e\n", e3);
/*---------- check error */
  cuda_check_err();
/*---------- Done, free */
  FreeCOO(&coo);  FreeCSR(&csr);
  free(x);   free(y0);  free(y);
}

