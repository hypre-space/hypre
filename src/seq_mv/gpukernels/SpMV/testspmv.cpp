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
    for (int i=0; i<csr->n; i++) {
      REAL r = 0.0;
      for (int j=csr->ia[i]; j<csr->ia[i+1]; j++)
        r += csr->a[j-1]*x[csr->ja[j-1]-1];
      y[i] = r;
    }
  }
  t2 = wall_timer() - t1;
/*--------------------------------------------------*/
  printf("\n=== [CPU] CSR Kernel ===\n");
  //printf("  number of threads %d\n", omp_get_num_threads());
  printf("  %.2f ms, %.2f GFLOPS \n",
  t2*1e3, 2*(csr->nnz)/t2/1e9*REPEAT);
}

/*-----------------------------------------*/
int main (int argc, char **argv) {
/*-----------------------------------------*
 *           y = A * x
 *     CPU CSR        kernel
 *     GPU JAD        kernel
 *-----------------------------------------*/
  int i, n, ret, nx=32, ny=32, nz=32, npts=7, flg=0, mm=1;
  REAL *x, *y0, *y;
  double e1, e2, e3, e4;
  struct coo_t coo;
  struct csr_t csr;
  struct jad_t jad;
  double t1, t2;
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
    read_coo_MM(&coo, fname, mm);
  } else {
    lapgen(nx, ny, nz, &coo, npts);
  }
/*-----------------------------------------*/
  n = coo.n;
  x  = (REAL *) malloc(n*sizeof(REAL));
  y0 = (REAL *) malloc(n*sizeof(REAL));
  y  = (REAL *) malloc(n*sizeof(REAL));
/*---------- randomly init x */
  for (i=0; i<n; i++)
    x[i] = rand() / (RAND_MAX + 1.0);
/*---------- convert COO to CSR */
  COO2CSR(&coo, &csr);
/*---------- CPU SpMV CSR kernel */
  spmv_csr_cpu(&csr, x, y0);
/*---------- GPU SpMV CSR-vector kernel */
  spmv_csr_vector(&csr, x, y);
  e2 = error_norm(y0, y, n);
  fprintf(stdout, "err norm %.2e\n", e2);
/*---------- convert CSR to JAD */
  t1 = wall_timer();
  CSR2JAD(&csr, &jad);
  t1 = wall_timer() - t1;
  fprintf(stdout, "JAD conversion time %f\n", t1);
/*---------- GPU SpMV JAD kernel */
  spmv_jad(&jad, x, y);
  e3 = error_norm(y0, y, n);
  fprintf(stdout, "err norm %.2e\n", e3);
/*---------- GPU SpMV CUSPARSE CSR kernel */
  spmv_cusparse_csr(&csr, x, y);
  e3 = error_norm(y0, y, n);
  fprintf(stdout, "err norm %.2e\n", e3);
/*---------- GPU SpMV CUSPARSE HYB kernel */
  spmv_cusparse_hyb(&csr, x, y);
  e3 = error_norm(y0, y, n);
  fprintf(stdout, "err norm %.2e\n", e3);
/*---------- check error */
  cuda_check_err();
/*---------- Done, free */  
  FreeCOO(&coo);  FreeCSR(&csr);  FreeJAD(&jad);  
  free(x);   free(y0);  free(y);
}

