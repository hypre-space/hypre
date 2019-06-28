#include <omp.h>
#include "spmv.h"

/*-----------------------------------------*/
HYPRE_Int main (HYPRE_Int argc, char **argv)
{
   /*-----------------------------------------*
    *           y = A * x
    *     CPU CSR        kernel
    *     GPU CSR        kernel
    *-----------------------------------------*/
   HYPRE_Int i, n, nx=32, ny=32, nz=32, npts=7, flg=0, mm=1;
   HYPRE_Real *x, *y0, *y;
   HYPRE_Real e2, e3;
   struct coo_t coo;
   hypre_CSRMatrix *csr;

   char fname[2048];
   /*-----------------------------------------*/
   flg = findarg("help", NA, NULL, argc, argv);
   if (flg)
   {
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
   if (flg == 1)
   {
      read_coo_MM(fname, mm, 0, &coo);
   } else
   {
      lapgen(nx, ny, nz, &coo, npts);
   }
   /*-----------------------------------------*/
   n = coo.nrows;
   x  = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   y0 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   y  = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   /*---------- randomly init x */
   for (i=0; i<n; i++)
      x[i] = rand() / (RAND_MAX + 1.0);
   /*---------- convert COO to CSR */
   coo_to_csr(0, &coo, &csr);
   /*---------- CPU SpMV CSR kernel */
   spmv_csr_cpu(csr, x, y0);
   /*---------- GPU SpMV CSR-vector kernel */
   spmv_csr_vector(csr, x, y);
   e2 = error_norm(y0, y, n);
   fprintf(stdout, "err norm %.2e\n", e2);
   /*---------- GPU SpMV CUSPARSE CSR kernel */
   spmv_cusparse_csr(csr, x, y);
   e3 = error_norm(y0, y, n);
   fprintf(stdout, "err norm %.2e\n", e3);
   /*---------- check error */
   cuda_check_err();
   /*---------- Done, free */
   hypre_CSRMatrixDestroy(csr);
   FreeCOO(&coo);
   free(x);   free(y0);  free(y);
}
