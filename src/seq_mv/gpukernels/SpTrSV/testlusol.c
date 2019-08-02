#include "spkernels.h"

int main(int argc, char *argv[])
{
   /*----------------------------------- *
    *  Driver program for GPU L/U solve  *
    *  x = U^{-1} * L^{-1} * b           *
    *  Kernels provided:                 *
    *  CPU L/U solve                     *
    *  GPU L/U solve w/ level-scheduling *
    *  GPU L/U solve w/ sync-free        *
    *----------------------------------- */
   int i,n,nnz,nx=32, ny=32, nz=32, npts=7, flg=0, mm=1, dotest=0;
   HYPRE_Real *h_b,*h_x0,*h_x1,*h_x2,*h_x3,*h_x4,*h_x5,*h_z;
   struct coo_t h_coo;
   hypre_CSRMatrix *h_csr;
   double e1,e2,e3,e4,e5;
   char fname[2048];
   int NTESTS = 10;
   int REP = 10;
   double err;

   /*-----------------------------------------*/
   flg = findarg("help", NA, NULL, argc, argv);
   if (flg)
   {
      printf("Usage: ./testL.ex -nx [int] -ny [int] -nz [int] -npts [int] -mat fname -mm [int] -rep [int] -dotest\n");
      return 0;
   }
   //srand (SEED);
   //srand(time(NULL));
   /*---------- Init GPU */
   cuda_init(argc, argv);
   /*---------- cmd line arg */
   findarg("nx", INT, &nx, argc, argv);
   findarg("ny", INT, &ny, argc, argv);
   findarg("nz", INT, &nz, argc, argv);
   findarg("npts", INT, &npts, argc, argv);
   flg = findarg("mat", STR, fname, argc, argv);
   findarg("mm", INT, &mm, argc, argv);
   findarg("rep", INT, &REP, argc, argv);
   dotest = findarg("dotest", NA, &dotest, argc, argv);
   /*---------- Read from Martrix Market file */
   if (flg == 1)
   {
      read_coo_MM(fname, mm, 0, &h_coo);
   } else
   {
      lapgen(nx, ny, nz, &h_coo, npts);
   }
   n = h_coo.nrows;
   nnz = h_coo.nnz;
   /*---------- COO -> CSR */
   coo_to_csr(0, &h_coo, &h_csr);
   /*--------------------- vector b */
   h_b = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   h_z = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   for (i = 0; i < n; i++)
   {
      //h_b[i] = rand() / (RAND_MAX + 1.0);
      //h_z[i] = rand() / (RAND_MAX + 1.0);
      h_b[i] = cos(i+1);
      h_z[i] = sin(i+1);
   }
   if (!dotest)
   {
      goto bench;
   }
   /*------------------------------------------------ */
   /*------------- Start testing kernels ------------ */
   /*------------------------------------------------ */
   printf("Test kernels for %d times ...\n", NTESTS);
   /*------------- CPU L/U Sol */
   h_x0 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   memcpy(h_x0, h_z, n*sizeof(HYPRE_Real));
   GaussSeidelCPU(n, nnz, h_b, h_x0, h_csr, 1, false);
   /*------------ GPU L/U Solv w/ Lev-Sched R32 */
   err = 0.0;
   h_x1 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   printf(" [GPU] G-S LEVR,       ");
   for (int i=0; i<NTESTS; i++)
   {
      memcpy(h_x1, h_z, n*sizeof(HYPRE_Real));
      GaussSeidelRowLevSchd<true>(h_csr, h_b, h_x1, 1, false);
      e1=error_norm(h_x0, h_x1, n);
      err = max(e1, err);
   }
   printf("err norm %.2e\n", err);
   free(h_x1);
   /*------------ GPU L/U Solv w/ Row Dyn-Sched */
   err = 0.0;
   h_x4 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   printf(" [GPU] G-S DYNR,       ");
   for (int i=0; i<NTESTS; i++)
   {
      memcpy(h_x4, h_z, n*sizeof(HYPRE_Real));
      GaussSeidelRowDynSchd<true>(h_csr, h_b, h_x4, 1, false);
      e4=error_norm(h_x0, h_x4, n);
      err = max(e4, err);
   }
   printf("err norm %.2e\n", err);
   free(h_x4);
exit(0);
   /*------------ GPU L/U Solv w/ Col Dyn-Sched */
   err = 0.0;
   h_x5 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   printf(" [GPU] G-S DYNC,       ");
   for (int i=0; i<NTESTS; i++)
   {
      memcpy(h_x5, h_z, n*sizeof(HYPRE_Real));
      GaussSeidelColDynSchd<true>(h_csr, h_b, h_x5, 1, false);
      e5=error_norm(h_x0, h_x5, n);
      err = max(e5, err);
   }
   printf("err norm %.2e\n", err);
   free(h_x5);
   /*----------- CUSPARSE-1 */
   err = 0.0;
   h_x2 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   printf(" [GPU] G-S CUSPARSE-1, ");
   for (int i=0; i<1; i++)
   {
      memcpy(h_x2, h_z, n*sizeof(HYPRE_Real));
      GaussSeidel_cusparse1(h_csr, h_b, h_x2, 1, false);
      e2=error_norm(h_x0, h_x2, n);
      err = max(e2, err);
   }
   printf("err norm %.2e\n", err);
   free(h_x2);
   /*----------- CUSPARSE-2 */
   err = 0.0;
   h_x3 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   printf(" [GPU] G-S CUSPARSE-2, ");
   for (int i=0; i<1; i++)
   {
      memcpy(h_x3, h_z, n*sizeof(HYPRE_Real));
      GaussSeidel_cusparse2(h_csr, h_b, h_x3, 1, false);
      e3=error_norm(h_x0, h_x3, n);
      err = max(e3, err);
   }
   printf("err norm %.2e\n", err);
   free(h_x3);
bench:
   printf("Benchmark kernels, repetition %d\n", REP);
   /*------------------------------------------------ */
   /*------------- Start benchmarking kernels ------- */
   /*------------------------------------------------ */
   /*------------- CPU L/U Sol */
   h_x0 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   memcpy(h_x0, h_z, n*sizeof(HYPRE_Real));
   GaussSeidelCPU(n, nnz, h_b, h_x0, h_csr, REP, true);
   /*------------ GPU L/U Solv w/ Lev-Sched R32 */
   h_x1 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   memcpy(h_x1, h_z, n*sizeof(HYPRE_Real));
   GaussSeidelRowLevSchd<false>(h_csr, h_b, h_x1, REP, true);
   e1=error_norm(h_x0, h_x1, n);
   printf("err norm %.2e\n", e1);
   free(h_x1);
   /*------------ GPU L/U Solv w/ Dyn-Sched R */
   h_x4 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   memcpy(h_x4, h_z, n*sizeof(HYPRE_Real));
   GaussSeidelRowDynSchd<false>(h_csr, h_b, h_x4, REP, true);
   e4=error_norm(h_x0, h_x4, n);
   printf("err norm %.2e\n", e4);
   free(h_x4);
   /*------------ GPU L/U Solv w/ Dyn-Sched C */
   h_x5 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   memcpy(h_x5, h_z, n*sizeof(HYPRE_Real));
   GaussSeidelColDynSchd<false>(h_csr, h_b, h_x5, REP, true);
   e5=error_norm(h_x0, h_x5, n);
   printf("err norm %.2e\n", e5);
   free(h_x5);
   /*----------- CUSPARSE-1 */
   h_x2 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   memcpy(h_x2, h_z, n*sizeof(HYPRE_Real));
   GaussSeidel_cusparse1(h_csr, h_b, h_x2, REP, true);
   e2=error_norm(h_x0, h_x2, n);
   printf("err norm %.2e\n", e2);
   free(h_x2);
   /*----------- CUSPARSE-2 */
   h_x3 = (HYPRE_Real *) malloc(n*sizeof(HYPRE_Real));
   memcpy(h_x3, h_z, n*sizeof(HYPRE_Real));
   GaussSeidel_cusparse2(h_csr, h_b, h_x3, REP, true);
   e3=error_norm(h_x0, h_x3, n);
   printf("err norm %.2e\n", e3);
   free(h_x3);
   /*----------- Done free */
   hypre_CSRMatrixDestroy(h_csr);
   FreeCOO(&h_coo);
   free(h_b);
   free(h_x0);
   /*---------- check error */
   cuda_check_err();
}

