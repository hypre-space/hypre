
#include "seq_mv.h"


int
main( int argc,
      char *argv[] )
{
   /* Read in matrices */
   printf("read\n");
   hypre_CSRMatrix *host_mat = hypre_CSRMatrixRead("A");
   hypre_CSRMatrix *A = hypre_CSRMatrixClone_v2(host_mat, 1, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixDestroy(host_mat);
   host_mat = hypre_CSRMatrixRead("B");
   hypre_CSRMatrix *B = hypre_CSRMatrixClone_v2(host_mat, 1, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixDestroy(host_mat);

   /* multiply matrices with oneMKL */
   printf("multiply\n");
   hypre_CSRMatrix *C;
   hypreDevice_CSRSpGemm(A, B, &C);

   printf("DONE\n");
   return 0;
}
