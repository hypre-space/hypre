
#include "HYPRE.h"
#include "seq_mv.h"


/****************************
 * main
 ****************************/

int
main( int argc,
      char *argv[] )
{
   HYPRE_Init();
   
   hypre_printf("Read in matrices\n");
   char filename[256];
   sprintf(filename, "matA.0");
   hypre_CSRMatrix *matA_host = hypre_CSRMatrixRead(filename);
   sprintf(filename, "matZ.0");
   hypre_CSRMatrix *matZ_host = hypre_CSRMatrixRead(filename);

   hypre_printf("Copy matrices to device\n");
   hypre_CSRMatrix *matA = hypre_CSRMatrixClone_v2(matA_host, 1, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrix *matZ = hypre_CSRMatrixClone_v2(matZ_host, 1, HYPRE_MEMORY_DEVICE);

   hypre_printf("Do matmat\n");
   hypre_CSRMatrix *matC = hypre_CSRMatrixMultiply(matA, matZ);

   hypre_printf("Print result\n");
   sprintf(filename, "matC_test");
   hypre_CSRMatrixPrint(matC, filename);

   return 0;
}
