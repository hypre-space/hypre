
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_struct_mv.h"

/*------------------------------------------------------------------------*/

int
hypre_StMatrixRead( FILE            *file,
                    int              id,
                    int              ndim,
                    hypre_StMatrix **matrix_ptr )
{
   hypre_StMatrix *matrix;
   hypre_StCoeff  *coeff;
   int             entry, d, size;

   fscanf(file, "%d\n", &size);

   hypre_StMatrixCreate(id, size, ndim, &matrix);

   for (d = 0; d < ndim; d++)
   {
      fscanf(file, "%d", &(matrix->rmap[d]));
   }
   for (d = 0; d < ndim; d++)
   {
      fscanf(file, "%d", &(matrix->dmap[d]));
   }
   for (entry = 0; entry < size; entry++)
   {
      for (d = 0; d < ndim; d++)
      {
         fscanf(file, "%d", &(matrix->shapes[entry][d]));
      }
      hypre_StCoeffCreate(1, &coeff);
      (coeff->terms[0].id) = id;
      (coeff->terms[0].entry) = entry;
      (matrix->coeffs[entry]) = coeff;
   }

   *matrix_ptr = matrix;

   return 0;
}

/*------------------------------------------------------------------------*/

int
main( int argc,
      char *argv[] )
{
   char             infile_default[] = "stencilmult.in";
   char            *infile;
   FILE            *file;
   hypre_StMatrix **matrices, *A, *B, *C;
   char            *matnames, transposechar;
   int             *termids, *termtrs;
   int              ndim, nmatrices, nprods, nterms, i, j, id, transpose;

   infile = infile_default;
   if (argc > 1)
   {
      infile = argv[1];
   }
   if ((file = fopen(infile, "r")) == NULL)
   {
      printf("Error: can't open input file %s\n", infile);
      exit(1);
   }

   fscanf(file, "%d %d\n", &ndim, &nmatrices);
   matrices = hypre_CTAlloc(hypre_StMatrix *, nmatrices);
   matnames = hypre_CTAlloc(char, nmatrices);

   for (i = 0; i < nmatrices; i++)
   {
      fscanf(file, "%d%s\n", &id, &matnames[i]);
      if (id != i)
      {
         printf("Matrix ID has incorrect value");
         exit(0);
      }
      hypre_StMatrixRead(file, i, ndim, &matrices[i]);
   }

   /* Read in requested matrix products, multiply, and print results */
   fscanf(file, "%d\n", &nprods);
   for (i = 0; i < nprods; i++)
   {
      fscanf(file, "%d", &nterms);
      termids = hypre_CTAlloc(int, nterms);
      termtrs = hypre_CTAlloc(int, nterms);
      for (j = 0; j < nterms; j++)
      {
         fscanf(file, "%d%c", &termids[j], &transposechar);
         if (transposechar == 't')
         {
            termtrs[j] = 1;
         }
      }

      id = termids[0]; transpose = termtrs[0];
      hypre_StMatrixClone(matrices[id], ndim, &C);
      if (transpose)
      {
         hypre_StMatrixTranspose(C, ndim);
      }
      for (j = 1; j < nterms; j++)
      {
         A = C;
         id = termids[j]; transpose = termtrs[j];
         hypre_StMatrixClone(matrices[id], ndim, &B);
         if (transpose)
         {
            hypre_StMatrixTranspose(B, ndim);
         }
         hypre_StMatrixMult(A, B, nmatrices, ndim, &C);
         hypre_StMatrixDestroy(A);
         hypre_StMatrixDestroy(B);
      }

      printf("\nMatrix product %d:", i);
      for (j = 0; j < nterms; j++)
      {
         id = termids[j]; transpose = termtrs[j];
         printf(" %c", matnames[id]);
         if (transpose)
         {
            printf("t");
         }
      }
      printf("\n\n");
      hypre_StMatrixPrint(C, matnames, ndim);

      hypre_TFree(termids);
      hypre_TFree(termtrs);
      hypre_StMatrixDestroy(C);
   }

   /* Clean up*/
   for (i = 0; i < nmatrices; i++)
   {
      hypre_StMatrixDestroy(matrices[i]);
   }
   hypre_TFree(matrices);
   hypre_TFree(matnames);
   
   fclose(file);

   return 0;
}
