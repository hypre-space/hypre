/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "_hypre_utilities.h"

HYPRE_Int hypre_mm_is_valid(MM_typecode matcode)
{
   if (!hypre_mm_is_matrix(matcode)) { return 0; }
   if (hypre_mm_is_dense(matcode) && hypre_mm_is_pattern(matcode)) { return 0; }
   if (hypre_mm_is_real(matcode) && hypre_mm_is_hermitian(matcode)) { return 0; }
   if (hypre_mm_is_pattern(matcode) && (hypre_mm_is_hermitian(matcode) || hypre_mm_is_skew(matcode))) { return 0; }
   return 1;
}

HYPRE_Int hypre_mm_read_banner(FILE *f, MM_typecode *matcode)
{
   char line[MM_MAX_LINE_LENGTH];
   char banner[MM_MAX_TOKEN_LENGTH];
   char mtx[MM_MAX_TOKEN_LENGTH];
   char crd[MM_MAX_TOKEN_LENGTH];
   char data_type[MM_MAX_TOKEN_LENGTH];
   char storage_scheme[MM_MAX_TOKEN_LENGTH];
   char *p;

   hypre_mm_clear_typecode(matcode);

   if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
   {
      return MM_PREMATURE_EOF;
   }

   if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
              storage_scheme) != 5)
   {
      return MM_PREMATURE_EOF;
   }

   for (p = mtx; *p != '\0'; *p = tolower(*p), p++); /* convert to lower case */
   for (p = crd; *p != '\0'; *p = tolower(*p), p++);
   for (p = data_type; *p != '\0'; *p = tolower(*p), p++);
   for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++);

   /* check for banner */
   if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
   {
      return MM_NO_HEADER;
   }

   /* first field should be "mtx" */
   if (strcmp(mtx, MM_MTX_STR) != 0)
   {
      return MM_UNSUPPORTED_TYPE;
   }
   hypre_mm_set_matrix(matcode);


   /* second field describes whether this is a sparse matrix (in coordinate
      storgae) or a dense array */


   if (strcmp(crd, MM_SPARSE_STR) == 0)
   {
      hypre_mm_set_sparse(matcode);
   }
   else if (strcmp(crd, MM_DENSE_STR) == 0)
   {
      hypre_mm_set_dense(matcode);
   }
   else
   {
      return MM_UNSUPPORTED_TYPE;
   }


   /* third field */

   if (strcmp(data_type, MM_REAL_STR) == 0)
   {
      hypre_mm_set_real(matcode);
   }
   else if (strcmp(data_type, MM_COMPLEX_STR) == 0)
   {
      hypre_mm_set_complex(matcode);
   }
   else if (strcmp(data_type, MM_PATTERN_STR) == 0)
   {
      hypre_mm_set_pattern(matcode);
   }
   else if (strcmp(data_type, MM_INT_STR) == 0)
   {
      hypre_mm_set_integer(matcode);
   }
   else
   {
      return MM_UNSUPPORTED_TYPE;
   }


   /* fourth field */

   if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
   {
      hypre_mm_set_general(matcode);
   }
   else if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
   {
      hypre_mm_set_symmetric(matcode);
   }
   else if (strcmp(storage_scheme, MM_HERM_STR) == 0)
   {
      hypre_mm_set_hermitian(matcode);
   }
   else if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
   {
      hypre_mm_set_skew(matcode);
   }
   else
   {
      return MM_UNSUPPORTED_TYPE;
   }

   return 0;
}

HYPRE_Int hypre_mm_read_mtx_crd_size(FILE *f, HYPRE_Int *M, HYPRE_Int *N, HYPRE_Int *nz )
{
   char line[MM_MAX_LINE_LENGTH];
   HYPRE_Int num_items_read;

   /* set return null parameter values, in case we exit with errors */
   *M = *N = *nz = 0;

   /* now continue scanning until you reach the end-of-comments */
   do
   {
      if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      {
         return MM_PREMATURE_EOF;
      }
   }
   while (line[0] == '%');

   /* line[] is either blank or has M,N, nz */
   if (hypre_sscanf(line, "%d %d %d", M, N, nz) == 3)
   {
      return 0;
   }
   else
   {
      do
      {
         num_items_read = hypre_fscanf(f, "%d %d %d", M, N, nz);
         if (num_items_read == EOF) { return MM_PREMATURE_EOF; }
      }
      while (num_items_read != 3);
   }

   return 0;
}

