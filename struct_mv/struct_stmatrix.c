/*BHEADER**********************************************************************
 * Copyright (c) 2014,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StIndexCopy( hypre_Index index1,
                   hypre_Index index2,
                   HYPRE_Int   ndim )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      index2[d] = index1[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StIndexNegate( hypre_Index index,
                     HYPRE_Int   ndim )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      index[d] = -index[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StIndexShift( hypre_Index index,
                    hypre_Index shift,
                    HYPRE_Int   ndim )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      index[d] += shift[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StIndexPrint( hypre_Index index,
                    char        lchar,
                    char        rchar,
                    HYPRE_Int   ndim )
{
   HYPRE_Int d;

/*   hypre_printf("(% d", index[0]);*/
   hypre_printf("%c% d", lchar, index[0]);
   for (d = 1; d < ndim; d++)
   {
      hypre_printf(",% d", index[d]);
   }
/*   hypre_printf(")");*/
   hypre_printf("%c", rchar);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StTermCopy( hypre_StTerm *term1,
                  hypre_StTerm *term2,
                  HYPRE_Int     ndim )
{
   HYPRE_Int  d;

   (term2->id)    = (term1->id);
   (term2->entry) = (term1->entry);
   for (d = 0; d < ndim; d++)
   {
      (term2->shift[d]) = (term1->shift[d]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StTermPrint( hypre_StTerm *term,
                   char         *matnames,
                   HYPRE_Int     ndim )
{
   hypre_printf("%c%d", matnames[(term->id)], (term->entry));
   hypre_StIndexPrint((term->shift), '(', ')', ndim);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StCoeffCreate( HYPRE_Int       nterms,
                     hypre_StCoeff **coeff_ptr )
{
   hypre_StCoeff *coeff;

   coeff = hypre_CTAlloc(hypre_StCoeff, 1);
   (coeff->nterms) = nterms;
   (coeff->terms) = hypre_CTAlloc(hypre_StTerm, nterms);
   (coeff->prev) = NULL;
   (coeff->next) = NULL;

   *coeff_ptr = coeff;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StCoeffClone( hypre_StCoeff  *coeff,
                    HYPRE_Int       ndim,
                    hypre_StCoeff **clone_ptr )
{
   hypre_StCoeff *clone, *lastclone;
   HYPRE_Int      nterms, t;

   *clone_ptr = lastclone = NULL;

   while (coeff != NULL)
   {
      nterms = (coeff->nterms);
      hypre_StCoeffCreate(nterms, &clone);
      for (t = 0; t < nterms; t++)
      {
         hypre_StTermCopy(&(coeff->terms[t]), &(clone->terms[t]), ndim);
      }
      if (lastclone == NULL)
      {
         *clone_ptr = clone;
      }
      else
      {
         (lastclone->next) = clone;
         (clone->prev) = lastclone;
      }
      lastclone = clone;
      
      coeff = (coeff->next);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StCoeffDestroy( hypre_StCoeff *coeff )
{
   hypre_StCoeff *next;

   while (coeff != NULL)
   {
      next = (coeff->next);
      hypre_TFree(coeff->terms);
      hypre_TFree(coeff);
      coeff = next;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StCoeffShift( hypre_StCoeff *coeff,
                    hypre_Index    shift,
                    HYPRE_Int      ndim )
{
   hypre_StTerm *terms;
   HYPRE_Int     nterms, t;

   while (coeff != NULL)
   {
      nterms = (coeff->nterms);
      terms  = (coeff->terms);
      for (t = 0; t < nterms; t++)
      {
         hypre_StIndexShift((terms[t].shift), shift, ndim);
      }
      
      coeff = (coeff->next);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StCoeffPush( hypre_StCoeff **stack_ptr,
                   hypre_StCoeff  *coeff )
{
   hypre_StCoeff *stack;

   /* Keep a reference to the current stack */
   stack = *stack_ptr;
   /* Make coeff the head of the new stack */
   *stack_ptr = coeff;

   /* Modify prev and next pointers in the linked list */
   (coeff->prev) = NULL;
   while ((coeff->next) != NULL)
   {
      coeff = (coeff->next);
   }
   (coeff->next) = stack;
   if (stack != NULL)
   {
      (stack->prev) = coeff;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StCoeffMult( hypre_StCoeff  *Acoeff,
                   hypre_StCoeff  *Bcoeff,
                   HYPRE_Int       ndim,
                   hypre_StCoeff **Ccoeff_ptr )
{
   hypre_StCoeff     *Bcoeffhead = Bcoeff;
   hypre_StCoeff     *Ccoeff = NULL;
   hypre_StCoeff     *coeff;
   HYPRE_Int          Ci, i;

   while (Acoeff != NULL)
   {
      Bcoeff = Bcoeffhead;

      while (Bcoeff != NULL)
      {
         hypre_StCoeffCreate( ((Acoeff->nterms) + (Bcoeff->nterms)) , &coeff);
         Ci = 0;
         for (i = 0; i < (Acoeff->nterms); i++)
         {
            hypre_StTermCopy(&(Acoeff->terms[i]), &(coeff->terms[Ci++]), ndim);
         }
         for (i = 0; i < (Bcoeff->nterms); i++)
         {
            hypre_StTermCopy(&(Bcoeff->terms[i]), &(coeff->terms[Ci++]), ndim);
         }
         hypre_StCoeffPush(&Ccoeff, coeff);

         Bcoeff = (Bcoeff->next);
      }

      Acoeff = (Acoeff->next);
   }

   *Ccoeff_ptr = Ccoeff;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StCoeffPrint( hypre_StCoeff *coeff,
                    char          *matnames,
                    HYPRE_Int      ndim )
{
   HYPRE_Int i;

   while (coeff != NULL)
   {
      hypre_StTermPrint(&(coeff->terms[0]), matnames, ndim);
      for (i = 1; i < (coeff->nterms); i++)
      {
         hypre_printf("*");
         hypre_StTermPrint(&(coeff->terms[i]), matnames, ndim);
      }
      coeff = (coeff->next);
      if (coeff != NULL)
      {
         hypre_printf(" + ");
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StMatrixCreate( HYPRE_Int        id,
                      HYPRE_Int        size,
                      HYPRE_Int        ndim,
                      hypre_StMatrix **matrix_ptr )
{
   hypre_StMatrix  *matrix;
   HYPRE_Int        d;

   matrix = hypre_TAlloc(hypre_StMatrix, 1);

   (matrix->id)   = id;
   (matrix->size) = size;
   for (d = 0; d < ndim; d++)
   {
      (matrix->rmap[d]) = 1;
      (matrix->dmap[d]) = 1;
   }
   (matrix->shapes)  = hypre_CTAlloc(hypre_Index, size);
   (matrix->coeffs)  = hypre_CTAlloc(hypre_StCoeff *, size);

   *matrix_ptr = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StMatrixClone( hypre_StMatrix  *matrix,
                     HYPRE_Int        ndim,
                     hypre_StMatrix **mclone_ptr )
{
   hypre_StMatrix *mclone;
   HYPRE_Int       entry;

   hypre_StMatrixCreate((matrix->id), (matrix->size), ndim, &mclone);

   hypre_StIndexCopy((matrix->rmap), (mclone->rmap), ndim);
   hypre_StIndexCopy((matrix->dmap), (mclone->dmap), ndim);

   for (entry = 0; entry < (matrix->size); entry++)
   {
      hypre_StIndexCopy((matrix->shapes[entry]), (mclone->shapes[entry]), ndim);
      hypre_StCoeffClone((matrix->coeffs[entry]), ndim, &(mclone->coeffs[entry]));
   }

   *mclone_ptr = mclone;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StMatrixDestroy( hypre_StMatrix *matrix )
{
   HYPRE_Int entry;

   for (entry = 0; entry < (matrix->size); entry++)
   {
      hypre_StCoeffDestroy(matrix->coeffs[entry]);
   }
   hypre_TFree(matrix->shapes);
   hypre_TFree(matrix->coeffs);
   hypre_TFree(matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StMatrixTranspose( hypre_StMatrix *matrix,
                         HYPRE_Int       ndim )
{
   HYPRE_Int       size   = (matrix->size);
   hypre_Index    *shapes = (matrix->shapes);
   hypre_StCoeff **coeffs = (matrix->coeffs);
   hypre_Index     tmap;
   HYPRE_Int       entry;

   for (entry = 0; entry < size; entry++)
   {
      hypre_StIndexNegate(shapes[entry], ndim);
      hypre_StCoeffShift(coeffs[entry], shapes[entry], ndim);
   }
   /* Swap maps */
   hypre_StIndexCopy((matrix->rmap), tmap,           ndim);
   hypre_StIndexCopy((matrix->dmap), (matrix->rmap), ndim);
   hypre_StIndexCopy(tmap,           (matrix->dmap), ndim);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StBoxRank( hypre_Index i,
                 hypre_Index lo,
                 hypre_Index hi,
                 HYPRE_Int   ndim )
{
   HYPRE_Int rank, size, d;

   rank = 0;
   size = 1;
   for (d = 0; d < ndim; d++)
   {
      rank += (i[d] - lo[d]) * size;
      size *= (hi[d] - lo[d] + 1);
   }

   return rank;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StMatrixMatmat( hypre_StMatrix  *A,
                      hypre_StMatrix  *B,
                      HYPRE_Int        Cid,
                      HYPRE_Int        ndim,
                      hypre_StMatrix **C_ptr )
{
   hypre_StMatrix *C, *Aclone, *Bclone;
   hypre_StCoeff  *Acoeff;
   HYPRE_Int       Aentry, Asize, *Ashape;
   hypre_StCoeff  *Bcoeff;
   HYPRE_Int       Bentry, Bsize, *Bshape;
   hypre_StCoeff  *Ccoeff;
   HYPRE_Int       Centry, Csize, *Crmap, *Cdmap;
   hypre_StCoeff  *shiftcoeff;
   hypre_Index     ABoff, ABi, ABlo, ABhi;
   HYPRE_Int      *ABbox, ABboxsize, ABii;
   HYPRE_Int       d, lo, hi, off, isrowstencil, indomain;

   /* Check domain and range compatibility */
   /* isrowstencil = {1, 0, -1, -2} -> is stencil type {row, col, either, error} */
   isrowstencil = -1;
   for (d = 0; d < ndim; d++)
   {
      HYPRE_Int m1 = (A->rmap[d]);
      HYPRE_Int m2 = (A->dmap[d]);
      HYPRE_Int m3 = (B->dmap[d]);

      if (m2 != (B->rmap[d]))
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Domain and range mismatch");
         return hypre_error_flag;
      }

      if ((m1 >= m2) && (m1%m2 == 0) && (m1 >= m3) && (m1%m3 == 0))
      {
         if (m1 != m3)
         {
            if (isrowstencil == 0)
            {
               isrowstencil = -2;
               break;
            }
            else
            {
               /* Range grid is a subset of domain grid */
               isrowstencil = 1;
            }
         }
      }
      else if ((m3 >= m2) && (m3%m2 == 0) && (m3 >= m1) && (m3%m1 == 0))
      {
         if (m3 != m1)
         {
            if (isrowstencil == 1)
            {
               isrowstencil = -2;
               break;
            }
            else
            {
               /* Domain grid is a subset of range grid */
               isrowstencil = 0;
            }
         }
      }
      else
      {
         isrowstencil = -2;
         break;
      }
   }
   if (isrowstencil == -2)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Invalid or incompatible matrices in product");
      return hypre_error_flag;
   }
   /* If either stencil type is okay, choose row type */
   if (isrowstencil == -1)
   {
      isrowstencil = 1;
   }

   /* Clone original matrices, then reset A and B pointers */
   hypre_StMatrixClone(A, ndim, &Aclone);
   hypre_StMatrixClone(B, ndim, &Bclone);
   if (isrowstencil)
   {
      A = Aclone;
      B = Bclone;
      hypre_StMatrixTranspose(B, ndim);
   }
   else
   {
      A = Bclone;
      hypre_StMatrixTranspose(A, ndim);
      B = Aclone;
   }
   Crmap = (A->rmap);
   Cdmap = (B->rmap);

   /* Initialize some things */
   Asize = (A->size);
   Bsize = (B->size);

   /* Compute box size info */
   ABboxsize = 1;
   for (d = 0; d < ndim; d++)
   {
      lo = hi = 0;
      for (Aentry = 0; Aentry < Asize; Aentry++)
      {
         off = (A->shapes[Aentry][d]);
         lo = hypre_min(lo, off);
         hi = hypre_max(hi, off);
      }
      ABlo[d] = lo;
      ABhi[d] = hi;

      lo = hi = 0;
      for (Bentry = 0; Bentry < Bsize; Bentry++)
      {
         off = -(B->shapes[Bentry][d]);
         lo = hypre_min(lo, off);
         hi = hypre_max(hi, off);
      }
      ABlo[d] += lo;
      ABhi[d] += hi;

      /* Adjust lo and hi based on domain grid */
      ABlo[d] /= Cdmap[d];
      ABhi[d] /= Cdmap[d];

      ABboxsize *= (ABhi[d] - ABlo[d] + 1);
   }

   /* Compute stenc(AB) = B^T stenc(A) */
   ABbox = hypre_TAlloc(HYPRE_Int, ABboxsize);
   for (ABii = 0; ABii < ABboxsize; ABii++)
   {
      ABbox[ABii] = -1;
   }
   hypre_StMatrixCreate(Cid, ABboxsize, ndim, &C);
   hypre_StIndexCopy(Crmap, (C->rmap), ndim);
   hypre_StIndexCopy(Cdmap, (C->dmap), ndim);
   Csize = 0;
   for (Bentry = 0; Bentry < Bsize; Bentry++)
   {
      Bshape = (B->shapes[Bentry]);
      Bcoeff = (B->coeffs[Bentry]);
      hypre_StIndexNegate(Bshape, ndim);

      for (Aentry = 0; Aentry < Asize; Aentry++)
      {
         Ashape = (A->shapes[Aentry]);
         Acoeff = (A->coeffs[Aentry]);

         hypre_StIndexCopy(Ashape, ABoff, ndim);
         hypre_StIndexShift(ABoff, Bshape, ndim);
         /* Check first that ABoff is in the domain grid */
         indomain = 1;
         for (d = 0; d < ndim; d++)
         {
            if (ABoff[d]%Cdmap[d] != 0)
            {
               indomain = 0;
               break;
            }
            /* Adjust index based on domain grid */
            ABi[d] = ABoff[d]/Cdmap[d];
         }
         if (indomain)
         {
            hypre_StCoeffClone(Bcoeff, ndim, &shiftcoeff);
            hypre_StCoeffShift(shiftcoeff, ABoff, ndim);
            hypre_StCoeffMult(shiftcoeff, Acoeff, ndim, &Ccoeff);
            hypre_StCoeffDestroy(shiftcoeff);
            ABii = hypre_StBoxRank(ABi, ABlo, ABhi, ndim);
            Centry = ABbox[ABii];
            if (Centry < 0)
            {
               Centry = Csize;
               Csize++;
               ABbox[ABii] = Centry;
               hypre_StIndexCopy(ABoff, (C->shapes[Centry]), ndim);
            }
            hypre_StCoeffPush(&(C->coeffs[Centry]), Ccoeff);
         }
      }
   }
   (C->size)    = Csize;
   (C->shapes)  = hypre_TReAlloc((C->shapes), hypre_Index, Csize);
   (C->coeffs)  = hypre_TReAlloc((C->coeffs), hypre_StCoeff *, Csize);
   if (!isrowstencil)
   {
      /* An StMatrix always has a row stencil */
      hypre_StMatrixTranspose(C, ndim);
   }

   /* Simplify? */

   /* Clean up */
   hypre_StMatrixDestroy(Aclone);
   hypre_StMatrixDestroy(Bclone);
   hypre_TFree(ABbox);

   *C_ptr = C;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StMatrixMatmult( HYPRE_Int        nmatrices,
                       hypre_StMatrix **matrices,
                       HYPRE_Int       *transposes,
                       HYPRE_Int        Cid,
                       HYPRE_Int        ndim,
                       hypre_StMatrix **C_ptr )
{
   hypre_StMatrix *A, *B, *C;
   HYPRE_Int       i;

   hypre_StMatrixClone(matrices[0], ndim, &C);
   if (transposes[0])
   {
      hypre_StMatrixTranspose(C, ndim);
   }
   for (i = 1; i < nmatrices; i++)
   {
      A = C;
      hypre_StMatrixClone(matrices[i], ndim, &B);
      if (transposes[i])
      {
         hypre_StMatrixTranspose(B, ndim);
      }
      hypre_StMatrixMatmat(A, B, Cid, ndim, &C);
      hypre_StMatrixDestroy(A);
      hypre_StMatrixDestroy(B);
   }

   *C_ptr = C;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StMatrixPrint( hypre_StMatrix *matrix,
                     char           *matnames,
                     HYPRE_Int       ndim )
{
   HYPRE_Int entry, d;

   hypre_printf("id   = %d\n", (matrix->id));
   hypre_printf("size = %d\n", (matrix->size));

   hypre_printf("rmap = ");
   hypre_StIndexPrint((matrix->rmap), '(', ')', ndim);
   hypre_printf("\n");

   hypre_printf("dmap = ");
   hypre_StIndexPrint((matrix->dmap), '(', ')', ndim);
   hypre_printf("\n");

   for (entry = 0; entry < (matrix->size); entry++)
   {
      hypre_printf("X%02d", entry);
      hypre_StIndexPrint((matrix->shapes[entry]), '[', ']', ndim);
      hypre_printf(" = ");
      hypre_StCoeffPrint((matrix->coeffs[entry]), matnames, ndim);
      hypre_printf("\n");
   }
   hypre_printf("\n");

   /* If 1D or 2D, print the stencil shape */
   if (ndim < 3)
   {
      hypre_Index  boxi, boxlo, boxhi;
      HYPRE_Int   *box, boxsize, off, ii;

      boxsize = 1;
      for (d = 0; d < ndim; d++)
      {
         boxlo[d] = boxhi[d] = 0;
         for (entry = 0; entry < (matrix->size); entry++)
         {
            off = (matrix->shapes[entry][d]);
            boxlo[d] = hypre_min(boxlo[d], off);
            boxhi[d] = hypre_max(boxhi[d], off);
         }
         boxsize *= (boxhi[d] - boxlo[d] + 1);
      }
      for (d = ndim; d < 2; d++)
      {
         boxlo[d] = boxhi[d] = 0;
      }

      box = hypre_TAlloc(HYPRE_Int, boxsize);
      for (ii = 0; ii < boxsize; ii++)
      {
         box[ii] = -1;
      }
      for (entry = 0; entry < (matrix->size); entry++)
      {
         ii = hypre_StBoxRank((matrix->shapes[entry]), boxlo, boxhi, ndim);
         box[ii] = entry;
      }

      for (boxi[1] = boxhi[1]; boxi[1] >= boxlo[1]; boxi[1]--)
      {
         for (boxi[0] = boxlo[0]; boxi[0] <= boxhi[0]; boxi[0]++)
         {
            ii = hypre_StBoxRank(boxi, boxlo, boxhi, ndim);
            entry = box[ii];
            if (entry > -1)
            {
               hypre_printf("   X%02d", entry);
            }
            else
            {
               hypre_printf("      ");
            }
         }
         hypre_printf("\n");
      }
      hypre_printf("\n");

      hypre_TFree(box);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StMatrixNEntryCoeffs( hypre_StMatrix *matrix,
                            HYPRE_Int       entry )
{
   HYPRE_Int      ncoeffs = 0;
   hypre_StCoeff *coeff;

   coeff = matrix->coeffs[entry];
   while (coeff != NULL)
   {
      ncoeffs++;
      coeff = (coeff->next);
   }

   return ncoeffs;
}
