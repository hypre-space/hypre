/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   void                   *relax_data;
   void                   *rb_relax_data;
   HYPRE_Int               relax_type;
   HYPRE_Real              jacobi_weight;

} hypre_PFMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_PFMGRelaxCreate( MPI_Comm  comm )
{
   hypre_PFMGRelaxData *pfmg_relax_data;

   pfmg_relax_data = hypre_CTAlloc(hypre_PFMGRelaxData,  1, HYPRE_MEMORY_HOST);
   (pfmg_relax_data -> relax_data) = hypre_PointRelaxCreate(comm);
   (pfmg_relax_data -> rb_relax_data) = hypre_RedBlackGSCreate(comm);
   (pfmg_relax_data -> relax_type) = 0;        /* Weighted Jacobi */
   (pfmg_relax_data -> jacobi_weight) = 0.0;

   return (void *) pfmg_relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxDestroy( void *pfmg_relax_vdata )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;

   if (pfmg_relax_data)
   {
      hypre_PointRelaxDestroy(pfmg_relax_data -> relax_data);
      hypre_RedBlackGSDestroy(pfmg_relax_data -> rb_relax_data);
      hypre_TFree(pfmg_relax_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelax( void               *pfmg_relax_vdata,
                 hypre_StructMatrix *A,
                 hypre_StructVector *b,
                 hypre_StructVector *x                )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;
   HYPRE_Int    relax_type = (pfmg_relax_data -> relax_type);
   HYPRE_Int    constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   switch (relax_type)
   {
      case 0:
      case 1:
         hypre_PointRelax((pfmg_relax_data -> relax_data), A, b, x);
         break;
      case 2:
      case 3:
         if (constant_coefficient)
         {
            hypre_RedBlackConstantCoefGS((pfmg_relax_data -> rb_relax_data),
                                         A, b, x);
         }
         else
         {
            hypre_RedBlackGS((pfmg_relax_data -> rb_relax_data), A, b, x);
         }

         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetup( void               *pfmg_relax_vdata,
                      hypre_StructMatrix *A,
                      hypre_StructVector *b,
                      hypre_StructVector *x                )
{
   hypre_PFMGRelaxData *pfmg_relax_data  = (hypre_PFMGRelaxData *)pfmg_relax_vdata;
   HYPRE_Int            relax_type       = (pfmg_relax_data -> relax_type);
   HYPRE_Real           jacobi_weight    = (pfmg_relax_data -> jacobi_weight);

   switch (relax_type)
   {
      case 0:
      case 1:
         hypre_PointRelaxSetup((pfmg_relax_data -> relax_data), A, b, x);
         break;
      case 2:
      case 3:
         hypre_RedBlackGSSetup((pfmg_relax_data -> rb_relax_data), A, b, x);
         break;
   }

   if (relax_type == 1)
   {
      hypre_PointRelaxSetWeight(pfmg_relax_data -> relax_data, jacobi_weight);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetType( void  *pfmg_relax_vdata,
                        HYPRE_Int    relax_type       )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;
   void                *relax_data = (pfmg_relax_data -> relax_data);

   (pfmg_relax_data -> relax_type) = relax_type;

   switch (relax_type)
   {
      case 0: /* Jacobi */
      {
         hypre_Index  stride;
         hypre_Index  indices[1];

         hypre_PointRelaxSetWeight(relax_data, 1.0);
         hypre_PointRelaxSetNumPointsets(relax_data, 1);

         hypre_SetIndex3(stride, 1, 1, 1);
         hypre_SetIndex3(indices[0], 0, 0, 0);
         hypre_PointRelaxSetPointset(relax_data, 0, 1, stride, indices);
      }
      break;

      case 2: /* Red-Black Gauss-Seidel */
      case 3: /* Red-Black Gauss-Seidel (non-symmetric) */
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetJacobiWeight(void  *pfmg_relax_vdata,
                               HYPRE_Real weight)
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;

   (pfmg_relax_data -> jacobi_weight)    = weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetPreRelax( void  *pfmg_relax_vdata )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;
   HYPRE_Int            relax_type = (pfmg_relax_data -> relax_type);

   switch (relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
         hypre_RedBlackGSSetStartRed((pfmg_relax_data -> rb_relax_data));
         break;

      case 3: /* Red-Black Gauss-Seidel (non-symmetric) */
         hypre_RedBlackGSSetStartRed((pfmg_relax_data -> rb_relax_data));
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetPostRelax( void  *pfmg_relax_vdata )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;
   HYPRE_Int            relax_type = (pfmg_relax_data -> relax_type);

   switch (relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
         hypre_RedBlackGSSetStartBlack((pfmg_relax_data -> rb_relax_data));
         break;

      case 3: /* Red-Black Gauss-Seidel (non-symmetric) */
         hypre_RedBlackGSSetStartRed((pfmg_relax_data -> rb_relax_data));
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetTol( void   *pfmg_relax_vdata,
                       HYPRE_Real  tol              )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;

   hypre_PointRelaxSetTol((pfmg_relax_data -> relax_data), tol);
   hypre_RedBlackGSSetTol((pfmg_relax_data -> rb_relax_data), tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetMaxIter( void  *pfmg_relax_vdata,
                           HYPRE_Int    max_iter         )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;

   hypre_PointRelaxSetMaxIter((pfmg_relax_data -> relax_data), max_iter);
   hypre_RedBlackGSSetMaxIter((pfmg_relax_data -> rb_relax_data), max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetZeroGuess( void  *pfmg_relax_vdata,
                             HYPRE_Int    zero_guess       )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;

   hypre_PointRelaxSetZeroGuess((pfmg_relax_data -> relax_data), zero_guess);
   hypre_RedBlackGSSetZeroGuess((pfmg_relax_data -> rb_relax_data), zero_guess);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGRelaxSetTempVec( void               *pfmg_relax_vdata,
                           hypre_StructVector *t                )
{
   hypre_PFMGRelaxData *pfmg_relax_data = (hypre_PFMGRelaxData *)pfmg_relax_vdata;

   hypre_PointRelaxSetTempVec((pfmg_relax_data -> relax_data), t);

   return hypre_error_flag;
}

