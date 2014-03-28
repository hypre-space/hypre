/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   void                   *relax_data;
   void                   *rb_relax_data;
   HYPRE_Int               relax_type;
   HYPRE_Real              jacobi_weight;

} hypre_BAMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_BAMGRelaxCreate( MPI_Comm  comm )
{
   hypre_BAMGRelaxData *bamg_relax_data;

   bamg_relax_data = hypre_CTAlloc(hypre_BAMGRelaxData, 1);
   (bamg_relax_data -> relax_data) = hypre_PointRelaxCreate(comm);
   (bamg_relax_data -> rb_relax_data) = hypre_RedBlackGSCreate(comm);
   (bamg_relax_data -> relax_type) = 0;        /* Weighted Jacobi */
   (bamg_relax_data -> jacobi_weight) = 0.0;

   return (void *) bamg_relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxDestroy( void *bamg_relax_vdata )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;

   if (bamg_relax_data)
   {
      hypre_PointRelaxDestroy(bamg_relax_data -> relax_data);
      hypre_RedBlackGSDestroy(bamg_relax_data -> rb_relax_data);
      hypre_TFree(bamg_relax_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelax( void               *bamg_relax_vdata,
                 hypre_StructMatrix *A,
                 hypre_StructVector *b,
                 hypre_StructVector *x                )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;
   HYPRE_Int    relax_type = (bamg_relax_data -> relax_type);
   HYPRE_Int    constant_coefficient= hypre_StructMatrixConstantCoefficient(A);

   switch(relax_type)
   {
      case 0:
      case 1:
         hypre_PointRelax((bamg_relax_data -> relax_data), A, b, x);
         break;
      case 2:
      case 3:
         if (constant_coefficient)
         {
            hypre_RedBlackConstantCoefGS((bamg_relax_data -> rb_relax_data), 
                                         A, b, x);
         }
         else
         {
            hypre_RedBlackGS((bamg_relax_data -> rb_relax_data), A, b, x);
         }
          
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxSetup( void               *bamg_relax_vdata,
                      hypre_StructMatrix *A,
                      hypre_StructVector *b,
                      hypre_StructVector *x                )
{
   hypre_BAMGRelaxData *bamg_relax_data  = bamg_relax_vdata;
   HYPRE_Int            relax_type       = (bamg_relax_data -> relax_type);
   HYPRE_Real           jacobi_weight    = (bamg_relax_data -> jacobi_weight); 

   switch(relax_type)
   {
      case 0:
      case 1:
         hypre_PointRelaxSetup((bamg_relax_data -> relax_data), A, b, x);
         break;
      case 2:
      case 3:
         hypre_RedBlackGSSetup((bamg_relax_data -> rb_relax_data), A, b, x);
         break;
   }

   if (relax_type==1)
   {
      hypre_PointRelaxSetWeight(bamg_relax_data -> relax_data, jacobi_weight);
   }
   
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxSetType( void  *bamg_relax_vdata,
                        HYPRE_Int    relax_type       )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;
   void                *relax_data = (bamg_relax_data -> relax_data);

   (bamg_relax_data -> relax_type) = relax_type;

   switch(relax_type)
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
hypre_BAMGRelaxSetJacobiWeight(void  *bamg_relax_vdata,
                               HYPRE_Real weight) 
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;

   (bamg_relax_data -> jacobi_weight)    = weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxSetPreRelax( void  *bamg_relax_vdata )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;
   HYPRE_Int            relax_type = (bamg_relax_data -> relax_type);

   switch(relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
         hypre_RedBlackGSSetStartRed((bamg_relax_data -> rb_relax_data));
         break;

      case 3: /* Red-Black Gauss-Seidel (non-symmetric) */
         hypre_RedBlackGSSetStartRed((bamg_relax_data -> rb_relax_data));
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxSetPostRelax( void  *bamg_relax_vdata )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;
   HYPRE_Int            relax_type = (bamg_relax_data -> relax_type);

   switch(relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
         hypre_RedBlackGSSetStartBlack((bamg_relax_data -> rb_relax_data));
         break;

      case 3: /* Red-Black Gauss-Seidel (non-symmetric) */
         hypre_RedBlackGSSetStartRed((bamg_relax_data -> rb_relax_data));
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxSetTol( void   *bamg_relax_vdata,
                       HYPRE_Real  tol              )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;

   hypre_PointRelaxSetTol((bamg_relax_data -> relax_data), tol);
   hypre_RedBlackGSSetTol((bamg_relax_data -> rb_relax_data), tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxSetMaxIter( void  *bamg_relax_vdata,
                           HYPRE_Int    max_iter         )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;

   hypre_PointRelaxSetMaxIter((bamg_relax_data -> relax_data), max_iter);
   hypre_RedBlackGSSetMaxIter((bamg_relax_data -> rb_relax_data), max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxSetZeroGuess( void  *bamg_relax_vdata,
                             HYPRE_Int    zero_guess       )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;

   hypre_PointRelaxSetZeroGuess((bamg_relax_data -> relax_data), zero_guess);
   hypre_RedBlackGSSetZeroGuess((bamg_relax_data -> rb_relax_data), zero_guess);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGRelaxSetTempVec( void               *bamg_relax_vdata,
                           hypre_StructVector *t                )
{
   hypre_BAMGRelaxData *bamg_relax_data = bamg_relax_vdata;

   hypre_PointRelaxSetTempVec((bamg_relax_data -> relax_data), t);

   return hypre_error_flag;
}

