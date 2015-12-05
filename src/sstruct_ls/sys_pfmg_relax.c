/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   void                   *relax_data;
   HYPRE_Int               relax_type;
   double                  jacobi_weight;

} hypre_SysPFMGRelaxData;

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SysPFMGRelaxCreate( MPI_Comm  comm )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data;

   sys_pfmg_relax_data = hypre_CTAlloc(hypre_SysPFMGRelaxData, 1);
   (sys_pfmg_relax_data -> relax_data) = hypre_NodeRelaxCreate(comm);
   (sys_pfmg_relax_data -> relax_type) = 0;        /* Weighted Jacobi */

   return (void *) sys_pfmg_relax_data;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxDestroy( void *sys_pfmg_relax_vdata )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   HYPRE_Int               ierr = 0;

   if (sys_pfmg_relax_data)
   {
      hypre_NodeRelaxDestroy(sys_pfmg_relax_data -> relax_data);
      hypre_TFree(sys_pfmg_relax_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelax( void                 *sys_pfmg_relax_vdata,
                    hypre_SStructPMatrix *A,
                    hypre_SStructPVector *b,
                    hypre_SStructPVector *x                )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   HYPRE_Int               ierr = 0;

   ierr = hypre_NodeRelax((sys_pfmg_relax_data -> relax_data), A, b, x);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetup( void                 *sys_pfmg_relax_vdata,
                         hypre_SStructPMatrix *A,
                         hypre_SStructPVector *b,
                         hypre_SStructPVector *x                )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   void                   *relax_data       = (sys_pfmg_relax_data -> relax_data);
   HYPRE_Int               relax_type       = (sys_pfmg_relax_data -> relax_type);
   double                  jacobi_weight    = (sys_pfmg_relax_data -> jacobi_weight);

   HYPRE_Int               ierr = 0;
  
   if (relax_type == 1)
   {
      hypre_NodeRelaxSetWeight(relax_data, jacobi_weight);
   }
                                                                                                                                     
   ierr = hypre_NodeRelaxSetup((sys_pfmg_relax_data -> relax_data), A, b, x);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetType( void  *sys_pfmg_relax_vdata,
                           HYPRE_Int    relax_type       )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);
   HYPRE_Int               ierr = 0;

   (sys_pfmg_relax_data -> relax_type) = relax_type;

   switch(relax_type)
   {
      case 0: /* Jacobi */
      {
         hypre_Index  stride;
         hypre_Index  indices[1];

         hypre_NodeRelaxSetWeight(relax_data, 1.0);
         hypre_NodeRelaxSetNumNodesets(relax_data, 1);

         hypre_SetIndex(stride, 1, 1, 1);
         hypre_SetIndex(indices[0], 0, 0, 0);
         hypre_NodeRelaxSetNodeset(relax_data, 0, 1, stride, indices);
      }
      break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         hypre_Index  stride;
         hypre_Index  indices[4];

         hypre_NodeRelaxSetNumNodesets(relax_data, 2);

         hypre_SetIndex(stride, 2, 2, 2);

         /* define red points (point set 0) */
         hypre_SetIndex(indices[0], 1, 0, 0);
         hypre_SetIndex(indices[1], 0, 1, 0);
         hypre_SetIndex(indices[2], 0, 0, 1);
         hypre_SetIndex(indices[3], 1, 1, 1);
         hypre_NodeRelaxSetNodeset(relax_data, 0, 4, stride, indices);

         /* define black points (point set 1) */
         hypre_SetIndex(indices[0], 0, 0, 0);
         hypre_SetIndex(indices[1], 1, 1, 0);
         hypre_SetIndex(indices[2], 1, 0, 1);
         hypre_SetIndex(indices[3], 0, 1, 1);
         hypre_NodeRelaxSetNodeset(relax_data, 1, 4, stride, indices);
      }
      break;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetPreRelax
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SysPFMGRelaxSetJacobiWeight(void  *sys_pfmg_relax_vdata,
                                  double weight)
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
                                                                                                                                     
  (sys_pfmg_relax_data -> jacobi_weight)    = weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetPreRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetPreRelax( void  *sys_pfmg_relax_vdata )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);
   HYPRE_Int               relax_type = (sys_pfmg_relax_data -> relax_type);
   HYPRE_Int               ierr = 0;

   switch(relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
      break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         hypre_NodeRelaxSetNodesetRank(relax_data, 0, 0);
         hypre_NodeRelaxSetNodesetRank(relax_data, 1, 1);
      }
      break;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetPostRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetPostRelax( void  *sys_pfmg_relax_vdata )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);
   HYPRE_Int               relax_type = (sys_pfmg_relax_data -> relax_type);
   HYPRE_Int               ierr = 0;

   switch(relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
      break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         hypre_NodeRelaxSetNodesetRank(relax_data, 0, 1);
         hypre_NodeRelaxSetNodesetRank(relax_data, 1, 0);
      }
      break;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetTol( void   *sys_pfmg_relax_vdata,
                          double  tol              )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   HYPRE_Int               ierr = 0;

   ierr = hypre_NodeRelaxSetTol((sys_pfmg_relax_data -> relax_data), tol);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetMaxIter( void  *sys_pfmg_relax_vdata,
                              HYPRE_Int    max_iter         )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   HYPRE_Int               ierr = 0;

   ierr = hypre_NodeRelaxSetMaxIter((sys_pfmg_relax_data -> relax_data),
                                     max_iter);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetZeroGuess( void  *sys_pfmg_relax_vdata,
                                HYPRE_Int    zero_guess       )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   HYPRE_Int            ierr = 0;

   ierr = hypre_NodeRelaxSetZeroGuess((sys_pfmg_relax_data -> relax_data),
                                       zero_guess);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGRelaxSetTempVec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetTempVec( void               *sys_pfmg_relax_vdata,
                              hypre_SStructPVector *t                )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = sys_pfmg_relax_vdata;
   HYPRE_Int            ierr = 0;

   ierr = hypre_NodeRelaxSetTempVec((sys_pfmg_relax_data -> relax_data), t);

   return ierr;
}

