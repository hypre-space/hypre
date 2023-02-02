/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   void                   *relax_data;
   HYPRE_Int               relax_type;
   HYPRE_Real              jacobi_weight;

} hypre_SysPFMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SysPFMGRelaxCreate( MPI_Comm  comm )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data;

   sys_pfmg_relax_data = hypre_CTAlloc(hypre_SysPFMGRelaxData,  1, HYPRE_MEMORY_HOST);
   (sys_pfmg_relax_data -> relax_data) = hypre_NodeRelaxCreate(comm);
   (sys_pfmg_relax_data -> relax_type) = 0;        /* Weighted Jacobi */

   return (void *) sys_pfmg_relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxDestroy( void *sys_pfmg_relax_vdata )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   if (sys_pfmg_relax_data)
   {
      hypre_NodeRelaxDestroy(sys_pfmg_relax_data -> relax_data);
      hypre_TFree(sys_pfmg_relax_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelax( void                 *sys_pfmg_relax_vdata,
                    hypre_SStructPMatrix *A,
                    hypre_SStructPVector *b,
                    hypre_SStructPVector *x                )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   hypre_NodeRelax((sys_pfmg_relax_data -> relax_data), A, b, x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetup( void                 *sys_pfmg_relax_vdata,
                         hypre_SStructPMatrix *A,
                         hypre_SStructPVector *b,
                         hypre_SStructPVector *x                )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;
   void                   *relax_data    = (sys_pfmg_relax_data -> relax_data);
   HYPRE_Int               relax_type    = (sys_pfmg_relax_data -> relax_type);
   HYPRE_Real              jacobi_weight = (sys_pfmg_relax_data -> jacobi_weight);

   if (relax_type == 1)
   {
      hypre_NodeRelaxSetWeight(relax_data, jacobi_weight);
   }

   hypre_NodeRelaxSetup((sys_pfmg_relax_data -> relax_data), A, b, x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetType( void  *sys_pfmg_relax_vdata,
                           HYPRE_Int    relax_type       )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);

   (sys_pfmg_relax_data -> relax_type) = relax_type;

   switch (relax_type)
   {
      case 0: /* Jacobi */
      {
         hypre_Index  stride;
         hypre_Index  indices[1];

         hypre_NodeRelaxSetWeight(relax_data, 1.0);
         hypre_NodeRelaxSetNumNodesets(relax_data, 1);

         hypre_SetIndex3(stride, 1, 1, 1);
         hypre_SetIndex3(indices[0], 0, 0, 0);
         hypre_NodeRelaxSetNodeset(relax_data, 0, 1, stride, indices);
      }
      break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         hypre_Index  stride;
         hypre_Index  indices[4];

         hypre_NodeRelaxSetNumNodesets(relax_data, 2);

         hypre_SetIndex3(stride, 2, 2, 2);

         /* define red points (point set 0) */
         hypre_SetIndex3(indices[0], 1, 0, 0);
         hypre_SetIndex3(indices[1], 0, 1, 0);
         hypre_SetIndex3(indices[2], 0, 0, 1);
         hypre_SetIndex3(indices[3], 1, 1, 1);
         hypre_NodeRelaxSetNodeset(relax_data, 0, 4, stride, indices);

         /* define black points (point set 1) */
         hypre_SetIndex3(indices[0], 0, 0, 0);
         hypre_SetIndex3(indices[1], 1, 1, 0);
         hypre_SetIndex3(indices[2], 1, 0, 1);
         hypre_SetIndex3(indices[3], 0, 1, 1);
         hypre_NodeRelaxSetNodeset(relax_data, 1, 4, stride, indices);
      }
      break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetJacobiWeight(void  *sys_pfmg_relax_vdata,
                                  HYPRE_Real weight)
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   (sys_pfmg_relax_data -> jacobi_weight)    = weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetPreRelax( void  *sys_pfmg_relax_vdata )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);
   HYPRE_Int               relax_type = (sys_pfmg_relax_data -> relax_type);

   switch (relax_type)
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetPostRelax( void  *sys_pfmg_relax_vdata )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);
   HYPRE_Int               relax_type = (sys_pfmg_relax_data -> relax_type);

   switch (relax_type)
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetTol( void   *sys_pfmg_relax_vdata,
                          HYPRE_Real  tol              )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   hypre_NodeRelaxSetTol((sys_pfmg_relax_data -> relax_data), tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetMaxIter( void  *sys_pfmg_relax_vdata,
                              HYPRE_Int    max_iter         )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   hypre_NodeRelaxSetMaxIter((sys_pfmg_relax_data -> relax_data), max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetZeroGuess( void  *sys_pfmg_relax_vdata,
                                HYPRE_Int    zero_guess       )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   hypre_NodeRelaxSetZeroGuess((sys_pfmg_relax_data -> relax_data), zero_guess);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGRelaxSetTempVec( void               *sys_pfmg_relax_vdata,
                              hypre_SStructPVector *t                )
{
   hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   hypre_NodeRelaxSetTempVec((sys_pfmg_relax_data -> relax_data), t);

   return hypre_error_flag;
}

