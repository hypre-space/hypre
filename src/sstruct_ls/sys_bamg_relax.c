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

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   void                   *relax_data;
   HYPRE_Int               relax_type;
   HYPRE_Real              jacobi_weight;
   HYPRE_SStructSolver     solver;

} hypre_SysBAMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SysBAMGRelaxCreate( MPI_Comm  comm, HYPRE_Int relax_type )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data;
   void * relax_data           = NULL;
   HYPRE_SStructSolver  solver = NULL;

   sys_bamg_relax_data = hypre_CTAlloc(hypre_SysBAMGRelaxData, 1);
   if (relax_type < 10)
   {
      relax_data = hypre_NodeRelaxCreate(comm);
   }
   else if(relax_type == 10)
   {
      HYPRE_SStructGMRESCreate(comm, &solver);
      
   }
   (sys_bamg_relax_data -> relax_data) = relax_data; 
   (sys_bamg_relax_data -> solver)     = solver; 
   (sys_bamg_relax_data -> relax_type) = relax_type;

   /* Do more solver specific setup */
   switch(relax_type)
   {
      case 0: /* Jacobi */
      {
         hypre_Index  stride;
         hypre_Index  indices[1];

         hypre_NodeRelaxSetWeight(relax_data, 1.0);
         hypre_NodeRelaxSetNumNodesets(relax_data, 1);

         hypre_SetIndex(stride, 1);
         hypre_SetIndex(indices[0], 0);
         hypre_NodeRelaxSetNodeset(relax_data, 0, 1, stride, indices);
      }
      break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         // XXX Hard-wired to MAXDIM=3

         hypre_Index  stride;
         hypre_Index  indices[4];

         hypre_NodeRelaxSetNumNodesets(relax_data, 2);

         hypre_SetIndex(stride, 2);

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



   return (void *) sys_bamg_relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxDestroy( void *sys_bamg_relax_vdata )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   HYPRE_Int               relax_type          = (sys_bamg_relax_data -> relax_type);

   if (sys_bamg_relax_data)
   {
      if(relax_type < 10)
      {
         hypre_NodeRelaxDestroy(sys_bamg_relax_data -> relax_data);
      }
      else if (relax_type == 10)
      {
         HYPRE_SStructGMRESDestroy(sys_bamg_relax_data -> solver);
      }
      hypre_TFree(sys_bamg_relax_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelax( void                 *sys_bamg_relax_vdata,
                    hypre_SStructPMatrix *A,
                    hypre_SStructPVector *b,
                    hypre_SStructPVector *x                )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   HYPRE_Int               relax_type          = (sys_bamg_relax_data -> relax_type);
   HYPRE_SStructSolver     solver              = (sys_bamg_relax_data -> solver);

   if (relax_type < 10)
   {
      hypre_NodeRelax((sys_bamg_relax_data -> relax_data), A, b, x);
   }
   else if (relax_type == 10)
   {
      HYPRE_SStructGMRESSolve(solver, A, b, x);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxSetup( void                 *sys_bamg_relax_vdata,
                         hypre_SStructPMatrix *A,
                         hypre_SStructPVector *b,
                         hypre_SStructPVector *x                )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   void                   *relax_data    = (sys_bamg_relax_data -> relax_data);
   HYPRE_Int               relax_type    = (sys_bamg_relax_data -> relax_type);
   HYPRE_Real              jacobi_weight = (sys_bamg_relax_data -> jacobi_weight);
   HYPRE_SStructSolver     solver        = (sys_bamg_relax_data -> solver);

   if (relax_type == 1)
   {
      hypre_NodeRelaxSetWeight(relax_data, jacobi_weight);
   }
   
   if (relax_type < 10)
   {
      hypre_NodeRelaxSetup((sys_bamg_relax_data -> relax_data), A, b, x);
   }
   else if (relax_type == 10)
   {
      HYPRE_SStructGMRESSetup(solver, A, b, x);
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxSetJacobiWeight(void  *sys_bamg_relax_vdata,
                                  HYPRE_Real weight)
{
   /* Only useful for relax_types < 10, the non-Krylov relaxation types */
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
                                                                                                                                     
   (sys_bamg_relax_data -> jacobi_weight)    = weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxSetPreRelax( void  *sys_bamg_relax_vdata )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   void                   *relax_data = (sys_bamg_relax_data -> relax_data);
   HYPRE_Int               relax_type = (sys_bamg_relax_data -> relax_type);

   /* Only useful for relax_types < 10, the non-Krylov relaxation types */
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxSetPostRelax( void  *sys_bamg_relax_vdata )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   void                   *relax_data = (sys_bamg_relax_data -> relax_data);
   HYPRE_Int               relax_type = (sys_bamg_relax_data -> relax_type);

   /* Only useful for relax_types < 10, the non-Krylov relaxation types */
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxSetTol( void   *sys_bamg_relax_vdata,
                          HYPRE_Real  tol              )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   HYPRE_Int               relax_type          = (sys_bamg_relax_data -> relax_type);
   HYPRE_SStructSolver     solver              = (sys_bamg_relax_data -> solver);

   if (relax_type < 10)
   {
      hypre_NodeRelaxSetTol((sys_bamg_relax_data -> relax_data), tol);
   }
   else if (relax_type == 10)
   {
      HYPRE_SStructGMRESSetTol(solver, tol);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxSetMaxIter( void  *sys_bamg_relax_vdata,
                              HYPRE_Int    max_iter         )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   HYPRE_Int               relax_type          = (sys_bamg_relax_data -> relax_type);
   HYPRE_SStructSolver     solver              = (sys_bamg_relax_data -> solver);

   if (relax_type < 10)
   {
      hypre_NodeRelaxSetMaxIter((sys_bamg_relax_data -> relax_data), max_iter);
   }
   else if (relax_type == 10)
   {
      HYPRE_SStructGMRESSetMaxIter(solver, max_iter);
      HYPRE_SStructGMRESSetKDim(solver, max_iter+2);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxSetZeroGuess( void  *sys_bamg_relax_vdata,
                                HYPRE_Int    zero_guess       )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   HYPRE_Int               relax_type          = (sys_bamg_relax_data -> relax_type);

   /* Only useful for relax_types < 10, the non-Krylov relaxation types */
   if(relax_type < 10)
   {
      hypre_NodeRelaxSetZeroGuess((sys_bamg_relax_data -> relax_data), zero_guess);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysBAMGRelaxSetTempVec( void               *sys_bamg_relax_vdata,
                              hypre_SStructPVector *t                )
{
   hypre_SysBAMGRelaxData *sys_bamg_relax_data = (hypre_SysBAMGRelaxData *)sys_bamg_relax_vdata;
   HYPRE_Int               relax_type          = (sys_bamg_relax_data -> relax_type);

   /* Only useful for relax_types < 10, the non-Krylov relaxation types */
   if(relax_type < 10)
   {
      hypre_NodeRelaxSetTempVec((sys_bamg_relax_data -> relax_data), t);
   }

   return hypre_error_flag;
}

