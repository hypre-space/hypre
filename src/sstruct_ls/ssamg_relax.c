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
 * Notes:
 *        1) relax_weight can vary across different parts
 *--------------------------------------------------------------------------*/

typedef struct hypre_SSAMGRelaxData_struct
{
   void                  **relax_data;
   HYPRE_Real             *relax_weight;
   HYPRE_Int               nparts;
   HYPRE_Int               relax_type;
   HYPRE_Int               zero_guess;
   HYPRE_Int               max_iter;

   hypre_SStructVector    *r;
   hypre_SStructVector    *e;
} hypre_SSAMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxCreate( MPI_Comm    comm,
                        HYPRE_Int   nparts,
                        void      **ssamg_relax_vdata_ptr )
{
   hypre_SSAMGRelaxData     *ssamg_relax_data;

   HYPRE_Int                 part;

   ssamg_relax_data = hypre_CTAlloc(hypre_SSAMGRelaxData, 1);
   (ssamg_relax_data -> nparts)       = nparts;
   (ssamg_relax_data -> zero_guess)   = 0;
   (ssamg_relax_data -> relax_type)   = 0;
   (ssamg_relax_data -> relax_data)   = hypre_CTAlloc(void *, nparts);
   (ssamg_relax_data -> relax_weight) = hypre_CTAlloc(HYPRE_Real, nparts);
   for (part = 0; part < nparts; part++)
   {
      (ssamg_relax_data -> relax_weight[part]) = 1.0;
      (ssamg_relax_data -> relax_data[part]) = hypre_NodeRelaxCreate(comm);
   }

   *ssamg_relax_vdata_ptr = (void *) ssamg_relax_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxDestroy( void *ssamg_relax_vdata )
{
   hypre_SSAMGRelaxData *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;

   HYPRE_Int             part, nparts;

   if (ssamg_relax_data)
   {
      nparts = (ssamg_relax_data -> nparts);
      for (part = 0; part < nparts; part++)
      {
         hypre_NodeRelaxDestroy((ssamg_relax_data -> relax_data[part]));
      }

      HYPRE_SStructVectorDestroy(ssamg_relax_data -> r);
      HYPRE_SStructVectorDestroy(ssamg_relax_data -> e);
      hypre_TFree(ssamg_relax_data -> relax_weight);
      hypre_TFree(ssamg_relax_data -> relax_data);
      hypre_TFree(ssamg_relax_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelax( void                 *ssamg_relax_vdata,
                  void                 *matvec_vdata,
                  hypre_SStructMatrix  *A,
                  hypre_SStructVector  *b,
                  hypre_SStructVector  *x )
{
   hypre_SSAMGRelaxData    *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *pr;
   hypre_SStructPVector    *pe;
   hypre_SStructPVector    *pb;
   hypre_SStructPVector    *px;
   hypre_SStructVector     *r = (ssamg_relax_data -> r);
   hypre_SStructVector     *e = (ssamg_relax_data -> e);
   void                   **relax_data = (ssamg_relax_data -> relax_data);
   HYPRE_Int                zero_guess = (ssamg_relax_data -> zero_guess);
   HYPRE_Int                max_iter   = (ssamg_relax_data -> max_iter);

   HYPRE_Int                nparts, nparts_A;
   HYPRE_Int                nparts_b, nparts_x;
   HYPRE_Int                part;
   HYPRE_Int                iter;

   nparts   = (ssamg_relax_data -> nparts);
   nparts_A = hypre_SStructMatrixNParts(A);
   nparts_b = hypre_SStructVectorNParts(b);
   nparts_x = hypre_SStructVectorNParts(x);

   if ( (nparts_A != nparts) || (nparts_b != nparts) || (nparts_x != nparts) )
   {
      // TODO: handle error
      return hypre_error_flag;
   }

   iter = 0;
   if (zero_guess)
   {
      for (part = 0; part < nparts; part++)
      {
         pA = hypre_SStructMatrixPMatrix(A, part);
         pb = hypre_SStructVectorPVector(b, part);
         px = hypre_SStructVectorPVector(x, part);

         hypre_NodeRelax(relax_data[part], pA, pb, px);
      }
      iter++;
   }

   while (iter < max_iter)
   {
      hypre_SStructCopy(b, r);
      hypre_SStructMatvecCompute(matvec_vdata, -1.0, A, x, 1.0, r);
      hypre_SSAMGRelaxSetZeroGuess(ssamg_relax_data, 1);

      for (part = 0; part < nparts; part++)
      {
         pA = hypre_SStructMatrixPMatrix(A, part);
         pr = hypre_SStructVectorPVector(r, part);
         pe = hypre_SStructVectorPVector(e, part);

         hypre_NodeRelax(relax_data[part], pA, pr, pe);
      }

      hypre_SStructAxpy(1.0, e, x);
      hypre_SSAMGRelaxSetZeroGuess(ssamg_relax_data, zero_guess);
      iter++;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetup( void                *ssamg_relax_vdata,
                       hypre_SStructMatrix *A,
                       hypre_SStructVector *b,
                       hypre_SStructVector *x )
{
   hypre_SSAMGRelaxData  *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;

   void                 **relax_data   = (ssamg_relax_data -> relax_data);
   HYPRE_Real            *relax_weight = (ssamg_relax_data -> relax_weight);
   HYPRE_Int              relax_type   = (ssamg_relax_data -> relax_type);

   MPI_Comm               comm = hypre_SStructMatrixComm(A);
   hypre_SStructGrid     *grid_b = hypre_SStructVectorGrid(b);
   hypre_SStructGrid     *grid_x = hypre_SStructVectorGrid(x);
   hypre_SStructVector   *r;
   hypre_SStructVector   *e;
   hypre_SStructPMatrix  *pA;
   hypre_SStructPVector  *pb;
   hypre_SStructPVector  *px;

   HYPRE_Int              nparts, nparts_A, nparts_b, nparts_x;
   HYPRE_Int              part;

   nparts   = (ssamg_relax_data -> nparts);
   nparts_A = hypre_SStructMatrixNParts(A);
   nparts_b = hypre_SStructVectorNParts(b);
   nparts_x = hypre_SStructVectorNParts(x);

   if ( (nparts_A != nparts) || (nparts_b != nparts) || (nparts_x != nparts) )
   {
      // TODO: handle error
      return hypre_error_flag;
   }

   HYPRE_SStructVectorCreate(comm, grid_b, &r);
   HYPRE_SStructVectorInitialize(r);
   HYPRE_SStructVectorAssemble(r);
   (ssamg_relax_data -> r) = r;

   HYPRE_SStructVectorCreate(comm, grid_x, &e);
   HYPRE_SStructVectorInitialize(e);
   HYPRE_SStructVectorAssemble(e);
   (ssamg_relax_data -> e) = e;

   for (part = 0; part < nparts_A; part++)
   {
      pA = hypre_SStructMatrixPMatrix(A, part);
      pb = hypre_SStructVectorPVector(b, part);
      px = hypre_SStructVectorPVector(x, part);

      if (relax_type == 1)
      {
         hypre_NodeRelaxSetWeight(relax_data[part], relax_weight[part]);
      }

      hypre_NodeRelaxSetup(relax_data[part], pA, pb, px);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetPreRelax( void  *ssamg_relax_vdata )
{
   hypre_SSAMGRelaxData  *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   void                 **relax_data = (ssamg_relax_data -> relax_data);
   HYPRE_Int              relax_type = (ssamg_relax_data -> relax_type);
   HYPRE_Int              nparts     = (ssamg_relax_data -> nparts);

   HYPRE_Int              part;

   switch(relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         for (part = 0; part < nparts; part++)
         {
            hypre_NodeRelaxSetNodesetRank(relax_data[part], 0, 0);
            hypre_NodeRelaxSetNodesetRank(relax_data[part], 1, 1);
         }
      }
      break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetPostRelax( void  *ssamg_relax_vdata )
{
   hypre_SSAMGRelaxData  *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   void                 **relax_data = (ssamg_relax_data -> relax_data);
   HYPRE_Int              relax_type = (ssamg_relax_data -> relax_type);
   HYPRE_Int              nparts     = (ssamg_relax_data -> nparts);

   HYPRE_Int              part;

   switch(relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         for (part = 0; part < nparts; part++)
         {
            hypre_NodeRelaxSetNodesetRank(relax_data[part], 0, 1);
            hypre_NodeRelaxSetNodesetRank(relax_data[part], 1, 0);
         }
      }
      break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetTol( void       *ssamg_relax_vdata,
                        HYPRE_Real  tol               )
{
   hypre_SSAMGRelaxData  *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   void                 **relax_data       = (ssamg_relax_data -> relax_data);

   HYPRE_Int              part, nparts;

   nparts = (ssamg_relax_data -> nparts);
   for (part = 0; part < nparts; part++)
   {
      hypre_NodeRelaxSetTol(relax_data[part], tol);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetWeights( void       *ssamg_relax_vdata,
                            HYPRE_Int  *pids,
                            HYPRE_Real *relax_weights     )
{
   hypre_SSAMGRelaxData  *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   HYPRE_Int              part, nparts;

   nparts = (ssamg_relax_data -> nparts);
   for (part = 0; part < nparts; part++)
   {
      (ssamg_relax_data -> relax_weight[part]) = relax_weights[pids[part]];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetMaxIter( void       *ssamg_relax_vdata,
                            HYPRE_Int   max_iter)
{
   hypre_SSAMGRelaxData  *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   void                 **relax_data       = (ssamg_relax_data -> relax_data);
   HYPRE_Int              nparts           = (ssamg_relax_data -> nparts);
   HYPRE_Int              part;

   (ssamg_relax_data -> max_iter) = max_iter;
   for (part = 0; part < nparts; part++)
   {
      hypre_NodeRelaxSetMaxIter(relax_data[part], 1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetZeroGuess( void      *ssamg_relax_vdata,
                              HYPRE_Int  zero_guess       )
{
   hypre_SSAMGRelaxData  *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   void                 **relax_data       = (ssamg_relax_data -> relax_data);
   HYPRE_Int              nparts           = (ssamg_relax_data -> nparts);

   HYPRE_Int              part;

   (ssamg_relax_data -> zero_guess) = zero_guess;
   for (part = 0; part < nparts; part++)
   {
      hypre_NodeRelaxSetZeroGuess(relax_data[part], zero_guess);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetType( void      *ssamg_relax_vdata,
                         HYPRE_Int  relax_type)
{
   hypre_SSAMGRelaxData *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   void                **relax_data       = (ssamg_relax_data -> relax_data);

   hypre_Index           stride;
   HYPRE_Int             part, nparts;

   nparts = (ssamg_relax_data -> nparts);
   (ssamg_relax_data -> relax_type) = relax_type;

   switch(relax_type)
   {
      case 0: /* Jacobi */
      {
         hypre_Index  indices[1];

         hypre_SetIndex(stride, 1);
         hypre_SetIndex(indices[0], 0);

         for (part = 0; part < nparts; part++)
         {
            hypre_NodeRelaxSetWeight(relax_data[part], 1.0);
            hypre_NodeRelaxSetNumNodesets(relax_data[part], 1);
            hypre_NodeRelaxSetNodeset(relax_data[part], 0, 1, stride, indices);
         }
      }
      break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         hypre_Index  red[4], black[4];

         hypre_SetIndex(stride, 2);

         /* define red points (point set 0) */
         hypre_SetIndex3(red[0], 1, 0, 0);
         hypre_SetIndex3(red[1], 0, 1, 0);
         hypre_SetIndex3(red[2], 0, 0, 1);
         hypre_SetIndex3(red[3], 1, 1, 1);

         /* define black points (point set 1) */
         hypre_SetIndex3(black[0], 0, 0, 0);
         hypre_SetIndex3(black[1], 1, 1, 0);
         hypre_SetIndex3(black[2], 1, 0, 1);
         hypre_SetIndex3(black[3], 0, 1, 1);

         for (part = 0; part < nparts; part++)
         {
            hypre_NodeRelaxSetNumNodesets(relax_data[part], 2);
            hypre_NodeRelaxSetNodeset(relax_data[part], 0, 4, stride, red);
            hypre_NodeRelaxSetNodeset(relax_data[part], 1, 4, stride, black);
         }
      }
      break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetTempVec( void                *ssamg_relax_vdata,
                            hypre_SStructVector *t                 )
{
   hypre_SSAMGRelaxData   *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   void                  **relax_data       = (ssamg_relax_data -> relax_data);
   hypre_SStructPVector   *pt;

   HYPRE_Int               part, nparts;

   nparts = (ssamg_relax_data -> nparts);
   for (part = 0; part < nparts; part++)
   {
      pt = hypre_SStructVectorPVector(t, part);
      hypre_NodeRelaxSetTempVec(relax_data[part], pt);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TODO: this and other assumes that relax_data is of type hypre_NodeRelaxData
 *       Need to make it general
 *--------------------------------------------------------------------------*/
#if 0
HYPRE_Int hypre_SSAMGRelaxGetMaxIter( void       *ssamg_relax_vdata,
                                      HYPRE_Int   part,
                                      HYPRE_Int  *max_iter )
{
   hypre_SSAMGRelaxData  *ssamg_relax_data = (hypre_SSAMGRelaxData *) ssamg_relax_vdata;
   void                 **relax_data       = (ssamg_relax_data -> relax_data);

   *max_iter = (relax_data[part] -> max_iter);

   return hypre_error_flag;
}
#endif
