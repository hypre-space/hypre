/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"
#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif

/*--------------------------------------------------------------------------
 * hypre_PFMGNewInterpOp
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_PFMGNewInterpOp( hypre_StructMatrix *A,
                       hypre_StructGrid   *cgrid,
                       int                 cdir  )
{
   hypre_StructMatrix   *P;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   int                   stencil_size;
   int                   stencil_dim;
                       
   int                   num_ghost[] = {1, 1, 1, 1, 1, 1};
                       
   int                   i;

   /* set up stencil */
   stencil_size = 2;
   stencil_dim = hypre_StructStencilDim(hypre_StructMatrixStencil(A));
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_SetIndex(stencil_shape[i], 0, 0, 0);
   }
   hypre_IndexD(stencil_shape[0], cdir) = -1;
   hypre_IndexD(stencil_shape[1], cdir) =  1;
   stencil = hypre_NewStructStencil(stencil_dim, stencil_size, stencil_shape);

   /* set up matrix */
   P = hypre_NewStructMatrix(hypre_StructMatrixComm(A), cgrid, stencil);
   hypre_SetStructMatrixNumGhost(P, num_ghost);
   hypre_InitializeStructMatrixShell(P);
 
   return P;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetupInterpOp
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetupInterpOp( hypre_StructMatrix *A,
                         int                 cdir,
                         hypre_Index         findex,
                         hypre_Index         stride,
                         hypre_StructMatrix *P      )
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *A_data_box;
   hypre_Box             *P_data_box;
                        
   int                    Ai;
   int                    Pi;
                        
   double                *Ap;
   double                *Pp0, *Pp1;
   double                 center;
                        
   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   int                    stencil_size;
   hypre_StructStencil   *P_stencil;
   hypre_Index           *P_stencil_shape;
                        
   int                    Astenc;
   int                    Pstenc0, Pstenc1;
                        
   hypre_Index            loop_size;
   hypre_Index            start;
   hypre_IndexRef         startc;
   hypre_Index            stridec;
                        
   int                    i, si;
   int                    loopi, loopj, loopk;

   int                    ierr = 0;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   P_stencil       = hypre_StructMatrixStencil(P);
   P_stencil_shape = hypre_StructStencilShape(P_stencil);

   hypre_SetIndex(stridec, 1, 1, 1);

   /*----------------------------------------------------------
    * Compute P
    *----------------------------------------------------------*/

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(P));
   hypre_ForBoxI(i, compute_boxes)
      {
         compute_box = hypre_BoxArrayBox(compute_boxes, i);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         P_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), i);

         Pp0 = hypre_StructMatrixBoxData(P, i, 0);
         Pp1 = hypre_StructMatrixBoxData(P, i, 1);

         Pstenc0 = hypre_IndexD(P_stencil_shape[0], cdir);
         Pstenc1 = hypre_IndexD(P_stencil_shape[1], cdir);

         startc  = hypre_BoxIMin(compute_box);
         hypre_PFMGMapCoarseToFine(startc, findex, stride, start);

         hypre_GetStrideBoxSize(compute_box, stridec, loop_size);
         hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                        A_data_box, start,  stride,  Ai,
                        P_data_box, startc, stridec, Pi,
                        {
                           center  = 0.0;
                           Pp0[Pi] = 0.0;
                           Pp1[Pi] = 0.0;

                           for (si = 0; si < stencil_size; si++)
                           {
                              Ap = hypre_StructMatrixBoxData(A, i, si);
                              Astenc = hypre_IndexD(stencil_shape[si], cdir);

                              if (Astenc == 0)
                              {
                                 center += Ap[Ai];
                              }
                              else if (Astenc == Pstenc0)
                              {
                                 Pp0[Pi] -= Ap[Ai];
                              }
                              else if (Astenc == Pstenc1)
                              {
                                 Pp1[Pi] -= Ap[Ai];
                              }
                           }

                           Pp0[Pi] /= center;
                           Pp1[Pi] /= center;
                        });
      }

   hypre_AssembleStructMatrix(P);

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return ierr;
}

