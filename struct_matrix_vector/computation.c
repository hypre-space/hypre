/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 * 
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_GetComputeInfo:
 *--------------------------------------------------------------------------*/

void
hypre_GetComputeInfo( hypre_BoxArrayArray  **send_boxes_ptr,
                    hypre_BoxArrayArray  **recv_boxes_ptr,
                    int               ***send_box_ranks_ptr,
                    int               ***recv_box_ranks_ptr,
                    hypre_BoxArrayArray  **indt_boxes_ptr,
                    hypre_BoxArrayArray  **dept_boxes_ptr,
                    hypre_StructGrid      *grid,
                    hypre_StructStencil   *stencil        )
{
   /* output variables */
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                  **send_box_ranks;
   int                  **recv_box_ranks;
   hypre_BoxArrayArray     *indt_boxes;
   hypre_BoxArrayArray     *dept_boxes;

   /* internal variables */
   hypre_BoxArray          *boxes;
   hypre_BoxArray          *all_boxes;

   hypre_Box               *box0;

   int                    i;

#ifdef OVERLAP_COMM_COMP
   hypre_BoxArray          *send_box_a;
   hypre_BoxArray          *recv_box_a;
   hypre_BoxArray          *indt_box_a;
   hypre_BoxArray          *dept_box_a;

   hypre_BoxArrayArray     *box_aa0;
   hypre_BoxArray          *box_a0;
   hypre_BoxArray          *box_a1;
                         
   int                    j, k;
#endif

   /*------------------------------------------------------
    * Extract needed grid info
    *------------------------------------------------------*/

   boxes     = hypre_StructGridBoxes(grid);
   all_boxes = hypre_StructGridAllBoxes(grid);

   /*------------------------------------------------------
    * Get communication info
    *------------------------------------------------------*/

   hypre_GetCommInfo(&send_boxes, &recv_boxes,
                   &send_box_ranks, &recv_box_ranks,
                   grid, stencil );

   /*------------------------------------------------------
    * Set up the dependent boxes
    *------------------------------------------------------*/

   dept_boxes = hypre_NewBoxArrayArray(hypre_BoxArraySize(boxes));

#ifdef OVERLAP_COMM_COMP

   hypre_ForBoxI(i, boxes)
   {
      /* grow `recv_boxes' by stencil transpose to get `box_aa0' */
      recv_box_a = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      box_aa0 = hypre_GrowBoxArrayByStencil(recv_box_a, stencil, 1);

      /* intersect `box_aa0' with `boxes' to create `dept_box_a' */
      dept_box_a = hypre_NewBoxArray();
      hypre_ForBoxArrayI(j, box_aa0)
      {
         box_a0 = hypre_BoxArrayArrayBoxArray(box_aa0, i);

         hypre_ForBoxI(k, box_a0)
         {
            box0 = hypre_IntersectBoxes(hypre_BoxArrayBox(box_a0, k),
                                      hypre_BoxArrayBox(boxes, i));
            if (box0)
               hypre_AppendBox(box0, dept_box_a);
         }
      }
      hypre_FreeBoxArrayArray(box_aa0);

      /* append `send_boxes' to `dept_box_a' */
      send_box_a = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      hypre_ForBoxI(j, send_box_a)
      {
         box0 = hypre_DuplicateBox(hypre_BoxArrayBox(send_box_a, j));
         hypre_AppendBox(box0, dept_box_a);
      }

      /* union `dept_box_a' to minimize size of `dept_boxes' */
      hypre_BoxArrayArrayBoxArray(dept_boxes, i) =
         hypre_UnionBoxArray(dept_box_a);

      hypre_FreeBoxArray(dept_box_a);
      hypre_FreeBoxArrayArray(box_aa0);
   }

#else

   hypre_ForBoxI(i, boxes)
   {
      box0 = hypre_DuplicateBox(hypre_BoxArrayBox(boxes, i));
      hypre_AppendBox(box0, hypre_BoxArrayArrayBoxArray(dept_boxes, i));
   }

#endif

   /*------------------------------------------------------
    * Set up the independent boxes
    *------------------------------------------------------*/

   indt_boxes = hypre_NewBoxArrayArray(hypre_BoxArraySize(boxes));

#ifdef OVERLAP_COMM_COMP

   /* subtract `dept_boxes' from `boxes' */
   hypre_ForBoxI(i, boxes)
   {
      dept_box_a = hypre_BoxArrayArrayBoxArray(dept_boxes, i);

      /* initialize `indt_box_a' */
      indt_box_a = hypre_BoxArrayArrayBoxArray(indt_boxes, i);
      hypre_AppendBox(hypre_DuplicateBox(hypre_BoxArrayBox(boxes, i)), indt_box_a);

      /* subtract `dept_box_a' from `indt_box_a' */
      hypre_ForBoxI(j, dept_box_a)
      {
         box_a0 = hypre_NewBoxArray();

         hypre_ForBoxI(k, indt_box_array)
         {
            box_a1 = hypre_SubtractBoxes(hypre_BoxArrayBox(indt_box_a, k),
                                       hypre_BoxArrayBox(dept_box_a, j));
            hypre_AppendBoxArray(box_a1, box_a0);
            hypre_FreeBoxArrayShell(box_a1);
         }

         hypre_FreeBoxArray(indt_box_a);
         indt_box_a = box_a0;
      }

      /* union `indt_box_a' to minimize size of `indt_boxes' */
      hypre_BoxArrayArrayBoxArray(indt_boxes, i) =
         hypre_UnionBoxArray(indt_box_a);
   }

#else

#endif

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_box_ranks_ptr = send_box_ranks;
   *recv_box_ranks_ptr = recv_box_ranks;
   *indt_boxes_ptr = indt_boxes;
   *dept_boxes_ptr = dept_boxes;
}

/*--------------------------------------------------------------------------
 * hypre_NewComputePkg:
 *--------------------------------------------------------------------------*/

hypre_ComputePkg *
hypre_NewComputePkg( hypre_SBoxArrayArray  *send_sboxes,
                   hypre_SBoxArrayArray  *recv_sboxes,
                   int                **send_box_ranks,
                   int                **recv_box_ranks,
                   hypre_SBoxArrayArray  *indt_sboxes,
                   hypre_SBoxArrayArray  *dept_sboxes,
                   hypre_StructGrid      *grid,
                   hypre_BoxArray        *data_space,
                   int                  num_values     )
{
   hypre_ComputePkg  *compute_pkg;

   compute_pkg = hypre_CTAlloc(hypre_ComputePkg, 1);

   hypre_ComputePkgCommPkg(compute_pkg)     =
      hypre_NewCommPkg(send_sboxes, recv_sboxes,
                     send_box_ranks, recv_box_ranks,
                     grid, data_space, num_values);

   hypre_ComputePkgIndtSBoxes(compute_pkg)   = indt_sboxes;
   hypre_ComputePkgDeptSBoxes(compute_pkg)   = dept_sboxes;

   hypre_ComputePkgGrid(compute_pkg)        = grid;
   hypre_ComputePkgDataSpace(compute_pkg)   = data_space;
   hypre_ComputePkgNumValues(compute_pkg)   = num_values;

   return compute_pkg;
}

/*--------------------------------------------------------------------------
 * hypre_FreeComputePkg:
 *--------------------------------------------------------------------------*/

void
hypre_FreeComputePkg( hypre_ComputePkg *compute_pkg )
{
   if (compute_pkg)
   {
      hypre_FreeCommPkg(hypre_ComputePkgCommPkg(compute_pkg));

      hypre_FreeSBoxArrayArray(hypre_ComputePkgIndtSBoxes(compute_pkg));
      hypre_FreeSBoxArrayArray(hypre_ComputePkgDeptSBoxes(compute_pkg));

      hypre_TFree(compute_pkg);
   }
}

/*--------------------------------------------------------------------------
 * hypre_InitializeIndtComputations:
 *--------------------------------------------------------------------------*/

hypre_CommHandle *
hypre_InitializeIndtComputations( hypre_ComputePkg *compute_pkg,
                                double         *data        )
{
   hypre_CommPkg *comm_pkg = hypre_ComputePkgCommPkg(compute_pkg);

   return hypre_InitializeCommunication(comm_pkg, data);
}

/*--------------------------------------------------------------------------
 * hypre_FinalizeIndtComputations:
 *--------------------------------------------------------------------------*/

void
hypre_FinalizeIndtComputations( hypre_CommHandle *comm_handle )
{
   hypre_FinalizeCommunication(comm_handle);
}
