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
 * zzz_GetComputeInfo:
 *--------------------------------------------------------------------------*/

void
zzz_GetComputeInfo( zzz_BoxArrayArray  **send_boxes_ptr,
                    zzz_BoxArrayArray  **recv_boxes_ptr,
                    int               ***send_box_ranks_ptr,
                    int               ***recv_box_ranks_ptr,
                    zzz_BoxArrayArray  **indt_boxes_ptr,
                    zzz_BoxArrayArray  **dept_boxes_ptr,
                    zzz_StructGrid      *grid,
                    zzz_StructStencil   *stencil        )
{
   /* output variables */
   zzz_BoxArrayArray     *send_boxes;
   zzz_BoxArrayArray     *recv_boxes;
   int                  **send_box_ranks;
   int                  **recv_box_ranks;
   zzz_BoxArrayArray     *indt_boxes;
   zzz_BoxArrayArray     *dept_boxes;

   /* internal variables */
   zzz_BoxArray          *boxes;
   zzz_BoxArray          *all_boxes;

   zzz_Box               *box0;

   int                    i;

#ifdef OVERLAP_COMM_COMP
   zzz_BoxArray          *send_box_a;
   zzz_BoxArray          *recv_box_a;
   zzz_BoxArray          *indt_box_a;
   zzz_BoxArray          *dept_box_a;

   zzz_BoxArrayArray     *box_aa0;
   zzz_BoxArray          *box_a0;
   zzz_BoxArray          *box_a1;
                         
   int                    j, k;
#endif

   /*------------------------------------------------------
    * Extract needed grid info
    *------------------------------------------------------*/

   boxes     = zzz_StructGridBoxes(grid);
   all_boxes = zzz_StructGridAllBoxes(grid);

   /*------------------------------------------------------
    * Get communication info
    *------------------------------------------------------*/

   zzz_GetCommInfo(&send_boxes, &recv_boxes,
                   &send_box_ranks, &recv_box_ranks,
                   grid, stencil );

   /*------------------------------------------------------
    * Set up the dependent boxes
    *------------------------------------------------------*/

   dept_boxes = zzz_NewBoxArrayArray(zzz_BoxArraySize(boxes));

#ifdef OVERLAP_COMM_COMP

   zzz_ForBoxI(i, boxes)
   {
      /* grow `recv_boxes' by stencil transpose to get `box_aa0' */
      recv_box_a = zzz_BoxArrayArrayBoxArray(recv_boxes, i);
      box_aa0 = zzz_GrowBoxArrayByStencil(recv_box_a, stencil, 1);

      /* intersect `box_aa0' with `boxes' to create `dept_box_a' */
      dept_box_a = zzz_NewBoxArray();
      zzz_ForBoxArrayI(j, box_aa0)
      {
         box_a0 = zzz_BoxArrayArrayBoxArray(box_aa0, i);

         zzz_ForBoxI(k, box_a0)
         {
            box0 = zzz_IntersectBoxes(zzz_BoxArrayBox(box_a0, k),
                                      zzz_BoxArrayBox(boxes, i));
            if (box0)
               zzz_AppendBox(box0, dept_box_a);
         }
      }
      zzz_FreeBoxArrayArray(box_aa0);

      /* append `send_boxes' to `dept_box_a' */
      send_box_a = zzz_BoxArrayArrayBoxArray(send_boxes, i);
      zzz_ForBoxI(j, send_box_a)
      {
         box0 = zzz_DuplicateBox(zzz_BoxArrayBox(send_box_a, j));
         zzz_AppendBox(box0, dept_box_a);
      }

      /* union `dept_box_a' to minimize size of `dept_boxes' */
      zzz_BoxArrayArrayBoxArray(dept_boxes, i) =
         zzz_UnionBoxArray(dept_box_a);

      zzz_FreeBoxArray(dept_box_a);
      zzz_FreeBoxArrayArray(box_aa0);
   }

#else

   zzz_ForBoxI(i, boxes)
   {
      box0 = zzz_DuplicateBox(zzz_BoxArrayBox(boxes, i));
      zzz_AppendBox(box0, zzz_BoxArrayArrayBoxArray(dept_boxes, i));
   }

#endif

   /*------------------------------------------------------
    * Set up the independent boxes
    *------------------------------------------------------*/

   indt_boxes = zzz_NewBoxArrayArray(zzz_BoxArraySize(boxes));

#ifdef OVERLAP_COMM_COMP

   /* subtract `dept_boxes' from `boxes' */
   zzz_ForBoxI(i, boxes)
   {
      dept_box_a = zzz_BoxArrayArrayBoxArray(dept_boxes, i);

      /* initialize `indt_box_a' */
      indt_box_a = zzz_BoxArrayArrayBoxArray(indt_boxes, i);
      zzz_AppendBox(zzz_DuplicateBox(zzz_BoxArrayBox(boxes, i)), indt_box_a);

      /* subtract `dept_box_a' from `indt_box_a' */
      zzz_ForBoxI(j, dept_box_a)
      {
         box_a0 = zzz_NewBoxArray();

         zzz_ForBoxI(k, indt_box_array)
         {
            box_a1 = zzz_SubtractBoxes(zzz_BoxArrayBox(indt_box_a, k),
                                       zzz_BoxArrayBox(dept_box_a, j));
            zzz_AppendBoxArray(box_a1, box_a0);
            zzz_FreeBoxArrayShell(box_a1);
         }

         zzz_FreeBoxArray(indt_box_a);
         indt_box_a = box_a0;
      }

      /* union `indt_box_a' to minimize size of `indt_boxes' */
      zzz_BoxArrayArrayBoxArray(indt_boxes, i) =
         zzz_UnionBoxArray(indt_box_a);
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
 * zzz_NewComputePkg:
 *--------------------------------------------------------------------------*/

zzz_ComputePkg *
zzz_NewComputePkg( zzz_SBoxArrayArray  *send_sboxes,
                   zzz_SBoxArrayArray  *recv_sboxes,
                   int                **send_box_ranks,
                   int                **recv_box_ranks,
                   zzz_SBoxArrayArray  *indt_sboxes,
                   zzz_SBoxArrayArray  *dept_sboxes,
                   zzz_StructGrid      *grid,
                   zzz_BoxArray        *data_space,
                   int                  num_values     )
{
   zzz_ComputePkg  *compute_pkg;

   compute_pkg = ctalloc(zzz_ComputePkg, 1);

   zzz_ComputePkgCommPkg(compute_pkg)     =
      zzz_NewCommPkg(send_sboxes, recv_sboxes,
                     send_box_ranks, recv_box_ranks,
                     grid, data_space, num_values);

   zzz_ComputePkgIndtSBoxes(compute_pkg)   = indt_sboxes;
   zzz_ComputePkgDeptSBoxes(compute_pkg)   = dept_sboxes;

   zzz_ComputePkgGrid(compute_pkg)        = grid;
   zzz_ComputePkgDataSpace(compute_pkg)   = data_space;
   zzz_ComputePkgNumValues(compute_pkg)   = num_values;

   return compute_pkg;
}

/*--------------------------------------------------------------------------
 * zzz_FreeComputePkg:
 *--------------------------------------------------------------------------*/

void
zzz_FreeComputePkg( zzz_ComputePkg *compute_pkg )
{
   if (compute_pkg)
   {
      zzz_FreeCommPkg(zzz_ComputePkgCommPkg(compute_pkg));

      zzz_FreeSBoxArrayArray(zzz_ComputePkgIndtSBoxes(compute_pkg));
      zzz_FreeSBoxArrayArray(zzz_ComputePkgDeptSBoxes(compute_pkg));

      tfree(compute_pkg);
   }
}

/*--------------------------------------------------------------------------
 * zzz_InitializeIndtComputations:
 *--------------------------------------------------------------------------*/

zzz_CommHandle *
zzz_InitializeIndtComputations( zzz_ComputePkg *compute_pkg,
                                double         *data        )
{
   zzz_CommPkg *comm_pkg = zzz_ComputePkgCommPkg(compute_pkg);

   return zzz_InitializeCommunication(comm_pkg, data);
}

/*--------------------------------------------------------------------------
 * zzz_FinalizeIndtComputations:
 *--------------------------------------------------------------------------*/

void
zzz_FinalizeIndtComputations( zzz_CommHandle *comm_handle )
{
   zzz_FinalizeCommunication(comm_handle);
}
