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

/*==========================================================================*/
/*==========================================================================*/
/** Return descriptions of communications and computations patterns
for a given grid-stencil computation.  If HYPRE\_OVERLAP\_COMM\_COMP
is defined, then the patterns are computed to allow for overlapping
communications and computations.  The default is no overlap.

{\bf Note:} This routine assumes that the grid boxes do not overlap.

{\bf Input files:}
headers.h

@return Error code.

@param grid [IN]
  computational grid
@param stencil [IN]
  computational stencil
@param send_boxes_ptr [OUT]
  description of the grid data to be sent to other processors.
@param recv_boxes_ptr [OUT]
  description of the grid data to be received from other processors.
@param send_processes_ptr [OUT]
  processors that data is to be sent to.
@param recv_processes_ptr [OUT]
  processors that data is to be received from.
@param indt_boxes_ptr [OUT]
  description of computations that do not depend on communicated data.
@param dept_boxes_ptr [OUT]
  description of computations that depend on communicated data.

@see hypre_CreateCommInfoFromStencil */
/*--------------------------------------------------------------------------*/

  int
  hypre_CreateComputeInfo( hypre_StructGrid      *grid,
                        hypre_StructStencil   *stencil,
                        hypre_BoxArrayArray  **send_boxes_ptr,
                        hypre_BoxArrayArray  **recv_boxes_ptr,
                        int                 ***send_processes_ptr,
                        int                 ***recv_processes_ptr,
                        hypre_BoxArrayArray  **indt_boxes_ptr,
                        hypre_BoxArrayArray  **dept_boxes_ptr     )
{
   int                      ierr = 0;

   /* output variables */
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                    **send_processes;
   int                    **recv_processes;
   hypre_BoxArrayArray     *indt_boxes;
   hypre_BoxArrayArray     *dept_boxes;

   /* internal variables */
   hypre_BoxArray          *boxes;

   hypre_BoxArray          *cbox_array;
   hypre_Box               *cbox;

   int                      i;

#ifdef HYPRE_OVERLAP_COMM_COMP
   hypre_Box               *rembox;
   hypre_Index             *stencil_shape;
   int                      border[3][2] = {{0, 0}, {0, 0}, {0, 0}};
   int                      cbox_array_size;
   int                      s, d;
#endif

   /*------------------------------------------------------
    * Extract needed grid info
    *------------------------------------------------------*/

   boxes = hypre_StructGridBoxes(grid);

   /*------------------------------------------------------
    * Get communication info
    *------------------------------------------------------*/

   hypre_CreateCommInfoFromStencil(grid, stencil,
                                   &send_boxes, &recv_boxes,
                                   &send_processes, &recv_processes);

#ifdef HYPRE_OVERLAP_COMM_COMP

   /*------------------------------------------------------
    * Compute border info
    *------------------------------------------------------*/

   stencil_shape = hypre_StructStencilShape(stencil);
   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
   {
      for (d = 0; d < 3; d++)
      {
         i = hypre_IndexD(stencil_shape[s], d);
         if (i < 0)
         {
            border[d][0] = hypre_max(border[d][0], -i);
         }
         else if (i > 0)
         {
            border[d][1] = hypre_max(border[d][1], i);
         }
      }
   }

   /*------------------------------------------------------
    * Set up the dependent boxes
    *------------------------------------------------------*/

   dept_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));

   rembox = hypre_BoxCreate();
   hypre_ForBoxI(i, boxes)
      {
         cbox_array = hypre_BoxArrayArrayBoxArray(dept_boxes, i);
         hypre_BoxArraySetSize(cbox_array, 6);

         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), rembox);
         cbox_array_size = 0;
         for (d = 0; d < 3; d++)
         {
            if ( (hypre_BoxVolume(rembox)) && (border[d][0]) )
            {
               cbox = hypre_BoxArrayBox(cbox_array, cbox_array_size);
               hypre_CopyBox(rembox, cbox);
               hypre_BoxIMaxD(cbox, d) =
                  hypre_BoxIMinD(cbox, d) + border[d][0] - 1;
               hypre_BoxIMinD(rembox, d) =
                  hypre_BoxIMinD(cbox, d) + border[d][0];
               cbox_array_size++;
            }
            if ( (hypre_BoxVolume(rembox)) && (border[d][1]) )
            {
               cbox = hypre_BoxArrayBox(cbox_array, cbox_array_size);
               hypre_CopyBox(rembox, cbox);
               hypre_BoxIMinD(cbox, d) =
                  hypre_BoxIMaxD(cbox, d) - border[d][1] + 1;
               hypre_BoxIMaxD(rembox, d) =
                  hypre_BoxIMaxD(cbox, d) - border[d][1];
               cbox_array_size++;
            }
         }
         hypre_BoxArraySetSize(cbox_array, cbox_array_size);
      }
   hypre_BoxDestroy(rembox);

   /*------------------------------------------------------
    * Set up the independent boxes
    *------------------------------------------------------*/

   indt_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));

   hypre_ForBoxI(i, boxes)
      {
         cbox_array = hypre_BoxArrayArrayBoxArray(indt_boxes, i);
         hypre_BoxArraySetSize(cbox_array, 1);
         cbox = hypre_BoxArrayBox(cbox_array, 0);
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), cbox);

         for (d = 0; d < 3; d++)
         {
            if ( (border[d][0]) )
            {
               hypre_BoxIMinD(cbox, d) += border[d][0];
            }
            if ( (border[d][1]) )
            {
               hypre_BoxIMaxD(cbox, d) -= border[d][1];
            }
         }
      }

#else

   /*------------------------------------------------------
    * Set up the independent boxes
    *------------------------------------------------------*/

   indt_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));

   /*------------------------------------------------------
    * Set up the dependent boxes
    *------------------------------------------------------*/

   dept_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));

   hypre_ForBoxI(i, boxes)
      {
         cbox_array = hypre_BoxArrayArrayBoxArray(dept_boxes, i);
         hypre_BoxArraySetSize(cbox_array, 1);
         cbox = hypre_BoxArrayBox(cbox_array, 0);
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), cbox);
      }

#endif

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_processes_ptr = send_processes;
   *recv_processes_ptr = recv_processes;
   *indt_boxes_ptr = indt_boxes;
   *dept_boxes_ptr = dept_boxes;

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Create a computation package from a grid-based description of a
communication-computation pattern.

{\bf Note:}
The input boxes and processes are destroyed.

{\bf Input files:}
headers.h

@return Error code.

@param send_boxes [IN]
  description of the grid data to be sent to other processors.
@param recv_boxes [IN]
  description of the grid data to be received from other processors.
@param send_stride [IN]
  stride to use for send data.
@param recv_stride [IN]
  stride to use for receive data.
@param send_processes [IN]
  processors that data is to be sent to.
@param recv_processes [IN]
  processors that data is to be received from.
@param indt_boxes_ptr [IN]
  description of computations that do not depend on communicated data.
@param dept_boxes_ptr [IN]
  description of computations that depend on communicated data.
@param stride [IN]
  stride to use for computations.
@param grid [IN]
  computational grid
@param data_space [IN]
  description of the stored data associated with the grid.
@param num_values [IN]
  number of data values associated with each grid index.
@param compute_pkg_ptr [OUT]
  pointer to a computation package

@see hypre_CommPkgCreate, hypre_ComputePkgDestroy */
/*--------------------------------------------------------------------------*/

  int
  hypre_ComputePkgCreate( hypre_BoxArrayArray   *send_boxes,
                          hypre_BoxArrayArray   *recv_boxes,
                          hypre_Index            send_stride,
                          hypre_Index            recv_stride,
                          int                  **send_processes,
                          int                  **recv_processes,
                          hypre_BoxArrayArray   *indt_boxes,
                          hypre_BoxArrayArray   *dept_boxes,
                          hypre_Index            stride,
                          hypre_StructGrid      *grid,
                          hypre_BoxArray        *data_space,
                          int                    num_values,
                          hypre_ComputePkg     **compute_pkg_ptr )
{
   int                ierr = 0;
   hypre_ComputePkg  *compute_pkg;

   compute_pkg = hypre_CTAlloc(hypre_ComputePkg, 1);

   hypre_ComputePkgCommPkg(compute_pkg)     =
      hypre_CommPkgCreate(send_boxes, recv_boxes,
                          send_stride, recv_stride,
                          data_space, data_space,
                          send_processes, recv_processes,
                          num_values, hypre_StructGridComm(grid),
                          hypre_StructGridPeriodic(grid));

   hypre_ComputePkgIndtBoxes(compute_pkg)   = indt_boxes;
   hypre_ComputePkgDeptBoxes(compute_pkg)   = dept_boxes;
   hypre_CopyIndex(stride, hypre_ComputePkgStride(compute_pkg));

   hypre_StructGridRef(grid, &hypre_ComputePkgGrid(compute_pkg));
   hypre_ComputePkgDataSpace(compute_pkg)   = data_space;
   hypre_ComputePkgNumValues(compute_pkg)   = num_values;

   *compute_pkg_ptr = compute_pkg;

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroy a computation package.

{\bf Input files:}
headers.h

@return Error code.

@param compute_pkg [IN/OUT]
  computation package.

@see hypre_ComputePkgCreate */
/*--------------------------------------------------------------------------*/

int
hypre_ComputePkgDestroy( hypre_ComputePkg *compute_pkg )
{
   int ierr = 0;

   if (compute_pkg)
   {
      hypre_CommPkgDestroy(hypre_ComputePkgCommPkg(compute_pkg));

      hypre_BoxArrayArrayDestroy(hypre_ComputePkgIndtBoxes(compute_pkg));
      hypre_BoxArrayArrayDestroy(hypre_ComputePkgDeptBoxes(compute_pkg));

      hypre_StructGridDestroy(hypre_ComputePkgGrid(compute_pkg));

      hypre_TFree(compute_pkg);
   }

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Initialize a non-blocking communication exchange.  The independent
computations may be done after a call to this routine, to allow for
overlap of communications and computations.

{\bf Input files:}
headers.h

@return Error code.

@param compute_pkg [IN]
  computation package.
@param data [IN]
  pointer to the associated data.
@param comm_handle [OUT]
  communication handle.

@see hypre_FinalizeIndtComputations, hypre_ComputePkgCreate,
hypre_InitializeCommunication */
/*--------------------------------------------------------------------------*/

int
hypre_InitializeIndtComputations( hypre_ComputePkg  *compute_pkg,
                                  double            *data,
                                  hypre_CommHandle **comm_handle_ptr )
{
   int            ierr = 0;
   hypre_CommPkg *comm_pkg = hypre_ComputePkgCommPkg(compute_pkg);

   ierr = hypre_InitializeCommunication(comm_pkg, data, data, comm_handle_ptr);

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Finalize a communication exchange.  The dependent computations may
be done after a call to this routine.

{\bf Input files:}
headers.h

@return Error code.

@param comm_handle [IN/OUT]
  communication handle.

@see hypre_InitializeIndtComputations, hypre_FinalizeCommunication */
/*--------------------------------------------------------------------------*/

int
hypre_FinalizeIndtComputations( hypre_CommHandle *comm_handle )
{
   int ierr = 0;

   ierr = hypre_FinalizeCommunication(comm_handle);

   return ierr;
}
