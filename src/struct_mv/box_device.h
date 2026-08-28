/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_BOX_DEVICE_H
#define HYPRE_BOX_DEVICE_H

#if defined(HYPRE_USING_GPU)

/* WM: todo - Naming convention for device subroutines?
 *            Is this something we do elsewhere in the code? */
/* WM: todo - passing the Box struct by value does not seemt to */
/*            work as expected with the sycl backend on Aurora */
static __device__ __forceinline__
HYPRE_Int
hypre_IndexInBoxDevice( HYPRE_Int    *index,
                        hypre_Box     box )
{
   HYPRE_Int d, inbox, ndim = box.ndim;

   inbox = 1;
   for (d = 0; d < ndim; d++)
   {
      if (!(index[d] >= box.imin[d] && index[d] <= box.imax[d]))
      {
         inbox = 0;
         break;
      }
   }

   return inbox;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

static __device__ __forceinline__
HYPRE_Int
hypre_BoxIndexRankDevice( hypre_Box    box,
                          HYPRE_Int   *index )
{
   HYPRE_Int  rank, size, d, ndim = box.ndim;

   rank = 0;
   size = 1;
   for (d = 0; d < ndim; d++)
   {
      rank += (index[d] - box.imin[d]) * size;
      /* WM: todo - make hypre_BoxSizeDevice() subroutine? */
      /* size *= hypre_BoxSizeD(box, d); */
#if defined(HYPRE_USING_SYCL)
      size *= sycl::max(0, box.imax[d] - box.imin[d] + 1);
#else
      size *= max(0, box.imax[d] - box.imin[d] + 1);
#endif
   }

   return rank;
}

#endif

#endif
