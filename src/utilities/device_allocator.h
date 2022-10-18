/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef DEVICE_ALLOCATOR_H
#define DEVICE_ALLOCATOR_H

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/* C++ style memory allocator for the device using the abstract memory model */
struct hypre_device_allocator
{
   typedef char value_type;

   hypre_device_allocator()
   {
      // constructor
   }

   ~hypre_device_allocator()
   {
      // destructor
   }

   char *allocate(std::ptrdiff_t num_bytes)
   {
      return hypre_TAlloc(char, num_bytes, HYPRE_MEMORY_DEVICE);
   }

   void deallocate(char *ptr, size_t n)
   {
      hypre_TFree(ptr, HYPRE_MEMORY_DEVICE);
   }
};

#endif /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */

#endif
