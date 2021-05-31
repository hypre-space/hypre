/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_UMPIRE_ALLOCATOR_H
#define HYPRE_UMPIRE_ALLOCATOR_H

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#if defined(HYPRE_USING_UMPIRE_DEVICE)

/*
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"
*/

struct hypre_umpire_device_allocator
{
   typedef char value_type;

   hypre_umpire_device_allocator()
   {
      // constructor
   }

   ~hypre_umpire_device_allocator()
   {
      // destructor
   }

   char *allocate(std::ptrdiff_t num_bytes)
   {
      char *ptr = NULL;
      hypre_umpire_device_pooled_allocate((void**) &ptr, num_bytes);

      return ptr;
   }

   void deallocate(char *ptr, size_t n)
   {
      hypre_umpire_device_pooled_free(ptr);
   }
};

#endif /* #ifdef HYPRE_USING_UMPIRE_DEVICE */
#endif /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */

#endif
