/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef HYPRE_USING_UMPIRE

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/util/Macros.hpp"
#include <string>

struct hypre_umpire_allocator
{
  typedef char value_type;

  hypre_umpire_allocator() {
  }

  ~hypre_umpire_allocator()
  {
    // auto hwm = umpire::ResourceManager::getInstance()
    //   .getAllocator("HYPRE_DEVICE_POOL")
    //   .getHighWatermark();
    // std::cout<<" High Water Mark is "<<hwm<<"bytes\n";
  }

  char *allocate(std::ptrdiff_t num_bytes)
  {
    //std::cout<<"umpire allocate "<<num_bytes<<"\n";
    umpire::ResourceManager &rma = umpire::ResourceManager::getInstance();
    auto allocator = rma.getAllocator("HYPRE_DEVICE_POOL");
    char *result = static_cast<char *>(allocator.allocate(num_bytes));
    return result;
  }

  void deallocate(char *ptr, size_t)
  {
    //std::cout<<"umpire de-allocate \n";
    umpire::ResourceManager &rma = umpire::ResourceManager::getInstance();
    auto allocator = rma.getAllocator("HYPRE_DEVICE_POOL");
    allocator.deallocate(ptr);
  }
};

#else /* #ifdef HYPRE_USING_UMPIRE */

// Dummy struct for when Umpire is not being used
struct hypre_umpire_allocator
{
  typedef char value_type;

  hypre_umpire_allocator() {
  }

  ~hypre_umpire_allocator(){}


  char *allocate(std::ptrdiff_t num_bytes)
  {
    return 0;
  }

  void deallocate(char *ptr, size_t)
  {
  }
};

#endif /* #ifdef HYPRE_USING_UMPIRE */
