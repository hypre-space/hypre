/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Umpire memory management utilities
 *
 *****************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_UMPIRE_HOST)

/*--------------------------------------------------------------------------
 * hypre_umpire_host_pooled_allocate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_umpire_host_pooled_allocate(void **ptr, size_t nbytes)
{
   hypre_Handle *handle = hypre_handle();
   const char *resource_name = "HOST";
   const char *pool_name = hypre_HandleUmpireHostPoolName(handle);

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       hypre_HandleUmpireHostPoolSize(handle),
                                                       hypre_HandleUmpireBlockSize(handle), &pooled_allocator);
      hypre_HandleOwnUmpireHostPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_umpire_host_pooled_free
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_umpire_host_pooled_free(void *ptr)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpireHostPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_umpire_host_pooled_realloc
 *--------------------------------------------------------------------------*/

void *
hypre_umpire_host_pooled_realloc(void *ptr, size_t size)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpireHostPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   ptr = umpire_resourcemanager_reallocate_with_allocator(rm_ptr, ptr, size, pooled_allocator);

   return ptr;
}
#endif

#if defined(HYPRE_USING_UMPIRE_DEVICE)

/*--------------------------------------------------------------------------
 * hypre_umpire_device_pooled_allocate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_umpire_device_pooled_allocate(void **ptr, size_t nbytes)
{
   hypre_Handle *handle = hypre_handle();
   hypre_int device_id;
   char resource_name[16];
   const char *pool_name = hypre_HandleUmpireDevicePoolName(handle);

   hypre_GetDevice(&device_id);
   hypre_sprintf(resource_name, "%s::%d", "DEVICE", device_id);

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       hypre_HandleUmpireDevicePoolSize(handle),
                                                       hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      hypre_HandleOwnUmpireDevicePool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_umpire_device_pooled_free
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_umpire_device_pooled_free(void *ptr)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpireDevicePoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}
#endif

#if defined(HYPRE_USING_UMPIRE_UM)

/*--------------------------------------------------------------------------
 * hypre_umpire_um_pooled_allocate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_umpire_um_pooled_allocate(void **ptr, size_t nbytes)
{
   hypre_Handle *handle = hypre_handle();
   const char *resource_name = "UM";
   const char *pool_name = hypre_HandleUmpireUMPoolName(handle);

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       hypre_HandleUmpireUMPoolSize(handle),
                                                       hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      hypre_HandleOwnUmpireUMPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_umpire_um_pooled_free
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_umpire_um_pooled_free(void *ptr)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpireUMPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}
#endif

#if defined(HYPRE_USING_UMPIRE_PINNED)

/*--------------------------------------------------------------------------
 * hypre_umpire_pinned_pooled_allocate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_umpire_pinned_pooled_allocate(void **ptr, size_t nbytes)
{
   hypre_Handle *handle = hypre_handle();
   const char *resource_name = "PINNED";
   const char *pool_name = hypre_HandleUmpirePinnedPoolName(handle);

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       hypre_HandleUmpirePinnedPoolSize(handle),
                                                       hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      hypre_HandleOwnUmpirePinnedPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_umpire_pinned_pooled_free
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_umpire_pinned_pooled_free(void *ptr)
{
   hypre_Handle *handle = hypre_handle();
   const char *pool_name = hypre_HandleUmpirePinnedPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(handle);

   hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return hypre_error_flag;
}
#endif

/******************************************************************************
 *
 * hypre Umpire
 *
 *****************************************************************************/

#if defined(HYPRE_USING_UMPIRE)

/*--------------------------------------------------------------------------
 * hypre_UmpireInit
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UmpireInit(hypre_Handle *hypre_handle_)
{
   umpire_resourcemanager_get_instance(&hypre_HandleUmpireResourceMan(hypre_handle_));

   hypre_HandleUmpireDevicePoolSize(hypre_handle_) = 4LL * (1 << 30); // 4 GiB
   hypre_HandleUmpireUMPoolSize(hypre_handle_)     = 4LL * (1 << 30); // 4 GiB
   hypre_HandleUmpireHostPoolSize(hypre_handle_)   = 4LL * (1 << 30); // 4 GiB
   hypre_HandleUmpirePinnedPoolSize(hypre_handle_) = 4LL * (1 << 30); // 4 GiB

   hypre_HandleUmpireBlockSize(hypre_handle_) = 512;

   strcpy(hypre_HandleUmpireDevicePoolName(hypre_handle_), "HYPRE_DEVICE_POOL");
   strcpy(hypre_HandleUmpireUMPoolName(hypre_handle_),     "HYPRE_UM_POOL");
   strcpy(hypre_HandleUmpireHostPoolName(hypre_handle_),   "HYPRE_HOST_POOL");
   strcpy(hypre_HandleUmpirePinnedPoolName(hypre_handle_), "HYPRE_PINNED_POOL");

   hypre_HandleOwnUmpireDevicePool(hypre_handle_) = 0;
   hypre_HandleOwnUmpireUMPool(hypre_handle_)     = 0;
   hypre_HandleOwnUmpireHostPool(hypre_handle_)   = 0;
   hypre_HandleOwnUmpirePinnedPool(hypre_handle_) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UmpireFinalize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UmpireFinalize(hypre_Handle *hypre_handle_)
{
   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(hypre_handle_);
   umpire_allocator allocator;

#if defined(HYPRE_USING_UMPIRE_HOST)
   if (hypre_HandleOwnUmpireHostPool(hypre_handle_))
   {
      const char *pool_name = hypre_HandleUmpireHostPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_DEVICE)
   if (hypre_HandleOwnUmpireDevicePool(hypre_handle_))
   {
      const char *pool_name = hypre_HandleUmpireDevicePoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_UM)
   if (hypre_HandleOwnUmpireUMPool(hypre_handle_))
   {
      const char *pool_name = hypre_HandleUmpireUMPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_PINNED)
   if (hypre_HandleOwnUmpirePinnedPool(hypre_handle_))
   {
      const char *pool_name = hypre_HandleUmpirePinnedPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UmpireMemoryGetUsage
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UmpireMemoryGetUsage(HYPRE_Real *memory)
{
   hypre_Handle                 *handle = hypre_handle();
   umpire_resourcemanager       *rm_ptr = &hypre_HandleUmpireResourceMan(handle);
   umpire_allocator              allocator;

   size_t                        memoryB[8] = {0, 0, 0, 0, 0, 0, 0, 0};
   HYPRE_Int                     i;

   if (!memory)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "memory is a NULL pointer!");
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_UMPIRE_HOST)
   if (hypre_HandleOwnUmpireHostPool(handle))
   {
      const char *pool_name = hypre_HandleUmpireHostPoolName(handle);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      memoryB[0] = umpire_allocator_get_current_size(&allocator);
      memoryB[1] = umpire_allocator_get_high_watermark(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_DEVICE)
   if (hypre_HandleOwnUmpireDevicePool(handle))
   {
      const char *pool_name = hypre_HandleUmpireDevicePoolName(handle);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      memoryB[2] = umpire_allocator_get_current_size(&allocator);
      memoryB[3] = umpire_allocator_get_high_watermark(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_UM)
   if (hypre_HandleOwnUmpireUMPool(handle))
   {
      const char *pool_name = hypre_HandleUmpireUMPoolName(handle);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      memoryB[4] = umpire_allocator_get_current_size(&allocator);
      memoryB[5] = umpire_allocator_get_high_watermark(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_PINNED)
   if (hypre_HandleOwnUmpirePinnedPool(handle))
   {
      const char *pool_name = hypre_HandleUmpirePinnedPoolName(handle);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      memoryB[6] = umpire_allocator_get_current_size(&allocator);
      memoryB[7] = umpire_allocator_get_high_watermark(&allocator);
   }
#endif

   for (i = 0; i < 8; i++)
   {
      memory[i] = ((HYPRE_Real) memoryB[i]) / ((HYPRE_Real) (1 << 30));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUmpireDevicePoolSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetUmpireDevicePoolSize(size_t nbytes)
{
   hypre_HandleUmpireDevicePoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUmpireUMPoolSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetUmpireUMPoolSize(size_t nbytes)
{
   hypre_HandleUmpireUMPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUmpireHostPoolSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetUmpireHostPoolSize(size_t nbytes)
{
   hypre_HandleUmpireHostPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUmpirePinnedPoolSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetUmpirePinnedPoolSize(size_t nbytes)
{
   hypre_HandleUmpirePinnedPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUmpireDevicePoolName
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetUmpireDevicePoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireDevicePoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUmpireUMPoolName
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetUmpireUMPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireUMPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUmpireHostPoolName
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetUmpireHostPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireHostPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetUmpirePinnedPoolName
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetUmpirePinnedPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpirePinnedPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_UMPIRE) */
