/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple devices.
 ******************************************************************************/

#ifndef HYPRE_CUB_ALLOCATOR_HEADER
#define HYPRE_CUB_ALLOCATOR_HEADER

#if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_DEVICE_POOL)

#include <set>
#include <map>

#if (__cplusplus > 199711L) || (defined(_MSC_VER) && _MSC_VER >= 1800)
    #include <mutex>
#else
    #if defined(_WIN32) || defined(_WIN64)
        #include <intrin.h>

        #define WIN32_LEAN_AND_MEAN
        #define NOMINMAX
        #include <windows.h>
        #undef WIN32_LEAN_AND_MEAN
        #undef NOMINMAX

        /**
         * Compiler read/write barrier
         */
        #pragma intrinsic(_ReadWriteBarrier)

    #endif
#endif

/**
 * Simple portable mutex
 *   - Wraps std::mutex when compiled with C++11 or newer (supported on all platforms)
 *   - Uses GNU/Windows spinlock mechanisms for pre C++11 (supported on x86/x64 when compiled with cl.exe or g++)
 */
struct hypre_cub_Mutex
{
#if (__cplusplus > 199711L) || (defined(_MSC_VER) && _MSC_VER >= 1800)

    std::mutex mtx;

    void Lock()
    {
        mtx.lock();
    }

    void Unlock()
    {
        mtx.unlock();
    }

    void TryLock()
    {
        mtx.try_lock();
    }

#else       //__cplusplus > 199711L

    #if defined(_MSC_VER)

        // Microsoft VC++
        typedef hypre_longint Spinlock;

    #else

        // GNU g++
        typedef hypre_int Spinlock;

        /**
         * Compiler read/write barrier
         */
        __forceinline__ void _ReadWriteBarrier()
        {
            __sync_synchronize();
        }

        /**
         * Atomic exchange
         */
        __forceinline__ hypre_longint _InterlockedExchange(volatile hypre_int * const Target, const hypre_int Value)
        {
            // NOTE: __sync_lock_test_and_set would be an acquire barrier, so we force a full barrier
            _ReadWriteBarrier();
            return __sync_lock_test_and_set(Target, Value);
        }

        /**
         * Pause instruction to prevent excess processor bus usage
         */
        __forceinline__ void YieldProcessor()
        {
        }

    #endif  // defined(_MSC_VER)

        /// Lock member
        volatile Spinlock lock;

        /**
         * Constructor
         */
        hypre_cub_Mutex() : lock(0) {}

        /**
         * Return when the specified spinlock has been acquired
         */
        __forceinline__ void Lock()
        {
            while (1)
            {
                if (!_InterlockedExchange(&lock, 1)) return;
                while (lock) YieldProcessor();
            }
        }


        /**
         * Release the specified spinlock
         */
        __forceinline__ void Unlock()
        {
            _ReadWriteBarrier();
            lock = 0;
        }

#endif      // __cplusplus > 199711L

};

#include <math.h>

/******************************************************************************
 * CachingDeviceAllocator (host use)
 ******************************************************************************/

/**
 * \brief A simple caching allocator for device memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe and stream-safe and is capable of managing cached
 * device allocations on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations from the allocator are associated with an \p active_stream.  Once freed,
 *   the allocation becomes available immediately for reuse within the \p active_stream
 *   with which it was associated with during allocation, and it becomes available for
 *   reuse within other streams when all prior work submitted to \p active_stream has completed.
 * - Allocations are categorized and cached by bin size.  A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused device allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations on a given device will exceed
 *   \p max_cached_bytes, allocations for that device are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingDeviceAllocator is configured with:
 * - \p bin_growth          = 8
 * - \p min_bin             = 3
 * - \p max_bin             = 7
 * - \p max_cached_bytes    = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per device
 *
 */
struct hypre_cub_CachingDeviceAllocator
{
   typedef char value_type;

   //---------------------------------------------------------------------
   // Constants
   //---------------------------------------------------------------------

   /// Out-of-bounds bin
   static const hypre_uint INVALID_BIN = (hypre_uint) -1;

   /// Invalid size
   static const size_t INVALID_SIZE = (size_t) -1;

   /// Invalid device ordinal
   static const hypre_int INVALID_DEVICE_ORDINAL = -1;

   //---------------------------------------------------------------------
   // Type definitions and helper types
   //---------------------------------------------------------------------

   /**
    * Descriptor for device memory allocations
    */
   struct BlockDescriptor
   {
      void*           d_ptr;              // Device pointer
      size_t          bytes;              // Size of allocation in bytes
      hypre_uint      bin;                // Bin enumeration
      hypre_int       device;             // device ordinal
      cudaStream_t    associated_stream;  // Associated associated_stream
      cudaEvent_t     ready_event;        // Signal when associated stream has run to the point at which this block was freed

      // Constructor (suitable for searching maps for a specific block, given its pointer and device)
      BlockDescriptor(void *d_ptr, hypre_int device) :
         d_ptr(d_ptr),
         bytes(0),
         bin(INVALID_BIN),
         device(device),
         associated_stream(0),
         ready_event(0)
      {}

      // Constructor (suitable for searching maps for a range of suitable blocks, given a device)
      BlockDescriptor(hypre_int device) :
         d_ptr(NULL),
         bytes(0),
         bin(INVALID_BIN),
         device(device),
         associated_stream(0),
         ready_event(0)
      {}

      // Comparison functor for comparing device pointers
      static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
      {
         if (a.device == b.device)
            return (a.d_ptr < b.d_ptr);
         else
            return (a.device < b.device);
      }

      // Comparison functor for comparing allocation sizes
      static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
      {
         if (a.device == b.device)
            return (a.bytes < b.bytes);
         else
            return (a.device < b.device);
      }
   };

   /// BlockDescriptor comparator function interface
   typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

   class TotalBytes {
      public:
         size_t free;
         size_t live;
         TotalBytes() { free = live = 0; }
   };

   /// Set type for cached blocks (ordered by size)
   typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

   /// Set type for live blocks (ordered by ptr)
   typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

   /// Map type of device ordinals to the number of cached bytes cached by each device
   typedef std::map<hypre_int, TotalBytes> GpuCachedBytes;


   //---------------------------------------------------------------------
   // Utility functions
   //---------------------------------------------------------------------

   /**
    * Integer pow function for unsigned base and exponent
    */
   static hypre_uint IntPow(
         hypre_uint base,
         hypre_uint exp)
   {
      hypre_uint retval = 1;
      while (exp > 0)
      {
         if (exp & 1) {
            retval = retval * base;        // multiply the result by the current base
         }
         base = base * base;                // square the base
         exp = exp >> 1;                    // divide the exponent in half
      }
      return retval;
   }


   /**
    * Round up to the nearest power-of
    */
   void NearestPowerOf(
         hypre_uint      &power,
         size_t          &rounded_bytes,
         hypre_uint       base,
         size_t           value)
   {
      power = 0;
      rounded_bytes = 1;

      if (value * base < value)
      {
         // Overflow
         power = sizeof(size_t) * 8;
         rounded_bytes = size_t(0) - 1;
         return;
      }

      while (rounded_bytes < value)
      {
         rounded_bytes *= base;
         power++;
      }
   }


   //---------------------------------------------------------------------
   // Fields
   //---------------------------------------------------------------------

   hypre_cub_Mutex mutex;              /// Mutex for thread-safety

   hypre_uint      bin_growth;         /// Geometric growth factor for bin-sizes
   hypre_uint      min_bin;            /// Minimum bin enumeration
   hypre_uint      max_bin;            /// Maximum bin enumeration

   size_t          min_bin_bytes;      /// Minimum bin size
   size_t          max_bin_bytes;      /// Maximum bin size
   size_t          max_cached_bytes;   /// Maximum aggregate cached bytes per device

   const bool      skip_cleanup;       /// Whether or not to skip a call to FreeAllCached() when destructor is called.  (The CUDA runtime may have already shut down for statically declared allocators)
   bool            debug;              /// Whether or not to print (de)allocation events to stdout

   GpuCachedBytes  cached_bytes;       /// Map of device ordinal to aggregate cached bytes on that device
   CachedBlocks    cached_blocks;      /// Set of cached device allocations available for reuse
   BusyBlocks      live_blocks;        /// Set of live device allocations currently in use

   bool            use_managed_memory; /// Whether to use managed memory or device memory

   //---------------------------------------------------------------------
   // Methods
   //---------------------------------------------------------------------

   /**
    * \brief Constructor.
    */
   hypre_cub_CachingDeviceAllocator(
         hypre_uint      bin_growth,                             ///< Geometric growth factor for bin-sizes
         hypre_uint      min_bin             = 1,                ///< Minimum bin (default is bin_growth ^ 1)
         hypre_uint      max_bin             = INVALID_BIN,      ///< Maximum bin (default is no max bin)
         size_t          max_cached_bytes    = INVALID_SIZE,     ///< Maximum aggregate cached bytes per device (default is no limit)
         bool            skip_cleanup        = false,            ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called (default is to deallocate)
         bool            debug               = false,            ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)
         bool            use_managed_memory  = false)            ///< Whether to use managed memory or device memory
      :
         bin_growth(bin_growth),
         min_bin(min_bin),
         max_bin(max_bin),
         min_bin_bytes(IntPow(bin_growth, min_bin)),
         max_bin_bytes(IntPow(bin_growth, max_bin)),
         max_cached_bytes(max_cached_bytes),
         skip_cleanup(skip_cleanup),
         debug(debug),
         use_managed_memory(use_managed_memory),
         cached_blocks(BlockDescriptor::SizeCompare),
         live_blocks(BlockDescriptor::PtrCompare)
   {}


   /**
    * \brief Default constructor.
    *
    * Configured with:
    * \par
    * - \p bin_growth          = 8
    * - \p min_bin             = 3
    * - \p max_bin             = 7
    * - \p max_cached_bytes    = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
    *
    * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
    * sets a maximum of 6,291,455 cached bytes per device
    */
   hypre_cub_CachingDeviceAllocator(
         bool skip_cleanup = false,
         bool debug = false,
         bool use_managed_memory = false)
      :
         bin_growth(8),
         min_bin(3),
         max_bin(7),
         min_bin_bytes(IntPow(bin_growth, min_bin)),
         max_bin_bytes(IntPow(bin_growth, max_bin)),
         max_cached_bytes((max_bin_bytes * 3) - 1),
         skip_cleanup(skip_cleanup),
         debug(debug),
         use_managed_memory(use_managed_memory),
         cached_blocks(BlockDescriptor::SizeCompare),
         live_blocks(BlockDescriptor::PtrCompare)
   {}


   /**
    * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
    *
    * Changing the ceiling of cached bytes does not cause any allocations (in-use or
    * cached-in-reserve) to be freed.  See \p FreeAllCached().
    */
   cudaError_t SetMaxCachedBytes(
         size_t max_cached_bytes)
   {
      // Lock
      mutex.Lock();

      if (debug) printf("Changing max_cached_bytes (%zu -> %zu)\n", this->max_cached_bytes, max_cached_bytes);

      this->max_cached_bytes = max_cached_bytes;

      // Unlock
      mutex.Unlock();

      return cudaSuccess;
   }


   /**
    * \brief Provides a suitable allocation of device memory for the given size on the specified device.
    *
    * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
    * with which it was associated with during allocation, and it becomes available for reuse within other
    * streams when all prior work submitted to \p active_stream has completed.
    */
   cudaError_t DeviceAllocate(
         hypre_int       device,             ///< [in] Device on which to place the allocation
         void            **d_ptr,            ///< [out] Reference to pointer to the allocation
         size_t          bytes,              ///< [in] Minimum number of bytes for the allocation
         cudaStream_t    active_stream = 0)  ///< [in] The stream to be associated with this allocation
   {
      *d_ptr                          = NULL;
      hypre_int entrypoint_device     = INVALID_DEVICE_ORDINAL;
      cudaError_t error               = cudaSuccess;

      if (device == INVALID_DEVICE_ORDINAL)
      {
         if ((error = cudaGetDevice(&entrypoint_device))) return error;
         device = entrypoint_device;
      }

      // Create a block descriptor for the requested allocation
      bool found = false;
      BlockDescriptor search_key(device);
      search_key.associated_stream = active_stream;
      NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

      if (search_key.bin > max_bin)
      {
         // Bin is greater than our maximum bin: allocate the request
         // exactly and give out-of-bounds bin.  It will not be cached
         // for reuse when returned.
         search_key.bin      = INVALID_BIN;
         search_key.bytes    = bytes;
      }
      else
      {
         // Search for a suitable cached allocation: lock
         mutex.Lock();

         if (search_key.bin < min_bin)
         {
            // Bin is less than minimum bin: round up
            search_key.bin      = min_bin;
            search_key.bytes    = min_bin_bytes;
         }

         // Iterate through the range of cached blocks on the same device in the same bin
         CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
         while ((block_itr != cached_blocks.end())
               && (block_itr->device == device)
               && (block_itr->bin == search_key.bin))
         {
            // To prevent races with reusing blocks returned by the host but still
            // in use by the device, only consider cached blocks that are
            // either (from the active stream) or (from an idle stream)
            if ((active_stream == block_itr->associated_stream) ||
                  (cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady))
            {
               // Reuse existing cache block.  Insert into live blocks.
               found = true;
               search_key = *block_itr;
               search_key.associated_stream = active_stream;
               live_blocks.insert(search_key);

               // Remove from free blocks
               cached_bytes[device].free -= search_key.bytes;
               cached_bytes[device].live += search_key.bytes;

               if (debug) printf("\tDevice %d reused cached block at %p (%zu bytes) for stream %p (previously associated with stream %p).\n",
                     device, search_key.d_ptr, search_key.bytes, search_key.associated_stream, block_itr->associated_stream);

               cached_blocks.erase(block_itr);

               break;
            }
            block_itr++;
         }

         // Done searching: unlock
         mutex.Unlock();
      }

      // Allocate the block if necessary
      if (!found)
      {
         // Set runtime's current device to specified device (entrypoint may not be set)
         if (device != entrypoint_device)
         {
            if ((error = cudaGetDevice(&entrypoint_device))) return error;
            if ((error = cudaSetDevice(device))) return error;
         }

         // Attempt to allocate

         if (use_managed_memory)
         {
            error = cudaMallocManaged(&search_key.d_ptr, search_key.bytes);
         }
         else
         {
            error = cudaMalloc(&search_key.d_ptr, search_key.bytes);
         }
         if ((error) == cudaErrorMemoryAllocation)
         {
            // The allocation attempt failed: free all cached blocks on device and retry
            if (debug) printf("\tDevice %d failed to allocate %zu bytes for stream %p, retrying after freeing cached allocations",
                  device, search_key.bytes, search_key.associated_stream);

            error = cudaSuccess;    // Reset the error we will return
            cudaGetLastError();     // Reset CUDART's error

            // Lock
            mutex.Lock();

            // Iterate the range of free blocks on the same device
            BlockDescriptor free_key(device);
            CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

            while ((block_itr != cached_blocks.end()) && (block_itr->device == device))
            {
               // No need to worry about synchronization with the device: cudaFree is
               // blocking and will synchronize across all kernels executing
               // on the current device

               // Free device memory and destroy stream event.
               if ((error = cudaFree(block_itr->d_ptr))) break;
               if ((error = cudaEventDestroy(block_itr->ready_event))) break;

               // Reduce balance and erase entry
               cached_bytes[device].free -= block_itr->bytes;

               if (debug) printf("\tDevice %d freed %zu bytes.\n\t\t  %zu available blocks cached (%zu bytes), %zu live blocks (%zu bytes) outstanding.\n",
                     device, block_itr->bytes, cached_blocks.size(), cached_bytes[device].free, live_blocks.size(), cached_bytes[device].live);

               cached_blocks.erase(block_itr);

               block_itr++;
            }

            // Unlock
            mutex.Unlock();

            // Return under error
            if (error) return error;

            // Try to allocate again

            if (use_managed_memory)
            {
               error = cudaMallocManaged(&search_key.d_ptr, search_key.bytes);
            }
            else
            {
               error = cudaMalloc(&search_key.d_ptr, search_key.bytes);
            }
            if ((error)) return error;
         }

         // Create ready event
         if ((error = cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming)))
            return error;

         // Insert into live blocks
         mutex.Lock();
         live_blocks.insert(search_key);
         cached_bytes[device].live += search_key.bytes;
         mutex.Unlock();

         if (debug) printf("\tDevice %d allocated new device block at %p (%zu bytes associated with stream %p).\n",
               device, search_key.d_ptr, search_key.bytes, search_key.associated_stream);

         // Attempt to revert back to previous device if necessary
         if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
         {
            if ((error = cudaSetDevice(entrypoint_device))) return error;
         }
      }

      // Copy device pointer to output parameter
      *d_ptr = search_key.d_ptr;

      if (debug) printf("\t\t%zu available blocks cached (%zu bytes), %zu live blocks outstanding(%zu bytes).\n",
            cached_blocks.size(), cached_bytes[device].free, live_blocks.size(), cached_bytes[device].live);

      return error;
   }


   /**
    * \brief Provides a suitable allocation of device memory for the given size on the current device.
    *
    * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
    * with which it was associated with during allocation, and it becomes available for reuse within other
    * streams when all prior work submitted to \p active_stream has completed.
    */
   cudaError_t DeviceAllocate(
         void            **d_ptr,            ///< [out] Reference to pointer to the allocation
         size_t          bytes,              ///< [in] Minimum number of bytes for the allocation
         cudaStream_t    active_stream = 0)  ///< [in] The stream to be associated with this allocation
   {
      return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
   }

   char * allocate(size_t bytes)
   {
      char *ptr;
      DeviceAllocate((void **) &ptr, bytes);

      return ptr;
   }


   /**
    * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator.
    *
    * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
    * with which it was associated with during allocation, and it becomes available for reuse within other
    * streams when all prior work submitted to \p active_stream has completed.
    */
   cudaError_t DeviceFree(
         hypre_int       device,
         void*           d_ptr)
   {
      hypre_int entrypoint_device     = INVALID_DEVICE_ORDINAL;
      cudaError_t error               = cudaSuccess;

      if (device == INVALID_DEVICE_ORDINAL)
      {
         if ((error = cudaGetDevice(&entrypoint_device)))
            return error;
         device = entrypoint_device;
      }

      // Lock
      mutex.Lock();

      // Find corresponding block descriptor
      bool recached = false;
      BlockDescriptor search_key(d_ptr, device);
      BusyBlocks::iterator block_itr = live_blocks.find(search_key);
      if (block_itr != live_blocks.end())
      {
         // Remove from live blocks
         search_key = *block_itr;
         live_blocks.erase(block_itr);
         cached_bytes[device].live -= search_key.bytes;

         // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
         if ((search_key.bin != INVALID_BIN) && (cached_bytes[device].free + search_key.bytes <= max_cached_bytes))
         {
            // Insert returned allocation into free blocks
            recached = true;
            cached_blocks.insert(search_key);
            cached_bytes[device].free += search_key.bytes;

            if (debug) printf("\tDevice %d returned %zu bytes from associated stream %p.\n\t\t %zu available blocks cached (%zu bytes), %zu live blocks outstanding. (%zu bytes)\n",
                  device, search_key.bytes, search_key.associated_stream, cached_blocks.size(),
                  cached_bytes[device].free, live_blocks.size(), cached_bytes[device].live);
         }
      }

      // Unlock
      mutex.Unlock();

      // First set to specified device (entrypoint may not be set)
      if (device != entrypoint_device)
      {
         if ((error = cudaGetDevice(&entrypoint_device))) return error;
         if ((error = cudaSetDevice(device))) return error;
      }

      if (recached)
      {
         // Insert the ready event in the associated stream (must have current device set properly)
         if ((error = cudaEventRecord(search_key.ready_event, search_key.associated_stream))) return error;
      }
      else
      {
         // Free the allocation from the runtime and cleanup the event.
         if ((error = cudaFree(d_ptr))) return error;
         if ((error = cudaEventDestroy(search_key.ready_event))) return error;

         if (debug) printf("\tDevice %d freed %zu bytes from associated stream %p.\n\t\t  %zu available blocks cached (%zu bytes), %zu live blocks (%zu bytes) outstanding.\n",
               device, search_key.bytes, search_key.associated_stream, cached_blocks.size(), cached_bytes[device].free, live_blocks.size(), cached_bytes[device].live);
      }

      // Reset device
      if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
      {
         if ((error = cudaSetDevice(entrypoint_device))) return error;
      }

      return error;
   }

   /**
    * \brief Frees a live allocation of device memory on the current device, returning it to the allocator.
    *
    * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
    * with which it was associated with during allocation, and it becomes available for reuse within other
    * streams when all prior work submitted to \p active_stream has completed.
    */
   cudaError_t DeviceFree(
         void*           d_ptr)
   {
      return DeviceFree(INVALID_DEVICE_ORDINAL, d_ptr);
   }

   void deallocate(char *ptr, size_t)
   {
      DeviceFree((void *) ptr);
   }

   /**
    * \brief Frees all cached device allocations on all devices
    */
   cudaError_t FreeAllCached()
   {
      cudaError_t error           = cudaSuccess;
      hypre_int entrypoint_device = INVALID_DEVICE_ORDINAL;
      hypre_int current_device    = INVALID_DEVICE_ORDINAL;

      mutex.Lock();

      while (!cached_blocks.empty())
      {
         // Get first block
         CachedBlocks::iterator begin = cached_blocks.begin();

         // Get entry-point device ordinal if necessary
         if (entrypoint_device == INVALID_DEVICE_ORDINAL)
         {
            if ((error = cudaGetDevice(&entrypoint_device))) break;
         }

         // Set current device ordinal if necessary
         if (begin->device != current_device)
         {
            if ((error = cudaSetDevice(begin->device))) break;
            current_device = begin->device;
         }

         // Free device memory
         if ((error = cudaFree(begin->d_ptr))) break;
         if ((error = cudaEventDestroy(begin->ready_event))) break;

         // Reduce balance and erase entry
         cached_bytes[current_device].free -= begin->bytes;

         if (debug) printf("\tDevice %d freed %zu bytes.\n\t\t  %zu available blocks cached (%zu bytes), %zu live blocks (%zu bytes) outstanding.\n",
               current_device, begin->bytes, cached_blocks.size(), cached_bytes[current_device].free, live_blocks.size(), cached_bytes[current_device].live);

         cached_blocks.erase(begin);
      }

      mutex.Unlock();

      // Attempt to revert back to entry-point device if necessary
      if (entrypoint_device != INVALID_DEVICE_ORDINAL)
      {
         if ((error = cudaSetDevice(entrypoint_device))) return error;
      }

      return error;
   }


   /**
    * \brief Destructor
    */
   virtual ~hypre_cub_CachingDeviceAllocator()
   {
      if (!skip_cleanup)
         FreeAllCached();
   }
};

#endif // #if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_DEVICE_POOL)
#endif // #ifndef HYPRE_CUB_ALLOCATOR_HEADER

