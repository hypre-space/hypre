/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/**
 * Hopscotch hash is modified from the code downloaded from
 * https://sites.google.com/site/cconcurrencypackage/hopscotch-hashing
 * with the following terms of usage
 */

////////////////////////////////////////////////////////////////////////////////
//TERMS OF USAGE
//------------------------------------------------------------------------------
//
//  Permission to use, copy, modify and distribute this software and
//  its documentation for any purpose is hereby granted without fee,
//  provided that due acknowledgments to the authors are provided and
//  this permission notice appears in all copies of the software.
//  The software is provided "as is". There is no warranty of any kind.
//
//Authors:
//  Maurice Herlihy
//  Brown University
//  and
//  Nir Shavit
//  Tel-Aviv University
//  and
//  Moran Tzafrir
//  Tel-Aviv University
//
//  Date: July 15, 2008.  
//
////////////////////////////////////////////////////////////////////////////////
// Programmer : Moran Tzafrir (MoranTza@gmail.com)
// Modified   : Jongsoo Park  (jongsoo.park@intel.com)
//              Oct 1, 2015.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef hypre_HOPSCOTCH_HASH_HEADER
#define hypre_HOPSCOTCH_HASH_HEADER

//#include <strings.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include <math.h>

#ifdef HYPRE_USING_OPENMP
#include <omp.h>
#endif

#include "_hypre_utilities.h"

// Potentially architecture specific features used here:
// __sync_val_compare_and_swap

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * This next section of code is here instead of in _hypre_utilities.h to get
 * around some portability issues with Visual Studio.  By putting it here, we
 * can explicitly include this '.h' file in a few files in hypre and compile
 * them with C++ instead of C (VS does not support C99 'inline').
 ******************************************************************************/

#ifdef HYPRE_USING_ATOMIC
static inline HYPRE_Int hypre_compare_and_swap(HYPRE_Int *ptr, HYPRE_Int oldval, HYPRE_Int newval)
{
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
  return __sync_val_compare_and_swap(ptr, oldval, newval);
//#elif defind _MSC_VER
  //return _InterlockedCompareExchange((long *)ptr, newval, oldval);
//#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
// JSP: not many compilers have implemented this, so comment out for now
  //_Atomic HYPRE_Int *atomic_ptr = ptr;
  //atomic_compare_exchange_strong(atomic_ptr, &oldval, newval);
  //return oldval;
#endif
}

static inline HYPRE_Int hypre_fetch_and_add(HYPRE_Int *ptr, HYPRE_Int value)
{
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
  return __sync_fetch_and_add(ptr, value);
//#elif defined _MSC_VER
  //return _InterlockedExchangeAdd((long *)ptr, value);
//#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
// JSP: not many compilers have implemented this, so comment out for now
  //_Atomic HYPRE_Int *atomic_ptr = ptr;
  //return atomic_fetch_add(atomic_ptr, value);
#endif
}
#else // !HYPRE_USING_ATOMIC
static inline HYPRE_Int hypre_compare_and_swap(HYPRE_Int *ptr, HYPRE_Int oldval, HYPRE_Int newval)
{
   if (*ptr == oldval)
   {
      *ptr = newval;
      return oldval;
   }
   else return *ptr;
}

static inline HYPRE_Int hypre_fetch_and_add(HYPRE_Int *ptr, HYPRE_Int value)
{
   HYPRE_Int oldval = *ptr;
   *ptr += value;
   return oldval;
}
#endif // !HYPRE_USING_ATOMIC

/******************************************************************************/

// Constants ................................................................
#define HYPRE_HOPSCOTCH_HASH_HOP_RANGE    (32)
#define HYPRE_HOPSCOTCH_HASH_INSERT_RANGE (4*1024)

#define HYPRE_HOPSCOTCH_HASH_EMPTY (0)
#define HYPRE_HOPSCOTCH_HASH_BUSY  (1)

// Small Utilities ..........................................................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
static inline HYPRE_Int first_lsb_bit_indx(hypre_uint x) 
{
  return ffs(x) - 1;
}
#endif
/**
 * hypre_Hash is adapted from xxHash with the following license.
 */
/*
   xxHash - Extremely Fast Hash algorithm
   Header File
   Copyright (C) 2012-2015, Yann Collet.

   BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   You can contact the author at :
   - xxHash source repository : https://github.com/Cyan4973/xxHash
*/

/***************************************
*  Constants
***************************************/
#define HYPRE_XXH_PRIME32_1   2654435761U
#define HYPRE_XXH_PRIME32_2   2246822519U
#define HYPRE_XXH_PRIME32_3   3266489917U
#define HYPRE_XXH_PRIME32_4    668265263U
#define HYPRE_XXH_PRIME32_5    374761393U

#define HYPRE_XXH_PRIME64_1 11400714785074694791ULL
#define HYPRE_XXH_PRIME64_2 14029467366897019727ULL
#define HYPRE_XXH_PRIME64_3  1609587929392839161ULL
#define HYPRE_XXH_PRIME64_4  9650029242287828579ULL
#define HYPRE_XXH_PRIME64_5  2870177450012600261ULL

#  define HYPRE_XXH_rotl32(x,r) ((x << r) | (x >> (32 - r)))
#  define HYPRE_XXH_rotl64(x,r) ((x << r) | (x >> (64 - r)))

#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
static inline HYPRE_BigInt hypre_BigHash(HYPRE_BigInt input)
{
    hypre_ulongint h64 = HYPRE_XXH_PRIME64_5 + sizeof(input);

    hypre_ulongint k1 = input;
    k1 *= HYPRE_XXH_PRIME64_2;
    k1 = HYPRE_XXH_rotl64(k1, 31);
    k1 *= HYPRE_XXH_PRIME64_1;
    h64 ^= k1;
    h64 = HYPRE_XXH_rotl64(h64, 27)*HYPRE_XXH_PRIME64_1 + HYPRE_XXH_PRIME64_4;

    h64 ^= h64 >> 33;
    h64 *= HYPRE_XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= HYPRE_XXH_PRIME64_3;
    h64 ^= h64 >> 32;

#ifndef NDEBUG
    if (HYPRE_HOPSCOTCH_HASH_EMPTY == h64) {
      hypre_printf("hash(%lld) = %d\n", h64, HYPRE_HOPSCOTCH_HASH_EMPTY);
      assert(HYPRE_HOPSCOTCH_HASH_EMPTY != h64);
    }
#endif 

    return h64;
}

#else
static inline HYPRE_Int hypre_BigHash(HYPRE_Int input)
{
    hypre_uint h32 = HYPRE_XXH_PRIME32_5 + sizeof(input);

    // 1665863975 is added to input so that
    // only -1073741824 gives HYPRE_HOPSCOTCH_HASH_EMPTY.
    // Hence, we're fine as long as key is non-negative.
    h32 += (input + 1665863975)*HYPRE_XXH_PRIME32_3;
    h32 = HYPRE_XXH_rotl32(h32, 17)*HYPRE_XXH_PRIME32_4;

    h32 ^= h32 >> 15;
    h32 *= HYPRE_XXH_PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= HYPRE_XXH_PRIME32_3;
    h32 ^= h32 >> 16;

    //assert(HYPRE_HOPSCOTCH_HASH_EMPTY != h32);

    return h32;
}
#endif

#ifdef HYPRE_BIGINT
static inline HYPRE_Int hypre_Hash(HYPRE_Int input)
{
    hypre_ulongint h64 = HYPRE_XXH_PRIME64_5 + sizeof(input);

    hypre_ulongint k1 = input;
    k1 *= HYPRE_XXH_PRIME64_2;
    k1 = HYPRE_XXH_rotl64(k1, 31);
    k1 *= HYPRE_XXH_PRIME64_1;
    h64 ^= k1;
    h64 = HYPRE_XXH_rotl64(h64, 27)*HYPRE_XXH_PRIME64_1 + HYPRE_XXH_PRIME64_4;

    h64 ^= h64 >> 33;
    h64 *= HYPRE_XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= HYPRE_XXH_PRIME64_3;
    h64 ^= h64 >> 32;

#ifndef NDEBUG
    if (HYPRE_HOPSCOTCH_HASH_EMPTY == h64) {
      hypre_printf("hash(%lld) = %d\n", h64, HYPRE_HOPSCOTCH_HASH_EMPTY);
      assert(HYPRE_HOPSCOTCH_HASH_EMPTY != h64);
    }
#endif 

    return h64;
}

#else
static inline HYPRE_Int hypre_Hash(HYPRE_Int input)
{
    hypre_uint h32 = HYPRE_XXH_PRIME32_5 + sizeof(input);

    // 1665863975 is added to input so that
    // only -1073741824 gives HYPRE_HOPSCOTCH_HASH_EMPTY.
    // Hence, we're fine as long as key is non-negative.
    h32 += (input + 1665863975)*HYPRE_XXH_PRIME32_3;
    h32 = HYPRE_XXH_rotl32(h32, 17)*HYPRE_XXH_PRIME32_4;

    h32 ^= h32 >> 15;
    h32 *= HYPRE_XXH_PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= HYPRE_XXH_PRIME32_3;
    h32 ^= h32 >> 16;

    //assert(HYPRE_HOPSCOTCH_HASH_EMPTY != h32);

    return h32;
}
#endif

static inline void hypre_UnorderedIntSetFindCloserFreeBucket( hypre_UnorderedIntSet *s,
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
                            hypre_HopscotchSegment* start_seg,
#endif
                            HYPRE_Int *free_bucket,
                            HYPRE_Int *free_dist )
{
  HYPRE_Int move_bucket = *free_bucket - (HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
  HYPRE_Int move_free_dist;
  for (move_free_dist = HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
  {
    hypre_uint start_hop_info = s->hopInfo[move_bucket];
    HYPRE_Int move_new_free_dist = -1;
    hypre_uint mask = 1;
    HYPRE_Int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1)
    {
      if (mask & start_hop_info)
      {
        move_new_free_dist = i;
        break;
      }
    }
    if (-1 != move_new_free_dist)
    {
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      hypre_HopscotchSegment*  move_segment = &(s->segments[move_bucket & s->segmentMask]);
      
      if(start_seg != move_segment)
        omp_set_lock(&move_segment->lock);
#endif

      if (start_hop_info == s->hopInfo[move_bucket])
      {
        // new_free_bucket -> free_bucket and empty new_free_bucket
        HYPRE_Int new_free_bucket = move_bucket + move_new_free_dist;
        s->key[*free_bucket]  = s->key[new_free_bucket];
        s->hash[*free_bucket] = s->hash[new_free_bucket];

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        ++move_segment->timestamp;
#pragma omp flush
#endif

        s->hopInfo[move_bucket] |= (1U << move_free_dist);
        s->hopInfo[move_bucket] &= ~(1U << move_new_free_dist);

        *free_bucket = new_free_bucket;
        *free_dist -= move_free_dist - move_new_free_dist;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
#endif

        return;
      }
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
#endif
    }
    ++move_bucket;
  }
  *free_bucket = -1; 
  *free_dist = 0;
}

static inline void hypre_UnorderedBigIntSetFindCloserFreeBucket( hypre_UnorderedBigIntSet *s,
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
                             hypre_HopscotchSegment* start_seg,
#endif
                             HYPRE_Int *free_bucket,
                             HYPRE_Int *free_dist )
{
  HYPRE_Int move_bucket = *free_bucket - (HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
  HYPRE_Int move_free_dist;
  for (move_free_dist = HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
  {
    hypre_uint start_hop_info = s->hopInfo[move_bucket];
    HYPRE_Int move_new_free_dist = -1;
    hypre_uint mask = 1;
    HYPRE_Int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1)
    {
      if (mask & start_hop_info)
      {
        move_new_free_dist = i;
        break;
      }
    }
    if (-1 != move_new_free_dist)
    {
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      hypre_HopscotchSegment*  move_segment = &(s->segments[move_bucket & s->segmentMask]);
      
      if(start_seg != move_segment)
        omp_set_lock(&move_segment->lock);
#endif

      if (start_hop_info == s->hopInfo[move_bucket])
      {
        // new_free_bucket -> free_bucket and empty new_free_bucket
        HYPRE_Int new_free_bucket = move_bucket + move_new_free_dist;
        s->key[*free_bucket]  = s->key[new_free_bucket];
        s->hash[*free_bucket] = s->hash[new_free_bucket];

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        ++move_segment->timestamp;
#pragma omp flush
#endif

        s->hopInfo[move_bucket] |= (1U << move_free_dist);
        s->hopInfo[move_bucket] &= ~(1U << move_new_free_dist);

        *free_bucket = new_free_bucket;
        *free_dist -= move_free_dist - move_new_free_dist;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
#endif

        return;
      }
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
#endif
    }
    ++move_bucket;
  }
  *free_bucket = -1; 
  *free_dist = 0;
}

static inline void hypre_UnorderedIntMapFindCloserFreeBucket( hypre_UnorderedIntMap *m,
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
                            hypre_HopscotchSegment* start_seg,
#endif
                            hypre_HopscotchBucket** free_bucket,
                            HYPRE_Int* free_dist)
{
  hypre_HopscotchBucket* move_bucket = *free_bucket - (HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
  HYPRE_Int move_free_dist;
  for (move_free_dist = HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
  {
    hypre_uint start_hop_info = move_bucket->hopInfo;
    HYPRE_Int move_new_free_dist = -1;
    hypre_uint mask = 1;
    HYPRE_Int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1)
    {
      if (mask & start_hop_info)
      {
        move_new_free_dist = i;
        break;
      }
    }
    if (-1 != move_new_free_dist)
    {
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      hypre_HopscotchSegment* move_segment = &(m->segments[(move_bucket - m->table) & m->segmentMask]);
      
      if (start_seg != move_segment)
        omp_set_lock(&move_segment->lock);
#endif

      if (start_hop_info == move_bucket->hopInfo)
      {
        // new_free_bucket -> free_bucket and empty new_free_bucket
        hypre_HopscotchBucket* new_free_bucket = move_bucket + move_new_free_dist;
        (*free_bucket)->data = new_free_bucket->data;
        (*free_bucket)->key  = new_free_bucket->key;
        (*free_bucket)->hash = new_free_bucket->hash;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        ++move_segment->timestamp;

#pragma omp flush
#endif

        move_bucket->hopInfo |= (1U << move_free_dist);
        move_bucket->hopInfo &= ~(1U << move_new_free_dist);

        *free_bucket = new_free_bucket;
        *free_dist -= move_free_dist - move_new_free_dist;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
#endif
        return;
      }
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
#endif
    }
    ++move_bucket;
  }
  *free_bucket = NULL; 
  *free_dist = 0;
}

static inline void hypre_UnorderedBigIntMapFindCloserFreeBucket( hypre_UnorderedBigIntMap *m,
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
                            hypre_HopscotchSegment* start_seg,
#endif
                            hypre_BigHopscotchBucket** free_bucket,
                            HYPRE_Int* free_dist)
{
  hypre_BigHopscotchBucket* move_bucket = *free_bucket - (HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
  HYPRE_Int move_free_dist;
  for (move_free_dist = HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
  {
    hypre_uint start_hop_info = move_bucket->hopInfo;
    HYPRE_Int move_new_free_dist = -1;
    hypre_uint mask = 1;
    HYPRE_Int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1)
    {
      if (mask & start_hop_info)
      {
        move_new_free_dist = i;
        break;
      }
    }
    if (-1 != move_new_free_dist)
    {
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      hypre_HopscotchSegment* move_segment = &(m->segments[(move_bucket - m->table) & m->segmentMask]);
      
      if (start_seg != move_segment)
        omp_set_lock(&move_segment->lock);
#endif

      if (start_hop_info == move_bucket->hopInfo)
      {
        // new_free_bucket -> free_bucket and empty new_free_bucket
        hypre_BigHopscotchBucket* new_free_bucket = move_bucket + move_new_free_dist;
        (*free_bucket)->data = new_free_bucket->data;
        (*free_bucket)->key  = new_free_bucket->key;
        (*free_bucket)->hash = new_free_bucket->hash;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        ++move_segment->timestamp;

#pragma omp flush
#endif

        move_bucket->hopInfo |= (1U << move_free_dist);
        move_bucket->hopInfo &= ~(1U << move_new_free_dist);

        *free_bucket = new_free_bucket;
        *free_dist -= move_free_dist - move_new_free_dist;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
#endif
        return;
      }
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
#endif
    }
    ++move_bucket;
  }
  *free_bucket = NULL; 
  *free_dist = 0;
}

void hypre_UnorderedIntSetCreate( hypre_UnorderedIntSet *s,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedBigIntSetCreate( hypre_UnorderedBigIntSet *s,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedIntMapCreate( hypre_UnorderedIntMap *m,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedBigIntMapCreate( hypre_UnorderedBigIntMap *m,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);

void hypre_UnorderedIntSetDestroy( hypre_UnorderedIntSet *s );
void hypre_UnorderedBigIntSetDestroy( hypre_UnorderedBigIntSet *s );
void hypre_UnorderedIntMapDestroy( hypre_UnorderedIntMap *m );
void hypre_UnorderedBigIntMapDestroy( hypre_UnorderedBigIntMap *m );

// Query Operations .........................................................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
static inline HYPRE_Int hypre_UnorderedIntSetContains( hypre_UnorderedIntSet *s,
                                                HYPRE_Int key )
{
  //CALCULATE HASH ..........................
#ifdef HYPRE_BIGINT
  HYPRE_Int hash = hypre_BigHash(key);
#else
  HYPRE_Int hash = hypre_Hash(key);
#endif

  //CHECK IF ALREADY CONTAIN ................
  hypre_HopscotchSegment *segment = &s->segments[hash & s->segmentMask];
  HYPRE_Int bucket = hash & s->bucketMask;
  hypre_uint hopInfo = s->hopInfo[bucket];

  if (0 == hopInfo)
    return 0;
  else if (1 == hopInfo )
  {
    if (hash == s->hash[bucket] && key == s->key[bucket])
      return 1;
    else return 0;
  }

  HYPRE_Int startTimestamp = segment->timestamp;
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    HYPRE_Int currElm = bucket + i;

    if (hash == s->hash[currElm] && key == s->key[currElm])
      return 1;
    hopInfo &= ~(1U << i);
  } 

  if (segment->timestamp == startTimestamp)
    return 0;

  HYPRE_Int i;
  for (i = 0; i< HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i)
  {
    if (hash == s->hash[bucket + i] && key == s->key[bucket + i])
      return 1;
  }
  return 0;
}

static inline HYPRE_Int hypre_UnorderedBigIntSetContains( hypre_UnorderedBigIntSet *s,
                                                HYPRE_BigInt key )
{
  //CALCULATE HASH ..........................
#if defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT)
  HYPRE_BigInt hash = hypre_BigHash(key);
#else
  HYPRE_BigInt hash = hypre_Hash(key);
#endif

  //CHECK IF ALREADY CONTAIN ................
  hypre_HopscotchSegment *segment = &s->segments[(HYPRE_Int)(hash & s->segmentMask)];
  HYPRE_Int bucket = (HYPRE_Int)(hash & s->bucketMask);
  hypre_uint hopInfo = s->hopInfo[bucket];

  if (0 == hopInfo)
    return 0;
  else if (1 == hopInfo )
  {
    if (hash == s->hash[bucket] && key == s->key[bucket])
      return 1;
    else return 0;
  }

  HYPRE_Int startTimestamp = segment->timestamp;
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    HYPRE_Int currElm = bucket + i;

    if (hash == s->hash[currElm] && key == s->key[currElm])
      return 1;
    hopInfo &= ~(1U << i);
  } 

  if (segment->timestamp == startTimestamp)
    return 0;

  HYPRE_Int i;
  for (i = 0; i< HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i)
  {
    if (hash == s->hash[bucket + i] && key == s->key[bucket + i])
      return 1;
  }
  return 0;
}

/**
 * @ret -1 if key doesn't exist
 */
static inline HYPRE_Int hypre_UnorderedIntMapGet( hypre_UnorderedIntMap *m,
                                           HYPRE_Int key)
{
  //CALCULATE HASH ..........................
#ifdef HYPRE_BIGINT
  HYPRE_Int hash = hypre_BigHash(key);
#else
  HYPRE_Int hash = hypre_Hash(key);
#endif

  //CHECK IF ALREADY CONTAIN ................
  hypre_HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
  hypre_HopscotchBucket *elmAry = &(m->table[hash & m->bucketMask]);
  hypre_uint hopInfo = elmAry->hopInfo;
  if (0 == hopInfo)
    return -1;
  else if (1 == hopInfo )
  {
    if (hash == elmAry->hash && key == elmAry->key)
      return elmAry->data;
    else return -1;
  }

  HYPRE_Int startTimestamp = segment->timestamp;
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    hypre_HopscotchBucket* currElm = elmAry + i;
    if (hash == currElm->hash && key == currElm->key)
      return currElm->data;
    hopInfo &= ~(1U << i);
  } 

  if (segment->timestamp == startTimestamp)
    return -1;

  hypre_HopscotchBucket *currBucket = &(m->table[hash & m->bucketMask]);
  HYPRE_Int i;
  for (i = 0; i< HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket)
  {
    if (hash == currBucket->hash && key == currBucket->key)
      return currBucket->data;
  }
  return -1;
}

static inline HYPRE_Int hypre_UnorderedBigIntMapGet( hypre_UnorderedBigIntMap *m,
                                           HYPRE_BigInt key)
{
  //CALCULATE HASH ..........................
#if defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT)
  HYPRE_BigInt hash = hypre_BigHash(key);
#else
  HYPRE_BigInt hash = hypre_Hash(key);
#endif

  //CHECK IF ALREADY CONTAIN ................
  hypre_HopscotchSegment *segment = &m->segments[(HYPRE_Int)(hash & m->segmentMask)];
  hypre_BigHopscotchBucket *elmAry = &(m->table[(HYPRE_Int)(hash & m->bucketMask)]);
  hypre_uint hopInfo = elmAry->hopInfo;
  if (0 == hopInfo)
    return -1;
  else if (1 == hopInfo )
  {
    if (hash == elmAry->hash && key == elmAry->key)
      return elmAry->data;
    else return -1;
  }

  HYPRE_Int startTimestamp = segment->timestamp;
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    hypre_BigHopscotchBucket* currElm = elmAry + i;
    if (hash == currElm->hash && key == currElm->key)
      return currElm->data;
    hopInfo &= ~(1U << i);
  } 

  if (segment->timestamp == startTimestamp)
    return -1;

  hypre_BigHopscotchBucket *currBucket = &(m->table[hash & m->bucketMask]);
  HYPRE_Int i;
  for (i = 0; i< HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket)
  {
    if (hash == currBucket->hash && key == currBucket->key)
      return currBucket->data;
  }
  return -1;
}
#endif

//status Operations .........................................................
static inline HYPRE_Int hypre_UnorderedIntSetSize(hypre_UnorderedIntSet *s)
{
  HYPRE_Int counter = 0;
  HYPRE_Int n = s->bucketMask + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
  HYPRE_Int i;
  for (i = 0; i < n; ++i)
  {
    if (HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i])
    {
      ++counter;
    }
  }
  return counter;
}   

static inline HYPRE_Int hypre_UnorderedBigIntSetSize(hypre_UnorderedBigIntSet *s)
{
  HYPRE_Int counter = 0;
  HYPRE_BigInt n = s->bucketMask + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
  HYPRE_Int i;
  for (i = 0; i < n; ++i)
  {
    if (HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i])
    {
      ++counter;
    }
  }
  return counter;
}   

static inline HYPRE_Int hypre_UnorderedIntMapSize(hypre_UnorderedIntMap *m)
{
  HYPRE_Int counter = 0;
  HYPRE_Int n = m->bucketMask + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
  HYPRE_Int i;
  for (i = 0; i < n; ++i)
  {
    if( HYPRE_HOPSCOTCH_HASH_EMPTY != m->table[i].hash )
    {
      ++counter;
    }
  }
  return counter;
}

static inline HYPRE_Int hypre_UnorderedBigIntMapSize(hypre_UnorderedBigIntMap *m)
{
  HYPRE_Int counter = 0;
  HYPRE_Int n = m->bucketMask + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
  HYPRE_Int i;
  for (i = 0; i < n; ++i)
  {
    if( HYPRE_HOPSCOTCH_HASH_EMPTY != m->table[i].hash )
    {
      ++counter;
    }
  }
  return counter;
}

HYPRE_Int *hypre_UnorderedIntSetCopyToArray( hypre_UnorderedIntSet *s, HYPRE_Int *len );
HYPRE_BigInt *hypre_UnorderedBigIntSetCopyToArray( hypre_UnorderedBigIntSet *s, HYPRE_Int *len );

//modification Operations ...................................................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
static inline void hypre_UnorderedIntSetPut( hypre_UnorderedIntSet *s,
                                      HYPRE_Int key )
{
  //CALCULATE HASH ..........................
#ifdef HYPRE_BIGINT
  HYPRE_Int hash = hypre_BigHash(key);
#else
  HYPRE_Int hash = hypre_Hash(key);
#endif

  //LOCK KEY HASH ENTERY ....................
  hypre_HopscotchSegment  *segment = &s->segments[hash & s->segmentMask];
  omp_set_lock(&segment->lock);
  HYPRE_Int bucket = hash&s->bucketMask;

  //CHECK IF ALREADY CONTAIN ................
  hypre_uint hopInfo = s->hopInfo[bucket];
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    HYPRE_Int currElm = bucket + i;

    if(hash == s->hash[currElm] && key == s->key[currElm])
    {
      omp_unset_lock(&segment->lock);
      return;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  HYPRE_Int free_bucket = bucket;
  HYPRE_Int free_dist = 0;
  for ( ; free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) && (HYPRE_HOPSCOTCH_HASH_EMPTY == hypre_compare_and_swap((HYPRE_Int *)&s->hash[free_bucket], (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_EMPTY, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_BUSY)) )
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE)
  {
    do
    {
      if (free_dist < HYPRE_HOPSCOTCH_HASH_HOP_RANGE)
      {
        s->key[free_bucket]  = key;
        s->hash[free_bucket] = hash;
        s->hopInfo[bucket]  |= 1U << free_dist;

        omp_unset_lock(&segment->lock);
        return;
      }
      hypre_UnorderedIntSetFindCloserFreeBucket(s,
                                                segment,
                                                &free_bucket, &free_dist);
    } while (-1 != free_bucket);
  }

  //NEED TO RESIZE ..........................
  hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR - RESIZE is not implemented\n");
  /*fprintf(stderr, "ERROR - RESIZE is not implemented\n");*/
  exit(1);
  return;
}

static inline void hypre_UnorderedBigIntSetPut( hypre_UnorderedBigIntSet *s,
                                      HYPRE_BigInt key )
{
  //CALCULATE HASH ..........................
#if defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT)
  HYPRE_BigInt hash = hypre_BigHash(key);
#else
  HYPRE_BigInt hash = hypre_Hash(key);
#endif

  //LOCK KEY HASH ENTERY ....................
  hypre_HopscotchSegment  *segment = &s->segments[hash & s->segmentMask];
  omp_set_lock(&segment->lock);
  HYPRE_Int bucket = (HYPRE_Int)(hash&s->bucketMask);

  //CHECK IF ALREADY CONTAIN ................
  hypre_uint hopInfo = s->hopInfo[bucket];
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    HYPRE_Int currElm = bucket + i;

    if(hash == s->hash[currElm] && key == s->key[currElm])
    {
      omp_unset_lock(&segment->lock);
      return;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  HYPRE_Int free_bucket = bucket;
  HYPRE_Int free_dist = 0;
  for ( ; free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) && (HYPRE_HOPSCOTCH_HASH_EMPTY == hypre_compare_and_swap((HYPRE_Int *)&s->hash[free_bucket], (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_EMPTY, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_BUSY)) )
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE)
  {
    do
    {
      if (free_dist < HYPRE_HOPSCOTCH_HASH_HOP_RANGE)
      {
        s->key[free_bucket]  = key;
        s->hash[free_bucket] = hash;
        s->hopInfo[bucket]  |= 1U << free_dist;

        omp_unset_lock(&segment->lock);
        return;
      }
      hypre_UnorderedBigIntSetFindCloserFreeBucket(s,
                                                segment,
                                                &free_bucket, &free_dist);
    } while (-1 != free_bucket);
  }

  //NEED TO RESIZE ..........................
  hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR - RESIZE is not implemented\n");
  /*fprintf(stderr, "ERROR - RESIZE is not implemented\n");*/
  exit(1);
  return;
}

static inline HYPRE_Int hypre_UnorderedIntMapPutIfAbsent( hypre_UnorderedIntMap *m, HYPRE_Int key, HYPRE_Int data)
{
  //CALCULATE HASH ..........................
#ifdef HYPRE_BIGINT
  HYPRE_Int hash = hypre_BigHash(key);
#else
  HYPRE_Int hash = hypre_Hash(key);
#endif

  //LOCK KEY HASH ENTERY ....................
  hypre_HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
  omp_set_lock(&segment->lock);
  hypre_HopscotchBucket* startBucket = &(m->table[hash & m->bucketMask]);

  //CHECK IF ALREADY CONTAIN ................
  hypre_uint hopInfo = startBucket->hopInfo;
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    hypre_HopscotchBucket* currElm = startBucket + i;
    if (hash == currElm->hash && key == currElm->key)
    {
      HYPRE_Int rc = currElm->data;
      omp_unset_lock(&segment->lock);
      return rc;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  hypre_HopscotchBucket* free_bucket = startBucket;
  HYPRE_Int free_dist = 0;
  for ( ; free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == free_bucket->hash) && (HYPRE_HOPSCOTCH_HASH_EMPTY == __sync_val_compare_and_swap((HYPRE_Int *)&free_bucket->hash, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_EMPTY, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_BUSY)) )
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE)
  {
    do
    {
      if (free_dist < HYPRE_HOPSCOTCH_HASH_HOP_RANGE)
      {
        free_bucket->data     = data;
        free_bucket->key      = key;
        free_bucket->hash     = hash;
        startBucket->hopInfo |= 1U << free_dist;
        omp_unset_lock(&segment->lock);
        return HYPRE_HOPSCOTCH_HASH_EMPTY;
      }
      hypre_UnorderedIntMapFindCloserFreeBucket(m,
                                                segment,
                                                &free_bucket, &free_dist);
    } while (NULL != free_bucket);
  }

  //NEED TO RESIZE ..........................
  hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR - RESIZE is not implemented\n");
  /*fprintf(stderr, "ERROR - RESIZE is not implemented\n");*/
  exit(1);
  return HYPRE_HOPSCOTCH_HASH_EMPTY;
}

static inline HYPRE_Int hypre_UnorderedBigIntMapPutIfAbsent( hypre_UnorderedBigIntMap *m, HYPRE_BigInt key, HYPRE_Int data)
{
  //CALCULATE HASH ..........................
#if defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT)
  HYPRE_BigInt hash = hypre_BigHash(key);
#else
  HYPRE_BigInt hash = hypre_Hash(key);
#endif

  //LOCK KEY HASH ENTERY ....................
  hypre_HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
  omp_set_lock(&segment->lock);
  hypre_BigHopscotchBucket* startBucket = &(m->table[hash & m->bucketMask]);

  //CHECK IF ALREADY CONTAIN ................
  hypre_uint hopInfo = startBucket->hopInfo;
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    hypre_BigHopscotchBucket* currElm = startBucket + i;
    if (hash == currElm->hash && key == currElm->key)
    {
      HYPRE_Int rc = currElm->data;
      omp_unset_lock(&segment->lock);
      return rc;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  hypre_BigHopscotchBucket* free_bucket = startBucket;
  HYPRE_Int free_dist = 0;
  for ( ; free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == free_bucket->hash) && (HYPRE_HOPSCOTCH_HASH_EMPTY == __sync_val_compare_and_swap((HYPRE_Int *)&free_bucket->hash, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_EMPTY, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_BUSY)) )
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE)
  {
    do
    {
      if (free_dist < HYPRE_HOPSCOTCH_HASH_HOP_RANGE)
      {
        free_bucket->data     = data;
        free_bucket->key      = key;
        free_bucket->hash     = hash;
        startBucket->hopInfo |= 1U << free_dist;
        omp_unset_lock(&segment->lock);
        return HYPRE_HOPSCOTCH_HASH_EMPTY;
      }
      hypre_UnorderedBigIntMapFindCloserFreeBucket(m,
                                                segment,
                                                &free_bucket, &free_dist);
    } while (NULL != free_bucket);
  }

  //NEED TO RESIZE ..........................
  hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR - RESIZE is not implemented\n");
  /*fprintf(stderr, "ERROR - RESIZE is not implemented\n");*/
  exit(1);
  return HYPRE_HOPSCOTCH_HASH_EMPTY;
}
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // hypre_HOPSCOTCH_HASH_HEADER
