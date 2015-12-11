/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

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

#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include <math.h>

#ifdef HYPRE_USING_OPENMP
#include <omp.h>
#endif

#include "_hypre_utilities.h"

// Potentially architecture specific features used here:
// __builtin_ffs
// __sync_val_compare_and_swap

#ifdef __cplusplus
extern "C" {
#endif

// Constants ................................................................
#define HYPRE_HOPSCOTCH_HASH_HOP_RANGE    (32U)
#define HYPRE_HOPSCOTCH_HASH_INSERT_RANGE (4*1024U)

#define HYPRE_HOPSCOTCH_HASH_EMPTY (0U)
#define HYPRE_HOPSCOTCH_HASH_BUSY  (1U)

// Small Utilities ..........................................................
static inline HYPRE_Int first_lsb_bit_indx(hypre_uint x) {
  if (0 == x) return -1;
  return __builtin_ffs(x) - 1;
}

// assumption: key is non-negative
static inline hypre_uint hypre_Hash(HYPRE_Int key) {
  key -= 339522179;
  key ^= (key << 15) ^ 0xcd7dcd7d;
  key ^= (key >> 10);
  key ^= (key <<  3);
  key ^= (key >>  6);
  key ^= (key <<  2) + (key << 14);
  key ^= (key >> 16);

  // only -2147483648 and -1073748731 gives HYPRE_HOPSCOTCH_HASH_EMPTY,
  // and we're fine as long as key is non-negative
  assert(HYPRE_HOPSCOTCH_HASH_EMPTY != key);
  return key;
}

static inline void hypre_UnorderedIntSetFindCloserFreeBucket( hypre_UnorderedIntSet *s,
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
                                                       hypre_HopscotchSegment* start_seg,
#endif
                                                       HYPRE_Int *free_bucket,
                                                       hypre_uint *free_dist )
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
                                                       hypre_uint* free_dist)
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

void hypre_UnorderedIntSetCreate( hypre_UnorderedIntSet *s,
                                  hypre_uint inCapacity,
                                  hypre_uint concurrencyLevel);
void hypre_UnorderedIntMapCreate( hypre_UnorderedIntMap *m,
                                  hypre_uint inCapacity,
                                  hypre_uint concurrencyLevel);

void hypre_UnorderedIntSetDestroy( hypre_UnorderedIntSet *s );
void hypre_UnorderedIntMapDestroy( hypre_UnorderedIntMap *m );

// Query Operations .........................................................
static inline HYPRE_Int hypre_UnorderedIntSetContains( hypre_UnorderedIntSet *s,
                                                HYPRE_Int key )
{
  //CALCULATE HASH ..........................
  hypre_uint hash = hypre_Hash(key);

  //CHECK IF ALREADY CONTAIN ................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  hypre_HopscotchSegment *segment = &s->segments[hash & s->segmentMask];
#endif
  HYPRE_Int bucket = hash & s->bucketMask;
  hypre_uint hopInfo = s->hopInfo[bucket];

  if (0 ==hopInfo)
    return 0;
  else if (1 == hopInfo )
  {
    if (hash == s->hash[bucket] && key == s->key[bucket])
      return 1;
    else return 0;
  }

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  hypre_uint startTimestamp = segment->timestamp;
#endif
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    HYPRE_Int currElm = bucket + i;

    if (hash == s->hash[currElm] && key == s->key[currElm])
      return 1;
    hopInfo &= ~(1U << i);
  } 

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  if (segment->timestamp == startTimestamp)
    return 0;
#endif

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
  hypre_uint hash = hypre_Hash(key);

  //CHECK IF ALREADY CONTAIN ................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  hypre_HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
#endif
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

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  hypre_uint startTimestamp = segment->timestamp;
#endif
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    hypre_HopscotchBucket* currElm = elmAry + i;
    if (hash == currElm->hash && key == currElm->key)
      return currElm->data;
    hopInfo &= ~(1U << i);
  } 

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  if (segment->timestamp == startTimestamp)
    return -1;
#endif

  hypre_HopscotchBucket *currBucket = &(m->table[hash & m->bucketMask]);
  HYPRE_Int i;
  for (i = 0; i< HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket)
  {
    if (hash == currBucket->hash && key == currBucket->key)
      return currBucket->data;
  }
  return -1;
}

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

HYPRE_Int *hypre_UnorderedIntSetCopyToArray( hypre_UnorderedIntSet *s, HYPRE_Int *len );

//modification Operations ...................................................
static inline void hypre_UnorderedIntSetPut( hypre_UnorderedIntSet *s,
                                      HYPRE_Int key )
{
  //CALCULATE HASH ..........................
  hypre_uint hash = hypre_Hash(key);

  //LOCK KEY HASH ENTERY ....................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  hypre_HopscotchSegment  *segment = &s->segments[hash & s->segmentMask];
  omp_set_lock(&segment->lock);
#endif
  HYPRE_Int bucket = hash&s->bucketMask;

  //CHECK IF ALREADY CONTAIN ................
  hypre_uint hopInfo = s->hopInfo[bucket];
  while (0 != hopInfo)
  {
    HYPRE_Int i = first_lsb_bit_indx(hopInfo);
    HYPRE_Int currElm = bucket + i;

    if(hash == s->hash[currElm] && key == s->key[currElm])
    {
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      omp_unset_lock(&segment->lock);
#endif
      return;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  HYPRE_Int free_bucket = bucket;
  hypre_uint free_dist = 0;
  for ( ; free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) && (HYPRE_HOPSCOTCH_HASH_EMPTY == hypre_compare_and_swap((HYPRE_Int *)&s->hash[free_bucket], (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_EMPTY, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_BUSY)) )
#else
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) )
#endif
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

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        omp_unset_lock(&segment->lock);
#endif
        return;
      }
      hypre_UnorderedIntSetFindCloserFreeBucket(s,
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
                                                segment,
#endif
                                                &free_bucket, &free_dist);
    } while (-1 != free_bucket);
  }

  //NEED TO RESIZE ..........................
  fprintf(stderr, "ERROR - RESIZE is not implemented - size %u\n", hypre_UnorderedIntSetSize(s));
  exit(1);
  return;
}

static inline HYPRE_Int hypre_UnorderedIntMapPutIfAbsent( hypre_UnorderedIntMap *m, HYPRE_Int key, HYPRE_Int data)
{
  //CALCULATE HASH ..........................
  hypre_uint hash = hypre_Hash(key);

  //LOCK KEY HASH ENTERY ....................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  hypre_HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
  omp_set_lock(&segment->lock);
#endif
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
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      omp_unset_lock(&segment->lock);
#endif
      return rc;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  hypre_HopscotchBucket* free_bucket = startBucket;
  hypre_uint free_dist = 0;
  for ( ; free_dist < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == free_bucket->hash) && (HYPRE_HOPSCOTCH_HASH_EMPTY == __sync_val_compare_and_swap((HYPRE_Int *)&free_bucket->hash, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_EMPTY, (HYPRE_Int)HYPRE_HOPSCOTCH_HASH_BUSY)) )
#else
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == free_bucket->hash) )
#endif
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
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
        omp_unset_lock(&segment->lock);
#endif
        return HYPRE_HOPSCOTCH_HASH_EMPTY;
      }
      hypre_UnorderedIntMapFindCloserFreeBucket(m,
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
                                                segment,
#endif
                                                &free_bucket, &free_dist);
    } while (NULL != free_bucket);
  }

  //NEED TO RESIZE ..........................
  fprintf(stderr, "ERROR - RESIZE is not implemented - size %u\n", hypre_UnorderedIntMapSize(m));
  exit(1);
  return HYPRE_HOPSCOTCH_HASH_EMPTY;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // hypre_HOPSCOTCH_HASH_HEADER
