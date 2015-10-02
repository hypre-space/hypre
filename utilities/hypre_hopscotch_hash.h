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
//	Permission to use, copy, modify and distribute this software and
//	its documentation for any purpose is hereby granted without fee,
//	provided that due acknowledgments to the authors are provided and
//	this permission notice appears in all copies of the software.
//	The software is provided "as is". There is no warranty of any kind.
//
//Authors:
//	Maurice Herlihy
//	Brown University
//	and
//	Nir Shavit
//	Tel-Aviv University
//	and
//	Moran Tzafrir
//	Tel-Aviv University
//
//	Date: July 15, 2008.  
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

#ifdef __cplusplus
extern "C" {
#endif

// Constants ................................................................
#define HYPRE_HOPSCOTCH_HASH_HOP_RANGE		 (32)
#define HYPRE_HOPSCOTCH_HASH_INSERT_RANGE	 (4*1024)

inline int first_lsb_bit_indx(int x) {
	if(0==x) 
		return -1;
	return __builtin_ffs(x)-1;
}

#define HYPRE_HOPSCOTCH_HASH_EMPTY (0)
#define HYPRE_HOPSCOTCH_HASH_BUSY (1)

// assumption: key is non-negative
inline unsigned int hypre_Hash(int key) {
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

// Small Utilities ..........................................................
inline void hypre_UnorderedIntSetFindCloserFreeBucket( hypre_UnorderedIntSet *s, const hypre_HopscotchHashSegment* const start_seg, int *free_bucket, int *free_distance )
{
  int move_bucket = *free_bucket - (HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
  int move_free_dist;
  for (move_free_dist = HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist) {
    int start_hop_info = s->hopInfo[move_bucket];
    int move_new_free_distance = -1;
    int mask = 1;
    int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1) {
      if (mask & start_hop_info) {
        move_new_free_distance = i;
        break;
      }
    }
    if (-1 != move_new_free_distance) {
      hypre_HopscotchHashSegment*	const move_segment = &(s->segments[(move_bucket >> s->segmentShift) & s->segmentMask]);
      
#ifdef HYPRE_USING_OPENMP
      if(start_seg != move_segment)
        omp_set_lock(&move_segment->lock);
#endif

      if (start_hop_info == s->hopInfo[move_bucket]) {
        int new_free_bucket = move_bucket + move_new_free_distance;
        s->key[*free_bucket]   = s->key[new_free_bucket];
        s->hash[*free_bucket] = s->hash[new_free_bucket];

        ++move_segment->timestamp;

#ifdef HYPRE_USING_OPENMP
#pragma omp flush
#endif

        s->hopInfo[move_bucket] |= (1U << move_free_dist);
        s->hopInfo[move_bucket] &= ~(1U << move_new_free_distance);

        *free_bucket = new_free_bucket;
        *free_distance -= move_free_dist;

#ifdef HYPRE_USING_OPENMP
        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
#endif

        return;
      }
#ifdef HYPRE_USING_OPENMP
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
#endif
    }
    ++move_bucket;
  }
  *free_bucket = -1; 
  *free_distance = 0;
}

inline void hypre_UnorderedIntMapFindCloserFreeBucket(hypre_UnorderedIntMap *m, const hypre_HopscotchHashSegment* const start_seg, BucketWithIntData** free_bucket, int* free_distance) {
  BucketWithIntData* move_bucket = *free_bucket - (HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
  int move_free_dist;
  for (move_free_dist = HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist) {
    int start_hop_info = move_bucket->hopInfo;
    int move_new_free_distance = -1;
    int mask = 1;
    int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1) {
      if (mask & start_hop_info) {
        move_new_free_distance = i;
        break;
      }
    }
    if (-1 != move_new_free_distance) {
      hypre_HopscotchHashSegment*	const move_segment = &(m->segments[((move_bucket - m->table) >> m->segmentShift) & m->segmentMask]);
      
#ifdef HYPRE_USING_OPENMP
      if(start_seg != move_segment)
        omp_set_lock(&move_segment->lock);
#endif

      if (start_hop_info == move_bucket->hopInfo) {
        BucketWithIntData* new_free_bucket = move_bucket + move_new_free_distance;
        (*free_bucket)->data  = new_free_bucket->data;
        (*free_bucket)->key   = new_free_bucket->key;
        (*free_bucket)->hash  = new_free_bucket->hash;

        ++(move_segment->timestamp);
#pragma omp flush

        move_bucket->hopInfo |= (1U << move_free_dist);
        move_bucket->hopInfo &= ~(1U << move_new_free_distance);

        *free_bucket = new_free_bucket;
        *free_distance -= move_free_dist;

#ifdef HYPRE_USING_OPENMP
        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
#endif
        return;
      }
#ifdef HYPRE_USING_OPENMP
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
#endif
    }
    ++move_bucket;
  }
  *free_bucket = 0; 
  *free_distance = 0;
}

/*inline void find_closer_free_bucket_with_pointer_data(HopscotchHashMapWithPointerData *h, const hypre_HopscotchHashSegment* const start_seg, BucketWithPointerData** free_bucket, int* free_distance) {
  BucketWithPointerData* move_bucket = *free_bucket - (HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
  int move_free_dist;
  for (move_free_dist = HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist) {
    int start_hop_info = move_bucket->hopInfo;
    int move_new_free_distance = -1;
    int mask = 1;
    int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1) {
      if (mask & start_hop_info) {
        move_new_free_distance = i;
        break;
      }
    }
    if (-1 != move_new_free_distance) {
      hypre_HopscotchHashSegment*	const move_segment = &(h->segments[((move_bucket - h->table) >> h->segmentShift) & h->segmentMask]);
      
      if(start_seg != move_segment)
        omp_set_lock(&move_segment->lock);

      if (start_hop_info == move_bucket->hopInfo) {
        BucketWithPointerData* new_free_bucket = move_bucket + move_new_free_distance;
        (*free_bucket)->data  = new_free_bucket->data;
        (*free_bucket)->key   = new_free_bucket->key;
        (*free_bucket)->hash  = new_free_bucket->hash;

        ++(move_segment->timestamp);
#pragma omp flush

        move_bucket->hopInfo |= (1U << move_free_dist);
        move_bucket->hopInfo &= ~(1U << move_new_free_distance);

        *free_bucket = new_free_bucket;
        *free_distance -= move_free_dist;

        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
        return;
      }
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
    }
    ++move_bucket;
  }
  *free_bucket = 0; 
  *free_distance = 0;
}*/

void hypre_UnorderedIntSetCreate( hypre_UnorderedIntSet *s,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);
void hypre_UnorderedIntMapCreate( hypre_UnorderedIntMap *m,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel);

/*inline void HopscotchHashMapWithPointerDataCreate(HopscotchHashMapWithPointerData *h, int inCapacity, int concurrencyLevel) */

void hypre_UnorderedIntSetDestroy( hypre_UnorderedIntSet *s );
void hypre_UnorderedIntMapDestroy( hypre_UnorderedIntMap *m );

/*inline void HopscotchHashMapWithPointerDataFree(HopscotchHashMapWithPointerData *h) {
  _mm_free(h->table);
  _mm_free(h->segments);
}*/

// Query Operations .........................................................
inline int hypre_UnorderedIntSetContains( hypre_UnorderedIntSet *s, int key )
{
  //CALCULATE HASH ..........................
  const unsigned int hash = hypre_Hash(key);

  //CHECK IF ALREADY CONTAIN ................
  const	hypre_HopscotchHashSegment	*segment = &s->segments[(hash >> s->segmentShift) & s->segmentMask];
  int pos = hash & s->bucketMask;
  int hopInfo = s->hopInfo[pos];

  if(0U ==hopInfo)
    return 0;
  else if(1U == hopInfo ) {
    if(hash == s->hash[pos] && key == s->key[pos])
      return 1;
    else return 0;
  }

  const	int startTimestamp = segment->timestamp;
  while(0U != hopInfo) {
    const int i = first_lsb_bit_indx(hopInfo);
    int currElm = pos + i;

    if(hash == s->hash[currElm] && key == s->key[currElm])
      return 1;
    hopInfo &= ~(1U << i);
  } 

  //-----------------------------------------
  if( segment->timestamp == startTimestamp)
    return 0;

  //-----------------------------------------
  int i;
  for(i = 0; i< HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i) {
    if(hash == s->hash[pos + i] && key == s->key[pos + i])
      return 1;
  }
  return 0;
}

/**
 * @ret -1 if key doesn't exist
 */
inline int hypre_UnorderedIntMapGet(hypre_UnorderedIntMap *m, int key)
{
  //CALCULATE HASH ..........................
  const unsigned int hash =  hypre_Hash(key);

  //CHECK IF ALREADY CONTAIN ................
  const	hypre_HopscotchHashSegment	*segment = &m->segments[(hash >> m->segmentShift) & m->segmentMask];
  const BucketWithIntData* const elmAry = &(m->table[hash & m->bucketMask]);
  int hopInfo = elmAry->hopInfo;
  if(0U ==hopInfo)
    return -1;
  else if(1U == hopInfo ) {
    if(hash == elmAry->hash && key == elmAry->key)
      return elmAry->data;
    else return -1;
  }

  const	int startTimestamp = segment->timestamp;
  while(0U != hopInfo) {
    const int i = first_lsb_bit_indx(hopInfo);
    const BucketWithIntData* currElm = elmAry + i;
    if(hash == currElm->hash && key == currElm->key)
      return currElm->data;
    hopInfo &= ~(1U << i);
  } 

  //-----------------------------------------
  if( segment->timestamp == startTimestamp)
    return -1;

  //-----------------------------------------
  const	BucketWithIntData* currBucket = &(m->table[hash & m->bucketMask]);
  int i;
  for(i = 0; i< HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket) {
    if(hash == currBucket->hash && key == currBucket->key)
      return currBucket->data;
  }
  return -1;
}

/*inline int containsKeyWithPointerData(HopscotchHashMapWithPointerData *h, int key) {
  //CALCULATE HASH ..........................
  const unsigned int hash =  hypre_Hash(key);

  //CHECK IF ALREADY CONTAIN ................
  const	hypre_HopscotchHashSegment	*segment = &h->segments[(hash >> h->segmentShift) & h->segmentMask];
  const BucketWithPointerData* const elmAry = &(h->table[hash & h->bucketMask]);
  int hopInfo = elmAry->hopInfo;

  if(0U ==hopInfo)
    return 0;
  else if(1U == hopInfo ) {
    if(hash == elmAry->hash && key == elmAry->key)
      return 1;
    else return 0;
  }

  const	int startTimestamp = segment->timestamp;
  while(0U != hopInfo) {
    const int i = first_lsb_bit_indx(hopInfo);
    const BucketWithPointerData* currElm = elmAry + i;
    if(hash == currElm->hash && key == currElm->key)
      return 1;
    hopInfo &= ~(1U << i);
  } 

  //-----------------------------------------
  if( segment->timestamp == startTimestamp)
    return 0;

  //-----------------------------------------
  const	BucketWithPointerData* currBucket = &(h->table[hash & h->bucketMask]);
  int i;
  for(i = 0; i< HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket) {
    if(hash == currBucket->hash && key == currBucket->key)
      return 1;
  }
  return 0;
}*/

//status Operations .........................................................
inline int hypre_UnorderedIntSetSize(hypre_UnorderedIntSet *s)
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

inline int hypre_UnorderedIntMapSize(hypre_UnorderedIntMap *m)	{
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

HYPRE_Int *hypre_UnorderedIntCreateArrayCopy( hypre_UnorderedIntSet *s, HYPRE_Int *len );

/*inline int HopscotchHashMapWithPointerDataSize(HopscotchHashMapWithPointerData *h)	{
  int counter = 0;
  const int num_elm = h->bucketMask + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
  int iElm;
  for(iElm=0; iElm < num_elm; ++iElm) {
    if( HYPRE_HOPSCOTCH_HASH_EMPTY != h->table[iElm].hash ) {
      ++counter;
    }
  }
  return counter;
}*/

//modification Operations ...................................................
inline void hypre_UnorderedIntSetPut( hypre_UnorderedIntSet *s,
                                                    int key )
{
  //CALCULATE HASH ..........................
  const unsigned int hash = hypre_Hash(key);

  //LOCK KEY HASH ENTERY ....................
  hypre_HopscotchHashSegment	*segment = &s->segments[(hash >> s->segmentShift) & s->segmentMask];
#ifdef HYPRE_USING_OPENMP
  omp_set_lock(&segment->lock);
#endif
  int pos = hash&s->bucketMask;

  //CHECK IF ALREADY CONTAIN ................
  int hopInfo = s->hopInfo[pos];
  while(0 != hopInfo) {
    const int i = first_lsb_bit_indx(hopInfo);
    int currElm = pos + i;

    if(hash == s->hash[currElm] && key == s->key[currElm]) {
#ifdef HYPRE_USING_OPENMP
      omp_unset_lock(&segment->lock);
#endif
      return;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  int free_bucket = pos;
  int free_distance = 0;
  for(; free_distance < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_distance, ++free_bucket) {
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) && (HYPRE_HOPSCOTCH_HASH_EMPTY == __sync_val_compare_and_swap(&s->hash[free_bucket], HYPRE_HOPSCOTCH_HASH_EMPTY, HYPRE_HOPSCOTCH_HASH_BUSY)) )
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_distance < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE) {
    do {
      if (free_distance < HYPRE_HOPSCOTCH_HASH_HOP_RANGE) {
        s->key[free_bucket]		= key;
        s->hash[free_bucket] = hash;
        s->hopInfo[pos] |= (1U << free_distance);

#ifdef HYPRE_USING_OPENMP
        omp_unset_lock(&segment->lock);
#endif
        return;
      }
      hypre_UnorderedIntSetFindCloserFreeBucket(s, segment, &free_bucket, &free_distance);
    } while (-1 != free_bucket);
  }

  //NEED TO RESIZE ..........................
  fprintf(stderr, "ERROR - RESIZE is not implemented - size %u\n", hypre_UnorderedIntSetSize(s));
  exit(1);
  return;
}

inline int hypre_UnorderedIntMapPut( hypre_UnorderedIntMap *m, int key, int data) {
  //CALCULATE HASH ..........................
  const unsigned int hash = hypre_Hash(key);

  //LOCK KEY HASH ENTERY ....................
  hypre_HopscotchHashSegment	*segment = &m->segments[(hash >> m->segmentShift) & m->segmentMask];
#ifdef HYPRE_USING_OPENMP
  omp_set_lock(&segment->lock);
#endif
  BucketWithIntData* const startBucket = &(m->table[hash & m->bucketMask]);

  //CHECK IF ALREADY CONTAIN ................
  int hopInfo = startBucket->hopInfo;
  while(0 != hopInfo) {
    const int i = first_lsb_bit_indx(hopInfo);
    const BucketWithIntData* currElm = startBucket + i;
    if(hash == currElm->hash && key == currElm->key) {
      int rc = currElm->data;
#ifdef HYPRE_USING_OPENMP
      omp_unset_lock(&segment->lock);
#endif
      return rc;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  BucketWithIntData* free_bucket = startBucket;
  int free_distance = 0;
  for(; free_distance < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_distance, ++free_bucket) {
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == free_bucket->hash) &&	(HYPRE_HOPSCOTCH_HASH_EMPTY == __sync_val_compare_and_swap(&(free_bucket->hash), HYPRE_HOPSCOTCH_HASH_EMPTY, HYPRE_HOPSCOTCH_HASH_BUSY)) )
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_distance < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE) {
    do {
      if (free_distance < HYPRE_HOPSCOTCH_HASH_HOP_RANGE) {
        free_bucket->data   = data;
        free_bucket->key		= key;
        free_bucket->hash   = hash;
        startBucket->hopInfo |= (1U << free_distance);
#ifdef HYPRE_USING_OPENMP
        omp_unset_lock(&segment->lock);
#endif
        return HYPRE_HOPSCOTCH_HASH_EMPTY;
      }
      hypre_UnorderedIntMapFindCloserFreeBucket(m, segment, &free_bucket, &free_distance);
    } while (0 != free_bucket);
  }

  //NEED TO RESIZE ..........................
  fprintf(stderr, "ERROR - RESIZE is not implemented - size %u\n", hypre_UnorderedIntMapSize(m));
  exit(1);
  return HYPRE_HOPSCOTCH_HASH_EMPTY;
}

/*inline void *putPointerIfAbsent(HopscotchHashMapWithPointerData *h, const int key,  void *data) {
  //CALCULATE HASH ..........................
  const unsigned int hash = hypre_Hash(key);

  //LOCK KEY HASH ENTERY ....................
  hypre_HopscotchHashSegment	*segment = &h->segments[(hash >> h->segmentShift) & h->segmentMask];
  omp_set_lock(&segment->lock);
  BucketWithPointerData* const startBucket = &(h->table[hash & h->bucketMask]);

  //CHECK IF ALREADY CONTAIN ................
  int hopInfo = startBucket->hopInfo;
  while(0 != hopInfo) {
    const int i = first_lsb_bit_indx(hopInfo);
    const BucketWithPointerData* currElm = startBucket + i;
    if(hash == currElm->hash && key == currElm->key) {
      void *rc = currElm->data;
      omp_unset_lock(&segment->lock);
      return rc;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  BucketWithPointerData* free_bucket = startBucket;
  int free_distance = 0;
  for(; free_distance < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_distance, ++free_bucket) {
    if( (HYPRE_HOPSCOTCH_HASH_EMPTY == free_bucket->hash) &&	(HYPRE_HOPSCOTCH_HASH_EMPTY == __sync_val_compare_and_swap(&(free_bucket->hash), HYPRE_HOPSCOTCH_HASH_EMPTY, HYPRE_HOPSCOTCH_HASH_BUSY)) )
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_distance < HYPRE_HOPSCOTCH_HASH_INSERT_RANGE) {
    do {
      if (free_distance < HYPRE_HOPSCOTCH_HASH_HOP_RANGE) {
        free_bucket->data   = data;
        free_bucket->key		= key;
        free_bucket->hash   = hash;
        startBucket->hopInfo |= (1U << free_distance);
      omp_unset_lock(&segment->lock);
        return HYPRE_HOPSCOTCH_HASH_EMPTY;
      }
      find_closer_free_bucket_with_pointer_data(h, segment, &free_bucket, &free_distance);
    } while (0 != free_bucket);
  }

  //NEED TO RESIZE ..........................
  fprintf(stderr, "ERROR - RESIZE is not implemented - size %u\n", HopscotchHashMapWithPointerDataSize(h));
  exit(1);
  return HYPRE_HOPSCOTCH_HASH_EMPTY;
}*/

#ifdef __cplusplus
} // extern "C"
#endif

#endif // hypre_HOPSCOTCH_HASH_HEADER
