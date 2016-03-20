#include "hypre_hopscotch_hash.h"

static HYPRE_Int NearestPowerOfTwo( HYPRE_Int value )
{
  HYPRE_Int rc = 1;
  while (rc < value) {
    rc <<= 1;
  }
  return rc;
}

static void InitBucket(hypre_HopscotchBucket *b)
{
  b->hopInfo = 0;
  b->hash = HYPRE_HOPSCOTCH_HASH_EMPTY;
}

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
static void InitSegment(hypre_HopscotchSegment *s)
{
  s->timestamp = 0;
  omp_init_lock(&s->lock);
}

static void DestroySegment(hypre_HopscotchSegment *s)
{
  omp_destroy_lock(&s->lock);
}
#endif

void hypre_UnorderedIntSetCreate( hypre_UnorderedIntSet *s,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel) 
{
  s->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
  if (inCapacity < s->segmentMask + 1)
  {
    inCapacity = s->segmentMask + 1;
  }

  //ADJUST INPUT ............................
  HYPRE_Int adjInitCap = NearestPowerOfTwo(inCapacity+4096);
  HYPRE_Int num_buckets = adjInitCap + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
  s->bucketMask = adjInitCap - 1;

  HYPRE_Int i;

  //ALLOCATE THE SEGMENTS ...................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  s->segments = hypre_TAlloc(hypre_HopscotchSegment, s->segmentMask + 1);
  for (i = 0; i <= s->segmentMask; ++i)
  {
    InitSegment(&s->segments[i]);
  }
#endif

  s->hopInfo = hypre_TAlloc(hypre_uint, num_buckets);
  s->key = hypre_TAlloc(HYPRE_Int, num_buckets);
  s->hash = hypre_TAlloc(HYPRE_Int, num_buckets);

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
#pragma omp parallel for
#endif
  for (i = 0; i < num_buckets; ++i)
  {
    s->hopInfo[i] = 0;
    s->hash[i] = HYPRE_HOPSCOTCH_HASH_EMPTY;
  }
}

void hypre_UnorderedIntMapCreate( hypre_UnorderedIntMap *m,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel) 
{
  m->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
  if (inCapacity < m->segmentMask + 1)
  {
    inCapacity = m->segmentMask + 1;
  }

  //ADJUST INPUT ............................
  HYPRE_Int adjInitCap = NearestPowerOfTwo(inCapacity+4096);
  HYPRE_Int num_buckets = adjInitCap + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
  m->bucketMask = adjInitCap - 1;

  HYPRE_Int i;

  //ALLOCATE THE SEGMENTS ...................
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  m->segments = hypre_TAlloc(hypre_HopscotchSegment, m->segmentMask + 1);
  for (i = 0; i <= m->segmentMask; i++)
  {
    InitSegment(&m->segments[i]);
  }
#endif

  m->table = hypre_TAlloc(hypre_HopscotchBucket, num_buckets);

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
#pragma omp parallel for
#endif
  for (i = 0; i < num_buckets; i++)
  {
    InitBucket(&m->table[i]);
  }
}

void hypre_UnorderedIntSetDestroy( hypre_UnorderedIntSet *s )
{
  hypre_TFree(s->hopInfo);
  hypre_TFree(s->key);
  hypre_TFree(s->hash);

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  HYPRE_Int i;
  for (i = 0; i <= s->segmentMask; i++)
  {
    DestroySegment(&s->segments[i]);
  }
  hypre_TFree(s->segments);
#endif
}

void hypre_UnorderedIntMapDestroy( hypre_UnorderedIntMap *m)
{
  hypre_TFree(m->table);

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
  HYPRE_Int i;
  for (i = 0; i <= m->segmentMask; i++)
  {
    DestroySegment(&m->segments[i]);
  }
  hypre_TFree(m->segments);
#endif
}

HYPRE_Int *hypre_UnorderedIntSetCopyToArray( hypre_UnorderedIntSet *s, HYPRE_Int *len )
{
  /*HYPRE_Int prefix_sum_workspace[hypre_NumThreads() + 1];*/
  HYPRE_Int *prefix_sum_workspace;
  HYPRE_Int *ret_array = NULL;

  prefix_sum_workspace = hypre_TAlloc(HYPRE_Int, hypre_NumThreads() + 1);

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
#pragma omp parallel
#endif
  {
    HYPRE_Int n = s->bucketMask + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
    HYPRE_Int i_begin, i_end;
    hypre_GetSimpleThreadPartition(&i_begin, &i_end, n);

    HYPRE_Int cnt = 0;
    HYPRE_Int i;
    for (i = i_begin; i < i_end; i++)
    {
      if (HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) cnt++;
    }

    hypre_prefix_sum(&cnt, len, prefix_sum_workspace);

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
#pragma omp barrier
#pragma omp master
#endif
    {
      ret_array = hypre_TAlloc(HYPRE_Int, *len);
    }
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
#pragma omp barrier
#endif

    for (i = i_begin; i < i_end; i++)
    {
      if (HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) ret_array[cnt++] = s->key[i];
    }
  }

  hypre_TFree(prefix_sum_workspace);

  return ret_array;
}
