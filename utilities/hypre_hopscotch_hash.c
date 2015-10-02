#include "hypre_hopscotch_hash.h"

static int NearestPowerOfTwo( int value )
{
  int rc = 1;
  while (rc < value) {
    rc <<= 1;
  }
  return rc;
}

static unsigned int CalcDivideShift( const unsigned int _value )
{
  unsigned int numShift = 0;
  unsigned int curr = 1;
  while (curr < _value)
  {
    curr <<= 1;
    ++numShift;
  }
  return numShift;
}

/*inline void InitBucket(Bucket *b)
{
  b->hopInfo = 0U;
  //b->hash = HYPRE_HOPSCOTCH_HASH_EMPTY;
  //b->key = _EMPTY_KEY;
}*/

static void InitBucketWithIntData(BucketWithIntData *b)
{
  b->hopInfo = 0U;
  b->hash = HYPRE_HOPSCOTCH_HASH_EMPTY;
}

static void InitBucketWithPointerData(BucketWithPointerData *b)
{
  b->hopInfo = 0U;
  b->hash = HYPRE_HOPSCOTCH_HASH_EMPTY;
}

static void InitSegment(hypre_HopscotchHashSegment *s)
{
  s->timestamp = 0;
#ifdef HYPRE_USING_OPENMP
  omp_init_lock(&s->lock);
#endif
}

static void DestroySegment(hypre_HopscotchHashSegment *s)
{
#ifdef HYPRE_USING_OPENMP
  omp_destroy_lock(&s->lock);
#endif
}

void hypre_UnorderedIntSetCreate( hypre_UnorderedIntSet *s,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel) 
{
  s->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
  s->segmentShift = CalcDivideShift(NearestPowerOfTwo(concurrencyLevel/(NearestPowerOfTwo(concurrencyLevel)))-1);

  //ADJUST INPUT ............................
  HYPRE_Int adjInitCap = NearestPowerOfTwo(inCapacity);
  HYPRE_Int adjConcurrencyLevel = NearestPowerOfTwo(concurrencyLevel);
  HYPRE_Int num_buckets = adjInitCap + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
  s->bucketMask = adjInitCap - 1;

  //ALLOCATE THE SEGMENTS ...................
  s->segments = hypre_TAlloc(hypre_HopscotchHashSegment, s->segmentMask + 1);
  s->hopInfo = hypre_TAlloc(HYPRE_Int, num_buckets);
  s->key = hypre_TAlloc(HYPRE_Int, num_buckets);
#ifdef BITMAP
  s->bitmap = hypre_TAlloc(__int64, (num_buckets + 63)/64);
#else
  s->hash = hypre_TAlloc(HYPRE_Int, num_buckets);
#endif

  HYPRE_Int i;
  for (i = 0; i <= s->segmentMask; ++i)
  {
    InitSegment(&s->segments[i]);
  }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for
#endif
  for (i=0; i < num_buckets; ++i)
  {
    s->hopInfo[i] = 0;
#ifndef BITMAP
    s->hash[i] = HYPRE_HOPSCOTCH_HASH_EMPTY;
#endif
  }

#ifdef BITMAP
#pragma omp parallel for
  for (i=0; i < (num_buckets + 63)/64; ++i)
  {
    s->bitmap[i] = 0;
  }
#endif
}

void hypre_UnorderedIntMapCreate( hypre_UnorderedIntMap *m,
                                  HYPRE_Int inCapacity,
                                  HYPRE_Int concurrencyLevel) 
{
  m->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
  m->segmentShift = CalcDivideShift(NearestPowerOfTwo(concurrencyLevel/(NearestPowerOfTwo(concurrencyLevel)))-1);

  //ADJUST INPUT ............................
  HYPRE_Int adjInitCap = NearestPowerOfTwo(inCapacity);
  HYPRE_Int adjConcurrencyLevel = NearestPowerOfTwo(concurrencyLevel);
  HYPRE_Int num_buckets = adjInitCap + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
  m->bucketMask = adjInitCap - 1;

  //ALLOCATE THE SEGMENTS ...................
  m->segments = hypre_TAlloc(hypre_HopscotchHashSegment, m->segmentMask + 1);
  m->table = hypre_TAlloc(BucketWithIntData, num_buckets);

  HYPRE_Int i;
  for (i = 0; i <= m->segmentMask; i++)
  {
    InitSegment(&m->segments[i]);
  }

  BucketWithIntData* curr_bucket = m->table;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for
#endif
  for (i = 0; i < num_buckets; i++)
  {
    InitBucketWithIntData(&m->table[i]);
  }
}

/*inline void HopscotchHashMapWithPointerDataCreate(HopscotchHashMapWithPointerData *h, int inCapacity, int concurrencyLevel) 
{
  h->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
  h->segmentShift = CalcDivideShift(NearestPowerOfTwo(concurrencyLevel/(NearestPowerOfTwo(concurrencyLevel)))-1);

  //ADJUST INPUT ............................
  const int adjInitCap = NearestPowerOfTwo(inCapacity);
  const int adjConcurrencyLevel = NearestPowerOfTwo(concurrencyLevel);
  const int num_buckets = adjInitCap + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
  h->bucketMask = adjInitCap - 1;

  //ALLOCATE THE SEGMENTS ...................
  h->segments = (Segment*) _mm_malloc( (h->segmentMask + 1) * sizeof(Segment), CACHE_LINE_SIZE );
  h->table = (BucketWithPointerData*) _mm_malloc( num_buckets * sizeof(BucketWithPointerData), CACHE_LINE_SIZE );

  Segment* curr_seg = h->segments;
  int iSeg;
  for (iSeg = 0; iSeg <= h->segmentMask; ++iSeg, ++curr_seg) {
    InitSegment(curr_seg);
  }

  BucketWithPointerData* curr_bucket = h->table;
  int iElm;
  for (iElm=0; iElm < num_buckets; ++iElm, ++curr_bucket) {
    InitBucketWithPointerData(curr_bucket);
  }
}*/


void hypre_UnorderedIntSetDestroy( hypre_UnorderedIntSet *s )
{
  hypre_TFree(s->hopInfo);
  hypre_TFree(s->key);
#ifdef BITMAP
  hypre_TFree(s->bitmap);
#else
  hypre_TFree(s->hash);
#endif

#ifdef HYPRE_USING_OPENMP
  HYPRE_Int i;
  for (i = 0; i < s->segmentMask; i++)
  {
    DestroySegment(&s->segments[i]);
  }
#endif
  hypre_TFree(s->segments);
}

void hypre_UnorderedIntMapDestroy( hypre_UnorderedIntMap *m)
{
  hypre_TFree(m->table);

#ifdef HYPRE_USING_OPENMP
  HYPRE_Int i;
  for (i = 0; i < m->segmentMask; i++)
  {
    DestroySegment(&m->segments[i]);
  }
#endif
  hypre_TFree(m->segments);
}

HYPRE_Int *hypre_UnorderedIntCreateArrayCopy( hypre_UnorderedIntSet *s, HYPRE_Int *len )
{
  HYPRE_Int prefix_sum_workspace[hypre_NumThreads() + 1];
  HYPRE_Int *ret_array = NULL;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
#endif
  {
    HYPRE_Int n = s->bucketMask + HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
    HYPRE_Int i_begin, i_end;
    hypre_GetSimpleThreadPartition(&i_begin, &i_end, n);

    HYPRE_Int cnt = 0;
    for (HYPRE_Int i = i_begin; i < i_end; i++)
    {
      if (HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) cnt++;
    }

    hypre_prefix_sum(&cnt, len, prefix_sum_workspace);

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#pragma omp master
#endif
    {
      ret_array = hypre_TAlloc(HYPRE_Int, *len);
    }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

    for (HYPRE_Int i = i_begin; i < i_end; i++)
    {
      if (HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) ret_array[cnt++] = s->key[i];
    }
  }

  return ret_array;
}
