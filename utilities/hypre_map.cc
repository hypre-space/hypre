#include "_hypre_utilities.h"

#include <cassert>
#include <unordered_set>
#include <unordered_map>

using namespace std;

extern "C"
{

typedef struct HYPRE_IntSet
{
   unordered_set<HYPRE_Int> set;
   unordered_set<HYPRE_Int>::iterator itr;
} HYPRE_IntSet;

HYPRE_IntSet *hypre_IntSetCreate()
{
   return new HYPRE_IntSet;
}

void hypre_IntSetDestroy( HYPRE_IntSet *set )
{
   delete set;
}

void hypre_IntSetInsert( HYPRE_IntSet *set, HYPRE_Int x)
{
   set->set.insert(x);
}

HYPRE_Int hypre_IntSetContain( HYPRE_IntSet *set, HYPRE_Int x)
{
   return set->set.find(x) != set->set.end();
}

HYPRE_Int hypre_IntSetSize( HYPRE_IntSet *set )
{
   return set->set.size();
}

void hypre_IntSetBegin( HYPRE_IntSet *set )
{
   set->itr = set->set.begin();
}

HYPRE_Int hypre_IntSetNext( HYPRE_IntSet *set )
{
   HYPRE_Int ret = *set->itr;
   ++set->itr;
   return ret;
}

HYPRE_Int hypre_IntSetHasNext( HYPRE_IntSet *set )
{
   return set->itr != set->set.end();
}

typedef unordered_map<HYPRE_Int, HYPRE_Int> HYPRE_Int2Int_;

HYPRE_Int2Int *hypre_Int2IntCreate()
{
   return (HYPRE_Int2Int *)(new HYPRE_Int2Int_());
}

void hypre_Int2IntDestroy(HYPRE_Int2Int *map)
{
   delete (HYPRE_Int2Int_ *)map;
}

void hypre_Int2IntInsert(HYPRE_Int2Int *map, HYPRE_Int key, HYPRE_Int value)
{
   ((HYPRE_Int2Int_ *)map)->insert(make_pair(key, value));
}

HYPRE_Int *hypre_Int2IntFind(HYPRE_Int2Int *map, HYPRE_Int key)
{
   HYPRE_Int2Int_ *m = (HYPRE_Int2Int_ *)map;
   HYPRE_Int2Int_::iterator itr = m->find(key);
   return itr == m->end() ? NULL : &itr->second;
}

typedef unordered_map<HYPRE_Int, HYPRE_IntSet> HYPRE_Int2IntSet_;

HYPRE_Int2IntSet *hypre_Int2IntSetCreate()
{
   return (HYPRE_Int2IntSet *)(new HYPRE_Int2IntSet_());
}

void hypre_Int2IntSetDestroy( HYPRE_Int2IntSet *map )
{
   delete (HYPRE_Int2IntSet_ *)map;
}

void hypre_Int2IntSetInsert( HYPRE_Int2IntSet *map, HYPRE_Int key, HYPRE_Int value )
{
   HYPRE_Int2IntSet_ *m = (HYPRE_Int2IntSet_ *)map;
   HYPRE_Int2IntSet_::iterator itr = m->find(key);
   if (itr == m->end())
   {
      itr = m->insert(make_pair(key, HYPRE_IntSet())).first;
   }
   itr->second.set.insert(value);
}

HYPRE_IntSet *hypre_Int2IntSetFind( HYPRE_Int2IntSet *map, HYPRE_Int key )
{
   HYPRE_Int2IntSet_ *m = (HYPRE_Int2IntSet_ *)map;
   HYPRE_Int2IntSet_::iterator itr = m->find(key);
   return itr == m->end() ? NULL : &itr->second;
}

HYPRE_Int hypre_sort_unique_and_inverse_map(
  HYPRE_Int *in, HYPRE_Int len, HYPRE_Int **out, HYPRE_Int2Int **inverse_map)
{
   if (len == 0)
   {
      *inverse_map = NULL;
      return 0;
   }

   HYPRE_Int *temp = hypre_TAlloc(HYPRE_Int, len);
   HYPRE_Int *duplicate_eliminated;
   HYPRE_Int new_len = hypre_merge_sort_unique2(in, temp, len, &duplicate_eliminated);
   *inverse_map = hypre_Int2IntCreate();
   HYPRE_Int i;
   for (i = 0; i < new_len; i++)
   {
      hypre_Int2IntInsert(*inverse_map, duplicate_eliminated[i], i);
   }
   if (duplicate_eliminated == in)
   {
      hypre_TFree(temp);
   }
   else
   {
      hypre_TFree(in);
   }
   *out = duplicate_eliminated;
   return new_len;
}

} // extern "C"
