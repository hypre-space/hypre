#include "_hypre_utilities.h"

#include <set>
#include <unordered_map>

using namespace std;

extern "C"
{

typedef struct HYPRE_IntSet
{
   set<HYPRE_Int> set;
   std::set<HYPRE_Int>::iterator itr;
} HYPRE_IntSet;

HYPRE_IntSet *hypre_IntSetCreate()
{
   return new HYPRE_IntSet;
}

void hypre_IntSetDestroy( HYPRE_IntSet *set )
{
   delete set;
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

HYPRE_Int2IntSet *hypre_Int2IntSetCreate()
{
   return (HYPRE_Int2IntSet *)(new unordered_map<HYPRE_Int, HYPRE_IntSet>());
}

void hypre_Int2IntSetDestroy( HYPRE_Int2IntSet *map )
{
   delete (unordered_map<HYPRE_Int, HYPRE_IntSet> *)map;
}

void hypre_Int2IntSetInsert( HYPRE_Int2IntSet *map, HYPRE_Int key, HYPRE_Int value )
{
   unordered_map<HYPRE_Int, HYPRE_IntSet> *m = (unordered_map<HYPRE_Int, HYPRE_IntSet> *)map;
   unordered_map<HYPRE_Int, HYPRE_IntSet>::iterator itr = m->find(key);
   if (itr == m->end())
   {
      itr = m->insert(make_pair(key, HYPRE_IntSet())).first;
   }
   itr->second.set.insert(value);
}

HYPRE_IntSet *hypre_Int2IntSetFind( HYPRE_Int2IntSet *map, HYPRE_Int key )
{
   unordered_map<HYPRE_Int, HYPRE_IntSet> *m = (unordered_map<HYPRE_Int, HYPRE_IntSet> *)map;
   unordered_map<HYPRE_Int, HYPRE_IntSet>::iterator itr = m->find(key);
   return itr == m->end() ? NULL : &itr->second;
}

} // extern "C"
