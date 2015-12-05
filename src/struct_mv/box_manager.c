/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.21 $
 ***********************************************************************EHEADER*/

/*******************************************************************************
            
BoxManager:

AHB 10/06, updated 10/09 (changes to info object)

purpose::  organize arbitrary information in a spatial way

misc. notes/considerations/open questions: 

  (1) In the struct code, we want to use Box Manager instead of
  current box neighbor stuff (see Struct function
  hypre_CreateCommInfoFromStencil.  For example, to get neighbors of
  box b, we can call Intersect with a larger box than b).

  (2) will associate a Box Manager with the struct grid (implement
  under the struct grid)

  (3) will interface with the Box Manager in the struct coarsen routine 

    the coarsen routine:

    (a) get all the box manager entries from the current level,
    coarsen them, and create a new box manager for the coarse grid,
    adding the boxes via AddEntry

    (b) check the max_distance value and see if we have
        all the neighbor info we need in the current box manager.  

    (c) if (b) is no, then call GatherEntries as needed on the coarse
    box manager


    (d) call assemble for the new coarse box manager (note: if gather
    entries has not been called, then no communication is required
          
  (4) We will associate an assumed partition with the box manager
      (this will be created in the box manager assemble routine)     

  (5) We use the box manager with sstruct "on the side" as
  the boxmap is now, (at issue is modifying
  the "info" associated with an entry after the box manager has
  already been assembled through the underlying struct grid)

  (6) In SStruct we will have a separate box manager for the 
      neighbor box information

********************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 hypre_BoxManEntrySetInfo - this is not used
 *--------------------------------------------------------------------------*/

#if 0
HYPRE_Int hypre_BoxManEntrySetInfo ( hypre_BoxManEntry *entry , void *info )
{

   /* TO DO*/

   return hypre_error_flag;
   
}
#endif

/*--------------------------------------------------------------------------
  hypre_BoxManEntryGetInfo 
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManEntryGetInfo (hypre_BoxManEntry *entry , void **info_ptr )
{
   
   HYPRE_Int position = hypre_BoxManEntryPosition(entry);
   hypre_BoxManager *boxman;
   
   boxman = (hypre_BoxManager *) hypre_BoxManEntryBoxMan(entry);

   *info_ptr =  hypre_BoxManInfoObject(boxman, position);

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
  hypre_BoxManEntryGetExtents
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManEntryGetExtents ( hypre_BoxManEntry *entry , hypre_Index imin ,
                                  hypre_Index imax )
{
   

   hypre_IndexRef  entry_imin = hypre_BoxManEntryIMin(entry);
   hypre_IndexRef  entry_imax = hypre_BoxManEntryIMax(entry);

   HYPRE_Int  d;

   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(imin, d) = hypre_IndexD(entry_imin, d);
      hypre_IndexD(imax, d) = hypre_IndexD(entry_imax, d);
   }


   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
  hypre_BoxManEntryCopy

  Warning: this does not copy the position! Also the info may need to
  be copied as well.

 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManEntryCopy( hypre_BoxManEntry *fromentry ,   
                           hypre_BoxManEntry *toentry)
{
   HYPRE_Int d;
   
   hypre_Index imin;
   hypre_Index imax;

   hypre_IndexRef      toentry_imin;
   hypre_IndexRef      toentry_imax;


   /* copy extents */
   hypre_BoxManEntryGetExtents( fromentry, imin, imax );

   toentry_imin = hypre_BoxManEntryIMin(toentry);
   toentry_imax = hypre_BoxManEntryIMax(toentry);

   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(toentry_imin, d) = hypre_IndexD(imin, d);
      hypre_IndexD(toentry_imax, d) = hypre_IndexD(imax, d);
   }
  
   /* copy proc and id */
   hypre_BoxManEntryProc(toentry) =  hypre_BoxManEntryProc(fromentry);
   hypre_BoxManEntryId(toentry) = hypre_BoxManEntryId(fromentry);

   /*copy ghost */
   for (d = 0; d < 6; d++)
   {
      hypre_BoxManEntryNumGhost(toentry)[d] =  
         hypre_BoxManEntryNumGhost(fromentry)[d];
   }

  /* copy box manager pointer */
   hypre_BoxManEntryBoxMan(toentry) = hypre_BoxManEntryBoxMan(fromentry) ;

   /* position - we don't copy this! */

  /* copy list pointer */
   hypre_BoxManEntryNext(toentry) =  hypre_BoxManEntryNext(fromentry);
   


   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 hypre_BoxManSetAllGlobalKnown 
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManSetAllGlobalKnown ( hypre_BoxManager *manager , HYPRE_Int known )
{
   
   hypre_BoxManAllGlobalKnown(manager) = known;

   return hypre_error_flag;
   
}
/*--------------------------------------------------------------------------
 hypre_BoxManGetAllGlobalKnown 
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManGetAllGlobalKnown ( hypre_BoxManager *manager , HYPRE_Int *known )
{
   
   *known = hypre_BoxManAllGlobalKnown(manager);

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
 hypre_BoxManSetIsEntriesSort
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManSetIsEntriesSort ( hypre_BoxManager *manager , HYPRE_Int is_sort )
{
   
   hypre_BoxManIsEntriesSort(manager) = is_sort;

   return hypre_error_flag;
   
}
/*--------------------------------------------------------------------------
 hypre_BoxManGetIsEntriesSort
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManGetIsEntriesSort ( hypre_BoxManager *manager , HYPRE_Int *is_sort )
{
   
  *is_sort  =  hypre_BoxManIsEntriesSort(manager);

   return hypre_error_flag;
   
}
/*--------------------------------------------------------------------------
 hypre_BoxManGetGlobalIsGatherCalled - did any proc call a gather?
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManGetGlobalIsGatherCalled( hypre_BoxManager *manager, 
                                         MPI_Comm  comm, 
                                         HYPRE_Int *is_gather )
{
   HYPRE_Int loc_is_gather;
   HYPRE_Int nprocs;
   
   hypre_MPI_Comm_size(comm, &nprocs);

   loc_is_gather = hypre_BoxManIsGatherCalled(manager);

   if (nprocs > 1)  
      hypre_MPI_Allreduce(&loc_is_gather, is_gather, 1, HYPRE_MPI_INT, hypre_MPI_LOR, comm);

   else /* just one proc */
      *is_gather = loc_is_gather;

   return hypre_error_flag;

}




/*--------------------------------------------------------------------------
  hypre_BoxManGetAssumedPartition
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManGetAssumedPartition ( hypre_BoxManager *manager,  
                                      hypre_StructAssumedPart **assumed_partition )
{
   
   *assumed_partition = hypre_BoxManAssumedPartition(manager);
   

   return hypre_error_flag;
   
}
/*--------------------------------------------------------------------------
  hypre_BoxManSetAssumedPartition
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManSetAssumedPartition ( hypre_BoxManager *manager,  
                                      hypre_StructAssumedPart *assumed_partition )
{
   
   hypre_BoxManAssumedPartition(manager) = assumed_partition;

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
  hypre_BoxManSetBoundingBox
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManSetBoundingBox ( hypre_BoxManager *manager,  
                                 hypre_Box *bounding_box )
{
   
   hypre_Box* bbox = hypre_BoxManBoundingBox(manager);
   
   hypre_BoxSetExtents(bbox,  hypre_BoxIMin(bounding_box),
                       hypre_BoxIMax(bounding_box));

   return hypre_error_flag;
   
}





/*--------------------------------------------------------------------------
  hypre_BoxManSetNumGhost
*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxManSetNumGhost( hypre_BoxManager *manager, HYPRE_Int  *num_ghost )
{

  HYPRE_Int  i;
  
  for (i = 0; i < 6; i++)
  {
    hypre_BoxManNumGhost(manager)[i] = num_ghost[i];
  }

  return hypre_error_flag;

}



/*--------------------------------------------------------------------------
  hypre_BoxManDeleteMultipleEntriesAndInfo

  Delete multiple entries (and their corresponding info object) 
  from the manager.  The indices correspond to the
  ordering of the entries.  Assumes indices given in ascending order - 
  this is meant for internal use inside the Assemble routime.

 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_BoxManDeleteMultipleEntriesAndInfo( hypre_BoxManager *manager, 
                                               HYPRE_Int*  indices , HYPRE_Int num )
{
   
   HYPRE_Int  i, j, start;
   HYPRE_Int  array_size = hypre_BoxManNEntries(manager);

   HYPRE_Int  info_size = hypre_BoxManEntryInfoSize(manager);

   void *to_ptr;
   void *from_ptr;
   
   hypre_BoxManEntry  *entries  = hypre_BoxManEntries(manager);

   if (num > 0) 
   {
      start = indices[0];

      j = 0;
   
      for (i = start; (i + j) < array_size; i++)
      {
         if (j < num)
         {
            while ((i+j) == indices[j]) /* see if deleting consecutive items */
            {
               j++; /*increase the shift*/
               if (j == num) break;
            }
         }
            
         if ( (i+j) < array_size)  /* if deleting the last item then no moving */
         {
            /*copy the entry */
            hypre_BoxManEntryCopy(&entries[i+j], &entries[i]);
            
            /* change the position */
            hypre_BoxManEntryPosition(&entries[i]) = i;
            
            /* copy the info object */
            to_ptr = hypre_BoxManInfoObject(manager, i);
            from_ptr = hypre_BoxManInfoObject(manager, i+j);

            memcpy(to_ptr, from_ptr, info_size);
         }
      }

      hypre_BoxManNEntries(manager) = array_size - num;
   }

   return hypre_error_flag;
   
}


/*--------------------------------------------------------------------------
  hypre_BoxManCreate:

  Allocate and initialize the box manager structure.  

  Notes: 

  (1) max_nentries indicates how much storage you think you will need
  for adding entries with BoxManAddEntry

  (2) info_size indicates the size (in bytes) of the info object that
  will be attached to each entry in this box manager. 

  (3) we will collect the bounding box - this is used by the AP

  (4) comm is needed for later calls to addentry - also used in the assemble
     

*--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManCreate ( HYPRE_Int max_nentries , HYPRE_Int info_size, HYPRE_Int dim,
                         hypre_Box *bounding_box, MPI_Comm comm,
                         hypre_BoxManager **manager_ptr )

{
   
   hypre_BoxManager   *manager;
   hypre_Box          *bbox;
   

   HYPRE_Int  i, d;                          
   /* allocate object */
   manager = hypre_CTAlloc(hypre_BoxManager, 1);


   /* initialize */
   hypre_BoxManComm(manager) = comm;
   hypre_BoxManMaxNEntries(manager) = max_nentries;
   hypre_BoxManEntryInfoSize(manager) = info_size;
   hypre_BoxManDim(manager)  = dim;
   hypre_BoxManIsAssembled(manager) = 0;

   for (d = 0; d < 3; d++)
   {
      hypre_BoxManIndexesD(manager, d)     = NULL;
   }
   

   hypre_BoxManNEntries(manager)   = 0;
   hypre_BoxManEntries(manager)    = hypre_CTAlloc(hypre_BoxManEntry, 
                                                   max_nentries);

   hypre_BoxManInfoObjects(manager)  = NULL;
   hypre_BoxManInfoObjects(manager)  = hypre_MAlloc(max_nentries*info_size);

   hypre_BoxManIndexTable(manager) = NULL;
   
   hypre_BoxManNumProcsSort(manager)     = 0;
   hypre_BoxManIdsSort(manager)          = hypre_CTAlloc(HYPRE_Int, max_nentries);
   hypre_BoxManProcsSort(manager)        = hypre_CTAlloc(HYPRE_Int, max_nentries);
   hypre_BoxManProcsSortOffsets(manager) = NULL;

   hypre_BoxManFirstLocal(manager)      = 0;
   hypre_BoxManLocalProcOffset(manager) = 0;

   hypre_BoxManIsGatherCalled(manager)  = 0;
   hypre_BoxManGatherRegions(manager)   = hypre_BoxArrayCreate(0); 
   hypre_BoxManAllGlobalKnown(manager)  = 0;

   hypre_BoxManIsEntriesSort(manager)   = 0;

   hypre_BoxManNumMyEntries(manager) = 0;
   hypre_BoxManMyIds(manager) = NULL;
   hypre_BoxManMyEntries(manager)  = NULL;
                            
   hypre_BoxManAssumedPartition(manager) = NULL;

   hypre_BoxManMyIds(manager) = hypre_CTAlloc(HYPRE_Int, max_nentries);
   hypre_BoxManMyEntries(manager) = hypre_CTAlloc(hypre_BoxManEntry *, 
                                                  max_nentries);

   bbox =  hypre_BoxCreate();
   hypre_BoxManBoundingBox(manager) = bbox;
   hypre_BoxSetExtents(bbox,  hypre_BoxIMin(bounding_box),
                       hypre_BoxIMax(bounding_box));


   hypre_BoxManNextId(manager) = 0;
      
  /* ghost points: we choose a default that will give zero everywhere..*/
  for (i = 0; i < 6; i++)
  {
    hypre_BoxManNumGhost(manager)[i] = 0;
  }
      

  /* return */
   *manager_ptr = manager;

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
  hypre_BoxManIncSize:

  Increase storage for entries (for future calls to BoxManAddEntry).


  Notes: 

  In addition, we will dynamically allocate more memory
  if needed when a call to BoxManAddEntry is made and there is not
  enough storage available.  

 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManIncSize ( hypre_BoxManager *manager , HYPRE_Int inc_size)
{
   

   HYPRE_Int   max_nentries = hypre_BoxManMaxNEntries(manager);
   HYPRE_Int  *ids          = hypre_BoxManIdsSort(manager);
   HYPRE_Int  *procs        = hypre_BoxManProcsSort(manager);
   HYPRE_Int   info_size    = hypre_BoxManEntryInfoSize(manager);
   
   void *info         = hypre_BoxManInfoObjects(manager);

   hypre_BoxManEntry  *entries = hypre_BoxManEntries(manager);

   /* increase size */
   max_nentries += inc_size;

   entries = hypre_TReAlloc(entries, hypre_BoxManEntry, max_nentries);
   ids = hypre_TReAlloc(ids, HYPRE_Int, max_nentries);
   procs =  hypre_TReAlloc(procs, HYPRE_Int, max_nentries);
   info = hypre_ReAlloc(info, max_nentries*info_size);
  

   /* update manager */
   hypre_BoxManMaxNEntries(manager) = max_nentries;
   hypre_BoxManEntries(manager)     = entries;
   hypre_BoxManIdsSort(manager)     = ids;
   hypre_BoxManProcsSort(manager)   = procs;
   hypre_BoxManInfoObjects(manager) = info;

   /* my ids temporary structure (destroyed in assemble) */
   {
      HYPRE_Int *my_ids = hypre_BoxManMyIds(manager);
      hypre_BoxManEntry  **my_entries = hypre_BoxManMyEntries(manager);
            
      my_ids = hypre_TReAlloc(my_ids, HYPRE_Int, max_nentries);

      my_entries = hypre_TReAlloc(my_entries, hypre_BoxManEntry *, max_nentries);
   
      hypre_BoxManMyIds(manager) = my_ids;
      hypre_BoxManMyEntries(manager) = my_entries;
   }
   


   return hypre_error_flag;
   
}



/*--------------------------------------------------------------------------
  hypre_BoxManDestroy:
  
  De-allocate the box manager structure.

 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManDestroy ( hypre_BoxManager *manager )

{
   HYPRE_Int d;

   if (manager)
   {

      for (d = 0; d < 3; d++)
      {
         hypre_TFree(hypre_BoxManIndexesD(manager, d));
      }

      hypre_TFree(hypre_BoxManEntries(manager));

      hypre_Free(hypre_BoxManInfoObjects(manager));
      
      hypre_TFree(hypre_BoxManIndexTable(manager));

      
      hypre_TFree(hypre_BoxManIdsSort(manager));
      hypre_TFree(hypre_BoxManProcsSort(manager));
      hypre_TFree(hypre_BoxManProcsSortOffsets(manager));
      
      hypre_BoxArrayDestroy(hypre_BoxManGatherRegions(manager));

      hypre_TFree(hypre_BoxManMyIds(manager));
      hypre_TFree(hypre_BoxManMyEntries(manager));

      hypre_StructAssumedPartitionDestroy(hypre_BoxManAssumedPartition(manager));

      hypre_BoxDestroy(hypre_BoxManBoundingBox(manager));

      hypre_TFree(manager);
   }


   return hypre_error_flag;
   
}


/*--------------------------------------------------------------------------
  hypre_BoxManAddEntry:

  Add a box (entry) to the box manager. Each entry is given a 
  unique id (proc_id, box_id).  Need to assemble after adding entries.

  Notes:

  (1) The id assigned may be any integer - though since (proc_id,
  box_id) is unique, duplicates will be eliminated in the assemble.


  (2) If there is not enough storage available for this entry, then
  increase the amount automatically

  (3) Only add entries whose boxes have non-zero volume.


  (4) The info object will be copied (according to the info size given in 
      the create) to storage within the box manager.

  (5) If the id passed in is negative (user doesn't care what it is) ,
  then use the next_id stored in the box manager to assign the id

*--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManAddEntry( hypre_BoxManager *manager , hypre_Index imin , 
                          hypre_Index imax , HYPRE_Int proc_id, HYPRE_Int box_id, 
                          void *info )

{
   HYPRE_Int           myid;
   HYPRE_Int           nentries = hypre_BoxManNEntries(manager);
   HYPRE_Int           info_size = hypre_BoxManEntryInfoSize(manager);

   hypre_BoxManEntry  *entries  = hypre_BoxManEntries(manager);
   hypre_BoxManEntry  *entry;
 
   hypre_IndexRef      entry_imin;
   hypre_IndexRef      entry_imax;
 
   HYPRE_Int           d;
   HYPRE_Int           *num_ghost = hypre_BoxManNumGhost(manager);  
   HYPRE_Int           volume;
   
   HYPRE_Int           id;
   

   hypre_Box           *box;
 
   /* can only use before assembling */
   if (hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   
  /* check to see if we have a non-zero box volume (only add if non-zero) */ 
   box = hypre_BoxCreate();
   hypre_BoxSetExtents( box, imin, imax );
   volume = hypre_BoxVolume(box);
   hypre_BoxDestroy(box);
   
   if (volume) 
   {
      
      hypre_MPI_Comm_rank(hypre_BoxManComm(manager), &myid );
      
      /* check to make sure that there is enough storage available
         for this new entry - if not add space for 10 more*/
      
      if (nentries + 1 > hypre_BoxManMaxNEntries(manager))
      {
         hypre_BoxManIncSize( manager, 10);
         
         entries  = hypre_BoxManEntries(manager);
      }
      
      /* we add this to the end entry list - get pointer to location*/
      entry = &entries[nentries];
      entry_imin = hypre_BoxManEntryIMin(entry);
      entry_imax = hypre_BoxManEntryIMax(entry);
      
      /* copy information into entry */
      for (d = 0; d < 3; d++)
      {
         hypre_IndexD(entry_imin, d) = hypre_IndexD(imin, d);
         hypre_IndexD(entry_imax, d) = hypre_IndexD(imax, d);
      }
      
      /* set the processor */
      hypre_BoxManEntryProc(entry) = proc_id;

      /* set the id */
      if (box_id >= 0 )
         id = box_id;
      else /* negative means use id from box manager */
      {
         id = hypre_BoxManNextId(manager);
         /* increment fir next time */
         hypre_BoxManNextId(manager) = id + 1;
      }
      
      hypre_BoxManEntryId(entry) = id;

      /* this is the current position in the entries array */
      hypre_BoxManEntryPosition(entry) = nentries; 


      /*this associates it with the box manager */
      hypre_BoxManEntryBoxMan(entry) = (void *) manager;

      /* copy the info object */
      {
           void *index_ptr;

           /*point in the info array */
           index_ptr =  hypre_BoxManInfoObject(manager, nentries);
           memcpy(index_ptr, info, info_size);
      }
            
      
      /* inherit and inject the numghost from manager into the entry (as
       * in boxmap) */
      for (d = 0; d < 6; d++)
      {
         hypre_BoxManEntryNumGhost(entry)[d] = num_ghost[d];
      }
      hypre_BoxManEntryNext(entry)= NULL;
      
      /* add proc and id to procs_sort and ids_sort array */
      hypre_BoxManProcsSort(manager)[nentries] = proc_id;
      hypre_BoxManIdsSort(manager)[nentries] = id;
      
      
      /* here we need to keep track of my entries separately just to improve
         speed at the beginning of the assemble - then this gets deleted when
         the entries are sorted. */
      
      if (proc_id == myid)
      {
         HYPRE_Int *my_ids =   hypre_BoxManMyIds(manager);
         hypre_BoxManEntry **my_entries = hypre_BoxManMyEntries(manager);
         HYPRE_Int num_my_entries = hypre_BoxManNumMyEntries(manager);
         
         my_ids[num_my_entries] = id;
         my_entries[num_my_entries] = &entries[nentries];
         num_my_entries++;
         
         hypre_BoxManNumMyEntries(manager) = num_my_entries;
      }
   

      /* increment number of entries */
      hypre_BoxManNEntries(manager) = nentries + 1;
   
   } /* end of  vol > 0 */
   

   return hypre_error_flag;
   
}


/*--------------------------------------------------------------------------
  hypre_BoxManGetEntry:

  Given an id: (proc_id, box_id), return a pointer to the box entry.  

  Notes: 

  (1) Use of this is generally to get back something that has been
  added by the above function.  If no entry is found, an error is returned.

  (2) This functionality will replace that previously provided by
  hypre_BoxManFindBoxProcEntry.

  (3) Need to store entry information such that this information is
  easily found. (During the assemble, we will sort on proc_id, then
  box_id, and provide a pointer to the entries.  Then we can do a
  search into the proc_id, and then into the box_id.)


*--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManGetEntry( hypre_BoxManager *manager , HYPRE_Int proc, HYPRE_Int id, 
                          hypre_BoxManEntry **entry_ptr )

{
   

   /* find proc_id in procs array.  then find id in ids array, then grab
      the corresponding entry */
  
   hypre_BoxManEntry *entry;

   HYPRE_Int  myid;
   HYPRE_Int  i, offset;
   HYPRE_Int  start, finish;
   HYPRE_Int  location;
   HYPRE_Int  first_local  = hypre_BoxManFirstLocal(manager);
   HYPRE_Int *procs_sort   = hypre_BoxManProcsSort(manager);
   HYPRE_Int *ids_sort     = hypre_BoxManIdsSort(manager);
   HYPRE_Int  nentries     = hypre_BoxManNEntries(manager);
   HYPRE_Int  num_proc     = hypre_BoxManNumProcsSort(manager);
   HYPRE_Int *proc_offsets =  hypre_BoxManProcsSortOffsets(manager);
  


  /* can only use after assembling */
   if (!hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_MPI_Comm_rank(hypre_BoxManComm(manager), &myid );

   if (nentries) 
   {

      /* check to see if it is the local id first - this will be the
       * case most of the time (currently it is only used in this 
       manner)*/
      if (proc == myid)
      {
         start = first_local;
         if (start >= 0 )
         {
            finish =  proc_offsets[hypre_BoxManLocalProcOffset(manager)+1];
         }
      }
      
      else /* otherwise find proc (TO DO: just have procs_sort not
            contain duplicates - then we could do a regular binary search
            (though this list is probably short)- this has to be changed in assemble,
            then also memory management in addentry - but currently this 
             is not necessary because proc = myid for all current hypre calls)*/
      {
         start = -1;
         for (i = 0; i< num_proc; i++)
         {
            offset = proc_offsets[i];
            if (proc == procs_sort[offset])
            {
               start = offset;
               finish = proc_offsets[i+1];
               break;
            }
         }
      }
      if (start >= 0 )
      {
         /* now look for the id - returns -1 if not found*/
         location = hypre_BinarySearch(&ids_sort[start], id, finish-start);
      }
      else
      {
         location = -1;
      }
      
   }
   else
   {
      location = -1;
   }
   

   if (location >= 0 )
   {
      /* this location is relative to where we started searching - so
       * fix if non-negative */
      location += start;
      /* now grab entry */ 
      entry =  &hypre_BoxManEntries(manager)[location];
   }
   else
      entry = NULL;
   
   *entry_ptr = entry;

   return hypre_error_flag;
   
}


/*--------------------------------------------------------------------------
  hypre_BoxManGetAllEntries

  Return a list of all of the entries in the box manager (and the
  number of entries). These are sorted by (proc, id) pairs.

  11/06 - changed to return the pointer to the boxman entries rather
  than a copy of the array (so calling code should not free this array!)

*--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManGetAllEntries( hypre_BoxManager *manager , HYPRE_Int *num_entries, 
                               hypre_BoxManEntry **entries)

{
  
 

   /* can only use after assembling */
   if (!hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

  
 
   /* return */
   *num_entries = hypre_BoxManNEntries(manager);
   *entries =  hypre_BoxManEntries(manager);

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
  hypre_BoxManGetAllEntriesBoxes

  Return a list of all of the boxes ONLY in the entries in the box manager.

  Notes: Should have already created the box array;


  TO DO: (?) Might want to just store the array of boxes seperate from the
  entries array so we don't have to create the array everytime this
  function is called.  (may be called quite a bit in some sstruct
  apps)

*--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManGetAllEntriesBoxes( hypre_BoxManager *manager, 
                                    hypre_BoxArray *boxes)

{
   

   hypre_BoxManEntry entry;
   
   HYPRE_Int          i, nentries;
   hypre_Index       ilower, iupper;

   hypre_BoxManEntry  *boxman_entries  = hypre_BoxManEntries(manager);

   
  /* can only use after assembling */
   if (!hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }


   /* set array size  */
   nentries = hypre_BoxManNEntries(manager);

   hypre_BoxArraySetSize(boxes, nentries);
   
   for (i= 0; i< nentries; i++)
   {
      entry = boxman_entries[i];
      hypre_BoxManEntryGetExtents(&entry, ilower, iupper);
      hypre_BoxSetExtents(hypre_BoxArrayBox(boxes,i), ilower, iupper);
   }

   /* return */

   return hypre_error_flag;
   
}
/*--------------------------------------------------------------------------
  hypre_BoxManGetLocalEntriesBoxes

  Return a list of all of the boxes ONLY in the entries in the box manager
  that belong to the calling processor.

  Notes: Should have already created the box array;


*--------------------------------------------------------------------------*/


HYPRE_Int hypre_BoxManGetLocalEntriesBoxes( hypre_BoxManager *manager, 
                                        hypre_BoxArray *boxes)

{
   

   hypre_BoxManEntry entry;
   
   HYPRE_Int          i;

   hypre_Index        ilower, iupper;

   HYPRE_Int  start = hypre_BoxManFirstLocal(manager);
   HYPRE_Int  finish;
   HYPRE_Int  num_my_entries = hypre_BoxManNumMyEntries(manager);

   hypre_BoxManEntry  *boxman_entries  = hypre_BoxManEntries(manager);

   HYPRE_Int *offsets = hypre_BoxManProcsSortOffsets(manager);
   
   
  /* can only use after assembling */
   if (!hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* set array size  */
   hypre_BoxArraySetSize(boxes, num_my_entries);

   finish =  offsets[hypre_BoxManLocalProcOffset(manager)+1];

   if ( num_my_entries && ((finish - start) != num_my_entries))
   {
      hypre_printf("error in GetLocalEntriesBoxes!\n");
   }

   for (i= 0; i< num_my_entries; i++)
   {
      entry = boxman_entries[start + i];
      hypre_BoxManEntryGetExtents(&entry, ilower, iupper);
      hypre_BoxSetExtents(hypre_BoxArrayBox(boxes,i), ilower, iupper);
   }


   /* return */

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
 *  Get the boxes and the proc ids. The input procs array should be NULL.
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_BoxManGetAllEntriesBoxesProc( hypre_BoxManager *manager,
                                        hypre_BoxArray *boxes,
                                        HYPRE_Int      *procs)
                                                                                                                                                
{
                                                                                                                                                
                                                                                                                                                
   hypre_BoxManEntry entry;
                                                                                                                                                
   HYPRE_Int          i, nentries;
   hypre_Index        ilower, iupper;
                                                                                                                                                
   hypre_BoxManEntry  *boxman_entries  = hypre_BoxManEntries(manager);
                                                                                                                                                
                                                                                                                                                
  /* can only use after assembling */
   if (!hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
                                                                                                                                                
                                                                                                                                                
   /* set array size  */
   nentries = hypre_BoxManNEntries(manager);
                                                                                                                                                
   hypre_BoxArraySetSize(boxes, nentries);
   procs= hypre_TAlloc(HYPRE_Int, nentries);
                                                                                                                                                
   for (i= 0; i< nentries; i++)
   {
      entry = boxman_entries[i];
      hypre_BoxManEntryGetExtents(&entry, ilower, iupper);
      hypre_BoxSetExtents(hypre_BoxArrayBox(boxes,i), ilower, iupper);
      procs[i]= hypre_BoxManEntryProc(&entry);
   }
                                                                                                                                                
   /* return */
                                                                                                                                                
   return hypre_error_flag;
                                                                                                                                                
}

/*--------------------------------------------------------------------------
 hypre_BoxManGatherEntries:

 All global entries that lie within the boxes supplied to this
 function are gathered from other processors during the assemble and
 stored in a processor's local box manager.  Multiple calls may be
 made to this function. The box extents supplied here are not retained
 after the assemble. 


 Note: 

 (1) This affects whether or not calls to BoxManIntersect() can be
 answered correctly.  In other words, the user needs to anticipate the
 areas of the grid where BoxManIntersect() calls will be made, and
 make sure that information has been collected.

 (2) when this is called, the boolean "is_gather_entries" is set and
 the box is added to gather_regions array.

 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManGatherEntries(hypre_BoxManager *manager , hypre_Index imin , 
                               hypre_Index imax )
{
   
   hypre_Box *box;

   hypre_BoxArray  *gather_regions;
   

   /* can only use before assembling */
   if (hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* initialize */
   hypre_BoxManIsGatherCalled(manager) = 1;
   gather_regions = hypre_BoxManGatherRegions(manager);
   
   
   /* add the box to the gather region array */
   box = hypre_BoxCreate();
   hypre_BoxSetExtents( box, imin, imax );
   hypre_AppendBox( box, gather_regions); /* this is a copy */
   

   /* clean up */
   hypre_BoxDestroy(box);
   hypre_BoxManGatherRegions(manager) = gather_regions; /* may have been 
                                                           a re-alloc */

   return hypre_error_flag;
   
}


/*--------------------------------------------------------------------------
  hypre_BoxManAssemble:

  In the assemble, we populate the local box manager with global box
  information to be used by calls to BoxManIntersect().  Global box
  information is gathered that corresponds to the regions input by calls
  to hypre_BoxManGatherEntries().

  Notes: 


  (1) In the assumed partition (AP) case, the boxes gathered are those
  that correspond to boxes living in the assumed partition regions
  that intersect the regions input to hypre_BoxManGatherEntries().
  (We will have to check for duplicates here as a box can be in more
  than one AP.)  

  (2) If a box is gathered from a neighbor processor, then all the boxes
  from that neighbor processor are retrieved.  So we can always assume that
  have all the local information from neighbor processors.

  (3) If hypre_BoxManGatherEntries() has *not* been called, then only
  the box information provided via calls to hypre_BoxManAddEntry will
  be in the box manager.  (There is a global communication to check if
  GatherEntires has been called on any processor).  In the non-AP
  case, if GatherEntries is called on *any* processor, then all
  processors get *all* boxes (via allgatherv).
 
   (Don't call gather entries if all is known already)
 
  (4) Need to check for duplicate boxes (and eliminate) - based on
  pair (proc_id, box_id).  Also sort this identifier pair so that
  GetEntry calls can be made more easily.

  (5) ****TO DO****Particularly in the AP case, might want to think
  about a "smart" algorithm to decide whether point-to-point
  communications or an AllGather is the best way to collect the needed
  entries resulting from calls to GatherEntries().  If this was done
  well, then the AP and non-AP would not have to be treated
  separately at all!


  **Assumptions: 

  1. A processor has used "add entry" to put all of the boxes
  that it owns into its box manager

  2. The assemble routine is only called once for a box manager (i.e., you
  don't assemble, then add more entries and then assemble again)

 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManAssemble ( hypre_BoxManager *manager)

{
   
   HYPRE_Int  myid, nprocs;
   HYPRE_Int  is_gather, global_is_gather;
   HYPRE_Int  nentries;
   HYPRE_Int *procs_sort, *ids_sort;
   HYPRE_Int  i,j, k;

   HYPRE_Int need_to_sort = 1; /* default it to sort */
   HYPRE_Int short_sort = 0; /*do abreviated sort */
   
   HYPRE_Int  non_ap_gather = 1; /* default to gather w/out ap*/

   HYPRE_Int  global_num_boxes = 0;

   hypre_BoxManEntry *entries;

   hypre_BoxArray  *gather_regions;

   MPI_Comm comm = hypre_BoxManComm(manager);
    

  
   /* cannot re-assemble */
   if (hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }


   /* initilize */
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &nprocs);

   gather_regions = hypre_BoxManGatherRegions(manager);
   nentries = hypre_BoxManNEntries(manager);
   entries =  hypre_BoxManEntries(manager);
   procs_sort = hypre_BoxManProcsSort(manager);
   
   ids_sort = hypre_BoxManIdsSort(manager);

   /* do we need to gather entries -check to see if ANY processor
      called a gather?*/

   if (!hypre_BoxManAllGlobalKnown(manager))
   {
      if (nprocs > 1)
      {
         is_gather = hypre_BoxManIsGatherCalled(manager);
         hypre_MPI_Allreduce(&is_gather, &global_is_gather, 1, HYPRE_MPI_INT, hypre_MPI_LOR, comm);
      }
      else /* just one proc */
      {
         global_is_gather = 0;
         hypre_BoxManAllGlobalKnown(manager) = 1;
      }
   }
   else /* global info is known - don't call a gather even if the use has
           called gather entries */
   {
      global_is_gather = 0;
   }
   

  /* ----------------------------GATHER? ------------------------------------*/  

   if (global_is_gather)
   {
      
      HYPRE_Int *my_ids         = hypre_BoxManMyIds(manager);
      HYPRE_Int  num_my_entries = hypre_BoxManNumMyEntries(manager);

      hypre_BoxManEntry **my_entries = hypre_BoxManMyEntries(manager);


      /* Need to be able to find our own entry, given the box
         number - for the second data exchange - so do some sorting now.
         Then we can use my_ids to quickly find an entry.  This will be
         freed when the sort table is created (it's redundant at that point).
         (Note: we may be creating the AP here, so this sorting
         needs to be done at the beginning for that too).  If non-ap, then
         we want the allgatherv to already be sorted - so this takes care
         of that*/  
  
      /* my entries may already be sorted (if all entries are then my entries are
         - so check first */
      
      if (hypre_BoxManIsEntriesSort(manager)==0)
         hypre_entryqsort2(my_ids, my_entries, 0, num_my_entries - 1);
      
      /* if AP, use AP to find out who owns the data we need.  In the 
         non-AP, then just gather everything for now. */

#ifdef HYPRE_NO_GLOBAL_PARTITION
      non_ap_gather = 0;
#else      
      non_ap_gather = 1;
#endif

      /* Goal: to gather the entries from the relevant processor and
         add to the *entries array.  Also add the proc and id to the
         procs_sort and ids_sort arrays */

      if (!non_ap_gather)   /*********** AP CASE! ***********/
      {
         HYPRE_Int  size;
         HYPRE_Int *tmp_proc_ids;
         HYPRE_Int  proc_count, proc_alloc, max_proc_count;
         HYPRE_Int *proc_array;
         HYPRE_Int *ap_proc_ids;
         HYPRE_Int  count;
        
         HYPRE_Int  max_response_size;
         HYPRE_Int  non_info_size, entry_size_bytes;
         HYPRE_Int *neighbor_proc_ids = NULL;
         HYPRE_Int *response_buf_starts;
         HYPRE_Int *response_buf;
         HYPRE_Int  response_size, tmp_int;

         HYPRE_Int *send_buf = NULL;
         HYPRE_Int *send_buf_starts = NULL;
         HYPRE_Int  d, proc, id, last_id;
         HYPRE_Int *tmp_int_ptr;
         HYPRE_Int *contact_proc_ids = NULL;

         HYPRE_Int max_regions, max_refinements, ologp;
         
         HYPRE_Int  *local_boxnums;

         HYPRE_Int statbuf[3];
         HYPRE_Int send_statbuf[3];
         


         HYPRE_Int dim = hypre_BoxManDim(manager);

         void *entry_response_buf;
         void *index_ptr;

         double gamma;
         double local_volume, global_volume;
         double sendbuf2[2], recvbuf2[2];

         hypre_BoxArray *gather_regions;
         hypre_BoxArray *local_boxes;

         hypre_Box *box;

         hypre_StructAssumedPart *ap;

         hypre_DataExchangeResponse  response_obj, response_obj2;
       
         hypre_BoxManEntry *entry_ptr;         

         hypre_Index imin, imax;

         hypre_IndexRef  min_ref, max_ref;
         
         /* 1.  Create an assumed partition? (may have been added in the coarsen routine) */   

         if (hypre_BoxManAssumedPartition(manager) == NULL)
         {
            
            /* create an array of local boxes.  get the global box size/volume
               (as a double). */

            local_boxes = hypre_BoxArrayCreate(num_my_entries);
            local_boxnums = hypre_CTAlloc(HYPRE_Int, num_my_entries);
            
            local_volume = 0.0;
         
            for (i=0; i< num_my_entries; i++)
            {
               /* get entry */           
               entry_ptr = my_entries[i];
               
               /* copy box info to local_boxes */
               min_ref = hypre_BoxManEntryIMin(entry_ptr);
               max_ref =  hypre_BoxManEntryIMax(entry_ptr);
               box = hypre_BoxArrayBox(local_boxes, i);
               hypre_BoxSetExtents( box, min_ref, max_ref );
               
               /* keep box num also */
               local_boxnums[i] =   hypre_BoxManEntryId(entry_ptr);
               
               /* calculate volume */ 
               local_volume += (double) hypre_BoxVolume(box);
               
               
            }/* end of local boxes */
     
            /* get the number of global entries and the global volume */

            sendbuf2[0] = local_volume;
            sendbuf2[1] = (double) num_my_entries;
            
            hypre_MPI_Allreduce(&sendbuf2, &recvbuf2, 2, hypre_MPI_DOUBLE, hypre_MPI_SUM, comm);   
            
            global_volume = recvbuf2[0];
            global_num_boxes = (HYPRE_Int) recvbuf2[1];
            
            /* estimates for the assumed partition */ 
            d = nprocs/2;
            ologp = 0;
            while ( d > 0)
            {
               d = d/2; /* note - d is an HYPRE_Int - so this is floored */
               ologp++;
            }
            
            max_regions =  hypre_min(pow(2, ologp+1), 10*ologp);
            max_refinements = ologp;
            gamma = .6; /* percentage a region must be full to 
                           avoid refinement */  
            
            
            hypre_StructAssumedPartitionCreate(dim, 
                                               hypre_BoxManBoundingBox(manager), 
                                               global_volume, 
                                               global_num_boxes,
                                               local_boxes, local_boxnums,
                                               max_regions, max_refinements, 
                                               gamma,
                                               comm, 
                                               &ap);
         
            
            hypre_BoxManAssumedPartition(manager) = ap;

            hypre_BoxArrayDestroy(local_boxes);
            hypre_TFree(local_boxnums);
         }
         else
         {
            ap = hypre_BoxManAssumedPartition(manager);
         }
         

         /* 2.  Now go thru gather regions and find out which processor's 
            AP region they intersect  - only do the rest if we have global boxes!*/

         if (global_num_boxes)
         {
            
            gather_regions = hypre_BoxManGatherRegions(manager);

            /*allocate space to store info from one box */  
            proc_count = 0;
            proc_alloc = 8;
            proc_array = hypre_CTAlloc(HYPRE_Int, proc_alloc);
            
            
            /* probably there will mostly be one proc per box -allocate
             * space for 2*/
            size = 2*hypre_BoxArraySize(gather_regions);
            tmp_proc_ids =  hypre_CTAlloc(HYPRE_Int, size);
            count = 0;
            
            /* loop through all boxes */
            hypre_ForBoxI(i, gather_regions) 
            {
               
               hypre_StructAssumedPartitionGetProcsFromBox(ap, 
                                                           hypre_BoxArrayBox(gather_regions, i), 
                                                           &proc_count, 
                                                           &proc_alloc, 
                                                           &proc_array);
               
               if ((count + proc_count) > size)       
               {
                  size = size + proc_count 
                     + 2*(hypre_BoxArraySize(gather_regions)-i);
                  tmp_proc_ids = hypre_TReAlloc(tmp_proc_ids, HYPRE_Int, size);
               }
               for (j = 0; j< proc_count; j++)
               {
                  tmp_proc_ids[count] = proc_array[j];
                  count++;
               }
            }
            
            hypre_TFree(proc_array);     
            
            /* now get rid of redundencies in tmp_proc_ids (since a box
               can lie in more than one AP - put in ap_proc_ids*/      
            qsort0(tmp_proc_ids, 0, count-1);
            proc_count = 0;
            ap_proc_ids = hypre_CTAlloc(HYPRE_Int, count);
            
            if (count)
            {
               ap_proc_ids[0] = tmp_proc_ids[0];
               proc_count++;
            }
            for (i = 1; i < count; i++)
            {
               if (tmp_proc_ids[i]  != ap_proc_ids[proc_count-1])
               {
                  ap_proc_ids[proc_count] = tmp_proc_ids[i];
                  proc_count++; 
               }
            }
            hypre_TFree(tmp_proc_ids);
            
            /* 3.  now we have a sorted list with no duplicates in ap_proc_ids */
            /* for each of these processor ids, we need to get infomation about the
               boxes in their assumed partition region */
            
            
            /* get some stats:
               check how many point to point communications? (what is the max?) */
            /* also get the max distinct AP procs and the max # of entries) */
            send_statbuf[0] = proc_count;
            send_statbuf[1] = hypre_StructAssumedPartMyPartitionNumDistinctProcs(ap);
            send_statbuf[2] = num_my_entries;



            hypre_MPI_Allreduce(send_statbuf, statbuf, 3, HYPRE_MPI_INT, hypre_MPI_MAX, comm);   

            max_proc_count = statbuf[0];
            
            /* we do not want a single processor to do a ton of point
               to point communications (relative to the number of
               total processors -  how much is too much?*/

            /* is there a better way to figure the threshold? */ 

            /* 3/07 - take out threshold calculation - shouldn't be a
             * problem on large number of processors if box sizes are
             * relativesly similar */ 

#if 0
            threshold = hypre_min(12*ologp, nprocs);

            if ( max_proc_count >=  threshold)
            {
                  /* too many! */
               /*if (myid == 0)
                 hypre_printf("TOO BIG: check 1: max_proc_count = %d\n", max_proc_count);*/

               /* change coarse midstream!- now we will just gather everything! */
               non_ap_gather = 1;

               /*clean up from above */ 
               hypre_TFree(ap_proc_ids);
               
            }
#endif
       
            if (!non_ap_gather)
            {
               
            
               /* EXCHANGE DATA information (2 required) :
               
               if we simply return  the boxes in the AP region, we will
               not have the entry information- in particular, we will not
               have the "info" obj.  So we have to get this info by doing
               a second communication where we contact the actual owners
               of the boxes and request the entry info...So:
               
               (1) exchange #1: contact the AP processor, get the ids of the 
               procs
               with boxes in that AP region (for now we ignore the 
               box numbers - since we will get all of the entries from each 
               processor)
               
               (2) exchange #2: use this info to contact the owner
               processors and from them get the rest of the entry infomation:
               box extents, info object, etc. ***note: we will get
               all of the entries from that processor, not just the ones 
               in a particular AP region (whose box numbers we ignored above) */
               
               
               /* exchange #1 - we send nothing, and the contacted proc
                * returns all of the procs with boxes in its AP region*/
               
               /* build response object*/
               response_obj.fill_response = hypre_FillResponseBoxManAssemble1;
               response_obj.data1 = ap; /* needed to fill responses*/ 
               response_obj.data2 = NULL;           
               
               send_buf = NULL;
               send_buf_starts = hypre_CTAlloc(HYPRE_Int, proc_count + 1);
               for (i=0; i< proc_count+1; i++)
               {
                  send_buf_starts[i] = 0;  
               }
               
               response_buf = NULL; /*this and the next are allocated in
                                     * exchange data */
               response_buf_starts = NULL;
               
               /*we expect back the proc id for each
                 box owned */
               size =  sizeof(HYPRE_Int);
               
               /* this parameter needs to be the same on all processors */ 
               /* max_response_size = (global_num_boxes/nprocs)*2;*/
               /* modification - should reduce data passed */
               max_response_size = statbuf[1]; /*max num of distinct procs */
               
               hypre_DataExchangeList(proc_count, ap_proc_ids, 
                                      send_buf, send_buf_starts, 
                                      0, size, &response_obj, max_response_size, 3, 
                                      comm, (void**) &response_buf,
                                      &response_buf_starts);
               
               
               /*how many items were returned? */
               size = response_buf_starts[proc_count];
               
               /* alias the response buffer */ 
               neighbor_proc_ids = response_buf;
                                            
               /*clean up*/
               hypre_TFree(send_buf_starts);
               hypre_TFree(ap_proc_ids);
               hypre_TFree(response_buf_starts);

               
               /* create a contact list of these processors (eliminate
                * duplicate procs and also my id ) */
               
               /*first sort on proc_id  */
               qsort0(neighbor_proc_ids, 0, size-1);
               
               
               /* new contact list: */
               contact_proc_ids = hypre_CTAlloc(HYPRE_Int, size);
               proc_count = 0; /* to determine the number of unique ids) */
               
               last_id = -1;
               
               for (i=0; i< size; i++)
               {
                  if (neighbor_proc_ids[i] != last_id)
                  {
                     if (neighbor_proc_ids[i] != myid)
                     {
                        contact_proc_ids[proc_count] = neighbor_proc_ids[i];
                        last_id =  neighbor_proc_ids[i];
                        proc_count++;
                     }
                  }
               }
            
                      
               /* check to see if we have any entries from a processor before
                  contacting(if we have one entry from a processor, then we have
                  all of the entries)
                  
                  we will do we only do this if we have sorted - otherwise we
                  can't easily seach the proc list - this will be most common usage 
                  anyways*/             
               
               if (hypre_BoxManIsEntriesSort(manager) && nentries)
               {
                  
                  /*so we can eliminate duplicate contacts */                

                  HYPRE_Int new_count = 0;
                  HYPRE_Int proc_spot = 0;
                  HYPRE_Int known_id, contact_id;
                

                  /* in this case, we can do the "short sort" because
                     we will not have any duplicate proc ids */
                  short_sort = 1;

                  for (i=0; i< proc_count; i++)
                  {
                     
                     contact_id = contact_proc_ids[i];
                     
                     while (proc_spot < nentries) 
                     {
                        known_id = procs_sort[proc_spot];
                        if (contact_id > known_id)
                        {
                           proc_spot++;
                        }
                        else if (contact_id == known_id)
                        {
                           /* known already - remove from contact list -
                              so go to next i and spot*/
                           proc_spot++;
                           break;
                        }
                        else /* contact_id < known_id */
                        {
                           /* this contact_id is not known already - 
                              keep in list*/
                           contact_proc_ids[new_count] = contact_id;
                           new_count++;
                           break;
                           
                        }
                     }
                     if (proc_spot == nentries) /* keep the rest */
                     {
                        contact_proc_ids[new_count] = contact_id;
                        new_count++;
                     }
                  }
                  
                  proc_count = new_count;
                  
               }
               /* also can do the short sort if we just have boxes that are
                  ours....here we also don't need to check for duplicates */
               if (nentries == num_my_entries)
               {
                  short_sort = 1;
               }


               send_buf_starts = hypre_CTAlloc(HYPRE_Int, proc_count + 1);
               for (i=0; i< proc_count+1; i++)
               {
                  send_buf_starts[i] = 0;  
               }
               send_buf = NULL;
               
               
               /* exchange #2 - now we contact processors (send nothing) and
                  that processor needs to send us all of their local entry
                  information*/ 
               
               entry_response_buf = NULL; /*this and the next are allocated
                                           * in exchange data */
               response_buf_starts = NULL;
               
               response_obj2.fill_response = hypre_FillResponseBoxManAssemble2;
               response_obj2.data1 = manager; /* needed to fill responses*/ 
               response_obj2.data2 = NULL;   
               
               
               /*how big is an entry? extents: 6 ints, proc: 1 HYPRE_Int, id: 1
                HYPRE_Int , + copy the info: info_size is in bytes*/
               /* note: for now, we do not need to send num_ghost, position or 
                boxman - this is just generated in addentry */

               non_info_size = 8;
               entry_size_bytes = non_info_size*sizeof(HYPRE_Int) 
                  + hypre_BoxManEntryInfoSize(manager);
               
               /* modification -  use an true max_response_size 
                  (should be faster and less communication*/
               max_response_size = statbuf[2]; /* max of num_my_entries */
               
               hypre_DataExchangeList(proc_count, contact_proc_ids, 
                                      send_buf, send_buf_starts, sizeof(HYPRE_Int),
                                      entry_size_bytes, &response_obj2, 
                                      max_response_size, 4, 
                                      comm,  &entry_response_buf, 
                                      &response_buf_starts);
               
               
               /* now we can add entries that are in response_buf
                  - we check for duplicates later  */
               
               /*how many entries do we have?*/
               response_size = response_buf_starts[proc_count];  
               
               /* do we need more storage ?*/
               if (nentries + response_size >  hypre_BoxManMaxNEntries(manager))
               {
                  HYPRE_Int inc_size;
                  
                  inc_size = (response_size + nentries 
                              - hypre_BoxManMaxNEntries(manager));
                  hypre_BoxManIncSize ( manager, inc_size);
                  
                  entries =  hypre_BoxManEntries(manager);
                  procs_sort = hypre_BoxManProcsSort(manager);
                  ids_sort = hypre_BoxManIdsSort(manager);
               }
               
               index_ptr = entry_response_buf; /* point into response buf */
               for (i = 0; i < response_size; i++)
               {
                  size = sizeof(HYPRE_Int);
                  /* imin */
                  for (d = 0; d < 3; d++)
                  {
                     memcpy( &tmp_int, index_ptr, size);
                     index_ptr =  (void *) ((char *) index_ptr + size);
                     hypre_IndexD(imin, d) = tmp_int;
                  }
                  
                  /*imax */
                  for (d = 0; d < 3; d++)
                  {
                     memcpy( &tmp_int, index_ptr, size);
                     index_ptr =  (void *) ((char *) index_ptr + size);
                     hypre_IndexD(imax, d) = tmp_int;
                  }
                  
                  /* proc */  
                  tmp_int_ptr = (HYPRE_Int *) index_ptr;
                  proc = *tmp_int_ptr;
                  index_ptr =  (void *) ((char *) index_ptr + size);
                  
                  /* id */
                  tmp_int_ptr = (HYPRE_Int *) index_ptr;
                  id = *tmp_int_ptr;
                  index_ptr =  (void *) ((char *) index_ptr + size);
                  
                  /* the info object (now pointer to by index_ptr) 
                     is copied by AddEntry*/
                  hypre_BoxManAddEntry( manager , imin , 
                                        imax , proc, id, 
                                        index_ptr );
                  
                  /* start of next entry */  
                  index_ptr = (void *) ((char *) index_ptr 
                                        + hypre_BoxManEntryInfoSize(manager));
               }
               
               /* clean up from this section of code*/
               hypre_TFree(entry_response_buf);
               hypre_TFree(response_buf_starts);
               hypre_TFree(send_buf_starts);
               hypre_TFree(contact_proc_ids);
               hypre_TFree(neighbor_proc_ids); /* response_buf - aliased */
               
               
               
            } /* end of nested non_ap_gather -exchange 1*/
            
         } /* end of if global boxes */
         
      } /********** end of gathering for the AP case *****************/
      
      if (non_ap_gather) /* beginning of gathering for the non-AP case */
      {

         /* collect global data - here we will just send each
          processor's local entries id = myid (not all of the entries
          in the table). Then we will just re-create the entries array
          instead of looking for duplicates and sorting */
         HYPRE_Int  entry_size_bytes;
         HYPRE_Int  send_count, send_count_bytes;
         HYPRE_Int *displs, *recv_counts;
         HYPRE_Int  recv_buf_size, recv_buf_size_bytes;
         HYPRE_Int  d;
         HYPRE_Int  size, non_info_size, position;
         HYPRE_Int  proc, id;
         HYPRE_Int  tmp_int;
         HYPRE_Int *tmp_int_ptr;
      

         void *send_buf = NULL;
         void *recv_buf = NULL;

         hypre_BoxManEntry  *entry;

         hypre_IndexRef index;

         hypre_Index imin, imax;

         void *index_ptr;
         void *info;
         

         /*how big is an entry? extents: 6 ints, proc: 1 HYPRE_Int, id: 1
          * HYPRE_Int , 6 ints, info: info_size is in bytes*/
         /* note: for now, we do not need to send num_ghost, 
            position or boxman - this
            is just generated in addentry anyhow */
         non_info_size = 8;
         entry_size_bytes = non_info_size*sizeof(HYPRE_Int) 
            + hypre_BoxManEntryInfoSize(manager);

         /* figure out how many entries each proc has - let the group know */ 
         send_count =  num_my_entries;
         send_count_bytes = send_count*entry_size_bytes;
         recv_counts = hypre_CTAlloc(HYPRE_Int, nprocs);
      
         hypre_MPI_Allgather(&send_count_bytes, 1, HYPRE_MPI_INT,
                       recv_counts, 1, HYPRE_MPI_INT, comm);

         displs = hypre_CTAlloc(HYPRE_Int, nprocs);
         displs[0] = 0;
         recv_buf_size_bytes = recv_counts[0];
         for (i = 1; i < nprocs; i++)
         {
            displs[i] = displs[i-1] + recv_counts[i-1];
            recv_buf_size_bytes += recv_counts[i];
         }
         recv_buf_size = recv_buf_size_bytes/ entry_size_bytes;
         /* mydispls = displs[myid]/entry_size_bytes; */

         global_num_boxes = recv_buf_size;

         /* populate the send buffer with my entries (note: these are
          sorted above by increasing id */
         send_buf = hypre_MAlloc(send_count_bytes);
         recv_buf = hypre_MAlloc(recv_buf_size_bytes);

         index_ptr = send_buf; /* step through send_buf with this pointer */
         /* loop over my entries */  
         for (i=0; i < send_count; i++)
         {
            entry = my_entries[i];

            size = sizeof(HYPRE_Int);

            /* imin */
            index = hypre_BoxManEntryIMin(entry); 
            for (d = 0; d < 3; d++)
            {
               tmp_int = hypre_IndexD(index, d);
               memcpy( index_ptr, &tmp_int, size);
               index_ptr =  (void *) ((char *) index_ptr + size);
            }

            /* imax */  
            index = hypre_BoxManEntryIMax(entry);
            for (d = 0; d < 3; d++)
            {
               tmp_int = hypre_IndexD(index, d);
               memcpy( index_ptr, &tmp_int, size);
               index_ptr =  (void *) ((char *) index_ptr + size);
            }

            /* proc */
            tmp_int = hypre_BoxManEntryProc(entry);
            memcpy( index_ptr, &tmp_int, size);
            index_ptr =  (void *) ((char *) index_ptr + size);

            /* id */
            tmp_int = hypre_BoxManEntryId(entry);
            memcpy( index_ptr, &tmp_int, size);
            index_ptr =  (void *) ((char *) index_ptr + size);
   
            /*info object*/
            size = hypre_BoxManEntryInfoSize(manager);
            position = hypre_BoxManEntryPosition(entry);
            info = hypre_BoxManInfoObject(manager, position);

            memcpy(index_ptr, info, size);
            index_ptr =  (void *) ((char *) index_ptr + size);
       
         } /* end of loop over my entries */

         /* now send_buf is ready to go! */  


         hypre_MPI_Allgatherv(send_buf, send_count_bytes, hypre_MPI_BYTE,
                        recv_buf, recv_counts, displs, hypre_MPI_BYTE, comm);

         /* unpack recv_buf into entries - let's just unpack them all
          into the entries table - this way they will already be
          sorted - so we set nentries to zero so that add entries
          starts at the beginning (i.e., we are deleting the current
          entries and re-creating)*/ 
 
         if (recv_buf_size > hypre_BoxManMaxNEntries(manager))
         {
            HYPRE_Int inc_size;
         
            inc_size = (recv_buf_size - hypre_BoxManMaxNEntries(manager));
            hypre_BoxManIncSize ( manager, inc_size);
            
            nentries = hypre_BoxManNEntries(manager);
            entries =  hypre_BoxManEntries(manager);
            procs_sort = hypre_BoxManProcsSort(manager);
            ids_sort = hypre_BoxManIdsSort(manager);
         }

         /* now "empty" the entries array */
         hypre_BoxManNEntries(manager) = 0;
         hypre_BoxManNumMyEntries(manager) = 0;
         

         /* point into recv buf and then unpack */
         index_ptr = recv_buf;
         for (i = 0; i < recv_buf_size; i++)
         {
            
            size = sizeof(HYPRE_Int);
            /* imin */
            for (d = 0; d < 3; d++)
            {
               memcpy( &tmp_int, index_ptr, size);
               index_ptr =  (void *) ((char *) index_ptr + size);
               hypre_IndexD(imin, d) = tmp_int;
            }
         
            /*imax */
            for (d = 0; d < 3; d++)
            {
               memcpy( &tmp_int, index_ptr, size);
               index_ptr =  (void *) ((char *) index_ptr + size);
               hypre_IndexD(imax, d) = tmp_int;
            }

            /* proc */  
            tmp_int_ptr = (HYPRE_Int *) index_ptr;
            proc = *tmp_int_ptr;
            index_ptr =  (void *) ((char *) index_ptr + size);

            /* id */
            tmp_int_ptr = (HYPRE_Int *) index_ptr;
            id = *tmp_int_ptr;
            index_ptr =  (void *) ((char *) index_ptr + size);
            
            /* info is copied by AddEntry and index_ptr is at info */            
            hypre_BoxManAddEntry( manager , imin , 
                                  imax , proc, id, 
                                  index_ptr );

            /* start of next entry */  
            index_ptr = (void *) ((char *) index_ptr + 
                                  hypre_BoxManEntryInfoSize(manager));
         }
         
         hypre_BoxManAllGlobalKnown(manager) = 1;
         
         hypre_TFree(send_buf);
         hypre_TFree(recv_buf);
         hypre_TFree(recv_counts);
         hypre_TFree(displs);
         
         /* now the entries and procs_sort and ids_sort are already
            sorted */
         need_to_sort = 0;
         hypre_BoxManIsEntriesSort(manager) = 1;
         
   
      } /********* end of non-AP gather *****************/


   }/* end of if (gather entries) for both AP and non-AP */
   else
   {
      /* no gather - so check to see if the entries have been sorted by the user -
         if so we don't need to sort! */
      if  (hypre_BoxManIsEntriesSort(manager))     
         need_to_sort = 0;
      
   }
   

   /*we don't need special access to my entries anymore - because we will
     create the sort table */

   hypre_TFree(hypre_BoxManMyIds(manager));
   hypre_TFree(hypre_BoxManMyEntries(manager));
   hypre_BoxManMyIds(manager) = NULL;
   hypre_BoxManMyEntries(manager) = NULL;

 


    /* -----------------------SORT--------------------------------------*/

   /* now everything we need is in entries, also ids and procs have *
    been added to procs_sort and ids_sort, but possibly not
    sorted. (check need_to_sort flag).  If sorted already, then
    duplicates have been removed. Also there may not be any duplicates
     in the AP case if a duplicate proc check was done (depends on if
     current entry info was sorted)*/

   /* check for and remove duplicate boxes - based on (proc, id) */
   /* at the same time sort the procs_sort and ids_sort and
    * then sort the entries*/   
   {
      HYPRE_Int *order_index = NULL;
      HYPRE_Int *delete_array = NULL;
      HYPRE_Int  tmp_id, start, index;
      HYPRE_Int  first_local;
      HYPRE_Int  num_procs_sort;
      HYPRE_Int *proc_offsets;
      HYPRE_Int  myoffset;
      HYPRE_Int size;

      hypre_BoxManEntry  *new_entries;
      
      /* (TO DO): if we are sorting after the ap gather, then the box ids may
         already be sorted within processor number (depends on if the check 
         for contacting duplicate processors was performed....if so, then
         there may be a faster way to sort the proc ids and not mess up the
         already sorted box ids - also there will not be any duplicates )*/


      /* initial... */
      nentries = hypre_BoxManNEntries(manager);
      entries =  hypre_BoxManEntries(manager);
 
      /* these are negative if a proc does not have any local entries
         in the manager */
      first_local = -1;
      myoffset = -1;
      
      if (need_to_sort)
      {

#if 0
         /* TO DO:  add code for the "short sort"  - which is don't check for duplicates and the boxids
         are already sorted within each processor id - but the proc ids are not sorted */

         if (short_sort)
         {
            /TO DO: write this */
         }
         else
         {
            
            /*stuff below */   
         }
#endif         

         order_index = hypre_CTAlloc(HYPRE_Int, nentries);
         delete_array =  hypre_CTAlloc(HYPRE_Int, nentries);
         index = 0;
               
         for (i=0; i< nentries; i++)
         {
            order_index[i] = i;
         }
         /* sort by proc_id */ 
         hypre_qsort3i(procs_sort, ids_sort, order_index, 0, nentries-1);
         num_procs_sort = 0;
         /* get first id */
         if (nentries)
         {
            tmp_id = procs_sort[0];
            num_procs_sort++;
         }
         
         /* now sort on ids within each processor number*/
         start = 0;
         for (i=1; i< nentries; i++)
         {
            if (procs_sort[i] != tmp_id) 
            {
               hypre_qsort2i(ids_sort, order_index, start, i-1);
               /*now find duplicate ids */ 
               for (j=start+1; j< i; j++)
               {
                  if (ids_sort[j] == ids_sort[j-1])
                  {
                     delete_array[index++] = j;
                  }
               }
               /* update start and tmp_id */  
               start = i;
               tmp_id = procs_sort[i];
               num_procs_sort++; 
            }
         }
         /* final sort and purge (the last group doesn't 
            get caught in the above loop) */
         if (nentries)
         {
            hypre_qsort2i(ids_sort, order_index, start, nentries-1);
            /*now find duplicate boxnums */ 
            for (j=start+1; j<nentries; j++)
            {
               if (ids_sort[j] == ids_sort[j-1])
               {
                  delete_array[index++] = j;
               }
            }
         }
         /* now index = the number to delete (in delete_array) */     
         
       

         if (index)
         {
            
            /* now delete from sort procs and sort ids -use delete_array
               because these have already been sorted.  also delete from
               order_index */
            start = delete_array[0];
            j = 0;
            for (i = start; (i + j) < nentries; i++)
            {
               if (j < index)
               {
                  while ((i+j) == delete_array[j]) /* see if deleting
                                                    * consec. items */
                  {
                     j++; /*increase the shift*/
                     if (j == index) break;
                  }
               }
               if ((i+j) < nentries) /* if deleting the last item then no moving */
               {
                  ids_sort[i] = ids_sort[i+j];
                  procs_sort[i] =  procs_sort[i+j];
                  order_index[i] = order_index[i+j];
               }
               
            }
         }
         
         
         /***create the new sorted entries and info arrays  - delete the old one****/
         {
            HYPRE_Int position;
            HYPRE_Int info_size = hypre_BoxManEntryInfoSize(manager);
            
            void *index_ptr;
            void *new_info;
            void *info;

            size = nentries - index;
            new_entries =  hypre_CTAlloc(hypre_BoxManEntry, size);
            
            new_info = hypre_MAlloc(size*info_size);
            index_ptr = new_info;
            
            for (i= 0; i< size; i++)
            {
               /* copy the entry */
               hypre_BoxManEntryCopy(&entries[order_index[i]], &new_entries[i]);

               /* set the new position */
               hypre_BoxManEntryPosition(&new_entries[i]) = i;
               
               /* copy the info object */
               position = hypre_BoxManEntryPosition(&entries[order_index[i]]);
               info = hypre_BoxManInfoObject(manager, position);
               
               memcpy(index_ptr, info, info_size);
               index_ptr =  (void *) ((char *) index_ptr + info_size);
               
            }
            hypre_TFree(entries);
            hypre_Free(hypre_BoxManInfoObjects(manager));

            hypre_BoxManEntries(manager) = new_entries;
            hypre_BoxManMaxNEntries(manager) = size;
            hypre_BoxManNEntries(manager) = size;

            hypre_BoxManInfoObjects(manager) = new_info;
            
         
            nentries = hypre_BoxManNEntries(manager);
            entries = hypre_BoxManEntries(manager);
         }
         
         
      } /* end of if (need_to_sort) */

      else
      {
         /* no sorting - just get num_procs_sort by looping through
          procs_sort array*/
        
         num_procs_sort = 0;
         if (nentries > 0)
         {
            tmp_id = procs_sort[0];
            num_procs_sort++;
         }
         for (i=1; i < nentries; i++) 
         {
            if (procs_sort[i] != tmp_id)
            {
               num_procs_sort++;
               tmp_id = procs_sort[i];
            }
         }
      }

      hypre_BoxManNumProcsSort(manager) = num_procs_sort;


      /* finally, create proc_offsets  (myoffset corresponds to local id position)
         first_local is the position in entries; */
      proc_offsets = hypre_CTAlloc(HYPRE_Int, num_procs_sort + 1);
      proc_offsets[0] = 0;
      if (nentries > 0)
      {
         j=1;
         tmp_id = procs_sort[0];
         if (myid == tmp_id) 
         {
            myoffset =0;
            first_local = 0;
         }
         
         for (i=0; i < nentries; i++) 
         {
            if (procs_sort[i] != tmp_id)
            {
               if (myid == procs_sort[i]) 
               {
                  myoffset = j;
                  first_local = i;
               }
               proc_offsets[j++] = i;
               tmp_id = procs_sort[i];
               
            }
         }
         proc_offsets[j] = nentries; /* last one */
      }
    
      hypre_BoxManProcsSortOffsets(manager) = proc_offsets;
      hypre_BoxManFirstLocal(manager) = first_local;
      hypre_BoxManLocalProcOffset(manager) = myoffset;

      /* clean up from this section of code */
      hypre_TFree(delete_array);
      hypre_TFree(order_index);






   }/* end bracket for all or the sorting stuff */
   
   

#ifdef HYPRE_NO_GLOBAL_PARTITION
   {
    /* for the assumed partition case, we can check to see if all the global
      information is known (is a gather has been done) 
      - this could prevent future comm costs */

      HYPRE_Int all_known = 0;
      HYPRE_Int global_all_known;
      
      nentries = hypre_BoxManNEntries(manager);
   
      if (!hypre_BoxManAllGlobalKnown(manager) && global_is_gather)
      {
         /*if every processor has its nentries = global_num_boxes, then all is known */  
            if (global_num_boxes == nentries) all_known = 1;
    
            hypre_MPI_Allreduce(&all_known, &global_all_known, 1, HYPRE_MPI_INT, hypre_MPI_LAND, comm);
            
            hypre_BoxManAllGlobalKnown(manager) = global_all_known;
         }
         
      }
   
#endif






   /*------------------------------INDEX TABLE ---------------------------*/

   /* now build the index_table and indexes array */
   /* Note: for now we are using the same scheme as in BoxMap  */
   {
      HYPRE_Int *indexes[3];
      HYPRE_Int  size[3];
      HYPRE_Int  iminmax[2];
      HYPRE_Int  index_not_there;
      HYPRE_Int  d, e;
      HYPRE_Int  mystart, myfinish;
      HYPRE_Int  imin[3];
      HYPRE_Int  imax[3];
      HYPRE_Int  start_loop[3];
      HYPRE_Int  end_loop[3];
      HYPRE_Int  loop, range, loop_num;
      HYPRE_Int *proc_offsets;


      HYPRE_Int location, spot;
      

      hypre_BoxManEntry  **index_table;
      hypre_BoxManEntry   *entry;

      hypre_IndexRef entry_imin;
      hypre_IndexRef entry_imax;

     
      /* initial */
      nentries     = hypre_BoxManNEntries(manager);
      entries      = hypre_BoxManEntries(manager);
      proc_offsets = hypre_BoxManProcsSortOffsets(manager);


      /*------------------------------------------------------
       * Set up the indexes array and record the processor's
       * entries. This will be used in ordering the link list
       * of BoxManEntry- ones on this processor listed first.
       *------------------------------------------------------*/
      for (d = 0; d < 3; d++)
      {
         indexes[d] = hypre_CTAlloc(HYPRE_Int, 2*nentries);/* room for min and max of 
                                                        each entry in each 
                                                        dimension */
         size[d] = 0;
      }
      /* loop through each entry and get index */
      for (e = 0; e < nentries; e++)
      {
         entry  = &entries[e]; /* grab the entry - get min and max extents */
         entry_imin = hypre_BoxManEntryIMin(entry);
         entry_imax = hypre_BoxManEntryIMax(entry);

         for (d = 0; d < 3; d++) /* in each dim, check if the min and max 
                                    position is there already in the index 
                                    table */
         {
            iminmax[0] = hypre_IndexD(entry_imin, d);
            iminmax[1] = hypre_IndexD(entry_imax, d) + 1;

            for (i = 0; i < 2; i++)/* do the min then the max */
            {
               /* find the new index position in the indexes array */
               index_not_there = 1;

               if (!i)
               {
                  location = hypre_BinarySearch2(indexes[d], iminmax[i], 0, size[d]-1, &j);
                  if (location != -1) index_not_there = 0;               
               }
               else /* for max, we can start seach at min position */
               {
                  location = hypre_BinarySearch2(indexes[d], iminmax[i], j, size[d]-1, &j);
                  if (location != -1) index_not_there = 0;
               }
               
               
               /* if the index is already there, don't add it again */
               if (index_not_there)
               {
                  for (k = size[d]; k > j; k--) /* make room for new index */
                  {
                     indexes[d][k] = indexes[d][k-1];
                  }
                  indexes[d][j] = iminmax[i];
                  size[d]++; /* increase the size in that dimension */
               }
            } /* end of for min and max */
         } /* end of for each dimension of the entry */
      } /* end of for each entry loop */
      

      if (nentries) 
      {
         for (d = 0; d < 3; d++)
         {
            size[d]--;
         }
      }


      /*------------------------------------------------------
       * Set up the table - do offprocessor then on-processor
       *------------------------------------------------------*/

      /* allocate space for table */
      index_table = hypre_CTAlloc(hypre_BoxManEntry *, 
                                  (size[0] * size[1] * size[2]));
            
      /* which are my entries? (on-processor) */ 
      mystart = hypre_BoxManFirstLocal(manager);
      if (mystart >= 0 ) /*  we have local entries) because 
                             firstlocal = -1 if no local entries */
      {
         loop_num = 3;
         /* basically we have need to do the same code fragment
            repeated three times so that we can do off-proc 
            then on proc entries - this ordering is because creating 
            the linked list for overlapping boxes */


         myfinish =  proc_offsets[hypre_BoxManLocalProcOffset(manager)+1];
         /* #1 do off proc. entries - lower range */
         start_loop[0] = 0;
         end_loop[0] = mystart;
         /* #2 do off proc. entries - upper range */
         start_loop[1]= myfinish;
         end_loop[1] = nentries;
         /* #3 do ON proc. entries */
         start_loop[2]= mystart;
         end_loop[2] = myfinish;
      }
      else /* no on-proc entries */
      {
         loop_num = 1;
         start_loop[0] = 0;
         end_loop[0] = nentries;
      }
      
      for (loop = 0; loop < loop_num; loop++)
      {
         for (range = start_loop[loop]; range < end_loop[loop]; range++)
         {
            entry = &entries[range];
            entry_imin = hypre_BoxManEntryIMin(entry);
            entry_imax = hypre_BoxManEntryIMax(entry);
            
            /* find the indexes corresponding to the current box - put in 
               imin and imax */
            for (d = 0; d < 3; d++)
            {
               /* need to go to size[d] because that contains the last element */             
               location = hypre_BinarySearch2(indexes[d], hypre_IndexD(entry_imin, d), 0, size[d], &spot);
               hypre_IndexD(imin, d) = location;

               location = hypre_BinarySearch2(indexes[d], hypre_IndexD(entry_imax, d) + 1 , 0, size[d], &spot);
               hypre_IndexD(imax, d) = location;

            } /* now have imin and imax location in index array*/
            
            
            /* set up index table */
            for (k = imin[2]; k < imax[2]; k++)
            {
               for (j = imin[1]; j < imax[1]; j++)
               {
                  for (i = imin[0]; i < imax[0]; i++)
                  {
                     if (!(index_table[((k) * size[1] + j) * size[0] + i]))
                        /* no entry- add one */
                     {
                        index_table[((k) * size[1] + j) * size[0] + i] = entry;
                     }
                     else  /* already and entry there - so add to link list for 
                              BoxMapEntry- overlapping */
                     {
                        hypre_BoxManEntryNext(entry)= 
                           index_table[((k) * size[1] + j) * size[0] + i];
                        index_table[((k) * size[1] + j) * size[0] + i]= entry;
                     }
                  }
               }
            }
         } /* end of subset of entries */
      }/* end of three loops over subsets */


      /* done with the index_table! */ 
      hypre_TFree( hypre_BoxManIndexTable(manager)); /* in case this is a 
                                                        re-assemble - shouldn't be though */
      hypre_BoxManIndexTable(manager) = index_table;

      for (d = 0; d < 3; d++)
      {
         hypre_TFree(hypre_BoxManIndexesD(manager, d));
         hypre_BoxManIndexesD(manager, d) = indexes[d];
         hypre_BoxManSizeD(manager, d) = size[d];
         hypre_BoxManLastIndexD(manager, d) = 0;
      }
      
   } /* end of building index table group */
   


   /* clean up and update*/

   hypre_BoxManNEntries(manager) = nentries;
   hypre_BoxManEntries(manager) = entries;
   
   hypre_BoxManIsGatherCalled(manager) = 0;
   hypre_BoxArrayDestroy(gather_regions);
   hypre_BoxManGatherRegions(manager) =  hypre_BoxArrayCreate(0); 

   hypre_BoxManIsAssembled(manager) = 1;



   return hypre_error_flag;
   
}




/*--------------------------------------------------------------------------

  hypre_BoxManIntersect:
 
  Given a box (lower and upper indices), return a list of boxes in the
  global grid that are intersected by this box. The user must insure
  that a processor owns the correct global information to do the
  intersection. For now this is virtually the same as the box map intersect.

  Notes: 
 
  (1) This function can also be used in the way that
  hypre_BoxMapFindEntry was previously used - just pass in iupper=ilower.

  (2) return NULL for entries if none are found

 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoxManIntersect ( hypre_BoxManager *manager , hypre_Index ilower , 
                            hypre_Index iupper , 
                            hypre_BoxManEntry ***entries_ptr , 
                            HYPRE_Int *nentries_ptr )

{
   HYPRE_Int  d, i, j, k;
   HYPRE_Int  find_index_d;
   HYPRE_Int  current_index_d;
   HYPRE_Int *man_indexes_d;
   HYPRE_Int  man_index_size_d;
   HYPRE_Int  cnt, nentries;
   HYPRE_Int *ii, *jj, *kk;
   HYPRE_Int *proc_ids, *ids, *unsort;
   HYPRE_Int  tmp_id, start;

   HYPRE_Int  man_ilower[3] = {0, 0, 0};
   HYPRE_Int  man_iupper[3] = {0, 0, 0};

   hypre_BoxManEntry **entries, **all_entries;
   hypre_BoxManEntry  *entry;

   /* can only use after assembling */
   if (!hypre_BoxManIsAssembled(manager))
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   
   /* Check whether the box manager contains any entries */
   if (hypre_BoxManNEntries(manager) == 0)
   {
      *entries_ptr  = NULL;
      *nentries_ptr = 0;
      return hypre_error_flag;
   }

   /* loop through each dimension */
   for (d = 0; d < 3; d++)
   {
      man_indexes_d = hypre_BoxManIndexesD(manager, d);
      man_index_size_d    = hypre_BoxManSizeD(manager, d);

      /* -----find location of ilower[d] in  indexes-----*/
      find_index_d = hypre_IndexD(ilower, d);

      /* Start looking in place indicated by last_index stored in map */
      current_index_d = hypre_BoxManLastIndexD(manager, d);

      /* Loop downward if target index is less than current location */
      while ( (current_index_d >= 0 ) &&
              (find_index_d < man_indexes_d[current_index_d]) )
      {
         current_index_d --;
      }

      /* Loop upward if target index is greater than current location */
      while ( (current_index_d <= (man_index_size_d - 1)) &&
              (find_index_d >= man_indexes_d[current_index_d + 1]) )
      {
         current_index_d ++;
      }

      if( current_index_d > (man_index_size_d - 1) )
      {
         *entries_ptr  = NULL;
         *nentries_ptr = 0;
         return hypre_error_flag;
      }
      else
      {
         man_ilower[d] = hypre_max(current_index_d, 0);
      }

      /* -----find location of iupper[d] in  indexes-----*/
      
      find_index_d = hypre_IndexD(iupper, d);

      /* Loop upward if target index is greater than current location */
      while ( (current_index_d <= (man_index_size_d-1)) &&
              (find_index_d >= man_indexes_d[current_index_d+1]) )
      {
         current_index_d ++;
      }
      if( current_index_d < 0 )
      {
         *entries_ptr  = NULL;
         *nentries_ptr = 0;
         return hypre_error_flag;
      }
      else
      {
         man_iupper[d] = hypre_min(current_index_d, (man_index_size_d-1)) + 1;
      }

   }
   
   /*-----------------------------------------------------------------
    * If code reaches this point, then set up the entries array and
    * eliminate duplicates. To eliminate duplicates, we need to
    * compare the BoxMapEntry link lists. We accomplish this using
    * the unique (proc, id) identifier.
    *-----------------------------------------------------------------*/
   
   /* how many entries */
   nentries = ((man_iupper[0] - man_ilower[0]) *
               (man_iupper[1] - man_ilower[1]) *
               (man_iupper[2] - man_ilower[2]));
   
   ii= hypre_CTAlloc(HYPRE_Int, nentries);
   jj= hypre_CTAlloc(HYPRE_Int, nentries);
   kk= hypre_CTAlloc(HYPRE_Int, nentries);
   
   nentries = 0;
   cnt= 0;
   for (k = man_ilower[2]; k < man_iupper[2]; k++)
   {
      for (j = man_ilower[1]; j < man_iupper[1]; j++)
      {
         for (i = man_ilower[0]; i < man_iupper[0]; i++)
         {
            /* the next 3 `if' statements eliminate duplicates */
            if (k > man_ilower[2])
            {
               if ( hypre_BoxManIndexTableEntry(manager, i, j, k) ==
                    hypre_BoxManIndexTableEntry(manager, i, j, (k-1)) )
               {
                  continue;
               }
            }
            if (j > man_ilower[1])
            {
               if ( hypre_BoxManIndexTableEntry(manager, i, j, k) ==
                    hypre_BoxManIndexTableEntry(manager, i, (j-1), k) )
               {
                  continue;
               }
            }
            if (i > man_ilower[0])
            {
               if ( hypre_BoxManIndexTableEntry(manager, i, j, k) ==
                    hypre_BoxManIndexTableEntry(manager, (i-1), j, k) )
               {
                  continue;
               }
            }
            
            entry = hypre_BoxManIndexTableEntry(manager, i, j, k);
            /* Record the indices for non-empty entries and count all
             * ManEntries. */
            if (entry != NULL)
            {
               ii[nentries]= i;
               jj[nentries]= j;
               kk[nentries++]= k;
               
               while (entry)
               {
                  cnt++;
                  entry= hypre_BoxManEntryNext(entry);
               }
            }
         }
      }
   }
   
   
   /* if no link lists of BoxManEntries, Just point to the unique
    * BoxManEntries */
   if (nentries == cnt)
   {
      entries = hypre_CTAlloc(hypre_BoxManEntry *, nentries);
      for (i= 0; i< nentries; i++)
      {
         entries[i]= hypre_BoxManIndexTableEntry(manager, ii[i], jj[i], kk[i]);
      }
   }

   /* otherwise: link lists of BoxManEntries. Sorting needed to
    * eliminate duplicates. */
   else
   {
   
      unsort      = hypre_CTAlloc(HYPRE_Int, cnt);
      proc_ids    = hypre_CTAlloc(HYPRE_Int, cnt);
      ids         = hypre_CTAlloc(HYPRE_Int, cnt);
      all_entries = hypre_CTAlloc(hypre_BoxManEntry *, cnt);

      cnt = 0;
      for (i= 0; i< nentries; i++)
      {
         entry = hypre_BoxManIndexTableEntry(manager, ii[i], jj[i], kk[i]);

         while (entry)
         {
             all_entries[cnt]= entry;
             unsort[cnt]     = cnt;

             ids[cnt] = hypre_BoxManEntryId(entry);
             proc_ids[cnt++] = hypre_BoxManEntryProc(entry);

             entry= hypre_BoxManEntryNext(entry);
         }
      }
      /* now we have a list of all of the entries - just want unique ones */
      /* first sort on proc id */
      hypre_qsort3i(proc_ids, ids, unsort, 0, cnt-1);
      /* now sort on ids within each processor number*/
      tmp_id = proc_ids[0];
      start = 0;
      for (i=1; i< cnt; i++)
      {
         if (proc_ids[i] != tmp_id) 
         {
            hypre_qsort2i(ids, unsort, start, i-1);
            /* update start and tmp_id */  
            start = i;
            tmp_id = proc_ids[i];
         }
      }
      if (cnt > 1) /*sort last group */
      {
         hypre_qsort2i(ids, unsort, start, nentries-1);
      }


      /* count the unique Entries */
      nentries = 1;
      for (i = 1; i< cnt; i++)
      {
         if (!(proc_ids[i] = proc_ids[i-1] && ids[i] == ids[i-1]))
         {
            nentries++;
         }
      }
         
      entries = hypre_CTAlloc(hypre_BoxManEntry *, nentries);

      /* extract the unique Entries */
      entries[0] = all_entries[unsort[0]];
      nentries = 1;

      for (i= 1; i< cnt; i++)
      {
         if (!(proc_ids[i] = proc_ids[i-1] && ids[i] == ids[i-1]))
         {
            entries[nentries++] = all_entries[unsort[i]];
         }
      }

      hypre_TFree(unsort);
      hypre_TFree(ids);
      hypre_TFree(proc_ids);
      hypre_TFree(all_entries);
   }

   hypre_TFree(ii);
   hypre_TFree(jj);
   hypre_TFree(kk);

   /* Reset the last index in the manager */
   for (d = 0; d < 3; d++)
   {
      hypre_BoxManLastIndexD(manager, d) = man_ilower[d];
   }

   *entries_ptr  = entries;
   *nentries_ptr = nentries;

   
  

   return hypre_error_flag;
   
}


/******************************************************************************
    hypre_fillResponseBoxManAssemble1 

    contact message is null.  need to return the (proc) id of each box
    in our assumed partition.

    1/07 - just returning distinct proc ids.

 *****************************************************************************/

HYPRE_Int
hypre_FillResponseBoxManAssemble1(void *p_recv_contact_buf, 
                             HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, 
                             MPI_Comm comm, void **p_send_response_buf, 
                             HYPRE_Int *response_message_size )
{
   

   HYPRE_Int    myid, i, index;
   HYPRE_Int    size, num_boxes, num_objects;
   HYPRE_Int   *proc_ids;
   HYPRE_Int   *send_response_buf = (HYPRE_Int *) *p_send_response_buf;
    
 
   hypre_DataExchangeResponse  *response_obj = ro;  
   hypre_StructAssumedPart     *ap = response_obj->data1;  

   HYPRE_Int overhead = response_obj->send_response_overhead;

   /*initialize stuff */
   hypre_MPI_Comm_rank(comm, &myid );
  
   proc_ids =  hypre_StructAssumedPartMyPartitionProcIds(ap);

   /*we need to send back the list of all the processor ids
     for the boxes */

   /* NOTE: in the AP, boxes with the same proc id are adjacent
    (but proc ids not in any sorted order) */

   /*how many boxes do we have in the AP?*/
   num_boxes = hypre_StructAssumedPartMyPartitionIdsSize(ap);
   /*how many procs do we have in the AP?*/
   num_objects = hypre_StructAssumedPartMyPartitionNumDistinctProcs(ap);
   
   /* num_objects is then how much we need to send*/
     
   /*check storage in send_buf for adding the information */
   /* note: we are returning objects that are 1 ints in size */

   if ( response_obj->send_response_storage  < num_objects  )
   {
      response_obj->send_response_storage =  hypre_max(num_objects, 10); 
      size =  1*(response_obj->send_response_storage + overhead);
      send_response_buf = hypre_TReAlloc( send_response_buf, HYPRE_Int, 
                                          size);
      *p_send_response_buf = send_response_buf;  
   }

   /* populate send_response_buf with distinct proc ids*/
   index = 0;

   if (num_objects > 0) 
      send_response_buf[index++] = proc_ids[0];

   for (i = 1; i < num_boxes && index < num_objects; i++)
   {
      /* processor id */
      if (proc_ids[i] != proc_ids[i-1])
         send_response_buf[index++] = proc_ids[i];
   }

   /* return variables */
   *response_message_size = num_objects;
   *p_send_response_buf = send_response_buf;

   return hypre_error_flag;
   

}
/******************************************************************************
 
  hypre_fillResponseBoxManAssemble2
 
  contact message is null.  the response needs to
  be the all our entries (with id = myid).

 *****************************************************************************/

HYPRE_Int
hypre_FillResponseBoxManAssemble2(void *p_recv_contact_buf, 
                             HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, 
                             MPI_Comm comm, void **p_send_response_buf, 
                             HYPRE_Int *response_message_size )
{
   

   HYPRE_Int   myid, i, d, size, position;
   HYPRE_Int   proc_id, box_id, tmp_int;
   HYPRE_Int   entry_size_bytes;

 
   void  *send_response_buf = (void *) *p_send_response_buf;
   void  *index_ptr;
 
   hypre_BoxManEntry *entry;

   hypre_IndexRef  index;
  
   hypre_DataExchangeResponse  *response_obj = ro;  

   hypre_BoxManager     *manager = response_obj->data1;  
   
   HYPRE_Int   overhead = response_obj->send_response_overhead;

   hypre_BoxManEntry **my_entries = hypre_BoxManMyEntries(manager) ;
 
   HYPRE_Int   num_my_entries = hypre_BoxManNumMyEntries(manager);

   void *info;
   

   /*initialize stuff */
   hypre_MPI_Comm_rank(comm, &myid );

   entry_size_bytes = 8*sizeof(HYPRE_Int) + hypre_BoxManEntryInfoSize(manager);
   
   /* num_my_entries is the amount of information to send */

   /*check storage in send_buf for adding the information */
   if ( response_obj->send_response_storage  < num_my_entries  )
   {
      response_obj->send_response_storage =  num_my_entries; 
      size =  entry_size_bytes*(response_obj->send_response_storage + overhead);
      send_response_buf = hypre_ReAlloc( send_response_buf, size);
      *p_send_response_buf = send_response_buf;  
   }

   index_ptr = send_response_buf; /* step through send_buf with this pointer */


   for (i=0; i < num_my_entries; i++)
   {
      entry = my_entries[i];
      
      /*pack response buffer with information */
      
      size = sizeof(HYPRE_Int);
      /* imin */
      index = hypre_BoxManEntryIMin(entry); 
      for (d = 0; d < 3; d++)
      {
         tmp_int = hypre_IndexD(index, d);
         memcpy( index_ptr, &tmp_int, size);
         index_ptr =  (void *) ((char *) index_ptr + size);
      }
      /* imax */  
      index = hypre_BoxManEntryIMax(entry);
      for (d = 0; d < 3; d++)
      {
         tmp_int = hypre_IndexD(index, d);
         memcpy( index_ptr, &tmp_int, size);
         index_ptr =  (void *) ((char *) index_ptr + size);
      }
      /* proc */
      proc_id =  hypre_BoxManEntryProc(entry);
      memcpy( index_ptr, &proc_id, size);
      index_ptr =  (void *) ((char *) index_ptr + size);
      
      /* id */
      box_id = hypre_BoxManEntryId(entry);
      memcpy( index_ptr, &box_id, size);
      index_ptr =  (void *) ((char *) index_ptr + size);
      
      /*info*/
      size = hypre_BoxManEntryInfoSize(manager);
      position = hypre_BoxManEntryPosition(entry);
      info = hypre_BoxManInfoObject(manager, position);

      memcpy(index_ptr, info, size);

      index_ptr =  (void *) ((char *) index_ptr + size);
      
   }
   
   /* now send_response_buf is full */  

   /* return variable */
   *response_message_size = num_my_entries;
   *p_send_response_buf = send_response_buf;

   return hypre_error_flag;
   

}



/******************************************************************************

 some specialized sorting routines

 *****************************************************************************/

/* sort on HYPRE_Int i, move entry pointers ent */

void hypre_entryqsort2( HYPRE_Int *v,
                       hypre_BoxManEntry ** ent,
                       HYPRE_Int  left,
                       HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_entryswap2( v, ent, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_entryswap2(v, ent, ++last, i);
      }
   }
   hypre_entryswap2(v, ent, left, last);
   hypre_entryqsort2(v, ent, left, last-1);
   hypre_entryqsort2(v, ent, last+1, right);
}


void hypre_entryswap2(HYPRE_Int  *v,
                      hypre_BoxManEntry ** ent,
                      HYPRE_Int  i,
                      HYPRE_Int  j )
{
   HYPRE_Int temp;

   hypre_BoxManEntry *temp_e;
   
   temp = v[i];
   v[i] = v[j];
   v[j] = temp;

   temp_e = ent[i];
   ent[i] = ent[j];
   ent[j] = temp_e;
}
