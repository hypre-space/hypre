/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/

#ifndef hypre_BOX_MANAGER_HEADER
#define hypre_BOX_MANAGER_HEADER



/*--------------------------------------------------------------------------
 * BoxManEntry
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxManEntry_struct
{
   hypre_Index  imin; /*extents of box */
   hypre_Index  imax;

   HYPRE_Int proc; /*this is a two-part unique id: (proc, id) */
   HYPRE_Int id;
   HYPRE_Int num_ghost[6];

   HYPRE_Int position; /* this indicates the location of the entry in the
                  * the box manager entries array and is used for
                  * pairing with the info object (populated in addentry) */
   
   void *boxman; /* the owning manager (populated in addentry)*/
   
   struct hypre_BoxManEntry_struct  *next;

} hypre_BoxManEntry;


/*---------------------------------------------------------------------------
 *
 * Box Manager: organizes arbitrary information in a spatial way
 *
 *----------------------------------------------------------------------------*/


typedef struct
{

   MPI_Comm            comm;

   HYPRE_Int           max_nentries;  /* storage in entries allocated to this 
                                         amount */

    
   HYPRE_Int           is_gather_called; /* boolean to indicate  whether GatherEntries
                                            function has been called  (prior to 
                                            assemble) - may not want this (can tell
                                            by the size of gather_regions array) */
   
   hypre_BoxArray     *gather_regions;  /*this is where we collect boxes input 
                                          by calls to BoxManGatherEntries - to be 
                                          gathered in the assemble.  These are then 
                                          deleted after the assemble */
   

   HYPRE_Int           all_global_known; /* Boolean to say that every
                                            processor already has all
                                            of the global data for
                                            this manager (this could be
                                            acessed by a coarsening routine, 
                                            for example) */
   
   HYPRE_Int           is_entries_sort;     /* Boolean to say that entries were 
                                            added in sorted order (id, proc)
                                            (this could be
                                            acessed by a coarsening routine, 
                                            for example) */


   HYPRE_Int           entry_info_size;  /* in bytes, the (max) size of the info 
                                            object for the entries */ 

   HYPRE_Int           is_assembled;        /* flag to indicate if the box manager has been 
                                            assembled (use to control whether or not
                                            functions can be used prior to assemble)*/
   

   /* storing the entries */
   HYPRE_Int           nentries;     /* number of entries stored */
   hypre_BoxManEntry  *entries;      /* These are the actual box manager entries - these
                                      are sorted by (proc, id) at the end of the assemble)*/  

   HYPRE_Int          *procs_sort;    /* the sorted procs corresponding to entries */
   HYPRE_Int          *ids_sort;      /* sorted ids corresponding to the entries */
 
   HYPRE_Int          num_procs_sort; /* number of distinct procs in *entries */
   HYPRE_Int          *procs_sort_offsets;  /* offsets for procs into the 
                                             *entry_sort array */
   HYPRE_Int          first_local;      /* position of local infomation in entries*/  
   HYPRE_Int          local_proc_offset;  /*position of local information in offsets */

   /* here is the table  that organizes the entries spatially (by index)*/
   hypre_BoxManEntry **index_table; /* this points into 'entries' array  
                                            and corresponds to the index arays*/

   HYPRE_Int          *indexes[3]; /* here we have the x,y,z indexes (ordered) 
                                      for the imin and imax
                                      of each box in the entries array*/
   HYPRE_Int           size[3];    /* how many indexes we have in each direction 
                                      - x,y,z */ 

   HYPRE_Int           last_index[3]; /* the last index used in the indexes map */

   HYPRE_Int           num_my_entries; /* number of entries with proc_id = myid */
   HYPRE_Int           *my_ids;        /* an array of ids corresponding to my entries */ 
   hypre_BoxManEntry   **my_entries;   /* points into *entries that are mine & corresponds to
                                          my_ids array.  This is destroyed in the assemble */
   
   void               *info_objects;    /* this is an array of info objects (of each is of 
                                         size entry_info_size) -this is managed byte-wise */ 
   

   hypre_StructAssumedPart *assumed_partition; /* the assumed partition object  - for now this is only
                                                  used during the assemble (where it is created)*/
   HYPRE_Int           dim;           /* problem dimension (known in the grid) */

   hypre_Box           *bounding_box;  /* bounding box - from associated grid */
   

   HYPRE_Int           next_id; /* counter to indicate the next id 
                                   that would be unique (regardless of proc id) */  

   /* ghost stuff  */

   HYPRE_Int          num_ghost[6]; 



} hypre_BoxManager;




/*--------------------------------------------------------------------------
 * Accessor macros:  hypre_BoxMan
 *--------------------------------------------------------------------------*/

#define hypre_BoxManComm(manager)               ((manager) -> comm)

#define hypre_BoxManMaxNEntries(manager)        ((manager) -> max_nentries)

#define hypre_BoxManIsGatherCalled(manager)     ((manager) -> is_gather_called)
#define hypre_BoxManIsEntriesSort(manager)      ((manager) -> is_entries_sort)
#define hypre_BoxManGatherRegions(manager)      ((manager) -> gather_regions)
#define hypre_BoxManAllGlobalKnown(manager)     ((manager) -> all_global_known)
#define hypre_BoxManEntryInfoSize(manager)      ((manager) -> entry_info_size)
#define hypre_BoxManNEntries(manager)           ((manager) -> nentries)
#define hypre_BoxManEntries(manager)            ((manager) -> entries)
#define hypre_BoxManInfoObjects(manager)        ((manager) -> info_objects)
#define hypre_BoxManIsAssembled(manager)        ((manager) -> is_assembled) 

#define hypre_BoxManProcsSort(manager)          ((manager) -> procs_sort)
#define hypre_BoxManIdsSort(manager)            ((manager) -> ids_sort)
#define hypre_BoxManNumProcsSort(manager)       ((manager) -> num_procs_sort)
#define hypre_BoxManProcsSortOffsets(manager)   ((manager) -> procs_sort_offsets)
#define hypre_BoxManLocalProcOffset(manager)    ((manager) -> local_proc_offset)

#define hypre_BoxManFirstLocal(manager)         ((manager) -> first_local)

#define hypre_BoxManIndexTable(manager)         ((manager) -> index_table)
#define hypre_BoxManIndexes(manager)            ((manager) -> indexes)
#define hypre_BoxManSize(manager)               ((manager) -> size)
#define hypre_BoxManLastIndex(manager)          ((manager) -> last_index)

#define hypre_BoxManNumMyEntries(manager)       ((manager) -> num_my_entries)
#define hypre_BoxManMyIds(manager)              ((manager) -> my_ids)
#define hypre_BoxManMyEntries(manager)          ((manager) -> my_entries)
#define hypre_BoxManAssumedPartition(manager)   ((manager) -> assumed_partition)
#define hypre_BoxManDim(manager)                ((manager) -> dim)
#define hypre_BoxManBoundingBox(manager)        ((manager) -> bounding_box)

#define hypre_BoxManNextId(manager)             ((manager) -> next_id)

#define hypre_BoxManNumGhost(manager)           ((manager) -> num_ghost)

#define hypre_BoxManIndexesD(manager, d)    hypre_BoxManIndexes(manager)[d]
#define hypre_BoxManSizeD(manager, d)       hypre_BoxManSize(manager)[d]
#define hypre_BoxManLastIndexD(manager, d)  hypre_BoxManLastIndex(manager)[d]
#define hypre_BoxManIndexTableEntry(manager, i, j, k) \
hypre_BoxManIndexTable(manager)[((k*hypre_BoxManSizeD(manager, 1) + j)*\
                           hypre_BoxManSizeD(manager, 0) + i)]

#define hypre_BoxManInfoObject(manager, i) \
(void *) ((char *)hypre_BoxManInfoObjects(manager) + i* hypre_BoxManEntryInfoSize(manager))



/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxManEntry
 *--------------------------------------------------------------------------*/

#define hypre_BoxManEntryIMin(entry)     ((entry) -> imin)
#define hypre_BoxManEntryIMax(entry)     ((entry) -> imax)
#define hypre_BoxManEntryProc(entry)     ((entry) -> proc)
#define hypre_BoxManEntryId(entry)       ((entry) -> id)
#define hypre_BoxManEntryPosition(entry) ((entry) -> position)
#define hypre_BoxManEntryNumGhost(entry) ((entry) -> num_ghost)
#define hypre_BoxManEntryNext(entry)     ((entry) -> next)
#define hypre_BoxManEntryBoxMan(entry)   ((entry) -> boxman)
#

#endif
