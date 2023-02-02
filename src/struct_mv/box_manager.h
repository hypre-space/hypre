/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_BOX_MANAGER_HEADER
#define hypre_BOX_MANAGER_HEADER

/*--------------------------------------------------------------------------
 * BoxManEntry
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxManEntry_struct
{
   hypre_Index imin; /* Extents of box */
   hypre_Index imax;
   HYPRE_Int   ndim; /* Number of dimensions */

   HYPRE_Int proc; /* This is a two-part unique id: (proc, id) */
   HYPRE_Int id;
   HYPRE_Int num_ghost[2 * HYPRE_MAXDIM];

   HYPRE_Int position; /* This indicates the location of the entry in the the
                        * box manager entries array and is used for pairing with
                        * the info object (populated in addentry) */

   void *boxman; /* The owning manager (populated in addentry) */

   struct hypre_BoxManEntry_struct  *next;

} hypre_BoxManEntry;

/*---------------------------------------------------------------------------
 * Box Manager: organizes arbitrary information in a spatial way
 *----------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm            comm;

   HYPRE_Int           max_nentries; /* storage allocated for entries */

   HYPRE_Int           is_gather_called; /* Boolean to indicate whether
                                            GatherEntries function has been
                                            called (prior to assemble) - may not
                                            want this (can tell by the size of
                                            gather_regions array) */

   hypre_BoxArray     *gather_regions; /* This is where we collect boxes input
                                          by calls to BoxManGatherEntries - to
                                          be gathered in the assemble.  These
                                          are then deleted after the assemble */


   HYPRE_Int           all_global_known; /* Boolean to say that every processor
                                            already has all of the global data
                                            for this manager (this could be
                                            accessed by a coarsening routine,
                                            for example) */

   HYPRE_Int           is_entries_sort; /* Boolean to say that entries were
                                           added in sorted order (id, proc)
                                           (this could be accessed by a
                                           coarsening routine, for example) */

   HYPRE_Int           entry_info_size; /* In bytes, the (max) size of the info
                                           object for the entries */

   HYPRE_Int           is_assembled; /* Flag to indicate if the box manager has
                                        been assembled (used to control whether
                                        or not functions can be used prior to
                                        assemble) */

   /* Storing the entries */
   HYPRE_Int          nentries; /* Number of entries stored */
   hypre_BoxManEntry *entries;  /* Actual box manager entries - sorted by
                                   (proc, id) at the end of the assemble) */

   HYPRE_Int         *procs_sort; /* The sorted procs corresponding to entries */
   HYPRE_Int         *ids_sort; /* Sorted ids corresponding to the entries */

   HYPRE_Int          num_procs_sort; /* Number of distinct procs in entries */
   HYPRE_Int         *procs_sort_offsets; /* Offsets for procs into the
                                             entry_sort array */
   HYPRE_Int          first_local; /* Position of local infomation in entries */
   HYPRE_Int          local_proc_offset; /* Position of local information in
                                            offsets */

   /* Here is the table  that organizes the entries spatially (by index) */
   hypre_BoxManEntry **index_table; /* This points into 'entries' array and
                                       corresponds to the index arays */

   HYPRE_Int          *indexes[HYPRE_MAXDIM]; /* Indexes (ordered) for imin and
                                                 imax of each box in the entries
                                                 array */
   HYPRE_Int           size[HYPRE_MAXDIM]; /* How many indexes in each
                                              direction */

   HYPRE_Int           last_index[HYPRE_MAXDIM]; /* Last index used in the
                                                    indexes map */

   HYPRE_Int           num_my_entries; /* Num entries with proc_id = myid */
   HYPRE_Int          *my_ids; /* Array of ids corresponding to my entries */
   hypre_BoxManEntry **my_entries; /* Points into entries that are mine and
                                      corresponds to my_ids array.  This is
                                      destroyed in the assemble. */

   void               *info_objects; /* Array of info objects (each of size
                                        entry_info_size), managed byte-wise */

   hypre_StructAssumedPart *assumed_partition; /* The assumed partition object.
                                                  For now this is only used
                                                  during the assemble (where it
                                                  is created). */
   HYPRE_Int           ndim; /* Problem dimension (known in the grid) */

   hypre_Box          *bounding_box; /* Bounding box from associated grid */

   HYPRE_Int           next_id; /* Counter to indicate the next id that would be
                                   unique (regardless of proc id) */

   /* Ghost stuff  */
   HYPRE_Int           num_ghost[2 * HYPRE_MAXDIM];

} hypre_BoxManager;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxMan
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
#define hypre_BoxManNDim(manager)               ((manager) -> ndim)
#define hypre_BoxManBoundingBox(manager)        ((manager) -> bounding_box)

#define hypre_BoxManNextId(manager)             ((manager) -> next_id)

#define hypre_BoxManNumGhost(manager)           ((manager) -> num_ghost)

#define hypre_BoxManIndexesD(manager, d)    hypre_BoxManIndexes(manager)[d]
#define hypre_BoxManSizeD(manager, d)       hypre_BoxManSize(manager)[d]
#define hypre_BoxManLastIndexD(manager, d)  hypre_BoxManLastIndex(manager)[d]

#define hypre_BoxManInfoObject(manager, i) \
(void *) ((char *)hypre_BoxManInfoObjects(manager) + i* hypre_BoxManEntryInfoSize(manager))

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxManEntry
 *--------------------------------------------------------------------------*/

#define hypre_BoxManEntryIMin(entry)     ((entry) -> imin)
#define hypre_BoxManEntryIMax(entry)     ((entry) -> imax)
#define hypre_BoxManEntryNDim(entry)     ((entry) -> ndim)
#define hypre_BoxManEntryProc(entry)     ((entry) -> proc)
#define hypre_BoxManEntryId(entry)       ((entry) -> id)
#define hypre_BoxManEntryPosition(entry) ((entry) -> position)
#define hypre_BoxManEntryNumGhost(entry) ((entry) -> num_ghost)
#define hypre_BoxManEntryNext(entry)     ((entry) -> next)
#define hypre_BoxManEntryBoxMan(entry)   ((entry) -> boxman)

#endif
