#ifndef hypre_BOX_MANAGER_HEADER
#define hypre_BOX_MANAGER_HEADER


/*---------------------------------------------------------------------------
 *
 * Box Manager: organizes arbitrary information in a spatial way
 *
 *----------------------------------------------------------------------------*/


typedef struct hypre_BoxManEntry_struct
{
   hypre_Index  imin; /*extents of box */
   hypre_Index  imax;

   int proc; /*this is a two-part unique id: (proc, id) */
   int id;
   int num_ghost[6];

   void *info; 

   struct hypre_BoxManEntry_struct  *next;

} hypre_BoxManEntry;


/*-----------------------------------------------------------------------------*/

typedef struct
{

   MPI_Comm            comm;

   int                 max_nentries;  /* storage in entries allocated to this 
                                         amount */

    
   int                 is_gather_called; /* boolean to indicate  whether GatherEntries
                                            function has been called  (prior to 
                                            assemble) - may not want this (can tell
                                            by the size of gather_regions array) */
   
   hypre_BoxArray     *gather_regions;  /*this is where we collect boxes input 
                                          by calls to BoxManGatherEntries - to be 
                                          gathered in the assemble.  These are then 
                                          deleted after the assemble */
   

   int                 all_global_known; /* Boolean to say that every
                                            processor already has all
                                            of the global data for
                                            this manager (this could be
                                            acessed by a coarsening routine, 
                                            for example) */
   

   int                 entry_info_size;  /* in bytes, the (max) size of the info 
                                            object for the entries */ 

   /* storing the entries */
   int                 nentries;     /* number of entries stored */
   hypre_BoxManEntry  *entries;      /* These are the actual box manager entries */  

 

   /* for accessing an entry via (proc, id) */

   hypre_BoxManEntry  **sort_table;   /* points into *entries and is sorted 
                                               by each entry's unique two-part id: 
                                               (proc, id) */
   
   int                *procs_sort;    /* the sorted procs corresponding to entries*/
   int                *ids_sort;      /* sorted ids corresponding to the entries */
 
   int                num_procs_sort; /* number of distinct procs in *entries */
   int                *procs_sort_offsets;  /* offsets for procs into the 
                                             *entry_sort array */
   int                first_local;      /* position of local infomation */  
   int                local_proc_offset;  /*position of local information in offsets */

   /* here is the table  that organizes the entires spatially (by index)*/
   hypre_BoxManEntry **index_table; /* this points into 'entries' array  
                                            and corresponds to the index arays*/

   int                *indexes[3]; /* here we have the x,y,z indexes (ordered) 
                                      for the imin and imax
                                      of each box in the entries array*/
   int                 size[3];    /* how many indexes we have in each direction 
                                      - x,y,z */ 

   int                 last_index[3]; /* the last index used in the indexes map */

   /* extra stuff needed for AP implementation */

   int                 num_my_entries; /* number of entries with proc_id = myid */
   int                 *my_ids;        /* an array of ids corresponding to my entries */ 
   hypre_BoxManEntry   **my_entries;   /* points into *entries that are mine & corresponds to
                                          my_ids array.  This is destroyed in the assemble */
   
   hypre_StructAssumedPart *assumed_partition; /* the assumed partition object  - for now this is only
                                           us ed during the assemble (where it is created)*/
   int                   dim;           /* problem dimension (known in the grid) */

   /* ghost stuff - leave for now */

   int                num_ghost[6]; 



} hypre_BoxManager;


/*--------------------------------------------------------------------------
 * Accessor macros:  hypre_BoxMan
 *--------------------------------------------------------------------------*/

#define hypre_BoxManComm(manager)               ((manager) -> comm)

#define hypre_BoxManMaxNEntries(manager)        ((manager) -> max_nentries)

#define hypre_BoxManIsGatherCalled(manager)     ((manager) -> is_gather_called)
#define hypre_BoxManGatherRegions(manager)      ((manager) -> gather_regions)
#define hypre_BoxManAllGlobalKnown(manager)     ((manager) -> all_global_known)
#define hypre_BoxManEntryInfoSize(manager)      ((manager) -> entry_info_size)
#define hypre_BoxManNEntries(manager)           ((manager) -> nentries)
#define hypre_BoxManEntries(manager)            ((manager) -> entries)

#define hypre_BoxManSortTable(manager)          ((manager) -> sort_table)
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

#define hypre_BoxManNumGhost(manager)           ((manager) -> num_ghost)

#define hypre_BoxManIndexesD(manager, d)    hypre_BoxManIndexes(manager)[d]
#define hypre_BoxManSizeD(manager, d)       hypre_BoxManSize(manager)[d]
#define hypre_BoxManLastIndexD(manager, d)  hypre_BoxManLastIndex(manager)[d]
#define hypre_BoxManIndexTableEntry(manager, i, j, k) \
hypre_BoxManIndexTable(manager)[((k*hypre_BoxManSizeD(manager, 1) + j)*\
                           hypre_BoxManSizeD(manager, 0) + i)]




/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxManEntry
 *--------------------------------------------------------------------------*/

#define hypre_BoxManEntryIMin(entry)     ((entry) -> imin)
#define hypre_BoxManEntryIMax(entry)     ((entry) -> imax)
#define hypre_BoxManEntryProc(entry)     ((entry) -> proc)
#define hypre_BoxManEntryId(entry)       ((entry) -> id)
#define hypre_BoxManEntryInfo(entry)     ((entry) -> info)
#define hypre_BoxManEntryNumGhost(entry) ((entry) -> num_ghost)
#define hypre_BoxManEntryNext(entry)     ((entry) -> next)




/*--------------------------------------------------------------------------
 * Info objects 
 *--------------------------------------------------------------------------*/



typedef struct
{
   int  type;
   int  proc;
   int  offset;
   int  box;
   int  ghoffset;

} hypre_BoxManInfoDefault;

#define hypre_BoxManInfoDType(info)            ((info) -> type)
#define hypre_BoxManInfoDProc(info)            ((info) -> proc)
#define hypre_BoxManInfoDOffset(info)          ((info) -> offset)
#define hypre_BoxManInfoDBox(info)             ((info) -> box)
#define hypre_BoxManInfoDGhoffset(info)        ((info) -> ghoffset)


#endif
