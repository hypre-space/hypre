/*******************************************************************************
            
BoxManager:

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

  (5) how will we use the box manager with sstruct?  "on the side" as
  the boxmap is now, or through the struct grid (at issue is modifying
  the "info" associated with an entry after the box manager has
  already been assembled through the underlying struct grid)

  (6) Populating the "info" in sstruct, we need to eliminate global
  storage when computing offsets (can probably use MPI_SCan as in
  parcsr_ls/par_coarse_parms.c)

  (7) In SStruct we will have a separate box manager for the 
      neighbor box information

********************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 hypre_BoxManEntrySetInfo 
 *--------------------------------------------------------------------------*/


int hypre_BoxManEntrySetInfo ( hypre_BoxManEntry *entry , void *info )
{
   
   hypre_BoxManEntryInfo(entry) = info;

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
  hypre_BoxManEntryGetInfo 
 *--------------------------------------------------------------------------*/


int hypre_BoxManEntryGetInfo ( hypre_BoxManEntry *entry , void **info_ptr )
{
   
   *info_ptr = hypre_BoxManEntryInfo(entry);

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
  hypre_BoxManEntryGetExtents
 *--------------------------------------------------------------------------*/


int hypre_BoxManEntryGetExtents ( hypre_BoxManEntry *entry , hypre_Index imin ,
                                  hypre_Index imax )
{
   

   hypre_IndexRef  entry_imin = hypre_BoxManEntryIMin(entry);
   hypre_IndexRef  entry_imax = hypre_BoxManEntryIMax(entry);

   int  d;

   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(imin, d) = hypre_IndexD(entry_imin, d);
      hypre_IndexD(imax, d) = hypre_IndexD(entry_imax, d);
   }


   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
  hypre_BoxManEntryCopy
 *--------------------------------------------------------------------------*/

int hypre_BoxManEntryCopy( hypre_BoxManEntry *fromentry ,   
                           hypre_BoxManEntry *toentry)
{
   int d;
   
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
      hypre_BoxManEntryNumGhost(toentry)[d] =  hypre_BoxManEntryNumGhost(fromentry)[d];
   }

  /* copy info */
   hypre_BoxManEntryInfo(toentry) = hypre_BoxManEntryInfo(fromentry) ;

  /* copy list pointer */
   hypre_BoxManEntryNext(toentry) =  hypre_BoxManEntryNext(fromentry);
   


   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
  hypre_BoxManDeleteMultipleEntries

  Delete multiple entries from the manager.  The indices correcpond to the
  ordering of the entires.  Assumes indices given in ascending order - 
  this is meant for internal use inside the Assemble routime.

 *--------------------------------------------------------------------------*/

int  hypre_BoxManDeleteMultipleEntries( hypre_BoxManager *manager, 
                                        int*  indices , int num )
{
   
   int  i, j, start;
   int  array_size = hypre_BoxManNEntries(manager);
 
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
            hypre_BoxManEntryCopy(&entries[i+j], &entries[i]);
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
  will be attached to each entry in this box manager. (In the future, may
  want to let the info size be stored in the entry - so specified with
  AddEntry call. Then we should adjust the ExchangeData function to
  allow for different sizes - probably a nontrivial change.)

  (3) we are *not* collecting the bounding box

  (4) comm is needed for later calls to addentry - also used in the assemble
     

*--------------------------------------------------------------------------*/

int hypre_BoxManCreate ( int max_nentries , int info_size, int dim, MPI_Comm comm,
                         hypre_BoxManager **manager_ptr )

{
   
   hypre_BoxManager   *manager;
   
   int  i, d;
                          
   /* allocate object */
   manager = hypre_CTAlloc(hypre_BoxManager, 1);


   /* initialize */
   hypre_BoxManComm(manager) = comm;
   hypre_BoxManMaxNEntries(manager) = max_nentries;
   hypre_BoxManEntryInfoSize(manager) = info_size;
   hypre_BoxManDim(manager)  = dim;
   
   for (d = 0; d < 3; d++)
   {
      hypre_BoxManIndexesD(manager, d)     = NULL;
   }
   

   hypre_BoxManNEntries(manager)   = 0;
   hypre_BoxManEntries(manager)    = hypre_CTAlloc(hypre_BoxManEntry, max_nentries);
   hypre_BoxManIndexTable(manager) = NULL;
   hypre_BoxManSortTable(manager)  = NULL;
   
   hypre_BoxManNumProcsSort(manager)     = 0;
   hypre_BoxManIdsSort(manager)          = hypre_CTAlloc(int, max_nentries);
   hypre_BoxManProcsSort(manager)        = hypre_CTAlloc(int, max_nentries);
   hypre_BoxManProcsSortOffsets(manager) = NULL;

   hypre_BoxManFirstLocal(manager) = 0;
   hypre_BoxManLocalProcOffset(manager) = 0;

   hypre_BoxManIsGatherCalled(manager)     = 0;
   hypre_BoxManGatherRegions(manager)      = hypre_BoxArrayCreate(0); 
   hypre_BoxManAllGlobalKnown(manager)     = 0;
   
   hypre_BoxManNumMyEntries(manager) = 0;
   hypre_BoxManMyIds(manager) = NULL;
   hypre_BoxManMyEntries(manager)  = NULL;
                            
   hypre_BoxManAssumedPartition(manager) = NULL;

#ifdef HYPRE_NO_GLOBAL_PARTITION
   hypre_BoxManMyIds(manager) = hypre_CTAlloc(int, max_nentries);
   hypre_BoxManMyEntries(manager) = hypre_CTAlloc(hypre_BoxManEntry *, max_nentries);
#endif

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

int hypre_BoxManIncSize ( hypre_BoxManager *manager , int inc_size)
{
   

   int   max_nentries = hypre_BoxManMaxNEntries(manager);
   int  *ids          = hypre_BoxManIdsSort(manager);
   int  *procs        = hypre_BoxManProcsSort(manager);

   hypre_BoxManEntry  *entries = hypre_BoxManEntries(manager);

   /* increase size */
   max_nentries += inc_size;
   entries = hypre_TReAlloc(entries, hypre_BoxManEntry, max_nentries);
   ids = hypre_TReAlloc(ids, int, max_nentries);
   procs =  hypre_TReAlloc(procs, int, max_nentries);



   /* update manager */
   hypre_BoxManMaxNEntries(manager) = max_nentries;
   hypre_BoxManEntries(manager)     = entries;
   hypre_BoxManIdsSort(manager)     = ids;
   hypre_BoxManProcsSort(manager)   = procs;



#ifdef HYPRE_NO_GLOBAL_PARTITION
   {
      int *my_ids = hypre_BoxManMyIds(manager);
      hypre_BoxManEntry  **my_entries = hypre_BoxManMyEntries(manager);
            
      my_ids = hypre_TReAlloc(my_ids, int, max_nentries);

      my_entries = hypre_TReAlloc(my_entries, hypre_BoxManEntry *, max_nentries);
   
      hypre_BoxManMyIds(manager) = my_ids;
      hypre_BoxManMyEntries(manager) = my_entries;
   }
   
#endif


   return hypre_error_flag;
   
}



/*--------------------------------------------------------------------------
  hypre_BoxManDestroy:
  
  De-allocate the box manager structure.

 *--------------------------------------------------------------------------*/

int hypre_BoxManDestroy ( hypre_BoxManager *manager )

{
   int d;

   if (manager)
   {

      for (d = 0; d < 3; d++)
      {
         hypre_TFree(hypre_BoxManIndexesD(manager, d));
      }

      hypre_TFree(hypre_BoxManEntries(manager));
      hypre_TFree(hypre_BoxManIndexTable(manager));
      hypre_TFree(hypre_BoxManSortTable(manager));
      
      hypre_TFree(hypre_BoxManIdsSort(manager));
      hypre_TFree(hypre_BoxManProcsSort(manager));
      hypre_TFree(hypre_BoxManProcsSortOffsets(manager));
      
      hypre_BoxArrayDestroy(hypre_BoxManGatherRegions(manager));

      hypre_TFree(hypre_BoxManMyIds(manager));
      hypre_TFree(hypre_BoxManMyEntries(manager));

      hypre_StructAssumedPartitionDestroy(hypre_BoxManAssumedPartition(manager));

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


  (2) if there is not enough storage available for this entry, then
  increase the amount automatically


*--------------------------------------------------------------------------*/

int hypre_BoxManAddEntry( hypre_BoxManager *manager , hypre_Index imin , 
                          hypre_Index imax , int proc_id, int box_id, 
                          void *info )

{
   int                 myid;
   int                 nentries = hypre_BoxManNEntries(manager);
   hypre_BoxManEntry  *entries  = hypre_BoxManEntries(manager);
   hypre_BoxManEntry  *entry;
   hypre_IndexRef      entry_imin;
   hypre_IndexRef      entry_imax;
   int                 d;
   int                 *num_ghost = hypre_BoxManNumGhost(manager);  


   MPI_Comm_rank(hypre_BoxManComm(manager), &myid );

   /* check to make sure that there is enough storage available
    for this new entry - if not add space for 5 more*/

   if (nentries + 1 > hypre_BoxManMaxNEntries(manager))
   {
      hypre_BoxManIncSize( manager, 5);
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
  
   hypre_BoxManEntryProc(entry) = proc_id;
   hypre_BoxManEntryId(entry) = box_id;
   hypre_BoxManEntryInfo(entry) = info;


   /* inherit and inject the numghost from mananger into the entry (as in boxmap) */
   for (d = 0; d < 6; d++)
   {
      hypre_BoxManEntryNumGhost(entry)[d] = num_ghost[d];
   }
   hypre_BoxManEntryNext(entry)= NULL;

   /* add proc and id to procs_sort and ids_sort array */
   hypre_BoxManProcsSort(manager)[nentries] = proc_id;
   hypre_BoxManIdsSort(manager)[nentries] = box_id;
   
  
#ifdef HYPRE_NO_GLOBAL_PARTITION
   /* here we need to keep track of my entries separately just to improve
      speed at the beginning of the assemble - then this gets deleted when
      the sort_table is created. */

   if (proc_id == myid)
   {
      int *my_ids =   hypre_BoxManMyIds(manager);
      hypre_BoxManEntry **my_entries = hypre_BoxManMyEntries(manager);
      int num_my_entries = hypre_BoxManNumMyEntries(manager);
      
      my_ids[num_my_entries] = box_id;
      my_entries[num_my_entries] = &entries[nentries];
      num_my_entries++;

      hypre_BoxManNumMyEntries(manager) = num_my_entries;


   }
   
#endif

         /* increment number of entries */
   hypre_BoxManNEntries(manager) = nentries + 1;


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

int hypre_BoxManGetEntry( hypre_BoxManager *manager , int proc, int id, 
                          hypre_BoxManEntry **entry_ptr )

{
   

   /* find proc_id in procs array.  then find id in ids array, then grab
      the corresponding entry */
  
   hypre_BoxManEntry *entry;

   int  myid;
   int  i, offset;
   int  start, finish;
   int  location;
   int  first_local  = hypre_BoxManFirstLocal(manager);
   int *procs_sort   = hypre_BoxManProcsSort(manager);
   int *ids_sort     = hypre_BoxManIdsSort(manager);
   int  nentries     = hypre_BoxManNEntries(manager);
   int  num_proc     = hypre_BoxManNumProcsSort(manager);
   int *proc_offsets =  hypre_BoxManProcsSortOffsets(manager);
  
   MPI_Comm_rank(hypre_BoxManComm(manager), &myid );

   if (nentries) 
   {


      if ( hypre_BoxManSortTable(manager) == NULL)
         hypre_error_in_arg(1);
      
      
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
            contain duplicates - then we could do a binary search and
            save on some storage - this has to be changed in assemble,
            then also memory management in addentry - but currently this 
             because proc = myid)*/
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
      /* this location is relative to where we started searching - so fix if non-negative */
      location += start;
      /* now grab entry */ 
      entry =  hypre_BoxManSortTable(manager)[location];
   }
   else
      entry = NULL;
   
   *entry_ptr = entry;

   return hypre_error_flag;
   
}


/*--------------------------------------------------------------------------
  hypre_BoxManGetAllEntries

  Return a list of all of the entries in the box manager (and the
  number of entries). These are not sorted in any particular way.

  We do not just return a pointer to the entries array in the box manager as
  we don't want the user to "mess with" the entries array - e.g., delete it :)

*--------------------------------------------------------------------------*/


int hypre_BoxManGetAllEntries( hypre_BoxManager *manager , int *num_entries, 
                               hypre_BoxManEntry ***entries_ptr)

{
   

   hypre_BoxManEntry **entries;
   int                i, nentries;
   
   hypre_BoxManEntry  *boxman_entries  = hypre_BoxManEntries(manager);


   nentries = hypre_BoxManNEntries(manager);
   entries = hypre_CTAlloc(hypre_BoxManEntry *, nentries);
   for (i= 0; i< nentries; i++)
   {
      entries[i]= &boxman_entries[i];;
   }


   /* return */
   *entries_ptr = entries;
   *num_entries = nentries;


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

int hypre_BoxManGatherEntries(hypre_BoxManager *manager , hypre_Index imin , 
                               hypre_Index imax )
{
   
   hypre_Box *box;

   hypre_BoxArray  *gather_regions;
   

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

  (2) If hypre_BoxManGatherEntries() has *not* been called, then only
  the box information provided via calls to hypre_BoxManAddEntry will
  be in the box manager.  (There is a global communication to check if
  GatherEntires has been called on any processor).  In the non-AP
  case, if GatherEntries is called on *any* processor, then all
  processors get all boxes (via allgatherv).
  
  (3) Need to check for duplicate boxes (and eliminate) - based on
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

int hypre_BoxManAssemble ( hypre_BoxManager *manager)

{
   
   int  myid, nprocs;
   int  is_gather, global_is_gather;
   int  nentries;
   int *procs_sort, *ids_sort;
   int  i,j, k;

   hypre_BoxManEntry *entries;

   hypre_BoxArray  *gather_regions;

   MPI_Comm comm = hypre_BoxManComm(manager);
   

   /* initilize */
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   gather_regions = hypre_BoxManGatherRegions(manager);
   nentries = hypre_BoxManNEntries(manager);
   entries =  hypre_BoxManEntries(manager);
   procs_sort = hypre_BoxManProcsSort(manager);
   
   ids_sort = hypre_BoxManIdsSort(manager);
       

   /* do we need to gather entries -check to see if ANY processor
      called a gather?*/
   is_gather = hypre_BoxManIsGatherCalled(manager);
   MPI_Allreduce(&is_gather, &global_is_gather, 1, MPI_INT, MPI_LOR, comm);

  /* ----------------------------GATHER? ------------------------------------*/  

   if (global_is_gather)
   {
      
      /* if AP, use AP to find out who owns the data we need.  In the 
         non-AP, then just gather everything. */

      /* Goal: to gather the entries from the relevant processor and
       * add to the *entries array.  Also add the proc and id to the
       * procs_sort and ids_sort arrays */

#ifdef HYPRE_NO_GLOBAL_PARTITION
      {
         int  size, index;
         int *tmp_proc_ids;
         int  proc_count, proc_alloc;
         int *proc_array;
         int *ap_proc_ids;
         int  count;
         int  global_num_boxes;
         int  max_response_size;
         int  non_info_size, entry_size_bytes;
         int *neighbor_proc_ids = NULL, *neighbor_boxnums = NULL;
         int *response_buf_starts;
         int *response_buf;
         int  response_size, tmp_int;

         int *send_buf = NULL;
         int *send_buf_starts = NULL;
         int *order_index = NULL;
         int *delete_array = NULL;
         int  tmp_id, start, d, proc, id;
         int *tmp_int_ptr;
         int *contact_proc_ids = NULL;

         int max_regions, max_refinements, ologp;
         
         int  grow, grow_array[6];
         int  sendbuf6[6], recvbuf6[6];
         
         int  *local_boxnums;

         int dim = hypre_BoxManDim(manager);

         int *my_ids         = hypre_BoxManMyIds(manager);
         int  num_my_entries = hypre_BoxManNumMyEntries(manager);
         
         void *entry_response_buf;
         void *index_ptr;

         double gamma;
         double local_volume, global_volume;
         double sendbuf2[2], recvbuf2[2];

         hypre_BoxArray *gather_regions;
         hypre_BoxArray *local_boxes;

         hypre_Box *box, *grow_box, *bounding_box;

         hypre_StructAssumedPart *ap;

         hypre_DataExchangeResponse  response_obj, response_obj2;
       
         hypre_BoxManEntry **my_entries = hypre_BoxManMyEntries(manager) ;
         hypre_BoxManEntry *entry_ptr;         

         hypre_Index imin, imax;

         hypre_IndexRef  min_ref, max_ref;
         
         /* 1.  Need to create an assumed partition */   

         /* create an array of local boxes.  also get the bounding box 
            and the global box size (as a double).  we don't use the bounding box
            from the grid since we care about the zero volume boxes */
         local_boxes = hypre_BoxArrayCreate(num_my_entries);
         local_boxnums = hypre_CTAlloc(int, num_my_entries);
         bounding_box = hypre_BoxCreate();
         grow_box = hypre_BoxCreate();
         local_volume = 0.0;
         for (d=0; d<3; d++) /* initialize */
         {
            hypre_IndexD(imin, d) = pow(2,30) ; 
            hypre_IndexD(imax, d) = -pow(2,30);
         }
         
         for (i=0; i< num_my_entries; i++)
         {
            
            entry_ptr = my_entries[i];
            min_ref = hypre_BoxManEntryIMin(entry_ptr);
            max_ref =  hypre_BoxManEntryIMax(entry_ptr);
            box = hypre_BoxArrayBox(local_boxes, i);
            hypre_BoxSetExtents( box, min_ref, max_ref );

            local_boxnums[i] =   hypre_BoxManEntryId(entry_ptr);

            local_volume += (double) hypre_BoxVolume(box);
  
            /* check for zero volume  boxes */

            if (hypre_BoxVolume(box) == 0) /* zero volume boxes - still count */
            {
               hypre_CopyBox(box, grow_box);
               for (d = 0; d < 3; d++)
               {
                  if(!hypre_BoxSizeD(box, d))
                  {
                     grow = (hypre_BoxIMinD(box, d) - hypre_BoxIMaxD(box, d) + 1)/2;
                     grow_array[2*d] = grow;
                     grow_array[2*d+1] = grow;
                  }
                  else
                  {
                     grow_array[2*d] = 0;
                     grow_array[2*d+1] = 0;
                  }
               }   
               /* expand the box */
               hypre_BoxExpand(grow_box, grow_array);
               box = grow_box; /*pointer copy*/
            }     
            /*now we have a vol > 0 box  - look for bounding box*/
             
            for (d = 0; d < 3; d++) /* for each dimension */
            {
               hypre_IndexD(imin, d) = hypre_min( hypre_IndexD(imin, d), 
                                                       hypre_BoxIMinD(box, d));
               hypre_IndexD(imax, d) = hypre_max( hypre_IndexD(imax, d), 
                                                       hypre_BoxIMaxD(box, d));
            }
 

            
         }/* end of local boxes */
         hypre_BoxSetExtents(bounding_box, imin, imax);
         /* if dim < 3 then set the extra dimensions to zero */
         for (d = dim; d < 3; d++)
         {
            hypre_BoxIMinD(bounding_box, d) = 0;
            hypre_BoxIMaxD(bounding_box, d) = 0;
         }

         /* now get the bounding box */
         for (d = 0; d < 3; d++) 
         {
            sendbuf6[d] = hypre_BoxIMinD(bounding_box, d);
            sendbuf6[d+3] = -hypre_BoxIMaxD(bounding_box, d);
         }
         MPI_Allreduce(sendbuf6, recvbuf6, 6, MPI_INT, MPI_MIN, comm);
         
         for (d = 0; d < 3; d++)
         {
            hypre_BoxIMinD(bounding_box, d) = recvbuf6[d];
            hypre_BoxIMaxD(bounding_box, d) = -recvbuf6[d+3];
         }
      
         if (hypre_BoxVolume(bounding_box) == 0) 
         {
            /* this shouldn't happen! */
            hypre_error(HYPRE_ERROR_GENERIC);
            return hypre_error_flag;
         }

         /* get the number of global entries and the global volume */

         sendbuf2[0] = local_volume;
         sendbuf2[1] = (double) num_my_entries;
         
         MPI_Allreduce(&sendbuf2, &recvbuf2, 2, MPI_DOUBLE, MPI_SUM, comm);   
          
         global_volume = recvbuf2[0];
         global_num_boxes = (int) recvbuf2[1];

         /* estimates for the assumed partition */ 
         d = nprocs/2;
         ologp = 0;
         while ( d > 0)
         {
            d = d/2; /* note - d is an int - so this is floored */
            ologp++;
         }
         
         max_regions =  hypre_min(pow(2, ologp+1), 10*ologp);
         max_refinements = ologp;
         gamma = .6; /* percentage a region must be full to 
                   avoid refinement */  
    

         hypre_StructAssumedPartitionCreate(dim, bounding_box, 
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
         hypre_BoxDestroy(bounding_box);
         hypre_BoxDestroy(grow_box);
         


         /* 2.  Need to be able to find our own entry, given the box
            number - for the second data exchange do some sorting.
            Then we can use my_ids to quickly find an entry.  This will be
            freed when the sort table is created (it's redundant).
            (Note: if we are creating the AP here, then this sorting
            may need to be done at the beginning of this function)*/  

         hypre_entryqsort2(my_ids, my_entries, 0, num_my_entries - 1);

         /* go thru gather regions and find out which processor's AP region
            they intersect */
         gather_regions = hypre_BoxManGatherRegions(manager);

         /*allocate space to store info from one box */  
         proc_count = 0;
         proc_alloc = 8;
         proc_array = hypre_CTAlloc(int, proc_alloc);


         /* probably there will mostly be one proc per box -allocate space for 2*/
         size = 2*hypre_BoxArraySize(gather_regions);
         tmp_proc_ids =  hypre_CTAlloc(int, size);
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
               size = size + proc_count + 2*(hypre_BoxArraySize(gather_regions)-i);
               tmp_proc_ids = hypre_TReAlloc(tmp_proc_ids, int, size);
            }
            for (j = 0; j< proc_count; j++)
            {
               tmp_proc_ids[count] = proc_array[j];
               count++;
            }
         }
         
         hypre_TFree(proc_array);     

         /* now get rid of redundencies in tmp_proc_ids (since a box can lie in more than one AP
            - put in ap_proc_ids*/      
         qsort0(tmp_proc_ids, 0, count-1);
         proc_count = 0;
         ap_proc_ids = hypre_CTAlloc(int, count);
         
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
         /* for each of these processor ids, we need to get the entries in their
            box manager */

         /* note: we are not ever re-assembling, so we won't worry about whether we
          * already have the info from some of these processors. */


         /* need to do an exchange data*/

         /* if we simply returned the boxes in the
            AP region, we will not have the entry information- in particular,
            we will not have the "info" obj.  So we have to get this info by doing a 
            second communication where we contact the actual owers of the boxes and
            request the entry info...So:  

            (1) exchange #1: contact the AP processor, get the proc ids and box numbers in that AP region

            (2) exchange #2: use this info to contact the owner processor and from them get the 
            entire entry infomation: box, info, etc.*/
         
        
         /* exchange #1 - we send nothing, and the contacted proc
          * returns all of its proc and box ids in its AP */

         /* build response object*/
         response_obj.fill_response = hypre_FillResponseBoxMapAssemble1;
         response_obj.data1 = ap; /* needed to fill responses*/ 
         response_obj.data2 = NULL;           

         send_buf = NULL;
         send_buf_starts = hypre_CTAlloc(int, proc_count + 1);
         for (i=0; i< proc_count+1; i++)
         {
            send_buf_starts[i] = 0;  
         }
     
         response_buf = NULL; /*this and the next are allocated in exchange data */
         response_buf_starts = NULL;
              
         /*we expect back a pair of ints: proc id, box number for each box owned */
         size =  2*sizeof(int);
     
         /* this needs to be the same for all processors */ 
         max_response_size = (global_num_boxes/nprocs)*2;

    
         hypre_DataExchangeList(proc_count, ap_proc_ids, 
                                send_buf, send_buf_starts, 
                                0, size, &response_obj, max_response_size, 3, 
                                comm, (void**) &response_buf, &response_buf_starts);


         /*how many entries do we have?*/
         size = response_buf_starts[proc_count];

         /* find a new list of procsessors to contact, eliminate duplicates */ 
         neighbor_proc_ids = hypre_CTAlloc(int, size);
         neighbor_boxnums =  hypre_CTAlloc(int, size);
         order_index =  hypre_CTAlloc(int, size);

         index = 0;
         for (i=0; i< size; i++) /* for each neighbor box */
         {
            neighbor_proc_ids[i] = response_buf[index++];
            neighbor_boxnums[i] = response_buf[index++];

            order_index[i] = i;
         }

         /*clean */
         hypre_TFree(send_buf_starts);
         hypre_TFree(ap_proc_ids);
         hypre_TFree(response_buf_starts);
         hypre_TFree(response_buf);

         /* look for duplicates and delete - also delete any with our proc_id */

         /*sort on proc_id  - move boxnums and order_index*/
         hypre_qsort3i(neighbor_proc_ids, neighbor_boxnums, order_index, 0, size-1);

         delete_array = hypre_CTAlloc(int, size);
         index = 0;
         proc_count = 0; /* keep track of # of distinct procs */
         
         /*now within each proc_id, we need to sort the boxnums and order index*/
         if (size)
         {
            tmp_id = neighbor_proc_ids[0]; 
            proc_count++;
         }
         start = 0;
         for (i=1; i< size; i++)
         {
            if (neighbor_proc_ids[i] != tmp_id) 
            {
               proc_count++;
               /*now find duplicate boxnums  - or delete all if myid = tmp_id*/ 
               if (myid == tmp_id)
               {
                  proc_count--;
                  
                  for (j=start; j< i; j++)
                  {
                     delete_array[index++] = j;
                  }
               }
               else
               {
                  hypre_qsort2i(neighbor_boxnums, order_index, start, i-1);
                  for (j=start+1; j< i; j++)
                  {
                     if (neighbor_boxnums[j] == neighbor_boxnums[j-1])
                     {
                        delete_array[index++] = j;
                     }
                  }
               }
               
               /* update start and tmp_id */  
               start = i;
               tmp_id = neighbor_proc_ids[i];
            }
         }

         /* final sort and purge (the last group doesn't 
            get caught in the above loop) */
         if (size)
         {
            /*now find duplicate boxnums  - or delete all if myid = tmp_id*/ 
            if (myid == tmp_id)
            {
               proc_count--;
               for (j=start; j< size; j++)
               {
                  delete_array[index++] = j;
               }
            }
            else
            {
               hypre_qsort2i(neighbor_boxnums, order_index, start, size-1);
               for (j=start+1; j< size; j++)
               {
                  if (neighbor_boxnums[j] == neighbor_boxnums[j-1])
                  {
                     delete_array[index++] = j;
                  }
               }
            }

         }

         /*now delete (index) duplicates from ids and boxnums */
         /* size = num of neighbor boxes 
            proc_count = # unique proc ids       */
         if (index)
         {
            start = delete_array[0];
            j = 0;
            for (i = start; (i + j) < size; i++)
            {
               if (j < index)
               {
                  while ((i+j) == delete_array[j]) /* see if deleting consec. items */
                  {
                     j++; /*increase the shift*/
                     if (j == index) break;
                  }
               }
               if ((i+j) < size) /* if deleting the last item then no moving */
               {
                  neighbor_boxnums[i] = neighbor_boxnums[i+j];
                  neighbor_proc_ids[i] =  neighbor_proc_ids[i+j];
               }
               
            }
         }
       
         /* finished removing duplicate (proc, id) combinations */
         /* new number of boxes is (size - index) */

         /* new contact: (send_buf is neighbor_boxnums)*/
         size = size - index;
         send_buf_starts = hypre_CTAlloc(int, proc_count + 1);
         send_buf_starts[0] = 0;
         contact_proc_ids = hypre_CTAlloc(int, proc_count);
         

         /* don't check if we have an entry before getting more info -
            eliminate duplicates later */ 

         /* create list of unique proc ids and starts buffer */
         if (proc_count)
         {
            contact_proc_ids[0] = neighbor_proc_ids[0];
         }
         
         index = 0; /* for indexing into unique contact_proc_ids */
         for (i= 1; i< size; i++)
         {
            if (neighbor_proc_ids[i] != contact_proc_ids[index])
            {
               index++;
               contact_proc_ids[index] = neighbor_proc_ids[i];
               send_buf_starts[index] = i;
            }
         }
         send_buf_starts[proc_count] = size;
         
         /* exchange #2 - now we contact processors with a box id and that processor needs
            to send us the entry information*/ 

         entry_response_buf = NULL; /*this and the next are allocated in exchange data */
         response_buf_starts = NULL;

         response_obj2.fill_response = hypre_FillResponseBoxMapAssemble2;
         response_obj2.data1 = manager; /* needed to fill responses*/ 
         response_obj2.data2 = NULL;   

        
         /*how big is an entry? extents: 6 ints, proc: 1 int, id: 1
          * int , num_ghost: 6 ints, info: info_size is in bytes*/
         /* note: for now, we do not need to send num_ghost - this
            is just copied in addentry anyhow */
         non_info_size = 8;
         entry_size_bytes = non_info_size*sizeof(int) + hypre_BoxManEntryInfoSize(manager);

         /* use same max_response_size as previous exchange */

         hypre_DataExchangeList(proc_count, contact_proc_ids, 
                                neighbor_boxnums, send_buf_starts, sizeof(int),
                                entry_size_bytes, &response_obj2, max_response_size, 4, 
                                comm,  &entry_response_buf, &response_buf_starts);


         /* now we can add entries that are in response_buf
            - we check for duplicates later */
 
         /*how many entries do we have?*/
         response_size = response_buf_starts[proc_count];  
 
         /* do we need more storage ?*/
         if (nentries + response_size >  hypre_BoxManMaxNEntries(manager))
         {
            int inc_size;
         
            inc_size = (response_size + nentries - hypre_BoxManMaxNEntries(manager));
            hypre_BoxManIncSize ( manager, inc_size);
            
            entries =  hypre_BoxManEntries(manager);
            procs_sort = hypre_BoxManProcsSort(manager);
            ids_sort = hypre_BoxManIdsSort(manager);
         }
         
         index_ptr = entry_response_buf; /* point into response buf */
         for (i = 0; i < response_size; i++)
         {
            size = sizeof(int);
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
            tmp_int_ptr = (int *) index_ptr;
            proc = *tmp_int_ptr;
            index_ptr =  (void *) ((char *) index_ptr + size);

            /* id */
            tmp_int_ptr = (int *) index_ptr;
            id = *tmp_int_ptr;
            index_ptr =  (void *) ((char *) index_ptr + size);
            
            /* info */
            /* index_ptr is at info */            
  
            hypre_BoxManAddEntry( manager , imin , 
                                  imax , proc, id, 
                                  index_ptr );

            /* start of next entry */  
            index_ptr = (void *) ((char *) index_ptr + hypre_BoxManEntryInfoSize(manager));
         }

         /* clean up from this section of code*/
         hypre_TFree(entry_response_buf);
         hypre_TFree(response_buf_starts);
         hypre_TFree(send_buf_starts);
         hypre_TFree(contact_proc_ids);
         hypre_TFree(neighbor_boxnums);
         hypre_TFree(neighbor_proc_ids);
         hypre_TFree(order_index);
         hypre_TFree(delete_array);
         

         /*we don't need special access to my entries anymore - because we will
           create the sort table */
         {
            
            hypre_TFree(hypre_BoxManMyIds(manager));
            hypre_TFree(hypre_BoxManMyEntries(manager));
            
            hypre_BoxManMyIds(manager) = NULL;
            hypre_BoxManMyEntries(manager) = NULL;
         }
         
      }
      
      /* end of gathering for the AP case */
#else

      /* beginning of gathering for the non-AP case */
      {

         /* collect global data */
         int  entry_size_bytes;
         int  send_count, send_count_bytes;
         int *displs, *recv_counts;
         int  recv_buf_size, recv_buf_size_bytes;
         int  d;
         int  size, non_info_size;
         int  proc, id;
         int  tmp_int;
         int *tmp_int_ptr;
         int  mydispls;

         void *send_buf = NULL;
         void *recv_buf = NULL;

         hypre_BoxManEntry  *entry;

         hypre_IndexRef index;

         hypre_Index imin, imax;

         void *index_ptr;

         /*how big is an entry? extents: 6 ints, proc: 1 int, id: 1
          * int , num_ghost: 6 ints, info: info_size is in bytes*/
         /* note: for now, we do not need to send num_ghost - this
            is just copied in addentry anyhow */
         non_info_size = 8;
         entry_size_bytes = non_info_size*sizeof(int) + hypre_BoxManEntryInfoSize(manager);

         /* figure out how many entries each proc has - let the group know */ 
         send_count = nentries;
         send_count_bytes = send_count*entry_size_bytes;
         recv_counts = hypre_CTAlloc(int, nprocs);
      
         MPI_Allgather(&send_count_bytes, 1, MPI_INT,
                       recv_counts, 1, MPI_INT, comm);

         displs = hypre_CTAlloc(int, nprocs);
         displs[0] = 0;
         recv_buf_size_bytes = recv_counts[0];
         for (i = 1; i < nprocs; i++)
         {
            displs[i] = displs[i-1] + recv_counts[i-1];
            recv_buf_size_bytes += recv_counts[i];
         }
         recv_buf_size = recv_buf_size_bytes/ entry_size_bytes;
         mydispls = displs[myid]/entry_size_bytes;

         /* populate the send buffer with my entries*/
         send_buf = hypre_MAlloc(send_count_bytes);
         recv_buf = hypre_MAlloc(recv_buf_size_bytes);
         
         index_ptr = send_buf; /* step through send_buf with this pointer */
         /* loop over my entries */  
         for (i=0; i < send_count; i++)
         {
            entry = &entries[i];

            size = sizeof(int);

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
   
            /* num_ghost (6 integers) - Don't send */
            /* size = 6*size;
            memcpy(index_ptr, hypre_BoxManEntryNumGhost(entry), size);
            index_ptr =  (void *) ((char *) index_ptr + size);*/

            /*info*/
            size = hypre_BoxManEntryInfoSize(manager);
            memcpy(index_ptr, hypre_BoxManEntryInfo(entry), size);
            index_ptr =  (void *) ((char *) index_ptr + size);
            
         } /* end of loop over my entries */

         /* now send_buf is full */  


         MPI_Allgatherv(send_buf, send_count_bytes, MPI_BYTE,
                        recv_buf, recv_counts, displs, MPI_BYTE, comm);

         /* unpack recv_buf into entries - note that the recv_buf size
          includes our own entries - which we don't need to readd (so
          size needs to be recv_buf_size - nentries)
         */ 
 
         if (recv_buf_size > hypre_BoxManMaxNEntries(manager))
         {
            int inc_size;
         
            inc_size = (recv_buf_size - hypre_BoxManMaxNEntries(manager));
            hypre_BoxManIncSize ( manager, inc_size);
            
            nentries = hypre_BoxManNEntries(manager);
            entries =  hypre_BoxManEntries(manager);
            procs_sort = hypre_BoxManProcsSort(manager);
            ids_sort = hypre_BoxManIdsSort(manager);
         }

         index_ptr = recv_buf; /* point into recv buf */

         for (i = 0; i < recv_buf_size; i++)
         {
            /* don't add my own entries */
            if ( i == mydispls)
            {
               /*increment i to skip over my entries */
               i += (send_count - 1); /* will be increased by 1 by loop */
                                         
              /* fix pointer */
               size = entry_size_bytes * send_count;
               index_ptr =  (void *) ((char *) index_ptr + size);
               continue;
               
            }
            

            size = sizeof(int);
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
            tmp_int_ptr = (int *) index_ptr;
            proc = *tmp_int_ptr;
            index_ptr =  (void *) ((char *) index_ptr + size);

            /* id */
            tmp_int_ptr = (int *) index_ptr;
            id = *tmp_int_ptr;
            index_ptr =  (void *) ((char *) index_ptr + size);
            
            /* num_ghost (6 integers) - didn't send */
            /* size = 6*size;
            memcpy(hypre_BoxManEntryNumGhost(entry), index_ptr, size);
            index_ptr =  (void *) ((char *) index_ptr + size); */

            /* info */
            /* index_ptr is at info */            
  
            hypre_BoxManAddEntry( manager , imin , 
                                  imax , proc, id, 
                                  index_ptr );

            /* start of next entry */  
            index_ptr = (void *) ((char *) index_ptr + hypre_BoxManEntryInfoSize(manager));
         }
         
         hypre_BoxManAllGlobalKnown(manager) = 1;
         
      } /* end of non-AP gather */
#endif

   }/* end of if (gather entries) for both AP and non-AP */
   

   nentries = hypre_BoxManNEntries(manager);
   entries =  hypre_BoxManEntries(manager);


    /* -----------------------SORT TABLE --------------------------------------*/

   /* now everything we need is in entries, also ids and procs have
    * been added, but not sorted */

   /* check for and remove duplicate boxes - based on (proc, id) */
   /* at the same time sort the procs_sort and ids_sort and
    * create the sort_table */   
   {
      int *order_index;
      int *delete_array;
      int *delete_array_orig_order;
      int *expand_delete_array;
      int  tmp_id, start, index;
      int  first_local;
      int  num_procs_sort;
      int *proc_offsets;
      int  myoffset;

      hypre_BoxManEntry **sort_table;

      order_index = hypre_CTAlloc(int, nentries);
      delete_array =  hypre_CTAlloc(int, nentries);
      index = 0;
   
      /* these are negative if a proc does not have any local entries
         in the manager */
      first_local = -1;
      myoffset = -1;
      

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
         if (tmp_id == myid ) first_local = 0;
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
           if(tmp_id == myid )
           {
              /* subtract the value of index as this is how many previous we will
                 delete */
              first_local = i - index;
           }
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
      /* need to delete dublicates in *entries, and build the sorted pointers to
         the entries */  

      
      if (index)
      {
         /* need to get the indices of the entries to delete as they 
            appear in the entries array */
         delete_array_orig_order = hypre_CTAlloc(int, index);
         for (i=0; i< index; i++)
         {
            delete_array_orig_order[i] = order_index[delete_array[i]];
         }
         qsort0(delete_array_orig_order, 0, index-1);

         /* delete entries - this also updates nentires in the box manager */
         hypre_BoxManDeleteMultipleEntries(manager, delete_array_orig_order, index );
         
         hypre_TFree(delete_array_orig_order);

         /* now delete from sort procs and sort ids -use delete_array because these
          have already been sorted.  also delete from order_index */
         start = delete_array[0];
         j = 0;
         for (i = start; (i + j) < nentries; i++)
         {
            if (j < index)
            {
               while ((i+j) == delete_array[j]) /* see if deleting consec. items */
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
         
         nentries =  hypre_BoxManNEntries(manager); /* updated if some entries were deleted */
         entries =  hypre_BoxManEntries(manager);
      
         /* need to adjust the indexes of the entries to be deleted by the number of
            entries delete prior to each index (expand_delete_array) */
         expand_delete_array = hypre_CTAlloc(int, nentries);
         j=0;
         start = delete_array[0];
         for (i=0; i < nentries; i++)
         {
            if (i == start)
            {
               j++;
               if (j < index) start = delete_array[j];
            }
            expand_delete_array[i] = j;
         }

      }
      else /* index ==0 (no deleting ) */
      {
         expand_delete_array = hypre_CTAlloc(int, nentries); /* all zeros */
      }
      

      /* populate sort table with links to entries - have to account for 
       deleted entries*/
      sort_table = hypre_CTAlloc(hypre_BoxManEntry *, nentries);

      for (i= 0; i< nentries; i++)
      {
         sort_table[i] = &entries[order_index[i] - expand_delete_array[i]];
      }
      
      /* create proc_offsets  (mmyoffset corresponds to local id)*/
      proc_offsets = hypre_CTAlloc(int, num_procs_sort + 1);
      proc_offsets[0] = 0;
     
      if (nentries > 0)
      {
         j=1;
         tmp_id = procs_sort[0];
         if (myid == tmp_id) myoffset =0;
         for (i=0; i < nentries; i++) 
         {
            if (procs_sort[i] != tmp_id)
            {
               if (myid == procs_sort[i]) myoffset = j;
               proc_offsets[j++] = i;
               tmp_id = procs_sort[i];
               
            }
         }
         proc_offsets[j] = nentries; /* last one */
      }
    

      /* clean up from this section of code */
      hypre_TFree(expand_delete_array);
      hypre_TFree(delete_array);
      hypre_TFree(order_index);

      hypre_BoxManProcsSortOffsets(manager) = proc_offsets;
      hypre_BoxManNumProcsSort(manager) = num_procs_sort;
      hypre_BoxManFirstLocal(manager) = first_local;
      hypre_BoxManSortTable(manager) = sort_table;
      hypre_BoxManLocalProcOffset(manager) = myoffset;


   }/* end bracket for all or the sorting stuff */
   

   /*------------------------------INDEX TABLE ---------------------------*/

   /* now build the index_table and indexes array */
   /* Note: for now we are using the same scheme as in BoxMap  */
   {
      int *indexes[3];
      int  size[3];
      int  iminmax[2];
      int  index_not_there;
      int  d, e;
      int  mystart, myfinish;
      int  imin[3];
      int  imax[3];
      int  start_loop[3];
      int  end_loop[3];
      int  loop, range, loop_num;
      int *proc_offsets;

      hypre_BoxManEntry  **index_table;
      hypre_BoxManEntry   *entry;
      hypre_BoxManEntry  **sort_table;

      hypre_IndexRef entry_imin;
      hypre_IndexRef entry_imax;

     
      /* initial */
      nentries     = hypre_BoxManNEntries(manager);
      entries      = hypre_BoxManEntries(manager);
      sort_table   = hypre_BoxManSortTable(manager);
      proc_offsets = hypre_BoxManProcsSortOffsets(manager);
      
  
      /*------------------------------------------------------
       * Set up the indexes array and record the processor's
       * entries. This will be used in ordering the link list
       * of BoxManEntry- ones on this processor listed first.
       *------------------------------------------------------*/
      for (d = 0; d < 3; d++)
      {
         indexes[d] = hypre_CTAlloc(int, 2*nentries);/* room for min and max of 
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

            for (i = 0; i < 2; i++)
            {
               /* find the new index position in the indexes array */
               index_not_there = 1;
               for (j = 0; j < size[d]; j++) /* size indicates current 
                                                size of index in that dimension*/
               {
                  if (iminmax[i] <= indexes[d][j])
                  {
                     if (iminmax[i] == indexes[d][j])
                     {
                        index_not_there = 0;
                     }
                     break;
                  }
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
      
      /* size was increased at the end - adjust when finished */
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
      index_table = hypre_CTAlloc(hypre_BoxManEntry *, (size[0] * size[1] * size[2]));
            
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
            entry = sort_table[range];
            entry_imin = hypre_BoxManEntryIMin(entry);
            entry_imax = hypre_BoxManEntryIMax(entry);
            
            /* find the indexes corresponding to the current box - put in 
               imin and imax */
            for (d = 0; d < 3; d++)
            {
               j = 0;
               while (hypre_IndexD(entry_imin, d) != indexes[d][j])
               {
                  j++;
               }
               hypre_IndexD(imin, d) = j;
               
               while (hypre_IndexD(entry_imax, d) + 1 != indexes[d][j])
               {
                  j++;
               }
               hypre_IndexD(imax, d) = j;
            } /* now have imin and imax location in index array*/
            
            
            /* set up index table */
            for (k = imin[2]; k < imax[2]; k++)
            {
               for (j = imin[1]; j < imax[1]; j++)
               {
                  for (i = imin[0]; i < imax[0]; i++)
                  {
                     if (!(index_table[((k) * size[1] + j) * size[0] + i])) /* no entry - 
                                                                               add one */
                     {
                        index_table[((k) * size[1] + j) * size[0] + i] = entry;
                     }
                     else  /* already and entry there - so add to link list for 
                              BoxMapEntry- overlapping */
                     {
                        hypre_BoxManEntryNext(entry)= index_table[((k) * size[1] + j) * size[0] + i];
                        index_table[((k) * size[1] + j) * size[0] + i]= entry;
                     }
                  }
               }
            }
         } /* end of subset of entries */
      }/* end of three loops over subsets */
    

      /* done with the index_table! */ 
      hypre_TFree( hypre_BoxManIndexTable(manager)); /* in case this is a 
                                                        re-assemble */
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
  (TEST THIS)

 *--------------------------------------------------------------------------*/

int hypre_BoxManIntersect ( hypre_BoxManager *manager , hypre_Index ilower , 
                            hypre_Index iupper , 
                            hypre_BoxManEntry ***entries_ptr , int *nentries_ptr )

{
   int  d, i, j, k;
   int  find_index_d;
   int  current_index_d;
   int *man_indexes_d;
   int  man_index_size_d;
   int  cnt, nentries;
   int *ii, *jj, *kk;
   int *proc_ids, *ids, *unsort;
   int  tmp_id, start;

   int  man_ilower[3] = {0, 0, 0};
   int  man_iupper[3] = {0, 0, 0};

   hypre_BoxManEntry **entries, **all_entries;
   hypre_BoxManEntry  *entry;

   
   
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
   
   ii= hypre_CTAlloc(int, nentries);
   jj= hypre_CTAlloc(int, nentries);
   kk= hypre_CTAlloc(int, nentries);
   
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
            /* Record the indices for non-empty entries and count all ManEntries. */
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
   
   
   /* if no link lists of BoxManEntries, Just point to the unique BoxManEntries */
   if (nentries == cnt)
   {
      entries = hypre_CTAlloc(hypre_BoxManEntry *, nentries);
      for (i= 0; i< nentries; i++)
      {
         entries[i]= hypre_BoxManIndexTableEntry(manager, ii[i], jj[i], kk[i]);
      }
   }

   /* otherwise: link lists of BoxManEntries. Sorting needed to eliminate duplicates. */
   else
   {
      unsort      = hypre_CTAlloc(int, cnt);
      proc_ids    = hypre_CTAlloc(int, cnt);
      ids         = hypre_CTAlloc(int, cnt);
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



/*------------------------------------------------------------------------------
 * hypre_BoxManSetNumGhost
 *-----------------------------------------------------------------------------*/


int
hypre_BoxManSetNumGhost( hypre_BoxManager *manager, int  *num_ghost )
{

  int  i;
  
  for (i = 0; i < 6; i++)
  {
     hypre_BoxManNumGhost(manager)[i] = num_ghost[i];
  }

  return hypre_error_flag;

}



/******************************************************************************
    hypre_fillResponseBoxManAssemble1 

    contact message is null.  need to return the id and boxnum of each box
    in our assumed partition.

 *****************************************************************************/

int
hypre_FillResponseBoxMapAssemble1(void *p_recv_contact_buf, 
                             int contact_size, int contact_proc, void *ro, 
                             MPI_Comm comm, void **p_send_response_buf, 
                             int *response_message_size )
{
   

   int    myid, i, index;
   int    size, num_objects;
   int   *proc_ids;
   int   *boxnums;
   int   *send_response_buf = (int *) *p_send_response_buf;
    
 
   hypre_DataExchangeResponse  *response_obj = ro;  
   hypre_StructAssumedPart     *ap = response_obj->data1;  

   int overhead = response_obj->send_response_overhead;

   /*initialize stuff */
   MPI_Comm_rank(comm, &myid );
  
   proc_ids =  hypre_StructAssumedPartMyPartitionProcIds(ap);
   boxnums = hypre_StructAssumedPartMyPartitionBoxnums(ap);

   /*we need to send back the list of all of the boxnums and
     corresponding processor id */

   /*how many boxes do we have in the AP?*/
   num_objects = hypre_StructAssumedPartMyPartitionIdsSize(ap);
  
   /* num_objects is then how much we need to send*/
  
   
   /*check storage in send_buf for adding the information */
   /* note: we are returning objects that are 2 ints in size */

   if ( response_obj->send_response_storage  < num_objects  )
   {
      response_obj->send_response_storage =  hypre_max(num_objects, 10); 
      size =  2*(response_obj->send_response_storage + overhead);
      send_response_buf = hypre_TReAlloc( send_response_buf, int, 
                                          size);
      *p_send_response_buf = send_response_buf;  
   }

   /* populate send_response_buf */
   index = 0;
   for (i = 0; i< num_objects; i++)
   {
      /* processor id + boxnum */
      send_response_buf[index++] = proc_ids[i];
      send_response_buf[index++] = boxnums[i];
   }

   /* return variables */
   *response_message_size = num_objects;
   *p_send_response_buf = send_response_buf;

   return hypre_error_flag;
   

}
/******************************************************************************
 
  hypre_fillResponseBoxManAssemble2
 
  contact message includes a box number (id).  the response needs to be the entry
  corresponding to the receiving processor and that id.

 *****************************************************************************/

int
hypre_FillResponseBoxMapAssemble2(void *p_recv_contact_buf, 
                             int contact_size, int contact_proc, void *ro, 
                             MPI_Comm comm, void **p_send_response_buf, 
                             int *response_message_size )
{
   

   int   myid, i, d, num_entries, location, size;
   int   proc_id, box_id, tmp_int;
   int   num_entries_found = 0;
   int   entry_size_bytes;

 
   int  *recv_contact_buf = (int *) p_recv_contact_buf;
  

   void  *send_response_buf = (void *) *p_send_response_buf;
   void  *index_ptr;
 
   hypre_BoxManEntry *entry;

   hypre_IndexRef  index;
  
   hypre_DataExchangeResponse  *response_obj = ro;  

   hypre_BoxManager     *manager = response_obj->data1;  
   
   int   overhead = response_obj->send_response_overhead;

   hypre_BoxManEntry **my_entries = hypre_BoxManMyEntries(manager) ;
 
   int  *my_ids =   hypre_BoxManMyIds(manager);
   int   num_my_entries = hypre_BoxManNumMyEntries(manager);

   /*initialize stuff */
   MPI_Comm_rank(comm, &myid );

   entry_size_bytes = 8*sizeof(int) + hypre_BoxManEntryInfoSize(manager);
   
   /* num_entries is one for each boxnum sent (corres. to our proc. id only)*/
   num_entries = contact_size;

   /*check storage in send_buf for adding the information */
   if ( response_obj->send_response_storage  < num_entries  )
   {
      response_obj->send_response_storage =  hypre_max(num_entries, 10); 
      size =  entry_size_bytes*(response_obj->send_response_storage + overhead);
      send_response_buf = hypre_ReAlloc( send_response_buf, size);
      *p_send_response_buf = send_response_buf;  
   }

   index_ptr = send_response_buf; /* step through send_buf with this pointer */

   if (num_my_entries == 0)
   {
      num_entries_found = 0;
   }
   else
   {
      for (i=0; i < num_entries; i++)
      {
         proc_id = myid;
         box_id = recv_contact_buf[i];
         
         /* find entry */ 
         location = hypre_BinarySearch(my_ids, box_id, num_my_entries);
         if (location  >= 0)
         {
            entry = my_entries[location];
            num_entries_found++;
            
            /*pack response buffer with information (check storage first) */
            
            size = sizeof(int);
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
            memcpy( index_ptr, &proc_id, size);
            index_ptr =  (void *) ((char *) index_ptr + size);
            
            /* id */
            memcpy( index_ptr, &box_id, size);
            index_ptr =  (void *) ((char *) index_ptr + size);
            
            /*info*/
            size = hypre_BoxManEntryInfoSize(manager);
            memcpy(index_ptr, hypre_BoxManEntryInfo(entry), size);
            index_ptr =  (void *) ((char *) index_ptr + size);
            
         }
      }
   }
   
   /* now send_response_buf is full */  

   /* return variable */
   *response_message_size = num_entries_found;
   *p_send_response_buf = send_response_buf;

   return hypre_error_flag;
   

}



/******************************************************************************

 some specialized sorting routines

 *****************************************************************************/

/* sort on int i, move entry pointers ent */

void hypre_entryqsort2( int *v,
                       hypre_BoxManEntry ** ent,
                       int  left,
                       int  right )
{
   int i, last;

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


void hypre_entryswap2(int  *v,
                      hypre_BoxManEntry ** ent,
                      int  i,
                      int  j )
{
   int temp;

   hypre_BoxManEntry *temp_e;
   
   temp = v[i];
   v[i] = v[j];
   v[j] = temp;

   temp_e = ent[i];
   ent[i] = ent[j];
   ent[j] = temp_e;
}
