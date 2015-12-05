/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



#include "headers.h"



int hypre_FillResponseStructAssembleAP(void *, int, int, void *, 
                                 MPI_Comm, void **, int *);




#define DEBUG 0
#define AP_STAT 0
#define NEIGH_PRINT 0

#if DEBUG
char       filename[255];
FILE      *file;
int        my_rank;
#endif



/*--------------------------------------------------------------------------
 * new version of 
 * hypre_StructGridAssemble (for use with "no global partition" option)
 * AHB 6/05
 *------------------------------------------------------------------------*/ 


int hypre_StructGridAssembleWithAP( hypre_StructGrid *grid )
{
 


   int                  ierr = 0;
   int                  tmp_i;
   
   int                  size, global_num_boxes, num_local_boxes;
   int                  i, j, d, k, index;
   int                  num_procs, myid;
   int                  *sendbuf8, *recvbuf8, *sendbuf2, *recvbuf2;
   int                  min_box_size, max_box_size;
   int                  global_min_box_size, global_max_box_size;
   int                 *ids;
   int                  max_regions, max_refinements, ologp;
   double               gamma;
   hypre_Index          min_index, max_index;
   
  
   int                  prune;
       
   hypre_Box           *box;
   

   MPI_Comm             comm         = hypre_StructGridComm(grid);
   hypre_Box           *bounding_box = hypre_StructGridBoundingBox(grid);
   hypre_BoxArray      *local_boxes  = hypre_StructGridBoxes(grid);
   int                  dim          = hypre_StructGridDim(grid);
   hypre_BoxNeighbors  *neighbors    = hypre_StructGridNeighbors(grid);
   int                  max_distance = hypre_StructGridMaxDistance(grid);
   hypre_IndexRef       periodic     = hypre_StructGridPeriodic(grid);

   int                 *local_boxnums;

   double               dbl_global_size, tmp_dbl;
   
   hypre_BoxArray       *my_partition;
   int                  *part_ids, *part_boxnums;
     
   int                  *proc_array, proc_count, proc_alloc, count;
   int                  *tmp_proc_ids = NULL;
   
   int                  max_response_size;
   int                  *ap_proc_ids, *send_buf, *send_buf_starts;
   int                  *response_buf, *response_buf_starts;

   hypre_BoxArray      *neighbor_boxes, *n_boxes_copy;
   int                 *neighbor_proc_ids, *neighbor_boxnums;

   int                 *order_index, *delete_array;
   int                 tmp_id, start, first_local;
   
   int                 grow, grow_array[6];
   hypre_Box           *grow_box;
   
   
   int                  *numghost;
   int                   ghostsize;
   hypre_Box            *ghostbox;

   hypre_StructAssumedPart     *assumed_part;
   hypre_DataExchangeResponse  response_obj;
 
   int                  px = hypre_IndexX(periodic);
   int                  py = hypre_IndexY(periodic);
   int                  pz = hypre_IndexZ(periodic);

   int                  i_periodic = px ? 1 : 0;
   int                  j_periodic = py ? 1 : 0;
   int                  k_periodic = pz ? 1 : 0;

   int                  num_periods, multiple_ap, p;
   hypre_Box           *result_box, *period_box;
   hypre_Index         *pshifts;

   hypre_IndexRef       pshift;

#if NEIGH_PRINT
   double               start_time, end_time;
   
#endif



/*---------------------------------------------
  Step 1:  Initializations
  -----------------------------------------------*/

   prune = 1; /* default is to prune */ 
   
   num_local_boxes = hypre_BoxArraySize(local_boxes);
  
   num_periods = (1+2*i_periodic) * (1+2*j_periodic) * (1+2*k_periodic);


   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);


 
/*---------------------------------------------
  Step 2:  Determine the global size, total number of boxes,
           and global bounding box.
           Also get the min and max box sizes
           since it is convenient to do so.
  -----------------------------------------------*/

   if (neighbors == NULL) 
   {
    
      /*these may not be needed - check later */
      ids =   hypre_TAlloc(int, num_local_boxes);
    
      /* for the vol and number of boxes */
      sendbuf2 = hypre_CTAlloc(int, 2);
      recvbuf2 = hypre_CTAlloc(int, 2);
      size = 0;
     
      bounding_box = hypre_BoxCreate();
      grow_box = hypre_BoxCreate();
      

      if (num_local_boxes) 
      {
         
         min_box_size = hypre_BoxVolume( hypre_BoxArrayBox(local_boxes, 0));
         max_box_size = hypre_BoxVolume( hypre_BoxArrayBox(local_boxes, 0));


         /* initialize min and max */
         for (d=0; d<3; d++)
         {
            hypre_IndexD(min_index, d) = pow(2,30); 
            hypre_IndexD(max_index, d) = -pow(2,30);
         }
         

         hypre_ForBoxI(i, local_boxes)
         {
            box = hypre_BoxArrayBox(local_boxes, i);
            /* get global size and number of boxes */ 
            tmp_i = hypre_BoxVolume(box);
            size += tmp_i;
            min_box_size = hypre_min(min_box_size, tmp_i);
            max_box_size = hypre_max(max_box_size, tmp_i);


            /* set id */  
            ids[i] = i;


            /* 1/3/05 we need this for the case of holes in the domain. (I had
               commented
               it out on 12/04 - as I thought this was not necessary. */
            
            
            /* zero volume boxes - still look at for getting the bounding box */
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
            /*now we have a vol > 0 box */
    
         
            for (d = 0; d < dim; d++) /* for each dimension */
            {
               hypre_IndexD(min_index, d) = hypre_min( hypre_IndexD(min_index, d), 
                                                       hypre_BoxIMinD(box, d));
               hypre_IndexD(max_index, d) = hypre_max( hypre_IndexD(max_index, d), 
                                                       hypre_BoxIMaxD(box, d));
            }
                        
         }/*end for each box loop */

         /* bounding box extents */ 
         hypre_BoxSetExtents(bounding_box, min_index, max_index);
   
      }
      else /* no boxes owned*/
      {
         for (d=0; d<3; d++)
         {
            hypre_BoxIMinD(bounding_box, d) = pow(2,30) ; 
            hypre_BoxIMaxD(bounding_box, d) = -pow(2,30);
         }
         
         min_box_size = 0;
         max_box_size = 0;
         size = 0;
      }
     
      
      /* if dim < 3 then set the extra dimensions to zero */
      for (d = dim; d < 3; d++)
      {
         hypre_BoxIMinD(bounding_box, d) = 0;
         hypre_BoxIMaxD(bounding_box, d) = 0;
      }


      /* populate the vol and number of boxes buffer */
      sendbuf2[0] = size;
      sendbuf2[1] = hypre_BoxArraySize(local_boxes);
      

   /* set local size (volume) */ 
      hypre_StructGridLocalSize(grid) = size;

      MPI_Allreduce(sendbuf2, recvbuf2, 2, MPI_INT, MPI_SUM, comm);
      
      /* now set global size */
      hypre_StructGridGlobalSize(grid) = recvbuf2[0]; /*this could easily overflow!
                                                        need to change to a double 
                                                        in struct grid data 
                                                        structure */
      global_num_boxes = recvbuf2[1];

      /* in case of overflow (until we change the datatype in the struct grid 
         object) - we use this global vol for our calculations */
      tmp_dbl = (double) size;
      MPI_Allreduce(&tmp_dbl, &dbl_global_size, 1, MPI_DOUBLE, MPI_SUM, comm);      
      hypre_TFree(sendbuf2);
      hypre_TFree(recvbuf2);

#if AP_STAT
      if (myid ==0)
      {
         printf("myid = %d, GLOBAL number of boxes = %d\n", myid, global_num_boxes);
      }
#endif

      /* don't need the grow_box */
      hypre_BoxDestroy(grow_box);

         
      /* now get the bounding box and min and max box sizes */
      
      sendbuf8 = hypre_CTAlloc(int, 8);
      recvbuf8 = hypre_CTAlloc(int, 8);
      

      /*get the min global lower extents and max upper extents */
      /* note: min(x)= -max(-x) */
 
      for (d = 0; d < 3; d++) 
      {
         sendbuf8[d] = hypre_BoxIMinD(bounding_box, d);
         sendbuf8[d+3] = -hypre_BoxIMaxD(bounding_box, d);
      }

      /*also collect the min and max box sizes*/
      sendbuf8[6] = min_box_size;
      sendbuf8[7] = -max_box_size;
      

      MPI_Allreduce(sendbuf8, recvbuf8, 8, MPI_INT, MPI_MIN, comm);


      for (d = 0; d < 3; d++)
      {
         hypre_BoxIMinD(bounding_box, d) = recvbuf8[d];
         hypre_BoxIMaxD(bounding_box, d) = -recvbuf8[d+3];
      }

      global_min_box_size = recvbuf8[6];
      global_max_box_size = -recvbuf8[7];
      

      hypre_TFree(sendbuf8);
      hypre_TFree(recvbuf8);



#if 0
      printf("myid = %d, GLOBAL min box size = %d\n", myid, global_min_box_size);
      printf("myid = %d, GLOBAL max box size = %d\n", myid, global_max_box_size);
#endif


      /* in theory there should be at least one box with vol > 0, but 
         just in case: */
      if (global_num_boxes == 0)
      {
         for (d = 0; d < 3; d++)
         {
            hypre_BoxIMinD(bounding_box, d) = 0;
            hypre_BoxIMaxD(bounding_box, d) = 0;
      }
         global_min_box_size = 1;
         global_max_box_size = 1;
         
         /* do we need to set the global grid size ? */
         hypre_StructGridGlobalSize(grid) = 1;
         
      }
      
      if (hypre_BoxVolume(bounding_box) == 0) 
      {
         if (myid ==0)  printf("Error: bounding box has zero volume - "
                               "this shouldn't happen!\n");
         return -1; 
      }
      
      hypre_StructGridBoundingBox(grid) = bounding_box; 

   


      /* ids are in order  r: 0 - num(local-boxes) - 1*/
      /* only set if they have not already been set! */   
      if (  hypre_StructGridIDs(grid) == NULL)
      {
         hypre_StructGridIDs(grid) = ids;
      }
      else
      {
         /* not needed! */
         hypre_TFree(ids);
      }
      


   }
   


  
/*---------------------------------------------
  Step 3:  Create an assumed partition 
  -----------------------------------------------*/

/* want #regions < #procs */
/* want #regions bounded */


 if (neighbors == NULL) 
 {
   /* Initializations - these should probably be
      adjusted depending on the problem */ 

    /* what is the max number of regions we can store ? */
    /* what is the max number of refinements we can do? */
    /* what fraction of the region do we want to be full? */
 
   /* estimate of log(num_procs) */
 
    d = num_procs/2;
    ologp = 0;
    while ( d > 0)
    {
       d = d/2; /* note - d is an int - so this is floored */
       ologp++;
    }
    
    max_regions =  hypre_min(pow(2, ologp+1), 10*ologp);
    /* max_regions = 100; */
    
    max_refinements = ologp;
    /* max_refinements = 1;*/

  /* TEMP - lots of refinement */
    /*max_regions = num_procs;
      max_refinements = ologp + 10;*/
    
    /* new 1/10/04*/
    /* max_refinements = ologp/2;
       max_regions = 64;*/
    

    
    gamma = .6; /* percentage a region must be full to 
                   avoid refinement */  
    

    /* assign boxnums */
    local_boxnums = hypre_CTAlloc(int, num_local_boxes);
    for (i = 0; i< num_local_boxes; i++)
    {
       local_boxnums[i] = i;
    }
 

    ierr = hypre_StructAssumedPartitionCreate(dim, bounding_box, dbl_global_size, 
                                              global_num_boxes,
                                              local_boxes, local_boxnums,
                                              max_regions, 
                                              max_refinements, gamma, 
                                              comm, &assumed_part);

    
    hypre_TFree(local_boxnums);
    


    /*Now we have the assumed partition built */

#if AP_STAT
    my_partition = hypre_StructAssumedPartMyPartition(assumed_part);
    hypre_ForBoxI(i, my_partition)
    {
       box = hypre_BoxArrayBox(my_partition, i);
       printf("*****myid = %d: MY ASSUMED Partitions (%d):  (%d, %d, %d)  "
              "x  (%d, %d, %d)\n",
              myid, i,
              hypre_BoxIMinX(box),
              hypre_BoxIMinY(box),
              hypre_BoxIMinZ(box),
              hypre_BoxIMaxX(box),
              hypre_BoxIMaxY(box),
           hypre_BoxIMaxZ(box));
     }
#endif
   
    my_partition =  hypre_StructAssumedPartMyPartitionBoxes(assumed_part);
    part_ids =  hypre_StructAssumedPartMyPartitionProcIds(assumed_part);
    part_boxnums = hypre_StructAssumedPartMyPartitionBoxnums(assumed_part);   
     
#if AP_STAT
    printf("*****myid = %d: I have %d boxes in my AP from %d procs\n", myid, 
           hypre_BoxArraySize(my_partition),  
           hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part));
 
#endif

#if AP_STAT
   
    hypre_ForBoxI(i, my_partition)
    {
       box = hypre_BoxArrayBox(my_partition, i);
       printf("*****myid = %d: BOXES in MY AP (number %d):  (%d, %d, %d)  "
              "x  (%d, %d, %d), boxnum = %d, owned by proc %d\n",
              myid, i,
              hypre_BoxIMinX(box),
              hypre_BoxIMinY(box),
              hypre_BoxIMinZ(box),
              hypre_BoxIMaxX(box),
              hypre_BoxIMaxY(box),
           hypre_BoxIMaxZ(box), part_boxnums[i], part_ids[i]);
     }
#endif

 }



/*---------------------------------------------
  Step 5: Use the assumed partition to find a 
          shortened list of *potential* neighbors
  -----------------------------------------------*/


/* set neighbors */
  if (neighbors == NULL)
  {    
    
     /* need to pass in a list of boxes (to BoxNeighborsAssemble)-
        not all the boxes - that
        contains potential nearest neighbors.  Also a corresponding list of
        processor numbers (same length).  Also provide an id 
        number for each of these and 
        an index that indicates which one is the first of the local boxes.
        Previously the list of boxes contained
        ALL of the boxes in the entire domain! */
     
     /*first determine the shifting needed for periodic boxes */  
     pshifts = hypre_CTAlloc(hypre_Index, num_periods); /* this is deallocated 
                                                           in boxneighbordestroy */
     pshift = pshifts[0];
     hypre_ClearIndex(pshift);
     
     if( num_periods > 1 )
     {
        int  ip, jp, kp;
        p = 1;
        for (ip = -i_periodic; ip <= i_periodic; ip++)
        {
           for (jp = -j_periodic; jp <= j_periodic; jp++)
           {
              for (kp = -k_periodic; kp <= k_periodic; kp++)
              {
                 if( !(ip == 0 && jp == 0 && kp == 0) )
                 {                  
                    pshift = pshifts[p];
                    hypre_SetIndex(pshift, ip*px, jp*py, kp*pz);
                    p++;
                 }
              }
           }
        }
     }
     
     /* now we go through all our boxes and 
        we grow them in each dimension before checking to see whose AP
        they lie in - in case we have any near boundaries.  We need to 
        store the list of processors form the AP that our boxes intersect*/


     /*to store info from one box */  
     proc_count = 0;
     proc_alloc = 8;
     proc_array = hypre_CTAlloc(int, proc_alloc);

     /* probably there will mostly be one proc per box */
     size = 2*hypre_BoxArraySize(local_boxes);
     tmp_proc_ids =  hypre_CTAlloc(int, size);
     count = 0;

     box = hypre_BoxCreate();
     result_box = hypre_BoxCreate();
     period_box = hypre_BoxCreate();

     /* loop through all boxes */
     hypre_ForBoxI(i, local_boxes) 
     {
        multiple_ap = 0;
        hypre_CopyBox(hypre_BoxArrayBox(local_boxes, i) ,box);
        hypre_BoxExpandConstant(box, max_distance);

        if (hypre_BoxVolume(box) == 0) /* zero volume boxes - still
                                           grow more*/
        {
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
           hypre_BoxExpand(box, grow_array);
        }

        if (num_periods > 1)
        {
           /*treat periodic separately if it's at the edge of the domain - 
             then we need to look for periodic neighbors*/
           hypre_IntersectBoxes( box, bounding_box, result_box);        
           if (  hypre_BoxVolume(result_box) <  hypre_BoxVolume(box) ) 
           /* on the edge of the bounding box */
           {
              multiple_ap = 1;
           }
        }
                
        if (!multiple_ap) /* only one assumed partition (AP) call */
        {
           hypre_StructAssumedPartitionGetProcsFromBox(assumed_part, box, &proc_count, 
                                              &proc_alloc, &proc_array);
           if ((count + proc_count) > size)       
           {
              size = size + proc_count + 2*(hypre_BoxArraySize(local_boxes)-i);
              tmp_proc_ids = hypre_TReAlloc(tmp_proc_ids, int, size);
           }
           for (j = 0; j< proc_count; j++)
           {
              tmp_proc_ids[count] = proc_array[j];
              count++;
           }
           
        }
        
        else /* this periodic box needs to be shifted as it is near a boundary - 
                so we'll have multiple AP calls*/
        {
           for (k=0; k < num_periods; k++)
           {
              /* get the periodic box (k=0 is the actual box) */  
              hypre_CopyBox(box, period_box);
              pshift = pshifts[k];
              hypre_BoxShiftPos(period_box, pshift);
              /* see if the shifted box intersects the domain */  
              hypre_IntersectBoxes(period_box, bounding_box, result_box);  
              if (hypre_BoxVolume(result_box) > 0)
              {
                 hypre_StructAssumedPartitionGetProcsFromBox(assumed_part, period_box, 
                                                    &proc_count, &proc_alloc, 
                                                    &proc_array);
                 if ((count + proc_count) > size)       
                 {
                    size = size + proc_count + 
                       2*(hypre_BoxArraySize(local_boxes)-i);
                    tmp_proc_ids = hypre_TReAlloc(tmp_proc_ids, int, size);
                 }
                 for (j = 0; j< proc_count; j++)
                 {
                    tmp_proc_ids[count] = proc_array[j];
                    count++;
                 }
              }
           }
        } /* end of if multiple AP calls */
        
     } /* end of loop through boxes */
     
     hypre_TFree(proc_array);     
     hypre_BoxDestroy(box);
     hypre_BoxDestroy(result_box);
     hypre_BoxDestroy(period_box);
     

     /* now get rid of redundencies in tmp_proc_ids  - put in ap_proc_ids*/      
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

#if NEIGH_PRINT
      printf("*****myid = %d: I have %d procs to contact for their AP boxes \n", myid, proc_count);
#endif



     /* now we have a sorted list with no duplicates in ap_proc_ids */
     /* for each of these processor ids, we need to get the boxes in their 
        assumed partition as these are potential neighbors */ 

                
     /* need to do an exchange data for this! (use flag 2 since flag 1 is in create
        assumed partition)*/
     /* contact message will be empty.  the response message should be all the
        boxes + corresp. proc id + boxnum in that processor's assumed partition 
        (so 8 ints for each)*/

     /*response object*/
     response_obj.fill_response = hypre_FillResponseStructAssembleAP;
     response_obj.data1 = assumed_part; /* needed to fill responses*/ 
     response_obj.data2 = NULL;           

     send_buf = NULL;
     send_buf_starts = hypre_CTAlloc(int, proc_count + 1);
     for (i=0; i< proc_count+1; i++)
     {
        send_buf_starts[i] = 0;  
     }
    

     response_buf = NULL; /*this and the next are allocated in exchange data */
     response_buf_starts = NULL;
    
     
     /*we expect back an array of items consisting of 8 integers*/
     size =  8*sizeof(int);
     
     /* this needs to be the same for all processors */ 
     max_response_size = global_num_boxes/num_procs +  10;   
     max_response_size = (global_num_boxes/num_procs)*2;


#if NEIGH_PRINT
 start_time = MPI_Wtime();
#endif


    
     hypre_DataExchangeList(proc_count, ap_proc_ids, 
                            send_buf, send_buf_starts, 
                            0, size, &response_obj, max_response_size, 2, 
                            comm, (void**) &response_buf, &response_buf_starts);




#if NEIGH_PRINT
 end_time = MPI_Wtime();
 end_time = end_time - start_time;
 printf("myid = %d   EXCHANGE 2 time =  %f sec.\n", myid, end_time);  
#endif

     /*clean right away */
     hypre_TFree(send_buf_starts);
     hypre_TFree(ap_proc_ids);
     /*we no longer need the assumed partition */
     hypre_StructAssumedPartitionDestroy(assumed_part);

#if NEIGH_PRINT
 start_time = MPI_Wtime();
#endif

     /*now we have the potential list of neighbor boxes and corresponding  
       processors in response_buf*/
 
     /*how many neighbor boxes do we have?*/
     size = response_buf_starts[proc_count];
     hypre_TFree(response_buf_starts);

     neighbor_proc_ids = hypre_CTAlloc(int, size);
     neighbor_boxnums =  hypre_CTAlloc(int, size);
     neighbor_boxes = hypre_BoxArrayCreate(size);
     box = hypre_BoxCreate();
     order_index = hypre_CTAlloc(int, size);

     index = 0;
     for (i=0; i< size; i++) /* for each neighbor box */
     {
        neighbor_proc_ids[i] = response_buf[index++];
        neighbor_boxnums[i] = response_buf[index++];
        for (d=0; d< 3; d++)
        {
           hypre_BoxIMinD(box, d) =  response_buf[index++];
           hypre_BoxIMaxD(box, d) =  response_buf[index++];
        }
        hypre_CopyBox(box, hypre_BoxArrayBox(neighbor_boxes, i));
        order_index[i] = i;
     }
     
     hypre_BoxDestroy(box);
     hypre_TFree(response_buf);
    
     

     /* now we have an array of boxes, boxnums, and proc_ids - 
        not in any order and there
        may be duplicate pairs of proc id and boxes  - this list includes all of MY
        boxes as well*/
#if 0
     printf("*****myid = %d: I have %d potential neighbor boxes \n", myid, size);
    
     hypre_ForBoxI(i, neighbor_boxes)
        {
           box = hypre_BoxArrayBox(neighbor_boxes, i);
           printf("*****myid = %d: BOXES in my list of potential neighbors "
                  "(number %d):  (%d, %d, %d)  x  (%d, %d, %d) , boxnum = %d, "
                  "proc %d\n",
                  myid, i,
                  hypre_BoxIMinX(box),
                  hypre_BoxIMinY(box),
                  hypre_BoxIMinZ(box),
                  hypre_BoxIMaxX(box),
                  hypre_BoxIMaxY(box),
                  hypre_BoxIMaxZ(box), neighbor_boxnums[i], neighbor_proc_ids[i]);
     }

#endif


      /* to set neighbors  - pass in reduced list of potential neighbor boxes.
         We'll put processors in ascending order.  Also we need to remove
         duplicate boxes first */

     size = hypre_BoxArraySize(neighbor_boxes);   
    
     /*sort on proc_id  - move boxnums and order_index*/
     hypre_qsort3i(neighbor_proc_ids, neighbor_boxnums, order_index, 0, size-1);
     
     delete_array = hypre_CTAlloc(int, size);
     index = 0;
     first_local = 0;
     
     /*now within each proc_id, we need to sort the boxnums and order index*/
     if (size)
     {
        tmp_id = neighbor_proc_ids[0]; 
     }
     start = 0;
     for (i=1; i< size; i++)
     {
        if (neighbor_proc_ids[i] != tmp_id) 
        {
           hypre_qsort2i(neighbor_boxnums, order_index, start, i-1);
           /*now find duplicate boxnums */ 
           for (j=start+1; j< i; j++)
           {
              if (neighbor_boxnums[j] == neighbor_boxnums[j-1])
              {
                 delete_array[index++] = j;
              }
           }
           /* update start and tmp_id */  
           start = i;
           tmp_id = neighbor_proc_ids[i];
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
     if (size > 1)
     {
        hypre_qsort2i(neighbor_boxnums, order_index, start, size-1);
        /*now find duplicate boxnums */ 
        for (j=start+1; j<size; j++)
        {
           if (neighbor_boxnums[j] == neighbor_boxnums[j-1])
           {
              delete_array[index++] = j;
           }
        }
     }
     
    /* now index = the number in delete_array */

     /*now sort the boxes according to index_order */        
    n_boxes_copy = hypre_BoxArrayDuplicate(neighbor_boxes);
    for (i=0; i< size; i++)
    {
       hypre_CopyBox(hypre_BoxArrayBox(n_boxes_copy, order_index[i]), 
                     hypre_BoxArrayBox(neighbor_boxes,i ));
    }
    hypre_TFree(order_index); 
    hypre_BoxArrayDestroy(n_boxes_copy);

     /*now delete (index) duplicates from boxes */
    hypre_DeleteMultipleBoxes( neighbor_boxes, delete_array, index);
    /*now delete (index) duplicates from ids and boxnums */
    /* size = num of neighbor boxes */
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
          if ((i+j) < size) /* if deleteing the last item then no moving */
          {
             neighbor_boxnums[i] = neighbor_boxnums[i+j];
             neighbor_proc_ids[i] =  neighbor_proc_ids[i+j];
          }
          
       }
    }
       
    /* finished removing duplicates! */
    /* new number of boxes is (size - index) */
     
    
   
    hypre_TFree(delete_array);


#if AP_STAT
   
    printf("!!!!!*myid = %d: I have %d potential "
                         "neighbor boxes \n", myid, size-index);
    
#endif

#if 0

/* now let's check the shortened list */
   
     printf("!!!!!*myid = %d: I have %d potential neighbor boxes \n", 
            myid, size-index);
     hypre_ForBoxI(i, neighbor_boxes)
        {
           box = hypre_BoxArrayBox(neighbor_boxes, i);
           printf("!!!!!myid = %d: BOXES in my list of potential neighbors "
                  "(number %d):  (%d, %d, %d)  x  (%d, %d, %d) , boxnum = %d,"
                  "proc %d\n",
                  myid, i,
                  hypre_BoxIMinX(box),
                  hypre_BoxIMinY(box),
                  hypre_BoxIMinZ(box),
                  hypre_BoxIMaxX(box),
                  hypre_BoxIMaxY(box),
                  hypre_BoxIMaxZ(box), neighbor_boxnums[i], neighbor_proc_ids[i]);
     }
#endif


     /* create the neighbors structure! */ 
     hypre_BoxNeighborsCreateWithAP(neighbor_boxes, neighbor_proc_ids, 
                                 neighbor_boxnums,
                                 first_local, num_local_boxes, pshifts, &neighbors);
    




 
     hypre_StructGridNeighbors(grid) = neighbors;


  }
  
  
#if NEIGH_PRINT
 end_time = MPI_Wtime();
 end_time = end_time - start_time;
 printf("myid = %d  Getting ready for neighbor assemble stuff time =  %f sec.\n", myid, end_time);  
#endif 


/*---------------------------------------------
  Step 6:  Find your neighbors
  -----------------------------------------------*/


#if NEIGH_PRINT
 start_time = MPI_Wtime();
#endif

      hypre_BoxNeighborsAssembleWithAP(neighbors, periodic, max_distance, prune);


#if NEIGH_PRINT
 end_time = MPI_Wtime();
 end_time = end_time - start_time;
 printf("myid = %d   Box Neighbor assemble time =  %f sec.\n", myid, end_time);  
#endif

/*---------------------------------------------
  Step 7:  Expand to include ghosts
-----------------------------------------------*/


   numghost = hypre_StructGridNumGhost(grid) ;
   ghostsize = 0;
   ghostbox = hypre_BoxCreate();
   hypre_ForBoxI(i, local_boxes)
   {
     box = hypre_BoxArrayBox(local_boxes, i);
          
     hypre_CopyBox(box, ghostbox);
     hypre_BoxExpand(ghostbox, numghost);         
            
     ghostsize += hypre_BoxVolume(ghostbox);        

   }
   
   hypre_StructGridGhlocalSize(grid) = ghostsize;
   hypre_BoxDestroy(ghostbox);




   /* clean up */
   
   /*neighbor_ids or neighbor_boxes, neighbor_boxnums, and
     neighbor_proc_ids 
     get aliased and destroyed elsewhere (BoxNeighborDestroy) */

 


/*---------------------------------------------
  DEBUG
-----------------------------------------------*/

#if DEBUG
{
   hypre_BoxNeighbors *neighbors;
   int                *procs, *boxnums;
   int                 num_neighbors, i;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   sprintf(filename, "zgrid.%05d", my_rank);

   if ((file = fopen(filename, "a")) == NULL)
   {
      printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   fprintf(file, "\n\n============================\n\n");

   hypre_StructGridPrint(file, grid);

   neighbors = hypre_StructGridNeighbors(grid);
   num_neighbors = hypre_BoxArraySize(hypre_BoxNeighborsBoxes(neighbors));
   procs   = hypre_BoxNeighborsProcs(neighbors);
   boxnums = hypre_BoxNeighborsBoxnums(neighbors);
   fprintf(file, "num_neighbors = %d\n", num_neighbors);
   for (i = 0; i < num_neighbors; i++)
   {
      fprintf(file, "%d: (%d, %d)\n", i, procs[i], boxnums[i]);
   }

   fflush(file);
   fclose(file);
}
#endif


   return ierr;
}









/******************************************************************************
 *
 *  fillResponseStructAssembleAP
 *
 *****************************************************************************/

int
hypre_FillResponseStructAssembleAP(void *p_recv_contact_buf, 
                             int contact_size, int contact_proc, void *ro, 
                             MPI_Comm comm, void **p_send_response_buf, 
                             int *response_message_size )
{
   

   int    myid, i, index, d;
   int    size, num_objects;
   int   *send_response_buf = (int *) *p_send_response_buf;
   int   *proc_ids;
   int   *boxnums;
   
    
   hypre_BoxArray  *box_array;
   hypre_Box      *box;
   

   hypre_DataExchangeResponse  *response_obj = ro;  
   hypre_StructAssumedPart     *assumed_part = response_obj->data1;  

   int overhead = response_obj->send_response_overhead;

   /*initialize stuff */
   MPI_Comm_rank(comm, &myid );
  

   proc_ids =  hypre_StructAssumedPartMyPartitionProcIds(assumed_part);
   box_array = hypre_StructAssumedPartMyPartitionBoxes(assumed_part);
   boxnums = hypre_StructAssumedPartMyPartitionBoxnums(assumed_part);

   /*we need to send back the list of all of the boxes in our
     assumed partition along with the corresponding processor id */

   /*how many boxes and ids do we have?*/
   num_objects = hypre_StructAssumedPartMyPartitionIdsSize(assumed_part);
   /* num_objects is then how much we need to send*/
  
   
   /*check storage in send_buf for adding the information */
   /* note: we are returning objects that are 8 ints in size */

   if ( response_obj->send_response_storage  < num_objects  )
   {
      response_obj->send_response_storage =  hypre_max(num_objects, 10); 
      size =  8*(response_obj->send_response_storage + overhead);
      send_response_buf = hypre_TReAlloc( send_response_buf, int, 
                                          size);
      *p_send_response_buf = send_response_buf;    /* needed when using ReAlloc */
   }

   /* populate send_response_buf */
   index = 0;
   for (i = 0; i< num_objects; i++)
   {
      /* processor id */
      send_response_buf[index++] = proc_ids[i];
      send_response_buf[index++] = boxnums[i];
      box = hypre_BoxArrayBox(box_array, i);
      
      for (d=0; d< 3; d++)
      {
         send_response_buf[index++] = hypre_BoxIMinD(box, d);
         send_response_buf[index++] = hypre_BoxIMaxD(box, d);
      }
   }

   /* return variable */
   *response_message_size = num_objects;
   *p_send_response_buf = send_response_buf;

   return(0);
   

}


/*--------------------------------------------------------------------------
 * hypre_StructGridSetIDs
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetIDs( hypre_StructGrid *grid,
                          int   *ids )
{
   int ierr = 0;

   hypre_TFree(hypre_StructGridIDs(grid));
   hypre_StructGridIDs(grid) = ids;

   return ierr;
}
