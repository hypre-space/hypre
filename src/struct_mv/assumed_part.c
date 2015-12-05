/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.17 $
 ***********************************************************************EHEADER*/



/* This is code for the struct assumed partition 
   AHB 6/05 */

#include "headers.h"


/* these are for debugging */
#define REGION_STAT 0
#define NO_REFINE   0
#define REFINE_INFO 0

/* note: functions used in this file to determine the partition - not elsewhere - 
have names that start with hypre_AP  */


/******************************************************************************
 *
 *  Given a region, subdivide the region equally a specified number of times.
 *  For dimension d, each "level" is a subdivison of 2^d.  The box_array  
 *  is adjusted to have space for l(2^d)^level boxes
 *  We are bisecting each dimension (level) times.
 *
 *  We may want to add min. size parameter for dimension of results 
 *  regions (currently 2) - i.e., don't bisect a dimention if it will 
 *  be smaller than 2 grid points, for example
 *****************************************************************************/

HYPRE_Int hypre_APSubdivideRegion( hypre_Box *region, HYPRE_Int dim, HYPRE_Int level, 
                     hypre_BoxArray *box_array, HYPRE_Int *num_new_boxes)
{
   
   HYPRE_Int    i, j, k, width, sz;
   HYPRE_Int    total, start, div[3];
   HYPRE_Int    extra, points, count;
   HYPRE_Int    *partition[3];
   HYPRE_Int    *xpart,*ypart, *zpart;
   
   HYPRE_Int    min_gridpts; /* this should probably be an input parameter */
   

   hypre_Index      isize, imin, imax;
   hypre_Box        *box;
   


   /* if level = 0 then no dividing */
   if (!level)
   {
      hypre_BoxArraySetSize(box_array, 1);
      hypre_CopyBox(region, hypre_BoxArrayBox(box_array, 0));
      *num_new_boxes = 1;
      return hypre_error_flag;
   }
   
  /* get the size of the box in each dimension */
  /* size = # of grid points, =1 for un-used dimensions. 
     note: cell-centered! */   

   hypre_BoxGetSize(region , isize);


  /* div = num of regions in each dimension */

  /* figure out the number of regions - make sure 
     the sizes will contain the min
     number of gridpoints - or divide less in that dimension
     -we are requiring at least min_gridpts in 
     a region dimension */

   min_gridpts = 4;
   

   for (i=0; i < 3; i++) 
   {
      div[i] = 1;
      sz = hypre_IndexD(isize, i);
      for (j = 0; j< level; j++) 
      {
         if ( sz >= 2*div[i]*min_gridpts)  /* each division cuts in half */
         {
            div[i] = div[i]*2;
         }
      }
   }
   
   /* space for each partition - aliasing makes debugging easier */ 
   xpart = hypre_TAlloc(HYPRE_Int, div[0]+1);
   ypart = hypre_TAlloc(HYPRE_Int, div[1]+1);
   zpart = hypre_TAlloc(HYPRE_Int, div[2]+1);
   partition[0] = xpart;
   partition[1] = ypart;
   partition[2] = zpart;
   
   /* total number of regions to create*/
   total = 1;
   for (i=0; i < 3; i++) 
   {
      total = total*(div[i]);
   }
   
   *num_new_boxes = total;
   
   /*prepare box array*/
   hypre_BoxArraySetSize(box_array, total);
      
   /*divide each dimension */ 

     for (j=0; j < 3; j++)
     {
        start = hypre_BoxIMinD(region, j);
        partition[j][0] =  start;
        /* we want to count grid points (these are cell-centered)*/
        points = hypre_IndexD(isize, j);
        width =  points/div[j]; 
        extra =  points - width*div[j];
        for (i = 1; i < div[j]; i++)
        {
           partition[j][i] = partition[j][i-1] + width;
           if (i <= extra) partition[j][i]++; 
        }
        partition[j][div[j]] = hypre_BoxIMaxD(region, j) + 1; 
     }
          
     /* now create the new regions */
     count = 0;
     
     for (i = 0; i < (div[0]); i++)
     {
        for (j = 0; j < (div[1]); j++)
        {
           for (k = 0; k < (div[2]); k++)
           {
              hypre_SetIndex(imin, partition[0][i], partition[1][j], 
                             partition[2][k]);
              hypre_SetIndex(imax, partition[0][i+1]-1, partition[1][j+1]-1, 
                             partition[2][k+1]-1);
              box = hypre_BoxArrayBox(box_array, count);
              hypre_BoxSetExtents( box, imin, imax );
              count++;
              
           }
        }
     }

     
     /* clean up */
     hypre_TFree(xpart);
     hypre_TFree(ypart);
     hypre_TFree(zpart);
     
     return hypre_error_flag;
   
   
}


/******************************************************************************
 *
 *   Given a list of regions, find out how many of *my* boxes are contained 
 *   in each region 
 *
 *****************************************************************************/

HYPRE_Int hypre_APFindMyBoxesInRegions( hypre_BoxArray *region_array, 
                                  hypre_BoxArray *my_box_array, 
                                  HYPRE_Int **p_count_array, double **p_vol_array)
{

   HYPRE_Int            i,j, d, grow;
   HYPRE_Int            num_boxes, num_regions;
   HYPRE_Int           *count_array;
   double              *vol_array;
   HYPRE_Int           grow_array[6];

   hypre_Box           *my_box, *result_box, *grow_box;
   hypre_Box           *region;
  

   num_boxes =  hypre_BoxArraySize(my_box_array);
   num_regions = hypre_BoxArraySize(region_array);

   count_array = *p_count_array;
   vol_array = *p_vol_array;
   

   /* may need to add some sort of sorting to make this
      more efficient  - though we shouldn't have many regions*/

   /* note: a box can be in more than one region */

   result_box = hypre_BoxCreate();
   grow_box = hypre_BoxCreate();

   for (i=0; i< num_regions; i++)
   {
      count_array[i] = 0;     
      vol_array[i] = 0.0;

      region = hypre_BoxArrayBox(region_array, i);

      for (j = 0; j< num_boxes; j++) 
      {
         my_box = hypre_BoxArrayBox(my_box_array, j);
         /* check if its a zero volume box - if so it still need to be counted - so 
            expand till vol. is non-zero then intersect */ 
	 if (hypre_BoxVolume(my_box) == 0)   
	 {
	   hypre_CopyBox(my_box, grow_box);
	   for (d = 0; d < 3; d++)
	   {
	     if(!hypre_BoxSizeD(my_box, d))
	     {
	       grow = (hypre_BoxIMinD(my_box, d) - hypre_BoxIMaxD(my_box, d) + 1)/2;
	       grow_array[2*d] = grow;
	       grow_array[2*d+1] = grow;
	     }
	     else
	     {
		 grow_array[2*d] = 0;
		 grow_array[2*d+1] = 0;
	     }
	   }
	   /* expand the grow box (leave our box untouched)*/
	   hypre_BoxExpand(grow_box, grow_array);
          /* do they intersect?*/   
	   hypre_IntersectBoxes( grow_box, region, result_box);
	 }
	 else
	 {
	   /* do they intersect?*/     
	   hypre_IntersectBoxes( my_box, region, result_box);
	 }
         if (  hypre_BoxVolume(result_box) > 0 )
         {
            count_array[i]++;            
            vol_array[i] += (double) hypre_BoxVolume(result_box);
         }
      }
   }


  /* clean up */
   hypre_BoxDestroy(result_box);
   hypre_BoxDestroy(grow_box);      

   /* output */
   *p_count_array = count_array;
   *p_vol_array = vol_array;
   
 
   return hypre_error_flag;

}

/******************************************************************************
 *
 *   Given a list of regions, find out how many global boxes are contained 
 *   in each region.
 * 
 *   assumes that p_count_array and p_vol_array have already been allocated    
 *
 *****************************************************************************/

HYPRE_Int hypre_APGetAllBoxesInRegions( hypre_BoxArray *region_array, 
                                  hypre_BoxArray *my_box_array, 
                                  HYPRE_Int **p_count_array, double **p_vol_array, 
                                  MPI_Comm comm )
{
   
   HYPRE_Int      i;
   HYPRE_Int     *count_array;
   HYPRE_Int      num_regions;
   HYPRE_Int     *send_buf_count;
   double  *send_buf_vol;
   double  *vol_array;
   double  *dbl_vol_and_count;
   

   count_array = *p_count_array;
   vol_array = *p_vol_array;
   
   
   /*first get a count and volume of my boxes in each region */
   num_regions = hypre_BoxArraySize(region_array);
   
   send_buf_count = hypre_CTAlloc(HYPRE_Int, num_regions);
   send_buf_vol =  hypre_CTAlloc(double, num_regions*2); /*allocate double*/

   dbl_vol_and_count =  hypre_CTAlloc(double, num_regions*2); /*allocate double*/

   hypre_APFindMyBoxesInRegions( region_array, my_box_array, &send_buf_count, 
                                 &send_buf_vol);


   /* copy ints to doubles so we can do one all_reduce */
   for (i=0; i< num_regions; i++)
   {
      send_buf_vol[num_regions+i] = (double) send_buf_count[i];
   }

   hypre_MPI_Allreduce(send_buf_vol, dbl_vol_and_count, num_regions*2, hypre_MPI_DOUBLE, 
                 hypre_MPI_SUM, comm);

   /*unpack*/
   for (i=0; i< num_regions; i++)
   { 
      vol_array[i] = dbl_vol_and_count[i];
      count_array[i] = (HYPRE_Int) dbl_vol_and_count[num_regions+i];
   }


   /* clean up*/
   hypre_TFree(send_buf_count);
   hypre_TFree(send_buf_vol);
   hypre_TFree(dbl_vol_and_count);
   

   /* output */
   *p_count_array = count_array;
   *p_vol_array = vol_array;
   
   return hypre_error_flag;
   
}


/******************************************************************************
 *
 *   Given a list of regions, shrink regions according to min and max extents
 *
 *****************************************************************************/

HYPRE_Int hypre_APShrinkRegions( hypre_BoxArray *region_array, 
                           hypre_BoxArray *my_box_array, MPI_Comm comm )
{
   
   /* These regions should all be non-empty at the global level*/


   HYPRE_Int            i,j, d;
   HYPRE_Int            num_boxes, num_regions;
   HYPRE_Int            *indices, *recvbuf;
   HYPRE_Int            count = 0;
   HYPRE_Int            grow, grow_array[6];
   

   hypre_Box           *my_box, *result_box, *grow_box;
   hypre_Box           *region;
   hypre_Index          imin, imax;
   

   num_boxes =  hypre_BoxArraySize(my_box_array);
   num_regions = hypre_BoxArraySize(region_array);

   indices = hypre_CTAlloc(HYPRE_Int, num_regions*6);
   recvbuf =  hypre_CTAlloc(HYPRE_Int, num_regions*6);

   result_box = hypre_BoxCreate();

   /* allocate for a grow box */
   grow_box = hypre_BoxCreate();

   /* Look locally at my boxes */
   /* for each region*/
   for (i=0; i< num_regions; i++)
   {
      count = 0; /*number of my boxes in this region */
      
      /* get the region box*/ 
      region = hypre_BoxArrayBox(region_array, i);
    
      
      /*go through each of my local boxes */
      for (j = 0; j< num_boxes; j++) 
      {
         my_box = hypre_BoxArrayBox(my_box_array, j);

         /* check if its a zero volume box - if so it still need to be checked - so 
            expand till vol. is non-zero then intersect */ 
	 if (hypre_BoxVolume(my_box) == 0)   
	 {
            hypre_CopyBox(my_box, grow_box);
            for (d = 0; d < 3; d++)
            {
               if(!hypre_BoxSizeD(my_box, d))
               {
                  grow = (hypre_BoxIMinD(my_box, d) - 
                          hypre_BoxIMaxD(my_box, d) + 1)/2;
                  grow_array[2*d] = grow;
                  grow_array[2*d+1] = grow;
               }
               else
               {
                  grow_array[2*d] = 0;
                  grow_array[2*d+1] = 0;
               }
            }
            /* expand the grow box (leave our box untouched)*/
            hypre_BoxExpand(grow_box, grow_array);
            /* do they intersect?*/   
            hypre_IntersectBoxes( grow_box, region, result_box);
	 }
	 else
	 {
            /* do they intersect?*/     
            hypre_IntersectBoxes( my_box, region, result_box);
         }
    
         if (  hypre_BoxVolume(result_box) > 0 ) /* they intersect*/
         {

            if (!count) /* set min and max for first box */
            {
               for (d = 0; d < 3; d++)
               {
                  indices[i*6+d] = hypre_BoxIMinD(result_box, d);
                  indices[i*6+3+d] = hypre_BoxIMaxD(result_box, d);
               } 
               
            }
            
            count++;

            /* boxes intersect - so get max and min extents 
                of the result box  (this keeps the bounds inside
                the region */
            for (d = 0; d < 3; d++)
            {
               indices[i*6+d] = hypre_min(indices[i*6+d], 
                                          hypre_BoxIMinD(result_box, d));
               indices[i*6+3+d] = hypre_max(indices[i*6+3+d], 
                                            hypre_BoxIMaxD(result_box, d));
            }
         }
      }
      /* now if we had no boxes in that region, set the min to the max 
         extents of the region and the max to the min! */ 

      if (!count) 
      {
         for (d = 0; d < 3; d++)
         {
            indices[i*6+d] = hypre_BoxIMaxD(region, d);
            indices[i*6+3+d] = hypre_BoxIMinD(region, d);
         } 
      }
      
      /*negate max indices for the allreduce */
      /* note: min(x)= -max(-x) */
      
      for (d = 0; d < 3; d++)
      {
         indices[i*6+3+d] = -indices[i*6+3+d];
      } 
                     

   }

   /* Now do an allreduce to get the global information */

   /* now do an all reduce on the size and volume */   
   hypre_MPI_Allreduce(indices, recvbuf, num_regions*6, HYPRE_MPI_INT, hypre_MPI_MIN, comm);

   /*now unpack the "shrunk" regions */
    /* for each region*/
   for (i=0; i< num_regions; i++)
   {
    /* get the region box*/ 
      region = hypre_BoxArrayBox(region_array, i);

      /*resize the box*/  
      hypre_SetIndex(imin, recvbuf[i*6],  recvbuf[i*6+1], recvbuf[i*6+2]);
      hypre_SetIndex(imax, -recvbuf[i*6+3],  -recvbuf[i*6+4], -recvbuf[i*6+5]);

      hypre_BoxSetExtents(region, imin, imax );
      
      /*add: check to see whether any shrinking is actually occuring*/  

   }
   

   /* clean up */
   hypre_TFree(recvbuf);
   hypre_TFree(indices);
   hypre_BoxDestroy(result_box);
   hypre_BoxDestroy(grow_box);
   

   return hypre_error_flag;
   

   
}



/******************************************************************************
 *
 *   Given a list of regions, eliminate empty regions 
 *
 *****************************************************************************/



HYPRE_Int hypre_APPruneRegions( hypre_BoxArray *region_array,  HYPRE_Int **p_count_array, 
                          double **p_vol_array )
{


  /*-------------------------------------------
    *  parameters: 
    *   
    *    region_array            = assumed partition regions
    *    count_array             = number of global boxes in each region 
    *-------------------------------------------*/


   HYPRE_Int          i, j;
   HYPRE_Int          num_regions;
   HYPRE_Int          count;
   HYPRE_Int          *delete_indices;

   HYPRE_Int          *count_array;
   double             *vol_array;
   

   count_array = *p_count_array;
   vol_array = *p_vol_array;
      

   num_regions = hypre_BoxArraySize(region_array);
   delete_indices = hypre_CTAlloc(HYPRE_Int, num_regions);
   count = 0;
   

   /* delete regions with zero elements */
   for (i=0; i< num_regions; i++) 
   {
      if (count_array[i] == 0)
      {
         delete_indices[count++] = i;      
      }
      
   }
      
   hypre_DeleteMultipleBoxes( region_array, delete_indices, count );
   
   /* adjust count and volume arrays */
   if (count > 0)
   {
         j=0;
         for (i = delete_indices[0]; (i + j) < num_regions; i++)
         {
            if (j < count)
            {
               while ((i+j) == delete_indices[j])
               {
                  j++; /*increase the shift*/
                  if (j == count) break;
               }
            }
            vol_array[i] = vol_array[i+j];
            count_array[i] = count_array[i+j];
         }
   }
   
   /* clean up */ 
   hypre_TFree(delete_indices);

   /* return variables */
   *p_count_array = count_array;
   *p_vol_array = vol_array;

   return hypre_error_flag;

}                                                


/******************************************************************************
 *
 *   Given a list of regions, and corresponding volumes contained in regions
 *   subdivide some of the regions that are not full enough
 *   
 *****************************************************************************/



HYPRE_Int hypre_APRefineRegionsByVol( hypre_BoxArray *region_array,  double *vol_array, 
                        HYPRE_Int max_regions, double gamma, HYPRE_Int dim, HYPRE_Int *return_code, 
                                MPI_Comm comm)
{


   HYPRE_Int          i, count, loop;
   HYPRE_Int          num_regions, init_num_regions;
   HYPRE_Int         *delete_indices;
   
   double            *fraction_full;
   HYPRE_Int         *order;
   HYPRE_Int          myid, num_procs, est_size;
   HYPRE_Int          new;
   /* HYPRE_Int  regions_intact; */

      
   hypre_BoxArray    *tmp_array;
   hypre_Box         *box;

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);

   num_regions = hypre_BoxArraySize(region_array);

   if (!num_regions)
   {
      *return_code = 1; /* no regions - so no subdividing*/
      return hypre_error_flag;
      
   }
   
   fraction_full = hypre_CTAlloc(double,  num_regions); 
   order = hypre_CTAlloc(HYPRE_Int,  num_regions); 
   delete_indices = hypre_CTAlloc(HYPRE_Int,  num_regions); 
  
   for (i= 0; i< num_regions; i++)
   {
      box = hypre_BoxArrayBox(region_array, i);
      fraction_full[i] = (vol_array[i])/((double) hypre_doubleBoxVolume(box));
      order[i] = i; /*this is what order to access the boxes */     
   }
      
   /* want to refine the regions starting with those that are the least full*/   
   /* sort the fraction AND the index */
   hypre_qsort2(order, fraction_full, 0, num_regions-1);
   

   /*now we can subdivide any that are not full enough*/
   /* when this is called, we know that size < max_regions */
   /* it is ok to subdivde such that we have slightly more
      regions that max_region, but we do not want more regions
      than processors */ 
 
   tmp_array = hypre_BoxArrayCreate(0);
   count = 0; /*how many regions subdivided */
   loop = 0; /* counts the loop number */   
   init_num_regions = num_regions;
   *return_code = 1; /* indicates all regions are at least gamma full  - 
                        and no subdividing occured*/

   while (fraction_full[loop] < gamma) 
   {

      /*some subdividing occurred*/
      *return_code = 2;

      /*we can't let the number of regions exceed the number
        of processors - only an issue for small proc numbers */
      est_size = num_regions + pow(2, dim) - 1;
      if (est_size > num_procs)
      {
         if (loop==0) 
         {
            *return_code = 4; /* some are less than gamma full, but we cannot 
                                 further subdivide due to max *processors* 
                                 limit (no subdividing occured)*/              
         }
         
         else
         {
            *return_code = 3;/* some subdividing occured, but there are some 
                                regions less than gamma full (max reached) 
                                that were not subdivided */
         }
         
         break;
      }
      
      box = hypre_BoxArrayBox(region_array, order[loop]);
      hypre_APSubdivideRegion(box, dim, 1, tmp_array, &new);

      if (new > 1) /* if new =1 then no subdividing occured */
      {
         num_regions = num_regions + new - 1; /* the orginal will be deleted*/
         
         delete_indices[count] = order[loop];
         count++; /* number of regions subdivided */
 
         /*append tmp_array to region array*/
         hypre_AppendBoxArray( tmp_array, region_array);

      }

     

      /* if we are on the last region */
      if  ((loop+1) == init_num_regions) 
      {
         break; 
      }

      /* clear tmp_array for next loop */
      hypre_BoxArraySetSize(tmp_array, 0);


      /* if we now have too many regions - don't want to subdivide anymore*/
      if (num_regions >= max_regions) 
      {
         /* see if next regions satifies gamma */
         if (fraction_full[order[loop+1]] > gamma)
         {
            *return_code = 5;/* all regions less than gamma full have been 
                                subdivided (and we have reached max)*/
         }
         else 
         {
            *return_code = 3; /* there are some regions less than gamma 
                                 full (but max is reached) */
         }
         break;
      }

      loop++; /* increment to repeat loop */
   }


   if (count ==0 )
   {
      *return_code = 1;
      /* no refining occured so don't do any more*/
   }
   else
   {

      /* so we subdivided count regions */
      
      /*delete the old regions*/  
      qsort0(delete_indices, 0, count-1); /* put delete indices in asc. order */
      hypre_DeleteMultipleBoxes( region_array, delete_indices, count );

   }

   /* TO DO: number of regions intact (beginning of region array is intact)- 
     may return this eventually*/
   /* regions_intact = init_num_regions - count;*/
   
  
   /* clean up */
   hypre_TFree(fraction_full);
   hypre_TFree(order);
   hypre_TFree(delete_indices);
   hypre_BoxArrayDestroy(tmp_array);
   


   return hypre_error_flag;
   
   
}


/******************************************************************************
 *
 *   Construct an assumed partition   
 *
 *   8/06 - changed the assumption that
 *   that the local boxes have boxnums 0 to num(local_boxes)-1 (now need to pass in
 *   ids)
 *
 *   10/06 - changed - no longer need to deal with negative boxes as this is 
 *   used through the box manager
 *
 *   3/6 - don't allow more regions than boxes (unless global boxes = 0)and
 *         don;'t partition into more procs than global number of boxes
 *****************************************************************************/


HYPRE_Int hypre_StructAssumedPartitionCreate(HYPRE_Int dim, hypre_Box *bounding_box, 
                                       double global_boxes_size, 
                                       HYPRE_Int global_num_boxes,
                                       hypre_BoxArray *local_boxes, HYPRE_Int *local_boxnums,
                                       HYPRE_Int max_regions, HYPRE_Int max_refinements, 
                                       double gamma,
                                       MPI_Comm comm, 
                                       hypre_StructAssumedPart **p_assumed_partition)
{
   
   HYPRE_Int          i, j, d;
   HYPRE_Int          size;
   HYPRE_Int          myid, num_procs;
   HYPRE_Int          num_proc_partitions;
   HYPRE_Int          count_array_size;
   HYPRE_Int          *count_array=NULL;
   double             *vol_array=NULL, one_volume;
   HYPRE_Int          return_code;
   HYPRE_Int          num_refine;
   HYPRE_Int          total_boxes, proc_count, max_position, tmp_num;
   HYPRE_Int          *proc_array=NULL;
   HYPRE_Int          i1, i2, i11, i22, pos1, pos2, pos0;
   HYPRE_Int          tmp, ti1, ti2, t_tmp, t_total;
   double             f1, f2, r,x_box, y_box, z_box, dbl_vol;
   HYPRE_Int          initial_level;

   

   hypre_Index        div_index;
   hypre_BoxArray     *region_array;
   hypre_Box          *box, *grow_box;

   hypre_StructAssumedPart *assumed_part;
   
 
   HYPRE_Int   proc_alloc, count, box_count;
   HYPRE_Int   max_response_size;
   HYPRE_Int  *response_buf = NULL, *response_buf_starts=NULL;
   HYPRE_Int  *tmp_box_nums = NULL, *tmp_proc_ids = NULL;
   HYPRE_Int  *proc_array_starts=NULL;
   

   hypre_BoxArray              *my_partition;
   hypre_DataExchangeResponse  response_obj;

   HYPRE_Int  *contact_boxinfo;
   HYPRE_Int  index;
   

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   
/* special case where ther are no boxes in the grid */
   if (global_num_boxes == 0)
   {
      region_array = hypre_BoxArrayCreate(0); 
      assumed_part = hypre_TAlloc(hypre_StructAssumedPart, 1);
      
      hypre_StructAssumedPartRegions(assumed_part) = region_array; 
      hypre_StructAssumedPartNumRegions(assumed_part) = 0;
      hypre_StructAssumedPartDivisions(assumed_part) =  NULL;                                             
   
      hypre_StructAssumedPartProcPartitions(assumed_part) = 
         hypre_CTAlloc(HYPRE_Int, 1); 
      hypre_StructAssumedPartProcPartition(assumed_part, 0) = 0;
      hypre_StructAssumedPartMyPartition(assumed_part) =  NULL;
      hypre_StructAssumedPartMyPartitionBoxes(assumed_part) 
         = hypre_BoxArrayCreate(0);
      hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = 0;
      hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = 0;
      hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part) = 0;
      hypre_StructAssumedPartMyPartitionBoxnums(assumed_part) 
         = NULL;
      hypre_StructAssumedPartMyPartitionProcIds(assumed_part) 
         = NULL;
      *p_assumed_partition = assumed_part;

     return hypre_error_flag;
   }
   
   /* end special case of zero boxes*/


   /*FIRST DO ALL THE GLOBAL PARTITION INFO */

   /*initially divide the bounding box*/


   if (!hypre_BoxVolume(bounding_box) && global_num_boxes)
   {
      hypre_error(HYPRE_ERROR_GENERIC);
      if (myid ==0) hypre_printf("ERROR: the bounding box has zero volume AND there are grid boxes  ");
   }
      

   /*first modify any input parameters if necessary */

   /*don't want the number of regions exceeding the number of processors*/
   /* note: this doesn't change the value in the caller's code */
    max_regions = hypre_min(num_procs, max_regions);


    /* don't want more regions than boxes either */
     if (global_num_boxes) max_regions = hypre_min(global_num_boxes, max_regions);

 
    /* start with a region array of size 0*/
    region_array = hypre_BoxArrayCreate(0); 
  
    /*if the bounding box is sufficiently covered by boxes then we will
      just have one region - the bounding box  - otherwise we will subdivide 
      probably just start with one level of division */ 

    /* one_volume = (double) hypre_BoxSizeX(bounding_box) * 
       (double) hypre_BoxSizeY(bounding_box) * (double) hypre_BoxSizeZ(bounding_box); */
    
    one_volume = (double) hypre_doubleBoxVolume(bounding_box);
    

    if( ((global_boxes_size/one_volume) > gamma) ||  
        ( global_num_boxes > one_volume) || global_num_boxes == 0 )
    {
       /*don't bother with any refinements - we are full enough - 
         or we have a small bounding box
         and we are not full because of neg. boxes */ 
       initial_level = 0;
       max_refinements = 0;
       
    }
    else
    {
       initial_level = 1; /*we could let this be an input parameter 
                            - but I think 1 division is probably
                            sufficient */

       /* start with the specified intial_levels for the original domain  - 
          unless we have a smaller number of procs */
       for (i= 0; i < initial_level; i++)
       {
          if ( pow(2,initial_level*dim) > num_procs) initial_level --;

          if (!initial_level) max_refinements = 0; /* we will not be able to do 
                                                      any refinements due to the
                                                      number of processors*/        
       }
    }
 

#if NO_REFINE
    max_refinements = 0;
    initial_level = 0;
#endif



#if REFINE_INFO
    if (myid ==0)
    {
       hypre_printf("gamma =  %g\n", gamma);
       hypre_printf("max_regions =  %d\n", max_regions);
       hypre_printf("max_refinements =  %d\n", max_refinements);
       hypre_printf("initial level =  %d\n", initial_level);
    }
    
#endif    

   
    /*divide the bounding box */    
    hypre_APSubdivideRegion(bounding_box, dim, initial_level, region_array, 
                            &size); 
    /* if no subdividing occured (becuz too small) then don't try to refine */     
    if (initial_level > 0 && size ==1) max_refinements = 0;
    
   /*need space for count and volume */
   size = hypre_BoxArraySize(region_array);
   count_array_size = size; /* memory allocation size */
   count_array = hypre_CTAlloc(HYPRE_Int,  size);
   vol_array =  hypre_CTAlloc(double,  size); 
   
   /* how many boxes are in each region (global count) and what the volume is*/
   hypre_APGetAllBoxesInRegions( region_array, local_boxes, &count_array, 
                                 &vol_array, comm );
  
   /* don't do any initial prune and shrink if we have only
      one region and we can't do any refinements */

   if ( !(size ==1 && max_refinements == 0))
   {
      
      /* get rid of regions with no boxes (and adjust count and vol arrays) */
      hypre_APPruneRegions( region_array, &count_array, &vol_array);
      
      /*shrink the extents*/
      hypre_APShrinkRegions( region_array, local_boxes, comm); 
    
   }
   
    /* keep track of refinements */ 
    num_refine = 0;
    

    /* now we can keep refining by dividing the regions that are not
       full enough and eliminating empty regions */
    while (( hypre_BoxArraySize(region_array)< max_regions) && 
           (num_refine < max_refinements))
    {
       num_refine++;
       
       /* now calculate how full the regions are and subdivide the 
          least full as needed */


       size = hypre_BoxArraySize(region_array);

       /* divide regions that are not full enough */
       hypre_APRefineRegionsByVol( region_array, vol_array, 
                                   max_regions, 
                                   gamma, dim, &return_code, comm );

     
       if (return_code == 1 || return_code == 4) /* 1 = all regions are at least 
                                                    gamma full - no subdividing 
                                                    occured */
       {                                         /* 4 = no subdividing occured 
                                                    due to num_procs limit on 
                                                    regions*/
          break; 
       }
       /* this is extraneous I think */     
       if (size == hypre_BoxArraySize(region_array))
       {
          /* no dividing occured -exit the loop */
           break; 
       }
       
 
       size = hypre_BoxArraySize(region_array);
       if (size >  count_array_size)
       {
          count_array = hypre_TReAlloc(count_array, HYPRE_Int,  size); 
          vol_array =  hypre_TReAlloc(vol_array, double,  size); 
          count_array_size =size;
        }

       /* FUTURE MOD - just count and prune and shrink in the modified regions
          from refineRegionsByVol - these are the last regions in the array */ 

       /* num boxes are in each region (global count) and what the volume is*/
       hypre_APGetAllBoxesInRegions( region_array, local_boxes, &count_array, 
                                     &vol_array, comm );

       /* get rid of regions with no boxes (and adjust count and vol arrays) */
       hypre_APPruneRegions( region_array, &count_array, &vol_array);

       /*shrink the extents*/
       hypre_APShrinkRegions( region_array, local_boxes, comm);        


       /* actually 3 and 5 may be ok after pruning..but if no pruning then
          exit the loop */  

       if ( (return_code == 3 || return_code == 5) 
            && size ==hypre_BoxArraySize(region_array)  ) 
                                                   /* 5 = all regions < gamma full
                                                     were subdivided and max 
                                                     reached */
                                                  /* 3 = some regions were divided 
                                                     (not all that needed) and max 
                                                     reached */
       {
          break;
       }

    }
    /* end of refinements */    


    /*error checking */  
    if (global_num_boxes)
    {
       hypre_ForBoxI(i, region_array)
       {
          if (hypre_BoxVolume(hypre_BoxArrayBox(region_array, i))==0)
          {
             hypre_error(HYPRE_ERROR_GENERIC);
             if (myid ==0) hypre_printf("ERROR: a region has zero volume!  "
                                  "(this should never happen unless there are 0 global boxes)\n");
          }
       }
    }
    


#if REGION_STAT
    if (myid == 0) hypre_printf("myid = %d: %d REGIONS (after refining %d times accord. "
                          "to vol\n", 
                          myid, hypre_BoxArraySize(region_array), num_refine);
#endif



#if REGION_STAT
    if (myid ==0)
    {
       hypre_printf("myid = %d, REGIONS (after refining %d times accord. to vol\n", 
              myid, num_refine);
    
       hypre_ForBoxI(i, region_array)
          {
             box = hypre_BoxArrayBox(region_array, i);
             hypre_printf("myid = %d, %d:  (%d, %d, %d)  x  (%d, %d, %d)\n",
                    myid, i,
                    hypre_BoxIMinX(box),
                    hypre_BoxIMinY(box),
                    hypre_BoxIMinZ(box),
                    hypre_BoxIMaxX(box),
                    hypre_BoxIMaxY(box),
                    hypre_BoxIMaxZ(box));
          }
    }
#endif


    hypre_TFree(vol_array);


/* ------------------------------------------------------------------------*/
    /*now we have the regions - construct the assumed partition */

    size = hypre_BoxArraySize(region_array);
    assumed_part = hypre_TAlloc(hypre_StructAssumedPart, 1);
    hypre_StructAssumedPartRegions(assumed_part) = region_array; 
   /*the above is aliased - so don't destroy region_array in this function */
    hypre_StructAssumedPartNumRegions(assumed_part) = size;
    hypre_StructAssumedPartDivisions(assumed_part) =  hypre_CTAlloc(hypre_Index, 
                                                                    size);
      
    /* first determine which processors (how many) to assign to each region*/
    proc_array = hypre_CTAlloc(HYPRE_Int,  size); 
    total_boxes = 0;        /* this is different than the total number of 
                               boxes as some boxes can be in more than
                               one region */
    proc_count = 0;
    d = -1;
    max_position = -1;
    /* calculate total number of boxes in the regions */
    for (i=0; i < size; i++) 
    {
       total_boxes += count_array[i];
    }
    /* calculate the fraction of actual boxes in each region, mult. by total number
       of proc partitons desired, put result in proc_array to
       assign  each region a number of processors proportional to the fraction
       of boxes */

    /* 3/6 - put a limit on the number of proc partitions - not larger
       than the total of boxes in the regions (at coarse levels may be
       many more procs than boxes - this should minimize some
       communication)*/
    num_proc_partitions = hypre_min(num_procs, total_boxes);
    

    for (i=0; i < size; i++) 
    {
 
       if (!total_boxes) /* in case there are no boxes in a grid */
       {
          proc_array[i] = 0;
       }
       else
       {
          proc_array[i] = (HYPRE_Int) hypre_round(((double)count_array[i]/ 
                                             (double)total_boxes)* (double) num_proc_partitions);
       }
       
       box =  hypre_BoxArrayBox(region_array, i); 
       /* vol = hypre_BoxVolume(box); */
       dbl_vol = hypre_doubleBoxVolume(box);
       
       /* can't have any zeros! */
       if (!proc_array[i]) proc_array[i] = 1;


       if (dbl_vol < (double) proc_array[i]) 
       {
          proc_array[i] = (HYPRE_Int) dbl_vol; /*don't let the the number of procs be 
                                           greater than the volume - safe to cast back to HYPRE_Int if
                                           this is true - then vol is not overflowing */
       }
       
       proc_count += proc_array[i];
       if (d < proc_array[i])
       {
          d = proc_array[i];
          max_position = i;
       }
       
       /* if (myid == 0) hypre_printf("proc array[%d] = %d\n", i, proc_array[i]);*/
    }

    
    hypre_TFree(count_array);
  


    /* adjust such that num_proc_partitions = proc_count (they should be close) */
    /* a processor is only assigned to ONE region*/

    /* if we need a few more processors assigned 
       in proc_array for proc_count to = num_proc_partitions
       (it is ok if we have fewer procs in proc_array
       due to volume constraints */
    while (num_proc_partitions > proc_count)
    {
       proc_array[max_position]++;
       
       if ( (double) proc_array[max_position] >  
           hypre_doubleBoxVolume(hypre_BoxArrayBox(region_array, max_position))) 
       {
          proc_array[max_position]--;
          break; /* some processors won't get assigned partitions */
       }
      proc_count++;
    }

    /* if we we need fewer processors in proc_array*/
    i = 0;
    while (num_proc_partitions < proc_count)
    {
       if (proc_array[max_position] != 1)
       {
          proc_array[max_position]--;
       }
       else
       {
          while (proc_array[i] <=1 && i < size) /* size is the number of regions */
          {
             i++;
          }
           proc_array[i]--;         
       }
       proc_count--;
    }
    /* the above logic would be flawed IF we allowed more regions 
       than processors, but this is not allowed! */


     /*now we have the number of processors in each region so create the 
       processor partition */
     /* size = # of regions */ 
     hypre_StructAssumedPartProcPartitions(assumed_part) = 
                                                   hypre_CTAlloc(HYPRE_Int, size+ 1); 
     hypre_StructAssumedPartProcPartition(assumed_part, 0) = 0;
     for (i=0; i< size; i++)
     {
        hypre_StructAssumedPartProcPartition(assumed_part, i+1) =  
           hypre_StructAssumedPartProcPartition(assumed_part, i) + proc_array[i];
     }


    /* now determine the NUMBER of divisions in the x, y amd z dir according
       to the number or processors assigned to the region */
     
     /* FOR EACH REGION */
     for (i = 0; i< size; i++)
     {
        proc_count =  proc_array[i];
        box =  hypre_BoxArrayBox(region_array, i);
        x_box =  (double) hypre_BoxSizeX(box);
        y_box =  (double) hypre_BoxSizeY(box);
        z_box =  (double) hypre_BoxSizeZ(box);
 

        /* this could be re-written so that the two 
           dimensions use the same code*/
        if (dim == 2)
        {
           
           /*dimension 2  */
           if  ( (x_box/y_box) > 1.0 )
           {
              f1 = x_box/y_box;         
              pos0 = 1; /* find y first */
              pos1 = 0;
              
           }
           else 
           {
              f1 = y_box/x_box;  
              pos0 = 0; /* find x first */
              pos1 = 1;
              
           }
           i1 = (HYPRE_Int) ceil(f1);
           i11 = (HYPRE_Int) floor(f1);

           /*case 1*/
           r = (double) proc_count/ (double) i1;
           ti1 = i1;
           t_tmp = (HYPRE_Int) ceil(sqrt(r));
           t_total =  t_tmp*t_tmp*i1;
           
           /* is this better?*/ 
           r = (double) proc_count/ (double) i11;
           tmp = (HYPRE_Int) ceil(sqrt(r));          
           tmp_num =  tmp*tmp*i11;
           if (tmp_num < t_total && tmp_num >= proc_count)
           {
              ti1 = i11;
              t_tmp = tmp;
              t_total = tmp_num;
           }
                   
           if ( t_total < proc_count ) 
           {
              hypre_error(HYPRE_ERROR_GENERIC);
              hypre_printf("ERROR: (this shouldn't happen) "
                     "the struct assumed partition doesn't"
                     "have enough partitions!!!\n");
           }
           hypre_IndexD(div_index, pos0) = t_tmp; 
           hypre_IndexD(div_index, pos1) = t_tmp*ti1; 
           hypre_IndexZ(div_index) = 1;

        }
        else
        {
                   
           tmp = hypre_min(x_box, y_box);
           tmp = hypre_min(tmp, z_box);
           
           if (tmp == x_box) /* x is smallest*/
           {
              f1 = (y_box/x_box);
              f2 = (z_box/x_box);
              pos0 = 0;
              pos1 = 1;
              pos2 = 2;
 
           }
           else if (tmp == y_box) /*y is smallest */
           {
              f1 = (x_box/y_box);
              f2 = (z_box/y_box);
              pos0 = 1;
              pos1 = 0;
              pos2 = 2;
           }
           else /*z is smallest */
           {
              f1 = (x_box/z_box);
              f2 = (y_box/z_box);
              pos0 = 2;
              pos1 = 0;
              pos2 = 1;
           }

           i1 = (HYPRE_Int) ceil(f1);
           i2 =  (HYPRE_Int) ceil(f2);
           i11 = (HYPRE_Int) floor(f1);
           i22 = (HYPRE_Int) floor(f2);
           
           /* case 0 - this will give a number of divisions > proc count */    
           r = (double) proc_count / (double) (i2*i1);
           tmp = (HYPRE_Int) ceil(pow(r,(1.0/3.0)));
           tmp_num = tmp*tmp*tmp*i1*i2;

           ti1 = i1;
           ti2 = i2;
           t_tmp = tmp;
           t_total = tmp_num;
         
           /* can we do better ? */  
           /*case 1: i11 and i2 */           
           r = (double) proc_count / (double) (i2*i11);
           tmp = (HYPRE_Int) ceil(pow(r,(1.0/3.0)));
           tmp_num = tmp*tmp*tmp*i11*i2;  
           if (tmp_num < t_total && tmp_num >= proc_count) 
           {
              ti1 = i11;
              ti2 = i2;
              t_tmp = tmp;
              t_total = tmp_num;
           }
           /*case 2: i11 and i22 */           
           r = (double) proc_count / (double) (i22*i11);
           tmp = (HYPRE_Int) ceil(pow(r,(1.0/3.0)));
           tmp_num = tmp*tmp*tmp*i11*i22;  
           if (tmp_num < t_total  && tmp_num >= proc_count) 
           {
              ti1 = i11;
              ti2 = i22;
              t_tmp = tmp;
              t_total = tmp_num;
           }
           /*case 3: i1 and i22 */           
           r = (double) proc_count / (double) (i22*i1);
           tmp = (HYPRE_Int) ceil(pow(r,(1.0/3.0)));
           tmp_num = tmp*tmp*tmp*i1*i22;  
           if (tmp_num < t_total  && tmp_num >= proc_count) 
           {
              ti1 = i1;
              ti2 = i22;
              t_tmp = tmp;
              t_total = tmp_num;
           }
          
           if ( t_total < proc_count ) 
           {
              hypre_error(HYPRE_ERROR_GENERIC);
              hypre_printf("ERROR: (this shouldn't happen)"
                     " the struct assumed partition "
                     "doesn't have enough partitions!!!\n");
           }
           
     
           /*set final division */           
           hypre_IndexD(div_index, pos0) = t_tmp; 
           hypre_IndexD(div_index, pos1) = t_tmp*ti1; 
           hypre_IndexD(div_index, pos2) = t_tmp*ti2; 

        }
        /* end of finding divisions */       


        /*see if we can further reduce the number of assumed regions -
          here we are effecting the aspect ratio but are avoiding an 
          excessive number of assumed regions
          per processor.  If the aspect ratio of the object is poor then 
          there can be many more assumed box divisions than processors for
          small numbers of processors - so this should only iterate multiple 
          times for poor aspect ratios*/

        while (1)
        {
           
           hypre_MaxIndexPosition(div_index, &d);
           tmp =  hypre_IndexD(div_index, d);
           if ((tmp-1)==0) break;
           hypre_IndexD(div_index, d) = tmp - 1;
           tmp_num = (hypre_IndexX(div_index))*(hypre_IndexY(div_index))*
              (hypre_IndexZ(div_index));
           if (tmp_num < proc_count) 
           {
              /* restore and exit */
              hypre_IndexD(div_index, d) = tmp;
              break;
           }
        } 
        
          /*check to see that we won't have more than 2 assumed partitions per proc 
            this should never happen according to my proof :) */   
        tmp_num = (hypre_IndexX(div_index))*(hypre_IndexY(div_index))*
           (hypre_IndexZ(div_index));
        if ( tmp_num > 2*proc_count ) 
        {
           hypre_error(HYPRE_ERROR_GENERIC);
           hypre_printf("ERROR: the struct assumed partition "
                  "has more than 2*procs (this "
                  "shouldn't happen) myid = %d, x = %d, y = %d, x = %d, proc_count = %d \n",myid, hypre_IndexX(div_index),
                  hypre_IndexY(div_index),hypre_IndexZ(div_index), proc_count);
        }
        
  
        hypre_CopyIndex(div_index, hypre_StructAssumedPartDivision(assumed_part, i));

#if REGION_STAT
        if ( myid ==0) hypre_printf("region = %d, proc_count = %d, divisions = [%d, %d, %d]\n", 
                              i, proc_count,  
                              hypre_IndexX(div_index), hypre_IndexY(div_index), 
                              hypre_IndexZ(div_index));
        
#endif
     } /* end of FOR EACH REGION loop */
     


    /* NOW WE HAVE COMPLETED GLOBAL INFO - START FILLING IN LOCAL INFO */


    /* now we need to populate the assumed partition object
       with info specific to each processor, like which assumed partition
       we own, which boxes are in that region, etc. */

    /* figure out my partition region and put it in the assumed_part structure*/   
     hypre_StructAssumedPartMyPartition(assumed_part) =  hypre_BoxArrayCreate(2);
     my_partition = hypre_StructAssumedPartMyPartition(assumed_part);
     hypre_StructAssumedPartitionGetRegionsFromProc(assumed_part, myid, my_partition);
#if 0
    hypre_ForBoxI(i, my_partition)
    {
       box = hypre_BoxArrayBox(my_partition, i);
       hypre_printf("myid = %d: MY ASSUMED Partitions (%d):  (%d, %d, %d)  x  "
              "(%d, %d, %d)\n",
              myid, i,
              hypre_BoxIMinX(box),
              hypre_BoxIMinY(box),
              hypre_BoxIMinZ(box),
              hypre_BoxIMaxX(box),
              hypre_BoxIMaxY(box),
           hypre_BoxIMaxZ(box));
     }
#endif


     /* find out the boxes in my partition - so I need to look through my boxes,
        figure our which assumed parition (AP) they fall in and contact that 
        processor. we need to use the exchange data functionality for this  */

     proc_alloc = 8;
     proc_array = hypre_TReAlloc(proc_array, HYPRE_Int, proc_alloc);
     
     /* probably there will mostly be one proc per box */
     /* don't want to allocate too much memory here */
     size = 1.2 * hypre_BoxArraySize(local_boxes);

     tmp_box_nums = hypre_CTAlloc(HYPRE_Int, size);
     tmp_proc_ids =  hypre_CTAlloc(HYPRE_Int, size);

     proc_count = 0;
     count = 0; /* current number of procs */
     grow_box = hypre_BoxCreate();
     
     hypre_ForBoxI(i, local_boxes)  
     {
           box = hypre_BoxArrayBox(local_boxes, i); 


           hypre_StructAssumedPartitionGetProcsFromBox(assumed_part, box, &proc_count, 
                                                       &proc_alloc, &proc_array);
           /* do we need more storage? */  
           if ((count + proc_count) > size)       
           {
              size = size + proc_count + 1.2*(hypre_BoxArraySize(local_boxes)-i);
              /* hypre_printf("myid = %d, *adjust* alloc size = %d\n", myid, size); */
              tmp_box_nums = hypre_TReAlloc(tmp_box_nums, HYPRE_Int, size);
              tmp_proc_ids = hypre_TReAlloc(tmp_proc_ids, HYPRE_Int, size);
           }
           for (j = 0; j< proc_count; j++)
           {

              /* tmp_box_nums[count] = i;*/ /*box numbers correspond to box order */
              tmp_box_nums[count] = local_boxnums[i];
              tmp_proc_ids[count] = proc_array[j];
              count++;
              
           }
     }


     hypre_BoxDestroy(grow_box);
     

     /*now we have two arrays: tmp_proc_ids and tmp_box_nums - these are 
       corresponding box  numbers and proc. ids - we need to sort the 
       processor ids and then create a new 
       buffer to send to the exchange data function */

     /*sort the proc_ids */
     hypre_qsort2i(tmp_proc_ids, tmp_box_nums, 0, count-1);
     
     /*now use proc_array for the processor ids to contact. we will use
       box array to get our boxes and then pass the array only (not the 
       structure) to exchange data */
     box_count = count;
     
     contact_boxinfo = hypre_CTAlloc(HYPRE_Int, box_count*7);
     
     proc_array = hypre_TReAlloc(proc_array, HYPRE_Int, box_count);
     proc_array_starts = hypre_CTAlloc(HYPRE_Int, box_count+1);
     proc_array_starts[0] = 0;
     
     proc_count = 0;
     index = 0;
     
     if (box_count)
     {
        proc_array[0] = tmp_proc_ids[0];

        contact_boxinfo[index++] = tmp_box_nums[0];
        box = hypre_BoxArrayBox(local_boxes, tmp_box_nums[0]);
        for (d=0; d< 3; d++)
        {
           contact_boxinfo[index++] = hypre_BoxIMinD(box, d);
           contact_boxinfo[index++] = hypre_BoxIMaxD(box, d);
        }
        proc_count++;
     }
     
     for (i=1; i< box_count; i++)
     {
        if (tmp_proc_ids[i]  != proc_array[proc_count-1])
        {
           proc_array[proc_count] = tmp_proc_ids[i];
           proc_array_starts[proc_count] = i;
           proc_count++;
           
        }

        /* these boxes are not copied in a particular order -   */
        
        contact_boxinfo[index++] = tmp_box_nums[i];
        box = hypre_BoxArrayBox(local_boxes, tmp_box_nums[i]);
        for (d=0; d< 3; d++)
        {
           contact_boxinfo[index++] = hypre_BoxIMinD(box, d);
           contact_boxinfo[index++] = hypre_BoxIMaxD(box, d);
        }
       
     }     
     proc_array_starts[proc_count] = box_count;

     /* clean up */
     hypre_TFree(tmp_proc_ids);
     hypre_TFree(tmp_box_nums);


     /* EXCHANGE DATA */

     /* prepare to populate the local info in the assumed partition */
     hypre_StructAssumedPartMyPartitionBoxes(assumed_part) 
        = hypre_BoxArrayCreate(box_count);
     hypre_BoxArraySetSize(hypre_StructAssumedPartMyPartitionBoxes(assumed_part), 0);
     hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = 0;
     hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = box_count;
     hypre_StructAssumedPartMyPartitionProcIds(assumed_part) 
        = hypre_CTAlloc(HYPRE_Int, box_count);
     hypre_StructAssumedPartMyPartitionBoxnums(assumed_part) 
        = hypre_CTAlloc(HYPRE_Int, box_count);
     hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part) = 0;

     /* set up for exchanging data */  
     /* the response we expect is just a confirmation*/
     response_buf = NULL;
     response_buf_starts = NULL;

     /*response object*/
     response_obj.fill_response = hypre_APFillResponseStructAssumedPart;
     response_obj.data1 = assumed_part; /* where we keep info from contacts*/ 
     response_obj.data2 = NULL;

     max_response_size = 0; /* no response data - just confirmation*/

     hypre_DataExchangeList(proc_count, proc_array, 
                            contact_boxinfo, proc_array_starts, 
                            7*sizeof(HYPRE_Int), 
                            sizeof(HYPRE_Int), &response_obj, max_response_size, 1, 
                            comm, (void**) &response_buf, &response_buf_starts);
     

     hypre_TFree(proc_array);
     hypre_TFree(proc_array_starts);
     hypre_TFree(response_buf);
     hypre_TFree(response_buf_starts);
     hypre_TFree(contact_boxinfo);




    /* return vars */
    *p_assumed_partition = assumed_part;
    

 return hypre_error_flag;
   
}

/******************************************************************************
 *
 *   Destroy the assumed partition   
 *
 *****************************************************************************/


HYPRE_Int hypre_StructAssumedPartitionDestroy(hypre_StructAssumedPart *assumed_part)
{

   if (assumed_part)
   {
      

      hypre_BoxArrayDestroy( hypre_StructAssumedPartRegions(assumed_part));
      hypre_TFree(hypre_StructAssumedPartProcPartitions(assumed_part));
      hypre_TFree(hypre_StructAssumedPartDivisions(assumed_part));
      hypre_BoxArrayDestroy( hypre_StructAssumedPartMyPartition(assumed_part));
      hypre_BoxArrayDestroy( hypre_StructAssumedPartMyPartitionBoxes(assumed_part));
      hypre_TFree(hypre_StructAssumedPartMyPartitionProcIds(assumed_part));
      hypre_TFree( hypre_StructAssumedPartMyPartitionBoxnums(assumed_part));
      
      /* this goes last! */
      hypre_TFree(assumed_part);
   }
   

   return hypre_error_flag;
   
   
}

/******************************************************************************
 *
 *   fillResponseStructAssumedPart
 *
 *****************************************************************************/

HYPRE_Int
hypre_APFillResponseStructAssumedPart(void *p_recv_contact_buf, 
                                 HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, 
                                 MPI_Comm comm, void **p_send_response_buf, 
                                       HYPRE_Int *response_message_size )
{
   
   HYPRE_Int    size,alloc_size, myid, i, d, index;
   HYPRE_Int    *ids, *boxnums;
   HYPRE_Int    *recv_contact_buf;

   hypre_Box    *box;
     
   hypre_BoxArray              *part_boxes;
   hypre_DataExchangeResponse  *response_obj = ro;  
   hypre_StructAssumedPart     *assumed_part = response_obj->data1;  

   
   /*initialize stuff */
   hypre_MPI_Comm_rank(comm, &myid );

   part_boxes =  hypre_StructAssumedPartMyPartitionBoxes(assumed_part);
   ids = hypre_StructAssumedPartMyPartitionProcIds(assumed_part);
   boxnums = hypre_StructAssumedPartMyPartitionBoxnums(assumed_part);


   size =  hypre_StructAssumedPartMyPartitionIdsSize(assumed_part);
   alloc_size = hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part);
   
   
   recv_contact_buf = (HYPRE_Int * ) p_recv_contact_buf;

   /*increment how many procs have contacted us */ 
   hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part)++;
   

   /*check to see if we need to allocate more space for ids and boxnums*/
   if ((size + contact_size) > alloc_size)
   {
 
      alloc_size = size + contact_size;
      ids = hypre_TReAlloc(ids, HYPRE_Int, alloc_size);
      boxnums = hypre_TReAlloc(boxnums, HYPRE_Int, alloc_size);
      hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = alloc_size;
   }
     
   
   box = hypre_BoxCreate();
   
   /* populate our assumed partition according to boxes received */
   index=0;
   for (i=0; i < contact_size; i++)
   {
      ids[size+i] = contact_proc; /*set the proc id*/
      boxnums[size+i] = recv_contact_buf[index++];
      for (d=0; d< 3; d++)
      {
         hypre_BoxIMinD(box, d)= recv_contact_buf[index++];
         hypre_BoxIMaxD(box, d) = recv_contact_buf[index++];
      }
      
      hypre_AppendBox(box, part_boxes);
   }
   /* adjust the size of the proc ids*/   
   hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = size + contact_size;

   /*in case more memory was allocated we have to assign these
     pointers back */
   hypre_StructAssumedPartMyPartitionBoxes(assumed_part) = part_boxes;
   hypre_StructAssumedPartMyPartitionProcIds(assumed_part) = ids;
   hypre_StructAssumedPartMyPartitionBoxnums(assumed_part) = boxnums;


   /*output - no message to return (confirmation) */
   *response_message_size = 0; 
  

   hypre_BoxDestroy(box);
   
   
   return hypre_error_flag;
   
}


/******************************************************************************
 *
 *   Given a processor id, get that processor's assumed region(s)   
 *
 *   At most a processor has 2 assumed regions- pass in a BoxArray of size 2.
 *****************************************************************************/


HYPRE_Int hypre_StructAssumedPartitionGetRegionsFromProc( hypre_StructAssumedPart *assumed_part , 
                                     HYPRE_Int proc_id, hypre_BoxArray *assumed_regions)
{
   

   HYPRE_Int    i;
   HYPRE_Int    in_region, proc_count, proc_start, num_partitions;
   HYPRE_Int    part_num, adj_part_num, x_row, y_row, extra, points, width;
   HYPRE_Int    plane, xyplane, adj_proc_id;
   HYPRE_Int    num_assumed, num_regions;
   
   
   hypre_Box   *region, *box;
   hypre_Index  div, isize, imin, imax;


   num_regions = hypre_StructAssumedPartNumRegions(assumed_part);
   

   /* check to see that this processor owns an assumed region - it is
      rare that it won't (only if the # of procs > vol. of bounding
      box or # procs > global # boxes) */
 
   if (proc_id >= hypre_StructAssumedPartProcPartition(assumed_part, 
                                                       num_regions))
   {
      /*owns no boxes */
      num_assumed = 0;
   }
   else   
   {
      
      /* which partition region am i in? */ 
      in_region = 0;
      if (  num_regions > 1)
      {
         while (proc_id >= hypre_StructAssumedPartProcPartition(assumed_part, 
                                                                in_region+1))
         {
            in_region++;
         }
      }
 
      /* how many processors in that region? */
      proc_count = hypre_StructAssumedPartProcPartition(assumed_part, in_region+1) 
         - hypre_StructAssumedPartProcPartition(assumed_part, in_region);
      /* first processor in the range */
      proc_start = hypre_StructAssumedPartProcPartition(assumed_part, in_region);
      /*get the region */   
      region = hypre_BoxArrayBox(hypre_StructAssumedPartRegions(assumed_part), in_region);
      /* size of the regions */
      hypre_BoxGetSize(region , isize);
      /* get the divisions in each dimension */
      hypre_CopyIndex(hypre_StructAssumedPartDivision(assumed_part, in_region), div);

      /* now calculate the assumed partition(s) (at most 2) that I own */
      /* given: region, number and range of processors, number of divisions in 
         each dimension */
      
      num_partitions = hypre_IndexX(div)*hypre_IndexY(div)*hypre_IndexZ(div);
      /* how many procs have 2 partitions instead of one*/
      extra =  num_partitions -  proc_count*(num_partitions/proc_count); 
   
      /* adjust the proc number to take into account the actual proceesor ids
         assigned to the particular region */
      /* now adj_proc_id range is 0 - (proc_count-1) */
      adj_proc_id = proc_id - proc_start;   

      /* the region is divided into partitions according to the number of
         divisions in each directions => num_partitions.  Some processors may 
         own more 
         than one partition (up to 2).  These partitions are numbered from 
         L to R and 
         bottom to top.  From the partition number then we can calculate the
         processor 
         id.  This is analogous to the 1-d ij case,
         but is of course more complicated....*/ 

      /* get my partition number */
      if (adj_proc_id < extra)
      {
         part_num = adj_proc_id*2;
         num_assumed = 2;
      }
      else 
      {
         part_num = extra+adj_proc_id;
         num_assumed = 1;
      }
   }
   
   /* make sure BoxArray has been allocated and reflects the number of 
      boxes we need */
   hypre_BoxArraySetSize(assumed_regions, num_assumed);
 
   /*where is part_num? numbering from L->R, bottom to top, 0 - (proc_count - 1)*/   
   for (i = 0; i< num_assumed; i++) 
   {
      
      part_num += i;
      
      /*find z coords first*/
      xyplane = hypre_IndexX(div)*hypre_IndexY(div);
      plane = part_num/xyplane;
   
      points =  hypre_IndexZ(isize);
      width =  points / hypre_IndexZ(div);
      extra =  points - width* hypre_IndexZ(div);

      /* z start */      
      hypre_IndexZ(imin) = width*plane;  
      hypre_IndexZ(imin) += hypre_min(extra, plane);
      /* z end */
      hypre_IndexZ(imax) =   width*(plane+1);  
      hypre_IndexZ(imax) += hypre_min(extra, plane+1);
      hypre_IndexZ(imax) -= 1;
      /*now change relative coordinates to absolute in z-dir*/
      hypre_IndexZ(imin) +=  hypre_BoxIMinZ(region);
      hypre_IndexZ(imax) +=  hypre_BoxIMinZ(region);
      
      /* now adjust the part_num to be in the lowest plane so
         that we can figure the x and y coords*/
      adj_part_num = part_num - plane*xyplane;
   
      /*x and y coords*/
      /* x_row and y_row indicate the row that the partion is in
      according to the number of divisions in each direction */
      y_row = adj_part_num / hypre_IndexX(div);
      x_row = adj_part_num % hypre_IndexX(div); /*mod */
   
      /* x */
      points =  hypre_IndexX(isize);
      width =  points / hypre_IndexX(div);
      extra =  points - width* hypre_IndexX(div);
      /* x start */      
      hypre_IndexX(imin) = width*x_row;  
      hypre_IndexX(imin) += hypre_min(extra, x_row);
      /* x end */
      hypre_IndexX(imax) =   width*(x_row+1);  
      hypre_IndexX(imax) += hypre_min(extra, x_row+1);
      hypre_IndexX(imax) -= 1;
      /*now change relative coordinates to absolute in x-dir*/
      hypre_IndexX(imin) +=  hypre_BoxIMinX(region);
      hypre_IndexX(imax) +=  hypre_BoxIMinX(region);

      /* y */
      points =  hypre_IndexY(isize);
      width =  points / hypre_IndexY(div);
      extra =  points - width* hypre_IndexY(div);
      /* y start */      
      hypre_IndexY(imin) = width*y_row;  
      hypre_IndexY(imin) += hypre_min(extra, y_row);
      /* y end */
      hypre_IndexY(imax) =   width*(y_row+1);  
      hypre_IndexY(imax) += hypre_min(extra, y_row+1);
      hypre_IndexY(imax) -= 1;
      /*now change relative coordinates to absolute in y-dir*/
      hypre_IndexY(imin) +=  hypre_BoxIMinY(region);
      hypre_IndexY(imax) +=  hypre_BoxIMinY(region);

      /* Note: a processor is only assigned a partition in 1 region */

      /* set the assumed region*/
      box = hypre_BoxArrayBox(assumed_regions, i);
      hypre_BoxSetExtents(box, imin, imax);
    
      
   }
   
   /* some thoughts: if x and y division are even (or if x is even and all the 
      extras
      the two partitions into one partition (they would be side by side in x-dir). 
      I think it may be better to leave them seperate in the interest of 
      reducing the search space for neighbor calculations */  

   return hypre_error_flag;
   
   
}



/******************************************************************************
 *
 *   Given a box, which processor(s) assumed partition does the box intersect  
 *
 *   proc_array should be allocated to size_alloc_proc_array
 *****************************************************************************/


HYPRE_Int hypre_StructAssumedPartitionGetProcsFromBox( hypre_StructAssumedPart *assumed_part , 
                                        hypre_Box *box, 
                                        HYPRE_Int *num_proc_array, 
                                        HYPRE_Int *size_alloc_proc_array, 
                                        HYPRE_Int **p_proc_array)
{
   
   HYPRE_Int       i,j,k,r,myid;
   HYPRE_Int       num_regions, in_regions, this_region, proc_count, proc_start;
   HYPRE_Int       adj_proc_id, extra, num_partitions, part_num;
   HYPRE_Int       gridpt[2], xyplane, extra_procs, switch_proc, points, width;
   HYPRE_Int       switch_pt;
   
   HYPRE_Int      *proc_array, proc_array_count;
   HYPRE_Int      *which_regions;
   HYPRE_Int      *proc_ids, num_proc_ids, size_proc_ids;
   

   hypre_Box      *region;
   hypre_Box      *result_box;
   hypre_Index     div, isize, part_row[2];
   hypre_BoxArray *region_array;


   /* need myid only for the hypre_printf statement*/
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   
   proc_array = *p_proc_array;
   region_array =   hypre_StructAssumedPartRegions(assumed_part);
   num_regions = hypre_StructAssumedPartNumRegions(assumed_part);

   /*first intersect the box to find out which region or regions(s) it lies in*/
   /*then determine which processor owns the assumed part of the particular 
     regions(s) that it is in */ 

   result_box = hypre_BoxCreate();
   which_regions = hypre_CTAlloc(HYPRE_Int, num_regions);
    
   size_proc_ids = 8;
   proc_ids = hypre_CTAlloc(HYPRE_Int, size_proc_ids);
   num_proc_ids = 0;
   

   /* which partition region(s) am i in? */ 
   in_regions = 0;
   for (i = 0; i< num_regions; i++) 
   {
      region = hypre_BoxArrayBox(region_array, i);  
      hypre_IntersectBoxes( box, region, result_box);
      if (  hypre_BoxVolume(result_box) > 0 )
      {
         which_regions[in_regions] = i;
         in_regions++;
      }
   }
#if 0
   if (in_regions == 0)  
   {

      /* 9/16/10 in hypre_SStructGridAssembleBoxManagers we grow boxes by 1
         before the gather boxes call because of shared variables, so
         now so we can get the situation that the gather box is outside of
         the assumed region */
      
         if (hypre_BoxVolume(box) > 0)
         { 
            hypre_error(HYPRE_ERROR_GENERIC);
            hypre_printf("MY_ID = %d Error: positive volume box (%d, %d, %d) x "
                   "(%d, %d, %d)  not in any assumed regions! (this should never"
                   " happen)\n", 
                   myid,
                   hypre_BoxIMinX(box),
                   hypre_BoxIMinY(box),
                   hypre_BoxIMinZ(box),
                   hypre_BoxIMaxX(box),
                   hypre_BoxIMaxY(box),
                   hypre_BoxIMaxZ(box));
             
         }
         else
         {
            /*this is ok 12/04 - current use of function is only called on 
              pos. boxes*/ 
            /* hypre_printf("MY_ID = %d Error: neg volume box not in partition\n", 
                       myid);*/
         }
      
   }
#endif   
   /*for each region, who is assumed to own this box? add the proc number
     to proc array */ 
   for (r = 0; r< in_regions; r++)
   {

      /* initialization for this particular region */
      this_region = which_regions[r];
      region = hypre_BoxArrayBox(region_array, this_region);  
      /* first processor in the range */
      proc_start = hypre_StructAssumedPartProcPartition(assumed_part, this_region);
      /* how many processors in that region? */
      proc_count = hypre_StructAssumedPartProcPartition(assumed_part, 
                                                        this_region+1) 
         -proc_start;
      /* size of the regions */
      hypre_BoxGetSize(region , isize);
      /* get the divisons in each dimension */
      hypre_CopyIndex(hypre_StructAssumedPartDivision(assumed_part, 
                                                      this_region), div);

      /* number of partitions in this region? */
      num_partitions = hypre_IndexX(div)*hypre_IndexY(div)*hypre_IndexZ(div);
      /* how many procs have 2 partitions instead of one*/
      extra_procs =  num_partitions -  proc_count*(num_partitions/proc_count); 
      switch_proc = 2*extra_procs;
      /* size of the xy-plane */
      xyplane = hypre_IndexX(div)*hypre_IndexY(div);

      /* get the location of a imin and imax indexes for this box
         in terms of
         the xrow, yrow, and zrow in the assumed partition 
         coordinates  - store these in part_row[0] and part_row[1]
         for imin and imax respectively */  

      /*check each dimension - get the partition row info an put in part_row*/
      for (i = 0; i< 3; i++)
      {
         /* first get the max and min grid point in this dimension
            and change to coords relative to
            the region  (so relative means the min is (0,0,0)
            - if a point is not in the region
            then take the regions end point */
         if (hypre_IndexDInBoxP(hypre_BoxIMin(box), i, region))
         {
            gridpt[0] =  hypre_BoxIMinD(box, i) - hypre_BoxIMinD(region, i);
         }
         else  /*grid point not in the region */
         {
            gridpt[0] = 0; /* 0 is the min *relative* corrd. */
         }
         if (hypre_IndexDInBoxP(hypre_BoxIMax(box), i, region))
         {
            gridpt[1] =  hypre_BoxIMaxD(box, i) - hypre_BoxIMinD(region, i);
         }
         else /*grid point not in the region */
         {
            gridpt[1] =  hypre_BoxIMaxD(region, i) - hypre_BoxIMinD(region, i); 
         }
         
         /* now find the correct part_row for this dimension*/
  
         points =  hypre_IndexD(isize, i); /*size of the region in this dimension */
         width =  points / hypre_IndexD(div, i); /*width of a partition */
         extra =  points - width* hypre_IndexD(div, i); 
         switch_pt = (width + 1) * extra;


         /* for imin(j=0) and imax (j=1) find the corresponding partition row*/ 
         for (j=0; j< 2; j++)
         {
            if (gridpt[j] >= switch_pt)
            {
               hypre_IndexD(part_row[j], i)= extra + (gridpt[j] - switch_pt)/width;
            }
            else
            {
               hypre_IndexD(part_row[j], i) = gridpt[j]/(width+1);
            }
         }
         
      } /*end of for each dimension */


      /*now we have the correct part_row for each of the box's corners since
        we checked the max and min indexes - these correct partition rows
        are stored in part_row */


      /*find the partition number and proc_id for each of the 8 corner points*/
      /* we get the location of the corner points from the location of imin and imax
         which was determined above and stored in part_row*/
    
     
      for (i = hypre_IndexX(part_row[0]); i<= hypre_IndexX(part_row[1]); i++) /* x*/
      { 
         for (j = hypre_IndexY(part_row[0]); j <=  hypre_IndexY(part_row[1]); j++) /*y*/
         {
            for (k = hypre_IndexZ(part_row[0]); k <= hypre_IndexZ(part_row[1]); k++) /*z*/
            {

               part_num = 0;
               /*move up in y */ 
               part_num += j* hypre_IndexX(div);
               /* move over in x*/
               part_num += i;
               /*adjust the plane for z*/ 
               part_num += k*xyplane;
               
               /*convert the partition number to a processor number*/
               if (part_num >= switch_proc)
               {
                  adj_proc_id =  extra_procs + (part_num - switch_proc);
               }
               else
               {    
                  adj_proc_id = part_num/2 ;
               }
                 
               if (num_proc_ids == size_proc_ids)
               {
                  size_proc_ids+=8;
                  proc_ids = hypre_TReAlloc(proc_ids, HYPRE_Int, size_proc_ids);
               }
               

               proc_ids[num_proc_ids] = adj_proc_id + proc_start;
               num_proc_ids++;
               
            }
         }
      }


   } /*end of for each region loop*/
   

   if (in_regions) 
   {
   
      /* now determine unique values in proc_id array (could be duplicates - 
         do to a processor owning more than one partiton in a region)*/
      /*sort the array*/
      qsort0(proc_ids, 0, num_proc_ids-1);
      
      /*make sure we have enough space from proc_array*/
      if (*size_alloc_proc_array < num_proc_ids)
      {
         proc_array = hypre_TReAlloc(proc_array, HYPRE_Int, num_proc_ids);
         *size_alloc_proc_array = num_proc_ids;
      }
   
      /*put unique values in proc_array*/
      proc_array[0] = proc_ids[0]; /*there will be at least one processor id */
      proc_array_count = 1;
      for (i=1; i< num_proc_ids; i++)
      {
         if  (proc_ids[i] != proc_array[proc_array_count-1]) 
         {
            proc_array[proc_array_count] = proc_ids[i];
            proc_array_count++;
         }
      }
   }
   else /*no processors for this box (neg. volume box) */
   {
      proc_array_count = 0;
   }

   

   /*return variables */
   *p_proc_array = proc_array;
   *num_proc_array = proc_array_count;
   
  

   /*clean up*/
   hypre_BoxDestroy(result_box);
   hypre_TFree(which_regions);
   hypre_TFree(proc_ids);
   

   return hypre_error_flag;
   
   
}

#if 0
{
/*
   UNFINISHED  
   Create a new assumed partition by coarsen the boxes from another
   assumed partition.

   unfinished because of a problem: can't figure out what the new id
   is since the zero boxes drop out - and we don't have all of the
   boxes from a particular processor in the AP */
 
*/ 


HYPRE_Int hypre_StructCoarsenAP(hypre_StructAssumedPart *ap,  hypre_Index index, hypre_Index stride,
                          hypre_StructAssumedPart **new_ap_ptr)
{
   
   HYPRE_Int num_regions;
   
   hypre_BoxArray *coarse_boxes;
   hypre_BoxArray *fine_boxes;
   hypre_BoxArray *regions_array;
   hypre_Box *box;

   hypre_StructAssumedPartition *new_ap;
   

   
   /* create new ap and copy global description information */
   new_ap = hypre_TAlloc(hypre_StructAssumedPart, 1);
   
   num_regions = hypre_StructAssumedPartNumRegions(ap);
   regions_array =    hypre_BoxArrayCreate(num_regions);

   hypre_StructAssumedPartRegions(new_ap) = regions_array;
   hypre_StructAssumedPartNumRegions(new_ap) = num_regions;
   hypre_StructAssumedPartProcPartitions(new_ap) = 
      hypre_CTAlloc(HYPRE_Int, num_regions+1); 
   hypre_StructAssumedPartDivisions(new_ap) = 
      hypre_CTAlloc(HYPRE_Int, num_regions); 

   hypre_StructAssumedPartProcPartitions(new_ap)[0] = 0;
   
   for (i=0; i< num_regions; i++)
   {
      box =  hypre_BoxArrayBox(hypre_StructAssumedPartRegions(ap), i);
                            
      hypre_CopyBox(box, hypre_BoxArrayBox(regions_array, i));

      hypre_StructAssumedPartDivision(new_ap, i) = 
         hypre_StructAssumedPartDivision(new_ap, i);

      hypre_StructAssumedPartProcPartition(new_ap, i+1)=
         hypre_StructAssumedPartProcPartition(ap, i+1);

   }
   
   /* copy my partition (at most 2 boxes)*/
   hypre_StructAssumedPartMyPartition(new_ap) 
      = hypre_BoxArrayCreate(2);
   for (i=0; i< 2; i++)
   {
       box = hypre_BoxArrayBox(hypre_StructAssumedPartMyPartition(ap), i);
       hypre_CopyBox(box, hypre_BoxArrayBox(hypre_StructAssumedPartMyPartition(new_ap), i));
   }

   /*create space for the boxes, ids and boxnums */
   size = hypre_StructAssumedPartMyPartitionIdsSize(ap);

   hypre_StructAssumedPartMyPartitionProcIds(new_ap) = hypre_CTAlloc(HYPRE_Int, size); 
   hypre_StructAssumedPartMyPartitionBoxnums(new_ap) = hypre_CTAlloc(HYPRE_Int, size); 

   hypre_StructAssumedPartMyPartitionBoxes(new_ap) 
      = hypre_BoxArrayCreate(size);

   hypre_StructAssumedPartMyPartitionIdsAlloc(new_ap) = size;
   hypre_StructAssumedPartMyPartitionIdsSize(new_ap) = size;

   /* now coarsen and copy the boxes - here we are going to prune - don't want size 0
      boxes */
   coarse_boxes = hypre_StructAssumedPartMyPartitionBoxes(new_ap);
   fine_boxes =  hypre_StructAssumedPartMyPartitionBoxes(ap);

   new_box = hypre_BoxCreate();

   hypre_ForBoxI(i, fine_boxes)
   {
      box =  hypre_BoxArrayBox(fine_boxes,i);
      
      hypre_CopyBox(box, new_box);
      
      hypre_StructCoarsenBox( new_box, index, stride);
  

   }
  

/* unfinished because of a problem: can't figure out what the new id
   is since the zero boxes drop out - and we don't have all of the
   boxes from a particular processor in the AP */


   /*  hypre_StructAssumedPartMyPartitionNumDistinctProcs(new_ap) */





   *new_ap_ptr = new_ap;


   return hypre_error_flag;
   
   
}
#endif
