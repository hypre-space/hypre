/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Routine for picking independent set.
 *
 *****************************************************************************/


#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_AMGIndepSet
 *--------------------------------------------------------------------------*/

int hypre_AMGIndepSet(ST_i, ST_j, num_variables,
                      measure_array, IS_array, IS_size)

/*--------------------------------------------------

  hypre_PickIndependentSet(ST, measure_array,
                               &IS_array[IS_start], &IS_size);

  ----------------------------------------------------- */



int             *ST_i;
int             *ST_j;
int              num_variables;
double          *measure_array;
int             *IS_array;
int             *IS_size;

{

  /*------------------------------
   * Variables I've added to be declared, or passed, etc.
   *--------------------------------------------------*/

   double       *local_measure;
   double        my_measure;
   double        s;
   int          *not_marked;

 
   int          i, j;
   int          mine_is_bigger;
   int          previous_num_indep_pts;
   int          num_points_in_graph;
   int          next_indep_point;
   int          point;
   int          nabor;

   not_marked = hypre_CTAlloc(int, num_variables);
   local_measure = hypre_CTAlloc(double, num_variables);
                      /*should use the number of remaining points
                        to be examined, rather than num_variables */

   /*-------------------------------------------------------
    * Copy measure array and add random value
    * Then find a set of nodes whose new measure value
    * is greater than that of all their neighbors.  This
    * is an independent set.
    *-------------------------------------------------------*/

   previous_num_indep_pts = 0;
   next_indep_point = 0;
   num_points_in_graph = num_variables;

   for (i = 0; i < num_variables; i++)
   {
      /* eliminate vertices already set as C or F points */
      if (measure_array[i] > 0)
      { 
         not_marked[i] = 1;
      }
      else
      {
         not_marked[i] = 0;
         --num_points_in_graph;
      } 

      s = hypre_Rand();
      local_measure[i] = measure_array[i] + s;
   }

 
  while (num_points_in_graph > 0)
   {
       for (i = 0; i < num_variables; i++)
       {
         if (not_marked[i])
         {
            my_measure = local_measure[i];
            mine_is_bigger = 1;
            if (ST_i[i] < ST_i[i+1])
            {
               for (j = ST_i[i]; j < ST_i[i+1]; j++)
               {
                  nabor = ST_j[j];
                  if (not_marked[nabor] && local_measure[nabor] > my_measure) 
                  {
                     mine_is_bigger = 0;
		     break;
                  }
               }

               if (mine_is_bigger)
               {
                  IS_array[next_indep_point] = i;
                  ++*IS_size;
                  ++next_indep_point;
               }
            }
         }
       }
      

      /*-----------------------------------------------------------
       * Mark points found and their neighbors
       *-----------------------------------------------------------*/

      for (i = previous_num_indep_pts; i < next_indep_point; i++)
      {
         point = IS_array[i];
         not_marked[point] = 0;
         --num_points_in_graph;
         for (j = ST_i[point]; j < ST_i[point+1]; j++)
         {
            nabor = ST_j[j];
            if (not_marked[nabor]) 
            {
               not_marked[nabor] = 0;
               --num_points_in_graph;
            }
         }
      }

      previous_num_indep_pts = next_indep_point;
   }

  return(0);
}
         

