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
 * Routine for automatic coarsening in unstructured multigrid codes
 *
 * Notes:
 *
 *   - The underlying matrix storage scheme is a PETSc AIJ matrix.
 *
 *   - We define the following temporary storage:
 *
 *     ST            - a CSR matrix representing the transpose of
 *                     the "strength matrix", S.
 *     measure_array - a double array containing the "measures" for
 *                     each of the fine-grid points
 *     IS_array      - an integer array containing the list of points
 *                     in the independent sets (it also naturally
 *                     contains the list of C-points)
 *
 *   - The graph of the "strength matrix" for A is a subgraph of
 *     the graph of A, but requires nonsymmetric storage even if
 *     A is symmetric.  This is because of the directional nature of
 *     the "strengh of dependence" notion (see below).  Since we are
 *     using nonsymmetric storage for A right now, this is not a problem.
 *     If we ever add the ability to store A symmetrically, then we
 *     could store the strength graph as floats instead of doubles to
 *     save space.
 *
 * Terminology:
 *  
 *   Ruge's terminology: A point is is "strongly connected to" j, or
 *   "strongly depends on" j, if -a_ij >= \theta max_(l != j) {-a_il}.
 *  
 *   Here, we retain some of this terminology, but with a more generalized
 *   notion of "strength".  We also retain the "natural" graph notation
 *   for representing the directed graph of a matrix.  That is, the
 *   nonzero entry a_ij is represented as:
 *  
 *       x --------> x
 *       i           j
 *  
 *   In the strength matrix, S, the entry s_ij is also graphically denoted
 *   as above, and means both of the following:
 *  
 *     - i "strongly depends on" j with "strength" s_ij
 *     - j "strongly influences" i with "strength" s_ij
 * 
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_AMGCoarsen
 *--------------------------------------------------------------------------*/

int hypre_AMGCoarsen(A, strong_threshold, CF_marker)

hypre_CSRMatrix    *S;
hypre_CSRMatrix    *A;
int                *CF_marker;
double              strong_threshold;

{
   double       *A_data = hypre_CSRMatrixData(A);
   int          *A_i = hypre_CSRMatrixI(A);
   int          *A_j = hypre_CSRMatrixJ(A);
   int           num_variables = hypre_CSRMatrixNumRows(A);

   /*--------------------------------------------------------------
    * CSR components of the transpose of the "strength matrix", S,
    * are stored in the hypre_Matrix ST. ***NOTE*** this matrix
    * violates the convention, in that there is *NO DIAGONAL ELEMENT*
    * on any row.  Caveat Emptor!
    *
    * For now, the "strength" of dependence/influence is defined in
    * the traditional way, e.g., i strongly depends on j if |aij| >
    * max (k != i ) |aik|. Then ST_ji = 1, else ST_ji = 0.
    *
    * For more sophistocated uses in future, ST_data will be of type
    * double.  For now, it is type int.
    *----------------------------------------------------------------*/

   int          *ST_data;  /* ST_data will be type double in future */
   int          *ST_i;
   int          *ST_j;   

   double       *measure_array;

   int          *IS_array;
   int           IS_start, IS_size;

   /*---------------------------------------------------
    * Variables I've added to be declared, or passed, etc.
    *--------------------------------------------------*/

   /*-------------
      int      num_unknowns;
      int      num_points;
      int     *iu;
      int     *ip;
      int     *iv;
    *-------------*/

   int      i, j, k, m, n;
   int      num_connections = 0;
   int      num_lost;
   int      next_open;
   int      now_checking;
   int      start_j;
   int      num_pts_not_assigned;
   int      Cpoint;
   int      nabor, nabor_two;
   double   RowMaxAbs;

   CF_marker = hypre_CTAlloc(int, num_variables);
   IS_array = hypre_CTAlloc(int, num_variables);
   ST_i = hypre_CTAlloc(int, num_variables+1);
   measure_array = hypre_CTAlloc(double, num_variables);
   num_pts_not_assigned = num_variables; /* will rewrite as num_points for
                                            systems */

   /*---------------------------------------------------
    * Compute a column-based strength matrix, S.
    * - No diagonal entry is stored.
    *
    * The first implementation will just use a 0 or
    * a 1 for the entries of S as defined by the
    * standard AMG definition of "strongly depends on".
    *---------------------------------------------------*/

   for (i = 0; i < num_variables; ++i)
   {
      RowMaxAbs = 0.0;
      for (j = A_i[i]+1; j < A_i[i+1]; j++)
      {
         ST_i[A_j[j]]++;
         RowMaxAbs = max(RowMaxAbs, fabs(A_data[j])); 
                       /* assumes correct sign of off-diagonal. 
                          fix later */
         ++num_connections;
      }
      measure_array[i] = RowMaxAbs;
   }

   /*---------------------------------------------------
    * Build first cut at ST_i
    *---------------------------------------------------*/

   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = num_connections;
      num_connections -= ST_i[i-1];
   }
   ST_i[0] = 0;

                                /* ST_data will be type double in future */
   ST_data = hypre_CTAlloc(int, ST_i[num_variables]);
   ST_j = hypre_CTAlloc(int, ST_i[num_variables]);

   /*-------------------------------------------------
    * Check each connection for strong dependence.
    * If true, make entry indicating strong influence
    * in ST.
    *-------------------------------------------------*/

   for (i = 0; i < num_variables; ++i)
   {
      for (j = A_i[i]+1; j < A_i[i+1]; ++j)
      {
         ST_j[ST_i[A_j[j]]] = i;
         if (fabs(A_data[j]) > strong_threshold * measure_array[i])
         {
            ST_data[ST_i[A_j[j]]] = 1;
         }
         ST_i[A_j[j]]++; 
      }
   }

   /*------------------------------------------------------------
    * ST_i[j] now points to the *end* of the jth row of entries
    * instead of the beginning.  Restore ST_i to front of row.
    *------------------------------------------------------------*/

   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i-1]; 
   }
   ST_i[0] = 0;

   /*----------------------------------------------------------
    * Compress ST to remove non-strong influences
    *----------------------------------------------------------*/

   next_open = 0;
   now_checking = 0;
   num_lost = 0;

   for (i = 0; i < num_variables; i++)
   {
      start_j = ST_i[i];
      ST_i[i] -= num_lost;

      for (j = start_j; j < ST_i[i+1]; j++)
      {
         if (ST_data[now_checking] == 0)
         {
            num_lost++;
            now_checking++;
         }
         else
         {
            ST_data[next_open] = 1;
            ST_j[next_open] = ST_j[now_checking];
            now_checking++;
            next_open++;
         }
      }
   }
   ST_i[num_variables] -= num_lost;

   /*----------------------------------------------------------
    * Compute "measures" for the fine-grid points,
    * and store in measure_array.
    *
    * The first implementation of this will just sum
    * the columns of S (rows of ST). Hence, measure_array[j]
    * is the *number* of strong influences variable j has.
    *----------------------------------------------------------*/

   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = ST_i[i+1] - ST_i[i];
      if (measure_array[i] == 0) 
      {
         /*mark points with no influences as fine points */
         CF_marker[i] = -1;
         --num_pts_not_assigned;
      }
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   IS_start = 0;

   while (num_pts_not_assigned > 0)
   {
      /*------------------------------------------------
       * Pick an independent set (maybe maximal) of
       * points with maximal measure.
       * - Each independent set is tacked onto the end
       *   of the array, IS_array.  This is how we
       *   keep a running list of C-points.
       *------------------------------------------------*/

      IS_size = 0;
      hypre_AMGIndepSet(ST_data, ST_i, ST_j, num_variables, 
                        measure_array, &IS_array[IS_start], &IS_size);

      /* check to see whether there are any points left */
/* printf("%d\n", IS_size); */

      if (IS_size == 0)  
         break;

      /*------------------------------------------------
       * For each new IS point, update the strength
       * matrix and the measure array.
       *------------------------------------------------*/

      for (i = IS_start; i < IS_start + IS_size; i++)
      {
         Cpoint = IS_array[i];
         if (CF_marker[Cpoint] == 0) /* point isn't already C or F */
         {
            CF_marker[Cpoint] = 1;
            --num_pts_not_assigned;
         
            /*---------------------------------------------
             * Heuristic:  1 goes here 
             *---------------------------------------------*/

            for (j = A_i[Cpoint]+1; j < A_i[Cpoint+1]; j++)
            {
               nabor = A_j[j];

               for (k = ST_i[nabor]; k < ST_i[nabor+1]; k++)
               {
                  if(ST_j[k] == Cpoint) /* Cpoint depends on nabor */
                  {
                     measure_array[nabor]--;
                     if (measure_array[nabor] == 0)  /* mark as fine point */
                     {
                        CF_marker[nabor] = -1;
                        --num_pts_not_assigned; 
                     }
                  ST_data[k] = -1;   /* change sign when ST_data is double */
                  break;                /* useable since ST_j are ordered */

                  }
               }
            }            
            
            /*---------------------------------------------
             * Heuristic: 2 goes here
             *---------------------------------------------*/

            for (n = ST_i[Cpoint]; n < ST_i[Cpoint+1]; ++n)
            {
               {
                  nabor = ST_j[n];
                  ST_data[n] = -1;     
                                                   
                  for (j = ST_i[nabor]; j < ST_i[nabor+1]; j++)
                  {
                                     /* nabor influences nabor_two. */
                      nabor_two = ST_j[j];

                                     /* see if nabor_two depends on Cpoint */
                      for (m = ST_i[Cpoint]; m < ST_i[Cpoint+1]; m++)
                      {
                          if (ST_j[m] == nabor_two)
                          {
                                     /* nabor_two DOES depend on Cpoint */
                             ST_data[j] = -1;
                             --measure_array[nabor];
                             if (measure_array[nabor] == 0) 
                             {
                                     /* new fine point */
                                CF_marker[nabor] = -1;
                                --num_pts_not_assigned; 
                             }        
                             break;  /* possible since ST is ordered */
                          }
                       }    
                  }
               }
            }
         }
         measure_array[Cpoint] = 0;
      } /* end loop over IS_array points */

      IS_start += IS_size;
   }

#if 1
/* Prints st.matx, st.coarse files, so that Matlab can display them */
   {

      FILE    *fp;
      char    fname[256];

      debug_out(ST_data,ST_i,ST_j,num_variables);

      sprintf(fname,"st.coarse");
      fp = fopen(fname, "w");

      for (j=0; j<IS_start;j++)
      {
         fprintf(fp, "%d\n", IS_array[j]);
      }
      fclose(fp);
   }
#endif

   return(0);
}



void debug_out(ST_data,ST_i,ST_j,num_variables)

int   *ST_data;
int   *ST_i;
int   *ST_j;
int    num_variables;

{
      hypre_CSRMatrix *ST;
      double  *matx_data;
      int      i;

      ST = hypre_CreateCSRMatrix(num_variables, num_variables, NULL);

      matx_data = hypre_CTAlloc(double,ST_i[num_variables]);
      for (i=0;i<ST_i[num_variables];i++)
      {
         matx_data[i] = (double) ST_data[i];
      }

      hypre_CSRMatrixData(ST) = matx_data;
      hypre_CSRMatrixI(ST)    = ST_i;
      hypre_CSRMatrixJ(ST)    = ST_j;
      hypre_CSRMatrixOwnsData(ST) = 0;  /* matrix does NOT own data */

      hypre_PrintCSRMatrix(ST,"st.matx");

      hypre_DestroyCSRMatrix(ST);
      hypre_TFree(matx_data);

}



