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

/*--------------------------------------------------------------------------
 * hypre_AMGCoarsen
 *--------------------------------------------------------------------------*/

int
hypre_AMGCoarsen( hypre_Matrix  *A,
                  int          **coarse_points_ptr )
{
   int          *coarse_points;

   double       *A_data;
   int          *A_i;
   int          *A_j;

   /* CSR components of the transpose of the "strength matrix", S */
   hypre_Matrix *ST;
   double       *ST_data;
   int          *ST_i;
   int          *ST_j;

   double       *measure_array;

   int          *IS_array;
   int           IS_start, IS_size;

   /* CSR indices are denoted by iiA, jjA, ... */

   /*---------------------------------------------------
    * Compute a column-based strength matrix, S.
    * - No diagonal entry is stored.
    *
    * The first implementation will just use a 0 or
    * a 1 for the entries of S as defined by the
    * standard AMG definition of "strongly depends on".
    *---------------------------------------------------*/

   /*---------------------------------------------------
    * Compute "measures" for the fine-grid points,
    * and store in measure_array.
    *
    * The first implementation of this will just sum
    * the columns of S
    *---------------------------------------------------*/

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   IS_start = 0;

   while (1)
   {
      /*------------------------------------------------
       * Pick an independent set (maybe maximal) of
       * points with maximal measure.
       * - Each independent set is tacked onto the end
       *   of the array, IS_array.  This is how we
       *   keep a running list of C-points.
       *------------------------------------------------*/

      hypre_PickIndependentSet(ST, measure_array,
                               &IS_array[IS_start], &IS_size);

      /* check to see whether there are any points left */
      if (IS_size == 0)
         break;

      /*------------------------------------------------
       * For each new IS point, update the strength
       * matrix and the measure array.
       *------------------------------------------------*/

      for (i = IS_start; i < IS_start + IS_size; i++)
      {
         /*---------------------------------------------
          * Heuristic: C-pts don't need to interpolate
          * from neighbors they strongly depend on.
          *
          * for each s_ij in the ith row of S (neighbors
          * that point i strongly depends on)
          * - subtract 1 from measure_array[j] and
          *   remove edge s_ij
          *
          * Notes:
          * - Since S is stored by columns, in order
          *   to loop through a row of S, we need to loop
          *   through rows of A, then check that there is
          *   a corresponding nonzero entry in S.
          * - To remove edge s_ij from S, we can replace
          *   it with the last column entry in the jth
          *   column, and decrement the size of this
          *   column (we need to keep an extra vector
          *   around with the column sizes to do this).
          *---------------------------------------------*/

         /*---------------------------------------------
          * Heuristic: neighbors that depend strongly on
          * a C-pt can get good interpolation data from
          * that C-pt, and hence are less dependent on
          * each other.
          *
          * for each s_ji in the ith column of S (neighbors
          * that point i strongly influences)
          * - remove edge s_ji
          * - for each s_jk in the jth row of S (neighbors
          *   that point j strongly depends on)
          *   - if point s_ki is nonzero (point i also
          *     strongly influences point k), subtract 1
          *     from measure_array[j] and remove edge s_jk
          * set measure_array[i] to 0
          *---------------------------------------------*/

      }

      IS_start += IS_size;
   }

   /*---------------------------------------------------
    * Allocate and set the coarse_points array.
    *---------------------------------------------------*/

   coarse_points = hypre_CTAlloc(int, nf);

   for (i = 0; k < IS_start; i++)
   {
      coarse_points[IS_array[i]] = 1;
   }

   *coarse_points_ptr = coarse_points;
}
