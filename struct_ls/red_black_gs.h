/*--------------------------------------------------------------------------
 * hypre_RedBlackGSData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   double                  tol;                /* not yet used */
   int                     max_iter;
   int                     rel_change;         /* not yet used */
   int                     zero_guess;
   int                     rb_start;

   hypre_StructMatrix     *A;
   hypre_StructVector     *b;
   hypre_StructVector     *x;

   int                     diag_rank;

   hypre_ComputePkg       *compute_pkg;

   /* log info (always logged) */
   int                     num_iterations;
   int                     time_index;
   int                     flops;

} hypre_RedBlackGSData;

