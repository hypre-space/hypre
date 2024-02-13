
/*--------------------------------------------------------------------------
 * hypre_ParChebyData
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Real          max_eig;
   HYPRE_Real          min_eig;
   HYPRE_Real          fraction;
   HYPRE_Int           order;     /* polynomial order */
   HYPRE_Int           scale;     /* scale by diagonal?*/
   HYPRE_Int           variant;
   HYPRE_Int           max_cg_it;
   HYPRE_Real         *coefs_ptr;
   HYPRE_Real         *ds_ptr;    /* initial/updated approximation */

} hypre_ParChebyData;

#define hypre_ParChebyDataMaxEig(cheby_data)       ((cheby_data) -> max_eig)
#define hypre_ParChebyDataMinEig(cheby_data)       ((cheby_data) -> min_eig)
#define hypre_ParChebyDataFraction(cheby_data)       ((cheby_data) -> fraction)
#define hypre_ParChebyDataOrder(cheby_data)       ((cheby_data) -> order)
#define hypre_ParChebyDataScale(cheby_data)       ((cheby_data) -> scale)
#define hypre_ParChebyDataVariant(cheby_data)       ((cheby_data) -> variant)
#define hypre_ParChebyDataMaxCGIterations(cheby_data)       ((cheby_data) -> max_cg_it)
#define hypre_ParChebyDataCoefs(cheby_data)       ((cheby_data) -> coefs_ptr)
#define hypre_ParChebyDataDs(cheby_data)       ((cheby_data) -> ds_ptr)
