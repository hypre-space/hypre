/**************************************************************************
 *
 *************************************************************************/

#ifndef __MLI_AMG_SAH__
#define __MLI_AMG_SAH__

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* ***********************************************************************
 * definition of the smoothed aggregation data structure                 *
 * ----------------------------------------------------------------------*/

typedef struct MLI_AMG_SA_Struct
{
   char     method_name[80];         /* store name of the AMG method     */
   MPI_Comm mpi_comm;                /* parallel communicator            */
   int      max_levels;              /* the finest level is max_levels-1 */
   int      curr_level;              /* current level being processed    */
   int      debug_level;             /* for diagnostics                  */
   int      node_dofs;               /* equation block size (fixed)      */
   double   threshold;               /* for creating aggregation graph   */
   int      nullspace_dim;           /* null space information (changes  */
   double   *nullspace_vec;          /* as curr_level)                   */
   double   P_weight;                /* weight for prolongator smoother  */
   double   *matrix_rowsums;         /* matrix row sums at each level    */ 
   int      *matrix_sizes;           /* matrix sizes at each level       */
   double   drop_tol_for_P;          /* tolerance for sparsify P         */
   int      **sa_counts;             /* store aggregation information at */
   int      **sa_data;               /* each level                       */
   double   *spectral_norms;         /* computed matrix norms            */
   int      calc_norm_scheme;        /* method to estimate matrix norm   */
   int      min_coarse_size;         /* tell when to stop aggregation    */
   int      coarsen_scheme;          /* different aggregation schemes    */
   int      *mat_complexity;         /* nnz of matrices at each level    */      
   int      *oper_complexity;        /* nnz of P and R at each level     */
   int      pre_smoothers;           /* denote which pre-smoother to use */
   int      postsmoothers;           /* denote which postsmoother to use */
   int      pre_smoother_num;        /* number of pre-smoother sweeps    */
   int      postsmoother_num;        /* number of postsmoother sweeps    */
   double   *pre_smoother_wgt;       /* weight used in pre-smoother      */
   double   *postsmoother_wgt;       /* weight used in postsmoother      */
   int      coarse_solver;           /* denote which coarse solver to use*/
   int      coarse_solver_num;       /* number of coarse solver sweeps   */
   double   coarse_solver_wgt;       /* weight used in coarse solver     */

} MLI_AMG_SA;

/* ***********************************************************************
 * functions to manipulate the aggregate data structure                  *
 * ********************************************************************* */

#ifdef __cplusplus
extern "C"
{
#endif

/* --------------------------------------------------------------------- *
 * constructor/destructor and level control                              *
 * --------------------------------------------------------------------- */

MLI_AMG_SA *MLI_AMG_SA_Create( MPI_Comm, int max_levels );
int MLI_AMG_SA_Destroy( MLI_AMG_SA * );
int MLI_AMG_SA_Set_DebugLevel( MLI_AMG_SA *, int debug_level );

/* --------------------------------------------------------------------- *
 * select parallel coarsening schemes                                    *
 * --------------------------------------------------------------------- */

int MLI_AMG_SA_Set_CoarsenScheme_Local( MLI_AMG_SA * );

/* --------------------------------------------------------------------- *
 * control when to stop coarsening                                       *
 * --------------------------------------------------------------------- */

int MLI_AMG_SA_Set_MinCoarseSize( MLI_AMG_SA *, int min_size );

/* --------------------------------------------------------------------- *
 * set threshold for pruning matrix graph                                *
 * --------------------------------------------------------------------- */

int MLI_AMG_SA_Set_Threshold( MLI_AMG_SA *, double thresh );

/* --------------------------------------------------------------------- *
 * damping factor for prolongator smoother                               *
 * --------------------------------------------------------------------- */

int MLI_AMG_SA_Set_Pweight( MLI_AMG_SA *, double weight );

/* --------------------------------------------------------------------- *
 * set up scheme to compute spectral radius of A at each level           *
 * --------------------------------------------------------------------- */

int MLI_AMG_SA_Set_CalcSpectralNorm( MLI_AMG_SA * );

/* --------------------------------------------------------------------- *
 * set null space for the finest grid                                    *
 * --------------------------------------------------------------------- */

int MLI_AMG_SA_Set_NullSpace(MLI_AMG_SA *, int node_dofs, int num_ns, 
                             double *null_vec, int length);

/* --------------------------------------------------------------------- *
 * activate coarsening                                                   *
 * --------------------------------------------------------------------- */

int MLI_AMG_SA_Gen_Prolongators(MLI_AMG_SA *,MLI_Matrix **A,MLI_Matrix **P);

#ifdef __cplusplus
}
#endif

#endif

