/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the top level MLI data structure
 *
 *****************************************************************************/

#ifndef __MLIAMGSAH__
#define __MLIAMGSAH__

#include "utilities/utilities.h"
#include <mpi.h>
#include "parcsr_mv/parcsr_mv.h"
#include "../base/mli.h"
#include "../base/mli_defs.h"
#include "../matrix/mli_matrix.h"

#define MLI_AMGSA_LOCAL 0

/* ***********************************************************************
 * definition of the aggregation based AMG data structure
 * ----------------------------------------------------------------------*/

class MLI_AMGSA
{
   MPI_Comm mpi_comm;
   char     method_name[80];         /* store name of the AMG method     */
   int      max_levels;              /* the finest level is 0            */
   int      num_levels;              /* number of levels requested       */
   int      curr_level;              /* current level being processed    */
   int      output_level;            /* for diagnostics                  */
   int      node_dofs;               /* equation block size (fixed)      */
   double   threshold;               /* for creating aggregation graph   */
   int      nullspace_dim;           /* null space information (changes  */
   int      nullspace_len;           /* length of nullspace in each dim  */
   double   *nullspace_vec;          /* as curr_level)                   */
   double   P_weight;                /* weight for prolongator smoother  */
   double   drop_tol_for_P;          /* tolerance for sparsifying P      */
   int      *sa_counts;              /* store aggregation information at */
   int      **sa_data;               /* each level                       */
   double   *spectral_norms;         /* computed matrix norms            */
   int      calc_norm_scheme;        /* method to estimate matrix norm   */
   int      min_coarse_size;         /* tell when to stop aggregation    */
   int      coarsen_scheme;          /* different aggregation schemes    */
   int      pre_smoother;            /* denote which pre-smoother to use */
   int      postsmoother;            /* denote which postsmoother to use */
   int      pre_smoother_num;        /* number of pre-smoother sweeps    */
   int      postsmoother_num;        /* number of postsmoother sweeps    */
   double   *pre_smoother_wgt;       /* weight used in pre-smoother      */
   double   *postsmoother_wgt;       /* weight used in postsmoother      */
   int      coarse_solver;           /* denote which coarse solver to use*/
   int      coarse_solver_num;       /* number of coarse solver sweeps   */
   double   *coarse_solver_wgt;      /* weight used in coarse solver     */
   int      calibration_size;        /* for calibration AMG method       */
   double   RAP_time;
   double   total_time;

public :

   MLI_AMGSA( MPI_Comm comm );
   ~MLI_AMGSA();
   int    setOutputLevel( int output_level );
   int    setNumLevels( int nlevels );
   int    setSmoother( int pre_post, int set_id, int num, double *wgt );
   int    setCoarseSolver( int set_id, int num, double *wgt );
   int    setCoarsenScheme( int scheme );
   int    setMinCoarseSize( int min_size );
   int    setStrengthThreshold( double thresh );
   int    setPweight( double weight );
   int    setCalcSpectralNorm();
   int    setNullSpace(int node_dofs, int num_ns, double *null_vec, int length);
   int    setCalibrationSize(int size);
   int    genMLStructure( MLI *mli );
   int    genMLStructureCalibration( MLI *mli );
   int    print();
   int    printStatistics(MLI *mli);
   double genPLocal( MLI_Matrix *Amat, MLI_Matrix **Pmat );
   int    getNullSpace(int &node_dofs,int &num_ns,double *&null_vec, int &leng);
   int    reinitialize();

private :

   int    formLocalGraph( hypre_ParCSRMatrix *Amat, hypre_ParCSRMatrix **graph);
   int    coarsenLocal( hypre_ParCSRMatrix *graph, int *naggr, int **aggr_info);
};

#endif

