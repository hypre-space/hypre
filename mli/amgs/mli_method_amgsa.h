/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the smoothed aggregation data structure
 *
 *****************************************************************************/

#ifndef __MLIMETHODAMGSAH__
#define __MLIMETHODAMGSAH__

#include "utilities/utilities.h"
#include <mpi.h>
#include "parcsr_mv/parcsr_mv.h"
#include "base/mli.h"
#include "base/mli_defs.h"
#include "matrix/mli_matrix.h"
#include "amgs/mli_method.h"

#define MLI_METHOD_AMGSA_LOCAL 0

/* ***********************************************************************
 * definition of the aggregation based AMG data structure
 * ----------------------------------------------------------------------*/

class MLI_Method_AMGSA : public MLI_Method
{
   int      max_levels;              /* the finest level is 0            */
   int      num_levels;              /* number of levels requested       */
   int      curr_level;              /* current level being processed    */
   int      output_level;            /* for diagnostics                  */
   int      node_dofs;               /* equation block size (fixed)      */
   int      curr_node_dofs;          /* current block size (this stage)  */
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

   MLI_Method_AMGSA( MPI_Comm comm );
   ~MLI_Method_AMGSA();
   int    setup( MLI *mli );
   int    setParams(char *name, int argc, char *argv[]);
   int    getParams(char *name, int *argc, char *argv[]);

   int    setOutputLevel( int outputLevel );
   int    setNumLevels( int nlevels );
   int    setSmoother( int prePost, int setID, int num, double *wgt );
   int    setCoarseSolver( int setID, int num, double *wgt );
   int    setCoarsenScheme( int scheme );
   int    setMinCoarseSize( int minSize );
   int    setStrengthThreshold( double thresh );
   int    setPweight( double weight );
   int    setCalcSpectralNorm();
   int    setAggregateInfo(int level, int naggr, int leng, int *aggrInfo);
   int    setNullSpace(int nodeDOF, int numNS, double *nullVec, int length);
   int    setNodalCoordinates(int nNodes,int nDOF,double *coor,double *scale);
   int    setCalibrationSize(int size);
   int    setupCalibration( MLI *mli );
   int    print();
   int    printStatistics(MLI *mli);
   int    getNullSpace(int &nodeDOF,int &numNS,double *&nullVec, int &leng);
   int    copy( MLI_Method * );

private :

   double genPLocal( MLI_Matrix *Amat, MLI_Matrix **Pmat, int, int * );
   int    formLocalGraph( hypre_ParCSRMatrix *Amat, hypre_ParCSRMatrix **graph);
   int    coarsenLocal( hypre_ParCSRMatrix *graph, int *nAggr, int **aggrInfo);
};

#endif

