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
#include "parcsr_mv/parcsr_mv.h"
#include "base/mli.h"
#include "matrix/mli_matrix.h"
#include "amgs/mli_method.h"

#define MLI_METHOD_AMGSA_LOCAL        0
#define MLI_METHOD_AMGSA_HYBRID       1

/* ********************************************************************* *
 * internal data structure used for domain decomposition
 * --------------------------------------------------------------------- */

typedef struct MLI_AMGSA_DD_Struct
{
   int nSends;
   int nRecvs;
   int *sendLengs;
   int *recvLengs;
   int *sendProcs;
   int *recvProcs;
   int *sendMap;
   int nSendMap;
   int NNodes;
   int *ANodeEqnList;
   int *SNodeEqnList;
   int dofPerNode;
}
MLI_AMGSA_DD;

/* ***********************************************************************
 * definition of the aggregation based AMG data structure
 * ----------------------------------------------------------------------*/

class MLI_Method_AMGSA : public MLI_Method
{
   int      maxLevels_;              /* the finest level is 0            */
   int      numLevels_;              /* number of levels requested       */
   int      currLevel_;              /* current level being processed    */
   int      outputLevel_;            /* for diagnostics                  */
   int      scalar_;                 /* aggregate in scalar manner       */
   int      nodeDofs_;               /* equation block size (fixed)      */
   int      currNodeDofs_ ;          /* current block size (this stage)  */
   double   threshold_;              /* for creating aggregation graph   */
   int      nullspaceDim_;           /* null space information (changes  */
   int      nullspaceLen_;           /* length of nullspace in each dim  */
   double   *nullspaceVec_;          /* as curr_level)                   */
   int      numSmoothVec_;           /* if nonzero, use smooth vectors   */
   double   Pweight_;                /* weight for prolongator smoother  */
   double   dropTolForP_  ;          /* tolerance for sparsifying P      */
   int      *saCounts_;              /* store aggregation information at */
   int      **saData_;               /* each level                       */
   int      **saLabels_;             /* labels for aggregation           */
   int      **saDataAux_;            /* for subdomain aggregates         */
   double   *spectralNorms_;         /* computed matrix norms            */
   int      calcNormScheme_;         /* method to estimate matrix norm   */
   int      minAggrSize_;            /* tell when to stop aggregation    */
   int      minCoarseSize_;          /* tell when to stop aggregation    */
   int      coarsenScheme_;          /* different aggregation schemes    */
   char     preSmoother_[20];        /* denote which pre-smoother to use */
   char     postSmoother_[20];       /* denote which postsmoother to use */
   int      preSmootherNum_;         /* number of pre-smoother sweeps    */
   int      postSmootherNum_;        /* number of postsmoother sweeps    */
   double   *preSmootherWgt_;        /* weight used in pre-smoother      */
   double   *postSmootherWgt_;       /* weight used in postsmoother      */
   int	    smootherPrintRNorm_;     /* tell smoother to print rnorm     */
   int	    smootherFindOmega_;      /* tell smoother to find good omega */
   char     coarseSolver_[20];       /* denote which coarse solver to use*/
   int      coarseSolverNum_;        /* number of coarse solver sweeps   */
   double   *coarseSolverWgt_;       /* weight used in coarse solver     */
   int      calibrationSize_;        /* for calibration AMG method       */
   int      symmetric_;              /* choose between symm/nonsymm      */
   int      useSAMGeFlag_;           /* element based method             */
   int      useSAMGDDFlag_;          /* domain decomposition (NN) method */
   double   RAPTime_;
   double   totalTime_;
   int      ARPACKSuperLUExists_;
   MLI_AMGSA_DD *ddObj_;
   char     paramFile_[100];
   int      printToFile_;

public :

   MLI_Method_AMGSA( MPI_Comm comm );
   ~MLI_Method_AMGSA();
   int    setup( MLI *mli );
   int    setParams(char *name, int argc, char *argv[]);
   int    getParams(char *name, int *argc, char *argv[]);

   int    setOutputLevel( int outputLevel );
   int    setNumLevels( int nlevels );
   int    setSmoother( int prePost, char *stype, int num, double *wgt );
   int    setCoarseSolver( char *stype, int num, double *wgt );
   int    setCoarsenScheme( int scheme );
   int    setMinAggregateSize( int minSize );
   int    setMinCoarseSize( int minSize );
   int    setStrengthThreshold( double thresh );
   int    setSmoothVec( int num );
   int    setPweight( double weight );
   int    setCalcSpectralNorm();
   int    setAggregateInfo(int level, int naggr, int leng, int *aggrInfo);
   int    setNullSpace(int nodeDOF, int numNS, double *nullVec, int length);
   int    resetNullSpaceComponents(int length, int start, int *indices);
   int    adjustNullSpace(double *vecAdjust);
   int    setNodalCoordinates(int nNodes,int nDOF, int nsDim, double *coor, 
                              int numNS, double *scale);
   int    setCalibrationSize(int size);
   int    setupCalibration( MLI *mli );
   int    setupFEDataBasedNullSpaces( MLI *mli );
   int    setupFEDataBasedAggregates( MLI *mli );
   int    setupFEDataBasedSuperLUSmoother( MLI *mli, int level );
   int    setupSFEIBasedNullSpaces( MLI *mli );
   int    setupSFEIBasedAggregates( MLI *mli );
   int    setupSFEIBasedSuperLUSmoother( MLI *mli, int level );
   int    print();
   int    printStatistics(MLI *mli);
   int    getNullSpace(int &nodeDOF,int &numNS,double *&nullVec, int &leng);
   int    copy( MLI_Method * );
   int    relaxNullSpaces(MLI_Matrix *mat);

private :

   double genP( MLI_Matrix *Amat, MLI_Matrix **Pmat, int, int * );
   double genPGlobal(hypre_ParCSRMatrix *Amat, MLI_Matrix **Pmat, int, int *);
   int    formLocalGraph( hypre_ParCSRMatrix *Amat, hypre_ParCSRMatrix **graph,
                          int *labels);
   int    formGlobalGraph(hypre_ParCSRMatrix *Amat,hypre_ParCSRMatrix **graph);
   int    coarsenLocal( hypre_ParCSRMatrix *graph, int *nAggr, int **aggrInfo);
   int    coarsenGlobal(hypre_ParCSRMatrix *graph, int *nAggr, int **aggrInfo);
};

#endif

