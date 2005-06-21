/*--------------------------------------------------------------------------
 * Header file for test drivers
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE.h"
 
#define HYPRE_BICGSTAB   99100
#define HYPRE_BOOMERAMG  99110
#define HYPRE_CGNR       99120
#define HYPRE_DIAGSCALE  99130
#define HYPRE_EUCLID     99140
#define HYPRE_GMRES      99150
#define HYPRE_HYBRID     99160
#define HYPRE_JACOBI     99170
#define HYPRE_PARASAILS  99180
#define HYPRE_PCG        99190
#define HYPRE_PFMG       99200
#define HYPRE_PILUT      99210
#define HYPRE_SCHWARTZ   99220
#define HYPRE_SMG        99230
#define HYPRE_SPARSEMSG  99240
#define HYPRE_SPLIT      99250
#define HYPRE_SPLITPFMG  99260
#define HYPRE_SPLITSMG   99270
#define HYPRE_SYSPFMG    99280

/****************************************************************************
 * Prototypes for testing routines
 ***************************************************************************/
int hypre_set_precond(int matrix_id, int solver_id, int precond_id, 
                      void *solver, void *precond);

int hypre_set_precond_params(int precond_id, void *precond);

/****************************************************************************
 * Prototypes for testing routines
 ***************************************************************************/
int      k_dim = 5;
int      gsmg_samples = 5;
int      poutdat = 1;
int      hybrid = 1;
int      coarsen_type = 6;
int      measure_type = 0;
int      smooth_type = 6;
int      num_functions = 1;
int      smooth_num_levels = 0;
int      smooth_num_sweeps = 1;
int      num_sweep = 1;
int      max_levels = 25;
int      variant = 0;
int      overlap = 1;
int      domain_type = 2;
int      nonzeros_to_keep = -1;

int      interp_type; 
int      cycle_type;
int      relax_default;
int     *dof_func;
int     *num_grid_sweeps;  
int     *grid_relax_type;   
int    **grid_relax_points;

double   tol = 1.e-8;
double   pc_tol = 0.;
double   drop_tol = -1.;
double   max_row_sum = 1.;
double   schwarz_rlx_weight = 1.;
double   sai_threshold = 0.1;
double   sai_filter = 0.1;

double   strong_threshold;
double   trunc_factor;
double  *relax_weight; 
double  *omega;
