#ifndef MATGENFD_DH_DH
#define MATGENFD_DH_DH

/*
option summary:
    
    -c <int>            [4]    dimension of each processor's subgrid
    -px <int>           [1]    px,py,pz, and threeD determine proc grid geometry
    -py <int>           [1]
    -pz <int>           [0]
    -threeD <bool>      [0]
    -xx_coeff <double>  [-1.0]  next 3 are coefficients for 2nd derivatives
    -yy_coeff <double>  [-1.0]
    -zz_coeff <double>  [-1.0]
    -debug_MatGenFD     [0]
    -blocked_matrix     [1]
*/


/*
   "essential" variables, i.e, those which must be set for run() to work;
   these are initialized from the parser_dh when a MatGenFD object is
   created:
     px, py, pz
     cc
     threeD
*/

#include "euclid_common.h"

struct _matgenfd {
  bool allocateMem; 
        /* If true, memory is allocated when run() is called, in which case
         * the caller is responsible for calling FREE_DH for the rp, cval,
         * aval, and rhs arrays.  If false, caller is assumed to have
         * allocated memory when run is called.  
         * Default is "true"
         */
  int px, py, pz;  /* Processor graph dimensions; default is 1 */
  int cc;          /* Dimension of each processor's subgrid; default is 3 */
  double hh;       /* Grid spacing; this is constant,  equal to 1.0/(px*c-1) */
  int nx, ny, nz;  /* Nodal grid dimensions */
  double xx, yy, zz;  
                   /* Coordinates of right, top, and up sides of nodal grid; */
  bool threeD;     /* if true, run() generates 3D problem; if false, generates
                    * 2D problem.  Default is false, but if pz > 1 will become
                    * true when run() is called.  This only needs to be set
                    * when pz=1, but a 3D problem is to be generated.
                    */
  int id;          /* the processor whose submatrix is to be generated */
  int np;          /* Number of processors */
  int m, nnz, M;   /* Local matrix dimensions and non-zero count, and
                    * global matrix dimension; M = np*n.
                    */
  double stencil[8];


  double a, b, c, d, e, f, g, h;
                  /* constant parts of the derivative's coefficients;
                   * default values: a = b = c = 1.0, remainder = 0.0 
                   */
  int first;
  bool debugFlag;
                
  /* The following return coefficients; default is konstant() */
  double (*A)(double coeff, double x, double y, double z);
  double (*B)(double coeff, double x, double y, double z);
  double (*C)(double coeff, double x, double y, double z);
  double (*D)(double coeff, double x, double y, double z);
  double (*E)(double coeff, double x, double y, double z);
  double (*F)(double coeff, double x, double y, double z);
  double (*G)(double coeff, double x, double y, double z);
  double (*H)(double coeff, double x, double y, double z);
};

extern void MatGenFD_Create(MatGenFD *mg);
extern void MatGenFD_Destroy(MatGenFD mg);
extern void MatGenFD_Run(MatGenFD mg, int id, Mat_dh *A, Vec_dh *rhs);
extern void MatGenFD_Print(MatGenFD mg, FILE *fp);

 /* =========== coefficient functions ============== */
extern double konstant(double coeff, double x, double y, double z);
extern double e2_xy(double coeff, double x, double y, double z);


#endif
