/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef MATGENFD_DH_DH
#define MATGENFD_DH_DH

/*=====================================================================
option summary:
---------------
processor topology
     -px <HYPRE_Int> -py <HYPRE_Int> -pz <HYPRE_Int>
     defaults:  -px 1 -py 1 -pz 0

grid topology
  -m <HYPRE_Int>
  if pz=0, each processor has a square grid of dimension m*m,
  hence there are m*m*px*py unknowns.
  if pz > 0, each local grid is of dimension m*m*m, hence
  there are m*m*m*px*py*pz unknowns.


diffusion coefficients (default is 1.0):
    -dx <HYPRE_Real> -dy <HYPRE_Real> -dz <HYPRE_Real>

convection coefficients (default is 0.0)
    -cx <HYPRE_Real> -cy <HYPRE_Real> -cz <HYPRE_Real>

grid dimension; if more than one mpi process, this is
the local size for each processor:
     -m <HYPRE_Int>

boundary conditions:
  This is very primitive; boundary conditions can only be generated for
  2D grids; the condition along each side is either dirichlet (constant),
  if bcXX >= 0, or neuman, if bcXX < 0.

   -bcx1 <HYPRE_Real>
   -bcx2 <HYPRE_Real>
   -bcy1 <HYPRE_Real>
   -bcy2 <HYPRE_Real>

Misc.
     -debug_matgen
     -striped (may not work?)
=====================================================================*/

/* #include "euclid_common.h" */

struct _matgenfd {
  bool allocateMem; 
        /* If true, memory is allocated when run() is called, in which case
         * the caller is responsible for calling FREE_DH for the rp, cval,
         * aval, and rhs arrays.  If false, caller is assumed to have
         * allocated memory when run is called.  
         * Default is "true"
         */
  HYPRE_Int px, py, pz;  /* Processor graph dimensions */
  bool threeD;  
  HYPRE_Int m;           /* number of matrix rows in local matrix */
  HYPRE_Int cc;          /* Dimension of each processor's subgrid */
  HYPRE_Real hh;       /* Grid spacing; this is constant,  equal to 1.0/(px*cc-1) */
  HYPRE_Int id;          /* the processor whose submatrix is to be generated */
  HYPRE_Int np;          /* number of subdomains (processors, mpi tasks) */
  HYPRE_Real stencil[8];


  /* derivative coefficients; a,b,c are 2nd derivatives, 
   * c,d,e are 1st derivatives; f,g,h not currently used.
   */
  HYPRE_Real a, b, c, d, e, f, g, h;

  HYPRE_Int first; /* global number of first locally owned row */
  bool debug;

  /* boundary conditions; if value is < 0, neumen; else, dirichelet */
  HYPRE_Real bcX1, bcX2;
  HYPRE_Real bcY1, bcY2;
  HYPRE_Real bcZ1, bcZ2;
                
  /* The following return coefficients; default is konstant() */
  HYPRE_Real (*A)(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
  HYPRE_Real (*B)(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
  HYPRE_Real (*C)(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
  HYPRE_Real (*D)(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
  HYPRE_Real (*E)(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
  HYPRE_Real (*F)(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
  HYPRE_Real (*G)(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
  HYPRE_Real (*H)(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
};

extern void MatGenFD_Create(MatGenFD *mg);
extern void MatGenFD_Destroy(MatGenFD mg);
extern void MatGenFD_Run(MatGenFD mg, HYPRE_Int id, HYPRE_Int np, Mat_dh *A, Vec_dh *rhs);

 /* =========== coefficient functions ============== */
extern HYPRE_Real konstant(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
extern HYPRE_Real e2_xy(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);



/* 3 boxes nested inside the unit square domain.
   diffusivity constants are: -dd1, -dd2, -dd3.
*/
/* box placement */
#define BOX1_X1 0.1
#define BOX1_X2 0.4
#define BOX1_Y1 0.1
#define BOX1_Y2 0.4

#define BOX2_X1 0.6
#define BOX2_X2 0.9
#define BOX2_Y1 0.1
#define BOX2_Y2 0.4

#define BOX3_X1 0.2
#define BOX3_X2 0.8
#define BOX3_Y1 0.6
#define BOX3_Y2 0.8

/* default diffusivity */
#define BOX1_DD  10
#define BOX2_DD  100
#define BOX3_DD  50

extern HYPRE_Real box_1(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);
  /* -bd2 is diffusion coeff outside box;
     -bd1 is diffusion coeff inside box.
  */
     


extern HYPRE_Real box_2(HYPRE_Real coeff, HYPRE_Real x, HYPRE_Real y, HYPRE_Real z);

#endif
