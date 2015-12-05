/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




#include "MatGenFD.h"
#include "Mat_dh.h"
#include "Vec_dh.h"
#include "Parser_dh.h"
#include "Mem_dh.h"
/* #include "graphColor_dh.h" */

static bool isThreeD;
 
  /* handles for values in the 5-point (2D) or 7-point (for 3D) stencil */
#define FRONT(a)  a[5]
#define SOUTH(a)  a[3]
#define WEST(a)   a[1]
#define CENTER(a) a[0]
#define EAST(a)   a[2]
#define NORTH(a)  a[4]
#define BACK(a)   a[6]
#define RHS(a)    a[7]

static void setBoundary_private(HYPRE_Int node, HYPRE_Int *cval, double *aval, HYPRE_Int len,
                 double *rhs, double bc, double coeff, double ctr, HYPRE_Int nabor);
static void generateStriped(MatGenFD mg, HYPRE_Int *rp, HYPRE_Int *cval, 
                                    double *aval, Mat_dh A, Vec_dh b);
static void generateBlocked(MatGenFD mg, HYPRE_Int *rp, HYPRE_Int *cval, double *aval, 
                                                         Mat_dh A, Vec_dh b);
static void getstencil(MatGenFD g, HYPRE_Int ix, HYPRE_Int iy, HYPRE_Int iz);

#if 0
static void fdaddbc(HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, HYPRE_Int *rp, HYPRE_Int *cval, 
             HYPRE_Int *diag, double *aval, double *rhs, double h, MatGenFD mg);
#endif

#undef __FUNC__
#define __FUNC__ "MatGenFDCreate"
void MatGenFD_Create(MatGenFD *mg)
{
  START_FUNC_DH
  struct _matgenfd* tmp =(struct _matgenfd*)MALLOC_DH(sizeof(struct _matgenfd)); CHECK_V_ERROR;
  *mg = tmp;

  tmp->debug = Parser_dhHasSwitch(parser_dh, "-debug_matgen");

  tmp->m = 9;
  tmp->px = tmp->py = 1;
  tmp->pz = 0;
  Parser_dhReadInt(parser_dh,"-m",&tmp->m);
  Parser_dhReadInt(parser_dh,"-px",&tmp->px);
  Parser_dhReadInt(parser_dh,"-py",&tmp->py);
  Parser_dhReadInt(parser_dh,"-pz",&tmp->pz);

  if (tmp->px < 1) tmp->px = 1;
  if (tmp->py < 1) tmp->py = 1;
  if (tmp->pz < 0) tmp->pz = 0;
  tmp->threeD = false;
  if (tmp->pz) {
    tmp->threeD = true;
  } else {
    tmp->pz = 1;
  }
  if (Parser_dhHasSwitch(parser_dh,"-threeD")) tmp->threeD = true;

  tmp->a = tmp->b = tmp->c = 1.0;
  tmp->d = tmp->e = tmp->f = 0.0;
  tmp->g = tmp->h = 0.0; 

  Parser_dhReadDouble(parser_dh,"-dx",&tmp->a);
  Parser_dhReadDouble(parser_dh,"-dy",&tmp->b);
  Parser_dhReadDouble(parser_dh,"-dz",&tmp->c);
  Parser_dhReadDouble(parser_dh,"-cx",&tmp->d);
  Parser_dhReadDouble(parser_dh,"-cy",&tmp->e);
  Parser_dhReadDouble(parser_dh,"-cz",&tmp->f);

  tmp->a = -1*fabs(tmp->a);
  tmp->b = -1*fabs(tmp->b);
  tmp->c = -1*fabs(tmp->c);

  tmp->allocateMem = true;

  tmp->A = tmp->B = tmp->C = tmp->D = tmp->E 
         =  tmp->F = tmp->G = tmp->H = konstant;

  tmp->bcX1 = tmp->bcX2 = tmp->bcY1 = tmp->bcY2
            = tmp->bcZ1 = tmp->bcZ2 = 0.0;
  Parser_dhReadDouble(parser_dh,"-bcx1",&tmp->bcX1);
  Parser_dhReadDouble(parser_dh,"-bcx2",&tmp->bcX2);
  Parser_dhReadDouble(parser_dh,"-bcy1",&tmp->bcY1);
  Parser_dhReadDouble(parser_dh,"-bcy2",&tmp->bcY2);
  Parser_dhReadDouble(parser_dh,"-bcz1",&tmp->bcZ1);
  Parser_dhReadDouble(parser_dh,"-bcz2",&tmp->bcZ2);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "MatGenFD_Destroy"
void MatGenFD_Destroy(MatGenFD mg)
{
  START_FUNC_DH
  FREE_DH(mg); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "MatGenFD_Run"
void MatGenFD_Run(MatGenFD mg, HYPRE_Int id, HYPRE_Int np, Mat_dh *AOut, Vec_dh *rhsOut)
{
/* What this function does:
 *   0. creates return objects (A and rhs)
 *   1. computes "nice to have" values;
 *   2. allocates storage, if required;
 *   3. calls generateBlocked() or generateStriped().
 *   4. initializes variable in A and rhs.
 */

  START_FUNC_DH
  Mat_dh A;
  Vec_dh rhs;
  bool threeD = mg->threeD;
  HYPRE_Int nnz;
  HYPRE_Int m = mg->m; /* local unknowns */
  bool debug = false, striped;

  if (mg->debug && logFile != NULL) debug = true;
  striped = Parser_dhHasSwitch(parser_dh,"-striped");

  /* 0. create objects */
  Mat_dhCreate(AOut); CHECK_V_ERROR;
  Vec_dhCreate(rhsOut); CHECK_V_ERROR;
  A = *AOut;
  rhs = *rhsOut;

  /* ensure that processor grid contains the same number of
     nodes as there are processors.
  */
  if (! Parser_dhHasSwitch(parser_dh, "-noChecks")) {
    if (!striped) {
      HYPRE_Int npTest = mg->px*mg->py;
      if (threeD) npTest *= mg->pz;
      if (npTest != np) {
        hypre_sprintf(msgBuf_dh, "numbers don't match: np_dh = %i, px*py*pz = %i", np, npTest);
        SET_V_ERROR(msgBuf_dh);
      }
    }
  }

  /* 1. compute "nice to have" values */
  /* each proc's subgrid dimension */
  mg->cc = m;
  if (threeD) { 
    m = mg->m = m*m*m;
  } else {
    m = mg->m = m*m;
  }    

  mg->first = id*m;
  mg->hh = 1.0/(mg->px*mg->cc - 1);
  
  if (debug) {
    hypre_sprintf(msgBuf_dh, "cc (local grid dimension) = %i", mg->cc);
    SET_INFO(msgBuf_dh);
    if (threeD) { hypre_sprintf(msgBuf_dh, "threeD = true"); }
    else            { hypre_sprintf(msgBuf_dh, "threeD = false"); }
    SET_INFO(msgBuf_dh);
    hypre_sprintf(msgBuf_dh, "np= %i  id= %i", np, id);
    SET_INFO(msgBuf_dh);
  }

  mg->id = id;
  mg->np = np;
  nnz = threeD ? m*7 : m*5;

  /* 2. allocate storage */
  if (mg->allocateMem) {
    A->rp = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
    A->rp[0] = 0;  
    A->cval = (HYPRE_Int*)MALLOC_DH(nnz*sizeof(HYPRE_Int)); CHECK_V_ERROR
    A->aval = (double*)MALLOC_DH(nnz*sizeof(double)); CHECK_V_ERROR;
    /* rhs->vals = (double*)MALLOC_DH(m*sizeof(double)); CHECK_V_ERROR; */
  }

  /* 4. initialize variables in A and rhs */
  rhs->n = m;
  A->m = m;
  A->n = m*mg->np;
  A->beg_row = mg->first;

  /* 3. generate matrix */
  isThreeD = threeD; /* yuck!  used in box_XX() */
  if (Parser_dhHasSwitch(parser_dh,"-striped")) {
    generateStriped(mg, A->rp, A->cval, A->aval, A, rhs); CHECK_V_ERROR;
  } else {
    generateBlocked(mg, A->rp, A->cval, A->aval, A, rhs); CHECK_V_ERROR;
  } 

  /* add in bdry conditions */
  /* only implemented for 2D mats! */
  if (! threeD) {
/*  fdaddbc(nx, ny, nz, rp, cval, diag, aval, rhs, h, mg); */
  }

  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "generateStriped"
void generateStriped(MatGenFD mg, HYPRE_Int *rp, HYPRE_Int *cval, double *aval, Mat_dh A, Vec_dh b)
{
  START_FUNC_DH
  HYPRE_Int mGlobal;
  HYPRE_Int m = mg->m;
  HYPRE_Int beg_row, end_row;
  HYPRE_Int i, j, k, row;
  bool threeD = mg->threeD;
  HYPRE_Int idx = 0;
  double *stencil = mg->stencil;
  bool debug = false;
  HYPRE_Int plane, nodeRemainder;
  HYPRE_Int naborx1, naborx2, nabory1, nabory2;
  double *rhs;

  bool applyBdry = true;
  double hhalf;
  double bcx1 = mg->bcX1;
  double bcx2 = mg->bcX2;
  double bcy1 = mg->bcY1;
  double bcy2 = mg->bcY2;
  /* double bcz1 = mg->bcZ1; */
  /* double bcz2 = mg->bcZ2; */
  HYPRE_Int nx, ny;

  printf_dh("@@@ using striped partitioning\n");

  if (mg->debug && logFile != NULL) debug = true;

  /* recompute values (yuck!) */
  m = 9;
  Parser_dhReadInt(parser_dh,"-m", &m);  /* global grid dimension */
  mGlobal = m*m;                         /* global unkknowns */
  if (threeD) mGlobal *= m;
  i = mGlobal/mg->np;                    /* unknowns per processor */
  beg_row = i*mg->id;                    /* global number of 1st local row */
  end_row = beg_row + i;
  if (mg->id == mg->np-1) end_row = mGlobal;
  nx = ny = m;

  mg->hh = 1.0/(m-1);
  hhalf = 0.5 * mg->hh;

  A->n = m*m;
  A->m = end_row - beg_row;
  A->beg_row = beg_row;

  Vec_dhInit(b, A->m); CHECK_V_ERROR;
  rhs = b->vals;

  plane = m*m;

  if (debug) {
    hypre_fprintf(logFile, "generateStriped: beg_row= %i; end_row= %i; m= %i\n", beg_row+1, end_row+1, m);
  }

  for (row = beg_row; row<end_row; ++row) {
        HYPRE_Int localRow = row-beg_row;

        /* compute current node's position in grid */
        k = (row / plane);      
        nodeRemainder = row - (k*plane); /* map row to 1st plane */
        j = nodeRemainder / m;
        i = nodeRemainder % m;

        if (debug) {
          hypre_fprintf(logFile, "row= %i  x= %i  y= %i  z= %i\n", row+1, i,j,k);
        }

        /* compute column values and rhs entry for the current node */
        getstencil(mg,i,j,k);

        /* only homogenous Dirichlet boundary conditions presently supported */

        /* down plane */
        if (threeD) {
          if (k > 0) {
            cval[idx]   = row - plane;
            aval[idx++] = BACK(stencil);
          }
        }

        /* south */
        if (j > 0) {
          nabory1 = cval[idx] = row - m;
          aval[idx++] = SOUTH(stencil);
        }

        /* west */
        if (i > 0) {
          naborx1 = cval[idx] = row - 1;
          aval[idx++] = WEST(stencil);
        }

        /* center node */
        cval[idx]   = row;
        aval[idx++] = CENTER(stencil);

        /* east */
        if (i < m-1) {
          naborx2 = cval[idx] = row + 1;
          aval[idx++] = EAST(stencil);
        }

        /* north */
        if (j < m-1) {
          nabory2 = cval[idx] = row + m;
          aval[idx++] = NORTH(stencil);
        }

        /* up plane */
        if (threeD) {
          if (k < m-1) {
            cval[idx]   = row + plane;
            aval[idx++] = FRONT(stencil);
          }
        }
       rhs[localRow] = 0.0;
       ++localRow;
       rp[localRow] = idx; 

       /* apply boundary conditions; only for 2D! */
       if (!threeD && applyBdry) {
         HYPRE_Int offset = rp[localRow-1];
         HYPRE_Int len = rp[localRow] - rp[localRow-1];
         double ctr, coeff;

/* hypre_fprintf(logFile, "globalRow = %i; naborx2 = %i\n", row+1, row); */

         if (i == 0) {         /* if x1 */
           coeff = mg->A(mg->a, i+hhalf,j,k);
           ctr   = mg->A(mg->a, i-hhalf,j,k);
           setBoundary_private(row, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcx1, coeff, ctr, naborx2);
         } else if (i == nx-1) {  /* if x2 */
           coeff = mg->A(mg->a, i-hhalf,j,k);
           ctr   = mg->A(mg->a, i+hhalf,j,k);
           setBoundary_private(row, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcx2, coeff, ctr, naborx1);
         } else if (j == 0) {  /* if y1 */
           coeff = mg->B(mg->b, i, j+hhalf,k);
           ctr   = mg->B(mg->b, i, j-hhalf,k);
           setBoundary_private(row, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcy1, coeff, ctr, nabory2);
         } else if (j == ny-1) {        /* if y2 */
           coeff = mg->B(mg->b, i, j-hhalf,k);
           ctr   = mg->B(mg->b, i, j+hhalf,k);
           setBoundary_private(row, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcy2, coeff, ctr, nabory1);
         }
       }
  }
  END_FUNC_DH
}


/* zero-based 
   (from Edmond Chow)
*/
/* 
   x,y,z       -  coordinates of row, wrt naturally ordered grid
   nz, ny, nz  -  local grid dimensions, wrt 0
   P, Q        -  subdomain grid dimensions in x and y directions
*/
HYPRE_Int rownum(const bool threeD, const HYPRE_Int x, const HYPRE_Int y, const HYPRE_Int z, 
   const HYPRE_Int nx, const HYPRE_Int ny, const HYPRE_Int nz, HYPRE_Int P, HYPRE_Int Q)
{
   HYPRE_Int p, q, r;
   HYPRE_Int lowerx, lowery, lowerz;
   HYPRE_Int id, startrow;


   /* compute x,y,z coordinates of subdomain to which
      this row belongs.
    */
   p = x/nx;
   q = y/ny;
   r = z/nz;

/*
if (myid_dh == 0) hypre_printf("nx= %i  ny= %i  nz= %i\n", nx, ny, nz);
if (myid_dh == 0) hypre_printf("x= %i y= %i z= %i  threeD= %i  p= %i q= %i r= %i\n",
              x,y,z,threeD, p,q,r);
*/

   /* compute the subdomain (processor) of the subdomain to which
      this row belongs.
    */
   if (threeD) {
     id = r*P*Q+q*P+p;
   } else {
     id = q*P+p;
   }

/*  if (myid_dh == 0) hypre_printf(" id= %i\n", id);
*/

   /* smallest row in the subdomain */
   startrow = id*(nx*ny*nz);

   /* x,y, and z coordinates of local grid of unknowns */
   lowerx = nx*p;
   lowery = ny*q;
   lowerz = nz*r;
   
   if (threeD) { 
     return startrow + nx*ny*(z-lowerz) + nx*(y-lowery) + (x-lowerx);
   } else {
     return startrow + nx*(y-lowery) + (x-lowerx);
   }
}



void getstencil(MatGenFD g, HYPRE_Int ix, HYPRE_Int iy, HYPRE_Int iz)
{
  HYPRE_Int k; 
  double h = g->hh;
  double hhalf = h*0.5;
  double x = h*ix;
  double y = h*iy;
  double z = h*iz;
  double cntr = 0.0;
  double *stencil = g->stencil;
  double coeff;
  bool threeD = g->threeD;

  for (k=0; k<8; ++k) stencil[k] = 0.0;

  /* differentiation wrt x */
  coeff = g->A(g->a, x+hhalf,y,z);
  EAST(stencil) += coeff;
  cntr += coeff;

  coeff = g->A(g->a, x-hhalf,y,z);
  WEST(stencil) += coeff;
  cntr += coeff;

  coeff = g->D(g->d, x,y,z)*hhalf;
  EAST(stencil) += coeff;
  WEST(stencil) -= coeff;

  /* differentiation wrt y */
  coeff = g->B(g->b,x,y+hhalf,z);
  NORTH(stencil) += coeff;
  cntr += coeff;

  coeff = g->B(g->b,x,y-hhalf,z);
  SOUTH(stencil) += coeff;
  cntr += coeff;

  coeff = g->E(g->e,x,y,z)*hhalf;
  NORTH(stencil) += coeff;
  SOUTH(stencil) -= coeff;

  /* differentiation wrt z */
  if (threeD) {
    coeff = g->C(g->c,x,y,z+hhalf);
    BACK(stencil) += coeff;
    cntr += coeff;

    coeff = g->C(g->c,x,y,z-hhalf);
    FRONT(stencil) += coeff;
    cntr += coeff;

    coeff = g->F(g->f,x,y,z)*hhalf;
    BACK(stencil) += coeff;
    FRONT(stencil) -= coeff;
  }

  /* contribution from function G: */
  coeff = g->G(g->g,x,y,z);
  CENTER(stencil) = h*h*coeff - cntr;

  RHS(stencil) = h*h*g->H(g->h,x,y,z);
}


double konstant(double coeff, double x, double y, double z)
{  return coeff; }

double e2_xy(double coeff, double x, double y, double z)
{ return exp(coeff*x*y); }

double boxThreeD(double coeff, double x, double y, double z);

/* returns diffusivity constant -bd1 if the point
   (x,y,z) is inside the box whose upper left and
   lower right points are (-bx1,-by1), (-bx2,-by2);
   else, returns diffusivity constant -bd2
*/
double box_1(double coeff, double x, double y, double z)
{
  static bool setup = false;
  double retval = coeff;
 
  /* dffusivity constants */
  static double dd1 = BOX1_DD;
  static double dd2 = BOX2_DD;
  static double dd3 = BOX3_DD;  

  /* boxes */
  static double ax1 = BOX1_X1, ay1 = BOX1_Y1;
  static double ax2 = BOX1_X2, ay2 = BOX1_Y2;
  static double bx1 = BOX2_X1, by1 = BOX2_Y1;
  static double bx2 = BOX2_X2, by2 = BOX2_Y2;
  static double cx1 = BOX3_X1, cy1 = BOX3_Y1;
  static double cx2 = BOX3_X2, cy2 = BOX3_Y2;

  if (isThreeD) {
    return(boxThreeD(coeff,x,y,z));
  }


  /* 1st time through, parse for dffusivity constants */
  if (!setup ) {
    dd1 = 0.1;
    dd2 = 0.1;
    dd3 = 10;
    Parser_dhReadDouble(parser_dh,"-dd1",&dd1);
    Parser_dhReadDouble(parser_dh,"-dd2",&dd2);
    Parser_dhReadDouble(parser_dh,"-dd3",&dd3);
    Parser_dhReadDouble(parser_dh,"-box1x1",&cx1);
    Parser_dhReadDouble(parser_dh,"-box1x2",&cx2);
    setup = true;
  }

  /* determine if point is inside box a */
  if (x > ax1 && x < ax2 && y > ay1 && y < ay2) {
    retval = dd1*coeff;
  }

  /* determine if point is inside box b */
  if (x > bx1 && x < bx2 && y > by1 && y < by2) {
    retval = dd2*coeff;
  }

  /* determine if point is inside box c */
  if (x > cx1 && x < cx2 && y > cy1 && y < cy2) {
    retval = dd3*coeff;
  }

  return retval;
} 

double boxThreeD(double coeff, double x, double y, double z)
{
  static bool setup = false;
  double retval = coeff;

  /* dffusivity constants */
  static double dd1 = 100;

  /* boxes */
  static double x1 = .2, x2 = .8;
  static double y1 = .3, y2 = .7;
  static double z1 = .4, z2 = .6;

  /* 1st time through, parse for diffusivity constants */
  if (!setup ) {
    Parser_dhReadDouble(parser_dh,"-dd1",&dd1);
    setup = true;
  }

  /* determine if point is inside the box */
  if (x > x1 && x < x2 && y > y1 && y < y2 && z > z1 && z < z2) {
    retval = dd1*coeff;
  }

  return retval;
} 

#if 0
double box_1(double coeff, double x, double y, double z)
{
  static double x1, x2, y1, y2;
  static double d1, d2;
  bool setup = false;
  double retval;

  /* 1st time through, parse for constants and
     bounding box definition
  */
  if (!setup ) {
    x1 = .25; x2 = .75; y1 = .25; y2 = .75;
    d1 = 1; d2 = 2;
    Parser_dhReadDouble(parser_dh,"-bx1",&x1);
    Parser_dhReadDouble(parser_dh,"-bx2",&x2);
    Parser_dhReadDouble(parser_dh,"-by1",&y1);
    Parser_dhReadDouble(parser_dh,"-by2",&y2);
    Parser_dhReadDouble(parser_dh,"-bd1",&d1);
    Parser_dhReadDouble(parser_dh,"-bd2",&d2);
    setup = true;
  }

  retval = d2;

  /* determine if point is inside box */
  if (x > x1 && x < x2 && y > y1 && y < y2) {
    retval = d1;
  }

  return -1*retval;
} 
#endif

/* divide square into 4 quadrants; return one of
   2 constants depending on the quadrant (checkerboard)
*/
double box_2(double coeff, double x, double y, double z)
{
  bool setup = false;
  static double d1, d2;
  double retval;

  if (!setup ) {
    d1 = 1; d2 = 2;
    Parser_dhReadDouble(parser_dh,"-bd1",&d1);
    Parser_dhReadDouble(parser_dh,"-bd2",&d2);
  }

  retval = d2;

  if (x < .5 && y < .5) retval = d1;
  if (x > .5 && y > .5) retval = d1;

  return -1*retval;
}


#undef __FUNC__
#define __FUNC__ "generateBlocked"
void generateBlocked(MatGenFD mg, HYPRE_Int *rp, HYPRE_Int *cval, double *aval, Mat_dh A, Vec_dh b)
{
  START_FUNC_DH
  bool applyBdry = true;
  double *stencil = mg->stencil;
  HYPRE_Int id = mg->id;
  bool threeD = mg->threeD;
  HYPRE_Int px = mg->px, py = mg->py, pz = mg->pz; /* processor grid dimensions */
  HYPRE_Int p, q, r; /* this proc's position in processor grid */
  HYPRE_Int cc = mg->cc; /* local grid dimension (grid of unknowns) */
  HYPRE_Int nx = cc, ny = cc, nz = cc;
  HYPRE_Int lowerx, upperx, lowery, uppery, lowerz, upperz;
  HYPRE_Int startRow;
  HYPRE_Int x, y, z;
  bool debug = false;
  HYPRE_Int idx = 0, localRow = 0; /* nabor; */
  HYPRE_Int naborx1, naborx2, nabory1, nabory2, naborz1, naborz2;
  double *rhs;

  double hhalf = 0.5 * mg->hh;
  double bcx1 = mg->bcX1;
  double bcx2 = mg->bcX2;
  double bcy1 = mg->bcY1;
  double bcy2 = mg->bcY2;
  /* double bcz1 = mg->bcZ1; */
  /* double bcz2 = mg->bcZ2; */

  Vec_dhInit(b, A->m); CHECK_V_ERROR;
  rhs = b->vals;

  if (mg->debug && logFile != NULL) debug = true;
  if (! threeD) nz = 1;

  /* compute p,q,r from P,Q,R and myid */
  p = id % px;
  q = (( id - p)/px) % py;
  r = ( id - p - px*q)/( px*py );

  if (debug) {
    hypre_sprintf(msgBuf_dh, "this proc's position in subdomain grid: p= %i  q= %i  r= %i", p,q,r);
    SET_INFO(msgBuf_dh);
  }

   /* compute ilower and iupper from p,q,r and nx,ny,nz */
   /* zero-based */

   lowerx = nx*p;
   upperx = lowerx + nx;
   lowery = ny*q;
   uppery = lowery + ny;
   lowerz = nz*r;
   upperz = lowerz + nz;

  if (debug) {
    hypre_sprintf(msgBuf_dh, "local grid parameters: lowerx= %i  upperx= %i", lowerx, upperx);
    SET_INFO(msgBuf_dh);
    hypre_sprintf(msgBuf_dh, "local grid parameters: lowery= %i  uppery= %i", lowery, uppery);
    SET_INFO(msgBuf_dh);
    hypre_sprintf(msgBuf_dh, "local grid parameters: lowerz= %i  upperz= %i", lowerz, upperz);
    SET_INFO(msgBuf_dh);
  }

  startRow = mg->first;
  rp[0] = 0;

  for (z=lowerz; z<upperz; z++) {
    for (y=lowery; y<uppery; y++) {
      for (x=lowerx; x<upperx; x++) {

        if (debug) {
          hypre_fprintf(logFile, "row= %i  x= %i  y= %i  z= %i\n", localRow+startRow+1, x, y, z);
        }

        /* compute row values and rhs, at the current node */
        getstencil(mg,x,y,z);

        /* down plane */
        if (threeD) {
          if (z > 0) {
            naborz1 = rownum(threeD, x,y,z-1,nx,ny,nz,px,py);
            cval[idx]   = naborz1;
            aval[idx++] = FRONT(stencil);
          }
        }

        /* south */
        if (y > 0) {
          nabory1 = rownum(threeD, x,y-1,z,nx,ny,nz,px,py);
          cval[idx]   = nabory1;
          aval[idx++] = SOUTH(stencil);
        }

        /* west */
        if (x > 0) {
          naborx1 = rownum(threeD, x-1,y,z,nx,ny,nz,px,py);
          cval[idx]   = naborx1;
          aval[idx++] = WEST(stencil);
/*hypre_fprintf(logFile, "--- row: %i;  naborx1= %i\n", localRow+startRow+1, 1+naborx1);
*/
        }
/*
else {
hypre_fprintf(logFile, "--- row: %i;  x >= nx*px-1; naborx1 has old value: %i\n", localRow+startRow+1,1+naborx1);
}
*/

        /* center node */
        cval[idx]   = localRow+startRow;
        aval[idx++] = CENTER(stencil);


        /* east */
        if (x < nx*px-1) {
          naborx2 = rownum(threeD,x+1,y,z,nx,ny,nz,px,py);
          cval[idx]   = naborx2;
          aval[idx++] = EAST(stencil);
        }
/*
else {
hypre_fprintf(logFile, "--- row: %i;  x >= nx*px-1; nobors2 has old value: %i\n", localRow+startRow,1+naborx2);
}
*/

        /* north */
        if (y < ny*py-1) {
          nabory2 = rownum(threeD,x,y+1,z,nx,ny,nz,px,py);
          cval[idx]   = nabory2;
          aval[idx++] = NORTH(stencil);
        }

        /* up plane */
        if (threeD) {
          if (z < nz*pz-1) {
            naborz2 = rownum(threeD,x,y,z+1,nx,ny,nz,px,py);
            cval[idx]   = naborz2;
            aval[idx++] = BACK(stencil);
          }
        }

       /* rhs[rhsIdx++] = RHS(stencil); */
       rhs[localRow] = 0.0;

       ++localRow;
       rp[localRow] = idx; 

       /* apply boundary conditions; only for 2D! */
       if (!threeD && applyBdry) {
         HYPRE_Int globalRow = localRow+startRow-1;
         HYPRE_Int offset = rp[localRow-1];
         HYPRE_Int len = rp[localRow] - rp[localRow-1];
         double ctr, coeff;

/* hypre_fprintf(logFile, "globalRow = %i; naborx2 = %i\n", globalRow+1, naborx2+1); */

         if (x == 0) {         /* if x1 */
           coeff = mg->A(mg->a, x+hhalf,y,z);
           ctr   = mg->A(mg->a, x-hhalf,y,z);
           setBoundary_private(globalRow, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcx1, coeff, ctr, naborx2);
         } else if (x == nx*px-1) {  /* if x2 */
           coeff = mg->A(mg->a, x-hhalf,y,z);
           ctr   = mg->A(mg->a, x+hhalf,y,z);
           setBoundary_private(globalRow, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcx2, coeff, ctr, naborx1);
         } else if (y == 0) {  /* if y1 */
           coeff = mg->B(mg->b, x, y+hhalf,z);
           ctr   = mg->B(mg->b, x, y-hhalf,z);
           setBoundary_private(globalRow, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcy1, coeff, ctr, nabory2);
         } else if (y == ny*py-1) {        /* if y2 */
           coeff = mg->B(mg->b, x, y-hhalf,z);
           ctr   = mg->B(mg->b, x, y+hhalf,z);
           setBoundary_private(globalRow, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcy2, coeff, ctr, nabory1);
         } else if (threeD) {
           if (z == 0) {
             coeff = mg->B(mg->b, x, y, z+hhalf);
             ctr   = mg->B(mg->b, x, y, z-hhalf);
             setBoundary_private(globalRow, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcy1, coeff, ctr, naborz2);
           } else if (z == nz*nx-1) {
             coeff = mg->B(mg->b, x, y, z-hhalf);
             ctr   = mg->B(mg->b, x, y, z+hhalf);
             setBoundary_private(globalRow, cval+offset, aval+offset, len,
                               &(rhs[localRow-1]), bcy1, coeff, ctr, naborz1);
           }
         }
       }
      }
    }
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "setBoundary_private"
void setBoundary_private(HYPRE_Int node, HYPRE_Int *cval, double *aval, HYPRE_Int len,
                               double *rhs, double bc, double coeff, double ctr, HYPRE_Int nabor)
{
  START_FUNC_DH
  HYPRE_Int i;

  /* case 1: Dirichlet Boundary condition  */
  if (bc >= 0) {
    /* set all values to zero, set the diagonal to 1.0, set rhs to "bc" */ 
    *rhs = bc;
    for (i=0; i<len; ++i) {
      if (cval[i] == node) {
        aval[i] = 1.0;
      } else {
        aval[i] = 0;
      }
    }
  }

  /* case 2: neuman */
  else {
/* hypre_fprintf(logFile, "node= %i  nabor= %i  coeff= %g\n", node+1, nabor+1, coeff); */
    /* adjust row values */
    for (i=0; i<len; ++i) {
      /* adjust diagonal */
      if (cval[i] == node) {
        aval[i] += (ctr - coeff);
      /* adust node's right neighbor */
      } else if (cval[i] == nabor) { 
        aval[i] = 2.0*coeff;
      }
    }
  }
  END_FUNC_DH
}
