#include "MatGenFD.h"
#include "Mat_dh.h"
#include "Vec_dh.h"
#include "Parser_dh.h"
#include "Mem_dh.h"

static char buf[128];

  /* handles for values in the 5-point (2D) or 7-point (for 3D) stencil */
#define FRONT(a)  a[5]
#define SOUTH(a)  a[3]
#define WEST(a)   a[1]
#define CENTER(a) a[0]
#define EAST(a)   a[2]
#define NORTH(a)  a[4]
#define BACK(a)   a[6]
#define RHS(a)    a[7]

  /* the following six functions return the ordering of the node's nabor,
   * or -1, if none exists.
   */
static int UpNabor(int node, int c, int firstUp, int ix, int iy, int iz);
static int DownNabor(int node, int c, int m, int firstDown, 
                                                 int ix, int iy, int iz); 
static int SouthNabor(int node, int c, int firstSouth, int ix, int iy, int iz);
static int NorthNabor(int node, int c, int firstNorth, int ix, int iy, int iz);
static int WestNabor(int node, int c, int firestWest, int ix, int iy, int iz);
static int EastNabor(int node, int c, int firstEast, int ix, int iy, int iz); 

static void generateStriped(MatGenFD mg, int *rp, int *cval, 
                                    double *aval, double *rhs);
static void generateBlocked(MatGenFD mg, int *rp, int *cval, 
                                    double *aval, double *rhs);
static void getstencil(MatGenFD g, int ix, int iy, int iz);
static void fdaddbc(int nx, int ny, int nz, int *rp, int *cval, 
             int *diag, double *aval, double *rhs, double h);

#undef __FUNC__
#define __FUNC__ "MatGenFDCreate"
void MatGenFD_Create(MatGenFD *mg)
{
  START_FUNC_DH
  struct _matgenfd* tmp =(struct _matgenfd*)MALLOC_DH(sizeof(struct _matgenfd)); CHECK_V_ERROR;
  *mg = tmp;
  Parser_dhReadInt(parser_dh,"-px",&tmp->px);
  Parser_dhReadInt(parser_dh,"-py",&tmp->py);
  Parser_dhReadInt(parser_dh,"-pz",&tmp->pz);
  Parser_dhReadInt(parser_dh,"-c",&tmp->cc);
  Parser_dhReadInt(parser_dh,"-threeD",&tmp->threeD);
  tmp->a = tmp->b = tmp->c = -1.0;
  tmp->d = tmp->e = tmp->f = tmp->g = tmp->h = 0.0; 

  Parser_dhReadDouble(parser_dh,"-xx_coeff",&tmp->a);
  Parser_dhReadDouble(parser_dh,"-yy_coeff",&tmp->b);
  Parser_dhReadDouble(parser_dh,"-zz_coeff",&tmp->c);

  tmp->allocateMem = true;
  if (Parser_dhHasSwitch(parser_dh, "-debug_MatGenFD")) {
    tmp->debugFlag = true;
  } else {
    tmp->debugFlag = false;
  }

  tmp->A = tmp->B = tmp->C = tmp->D = tmp->E 
         =  tmp->F = tmp->G = tmp->H = konstant;
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
void MatGenFD_Run(MatGenFD mg, int id, Mat_dh *AOut, Vec_dh *rhsOut)
{
/* What this function does:
 *   0. creates return objects (A and rhs)
 *   1. computes "nice to have" values;
 *   2. allocates storage, if required;
 *   3. calls generateBlocked() or generateStriped().
 *   4. initializes variable in A and rhs.
 */

  START_FUNC_DH
  /* 0. create objects */
  Mat_dh A;
  Vec_dh rhs;
  Mat_dhCreate(AOut); CHECK_V_ERROR;
  Vec_dhCreate(rhsOut); CHECK_V_ERROR;
  A = *AOut;
  rhs = *rhsOut;

  if (! mg->debugFlag) {
    int np = mg->px;
    if (mg->py) np *= mg->py;
    if (mg->threeD) np *= mg->pz;
    if (np != np_dh) {
      sprintf(buf, "numbers don't match: np_dh = %i, px*py*pz = %i", np_dh, np);
      SET_V_ERROR(buf);
    }
  }


  /* 1. compute "nice to have" values */
  if (mg->pz > 1) mg->threeD = true;
  mg->nx = mg->px*mg->cc;
  mg->ny = mg->py*mg->cc;
  if (mg->threeD) mg->nz = mg->pz*mg->cc; 
  else            mg->nz = 0;
  mg->hh = 1.0/(mg->nx - 1);
  mg->np = mg->px*mg->py*mg->pz;
  mg->id = id;
  if (mg->threeD) {
    mg->M = mg->nx*mg->ny*mg->nz;
    mg->m = mg->cc*mg->cc*mg->cc;
    mg->nnz = mg->m*7;
  } else {
    mg->M = mg->nx*mg->ny;
    mg->m = mg->cc*mg->cc;
    mg->nnz = mg->m*5;
  }
  mg->first = mg->id*mg->m;

  if (mg->debugFlag) {
    MatGenFD_Print(mg, logFile);
    fflush(logFile);
  }

  /* 2. allocate storage, if required */
  if (mg->allocateMem) {
    A->rp = (int*)MALLOC_DH((2*mg->m)*sizeof(int)); CHECK_V_ERROR;
    A->cval = (int*)MALLOC_DH(mg->nnz*sizeof(int)); CHECK_V_ERROR
    A->aval = (double*)MALLOC_DH(mg->nnz*sizeof(double)); CHECK_V_ERROR;
    rhs->vals = (double*)MALLOC_DH(mg->m*sizeof(double)); CHECK_V_ERROR;
  }

  /* 3. generate matrix */
  if (Parser_dhHasSwitch(parser_dh,"-blocked_matrix")) {
    if (mg->debugFlag) SET_INFO("calling generateBlocked");
    generateBlocked(mg, A->rp, A->cval, A->aval, rhs->vals); CHECK_V_ERROR;
  } else {
    if (mg->debugFlag) SET_INFO("calling generateStriped");
    generateStriped(mg, A->rp, A->cval, A->aval, rhs->vals); CHECK_V_ERROR;
  }
  /* add in bdry conditions */
/*  fdaddbc(nx, ny, nz, rp, cval, diag, aval, rhs, h); */

  /* 4. initialize variables in A and rhs */
  rhs->n = mg->m;

  A->m = mg->m;
  A->n = mg->M;
  A->beg_row = mg->first;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "generateStriped"
void generateStriped(MatGenFD mg, int *rp, int *cval, double *aval, double *rhs)
{
  START_FUNC_DH
  int i, j, k;
  int yStart, yEnd, zStart, zEnd;
  int xStart = 0, xEnd = mg->nx;
  int idx  = 0;   /* indexes cval and aval arrays */
  int row;
  bool threeD = mg->threeD;
  double *stencil = mg->stencil;
  int nx, ny, nz;
  int first;
  int pp;  /* number of nodes in a plane */
  int slice;


  /* adjust some values to ensure all processors have the same
   * number of matrix rows.
   */
  if (threeD) {
    slice = (mg->pz*mg->cc)/mg->np;
    zStart = mg->id*slice;
    zEnd   = zStart+slice;
    yStart = 0;
    yEnd   = mg->ny; 
    mg->nz = slice*mg->np;
    mg->M = mg->nx*mg->ny*mg->nz;
    mg->m = slice*mg->nx*mg->ny;
  } else {
    slice = (mg->py*mg->cc)/mg->np;
    yStart = mg->id*slice;
    yEnd = yStart+slice;
    zStart = 0;
    zEnd = 1;
    mg->ny = slice*mg->np;
    mg->M = mg->nx*mg->ny;
    mg->m = slice*mg->nx;
  }
  first = mg->first = mg->id*mg->m;
  row = mg->first;

  if (mg->debugFlag) {
    fprintf(logFile, "@@@ MatGenFD possibly changed values:\n");
    fprintf(logFile, "   slice = %i ny = %i nz = %i  m = %i  M = %i  first = %i\n",
                             slice, mg->ny, mg->nz, mg->m, mg->M, mg->first);
    fflush(logFile);
  }

  nx = mg->nx;
  ny = mg->ny;
  nz = mg->nz;
  pp = nx*ny;  /* number of nodes in plane */
  rp[0] = 0;  

  for (i = zStart; i < zEnd; ++i) {   /* loop over planes (z-direction) */
    for (j = yStart; j < yEnd; ++j) {  /* loop over rows (y-direction) */
      for (k = xStart; k < xEnd; ++k) { /* loop over columns (x-direction) */
        bool bdryFlag = false;

        /* compute column values and rhs entry for the current node */
        getstencil(mg,k,j,i);


        /* only homogenous Dirichlet boundary conditions presently supported */
        if (threeD) {
          if (j==0 || k==0 || j==ny-1 || k==nx-1 || i==0 || i==nz-1) {
            bdryFlag = true; 
          }
        } else {
          if (j==0 || k==0 || j==ny-1 || k==nx-1) {
            bdryFlag = true; 
          }
        }

        /* it's a boundary node */
/*
        if (bdryFlag) {
          cval[idx] = row;
          aval[idx++] = 1.0;
        }
*/

       /* it's an iterior node */
       /* else */{

        /* down plane */
        if (threeD) {
          if (i > 0) {
            cval[idx]   = row - pp;
            aval[idx++] = BACK(stencil);
          }
        }

        /* south */
        if (j > 0) {
          cval[idx] = row - nx;
          aval[idx++] = SOUTH(stencil);
        }

        /* west */
        if (k > 0) {
          cval[idx] = row - 1;
          aval[idx++] = WEST(stencil);
        }

        /* center node */
        cval[idx]   = row;
        aval[idx++] = CENTER(stencil);

        /* east */
        if (k < nx-1) {
          cval[idx] = row + 1;
          aval[idx++] = EAST(stencil);
        }

        /* north */
        if (j < ny-1) {
          cval[idx] = row + nx;
          aval[idx++] = NORTH(stencil);
        }

        /* up plane */
        if (threeD) {
          if (i < nz-1) {
            cval[idx]   = row + pp;
            aval[idx++] = FRONT(stencil);
          }
        }

       } /* else, interior node */
        ++row;
        rp[row-first] = idx;
      }
    }
  }
  END_FUNC_DH
}

int UpNabor(int node, int c, int firstUp, int ix, int iy, int iz) {
  if (iz == c-1 && firstUp == -1) return -1;
  if (iz < c-1) return(node + c*c );
  return(firstUp + iy*c+ix);
}

int DownNabor(int node, int c, int m, int firstDown, int ix, int iy, int iz) {
  if (iz == 0 && firstDown == -1) return -1;
  if (iz > 0) return(node - c*c );
  return(firstDown + m - c*c +  iy*c+ix);
}

int SouthNabor(int node, int c, int firstSouth, int ix, int iy, int iz) {
  if (iy == 0 && firstSouth ==  -1) return -1;
  if (iy > 0) return (node-c);
  return(firstSouth + (iz+1)*c*c - c+ix);
}

int NorthNabor(int node, int c, int firstNorth, int ix, int iy, int iz) {
  if (iy == c-1 && firstNorth ==  -1) return -1;
  if (iy < c-1) return (node+c);
  return (firstNorth + iz*c*c + ix);
}

int WestNabor(int node, int c, int firstWest, int ix, int iy, int iz) {
  if (ix == 0 && firstWest == -1) return -1;
  if (ix) return(node-1);
  return(firstWest+(iy+1)*c-1+iz*c*c);
}

int EastNabor(int node, int c, int firstEast, int ix, int iy, int iz) {
  if (ix == c-1 && firstEast == -1) return -1;
  if (ix < c-1) return(node+1);
  return(firstEast+(iy*c)+iz*c*c);
}

void getstencil(MatGenFD g, int ix, int iy, int iz)
{
  int k; 
  double h = g->hh;
  double hhalf = h*0.5;
  double x = h*ix;
  double y = h*iy;
  double z = h*iz;
  double cntr = 0.0;
  double *stencil = g->stencil;
  double coeff;

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
  if (g->threeD) {
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
{ return coeff; }

double e2_xy(double coeff, double x, double y, double z)
{ return exp(coeff*x*y); }

#undef __FUNC__
#define __FUNC__ "MatGenFD_Print"
void MatGenFD_Print(MatGenFD g, FILE *fp)
{
  START_FUNC_DH
  fprintf(fp, "--------------------------- MatGenFD_Print:\n");
  fprintf(fp, "allocateMem = %i\n", g->allocateMem);
  fprintf(fp, "px = %i  py = %i  pz = %i\n", g->px, g->py, g->pz);
  fprintf(fp, "cc = %i  hh = %g\n", g->cc, g->hh);
  fprintf(fp, "nx = %i  ny = %i  nz = %i\n", g->nx, g->ny, g->nz);
  fprintf(fp, "threeD = %i\n", g->threeD);
  fprintf(fp, "id = %i  np = %i\n", g->id, g->np);
  fprintf(fp, "m = %i  nnz = %i  M = %i\n", g->m, g->nnz, g->M);
  fprintf(fp, "a = %g  b = %g  c = %g\n", g->a, g->b, g->c);
  fprintf(fp, "d = %g  e = %g  f = %g \n", g->d, g->e, g->f);
  fprintf(fp, "g = %g  h = %g\n", g->g, g->h);
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "generateBlocked"
void generateBlocked(MatGenFD mg, int *rp, int *cval, double *aval, double *rhs)
{
  START_FUNC_DH
  double *stencil = mg->stencil;
  int idx  = 0, rhsIdx=0, i,j,k, c=mg->cc, zEnd;
  int id = mg->id, idTmp = id, np = mg->np, m = mg->m;
  int nabor, node = id*m;
  bool threeD = mg->threeD;
  int px = mg->px, py = mg->py, pp = px*py;
  int firstUp = -1, firstDown= -1, firstNorth = -1, firstSouth = -1, 
      firstEast = -1, firstWest = -1, first = id*m;

  if (threeD) {
    if (id >= pp) firstDown = (id-pp)*m;
    if (id < np-pp) firstUp = (id+pp)*m;
    idTmp = id % pp;  /* project proc's id to plane 1 */
  } 
/*SET_INFO(setMsg("idTMP === %i, px= %i  py= %i", idTmp, px, py));
*/
  if (idTmp < pp-px) firstNorth = (id+px)*m;
  if (idTmp >= px)   firstSouth = (id-px)*m;
  if (py > 1) idTmp = idTmp % px; /* progect proc's id to row 1 */
/*SET_INFO(setMsg("idTMP === %i, px= %i  py= %i", idTmp, px, py));
*/
  if (idTmp > 0) firstWest = (id-1)*m;
  if (idTmp < px-1) firstEast = (id+1)*m;


  /* get the x, y, and z dimensions of this processor's grid points */
  /* getSubgridIndices(mg, &xStart, &yStart, &zStart); */
  zEnd = mg->threeD ? mg->cc : 1;
/*
fprintf(logFile, "\nfirstNorth=%i  firstUp=%i  firstEast=%i\n", firstNorth,firstUp,firstEast);
fprintf(logFile, "firstSouth=%i  firstDown=%i  firstWest=%i\n", 
          firstSouth,firstDown,firstWest);
mg->print(mg, stdout);
*/


  rp[0] = 0;
  for (i = 0; i < zEnd; ++i) {  
    for (j = 0; j < c; ++j) {  
      for (k = 0; k < c; ++k) {


        /* only homogenous Dirichlet boundary conditions presently supported */
        bool bdryFlag = false;
        if ((j==0 && firstSouth==-1) || 
            (j==c-1 && firstNorth==-1) || 
            (k==0 && firstWest==-1) || 
            (k==c-1 && firstEast==-1)) { 
          bdryFlag = true; 
        } else if (threeD && 
                   ((i==0 && firstDown==-1) || 
                    (i==c-1 && firstUp==-1))) {
          bdryFlag = true; 
        }

        /* compute row values and rhs, at the current node */
        getstencil(mg,k,j,i);

        /* it's a boundary node */
/*
        if (bdryFlag) {
          cval[idx] = node;
          aval[idx++] = 1.0;
        }
*/

       /* it's an interior node */
/*       else*/ {

        /* down plane */
        if (threeD) {
          if ((nabor = DownNabor(node,c,m,firstDown,k,j,i)) != -1) {
            cval[idx]   = nabor;
            aval[idx++] = FRONT(stencil);
          }
        }

        /* south */
        if ((nabor = SouthNabor(node,c,firstSouth,k,j,i)) > -1) { 
          cval[idx]   = nabor;
          aval[idx++] = SOUTH(stencil);
        }

        /* west */
        if ((nabor = WestNabor(node,c,firstWest,k,j,i)) > -1) { 
          cval[idx]   = nabor;
          aval[idx++] = WEST(stencil);
        }

        /* center node */
        cval[idx]   = node;
        aval[idx++] = CENTER(stencil);

        /* east */
        if ((nabor = EastNabor(node,c,firstEast,k,j,i)) != -1) { 
          cval[idx]   = nabor;
          aval[idx++] = EAST(stencil);
        }

        /* north */
        if ((nabor = NorthNabor(node,c,firstNorth,k,j,i)) != -1) { 
          cval[idx]   = nabor;
          aval[idx++] = NORTH(stencil);
        }

        /* up plane */
        if (threeD) {
          if ((nabor = UpNabor(node,c,firstUp,k,j,i)) != -1) {
            cval[idx]   = nabor;
            aval[idx++] = BACK(stencil);
          }
        }
       } /* else, interior node */

       /* rhs[rhsIdx++] = RHS(stencil); */
       rhs[rhsIdx++] = 5.5;

        ++node;
        rp[node-first] = idx; 
      }
    }
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "fdaddbc"
void fdaddbc(int nx, int ny, int nz, int *rp, int *cval, 
             int *diag, double *aval, double *rhs, double h)
{
/*
c-----------------------------------------------------------------------
c This subroutine will add the boundary condition to the system with
c the right interior point values.
c
c The Boundary condition is specified in the following form:
c           du
c     alpha -- + beta u = gamma
c           dn
c-----------------------------------------------------------------------
*/

/*

  int i,j,k,node,nbr,ly,uy,lx,ux, count;
  double coeff, ctr, x, y, z, *ptr;
 
  double hhalf = 0.5 * h;
  int kx = 1;
  int ky = nx;
  int kz = nx*ny;
  double *alpha = coeff_->alpha();

  //adjust nodes on X1 side of the grid
  x = 0.0;
  PDE_Coeff::Side s = PDE_Coeff::X1;

  //note: for 2D, nx = ny = n, nz = 1
  //      for 3D, nx = ny = nz = n
  //      (original code permitted different n to vary in x, y, and z)

  for (k=0; k<nz; ++k) {
    z = k*h;
    for (j=0; j<ny; ++j) {
      y = j*h;
      node = j*ky+k*kz;
 
      //case 1: Dirichlet Boundary condition 
      if (alpha[0] == 0.0) {
        ptr = aval+rp[node];
        count = rp[node+1]-rp[node];
        for (i=count; i>0; --i) *ptr++ = 0.0;
        aval[diag[node]] = coeff_->beta(x,y,z,s);
        rhs[node] = coeff_->gamma(x,y,z,s);
      }

      //case 2: general boundary condition
      else {
        //adjust rhs
        coeff = 2.0*coeff_->Afun(x,y,z);
        ctr   = (h*coeff_->Dfun(x,y,z) - coeff)*h/alpha[0];
        rhs[node] += (ctr * coeff_->gamma(x,y,z,s));
        //adjust diagonal
        ctr   = coeff_->Afun(x-hhalf,y,z) + ctr * coeff_->beta(x,y,z,s);
        coeff = coeff_->Afun(x+hhalf,y,z);
        aval[diag[node]] += (ctr - coeff);
        //find array index of node's right neighbor and adjust
        nbr = rp[node];
        while (cval[nbr] != node+kx) ++nbr;
        aval[nbr] = 2.0*coeff;
      }
    }
  }

  //adjust nodes on X2 side of the grid
  x = 1.0;
  s = PDE_Coeff::X2;

  for (k=0; k<nz; ++k) {
    z = k*h;
    for (j=0; j<ny; ++j) {
      y = j*h;
      node = (j+1)*ky+k*kz-1;

      //case 1: Dirichlet Boundary condition 
      if (alpha[1] == 0.0) {
        ptr = aval+rp[node];
        count = rp[node+1]-rp[node];
        for (i=count; i>0; --i) *ptr++ = 0.0;
        aval[diag[node]] = coeff_->beta(x,y,z,s);
        rhs[node] = coeff_->gamma(x,y,z,s);
      }

      //case 2: general boundary condition
      else {
        //adjust rhs
        coeff = 2.0*coeff_->Afun(x,y,z);
        ctr   = (h*coeff_->Dfun(x,y,z) - coeff)*h/alpha[1];
        rhs[node] += (ctr * coeff_->gamma(x,y,z,s));
        //adjust diagonal
        ctr   = coeff_->Afun(x+hhalf,y,z) + ctr * coeff_->beta(x,y,z,s);
        coeff = coeff_->Afun(x-hhalf,y,z);
        aval[diag[node]] += (ctr - coeff);
        //find array index of node's right neighbor and adjust
        nbr = rp[node];
        while (cval[nbr] != node-kx) ++nbr;
        aval[nbr] = 2.0*coeff;
      }
    }
  }
*/

/*
c
c     the bottom (south) side suface, This similar to the situation
c     with the left side, except all the function and realted variation
c     should be on the y.
c
c     These two block if statment here is to resolve the possible conflict
            (2 trinary statements!)
c     of assign the boundary value differently by different side of the
c     Dirichlet Boundary Conditions. They ensure that the edges that have
c     be assigned a specific value will not be reassigned.
c
*/

/*
  //adjust nodes on Y1 side of the grid
  y = 0.0;
  s = PDE_Coeff::Y1;

  lx = (alpha[0] == 0.0) ? 1 : 0;
  ux = (alpha[1] == 0.0) ? nz : nx-1;

  for (k=0; k<nz; ++k) {
    z = k*h;
    for (i=lx; i<ux; ++i) {
      x = i*h;
      node = i+k*kz;
 
      //case 1: Dirichlet Boundary condition
      if (alpha[2] == 0.0) {
        ptr = aval+rp[node];
        count = rp[node+1]-rp[node];
        for (i=count; i>0; --i) *ptr++ = 0.0;
        aval[diag[node]] = coeff_->beta(x,y,z,s);
        rhs[node] = coeff_->gamma(x,y,z,s);
      }

      //case 2: general boundary condition
      else {
        //adjust rhs
        coeff = 2.0*coeff_->Bfun(x,y,z);
        ctr   = (h*coeff_->Efun(x,y,z) - coeff)*h/alpha[2];
        rhs[node] += (ctr * coeff_->gamma(x,y,z,s));
        //adjust diagonal
        ctr   = coeff_->Bfun(x,y-hhalf,z) + ctr * coeff_->beta(x,y,z,s);
        coeff = coeff_->Bfun(x,y+hhalf,z);
        aval[diag[node]] += (ctr - coeff);
        //find array index of node's right neighbor and adjust
        nbr = rp[node];
        while (cval[nbr] != node+ky) ++nbr;
        aval[nbr] = 2.0*coeff;
      }
    }
  }

  //adjust nodes on Y2 side of the grid
  y = (ny-1)*h;
  s = PDE_Coeff::Y2;

  for (k=0; k<nz; ++k) {
    z = k*h;
    for (i=lx; i<ux; ++i) {
      x = i*h;
      node = k*kz+ny*ky+i; 

      //case 1: Dirichlet Boundary condition
      if (alpha[3] == 0.0) {
        ptr = aval+rp[node];
        count = rp[node+1]-rp[node];
        for (i=count; i>0; --i) *ptr++ = 0.0;
        aval[diag[node]] = coeff_->beta(x,y,z,s);
        rhs[node] = coeff_->gamma(x,y,z,s);
      }

      //case 2: general boundary condition
      else {
        //adjust rhs
        coeff = 2.0*coeff_->Bfun(x,y,z);
        ctr   = (h*coeff_->Efun(x,y,z) - coeff)*h/alpha[3];
        rhs[node] += (ctr * coeff_->gamma(x,y,z,s));
        //adjust diagonal
        ctr   = coeff_->Bfun(x,y+hhalf,z) + ctr * coeff_->beta(x,y,z,s);
        coeff = coeff_->Bfun(x,y-hhalf,z);
        aval[diag[node]] += (ctr - coeff);
        //find array index of node's right neighbor and adjust
        nbr = rp[node];
        while (cval[nbr] != node-ky) ++nbr;
        aval[nbr] = 2.0*coeff;
      }
    }
  }

//===============================================================
// if grid is 2D, we're finished
//===============================================================
  if (nz == 1) return;

  //adjust nodes on Z1 side of the grid
  y = (ny-1)*h;
  s = PDE_Coeff::Z1;

  ly = (alpha[2] == 0.0) ? 1 : 0;
  uy = (alpha[3] == 0.0) ? ny : ny-1;

  for (j=ly; j<uy; ++j) {
    y = j*h;
    for (i=lx; i<ux; ++i) {
      x = i*h;
      node = i+j*ky;

      //case 1: Dirichlet Boundary condition
      if (alpha[4] == 0.0) {
        ptr = aval+rp[node];
        count = rp[node+1]-rp[node];
        for (i=count; i>0; --i) *ptr++ = 0.0;
        aval[diag[node]] = coeff_->beta(x,y,z,s);
        rhs[node] = coeff_->gamma(x,y,z,s);
      }

      //case 2: general boundary condition
      else {
        //adjust rhs
        coeff = 2.0*coeff_->Cfun(x,y,z);
        ctr   = (h*coeff_->Ffun(x,y,z) - coeff)*h/alpha[4];
        rhs[node] += (ctr * coeff_->gamma(x,y,z,s));
        //adjust diagonal
        ctr   = coeff_->Cfun(x,y,z-hhalf) + ctr * coeff_->beta(x,y,z,s);
        coeff = coeff_->Cfun(x,y,z+hhalf);
        aval[diag[node]] += (ctr - coeff);
        //find array index of node's right neighbor and adjust
        nbr = rp[node];
        while (cval[i] != node+kz) ++nbr;
        aval[nbr] = 2.0*coeff;
      }
    }
  }

  //adjust nodes on Z2 side of the grid
  y = (ny-1)*h;
  s = PDE_Coeff::Z1;
  for (j=ly; j<uy; ++j) {
    y = j*h;
    for (i=lx; i<ux; ++i) {
      x = i*h;
      node = nz*kz+j*ky+i;

      //case 1: Dirichlet Boundary condition
      if (alpha[5] == 0.0) {
        ptr = aval+rp[node];
        count = rp[node+1]-rp[node];
        for (i=count; i>0; --i) *ptr++ = 0.0;
        aval[diag[node]] = coeff_->beta(x,y,z,s);
        rhs[node] = coeff_->gamma(x,y,z,s);
      }

      //case 2: general boundary condition
      else {
        //adjust rhs
        coeff = 2.0*coeff_->Cfun(x,y,z);
        ctr   = (h*coeff_->Ffun(x,y,z) - coeff)*h/alpha[5];
        rhs[node] += (ctr * coeff_->gamma(x,y,z,s));
        //adjust diagonal
        ctr   = coeff_->Cfun(x,y,z+hhalf) + ctr * coeff_->beta(x,y,z,s);
        coeff = coeff_->Cfun(x,y,z-hhalf);
        aval[diag[node]] += (ctr - coeff);
        //find array index of node's right neighbor and adjust
        nbr = rp[node];
        while (cval[i] != node-kz) ++nbr;
        aval[nbr] = 2.0*coeff;
      }
    }
  }
*/
}
