#include <spmv.h>
extern "C" {
#include <mmio.h>
}
#define MAX_LINE 200

/*---------------------------------------------*
 *             READ COO Matrix Market          *
 *---------------------------------------------*/
int read_coo_MM(struct coo_t *coo, char *matfile, int mmidx) 
{
  int idx = mmidx == 0;

  MM_typecode matcode;
  FILE *p = fopen(matfile,"r");
  if (p == NULL) {
    printf("Unable to open file %s\n", matfile);
    exit(1);
  }
/*----------- READ MM banner */
  if (mm_read_banner(p, &matcode) != 0){
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }
  if (!mm_is_valid(matcode)){
    printf("Invalid Matrix Market file.\n");
    exit(1);
  }
  if (!(mm_is_real(matcode) && mm_is_coordinate(matcode) 
        && mm_is_sparse(matcode))) {
    printf("Only sparse real-valued coordinate \
    matrices are supported\n");
    exit(1);
  }
  int nrow, ncol, nnz, nnz2, k, j;
  char line[MAX_LINE];
/*------------- Read size */
  if (mm_read_mtx_crd_size(p, &nrow, &ncol, &nnz) !=0) {
    printf("MM read size error !\n");
    exit(1);
  }
  if (nrow != ncol) {
    fprintf(stdout,"This is not a square matrix!\n");
    exit(1);
  }
/*--------------------------------------
 * symmetric case : only L part stored,
 * so nnz2 := 2*nnz - nnz of diag,
 * so nnz2 <= 2*nnz 
 *-------------------------------------*/
  if (mm_is_symmetric(matcode))
    nnz2 = 2*nnz;
  else
    nnz2 = nnz;
/*-------- Allocate mem for COO */
  coo->ir  = (int *)  malloc(nnz2 * sizeof(int));
  coo->jc  = (int *)  malloc(nnz2 * sizeof(int));
  coo->val = (REAL *) malloc(nnz2 * sizeof(REAL));
/*-------- read line by line */
  char *p1, *p2;
  for (k=0; k<nnz; k++) {
    fgets(line, MAX_LINE, p);
    for( p1 = line; ' ' == *p1; p1++ );
/*----------------- 1st entry - row index */
    for( p2 = p1; ' ' != *p2; p2++ ); 
    *p2 = '\0';
    double tmp1 = atof(p1);
    //coo->ir[k] = atoi(p1);
    coo->ir[k] = (int) tmp1 + idx;
/*-------------- 2nd entry - column index */
    for( p1 = p2+1; ' ' == *p1; p1++ );
    for( p2 = p1; ' ' != *p2; p2++ );
    *p2 = '\0';
    double tmp2 = atof(p1);
    coo->jc[k] = (int) tmp2 + idx;
    //coo->jc[k]  = atoi(p1);      
/*------------- 3rd entry - nonzero entry */
    p1 = p2+1;
    coo->val[k] = atof(p1); 
  }
/*------------------ Symmetric case */
  j = nnz;
  if (mm_is_symmetric(matcode)) {
    for (k=0; k<nnz; k++)
      if (coo->ir[k] != coo->jc[k]) {
/*------------------ off-diag entry */
        coo->ir[j] = coo->jc[k];
        coo->jc[j] = coo->ir[k];
        coo->val[j] = coo->val[k];
        j++;
      }
    if (j != nnz2) {
      coo->ir  = (int *)realloc(coo->ir, j*sizeof(int));
      coo->jc  = (int *)realloc(coo->jc, j*sizeof(int));
      coo->val = (REAL*)realloc(coo->val,j*sizeof(REAL));
    }
  }
  coo->n = nrow;
  coo->nnz = j;
  printf("Matrix N = %d, NNZ = %d\n", nrow, j);
  fclose(p);

  return 0;
}

int computeidx(int nx, int ny, int nz, int ix, int iy, int iz, 
               int dx, int dy, int dz) {
   ix += dx;
   iy += dy;
   iz += dz;
   if (ix < 0 || ix >= nx) { return -1; }
   if (iy < 0 || iy >= ny) { return -1; }
   if (iz < 0 || iz >= nz) { return -1; }

   return (ix + nx*iy + nx*ny*iz);
}
/**-----------------------------------------------------------------------
 *
 * @brief Laplacean Matrix generator 
 *
 * @param[in] nx  Number of points in x-direction
 * @param[in] ny  Number of points in y-direction
 * @param[in] nz  Number of points in z-direction
 * @param[out] *Acoo matrix in coordinate format. 
 *
 -----------------------------------------------------------------------**/
int lapgen(int nx, int ny, int nz, struct coo_t *Acoo, int npts) {
  int n = nx * ny * nz;
  Acoo->n = n;

  if (nz > 1) {
     /* 3D */
    if (npts > 7) {
       npts = 27;
    } else {
       npts = 7;
    }
  } else {
     /* 2D */
    if (npts > 5) {
      npts = 9;
    } else {
      npts = 5;
    }
  }

  int nzmax = npts*n;
  Acoo->ir = (int*) malloc(nzmax*sizeof(int));
  Acoo->jc = (int*) malloc(nzmax*sizeof(int));
  Acoo->val = (REAL *) malloc(nzmax*sizeof(REAL));

  int ii, nnz=0;
  for (ii=0; ii<n; ii++) {
    double v = -1.0;
    int iz = ii / (nx*ny);
    int iy = (ii - iz*nx*ny) / nx;
    int ix = ii - iz*nx*ny - iy*nx;

    int jj;
    
    // front
    if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 0, 0, -1)) >= 0 ) {
      Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
    }
    // back
    if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 0, 0, 1)) >= 0 ) {
      Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
    }
    // down
    if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 0, -1, 0)) >= 0 ) {
      Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
    }
    // up
    if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 0, 1, 0)) >= 0 ) {
      Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
    }
    // left
    if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, 0, 0)) >= 0 ) {
      Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
    }
    // right
    if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, 0, 0)) >= 0 ) {
      Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
    }

    if (npts == 27 || npts == 9) {
      // front-down-left
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, -1, -1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // front-down
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 0, -1, -1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // front-down-right
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, -1, -1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // front-left
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, 0, -1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // front-right
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, 0, -1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // front-up-left
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, 1, -1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // front-up
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 0, 1, -1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // front-up-right
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, 1, -1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // down-left
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, -1, 0)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // down-right
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, -1, 0)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // up-left
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, 1, 0)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // up-right
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, 1, 0)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // back-down-left
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, -1, 1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // back-down
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 0, -1, 1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // back-down-right
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, -1, 1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // back-left
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, 0, 1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // back-right
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, 0, 1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // back-up-left
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, -1, 1, 1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // back-up
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 0, 1, 1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
      // back-up-right
      if ( (jj = computeidx(nx, ny, nz, ix, iy, iz, 1, 1, 1)) >= 0 ) {
        Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = jj;  Acoo->val[nnz] = v;  nnz++;
      }
    }

    v = npts - 1.0;
    Acoo->ir[nnz] = ii;  Acoo->jc[nnz] = ii;  Acoo->val[nnz] = v;  nnz++;
  }

  Acoo->nnz = nnz;

  printf("Lapcian Matrix N = %d, NNZ = %d\n", n, nnz);
  
  // change to 1-based index
  for (ii=0; ii<nnz; ii++) {
     Acoo->ir[ii] ++;
     Acoo->jc[ii] ++;
  }

  return 0;
}

// parse command-line input parameters
int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv) {
  int *outint;
  double *outdouble;
  char *outchar;
  int i;
  for (i=0; i<argc; i++) {
    if (argv[i][0] != '-') {
      continue;
    }
    if (!strcmp(argname, argv[i]+1)) {
      if (type == NA) {
        return 1;
      } else {
        if (i+1 >= argc /*|| argv[i+1][0] == '-'*/) {
          return 0;
        }
        switch (type) {
          case INT:
            outint = (int *) val;
            *outint = atoi(argv[i+1]);
            return 1;
            break;
          case DOUBLE:
            outdouble = (double *) val;
            *outdouble = atof(argv[i+1]);
            return 1;
            break;
          case STR:
            outchar = (char *) val;
            sprintf(outchar, "%s", argv[i+1]);
            return 1;
            break;
          default:
            printf("unknown arg type\n");
        }
      }
    }
  }
  return 0;
}

