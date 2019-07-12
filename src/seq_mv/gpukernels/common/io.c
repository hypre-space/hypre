#include "spkernels.h"
#include "mmio.h"
#define MAX_LINE 200

/*---------------------------------------------*
 *             READ COO Matrix Market          *
 *---------------------------------------------*/
HYPRE_Int read_coo_MM(const char *matfile, HYPRE_Int idxin, HYPRE_Int idxout, struct coo_t *Acoo) {
  MM_typecode matcode;
  FILE *p = fopen(matfile,"r");
  HYPRE_Int i;
  if (p == NULL) {
    printf("Unable to open mat file %s\n", matfile);
    exit(-1);
  }
  /*----------- READ MM banner */
  if (mm_read_banner(p, &matcode) != 0){
    printf("Could not process Matrix Market banner.\n");
    return 1;
  }
  if (!mm_is_valid(matcode)){
    printf("Invalid Matrix Market file.\n");
    return 1;
  }
  if ( !( (mm_is_real(matcode) || mm_is_integer(matcode)) && mm_is_coordinate(matcode)
        && mm_is_sparse(matcode) ) ) {
    printf("Only sparse real-valued/integer coordinate \
        matrices are supported\n");
    return 1;
  }
  HYPRE_Int nrow, ncol, nnz, nnz2, k, j;
  char line[MAX_LINE];
  /*------------- Read size */
  if (mm_read_mtx_crd_size(p, &nrow, &ncol, &nnz) !=0) {
    printf("MM read size error !\n");
    return 1;
  }
  if (nrow != ncol) {
    fprintf(stdout,"This is not a square matrix!\n");
    return 1;
  }
  /*--------------------------------------
   * symmetric case : only L part stored,
   * so nnz2 := 2*nnz - nnz of diag,
   * so nnz2 <= 2*nnz
   *-------------------------------------*/
  if (mm_is_symmetric(matcode)){
    /* printf(" * * * *  matrix is symmetric * * * * \n"); */
    nnz2 = 2*nnz;
  } else {
    nnz2 = nnz;
  }
  /*-------- Allocate mem for COO */
  HYPRE_Int* IR = (HYPRE_Int *) malloc(nnz2 * sizeof(HYPRE_Int));
  HYPRE_Int* JC = (HYPRE_Int *) malloc(nnz2 * sizeof(HYPRE_Int));
  HYPRE_Real* VAL = (HYPRE_Real *) malloc(nnz2 * sizeof(HYPRE_Real));
  /*-------- read line by line */
  char *p1, *p2;
  for (k=0; k<nnz; k++) {
    if (fgets(line, MAX_LINE, p) == NULL) { return -1; }
    for( p1 = line; ' ' == *p1; p1++ );
    /*----------------- 1st entry - row index */
    for( p2 = p1; ' ' != *p2; p2++ );
    *p2 = '\0';
    float tmp1 = atof(p1);
    //coo.ir[k] = atoi(p1);
    IR[k] = (HYPRE_Int) tmp1;
    /*-------------- 2nd entry - column index */
    for( p1 = p2+1; ' ' == *p1; p1++ );
    for( p2 = p1; ' ' != *p2; p2++ );
    *p2 = '\0';
    float tmp2 = atof(p1);
    JC[k] = (HYPRE_Int) tmp2;
    //coo.jc[k]  = atoi(p1);
    /*------------- 3rd entry - nonzero entry */
    p1 = p2+1;
    VAL[k] = atof(p1);
  }
  /*------------------ Symmetric case */
  j = nnz;
  if (mm_is_symmetric(matcode)) {
    for (k=0; k<nnz; k++)
      if (IR[k] != JC[k]) {
        /*------------------ off-diag entry */
        IR[j] = JC[k];
        JC[j] = IR[k];
        VAL[j] = VAL[k];
        j++;
      }
    if (j != nnz2) {
      nnz2 = j;
    }
  }
  HYPRE_Int offset = idxout - idxin;
  if (offset) {
    for (i=0; i<nnz2; i++) {
      IR[i] += offset;
      JC[i] += offset;
    }
  }
  //  printf("nrow = %d, ncol = %d, nnz = %d\n", nrow, ncol, j);
  fclose(p);
  Acoo->nrows = nrow;
  Acoo->ncols = ncol;
  Acoo->nnz = nnz2;
  Acoo->ir = IR;
  Acoo->jc = JC;
  Acoo->val = VAL;
  return 0;
}

HYPRE_Int computeidx(HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, HYPRE_Int ix, HYPRE_Int iy, HYPRE_Int iz,
               HYPRE_Int dx, HYPRE_Int dy, HYPRE_Int dz) {
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
HYPRE_Int lapgen(HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, struct coo_t *Acoo, HYPRE_Int npts) {
  HYPRE_Int n = nx * ny * nz;
  Acoo->nrows = n;
  Acoo->ncols = n;

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

  HYPRE_Int nzmax = npts*n;
  Acoo->ir = (HYPRE_Int*) malloc(nzmax*sizeof(HYPRE_Int));
  Acoo->jc = (HYPRE_Int*) malloc(nzmax*sizeof(HYPRE_Int));
  Acoo->val = (HYPRE_Real *) malloc(nzmax*sizeof(HYPRE_Real));

  HYPRE_Int ii, nnz=0;
  for (ii=0; ii<n; ii++) {
    HYPRE_Real v = -1.0;
    HYPRE_Int iz = ii / (nx*ny);
    HYPRE_Int iy = (ii - iz*nx*ny) / nx;
    HYPRE_Int ix = ii - iz*nx*ny - iy*nx;

    HYPRE_Int jj;

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

  printf("Lapcian Matrix N = %d, NNZ = %d, NNZ/N = %d\n", n, nnz, nnz/n );

  // change to 1-based index
  /*
  for (ii=0; ii<nnz; ii++) {
     Acoo->ir[ii] ++;
     Acoo->jc[ii] ++;
  }
  */

  return 0;
}

// parse command-line input parameters
HYPRE_Int findarg(const char *argname, ARG_TYPE type, void *val, HYPRE_Int argc, char **argv) {
  HYPRE_Int *outint;
  hypre_double *outdouble;
  char *outchar;
  HYPRE_Int i;
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
            outint = (HYPRE_Int *) val;
            *outint = atoi(argv[i+1]);
            return 1;
          case DOUBLE:
            outdouble = (hypre_double *) val;
            *outdouble = atof(argv[i+1]);
            return 1;
          case STR:
            outchar = (char *) val;
            sprintf(outchar, "%s", argv[i+1]);
            return 1;
          default:
            printf("unknown arg type\n");
        }
      }
    }
  }
  return 0;
}

