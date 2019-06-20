#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <curand.h>
#include "../../seq_mv.h"

#define min(a, b) (a) > (b) ? (b) : (a)
#define max(a, b) (a) > (b) ? (a) : (b)
#define SEED 200
#define REPEAT 100
#define WARP 32
#define HALFWARP 16
#define BLOCKDIM 512
#define MAXTHREADS (30*1024*60)

#if DOUBLEPRECISION
#define REAL double
#else
#define REAL float
#endif

#define FORT(name) name ## _
//#define FORT(name) name

/* CSR format type */
struct csr_t {
  int nrows;
  int ncols;
  int *ia;
  int *ja;
  REAL *a;
};

/* types of user command-line input */
typedef enum {
  INT,
  DOUBLE,
  STR,
  NA
} ARG_TYPE;

/* COO format type */
struct coo_t {
  int nrows;
  int ncols;
  int nnz;
  int *ir;
  int *jc;
  REAL *val;
};

#include "protos.h"

