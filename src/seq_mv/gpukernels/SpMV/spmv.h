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
#define BLOCKDIM 512

#define FORT(name) name ## _
//#define FORT(name) name

/* types of user command-line input */
typedef enum {
  INT,
  DOUBLE,
  STR,
  NA
} ARG_TYPE;

/* COO format type */
struct coo_t {
  HYPRE_Int nrows;
  HYPRE_Int ncols;
  HYPRE_Int nnz;
  HYPRE_Int *ir;
  HYPRE_Int *jc;
  HYPRE_Real *val;
};

#include "protos.h"

