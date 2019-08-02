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

#define SPMV_BLOCKDIM 512
#define SPTRSV_BLOCKDIM 1024

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

struct level_t {
  // L
  HYPRE_Int nlevL;
  HYPRE_Int num_klevL;
  HYPRE_Int *jlevL;
  HYPRE_Int *ilevL;
  HYPRE_Int *klevL;
  // U
  HYPRE_Int nlevU;
  HYPRE_Int num_klevU;
  HYPRE_Int *jlevU;
  HYPRE_Int *ilevU;
  HYPRE_Int *klevU;
  // level
  HYPRE_Int *levL;
  HYPRE_Int *levU;
};

#include "protos.h"

