/* supporting routines used by both mpi and seq versions */

/*

?????????????
void print_triples_to_file_private(int n, int m, int beg_row, int *rp, 
*/

#include "Euclid_dh.h"
#include "Mem_dh.h"
#include "Mat_dh.h"
#include "Hash_dh.h"  /* needed for print_triples_to_file_private */

static char *algo_par_strings[] = { "None", "Block Jacobi", 
                                     "Parallel ILU", "Graph Color ILU" };
static char *algo_ilu_strings[] = { "None", "ILU(k)", "ILUT(dt)" };

#undef __FUNC__
#define __FUNC__ "Euclid_dhCreate"
void Euclid_dhCreate(Euclid_dh *ctxOUT)
{
  START_FUNC_DH
  struct _mpi_interface_dh * ctx =
     (struct _mpi_interface_dh*)MALLOC_DH(sizeof(struct _mpi_interface_dh)); CHECK_V_ERROR;

  *ctxOUT = ctx;

  ctx->n = 0;
  ctx->m = 0;
  ctx->beg_row = 0;
  ctx->first_bdry = 0;
  ctx->nzA    = 0;
  ctx->nzF    = 0;
  ctx->nzAglobal = 0;
  ctx->nzFglobal = 0;
  ctx->rho_init = 2.0;
  ctx->rho_final = 0.0;

  ctx->ownsAstruct = false;
  ctx->A = NULL;

  ctx->isSinglePrecision = true;
  ctx->rpF  = NULL;
  ctx->cvalF = NULL;
  ctx->avalF = NULL;
  ctx->avalFD = NULL;
  ctx->diagF = NULL;
  ctx->fillF = NULL;
  ctx->allocF = 0;

  ctx->partMethod = SIMPLE_PART;
  ctx->blockCount = 1;
  ctx->block[0].beg_row = 0;
  ctx->block[0].end_row = 0;
  ctx->block[0].first_bdry = 0;
  ctx->colorCounter = 0;

  ctx->n2o_row = NULL;
  ctx->n2o_col = NULL;
  ctx->n2o_nonLocal = NULL;
  ctx->o2n_nonLocal = NULL;
  ctx->isNaturallyOrdered = true;
  ctx->isSymOrdered = true;
  ctx->orderMethod = NATURAL_ORDER;

  ctx->beg_rows = NULL;
  ctx->end_rows = NULL;
  ctx->bdryNodeCounts = 0;
  ctx->bdryRowNzCounts = 0;
  ctx->naborCount = 0;
  ctx->nabors = 0;
  ctx->externalRows = 0;

  ctx->scale = NULL;
  ctx->scaleD = NULL;
  ctx->isScaled = false;
  ctx->work = NULL;
  ctx->workD = NULL;
  ctx->from = 0;
  ctx->to = 0;

  ctx->algo_par = BJILU_PAR;
  ctx->algo_ilu = ILUK_ILU;
  ctx->level = 1;
  ctx->cLevel = -1;
  ctx->droptol = 0.0;
  ctx->sparseTolA = 0.0;
  ctx->sparseTolF = 0.0;
  ctx->pivotMin = 0.0;
  ctx->pivotFix = PIVOT_FIX_DEFAULT;
  ctx->maxVal = 0.0;

  ctx->krylovMethod = EUCLID_NONE;
  ctx->maxIts = 200;
  ctx->rtol = 1e-5;
  ctx->atol = 1e-50;
  ctx->itsOUT = 0;
  ctx->A = NULL;

  ctx->zeroDiags = 0;
  ctx->zeroPivots = 0;
  ctx->symbolicZeroDiags = 0;
  ctx->printProfile = false;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Euclid_dhDestroy"
void Euclid_dhDestroy(Euclid_dh ctx)
{
  START_FUNC_DH
  if (ctx->rpF != NULL) { FREE_DH(ctx->rpF); CHECK_V_ERROR; }
  if (ctx->cvalF != NULL) { FREE_DH(ctx->cvalF); CHECK_V_ERROR; }
  if (ctx->avalF != NULL) { FREE_DH(ctx->avalF); CHECK_V_ERROR; }
  if (ctx->avalFD != NULL) { FREE_DH(ctx->avalFD); CHECK_V_ERROR; }
  if (ctx->fillF != NULL) { FREE_DH(ctx->fillF); CHECK_V_ERROR; }
  if (ctx->diagF != NULL) { FREE_DH(ctx->diagF); CHECK_V_ERROR; }
  if (ctx->n2o_row!= NULL) { FREE_DH(ctx->n2o_row); CHECK_V_ERROR; }
  if (ctx->beg_rows != NULL) { FREE_DH(ctx->beg_rows); CHECK_V_ERROR; }
  if (ctx->end_rows != NULL) { FREE_DH(ctx->end_rows); CHECK_V_ERROR; }
  if (ctx->bdryNodeCounts != NULL) { FREE_DH(ctx->bdryNodeCounts); CHECK_V_ERROR; }
  if (ctx->bdryRowNzCounts != NULL) { FREE_DH(ctx->bdryRowNzCounts); CHECK_V_ERROR; }
  if (ctx->nabors != NULL) { FREE_DH(ctx->nabors); CHECK_V_ERROR; }
  if (ctx->n2o_col!= NULL  && (! ctx->isSymOrdered)) {
    FREE_DH(ctx->n2o_col); CHECK_V_ERROR; 
  }
  if (ctx->colorCounter != NULL) { FREE_DH(ctx->colorCounter); CHECK_V_ERROR; }
  if (ctx->externalRows != NULL) { Hash_dhDestroy(ctx->externalRows); CHECK_V_ERROR; }
  if (ctx->scale != NULL) { FREE_DH(ctx->scale); CHECK_V_ERROR; }
  if (ctx->scaleD != NULL) { FREE_DH(ctx->scaleD); CHECK_V_ERROR; }
  if (ctx->work != NULL) { FREE_DH(ctx->work); CHECK_V_ERROR; }
  if (ctx->workD != NULL) { FREE_DH(ctx->workD); CHECK_V_ERROR; }
  if (ctx->n2o_nonLocal != NULL) { Hash_dhDestroy(ctx->n2o_nonLocal); CHECK_V_ERROR; }
  if (ctx->o2n_nonLocal != NULL) { Hash_dhDestroy(ctx->o2n_nonLocal); CHECK_V_ERROR; }

  /* this may not be a really good idea . . . */
  if (ctx->ownsAstruct) { 
   Mat_dh A = (Mat_dh) ctx->A;
   A->rp = NULL;
   A->cval = NULL;
   A->aval = NULL;
   Mat_dhDestroy(A); CHECK_V_ERROR;
  }

  FREE_DH(ctx); CHECK_V_ERROR; 
  END_FUNC_DH
}


/* on entry, the following fields have been
   initialized by the user (probably by calling Euclid_dhInputXxxMat):
        m, n, beg_row, A, ownsAstruct.
*/
#undef __FUNC__
#define __FUNC__ "Euclid_dhSetup"
void Euclid_dhSetup(Euclid_dh ctx)
{
  START_FUNC_DH
  int m = ctx->m;
  bool printMat = Parser_dhHasSwitch(parser_dh, "-printMat");

  /* for debugging: optionally print input matrix as triples */
  if (printMat) {
    PrintMatUsingGetRow(ctx->A, ctx->beg_row, ctx->m, NULL, NULL, "A.trip"); CHECK_V_ERROR;
  }

  /* query parser for runtime parameters */
  get_runtime_params_private(ctx); CHECK_V_ERROR;

  /* record initial size of matrix (ctx->nzA, ctx->nzAglobal) */
  find_nzA_private(ctx); CHECK_V_ERROR;

  /* set initial storage allocation for factor; heuristic: pad by 5%, 
     in case user rounded off */
  ctx->allocF = ctx->rho_init*ctx->nzA*1.05;

  /* initialize natural ordering */
  if (ctx->n2o_row == NULL) {
    int i, *tmp = ctx->n2o_row = (int*)MALLOC_DH(m*sizeof(int));  CHECK_V_ERROR;
    for (i=0; i<m; ++i) tmp[i] = i;
    ctx->n2o_col = ctx->n2o_row;
    ctx->isSymOrdered = true;
    ctx->isNaturallyOrdered = true;
  }

  if (np_dh == 1) {
    euclid_setup_private_seq(ctx); CHECK_V_ERROR;
  } else {
    euclid_setup_private_mpi(ctx); CHECK_V_ERROR;
  }

  /* allocate and initialize storage for row-scaling values */
  if (ctx->algo_ilu != NONE_ILU) {
    int i; 
    if (ctx->isSinglePrecision) {
      float *tmp = ctx->scale = (float*)MALLOC_DH(m*sizeof(float)); CHECK_V_ERROR;
      for (i=0; i<m; ++i) tmp[i] = 1.0;
    } else {
      double *tmp = ctx->scaleD = (double*)MALLOC_DH(m*sizeof(double)); CHECK_V_ERROR;
      for (i=0; i<m; ++i) tmp[i] = 1.0;
    }
  }

  /* order subdomain interiors */
  if (ctx->orderMethod != NATURAL_ORDER) {
    order_interiors_private(ctx);
  }

  /* perform symbolic and numeric factorization */
  if (! Parser_dhHasSwitch(parser_dh, "-doNotFactor")) {

    factor_private(ctx); CHECK_V_ERROR;

    /* record size of factor (ctx->nzF, ctx->nzFglobal, ctx->rhow_final) */
    find_nzF_private(ctx); CHECK_V_ERROR;

    /* invert diagonals, for faster triangular solves */
    invert_diagonals_private(ctx); CHECK_V_ERROR;
  }

  /* for debugging: optionally print permuted blocked input matrix */
  if (printMat) {
    print_factor_private(ctx, "F.trip"); CHECK_V_ERROR;
    if (myid_dh == 1) {
      printf("@@@@@ factor printed to 'F.trip' as triples\n");
      printf("@@@@@ local matrices printed to logFile (if opened)\n");
    }
  }

  /* allocate work vector for triangular solves */
  if (ctx->algo_ilu != NONE_ILU) {
    if (ctx->isSinglePrecision) {
      ctx->work = (float*)MALLOC_DH(ctx->m*sizeof(float)); CHECK_V_ERROR;
    } else {
      ctx->workD = (double*)MALLOC_DH(ctx->m*sizeof(double)); CHECK_V_ERROR;
    }
  }

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "get_runtime_params_private"
void get_runtime_params_private(Euclid_dh ctx)
{
  START_FUNC_DH
  ctx->isSinglePrecision = (! Parser_dhHasSwitch(parser_dh, "-eu_double"));
  ctx->isScaled = (! Parser_dhHasSwitch(parser_dh, "-doNotScale"));
  Parser_dhReadInt(parser_dh,    "-level",&(ctx->level));    /* for ILUK */
  Parser_dhReadInt(parser_dh, "-gColor",&(ctx->cLevel));  /* for graph color ILU */
  Parser_dhReadDouble(parser_dh, "-dt",&(ctx->droptol));     /* for ILUT */
  Parser_dhReadDouble(parser_dh, "-sparseA",&(ctx->sparseTolA));  /* sparsify A before factoring */
  Parser_dhReadDouble(parser_dh, "-sparseF",&(ctx->sparseTolF));  /* sparsify after factoring */
  Parser_dhReadDouble(parser_dh, "-pivotMin", &(ctx->pivotMin));    /* adjust pivots if smaller than this */
  Parser_dhReadDouble(parser_dh, "-pivotFix", &(ctx->pivotFix));    /* how to adjust pivots */

  Parser_dhReadDouble(parser_dh, "-rho", &(ctx->rho_init)); /* inital storage allocation for factor */

  Parser_dhReadInt(parser_dh,    "-profile",&(ctx->printProfile)); /* prints info about matrix to stdout */

  Parser_dhReadInt(parser_dh,    "-maxIts",&(ctx->maxIts)); 
  Parser_dhReadDouble(parser_dh, "-rtol", &(ctx->rtol));
  Parser_dhReadDouble(parser_dh, "-atol", &(ctx->atol));

  /* set parallelization strategy */
  if (Parser_dhHasSwitch(parser_dh, "-blockJacobi")) {
    Parser_dhReadInt(parser_dh, "-blockJacobi", &(ctx->blockCount));
    ctx->algo_par = BJILU_PAR;
  } 

  if  (Parser_dhHasSwitch(parser_dh, "-pilu")) {
    Parser_dhReadInt(parser_dh, "-pilu", &(ctx->blockCount));
    ctx->algo_par = PILU_PAR;
  } 

  if  (Parser_dhHasSwitch(parser_dh, "-gColor")) {
    Parser_dhReadInt(parser_dh, "-gColor", &(ctx->blockCount));
    ctx->algo_par = GRAPHCOLOR_PAR;
  }

  /* set ILU method */
  ctx->algo_ilu = ILUK_ILU;
  if (Parser_dhHasSwitch(parser_dh, "-ilut")) {
    ctx->algo_ilu = ILUT_ILU;
  }
  if (Parser_dhHasSwitch(parser_dh, "-iluNONE")) {
    ctx->algo_ilu = NONE_ILU;
    ctx->algo_par = NONE_PAR;
  }

  /* partitioning method */
  ctx->partMethod = SIMPLE_PART;
  if (Parser_dhHasSwitch(parser_dh, "-useMetis")) {
    ctx->partMethod= METIS_PART;
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Euclid_dhPrintParams"
void Euclid_dhPrintParams(Euclid_dh ctx, FILE *fp)
{
  START_FUNC_DH
  if (myid_dh == 0) {
    fprintf(fp, "\n------------------------------ Euclid Settings Summary\n");
    fprintf(fp, "preconditioner density:  %0.2f (nonzeros in factor/nonzeros in A)\n", ctx->rho_final);

    if (ctx->isSinglePrecision) {
      fprintf(fp, "SINGLE precision\n");
    } else {
      fprintf(fp, "DOUBLE precision\n");
    }

    fprintf(fp, "global matrix dimension: %i\n", ctx->n);
    fprintf(fp, "number of MPI tasks:     %i\n", np_dh);
    fprintf(fp, "nonzeros in input:       %i\n", ctx->nzA);

    if (ctx->isScaled) {
      fprintf(fp, "row scaling in effect\n");
    } else {
      fprintf(fp, "row scaling NOT in effect\n");
    }

    fprintf(fp, "level:   %i (used for ILU(k))\n", ctx->level);
    fprintf(fp, "droptol: %0.3e (used for ILUT(dt))\n", ctx->droptol);
    fprintf(fp, "sparseA: %0.3e (used for sparsification of input)\n", ctx->sparseTolA);
    fprintf(fp, "parallelization strategy: %s\n", algo_par_strings[ctx->algo_par]);
    fprintf(fp, "factorization method:     %s\n", algo_ilu_strings[ctx->algo_ilu]);
  }
  END_FUNC_DH
}



#undef __FUNC__
#define __FUNC__ "reallocate_private"
void reallocate_private(int row, int newEntries, int *nzHave, 
                int **rp, int **cval, float **aval, double **avalD, int **fill)
{
  START_FUNC_DH
  int    *cvalTMP, *fillTMP;
  float *avalTMP;
  double *avalTMP_D;
  int *rpIN = *rp;
  int idx = rpIN[row];
  int nzNeed = idx+newEntries;
  int nz = *nzHave;

  while (nz < nzNeed) {
    nz *= 2;
  }

  sprintf(msgBuf_dh, "reallocating; old= %i; new= %i", *nzHave, nz);
  SET_INFO(msgBuf_dh);

  *nzHave = nz;

  if (cval != NULL) {
    cvalTMP = (int*)MALLOC_DH(nz*sizeof(int)); 
    memcpy(cvalTMP, *cval, idx*sizeof(int));
    FREE_DH(*cval);
    *cval = cvalTMP;
  }
  if (fill != NULL) {
    fillTMP = (int*)MALLOC_DH(nz*sizeof(int)); 
    memcpy(fillTMP, *fill, idx*sizeof(int));
    FREE_DH(*fill);
    *fill = fillTMP;
  }
  if (aval != NULL) {
    avalTMP = (float*)MALLOC_DH(nz*sizeof(float)); 
    memcpy(avalTMP, *aval, idx*sizeof(float));
    FREE_DH(*aval);
    *aval = avalTMP;
  }
  if (avalD != NULL) {
    avalTMP_D = (double*)MALLOC_DH(nz*sizeof(double)); 
    memcpy(avalTMP_D, *avalD, idx*sizeof(double));
    FREE_DH(*avalD);
    *avalD = avalTMP_D;
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "print_triples_to_file_private"
void print_triples_to_file_private(int n, int m, int beg_row, int *rp, 
                 int *cval, float *avalF, double *avalD, int *n2o_row, 
                 int *n2o_col, Hash_dh o2n_globalCol, char *filename)
{
  START_FUNC_DH
  int i, j, pe, end_row = beg_row+m; 
  FILE *fp;

  for (pe=0; pe<np_dh; ++pe) {

    MPI_Barrier(comm_dh);

    if (myid_dh == pe) {
      if (pe == 0) {
        fp=fopen(filename, "w");
      } else {
        fp=fopen(filename, "a");
      }
      if (fp == NULL) {
        sprintf(msgBuf_dh, "can't open %s for writing\n", filename);
        SET_V_ERROR(msgBuf_dh);
      }

      if (n2o_row == NULL) {
        for (i=0; i<m; ++i) {
          for (j=rp[i]; j<rp[i+1]; ++j) {
            if (avalD == NULL && avalF == NULL) {
              fprintf(fp, "%i %i 1\n", 1+beg_row+i, 1+cval[j]);
            } else if (avalF == NULL) {
              fprintf(fp, "%i %i %g\n", 1+beg_row+i, 1+cval[j], avalD[j]);
            } else {
              fprintf(fp, "%i %i %g\n", 1+beg_row+i, 1+cval[j], avalF[j]);
            }
          }
        } 
      }

      else {
        int *o2n_col = (int*)MALLOC_DH(n*sizeof(int)); CHECK_V_ERROR;
        for (i=0; i<m; ++i) o2n_col[n2o_col[i]] = i;
        for (i=0; i<m; ++i) {
          int localRow = n2o_row[i];
          for (j=rp[localRow]; j<rp[localRow+1]; ++j) {
            int col = cval[j];
            if (col < beg_row || col >= end_row) {
              if (o2n_globalCol == NULL) {
                col = -1;
              } else { 
                HashData *r = Hash_dhLookup(o2n_globalCol, col);
                if (r == NULL) {
                  sprintf(msgBuf_dh, "lookup for nonlocal column %i failed", 1+col);
                  SET_INFO(msgBuf_dh);
                } else {
int tmp = col;
                  col = r->iData;
sprintf(msgBuf_dh, "mapped nonLocal %i to %i;  beg= %i  end= %i\n", tmp, col, beg_row, beg_row+m);
                }
              }
            } else {
              col -= beg_row;
              col = o2n_col[col]+beg_row+1;
            }
            if (col > -1) { 
              if (avalD == NULL && avalF == NULL) {
                fprintf(fp, "%i %i 1\n", 1+beg_row+i, col);
              } else if (avalF == NULL) {
                fprintf(fp, "%i %i %g\n", 1+beg_row+i, col, avalD[j]);
              } else {
                fprintf(fp, "%i %i %g\n", 1+beg_row+i, col, avalF[j]);
              }
            } else {
               sprintf(msgBuf_dh, "can't get o2n for nonlocal col= %i", 1+cval[j]);
               SET_INFO(msgBuf_dh);
            }
          }
        }
        FREE_DH(o2n_col); CHECK_V_ERROR;
      }
      fclose(fp);
    }
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "invert_diagonals_private"
void invert_diagonals_private(Euclid_dh ctx)
{
  START_FUNC_DH
  if (ctx->algo_ilu != NONE_ILU) {
    int i, m = ctx->m, *diag = ctx->diagF;

    if (ctx->isSinglePrecision) {
      float *aval = ctx->avalF;
      for (i=0; i<m; ++i) {
        if (aval[diag[i]] != 0.0) aval[diag[i]] = 1.0/aval[diag[i]]; 
      }
    } else {
      double *aval = ctx->avalFD;
      for (i=0; i<m; ++i) {
        if (aval[diag[i]] != 0.0) aval[diag[i]] = 1.0/aval[diag[i]]; 
      }
    }
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "order_interiors_private"
void order_interiors_private(Euclid_dh ctx)
{
  START_FUNC_DH
  if (ctx->orderMethod != NATURAL_ORDER) {
    SET_V_ERROR("unknown ordering method");
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "factor_private"
void factor_private(Euclid_dh ctx)
{
  START_FUNC_DH
  int  m = ctx->m, nzmax = ctx->allocF;

  /* special case, mostly for debugging: no preconditioning */
  if (ctx->algo_ilu == NONE_ILU) {
    ctx->rpF   = (int*)MALLOC_DH(2*sizeof(int));  CHECK_V_ERROR;
    ctx->rpF[0] = ctx->rpF[1] = 0;
    goto DO_NOTHING;
  }

  if (ctx->diagF == NULL) {
    ctx->diagF = (int*)MALLOC_DH(m*sizeof(int));      CHECK_V_ERROR;
  }
  if (ctx->rpF == NULL) {
    ctx->rpF   = (int*)MALLOC_DH((1+m)*sizeof(int));  CHECK_V_ERROR;
    ctx->rpF[0] = 0;
  }
  if (ctx->algo_ilu == ILUK_ILU && ctx->fillF == NULL) {
    ctx->fillF = (int*)MALLOC_DH(nzmax*sizeof(int));  CHECK_V_ERROR;
  }
  if (ctx->cvalF == NULL) {
    ctx->cvalF = (int*)MALLOC_DH(nzmax*sizeof(int));  CHECK_V_ERROR;
  }

  if (ctx->isSinglePrecision && ctx->avalF == NULL) {
    ctx->avalF = (float*)MALLOC_DH(nzmax*sizeof(float));  CHECK_V_ERROR;
  } else if (ctx->avalFD == NULL) {
    ctx->avalFD = (double*)MALLOC_DH(nzmax*sizeof(double));  CHECK_V_ERROR;
  }

  /* case for block-jacobi with multiple mpi tasks;
   * also for block-jacobi and pilu for single mpi task.
   */ 
  if (ctx->algo_par == BJILU_PAR || np_dh == 1) {

    ctx->from = 0;
    ctx->to = m;
    if (ctx->isSinglePrecision) {
      iluk_seq(ctx); CHECK_V_ERROR;
    } else {
      iluk_seq_D(ctx); CHECK_V_ERROR;
    }
    ctx->nzF = ctx->rpF[m];
  }

  else if (ctx->algo_par == PILU_PAR) {

/*    SET_V_ERROR("broken"); */

    /* factor interior rows */
    ctx->from = 0;
    ctx->to = ctx->first_bdry;
    iluk_mpi(ctx); CHECK_V_ERROR;

    /* receive bdry rows */

    /* factor boundary rows */
    ctx->from = ctx->first_bdry;
    ctx->to   = ctx->m;
    iluk_mpi(ctx); CHECK_V_ERROR;

    /* send bdry rows */

/* exchange_bdry_rows_private(ctx); CHECK_V_ERROR; */
  }

  else {
    sprintf(msgBuf_dh, "unknown parallel method: %i\n", ctx->algo_par);
    SET_V_ERROR(msgBuf_dh);
  }

DO_NOTHING: ;

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Euclid_dhApply"
int Euclid_dhApply(void *ctx_in, void *xx, void *yy)
{
  START_FUNC_DH
  Euclid_dh  ctx = (Euclid_dh)ctx_in;

  #ifdef PETSC_MODE

    Vec x_vec = (Vec) xx;
    Vec y_vec = (Vec) xx;
    double *x, *y;
    int ierr;

    ierr = VecGetArray(x_vec, &x); CHECK_P_ERROR(ierr, "call to VecGetArray() failed");
    ierr = VecGetArray(y_vec, &y); CHECK_P_ERROR(ierr, "call to  VecGetArray() faild");

    Euclid_dhApply_private(ctx, x, y); CHECK_ERROR(errFlag_dh);

    ierr = VecRestoreArray(x_vec, &x); CHECK_P_ERROR(ierr, "call to VecRestoreArray() failed");
    ierr = VecRestoreArray(y_vec, &y); CHECK_P_ERROR(ierr, "call to VecGetArray() failed");

  #else

    double *x = (double*) xx;
    double *y = (double*) yy;

    Euclid_dhApply_private(ctx, x, y); CHECK_ERROR(errFlag_dh);

  #endif /* #ifdef PETSC_MODE ... #else */
  END_FUNC_VAL(0)
}

#undef __FUNC__
#define __FUNC__ "print_factor_private"
void print_factor_private(Euclid_dh ctx, char *filename)
{
  START_FUNC_DH
  int pe, m = ctx->m;

  /* block jacobi and single MPI-task case */
  if (ctx->algo_par == BJILU_PAR || np_dh == 1) {
    FILE *fp;
    int i, j, beg_row = ctx->beg_row;

      if (logFile != NULL) {
        for (i=0; i<m; ++i) {
          fprintf(logFile, "row= %2i :: ", i+1);
          for (j=ctx->rpF[i]; j<ctx->rpF[i+1]; ++j) {
            if (ctx->avalF != NULL) {
              fprintf(logFile, "%i,%g  ", 1+ctx->cvalF[j], ctx->avalF[j]);
            } else if (ctx->avalFD != NULL) {
              fprintf(logFile, "%i,%g  ", 1+ctx->cvalF[j], ctx->avalFD[j]);
            } else {
              SET_V_ERROR("both avalF and avalFD are NULL");
            }
          }
          fprintf(logFile, "\n");
        }
      }
  
      for (pe=0; pe<np_dh; ++pe) {
        MPI_Barrier(comm_dh);
        if (myid_dh == pe) {
          if (pe == 0) { fp=fopen(filename, "w"); } 
          else { fp=fopen(filename, "a"); }

          for (i=0; i<m; ++i) {
            for (j=ctx->rpF[i]; j<ctx->rpF[i+1]; ++j) {
              if (ctx->avalF != NULL) {
                fprintf(fp, "%i %i %g\n", 1+i+beg_row, 1+ctx->cvalF[j]+beg_row, ctx->avalF[j]);
              } else if (ctx->avalFD != NULL) {
                fprintf(fp, "%i %i %g\n", 1+i+beg_row, 1+ctx->cvalF[j]+beg_row, ctx->avalFD[j]);
              } else {
                SET_V_ERROR("both avalF and avalFD are NULL");
              }
            }
          }
          fclose(fp);
        } 
      } 
  }

  else {
    SET_V_ERROR("not implemented");
  }

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "find_nzA_private"
void find_nzA_private(Euclid_dh ctx)
{
  START_FUNC_DH
  int row, len, beg_row = ctx->beg_row;
  int end_row = beg_row + ctx->m;
  int nzLocal = 0, nzGlobal;

  for (row=beg_row; row<end_row; ++row) {
    EuclidGetRow(ctx->A, row, &len, NULL, NULL); CHECK_V_ERROR;
    nzLocal += len;
    EuclidRestoreRow(ctx->A, row, &len, NULL, NULL); CHECK_V_ERROR;
  }

  if (np_dh > 1) {
    MPI_Allreduce(&nzLocal, &nzGlobal, 1, MPI_INT, MPI_SUM, comm_dh);
  } else {
    nzGlobal = nzLocal;
  }
  ctx->nzA = nzGlobal;
  ctx->nzAglobal = nzGlobal;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "find_nzF_private"
void find_nzF_private(Euclid_dh ctx)
{
  START_FUNC_DH
  int nzLocal = 0, nzGlobal;
  double rhoLocal;


  if (ctx->algo_ilu == NONE_ILU) {
    ctx->nzF = ctx->nzFglobal = 0;
    ctx->rho_final = -1;
  } else {
    nzLocal = ctx->rpF[ctx->m];
    rhoLocal = (double)nzLocal/(double)ctx->nzA;
    ctx->nzF = nzLocal;

    if (np_dh == 1) {
      ctx->nzFglobal = nzLocal;
      ctx->rho_final = rhoLocal;
    } else {
      double rhoGlobal;
      MPI_Allreduce(&nzLocal, &nzGlobal, 1, MPI_INT, MPI_SUM, comm_dh);
      ctx->nzFglobal = nzGlobal;
      MPI_Allreduce(&rhoLocal, &rhoGlobal, 1, MPI_DOUBLE, MPI_MAX, comm_dh);
      ctx->rho_final = rhoGlobal;
    }
  }
  END_FUNC_DH
}
