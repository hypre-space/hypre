#include "io_dh.h"
#include "Mat_dh.h"
#include "Mem_dh.h"


#undef __FUNC__
#define __FUNC__ "readEuclid"
void readEuclid(char *filename, Mat_dh *A)
{
  START_FUNC_DH
  FILE *fp;
  Mat_dh AA;
  int n, nz;

  Mat_dhCreate(&AA); CHECK_V_ERROR;

  if ((fp = fopen(filename, "r")) == NULL) {
    printf("can't open '%s' for reading\n", filename);
    exit(-1);
  }

  if (fread(&n, sizeof(int), 1, fp)  != 1) {
    SET_V_ERROR("fread error for 'n'");
  }
  if (fread(&nz, sizeof(int), 1, fp)  != 1) {
    SET_V_ERROR("fread error for 'nz'");
  }

  AA->rp = (int*)MALLOC_DH((n+1)*sizeof(int)); CHECK_V_ERROR;
  AA->cval = (int*)MALLOC_DH(nz*sizeof(int));  CHECK_V_ERROR;
  AA->aval = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;

  if (fread(AA->rp, sizeof(int), n+1, fp)  != n+1) {
    SET_V_ERROR("fread error for 'rp'");
  }

  if (fread(AA->cval, sizeof(int), nz, fp)  != nz) {
    SET_V_ERROR("fread error for 'cval'");
  }

  if (fread(AA->aval, sizeof(double), nz, fp)  != nz) {
    SET_V_ERROR("fread error for 'aval'");
  }

  fclose(fp);

  AA->m = AA->n = n;
  *A = AA;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "writeEuclid"
void writeEuclid(int n, int *rp, int *cval, double *aval, char *filename)
{
  START_FUNC_DH
  FILE *fp;
  int nz = rp[n];

  if ((fp = fopen(filename, "w")) == NULL) {
    sprintf(msgBuf_dh, "can't open '%s' for writing", filename);
    SET_V_ERROR(msgBuf_dh);
  }

  if (fwrite(&n, sizeof(int), 1, fp)  != 1) {
    SET_V_ERROR("writeMybin::fwrite error for 'n'");
  }
  if (fwrite(&nz, sizeof(int), 1, fp)  != 1) {
    SET_V_ERROR("writeMybin::fwrite error for 'nz'");
  }
  if (fwrite(rp, sizeof(int), n+1, fp)  != n+1) {
    SET_V_ERROR("writeMybin::fwrite error for 'rp'");
  }
  if (fwrite(cval, sizeof(int), nz, fp)  != nz) {
    SET_V_ERROR("writeMybin::fwrite error for 'cval'");
  }
  if (fwrite(aval, sizeof(double), nz, fp)  != nz) {
    SET_V_ERROR("writeMybin::fwrite error for 'aval'");
  }
  fclose(fp);
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "readSCR_seq"
void readSCR_seq(char *filename, Mat_dh *A)
{
  START_FUNC_DH
  FILE *fp;
  int  i, n, nz;
  int *rp, *cval;
  double *aval;
  Mat_dh AA;

  Mat_dhCreate(&AA);

  if ((fp=fopen(filename, "r")) == NULL) {
    sprintf(msgBuf_dh, "can't open %s for reading", filename);
    SET_V_ERROR(msgBuf_dh);
  }

  /* get matrix dimensions */
  fscanf(fp, "%i %i", &n, &nz);
  sprintf(msgBuf_dh, "filename= %s  n= %i  nz= %i", filename, n, nz);
  SET_INFO(msgBuf_dh);

  /* allocate storage */
  rp = (int*)MALLOC_DH((n+1)*sizeof(int)); CHECK_V_ERROR;
  cval = (int*)MALLOC_DH(nz*sizeof(int)); CHECK_V_ERROR;
  aval = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;

  /* read rp, cval, and aval blocks */
  for (i=0; i<(n+1); ++i) fscanf(fp, "%i", &(rp[i]));
  for (i=0; i<nz; ++i)    fscanf(fp, "%i", &(cval[i]));
  for (i=0; i<nz; ++i)    fscanf(fp, "%lf", &(aval[i]));

  AA->m = AA->n = n;
  AA->rp = rp;
  AA->cval = cval;
  AA->aval = aval;
  *A = AA;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "printTriplesToFile"
void printTriplesToFile(int n, int *rp, int *cval, double *aval,
                                   int *n2o_row, int *n2o_col, char *filename)
{
  START_FUNC_DH
  int i, j;
  FILE *fp;

  if ((fp=fopen(filename, "w")) == NULL) {
    sprintf(msgBuf_dh, "can't open %s for writing\n", filename);
    SET_V_ERROR(msgBuf_dh);
  }

  if (n2o_row == NULL) {
    for (i=0; i<n; ++i) {
      for (j=rp[i]; j<rp[i+1]; ++j) {
        fprintf(fp, "%i %i %g\n", i+1, 1+cval[j], aval[j]);
      }
    }
  }

  else {
    int *o2n_col = (int*)MALLOC_DH(n*sizeof(int)); CHECK_V_ERROR;
    for (i=0; i<n; ++i) o2n_col[n2o_col[i]] = i;

    for (i=0; i<n; ++i) {
      int row = n2o_row[i];
      for (j=rp[row]; j<rp[row+1]; ++j) {
        fprintf(fp, "%i %i %g\n", i+1, 1+o2n_col[cval[j]], aval[j]);
      }
    }
    FREE_DH(o2n_col); CHECK_V_ERROR;
  }

  fclose(fp);
  END_FUNC_DH
}
