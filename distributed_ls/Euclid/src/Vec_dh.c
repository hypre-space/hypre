#include "Vec_dh.h"
#include "Mem_dh.h"

#undef __FUNC__
#define __FUNC__ "Vec_dhCreate"
void Vec_dhCreate(Vec_dh *v)
{
  START_FUNC_DH
  struct _vec_dh* tmp = (struct _vec_dh*)MALLOC_DH(sizeof(struct _vec_dh)); CHECK_V_ERROR;
  *v = tmp;
  tmp->n = 0;
  tmp->vals = NULL;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhDestroy"
void Vec_dhDestroy(Vec_dh v)
{
  START_FUNC_DH
  if (v->vals != NULL) FREE_DH(v->vals); CHECK_V_ERROR;
  FREE_DH(v); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhPrint"
void Vec_dhPrint(Vec_dh v, FILE *fp)
{
  START_FUNC_DH
  int i;
  fprintf(fp, "--------------------------------------------- Vec_dhPrint:\n");
  if (v->vals == NULL) {
    fprintf(fp, "v->vals = NULL; nothing to print!\n");
  } else {
    for (i=0; i<v->n; ++i) fprintf(fp, "%g  ", v->vals[i]);
  }
  fprintf(fp, "\n\n");
  fflush(fp);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Vec_dhInit"
void Vec_dhInit(Vec_dh v, int size)
{
  START_FUNC_DH
  v->n = size;
  v->vals = (double*)MALLOC_DH(size*sizeof(double)); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhCopy"
void Vec_dhCopy(Vec_dh x, Vec_dh y)
{
  START_FUNC_DH
  if (x->vals == NULL) SET_V_ERROR("x->vals is NULL");
  if (y->vals == NULL) SET_V_ERROR("y->vals is NULL");
  if (x->n != y->n) SET_V_ERROR("x and y are different lengths");
  memcpy(y->vals, x->vals, x->n*sizeof(double));
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Vec_dhDuplicate"
void Vec_dhDuplicate(Vec_dh v, Vec_dh *out)
{
  START_FUNC_DH
  Vec_dh tmp; 
  int size = v->n;
  if (v->vals == NULL) SET_V_ERROR("v->vals is NULL");
  Vec_dhCreate(out); CHECK_V_ERROR;
  tmp = *out;
  tmp->n = size;
  tmp->vals = (double*)MALLOC_DH(size*sizeof(double)); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhSet"
void Vec_dhSet(Vec_dh v, double value)
{
  START_FUNC_DH
  int i, m = v->n;
  double *vals = v->vals;
  if (v->vals == NULL) SET_V_ERROR("v->vals is NULL");
  for (i=0; i<m; ++i) vals[i] = value;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Vec_dhSetRand"
void Vec_dhSetRand(Vec_dh v)
{
  START_FUNC_DH
  int i, m = v->n;
  double max = 0.0;
  double *vals = v->vals;

  if (v->vals == NULL) SET_V_ERROR("v->vals is NULL");

  for (i=0; i<m; ++i) vals[i] = random();

  /* find largest value in vector, and scale vector,
   * so all values are in [0.0,1.0]
   */
  for (i=0; i<m; ++i) max = MAX(max, vals[i]);
  for (i=0; i<m; ++i) vals[i] = vals[i]/max; 
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Vec_dhPrintToFile"
void Vec_dhPrintToFile(Vec_dh v, char *filename)
{

  START_FUNC_DH
  int i, size = v->n;
  double *vals = v->vals;
  char buf[200];
  FILE *fp;

  if (v->vals == NULL) SET_V_ERROR("v->vals is NULL");
  sprintf(buf, "%s.vec", filename);
  if ((fp = fopen(buf, "w")) == NULL) {
    sprintf(msgBuf_dh, "can't open %s for writing", buf);
    SET_V_ERROR(buf);
  } else {
    for (i=0; i<size; ++i) fprintf(fp, "%g\n", vals[i]);
    fclose(fp);
  }
  END_FUNC_DH
}
