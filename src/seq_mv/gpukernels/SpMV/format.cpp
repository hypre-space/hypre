#include "spmv.h"

/*--------------------------------------------------*/
void COO2CSR(struct coo_t *coo, struct csr_t *csr) {
  //Allocate CSR
  csr->n = coo->n;
  csr->nnz = coo->nnz;
  csr->ia = (int *) malloc((csr->n+1)*sizeof(int));
  csr->ja = (int *) malloc(csr->nnz*sizeof(int));
  csr->a = (REAL *) malloc(csr->nnz*sizeof(REAL));
  // COO -> CSR
  FORT(coocsr)(&coo->n, &coo->nnz, coo->val, coo->ir, coo->jc, 
          csr->a, csr->ja, csr->ia);
}

/*-------------------------------------------------*/
void CSR2JAD(struct csr_t *csr, struct jad_t *jad) {
  // Allocate JAD
  jad->n = csr->n;
  jad->nnz = csr->nnz;
  jad->ia = (int *) malloc((csr->n+1)*sizeof(int));
  jad->ja = (int *) malloc(csr->nnz*sizeof(int));
  jad->a = (REAL *) malloc(csr->nnz*sizeof(REAL));
  jad->perm = (int *) malloc(csr->n*sizeof(int));
  // CSR -> JAD
  FORT(csrjad)(&csr->n, csr->a, csr->ja, csr->ia,
          &jad->njad, jad->perm, jad->a, jad->ja, jad->ia);
/*------ pad jad each jad to have multiple of 32 */
  PadJAD32(jad);
}

/*---------------------------------------------------------*/
void PadJAD32(struct jad_t *jad) {
  int i;
  int nnz2 = 0;
  int *oldia = jad->ia;
  jad->ia = (int *) malloc((jad->njad+1)*sizeof(int));
  jad->ia[0] = 1;

  for (i=0; i<jad->njad; i++) {
    jad->ia[i+1] = jad->ia[i]+(oldia[i+1]-oldia[i]+31)/32*32;
    nnz2 += (jad->ia[i+1]-jad->ia[i]);
  }

  REAL *olda = jad->a;
  int *oldja = jad->ja;

  //printf("Pading with zeros %.2f\n",\
  (double)nnz2/(double)jad->nnz);

  jad->nnz = nnz2;
  jad->a = (REAL *) calloc(nnz2, sizeof(REAL));
  jad->ja = (int *) malloc(nnz2*sizeof(int));
  for (i=0; i<nnz2; i++)
    jad->ja[i] = 1;

  for (i=0; i<jad->njad; i++) {
    memcpy(&jad->a[jad->ia[i]-1], &olda[oldia[i]-1], 
    (oldia[i+1]-oldia[i])*sizeof(REAL)); 
    memcpy(&jad->ja[jad->ia[i]-1], &oldja[oldia[i]-1],
    (oldia[i+1]-oldia[i])*sizeof(int));    
  }

  free(olda);
  free(oldia);
  free(oldja);
}

