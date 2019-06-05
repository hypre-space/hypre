void cuda_init(int argc, char **argv);
void cuda_check_err();
void print_header();
int read_coo_MM(struct coo_t *coo, char *matfile, int);
void COO2CSR(struct coo_t *coo, struct csr_t *csr);
void CSR2JAD(struct csr_t *csr, struct jad_t *jad);
int CSR2DIA(struct csr_t *csr, struct dia_t *dia);
double wall_timer();
void spmv_csr_cpu(struct csr_t *csr, REAL *x, REAL *y);
void spmv_csr_scalar(struct csr_t *csr, REAL *x, REAL *y);
void spmv_csr_vector(struct csr_t *csr, REAL *x, REAL *y);
void spmv_jad(struct jad_t *jad, REAL *x, REAL *y);
void spmv_cusparse_csr(struct csr_t *csr, REAL *x, REAL *y);
void spmv_cusparse_hyb(struct csr_t *csr, REAL *x, REAL *y);
extern "C" {
void FORT(coocsr)(int*, int*, REAL*, int*, int*, REAL*, int*, int*);
void FORT(csrjad)(int*, REAL*, int*, int*, int*, int*, REAL*, int*, int*); 
}
double error_norm(REAL *x, REAL *y, int n);
void PadJAD32(struct jad_t *jad);
void FreeCOO(struct coo_t *coo);
void FreeCSR(struct csr_t *csr);
void FreeJAD(struct jad_t *jad);
void FreeDIA(struct dia_t *dia);

int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv);
int lapgen(int nx, int ny, int nz, struct coo_t *Acoo, int npts);

