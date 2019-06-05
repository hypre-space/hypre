void cuda_init(int argc, char **argv);
void cuda_check_err();
void print_header();
double wall_timer();
void spmv_csr_cpu(struct csr_t *csr, REAL *x, REAL *y);
void spmv_csr_vector(struct csr_t *csr, REAL *x, REAL *y);
void spmv_cusparse_csr(struct csr_t *csr, REAL *x, REAL *y);
double error_norm(REAL *x, REAL *y, int n);
void FreeCOO(struct coo_t *coo);
void FreeCSR(struct csr_t *csr);

int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv);
int lapgen(int nx, int ny, int nz, struct coo_t *Acoo, int npts);

int coo_to_csr(int cooidx, struct coo_t *coo, struct csr_t *csr);
int read_coo_MM(const char *matfile, int idxin, int idxout, struct coo_t *Acoo);
void sortrow(struct csr_t *A);

