void cuda_init(HYPRE_Int argc, char **argv);
void cuda_check_err();
void print_header();
void spmv_csr_cpu(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y);
void spmv_csr_vector(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y);
void spmv_cusparse_csr(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y);
HYPRE_Real error_norm(HYPRE_Real *x, HYPRE_Real *y, HYPRE_Int n);
void FreeCOO(struct coo_t *coo);

HYPRE_Int findarg(const char *argname, ARG_TYPE type, void *val, HYPRE_Int argc, char **argv);
HYPRE_Int lapgen(HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, struct coo_t *Acoo, HYPRE_Int npts);

HYPRE_Int coo_to_csr(HYPRE_Int cooidx, struct coo_t *coo, hypre_CSRMatrix **csr);
HYPRE_Int read_coo_MM(const char *matfile, HYPRE_Int idxin, HYPRE_Int idxout, struct coo_t *Acoo);
void sortrow(hypre_CSRMatrix *A);

hypre_double wall_timer();
