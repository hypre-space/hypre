void cuda_init(HYPRE_Int argc, char **argv);
void cuda_check_err();
void print_header();
void spmv_csr_cpu(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y, HYPRE_Int REPEAT);
void spmv_csr_vector(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y, HYPRE_Int REPEAT);
void spmv_cusparse_csr(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y, HYPRE_Int REPEAT);
HYPRE_Real error_norm(HYPRE_Real *x, HYPRE_Real *y, HYPRE_Int n);
HYPRE_Int findarg(const char *argname, ARG_TYPE type, void *val, HYPRE_Int argc, char **argv);
HYPRE_Int lapgen(HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz, struct coo_t *Acoo, HYPRE_Int npts);
HYPRE_Int coo_to_csr(HYPRE_Int cooidx, struct coo_t *coo, hypre_CSRMatrix **csr);
HYPRE_Int read_coo_MM(const char *matfile, HYPRE_Int idxin, HYPRE_Int idxout, struct coo_t *Acoo);
void sortrow(hypre_CSRMatrix *A);
void cudaMallocCSR(int n, int nnz, hypre_CSRMatrix *d_csr);
void FreeCOO(struct coo_t *coo);
void FreeCSR(hypre_CSRMatrix *csr);
void FreeLev(struct level_t *h_lev);
void CudaFreeCSR(hypre_CSRMatrix *d_csr);
void CudaFreeLev(struct level_t *h_lev);
void makeLevel(hypre_CSRMatrix *, struct level_t *);
void makeLevelCSR(int n, int *ia, int *ja, struct level_t *h_lev);
void GaussSeidelCPU(int n, int nnz, HYPRE_Real *b, HYPRE_Real *x, hypre_CSRMatrix *csr, int, bool);
template <bool TEST>
HYPRE_Int GaussSeidelRowLevSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
template <bool TEST>
HYPRE_Int GaussSeidelColLevSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
template <bool TEST>
HYPRE_Int GaussSeidelRowDynSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
template <bool TEST>
HYPRE_Int GaussSeidelRowDynSchd_v2(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
template <bool TEST>
HYPRE_Int GaussSeidelColDynSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
HYPRE_Int GaussSeidelColGSF(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
double wall_timer();
void GaussSeidel_cusparse1(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int, bool);
void GaussSeidel_cusparse2(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int, bool);
void allocLevel(int n, struct level_t *lev);
HYPRE_Int hypre_SeqCSRMatvecDevice(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Complex *d_x, HYPRE_Complex *d_y);
