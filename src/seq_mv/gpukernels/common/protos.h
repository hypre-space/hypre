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
HYPRE_Int GaussSeidelRowLevSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
void luSolvLevR16(int n, int nnz, hypre_CSRMatrix *, HYPRE_Real *d_x, HYPRE_Real *d_b, int, bool);
void luSolvLevR32(int n, int nnz, hypre_CSRMatrix *, HYPRE_Real *d_x, HYPRE_Real *d_b, int, bool);
void luSolvLevC16(int n, int nnz, hypre_CSRMatrix *, HYPRE_Real *d_x, HYPRE_Real *d_b, int, bool);
void luSolvLevC32(int n, int nnz, hypre_CSRMatrix *, HYPRE_Real *d_x, HYPRE_Real *d_b, int, bool);
double wall_timer();
void GaussSeidel_cusparse1(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int, bool);
void GaussSeidel_cusparse2(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int, bool);
void CreateSyncfree(hypre_CSRMatrix *csr, struct syncfree_t *syncf);
void FreeSyncfree(struct syncfree_t *syncf);
void luSolvDYNR(int n, int nnz, hypre_CSRMatrix *csr,
               HYPRE_Real *x, HYPRE_Real *b, int, bool);
void allocLevel(int n, struct level_t *lev);
void checktopo(int n, int *ib, int *jb, int *db, int *d_jlevL,
               int *d_jlevU, int *d_dp);

