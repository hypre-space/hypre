# Device level-set ILU0 regression tests

ILU type 60 supports CUDA/HIP device execution, zero fill, direct triangular
solves, and noniterative setup. CPU and SYCL execution are unsupported.
See the ILU user manual for the structural and pivot requirements.

Build with `HYPRE_BUILD_TESTS=ON`, then run from the CMake build directory:

```sh
cmake --build . --target test_ilu60
ctest --output-on-failure -R '^test_ilu60($|_)'
```

`test_ilu60` compares factors and corrections against an independent host
ILU0 oracle, exercises identity/RCM ordering, asymmetric and unsorted patterns,
invalid input and pivots, and repeated setup/destruction. `test_ilu60_stream`
checks stream/RHS changes and uncaptured scalar substitutions. `test_ilu60_mpi`
uses three ranks to cover an empty rank, collective failure agreement, and
successful recovery after a rank-local failed setup. Stream tests require a
CUDA/HIP build; the MPI test additionally requires an MPI build and launcher.
Tests run serially to avoid competing GPU work. Exit status 77 means a skipped
test, including unavailable CUDA/HIP hardware; it is not correctness evidence.

The equivalent direct invocations, from the build's test directory, are:

```sh
./test_ilu60
./test_ilu60 --stream
mpirun -np 3 ./test_ilu60 --mpi
```

For the Makefile workflow, `make test_ilu60` (or `make tests`) builds the driver;
run the direct commands above to execute it. Exit status 77 indicates a
skip in this workflow as well. Select hardware supported by the build's
GPU architecture; a visible GPU alone does not guarantee compatibility.

Allocation accounting requires a separate build with
`HYPRE_ENABLE_MEMORY_TRACKER=ON`. The default test then checks live allocation
counts across repeated setups and destruction; without the tracker it reports
that accounting remains pending. Keep tracker overhead out of timing builds.
Release and Debug builds use the same scalar, per-level execution path.
Passing these tests does not replace device memory/race sanitizer validation.

# Implementation and performance scope

Type 60 complements type 0 for large sparse local blocks with short rows and
wide dependency levels. Each GPU thread factors or solves one row. The existing
type 0 remains available for other sparsity patterns; type 60 does not choose
between implementations automatically.

The implementation keeps the zero-fill pattern and serial pivot order. It uses
sorted off-diagonals and stores factors in initial identity/RCM coordinates.
Lower and upper schedules are computed independently on the host, supporting
general asymmetric patterns. Each dependency level uses a separate scalar
kernel launch. Setup and destruction synchronize before retiring owned storage.

Compare type 60 with type 0 using matching Release builds, solver settings,
precision, architecture, ordering, MPI, and allocators. Measure complete
converged solves, setup time, first-use costs, iteration counts, and independent
residuals. Reverse execution order and report all retained repetitions rather
than the fastest sample. Include large sparse matrices and denser controls.
Keep profiling and allocation-accounting runs separate from timing runs.
Performance results from earlier implementations do not validate later edits;
repeat numerical tests and paired measurements after changing kernels.
