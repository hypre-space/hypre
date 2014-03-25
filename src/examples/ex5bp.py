"""Example of using hypre through the Babel-Python interface"""

# To build, do the following
# 1. Install Python 2.4 or later.
# 2. Install a version of Numeric Python as required by Babel.
# 3. Install pyMPI.  Probably this could be made to work with other MPI extensions of Python
#    with minor changes in this example; but I haven't tried anything else.
# 4. configure --enable-shared --with-babel --enable-python=pyMPI
#    If you have trouble with missing stdc++ functions, also use  --without-fei
# 5. make
# 6. Make sure you have the following environment variables:
#    SIDL_DLL_PATH={your top-level hypre directory}/babel/bHYPREClient-P/libbHYPRE.scl
#    LD_LIBRARY_PATH={your top-level hypre directory}/hypre/lib

import mpi,sys
import bHYPRE
import bHYPRE.MPICommunicator
# It is important to import bHYPRE.MPICommunicator first.  The other classes
# require it, so the build system assumes you will have loaded it first.
import bHYPRE.IJParCSRMatrix
import bHYPRE.IJParCSRVector
import bHYPRE.BoomerAMG
import bHYPRE.PCG
import bHYPRE.IdentitySolver
from array import *
from struct import *
from Numeric import *

def solver( solver_id=0 ):
    # solver_id values : 0(default, AMG), 1 (PCG-AMG), 8 (ParaSails), 50 (PCG)
    #
    myid = mpi.COMM_WORLD.rank
    num_procs = mpi.COMM_WORLD.size
    #>>>> doesn't work, but should ...mpi_comm = bHYPRE.MPICommunicator.CreateC( mpi.COMM_WORLD )
    #>>>> temporary substitute>>>
    mpi_comm = bHYPRE.MPICommunicator.Create_MPICommWorld()
    #
    #   for now, don't bother to read any inputs
    #
    n = 33
    print_solution = 0
    #
    # Preliminaries: want at least one processor per row
    if n*n < num_procs :
        n = int( num_procs**0.5 ) + 1
    N = n*n            # global number of rows
    h = 1.0/(n+1)      # mesh size
    h2 = h*h
    #
    # Each processor knows only of its own rows - the range is denoted by ilower
    # and upper.  Here we partition the rows. We account for the fact that
    # N may not divide evenly by the number of processors.
    local_size = N/num_procs
    extra = N - local_size*num_procs
    #
    ilower = local_size*myid
    ilower += min(myid, extra)
    #
    iupper = local_size*(myid+1)
    iupper += min(myid+1, extra)
    iupper = iupper - 1
    #
    # How many rows do I have? 
    local_size = iupper - ilower + 1
    #
    # Create the matrix.
    # Note that this is a square matrix, so we indicate the row partition
    # size twice (since number of rows = number of cols)
    parcsr_A = bHYPRE.IJParCSRMatrix.Create( mpi_comm, ilower, iupper, ilower, iupper )
    #
    # Choose a parallel csr format storage (see the User's Manual)
    # Note: Here the HYPRE interface requires a SetObjectType call.
    # I am using the bHYPRE interface in a way which does not because
    # the object type is already specified through the class name. 
    #
    # Initialize before setting coefficients 
    parcsr_A.Initialize()
    #
    # Now go through my local rows and set the matrix entries.
    # Each row has at most 5 entries. For example, if n=3:
    #
    #   A = [M -I 0; -I M -I; 0 -I M]
    #   M = [4 -1 0; -1 4 -1; 0 -1 4]
    #
    # Note that here we are setting one row at a time, though
    #  one could set all the rows together (see the User's Manual).
    values = zeros(5)
    cols = zeros(5)
    i = ilower
    while i<=iupper:
        nnz = 0
        # The left identity block: position i-n
        if (i-n) >= 0 :
            cols[nnz] = i-n
            values[nnz] = -1.0
            nnz = nnz + 1
        # The left -1: position i-1
        if (i%n):
            cols[nnz] = i-1
            values[nnz] = -1.0
            nnz = nnz + 1
        # Set the diagonal: position i
        cols[nnz] = i
        values[nnz] = 4.0
        nnz = nnz + 1
        # The right -1: position i+1
        if ((i+1)%n):
            cols[nnz] = i+1
            values[nnz] = -1.0
            nnz = nnz + 1
        # The right identity block:position i+n
        if (i+n) < N:
            cols[nnz] = i+n
            values[nnz] = -1.0
            nnz = nnz + 1
        # Set the values for row i
        parcsr_A.SetValues( array([nnz]), array([i]), cols, values )
        i = i + 1
    #
    # Assemble after setting the coefficients
    parcsr_A.Assemble()
    #
    # Create the rhs and solution
    par_b = bHYPRE.IJParCSRVector.Create( mpi_comm, ilower, iupper )
    par_b.Initialize()
    par_x = bHYPRE.IJParCSRVector.Create( mpi_comm, ilower, iupper )
    par_x.Initialize()
    #
    # Set the rhs values to h^2 and the solution to zero
    rhs_values = zeros(local_size)*1.1
    x_values = zeros(local_size)*1.1
    rows = zeros(local_size)
    i = 0
    while i<local_size:
        rhs_values[i] = h2
        rows[i] = ilower + i
        i = i + 1
    par_b.SetValues( rows, rhs_values )
    par_x.SetValues( rows, x_values )
    #
    par_b.Assemble()
    par_x.Assemble()
    #
    # Choose a solver and solve the system
    #
    # AMG
    if solver_id == 0:
        # Create solver
        solver = bHYPRE.BoomerAMG.Create( mpi_comm, parcsr_A )
        # Set some parameters (See Reference Manual for more parameters)
        solver.SetIntParameter( "PrintLevel", 3 )  # print solve info + parameters
        solver.SetIntParameter( "CoarsenType", 6 ) # Falgout coarsening
        solver.SetIntParameter( "RelaxType", 3 )   # G-S/Jacobi hybrid relaxation
        solver.SetIntParameter( "NumSweeps", 1 )   # Sweeeps on each level
        solver.SetIntParameter( "MaxLevels", 20 )  # maximum number of levels
        solver.SetDoubleParameter( "Tolerance", 1e-7 )      # conv. tolerance
        #
        # Now setup and solve!
        solver.Setup( par_b, par_x )
        solver.Apply( par_b, par_x )
        #
        # Run info - needed logging turned on
        # The 0-th return value of Get*Value is the error flag.
        num_iterations = solver.GetIntValue( "NumIterations" )[1]
        final_res_norm = solver.GetDoubleValue( "RelResidualNorm" )[1]
        #
        if myid == 0:
            print "Iterations = ", num_iterations
            print "Final Relative Residual Norm = ", final_res_norm
    elif solver_id == 50:
        # Create solver
        solver = bHYPRE.PCG.Create( mpi_comm, parcsr_A )
        # Set some parameters (See Reference Manual for more parameters)
        solver.SetIntParameter( "MaxIter", 1000 ) # max iterations
        solver.SetDoubleParameter( "Tolerance", 1e-7 ) # conv. tolerance
        solver.SetIntParameter( "TwoNorm", 1 ) # use the two norm as the stopping criteria
        solver.SetIntParameter( "PrintLevel", 2 ) # prints out the iteration info
        solver.SetIntParameter( "Logging", 1 ) # needed to get run info later
        #
        precond = bHYPRE.IdentitySolver.Create( mpi_comm )
        solver.SetPreconditioner( precond )
        #
        # Now setup and solve!
        solver.Setup( par_b, par_x )
        solver.Apply( par_b, par_x )
        #
        # Run info - needed logging turned on
        # The 0-th return value of Get*Value is the error flag.
        num_iterations = solver.GetIntValue( "NumIterations" )[1]
        final_res_norm = solver.GetDoubleValue( "RelResidualNorm" )[1]
        #
        if myid == 0:
            print "Iterations = ", num_iterations
            print "Final Relative Residual Norm =", final_res_norm
    else:
        print "Solver ", solver_id, " is not supported."
#
if __name__ == "__main__":
    solver(0)
