
#Options
  PETSC_OPT = g
  #PETSC_OPT = O

  #MATREAD = HB
  #MATREAD = Amg
  MATREAD = UNSYM


  KSP = Ksp
  PC = BlockJacobi
  #LOCAL_SOLVER = Amg
  LOCAL_SOLVER = ILU
  #LOCAL_SOLVER = IC

  COMM = mpi
  #COMM = pvm



  PETSC_INC = -I$(PETSC_DIR)/src/mat/impls/aij/mpi -I$(PETSC_DIR)/src/mat/impls/aij/seq \
        -I$(PETSC_DIR)/src/mat -I$(PETSC_DIR)
  PETSC_PRIV_INC = -I$(PETSC_DIR)/include
  MPI_INC = -I/usr/local/mpi/mpich/include
  Amg_INC = -I/home/cleary/amg/install.SunOS/include \
            -I/usr/local/include -I/home/casc/include
  ILU_INC = -I../../seq_linear_solvers/ilut
  IC_INC = -I../../seq_linear_solvers/ict
  LIN_SOLVE_INC = -I../.. -I../../utilities

  INC = $(PETSC_INC) $(PETSC_PRIV_INC) $(MPI_INC) $($(LOCAL_SOLVER)_INC) \
        $(LIN_SOLVE_INC) -I/home/casc/include

  CFLAGS = $(INC) -g -DReadMPIAIJ_matread=ReadMPIAIJ_$(MATREAD) \
           -DLocalSolver$(LOCAL_SOLVER)
  FFLAGS = 


#compiling/linking specifications
#CC       = cc -DPARCH_solaris
CC       = gcc
FC       = f77
LINK     = gcc -Wl,"-zmuldefs"
AR       = ar


#Implicit compilation rules
.c.o:
	$(CC)  -c $(CFLAGS) $*.c
.f.o: 
	$(FC)  -c $(FFLAGS) $*.f


#libraries to link to

LDIR	      = $(PETSC_DIR)/lib/lib$(PETSC_OPT)/$(PETSC_ARCH)
include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base.site
PETSC    =	-L$(LDIR) -lpetscsles  \
		-lpetscmat  -lpetscvec  \
		-lpetscsys \
		$(SUPERLU_LIB) $(BS_LIB) $(LAPACK_LIB) $(BLAS_LIB) \
                $(X11_LIB) \
		$(MPE_LIB) $(MPI_LIB) $(SYS_LIB)


Amg   = -L$(AMG_DIR)/lib -lamg -L/usr/local/lib 

ILU   = -L../../seq_linear_solvers/ilut -lilu

IC   = -L../../seq_linear_solvers/ict -lic

SEQ_MATRIX = -L../../seq_matrix_vector -lHYPRE_seq_mat_vec

SPARSKIT = -L../../seq_linear_solvers/sparskit -ldsparskit

DEBUG_LIBS = -L/home/casc/lib -lcegdb



LIBS = $($(LOCAL_SOLVER)) $(PETSC) $(SPARSKIT) $(SEQ_MATRIX) $(DEBUG_LIBS) \
       -L/home/casc/g77/lib -lf2c -lm

#Some source preprocessing using SED
#Initialize
BlockJacobiILUPcKspInitialize.c: BlockJacobiPcKspInitialize.c
	sed -e s/incfact/ilu/g < BlockJacobiPcKspInitialize.c \
        | sed -e s/INCFACT/ILU/g \
        > BlockJacobiILUPcKspInitialize.c

BlockJacobiICPcKspInitialize.c: BlockJacobiPcKspInitialize.c
	sed -e s/incfact/ic/g < BlockJacobiPcKspInitialize.c \
        | sed -e s/INCFACT/IC/g \
        > BlockJacobiICPcKspInitialize.c

#Options
BlockJacobiILUPcKspOptional.c: BlockJacobiPcKspOptional.c
	sed -e s/incfact/ilu/g < BlockJacobiPcKspOptional.c \
        | sed -e s/INCFACT/ILU/g \
        > BlockJacobiILUPcKspOptional.c

BlockJacobiICPcKspOptional.c: BlockJacobiPcKspOptional.c
	sed -e s/incfact/ic/g < BlockJacobiPcKspOptional.c \
        | sed -e s/INCFACT/IC/g \
        > BlockJacobiICPcKspOptional.c

#Setup
BlockJacobiILUPcKspSetup.c: BlockJacobiPcKspSetup.c
	sed -e s/incfact/ilu/g < BlockJacobiPcKspSetup.c \
        | sed -e s/INCFACT/ILU/g \
        > BlockJacobiILUPcKspSetup.c

BlockJacobiICPcKspSetup.c: BlockJacobiPcKspSetup.c
	sed -e s/incfact/ic/g < BlockJacobiPcKspSetup.c \
        | sed -e s/INCFACT/IC/g \
        > BlockJacobiICPcKspSetup.c

#Solver
BlockJacobiILUPcKspSolve.c: BlockJacobiPcKspSolve.c
	sed -e s/ILU/ILU/g < BlockJacobiPcKspSolve.c \
        | sed -e s/INCFACT/ILU/g \
        > BlockJacobiILUPcKspSolve.c

BlockJacobiICPcKspSolve.c: BlockJacobiPcKspSolve.c
	sed -e s/ILU/IC/g < BlockJacobiPcKspSolve.c \
        | sed -e s/INCFACT/IC/g \
        > BlockJacobiICPcKspSolve.c

#"Apply" routines
ILU_Apply.c: Apply.c
	sed -e s/incfact/ilu/g < Apply.c \
          | sed -e s/INCFACT/ILU/g \
          | sed -e s/ksp/gmres/g \
          | sed -e s/KSP/GMRES/g > ILU_Apply.c


IC_Apply.c: Apply.c
	sed -e s/incfact/ic/g < Apply.c \
          | sed -e s/INCFACT/IC/g \
          | sed -e s/ksp/cg/g \
          | sed -e s/KSP/CG/g \
        > IC_Apply.c


#Driver routines
driver.Amg.c: driver.c
	sed -e s/Amg/Amg/g < driver.c > driver.Amg.c

driver.ILU.c: driver.c
	sed -e s/Amg/ILU/g < driver.c > driver.ILU.c

driver.IC.c: driver.c
	sed -e s/Amg/IC/g < driver.c > driver.IC.c

driver.$(LOCAL_SOLVER).o: driver.$(LOCAL_SOLVER).c

DRIVER = driver.$(LOCAL_SOLVER).o

READER = ReadMPIAIJ_$(MATREAD).o ReadMPIVec.o



#Tools
AmgTOOLS = CsrConvert.o
ILUTOOLS = 
ICTOOLS = 
TOOLS = $($(LOCAL_SOLVER)TOOLS)

#Debugging tools
DEBUG = -lmalloc
#DEBUG = 

#Solver files
SOLVER_NAME = $(PC)$(LOCAL_SOLVER)Pc$(KSP)

SOLVER_HEADERS = $(PC)Pc$(KSP).h
SOLVER = $(SOLVER_NAME)Initialize.o $(SOLVER_NAME)Optional.o \
         $(SOLVER_NAME)Setup.o $(SOLVER_NAME)Solve.o


#"Apply" routine
APPLY = $(LOCAL_SOLVER)_Apply.o

#Headers
HEADERS = $(PETSC_HEADERS) $(AMG_HEADERS) $(SOLVER_HEADERS)

lib: $(SOLVER) $(APPLY) $(TOOLS)
	$(AR) -r libHYPRE$(LOCAL_SOLVER).a $(SOLVER) $(APPLY) $(TOOLS)

exe: $(HEADERS) $(DRIVER) $(READER) lib
	$(LINK) -o driver.$(LOCAL_SOLVER) $(DRIVER) $(READER) \
                 libHYPRE$(LOCAL_SOLVER).a $(DEBUG) $(LIBS)
