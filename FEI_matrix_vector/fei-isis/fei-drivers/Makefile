################################################################
#
#	Central makefile for the ISIS++ FEI sparse solver drivers.
#
################################################################

SHELL		= /bin/sh

include ../make.options

CINCLUDES	= -I.  -I$(ISIS_ROOT)
CDEFINES	= -D$(ISIS_ARCH) $(ISIS_BOOL) -D$(ISIS_TARGET)

C++FLAGS	= $(LD_OPT)

C++DEFINES	= $(ISIS_BOOL) -D$(ISIS_TARGET)
C++INCLUDES_MPI = $(C++INCLUDES) -I$(MPI_INCLUDE)

LD		= $(CPP)

CFE_OBJS	= CFE_Elem.o CFE_Node.o
FE_OBJS		= FE_Elem.o FE_Node.o
Poisson_OBJS    = Poisson_Elem.o PoissonData.o

PLL_FEI_PGMS = distBeamDriver.x poisson-isis.x distExtBeamDriver.x \
				distExtPenDriver.x
PLL_CFEI_PGMS = distCFEBeamDriver.x

PLIB = $(FEI_ROOT)/lib/$(ISIS_ARCH)/libfei_isis.a

#	External targets.
default: $(PLL_FEI_PGMS) $(PLL_CFEI_PGMS)

all:
	make $(PLL_PGMS)
	make $(PLL_FEI_PGMS)
	make $(PLL_CFEI_PGMS)
	echo "===> Make drivers complete"

vardef:
	echo $(PLIB)
	echo $(OBJS)
	echo $(C++FLAGS)
	echo $(C++INCLUDES_MPI)
	echo $(C++DEFINES)
	echo $(LDFLAGS)
	echo $(SuperLU_INCLUDE_PATH)
	echo $(libraries_mpi)
	echo $(C)
	echo $(CINCLUDES) 
	echo $(CDEFINES) 
	echo $(CFLAGS)
	
test:
	./runBEAM

distBeamDriver.x: distBeamDriver.cc $(FE_OBJS) $(PLIB)
	$(CPP) $(LD_OPT) $(C++INCLUDES_MPI) -I. $(C++DEFINES) $(LDFLAGS) \
		distBeamDriver.cc -o distBeamDriver.x $(FE_OBJS) $(libraries_mpi)

poisson-isis.x: poisson-isis.cpp $(Poisson_OBJS) $(PLIB)
	$(CPP) $(LD_OPT) $(C++INCLUDES_MPI) \
		-I. $(C++DEFINES) $(LDFLAGS) \
		poisson-isis.cpp -o poisson-isis.x $(Poisson_OBJS) \
		$(libraries_mpi)

distExtBeamDriver.x: distExtBeamDriver.cc $(FE_OBJS) $(PLIB)
	$(CPP) $(LD_OPT) $(C++INCLUDES_MPI) -I. $(C++DEFINES) $(LDFLAGS) \
		distExtBeamDriver.cc -o distExtBeamDriver.x $(FE_OBJS) $(libraries_mpi)

distExtPenDriver.x: distExtPenDriver.cc $(FE_OBJS) $(PLIB)
	$(CPP) $(LD_OPT) $(C++INCLUDES_MPI) -I. $(C++DEFINES) $(LDFLAGS) \
		distExtPenDriver.cc -o distExtPenDriver.x $(FE_OBJS) $(libraries_mpi)

distCFEBeamDriver.x: distCFEBeamDriver.c $(PLL_OBJS) $(CFE_OBJS) $(PLIB)
	$(C) $(CFLAGS) $(CDEFINES) $(C++INCLUDES_MPI) -I. \
		-c distCFEBeamDriver.c
	$(CPP) $(LDFLAGS) distCFEBeamDriver.o $(CFE_OBJS) \
		-o distCFEBeamDriver.x $(libraries_mpi)
  
PoissonData.o:	PoissonData.cpp
	$(CPP) $(C++FLAGS) -c $(C++INCLUDES) \
		$(C++DEFINES) PoissonData.cpp

Poisson_Elem.o:	Poisson_Elem.cc
	$(CPP) $(C++FLAGS) -c $(C++INCLUDES) \
		$(C++DEFINES) Poisson_Elem.cc

FE_Elem.o:	FE_Elem.cc
	$(CPP) $(C++FLAGS) $(C++DEFINES) \
		$(C++INCLUDES) \
		-c FE_Elem.cc

FE_Node.o:	FE_Node.cc
	$(CPP) $(C++FLAGS) $(C++DEFINES) \
		$(C++INCLUDES) \
		-c FE_Node.cc

CFE_Elem.o:	CFE_Elem.c
	$(C) $(CFLAGS) -c $(C++INCLUDES) $(CDEFINES) CFE_Elem.c

CFE_Node.o:	CFE_Node.c
	$(C) $(CFLAGS) -c $(C++INCLUDES) $(CDEFINES) CFE_Node.c

clean:
	@touch erase.ii
	rm -rf core erase.* PI* *~ *.x *.o *.out $(OBJS) ./ptrepository residuals *.ii *debug* A_*ISIS* b_*ISIS* x_*ISIS*
	@echo "===> Make fei-drivers clean complete"

