OBJS=cg_driver.o ict_driver.o \
       ic_setup.o ic_Initialize.o ic_solve.o GetICDataIpar.o SetICMode.o

HEADERS = ic_data.h ict_facsol_f.h

INC = -I.. -I/home/casc/include -I../../includes

SPARSKIT_DIR = ../dsparskit
LIB = -L$(SPARSKIT_DIR) -ldsparskit

#Source preprocessing from parent directory
ic_data.h: ../if_data.h
	sed -e s/incfact/ic/g < ../if_data.h \
          | sed -e s/INCFACT/IC/g \
          | sed -e s/ksp/cg/g \
          | sed -e s/KSP/CG/g \
          > ic_data.h

ict_facsol_f.h: ../ift_facsol_f.h
	sed -e s/incfact/ic/g < ../ift_facsol_f.h \
          | sed -e s/INCFACT/IC/g \
          | sed -e s/ksp/cg/g \
          | sed -e s/KSP/CG/g \
          > ict_facsol_f.h

GetICDataIpar.c: $(HEADERS) ../GetIFDataIpar.c
	sed -e s/incfact/ic/g < ../GetIFDataIpar.c \
          | sed -e s/INCFACT/IC/g \
          | sed -e s/ksp/cg/g \
          | sed -e s/KSP/CG/g \
          > GetICDataIpar.c

SetICMode.c: $(HEADERS) ../SetIFMode.c
	sed -e s/incfact/ic/g < ../SetIFMode.c \
          | sed -e s/INCFACT/IC/g \
          | sed -e s/ksp/cg/g \
          | sed -e s/KSP/CG/g \
          > SetICMode.c

ic_Initialize.c: $(HEADERS) ../if_Initialize.c
	sed -e s/incfact/ic/g < ../if_Initialize.c \
          | sed -e s/INCFACT/IC/g \
          | sed -e s/ksp/cg/g \
          | sed -e s/KSP/CG/g \
          > ic_Initialize.c

ic_setup.c: $(HEADERS) ../if_setup.c
	sed -e s/incfact/ic/g < ../if_setup.c \
          | sed -e s/INCFACT/IC/g \
          | sed -e s/ksp/cg/g \
          | sed -e s/KSP/CG/g \
          > ic_setup.c

ic_solve.c: $(HEADERS) ../if_solve.c
	sed -e s/incfact/ic/g < ../if_solve.c \
          | sed -e s/INCFACT/IC/g \
          | sed -e s/ksp/cg/g \
          | sed -e s/KSP/CG/g \
          > ic_solve.c



ict_main  : $(OBJS)
	/home/casc/g77/bin/g77 -g -o ict_main $(OBJS) $(LIB)

lib: $(HEADERS) $(OBJS)
	ar -r libic.a $(OBJS)


#.f.o : ;	f77  -O -c $*.f  
.F.o : ;        /home/casc/g77/bin/g77 -g -c $*.F
.f.o : ;	/home/casc/g77/bin/g77 -g -c $*.f  
.c.o : ;        gcc -g -c $(INC) $*.c
