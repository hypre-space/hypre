OBJS=gmres_driver.o ilut_driver.o ilu_setup.o ilu_Initialize.o \
       ilu_solve.o GetILUDataIpar.o SetILUMode.o

HEADERS = ilu_data.h ilut_facsol_f.h

INC = -I.. -I/home/casc/include -I../../includes

SPARSKIT_DIR = ../dsparskit
LIB = -L$(SPARSKIT_DIR) -ldsparskit

#Source preprocessing from parent directory
ilu_data.h: ../if_data.h
	sed -e s/incfact/ilu/g < ../if_data.h \
          | sed -e s/INCFACT/ILU/g \
          | sed -e s/ksp/gmres/g \
          | sed -e s/KSP/GMRES/g \
          > ilu_data.h

ilut_facsol_f.h: ../ift_facsol_f.h
	sed -e s/incfact/ilu/g < ../ift_facsol_f.h \
          | sed -e s/INCFACT/ILU/g \
          | sed -e s/ksp/gmres/g \
          | sed -e s/KSP/GMRES/g \
          > ilut_facsol_f.h

GetILUDataIpar.c: $(HEADERS) ../GetIFDataIpar.c
	sed -e s/incfact/ilu/g < ../GetIFDataIpar.c \
          | sed -e s/INCFACT/ILU/g \
          | sed -e s/ksp/gmres/g \
          | sed -e s/KSP/GMRES/g \
          > GetILUDataIpar.c

SetILUMode.c: $(HEADERS) ../SetIFMode.c
	sed -e s/incfact/ilu/g < ../SetIFMode.c \
          | sed -e s/INCFACT/ILU/g \
          | sed -e s/ksp/gmres/g \
          | sed -e s/KSP/GMRES/g \
          > SetILUMode.c

ilu_Initialize.c: $(HEADERS) ../if_Initialize.c
	sed -e s/incfact/ilu/g < ../if_Initialize.c \
          | sed -e s/INCFACT/ILU/g \
          | sed -e s/ksp/gmres/g \
          | sed -e s/KSP/GMRES/g \
          > ilu_Initialize.c

ilu_setup.c: $(HEADERS) ../if_setup.c
	sed -e s/incfact/ilu/g < ../if_setup.c \
          | sed -e s/INCFACT/ILU/g \
          | sed -e s/ksp/gmres/g \
          | sed -e s/KSP/GMRES/g \
          > ilu_setup.c

ilu_solve.c: $(HEADERS) ../if_solve.c
	sed -e s/incfact/ilu/g < ../if_solve.c \
          | sed -e s/INCFACT/ILU/g \
          | sed -e s/ksp/gmres/g \
          | sed -e s/KSP/GMRES/g \
          > ilu_solve.c



ilut_main  : $(OBJS)
	/home/casc/g77/bin/g77 -g -o ilut_main $(OBJS) $(LIB)

lib: $(HEADERS) $(OBJS)
	ar -r libilu.a $(OBJS)


#.f.o : ;	f77  -O -c $*.f  
.f.o : ;	/home/casc/g77/bin/g77 -g -c $*.f  
.c.o : ;        gcc -g -c $(INC) $*.c
