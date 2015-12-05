#
# compiler and options
#
FC = f77
FFLAGS  = -O2
CC      = cc
CFLAGS  = -O2

# 
# libraries
#
SUPLDLT = ../SUPLDLT/libsupldlt.a 
METIS   = ../../metis-4.0/libmetis.a
ARPACK  = ../ARPACK/libarpack_alpha.a
LIBS   = $(SUPLDLT) $(ARPACK)

.f.o:
	${FC} ${FFLAGS} -c $?
.c.o:
	$(CC) -DTIMING ${CFLAGS} -c $?

dssdrv: dssdrv.o dspssev.o
	${FC} -o dssdrv  dssdrv.o dspssev.o ${LIBS} ${ARPACK} -lm 

dsgdrv: dsgdrv.o dspsgev.o dnzcnt.o dshftab.o ${LIBS} 
	${FC} -o dsgdrv dsgdrv.o dspsgev.o dnzcnt.o dshftab.o\
        ${LIBS} ${ARPACK} -lm

nd3kev: nd3kev.o dspssev.o ${files} 
	${FC} -o nd3kev nd3kev.o dspssev.o ${files} \
	${METIS} ${ARPACK} -lm

clean:
	rm -f *.o dssdrv dsgdrv
