#***********************************************************************
#***********************************************************************
#
# Makefile for support routine to link to ARPACK
#
#***********************************************************************
#***********************************************************************

FORTRAN = f77
FFLAGS  = -O3
CC      = cc
CFLAGS  = -O3

#files =	dmvm.o \
#Dinterface.o \
#Zinterface.o \
#dnexst.o \
#dnexge.o \
#dngeev.o \
#dngeres.o \
#dninge.o \
#dninst.o \
#dnstev.o \
#dnstres.o \
#dnzcnt.o \
#dshftab.o \
#dshftab2.o \
#dtrsv.o \
#ztrsv.o

files =	dmvm.o \
Dinterface.o \
dnexst.o \
dnexge.o \
dngeev.o \
dngeres.o \
dninge.o \
dninst.o \
dnstev.o \
dnstres.o \
dnzcnt.o \
dshftab.o \
dshftab2.o \
dtrsv.o \

.f.o:
	${FORTRAN} ${FFLAGS} -c $?

.c.o:
	$(CC) ${CFLAGS} -c $?

lib: $(files)
	ar rv libarpacksub.a $(files)
	ranlib libarpacksub.a

clean:
	\rm -f *.o *.a
