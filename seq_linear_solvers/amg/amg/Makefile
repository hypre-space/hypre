
.SUFFIXES:
.SUFFIXES: .c .f .${AMG_ARCH}.o

AMG_SRC =\
 amg.f\
 amgs01.f\
 bsetup.f\
 cforce.f\
 color.f\
 crsgd.f\
 crushb.f\
 ctest.f\
 fftests.f\
 form.f\
 iinvert.f\
 input.f\
 memoryx.f\
 misc.f\
 ms.f\
 opdfn.f\
 rstdf0.f\
 setc.f\
 setcw.f\
 setdep.f\
 setup.f\
 setws.f\
 strong.f\
 tests.f\
 testsB.f\
 trunc.f

AMG_OBJS1 = ${AMG_SRC:.c=.${AMG_ARCH}.o}
AMG_OBJS  = ${AMG_OBJS1:.f=.${AMG_ARCH}.o}

LAPNOBJS  = lapn.o 

##################################################################
# Rules
##################################################################

${AMG_PROGRAM}.${AMG_ARCH}: ${AMG_OBJS}
	@echo  "Linking" ${AMG_PROGRAM} "... "
	@${AMG_LINK} -o ${AMG_PROGRAM}.${AMG_ARCH} \
		${LFLAGS} ${AMG_OBJS} ${AMG_LIBS}

lapn: ${LAPNOBJS} ${AMG_OBJS}
	@echo  "Linking" lapn "... "
	@${AMG_LINK} -o lapn \
		${LFLAGS} ${LAPNOBJS} ${AMG_OBJS} ${AMG_LIBS}

.c.${AMG_ARCH}.o:
	@echo "Making (c) " $@
	@${AMG_CC} -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${AMG_F77} -o $@ -c ${FFLAGS} $<

amgs01.o: params.amg

