COMP      = f77 
LIBS      =  
CFLAGS    = -g
LDFLAGS   = -g

AMGFILES   = amgs01.f input.f setup.f opdfn.f form.f trunc.f crushb.f tests.f \
	memoryx.f crsgd.f setdep.f strong.f cforce.f ctest.f setcw.f \
	color.f testsB.f setc.f setws.f iinvert.f bsetup.f rstdf0.f \
	fftests.f misc.f

MSFILES   = ms.f


AMGDRVOBS  = amgdrv.o
LAPNOBJS  = lapn.o 
LAPNSHOBJS = lapnshow.o

MSOBJS    = ms.o

AMGOBJS   = amgs01.o input.o setup.o opdfn.o form.o trunc.o crushb.o tests.o \
	memoryx.o crsgd.o setdep.o strong.o cforce.o ctest.o setcw.o  \
	color.o testsB.o setc.o setws.o iinvert.o bsetup.o rstdf0.o \
	fftests.o misc.o




# Rules...


amgdrv: $(AMGDRVOBS) $(AMGOBJS) $(MSOBJS)
	$(COMP) $(LDFLAGS) -o $(HOME)/bin/amgdrv $(AMGDRVOBS) $(AMGOBJS) $(MSOBJS) $(LIBS)

lapn:  $(LAPNOBJS) $(AMGOBJS) $(MSOBJS)
	$(COMP) $(LDFLAGS) -o $(HOME)/bin/lapn $(LAPNOBJS) $(AMGOBJS) $(MSOBJS) $(LIBS)

lapnshow: $(LAPNSHOBJS) $(AMGOBJS) $(MSOBJS)
	$(COMP) $(LDFLAGS) -o $(HOME)/bin/lapnshow $(LAPNSHOBJS) $(AMGOBJS) $(MSOBJS) $(LIBS)

amgs01.o: params.amg

.f.o: 
	$(COMP) -c $(CFLAGS) $?




#
   



