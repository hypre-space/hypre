COMP      = f77 
LIBS      =  
CFLAGS    = -g
LDFLAGS   = -g

AMGFILES   = amgs01.f input.f setup.f opdfn.f form.f trunc.f crushb.f tests.f \
	memoryx.f crsgd.f setdep.f strong.f cforce.f ctest.f setcw.f \
	color.f testsB.f setc.f setws.f iinvert.f bsetup.f rstdf0.f \
	fftests.f misc.f

MSFILES   = ms.f


AMGOB  = amg.o
LAPNOBJS  = lapn.o 

MSOBJS    = ms.o

AMGS01OBJS   = amgs01.o input.o setup.o opdfn.o form.o trunc.o crushb.o tests.o \
	memoryx.o crsgd.o setdep.o strong.o cforce.o ctest.o setcw.o  \
	color.o testsB.o setc.o setws.o iinvert.o bsetup.o rstdf0.o \
	fftests.o misc.o




# Rules...

# all: amg lapn

amg: $(AMGOB) $(AMGS01OBJS) $(MSOBJS)
	$(COMP) $(LDFLAGS) -o amg $(AMGOB) $(AMGS01OBJS) $(MSOBJS) $(LIBS)

lapn:  $(LAPNOBJS) $(AMGS01OBJS) $(MSOBJS)
	$(COMP) $(LDFLAGS) -o lapn $(LAPNOBJS) $(AMGS01OBJS) $(MSOBJS) $(LIBS)


amgs01.o: params.amg

.f.o: 
	$(COMP) -c $(CFLAGS) $?




#
   



