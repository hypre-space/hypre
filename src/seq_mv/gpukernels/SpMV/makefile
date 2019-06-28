include ../../../config/Makefile.config

CFLAGS+= -I../../../hypre/include ${INCLUDES}
OBJS2 = spmv.o io.o testspmv.o mmio.o format.o auxil.o #skit.o

default: spmv.ex
spmv.ex: $(OBJS2)
	$(LINK_CXX) ${LDFLAGS} -o spmvD.ex $(OBJS2) $(LIBS) -L../../../lib/ -lHYPRE
clean:
	find . | egrep "#" | xargs rm -f
	find . | egrep "\~" | xargs rm -f
	rm -f *.o
	rm -f *.ex

