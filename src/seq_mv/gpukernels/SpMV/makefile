include ../../../config/Makefile.config

.c.o:
	$(CC) $(CFLAGS) -o $@ -c $<

CFLAGS+= -I../../../hypre/include -I../common ${INCLUDES}
OBJS = testspmv.o spmv.o ../common/io.o ../common/mmio.o ../common/format.o ../common/auxil.o 

default: spmv.ex
spmv.ex: $(OBJS)
	$(LINK_CXX) ${LDFLAGS} -o spmvD.ex $(OBJS) $(LIBS) -L../../../lib/ -lHYPRE
clean:
	find . | egrep "#" | xargs rm -f
	find . | egrep "\~" | xargs rm -f
	rm -f *.o
	rm -f *.ex

