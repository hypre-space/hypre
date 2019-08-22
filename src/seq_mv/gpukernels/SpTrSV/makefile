include ../../../config/Makefile.config

.c.o:
	$(CC) $(CFLAGS) -o $@ -c $<

CFLAGS+= -g -I../../../hypre/include -I../common ${INCLUDES}

OBJS = testlusol.o ../common/io.o ../common/mmio.o ../common/auxil.o ../common/format.o \
       level.o gsLevR.o gsLevC.o gsDynR.o gsGSF.o gsDynR_2.o gsDynC.o cusparse.o ../SpMV/spmv.o

default: lusol.ex
lusol.ex: $(OBJS)
	$(LINK_CXX) ${LDFLAGS} -o lusol.ex $(OBJS) $(LIBS) -lcublas -L../../../lib/ -lHYPRE

clean:
	find . | egrep "#" | xargs rm -f
	find . | egrep "\~" | xargs rm -f
	rm -f *.o *.o_s *.o_d ${OBJS}
	rm -f *.ex

