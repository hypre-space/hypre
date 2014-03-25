OBJS=csrcsrt.o csrssr.o dlinal.o dsparskit.o dsparskit2.o \
 ict.o iters.o pcg.o


ALL_OBJS = $(OBJS)

HEADERS = 

INC = .


lib: $(HEADERS) $(WRAPPER_OBJS) $(OBJS)
	ar -r libdsparskit.a $(WRAPPER_OBJS) $(OBJS)


#.f.o : ;	f77  -O -c $*.f  
.F.o : ;        /home/casc/g77/bin/g77 -g -c $*.F
.f.o : ;	/home/casc/g77/bin/g77 -g -c $*.f  
.c.o : ;        gcc -g -c -I$(INC) $*.c
