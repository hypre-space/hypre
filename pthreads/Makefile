M4 = m4
CC = cc -g
CLINKER = ${CC}
M4FLAGS = -DTHREADED=1
LDFLAGS = -lpthread


hello:  hello_world.o pthreads_interface.o
	${CLINKER} -o hello hello_world.o pthreads_interface.o ${LDFLAGS}

hello_world.o: hello_world.c
	${CC} -c hello_world.c

hello_world.c: hello_world.m4 pthreads_c_definitions.m4
	${M4} ${M4FLAGS} hello_world.m4 > hello_world.c

pthreads_interface.o: pthreads_interface.c
	${CC} -c pthreads_interface.c

clean:
	rm -rf hello hello_world.o hello_world.c pthreads_interface.o

