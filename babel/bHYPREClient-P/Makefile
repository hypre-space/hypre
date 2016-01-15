# Python has its own build system, so this makefile just runs python.

include ../../config/Makefile.config

PYTHON_HYPRE_LIBRARY_DIR = ${HOME}/lib/pyMPI2.4/site-packages/bHYPRE
HYPRE_BABEL_RUNTIME_SIDL_DIR = ${HYPRE_SRC_TOP_DIR}/babel-runtime/sidl
HYPRE_BABEL_RUNTIME_PYTHON_DIR = ${HYPRE_SRC_TOP_DIR}/babel-runtime/python

all: libbHYPRE.scl
	${PYTHON} ./setup.py build \
	--include-dirs=${HYPRE_INC_INSTALL} \
	--include-dirs=${HYPRE_BABEL_RUNTIME_SIDL_DIR} \
	--include-dirs=${HYPRE_BABEL_RUNTIME_PYTHON_DIR} \
	--library-dirs=${HYPRE_LIB_INSTALL} \
	--library-dirs=${HYPRE_BABEL_RUNTIME_SIDL_DIR}/.libs \
	--extra-library='bHYPRE' --extra-library='HYPRE'
	${PYTHON} ./setup.py install 
	cp ${HYPRE_LIB_INSTALL}/libbHYPRE${HYPRE_LIB_SUFFIX} ${PYTHON_HYPRE_LIBRARY_DIR}

libbHYPRE.scl: 
	sed -e 's,@@DIR@@,$(HYPRE_LIB_INSTALL),g' < $(srcdir)/libbHYPRE.scl.in > libbHYPRE.scl

install: all
 
clean:
	rm -rf build libbHYPRE.scl

distclean: clean
