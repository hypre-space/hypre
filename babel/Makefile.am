SUBDIRS = Hypre test
AUTOMAKE_OPTIONS = foreign no-dependencies
BABELDIR = @babeldir@
EXTRA_DIST = Interfaces.idl

all: all-redirect

all-redirect: all-local all-recursive

all-local:
	@-if find Hypre -name "*Skel.c" > /dev/null ; then \
	  echo "Compiling Interfaces.idl..."; \
	  ${BABELDIR}/bin/babel -c${CC} -lC -aSolaris Interfaces.idl ; \
	else  \
	  echo "Already compiled Interfaces.idl..."; \
	fi

