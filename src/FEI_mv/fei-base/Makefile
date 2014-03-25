######################################################################
#
#	Top-level makefile for the FEI-isis code.
#
#######################################################################

include ../../config/Makefile.config

PACKAGE=fei-isis

#######################################################################
# configure generates make.options, and the user
# should never have to edit this main make file.
# If some old bsd 4.3 derived version of make does not
# support the include directive, then the text of
# make.options should be pasted here directly by
# the user.
#######################################################################

HYPRE_ROOT = ${HYPRE_SRC_TOP_DIR}
FEI_ROOT= ${HYPRE_ROOT}/FEI_mv/fei-base
HYPRE_ARCH=@ARCH@
FEI_TARGET=FEI_PLL

C= mpicc
CPP= mpiCC
C_COMPILE_FLAGS=-O2
CXX_COMPILE_FLAGS=-O2
F77_COMPILE_FLAGS=
CINCLUDES= 
CDEFS =

OPT=-w 
LD_OPT=-g
CFLAGS=${C_COMPILE_FLAGS}
INSTALL_ROOT=.
C++INCLUDES=-I$(FEI_ROOT) -I$(HYPRE_ROOT) \
            -I$(FEI_ROOT)/../fei-hypre \
            -I$(HYPRE_ROOT)/hypre/include \
            -I$(HYPRE_ROOT)/utilities \
            -I$(HYPRE_ROOT)/parcsr_mv \
            -I$(HYPRE_ROOT)/seq_mv \
            ${CINCLUDES}
CXXFLAGS=${CXX_COMPILE_FLAGS} $(OPT) -c

# std unixisms, we hope, that still need fixing sometimes
CP=/bin/cp
RM=/bin/rm
LN=/bin/ln
MKDIR=/bin/mkdir
CHMOD=chmod
RANLIB=ranlib

# our dependency stuff for linux
DEPEND_CPP=g++
DEPEND_ALL=$(FEI_ROOT)/util/g++depinclude
DEPEND_NOSYS=$(FEI_ROOT)/util/g++depinclude_nosys
DEPEND_NONE=$(FEI_ROOT)/util/g++nodepinclude
# DEPEND_FLAGS should be redefined in the applying makefile if needed
DEPEND_FLAGS=$(CXXFLAGS) $(C++INCLUDES) $(C++DEFINES)
# misc stuff that'll probably need fixing on some platforms:

#
# defines and other flags composed from info in make.options.
#

# other flags of convenience
IT_DEFINES	= -D$(FEI_TARGET)
C++DEFINES      = $(ISIS_BOOL) $(IT_DEFINES)
C++INCLUDES_MPI = $(C++INCLUDES)

# our primary targets defined
SLIB = $(FEI_ROOT)/lib/$(HYPRE_ARCH)/libHYPRE_FEI_BASE${HYPRE_LIB_SUFFIX}
PLIB = $(FEI_ROOT)/lib/$(HYPRE_ARCH)/libHYPRE_FEI_BASE${HYPRE_LIB_SUFFIX}

#######################################################################
#
# object lists:
#
#######################################################################

#############################
# the FEI targets
#############################
FEI_OBJS = \
	src/BCRecord.o \
	src/BCManager.o \
	src/NodeCommMgr.mp.o \
	src/ProcEqns.o \
	src/EqnBuffer.o \
	src/EqnCommMgr.mp.o \
	src/BlockDescriptor.o \
	src/FieldRecord.o \
	src/MultConstRecord.o \
	src/NodeDescriptor.o \
	src/ProblemStructure.mp.o \
	src/PenConstRecord.o \
	src/SLE_utils.o \
	src/Utils.o \
	src/BASE_FEI.mp.o \
	src/FEI_Implementation.mp.o 

#src/cfei.mp.o
#src/ISIS_LinSysCore.mp.o \

FEI_SRCS        = $(FEI_OBJS:.o=.cc)

#############################
# the 'other' target
#############################
OTHER_OBJ = \
	other/mpi.o

OTHER_SRC = $(OTHER_OBJ:.o=.c)

#############################
# object lists for wrappers of externally defined pc classes
# are small and infrequently changed, therefore we
# keep them with the build targets for the wrappers
# for ease of maintenance. 
#############################

# combined lists:
MPI_DEPENDENT_SRCS=

# install directories that will be created
# these must be given in order when there is a dependency
PACKAGE_DIR=$(INSTALL_ROOT)/fei-isis
FEI_SUBDIRS= \
	lib \
	lib/$(HYPRE_ARCH) \
	src \
	other

#######################################################################
#
# top level flow of control and messages
#
#######################################################################

default: subdirs libs archive

libs: src

lib: libs

archive: $(PLIB)
	$(RANLIB) $(PLIB)

#
# libs plus FEI
all:
	make libs;
	make driver;
	make fe;

src fe: $(FEI_OBJS)
	ar rs $(PLIB) $(FEI_OBJS)

other: $(OTHER_OBJ)
	ar rs $(PLIB) $(OTHER_OBJ)

drivers driver fei-drivers:
	cd fei-drivers; make FEI_ROOT="$(FEI_ROOT)";

test tests unit-tests:
	cd unit-tests; make FEI_ROOT="$(FEI_ROOT)";

subdirs:
#   Create sub-directories if needed.
	@if [ -d lib/$(HYPRE_ARCH) ] ; then \
		echo "lib/$(HYPRE_ARCH) exists"; \
	else \
		$(MKDIR) lib/$(HYPRE_ARCH); \
	fi

util utils:
	make libs;
	cd util; make $@ FEI_ROOT="$(FEI_ROOT)"

#######################################################################
#
# generic build rules
#
#######################################################################

.SUFFIXES:
.SUFFIXES: .c .cc .mp.o .o
# standard C (.o)
.c.o:
	$(C) $(CFLAGS) -c -o $@ $<

# standard C++ (.o)
.cc.o:
	$(CPP) $(CXXFLAGS) $(C++DEFINES) \
		$(C++INCLUDES) \
		-o $@ $<

# mpi dependent c++ (.mp.o)
.cc.mp.o:
	$(CPP) $(CXXFLAGS) $(C++DEFINES) \
		$(C++INCLUDES_MPI) \
		-o $@ $<

###################
# special definitions and rules for external library preconditioners.
###################

#######################################################################
#
# distribution rules. not current.
#
#######################################################################
release:
	cd $(FEI_ROOT); make clean
	cd $(FEI_ROOT)/..; tar -cf fei-isis-release.tar fei-isis


#######################################################################
#
# install rules, in sort of generic form until we get fei outa here.
#
#######################################################################
uninstall: $(PACKAGE_DIR)
	/bin/rm -rf $(PACKAGE_DIR)

install: lib
	@cp -f ${PLIB} $$HYPRE_INSTALL_DIR/lib

install.$(PACKAGE):	install-dirs.$(PACKAGE) \
			install-header.$(PACKAGE) \
			install-lib.$(PACKAGE) \
			install-readme.$(PACKAGE)

install-dirs.$(PACKAGE): $(INSTALL_ROOT) have-permission
	@if ! [ -d $(PACKAGE_DIR) ] ; then \
		$(MKDIR) $(PACKAGE_DIR); \
		make install-subdirs.$(PACKAGE); \
	else  \
		echo "* * Will not install on top of existing $(PACKAGE)!"; \
		echo "* * Move existing $(PACKAGE_DIR), or"; \
		echo "* * 'make uninstall' to delete $(PACKAGE_DIR)"; \
	fi

install-subdirs.$(PACKAGE): have-permission
	@if [ -d $(PACKAGE_DIR) ] ; then \
		for newdir in $(FEI_SUBDIRS) ; do \
			if ! [ -d $(PACKAGE_DIR)/$$newdir ] ; then \
				$(MKDIR) $(PACKAGE_DIR)/$$newdir; \
			fi; \
		done \
	fi

install-header.$(PACKAGE): have-permission
	@$(CP) fei-isis.h $(PACKAGE_DIR)
	@$(CHMOD) 644 $(PACKAGE_DIR)/fei-*.h
	@for hdir in src other ; do \
		if [ -d $(PACKAGE_DIR)/$$hdir ] ; then \
			$(CP) $$hdir/*.h $(PACKAGE_DIR)/$$hdir; \
			$(CHMOD) 644 $(PACKAGE_DIR)/$$hdir/*.h; \
		else \
			echo "Failed to install $(PACKAGE_DIR)/$$hdir"; \
		fi; \
	done

install-lib.$(PACKAGE): $(PLIB) have-permission
	@$(CP) $(PLIB) $(PACKAGE_DIR)/lib/$(HYPRE_ARCH)
	@$(RANLIB) $(PACKAGE_DIR)/lib/$(HYPRE_ARCH)/libfei_isis.a
	@$(CHMOD) 644 $(PACKAGE_DIR)/lib/$(HYPRE_ARCH)/libfei_isis.a

install-readme.$(PACKAGE): have-permission
	-@$(RM) $(PACKAGE_DIR)/README.$(PACKAGE).txt
	@echo "$(CPP) -I$(PACKAGE_DIR) -L$(PACKAGE_DIR)/lib/$(HYPRE_ARCH) -lfei_isis" \
	> $(PACKAGE_DIR)/README.$(PACKAGE).txt
	@echo "#include <fei-isis.h> in C++." \
	>> $(PACKAGE_DIR)/README.$(PACKAGE).txt
	@echo "$(CPP) -I$(PACKAGE_DIR) to compile with $(PACKAGE)"
	@echo "$(CPP) -L$(PACKAGE_DIR)/lib/$(HYPRE_ARCH) -lfei_isis to link $(PACKAGE)."

have-permission:
	@touch $(INSTALL_ROOT)/.fei.writable.install

#######################################################################
#
# clean rules
#
#######################################################################
clean cleandir veryclean:
	@touch erase.this
	-$(RM) -f PI* erase.* *.out #* *~ emake.log
	@cd other; make clean FEI_ROOT="$(FEI_ROOT)"
	cd src; make clean FEI_ROOT="$(FEI_ROOT)"
	cd lib; make clean FEI_ROOT="$(FEI_ROOT)" HYPRE_ARCH="$(HYPRE_ARCH)"
	cd fei-drivers; make clean FEI_ROOT="$(FEI_ROOT)"
	cd util; make clean FEI_ROOT="$(FEI_ROOT)"

neat:
	touch erase.this
	rm -f PI* erase.* *.out #* *~
	cd other; make $@ FEI_ROOT="$(FEI_ROOT)"
	cd fei-drivers; make clean FEI_ROOT="$(FEI_ROOT)"
	cd util; make clean FEI_ROOT="$(FEI_ROOT)"

clean.$(FEI_DIR):
	@cd $(FEI_DIR); touch erase.ii
	@cd $(FEI_DIR); $(RM) -fr *~ *.o *.mp.o \
		./ptrepository erase.* *.ii core ii_files/*.ii emake.log
	@echo "=============> Make $(FEI_DIR) clean complete"

distclean:	clean
	$(RM) -rf */ii_files make.options

#######################################################################
#
# help rules
#
#######################################################################

targets help:
	@echo '### FEI-ISIS top level build targets ###'
	@echo 'default: builds libfei files'
	@echo 'install: installs FEI, usually in /usr/local/fei-isis'
	@echo 'uninstall: removes installed FEI-ISIS (usually /usr/local/fei-isis) entirely
	@echo 'clean: removes .o and other intermediate files'
	@echo 'distclean: removes configure generated files, template dirs, libs'
	@echo 'test: runs some simple tests'
	@echo 'depend-portable: (linux only) (broken) generate Makefile.depend'
	@echo '### FEI subdirectory build targets ###'
	@echo 'default: builds local directory and updates libisis files'
	@echo 'clean: removes .o and other intermediate files'
	@echo '### FEI drivers subdirectory build targets ###'
	@echo 'See subdirectory makefiles for more details'

# if tags targets are to exist, they should be here. etags ctags

#######################################################################
#
# generate dependencies on isis sources for isis and wrapper classes.
# move to separate makefile.developers
#
#######################################################################
DEPEND_FLAGS = $(C++INCLUDES_MPI) $(C++DEFINES) -DSLOPPY_DEPENDS
DEPEND_SRCS= \
	$(OTHER_SRCS) \
	$(FEI__SRCS)

nodepend:
	\$(DEPEND_NONE) \$(DEPEND_FLAGS) \$?

Makefile.depends:
	\$(DEPEND_NONE) \$(DEPEND_FLAGS) \$?

#does require g++ or a CC which emits dependencies when properly flagged.
depend-nosys depend-no-sys: $(DEPEND_SRCS) 
	if [ $(HYPRE_ARCH) = LINUX ]; then \
		$(DEPEND_NOSYS) $(DEPEND_FLAGS) $(DEPEND_INCLUDES) $? ; \
	else \
		echo 'depend-nosys requires the g++ compiler' ; \
	fi

depend: $(FEI_SRCS)
	@if [ $(HYPRE_ARCH) = LINUX ]; then \
		$(DEPEND_ALL) $(DEPEND_FLAGS) $(DEPEND_INCLUDES) $? ; \
	else \
		echo 'depend requires the g++ compiler' ; \
	fi

depend-portable: depend-nosys
	cat Makefile.depends | \
	sed -e 's;$(FEI_ROOT)/;;g' -e 's;^ *;;g' | \
	sed -e 's;^ ;@;g' -e 's;^ ;@;g' -e 's;^ ;@;g' >Makefile.nd
	$(CP) -f Makefile.nd Makefile.depends
	$(RM) -f Makefile.nd

# don't put anything after this line.
# don't move the next line up in the file, either.

include Makefile.depends
