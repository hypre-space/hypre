#! /usr/bin/env python
# Build file for Python modules
import sys
from re import compile
from distutils.core import setup, Extension

inc_re = compile('^--include-dirs=(.*)$')
lib_re = compile('^--library-dirs=(.*)$')
old_argv = sys.argv
sys.argv = []
inc_dirs = ['.']
lib_dirs = []

for i in old_argv:
  m = inc_re.match(i)
  if (m):
    if (len(m.group(1))): inc_dirs.append(m.group(1))
  else:
    m = lib_re.match(i)
    if (m):
      if (len(m.group(1))): lib_dirs.append(m.group(1))
    else:
      sys.argv.append(i)
setup(name='babel',
  include_dirs=inc_dirs,
  headers = [
    'SIDL_BaseClass_Module.h',
    'SIDL_BaseException_Module.h',
    'SIDL_BaseInterface_Module.h',
    'SIDL_ClassInfoI_Module.h',
    'SIDL_ClassInfo_Module.h',
    'SIDL_DLL_Module.h',
    'SIDL_Loader_Module.h'
  ],
  packages = [
    'SIDL'
  ],
  ext_modules = [
    Extension('SIDL.BaseException',
      ["SIDL/BaseException_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=["sidl"]),
    Extension('SIDL.BaseInterface',
      ["SIDL/BaseInterface_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=["sidl"]),
    Extension('SIDL.BaseClass',
      ["SIDL/BaseClass_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=["sidl"]),
    Extension('SIDL.ClassInfo',
      ["SIDL/ClassInfo_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=["sidl"]),
    Extension('SIDL.DLL',
      ["SIDL/DLL_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=["sidl"]),
    Extension('SIDL.ClassInfoI',
      ["SIDL/ClassInfoI_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=["sidl"]),
    Extension('SIDL.Loader',
      ["SIDL/Loader_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=["sidl"])
  ])
