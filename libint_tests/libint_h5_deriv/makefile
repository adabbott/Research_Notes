CC      := g++
# These need to be absolute paths to Python headers, pybind11 headers, Eigen headers, libint API files and headers, and the libint static library libint2.a
LIBINT_PREFIX := /home/adabbott/Git/libint_am3/libint/build/libint-2.7.0-beta.6/PREFIX

INCL1   := /home/adabbott/anaconda3/envs/psijax/include/python3.6m
INCL2   := /home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include
INCL3   := /home/adabbott/anaconda3/envs/psijax/include/eigen3
INCL4   := $(LIBINT_PREFIX)/include
INCL5   := $(LIBINT_PREFIX)/include/libint2 
INCL6   := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include
LIBS    := $(LIBINT_PREFIX)/lib
LIBHDF5 := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib 
RPATH   := -Wl,-rpath,"/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib"


# This target suffix should be set to whatever is returned by the command `python3-config --extension-suffix` entered on command line  
# and it should match the same python version referenced in the above includes INCL1 INCL2  (3.6 in this case)
TARGETS := pylibint.cpython-36m-x86_64-linux-gnu.so
OBJ     := pylibint.o
DEPS    := 

.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

# Way faster with -O3
$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ -fPIC -I $(INCL1) -I $(INCL2) -I $(INCL3) -I $(INCL4) -I $(INCL5) -lint2 -L $(LIBS) -I $(INCL6) -lhdf5 -lhdf5_cpp -L $(LIBHDF5) $(RPATH)

$(TARGETS): $(OBJ)
	$(CC)  $^ -o $@ -shared -fPIC -I $(INCL1) -I $(INCL2) -I $(INCL3) -I $(INCL4) -I $(INCL5) -lint2 -L $(LIBS) -I $(INCL6) -lhdf5 -lhdf5_cpp -L $(LIBHDF5) $(RPATH)
