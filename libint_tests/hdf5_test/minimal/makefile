CC       := g++
INCL1    := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include
LIB_HDF5 := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib 
#RPATH   := -Wl,-rpath,"/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib"

TARGETS := create 
OBJ     := create.o 
DEPS    := 

.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ -fPIC -std=c++11 -I $(INCL1) -lhdf5 -lhdf5_cpp -L $(LIB_HDF5)

$(TARGETS): $(OBJ)
	$(CC) $^ -o $@ -fPIC -shared -std=c++11 -I $(INCL1) -lhdf5 -lhdf5_cpp -L $(LIB_HDF5)

# g++ create.cpp -I/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include -L/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib -lhdf5 -lhdf5_cpp
# g++ create.cc -I/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include -L/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib -lhdf5 -lhdf5_cpp -Wl,-rpath,"/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib"


