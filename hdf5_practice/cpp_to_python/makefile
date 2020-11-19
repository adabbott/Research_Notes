CC      := g++
INCL1   := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include
LIBS    := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib 
RPATH   := -Wl,-rpath,"/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib"

TARGETS := test
OBJ     := test.o 
DEPS    := 

.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ -O3 -std=c++11 -I $(INCL1) -lhdf5 -lhdf5_cpp -L $(LIBS) $(RPATH)

$(TARGETS): $(OBJ)
	$(CC)  $^ -o $@ -O3 -std=c++11 -I $(INCL1) -lhdf5 -lhdf5_cpp -L $(LIBS) $(RPATH) 
#-fPIC -shared 

# g++ create.cpp -I/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include -L/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib -lhdf5 -lhdf5_cpp
# g++ create.cc -I/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include -L/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib -lhdf5 -lhdf5_cpp -Wl,-rpath,"/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib"


# g++ create.cc -fPIC -shared -I/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include -L/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib -lhdf5 -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_hl -Wl,-rpath,"/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib"
