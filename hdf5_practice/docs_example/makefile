CC      := g++ 
INCL1   := /usr/include/hdf5/serial
LIBS    := /usr/lib/x86_64-linux-gnu/hdf5/serial
RPATH   := -Wl,-rpath,"/usr/lib/x86_64-linux-gnu/hdf5/serial"

TARGETS := control
OBJ     := control.o
DEPS    := 

.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

#$(OBJ): %.o : %.cc $(DEPS)
##	$(CC) -c $< -o $@ -I $(INCL1) -lhdf5 -L $(LIBS) $(RPATH)
#	$(CC) -c $< -o $@ $(RPATH) -I $(INCL1) -lhdf5 
#
#$(TARGETS): $(OBJ)
#	#$(CC)  $^ -o $@ -I $(INCL1) -lhdf5 -L $(LIBS) $(RPATH)
#	$(CC) -c $< -o $@ $(RPATH) -I $(INCL1) -lhdf5 

$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -std=c++11 -c $< -o $@ $(RPATH) -I $(INCL1) -lhdf5 -L $(LIBS)

$(TARGETS): $(OBJ)
	$(CC) -std=c++11 $^ -o $@ $(RPATH) -I $(INCL1) -lhdf5 -L $(LIBS)
