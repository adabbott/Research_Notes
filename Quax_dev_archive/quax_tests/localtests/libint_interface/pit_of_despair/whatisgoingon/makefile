CC      := g++
INCL1   := /home/adabbott/anaconda3/envs/psijax/include/eigen3
INCL2   := /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include
INCL3   := /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 
LIBS    := /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ 

TARGETS := test
OBJ     := test.o
DEPS    := 

.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ -I $(INCL1) -I $(INCL2) -I $(INCL3) -lint2 -L $(LIBS)

$(TARGETS): $(OBJ)
	$(CC)  $^ -o $@ -I $(INCL1) -I $(INCL2) -I $(INCL3) -lint2 -L $(LIBS)
