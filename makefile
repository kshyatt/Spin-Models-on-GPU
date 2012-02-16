CC = nvcc
CFLAGS = -w -g -G -arch=sm_21
#LANCZLIBS = -lcublas -lcusparse 
HAMLIBS = -Lsort/sort/gnu/release -lmgpusort -lcuda -lcudart
OBJS = heisenberg.o lanczos.o hamiltonian.o lattice.o

a.out : $(OBJS) 
	$(CC) $(CFLAGS) $(OBJS) -o a.out $(HAMLIBS) $(LANCZLIBS)

heisenberg.o : heisenberg.cu hamiltonian.h lattice.h
	$(CC) $(CFLAGS) -c heisenberg.cu

#lanczos.o : lanczos.cu lanczos.h 
#	$(CC) $(CFLAGS) -c $(LANCZLIBS) lanczos.cu

hamiltonian.o : hamiltonian.cu hamiltonian.h
	$(CC) $(CFLAGS) -c $(HAMLIBS) hamiltonian.cu

lattice.o : lattice.cpp lattice.h
	$(CC) $(CFLAGS) -c lattice.cpp

clean : 
	rm *.o
