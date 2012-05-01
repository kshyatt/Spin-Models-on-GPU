CC = nvcc
CFLAGS = -g -G -arch=sm_21
LANCZLIBS = -lcublas -lcusparse 
HAMLIBS = -Lsort/sort/gnu/release -lmgpusort -lcuda -lcudart
OBJS = heisenberg.o lanczos.o hamiltonian.o lattice.o

a.out : $(OBJS) 
	$(CC) $(CFLAGS) $(OBJS) -o a.out $(HAMLIBS) $(LANCZLIBS)

heisenberg.o : heisenberg.cu lanczos2.h lattice.h
	$(CC) $(CFLAGS) -c heisenberg.cu

lanczos.o : lanczos2.cu lanczos2.h 
	$(CC) $(CFLAGS) -c $(LANCZLIBS) lanczos2.cu

hamiltonian.o : hamiltonian.cu hamiltonian.h
	$(CC) $(CFLAGS) -c $(HAMLIBS) hamiltonian.cu

lattice.o : lattice.cpp lattice.h
	$(CC) $(CFLAGS) -c lattice.cpp

clean : 
	rm *.o
