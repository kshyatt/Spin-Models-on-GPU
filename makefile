CC = nvcc
CFLAGS = -g -G -gencode arch=compute_20,code=sm_21
LANCZLIBS = -lcublas -lcusparse
HAMLIBS = -Lsort/sort/gnu/release -lmgpusort -lcuda -lcudart
OBJS = launcher.o lanczos.o hamiltonian.o heisenberg.o xy.o tfising.o lattice.o

a.out : $(OBJS) 
	$(CC) $(CFLAGS) $(OBJS) -o a.out $(HAMLIBS) $(LANCZLIBS)

launcher.o : launcher.cu lanczos.h lattice.h
	$(CC) $(CFLAGS) -c launcher.cu

lanczos.o : lanczos.cu lanczos.h 
	$(CC) $(CFLAGS) -c $(LANCZLIBS) lanczos.cu

hamiltonian.o : hamiltonian.cu hamiltonian.h
	$(CC) $(CFLAGS) -c $(HAMLIBS) hamiltonian.cu

heisenberg.o : heisenberg.cu hamiltonian.h
	$(CC) $(CFLAGS) -c heisenberg.cu

xy.o : xy.cu hamiltonian.h
	$(CC) $(CFLAGS) -c xy.cu

tfising.o : tfising.cu hamiltonian.h
	$(CC) $(CFLAGS) -c tfising.cu

lattice.o : lattice.cpp lattice.h
	$(CC) $(CFLAGS) -c lattice.cpp

clean : 
	rm *.o
