CC = nvcc
CFLAGS = -w -g -G -arch=sm_21
LANCZLIBS = -lcublas -lcusparse 
HAMLIBS = -Lsort/sort/gnu/release -lmgpusort -lcuda -lcudart
OBJS = heisenberg.o lanczos.o testhamiltonian.o 

a.out : $(OBJS) 
	$(CC) $(CFLAGS) $(OBJS) -o a.out $(HAMLIBS) $(LANCZLIBS)

heisenberg.o : heisenberg.cpp testhamiltonian.h 
	$(CC) $(CFLAGS) -c heisenberg.cpp

lanczos.o : lanczos.cu lanczos.h 
	$(CC) $(CFLAGS) -c $(LANCZLIBS) lanczos.cu

testhamiltonian.o : testhamiltonian.cu testhamiltonian.h
	$(CC) $(CFLAGS) -c $(HAMLIBS) testhamiltonian.cu

clean : 
	rm *.o
