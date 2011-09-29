CC = nvcc
CFLAGS = -w -g -G -arch=sm_21
LIBS = -lcublas -lcusparse -Lsort/sort/gnu/release -lmgpusort -lcuda -lcudart
OBJS = heisenberg.o lanczos.o testhamiltonian.o 

a.out : $(OBJS) 
	$(CC) $(CFLAGS) $(OBJS) -o a.out $(LIBS)

heisenberg.o : heisenberg.cpp testhamiltonian.h 
	$(CC) $(CFLAGS) -c heisenberg.cpp

lanczos.o : lanczos.cu lanczos.h 
	$(CC) $(CFLAGS) -c $(LIBS) lanczos.cu

testhamiltonian.o : testhamiltonian.cu testhamiltonian.h
	$(CC) $(CFLAGS) -c $(LIBS) testhamiltonian.cu

clean : 
	rm *.o
