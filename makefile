CC = nvcc
CFLAGS = -w -g -G -DTHRUST_DEBUG -arch=sm_20
LIBS = -lcublas -lcusparse
OBJS = heisenberg.o lanczos.o testhamiltonian.o 

a.out : $(OBJS) 
	$(CC) $(CFLAGS) $(OBJS) -o a.out $(LIBS)

heisenberg.o : heisenberg.cpp testhamiltonian.h 
	$(CC) $(CFLAGS) -c heisenberg.cpp

lanczos.o : lanczos.cu lanczos.h
	$(CC) $(CFLAGS) -c $(LIBS) lanczos.cu

tqli.o : tqli.c tqli.h
	$(CC) $(CFLAGS) -c tqli.c

testhamiltonian.o : testhamiltonian.cu testhamiltonian.h
	$(CC) $(CFLAGS) -c testhamiltonian.cu

clean : 
	rm *.o
