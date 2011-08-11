// header for tqli.c
#include<stdio.h>
#include<math.h>
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

int tqli(double* d, double* e, int n, double* z);

double pythag(double a, double b);
