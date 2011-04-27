/// 
/// @file lapack.h
/// @Synopsis Definition of the wrapper function for CLAPACK diag
/// @author Ivan Gonzalez
/// @date 2007-02-06
/// 
/// In the case of complex wavefunction the set tqli and tred3 does 
/// not diagonalize the DM matrix properly. Here we define a wrapper 
/// function that use the standard LAPACK function zheev to diagonalize
/// a complex matrix. The wrapper knows about blitz arrays and the 
/// different ordering between C and FORTRAN.
///
#ifndef LAPACK_H
#define LAPACK_H
//#include "precompile_methods.h"
#include<complex>
#include <blitz/array.h>
#include<vector>
BZ_USING_NAMESPACE(blitz)

/*****************************************************************************/
///
/// Extern declaration of CLAPACK functions
///
extern "C" {
        
    int zheev_(char *jobz, char *uplo, int *n, complex<double> 
	    *a, int *lda, double *w, complex<double> *work, int *lwork, 
	    double *rwork, int *info);
 
    int zheevd_(char *jobz, char *uplo, int *n, 
	    complex<double> *a, int *lda, double *w, complex<double> *work, 
	    int *lwork, double *rwork, int *lrwork, int *iwork, 
    	    int *liwork, int *info);

    int dsyev_(char *jobz, char *uplo, int *n, double *a,
	    int *lda, double *w, double *work, int *lwork, int *info);
}
/*****************************************************************************/
///
/// Function to take a complex my_Matrix and diag it
///
void diagWithLapack(Array<complex<double>, 2>& DMpart, 
	vector<double>& EigenVals);
/*****************************************************************************/
///
/// Function to take a real my_Matrix  and diag it with lapack
///
void diagWithLapack_R(Array<double, 2>& DMpart, vector<double>& EigenVals);
#endif
