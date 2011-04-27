//c++ class for creating a general Hamiltonian in c++ vectors
//Roger Melko, November 2007
#ifndef GenHam_H
#define GenHam_H

#include <iostream>
#include <vector>
using namespace std;

#include <blitz/array.h>
BZ_USING_NAMESPACE(blitz)

typedef long double h_float;  //precision for Hamiltonian storage

class GENHAM{

  public:
    int Fdim; //"full" Hilbert space
    int Vdim; //dimenson of reduced Hilbert space

    vector<vector<long> > PosHam;
    vector<vector<h_float> > ValHam;
    //vector<double> DiagHam;

    vector<long> Basis;
    vector<long> BasPos;

    Array<int,2> Bond;
    Array<int,2> OtherTwoX; //sites on the plaquette that are not in bond
    Array<int,2> OtherTwoY;
    Array<int,2> PlaqX;
    Array<int,2> PlaqY;
    
    Array<double,2> Ham;  //full hamiltonian

    GENHAM(const int,const h_float J_, const h_float J2_,const h_float Q_,const int Sz); 
    void printg();
    //double at(const int , const int );
    Array<double,1> apply(const Array<double,1>&);

    void Bonds_16A();
    void Bonds_16B();
    void Bonds_18A();
    void Bonds_20A();
    void Bonds_22A();
    void Bonds_24A();
    void Bonds_24R();
    void Bonds_24M();
    void Bonds_26A();
    void Bonds_30R();
    void SparseHamJQ();
    void FullHamJQ();

  private:
    int Nsite; //number sites
    unsigned long SpinInv; //SpinInv integer = 0 for no spin inversion symmetry


    h_float JJ; //heisenberg exchange value
    h_float J2; //next-nearest neighbor exchange value
    h_float QQ; //ring-exchange value

    double HdiagPart(const long);
    double HOFFdBondX(const int, const long);
    double HOFFdBondY(const int, const long);
    double HOFFdBond_02(const int, const long);
    double HOFFdBond_13(const int, const long);
    double HOFFdPlaq(const int, const long);

};

#endif
