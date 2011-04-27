#ifndef PARAMSIM_H
#define PARAMSIM_H


//Class to read in the simulation parameters from a file
// Adapted to the JQ ED code

class PARAMS
{
  public:
    double JJ_; //the heisenberg exchange
    double J2_; //the next-nearest neighbor heisenberg exchange
    double QQ_; //the 4-particle SU(2) exchange
    int Sz_; // z-component of total spin
    // FOR LANCZOS
    int Neigen_;    //: # of eigenvalues to converge
    int valvec_; //  1 for -values only, 2 for vals AND vectors
    // FULL_DIAG?

    PARAMS(){
      //initializes commonly used parameters from a file
      ifstream pfin;
      pfin.open("param.dat");
    
      pfin >> JJ_;
      pfin >> J2_;
      pfin >> QQ_;
      pfin >> Sz_;
      pfin >> Neigen_;
      pfin >> valvec_;
    
      pfin.close();
    }//constructor

}; //PARAMS

#endif
