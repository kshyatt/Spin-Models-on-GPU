#include "GenHam.h"

//----------------------------------------------------------
GENHAM::GENHAM(const int Ns, const h_float J_, const h_float J2_, const h_float Q_, const int Sz)  
               : JJ(J_), J2(J2_), QQ(Q_) 
//create bases and determine dim of full Hilbert space
{
  int Dim;
  Nsite = Ns;

  Dim = 2;  //  S=1/2 models : two states
  for (int ch=1; ch<Nsite; ch++) Dim *=2;
  Fdim = Dim;

  BasPos.resize(Dim,-1); //initialization 

  Vdim=0;
  unsigned long temp;    //create basis (16 site cluster)

  for (unsigned long i1=0; i1<Dim; i1++) 
  {
      temp = 0;
      for (int sp =0; sp<Nsite; sp++)
          temp += (i1>>sp)&1;  //unpack bra
      if (temp==(Nsite/2+Sz) ){ 
          Basis.push_back(i1);
          BasPos.at(i1)=Basis.size()-1;
          Vdim++;
      }
  }//Dim

  cout<<"Vdim "<<Vdim<<" "<<Dim<<endl;

}//constructor


//----------------------------------------------------------
void GENHAM::printg()
{
  int i,j;
  vector<int> tempP;
  vector<h_float> tempV;

  for (i=0; i<PosHam.size(); i++){
    //cout<<PosHam[i][0]<<" * ";
    cout<<i+1<<" * ";
    for (j=0; j<=PosHam[i][0]; j++){
     cout<<"("<<PosHam[i][j]+1<<","<<ValHam[i][j]<<") ";
   }
   cout<<endl;
  }

}//print


//----------------------------------------------------------
void GENHAM::FullHamJQ(){

  int ii, tempI;
  vector<long> revBas(Fdim,-1);

  for (ii=0; ii<Basis.size(); ii++) { //reverse lookup
    tempI = Basis.at(ii);
    revBas.at(tempI) = ii;
  }
    
  Ham.resize(Vdim,Vdim);
  Ham = 0;

  unsigned long tempi, tempj, tempod;
  double tempD;
  int si,sj,sk,sl, revPos;
  for (ii=0; ii<Basis.size(); ii++){
    tempi = Basis.at(ii);

    //Hamiltonian for diagonal
    tempD = (*this).HdiagPart(tempi);
    Ham(ii,ii) = tempD;

    for (int T0=0; T0<Nsite; T0++){ // Now generate off-diagonal part

      si = Bond(T0,0);
      //if (si != T0) cout<<"Square error \n";
      // X Bond
      tempod = tempi;
      sj = Bond(T0,1);
      tempod ^= (1<<si);   //toggle bit 
      tempod ^= (1<<sj);   //toggle bit 
      revPos = revBas.at(tempod);
      if (revPos != -1){
        tempD = (*this).HOFFdBondX(T0,tempi);
        Ham(ii,revPos) = tempD;
      }

      // Y Bond
      tempod = tempi;
      sj = Bond(T0,2);
      tempod ^= (1<<si);   //toggle bit 
      tempod ^= (1<<sj);   //toggle bit 
      revPos = revBas.at(tempod);
      if (revPos != -1){
        tempD = (*this).HOFFdBondY(T0,tempi);
        Ham(ii,revPos) = tempD;
      }

    }//si

  }//ii

}//FullHamHeis

//----------------------------------------------------------
void GENHAM::SparseHamJQ()
{
  int ii, jj;

  int Rsize;
  vector<long> tempBas;
  //vector<long> tempBas2;
  vector<h_float> tempH;
  unsigned long tempi, tempj, tempod;
  int si, sj,sk,sl;
  double tempD;

  for (ii=0; ii<Basis.size(); ii++){
    tempH.clear(); 
    tempBas.clear();

    tempi = Basis.at(ii);
    tempBas.push_back(0); //first element (Row size)
    tempH.push_back(0); //make base 0

    //-----1:   diagonal 
    tempBas.push_back(BasPos.at(tempi));  
    tempD = (*this).HdiagPart(tempi);
    tempH.push_back(tempD); 

    for (int T0=0; T0<Nsite; T0++){ //T0 is your square index

      si = Bond(T0,0); //the lower left bond spin is not always T0
      //if (si != T0) cout<<"Square error 2\n";
      //-----2:   first bond (Horizontal)
      tempod = tempi;
      sj = Bond(T0,1);
      tempod ^= (1<<si);   //toggle bit 
      tempod ^= (1<<sj);   //toggle bit 
      if (BasPos.at(tempod) != -1 && BasPos.at(tempod) > ii){ //build only upper half of matrix
        tempBas.push_back(BasPos.at(tempod));
        tempD = (*this).HOFFdBondX(T0,tempi);
        tempH.push_back(tempD); 
      }
 
       //-----3:   second bond (Vertical)
      tempod = tempi;
      sj = Bond(T0,2);
      tempod ^= (1<<si);   //toggle bit 
      tempod ^= (1<<sj);   //toggle bit 
      if (BasPos.at(tempod) != -1 && BasPos.at(tempod) > ii){ 
        tempBas.push_back(BasPos.at(tempod));
        tempD = (*this).HOFFdBondY(T0,tempi);
        tempH.push_back(tempD); 
      }

    }//si

    tempBas.at(0) = tempBas.size()-1;
    //cout<<tempBas.at(0)<<" "<<tempBas.size()<<" "<<tempH.size()<<endl;

    //bubble sort (slow) 
    long stemp;
    bool noswap = false;
    while (noswap == false){
      noswap = true; 
      for (int i2=1; i2<tempBas.size()-1; i2++){ //ignore 0 element
        if (tempBas.at(i2) > tempBas.at(i2+1) ) {
          stemp = tempBas.at(i2);
          tempBas.at(i2) = tempBas.at(i2+1);
          tempBas.at(i2+1) = stemp;
          tempD = tempH.at(i2);
          tempH.at(i2) = tempH.at(i2+1);
          tempH.at(i2+1) = tempD;
          noswap = false;
        }
      }//i2
    }//while

    PosHam.push_back(tempBas);
    ValHam.push_back(tempH);

  }//ii       

}//Heisenberg

//----------------------------------------------------------
double GENHAM::HdiagPart(const long bra){

  int S0b,S1b ;  //spins (bra 
  int T0,T1;  //site
  int P0, P1, P2, P3; //sites for plaquette (Q)
  int s0p, s1p, s2p, s3p;
  double valH = 0;

  for (int Ti=0; Ti<Nsite; Ti++){
    //***HEISENBERG PART

    T0 = Bond(Ti,0); //lower left spin
    S0b = (bra>>T0)&1;  
    //if (T0 != Ti) cout<<"Square error 3\n";
    T1 = Bond(Ti,1); //first bond
    S1b = (bra>>T1)&1;  //unpack bra
    valH += JJ*(S0b-0.5)*(S1b-0.5);
    T1 = Bond(Ti,2); //second bond
    S1b = (bra>>T1)&1;  //unpack bra
    valH += JJ*(S0b-0.5)*(S1b-0.5);

  }//T0

  //cout<<bra<<" "<<valH<<endl;

  return valH;

}//HdiagPart

//----------------------------------------------------------
double GENHAM::HOFFdBondX(const int si, const long bra){

  double valH;
  int S0, S1;
  int T0, T1;

  valH = JJ*0.5; //contribution from the J part of the Hamiltonian

  return valH;

}//HOFFdPart

//----------------------------------------------------------
double GENHAM::HOFFdBondY(const int si, const long bra){

  double valH;
  int S0, S1;
  int T0, T1;

  valH = JJ*0.5; //contribution from the J part of the Hamiltonian

  return valH;

}//HOFFdPart

