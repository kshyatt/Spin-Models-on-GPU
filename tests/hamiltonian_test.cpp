#include "../lanczos.h"
#include "gtest/gtest.h"
#include <string>

class hamiltonianTest : public ::testing::Test
{
    public:

        std::string* correct_basis;
        std::string* correct_basis_position;
        std::string* test_basis;
        std::string* test_basis_position;

        d_hamiltonian* hamil;
        parameters* data;
        int** Bond;
        int* num_elem;
        double** energies;
        double** groundstate;

        //hamiltonianTest();
        //~hamiltonianTest();

};


class isingTest : public hamiltonianTest
{
    public:
        isingTest();
        ~isingTest();
};

isingTest::isingTest()
{
    correct_basis = (string*)malloc(sizeof(string));
    correct_basis_position = (string*)malloc(sizeof(string));

    correct_basis[0] = "../isingbasis.dat";
    correct_basis_position[0] = "../isingbasisposition.dat";
    
    test_basis = (string*)malloc(sizeof(string));
    test_basis_position = (string*)malloc(sizeof(string));

    test_basis[0] = "../testisingbasis.dat";
    test_basis_position[0] = "../testisingbasisposition.dat";

    num_elem = (int*)malloc(sizeof(int));
    data = (parameters*)malloc(sizeof(parameters));
    data[0].nsite = 16;
    data[0].modelType = 2;
    data[0].dimension = 1;

    hamil = (d_hamiltonian*)malloc(sizeof(d_hamiltonian));
    Bond = (int**)malloc(sizeof(int*));
    Bond[0] = (int*)malloc(2*data[0].nsite*sizeof(int));
    for( int i = 0; i < data[0].nsite; i++)
    {
        Bond[0][i] = i;
        Bond[0][i + data[0].nsite] = (i + 1)%data[0].nsite;
    }

};

isingTest::~isingTest()
{
    free(Bond[0]);
    free(Bond);
    free(hamil);
    free(data);
    free(num_elem);
    free(correct_basis);
    free(correct_basis_position);
    free(test_basis);
    free(test_basis_position);
};

TEST_F(isingTest, GetBasisCorrect)
{

    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);

    ASSERT_EQ(0, test_basis[0].compare(correct_basis[0]));
    ASSERT_EQ(0, test_basis_position[0].compare(correct_basis_position[0]));
}    

TEST_F(isingTest, GetNumElemCorrect)
{

    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);
    ASSERT_EQ(num_elem[0], 11212111);
}

TEST_F(isingTest, GetHamiltonianCorrect)
{
    //We want to test for two cases: J1 = 0, J2 = 1; J1 = 2; J2 = 0;
    data[0].Sz = 0;
    data[0].J1 = 2.f;
    data[0].J2 = 0.f;
    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);

    string correct_hamil_rows = "../isingcorrectrows1.dat";
    string correct_hamil_cols = "../isingcorrectcols1.dat";
    string correct_hamil_vals = "../isingcorrectvals1.dat";

    string test_hamil_rows = "../isingtestrows1.dat";
    string test_hamil_cols = "../isingtestcols1.dat";
    string test_hamil_vals = "../isingtestvals1.dat";

    ASSERT_EQ(0, test_hamil_rows.compare(correct_hamil_rows));
    ASSERT_EQ(0, test_hamil_cols.compare(correct_hamil_cols));
    ASSERT_EQ(0, test_hamil_vals.compare(correct_hamil_vals));

    data[0].J1 = 0.f;
    data[0].J2 = 1.f;

    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);

    correct_hamil_rows = "../isingcorrectrows2.dat";
    correct_hamil_cols = "../isingcorrectcols2.dat";
    correct_hamil_vals = "../isingcorrectvals2.dat";

    test_hamil_rows = "../isingtestrows2.dat";
    test_hamil_cols = "../isingtestcols2.dat";
    test_hamil_vals = "../isingtestvals2.dat";

    ASSERT_EQ(0, test_hamil_rows.compare(correct_hamil_rows));
    ASSERT_EQ(0, test_hamil_cols.compare(correct_hamil_cols));
    ASSERT_EQ(0, test_hamil_vals.compare(correct_hamil_vals));

}

class heisenbergTest : public hamiltonianTest
{
    public:
        heisenbergTest();
        ~heisenbergTest();
};

heisenbergTest::heisenbergTest()
{
    correct_basis = (string*)malloc(8*sizeof(string));
    correct_basis_position = (string*)malloc(8*sizeof(string));

    correct_basis[0] = "../s0basis.dat";
    correct_basis[1] = "../s1basis.dat";
    correct_basis[2] = "../s2basis.dat";
    correct_basis[3] = "../s3basis.dat";
    correct_basis[4] = "../s4basis.dat";
    correct_basis[5] = "../s5basis.dat";
    correct_basis[6] = "../s6basis.dat";
    correct_basis[7] = "../s7basis.dat";
    correct_basis_position[0] = "../s0basisposition.dat";
    correct_basis_position[1] = "../s1basisposition.dat";
    correct_basis_position[2] = "../s2basisposition.dat";
    correct_basis_position[3] = "../s3basisposition.dat";
    correct_basis_position[4] = "../s4basisposition.dat";
    correct_basis_position[5] = "../s5basisposition.dat";
    correct_basis_position[6] = "../s6basisposition.dat";
    correct_basis_position[7] = "../s7basisposition.dat";
    
    test_basis = (string*)malloc(8*sizeof(string));
    test_basis_position = (string*)malloc(8*sizeof(string));

    test_basis[0] = "../tests0basis.dat";
    test_basis[1] = "../tests1basis.dat";
    test_basis[2] = "../tests2basis.dat";
    test_basis[3] = "../tests3basis.dat";
    test_basis[4] = "../tests4basis.dat";
    test_basis[5] = "../tests5basis.dat";
    test_basis[6] = "../tests6basis.dat";
    test_basis[7] = "../tests7basis.dat";
    test_basis_position[0] = "../tests0basisposition.dat";
    test_basis_position[1] = "../tests1basisposition.dat";
    test_basis_position[2] = "../tests2basisposition.dat";
    test_basis_position[3] = "../tests3basisposition.dat";
    test_basis_position[4] = "../tests4basisposition.dat";
    test_basis_position[5] = "../tests5basisposition.dat";
    test_basis_position[6] = "../tests6basisposition.dat";
    test_basis_position[7] = "../tests7basisposition.dat";
    
    num_elem = (int*)malloc(sizeof(int));
    data = (parameters*)malloc(sizeof(parameters));
    data[0].J1 = 1.f;
    data[0].J2 = 0.f;
    data[0].nsite = 16;
    data[0].modelType = 0;
    data[0].dimension = 2;
    hamil = (d_hamiltonian*)malloc(sizeof(d_hamiltonian));
    Bond = (int**)malloc(sizeof(int*));
    Bond[0] = (int*)malloc(3*data[0].nsite*sizeof(int));
    Fill_Bonds_16B(Bond[0]);
}

heisenbergTest::~heisenbergTest()
{
    free(Bond[0]);
    free(Bond);
    free(hamil);
    free(num_elem);
    free(data);
    free(correct_basis);
    free(correct_basis_position);
    free(test_basis);
    free(test_basis_position);
}

TEST_F(heisenbergTest, GetBasisCorrect)
{


    for(int i = 0; i < 8; i++)
    {
        data[0].Sz = i;
        ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);

        ASSERT_EQ(0, test_basis[i].compare(correct_basis[i]));
        ASSERT_EQ(0, test_basis_position[i].compare(correct_basis_position[i]));    
    }

}

TEST_F(heisenbergTest, GetNumElemCorrect)
{
    data[0].Sz = 0;
    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);
    ASSERT_EQ(num_elem[0], 232518);
}

TEST_F(heisenbergTest, GetHamiltonianCorrect)
{
    data[0].Sz = 0;
    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);

    string correct_hamil_rows = "../heiscorrectrows.dat";
    string correct_hamil_cols = "../heiscorrectcols.dat";
    string correct_hamil_vals = "../heiscorrectvals.dat";

    string test_hamil_rows = "../heistestrows.dat";
    string test_hamil_cols = "../heistestcols.dat";
    string test_hamil_vals = "../heistestvals.dat";

    ASSERT_EQ(0, test_hamil_rows.compare(correct_hamil_rows));
    ASSERT_EQ(0, test_hamil_cols.compare(correct_hamil_cols));
    ASSERT_EQ(0, test_hamil_vals.compare(correct_hamil_vals));
}

TEST_F(heisenbergTest, HeisenbergCorrect)
{
    //Diagonalize the Hamiltonian to see if we get correct answers

    energies = new double* [1];
    energies[0] = new double[3];

    groundstate = new double* [1];
    groundstate[0] = new double[hamil[0].sectorDim];

    lanczos(1, num_elem, hamil, groundstate, energies, 200, 3, 1e-10);

    ASSERT_NEAR(energies[0][0], -11.2282329996, 1e-3);
    ASSERT_NEAR(energies[0][1], -10.6499034854, 1e-3);
    ASSERT_NEAR(energies[0][2], -9.51748118382, 1e-3);

    delete [] groundstate[0];
    delete [] groundstate;

    delete [] energies[0];
    delete [] energies;
}

TEST_F(isingTest, IsingCorrect)
{

    data[0].J1 = 4.f;
    data[0].J2 = 0.f;

    energies = new double* [1];
    energies[0] = new double[3];

    groundstate = new double* [1];
    groundstate[0] = new double[hamil[0].sectorDim];
    
    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);
    lanczos(1, num_elem, hamil, groundstate, energies, 200, 3, 1e-10);

    ASSERT_NEAR(energies[0][0], -16.0, 1e-3);
    ASSERT_NEAR(energies[0][1], -12.0, 1e-3);
    ASSERT_NEAR(energies[0][2], -8.0, 1e-3);

    data[0].J1 = 0.f;
    data[0].J2 = 2.f;

    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);
    lanczos(1, num_elem, hamil, groundstate, energies, 200, 3, 1e-10);

    ASSERT_NEAR(energies[0][0], -16.0, 1e-3);
    ASSERT_NEAR(energies[0][1], -12.0, 1e-3);
    ASSERT_NEAR(energies[0][2], -8.0, 1e-3);

    data[0].J1 = 4.f;
    
    ConstructSparseMatrix(1, Bond, hamil, data, num_elem, 0);
    lanczos(1, num_elem, hamil, groundstate, energies, 200, 3, 1e-10);

    ASSERT_NEAR(energies[0][0], -20.4044129223, 1e-3);
    ASSERT_NEAR(energies[0][1], -20.3063407044, 1e-3);
    ASSERT_NEAR(energies[0][2], -19.6204585904, 1e-3);

    delete [] groundstate[0];
    delete [] energies;

    delete [] energies[0];
    delete [] energies;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
