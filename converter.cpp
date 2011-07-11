void converter( vector< vector<long> > PosHam, vector< vector<h_float> > ValHam,long* num_Elem, long* row_index, long* col_index, cuDoubleComplex* values){

  unsigned long dim = PosHam.size();

  for (int hh = 0; hh < dim; hh++){
    *(num_Elem) += PosHam[hh][0];
  }

  row_index = (long*)malloc(dim*sizeof(long));
  col_index = (long*)malloc(*(num_Elem)*sizeof(long));
  values = (cuDoubleComplex*)malloc(*(num_Elem)*sizeof(cuDoubleComplex));

  for( long ii = 0; ii < dim; ii++){

    long start = 0;

    for ( long jj = 0; jj < ii; jj++){
      start += PosHam[jj][0];
    }

    row_index[ii] = start;

    for ( long kk = 0; kk < PosHam[ii][0]; kk++){

      col_index[start + kk] = PosHam[ii][kk+1];
      values[start + kk] = make_cuDoubleComplex(ValHam[ii][kk], 0);

    }

  }

}
