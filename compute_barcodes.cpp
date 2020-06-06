#include "boost/multi_array.hpp"
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <tuple>
#include <vector>

using namespace std;
using namespace boost;

template <class T> void printmat(multi_array<T, 2> a, multi_array<T, 2> b) {
  for (int i = 0; i < a.shape()[0]; ++i) {
    for (int j = 0; j < a.shape()[1]; ++j) {
      if (a[i][j] == 1) {
        cout << "t^" << b[i][j] << "\t";
      } else {
        cout << "0  \t";
      }
    }
    cout << endl;
  }
  cout << endl;
}

double dot(const vector<double>& v1, const vector<double>& v2, int n) {
  double t = 0;
  for (int k = 0; k < n; ++k) {
    t += v1[k] * v2[k];
  }
  return t;
}

double norm(const vector<double>& v1, int n) { return pow(dot(v1, v1, n), 0.5); }

double cosine_dist(const vector<double>& v1, const vector<double>& v2, int n) {
  return 1.0-(dot(v1, v2, n) / norm(v1, n) / norm(v2, n));
}

void get_barcodes(const vector<vector<double>> &data, vector<double> &barcodes,
                  int num_threads) {

  int N = data.size();
  int dim = 0;
  if (N > 0)
    dim = data[0].size();

  // create VR filtration
  const int E = N * (N - 1) / 2; // number edges
  vector<tuple<double, int, int>> edges(E);
  int count = 0;
  for (int i = 0; i < N; ++i) {

//#pragma omp parallel num_threads(6)
{
//#pragma omp for

    for (int j = i + 1; j < N; ++j) {
      edges[count] = make_tuple(cosine_dist(data[i], data[j], dim), i, j);
      count += 1;
    }
}

  }

  sort(edges.begin(), edges.end(),
       [](tuple<double, int, int> t1, tuple<double, int, int> t2) {
         return get<0>(t1) < get<0>(t2);
       });

  // create matrix
  multi_array<int, 2> bdd1{extents[N][E]};
  multi_array<int, 2> bdd2{extents[N][E]};

  // populate matrix
//#pragma omp parallel num_threads(6)
{
//#pragma omp for
  for (int e = 0; e < E; ++e) {
    double d;
    int i, j;
    tie(d, i, j) = edges[e];

    int deg = e + 1;
    bdd1[i][e] = 1;
    bdd2[i][e] = deg;

    bdd1[j][e] = 1;
    bdd2[j][e] = deg;
  }
}

  // reduce matrix
  int row_offset = 0;
  for (int c = 0; c < min(E - 1, N - 1) + row_offset; ++c) {
    bool skip = false;

    // check if need to reduce column
    while (skip == false && row_offset < N - c) {
      int numZero = 0;
      for (int i = 0; i < E - c; ++i) {
        if (bdd1[c + row_offset][c + i] == 0) {
          ++numZero;
        }
      }
      if (numZero == E - c) {
        ++row_offset;
      } else {
        skip = true;
      }
    }

    if (skip == true) {
      bool skip2 = false;
      if (bdd1[c + row_offset][c] == 0) { // pivot column
        int c2 = c;
        while (c2 < E && bdd1[c + row_offset][c2] == 0) {
          ++c2;
        }
        if (c2 >= E) {
          skip2 = true;
        } else {
//#pragma omp parallel num_threads(6)
{
//#pragma omp for
          for (int i = 0; i < N; ++i) { // swap columns
            int t;

            t = bdd1[i][c];
            bdd1[i][c] = bdd1[i][c2];
            bdd1[i][c2] = t;

            t = bdd2[i][c];
            bdd2[i][c] = bdd2[i][c2];
            bdd2[i][c2] = t;
          }
}
        }
      }

      if (skip2 == false) {
        assert(bdd1[c + row_offset][c] == 1); // check we pivotted
//#pragma omp parallel num_threads(6)
{
//#pragma omp for
        for (int c2 = c + 1; c2 < E; ++c2) {
          if (bdd1[c + row_offset][c2] != 0) {

            int deg_diff = bdd2[c + row_offset][c2] - bdd2[c + row_offset][c];

            for (int i = 0; i < N - c - row_offset; ++i) {
              if (bdd1[c + i + row_offset][c] != 0) {
                if (deg_diff + bdd2[c + i + row_offset][c] ==
                    bdd2[c + i + row_offset][c2]) {
                  if (bdd1[c + i + row_offset][c2] != 0) {
                    bdd1[c + i + row_offset][c2] = 0;
                  } else {
                    bdd1[c + i + row_offset][c2] = 1;
                  }
                } else if (deg_diff + bdd2[c + i + row_offset][c] >
                           bdd2[c + i + row_offset][c2]) {
                  bdd1[c + i + row_offset][c2] = 1;
                  bdd2[c + i + row_offset][c2] =
                      deg_diff + bdd2[c + i + row_offset][c];
                }
              }
            }
          }
        }
}
      }
    }
  }

  // retrieve barcodes

  barcodes.resize(min(N, E));
  for (int i = 0; i < min(N, E); ++i) {
    if (i + row_offset >= N) {
      break;
    }

    int row_offset = 0;
    bool skip = false;
    while (skip == false && row_offset < N - i) {
      if (bdd1[i + row_offset][i] == 0) {
        ++row_offset;
      } else {
        skip = true;
      }
    }

    if (row_offset >= N - i) {
      barcodes[i] = std::numeric_limits<double>::infinity();
    } else {
      barcodes[i] = get<0>(edges[bdd2[i + row_offset][i] - 1]);
    }
  }
}

int main(int argc, char **argv) {

  // problem parameters
  const string instance = argv[1];
  const int max_num_threads = 1;    // range of threads to run
  const int number_experiments = 1; // experiments per thread limit

  // read data
  vector<vector<double>> data;
  string line;
  ifstream file("temp.txt."+instance);
  if (file.is_open()) {
    int i = 0;
    while (getline(file, line)) {
      data.push_back(vector<double>());
      stringstream line_stream(line);
      string token;
      while (getline(line_stream, token, ' ')) {
        data[i].push_back(stod(token));
      }
      ++i;
    }
    file.close();
  } else {
    cout << "Unable to open file: "<< "temp.txt."+instance <<"\n";
    return 0;
  }

  for (int num_threads = 1; num_threads <= max_num_threads; ++num_threads) {
    for (int j = 0; j < number_experiments; ++j) {

      vector<double> barcodes;
      get_barcodes(data, barcodes, num_threads);

      for (int i = 0; i < barcodes.size(); ++i) {
        cout << barcodes[i] << " ";
      }
      cout << endl;
    }
  }
  return 0;
}
