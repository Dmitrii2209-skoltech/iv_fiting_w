#pragma once
# include <cstdlib>
# include <cstdio>
# include <cmath>
# include <iostream>
# include <fstream>
# include <iomanip>
# include <ctime>
# include <cstring>
#include <vector>

using namespace std;

int ma_in(int N, int option);
void hermite_compute(int order, double xtab[], double weight[]);
void hermite_handle(int order, double alpha, int option, string output);
void hermite_handle(int order, double alpha, int option, vector<double>*, vector<double>*);
void hermite_recur(double* p2, double* dp2, double* p1, double x, int order);
void hermite_root(double* x, int order, double* dp2, double* p1);
double r8_abs(double x);
double r8_epsilon();
double r8_gamma(double x);
double r8_huge();
void r8mat_write(string output_filename, int m, int n, double table[]);
void timestamp();