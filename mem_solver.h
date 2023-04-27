#ifndef MEM_SOLVER_H
#define MEM_SOLVER_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

#include "models.h"

using namespace std;

struct data_ {
    size_t n;
    double * t;
    double * y;
  };


class MemSolver
{
public:
    MemSolver(Model*);
    Model* model;

    void init_solver();

    int error_f(const gsl_vector*, void*, gsl_vector*);

    static void callback(const size_t, void*, const gsl_multifit_nlinear_workspace*);

    double solve(int(*)(const gsl_vector*, void*, gsl_vector*),uint64_t, double, double, double);

private:

    size_t n;
    size_t p;

    data_ d;

    gsl_multifit_nlinear_workspace* w;
    gsl_multifit_nlinear_fdf fdf;
    const gsl_multifit_nlinear_type* T;
    gsl_multifit_nlinear_parameters fdf_params;

    uint64_t duration_ms_min, duration_ms_max, duration_ms_avg, memsolver_steps, duration_full;
};

#endif // MEM_SOLVER_H
