#include "mem_solver.h"
#include <vector>
#include <chrono>


using namespace std::chrono;


MemSolver::MemSolver(Model* model)
{
    this->model = model;
}

int MemSolver::error_f(const gsl_vector* p, void* data, gsl_vector* res)
{
    auto start_memsolver_step = high_resolution_clock::now();
    size_t n = ((struct data_*)data)->n;
    double* t = ((struct data_*)data)->t;
    double* y = ((struct data_*)data)->y;

    size_t it;

    vector<double> x = model->solve_x();
    vector<double> i = model->solve_i();
    for (it = 0; it < i.size(); it++)
    {
        gsl_vector_set(res, it, i[it] - model->i_array[it]);
    }

    auto stop_memsolver_step = high_resolution_clock::now();
    auto duration = (duration_cast<milliseconds>(stop_memsolver_step - start_memsolver_step)).count();
    if (duration < duration_ms_min)
        duration_ms_min = duration;
    if (duration > duration_ms_max)
        duration_ms_max = duration;
    duration_full += duration;
    memsolver_steps += 1;
    return GSL_SUCCESS;
};

void MemSolver::callback(const size_t iter, void* params, const gsl_multifit_nlinear_workspace* w)
{
    gsl_vector* f = gsl_multifit_nlinear_residual(w);
#ifdef MODE_DEBUG
    cout << iter << '\t' << gsl_blas_dnrm2(f) << endl;
#endif
};

void MemSolver::init_solver()
{
    T = gsl_multifit_nlinear_trust;
    fdf_params = gsl_multifit_nlinear_default_parameters();
    fdf_params.solver = gsl_multifit_nlinear_solver_svd;
    fdf_params.avmax = 0.001;
    fdf_params.factor_up = 1.02;
    fdf_params.factor_down = 1.02;
    fdf_params.scale = gsl_multifit_nlinear_scale_levenberg;
    n = model->t_array.size();
    p = model->get_params().size();
}

double MemSolver::solve(int(*err_f)(const gsl_vector*, void*, gsl_vector*),uint64_t max_iter, double xtol, double gtol, double ftol)
{
    duration_ms_min = 0xFFFFFFFFFFFFFFFF;
    duration_ms_max = 0;
    duration_ms_avg = 0;
    duration_full = 0;
    memsolver_steps = 0;

    gsl_vector* f;
    gsl_matrix *J;
    gsl_matrix *covar = gsl_matrix_alloc (p, p);
    double* t = new double[n];
    double* y = new double[n];

    d = {n, t, y};

    gsl_vector* p_init = gsl_vector_alloc(p);
    vector<double> def_params = model->get_params_mhc();
    for (size_t it = 0; it < p; it++)
    {
        gsl_vector_set(p_init, it, def_params[it]);
    }


    gsl_vector* wts = gsl_vector_alloc(n);
    for(int it = 0; it < n; it++)
    {
        gsl_vector_set(wts, it, 1.);
    }

    gsl_rng *r;
    double chisq, chisq0;
    int status, info;

    gsl_rng_env_setup();
    r = gsl_rng_alloc(gsl_rng_default);

    /* define the function to be minimized */
    fdf.f = err_f;
    fdf.df = NULL;
    fdf.fvv = NULL;
    fdf.n = n;
    fdf.p = p;
    fdf.params = &d;

    /* load data to be fitted */
    for (size_t it = 0; it < n; it++)
    {
        t[it] = model->t_array[it];
        y[it] = model->i_array[it];
    }

    /* allocate workspace with default parameters */
    w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, p);

    /* initialize solver with starting point and weights */
    gsl_multifit_nlinear_winit (p_init, wts, &fdf, w);

    /* compute initial cost function */
    f = gsl_multifit_nlinear_residual(w);
    gsl_blas_ddot(f, f, &chisq0);

    /* solve the system */
    status = gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol, MemSolver::callback, NULL, &info, w);


    /* compute covariance of best fit parameters */
    J = gsl_multifit_nlinear_jac(w);
    gsl_multifit_nlinear_covar (J, 0.0, covar);

    /* compute final cost */
    gsl_blas_ddot(f, f, &chisq);

    double cost = sqrt(chisq);

    cout << "Fitting finished, status: " << gsl_strerror(status) << endl;
    cout<< "Cost drop from " << sqrt(chisq0) << " to " << sqrt(chisq) << endl;
    cout << "Full memsolver steps duration: " << duration_full << " ms." << endl;
    duration_ms_avg = (uint64_t)((float)duration_full / memsolver_steps);
    cout << "Min/Avg/Max duration: " << duration_ms_min << ' ' << duration_ms_avg << ' ' << duration_ms_max << ' ' << "ms." << endl;
    cout << "Total solver steps :" << memsolver_steps << endl;
    cout << "-------" << endl;

    vector<double> params_fitted;
    for (size_t it = 0; it < p; it++)
    {
        params_fitted.push_back(gsl_vector_get(w->x,it));
    }
    model->set_params_mhc(&params_fitted);

    gsl_multifit_nlinear_free (w);
    gsl_matrix_free (covar);
    gsl_rng_free (r);
    gsl_vector_free(p_init);
    gsl_vector_free(wts);
    delete [] t;
    delete [] y;

    return cost;
}
