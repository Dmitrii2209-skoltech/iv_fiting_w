#include <iostream>
#include <cstring>
#include "models.h"
#include "mem_solver.h"
#include <random>
#include <chrono>
/************************************/
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])
/**************************************/

using namespace std;
using namespace std::chrono;

QDeformedMHCFrac qdefm;
MemSolver solver(&qdefm);
uint64_t MAX_STEPS = 50;

/***************************************************/
void conv_test(double* x1, double* x2, size_t N, double* y)
{
    size_t convolve_size = (N+1) + N;
    /* here fft is set up */
    gsl_fft_complex_wavetable* wavetable = gsl_fft_complex_wavetable_alloc(convolve_size);
    gsl_fft_complex_workspace* fft_w = gsl_fft_complex_workspace_alloc(convolve_size);


    double* data1 = new double [convolve_size * 2];
    double* data2 = new double [convolve_size * 2];
    double* data3 = new double [convolve_size * 2];
    //double* res = new double [convolve_size * 2];





    for (int j = 0; j < convolve_size; j++)
    {
        if (j < N+1)
        {
            REAL(data1,j) = x1[j];
            IMAG(data1,j) = 0;
            REAL(data2,j) = x2[j];
            IMAG(data2,j) = 0;
        }
        else
        {
            REAL(data1,j) = 0;
            IMAG(data1,j) = 0;
            REAL(data2,j) = 0;
            IMAG(data2,j) = 0;
        }

    }


    /* slow convolution *
    for (int i = 0; i < convolve_size; i++)
    {
        REAL(res,i) = 0;
        IMAG(res,i) = 0;
    }
    for (int i = 0; i < convolve_size; i++)
    {
        size_t sum_n = i + 1;
        size_t start_point = i;
        size_t offset = 0;
        if (sum_n > N+1)
            sum_n = N+1;
        if (i >= N+1)
        {
            start_point = N;
            offset= i - N;
        }
        for (int j = 0; j < sum_n - offset; j++)
            REAL(res,i) += REAL(data1, start_point + j)*REAL(data2,j+offset);
    }
    ********************/

    cout << "Arr1:\t";
    for (int i = 0; i < convolve_size-1; i++)
    {
        cout << REAL(data1, i) << '\t';
    }
    cout << REAL(data1,convolve_size) << endl;

    cout << "Arr2:\t";
    for (int i = 0; i < convolve_size-1; i++)
    {
        cout << REAL(data2, i) << '\t';
    }
    cout << REAL(data2,convolve_size) << endl;


    gsl_fft_complex_forward(data1,1,convolve_size,wavetable,fft_w);
    gsl_fft_complex_forward(data2,1,convolve_size,wavetable,fft_w);
    for (int j = 0; j < convolve_size; j++)
    {
        REAL(data3, j) = REAL(data1,j)*REAL(data2,j) - IMAG(data1,j)*IMAG(data2,j);
        IMAG(data3, j) = REAL(data1,j)*IMAG(data2,j) + IMAG(data1,j)*REAL(data2,j);
    }
    gsl_fft_complex_inverse(data3,1,convolve_size,wavetable,fft_w);
    for (int j = 0; j < convolve_size; j++)
    {
        y[j] = REAL(data3,j);
    }

    /*cout << "Conv:\t";
    for (int i = 0; i < convolve_size-1; i++)
    {
        cout << REAL(res, i) << '\t';
    }
    cout << REAL(res,convolve_size-1) << endl;*/

    cout << "FFTC R:\t";
    for (int i = 0; i < convolve_size - 1; i++)
    {
        cout << REAL(data3, i) << '\t';
    }
    cout << REAL(data3,convolve_size) << endl;

    cout << "FFTC Im:\t";
    for (int i = 0; i < convolve_size - 1; i++)
    {
        cout << IMAG(data3, i) << '\t';
    }
    cout << IMAG(data3,convolve_size) << endl;

    /*for (int i = 0; i < N*2 + 2; i++)
    {
        res[i] = abs(res[i] - data3[i]);
    }

    cout << "##############################" << endl;
    cout << "Error R:\t";
    for (int i = 0; i < N; i++)
    {
        cout << REAL(res, i) << '\t';
    }
    cout << REAL(res,N) << endl;

    cout << "Error Im:\t";
    for (int i = 0; i < N; i++)
    {
        cout << IMAG(res, i) << '\t';
    }
    cout << IMAG(res,N) << endl;*/

    delete [] data1;
    delete [] data2;
    delete [] data3;
    //delete [] res;

    gsl_fft_complex_wavetable_free(wavetable);
    gsl_fft_complex_workspace_free(fft_w);
}
/*************************************************/

int residual_wrapper(const gsl_vector* p, void* data, gsl_vector* res)
{
    vector<double> new_params;
    for (size_t i = 0; i < p->size; i++)
    {
        new_params.push_back(gsl_vector_get(p,i));
    }
    qdefm.set_params_mhc(&new_params);
    new_params = qdefm.get_params_mhc();



    gsl_vector* new_vec = gsl_vector_alloc(p->size);
    for (size_t i = 0; i < p->size; i++)
    {
        gsl_vector_set(new_vec,i,new_params[i]);
    }
    memcpy(p->data,new_vec->data,p->size);
    gsl_vector_free(new_vec);

    return solver.error_f(p, data, res);
}


vector<double> select_params(vector<vector<double>> best_params_array, vector<double> costs_array)
{
    vector <double> sum_array;
    vector<double> weights, weights_cum;
    for (int i = 0; i < costs_array.size(); i++)
    {
        double sum = 0;
        double weight;
        for (int k = 0; k < best_params_array[i].size(); k++)
        {
            sum += best_params_array[i].at(k);
        }
        sum_array.push_back(sum);
        weight = 1./costs_array[i] + 1./sum;
        weights.push_back(weight);
    }
    double weights_sum = 0;
    for (int i = 0; i < costs_array.size(); i++)
    {
        weights_sum += weights[i];
    }
    for (int i = 0; i < costs_array.size(); i++)
    {
        weights[i] = weights[i]/weights_sum;
    }
    for (int i = 0; i < weights.size(); i++)
    {
        double sum = 0;
        for (int k = 0; k <= i; k++)
        {
            sum += weights[k];
        }
        weights_cum.push_back(sum);
    }

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double X = dis(gen);
    size_t index = 0;
    while (weights_cum[index] < X)
    {
        index++;
    }
    return best_params_array[index];
}

void randomize(Model* model,vector<double>* curr_params)
{
    for (int i = 0; i < curr_params->size(); i++)
    {
        double s = sqrt(abs(curr_params->at(i))) + 1;
        random_device rd;  // Will be used to obtain a seed for the random number engine
        mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        normal_distribution<> dis(0.0, s);
        curr_params->at(i) += dis(gen);
    }
    model->limit_params_mhc(curr_params);
}

void apply_random_step(Model* model, vector<double>* curr_params)
{
    for (int i = 0; i < curr_params->size(); i++)
    {
        double d = sqrt(abs(curr_params->at(i))+1)/4.;
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-d, d);
        curr_params->at(i) += dis(gen);
    }
    model->limit_params_mhc(curr_params);
}

double roland_fitting(Model* model, size_t nruns, double start_temp, double end_temp, double alpha_temp)
{
    double temp = start_temp;
    size_t counter = 0;
    while (temp > end_temp)
    {
        counter++;
        temp *= alpha_temp;
    }

    cout << endl << "################ START ROLAND FITTING PROCEDURE ################" << endl;
    cout << "######### Expected " << counter << " fitting procedures per run" << endl;
    vector<vector<double>> best_params_array;
    vector<double> costs_array;

    costs_array.push_back(solver.solve(residual_wrapper,MAX_STEPS,1e-8, 1e-8, 1e-8));
    best_params_array.push_back(model->get_params_mhc());
    cout << endl;

    model->solve_x();
    model->solve_i();

    string filename = "out_";
    char ch[4];
    snprintf(ch, 4, "%03d", 0);
    filename += ch;
    model->set_params_from_mhc();
    model->write_results(true, filename, costs_array.back());

    vector<double> curr_params;
    for (int run = 1; run < nruns+1; run++)
    {
        auto start_run = high_resolution_clock::now();
        cout << "####### Run # " << run << endl;
        curr_params = select_params(best_params_array, costs_array);
        randomize(model, &curr_params);
        model->set_params_mhc(&curr_params);
        double curr_min_cost = numeric_limits<double>::infinity();
        vector<double> curr_best_params;
        temp = start_temp;
        while (temp > end_temp)
        {
            auto start_1iter = high_resolution_clock::now();
            double cost = solver.solve(residual_wrapper,MAX_STEPS,1e-8, 1e-8, 1e-8);

            vector<double> params = model->get_params_mhc();
            if (cost < curr_min_cost)
            {
                curr_min_cost = cost;
                curr_best_params = params;
            }
            else if (cost == cost)
            {
                random_device rd;  // Will be used to obtain a seed for the random number engine
                mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
                uniform_real_distribution<> dis(0.0, 1.0);
                double X = dis(gen);
                if (X < exp(-(cost - curr_min_cost)/temp))
                {

                    curr_min_cost = cost;
                    curr_best_params = params;
                }
            }
            apply_random_step(model, &curr_params);
            model->set_params_mhc(&curr_params);
            //cout << "### Temp was: " << temp << ", Min cost achived: " << curr_min_cost << endl;
            temp *= alpha_temp;
            auto stop_1iter = high_resolution_clock::now();
            uint64_t while_iter_duration = (duration_cast<milliseconds>(stop_1iter - start_1iter)).count();
            cout << "While step duration: " << while_iter_duration << " ms." << endl;
        }
        best_params_array.push_back(curr_best_params);
        costs_array.push_back(curr_min_cost);
        model->set_params_mhc(&curr_best_params);
        model->solve_x();
        model->solve_i();
        string filename = "out_";
        char ch[4];
        snprintf(ch, 4, "%03d", run);
        filename += ch;
        model->set_params_from_mhc();
        model->write_results(true, filename, costs_array.back());
        auto stop_run = high_resolution_clock::now();
        uint64_t run_duration = (duration_cast<milliseconds>(stop_run - start_run)).count();

        cout << "####### Run # " << run << " min cost: " << curr_min_cost << endl;
        cout << "####### Run duration: " << run_duration << " ms." << endl;
        cout << endl;
    }

    cout << "################ STOP ROLAND FITTING PROCEDURE ################" << endl;


    size_t index = 0;
    for (int i = 1; i < costs_array.size(); i++)
    {
        if (costs_array[i] < costs_array[index])
            index = i;
    }
    cout << "Costs | Best params array: " << endl;
    cout << "run cost " << endl;
    map<string, double> curr_p = model->get_params();
    auto it = curr_p.begin();
    for (int i = 0; i < curr_p.size(); i++)
    {
        cout << it->first << " ";
        it++;
    }
    cout << endl;
    for (int i = 0; i < costs_array.size(); i++)
    {
        cout << i << " ";
        model->set_params_mhc(&best_params_array[i]);
        model->set_params_from_mhc();
        curr_p = model->get_params();
        cout << costs_array[i] << ' ';
        it = curr_p.begin();
        for (int i = 0; i < curr_p.size(); i++)
        {
            cout << it->second << " ";
            it++;
        }
        cout << endl;
    }
    cout << endl;
    model->set_params_mhc(&best_params_array[index]);
    model->set_params_from_mhc();

    return costs_array[index];
}

int main()
{
    /*double a[] = {1., 2.1, 3, 4, 5, 6};
    fftw_test(a, 6);
    return 0;*/


    qdefm = QDeformedMHCFrac();
    qdefm.setup_pols(25);
    qdefm.upload_dataset("roland_full.txt");
    qdefm.set_ivp_multiplier(1.0);
    qdefm.setup_convolution_environment();
    qdefm.final_init_routine();

    qdefm.solve_x();
    qdefm.solve_i();

    qdefm.write_results(true, "initial");

    solver.init_solver();
    solver.solve(residual_wrapper,MAX_STEPS,1e-8, 1e-8, 1e-8);
    cout << endl;
    qdefm.write_results(true, "init_solve");

    double final_cost = roland_fitting(&qdefm, 20, 10, 0.5, 0.5);

    qdefm.solve_x();
    qdefm.solve_i();

    cout << endl << "*****************************" << endl << endl;
    cout << "Final cost: " << final_cost << endl;
    cout << "Final parameters set:" << endl;
    map <string,double> f_par = qdefm.get_params();
    auto it = f_par.begin();
    for (int i = 0; i < f_par.size(); i++)
    {
        cout << it->first << ": " << it->second << ' ' << endl;
        it++;
    }
    cout << endl;
    qdefm.write_results(true, "out", final_cost);
    qdefm.reset_convolution_environment();
    qdefm.final_reset_routine();

    return 0;
}

