#ifndef MODEL_H
#define MODEL_H

#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_interp.h>
#include <string>

using namespace std;

void fftw_test(double* x, size_t N);

struct params_bouns_struct
{
	params_bouns_struct(double min, double max)
	{
		this->min = min;
		this->max = max;
	}

	double min;
	double max;
};

class Model
{
protected:
	size_t NPOINTS;
	uint32_t ivp_steps;
public:
	Model();

	map<string, double> params, def_params;
	map<string, params_bouns_struct> params_bounds;

	vector<double> t_array, i_array, v_array, x_array, i_modelled;

	virtual vector<double> solve_x() = 0;
	virtual vector<double> solve_i() = 0;

	void limit_params();
	void limit_params(map<string, double>*);
	void set_params(map<string, double>);
	map<string, double> get_params();
	void set_params_as_vector(vector<double>);
	vector<double> get_params_as_vector();

	void upload_dataset(string);
	void write_results(bool, string);
	void write_results(bool, string, double);

	void set_ivp_multiplier(double);

	virtual double x_der(double, double) = 0;
	virtual vector<double> get_params_mhc();
	virtual void set_params_mhc(vector<double>*);
	virtual void limit_params_mhc(vector<double>*);
	virtual void set_params_from_mhc();
};


class GenModel : public Model
{
private:

	double w_p(double);
	double w_n(double);
	double f_x(double, double);
	double g(double);
	double x(double);

	double alphap = 10.227;
	double alphan = 18.768;
	double xp = 10.0;
	double xn = 0.9;
	double Ap = 10.2;
	double An = 12.5;
	double vp = 4.0;
	double vn = 5.24;
	double a1 = 1.10;
	double a2 = 12.20;
	double b = 5.451;
	double x0 = 0.0;

	double alphap_min = 0;
	double alphan_min = 0;
	double xp_min = 0;
	double xn_min = 0;
	double Ap_min = 0;
	double An_min = 0;
	double vp_min = 0;
	double vn_min = 0;
	double a1_min = 0;
	double a2_min = 0;
	double b_min = 0;
	double x0_min = 0.0;

	double alphap_max = 50;
	double alphan_max = 50;
	double xp_max = 0.999;
	double xn_max = 0.999;
	double Ap_max = 50;
	double An_max = 50;
	double vp_max = 10;
	double vn_max = 10;
	double a1_max = 50;
	double a2_max = 50;
	double b_max = 50;
	double x0_max = 0.0;

public:
	GenModel();

	/*map<string,double> def_params = {{"alphap", 1.0},
								{"alphan", 5.0},
								{"xp", 0.4},
								{"xn", 0.5},
								{"Ap", 7.2},
								{"An", 3.0},
								{"vp", 0.5},
								{"vn", 0.75},
								{"a1", 0.11},
								{"a2", 0.11},
								{"b", 0.5},
								{"x0", 0.1}
	};

	map<string,params_bouns_struct> params_bounds = {{"alphap", params_bouns_struct(0.0,10.0)},
								{"alphan",  params_bouns_struct(0.0,10.0)},
								{"xp",  params_bouns_struct(0.0,10.0)},
								{"xn",  params_bouns_struct(0.0,10.0)},
								{"Ap",  params_bouns_struct(0.0,10.0)},
								{"An",  params_bouns_struct(0.0,10.0)},
								{"vp",  params_bouns_struct(0.0,10.0)},
								{"vn",  params_bouns_struct(0.0,10.0)},
								{"a1",  params_bouns_struct(0.0,10.0)},
								{"a2",  params_bouns_struct(0.0,10.0)},
								{"b",  params_bouns_struct(0.0,10.0)},
								{"x0",  params_bouns_struct(0.0,10.0)}
	};*/

	double x_der(double, double) override;

	vector<double> solve_x() override;
	vector<double> solve_i() override;

	vector<double> get_params_mhc();
	void set_params_mhc(vector<double>*);
	void limit_params_mhc(vector<double>*);
	void set_params_from_mhc();
};

class QDeformed : public Model
{
private:
	/* convolution members*/
	gsl_fft_complex_wavetable* wavetable;
	gsl_fft_complex_workspace* fft_w;
	double* data1;
	double* data2;
	double* data3;
	void gsl_fftconvolve(double*, double*, size_t, double*);
	void set_def_parameters();

protected:
	void fftw_fftconvolve(double*, double*, size_t, double*);

public:
	QDeformed();
	QDeformed(double);

	vector<double> solve_x() override;
	vector<double> solve_i() override;

	double wp(double);
	double wn(double);
	double f_x(double, double);
	double g(double);
	double qexp(double);
	double qsinh(double);
	double x_der(double, double) override;

	void frac_solver(vector<double>*, double, double, size_t);
	void setup_convolution_environment();
	void reset_convolution_environment();
	virtual void final_init_routine();
	virtual void final_reset_routine();

};

class QDeformedMHC : public QDeformed
{
private:
	/* convolution members*/
	gsl_fft_complex_wavetable* wavetable;
	gsl_fft_complex_workspace* fft_w;
	double* data1;
	double* data2;
	double* data3;
	//void gsl_fftconvolve(double*, double*, size_t, double*);
	void set_def_parameters();
	vector<double> pols_x;
	vector<double> pols_w;
	double fermi(double);

	double xp;
	double xn;
	double Ap;
	double An;
	double vp;
	double vn;
	double gamma1;
	double gamma2;
	double delta1;
	double delta2;
	double x0;
	double lambda;
	double A;

	double xp_min;
	double xn_min;
	double Ap_min;
	double An_min;
	double vp_min;
	double vn_min;
	double gamma1_min;
	double gamma2_min;
	double delta1_min;
	double delta2_min;
	double x0_min;
	double lambda_min;
	double A_min;

	double xp_max;
	double xn_max;
	double Ap_max;
	double An_max;
	double vp_max;
	double vn_max;
	double gamma1_max;
	double gamma2_max;
	double delta1_max;
	double delta2_max;
	double x0_max;
	double lambda_max;
	double A_max;

	double wp(double);
	double wn(double);
	double x_der(double, double) override;
	void x_solver(vector<double>*, double, double, size_t);

public:
	QDeformedMHC();

	vector<double> solve_i() override;
	vector<double> solve_x() override;
	double i_MHC(double);
	void setup_pols(size_t);
	void reset_pols();

	vector<double> get_params_mhc();
	void set_params_mhc(vector<double>*);
	void limit_params_mhc(vector<double>*);
	void set_params_from_mhc();
	double f_x(double, double);
	double g(double);
};

class QDeformedMHCFrac : public QDeformed
{
private:
	/* convolution members*/
	gsl_fft_complex_wavetable* wavetable;
	gsl_fft_complex_workspace* fft_w;
	double* data1;
	double* data2;
	double* data3;
	//void gsl_fftconvolve(double*, double*, size_t, double*);
	void set_def_parameters();
	vector<double> pols_x;
	vector<double> pols_w;
	double fermi(double);

	double xp;
	double xn;
	double Ap;
	double An;
	double vp;
	double vn;
	double gamma1;
	double gamma2;
	double delta1;
	double delta2;
	double x0;
	double lambda;
	double A;
	double alpha;

	double xp_min;
	double xn_min;
	double Ap_min;
	double An_min;
	double vp_min;
	double vn_min;
	double gamma1_min;
	double gamma2_min;
	double delta1_min;
	double delta2_min;
	double x0_min;
	double lambda_min;
	double A_min;
	double alpha_min;

	double xp_max;
	double xn_max;
	double Ap_max;
	double An_max;
	double vp_max;
	double vn_max;
	double gamma1_max;
	double gamma2_max;
	double delta1_max;
	double delta2_max;
	double x0_max;
	double lambda_max;
	double A_max;
	double alpha_max;

	double wp(double);
	double wn(double);
	double f_x(double, double);
	double g(double);
	double x_der(double, double) override;
	void frac_solver_init(double, size_t);
	void frac_solver(vector<double>*, double, double);
	void frac_solver_reset();


	// frac_solver_members:
	double h_fs, ha, ga1, ga2;
	double* t_fs;
	double* x_fs;
	double* k_fs;
	double* a_fs;
	double* b_fs;
	double* t_orig;
	double* v_orig;
	double* f_fs;
	double* conva;
	double* convb;
	double* v_interp;
	gsl_interp* intrpl1;
	gsl_interp* intrpl2;
	gsl_interp_accel* accel1;
	gsl_interp_accel* accel2;

public:
	QDeformedMHCFrac();
	QDeformedMHCFrac(double);

	void final_init_routine() override;
	void final_reset_routine() override;

	vector<double> solve_i() override;
	vector<double> solve_x() override;
	double i_MHC(double);
	void setup_pols(size_t);
	void reset_pols();

	vector<double> get_params_mhc();
	void set_params_mhc(vector<double>*);
	void limit_params_mhc(vector<double>*);
	void set_params_from_mhc();
};

#endif // MODEL_H
