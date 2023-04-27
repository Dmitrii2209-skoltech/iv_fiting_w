#include "models.h"
#include <fstream>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <iostream>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_fft_complex.h>
#include "fft.h"
#include <chrono>
#include <fftw3.h>
#include <complex>
#include "hermite_rule_ss.h"


using namespace std::chrono;

#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;

uint64_t fft_duration;

using namespace std;

struct ivp_params
{
	Model* model_pointer;
	double* volt;
};


int x_ivp(double t, const double x[], double x_der[], void* params)
{
	(void)(t);
	ivp_params* p = (ivp_params*)params;
	x_der[0] = p->model_pointer->x_der(x[0], *p->volt);

	return GSL_SUCCESS;
}

void fftw_test(double* x, size_t N)
{
	double* gsl_in = new double[N * 2];
	double* gsl_out = new double[N * 2];
	fftw_complex* in, * out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

	for (int i = 0; i < N; i++)
	{
		REAL(gsl_in, i) = x[i];
		IMAG(gsl_in, i) = 0;
		in[i][0] = x[i];
		in[i][1] = 0;
		cout << in[i][0] << ' ';
	}
	cout << endl;

	fftw_plan p1, p2;
	p1 = fftw_plan_dft_r2c_1d(N, x, in, FFTW_ESTIMATE);
	fftw_execute(p1);
	fftw_destroy_plan(p1);

	for (int i = 0; i < N; i++)
	{
		cout << in[i][0] << ' ';
	}
	cout << endl;
	p2 = fftw_plan_dft_1d(N, in, in, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p2);
	fftw_destroy_plan(p2);

	for (int i = 0; i < N; i++)
	{
		cout << in[i][0] << ' ';
	}
	cout << endl;

	fftw_free(in);
	fftw_free(out);
	delete[] gsl_in;
	delete[] gsl_out;
}

void QDeformed::gsl_fftconvolve(double* x1, double* x2, size_t N, double* y)
{

	size_t convolve_size = (N + 1) + N;
	for (int j = 0; j < N + 1; j++)
	{
		REAL(data1, j) = x1[j];
		REAL(data2, j) = x2[j];
		IMAG(data1, j) = 0;
		IMAG(data2, j) = 0;
	}
	for (int j = N + 1; j < convolve_size; j++)
	{
		REAL(data1, j) = 0;
		REAL(data2, j) = 0;
		IMAG(data1, j) = 0;
		IMAG(data2, j) = 0;
	}
	auto start = high_resolution_clock::now();
	gsl_fft_complex_forward(data1, 1, convolve_size, wavetable, fft_w);
	gsl_fft_complex_forward(data2, 1, convolve_size, wavetable, fft_w);

	for (int i = 0; i < convolve_size; i++)
	{
		REAL(data3, i) = (REAL(data1, i) * REAL(data2, i) - IMAG(data1, i) * IMAG(data2, i));
		IMAG(data3, i) = (REAL(data1, i) * IMAG(data2, i) + IMAG(data1, i) * REAL(data2, i));
	}

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	fft_duration += duration.count();
	start = high_resolution_clock::now();
	gsl_fft_complex_inverse(data3, 1, convolve_size, wavetable, fft_w);
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	fft_duration += duration.count();
	for (int j = 0; j < convolve_size; j++)
	{
		y[j] = REAL(data3, j);
	}
}

void QDeformed::fftw_fftconvolve(double* x1, double* x2, size_t N, double* y)
{
	size_t convolve_size = (N + 1) + N;
	fftw_complex* in1, * in2, * out1, * out2;
	in1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * convolve_size);
	in2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * convolve_size);
	out1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * convolve_size);
	out2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * convolve_size);
	for (int j = 0; j < N + 1; j++)
	{
		REAL(data1, j) = x1[j];
		REAL(data2, j) = x2[j];
		IMAG(data1, j) = 0;
		IMAG(data2, j) = 0;
	}
	for (int j = N + 1; j < convolve_size; j++)
	{
		REAL(data1, j) = 0;
		REAL(data2, j) = 0;
		IMAG(data1, j) = 0;
		IMAG(data2, j) = 0;
	}
	auto start = high_resolution_clock::now();
	for (int i = 0; i < convolve_size; i++)
	{
		in1[i][0] = REAL(data1, i);
		in1[i][1] = IMAG(data1, i);
	}
	for (int i = 0; i < convolve_size; i++)
	{
		in2[i][0] = REAL(data2, i);
		in2[i][1] = IMAG(data2, i);
	}
	fftw_plan p1, p2, p3;
	p1 = fftw_plan_dft_1d(convolve_size, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p1);
	fftw_destroy_plan(p1);

	p2 = fftw_plan_dft_1d(convolve_size, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p2);
	fftw_destroy_plan(p2);
	for (int i = 0; i < convolve_size; i++)
	{
		in1[i][0] = (out1[i][0] * out2[i][0] - out1[i][1] * out2[i][1]) / convolve_size / convolve_size;
		in1[i][1] = (out1[i][0] * out2[i][1] + out1[i][1] * out2[i][0]) / convolve_size / convolve_size;
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	fft_duration += duration.count();
	start = high_resolution_clock::now();
	p3 = fftw_plan_dft_1d(convolve_size, in1, in1, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p3);
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	fft_duration += duration.count();
	for (int i = 0; i < convolve_size; i++)
	{
		REAL(data3, i) = in1[i][0] * convolve_size;
		IMAG(data3, i) = in1[i][1] * convolve_size;
	}
	for (int j = 0; j < convolve_size; j++)
	{
		y[j] = REAL(data3, j);
	}
	fftw_destroy_plan(p3);
	fftw_free(in1);
	fftw_free(in2);
	fftw_free(out1);
	fftw_free(out2);
}

Model::Model()
{

}

void Model::limit_params()
{
	auto it1 = this->params.begin();
	/*auto it2 = this->params_bounds.begin();
	for (int i = 0; i < this->params.size(); i++)
	{
		if (it1->second < it2->second.min)
			it1->second = it2->second.min;
		else if (it1->second > it2->second.max)
			it1->second = it2->second.max;
		it1++;
		it2++;
	}*/
	for (int i = 0; i < this->params.size(); i++)
	{
		if (it1->second < this->params_bounds.at(it1->first).min)
			it1->second = this->params_bounds.at(it1->first).min;
		else if (it1->second > this->params_bounds.at(it1->first).max)
			it1->second = this->params_bounds.at(it1->first).max;
		it1++;
	}
}

void Model::limit_params(map<string, double>* ext_params)
{
	auto it1 = ext_params->begin();
	for (int i = 0; i < ext_params->size(); i++)
	{
		if (it1->second < this->params_bounds.at(it1->first).min)
			it1->second = this->params_bounds.at(it1->first).min;
		else if (it1->second > this->params_bounds.at(it1->first).max)
			it1->second = this->params_bounds.at(it1->first).max;
		it1++;
	}
}

void Model::set_params(map<string, double> params)
{
	this->params = params;
	limit_params();
};
void Model::set_params_as_vector(vector<double> params)
{
	uint8_t p_size = this->params.size();
	auto it = this->params.begin();
	for (int i = 0; i < p_size; i++)
	{
		it->second = params[i];
		it++;
	}
	limit_params();
};

void Model::set_ivp_multiplier(double mult)
{
	this->ivp_steps = floor(NPOINTS * mult);
}


map<string, double> Model::get_params() { return this->params; };
vector<double> Model::get_params_as_vector()
{
	vector<double> v;
	for (auto it = this->params.begin(); it != this->params.end(); it++)
	{
		v.push_back(it->second);
	}
	return v;
};

void Model::upload_dataset(string filename)
{
	this->t_array.clear();
	this->v_array.clear();
	this->i_array.clear();

	ifstream is;
	is.open(filename);

	double counter = 0;
	while (!is.eof())
	{
		double t, v, i;
		//t = counter * 0.001;
		is >> t >> v >> i;
		if (!t_array.empty() && t == t_array.back())
		{
			continue;
		}
		t_array.push_back(t);
		v_array.push_back(v);
		i_array.push_back(i);
		counter++;
	}
	is.close();

	this->NPOINTS = t_array.size();
	this->ivp_steps = this->NPOINTS * 10;
}

void Model::write_results(bool include_origin_data, string filename)
{
	ofstream out;

	out.open(filename + ".txt");
	out << "TIMESTAMP" << '\t' << "VOLTAGE" << '\t' << "X" << '\t' << "MODEL_CURRENT";
	if (include_origin_data)
		out << '\t' << "EXP_CURRENT" << endl;
	else
		out << endl;
	for (size_t i = 0; i < NPOINTS; i++)
	{
		out << t_array[i] << '\t' << v_array[i] << '\t' << x_array[i] << '\t' << i_modelled[i];
		if (include_origin_data)
			out << '\t' << i_array[i] << endl;
		else
			out << endl;
	}
	out.close();
	return;
}

void Model::write_results(bool include_origin_data, string filename, double cost)
{
	ofstream out;

	out.open(filename + ".txt");
	out << "TIMESTAMP" << '\t' << "VOLTAGE" << '\t' << "X" << '\t' << "MODEL_CURRENT";
	if (include_origin_data)
		out << '\t' << "EXP_CURRENT" << endl;
	else
		out << endl;
	for (size_t i = 0; i < NPOINTS; i++)
	{
		out << t_array[i] << '\t' << v_array[i] << '\t' << x_array[i] << '\t' << i_modelled[i];
		if (include_origin_data)
			out << '\t' << i_array[i] << endl;
		else
			out << endl;
	}
	out.close();

	out.open(filename + "_params.txt");
	out << "Cost: " << cost << " Params: " << endl;
	auto it = this->params.begin();
	for (int i = 0; i < params.size(); i++)
	{
		out << it->first << ": " << it->second << ' ';
		it++;
	}
	out << endl;
	out.close();
	return;
}

vector<double> Model::get_params_mhc()
{
	vector<double> pp;
	cout << "Error, interface class is reffered" << endl;
	return pp;
};

void Model::set_params_mhc(vector<double>* pp)
{
	cout << "Error, interface class is reffered" << endl;
};

void Model::limit_params_mhc(vector<double>* pp)
{
	cout << "Error, interface class is reffered" << endl;
};

void Model::set_params_from_mhc()
{
	cout << "Error, interface class is reffered" << endl;
};


GenModel::GenModel()
{
	this->params = { {"alphap", 10.227},
					{"alphan", 18.768},
					{"xp", 10.0},
					{"xn", 0.9},
					{"Ap", 10.2},
					{"An", 12.5},
					{"vp", 4.0},
					{"vn", 5.24},
					{"a1", 1.10},
					{"a2", 12.20},
					{"b", 5.451},
					{"x0", 0.0}
	};

	this->params_bounds = { {"alphap", params_bouns_struct(0.0,50.0)},
						   {"alphan",  params_bouns_struct(0.0,50.0)},
						   {"xp",  params_bouns_struct(0.0,1.0)},
						   {"xn",  params_bouns_struct(0.0,1.0)},
						   {"Ap",  params_bouns_struct(0.0,50.0)},
						   {"An",  params_bouns_struct(0.0,50.0)},
						   {"vp",  params_bouns_struct(0.0,10.0)},
						   {"vn",  params_bouns_struct(0.0,10.0)},
						   {"a1",  params_bouns_struct(0.0,50.0)},
						   {"a2",  params_bouns_struct(0.0,50.0)},
						   {"b",  params_bouns_struct(0.0,50.0)},
						   {"x0",  params_bouns_struct(0.0,0.0)}
	};


	this->params = { {"alphap", 10.227},
					{"alphan", 18.768},
					{"xp", 10.0},
					{"xn", 0.9},
					{"Ap", 10.2},
					{"An", 12.5},
					{"vp", 4.0},
					{"vn", 5.24},
					{"a1", 1.10},
					{"a2", 12.20},
					{"b", 5.451},
					{"x0", 0.0}
	};

	this->params_bounds = { {"alphap", params_bouns_struct(0.0,50.0)},
						   {"alphan",  params_bouns_struct(0.0,50.0)},
						   {"xp",  params_bouns_struct(0.0,1.0)},
						   {"xn",  params_bouns_struct(0.0,1.0)},
						   {"Ap",  params_bouns_struct(0.0,50.0)},
						   {"An",  params_bouns_struct(0.0,50.0)},
						   {"vp",  params_bouns_struct(0.0,10.0)},
						   {"vn",  params_bouns_struct(0.0,10.0)},
						   {"a1",  params_bouns_struct(0.0,50.0)},
						   {"a2",  params_bouns_struct(0.0,50.0)},
						   {"b",  params_bouns_struct(0.0,50.0)},
						   {"x0",  params_bouns_struct(0.0,0.0)}
	};
};


vector<double> GenModel::solve_x()
{
	this->x_array.clear();
	double* t_integr = new double[this->ivp_steps + 1];
	double* x_integr = new double[this->ivp_steps + 1];
	double* v_integr = new double[this->ivp_steps + 1];
	double* t_orig = new double[this->NPOINTS];
	double* v_orig = new double[this->NPOINTS];

	for (int it = 0; it < NPOINTS; it++)
	{
		t_orig[it] = this->t_array[it];
		v_orig[it] = this->v_array[it];
	}

	gsl_interp* intrp1 = gsl_interp_alloc(gsl_interp_linear, NPOINTS);
	gsl_interp_init(intrp1, t_orig, v_orig, NPOINTS);
	gsl_interp_accel* intrp_acc1 = gsl_interp_accel_alloc();
	for (int i = 0; i <= this->ivp_steps; i++)
	{
		v_integr[i] = gsl_interp_eval(intrp1, t_orig, v_orig, i * this->t_array.back() / (double)this->ivp_steps, intrp_acc1);
	}
	gsl_interp_free(intrp1);
	gsl_interp_accel_free(intrp_acc1);

	double voltage;
	volatile ivp_params par;
	par.model_pointer = this;
	par.volt = &voltage;

	gsl_odeiv2_system sys = { x_ivp, NULL,1,(void*)&par };
	gsl_odeiv2_step* stepper = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk2, 1);
	double t = 0;
	double x[1] = { this->x0 };
	t_integr[0] = 0;
	x_integr[0] = x[0];
	voltage = v_integr[0];
	double odeerror;
	double h = this->t_array.back() / (double)this->ivp_steps;
	for (int i = 0; i <= this->ivp_steps; i++)
	{
		voltage = v_integr[i];
		gsl_odeiv2_step_reset(stepper);
		double ti = i * h;
		int status = gsl_odeiv2_step_apply(stepper, ti, h, x, &odeerror, NULL, NULL, &sys);
		if (status != GSL_SUCCESS)
		{
			cout << "error, return value=" << status << endl;
			break;
		}
		t_integr[i] = ti;
		x_integr[i] = x[0];
	}
	gsl_odeiv2_step_free(stepper);

	gsl_interp* intrp2 = gsl_interp_alloc(gsl_interp_linear, this->ivp_steps + 1);
	gsl_interp_init(intrp2, t_integr, x_integr, this->ivp_steps + 1);
	gsl_interp_accel* intrp_acc2 = gsl_interp_accel_alloc();
	for (int i = 0; i < this->t_array.size(); i++)
	{
		this->x_array.push_back(gsl_interp_eval(intrp2, t_integr, x_integr, this->t_array[i], intrp_acc2));
	}

	gsl_interp_free(intrp2);
	gsl_interp_accel_free(intrp_acc2);

	delete[] t_integr;
	delete[] x_integr;
	delete[] v_integr;
	delete[] t_orig;
	delete[] v_orig;

	return this->x_array;
};

vector<double> GenModel::solve_i()
{
	this->i_modelled.clear();
	for (size_t it = 0; it < t_array.size(); it++)
	{
		if (this->v_array[it] >= 0)
		{
			this->i_modelled.push_back(min(this->a1 * this->x_array[it] * sinh(this->b * this->v_array[it]), 20.));
		}
		else
		{
			this->i_modelled.push_back(this->a2 * this->x_array[it] * sinh(this->b * this->v_array[it]));
		}
		if (this->i_modelled.back() > 5.0)
			this->i_modelled.back() = 5.0;

	}

#ifdef MODE_DEBUG
	for (map<string, double>::const_iterator it = this->params.begin(); it != this->params.end(); ++it)
	{
		cout << it->first << " " << it->second << " ||| ";

	}
	cout << endl;
#endif

	return this->i_modelled;
};

double GenModel::w_p(double x)
{
	return (this->xp - x) / (1 - this->xp) + 1;
}

double GenModel::w_n(double x)
{
	return x / (1 - this->xn);
}

double GenModel::f_x(double v, double x)
{
	double f_x = 1;
	if (v >= 0)
	{
		if (x >= this->xp)
			f_x = exp(-this->alphap * (x - this->xp)) * this->w_p(x);
	}
	else
	{
		if (x <= 1 - this->xn)
		{
			f_x = exp(this->alphan * (x + this->xn - 1)) * this->w_n(x);
		}
	}

	return f_x;
}

double GenModel::g(double v)
{
	double g = 0.;
	if (v > this->vp)
		g = this->Ap * (exp(v) - exp(this->vp));
	else if (v < -this->vn)
		g = -this->An * (exp(-v) - exp(this->vn));
	return g;
}

double GenModel::x_der(double x, double v)
{
	return this->g(v) * this->f_x(v, x);
}

vector<double> GenModel::get_params_mhc()
{
	vector<double> pp;
	pp.push_back(this->alphap);
	pp.push_back(this->alphan);
	pp.push_back(this->xp);
	pp.push_back(this->xn);
	pp.push_back(this->Ap);
	pp.push_back(this->An);
	pp.push_back(this->vp);
	pp.push_back(this->vn);
	pp.push_back(this->a1);
	pp.push_back(this->a2);
	pp.push_back(this->b);
	pp.push_back(this->x0);
	return pp;
};

void GenModel::set_params_mhc(vector<double>* pp)
{
	limit_params_mhc(pp);
	this->alphap = pp->at(0);
	this->alphan = pp->at(1);
	this->xp = pp->at(2);
	this->xn = pp->at(3);
	this->Ap = pp->at(4);
	this->An = pp->at(5);
	this->vp = pp->at(6);
	this->vn = pp->at(7);
	this->a1 = pp->at(8);
	this->a2 = pp->at(9);
	this->b = pp->at(10);
	this->x0 = pp->at(11);
};

void GenModel::limit_params_mhc(vector<double>* pp)
{
	if (pp->at(0) < this->alphap_min)
		pp->at(0) = alphap_min;
	else if (pp->at(0) > this->alphap_max)
		pp->at(0) = alphap_max;
	if (pp->at(1) < this->alphan_min)
		pp->at(1) = alphan_min;
	else if (pp->at(1) > this->alphan_max)
		pp->at(1) = alphan_max;
	if (pp->at(2) < this->xp_min)
		pp->at(2) = xp_min;
	else if (pp->at(2) > this->xp_max)
		pp->at(2) = xp_max;
	if (pp->at(3) < this->xn_min)
		pp->at(3) = xn_min;
	else if (pp->at(3) > this->xn_max)
		pp->at(3) = xn_max;
	if (pp->at(4) < this->Ap_min)
		pp->at(4) = Ap_min;
	else if (pp->at(4) > this->Ap_max)
		pp->at(4) = Ap_max;
	if (pp->at(5) < this->An_min)
		pp->at(5) = An_min;
	else if (pp->at(5) > this->An_max)
		pp->at(5) = An_max;
	if (pp->at(6) < this->vp_min)
		pp->at(6) = vp_min;
	else if (pp->at(6) > this->vp_max)
		pp->at(6) = vp_max;
	if (pp->at(7) < this->vn_min)
		pp->at(7) = vn_min;
	else if (pp->at(7) > this->vn_max)
		pp->at(7) = vn_max;
	if (pp->at(8) < this->a1_min)
		pp->at(8) = a1_min;
	else if (pp->at(8) > this->a1_max)
		pp->at(8) = a1_max;
	if (pp->at(9) < this->a2_min)
		pp->at(9) = a2_min;
	else if (pp->at(9) > this->a2_max)
		pp->at(9) = a2_max;
	if (pp->at(10) < this->b_min)
		pp->at(10) = b_min;
	else if (pp->at(10) > this->b_max)
		pp->at(10) = b_max;
	if (pp->at(11) < this->x0_min)
		pp->at(11) = x0_min;
	else if (pp->at(11) > this->x0_max)
		pp->at(11) = x0_max;
};

void GenModel::set_params_from_mhc()
{
	this->params.at("alphap") = this->alphap;
	this->params.at("alphan") = this->alphan;
	this->params.at("xp") = this->xp;
	this->params.at("xn") = this->xn;
	this->params.at("Ap") = this->Ap;
	this->params.at("An") = this->An;
	this->params.at("vp") = this->vp;
	this->params.at("vn") = this->vn;
	this->params.at("a1") = this->a1;
	this->params.at("a2") = this->a2;
	this->params.at("b") = this->b;
	this->params.at("x0") = this->x0;
};

QDeformed::QDeformed()
{
	set_def_parameters();
}

QDeformed::QDeformed(double alpha)
{
	set_def_parameters();

	if (alpha < this->params_bounds.at("alpha").min)
	{
		alpha = this->params_bounds.at("alpha").min;
		cout << "WARNING. Alpha value below min bound. Set to " << alpha << "." << endl;
	}
	else if (alpha > this->params_bounds.at("alpha").max)
	{
		alpha = this->params_bounds.at("alpha").max;
		cout << "WARNING. Alpha value exceeds max bound. Set to " << alpha << "." << endl;
	}
	this->params.at("alpha") = alpha;
}

void QDeformed::final_init_routine() {};
void QDeformed::final_reset_routine() {};


void QDeformed::set_def_parameters()
{
	this->params = { {"xp", 0.2103},
					{"xn", 0.5712},
					{"Ap", 0.3213},
					{"An", 0.0492},
					{"vp", 4.5434},
					{"vn", 0.0},
					{"q", 0.7261},
					{"gamma1", 0.2266},
					{"gamma2", 0.001},
					{"delta1", 1.0206},
					{"delta2", 5.373},
					{"x0", 0.1},
					{"alpha", 1.0}
	};

	this->params_bounds = { {"xp",  params_bouns_struct(0.0,0.99)},
						   {"xn",  params_bouns_struct(0.0,0.99)},
						   {"Ap",  params_bouns_struct(0.0,50.0)},
						   {"An",  params_bouns_struct(0.0,50.0)},
						   {"vp",  params_bouns_struct(0.0,10.0)},
						   {"vn",  params_bouns_struct(0.0,10.0)},
						   {"q",  params_bouns_struct(0.0,10.0)},
						   {"gamma1",  params_bouns_struct(0.0,50.0)},
						   {"gamma2",  params_bouns_struct(0.0,50.0)},
						   {"delta1",  params_bouns_struct(0.0,50.0)},
						   {"delta2",  params_bouns_struct(0.0,50.0)},
						   {"x0",  params_bouns_struct(0.0,1.0)},
						   {"alpha",  params_bouns_struct(0.0,2.0)}
	};
}

double QDeformed::wp(double x)
{
	return (this->params.at("xp") - x) / (1.0 - this->params.at("xp")) + 1.;
}

double QDeformed::wn(double x)
{
	return x / (1.0 - this->params.at("xn"));
}

double QDeformed::f_x(double k, double x)
{
	double ret;
	if (k > 0 && x >= this->params.at("xp"))
	{
		//ret = exp(-(x - this->params.at("xp"))) * wp(x);
		return exp(-(x - this->params.at("xp"))) * wp(x);
	}
	else if (k <= 0 && x <= 1.0 - this->params.at("xn"))
	{
		/*double ret1 = wn(x);
		double ret2 = exp(x + this->params.at("xn") - 1);
		double ret3 = x + this->params.at("xn") - 1;
		ret = exp(x + this->params.at("xn") - 1) * wn(x);*/
		return exp(x + this->params.at("xn") - 1) * wn(x);
	}
	else
		return 1.;
}

double QDeformed::g(double v)
{
	double ret;
	if (v > this->params.at("vp"))
	{
		ret = this->params.at("Ap") * (exp(v) - exp(this->params.at("vp")));
		return this->params.at("Ap") * (exp(v) - exp(this->params.at("vp")));
	}
	else if (v < -this->params.at("vn"))
	{
		ret = -this->params.at("An") * (exp(-v) - exp(this->params.at("vn")));
		return -this->params.at("An") * (exp(-v) - exp(this->params.at("vn")));
	}
	else
		return 0;
}

double QDeformed::qexp(double arg)
{
	if (this->params.at("q") == 1)
		return exp(arg);
	else
	{
		if (1.0 + (1.0 - this->params.at("q")) * arg > 0)
			return pow(1.0 + (1.0 - this->params.at("q")) * arg, 1. / (1. - this->params.at("q")));
		else
			return 0;
	}
}

double QDeformed::qsinh(double arg)
{
	return 0.5 * (this->qexp(arg) - this->qexp(-arg));
}

double QDeformed::x_der(double x, double v)
{
	return f_x(v, x) * g(v);
}

void QDeformed::frac_solver(vector<double>* x_ret, double x0, double T, size_t N)
{
	double alpha = this->params.at("alpha");
	fft_duration = 0;
	auto start = high_resolution_clock::now();

	// TODO : EXTRACT CODE FROM FUNC IN INITIALIZER OR SMTH
	double h = T / (double)N;
	double ha = pow(h, alpha);
	double ga1 = tgamma(alpha + 1);
	double ga2 = tgamma(alpha + 2);
	double* t = new double[N + 1];
	double* x = new double[N + 1];
	double* k = new double[N + 1];
	double* a = new double[N + 1];
	double* b = new double[N + 1];
	double* t_orig = new double[this->t_array.size()];
	double* v_orig = new double[this->v_array.size()];
	double* f = new double[N + 1];
	double* conva = new double[N + 1 + N];
	double* convb = new double[N + 1 + N];
	for (int i = 0; i < this->NPOINTS; i++)
	{
		t_orig[i] = this->t_array[i];
		v_orig[i] = this->v_array[i];
	}
	double* v_interp = new double[N + 1];
	gsl_interp* intrpl1 = gsl_interp_alloc(gsl_interp_linear, this->NPOINTS);
	gsl_interp_accel* accel = gsl_interp_accel_alloc();
	gsl_interp_init(intrpl1, t_orig, v_orig, NPOINTS);
	for (int i = 0; i < N + 1; i++)
	{
		t[i] = (double)i * h;
		if (i == N)
			t[i] = T;
		v_interp[i] = gsl_interp_eval(intrpl1, t_orig, v_orig, t[i], accel);
		k[i] = i + 1;
		a[i] = pow(k[i] + 1, alpha + 1) - 2.0 * pow(k[i], alpha + 1) + pow(k[i] - 1, alpha + 1);
		b[i] = pow(k[i], alpha) - pow(k[i] - 1, alpha);
		f[i] = 0;
		x[i] = 0;
	}
	x[0] = x0;
	f[0] = x_der(x[0], v_interp[0]);
	gsl_interp_free(intrpl1);
	gsl_interp_accel_free(accel);


	for (int i = 1; i < N + 1; i++)
	{
		for (int j = 0; j < i + 1; j++)
		{
			if (x[j] != x[j])
				cout << "Hah, x!" << endl;
			if (v_interp[j] != v_interp[j])
				cout << "Hah, v_interp!" << endl;
			f[j] = this->x_der(x[j], v_interp[j]);
			if (f[j] != f[j])
			{
				cout << x[j - 2] << ' ' << x[j - 1] << ' ' << x[j] << endl;
				cout << "Hah, f!: " << f[j] << endl;
				f[j] = this->x_der(x[j], v_interp[j]);
			}
		}
		fftw_fftconvolve(a, f, N, conva); // TODO : replace N with i
		fftw_fftconvolve(b, f, N, convb);
		double pred = x0 + ha / ga1 * convb[i - 1];
		x[i] = x0 + ha / ga2 * (x_der(pred, v_interp[i - 1]) - (pow(i - 1.0, alpha + 1) - (i - 1.0 - alpha) * pow((double)i, alpha)) * f[0] + conva[i - 1]);
		if (x[i] != x[i])
			cout << "Here fail!" << endl;
	}

	gsl_interp* intrpl2 = gsl_interp_alloc(gsl_interp_linear, N + 1);
	gsl_interp_accel* accel2 = gsl_interp_accel_alloc();
	gsl_interp_init(intrpl2, t, x, N + 1);
	x_ret->clear();
	for (int i = 0; i < this->NPOINTS; i++)
	{
		x_ret->push_back(gsl_interp_eval(intrpl2, t, x, this->t_array[i], accel2));
	}

	gsl_interp_free(intrpl2);
	gsl_interp_accel_free(accel2);

	delete[] t;
	delete[] x;
	delete[] k;
	delete[] a;
	delete[] b;
	delete[] t_orig;
	delete[] v_orig;
	delete[] v_interp;
	delete[] f;
	delete[] conva;
	delete[] convb;
	static size_t frac_solver_counter = 0;
	frac_solver_counter++;
	//cout << "Frac solver worked " << frac_solver_counter << " times." << endl;

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	uint64_t duration_wa_fft = duration.count() - fft_duration;
	//cout << "Only FFT duration:\t" << fft_duration << "ms." << endl << "Fracsolver duration excl. FFT:\t" << duration_wa_fft << "ms." << endl << "Fracsolver total:\t" << duration.count() << "ms." << endl;
}



vector<double> QDeformed::solve_x()
{
	size_t N = this->ivp_steps;
	this->frac_solver(&this->x_array, this->params.at("x0"), t_array.back(), (double)N);
	return this->x_array;
}

vector<double> QDeformed::solve_i()
{
	this->i_modelled.clear();

	for (int i = 0; i < this->NPOINTS; i++)
	{
		this->i_modelled.push_back(this->params.at("gamma1") * this->x_array[i] * this->qsinh(this->params.at("delta1") * this->v_array[i]) + this->params.at("gamma2") * (1 - this->x_array[i]) * this->qsinh(this->params.at("delta2") * this->v_array[i]));
	}
	return this->i_modelled;
}

void QDeformed::setup_convolution_environment()
{
	size_t N = this->ivp_steps;
	size_t convolve_size = (N + 1) + N;

	/*uint16_t power = 1;
	while (pow(2,power)<convolve_size)
		power++;
	size_t old_size = convolve_size;
	convolve_size = pow(2,power);*/

	/* here fft is set up */
	this->wavetable = gsl_fft_complex_wavetable_alloc(convolve_size);
	this->fft_w = gsl_fft_complex_workspace_alloc(convolve_size);

	this->data1 = new double[convolve_size * 2];
	this->data2 = new double[convolve_size * 2];
	this->data3 = new double[convolve_size * 2];

	for (int j = 0; j < convolve_size; j++)
	{
		if (j < N + 1)
		{
			IMAG(data1, j) = 0;
			IMAG(data2, j) = 0;
		}
		else
		{
			REAL(data1, j) = 0;
			IMAG(data1, j) = 0;
			REAL(data2, j) = 0;
			IMAG(data2, j) = 0;
		}
	}
}

void QDeformed::reset_convolution_environment()
{
	gsl_fft_complex_wavetable_free(this->wavetable);
	gsl_fft_complex_workspace_free(this->fft_w);

	delete[] this->data1;
	delete[] this->data2;
	delete[] this->data3;
}

QDeformedMHC::QDeformedMHC()
{
	set_def_parameters();
}

void QDeformedMHC::set_def_parameters()
{
	this->xp = 6.188413833797580255e-01;
	this->xn = 1.917203358770194299e+01;
	this->Ap = 7.139129118557706322e-02;
	this->An = 5.691862660926473062e-03;
	this->vp = 4.717732974303751270e+00;
	this->vn = 7.541510802991451782e-06;
	this->gamma1 = 1.746164607742726860e+00;
	this->gamma2 = 2.520196716597585151e+00;
	this->delta1 = 4.120514766418048147e+00;
	this->delta2 = 2.164854963492383710e+00;
	this->x0 = 0.0;
	this->lambda = 1.595113878010011632e+01;
	this->A = 1.372215502635717455e+00;

	this->xp_min = 0.;
	this->xn_min = 0.;
	this->Ap_min = 0.;
	this->An_min = 0.;
	this->vp_min = 0.;
	this->vn_min = 0.;
	this->gamma1_min = 0.;
	this->gamma2_min = 0.;
	this->delta1_min = 0.;
	this->delta2_min = 0.;
	this->x0_min = 0.0;
	this->lambda_min = 0.;
	this->A_min = 0.;

	this->xp_max = 0.99;
	this->xn_max = 0.99;
	this->Ap_max = 100.0;
	this->An_max = 100.0;
	this->vp_max = 10.;
	this->vn_max = 10.;
	this->gamma1_max = 100.;
	this->gamma2_max = 100.;
	this->delta1_max = 100.;
	this->delta2_max = 100.;
	this->x0_max = 1.5;
	this->lambda_max = 100.;
	this->A_max = 100.;

	this->params = { {"xp", 6.188413833797580255e-01},
					{"xn", 1.917203358770194299e+01},
					{"Ap", 7.139129118557706322e-02},
					{"An", 5.691862660926473062e-03},
					{"vp", 4.717732974303751270e+00},
					{"vn", 7.541510802991451782e-06},
					{"gamma1", 1.746164607742726860e+00},
					{"gamma2", 2.520196716597585151e+00},
					{"delta1", 4.120514766418048147e+00},
					{"delta2", 2.164854963492383710e+00},
					{"x0", 0.0},
					{"lambda", 1.595113878010011632e+01},
					{"A", 1.372215502635717455e+00}
	};

	this->params_bounds = { {"xp",  params_bouns_struct(0.0,0.99)},
						   {"xn",  params_bouns_struct(0.0,0.99)},
						   {"Ap",  params_bouns_struct(0.0,50.0)},
						   {"An",  params_bouns_struct(0.0,50.0)},
						   {"vp",  params_bouns_struct(0.0,10.0)},
						   {"vn",  params_bouns_struct(0.0,10.0)},
						   {"gamma1",  params_bouns_struct(0.0,50.0)},
						   {"gamma2",  params_bouns_struct(0.0,50.0)},
						   {"delta1",  params_bouns_struct(0.0,50.0)},
						   {"delta2",  params_bouns_struct(0.0,50.0)},
						   {"x0",  params_bouns_struct(0.0,1.0)},
						   {"lambda",  params_bouns_struct(0.0,50.0)},
						   {"A",  params_bouns_struct(0.0,5.0)}
	};
}

vector <double> QDeformedMHC::solve_i()
{
	this->i_modelled.clear();

	for (int i = 0; i < this->NPOINTS; i++)
	{
		/*if (this->params.at("gamma1") != this->params.at("gamma1"))
			cout << "Error gamma1" << endl;
		if (this->params.at("gamma2") != this->params.at("gamma2"))
			cout << "Error gamma2" << endl;
		if (this->params.at("delta1") != this->params.at("delta1"))
			cout << "Error delta1" << endl;
		if (this->params.at("delta2") != this->params.at("delta2"))
			cout << "Error delta2" << endl;
		if (this->v_array[i] != this->v_array[i])
			cout << "Error v array" << endl;
		if (this->params.at("gamma1") * this->x_array[i] * i_MHC(this->params.at("delta1") * this->v_array[i]) + this->params.at("gamma2") * (1. - this->x_array[i]) * i_MHC(this->params.at("delta2") * this->v_array[i]) != this->params.at("gamma1") * this->x_array[i] * i_MHC(this->params.at("delta1") * this->v_array[i]) + this->params.at("gamma2") * (1. - this->x_array[i]) * i_MHC(this->params.at("delta2") * this->v_array[i]))
		{
			cout << "Error is near " << this->params.at("gamma1") * this->x_array[i] * i_MHC(this->params.at("delta1") * this->v_array[i]) + this->params.at("gamma2") * (1. - this->x_array[i]) * i_MHC(this->params.at("delta2") * this->v_array[i]) << endl;
			cout << "i_MHC: " << i_MHC(this->params.at("delta1") * this->v_array[i]) << endl;
			cout << "x: " << this->x_array[i] << endl;
			cout << "g: " << this->params.at("gamma1") << endl;
		}*/
		double i_ = this->gamma1 * this->x_array[i] * i_MHC(this->delta1 * this->v_array[i]) + this->gamma2 * (1. - this->x_array[i]) * i_MHC(this->delta2 * this->v_array[i]);
		if (i_ < 5.0)
			this->i_modelled.push_back(this->gamma1 * this->x_array[i] * i_MHC(this->delta1 * this->v_array[i]) + this->gamma2 * (1. - this->x_array[i]) * i_MHC(this->delta2 * this->v_array[i]));
		else
			this->i_modelled.push_back(5.0);
	}
	return this->i_modelled;
}

double QDeformedMHC::fermi(double val)
{
	/*if (1. / (1. + exp(val)) != 1. / (1. + exp(val)))
		cout << "Haha,found, it's fermi!" << endl;*/
	return 1. / (1. + exp(val));
}

double QDeformedMHC::i_MHC(double V)
{
	double sum1 = 0;
	for (int i = 0; i < this->pols_x.size(); i++)
	{
		/*/if (this->pols_x[i] != this->pols_x[i])
			cout << "pols_x Wrong!" << endl;
		if (this->pols_w[i] != this->pols_w[i])
			cout << "pols_w Wrong!" << endl;*/
		sum1 += this->pols_w[i] * this->fermi(2.0 * sqrt(this->lambda) * this->pols_x[i] + this->lambda - V);
	}

	double sum2 = 0;
	for (int i = 0; i < this->pols_x.size(); i++)
	{
		sum2 += this->pols_w[i] * this->fermi(2.0 * sqrt(this->lambda) * this->pols_x[i] + this->lambda + V);
	}

	/*double temp = this->params.at("A");
	cout << temp << endl;*/
	/*if (this->params.at("A") != this->params.at("A"))
		cout << "A is Wrong, lol!" << endl;
	if (sum1 != sum1)
		cout << "sum2 is Wrong, lol!" << endl;
	if (sum1 * this->params.at("A") - sum2 * this->params.at("A") != sum1 * this->params.at("A") - sum2 * this->params.at("A"))
		cout << "whole i_MHC!!! is Wrong, lol!" << endl;
	cout << sum1 * this->params.at("A") - sum2 * this->params.at("A") << endl;*/
	return sum1 * this->A - sum2 * this->A;
}

void QDeformedMHC::setup_pols(size_t N)
{
	hermite_handle(N, 0.0, 1, &this->pols_x, &this->pols_w);
}

void QDeformedMHC::reset_pols()
{
	this->pols_x.clear();
	this->pols_w.clear();
}

double QDeformedMHC::wp(double x)
{
	return (this->xp - x) / (1.0 - this->xp) + 1.;
}

double QDeformedMHC::wn(double x)
{
	return x / (1.0 - this->xn);
}

double QDeformedMHC::f_x(double k, double x)
{
	double ret;
	if (k > 0 && x >= this->xp)
	{
		//ret = exp(-(x - this->params.at("xp"))) * wp(x);
		return exp(-(x - this->xp)) * wp(x);
	}
	else if (k <= 0 && x <= 1.0 - this->xn)
	{
		/*double ret1 = wn(x);
		double ret2 = exp(x + this->params.at("xn") - 1);
		double ret3 = x + this->params.at("xn") - 1;
		ret = exp(x + this->params.at("xn") - 1) * wn(x);*/
		return exp(x + this->xn - 1) * wn(x);
	}
	else
		return 1.;
}

double QDeformedMHC::g(double v)
{
	double ret;
	if (v > this->vp)
	{
		ret = this->Ap * (exp(v) - exp(this->vp));
		return ret;
	}
	else if (v < -this->vn)
	{
		ret = -this->An * (exp(-v) - exp(this->vn));
		return ret;
	}
	else
		return 0;
}

double QDeformedMHC::x_der(double x, double v)
{
	return f_x(v, x) * g(v);
}

struct ode_params
{
	double* v;
	QDeformedMHC* model;
};

int x_der_gsl(double t, const double *x, double *f, void* params)
{
	(void)(t);
	ode_params p = *(ode_params*)params;
	*f = p.model->f_x(*p.v, *x) * p.model->g(*p.v);
	return GSL_SUCCESS;
}

void QDeformedMHC::x_solver(vector<double>* x_ret, double x0, double T, size_t N)
{
	fft_duration = 0;
	auto start = high_resolution_clock::now();

	// TODO : EXTRACT CODE FROM FUNC IN INITIALIZER OR SMTH
	double h = T / (double)N;
	double* t = new double[N + 1];
	double* x = new double[N + 1];
	double* t_orig = new double[this->t_array.size()];
	double* v_orig = new double[this->v_array.size()];
	for (int i = 0; i < this->NPOINTS; i++)
	{
		t_orig[i] = this->t_array[i];
		v_orig[i] = this->v_array[i];
	}
	double* v_interp = new double[N + 1];
	gsl_interp* intrpl1 = gsl_interp_alloc(gsl_interp_linear, this->NPOINTS);
	gsl_interp_accel* accel = gsl_interp_accel_alloc();
	gsl_interp_init(intrpl1, t_orig, v_orig, NPOINTS);
	for (int i = 0; i < N + 1; i++)
	{
		t[i] = (double)i * h;
		if (i == N)
			t[i] = T;
		v_interp[i] = gsl_interp_eval(intrpl1, t_orig, v_orig, t[i], accel);
	}
	x[0] = x0;

	gsl_interp_free(intrpl1);
	gsl_interp_accel_free(accel);

	ode_params p;
	p.v = &v_interp[0];
	p.model = this;

	gsl_odeiv2_system sys = {x_der_gsl, nullptr, 1, &p};
	gsl_odeiv2_driver* d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd,
		h, 1e-6, 0.0);

	int i;
	double t0 = 0.0, t1 = t[N];
	double x_ode;
	x_ode = x[0];

	for (i = 1; i < N+1; i++)
	{
		int status = gsl_odeiv2_driver_apply(d, &t0, t[i], &x_ode);

		if (status != GSL_SUCCESS)
		{
			printf("error, return value=%d\n", status);
			break;
		}
		x[i] = x_ode;
		p.v = &v_interp[i];
	}

	gsl_odeiv2_driver_free(d);

	gsl_interp* intrpl2 = gsl_interp_alloc(gsl_interp_linear, N + 1);
	gsl_interp_accel* accel2 = gsl_interp_accel_alloc();
	gsl_interp_init(intrpl2, t, x, N + 1);
	x_ret->clear();
	for (int i = 0; i < this->NPOINTS; i++)
	{
		x_ret->push_back(gsl_interp_eval(intrpl2, t, x, this->t_array[i], accel2));
	}

	gsl_interp_free(intrpl2);
	gsl_interp_accel_free(accel2);

	delete[] t;
	delete[] x;
	delete[] t_orig;
	delete[] v_orig;
	delete[] v_interp;
	static size_t frac_solver_counter = 0;
	frac_solver_counter++;
	//cout << "Frac solver worked " << frac_solver_counter << " times." << endl;

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	uint64_t duration_wa_fft = duration.count() - fft_duration;
	//cout << "Only FFT duration:\t" << fft_duration << "ms." << endl << "Fracsolver duration excl. FFT:\t" << duration_wa_fft << "ms." << endl << "Fracsolver total:\t" << duration.count() << "ms." << endl;
}



vector<double> QDeformedMHC::solve_x()
{
	size_t N = this->ivp_steps;
	this->x_solver(&this->x_array, this->x0, t_array.back(), (double)N);
	return this->x_array;
}

vector<double> QDeformedMHC::get_params_mhc()
{
	vector<double> pp;
	pp.push_back(this->xp);
	pp.push_back(this->xn);
	pp.push_back(this->Ap);
	pp.push_back(this->An);
	pp.push_back(this->vp);
	pp.push_back(this->vn);
	pp.push_back(this->gamma1);
	pp.push_back(this->gamma2);
	pp.push_back(this->delta1);
	pp.push_back(this->delta2);
	pp.push_back(this->x0);
	pp.push_back(this->lambda);
	pp.push_back(this->A);
	return pp;
};

void QDeformedMHC::set_params_mhc(vector<double>* pp)
{
	limit_params_mhc(pp);
	this->xp = pp->at(0);
	this->xn = pp->at(1);
	this->Ap = pp->at(2);
	this->An = pp->at(3);
	this->vp = pp->at(4);
	this->vn = pp->at(5);
	this->gamma1 = pp->at(6);
	this->gamma2 = pp->at(7);
	this->delta1 = pp->at(8);
	this->delta2 = pp->at(9);
	this->x0 = pp->at(10);
	this->lambda = pp->at(11);
	this->A = pp->at(12);
};

void QDeformedMHC::limit_params_mhc(vector<double>* pp)
{
	if (pp->at(0) < this->xp_min)
		pp->at(0) = xp_min;
	else if (pp->at(0) > this->xp_max)
		pp->at(0) = xp_max;
	if (pp->at(1) < this->xn_min)
		pp->at(1) = xn_min;
	else if (pp->at(1) > this->xn_max)
		pp->at(1) = xn_max;
	if (pp->at(2) < this->Ap_min)
		pp->at(2) = Ap_min;
	else if (pp->at(2) > this->Ap_max)
		pp->at(2) = Ap_max;
	if (pp->at(3) < this->An_min)
		pp->at(3) = An_min;
	else if (pp->at(3) > this->An_max)
		pp->at(3) = An_max;
	if (pp->at(4) < this->vp_min)
		pp->at(4) = vp_min;
	else if (pp->at(4) > this->vp_max)
		pp->at(4) = vp_max;
	if (pp->at(5) < this->vn_min)
		pp->at(5) = vn_min;
	else if (pp->at(5) > this->vn_max)
		pp->at(5) = vn_max;
	if (pp->at(6) < this->gamma1_min)
		pp->at(6) = gamma1_min;
	else if (pp->at(6) > this->gamma1_max)
		pp->at(6) = gamma1_max;
	if (pp->at(7) < this->gamma2_min)
		pp->at(7) = gamma2_min;
	else if (pp->at(7) > this->gamma2_max)
		pp->at(7) = gamma2_max;
	if (pp->at(8) < this->delta1_min)
		pp->at(8) = delta1_min;
	else if (pp->at(8) > this->delta1_max)
		pp->at(8) = delta1_max;
	if (pp->at(9) < this->delta2_min)
		pp->at(9) = delta2_min;
	else if (pp->at(9) > this->delta2_max)
		pp->at(9) = delta2_max;
	if (pp->at(10) < this->x0_min)
		pp->at(10) = x0_min;
	else if (pp->at(10) > this->x0_max)
		pp->at(10) = x0_max;
	if (pp->at(11) < this->lambda_min)
		pp->at(11) = lambda_min;
	else if (pp->at(11) > this->lambda_max)
		pp->at(11) = lambda_max;
	if (pp->at(12) < this->A_min)
		pp->at(12) = A_min;
	else if (pp->at(12) > this->A_max)
		pp->at(12) = A_max;
};

void QDeformedMHC::set_params_from_mhc()
{
	this->params.at("xp") = this->xp;
	this->params.at("xn") = this->xn;
	this->params.at("Ap") = this->Ap;
	this->params.at("An") = this->An;
	this->params.at("vp") = this->vp;
	this->params.at("vn") = this->vn;
	this->params.at("gamma1") = this->gamma1;
	this->params.at("gamma2") = this->gamma2;
	this->params.at("delta1") = this->delta1;
	this->params.at("delta2") = this->delta2;
	this->params.at("x0") = this->x0;
	this->params.at("lambda") = this->lambda;
	this->params.at("A") = this->A;
};

QDeformedMHCFrac::QDeformedMHCFrac()
{
	set_def_parameters();
}

QDeformedMHCFrac::QDeformedMHCFrac(double alpha)
{
	set_def_parameters();

	if (alpha < this->alpha_min)
	{
		alpha = this->alpha_min;
		cout << "WARNING. Alpha value below min bound. Set to " << alpha << "." << endl;
	}
	else if (alpha > this->alpha_max)
	{
		alpha = this->alpha_max;
		cout << "WARNING. Alpha value exceeds max bound. Set to " << alpha << "." << endl;
	}
	this->alpha = alpha;
}

void QDeformedMHCFrac::final_init_routine()
{
	this->frac_solver_init(this->t_array[this->t_array.size() - 1], this->ivp_steps);
}

void QDeformedMHCFrac::final_reset_routine()
{
	this->frac_solver_reset();
}

void QDeformedMHCFrac::set_def_parameters()
{
	this->xp = 6.188413833797580255e-01;
	this->xn = 1.917203358770194299e+01;
	this->Ap = 7.139129118557706322e-02;
	this->An = 5.691862660926473062e-03;
	this->vp = 4.717732974303751270e+00;
	this->vn = 7.541510802991451782e-06;
	this->gamma1 = 1.746164607742726860e+00;
	this->gamma2 = 2.520196716597585151e+00;
	this->delta1 = 4.120514766418048147e+00;
	this->delta2 = 2.164854963492383710e+00;
	this->x0 = 0.0;
	this->lambda = 1.595113878010011632e+01;
	this->A = 1.372215502635717455e+00;
	this->alpha = 6.971426484505444110e-01;

	this->xp_min = 0.;
	this->xn_min = 0.;
	this->Ap_min = 0.;
	this->An_min = 0.;
	this->vp_min = 0.;
	this->vn_min = 0.;
	this->gamma1_min = 0.;
	this->gamma2_min = 0.;
	this->delta1_min = 0.;
	this->delta2_min = 0.;
	this->x0_min = 0.0;
	this->lambda_min = 0.;
	this->A_min = 0.;
	this->alpha_min = 1.;

	this->xp_max = 0.99;
	this->xn_max = 0.99;
	this->Ap_max = 100.0;
	this->An_max = 100.0;
	this->vp_max = 10.;
	this->vn_max = 10.;
	this->gamma1_max = 100.;
	this->gamma2_max = 100.;
	this->delta1_max = 100.;
	this->delta2_max = 100.;
	this->x0_max = 1.5;
	this->lambda_max = 100.;
	this->A_max = 100.;
	this->alpha_max = 1.0;

	this->params = { {"xp", 6.188413833797580255e-01},
					{"xn", 1.917203358770194299e+01},
					{"Ap", 7.139129118557706322e-02},
					{"An", 5.691862660926473062e-03},
					{"vp", 4.717732974303751270e+00},
					{"vn", 7.541510802991451782e-06},
					{"gamma1", 1.746164607742726860e+00},
					{"gamma2", 2.520196716597585151e+00},
					{"delta1", 4.120514766418048147e+00},
					{"delta2", 2.164854963492383710e+00},
					{"x0", 0.0},
					{"lambda", 1.595113878010011632e+01},
					{"A", 1.372215502635717455e+00},
					{"alpha", 6.971426484505444110e-01}
	};

	this->params_bounds = { {"xp",  params_bouns_struct(0.0,0.99)},
						   {"xn",  params_bouns_struct(0.0,0.99)},
						   {"Ap",  params_bouns_struct(0.0,50.0)},
						   {"An",  params_bouns_struct(0.0,50.0)},
						   {"vp",  params_bouns_struct(0.0,10.0)},
						   {"vn",  params_bouns_struct(0.0,10.0)},
						   {"gamma1",  params_bouns_struct(0.0,50.0)},
						   {"gamma2",  params_bouns_struct(0.0,50.0)},
						   {"delta1",  params_bouns_struct(0.0,50.0)},
						   {"delta2",  params_bouns_struct(0.0,50.0)},
						   {"x0",  params_bouns_struct(0.0,1.0)},
						   {"lambda",  params_bouns_struct(0.0,50.0)},
						   {"A",  params_bouns_struct(0.0,5.0)},
						   {"alpha",  params_bouns_struct(0.0,1.0)}
	};
}

vector <double> QDeformedMHCFrac::solve_i()
{
	this->i_modelled.clear();

	for (int i = 0; i < this->NPOINTS; i++)
	{
		double i_ = this->gamma1 * this->x_array[i] * i_MHC(this->delta1 * this->v_array[i]) + this->gamma2 * (1. - this->x_array[i]) * i_MHC(this->delta2 * this->v_array[i]);
		if (i_ < 5.0)
			this->i_modelled.push_back(this->gamma1 * this->x_array[i] * i_MHC(this->delta1 * this->v_array[i]) + this->gamma2 * (1. - this->x_array[i]) * i_MHC(this->delta2 * this->v_array[i]));
		else
			this->i_modelled.push_back(5.0);
	}
	return this->i_modelled;
}

double QDeformedMHCFrac::fermi(double val)
{
	return 1. / (1. + exp(val));
}

double QDeformedMHCFrac::i_MHC(double V)
{
	double sum1 = 0;
	for (int i = 0; i < this->pols_x.size(); i++)
	{
		sum1 += this->pols_w[i] * this->fermi(2.0 * sqrt(this->lambda) * this->pols_x[i] + this->lambda - V);
	}

	double sum2 = 0;
	for (int i = 0; i < this->pols_x.size(); i++)
	{
		sum2 += this->pols_w[i] * this->fermi(2.0 * sqrt(this->lambda) * this->pols_x[i] + this->lambda + V);
	}

	return sum1 * this->A - sum2 * this->A;
}

void QDeformedMHCFrac::setup_pols(size_t N)
{
	hermite_handle(N, 0.0, 1, &this->pols_x, &this->pols_w);
}

void QDeformedMHCFrac::reset_pols()
{
	this->pols_x.clear();
	this->pols_w.clear();
}

double QDeformedMHCFrac::wp(double x)
{
	return (this->xp - x) / (1.0 - this->xp) + 1.;
}

double QDeformedMHCFrac::wn(double x)
{
	return x / (1.0 - this->xn);
}

double QDeformedMHCFrac::f_x(double k, double x)
{
	double ret;
	if (k > 0 && x >= this->xp)
	{
		return exp(-(x - this->xp)) * wp(x);
	}
	else if (k <= 0 && x <= 1.0 - this->xn)
	{
		return exp(x + this->xn - 1) * wn(x);
	}
	else
		return 1.;
}

double QDeformedMHCFrac::g(double v)
{
	double ret;
	if (v > this->vp)
	{
		ret = this->Ap * (exp(v) - exp(this->vp));
		return ret;
	}
	else if (v < -this->vn)
	{
		ret = -this->An * (exp(-v) - exp(this->vn));
		return ret;
	}
	else
		return 0;
}

double QDeformedMHCFrac::x_der(double x, double v)
{
	return f_x(v, x) * g(v);
}

void QDeformedMHCFrac::frac_solver_init(double T, size_t N)
{
	h_fs = T / (double)(N - 1);
	ha = pow(h_fs, alpha);
	ga1 = tgamma(alpha + 1);
	ga2 = tgamma(alpha + 2);
	t_fs = new double[N];
	x_fs = new double[N];
	k_fs = new double[N];
	a_fs = new double[N];
	b_fs = new double[N];
	t_orig = new double[this->t_array.size()];
	v_orig = new double[this->v_array.size()];
	f_fs = new double[N];
	conva = new double[N - 1 + N];
	convb = new double[N - 1 + N];
	for (int i = 0; i < this->NPOINTS; i++)
	{
		t_orig[i] = this->t_array[i];
		v_orig[i] = this->v_array[i];
	}

	gsl_interp* intrpl1 = gsl_interp_alloc(gsl_interp_linear, this->NPOINTS);
	gsl_interp_accel* accel1 = gsl_interp_accel_alloc();
	gsl_interp_init(intrpl1, t_orig, v_orig, NPOINTS);
	v_interp = new double[N];
	intrpl1 = gsl_interp_alloc(gsl_interp_linear, this->NPOINTS);
	accel1 = gsl_interp_accel_alloc();
	gsl_interp_init(intrpl1, t_orig, v_orig, NPOINTS);
	for (int i = 0; i < N; i++)
	{
		t_fs[i] = (double)i * h_fs;
		if (i == N - 1)
			t_fs[i] = T;
		v_interp[i] = gsl_interp_eval(intrpl1, t_orig, v_orig, t_fs[i], accel1);
	}
	gsl_interp_free(intrpl1);
	gsl_interp_accel_free(accel1);


	intrpl2 = gsl_interp_alloc(gsl_interp_linear, N);
	accel2 = gsl_interp_accel_alloc();

	for (int i = 0; i < N; i++)
	{
		k_fs[i] = i + 1;
		a_fs[i] = pow(k_fs[i] + 1, alpha + 1) - 2.0 * pow(k_fs[i], alpha + 1) + pow(k_fs[i] - 1, alpha + 1);
		b_fs[i] = pow(k_fs[i], alpha) - pow(k_fs[i] - 1, alpha);
	}
}

void QDeformedMHCFrac::frac_solver_reset()
{
	gsl_interp_free(intrpl2);
	gsl_interp_accel_free(accel2);
	delete[] t_fs;
	delete[] x_fs;
	delete[] k_fs;
	delete[] a_fs;
	delete[] b_fs;
	delete[] t_orig;
	delete[] v_orig;
	delete[] v_interp;
	delete[] f_fs;
	delete[] conva;
	delete[] convb;
}

void QDeformedMHCFrac::frac_solver(vector<double>* x_ret, double x0, double T)
{
	size_t N = this->ivp_steps;
	fft_duration = 0;
	auto start = high_resolution_clock::now();

	for (int i = 0; i < N - 1 + N; i++)
	{
		conva[i] = 0;
		convb[i] = 0;
	}
	for (int i = 0; i < N; i++)
	{
		f_fs[i] = 0;
		x_fs[i] = 0;
	}
	x_fs[0] = x0;
	f_fs[0] = x_der(x_fs[0], v_interp[0]);

	for (int i = 1; i < N; i++)
	{
		for (int j = 0; j < i; j++)
		{
			f_fs[j] = this->x_der(x_fs[j], v_interp[j]);
		}
		fftw_fftconvolve(a_fs, f_fs, i, conva); // TODO : replace N with i
		fftw_fftconvolve(b_fs, f_fs, i, convb);
		double pred = x0 + ha / ga1 * convb[i - 1];
		x_fs[i] = x0 + ha / ga2 * (x_der(pred, v_interp[i - 1]) - (pow(i - 1.0, alpha + 1) - (i - 1.0 - alpha) * pow((double)i, alpha)) * f_fs[0] + conva[i - 1]);
	}

	x_ret->clear();
	gsl_interp_init(intrpl2, t_fs, x_fs, N);
	for (int i = 0; i < this->NPOINTS; i++)
	{
		x_ret->push_back(gsl_interp_eval(intrpl2, t_fs, x_fs, this->t_array[i], accel2));
	}

	static size_t frac_solver_counter = 0;
	frac_solver_counter++;
	//cout << "Frac solver worked " << frac_solver_counter << " times." << endl;

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	uint64_t duration_wa_fft = duration.count() - fft_duration;
	//cout << "Only FFT duration:\t" << fft_duration << "ms." << endl << "Fracsolver duration excl. FFT:\t" << duration_wa_fft << "ms." << endl << "Fracsolver total:\t" << duration.count() << "ms." << endl;
}



vector<double> QDeformedMHCFrac::solve_x()
{
	size_t N = this->ivp_steps;
	this->frac_solver(&this->x_array, this->x0, t_array.back());
	return this->x_array;
}

vector<double> QDeformedMHCFrac::get_params_mhc()
{
	vector<double> pp;
	pp.push_back(this->xp);
	pp.push_back(this->xn);
	pp.push_back(this->Ap);
	pp.push_back(this->An);
	pp.push_back(this->vp);
	pp.push_back(this->vn);
	pp.push_back(this->gamma1);
	pp.push_back(this->gamma2);
	pp.push_back(this->delta1);
	pp.push_back(this->delta2);
	pp.push_back(this->x0);
	pp.push_back(this->lambda);
	pp.push_back(this->A);
	pp.push_back(this->alpha);
	return pp;
};

void QDeformedMHCFrac::set_params_mhc(vector<double>* pp)
{
	limit_params_mhc(pp);
	this->xp = pp->at(0);
	this->xn = pp->at(1);
	this->Ap = pp->at(2);
	this->An = pp->at(3);
	this->vp = pp->at(4);
	this->vn = pp->at(5);
	this->gamma1 = pp->at(6);
	this->gamma2 = pp->at(7);
	this->delta1 = pp->at(8);
	this->delta2 = pp->at(9);
	this->x0 = pp->at(10);
	this->lambda = pp->at(11);
	this->A = pp->at(12);
	this->alpha = pp->at(13);
};

void QDeformedMHCFrac::limit_params_mhc(vector<double>* pp)
{
	if (pp->at(0) < this->xp_min)
		pp->at(0) = xp_min;
	else if (pp->at(0) > this->xp_max)
		pp->at(0) = xp_max;
	if (pp->at(1) < this->xn_min)
		pp->at(1) = xn_min;
	else if (pp->at(1) > this->xn_max)
		pp->at(1) = xn_max;
	if (pp->at(2) < this->Ap_min)
		pp->at(2) = Ap_min;
	else if (pp->at(2) > this->Ap_max)
		pp->at(2) = Ap_max;
	if (pp->at(3) < this->An_min)
		pp->at(3) = An_min;
	else if (pp->at(3) > this->An_max)
		pp->at(3) = An_max;
	if (pp->at(4) < this->vp_min)
		pp->at(4) = vp_min;
	else if (pp->at(4) > this->vp_max)
		pp->at(4) = vp_max;
	if (pp->at(5) < this->vn_min)
		pp->at(5) = vn_min;
	else if (pp->at(5) > this->vn_max)
		pp->at(5) = vn_max;
	if (pp->at(6) < this->gamma1_min)
		pp->at(6) = gamma1_min;
	else if (pp->at(6) > this->gamma1_max)
		pp->at(6) = gamma1_max;
	if (pp->at(7) < this->gamma2_min)
		pp->at(7) = gamma2_min;
	else if (pp->at(7) > this->gamma2_max)
		pp->at(7) = gamma2_max;
	if (pp->at(8) < this->delta1_min)
		pp->at(8) = delta1_min;
	else if (pp->at(8) > this->delta1_max)
		pp->at(8) = delta1_max;
	if (pp->at(9) < this->delta2_min)
		pp->at(9) = delta2_min;
	else if (pp->at(9) > this->delta2_max)
		pp->at(9) = delta2_max;
	if (pp->at(10) < this->x0_min)
		pp->at(10) = x0_min;
	else if (pp->at(10) > this->x0_max)
		pp->at(10) = x0_max;
	if (pp->at(11) < this->lambda_min)
		pp->at(11) = lambda_min;
	else if (pp->at(11) > this->lambda_max)
		pp->at(11) = lambda_max;
	if (pp->at(12) < this->A_min)
		pp->at(12) = A_min;
	else if (pp->at(12) > this->A_max)
		pp->at(12) = A_max;
	if (pp->at(13) < this->alpha_min)
		pp->at(13) = alpha_min;
	else if (pp->at(13) > this->alpha_max)
		pp->at(13) = alpha_max;
};

void QDeformedMHCFrac::set_params_from_mhc()
{
	this->params.at("xp") = this->xp;
	this->params.at("xn") = this->xn;
	this->params.at("Ap") = this->Ap;
	this->params.at("An") = this->An;
	this->params.at("vp") = this->vp;
	this->params.at("vn") = this->vn;
	this->params.at("gamma1") = this->gamma1;
	this->params.at("gamma2") = this->gamma2;
	this->params.at("delta1") = this->delta1;
	this->params.at("delta2") = this->delta2;
	this->params.at("x0") = this->x0;
	this->params.at("lambda") = this->lambda;
	this->params.at("A") = this->A;
	this->params.at("alpha") = this->alpha;
};
