# include "hermite_rule_ss.h"

using namespace std;

//****************************************************************************80

int ma_in(int N, int option)

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for HERMITE_RULE.
//
//  Discussion:
//
//    This program computes a Gauss-Hermite quadrature rule 
//    and writes it to a file.
//
//    The user specifies:
//    * the ORDER (number of points) in the rule
//    * the OPTION (unweighted/physicist weight/probabilist weight)
//    * the root name of the OUTPUT files.
//
//    The parameter ALPHA is fixed to zero in this program.  If nonzero, a
//    generalized Gauss-Hermite quadrature rule would be generated, in which
//    the weight function would be multiplied by |x|^(2*ALPHA).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 May 2009
//
//  Author:
//
//    John Burkardt
//
{
	double alpha = 0.0;
	int order = N;
	string output = "hg_pols";

	cout << "\n";
	//timestamp();
	cout << "\n";
	cout << "HERMITE_RULE\n";
	cout << "  C++ version\n";
	cout << "\n";
	cout << "  Compiled on " << __DATE__ << " at " << __TIME__ << ".\n";
	cout << "\n";
	cout << "  Compute a Gauss-Hermite quadrature rule for approximating\n";
	cout << "\n";
	cout << "    Integral ( -oo < x < +oo ) w(x) f(x) dx\n";
	cout << "\n";
	cout << "  of order ORDER.\n";
	cout << "\n";
	cout << "  The user specifies ORDER, OPTION, and OUTPUT.\n";
	cout << "\n";
	cout << "  OPTION specifies the weight function w(x):\n";
	cout << "\n";
	cout << "    0, the unweighted rule for:\n";
	cout << "      Integral ( -oo < x < +oo )             f(x) dx\n";
	cout << "\n";
	cout << "    1, the physicist weighted rule for:\n";
	cout << "      Integral ( -oo < x < +oo ) exp(-x*x)   f(x) dx\n";
	cout << "\n";
	cout << "    2, the probabilist weighted rule for:\n";
	cout << "      Integral ( -oo < x < +oo ) exp(-x*x/2) f(x) dx\n";
	cout << "\n";
	cout << "  OUTPUT is:\n";
	cout << "\n";
	cout << "  \"C++\" for printed C++ output;\n";
	cout << "  \"F77\" for printed Fortran77 output;\n";
	cout << "  \"F90\" for printed Fortran90 output;\n";
	cout << "  \"MAT\" for printed MATLAB output;\n";
	cout << "\n";
	cout << "  or:\n";
	cout << "\n";
	cout << "  \"filename\" to generate 3 files:\n";
	cout << "\n";
	cout << "    filename_w.txt - the weight file\n";
	cout << "    filename_x.txt - the abscissa file.\n";
	cout << "    filename_r.txt - the region file.\n";

	cout << "\n";
	cout << "  ORDER =  " << order << "\n";
	cout << "  OPTION = " << option << "\n";
	cout << "  OUTPUT = \"" << output << "\".\n";
	//
	//  Construct the rule and output it.
	//
	hermite_handle(order, alpha, option, output);

	cout << "\n";
	cout << "HERMITE_RULE:\n";
	cout << "  Normal end of execution.\n";

	cout << "\n";
	//timestamp();

	return 0;
}
//****************************************************************************80

void hermite_compute(int order, double xtab[], double weight[])

//****************************************************************************80
//
//  Purpose:
//
//    HERMITE_COMPUTE computes a Gauss-Hermite quadrature rule.
//
//  Discussion:
//
//    The abscissas are the zeros of the N-th order Hermite polynomial.
//
//    The integration interval is ( -oo, +oo ).
//
//    The weight function is w(x) = exp ( - x*x ).
//
//    The integral to approximate:
//
//      Integral ( -oo < X < +oo ) exp ( - X*X ) * F(X) dX
//
//    The quadrature rule:
//
//      Sum ( 1 <= I <= ORDER ) WEIGHT(I) * F ( XTAB(I) )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    18 February 2008
//
//  Author:
//
//    Original FORTRAN77 version by Arthur Stroud, Don Secrest.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Arthur Stroud, Don Secrest,
//    Gaussian Quadrature Formulas,
//    Prentice Hall, 1966,
//    LC: QA299.4G3S7.
//
//  Parameters:
//
//    Input, int ORDER, the order of the formula to be computed.
//
//    Output, double XTAB[ORDER], the Gauss-Hermite abscissas.
//
//    Output, double WEIGHT[ORDER], the Gauss-Hermite weights.
//
{
	double cc;
	double dp2;
	int i;
	double p1;
	double s;
	double temp;
	double x;

	cc = 1.7724538509 * r8_gamma((double)(order))
		/ pow(2.0, order - 1);

	s = pow(2.0 * (double)(order)+1.0, 1.0 / 6.0);

	for (i = 0; i < (order + 1) / 2; i++)
	{
		if (i == 0)
		{
			x = s * s * s - 1.85575 / s;
		}
		else if (i == 1)
		{
			x = x - 1.14 * pow((double)(order), 0.426) / x;
		}
		else if (i == 2)
		{
			x = 1.86 * x - 0.86 * xtab[0];
		}
		else if (i == 3)
		{
			x = 1.91 * x - 0.91 * xtab[1];
		}
		else
		{
			x = 2.0 * x - xtab[i - 2];
		}

		hermite_root(&x, order, &dp2, &p1);

		xtab[i] = x;
		weight[i] = (cc / dp2) / p1;

		xtab[order - i - 1] = -x;
		weight[order - i - 1] = weight[i];
	}
	//
	//  Reverse the order of the abscissas.
	//
	for (i = 1; i <= order / 2; i++)
	{
		temp = xtab[i - 1];
		xtab[i - 1] = xtab[order - i];
		xtab[order - i] = temp;
	}

	return;
}
//****************************************************************************80

void hermite_handle(int order, double alpha, int option, string output)

//****************************************************************************80
//
//  Purpose:
//
//    HERMITE_HANDLE computes the requested Gauss-Hermite rule and outputs it.
//
//  Discussion:
//
//    This routine assumes that the input parameter ALPHA is zero, so that
//    a basic Gauss-Hermite rule is requested, not a generalized rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 May 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int ORDER, the order of the rule.
//
//    Input, double ALPHA, the parameter.
//
//    Input, int OPTION.
//    0, the integral has the form Integral ( -oo < x < +oo )             f(x) dx.
//    1, the integral has the form Integral ( -oo < x < +oo ) exp(-x*x)   f(x) dx
//    2, the integral has the form Integral ( -oo < x < +oo ) exp(-x*x/2) f(x) dx
//
//    Input, string OUTPUT, specifies the output.
//    * "C++'", print as C++ code.
//    * "F77", print as FORTRAN77 code.
//    * "F90", print as FORTRAN90 code.
//    * "MAT", print as MATLAB code.
//    * file,  write files "file_w.txt", "file_x.txt", "file_r.txt" 
//      defining weights, abscissas, and region.
// 
{
	int i;
	string output_r;
	string output_w;
	string output_x;
	double* r;
	double* w;
	double* x;

	r = new double[2];
	w = new double[order];
	x = new double[order];

	r[0] = -r8_huge();
	r[1] = r8_huge();

	hermite_compute(order, x, w);
	//
	//  Modify weights if requested.
	//
	if (option == 0)
	{
		for (i = 0; i < order; i++)
		{
			w[i] = exp(x[i] * x[i]) * w[i];
		}
	}
	else if (option == 2)
	{
		for (i = 0; i < order; i++)
		{
			w[i] = exp(x[i] * x[i] / 2.0) * w[i];
		}
	}

	if (output == "C++" || output == "c++")
	{
		cout << "//\n";
		cout << "//  Weights W, abscissas X and range R\n";
		cout << "//  for a Gauss-Hermite quadrature rule\n";
		cout << "//  ORDER = " << order << "\n";
		cout << "//\n";

		if (option == 0)
		{
			cout << "//  OPTION = 0, unweighted rule:\n";
			cout << "//    Integral ( -oo < x < +oo ) f(x) dx\n";
			cout << "//    is to be approximated by\n";
			cout << "//    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";

		}
		else if (option == 1)
		{
			cout << "//  OPTION = 1, physicist weighted rule:\n";
			cout << "//    Integral ( -oo < x < +oo ) exp(-x*x) f(x) dx\n";
			cout << "//    is to be approximated by\n";
			cout << "//    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		else if (option == 2)
		{
			cout << "//  OPTION = 2, probabilist weighted rule:\n";
			cout << "//    Integral ( -oo < x < +oo ) exp(-x*x/2) f(x) dx\n";
			cout << "//    is to be approximated by\n";
			cout << "//    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		cout << "//\n";

		for (i = 0; i < order; i++)
		{
			cout << "  w[" << i << "] = "
				<< setprecision(16) << w[i] << ";\n";
		}
		cout << "\n";
		for (i = 0; i < order; i++)
		{
			cout << "  x[" << i << "] = "
				<< setprecision(16) << x[i] << ";\n";
		}
		cout << "\n";
		for (i = 0; i < 2; i++)
		{
			cout << "  r[" << i << "] = " << r[i] << ";\n";
		}
	}
	else if (output == "F77" || output == "f77")
	{
		cout << "c\n";
		cout << "c  Weights W, abscissas X and range R\n";
		cout << "c  for a Gauss-Hermite quadrature rule\n";
		cout << "c  ORDER = " << order << "\n";
		cout << "c\n";
		if (option == 0)
		{
			cout << "c  OPTION = 0, unweighted rule:\n";
			cout << "c    Integral ( -oo < x < +oo ) f(x) dx\n";
			cout << "c    is to be approximated by\n";
			cout << "c    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		else if (option == 1)
		{
			cout << "c  OPTION = 1, physicist weighted rule:\n";
			cout << "c    Integral ( -oo < x < +oo ) exp(-x*x) f(x) dx\n";
			cout << "c    is to be approximated by\n";
			cout << "c    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		else if (option == 2)
		{
			cout << "c  OPTION = 2, probabilist weighted rule:\n";
			cout << "c    Integral ( -oo < x < +oo ) exp(-x*x/2) f(x) dx\n";
			cout << "c    is to be approximated by\n";
			cout << "c    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		cout << "c\n";

		for (i = 0; i < order; i++)
		{
			cout << "      w(" << i + 1 << ") = "
				<< setprecision(16) << w[i] << "\n";
		}
		cout << "\n";
		for (i = 0; i < order; i++)
		{
			cout << "      x(" << i + 1 << ") = "
				<< setprecision(16) << x[i] << "\n";
		}
		cout << "\n";
		for (i = 0; i < 2; i++)
		{
			cout << "      r(" << i + 1 << ") = " << r[i] << "\n";
		}
	}
	else if (output == "F90" || output == "f90")
	{
		cout << "!\n";
		cout << "!  Weights W, abscissas X and range R\n";
		cout << "!  for a Gauss-Hermite quadrature rule\n";
		cout << "!  ORDER = " << order << "\n";
		cout << "!\n";
		if (option == 0)
		{
			cout << "!  OPTION = 0, unweighted rule rule:\n";
			cout << "!    Integral ( -oo < x < +oo ) f(x) dx\n";
			cout << "!    is to be approximated by\n";
			cout << "!    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		else if (option == 1)
		{
			cout << "!  OPTION = 1, physicist weighted rule:\n";
			cout << "!    Integral ( -oo < x < +oo ) exp(-x*x) f(x) dx\n";
			cout << "!    is to be approximated by\n";
			cout << "!    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		else if (option == 2)
		{
			cout << "!  OPTION = 2, probabilist weighted rule:\n";
			cout << "!    Integral ( -oo < x < +oo ) exp(-x*x/2) f(x) dx\n";
			cout << "!    is to be approximated by\n";
			cout << "!    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		cout << "!\n";

		for (i = 0; i < order; i++)
		{
			cout << "  w(" << i + 1 << ") = "
				<< setprecision(16) << w[i] << "\n";
		}
		cout << "\n";
		for (i = 0; i < order; i++)
		{
			cout << "  x(" << i + 1 << ") = "
				<< setprecision(16) << x[i] << "\n";
		}
		cout << "\n";
		for (i = 0; i < 2; i++)
		{
			cout << "  r(" << i + 1 << ") = " << r[i] << "\n";
		}
	}
	else if (output == "MAT" || output == "mat")
	{
		cout << "%\n";
		cout << "%  Weights W, abscissas X and range R\n";
		cout << "%  for a Gauss-Hermite quadrature rule\n";
		cout << "%  ORDER = " << order << "\n";
		cout << "%\n";
		if (option == 0)
		{
			cout << "%  OPTION = 0, unweighted rule:\n";
			cout << "%    Integral ( -oo < x < +oo ) f(x) dx\n";
			cout << "%    is to be approximated by\n";
			cout << "%    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		else if (option == 1)
		{
			cout << "%  OPTION = 1, physicist weighted rule:\n";
			cout << "%    Integral ( -oo < x < +oo ) exp(-x*x) f(x) dx\n";
			cout << "%    is to be approximated by\n";
			cout << "%    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		else if (option == 2)
		{
			cout << "%  OPTION = 2, probabilist weighted rule:\n";
			cout << "%    Integral ( -oo < x < +oo ) exp(-x*x) f(x) dx\n";
			cout << "%    is to be approximated by\n";
			cout << "%    sum ( 1 <= I <= ORDER ) w(i) * f(x(i)).\n";
		}
		cout << "%\n";

		for (i = 0; i < order; i++)
		{
			cout << "  w(" << i + 1 << ") = "
				<< setprecision(16) << w[i] << ";\n";
		}
		cout << "\n";
		for (i = 0; i < order; i++)
		{
			cout << "  x(" << i + 1 << ") = "
				<< setprecision(16) << x[i] << ";\n";
		}
		cout << "\n";
		for (i = 0; i < 2; i++)
		{
			cout << "  r(" << i + 1 << ") = " << r[i] << ";\n";
		}
	}
	else
	{
		if (option == 0)
		{
			output_w = output + "_unweighted_w.txt";
			output_x = output + "_unweighted_x.txt";
			output_r = output + "_unweighted_r.txt";
		}
		else if (option == 1)
		{
			output_w = output + "_physicist_w.txt";
			output_x = output + "_physicist_x.txt";
			output_r = output + "_physicist_r.txt";
		}
		else if (option == 2)
		{
			output_w = output + "_probabilist_w.txt";
			output_x = output + "_probabilist_x.txt";
			output_r = output + "_probabilist_r.txt";
		}
		cout << "\n";
		cout << "  Creating quadrature files.\n";
		cout << "\n";
		cout << "  Root file name is     \"" << output << "\".\n";
		cout << "\n";
		cout << "  Weight file will be   \"" << output_w << "\".\n";
		cout << "  Abscissa file will be \"" << output_x << "\".\n";
		cout << "  Region file will be   \"" << output_r << "\".\n";

		r8mat_write(output_w, 1, order, w);
		r8mat_write(output_x, 1, order, x);
		r8mat_write(output_r, 1, 2, r);
	}

	delete[] r;
	delete[] w;
	delete[] x;

	return;
}
//****************************************************************************80

void hermite_handle(int order, double alpha, int option, vector<double>* output_x, vector<double>* output_w)

//****************************************************************************80
//
//  Purpose:
//
//    HERMITE_HANDLE computes the requested Gauss-Hermite rule and outputs it.
//
//  Discussion:
//
//    This routine assumes that the input parameter ALPHA is zero, so that
//    a basic Gauss-Hermite rule is requested, not a generalized rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 May 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int ORDER, the order of the rule.
//
//    Input, double ALPHA, the parameter.
//
//    Input, int OPTION.
//    0, the integral has the form Integral ( -oo < x < +oo )             f(x) dx.
//    1, the integral has the form Integral ( -oo < x < +oo ) exp(-x*x)   f(x) dx
//    2, the integral has the form Integral ( -oo < x < +oo ) exp(-x*x/2) f(x) dx
//
//    Input, string OUTPUT, specifies the output.
//    * "C++'", print as C++ code.
//    * "F77", print as FORTRAN77 code.
//    * "F90", print as FORTRAN90 code.
//    * "MAT", print as MATLAB code.
//    * file,  write files "file_w.txt", "file_x.txt", "file_r.txt" 
//      defining weights, abscissas, and region.
// 
{
	int i;
	double* w;
	double* x;

	w = new double[order];
	x = new double[order];

	hermite_compute(order, x, w);
	//
	//  Modify weights if requested.
	//
	output_x->clear();
	output_w->clear();

	for (int i = 0; i < order; i++)
	{
		output_x->push_back(x[i]);
		output_w->push_back(w[i]);
	}

	delete[] w;
	delete[] x;

	return;
}
//****************************************************************************80

void hermite_recur(double* p2, double* dp2, double* p1, double x, int order)

//****************************************************************************80
//
//  Purpose:
//
//    HERMITE_RECUR finds the value and derivative of a Hermite polynomial.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    20 February 2008
//
//  Author:
//
//    Original FORTRAN77 version by Arthur Stroud, Don Secrest.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Arthur Stroud, Don Secrest,
//    Gaussian Quadrature Formulas,
//    Prentice Hall, 1966,
//    LC: QA299.4G3S7.
//
//  Parameters:
//
//    Output, double *P2, the value of H(ORDER)(X).
//
//    Output, double *DP2, the value of H'(ORDER)(X).
//
//    Output, double *P1, the value of H(ORDER-1)(X).
//
//    Input, double X, the point at which polynomials are evaluated.
//
//    Input, int ORDER, the order of the polynomial to be computed.
//
{
	int i;
	double dq0;
	double dq1;
	double dq2;
	double q0;
	double q1;
	double q2;

	q1 = 1.0;
	dq1 = 0.0;

	q2 = x;
	dq2 = 1.0;

	for (i = 2; i <= order; i++)
	{
		q0 = q1;
		dq0 = dq1;

		q1 = q2;
		dq1 = dq2;

		q2 = x * q1 - 0.5 * ((double)(i)-1.0) * q0;
		dq2 = x * dq1 + q1 - 0.5 * ((double)(i)-1.0) * dq0;
	}

	*p2 = q2;
	*dp2 = dq2;
	*p1 = q1;

	return;
}
//****************************************************************************80

void hermite_root(double* x, int order, double* dp2, double* p1)

//****************************************************************************80
//
//  Purpose:
//
//    HERMITE_ROOT improves an approximate root of a Hermite polynomial.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 February 2008
//
//  Author:
//
//    Original FORTRAN77 version by Arthur Stroud, Don Secrest.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Arthur Stroud, Don Secrest,
//    Gaussian Quadrature Formulas,
//    Prentice Hall, 1966,
//    LC: QA299.4G3S7.
//
//  Parameters:
//
//    Input/output, double *X, the approximate root, which
//    should be improved on output.
//
//    Input, int ORDER, the order of the Hermite polynomial.
//
//    Output, double *DP2, the value of H'(ORDER)(X).
//
//    Output, double *P1, the value of H(ORDER-1)(X).
//
{
	double d;
	double eps;
	double p2;
	int step;
	int step_max = 10;

	eps = r8_epsilon();

	for (step = 1; step <= step_max; step++)
	{
		hermite_recur(&p2, dp2, p1, *x, order);

		d = p2 / (*dp2);
		*x = *x - d;

		if (r8_abs(d) <= eps * (r8_abs(*x) + 1.0))
		{
			return;
		}
	}

	return;
}
//****************************************************************************80

double r8_abs(double x)

//****************************************************************************80
//
//  Purpose:
//
//    R8_ABS returns the absolute value of an R8.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    14 November 2006
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, the quantity whose absolute value is desired.
//
//    Output, double R8_ABS, the absolute value of X.
//
{
	double value;

	if (0.0 <= x)
	{
		value = x;
	}
	else
	{
		value = -x;
	}
	return value;
}
//****************************************************************************80

double r8_epsilon()

//****************************************************************************80
//
//  Purpose:
//
//    R8_EPSILON returns the R8 roundoff unit.
//
//  Discussion:
//
//    The roundoff unit is a number R which is a power of 2 with the 
//    property that, to the precision of the computer's arithmetic,
//      1 < 1 + R
//    but 
//      1 = ( 1 + R / 2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    01 July 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_EPSILON, the R8 round-off unit.
//
{
	double value;

	value = 1.0;

	while (1.0 < (double)(1.0 + value))
	{
		value = value / 2.0;
	}

	value = 2.0 * value;

	return value;
}
//****************************************************************************80

double r8_gamma(double x)

//****************************************************************************80
//
//  Purpose:
//
//    R8_GAMMA evaluates Gamma(X) for a real argument.
//
//  Discussion:
//
//    This routine calculates the gamma function for a real argument X.
//
//    Computation is based on an algorithm outlined in reference 1.
//    The program uses rational functions that approximate the gamma
//    function to at least 20 significant decimal digits.  Coefficients
//    for the approximation over the interval (1,2) are unpublished.
//    Those for the approximation for 12 <= X are from reference 2.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    18 January 2008
//
//  Author:
//
//    Original FORTRAN77 version by William Cody, Laura Stoltz.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    William Cody,
//    An Overview of Software Development for Special Functions,
//    in Numerical Analysis Dundee, 1975,
//    edited by GA Watson,
//    Lecture Notes in Mathematics 506,
//    Springer, 1976.
//
//    John Hart, Ward Cheney, Charles Lawson, Hans Maehly,
//    Charles Mesztenyi, John Rice, Henry Thatcher,
//    Christoph Witzgall,
//    Computer Approximations,
//    Wiley, 1968,
//    LC: QA297.C64.
//
//  Parameters:
//
//    Input, double X, the argument of the function.
//
//    Output, double R8_GAMMA, the value of the function.
//
{
	//
	//  Coefficients for minimax approximation over (12, INF).
	//
	double c[7] = {
	 -1.910444077728E-03,
	  8.4171387781295E-04,
	 -5.952379913043012E-04,
	  7.93650793500350248E-04,
	 -2.777777777777681622553E-03,
	  8.333333333333333331554247E-02,
	  5.7083835261E-03 };
	double eps = 2.22E-16;
	double fact;
	double half = 0.5;
	int i;
	int n;
	double one = 1.0;
	double p[8] = {
	-1.71618513886549492533811E+00,
	 2.47656508055759199108314E+01,
	-3.79804256470945635097577E+02,
	 6.29331155312818442661052E+02,
	 8.66966202790413211295064E+02,
	-3.14512729688483675254357E+04,
	-3.61444134186911729807069E+04,
	 6.64561438202405440627855E+04 };
	bool parity;
	double pi = 3.1415926535897932384626434;
	double q[8] = {
	-3.08402300119738975254353E+01,
	 3.15350626979604161529144E+02,
	-1.01515636749021914166146E+03,
	-3.10777167157231109440444E+03,
	 2.25381184209801510330112E+04,
	 4.75584627752788110767815E+03,
	-1.34659959864969306392456E+05,
	-1.15132259675553483497211E+05 };
	double res;
	double sqrtpi = 0.9189385332046727417803297;
	double sum;
	double twelve = 12.0;
	double two = 2.0;
	double value;
	double xbig = 171.624;
	double xden;
	double xinf = 1.79E+308;
	double xminin = 2.23E-308;
	double xnum;
	double y;
	double y1;
	double ysq;
	double z;
	double zero = 0.0;;

	parity = false;
	fact = one;
	n = 0;
	y = x;
	//
	//  Argument is negative.
	//
	if (y <= zero)
	{
		y = -x;
		y1 = (double)(int)(y);
		res = y - y1;

		if (res != zero)
		{
			if (y1 != (double)(int)(y1 * half) * two)
			{
				parity = true;
			}

			fact = -pi / sin(pi * res);
			y = y + one;
		}
		else
		{
			res = xinf;
			value = res;
			return value;
		}
	}
	//
	//  Argument is positive.
	//
	if (y < eps)
	{
		//
		//  Argument < EPS.
		//
		if (xminin <= y)
		{
			res = one / y;
		}
		else
		{
			res = xinf;
			value = res;
			return value;
		}
	}
	else if (y < twelve)
	{
		y1 = y;
		//
		//  0.0 < argument < 1.0.
		//
		if (y < one)
		{
			z = y;
			y = y + one;
		}
		//
		//  1.0 < argument < 12.0.
		//  Reduce argument if necessary.
		//
		else
		{
			n = (int)(y)-1;
			y = y - (double)(n);
			z = y - one;
		}
		//
		//  Evaluate approximation for 1.0 < argument < 2.0.
		//
		xnum = zero;
		xden = one;
		for (i = 0; i < 8; i++)
		{
			xnum = (xnum + p[i]) * z;
			xden = xden * z + q[i];
		}
		res = xnum / xden + one;
		//
		//  Adjust result for case  0.0 < argument < 1.0.
		//
		if (y1 < y)
		{
			res = res / y1;
		}
		//
		//  Adjust result for case 2.0 < argument < 12.0.
		//
		else if (y < y1)
		{
			for (i = 1; i <= n; i++)
			{
				res = res * y;
				y = y + one;
			}
		}
	}
	else
	{
		//
		//  Evaluate for 12.0 <= argument.
		//
		if (y <= xbig)
		{
			ysq = y * y;
			sum = c[6];
			for (i = 0; i < 6; i++)
			{
				sum = sum / ysq + c[i];
			}
			sum = sum / y - y + sqrtpi;
			sum = sum + (y - half) * log(y);
			res = exp(sum);
		}
		else
		{
			res = xinf;
			value = res;
			return value;
		}
	}
	//
	//  Final adjustments and return.
	//
	if (parity)
	{
		res = -res;
	}

	if (fact != one)
	{
		res = fact / res;
	}

	value = res;

	return value;
}
//****************************************************************************80

double r8_huge()

//****************************************************************************80
//
//  Purpose:
//
//    R8_HUGE returns a "huge" R8.
//
//  Discussion:
//
//    The value returned by this function is NOT required to be the
//    maximum representable R8.  This value varies from machine to machine,
//    from compiler to compiler, and may cause problems when being printed.
//    We simply want a "very large" but non-infinite number.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    06 October 2007
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_HUGE, a "huge" R8 value.
//
{
	double value;

	value = 1.0E+30;

	return value;
}
//****************************************************************************80

void r8mat_write(string output_filename, int m, int n, double table[])

//****************************************************************************80
//
//  Purpose:
//
//    R8MAT_WRITE writes an R8MAT file with no header.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    29 June 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, string OUTPUT_FILENAME, the output filename.
//
//    Input, int M, the spatial dimension.
//
//    Input, int N, the number of points.
//
//    Input, double TABLE[M*N], the table data.
//
{
	int i;
	int j;
	ofstream output;
	//
	//  Open the file.
	//
	output.open(output_filename.c_str());

	if (!output)
	{
		cerr << "\n";
		cerr << "R8MAT_WRITE - Fatal error!\n";
		cerr << "  Could not open the output file.\n";
		return;
	}
	//
	//  Write the data.
	//
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < m; i++)
		{
			output << "  " << setw(24) << setprecision(16) << table[i + j * m];
		}
		output << "\n";
	}
	//
	//  Close the file.
	//
	output.close();

	return;
}
//****************************************************************************80

void timestamp()

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 July 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

	static char time_buffer[TIME_SIZE];
	struct std::tm* tm_ptr;
	std::time_t now;

	now = std::time(NULL);
	tm_ptr = new struct std::tm();
	//localtime_s(tm_ptr, &now);

	std::strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr);

	std::cout << time_buffer << "\n";

	return;
# undef TIME_SIZE
}
