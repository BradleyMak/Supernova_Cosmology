import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import math

c = 299792458 #m/s

#function to read data from txt file
def read_data(filename):
    """
    Args:
        filename (_string_): .txt filename

    Returns:
        _array_: returns arrays containing name, redshift, effective peak magnitude, peak magnitude error data.
    """
    alldata = np.loadtxt(filename, dtype='str', comments='#') #2D matrix
    name = alldata[:, 0]
    redshift = np.array(alldata[:, 1], dtype = float)
    eff_peak_mag = np.array(alldata[:, 2], dtype = float)
    mag_err = np.array(alldata[:, 3], dtype = float)
    return name, redshift, eff_peak_mag, mag_err



#function to calculate peak flux from effective peak magnitudes (and the respective errors)
def get_flux(eff_peak_mag, mag_err, m_0):
    peak_flux = 10**(0.4*(m_0-eff_peak_mag))*10**-7*10**4 # in Wm^-2
    peak_flux_err = (10**(0.4*(m_0-(eff_peak_mag+mag_err)))*10**-7*10**4 - peak_flux) # in Wm^-2
    return peak_flux, peak_flux_err



#function to calculate the comoving distance for a given redshift (approximation for z<0.1)
def get_comoving_distance_low_z(redshift, H_0):
    comoving_distance = (c*redshift/H_0) #in m
    return comoving_distance



#function to calculate the luminosity distance and its error from peak luminosity and flux.
def get_luminosity_distance(redshift, comoving_distance):
    d_L = (1+redshift)*comoving_distance
    return d_L



def get_norm_residuals(model_params, model_funct, x_data, y_data, y_err):
    norm_residuals = []
    for i in range(0,len(x_data)):
        norm_residual = (y_data[i] - (model_funct(x_data[i], *model_params)))/y_err[i]
        norm_residuals.append(norm_residual)
    norm_residuals = np.array(norm_residuals)
    return norm_residuals



def chisq(model_params, model_funct, x_data, y_data, y_err):
        chisqval = (get_norm_residuals(model_params, model_funct, x_data, y_data, y_err))**2
        return chisqval



def automated_curve_fitting(xval, yval, yerr, model_funct, initial, xlabel, ylabel):
    #order arrays in ascending order
    order = np.argsort(xval)
    xval = xval[order]; yval = yval[order]; yerr = yerr[order]

    deg_freedom = xval.size - initial.size
    print('DoF = {}'.format(deg_freedom))

    popt, cov = scipy.optimize.curve_fit(model_funct, # function to fit
                                        xval, # x data
                                        yval, # y data
                                        sigma=yerr, # set yerr as the array of error bars for the fit
                                        absolute_sigma=True, # errors bars DO represent 1 std error
                                        p0=initial, # starting point for fit
                                        check_finite=True) # raise ValueError if NaN encountered (don't allow errors to pass)

    print('Optimised parameters = ', popt, '\n')
    print('Covariance matrix = \n', cov)

    plt.figure()
    plt.errorbar(xval, yval, yerr=yerr, marker='o', linestyle='None', capsize = 3, color = 'black')

    # Generate best fit line using model function and best fit parameters, and add to plot. 
    plt.plot(xval, 
                model_funct(xval, *popt),  # NOTE that now we need to 'unpack' our optimised parameters with '*'.
                'k', label='optimised', color = 'red')

    # We can also plot a calculation based on our initial conditions to ensure that something has actually happened!
    plt.plot(xval, 
                model_funct(xval, *initial), # We need to 'unpack' our initial parameters with '*'.
                'r', label='initial', color = 'black')
    plt.legend()
    plt.show()

    popt = popt.tolist()

    norm_residuals = get_norm_residuals(popt, model_funct, xval, yval, yerr)
    chisq_min = np.sum(chisq(popt, model_funct, xval, yval, yerr))
    print('chi^2_min = {}'.format(chisq_min))
    chisq_reduced = chisq_min/deg_freedom
    print('reduced chi^2 = {}'.format(chisq_reduced))
    P = scipy.stats.chi2.sf(chisq_min, deg_freedom)
    print('$P(chi^2_min, DoF)$ = {}'.format(P))

    #plotting final graph with data overlaid by best-fit curve, with normalised residuals and their probability density subplots.
    plt.figure(1)
    plt.figure(1).add_axes((0,0,0.8,0.8))
    plt.errorbar(xval, yval, yerr=yerr, marker='o', linestyle='None', color = 'black', capsize = 3)
    smooth_xval = np.linspace(xval[0], xval[-1], 1001)
    plt.plot(smooth_xval, model_funct(smooth_xval, *popt), color = 'red')
    plt.annotate('$χ^2_{min}$' + f'= {np.round(chisq_min, 2)}' + '\n' 
                 + f'DoF = {deg_freedom}' + '\n'
                 + '$χ^2_{red}$' + f'= {np.round(chisq_reduced, 2)}' + '\n' 
                 + f'P = {np.round(P, 3)}', 
                 xycoords = 'axes fraction', xy = (0.65, 0.07), backgroundcolor = 'white')
    plt.ylabel(ylabel, weight = 'bold')
    plt.gca().set_xticks([])

    plt.figure(1).add_axes((0,-0.25,0.8,0.25))
    plt.scatter(xval, norm_residuals, color='red', marker = 'd')
    plt.ylabel("""Normalised
Residuals""", weight = 'bold')
    plt.ylim(-3, 3)
    plt.axhline(y = 0, linestyle = '--', color = 'black')
    plt.axhline(y = 1, linestyle = ':', color = 'dimgrey'); plt.axhline(y = -1, linestyle = ':', color = 'dimgrey')
    plt.axhline(y = 2, linestyle = ':', color = 'lightgrey'); plt.axhline(y = -2, linestyle = ':', color = 'lightgrey')
    plt.xlabel(xlabel, weight = 'bold')

    plt.figure(1).add_axes((0.8,-0.25,0.15,0.25))
    plt.hist(norm_residuals, bins=np.arange(-100, 100, 0.5), alpha = 0.5, density = True, orientation = 'horizontal')
    plt.ylim(3, -3)
    plt.xticks([0.2,0.4], fontsize = 13)
    mu = 0; variance = 1
    sigma = math.sqrt(variance)
    y = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(stats.norm.pdf(y, mu, sigma), y, color = 'blue')
    plt.xlabel("""Probability
Density""", fontsize = 14, weight = 'bold')
    plt.gca().set_yticks([])
    plt.gca().invert_yaxis()

    plt.show()

    popt_errs = np.sqrt(np.diag(cov))
    for i in range(len(popt)):
        print('optimised parameter[{}] = {} +/- {}'.format(i, popt[i], popt_errs[i]))
    
    return popt, popt_errs



def mag_model(redshift, L_peak, H_0, m_0, k, omega_lambda0, omega_M0):
    flux_predicted = L_peak/(4*np.pi*(1+redshift)**2*(get_transverse_comoving_distance(redshift, H_0, k, omega_lambda0, omega_M0))**2) #W/m^2/Ang
    flux_predicted = flux_predicted * 10**7 * 10**-4 #erg/s/cm^2/Angstrom
    mag_predicted = m_0 - 2.5*np.log10(flux_predicted) 
    return mag_predicted



def comoving_distance_integrand(z, H_0, k, omega_lambda0, omega_M0):
    if k == 0:
        return ((c)/(H_0*((1-(1+z)**3)*omega_lambda0 + (1+z)**3)**0.5))
    elif k == 1 or k == -1:
        try:
            R_0_sq = ((k*c**2)/(H_0**2*(omega_lambda0 + omega_M0 - 1)))
            return ((c)/(H_0**2*(omega_M0*(1+z)**3+omega_lambda0)-k*c**2*(1+z)**2/R_0_sq**2)**0.5)
        except:
            return ((c)/(H_0*((1-(1+z)**3)*omega_lambda0 + (1+z)**3)**0.5))
    else:
        return ("k must equal 0, 1, or -1.")



def integrate_array(function, lower_limit_array, upper_limit_array, args):
    output = []
    assert len(lower_limit_array) == len(upper_limit_array)
    for i in range(0, len(upper_limit_array)):
        output.append(scipy.integrate.quad(function, lower_limit_array[i], upper_limit_array[i], args = args)[0])
    return np.array(output)



def find_omega_lambda_and_error(x, actual, error, L_peak, H_0, m_0):
    no_of_values = 10000
    omega_lambdas = np.linspace(0, 1.1, no_of_values) #trial values for omega_lambda,0
    chi_squared = []
    for omega_lambda in omega_lambdas:
        predicted = mag_model(x, L_peak, H_0, m_0, 0, omega_lambda, None)
        chi_squared.append(np.sum((actual-predicted)**2/error**2))

    min_index = np.argmin(chi_squared)
    omega_lambda = omega_lambdas[min_index]
    min_chisq = chi_squared[min_index]
    print(min_chisq)
    counter = 0
    errors = []
    try:
        while True:
            if chi_squared[min_index+counter]-min_chisq > 1:
                errors.append(omega_lambdas[min_index+counter]-omega_lambda)
                break
            else:
                counter += 1
    except:
        print('Error calculation exceeds maximum trial value!')

    counter = 0
    try:
        while True:
            if chi_squared[min_index-counter]-min_chisq > 1:
                errors.append(omega_lambdas[min_index-counter]-omega_lambda)
                break
            else:
                counter += 1
    except:
        print('Error calculation exceeds minimum trial value!')

    omega_lambda = np.round(omega_lambdas[min_index], 3)
    omega_lambda_err = np.round(np.max(errors), 3)

    plt.scatter(omega_lambdas, chi_squared, marker = 'x', s = 2, color = 'red')
    plt.xlabel('$Ω_{Λ,0}$')
    plt.ylabel('$χ^2$')
    plt.axvline(x = omega_lambda, color = 'black', linestyle = 'dashed')
    plt.axvline(x = omega_lambda + errors[0], color = 'grey', linestyle = 'dotted'); plt.axvline(x = omega_lambda + errors[1], color = 'grey', linestyle = 'dotted')
    plt.annotate('$Ω_{Λ,0}$' + f'= {omega_lambda} +/- {omega_lambda_err}', xycoords = 'axes fraction', xy = (0.05, 0.9), backgroundcolor = 'white')
    plt.show()

    return omega_lambda, omega_lambda_err



def get_transverse_comoving_distance(redshift, H_0, k, omega_lambda0, omega_M0):
    try:
        r_c = integrate_array(comoving_distance_integrand, np.zeros(len(redshift.tolist())), redshift, args=(H_0, k, omega_lambda0, omega_M0))
    except:
        r_c = integrate_array(comoving_distance_integrand, [0], [redshift], args=(H_0, k, omega_lambda0, omega_M0))
    if k == 0:
        omega_M0 = 1 - omega_lambda0
        return r_c
    elif k == 1:
        R_0 = ((k*c**2)/(H_0**2*(omega_lambda0 + omega_M0 - 1)))**0.5
        return R_0*np.sin(r_c/R_0)
    elif k == -1:
        R_0 = ((k*c**2)/(H_0**2*(omega_lambda0 + omega_M0 - 1)))**0.5
        return R_0*np.sinh(r_c/R_0)
    else:
        print("Invalid value of k encountered - k = -1, 0, or 1 only.")
    

