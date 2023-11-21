#import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import math
import matplotlib.ticker as ticker

#constants
c = 299792458 #m/s

#function to read data from txt file
def read_data(filename):
    """
    Args:
        filename (_string_): .txt filename

    Returns:
        _array_: returns arrays containing name, redshift, effective peak magnitude, peak magnitude error data.
    """
    #reads all data into a matrix
    alldata = np.loadtxt(filename, dtype='str', comments='#') #2D matrix
    columns = []
    #for loop iterates through columns, appending each column to the columns list, which is returned once this process is complete.
    for i in range(len(alldata[0])):
        try:
            columns.append(np.array(alldata[:, i], dtype = float))
        except:
            columns.append(alldata[:, i])
    return columns



#function to calculate peak flux from effective peak magnitudes (and the respective errors)
def get_flux(eff_peak_mag, mag_err, m_0):
    peak_flux = 10**(0.4*(m_0-eff_peak_mag))*10**-7*10**4 # in Wm^-2
    #error calculated via functional approach.
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



#function to calculate and return an array of normalised residuals for a given dataset and model function.
def get_norm_residuals(model_params, model_funct, x_data, y_data, y_err):
    norm_residuals = []
    #iterates through all data, calculating each normalised residual, and adding them to a list of normalised residuals to be returned at the end.
    for i in range(0,len(x_data)):
        norm_residual = (y_data[i] - (model_funct(x_data[i], *model_params)))/y_err[i]
        norm_residuals.append(norm_residual)
    norm_residuals = np.array(norm_residuals)
    return norm_residuals



#calculates and returns an array of chi squared values for a dataset and model function. This array can be summed over to find the total chi-squared for a given fit.
def chisq(model_params, model_funct, x_data, y_data, y_err):
        chisqval = (get_norm_residuals(model_params, model_funct, x_data, y_data, y_err))**2
        return chisqval



#a function to carry out an automated chi-squared fitting - fits a given model function with unknown parameters to a given dataset
#by varying the unknown parameters in such a way as to minimise the chi-squared statistic.
#The value of the unknown parameters and their errors, as well as a plot showing the raw data, best fit line, and the normalised residuals and their distribution is also returned.
def automated_curve_fitting(xval, yval, yerr, model_funct, initial, xlabel, ylabel):
    #order arrays in ascending order
    order = np.argsort(xval)
    xval = xval[order]; yval = yval[order]; yerr = yerr[order]

    #calculates degrees of freedom.
    deg_freedom = xval.size - initial.size
    print('DoF = {}'.format(deg_freedom))

    #fits model function to dataset by varying unknown parameters. Returns both the best fit values, and a covariance matrix which is used to calculate the errors on these parameters.
    popt, cov = scipy.optimize.curve_fit(model_funct, # function to fit
                                        xval, # x data
                                        yval, # y data
                                        sigma=yerr, # set yerr as the array of error bars for the fit
                                        absolute_sigma=True, # errors bars DO represent 1 std error
                                        p0=initial, # starting point for fit
                                        check_finite=True) # raise ValueError if NaN encountered (don't allow errors to pass)

    print('Optimised parameters = ', popt, '\n')
    print('Covariance matrix = \n', cov)

    #plots the model function for both the initial parameters, and the optimised parameters after fitting.
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

    #to solve indexing errors
    popt = popt.tolist()

    #calculates chi-sq statistics
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

    #normalised residual plot
    plt.figure(1).add_axes((0,-0.25,0.8,0.25))
    plt.scatter(xval, norm_residuals, color='red', marker = 'd')
    plt.ylabel("""Normalised
Residuals""", weight = 'bold')
    plt.ylim(-3, 3)
    plt.axhline(y = 0, linestyle = '--', color = 'black')
    plt.axhline(y = 1, linestyle = ':', color = 'dimgrey'); plt.axhline(y = -1, linestyle = ':', color = 'dimgrey')
    plt.axhline(y = 2, linestyle = ':', color = 'lightgrey'); plt.axhline(y = -2, linestyle = ':', color = 'lightgrey')
    plt.xlabel(xlabel, weight = 'bold')

    #normalised residual distribution histogram.
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

    """#produce chisq contour plot
    if len(initial) == 2:
        a_low, a_high = popt[0]-0.5, popt[0]+0.5
        b_low, b_high = popt[1]-0.5, popt[1]+0.5

        # Generate grid and data
        da = (a_high - a_low)/100.0
        db = (b_high - b_low)/100.0
        a_axis = np.arange(a_low, a_high, da)
        b_axis = np.arange(b_low, b_high, db)
        plot_data = np.zeros((len(a_axis), len(b_axis)))
        for i, bval in enumerate(b_axis):
            for j, aval in enumerate(a_axis):
                plot_data[i][j] = np.sum(chisq([aval, bval], model_funct, xval, yval, yerr))
        X, Y = np.meshgrid(a_axis, b_axis, indexing='xy')
        contour_data = plot_data - chisq_min

        # Contour levels to plot - delta chi-squared of 1, 4 & 9 correspond to 1, 2 & 3 standard deviations
        levels = [1, 4, 9]
        C_im = plt.contour(X, Y, contour_data, levels = levels, colors='b', origin = 'lower')
        plt.clabel(C_im, levels, fontsize=12, inline=1, fmt=r'$\chi^2 = \chi^2_{min}+%1.0f$') 

        # Axis labels
        plt.xlabel('a (units?)')
        plt.ylabel('b (units?)')

        # This allows you to modify the tick markers to assess the errors from the chi-squared contour plots.
        #xtick_spacing = 5
        #ytick_spacing = 1
        #ax = plt.gca()
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))

        # Add in best fit point and dashed lines to axes
        plt.plot(popt[0], popt[1], 'ro')
        plt.plot((popt[0], popt[0]), (b_low, popt[1]), linestyle='--', color='r')
        plt.plot((a_low, popt[0]), (popt[1], popt[1]), linestyle='--', color='r')
        plt.show()"""

    #calculates errors on parameters by diagonalising the covariance matrix and square rooting the disgonal elements. These square rooted diagonal elements are the errors on the best fit parameters.
    #NOTE: this covariance matrix takes any correlation between parameter errors into account.
    popt_errs = np.sqrt(np.diag(cov))
    for i in range(len(popt)):
        print('optimised parameter[{}] = {} +/- {}'.format(i, popt[i], popt_errs[i]))
    
    #best fit parameters and their errors are returned.
    return popt, popt_errs



#a model function for magnitudes.
def mag_model(redshift, L_peak, H_0, m_0, omega_lambda0, omega_M0):
    flux_predicted = L_peak/(4*np.pi*(1+redshift)**2*(get_transverse_comoving_distance(redshift, H_0, omega_lambda0, omega_M0))**2) #W/m^2/Ang
    flux_predicted = flux_predicted * 10**7 * 10**-4 #erg/s/cm^2/Angstrom
    mag_predicted = m_0 - 2.5*np.log10(flux_predicted) 
    return mag_predicted



#the integrand in the comoving distance integral.
def comoving_distance_integrand(z, H_0, omega_lambda0, omega_M0):
    return ((c)/(H_0**2*(omega_M0*(1+z)**3+omega_lambda0)-(1+z)**2*H_0**2*(omega_lambda0+omega_M0-1))**0.5)



#integrates a function for an array of limits using the scipy.integrate.quad method.
def integrate_array(function, lower_limit_array, upper_limit_array, args):
    output = []
    assert len(lower_limit_array) == len(upper_limit_array)
    for i in range(0, len(upper_limit_array)):
        output.append(scipy.integrate.quad(function, lower_limit_array[i], upper_limit_array[i], args = args)[0])
    return np.array(output)



#ONLY VALID FOR MILESTONE! Calculates omega_lambda,0 abd its error using the trial value method rather than an automated chisq fitting.
def find_omega_lambda_and_error(x, actual, error, L_peak, H_0, m_0):
    #number of trial values.
    no_of_values = 10000
    #trial values.
    omega_lambdas = np.linspace(0, 1.1, no_of_values) #trial values for omega_lambda,0
    chi_squared = []
    #for loop calculates chi squared value for each trial value of omega_lambda,0.
    for omega_lambda in omega_lambdas:
        predicted = mag_model(x, L_peak, H_0, m_0, omega_lambda, 1-omega_lambda)
        chi_squared.append(np.sum((actual-predicted)**2/error**2))

    #calculates the index of the minimum chi-squared value in the chi_squared list.
    min_index = np.argmin(chi_squared)
    #this minimum chi-squared value corresponds to a certain trial value of omega_lambda,0.
    omega_lambda = omega_lambdas[min_index]
    #the actual minimum chi-squared statistic.
    min_chisq = chi_squared[min_index]
    print(min_chisq)

    #calculating the error.
    counter = 0
    errors = []
    #try and except is to stop algorithm returning an error when it goes above or below the maximum or minimum omega_lambda,0 trial values defined earlier.
    try:
        #this while loop moves from the minimum chisquared value in the positive direction until we reach delta chi-sq = +/- 1.
        while True:
            if chi_squared[min_index+counter]-min_chisq > 1:
                #if we reach delta chisq = +1, then the error is taken to be the difference between the omega_lambda,0 trial value at that point and the omega_lambda,0 trial value at the minimum chi-sq position.
                errors.append(omega_lambdas[min_index+counter]-omega_lambda)
                break
            else:
                #if we don't reach delta chisq = +1, counter ticks over to point to the next chisq value.
                counter += 1
    except:
        print('Error calculation exceeds maximum trial value!')

    #counter reset.
    counter = 0
    #try and except is to stop algorithm returning an error when it goes above or below the maximum or minimum omega_lambda,0 trial values defined earlier.
    try:
        #this while loop moves from the minimum chisquared value in the negative direction until we reach delta chi-sq = +/- 1.
        while True:
            if chi_squared[min_index-counter]-min_chisq > 1:
                #if we reach delta chisq = +1, then the error is taken to be the difference between the omega_lambda,0 trial value at that point and the omega_lambda,0 trial value at the minimum chi-sq position.
                errors.append(omega_lambdas[min_index-counter]-omega_lambda)
                break
            else:
                #if we don't reach delta chisq = +1, counter ticks over to point to the next chisq value.
                counter += 1
    except:
        print('Error calculation exceeds minimum trial value!')

    #rounding value and error to 3d.p. Error is taken to be the maximum of the error in the positive and negative directions.
    omega_lambda = np.round(omega_lambdas[min_index], 3)
    omega_lambda_err = np.round(np.max(errors), 3)

    #plotting chisq values against omega_lambda,0 trial values with vertical lines to indicate the positions of minimum chisq and delta chisq = +/- 1.
    plt.scatter(omega_lambdas, chi_squared, marker = 'x', s = 2, color = 'red')
    plt.xlabel('$Ω_{Λ,0}$')
    plt.ylabel('$χ^2$')
    plt.axvline(x = omega_lambda, color = 'black', linestyle = 'dashed')
    plt.axvline(x = omega_lambda + errors[0], color = 'grey', linestyle = 'dotted'); plt.axvline(x = omega_lambda + errors[1], color = 'grey', linestyle = 'dotted')
    plt.annotate('$Ω_{Λ,0}$' + f'= {omega_lambda} +/- {omega_lambda_err}', xycoords = 'axes fraction', xy = (0.05, 0.9), backgroundcolor = 'white')
    plt.show()

    #best fit value and error returned.
    return omega_lambda, omega_lambda_err



def get_transverse_comoving_distance(redshift, H_0, omega_lambda0, omega_M0):
    #the following try/except statement allows the function to deal with either an array of redshift values or a single redshift value being passed to it without throwing up an error.
    try:
        #calculates comoving distance for an array of redshift values.
        r_c = integrate_array(comoving_distance_integrand, np.zeros(len(redshift.tolist())), redshift, args=(H_0, omega_lambda0, omega_M0))
    except:
        #calculates comoving distance for a single redshift value.
        r_c = integrate_array(comoving_distance_integrand, [0], [redshift], args=(H_0, omega_lambda0, omega_M0))
    #determining whether k = -1, 0, or 1.
    k = np.sign(omega_lambda0+omega_M0-1)

    #Checks the value of k (universe geometry), and returns the relevant transverse comoving distance.
    if k == 0:
        return r_c
    elif k == 1:
        R_0 = ((k*c**2)/(H_0**2*(omega_lambda0 + omega_M0 - 1)))**0.5
        return R_0*np.sin(r_c/R_0)
    elif k == -1:
        R_0 = ((k*c**2)/(H_0**2*(omega_lambda0 + omega_M0 - 1)))**0.5
        return R_0*np.sinh(r_c/R_0)
    

