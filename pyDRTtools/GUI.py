# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Baptiste Py, Ting Hei Wan, Adeleke Maradesa'

__date__ = '20th August 2023'

# import sys
# import csv
# import numpy as np
# from numpy import log10, absolute, angle
# from PyQt5 import QtGui, QtWidgets
# from PyQt5.QtWidgets import QFileDialog
# from . import layout
# #from .pyDRTtools_layout import Ui_MainWindow
# #from runs import *
# #from runs import *
# from .runs import *
# #from .runs import EIS_Object, simple_run, Bayesian_run,BHT_run,peak_analysis
# import matplotlib as mpl
# mpl.use("Qt5Agg")
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# class GUI(QtWidgets.QMainWindow):
#     def __init__(self):
    
#         # Initalize parent
#         QtWidgets.QMainWindow.__init__(self)
        
#         # Initalize GUI layout
#         self.ui = layout.Ui_MainWindow()  # layout
#         self.ui.setupUi(self)
       
#         # setting data to be initially none
#         self.data = None
       
#         # linking buttons with various functions
#         # import button
#         self.ui.import_button.clicked.connect(self.import_file)
#         self.ui.induct_choice.currentIndexChanged.connect(self.inductance_callback) # activated when item change
        
#         # show buttons
#         self.ui.show_EIS.clicked.connect(lambda: self.plotting_callback('EIS_data'))
#         self.ui.show_mag.clicked.connect(lambda: self.plotting_callback('Magnitude'))
#         self.ui.show_phase.clicked.connect(lambda: self.plotting_callback('Phase'))
#         self.ui.show_re.clicked.connect(lambda: self.plotting_callback('Re_data'))
#         self.ui.show_im.clicked.connect(lambda: self.plotting_callback('Im_data'))
#         self.ui.show_re_res.clicked.connect(lambda: self.plotting_callback('Re_residual'))
#         self.ui.show_im_res.clicked.connect(lambda: self.plotting_callback('Im_residual'))
#         self.ui.show_DRT.clicked.connect(lambda: self.plotting_callback('DRT_data'))
#         self.ui.show_score.clicked.connect(lambda: self.plotting_callback('Score'))
        
#         # run buttons
#         self.ui.simple_run_button.clicked.connect(self.simple_run_callback)
#         self.ui.bayesian_button.clicked.connect(self.bayesian_run_callback)
#         self.ui.HT_button.clicked.connect(self.BHT_run_callback)
#         self.ui.peak_decon_button.clicked.connect(self.peak_analysis_run_callback)
        
#         # export result buttons
#         self.ui.export_DRT_button.clicked.connect(self.export_DRT)
#         self.ui.export_EIS_button.clicked.connect(self.export_EIS)
#         self.ui.export_fig_button.clicked.connect(self.export_fig)
       
#     def import_file(self):
        
#         # file dialog pop up, the users can choose a csv or txt file
#         path, ext = QFileDialog.getOpenFileName(None, "Please choose a file", "", "All Files (*);; CSV files (*.csv);; TXT files (*.txt)") 
        
#         # return if there is no path or the file format is not correct
#         if not (path.endswith('.csv') or path.endswith('.txt')):
#             print('return')
#             return
            
#         self.data = EIS_object.from_file(path)   ######### Here
        
#         # discard inductance data if necessary
#         self.inductance_callback()
#         self.statusBar().showMessage('Imported file: %s' % path, 1000)
    
#     def inductance_callback(self): 
        
#         if self.data is None:
#             return
        
#         elif self.ui.induct_choice.currentIndex() == 2: # discard inductive data
#             self.data.freq = self.data.freq[-self.data.Z_double_prime>0]
#             self.data.Z_prime = self.data.Z_prime[-self.data.Z_double_prime>0]
#             self.data.Z_double_prime = self.data.Z_double_prime[-self.data.Z_double_prime>0]
#             self.data.Z_exp = self.data.Z_prime + self.data.Z_double_prime*1j
            
#         else: # use original data otherwise
#             self.data.freq = self.data.freq_0
#             self.data.Z_prime = self.data.Z_prime_0
#             self.data.Z_double_prime = self.data.Z_double_prime_0
#             self.data.Z_exp = self.data.Z_exp_0
        
#         self.data.tau = 1/self.data.freq # we assume that the collocation points are equal to 1/freq as default
#         self.data.tau_fine  = np.logspace(log10(self.data.tau.min())-0.5,log10(self.data.tau.max())+0.5,10*self.data.freq.shape[0])
#         self.data.method = 'none'
        
#         self.plotting_callback('EIS_data')
    
#     def simple_run_callback(self): # callback for simple ridge regularization
        
#         if self.data is None:
#             return
        
#         # we first read the parameters chosen by the users
#         rbf_type = str(self.ui.discre_choice.currentText())
#         data_used = str(self.ui.data_used_choice.currentText())
#         induct_used = int(self.ui.induct_choice.currentIndex())
#         der_used = str(self.ui.der_choice.currentText())
#         cv_type = str(self.ui.lambda_choice.currentText())
#         reg_param = float(self.ui.reg_param_entry.text())
#         shape_control = str(self.ui.shape_control_choice.currentText())
#         coeff = float(self.ui.FWHM_entry.text())
        
#         # we perform the computation
#         self.data = simple_run(self.data, rbf_type = rbf_type, data_used = data_used, induct_used = induct_used,
#                                 der_used = der_used, cv_type = cv_type, reg_param = reg_param, shape_control = shape_control, coeff = coeff)
#         self.plotting_callback('DRT_data')

#     def bayesian_run_callback(self): # callback for Bayesian regularization
        
#         if self.data is None:
#             return
        
#         # we first read the parameters chosen by the users
#         rbf_type = str(self.ui.discre_choice.currentText())
#         data_used = str(self.ui.data_used_choice.currentText())
#         induct_used = int(self.ui.induct_choice.currentIndex())
#         der_used = str(self.ui.der_choice.currentText())
#         cv_type = str(self.ui.lambda_choice.currentText())
#         reg_param = float(self.ui.reg_param_entry.text())
#         shape_control = str(self.ui.shape_control_choice.currentText())
#         coeff = float(self.ui.FWHM_entry.text())
#         sample_number = int(self.ui.sample_no_entry.text())
        
#         # we perform the computation
#         self.data = Bayesian_run(self.data, rbf_type = rbf_type, data_used = data_used, induct_used = induct_used, 
#                                  der_used = der_used, cv_type = cv_type, reg_param = reg_param, shape_control = shape_control,
#                                  coeff = coeff, NMC_sample = sample_number)        
#         self.plotting_callback('DRT_data')
        
#     def BHT_run_callback(self): # callback for Hilbert transform run
        
#         if self.data is None:
#             return
        
#         # we first read the parameters chosen by the users
#         rbf_type = str(self.ui.discre_choice.currentText())
#         der_used = str(self.ui.der_choice.currentText())
#         shape_control = str(self.ui.shape_control_choice.currentText())
#         coeff = float(self.ui.FWHM_entry.text()) 
        
#         # we perform the computation
#         self.data = BHT_run(self.data, rbf_type, der_used, shape_control, coeff)
#         self.plotting_callback('DRT_data')
    
#     def peak_analysis_run_callback(self): # callback for peak analysis
        
#         if self.data is None:
#             return
        
#         # we first read the parameters chosen by the users
#         rbf_type = str(self.ui.discre_choice.currentText())
#         data_used = str(self.ui.data_used_choice.currentText())
#         induct_used = int(self.ui.induct_choice.currentIndex())
#         der_used = str(self.ui.der_choice.currentText())
#         cv_type = str(self.ui.lambda_choice.currentText())
#         reg_param = float(self.ui.reg_param_entry.text())
#         shape_control = str(self.ui.shape_control_choice.currentText())
#         coeff = float(self.ui.FWHM_entry.text())
#         peak_method = str(self.ui.peak_method_choice.currentText())
#         N_peaks = float(self.ui.peak_num_entry.text())
        
#         # we perform the computation
#         self.data = peak_analysis(self.data,rbf_type = rbf_type, data_used = data_used, induct_used = induct_used, 
#                         der_used = der_used, cv_type = cv_type, reg_param = reg_param, shape_control = shape_control, coeff = coeff, peak_method=peak_method, N_peaks=N_peaks)
#         self.plotting_callback('DRT_data')
        
#     def plotting_callback(self, plot_to_show):
        
#         # not plotting if no data has been imported
#         if self.data == None:
#             return
        
#         # initalize the figure object
#         fig = Figure_Canvas()
        
#         # rendering the figure object on to the gui
#         fig_to_plot = getattr(fig,plot_to_show)
#         fig_to_plot(self.data)
#         graphicscene = QtWidgets.QGraphicsScene(20, 25, 650, 610)
#         graphicscene.addWidget(fig)
#         self.ui.plot_panel.setScene(graphicscene)
#         self.ui.plot_panel.show()
    
#     def export_DRT(self): # callback for exporting the DRT results
        
#         # return None if the users have not conducted any computation
#         if self.data == None:
#             return
        
#         # select path to save the results
#         path, ext = QFileDialog.getSaveFileName(None, "Please directory to save the DRT result", 
#                                                 "", "CSV files (*.csv);; TXT files (*.txt)")
        
#         if self.data.method == 'simple':
#             with open(path, 'w', newline='') as save_file:
#                 writer = csv.writer(save_file)
                
#                 # first save L and R
#                 writer.writerow(['L', self.data.L])
#                 writer.writerow(['R', self.data.R])
#                 writer.writerow(['lambda_value', self.data.lambda_value])
#                 writer.writerow(['tau','gamma'])
                
#                 # after that, save tau and gamma
#                 for n in range(self.data.out_tau_vec.shape[0]):
#                     writer.writerow([self.data.out_tau_vec[n], self.data.gamma[n]])
            
#         elif self.data.method == 'credit':
#             with open(path, 'w', newline='') as save_file:
#                 writer = csv.writer(save_file)
                
#                 # first save L and R
#                 writer.writerow(['L', self.data.L])
#                 writer.writerow(['R', self.data.R])
#                 writer.writerow(['tau','MAP','Mean','Upperbound','Lowerbound'])
                
#                 # after that, save tau, gamma, mean, upper bound, and lower bound
#                 for n in range(self.data.out_tau_vec.shape[0]):
#                     writer.writerow([self.data.out_tau_vec[n], self.data.gamma[n],
#                                     self.data.mean[n], self.data.upper_bound[n], 
#                                     self.data.lower_bound[n]] )
                        
#         elif self.data.method == 'BHT':
#             with open(path, 'w', newline='') as save_file:
#                 writer = csv.writer(save_file)
                
#                 # first save L and R
#                 writer.writerow(['L', self.data.mu_L_0])
#                 writer.writerow(['R', self.data.mu_R_inf])
#                 writer.writerow(['tau','gamma_Re','gamma_Im'])
                
#                 # after that, save tau, gamma_re, and gamma_im 
#                 for n in range(self.data.out_tau_vec.shape[0]):
#                     writer.writerow([self.data.out_tau_vec[n], 
#                                      self.data.mu_gamma_fine_re[n],
#                                      self.data.mu_gamma_fine_im[n]])
    
#     def export_EIS(self): # callback for exporting the EIS fitting results
        
#         # return None if the users have not conducted any computation
#         if self.data == None:
#             return
        
#         # select path to save the result
#         path, ext = QFileDialog.getSaveFileName(None, "Please directory to save the EIS fitting result", 
#                                                 "", "CSV files (*.csv);; TXT files (*.txt)")
                        
#         if self.data.method == 'BHT': # save result for BHT
#             with open(path, 'w', newline='') as save_file:
#                 writer = csv.writer(save_file)
#                 writer.writerow(['s_res_re', self.data.out_scores['s_res_re'][0],
#                                   self.data.out_scores['s_res_re'][1],
#                                   self.data.out_scores['s_res_re'][2]])
#                 writer.writerow(['s_res_im', self.data.out_scores['s_res_im'][0],
#                                   self.data.out_scores['s_res_im'][1],
#                                   self.data.out_scores['s_res_im'][2]])
#                 writer.writerow(['s_mu_re', self.data.out_scores['s_mu_re']])
#                 writer.writerow(['s_mu_im', self.data.out_scores['s_mu_im']])
#                 writer.writerow(['s_HD_re', self.data.out_scores['s_HD_re']])
#                 writer.writerow(['s_HD_im', self.data.out_scores['s_HD_im']])
#                 writer.writerow(['s_JSD_re', self.data.out_scores['s_JSD_re']])
#                 writer.writerow(['s_JSD_im', self.data.out_scores['s_JSD_im']])
#                 writer.writerow(['freq', 'mu_Z_re','mu_Z_im','mu_H_re','mu_H_im',
#                                  'Z_H_re_band','Z_H_im_band','Z_H_re_res','Z_H_im_res'])
#                 # save frequency, the fitted impedance and the residual
#                 for n in range(self.data.freq.shape[0]):
#                     writer.writerow([self.data.freq[n], self.data.mu_Z_re[n],
#                                      self.data.mu_Z_im[n], self.data.mu_Z_H_im_agm[n],
#                                      self.data.band_re_agm[n], self.data.band_im_agm[n],
#                                      self.data.res_H_re[n], self.data.res_H_im[n]])
                
#         else: # save result for simple and bayesian run
#             with open(path, 'w', newline='') as save_file:
#                 writer = csv.writer(save_file)
#                 writer.writerow(['freq','mu_Z_re','mu_Z_im','Z_re_res','Z_im_res'])
#                 # save frequency, the fitted impedance and the residual
#                 for n in range(self.data.freq.shape[0]):
#                     writer.writerow([self.data.freq[n], self.data.mu_Z_re[n],
#                                      self.data.mu_Z_im[n], self.data.res_re[n],
#                                      self.data.res_im[n]])
    
#     def export_fig(self): # export the figures as png
        
#         # return if users have not conducted any computation
#         if self.data == None:
#             return
        
#         file_choices = "PNG (*.png)"        
#         path, ext = QFileDialog.getSaveFileName(self,'Save file', '', file_choices)
        
#         pixmap = QtGui.QPixmap(self.ui.plot_panel.viewport().size())
#         self.ui.plot_panel.viewport().render(pixmap)
#         pixmap.save(path)
        
#         if path:
#             self.statusBar().showMessage('Saved to %s' % path, 1000)
            

# class Figure_Canvas(FigureCanvas):
    
#     def __init__(self, parent=None, width=7.5, height=6.5, dpi=100): # create Figure under matplotlib.pyplot
        
#         # deactivate the popping of another figure panel
#         plt.ioff()
        
#         # plt.rc('text', usetex=True)
#         plt.rc('font', family='serif', size = 20)
#         plt.rc('xtick', labelsize = 15)
#         plt.rc('ytick', labelsize = 15)
#         fig = plt.figure(figsize=(width, height), dpi=dpi) 
        
#         # initalizing parent
#         FigureCanvas.__init__(self, fig) 
#         self.setParent(parent)
#         self.axes = fig.add_subplot(111) # using add_subplot method
#         fig.tight_layout()
#         fig.subplots_adjust(left=0.14, bottom=0.13, right=0.9, top=0.9)
        
#     def EIS_data(self, entry): # plot the data
        
#         if entry.method == 'BHT': # for BHT run
#             self.axes.plot(entry.mu_Z_re, -entry.mu_Z_im, 
#                            'k', label = '$Z_\mu$(Regressed)', linewidth=3)
#             self.axes.plot(entry.mu_Z_H_re_agm, -entry.mu_Z_H_im_agm, 
#                            'b', label = '$Z_H$(Hilbert transform)', linewidth=3)
#             self.axes.plot(entry.Z_prime, -entry.Z_double_prime, 'or')
#             self.axes.legend(frameon = False, fontsize = 15, loc = 'upper left')
            
#         elif entry.method != 'none': # for simple or Bayesian run
#             self.axes.plot(entry.mu_Z_re, -entry.mu_Z_im, 'k', linewidth=2)
#             self.axes.plot(entry.Z_prime, -entry.Z_double_prime, 'or')
        
#         else: # no computation of the imported data yet
#             self.axes.plot(entry.Z_prime, -entry.Z_double_prime, 'or')
        
#         self.axes.set_xlabel('$Z^{\prime}/\Omega$')
#         self.axes.set_ylabel('-$Z^{\prime \prime}/\Omega$')
#         self.axes.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
#         self.axes.axis('equal')

#     def Magnitude(self, entry): # plot the magnitude
        
#         if entry.method == 'BHT': # for BHT run
#             self.axes.semilogx(entry.freq, absolute(entry.mu_Z_re+1j*entry.mu_Z_im), 
#                                'k', label = '$Z_\mu$(Regressed)', linewidth=2)
#             self.axes.semilogx(entry.freq, absolute(entry.mu_Z_H_re_agm+1j*entry.mu_Z_H_im_agm),
#                                'b', label = '$Z_H$(Hilbert transform)', linewidth=2)
#             self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')
#             self.axes.legend(frameon = False, fontsize = 15, loc = 'upper left')
            
#         elif entry.method != 'none': # for simple or Bayesian run
#             self.axes.semilogx(entry.freq, absolute(entry.mu_Z_re+1j*entry.mu_Z_im), 'k', linewidth=3)
#             self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')
            
#         else: # no computation of the imported data yet
#             self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')
        
#         self.axes.set_xlim([min(entry.freq), max(entry.freq)])    
#         self.axes.set_xlabel('$f/Hz$')
#         self.axes.set_ylabel('$|Z|/\Omega$')
        
#     def Phase(self, entry): # plot the phase
        
#         if entry.method == 'BHT': # for BHT run
#             self.axes.semilogx(entry.freq, angle(entry.mu_Z_re+1j*entry.mu_Z_im, deg = True),
#                                'k', label = '$Z_\mu$(Regressed)', linewidth=2)
#             self.axes.semilogx(entry.freq, angle(entry.mu_Z_H_re_agm+1j*entry.mu_Z_H_im_agm, deg = True),
#                                'b', label = '$Z_H$(Hilbert transform)', linewidth=2)
#             self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg = True), 'or')
#             self.axes.legend(frameon = False, fontsize = 15, loc = 'upper left')
        
#         elif entry.method != 'none': # for simple or Bayesian run
#             self.axes.semilogx(entry.freq, angle(entry.mu_Z_re+1j*entry.mu_Z_im, deg = True), 'k', linewidth=3)
#             self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg = True), 'or')
            
#         else: # no computation of the imported data yet
#             self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg = True), 'or')
        
#         self.axes.set_xlim([min(entry.freq), max(entry.freq)])    
#         self.axes.set_xlabel('$f/Hz$')
#         self.axes.set_ylabel('$angle/^\circ$')
        
#     def Re_data(self, entry): # plot the real part
        
#         if entry.method == 'BHT': # for BHT run
#             self.axes.fill_between(entry.freq, entry.mu_Z_H_re_agm-3*entry.band_re_agm, entry.mu_Z_H_re_agm+3*entry.band_re_agm,  facecolor='lightgrey')
#             self.axes.semilogx(entry.freq, entry.mu_Z_re, 'k', label = '$Z_\mu$(Regressed)', linewidth=3)
#             self.axes.semilogx(entry.freq, entry.mu_Z_H_re_agm, 'b', label = '$Z_H$(Hilbert transform)', linewidth=3)
#             self.axes.semilogx(entry.freq, entry.Z_prime, 'or')
#             self.axes.legend(frameon = False, fontsize = 15, loc = 'upper left')
        
#         elif entry.method != 'none': # for simple or Bayesian run
#             self.axes.semilogx(entry.freq, entry.mu_Z_re, 'k', linewidth=3)
#             self.axes.semilogx(entry.freq, entry.Z_prime, 'or')
            
#         else: # no computation of the imported data yet
#             self.axes.semilogx(entry.freq, entry.Z_prime, 'or')
        
#         self.axes.set_xlim([min(entry.freq), max(entry.freq)])    
#         self.axes.set_xlabel('$f/Hz$')
#         self.axes.set_ylabel('$Z^{\prime}/\Omega$')
        
#     def Im_data(self, entry): # plot of the imaginary part
        
#         if entry.method == 'BHT': # for BHT run
#             self.axes.fill_between(entry.freq, -entry.mu_Z_H_im_agm-3*entry.band_im_agm, -entry.mu_Z_H_im_agm+3*entry.band_im_agm,  facecolor='lightgrey')
#             self.axes.semilogx(entry.freq, -entry.mu_Z_im, 'k', label = '$Z_\mu$(Regressed)', linewidth=3)
#             self.axes.semilogx(entry.freq, -entry.mu_Z_H_im_agm, 'b', label = '$Z_H$(Hilbert transform)', linewidth=3)
#             self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')
#             self.axes.legend(frameon = False, fontsize = 15, loc = 'upper left')
            
#         elif entry.method != 'none':# for simple or bayesian run
#             self.axes.semilogx(entry.freq, -entry.mu_Z_im, 'k')
#             self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')
        
#         else: # no computation of the imported data yet
#             self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')
        
#         self.axes.set_xlim([min(entry.freq), max(entry.freq)])    
#         self.axes.set_xlabel('$f/Hz$')
#         self.axes.set_ylabel('-$Z^{\prime \prime}/\Omega$')
        
#     def Re_residual(self, entry): # plot the residuals of the real part of the impedance
        
#         if entry.method == 'none' : # or str(entry.ui.data_used_choice.currentText()) == 'Im Data':
#             return
        
#         elif entry.method == 'BHT': # for BHT run
#             self.axes.fill_between(entry.freq, -3*entry.band_re_agm, 3*entry.band_re_agm,  facecolor='lightgrey')
#             self.axes.semilogx(entry.freq, entry.res_H_re, 'or')
#             self.axes.set_xlabel('$f/Hz$')
#             self.axes.set_ylabel('$R_{\infty}+Z^{\prime}_{H}-Z^{\prime}_{exp}/\Omega$')
#             y_max = max(3*entry.band_re_agm)
        
#         else: # for simple or Bayesian run
#             self.axes.semilogx(entry.freq, entry.res_re, 'or')
#             self.axes.set_xlabel('$f/Hz$')
#             self.axes.set_ylabel('$Z^{\prime}_{DRT}-Z^{\prime}_{exp}/\Omega$')
#             y_max = max(absolute(entry.res_re))
        
#         self.axes.set_xlim([min(entry.freq), max(entry.freq)])    
#         self.axes.set_ylim([-1.1*y_max, 1.1*y_max])
#         self.axes.ticklabel_format(axis= 'y', style="sci", scilimits=(0,0))
          
#     def Im_residual(self, entry): # plot the residuals of the imaginary part of the impedance
        
#         if entry.method == 'none' : # or str(entry.ui.data_used_choice.currentText()) == 'Re Data':
#             return
        
#         elif entry.method == 'BHT': # for BHT run
#             self.axes.fill_between(entry.freq, -3*entry.band_im_agm, 3*entry.band_im_agm,  facecolor='lightgrey')
#             self.axes.semilogx(entry.freq, entry.res_H_im, 'or')
#             self.axes.set_xlabel('$f/Hz$')
#             self.axes.set_ylabel('$\omega L_0+Z^{\prime \prime}_{H}-Z^{\prime\prime}_{exp}/\Omega$')
#             y_max = max(3*entry.band_im_agm)
            
#         else: # for simple or Bayesian run
#             self.axes.semilogx(entry.freq, entry.res_im, 'or')
#             self.axes.set_xlabel('$f/Hz$')
#             self.axes.set_ylabel('$Z^{\prime \prime}_{DRT}-Z^{\prime \prime}_{exp}/\Omega$')
#             y_max = max(absolute(entry.res_im))
            
#         self.axes.set_xlim([min(entry.freq), max(entry.freq)])    
#         self.axes.set_ylim([-1.1*y_max, 1.1*y_max])
#         self.axes.ticklabel_format(axis = 'y', style="sci", scilimits=(0,0))
        
#     def DRT_data(self, entry): # plot the DRT
        
#         if entry.method == 'none':
#             return
        
#         elif entry.method == 'simple':
#             self.axes.semilogx(entry.out_tau_vec, entry.gamma, 'k', linewidth=3)
#             y_min = 0
#             y_max = max(entry.gamma)
            
#         elif entry.method == 'credit':
#             self.axes.fill_between(entry.out_tau_vec, entry.lower_bound, entry.upper_bound,  facecolor='lightgrey')
#             self.axes.semilogx(entry.out_tau_vec, entry.gamma, color='black', label='MAP', linewidth=3)
#             self.axes.semilogx(entry.out_tau_vec, entry.mean, color='blue', label='mean', linewidth=3)
#             self.axes.semilogx(entry.out_tau_vec, entry.lower_bound, color='black')
#             self.axes.semilogx(entry.out_tau_vec, entry.upper_bound, color='black')
#             self.axes.legend(frameon = False, fontsize = 15)
#             y_min = 0
#             y_max = max(entry.upper_bound)
            
#         elif entry.method == 'BHT':
#             self.axes.semilogx(entry.out_tau_vec, entry.mu_gamma_fine_re, 'b', linewidth = 3, label = '$Mean Re$')
#             self.axes.semilogx(entry.out_tau_vec, entry.mu_gamma_fine_im, 'k', linewidth = 3, label = '$Mean Im$')
#             y_min = min(np.concatenate((entry.mu_gamma_fine_re,entry.mu_gamma_fine_im)))
#             y_max = max(np.concatenate((entry.mu_gamma_fine_re,entry.mu_gamma_fine_im)))
            
#         elif entry.method == 'peak':
            
#             self.axes.semilogx(entry.out_tau_vec, entry.gamma, color='black', linewidth=3)
#             color = ['red', 'green', 'cyan', 'yellow', 'orange', 'blue', 'grey', 'brown', 'coral', 'darkblue', 'darkgreen', 'gold']
            
#             if len(entry.out_gamma_fit)==entry.N_peaks: # separate fit of the peaks
#                 for i in range(entry.N_peaks):
#                     self.axes.semilogx(entry.out_tau_vec, entry.out_gamma_fit[i], color=color[i], linewidth=3)
            
#             else: # combine fit of the DRT peaks
#                 self.axes.semilogx(entry.out_tau_vec, entry.out_gamma_fit, color = 'green', linewidth=3)
#             y_min = 0
#             y_max = max(entry.gamma)
        
#         self.axes.set_xlabel(r'$\tau/s$')
#         self.axes.set_ylabel(r'$\gamma(log \tau)/\Omega$')
#         self.axes.set_ylim([y_min, 1.1*y_max])
#         self.axes.set_xlim([min(entry.out_tau_vec), max(entry.out_tau_vec)])
    
#     def Score(self, entry): # plot the EIS score
        
#         if entry.method != 'BHT':
#             return
        
#         else: # for BHT method
#             Re_part = np.array([entry.out_scores['s_res_re'][0], entry.out_scores['s_res_re'][1], entry.out_scores['s_res_re'][2],
#                        entry.out_scores['s_mu_re'], entry.out_scores['s_HD_re'], entry.out_scores['s_JSD_re']])
#             Im_part = np.array([entry.out_scores['s_res_im'][0], entry.out_scores['s_res_im'][1], entry.out_scores['s_res_im'][2],
#                        entry.out_scores['s_mu_im'], entry.out_scores['s_HD_im'], entry.out_scores['s_JSD_im']])
#             x = np.arange(6)  # the label locations
#             width = 0.35  # the width of the bars
            
#             self.axes.bar(x - width/2, Re_part*100, width, label='Re part', color = 'blue')
#             self.axes.bar(x + width/2, Im_part*100, width, label='Im part', color = 'black')
#             self.axes.plot([-0.5,5.5], [100,100], '--k')
            
#             self.axes.legend(frameon = False, fontsize = 15)
#             self.axes.set_ylim([0, 125])
#             self.axes.set_xlim([-0.5, 5.5])
#             self.axes.set_xticks(x) 
#             self.axes.set_xticklabels((r'$s_{1\sigma}$', r'$s_{2\sigma}$', r'$s_{3\sigma}$', r'$s_{\mu}$', r'$s_{\rm HD}$', r'$s_{\rm JSD}$'))
#             self.axes.set_yticks([0,50,100]) 
#             self.axes.set_ylabel(r'$\rm Scores (\%)$')
            
# if __name__ == "__main__": # starting the GUI when users run this file 
    
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = GUI()
#     MainWindow.show()
#     sys.exit(app.exec_())


# def launch_gui():
#     app = QtWidgets.QApplication([])
#     MainWindow = GUI()
#     MainWindow.show()
#     app.exec_()    

import streamlit as st
import pandas as pd
import numpy as np
from numpy import log10, absolute, angle
import matplotlib.pyplot as plt
from . import layout
from .runs import *

def EIS_data_plot(ax, data):
    if data.method == 'BHT':
        ax.plot(data.mu_Z_re, -data.mu_Z_im, 'k', label='$Z_\\mu$(Regressed)', linewidth=3)
        ax.plot(data.mu_Z_H_re_agm, -data.mu_Z_H_im_agm, 'b', label='$Z_H$(Hilbert transform)', linewidth=3)
        ax.plot(data.Z_prime, -data.Z_double_prime, 'or')
        ax.legend(frameon=False, fontsize=15, loc='upper left')
    elif data.method != 'none':
        ax.plot(data.mu_Z_re, -data.mu_Z_im, 'k', linewidth=2)
        ax.plot(data.Z_prime, -data.Z_double_prime, 'or')
    else:
        ax.plot(data.Z_prime, -data.Z_double_prime, 'or')
    ax.set_xlabel('$Z^{\\prime}/\\Omega$')
    ax.set_ylabel('-$Z^{\\prime \\prime}/\\Omega$')
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.axis('equal')

def Magnitude_plot(ax, data):
    if data.method == 'BHT':
        ax.semilogx(data.freq, absolute(data.mu_Z_re + 1j * data.mu_Z_im), 'k', label='$Z_\\mu$(Regressed)', linewidth=2)
        ax.semilogx(data.freq, absolute(data.mu_Z_H_re_agm + 1j * data.mu_Z_H_im_agm), 'b', label='$Z_H$(Hilbert transform)', linewidth=2)
        ax.semilogx(data.freq, absolute(data.Z_exp), 'or')
        ax.legend(frameon=False, fontsize=15, loc='upper left')
    elif data.method != 'none':
        ax.semilogx(data.freq, absolute(data.mu_Z_re + 1j * data.mu_Z_im), 'k', linewidth=3)
        ax.semilogx(data.freq, absolute(data.Z_exp), 'or')
    else:
        ax.semilogx(data.freq, absolute(data.Z_exp), 'or')
    ax.set_xlim([min(data.freq), max(data.freq)])
    ax.set_xlabel('$f/Hz$')
    ax.set_ylabel('$|Z|/\\Omega$')

def Phase_plot(ax, data):
    if data.method == 'BHT':
        ax.semilogx(data.freq, angle(data.mu_Z_re + 1j * data.mu_Z_im, deg=True), 'k', label='$Z_\\mu$(Regressed)', linewidth=2)
        ax.semilogx(data.freq, angle(data.mu_Z_H_re_agm + 1j * data.mu_Z_H_im_agm, deg=True), 'b', label='$Z_H$(Hilbert transform)', linewidth=2)
        ax.semilogx(data.freq, angle(data.Z_exp, deg=True), 'or')
        ax.legend(frameon=False, fontsize=15, loc='upper left')
    elif data.method != 'none':
        ax.semilogx(data.freq, angle(data.mu_Z_re + 1j * data.mu_Z_im, deg=True), 'k', linewidth=3)
        ax.semilogx(data.freq, angle(data.Z_exp, deg=True), 'or')
    else:
        ax.semilogx(data.freq, angle(data.Z_exp, deg=True), 'or')
    ax.set_xlim([min(data.freq), max(data.freq)])
    ax.set_xlabel('$f/Hz$')
    ax.set_ylabel('$angle/^\\circ$')

def Re_data_plot(ax, data):
    if data.method == 'BHT':
        ax.fill_between(data.freq, data.mu_Z_H_re_agm - 3 * data.band_re_agm, data.mu_Z_H_re_agm + 3 * data.band_re_agm, facecolor='lightgrey')
        ax.semilogx(data.freq, data.mu_Z_re, 'k', label='$Z_\\mu$(Regressed)', linewidth=3)
        ax.semilogx(data.freq, data.mu_Z_H_re_agm, 'b', label='$Z_H$(Hilbert transform)', linewidth=3)
        ax.semilogx(data.freq, data.Z_prime, 'or')
        ax.legend(frameon=False, fontsize=15, loc='upper left')
    elif data.method != 'none':
        ax.semilogx(data.freq, data.mu_Z_re, 'k', linewidth=3)
        ax.semilogx(data.freq, data.Z_prime, 'or')
    else:
        ax.semilogx(data.freq, data.Z_prime, 'or')
    ax.set_xlim([min(data.freq), max(data.freq)])
    ax.set_xlabel('$f/Hz$')
    ax.set_ylabel('$Z^{\\prime}/\\Omega$')

def Im_data_plot(ax, data):
    if data.method == 'BHT':
        ax.fill_between(data.freq, -data.mu_Z_H_im_agm - 3 * data.band_im_agm, -data.mu_Z_H_im_agm + 3 * data.band_im_agm, facecolor='lightgrey')
        ax.semilogx(data.freq, -data.mu_Z_im, 'k', label='$Z_\\mu$(Regressed)', linewidth=3)
        ax.semilogx(data.freq, -data.mu_Z_H_im_agm, 'b', label='$Z_H$(Hilbert transform)', linewidth=3)
        ax.semilogx(data.freq, -data.Z_double_prime, 'or')
        ax.legend(frameon=False, fontsize=15, loc='upper left')
    elif data.method != 'none':
        ax.semilogx(data.freq, -data.mu_Z_im, 'k')
        ax.semilogx(data.freq, -data.Z_double_prime, 'or')
    else:
        ax.semilogx(data.freq, -data.Z_double_prime, 'or')
    ax.set_xlim([min(data.freq), max(data.freq)])
    ax.set_xlabel('$f/Hz$')
    ax.set_ylabel('-$Z^{\\prime \\prime}/\\Omega$')

def Re_residual_plot(ax, data):
    if data.method == 'BHT':
        ax.fill_between(data.freq, -3 * data.band_re_agm, 3 * data.band_re_agm, facecolor='lightgrey')
        ax.semilogx(data.freq, data.res_H_re, 'or')
        ax.set_xlabel('$f/Hz$')
        ax.set_ylabel('$R_{\\infty}+Z^{\\prime}_{H}-Z^{\\prime}_{exp}/\\Omega$')
        y_max = max(3 * data.band_re_agm)
    elif data.method != 'none':
        ax.semilogx(data.freq, data.res_re, 'or')
        ax.set_xlabel('$f/Hz$')
        ax.set_ylabel('$Z^{\\prime}_{DRT}-Z^{\\prime}_{exp}/\\Omega$')
        y_max = max(absolute(data.res_re))
    else:
        return
    ax.set_xlim([min(data.freq), max(data.freq)])
    ax.set_ylim([-1.1 * y_max, 1.1 * y_max])
    ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))

def Im_residual_plot(ax, data):
    if data.method == 'BHT':
        ax.fill_between(data.freq, -3 * data.band_im_agm, 3 * data.band_im_agm, facecolor='lightgrey')
        ax.semilogx(data.freq, data.res_H_im, 'or')
        ax.set_xlabel('$f/Hz$')
        ax.set_ylabel('$\\omega L_0+Z^{\\prime \\prime}_{H}-Z^{\\prime\\prime}_{exp}/\\Omega$')
        y_max = max(3 * data.band_im_agm)
    elif data.method != 'none':
        ax.semilogx(data.freq, data.res_im, 'or')
        ax.set_xlabel('$f/Hz$')
        ax.set_ylabel('$Z^{\\prime \\prime}_{DRT}-Z^{\\prime \\prime}_{exp}/\\Omega$')
        y_max = max(absolute(data.res_im))
    else:
        return
    ax.set_xlim([min(data.freq), max(data.freq)])
    ax.set_ylim([-1.1 * y_max, 1.1 * y_max])
    ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))

def DRT_data_plot(ax, data, drt_type):
    if data.method == 'none':
        return
    elif data.method == 'simple':
        if drt_type == "Gamma vs Tau":
            ax.semilogx(1./data.out_tau_vec, data.gamma, 'k', linewidth=3)
            y_min = 0
            y_max = max(data.gamma)
            xlabel = r'$\tau/s$'
            ylabel = r'$\gamma(\ln\tau)/\Omega$'
        elif drt_type == "Gamma vs Frequency":
            ax.semilogx(data.freq_fine, data.gamma_ridge_fine, 'k', linewidth=3)
            y_min = 0
            y_max = max(data.gamma_ridge_fine)
            xlabel = r'$f$/Hz'
            ylabel = r'$\gamma(\ln f)/\Omega$'
        elif drt_type == "G vs Tau":
            ax.semilogx(1./data.out_tau_vec, data.gamma*data.freq_fine, 'k', linewidth=3)
            y_min = 0
            y_max = max(data.gamma*data.freq_fine)
            xlabel = r'$\tau/s$'
            ylabel = r'$g(\tau)/(\Omega/\rm s)$'
        elif drt_type == "G vs Frequency":
            ax.semilogx(data.freq_fine, data.gamma*data.freq_fine, 'k', linewidth=3)
            y_min = 0
            y_max = max(data.gamma*data.freq_fine)
            xlabel = r'$f$/Hz'
            ylabel = r'$g(f)/(\Omega/\rm s)$'
    elif data.method == 'credit':
        if drt_type == "Gamma vs Tau":
            ax.fill_between(1./data.out_tau_vec, data.lower_bound, data.upper_bound, facecolor='lightgrey')
            ax.semilogx(1./data.out_tau_vec, data.gamma, color='black', label='MAP', linewidth=3)
            ax.semilogx(1./data.out_tau_vec, data.mean, color='blue', label='mean', linewidth=3)
            ax.semilogx(1./data.out_tau_vec, data.lower_bound, color='black')
            ax.semilogx(1./data.out_tau_vec, data.upper_bound, color='black')
            ax.legend(frameon=False, fontsize=15)
            y_min = 0
            y_max = max(data.upper_bound)
            xlabel = r'$\tau/s$'
            ylabel = r'$\gamma(\ln\tau)/\Omega$'
        elif drt_type == "Gamma vs Frequency":
            ax.fill_between(data.freq_fine, data.lower_bound_fine, data.upper_bound_fine, facecolor='lightgrey')
            ax.semilogx(data.freq_fine, data.gamma_ridge_fine, color='black', label='MAP', linewidth=3)
            ax.semilogx(data.freq_fine, data.gamma_mean_fine, color='blue', label='mean', linewidth=3)
            ax.semilogx(data.freq_fine, data.lower_bound_fine, color='black')
            ax.semilogx(data.freq_fine, data.upper_bound_fine, color='black')
            ax.legend(frameon=False, fontsize=15)
            y_min = 0
            y_max = max(data.upper_bound_fine)
            xlabel = r'$f$/Hz'
            ylabel = r'$\gamma(\ln f)/\Omega$'
        elif drt_type == "G vs Tau":
            ax.fill_between(1./data.out_tau_vec, data.lower_bound*data.freq_fine, data.upper_bound*data.freq_fine, facecolor='lightgrey')
            ax.semilogx(1./data.out_tau_vec, data.gamma*data.freq_fine, color='black', label='MAP', linewidth=3)
            ax.semilogx(1./data.out_tau_vec, data.mean*data.freq_fine, color='blue', label='mean', linewidth=3)
            ax.semilogx(1./data.out_tau_vec, data.lower_bound*data.freq_fine, color='black')
            ax.semilogx(1./data.out_tau_vec, data.upper_bound*data.freq_fine, color='black')
            ax.legend(frameon=False, fontsize=15)
            y_min = 0
            y_max = max(data.upper_bound*data.freq_fine)
            xlabel = r'$\tau/s$'
            ylabel = r'$g(\tau)/(\Omega/\rm s)$'
        elif drt_type == "G vs Frequency":
            ax.fill_between(data.freq_fine, data.lower_bound_fine*data.freq_fine, data.upper_bound_fine*data.freq_fine, facecolor='lightgrey')
            ax.semilogx(data.freq_fine, data.gamma_ridge_fine*data.freq_fine, color='black', label='MAP', linewidth=3)
            ax.semilogx(data.freq_fine, data.gamma_mean_fine*data.freq_fine, color='blue', label='mean', linewidth=3)
            ax.semilogx(data.freq_fine, data.lower_bound_fine*data.freq_fine, color='black')
            ax.semilogx(data.freq_fine, data.upper_bound_fine*data.freq_fine, color='black')
            ax.legend(frameon=False, fontsize=15)
            y_min = 0
            y_max = max(data.upper_bound_fine*data.freq_fine)
            xlabel = r'$f$/Hz'
            ylabel = r'$g(f)/(\Omega/\rm s)$'
    elif data.method == 'BHT':
        if drt_type == "Gamma vs Tau":
            ax.semilogx(1./data.out_tau_vec, data.mu_gamma_fine_re, 'b', linewidth=3, label='$Mean Re$')
            ax.semilogx(1./data.out_tau_vec, data.mu_gamma_fine_im, 'k', linewidth=3, label='$Mean Im$')
            y_min = min(np.concatenate((data.mu_gamma_fine_re, data.mu_gamma_fine_im)))
            y_max = max(np.concatenate((data.mu_gamma_fine_re, data.mu_gamma_fine_im)))
            xlabel = r'$\tau/s$'
            ylabel = r'$\gamma(\ln\tau)/\Omega$'
        elif drt_type == "Gamma vs Frequency":
            ax.semilogx(data.freq_fine, data.mu_gamma_fine_re, 'b', linewidth=3, label='$Mean Re$')
            ax.semilogx(data.freq_fine, data.mu_gamma_fine_im, 'k', linewidth=3, label='$Mean Im$')
            y_min = min(np.concatenate((data.mu_gamma_fine_re, data.mu_gamma_fine_im)))
            y_max = max(np.concatenate((data.mu_gamma_fine_re, data.mu_gamma_fine_im)))
            xlabel = r'$f$/Hz'
            ylabel = r'$\gamma(\ln f)/\Omega$'
        elif drt_type == "G vs Tau":
            ax.semilogx(1./data.out_tau_vec, data.mu_gamma_fine_re*data.freq_fine, 'b', linewidth=3, label='$Mean Re$')
            ax.semilogx(1./data.out_tau_vec, data.mu_gamma_fine_im*data.freq_fine, 'k', linewidth=3, label='$Mean Im$')
            y_min = min(np.concatenate((data.mu_gamma_fine_re*data.freq_fine, data.mu_gamma_fine_im*data.freq_fine)))
            y_max = max(np.concatenate((data.mu_gamma_fine_re*data.freq_fine, data.mu_gamma_fine_im*data.freq_fine)))
            xlabel = r'$\tau/s$'
            ylabel = r'$g(\tau)/(\Omega/\rm s)$'
        elif drt_type == "G vs Frequency":
            ax.semilogx(data.freq_fine, data.mu_gamma_fine_re*data.freq_fine, 'b', linewidth=3, label='$Mean Re$')
            ax.semilogx(data.freq_fine, data.mu_gamma_fine_im*data.freq_fine, 'k', linewidth=3, label='$Mean Im$')
            y_min = min(np.concatenate((data.mu_gamma_fine_re*data.freq_fine, data.mu_gamma_fine_im*data.freq_fine)))
            y_max = max(np.concatenate((data.mu_gamma_fine_re*data.freq_fine, data.mu_gamma_fine_im*data.freq_fine)))
            xlabel = r'$f$/Hz'
            ylabel = r'$g(f)/(\Omega/\rm s)$'
    elif data.method == 'peak':
        if drt_type == "Gamma vs Tau":
            ax.semilogx(data.out_tau_vec, data.gamma, color='black', linewidth=3)
            color = ['red', 'green', 'cyan', 'yellow', 'orange', 'blue', 'grey', 'brown', 'coral', 'darkblue', 'darkgreen', 'gold']
            if len(data.out_gamma_fit) == data.N_peaks:
                for i in range(data.N_peaks):
                    ax.semilogx(data.out_tau_vec, data.out_gamma_fit[i], color=color[i], linewidth=3)
            else:
                ax.semilogx(data.out_tau_vec, data.out_gamma_fit, color='green', linewidth=3)
            y_min = 0
            y_max = max(data.gamma)
            xlabel = r'$\tau/s$'
            ylabel = r'$\gamma(\ln\tau)/\Omega$'
        elif drt_type == "Gamma vs Frequency":
            ax.semilogx(data.freq_fine, data.gamma_ridge_fine, color='black', linewidth=3)
            color = ['red', 'green', 'cyan', 'yellow', 'orange', 'blue', 'grey', 'brown', 'coral', 'darkblue', 'darkgreen', 'gold']
            if len(data.gamma_gauss_mat) == data.N_peaks:
                for i in range(data.N_peaks):
                    ax.semilogx(data.freq_fine, data.gamma_gauss_mat[:, i], color=color[i], linewidth=3)
            else:
                ax.semilogx(data.freq_fine, data.gamma_gauss_mat, color='green', linewidth=3)
            y_min = 0
            y_max = max(data.gamma_ridge_fine)
            xlabel = r'$f$/Hz'
            ylabel = r'$\gamma(\ln f)/\Omega$'
        elif drt_type == "G vs Tau":
            ax.semilogx(1./data.out_tau_vec, data.gamma*data.freq_fine, color='black', linewidth=3)
            color = ['red', 'green', 'cyan', 'yellow', 'orange', 'blue', 'grey', 'brown', 'coral', 'darkblue', 'darkgreen', 'gold']
            if len(data.g_gauss_mat) == data.N_peaks:
                for i in range(data.N_peaks):
                    ax.semilogx(1./data.out_tau_vec, data.g_gauss_mat[:, i], color=color[i], linewidth=3)
            else:
                ax.semilogx(1./data.out_tau_vec, data.g_gauss_mat, color='green', linewidth=3)
            y_min = 0
            y_max = max(data.gamma*data.freq_fine)
            xlabel = r'$\tau/s$'
            ylabel = r'$g(\tau)/(\Omega/\rm s)$'
        elif drt_type == "G vs Frequency":
            ax.semilogx(data.freq_fine, data.gamma_ridge_fine*data.freq_fine, color='black', linewidth=3)
            color = ['red', 'green', 'cyan', 'yellow', 'orange', 'blue', 'grey', 'brown', 'coral', 'darkblue', 'darkgreen', 'gold']
            if len(data.g_gauss_mat) == data.N_peaks:
                for i in range(data.N_peaks):
                    ax.semilogx(data.freq_fine, data.g_gauss_mat[:, i], color=color[i], linewidth=3)
            else:
                ax.semilogx(data.freq_fine, data.g_gauss_mat, color='green', linewidth=3)
            y_min = 0
            y_max = max(data.gamma_ridge_fine*data.freq_fine)
            xlabel = r'$f$/Hz'
            ylabel = r'$g(f)/(\Omega/\rm s)$'
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([y_min, 1.1 * y_max])
    ax.set_xlim([min(data.out_tau_vec), max(data.out_tau_vec)])

def Score_plot(ax, data):
    if data.method != 'BHT':
        return
    else:
        Re_part = np.array([data.out_scores['s_res_re'][0], data.out_scores['s_res_re'][1], data.out_scores['s_res_re'][2],
                            data.out_scores['s_mu_re'], data.out_scores['s_HD_re'], data.out_scores['s_JSD_re']])
        Im_part = np.array([data.out_scores['s_res_im'][0], data.out_scores['s_res_im'][1], data.out_scores['s_res_im'][2],
                            data.out_scores['s_mu_im'], data.out_scores['s_HD_im'], data.out_scores['s_JSD_im']])
        x = np.arange(6)
        width = 0.35
        ax.bar(x - width / 2, Re_part * 100, width, label='Re part', color='blue')
        ax.bar(x + width / 2, Im_part * 100, width, label='Im part', color='black')
        ax.plot([-0.5, 5.5], [100, 100], '--k')
        ax.legend(frameon=False, fontsize=15)
        ax.set_ylim([0, 125])
        ax.set_xlim([-0.5, 5.5])
        ax.set_xticks(x)
        ax.set_xticklabels((r'$s_{1\sigma}$', r'$s_{2\sigma}$', r'$s_{3\sigma}$', r'$s_{\mu}$', r'$s_{\rm HD}$', r'$s_{\rm JSD}$'))
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel(r'$\rm Scores (\%)$')
    
