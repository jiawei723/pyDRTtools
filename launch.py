# from pyDRTtools.GUI import launch_gui

# def main():
#     launch_gui()

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
from numpy import log10, absolute, angle
import matplotlib.pyplot as plt
from pyDRTtools import layout
from pyDRTtools.runs import *

def launch_gui():
    st.set_page_config(page_title="DRTtools", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("DRTtools")

    # Initialize data
    data = None

    # Sidebar widgets
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])

    induct_options = ["Fitting w/o Inductance", "Fitting with Inductance", "Discard Inductive Data"]
    induct_choice = st.sidebar.selectbox("Inductance Included", induct_options)

    if induct_choice == "Fitting w/o Inductance" :
        induct_used = 1
    elif induct_choice == "Fitting with Inductance" :
        induct_used = 2
    else:
        induct_used = 0

    # Processing parameters
    rbf_type = st.sidebar.selectbox("RBF Type", ["Gaussian", "C0 Matern", "C2 Matern", "C4 Matern", "C6 Matern", "Inverse Quadratic", "Inverse Quadric", "Cauchy"])
    data_used = st.sidebar.selectbox("Data Used", ["Combined Re-Im Data", "Re Data", "Im Data"])
    der_used = st.sidebar.selectbox("Regularization Derivative", ["1st order", "2nd order"])
    cv_type = st.sidebar.selectbox("Regularization Method", ["custom", "GCV", "mGCV", "rGCV", "LC", "re-im", "kf"])
    reg_param = st.sidebar.number_input("Regularization Parameter", value=1e-3)


    sidebar_container = st.sidebar.container()

    with sidebar_container:
        with st.expander("Options for RBF"):
            shape_control = st.selectbox("RBF Shape Control", ["FWHM Coefficient", "Shape Factor"])
            coeff = st.number_input("FWHM Control", value=0.5)

    if uploaded_file is not None:
        # Read data
        data = EIS_object.from_file(uploaded_file)

        # Discard inductance data if necessary
        if induct_choice == "Discard Inductive Data":
            data.freq = data.freq[-data.Z_double_prime > 0]
            data.Z_prime = data.Z_prime[-data.Z_double_prime > 0]
            data.Z_double_prime = data.Z_double_prime[-data.Z_double_prime > 0]
            data.Z_exp = data.Z_prime + data.Z_double_prime * 1j
        else:
            data.freq = data.freq_0
            data.Z_prime = data.Z_prime_0
            data.Z_double_prime = data.Z_double_prime_0
            data.Z_exp = data.Z_exp_0

        data.tau = 1 / data.freq
        data.tau_fine = np.logspace(log10(data.tau.min()) - 0.5, log10(data.tau.max()) + 0.5, 10 * data.freq.shape[0])
        data.method = "none"

    # Main content
    if data is not None:
        # Display data
        # st.write(data)

        # Plot options
        plot_options = ["EIS_data", "Magnitude", "Phase", "Re_data", "Im_data", "Re_residual", "Im_residual", "DRT_data", "Score"]
        selected_plot = st.selectbox("Select Plot", plot_options)

        # Plot data
        fig, ax = plt.subplots()
        if selected_plot == "EIS_data":
            ax.plot(data.Z_prime, -data.Z_double_prime, 'or')
            ax.set_xlabel('$Z^{\prime}/\Omega$')
            ax.set_ylabel('-$Z^{\prime \prime}/\Omega$')
            ax.axis('equal')
        elif selected_plot == "Magnitude":
            ax.semilogx(data.freq, absolute(data.Z_exp), 'or')
            ax.set_xlabel('$f/Hz$')
            ax.set_ylabel('$|Z|/\Omega$')
        # Add more plotting options based on the original code
        elif selected_plot == "Phase":
            ax.semilogx(data.freq, angle(data.Z_exp), 'or')
            ax.set_xlabel('$f/Hz$')
            ax.set_ylabel('$\phi/deg$')
        elif selected_plot == "Re_data":
            ax.semilogx(data.freq, data.Z_prime, 'or')
            ax.set_xlabel('$f/Hz$')
            ax.set_ylabel('$Z^{\prime}/\Omega$')
        elif selected_plot == "Im_data":
            ax.semilogx(data.freq, data.Z_double_prime, 'or')
            ax.set_xlabel('$f/Hz$')
            ax.set_ylabel('$Z^{\prime \prime}/\Omega$')
        elif selected_plot == "Re_residual":
            ax.semilogx(data.freq, data.Z_prime - data.Z_prime_fit, 'or')
            ax.set_xlabel('$f/Hz$')
            ax.set_ylabel('$Z^{\prime}_{residual}/\Omega$')
        elif selected_plot == "Im_residual":
            ax.semilogx(data.freq, data.Z_double_prime - data.Z_double_prime_fit, 'or')
            ax.set_xlabel('$f/Hz$')
            ax.set_ylabel('$Z^{\prime \prime}_{residual}/\Omega$')
        # elif selected_plot == "DRT_data":
        #     ax.semilogx(data.out_tau_vec, data.gamma, 'or')
        #     ax.set_xlabel(r'$\tau/s$')
        #     ax.set_ylabel(r'$\gamma(log \tau)/\Omega$')
        #     ax.set_ylim([0, 1.1 * max(data.gamma)])
        #     ax.set_xlim([min(data.out_tau_vec), max(data.out_tau_vec)])
        elif selected_plot == "Score":
            ax.plot(data.score, 'or')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')

        st.pyplot(fig)

        # Processing options
        process_options = ["Simple Run", "Bayesian Run", "BHT Run", "Peak Analysis Run"]
        selected_process = st.selectbox("Select Processing", process_options)

        # Run processing
        if st.button("Run Processing"):
            if selected_process == "Simple Run":
                data = simple_run(data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                  der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                                  shape_control=shape_control, coeff=coeff)
            elif selected_process == "Bayesian Run":
                sample_number = st.number_input("Sample Number", value=2000)
                data = Bayesian_run(data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                    der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                                    shape_control=shape_control, coeff=coeff, NMC_sample=sample_number)
            elif selected_process == "BHT Run":
                data = BHT_run(data, rbf_type, der_used, shape_control, coeff)
            elif selected_process == "Peak Analysis Run":
                peak_method = st.selectbox("Peak Method", ["separate", "combine"])
                N_peaks = st.number_input("Number of Peaks", value=1)
                data = peak_analysis(data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                     der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                                     shape_control=shape_control, coeff=coeff, peak_method=peak_method, N_peaks=N_peaks)

            # Update plot after processing
            if selected_plot == "DRT_data":
                fig, ax = plt.subplots()
                ax.semilogx(data.out_tau_vec, data.gamma, 'k', linewidth=3)
                ax.set_xlabel(r'$\tau/s$')
                ax.set_ylabel(r'$\gamma(log \tau)/\Omega$')
                ax.set_ylim([0, 1.1 * max(data.gamma)])
                ax.set_xlim([min(data.out_tau_vec), max(data.out_tau_vec)])
                st.pyplot(fig)

        # Export options
        export_options = ["None", "Export DRT", "Export EIS", "Export Figure"]
        selected_export = st.selectbox("Select Export", export_options, index=0)

        if selected_export == "Export DRT":
            export_data = data.out_tau_vec, data.gamma
            export_filename = st.text_input("Enter filename for DRT export", value="drt_export.csv")
            if st.button("Export DRT"):
                pd.DataFrame({"tau": export_data[0], "gamma": export_data[1]}).to_csv(export_filename, index=False)
                st.success(f"DRT exported to {export_filename}")
        elif selected_export == "Export EIS":
            export_data = data.freq, data.mu_Z_re, data.mu_Z_im
            export_filename = st.text_input("Enter filename for EIS export", value="eis_export.csv")
            if st.button("Export EIS"):
                pd.DataFrame({"freq": export_data[0], "mu_Z_re": export_data[1], "mu_Z_im": export_data[2]}).to_csv(export_filename, index=False)
                st.success(f"EIS exported to {export_filename}")
        elif selected_export == "Export Figure":
            export_filename = st.text_input("Enter filename for figure export", value="figure.png")
            if st.button("Export Figure"):
                fig.savefig(export_filename)
                st.success(f"Figure exported to {export_filename}")
        else:
            st.write("No export selected")

def main():
    launch_gui()

if __name__ == "__main__":
    main()