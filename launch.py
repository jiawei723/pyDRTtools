import streamlit as st
import pandas as pd
import numpy as np
from numpy import log10, absolute, angle
import matplotlib.pyplot as plt
from pyDRTtools import layout
from pyDRTtools.runs import *
from pyDRTtools.GUI import *

processed_data = None

def launch_gui():
    global processed_data

    st.set_page_config(page_title="DRTtools", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("DRTtools")

    # Initialize data
    data = None

    uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])

    if uploaded_file is not None:
        # Read data
        data = EIS_object.from_file(uploaded_file)

    sidebar_container_setting = st.sidebar.container()
    with sidebar_container_setting:
        with st.expander("Setting"):
            # Processing parameters
            induct_options = ["Fitting w/o Inductance", "Fitting with Inductance", "Discard Inductive Data"]
            induct_choice = st.selectbox("Inductance Included", induct_options)
            if induct_choice == "Discard Inductive Data" and data is not None:
                data.freq = data.freq[-data.Z_double_prime > 0]
                data.Z_prime = data.Z_prime[-data.Z_double_prime > 0]
                data.Z_double_prime = data.Z_double_prime[-data.Z_double_prime > 0]
                data.Z_exp = data.Z_prime + data.Z_double_prime * 1j
            elif data is not None:
                data.freq = data.freq_0
                data.Z_prime = data.Z_prime_0
                data.Z_double_prime = data.Z_double_prime_0
                data.Z_exp = data.Z_exp_0

            if data is not None:
                data.tau = 1 / data.freq
                data.tau_fine = np.logspace(log10(data.tau.min()) - 0.5, log10(data.tau.max()) + 0.5, 10 * data.freq.shape[0])
                data.method = "none"
                                
            drt_type_options = ["Gamma vs Tau", "Gamma vs Frequency", "G vs Tau", "G vs Frequency"] 
            drt_type = st.selectbox("DRT Type", drt_type_options)
            # drt_type = "Gamma vs Tau"
            rbf_type = st.selectbox("Method of Discretization", ["Gaussian", "C0 Matern", "C2 Matern", "C4 Matern", "C6 Matern", "Inverse Quadratic", "Inverse Quadric", "Cauchy"])
            data_used = st.selectbox("Data Used", ["Combined Re-Im Data", "Re Data", "Im Data"])
            
            
            der_used = st.selectbox("Regularization Derivative", ["1st order", "2nd order"])
            cv_type = st.selectbox("Regularization Method", ["custom", "GCV", "mGCV", "rGCV", "LC", "re-im", "kf"])
            reg_param = st.number_input("Regularization Parameter", value=1e-3, format="%.6f")

    if induct_choice == "Fitting w/o Inductance" :
        induct_used = 1
    elif induct_choice == "Fitting with Inductance" :
        induct_used = 2
    else:
        induct_used = 0

    sidebar_container_rbf = st.sidebar.container()
    with sidebar_container_rbf:
        with st.expander("Options for RBF"):
            shape_control = st.selectbox("RBF Shape Control", ["FWHM Coefficient", "Shape Factor"])
            coeff = st.number_input("FWHM Control", value=0.5)

    sidebar_container_Baye = st.sidebar.container()
    with sidebar_container_Baye:
        with st.expander("Options for Bayesian Run"):
            sample_number = st.number_input("Sample Number", value=2000, min_value=1, step=1)

    sidebar_container_Peak = st.sidebar.container()
    with sidebar_container_Peak:
        with st.expander("Options for Peak Analysis"):
            peak_method = st.selectbox("Peak Method", ["separate", "combine"])
            N_peaks = st.number_input("Number of Peaks", value=1.0, min_value=1.0, step=1.0)

    # Main content
    if data is not None:

        # Plot options
        plot_options = ["EIS_data", "Magnitude", "Phase", "Re_data", "Im_data", "Re_residual", "Im_residual"]
        selected_plot = st.selectbox("Select Plot", plot_options)

        # Plot data
        fig, ax = plt.subplots()
        if selected_plot == "EIS_data":
            EIS_data_plot(ax, data)
        elif selected_plot == "Magnitude":
            Magnitude_plot(ax, data)
        elif selected_plot == "Phase":
            Phase_plot(ax, data)
        elif selected_plot == "Re_data":
            Re_data_plot(ax, data)
        elif selected_plot == "Im_data":
            Im_data_plot(ax, data)
        elif selected_plot == "Re_residual":
            Re_residual_plot(ax, data)
        elif selected_plot == "Im_residual":
            Im_residual_plot(ax, data)
        elif selected_plot == "DRT_data":
            DRT_data_plot(ax, data)
        elif selected_plot == "Score":
            Score_plot(ax, data)

        st.pyplot(fig)

        # Processing options
        process_options = ["Simple Run", "Bayesian Run", "BHT Run", "Peak Analysis Run"]
        selected_process = st.selectbox("Select Processing", process_options)

        # Run processing
        if st.button("Run Processing"):
            if selected_process == "Simple Run":
                processed_data = simple_run(data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                  der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                                  shape_control=shape_control, coeff=coeff)
            elif selected_process == "Bayesian Run":
                processed_data = Bayesian_run(data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                    der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                                    shape_control=shape_control, coeff=coeff, NMC_sample=sample_number)
            elif selected_process == "BHT Run":
                processed_data = BHT_run(data, rbf_type, der_used, shape_control, coeff)
            elif selected_process == "Peak Analysis Run":
                processed_data = peak_analysis(data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                     der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                                     shape_control=shape_control, coeff=coeff, peak_method=peak_method, N_peaks=N_peaks)

            # Update plot after processing
            fig, ax = plt.subplots()
            DRT_data_plot(ax, processed_data, drt_type)
            st.pyplot(fig)

        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        def covert_drt_format(df, drt_type):

            return 0
        # Export options
        export_options = ["Export DRT", "Export EIS", "Export Figure"]
        selected_export = st.selectbox("Select Export", export_options, index=0)      

        if selected_export == "Export DRT":
            if processed_data is not None and hasattr(processed_data, 'out_tau_vec') and hasattr(processed_data, 'gamma'):
                export_data = processed_data.out_tau_vec, processed_data.gamma
                export_filename = st.text_input("Enter filename for DRT export", value="drt_export.csv")
                if st.button("Export DRT"):
                    
                    np.savetxt(export_filename, np.column_stack(export_data), delimiter=",", header="tau,gamma", comments="")
                    st.success(f"DRT exported to {export_filename}")
            else:
                st.warning("No data available for export. Please run the processing first.")
        elif selected_export == "Export EIS":
            if processed_data is not None and hasattr(processed_data, 'freq') and hasattr(processed_data, 'mu_Z_re') and hasattr(processed_data, 'mu_Z_im'):
                export_data = processed_data.freq, processed_data.mu_Z_re, processed_data.mu_Z_im
                export_filename = st.text_input("Enter filename for EIS export", value="eis_export.csv")
                if st.button("Export EIS"):
                    np.savetxt(export_filename, np.column_stack(export_data), delimiter=",", header="freq,mu_Z_re,mu_Z_im", comments="")
                    st.success(f"EIS exported to {export_filename}")
            else:
                st.warning("No data available for export. Please run the processing first.")
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