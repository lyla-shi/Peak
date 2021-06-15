import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymssa import MSSA

#-----------------------methods----------------------------------------------#
def mssa(dataframe, decompose_num, reconstruct_num):

    #split test/training set
    total_rows = len(dataframe)
    split = int(total_rows * 0.8)   #use 80% of the dataset for training
    train = dataframe.iloc[:split]
    test = dataframe.iloc[split+1:]
    train_time = train.iloc[:,[0]].values
    train_data = train.iloc[:,[1]].values
    test_time = test.iloc[:, [0]].values
    test_data = test.iloc[:, [1]].values

    #center the data for decomposition
    train_data_cen = train_data - np.mean(train_data)
    test_data_cen = test_data - np.mean(test_data)

    #fit simple MSSA
    # st.write("#The number of components.")
    mssa = MSSA(n_components=decompose_num,  #maximum number of components
                window_size=None,   #size to take the transposition of the hankel matrix of this timeseries
                verbose=True)   #see the steps taken in the fit procedure
    mssa.fit(train_data_cen)

    #plot components
    print("components shape:",mssa.components_.shape)   #component matrix(input columns num, observation num, rank num of components output)

    st.write("Decomposition")
    for comp in range(decompose_num):
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.plot(train_time, train_data_cen, lw=3, alpha=0.2, c='k', label='Original')
        ax.plot(train_time, mssa.components_[0, :, comp], lw=3, c='steelblue', alpha=0.8,
                label='component={}'.format(comp))
        ax.legend()
        st.pyplot(fig)
        # plt.show()

    #Reconstruction
    st.write("Reconstruction")
    cumulative_recon = np.zeros_like(mssa.components_[0,:,0])

    for comp in range(reconstruct_num):
        fig, ax = plt.subplots(figsize=(18, 7))
        current_component = mssa.components_[0, :, comp]
        cumulative_recon = cumulative_recon + current_component

        ax.plot(train_time, train_data_cen, lw=3, alpha=0.2, c='k', label="Original")
        ax.plot(train_time, cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
        ax.plot(train_time, current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))

        ax.legend()
        st.pyplot(plt)
        # plt.show()

#-----------------------methods end----------------------------------------------#


"""
# Hello Plot
Here's my first attempt at using data to create a plot.

Please upload .csv data file here:
"""

#upload file
uploaded_file = st.file_uploader("Upload File",type=['csv'])

#get file
if uploaded_file is not None:

    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    # print file detail
    """File details: """
    file_details

    # read and save data
    df = pd.read_csv(uploaded_file, encoding = 'unicode_escape')
    st.dataframe(df)

    #draw the plot
    #matlib plot
    fig = plt.figure(figsize=(18, 7))
    plt.plot(df['t [s]'], df['motion [µm/s]'])
    st.pyplot(fig)

    #fetch parameter
    ##get time range
    time_min = float(df.iloc[:,[0]].min())
    time_max = float(df.iloc[:,[0]].max())

    values = st.slider('Select a range of values', time_min, time_max, (time_min, time_max))
    st.write('Time bound for decomposition:', values)
    (lowbound, upbound) = values

    ##get number of components for decomposition
    de_number = st.number_input('Input the number of components for decomposition', min_value=1, max_value=15, step=1)
    st.write('The number of components for decomposition is', de_number)

    ##get number of components for reconstruct
    re_number = st.number_input('Input the number of components for reconstruction', min_value=1, max_value=de_number, step=1)
    st.write('The number of components for reconstruction is', re_number)

    #run MSSA
    new_df = df.loc[(df['t [s]']>=lowbound) & (df['t [s]']<=upbound),:]
    new_df.shape
    mssa(new_df,de_number,re_number)


