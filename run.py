import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os
import sys
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC

#helper function to clean data, helps correlate e columns
def get_size_ind(x,dirs,col):
    indices = []
    for i in range(len(x.packet_dirs)):
        if(x.packet_dirs[i]==int(dirs)):
            indices.append(x[col][i])
    return indices

#function that will extract features from each dataset
def build_features(datasets,streaming=None):

    #store feature for every dataset
    if (streaming is not None):
        fdf = pd.DataFrame(columns = ['%peaks_download','%peaks_upload','peak_heights_std_download','peak_heights_std_upload','time_between_peaks_std_download','time_between_peaks_std_upload','download_packets_cv','upload_packets_cv','upload_time_cv','download_time_cv','std_upload_time','std_download_time','is_streaming'])
    else:
        fdf = pd.DataFrame(columns = ['%peaks_download','%peaks_upload','peak_heights_std_download','peak_heights_std_upload','time_between_peaks_std_download','time_between_peaks_std_upload','download_packets_cv','upload_packets_cv','upload_time_cv','download_time_cv','std_upload_time','std_download_time'])
    index=0
    #go through every dataset
    for path in datasets:
        df = pd.read_csv(path)

        #clean datasets
        df['packet_times'] = df['packet_times'].str.strip(';').str.split(';').apply(lambda x:[int(i) for i in x])
        df['packet_sizes'] = df['packet_sizes'].str.strip(';').str.split(';').apply(lambda x:[int(i) for i in x])
        df['packet_dirs'] = df['packet_dirs'].str.strip(';').str.split(';').apply(lambda x: [int(i) for i in x])

        df['psize_from1'] = df.apply(get_size_ind,args=('1','packet_sizes'),axis=1)
        df['psize_from2'] = df.apply(get_size_ind,args=('2','packet_sizes'),axis=1)
        df['ptime_from1'] = df.apply(get_size_ind,args=('1','packet_times'),axis=1)
        df['ptime_from2'] = df.apply(get_size_ind,args=('2','packet_times'),axis=1)

        #threshold for peaks
        upload_peak = 100000
        download_peak = 250000

        # keep track of row values
        row = []
        # %peaks_download values gets the number of peaks in relation to the size of the datset
        row.append(len(find_peaks(df['2->1Bytes'], height=download_peak)[0])/len(df))
        # %peaks_upload values gets the number of peaks in relation to the size of the datset
        row.append(len(find_peaks(df['1->2Bytes'], height=upload_peak)[0])/len(df))
        # 'peak_heights_std_download' gets the standard deviation of the peak heights
        row.append(np.std(find_peaks(df['2->1Bytes'], height=download_peak)[1]['peak_heights']))
        # 'peak_heights_std_upload' gets the standard deviation of the peak heights
        row.append(np.std(find_peaks(df['1->2Bytes'], height=upload_peak)[1]['peak_heights']))
        #get the time between peaks
        peak_times_download = df.iloc[find_peaks(df['2->1Bytes'], height=download_peak)[0]]['Time'].values
        peak_times_upload = df.iloc[find_peaks(df['1->2Bytes'], height=upload_peak)[0]]['Time'].values
        # 'time_between_peaks_std_download' standard deviation of times between peaks
        row.append(np.std(([peak_times_download[i]-peak_times_download[i-1] for i in range(len(peak_times_download)) if i!=0])))
        # 'time_between_peaks_std_upload' standard deviation of times between peaks
        row.append(np.std(([peak_times_upload[i]-peak_times_upload[i-1] for i in range(len(peak_times_upload)) if i!=0])))
        # calculate standard deviation to the mean of individual packet sizes being downloaded (coefficient of variation)
        row.append((np.std(df['psize_from2'].sum())/np.mean(df['psize_from2'].sum())))
        # calculate standard deviation to the mean of individual packet sizes being uploaded (coefficient of variation)
        row.append((np.std(df['psize_from1'].sum())/np.mean(df['psize_from1'].sum())))
        #get (coefficient of variation) upload time
        row.append(np.std(df['ptime_from1'].apply(lambda x: max(x)-min(x) if x!=[] else 0))/np.mean(df['ptime_from1'].apply(lambda x: max(x)-min(x) if x!=[] else 0)))
        #get (coefficient of variation) download time
        row.append(np.std(df['ptime_from2'].apply(lambda x: max(x)-min(x) if x!=[] else 0))/np.mean(df['ptime_from2'].apply(lambda x: max(x)-min(x) if x!=[] else 0)))
        #get std upload time
        row.append(np.std(df['ptime_from1'].apply(lambda x: max(x)-min(x) if x!=[] else 0)))
        #get std download time
        row.append(np.std(df['ptime_from2'].apply(lambda x: max(x)-min(x) if x!=[] else 0)))

        #indicate if it is streaming or not
        if (streaming is not None):
            row.append(streaming)

        fdf.loc[index] = row
        index+=1

    #if a chunk doesnt have a peak fillnan -- still trying to figure out a better way of filling nans
    fdf=fdf.fillna(0)
    #fdf['peak_heights_std']=fdf['peak_heights_std'].fillna(0)
    #fdf['time_between_peaks_std']=fdf['time_between_peaks_std'].fillna(0)

    return fdf

#train the model on train data and apply model
def train_run(data):
    #get paths for training data
    streaming_paths = ['data/streaming/'+x for x in os.listdir('data/streaming/') if x !='.DS_Store']
    non_streaming_paths = ['data/nonstreaming/'+x for x in os.listdir('data/nonstreaming/')if x !='.DS_Store']

    #clean and build features for training data
    print('cleaning and building features for training data...')
    fns = build_features(non_streaming_paths,False)
    fs = build_features(streaming_paths,True)
    fdf = fns.append(fs)

    #get training columns
    X_train = fdf.drop('is_streaming',axis=1)
    y_train = fdf['is_streaming'].astype(bool)

    #train model
    print('training model...')
    #model = SVC().fit(X_train,y_train)
    model = RandomForestClassifier(random_state=43).fit(X_train, y_train)
    #get prediction
    print('calculating prediction...')
    prediction = model.predict(data.drop('is_streaming',axis=1,errors='ignore'))
    #sum(prediction == data['is_streaming'])/len(prediction)
    return prediction

#get input file path
fp = ([sys.argv[1]])
#clean and build features for input dataset
print('cleaning and building features for input data...')
fdf = build_features(fp)
#output prediction
print('The model predicts streaming presence is: ',train_run(fdf)[0])
