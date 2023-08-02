import os
import numpy as np
import operation_code_v2 as operation_code
global pos_spots, neg_spots, test_spots, n_sigma, n_spots, accum_flag, x_data

'''
Specify parameters for the code
'''


m_thresh_bool = 1
m_thresh = 0.001
time_touse = 4
time_offset = 0 #Time offset from 0 min
neg_analysis_flag = 1 #Whether to do outliers exclusion for zero
zero_subtraction_flag = 0 #Whether to subtract 0 signal from all sensors or not

#CHANGE THIS VALUE FOR EXLUDING OVERLAPING SPOTS
accum_flag = 1 #whether to keep or exclude overlapping spots 0 - don't exclude, only borders (default), 1 - exclude all if >=3 spots are exluded, 2 - exclude all if >=1 spots are exluded

differential_flag = 1
keep_zero_time = 0

metric_flag = 1






time_touse = time_touse + time_offset
n_sigma = 1
n_spots = 17
pos_spots = [0, 15]
neg_spots = [3, 4, 11, 12, ]
test_spots = [1, 2, 5, 6, 7, 8, 9, 10, 13, 14]
'''
Testing of a new sample
'''


'''
We will have two modes in running this, either it has been passed through or the file itself had been run
Thus, we will use a boolean flag to check.
'''

def encapsulate(file_path, output_path, prediction_folder_path, features_folder, zero, test):
    data = np.loadtxt(open("/home/pi/Desktop/alan/data.csv"), delimiter=",") #this is a constant as per convo with Artem
    data_size = data.shape

    x_data = data[2: data_size[0], 2: data_size[1]]
    y_labels = data[1, 2:data_size[1]]
    zero_cols = np.where(y_labels==0)[0]

    testing_data = np.loadtxt(open(file_path + '/' + test + '.csv'), delimiter=",")
    data_zero = np.loadtxt(open(file_path + '/' + zero + '.csv'), delimiter=",")

    data_size = testing_data.shape
    testing_data = testing_data[1:data_size[0], 1:data_size[1]]
    data_zero = data_zero[1:data_size[0], 1:data_size[1]]

    data_size = testing_data.shape

    testing_data = np.resize(np.transpose(testing_data), (data_size[0]*data_size[1] ,1))
    data_zero = np.resize(np.transpose(data_zero), (data_size[0]*data_size[1] ,1))

    n_samples = testing_data.shape[1]

    data_inclusion_flag = np.zeros((time_touse*n_spots,1))
    data_inclusion_flag_0 = np.zeros(((time_touse+1)*n_spots,1))

    for time in range(time_touse):
        neg_spots_shifted = [x + time * n_spots for x in neg_spots]
        test_spots_shifted = [x + time * n_spots for x in test_spots]
        if time < time_touse:
            data_inclusion_flag[neg_spots_shifted, :] = 2
            data_inclusion_flag[test_spots_shifted, :] = 1
            data_inclusion_flag_0[neg_spots_shifted, :] = 2
            data_inclusion_flag_0[test_spots_shifted, :] = 1
        else:
            data_inclusion_flag_0[neg_spots_shifted, :] = 2
            data_inclusion_flag_0[test_spots_shifted, :] = 1

    r2_sequence0, cv_sequence0, loss_sequence0, spot_exclusion_sequence0, feature_set_opt0,\
        = operation_code.feature_selection(x_data, y_labels, data_inclusion_flag_0, time_touse, zero_cols, zero_subtraction_flag, metric_flag)

    output_features0 = operation_code.generate_output_features(r2_sequence0, cv_sequence0, loss_sequence0, spot_exclusion_sequence0, feature_set_opt0)

    feature_set_opt_0 = np.ones((n_spots,1))
    xzero_mean_0 = operation_code.calculate_zero_1sample(data_zero, data_inclusion_flag_0, feature_set_opt0, accum_flag, time_touse)

    signal_feat, xdata_raw_feat, xdata_raw_zerosub_feat = operation_code.predict_concentration(testing_data, data_inclusion_flag_0, feature_set_opt0, time_touse, xzero_mean_0,
                                                                zero_subtraction_flag, accum_flag, time_offset)

    xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat, data_inclusion_flag_feat = operation_code.prep_test_sample_features(testing_data, data_inclusion_flag_0, feature_set_opt0, time_touse)

    xsample_features = np.transpose(np.resize(xsample_features, (time_touse, data_size[0])))
    data_inclusion_flag_feat = np.transpose(np.resize(data_inclusion_flag_feat, (time_touse, data_size[0])))

    raw_data_feat = operation_code.generate_output_data(xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat, data_inclusion_flag_feat, xdata_raw_feat, time_touse)


    testing_data_mean, testing_data_mean_control, testing_data_std, testing_data_std_control, xsample_processed, data_inclusion_flag, thresh_all = operation_code.spot_exclusion_all_data(testing_data, data_inclusion_flag, n_sigma, accum_flag, time_touse, n_spots,m_thresh,m_thresh_bool, 0, 0, keep_zero_time)
    operation_code.print_control1(data_inclusion_flag, n_sigma)

    cvs_spotwise = np.concatenate((testing_data_std_control, testing_data_std), axis = 1)

    cv = testing_data_std/testing_data_mean
    cv_control = testing_data_std_control/testing_data_mean_control

    data_checked = testing_data
    testing_data_control = np.transpose(np.resize(data_checked, (time_touse, data_size[0])))



    data_inclusion_flag_test = np.transpose(np.resize(data_inclusion_flag, (time_touse, data_size[0])))
    data_inclusion_flag_control = np.transpose(np.resize(data_inclusion_flag_0, (time_touse, data_size[0])))


    xzero_mean_q = operation_code.calculate_zero_1sample(data_zero, data_inclusion_flag, feature_set_opt_0, accum_flag, time_touse)

    r2_sequence, cv_sequence, loss_sequence, spot_exclusion_sequence, feature_set_opt,\
        = operation_code.feature_selection(x_data, y_labels, data_inclusion_flag, time_touse, zero_cols, zero_subtraction_flag, metric_flag)

    output_features = operation_code.generate_output_features(r2_sequence, cv_sequence, loss_sequence, spot_exclusion_sequence, feature_set_opt)


    xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat, data_inclusion_flag_feat = operation_code.prep_test_sample_features(testing_data, data_inclusion_flag, feature_set_opt, time_touse)
    xsample_features = np.transpose(np.resize(xsample_features, (time_touse, data_size[0])))
    data_inclusion_flag_feat = np.transpose(np.resize(data_inclusion_flag_feat, (time_touse, data_size[0])))

    signal_0, xdata_raw_0, xdata_raw_0_zerosub  = operation_code.predict_concentration(data_checked, data_inclusion_flag_0, feature_set_opt_0, time_touse, xzero_mean_0,
                                                                zero_subtraction_flag, accum_flag, time_offset)
    signal_q, xdata_raw_q, xdata_raw_q_zerosub = operation_code.predict_concentration(data_checked, data_inclusion_flag, feature_set_opt_0, time_touse, xzero_mean_q,
                                                                zero_subtraction_flag, accum_flag, time_offset)

    signal_q_feat, xdata_raw_q_feat, xdata_raw_q_zerosub_feat = operation_code.predict_concentration(data_checked, data_inclusion_flag, feature_set_opt, time_touse, xzero_mean_q,
                                                                zero_subtraction_flag, accum_flag, time_offset)


    print('Prediction for raw data:')
    print('Signal = ' + str(signal_0))

    print('Prediction for data after quality check:')
    print('Signal = ' + str(signal_q))

    raw_data_control = operation_code.generate_output_data(testing_data_control, testing_data_mean_control, testing_data_std_control, cv_control, data_inclusion_flag_control, xdata_raw_0, time_touse)

    raw_data_q = operation_code.generate_output_data(xsample_processed, testing_data_mean, testing_data_std, cv, data_inclusion_flag_test, xdata_raw_q, time_touse)

    raw_data_q_feat = operation_code.generate_output_data(xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat, data_inclusion_flag_feat, xdata_raw_q_feat, time_touse)

    sep = [[' '] for i in range(len(raw_data_q))]

    raw_data_combined = list(raw_data_control + sep for raw_data_control, sep in zip(raw_data_control, sep))
    raw_data_combined = list(raw_data_combined + raw_data_q for raw_data_combined, raw_data_q in zip(raw_data_combined, raw_data_q))

    raw_data_feat_combined = list(raw_data_feat + sep for raw_data_feat, sep in zip(raw_data_feat, sep))
    raw_data_feat_combined = list(raw_data_feat_combined + raw_data_q_feat for raw_data_feat_combined, raw_data_q_feat in zip(raw_data_feat_combined, raw_data_q_feat))

    output_features_combined = list(output_features0 + sep for output_features0, sep in zip(output_features0, sep))
    output_features_combined = list(output_features_combined + output_features for output_features_combined, output_features in zip(output_features_combined, output_features))

    if neg_analysis_flag:
        data_inclusion_flag = np.zeros((time_touse * n_spots, 1))
        for time in range(time_touse):
            neg_spots_shifted = [x + time * n_spots for x in neg_spots]
            test_spots_shifted = [x + time * n_spots for x in test_spots]
            if time < time_touse:
                data_inclusion_flag[neg_spots_shifted, :] = 2
                data_inclusion_flag[test_spots_shifted, :] = 1

        testing_data_mean, testing_data_mean_control, testing_data_std, testing_data_std_control, xsample_processed, data_inclusion_flag, thresh_all = operation_code.spot_exclusion_all_data(
            testing_data, data_inclusion_flag, n_sigma, accum_flag, time_touse, n_spots, m_thresh, m_thresh_bool,
            neg_analysis_flag, 0, keep_zero_time)



        cv = testing_data_std / testing_data_mean
        data_inclusion_flag_test = np.transpose(np.resize(data_inclusion_flag, (time_touse, data_size[0])))

        xzero_mean_q = operation_code.calculate_zero_1sample(data_zero, data_inclusion_flag, feature_set_opt_0, accum_flag,time_touse)
        signal_q_neg, xdata_raw_q_neg, xdata_raw_q_zerosub = operation_code.predict_concentration(data_checked, data_inclusion_flag,feature_set_opt_0, time_touse,xzero_mean_q,zero_subtraction_flag, accum_flag,time_offset)

        raw_data_q = operation_code.generate_output_data(xsample_processed, testing_data_mean, testing_data_std, cv,data_inclusion_flag_test, xdata_raw_q_neg, time_touse)

        raw_data_combined = list(raw_data_combined + sep for raw_data_combined, sep in zip(raw_data_combined, sep))
        raw_data_combined = list(raw_data_combined + raw_data_q for raw_data_combined, raw_data_q in zip(raw_data_combined, raw_data_q))

        r2_sequence, cv_sequence, loss_sequence, spot_exclusion_sequence, feature_set_opt, \
            = operation_code.feature_selection(x_data, y_labels, data_inclusion_flag, time_touse, zero_cols,
                                            zero_subtraction_flag, metric_flag)

        output_features = operation_code.generate_output_features(r2_sequence, cv_sequence, loss_sequence,
                                                                spot_exclusion_sequence, feature_set_opt)

        xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat, data_inclusion_flag_feat = operation_code.prep_test_sample_features(
            testing_data, data_inclusion_flag, feature_set_opt, time_touse)
        xsample_features = np.transpose(np.resize(xsample_features, (time_touse, data_size[0])))
        data_inclusion_flag_feat = np.transpose(np.resize(data_inclusion_flag_feat, (time_touse, data_size[0])))

        signal_q_feat, xdata_raw_q_feat, xdata_raw_q_zerosub_feat = operation_code.predict_concentration(data_checked,data_inclusion_flag,feature_set_opt,time_touse,
                                                                                                        xzero_mean_q,zero_subtraction_flag,accum_flag,time_offset)

        raw_data_q_feat = operation_code.generate_output_data(xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat,
                                                            data_inclusion_flag_feat, xdata_raw_q_feat, time_touse)

        raw_data_feat_combined = list(raw_data_feat_combined + sep for raw_data_feat_combined, sep in zip(raw_data_feat_combined, sep))
        raw_data_feat_combined = list(raw_data_feat_combined + raw_data_q_feat for raw_data_feat_combined, raw_data_q_feat in zip(raw_data_feat_combined, raw_data_q_feat))


    if differential_flag:
        data_inclusion_flag = np.zeros((time_touse * n_spots, 1))
        for time in range(time_touse):
            neg_spots_shifted = [x + time * n_spots for x in neg_spots]
            test_spots_shifted = [x + time * n_spots for x in test_spots]
            if time < time_touse:
                data_inclusion_flag[neg_spots_shifted, :] = 2
                data_inclusion_flag[test_spots_shifted, :] = 1

        testing_data_mean, testing_data_mean_control, testing_data_std, testing_data_std_control, xsample_processed, data_inclusion_flag, thresh_all = operation_code.spot_exclusion_all_data(
            testing_data, data_inclusion_flag, n_sigma, accum_flag, time_touse, n_spots, m_thresh, m_thresh_bool,
            0, differential_flag, keep_zero_time)

        cv = testing_data_std / testing_data_mean
        data_inclusion_flag_test = np.transpose(np.resize(data_inclusion_flag, (time_touse, data_size[0])))

        xzero_mean_q = operation_code.calculate_zero_1sample(data_zero, data_inclusion_flag, feature_set_opt_0, accum_flag,
                                                            time_touse)
        signal_q_neg, xdata_raw_diff, xdata_raw_q_zerosub = operation_code.predict_concentration(data_checked, data_inclusion_flag, feature_set_opt_0, time_touse, xzero_mean_q,
                                                                                                zero_subtraction_flag, accum_flag, time_offset)

        raw_data_q = operation_code.generate_output_data(xsample_processed, testing_data_mean, testing_data_std, cv,
                                                        data_inclusion_flag_test, xdata_raw_diff, time_touse)

        raw_data_combined = list(raw_data_combined + sep for raw_data_combined, sep in zip(raw_data_combined, sep))
        raw_data_combined = list(raw_data_combined + raw_data_q for raw_data_combined, raw_data_q in zip(raw_data_combined, raw_data_q))

        r2_sequence, cv_sequence, loss_sequence, spot_exclusion_sequence, feature_set_opt, \
            = operation_code.feature_selection(x_data, y_labels, data_inclusion_flag, time_touse, zero_cols,
                                            zero_subtraction_flag, metric_flag)

        output_features = operation_code.generate_output_features(r2_sequence, cv_sequence, loss_sequence,
                                                                spot_exclusion_sequence, feature_set_opt)

        xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat, data_inclusion_flag_feat = operation_code.prep_test_sample_features(
            testing_data, data_inclusion_flag, feature_set_opt, time_touse)
        xsample_features = np.transpose(np.resize(xsample_features, (time_touse, data_size[0])))
        data_inclusion_flag_feat = np.transpose(np.resize(data_inclusion_flag_feat, (time_touse, data_size[0])))

        signal_q_feat, xdata_raw_q_feat, xdata_raw_q_zerosub_feat = operation_code.predict_concentration(data_checked,data_inclusion_flag,feature_set_opt,time_touse,
                                                                                                        xzero_mean_q,zero_subtraction_flag,accum_flag,time_offset)

        raw_data_q_feat = operation_code.generate_output_data(xsample_features, xsample_mean_feat, xsample_std_feat,xsample_cv_feat,
                                                            data_inclusion_flag_feat, xdata_raw_q_feat, time_touse)

        raw_data_feat_combined = list(raw_data_feat_combined + sep for raw_data_feat_combined, sep in zip(raw_data_feat_combined, sep))
        raw_data_feat_combined = list(raw_data_feat_combined + raw_data_q_feat for raw_data_feat_combined, raw_data_q_feat in zip(raw_data_feat_combined, raw_data_q_feat))


    if differential_flag and neg_analysis_flag:
        data_inclusion_flag = np.zeros((time_touse * n_spots, 1))
        for time in range(time_touse):
            neg_spots_shifted = [x + time * n_spots for x in neg_spots]
            test_spots_shifted = [x + time * n_spots for x in test_spots]
            if time < time_touse:
                data_inclusion_flag[neg_spots_shifted, :] = 2
                data_inclusion_flag[test_spots_shifted, :] = 1

        testing_data_mean, testing_data_mean_control, testing_data_std, testing_data_std_control, xsample_processed, data_inclusion_flag, thresh_all = operation_code.spot_exclusion_all_data(
            testing_data, data_inclusion_flag, n_sigma, accum_flag, time_touse, n_spots, m_thresh, m_thresh_bool,
            neg_analysis_flag, differential_flag, keep_zero_time)

        cv = testing_data_std / testing_data_mean
        data_inclusion_flag_test = np.transpose(np.resize(data_inclusion_flag, (time_touse, data_size[0])))

        xzero_mean_q = operation_code.calculate_zero_1sample(data_zero, data_inclusion_flag, feature_set_opt_0, accum_flag,
                                                            time_touse)
        signal_q_neg, xdata_raw_diff_neg, xdata_raw_q_zerosub = operation_code.predict_concentration(data_checked, data_inclusion_flag, feature_set_opt_0, time_touse, xzero_mean_q,
                                                                                                zero_subtraction_flag, accum_flag, time_offset)

        raw_data_q = operation_code.generate_output_data(xsample_processed, testing_data_mean, testing_data_std, cv,
                                                        data_inclusion_flag_test, xdata_raw_diff_neg, time_touse)

        raw_data_combined = list(raw_data_combined + sep for raw_data_combined, sep in zip(raw_data_combined, sep))
        raw_data_combined = list(raw_data_combined + raw_data_q for raw_data_combined, raw_data_q in zip(raw_data_combined, raw_data_q))

        r2_sequence, cv_sequence, loss_sequence, spot_exclusion_sequence, feature_set_opt, \
            = operation_code.feature_selection(x_data, y_labels, data_inclusion_flag, time_touse, zero_cols,
                                            zero_subtraction_flag, metric_flag)

        output_features = operation_code.generate_output_features(r2_sequence, cv_sequence, loss_sequence, spot_exclusion_sequence, feature_set_opt)

        xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat, data_inclusion_flag_feat = operation_code.prep_test_sample_features(
            testing_data, data_inclusion_flag, feature_set_opt, time_touse)

        xsample_features = np.transpose(np.resize(xsample_features, (time_touse, data_size[0])))
        data_inclusion_flag_feat = np.transpose(np.resize(data_inclusion_flag_feat, (time_touse, data_size[0])))

        signal_q_feat, xdata_raw_q_feat, xdata_raw_q_zerosub_feat = operation_code.predict_concentration(data_checked,data_inclusion_flag,feature_set_opt,
                                                        time_touse,xzero_mean_q,zero_subtraction_flag,accum_flag, time_offset)

        raw_data_q_feat = operation_code.generate_output_data(xsample_features, xsample_mean_feat, xsample_std_feat, xsample_cv_feat,data_inclusion_flag_feat,
                                                            xdata_raw_q_feat, time_touse)

        raw_data_feat_combined = list(raw_data_feat_combined + sep for raw_data_feat_combined, sep in zip(raw_data_feat_combined, sep))
        raw_data_feat_combined = list(raw_data_feat_combined + raw_data_q_feat for raw_data_feat_combined, raw_data_q_feat in zip(raw_data_feat_combined, raw_data_q_feat))




    raw_data = np.concatenate((xsample_processed, xdata_raw_0, xdata_raw_q), axis=0)
    if os.path.isdir(output_path + '/' + prediction_folder_path) == False:
        os.mkdir(output_path + '/' + prediction_folder_path)

    np.savetxt(output_path + '/' + prediction_folder_path + '/' + test + '_output_sig.csv', [signal_0, signal_q],
                delimiter=",")
    np.savetxt(output_path + '/' + prediction_folder_path + '/' + test + '_margins.csv', thresh_all,
                delimiter=",")
    np.savetxt(output_path + '/' + prediction_folder_path + '/' + test + '_rawdata.csv', raw_data_combined, fmt='%s',
                delimiter=",")

    if os.path.isdir(output_path + '/' + features_folder) == False:
        os.mkdir(output_path + '/' + features_folder)

    np.savetxt(output_path + '/' + features_folder + '/' + test_spots + '_rawdata_features.csv', raw_data_feat_combined, fmt='%s',
                delimiter=",")

    np.savetxt(output_path + '/' + features_folder + '/' + test + '_rawdata_iterations.csv', output_features_combined, fmt='%s',
                delimiter=",")


def main(): 
    print('hello')
    output_path = r'/home/pi/Desktop/Auionreduction/output_data_2023-07-12-21:58:04/csv' #main path with the code
    sample_folder = r'/home/pi/Desktop/alan/samples_10_19_22' #path with the data
    prediction_folder = 'samples_10_19_22' #folder to save predicted concentration
    features_folder = 'predic/home/pi/Desktop/alanted_concentration_features' #folder to save predicted concentration
    test_filename = 'testing_sample_0ng.mL_1' #name of the data file to test
    zero_filename = 'testing_sample_0ng.mL_2' #name of the data file to test
    encapsulate(sample_folder, output_path, prediction_folder, features_folder, zero_filename, test_filename)

if __name__ == "__main__":
    print('flag1')
    main()
