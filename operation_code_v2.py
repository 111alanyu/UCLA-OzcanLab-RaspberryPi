import numpy as np
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import curve_fit


def spot_exclusion_all_data(xdata, data_inclusion_flag, n_sigma, accum_flag, n_times, n_spots,m_thresh,m_thresh_bool,neg_analysis_flag, differential_flag, keep_zero_time):
    n_samples = xdata.shape[1]
    for n_sample in range(n_samples):
        xdata_sample = xdata[:, n_sample]
        data_inclusion_flag_sample = data_inclusion_flag[:, n_sample]
        if differential_flag == 0:
            xdata_sample_mean, xdata_sample_mean_0, xsample_std, xsample_std_0, xsample_processed, data_inclusion_flag_sample, thresh_sample = spot_exclusion_1sample(xdata_sample, data_inclusion_flag_sample,
                                                                                n_sigma, accum_flag, n_times, n_spots,m_thresh,m_thresh_bool,neg_analysis_flag)
        else:

            xdata_sample_mean, xdata_sample_mean_0, xsample_std, xsample_std_0, xsample_processed, data_inclusion_flag_sample, thresh_sample = spot_exclusion_1sample_differential(xdata_sample, data_inclusion_flag_sample, n_sigma,
                                                                                accum_flag, n_times, n_spots, m_thresh, m_thresh_bool, neg_analysis_flag, keep_zero_time)

        if n_sample == 0:
            xdata_mean = [xdata_sample_mean]
            xdata_mean_control = [xdata_sample_mean_0]
            thresh_data = [thresh_sample]
            xdata_std = [xsample_std]
            xdata_std_control = [xsample_std_0]
            xsample_processed_all = [xsample_processed]

            data_inclusion_flag_updated = [data_inclusion_flag_sample.tolist()]
        else:
            xdata_mean += [xdata_sample_mean]
            xdata_mean_control += [xdata_sample_mean_0]
            thresh_data += [thresh_sample]

            xdata_std += [xsample_std]
            xdata_std_control += [xsample_std_0]

            data_inclusion_flag_updated += [data_inclusion_flag_sample.tolist()]
            xsample_processed_all += [xsample_processed]

    xdata_mean = np.transpose(np.array(xdata_mean))
    xdata_mean_control = np.transpose(np.array(xdata_mean_control))
    xdata_std = np.transpose(np.array(xdata_std))
    xdata_std_control = np.transpose(np.array(xdata_std_control))
    thresh_data = np.transpose(np.array(thresh_data))

    data_inclusion_flag_updated = np.transpose(np.array(data_inclusion_flag_updated))

    if n_samples == 1:
        xsample_processed_all = np.transpose(np.resize(xsample_processed_all, (n_times, n_spots)))
    return xdata_mean, xdata_mean_control, xdata_std, xdata_std_control, xsample_processed_all, data_inclusion_flag_updated, thresh_data

def spot_exclusion_1sample(xsample, data_inclusion_flag_sample, n_sigma, accum_flag, n_times, n_spots,m_thresh,m_thresh_bool, neg_analysis_flag):
    xsample_processed = np.zeros((xsample.shape))

    for time in range(n_times):
        xsample_single = np.array(xsample[time * n_spots:(time + 1) * n_spots])
        data_inclusion_flag_single = np.array(data_inclusion_flag_sample[time * n_spots:(time + 1) * n_spots])

        test_mean_red = np.mean(xsample_single[data_inclusion_flag_single == 1])
        neg_mean_red = np.mean(xsample_single[data_inclusion_flag_single == 2])
        pos_mean = np.mean(xsample_single[data_inclusion_flag_single == 0])

        test_std_red = np.std(xsample_single[data_inclusion_flag_single == 1])
        neg_std_red = np.std(xsample_single[data_inclusion_flag_single == 2])
        pos_std = np.std(xsample_single[data_inclusion_flag_single == 0])

        if time == 0:
            xsample_mean_0 = [test_mean_red, neg_mean_red, pos_mean]
            xsample_std_0 = [test_std_red, neg_std_red, pos_std]
        else:
            xsample_mean_0 += [test_mean_red, neg_mean_red, pos_mean]
            xsample_std_0 += [test_std_red, neg_std_red, pos_std]

    for time in range(n_times):
        xsample_single = np.array(xsample[time * n_spots:(time + 1) * n_spots])
        data_inclusion_flag_single = np.array(data_inclusion_flag_sample[time * n_spots:(time + 1) * n_spots])

        test_mean = np.mean(xsample_single[data_inclusion_flag_single == 1])
        test_std = np.std(xsample_single[data_inclusion_flag_single == 1], ddof=1)

        neg_mean = np.mean(xsample_single[data_inclusion_flag_single == 2])
        neg_std = np.std(xsample_single[data_inclusion_flag_single == 2], ddof=1)

        pos_mean = np.mean(xsample_single[data_inclusion_flag_single == 0])

        low_test = test_mean - n_sigma * test_std*test_multiplier
        high_test = test_mean + n_sigma * test_std*test_multiplier

        low_neg = neg_mean - n_sigma * neg_std*neg_multiplier
        high_neg = neg_mean + n_sigma * neg_std*neg_multiplier

        shifted_test_spots = [x + time * n_spots for x in test_spots]
        for spot in shifted_test_spots:
            if m_thresh_bool:
                if np.round(xsample[spot],3) < np.round(low_test,3)-m_thresh or np.round(xsample[spot],3) > np.round(high_test,3)+m_thresh:
                    data_inclusion_flag_sample[spot] = -1
                    xsample_processed[spot] = -1
                elif (np.round(xsample[spot],3) >= np.round(low_test,3) - m_thresh and np.round(xsample[spot],3) <= np.round(low_test,3)) or (np.round(xsample[spot],3) <= np.round(high_test,3) + m_thresh and np.round(xsample[spot],3) >= np.round(high_test,3)):
                    data_inclusion_flag_sample[spot] = -2
                    xsample_processed[spot] = -2
                else:
                    xsample_processed[spot] = xsample[spot]
            else:
                if np.round(xsample[spot],3) < np.round(low_test,3) or np.round(xsample[spot],3) > np.round(high_test,3):
                    data_inclusion_flag_sample[spot] = -1
                    xsample_processed[spot] = -1
                else:
                    xsample_processed[spot] = xsample[spot]

        shifted_neg_spots = [x + time * n_spots for x in neg_spots]

        if neg_analysis_flag == 1:
            for spot in shifted_neg_spots:
                if m_thresh_bool:
                    if xsample[spot] < np.round(low_neg,3)-m_thresh or xsample[spot] > np.round(high_neg,3)+m_thresh:
                        data_inclusion_flag_sample[spot] = -1
                        xsample_processed[spot] = -1
                    elif (xsample[spot] >= np.round(low_neg,3)-m_thresh and xsample[spot] <= np.round(low_neg,3)) or (xsample[spot] <= np.round(high_neg,3)+m_thresh and xsample[spot] >= np.round(high_neg,3)):
                        data_inclusion_flag_sample[spot] = -2
                        xsample_processed[spot] = -2
                    else:
                        xsample_processed[spot] = xsample[spot]
                else:
                    if xsample[spot] < np.round(low_neg,3) or xsample[spot] > np.round(high_neg,3):
                        data_inclusion_flag_sample[spot] = -1
                        xsample_processed[spot] = -1
                    else:
                        xsample_processed[spot] = xsample[spot]
        else:
            for spot in shifted_neg_spots:
                xsample_processed[spot] = xsample[spot]


        if time == 0:
            thresh_all = [low_test, high_test]
        else:
            thresh_all += [low_test, high_test]

    for time in range(n_times):
        shifted_pos_spots = [x + time * n_spots for x in pos_spots]
        for spot in shifted_pos_spots:
            xsample_processed[spot] = xsample[spot]

    xsample_processed, data_inclusion_flag_sample = error_accumulation(data_inclusion_flag_sample, xsample_processed, xsample, accum_flag, n_times)

    for time in range(n_times):
        xsample_single = np.array(xsample[time * n_spots:(time + 1) * n_spots])

        data_inclusion_flag_single = np.array(data_inclusion_flag_sample[time * n_spots:(time + 1) * n_spots])

        test_mean_red = np.mean(xsample_single[data_inclusion_flag_single == 1])
        neg_mean_red = np.mean(xsample_single[data_inclusion_flag_single == 2])
        pos_mean = np.mean(xsample_single[data_inclusion_flag_single == 0])

        test_std_red = np.std(xsample_single[data_inclusion_flag_single == 1])
        neg_std_red = np.std(xsample_single[data_inclusion_flag_single == 2])
        pos_std = np.std(xsample_single[data_inclusion_flag_single == 0])

        if time == 0:
            xsample_mean = [test_mean_red, neg_mean_red, pos_mean]
            xsample_std = [test_std_red, neg_std_red, pos_std]
        else:
            xsample_mean += [test_mean_red, neg_mean_red, pos_mean]
            xsample_std += [test_std_red, neg_std_red, pos_std]

    return xsample_mean, xsample_mean_0, xsample_std, xsample_std_0, xsample_processed, data_inclusion_flag_sample, thresh_all

def spot_exclusion_1sample_differential(xsample, data_inclusion_flag_sample, n_sigma, accum_flag, n_times, n_spots,m_thresh,m_thresh_bool, neg_analysis_flag, keep_zero_time):
    xsample_processed = np.zeros((xsample.shape))

    xsample_differential = np.zeros((xsample.shape))
    xsample_differential[0:n_spots] = xsample[0:n_spots]
    for time in range(n_times-1):
        xsample_differential[n_spots*(time+1):n_spots*(time+2)] = xsample[n_spots*(time+1):n_spots*(time+2)] - xsample[n_spots*time:n_spots*(time+1)]

    for time in range(n_times):
        xsample_single = np.array(xsample[time * n_spots:(time + 1) * n_spots])
        data_inclusion_flag_single = np.array(data_inclusion_flag_sample[time * n_spots:(time + 1) * n_spots])

        test_mean_red = np.mean(xsample_single[data_inclusion_flag_single == 1])
        neg_mean_red = np.mean(xsample_single[data_inclusion_flag_single == 2])
        pos_mean = np.mean(xsample_single[data_inclusion_flag_single == 0])

        test_std_red = np.std(xsample_single[data_inclusion_flag_single == 1])
        neg_std_red = np.std(xsample_single[data_inclusion_flag_single == 2])
        pos_std = np.std(xsample_single[data_inclusion_flag_single == 0])

        if time == 0:
            xsample_mean_0 = [test_mean_red, neg_mean_red, pos_mean]
            xsample_std_0 = [test_std_red, neg_std_red, pos_std]
        else:
            xsample_mean_0 += [test_mean_red, neg_mean_red, pos_mean]
            xsample_std_0 += [test_std_red, neg_std_red, pos_std]


    for time in range(n_times):

        xsample_single_differential = np.array(xsample_differential[time * n_spots:(time + 1) * n_spots])
        data_inclusion_flag_single = np.array(data_inclusion_flag_sample[time * n_spots:(time + 1) * n_spots])

        test_mean = np.mean(xsample_single_differential[data_inclusion_flag_single == 1])
        test_std = np.std(xsample_single_differential[data_inclusion_flag_single == 1], ddof=1)

        neg_mean = np.mean(xsample_single_differential[data_inclusion_flag_single == 2])
        neg_std = np.std(xsample_single_differential[data_inclusion_flag_single == 2], ddof=1)

        pos_mean = np.mean(xsample_single_differential[data_inclusion_flag_single == 0])

        low_test = test_mean - n_sigma * test_std*test_multiplier
        high_test = test_mean + n_sigma * test_std*test_multiplier

        low_neg = neg_mean - n_sigma * neg_std*neg_multiplier
        high_neg = neg_mean + n_sigma * neg_std*neg_multiplier

        shifted_test_spots = [x + time * n_spots for x in test_spots]

        for spot in shifted_test_spots:
            if m_thresh_bool:
                if np.round(xsample_differential[spot],3) < np.round(low_test,3)-m_thresh or np.round(xsample_differential[spot],3) > np.round(high_test,3)+m_thresh:
                    data_inclusion_flag_sample[spot] = -1
                    xsample_processed[spot] = -1
                elif (np.round(xsample_differential[spot],3) >= np.round(low_test,3) - m_thresh and np.round(xsample_differential[spot],3) <= np.round(low_test,3)) or (np.round(xsample_differential[spot],3) <= np.round(high_test,3) + m_thresh and np.round(xsample_differential[spot],3) >= np.round(high_test,3)):
                    #data_inclusion_flag_sample[spot] = -2
                    #xsample_processed[spot] = -2
                    xsample_processed[spot] = xsample[spot]
                else:
                    xsample_processed[spot] = xsample[spot]
            else:
                if np.round(xsample_differential[spot],3) < np.round(low_test,3) or np.round(xsample_differential[spot],3) > np.round(high_test,3):
                    data_inclusion_flag_sample[spot] = -1
                    xsample_processed[spot] = -1
                else:
                    xsample_processed[spot] = xsample[spot]

        shifted_neg_spots = [x + time * n_spots for x in neg_spots]

        if neg_analysis_flag == 1:
            for spot in shifted_neg_spots:
                if m_thresh_bool:
                    if xsample_differential[spot] < np.round(low_neg,3)-m_thresh or xsample_differential[spot] > np.round(high_neg,3)+m_thresh:
                        data_inclusion_flag_sample[spot] = -1
                        xsample_processed[spot] = -1
                    elif (xsample_differential[spot] >= np.round(low_neg,3)-m_thresh and xsample_differential[spot] <= np.round(low_neg,3)) or (xsample_differential[spot] <= np.round(high_neg,3)+m_thresh and xsample_differential[spot] >= np.round(high_neg,3)):
                        #data_inclusion_flag_sample[spot] = -2
                        #xsample_processed[spot] = -2
                        xsample_processed[spot] = xsample[spot]
                    else:
                        xsample_processed[spot] = xsample[spot]
                else:
                    if xsample_differential[spot] < np.round(low_neg,3) or xsample_differential[spot] > np.round(high_neg,3):
                        data_inclusion_flag_sample[spot] = -1
                        xsample_processed[spot] = -1
                    else:
                        xsample_processed[spot] = xsample[spot]
        else:
            for spot in shifted_neg_spots:
                xsample_processed[spot] = xsample[spot]


        if time == 0:
            thresh_all = [low_test, high_test]
        else:
            thresh_all += [low_test, high_test]

    for time in range(n_times):
        shifted_pos_spots = [x + time * n_spots for x in pos_spots]
        for spot in shifted_pos_spots:
            xsample_processed[spot] = xsample[spot]


    for spot in range(n_spots):
        if keep_zero_time == 0:
            spot_array = [np.int32(x) for x in np.linspace(0, n_times - 1, n_times) * n_spots + spot]
        else:
            spot_array = [np.int32(x) for x in np.linspace(1, n_times - 1, n_times-1) * n_spots + spot]
            data_inclusion_flag_sample[neg_spots] = 2
            data_inclusion_flag_sample[test_spots] = 1
            xsample_processed[neg_spots] = xsample[neg_spots]
            xsample_processed[test_spots] = xsample[test_spots]

        for spot_input_ind in range(len(spot_array)):
            if data_inclusion_flag_sample[spot_array[spot_input_ind]] == -1:
                data_inclusion_flag_sample[spot_array[spot_input_ind:len(spot_array)]] = -1
                xsample_processed[spot_array[spot_input_ind:len(spot_array)]] = -1
                break


    for time in range(n_times):
        xsample_single = np.array(xsample[time * n_spots:(time + 1) * n_spots])

        data_inclusion_flag_single = np.array(data_inclusion_flag_sample[time * n_spots:(time + 1) * n_spots])

        test_mean_red = np.mean(xsample_single[data_inclusion_flag_single == 1])
        neg_mean_red = np.mean(xsample_single[data_inclusion_flag_single == 2])
        pos_mean = np.mean(xsample_single[data_inclusion_flag_single == 0])

        test_std_red = np.std(xsample_single[data_inclusion_flag_single == 1])
        neg_std_red = np.std(xsample_single[data_inclusion_flag_single == 2])
        pos_std = np.std(xsample_single[data_inclusion_flag_single == 0])

        if time == 0:
            xsample_mean = [test_mean_red, neg_mean_red, pos_mean]
            xsample_std = [test_std_red, neg_std_red, pos_std]
        else:
            xsample_mean += [test_mean_red, neg_mean_red, pos_mean]
            xsample_std += [test_std_red, neg_std_red, pos_std]

    return xsample_mean, xsample_mean_0, xsample_std, xsample_std_0, xsample_processed, data_inclusion_flag_sample, thresh_all

def sensor_exclusion(ylabels, xdata, xdata_mean, xdata_mean_control, data_inclusion_flag, n_sigma_sensor, cv_sensor, n_times):

    exclusion_array_neg = np.zeros((xdata_mean.shape))
    exclusion_array_pos = np.zeros((xdata_mean.shape))
    exclusion_array_cv = np.zeros((xdata_mean.shape))
    exclusion_array_nodata = np.zeros((xdata_mean.shape))

    neg_low_arr = []
    neg_high_arr = []

    pos_low_arr = []
    pos_high_arr = []

    for n_time in range(n_times):
        neg_mean = np.mean(xdata_mean[3 * n_time + 1, :])
        neg_std = np.std(xdata_mean[3 * n_time + 1, :])
        neg_low = neg_mean - n_sigma_sensor * neg_std
        neg_low_arr.append(neg_low)
        neg_high = neg_mean + n_sigma_sensor * neg_std
        neg_high_arr.append(neg_high)

        pos_mean = np.mean(xdata_mean[3 * n_time + 2, :])
        pos_std = np.std(xdata_mean[3 * n_time + 2, :])
        pos_low = pos_mean - n_sigma_sensor * pos_std
        pos_low_arr.append(pos_low)
        pos_high = pos_mean + n_sigma_sensor * pos_std
        pos_high_arr.append(pos_high)

        for n_sample in range(xdata.shape[1]):
            if xdata_mean[3 * n_time + 1, n_sample] < neg_low or xdata_mean[3 * n_time + 1, n_sample] > neg_high:
                exclusion_array_neg[3 * n_time + 1, n_sample] = 1
                data_inclusion_flag[:, n_sample] = 0
            if xdata_mean[3 * n_time + 2, n_sample] < pos_low or xdata_mean[3 * n_time + 2, n_sample] > pos_high:
                exclusion_array_pos[3 * n_time + 1, n_sample] = 1
                data_inclusion_flag[:, n_sample] = 0

            pos_spots_shifted = [x + n_time * n_spots for x in pos_spots]
            if np.std(xdata[pos_spots_shifted, n_sample])/np.mean(xdata[pos_spots_shifted, n_sample]) > cv_sensor:
                exclusion_array_cv[3 * n_time + 2, n_sample] = 1
                data_inclusion_flag[:, n_sample] = 0
            if xdata_mean[3 * n_time, n_sample] == 1:
                exclusion_array_nodata[3 * n_time, n_sample] = 1
                data_inclusion_flag[:, n_sample] = 0


    flag_array = np.zeros((xdata.shape[1]))
    for n_sample in range(xdata.shape[1]):
        if np.sum(exclusion_array_neg[:,n_sample]) > 0:
            flag_array[n_sample] = 1
            data_inclusion_flag[:,n_sample] += -11
        if np.sum(exclusion_array_pos[:,n_sample]) > 0:
            flag_array[n_sample] = 2
            data_inclusion_flag[:, n_sample] += -12*100
        if np.sum(exclusion_array_cv[:,n_sample]) > 0:
            flag_array[n_sample] = 3
            data_inclusion_flag[:, n_sample] += -13*10000
        if np.sum(exclusion_array_nodata[:,n_sample]) > 0:
            flag_array[n_sample] = 4
            data_inclusion_flag[:, n_sample] += -14*1000000

    samples = np.linspace(0, xdata.shape[1]-1, xdata.shape[1])
    del_indexes = flag_array > 0

    ylabels_del = np.delete(ylabels, del_indexes)
    xdata_del = np.delete(xdata, del_indexes, 1)
    xdata_mean_del = np.delete(xdata_mean, del_indexes, 1)
    xdata_mean_control_del = np.delete(xdata_mean_control, del_indexes, 1)
    data_inclusion_flag_del = np.delete(data_inclusion_flag, del_indexes, 1)

    return xdata_del, xdata_mean_del, xdata_mean_control_del, ylabels_del, flag_array, data_inclusion_flag, data_inclusion_flag_del, neg_low_arr, neg_high_arr, pos_low_arr, pos_high_arr

def test_sample_control(testing_data, testing_data_mean, data_inclusion_flag, neg_low, neg_high, pos_low, pos_high, n_sigma_sensor, cv_sensor, n_times):

    xdata = testing_data
    xdata_mean = testing_data_mean
    exclusion_array_neg = np.zeros((xdata_mean.shape))
    exclusion_array_pos = np.zeros((xdata_mean.shape))
    exclusion_array_cv = np.zeros((xdata_mean.shape))
    exclusion_array_nodata = np.zeros((xdata_mean.shape))

    for n_time in range(n_times):
        if xdata_mean[3 * n_time + 1] < neg_low[n_time] or xdata_mean[3 * n_time + 1] > neg_high[n_time]:
            exclusion_array_neg[3 * n_time + 1] = 1
            data_inclusion_flag[:] = 0
        if xdata_mean[3 * n_time + 2] < pos_low[n_time] or xdata_mean[3 * n_time + 2] > pos_high[n_time]:
            exclusion_array_pos[3 * n_time + 1] = 1
            data_inclusion_flag[:] = 0

        pos_spots_shifted = [x + n_time * n_spots for x in pos_spots]
        if np.std(xdata[pos_spots_shifted]) / np.mean(xdata[pos_spots_shifted]) > cv_sensor:
            exclusion_array_cv[3 * n_time + 2] = 1
            data_inclusion_flag[:] = 0
        if xdata_mean[3 * n_time] == 1:
            exclusion_array_nodata[3 * n_time] = 1
            data_inclusion_flag[:] = 0


    if np.sum(exclusion_array_neg) > 0:
        data_inclusion_flag[:] += -11
    if np.sum(exclusion_array_pos) > 0:
        data_inclusion_flag[:] += -12*100
    if np.sum(exclusion_array_cv) > 0:
        data_inclusion_flag[:] += -13*10000
    if np.sum(exclusion_array_nodata) > 0:
        data_inclusion_flag[:] += -14*1000000

    return testing_data, data_inclusion_flag

def prep_data(x_data, iteration_features, data_inclusion_flag, time_touse, zero_cols, zero_subtraction_flag):
    n_samples = x_data.shape[1]
    for n_sample in range(n_samples):
        for n_time in range(time_touse):

            current_data_inclusion_flag = data_inclusion_flag[n_time*n_spots : (n_time+1) * n_spots, 0]

            current_xdata = x_data[n_time*n_spots : (n_time+1) * n_spots, n_sample]
            test_spots_bool = np.logical_and(current_data_inclusion_flag == 1, iteration_features == 1)
            neg_spots_bool = np.logical_and(current_data_inclusion_flag == 2, iteration_features == 1)
            pos_spots_bool = np.logical_and(current_data_inclusion_flag == 0, iteration_features == 1)
            x_test_mean = np.mean(current_xdata[test_spots_bool])
            x_neg_mean = np.mean(current_xdata[neg_spots_bool])
            x_pos_mean = np.mean(current_xdata[pos_spots_bool])
            if np.isnan(x_pos_mean):
                x_pos_mean = 0
            if n_time == 0:
                xmean_sample = [x_test_mean, x_neg_mean, x_pos_mean]
            else:
                xmean_sample += [x_test_mean, x_neg_mean, x_pos_mean]

        if n_sample == 0:
            xmean_data = [xmean_sample]
        else:
            xmean_data += [xmean_sample]

    xmean_data = np.transpose(np.array(xmean_data))
    xmean_data, xzero_mean = normalize_signals(xmean_data, time_touse, zero_cols, zero_subtraction_flag)
    x_zero = calculate_zero(xmean_data, time_touse, zero_cols)

    x_data_final = np.zeros((n_samples))
    for n_sample in range(n_samples):
        if time_touse == 1:
            sig = xmean_data[0, n_sample]
        else:
            sig = 0
            for n_time in range(time_touse-1):
                x_test = (xmean_data[n_time, n_sample]+xmean_data[n_time+1, n_sample])*0.5/2
                sig += x_test
        #x_data_final[n_sample] = sig/x_zero
        x_data_final[n_sample] = sig
    return x_data_final, xzero_mean

def calc_final_data(xmean_data, n_samples, time_touse):
    x_data_final = np.zeros((n_samples))
    for n_sample in range(n_samples):
        if time_touse == 1:
            sig = xmean_data[0, n_sample]
        else:
            sig = 0
            for n_time in range(time_touse - 1):
                x_test = (xmean_data[n_time, n_sample] + xmean_data[n_time + 1, n_sample]) * 0.5 / 2
                sig += x_test
        # x_data_final[n_sample] = sig/x_zero
        x_data_final[n_sample] = sig
    return x_data_final

def calc_calibration_mean_data(xmean_data, y_labels, time_touse, zero_cols, zero_subtraction_flag, include_yshift_flag):

    xmean_data, xzero_mean = normalize_signals(xmean_data, time_touse, zero_cols, zero_subtraction_flag)
    n_samples = xmean_data.shape[1]

    x_data_final = calc_final_data(xmean_data, n_samples, time_touse)

    if include_yshift_flag:
        pars, cov = curve_fit(f=power_law_shifted, xdata=y_labels, ydata=x_data_final, p0=[0, 0, 0], bounds=(0, np.inf))
    else:
        pars, cov = curve_fit(f=power_law, xdata=y_labels, ydata=x_data_final, p0=[0, 0], bounds=(-np.inf, np.inf))

    return x_data_final, pars

def calc_calibration(r2_sequence, cv_sequence, loss_sequence, xdata_opt_array, y_labels, metric_flag, include_yshift_flag):

    if metric_flag == 0:
        opt_ind = np.argmax(r2_sequence)
    elif metric_flag == 1:
        opt_ind = np.argmin(cv_sequence)
    elif metric_flag == 2:
        opt_ind = np.argmin(loss_sequence)

    xdata_opt = xdata_opt_array[opt_ind, :]

    if include_yshift_flag:
        pars, cov = curve_fit(f=power_law_shifted, xdata=y_labels, ydata=xdata_opt, p0=[0, 0, 0], bounds=(0, np.inf))
    else:
        pars, cov = curve_fit(f=power_law, xdata=y_labels, ydata=xdata_opt, p0=[0, 0], bounds=(-np.inf, np.inf))

    return [r2_sequence[opt_ind], cv_sequence[opt_ind], loss_sequence[opt_ind]], xdata_opt, pars

def feature_selection(x_data, y_labels, data_inclusion_flag, time_touse, zero_cols, zero_subtraction_flag, metric_flag):

    feature_set_bool = np.ones((n_spots))
    feature_set_bool[0] = 0
    feature_set_bool[15] = 0
    feature_set_opt = feature_set_bool
    r2_array = []
    cv_array = []
    loss_array = []
    ind_array = []


    n_samples = x_data.shape[1]
    loss_opt_all = 100
    cv_opt_all = 1
    r2_opt_all = 0
    for iter in range(15):
        left_feature_inds = np.where(feature_set_bool == 1)[0]
        for n_sample in range(n_samples):
            for n_time in range(time_touse):
                data_inclusion_flag_one = data_inclusion_flag[n_time * n_spots: (n_time+1) * n_spots]


                feature_set_quality_control = np.logical_and(np.transpose(data_inclusion_flag_one), feature_set_bool)

                if np.sum(feature_set_quality_control[0,neg_spots]) == 1:
                    left_ind = np.argmax(feature_set_quality_control[0,neg_spots])
                    left_feature_inds = np.delete(left_feature_inds, np.where(left_feature_inds == neg_spots[left_ind]))


                if np.sum(feature_set_quality_control[0,test_spots]) == 1:
                    left_ind = np.argmax(feature_set_quality_control[0,test_spots])
                    left_feature_inds = np.delete(left_feature_inds, np.where(left_feature_inds == test_spots[left_ind]))

        r2 = 0
        cv = 1
        loss = 100
        for remove_feature in range(len(left_feature_inds)):
            iteration_features = np.ones((n_spots))
            iteration_features[np.where(feature_set_bool == 0)] = 0
            ind_toadd = left_feature_inds[remove_feature]
            iteration_features[ind_toadd] = 0

            x_data_sig, xzero_mean = prep_data(x_data, iteration_features, data_inclusion_flag, time_touse, zero_cols, zero_subtraction_flag)

            #r2_iter = r2_score(x_data_sig*10, y_labels)
            correl = np.corrcoef(x_data_sig, y_labels)
            r2_iter = correl[0,1]*correl[0,1]
            cv_iter = calc_cv(x_data_sig, y_labels)
            #loss_iter = calc_loss(x_data_sig, y_labels)
            loss_iter = cv_iter

            if metric_flag == 0:
                if r2_iter > r2:
                    loss = loss_iter
                    r2 = r2_iter
                    cv = cv_iter
                    opt_ind = ind_toadd
                    x_data_opt = x_data_sig
            elif metric_flag == 1:
                if cv_iter < cv:
                    loss = loss_iter
                    r2 = r2_iter
                    cv = cv_iter
                    opt_ind = ind_toadd
                    x_data_opt = x_data_sig
            elif metric_flag == 2:
                if loss_iter < loss:
                    loss = loss_iter
                    r2 = r2_iter
                    cv = cv_iter
                    opt_ind = ind_toadd
                    x_data_opt = x_data_sig

        feature_set_bool[opt_ind] = 0
        r2_array.append(r2)
        cv_array.append(cv)
        loss_array.append(loss)
        ind_array.append(opt_ind)
        if iter == 0:
            xdata_opt_array = np.array([x_data_opt])
        else:
            xdata_opt_array = np.concatenate((xdata_opt_array, np.array([x_data_opt])), axis = 0)


        if metric_flag == 2 and loss < loss_opt_all:
            loss_opt_all = loss
            feature_set_opt = np.ones((n_spots))
            feature_set_opt[np.where(feature_set_bool == 0)]=0
            feature_set_opt[0] = 1
            feature_set_opt[15] = 1
        elif metric_flag == 1 and cv < cv_opt_all:
            cv_opt_all = cv
            feature_set_opt = np.ones((n_spots))
            feature_set_opt[np.where(feature_set_bool == 0)]=0
            feature_set_opt[0] = 1
            feature_set_opt[15] = 1
        elif metric_flag == 0 and r2 > r2_opt_all:
            r2_opt_all = r2
            feature_set_opt = np.ones((n_spots))
            feature_set_opt[np.where(feature_set_bool == 0)]=0
            feature_set_opt[0] = 1
            feature_set_opt[15] = 1
            print(iter)
            print(feature_set_opt)

    print(feature_set_opt)
    return r2_array, cv_array, loss_array, ind_array, np.transpose(np.array([feature_set_opt]))

def prep_test_sample_features(xsample, data_inclusion_flag, feature_set_opt,  n_times):

    xsample_features = np.zeros((xsample.shape))
    data_inclusion_flag_feat = np.zeros((data_inclusion_flag.shape))

    for time in range(n_times):
        xsample_single = np.array(xsample[time * n_spots:(time + 1) * n_spots])
        data_inclusion_flag_single = np.array(data_inclusion_flag[time * n_spots:(time + 1) * n_spots])

        test_spots_bool = np.logical_and(data_inclusion_flag_single == 1, feature_set_opt == 1)
        neg_spots_bool = np.logical_and(data_inclusion_flag_single == 2, feature_set_opt == 1)
        pos_spots_bool = np.logical_and(data_inclusion_flag_single == 0, feature_set_opt == 1)

        test_spots_shifted = [x + time * n_spots for x in test_spots]
        neg_spots_shifted = [x + time * n_spots for x in neg_spots]
        pos_spots_shifted = [x + time * n_spots for x in pos_spots]

        for spot in range(test_spots_bool.shape[0]):
            if test_spots_bool[spot]:
                xsample_features[spot+time*n_spots] = xsample[spot]
                data_inclusion_flag_feat[spot+time*n_spots] = 1

        for spot in range(neg_spots_bool.shape[0]):
            if neg_spots_bool[spot]:
                xsample_features[spot+time*n_spots] = xsample[spot]
                data_inclusion_flag_feat[spot+time*n_spots] = 2

        for spot in range(pos_spots_bool.shape[0]):
            if pos_spots_bool[spot]:
                xsample_features[spot+time*n_spots] = xsample[spot]
                data_inclusion_flag_feat[spot+time*n_spots] = 0

        for spot in range(pos_spots_bool.shape[0]):
            if np.max([test_spots_bool[spot],neg_spots_bool[spot],pos_spots_bool[spot]])==0:
                data_inclusion_flag_feat[spot + time * n_spots] = -1


        test_mean = np.mean(xsample_single[test_spots_bool])
        neg_mean = np.mean(xsample_single[neg_spots_bool])
        pos_mean = np.mean(xsample_single[pos_spots_bool])

        test_std = np.std(xsample_single[data_inclusion_flag_single == 1])
        neg_std = np.std(xsample_single[data_inclusion_flag_single == 2])
        pos_std = np.std(xsample_single[data_inclusion_flag_single == 0])

        if time == 0:
            xsample_mean = [test_mean, neg_mean, pos_mean]
            xsample_std = [test_std, neg_std, pos_std]
            xsample_cv = [test_std/test_mean, neg_std/neg_mean, pos_std/pos_mean]
        else:
            xsample_mean+= [test_mean, neg_mean, pos_mean]
            xsample_std += [test_std, neg_std, pos_std]
            xsample_cv += [test_std/test_mean, neg_std/neg_mean, pos_std/pos_mean]

    return xsample_features, xsample_mean, xsample_std, xsample_cv, data_inclusion_flag_feat


def calc_cv_arr(xdata, y_labels, time_touse):
    n_samples = xdata.shape[1]
    xdata = calc_final_data(xdata, n_samples, time_touse)
    y_labels_single = []
    xdata_means = []
    xdata_stds = []
    xdata_cvs = []
    xcur = [xdata[0]]
    for n_sample in range(n_samples-1):
        n_sample = n_sample + 1
        if (n_sample == 1) and (y_labels[n_sample] != y_labels[n_sample-1]):
            xdata_means.append(xdata[0])
            xdata_stds.append(0)
            xdata_cvs.append(0)
            xcur = [xdata[n_sample]]
            y_labels_single.append(y_labels[0])
        if y_labels[n_sample] != y_labels[n_sample-1]:
            if len(xcur)>1:
                xdata_means.append(np.mean(xcur))
                xdata_stds.append(np.std(xcur))
                xdata_cvs.append(np.std(xcur)/np.mean(xcur))
                y_labels_single.append(y_labels[n_sample])
            else:
                xdata_means.append(xcur[0])
                xdata_stds.append(0)
                xdata_cvs.append(0)
                y_labels_single.append(y_labels[n_sample])
            xcur = [xdata[n_sample]]
        else:
            xcur.append(xdata[n_sample])

    if len(xcur) > 1:
        xdata_means.append(np.mean(xcur))
        xdata_stds.append(np.std(xcur))
        xdata_cvs.append(np.std(xcur) / np.mean(xcur))
        y_labels_single.append(y_labels[n_sample])
    else:
        xdata_means.append(xcur[0])
        xdata_stds.append(0)
        xdata_cvs.append(0)
        y_labels_single.append(y_labels[n_sample])
    mean_cv_data = [y_labels_single, xdata_means, xdata_stds, xdata_cvs]

    return mean_cv_data

def calc_cv_test(xdata, data_size, data_inclusion_flag, feature_set_opt, time_touse):

    xdata = np.transpose(np.resize(xdata, (data_size[0], data_size[1])))
    data_inclusion_flag = np.transpose(np.resize(data_inclusion_flag, (data_size[0], data_size[1])))
    return data_inclusion_flag

def calculate_zero_1sample(data_zero, data_inclusion_flag, feature_set_opt, accum_flag, time_touse):

    for n_time in range(time_touse):

        current_xdata = data_zero[n_time * n_spots: (n_time + 1) * n_spots]


        current_data_inclusion_flag = data_inclusion_flag[n_time * n_spots : (n_time + 1) * n_spots]


        test_spots_bool = np.logical_and(current_data_inclusion_flag == 1, feature_set_opt == 1)
        neg_spots_bool = np.logical_and(current_data_inclusion_flag == 2, feature_set_opt == 1)
        pos_spots_bool = np.logical_and(current_data_inclusion_flag == 0, feature_set_opt == 1)

        if np.sum(test_spots_bool[test_spots])>0:
            x_test_mean = np.mean(current_xdata[test_spots_bool])
        else:
            current_data_inclusion_flag[test_spots] = 1
            test_spots_bool = np.logical_and(current_data_inclusion_flag == 1, feature_set_opt == 1)
            x_test_mean = np.mean(current_xdata[test_spots_bool])
        if np.sum(neg_spots_bool[neg_spots]) > 0:
            x_neg_mean = np.mean(current_xdata[neg_spots_bool])
        else:
            current_data_inclusion_flag[neg_spots] = 2
            test_spots_bool = np.logical_and(current_data_inclusion_flag == 2, feature_set_opt == 1)
            x_neg_mean = np.mean(current_xdata[test_spots_bool])

        x_pos_mean = np.mean(current_xdata[pos_spots_bool])
        if np.isnan(x_pos_mean):
            x_pos_mean = 0
        if n_time == 0:
            xmean_data = [x_test_mean, x_neg_mean, x_pos_mean]
        else:
            xmean_data += [x_test_mean, x_neg_mean, x_pos_mean]

    xdata_norm = np.zeros((time_touse))
    for n_time in range(time_touse):
        xdata_norm[n_time] = xmean_data[n_time * 3] - xmean_data[n_time * 3 + 1]

    xzero = xdata_norm[0]

    return xzero

def predict_concentration(data_checked, data_inclusion_flag, feature_set_opt, time_touse, xzero_mean, zero_subtraction_flag, accum_flag, time_offset):
    for n_time in range(time_touse):

        current_xdata = data_checked[n_time * n_spots: (n_time + 1) * n_spots]


        current_data_inclusion_flag = data_inclusion_flag[n_time * n_spots : (n_time + 1) * n_spots]

        test_spots_bool = np.logical_and(current_data_inclusion_flag == 1, feature_set_opt == 1)
        neg_spots_bool = np.logical_and(current_data_inclusion_flag == 2, feature_set_opt == 1)
        pos_spots_bool = np.logical_and(current_data_inclusion_flag == 0, feature_set_opt == 1)

        if np.sum(test_spots_bool[test_spots])>0:
            x_test_mean = np.mean(current_xdata[test_spots_bool])
        else:
            current_data_inclusion_flag[test_spots] = 1
            test_spots_bool = np.logical_and(current_data_inclusion_flag == 1, feature_set_opt == 1)
            x_test_mean = np.mean(current_xdata[test_spots_bool])
        if np.sum(neg_spots_bool[neg_spots]) > 0:
            x_neg_mean = np.mean(current_xdata[neg_spots_bool])
        else:
            current_data_inclusion_flag[neg_spots] = 2
            test_spots_bool = np.logical_and(current_data_inclusion_flag == 2, feature_set_opt == 1)
            x_neg_mean = np.mean(current_xdata[test_spots_bool])

        x_pos_mean = np.mean(current_xdata[pos_spots_bool])
        if np.isnan(x_pos_mean):
            x_pos_mean = 0
        if n_time == 0:
            xmean_data = [x_test_mean, x_neg_mean, x_pos_mean]
        else:
            xmean_data += [x_test_mean, x_neg_mean, x_pos_mean]

    xdata_norm = np.zeros((time_touse))
    xdata_norm_raw = np.zeros((1, time_touse))

    for n_time in range(time_touse):
        xdata_norm[n_time] = xmean_data[n_time * 3] - xmean_data[n_time * 3 + 1]
        xdata_norm_raw[0,n_time] = xmean_data[n_time * 3] - xmean_data[n_time * 3 + 1]

    if zero_subtraction_flag:
        xdata_norm = np.maximum(xdata_norm - xzero_mean, np.zeros(xdata_norm.shape))

    if time_touse == 1:
        sig = xdata_norm[0]
    else:
        sig = 0
        for n_time in range(time_touse - time_offset - 1):
            n_timef = n_time + time_offset
            x_test = (xdata_norm[n_timef] + xdata_norm[n_timef + 1]) * 0.5 / 2
            sig += x_test


    return sig, xdata_norm_raw, xdata_norm

def error_accumulation(data_inclusion_flag_sample, xsample_processed, xsample, accum_flag, n_times):
    thresh = np.floor(n_times)+1

    if accum_flag == 0:
        for spot in range(n_spots):
            times = np.linspace(0, n_times - 1, n_times)
            spot_array = np.linspace(0, n_times - 1, n_times) * n_spots + spot
            spot_array = np.array([np.int32(x) for x in spot_array])
            bool_array = np.logical_or(data_inclusion_flag_sample[spot_array]==-1, data_inclusion_flag_sample[spot_array]==-1)
            for spot_input in spot_array:
                if (xsample_processed[spot_input] == -2 and np.sum(bool_array)<thresh):
                    xsample_processed[spot_input] = xsample[spot_input]
                    #if spot in neg_spots:
                    #    data_inclusion_flag_sample[spot_input] = 2
                    #elif spot in test_spots:
                    #    data_inclusion_flag_sample[spot_input] = 1
    else:
        if accum_flag == 1:
            and_thresh = np.floor(n_times)+1
        elif accum_flag == 2:
            and_thresh = 1
        else:
            and_thresh = 10
        for spot in range(n_spots):
            spot_array = np.linspace(0, n_times - 1, n_times) * n_spots + spot
            spot_array = np.array([np.int32(x) for x in spot_array])
            if accum_flag == 1:
                bool_array = np.logical_or(data_inclusion_flag_sample[spot_array]==-1, data_inclusion_flag_sample[spot_array]==-1)
            elif accum_flag == 2:
                bool_array = np.logical_or(data_inclusion_flag_sample[spot_array]==-1, data_inclusion_flag_sample[spot_array]==-1)
            for spot_input in spot_array:
                if np.sum(bool_array) >= and_thresh:
                    xsample_processed[spot_input] = -1
                elif (np.sum(data_inclusion_flag_sample[spot_array]==-1) < thresh and xsample_processed[spot_input] == -2):
                    xsample_processed[spot_input] = xsample[spot_input]


    return xsample_processed, data_inclusion_flag_sample

def normalize_signals(xmean_data, time_touse, zero_cols, zero_subtraction_flag):
    n_samples = xmean_data.shape[1]

    xdata_norm = np.zeros((time_touse, n_samples))
    for n_time in range(time_touse):
        xdata_norm[n_time, :] = xmean_data[n_time*3, :]-xmean_data[n_time*3+1, :]

    xnorm_zero = xdata_norm[:, zero_cols]
    x_zero_min = np.min(xnorm_zero[0,:])

    if zero_subtraction_flag:
        xdata_norm = np.maximum(xdata_norm - x_zero_min, np.zeros(xdata_norm.shape))

    return xdata_norm, x_zero_min

def calculate_zero(xmean_data, time_touse, zero_cols):
    xmean_zero = xmean_data[:, zero_cols]
    #xmean_zero = np.transpose(np.array([0, 0.001, 0.002, 0.005]))

    n_zeros = xmean_zero.shape[1]
    x_zeros = np.zeros((n_zeros))
    for n_zero in range(n_zeros):
        if time_touse == 0:
            sig = xmean_zero[0, n_zero]
        else:
            sig = 0
            for n_time in range(time_touse-1):
                x_test = (xmean_zero[n_time, n_zero]+xmean_zero[n_time+1, n_zero])*0.5/2
                sig += x_test
        x_zeros[n_zero] = sig

    x_zero_final = np.mean(x_zeros)
    return x_zero_final

def calc_cv(x_data, y_labels):
    n_samples = x_data.shape[0]
    nbins = 0
    cv_sum = 0
    x_list = [x_data[0]]
    for n_sample in range(n_samples-1):
        n_sample = n_sample + 1
        if y_labels[n_sample] == y_labels[n_sample-1]:
            x_list.append(x_data[n_sample])
        else:
            if len(x_list)> 1 and np.mean(x_list)>0:
                cv_sum += np.std(x_list)/np.mean(x_list)
                nbins += 1
            x_list = []
    if nbins>0:
        cv_av = cv_sum/nbins
    else:
        cv_av = cv_sum

    return cv_av
def power_law(x, a, b):
    return a * np.power(x, b)

def power_law_shifted(x, a, b, c):
    return a * np.power(x, b) + c

def calc_loss(x_data_sig, y_labels):

    pars, cov = curve_fit(f=power_law, xdata=x_data_sig, ydata=y_labels, p0=[0, 0], bounds=(-np.inf, np.inf))
    loss = mean_squared_log_error(x_data_sig, power_law(y_labels, *pars))

    return loss

def print_control1(data_inclusion_flag, n_sigma):

    neg_excluded = []
    test_excluded = []
    for n_spot in range(n_spots):
        if data_inclusion_flag[n_spot] == -1 and n_spot in test_spots:
            test_excluded.append(n_spot)
        if data_inclusion_flag[n_spot] == -1 and n_spot in neg_spots:
            neg_excluded.append(n_spot)
    print('Spotwise exclusion using ' + str(n_sigma) + ' sigma rule is completed.')
    if len(neg_excluded) == 0 and len(test_excluded) == 0:
        print('No spots were excluded at this step')
    elif len(neg_excluded) > 0:
        out_str = 'Negative control spots # '
        for spot in neg_excluded:
            out_str += str(spot+1)
            out_str += ', '
        out_str += 'were excluded.'
        print(out_str)
    elif len(test_excluded) > 0:
        out_str = 'Test spots # '
        for spot in test_excluded:
            out_str += str(spot+1)
            out_str += ', '
        out_str += 'were excluded.'
        print(out_str)

    return


def generate_output_data(testing_data, testing_data_mean, testing_data_std, testing_cv, data_inclusion_flag, xdata_raw, time_touse):

    testing_data_mean = np.transpose(np.resize(testing_data_mean, (time_touse, 3)))
    testing_data_std = np.transpose(np.resize(testing_data_std, (time_touse, 3)))
    testing_cv = np.transpose(np.resize(testing_cv, (time_touse, 3)))

    raw_data = np.concatenate((testing_data, testing_data_mean, testing_data_std, testing_cv, xdata_raw), axis=0)

    times_array = np.zeros((1, time_touse))
    for ntime in range(time_touse):
        times_array[0, ntime] = 0.5 * ntime

    raw_data = np.concatenate((times_array, data_inclusion_flag,
                               np.empty((1, times_array.shape[1]), dtype=object), times_array,
                               raw_data), axis=0)

    col1 = np.zeros((raw_data.shape[0], 1))
    col1[0, 0] = None
    for nspot in range(n_spots):
        col1[nspot + 1, 0] = np.int32(nspot + 1)
        col1[nspot + 1 + n_spots + 2, 0] = np.int32(nspot + 1)

    col1[1 + n_spots, 0] = None
    col1[1 + n_spots + 1, 0] = None

    for n in range(col1.shape[0] - n_spots * 2 - 3):
        col1[n + 2 * n_spots + 3, 0] = None

    raw_data = np.concatenate((col1, raw_data), axis=1)

    raw_data = raw_data.tolist()

    for row in range(n_spots):
        for col in range(time_touse):
            if raw_data[row+20][col+1] < 0:
                raw_data[row + 20][col + 1] = ' '

    raw_data[0][0] = 'Time'
    raw_data[1 + n_spots + 1][0] = 'Time'
    for n in range(time_touse + 1):
        raw_data[1 + n_spots][n] = ' '

    raw_data[2 * n_spots + 3][0] = 'Mean(test)'
    raw_data[2 * n_spots + 4][0] = 'Mean(neg)'
    raw_data[2 * n_spots + 5][0] = 'Mean(pos)'

    raw_data[2 * n_spots + 6][0] = 'STD(test)'
    raw_data[2 * n_spots + 7][0] = 'STD(neg)'
    raw_data[2 * n_spots + 8][0] = 'STD(pos)'

    raw_data[2 * n_spots + 9][0] = 'CV(test)'
    raw_data[2 * n_spots + 10][0] = 'CV(neg)'
    raw_data[2 * n_spots + 11][0] = 'CV(pos)'

    raw_data[2 * n_spots + 12][0] = 'test-neg'


    return raw_data

def generate_output_features(r2_sequence, cv_sequence, loss_sequence, spot_exclusion_sequence, feature_set_opt):

    metrics_array = np.concatenate(([r2_sequence], [cv_sequence], [loss_sequence], [spot_exclusion_sequence]), axis = 0)
    iteration_array = np.linspace(1, len(r2_sequence), len(r2_sequence))
    metrics_array = np.concatenate(([iteration_array], metrics_array), axis = 0)
    zero_array = np.zeros((5,1))
    metrics_array = np.concatenate((zero_array, metrics_array), axis=1)
    metrics_array = metrics_array.tolist()
    metrics_array[0][0] = 'Iteration'
    metrics_array[1][0] = 'Pearson correlation'
    metrics_array[2][0] = 'CV'
    metrics_array[3][0] = 'Custom loss (=CV)'
    metrics_array[4][0] = 'Exclusion sequence'
    return metrics_array

n_spots = 17

pos_spots = [0, 15]
neg_spots = [3, 4, 11, 12, ]
test_spots = [1, 2, 5, 6, 7, 8, 9, 10, 13, 14]

#test_multiplier = 2.26215716279821/np.sqrt(10)
neg_multiplier = 2.228/np.sqrt(11)
test_multiplier = 3.182/np.sqrt(4)
#test_multiplier = 1
#neg_multiplier = 1
def save_quality_control(path, quality_control_folder, y_labels, x_data, xdata_mean, flag_array, data_inclusion_flag):
    np.savetxt(path + '\\' + quality_control_folder + '\\y_labels.csv', y_labels, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + quality_control_folder + '\\x_data.csv', x_data, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + quality_control_folder + '\\xdata_mean.csv', xdata_mean, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + quality_control_folder + '\\sensor_exclusion_flags.csv', flag_array, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + quality_control_folder + '\\spot_exclusion_flags.csv', data_inclusion_flag, '%1.10f', delimiter=",")
    return


def save_optimal_calibration(path, calibration_folder, r2_sequence, cv_sequence, loss_sequence, spot_exclusion_sequence,
                             xdata_opt_array, metric_optimal, xdata_optimal, power_coeffs):

    np.savetxt(path + '\\' + calibration_folder + '\\r2_sequence.csv', r2_sequence, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + calibration_folder + '\\cv_sequence.csv', cv_sequence, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + calibration_folder + '\\loss_sequence.csv', loss_sequence, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + calibration_folder + '\\spot_exclusion_sequence.csv', spot_exclusion_sequence, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + calibration_folder + '\\xdata_opt_array.csv', xdata_opt_array, '%1.10f', delimiter=",")

    np.savetxt(path + '\\' + calibration_folder + '\\metric_optimal.csv', metric_optimal, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + calibration_folder + '\\xdata_optimal.csv', xdata_optimal, '%1.10f', delimiter=",")
    np.savetxt(path + '\\' + calibration_folder + '\\power_coeffs.csv', power_coeffs, '%1.10f', delimiter=",")
    return