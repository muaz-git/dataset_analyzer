import matplotlib

matplotlib.use('Agg')

from Dataset_Handler.CS_Iterator import CS_Iterator
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from labels import *
import sys
from scipy.stats import kde


def valid_disparity_elems(disparity):
    valid_elems = np.greater(disparity, 1)
    return disparity[valid_elems]


def invalid_disparity_elems(disparity):
    invalid_elems = np.less_equal(disparity, 0)
    return disparity[invalid_elems]


def getLabelId():
    return 26


def filter_labelId(labelId, labelId_image, to_filter):
    filtered_img = np.zeros_like(to_filter)
    filtered_img[np.equal(labelId_image, labelId)] = to_filter[np.equal(labelId_image, labelId)]
    return filtered_img


maxLabelID = 33
ncols = maxLabelID + 1


def onehot_initialization_v2(a):
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def get_class_freq_holder():
    return np.zeros((1024, 2048, ncols), dtype=np.uint16)


def get_class_frequency(data_split, debug=True):
    def replace_value(labelId_image):
        np.place(labelId_image, labelId_image == -1, [34])

    def counter(labelId_image):
        count = np.where(labelId_image < 0)
        if len(count[0]) > 0:
            print("counted ", len(count[0]), " elems.")

    cs_iter = CS_Iterator(cs_loc, data_split, debug=debug)
    class_frequency = get_class_freq_holder()
    elems_counter = 0
    for _, labelId_image, _, leftImg_image, _, disparity_image in cs_iter:
        print("\rImages Processed: {}".format(elems_counter + 1), end=' ')
        sys.stdout.flush()
        counter(labelId_image)
        replace_value(labelId_image)
        onehot_label_matrix = onehot_initialization_v2(labelId_image)
        class_frequency = np.add(class_frequency, onehot_label_matrix)

        elems_counter += 1

    # class_frequency = np.divide(class_frequency, idx)
    return class_frequency, np.array(elems_counter)


def calculating_frequency(debug=True):
    data_analysis = {}
    for split in ["train", "val"]:
        print("Going for ", split)
        data_analysis[split] = {}
        class_frequency, elems_counter = get_class_frequency(split, debug=debug)

        data_analysis[split]["class_frequency"] = class_frequency
        data_analysis[split]["total_images"] = elems_counter

        print("\nThere are ", elems_counter, " images in the split: ", split)

    print("Not Saving")
    # np.save(frequency_filename, data_analysis)
    print("Done")


def load_class_frequency():
    return np.load(frequency_filename)[()]


def plot_density(split_name, label_data, analysis_mat):
    label_id = label_data.id
    plt.clf()
    sl = np.array(analysis_mat["class_frequency"][:, :, label_id], dtype=np.float64)
    total = np.array(analysis_mat["total_images"], dtype=np.float64)
    class_slice = np.divide(sl, total)
    class_slice = np.divide(class_slice, ncols)

    # class_slice = np.divide(analysis_mat["class_frequency"][:, :, label_id], analysis_mat["total_images"])
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    label = label_data.name

    ax.set_title('Density for class \"' + label + "\" in \"" + split_name + "\" split.")

    im = plt.imshow(class_slice, interpolation='nearest', origin='upper', cmap=plt.cm.Blues)

    plt.colorbar()
    plt.axis('off')

    plot_folder = "./data/plots/class_density/"
    plot_path = plot_folder + label + "_" + split_name + ".png"

    plt.savefig(plot_path)


def plot_classes_per_split(data_analysis):
    for split in ["train", "val"]:
        split_analysis = data_analysis[split]
        for l in labels:
            if not l.ignoreInEval:
                plot_density(split, l, split_analysis)


def get_categorical_avg(data_analysis):
    class_avg_dict = {}
    for split in ["train", "val"]:
        split_analysis = data_analysis[split]

        class_avg_dict[split] = {}

        overall_avg = 0.0
        for label_data in labels:
            label_id = label_data.id
            if label_id >= 0:
                if label_data.category not in class_avg_dict[split]:
                    class_avg_dict[split][label_data.category] = {}

                label = label_data.name

                sl = np.array(split_analysis["class_frequency"][:, :, label_id], dtype=np.float64)

                total_imgs = np.array(split_analysis["total_images"], dtype=np.float64)
                # print("Sum : ",np.sum(sl))
                # print("total_imgs : ",total_imgs)
                class_slice = np.divide(sl, total_imgs)

                class_avg = np.mean(class_slice, dtype=np.float64)
                # print(class_avg)
                # exit()

                class_avg_dict[split][label_data.category][label] = class_avg
                # class_avg_dict[split][label_data.category][label] = np.sum(sl)
                # print("Class : ", label, " avg = ", class_avg)
                overall_avg += class_avg

    return class_avg_dict


def plot_categorical_avg(split_class_avg, split_name):
    plt.clf()
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)

    def autolabel(rects, xtick_labels):
        """
        Attach a text label above each bar displaying its height
        """
        for rect, l in zip(rects, xtick_labels):
            height = rect.get_height()

            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.01,
                    '%s' % l,
                    ha='center', va='bottom', rotation="vertical")

    margin = 6.0
    width = 5
    c = 0

    for c_idx, category in enumerate(split_class_avg):
        xtick_labels = []
        N = len(split_class_avg[category])

        ind = np.arange(N)

        ind = [c + 0.5 * width + idx * width for idx in ind]

        c = ind[-1] + margin
        avg = [split_class_avg[category][k] for k in split_class_avg[category]]

        for k in split_class_avg[category]:
            xtick_labels.append(k)

        colors = [list(l.color) for k in split_class_avg[category] for l in labels if l.name == k]
        colors_new = []
        for color in colors:
            color = np.array(color, dtype=np.float64)
            new_c = np.divide(color, 255)
            colors_new.append(list(new_c))

        rects1 = ax.bar(ind, avg, width, color=colors_new, label="tempo")
        autolabel(rects1, xtick_labels)

    ax.set_ylim([0, 0.5])

    plot_folder = "./data/plots/class_average/"
    plot_path = plot_folder + split_name + "_class_avg.png"
    plt.savefig(plot_path)


server_addr = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
# server_addr = ""
cs_loc = server_addr + "/home/mumu01/models/deeplab/datasets/cityscapes"

frequency_filename = "./data/class_frequency.npy"

# calculating_frequency(False)
# exit()
data_analysis = load_class_frequency()

# plot_classes_per_split(data_analysis)

class_avg_dict = get_categorical_avg(data_analysis)

for split in ['train', 'val']:
    split_class_avg = class_avg_dict[split]
    plot_categorical_avg(split_class_avg, split)

exit()

ind = np.arange(len(train_void_cat))
width = 0.35
avg = []
for cl in train_void_cat:
    avg.append(train_void_cat[cl])

exit()
# upper_bound = 32257
# lower_bound = 1
# bin_gap = 10
# bins = np.arange(lower_bound, np.ceil(upper_bound / bin_gap) * bin_gap + 1, bin_gap)
#
# idx = 0
# for _, labelId_image, _, leftImg_image, _, disparity_image in cs_iter:
#     onehot_label_matrix = onehot_initialization_v2(labelId_image)
#     class_frequency = np.add(class_frequency, onehot_label_matrix)
#
#     # print(np.shape(onehot_label_matrix))
#     # exit()
#     # labelId = getLabelId()
#     # filtered_img = filter_labelId(labelId, labelId_image, leftImg_image)
#     #
#     # cv2.imshow("filtered img ", filtered_img)
#     # cv2.waitKey()
#     # valid_elems_arr = valid_disparity_elems(disparity_image)
#     # print(valid_elems_arr[1:])
#     # hist, bin_edges = np.histogram(valid_elems_arr, bins=bins)
#
#     # hist = hist / hist.sum()
#     # print(np.amax(hist))
#     # plt.figure(figsize=(15, 7.5))
#     # ax = plt.subplot(111)
#     #
#     # plt.hist(valid_elems_arr, bins=bins, density=True)
#     # plt.title("Histogram with bins")
#     # # ax.set_ylim([0, 500])
#     #
#     # plt.savefig("tmp2.png")
#     # exit()
#     idx += 1
#
# # carslice = class_frequency[:, :, 7]
# #
# # cv2.imshow("car", carslice)
# # cv2.waitKey()
# #
