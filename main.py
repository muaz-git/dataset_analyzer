import matplotlib

matplotlib.use('Agg')

from Dataset_Handler.CS_Iterator import CS_Iterator
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from labels import *


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
    cs_iter = CS_Iterator(cs_loc, data_split, debug=debug)
    class_frequency = get_class_freq_holder()
    elems_counter = 0
    for _, labelId_image, _, leftImg_image, _, disparity_image in cs_iter:
        onehot_label_matrix = onehot_initialization_v2(labelId_image)
        class_frequency = np.add(class_frequency, onehot_label_matrix)

        elems_counter += 1

    # class_frequency = np.divide(class_frequency, idx)
    return class_frequency, elems_counter


server_addr = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
cs_loc = server_addr + "/home/mumu01/models/deeplab/datasets/cityscapes"

# data_analysis = np.load("data.npy")
# print(np.shape(data_analysis["train"]))
# exit()
data_analysis = {}
for split in ["train", "val", "test"]:
    print("Going for ", split)
    data_analysis[split] = {}
    class_frequency, elems_counter = get_class_frequency(split)

    data_analysis[split]["class_frequency"] = class_frequency
    data_analysis[split]["total_images"] = elems_counter

    print("shape is ", np.shape(class_frequency))
    print("min is ", np.amin(class_frequency))
    print("max is ", np.amax(class_frequency))
    print("total images considered ", elems_counter)

print()
print(type(data_analysis))
for key in data_analysis:
    print("\t", type(data_analysis[key]))
    print("\t", key)
    for kk in data_analysis[key]:
        print("\t\t", type(data_analysis[key][kk]))
        print("\t\t", kk)
# print("Saving")
# np.save("data.npy", data_analysis)
# print("Done")
exit()
upper_bound = 32257
lower_bound = 1
bin_gap = 10
bins = np.arange(lower_bound, np.ceil(upper_bound / bin_gap) * bin_gap + 1, bin_gap)

idx = 0
for _, labelId_image, _, leftImg_image, _, disparity_image in cs_iter:
    onehot_label_matrix = onehot_initialization_v2(labelId_image)
    class_frequency = np.add(class_frequency, onehot_label_matrix)

    # print(np.shape(onehot_label_matrix))
    # exit()
    # labelId = getLabelId()
    # filtered_img = filter_labelId(labelId, labelId_image, leftImg_image)
    #
    # cv2.imshow("filtered img ", filtered_img)
    # cv2.waitKey()
    # valid_elems_arr = valid_disparity_elems(disparity_image)
    # print(valid_elems_arr[1:])
    # hist, bin_edges = np.histogram(valid_elems_arr, bins=bins)

    # hist = hist / hist.sum()
    # print(np.amax(hist))
    # plt.figure(figsize=(15, 7.5))
    # ax = plt.subplot(111)
    #
    # plt.hist(valid_elems_arr, bins=bins, density=True)
    # plt.title("Histogram with bins")
    # # ax.set_ylim([0, 500])
    #
    # plt.savefig("tmp2.png")
    # exit()
    idx += 1

# carslice = class_frequency[:, :, 7]
#
# cv2.imshow("car", carslice)
# cv2.waitKey()
