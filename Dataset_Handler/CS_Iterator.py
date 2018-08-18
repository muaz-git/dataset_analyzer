import os
import glob
import cv2


class CS_Iterator(object):
    def __init__(self, cs_base_path, split, debug=False):
        if split not in ["train", "val", "test"]:
            raise ValueError("Split : ", split, " not valid.")
        self.cs_base_path = cs_base_path
        self.split = split

        self.labelId_filenames = self.__get_labelId_filepaths()
        self.disparity_filenames = self.__get_disparity_filepaths()
        self.leftImg_filenames = self.__get_img_filepaths()

        if not ((len(self.labelId_filenames) == len(self.disparity_filenames)) and (
                len(self.disparity_filenames) == len(self.leftImg_filenames))):
            raise ValueError("Length of files do not match.")
        self.iter_index = 0
        self.debug = debug

    def __iter__(self):
        return self

    def get_single_cv_image(self, image_file):
        """
        Returns the OpenCV image of the given filepath
        """
        image_file_path = image_file

        im = cv2.imread(image_file_path, -1)
        return im

    def __next__(self):

        if self.iter_index < len(self.labelId_filenames):
            if self.debug and self.iter_index > 5:
                raise StopIteration()
            labelId_file_name = self.labelId_filenames[self.iter_index]
            disparity_file_name = self.disparity_filenames[self.iter_index]
            leftImg_file_name = self.leftImg_filenames[self.iter_index]

            labelId_image = self.get_single_cv_image(labelId_file_name)
            disparity_image = self.get_single_cv_image(disparity_file_name)
            leftImg_image = self.get_single_cv_image(leftImg_file_name)

            self.iter_index += 1

            return labelId_file_name, labelId_image, leftImg_file_name, leftImg_image, disparity_file_name, disparity_image

        else:
            raise StopIteration()

    def __get_filepaths_from_search(self, search):
        ImgList = glob.glob(search)
        ImgList.sort()
        return ImgList

    def __get_labelId_filepaths(self):
        labelIdSearch = os.path.join(self.cs_base_path, "gtFine", self.split, "*", "*_gtFine_labelIds.png")
        return self.__get_filepaths_from_search(labelIdSearch)

    def __get_disparity_filepaths(self):
        disparitySearch = os.path.join(self.cs_base_path, "disparity", self.split, "*", "*_disparity.png")
        return self.__get_filepaths_from_search(disparitySearch)

    def __get_img_filepaths(self):
        leftImgSearch = os.path.join(self.cs_base_path, "leftImg8bit", self.split, "*", "*_leftImg8bit.png")
        return self.__get_filepaths_from_search(leftImgSearch)
