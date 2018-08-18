from Dataset_Handler.CS_Iterator import CS_Iterator
import os

server_addr = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
cs_loc = server_addr + "/home/mumu01/models/deeplab/datasets/cityscapes"

cs_iter = CS_Iterator(cs_loc, "val", debug=True)

idx = 0
for _, labelId_image, _, leftImg_image, _, disparity_image in cs_iter:
    idx += 1
