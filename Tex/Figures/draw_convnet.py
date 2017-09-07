"""
Copyright (c) 2016, Gavin Weiguang Ding
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


NumConvMaxbig = 10
NumConvMaxsmall = 24
NumConvMaxtiny = 50
SmallOnFrom = 3
TinyOnFrom = 4
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Black = 0.


def add_layer(patches, colors, sizex=24, sizey=24, num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, sizex])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, sizex, sizey))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_mapping(patches, colors, start_ratio, patch_sizex, patch_sizey, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * size_list[ind_bgn],
                    -start_ratio[1] * size_list[ind_bgn]])

    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) \
        * np.array(loc_diff_list[ind_bgn + 1]) \
        + np.array([(start_ratio[0] + .5 * patch_sizex / size_list[ind_bgn]) *
                    size_list[ind_bgn + 1],
                    -(start_ratio[1] - .5 * patch_sizex / size_list[ind_bgn]) *
                    size_list[ind_bgn + 1]])

    patches.append(Rectangle(start_loc, patch_sizex, patch_sizey))
    colors.append(Dark)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0] + patch_sizex, end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1] + patch_sizey, end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0] + patch_sizex, end_loc[0]],
                          [start_loc[1] + patch_sizey, end_loc[1]]))
    colors.append(Black)


def label(xy, text, xy_off=[0, 17]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8)


if __name__ == '__main__':

    fc_unit_size = 2
    layer_width = 34

    patches = []
    colors = []

    fig, ax = plt.subplots()


    ############################
    # conv layers
    sizex_list = [30, 14, 8, 2, 1]
    sizey_list = [45, 14, 8, 2, 1]
    num_list = [8, 32, 64, 64, 256]
    x_diff_list = [0, 54, layer_width, layer_width, layer_width]
    text_list = ['Convolutional\ninputs'] + ['Feature\nmaps'] * (len(sizex_list) - 1)
    loc_diff_list = [[3, -3]] * SmallOnFrom + [[2, -2]] * (TinyOnFrom-SmallOnFrom) + [[1,-1]] * (len(sizex_list)-TinyOnFrom)

    bigs = num_list[:SmallOnFrom]
    smalls = num_list[SmallOnFrom:TinyOnFrom]
    tinys = num_list[TinyOnFrom:]
    num_show_list1 = list(map(min, bigs, [NumConvMaxbig] * len(bigs)))
    num_show_list2 = list(map(min, smalls, [NumConvMaxsmall] * len(smalls)))
    num_show_list3 = list(map(min, smalls, [NumConvMaxtiny] * len(tinys)))
    num_show_list = num_show_list1 + num_show_list2 +  num_show_list3
    
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    
    for ind in range(len(sizex_list)):
        add_layer(patches, colors, sizex=sizex_list[ind], sizey=sizey_list[ind],
                  num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
            num_list[ind], sizex_list[ind], sizey_list[ind]))


    ############################
    # in between layers
    start_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0, 1]]
    patch_sizex_list = [4, 4, 3, 4]
    patch_sizey_list = [6, 4, 3, 4]
    stridex_list = [2, 2, 2, 2]
    stridey_list = [3, 2, 2, 2  ]
    ind_bgn_list = range(len(patch_sizex_list))
    text_list = ['Convolution', 'Convolution', 'Convolution', 'Convolution']

    for ind in range(len(patch_sizex_list)):
        add_mapping(patches, colors, start_ratio_list[ind],
                    patch_sizex_list[ind], patch_sizey_list[ind], ind,
                    top_left_list, loc_diff_list, num_show_list, sizex_list)
        label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel\nstride ({},{})'.format(
            patch_sizex_list[ind], patch_sizey_list[ind], stridex_list[ind], stridey_list[ind]), xy_off=[26, -70])


    ############################
    # fully connected layers
    size_list = [fc_unit_size]
    num_list = [256]
    num_show_list = [20]
    x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden\nunits'] #* (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]))

    text_list = ['Flatten\n', 'Fully\nconnected', 'Fully\nconnected']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -65])



    ############################
    # fully connected layers NUMBER TWO
    size_list = [fc_unit_size, fc_unit_size]
    num_list = [128, 64]
    num_show_list = [18, 10]
    x_diff_list = [4*layer_width+25, layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))-94]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Flat inputs', 'Hidden units'] #* (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]), xy_off=[0,5])

    text_list = ['', 'Fully\nconnected', 'Fully\nconnected']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -55])

    ############################
    # fully connected layers CONCAT
    size_list = [fc_unit_size, fc_unit_size]
    num_list = [384, 256]
    num_show_list = [22, 20]
    x_diff_list = [7.1*layer_width,layer_width,layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))-40]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden units', 'Hidden units'] #* (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]), xy_off=[0,5])

    text_list = ['Concat', 'Fully\nconnected', 'Split']

    
    label(top_left_list[0], text_list[0], xy_off=[0,-35])
    label(top_left_list[1], text_list[1], xy_off=[-10, -55])
        
    ############################
    # fully connected layers SPLIT 1
    size_list = [fc_unit_size, fc_unit_size]
    num_list = [128, 21]
    num_show_list = [11, 7]
    x_diff_list = [9.4*layer_width, layer_width*.8]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))-20]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Advantage', ''] #* (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]), xy_off=[0,5])

    text_list = ['Fully\nConnected', '', '']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[10, -35])

        
    ############################
    # fully connected layers SPLIT 2
    size_list = [fc_unit_size, fc_unit_size]
    num_list = [128, 1]
    num_show_list = [11, 1]
    x_diff_list = [9.4*layer_width, layer_width*.8]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))-70]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Value', ''] #* (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]), xy_off=[0,5])

    text_list = ['Fully\nConnected', '', '']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[10, -35])
       
        
        
    ############################
    # fully connected layers FINAL CONCAT
    size_list = [fc_unit_size]
    num_list = [21]
    num_show_list = [7]
    x_diff_list = [11*layer_width,layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))-40]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Output'] #* (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]), xy_off=[0,5])

    text_list = ['Matmul']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -19])           




    ############################
    colors += [0, 1]
    collection = PatchCollection(patches, cmap=plt.cm.gray)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig.set_size_inches(8, 2.5)

    fig_dir = './'
    fig_ext = '.png'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)
