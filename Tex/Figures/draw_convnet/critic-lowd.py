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
from copy import deepcopy


NumConvMaxbig = 10
NumConvMaxsmall = 13
NumConvMaxtiny = 10
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

    top = loc_start; top[1] -= 0.5*loc_diff[1]; top[0] += loc_diff[0]
    bottom = loc_start + (num-1) * loc_diff
    return top, bottom 

def add_fc_mapping(patches, colors, loc_start1, loc_end1, loc_start2, loc_end2, loc_size=[2, 0], loc_diff=[2, -2], probab=0.3):

    loc_start2n = deepcopy(loc_start2); loc_start2n[0] -= loc_size[0]
    loc_end2n = deepcopy(loc_end2); loc_end2n[0] -= loc_size[0]
    
    l1 = int(round((loc_end1[0]-loc_start1[0])/(loc_diff[0])))
    l2 = int(round((loc_end2[0]-loc_start2[0])/(loc_diff[0])))
    
    if l1 < 3 or l2 < 3:
        probab = 1
    elif l1 < 6 or l2 < 6:
        probab = 0.4

    lstart = deepcopy(loc_start1)
    lend = deepcopy(loc_start2n)

    for j in range(l2+1):
        lstart = deepcopy(loc_start1)
        for i in range(l1+1):
            if np.random.random() < probab: 
                patches.append(Line2D([lstart[0], lend[0]],[lstart[1], lend[1]]))
            lstart += loc_diff
        lend += loc_diff
        
        
def add_XToY_mapping(patches, colors, loc_start1, loc_end1, loc_start2, loc_end2, loc_size=[2, 0], loc_diff=[2, -2], x=1, y=1):

    loc_start2n = deepcopy(loc_start2); loc_start2n[0] -= loc_size[0]
    loc_end2n = deepcopy(loc_end2); loc_end2n[0] -= loc_size[0]
    
    l1 = int(round((loc_end1[0]-loc_start1[0])/(loc_diff[0])))
    l2 = int(round((loc_end2[0]-loc_start2[0])/(loc_diff[0])))
 
    lstart = deepcopy(loc_start1)
    lend = deepcopy(loc_start2n)
    
    if x == y == 1:
        for j in range(min(l1,l2)+1):
            patches.append(Line2D([lstart[0], lend[0]],[lstart[1], lend[1]]))
            lstart += loc_diff
            lend += loc_diff           
    else:
        for j in range(max(l1,l2)+1):
            patches.append(Line2D([lstart[0], lend[0]],[lstart[1], lend[1]]))
            if j % 2 == 0: 
                lstart += loc_diff
            lend += loc_diff       
        

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


def label(xy, text, xy_off=[0, 5]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=9)


if __name__ == '__main__':

    fc_unit_size = 2
    layer_width = 34

    patches = []
    colors = []
    lines = []

    fig, ax = plt.subplots()


    ############################
    size_list = [fc_unit_size, fc_unit_size]
    num_list = [154, 400]
    num_show_list = [14, 24]
    x_diff_list = [layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Inputs', 'Hidden\nunits'] #* (len(size_list) - 1) + ['Outputs']

    oldtop, oldbottom = None, None
    for ind in range(len(size_list)):
        top, bottom = add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]))
        
        if oldtop != None:
            add_fc_mapping(lines, colors, oldtop, oldbottom, top, bottom, probab=0.2)
                
        oldtop, oldbottom = top, bottom

    text_list = ['', 'Fully\nconnected']

    label(top_left_list[1], text_list[1], xy_off=[-10, -38])
    

    ############################
    # fully connected layers NUMBER TWO
    size_list = [fc_unit_size]
    num_list = [3]
    num_show_list = [3]
    x_diff_list = [2*layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))-61]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Action inputs', 'Hidden units'] #* (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        tmptop, tmpbottom = add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]), xy_off=[0,5])

    text_list = ['Concat', '']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[40, -10])
        

    ############################
    size_list = [fc_unit_size, fc_unit_size, fc_unit_size, fc_unit_size]
    num_list = [403, 300, 20, 1]
    num_show_list = [27, 20, 5, 1]
    x_diff_list = [layer_width*3.4, layer_width, layer_width, layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))-10]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden\nunits', 'Hidden\nunits', 'Hidden\nunits', 'Q'] #* (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        top, bottom = add_layer(patches, colors, sizex=size_list[ind], sizey=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]))
        
        
         
        if oldtop != None:
            if ind == 0:
                add_XToY_mapping(lines, colors, oldtop, oldbottom, top, bottom)
                add_XToY_mapping(lines, colors, tmptop, tmpbottom, bottom-[4,-4], bottom, x=1, y=1)
            else:
                add_fc_mapping(lines, colors, oldtop, oldbottom, top, bottom, probab=0.2)
                
        oldtop, oldbottom = top, bottom

    text_list = ['', 'Fully\nconnected', 'Fully\nconnected', 'Fully\nconnected', 'Scale']

    
    label(top_left_list[1], text_list[1], xy_off=[-10, -68])
    label(top_left_list[2], text_list[2], xy_off=[-10, -60])
    label(top_left_list[3], text_list[3], xy_off=[-15, -20])
    



    ############################
    colors += [0, 1]
    collection = PatchCollection(patches, cmap=plt.cm.gray)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    linescol = PatchCollection(lines, edgecolors="0.86")
    ax.add_collection(linescol)
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig.set_size_inches(8, 2.5)

    fig_dir = './'
    fig_ext = '.png'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)

