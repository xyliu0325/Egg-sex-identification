import xlrd
import xlwt
from xlutils.copy import copy
import pandas as pd
import cv2 as cv
import numpy as np
import os
import openpyxl



def dataProcess(data1,data2,res1,res2):
    df1 = pd.read_excel(data1, usecols=['Blue','Green','Red'])
    df2 = pd.read_excel(data2, usecols=['Blue','Green','Red'])

    df_res1 = pd.DataFrame(np.zeros(df1.shape), columns=df1.columns, index=df1.index)
    df_res1['Blue'] = abs(df1['Blue'] - df2['Blue'])
    df_res1['Green'] = abs(df1['Green'] - df2['Green'])
    df_res1['Red'] = abs(df1['Red'] - df2['Red'])

    df_res2 = pd.DataFrame(np.zeros(df1.shape), columns=df1.columns, index=df1.index)
    df_res2['Blue'] = abs((abs(df1['Blue'] - df2['Blue']) - 4) * 17)
    df_res2['Green'] = abs((abs(df1['Green'] - df2['Green']) - 4) * 17)
    df_res2['Red'] = abs((abs(df1['Red'] - df2['Red']) - 4) * 17)


    df_res1.to_excel(res1)
    df_res2.to_excel(res2)

    print("\nPreprocessing completed.")


def mapProcess(datapath,savepath):
    file_name = datapath
    df = pd.read_excel(file_name, header=None)
    bgr_list = []

    for i in range(1, 3997):
        exper = df.values[i, 1:4]
        b, g, r = round(min(exper[0], 255)), round(min(exper[1], 255)), round(min(exper[2], 255))
        bgr_list.append([b, g, r])


    blue_col, green_col, red_col = 1, 2, 3
    df = pd.read_excel(file_name, header=None)
    total_row = 3997

    H, W, C = 3000, 4000, 3
    R = 200
    row_num = 3
    col_num = 6

    row_pos = list(round(H/4*(row+1)) for row in range(row_num))
    col_pos = list(round(W/7*(col+1)) for col in range(col_num))
    pos_list = []
    for j in range(row_num):
        for k in range(col_num):
            pos_list.append((col_pos[k], row_pos[j]))

    num_flag = 0
    black_img = np.zeros((H, W, C), np.uint8)
    name_num = 1

    for i in range(1, total_row):
        if i%18 == 1:
            print("\n*********** Image %d ***********" % (name_num))
        if num_flag != row_num * col_num:
            b, g, r = df.values[i, blue_col:red_col+1]
            b, g, r = round(b), round(g), round(r)
            print(i,'--- BGR values: ',b,g,r)
            cv.circle(black_img, pos_list[num_flag], R, (b, g, r), -1, cv.LINE_AA)
            num_flag += 1

        if num_flag == row_num * col_num:
            cv.imwrite(os.path.join(savepath, str(name_num)+'.png'), black_img)


            name_num += 1
            num_flag = 0
            black_img = np.zeros((H, W, C), np.uint8)

    print("\nImage generation completed.")

if __name__ == '__main__':

    data1 = r"C:\Users\41673\Desktop\0.xlsx"
    data2 = r"C:\Users\41673\Desktop\4.xlsx"
    res1 = r"C:\Users\41673\Desktop\4\res1.xlsx"
    res2 = r"C:\Users\41673\Desktop\4\res2.xlsx"


    dataProcess(data1,data2,res1,res2)

    datapath2 = res2
    savepath2 = r"C:\Users\41673\Desktop\4"
    mapProcess(datapath2,savepath2)