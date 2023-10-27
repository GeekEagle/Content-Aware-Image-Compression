import cv2
import numpy as np
import time
import concurrent.futures
import math
import threading
import os


fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def matrix_multiply(matrix):
    thread_id = threading.get_ident()
    # print("Task running in thread {}".format(thread_id))
    rows,columns = matrix.shape
    matrix = np.pad(matrix,pad_width=1,mode='constant',constant_values=0)
    Gx = Gy = matrix.copy()
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            Gx[i][j] = np.sum(fx*matrix[i-1:i+2,j-1:j+2])
            Gy[i][j] = np.sum(fy*matrix[i-1:i+2,j-1:j+2])
    Gx = Gx[1:-1,1:-1]
    Gy = Gy[1:-1,1:-1]
    return Gx,Gy

def init_energy_function(gray_scale):
    rows, columns = gray_scale.shape
    print("init:",gray_scale.shape)
    mat = [[] for _ in range(8)]
    # gray_scale = np.pad(gray_scale, pad_width=1, mode='constant', constant_values=0)
    row_sig = math.floor(rows/2)
    col_sig1 = math.floor(columns/4)
    col_sig2 = math.floor(columns/2)
    col_sig3 = math.floor(3*columns/4)
    mat[0] = list(gray_scale[0:row_sig,0:col_sig1])
    mat[1] = list(gray_scale[0:row_sig,col_sig1:col_sig2])
    mat[2] = list(gray_scale[0:row_sig, col_sig2:col_sig3])
    mat[3] = list(gray_scale[0:row_sig, col_sig3:columns])
    mat[4] = list(gray_scale[row_sig:rows, 0:col_sig1])
    mat[5] = list(gray_scale[row_sig:rows, col_sig1:col_sig2])
    mat[6] = list(gray_scale[row_sig:rows, col_sig2:col_sig3])
    mat[7] = list(gray_scale[row_sig:rows, col_sig3:columns])
    exec = [np.array(mat[i]) for i in range(8)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = []
        for i in range(8):
            result = executor.submit(matrix_multiply, exec[i])
            results.append(result)
        concurrent.futures.wait(results)

    res = [result.result() for result in results]
    Gx_up = np.array(res[0][0])
    Gx_down = np.array(res[4][0])
    Gy_up = np.array(res[0][1])
    Gy_down = np.array(res[4][1])
    for i in range(1,4):
        gx_res_up = np.array(res[i][0])
        gx_res_down = np.array(res[i+4][0])
        tempx_up = np.hstack((Gx_up,gx_res_up))
        tempx_down = np.hstack((Gx_down,gx_res_down))
        Gx_up = tempx_up
        Gx_down = tempx_down
        gy_res_up = np.array(res[i][1])
        gy_res_down = np.array(res[i + 4][1])
        tempy_up = np.hstack((Gy_up, gy_res_up))
        tempy_down = np.hstack((Gy_down, gy_res_down))
        Gy_up = tempy_up
        Gy_down = tempy_down

    Gx = np.vstack((Gx_up,Gx_down))
    Gy = np.vstack((Gy_up, Gy_down))

    Energy = np.sqrt(Gx ** 2 + Gy ** 2)
    return Energy

def update_energy_function(gray_scale,seam,energy,mode):
    rows,columns = energy.shape
    new_energy = energy.copy()
    gx = np.zeros(6)
    gy = np.zeros(6)
    e_fun = np.zeros(6)

    gray_scale = np.pad(gray_scale, pad_width=1, mode='constant', constant_values=0)
    if mode == 'col':
        for i in range(1,rows+1):
            if seam[i-1]<columns-1:
                j = seam[i-1]+1
            else: j = seam[i-1]
            if i == 1 and j == 1:
                gx[0] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[0] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                e_fun[0] = np.sqrt(gx[0] ** 2 + gy[0] ** 2)
                new_energy[i-1,j-1] = e_fun[0]
                new_energy[i-1,j:columns-1] = energy[i-1,j+1:columns]
            elif i == 1 and j > 1:
                gx[0] = np.sum(fx * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gx[1] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[0] = np.sum(fy * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gy[1] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                for k in range(0, 2):
                    e_fun[k] = np.sqrt(gx[k] ** 2 + gy[k] ** 2)
                new_energy[i - 1, j - 2] = e_fun[0]
                new_energy[i - 1, j-1] = e_fun[1]
                new_energy[i - 1, 0:j - 2] = energy[i - 1, 0:j - 2]
                new_energy[i - 1, j:columns - 1] = energy[i - 1, j + 1:columns]
            elif i > 1 and j < 3:
                gx[0] = np.sum(fx * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gx[1] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gx[2] = np.sum(fx * gray_scale[i - 1:i + 2, j:j + 3])
                gy[0] = np.sum(fy * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gy[1] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[2] = np.sum(fy * gray_scale[i - 1:i + 2, j:j + 3])
                for k in range(0, 3):
                    e_fun[k] = np.sqrt(gx[k] ** 2 + gy[k] ** 2)
                new_energy[i - 2, j - 1] = e_fun[0]
                new_energy[i - 2, j:columns-1] = energy[i - 2, j + 1:columns]
                new_energy[i - 1, j - 1] = e_fun[1]
                new_energy[i - 1, j] = e_fun[2]
                new_energy[i - 1, j + 1:columns - 1] = energy[i - 1, j + 2:columns]
            elif i > 1 and j == columns-1:
                gx[0] = np.sum(fx * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gx[1] = np.sum(fx * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gx[2] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[0] = np.sum(fy * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gy[1] = np.sum(fy * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gy[2] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                for k in range(0, 3):
                    e_fun[k] = np.sqrt(gx[k] ** 2 + gy[k] ** 2)
                new_energy[i - 2, j-1] = e_fun[0]
                new_energy[i - 2, 0:j-1] = energy[i - 2, 0:j - 1]
                new_energy[i - 1, j - 2] = e_fun[1]
                new_energy[i - 1, j - 1] = e_fun[2]
                new_energy[i - 1, 0:j - 2] = energy[i - 1, 0:j - 2]
            else:
                gx[0] = np.sum(fx * gray_scale[i - 2:i + 1, j - 2:j + 1])
                gx[1] = np.sum(fx * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gx[2] = np.sum(fx * gray_scale[i - 1:i + 2, j - 3:j])
                gx[3] = np.sum(fx * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gx[4] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gx[5] = np.sum(fx * gray_scale[i - 1:i + 2, j:j + 3])
                gy[0] = np.sum(fy * gray_scale[i - 2:i + 1, j - 2:j + 1])
                gy[1] = np.sum(fy * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gy[2] = np.sum(fy * gray_scale[i - 1:i + 2, j - 3:j])
                gy[3] = np.sum(fy * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gy[4] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[5] = np.sum(fy * gray_scale[i - 1:i + 2, j:j + 3])
                for k in range(0, 6):
                    e_fun[k] = np.sqrt(gx[k] ** 2 + gy[k] ** 2)
                new_energy[i - 2, j - 2] = e_fun[0]
                new_energy[i - 2, j - 1] = e_fun[1]
                new_energy[i - 2, 0:j - 2] = energy[i - 2, 0:j - 2]
                new_energy[i - 2, j:columns - 1] = energy[i - 2, j + 1:columns]
                new_energy[i - 1, j - 3] = e_fun[2]
                new_energy[i - 1, j - 2] = e_fun[3]
                new_energy[i - 1, j - 1] = e_fun[4]
                new_energy[i - 1, j] = e_fun[5]
                new_energy[i - 1, 0:j - 3] = energy[i - 1, 0:j - 3]
                new_energy[i - 1, j + 1:columns - 1] = energy[i - 1, j + 2:columns]
        new_energy = new_energy[:, 0:-1]
    elif mode == 'row':
        for j in range(1,columns+1):
            if seam[j-1]<rows-1:
                i = seam[j-1]+1
            else: i = seam[j-1]
            if i == 1 and j == 1:
                gx[0] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[0] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                e_fun[0] = np.sqrt(gx[0] ** 2 + gy[0] ** 2)
                new_energy[i-1,j-1] = e_fun[0]
                new_energy[i:rows-1,j-1] = energy[i+1:rows,j-1]
            elif i > 1 and j == 1:
                gx[0] = np.sum(fx * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gx[1] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[0] = np.sum(fy * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gy[1] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                for k in range(0, 2):
                    e_fun[k] = int(np.sqrt(gx[k] ** 2 + gy[k] ** 2))
                new_energy[i - 2, j - 1] = e_fun[0]
                new_energy[i - 1, j - 1] = e_fun[1]
                new_energy[0:i - 2, j - 1] = energy[0:i - 2, j - 1]
                new_energy[i:rows - 1, j - 1] = energy[i + 1:rows, j - 1]
            elif i < 3 and j > 1:
                gx[0] = np.sum(fx * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gx[1] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gx[2] = np.sum(fx * gray_scale[i:i + 3, j - 1:j + 2])
                gy[0] = np.sum(fy * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gy[1] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[2] = np.sum(fy * gray_scale[i:i + 3, j - 1:j + 2])
                for k in range(0, 3):
                    e_fun[k] = int(np.sqrt(gx[k] ** 2 + gy[k] ** 2))
                new_energy[i - 1, j - 2] = e_fun[0]
                new_energy[i:rows - 1, j - 2] = energy[i + 1:rows, j - 2]
                new_energy[i - 1, j - 1] = e_fun[1]
                new_energy[i, j - 1] = e_fun[2]
                new_energy[i + 1:rows - 1, j - 1] = energy[i + 2:rows, j - 1]
            elif i == rows-1 and j > 1:
                gx[0] = np.sum(fx * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gx[1] = np.sum(fx * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gx[2] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[0] = np.sum(fy * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gy[1] = np.sum(fy * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gy[2] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                for k in range(0, 3):
                    e_fun[k] = int(np.sqrt(gx[k] ** 2 + gy[k] ** 2))
                new_energy[0:i - 1,j - 2] = energy[0:i - 1,j - 2]
                new_energy[i - 1, j - 2] = e_fun[0]
                new_energy[i - 2, j - 1] = e_fun[1]
                new_energy[i - 1, j - 1] = e_fun[2]
                new_energy[0:i - 2, j - 1] = energy[0:i - 2, j - 1]
            else:
                gx[0] = np.sum(fx * gray_scale[i - 2:i + 1, j - 2:j + 1])
                gx[1] = np.sum(fx * gray_scale[i - 1:i + 2,j - 2:j + 1])
                gx[2] = np.sum(fx * gray_scale[i - 3:i, j - 1:j + 2])
                gx[3] = np.sum(fx * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gx[4] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gx[5] = np.sum(fx * gray_scale[i:i + 3,j - 1:j + 2])
                gy[0] = np.sum(fy * gray_scale[i - 2:i + 1, j - 2:j + 1])
                gy[1] = np.sum(fy * gray_scale[i - 1:i + 2, j - 2:j + 1])
                gy[2] = np.sum(fy * gray_scale[i - 3:i, j - 1:j + 2])
                gy[3] = np.sum(fy * gray_scale[i - 2:i + 1, j - 1:j + 2])
                gy[4] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
                gy[5] = np.sum(fy * gray_scale[i:i + 3, j - 1:j + 2])
                for k in range(0, 6):
                    e_fun[k] = np.sqrt(gx[k] ** 2 + gy[k] ** 2)
                new_energy[i - 2, j - 2] = e_fun[0]
                new_energy[i - 1, j - 2] = e_fun[1]
                new_energy[0:i - 2, j - 2] = energy[0:i - 2, j - 2]
                new_energy[i:rows-1, j - 2] = energy[i+1:rows, j - 2]
                new_energy[i - 3, j - 1] = e_fun[2]
                new_energy[i - 2, j - 1] = e_fun[3]
                new_energy[i - 1, j - 1] = e_fun[4]
                new_energy[i, j - 1] = e_fun[5]
                new_energy[0:i - 3, j - 1] = energy[0:i - 3, j - 1]
                new_energy[i + 1:rows - 1, j - 1] = energy[i + 2:rows, j - 1]
        new_energy = new_energy[0:-1,:]
    return new_energy

def search_column_seam(energy):
    rows, columns = energy.shape
    column_seam = np.zeros(rows,dtype=int)
    dp = energy.copy()
    for i in range(1,rows):
        for j in range(0,columns):
            if j == 0:
                dp[i,j] = dp[i,j]+min(dp[i - 1,j], dp[i - 1,j + 1])
            elif j == columns - 1:
                dp[i, j] = dp[i, j] + min(dp[i - 1, j], dp[i - 1, j - 1])
            else:
                dp[i, j] = dp[i, j] + min(dp[i-1,j-1], dp[i - 1, j], dp[i - 1, j + 1])
    column_seam[-1] = np.argmin(dp[-1])
    for i in range(rows-2,-1,-1):
        next_row = column_seam[i+1]
        if next_row == 0:
            column_seam[i] = np.argmin(dp[i,:2])
        elif next_row == columns-1:
            column_seam[i] = next_row-1+np.argmin(dp[i,columns-2:])
        else:
            column_seam[i] = next_row-1+np.argmin(dp[i,next_row-1:next_row+2])
    return column_seam

def search_row_seam(energy):
    rows, columns = energy.shape
    row_seam = np.zeros(columns,dtype=int)
    dp = energy.copy()

    for j in range(1,columns):
        for i in range(0,rows):
            if i == 0:
                dp[i,j] = dp[i,j]+min(dp[i, j-1], dp[i+1,j-1])
            elif i == rows-1:
                dp[i, j] = dp[i, j] + min(dp[i, j - 1], dp[i - 1, j - 1])
            else:
                dp[i, j] = dp[i, j] + min(dp[i-1,j-1],dp[i, j - 1], dp[i + 1, j - 1])

    row_seam[-1] = np.argmin(dp[:,-1])
    for i in range(columns - 2, -1, -1):
        next_col = row_seam[i + 1]
        if next_col == 0:
            row_seam[i] = np.argmin(dp[:2,i])
        elif next_col == rows-1:
            row_seam[i] = next_col - 1+int(np.argmin(dp[rows-2:,i]))
        else:
            row_seam[i] = next_col - 1 + int(np.argmin(dp[next_col-1:next_col+2, i]))
    return row_seam

def delete_row(gray_scale,row_seam):
    row_num,column_num= gray_scale.shape
    result = np.zeros((row_num-1,column_num))
    for j in range(0,column_num):
        row = row_seam[j]
        result[:row,j] = gray_scale[:row,j]
        result[row:,j] = gray_scale[row+1:,j]
    return result

def delete_column(gray_scale,column_seam):
    row_num,column_num= gray_scale.shape
    result = np.zeros((row_num,column_num-1))
    for i in range(0,row_num):
        col = column_seam[i]
        result[i,:col] = gray_scale[i,:col]
        result[i,col:] = gray_scale[i,col+1:]
    return result

def calculate_row(gray_scale,row_seam):
    row_num, column_num = gray_scale.shape
    insert_energy = 0
    for j in range(1, column_num):
        i = row_seam[j]
        if row_seam[j - 1] == row_seam[j] and i > 0 and i < row_num - 1:
            insert_energy += np.abs(gray_scale[i + 1][j] - gray_scale[i - 1][j])
        elif row_seam[j - 1] == row_seam[j] - 1 and i < row_num - 1:
            insert_energy += np.abs(gray_scale[i - 1][j] - gray_scale[i + 1][j]) + \
                             np.abs(gray_scale[i - 1][j] - gray_scale[i][j - 1])
        elif row_seam[j - 1] == row_seam[j] + 1 and i > 0:
            insert_energy += np.abs(gray_scale[i + 1][j] - gray_scale[i][j - 1]) + \
                             np.abs(gray_scale[i + 1][j] - gray_scale[i - 1][j])
    return insert_energy

def calculate_column(gray_scale,column_seam):
    row_num, column_num = gray_scale.shape
    insert_energy = 0
    for i in range(1, row_num):
        j = column_seam[i]
        if column_seam[i - 1] == column_seam[i] and j > 0 and j < column_num - 1:
            insert_energy += np.abs(gray_scale[i][j - 1] - gray_scale[i][j + 1])
        elif column_seam[i - 1] == column_seam[i] - 1 and j < column_num - 1:
            insert_energy += np.abs(gray_scale[i][j - 1] - gray_scale[i - 1][j - 1]) + \
                             np.abs(gray_scale[i][j - 1] - gray_scale[i][j + 1])
        elif column_seam[i - 1] == column_seam[i] + 1 and j > 0:
            insert_energy += np.abs(gray_scale[i][j - 1] - gray_scale[i][j + 1]) + \
                             np.abs(gray_scale[i][j + 1] - gray_scale[i - 1][j + 1])
    return insert_energy

if __name__ == "__main__":
    image_name = 'Dolphin.bmp'
    image = cv2.imread('./Images/{}'.format(image_name))
    # rows_columns = input("please input need removed rows and colunms")
    need_removed_rows = need_removed_columns = 40  # rows_columns.split()
    image = cv2.imread('./Images/{}'.format(image_name))
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_scale = np.array(gray_scale)
    color_image = []
    removed_rows = 0
    removed_columns = 0
    red = np.array(image[:, :, 0])
    green = np.array(image[:, :, 1])
    blue = np.array(image[:, :, 2])
    energy = init_energy_function(gray_scale)

    t1 = time.time()
    while removed_columns < need_removed_columns or removed_rows < need_removed_rows:
        print(gray_scale.shape)
        column_seam = search_column_seam(energy)
        row_seam = search_row_seam(energy)
        row_insert_energy = calculate_row(gray_scale, row_seam)
        column_insert_energy = calculate_column(gray_scale, column_seam)
        if row_insert_energy < column_insert_energy:
            if removed_rows < need_removed_rows:
                gray_scale = delete_row(gray_scale, row_seam)
                red = delete_row(red, row_seam)
                green = delete_row(green, row_seam)
                blue = delete_row(blue, row_seam)
                energy = update_energy_function(gray_scale, row_seam, energy, 'row')
                removed_rows += 1
            else:
                gray_scale = delete_column(gray_scale, column_seam)
                red = delete_column(red, column_seam)
                green = delete_column(green, column_seam)
                blue = delete_column(blue, column_seam)
                energy = update_energy_function(gray_scale, column_seam, energy, 'col')
                removed_columns += 1
        elif column_insert_energy < row_insert_energy:
            if removed_columns < need_removed_columns:
                gray_scale = delete_column(gray_scale, column_seam)
                red = delete_column(red, column_seam)
                green = delete_column(green, column_seam)
                blue = delete_column(blue, column_seam)
                energy = update_energy_function(gray_scale, column_seam, energy, 'col')
                removed_columns += 1
            else:
                gray_scale = delete_row(gray_scale, row_seam)
                red = delete_row(red, row_seam)
                green = delete_row(green, row_seam)
                blue = delete_row(blue, row_seam)
                energy = update_energy_function(gray_scale, row_seam, energy, 'row')
                removed_rows += 1

    t2 = time.time()
    print("Time of Fast and multi-thread running in {}:".format(image_name), t2 - t1)
    color_image.append(red)
    color_image.append(green)
    color_image.append(blue)
    # gray_scale = Image.fromarray(gray_scale.astype('uint8'))
    color_image = np.array(color_image)
    color_image = np.transpose(color_image, axes=(1, 2, 0))
    color_image = color_image.astype('uint8')
    cv2.imwrite('./fast_with_parallel_resized_{}'.format(image_name), color_image)
    print("done")