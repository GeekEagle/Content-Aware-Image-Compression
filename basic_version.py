import cv2
import numpy as np
import time
import multiprocessing
from PIL import Image

fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
def energy_function(gray_scale):
    rows,columns = gray_scale.shape
    Gx = np.zeros((rows,columns))
    Gy = np.zeros((rows,columns))
    gray_scale = np.pad(gray_scale, pad_width=1, mode='constant', constant_values=0)
    for i in range(1,rows+1):
        for j in range(1,columns+1):
            Gx[i - 1][j - 1] = np.sum(fx * gray_scale[i - 1:i + 2, j - 1:j + 2])
            Gy[i - 1][j - 1] = np.sum(fy * gray_scale[i - 1:i + 2, j - 1:j + 2])
    Energy = np.sqrt(Gx ** 2 + Gy ** 2)
    return Energy


def search_column_seam(energy):
    rows, columns = energy.shape
    column_seam = np.zeros(rows,dtype=int)
    dp = energy.copy()
    for i in range(1,rows):
        for j in range(0,columns):
            if j == 0:
                dp[i,j] = dp[i,j]+min(dp[i-1,j], dp[i-1,j+1])
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
    need_removed_rows = need_removed_columns = 40
    # image_name = input()
    image_name = 'Image2.bmp'
    image = cv2.imread('./Images/{}'.format(image_name))
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_scale = np.array(gray_scale)
    color_image = []
    removed_rows = 0
    removed_columns = 0
    red = np.array(image[:,:,0])
    green = np.array(image[:,:,1])
    blue = np.array(image[:,:,2])

    t1 = time.time()
    while removed_columns < need_removed_columns or removed_rows < need_removed_rows:
        print(gray_scale.shape)
        energy = energy_function(gray_scale)
        column_seam = search_column_seam(energy)
        row_seam = search_row_seam(energy)
        row_insert_energy = calculate_row(gray_scale, row_seam)
        column_insert_energy = calculate_column(gray_scale, column_seam)
        if row_insert_energy < column_insert_energy:
            if removed_rows < need_removed_rows:
                gray_scale = delete_row(gray_scale, row_seam)
                red = delete_row(red,row_seam)
                green = delete_row(green,row_seam)
                blue = delete_row(blue,row_seam)
                removed_rows += 1
            else:
                gray_scale = delete_column(gray_scale, column_seam)
                red = delete_column(red,column_seam)
                green = delete_column(green,column_seam)
                blue = delete_column(blue,column_seam)
                removed_columns += 1
        elif column_insert_energy < row_insert_energy:
            if removed_columns < need_removed_columns:
                gray_scale = delete_column(gray_scale, column_seam)
                red = delete_column(red, column_seam)
                green = delete_column(green, column_seam)
                blue = delete_column(blue, column_seam)
                removed_columns += 1
            else:
                gray_scale = delete_row(gray_scale, row_seam)
                red = delete_row(red, row_seam)
                green = delete_row(green, row_seam)
                blue = delete_row(blue, row_seam)
                removed_rows += 1

    t2 = time.time()
    print("Time of Normal way  running in {}:".format(image_name), t2 - t1)
    color_image.append(red)
    color_image.append(green)
    color_image.append(blue)
    # gray_scale = Image.fromarray(gray_scale.astype('uint8'))
    color_image = np.array(color_image)
    color_image = np.transpose(color_image,axes=(1,2,0))
    color_image = color_image.astype('uint8')
    # color_image = cv2.cvtColor(gray_scale, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('./basic_version_resized_{}.jpg'.format(image_name), color_image)
    print("done")
    # cv2.imshow('resized_image',color_image)




