
from Levenshtein import distance
from scipy.spatial.distance import hamming
import numpy as np

def levenshtein_distance(str_list):
    
    dist_matrix = np.zeros(shape=(len(str_list), len(str_list)))
    #t0 = time()
    #print("Starting to build distance matrix. This will iterate from 0 till ", len(str_list)) 
    for i in range(0, len(str_list)):
        #print(i)
        for j in range(i+1, len(str_list)):
                dist_matrix[i][j] = distance(str_list[i], str_list[j]) 
    for i in range(0, len(str_list)):
        for j in range(0, len(str_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
    #t1 = time()
    #print("took", (t1-t0), "seconds")
    return dist_matrix

#dist =get_levenshtein_distance_matrix(FL)

def make_lists_equal_length(list1, list2):
    if len(list1) == len(list2):
        return list1, list2

    if len(list1) < len(list2):
        list1 += ['a'] * (len(list2) - len(list1))
    else:
        list2 += ['a'] * (len(list1) - len(list2))

    return list1, list2

def hamming_distance(list_list):
    #distance = 0
    dist_matrix = np.zeros(shape=(len(list_list), len(list_list)))
    for i in range(0, len(list_list)):
        #print(i)
        for j in range(i+1, len(list_list)):
            if len(list_list[i]) != len(list_list[j]):
                list_list[i], list_list[j]= make_lists_equal_length(list_list[i], list_list[j])

            dist_matrix[i][j] = hamming(list(list_list[i]), list(list_list[j]))*len(list(list_list[i]))
        
    for i in range(0, len(list_list)):
        for j in range(0, len(list_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
    
    return dist_matrix

def custom_distance(str_list):
    
    dist_matrix = np.zeros(shape=(len(str_list), len(str_list)))
    for i in range(0, len(str_list)):
        for j in range(i+1, len(str_list)):
            if str_list[i][-1] == str_list[j][-1]:
                dist_matrix[i][j] = abs(round(float(str_list[i][0])-float(str_list[j][0]),1))
            else:           
                dist_matrix[i][j] = abs(round(float(str_list[i][0])+float(str_list[j][0])-1,1))
    for i in range(0, len(str_list)):
        for j in range(0, len(str_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
    return dist_matrix

def custom_distanceW(str_list):
    
    dist_matrix = np.zeros(shape=(len(str_list), len(str_list)))
    for i in range(0, len(str_list)):
        for j in range(i+1, len(str_list)):
            dist_matrix[i][j] = sum(abs(str_list[i]-str_list[j]))

    for i in range(0, len(str_list)):
        for j in range(0, len(str_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
    return dist_matrix

