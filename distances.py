
from Levenshtein import distance
from scipy.spatial.distance import hamming
import numpy as np
from .utils import make_lists_equal_length

def levenshtein_distance(Lst_list):
    '''Calculate levenshtein distance between all pairs of list in a given list
    
    Parameter:
    Lst_list (list): List of list

    Returns:
    dist_matrix (numpy.ndarray): square matrix containing distance between all pair of list    
    '''
    
    dist_matrix = np.zeros(shape=(len(Lst_list), len(Lst_list)))
    for i in range(0, len(Lst_list)):
        for j in range(i+1, len(Lst_list)):
                dist_matrix[i][j] = distance(Lst_list[i], Lst_list[j]) 
    for i in range(0, len(Lst_list)):
        for j in range(0, len(Lst_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]

    return dist_matrix

def hamming_distance(Lst_list):
    '''Calculate Hamming distance between all pairs of list in a given list
    
    Parameters:
    Lst_list (list): List of list for which distance are to be calculated

    Returns:
    dist_matrix (numpy.ndarray): square matrix containing distance between all pair of list    
    '''

    dist_matrix = np.zeros(shape=(len(Lst_list), len(Lst_list)))
    for i in range(0, len(Lst_list)):
        for j in range(i+1, len(Lst_list)):
            if len(Lst_list[i]) != len(Lst_list[j]):
                Lst_list[i], Lst_list[j]= make_lists_equal_length(Lst_list[i], Lst_list[j])

            dist_matrix[i][j] = hamming(list(Lst_list[i]), list(Lst_list[j]))*len(list(Lst_list[i]))
        
    for i in range(0, len(Lst_list)):
        for j in range(0, len(Lst_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
    
    return dist_matrix

def custom_distance(Lst_list):
    
    dist_matrix = np.zeros(shape=(len(Lst_list), len(Lst_list)))
    for i in range(0, len(Lst_list)):
        for j in range(i+1, len(Lst_list)):
            if Lst_list[i][-1] == Lst_list[j][-1]:
                dist_matrix[i][j] = abs(round(float(Lst_list[i][0])-float(Lst_list[j][0]),1))
            else:           
                dist_matrix[i][j] = abs(round(float(Lst_list[i][0])+float(Lst_list[j][0])-1,1))
    for i in range(0, len(Lst_list)):
        for j in range(0, len(Lst_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
    return dist_matrix

def custom_distanceW(Lst_list):
    
    dist_matrix = np.zeros(shape=(len(Lst_list), len(Lst_list)))
    for i in range(0, len(Lst_list)):
        for j in range(i+1, len(Lst_list)):
            dist_matrix[i][j] = sum(abs(Lst_list[i]-Lst_list[j]))

    for i in range(0, len(Lst_list)):
        for j in range(0, len(Lst_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
    return dist_matrix

