import copy

def data_preprocess(path_train, Dist):
    '''
    Preprocess the dataset for training the DFA

    Parameters:
    path_train (str): path of a training dataset
    Dist (function): Distance function to use

    Returns:
    tuple: A tuple containing alphabet, Pref_S, Lst, FL, dist_matrix
    '''

    with open(path_train, "r") as my_file:
        Lst = [line.strip() for line in my_file.readlines()]
    
    #FL is required to calculate the distance
    FL = [l.split(",") for l in Lst]
        
    uniq_list=set(Lst)
    alphabet = set(item for string in uniq_list for item in string.split(",") if item != ',')
    
    Pref_S = set()
    for string in uniq_list:
        prefixes = string.split(',')
        for i in range(1, len(prefixes) + 1):
            prefix = ','.join(prefixes[:i])
            Pref_S.add(prefix)
    Pref_S.add('')

    dist_matrix = Dist(copy.deepcopy(FL))
    #np.savetxt('C:/Users/bchan/Downloads/result_dist.txt', dist, fmt='%.0f')
    #print(np.max(dist))
    
    return alphabet, Pref_S, Lst, FL, dist_matrix