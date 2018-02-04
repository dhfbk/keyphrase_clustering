
import codecs
import collections
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def make_keywords_dicts(file):
    keyword_dict = dict()
    tfidf_dict = dict()
    max_kw_lenght=0
    for line in codecs.open(file, 'r', "utf-8"):
        splitted_line = line.split('\t')
        tokens=splitted_line[0].count('_')+1
        if tokens>max_kw_lenght:
            max_kw_lenght=tokens
        if tokens not in keyword_dict.keys():
             keyword_dict[tokens] = list()
        keyword_dict[tokens].append(splitted_line[0].rstrip())
        tfidf_dict[splitted_line[0]]=splitted_line[1]
    return (keyword_dict,tfidf_dict)

def list_to_keywords_dict(keyword_list):
    keyword_dict = dict()
    tfidf_dict = dict()
    max_kw_lenght=0
    for line in keyword_list:
        splitted_line = line
        tokens=splitted_line.count('_')+1
        if tokens>max_kw_lenght:
            max_kw_lenght=tokens
        if tokens not in keyword_dict.keys():
             keyword_dict[tokens] = list()
        keyword_dict[tokens].append(splitted_line.rstrip())
        tfidf_dict[splitted_line]=1
    return (keyword_dict,tfidf_dict)


def average_keyword_vectors(vectors_dic, keywords_dic):
    max_kw_lenght=max(keywords_dic.keys())
    mean_kw_vectors=dict()
    for n in range (1,max_kw_lenght+1):
        for k in keywords_dic[n]:
            word_found=False
            words = k.rstrip().split("_")
            vectorlist = list()
            for w in words:
                if w in vectors_dic:
                    word_found=True
                    vectorlist.append(vectors_dic[w])
            if word_found==False:
                empty_list=[]
                vectorlist.append(empty_list)
            a = np.array(vectorlist)
            mean_vec = np.mean(a, axis=0)
            mean_kw_vectors[k] = mean_vec
    return mean_kw_vectors

def compute_keywords_cosine(keywords_dic, mean_kw_vectors, threshold_multiwords, threshold_single):
    cosines_dict=dict()
    max_kw_lenght = max(keywords_dic.keys())
    for x in range(max_kw_lenght, 1, -1):
        if x in keywords_dic:
            for k1 in keywords_dic[x]:
                for y in range(x - 1, 0, -1):
                    if y in keywords_dic:
                        for k2 in keywords_dic[y]:
                            if k2 in k1:
                                if len(mean_kw_vectors[k1]) > 1 and len(mean_kw_vectors[k2]) > 1:
                                    vector_keyword_one = mean_kw_vectors[k1].reshape(1, -1)
                                    vector_keyword_two = mean_kw_vectors[k2].reshape(1, -1)
                                    cosine_value = cosine_similarity(vector_keyword_one, vector_keyword_two)
                                    if cosine_value[0][0] > threshold_multiwords:
                                        pair_string = k1 + ' ' + k2
                                        cosines_dict[pair_string] = cosine_value[0][0]
    if 1 in keywords_dic:
        for k1 in keywords_dic[1]:
            for k2 in keywords_dic[1]:
                if k1 != k2:
                    if len(mean_kw_vectors[k1]) > 1 and len(mean_kw_vectors[k2]) > 1:
                        vector_keyword_one = mean_kw_vectors[k1].reshape(1, -1)
                        vector_keyword_two = mean_kw_vectors[k2].reshape(1, -1)
                        cosine_value = cosine_similarity(vector_keyword_one, vector_keyword_two)
                        if cosine_value[0][0] > threshold_single:
                            pair_string= k1 + ' ' + k2
                            cosines_dict[pair_string]=cosine_value[0][0]
    return cosines_dict


def compute_keywords_cosine_compare_all(keywords_dic, mean_kw_vectors, threshold_multiwords, threshold_single):
    cosines_dict = dict()
    max_kw_lenght = max(keywords_dic.keys())
    for x in range(max_kw_lenght, 1, -1):
        if x in keywords_dic:
            for k1 in keywords_dic[x]:
                for y in range(x - 1, 0, -1):
                    if y in keywords_dic:
                        for k2 in keywords_dic[y]:
                            if len(mean_kw_vectors[k1]) > 1 and len(mean_kw_vectors[k2]) > 1:
                                vector_keyword_one = mean_kw_vectors[k1].reshape(1, -1)
                                vector_keyword_two = mean_kw_vectors[k2].reshape(1, -1)
                                cosine_value = cosine_similarity(vector_keyword_one, vector_keyword_two)
                                if cosine_value[0][0] > threshold_multiwords:
                                    pair_string = k1 + ' ' + k2
                                    cosines_dict[pair_string] = cosine_value[0][0]
    if 1 in keywords_dic:
        for k1 in keywords_dic[1]:
            for k2 in keywords_dic[1]:
                if k1 != k2:
                    if len(mean_kw_vectors[k1]) > 1 and len(mean_kw_vectors[k2]) > 1:
                        vector_keyword_one = mean_kw_vectors[k1].reshape(1, -1)
                        vector_keyword_two = mean_kw_vectors[k2].reshape(1, -1)
                        cosine_value = cosine_similarity(vector_keyword_one, vector_keyword_two)
                        if cosine_value[0][0] > threshold_single:
                            pair_string = k1 + ' ' + k2
                            cosines_dict[pair_string] = cosine_value[0][0]

    return cosines_dict


def compute_keywords_cosine_all(keywords_dic, mean_kw_vectors, threshold_multiwords, threshold_single):
    cosines_dict = dict()
    max_kw_lenght = max(keywords_dic.keys())
    for x in range(max_kw_lenght, 1, -1):
        if x in keywords_dic:
            for k1 in keywords_dic[x]:
                for y in range(x - 1, 0, -1):
                    if y in keywords_dic:
                        for k2 in keywords_dic[y]:
                            if len(mean_kw_vectors[k1]) > 1 and len(mean_kw_vectors[k2]) > 1:
                                vector_keyword_one = mean_kw_vectors[k1].reshape(1, -1)
                                vector_keyword_two = mean_kw_vectors[k2].reshape(1, -1)
                                cosine_value = cosine_similarity(vector_keyword_one, vector_keyword_two)
                                if cosine_value[0][0] > threshold_multiwords:
                                    pair_string = k1 + ' ' + k2
                                    cosines_dict[pair_string] = cosine_value[0][0]

    if 1 in keywords_dic:
        for k1 in keywords_dic[1]:
            for k2 in keywords_dic[1]:
                if k1 != k2:
                    if len(mean_kw_vectors[k1]) > 1 and len(mean_kw_vectors[k2]) > 1:
                        vector_keyword_one = mean_kw_vectors[k1].reshape(1, -1)
                        vector_keyword_two = mean_kw_vectors[k2].reshape(1, -1)
                        cosine_value = cosine_similarity(vector_keyword_one, vector_keyword_two)
                        if cosine_value[0][0] > threshold_single:
                            pair_string = k1 + ' ' + k2
                            cosines_dict[pair_string] = cosine_value[0][0]
                            
    return cosines_dict


def sort_keyword_pairs(cosines_dict,tfidf_dict):
    sorted_pairs_cosines_dict=dict()
    for pair in cosines_dict.keys():
        kw1, kw2 = pair.rstrip().split(" ")
        tfidf_a = tfidf_dict[kw1]
        tfidf_b = tfidf_dict[kw2]
        if tfidf_a > tfidf_b:
            pair_string = kw1 + ' ' + kw2
            sorted_pairs_cosines_dict[pair_string] = cosines_dict[pair]
        elif tfidf_a < tfidf_b:
            pair_string = kw2 + ' ' + kw1
            sorted_pairs_cosines_dict[pair_string] = cosines_dict[pair]
        elif tfidf_a == tfidf_b:
            pair_string = kw1 + ' ' + kw2
            sorted_pairs_cosines_dict[pair_string] = cosines_dict[pair]
            pair_string = kw2 + ' ' + kw1
            sorted_pairs_cosines_dict[pair_string] = cosines_dict[pair]
    return sorted_pairs_cosines_dict

def sort_keyword_pairs_reverse(cosines_dict,tfidf_dict):
    sorted_pairs_cosines_dict=dict()
    for pair in cosines_dict.keys():
        kw1, kw2 = pair.rstrip().split(" ")
        tfidf_a = tfidf_dict[kw1]
        tfidf_b = tfidf_dict[kw2]
        if tfidf_a < tfidf_b:
            pair_string = kw1 + ' ' + kw2
            sorted_pairs_cosines_dict[pair_string] = cosines_dict[pair]
        elif tfidf_a > tfidf_b:
            pair_string = kw2 + ' ' + kw1
            sorted_pairs_cosines_dict[pair_string] = cosines_dict[pair]
        elif tfidf_a == tfidf_b:
            pair_string = kw1 + ' ' + kw2
            sorted_pairs_cosines_dict[pair_string] = cosines_dict[pair]
            pair_string = kw2 + ' ' + kw1
            sorted_pairs_cosines_dict[pair_string] = cosines_dict[pair]
    return sorted_pairs_cosines_dict

def merge_on_first_keyword(sorted_pairs_cosines_dict):
    groups_dict=dict()
    for pair in sorted_pairs_cosines_dict.keys():
        kwords=pair.rstrip().split('\t')[0].split(' ')
        kw1=kwords[0]
        kw2=kwords[1]
        if kw1 not in groups_dict.keys():
            empty=[]
            groups_dict[kw1] = empty
            groups_dict[kw1].append(kw1)
        groups_dict[kw1].append(kw2)
    groups_list = []
    for g in groups_dict.keys():
        groups_list.append(groups_dict[g])
    return groups_list

def merge_overlapping_groups(groups_list, overlap_value):
    lists_list = list()
    items_to_remove = dict()
    new_groups = dict()
    items_to_remove = dict()
    new_groups = dict()
    for g in groups_list:
        lists_list.append(sorted(list(set(g))))
    lists_list=list(set(tuple(i) for i in lists_list))
    overlap_found = True
    while overlap_found==True:
        overlap_found=False
        max_index=len(lists_list)-1
        for x in range(0, max_index,+1):
            for y in range(x+1, max_index+1, +1):
                a = lists_list[x]
                b = lists_list[y]
                
                a_multiset = collections.Counter(a)
                b_multiset = collections.Counter(b)
                overlap = list((a_multiset & b_multiset).elements())


                if  len(overlap) > overlap_value*(min(len(a), len(b))):
                    

                    overlap_found=True
                    items_to_remove[x]=1
                    items_to_remove[y]=1
                    if len(lists_list[x])<len(lists_list[y]):
                        if y not in new_groups.keys():
                            new_groups[y] = list(lists_list[y])
                        new_groups[y].extend(lists_list[x])
                    else:
                        if x not in new_groups.keys():
                            new_groups[x] = list(lists_list[x])
                        new_groups[x].extend(lists_list[y])
        groups_next_cycle=list()
        
        for x in range(0, max_index+1, +1):
            if x in new_groups.keys():
                groups_next_cycle.append(new_groups[x])
            elif x not in items_to_remove.keys():
                groups_next_cycle.append(lists_list[x])
        lists_list=list()
        for i in groups_next_cycle:
            lists_list.append(sorted(list(set(i))))
        
        lists_list=list(set(tuple(i) for i in lists_list))
        
        items_to_remove = dict()
        new_groups = dict()
    
    return lists_list

def merge_overlapping_groups_of_2(groups_list):
    lists_list = list()
    items_to_remove = dict()
    new_groups = dict()
    for g in groups_list:
        lists_list.append(sorted(list(set(g))))
    lists_list = list(set(tuple(i) for i in lists_list))
    overlap_found = True
    while overlap_found == True:
        new_list = list()
        overlap_found = False
        max_index = len(lists_list) - 1
        for x in range(0, max_index, +1):
            if len(lists_list[x]) == 2:
                for y in range(x + 1, max_index + 1, +1):
                    if len(lists_list[y]) == 2:
                        a = lists_list[x]
                        b = lists_list[y]
                        a_multiset = collections.Counter(a)
                        b_multiset = collections.Counter(b)
                        overlap = list((a_multiset & b_multiset).elements())
                        if len(overlap) > 0.49 * (min(len(a), len(b))):
                            overlap_found = True
                            items_to_remove[x] = 1
                            items_to_remove[y] = 1
                            if x not in new_groups.keys():
                                new_groups[x] = list(lists_list[x])
                            new_groups[x].extend(lists_list[y])
        groups_next_cycle = list()
        for x in range(0, max_index+1, +1):
            if x in new_groups.keys():
                groups_next_cycle.append(new_groups[x])
            elif x not in items_to_remove.keys():
                groups_next_cycle.append(lists_list[x])
        lists_list = list()
        for i in groups_next_cycle:
            lists_list.append(sorted(list(set(i))))
        items_to_remove = dict()
        new_groups = dict()
    return lists_list

def clean_by_standard_deviation(group_list, vectors_dic, threshold, min_groups_length):
    lists_list = list()
    for splitted_group in group_list:
        if len(splitted_group) > min_groups_length-1:
            vectorlist = list()
            for kw in splitted_group:
                if kw in vectors_dic:
                    vectorlist.append(vectors_dic[kw])
            a = np.array(vectorlist)
            mean_vec = np.mean(a, axis=0)
            cosines_dict = dict()
            cosine_list = list()
            for kw in splitted_group:
                mean_vec = mean_vec.reshape(1, -1)
                vectors_to_compare = vectors_dic[kw].reshape(1, -1)
                cosine_value = 1 - cosine_similarity(mean_vec, vectors_to_compare)
                for i in cosine_value.tolist():
                    cosine_list.extend(i)
                for i in cosine_value.tolist():
                    cosines_dict[kw] = i
            mean_cosine = np.mean(cosine_list)
            stdev = np.std(cosine_list, dtype=np.float64)
            counter = 0
            out_list = list()
            for c in cosine_list:
                if c <= (mean_cosine + (threshold* stdev)):
                    out_list.append(splitted_group[counter])
                counter = counter + 1
            lists_list.append(sorted(list(set(out_list))))
    return lists_list
