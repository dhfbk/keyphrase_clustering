import os
import sys
import codecs
from kpaggregator import functions
import getopt
import operator

threshold_multi=0.8
threshold_single=0.8
threshold_multi_split=0.8
threshold_single_split=0.8
maxlen=10000

vec_file='glove.6B.50d.txt'

try:
    opts, args = getopt.getopt(sys.argv[1:], "v:d:t:s:m:",["","","","",""])
except getopt.GetoptError:
    sys.exit(2)
for opt, arg in opts:
    if opt == '-v':
        vec_file = arg
    if opt == '-d':
        keywords_input_dir= arg
    if opt == '-m':
        maxlen= int(arg)
    if opt == '-t':
        threshold_multi = float(arg)
        threshold_single = threshold_multi
        threshold_multi_split = float(arg)
        threshold_single_split = threshold_multi_split
    if opt == '-s':
        threshold_multi_split = float(arg)
        threshold_single_split = threshold_multi_split


sys.stdout.write("Loading vectors")
glove_vectors_dic=dict()
for line in codecs.open(vec_file, 'r',  "utf-8"):
    parts = line.rstrip().split(" ")
    word, weights = parts[0], list(map(float, parts[1:]))
    glove_vectors_dic[word]=weights
sys.stdout.write("...done\n\n")

for filename in os.listdir(keywords_input_dir):
    if not filename.startswith('.'):
        sys.stdout.write("Processing ")
        sys.stdout.write(filename)
        sys.stdout.write("\n")

        sys.stdout.write("Making keyword dicts")
        (keywords_dic,tfidf_dict) = functions.make_keywords_dicts(os.path.join(keywords_input_dir, filename))
        sys.stdout.write("...Done\n")
        sys.stdout.write("Average keyword vectors")
        mean_kw_vectors  =functions.average_keyword_vectors(glove_vectors_dic, keywords_dic)
        sys.stdout.write("...Done\n")
        sys.stdout.write("Building graph")
        
        cosines_dict = functions.compute_keywords_cosine(keywords_dic, mean_kw_vectors, threshold_multi, threshold_single)
        sys.stdout.write("...Done\n")
        sys.stdout.write("Creating clusters")
        sorted_pairs_cosines_dict=functions.sort_keyword_pairs(cosines_dict,tfidf_dict)
        groups_list=functions.merge_on_first_keyword(sorted_pairs_cosines_dict)
        groups_list=functions.merge_overlapping_groups(groups_list, 0.5)
        groups_list=functions.merge_overlapping_groups_of_2(groups_list)
        groups_list=functions.merge_overlapping_groups(groups_list, 0.5)
        sys.stdout.write("...Done\n")

        sys.stdout.write("Cleaning clusters")
        groups_list=functions.clean_by_standard_deviation(groups_list, mean_kw_vectors, 1.5, 3)
        sys.stdout.write("...Done\n")
        sys.stdout.write("Splitting large clusters")
        groups_output_list=list()
        for group in groups_list:
            if len(group) <maxlen:
                groups_output_list.append(group)
            else:
                tmp_group_list=list()
                (tmp_k_dict, tmp_t_dict)=functions.list_to_keywords_dict(group)
                cosines_dict = functions.compute_keywords_cosine_all(tmp_k_dict, mean_kw_vectors, threshold_multi_split, threshold_single_split)
                sorted_pairs_cosines_dict = functions.sort_keyword_pairs(cosines_dict, tfidf_dict)
                tmp_group_list = functions.merge_on_first_keyword(sorted_pairs_cosines_dict)
                tmp_group_list = functions.merge_overlapping_groups(tmp_group_list, 0.5)
                tmp_group_list = functions.merge_overlapping_groups_of_2(tmp_group_list)
                tmp_group_list = functions.merge_overlapping_groups(tmp_group_list, 0.5)
                tmp_group_list = functions.clean_by_standard_deviation(tmp_group_list, mean_kw_vectors, 1.5, 3)
                groups_output_list.extend(tmp_group_list)
        sys.stdout.write("...Done\n")

        sys.stdout.write("Printing clusters")
        outname = filename+"_clusters.txt"
        groups_out = open(outname, 'w')
        for g in groups_output_list:
            for i in g:
                groups_out.write(i)
                groups_out.write("\t")
            groups_out.write("\n")
        sys.stdout.write("...Done\n")


        sys.stdout.write(str(len(groups_output_list)))
        sys.stdout.write(" clusters found\n\n")
