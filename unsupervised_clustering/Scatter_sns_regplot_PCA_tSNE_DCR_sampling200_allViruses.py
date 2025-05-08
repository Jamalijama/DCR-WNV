
from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import math
from math import cos,sin,radians
from sklearn.linear_model import LinearRegression
import os

amino_table = ['I', 'D', 'M', 'H', 'E', 'W', 'R', 'L', 'Y', 'Q', 'G', 'A', 'S', 'P', 'C', 'T', 'V', 'F', 'N', 'K']
nt_table = ['t','c', 'a', 'g']
dnt_table = [nt1+nt2 for nt1 in nt_table for nt2 in nt_table]
dnts_table = [dnt1+dnt2 for dnt1 in dnt_table for dnt2 in dnt_table]
codon_table = [nt1+nt2+nt3 for nt1 in nt_table for nt2 in nt_table for nt3 in nt_table]
codon_table1 = codon_table.copy()
codon_table1.remove('taa')
codon_table1.remove('tag')
codon_table1.remove('tga')
codonpair_table = [codon0 + codon1 for codon0 in codon_table1 for codon1 in codon_table1]
dnt_category = ['n12', 'n23','n31']
dntpair_category = ['n12m12', 'n23m23','n31m31','n12n31','n23m12','n31m23']
dnts_cols_list = ['Freq_'+ dnt + '_' + dnt_cat for dnt_cat in dnt_category for dnt in dnt_table]
dntpair_cols_list = ['Freq_'+ dnts + '_' + dnts_cat for dnts_cat in dntpair_category for dnts in dnts_table]
codon_cols_list = ['Freq_'+ codon for codon in codon_table]
codonpair_cols_list = ['Freq_'+ codonpair for codonpair in codonpair_table]
amino_cols_list = ['Freq_'+ amino for amino in amino_table]
# full_cols_list0 = dnts_cols_list + dntpair_cols_list + codon_cols_list + codonpair_cols_list + amino_cols_list

dcr_set_name_list = ['dnts', 'codons', 'aminos', 'DCR','codonpair']
dcr_set_list = [dnts_cols_list,codon_cols_list, amino_cols_list,  dntpair_cols_list, codonpair_cols_list]

cnames = {
'lightblue':            '#ADD8E6',
'deepskyblue':          '#00BFFF',
'cadetblue':            '#5F9EA0',
'cyan':                 '#00FFFF',
'purple':               '#800080',
'orchid':               '#DA70D6',
'lightgreen':           '#90EE90',
'darkgreen':            '#006400',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32',
'deeppink':             '#FF1493',
'burlywood':            '#DEB887',
'red':                  '#FF0000',
'indianred':            '#CD5C5C',
'darkred':              '#8B0000',
    }

cnames = {
'lightblue':            '#ADD8E6',
'deepskyblue':          '#00BFFF',
'purple':               '#800080',
'orchid':               '#DA70D6',
'lightgreen':           '#90EE90',
'darkgreen':            '#006400',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32',
'deeppink':             '#FF1493',
'burlywood':            '#DEB887',
'red':                  '#FF0000',
'indianred':            '#CD5C5C',
'darkred':              '#8B0000',
    }
# color_num_list = list (range(1,16,1))
color_num_list = list (range(1,16,1))

# print (len(color_num_list))
color_dict = dict(zip(color_num_list,cnames.values()))
# print (color_dict)
color_list0 = list(color_dict.values())


color_list_all = []
for i in color_list0:
    color_list_all.append(i)
# print (color_list_all)


df_labels = pd.read_csv ('../data/df_concat_cds_record_information_HPV_CompleteGenome6.5to8k.csv', index_col = 'seqID', encoding = 'gbk')
label_lst0 = df_labels['type'].tolist()
# y_types = list(set(label_lst))
# print (y_types)
subtype_lst0 = df_labels['organism'].tolist()

path = '../counting_seq/'
for file in os.listdir(path):
    if file.endswith ('.csv') & file.startswith('df_full_counting'):
        
        gene = file[-6:-4]
        print (file, gene)
        
        df_DCR0 = pd.read_csv (path + file, index_col = 'accession')
        df_DCR0['label'] = label_lst0
        df_DCR0['subtype'] = subtype_lst0

        
        print (df_DCR0.shape)
        df_DCR0 = df_DCR0[df_DCR0['label'] != 'Zurh']
        
        df_alpha = df_DCR0[df_DCR0['label'] == 'Alpha']
        df_alpha = df_alpha.sample(n = 200, random_state = 1)
 
        df_gamma = df_DCR0[df_DCR0['label'] == 'Gamma']
        df_gamma = df_gamma.sample(n = 200, random_state = 1)


        df_others = df_DCR0[(df_DCR0['label'] != 'Alpha')&(df_DCR0['label'] != 'Gamma')]
        df_DCR = pd.concat([df_alpha, df_gamma, df_others], axis = 0)
        num = df_DCR.shape[0]
        
        index_lst = df_DCR.index.tolist()
        subtypes = df_DCR['subtype'].tolist()
        
        
        label_lst = df_DCR['label'].tolist()
        y_types = list(set(label_lst))
        
        # print (y_types)
        y_num = len(y_types)
        label_cate_list = y_types
        label_c_list = list(range(1,6,1))
        # print (label_c_list)
    
        dict1 = dict(zip(label_cate_list,label_c_list))
        dict2 = dict(zip(label_cate_list,color_dict))
        label_list1 = [dict1[i] for i in label_cate_list]
        # print (label_list1)
        color_list = [color_dict[i] for i in label_list1]
        # print(color_list)
    
        for set_i in range(5):
            set_name = dcr_set_name_list[set_i]
            dcr_set = dcr_set_list[set_i]
            # print (set_name, len(dcr_set))
            data = np.array (df_DCR[dcr_set])
            # print (data.shape)
        
            X_tsne = TSNE(learning_rate=100).fit_transform(data)
            X_pca = PCA(n_components = 2).fit_transform(data)
            # print (X_tsne.shape)
            # print ('X_pca', X_pca.shape)
    
    
            df_tsne = pd.DataFrame (X_tsne,columns = ['t_SNE1','t_SNE2'])
            df_tsne = (df_tsne - df_tsne.min()) / (df_tsne.max() - df_tsne.min())
            df_tsne ['label'] = label_lst
            # print (df_tsne.head(2))
    
            df_pca = pd.DataFrame (X_pca,columns = ['PCA1','PCA2'])
            df_pca = (df_pca - df_pca.min()) / (df_pca.max() - df_pca.min())
    
            df_pca ['label'] = label_lst
            # print (df_pca.head(2))
            
            file_name0 = 'sns_scatterplot_tSNE_PCA_all_' +gene + '_' + set_name + '_sampling' + str(num) + '.png'
            file_name1 = 'df_PCA_all_' + gene + '_' + set_name + '_sampling' + str(num) + '.csv'
    
    
    
            plt.figure(figsize=(8, 4))
            plt.subplot(121)
            sns.scatterplot(data = df_tsne, x = 't_SNE2', y = 't_SNE1', markers=True, hue = 'label',palette = color_list0[:y_num],hue_order = y_types)
            plt.legend(scatterpoints=1)
            plt.subplot(122)
    
            for y_i in range(y_num):
                y_ = y_types[y_i]
                # print (y_i)
                color = color_list0[y_i]
                plt.xlim([-0.1,1.1])
                plt.ylim([-0.1,1.1])
    
                df_X_pca_label = df_pca[df_pca['label'] == y_]
                # print (df_X_pca_label.shape)
                sns.scatterplot(x = df_X_pca_label['PCA2'], y = df_X_pca_label['PCA1'], color = color) # ,x_estimator=np.mean
    
                plt.legend(scatterpoints=1)
            plt.savefig(file_name0, dpi = 300, bbox_inches = 'tight')
    #         plt.savefig('sns_regplot_tSNE_PCA_all_' +gene + '_' + set_name + '_sampling200_noFilo_noFlavi.png', dpi = 600, bbox_inches = 'tight')
    
    #         fw = open ('sns_scatter_details_' + gene + '_' + set_name + '_sampling200.txt', 'w')
    #         file_name = ('Fig details_' + gene + '_' + set_name + '_sampling200.csv')
            df_pca['accession'] = index_lst
            df_pca['subtype'] = subtypes
            df_pca.to_csv (file_name1)
    #         print ('Color dictionary: ', dict2,'\n','label_num_list: ', label_c_list,'\n', 'label_list: ', label_cate_list,'\n', file = fw)
    #         fw.close()
    #         plt.show()
    # plt.figure(figsize=(1, 3))
    # plt.scatter([1]*4, list(range(1,5,1)), c=color_list_all[:4], s = 40)
    # plt.savefig('color_list_sampling200_RdRp_Gp_all_noFilo_noFlavi.png', dpi = 200, bbox_inches = 'tight')
    
