
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os



nt_table = ['t','c', 'a', 'g']
dnt_table = [nt1+nt2 for nt1 in nt_table for nt2 in nt_table]
dnts_table = [dnt1+dnt2 for dnt1 in dnt_table for dnt2 in dnt_table]
dntpair_category = ['n12m12', 'n23m23','n31m31','n12n31','n23m12','n31m23']
dntpair_cols_list = ['Freq_'+ dnts + '_' + dnts_cat for dnts_cat in dntpair_category for dnts in dnts_table]

model_name = './DCR_CNN_model_WNV_avian(1)_mosquito(0).txt'

class CNN (nn.Module):
    def __init__ (self):
        super (CNN, self).__init__()
        self.conv1 = nn.Sequential ( 
            nn.Conv3d ( in_channels = 1,
                        out_channels = 8,
                        kernel_size = (1, 3, 3), # kernel only for 2d data
                        stride =(1,1,1),
                        padding = (0,1,1),
                        bias = True
                        ),            # 
            nn.ReLU (),
            nn.AvgPool3d (kernel_size = (1,2,2)) 
        )
        self.conv2 = nn.Sequential ( 
            nn.Conv3d ( in_channels = 8,
                        out_channels = 16,
                        kernel_size = (1, 3, 3),# kernel only for 2d data
                        stride =(1,1,1),
                        padding = (0,1,1),
                        bias = True
                        ),            # 
            nn.ReLU (),
            nn.AvgPool3d (kernel_size = (1, 2, 2)) # Max or Avg
        )
        self.conv3 = nn.Sequential (
            nn.Conv3d ( in_channels = 16,
                        out_channels = 32,
                        kernel_size = (1, 3, 3),# kernel only for 2d data
                        stride =(1,1,1),
                        bias = True,
                        padding = (0,1,1)
                        ),
            nn.ReLU (),
            nn.AvgPool3d (kernel_size = (1, 2, 2)) #MaxPool3d
        )
        self.fc1 = nn.Linear (768, 192)  # adding this step, too slow
        self.fc2 = nn.Linear (192, 2)  # adding this step, too slow
        # self.fc3 = nn.Linear (48, 2)

    def forward (self, x):  # x is the a matrix, however, only lowcase is suggeted to use
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view (x.size(0), -1) # flat x, similar to reshape of numpy
        fc_full = x
        # print ('before sigmoid:', x.shape)
        x = self.fc1(x)
        # x = self.fc2(x)
        pred_ = self.fc2(x)
        prob_ = F.softmax (self.fc2(x))
        # x = F.sigmoid (self.fc3(x))        # to activate x
        # print ('after sigmoid:', x.shape)
        # print (prob_)
        return pred_, prob_, fc_full # also to output prob_ 

    
# cnn = CNN()
cnn = torch.load(model_name)
# device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")
# cnn.to(device)
# print (cnn)
if torch.cuda.is_available():
    cnn = cnn.cuda()
    cnn.to(torch.device("cuda:0"))


path = '../counting/'
for file in os.listdir(path):
    if file.startswith ('df_full_counting')&file.endswith('.csv'):
        print (file)
        df_DCR = pd.read_csv (path + file)
        print (df_DCR.shape)

        accession_list = df_DCR.loc[:,'accession'].tolist()
        df_DCR = df_DCR.loc[:,dntpair_cols_list]
        print (df_DCR.shape)
        DCR_array = np.array(df_DCR)
        num = DCR_array.shape[0]
        DCR_array2 = DCR_array.reshape(num,1,6,16,16)
        print (DCR_array2.shape)
        DCR_tensor0 = torch.tensor(DCR_array2)
        DCR_tensor = DCR_tensor0.to(torch.float32)
        print (DCR_tensor.shape)

        data = Variable(DCR_tensor).cuda()
        pred0 = cnn (data)
        pred = pred0[0].cpu()
        prob_array = pred0[1].cpu().data.numpy()
        print (prob_array.shape)
        pred_labels = torch.max (pred, 1)[1].data
        
        fc_array = pred0[2].cpu().data.numpy()
        

        pred_results = pd.DataFrame ({'Accession': accession_list,'pred_label': pred_labels})

        pred_results [['Score_0','Score_1']] = prob_array
        print (pred_results.shape)
        file_name = 'df_Pred_3dCNN_' + file[3:-4] + '.csv'
        pred_results.to_csv (file_name, index = False)
        
        df_fc = pd.DataFrame ({'Accession': accession_list})
        df_fc [list(range(fc_array.shape[1]))] = fc_array

        file_name2 = 'df_fc_data_3dCNN_' + file[3:-4] + '.csv'
        
        df_fc.to_csv (file_name2)
        
        