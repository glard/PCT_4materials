import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

#def download():
#    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#    DATA_DIR = os.path.join(BASE_DIR, 'data')
#    if not os.path.exists(DATA_DIR):
#        os.mkdir(DATA_DIR)
#    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#        zipfile = os.path.basename(www)
#        os.system('wget %s; unzip %s' % (www, zipfile))
#        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#        os.system('rm %s' % (zipfile))

#def load_data(partition):
#    download()
#    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#    DATA_DIR = os.path.join(BASE_DIR, 'data')
#    all_data = []
#    all_label = []
#    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
#        f = h5py.File(h5_name)
#        data = f['data'][:].astype('float32')
#        label = f['label'][:].astype('int64')
#        f.close()
#        all_data.append(data)
#        all_label.append(label)
#    all_data = np.concatenate(all_data, axis=0)
#    all_label = np.concatenate(all_label, axis=0)
#    return all_data, all_label


def load_bandgap_data(data_dic, partition, num_points):
    """
    retur all_data and all_label for PCT framework
    """
    df_whole = pd.read_csv(data_dic + 'band_gap67039.csv')
    df = pd.read_csv(data_dic + 'band_gap' + str(num_points) + '.csv')
    print("original df rows: "+ str(len(df)))
    # shuffle df with seed 456 for reproducable experiments
    df = df.sample(frac=1, random_state=456)
    # df = df.reset_index()
    # df = df.round({'band_gap': 4})
    # print("after round df rows: "+ str(len(df)))
    #assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    ind_to_keep = ~df['band_gap'].isin([np.nan, np.inf, -np.inf])
    df = df[ind_to_keep]

    ind_tk2 = df['mp-id'].isin(df_whole['mp-id'].tolist())
    df = df[ind_tk2]

    #df = df[df['band_gap'] > 0.01]
    print("after dropna, inf df rows and some unavailble data: "+ str(len(df)))
    
    # 6 2 2 split for training, validation and test dataset
    train, val, test = np.split(df, [int(.6*len(df)), int(.8*len(df))])

    #all_data = []
    #all_label = []
    #for id, label in zip(df['mp-id'], df['band_gap']):
    #    latent_vec = np.load(data_dic + str(id) + '.npy')

    #    all_data.append(latent_vec)
    #    all_label.append(label)

    #     print(all_label)
    #all_data = np.array(all_data).astype(np.float32)
    #all_label = np.array(all_label).astype(np.float32)
   


    data = []
    label = []
    if partition == 'train':
        for i, l in zip(train['mp-id'], train['band_gap']):
            latent_vec = np.load(data_dic + str(i) + '.npy')
            data.append(latent_vec)
            label.append(l)
        #data = all_data[int(num_points/5):]
        #label = all_label[int(num_points/5):]
    elif partition == 'test':
        for i, l in zip(test['mp-id'], test['band_gap']):
            latent_vec = np.load(data_dic + str(i) + '.npy')
            data.append(latent_vec)
            label.append(l)
        #data = all_data[:int(num_points/5)]
        #label = all_label[:int(num_points/5)]
    elif partition == 'validation':
        for i, l in zip(val['mp-id'], val['band_gap']):
            latent_vec = np.load(data_dic + str(i) + '.npy')
            data.append(latent_vec)
            label.append(l)
        #data = all_data[:int(num_points/5)]
        #label = all_label[:int(num_points/5)]

    return np.array(data).astype(np.float32), np.array(label).astype(np.float32)


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    # revised for bandgap prediction
    # xyz1 = np.random.uniform(low=2./3., high=3./2., size=[64])
    # xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[64])
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[203])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[203])
    # print(f"pointcloud shape is{pointcloud.shape}")
    # print(f"xyz1 is{xyz1.shape}")
    # print(f"xyz2 is{xyz2.shape}")
    # exit(0)
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')

    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        #self.data, self.label = load_data(partition)
        self.data, self.label = load_bandgap_data('data/processed_data/', partition, num_points)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
