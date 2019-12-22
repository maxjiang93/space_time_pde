import numpy as np
import matplotlib. pyplot as plt
import random
from scipy import interpolate
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data


class DataSet():
    
    def __init__(self, data_PATH="./", data_FILENAME="rb2d_ra1e6_s42.npz", Number_of_augm_samples_in_time=256):
        
        """
        
        Initialize DataSet
        args:
          @data_PATH: path to the dataset folder, default="./"
          @data_FILENAME: name of the dataset file, default="rb2d_ra1e6_s42"
          @Number_of_augm_samples_in_time: number of samples that we want to randomly query in time 
           NOTE: that two similar points in time will have different spatial information as we query 
           in space randomly as well
           
        """
        
        self.data_PATH = data_PATH
        self.data_FILENAME = data_FILENAME
        self.Number_of_augm_samples_in_time = Number_of_augm_samples_in_time
        self.lowResolutionData = None
        self.highResolutionData = None
        
        
    def getHighResData(self):
        
        """
        function for 
         - reading the high resolution data provided by the simulator
         - concatenating the pressure, temperature, x-velocity, and z-velocity as 4 channel array: pbuw
         - randomly creating an augmented dataset in space by concatenating two randomly chosen consecutive time steps
         - returning: high resolution dataset as "d_ts01_high_res_data" with shape of
         -- Number_of_augm_samples_in_time*(4+4)*ny*ny where ny = min(nx,ny)
        """
    
        d = np.load(self.data_PATH + self.data_FILENAME)

        # concatenating pressure, temperature, x-velocity, and z-velocity as a 4 channel array: pbuw
        # shape: (200,4,512,128) 
        data = np.stack([d['p'],d['b'],d['u'],d['w']], axis = 1) 

        # nt = 200 - no. time steps - obtained from the simulator
        # nc = 4   - no. channels - pbuw
        # nx = 512 - no. x-driection grids - obtained from the simulator
        # ny = 128 - no. y-driection grids - obtained from the simulator
        [nt, nc, nx, ny] = data.shape

        # random starting points in x-direction for choosing a square of size ny*ny - ny = min(nx,ny)
        rand_pos_x = random.randint(0,nx-ny-1)

        #N_r: number of random samples in time to get selected for data augmentation in space
        N_r = self.Number_of_augm_samples_in_time
        #t_s0: list of randomly chosen 0th time step indices
        t_s0 = np.random.choice(nt-1, N_r, replace=True) 

        # concatenating data for data from two consecutive time steps as 
        # d_ts0(@t=t_s0), d_ts1(@t=t_s1) where t_s1=t_s0+1
        # shape: N_r*8*ny*ny
        d_ts0  = data[t_s0  , :, rand_pos_x:rand_pos_x+ny, :]
        d_ts1  = data[t_s0+1, :, rand_pos_x:rand_pos_x+ny, :]
        d_ts01_high_res_data = np.concatenate([d_ts0, d_ts1], axis = 1)

        self.highResolutionData = d_ts01_high_res_data



    def down_res(self, arr, factor=2, kind_="cubic"):
        
        """
        function for 
         - converting an image represented by "arr" to lower resolution by a factor specified as "factor"
         - using 2d interpolation, where kind_="cubic is chosen as a default interpolation scheme
         - kind_ in : {‘linear’, ‘cubic’, ‘quintic’}
        """
    
        H, W = arr.shape
        new_H, new_W = (int(H/factor),int(W/factor))

        f = interpolate.interp2d(np.linspace(0, 1, H), np.linspace(0, 1, W), arr, kind=kind_)
        new_arr = f(np.linspace(0, 1, new_H),np.linspace(0, 1, new_W))

        return new_arr


    def createLowResData(self, down_res_factor = 2):
        
        """
        function for 
         - creating a low resolution data, from the high resolution data read from the simulator
         - down_res_factor=2 is the factor by which we upscale(lower the resolution) of highRes data
        """
        
        d_ts01_low_res = np.zeros((self.highResolutionData.shape[0],8,
                                   self.highResolutionData.shape[2]//down_res_factor,
                                   self.highResolutionData.shape[3]//down_res_factor))
        for t in range(self.highResolutionData.shape[0]):
            for i in range(8):
                d_ts01_low_res[t,i,:,:] = down_res(self.highResolutionData[t,i,:,:], 
                                                   factor=down_res_factor, kind_="linear") 

        self.lowResolutionData = d_ts01_low_res
    
    
    def plot_lowRes_HighRes_Data_for_check(self):
        
        """
        function for 
         - plotting low and high res data at two consecutive times steps at a randomly chosen data sample
        """
        
        d_ts01_high_res = self.highResolutionData
        d_ts01_low_res = self.lowResolutionData

        N_r = highResData.shape[0]
        t = random.randint(0,N_r-1)

        fig = plt.figure(figsize=(16,12))
        subtitle = ["pressure", "temperature", "x-velocity", "z-velocity"]
        for i in range(8):
            plt.subplot(4, 4, i+1)
            if i>3: plt.title(subtitle[i%4] + " high res")
            plt.imshow(np.transpose(d_ts01_high_res[t,i,:,:]),cmap="gist_rainbow")
            plt.axis('off')

        j = 0
        for i in range(8,16):
            plt.subplot(4, 4, i+1)
            if j>3: plt.title(subtitle[j%4] + " low res")
            plt.imshow(np.transpose(d_ts01_low_res[t,j,:,:]),cmap="gist_rainbow")
            j+=1
            plt.axis('off')
            
        plt.savefig('HighResAndLowResData_SamplePlot.png')
            
        
        
class DataLoaderDeepRes():
    
    """
        data loader class that desgined to:
        pass a tuple of (low_res_images and high_res_images) to torch.utils.data.DataLoader
    """
    
    def __init__(self, lowResolutionData, highResolutionData):
        super(DataLoaderDeepRes, self).__init__()
        
        self.high_res_images = highResolutionData
        self.low_res_images = lowResolutionData
        

    def __getitem__(self, index):
            return (self.low_res_images[index], self.high_res_images[index])

    def __len__(self):
        return len(self.high_res_images)

    
    
if __name__ == "__main__":
    
    """
    Testing the 
    high resolution dataset preparation, 
    low resolution image creation and dataset preparation, 
    and creating batches by dataLoader.
    """

    ### gettting lowRes and higRes data
    dataset = DataSet()
    dataset.getHighResData()
    dataset.createLowResData()
    
    lowResData = dataset.lowResolutionData
    highResData = dataset.highResolutionData
    
    # dataset.plot_lowRes_HighRes_Data_for_check()
    
    ### example for using the data loader
    data_loader = DataLoaderDeepRes(dataset.lowResolutionData,dataset.highResolutionData)
    data_batches = torch.utils.data.DataLoader(data_loader, batch_size=16, shuffle=True, num_workers=1)

    for batch_idx, (lowres_input_batch, highres_output_batch) in enumerate(data_batches):
        print("Reading batch #{}:\t with lowres inputs of size {} and highres outputs of size {}"
              .format(batch_idx+1, list(lowres_input_batch.shape),  list(highres_output_batch.shape)))
