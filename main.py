import os
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm

import FlowNetSD
#from utils import se3qua
import se3qua



class MyDataset:
    def __init__(self, data_dir):
        # 
        self.data_dir = data_dir

        ## Set images folder
        self.base_path_img = self.data_dir + '/cam0/data/'
        print("\nRead images from folder:", self.base_path_img)
        
        # Get all image names without image paths    
        self.data_files = os.listdir(self.data_dir + '/cam0/data/')
        self.data_files.sort()
        print("Found {} images from the images folder.\n".format(len(self.data_files)))
        
        ## relative camera pose
        self.trajectory_relative = self.read_R6TrajFile('/vicon0/sampled_relative_R6.csv')
        
        ## abosolute camera pose (global)
        self.trajectory_abs = self.readTrajectoryFile('/vicon0/sampled.csv')

        ## imu
        self.imu = self.readIMU_File('/imu0/data.csv')
        
        self.imu_seq_len = 5
    
    def readTrajectoryFile(self, path):
        traj = []
        with open(self.data_dir + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                traj.append(parsed)
                
        return np.array(traj)
    
    def read_R6TrajFile(self, path):
        traj = []
        with open(self.data_dir + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6])]
                traj.append(parsed)
                
        return np.array(traj)
    
    def readIMU_File(self, path):
        imu = []
        count = 0
        with open(self.data_dir + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if count == 0:
                    count += 1
                    continue
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6])]
                imu.append(parsed)
                
        return np.array(imu)
    
    def getTrajectoryAbs(self, idx):
        return self.trajectory_abs[idx]
    
    def getTrajectoryAbsAll(self):
        return self.trajectory_abs
    
    def getIMU(self):
        return self.imu
    
    def __len__(self):
        return len(self.trajectory_relative)
    
    def load_img_bat(self, idx, batch):
        batch_x = []
        batch_imu = []
        for i in range(batch):
            # Read images and resize them to 512 x 384 pixels
            x_data_np_1 = np.array(Image.open(self.base_path_img + self.data_files[idx + i]).resize((512,384)))
            x_data_np_2 = np.array(Image.open(self.base_path_img + self.data_files[idx+1 + i]).resize((512,384)))

            ## Images are gray scale, copy same channel value to all 3 channels
            x_data_np_1 = np.array([x_data_np_1, x_data_np_1, x_data_np_1])
            x_data_np_2 = np.array([x_data_np_2, x_data_np_2, x_data_np_2])

            # Concatenate both images into the same variable and add it to the images batch list
            X = np.array([x_data_np_1, x_data_np_2])
            batch_x.append(X)

            # Read IMU data and add to the IMU data batch list
            tmp = np.array(self.imu[idx-self.imu_seq_len+1 + i:idx+1 + i])
            batch_imu.append(tmp)
            

        batch_x = np.array(batch_x)
        batch_imu = np.array(batch_imu)
        
        X = Variable(torch.from_numpy(batch_x).type(torch.FloatTensor).cuda())    
        X2 = Variable(torch.from_numpy(batch_imu).type(torch.FloatTensor).cuda())    
        
        ## F2F gt
        Y = Variable(torch.from_numpy(self.trajectory_relative[idx+1:idx+1+batch]).type(torch.FloatTensor).cuda())
        
        ## global pose gt
        Y2 = Variable(torch.from_numpy(self.trajectory_abs[idx+1:idx+1+batch]).type(torch.FloatTensor).cuda())
        
        return X, X2, Y, Y2

class Vinet(nn.Module):
    def __init__(self):
        super(Vinet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=49165,#49152,#24576, 
            hidden_size=1024,#64, 
            num_layers=2,
            batch_first=True)
        self.rnn.cuda()
        
        self.rnnIMU = nn.LSTM(
            input_size=6, 
            hidden_size=6,
            num_layers=2,
            batch_first=True)
        self.rnnIMU.cuda()
        
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, 6)
        #self.linear3 = nn.Linear(128, 6)
        self.linear1.cuda()
        self.linear2.cuda()
        #self.linear3.cuda()
        
        self.flownet_sd = FlowNetSD.FlowNetSD(batchNorm=False)
        self.flownet_sd.cuda()

    def forward(self, image, imu, xyzQ):
        batch_size, timesteps, C, H, W = image.size()
        
        ## Input1: Feed image pairs to FlownetC
        c_in = image.view(batch_size, timesteps * C, H, W)
        c_out = self.flownet_sd(c_in)
        #r_in = c_out.view(batch_size, timesteps, -1)
        r_in = c_out.view(batch_size, 1, -1)
        
        ## Input2: Feed IMU records to small LSTM
        imu_out, (imu_n, imu_c) = self.rnnIMU(imu)
        imu_out = imu_out[:, -1, :]
        imu_out = imu_out.unsqueeze(1)
        
        ## Combine the output of Flownet and IMU LSTM and xyzQ
        cat_out = torch.cat((r_in, imu_out), 2)#1 1 49158
        cat_out = torch.cat((cat_out, xyzQ), 2)#1 1 49165
        
        ## Run main LSTM and flatten output
        r_out, (h_n, h_c) = self.rnn(cat_out)
        l_out1 = self.linear1(r_out[:,-1,:])
        l_out2 = self.linear2(l_out1)
        #l_out3 = self.linear3(l_out2)
        #print('\nFinal output of Deep learning network:', l_out2.shape)

        return l_out2
    
    
def train():

    # Set training parameters
    epoch = 1
    batch = 1 # Does not work (yet) with bigger patch size
    
    # Initialize summary writer 
    writer = SummaryWriter()
    
    # Get GPU device which will be used in training
    device = torch.device("cuda")

    # Get the model
    model = Vinet()

    # Load trained model checkpoint
    checkpoint = torch.load('model_checkpoints/vinet_last.pt') # Options: vinet_best.pt or vinet_last.pt
    model.load_state_dict(checkpoint['model_state_dict'])

    # Transfer model from CPU to GPU
    model.to(device)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # Set model to training state
    model.train()

    # Path to where to read data in training process
    mydataset = MyDataset('data/V1_03_difficult/mav0')

    # Define loss function
    #criterion  = nn.MSELoss()
    criterion  = nn.L1Loss(size_average=False)
    

    start = 5
    end = len(mydataset)-batch
    batch_num = (end - start) #/ batch
   

    # Get the lowest loss from checkpoint. Used when searching for the new best model.
    lowest_loss = checkpoint['loss'] + 100 # Add 1 to make training happen, if you change dataset as loss is based on previous dataset
    print("\nLoss in the loaded checkpoint is:", lowest_loss)
    
   # Run training loop
    for k in tqdm(range(epoch)):
        for i in tqdm(range(start, end), leave=False):
            data, data_imu, target_f2f, target_global = mydataset.load_img_bat(i, batch)
            data, data_imu, target_f2f, target_global = \
                data.cuda(), data_imu.cuda(), target_f2f.cuda(), target_global.cuda()

            optimizer.zero_grad()
            
            if i == start:
                ## load first SE3 pose xyzQuaternion
                abs_traj = mydataset.getTrajectoryAbs(start)
                abs_traj_input = np.expand_dims(abs_traj, axis=0)
                abs_traj_input = np.expand_dims(abs_traj_input, axis=0)
                abs_traj_input = Variable(torch.from_numpy(abs_traj_input).type(torch.FloatTensor).cuda())

            ## Forward
            output = model(data, data_imu, abs_traj_input)
            
            ## Accumulate pose
            numarr = output.data.cpu().numpy()
            abs_traj = se3qua.accu(abs_traj, numarr)
            abs_traj_input = np.expand_dims(abs_traj, axis=0)
            abs_traj_input = np.expand_dims(abs_traj_input, axis=0)
            abs_traj_input = Variable(torch.from_numpy(abs_traj_input).type(torch.FloatTensor).cuda()) 


            ## (F2F loss) + (Global pose loss)
            ## Global pose: Full concatenated pose relative to the start of the sequence
            loss = criterion(output, target_f2f) + criterion(abs_traj_input, target_global)

            loss.backward()
            optimizer.step()

            # Loss from tensor to float
            loss_float = loss.item()
            
            # Save loss to the tensorboard file
            writer.add_scalar('Loss/train', loss_float, k*batch_num + i)

            # Check if loss in lower than ever before
            # But require that iteration is larger than start+50, as loss is small at the beginning
            if (loss_float <= lowest_loss) and (i > 100): 
                lowest_loss = loss_float
                #print('New lowest loss found, it is:', lowest_loss)
                torch.save({
                    'epoch': k,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_float,
                    }, 'model_checkpoints/vinet_best.pt')
    
    # Save also the last checkpoint
    torch.save({
            'epoch': k,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_float,
            }, 'model_checkpoints/vinet_last.pt')

    # Save tensorboard file
    writer.flush()
    writer.close()

def test():
    checkpoint_pytorch = '/notebooks/vinet/vinet_v1_01.pt'
    if os.path.isfile(checkpoint_pytorch):
        checkpoint = torch.load(checkpoint_pytorch,\
                            map_location=lambda storage, loc: storage.cuda(0))
        #best_err = checkpoint['best_EPE']
    else:
        print('No checkpoint')
    

    model = Vinet()
    model.load_state_dict(checkpoint)  
    model.cuda()
    model.eval()
    mydataset = MyDataset('/notebooks/EuRoC_modify/', 'V2_03_difficult')
    
    
    err = 0
    ans = []
    abs_traj = None
    start = 5
    #for i in range(len(mydataset)-1):
    for i in range(start, 100):
        data, data_imu, target, target2 = mydataset.load_img_bat(i, 1)
        data, data_imu, target, target2 = data.cuda(), data_imu.cuda(), target.cuda(), target2.cuda()

        if i == start:
            ## load first SE3 pose xyzQuaternion
            abs_traj = mydataset.getTrajectoryAbs(start)
            abs_traj = np.expand_dims(abs_traj, axis=0)
            abs_traj = np.expand_dims(abs_traj, axis=0)
            abs_traj = Variable(torch.from_numpy(abs_traj).type(torch.FloatTensor).cuda()) 
                    
        output = model(data, data_imu, abs_traj)
        
        err += float(((target - output) ** 2).mean())
        
        output = output.data.cpu().numpy()

        xyzq = se3qua.se3R6toxyzQ(output)
                
        abs_traj = abs_traj.data.cpu().numpy()[0]
        numarr = output
        
        abs_traj = se3qua.accu(abs_traj, numarr)
        abs_traj = np.expand_dims(abs_traj, axis=0)
        abs_traj = np.expand_dims(abs_traj, axis=0)
        abs_traj = Variable(torch.from_numpy(abs_traj).type(torch.FloatTensor).cuda()) 
        
        ans.append(xyzq)
        print(xyzq)
        print('{}/{}'.format(str(i+1), str(len(mydataset)-1)) )
        
        
    print('err = {}'.format(err/(len(mydataset)-1)))  
    trajectoryAbs = mydataset.getTrajectoryAbsAll()
    print(trajectoryAbs[0])
    x = trajectoryAbs[0].astype(str)
    x = ",".join(x)
    
    with open('/notebooks/EuRoC_modify/V2_01_easy/vicon0/sampled_relative_ans.csv', 'w+') as f:
        tmpStr = x
        f.write(tmpStr + '\n')        
        
        for i in range(len(ans)-1):
            tmpStr = ans[i].astype(str)
            tmpStr = ",".join(tmpStr)
            print(tmpStr)
            print(type(tmpStr))
            f.write(tmpStr + '\n')      
    
def main():
    train()
          
    #test()

    
        

if __name__ == '__main__':
    main()
