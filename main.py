# Import packages
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

# Import functions
import FlowNetSD
import se3qua



class MyDataset:
    def __init__(self, data_dir):
        # 
        self.data_dir = data_dir

        ## Set images folder
        self.base_path_img = self.data_dir + '/cam0/data/' # EuRoC MAV
        #self.base_path_img = self.data_dir + '/cam1/' # Own dataset
        print("\nRead images from folder:", self.base_path_img)
        
        # Get all image names without image paths    
        self.data_files = os.listdir(self.data_dir + '/cam0/data/') # EuRoC MAV
        #self.data_files = os.listdir(self.data_dir + '/cam1') # Own dataset
        self.data_files.sort() # Order images by name
        print("Found {} images from the images folder.\n".format(len(self.data_files)))
        
        ## relative camera pose
        self.trajectory_relative = self.read_R6TrajFile('/vicon0/sampled_relative_R6.csv') # EuRoC MAV
        #self.trajectory_relative = self.read_R6TrajFile('/reference/newrefence-quat-10hz.csv') #  Own dataset
        
        ## abosolute camera pose (global)
        self.trajectory_abs = self.readTrajectoryFile('/vicon0/sampled.csv') # EuRoC MAV
        #self.trajectory_abs = self.readTrajectoryFile('/reference/newrefence-quat-10hz.csv') #  Own dataset

        ## imu
        self.imu = self.readIMU_File('/imu0/data.csv') # EuRoC MAV
        #self.imu = self.readIMU_File('/imu/xsens-1597237222976.csv') #  Own dataset

        # Number of imu steps in one image step
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
        # Initialize list where to read imu data
        imu_data = []
        # Open imu data file
        with open(self.data_dir + path) as csvfile:
            # Skip first row (only data headers)
            _ = csvfile.readline()
            # Read the whole imu data file with csv reader
            imu_file = csv.reader(csvfile, delimiter=',', quotechar='|')
            # Then read imu_data row by row and add to list
            for row in imu_file:
                parsed_row = [float(row[1]), float(row[2]), float(row[3]), 
                            float(row[4]), float(row[5]), float(row[6])]
                imu_data.append(parsed_row)
        # Convert list to numpy array and return results
        return np.array(imu_data)
    
    def getTrajectoryAbs(self, idx):
        return self.trajectory_abs[idx]
    
    def getTrajectoryAbsAll(self):
        return self.trajectory_abs
    
    def getIMU(self):
        return self.imu
    
    def __len__(self):
        return len(self.trajectory_relative)
    
    def load_img_bat(self, idx, batch):

        # Initialize images and imu data lists, where to append batch data
        img_data = []
        imu_data = []

        # Read the data and append to lists
        for i in range(batch):
            # Read images and resize them to 512 x 384 pixels
            img_01 = np.array(Image.open(self.base_path_img + self.data_files[idx + i]).resize((512,384)))
            img_02 = np.array(Image.open(self.base_path_img + self.data_files[idx+1 + i]).resize((512,384)))

            ## Images are gray scale, copy same channel value to all 3 channels
            img_01 = np.array([img_01, img_01, img_01])
            img_02 = np.array([img_02, img_02, img_02])

            # Concatenate both images into the same variable and add it to the images batch list
            img_concat = np.array([img_01, img_02])
            img_data.append(img_concat)

            # Read IMU data and add to the IMU data batch list
            imu_batch = np.array(self.imu[idx-self.imu_seq_len+1 + i:idx+1 + i])
            imu_data.append(imu_batch)

        # Lists to numpy array
        img_data = np.array(img_data)
        imu_data = np.array(imu_data)
        
        # Numpy arrays to tensors
        img_data_gpu = Variable(torch.from_numpy(img_data).type(torch.FloatTensor).cuda())    
        imu_data_gpu = Variable(torch.from_numpy(imu_data).type(torch.FloatTensor).cuda())    
        
        ## F2F gt
        trajectory_relative_gpu = Variable(torch.from_numpy(self.trajectory_relative[idx+1:idx+1+batch]).type(torch.FloatTensor).cuda())
        
        ## global pose gt
        trajectory_abs_gpu = Variable(torch.from_numpy(self.trajectory_abs[idx+1:idx+1+batch]).type(torch.FloatTensor).cuda())
        
        return img_data_gpu, imu_data_gpu, trajectory_relative_gpu, trajectory_abs_gpu

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
    
    
def train(dataset_base_path):

    # Set training parameters
    epoch = 1000 # Number of epochs
    batch = 1 # Does not work (yet) with bigger patch size

    # Initialize summary writer 
    writer = SummaryWriter()
    
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    mydataset = MyDataset(dataset_base_path)

    # Define loss function
    #criterion  = nn.MSELoss()
    criterion  = nn.L1Loss(reduction='mean')

    # Get the lowest loss from checkpoint. Used when searching for the new best model.
    lowest_loss = checkpoint['loss'] + 100 # Add 1 to make training happen, if you change dataset as loss is based on previous dataset
    print("\nLoss in the loaded checkpoint is:", lowest_loss)


    start = 5 # Index of first data point
    end = len(mydataset)-batch # 
    batch_num = (end - start) # Number of batches
    
   # Run training loop k number of epochs
    for k in tqdm(range(epoch)):

        # Each epoch has i number of iterations
        for i in tqdm(range(start, end), leave=False):

            # Data: img_data_gpu, imu_data_gpu, trajectory_relative_gpu, trajectory_abs_gpu
            data, data_imu, target_f2f, target_global = mydataset.load_img_bat(i, batch)

            # Do not calculate gradients, as it is not needed in testing
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
    print("\nSaved last checkpoint to file: model_checkpoints/vinet_last.pt")

    # Save tensorboard file
    writer.flush()
    writer.close()

def test(dataset_base_path):
    
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get GPU device which will be used in trimu
    # Get the model
    model = Vinet()

    # Load trained model checkpoint
    checkpoint = torch.load('model_checkpoints/vinet_last.pt') # Options: vinet_best.pt or vinet_last.pt
    model.load_state_dict(checkpoint['model_state_dict'])

    # Transfer model from CPU to GPU
    model.to(device)
    
    # Define folder where to read data
    mydataset = MyDataset(dataset_base_path)
    
    
    err = 0
    ans = []
    abs_traj = None
    start = 5

    # Start evaluation process
    for i in tqdm(range(start, len(mydataset)-10)):

        # Load data for batch i
        # Data: img_data_gpu, imu_data_gpu, trajectory_relative_gpu, trajectory_abs_gpu
        img_data, imu_data, rel_trajectory, _ = mydataset.load_img_bat(i, 1)

        if i == start:
            ## load first SE3 pose xyzQuaternion
            abs_traj = mydataset.getTrajectoryAbs(start)
            abs_traj = np.expand_dims(abs_traj, axis=0)
            abs_traj = np.expand_dims(abs_traj, axis=0)
            abs_traj = Variable(torch.from_numpy(abs_traj).type(torch.FloatTensor).cuda()) 
        
        # Run model inference
        output = model(img_data, imu_data, abs_traj)
        
        # Add error to variable ett
        err += float(((rel_trajectory - output) ** 2).mean())
        if i > 2888:
            print("\ntarget:", rel_trajectory)
            print("\output:", output)
        
        output = output.data.cpu().numpy()
        xyzq = se3qua.se3R6toxyzQ(output)
        abs_traj = abs_traj.data.cpu().numpy()[0]
        numarr = output

        abs_traj = se3qua.accu(abs_traj, numarr)
        abs_traj = np.expand_dims(abs_traj, axis=0)
        abs_traj = np.expand_dims(abs_traj, axis=0)
        abs_traj = Variable(torch.from_numpy(abs_traj).type(torch.FloatTensor).cuda()) 
        
        ans.append(xyzq)
        #print(xyzq)
        #print('{}/{}'.format(str(i+1), str(len(mydataset)-1)) )
        
    
    # Print error
    print('err = {}'.format(err/(len(mydataset)-1)))  ## Why it is NaN????
    trajectoryAbs = mydataset.getTrajectoryAbsAll()
    x = trajectoryAbs[0].astype(str)
    x = ",".join(x)
    
    # Write results to file
    with open('data/V1_01_easy/mav0/vicon0/sampled_relative_ans.csv', 'w+') as f:
        tmpStr = x
        f.write(tmpStr + '\n')
        for i in range(len(ans)-1):
            tmpStr = ans[i].astype(str)
            tmpStr = ",".join(tmpStr)
            f.write(tmpStr + '\n')
    
def main():
    # Choose if you want to do model training or testing (ADD INFERENCE OPTION!!)

    # Train options (EuRoC MAV):
    #train(dataset_base_path = 'data/V1_01_easy/mav0')
    #train(dataset_base_path = 'data/V1_02_medium/mav0')
    #train(dataset_base_path = 'data/V1_03_difficult/mav0')
    train(dataset_base_path = 'data/V2_01_easy/mav0')
    #train(dataset_base_path = 'data/V2_02_medium/mav0')
    #train(dataset_base_path = 'data/V2_03_difficult/mav0')

    # Test options (EuRoC MAV):
    #test(dataset_base_path = 'data/V1_01_easy/mav0')
    #test(dataset_base_path = 'data/V1_02_medium/mav0')
    #test(dataset_base_path = 'data/V1_03_difficult/mav0')
    #test(dataset_base_path = 'data/V2_01_easy/mav0')
    #test(dataset_base_path = 'data/V2_02_medium/mav0')
    #test(dataset_base_path = 'data/V2_03_difficult/mav0')

    # Train and test (university of Helsinki dataset)
    #train(dataset_base_path = 'data/hy-data')
    #test(dataset_base_path = 'data/hy-data')

if __name__ == '__main__':
    main()