import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import torch.utils.data as Data
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from models import *
from model_hess import *
from datasets import *
from utils import *
import torch
from tqdm import tqdm
from torchsummary import summary



class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        """
        Args:
            patience (int): 容忍验证损失不改善的次数.
            min_delta (float): 视为改善的最小变化.
            save_path (str): 最优模型的保存路径.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None  # 用于保存最好的模型

    def __call__(self, validation_loss, model,dataset_name, epoch):
        score = -validation_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model,dataset_name, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model,dataset_name, epoch)

    def save_checkpoint(self, model,dataset_name, epoch):
        """保存当前最好的模型"""
        print(f"Validation loss improved.")
        os.makedirs("../saved_models/%s/best" % dataset_name, exist_ok=True)
        torch.save(model.state_dict(), "../saved_models/%s/best/G_AB_%d.pth" % (dataset_name, epoch))









def main(epoch,n_epochs,data_split,dataset_name,batch_size,lr,b1,b2,decay_epoch,patience,img_height,img_width,channels,sample_interval
         ,checkpoint_interval,n_residual_blocks,lambda_cyc,lambda_id):
    # Create sample and checkpoint directories

    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    os.makedirs("../result/%s" % dataset_name, exist_ok=True)
    os.makedirs("../saved_models/%s" % dataset_name, exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()

    input_shape = (channels, img_height, img_width)

    # Initialize generator and discriminator
    # G_AB = GeneratorResNet(input_shape, n_residual_blocks)
    # G_BA = GeneratorResNet(input_shape,  n_residual_blocks)
    # D_A = Discriminator(input_shape)
    # D_B = Discriminator(input_shape)




    # Initialize generator and discriminator(hess)
    G_AB = Generator_hess()
    G_BA = Generator_hess()
    D_A = Discriminator_hess()
    D_B = Discriminator_hess()



    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("../saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch)))
        G_BA.load_state_dict(torch.load("../saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch)))
        D_A.load_state_dict(torch.load("../saved_models/%s/D_A_%d.pth" % (dataset_name, epoch)))
        D_B.load_state_dict(torch.load("../saved_models/%s/D_B_%d.pth" % (dataset_name, epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
    summary(G_AB, input_size=(1, 72, 144))
    # Optimizers
    # lr = np.array(0.2, dtype=np.float16).astype(np.float16)
    # b1 = np.array(0.5, dtype=np.float16).astype(np.float16)
    # b2 = np.array(0.999, dtype=np.float16).astype(np.float16)
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Load data
    mpi = np.load('../data/rh_mpi_1950_2014.npy')
    necp = np.load('../data/rh_necp_1950_2014.npy')

    date1 = -data_split[0]
    date2 = -data_split[1]

    # 1950-2000
    mpi_train = mpi[:date1]
    necp_train = necp[:date1]

    # 2001-2004
    mpi_val = mpi[date1:date2]
    necp_val = necp[date1:date2]

    # 2005 - 2014
    mpi_test = mpi[date2:]
    necp_test = necp[date2:]



    mpi_train=torch.tensor(mpi_train).unsqueeze(1).cuda()
    necp_train = torch.tensor(necp_train).unsqueeze(1).cuda()

    mpi_val = torch.tensor(mpi_val).unsqueeze(1).cuda()
    necp_val = torch.tensor(necp_val).unsqueeze(1).cuda()

    mpi_test = torch.tensor(mpi_test).unsqueeze(1).cuda()
    necp_test = torch.tensor(necp_test).unsqueeze(1).cuda()


    dataset=Data.TensorDataset(mpi_train,necp_train )
    dataset_val = Data.TensorDataset(mpi_val, necp_val)
    dataset_test = Data.TensorDataset(mpi_test, necp_test)
    # Training data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    # Test data loader

    # Val data loader
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


    test_dataloader = DataLoader(
        dataset_test,
        batch_size=5,
        shuffle=True,
        num_workers=0,
    )
    print('load data end')
    def sample_images(batches_done,num):
        """Saves a generated sample from the test set"""
        imgs = next(iter(test_dataloader))


        G_AB.eval()
        G_BA.eval()
        real_A = imgs[0]
        fake_B = G_AB(real_A)
        real_B = imgs[1]
        fake_A = G_BA(real_B)
        recover_B = G_BA(fake_B)



        # Arange images along x-axis
        real_A = make_grid(torch.flipud(real_A), nrow=5, normalize=True)
        real_B = make_grid(torch.flipud(real_B), nrow=5, normalize=True)
        fake_A = make_grid(torch.flipud(fake_A), nrow=5, normalize=True)
        fake_B = make_grid(torch.flipud(fake_B), nrow=5, normalize=True)
        recover_B = make_grid(torch.flipud(recover_B), nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B,recover_B, real_B, fake_A), 1)
        save_image(image_grid, "../result/%s/%s.png" % (dataset_name, batches_done), normalize=False)

    # ----------
    #  Training
    # ----------
    prev_time = time.time()
    for epoch in range(epoch, n_epochs):
        for i, (batch_A,batch_B) in tqdm(enumerate(dataloader)):
            print(i)

            # Set model input
            real_A = batch_A
            real_B = batch_B

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), 1,7,16))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), 1,7,16))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(real_A,G_BA(real_A))
            loss_id_B = criterion_identity(real_B,G_AB(real_B))

            loss_identity = (loss_id_A + loss_id_B) / 2


            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(real_A,recov_A )
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(real_B,recov_B )

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            #Total loss
            loss_GAN_AB=loss_GAN_AB+(lambda_id * loss_identity)+(lambda_cyc * loss_cycle)
            loss_GAN_BA=loss_GAN_AB+(lambda_id * loss_identity)+(lambda_cyc * loss_cycle)
            loss_G = loss_GAN_AB + loss_GAN_BA + (lambda_cyc * loss_cycle)


            # loss_G = loss_GAN_AB + loss_GAN_BA + (lambda_cyc * loss_cycle)


            loss_G.backward()
            optimizer_G.step()

            (G_AB,real_B)

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d]  [G loss: %f, adv: %f, cycle: %f, identity: %f,loss_cycle_A: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    loss_cycle_A.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done,i)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        torch.save(G_AB.state_dict(), "../saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch))
        torch.save(G_BA.state_dict(), "../saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch))
        torch.save(D_A.state_dict(), "../saved_models/%s/D_A_%d.pth" % (dataset_name, epoch))
        torch.save(D_B.state_dict(), "../saved_models/%s/D_B_%d.pth" % (dataset_name, epoch))

        total_loss = 0
        G_AB.eval()
        with torch.no_grad():
            for i, (batch_A_val, batch_B_val) in tqdm(enumerate(val_dataloader)):
                real_A = batch_A_val
                real_B = batch_B_val
                loss = criterion_GAN(real_B, G_AB(real_A))
                total_loss = total_loss + loss.item() * real_A.size(0)

        total_loss /= len(dataset_val)
        early_stopping(total_loss, G_AB, dataset_name, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break


        # if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        #     # Save model checkpoints
        #     torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch))
        #     torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch))
        #     torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (dataset_name, epoch))
        #     torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (dataset_name, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--data_split", type=int, default=[5113,3652], help="split the dataset")
    parser.add_argument("--dataset_name", type=str, default="mpi_necp_hess_1950_2014_p20_2", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=2, help="epoch from which to start lr decay")
    parser.add_argument("--patience", type=int, default=20, help="early stop The number of training sessions required to stop training")
    parser.add_argument("--img_height", type=int, default=72, help="size of image height")
    parser.add_argument("--img_width", type=int, default=144, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=1, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=0.5, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=1, help="identity loss weight")
    opt = parser.parse_args()
    print(opt)
    main(opt.epoch,opt.n_epochs,opt.data_split,opt.dataset_name,opt.batch_size,opt.lr,opt.b1,opt.b2,opt.decay_epoch,opt.patience,opt.img_height,opt.img_width,opt.channels,opt.sample_interval
         ,opt.checkpoint_interval,opt.n_residual_blocks,opt.lambda_cyc,opt.lambda_id)