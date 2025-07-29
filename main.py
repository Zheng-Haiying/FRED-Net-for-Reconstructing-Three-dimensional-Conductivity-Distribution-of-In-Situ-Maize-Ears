import argparse
import torch
import random
import torch.backends.cudnn as cudnn
from Networks.resnet50 import ResNet50
from Networks.unet import UNet
from Networks.resunet import ResUNet
from Networks.ds_lstm import DoubleStageLSTM
from 上传至github.networks.ea_resnet import EAResNet50
from torch.utils.data import Dataset, DataLoader
import h5py
import datetime
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm
import numpy
from scipy.signal import convolve


def set_device(args):
    """Set computing device (CPU or GPU)"""
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu} for training; Device name: {torch.cuda.get_device_name(args.gpu)}")
        return torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        print(f"Using GPU; Device count: {torch.cuda.device_count()}; Device name: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")
    else:
        print('Using CPU')
        return torch.device("cpu")


def set_random_seed(args):
    """Set random seed for reproducibility"""
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True


class H5DatasetWithIndex(Dataset):
    """Dataset for loading HDF5 data, returns index and corresponding sample"""

    def __init__(self, file_path, transform=None):
        with h5py.File(file_path, 'r') as file:
            # Auto-detect dataset
            for key in file.keys():
                self.data = list(file[key])
                break
            self.indexes = list(range(len(self.data)))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        index = self.indexes[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, index


class PairedDataset(Dataset):
    """Handle two datasets, return samples and corresponding indices"""

    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1, idx1 = self.dataset1[index]
        x2, idx2 = self.dataset2[index]
        return x1, x2, idx1

    def __len__(self):
        return len(self.dataset1)


def split_dataset(dataset, ratio):
    """Split dataset into training and validation sets"""
    size = len(dataset)
    train_size = int(size * ratio)
    train_samples = [dataset[i] for i in range(train_size)]
    val_samples = [dataset[i] for i in range(train_size, size)]
    return train_samples, val_samples


def save_losses_to_csv_and_plot(all_train_losses, all_val_losses, save_time, directory='./output/save_loss/',
                                filename_prefix='losses'):
    """Save losses to CSV and create loss plot"""
    csv_filepath = os.path.join(directory, f'{filename_prefix}_{save_time}.csv')
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

    with open(csv_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        for epoch in range(len(all_train_losses)):
            train_loss = all_train_losses[epoch]
            val_loss = all_val_losses[epoch]
            writer.writerow([epoch + 1, train_loss, val_loss])

    # Create loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(all_train_losses, label='Training Loss')
    plt.plot(all_val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plot_filepath = os.path.join(directory, f'{filename_prefix}_loss_plot_{save_time}.png')
    os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
    plt.savefig(plot_filepath)
    plt.close()
    return csv_filepath, plot_filepath


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def early_stopping(args, val_loss_avg, best_loss, epoch, early_stop_counter, patience):
    """Early stopping logic"""
    if args.early_stop:
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch} due to no improvement in validation loss.')
            return True, best_loss, early_stop_counter
    return False, best_loss, early_stop_counter


def moving_std(x, window_size):
    """Calculate moving standard deviation"""
    mean = convolve(x, numpy.ones(window_size, dtype=int), 'valid') / window_size
    std = numpy.sqrt(convolve(x ** 2, numpy.ones(window_size, dtype=int), 'valid') / window_size - mean ** 2)
    return numpy.concatenate([numpy.full(window_size - 1, 0), std])


def train(args, epoch, model, criterion, optimizer, train_loader, device, train_losses, multiple, losses=None):
    """Training loop"""
    model.train()
    window_size = 45
    if losses is None:
        losses = AverageMeter()

    with tqdm(total=len(train_loader.dataset), desc=f'Train[{epoch}/{args.epochs}]') as pbar:
        for i, (BV_samples, DDL_samples, idx) in enumerate(train_loader):
            # Prepare input based on channel configuration
            if args.set_channel == 1:
                input_sample = BV_samples.to(device).float().unsqueeze(1)
            elif args.set_channel == 3:
                input_data = BV_samples.to(device).float()
                # FFT for magnitude spectrum
                fft_results = torch.fft.fft(input_data)
                magnitude_spectrum = torch.abs(fft_results)
                # Calculate volatility
                input_data_np = input_data.cpu().numpy()
                volatility = numpy.array([moving_std(x, window_size)[None, :] for x in input_data_np])
                volatility = torch.from_numpy(volatility).float().to(device)
                # Combine three channels
                combined_data = torch.cat((input_data.unsqueeze(1), magnitude_spectrum.unsqueeze(1), volatility), dim=1)
                input_sample = combined_data

            label_sample = DDL_samples.to(device).float()
            optimizer.zero_grad()
            output = model(input_sample)

            # Data loss
            loss_data = criterion(output, label_sample * multiple)
            losses.update(loss_data.item(), input_sample.size(0))
            loss_data.backward()
            optimizer.step()

            avg_loss = loss_data.item() / len(input_sample)
            train_losses.append(avg_loss)
            pbar.set_postfix({'Loss': f'{avg_loss:.8f}'})
            pbar.update(input_sample.size(0))

        loss_avg = losses.avg
    return loss_avg, train_losses


def validate(args, epoch, model, criterion, valid_loader, device, valid_losses, multiple, losses=None):
    """Validation loop"""
    window_size = 45
    model.eval()
    if losses is None:
        losses = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(valid_loader.dataset), desc=f'Val[{epoch}/{args.epochs}') as pbar:
            for i, (BV_samples, DDL_samples, idx) in enumerate(valid_loader):
                # Prepare input based on channel configuration
                if args.set_channel == 1:
                    input_sample = BV_samples.to(device).float().unsqueeze(1)
                elif args.set_channel == 3:
                    input_data = BV_samples.to(device).float()
                    # FFT for magnitude spectrum
                    fft_results = torch.fft.fft(input_data)
                    magnitude_spectrum = torch.abs(fft_results)
                    # Calculate volatility
                    input_data_np = input_data.cpu().numpy()
                    volatility = numpy.array([moving_std(x, window_size)[None, :] for x in input_data_np])
                    volatility = torch.from_numpy(volatility).float().to(device)
                    # Combine three channels
                    combined_data = torch.cat((input_data.unsqueeze(1), magnitude_spectrum.unsqueeze(1), volatility),
                                              dim=1)
                    input_sample = combined_data

                label_sample = DDL_samples.to(device).float()
                output = model(input_sample)
                loss = criterion(output, label_sample * multiple)
                losses.update(loss.item(), input_sample.size(0))

                avg_loss = loss.item() / len(input_sample)
                valid_losses.append(avg_loss)
                pbar.set_postfix({'Loss': f'{avg_loss:.8f}'})
                pbar.update(input_sample.size(0))

            loss_avg = losses.avg
    return loss_avg, valid_losses


def select_sub_dataset(dataset, sub_rate):
    """Select subset of dataset"""
    num_samples = int(len(dataset) * sub_rate)
    sub_data = [dataset[i] for i in range(num_samples)]
    return sub_data


def main(args, file_path_bv):
    """Main training function"""
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Set device and random seed
    device = set_device(args)
    set_random_seed(args)

    # Create model instance
    if args.model == 'resnet50':
        model = ResNet50(in_channel=args.set_channel, num_elements=args.num_ddl).to(device)
    elif args.model == 'unet':
        model = UNet(in_channels=args.set_channel, out_channels=1, num_elements=args.num_ddl).to(device)
    elif args.model == 'resunet':
        model = ResUNet(in_channels=args.set_channel, out_channels=1, num_elements=args.num_ddl).to(device)
    elif args.model == 'ds_LSTM':
        model = DoubleStageLSTM(input_size=2160, hidden_size=2048, output_size=args.num_ddl).to(device)
    elif args.model == 'ea_resnet':
        model = EAResNet50(in_channel=args.set_channel, num_elements=args.num_ddl).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    print("Loading model:", args.model)

    # Set loss function and optimizer
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True,
                                                           min_lr=0.000001)
    cudnn.benchmark = True

    # Load datasets
    dataset_bv_samples = H5DatasetWithIndex(file_path_bv)
    dataset_ddl_samples = H5DatasetWithIndex(args.ddl_file)
    print("Selected datasets:\n", file_path_bv, "\n", args.ddl_file)

    # Select subset of data for training and validation
    sub_rate = 1
    print("Data usage ratio:", sub_rate)
    sub_dataset_bv_samples = select_sub_dataset(dataset_bv_samples, sub_rate)
    sub_dataset_ddl_samples = select_sub_dataset(dataset_ddl_samples, sub_rate)

    train_dataset_bv, val_dataset_bv = split_dataset(sub_dataset_bv_samples, args.train_ratio)
    train_dataset_ddl, val_dataset_ddl = split_dataset(sub_dataset_ddl_samples, args.train_ratio)

    # Create paired datasets
    train_dataset = PairedDataset(train_dataset_bv, train_dataset_ddl)
    val_dataset = PairedDataset(val_dataset_bv, val_dataset_ddl)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Training data loaded successfully!")

    # Training loop initialization
    best_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []
    all_train_losses = []
    all_val_losses = []
    early_stop_counter = 0

    for epoch in range(args.epochs):
        train_loss_avg, train_losses = train(args, epoch, model, criterion, optimizer, train_dataloader,
                                             device, train_losses, multiple=args.multiple, losses=None)
        val_loss_avg, val_losses = validate(args, epoch, model, criterion, val_dataloader, device, val_losses,
                                            multiple=args.multiple)

        # Save epoch losses
        all_train_losses.append(train_loss_avg)
        all_val_losses.append(val_loss_avg)

        if val_loss_avg < best_loss:
            best_model = model
            print(f"Epoch {epoch}: Updated best_model, current train loss: {train_loss_avg}; val loss: {val_loss_avg}")

        # Early stopping check
        stop_training, best_loss, early_stop_counter = early_stopping(args, val_loss_avg, best_loss, epoch,
                                                                      early_stop_counter, patience=20)
        if stop_training:
            break
        scheduler.step(val_loss_avg)

    # Save best model
    save_directory = './output/save_model/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
    filename = f'{save_directory}BestRMSE_{args.model}_model_{args.epochs}epochs_{args.lr}_{save_time}.pth'

    if best_model is not None:
        torch.save(best_model.state_dict(), filename)
        print(f"Best model saved to {filename}")
    else:
        print("Warning: best_model is None, not saving.")

    # Save losses to CSV and plot
    save_losses_to_csv_and_plot(all_train_losses, all_val_losses, save_time, directory='./output/save_loss/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch 3D-EIT Reconstruction")

    parser.add_argument('--ddl_file', type=str,
                        default='your path/DDL_samples.h5',
                        help="Conductivity data file path")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=19940806, help='Random seed')
    parser.add_argument('--gpu', type=str, default=0, help='GPU device')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--early_stop', action='store_true', default=True, help='Enable early stopping')
    parser.add_argument('--model', type=str, default='unet',
                        help='Model type: resnet50, unet, resunet, ds_LSTM, ea_resnet')
    parser.add_argument('--multiple', type=int, default=1000, help='Scaling factor')
    parser.add_argument('--num_ddl', type=int, default=95295, help='Number of conductivity elements')
    parser.add_argument('--set_channel', type=int, default=3, help='Input channels (1 or 3)')

    args = parser.parse_args()

    print("Using model UNet")
    main(args, file_path_bv=r"your path/BV_samples.h5")