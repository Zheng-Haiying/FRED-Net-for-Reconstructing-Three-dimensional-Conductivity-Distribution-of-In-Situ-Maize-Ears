import os
import time
import datetime
import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pandas
import sys
import h5py
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from scipy.signal import convolve
from sklearn.metrics import r2_score


def compute_icc(P, Q):
    """Compute Intraclass Correlation Coefficient (ICC)"""
    P, Q = np.asarray(P).flatten(), np.asarray(Q).flatten()
    assert P.shape == Q.shape, "Input variables P and Q must have the same shape"

    # Calculate means
    mean_P = np.mean(P)
    mean_Q = np.mean(Q)
    mean_overall = np.mean([P, Q])
    n = len(P)

    # Calculate sum of squares
    ss_between = ((mean_P - mean_overall) ** 2 + (mean_Q - mean_overall) ** 2) * n
    ss_within = np.sum((P - mean_P) ** 2 + (Q - mean_Q) ** 2)
    ss_between_measurements = ((np.mean(P) - mean_overall) ** 2 + (np.mean(Q) - mean_overall) ** 2) * n
    ss_residual = ss_within - ss_between_measurements

    # Calculate mean squares
    ms_between = ss_between
    ms_residual = ss_residual / (n - 1)

    # Calculate ICC
    icc = (ms_between - ms_residual) / (ms_between + (n - 1) * ms_residual)
    return icc


def calculate_ssim(P, Q, win_size=11, C3=0.03 ** 2, stride=1):
    """Calculate Structural Similarity Index (SSIM)"""
    if P.shape != Q.shape:
        raise ValueError("Input images must have the same dimensions.")

    if len(P.shape) == 2:
        P = np.transpose(P)
        Q = np.transpose(Q)
    elif len(P.shape) > 2:
        raise ValueError("Please provide a vector.")

    # Convert to float
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)

    # 1D window for vector processing
    window = np.ones(win_size) / win_size

    # Compute averages using convolution
    mu_P = convolve(P, window, mode='valid')[::stride]
    mu_Q = convolve(Q, window, mode='valid')[::stride]

    # Compute variances and covariance
    P_sq = P ** 2
    Q_sq = Q ** 2
    P_Q = P * Q
    sigma_P_sq = convolve(P_sq, window, mode='valid')[::stride] - mu_P ** 2
    sigma_Q_sq = convolve(Q_sq, window, mode='valid')[::stride] - mu_Q ** 2
    sigma_PQ = convolve(P_Q, window, mode='valid')[::stride] - mu_P * mu_Q

    # Compute SSIM components
    l = (2 * mu_P * mu_Q + C3) / (mu_P ** 2 + mu_Q ** 2 + C3)
    c = (2 * np.sqrt(sigma_P_sq) * np.sqrt(sigma_Q_sq) + C3) / (sigma_P_sq + sigma_Q_sq + C3)
    s = (sigma_PQ + C3 / 2) / (np.sqrt(sigma_P_sq) * np.sqrt(sigma_Q_sq) + C3 / 2)

    ssim = l * c * s
    return np.mean(ssim)


def RIE(P, Q):
    """Calculate Relative Image Error (RIE)"""
    return np.sum(np.abs(Q - P)) / np.sum(Q)


def RMSE(P, Q):
    """Calculate Root Mean Square Error (RMSE)"""
    return np.sqrt(np.mean((P - Q) ** 2))


class H5Dataset(torch.utils.data.Dataset):
    """Dataset class for loading HDF5 files"""

    def __init__(self, file_path, *dataset_names):
        self.file_path = file_path
        self.dataset_names = dataset_names

        # Check if file exists and dataset names are correct
        with h5py.File(self.file_path, 'r') as f:
            for name in self.dataset_names:
                if name not in f:
                    raise ValueError(f"Dataset {name} not found in the file")

        # Check if dataset lengths are consistent
        with h5py.File(self.file_path, 'r') as f:
            lengths = [len(f[name]) for name in self.dataset_names]
            if not all(length == lengths[0] for length in lengths):
                raise ValueError("All dataset lengths must be the same")

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f[self.dataset_names[0]])

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            return tuple(torch.from_numpy(f[name][idx]) for name in self.dataset_names)


class Logger(object):
    """Logger class for GUI text widget and file output"""

    def __init__(self, text_widget, save_path, file_prefix="RMSEs_Loss_Time"):
        self.text_widget = text_widget
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file_name = f"{file_prefix}_{current_time}.log"
        self.log_path = os.path.join(save_path, log_file_name)
        self.log_file = open(self.log_path, "a")

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.log_file.write(message)

    def flush(self):
        pass


def set_device(gpu_id):
    """Set computing device (CPU or GPU)"""
    if gpu_id is not None:
        print(f"Using GPU: {gpu_id} for inference; Device name: {torch.cuda.get_device_name(int(gpu_id))}")
        return torch.device(f"cuda:{gpu_id}")
    elif torch.cuda.is_available():
        print(
            f"Using GPU device; Number of devices: {torch.cuda.device_count()}; Device name: {torch.cuda.get_device_name()}")
        return torch.device("cuda:0")
    else:
        print('Using CPU device')
        return torch.device("cpu")


def run_prediction(datadir, main_modelpath, main_style, main_savepath, batch_size=1, gpu=None, selected_datasets=None):
    """Main prediction function with comprehensive evaluation metrics"""
    # Set computing device
    device = set_device(gpu)
    rmses = []
    iccs = []
    ssims = []
    ries = []
    losses = []
    times_invp = []
    r2s = []

    sys.stdout = Logger(log_text, save_path=main_savepath)
    sys.stderr = Logger(log_text, save_path=main_savepath)

    # Load data
    with h5py.File(datadir, 'r') as f:
        dataset_names = list(f.keys())
        print("Available datasets:", dataset_names)

    dataset = H5Dataset(datadir, *selected_datasets)
    print("Selected datasets:", dataset.dataset_names)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load trained model
    model = torch.load(main_modelpath)
    print('Model name:', main_modelpath)
    model.cuda()

    criterion = nn.MSELoss().cuda()

    for i, (BV_samples, DDL_samples, _) in enumerate(dataloader):
        # Set to evaluation mode
        model.eval()
        m = i + 1

        input_sample = BV_samples.to(device).to(torch.float32)  # Boundary voltage input
        label_sample = DDL_samples.to(device).to(torch.float32)  # True conductivity

        # Compute model output
        start_time = time.time()
        output = model(input_sample)
        output = output / 1000  # Scale adjustment if needed
        end_time = time.time()

        loss = criterion(output, label_sample)
        rmse = torch.sqrt(loss)

        output, label_sample = output.flatten(), label_sample.flatten()

        # Calculate evaluation metrics
        icc = compute_icc(output.cpu().detach().numpy(), label_sample.cpu().detach().numpy())
        ssim = calculate_ssim(output.cpu().detach().numpy(), label_sample.cpu().detach().numpy())
        rie = RIE(output.cpu().detach().numpy(), label_sample.cpu().detach().numpy())
        r2 = r2_score(label_sample.cpu().detach().numpy(), output.cpu().detach().numpy())

        # Save output
        save_output = output.data.cpu()
        save_path_output = os.path.join(main_savepath, 'Output_{}'.format(main_style))
        if not os.path.exists(save_path_output):
            os.makedirs(save_path_output)
        save_name_output = os.path.join(save_path_output, str(m) + '_ddl' + '.csv')
        numpy.savetxt(save_name_output, save_output, fmt='%.4f', delimiter=',')

        # Store metrics
        losses.append(loss.data.cpu().numpy())
        rmses.append(rmse.data.cpu().numpy())
        iccs.append(icc)
        ssims.append(ssim)
        ries.append(rie)
        r2s.append(r2)

        # Calculate inference time
        time_invp = end_time - start_time
        times_invp.append(time_invp)

    # Calculate statistics
    losses_mean = numpy.mean(losses)
    RMSEs_mean = numpy.mean(rmses)
    iccs_mean = numpy.mean(iccs)
    ssims_mean = numpy.mean(ssims)
    ries_mean = numpy.mean(ries)
    r2s_mean = numpy.mean(r2s)

    losses_std = numpy.std(losses)
    RMSEs_std = numpy.std(rmses)
    iccs_std = numpy.std(iccs)
    ssims_std = numpy.std(ssims)
    ries_std = numpy.std(ries)
    r2s_std = numpy.std(r2s)

    InvpTime_mean = numpy.mean(times_invp)
    InvpTime_std = numpy.std(times_invp)

    print('-------------------------------- Evaluation Results -------------------------------')
    print('Loss mean:', losses_mean, '; Loss std:', losses_std)
    print('RMSE mean:', RMSEs_mean, '; RMSE std:', RMSEs_std)
    print('ICC mean:', iccs_mean, '; ICC std:', iccs_std)
    print('SSIM mean:', ssims_mean, '; SSIM std:', ssims_std)
    print('RIE mean:', ries_mean, '; RIE std:', ries_std)
    print('R² mean:', r2s_mean, '; R² std:', r2s_std)
    print('Inference time mean:', InvpTime_mean, '; std:', InvpTime_std)

    # Save results to DataFrame
    data = {
        'Loss': losses,
        'RMSE': rmses,
        'ICC': iccs,
        'SSIM': ssims,
        'RIE': ries,
        'R²': r2s,
        'Inference_Time': times_invp
    }
    df = pandas.DataFrame(data)

    # Save to CSV file
    save_name = os.path.join(main_savepath, 'Results_{}'.format(main_style) + '.csv')
    df.to_csv(save_name, index=False)


def browse_input_file():
    """Browse for input data file"""
    filename = filedialog.askopenfilename()
    entry_input.delete(0, tk.END)
    entry_input.insert(0, filename)


def browse_model_file():
    """Browse for model file"""
    filename = filedialog.askopenfilename()
    entry_model.delete(0, tk.END)
    entry_model.insert(0, filename)


def browse_save_dir():
    """Browse for save directory"""
    save_dir = filedialog.askdirectory()
    entry_savepath.delete(0, tk.END)
    entry_savepath.insert(0, save_dir)


def run_prediction_gui():
    """Run prediction from GUI inputs"""
    datadir = entry_input.get()
    main_modelpath = entry_model.get()
    main_style = entry_style.get()
    main_savepath = entry_savepath.get()
    batch_size = int(entry_batch_size.get())
    gpu = entry_gpu.get()
    selected_datasets = [var.get() for var in dataset_vars if var.get()]

    if 'DDL_samples' not in selected_datasets or 'Vol_samples' not in selected_datasets:
        messagebox.showerror("Error", "DDL_samples and Vol_samples are required!")
        return

    run_prediction(datadir, main_modelpath, main_style, main_savepath, batch_size, gpu, selected_datasets)
    root.destroy()


# Create GUI
root = tk.Tk()
root.title("EIT-3D Reconstruction Evaluation GUI")

# Input file selection
label_input = tk.Label(root, text="Select input file:")
label_input.grid(row=0, column=0, sticky="w")
entry_input = tk.Entry(root, width=50)
entry_input.grid(row=0, column=1, padx=5, pady=5)
button_browse_input = tk.Button(root, text="Browse", command=browse_input_file)
button_browse_input.grid(row=0, column=2)

# Model file selection
label_model = tk.Label(root, text="Select model file:")
label_model.grid(row=1, column=0, sticky="w")
entry_model = tk.Entry(root, width=50)
entry_model.grid(row=1, column=1, padx=5, pady=5)
button_browse_model = tk.Button(root, text="Browse", command=browse_model_file)
button_browse_model.grid(row=1, column=2)

# Save keyword
label_style = tk.Label(root, text="Save keyword:")
label_style.grid(row=2, column=0, sticky="w")
entry_style = tk.Entry(root)
entry_style.grid(row=2, column=1, padx=5, pady=5)

# Save path
label_savepath = tk.Label(root, text="Save path:")
label_savepath.grid(row=3, column=0, sticky="w")
entry_savepath = tk.Entry(root, width=50)
entry_savepath.grid(row=3, column=1, padx=5, pady=5)
button_browse_save = tk.Button(root, text="Select folder", command=browse_save_dir)
button_browse_save.grid(row=3, column=2)

# Dataset selection frame
dataset_frame = tk.Frame(root)
dataset_frame.grid(row=4, column=0, columnspan=3, sticky="w")

label_datasets = tk.Label(dataset_frame, text="Select datasets:")
label_datasets.grid(row=0, column=0, sticky="w")

datasets = ['BV_20dB_samples', 'BV_30dB_samples', 'BV_40dB_samples', 'BV_50dB_samples', 'BV_samples', 'DDL_samples',
            'Vol_samples']
dataset_vars = [tk.StringVar() for _ in datasets]
for idx, dataset in enumerate(datasets):
    tk.Checkbutton(dataset_frame, text=dataset, variable=dataset_vars[idx],
                   onvalue=dataset, offvalue='').grid(row=idx + 1, column=0, sticky="w")

# Workers setting
label_workers = tk.Label(root, text="Data loading workers:")
label_workers.grid(row=8, column=0, sticky="w")
entry_workers = tk.Entry(root)
entry_workers.insert(0, "4")
entry_workers.grid(row=8, column=1, padx=5, pady=5)

# Batch size setting
label_batch_size = tk.Label(root, text="Batch size:")
label_batch_size.grid(row=9, column=0, sticky="w")
entry_batch_size = tk.Entry(root)
entry_batch_size.insert(0, "1")
entry_batch_size.grid(row=9, column=1, padx=5, pady=5)

# GPU setting
label_gpu = tk.Label(root, text="GPU ID:")
label_gpu.grid(row=11, column=0, sticky="w")
entry_gpu = tk.Entry(root)
entry_gpu.insert(0, "0")
entry_gpu.grid(row=11, column=1, padx=5, pady=5)

# Log display
log_text = tk.Text(root, width=60, height=10)
log_text.grid(row=13, columnspan=3)

# Run button
button_run = tk.Button(root, text="Run Evaluation", command=run_prediction_gui)
button_run.grid(row=12, column=1, pady=10)

# Start GUI
root.mainloop()