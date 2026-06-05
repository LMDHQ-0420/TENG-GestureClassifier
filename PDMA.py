import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.signal import butter, filtfilt, iirnotch, stft, istft

def spectral_subtraction(signal, fs, nperseg=256):
    """通过谱减法抑制各通道独立的底噪"""
    f, t_spec, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    energy = np.sum(magnitude**2, axis=0)
    noise_idx = max(1, int(len(t_spec) * 0.1))
    noise_frames = np.argsort(energy)[:noise_idx]
    noise_mu = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    
    alpha, beta = 2.0, 0.02
    
    subtracted_mag = np.maximum(magnitude**2 - (alpha * noise_mu**2), (beta * noise_mu)**2)**0.5
    
    Zxx_new = subtracted_mag * np.exp(1j * phase)
    _, signal_clean = istft(Zxx_new, fs=fs, nperseg=nperseg)
    
    if len(signal_clean) < len(signal):
        signal_clean = np.pad(signal_clean, (0, len(signal) - len(signal_clean)))
    else:
        signal_clean = signal_clean[:len(signal)]
    return signal_clean

def detect_artifact_boundary(signal, fs, search_range_ms=500):
    """自适应探测并返回稳定信号的边界索引"""
    envelope = np.abs(signal)
    search_samples = int((search_range_ms / 1000) * fs)
 
    head_segment = envelope[:search_samples]
    head_idx = np.argmin(head_segment) + 1 if len(head_segment) > 0 else 0
        
    tail_segment = envelope[-search_samples:]
    tail_idx = len(signal) - (len(tail_segment) - np.argmin(tail_segment)) - 1 if len(tail_segment) > 0 else len(signal)
        
    return head_idx, tail_idx

def calculate_9d_vector(stable_data, fs):
    """提取全局9维手势特征向量"""
    mav = np.mean(np.abs(stable_data), axis=0)
    duration = len(stable_data) / fs
    wl = np.sum(np.abs(np.diff(stable_data, axis=0)), axis=0) / duration
    rms = np.sqrt(np.mean(stable_data**2, axis=0))
    ratio = rms / np.sum(rms) if np.sum(rms) > 0 else np.zeros(3)
    return np.concatenate([mav, wl, ratio])

def calculate_spatial_features(data_array, fs, window_ms=100, step_ms=50):
    """计算滑动窗口空间分布特征（Ratio）"""
    window_samples = int((window_ms / 1000) * fs)
    step_samples = int((step_ms / 1000) * fs)
    feature_list = []
    num_samples = data_array.shape[0]
    
    for start in range(0, num_samples - window_samples, step_samples):
        end = start + window_samples
    
        window = data_array[start:end, :]
        rms_values = np.sqrt(np.mean(window**2, axis=0))
        total_rms = np.sum(rms_values)
        ratios = rms_values / total_rms if total_rms > 0 else np.zeros(3)
        feature_list.append([start / fs, ratios[0], ratios[1], ratios[2]])
        
    return pd.DataFrame(feature_list, columns=['Window_Start_Time(s)', 'CH1_Ratio', 'CH3_Ratio', 'CH5_Ratio'])

def process_selected_files():
    """主程序：执行数据读取、处理与分类保存"""
    root = tk.Tk()
    root.withdraw()
    
    file_paths = filedialog.askopenfilenames(title="选择肌电信号CSV文件", filetypes=[("CSV files", "*.csv")])
 
    if not file_paths:
        return

    fs = 1000
    nyq = 0.5 * fs
    b_band, a_band = butter(4, [20/nyq, 450/nyq], btype='band')
    b_notch, a_notch = iirnotch(50/nyq, 30)

    for path in file_paths:
        dir_name = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        target_dir = os.path.join(dir_name, base_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        try:
            df = pd.read_csv(path, header=None)
            selected_data = df[[0, 2, 4]].values
            
            processed_channels = []
            head_indices, tail_indices = [], []

            for i in range(3):
                col = selected_data[:, i] - np.mean(selected_data[:, i])
                col = filtfilt(b_band, a_band, col)
                col = filtfilt(b_notch, a_notch, col)
                clean_col = spectral_subtraction(col, fs)
           
                processed_channels.append(clean_col)
                h, t = detect_artifact_boundary(clean_col, fs)
                head_indices.append(h)
                tail_indices.append(t)

            final_head = max(head_indices)
            final_tail = min(tail_indices)
 
            # 1. 保存清洗后的时域信号数据
            cleaned_array = np.array(processed_channels).T
            time_axis = np.arange(len(selected_data)) / fs
            output_df = pd.DataFrame(cleaned_array, columns=['CH1_Clean', 'CH3_Clean', 'CH5_Clean'])
            output_df.insert(0, 'Time(s)', time_axis)
            final_signal_df = output_df.iloc[final_head : final_tail].copy()
            final_signal_df.to_csv(os.path.join(target_dir, f"cleaned_{base_name}.csv"), index=False)

            # 2. 保存全局9维特征向量
            stable_vals = final_signal_df[['CH1_Clean', 'CH3_Clean', 'CH5_Clean']].values
            vector_values = calculate_9d_vector(stable_vals, fs)
            feature_names = ['MAV_CH1', 'MAV_CH3', 'MAV_CH5', 'WL_CH1', 'WL_CH3', 'WL_CH5', 'Ratio_CH1', 'Ratio_CH3', 'Ratio_CH5']
            pd.DataFrame([vector_values], columns=feature_names).to_csv(os.path.join(target_dir, f"vector_{base_name}.csv"), index=False)

            # 3. 保存滑动窗口空间特征数据 (features)
            feature_df = calculate_spatial_features(stable_vals, fs)
            feature_df['Window_Start_Time(s)'] += (final_head / fs)
            feature_df.to_csv(os.path.join(target_dir, f"features_{base_name}.csv"), index=False)

            # 4. 保存统计摘要数据 (summary)
            ratio_cols = ['CH1_Ratio', 'CH3_Ratio', 'CH5_Ratio']
            summary_df = pd.DataFrame({
                'Channel': ratio_cols,
                'Mean': feature_df[ratio_cols].mean().values,
                'Std_Dev': feature_df[ratio_cols].std().values,
                'Max': feature_df[ratio_cols].max().values
            })
            summary_df.to_csv(os.path.join(target_dir, f"summary_{base_name}.csv"), index=False)

            print(f"文件 {base_name} 处理完成。输出目录: {target_dir}")

        except Exception as e:
            print(f"处理文件 {base_name} 时发生错误: {e}")

if __name__ == "__main__":
    process_selected_files()