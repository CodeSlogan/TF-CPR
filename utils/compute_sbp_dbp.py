import numpy as np
from scipy.signal import find_peaks

def calculate_sbp_dbp_from_abp_waveform(abp_waveform, sampling_rate=125):
    """
    从单条ABP波形中计算SBP和DBP 适配10秒窗口波形 
    逻辑与MIMIC-BP数据集一致:SBP=峰值最大值,DBP=谷值最小值
    参数：
        abp_waveform: numpy数组,长度=1250(10秒x125Hz)
        sampling_rate: 采样率,默认125Hz
    返回：
        sbp: float,收缩压;无有效峰值返回np.nan
        dbp: float,舒张压;无有效谷值返回np.nan
    """
    min_dist = int(0.2 * sampling_rate) 
    
    sbp_peaks, _ = find_peaks(
        abp_waveform,
        distance=min_dist,     
        height=(60, 200)     
    )
    
    if len(sbp_peaks) == 0:
        return np.nan, np.nan
    sbp = abp_waveform[sbp_peaks].max()

    abp_negated = -abp_waveform
    dbp_valleys, _ = find_peaks(
        abp_negated,
        distance=min_dist,    
        height=(-120, -30)   
    )
    
    if len(dbp_valleys) == 0:
        return np.nan, np.nan

    dbp = abp_waveform[dbp_valleys].min()
    return sbp, dbp


if __name__ == "__main__":
    test_waveform = np.array([80, 85, 90, 95, 100, 95, 90, 85, 80, 75,
                              70, 65, 60, 65, 70, 75, 80, 85, 90, 95,
                              100, 105, 111, 105, 100, 95, 90, 85, 80, 75,
                              70, 65, 60, 65, 70, 75, 80, 85, 90, 95,
                              100, 105, 115, 105, 100, 95, 90, 85, 80, 75,
                              70, 65, 60, 65, 70, 75, 80, 85, 90, 95,
                              100, 105, 120, 105, 100, 95, 
                              ])
    sbp, dbp = calculate_sbp_dbp_from_abp_waveform(test_waveform)
    print(f"Calculated SBP: {sbp}, DBP: {dbp}")
