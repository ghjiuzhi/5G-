import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from channel_model import ChannelModel

class MultipathModel(ChannelModel):
    def __init__(self, freq_ghz=28, tx_height=30, rx_height=1.5):
        """
        初始化多径传播模型
        """
        super().__init__(freq_ghz, tx_height, rx_height)
        
    def two_ray_path_loss(self, d, reflection_coef=-1.0):
        """
        计算双径模型路径损耗
        
        参数:
            d: 水平距离(m)
            reflection_coef: 地面反射系数，默认为-1(完全反射)
            
        返回:
            双径模型路径损耗(dB)
        """
        h_t = self.tx_height
        h_r = self.rx_height
        
        d_los = np.sqrt(d**2 + (h_t - h_r)**2)  # 直射路径长度
        d_ref = np.sqrt(d**2 + (h_t + h_r)**2)  # 反射路径长度
        
        # 相位差
        phase_diff = 2 * np.pi * (d_ref - d_los) / self.wavelength
        
        # 计算总场强
        E_los = 1/d_los * np.exp(-1j*2*np.pi*d_los/self.wavelength)
        E_ref = reflection_coef/d_ref * np.exp(-1j*2*np.pi*d_ref/self.wavelength)
        E_total = np.abs(E_los + E_ref)
        
        # 参考自由空间场强
        E_free = 1/d_los
        
        # 计算路径损耗
        PL = -20 * np.log10(E_total/E_free) + 20 * np.log10(4*np.pi*d_los/self.wavelength)
        
        return PL
    
    def frequency_selective_fading(self, d, bandwidth, num_points=1000):
        """
        计算频率选择性衰落
        
        参数:
            d: 水平距离(m)
            bandwidth: 带宽(Hz)
            num_points: 频率点数
            
        返回:
            频率点和对应的信道响应
        """
        h_t = self.tx_height
        h_r = self.rx_height
        
        d_los = np.sqrt(d**2 + (h_t - h_r)**2)
        d_ref = np.sqrt(d**2 + (h_t + h_r)**2)
        
        # 时延差
        delay_diff = (d_ref - d_los) / self.c
        
        # 中心频率
        f_c = self.freq
        
        # 频率范围
        f = np.linspace(f_c - bandwidth/2, f_c + bandwidth/2, num_points)
        
        # 计算每个频率点的信道响应
        H = np.zeros(num_points, dtype=complex)
        for i, freq in enumerate(f):
            wavelength = self.c / freq
            
            # 直射路径
            E_los = 1/d_los * np.exp(-1j*2*np.pi*d_los/wavelength)
            
            # 反射路径
            E_ref = -1/d_ref * np.exp(-1j*2*np.pi*d_ref/wavelength)
            
            # 总信道响应
            H[i] = E_los + E_ref
        
        return f, H
    
    def rayleigh_fading(self, snr_db, samples=1000):
        """
        生成瑞利衰落信道
        
        参数:
            snr_db: 信噪比(dB)
            samples: 样本数
            
        返回:
            瞬时SNR
        """
        snr = 10**(snr_db/10)
        
        # 生成复高斯随机变量
        h_real = np.random.normal(0, np.sqrt(0.5), samples)
        h_imag = np.random.normal(0, np.sqrt(0.5), samples)
        h = h_real + 1j*h_imag
        
        # 计算瞬时SNR
        gamma = snr * np.abs(h)**2
        return gamma
    
    def rician_fading(self, snr_db, K_factor=1, samples=1000):
        """
        生成莱斯衰落信道
        
        参数:
            snr_db: 信噪比(dB)
            K_factor: 莱斯K因子，表示直射分量与散射分量功率比
            samples: 样本数
            
        返回:
            瞬时SNR
        """
        snr = 10**(snr_db/10)
        
        # 计算直射分量和散射分量的功率
        p_direct = K_factor / (K_factor + 1)
        p_scatter = 1 / (K_factor + 1)
        
        # 生成直射分量(确定性)
        direct = np.sqrt(p_direct)
        
        # 生成散射分量(随机)
        scatter_real = np.random.normal(0, np.sqrt(p_scatter/2), samples)
        scatter_imag = np.random.normal(0, np.sqrt(p_scatter/2), samples)
        scatter = scatter_real + 1j*scatter_imag
        
        # 总信道增益
        h = direct + scatter
        
        # 计算瞬时SNR
        gamma = snr * np.abs(h)**2
        return gamma
    
    def calculate_ber_qpsk(self, snr_db, channel_type='rayleigh', K_factor=1, samples=10000):
        """
        计算QPSK调制下的误比特率
        
        参数:
            snr_db: 信噪比(dB)
            channel_type: 信道类型，'rayleigh'或'rician'
            K_factor: 莱斯K因子
            samples: 样本数
            
        返回:
            误比特率
        """
        if channel_type.lower() == 'rayleigh':
            gamma = self.rayleigh_fading(snr_db, samples)
        elif channel_type.lower() == 'rician':
            gamma = self.rician_fading(snr_db, K_factor, samples)
        else:
            raise ValueError("信道类型必须是'rayleigh'或'rician'")
        
        # QPSK调制的理论BER
        ber = 0.5 * special.erfc(np.sqrt(gamma/2))
        
        return np.mean(ber)