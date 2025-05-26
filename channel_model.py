import numpy as np
import matplotlib.pyplot as plt
from scipy import special

class ChannelModel:
    def __init__(self, freq_ghz=28, tx_height=30, rx_height=1.5):
        """
        初始化信道模型
        
        参数:
            freq_ghz: 频率(GHz)
            tx_height: 发射机高度(m)
            rx_height: 接收机高度(m)
        """
        self.c = 3e8  # 光速(m/s)
        self.freq = freq_ghz * 1e9  # 频率转换为Hz
        self.wavelength = self.c / self.freq
        self.tx_height = tx_height
        self.rx_height = rx_height
        
    def fresnel_zone_radius(self, d, n=1):
        """
        计算第n个菲涅尔区半径
        
        参数:
            d: 发射机与接收机之间的距离(m)
            n: 菲涅尔区序号
            
        返回:
            第n个菲涅尔区半径(m)
        """
        d1 = d/2  # 假设发射机和接收机之间的距离平分
        return np.sqrt((n * self.wavelength * d1 * (d-d1)) / d)
    
    def obstacle_loss(self, h_c, F1):
        """
        计算障碍物造成的附加路径损耗
        
        参数:
            h_c: 障碍物高度(m)
            F1: 第一菲涅尔区半径(m)
            
        返回:
            附加路径损耗(dB)
        """
        v = h_c / F1  # 相对余隙
        # 基于ITU-R P.526-15模型计算衍射损耗
        if v > -0.7:
            J = 6.9 + 20 * np.log10(np.sqrt((v-0.1)**2 + 1) + v - 0.1)
        else:
            J = 0  # 当障碍物不显著时，衍射损耗可忽略
        return J

    def path_loss(self, distance):
        """
        计算基本路径损耗（自由空间路径损耗）
        
        参数:
            distance: 距离(m)
            
        返回:
            路径损耗(dB)
        """
        PL = 20 * np.log10(4 * np.pi * distance * self.freq / self.c)
        return PL
    
    def mmwave_path_loss(self, distance, alpha=2.1, shadow_std=4.0):
        """
        计算毫米波特定的路径损耗模型
        
        参数:
            distance: 距离(m)
            alpha: 路径损耗指数
            shadow_std: 阴影衰落标准差(dB)
            
        返回:
            毫米波路径损耗(dB)
        """
        # 参考距离1米处的路径损耗
        PL_0 = 20 * np.log10(4 * np.pi * self.freq / self.c)
        
        # 距离相关路径损耗
        PL = PL_0 + 10 * alpha * np.log10(distance)
        
        # 添加阴影衰落
        if shadow_std > 0:
            shadow_fading = np.random.normal(0, shadow_std)
            PL += shadow_fading
            
        return PL