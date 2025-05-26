import numpy as np
from scipy.linalg import inv, svd

class MIMOBeamforming:
    def __init__(self, num_tx_ant, num_users):
        """
        初始化MIMO波束赋形
        
        参数:
            num_tx_ant: 发射天线数
            num_users: 用户数
        """
        self.Nt = num_tx_ant
        self.K = num_users
        
    def generate_channel(self, channel_type='rayleigh'):
        """
        生成信道矩阵
        
        参数:
            channel_type: 信道类型，'rayleigh'或'los'(视距)
            
        返回:
            信道矩阵H (K x Nt)
        """
        if channel_type.lower() == 'rayleigh':
            # 瑞利信道
            H = (np.random.normal(0, 1, (self.K, self.Nt)) + 
                 1j * np.random.normal(0, 1, (self.K, self.Nt))) / np.sqrt(2)
        elif channel_type.lower() == 'los':
            # 视距信道(简化模型)
            H = np.zeros((self.K, self.Nt), dtype=complex)
            for k in range(self.K):
                # 随机方向角
                theta = np.random.uniform(0, np.pi)
                # 生成导向矢量
                a = np.exp(1j * np.pi * np.arange(self.Nt) * np.sin(theta))
                H[k, :] = a / np.sqrt(self.Nt)
        else:
            raise ValueError("信道类型必须是'rayleigh'或'los'")
            
        return H
    
    def zf_beamforming(self, H):
        """
        零强制波束赋形
        
        参数:
            H: 信道矩阵 (K x Nt)
            
        返回:
            波束赋形矩阵W (Nt x K)
        """
        # 伪逆计算
        W = H.conj().T @ inv(H @ H.conj().T)
        
        # 功率归一化
        W_norm = np.sqrt(np.sum(np.abs(W)**2, axis=0))
        W = W / W_norm
        
        return W
    
    def mmse_beamforming(self, H, snr):
        """
        MMSE波束赋形
        
        参数:
            H: 信道矩阵 (K x Nt)
            snr: 信噪比(线性)
            
        返回:
            波束赋形矩阵W (Nt x K)
        """
        K = H.shape[0]
        I = np.eye(K)
        
        # MMSE预编码矩阵
        W = H.conj().T @ inv(H @ H.conj().T + I/snr)
        
        # 功率归一化
        W_norm = np.sqrt(np.sum(np.abs(W)**2, axis=0))
        W = W / W_norm
        
        return W
    
    def mrt_beamforming(self, H):
        """
        最大比合并发送(MRT)波束赋形
        
        参数:
            H: 信道矩阵 (K x Nt)
            
        返回:
            波束赋形矩阵W (Nt x K)
        """
        # MRT就是信道共轭
        W = H.conj().T
        
        # 功率归一化
        W_norm = np.sqrt(np.sum(np.abs(W)**2, axis=0))
        W = W / W_norm
        
        return W
    
    def calculate_sinr(self, H, W, snr):
        """
        计算SINR
        
        参数:
            H: 信道矩阵 (K x Nt)
            W: 波束赋形矩阵 (Nt x K)
            snr: 信噪比(线性)
            
        返回:
            每个用户的SINR
        """
        K = H.shape[0]
        sinr = np.zeros(K)
        
        for k in range(K):
            # 有用信号功率
            signal = np.abs(H[k] @ W[:, k])**2
            
            # 干扰功率
            interference = 0
            for j in range(K):
                if j != k:
                    interference += np.abs(H[k] @ W[:, j])**2
            
            # 噪声功率
            noise = 1/snr
            
            # 计算SINR
            sinr[k] = signal / (interference + noise)
            
        return sinr
    
    def calculate_capacity(self, sinr):
        """
        计算容量
        
        参数:
            sinr: 信干噪比
            
        返回:
            容量(bps/Hz)
        """
        return np.log2(1 + sinr)