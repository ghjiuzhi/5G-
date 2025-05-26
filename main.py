import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from channel_model import ChannelModel
from multipath_model import MultipathModel
from mimo_beamforming import MIMOBeamforming

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def plot_path_loss():
    """分析不同障碍物高度下的路径损耗"""
    channel = ChannelModel(freq_ghz=28)
    distances = np.linspace(10, 1000, 100) # 距离范围 10m 到 1000m

    plt.figure(figsize=(10, 6))

    # 绘制自由空间路径损耗作为参考
    free_space_loss = [channel.path_loss(d) for d in distances]
    plt.plot(distances, free_space_loss, 'k--', label='自由空间路径损耗')

    # 分析不同 h_c/F1 比例下的总路径损耗
    # 注意：这里的 h_c/F1 比例仅用于设定障碍物高度 h_c，
    # obstacle_loss 函数内部会根据 d 和 h_c 计算标准的衍射参数 v
    for h_c_ratio in [0, 0.3, 0.6, 1.0]:
        total_loss = []
        for d in distances:
            # 计算当前距离下的第一菲涅尔区半径 (假设障碍物在中间)
            F1 = channel.fresnel_zone_radius(d)
            # 根据比例计算障碍物超出视线的高度
            h_c = h_c_ratio * F1
            # 计算障碍物造成的附加损耗，传递距离 d 和高度 h_c
            additional_loss = channel.obstacle_loss(d, h_c)
            # 计算总路径损耗 (自由空间损耗 + 附加损耗)
            path_loss = channel.path_loss(d)
            total_loss.append(path_loss + additional_loss)
        # 绘制结果，标签仍然使用 h_c/F1 比例，因为它代表了场景设置
        plt.plot(distances, total_loss, label=f'h_c/F1 ≈ {h_c_ratio}')

    plt.grid(True)
    plt.xlabel('距离 (m)')
    plt.ylabel('路径损耗 (dB)')
    plt.title('不同障碍物相对余隙下的路径损耗')
    plt.legend()
    # 保存图表到 results 目录
    results_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, 'path_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_frequency_selective_fading():
    """分析频率选择性衰落"""
    multipath = MultipathModel(freq_ghz=28)
    
    # 不同距离下的频率选择性衰落
    distances = [100, 200, 500]
    bandwidth = 400e6  # 400MHz带宽
    
    plt.figure(figsize=(12, 8))
    
    for d in distances:
        f, H = multipath.frequency_selective_fading(d, bandwidth)
        f_ghz = (f - 28e9) / 1e6  # 转换为MHz偏移
        plt.plot(f_ghz, 20*np.log10(np.abs(H)), label=f'd = {d}m')
    
    plt.xlabel('频率偏移 (MHz)')
    plt.ylabel('信道增益 (dB)')
    plt.title('不同距离下的频率选择性衰落')
    plt.legend()
    plt.grid(True)
    plt.savefig('frequency_selective_fading.png', dpi=300)
    plt.show()

def plot_ber_vs_snr():
    """分析BER与SNR的关系"""
    multipath = MultipathModel(freq_ghz=28)
    snr_db = np.linspace(0, 30, 16)
    
    # 计算不同信道下的BER
    ber_rayleigh = []
    ber_rician_k5 = []
    ber_rician_k10 = []
    
    for snr in snr_db:
        ber_rayleigh.append(multipath.calculate_ber_qpsk(snr, 'rayleigh'))
        ber_rician_k5.append(multipath.calculate_ber_qpsk(snr, 'rician', K_factor=5))
        ber_rician_k10.append(multipath.calculate_ber_qpsk(snr, 'rician', K_factor=10))
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db, ber_rayleigh, 'b-o', label='瑞利衰落')
    plt.semilogy(snr_db, ber_rician_k5, 'r-s', label='莱斯衰落 (K=5)')
    plt.semilogy(snr_db, ber_rician_k10, 'g-^', label='莱斯衰落 (K=10)')
    
    # 添加AWGN信道理论BER作为参考
    ber_awgn = 0.5 * special.erfc(np.sqrt(10**(snr_db/10)/2))
    plt.semilogy(snr_db, ber_awgn, 'k--', label='AWGN信道')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('误比特率 (BER)')
    plt.title('QPSK调制下不同信道的BER性能')
    plt.grid(True)
    plt.legend()
    plt.savefig('ber_vs_snr.png', dpi=300)
    plt.show()

def analyze_mimo_performance():
    """分析MIMO波束赋形性能"""
    # 设置参数
    Nt_values = [16, 32, 64, 128]  # 不同的发射天线数
    K = 4    # 用户数
    snr_db = np.linspace(0, 30, 16)
    
    plt.figure(figsize=(12, 8))
    
    for Nt in Nt_values:
        mimo = MIMOBeamforming(Nt, K)
        
        # 生成随机信道矩阵
        H = mimo.generate_channel('rayleigh')
        
        # 存储不同SNR下的容量
        zf_capacity = []
        mmse_capacity = []
        
        for snr in snr_db:
            snr_linear = 10**(snr/10)
            
            # ZF波束赋形
            W_zf = mimo.zf_beamforming(H)
            sinr_zf = mimo.calculate_sinr(H, W_zf, snr_linear)
            zf_capacity.append(np.mean(mimo.calculate_capacity(sinr_zf)))
            
            # MMSE波束赋形
            W_mmse = mimo.mmse_beamforming(H, snr_linear)
            sinr_mmse = mimo.calculate_sinr(H, W_mmse, snr_linear)
            mmse_capacity.append(np.mean(mimo.calculate_capacity(sinr_mmse)))
        
        plt.plot(snr_db, zf_capacity, '--', label=f'ZF (Nt={Nt})')
        plt.plot(snr_db, mmse_capacity, '-', label=f'MMSE (Nt={Nt})')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('平均用户容量 (bps/Hz)')
    plt.title('不同天线数下ZF与MMSE波束赋形性能对比')
    plt.grid(True)
    plt.legend()
    plt.savefig('mimo_performance.png', dpi=300)
    plt.show()

def analyze_user_interference():
    """分析多用户干扰抑制能力"""
    Nt = 64  # 发射天线数
    K_values = [2, 4, 8, 16]  # 不同的用户数
    snr_db = 20  # 固定SNR
    snr_linear = 10**(snr_db/10)
    
    # 存储不同算法的性能
    zf_sinr = []
    mmse_sinr = []
    mrt_sinr = []
    
    for K in K_values:
        mimo = MIMOBeamforming(Nt, K)
        
        # 生成随机信道矩阵
        H = mimo.generate_channel('rayleigh')
        
        # ZF波束赋形
        W_zf = mimo.zf_beamforming(H)
        sinr_zf = mimo.calculate_sinr(H, W_zf, snr_linear)
        zf_sinr.append(np.mean(sinr_zf))
        
        # MMSE波束赋形
        W_mmse = mimo.mmse_beamforming(H, snr_linear)
        sinr_mmse = mimo.calculate_sinr(H, W_mmse, snr_linear)
        mmse_sinr.append(np.mean(sinr_mmse))
        
        # MRT波束赋形
        W_mrt = mimo.mrt_beamforming(H)
        sinr_mrt = mimo.calculate_sinr(H, W_mrt, snr_linear)
        mrt_sinr.append(np.mean(sinr_mrt))
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, 10*np.log10(zf_sinr), 'b-o', label='ZF')
    plt.plot(K_values, 10*np.log10(mmse_sinr), 'r-s', label='MMSE')
    plt.plot(K_values, 10*np.log10(mrt_sinr), 'g-^', label='MRT')
    
    plt.xlabel('用户数 (K)')
    plt.ylabel('平均SINR (dB)')
    plt.title(f'不同波束赋形算法的多用户干扰抑制能力 (Nt={Nt}, SNR={snr_db}dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('user_interference.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("开始5G毫米波信道模型仿真...")
    
    # 路径损耗分析
    plot_path_loss()
    
    # 频率选择性衰落分析
    analyze_frequency_selective_fading()
    
    # BER性能分析
    plot_ber_vs_snr()
    
    # MIMO波束赋形性能分析
    analyze_mimo_performance()
    
    # 多用户干扰抑制能力分析
    analyze_user_interference()
    
    print("仿真完成，结果已保存为图片文件。")
    
    # 生成完整报告
    generate_report = input("是否生成完整技术报告？(y/n): ")
    if generate_report.lower() == 'y':
        try:
            from generate_report import generate_report
            report_file = generate_report()
            print(f"报告已生成并保存到: {report_file}")
        except ImportError:
            print("未找到报告生成模块，请确保generate_report.py文件存在。")