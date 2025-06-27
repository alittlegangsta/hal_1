"""
数据加载和预处理模块
"""
import numpy as np
import scipy.io
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """测井数据加载器"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.cast_data = None
        self.xsilmr_data = {}
        
    def load_cast_data(self) -> Dict:
        """加载CAST超声测井数据"""
        logger.info("加载CAST数据...")
        cast_file = self.data_dir / "CAST.mat"
        
        if not cast_file.exists():
            raise FileNotFoundError(f"CAST文件不存在: {cast_file}")
            
        mat_data = scipy.io.loadmat(str(cast_file))
        cast_struct = mat_data['CAST'][0, 0]
        
        self.cast_data = {
            'Depth': cast_struct['Depth'].flatten(),  # 形状: (24750,)
            'Zc': cast_struct['Zc']  # 形状: (180, 24750)
        }
        
        logger.info(f"CAST数据加载完成: 深度点数={len(self.cast_data['Depth'])}, "
                   f"方位角数={self.cast_data['Zc'].shape[0]}")
        return self.cast_data
    
    def load_xsilmr_data(self) -> Dict:
        """加载所有XSILMR阵列声波测井数据"""
        logger.info("加载XSILMR数据...")
        xsilmr_dir = self.data_dir / "XSILMR"
        
        # 方位接收器标识
        sides = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        for receiver_idx in range(1, 14):  # 1-13号接收器
            mat_file = xsilmr_dir / f"XSILMR{receiver_idx:02d}.mat"
            
            if not mat_file.exists():
                logger.warning(f"文件不存在，跳过: {mat_file}")
                continue
                
            mat_data = scipy.io.loadmat(str(mat_file))
            xsilmr_struct = mat_data[f'XSILMR{receiver_idx:02d}'][0, 0]
            
            receiver_data = {
                'Depth': xsilmr_struct['Depth'].flatten(),
                'Tad': float(xsilmr_struct['Tad']) if 'Tad' in xsilmr_struct.dtype.names else 10.0
            }
            
            # 加载8个方位的波形数据
            for side in sides:
                wave_key = f'WaveRng{receiver_idx:02d}Side{side}'
                if wave_key in xsilmr_struct.dtype.names:
                    receiver_data[f'Side{side}'] = xsilmr_struct[wave_key]
                    
            self.xsilmr_data[receiver_idx] = receiver_data
            logger.info(f"接收器{receiver_idx}数据加载完成: 深度点数={len(receiver_data['Depth'])}")
            
        logger.info(f"XSILMR数据加载完成: 共{len(self.xsilmr_data)}个接收器")
        return self.xsilmr_data
    
    def filter_depth_range(self, min_depth: float = 2732.0, max_depth: float = 4132.0) -> Tuple[Dict, Dict]:
        """筛选指定深度范围的数据"""
        logger.info(f"筛选深度范围: {min_depth} - {max_depth} ft")
        
        # 筛选CAST数据
        cast_mask = (self.cast_data['Depth'] >= min_depth) & (self.cast_data['Depth'] <= max_depth)
        filtered_cast = {
            'Depth': self.cast_data['Depth'][cast_mask],
            'Zc': self.cast_data['Zc'][:, cast_mask]
        }
        
        # 筛选XSILMR数据
        filtered_xsilmr = {}
        for receiver_idx, data in self.xsilmr_data.items():
            mask = (data['Depth'] >= min_depth) & (data['Depth'] <= max_depth)
            filtered_data = {
                'Depth': data['Depth'][mask],
                'Tad': data['Tad']
            }
            
            # 筛选所有方位的波形数据
            for side in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                if f'Side{side}' in data:
                    filtered_data[f'Side{side}'] = data[f'Side{side}'][:, mask]
                    
            filtered_xsilmr[receiver_idx] = filtered_data
            
        logger.info(f"深度筛选完成: CAST点数={len(filtered_cast['Depth'])}, "
                   f"XSILMR点数={len(filtered_xsilmr[7]['Depth'])}")
        
        return filtered_cast, filtered_xsilmr
    
    def calculate_absolute_depths(self, xsilmr_data: Dict) -> Dict:
        """计算每个接收器的绝对深度"""
        logger.info("计算接收器绝对深度...")
        
        # 以第7个接收器为基准
        base_depths = xsilmr_data[7]['Depth']  # 基准深度
        
        for receiver_idx in range(1, 14):
            if receiver_idx in xsilmr_data:
                # 计算绝对深度: D_actual(i) = D_base + (7 - i) × 0.5 ft
                offset = (7 - receiver_idx) * 0.5
                absolute_depths = base_depths + offset
                xsilmr_data[receiver_idx]['AbsoluteDepth'] = absolute_depths
                
                logger.debug(f"接收器{receiver_idx}: 偏移量={offset} ft")
                
        logger.info("绝对深度计算完成")
        return xsilmr_data 