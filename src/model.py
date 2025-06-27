"""
深度学习模型模块 - 支持HDF5数据源的内存高效训练
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, Dict, List, Optional, Generator
import logging
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from .hdf5_manager import HDF5DataManager

logger = logging.getLogger(__name__)

class HDF5DataGenerator(tf.keras.utils.Sequence):
    """HDF5数据生成器 - 用于内存高效的模型训练"""
    
    def __init__(self, 
                 hdf5_path: str,
                 indices: List[int],
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        初始化HDF5数据生成器
        
        Args:
            hdf5_path: HDF5文件路径
            indices: 数据索引列表
            batch_size: 批次大小
            shuffle: 是否打乱数据
        """
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 创建HDF5管理器
        self.hdf5_manager = HDF5DataManager(hdf5_path, mode='r')
        
        self.on_epoch_end()
        
    def __len__(self):
        """返回每个epoch的批次数"""
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        """获取一个批次的数据"""
        # 计算批次索引范围
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        # 读取数据
        batch_data = self._load_batch_data(batch_indices)
        
        # 返回 (inputs, outputs) 格式
        X = [batch_data['image_features'], batch_data['vector_features']]
        y = batch_data['labels']
        
        return X, y
    
    def _load_batch_data(self, batch_indices: List[int]) -> Dict:
        """从HDF5文件加载批次数据"""
        with self.hdf5_manager.open_file() as f:
            batch_data = {
                'image_features': f['image_features'][batch_indices],
                'vector_features': f['vector_features'][batch_indices],
                'labels': f['labels'][batch_indices]
            }
        
        return batch_data
    
    def on_epoch_end(self):
        """在每个epoch结束时调用"""
        if self.shuffle:
            np.random.shuffle(self.indices)


class HybridChannelingModel:
    """混合输入窜槽检测模型 - 支持HDF5数据源"""
    
    def __init__(self, image_shape: Tuple[int, int] = (127, 1024), vector_dim: int = 8):
        """
        初始化模型
        
        Args:
            image_shape: 图像输入形状
            vector_dim: 向量输入维度
        """
        self.image_shape = image_shape
        self.vector_dim = vector_dim
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        logger.info(f"初始化混合窜槽检测模型: 图像形状={image_shape}, 向量维度={vector_dim}")
    
    def build_model(self) -> Model:
        """构建混合输入CNN模型"""
        logger.info("构建混合输入CNN模型...")
        
        # 图像输入分支 (尺度图)
        image_input = layers.Input(shape=(*self.image_shape, 1), name='image_input')
        
        # CNN特征提取
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # CNN特征后处理
        cnn_features = layers.Dense(128, activation='relu')(x)
        cnn_features = layers.Dropout(0.4)(cnn_features)
        
        # 数值特征输入分支
        vector_input = layers.Input(shape=(self.vector_dim,), name='vector_input')
        vector_features = layers.Dense(64, activation='relu')(vector_input)
        vector_features = layers.BatchNormalization()(vector_features)
        vector_features = layers.Dropout(0.3)(vector_features)
        
        # 特征融合
        merged = layers.Concatenate(name='feature_fusion')([cnn_features, vector_features])
        
        # 预测头
        x = layers.Dense(128, activation='relu')(merged)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # 输出层 (回归，使用Sigmoid确保输出在[0,1]范围)
        output = layers.Dense(1, activation='sigmoid', name='channeling_ratio')(x)
        
        # 创建模型
        model = Model(inputs=[image_input, vector_input], outputs=output, name='HybridChannelingModel')
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info(f"模型构建完成: 参数总数={model.count_params()}")
        return model
    
    def prepare_data_from_hdf5(self, 
                             hdf5_path: str,
                             test_size: float = 0.2,
                             val_size: float = 0.1,
                             random_seed: int = 42) -> Tuple:
        """
        从HDF5文件准备训练数据
        
        Args:
            hdf5_path: HDF5文件路径
            test_size: 测试集比例
            val_size: 验证集比例 
            random_seed: 随机种子
            
        Returns:
            (train_indices, val_indices, test_indices, total_samples)
        """
        logger.info("从HDF5文件准备训练数据...")
        
        # 创建HDF5管理器获取数据信息
        hdf5_manager = HDF5DataManager(hdf5_path, mode='r')
        data_info = hdf5_manager.get_data_info()
        total_samples = data_info['total_samples']
        
        logger.info(f"HDF5数据集信息:")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  图像特征形状: {data_info['image_shape']}")
        logger.info(f"  向量特征维度: {data_info['vector_dim']}")
        logger.info(f"  文件大小: {data_info['file_size_mb']:.2f} MB")
        
        # 生成随机索引
        np.random.seed(random_seed)
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)
        
        # 划分数据集
        n_test = int(total_samples * test_size)
        n_val = int(total_samples * val_size)
        n_train = total_samples - n_test - n_val
        
        test_indices = all_indices[:n_test].tolist()
        val_indices = all_indices[n_test:n_test+n_val].tolist()
        train_indices = all_indices[n_test+n_val:].tolist()
        
        logger.info(f"数据划分完成:")
        logger.info(f"  训练集: {len(train_indices)} 样本 ({len(train_indices)/total_samples:.1%})")
        logger.info(f"  验证集: {len(val_indices)} 样本 ({len(val_indices)/total_samples:.1%})")
        logger.info(f"  测试集: {len(test_indices)} 样本 ({len(test_indices)/total_samples:.1%})")
        
        # 计算数值特征的标准化参数
        self._fit_scaler_from_hdf5(hdf5_path, train_indices)
        
        return train_indices, val_indices, test_indices, total_samples
    
    def _fit_scaler_from_hdf5(self, hdf5_path: str, train_indices: List[int]):
        """从HDF5文件的训练数据拟合标准化器"""
        logger.info("计算数值特征标准化参数...")
        
        hdf5_manager = HDF5DataManager(hdf5_path, mode='r')
        
        # 分批读取训练数据来计算统计量
        batch_size = 1000
        all_vector_features = []
        
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            batch_data = hdf5_manager.read_batch(0, len(batch_indices))  # 这里需要修改read_batch方法
            
            # 临时解决方案：直接读取指定索引的数据
            with hdf5_manager.open_file() as f:
                vector_features = f['vector_features'][batch_indices]
                all_vector_features.append(vector_features)
        
        # 合并所有批次的数据
        train_vector_features = np.vstack(all_vector_features)
        
        # 拟合标准化器
        self.scaler.fit(train_vector_features)
        
        logger.info("数值特征标准化参数计算完成")
    
    def train_from_hdf5(self, 
                       hdf5_path: str,
                       epochs: int = 100,
                       batch_size: int = 32,
                       test_size: float = 0.2,
                       val_size: float = 0.1) -> Dict:
        """
        从HDF5文件训练模型
        
        Args:
            hdf5_path: HDF5文件路径
            epochs: 训练轮数
            batch_size: 批次大小
            test_size: 测试集比例
            val_size: 验证集比例
            
        Returns:
            训练历史
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"开始从HDF5文件训练模型: {hdf5_path}")
        
        # 准备数据
        train_indices, val_indices, test_indices, total_samples = self.prepare_data_from_hdf5(
            hdf5_path, test_size, val_size)
        
        # 创建数据生成器
        train_generator = HDF5DataGenerator(
            hdf5_path, train_indices, batch_size, shuffle=True)
        val_generator = HDF5DataGenerator(
            hdf5_path, val_indices, batch_size, shuffle=False)
        
        # 设置回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'outputs/models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # 创建输出目录
        Path("outputs/models").mkdir(parents=True, exist_ok=True)
        
        # 训练模型
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        logger.info("模型训练完成")
        
        # 评估模型
        if len(test_indices) > 0:
            test_generator = HDF5DataGenerator(
                hdf5_path, test_indices, batch_size, shuffle=False)
            
            logger.info("评估测试集性能...")
            test_metrics = self.model.evaluate(test_generator, verbose=0)
            
            logger.info("测试集评估结果:")
            for name, value in zip(self.model.metrics_names, test_metrics):
                logger.info(f"  {name}: {value:.4f}")
        
        return self.history
    
    def prepare_data(self, X_images: List, X_vectors: List, y_labels: List, 
                    test_size: float = 0.2) -> Tuple:
        """
        准备训练数据 (传统方法，兼容性保持)
        
        Args:
            X_images: 图像特征列表
            X_vectors: 数值特征列表  
            y_labels: 标签列表
            test_size: 测试集比例
            
        Returns:
            训练和测试数据元组
        """
        logger.info("准备训练数据...")
        
        # 转换为numpy数组
        X_images = np.array(X_images)
        X_vectors = np.array(X_vectors)
        y_labels = np.array(y_labels)
        
        logger.info(f"数据形状: 图像={X_images.shape}, 向量={X_vectors.shape}, 标签={y_labels.shape}")
        
        # 逐样本归一化图像数据
        X_images = X_images[..., np.newaxis]  # 添加通道维度
        
        # 对每个尺度图独立进行归一化
        for i in range(len(X_images)):
            max_val = np.max(X_images[i])
            if max_val > 1e-8:
                X_images[i] = X_images[i] / max_val
            else:
                logger.warning(f"样本 {i} 的最大值过小: {max_val}")
        
        logger.info("完成逐样本图像归一化")
        
        # 标准化数值特征
        X_vectors = self.scaler.fit_transform(X_vectors)
        
        # 划分训练和测试集
        indices = np.arange(len(X_images))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        X_images_train, X_images_test = X_images[train_idx], X_images[test_idx]
        X_vectors_train, X_vectors_test = X_vectors[train_idx], X_vectors[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        logger.info(f"数据划分完成: 训练集={len(train_idx)}, 测试集={len(test_idx)}")
        
        return (X_images_train, X_vectors_train, y_train), (X_images_test, X_vectors_test, y_test)
    
    def train(self, train_data: Tuple, val_data: Tuple = None, 
             epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        训练模型 (传统方法，兼容性保持)
        
        Args:
            train_data: 训练数据 (X_images, X_vectors, y)
            val_data: 验证数据 (可选)
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练历史
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"开始训练模型: epochs={epochs}, batch_size={batch_size}")
        
        X_images_train, X_vectors_train, y_train = train_data
        
        # 设置回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_data else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # 训练模型
        validation_data = None
        if val_data is not None:
            X_images_val, X_vectors_val, y_val = val_data
            validation_data = ([X_images_val, X_vectors_val], y_val)
        
        history = self.model.fit(
            x=[X_images_train, X_vectors_train],
            y=y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        logger.info("模型训练完成")
        
        return self.history
    
    def evaluate(self, test_data: Tuple) -> Dict:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            
        Returns:
            评估指标字典
        """
        logger.info("评估模型性能...")
        
        X_images_test, X_vectors_test, y_test = test_data
        
        # 预测
        y_pred = self.model.predict([X_images_test, X_vectors_test])
        y_pred = y_pred.flatten()
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        logger.info(f"评估结果: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return metrics, y_pred
    
    def predict_channeling(self, image_features: np.ndarray, 
                          vector_features: np.ndarray) -> np.ndarray:
        """
        预测窜槽比例
        
        Args:
            image_features: 图像特征
            vector_features: 数值特征
            
        Returns:
            预测的窜槽比例
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 预处理图像特征
        if image_features.ndim == 2:
            image_features = image_features[np.newaxis, ..., np.newaxis]
        elif image_features.ndim == 3:
            image_features = image_features[..., np.newaxis]
        
        # 逐样本归一化图像数据（与训练时保持一致）
        for i in range(len(image_features)):
            max_val = np.max(image_features[i])
            if max_val > 1e-8:
                image_features[i] = image_features[i] / max_val
        
        # 预处理数值特征
        if vector_features.ndim == 1:
            vector_features = vector_features.reshape(1, -1)
        
        vector_features = self.scaler.transform(vector_features)
        
        # 预测
        predictions = self.model.predict([image_features, vector_features])
        return predictions.flatten()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        if self.history is None:
            logger.warning("没有训练历史可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training History', fontsize=16)
        
        # 训练和验证损失
        axes[0, 0].plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(self.history['mae'], label='Training MAE')
        if 'val_mae' in self.history:
            axes[0, 1].plot(self.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率 (如果有)
        if 'lr' in self.history:
            axes[1, 0].plot(self.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # MSE
        axes[1, 1].plot(self.history['mse'], label='Training MSE')
        if 'val_mse' in self.history:
            axes[1, 1].plot(self.history['val_mse'], label='Validation MSE')
        axes[1, 1].set_title('Mean Squared Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图表已保存: {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        save_path: Optional[str] = None):
        """绘制预测结果对比"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 预测 vs 真实值散点图
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[0].set_xlabel('True Channeling Ratio')
        axes[0].set_ylabel('Predicted Channeling Ratio')
        axes[0].set_title('Predicted vs True Values')
        axes[0].grid(True, alpha=0.3)
        
        # 残差分布
        residuals = y_pred - y_true
        axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals (Predicted - True)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图表已保存: {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str, scaler_path: str):
        """保存模型和标准化器"""
        if self.model is not None:
            self.model.save(model_path)
            logger.info(f"模型已保存: {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"标准化器已保存: {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str):
        """加载模型和标准化器"""
        self.model = keras.models.load_model(model_path)
        logger.info(f"模型已加载: {model_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"标准化器已加载: {scaler_path}")
    
    def get_model_summary(self):
        """获取模型摘要"""
        if self.model is not None:
            return self.model.summary()
        else:
            return "模型尚未构建" 