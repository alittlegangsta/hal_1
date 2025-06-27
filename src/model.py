"""
深度学习模型模块 - 混合输入CNN模型
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import logging
import joblib

logger = logging.getLogger(__name__)

class HybridChannelingModel:
    """混合输入窜槽预测模型"""
    
    def __init__(self, image_shape: Tuple[int, int] = (127, 1024), vector_dim: int = 8):
        """
        初始化模型
        
        Args:
            image_shape: 尺度图输入形状 (scales, time_samples)
            vector_dim: 数值特征向量维度
        """
        self.image_shape = image_shape
        self.vector_dim = vector_dim
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        # 设置随机种子
        tf.random.set_seed(42)
        np.random.seed(42)
        
        logger.info(f"模型初始化: 图像形状={image_shape}, 向量维度={vector_dim}")
    
    def build_model(self) -> Model:
        """构建混合输入模型"""
        logger.info("构建混合输入模型...")
        
        # 图像输入分支 (CNN)
        image_input = layers.Input(shape=(*self.image_shape, 1), name='image_input')
        
        # CNN层
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        cnn_features = layers.Dense(64, activation='relu', name='cnn_features')(x)
        
        # 数值输入分支
        vector_input = layers.Input(shape=(self.vector_dim,), name='vector_input')
        vector_features = layers.Dense(32, activation='relu', name='vector_features')(vector_input)
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
    
    def prepare_data(self, X_images: List, X_vectors: List, y_labels: List, 
                    test_size: float = 0.2) -> Tuple:
        """
        准备训练数据
        
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
        
        # 归一化图像数据
        X_images = X_images[..., np.newaxis]  # 添加通道维度
        X_images = X_images / (np.max(X_images) + 1e-8)  # 归一化到[0,1]
        
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
        训练模型
        
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
        
        # 预处理
        if image_features.ndim == 2:
            image_features = image_features[np.newaxis, ..., np.newaxis]
        elif image_features.ndim == 3:
            image_features = image_features[..., np.newaxis]
        
        image_features = image_features / (np.max(image_features) + 1e-8)
        
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
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失
        axes[0, 0].plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history['mae'], label='Training MAE')
        if 'val_mae' in self.history:
            axes[0, 1].plot(self.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # MSE
        axes[1, 0].plot(self.history['mse'], label='Training MSE')
        if 'val_mse' in self.history:
            axes[1, 0].plot(self.history['val_mse'], label='Validation MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        
        # 学习率 (如果可用)
        if 'lr' in self.history:
            axes[1, 1].plot(self.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate Data Not Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图保存至: {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        save_path: Optional[str] = None):
        """绘制预测结果对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 散点图
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0].set_xlabel('True Channeling Ratio')
        axes[0].set_ylabel('Predicted Channeling Ratio')
        axes[0].set_title('Predictions vs True Values')
        axes[0].grid(True, alpha=0.3)
        
        # 残差图
        residuals = y_pred - y_true
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Channeling Ratio')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Analysis')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图保存至: {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str, scaler_path: str):
        """保存模型和预处理器"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"模型保存至: {model_path}")
        logger.info(f"预处理器保存至: {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str):
        """加载模型和预处理器"""
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        logger.info(f"模型加载自: {model_path}")
        logger.info(f"预处理器加载自: {scaler_path}")
    
    def get_model_summary(self):
        """获取模型摘要"""
        if self.model is None:
            logger.warning("模型尚未构建")
            return None
        
        return self.model.summary() 