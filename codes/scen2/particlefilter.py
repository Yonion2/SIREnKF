# particlefilter.py
# 粒子滤波
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class ParticleFilter:
    def __init__(self, num_particles, state_dim, process_noise, measurement_noise):
        """
        初始化粒子滤波器
        :param num_particles: 粒子数量
        :param state_dim: 状态维度
        :param process_noise: 过程噪声的标准差
        :param measurement_noise: 观测噪声的标准差
        """
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(num_particles, state_dim)  # 随机初始化粒子
        self.weights = np.ones(num_particles) / num_particles  # 初始化权重

    def predict(self):
        """
        预测阶段：根据动态模型更新粒子
        """
        # 假设动态模型为 x_t = x_{t-1} + 过程噪声
        noise = np.random.randn(self.num_particles, self.state_dim) * self.process_noise
        self.particles += noise

    def update(self, measurement):
        """
        更新阶段：根据观测模型更新粒子权重
        :param measurement: 当前观测值
        """
        # 假设观测模型为 z_t = x_t + 观测噪声
        predicted_measurements = self.particles
        residuals = measurement - predicted_measurements
        likelihood = np.exp(-0.5 * (residuals / self.measurement_noise) ** 2)
        self.weights *= likelihood
        self.weights /= np.sum(self.weights)  # 归一化权重

    def resample(self):
        """
        重采样阶段：根据权重重新采样粒子
        """
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        """
        输出状态估计
        """
        return np.sum(self.particles * self.weights[:, None], axis=0)