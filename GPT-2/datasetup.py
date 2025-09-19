import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

class MARSDatasetPreprocessor:
    """MARS 데이터셋 전처리 클래스"""
    
    def __init__(self, data_dir='./combined_dataset'):
        self.data_dir = data_dir
        self.X = None
        self.y = None
        self.X_scaler = None
        self.y_scaler = None
        
    def load_raw_data(self):
        """원본 데이터 로드"""
        print("=== 원본 데이터 로딩 ===")
        
        # X 데이터 로드 및 결합
        default_inputs = np.load(os.path.join(self.data_dir, 'default_mars_inputs.npy'))
        passive_safety = np.load(os.path.join(self.data_dir, 'passive_safety_system_functionalities.npy'))
        
        print(f"default_mars_inputs 형태: {default_inputs.shape}")
        print(f"passive_safety 형태: {passive_safety.shape}")
        
        # X 결합 (2999, 15) + (2999, 3) = (2999, 18)
        self.X = np.concatenate([default_inputs, passive_safety], axis=1)
        
        # y 데이터 로드
        self.y = np.load(os.path.join(self.data_dir, 'pcts.npy'))
        
        print(f"결합된 X 형태: {self.X.shape}")
        print(f"y 형태: {self.y.shape}")
        
        # 기본 통계 확인
        self._print_basic_stats()
        
        return self.X, self.y
    
    def _print_basic_stats(self):
        """기본 통계 출력"""
        print("\n=== 기본 통계 ===")
        print("X 통계:")
        print(f"  평균: {self.X.mean():.4f}")
        print(f"  표준편차: {self.X.std():.4f}")
        print(f"  최솟값: {self.X.min():.4f}")
        print(f"  최댓값: {self.X.max():.4f}")
        print(f"  결측값: {np.isnan(self.X).sum()}")
        
        print("\ny 통계:")
        print(f"  평균: {self.y.mean():.4f}")
        print(f"  표준편차: {self.y.std():.4f}")
        print(f"  최솟값: {self.y.min():.4f}")
        print(f"  최댓값: {self.y.max():.4f}")
        print(f"  결측값: {np.isnan(self.y).sum()}")
    
    def check_outliers(self, plot=True):
        """이상치 검사"""
        print("\n=== 이상치 분석 ===")
        
        # X 이상치 (IQR 방법)
        X_outliers = []
        for i in range(self.X.shape[1]):
            Q1 = np.percentile(self.X[:, i], 25)
            Q3 = np.percentile(self.X[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.sum((self.X[:, i] < lower_bound) | (self.X[:, i] > upper_bound))
            X_outliers.append(outliers)
            
        print(f"X 피처별 이상치 개수: {X_outliers}")
        print(f"X 총 이상치: {sum(X_outliers)}")
        
        # y 이상치
        y_flat = self.y.flatten()
        Q1 = np.percentile(y_flat, 25)
        Q3 = np.percentile(y_flat, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        y_outliers = np.sum((y_flat < lower_bound) | (y_flat > upper_bound))
        print(f"y 이상치 개수: {y_outliers} / {len(y_flat)} ({y_outliers/len(y_flat)*100:.2f}%)")
        
        if plot:
            self._plot_distributions()
    
    def _plot_distributions(self):
        """데이터 분포 시각화"""
        plt.figure(figsize=(15, 10))
        
        # X 분포 (처음 6개 피처만)
        for i in range(min(6, self.X.shape[1])):
            plt.subplot(3, 4, i+1)
            plt.hist(self.X[:, i], bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'X Feature {i+1}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        # y 분포 (전체)
        plt.subplot(3, 4, 7)
        plt.hist(self.y.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
        plt.title('y Distribution (All)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # y 분포 (샘플별 평균)
        plt.subplot(3, 4, 8)
        y_means = self.y.mean(axis=1)
        plt.hist(y_means, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.title('y Sample Means')
        plt.xlabel('Mean Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 시계열 예시
        plt.subplot(3, 4, 9)
        for i in range(3):
            plt.plot(self.y[i][:100], alpha=0.7, label=f'Sample {i+1}')
        plt.title('y Time Series Examples (First 100 points)')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # X 상관관계 히트맵 (서브샘플)
        plt.subplot(3, 4, 10)
        if self.X.shape[1] <= 10:
            corr_matrix = np.corrcoef(self.X.T)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('X Features Correlation')
        else:
            # 너무 많으면 처음 10개만
            corr_matrix = np.corrcoef(self.X[:, :10].T)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('X Features Correlation (First 10)')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, X_method='minmax', y_method='minmax', 
                       handle_outliers=False, outlier_method='clip'):
        """데이터 전처리"""
        print(f"\n=== 데이터 전처리 (X: {X_method}, y: {y_method}) ===")
        
        # 이상치 처리 (선택적)
        if handle_outliers:
            self.X, self.y = self._handle_outliers(self.X, self.y, method=outlier_method)
        
        # X 정규화
        if X_method == 'standard':
            self.X_scaler = StandardScaler()
        elif X_method == 'minmax':
            self.X_scaler = MinMaxScaler()
        elif X_method == 'robust':
            self.X_scaler = RobustScaler()
        else:
            raise ValueError("X_method는 'standard', 'minmax', 'robust' 중 하나여야 합니다")
        
        X_normalized = self.X_scaler.fit_transform(self.X)
        
        # y 정규화
        if y_method == 'standard':
            self.y_scaler = StandardScaler()
        elif y_method == 'minmax':
            self.y_scaler = MinMaxScaler()
        elif y_method == 'robust':
            self.y_scaler = RobustScaler()
        else:
            raise ValueError("y_method는 'standard', 'minmax', 'robust' 중 하나여야 합니다")
        
        # y는 2D로 reshape 필요 (scaler가 2D를 기대함)
        y_reshaped = self.y.reshape(-1, 1)
        y_normalized_flat = self.y_scaler.fit_transform(y_reshaped)
        y_normalized = y_normalized_flat.reshape(self.y.shape)
        
        # 전처리 후 통계
        print("전처리 후 통계:")
        print(f"X - 평균: {X_normalized.mean():.4f}, 표준편차: {X_normalized.std():.4f}")
        print(f"X - 범위: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
        print(f"y - 평균: {y_normalized.mean():.4f}, 표준편차: {y_normalized.std():.4f}")
        print(f"y - 범위: [{y_normalized.min():.4f}, {y_normalized.max():.4f}]")
        
        return X_normalized, y_normalized
    
    def _handle_outliers(self, X, y, method='clip'):
        """이상치 처리"""
        print(f"이상치 처리 중 (방법: {method})")
        
        if method == 'clip':
            # X 이상치 클리핑
            for i in range(X.shape[1]):
                Q1 = np.percentile(X[:, i], 25)
                Q3 = np.percentile(X[:, i], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X[:, i] = np.clip(X[:, i], lower_bound, upper_bound)
            
            # y 이상치 클리핑
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            y = np.clip(y, lower_bound, upper_bound)
            
        elif method == 'remove':
            # 이상치 샘플 제거 (구현 가능하지만 복잡)
            print("이상치 샘플 제거는 구현되지 않았습니다. 'clip' 사용")
            return self._handle_outliers(X, y, method='clip')
        
        return X, y
    
    def save_processed_data(self, X_processed, y_processed, 
                           output_file='preprocessed_data.pkl'):
        """전처리된 데이터 저장"""
        print(f"\n=== 데이터 저장: {output_file} ===")
        
        data_dict = {
            'X': X_processed,
            'y': y_processed,
            'X_scaler': self.X_scaler,
            'y_scaler': self.y_scaler,
            'X_original_shape': self.X.shape,
            'y_original_shape': self.y.shape,
            'preprocessing_info': {
                'X_method': type(self.X_scaler).__name__,
                'y_method': type(self.y_scaler).__name__,
                'X_original_stats': {
                    'mean': self.X.mean(),
                    'std': self.X.std(),
                    'min': self.X.min(),
                    'max': self.X.max()
                },
                'y_original_stats': {
                    'mean': self.y.mean(),
                    'std': self.y.std(),
                    'min': self.y.min(),
                    'max': self.y.max()
                }
            }
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"저장 완료:")
        print(f"  X: {X_processed.shape}")
        print(f"  y: {y_processed.shape}")
        print(f"  파일 크기: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    def load_processed_data(self, input_file='preprocessed_data.pkl'):
        """전처리된 데이터 로드"""
        print(f"전처리된 데이터 로딩: {input_file}")
        
        with open(input_file, 'rb') as f:
            data_dict = pickle.load(f)
        
        X = data_dict['X']
        y = data_dict['y']
        self.X_scaler = data_dict['X_scaler']
        self.y_scaler = data_dict['y_scaler']
        
        print(f"로딩 완료: X {X.shape}, y {y.shape}")
        return X, y
    
    def inverse_transform(self, X_processed=None, y_processed=None):
        """정규화 역변환"""
        results = {}
        
        if X_processed is not None and self.X_scaler is not None:
            results['X'] = self.X_scaler.inverse_transform(X_processed)
        
        if y_processed is not None and self.y_scaler is not None:
            if y_processed.ndim == 2:
                y_reshaped = y_processed.reshape(-1, 1)
                y_inverse_flat = self.y_scaler.inverse_transform(y_reshaped)
                results['y'] = y_inverse_flat.reshape(y_processed.shape)
            else:
                results['y'] = self.y_scaler.inverse_transform(y_processed.reshape(-1, 1)).flatten()
        
        return results

def main():
    """메인 실행 함수"""
    # 전처리기 초기화
    preprocessor = MARSDatasetPreprocessor('./combined_dataset')
    
    # 1. 원본 데이터 로드
    X_raw, y_raw = preprocessor.load_raw_data()
    
    # 2. 이상치 분석
    preprocessor.check_outliers(plot=True)
    
    # 3. 전처리 수행
    X_processed, y_processed = preprocessor.preprocess_data(
        X_method='minmax',      # X는 MinMax 정규화
        y_method='minmax',      # y도 MinMax 정규화
        handle_outliers=True,   # 이상치 처리
        outlier_method='clip'   # 클리핑 방식
    )
    
    # 4. 전처리된 데이터 저장
    preprocessor.save_processed_data(X_processed, y_processed, 
                                   'preprocessed_mars_data.pkl')
    
    print("\n=== 전처리 완료! ===")
    print("다음 단계: Conditional Spline Flow 모델 훈련")

if __name__ == "__main__":
    main()