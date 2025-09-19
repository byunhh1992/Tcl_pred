import pickle as pkl
import numpy as np
import torch
from transformers import GPT2LMHeadModel
from model_with_training import ModelConfig
import matplotlib.pyplot as plt
import os
import time


def predict_single(X: np.ndarray, model: GPT2LMHeadModel, config, device) -> np.ndarray:
    """단일 X에 대한 y예측 함수"""
    # 입력을 토큰으로 변환
    X_tokens = (X * (config.vocab_size - 1)).astype(int).tolist()

    # SEP 토큰 추가
    input_sequence = X_tokens + [config.vocab_size]  # 1000이 SEP 토큰
    input_ids = torch.tensor([input_sequence]).to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=len(input_sequence) + config.output_dim,
            do_sample=True,
            pad_token_id=0,
        )
    
    # 결과 추출
    generated = outputs[0].cpu().tolist()
    sep_index = generated.index(config.vocab_size)
    y_tokens = generated[sep_index + 1:][:config.output_dim]

    # 토큰을 0-1 float로 변환
    y_pred = np.array(y_tokens) / (config.vocab_size - 1)
    return y_pred

def predict_single_case_single_time(X_val: np.ndarray, y_val: np.ndarray, model: GPT2LMHeadModel, config, device) -> None:
    """예측값, 계측값 비교 그림까지 그려서 저장; 하나의 시나리오 대상 한 번의 예측"""
    # 예측값 계산
    y_prd = predict_single(X_val, model, config, device)

    # 예측 vs 계측 그림 비교
    plt.figure(figsize=(12, 6))
    plt.plot(y_val, 'b-', label='y_val', linewidth=2)
    plt.plot(y_prd, 'r--', label='y_prd', linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 그림 저장
    os.makedirs('./validation_results', exist_ok=True)
    plt.savefig('./validation_results/predict_single_case_single_time.png', dpi=300)

def predict_single_case_multiple_times(X_val: np.ndarray, y_val: np.ndarray, model: GPT2LMHeadModel, config, device, n: int) -> None:
    """예측값, 계측값 비교 그림까지 그려서 저장; 하나의 시나리오 대상 여러 번의 예측"""
    # 예측값 계산
    start_time = time.time()
    y_prds = []
    for i in range(n):
        y_prd = predict_single(X_val, model, config, device)
        if y_prd.shape == y_val.shape:
            y_prds.append(y_prd)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Predictions (n={n}, elapsed time: {elapsed_time:.2f}s)")
    
    # 예측값 먼저 그리고 계측값 그림
    plt.figure(figsize=(12, 6))
    for i in range(len(y_prds)):
        plt.plot(y_prds[i], 'b-', linewidth=2, alpha=0.1)
    plt.plot(y_val, 'r--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 예측값과 계측값 겹쳐서 그린 결과치 저장
    os.makedirs('./validation_results', exist_ok=True)
    plt.savefig('./validation_results/predict_single_case_multiple_times.png', dpi=300)

    # ====================================================================================
    # 예측값들의 통계 계산
    y_prds_array = np.array(y_prds)
    y_mean = np.mean(y_prds_array, axis=0)
    y_std = np.std(y_prds_array, axis=0)

    # 평균과 분산을 이용한 그래프
    plt.figure(figsize=(12, 8))
    plt.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, alpha=0.2, color='blue', label='±1σ') # 평균과 표준편차 영역 그리기
    plt.plot(y_mean, 'b-', linewidth=3, label='Mean prediction') # 평균값 그리기
    plt.plot(y_val, 'r--', linewidth=2, label='True value') # 계측값 그리기
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 통계 분석 그래프 저장
    plt.savefig('./validation_results/predict_single_case_statistic_results.png', dpi=300)


def evaluate_all_samples_and_select_best(X_val: np.ndarray, y_val: np.ndarray, model: GPT2LMHeadModel, config, device, n_predictions: int = 10, n_best: int = 8) -> list:
    """전체 테스트 샘플에 대해 n_predictions번씩 예측하여 성능이 가장 좋은 n_best개 샘플의 인덱스 반환"""
    print(f"전체 {len(X_val)}개 샘플에 대해 {n_predictions}번씩 예측 중...")
    
    sample_errors = []
    
    for i in range(len(X_val)):
        if i % 50 == 0:  # 진행상황 출력
            print(f"진행률: {i}/{len(X_val)}")
            
        predictions = []
        for j in range(n_predictions):
            y_pred = predict_single(X_val[i], model, config, device)
            if y_pred.shape == y_val[i].shape:
                predictions.append(y_pred)
        
        if predictions:  # 예측이 성공한 경우만
            predictions_array = np.array(predictions)
            mean_prediction = np.mean(predictions_array, axis=0)
            
            # MSE 계산
            mse = np.mean((mean_prediction - y_val[i]) ** 2)
            sample_errors.append((i, mse))
        else:
            sample_errors.append((i, float('inf')))  # 실패한 경우 무한대 오차
    
    # 오차가 작은 순서로 정렬하여 상위 n_best개 선택
    sample_errors.sort(key=lambda x: x[1])
    best_sample_indices = [idx for idx, _ in sample_errors[:n_best]]
    
    print(f"가장 성능이 좋은 {n_best}개 샘플 선택 완료")
    print(f"선택된 샘플 인덱스: {best_sample_indices}")
    print(f"해당 샘플들의 MSE: {[error for _, error in sample_errors[:n_best]]}")
    
    return best_sample_indices


def predict_best_samples_extensively(X_val: np.ndarray, y_val: np.ndarray, model: GPT2LMHeadModel, config, device, best_indices: list, n_extensive: int = 100) -> None:
    """선별된 최고 성능 샘플들에 대해 extensive prediction 수행 및 시각화"""
    print(f"선별된 {len(best_indices)}개 샘플에 대해 {n_extensive}번씩 예측 중...")
    
    # 서브플롯 설정 (2x4 그리드)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for plot_idx, sample_idx in enumerate(best_indices):
        print(f"샘플 {sample_idx} 처리 중... ({plot_idx + 1}/{len(best_indices)})")
        
        start_time = time.time()
        predictions = []
        
        for i in range(n_extensive):
            y_pred = predict_single(X_val[sample_idx], model, config, device)
            if y_pred.shape == y_val[sample_idx].shape:
                predictions.append(y_pred)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"샘플 {sample_idx} 완료 (소요시간: {elapsed_time:.2f}s)")
        
        if predictions:
            predictions_array = np.array(predictions)
            y_mean = np.mean(predictions_array, axis=0)
            y_std = np.std(predictions_array, axis=0)
            
            # 각 서브플롯에 결과 그리기
            ax = axes[plot_idx]
            ax.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, 
                          alpha=0.2, color='blue', label='±1σ')
            ax.plot(y_mean, 'b-', linewidth=2, label='Mean prediction')
            ax.plot(y_val[sample_idx], 'r--', linewidth=2, label='True value')
            ax.set_title(f'Sample {sample_idx} (n={len(predictions)})', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # 그림 저장
    os.makedirs('./validation_results', exist_ok=True)
    plt.savefig('./validation_results/best_samples_extensive_predictions.png', dpi=300, bbox_inches='tight')
    print("베스트 샘플들의 extensive prediction 결과 저장 완료")
    
    # 개별 샘플별로도 저장
    for plot_idx, sample_idx in enumerate(best_indices):
        predictions = []
        for i in range(n_extensive):
            y_pred = predict_single(X_val[sample_idx], model, config, device)
            if y_pred.shape == y_val[sample_idx].shape:
                predictions.append(y_pred)
        
        if predictions:
            predictions_array = np.array(predictions)
            y_mean = np.mean(predictions_array, axis=0)
            y_std = np.std(predictions_array, axis=0)
            
            plt.figure(figsize=(12, 8))
            plt.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, 
                           alpha=0.2, color='blue', label='±1σ')
            plt.plot(y_mean, 'b-', linewidth=3, label='Mean prediction')
            plt.plot(y_val[sample_idx], 'r--', linewidth=2, label='True value')
            plt.title(f'Sample {sample_idx} - Extensive Predictions (n={len(predictions)})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(f'./validation_results/sample_{sample_idx}_extensive_prediction.png', dpi=300)
            plt.close()
    
    print("개별 샘플별 그래프 저장 완료")


def predict_specific_sample_multiple_times(X_val: np.ndarray, y_val: np.ndarray, model: GPT2LMHeadModel, config, device, sample_idx: int, n: int) -> None:
    """특정 샘플에 대해 여러번 시뮬레이션 수행 및 시각화"""
    print(f"Sample {sample_idx}에 대해 {n}번 시뮬레이션 수행 중...")
    
    # 해당 샘플의 데이터 추출
    X_sample = X_val[sample_idx]
    y_sample = y_val[sample_idx]
    
    # 예측값 계산
    start_time = time.time()
    y_prds = []
    for i in range(n):
        if i % 20 == 0:  # 진행상황 출력
            print(f"진행률: {i}/{n}")
        y_prd = predict_single(X_sample, model, config, device)
        if y_prd.shape == y_sample.shape:
            y_prds.append(y_prd)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Sample {sample_idx} 예측 완료 (n={len(y_prds)}, 소요시간: {elapsed_time:.2f}s)")
    
    # 예측값들의 통계 계산
    y_prds_array = np.array(y_prds)
    y_mean = np.mean(y_prds_array, axis=0)
    y_std = np.std(y_prds_array, axis=0)
    
    # MSE 계산
    mse = np.mean((y_mean - y_sample) ** 2)
    print(f"Sample {sample_idx} MSE: {mse:.6f}")
    
    # 1. 모든 예측값과 실제값을 겹쳐서 그린 그래프
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 모든 예측값과 실제값
    plt.subplot(2, 1, 1)
    for i in range(len(y_prds)):
        plt.plot(y_prds[i], 'b-', linewidth=1, alpha=0.05)
    plt.plot(y_sample, 'r-', linewidth=3, label='True value')
    plt.title(f'Sample {sample_idx} - All Predictions (n={len(y_prds)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: 평균과 표준편차
    plt.subplot(2, 1, 2)
    plt.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, 
                     alpha=0.2, color='blue', label='±1σ')
    plt.plot(y_mean, 'b-', linewidth=3, label='Mean prediction')
    plt.plot(y_sample, 'r-', linewidth=3, label='True value')
    plt.title(f'Sample {sample_idx} - Statistical Summary (MSE: {mse:.6f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 그림 저장
    os.makedirs('./validation_results', exist_ok=True)
    plt.savefig(f'./validation_results/sample_{sample_idx}_multiple_simulations.png', dpi=300, bbox_inches='tight')
    print(f"Sample {sample_idx} 시뮬레이션 결과 저장 완료")
    
    # 2. 통계 정보만 별도로 그린 상세 그래프
    plt.figure(figsize=(12, 8))
    plt.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, 
                     alpha=0.2, color='blue', label='±1σ')
    plt.fill_between(range(len(y_mean)), y_mean - 2*y_std, y_mean + 2*y_std, 
                     alpha=0.1, color='blue', label='±2σ')
    plt.plot(y_mean, 'b-', linewidth=3, label='Mean prediction')
    plt.plot(y_sample, 'r-', linewidth=3, label='True value')
    plt.title(f'Sample {sample_idx} - Detailed Statistical Analysis\n'
              f'Predictions: {len(y_prds)}, MSE: {mse:.6f}, Elapsed time: {elapsed_time:.2f}s')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'./validation_results/sample_{sample_idx}_statistical_detail.png', dpi=300, bbox_inches='tight')
    print(f"Sample {sample_idx} 상세 통계 그래프 저장 완료")


def predict_specific_sample_filtered(X_val: np.ndarray, y_val: np.ndarray, model: GPT2LMHeadModel, config, device, sample_idx: int, n_total: int = 200, n_keep: int = 100) -> None:
    """특정 샘플에 대해 n_total번 예측한 후 평균에 가까운 n_keep개만 선별하여 시각화"""
    print(f"Sample {sample_idx}에 대해 {n_total}번 시뮬레이션 수행 후 상위 {n_keep}개 선별...")
    
    # 해당 샘플의 데이터 추출
    X_sample = X_val[sample_idx]
    y_sample = y_val[sample_idx]
    
    # 1단계: n_total번 예측값 계산
    start_time = time.time()
    y_prds = []
    for i in range(n_total):
        if i % 20 == 0:  # 진행상황 출력
            print(f"예측 진행률: {i}/{n_total}")
        y_prd = predict_single(X_sample, model, config, device)
        if y_prd.shape == y_sample.shape:
            y_prds.append(y_prd)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"전체 {len(y_prds)}개 예측 완료 (소요시간: {elapsed_time:.2f}s)")
    
    # 2단계: 초기 평균 계산
    y_prds_array = np.array(y_prds)
    y_initial_mean = np.mean(y_prds_array, axis=0)
    
    # 3단계: 각 예측값과 초기 평균 간의 거리 계산 (MSE 기준)
    distances = []
    for i, y_pred in enumerate(y_prds):
        distance = np.mean((y_pred - y_initial_mean) ** 2)
        distances.append((i, distance))
    
    # 4단계: 거리가 가까운 순서로 정렬하여 상위 n_keep개 선택
    distances.sort(key=lambda x: x[1])
    selected_indices = [idx for idx, _ in distances[:n_keep]]
    selected_predictions = [y_prds[i] for i in selected_indices]
    
    print(f"평균에 가까운 상위 {len(selected_predictions)}개 예측값 선별 완료")
    print(f"선별된 예측값들의 평균 거리: {np.mean([dist for _, dist in distances[:n_keep]]):.6f}")
    print(f"제외된 예측값들의 평균 거리: {np.mean([dist for _, dist in distances[n_keep:]]):.6f}")
    
    # 5단계: 선별된 예측값들의 통계 계산
    selected_array = np.array(selected_predictions)
    y_filtered_mean = np.mean(selected_array, axis=0)
    y_filtered_std = np.std(selected_array, axis=0)
    
    # MSE 계산
    mse_filtered = np.mean((y_filtered_mean - y_sample) ** 2)
    mse_initial = np.mean((y_initial_mean - y_sample) ** 2)
    
    print(f"Sample {sample_idx} 전체 평균 MSE: {mse_initial:.6f}")
    print(f"Sample {sample_idx} 필터링 후 MSE: {mse_filtered:.6f}")
    
    # 6단계: 선별된 예측값들만 시각화 (두 번째 figure만)
    plt.figure(figsize=(12, 8))
    for pred in selected_predictions:
        plt.plot(pred, 'b-', linewidth=1, alpha=0.3)
    plt.plot(y_sample, 'r-', linewidth=3, label='True value')
    plt.ylabel('Normalized PCT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 그림 저장
    os.makedirs('./validation_results', exist_ok=True)
    plt.savefig(f'./validation_results/sample_{sample_idx}_filtered_predictions.png', dpi=300, bbox_inches='tight')
    print(f"Sample {sample_idx} 필터링된 예측 결과 저장 완료")
    
    # 7단계: 상세 통계 그래프 (필터링된 결과만) - 제목 제거, y축 라벨 변경
    plt.figure(figsize=(12, 8))
    plt.fill_between(range(len(y_filtered_mean)), y_filtered_mean - y_filtered_std, y_filtered_mean + y_filtered_std, 
                     alpha=0.2, color='blue', label='±1σ')
    plt.fill_between(range(len(y_filtered_mean)), y_filtered_mean - 2*y_filtered_std, y_filtered_mean + 2*y_filtered_std, 
                     alpha=0.1, color='blue', label='±2σ')
    plt.plot(y_filtered_mean, 'b-', linewidth=3, label='Filtered mean prediction')
    plt.plot(y_sample, 'r-', linewidth=3, label='True value')
    plt.xlabel('Time step')
    plt.ylabel('Normalized PCT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'./validation_results/sample_{sample_idx}_filtered_detail.png', dpi=300, bbox_inches='tight')
    print(f"Sample {sample_idx} 필터링된 상세 통계 그래프 저장 완료")


def predict_specific_sample_weighted(X_val: np.ndarray, y_val: np.ndarray, model: GPT2LMHeadModel, config, device, sample_idx: int, n_total: int = 200) -> None:
    """특정 샘플에 대해 n_total번 예측한 후 평균과의 거리에 따라 가중치를 부여하여 시각화"""
    print(f"Sample {sample_idx}에 대해 {n_total}번 시뮬레이션 후 가중평균 계산...")
    
    # 해당 샘플의 데이터 추출
    X_sample = X_val[sample_idx]
    y_sample = y_val[sample_idx]
    
    # 1단계: n_total번 예측값 계산
    start_time = time.time()
    y_prds = []
    for i in range(n_total):
        if i % 20 == 0:  # 진행상황 출력
            print(f"예측 진행률: {i}/{n_total}")
        y_prd = predict_single(X_sample, model, config, device)
        if y_prd.shape == y_sample.shape:
            y_prds.append(y_prd)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"전체 {len(y_prds)}개 예측 완료 (소요시간: {elapsed_time:.2f}s)")
    
    # 2단계: 초기 평균 계산
    y_prds_array = np.array(y_prds)
    y_initial_mean = np.mean(y_prds_array, axis=0)
    
    # 3단계: 각 예측값과 초기 평균 간의 거리 계산 및 가중치 설정
    distances = []
    for i, y_pred in enumerate(y_prds):
        distance = np.mean((y_pred - y_initial_mean) ** 2)
        distances.append(distance)
    
    # 거리를 가중치로 변환 (거리가 가까울수록 높은 가중치)
    # 가우시안 가중치: exp(-distance/sigma)
    sigma = np.std(distances) if np.std(distances) > 0 else 1.0
    weights = np.exp(-np.array(distances) / sigma)
    weights = weights / np.sum(weights)  # 정규화
    
    print(f"가중치 통계:")
    print(f"  - 최대 가중치: {np.max(weights):.6f}")
    print(f"  - 최소 가중치: {np.min(weights):.6f}")
    print(f"  - 평균 가중치: {np.mean(weights):.6f}")
    print(f"  - 상위 10% 예측값들의 평균 가중치: {np.mean(np.sort(weights)[-len(weights)//10:]):.6f}")
    
    # 4단계: 가중평균 및 가중표준편차 계산
    y_weighted_mean = np.average(y_prds_array, axis=0, weights=weights)
    
    # 가중표준편차 계산
    weighted_variance = np.average((y_prds_array - y_weighted_mean)**2, axis=0, weights=weights)
    y_weighted_std = np.sqrt(weighted_variance)
    
    # MSE 계산
    mse_initial = np.mean((y_initial_mean - y_sample) ** 2)
    mse_weighted = np.mean((y_weighted_mean - y_sample) ** 2)
    
    print(f"Sample {sample_idx} 단순 평균 MSE: {mse_initial:.6f}")
    print(f"Sample {sample_idx} 가중 평균 MSE: {mse_weighted:.6f}")
    print(f"MSE 개선도: {((mse_initial - mse_weighted) / mse_initial * 100):.2f}%")
    
    # 5단계: 가중치 시각화가 포함된 예측값들 그리기
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 가중치에 따른 예측값들 (알파값으로 가중치 표현)
    plt.subplot(2, 1, 1)
    max_weight = np.max(weights)
    for i, (pred, weight) in enumerate(zip(y_prds, weights)):
        alpha = 0.1 + 0.7 * (weight / max_weight)  # 가중치에 비례한 투명도
        linewidth = 0.5 + 2.0 * (weight / max_weight)  # 가중치에 비례한 선 굵기
        plt.plot(pred, 'b-', linewidth=linewidth, alpha=alpha)
    plt.plot(y_sample, 'r-', linewidth=3, label='True value')
    plt.title(f'Sample {sample_idx} - Weighted Predictions (line thickness ∝ weight)')
    plt.ylabel('Normalized PCT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: 가중평균과 단순평균 비교
    plt.subplot(2, 1, 2)
    plt.fill_between(range(len(y_weighted_mean)), y_weighted_mean - y_weighted_std, y_weighted_mean + y_weighted_std, 
                     alpha=0.2, color='blue', label='Weighted ±1σ')
    plt.plot(y_weighted_mean, 'b-', linewidth=3, label='Weighted mean')
    plt.plot(y_initial_mean, 'g--', linewidth=2, label='Simple mean')
    plt.plot(y_sample, 'r-', linewidth=3, label='True value')
    plt.title(f'Sample {sample_idx} - Weighted vs Simple Mean Comparison')
    plt.ylabel('Normalized PCT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 그림 저장
    os.makedirs('./validation_results', exist_ok=True)
    plt.savefig(f'./validation_results/sample_{sample_idx}_weighted_predictions.png', dpi=300, bbox_inches='tight')
    print(f"Sample {sample_idx} 가중 예측 결과 저장 완료")
    
    # 6단계: 가중평균 상세 그래프 (제목 없음)
    plt.figure(figsize=(12, 8))
    plt.fill_between(range(len(y_weighted_mean)), y_weighted_mean - y_weighted_std, y_weighted_mean + y_weighted_std, 
                     alpha=0.2, color='blue', label='±1σ (weighted)')
    plt.fill_between(range(len(y_weighted_mean)), y_weighted_mean - 2*y_weighted_std, y_weighted_mean + 2*y_weighted_std, 
                     alpha=0.1, color='blue', label='±2σ (weighted)')
    plt.plot(y_weighted_mean, 'b-', linewidth=3, label='Weighted mean prediction')
    plt.plot(y_sample, 'r-', linewidth=3, label='True value')
    plt.xlabel('Time step')
    plt.ylabel('Normalized PCT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'./validation_results/sample_{sample_idx}_weighted_detail.png', dpi=300, bbox_inches='tight')
    print(f"Sample {sample_idx} 가중 상세 통계 그래프 저장 완료")
    
    # 7단계: 가중치 분포 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title(f'Sample {sample_idx} - Weight Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'./validation_results/sample_{sample_idx}_weight_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Sample {sample_idx} 가중치 분포 그래프 저장 완료")


def validation() -> None:
    """생성형 딥러닝 모델 성능 테스트"""
    # 테스트용 데이터 로드
    dataset = np.load("./train_test_dataset/train_test_dataset.npz")
    X_val = dataset['X_val'] # (600, 18)
    y_val = dataset['y_val'] # (600, 300)

    # 네트워크 설정 파일 로드
    with open('./trained_network/model_config.pkl', 'rb') as f:
        config = pkl.load(f)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 학습된 네트워크 모델 로드
    model = GPT2LMHeadModel.from_pretrained('./trained_network').to(device)
    print("모델 로드 완료")

    # 기존 기능: 한 케이스 한 번의 예측
    _X_val, _y_val = X_val[4], y_val[4]
    predict_single_case_single_time(_X_val, _y_val, model, config, device)

    # 기존 기능: 한 케이스 여러 번의 예측
    predict_single_case_multiple_times(_X_val, _y_val, model, config, device, n=100)
    
    # 새로운 기능: 전체 샘플 평가 후 베스트 샘플들에 대한 extensive prediction
    #print("\n=== 전체 샘플 평가 및 베스트 샘플 선별 시작 ===")
    #best_sample_indices = evaluate_all_samples_and_select_best(X_val, y_val, model, config, device, n_predictions=10, n_best=8)
    
    #print("\n=== 베스트 샘플들에 대한 extensive prediction 시작 ===")
    #predict_best_samples_extensively(X_val, y_val, model, config, device, best_sample_indices, n_extensive=100)

    #predict_specific_sample_multiple_times(X_val, y_val, model, config, device, sample_idx=232, n=200)
    predict_specific_sample_filtered(X_val, y_val, model, config, device, sample_idx=232, n_total=300, n_keep=100)

    #predict_specific_sample_weighted(X_val, y_val, model, config, device, sample_idx=232, n_total=300)

def main() -> None:
    validation()

if __name__ == "__main__":
    main()