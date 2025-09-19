import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle as pkl
from transformers import GPT2LMHeadModel
import time
import os
from LSTM_model_with_training import load_trained_lstm_model, LSTMConfig

# Define ModelConfig class to handle pickle loading
class ModelConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def predict_single_gpt2(X: np.ndarray, model: GPT2LMHeadModel, config, device) -> np.ndarray:
    """GPT-2 단일 예측"""
    X_tokens = (X * (config.vocab_size - 1)).astype(int).tolist()
    input_sequence = X_tokens + [config.vocab_size]
    input_ids = torch.tensor([input_sequence]).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=len(input_sequence) + config.output_dim,
            do_sample=True,
            pad_token_id=0,
        )
    
    generated = outputs[0].cpu().tolist()
    sep_index = generated.index(config.vocab_size)
    y_tokens = generated[sep_index + 1:][:config.output_dim]
    y_pred = np.array(y_tokens) / (config.vocab_size - 1)
    return y_pred

def predict_single_lstm(X: np.ndarray, model_wrapper, device) -> np.ndarray:
    """LSTM 단일 예측"""
    return model_wrapper.predict(X).flatten()

def compare_models_on_sample(X_val: np.ndarray, y_val: np.ndarray, sample_idx: int, 
                           gpt2_model, gpt2_config, gpt2_device,
                           lstm_model_wrapper, n_predictions: int = 100):
    """특정 샘플에 대해 두 모델 성능 비교"""
    
    print(f"Sample {sample_idx}에 대해 두 모델 비교 중... (각각 {n_predictions}번 예측)")
    
    X_sample = X_val[sample_idx]
    y_sample = y_val[sample_idx]
    
    # GPT-2 예측
    print("GPT-2 예측 중...")
    start_time = time.time()
    gpt2_predictions = []
    for i in range(n_predictions):
        if i % 20 == 0:
            print(f"  GPT-2 진행률: {i}/{n_predictions}")
        try:
            y_pred = predict_single_gpt2(X_sample, gpt2_model, gpt2_config, gpt2_device)
            if y_pred.shape == y_sample.shape:
                gpt2_predictions.append(y_pred)
        except:
            continue
    gpt2_time = time.time() - start_time
    
    # LSTM 예측
    print("LSTM 예측 중...")
    start_time = time.time()
    lstm_predictions = []
    for i in range(n_predictions):
        if i % 20 == 0:
            print(f"  LSTM 진행률: {i}/{n_predictions}")
        try:
            y_pred = predict_single_lstm(X_sample, lstm_model_wrapper, lstm_model_wrapper.device)
            if y_pred.shape == y_sample.shape:
                lstm_predictions.append(y_pred)
        except:
            continue
    lstm_time = time.time() - start_time
    
    print(f"GPT-2: {len(gpt2_predictions)}개 성공 예측 ({gpt2_time:.2f}s)")
    print(f"LSTM: {len(lstm_predictions)}개 성공 예측 ({lstm_time:.2f}s)")
    
    # 통계 계산
    if len(gpt2_predictions) > 0:
        gpt2_array = np.array(gpt2_predictions)
        gpt2_mean = np.mean(gpt2_array, axis=0)
        gpt2_std = np.std(gpt2_array, axis=0)
        gpt2_mse = np.mean((gpt2_mean - y_sample) ** 2)
    else:
        gpt2_mean = gpt2_std = None
        gpt2_mse = float('inf')
    
    if len(lstm_predictions) > 0:
        lstm_array = np.array(lstm_predictions)
        lstm_mean = np.mean(lstm_array, axis=0)
        lstm_std = np.std(lstm_array, axis=0)
        lstm_mse = np.mean((lstm_mean - y_sample) ** 2)
    else:
        lstm_mean = lstm_std = None
        lstm_mse = float('inf')
    
    print(f"GPT-2 MSE: {gpt2_mse:.6f}")
    print(f"LSTM MSE: {lstm_mse:.6f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. GPT-2 예측들
    ax = axes[0, 0]
    if len(gpt2_predictions) > 0:
        for pred in gpt2_predictions:
            ax.plot(pred, 'b-', alpha=0.1, linewidth=1)
        ax.plot(gpt2_mean, 'b-', linewidth=3, label='GPT-2 Mean')
    ax.plot(y_sample, 'r-', linewidth=3, label='True Value')
    ax.set_title(f'GPT-2 Predictions (n={len(gpt2_predictions)}, MSE={gpt2_mse:.6f})')
    ax.set_ylabel('Normalized PCT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. LSTM 예측들
    ax = axes[0, 1]
    if len(lstm_predictions) > 0:
        for pred in lstm_predictions:
            ax.plot(pred, 'g-', alpha=0.1, linewidth=1)
        ax.plot(lstm_mean, 'g-', linewidth=3, label='LSTM Mean')
    ax.plot(y_sample, 'r-', linewidth=3, label='True Value')
    ax.set_title(f'LSTM Predictions (n={len(lstm_predictions)}, MSE={lstm_mse:.6f})')
    ax.set_ylabel('Normalized PCT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 통계 비교
    ax = axes[1, 0]
    if gpt2_mean is not None:
        ax.fill_between(range(len(gpt2_mean)), gpt2_mean - gpt2_std, gpt2_mean + gpt2_std, 
                       alpha=0.2, color='blue', label='GPT-2 ±1σ')
        ax.plot(gpt2_mean, 'b-', linewidth=3, label='GPT-2 Mean')
    if lstm_mean is not None:
        ax.fill_between(range(len(lstm_mean)), lstm_mean - lstm_std, lstm_mean + lstm_std, 
                       alpha=0.2, color='green', label='LSTM ±1σ')
        ax.plot(lstm_mean, 'g-', linewidth=3, label='LSTM Mean')
    ax.plot(y_sample, 'r-', linewidth=3, label='True Value')
    ax.set_title('Statistical Comparison')
    ax.set_ylabel('Normalized PCT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 성능 메트릭 텍스트 비교
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics_text = f"""
    Performance Metrics Comparison
    
    GPT-2:
      • MSE: {gpt2_mse:.6f}
      • Avg Prediction Time: {gpt2_time/n_predictions:.4f}s
      • Success Rate: {len(gpt2_predictions)/n_predictions*100:.1f}%
      • Total Predictions: {len(gpt2_predictions)}/{n_predictions}
    
    LSTM:
      • MSE: {lstm_mse:.6f}
      • Avg Prediction Time: {lstm_time/n_predictions:.4f}s
      • Success Rate: {len(lstm_predictions)/n_predictions*100:.1f}%
      • Total Predictions: {len(lstm_predictions)}/{n_predictions}
    
    Winner:
      • Lower MSE: {'GPT-2' if gpt2_mse < lstm_mse else 'LSTM' if lstm_mse < gpt2_mse else 'Tie'}
      • Faster: {'GPT-2' if gpt2_time < lstm_time else 'LSTM'}
      • More Reliable: {'GPT-2' if len(gpt2_predictions) > len(lstm_predictions) else 'LSTM' if len(lstm_predictions) > len(gpt2_predictions) else 'Tie'}
    """
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # 저장
    os.makedirs('./comparison_results', exist_ok=True)
    plt.savefig(f'./comparison_results/sample_{sample_idx}_model_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample {sample_idx} 비교 결과 저장 완료")
    
    return {
        'sample_idx': sample_idx,
        'gpt2_mse': gpt2_mse,
        'lstm_mse': lstm_mse,
        'gpt2_time': gpt2_time,
        'lstm_time': lstm_time,
        'gpt2_success_rate': len(gpt2_predictions) / n_predictions,
        'lstm_success_rate': len(lstm_predictions) / n_predictions,
        'gpt2_predictions': len(gpt2_predictions),
        'lstm_predictions': len(lstm_predictions)
    }

def batch_model_comparison(sample_indices: list, n_predictions: int = 50):
    """여러 샘플에 대해 일괄 모델 비교"""
    
    # 데이터 로드
    dataset = np.load("./train_test_dataset/train_test_dataset.npz")
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    
    # GPT-2 모델 로드
    print("GPT-2 모델 로드 중...")
    with open('./trained_network/model_config.pkl', 'rb') as f:
        gpt2_config = pkl.load(f)
    gpt2_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt2_model = GPT2LMHeadModel.from_pretrained('./trained_network').to(gpt2_device)
    print("GPT-2 모델 로드 완료")
    
    # LSTM 모델 로드
    print("LSTM 모델 로드 중...")
    lstm_model_wrapper, lstm_config = load_trained_lstm_model('./trained_lstm_network')
    print("LSTM 모델 로드 완료")
    
    # 각 샘플에 대해 비교 수행
    results = []
    for sample_idx in sample_indices:
        print(f"\n=== Sample {sample_idx} 비교 시작 ===")
        result = compare_models_on_sample(
            X_val, y_val, sample_idx,
            gpt2_model, gpt2_config, gpt2_device,
            lstm_model_wrapper, n_predictions
        )
        results.append(result)
    
    # 전체 결과 요약
    print("\n" + "="*60)
    print("전체 비교 결과 요약")
    print("="*60)
    
    gpt2_wins = 0
    lstm_wins = 0
    ties = 0
    
    for result in results:
        sample_idx = result['sample_idx']
        gpt2_mse = result['gpt2_mse']
        lstm_mse = result['lstm_mse']
        
        if gpt2_mse < lstm_mse:
            winner = "GPT-2"
            gpt2_wins += 1
        elif lstm_mse < gpt2_mse:
            winner = "LSTM"
            lstm_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        print(f"Sample {sample_idx}: GPT-2 MSE={gpt2_mse:.6f}, LSTM MSE={lstm_mse:.6f} → {winner}")
    
    print(f"\n최종 스코어: GPT-2 {gpt2_wins}승, LSTM {lstm_wins}승, 무승부 {ties}개")
    
    # 평균 성능 계산
    avg_gpt2_mse = np.mean([r['gpt2_mse'] for r in results if r['gpt2_mse'] != float('inf')])
    avg_lstm_mse = np.mean([r['lstm_mse'] for r in results if r['lstm_mse'] != float('inf')])
    avg_gpt2_time = np.mean([r['gpt2_time'] for r in results])
    avg_lstm_time = np.mean([r['lstm_time'] for r in results])
    
    print(f"\n평균 성능:")
    print(f"  GPT-2 평균 MSE: {avg_gpt2_mse:.6f}")
    print(f"  LSTM 평균 MSE: {avg_lstm_mse:.6f}")
    print(f"  GPT-2 평균 시간: {avg_gpt2_time:.2f}s")
    print(f"  LSTM 평균 시간: {avg_lstm_time:.2f}s")
    
    # 결과 저장
    with open('./comparison_results/batch_comparison_results.pkl', 'wb') as f:
        pkl.dump(results, f)
    
    return results

def quick_single_comparison(sample_idx: int = 232, n_predictions: int = 100):
    """단일 샘플에 대한 빠른 비교"""
    
    # 데이터 로드
    dataset = np.load("./train_test_dataset/train_test_dataset.npz")
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    
    # GPT-2 모델 로드
    print("GPT-2 모델 로드 중...")
    with open('./trained_network/model_config.pkl', 'rb') as f:
        gpt2_config = pkl.load(f)
    gpt2_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt2_model = GPT2LMHeadModel.from_pretrained('./trained_network').to(gpt2_device)
    
    # LSTM 모델 로드
    print("LSTM 모델 로드 중...")
    lstm_model_wrapper, lstm_config = load_trained_lstm_model('./trained_lstm_network')
    
    # 비교 수행
    result = compare_models_on_sample(
        X_val, y_val, sample_idx,
        gpt2_model, gpt2_config, gpt2_device,
        lstm_model_wrapper, n_predictions
    )
    
    return result

def main():
    """메인 실행 함수"""
    print("모델 성능 비교 스크립트")
    print("1. 단일 샘플 비교 (Sample 232)")
    print("2. 여러 샘플 일괄 비교")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        # 단일 샘플 비교
        sample_idx = int(input("샘플 인덱스를 입력하세요 (기본값 232): ") or "232")
        n_predictions = int(input("예측 횟수를 입력하세요 (기본값 100): ") or "100")
        
        result = quick_single_comparison(sample_idx, n_predictions)
        print(f"\n비교 완료! 결과는 ./comparison_results/sample_{sample_idx}_model_comparison.png에서 확인하세요.")
        
    elif choice == "2":
        # 일괄 비교
        sample_list = input("샘플 인덱스들을 쉼표로 구분해서 입력하세요 (예: 50,100,150,200,232): ")
        if sample_list.strip():
            sample_indices = [int(x.strip()) for x in sample_list.split(',')]
        else:
            sample_indices = [50, 100, 150, 200, 232]  # 기본값
        
        n_predictions = int(input("각 샘플당 예측 횟수를 입력하세요 (기본값 50): ") or "50")
        
        results = batch_model_comparison(sample_indices, n_predictions)
        print(f"\n일괄 비교 완료! 결과는 ./comparison_results/ 폴더에서 확인하세요.")
        
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()