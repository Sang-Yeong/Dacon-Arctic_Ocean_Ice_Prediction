experiment_params = {
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "normalize_flag": True,
    "model": "weather_model",
    "device": 'cuda'
}


model_params = {
    "weather_model": {
        "batch_gen": {
            "input_dim": [0],       # 학습에 사용할 데이터 인덱스, 본 대회에서는 5개 중 제일 첫번째 사진만 사용
            "output_dim": 0,        # 예측 결과로 나와야 할 데이터 인덱스
            "window_in_len": 12,    # 한번 학습할 때마다 사용할 데이터 개수 (time축)
            "window_out_len": 12,   # 한번 학습하고 결과 예측할 때마다 사용할 데이터 개수 (time축)
            "batch_size": 1,
            "shuffle": True,
        },
        "trainer": {
            "num_epochs": 3,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": 0.0005,
            "clip": 5,
            "early_stop_tolerance": 5
        },
        "core": {
            "input_size": (448, 304),
            "window_in": 12,        # batch_gen["window_in_len"]와 동일하게 설정
            "window_out": 12,       # batch_gen["window_out_len"]와 동일하게 설정
            "num_layers": 3,
            "selected_dim": 0,      # batch_gen["input_dim"]에서 설정한 인덱스 중, batch_gen["output_dim"]에 해당하는 인덱스
            "input_attn_params": {  # model 구성 중, attention block에 대한 파라미터 설정
                "input_dim": 12,
                "hidden_dim": 32,
                "attn_channel": 1,
                "kernel_size": 3
            },
            "encoder_params": {
                "input_dim": 1,     # input_dim list 원소개수와 동일하게 설정
                "hidden_dims": [32, 32, 16],
                "kernel_size": [5, 3, 1],
                "bias": False,
                "peephole_con": False
            },
            "decoder_params": {
                "input_dim": 1,
                "hidden_dims": [16, 32, 32],
                "kernel_size": [3, 3, 3],
                "bias": False,
                "peephole_con": False
            },
            "output_conv_params": {
                "mid_channel": 5,
                "in_kernel": 3,
                "out_kernel": 1
            }
        }

    }
}
