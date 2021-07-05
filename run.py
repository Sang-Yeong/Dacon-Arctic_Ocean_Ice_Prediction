import os
import shutil
import pandas as pd
import numpy as np
import glob
import cv2

from config import experiment_params, model_params
from data_creator import DataCreator
from batch_generator import BatchGenerator
from experiment import predict, train
from models.weather.weather_model import WeatherModel
from models.baseline.moving_avg import MovingAvg
from models.baseline.convlstm import ConvLSTM
from models.baseline.u_net import UNet

from skimage.transform import resize


def run():
    '''
    #############################
    ### experiment_params 설정 ###
    #############################

    "val_ratio": 0.1(train), 0(predict)
    "test_ratio": 0.1(train), 1(predict)
    "normalize_flag": True,
    "model": "weather_model",
    "device": 'cuda'
    '''

    val_ratio = experiment_params['val_ratio']
    test_ratio = experiment_params['test_ratio']
    normalize_flag = experiment_params['normalize_flag']
    model_name = experiment_params['model']
    device = experiment_params['device']
    model_dispatcher = {
        'moving_avg': MovingAvg,
        'convlstm': ConvLSTM,
        'u_net': UNet,
        'weather_model': WeatherModel,
    }

    # 데이터 파일 경로 불러오기
    weather_data = np.array(glob.glob('./data/data_dump/*.*', recursive=True))

    selected_model_params = model_params[model_name]["core"]
    batch_gen_params = model_params[model_name]["batch_gen"]
    trainer_params = model_params[model_name]["trainer"]
    config = {
        "experiment_params": experiment_params,
        f"{model_name}_params": model_params[model_name]
    }

    '''
    ###############################
    ##### 데이터셋 픽셀범위 변경 #####
    ###############################

    본 코드의 input 값의 픽셀범위는 [-1, 1]사이의 값으로 구성됨.
    해당 대회에서 주어진 dataset의 픽셀범위는 [0, 250]으로 설정되어있음.

    step1. from skimage.transform import resize로 이미지의 크기를 변경해주면, [0, 1]사이의 값으로 변경됨.
    step2. cv2.normalize를 통해 [-1, 1]사이의 값으로 normalize 수행

    '''
    # normalize 한 결과를 저장하기 위해 빈 배열 선언
    img_norm2 = np.zeros((448, 304, 5))

    # input image를 차례로 불러와 normalize 수행
    for path in weather_data:
        img_array = np.load(path)
        img_array = resize(img_array, (448, 304, 5))
        for d in range(5):
            img_float = img_array[..., d]
            img_norm2[..., d] = cv2.normalize(img_float, None, -1, 1, cv2.NORM_MINMAX)

        np.save('./data/data_use/' + path.split('/')[-1], img_norm2)

    print("success to load image")

    '''
    #############################
    ######### training ##########
    #############################     
    '''
    # 전처리를 마친 이미지 불러오기
    weather_data_train = np.array(
        sorted(glob.glob('./data/data_use/*.*', recursive=True)))

    # 학습을 위한 data generator 생성
    # batch_generator.py
    batch_generator = BatchGenerator(weather_data=weather_data_train,
                                     val_ratio=val_ratio,
                                     test_ratio=test_ratio,
                                     params=batch_gen_params,
                                     normalize_flag=normalize_flag)

    model = model_dispatcher[model_name](device=device, **selected_model_params)

    train(model_name=model_name,
          model=model,
          batch_generator=batch_generator,
          trainer_params=trainer_params,
          date_r=None,
          config=config,
          device=device)

    '''
    ############################
    ######### predict ##########
    ############################
    step1. predict에 사용할 데이터 목록에 대한 경로 차례로 불러오기
    step2. predict를 위한 data generator 생성
    step3. predict 수행
    step4. 대회에서 요구하는 결과 형태로 맞추기
    step5. sample_submission 사용하여 제출형태 맞추기
    '''
    # step1
    # 대회에서 주어진 csv파일로 predict에 사용할 데이터 경로 불러오기
    test1 = pd.read_csv('dacon_data/data/data/data_v2/public_weekly_test.csv')
    test_path1 = './data/data_use/' + test1.tail(12)['week_file_nm']

    test2 = pd.read_csv('dacon_data/data/data/data_v2/private_weekly_test.csv')
    test_path2 = './data/data_use/' + test2.tail(12)['week_file_nm']

    # 시계열 데이터이기 때문에, 데이터 순서 정렬해서 가져오기
    weather_data_predict1 = np.array(sorted(test_path1))
    weather_data_predict2 = np.array(sorted(test_path2))

    # step2
    # predict를 위한 data generator 생성
    batch_generator1 = BatchGenerator(weather_data=weather_data_predict1,
                                      val_ratio=val_ratio,
                                      test_ratio=test_ratio,
                                      params=batch_gen_params,
                                      normalize_flag=normalize_flag)

    batch_generator2 = BatchGenerator(weather_data=weather_data_predict2,
                                      val_ratio=val_ratio,
                                      test_ratio=test_ratio,
                                      params=batch_gen_params,
                                      normalize_flag=normalize_flag)

    # step3
    # 각 데이터셋에 대한 predict 수행
    predict_img1 = predict(model_name=model_name, batch_generator=batch_generator1, device=device, exp_num=1)
    predict_img2 = predict(model_name=model_name, batch_generator=batch_generator2, device=device, exp_num=1)

    # step4
    # 대회에서 요구하는 결과 형태로 맞추기
    # 12(time), 448, 304(data shape), 1(총 5장 중 결과 1장)
    predict_img1 = predict_img1.detach().to('cpu').numpy()
    predict_img1 = cv2.normalize(predict_img1, None, 0, 250, cv2.NORM_MINMAX)
    predict_img1 = predict_img1.reshape([12, 448, 304, 1])

    predict_img2 = predict_img2.detach().to('cpu').numpy()
    predict_img2 = cv2.normalize(predict_img2, None, 0, 250, cv2.NORM_MINMAX)
    predict_img2 = predict_img2.reshape([12, 448, 304, 1])

    # step5
    # sample_submission 사용하여 제출형태 맞추기
    submission = pd.read_csv('./dacon_data/data/data/data_v2/sample_submission.csv')

    index = [i for i in range(12, 24)]
    idx = pd.DataFrame(index, columns=['idx'])

    sub_2020 = submission.loc[:11, ['week_start']].copy()
    sub_2021 = submission.loc[12:, ['week_start']].copy()
    pd1 = pd.DataFrame(predict_img2.reshape([12, -1]))

    pd1 = pd.concat([idx, pd1], axis=1)
    pd1.set_index('idx', append=False, inplace=True)

    # 최종 제출형식
    sub_2020 = pd.concat([sub_2020, (pd.DataFrame(predict_img1.reshape([12, -1])))], axis=1)
    sub_2021 = pd.concat([sub_2021, pd1], axis=1)
    submission = pd.concat([sub_2020, sub_2021], axis=0)

    submission.to_csv('baseline.csv', index=False)


if __name__ == '__main__':
    run()
