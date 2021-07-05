from dataset import WeatherDataset
from models.adaptive_normalizer import AdaptiveNormalizer

'''
- batch_generator.py
    - __split_data: 정해준 비율로 'train', 'val', 'test' split
    - generate: selected_loader.next() --> dataset.py
    - dataset.py의 next()
        - x = batch_data[:, :self.window_in_len, ..., self.input_dim]
        - y = batch_data[:, self.window_in_len:, ..., [self.output_dim]]
        - x, y, f_x 만들어서 training시 아래 수행 
            pred = model.forward(x=x, f_x=f_x, hidden=hidden)
            loss = self.criterion(pred, y)
'''


class BatchGenerator:
    def __init__(self, weather_data, val_ratio, test_ratio, normalize_flag, params):
        self.weather_data = weather_data
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.dataset_params = params
        self.normalize_flag = normalize_flag

        if self.normalize_flag:
            self.normalizer = AdaptiveNormalizer(output_dim=params['output_dim'])
        else:
            self.normalizer = None

        self.weather_dict = self.__split_data(self.weather_data)
        self.dataset_dict = self.__create_sets()

    def __split_data(self, in_data):
        data_len = len(in_data)
        val_count = int(data_len * self.val_ratio)
        test_count = int(data_len * self.val_ratio)

        train_count = data_len - val_count - test_count

        data_dict = {
            'train': in_data[:train_count],
            'val': in_data[train_count:train_count+val_count],
            'test': in_data[train_count+val_count:]
        }

        return data_dict

    def __create_sets(self):
        hurricane_dataset = {}
        for i in ['train', 'val', 'test']:
            dataset = WeatherDataset(weather_data=self.weather_dict[i],
                                     normalizer=self.normalizer,
                                     **self.dataset_params)
            hurricane_dataset[i] = dataset

        return hurricane_dataset

    def num_iter(self, dataset_name):
        return self.dataset_dict[dataset_name].num_iter

    def generate(self, dataset_name):
        selected_loader = self.dataset_dict[dataset_name]
        '''
            def next(self):
        """
        Iterator function. It yields x and y with the following shapes
        x: (B, T, M, N, self.input_dim)
        y: (B, T, M, N, 1)

        :return: x, y
        :rtype: torch.tensor, torch.tensor
        """
        weather_data = self.__create_buffer(in_data=self.weather_data)
        self.num_iter = len(weather_data)

        prev_batch = None
        for i in range(self.num_iter):
            batch_data = torch.from_numpy(self.__load_batch(batch=weather_data[i]))

            if self.normalizer:
                batch_data = self.normalizer.norm(batch_data)

            # create x and y
            x = batch_data[:, :self.window_in_len, ..., self.input_dim]
            y = batch_data[:, self.window_in_len:, ..., [self.output_dim]]

            # create flow matrix
            if prev_batch is None:
                prev_batch = torch.zeros_like(batch_data)

            flow_batch = torch.cat([prev_batch[:, [-1]], batch_data], dim=1)  # (B, T+1, M, N, D)
            f_x = self.create_flow_mat(flow_batch)

            yield x, y, f_x
        
        '''

        yield from selected_loader.next()
