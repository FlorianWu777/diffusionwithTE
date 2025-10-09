import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
import pandas as pd
from dateutil.relativedelta import relativedelta

class IceNetDataset(Dataset):
    def __init__(self, config_path, mode='train'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.mode = mode
        self.dataset_path = os.path.join('datasets', self.config['dataset_name'])
        self.raw_shape = tuple(self.config['raw_data_shape'])
        self.n_forecast_months = self.config['n_forecast_months']
        self.input_data = self.config['input_data']
        self.forecast_dates = self._load_forecast_dates()
        self.variable_paths = self._build_variable_paths()
        self.num_channels = self._calculate_input_channels()

    def _load_forecast_dates(self):
        start, end = self.config['sample_IDs'][f'obs_{self.mode}_dates']
        return list(pd.date_range(start=start, end=end, freq='MS', closed='right'))

    def _build_variable_paths(self):
        paths = {}
        for varname, vardict in self.input_data.items():
            if 'metadata' not in vardict:
                for fmt in vardict:
                    if vardict[fmt]['include']:
                        paths[f'{varname}_{fmt}'] = os.path.join(
                            self.dataset_path, 'obs', varname, fmt, '{:04d}_{:02d}.npy')
            else:
                if vardict['include']:
                    if varname == 'land':
                        paths['land'] = os.path.join(self.dataset_path, 'meta', 'land.npy')
                    if varname == 'circmonth':
                        paths['circmonth_cos'] = os.path.join(self.dataset_path, 'meta', 'cos_month_{:02d}.npy')
                        paths['circmonth_sin'] = os.path.join(self.dataset_path, 'meta', 'sin_month_{:02d}.npy')
        return paths

    def _calculate_input_channels(self):
        total = 0
        for varname, vardict in self.input_data.items():
            if 'metadata' not in vardict:
                for fmt in vardict:
                    if vardict[fmt]['include']:
                        if fmt != 'linear_trend':
                            total += vardict[fmt]['max_lag']
                        else:
                            total += self.n_forecast_months
            else:
                if vardict['include']:
                    if varname == 'land':
                        total += 1
                    elif varname == 'circmonth':
                        total += 2
        return total

    def __len__(self):
        return len(self.forecast_dates)

    def __getitem__(self, idx):
        date = self.forecast_dates[idx]
        input_tensor = np.zeros((*self.raw_shape, self.num_channels), dtype=np.float32)
        output_tensor = np.zeros((*self.raw_shape, self.n_forecast_months), dtype=np.float32)

        channel_idx = 0
        present_date = date - relativedelta(months=1)

        for varname, vardict in self.input_data.items():
            if 'metadata' not in vardict:
                for fmt in vardict:
                    if vardict[fmt]['include']:
                        if fmt != 'linear_trend':
                            for lag in range(1, vardict[fmt]['max_lag']+1):
                                d = present_date - relativedelta(months=lag-1)
                                fpath = self.variable_paths[f'{varname}_{fmt}'].format(d.year, d.month)
                                input_tensor[:, :, channel_idx] = np.load(fpath)
                                channel_idx += 1
                        else:
                            for lead in range(1, self.n_forecast_months+1):
                                d = present_date + relativedelta(months=lead)
                                fpath = self.variable_paths[f'{varname}_{fmt}'].format(d.year, d.month)
                                input_tensor[:, :, channel_idx] = np.load(fpath)
                                channel_idx += 1
            else:
                if vardict['include']:
                    if varname == 'land':
                        input_tensor[:, :, channel_idx] = np.load(self.variable_paths['land'])
                        channel_idx += 1
                    elif varname == 'circmonth':
                        cos_path = self.variable_paths['circmonth_cos'].format(date.month)
                        sin_path = self.variable_paths['circmonth_sin'].format(date.month)
                        input_tensor[:, :, channel_idx] = np.load(cos_path)
                        input_tensor[:, :, channel_idx+1] = np.load(sin_path)
                        channel_idx += 2

        for lead in range(self.n_forecast_months):
            future_date = date + relativedelta(months=lead)
            sic_path = self.variable_paths['siconca_abs'].format(future_date.year, future_date.month)
            output_tensor[:, :, lead] = np.load(sic_path)

        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # C, H, W
        output_tensor = np.transpose(output_tensor, (2, 0, 1))  # T, H, W
        return torch.tensor(input_tensor), torch.tensor(output_tensor)
