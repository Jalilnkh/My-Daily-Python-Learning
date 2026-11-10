import numpy as np
import pandas as pd
import torch

from timesfm3_stock import (read_series, split_origin_indices, make_loader,
                            forward_prices, relative_huber)


class ToyTorchModel(torch.nn.Module):
    """Cheap shape/gradient contract test, NOT TimesFM-3 nor a stock model."""
    input_patch_len = 32
    num_quantiles = 9
    def __init__(self):
        super().__init__()
        self.output_head = torch.nn.Linear(1, 64 * 9)

    def forward(self, inputs):
        x = inputs['values'].mean(dim=-1, keepdim=True)
        pred = self.output_head(x).reshape(x.shape[0], 1, x.shape[2], 64, 9)
        return {'logits': pred}


def test_window_splits_no_future_leak():
    s = split_origin_indices(1100, 256, 5, 126, 126, 5)
    assert len(s.train) > 10 and len(s.val) > 10 and len(s.test) > 10
    assert (s.train + 5 < s.val_start).all()
    assert (s.val + 1 >= s.val_start).all()
    assert (s.val + 5 < s.test_start).all()
    assert (s.test + 1 >= s.test_start).all()
    assert (s.test + 5 < 1100).all()


def test_toy_forward_has_gradients():
    toy = ToyTorchModel()
    x = torch.linspace(30, 90, 256).repeat(3, 1)
    y = torch.ones(3, 5) * 90
    pred = forward_prices(toy, x, 5)
    assert pred.shape == (3, 5)
    loss = relative_huber(pred, y, x[:, -1:])
    loss.backward()
    assert toy.output_head.weight.grad is not None
    assert torch.isfinite(toy.output_head.weight.grad).all()


def test_csv_long_schema(tmp_path):
    days = pd.bdate_range('2024-01-01', periods=400)
    frame = pd.DataFrame({'date': days, 'ticker': ['NVDA'] * 400,
                          'close': np.arange(400) + 50., 'adj_close': np.arange(400) + 49.})
    f = tmp_path / 'prices_100_stocks_daily.csv'
    frame.to_csv(f, index=False)
    stock = read_series(f, 'nvda')
    assert len(stock) == 400
    s = split_origin_indices(len(stock), 32, 5, 63, 63, 1)
    loader = make_loader(stock.price.to_numpy(np.float32), s.train, 32, 5, 4, False)
    x, y, origin = next(iter(loader))
    assert x.shape == (4, 32) and y.shape == (4, 5)
    assert np.isclose(y[0, 0], stock.price.iloc[int(origin[0]) + 1])
