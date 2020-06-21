import numpy as np


def call_price(stock_price, strike_price):
    return np.maximum(0, stock_price - strike_price)


def put_price(stock_price, strike_price):
    return np.maximum(0, strike_price - stock_price)
