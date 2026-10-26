import numpy as np

def matrix_sim(
    alpha: np.ndarray, 
    returns: np.ndarray, 
    book_size: float|np.ndarray|None = None, 
    close: np.ndarray|None = None, 
    trade_price: np.ndarray|None = None,
    slippage: float|np.ndarray|None = None,
    impact_factor: np.ndarray|None = None, 
    stats: list[np.ndarray]|dict[str, np.ndarray]|None = None,
    as_dict: bool = True
    ) -> dict[str, np.ndarray]|list:
    """
    Args:
        alpha: di x ii alpha to be evaluated
        returns: di x ii returns of instruments
        book_size: book size for turnover calculation. If not given, daily booksize is computed from alpha.
        close: di x ii close price of instruments (used for PNL marking)
        trade_price: di x ii trade price of instruments (can be vwap or twap, for instance)
        slippage: float or di x ii matrix for slippage calculation
        impact_factor: di x ii impact factor
        stats: list of string for stat to be computed or dict of (str, np.ndarray) for output. If None or not given, computes pnl and turnover only.
        as_dict: indicate whether to return stats as dict or as list
    Returns:
        dict[(str, np.ndarray)] or list[np.ndarray]: Stats computed
    """
    if stats is None:
        stats = ['pnl', 'turnover']
    if isinstance(stats, dict):
        assert as_dict, 'If stats is given as a dict, as_dict must be True.'
    else:
        stats = { k: np.zeros(alpha.shape[:-1], dtype=np.float64) for k in stats }

    if close is None or trade_price is None:
        close_to_price_diff = None
        """:type: None|np.ndarray"""
    else:
        close_to_price_diff = close / trade_price - 1.
        assert isinstance(close_to_price_diff, np.ndarray)
        close_to_price_diff[~np.isfinite(close_to_price_diff)] = 0.

    alpha_abs = np.fabs(alpha)
    daily_pos_change = np.diff(alpha, 1, axis=0)
    daily_pos_change_abs = np.fabs(daily_pos_change)

    if close_to_price_diff is None:
        np.nansum(alpha[:-1] * returns[1:], out=stats['pnl'][1:], axis=-1)
    else:
        np.nansum(alpha[:-1] * returns[1:] + daily_pos_change * close_to_price_diff[1:], out=stats['pnl'][1:], axis=-1)

    if 'hold_value' in stats:
        np.nansum(alpha_abs, out=stats['hold_value'], axis=-1)

    if book_size is None:
        if 'hold_value' in stats:
            book_size = stats['hold_value'][:-1]
        else:
            book_size = np.nansum(alpha_abs, axis=-1)[:-1]

    stats['turnover'][1:] = np.nansum(daily_pos_change_abs / book_size[..., None], axis=-1)

    if 'differential_pnl' in stats:
        if close_to_price_diff is None:
            np.nansum(daily_pos_change[:-1] * returns[2:], out=stats['differential_pnl'][2:], axis=-1)
        else:
            np.nansum(daily_pos_change[:-1] * returns[2:] + np.diff(daily_pos_change, 1, axis=0) * close_to_price_diff[2:], out=stats['differential_pnl'][2:], axis=-1)

    if 'long_value' in stats:
        np.nansum(alpha * (alpha > 0), out=stats['long_value'], axis=-1)
    if 'short_value' in stats:
        np.nansum(alpha * (alpha < 0), out=stats['short_value'], axis=-1)

    if 'ret' in stats:
        stats['ret'][1:] = stats['pnl'][1:]
        stats['ret'][1:] /= book_size

    if 'traded_value' in stats:
        np.nansum(daily_pos_change_abs, out=stats['traded_value'][1:], axis=-1)

    if 'hold_shares' in stats:
        np.nansum(alpha_abs / close, out=stats['hold_shares'], axis=-1)

    if 'trade_shares' in stats:
        np.nansum(daily_pos_change_abs / close[:-1], out=stats['trade_shares'][1:], axis=-1)

    if 'long_count' in stats:
        np.sum(alpha > 0, out=stats['long_count'], axis=-1)
    if 'short_count' in stats:
        np.sum(alpha < 0, out=stats['short_count'], axis=-1)

    if 'slippage' in stats:
        if isinstance(slippage, float):
            slp = slippage * daily_pos_change_abs
        else:
            slp = slippage[1:] / close[1:] * daily_pos_change_abs
        np.nansum(slp, out=stats['slippage'][1:], axis=-1)

    if 'impact' in stats:
        stats['impact'][1:] = 0.5 * np.nansum(impact_factor[1:] * np.square(daily_pos_change), axis=-1)

    if as_dict:
        return stats
    else:
        return list(stats.values())