import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string

#allowed imports
import pandas as pd
import numpy as np
import statistics as stats
import math
import typing
import jsonpickle

#native libraries
import copy
import collections

import sys


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:

    def update_prev_prices(self, state, mp_price_history, product):
        num_past_prices = 20

        if len(mp_price_history[product]) >= num_past_prices:
            mp_price_history[product].pop(0)
        mp_price_history[product].append(self.depths_calc_midP(state.order_depths[product]))
        return mp_price_history

    def depths_calc_midP(self, order_depths):
        """
        Calculates: midprice of a order_depths object

        Parameters:
        - price_dict: `state.order_depths[product]`
        """
        _, best_bid, _ = self.calc_metrics_bids(order_depths.buy_orders)
        _, best_ask, _ = self.calc_metrics_bids(order_depths.sell_orders)
        return (best_bid+best_ask)/2

    def calc_regression(self, past_prices):
        """
        Simple linear regression of 5 evenly spaced prices to a sixth prce (one unit from the 5th)

        Parameters: 
        - past_prices: `List[float]` - 5 evenly spaced past prices

        Returns:
        - next_price: `float` - next price (one unit from the 5th)
        """
        n = len(past_prices)
        if n <= 1:
            return past_prices[0]
        else:
            x = np.arange(0, n, 1)  
            y = np.array(past_prices) 
            poly_coeffs = np.polyfit(x, y, deg=1)
            next_price = np.polyval(poly_coeffs, n+1)
            return next_price

    def calc_metrics_bids(self, price_dict):
        volume = 0 
        highest_bid = 0
        bids_vwap = 0
        print(price_dict)

        #In this function we are looping through the bids from highest (most attractive) to lowest (least attractive)
        #We use this function to find the three most important metrics: total volume of bids, highest (most attractive) bid, and the total vwap of bids
        if len(price_dict) > 0:
            price_dict = collections.OrderedDict(sorted(price_dict.items(), reverse=True))
            for index, (key,value) in enumerate(price_dict.items()):
                if index == 0:
                    highest_bid = key
                volume += value
                bids_vwap += key*value
            
            if volume != 0:
                bids_vwap /= volume
            else:
                bids_vwap /= 1

        return volume, highest_bid, bids_vwap
    
    def calc_metrics_asks(self, price_dict): 
        volume = 0 
        lowest_ask = 0
        asks_vwap = 0

        #In this function we are looping through the asks from lowest (most attractive) to highest (least attractive)
        #We use this function to find the three most important metrics: total volume of asks, lowest (most attractive) bid, and the total vwap of asks
        if len(price_dict) > 0:
            price_dict = collections.OrderedDict(sorted(price_dict.items()))
            for index, (key,value) in enumerate(price_dict.items()):
                if index == 0:
                    lowest_ask = key
                volume += -value
                asks_vwap += key*-value
            
            if volume != 0:
                asks_vwap /= volume
            else:
                asks_vwap /= 1

        return volume, lowest_ask, asks_vwap

        d_plus = (math.log(S_0 / K) + (r + (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))
        d_minus = (math.log(S_0 / K) + (r - (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))

        return S_0 * stats.NormalDist().cdf(d_plus) - K * math.exp(-r*tau) * stats.NormalDist().cdf(d_minus)
    
    def BS_call_calc(self, S_0, K, sig, tau, r=0):
        """
        Calculates: midprice of a order_depths object

        Parameters:
        - `S_0` - Current underlying price
        - `K` - Strike Price
        - `sig` - Annualized volatility of returns
        - `tau` - Time until maturity in days (will be converted to years for calc)
        - `r` - "RF" Interest rate
        """
        tau /= 252
        d_plus = (math.log(S_0 / K) + (r + (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))
        d_minus = (math.log(S_0 / K) + (r - (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))
        return S_0 * stats.NormalDist().cdf(d_plus) - K * math.exp(-r*tau) * stats.NormalDist().cdf(d_minus)

    def stat_arb_orders(self, product1, product2, product1str, product2str, enter_at, exit_at):
        print("lol")
    #AMETHYSTS
    def order_gen_AMETHYSTS_MT(self, order_book, starting_pos, pos_limit, signal):
        orders_to_submit: List[Order] = []
        buy_signal = sell_signal = signal
        cur_pos = starting_pos
        ob_bids = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        ob_asks = collections.OrderedDict(sorted(order_book.sell_orders.items()))
        
        buy_volume_avail = pos_limit - starting_pos
        sell_volume_avail = abs(-pos_limit - starting_pos)

        assert buy_volume_avail+sell_volume_avail == 40 and buy_volume_avail <= 40 and sell_volume_avail <= 40, "someting wrong"

        #Market taking starting with lowest ask, here we are doing BUYING (if the price is below buy_signal)
        for ask, avolume in ob_asks.items():
            if (ask < buy_signal or (ask == buy_signal and starting_pos > 0)) and buy_volume_avail > 0:
                max_vol_tradable = min(abs(avolume), buy_volume_avail)
                orders_to_submit.append(Order("AMETHYSTS", int(ask), int(max_vol_tradable)))
                buy_volume_avail -= max_vol_tradable
                cur_pos += max_vol_tradable

        assert abs(cur_pos) <= pos_limit, f"error in {cur_pos, buy_volume_avail, sell_volume_avail} amethysts MT buy_pos"
        assert buy_volume_avail >= 0, f"error in {cur_pos, buy_volume_avail, sell_volume_avail} amethysts MT buy_vol"
        
                
        #Market taking starting with highest bid, here we are doing SELLING (if the price is above sell_signal)
        for bid, bvolume in ob_bids.items():
            if (bid > sell_signal or (bid == sell_signal and starting_pos < 0)) and sell_volume_avail > 0:
                max_vol_tradable = min(bvolume, sell_volume_avail)
                orders_to_submit.append(Order("AMETHYSTS", int(bid), int(-max_vol_tradable)))
                sell_volume_avail -= max_vol_tradable
                cur_pos -= max_vol_tradable

        assert abs(cur_pos) <= pos_limit, f"error in {cur_pos, buy_volume_avail, sell_volume_avail} amethysts MT sell_pos"
        assert sell_volume_avail >= 0, f"error in {cur_pos, buy_volume_avail, sell_volume_avail} amethysts MT sell_vol"
                
        return cur_pos, buy_volume_avail, sell_volume_avail, orders_to_submit
    
    def order_gen_AMETHYSTS_MM(self, order_book, current_pos, buy_vol_avail, sell_vol_avail, pos_limit, signal):     
        orders_to_submit: List[Order] = []

        buy_signal = sell_signal = signal

        filtered_bids = {k: v for k, v in order_book.buy_orders.items() if k < signal}
        filtered_asks = {k: v for k, v in order_book.sell_orders.items() if k > signal}

        ob_bids = collections.OrderedDict(sorted(filtered_bids.items(), reverse=True))
        ob_asks = collections.OrderedDict(sorted(filtered_asks.items()))

        _, highest_bid, _ = self.calc_metrics_bids(ob_bids)
        _, lowest_ask, _ = self.calc_metrics_asks(ob_asks)

        buy_volume_avail = buy_vol_avail
        sell_volume_avail = sell_vol_avail


        undercut_ask = lowest_ask - 1
        undercut_bid = highest_bid + 1

        bid_am = min(undercut_bid, buy_signal-1) 
        sell_am = max(undercut_ask, sell_signal+1)

        #Market Making for sending outstanding BIDS:
        #Case 1: If our initial pos at start of tick is greatly negative, we want to buy (to get our position to 0), so we "relax" the "undercut" bid a little, but still min it with the buy_signal to make sure we are sending "good orders" (under the buy signal)
        if buy_volume_avail > 0:
            orders_to_submit.append(Order("AMETHYSTS", int(bid_am), int(buy_volume_avail)))
            buy_vol_avail-=buy_vol_avail

        assert abs(current_pos) <= pos_limit, f"error in {current_pos,buy_vol_avail,sell_vol_avail} amethysts MM buy_pos"
        assert sell_vol_avail >= 0, f"error in {sell_vol_avail,current_pos} amethysts MM buy_vol"
        #Market Making for sending outstanding ASKS (cases are the reverse of bids)

        #Case 1: Greatly positive, so agressive selling
        if sell_volume_avail:
            orders_to_submit.append(Order("AMETHYSTS", int(sell_am), int(-sell_volume_avail)))
            sell_vol_avail -= sell_volume_avail

        assert abs(current_pos) <= pos_limit, f"error in {current_pos,buy_vol_avail,sell_vol_avail} amethysts MM sell_pos"
        assert sell_vol_avail >= 0, f"error in {sell_vol_avail,current_pos} amethysts MM sell_vol"
        return orders_to_submit

    #STARFRUIT
    def order_gen_STARFRUIT_MT(self, order_book, starting_pos, pos_limit):
        orders_to_submit: List[Order] = []

        ob_bids = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        ob_asks = collections.OrderedDict(sorted(order_book.sell_orders.items()))

        bids_volume,_,bvwap = self.calc_metrics_bids(ob_bids)
        asks_volume,_,avwap = self.calc_metrics_asks(ob_asks)

        buy_signal = sell_signal = (bvwap*bids_volume+avwap*asks_volume)/(bids_volume+asks_volume)

        buy_volume_avail = pos_limit - starting_pos
        sell_volume_avail = -pos_limit - starting_pos
        cur_pos = starting_pos


        #Market taking starting with lowest ask, here we are doing BUYING (if the price is below buy_signal)
        for ask, avolume in ob_asks.items():
            if (ask < buy_signal or (ask == buy_signal and starting_pos > 0)) and buy_volume_avail > 0:
                max_vol_tradable = min(abs(avolume), buy_volume_avail)
                orders_to_submit.append(Order("STARFRUIT", int(ask), int(max_vol_tradable)))
                buy_volume_avail -= max_vol_tradable
                cur_pos += max_vol_tradable

        assert abs(cur_pos) <= pos_limit, f"error in {cur_pos, buy_volume_avail, sell_volume_avail} starfruit MT buy_pos"
        assert buy_volume_avail >= 0, f"error in {cur_pos, buy_volume_avail, sell_volume_avail} starfruit MT buy_vol"
                
        #Market taking starting with highest bid, here we are doing SELLING (if the price is above sell_signal)
        for bid, bvolume in ob_bids.items():
            if (bid > sell_signal  or (bid == sell_signal and starting_pos < 0)) and sell_volume_avail > 0:
                max_vol_tradable = min(bvolume, sell_volume_avail)
                orders_to_submit.append(Order("STARFRUIT", int(bid), int(-max_vol_tradable)))
                sell_volume_avail -= max_vol_tradable
                cur_pos -= max_vol_tradable

        assert abs(cur_pos) <= pos_limit, f"error in {cur_pos, buy_volume_avail, sell_volume_avail} starfruit MT sell_pos"
        assert buy_volume_avail >= 0, f"error in {cur_pos, buy_volume_avail, sell_volume_avail} starfruit MT sell_vol"

        return cur_pos, buy_volume_avail, sell_volume_avail, orders_to_submit
        
    def order_gen_STARFRUIT_MM(self, order_book, current_pos, buy_vol_avail, sell_vol_avail, pos_limit):   
        orders_to_submit: List[Order] = []

        #Note that after MM, current_pos might be inaccurate due to orders not being filled

        ob_bids = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        ob_asks = collections.OrderedDict(sorted(order_book.sell_orders.items()))

        _,_,highest_bid = self.calc_metrics_bids(ob_bids)
        _,_,lowest_ask = self.calc_metrics_asks(ob_asks)

        buy_signal = sell_signal = (highest_bid+lowest_ask)/2

        buy_volume_avail = buy_vol_avail
        sell_volume_avail = sell_vol_avail

        undercut_ask = lowest_ask - 1
        undercut_bid = highest_bid + 1

        bid_am = min(undercut_bid, buy_signal-1) 
        sell_am = max(undercut_ask, sell_signal+1)
        
        #Market Making for sending outstanding BIDS:

        #Case 1: If our initial pos at start of tick is greatly negative, we want to buy (to get our position to 0), so we "relax" the "undercut" bid a little, but still min it with the buy_signal to make sure we are sending "good orders" (under the buy signal)
        if (buy_volume_avail > 0) and (current_pos < -7):
            orders_to_submit.append(Order("STARFRUIT", int(min(undercut_bid + 1, buy_signal-1)), int(buy_volume_avail)))
            buy_vol_avail-=buy_vol_avail
            current_pos += buy_volume_avail
        #Case 2: If our initial pos is greatly positive, we are ok with being more restrictive with selling, so we do the opposite of above
        elif (buy_volume_avail > 0) and (current_pos > 7):
            orders_to_submit.append(Order("STARFRUIT", int(min(undercut_bid - 1, buy_signal-1)), int(buy_volume_avail)))
            buy_vol_avail-=buy_vol_avail
            current_pos += buy_volume_avail
        #Case 3: If neither, just use send in relatively strong orders
        elif buy_volume_avail > 0:
            orders_to_submit.append(Order("STARFRUIT", int(bid_am), int(buy_volume_avail)))
            buy_vol_avail-=buy_vol_avail
            current_pos += buy_volume_avail

        assert abs(current_pos) <= pos_limit, f"error in {current_pos, buy_volume_avail, sell_volume_avail} starfruit MM buy_pos"
        assert buy_vol_avail >= 0, f"error in {current_pos, buy_volume_avail, sell_volume_avail} starfruit MM buy_vol"

        #Market Making for sending outstanding ASKS (cases are the reverse of bids)
        if (sell_volume_avail) and (current_pos > 7):
            orders_to_submit.append(Order("STARFRUIT", int(max(undercut_ask-1, sell_signal+1)), int(sell_volume_avail)))
            sell_vol_avail-=sell_volume_avail
            current_pos -= buy_volume_avail
        elif (sell_volume_avail) and (current_pos < -7):
            orders_to_submit.append(Order("STARFRUIT", int(max(undercut_ask+1, sell_signal+1)), int(sell_volume_avail)))
            sell_vol_avail-=sell_volume_avail
            current_pos -= buy_volume_avail
        elif sell_volume_avail:
            orders_to_submit.append(Order("STARFRUIT", int(sell_am), int(sell_volume_avail)))
            sell_vol_avail-=sell_volume_avail
            current_pos -= buy_volume_avail

        assert abs(current_pos) <= pos_limit, f"error in {current_pos, buy_volume_avail, sell_volume_avail} starfruit MM sell_pos"
        assert sell_vol_avail >= 0, f"error in {current_pos, buy_volume_avail, sell_volume_avail} starfruit MM sell_vol"
        return orders_to_submit
    
    #ORCHIDS
    def arb_orders_ORCHID(self, state, pos, pos_limit):  
        #since we can only take a long/short conversion each timestep:
        #
        #   1) find any arb opportunities: buy local - sell from south OR sell local - buy from south
        #   2) see which opportunity will yield a higher pnl

        bid_orders_to_submit: List[Order] = []
        ask_orders_to_submit: List[Order] = []

        order_book              = state.order_depths["ORCHIDS"]
        local_bid_prices       = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        local_ask_prices        = collections.OrderedDict(sorted(order_book.sell_orders.items()))
        buy_volume_avail        = pos_limit - pos
        sell_volume_avail       = abs(-pos_limit - pos)

        south_island_info       = state.observations.conversionObservations["ORCHIDS"]
        adj_south_ask_price     = south_island_info.askPrice + south_island_info.transportFees + south_island_info.importTariff
        adj_south_bid_price    = south_island_info.bidPrice - south_island_info.transportFees - south_island_info.exportTariff

        bLocal_sSouth_pnl = 0
        sLocal_bSouth_pnl = 0

        #here we buy local (at the ask), and will sell at the `adj_south_ask_price`
        for local_ask_price, ask_volume in local_ask_prices.items():
            if (local_ask_price < adj_south_bid_price) and buy_volume_avail > 0:
                logger.print("g1 bLocal,sSouth:", local_ask_price, adj_south_ask_price)
                max_vol_tradable = min(abs(ask_volume), buy_volume_avail)
                bid_orders_to_submit.append(Order("ORCHIDS", int(local_ask_price), int(max_vol_tradable)))
                bLocal_sSouth_pnl += (adj_south_bid_price-local_ask_price)*max_vol_tradable
                local_ask_prices[local_ask_price] -= max_vol_tradable
                local_ask_prices = {k: v for k, v in local_ask_prices.items() if k > 0}
                buy_volume_avail -= max_vol_tradable
                
        #here we sell local (at the bid), and will buy at the `adj_south_bid_price`
        for local_bid_price, bid_volume in local_bid_prices.items():
            if (adj_south_ask_price < local_bid_price) and sell_volume_avail > 0:
                logger.print("g2 sLocal,bSouth:", local_bid_price, adj_south_bid_price)
                max_vol_tradable = min(bid_volume, sell_volume_avail)
                ask_orders_to_submit.append(Order("ORCHIDS", int(local_bid_price), int(-max_vol_tradable)))
                sLocal_bSouth_pnl += (local_bid_price-adj_south_ask_price)*max_vol_tradable
                local_bid_prices[local_bid_price] -= max_vol_tradable
                local_bid_prices = {k: v for k, v in local_bid_prices.items() if k > 0}
                sell_volume_avail -= max_vol_tradable

        #Now we decide which side to MM on, bLocal,sSouth or sLocal,bSouth BECAUSE you can only do conversions simalaniously in short/long
                
        #hyperparam
        margin = 1
    
        #if there's sure gains in one direction
        if bLocal_sSouth_pnl > sLocal_bSouth_pnl:
            if buy_volume_avail > 0:
                logger.print(adj_south_bid_price,adj_south_ask_price)
                bid_orders_to_submit.append(Order("ORCHIDS", int(round(adj_south_bid_price,0)-margin), buy_volume_avail))
                return bid_orders_to_submit
        elif bLocal_sSouth_pnl < sLocal_bSouth_pnl:
            if sell_volume_avail > 0:
                logger.print(adj_south_bid_price,adj_south_ask_price)
                ask_orders_to_submit.append(Order("ORCHIDS", int(round(adj_south_ask_price,0)+margin), -sell_volume_avail))
                return ask_orders_to_submit
        #if not MM based on closest strat to an arb (both will be negative)
        else:
            logger.print("nope! MM")
            _,lowest_local_ask_price,_ = self.calc_metrics_asks(local_ask_prices)
            _,highest_local_bid_price,_ = self.calc_metrics_bids(local_bid_prices)

            logger.print(adj_south_bid_price,adj_south_ask_price)

            bLocal_sSouth_pnl = adj_south_bid_price - lowest_local_ask_price
            sLocal_bSouth_pnl = highest_local_bid_price - adj_south_ask_price

            logger.print(bLocal_sSouth_pnl,"?", sLocal_bSouth_pnl)

            if bLocal_sSouth_pnl > sLocal_bSouth_pnl: 
                return [Order("ORCHIDS", int(round(adj_south_bid_price,0)-margin), buy_volume_avail)]
            else:
                return [Order("ORCHIDS", int(round(adj_south_ask_price,0)+margin), -sell_volume_avail)]
        
    #GIFT_BASKET products
    def order_gen_GIFT_BASKET(self, state):
        basket_orders = []
        strawb_orders = []
        choc_orders = []
        rose_orders = []
        basket_pos_limit = 54
        strawb_pos_limit = 315
        choc_pos_limit = 225
        rose_pos_limit = 54

        #getting the current position, buy/sell vol available (this is for liquidation), and the best bid/ask for all products
        current_basket_pos = state.position.get("GIFT_BASKET", 0)
        basket_buy_vol = basket_pos_limit - current_basket_pos
        basket_sell_vol = abs(-basket_pos_limit - current_basket_pos)
        basket_buy_vol_ext = 60 - current_basket_pos
        basket_sell_vol_ext = abs(-60- current_basket_pos)
        _,basket_sell_price,_ = self.calc_metrics_bids(state.order_depths["GIFT_BASKET"].buy_orders) #buying at the ask
        _,basket_buy_price,_ = self.calc_metrics_asks(state.order_depths["GIFT_BASKET"].sell_orders) #selling at the bid

        current_strawb_pos = state.position.get("STRAWBERRIES", 0) 
        strawb_buy_vol = strawb_pos_limit - current_strawb_pos 
        strawb_sell_vol = abs(-strawb_pos_limit - current_strawb_pos)
        strawb_buy_vol_ext = 350 - current_strawb_pos 
        strawb_sell_vol_ext = abs(-350 - current_strawb_pos)
        _,strawb_highest_bid,_ = self.calc_metrics_bids(state.order_depths["STRAWBERRIES"].buy_orders)
        _,strawb_lowest_ask,_ = self.calc_metrics_asks(state.order_depths["STRAWBERRIES"].sell_orders)

        current_choc_pos = state.position.get("CHOCOLATE", 0) 
        choc_buy_vol = choc_pos_limit - current_choc_pos
        choc_sell_vol = abs(-choc_pos_limit - current_choc_pos)
        choc_buy_vol_ext = 250 - current_choc_pos
        choc_sell_vol_ext = abs(-250 - current_choc_pos)
        _,choc_highest_bid,_ = self.calc_metrics_bids(state.order_depths["CHOCOLATE"].buy_orders)
        _,choc_lowest_ask,_ = self.calc_metrics_asks(state.order_depths["CHOCOLATE"].sell_orders)

        current_rose_pos = state.position.get("ROSES", 0) 
        rose_buy_vol = rose_pos_limit - current_rose_pos
        rose_sell_vol = abs(-rose_pos_limit - current_rose_pos)
        rose_buy_vol_ext = 60 - current_rose_pos
        rose_sell_vol_ext = abs(-60 - current_rose_pos)
        _,rose_highest_bid,_ = self.calc_metrics_bids(state.order_depths["ROSES"].buy_orders)
        _,rose_lowest_ask,_ = self.calc_metrics_asks(state.order_depths["ROSES"].sell_orders)

        adjusted_debasket_buy_price = strawb_lowest_ask*6+choc_lowest_ask*4+rose_lowest_ask + 380       #buying at the ask
        adjusted_debasket_sell_price = strawb_highest_bid*6+choc_highest_bid*4+rose_highest_bid + 380   #selling at the bid
        basket_mid_p = (basket_buy_price+basket_sell_price)/2
        debasket_mid_p = (adjusted_debasket_buy_price+adjusted_debasket_sell_price)/2
        
        logger.print("basket mid", basket_mid_p)
        logger.print("debasket mid", debasket_mid_p)
        logger.print(basket_buy_price-debasket_mid_p)


        #Start averaging into a position
        if abs(basket_mid_p-debasket_mid_p) > 90:
            if basket_mid_p < debasket_mid_p:
                logger.print("enter")
                logger.print(basket_buy_vol)
                basket_orders.append(Order("GIFT_BASKET", basket_buy_price, min(1,basket_buy_vol_ext)))
                strawb_orders.append(Order("STRAWBERRIES", strawb_lowest_ask, min(6,strawb_sell_vol_ext)))
                choc_orders.append(Order("CHOCOLATE", choc_lowest_ask, min(4,choc_sell_vol_ext)))
                rose_orders.append(Order("ROSES", rose_lowest_ask, min(1,rose_sell_vol_ext)))
            else:
                logger.print("enter2")
                basket_orders.append(Order("GIFT_BASKET", basket_sell_price, -min(1,basket_sell_vol_ext)))
                strawb_orders.append(Order("STRAWBERRIES", strawb_highest_bid, -min(6,strawb_buy_vol_ext)))
                choc_orders.append(Order("CHOCOLATE", choc_highest_bid, -min(4, choc_buy_vol_ext)))
                rose_orders.append(Order("ROSES", rose_highest_bid, -min(1, rose_buy_vol_ext)))
        elif abs(basket_mid_p-debasket_mid_p) > 50:
            if basket_mid_p < debasket_mid_p:
                logger.print("enter")
                logger.print(basket_buy_vol)
                basket_orders.append(Order("GIFT_BASKET", basket_buy_price, min(1,basket_buy_vol)))
                strawb_orders.append(Order("STRAWBERRIES", strawb_lowest_ask, min(6,strawb_sell_vol)))
                choc_orders.append(Order("CHOCOLATE", choc_lowest_ask, min(4,choc_sell_vol)))
                rose_orders.append(Order("ROSES", rose_lowest_ask, min(1,rose_sell_vol)))
            else:
                logger.print("enter2")
                basket_orders.append(Order("GIFT_BASKET", basket_sell_price, -min(1,basket_sell_vol)))
                strawb_orders.append(Order("STRAWBERRIES", strawb_highest_bid, -min(6,strawb_buy_vol)))
                choc_orders.append(Order("CHOCOLATE", choc_highest_bid, -min(4, choc_buy_vol)))
                rose_orders.append(Order("ROSES", rose_highest_bid, -min(1, rose_buy_vol)))
        
        elif abs(basket_mid_p-debasket_mid_p) < 15:
            logger.print("exit")
            basket_pos = state.position.get('GIFT_BASKET', 0)
            strawb_pos = state.position.get('STRAWBERRIES', 0)
            choc_pos = state.position.get('CHOCOLATE', 0)
            rose_pos = state.position.get('ROSES', 0)
            if(basket_pos < 0):
                basket_orders.append(Order("GIFT_BASKET", basket_buy_price, -basket_pos))
            else:
                basket_orders.append(Order("GIFT_BASKET", basket_sell_price, -basket_pos))

            if(strawb_pos < 0):
                strawb_orders.append(Order("STRAWBERRIES", strawb_lowest_ask, -strawb_pos))
            else:
                strawb_orders.append(Order("STRAWBERRIES", strawb_highest_bid, -strawb_pos))

            if(choc_pos < 0):
                choc_orders.append(Order("CHOCOLATE", choc_lowest_ask, -choc_pos))
            else:
                choc_orders.append(Order("CHOCOLATE", choc_highest_bid, -choc_pos))

            if(rose_pos < 0):
                rose_orders.append(Order("ROSES", rose_lowest_ask, -rose_pos))
            else:
                rose_orders.append(Order("ROSES", rose_highest_bid, -rose_pos))
        
        return basket_orders, strawb_orders, choc_orders, rose_orders

    #COCONUT and COCONUT_COUPON
    def order_gen_COCONUT(self,state):
        coconut_pos = state.position.get("COCONUT", 0)
        coconut_limit = 300
        coconut_orders = []
        coconut_buy_volume_avail = coconut_limit - coconut_pos
        coconut_sell_volume_avail = abs(-coconut_limit - coconut_pos)

        coconut_ask_volume, coconut_best_ask, coconut_ask_vwap = self.calc_metrics_asks(state.order_depths["COCONUT"].sell_orders)
        coconut_bid_volume, coconut_best_bid, coconut_bid_vwap = self.calc_metrics_bids(state.order_depths["COCONUT"].buy_orders)
        coconut_midprice = (coconut_best_ask+coconut_best_bid)/2

        coupon_pos = state.position.get("COCONUT_COUPON",0)
        coupon_limit = 600
        coupon_orders = []
        coupon_buy_volume_avail = coupon_limit - coupon_pos
        coupon_sell_volume_avail = abs(-coupon_limit - coupon_pos)

        coupon_ask_volume, coupon_best_ask, coupon_ask_vwap = self.calc_metrics_asks(state.order_depths["COCONUT_COUPON"].sell_orders)
        coupon_bid_volume, coupon_best_bid, coupon_bid_vwap = self.calc_metrics_bids(state.order_depths["COCONUT_COUPON"].buy_orders)
        coupon_midprice = (coupon_best_ask+coupon_best_bid)/2


        coconut_volatility = 0.16
        theo_price = round(self.BS_call_calc(coconut_midprice, 10000, coconut_volatility, (246-(state.timestamp/(100*10000)))))


        #hyperparam
        enter_trade = 0
        exit_trade = 0
        assert exit_trade <= enter_trade

        logger.print(coconut_pos, coconut_buy_volume_avail, coconut_sell_volume_avail)
        logger.print(coupon_pos, coupon_buy_volume_avail, coupon_sell_volume_avail)

        if abs(coupon_midprice - theo_price) > enter_trade:
            #entering position
            if coupon_midprice < theo_price:
                coupon_orders.append(Order("COCONUT_COUPON", coupon_best_ask, coupon_buy_volume_avail))
                coconut_orders.append(Order("COCONUT", coconut_best_bid, -coconut_sell_volume_avail))
            elif coupon_midprice > theo_price:
                coupon_orders.append(Order("COCONUT_COUPON", coupon_best_bid, -coupon_sell_volume_avail))
                coconut_orders.append(Order("COCONUT", coconut_best_ask, coconut_buy_volume_avail))
        else:
            #liquidation
            if coupon_pos < 0:
                coupon_orders.append(Order("COCONUT_COUPON", coupon_best_ask, -coupon_pos))
            elif coupon_pos > 0:
                coupon_orders.append(Order("COCONUT_COUPON", coupon_best_bid, -coupon_pos))

            if coconut_pos < 0:
                coconut_orders.append(Order("COCONUT", coconut_best_ask, -coconut_pos))
            elif coconut_pos > 0:
                coconut_orders.append(Order("COCONUT", coconut_best_bid, -coconut_pos))

        return coconut_orders, coupon_orders


    #Live function
    def run(self, state: TradingState):
        conversions = 0
        result = {}
        traderData = ""

        if state.traderData:
            mp_price_history = jsonpickle.decode(state.traderData)
        else:
            mp_price_history : Dict[str, List[int]] = {"STARFRUIT": [], "AMETHYSTS": [], "ORCHIDS_LOCAL": [], "ORCHIDS_SOUTH": [], "GIFT_BASKET": [], "STRAWBERRIES": [], "CHOCOLATE": [], "ROSES": []}    

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            # if product == "AMETHYSTS":
            #     current_am_pos = state.position.get("AMETHYSTS", 0)
            #     mp_price_history = self.update_prev_prices(state, mp_price_history, product)
            #     pos_after_mt, buy_vol_remain, sell_vol_remain, orders_MT = self.order_gen_AMETHYSTS_MT(order_depth, current_am_pos, 20, 10000)
            #     orders_MM = self.order_gen_AMETHYSTS_MM(order_depth, pos_after_mt, buy_vol_remain, sell_vol_remain, 20, 10000)
            #     orders = orders_MT + orders_MM
            #     result[product] = orders
            # elif product == "STARFRUIT":
            #     current_star_pos = state.position.get("STARFRUIT", 0)
            #     mp_price_history = self.update_prev_prices(state, mp_price_history, product)
            #     star_signal = self.calc_regression(mp_price_history[product])
            #     pos_after_mt, buy_vol_remain, sell_vol_remain, orders_MT = self.order_gen_STARFRUIT_MT(order_depth, current_star_pos, 20)
            #     orders_MM = self.order_gen_STARFRUIT_MM(order_depth, pos_after_mt, buy_vol_remain, sell_vol_remain, star_signal)
            #     orders = orders_MT + orders_MM
            #     result[product] = orders
            #     result[product] = orders
            if product == "ORCHIDS":
                current_orch_pos = state.position.get("ORCHIDS", 0) 
                logger.print("cur_pos = ",current_orch_pos)
                order_depth: OrderDepth = state.order_depths[product]
                logger.print("lSell @:", order_depth.buy_orders)
                logger.print("lBuy @:", order_depth.sell_orders)
                orders = self.arb_orders_ORCHID(state, current_orch_pos, 100)
                logger.print("orders",orders)
                conversions = current_orch_pos*-1
                result[product] = orders
            # if product == "GIFT_BASKET":
            #     basket_orders, strawb_orders, choc_orders, rose_orders = self.order_gen_GIFT_BASKET(state)
            #     logger.print(basket_orders)
            #     logger.print(strawb_orders)
            #     logger.print(choc_orders)
            #     logger.print(rose_orders)
            #     result[product] = basket_orders

             #   Not trading on these products because they're way too stochastic to predict in a meaningful way
                # result["STRAWBERRIES"] = strawb_orders
                # result["CHOCOLATE"] = choc_orders 
                # result["ROSES"] = rose_orders
            # if product == "COCONUT":
            #     coconut_orders, coupon_orders = self.order_gen_COCONUT(state)
            #     result["COCONUT_COUPON"] = coupon_orders
            #     result["COCONUT"] = coconut_orders
 

        traderData = jsonpickle.encode(mp_price_history)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData