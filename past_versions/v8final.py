import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string

#allowed imports
import pandas as pd
import numpy as np
import statistics
import math
import typing
import jsonpickle

#native libraries
import copy
import collections

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
    def calc_metrics_bids(self, price_dict):
        volume = 0 
        highest_bid = -1
        bids_vwap = -1

        #In this function we are looping through the bids from highest (most attractive) to lowest (least attractive)
        #We use this function to find the three most important metrics: total volume of bids, highest (most attractive) bid, and the total vwap of bids
        if len(price_dict) > 0:
            price_dict = collections.OrderedDict(sorted(price_dict.items(), reverse=True))
            for index, (key,value) in enumerate(price_dict.items()):
                if index == 0:
                    highest_bid = key
                volume += value
                bids_vwap += key*value
            
            bids_vwap /= volume

        return volume, highest_bid, bids_vwap
    
    def calc_metrics_ask(self, price_dict):
        volume = 0 
        lowest_ask = -1
        ask_vwap = -1

        #In this function we are looping through the asks from lowest (most attractive) to highest (least attractive)
        #We use this function to find the three most important metrics: total volume of asks, lowest (most attractive) bid, and the total vwap of asks
        if len(price_dict) > 0:
            price_dict = collections.OrderedDict(sorted(price_dict.items()))
            for index, (key,value) in enumerate(price_dict.items()):
                if index == 0:
                    lowest_ask = key
                volume += value
                ask_vwap += key*value
            
            ask_vwap /= volume

        return -1*volume, lowest_ask, ask_vwap


    #AMETHYSTS
    def order_gen_AMETHYSTS_MT(self, order_book, starting_pos, pos_limit, buy_signal, sell_signal):
        orders_to_submit: List[Order] = []

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
    
    def order_gen_AMETHYSTS_MM(self, order_book, current_pos, buy_vol_avail, sell_vol_avail, pos_limit, buy_signal, sell_signal):     
        orders_to_submit: List[Order] = []



        filtered_bids = {k: v for k, v in order_book.buy_orders.items() if k < 10000}
        filtered_asks = {k: v for k, v in order_book.sell_orders.items() if k > 10000}

        ob_bids = collections.OrderedDict(sorted(filtered_bids.items(), reverse=True))
        ob_asks = collections.OrderedDict(sorted(filtered_asks.items()))

        _, highest_bid, _ = self.calc_metrics_bids(ob_bids)
        _, lowest_ask, _ = self.calc_metrics_ask(ob_asks)

        buy_volume_avail = buy_vol_avail
        sell_volume_avail = sell_vol_avail

        #Market Making:
        current_theo = (buy_signal+sell_signal)/2


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
    #STARFRUIT
    def order_gen_STARFRUIT_MT(self, order_book, starting_pos, pos_limit):
        orders_to_submit: List[Order] = []

        ob_bids = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        ob_asks = collections.OrderedDict(sorted(order_book.sell_orders.items()))

        bids_volume,_,bvwap = self.calc_metrics_bids(ob_bids)
        asks_volume,_,avwap = self.calc_metrics_ask(ob_asks)

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
            if (bid > sell_signal or (bid == sell_signal and starting_pos < 0)) and sell_volume_avail > 0:
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
        _,_,lowest_ask = self.calc_metrics_ask(ob_asks)

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


    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        trader_data = ""
        

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "AMETHYSTS":
                logger.print(state.order_depths[product].buy_orders)
                current_am_pos = state.position.get("AMETHYSTS", 0)
                pos_after_mt, buy_vol_remain, sell_vol_remain, orders_MT = self.order_gen_AMETHYSTS_MT(order_depth, current_am_pos, 20, 10000, 10000)
                orders_MM = self.order_gen_AMETHYSTS_MM(order_depth, pos_after_mt, buy_vol_remain, sell_vol_remain, 20, 10000, 10000)
                orders = orders_MT + orders_MM
                
            elif product == "STARFRUIT":
                

                current_str_pos = state.position.get("STARFRUIT", 0)
                pos_after_mt, buy_vol_remain, sell_vol_remain, orders_MT = self.order_gen_STARFRUIT_MT(order_depth, current_str_pos, 20)
                orders_MM = self.order_gen_STARFRUIT_MM(order_depth, pos_after_mt, buy_vol_remain, sell_vol_remain, 20)
                orders = orders_MT + orders_MM

            result[product] = orders
        
        conversions = 1
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data