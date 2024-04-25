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

        logger.print(self.to_json([
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
        if len(mp_price_history[product]) >= 4:
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
        assert n <= 5 and n >= 1, "bad input!"
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
    
    def calc_metrics_asks(self, price_dict): 
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
        buy_orders_to_submit: List[Order] = []
        buy_orders_to_submit_dict : Dict[int,int] = {}
        sell_orders_to_submit: List[Order] = []
        sell_orders_to_submit_dict : Dict[int,int] = {}

        buy_volume_avail = pos_limit - pos
        sell_volume_avail = abs(-pos_limit - pos)
        south_island_info = state.observations.conversionObservations["ORCHIDS"]
        adj_south_buy_price = south_island_info.askPrice + south_island_info.transportFees + south_island_info.importTariff
        adj_south_sell_price = south_island_info.bidPrice - south_island_info.transportFees - south_island_info.exportTariff
        order_book = state.order_depths["ORCHIDS"]
        local_sell_prices = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        local_buy_prices = collections.OrderedDict(sorted(order_book.sell_orders.items()))

        #here we buy local (at the ask), and will sell at the `adj_south_sell_price`
        for local_buy_price, buy_volume in local_buy_prices.items():
            if (local_buy_price < adj_south_sell_price) and buy_volume_avail > 0:
                print("good1: buy,sell: ", local_buy_price, adj_south_sell_price)
                max_vol_tradable = min(abs(buy_volume), buy_volume_avail)

                buy_orders_to_submit.append(Order("ORCHIDS", int(local_buy_price), int(max_vol_tradable)))
                buy_orders_to_submit_dict[int(local_buy_price)] = int(max_vol_tradable)
                buy_volume_avail -= max_vol_tradable
                
        #here we sell local (at the bid), and will buy at the `adj_south_buy_price`
        for local_sell_price, sell_volume in local_sell_prices.items():
            if (adj_south_buy_price < local_sell_price) and sell_volume_avail > 0:
                print("good2 sell,buy: ", local_sell_price, adj_south_buy_price)
                max_vol_tradable = min(sell_volume, sell_volume_avail)

                sell_orders_to_submit.append(Order("ORCHIDS", int(local_sell_price), int(-max_vol_tradable)))
                sell_orders_to_submit_dict[int(local_sell_price)] = int(-max_vol_tradable)
                sell_volume_avail -= max_vol_tradable

        bvolume,_,bvwap = self.calc_metrics_bids(buy_orders_to_submit_dict)
        avolume,_,avwap = self.calc_metrics_asks(sell_orders_to_submit_dict) #could use bids too if our values werent neg

        bLocal_sSouth_pnl = abs(bvwap-adj_south_buy_price)*bvolume
        sLocal_bSouth_pnl = abs(avwap-adj_south_sell_price)*avolume

        #since we can only take a long/short conversion each timestep:
        #
        #   1) find any arb opportunities: buy local - sell from south OR sell local - buy from south
        #   2) see which opportunity will yield a higher pnl
        #

        print("buy_orders_to_submit",buy_orders_to_submit)
        print("sell_orders_to_submit",sell_orders_to_submit)

        if bLocal_sSouth_pnl<=0 and sLocal_bSouth_pnl <= 0:
            _,highest_local_sell_price,_ = self.calc_metrics_bids(local_buy_prices)
            _,lowest_local_buy_price,_ = self.calc_metrics_bids(local_sell_prices)
            #if there is no outstsanding arb ops: MM on the side (long/short)that is the closest to a possible arb and trying to snag market orders
            if adj_south_sell_price-lowest_local_buy_price < adj_south_buy_price-highest_local_sell_price: 
                return [Order("ORCHIDS", int(adj_south_sell_price-2), buy_volume_avail)]
            else:
                return [Order("ORCHIDS", int(adj_south_buy_price+2), -1*sell_volume_avail)]
        elif(bLocal_sSouth_pnl > sLocal_bSouth_pnl) or (bLocal_sSouth_pnl == sLocal_bSouth_pnl and bvolume>avolume):
            buy_orders_to_submit.append(Order("ORCHIDS", int(adj_south_sell_price-2), buy_volume_avail))
            return buy_orders_to_submit
        else:
            sell_orders_to_submit.append(Order("ORCHIDS", int(adj_south_buy_price+2), -1*sell_volume_avail))
            return sell_orders_to_submit
        
    #GIFT_BASKET products
    def orders_gen_GIFT_BASKET(self, state):
        basket_orders = []
        strawb_orders = []
        choc_orders = []
        rose_orders = []
        basket_pos_limit = 60
        strawb_pos_limit = 350
        choc_pos_limit = 250
        rose_pos_limit = 60

        current_basket_pos = state.position.get("GIFT_BASKET", 0)
        basket_buy_vol = basket_pos_limit - current_basket_pos
        basket_sell_vol = abs(-basket_pos_limit - current_basket_pos)
        _,basket_sell_price,_ = self.calc_metrics_bids(state.order_depths["GIFT_BASKET"].buy_orders) #buying at the ask
        _,basket_buy_price,_ = self.calc_metrics_asks(state.order_depths["GIFT_BASKET"].sell_orders) #selling at the bid

        current_strawb_pos = state.position.get("STRAWBERRIES", 0) 
        strawb_buy_vol = strawb_pos_limit - current_strawb_pos 
        strawb_sell_vol = abs(-strawb_pos_limit - current_strawb_pos)
        _,strawb_highest_bid,_ = self.calc_metrics_bids(state.order_depths["STRAWBERRIES"].buy_orders)
        _,strawb_lowest_ask,_ = self.calc_metrics_asks(state.order_depths["STRAWBERRIES"].sell_orders)

        current_choc_pos = state.position.get("CHOCOLATE", 0) 
        choc_buy_vol = choc_pos_limit - current_choc_pos
        choc_sell_vol = abs(-choc_pos_limit - current_choc_pos)
        _,choc_highest_bid,_ = self.calc_metrics_bids(state.order_depths["CHOCOLATE"].buy_orders)
        _,choc_lowest_ask,_ = self.calc_metrics_asks(state.order_depths["CHOCOLATE"].sell_orders)

        current_rose_pos = state.position.get("ROSES", 0) 
        rose_buy_vol = rose_pos_limit - current_rose_pos
        rose_sell_vol = abs(-rose_pos_limit - current_rose_pos)
        _,rose_highest_bid,_ = self.calc_metrics_bids(state.order_depths["ROSES"].buy_orders)
        _,rose_lowest_ask,_ = self.calc_metrics_asks(state.order_depths["ROSES"].sell_orders)

        if abs(current_basket_pos) <= basket_pos_limit and abs(current_strawb_pos) <= strawb_pos_limit and abs(current_choc_pos) <= choc_pos_limit and abs(current_rose_pos) <= rose_pos_limit:
            adjusted_debasket_buy_price = strawb_lowest_ask*6+choc_lowest_ask*4+rose_lowest_ask + 380     #buying at the ask
            adjusted_debasket_sell_price = strawb_highest_bid*6+choc_highest_bid*4+rose_highest_bid + 380 #selling at the bid
            basket_mid_p = (basket_buy_price+basket_sell_price)/2
            debasket_mid_p = (adjusted_debasket_buy_price+adjusted_debasket_sell_price)/2
            logger.print("basket mid", basket_mid_p)
            logger.print("debasket mid", debasket_mid_p)
            logger.print(basket_buy_price-debasket_mid_p)
            #Profit off of buying basket and selling for parts
            if abs(basket_mid_p-debasket_mid_p) > 40:
                if basket_mid_p < debasket_mid_p:
                    logger.print("enter")
                    logger.print(basket_buy_vol)
                    basket_orders.append(Order("GIFT_BASKET", basket_buy_price-1, min(1,basket_buy_vol)-1)+1)
                    # strawb_orders.append(Order("STRAWBERRIES", strawb_highest_bid, -1*min(6,strawb_sell_vol)))
                    # choc_orders.append(Order("CHOCOLATE", choc_highest_bid, -1*min(4,choc_sell_vol)))
                    # rose_orders.append(Order("ROSES", rose_highest_bid, -1*min(1,rose_sell_vol)))
                else:
                    logger.print("enter2")
                    basket_orders.append(Order("GIFT_BASKET", basket_sell_price+1, -1*min(1,basket_sell_vol)))
                    # strawb_orders.append(Order("STRAWBERRIES", strawb_lowest_ask, min(6,strawb_buy_vol)))
                    # choc_orders.append(Order("CHOCOLATE", choc_lowest_ask, min(4, choc_buy_vol)))
                    # rose_orders.append(Order("ROSES", rose_lowest_ask, min(1, rose_buy_vol)))
            elif abs(basket_mid_p-debasket_mid_p) < 10:
                logger.print("exit")
                basket_pos = state.position.get('GIFT_BASKET',0)
                strawb_pos = state.position.get('STRAWBERRIES',0)
                choc_pos = state.position.get('CHOCOLATE',0)
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


    def run(self, state: TradingState):
        conversions = 0
        result = {}
        traderData = ""
        needs_orders = True

        if state.traderData:
            mp_price_history = jsonpickle.decode(state.traderData)
        else:
            mp_price_history : Dict[str, List[int]] = {"STARFRUIT": [], "AMETHYSTS": []}    

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
            # elif product == "ORCHIDS":
            #     current_orch_pos = state.position.get("ORCHIDS", 0) 
            #     orders = self.arb_orders_ORCHID(state, current_orch_pos, 100)
            #     conversions = current_orch_pos*-1
            #     result[product] = orders
            if product == "GIFT_BASKET":
                basket_orders, strawb_orders, choc_orders, rose_orders = self.orders_gen_GIFT_BASKET(state)
                logger.print(basket_orders)
                logger.print(strawb_orders)
                logger.print(choc_orders)
                logger.print(rose_orders)
                result[product] = basket_orders
                result["STRAWBERRIES"] = strawb_orders
                result["CHOCOLATE"] = choc_orders
                result["ROSES"] = rose_orders
        
        traderData = jsonpickle.encode(mp_price_history)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData