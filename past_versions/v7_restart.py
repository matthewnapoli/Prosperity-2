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

Product = str

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

        return volume, lowest_ask, ask_vwap
    
    def order_gen_starfruit(self, order_book, starting_pos, pos_limit):
        orders_to_submit: List[Order] = []
        print("Position before MM/MT Starfruit:", starting_pos)

        ob_bids = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        ob_asks = collections.OrderedDict(sorted(order_book.sell_orders.items()))

        _,_,highest_bid = self.calc_metrics_bids(ob_bids)
        _,_,lowest_ask = self.calc_metrics_ask(ob_asks)

        buy_signal = sell_signal = (highest_bid+lowest_ask)/2

        buy_volume_avail = pos_limit - starting_pos
        sell_volume_avail = -pos_limit - starting_pos

        #Market taking starting with lowest ask, here we are doing BUYING (if the price is below buy_signal)
        for ask, avolume in ob_asks.items():
            if (ask < buy_signal or (ask == buy_signal and starting_pos > 0)) and buy_volume_avail > 0:
                max_vol_tradable = min(avolume, buy_volume_avail)
                orders_to_submit.append(Order("STARFRUIT", int(ask), max_vol_tradable))
                buy_volume_avail -= max_vol_tradable
                
        #Market taking starting with highest bid, here we are doing SELLING (if the price is above sell_signal)
        for bid, bvolume in ob_bids.items():
            if (bid > sell_signal or (bid == sell_signal and starting_pos < 0)) and sell_volume_avail > 0:
                max_vol_tradable = min(bvolume, sell_volume_avail)
                orders_to_submit.append(Order("STARFRUIT", int(bid), -max_vol_tradable))
                sell_volume_avail -= max_vol_tradable
            
        #Market Making:
        market_midprice = (highest_bid+lowest_ask)/2
        current_theo = (buy_signal+sell_signal)/2

        undercut_ask = lowest_ask - 1
        undercut_bid = highest_bid + 1

        bid_am = min(undercut_bid, buy_signal-1) 
        sell_am = max(undercut_ask, sell_signal+1)
        
        #Market Making for sending outstanding BIDS:

        #Case 1: If our initial pos at start of tick is greatly negative, we want to buy (to get our position to 0), so we "relax" the "undercut" bid a little, but still min it with the buy_signal to make sure we are sending "good orders" (under the buy signal)
        if (buy_volume_avail > 0) and (starting_pos < -10):
            orders_to_submit.append(Order("STARFRUIT", int(min(undercut_bid + 1, buy_signal-1)), buy_volume_avail))
        #Case 2: If our initial pos is greatly positive, we are ok with being more restrictive with selling, so we do the opposite of above
        elif (buy_volume_avail > 0) and (starting_pos > 10):
            orders_to_submit.append(Order("STARFRUIT", int(min(undercut_bid - 1, buy_signal-1)), buy_volume_avail))
        #Case 3: If neither, just use send in relatively strong orders
        elif buy_volume_avail > 0:
            orders_to_submit.append(Order("STARFRUIT", int(bid_am), buy_volume_avail))

        #Market Making for sending outstanding ASKS (cases are the reverse of bids)
        if (sell_volume_avail) and (starting_pos > 10):
            orders_to_submit.append(Order("STARFRUIT", int(max(undercut_ask-1, sell_signal+1)), sell_volume_avail))
        elif (sell_volume_avail) and (starting_pos < -10):
            orders_to_submit.append(Order("STARFRUIT", int(max(undercut_ask+1, sell_signal+1)), sell_volume_avail))
        elif sell_volume_avail:
            orders_to_submit.append(Order("STARFRUIT", int(sell_am), sell_volume_avail))

        return orders_to_submit


    #"TO DO"
    def order_gen_amethysts(self, order_book, starting_pos, pos_limit, buy_signal, sell_signal):
        orders_to_submit: List[Order] = []

        cur_pos = starting_pos
        print("Position before MM/MT AM:", cur_pos)
        ob_bids = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        ob_asks = collections.OrderedDict(sorted(order_book.sell_orders.items()))

        _,_,highest_bid = self.calc_metrics_bids(ob_bids)
        _,_,lowest_ask = self.calc_metrics_ask(ob_asks)
        
        buy_volume_avail = pos_limit - starting_pos
        sell_volume_avail = -pos_limit - starting_pos

        #Market taking starting with lowest ask, here we are doing BUYING (if the price is below buy_signal)
        for ask, avolume in ob_asks.items():
            if (ask < buy_signal or (ask == buy_signal and starting_pos > 0)) and buy_volume_avail > 0:
                max_vol_tradable = min(avolume, buy_volume_avail)
                orders_to_submit.append(Order("AMETHYSTS", int(ask), max_vol_tradable))
                buy_volume_avail -= max_vol_tradable
                
        #Market taking starting with highest bid, here we are doing SELLING (if the price is above sell_signal)
        for bid, bvolume in ob_bids.items():
            if (bid > sell_signal or (bid == sell_signal and starting_pos < 0)) and sell_volume_avail > 0:
                max_vol_tradable = min(bvolume, sell_volume_avail)
                orders_to_submit.append(Order("AMETHYSTS", int(bid), -max_vol_tradable))
                sell_volume_avail -= max_vol_tradable
            
        #Market Making:
        market_midprice = (highest_bid+lowest_ask)/2
        current_theo = (buy_signal+sell_signal)/2

        undercut_ask = lowest_ask - 1
        undercut_bid = highest_bid + 1

        bid_am = min(undercut_bid, buy_signal-1) 
        sell_am = max(undercut_ask, sell_signal+1)
        
        #Market Making for sending outstanding BIDS:

        #Case 1: If our initial pos at start of tick is greatly negative, we want to buy (to get our position to 0), so we "relax" the "undercut" bid a little, but still min it with the buy_signal to make sure we are sending "good orders" (under the buy signal)
        if (buy_volume_avail > 0) and (starting_pos < -10):
            orders_to_submit.append(Order("AMETHYSTS", int(min(undercut_bid + 1, buy_signal-1)), buy_volume_avail))
        #Case 2: If our initial pos is greatly positive, we are ok with being more restrictive with selling, so we do the opposite of above
        elif (buy_volume_avail > 0) and (starting_pos > 10):
            orders_to_submit.append(Order("AMETHYSTS", int(min(undercut_bid - 1, buy_signal-1)), buy_volume_avail))
        #Case 3: If neither, just use send in relatively strong orders
        elif buy_volume_avail > 0:
            orders_to_submit.append(Order("AMETHYSTS", int(bid_am), buy_volume_avail))

        #Market Making for sending outstanding ASKS (cases are the reverse of bids)
        if (sell_volume_avail) and (starting_pos > 10):
            orders_to_submit.append(Order("AMETHYSTS", int(max(undercut_ask-1, sell_signal+1)), sell_volume_avail))
        elif (sell_volume_avail) and (starting_pos < -10):
            orders_to_submit.append(Order("AMETHYSTS", int(max(undercut_ask+1, sell_signal+1)), sell_volume_avail))
        elif sell_volume_avail:
            orders_to_submit.append(Order("AMETHYSTS", int(sell_am), sell_volume_avail))

        return orders_to_submit





    def run(self, state: TradingState):
        result: Dict[Product, List[Order]] = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "AMETHYSTS":
                print("Market prices AM:  ", order_depth.buy_orders, order_depth.sell_orders)
                orders = self.order_gen_amethysts(order_depth, state.position.get("AMETHYSTS", 0), 20, 10000, 10000)
                print("Submitted orders am: ", orders)
            elif product == "STARFRUIT":
                print("Market prices STARFRUIT:  ", order_depth.buy_orders, order_depth.sell_orders)
                orders = self.order_gen_starfruit(order_depth, state.position.get("STARFRUIT", 0), 20)
                print("Submitted orders star: ", orders)

            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData