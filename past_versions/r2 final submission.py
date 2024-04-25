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
        avolume,_,avwap = self.calc_metrics_ask(sell_orders_to_submit_dict) #could use bids too if our values werent neg

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
        

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        if state.traderData:
            mp_price_history = jsonpickle.decode(state.traderData)
        else:
            mp_price_history : Dict[str, List[int]] = {"STARFRUIT": [], "AMETHYSTS": []}    

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if product == "AMETHYSTS":
                current_am_pos = state.position.get("AMETHYSTS", 0)
                mp_price_history = self.update_prev_prices(state, mp_price_history, product)
                pos_after_mt, buy_vol_remain, sell_vol_remain, orders_MT = self.order_gen_AMETHYSTS_MT(order_depth, current_am_pos, 20, 10000)
                orders_MM = self.order_gen_AMETHYSTS_MM(order_depth, pos_after_mt, buy_vol_remain, sell_vol_remain, 20, 10000)
                orders = orders_MT + orders_MM
            elif product == "STARFRUIT":
                current_star_pos = state.position.get("STARFRUIT", 0)
                mp_price_history = self.update_prev_prices(state, mp_price_history, product)
                star_signal = self.calc_regression(mp_price_history[product])
                pos_after_mt, buy_vol_remain, sell_vol_remain, orders_MT = self.order_gen_STARFRUIT_MT(order_depth, current_star_pos, 20)
                orders_MM = self.order_gen_STARFRUIT_MM(order_depth, pos_after_mt, buy_vol_remain, sell_vol_remain, star_signal)
                orders = orders_MT + orders_MM
            if product == "ORCHIDS":
                current_orch_pos = state.position.get("ORCHIDS", 0) 
                orders = self. arb_orders_ORCHID(state, current_orch_pos, 100)
                conversions = current_orch_pos*-1

            result[product] = orders
        
        traderData = jsonpickle.encode(mp_price_history)
        return result, conversions, traderData