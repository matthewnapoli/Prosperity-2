from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import numpy as np

class Trader:
    def calculate_vwap(self, orderDepth_obj):
        buys = orderDepth_obj.buy_orders
        sells = orderDepth_obj.sell_orders
        order_book = {**buys, **sells}
        total_volume = sum(order_book.values())
        price_volume_sum = sum(price * volume for price, volume in order_book.items())
        if total_volume > 0:    
            vwap = price_volume_sum / total_volume
            return vwap
        else:
            return None

    def gen_orders(self, product, current_self_orders, MM_price, market_buy_orders, market_sell_orders, current_pos):
        #the market orders should already be sorted lowest to highest
        buy_volume_avail = 20 - current_pos
        sell_volume_avail = current_pos + 20 

        #looking for the lowests asks to buy in at
        for market_ask_price in market_sell_orders:
            if market_ask_price < MM_price and buy_volume_avail > 0:
                current_self_orders.append(Order(product, market_ask_price, min(buy_volume_avail, market_sell_orders[market_ask_price])))
                buy_volume_avail -= min(buy_volume_avail, market_sell_orders[market_ask_price])
            else:
                break

        #looking for the highests bids to sell off at
        for market_bid_price in market_buy_orders:
            if market_bid_price > MM_price and sell_volume_avail > 0:
                current_self_orders.append(Order(product, market_bid_price, -1*min(sell_volume_avail, market_buy_orders[market_bid_price])))
                sell_volume_avail -= min(sell_volume_avail, market_buy_orders[market_bid_price])
            else:
                break
        
        return current_self_orders
            
            
        
    def run(self, state: TradingState):
		# Orders to be placed on exchange matching engine
        result = {}
        next_tick = np.zeros(2)
        
        if len(state.traderData) > 0:
            print(state.traderData)
            starfruit_MM_price = float(state.traderData.split(",")[0])
            amethysts_MM_price = float(state.traderData.split(",")[1])
        else:
            starfruit_MM_price = self.calculate_vwap(state.order_depths["STARFRUIT"])
            amethysts_MM_price = self.calculate_vwap(state.order_depths["AMETHYSTS"])

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == "STARFRUIT":
                pos = state.position.get("AMETHYSTS", 0)
                if state.traderData is None:
                    break
                else:
                    orders = self.gen_orders(product, orders, starfruit_MM_price, dict(sorted(order_depth.buy_orders.items(),reverse=True)), dict(sorted(order_depth.sell_orders.items())), pos)
                    next_tick_star = self.calculate_vwap(order_depth)
                    if next_tick_star is not None: 
                        next_tick[0] = next_tick_star
                    else:
                        next_tick[0] = starfruit_MM_price

            if product == "AMETHYSTS":
                pos = state.position.get("AMETHYSTS", 0)
                if state.traderData is None:
                    break
                else:
                    orders = self.gen_orders(product, orders, amethysts_MM_price, dict(sorted(order_depth.buy_orders.items(),reverse=True)), dict(sorted(order_depth.sell_orders.items())), pos)
                    next_tick_ameth = self.calculate_vwap(order_depth)
                    if next_tick_ameth is not None: 
                        next_tick[1] = next_tick_ameth
                    else:
                        next_tick[1] = amethysts_MM_price

            result[product] = orders

        
        traderData = str(next_tick[0]) + "," + str(next_tick[1])        
        conversions = 1
        return result, conversions, traderData 