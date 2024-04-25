from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            if product == "STARFRUIT":
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []
                acceptable_price = 10  # Participant should calculate this value
                print("Acceptable price : " + str(acceptable_price))
                print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
        
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) < acceptable_price:
                        print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))
        
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if int(best_bid) > acceptable_price:
                        print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))

            if product == "AMETHYSTS":
                    pos = state.position.get("AMETHYSTS", 0)
                    filt_buy_orders = {key: value for key, value in order_depth.buy_orders.items() if key > 10000}
                    filt_sell_orders = {key: value for key, value in order_depth.sell_orders.items() if key < 10000}
                    while filt_buy_orders or filt_sell_orders:
                        if filt_buy_orders and abs(pos + 1) <= 20:
                            best_bid = min(filt_buy_orders.keys())
                            orders.append(Order(product, best_bid, 1))
                            filt_buy_orders
                        elif filt_sell_orders and abs(pos - 1) <= 20:

                            orders.append(Order(product, x, -1))
                        else:
                            break





            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData