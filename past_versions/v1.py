from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):

		# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # if product == "STARFRUIT":

            #     #the basic idea here is to just sell and hold since we see the general downtrend
            #     #NOTE: THIS IS JUST TO SEE what might be the upper limit, this is extremely stupid in practice

            #     pos = state.position.get("STARFRUIT", 0)
            #     print("STARFRUIT Position: ", pos) 
            #     best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            #     if(abs(pos) < 20):
            #         orders.append(Order(product, best_bid, -best_bid_amount))
            #     elif(pos == 20):
            #         print("Done")

            if product == "AMETHYSTS":

                #the basic idea here is the market make around the static paremeter of 10,000
                #this strategy ends up being not that useful.

                pos = state.position.get("AMETHYSTS", 0)
                curr_buys = 0
                curr_sells = 0
                print("AMETHYSTS Position: ", pos) #
                filt_buy_orders = {key: value for key, value in order_depth.buy_orders.items() if key > 10000}
                #print(filt_buy_orders)
                filt_sell_orders = {key: value for key, value in order_depth.sell_orders.items() if key < 10000}
                # print(filt_sell_orders)
                
                while filt_buy_orders or filt_sell_orders:
                    if filt_buy_orders and abs(pos - curr_sells - 1) <= 20:
                        best_bid = max(filt_buy_orders.keys())
                        orders.append(Order(product, best_bid, -1))
                        filt_buy_orders[best_bid]-=1
                        # print("SELL @ ", str(best_bid)) # we sell when there is buy orders
                        filt_buy_orders = {key: value for key, value in filt_buy_orders.items() if value > 0}
                        curr_sells+=1
                    elif filt_sell_orders and abs(pos + curr_buys + 1) <= 20:
                        best_ask = min(filt_sell_orders.keys())
                        filt_sell_orders[best_ask]-=1
                        orders.append(Order(product, best_ask, 1))
                        # print("BUY @ ", str(best_ask)) # we buy when there is sell orders
                        filt_sell_orders = {key: value for key, value in filt_sell_orders.items() if value > 0}
                        curr_buys+=1
                    else:
                        break
                # print("Orders: ", orders) #


            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData#