from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string




class Trader:
    def 
    def run(self, state: TradingState):
        # print("traderData: " + state.traderData)
        # print("Observations: " + str(state.observations))

        result = {}
        for product in state.order_depths:
          order_depth: OrderDepth = state.order_depths[product]
          orders: List[Order] = []
            
          if(product == "AMETHYSTS"):
            curr_am_pos = state.position.get("AMETHYSTS", 0)
            acceptable_price = 10000

            #checking to see if we can buy
            if len(order_depth.sell_orders) != 0 and curr_am_pos < 20:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, best_ask_amount))

             #checking to see if we can sell
            if len(order_depth.buy_orders) != 0 and curr_am_pos > -20:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
                    
          result[product] = orders
      
        for product, position in state.position.items():
          print(f"Position for {product}: {position}")
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData