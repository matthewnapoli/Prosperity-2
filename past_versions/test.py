        bid_orders_to_submit: List[Order] = []
        bid_orders_to_submit_dict : Dict[int,int] = {}
        ask_orders_to_submit: List[Order] = []
        ask_orders_to_submit_dict : Dict[int,int] = {}

        order_book              = state.order_depths["ORCHIDS"]
        local_bid_prices       = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
        local_ask_prices        = collections.OrderedDict(sorted(order_book.sell_orders.items()))
        bid_volume_avail        = pos_limit - pos
        ask_volume_avail       = abs(-pos_limit - pos)

        south_island_info       = state.observations.conversionObservations["ORCHIDS"]
        adj_south_ask_price     = south_island_info.askPrice + south_island_info.transportFees + south_island_info.importTariff
        adj_south_bid_price    = south_island_info.bidPrice - south_island_info.transportFees - south_island_info.exportTariff
        bvolume,_,bvwap         = self.calc_metrics_bids(bid_orders_to_submit_dict)
        avolume,_,avwap         = self.calc_metrics_asks(ask_orders_to_submit_dict)

        logger.print("sBuy @", adj_south_ask_price)
        logger.print("sSell @", adj_south_bid_price)
        bLocal_sSouth_pnl = 0
        sLocal_bSouth_pnl = 0

        #here we buy local (at the ask), and will sell at the `adj_south_ask_price`
        for local_bid_price, bid_volume in local_ask_prices.items():
            if (local_bid_price < adj_south_ask_price) and bid_volume_avail > 0:
                logger.print("good1: buy,sell: ", local_bid_price, adj_south_ask_price)
                max_vol_tradable = min(abs(bid_volume), bid_volume_avail)
                bid_orders_to_submit.append(Order("ORCHIDS", int(local_bid_price), int(max_vol_tradable)))


                bid_volume_avail -= max_vol_tradable
                
        #here we sell local (at the bid), and will buy at the `adj_south_bid_price`
        for local_ask_price, ask_volume in local_bid_prices.items():
            if (adj_south_bid_price < local_ask_price) and ask_volume_avail > 0:
                logger.print("good2 sell,buy: ", local_ask_price, adj_south_bid_price)
                max_vol_tradable = min(ask_volume, ask_volume_avail)
                ask_orders_to_submit.append(Order("ORCHIDS", int(local_ask_price), int(-max_vol_tradable)))

                local_bid_prices[local_ask_price]
                ask_volume_avail -= max_vol_tradable

        if bid_volume_avail > 0 or ask_volume_avail > 0:

            logger.print("MM-ing")
            _,lowest_local_bid_price,_ = self.calc_metrics_asks(local_ask_prices)
            _,highest_local_ask_price,_ = self.calc_metrics_bids(local_bid_prices)
            #if there is no outstsanding arb ops: MM on the side (long/short)that is the closest to a possible arb and trying to snag market orders
            logger.print(adj_south_ask_price-lowest_local_bid_price,"?",highest_local_ask_price-adj_south_bid_price)
            if adj_south_ask_price-lowest_local_bid_price < highest_local_ask_price-adj_south_bid_price: 
                #sLocal, bSouth (right side of ineq), gives us a less negative thing, so do it
                return [Order("ORCHIDS", int(adj_south_ask_price+1), -1*ask_volume_avail)]
            else:
                #bLocal, sSouth (left side of ineq), gives us a less negative thing, so do it
                return [Order("ORCHIDS", int(adj_south_bid_price-1), bid_volume_avail)]