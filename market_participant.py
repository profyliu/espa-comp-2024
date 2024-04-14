# This is a test dummy algorithm to get the opportunity cost curves
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
import argparse
import json
import datetime
from itertools import accumulate
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



batt_attr = {'cost_rgu': 3, 'cost_rgd': 3, 'cost_spr': 0, 'cost_nsp': 0, 'init_en': 0, 'init_status': 1, 'ramp_dn': 9999, 'ramp_up': 9999, 'block_ch_mc': '0', 'block_dc_mc': '0', 'block_soc_mc': '0', 'chmax': 125.0, 'dcmax': 125.0, 'block_ch_mq': '125.0', 'block_dc_mq': '125.0', 'block_soc_mq': '608.0', 'soc_end': 128.0, 'soc_begin': 480.0, 'socmax': 608.0, 'socmin': 128.0, 'eff_ch': 0.8919999999999999, 'eff_dc': 1.0, 'imax': 1000, 'imin': -1000, 'vmax': 820, 'vmin': 680, 'ch_cap': 3413.3333333333335, 'eff_coul': 0.946, 'eff_inv0': 0.0, 'eff_inv1': 0.99531, 'eff_inv2': -0.00027348, 'voc0': 669.282, 'voc1': 201.004, 'voc2': -368.742, 'voc3': 320.377, 'resis': 0.000365333, 'therm_cap': 36000, 'temp_max': 60, 'temp_min': -20, 'temp_ref': 20, 'Utherm': 4.800000000000001, 'deg_DoD0': 0.0616, 'deg_DoD1': 0.537, 'deg_DoD2': 3.3209, 'deg_DoD3': -6.8292, 'deg_DoD4': 5.7905, 'deg_soc': 1.04, 'deg_therm': 0.0693, 'deg_time': 5.708e-06, 'cycle_life': 10950, 'cost_EoL': -187200000.0, 'socref': 320.0, 'soc_capacity': 640, 'cell_count': 250}

#calculate the opportunity cost for charge/discharge in the DA market
def da_offers(prices, cur_soc, required_times):
    # battery parameters
    global batt_attr
    socmax = batt_attr['socmax']  # 608 MWh
    socmin = batt_attr['socmin']  # 128 MWh
    effcy=batt_attr['eff_ch']
    chmax = batt_attr['chmax']  # 125 MW
    dcmax = batt_attr['dcmax']  # 125 MW
    beta1 = 77.3 / 24 # dollars per MWh in SOC per hour
    target_midday_soc = 500  # three-hours buffer for sell in extreme high lmp events
    
    
    block_ch_mc = {required_times[i]:0 for i in range(len(required_times))}
    block_ch_mq = {required_times[i]:0 for i in range(len(required_times))}
    block_dc_mc = {required_times[i]:0 for i in range(len(required_times))}
    block_dc_mq = {required_times[i]:0 for i in range(len(required_times))}

    for i in [9,10,11]:
        block_ch_mc[required_times[i]] = 5
        block_ch_mq[required_times[i]] = 125
    for i in [19, 20, 21]:
        block_dc_mc[required_times[i]] = 15
        block_dc_mq[required_times[i]] = 125
    return block_ch_mc, block_ch_mq, block_dc_mc, block_dc_mq

def rt_offers(this_hour_da_price, this_hour_da_dispatch, last_interval_rt_price, offer_timestamp):
    global batt_attr
    socmax = batt_attr['socmax']  # 608 MWh
    socmin = batt_attr['socmin']  # 128 MWh
    eff_ch = batt_attr['eff_ch']  # 0.8919
    eff_dc = batt_attr['eff_dc']  # 1.0
    chmax = batt_attr['chmax']  # 125 MW
    dcmax = batt_attr['dcmax']  # 125 MW    
    n_blocks = 10
    block_ch_mc = {offer_timestamp: 0}
    block_ch_mq = {offer_timestamp: 0}
    block_dc_mc = {offer_timestamp: 0}
    block_dc_mq = {offer_timestamp: 0}
    
    if this_hour_da_dispatch > 0:  # discharge
        price_points = np.linspace(start = this_hour_da_price/(n_blocks/2), stop = 2*this_hour_da_price, num = n_blocks)
        quantity_points = [this_hour_da_dispatch/(n_blocks/2) for i in range(n_blocks//2)] + [(dcmax-this_hour_da_dispatch)/(n_blocks/2) for i in range(n_blocks//2)]
        block_dc_mc = {offer_timestamp: list(price_points)}
        block_dc_mq = {offer_timestamp: quantity_points}
    elif this_hour_da_dispatch < 0:  # charge
        price_points = np.linspace(start = (2 - 1/(n_blocks/2))*this_hour_da_price, stop = 0, num = n_blocks)
        quantity_points = [-this_hour_da_dispatch/(n_blocks/2) for i in range(n_blocks//2)] + [(chmax + this_hour_da_dispatch)/(n_blocks/2) for i in range(n_blocks//2)]
        block_ch_mc = {offer_timestamp:list(price_points)}
        block_ch_mq = {offer_timestamp:quantity_points}
    else:
        if last_interval_rt_price < 3*this_hour_da_price:
            block_dc_mc = {offer_timestamp: [5*last_interval_rt_price]}
            block_dc_mq = {offer_timestamp: [dcmax]}
        else:
            block_dc_mc = {offer_timestamp: [4*last_interval_rt_price/5]}
            block_dc_mq = {offer_timestamp: [dcmax]}
            block_ch_mc = {offer_timestamp: [this_hour_da_price]}
            block_ch_mq = {offer_timestamp: [chmax]}
    return block_ch_mc, block_ch_mq, block_dc_mc, block_dc_mq

if __name__ == '__main__':
    # Add argument parser for three required input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('time_step', type=int, help='Integer time step tracking the progress of the\
                        simulated market.')
    parser.add_argument('market_file', help='json formatted dictionary with market information.')
    parser.add_argument('resource_file', help='json formatted dictionary with resource information.')

    args = parser.parse_args()

    # Parse json inputs into python dictionaries
    time_step = args.time_step
    args = parser.parse_args()
    with open(args.market_file, 'r') as f:
        market_info = json.load(f)
    with open(args.resource_file, 'r') as f:
        resource_info = json.load(f)
    
    #rid = 'R00229'  # where does this designation come from? Need to verify. 
    rid = resource_info['rid']
    bus_id = resource_info['bus']
    # Read in information from the market
    uid =market_info["uid"]
    market_type = market_info["market_type"]
    if 'DAM' in market_type:
        prices = market_info["previous"]["TSDAM"]["prices"]["EN"][bus_id]
        required_times = [t for t in market_info['timestamps']]
        # price_dict = {required_times[i]:prices[i] for i in range(len(required_times))}
        # Writing prices to a local JSON file
        # file_path = "DAM_da_prices_" + str(time_step) + ".json"
        # with open(file_path, "w") as file:
        #     json.dump(price_dict, file)
        prices = np.array(prices)  # this is the da price of the previous da settlement mapped to the required_times of the upcoming da settlement
        cur_soc = resource_info['status'][rid]['soc']
        # Make the offer curves and unload into arrays
        block_ch_mc, block_ch_mq, block_dc_mc, block_dc_mq = da_offers(prices, cur_soc, required_times)
        block_soc_mc = {}
        block_soc_mq = {}
        for i in range(len(required_times)):
            block_soc_mc[required_times[i]] = 0
            block_soc_mq[required_times[i]] = 0

        reg = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp']
        zero_arr = np.zeros(len(required_times))
        rgu_dict = {}
        for r in reg:
            rgu_dict[r] = {}
        for t in required_times:
            rgu_dict['cost_rgu'][t] = 100
        for t in required_times:
            rgu_dict['cost_rgd'][t] = 100       
        for t in required_times:
            rgu_dict['cost_spr'][t] = 200
        for t in required_times:
            rgu_dict['cost_nsp'][t] = 150

        max_dict = {}
        for mx in ['chmax', 'dcmax']:
            max_dict[mx] = {}
            for t in required_times:
                max_dict[mx][t] = 125

        constants = {}
        constants['soc_begin'] = cur_soc
        constants['init_en'] = 0
        constants['init_status'] = 0
        constants['ramp_dn'] = 9999
        constants['ramp_up'] = 9999
        constants['socmax'] = 608
        constants['socmin'] = 128
        constants['eff_ch'] = 0.892
        constants['eff_dc'] = 1.0
        constants['soc_end'] = 128
        constants['bid_soc'] = False

        # Pacakge the dictionaries into an output formatted dictionary
        
        offer_out_dict = {rid:{}}
        offer_out_dict[rid] = {"block_ch_mc":block_ch_mc, "block_ch_mq":block_ch_mq, "block_dc_mc":block_dc_mc, "block_dc_mq":block_dc_mq, "block_soc_mc":block_soc_mc, "block_soc_mq":block_soc_mq}
        offer_out_dict[rid].update(rgu_dict)
        offer_out_dict[rid].update(max_dict)
        offer_out_dict[rid].update(constants)

        # Save as json file in the current directory with name offer_{time_step}.json
        with open(f'offer_{time_step}.json', 'w') as f:
            json.dump(offer_out_dict, f, indent=4, cls=NpEncoder)
    elif 'RTM' in market_type:
        # price_path = "da_prices.json"
        # with open(price_path, "r") as file:
        #     da_prices = json.load(file)
        #     dam_times = [key for key in da_prices.keys()]
        #     prices = [value for value in da_prices.values()
        da_timestamps = market_info['previous']['TSDAM']['times']
        da_prices = market_info["previous"]["TSDAM"]["prices"]["EN"][bus_id]
        # price_dict = {da_timestamps[i]:da_prices[i] for i in range(len(da_timestamps))}
        # date_value = da_timestamps[0][0:-4]
        # Writing prices to a local JSON file
        # file_path = "RTM_da_prices_" + date_value + ".json"
        # if not os.path.exists(file_path):
        #     with open(file_path, "w") as file:
        #         json.dump(price_dict, file)
        #print("da_prices: ")
        #print(da_prices)
        # Read in information from the resource
        en_schedule_list = [z[0] for z in resource_info["ledger"][rid]["EN"].values()]
        sch_time = [z for z in resource_info["ledger"][rid]["EN"].keys()]
        # print(en_schedule_list)
        # print(sch_time)


        offer_timestamp = market_info['timestamps'][0]  # the timestamp for which an RT offer is required 
        ledger_EN = resource_info['ledger'][rid]['EN']
        this_hour_timestamp = offer_timestamp[:-2] + "00"
        # print(f"offer_ts: ", offer_timestamp, " this hour ts: ", this_hour_timestamp)
        this_hour_da_ledger = ledger_EN.get(this_hour_timestamp, [])
        # print(f"this hour da dispatch: {this_hour_da_ledger}")
        if len(this_hour_da_ledger) == 0:
            this_hour_da_dispatch = 0
        else:
            max_price = 0  # assuming when multiple entries exist, the one with the largest price is the da dispatch
            max_price_index = -1000
            for i in range(len(this_hour_da_ledger)):
                this_entry = this_hour_da_ledger[i]
                if this_entry[1] > max_price:
                    max_price = this_entry[1]
                    max_price_index = i
            this_hour_da_dispatch = this_hour_da_ledger[i][0]
        this_hour_da_price = da_prices[da_timestamps.index(this_hour_timestamp)]  # assuming for now that this is the actual da lmp. pending verification
        # print(f"this_hour_da_price = {this_hour_da_price}")
        TSRTM_times = market_info['previous']['TSRTM']['times']
        k = TSRTM_times.index(offer_timestamp)
        last_interval_rt_price = market_info['previous']['TSRTM']['prices']['EN'][bus_id][k]*12  # multiply by 12 because of the way the TSRTM value is provided
        if last_interval_rt_price == 0:
            last_interval_rt_price = this_hour_da_price  # if cannot find rt price, use this hour da price instead
        print(f"last_interval_rt_price = {last_interval_rt_price}")
        # initialize 
        block_ch_mc = {}
        block_ch_mq = {}
        block_dc_mc = {}
        block_dc_mq = {}
        block_soc_mc = {}
        block_soc_mq = {}
        required_times = [t for t in market_info['timestamps']]

        block_ch_mc_1, block_ch_mq_1, block_dc_mc_1, block_dc_mq_1 = rt_offers(this_hour_da_price, 
                                                                       this_hour_da_dispatch, 
                                                                       last_interval_rt_price, 
                                                                       offer_timestamp)        
        for i in range(len(required_times)):
            block_ch_mc[required_times[i]] = block_ch_mc_1[offer_timestamp]
            block_ch_mq[required_times[i]] = block_ch_mq_1[offer_timestamp]
            block_dc_mc[required_times[i]] = block_dc_mc_1[offer_timestamp]
            block_dc_mq[required_times[i]] = block_dc_mq_1[offer_timestamp]
            block_soc_mc[required_times[i]] = 0.0
            block_soc_mq[required_times[i]] = 0.0

        reg = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp']
        zero_arr = np.zeros(len(required_times))
        rgu_dict = {}
        for r in reg:
            rgu_dict[r] = {}
            for t in required_times:
                rgu_dict[r][t] = 0

        max_dict = {}
        for mx in ['chmax', 'dcmax']:
            max_dict[mx] = {}
            for t in required_times:
                max_dict[mx][t] = 125

        constants = {}
        constants['soc_begin'] = 128
        constants['init_en'] = 0
        constants['init_status'] = 0
        constants['ramp_dn'] = 9999
        constants['ramp_up'] = 9999
        constants['socmax'] = 608
        constants['socmin'] = 128
        constants['eff_ch'] = 0.892
        constants['eff_dc'] = 1.0
        constants['soc_end'] = 128
        constants['bid_soc'] = True

        # Pacakge the dictionaries into an output formatted dictionary
        offer_out_dict = {rid:{}}
        offer_out_dict[rid] = {"block_ch_mc":block_ch_mc, "block_ch_mq":block_ch_mq, "block_dc_mc":block_dc_mc, "block_dc_mq":block_dc_mq, "block_soc_mc":block_soc_mc, "block_soc_mq":block_soc_mq}
        offer_out_dict[rid].update(rgu_dict)
        offer_out_dict[rid].update(max_dict)
        offer_out_dict[rid].update(constants)
        # Save as json file in the current directory with name offer_{time_step}.json
        with open(f'offer_{time_step}.json', 'w') as f:
            json.dump(offer_out_dict, f, indent=4, cls=NpEncoder)
