import pandas as pd
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
from obspy import read_inventory


def get_element(root, elem_name):
    # root = parse_inventory(inv_path)

    for elem in root[-1][-1][-1]:
        if elem_name in elem.tag:
            return elem.text


inventory_dir = join('/', 'home', 'eq', 'PycharmProjects', 'Q_RMT', 'Analysis', 'tanimbar2023', 'data', 'inventory')

sts_list = pd.DataFrame(columns=['Network_code', 'Station_code', 'Location_code',
                                 'Channel_code'  'Station_lat', 'Station_lon'])

sts_inv_list = [file for file in listdir(inventory_dir) if file.endswith(".xml")]
for inv in sts_inv_list:
    net, stn, ext = inv.split(".")
    if stn not in sts_list['Station_code'].values:
        inv = read_inventory(join(inventory_dir, inv))
        list_channel = [cha.code for cha in inv[0][0]]
        net = inv[0].code
        sta = inv[0][0].code
        loc = inv[0][0][0].location_code
        for i in range(len(list_channel)-1, -1, -1):
            if list_channel[i][:2] == 'BH' or list_channel[i][:2] == 'SH':
                cha = inv[0][0][i].code
                break
        lat = inv[0][0].latitude
        lon = inv[0][0].longitude
        new_sts_data = {
            'Network_code': net,
            'Station_code': sta,
            'Location_code': loc,
            'Channel_code': cha,
            'Station_lat': lat,
            'Station_lon': lon}
        sts_list = pd.concat([sts_list, pd.DataFrame([new_sts_data])], ignore_index=True)
        sts_list = sts_list.sort_values(by=['Network_code', 'Station_code'], ascending=[True, True])
        sts_list = sts_list.reset_index(drop=True)

out_sta = open('stations.dat', 'w')
out_sta.write('sts     lat     lon\n')
for i, r in sts_list.iterrows():
    out_sta.write(f"{r['Network_code']}:{r['Station_code']}:{r['Location_code']}:{r['Channel_code'][:-1]}"
                  f"   {r['Station_lat']}    {r['Station_lon']}\n")
out_sta.close()
