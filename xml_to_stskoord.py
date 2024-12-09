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


inventory_dir = join('/', 'home', 'eq', 'seiscomp', 'etc', 'inventory', 'fsdn')

# sts_list = pd.DataFrame(columns=['Network_code', 'Station_code', 'Location_code',
#                                  'Channel_code'  'Station_lat', 'Station_lon'])
sts_list = pd.DataFrame(columns=['net', 'sta', 'loc', 'ch', 'lat', 'lon', 'wZ', 'wN', 'wE', 'starttime'])

sts_inv_list = [file for file in listdir(inventory_dir) if file.endswith(".xml")]
for inv in sts_inv_list:
    net, stn, ext = inv.split(".")
    # if stn not in sts_list['Station_code'].values:
    if stn not in sts_list['sta'].values:
        inv = read_inventory(join(inventory_dir, inv))
        list_channel = [cha.code for cha in inv[0][0]]
        net = inv[0].code
        sta = inv[0][0].code
        loc = inv[0][0][0].location_code
        for i in range(len(list_channel)-1, -1, -1):
            if list_channel[i][:2] == 'BH' or list_channel[i][:2] == 'SH':
                cha = inv[0][0][i].code[:2]
                break
        lat = inv[0][0].latitude
        lon = inv[0][0].longitude
        sts_start = inv[0][0][0].start_date.strftime('%Y-%m-%dT%H:%M:%S')
        new_sts_data = {
            'net': net,
            'sta': sta,
            'loc': loc,
            'ch': cha,
            'lat': lat,
            'lon': lon,
            'wZ': 1.0,
            'wN': 1.0,
            'wE': 1.0,
            'starttime': sts_start}
        sts_list = pd.concat([sts_list, pd.DataFrame([new_sts_data])], ignore_index=True)

# sts_list = sts_list.sort_values(by=['Network_code', 'Station_code'], ascending=[True, True])
sts_list = sts_list.sort_values(by=['net', 'sta'], ascending=[True, True])
sts_list = sts_list.reset_index(drop=True)

sts_list['loc'] = sts_list['loc'].astype(str)
sts_list[['wZ', 'wN', 'wE']] = sts_list[['wZ', 'wN', 'wE']].round(1)
sts_list.to_csv('stations.dat', index=False)


# out_sta = open('stations.dat', 'w')
# out_sta.write('sts     lat     lon\n')
# for i, r in sts_list.iterrows():
#     out_sta.write(f"{r['Network_code']}:{r['Station_code']}:{r['Location_code']}:{r['Channel_code'][:-1]}"
#                   f"   {r['Station_lat']}    {r['Station_lon']}\n")
# out_sta.close()
