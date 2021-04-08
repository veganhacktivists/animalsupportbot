import os
from collections import OrderedDict

import gspread
import numpy as np
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


def get_myth_df(spread):
    wks = spread.sheet1
    data = wks.get_all_values()
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    return df


def get_myth_egs_df(spread):
    wks = spread.get_worksheet(1)
    egs_data = wks.get_all_values()
    args = egs_data.pop(0)
    egs_df = pd.DataFrame(egs_data, columns=args)
    return egs_df.astype(str)


def get_eg_row(idx, arg_dict):
    # Populate a dataframe row from the arg dict
    row = []
    for key in arg_dict:
        if idx >= len(arg_dict[key]):
            row.append('')
        else:
            row.append(arg_dict[key][idx])
    return row


def make_newdf_from_dict(arg_dict, columns):
    egs_df = pd.DataFrame(columns=columns)
    max_len = max([len(arg_dict[k]) for k in arg_dict])
    for i in range(max_len + 10):
        egs_df.loc[i] = get_eg_row(i, arg_dict)
    return egs_df.fillna('')


def read_eg_df(file):
    df = pd.read_csv(file)
    return df

def arg_dict_from_df(egs_df):
    args = egs_df.columns
    arg_dict = OrderedDict({})
    for arg in args:
        arg_dict[arg] = egs_df[arg].values[egs_df[arg].str.len() > 0].tolist()
    return arg_dict


if __name__ == "__main__":
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    keyfile = './gdrive-keyfile.json'
    assert os.path.isfile(keyfile), "Need {}".format(keyfile)

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        keyfile, scope)
    gc = gspread.authorize(credentials)

    spread = gc.open_by_key("1epSnuOAhkv97UDs3NPAUtYMBNUzRo783GizPrADY4T8")

    print('Getting Myth and Example spreadsheets from Gdrive')
    myth_df = get_myth_df(spread)
    egs_df = get_myth_egs_df(spread)

    print('Checking that the data matches up okay')
    assert len(set(myth_df['Title'].values)) == len(set(egs_df.columns))
    assert set(myth_df['Title'].values) == set(egs_df.columns)

    myth_df.to_csv('./knowledge/myths.csv', index=False)
    egs_df.to_csv('./knowledge/myths_egs.csv', index=False)
    print('Wrote to ./knowledge/')
