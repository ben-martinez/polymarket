import json
import torch
import torch_geometric
# from torch_geometric import Data
from tqdm import tqdm
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas

def flatten_dataset(file_path, outfile_users, outfile_markets, outfile_positions):
    with open(file_path, 'r') as file:
        data = json.load(file)

    output_users = []
    output_markets = []
    output_positions = []
    # make it such that each entry is flat row instead of a nested dict
    marketPositions = data["data"]["marketPositions"]
    for i in tqdm(range(len(marketPositions))):
        pos = marketPositions[i]

        # may contain duplicate user
        output_users.append({
            "user_id": pos["user"]["id"],
            "position_id": pos["id"],
            "lastTradedTimestamp": pos["user"]["lastTradedTimestamp"],
            "user_creationTimestamp": pos["user"]["creationTimestamp"],
            "collateralVolume": pos["user"]["collateralVolume"],
            "numTrades": pos["user"]["numTrades"],
            "scaledProfit": pos["user"]["scaledProfit"],
            "market_id": pos["market"]["id"]
        })

        output_markets.append({
            "user_id": pos["user"]["id"],
            # "position_id": pos["id"],
            "market_id": pos["market"]["id"],
            "market_price": pos["market"]["priceOrderbook"]
        })

        output_positions.append({
            # "user_id": pos["user"]["id"],
            "position_id": pos["id"],
            # "market_id": pos["market"]["id"],
            "netValue": pos["netValue"],
            "feesPaid": pos["feesPaid"],
            "netQuantity": pos["netQuantity"]
        })

    with open(outfile_users, 'w') as f:
        json.dump(output_users, f)
    with open(outfile_markets, 'w') as f:
        json.dump(output_markets, f)
    with open(outfile_positions, 'w') as f:
        json.dump(output_positions, f)

def create_pyg_dataset(users_path, markets_path, positions_path):
    users_df = pandas.read_json(users_path)
    markets_df = pandas.read_json(markets_path)
    positions_df = pandas.read_json(positions_path)

    # map each node type id to a new id (0 to num_unique_<node_type>)
    unique_users = users_df["user_id"].unique()
    unique_user_ids = pandas.DataFrame(data={
        'user_id': unique_users,
        'user_index': pandas.RangeIndex(len(unique_users)),
    })
    df = users_df.join(unique_user_ids.set_index('user_id'), on='user_id', how='left')

    unique_markets = markets_df['market_id'].unique()
    # print("Markets", unique_markets)
    unique_market_ids = pandas.DataFrame(data={
        'market_id': unique_markets,
        'market_index': pandas.RangeIndex(len(unique_markets)),
    })
    df = df.join(unique_market_ids.set_index('market_id'), on='market_id', how='left')
    markets_df = markets_df.join(unique_market_ids.set_index('market_id'), on='market_id', how='left')

    unique_positions = positions_df['position_id'].unique()
    unique_position_ids = pandas.DataFrame(data={
        'position_id': unique_positions,
        'position_index': pandas.RangeIndex(len(unique_positions)),
    })
    df = df.join(unique_position_ids.set_index('position_id'), on='position_id', how='left')
    positions_df = positions_df.join(unique_position_ids.set_index('position_id'), on='position_id', how='left')

    # shape of [2, num_edges]
    edge_index_users_to_position = []
    edge_index_position_to_market = []

    for index, entry in df.iterrows():
        # print(entry)
        user_index = entry["user_index"]
        market_index = entry["market_index"]
        position_index = entry["position_index"]
        edge_index_users_to_position.append([user_index, position_index])
        edge_index_position_to_market.append([position_index, market_index])

    # print(edge_index_users_to_position)
    # print(edge_index_position_to_market)

    # remove the col(s) of user_id, market_id, etc. because they aren't 0-indexed and are confusing
    users = df[["user_index", "lastTradedTimestamp", "user_creationTimestamp", "collateralVolume", "numTrades", "scaledProfit"]]
    users = users.groupby("user_index").agg("mean")

    markets = markets_df[["market_index", "user_id", "market_price"]]
    # get the number of unique users to be the feature for markets
    markets = markets.groupby("market_index").agg({"user_id": lambda x: len(set(x)), "market_price": "mean"})

    positions = positions_df[["position_index", "netValue", "feesPaid", "netQuantity"]]
    positions = positions.groupby("position_index").agg("mean")
    #unique_markets = markets_df.groupby("market_id").agg("mean")
    #unique_positions = positions_df.groupby("position_id").agg("mean")

    # Reference: https://pytorch-geometric.readthedocs.io/en/2.6.0/notes/heterogeneous.html
    data = HeteroData()
    # shape [num unique users, num user features = 5]
    data['user'].x = torch.from_numpy(users.values)
    # shape [num_unique_markets, num market features = 2 (number of unique users betting in it currently, ave market price)]
    data['market'].x = torch.from_numpy(markets.values)
    # shape [num_unique positions, num position features = 3]
    data['position'].x = torch.from_numpy(positions.values)

    data['user', 'bets', 'position'].edge_index = torch.Tensor(edge_index_users_to_position).T
    data['position', 'active', 'market'].edge_index = torch.Tensor(edge_index_position_to_market).T

    print(data)
    """
        HeteroData(
      user={ x=[10, 5] },
      market={ x=[91, 2] },
      position={ x=[100, 3] },
      (user, bets, position)={ edge_index=[2, 100] },
      (position, active, market)={ edge_index=[2, 100] }
    )"""
    # Reference: https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
    # reference: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.NodePropertySplit.html
    # Split into training, validation, test nodes

    transform = T.RandomNodeSplit(
        num_val=0.2,
        num_test=0.2,
        # the ground truth label. default is y
        key=""
    )
    train_data, val_data, test_data = transform(data)

    return data, train_data, val_data, test_data

# Initial data pulled from the API
file_path = "query_1.json"

users_path = "users.json"
markets_path = "markets.json"
positions_path = "positions.json"

flatten_dataset(file_path, users_path, markets_path, positions_path)

pyg_data = create_pyg_dataset(users_path, markets_path, positions_path)#, "users.json", "markets.json", "positions.json")


