import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import re

def get_kraken_results():
    result_path = f"output"
    columns = ['percentage', 'count', 'coverage', 'taxon', 'taxonomy_id', 'name']
    df = pd.read_table(f'{result_path}/report.txt',header=None,names=columns)
    columns_=['status','seq_id','taxonomy_id','length','mapping']
    results_df = pd.read_csv(f'{result_path}/output.txt',delimiter='\t',header=None, names=columns_)
    merged_df = pd.merge(results_df[['seq_id','status','taxonomy_id']],df[['taxonomy_id','taxon','name']],on='taxonomy_id',how='left')
    merged_df.to_csv(f'{result_path}/kraken2.csv',index=False)
    return

def get_kraken_taxonomic():
    filepath = f"output"
    kraken_df = pd.read_csv(f'{filepath}/kraken2.csv')
    
    columns = ['percentage', 'count', 'coverage', 'taxon', 'taxonomy_id', 'name']
    df = pd.read_table(f'{filepath}/report.txt',header=None,names=columns)
    
    #Get Species Map
    species_map = { }
    idx = 0
    while idx < df.shape[0]:
      taxon = df.iloc[idx,-3]
      name = df.iloc[idx,-1].strip()
      if taxon == 'S':
        species = name
        species_map[name] = species
      if bool(re.match(r'^S\d', taxon)):
        # print(name)
        species_map[name] = species
      idx+=1
    
    #Get Genus Map
    genus_map = { }
    idx = 0
    while idx < df.shape[0]:
      taxon = df.iloc[idx,-3]
      name = df.iloc[idx,-1].strip()
      if taxon == 'G':
        genus = name
        genus_map[name] = genus
      if bool(re.match(r'^G\d', taxon)) or taxon.startswith('S'):
        # print(name)
        genus_map[name] = genus
      idx+=1
      
    kraken_df['name'] = kraken_df['name'].apply(lambda x:x.strip())
    kraken_df['species'] = kraken_df['name'].map(species_map).fillna('unknown')
    kraken_df['genus'] = kraken_df['name'].map(genus_map).fillna('unknown')
    
    train_idx = kraken_df[kraken_df['species'] != 'unknown'].index
    test_idx = kraken_df[kraken_df['species'] == 'unknown'].index
    
    labels = kraken_df['species'].to_numpy()
    
    kraken_df.to_csv(f'{filepath}/kraken_final.csv',index=None)
    np.save(f'{filepath}/train_idx.npy',train_idx)
    np.save(f'{filepath}/test_idx.npy',test_idx)
    np.save(f'{filepath}/labels.npy',labels)
    
    return

def get_weights():
    path = f'output'
    columns_=['status','seq_id','taxonomy_id','length','mapping']
    results_df = pd.read_csv(f'{path}/output.txt',delimiter='\t',header=None, names=columns_)
    train_idx = np.load(f'{path}/train_idx.npy')
    
    results_df = results_df.iloc[train_idx,:]
    
    #weight calculation
    mapping_list = []

    for _, row in results_df.iterrows():
        seq_id = row['seq_id']
        taxonomic_id = row['taxonomy_id']
        mapping = row['mapping']
        mapping_dict = {}

        # Splitting each entry on ':'
        mappings = mapping.split(' ')
        if mappings[-1] == '':
          mappings=mappings.pop()

        for i in range(0, len(mappings)):
            left_val, right_val = mappings[i].split(':')
            # Adding the count to the corresponding key in the dictionary
            mapping_dict[left_val] = mapping_dict.get(left_val, 0) + int(right_val)

        list_row = [seq_id, taxonomic_id, mapping_dict]

        # Calculate the weight
        total_mapping_sum = sum(mapping_dict.values())
        weight = mapping_dict.get(str(taxonomic_id), 0) / total_mapping_sum if total_mapping_sum > 0 else 0

        list_row.append(weight)
        mapping_list.append(list_row)
        
    weights_df = pd.DataFrame(mapping_list, columns=['seq_id', 'taxonomic_id', 'mapping', 'weight'])
    weights = weights_df['weight'].values.reshape(-1, 1)

    # Initialize the StandardScaler
    scaler = MinMaxScaler()

    # Fit and transform the weights
    weights_df['weight_standardized'] = scaler.fit_transform(weights)
    sample_weights = weights_df['weight_standardized'].to_numpy()
    
    np.save(f'{path}/sample_weights.npy',sample_weights)
    
    return