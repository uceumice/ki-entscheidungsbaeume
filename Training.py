#TRAININGSPROZESS + PRUNNING

from PyTree import ClassificationTree as ct
import pandas as pd
import pickle

data = pd.read_csv("DATENSATZ.csv",sep=",")
parkplaetze = data.iloc[:,1:12].columns

parkplaetze_limits = {
        'P1AK': lambda x: x/640,
        'P2AK': lambda x: x/540,
        'P3AK': lambda x: x/540,
        'P41AK': lambda x: x/670,
        'P42AK': lambda x: x/670,
        'P5AK': lambda x: x/500,
        'P61AK': lambda x: x/290,
        'P62AK': lambda x: x/290,
        'P63AK': lambda x: x/290,
        'P7AK': lambda x: x/290,
        'P8AK': lambda x: x/160
    }

def quantity_to_procent(parkplatz):
    return parkplaetze_limits[parkplatz]


"""
DATA -> NUR WETTERDATEN
"""
def data_train_test_W(parkplatz_name: str):
    data = pd.read_csv("DATENSATZ.csv",sep=",")
    # Zielspalten aussuchen
    parkplaetze = data.iloc[:,1:12].columns
    ## INDEX WICHTIG
    # Extra Spalten aussuchen
    extra_spalten = [parkplatz_ for parkplatz_ in parkplaetze if parkplatz_!=parkplatz_name]
    # Extra-Spalten löschen (inklusive TimeStamp)
    data = data.drop(columns=extra_spalten+["TimeStamp"])
    data[parkplatz_name] = data[parkplatz_name].apply(quantity_to_procent(parkplatz_name)).round(decimals=1)
    
    # Erstelle eine Zielvariable -> (vorhersage für 1 Stunde)
    temp = data[parkplatz_name].tolist()[1:]
    temp.append(float("nan"))
    data["ZielVariable"] =  temp
    data = data.drop(columns=[parkplatz_name])
    
    # Trenne Trainingdaten und Testdaten
    data = data.sample(frac=1).reset_index(drop=True)
    data_train, data_test = data[:23016], data[23016:]
    return data_train, data_test

"""
DATA -> NUR WETTERDATEN + PARKPLATZDATEN ZUM PARKPLATZ DEREN BELEGUNG VORHERGESAGT WIRD
"""
def data_train_test_WPB1(parkplatz_name: str):
    data = pd.read_csv("DATENSATZ.csv",sep=",")
    # Zielspalten aussuchen
    parkplaetze = data.iloc[:,1:12].columns
    ## INDEX WICHTIG
    # Extra Spalten aussuchen
    extra_spalten = [parkplatz_ for parkplatz_ in parkplaetze if parkplatz_!=parkplatz_name]
    # Extra-Spalten löschen (inklusive TimeStamp)
    data = data.drop(columns=extra_spalten+["TimeStamp"])
    data[parkplatz_name] = data[parkplatz_name].apply(quantity_to_procent(parkplatz_name)).round(decimals=1)
    
    # Erstelle eine Zielvariable -> (vorhersage für 1 Stunde)
    temp = data[parkplatz_name].tolist()[1:]
    temp.append(float("nan"))
    data["ZielVariable"] =  temp

    # Trenne Trainingdaten und Testdaten
    data = data.sample(frac=1).reset_index(drop=True)
    data_train, data_test = data[:23016], data[23016:]
    return data_train, data_test

"""
DATA -> NUR WETTERDATEN + PARKPLATZDATEN ZU ALLEN PARKPLÄTZEN
"""
def data_train_test_WPBALL(parkplatz_name: str):
    data = pd.read_csv("DATENSATZ.csv",sep=",")
    # Zielspalten aussuchen
    parkplaetze = data.iloc[:,1:12].columns
    ## INDEX WICHTIG
    # Extra Spalten aussuchen
    # Extra-Spalten löschen (inklusive TimeStamp)
    data = data.drop(columns=["TimeStamp"])

    # inklusive alle Parkplätze
    for parkplatz_n in parkplaetze:
        data[parkplatz_n] = data[parkplatz_n].apply(quantity_to_procent(parkplatz_n)).round(decimals=1)
    
    # Erstelle eine Zielvariable -> (vorhersage für 1 Stunde)
    temp = data[parkplatz_name].tolist()[1:]
    temp.append(float("nan"))
    data["ZielVariable"] =  temp

    # Trenne Trainingdaten und Testdaten
    data = data.sample(frac=1).reset_index(drop=True)
    data_train, data_test = data[:23016], data[23016:]
    return data_train, data_test


def make_tree(data_train: str) -> tuple:
    # erstelle Baum
    tree: ct.DecisionTree = ct.DecisionTree()
    tree.grow_tree(data_train, "ZielVariable", min_leaf_cases=10)
    return tree

def save_to_pkl(filename: str, tree: ct.DecisionTree):
    if filename[-4:] != ".pkl":
        filename+=".pkl"
    
    with open(filename, 'wb') as outp:
        # fordert Python > 3.8
        pickle.dump(tree, outp, pickle.HIGHEST_PROTOCOL)

def load_from_pkl(filename: str):
    if filename[-4:] != ".pkl":
        filename+=".pkl"
        
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def save_to_tree(filename: str, tree: ct.DecisionTree):
    tree.export(filename=parkplaetze[0])

def load_from_tree(filename: str):
    return ct.import_tree(filename=filename)

def validation(tree: ct.DecisionTree, test_data: pd.DataFrame):
    return tree.validation_pruning(test_data)

def main(parkplatz_index, mode=0, limit_training_data=None):
    """
    parkplatz_index:    0 - 11

    mode:   0  = nur WetterDaten
            1 = wetterdaten + parkplatzdaten zum parkplatz der vorhergesagt wird
            2 = wetterdaten + alle parkplatzdaten
            3 = selbstdefinierte Spalten

    return tree, data_train, data_test
    """

    if 11 < parkplatz_index < 0:
        raise ValueError("PArkplatz existiert nicht")

    if mode==0:
        data_train, data_test = data_train_test_W(parkplaetze[parkplatz_index])
    elif mode==1:
        data_train, data_test = data_train_test_WPB1(parkplaetze[parkplatz_index])
    elif mode==2:
        data_train, data_test = data_train_test_WPBALL(parkplaetze[parkplatz_index])
    elif mode==3:
        raise NotImplementedError()


    print("DATASETS CREATED")


    if limit_training_data:
        limit = int(len(data_train)*limit_training_data)
        tree = make_tree(data_train[:limit])
    else:
        tree = make_tree(data_train)
    print("TREE CREATED")


    return tree, data_train, data_test

class Training():
    def __init__(self,parkplatz_index, mode=0, limit_training_data=None) -> None:
        t, trs,tes = main(parkplatz_index,
                          mode,
                          limit_training_data,
                          )
        self.tree: ct.DecisionTree = t
        self.data_train = trs
        self.data_test = tes
        self.parkplatz_index = parkplatz_index
        
    def validation_pruning(self, validation_sample, root_node=None, limit_pruning_data=None):
        if limit_pruning_data:
            limit = int(len(self.data_test)*limit_pruning_data)        
        self.tree.validation_pruning(validation_sample[:limit], root_node)
    
    def save_to_pkl(self, location_pkl=""):
        save_to_pkl(filename=location_pkl+parkplaetze[self.parkplatz_index], tree = self.tree)

    def save_to_tree(self, location_tree=""):
        save_to_tree(filename=location_tree+parkplaetze[self.parkplatz_index], tree = self.tree)

    def save_train_set(self, location_trainset=""):
        self.data_train.to_csv(location_trainset+parkplaetze[self.parkplatz_index])

    def save_test_set(self, location_testset=""):
        self.data_test.to_csv(location_testset+parkplaetze[self.parkplatz_index])
