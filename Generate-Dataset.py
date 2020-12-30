import numpy as np
from os import sys, path
import os

from sklearn.preprocessing import scale, robust_scale, normalize, MinMaxScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error, r2_score
from RegscorePy import *
import csv


def addRelativePathToSystemPath(relPath):
    if __name__ == '__main__' and __package__ is None:
        sys.path.append(path.join(path.dirname(path.abspath(__file__)), relPath))

addRelativePathToSystemPath("../GrooveToolbox/")
addRelativePathToSystemPath("../shared/")

from Groove import *
from LoadGrooveFromBFDPalette import *

path = '/home/fred/BFD/python/grooves/Small_AL_Tester/'
all_features = []
all_names = []
all_palette_names = []

def generate_dataset_unweighted_test():
    for i in os.listdir(path):
        hits, timings, names = get_all_grooves_from_BFD_palette(path+str(i))
        pal = []
        for j in range(len(names)):
             groove = NewGroove(hits[j], timings[j], names[j], velocity_type="None", extract_features=False)
             reduction = groove.reduce_groove().flatten()
             lowD = groove.RhythmFeatures.get_low_density()
             midD = groove.RhythmFeatures.get_mid_density()
             highD = groove.RhythmFeatures.get_high_density()
             totalD = groove.RhythmFeatures.get_total_density()
             combinedSync = groove.RhythmFeatures.get_combined_syncopation()
             polySync = groove.RhythmFeatures.get_polyphonic_syncopation()
             symmetry = groove.RhythmFeatures.get_total_symmetry()
             laidbackness = groove.MicrotimingFeatures.get_laidbackness()
             swingness = groove.MicrotimingFeatures.get_swingness()
             averageIntensity = groove.RhythmFeatures.get_total_average_intensity()

             grooveFeature = np.hstack([reduction, lowD, midD, highD, totalD, combinedSync, polySync, symmetry, laidbackness,
                                       swingness, averageIntensity])
             #print(names[j])
             all_features.append(grooveFeature)
             all_names.append(names[j])
             all_palette_names.append(i)
    return all_features, all_names, all_palette_names

def generate_dataset_format_for_PLS():
    all_features = []
    all_names = []
    all_palette_names = []
    for i in os.listdir(path):
        hits, timings, names = get_all_grooves_from_BFD_palette(path+str(i))
        pal = []
        for j in range(len(names)):
            groove = NewGroove(hits[j], timings[j], names[j], velocity_type="None", extract_features=True)
            rhythm_timing_features = np.hstack([groove.RhythmFeatures.get_all_features(),
                                      groove.MicrotimingFeatures.get_all_features()])
            reduction = groove.reduce_groove()
            groove_features = np.append(rhythm_timing_features, reduction)

            all_features.append(groove_features)
            all_names.append(names[j])
            all_palette_names.append(i)
            print(names[j])
    return all_features, all_names, all_palette_names

def get_difference_matrix():
    namesA = []
    namesB = []
    palettesA = []
    palettesB = []
    groovesA = []
    groovesB = []
    differenceMatrix = []
    i = 0

    path = 'Grooves/'

    with open('eval-pairings-palettes.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            if i < 1000:
                namesA.append(row[0])
                palettesA.append(path+row[1]+".bfd3pal")
                hitsMatrixA, timingMatrixA, tempoA = get_groove_from_BFD_palette(palettesA[i], namesA[i])
                groovesA.append(NewGroove(hitsMatrixA, timingMatrixA, tempoA,
                                          velocity_type="Transform", extract_features=True, name=namesA[i]))
                all_featuresA = np.hstack([groovesA[i].RhythmFeatures.get_all_features(),
                                         groovesA[i].MicrotimingFeatures.get_all_features()])

                namesB.append(row[2])
                palettesB.append(path+row[3]+".bfd3pal")
                hitsMatrixB, timingMatrixB, tempoB = get_groove_from_BFD_palette(palettesB[i], namesB[i])
                groovesB.append(NewGroove(hitsMatrixB, timingMatrixB, tempoB,
                                          velocity_type="Transform", extract_features=True,name=namesB[i]))
                all_featuresB = np.hstack([groovesB[i].RhythmFeatures.get_all_features(),
                                         groovesB[i].MicrotimingFeatures.get_all_features()])

                structureDifference = groovesA[i].reduce_groove() -  groovesB[i].reduce_groove()
                diff = np.abs(np.append((all_featuresA-all_featuresB), structureDifference))
                differenceMatrix.append(diff) #  features total
                print(diff.shape)

                print(groovesA[i].name,groovesB[i].name)
            i+=1
    np.save("difference matrix - features and structure.npy", differenceMatrix) #103 features
    return differenceMatrix

def getPLSRegression(differenceMatrix):
    similarityRatings = np.zeros([80])
    feature_length = 103

    X1 = np.nan_to_num(similarityRatings)  # (80,)
    featureSet = scale(differenceMatrix) # (80, 103) - 103 diff. values for 80 ratings

    i = 0
    with open('average-similarity-ratings.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            similarityRatings[i] = float(row[0])
            i += 1

    normalizedDifference = np.zeros([80, feature_length])
    differenceArray = np.zeros([80, feature_length])
    for i in range(80):
        differenceArray[i, :] = differenceMatrix[i]

    y = np.nan_to_num(similarityRatings)  # (80,)

    PLSreg = PLSRegression(n_components=16)
    PLSreg.fit(featureSet, y)
    print(featureSet.shape, 'featureset shape')
    return PLSreg, similarityRatings

differenceMatrix = np.load("difference matrix - features and structure.npy")
#
PLSReg, similarity_ratings = getPLSRegression(differenceMatrix)
#
all_features = np.load("Small_AL_Tester_unweighted.npy")
print(all_features[1].shape, 'allfeatures[1] shape')
print(len(all_features), 'allfeatures[1] shape')
PLS_reduced_features = []
for i in range(len(all_features)):
    print(all_features[i].shape)
    transformed_features = PLSReg.transform(all_features[i].reshape(all_features[i].shape[0],-1).T)

    PLS_reduced_features.append(transformed_features.flatten())
    print(i)
    print(PLS_reduced_features[i].shape)

np.save("Small_AL_Tester.npy", PLS_reduced_features)

#all_features, all_names, all_palette_names = generate_dataset_format_for_PLS()
#np.save("Small_AL_TesterNames.npy", [all_names,all_palette_names])
#np.save("Small_AL_Tester_unweighted.npy", all_features)

PLSReg = getPLSRegression(differenceMatrix)
print(len(all_features))
print(all_features[0].shape)