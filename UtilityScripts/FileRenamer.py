# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:12:49 2022

@author: JosephVermeil
"""

# %% Import

import os
import re
import pandas as pd

dateFormat1 = re.compile(r'\d{2}-\d{2}-\d{2}')
#os.rename(r'file path\OLD file name.file type',r'file path\NEW file name.file type')

# -----------------------------------------------------------------------------------------------
# %% Test
string = 'M3-1D4P6'

target_pattern = re.compile(r'M([\d-]+)D([\d-]+)P([\d-]+)')
match = target_pattern.match(string)
newStr = 'M{}_D{}_P{}'.format(*match.groups())

print(newStr) # 2019-04-01_23_59.jpg

target_pattern = r'M([\d-]+)D([\d-]+)P([\d-]+)'
replacement = r'M\1_D\2_P\3'
newStr = re.sub(target_pattern, replacement, string)
print(newStr)

# -----------------------------------------------------------------------------------------------
# %% Functions

def inverseDate(path, target='file', test = True, recursiveAction = False, exceptStrings = []):
    listAll = os.listdir(path)
    listFiles = []
    listDir = []
    listTarget = []
    for f in listAll:
        if os.path.isfile(os.path.join(path,f)):
            listFiles.append(f)
        elif os.path.isdir(os.path.join(path,f)):
            listDir.append(f)
    if target == 'file':
        listTarget = listFiles
    elif target == 'dir':
        listTarget = listDir
    elif target == 'all':
        listTarget = listAll
    renamedListTarget = []
    for f in listTarget:
        searchDate = re.search(dateFormat1, f)
        if searchDate:
            doExcept = False
            for s in exceptStrings:
                if s in f:
                    doExcept = True
                    print('Exception for ' + os.path.join(path,f))
            if not doExcept:
                foundDate = f[searchDate.start():searchDate.end()]
                newDate = foundDate[-2:] + foundDate[2:-2] + foundDate[:2]
                newFileName = f[:searchDate.start()] + newDate + f[searchDate.end():]
                if newDate[:2] not in [str(i) for i in range(18, 23)]:
                    print("newDate = " + newDate)
                    print('error detected in : ' + os.path.join(path,f))
                else:
                    renamedListTarget.append(newFileName)
                if not test:
                    os.rename(r''+os.path.join(path,f),r''+os.path.join(path,newFileName))
    if recursiveAction:
        # Update ListDir after potential renaming
        listAll = os.listdir(path)
        listDir = []
        for f in listAll:
            if os.path.isdir(os.path.join(path,f)):
                listDir.append(f)
        # Start going recursive
        for d in listDir:
            print("Let's go into " + os.path.join(path,d))
            inverseDate(os.path.join(path,d), target, test = test, recursiveAction = True, exceptStrings = exceptStrings)
    print(renamedListTarget)
    
def inverseDateInsideFile(path, test = True):
    f = open(path, 'r')
    data = f.read()
    searchedDates = re.findall(dateFormat1, data)
    for date in searchedDates:
        foundDate = date
        newDate = foundDate[-2:] + foundDate[2:-2] + foundDate[:2]
        data = data.replace(foundDate, newDate)
        if newDate[:2] != '20' and newDate[:2] != '21':
            print('error detected with date : ' + foundDate)
    f.close()
    print(data)
    if not test:
        f = open(path, 'w')
        f.write(data)
        f.close()

def findAndRename(path, target_string, new_string, target='file', 
                  test = True, recursiveAction = False, exceptStrings = []):
    listAll = os.listdir(path)
    listFiles = []
    listDir = []
    listTarget = []
    for f in listAll:
        if os.path.isfile(os.path.join(path,f)):
            listFiles.append(f)
        elif os.path.isdir(os.path.join(path,f)):
            listDir.append(f)
    if target == 'file':
        listTarget = listFiles
    elif target == 'dir':
        listTarget = listDir
    elif target == 'all':
        listTarget = listAll
    renamedListTarget = []
    for f in listTarget:
        searchString = re.search(target_string, f)
        if searchString:
            doExcept = False
            for s in exceptStrings:
                if s in f:
                    doExcept = True
                    print('Exception for ' + os.path.join(path,f))
            if not doExcept:
                foundString = f[searchString.start():searchString.end()]
                newFileName = f[:searchString.start()] + new_string + f[searchString.end():]
                # newFileName = f[:searchString.start()] + foundString[:2] + f[searchString.end():]
                renamedListTarget.append(newFileName)
                if not test:
                    new_path = os.path.join(path, newFileName)
                    if not os.path.isfile(new_path):
                        os.rename(r''+os.path.join(path, f),r''+os.path.join(path, newFileName))  
                # else:
                #     print(f)
    if recursiveAction:
        # Update ListDir after potential renaming
        listAll = os.listdir(path)
        listDir = []
        for f in listAll:
            if os.path.isdir(os.path.join(path,f)):
                listDir.append(f)
        # Start going recursive
        for d in listDir:
            print("Let's go into " + os.path.join(path,d))
            rlT = findAndRename(os.path.join(path,d), target_string, new_string, 
                          target=target, test = test, recursiveAction = True, exceptStrings = exceptStrings)
            renamedListTarget += rlT
    print(renamedListTarget)
    print(len(renamedListTarget))
    return(renamedListTarget)



def findPatternAndRename(path, target_pattern, new_pattern, target='file', 
                         test = True, recursiveAction = False, exceptStrings = []):
    listAll = os.listdir(path)
    listFiles = []
    listDir = []
    listTarget = []
    for f in listAll:
        if os.path.isfile(os.path.join(path,f)):
            listFiles.append(f)
        elif os.path.isdir(os.path.join(path,f)):
            listDir.append(f)
    if target == 'file':
        listTarget = listFiles
    elif target == 'dir':
        listTarget = listDir
    elif target == 'all':
        listTarget = listAll
    renamedListTarget = []
    for f in listTarget:
        searchPattern = re.search(target_pattern, f)
        if searchPattern:
            doExcept = False
            for s in exceptStrings:
                if s in f:
                    doExcept = True
                    print('Exception for ' + os.path.join(path,f))
            if not doExcept:
                foundString = f[searchPattern.start():searchPattern.end()]
                newString = re.sub(target_pattern, new_pattern, foundString)
                newFileName = f[:searchPattern.start()] + newString + f[searchPattern.end():]
                renamedListTarget.append(newFileName)
                if not test:
                    os.rename(r''+os.path.join(path,f),r''+os.path.join(path,newFileName))
    if recursiveAction:
        # Update ListDir after potential renaming
        listAll = os.listdir(path)
        listDir = []
        for f in listAll:
            if os.path.isdir(os.path.join(path,f)):
                listDir.append(f)
        # Start going recursive
        for d in listDir:
            print("Let's go into " + os.path.join(path,d))
            findPatternAndRename(os.path.join(path,d), target_pattern, new_pattern, target=target, 
                         test = test, recursiveAction = True, exceptStrings = exceptStrings)
    print(renamedListTarget)




def findAndRemove(path, target_string, target='file', test = True, 
                  recursiveAction = False, exceptStrings = []):
    listAll = os.listdir(path)
    listFiles = []
    listDir = []
    listTarget = []
    for f in listAll:
        if os.path.isfile(os.path.join(path,f)):
            listFiles.append(f)
        elif os.path.isdir(os.path.join(path,f)):
            listDir.append(f)
    if target == 'file':
        listTarget = listFiles
    elif target == 'dir':
        listTarget = listDir
    elif target == 'all':
        listTarget = listAll
    renamedListTarget = []
    for f in listTarget:
        searchString = re.search(target_string, f)
        if searchString:
            doExcept = False
            for s in exceptStrings:
                if s in f:
                    doExcept = True
                    print('Exception for ' + os.path.join(path,f))
            if not doExcept:
                foundString = f[searchString.start():searchString.end()]
                newFileName = f[:searchString.start()] + f[searchString.end():]
                # newFileName = f[:searchString.start()] + foundString[:2] + f[searchString.end():]
                renamedListTarget.append(newFileName)
                if not test:
                    new_path = os.path.join(path, newFileName)
                    if not os.path.isfile(new_path):
                        os.rename(r''+os.path.join(path, f),r''+os.path.join(path, newFileName))
                        
                # else:
                #     print(f)
                    
    if recursiveAction:
        # Update ListDir after potential renaming
        listAll = os.listdir(path)
        listDir = []
        for f in listAll:
            if os.path.isdir(os.path.join(path,f)):
                listDir.append(f)
        # Start going recursive
        for d in listDir:
            print("Let's go into " + os.path.join(path,d))
            rlT = findAndRemove(os.path.join(path,d), target_string, 
                          target=target, test = test, recursiveAction = True, exceptStrings = exceptStrings)
            renamedListTarget += rlT
            
    print(renamedListTarget)
    print(len(renamedListTarget))
    return(renamedListTarget)

# -----------------------------------------------------------------------------------------------
# %% Script Other renaming

# target_pattern = r'M([\d-]+)D([\d-]+)P([\d-]+)'
# new_pattern = r'M\1-D\2-P\3'

target_pattern = r'M([\d]+)_D([\d]+)_P([\d]+)_([^B])'
new_pattern = r'M\1_D\2_P\3_B1_\4'

path0 = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Analysis_Pulls/Tracks/25-09-19'

sub = ''
path = path0 + sub

findPatternAndRename(path, target_pattern, new_pattern, 
                    target = 'all', test = False, recursiveAction = True, exceptStrings = [])



# -----------------------------------------------------------------------------------------------
# %% Script Other renaming

# target_pattern = r'M([\d-]+)D([\d-]+)P([\d-]+)'
# new_pattern = r'M\1-D\2-P\3'

target_pattern = r'M([\d]+)_D([\d]+)_P([\d]+)_([\w-]+)_Tracks_B([\d]+)'
new_pattern = r'M\1_D\2_P\3_B\5_\4_Tracks'

path0 = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Analysis_Pulls/Tracks/25-09-19'

sub = ''
path = path0 + sub

findPatternAndRename(path, target_pattern, new_pattern, 
                    target = 'all', test = False, recursiveAction = True, exceptStrings = [])



# -----------------------------------------------------------------------------------------------
# %% Script Other renaming

# target_pattern = r'M([\d-]+)D([\d-]+)P([\d-]+)'
# new_pattern = r'M\1-D\2-P\3'

target_pattern = r'D([\d-]+)P([\d-]+)'
new_pattern = r'M1-D\1-P\2'

path0 = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Leica/25-08-26'
sub = ''
path = path0 + sub

findPatternAndRename(path, target_pattern, new_pattern, 
                    target = 'all', test = False, recursiveAction = True, exceptStrings = [])

# -----------------------------------------------------------------------------------------------
# %% Script Other renaming

s1 = '26-02-09'
s2 = '26-02-11'

path0 = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/26-02-11_UVonCytoplasmAndBeads'
sub = ''
path = path0 + sub

findAndRename(path, s1, s2, 
              target = 'file', test = False, recursiveAction = False, exceptStrings = [])

# -----------------------------------------------------------------------------------------------
# %% Script Other renaming

s1 = r'D\dP\d'
s2 = r'D\d-P\d'

path0 = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Leica'
sub = ''
path = path0 + sub

findAndRename(path, s1, s2, 
              target = 'all', test = False, recursiveAction = True, exceptStrings = [])

# -----------------------------------------------------------------------------------------------
# %% Remove spaces
        
path0 = 'E:/24-06-14_Chameleon_Compressions/All_tifs/'
sub = ''
path = path0 + sub

findAndRemove(path, r" +", 
              target = 'all', test = True, recursiveAction = True, exceptStrings = [])

# -----------------------------------------------------------------------------------------------
# %% Script Dates
        
inverseDate('D://MagneticPincherData//Raw_DC//Raw_DC_JV//', target = 'all', 
            test = True, recursiveAction = True, exceptStrings = ['Deptho'])

path0 = 'D://MagneticPincherData//Raw//'

# -----------------------------------------------------------------------------------------------