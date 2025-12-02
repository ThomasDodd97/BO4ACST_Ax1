from ax.service.ax_client import AxClient
from MiscMethods import MiscMethods_class
from PipettingMethods import PipettingMethods_class
from CSTMethods import CSTMethods_class
import numpy as np
from pathlib import Path
import pandas as pd
import os
from torch import Tensor
from botorch.test_functions.multi_objective import BraninCurrin

def BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis):
    """
    This function firstly ascertains whether there is a 'backup' file into which experimental
    values can be saved (values used during execution of experimental work), if there isn't
    then the function creates a new one and fills this with the latest experimental values.
    """
    try:
        my_file = Path(OptimisationSetup_obj.BackupPath_str)
        if my_file.is_file():
            df1 = pd.read_csv(OptimisationSetup_obj.BackupPath_str)
            df2 = pd.DataFrame({BackupVariableNames_lis[i]: BackupVariablesArrays_mat[i].tolist() for i in range(len(BackupVariableNames_lis))})
            df3 = pd.concat([df1,df2])
            df3 = df3.astype({f'{BackupVariableNames_lis[0]}':'int'})
            df3.to_csv(OptimisationSetup_obj.BackupPath_str,index=False)
        else:
            raise ValueError('No previous backup file available.')
    except:
        df = pd.DataFrame({BackupVariableNames_lis[i]: BackupVariablesArrays_mat[i].tolist() for i in range(len(BackupVariableNames_lis))})
        df = df.astype({f'{BackupVariableNames_lis[0]}':'int'})
        df.to_csv(OptimisationSetup_obj.BackupPath_str,index=False)

def ExecutionCheckerMethod20241024_func(client_obj,OptimisationSetup_obj):
    """
    This function checks whether the mixing method has been executed so that
    other parts of the program know when it is time to complete the trials.
    The first try except clause looks at whether there is a backup csv file
    and additionally the highest index values of this file and the Ax experiment
    dataframe to ascertain whether execution has taken place. If it has, it carries
    out the except clause.
    In the except clause a second check takes place- the running trials indices are
    cross-compared with the numbers of the files in the mechanical testing folder
    if these are equal then data is available for consideration.
    """
    checker_bool = False
    try:
        PreviousTrials_df = client_obj.summarize()
        RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        if RunningTrials_df.empty:
            print("There are no trials running! Therefore, none can be completed.")
        else:
            print("There are trials running!")
            my_file = Path(OptimisationSetup_obj.BackupPath_str)
            if my_file.is_file():
                print("Backup file available!")
                Backup_df = pd.read_csv(OptimisationSetup_obj.BackupPath_str)
                BackupIdx_arr = np.array(Backup_df["TrialIdx"])
                if RunningTrialsIdx_arr[-1] == BackupIdx_arr[-1]:
                    raise ValueError('Trials executed but uncompleted.')
                else:
                    print("Trials not executed so cannot be completed.")
            else:
                print("No backup file available!")
    except:
        print("There are trials that can be completed.")
        try:
            dir_lis = os.listdir(OptimisationSetup_obj.RawDataMT_str)
        except:
            print("Couldn't find a mechanical test data directory nor any mechanical test data, creating one, please place mechanical test data here.")
            os.mkdir(OptimisationSetup_obj.RawDataMT_str)
        dir_lis = os.listdir(OptimisationSetup_obj.RawDataMT_str)
        if dir_lis == []:
            print("No Files in Mechanical Testing Directory...")
        else:
            dir_lis = sorted([int(dir_str.rstrip('.csv')) for dir_str in dir_lis])
            if dir_lis == BackupIdx_arr.tolist():
                print("Mechanical Test Data Available - trials can be completed.")
                checker_bool = True
            else:
                print("Mechanical test data unavailable - please place in folder.")
    return checker_bool

def TargetRetrievalMethod20241023_func(client_obj,OptimisationSetup_obj):
    """
    This function is set up to retrieve mechanical test data and process it before passing
    an array of values to be saved into the Ax experiment.
    This function firstly checks whether there is a file 'Dimensions' dictating the dimensions
    of each sample generated in each of the trials executed. If this is lacking, then the function
    creates a new one. Secondly this function checks whether these is any data in the dimensions
    dataframe that could be of use using a try except clause. Within the try clause there is an if
    clause that makes a check for the correct contents and if apparent will retrieve the targets
    as required by invoking the appraisal object.
    """
    DownsamplingFactor_int = OptimisationSetup_obj.DownsamplingFactor_int
    HorizonValue_int = OptimisationSetup_obj.HorizonValue_int
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_arr = np.empty(0)
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                t_flt = CSTMethods_obj.Appraisal_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int)
                t_arr = np.append(arr=t_arr,values=t_flt)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

class Method20241026Dim1_class():
    """
    This class governs the tailored 2D experiment that involves the synthesis of IAUPR1
    which is carried out as follows:
    StockUPR1 (c1)
    I1 (c2)
    StockUPR1 + DeI -> UPR1 (x1)
    UPR1 + DMA -> AUPR1 (c3)
    AUPR1 + I1 -> IAUPR1 (c4)
    Once the function has prepared each of these trials with the operator it then saves
    all the calculated components and inputted notes into a backup csv so that the experimenter
    knows what was fundamentally done to each trial.
    Specified Variables in Script:
    # TrialIdx = Index of a Trial
    # x1 = decimal percentage w/w concentration of the UP1 in UPR1
    # c1 = decimal percentage w/w concentration of the UP1 in StockUPR1
    # c2 = decimal percentage w/w concentration of the CS in I1
    # c3 = decimal percentage w/w concentration of the UPR1 in AUPR1
    # c4 = decimal percentage w/w concentration of the AUPR1 in IAUPR1
    # a1 = g mass of UP1 in StockUPR1
    # a2 = g mass of DeI in StockUPR1
    # a3 = g mass of DeI added to StockUPR1 to make UPR1
    # a4 = g mass of DeI in UPR1
    # a5 = g mass of DMA added to UPR1 to make AUPR1
    # a6 = g mass of I1 added to AUPR1 to make the IAUPR1
    # a7 = g mass of CS in I1
    # a8 = g mass of BP in I1
    """
    def __init__(self):
        self.name = "Inner Class - Tailored Method20241026Dim1 Class"
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrievalMethod20241023_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()
        # Retrieve the trials data:
        print("Retrieving trials.")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])

        print(f"\n===== Preliminary Setup =====")
        print(f"Pre-Heat the Oven to {OptimisationSetup_obj.CuringRegime_lis[0].get('temperature_oc_flt')}oC.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Initial Pouring of StockUPR1 =====")
        StockUPRMass_g_arr = np.empty(0)
        print(f"Use a P10mL pipette set to 2 ml to transfer StockUPR1 to {len(RunningTrialsIdx_arr)} mould gaps.")
        for TrialIdx_int in RunningTrialsIdx_arr:
            print(f"\n-----Trial {TrialIdx_int}-----")
            StockUPRMass_g_arr = np.append(arr=StockUPRMass_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))

        print(f"\n===== Mixing UPR1 from StockUPR1 & DeI =====")
        UPR1Mass_g_arr = np.empty(0)
        DeIMass_g_arr = np.empty(0)
        UP1InStockUPR_g_arr = np.empty(0)
        DeIInStockUPR_g_arr = np.empty(0)
        DeIInUPR1_g_arr = np.empty(0)
        for TrialIdx_int,StockUPRMass_g_flt,RunningTrialsX1_flt in zip(RunningTrialsIdx_arr,StockUPRMass_g_arr,RunningTrialsX1_arr):
            a = StockUPRMass_g_flt # Mass of StockUPR1 in a mould gap 
            b = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt # Current percentage UP in StockUPR1
            c = RunningTrialsX1_flt # Desired percentage UP in final UPR1
            d = (a*b) # Mass of UP in the StockUPR1
            e = (a*(1-b)) # Mass of DeI in the StockUPR1
            f = (d/c)-a # Mass of DeI to be added to create UPR1 from StockUPR1
            g = d+e+f # The total mass of UPR1 generated
            h = e + f # Total mass of DeI in UPR1 generated
            UPR1Mass_g_arr = np.append(arr=UPR1Mass_g_arr,values=g)
            DeIMass_g_arr = np.append(arr=DeIMass_g_arr,values=f)
            UP1InStockUPR_g_arr = np.append(arr=UP1InStockUPR_g_arr,values=d)
            DeIInStockUPR_g_arr = np.append(arr=DeIInStockUPR_g_arr,values=e)
            DeIInUPR1_g_arr = np.append(arr=DeIInUPR1_g_arr,values=h)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(f,3)} g DeI")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(f,"DeI")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Mixing AUPR1 from UPR1 & DMA =====")
        AUPR1Mass_g_arr = np.empty(0)
        DMAMass_g_arr = np.empty(0)
        for TrialIdx_int,UPR1Mass_g_flt in zip(RunningTrialsIdx_arr,UPR1Mass_g_arr):
            a = UPR1Mass_g_flt # Mass of UPR1 in trial.
            b = OptimisationSetup_obj.AUPR1_UPR1vsDMA_Constant_DecPct_flt # Desired Decimalised Percentage of UPR in AUPR1
            c = 1-b # Desired decimalised percentage of DMA in AUPR1
            d = (a/b)-a # Dimethyl aniline to be added
            e = a + d # End mass of Accelerated Unsaturated Polyester Resin
            AUPR1Mass_g_arr = np.append(arr=AUPR1Mass_g_arr,values=e)
            DMAMass_g_arr = np.append(arr=DMAMass_g_arr,values=d)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(d,4)} g DMA")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(d,"DMA")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Mixing IAUPR1 from AUPR1 & I1 =====")
        IAUPRMass_g_arr = np.empty(0)
        I1Mass_g_arr = np.empty(0)
        CSinI1_g_arr = np.empty(0)
        BPinI1_g_arr = np.empty(0)
        for TrialIdx_int,AUPR1Mass_g_flt in zip(RunningTrialsIdx_arr,AUPR1Mass_g_arr):
            a = AUPR1Mass_g_flt # Mass of AUPR1 in trial.
            b = OptimisationSetup_obj.IAUPR1_AUPR1vsI1_Bounds_DecPct_lis # Desired decimalised percentage of AUPR1 in IAUPR1
            c = 1-b # Desired decimalised percentage of I1 in IAUPR1
            d = (a/b)-a # The mass of I1 to add.
            e = a + d # The total mass of initiated and accelerated unsaturated polyester resin IAUPR1
            f = d * OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt # Mass of CS in I1
            g = d * (1-OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt) # Mass of BP in I1
            IAUPRMass_g_arr = np.append(arr=IAUPRMass_g_arr,values=e)
            I1Mass_g_arr = np.append(arr=I1Mass_g_arr,values=d)
            CSinI1_g_arr = np.append(arr=CSinI1_g_arr,values=f)
            BPinI1_g_arr = np.append(arr=BPinI1_g_arr,values=g)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(d,2)} g I1")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")
        
        print(f"\n===== Curing Regime =====")
        for counter_int,CuringRegime_dict in enumerate(OptimisationSetup_obj.CuringRegime_lis):
            print(f"Stage{counter_int+1}: {CuringRegime_dict.get('time_mins_flt')} mins @ {CuringRegime_dict.get('temperature_oc_flt')} oC")

        # Backing up the variables calculated during the trials.
        BackupVariableNames_lis = ["TrialIdx","a1","a2","a3","a4","a5","a6","a7","a8"]
        BackupVariablesArrays_mat = np.array([RunningTrialsIdx_arr,UP1InStockUPR_g_arr,DeIInStockUPR_g_arr,DeIMass_g_arr,DeIInUPR1_g_arr,DMAMass_g_arr,I1Mass_g_arr,CSinI1_g_arr,BPinI1_g_arr])
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

class Method20241026Dim2_class():
    """
    This class governs the tailored 2D experiment that involves the synthesis of IAUPR1
    which is carried out as follows:
    StockUPR1 (c1)
    I1 (c2)
    StockUPR1 + DeI -> UPR1 (x1)
    UPR1 + DMA -> AUPR1 (c3)
    AUPR1 + I1 -> IAUPR1 (x2)
    Once the function has prepared each of these trials with the operator it then saves
    all the calculated components and inputted notes into a backup csv so that the experimenter
    knows what was fundamentally done to each trial.
    Specified Variables in Script:
    # TrialIdx = Index of a Trial
    # x1 = decimal percentage w/w concentration of the UP1 in UPR1
    # x2 = decimal percentage w/w concentration of the AUPR1 in IAUPR1
    # c1 = decimal percentage w/w concentration of the UP1 in StockUPR1
    # c2 = decimal percentage w/w concentration of the UPR1 in AUPR1
    # a1 = g mass of UP1 in StockUPR1
    # a2 = g mass of DeI in StockUPR1
    # a3 = g mass of DeI added to StockUPR1 to make UPR1
    # a4 = g mass of DeI in UPR1
    # a5 = g mass of DMA added to UPR1 to make AUPR1
    # a6 = g mass of I1 added to AUPR1 to make the IAUPR1
    # a7 = g mass of CS in I1
    # a8 = g mass of BP in I1
    """
    def __init__(self):
        self.name = "Inner Class - Tailored Method20241026Dim2 Class"
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrievalMethod20241023_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()
        # Retrieve the trials data:
        print("Retrieving trials.")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])

        print(f"\n===== Preliminary Setup =====")
        print(f"Pre-Heat the Oven to {OptimisationSetup_obj.CuringRegime_lis[0].get('temperature_oc_flt')}oC.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Initial Pouring of StockUPR1 =====")
        StockUPRMass_g_arr = np.empty(0)
        print(f"Use a P10mL pipette set to 2 ml to transfer StockUPR1 to {len(RunningTrialsIdx_arr)} mould gaps.")
        for TrialIdx_int in RunningTrialsIdx_arr:
            print(f"\n-----Trial {TrialIdx_int}-----")
            StockUPRMass_g_arr = np.append(arr=StockUPRMass_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))

        print(f"\n===== Mixing UPR1 from StockUPR1 & DeI =====")
        UPR1Mass_g_arr = np.empty(0)
        DeIMass_g_arr = np.empty(0)
        UP1InStockUPR_g_arr = np.empty(0)
        DeIInStockUPR_g_arr = np.empty(0)
        DeIInUPR1_g_arr = np.empty(0)
        for TrialIdx_int,StockUPRMass_g_flt,RunningTrialsX1_flt in zip(RunningTrialsIdx_arr,StockUPRMass_g_arr,RunningTrialsX1_arr):
            a = StockUPRMass_g_flt # Mass of StockUPR1 in a mould gap 
            b = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt # Current percentage UP in StockUPR1
            c = RunningTrialsX1_flt # Desired percentage UP in final UPR1
            d = (a*b) # Mass of UP in the StockUPR1
            e = (a*(1-b)) # Mass of DeI in the StockUPR1
            f = (d/c)-a # Mass of DeI to be added to create UPR1 from StockUPR1
            g = d+e+f # The total mass of UPR1 generated
            h = e + f # Total mass of DeI in UPR1 generated
            UPR1Mass_g_arr = np.append(arr=UPR1Mass_g_arr,values=g)
            DeIMass_g_arr = np.append(arr=DeIMass_g_arr,values=f)
            UP1InStockUPR_g_arr = np.append(arr=UP1InStockUPR_g_arr,values=d)
            DeIInStockUPR_g_arr = np.append(arr=DeIInStockUPR_g_arr,values=e)
            DeIInUPR1_g_arr = np.append(arr=DeIInUPR1_g_arr,values=h)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(f,3)} g DeI")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(f,"DeI")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Mixing AUPR1 from UPR1 & DMA =====")
        AUPR1Mass_g_arr = np.empty(0)
        DMAMass_g_arr = np.empty(0)
        for TrialIdx_int,UPR1Mass_g_flt in zip(RunningTrialsIdx_arr,UPR1Mass_g_arr):
            a = UPR1Mass_g_flt # Mass of UPR1 in trial.
            b = OptimisationSetup_obj.AUPR1_UPR1vsDMA_Constant_DecPct_flt # Desired Decimalised Percentage of UPR in AUPR1
            c = 1-b # Desired decimalised percentage of DMA in AUPR1
            d = (a/b)-a # Dimethyl aniline to be added
            e = a + d # End mass of Accelerated Unsaturated Polyester Resin
            AUPR1Mass_g_arr = np.append(arr=AUPR1Mass_g_arr,values=e)
            DMAMass_g_arr = np.append(arr=DMAMass_g_arr,values=d)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(d,4)} g DMA")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(d,"DMA")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Mixing IAUPR1 from AUPR1 & I1 =====")
        IAUPRMass_g_arr = np.empty(0)
        I1Mass_g_arr = np.empty(0)
        CSinI1_g_arr = np.empty(0)
        BPinI1_g_arr = np.empty(0)
        for TrialIdx_int,AUPR1Mass_g_flt,RunningTrialsX2_flt in zip(RunningTrialsIdx_arr,AUPR1Mass_g_arr,RunningTrialsX2_arr):
            a = AUPR1Mass_g_flt # Mass of AUPR1 in trial.
            b = RunningTrialsX2_flt # Desired decimalised percentage of AUPR1 in IAUPR1
            c = 1-b # Desired decimalised percentage of I1 in IAUPR1
            d = (a/b)-a # The mass of I1 to add.
            e = a + d # The total mass of initiated and accelerated unsaturated polyester resin IAUPR1
            f = d * OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt # Mass of CS in I1
            g = d * (1-OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt) # Mass of BP in I1
            IAUPRMass_g_arr = np.append(arr=IAUPRMass_g_arr,values=e)
            I1Mass_g_arr = np.append(arr=I1Mass_g_arr,values=d)
            CSinI1_g_arr = np.append(arr=CSinI1_g_arr,values=f)
            BPinI1_g_arr = np.append(arr=BPinI1_g_arr,values=g)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(d,2)} g I1")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")
        
        print(f"\n===== Curing Regime =====")
        for counter_int,CuringRegime_dict in enumerate(OptimisationSetup_obj.CuringRegime_lis):
            print(f"Stage{counter_int+1}: {CuringRegime_dict.get('time_mins_flt')} mins @ {CuringRegime_dict.get('temperature_oc_flt')} oC")

        # Backing up the variables calculated during the trials.
        BackupVariableNames_lis = ["TrialIdx","a1","a2","a3","a4","a5","a6","a7","a8"]
        BackupVariablesArrays_mat = np.array([RunningTrialsIdx_arr,UP1InStockUPR_g_arr,DeIInStockUPR_g_arr,DeIMass_g_arr,DeIInUPR1_g_arr,DMAMass_g_arr,I1Mass_g_arr,CSinI1_g_arr,BPinI1_g_arr])
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

class Method20241026Dim3_class():
    """
    This class governs the tailored 3D experiment that involves the synthesis of IAUPR1
    which is carried out as follows:
    StockUPR1 (c1)
    StockUPR1 + DeI -> UPR1 (x1)
    UPR1 + DMA -> AUPR1 (x2)
    AUPR1 + I1 -> IAUPR1 (x3)
    Once the function has prepared each of these trials with the operator it then saves
    all the calculated components and inputted notes into a backup csv so that the experimenter
    knows what was fundamentally done to each trial.
    Specified Variables in Script:
    # TrialIdx = Index of a Trial
    # x1 = decimal percentage w/w concentration of the UP1 in UPR1
    # x2 = decimal percentage w/w concentration of the UPR1 in AUPR1
    # x3 = decimal percentage w/w concentration of the AUPR1 in IAUPR1
    # c1 = decimal percentage w/w concentration of the UP1 in StockUPR1
    # a1 = g mass of UP1 in StockUPR1
    # a2 = g mass of DeI in StockUPR1
    # a3 = g mass of DeI added to StockUPR1 to make UPR1
    # a4 = g mass of DeI in UPR1
    # a5 = g mass of DMA added to UPR1 to make AUPR1
    # a6 = g mass of I1 added to AUPR1 to make the IAUPR1
    # a7 = g mass of CS in I1
    # a8 = g mass of BP in I1
    """
    def __init__(self):
        self.name = "Inner Class - Tailored Method20241026Dim3 Class"
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrievalMethod20241023_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()
        # Retrieve the trials data:
        print("Retrieving trials.")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
        RunningTrialsX3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].name}'])

        print(f"\n===== Preliminary Setup =====")
        print(f"Pre-Heat the Oven to {OptimisationSetup_obj.CuringRegime_lis[0].get('temperature_oc_flt')}oC.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Initial Pouring of StockUPR1 =====")
        StockUPRMass_g_arr = np.empty(0)
        print(f"Use a P10mL pipette set to 2 ml to transfer StockUPR1 to {len(RunningTrialsIdx_arr)} mould gaps.")
        for TrialIdx_int in RunningTrialsIdx_arr:
            print(f"\n-----Trial {TrialIdx_int}-----")
            StockUPRMass_g_arr = np.append(arr=StockUPRMass_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))

        print(f"\n===== Mixing UPR1 from StockUPR1 & DeI =====")
        UPR1Mass_g_arr = np.empty(0)
        DeIMass_g_arr = np.empty(0)
        UP1InStockUPR_g_arr = np.empty(0)
        DeIInStockUPR_g_arr = np.empty(0)
        DeIInUPR1_g_arr = np.empty(0)
        for TrialIdx_int,StockUPRMass_g_flt,RunningTrialsX1_flt in zip(RunningTrialsIdx_arr,StockUPRMass_g_arr,RunningTrialsX1_arr):
            a = StockUPRMass_g_flt # Mass of StockUPR1 in a mould gap 
            b = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt # Current percentage UP in StockUPR1
            c = RunningTrialsX1_flt # Desired percentage UP in final UPR1
            d = (a*b) # Mass of UP in the StockUPR1
            e = (a*(1-b)) # Mass of DeI in the StockUPR1
            f = (d/c)-a # Mass of DeI to be added to create UPR1 from StockUPR1
            g = d+e+f # The total mass of UPR1 generated
            h = e + f # Total mass of DeI in UPR1 generated
            UPR1Mass_g_arr = np.append(arr=UPR1Mass_g_arr,values=g)
            DeIMass_g_arr = np.append(arr=DeIMass_g_arr,values=f)
            UP1InStockUPR_g_arr = np.append(arr=UP1InStockUPR_g_arr,values=d)
            DeIInStockUPR_g_arr = np.append(arr=DeIInStockUPR_g_arr,values=e)
            DeIInUPR1_g_arr = np.append(arr=DeIInUPR1_g_arr,values=h)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(f,3)} g DeI")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(f,"DeI")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Mixing AUPR1 from UPR1 & DMA =====")
        AUPR1Mass_g_arr = np.empty(0)
        DMAMass_g_arr = np.empty(0)
        for TrialIdx_int,UPR1Mass_g_flt,RunningTrialsX2_flt in zip(RunningTrialsIdx_arr,UPR1Mass_g_arr,RunningTrialsX2_arr):
            a = UPR1Mass_g_flt # Mass of UPR1 in trial.
            b = RunningTrialsX2_flt # Desired Decimalised Percentage of UPR in AUPR1
            c = 1-b # Desired decimalised percentage of DMA in AUPR1
            d = (a/b)-a # Dimethyl aniline to be added
            e = a + d # End mass of Accelerated Unsaturated Polyester Resin
            AUPR1Mass_g_arr = np.append(arr=AUPR1Mass_g_arr,values=e)
            DMAMass_g_arr = np.append(arr=DMAMass_g_arr,values=d)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(d,4)} g DMA")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(d,"DMA")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Mixing IAUPR1 from AUPR1 & I1 =====")
        IAUPRMass_g_arr = np.empty(0)
        I1Mass_g_arr = np.empty(0)
        CSinI1_g_arr = np.empty(0)
        BPinI1_g_arr = np.empty(0)
        for TrialIdx_int,AUPR1Mass_g_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,AUPR1Mass_g_arr,RunningTrialsX3_arr):
            a = AUPR1Mass_g_flt # Mass of AUPR1 in trial.
            b = RunningTrialsX3_flt # Desired decimalised percentage of AUPR1 in IAUPR1
            c = 1-b # Desired decimalised percentage of I1 in IAUPR1
            d = (a/b)-a # The mass of I1 to add.
            e = a + d # The total mass of initiated and accelerated unsaturated polyester resin IAUPR1
            f = d * OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt # Mass of CS in I1
            g = d * (1-OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt) # Mass of BP in I1
            IAUPRMass_g_arr = np.append(arr=IAUPRMass_g_arr,values=e)
            I1Mass_g_arr = np.append(arr=I1Mass_g_arr,values=d)
            CSinI1_g_arr = np.append(arr=CSinI1_g_arr,values=f)
            BPinI1_g_arr = np.append(arr=BPinI1_g_arr,values=g)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(d,2)} g I1")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")
        
        print(f"\n===== Curing Regime =====")
        for counter_int,CuringRegime_dict in enumerate(OptimisationSetup_obj.CuringRegime_lis):
            print(f"Stage{counter_int+1}: {CuringRegime_dict.get('time_mins_flt')} mins @ {CuringRegime_dict.get('temperature_oc_flt')} oC")

        # Backing up the variables calculated during the trials.
        BackupVariableNames_lis = ["TrialIdx","a1","a2","a3","a4","a5","a6","a7","a8"]
        BackupVariablesArrays_mat = np.array([RunningTrialsIdx_arr,UP1InStockUPR_g_arr,DeIInStockUPR_g_arr,DeIMass_g_arr,DeIInUPR1_g_arr,DMAMass_g_arr,I1Mass_g_arr,CSinI1_g_arr,BPinI1_g_arr])
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

def goldstein(x,y):
    tmp1 = (1+np.power(x+y+1,2)*(19-14*x+3*np.power(x,2)-14*y+6*x*y+3*np.power(y,2)))
    tmp2 = (30+(np.power(2*x-3*y,2)*(18-32*x+12*np.power(x,2)+48*y-36*x*y+27*np.power(y,2))))
    return tmp1*tmp2

def booth(x,y):
    tmp1 = (x+(2*y)-7)**2
    tmp2 = ((2*x)+y-5)**2
    return tmp1+tmp2

def TargetRetriever20241028_func(client_obj,OptimisationSetup_obj):
    AllTrials_df = client_obj.summarize()
    RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
    RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
    t_arr = np.empty(0)
    for RunningTrialsIdx_int,RunningTrialsX1_flt,RunningTrialsX2_arr in zip(RunningTrialsIdx_arr,RunningTrialsX1_arr,RunningTrialsX2_arr):
        t_flt = booth(RunningTrialsX1_flt,RunningTrialsX2_arr)
        t_arr = np.append(arr=t_arr,values=t_flt)
    return t_arr

class Method20241028Dim2_class():
    """
    This function governs the tailored method retrieval of target values from
    a simple 2D function.
    """
    def __init__(self):
        self.name = "Inner Class - Tailored Method20241028Dim2_class Class"
        self.TargetRetriever = TargetRetriever20241028_func

class Method20250310Dim3_class():
    """
    This class governs the tailored 3D experiment that involves the
    synthesis of IUPR1. It is associated with a wider experiment which
    focusses upon optimising the curing system to be used with the
    2 stage bio-based thermosetting systems researched.

    It is to be carried out as follows:
    UP1 + DeI -> StockUPR1      [c1]    (Carried out separate from optimisation routine)
    CS + DBP -> I1              [c2]    (Carried out by the manufacturer of the initiating powder I1)
    StockUPR1 + DeI -> UPR1     [x1]    (Operator changes this during optimisation routine)
    I1 + CS -> I2               [x2]    (Operator changes this during optimisation routine)
    UPR1 + I2 -> IUPR1          [x3]    (Operator changes this during optimisation routine)

    Once the function has aided the operator in the preparation of the
    samples it makes sure to save notes about all the variables in a backup
    csv so that a human can return to any sample and recreate it at a later
    date with ease.

    Specified parameters:
    j1 = decimal percentage w/w concentration of the UP1 in StockUPR1
    j2 = decimal percentage w/w concentration of the CS in I1
    x1 = decimal percentage w/w concentration of the StockUPR1 in UPR1
    x2 = decimal percentage w/w concentration of the I1 in I2
    x3 = decimal percentage w/w concentration of the UPR1 in IUPR1
    a1 = Mass of UP1 in Stock-UPR1
    a2 = Mass of DeI in Stock-UPR1
    a3 = Mass of Stock-UPR1 in UPR1
    a4 = Mass of CS in I1
    a5 = Mass of DBP in I1
    a6 = Mass of I1 in I2
    a7 = Mass of DeI in UPR1
    a8 = Mass of UPR1 in IUPR1
    a9 = Mass of CS in I2
    a10 = Mass of I2 in IUPR1
    a11 = Mass of IUPR1
    b1 = Total Mass of DeI
    b2 = Total Mass of CS
    b3 = Total Mass of DBP
    b4 = Total Mass of UP1
    c1 = UP1-Normalised Total Mass of DeI
    c2 = UP1-Normalised Total Mass of CS
    c3 = UP1-Normalised Total Mass of DBP
    """
    def __init__(self):
        self.name = "Inner Class - Tailored Method20250310Dim3 Class"
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrievalMethod20241023_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()

        print("Retrieving trials.")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
        RunningTrialsX3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].name}'])

        print(f"\n===== Preliminary Setup =====")
        print(f"Pre-Heat the Oven to {OptimisationSetup_obj.CuringRegime_lis[0].get('temperature_oc_flt')}oC.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Initial Pouring of StockUPR1 =====")
        a3_g_arr = np.empty(0)
        print(f"Use a P10mL pipette set to 2 ml to transfer StockUPR1 to {len(RunningTrialsIdx_arr)} mould gaps.")
        for TrialIdx_int,RunningTrialsX1_flt,RunningTrialsX2_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,RunningTrialsX1_arr,RunningTrialsX2_arr,RunningTrialsX3_arr):
            j1_flt = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt   # Decimalised Percentage UP1 in StockUPR1
            j2_flt = OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt       # Decimalised Percentage CS in I1
            j3_flt = OptimisationSetup_obj.TargetMassOfIUPR1a11_Constant_g_flt       # Decimalised Percentage CS in I1
            x1_flt = RunningTrialsX1_flt        # Value of x1
            x2_flt = RunningTrialsX2_flt        # Value of x2
            x3_flt = RunningTrialsX3_flt        # Value of x3
            a3_flt = j3_flt                     # Mass of Stock-UPR1 in UPR1
            a1_flt = a3_flt*j1_flt              # Mass of UP1 in StockUPR1
            a2_flt = a3_flt*(1-j1_flt)          # Mass of DeI in StockUPR1
            a7_flt = (a3_flt/x1_flt)*(1-x1_flt) # Mass of DeI in UPR1 (a1_flt/x1_flt)-a3_flt 
            a8_flt = a3_flt+a7_flt              # Mass of UPR1 in IUPR1
            a10_flt = (a8_flt/x3_flt)-a8_flt    # Mass of I2 in IUPR1
            a6_flt = a10_flt*x2_flt             # Mass of I1 in I2
            a5_flt = a6_flt*(1-j2_flt)          # Mass of DBP in I1
            a4_flt = a6_flt*j2_flt              # Mass of CS is I1
            a9_flt = a10_flt*(1-x2_flt)         # Mass of CS in I2
            a11_flt = a8_flt+a10_flt            # Mass of IUPR1
            b1_flt = a5_flt                     # Total Mass of DBP
            b2_flt = a2_flt+a7_flt              # Total Mass of DEI
            b3_flt = a4_flt+a9_flt              # Total Mass of CS
            b4_flt = a1_flt                     # Total Mass of UP1
            c1_flt = b1_flt/b4_flt              # UP1-Normalised Total Mass of DeI
            c2_flt = b2_flt/b4_flt              # UP1-Normalised Total Mass of CS
            c3_flt = b3_flt/b4_flt              # UP1-Normalised Total Mass of DBP
            c1_flt = (j3_flt/a11_flt)*j3_flt    # Recommended Mass of UP1 at Beginning
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Recommended mass of Stock-UPR1: {round(c1_flt,2)} g")
            a3_g_arr = np.append(arr=a3_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))
        
        print(f"\n===== Mixing UPR1 from StockUPR1 & DeI =====")
        j1_arr = np.empty(0)
        j2_arr = np.empty(0)
        x1_arr = np.empty(0)
        x2_arr = np.empty(0)
        x3_arr = np.empty(0)
        a1_arr = np.empty(0)
        a2_arr = np.empty(0)
        a3_arr = np.empty(0)
        a4_arr = np.empty(0)
        a5_arr = np.empty(0)
        a6_arr = np.empty(0)
        a7_arr = np.empty(0)
        a8_arr = np.empty(0)
        a9_arr = np.empty(0)
        a10_arr = np.empty(0)
        a11_arr = np.empty(0)
        b1_arr = np.empty(0)
        b2_arr = np.empty(0)
        b3_arr = np.empty(0)
        b4_arr = np.empty(0)
        c1_arr = np.empty(0)
        c2_arr = np.empty(0)
        c3_arr = np.empty(0)
        for TrialIdx_int,a3_g_flt,RunningTrialsX1_flt,RunningTrialsX2_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,a3_g_arr,RunningTrialsX1_arr,RunningTrialsX2_arr,RunningTrialsX3_arr):
            j1_flt = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt   # Decimalised Percentage UP1 in StockUPR1
            j2_flt = OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt       # Decimalised Percentage CS in I1
            x1_flt = RunningTrialsX1_flt        # Value of x1
            x2_flt = RunningTrialsX2_flt        # Value of x2
            x3_flt = RunningTrialsX3_flt        # Value of x3
            a3_flt = a3_g_flt                   # Mass of Stock-UPR1 in UPR1
            a1_flt = a3_flt*j1_flt              # Mass of UP1 in StockUPR1
            a2_flt = a3_flt*(1-j1_flt)          # Mass of DeI in StockUPR1
            a7_flt = (a3_flt/x1_flt)*(1-x1_flt) # Mass of DeI in UPR1 (a1_flt/x1_flt)-a3_flt 
            a8_flt = a3_flt+a7_flt              # Mass of UPR1 in IUPR1
            a10_flt = (a8_flt/x3_flt)-a8_flt    # Mass of I2 in IUPR1
            a6_flt = a10_flt*x2_flt             # Mass of I1 in I2
            a5_flt = a6_flt*(1-j2_flt)          # Mass of DBP in I1
            a4_flt = a6_flt*j2_flt              # Mass of CS is I1
            a9_flt = a10_flt*(1-x2_flt)         # Mass of CS in I2
            a11_flt = a8_flt+a10_flt            # Mass of IUPR1
            b1_flt = a5_flt                     # Total Mass of DBP
            b2_flt = a2_flt+a7_flt              # Total Mass of DEI
            b3_flt = a4_flt+a9_flt              # Total Mass of CS
            b4_flt = a1_flt                     # Total Mass of UP1
            c1_flt = b1_flt/b4_flt              # UP1-Normalised Total Mass of DeI
            c2_flt = b2_flt/b4_flt              # UP1-Normalised Total Mass of CS
            c3_flt = b3_flt/b4_flt              # UP1-Normalised Total Mass of DBP
            j1_arr = np.append(arr=j1_arr,values=j1_flt)
            j2_arr = np.append(arr=j2_arr,values=j2_flt)
            x1_arr = np.append(arr=x1_arr,values=x1_flt)
            x2_arr = np.append(arr=x2_arr,values=x2_flt)
            x3_arr = np.append(arr=x3_arr,values=x3_flt)
            a1_arr = np.append(arr=a1_arr,values=a1_flt)
            a2_arr = np.append(arr=a2_arr,values=a2_flt)
            a3_arr = np.append(arr=a3_arr,values=a3_flt)
            a4_arr = np.append(arr=a4_arr,values=a4_flt)
            a5_arr = np.append(arr=a5_arr,values=a5_flt)
            a6_arr = np.append(arr=a6_arr,values=a6_flt)
            a7_arr = np.append(arr=a7_arr,values=a7_flt)
            a8_arr = np.append(arr=a8_arr,values=a8_flt)
            a9_arr = np.append(arr=a9_arr,values=a9_flt)
            a10_arr = np.append(arr=a10_arr,values=a10_flt)
            a11_arr = np.append(arr=a11_arr,values=a11_flt)
            b1_arr = np.append(arr=b1_arr,values=b1_flt)
            b2_arr = np.append(arr=b2_arr,values=b2_flt)
            b3_arr = np.append(arr=b3_arr,values=b3_flt)
            b4_arr = np.append(arr=b4_arr,values=b4_flt)
            c1_arr = np.append(arr=c1_arr,values=c1_flt)
            c2_arr = np.append(arr=c2_arr,values=c2_flt)
            c3_arr = np.append(arr=c3_arr,values=c3_flt)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(a7_flt,3)} g DmI")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(a7_flt,"DmI")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")
        
        print(f"\n===== Mixing I2 from CS & I1 before adding I2 to UPR1 to form IUPR1 =====")
        for TrialIdx_int,a6_flt,a9_flt in zip(RunningTrialsIdx_arr,a6_arr.tolist(),a9_arr.tolist()):
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"In a pair of weighing boats:")
            print(f"Add {round(a6_flt,3)} g I1")
            print(f"Add {round(a9_flt,3)} g CS")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Curing Regime =====")
        for counter_int,CuringRegime_dict in enumerate(OptimisationSetup_obj.CuringRegime_lis):
            print(f"Stage{counter_int+1}: {CuringRegime_dict.get('time_mins_flt')} mins @ {CuringRegime_dict.get('temperature_oc_flt')} oC")

        # Backing up the variables calculated during the trials.
        BackupVariableNames_lis = ["TrialIdx","j1","j2","x1","x2","x3","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","b1","b2","b3","b4","c1","c2","c3"]
        BackupVariablesArrays_mat = np.array([RunningTrialsIdx_arr,j1_arr,j2_arr,x1_arr,x2_arr,x3_arr,a1_arr,a2_arr,a3_arr,a4_arr,a5_arr,a6_arr,a7_arr,a8_arr,a9_arr,a10_arr,a11_arr,b1_arr,b2_arr,b3_arr,b4_arr,c1_arr,c2_arr,c3_arr])
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

def UCSTargetRetrievalMethod20250518_func(client_obj,OptimisationSetup_obj):
    """
    This function is set up to retrieve mechanical test data and process it before passing
    an array of values to be saved into the Ax experiment.
    This function firstly checks whether there is a file 'Dimensions' dictating the dimensions
    of each sample generated in each of the trials executed. If this is lacking, then the function
    creates a new one. Secondly this function checks whether these is any data in the dimensions
    dataframe that could be of use using a try except clause. Within the try clause there is an if
    clause that makes a check for the correct contents and if apparent will retrieve the targets
    as required by invoking the appraisal object.
    The target values will be ultimate compressive strength.
    """
    DownsamplingFactor_int = OptimisationSetup_obj.DownsamplingFactor_int
    HorizonValue_int = OptimisationSetup_obj.HorizonValue_int
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_arr = np.empty(0)
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                t_flt = CSTMethods_obj.UltimateCompressiveStrengther_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int)
                t_arr = np.append(arr=t_arr,values=t_flt)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def YMTargetRetrieverMethod20250518_func(client_obj,OptimisationSetup_obj):
    """
    This function is set up to retrieve mechanical test data and process it before passing
    an array of values to be saved into the Ax experiment.
    This function firstly checks whether there is a file 'Dimensions' dictating the dimensions
    of each sample generated in each of the trials executed. If this is lacking, then the function
    creates a new one. Secondly this function checks whether these is any data in the dimensions
    dataframe that could be of use using a try except clause. Within the try clause there is an if
    clause that makes a check for the correct contents and if apparent will retrieve the targets
    as required by invoking the appraisal object.
    The target values will be young's modulus.
    """
    DownsamplingFactor_int = OptimisationSetup_obj.DownsamplingFactor_int
    HorizonValue_int = OptimisationSetup_obj.HorizonValue_int
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_arr = np.empty(0)
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                t_flt = CSTMethods_obj.YoungsModder_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int)
                t_arr = np.append(arr=t_arr,values=t_flt)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr
    
def PLTargetRetrieverMethod20250903_func(client_obj,OptimisationSetup_obj):
    """
    This function finds the proportionality limit. It does this by finding the inflection point (in the same fashion as
    for calculating the young's modulus), then it takes the stress achieved at this point as the limit of proportionality.
    """
    DownsamplingFactor_int = OptimisationSetup_obj.DownsamplingFactor_int
    HorizonValue_int = OptimisationSetup_obj.HorizonValue_int
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_arr = np.empty(0)
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                t_flt = CSTMethods_obj.ProportionalLimiter_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int)
                t_arr = np.append(arr=t_arr,values=t_flt)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def PLTargetRetrieverMethod20250911_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_arr = np.empty(0)
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                LimitOfProportionality_flt = CSTMethods_obj.LimitOfProportionality_func(StressStrain_mat,PeakStrain_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                t_arr = np.append(arr=t_arr,values=LimitOfProportionality_flt)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def YMTargetRetrieverMethod20250911_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_arr = np.empty(0)
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                YoungsModulus_flt = CSTMethods_obj.YoungsModulus_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                AbsoluteYoungsModulus_flt = abs(YoungsModulus_flt)
                t_arr = np.append(arr=t_arr,values=AbsoluteYoungsModulus_flt)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def YBPTargetRetrieverMethod20250911_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_arr = np.empty(0)
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                YieldBreakPoint_flt = CSTMethods_obj.YieldBreakPoint_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,StressStrain_mat,ArbitraryGradientCutoff=ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ArbitrarySustainedRise_int=ArbitrarySustainedRise_int,ChallengePlotAcceptability_bool=True)
                t_arr = np.append(arr=t_arr,values=YieldBreakPoint_flt)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def PLYMTargetRetrieverMethod20250911_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_lis = []
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                LimitOfProportionality_flt = CSTMethods_obj.LimitOfProportionality_func(StressStrain_mat,PeakStrain_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                t1_flt = LimitOfProportionality_flt
                YoungsModulus_flt = CSTMethods_obj.YoungsModulus_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                AbsoluteYoungsModulus_flt = abs(YoungsModulus_flt)
                t2_flt = AbsoluteYoungsModulus_flt
                t_lis.append([t1_flt,t2_flt])
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr


def YBPYMTargetRetrieverMethod20251019_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_lis = []
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                LimitOfProportionality_flt = CSTMethods_obj.LimitOfProportionality_func(StressStrain_mat,PeakStrain_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                YieldBreakPoint_flt = CSTMethods_obj.YieldBreakPoint_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,StressStrain_mat,ArbitraryGradientCutoff=ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ArbitrarySustainedRise_int=ArbitrarySustainedRise_int,ChallengePlotAcceptability_bool=True)
                t1_flt = YieldBreakPoint_flt
                YoungsModulus_flt = CSTMethods_obj.YoungsModulus_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                AbsoluteYoungsModulus_flt = abs(YoungsModulus_flt)
                t2_flt = AbsoluteYoungsModulus_flt
                t_lis.append([t1_flt,t2_flt])
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def PLYBPTargetRetrieverMethod20251019_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_lis = []            
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                LimitOfProportionality_flt = CSTMethods_obj.LimitOfProportionality_func(StressStrain_mat,PeakStrain_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                t1_flt = LimitOfProportionality_flt
                YieldBreakPoint_flt = CSTMethods_obj.YieldBreakPoint_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,StressStrain_mat,ArbitraryGradientCutoff=ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ArbitrarySustainedRise_int=ArbitrarySustainedRise_int,ChallengePlotAcceptability_bool=True)
                t2_flt = YieldBreakPoint_flt
                t_lis.append([t1_flt,t2_flt])
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def PLYBPYMTargetRetrieverMethod20251013_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_lis = []
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                LimitOfProportionality_flt = CSTMethods_obj.LimitOfProportionality_func(StressStrain_mat,PeakStrain_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                t1_flt = LimitOfProportionality_flt
                YieldBreakPoint_flt = CSTMethods_obj.YieldBreakPoint_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,StressStrain_mat,ArbitraryGradientCutoff=ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ArbitrarySustainedRise_int=ArbitrarySustainedRise_int,ChallengePlotAcceptability_bool=True) # True
                t2_flt = YieldBreakPoint_flt
                YoungsModulus_flt = CSTMethods_obj.YoungsModulus_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                AbsoluteYoungsModulus_flt = abs(YoungsModulus_flt)
                t3_flt = AbsoluteYoungsModulus_flt
                t_lis.append([t1_flt,t2_flt,t3_flt])
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

    
def AddUCSvsYMRetrieverMethod20250518_func(client_obj,OptimisationSetup_obj):
    """
    This function is set up to retrieve mechanical test data and process it before passing
    an array of values to be saved into the Ax experiment.
    This function firstly checks whether there is a file 'Dimensions' dictating the dimensions
    of each sample generated in each of the trials executed. If this is lacking, then the function
    creates a new one. Secondly this function checks whether these is any data in the dimensions
    dataframe that could be of use using a try except clause. Within the try clause there is an if
    clause that makes a check for the correct contents and if apparent will retrieve the targets
    as required by invoking the appraisal object.
    The target values will be an additive and then penalised mixture of both youngs modulus and
    ultimate compressive strength.
    """
    DownsamplingFactor_int = OptimisationSetup_obj.DownsamplingFactor_int
    HorizonValue_int = OptimisationSetup_obj.HorizonValue_int
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_arr = np.empty(0)
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                AvUCS_flt,AvYM_flt,StdUCS_flt,StdYM_flt = CSTMethods_obj.Penaliser_func(OptimisationSetup_obj)
                UCS_flt = CSTMethods_obj.UltimateCompressiveStrengther_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int)
                YM_flt = CSTMethods_obj.YoungsModder_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int)
                t_flt = UCS_flt+((StdUCS_flt/StdYM_flt)*YM_flt)
                t_arr = np.append(arr=t_arr,values=t_flt)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

class Method20250518Dim3_class():
    def __init__(self):
        self.name = "Inner Class - Tailored Method20250518Dim3 Class"
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.UCSTargetRetriever = UCSTargetRetrievalMethod20250518_func
        self.YMTargetRetriever = YMTargetRetrieverMethod20250518_func
        self.AddUCSvsYMRetriever = AddUCSvsYMRetrieverMethod20250518_func
        self.PLTargetRetriever = PLTargetRetrieverMethod20250903_func

def BraninCurrinOne_func(x_1,x_2):
    tens = Tensor([x_1,x_2])
    BC_obj = BraninCurrin()
    return np.float64(BC_obj._evaluate_true(tens)[0])

def BraninCurrinTwo_func(x_1,x_2):
    tens = Tensor([x_1,x_2])
    BC_obj = BraninCurrin()
    return np.float64(BC_obj._evaluate_true(tens)[1])

def TargetRetriever20250623_func(client_obj,OptimisationSetup_obj):
    AllTrials_df = client_obj.summarize()
    RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
    RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
    t_lis = []
    for RunningTrialsIdx_int,RunningTrialsX1_flt,RunningTrialsX2_flt in zip(RunningTrialsIdx_arr,RunningTrialsX1_arr,RunningTrialsX2_arr):
        t1_flt = BraninCurrinOne_func(RunningTrialsX1_flt,RunningTrialsX2_flt)
        t2_flt = BraninCurrinTwo_func(RunningTrialsX1_flt,RunningTrialsX2_flt)
        t_lis.append([t1_flt,t2_flt])
    t_arr = np.array(t_lis)
    return t_arr

class Method20250623Dim2_class():
    def __init__(self):
        self.name = "Tailored Method20250623Dim2 Branin Currin Test Function Class"
        self.TargetRetriever = TargetRetriever20250623_func

def TargetRetrievalMethod20250623_func(client_obj,OptimisationSetup_obj):
    """
    This function is set up to retrieve mechanical test data and process it before passing
    an array of values to be saved into the Ax experiment.
    This function firstly checks whether there is a file 'Dimensions' dictating the dimensions
    of each sample generated in each of the trials executed. If this is lacking, then the function
    creates a new one. Secondly this function checks whether these is any data in the dimensions
    dataframe that could be of use using a try except clause. Within the try clause there is an if
    clause that makes a check for the correct contents and if apparent will retrieve the targets
    as required by invoking the appraisal object.
    """
    DownsamplingFactor_int = OptimisationSetup_obj.DownsamplingFactor_int
    HorizonValue_int = OptimisationSetup_obj.HorizonValue_int
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_lis = []
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                t2_flt = CSTMethods_obj.YoungsModder_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int)
                print(t2_flt)
                t1_flt = CSTMethods_obj.UltimateCompressiveStrengther_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int)
                print(t1_flt)
                t_lis.append([t1_flt,t2_flt])
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

class Method20250623Dim3_class():
    """
    This class governs the tailored 3D experiment that involves the
    synthesis of IUPR1. It is associated with a wider experiment which
    focusses upon optimising the curing system to be used with the
    2 stage bio-based thermosetting systems researched.
    [THIS IS AN ADAPTED FORM OF THE EXPERIMENT FOR STYRENE-BASED RESIN SYSTEM]

    It is to be carried out as follows:
    UP1 + STY -> StockUPR1      [c1]    (Carried out separate from optimisation routine)
    CS + DBP -> I1              [c2]    (Carried out by the manufacturer of the initiating powder I1)
    StockUPR1 + STY -> UPR1     [x1]    (Operator changes this during optimisation routine)
    I1 + CS -> I2               [x2]    (Operator changes this during optimisation routine)
    UPR1 + I2 -> IUPR1          [x3]    (Operator changes this during optimisation routine)

    Once the function has aided the operator in the preparation of the
    samples it makes sure to save notes about all the variables in a backup
    csv so that a human can return to any sample and recreate it at a later
    date with ease.

    Specified parameters:
    j1 = decimal percentage w/w concentration of the UP1 in StockUPR1
    j2 = decimal percentage w/w concentration of the CS in I1
    x1 = decimal percentage w/w concentration of the StockUPR1 in UPR1
    x2 = decimal percentage w/w concentration of the I1 in I2
    x3 = decimal percentage w/w concentration of the UPR1 in IUPR1
    a1 = Mass of UP1 in Stock-UPR1
    a2 = Mass of STY in Stock-UPR1
    a3 = Mass of Stock-UPR1 in UPR1
    a4 = Mass of CS in I1
    a5 = Mass of DBP in I1
    a6 = Mass of I1 in I2
    a7 = Mass of STY in UPR1
    a8 = Mass of UPR1 in IUPR1
    a9 = Mass of CS in I2
    a10 = Mass of I2 in IUPR1
    a11 = Mass of IUPR1
    b1 = Total Mass of STY
    b2 = Total Mass of CS
    b3 = Total Mass of DBP
    b4 = Total Mass of UP1
    c1 = UP1-Normalised Total Mass of STY
    c2 = UP1-Normalised Total Mass of CS
    c3 = UP1-Normalised Total Mass of DBP
    """
    def __init__(self):
        self.name = "Tailored Method20250623Dim3 Styrene-based Resin Dual-Objective Optimisation Class"
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrievalMethod20250623_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()

        print("Retrieving trials.")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
        RunningTrialsX3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].name}'])

        print(f"\n===== Preliminary Setup =====")
        print(f"Pre-Heat the Oven to {OptimisationSetup_obj.CuringRegime_lis[0].get('temperature_oc_flt')}oC.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Initial Pouring of StockUPR1 =====")
        a3_g_arr = np.empty(0)
        print(f"Use a P10mL pipette set to 2 ml to transfer StockUPR1 to {len(RunningTrialsIdx_arr)} mould gaps.")
        for TrialIdx_int,RunningTrialsX1_flt,RunningTrialsX2_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,RunningTrialsX1_arr,RunningTrialsX2_arr,RunningTrialsX3_arr):
            j1_flt = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt   # Decimalised Percentage UP1 in StockUPR1
            j2_flt = OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt       # Decimalised Percentage CS in I1
            j3_flt = OptimisationSetup_obj.TargetMassOfIUPR1a11_Constant_g_flt       # Decimalised Percentage CS in I1
            x1_flt = RunningTrialsX1_flt        # Value of x1
            x2_flt = RunningTrialsX2_flt        # Value of x2
            x3_flt = RunningTrialsX3_flt        # Value of x3
            a3_flt = j3_flt                     # Mass of Stock-UPR1 in UPR1
            a1_flt = a3_flt*j1_flt              # Mass of UP1 in StockUPR1
            a2_flt = a3_flt*(1-j1_flt)          # Mass of STY in StockUPR1
            a7_flt = (a3_flt/x1_flt)*(1-x1_flt) # Mass of STY in UPR1 (a1_flt/x1_flt)-a3_flt 
            a8_flt = a3_flt+a7_flt              # Mass of UPR1 in IUPR1
            a10_flt = (a8_flt/x3_flt)-a8_flt    # Mass of I2 in IUPR1
            a6_flt = a10_flt*x2_flt             # Mass of I1 in I2
            a5_flt = a6_flt*(1-j2_flt)          # Mass of DBP in I1
            a4_flt = a6_flt*j2_flt              # Mass of CS is I1
            a9_flt = a10_flt*(1-x2_flt)         # Mass of CS in I2
            a11_flt = a8_flt+a10_flt            # Mass of IUPR1
            b1_flt = a5_flt                     # Total Mass of DBP
            b2_flt = a2_flt+a7_flt              # Total Mass of STY
            b3_flt = a4_flt+a9_flt              # Total Mass of CS
            b4_flt = a1_flt                     # Total Mass of UP1
            c1_flt = b1_flt/b4_flt              # UP1-Normalised Total Mass of STY
            c2_flt = b2_flt/b4_flt              # UP1-Normalised Total Mass of CS
            c3_flt = b3_flt/b4_flt              # UP1-Normalised Total Mass of DBP
            c1_flt = (j3_flt/a11_flt)*j3_flt    # Recommended Mass of UP1 at Beginning
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Recommended mass of Stock-UPR1: {round(c1_flt,2)} g")
            a3_g_arr = np.append(arr=a3_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))
        
        print(f"\n===== Mixing UPR1 from StockUPR1 & STY =====")
        j1_arr = np.empty(0)
        j2_arr = np.empty(0)
        x1_arr = np.empty(0)
        x2_arr = np.empty(0)
        x3_arr = np.empty(0)
        a1_arr = np.empty(0)
        a2_arr = np.empty(0)
        a3_arr = np.empty(0)
        a4_arr = np.empty(0)
        a5_arr = np.empty(0)
        a6_arr = np.empty(0)
        a7_arr = np.empty(0)
        a8_arr = np.empty(0)
        a9_arr = np.empty(0)
        a10_arr = np.empty(0)
        a11_arr = np.empty(0)
        b1_arr = np.empty(0)
        b2_arr = np.empty(0)
        b3_arr = np.empty(0)
        b4_arr = np.empty(0)
        c1_arr = np.empty(0)
        c2_arr = np.empty(0)
        c3_arr = np.empty(0)
        for TrialIdx_int,a3_g_flt,RunningTrialsX1_flt,RunningTrialsX2_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,a3_g_arr,RunningTrialsX1_arr,RunningTrialsX2_arr,RunningTrialsX3_arr):
            j1_flt = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt   # Decimalised Percentage UP1 in StockUPR1
            j2_flt = OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt       # Decimalised Percentage CS in I1
            x1_flt = RunningTrialsX1_flt        # Value of x1
            x2_flt = RunningTrialsX2_flt        # Value of x2
            x3_flt = RunningTrialsX3_flt        # Value of x3
            a3_flt = a3_g_flt                   # Mass of Stock-UPR1 in UPR1
            a1_flt = a3_flt*j1_flt              # Mass of UP1 in StockUPR1
            a2_flt = a3_flt*(1-j1_flt)          # Mass of STY in StockUPR1
            a7_flt = (a3_flt/x1_flt)*(1-x1_flt) # Mass of STY in UPR1 (a1_flt/x1_flt)-a3_flt 
            a8_flt = a3_flt+a7_flt              # Mass of UPR1 in IUPR1
            a10_flt = (a8_flt/x3_flt)-a8_flt    # Mass of I2 in IUPR1
            a6_flt = a10_flt*x2_flt             # Mass of I1 in I2
            a5_flt = a6_flt*(1-j2_flt)          # Mass of DBP in I1
            a4_flt = a6_flt*j2_flt              # Mass of CS is I1
            a9_flt = a10_flt*(1-x2_flt)         # Mass of CS in I2
            a11_flt = a8_flt+a10_flt            # Mass of IUPR1
            b1_flt = a5_flt                     # Total Mass of DBP
            b2_flt = a2_flt+a7_flt              # Total Mass of STY
            b3_flt = a4_flt+a9_flt              # Total Mass of CS
            b4_flt = a1_flt                     # Total Mass of UP1
            c1_flt = b1_flt/b4_flt              # UP1-Normalised Total Mass of STY
            c2_flt = b2_flt/b4_flt              # UP1-Normalised Total Mass of CS
            c3_flt = b3_flt/b4_flt              # UP1-Normalised Total Mass of DBP
            j1_arr = np.append(arr=j1_arr,values=j1_flt)
            j2_arr = np.append(arr=j2_arr,values=j2_flt)
            x1_arr = np.append(arr=x1_arr,values=x1_flt)
            x2_arr = np.append(arr=x2_arr,values=x2_flt)
            x3_arr = np.append(arr=x3_arr,values=x3_flt)
            a1_arr = np.append(arr=a1_arr,values=a1_flt)
            a2_arr = np.append(arr=a2_arr,values=a2_flt)
            a3_arr = np.append(arr=a3_arr,values=a3_flt)
            a4_arr = np.append(arr=a4_arr,values=a4_flt)
            a5_arr = np.append(arr=a5_arr,values=a5_flt)
            a6_arr = np.append(arr=a6_arr,values=a6_flt)
            a7_arr = np.append(arr=a7_arr,values=a7_flt)
            a8_arr = np.append(arr=a8_arr,values=a8_flt)
            a9_arr = np.append(arr=a9_arr,values=a9_flt)
            a10_arr = np.append(arr=a10_arr,values=a10_flt)
            a11_arr = np.append(arr=a11_arr,values=a11_flt)
            b1_arr = np.append(arr=b1_arr,values=b1_flt)
            b2_arr = np.append(arr=b2_arr,values=b2_flt)
            b3_arr = np.append(arr=b3_arr,values=b3_flt)
            b4_arr = np.append(arr=b4_arr,values=b4_flt)
            c1_arr = np.append(arr=c1_arr,values=c1_flt)
            c2_arr = np.append(arr=c2_arr,values=c2_flt)
            c3_arr = np.append(arr=c3_arr,values=c3_flt)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(a7_flt,3)} g STY")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(a7_flt,"STY")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")
        
        print(f"\n===== Mixing I2 from CS & I1 before adding I2 to UPR1 to form IUPR1 =====")
        for TrialIdx_int,a6_flt,a9_flt in zip(RunningTrialsIdx_arr,a6_arr.tolist(),a9_arr.tolist()):
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"In a pair of weighing boats:")
            print(f"Add {round(a6_flt,3)} g I1")
            print(f"Add {round(a9_flt,3)} g CS")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Curing Regime =====")
        for counter_int,CuringRegime_dict in enumerate(OptimisationSetup_obj.CuringRegime_lis):
            print(f"Stage{counter_int+1}: {CuringRegime_dict.get('time_mins_flt')} mins @ {CuringRegime_dict.get('temperature_oc_flt')} oC")

        # Backing up the variables calculated during the trials.
        BackupVariableNames_lis = ["TrialIdx","j1","j2","x1","x2","x3","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","b1","b2","b3","b4","c1","c2","c3"]
        BackupVariablesArrays_mat = np.array([RunningTrialsIdx_arr,j1_arr,j2_arr,x1_arr,x2_arr,x3_arr,a1_arr,a2_arr,a3_arr,a4_arr,a5_arr,a6_arr,a7_arr,a8_arr,a9_arr,a10_arr,a11_arr,b1_arr,b2_arr,b3_arr,b4_arr,c1_arr,c2_arr,c3_arr])
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

def TargetRetrievalMethod20250625_func(client_obj,OptimisationSetup_obj):
    """
    This function is set up to retrieve mechanical test data and process it before passing
    an array of values to be saved into the Ax experiment.
    This function firstly checks whether there is a file 'Dimensions' dictating the dimensions
    of each sample generated in each of the trials executed. If this is lacking, then the function
    creates a new one. Secondly this function checks whether these is any data in the dimensions
    dataframe that could be of use using a try except clause. Within the try clause there is an if
    clause that makes a check for the correct contents and if apparent will retrieve the targets
    as required by invoking the appraisal object.
    """
    DownsamplingFactor_int = OptimisationSetup_obj.DownsamplingFactor_int
    HorizonValue_int = OptimisationSetup_obj.HorizonValue_int
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_lis = []
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                t2_flt = CSTMethods_obj.YoungsModder_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int)
                print()
                print(t2_flt)
                t1_flt = CSTMethods_obj.UltimateCompressiveStrengther_func(CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int)
                print(t1_flt)
                t_lis.append([t1_flt,t2_flt])
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr


class Method20250625Dim3_class():
    """
    This class governs the tailored 3D experiment that involves the
    synthesis of IUPR1. It is associated with a wider experiment which
    focusses upon optimising the curing system to be used with the
    2 stage bio-based thermosetting systems researched.
    [THIS IS AN ADAPTED FORM OF THE EXPERIMENT FOR A DUAL OBJECTIVE SEARCH OF THE PARAMETER SPACE]

    It is to be carried out as follows:
    UP1 + DEI -> StockUPR1      [c1]    (Carried out separate from optimisation routine)
    CS + DBP -> I1              [c2]    (Carried out by the manufacturer of the initiating powder I1)
    StockUPR1 + DEI -> UPR1     [x1]    (Operator changes this during optimisation routine)
    I1 + CS -> I2               [x2]    (Operator changes this during optimisation routine)
    UPR1 + I2 -> IUPR1          [x3]    (Operator changes this during optimisation routine)

    Once the function has aided the operator in the preparation of the
    samples it makes sure to save notes about all the variables in a backup
    csv so that a human can return to any sample and recreate it at a later
    date with ease.

    Specified parameters:
    j1 = decimal percentage w/w concentration of the UP1 in StockUPR1
    j2 = decimal percentage w/w concentration of the CS in I1
    x1 = decimal percentage w/w concentration of the StockUPR1 in UPR1
    x2 = decimal percentage w/w concentration of the I1 in I2
    x3 = decimal percentage w/w concentration of the UPR1 in IUPR1
    a1 = Mass of UP1 in Stock-UPR1
    a2 = Mass of DEI in Stock-UPR1
    a3 = Mass of Stock-UPR1 in UPR1
    a4 = Mass of CS in I1
    a5 = Mass of DBP in I1
    a6 = Mass of I1 in I2
    a7 = Mass of DEI in UPR1
    a8 = Mass of UPR1 in IUPR1
    a9 = Mass of CS in I2
    a10 = Mass of I2 in IUPR1
    a11 = Mass of IUPR1
    b1 = Total Mass of DEI
    b2 = Total Mass of CS
    b3 = Total Mass of DBP
    b4 = Total Mass of UP1
    c1 = UP1-Normalised Total Mass of DEI
    c2 = UP1-Normalised Total Mass of CS
    c3 = UP1-Normalised Total Mass of DBP
    """
    def __init__(self):
        self.name = "Tailored Method20250623Dim3 DEI-based Resin Dual-Objective Optimisation Class"
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrievalMethod20250625_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()

        print("Retrieving trials.")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
        RunningTrialsX3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].name}'])

        print(f"\n===== Preliminary Setup =====")
        print(f"Pre-Heat the Oven to {OptimisationSetup_obj.CuringRegime_lis[0].get('temperature_oc_flt')}oC.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Initial Pouring of StockUPR1 =====")
        a3_g_arr = np.empty(0)
        print(f"Use a P10mL pipette set to 2 ml to transfer StockUPR1 to {len(RunningTrialsIdx_arr)} mould gaps.")
        for TrialIdx_int,RunningTrialsX1_flt,RunningTrialsX2_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,RunningTrialsX1_arr,RunningTrialsX2_arr,RunningTrialsX3_arr):
            j1_flt = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt   # Decimalised Percentage UP1 in StockUPR1
            j2_flt = OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt       # Decimalised Percentage CS in I1
            j3_flt = OptimisationSetup_obj.TargetMassOfIUPR1a11_Constant_g_flt       # Decimalised Percentage CS in I1
            x1_flt = RunningTrialsX1_flt        # Value of x1
            x2_flt = RunningTrialsX2_flt        # Value of x2
            x3_flt = RunningTrialsX3_flt        # Value of x3
            a3_flt = j3_flt                     # Mass of Stock-UPR1 in UPR1
            a1_flt = a3_flt*j1_flt              # Mass of UP1 in StockUPR1
            a2_flt = a3_flt*(1-j1_flt)          # Mass of DeI in StockUPR1
            a7_flt = (a3_flt/x1_flt)*(1-x1_flt) # Mass of DeI in UPR1 (a1_flt/x1_flt)-a3_flt 
            a8_flt = a3_flt+a7_flt              # Mass of UPR1 in IUPR1
            a10_flt = (a8_flt/x3_flt)-a8_flt    # Mass of I2 in IUPR1
            a6_flt = a10_flt*x2_flt             # Mass of I1 in I2
            a5_flt = a6_flt*(1-j2_flt)          # Mass of DBP in I1
            a4_flt = a6_flt*j2_flt              # Mass of CS is I1
            a9_flt = a10_flt*(1-x2_flt)         # Mass of CS in I2
            a11_flt = a8_flt+a10_flt            # Mass of IUPR1
            b1_flt = a5_flt                     # Total Mass of DBP
            b2_flt = a2_flt+a7_flt              # Total Mass of DeI
            b3_flt = a4_flt+a9_flt              # Total Mass of CS
            b4_flt = a1_flt                     # Total Mass of UP1
            c1_flt = b1_flt/b4_flt              # UP1-Normalised Total Mass of DeI
            c2_flt = b2_flt/b4_flt              # UP1-Normalised Total Mass of CS
            c3_flt = b3_flt/b4_flt              # UP1-Normalised Total Mass of DBP
            c1_flt = (j3_flt/a11_flt)*j3_flt    # Recommended Mass of UP1 at Beginning
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Recommended mass of Stock-UPR1: {round(c1_flt,2)} g")
            a3_g_arr = np.append(arr=a3_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))
        
        print(f"\n===== Mixing UPR1 from StockUPR1 & DeI =====")
        j1_arr = np.empty(0)
        j2_arr = np.empty(0)
        x1_arr = np.empty(0)
        x2_arr = np.empty(0)
        x3_arr = np.empty(0)
        a1_arr = np.empty(0)
        a2_arr = np.empty(0)
        a3_arr = np.empty(0)
        a4_arr = np.empty(0)
        a5_arr = np.empty(0)
        a6_arr = np.empty(0)
        a7_arr = np.empty(0)
        a8_arr = np.empty(0)
        a9_arr = np.empty(0)
        a10_arr = np.empty(0)
        a11_arr = np.empty(0)
        b1_arr = np.empty(0)
        b2_arr = np.empty(0)
        b3_arr = np.empty(0)
        b4_arr = np.empty(0)
        c1_arr = np.empty(0)
        c2_arr = np.empty(0)
        c3_arr = np.empty(0)
        for TrialIdx_int,a3_g_flt,RunningTrialsX1_flt,RunningTrialsX2_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,a3_g_arr,RunningTrialsX1_arr,RunningTrialsX2_arr,RunningTrialsX3_arr):
            j1_flt = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt   # Decimalised Percentage UP1 in StockUPR1
            j2_flt = OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt       # Decimalised Percentage CS in I1
            x1_flt = RunningTrialsX1_flt        # Value of x1
            x2_flt = RunningTrialsX2_flt        # Value of x2
            x3_flt = RunningTrialsX3_flt        # Value of x3
            a3_flt = a3_g_flt                   # Mass of Stock-UPR1 in UPR1
            a1_flt = a3_flt*j1_flt              # Mass of UP1 in StockUPR1
            a2_flt = a3_flt*(1-j1_flt)          # Mass of DeI in StockUPR1
            a7_flt = (a3_flt/x1_flt)*(1-x1_flt) # Mass of DeI in UPR1 (a1_flt/x1_flt)-a3_flt 
            a8_flt = a3_flt+a7_flt              # Mass of UPR1 in IUPR1
            a10_flt = (a8_flt/x3_flt)-a8_flt    # Mass of I2 in IUPR1
            a6_flt = a10_flt*x2_flt             # Mass of I1 in I2
            a5_flt = a6_flt*(1-j2_flt)          # Mass of DBP in I1
            a4_flt = a6_flt*j2_flt              # Mass of CS is I1
            a9_flt = a10_flt*(1-x2_flt)         # Mass of CS in I2
            a11_flt = a8_flt+a10_flt            # Mass of IUPR1
            b1_flt = a5_flt                     # Total Mass of DBP
            b2_flt = a2_flt+a7_flt              # Total Mass of DeI
            b3_flt = a4_flt+a9_flt              # Total Mass of CS
            b4_flt = a1_flt                     # Total Mass of UP1
            c1_flt = b1_flt/b4_flt              # UP1-Normalised Total Mass of DeI
            c2_flt = b2_flt/b4_flt              # UP1-Normalised Total Mass of CS
            c3_flt = b3_flt/b4_flt              # UP1-Normalised Total Mass of DBP
            j1_arr = np.append(arr=j1_arr,values=j1_flt)
            j2_arr = np.append(arr=j2_arr,values=j2_flt)
            x1_arr = np.append(arr=x1_arr,values=x1_flt)
            x2_arr = np.append(arr=x2_arr,values=x2_flt)
            x3_arr = np.append(arr=x3_arr,values=x3_flt)
            a1_arr = np.append(arr=a1_arr,values=a1_flt)
            a2_arr = np.append(arr=a2_arr,values=a2_flt)
            a3_arr = np.append(arr=a3_arr,values=a3_flt)
            a4_arr = np.append(arr=a4_arr,values=a4_flt)
            a5_arr = np.append(arr=a5_arr,values=a5_flt)
            a6_arr = np.append(arr=a6_arr,values=a6_flt)
            a7_arr = np.append(arr=a7_arr,values=a7_flt)
            a8_arr = np.append(arr=a8_arr,values=a8_flt)
            a9_arr = np.append(arr=a9_arr,values=a9_flt)
            a10_arr = np.append(arr=a10_arr,values=a10_flt)
            a11_arr = np.append(arr=a11_arr,values=a11_flt)
            b1_arr = np.append(arr=b1_arr,values=b1_flt)
            b2_arr = np.append(arr=b2_arr,values=b2_flt)
            b3_arr = np.append(arr=b3_arr,values=b3_flt)
            b4_arr = np.append(arr=b4_arr,values=b4_flt)
            c1_arr = np.append(arr=c1_arr,values=c1_flt)
            c2_arr = np.append(arr=c2_arr,values=c2_flt)
            c3_arr = np.append(arr=c3_arr,values=c3_flt)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(a7_flt,3)} g DeI")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(a7_flt,"DeI")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")
        
        print(f"\n===== Mixing I2 from CS & I1 before adding I2 to UPR1 to form IUPR1 =====")
        for TrialIdx_int,a6_flt,a9_flt in zip(RunningTrialsIdx_arr,a6_arr.tolist(),a9_arr.tolist()):
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"In a pair of weighing boats:")
            print(f"Add {round(a6_flt,3)} g I1")
            print(f"Add {round(a9_flt,3)} g CS")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Curing Regime =====")
        for counter_int,CuringRegime_dict in enumerate(OptimisationSetup_obj.CuringRegime_lis):
            print(f"Stage{counter_int+1}: {CuringRegime_dict.get('time_mins_flt')} mins @ {CuringRegime_dict.get('temperature_oc_flt')} oC")

        # Backing up the variables calculated during the trials.
        BackupVariableNames_lis = ["TrialIdx","j1","j2","x1","x2","x3","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","b1","b2","b3","b4","c1","c2","c3"]
        BackupVariablesArrays_mat = np.array([RunningTrialsIdx_arr,j1_arr,j2_arr,x1_arr,x2_arr,x3_arr,a1_arr,a2_arr,a3_arr,a4_arr,a5_arr,a6_arr,a7_arr,a8_arr,a9_arr,a10_arr,a11_arr,b1_arr,b2_arr,b3_arr,b4_arr,c1_arr,c2_arr,c3_arr])
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

class Method20250627Dim3_class():
    """
    This class governs the tailored 3D experiment that involves the
    synthesis of IUPR1. It is associated with a wider experiment which
    focusses upon optimising the curing system to be used with the
    2 stage bio-based thermosetting systems researched.
    [THIS IS AN ADAPTED FORM OF THE EXPERIMENT FOR STYRENE-BASED RESIN SYSTEM]

    It is to be carried out as follows:
    UP1 + STY -> StockUPR1      [c1]    (Carried out separate from optimisation routine)
    CS + DBP -> I1              [c2]    (Carried out by the manufacturer of the initiating powder I1)
    StockUPR1 + STY -> UPR1     [x1]    (Operator changes this during optimisation routine)
    I1 + CS -> I2               [x2]    (Operator changes this during optimisation routine)
    UPR1 + I2 -> IUPR1          [x3]    (Operator changes this during optimisation routine)

    Once the function has aided the operator in the preparation of the
    samples it makes sure to save notes about all the variables in a backup
    csv so that a human can return to any sample and recreate it at a later
    date with ease.

    Specified parameters:
    j1 = decimal percentage w/w concentration of the UP1 in StockUPR1
    j2 = decimal percentage w/w concentration of the CS in I1
    x1 = decimal percentage w/w concentration of the StockUPR1 in UPR1
    x2 = decimal percentage w/w concentration of the I1 in I2
    x3 = decimal percentage w/w concentration of the UPR1 in IUPR1
    a1 = Mass of UP1 in Stock-UPR1
    a2 = Mass of STY in Stock-UPR1
    a3 = Mass of Stock-UPR1 in UPR1
    a4 = Mass of CS in I1
    a5 = Mass of DBP in I1
    a6 = Mass of I1 in I2
    a7 = Mass of STY in UPR1
    a8 = Mass of UPR1 in IUPR1
    a9 = Mass of CS in I2
    a10 = Mass of I2 in IUPR1
    a11 = Mass of IUPR1
    b1 = Total Mass of STY
    b2 = Total Mass of CS
    b3 = Total Mass of DBP
    b4 = Total Mass of UP1
    c1 = UP1-Normalised Total Mass of STY
    c2 = UP1-Normalised Total Mass of CS
    c3 = UP1-Normalised Total Mass of DBP
    """
    def __init__(self):
        self.name = "Tailored Method20250627Dim3 Styrene-based Resin Single-Objective Optimisation Class"
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()

        print("Retrieving trials.")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
        RunningTrialsX3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].name}'])

        print(f"\n===== Preliminary Setup =====")
        print(f"Pre-Heat the Oven to {OptimisationSetup_obj.CuringRegime_lis[0].get('temperature_oc_flt')}oC.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Initial Pouring of StockUPR1 =====")
        a3_g_arr = np.empty(0)
        print(f"Use a P10mL pipette set to 2 ml to transfer StockUPR1 to {len(RunningTrialsIdx_arr)} mould gaps.")
        for TrialIdx_int,RunningTrialsX1_flt,RunningTrialsX2_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,RunningTrialsX1_arr,RunningTrialsX2_arr,RunningTrialsX3_arr):
            j1_flt = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt   # Decimalised Percentage UP1 in StockUPR1
            j2_flt = OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt       # Decimalised Percentage CS in I1
            j3_flt = OptimisationSetup_obj.TargetMassOfIUPR1a11_Constant_g_flt       # Decimalised Percentage CS in I1
            x1_flt = RunningTrialsX1_flt        # Value of x1
            x2_flt = RunningTrialsX2_flt        # Value of x2
            x3_flt = RunningTrialsX3_flt        # Value of x3
            a3_flt = j3_flt                     # Mass of Stock-UPR1 in UPR1
            a1_flt = a3_flt*j1_flt              # Mass of UP1 in StockUPR1
            a2_flt = a3_flt*(1-j1_flt)          # Mass of DeI in StockUPR1
            a7_flt = (a3_flt/x1_flt)*(1-x1_flt) # Mass of DeI in UPR1 (a1_flt/x1_flt)-a3_flt 
            a8_flt = a3_flt+a7_flt              # Mass of UPR1 in IUPR1
            a10_flt = (a8_flt/x3_flt)-a8_flt    # Mass of I2 in IUPR1
            a6_flt = a10_flt*x2_flt             # Mass of I1 in I2
            a5_flt = a6_flt*(1-j2_flt)          # Mass of DBP in I1
            a4_flt = a6_flt*j2_flt              # Mass of CS is I1
            a9_flt = a10_flt*(1-x2_flt)         # Mass of CS in I2
            a11_flt = a8_flt+a10_flt            # Mass of IUPR1
            b1_flt = a5_flt                     # Total Mass of DBP
            b2_flt = a2_flt+a7_flt              # Total Mass of DEI
            b3_flt = a4_flt+a9_flt              # Total Mass of CS
            b4_flt = a1_flt                     # Total Mass of UP1
            c1_flt = b1_flt/b4_flt              # UP1-Normalised Total Mass of DeI
            c2_flt = b2_flt/b4_flt              # UP1-Normalised Total Mass of CS
            c3_flt = b3_flt/b4_flt              # UP1-Normalised Total Mass of DBP
            c1_flt = (j3_flt/a11_flt)*j3_flt    # Recommended Mass of UP1 at Beginning
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Recommended mass of Stock-UPR1: {round(c1_flt,2)} g")
            a3_g_arr = np.append(arr=a3_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))
        
        print(f"\n===== Mixing UPR1 from StockUPR1 & STY =====")
        j1_arr = np.empty(0)
        j2_arr = np.empty(0)
        x1_arr = np.empty(0)
        x2_arr = np.empty(0)
        x3_arr = np.empty(0)
        a1_arr = np.empty(0)
        a2_arr = np.empty(0)
        a3_arr = np.empty(0)
        a4_arr = np.empty(0)
        a5_arr = np.empty(0)
        a6_arr = np.empty(0)
        a7_arr = np.empty(0)
        a8_arr = np.empty(0)
        a9_arr = np.empty(0)
        a10_arr = np.empty(0)
        a11_arr = np.empty(0)
        b1_arr = np.empty(0)
        b2_arr = np.empty(0)
        b3_arr = np.empty(0)
        b4_arr = np.empty(0)
        c1_arr = np.empty(0)
        c2_arr = np.empty(0)
        c3_arr = np.empty(0)
        for TrialIdx_int,a3_g_flt,RunningTrialsX1_flt,RunningTrialsX2_flt,RunningTrialsX3_flt in zip(RunningTrialsIdx_arr,a3_g_arr,RunningTrialsX1_arr,RunningTrialsX2_arr,RunningTrialsX3_arr):
            j1_flt = OptimisationSetup_obj.StockUPR1_UP1vsDeI_Constant_DecPct_flt   # Decimalised Percentage UP1 in StockUPR1
            j2_flt = OptimisationSetup_obj.StockI1_CSvsBP_Constant_DecPct_flt       # Decimalised Percentage CS in I1
            x1_flt = RunningTrialsX1_flt        # Value of x1
            x2_flt = RunningTrialsX2_flt        # Value of x2
            x3_flt = RunningTrialsX3_flt        # Value of x3
            a3_flt = a3_g_flt                   # Mass of Stock-UPR1 in UPR1
            a1_flt = a3_flt*j1_flt              # Mass of UP1 in StockUPR1
            a2_flt = a3_flt*(1-j1_flt)          # Mass of DeI in StockUPR1
            a7_flt = (a3_flt/x1_flt)*(1-x1_flt) # Mass of DeI in UPR1 (a1_flt/x1_flt)-a3_flt 
            a8_flt = a3_flt+a7_flt              # Mass of UPR1 in IUPR1
            a10_flt = (a8_flt/x3_flt)-a8_flt    # Mass of I2 in IUPR1
            a6_flt = a10_flt*x2_flt             # Mass of I1 in I2
            a5_flt = a6_flt*(1-j2_flt)          # Mass of DBP in I1
            a4_flt = a6_flt*j2_flt              # Mass of CS is I1
            a9_flt = a10_flt*(1-x2_flt)         # Mass of CS in I2
            a11_flt = a8_flt+a10_flt            # Mass of IUPR1
            b1_flt = a5_flt                     # Total Mass of DBP
            b2_flt = a2_flt+a7_flt              # Total Mass of DEI
            b3_flt = a4_flt+a9_flt              # Total Mass of CS
            b4_flt = a1_flt                     # Total Mass of UP1
            c1_flt = b1_flt/b4_flt              # UP1-Normalised Total Mass of DeI
            c2_flt = b2_flt/b4_flt              # UP1-Normalised Total Mass of CS
            c3_flt = b3_flt/b4_flt              # UP1-Normalised Total Mass of DBP
            j1_arr = np.append(arr=j1_arr,values=j1_flt)
            j2_arr = np.append(arr=j2_arr,values=j2_flt)
            x1_arr = np.append(arr=x1_arr,values=x1_flt)
            x2_arr = np.append(arr=x2_arr,values=x2_flt)
            x3_arr = np.append(arr=x3_arr,values=x3_flt)
            a1_arr = np.append(arr=a1_arr,values=a1_flt)
            a2_arr = np.append(arr=a2_arr,values=a2_flt)
            a3_arr = np.append(arr=a3_arr,values=a3_flt)
            a4_arr = np.append(arr=a4_arr,values=a4_flt)
            a5_arr = np.append(arr=a5_arr,values=a5_flt)
            a6_arr = np.append(arr=a6_arr,values=a6_flt)
            a7_arr = np.append(arr=a7_arr,values=a7_flt)
            a8_arr = np.append(arr=a8_arr,values=a8_flt)
            a9_arr = np.append(arr=a9_arr,values=a9_flt)
            a10_arr = np.append(arr=a10_arr,values=a10_flt)
            a11_arr = np.append(arr=a11_arr,values=a11_flt)
            b1_arr = np.append(arr=b1_arr,values=b1_flt)
            b2_arr = np.append(arr=b2_arr,values=b2_flt)
            b3_arr = np.append(arr=b3_arr,values=b3_flt)
            b4_arr = np.append(arr=b4_arr,values=b4_flt)
            c1_arr = np.append(arr=c1_arr,values=c1_flt)
            c2_arr = np.append(arr=c2_arr,values=c2_flt)
            c3_arr = np.append(arr=c3_arr,values=c3_flt)
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(a7_flt,3)} g STY")
            PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str = PipettingM_obj.MassToVolumeToSettingToStrategy_func(a7_flt,"STY")
            print(f"Use {PipetteName_str} equipped with {TipName_str}.")
            print(f"Set to {round(PipetteSetting_flt,3)} {PipetteSettingUnits_str} and make {NumberOfPipettingRounds_int} transfer(s).")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")
        
        print(f"\n===== Mixing I2 from CS & I1 before adding I2 to UPR1 to form IUPR1 =====")
        for TrialIdx_int,a6_flt,a9_flt in zip(RunningTrialsIdx_arr,a6_arr.tolist(),a9_arr.tolist()):
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"In a pair of weighing boats:")
            print(f"Add {round(a6_flt,3)} g I1")
            print(f"Add {round(a9_flt,3)} g CS")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        print(f"\n===== Curing Regime =====")
        for counter_int,CuringRegime_dict in enumerate(OptimisationSetup_obj.CuringRegime_lis):
            print(f"Stage{counter_int+1}: {CuringRegime_dict.get('time_mins_flt')} mins @ {CuringRegime_dict.get('temperature_oc_flt')} oC")

        # Backing up the variables calculated during the trials.
        BackupVariableNames_lis = ["TrialIdx","j1","j2","x1","x2","x3","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","b1","b2","b3","b4","c1","c2","c3"]
        BackupVariablesArrays_mat = np.array([RunningTrialsIdx_arr,j1_arr,j2_arr,x1_arr,x2_arr,x3_arr,a1_arr,a2_arr,a3_arr,a4_arr,a5_arr,a6_arr,a7_arr,a8_arr,a9_arr,a10_arr,a11_arr,b1_arr,b2_arr,b3_arr,b4_arr,c1_arr,c2_arr,c3_arr])
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

class Method20250817Dim4_class():
    def __init__(self):
        self.name = "Tailored Method20250817Dim4: A mixed-monomer resin. Optimisation of compressive strength by adaptation of monomer content and overall RD content."
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.YMTargetRetriever = YMTargetRetrieverMethod20250911_func
        self.PLTargetRetriever = PLTargetRetrieverMethod20250911_func
        self.PLYMTargetRetriever = PLYMTargetRetrieverMethod20250911_func
        self.YBPTargetRetriever = YBPTargetRetrieverMethod20250911_func
        self.PLYBPYMTargetRetriever = PLYBPYMTargetRetrieverMethod20251013_func
        self.PLYBPTargetRetriever = PLYBPTargetRetrieverMethod20251019_func
        self.YBPYMTargetRetriever = YBPYMTargetRetrieverMethod20251019_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        ChemicalData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + MiscMethods_obj.ChemicalDependencyFileLocation_str)
        PipettingM_obj = PipettingMethods_class()
        from SamplingMethods import CoordinateConverterFromQsToXyz
        from SamplingMethods import CoordinateConverterFromXyzToAs

        print("Retrieving trials...")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])

        RunningTrialsQ1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrialsQ2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
        RunningTrialsQ3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].name}'])
        RunningTrialsQ4_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[3].name}'])

        qs_mat = np.concat(([RunningTrialsQ2_arr],[RunningTrialsQ3_arr],[RunningTrialsQ4_arr]),axis=0)
        xyz_mat = CoordinateConverterFromQsToXyz(qs_mat)
        as_mat = CoordinateConverterFromXyzToAs(xyz_mat)
        a1_arr = as_mat[0,:]
        a2_arr = as_mat[1,:]
        a3_arr = as_mat[2,:]
        a4_arr = as_mat[3,:]

        print(f"\n===== Preliminary Setup =====")
        print(f"Pre-Heat the Oven to {OptimisationSetup_obj.CuringRegime_lis[0].get('temperature_oc_flt')}oC.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        RequiredUPR1_g_arr = np.empty(0)
        RequiredDMI2_g_arr = np.empty(0)
        for TrialIdx_int,RunningTrialsQ1_flt,RunningTrialsQ2_flt,RunningTrialsQ3_flt,RunningTrialsQ4_flt,RunningTrialsA1_flt,RunningTrialsA2_flt,RunningTrialsA3_flt,RunningTrialsA4_flt in zip(RunningTrialsIdx_arr,RunningTrialsQ1_arr,RunningTrialsQ2_arr,RunningTrialsQ3_arr,RunningTrialsQ4_arr,a1_arr,a2_arr,a3_arr,a4_arr):    
            q1_flt = RunningTrialsQ1_flt        # Ratio of UP (vs RD) in UPR2 (dec. %)
            q2_flt = RunningTrialsQ2_flt        # Abstract x-axis simplex coordinates (Tetrahedron Parameter Space)
            q3_flt = RunningTrialsQ3_flt        # Abstract y-axis simplex coordinates (Tetrahedron Parameter Space)
            q4_flt = RunningTrialsQ4_flt        # Abstract z-axis simplex coordinates (Tetrahedron Parameter Space)

            a1_flt = RunningTrialsA1_flt        # Mole fraction value for DMI (vs DEI,DPI,MM) in RD (#)
            a2_flt = RunningTrialsA2_flt        # Mole fraction value for DEI (vs DMI,DPI,MM) in RD (#)
            a3_flt = RunningTrialsA3_flt        # Mole fraction value for DPI (vs DMI,DEI,MM) in RD (#)
            a4_flt = RunningTrialsA4_flt        # Mole fraction value for MM (vs DMI,DEI,DPI) in RD (#)

            j1_flt = OptimisationSetup_obj.j1_IUPR_UPR1vsI1_DecPct_flt                  # Percentage UPR2 (vs I) in IUPR1 (dec. %)
            j2_flt = OptimisationSetup_obj.j2_UPR1_UPvsRD1_DecPct_flt                   # Percentage UP1 (vs RD1) in UPR1 (dec. %)
            j3_flt = OptimisationSetup_obj.j3_I1_CSvsDBP_DecPct_flt                     # Percentage CS (vs DBP) in I (dec. %)
            j4_flt = OptimisationSetup_obj.j4_RD1_DMI1vsDEI1vsDPI1vsMM1_Stoich_flt      # Mole fraction value for DMI1 (vs DEI1,DPI1,MM1) in RD1 (#)
            j5_flt = OptimisationSetup_obj.j5_RD1_DEI1vsDMI1vsDPI1vsMM1_Stoich_flt      # Mole fraction value for DEI1 (vs DMI1,DPI1,MM1) in RD1 (#)
            j6_flt = OptimisationSetup_obj.j6_RD1_DPI1vsDMI1vsDEI1vsMM1_Stoich_flt      # Mole fraction value for DPI1 (vs DMI1,DEI1,MM1) in RD1 (#)
            j7_flt = OptimisationSetup_obj.j7_RD1_MM1vsDMI1vsDEI1vsDPI1_Stoich_flt      # Mole fraction value for MM1 (vs DMI1,DEI1,DPI1) in RD1 (#)
            j8_flt = OptimisationSetup_obj.j8_IUPR_TargetMass_flt                       # Target mass of IUPR (g)
            j9_flt = OptimisationSetup_obj.j9_UPR1_InitGuessMass_flt                    # Guesstimated mass of UPR1 (g)

            m1_flt = ChemicalData_dict['chemicals']['DmI']['mr_flt']    # Molecular mass of DMI (g mol^-1)
            m2_flt = ChemicalData_dict['chemicals']['DeI']['mr_flt']    # Molecular mass of DEI (g mol^-1)
            m3_flt = ChemicalData_dict['chemicals']['DpI']['mr_flt']    # Molecular mass of DPI (g mol^-1)
            m4_flt = ChemicalData_dict['chemicals']['MM']['mr_flt']     # Molecular mass of MM (g mol^-1)

            b1_flt = j9_flt                                                             # Mass of UPR1 (g)
            b2_flt = j2_flt*b1_flt                                                      # Mass of UP (vs RD1) in UPR1 (g)
            b3_flt = (1-j2_flt)*b1_flt                                                  # Mass of RD1 (vs UP) in UPR1 (g)
            b4_flt = (j4_flt*m1_flt)+(j5_flt*m2_flt)+(j6_flt*m3_flt)+(j7_flt*m4_flt)    # Average molar mass of RD1 mixture (g mol^-1)
            b5_flt = b3_flt/b4_flt                                                      # Total moles in the mixture RD1 (moles)
            b6_flt = m1_flt*(j4_flt*b5_flt)                                             # Mass of DMI1 (vs DEI1,DPI1,MM1) in RD1 (g)
            b7_flt = m2_flt*(j5_flt*b5_flt)                                             # Mass of DEI1 (vs DMI1,DPI1,MM1) in RD1 (g)
            b8_flt = m3_flt*(j6_flt*b5_flt)                                             # Mass of DPI1 (vs DMI1,DEI1,MM1) in RD1 (g)
            b9_flt = m4_flt*(j7_flt*b5_flt)                                             # Mass of MM1 (vs DMI1,DEI1,DPI1) in RD1 (g)
            b10_flt = (b2_flt/q1_flt)*(1-q1_flt)                                        # Mass of RD (vs UP) (RD=RD1+RD2) in UPR2 (g)
            b11_flt = b10_flt-b3_flt                                                    # Mass of RD2 (vs UPR1) in UPR2 (g)
            b12_flt = (a1_flt*m1_flt)+(a2_flt*m2_flt)+(a3_flt*m3_flt)+(a4_flt*m4_flt)   # Average molar mass of RD2 mixture (g mol^-1)
            b13_flt = b11_flt/b12_flt                                                   # Total moles in mixture RD2 (moles)
            b14_flt = m1_flt*(a1_flt*b13_flt)                                           # Mass of DMI2 (vs DEI2,DPI2,MM2) in RD2 (g)
            b15_flt = m2_flt*(a2_flt*b13_flt)                                           # Mass of DEI2 (vs DMI2,DPI2,MM2) in RD2 (g)
            b16_flt = m3_flt*(a3_flt*b13_flt)                                           # Mass of DPI2 (vs DMI2,DEI2,MM2) in RD2 (g)
            b17_flt = m4_flt*(a4_flt*b13_flt)                                           # Mass of MM2 (vs DMI2,DEI2,DPI2) in RD2 (g)
            b18_flt = b6_flt+b14_flt                                                    # Mass of DMI (vs DEI,DPI,MM) in RD (g)
            b19_flt = b7_flt+b15_flt                                                    # Mass of DEI (vs DMI,DPI,MM) in RD (g)
            b20_flt = b8_flt+b16_flt                                                    # Mass of DPI (vs DMI,DEI,MM) in RD (g)
            b21_flt = b9_flt+b17_flt                                                    # Mass of MM (vs DMI,DEI,DPI) in RD (g)
            b22_flt = b10_flt+b2_flt                                                    # Mass of UPR2 (vs I) in IUPR (g)
            b23_flt = (b22_flt/j1_flt)*(1-j1_flt)                                       # Mass of I (vs UPR2) in IUPR (g)
            b24_flt = (1-j3_flt)*b23_flt                                                # Mass of DBP (vs CS) in I (g)
            b25_flt = j3_flt*b23_flt                                                    # Mass of CS (vs DBP) in I (g)
            b26_flt = b22_flt+b23_flt                                                   # Mass of IUPR (g)
            b27_flt = (j8_flt/b26_flt)*j9_flt                                           # Suggested mass of UPR1 to achieve target mass of IUPR (g) (uses b1)

            RequiredUPR1_g_arr = np.append(RequiredUPR1_g_arr,b27_flt)
            RequiredDMI2_g_arr = np.append(RequiredDMI2_g_arr,b14_flt)
        
        RequiredDMI2_g_float = np.sum(RequiredDMI2_g_arr)

        print(f"Place {round(RequiredDMI2_g_float,2)} g of DMI into a heated vessel to melt.")
        MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")
        
        print(f"\n===== Initial Pouring of UPR1 =====")
        b1_g_arr = np.empty(0)
        for TrialIdx_int,TargetMassOfUPR1_g_float in zip(RunningTrialsIdx_arr,RequiredUPR1_g_arr):
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Use a pasteur pipette to transfer around {round(TargetMassOfUPR1_g_float,2)} g UPR1 to the silicone mould gap.")
            b1_g_arr = np.append(arr=b1_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))

        print(f"\n===== Mixing UPR2 from UPR1 & RD2 =====")
        BackupVariableNames_lis = ["TrialIdx","q1","q2","q3","q4","a1","a2","a3","a4","j1","j2","j3","j4","j5","j6","j7","j8","j9","m1","m2","m3","m4","b1","b2","b3","b4","b5","b6","b7","b8","b9","b10","b11","b12","b13","b14","b15","b16","b17","b18","b19","b20","b21","b22","b23","b24","b25","b26","b27"]
        
        BackupVariablesMatrix_lis = []
        for i in BackupVariableNames_lis:
            BackupVariablesMatrix_lis.append(np.empty(0))

        for TrialIdx_int,b1_g_flt,RunningTrialsQ1_flt,RunningTrialsQ2_flt,RunningTrialsQ3_flt,RunningTrialsQ4_flt,RunningTrialsA1_flt,RunningTrialsA2_flt,RunningTrialsA3_flt,RunningTrialsA4_flt in zip(RunningTrialsIdx_arr,b1_g_arr,RunningTrialsQ1_arr,RunningTrialsQ2_arr,RunningTrialsQ3_arr,RunningTrialsQ4_arr,a1_arr,a2_arr,a3_arr,a4_arr):
            q1_flt = RunningTrialsQ1_flt        # Ratio of UP (vs RD) in UPR2 (dec. %)
            q2_flt = RunningTrialsQ2_flt        # Abstract x-axis simplex coordinates (Tetrahedron Parameter Space)
            q3_flt = RunningTrialsQ3_flt        # Abstract y-axis simplex coordinates (Tetrahedron Parameter Space)
            q4_flt = RunningTrialsQ4_flt        # Abstract z-axis simplex coordinates (Tetrahedron Parameter Space)
            q_vals_lis = [q1_flt,q2_flt,q3_flt,q4_flt]

            a1_flt = RunningTrialsA1_flt        # Mole fraction value for DMI (vs DEI,DPI,MM) in RD (#)
            a2_flt = RunningTrialsA2_flt        # Mole fraction value for DEI (vs DMI,DPI,MM) in RD (#)
            a3_flt = RunningTrialsA3_flt        # Mole fraction value for DPI (vs DMI,DEI,MM) in RD (#)
            a4_flt = RunningTrialsA4_flt        # Mole fraction value for MM (vs DMI,DEI,DPI) in RD (#)
            a_vals_lis = [a1_flt,a2_flt,a3_flt,a4_flt]

            j1_flt = OptimisationSetup_obj.j1_IUPR_UPR1vsI1_DecPct_flt                  # Percentage UPR2 (vs I) in IUPR1 (dec. %)
            j2_flt = OptimisationSetup_obj.j2_UPR1_UPvsRD1_DecPct_flt                   # Percentage UP1 (vs RD1) in UPR1 (dec. %)
            j3_flt = OptimisationSetup_obj.j3_I1_CSvsDBP_DecPct_flt                     # Percentage CS (vs DBP) in I (dec. %)
            j4_flt = OptimisationSetup_obj.j4_RD1_DMI1vsDEI1vsDPI1vsMM1_Stoich_flt      # Mole fraction value for DMI1 (vs DEI1,DPI1,MM1) in RD1 (#)
            j5_flt = OptimisationSetup_obj.j5_RD1_DEI1vsDMI1vsDPI1vsMM1_Stoich_flt      # Mole fraction value for DEI1 (vs DMI1,DPI1,MM1) in RD1 (#)
            j6_flt = OptimisationSetup_obj.j6_RD1_DPI1vsDMI1vsDEI1vsMM1_Stoich_flt      # Mole fraction value for DPI1 (vs DMI1,DEI1,MM1) in RD1 (#)
            j7_flt = OptimisationSetup_obj.j7_RD1_MM1vsDMI1vsDEI1vsDPI1_Stoich_flt      # Mole fraction value for MM1 (vs DMI1,DEI1,DPI1) in RD1 (#)
            j8_flt = OptimisationSetup_obj.j8_IUPR_TargetMass_flt                       # Target mass of IUPR (g)
            j9_flt = OptimisationSetup_obj.j9_UPR1_InitGuessMass_flt                    # Guesstimated mass of UPR1 (g)
            j_vals_lis = [j1_flt,j2_flt,j3_flt,j4_flt,j5_flt,j6_flt,j7_flt,j8_flt,j9_flt]

            m1_flt = ChemicalData_dict['chemicals']['DmI']['mr_flt']    # Molecular mass of DMI (g mol^-1)
            m2_flt = ChemicalData_dict['chemicals']['DeI']['mr_flt']    # Molecular mass of DEI (g mol^-1)
            m3_flt = ChemicalData_dict['chemicals']['DpI']['mr_flt']    # Molecular mass of DPI (g mol^-1)
            m4_flt = ChemicalData_dict['chemicals']['MM']['mr_flt']     # Molecular mass of MM (g mol^-1)
            m_vals_lis = [m1_flt,m2_flt,m3_flt,m4_flt]

            b1_flt = b1_g_flt                                                           # Mass of UPR1 (g)
            b2_flt = j2_flt*b1_flt                                                      # Mass of UP (vs RD1) in UPR1 (g)
            b3_flt = (1-j2_flt)*b1_flt                                                  # Mass of RD1 (vs UP) in UPR1 (g)
            b4_flt = (j4_flt*m1_flt)+(j5_flt*m2_flt)+(j6_flt*m3_flt)+(j7_flt*m4_flt)    # Average molar mass of RD1 mixture (g mol^-1)
            b5_flt = b3_flt/b4_flt                                                      # Total moles in the mixture RD1 (moles)
            b6_flt = m1_flt*(j4_flt*b5_flt)                                             # Mass of DMI1 (vs DEI1,DPI1,MM1) in RD1 (g)
            b7_flt = m2_flt*(j5_flt*b5_flt)                                             # Mass of DEI1 (vs DMI1,DPI1,MM1) in RD1 (g)
            b8_flt = m3_flt*(j6_flt*b5_flt)                                             # Mass of DPI1 (vs DMI1,DEI1,MM1) in RD1 (g)
            b9_flt = m4_flt*(j7_flt*b5_flt)                                             # Mass of MM1 (vs DMI1,DEI1,DPI1) in RD1 (g)
            b10_flt = (b2_flt/q1_flt)*(1-q1_flt)                                        # Mass of RD (vs UP) (RD=RD1+RD2) in UPR2 (g)
            b11_flt = b10_flt-b3_flt                                                     # Mass of RD2 (vs UPR1) in UPR2 (g)
            b12_flt = (a1_flt*m1_flt)+(a2_flt*m2_flt)+(a3_flt*m3_flt)+(a4_flt*m4_flt)   # Average molar mass of RD2 mixture (g mol^-1)
            b13_flt = b11_flt/b12_flt                                                   # Total moles in mixture RD2 (moles)
            b14_flt = m1_flt*(a1_flt*b13_flt)                                           # Mass of DMI2 (vs DEI2,DPI2,MM2) in RD2 (g)
            b15_flt = m2_flt*(a2_flt*b13_flt)                                           # Mass of DEI2 (vs DMI2,DPI2,MM2) in RD2 (g)
            b16_flt = m3_flt*(a3_flt*b13_flt)                                           # Mass of DPI2 (vs DMI2,DEI2,MM2) in RD2 (g)
            b17_flt = m4_flt*(a4_flt*b13_flt)                                           # Mass of MM2 (vs DMI2,DEI2,DPI2) in RD2 (g)
            b18_flt = b6_flt+b14_flt                                                    # Mass of DMI (vs DEI,DPI,MM) in RD (g)
            b19_flt = b7_flt+b15_flt                                                    # Mass of DEI (vs DMI,DPI,MM) in RD (g)
            b20_flt = b8_flt+b16_flt                                                    # Mass of DPI (vs DMI,DEI,MM) in RD (g)
            b21_flt = b9_flt+b17_flt                                                    # Mass of MM (vs DMI,DEI,DPI) in RD (g)
            b22_flt = b10_flt+b2_flt                                                    # Mass of UPR2 (vs I) in IUPR (g)
            b23_flt = (b22_flt/j1_flt)*(1-j1_flt)                                       # Mass of I (vs UPR2) in IUPR (g)
            b24_flt = (1-j3_flt)*b23_flt                                                # Mass of DBP (vs CS) in I (g)
            b25_flt = j3_flt*b23_flt                                                    # Mass of CS (vs DBP) in I (g)
            b26_flt = b22_flt+b23_flt                                                   # Mass of IUPR (g)
            b27_flt = (j8_flt/b26_flt)*j9_flt                                           # Suggested mass of UPR1 to achieve target mass of IUPR (g) (uses b1)
            b_vals_lis = [b1_flt,b2_flt,b3_flt,b4_flt,b5_flt,b6_flt,b7_flt,b8_flt,b9_flt,b10_flt,b11_flt,b12_flt,b13_flt,b14_flt,b15_flt,b16_flt,b17_flt,b18_flt,b19_flt,b20_flt,b21_flt,b22_flt,b23_flt,b24_flt,b25_flt,b26_flt,b27_flt]

            var_vals_lis = [TrialIdx_int]+q_vals_lis+a_vals_lis+j_vals_lis+m_vals_lis+b_vals_lis
            for count,(val,BackupVariable_arr) in enumerate(zip(var_vals_lis,BackupVariablesMatrix_lis)):
                BackupVariablesMatrix_lis[count] = np.append(BackupVariable_arr,val)
            
            MiscMethods_obj = MiscMethods_class()
            PipettingMethods_obj = PipettingMethods_class()
            PipetteData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + PipettingMethods_obj.DependencyFileLocation_str)
            ChemicalData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + MiscMethods_obj.ChemicalDependencyFileLocation_str)
            pipette_lis = ["P200","P20"]
            PipetteData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + PipettingMethods_obj.DependencyFileLocation_str)

            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Add {round(b14_flt,3)} g DMI")
            SubstanceName_str = "DmI"
            SubstanceInfo_dict = ChemicalData_dict["chemicals"][SubstanceName_str]
            SubstanceTemperature_flt = 40.0
            print(f"Use either:")
            for pipette_str in pipette_lis:
                print(f"> {pipette_str} equipped with {PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict)}")
                CalibrationDataLocation_str = PipettingMethods_obj.CalibrationDataAvailabilityChecker_func(SubstanceInfo_dict=SubstanceInfo_dict,PipetteCalibrationData_dict=PipetteData_dict,PipetteName_str=pipette_str,TipName_str=PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict),Temperature_oc_flt=SubstanceTemperature_flt,PackageLocation_str=MiscMethods_obj.RootPackageLocation_str,PipetteDependenciesLocation_str=PipettingMethods_obj.DependencyFolderLocation_str)
                CalibrationStraightLineEquationParameters_arr = PipettingMethods_obj.CalibrationEquationGenerator_func(CalibrationDataLocation_str)
                Pipettings_int,Setting_flt = PipettingMethods_obj.PipettingStrategyElucidator(b14_flt,CalibrationStraightLineEquationParameters_arr,CalibrationDataLocation_str)
                print(f"Set to {round(Setting_flt,3)} {PipettingMethods_obj.UnitRetriever_func(pipette_str,PipetteData_dict)} and make {Pipettings_int} transfer(s) at {SubstanceTemperature_flt}oC.")
            print()

            print(f"Add {round(b15_flt,3)} g DEI")
            SubstanceName_str = "DeI"
            SubstanceInfo_dict = ChemicalData_dict["chemicals"][SubstanceName_str]
            SubstanceTemperature_flt = 3.0
            print(f"Use either:")
            for pipette_str in pipette_lis:
                print(f"> {pipette_str} equipped with {PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict)}")
                CalibrationDataLocation_str = PipettingMethods_obj.CalibrationDataAvailabilityChecker_func(SubstanceInfo_dict=SubstanceInfo_dict,PipetteCalibrationData_dict=PipetteData_dict,PipetteName_str=pipette_str,TipName_str=PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict),Temperature_oc_flt=SubstanceTemperature_flt,PackageLocation_str=MiscMethods_obj.RootPackageLocation_str,PipetteDependenciesLocation_str=PipettingMethods_obj.DependencyFolderLocation_str)
                CalibrationStraightLineEquationParameters_arr = PipettingMethods_obj.CalibrationEquationGenerator_func(CalibrationDataLocation_str)
                Pipettings_int,Setting_flt = PipettingMethods_obj.PipettingStrategyElucidator(b15_flt,CalibrationStraightLineEquationParameters_arr,CalibrationDataLocation_str)
                print(f"Set to {round(Setting_flt,3)} {PipettingMethods_obj.UnitRetriever_func(pipette_str,PipetteData_dict)} and make {Pipettings_int} transfer(s) at {SubstanceTemperature_flt}oC.")
            print()

            print(f"Add {round(b16_flt,3)} g DPI")
            SubstanceName_str = "DpI"
            SubstanceInfo_dict = ChemicalData_dict["chemicals"][SubstanceName_str]
            SubstanceTemperature_flt = 3.0
            print(f"Use either:")
            for pipette_str in pipette_lis:
                print(f"> {pipette_str} equipped with {PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict)}")
                CalibrationDataLocation_str = PipettingMethods_obj.CalibrationDataAvailabilityChecker_func(SubstanceInfo_dict=SubstanceInfo_dict,PipetteCalibrationData_dict=PipetteData_dict,PipetteName_str=pipette_str,TipName_str=PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict),Temperature_oc_flt=SubstanceTemperature_flt,PackageLocation_str=MiscMethods_obj.RootPackageLocation_str,PipetteDependenciesLocation_str=PipettingMethods_obj.DependencyFolderLocation_str)
                CalibrationStraightLineEquationParameters_arr = PipettingMethods_obj.CalibrationEquationGenerator_func(CalibrationDataLocation_str)
                Pipettings_int,Setting_flt = PipettingMethods_obj.PipettingStrategyElucidator(b16_flt,CalibrationStraightLineEquationParameters_arr,CalibrationDataLocation_str)
                print(f"Set to {round(Setting_flt,3)} {PipettingMethods_obj.UnitRetriever_func(pipette_str,PipetteData_dict)} and make {Pipettings_int} transfer(s) at {SubstanceTemperature_flt}oC.")
            print()

            print(f"Add {round(b17_flt,3)} g MM")
            SubstanceName_str = "MM"
            SubstanceInfo_dict = ChemicalData_dict["chemicals"][SubstanceName_str]
            SubstanceTemperature_flt = 3.0
            print(f"Use either:")
            for pipette_str in pipette_lis:
                print(f"> {pipette_str} equipped with {PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict)}")
                CalibrationDataLocation_str = PipettingMethods_obj.CalibrationDataAvailabilityChecker_func(SubstanceInfo_dict=SubstanceInfo_dict,PipetteCalibrationData_dict=PipetteData_dict,PipetteName_str=pipette_str,TipName_str=PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict),Temperature_oc_flt=SubstanceTemperature_flt,PackageLocation_str=MiscMethods_obj.RootPackageLocation_str,PipetteDependenciesLocation_str=PipettingMethods_obj.DependencyFolderLocation_str)
                CalibrationStraightLineEquationParameters_arr = PipettingMethods_obj.CalibrationEquationGenerator_func(CalibrationDataLocation_str)
                Pipettings_int,Setting_flt = PipettingMethods_obj.PipettingStrategyElucidator(b17_flt,CalibrationStraightLineEquationParameters_arr,CalibrationDataLocation_str)
                print(f"Set to {round(Setting_flt,3)} {PipettingMethods_obj.UnitRetriever_func(pipette_str,PipetteData_dict)} and make {Pipettings_int} transfer(s) at {SubstanceTemperature_flt}oC.")

            print(f"\n===== Weighing out I to be mixed with UPR2 to form IUPR =====")
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"In a pair of weighing boats:")
            print(f"Add {round(b23_flt,3)} g I")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        # Backing up the variables calculated during the trials.
        BackupVariablesArrays_mat = np.array(BackupVariablesMatrix_lis)
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

def TargetRetrieverMethod20251120_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == (((RunningTrialsIdx_arr[-1]+1)*3)-1):
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t1_lis = []
            t2_lis = []
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                t1_lis.append(PreviousTrials_df["x2"][RunningTrialIdx_int])
                t2_avg_lis = []
                for MechDataIdx_int in list(range(((RunningTrialIdx_int+1)*3)-3,((RunningTrialIdx_int+1)*3))):
                    ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                    StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                    ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                    DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == MechDataIdx_int, 'DiameterAvg_mm']).iloc[0])
                    HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == MechDataIdx_int, 'HeightAvg_mm']).iloc[0])
                    CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(MechDataIdx_int) + ".csv"
                    corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                    cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                    corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                    ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                    StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                    SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                    DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                    PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                    LimitOfProportionality_flt = CSTMethods_obj.LimitOfProportionality_func(StressStrain_mat,PeakStrain_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                    YieldBreakPoint_flt = CSTMethods_obj.YieldBreakPoint_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,StressStrain_mat,ArbitraryGradientCutoff=ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ArbitrarySustainedRise_int=ArbitrarySustainedRise_int,ChallengePlotAcceptability_bool=True)
                    t2_avg_lis.append(YieldBreakPoint_flt)
                t2_avg = np.average(np.array(t2_avg_lis))
                t2_lis.append(t2_avg)
            t_lis = []
            for t1,t2 in zip(t1_lis,t2_lis):
                t_lis.append([t1,t2])
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def TargetRetrieverMethod20251121_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_lis = []
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False) # True
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                print(PeakStrain_flt)
                if PeakStrain_flt == 6969696.69:
                    print("time to work out the rubber system")
                    YieldBreakPoint_flt = CSTMethods_obj.YieldBreakForUnusualSamples_func(DerivativeStressStrain_mat,SmoothedStressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                else:
                    print("normal conditions")
                    YieldBreakPoint_flt = CSTMethods_obj.YieldBreakPoint_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,StressStrain_mat,ArbitraryGradientCutoff=ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ArbitrarySustainedRise_int=ArbitrarySustainedRise_int,ChallengePlotAcceptability_bool=True) # True
                print(f"The yield break point was : {YieldBreakPoint_flt}")
                t1_flt = np.array(PreviousTrials_df["x1"])[RunningTrialIdx_int]
                t2_flt = YieldBreakPoint_flt
                t_lis.append([t1_flt,t2_flt])
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

def ExecutionCheckerMethod20251120_func(client_obj,OptimisationSetup_obj):
    """
    This function checks whether the mixing method has been executed so that
    other parts of the program know when it is time to complete the trials.
    The first try except clause looks at whether there is a backup csv file
    and additionally the highest index values of this file and the Ax experiment
    dataframe to ascertain whether execution has taken place. If it has, it carries
    out the except clause.
    In the except clause a second check takes place- the running trials indices are
    cross-compared with the numbers of the files in the mechanical testing folder
    if these are equal then data is available for consideration.
    """
    checker_bool = False
    try:
        PreviousTrials_df = client_obj.summarize()
        RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        if RunningTrials_df.empty:
            print("There are no trials running! Therefore, none can be completed.")
        else:
            print("There are trials running!")
            my_file = Path(OptimisationSetup_obj.BackupPath_str)
            if my_file.is_file():
                print("Backup file available!")
                Backup_df = pd.read_csv(OptimisationSetup_obj.BackupPath_str)
                BackupIdx_arr = np.array(Backup_df["TrialIdx"])
                if RunningTrialsIdx_arr[-1] == BackupIdx_arr[-1]:
                    raise ValueError('Trials executed but uncompleted.')
                else:
                    print("Trials not executed so cannot be completed.")
            else:
                print("No backup file available!")
    except:
        print("There are trials that can be completed.")
        try:
            dir_lis = os.listdir(OptimisationSetup_obj.RawDataMT_str)
        except:
            print("Couldn't find a mechanical test data directory nor any mechanical test data, creating one, please place mechanical test data here.")
            os.mkdir(OptimisationSetup_obj.RawDataMT_str)
        dir_lis = os.listdir(OptimisationSetup_obj.RawDataMT_str)
        if dir_lis == []:
            print("No Files in Mechanical Testing Directory...")
        else:
            dir_lis = sorted([int(dir_str.rstrip('.csv')) for dir_str in dir_lis])
            if dir_lis[-1] == ((BackupIdx_arr.tolist()[-1]+1)*3)-1:
                print("Mechanical Test Data Available - trials can be completed.")
                checker_bool = True
            else:
                print("Mechanical test data unavailable - please place in folder.")
    return checker_bool
 
class Method20251120Dim2_class():
    def __init__(self):
        self.name = "Tailored Method20251120Dim2: A 20% DMI-based resin. Dual-objective optimisation of both compressive strength and oven time."
        # self.ExecutionChecker = ExecutionCheckerMethod20251120_func
        # self.TargetRetriever = TargetRetrieverMethod20251120_func
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrieverMethod20251121_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        ChemicalData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + MiscMethods_obj.ChemicalDependencyFileLocation_str)
        PipettingMethods_obj = PipettingMethods_class()
        PipetteData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + PipettingMethods_obj.DependencyFileLocation_str)
        pipette_lis = ["P200","P20"]

        print("Retrieving trials...")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])

        RunningTrials_x1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrials_x2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])

        print(f"\n===== Initial Pouring of UPR1 =====")
        a1_g_arr = np.empty(0)
        TargetMassOfUPR1_g_float = (OptimisationSetup_obj.j5_IUPR_TargetMass_flt*OptimisationSetup_obj.j1_IUPR_UPR2vsI1_DecPct_flt)*OptimisationSetup_obj.j3_UPR2_UPR1vsRD2_DecPct_flt

        for TrialIdx_int in RunningTrialsIdx_arr:
            print(f"\n-----Trial {TrialIdx_int}-----")
            print(f"Use a pasteur pipette to transfer around {round(TargetMassOfUPR1_g_float,2)} g UPR1 to the silicone mould gap.")
            a1_g_arr = np.append(arr=a1_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))

        print(f"\n===== Synthesis and Oven Time =====")
        BackupVariableNames_lis = ["TrialIdx","x1","x2","j1","j2","j3","j4","j5","a1","a2","a3","a4","a5","a6","a7","a8"]
        BackupVariablesMatrix_lis = []
        for i in BackupVariableNames_lis:
            BackupVariablesMatrix_lis.append(np.empty(0))

        for TrialIdx_int,a1_g_flt,RunningTrials_x1_flt,RunningTrials_x2_flt in zip(RunningTrialsIdx_arr,a1_g_arr,RunningTrials_x1_arr,RunningTrials_x2_arr):
            print(f"\n-----Trial {TrialIdx_int}-----")
            OvenTemperatureSetting,HeraeusTemperatureSetting = MiscMethods_obj.ActualTemperatureToOvenSetting(RunningTrials_x2_flt)
            print(f"Temperature Target: {round(RunningTrials_x2_flt,1)}oC")
            print(f"Pre-Heat the Oven to {round(OvenTemperatureSetting,1)}oC.")
            print(f"Pre-Heat the Heraeus to {round(HeraeusTemperatureSetting,1)}oC.")
            print()

            x1_flt = RunningTrials_x1_flt       # Oven time
            x2_flt = RunningTrials_x2_flt       # Oven temperature
            x_vals_lis = [x1_flt,x2_flt]

            j1_flt = OptimisationSetup_obj.j1_IUPR_UPR2vsI1_DecPct_flt
            j2_flt = OptimisationSetup_obj.j2_UPR1_UPvsRD1_DecPct_flt
            j3_flt = OptimisationSetup_obj.j3_UPR2_UPR1vsRD2_DecPct_flt
            j4_flt = OptimisationSetup_obj.j4_I1_CSvsDBP_DecPct_flt
            j5_flt = OptimisationSetup_obj.j5_IUPR_TargetMass_flt
            j_vals_lis = [j1_flt,j2_flt,j3_flt,j4_flt,j5_flt]

            a1_flt = a1_g_flt                       # Mass of UPR1 (g)
            a2_flt = a1_flt*j2_flt                  # Mass of UP (vs RD1) in UPR1 (g)
            a3_flt = a1_flt*(1-j2_flt)              # Mass of RD1 (vs UP) in UPR1 (g)
            a4_flt = (a3_flt/j3_flt)*(1-j3_flt)     # Mass of RD2 (vs UPR1) in UPR2 (g)     [ADD THIS]
            a5_flt = a4_flt+a1_flt                  # Mass of UPR2 (g)
            a6_flt = (a5_flt/j1_flt)*(1-j1_flt)     # Mass of I1 (vs UPR2) in IUPR (g)      [ADD THIS]
            a7_flt = a6_flt*j4_flt                  # Mass of CS (vs DBP) in I1 (g)
            a8_flt = a6_flt*(1-j4_flt)              # Mass of DBP (vs CS) in I1 (g)
            a_vals_lis = [a1_flt,a2_flt,a3_flt,a4_flt,a5_flt,a6_flt,a7_flt,a8_flt]

            var_vals_lis = [TrialIdx_int]+x_vals_lis+j_vals_lis+a_vals_lis
            for count,(val,BackupVariable_arr) in enumerate(zip(var_vals_lis,BackupVariablesMatrix_lis)):
                BackupVariablesMatrix_lis[count] = np.append(BackupVariable_arr,val)

            print(f"Add {round(a4_flt,3)} g DMI")
            SubstanceName_str = "DmI"
            SubstanceInfo_dict = ChemicalData_dict["chemicals"][SubstanceName_str]
            SubstanceTemperature_flt = 40.0
            print(f"Use either:")
            for pipette_str in pipette_lis:
                print(f"> {pipette_str} equipped with {PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict)}")
                CalibrationDataLocation_str = PipettingMethods_obj.CalibrationDataAvailabilityChecker_func(SubstanceInfo_dict=SubstanceInfo_dict,PipetteCalibrationData_dict=PipetteData_dict,PipetteName_str=pipette_str,TipName_str=PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict),Temperature_oc_flt=SubstanceTemperature_flt,PackageLocation_str=MiscMethods_obj.RootPackageLocation_str,PipetteDependenciesLocation_str=PipettingMethods_obj.DependencyFolderLocation_str)
                CalibrationStraightLineEquationParameters_arr = PipettingMethods_obj.CalibrationEquationGenerator_func(CalibrationDataLocation_str)
                Pipettings_int,Setting_flt = PipettingMethods_obj.PipettingStrategyElucidator(a4_flt,CalibrationStraightLineEquationParameters_arr,CalibrationDataLocation_str)
                print(f"Set to {round(Setting_flt,3)} {PipettingMethods_obj.UnitRetriever_func(pipette_str,PipetteData_dict)} and make {Pipettings_int} transfer(s) at {SubstanceTemperature_flt}oC.")
            print()
            print(f"In a weighing boat:")
            print(f"Add {round(a6_flt,3)} g I1")
            print()
            Hours_int,Minutes_int,Seconds_int = MiscMethods_obj.SecondsToHoursMinutesSeconds(RunningTrials_x1_flt)
            print(f"Heat sample for {Hours_int} hours, {Minutes_int} minutes, {Seconds_int} seconds.")
            MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        # Backing up the variables calculated during the trials.
        BackupVariablesArrays_mat = np.array(BackupVariablesMatrix_lis)
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

class Method20251123Dim3_class():
    def __init__(self):
        self.name = "Tailored Method20251123Dim3: Dual-objective optimisation of both compressive strength and oven time."
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrieverMethod20251121_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        ChemicalData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + MiscMethods_obj.ChemicalDependencyFileLocation_str)
        PipettingMethods_obj = PipettingMethods_class()
        PipetteData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + PipettingMethods_obj.DependencyFileLocation_str)
        pipette_lis = ["P200","P20"]

        print("Retrieving trials...")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])

        RunningTrials_x1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrials_x2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
        RunningTrials_x3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].name}'])

        print(f"\n===== Initial Pouring of UPR1 =====")
        a1_g_arr = np.empty(0)

        for TrialIdx_int,x3 in zip(RunningTrialsIdx_arr,RunningTrials_x3_arr):
            print(f"\n-----Trial {TrialIdx_int}-----")
            TargetMassOfUPR1_g_float = (OptimisationSetup_obj.j4_IUPR_TargetMass_flt*OptimisationSetup_obj.j1_IUPR_UPR2vsI1_DecPct_flt)*x3
            print(f"Use a pasteur pipette to transfer around {round(TargetMassOfUPR1_g_float,2)} g UPR1 to the silicone mould gap.")
            a1_g_arr = np.append(arr=a1_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))

        print(f"\n===== Synthesis and Oven Time =====")
        BackupVariableNames_lis = ["TrialIdx","x1","x2","x3","j1","j2","j3","j4","a1","a2","a3","a4","a5","a6","a7","a8"]
        BackupVariablesMatrix_lis = []
        for i in BackupVariableNames_lis:
            BackupVariablesMatrix_lis.append(np.empty(0))

        for TrialIdx_int,a1_g_flt,RunningTrials_x1_flt,RunningTrials_x2_flt,RunningTrials_x3_flt in zip(RunningTrialsIdx_arr,a1_g_arr,RunningTrials_x1_arr,RunningTrials_x2_arr,RunningTrials_x3_arr):
            print(f"\n-----Trial {TrialIdx_int}-----")
            OvenTemperatureSetting,HeraeusTemperatureSetting = MiscMethods_obj.ActualTemperatureToOvenSetting(RunningTrials_x2_flt)
            print(f"Temperature Target: {round(RunningTrials_x2_flt,1)}oC")
            print(f"Pre-Heat the Oven to {round(OvenTemperatureSetting,1)}oC.")
            print(f"Pre-Heat the Heraeus to {round(HeraeusTemperatureSetting,1)}oC.")
            print()

            x1_flt = RunningTrials_x1_flt       # Oven time
            x2_flt = RunningTrials_x2_flt       # Oven temperature
            x3_flt = RunningTrials_x3_flt       # UPR1 (vs RD2) in UPR2
            x_vals_lis = [x1_flt,x2_flt,x3_flt]

            j1_flt = OptimisationSetup_obj.j1_IUPR_UPR2vsI1_DecPct_flt
            j2_flt = OptimisationSetup_obj.j2_UPR1_UPvsRD1_DecPct_flt
            j3_flt = OptimisationSetup_obj.j3_I1_CSvsDBP_DecPct_flt
            j4_flt = OptimisationSetup_obj.j4_IUPR_TargetMass_flt
            j_vals_lis = [j1_flt,j2_flt,j3_flt,j4_flt]

            a1_flt = a1_g_flt                       # Mass of UPR1 (g)
            a2_flt = a1_flt*j2_flt                  # Mass of UP (vs RD1) in UPR1 (g)
            a3_flt = a1_flt*(1-j2_flt)              # Mass of RD1 (vs UP) in UPR1 (g)
            a4_flt = ((a2_flt/x3_flt)*(1-x3_flt))-a3_flt # Mass of RD2 (vs UP) in UPR2
            a5_flt = a4_flt+a1_flt                  # Mass of UPR2 (g)
            a6_flt = (a5_flt/j1_flt)*(1-j1_flt)     # Mass of I1 (vs UPR2) in IUPR (g)      [ADD THIS]
            a7_flt = a6_flt*j3_flt                  # Mass of CS (vs DBP) in I1 (g)
            a8_flt = a6_flt*(1-j3_flt)              # Mass of DBP (vs CS) in I1 (g)
            a_vals_lis = [a1_flt,a2_flt,a3_flt,a4_flt,a5_flt,a6_flt,a7_flt,a8_flt]

            var_vals_lis = [TrialIdx_int]+x_vals_lis+j_vals_lis+a_vals_lis
            for count,(val,BackupVariable_arr) in enumerate(zip(var_vals_lis,BackupVariablesMatrix_lis)):
                BackupVariablesMatrix_lis[count] = np.append(BackupVariable_arr,val)

            print(f"Add {round(a4_flt,3)} g {OptimisationSetup_obj.SubstanceName_str}")
            SubstanceName_str = OptimisationSetup_obj.SubstanceName_str
            SubstanceInfo_dict = ChemicalData_dict["chemicals"][SubstanceName_str]
            SubstanceTemperature_flt = OptimisationSetup_obj.SubstanceTemperature_flt
            print(f"Use either:")
            for pipette_str in pipette_lis:
                print(f"> {pipette_str} equipped with {PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict)}")
                CalibrationDataLocation_str = PipettingMethods_obj.CalibrationDataAvailabilityChecker_func(SubstanceInfo_dict=SubstanceInfo_dict,PipetteCalibrationData_dict=PipetteData_dict,PipetteName_str=pipette_str,TipName_str=PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict),Temperature_oc_flt=SubstanceTemperature_flt,PackageLocation_str=MiscMethods_obj.RootPackageLocation_str,PipetteDependenciesLocation_str=PipettingMethods_obj.DependencyFolderLocation_str)
                CalibrationStraightLineEquationParameters_arr = PipettingMethods_obj.CalibrationEquationGenerator_func(CalibrationDataLocation_str)
                Pipettings_int,Setting_flt = PipettingMethods_obj.PipettingStrategyElucidator(a4_flt,CalibrationStraightLineEquationParameters_arr,CalibrationDataLocation_str)
                print(f"Set to {round(Setting_flt,3)} {PipettingMethods_obj.UnitRetriever_func(pipette_str,PipetteData_dict)} and make {Pipettings_int} transfer(s) at {SubstanceTemperature_flt}oC.")
            print()
            print(f"In a weighing boat:")
            print(f"Add {round(a6_flt,3)} g I1")
            print()
            Hours_int,Minutes_int,Seconds_int = MiscMethods_obj.SecondsToHoursMinutesSeconds(RunningTrials_x1_flt)
            print(f"Heat sample for {Hours_int} hours, {Minutes_int} minutes, {Seconds_int} seconds.")
            # MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        # Backing up the variables calculated during the trials.
        BackupVariablesArrays_mat = np.array(BackupVariablesMatrix_lis)
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

def TargetRetrieverMethod20251202_func(client_obj,OptimisationSetup_obj):
    CSTMethods_obj = CSTMethods_class()
    MiscMethods_obj = MiscMethods_class()
    PreviousTrials_df = client_obj.summarize()
    RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
    except:
        print("There is currently no dimensions file for the mechanical test data - creating one - place dimensional data here before attempting target retrieval and processing.")
        df = pd.DataFrame(columns=['TrialIdx','Diameter1_mm','Diameter2_mm',"Diameter3_mm","Height1_mm","Height2_mm","Height3_mm"])
        df.to_csv(OptimisationSetup_obj.RawDataMTDims_str,index=False)
        print("The dimensional dataframe is now available.")
    try:
        dim_df = CSTMethods_obj.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        if dim_df['TrialIdx'].tolist()[-1] == RunningTrialsIdx_arr[-1]:
            print("The dimensional dataframe is up to date - extracting mechanical test data for evaluation.")
            dim_df["DiameterAvg_mm"] = (dim_df["Diameter1_mm"] + dim_df["Diameter2_mm"] + dim_df["Diameter3_mm"]) / 3
            dim_df["HeightAvg_mm"] =  (dim_df["Height1_mm"] + dim_df["Height2_mm"] + dim_df["Height3_mm"]) / 3
            t_lis = []
            for RunningTrialIdx_int in RunningTrialsIdx_arr:
                ArbitrarySustainedRise_int = OptimisationSetup_obj.ArbitrarySustainedRise_int
                StandardDeviationParameter_flt = OptimisationSetup_obj.StandardDeviationParameter_flt
                ArbitraryGradientCutoff_flt = OptimisationSetup_obj.ArbitraryGradientCutoff_flt
                DiameterAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'DiameterAvg_mm']).iloc[0])
                HeightAvg_mm = float((dim_df.loc[dim_df['TrialIdx'] == RunningTrialIdx_int, 'HeightAvg_mm']).iloc[0])
                CsvRawDataTrialMT_str = OptimisationSetup_obj.RawDataMT_str + "/" + str(RunningTrialIdx_int) + ".csv"
                corr_df = pd.read_csv(CSTMethods_obj.RootPackageLocation_str+CSTMethods_obj.CorrectionalFilePath_str)
                cst_df = pd.read_csv(CsvRawDataTrialMT_str)
                corr_df = CSTMethods_obj.SmoothedForceDisplacement_func(corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=False)
                ForceDisplacement_mat = CSTMethods_obj.AlternativeDataFrameCorrector_func(cst_df,corr_df,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False) # True
                StressStrain_mat = CSTMethods_obj.StressStrain_func(ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool=False,ChallengePlotAcceptability_bool=False)
                SmoothedStressStrain_mat = CSTMethods_obj.SmoothedStressStrain_func(StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                DerivativeStressStrain_mat = CSTMethods_obj.DerivativeStressStrain_func(SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                PeakStrain_flt = CSTMethods_obj.PeakFinder_func(DerivativeStressStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True) # True
                print(PeakStrain_flt)
                UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Rubber?","y","n")
                if UserInput_str == "y":
                    print("time to work out the rubber system")
                    YieldBreakPoint_flt = CSTMethods_obj.YieldBreakForUnusualSamples_func(DerivativeStressStrain_mat,SmoothedStressStrain_mat,ToPlotOrNotToPlot_bool=True,ChallengePlotAcceptability_bool=True)
                else:
                    print("normal conditions")
                    YieldBreakPoint_flt = CSTMethods_obj.YieldBreakPoint_func(DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,StressStrain_mat,ArbitraryGradientCutoff=ArbitraryGradientCutoff_flt,ToPlotOrNotToPlot_bool=True,ArbitrarySustainedRise_int=ArbitrarySustainedRise_int,ChallengePlotAcceptability_bool=True) # True
                print(f"The yield break point was : {YieldBreakPoint_flt}")
                t1_flt = np.array(PreviousTrials_df["x1"])[RunningTrialIdx_int]
                t2_flt = YieldBreakPoint_flt
                # t_lis.append([t1_flt,t2_flt])
                t_lis.append(t2_flt)
            t_arr = np.array(t_lis)
            return t_arr
    except:
        print("Please enter the relevant data into the dimensions csv.")
        t_arr = [float(69.69696969696969)]
        return t_arr

class Method20251202Dim3_class():
    def __init__(self):
        self.name = "Tailored Method20251123Dim3: Dual-objective optimisation of both compressive strength and oven time."
        self.ExecutionChecker = ExecutionCheckerMethod20241024_func
        self.TargetRetriever = TargetRetrieverMethod20251202_func
    def MixingProcedure_func(self,client_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        ChemicalData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + MiscMethods_obj.ChemicalDependencyFileLocation_str)
        PipettingMethods_obj = PipettingMethods_class()
        PipetteData_dict = MiscMethods_obj.jsonOpener_func(MiscMethods_obj.RootPackageLocation_str + PipettingMethods_obj.DependencyFileLocation_str)
        pipette_lis = ["P200","P20"]

        print("Retrieving trials...")
        AllTrials_df = client_obj.summarize()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])

        RunningTrials_x1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].name}'])
        RunningTrials_x2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].name}'])
        RunningTrials_x3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].name}'])

        print(f"\n===== Initial Pouring of UPR1 =====")
        a1_g_arr = np.empty(0)

        for TrialIdx_int,x3 in zip(RunningTrialsIdx_arr,RunningTrials_x3_arr):
            print(f"\n-----Trial {TrialIdx_int}-----")
            TargetMassOfUPR1_g_float = (OptimisationSetup_obj.j4_IUPR_TargetMass_flt*OptimisationSetup_obj.j1_IUPR_UPR2vsI1_DecPct_flt)*x3
            print(f"Use a pasteur pipette to transfer around {round(TargetMassOfUPR1_g_float,2)} g UPR1 to the silicone mould gap.")
            a1_g_arr = np.append(arr=a1_g_arr,values=MiscMethods_obj.NumericUserInputRetriever_func("Mass of StockUPR1 transferred? (g)"))

        print(f"\n===== Synthesis and Oven Time =====")
        BackupVariableNames_lis = ["TrialIdx","x1","x2","x3","j1","j2","j3","j4","a1","a2","a3","a4","a5","a6","a7","a8"]
        BackupVariablesMatrix_lis = []
        for i in BackupVariableNames_lis:
            BackupVariablesMatrix_lis.append(np.empty(0))

        for TrialIdx_int,a1_g_flt,RunningTrials_x1_flt,RunningTrials_x2_flt,RunningTrials_x3_flt in zip(RunningTrialsIdx_arr,a1_g_arr,RunningTrials_x1_arr,RunningTrials_x2_arr,RunningTrials_x3_arr):
            print(f"\n-----Trial {TrialIdx_int}-----")
            OvenTemperatureSetting,HeraeusTemperatureSetting = MiscMethods_obj.ActualTemperatureToOvenSetting(RunningTrials_x2_flt)
            print(f"Temperature Target: {round(RunningTrials_x2_flt,1)}oC")
            print(f"Pre-Heat the Oven to {round(OvenTemperatureSetting,1)}oC.")
            print(f"Pre-Heat the Heraeus to {round(HeraeusTemperatureSetting,1)}oC.")
            print()

            x1_flt = RunningTrials_x1_flt       # Oven time
            x2_flt = RunningTrials_x2_flt       # Oven temperature
            x3_flt = RunningTrials_x3_flt       # UPR1 (vs RD2) in UPR2
            x_vals_lis = [x1_flt,x2_flt,x3_flt]

            j1_flt = OptimisationSetup_obj.j1_IUPR_UPR2vsI1_DecPct_flt
            j2_flt = OptimisationSetup_obj.j2_UPR1_UPvsRD1_DecPct_flt
            j3_flt = OptimisationSetup_obj.j3_I1_CSvsDBP_DecPct_flt
            j4_flt = OptimisationSetup_obj.j4_IUPR_TargetMass_flt
            j_vals_lis = [j1_flt,j2_flt,j3_flt,j4_flt]

            a1_flt = a1_g_flt                       # Mass of UPR1 (g)
            a2_flt = a1_flt*j2_flt                  # Mass of UP (vs RD1) in UPR1 (g)
            a3_flt = a1_flt*(1-j2_flt)              # Mass of RD1 (vs UP) in UPR1 (g)
            a4_flt = ((a2_flt/x3_flt)*(1-x3_flt))-a3_flt # Mass of RD2 (vs UP) in UPR2
            a5_flt = a4_flt+a1_flt                  # Mass of UPR2 (g)
            a6_flt = (a5_flt/j1_flt)*(1-j1_flt)     # Mass of I1 (vs UPR2) in IUPR (g)      [ADD THIS]
            a7_flt = a6_flt*j3_flt                  # Mass of CS (vs DBP) in I1 (g)
            a8_flt = a6_flt*(1-j3_flt)              # Mass of DBP (vs CS) in I1 (g)
            a_vals_lis = [a1_flt,a2_flt,a3_flt,a4_flt,a5_flt,a6_flt,a7_flt,a8_flt]

            var_vals_lis = [TrialIdx_int]+x_vals_lis+j_vals_lis+a_vals_lis
            for count,(val,BackupVariable_arr) in enumerate(zip(var_vals_lis,BackupVariablesMatrix_lis)):
                BackupVariablesMatrix_lis[count] = np.append(BackupVariable_arr,val)

            print(f"Add {round(a4_flt,3)} g {OptimisationSetup_obj.SubstanceName_str}")
            SubstanceName_str = OptimisationSetup_obj.SubstanceName_str
            SubstanceInfo_dict = ChemicalData_dict["chemicals"][SubstanceName_str]
            SubstanceTemperature_flt = OptimisationSetup_obj.SubstanceTemperature_flt
            print(f"Use either:")
            for pipette_str in pipette_lis:
                print(f"> {pipette_str} equipped with {PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict)}")
                CalibrationDataLocation_str = PipettingMethods_obj.CalibrationDataAvailabilityChecker_func(SubstanceInfo_dict=SubstanceInfo_dict,PipetteCalibrationData_dict=PipetteData_dict,PipetteName_str=pipette_str,TipName_str=PipettingMethods_obj.TipSelector_func(pipette_str,PipetteData_dict),Temperature_oc_flt=SubstanceTemperature_flt,PackageLocation_str=MiscMethods_obj.RootPackageLocation_str,PipetteDependenciesLocation_str=PipettingMethods_obj.DependencyFolderLocation_str)
                CalibrationStraightLineEquationParameters_arr = PipettingMethods_obj.CalibrationEquationGenerator_func(CalibrationDataLocation_str)
                Pipettings_int,Setting_flt = PipettingMethods_obj.PipettingStrategyElucidator(a4_flt,CalibrationStraightLineEquationParameters_arr,CalibrationDataLocation_str)
                print(f"Set to {round(Setting_flt,3)} {PipettingMethods_obj.UnitRetriever_func(pipette_str,PipetteData_dict)} and make {Pipettings_int} transfer(s) at {SubstanceTemperature_flt}oC.")
            print()
            print(f"In a weighing boat:")
            print(f"Add {round(a6_flt,3)} g I1")
            print()
            Hours_int,Minutes_int,Seconds_int = MiscMethods_obj.SecondsToHoursMinutesSeconds(RunningTrials_x1_flt)
            print(f"Heat sample for {Hours_int} hours, {Minutes_int} minutes, {Seconds_int} seconds.")
            # MiscMethods_obj.CheckpointUserInputRetriever_func("Continue? (y)", "y")

        # Backing up the variables calculated during the trials.
        BackupVariablesArrays_mat = np.array(BackupVariablesMatrix_lis)
        BackupCsvSaving(OptimisationSetup_obj,BackupVariablesArrays_mat,BackupVariableNames_lis)

class TailoredMethods_class(object):
    def __init__(self):
        self.name = "Outer Class - Tailored Methods Class"
        self.Method20241026Dim1 = Method20241026Dim1_class()
        self.Method20241026Dim2 = Method20241026Dim2_class()
        self.Method20241026Dim3 = Method20241026Dim3_class()
        self.Method20241028Dim2 = Method20241028Dim2_class()
        self.Method20250310Dim3 = Method20250310Dim3_class()
        self.Method20250518Dim3 = Method20250518Dim3_class()
        self.Method20250623Dim2 = Method20250623Dim2_class()
        self.Method20250623Dim3 = Method20250623Dim3_class()
        self.Method20250625Dim3 = Method20250625Dim3_class()
        self.Method20250627Dim3 = Method20250627Dim3_class()
        self.Method20250817Dim4 = Method20250817Dim4_class()
        self.Method20251120Dim2 = Method20251120Dim2_class()
        self.Method20251123Dim3 = Method20251123Dim3_class()
        self.Method20251202Dim3 = Method20251202Dim3_class()