from ax.service.ax_client import AxClient
from MiscMethods import MiscMethods_class
from PipettingMethods import PipettingMethods_class
from CSTMethods import CSTMethods_class
import numpy as np
from pathlib import Path
import pandas as pd
import os

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

def ExecutionCheckerMethod20241024_func(AxClient_obj,OptimisationSetup_obj):
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
        PreviousTrials_df = AxClient_obj.get_trials_data_frame()
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

def TargetRetrievalMethod20241023_func(AxClient_obj,OptimisationSetup_obj):
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
    PreviousTrials_df = AxClient_obj.get_trials_data_frame()
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
    def MixingProcedure_func(self,AxClient_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()
        # Retrieve the trials data:
        print("Retrieving trials.")
        AllTrials_df = AxClient_obj.get_trials_data_frame()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].get("name")}'])

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
    def MixingProcedure_func(self,AxClient_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()
        # Retrieve the trials data:
        print("Retrieving trials.")
        AllTrials_df = AxClient_obj.get_trials_data_frame()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].get("name")}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].get("name")}'])

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
    def MixingProcedure_func(self,AxClient_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()
        # Retrieve the trials data:
        print("Retrieving trials.")
        AllTrials_df = AxClient_obj.get_trials_data_frame()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].get("name")}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].get("name")}'])
        RunningTrialsX3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].get("name")}'])

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

def TargetRetriever20241028_func(AxClient_obj,OptimisationSetup_obj):
    AllTrials_df = AxClient_obj.get_trials_data_frame()
    RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
    RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
    RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].get("name")}'])
    RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].get("name")}'])
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
    def MixingProcedure_func(self,AxClient_obj:AxClient,OptimisationSetup_obj):
        MiscMethods_obj = MiscMethods_class()
        PipettingM_obj = PipettingMethods_class()

        print("Retrieving trials.")
        AllTrials_df = AxClient_obj.get_trials_data_frame()
        RunningTrials_df = AllTrials_df[AllTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        RunningTrialsX1_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[0].get("name")}'])
        RunningTrialsX2_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[1].get("name")}'])
        RunningTrialsX3_arr = np.array(RunningTrials_df[f'{OptimisationSetup_obj.Parameters_lis[2].get("name")}'])

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

def UCSTargetRetrievalMethod20250518_func(AxClient_obj,OptimisationSetup_obj):
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
    PreviousTrials_df = AxClient_obj.get_trials_data_frame()
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

def YMTargetRetrieverMethod20250518_func(AxClient_obj,OptimisationSetup_obj):
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
    PreviousTrials_df = AxClient_obj.get_trials_data_frame()
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

def AddUCSvsYMRetrieverMethod20250518_func(AxClient_obj,OptimisationSetup_obj):
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
    PreviousTrials_df = AxClient_obj.get_trials_data_frame()
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

class TailoredMethods_class(object):
    def __init__(self):
        self.name = "Outer Class - Tailored Methods Class"
        self.Method20241026Dim1 = Method20241026Dim1_class()
        self.Method20241026Dim2 = Method20241026Dim2_class()
        self.Method20241026Dim3 = Method20241026Dim3_class()
        self.Method20241028Dim2 = Method20241028Dim2_class()
        self.Method20250310Dim3 = Method20250310Dim3_class()
        self.Method20250518Dim3 = Method20250518Dim3_class()