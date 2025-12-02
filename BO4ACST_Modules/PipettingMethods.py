import numpy as np
import pandas as pd
from MiscMethods import MiscMethods_class
class PipettingMethods_class():
    def __init__(self):
        self.name = "PipettingMethods"
        self.DependencyFolderLocation_str = "/BO4ACST_Dependencies/PipettingDependencies"
        self.DependencyFileLocation_str = "/BO4ACST_Dependencies/PipettingDependencies/data_2024-10-04_PipetteData.json"
    def VolumeFinder_func(self,SubstanceMass_g_flt:float,SubstanceInfo_dict:dict)->float:
        """
        This function takes the desired mass of substance to be pipetted and the substances
        dictionary of fundamental characteristics (from which it draws a density value if available);
        it therewith calculated the volume of the desired substance.
        Takes:
            SubstanceMass_g_flt = float, mass of desired substance to be pipetted in g.
            SubstanceInfo_dict = dictionary, important data about the substance to be pipetted.
        Returns:
            SubstanceVolume_ul_flt = float, volume of desired substance to be pipetted in ul.
        """
        try:
            SubstanceDensity = SubstanceInfo_dict.get("density_flt")
            if SubstanceDensity == None:
                raise Exception()
        except:
            print(f"No density available for substance named {SubstanceInfo_dict.get('names_str')}!")
        SubstanceVolume_ml_flt = SubstanceMass_g_flt / SubstanceDensity
        SubstanceVolume_ul_flt = SubstanceVolume_ml_flt * 1000
        return SubstanceVolume_ul_flt
    def PipetteSelector_func(self,SubstanceVolume_ul_flt:float,PipetteData_dict:dict)->str:
        """
        This function takes the desired volume of substance to be pipetted and a dictionary of
        information regarding the pipettes available and finally returns the name of a suitable pipette
        for the job.
        Takes:
            SubstanceVolume_ul_flt = float, volume of desired substance to be pipetted in ul.
            PipetteData_dict = dictionary, important data about the pipettes available to be used.
        Returns:
            PipetteName_str = string, the name of the pipette selected for the job.
        """
        import numpy as np
        # A couple of lists are generated describing the pipettes names and their various pipettable ranges.
        PipetteNames_lis = []
        PipetteMaxVols_ul_lis = []
        PipetteMinVols_ul_lis = []
        for PipetteModel_str in PipetteData_dict["pipettes"].keys():
            PipetteNames_lis.append(PipetteModel_str)
            PipetteMaxVols_ul_lis.append(PipetteData_dict["pipettes"][f"{PipetteModel_str}"].get("maxVol_ul"))
            PipetteMinVols_ul_lis.append(PipetteData_dict["pipettes"][f"{PipetteModel_str}"].get("minVol_ul"))
        # This clause gives the user the correct pipette for the job if it is within the operating bounds of the pipette.
        # In these cases the largest pipette is chosen and multiple pipettings required to reach the desired volume.
        for PipetteName_str,PipetteMaxVol_ul_flt,PipetteMinVol_ul_flt in zip(PipetteNames_lis,PipetteMaxVols_ul_lis,PipetteMinVols_ul_lis):
            if SubstanceVolume_ul_flt >= PipetteMinVol_ul_flt and SubstanceVolume_ul_flt <= PipetteMaxVol_ul_flt:
                return PipetteName_str
        # This if clause tells the user to use the largest pipette if the required volumes are exceptionally high.
        # In these cases the perfect pipette has successfully been found, a single pipetting will suffice to reach the desired volume.
        if SubstanceVolume_ul_flt > max(PipetteMaxVols_ul_lis):
            return PipetteNames_lis[PipetteMaxVols_ul_lis.index(max(PipetteMaxVols_ul_lis))]
        PossibleClosestMaxVolumes = []
        # This clause throws an exception where desired volumes are too low for any of the pipettes available.
        try:
            if SubstanceVolume_ul_flt < min(PipetteMinVols_ul_lis):
                raise Exception()
        except:
            print(f"No pipette available with a range low enough to pipette {SubstanceVolume_ul_flt} ul!")
        # The below code is used on the occasions when volumes are required between the pipettes available.
        # In these cases the smaller pipette is elected and a multiple pipettings required to reach the desired volume.
        for PipetteMaxVol_ul_flt in PipetteMaxVols_ul_lis:
            PossibleClosestMaxVolumes.append(SubstanceVolume_ul_flt - PipetteMaxVol_ul_flt)
        for counter,value_flt in enumerate(PossibleClosestMaxVolumes):
            if value_flt < 0:
                PossibleClosestMaxVolumes[counter] = np.max(PossibleClosestMaxVolumes)
        MinIdxPosition_int = PossibleClosestMaxVolumes.index(min(PossibleClosestMaxVolumes))
        PipetteName_str = PipetteNames_lis[MinIdxPosition_int]
        # The pipette which has been chosen for the job is proffered.
        return PipetteName_str
    def TipSelector_func(self,PipetteName_str:str,PipetteData_dict:dict)->str:
        """
        This function takes the pipette data dictionary and the currently selected pipette for
        job's name, it therewith looks for the default pipette and returns this. Later this function
        could be adapted for customisation- such as selecting an alternative pipette tip for a job.
        Takes:
            PipetteName_str = string, the name of the pipette selected for the job.
            PipetteData_dict = dictionary, important data about the pipettes available to be used.
        Returns:
            TipName_str = string, the name of the pipette tip to be used.
        """
        TipName_str = PipetteData_dict["pipettes"][f"{PipetteName_str}"].get("DefaultTip")
        return TipName_str

    def UnitRetriever_func(self,PipetteName_str:str,PipetteData_dict:dict)->str:
        UnitName_str = PipetteData_dict["pipettes"][f"{PipetteName_str}"].get("settingUnit")
        return UnitName_str
    def CalibrationDataAvailabilityChecker_func(self,SubstanceInfo_dict:dict,PipetteCalibrationData_dict:dict,PipetteName_str:str,TipName_str:str,PackageLocation_str:str,PipetteDependenciesLocation_str:str,Temperature_oc_flt:float)->str:
        """
        This function takes data available about the substance in use as well as information available
        about currently held data on calibrations and trawls these using the currently selected pipette
        and tips to find out whether our database holds and calibrations suitable for the job at hand.
        Takes:
            SubstanceInfo_dict = dictionary, important data about the substance to be pipetted.
            PipetteCalibrationData_dict = dictionary, dataset over substances which have been calibrated for.
            PipetteName_str = string, the name of the pipette selected for the job.
            TipName_str = string, the name of the pipette tip to be used.
            PackageLocation_str = string, the absolute path of the package location.
            PipetteDependenciesLocation_str = string, the relative location of the pipette dependencies folder from the package location.
        Returns:
            CalibrationDataLocation_str = string, the location of the calibration file suitable for this substance and the selected pipette and tips.
        """
        import json
        SubstanceCAS_str = SubstanceInfo_dict.get("cas_str")
        try:
            PipetteCalibrationMetadataLocation_str = PipetteCalibrationData_dict["calibrations"][f"{SubstanceCAS_str}"].get("location")
        except:
            print(f"No calibration metadata files available on this substance (CAS:{SubstanceCAS_str})!")
        with open(f'{PackageLocation_str + PipetteDependenciesLocation_str + PipetteCalibrationMetadataLocation_str}') as f:
            PipetteCalibrationMetadata_dict = json.load(f)
        CalibrationName_str = "None"
        for item in PipetteCalibrationMetadata_dict["PipetteCalibration"].items():
            if item[1].get("pipette") == PipetteName_str and item[1].get("tip") == TipName_str and item[1].get("TemperatureOfSubstanceOC") == Temperature_oc_flt:
                CalibrationName_str = item[0]
        try:
            if CalibrationName_str == "None":
                raise Exception()
        except:
            print(f"No calibrations have been done for this substance with the selected pipette ({PipetteName_str}) and tip ({TipName_str}) and subtance temperature!")
        CalibrationDataLocation_str = PackageLocation_str + PipetteDependenciesLocation_str + PipetteCalibrationMetadata_dict["PipetteCalibration"][f"{CalibrationName_str}"].get("location")
        return CalibrationDataLocation_str
    def CalibrationEquationGenerator_func(self,CalibrationDataLocation_str:str)->np.ndarray:
        """
        This function takes the location of the calibration data to be used and provides a calibration
        equation, a straight line fit, which can be used to interconvert between the setting of the 
        pipette and the grammage desired.
        Takes:
            CalibrationDataLocation_str = string, the location of the calibration file suitable for this substance and the selected pipette and tips.
        Returns:
            CalibrationStraightLineEquationParameters_arr
        """
        import pandas as pd
        import numpy as np
        Calibration_df = pd.read_csv(f"{CalibrationDataLocation_str}")
        CalibrationStraightLineEquationParameters_arr = np.polyfit(x=Calibration_df["Mass_g"],y=Calibration_df["Setting"],deg=1)
        # CalibrationStraightLineEquationParameters_arr = np.polyfit(x=Calibration_df["Mass_g"],y=Calibration_df["Setting"],deg=2)
        return CalibrationStraightLineEquationParameters_arr
    def PipettingStrategyDesigner_func(self,PipetteName_str:str,TipName_str:str,PipetteData_dict:dict,SubstanceVolume_ul_flt:float)->tuple[int,float]:
        """
        This function takes into account the desired grammage, pipette, and tips and constructs a strategy for
        pipetting the mass. If the volume is within the operating range of the pipette and its tips then a single
        pipetting will suffice, larger volumes require multiple pipettes worth of substance.
        Takes:
            PipetteName_str = string, the name of the pipette selected for the job.
            TipName_str = string, the name of the pipette tip to be used.
            PipetteData_dict = dictionary, important data about the pipettes available to be used.
            SubstanceVolume_ul_flt = float, volume of desired substance to be pipetted in ul.
        Returns:
            NumberOfPipettingRounds_int = integer, number of rounds of pipetting required.
            VolumePerRound_ul_flt = float, volume to be pipetted per round.
        """
        MaxVol_lis = []
        MaxVol_lis.append(PipetteData_dict["pipettes"][f"{PipetteName_str}"].get("maxVol_ul"))
        MaxVol_lis.append(PipetteData_dict["tips"][f"{TipName_str}"].get("maxVol_ul"))
        LimitingMaxVol_ul_flt = min(MaxVol_lis)
        NumberOfPipettingRounds_int = 0
        if SubstanceVolume_ul_flt < LimitingMaxVol_ul_flt:
            NumberOfPipettingRounds_int = 1
            VolumePerRound_ul_flt = SubstanceVolume_ul_flt
        else:
            StopStart_bool = False
            i = 1
            while StopStart_bool == False:
                DividedSubstanceVolume_ul_flt = SubstanceVolume_ul_flt / i
                if DividedSubstanceVolume_ul_flt < LimitingMaxVol_ul_flt:
                    StopStart_bool = True
                else:
                    i += 1
            NumberOfPipettingRounds_int = i
            VolumePerRound_ul_flt = DividedSubstanceVolume_ul_flt
        return NumberOfPipettingRounds_int,VolumePerRound_ul_flt
    def PipetteSettingFinder_func(self,NumberOfPipettingRounds_int:int,SubstanceMass_g_flt:float,PipetteName_str:str,PipetteData_dict:dict,CalibrationStraightLineEquationParameters_arr:np.ndarray)->tuple[float,str]:
        """
        This function takes into consideration the pipetting strategy chosen and calculates the masses intended
        per round of pipetting, after which, it calculates the setting required on the pipette to achieve this strategy.
        This is achieved by taking into account the pipette's setting scale as well as the calibration curve equations
        which have been derived for this particular set up of substance, pipette, and tips.
        Takes:
            NumberOfPipettingRounds_int = integer, number of rounds of pipetting to be performed.
            SubstanceMass_g_flt = float, desired mass of the substance to be pipetted.
            PipetteName_str = string, name of the selected pipette to be used.
            PipetteData_dict = dictionary, data regarding pipettes, tips, etc
            CalibrationStraightLineEquationParameters_arr = array, equation for calibration curve.
        Returns:
            PipetteSetting_flt = float, the pipette setting to be set up.
            PipetteSettingUnits_str = string, the unit of the setting for the pipette.
        """
        MassPerRound_g_flt = SubstanceMass_g_flt / NumberOfPipettingRounds_int
        # What are the units of the y axis?
        PipetteSettingUnits_str = PipetteData_dict["pipettes"][f"{PipetteName_str}"].get("settingUnit")
        PipetteSetting_flt = (CalibrationStraightLineEquationParameters_arr[0] * MassPerRound_g_flt) + CalibrationStraightLineEquationParameters_arr[1]
        # PipetteSetting_flt = (CalibrationStraightLineEquationParameters_arr[0] * (MassPerRound_g_flt ** 2)) + (CalibrationStraightLineEquationParameters_arr[1] * MassPerRound_g_flt) + CalibrationStraightLineEquationParameters_arr[2]
        return PipetteSetting_flt,PipetteSettingUnits_str
    def MassToVolumeToSettingToStrategy_func(self,SubstanceMass_g_flt,SubstanceAcronym_str):
        # Instantiating objects of utility.
        MiscM_obj = MiscMethods_class()
        # Useful dictionaries opened.
        PipetteData_dict = MiscM_obj.jsonOpener_func(MiscM_obj.RootPackageLocation_str + self.DependencyFileLocation_str)
        ChemicalData_dict = MiscM_obj.jsonOpener_func(MiscM_obj.RootPackageLocation_str + MiscM_obj.ChemicalDependencyFileLocation_str)
        # A proposed substance.
        SubstanceInfo_dict = ChemicalData_dict["chemicals"][SubstanceAcronym_str]
        # Finding the volume of substance to be pipetted using the VolumeFinder_func function.
        SubstanceVolume_ul_flt = self.VolumeFinder_func(SubstanceMass_g_flt,SubstanceInfo_dict)
        # Deduced pipette to be used.
        PipetteName_str = self.PipetteSelector_func(SubstanceVolume_ul_flt,PipetteData_dict)
        # Deduced tip to be used.
        TipName_str = self.TipSelector_func(PipetteName_str,PipetteData_dict)
        # Finding out if there is a calibration dataset for this substance and the equipment in question.
        CalibrationDataLocation_str = self.CalibrationDataAvailabilityChecker_func(SubstanceInfo_dict,PipetteData_dict,PipetteName_str,TipName_str,MiscM_obj.RootPackageLocation_str,self.DependencyFolderLocation_str)
        # The calibration curve is fitted in the form of a simple straight line equation.
        # y = ax + b
        # y = (CalibrationStraightLineEquationParameters_arr[0] * x) + CalibrationStraightLineEquationParameters_arr[1]
        CalibrationStraightLineEquationParameters_arr = self.CalibrationEquationGenerator_func(CalibrationDataLocation_str)
        # Retrieving a pipetting strategy.
        NumberOfPipettingRounds_int,VolumePerRound_ul_flt = self.PipettingStrategyDesigner_func(PipetteName_str,TipName_str,PipetteData_dict,SubstanceVolume_ul_flt)
        # Retrieving a pipetting setting for the strategy.
        PipetteSetting_flt,PipetteSettingUnits_str = self.PipetteSettingFinder_func(NumberOfPipettingRounds_int,SubstanceMass_g_flt,PipetteName_str,PipetteData_dict,CalibrationStraightLineEquationParameters_arr)
        return PipetteName_str,TipName_str,NumberOfPipettingRounds_int,PipetteSetting_flt,PipetteSettingUnits_str
    def PipettingStrategyElucidator(self,Mass_g_flt,CalibrationStraightLineEquationParameters_arr,CalibrationDataLocation_str):
        import math
        df = pd.read_csv(f"{CalibrationDataLocation_str}")
        MaxSetting_flt = float(df["Setting"][np.argmax(df["Mass_g"])])
        MaxMass_flt = (MaxSetting_flt - CalibrationStraightLineEquationParameters_arr[1]) / CalibrationStraightLineEquationParameters_arr[0]
        Pipettings_int = math.ceil(Mass_g_flt / MaxMass_flt)
        MassPerPipetting = Mass_g_flt / Pipettings_int
        Setting_flt = (CalibrationStraightLineEquationParameters_arr[0] * MassPerPipetting) + CalibrationStraightLineEquationParameters_arr[1]
        return Pipettings_int,Setting_flt