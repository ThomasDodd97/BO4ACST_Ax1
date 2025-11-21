import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from MiscMethods import MiscMethods_class
import scipy as sp

class CSTMethods_class():
    def __init__(self):
        self.name = "CSTMethods"
        self.CorrectionalFilePath_str = "/BO4ACST_Dependencies/MiscellaneousDependencies/data_2024-10-25_RawDataForCorrectionOfCST.csv"
        self.RootPackageLocation_str = str(Path(os.path.abspath(__file__)).parent.absolute().parent.absolute())
    def DataframeGetter_func(self,path_str):
        """
        This function takes a path string of a csv file and returns a pandas dataframe of it.
        Takes:
            path_str = string, file path to csv.
        Returns:
            cst_df = pandas dataframe, compressive strength test dataframe.
        """
        import pandas as pd
        cst_df = pd.read_csv(path_str)
        return cst_df

    def DataFrameCorrector(self,cst_df,corr_df):
        """
        This function takes the raw data in the form of a pandas dataframe, as well as a correctional dataframe.
        Takes:
            cst_df = pandas dataframe, compressive strength test dataframe.
            corr_df = pandas dataframe, correctional compressive strength test dataframe from universal test machine without a sample.
        Returns:
            cst_df = pandas dataframe, compressive strength test dataframe (With corrected positions).
        """
        import numpy as np
        import pandas as pd
        CorrectionalPosition_arr = np.array(corr_df["Position (mm)"])
        CorrectionalForce_arr = np.array(corr_df["Force (N)"])
        Force_arr = np.array(cst_df["Force (N)"])
        UncorrectedPosition_arr = np.array(cst_df["Position (mm)"])
        CorrectedPosition_arr = np.empty(shape=0)
        for count,(UncorrectedPosition_flt,Force_flt) in enumerate(zip(UncorrectedPosition_arr,Force_arr)):
            for count,(CorrectionalPosition_flt, CorrectionalForce_flt) in enumerate(zip(CorrectionalPosition_arr,CorrectionalForce_arr)):
                if Force_flt < CorrectionalForce_flt:
                    CorrectedPosition = UncorrectedPosition_flt - CorrectionalPosition_arr[count]
                    CorrectedPosition_arr = np.append(arr=CorrectedPosition_arr,values=CorrectedPosition)
                    break
        cst_df["CorrectedPosition (mm)"] = CorrectedPosition_arr
        return cst_df

    def StressCalculator_func(self,cst_df,sampledia_mm_flt):
        """
        This function takes a compressive strength test dataframe, analyses the raw data within,
        and appends an array of stresses.
        Takes:
            cst_df = pandas dataframe, compressive strength test dataframe.
            sampledia_mm_flt = float, sample diameter in mm.
        Returns:
            cst_df = pandas dataframe, compressive strength test dataframe (with stresses).
        """
        import pandas as pd
        import numpy as np
        # stress = force / area
        # N / mm^2
        sampleradius_mm_flt = sampledia_mm_flt / 2
        samplearea_mm2_flt = np.pi * (sampleradius_mm_flt ** 2)
        cst_df["Stress (N mm^-2)"] = cst_df["Force (N)"] / samplearea_mm2_flt
        return cst_df

    def StrainCalculator_func(self,cst_df,samplelength_mm_flt):
        """
        This function takes a compressive strength test dataframe, analyses the raw data within,
        and appends an array of strains.
        Takes:
            cst_df = pandas dataframe, compressive strength test dataframe.
            samplelength_mm_flt = float, sample start length in mm.
        Returns:
            cst_df = pandas dataframe, compressive strength test dataframe (with strains).
        """
        import pandas as pd
        # strain = (Length of specimen at applied stress - Initial sample length) / Initial sample length
        # unitless
        cst_df["Strain"] = (((samplelength_mm_flt - cst_df["CorrectedPosition (mm)"]) - samplelength_mm_flt) / samplelength_mm_flt) * -1
        return cst_df

    def PandasToNumpy_func(self,cst_df):
        """
        This function takes a compressive strength test dataframe and places it into a numpy
        array format.
        Takes:
            cst_df = pandas dataframe, compressive strength test dataframe.
            FullResCst_arr = numpy array, compressive strength test data array (NO DOWNSAMPLING).
        """
        import pandas as pd
        import numpy as np

        compdist_arr = np.array([list(cst_df["CorrectedPosition (mm)"])])
        FullResCst_arr = np.zeros(shape=(0,len(list(cst_df["CorrectedPosition (mm)"]))))
        index_arr = [np.linspace(0,len(list(cst_df["CorrectedPosition (mm)"]))-1,len(list(cst_df["CorrectedPosition (mm)"])))]

        force_arr = (np.array([list(cst_df["Force (N)"])]))
        strain_arr = (np.array([list(cst_df["Strain"])]))
        stress_arr = (np.array([list(cst_df["Stress (N mm^-2)"])]))

        FullResCst_arr = np.append(arr=FullResCst_arr,values=index_arr,axis=0)
        FullResCst_arr = np.append(arr=FullResCst_arr,values=compdist_arr,axis=0)
        FullResCst_arr = np.append(arr=FullResCst_arr,values=force_arr,axis=0)
        FullResCst_arr = np.append(arr=FullResCst_arr,values=strain_arr,axis=0)
        FullResCst_arr = np.append(arr=FullResCst_arr,values=stress_arr,axis=0)
        StressDiff_arr = np.empty(shape=0)
        j = 0
        for i in list(cst_df["Stress (N mm^-2)"]):
            k = i - j
            j = i
            StressDiff_arr = np.append(StressDiff_arr,k)
        # StressDiff_arr = [np.delete(StressDiff_arr, (0))]
        StressDiff_arr = np.array(object=[list(StressDiff_arr)])
        FullResCst_arr = np.append(arr=FullResCst_arr,values=StressDiff_arr,axis=0)

        return FullResCst_arr

    def PandasToDownsampledNumpy_func(self,cst_df,Downsamplingfactor_int):
        """
        This function takes a compressive strength test dataframe and places it into a numpy
        array format. Additionally it downsamples it.
        Takes:
            cst_df = pandas dataframe, compressive strength test dataframe.
            Downsamplingfactor_int = An unitless downsampling factor.
        Returns:
            cst_arr = numpy array, compressive strength test data array. This takes the shape of
                (6,?) where ? will be the downscaled length of the arrays. The order of the contents
                are as follows:
                [[Index],[Displacement in mm],[Force in N],[Strain],[Stress in N mm^-2],[Change in Stress in N mm^-2]]
        """
        import pandas as pd
        import numpy as np

        compdist_lis = list((np.array(cst_df["CorrectedPosition (mm)"])).reshape(-1, Downsamplingfactor_int).mean(axis=1))
        compdist_arr = np.array(object=[compdist_lis])
        cst_arr = np.zeros(shape=(0,len(compdist_lis)))
        index_arr = [np.linspace(0,len(compdist_lis)-1,len(compdist_lis))]

        force_lis = list((np.array([cst_df["Force (N)"]])).reshape(-1, Downsamplingfactor_int).mean(axis=1))
        force_arr = np.array(object=[force_lis])
        strain_lis = list((np.array([cst_df["Strain"]])).reshape(-1, Downsamplingfactor_int).mean(axis=1))
        strain_arr = np.array(object=[strain_lis])
        stress_lis = list((np.array([cst_df["Stress (N mm^-2)"]])).reshape(-1, Downsamplingfactor_int).mean(axis=1))
        stress_arr = np.array(object=[stress_lis])

        # The averaging divides the original array into sections which are as long as the arbitrary downsampling
        # factor. These sections are then averaged and the means are placed in a second downsampled array.

        cst_arr = np.append(arr=cst_arr,values=index_arr,axis=0)
        cst_arr = np.append(arr=cst_arr,values=compdist_arr,axis=0)
        cst_arr = np.append(arr=cst_arr,values=force_arr,axis=0)
        cst_arr = np.append(arr=cst_arr,values=strain_arr,axis=0)
        cst_arr = np.append(arr=cst_arr,values=stress_arr,axis=0)

        StressDiff_arr = np.empty(shape=0)
        j = 0
        for i in stress_lis:
            k = i - j
            j = i
            StressDiff_arr = np.append(StressDiff_arr,k)
        # StressDiff_arr = [np.delete(StressDiff_arr, (0))]
        StressDiff_arr = np.array(object=[list(StressDiff_arr)])
        cst_arr = np.append(arr=cst_arr,values=StressDiff_arr,axis=0)

        return cst_arr

    def PlotStressStrain(self,cst_arr):
        """
        This function takes the compressive strength testing data array, and
        produces an overview graphic.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
        Outputs:
            A single graphic of stress vs strain.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]

        figure,(plot1) = plt.subplots(1,1, figsize=(10, 3))

        plot1.scatter(strain_arr,stress_arr,s=1)
        plot1.set_xlabel("Strain")
        plot1.set_ylabel("Stress (N mm^-2)")
        plot1.set_xlim(left=0,right=np.max(strain_arr))
        plot1.set_ylim(bottom=0,top=np.max(stress_arr))
        plt.show()
    
    def PlotStressStrainWithPeakStress(self,cst_arr,ucs_flt):
        """
        This function takes the compressive strength testing data array, and
        produces an overview graphic. Also adds in a horizontal line
        describing the max stress attained.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
            ucs_flt = float, maximum compressive strength discovered.
        Outputs:
            A single graphic of stress vs strain.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]

        figure,(plot1) = plt.subplots(1,1, figsize=(10, 3))

        plot1.scatter(strain_arr,stress_arr,s=1)
        plot1.set_xlabel("Strain")
        plot1.set_ylabel("Stress (N mm^-2)")
        plot1.set_xlim(left=0,right=np.max(strain_arr))
        plot1.set_ylim(bottom=0,top=np.max(stress_arr)+20)
        plot1.hlines(y=[ucs_flt],xmin=[min(strain_arr)],xmax=[max(strain_arr)],colors=["red"])
        plt.show()

    def InflectionFinder(self,cst_arr,HorizonValue_int):
        """
        This function takes the compressive strength testing data array, and figures
        out the index position of the initial inflection point within the stress change vs
        strain graph, this locates a crucial position on the curve that can provide the
        young's modulus.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
            HorizonValue_int = integer, the number of points backwards and forwards the algorithm can 'look'.
        Returns:
            InflectionIdx_int = integer, the index of the point which sits on the initial inflection.
        """
        import numpy as np
        strain_arr = cst_arr[3]
        ForceDiff_arr = cst_arr[5]
        for counter,(strain,forcechange) in enumerate(zip(strain_arr,ForceDiff_arr)):
            if counter < HorizonValue_int:
                continue
            elif counter > (len(ForceDiff_arr) - (HorizonValue_int+1)):
                continue
            else:
                PrevIdx_arr = np.empty(shape=[0])
                FutIdx_arr = np.empty(shape=[0])
                for x in range(HorizonValue_int):
                    PrevIdx_arr = np.append(PrevIdx_arr,(counter-(x+1)))
                    FutIdx_arr = np.append(FutIdx_arr,(counter+(x+1)))
                PrevValsForceDiff = np.empty(shape=[0])
                for PrevIdx in PrevIdx_arr:
                    PrevValsForceDiff = np.append(PrevValsForceDiff,ForceDiff_arr[int(PrevIdx)])
                FutValsForceDiff = np.empty(shape=[0])
                for FutIdx in FutIdx_arr:
                    FutValsForceDiff = np.append(FutValsForceDiff,ForceDiff_arr[int(FutIdx)])
                AvgPrevValsForceDiff = np.average(PrevValsForceDiff)
                AvgFutValsForceDiff = np.average(FutValsForceDiff)
                if AvgPrevValsForceDiff > AvgFutValsForceDiff:
                    InflectionIdx_int = counter
                    return(InflectionIdx_int)

    def PlotInflectionPoint(self,cst_arr,InflectionIdx_int,HorizonValue_int):
        """
        This function takes the compressive strength testing data array, and
        produces an overview graphic.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
        Outputs:
            Two graphics; one stress over strain and the other change in stress over strain
            (both of these are annotated with the initial inflection point)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]
        stresschange_arr = cst_arr[5]

        HighHorizonIdx_int = InflectionIdx_int + HorizonValue_int
        LowHorizonIdx_int = InflectionIdx_int - HorizonValue_int

        figure,([plot1,plot2]) = plt.subplots(2,1, figsize=(10, 5))

        plot1.scatter(strain_arr,stress_arr,s=1)
        plot1.set_ylabel("Stress (N mm^-2)")
        plot1.axvline(x=strain_arr[InflectionIdx_int])
        plot1.scatter(strain_arr[LowHorizonIdx_int:HighHorizonIdx_int],stress_arr[LowHorizonIdx_int:HighHorizonIdx_int],s=1)
        plot1.set_xlim(left=0,right=np.max(strain_arr))
        plot1.set_ylim(bottom=0,top=np.max(stress_arr))

        plot2.scatter(strain_arr,stresschange_arr,s=1)
        plot2.set_xlabel("Strain")
        plot2.set_ylabel("Stress Change (N mm^-2)")
        plot2.axvline(x=strain_arr[InflectionIdx_int])
        plot2.scatter(strain_arr[LowHorizonIdx_int:HighHorizonIdx_int],stresschange_arr[LowHorizonIdx_int:HighHorizonIdx_int],s=1)
        plot2.set_xlim(left=0,right=np.max(strain_arr))
        plot2.set_ylim(bottom=0,top=np.max(stresschange_arr))
        plt.show()

    def InflectionTangentFinder(self,cst_arr,InflectionIdx_int,HorizonValue_int):
        """
        This function takes the compressive strength testing data array, the location of the
        initial inflection, and the horizon value so that it may find the gradient of a tangent
        set against the curve at this point on the stress strain graph. This gradient is equal to
        the young's modulus of the material.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
            InflectionIdx_int = integer, the index value within the cst_arr of the initial inflection point.
            HorizonValue_int = integer, the number of points backwards and forwards the algorithm can 'look'.
        Returns:
            FittedTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the tangent.
            YoungsModulus_flt = float, the 'm' value from the FittedTangentCoeffs_arr, which is the Young's Modulus.
        """
        import numpy as np
        HighHorizonIdx_int = InflectionIdx_int + HorizonValue_int
        LowHorizonIdx_int = InflectionIdx_int - HorizonValue_int

        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]

        FittedTangentCoeffs_arr = np.polyfit(strain_arr[LowHorizonIdx_int:HighHorizonIdx_int],stress_arr[LowHorizonIdx_int:HighHorizonIdx_int],1)
        YoungsModulus_flt = float(FittedTangentCoeffs_arr[0])

        return FittedTangentCoeffs_arr,YoungsModulus_flt

    def PlotTangent(self,cst_arr,FittedTangentCoeffs_arr):
        """
        Function which takes the compressive strength testing data array and the fitted tangent coefficients,
        so that it can produce a graphic of stress vs strain with the tangent overlain. This is a sanity check
        to make sure the algorithm is fitting line correctly- this line's gradient is equal to the vital
        parameter of young's modulus.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
            FittedTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the tangent.
        Outputs:
            A single graphic of stress vs strain with a tangent drawn on it.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]

        HighestStrain_flt = np.max(strain_arr)
        TangentCutOffPoint_flt = HighestStrain_flt / 2

        ArtificialStrainTangentVals_arr = np.linspace(0,TangentCutOffPoint_flt,100)
        ArtificialStressTangentVals_arr = np.zeros(shape=(0,1))
        for ArtificialStrainTangentVal in ArtificialStrainTangentVals_arr:
            ArtificialStressTangentVals_arr = np.append(arr=ArtificialStressTangentVals_arr,values=((FittedTangentCoeffs_arr[0] * ArtificialStrainTangentVal) + FittedTangentCoeffs_arr[1]))

        figure,(plot1) = plt.subplots(1,1, figsize=(10, 3))

        plot1.scatter(cst_arr[3],cst_arr[4],s=1)
        plot1.set_xlabel("Strain")
        plot1.set_ylabel("Stress (N mm^-2)")
        plot1.scatter(ArtificialStrainTangentVals_arr,ArtificialStressTangentVals_arr,s=1)
        plot1.set_xlim(left=0,right=np.max(strain_arr))
        plot1.set_ylim(bottom=0,top=np.max(stress_arr))
        plt.show()

    def PlotTangentWithProportionalLimit(self,cst_arr,FittedTangentCoeffs_arr,PropLimitStress,PropLimitStrain):
        """
        Function which takes the compressive strength testing data array and the fitted tangent coefficients,
        so that it can produce a graphic of stress vs strain with the tangent overlain. This is a sanity check
        to make sure the algorithm is fitting line correctly- this line's gradient is equal to the vital
        parameter of young's modulus.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
            FittedTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the tangent.
        Outputs:
            A single graphic of stress vs strain with a tangent drawn on it.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]

        HighestStrain_flt = np.max(strain_arr)
        TangentCutOffPoint_flt = HighestStrain_flt / 2

        ArtificialStrainTangentVals_arr = np.linspace(0,TangentCutOffPoint_flt,100)
        ArtificialStressTangentVals_arr = np.zeros(shape=(0,1))
        for ArtificialStrainTangentVal in ArtificialStrainTangentVals_arr:
            ArtificialStressTangentVals_arr = np.append(arr=ArtificialStressTangentVals_arr,values=((FittedTangentCoeffs_arr[0] * ArtificialStrainTangentVal) + FittedTangentCoeffs_arr[1]))

        figure,(plot1) = plt.subplots(1,1, figsize=(10, 3))

        plot1.scatter(cst_arr[3],cst_arr[4],s=1)
        plot1.scatter(PropLimitStrain,PropLimitStress,s=20)
        plot1.set_xlabel("Strain")
        plot1.set_ylabel("Stress (N mm^-2)")
        plot1.scatter(ArtificialStrainTangentVals_arr,ArtificialStressTangentVals_arr,s=1)
        plot1.set_xlim(left=0,right=np.max(strain_arr))
        plot1.set_ylim(bottom=0,top=np.max(stress_arr))
        plt.show()

    def OffsetTangentFinder(self,cst_arr,FittedTangentCoeffs_arr,OffsetValue_DecPct_flt):
        """
        Function which takes the compressive strength testing data array, the fitted tangent coefficients,
        and an offset value and provides the coefficients for a line which remains parralel but offset
        along by a positive amount along the strain axis.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
            FittedTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the tangent.
            OffsetValue_DecPct_flt = float, the decimalised percentage offset in the positive direction along the strain axis.
        Returns:
            OffsetTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the offset tangent.
        """
        import numpy as np
        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]

        HighestStrain_flt = np.max(strain_arr)
        TangentCutOffPoint_flt = HighestStrain_flt / 2

        ArtificialStrainTangentVals_arr = np.linspace(0,TangentCutOffPoint_flt,100)
        ArtificialStressTangentVals_arr = np.zeros(shape=(0,1))
        for ArtificialStrainTangentVal in ArtificialStrainTangentVals_arr:
            ArtificialStressTangentVals_arr = np.append(arr=ArtificialStressTangentVals_arr,values=((FittedTangentCoeffs_arr[0] * ArtificialStrainTangentVal) + FittedTangentCoeffs_arr[1]))

        ArtificialOffsetStrainTangentVals_arr = np.zeros(shape=(0,1))
        for ArtificialStrainTangentVal_arr in ArtificialStrainTangentVals_arr:
            ArtificialOffsetStrainTangentVals_arr = np.append(arr=ArtificialOffsetStrainTangentVals_arr,values=(ArtificialStrainTangentVal_arr + OffsetValue_DecPct_flt))

        OffsetTangentCoeffs_arr = np.polyfit(ArtificialOffsetStrainTangentVals_arr,ArtificialStressTangentVals_arr,1)

        return OffsetTangentCoeffs_arr

    def PlotOffsetTangent(self,cst_arr,FittedTangentCoeffs_arr,OffsetTangentCoeffs_arr):
        """
        Function which takes the compressive strength testing data array and the fitted tangent coefficients,
        so that it can produce a graphic of stress vs strain with the tangent overlain. This is a sanity check
        to make sure the algorithm is fitting line correctly- this line's gradient is equal to the vital
        parameter of young's modulus.
        Takes:
            cst_arr = numpy array, compressive strength test data array.
            FittedTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the tangent.
            OffsetTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the offset tangent.
        Outputs:
            A single graphic of stress vs strain with a tangent and the offset tangent drawn on it.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]

        HighestStrain_flt = np.max(strain_arr)
        TangentCutOffPoint_flt = HighestStrain_flt / 2

        ArtificialStrainTangentVals_arr = np.linspace(0,TangentCutOffPoint_flt,100)
        ArtificialStressTangentVals_arr = np.zeros(shape=(0,1))
        ArtificialStressOffsetTangentVals_arr = np.zeros(shape=(0,1))
        for ArtificialStrainTangentVal in ArtificialStrainTangentVals_arr:
            ArtificialStressTangentVals_arr = np.append(arr=ArtificialStressTangentVals_arr,values=((FittedTangentCoeffs_arr[0] * ArtificialStrainTangentVal) + FittedTangentCoeffs_arr[1]))
            ArtificialStressOffsetTangentVals_arr = np.append(arr=ArtificialStressOffsetTangentVals_arr,values=((OffsetTangentCoeffs_arr[0] * ArtificialStrainTangentVal) + OffsetTangentCoeffs_arr[1]))

        figure,(plot1) = plt.subplots(1,1, figsize=(10, 3))

        plot1.scatter(cst_arr[3],cst_arr[4],s=1)
        plot1.set_xlabel("Strain")
        plot1.set_ylabel("Stress (N mm^-2)")
        plot1.scatter(ArtificialStrainTangentVals_arr,ArtificialStressTangentVals_arr,s=1)
        plot1.scatter(ArtificialStrainTangentVals_arr,ArtificialStressOffsetTangentVals_arr,s=1,c="green")
        plot1.set_xlim(left=0,right=np.max(strain_arr))
        plot1.set_ylim(bottom=0,top=np.max(stress_arr))

    def ProxyOffsetYieldStrengthFinder(self,FullResCst_arr,InflectionIdx_int,OffsetTangentCoeffs_arr):
        """
        Function which finds the crossover point between the offset tangent and the
        curve on the stress-strain diagram.
        Takes:
            FullResCst_arr = numpy array, compressive strength test data array. (NO DOWNSAMPLING)
            InflectionIdx_int = integer, the index of the point which sits on the initial inflection.
        Returns:
            OffsetYieldPointIdx_int = integer, the index of the crossover point, and therewith the
                proxy offset yield strength.
            OffsetYieldStrength_flt = float, the offset yield strength predicted by the model.
        """

        strain_lis = list(FullResCst_arr[3])
        stress_lis = list(FullResCst_arr[4])

        for counter,(stress_sca,strain_sca) in enumerate(zip(stress_lis,strain_lis)):
            if counter < InflectionIdx_int:
                continue
            else:
                offsettedlineyposition = (OffsetTangentCoeffs_arr[0] * strain_sca) + OffsetTangentCoeffs_arr[1]
                if offsettedlineyposition < stress_sca:
                    continue
                else:
                    OffsetYieldPointIdx_int = int(counter)
                    OffsetYieldStrength_flt = stress_lis[OffsetYieldPointIdx_int]
                    return OffsetYieldPointIdx_int,OffsetYieldStrength_flt
                    break

    def PlotOffsetYieldStrengthTangentCrossover(self,FullResCst_arr,OffsetYieldPointIdx_int,FittedTangentCoeffs_arr,OffsetTangentCoeffs_arr):
        """
        Function which takes the compressive strength testing data array and the offset yield points
        index position to draw a graph illustrating where this value sits. Another sanity check really.
        Takes:
            FullResCst_arr = numpy array, compressive strength test data array. (NO DOWNSAMPLING)
            OffsetYieldPointIdx_int = integer, the index of the crossover point, and therewith the
                proxy offset yield strength.
            FittedTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the tangent.
            OffsetTangentCoeffs_arr = array, the 'm' and 'c' values from y=mx+c describing the offset tangent.
        Outputs:
            A single graphic of stress vs strain.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        strain_arr = FullResCst_arr[3]
        stress_arr = FullResCst_arr[4]

        HighestStrain_flt = np.max(strain_arr)
        TangentCutOffPoint_flt = HighestStrain_flt / 2

        ArtificialStrainTangentVals_arr = np.linspace(0,TangentCutOffPoint_flt,100)
        # ArtificialStressTangentVals_arr = np.zeros(shape=(0,1))
        ArtificialStressOffsetTangentVals_arr = np.zeros(shape=(0,1))
        for ArtificialStrainTangentVal in ArtificialStrainTangentVals_arr:
            # ArtificialStressTangentVals_arr = np.append(arr=ArtificialStressTangentVals_arr,values=((FittedTangentCoeffs_arr[0] * ArtificialStrainTangentVal) + FittedTangentCoeffs_arr[1]))
            ArtificialStressOffsetTangentVals_arr = np.append(arr=ArtificialStressOffsetTangentVals_arr,values=((OffsetTangentCoeffs_arr[0] * ArtificialStrainTangentVal) + OffsetTangentCoeffs_arr[1]))

        figure,(plot1) = plt.subplots(1,1, figsize=(10, 3))

        plot1.scatter(FullResCst_arr[3],FullResCst_arr[4],s=1)
        plot1.set_xlabel("Strain")
        plot1.set_ylabel("Stress (N mm^-2)")
        # plot1.scatter(ArtificialStrainTangentVals_arr,ArtificialStressTangentVals_arr,s=1)
        plot1.scatter(ArtificialStrainTangentVals_arr,ArtificialStressOffsetTangentVals_arr,s=1,c="green")
        plot1.axhline(y=stress_arr[OffsetYieldPointIdx_int],c="red")
        plot1.set_xlim(left=0,right=np.max(strain_arr))
        plot1.set_ylim(bottom=0,top=np.max(stress_arr))
        plt.show()

    def UltimateCompressiveStrengthFinder(self,FullResCst_arr):
        """
        Function which takes the compressive strength testing data array and finds the ultimate compressive strength attained.
        Takes:
            FullResCst_arr = numpy array, compressive strength test data array. (NO DOWNSAMPLING)
        Outputs:
            UCS_flt = float, ultimate compressive strength in N
        """
        import numpy as np
        force_arr = FullResCst_arr[4]
        UCS_flt = np.max(force_arr)
        return UCS_flt

    def Appraisal_func(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int):
        """
        The appraisal function is called so that a user can make sure that fits of curves are correct- These fit curves
        are for the deduction of the offset yield strength from raw mechanical test data.
        CsvRawDataTrialMT_str = string, path to raw compressive strength test data.
        DiameterAvg_mm = float, average diamater of sample associated with the raw compressive strength test data.
        HeightAvg_mm = float, average height of sample associated with the raw compressive strength test data.
        DownsamplingFactor_int = integer, hyperparameter for model fitting. Governs how to downsample the raw data.
        HorizonValue_int = integer, hyperparameter for model fitting. Governs how far forward and backward an averaging tool can 'see'.
        """
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        InflectionIdx_int = self.InflectionFinder(cst_arr,HorizonValue_int)
        FittedTangentCoeffs_arr,YoungsModulus_flt = self.InflectionTangentFinder(cst_arr,InflectionIdx_int,HorizonValue_int)
        OffsetValue_DecPct_flt = 0.002
        OffsetTangentCoeffs_arr = self.OffsetTangentFinder(cst_arr,FittedTangentCoeffs_arr,OffsetValue_DecPct_flt)
        OffsetYieldPointIdx_int,OffsetYieldStrength_float = self.ProxyOffsetYieldStrengthFinder(FullResCst_arr,InflectionIdx_int,OffsetTangentCoeffs_arr)
        self.PlotStressStrain(cst_arr)
        self.PlotInflectionPoint(cst_arr,InflectionIdx_int,HorizonValue_int)
        self.PlotTangent(cst_arr,FittedTangentCoeffs_arr)
        self.PlotOffsetYieldStrengthTangentCrossover(FullResCst_arr,OffsetYieldPointIdx_int,FittedTangentCoeffs_arr,OffsetTangentCoeffs_arr)
        UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
        try:
            if UserInput_str != "y":
                raise TypeError("The curve fitting for offset strength is deemed unsuitable.")
            else:
                return OffsetYieldStrength_float
        except:
            print("Fits deemed unacceptable, please change the horizon and downsampling factors to allow for alternative fitting.")

    def YoungsModder_func(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int):
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        InflectionIdx_int = self.InflectionFinder(cst_arr,HorizonValue_int)
        FittedTangentCoeffs_arr,YoungsModulus_flt = self.InflectionTangentFinder(cst_arr,InflectionIdx_int,HorizonValue_int)
        self.PlotInflectionPoint(cst_arr,InflectionIdx_int,HorizonValue_int)
        self.PlotTangent(cst_arr,FittedTangentCoeffs_arr)
        UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
        try:
            if UserInput_str != "y":
                raise TypeError("The curve fitting has manually been deemed unsuitable.")
            else:
                return YoungsModulus_flt
        except:
            print("Fits deemed unacceptable, please change the horizon and downsampling factors to allow for alternative fitting.")

    def YoungsModderWithoutUserInput_func(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int):
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        InflectionIdx_int = self.InflectionFinder(cst_arr,HorizonValue_int)
        FittedTangentCoeffs_arr,YoungsModulus_flt = self.InflectionTangentFinder(cst_arr,InflectionIdx_int,HorizonValue_int)
        return YoungsModulus_flt

    def ProportionalLimiter_func(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int):
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        InflectionIdx_int = self.InflectionFinder(cst_arr,HorizonValue_int)
        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]
        PropLimitStrain = strain_arr[InflectionIdx_int]
        PropLimitStress = stress_arr[InflectionIdx_int]
        FittedTangentCoeffs_arr,YoungsModulus_flt = self.InflectionTangentFinder(cst_arr,InflectionIdx_int,HorizonValue_int)
        self.PlotInflectionPoint(cst_arr,InflectionIdx_int,HorizonValue_int)
        self.PlotTangentWithProportionalLimit(cst_arr,FittedTangentCoeffs_arr,PropLimitStress,PropLimitStrain)
        UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
        try:
            if UserInput_str != "y":
                raise TypeError("The curve fitting has manually been deemed unsuitable.")
            else:
                return PropLimitStress
        except:
            print("Fits deemed unacceptable, please change the horizon and downsampling factors to allow for alternative fitting.")

    def ProportionalLimiterWithoutUserInput_func(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int):
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        InflectionIdx_int = self.InflectionFinder(cst_arr,HorizonValue_int)
        strain_arr = cst_arr[3]
        stress_arr = cst_arr[4]
        PropLimitStrain = strain_arr[InflectionIdx_int]
        PropLimitStress = stress_arr[InflectionIdx_int]
        FittedTangentCoeffs_arr,YoungsModulus_flt = self.InflectionTangentFinder(cst_arr,InflectionIdx_int,HorizonValue_int)
        return PropLimitStress
        

    def UltimateCompressiveStrengther_func(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int):
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        ucs_flt = self.UltimateCompressiveStrengthFinder(FullResCst_arr)
        self.PlotStressStrainWithPeakStress(cst_arr,ucs_flt)
        UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
        try:
            if UserInput_str != "y":
                raise TypeError("The curve fitting has manually been deemed unsuitable.")
            else:
                return ucs_flt
        except:
            print("Fits deemed unacceptable, please change the horizon and downsampling factors to allow for alternative fitting.")

    def UltimateCompressiveStrengtherWithoutUserInput_func(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int):
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        ucs_flt = self.UltimateCompressiveStrengthFinder(FullResCst_arr)
        return ucs_flt

    def Penaliser_func(self,OptimisationSetup_obj):
        import numpy as np
        PriorYM_lis = []
        PriorUCS_lis = []
        dim_df = self.DataframeGetter_func(OptimisationSetup_obj.RawDataMTDims_str)
        for i in range(0,OptimisationSetup_obj.NoOfPriorSamples_int):
            DownsamplingFactor_int = OptimisationSetup_obj.DownsamplingFactor_int
            HorizonValue_int = OptimisationSetup_obj.HorizonValue_int
            AvgDiameter_flt = (dim_df.at[i,'Diameter1_mm'] + dim_df.at[i,'Diameter2_mm'] + dim_df.at[i,'Diameter3_mm']) / 3
            AvgHeight_flt = (dim_df.at[i,'Height1_mm'] + dim_df.at[i,'Height2_mm'] + dim_df.at[i,'Height3_mm']) / 3
            csv_str = OptimisationSetup_obj.RawDataMT_str + f"/{i}" + ".csv"
            PriorYM_lis.append(self.YoungsModderWithoutUserInput_func(csv_str,AvgDiameter_flt,AvgHeight_flt,DownsamplingFactor_int,HorizonValue_int))
            PriorUCS_lis.append(self.UltimateCompressiveStrengtherWithoutUserInput_func(csv_str,AvgDiameter_flt,AvgHeight_flt,DownsamplingFactor_int))
        AvUCS_flt = np.average(PriorUCS_lis)
        AvYM_flt = np.average(PriorYM_lis)
        StdUCS_flt = np.std(PriorUCS_lis)
        StdYM_flt = np.std(PriorYM_lis)
        return AvUCS_flt,AvYM_flt,StdUCS_flt,StdYM_flt

    def AdditivePenalisedUltimateCompressiveStrengthVsYoungsModulus_func(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int,PenalisationFactor_flt):
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        ucs_flt = self.UltimateCompressiveStrengthFinder(FullResCst_arr)
        InflectionIdx_int = self.InflectionFinder(cst_arr,HorizonValue_int)
        FittedTangentCoeffs_arr,YoungsModulus_flt = self.InflectionTangentFinder(cst_arr,InflectionIdx_int,HorizonValue_int)

        print(f"{ucs_flt} + {YoungsModulus_flt}")
        print(f"{ucs_flt} + {(YoungsModulus_flt*PenalisationFactor_flt)}")
        print(f"{ucs_flt + (YoungsModulus_flt*PenalisationFactor_flt)}")

        self.PlotStressStrainWithPeakStress(cst_arr,ucs_flt)
        self.PlotInflectionPoint(cst_arr,InflectionIdx_int,HorizonValue_int)
        self.PlotTangent(cst_arr,FittedTangentCoeffs_arr)

        UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
        try:
            if UserInput_str != "y":
                raise TypeError("The curve fitting has manually been deemed unsuitable.")
            else:
                return ucs_flt+(YoungsModulus_flt*PenalisationFactor_flt)
        except:
            print("Fits deemed unacceptable, please change the horizon and downsampling factors to allow for alternative fitting.")

    def CSTARRAYRETURNER(self,CsvRawDataTrialMT_str,DiameterAvg_mm,HeightAvg_mm,DownsamplingFactor_int,HorizonValue_int):
        MiscMethods_obj = MiscMethods_class()
        cst_df = self.DataframeGetter_func(CsvRawDataTrialMT_str)
        corr_df = self.DataframeGetter_func(self.RootPackageLocation_str + self.CorrectionalFilePath_str)
        cst_df = self.DataFrameCorrector(cst_df,corr_df)
        cst_df = self.StressCalculator_func(cst_df,DiameterAvg_mm)
        cst_df = self.StrainCalculator_func(cst_df,HeightAvg_mm)
        FullResCst_arr = self.PandasToNumpy_func(cst_df)
        cst_arr = self.PandasToDownsampledNumpy_func(cst_df,DownsamplingFactor_int)
        return cst_arr

    def SmoothedForceDisplacement_func(self,corr_df,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool):
        Force_arr = np.array(corr_df["Force (N)"])
        Position_arr = np.array(corr_df["Position (mm)"])
        window = sp.signal.windows.gaussian(M=len(Force_arr),std=StandardDeviationParameter_flt,sym=True)
        normalised_window = window/np.sum(window)
        SmoothedForce_arr = sp.ndimage.convolve1d(input=Force_arr,weights=normalised_window)
        SmoothedDisplacement_arr = sp.ndimage.convolve1d(input=Position_arr,weights=normalised_window)
        SmoothedForceDisplacement_mat = np.array([SmoothedForce_arr,SmoothedDisplacement_arr])
        corr_df["Force (N)"] = SmoothedForce_arr
        corr_df["Position (mm)"] = SmoothedDisplacement_arr
        if ToPlotOrNotToPlot_bool == True:
            # Plotting the Force-Displacement plot
            plt.scatter(Position_arr,Force_arr,color="black",s=10,label="Original Dataset")
            plt.scatter(SmoothedForceDisplacement_mat[1],SmoothedForceDisplacement_mat[0],color="yellow",s=1,label="Smoothed Dataset")
            plt.xlabel("Displacement")
            plt.ylabel("Force (N)")
        return corr_df

    def AlternativeDataFrameCorrector_func(self,cst_df,corr_df,ToPlotOrNotToPlot_bool,ChallengePlotAcceptability_bool):
        # Noting the original positions and forces for later plotting.
        OriginalXes = np.array(cst_df["Position (mm)"])
        OriginalYes = np.array(cst_df["Force (N)"])
        # Creating an array of both the forces and correctional positions from the correctional dataframe.
        CorrectionalPosition_arr = np.array(corr_df["Position (mm)"])
        CorrectionalForce_arr = np.array(corr_df["Force (N)"])
        # Creating an array of both force and uncorrected positions
        Force_arr = np.array(cst_df["Force (N)"])
        UncorrectedPosition_arr = np.array(cst_df["Position (mm)"])
        # Setting up an new array to take the corrected positions.
        CorrectedPosition_arr = np.empty(shape=0)
        # Iterating through the two sets of arrays siumultaneously to correct the positions.
        for count,(UncorrectedPosition_flt,Force_flt) in enumerate(zip(UncorrectedPosition_arr,Force_arr)):
            for count,(CorrectionalPosition_flt, CorrectionalForce_flt) in enumerate(zip(CorrectionalPosition_arr,CorrectionalForce_arr)):
                if Force_flt < CorrectionalForce_flt:
                    CorrectedPosition = UncorrectedPosition_flt - CorrectionalPosition_arr[count]
                    CorrectedPosition_arr = np.append(arr=CorrectedPosition_arr,values=CorrectedPosition)
                    break

        cst_df["CorrectedPosition (mm)"] = CorrectedPosition_arr
        ForceDisplacement_mat = np.array([cst_df["Force (N)"],cst_df["Position (mm)"]])
        # Plotting the various datasets and how the corrections have affected them.
        if ToPlotOrNotToPlot_bool == True:
            plt.figure()
            CorrectionalXes = np.array(corr_df["Position (mm)"])
            CorrectionalYes = np.array(corr_df["Force (N)"])
            CorrectedXes = np.array(cst_df["CorrectedPosition (mm)"])
            CorrectedYes = np.array(cst_df["Force (N)"])
            plt.plot(CorrectionalXes,CorrectionalYes,c="blue",label="Correctional Dataset")
            plt.plot(OriginalXes,OriginalYes,c="red",label="Raw Data")
            plt.plot(CorrectedXes,CorrectedYes,c="green",label="Corrected Data")
            plt.xlabel("Displacement (mm)")
            plt.ylabel("Force (N)")
            plt.legend()
            plt.show()
            plt.close()
        if ChallengePlotAcceptability_bool == True:
            MiscMethods_obj = MiscMethods_class()
            UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
            try:
                if UserInput_str != "y":
                    raise TypeError("Plots deemed unsuitable.")
            except:
                print("Plots deemed unacceptable, please change the arbitrary constraints to allow for alternative fitting.")
        # Returning the ForceDisplacement matrix
        return ForceDisplacement_mat
    
    def StressStrain_func(self,ForceDisplacement_mat,DiameterAvg_mm,HeightAvg_mm,ToPlotOrNotToPlot_bool,ChallengePlotAcceptability_bool):
        # Calculating the surface area of a sample.
        SurfaceArea_mm = np.pi*((DiameterAvg_mm/2)**2)
        # Calculating the stress array.
        stress_arr = ForceDisplacement_mat[0]/SurfaceArea_mm
        # Calculating the strain array.
        strain_arr = ((HeightAvg_mm-ForceDisplacement_mat[1])-HeightAvg_mm)/HeightAvg_mm
        if ToPlotOrNotToPlot_bool == True:
            # Plotting the stress-strain dataset generated.
            plt.figure()
            plt.scatter(strain_arr,stress_arr,color="black",s=10)
            plt.gca().invert_xaxis()
            plt.xlabel("Strain")
            plt.ylabel("Stress (MPa)")
            plt.show()
            plt.close()
        if ChallengePlotAcceptability_bool == True:
            MiscMethods_obj = MiscMethods_class()
            UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
            try:
                if UserInput_str != "y":
                    raise TypeError("Plots deemed unsuitable.")
            except:
                print("Plots deemed unacceptable, please change the arbitrary constraints to allow for alternative fitting.")
        # Assembling the stress-strain matrix and returning it.
        StressStrain_mat = np.array([stress_arr,strain_arr])
        return StressStrain_mat


    def SmoothedStressStrain_func(self,StressStrain_mat,StandardDeviationParameter_flt,ToPlotOrNotToPlot_bool,ChallengePlotAcceptability_bool):
        window = sp.signal.windows.gaussian(M=len(StressStrain_mat[0]),std=StandardDeviationParameter_flt,sym=True)
        normalised_window = window/np.sum(window)
        SmoothedStressStrain_mat = np.array([sp.ndimage.convolve1d(input=StressStrain_mat[0],weights=normalised_window),sp.ndimage.convolve1d(input=StressStrain_mat[1],weights=normalised_window)])
        if ToPlotOrNotToPlot_bool == True:
            # Plotting the stress-strain dataset generated.
            plt.figure()
            plt.scatter(StressStrain_mat[1],StressStrain_mat[0],color="black",s=10,label="Original Dataset")
            plt.scatter(SmoothedStressStrain_mat[1],SmoothedStressStrain_mat[0],color="yellow",s=1,label="Smoothed Dataset")
            plt.gca().invert_xaxis()
            plt.xlabel("Strain")
            plt.ylabel("Stress (MPa)")
            plt.show()
            plt.close()
        if ChallengePlotAcceptability_bool == True:
            MiscMethods_obj = MiscMethods_class()
            UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
            try:
                if UserInput_str != "y":
                    raise TypeError("Plots deemed unsuitable.")
            except:
                print("Plots deemed unacceptable, please change the arbitrary constraints to allow for alternative fitting.")
        return SmoothedStressStrain_mat

    def DerivativeStressStrain_func(self,SmoothedStressStrain_mat,StressStrain_mat,ToPlotOrNotToPlot_bool,ChallengePlotAcceptability_bool):
        StressChanges = np.gradient(SmoothedStressStrain_mat[0])
        StrainChanges = np.gradient(SmoothedStressStrain_mat[1])
        StressStrainDerivative = StressChanges/StrainChanges
        DerivativeStressStrain_mat = np.array([StressStrainDerivative,SmoothedStressStrain_mat[1]])
        if ToPlotOrNotToPlot_bool == True:
            plt.figure()
            plt.scatter(StressStrain_mat[1],(np.gradient(StressStrain_mat[0])/np.gradient(StressStrain_mat[1])),color="black",s=5,label="Derivative of Original Dataset")
            plt.scatter(DerivativeStressStrain_mat[1],DerivativeStressStrain_mat[0],color="yellow",s=1,label="Derivative of Smoothed Dataset")
            plt.gca().invert_xaxis()
            plt.xlabel("Strain")
            plt.ylabel("d Stress / d Strain")
            plt.legend()
            plt.show()
            plt.close()
        if ChallengePlotAcceptability_bool == True:
            MiscMethods_obj = MiscMethods_class()
            UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
            try:
                if UserInput_str != "y":
                    raise TypeError("Plots deemed unsuitable.")
            except:
                print("Plots deemed unacceptable, please change the arbitrary constraints to allow for alternative fitting.")
        return DerivativeStressStrain_mat

    def PeakFinder_func(self,DerivativeStrain_mat,ArbitrarySustainedRise_int,ToPlotOrNotToPlot_bool,ChallengePlotAcceptability_bool):
        # Setting an empty array into which sustained rises will be logged as we index through the stress strain curve.
        SustainedRise = np.empty(0)
        for counter,(Derivative_flt,Strain_flt) in enumerate(zip(DerivativeStrain_mat[0],DerivativeStrain_mat[1])):
            # We can check the next value for its value
            NextDerivative = DerivativeStrain_mat[0][counter+1]
            # We can find the change between the previous derivative and the current derivative
            DerivativeChange = NextDerivative-Derivative_flt
            # We can store a tentative initial point of positivity
            if DerivativeChange > 0:
                SustainedRise = np.append(SustainedRise,counter)
            # We can reset our array of positivities in the event of a negativity
            if DerivativeChange < 0:
                SustainedRise = np.empty(0)
            # We can claim victory and return the index at the arbitrary sustained rise value.
            if len(SustainedRise) == ArbitrarySustainedRise_int:
                if ToPlotOrNotToPlot_bool == True:
                    plt.figure()
                    # Plotting the derivative vs strain graph and the located peak.
                    plt.scatter(DerivativeStrain_mat[1],DerivativeStrain_mat[0],s=1)
                    plt.axvline(x=DerivativeStrain_mat[1][int(SustainedRise[0])],alpha=0.3)
                    # plt.xlim(np.min(DerivativeStrain_mat[1]),np.max(DerivativeStrain_mat[1]))
                    # plt.ylim(np.min(DerivativeStrain_mat[0]),np.max(DerivativeStrain_mat[0])*1.1)
                    plt.gca().invert_xaxis()
                    plt.xlabel("Strain")
                    plt.ylabel("d Stress / d Strain (MPa)")
                    plt.show()
                    plt.close()
                if ChallengePlotAcceptability_bool == True:
                    MiscMethods_obj = MiscMethods_class()
                    UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
                    try:
                        if UserInput_str != "y":
                            raise TypeError("Plots deemed unsuitable.")
                    except:
                        print("Plots deemed unacceptable, please change the arbitrary constraints to allow for alternative fitting.")
                # Returning the strain at the peaks location.
                return DerivativeStrain_mat[1][int(SustainedRise[0])]

    def LimitOfProportionality_func(self,StressStrain_mat,PeakStrain_flt,ToPlotOrNotToPlot_bool,ChallengePlotAcceptability_bool):
        # Finding the proximity of all points in the original dataset to the strain of the first inflection point.
        Proximities_arr = np.empty(0)
        for strain_flt in StressStrain_mat[1]:
            Proximities_arr = np.append(Proximities_arr,abs(strain_flt-PeakStrain_flt))
        # Finding the stress and strain associated with the point with the lowest proximity to the strain of the first inflection point.
        InflectionStressStrain = StressStrain_mat.T[np.argmin(Proximities_arr)]
        if ToPlotOrNotToPlot_bool == True:
            plt.figure()
            # Plotting the location  of the limit of proportionality.
            plt.plot(StressStrain_mat[1],StressStrain_mat[0])
            plt.axvline(x=InflectionStressStrain[1],alpha=0.3)
            plt.axhline(y=InflectionStressStrain[0],alpha=0.3)
            # plt.xlim(np.min(StressStrain_mat[1]),np.max(StressStrain_mat[1]))
            # plt.ylim(np.min(StressStrain_mat[0]),np.max(StressStrain_mat[0]))
            plt.gca().invert_xaxis()
            plt.xlabel("Strain")
            plt.ylabel("Stress (MPa)")
            plt.show()
            plt.close()
        if ChallengePlotAcceptability_bool == True:
            MiscMethods_obj = MiscMethods_class()
            UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
            try:
                if UserInput_str != "y":
                    raise TypeError("Plots deemed unsuitable.")
            except:
                print("Plots deemed unacceptable, please change the arbitrary constraints to allow for alternative fitting.")
        # Returning the metric of interest, limit of proportionality.
        return InflectionStressStrain[0]

    def YoungsModulus_func(self,DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,ArbitraryGradientCutoff,ToPlotOrNotToPlot_bool,ChallengePlotAcceptability_bool):
        # Here we recollect the inflection point's stress as being equal to the stress at the first derivatives peak.
        InflectionStrain_flt = PeakStrain_flt
        # This empty array will serve to hold all the proximity values between all strain points in the DerivativeVsStrain matrix
        Proximities_arr = np.empty(0)
        for StrainVal_flt in DerivativeStressStrain_mat[1]:
            Proximities_arr = np.append(Proximities_arr,abs(PeakStrain_flt-StrainVal_flt))
        # We will find the index of the point with the lowest proximity to the inflection strain.
        PeakStressIndex_int = np.argmin(Proximities_arr)
        # The aformentionend index position will then be used to retrieve the stress of that corresponding point.
        InflectionStress_flt = DerivativeStressStrain_mat[0][PeakStressIndex_int]

        # We crop the original matrix so that we only consider all the data points which preceed the inflection point.
        DatasetOne = DerivativeStressStrain_mat.T[0:PeakStressIndex_int].T
        # We invert the matrix so that we can iterate backward from the inflection point towards the start of the compressive strength test.
        DataSetOneDirectional = np.flip(DatasetOne,axis=1)

        # We will iterate back from the inflection point calculating the gradients from both the derivative and strain values.
        PrevStress = 999
        PrevStrain = 999
        for counter,(Stress_flt,Strain_flt) in enumerate(zip(DataSetOneDirectional[0],DataSetOneDirectional[1])):
            # We must skip the first point as we have no previous data with which to calculate a gradient.
            if counter == 0:
                PrevStress = Stress_flt
                PrevStrain = Strain_flt
                continue
            # In-situ gradient is calculated.
            InSituGradient = abs(Stress_flt-PrevStress)/abs(Strain_flt-PrevStrain)
            # If the in-situ gradient exceeds an arbitrary cut-ff we have set, then we stop iterating and return the stress and strain of that in-situ position.
            if InSituGradient > ArbitraryGradientCutoff:
                DataSetOneDeterminedStress = Stress_flt
                DataSetOneDeterminedStrain = Strain_flt
                break

        # We then consider the smoothed StressStrain dataset and begin by setting the upper and lower strain bounds for a Young's modulus calculation.
        # The lower bound is the previously found strain value at which the gradient of the derivative curve hits the arbitrary cur-off.
        # The upper bound is the strain at which we found the limit of proportionality (aka the inflection point).
        LowerStrainValue = DataSetOneDeterminedStrain
        UpperStrainValue = InflectionStrain_flt
        # We then iterate through the smoothed stress-strain dataset to obtain all the stresses and strains associated with points within the brackets we have set.
        ToBeFittedStresses_arr = np.empty(0)
        ToBeFittedStrains_arr = np.empty(0)
        for StressValue_flt,StrainValue_flt in zip(SmoothedStressStrain_mat[0],SmoothedStressStrain_mat[1]):
            if StrainValue_flt < LowerStrainValue and StrainValue_flt > UpperStrainValue:
                ToBeFittedStresses_arr = np.append(ToBeFittedStresses_arr,StressValue_flt)
                ToBeFittedStrains_arr = np.append(ToBeFittedStrains_arr,StrainValue_flt)
        # We can then fit a straight line through the points (so that we might obtain the gradient which will be equivalent to the Young's modulus.)
        mc = np.polyfit(ToBeFittedStrains_arr,ToBeFittedStresses_arr,deg=1)
        YoungsModulus_flt = mc[0]

        # Here we generate some data with which we may plot the line on a graph of the smoothed StressStrain data
        xs = np.linspace(np.min(ToBeFittedStrains_arr),np.max(ToBeFittedStrains_arr),100)
        ys = np.empty(0)
        def StraightLineEquation(m,c,x):
            y = m*x+c
            return y
        for x in xs:
            ys = np.append(ys,StraightLineEquation(mc[0],mc[1],x))
        # Here we check if the user would like plots to be shown and evaluated, if so we plot the vitals.
        if ToPlotOrNotToPlot_bool == True:
            # The first plot considers the work done with the DerivativeStressStrain curve.
            plt.figure()
            plt.scatter(DerivativeStressStrain_mat[1],DerivativeStressStrain_mat[0],s=1)
            plt.scatter(InflectionStrain_flt,InflectionStress_flt)
            plt.scatter(DataSetOneDeterminedStrain,DataSetOneDeterminedStress)
            plt.gca().invert_xaxis()
            plt.xlabel("Strain")
            plt.ylabel("d Stress / d Strain (MPa)")
            plt.show()
            plt.close()
            # The second plot considers the work done with the Smoothed StressStrain curve.
            plt.figure()
            plt.scatter(SmoothedStressStrain_mat[1],SmoothedStressStrain_mat[0],s=1)
            plt.scatter(LowerStrainValue,ToBeFittedStresses_arr[0])
            plt.scatter(UpperStrainValue,ToBeFittedStresses_arr[-1])
            plt.plot(xs,ys,color="yellow")
            plt.gca().invert_xaxis()
            plt.xlabel("Strain")
            plt.ylabel("Stress (MPa)")
            plt.show()
            plt.close()
        if ChallengePlotAcceptability_bool == True:
            MiscMethods_obj = MiscMethods_class()
            UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
            try:
                if UserInput_str != "y":
                    raise TypeError("Plots deemed unsuitable.")
            except:
                print("Plots deemed unacceptable, please change the arbitrary constraints to allow for alternative fitting.")
        return YoungsModulus_flt
    
    def YieldBreakPoint_func(self,DerivativeStressStrain_mat,PeakStrain_flt,SmoothedStressStrain_mat,StressStrain_mat,ArbitraryGradientCutoff,ToPlotOrNotToPlot_bool,ArbitrarySustainedRise_int,ChallengePlotAcceptability_bool):
        # Here we recollect the inflection point's stress as being equal to the stress at the first derivatives peak.
        InflectionStrain_flt = PeakStrain_flt
        # This empty array will serve to hold all the proximity values between all strain points in the DerivativeVsStrain matrix
        Proximities_arr = np.empty(0)
        for StrainVal_flt in DerivativeStressStrain_mat[1]:
            Proximities_arr = np.append(Proximities_arr,abs(PeakStrain_flt-StrainVal_flt))
        # We will find the index of the point with the lowest proximity to the inflection strain.
        PeakStressIndex_int = np.argmin(Proximities_arr)
        # The aformentionend index position will then be used to retrieve the stress of that corresponding point.
        InflectionStress_flt = DerivativeStressStrain_mat[0][PeakStressIndex_int]

        # We crop the original matrix so that we only consider all the data points which are after the inflection point.
        DatasetOne = DerivativeStressStrain_mat.T[PeakStressIndex_int::].T

        # We will iterate forward from the inflection point calculating the gradients from both the derivative and strain values.
        PrevStress = 999
        PrevStrain = 999
        for counter,(Stress_flt,Strain_flt) in enumerate(zip(DatasetOne[0],DatasetOne[1])):
            # We must skip the first point as we have no previous data with which to calculate a gradient.
            if counter == 0:
                PrevStress = Stress_flt
                PrevStrain = Strain_flt
                continue
            # In-situ gradient is calculated.
            InSituGradient = abs(Stress_flt-PrevStress)/abs(Strain_flt-PrevStrain)
            # If the in-situ gradient exceeds an arbitrary cut-off we have set, then we stop iterating and return the index of that point.
            if InSituGradient > ArbitraryGradientCutoff:
                DataSetOneDeterminedIndex = counter
                DataSetOneDeterminedStress = Stress_flt
                DataSetOneDeterminedStrain = Strain_flt
                break
        
        # Now we have the start point we must make the final dataset through which we must iterate to find the yield point's strain value
        DatasetTwo = DatasetOne.T[DataSetOneDeterminedIndex::].T

        # Setting an empty array into which sustained rises will be logged as we index through the stress strain curve.
        SustainedRise = np.empty(0)
        for counter,(Derivative_flt,Strain_flt) in enumerate(zip(DatasetTwo[0],DatasetTwo[1])):
            # We can check the next value for its value
            NextDerivative = DatasetTwo[0][counter+1]
            # print(NextDerivative)
            # We can find the change between the previous derivative and the current derivative
            DerivativeChange = NextDerivative-Derivative_flt
            # print(DerivativeChange)
            # We can store a tentative initial point of positivity
            if DerivativeChange < 0:
                SustainedRise = np.append(SustainedRise,counter)
            # We can reset our array of positivities in the event of a negativity
            if DerivativeChange > 0:
                SustainedRise = np.empty(0)
            # We can claim victory and return the index at the arbitrary sustained rise value.
            if len(SustainedRise) == ArbitrarySustainedRise_int:
                EndStrain = DatasetTwo[1][int(SustainedRise[0])]
                break
            # print(counter,len(DatasetTwo[0]))
            if counter == len(DatasetTwo[0])-2:
                EndStrain = Strain_flt
                break

        # Finding the proximity of all points in the original dataset to the strain of the second inflection point.
        Proximities_arr = np.empty(0)
        for strain_flt in StressStrain_mat[1]:
            Proximities_arr = np.append(Proximities_arr,abs(strain_flt-EndStrain))
        # Finding the stress and strain associated with the point with the lowest proximity to the strain of the first inflection point.
        InflectionStressStrain = StressStrain_mat.T[np.argmin(Proximities_arr)]

        if ToPlotOrNotToPlot_bool == True:
            # # The first plot considers the work done with the DerivativeStressStrain curve.
            plt.figure()
            plt.scatter(DerivativeStressStrain_mat[1],DerivativeStressStrain_mat[0],s=1)
            plt.scatter(InflectionStrain_flt,InflectionStress_flt)
            plt.scatter(DataSetOneDeterminedStrain,DataSetOneDeterminedStress)
            plt.axvline(x=EndStrain)
            plt.gca().invert_xaxis()
            plt.xlabel("Strain")
            plt.ylabel("Derivative (MPa)")
            plt.show()
            plt.close()
            plt.figure()
            # Plotting the location  of the yield point.
            plt.plot(StressStrain_mat[1],StressStrain_mat[0])
            plt.axvline(x=InflectionStressStrain[1],alpha=0.3)
            plt.axhline(y=InflectionStressStrain[0],alpha=0.3)
            plt.gca().invert_xaxis()
            plt.xlabel("Strain")
            plt.ylabel("Stress (MPa)")
            plt.show()
            plt.close()
        if ChallengePlotAcceptability_bool == True:
            MiscMethods_obj = MiscMethods_class()
            UserInput_str = MiscMethods_obj.StringUserInputRetriever_func("Were the fits acceptable?","y","n")
            try:
                if UserInput_str != "y":
                    raise TypeError("Plots deemed unsuitable.")
            except:
                print("Plots deemed unacceptable, please change the arbitrary constraints to allow for alternative fitting.")

        # Returning the metric of interest, yield point.
        return InflectionStressStrain[0]