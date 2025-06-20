import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from MiscMethods import MiscMethods_class

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
