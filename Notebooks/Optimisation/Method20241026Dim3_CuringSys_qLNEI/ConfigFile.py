import os
from pathlib import Path

from botorch.models.gp_regression import SingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

class OptimisationSetup_class(object):
    def __init__(self):
        # Set the name of the experiment.
        self.name = "TrialIn3D"

        # Set the tailored experiment:
        self.TailoredExperiment_str = "Method20241026Dim3"
        self.StockUPR1_UP1vsDeI_Constant_DecPct_flt = 0.686386 # 68.64% UP1 (31.36% DeI) # c1
        self.StockI1_CSvsBP_Constant_DecPct_flt = 0.8 # 80% CS (20% BP) # c2
        self.UPR1_UP1vsDeI_Bounds_DecPct_lis = [0.30,0.65] # 30-65% UP1 (70-35% DeI) # x1
        self.AUPR1_UPR1vsDMA_Bounds_DecPct_flt = [0.99,0.9995] # 99%-99.95 UPR1 (1-0.05% DMA) # x2
        self.IAUPR1_AUPR1vsI1_Bounds_DecPct_lis = [0.5,0.9] # 50-90% AUPR1 (50-10% I1) # x3
        self.CuringRegime_lis = [
            {"time_mins_flt":180,"temperature_oc_flt":80},
            {"time_mins_flt":60,"temperature_oc_flt":120}
        ]
        # Set the parameters of the parameter space:
        self.Parameters_lis = [
            {"name":"x1", "type":"range","bounds":self.UPR1_UP1vsDeI_Bounds_DecPct_lis,"value_type":"float"},
            {"name":"x2", "type":"range","bounds":self.AUPR1_UPR1vsDMA_Bounds_DecPct_flt,"value_type":"float"},
            {"name":"x3", "type":"range","bounds":self.IAUPR1_AUPR1vsI1_Bounds_DecPct_lis,"value_type":"float"}
        ]

        # Set the objective properties dictionary:
        self.Objectives_dict = {"t1":ObjectiveProperties(minimize=False)}
        # Set the objective retrieval methods vital parameters:
        self.DownsamplingFactor_int = int(5)
        self.HorizonValue_int = int(5)

        # Set the number of trials to be used to build a prior over the parameter space.
        self.NoOfPriorSamples_int = int(27)
        # Deterministic sampling method (grid,pseudorandom,quasirandom)
        self.DeterministicSamplingMethod_str = "quasirandom"

        # Set the sequential optimisation routine parameters:
        self.SequentialTechnique = "Ax"
        # Set the surrogate:
        self.Surrogate = Surrogate(SingleTaskGP)
        # Set the acquisition function:
        self.AcquisitionFunction = qLogNoisyExpectedImprovement

        # Set the number of trials to be made at each sequential iteration.
        self.NoOfTrialsPerIteration_int = int(1)
        # Set the number of sequential iterations to be carried out in this experiment.
        self.NoOfIterations_int = int(6)
        # The maximum number of trials for this experiment.
        self.MaxNoOfTrials_int = int(self.NoOfPriorSamples_int + (self.NoOfIterations_int * self.NoOfTrialsPerIteration_int))

        # The path of this file.
        self.InputFilePath_str = str(Path(os.path.abspath(__file__)))
        # The home directory for this experiment.
        self.HomeDirectoryPath_str = str(Path(os.path.abspath(__file__)).parent.absolute())
        # Set the path of the experiments Ax json file.
        self.ExperimentFilePath_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/AxExperiment.json"
        # Set the path of the parameter backup file.
        self.BackupPath_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/Backup.csv"
        # Set the path of the raw mechanical test data directory
        self.RawDataMT_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/RawDataMT"
        # Set the path of the raw mechanical test data dimensions file.
        self.RawDataMTDims_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/Dimensions.csv"