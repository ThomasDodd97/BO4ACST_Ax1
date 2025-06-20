import os
from pathlib import Path
from ax.service.utils.instantiation import ObjectiveProperties

from botorch.models.gp_regression import SingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

class OptimisationSetup_class(object):
    def __init__(self):
        # Set the name of the experiment.
        self.name = "TrialIn2D"

        # Set the tailored experiment:
        self.TailoredExperiment_str = "Method20241028Dim2"
        # Set the parameters of the parameter space:
        self.Parameters_lis = [
            {"name":"x1", "type":"range","bounds":list([-10.0,10.0]),"value_type":"float"},
            {"name":"x2", "type":"range","bounds":list([-10.0,10.0]),"value_type":"float"}
        ]

        # Set the objective properties dictionary:
        self.Objectives_dict = {"t1":ObjectiveProperties(minimize=True)}

        # Set the number of trials to be used to build a prior over the parameter space.
        self.NoOfPriorSamples_int = int(8)
        # Deterministic sampling method (grid,pseudorandom,quasirandom)
        self.DeterministicSamplingMethod_str = "grid"

        # Set the sequential optimisation routine parameters:
        self.SequentialTechnique = "Ax"
        # Set the surrogate:
        self.Surrogate = Surrogate(SingleTaskGP)
        # Set the acquisition function:
        self.AcquisitionFunction = qMaxValueEntropy

        # Set the number of trials to be made at each sequential iteration.
        self.NoOfTrialsPerIteration_int = int(1)
        # Set the number of sequential iterations to be carried out in this experiment.
        self.NoOfIterations_int = int(16)
        # The maximum number of trials for this experiment.
        self.MaxNoOfTrials_int = int(self.NoOfPriorSamples_int + (self.NoOfIterations_int * self.NoOfTrialsPerIteration_int))

        # The path of this file.
        self.InputFilePath_str = str(Path(os.path.abspath(__file__)))
        # The home directory for this experiment.
        self.HomeDirectoryPath_str = str(Path(os.path.abspath(__file__)).parent.absolute())
        # Set the path of the experiments Ax json file.
        self.ExperimentFilePath_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/AxExperiment.json"