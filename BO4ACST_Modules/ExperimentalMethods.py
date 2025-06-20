import os
import glob
import ax
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
import pandas as pd
from SamplingMethods import Sampler_class
from TailoredMethods import TailoredMethods_class
from pathlib import Path
import numpy as np

class ExperimentalMethods_class(object):
    def __init__(self):
        self.name = "ExptMethods"
    def TrialAppender_func(self,x_mat,AxClient_obj,OptimisationSetup_obj):
        """
        This function takes a set of recently generated potential trials and appends them into the
        Ax client. These trials will be left marked as RUNNING until they are COMPLETED later
        when the target value has been evaluated for each trial. This function also ensures that
        the .json file for the experiment is updated accordingly.
        Takes:
            x_mat = matrix, a numpy matrix of parameterisations to be added to the ax experiment.
            AxClient_obj = object, an object based on the contents of the experiment.json file.
                                    This object has a wide variety of functionalities, but most importantly,
                                    keeps track of the predictor and target values we have accrued thus far.
            OptimisationSetup_obj = object, an object based on the class defined in the input.py file.
                                            This provides information about the configuration of the
                                            current experiment.
        Returns:
            The current ax instance is saved, with all the new trials appended.
        """
        ParameterNames_lis = []
        for i in range(len(OptimisationSetup_obj.Parameters_lis)):
            ParameterNames_lis.append(OptimisationSetup_obj.Parameters_lis[i].get("name"))
        for j in range(len(x_mat[0])):
            AxClient_obj.attach_trial(parameters={ParameterNames_lis[i]: x_mat.T[:][j].tolist()[i] for i in range(len(ParameterNames_lis))})
        AxClient_obj.save_to_json_file(OptimisationSetup_obj.ExperimentFilePath_str)

    def DeterministicMethodExecutor_func(self,OptimisationSetup_obj)->np.ndarray:
        """
        This function selects a deterministic sampler of choice and returns some potential trials.
        Takes:
            OptimisationSetup_obj = object, an object based on the class defined in the input.py file.
                                            This provides information about the configuration of the
                                            current experiment.
        Returns:
            A matrix of the appropriate dimensions to the problem. Within this matrix are deterministically
            spaced samples from across a desired parameter space.
        """
        Sampler_obj = Sampler_class()
        if OptimisationSetup_obj.DeterministicSamplingMethod_str == "grid":
            if len(OptimisationSetup_obj.Parameters_lis) == 1:
                return Sampler_obj.one.GridSampler1D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            elif len(OptimisationSetup_obj.Parameters_lis) == 2:
                return Sampler_obj.two.GridSampler2D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            elif len(OptimisationSetup_obj.Parameters_lis) == 3:
                return Sampler_obj.three.GridSampler3D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            else:
                print("Dimensions out of range...")
        elif OptimisationSetup_obj.DeterministicSamplingMethod_str == "pseudorandom":
            if len(OptimisationSetup_obj.Parameters_lis) == 1:
                return Sampler_obj.one.PseudorandomSampler1D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            elif len(OptimisationSetup_obj.Parameters_lis) == 2:
                return Sampler_obj.two.PseudorandomSampler2D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            elif len(OptimisationSetup_obj.Parameters_lis) == 3:
                return Sampler_obj.three.PseudorandomSampler3D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            else:
                print("Dimensions out of range...")
        elif OptimisationSetup_obj.DeterministicSamplingMethod_str == "quasirandom":
            if len(OptimisationSetup_obj.Parameters_lis) == 1:
                return Sampler_obj.one.QuasirandomSampler1D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            elif len(OptimisationSetup_obj.Parameters_lis) == 2:
                return Sampler_obj.two.QuasirandomSampler2D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            elif len(OptimisationSetup_obj.Parameters_lis) == 3:
                return Sampler_obj.three.QuasirandomSampler3D_func(OptimisationSetup_obj.NoOfPriorSamples_int,OptimisationSetup_obj.Parameters_lis)
            else:
                print("Dimensions out of range...")
        elif OptimisationSetup_obj.DeterministicSamplingMethod_str == "manual":
            x_mat = OptimisationSetup_obj.x_mat
            return x_mat
    
    def ExperimentInitialiser_func(self,OptimisationSetup_obj):
        """
        This function is responsible for retrieving the ax client for the current experiment
        by loading it from a local .json file. If no such file exists, then the function will
        generate a new one based on data from the the optimisation setup object.
        Takes:
            OptimisationSetup_obj = object, an object based on the class defined in the input.py file.
                                            This provides information about the configuration of the
                                            current experiment.
        Returns:
            AxClient_obj = object, an object based on the contents of the experiment.json file.
                                    This object has a wide variety of functionalities, but most importantly,
                                    keeps track of the predictor and target values we have accrued thus far.
        """
        try:
            my_file = Path(OptimisationSetup_obj.ExperimentFilePath_str)
            if my_file.is_file():
                AxClient_obj = AxClient()
                AxClient_obj = AxClient_obj.load_from_json_file(OptimisationSetup_obj.ExperimentFilePath_str)
                print("The current experiment has been initialised.")
            else:
                raise ValueError('No previous experimental Ax file available')
        except:
            print("New experiment created.")
            gs = GenerationStrategy(
                steps=[
                    GenerationStep(
                        # model=Models.GPEI,
                        model=Models.BOTORCH_MODULAR,
                        num_trials=-1,  # No limitation on how many trials should be produced from this step
                        model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                            "surrogate": OptimisationSetup_obj.Surrogate,
                            "botorch_acqf_class": OptimisationSetup_obj.AcquisitionFunction,
                        },
                    ),
                ]
            )
            # Setup the new experiment
            AxClient_obj = AxClient(generation_strategy=gs)
            AxClient_obj.create_experiment(
                name="Example",
                parameters=OptimisationSetup_obj.Parameters_lis,
                objectives=OptimisationSetup_obj.Objectives_dict
            )
            # Save the experiment as a .json
            AxClient_obj.save_to_json_file(OptimisationSetup_obj.ExperimentFilePath_str)
            print("A new experiment has been created and initialised.")
            return AxClient_obj
        return AxClient_obj

    def TrialsStartor_func(self,AxClient_obj,OptimisationSetup_obj):
        """
        This function firstly checks whether deterministically drawn trials need initialising or
        Bayesian ones by checking the ax experiments state for previously run trials (If there are
        none then the initial deterministically drawn ones are needed.). Secondly the function checks
        if any of the previous trials are running, if so it stops and awaits the users execution and
        completion of these trials before setting up some new Bayesian ones. Thirdly this function
        checks the upper limit one trials to be made during the course of this experiment- if the
        limit is about to be breached by drawing the next Bayesian iteration's worth of trials, then
        it halts the process and advises the user that the experiment is complete. Otherwise it draws
        the desired number of trials for the next Bayesian iteration.
        Takes:
            AxClient_obj = object, an object based on the contents of the experiment.json file.
                                    This object has a wide variety of functionalities, but most importantly,
                                    keeps track of the predictor and target values we have accrued thus far.
            OptimisationSetup_obj = object, an object based on the class defined in the input.py file.
                                            This provides information about the configuration of the
                                            current experiment.
        Returns:
            An updated ax experiment json file is saved if new trials have been appended. Otherwise it
            will simply print a note stating that trials are yet to be completed.
        """
        Sampler_obj = Sampler_class()
        PreviousTrials_df = AxClient_obj.get_trials_data_frame()
        if PreviousTrials_df.empty:
            print("No previous or current trials detected - initialising prior-forming trials.")
            x_mat = self.DeterministicMethodExecutor_func(OptimisationSetup_obj)
            self.TrialAppender_func(x_mat,AxClient_obj,OptimisationSetup_obj)
        else:
            if PreviousTrials_df["trial_status"].isin(["RUNNING"]).any().any():
                print("There are trials which are currently running - awaiting completion of trials.")
            else:
                CurrentTrialCount = (PreviousTrials_df["trial_index"].tolist())[-1]
                NextTrialCount = CurrentTrialCount + OptimisationSetup_obj.NoOfTrialsPerIteration_int
                if NextTrialCount >= OptimisationSetup_obj.MaxNoOfTrials_int:
                    print("The next Bayesian iteration will breach the trial limit set for this experiment.")
                    print(f"\tMaximum Number of Trials: {OptimisationSetup_obj.MaxNoOfTrials_int}")
                    print(f"\tCurrent Trial: {CurrentTrialCount}")
                    print(f"\tTrials per Bayesian Iteration: {OptimisationSetup_obj.NoOfTrialsPerIteration_int}")
                    print(f"\tCurrent Trial after next Iteration: {NextTrialCount}")
                else:
                    print("There are no trials currently running - initialising Bayesian iteration trials.")
                    if OptimisationSetup_obj.SequentialTechnique == "Ax":
                        print(OptimisationSetup_obj.NoOfTrialsPerIteration_int)
                        AxClient_obj.get_next_trials(max_trials=OptimisationSetup_obj.NoOfTrialsPerIteration_int)
                        AxClient_obj.save_to_json_file(OptimisationSetup_obj.ExperimentFilePath_str)
                    elif OptimisationSetup_obj.SequentialTechnique == "McIntersiteProj":
                        x_mat = Sampler_obj.McIntersiteProj.GetNextTrials_func(OptimisationSetup_obj,AxClient_obj)
                        x_mat = x_mat.T
                        self.TrialAppender_func(x_mat,AxClient_obj,OptimisationSetup_obj)


    def TrialsExecutor_func(self,AxClient_obj,OptimisationSetup_obj):
        """
        This function determines whether there are running trials that have yet to be executed by the operator.
        If there are, then it makes the operator execute that work.
        Takes:
            AxClient_obj = object, an object based on the contents of the experiment.json file.
                                    This object has a wide variety of functionalities, but most importantly,
                                    keeps track of the predictor and target values we have accrued thus far.
            OptimisationSetup_obj = object, an object based on the class defined in the input.py file.
                                            This provides information about the configuration of the
                                            current experiment.
        """
        # Try ascertains whether there are any trials which have not already been executed by the operator.
        try:
            PreviousTrials_df = AxClient_obj.get_trials_data_frame()
            RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
            RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
            if RunningTrials_df.empty:
                print("No trials are running. - Begin some using the TrialsStartor_func")
            else:
                print("Trials are running. - Searching for parameterisation backup file")
                my_file = Path(OptimisationSetup_obj.BackupPath_str)
                if my_file.is_file():
                    print("Backup file available!")
                    Backup_df = pd.read_csv(OptimisationSetup_obj.BackupPath_str)
                    BackupIdx_arr = np.array(Backup_df["TrialIdx"])
                    if RunningTrialsIdx_arr[-1] == BackupIdx_arr[-1]:
                        print("Running trials have been executed by the operator but uncompleted - Use the TrialsCompletor_func to complete these.")
                    else:
                        raise ValueError('Running Bayesian trials have not been executed by the operator.')
                else:
                    print("No backup file available!")
                    if RunningTrialsIdx_arr[-1]+1 == (OptimisationSetup_obj.NoOfPriorSamples_int):
                        raise ValueError('Running initial deterministic trials have not been executed by the operator.')
                    else:
                        print("Exception... (If you have done a grid sample the number of prior samples specified in config file may need re-adaping!)")
        # Except is run when it is found to be the case that the operator has not yet executed running samples.
        except:
            print("Running trials have not executed by the operator - executing...")
            TMethods_obj = TailoredMethods_class()
            if OptimisationSetup_obj.TailoredExperiment_str == "Method20241026Dim1":
                TMethods_obj.Method20241026Dim1.MixingProcedure_func(AxClient_obj,OptimisationSetup_obj)
            elif OptimisationSetup_obj.TailoredExperiment_str == "Method20241026Dim2":
                TMethods_obj.Method20241026Dim2.MixingProcedure_func(AxClient_obj,OptimisationSetup_obj)
            elif OptimisationSetup_obj.TailoredExperiment_str == "Method20241026Dim3":
                TMethods_obj.Method20241026Dim3.MixingProcedure_func(AxClient_obj,OptimisationSetup_obj)
            elif OptimisationSetup_obj.TailoredExperiment_str == "Method20250310Dim3":
                TMethods_obj.Method20250310Dim3.MixingProcedure_func(AxClient_obj,OptimisationSetup_obj)

    def TrialsCompletor_func(self,AxClient_obj,OptimisationSetup_obj):
        """
        This function searches for completed mechanical tests, extracts the offset yield strengths,
        checks with the operator that these are suitable, then adds these to the experiments .json
        file.
        This function firstly checks whether an operator has successfully executed an experimental procedure.
        If execution succesful then the raw data generated can be retrieved, processed, and the target values
        obtained. Finally, this function takes the successfully pulled target values and places these into
        the ax experiments json file alongside the associated parameterisations and trial indices.
        """
        PreviousTrials_df = AxClient_obj.get_trials_data_frame()
        RunningTrials_df = PreviousTrials_df[PreviousTrials_df['trial_status'] == "RUNNING"]
        RunningTrialsIdx_arr = np.array(RunningTrials_df['trial_index'])
        # Tailored method for checking that execution has been finalised
        TMethods_obj = TailoredMethods_class()
        t_arr = np.array([float(69.69696969696969)])
        if OptimisationSetup_obj.TailoredExperiment_str == "Method20241026Dim1":
            if TMethods_obj.Method20241026Dim1.ExecutionChecker(AxClient_obj,OptimisationSetup_obj) == True:
                t_arr = TMethods_obj.Method20241026Dim1.TargetRetriever(AxClient_obj,OptimisationSetup_obj)
        elif OptimisationSetup_obj.TailoredExperiment_str == "Method20241026Dim2":
            if TMethods_obj.Method20241026Dim2.ExecutionChecker(AxClient_obj,OptimisationSetup_obj) == True:
                t_arr = TMethods_obj.Method20241026Dim2.TargetRetriever(AxClient_obj,OptimisationSetup_obj)
        elif OptimisationSetup_obj.TailoredExperiment_str == "Method20241026Dim3":
            if TMethods_obj.Method20241026Dim3.ExecutionChecker(AxClient_obj,OptimisationSetup_obj) == True:
                t_arr = TMethods_obj.Method20241026Dim3.TargetRetriever(AxClient_obj,OptimisationSetup_obj)
        elif OptimisationSetup_obj.TailoredExperiment_str == "Method20241028Dim2":
            t_arr = TMethods_obj.Method20241028Dim2.TargetRetriever(AxClient_obj,OptimisationSetup_obj)
        elif OptimisationSetup_obj.TailoredExperiment_str == "Method20250310Dim3":
            if OptimisationSetup_obj.ObjectivesType_str == "OffsetYieldStrength":
                if TMethods_obj.Method20250310Dim3.ExecutionChecker(AxClient_obj,OptimisationSetup_obj) == True:
                    t_arr = TMethods_obj.Method20250310Dim3.TargetRetriever(AxClient_obj,OptimisationSetup_obj)
            elif OptimisationSetup_obj.ObjectivesType_str == "UltimateCompressiveStrength":
                if TMethods_obj.Method20250518Dim3.ExecutionChecker(AxClient_obj,OptimisationSetup_obj) == True:
                    t_arr = TMethods_obj.Method20250518Dim3.UCSTargetRetriever(AxClient_obj,OptimisationSetup_obj)
            elif OptimisationSetup_obj.ObjectivesType_str == "YoungsModulus":
                if TMethods_obj.Method20250518Dim3.ExecutionChecker(AxClient_obj,OptimisationSetup_obj) == True:
                    t_arr = TMethods_obj.Method20250518Dim3.YMTargetRetriever(AxClient_obj,OptimisationSetup_obj)
            elif OptimisationSetup_obj.ObjectivesType_str == "AdditivePenalisedUCSvsYM":
                if TMethods_obj.Method20250518Dim3.ExecutionChecker(AxClient_obj,OptimisationSetup_obj) == True:
                    t_arr = TMethods_obj.Method20250518Dim3.AddUCSvsYMRetriever(AxClient_obj,OptimisationSetup_obj)

        # A clause is used to check that the target retrieval was successful before attempting final completion.
        if np.sum(t_arr) == float(69.69696969696969):
            print("Target Retrieval was Unsuccessful...")
        else:
            for RunningTrialsIdx_flt,t_flt in zip(RunningTrialsIdx_arr,t_arr):
                AxClient_obj.complete_trial(trial_index=int(RunningTrialsIdx_flt),raw_data={f"{list(OptimisationSetup_obj.Objectives_dict.keys())[0]}": (t_flt)})
                # The updated ax object is saved, overwriting the old state of the ax experiment.
                AxClient_obj.save_to_json_file(OptimisationSetup_obj.ExperimentFilePath_str)