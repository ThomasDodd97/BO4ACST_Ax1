import ipywidgets as widgets
from ipywidgets import interactive
from mpl_toolkits import mplot3d
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from ax.modelbridge.registry import Models
from ax.core import observation
from ax.core import data
import plotly.graph_objects as go
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

class PlottingMethods_class(object):
    def __init__(self):
        self.name = "Class of PlottingMethods"
    def InteractiveSamplesOverviewPlot(self,AxClient_obj,OptimisationSetup_obj):
        # Get the data to be handled.
        df = AxClient_obj.get_trials_data_frame()
        # Get the names of all the parameters.
        ParameterNames_lis = []
        for i in OptimisationSetup_obj.Parameters_lis:
            ParameterNames_lis.append(i['name'])

        # Get the index values of the prior and sequential samples.
        PriorIdx_lis = []
        SequentialIdx_lis = []
        for i in df.index:
            if i <= OptimisationSetup_obj.NoOfPriorSamples_int - 1:
                PriorIdx_lis.append(i)
            elif i <= OptimisationSetup_obj.MaxNoOfTrials_int:
                SequentialIdx_lis.append(i)

        # Create a blank column of colours.
        df['Color'] = None

        # Set the generation methods accordingly.
        for i in PriorIdx_lis:
            df.at[i,'generation_method'] = OptimisationSetup_obj.DeterministicSamplingMethod_str
        
        for i in df.index:
            if df.at[i,'generation_method'] == "grid":
                df.at[i,'Color'] = str('#588157')
            elif df.at[i,'generation_method'] == "pseudorandom":
                df.at[i,'Color'] = str('#3a5a40')
            elif df.at[i,'generation_method'] == "quasirandom":
                df.at[i,'Color'] = str('#344e41')
            elif df.at[i,'generation_method'] == "BoTorch":
                df.at[i,'Color'] = str('#669bbc')
            elif df.at[i,'generation_method'] == "Manual":
                df.at[i,'Color'] = str('#669bbc')
        
        SequentialIterationsToDisplayStep_int = OptimisationSetup_obj.NoOfTrialsPerIteration_int
        SequentialIterationsToDisplayMin_int = OptimisationSetup_obj.NoOfPriorSamples_int
        SequentialIterationsToDisplayMax_int = SequentialIterationsToDisplayMin_int + len(SequentialIdx_lis)

        if len(OptimisationSetup_obj.Parameters_lis) == 1:
            plt.close("all")
            print("1D Interactive Plot")

            def plotter(PointsToDisplay_int):
                IndexPoints_lis = []
                for i in range(int(PointsToDisplay_int)):
                    IndexPoints_lis.append(i)
                Specific_df = df.iloc[IndexPoints_lis]
        
                fig=plt.figure(figsize=[5,1])
                ax=plt.axes()

                x = np.array(Specific_df[f'{ParameterNames_lis[0]}'])
                y = np.zeros(len(x))
                c = np.array(Specific_df['Color'].tolist())
                p = ax.scatter(x,y,c=c,alpha=1)
                ax.set_xlabel(r"$x_1$",fontsize=20)
                ax.set_xlim(OptimisationSetup_obj.Parameters_lis[0]['bounds'][0],OptimisationSetup_obj.Parameters_lis[0]['bounds'][1])
                ax.get_yaxis().set_visible(False)
                plt.tight_layout()
                plt.show()

            iplot=interactive(plotter,PointsToDisplay_int= widgets.IntSlider(min=SequentialIterationsToDisplayMin_int,max=SequentialIterationsToDisplayMax_int,step=SequentialIterationsToDisplayStep_int))
            return iplot


        elif len(OptimisationSetup_obj.Parameters_lis) == 2:
            plt.close("all")
            print("2D Interactive Plot")
            def plotter(PointsToDisplay_int):
                IndexPoints_lis = []
                for i in range(int(PointsToDisplay_int)):
                    IndexPoints_lis.append(i)
                Specific_df = df.iloc[IndexPoints_lis]
        
                fig=plt.figure(figsize=[5,5])
                ax=plt.axes()

                x = np.array(Specific_df[f'{ParameterNames_lis[0]}'])
                y = np.array(Specific_df[f'{ParameterNames_lis[1]}'])
                c = np.array(Specific_df['Color'].tolist())
                p = ax.scatter(x,y,c=c,alpha=1)

                ax.set_xlabel(r"$x_1$",fontsize=20)
                ax.set_ylabel(r'$x_2$',fontsize=20)
                ax.set_xlim(OptimisationSetup_obj.Parameters_lis[0]['bounds'][0],OptimisationSetup_obj.Parameters_lis[0]['bounds'][1])
                ax.set_ylim(OptimisationSetup_obj.Parameters_lis[1]['bounds'][0],OptimisationSetup_obj.Parameters_lis[1]['bounds'][1])
                plt.tight_layout()
                plt.show()
            iplot=interactive(plotter,PointsToDisplay_int= widgets.IntSlider(min=SequentialIterationsToDisplayMin_int,max=SequentialIterationsToDisplayMax_int,step=SequentialIterationsToDisplayStep_int))
            return iplot

        elif len(OptimisationSetup_obj.Parameters_lis) == 3:
            plt.close("all")
            print("3D Interactive Plot")

            def plotter(E,A,PointsToDisplay_int):
                IndexPoints_lis = []
                for i in range(int(PointsToDisplay_int)):
                    IndexPoints_lis.append(i)
                Specific_df = df.iloc[IndexPoints_lis]

                fig=plt.figure(figsize=[10,10])
                ax=plt.axes(projection='3d')

                x = np.array(Specific_df[f'{ParameterNames_lis[0]}'])
                y = np.array(Specific_df[f'{ParameterNames_lis[1]}'])
                z = np.array(Specific_df[f'{ParameterNames_lis[2]}'])
                c = np.array(Specific_df['t1'].tolist())

                p = ax.scatter(x,y,z,c=c,cmap="plasma",s=50)

                ax.view_init(elev=E,azim=A)
                ax.set_xlabel(r"$x_1$",fontsize=20)
                ax.set_ylabel(r'$x_2$',fontsize=20)
                ax.set_zlabel(r'$x_3$',fontsize=20)
                ax.set_box_aspect(aspect=None, zoom=0.8)
                ax.set_xlim(OptimisationSetup_obj.Parameters_lis[0]['bounds'][0],OptimisationSetup_obj.Parameters_lis[0]['bounds'][1])
                ax.set_ylim(OptimisationSetup_obj.Parameters_lis[1]['bounds'][0],OptimisationSetup_obj.Parameters_lis[1]['bounds'][1])
                ax.set_zlim(OptimisationSetup_obj.Parameters_lis[2]['bounds'][0],OptimisationSetup_obj.Parameters_lis[2]['bounds'][1])
                
                plt.colorbar(p,shrink=0.5)
                plt.show()

            iplot=interactive(plotter,E=widgets.IntSlider(value=25, description='Elevation', max=90, min=-90, step=5),A = widgets.IntSlider(value=55, description='Azimuth', max=90, min=-90, step=5),PointsToDisplay_int= widgets.IntSlider(min=SequentialIterationsToDisplayMin_int,max=SequentialIterationsToDisplayMax_int,step=SequentialIterationsToDisplayStep_int))
            return iplot
    def InteractiveMeanFunctionPlot(self,AxClient_obj,OptimisationSetup_obj,TypeOfJob_str,Resolution_int):
        # Get the data to be handled.
        df = AxClient_obj.get_trials_data_frame()
        # Get the names of all the parameters.
        ParameterNames_lis = []
        for i in OptimisationSetup_obj.Parameters_lis:
            ParameterNames_lis.append(i['name'])
        if len(OptimisationSetup_obj.Parameters_lis) == 1:
            print("1D Interactive Plot")
        elif len(OptimisationSetup_obj.Parameters_lis) == 2:
            print("2D Interactive Plot")
        elif len(OptimisationSetup_obj.Parameters_lis) == 3:
            print("3D Interactive Plot")

            df2 = AxClient_obj.get_trials_data_frame()
            df2["mean"] = df2["t1"]
            df2["sem"] = 0
            df2["metric_name"] = "t1"
            df2.drop(axis=0, columns=['t1', 'trial_status', 'generation_method'],inplace=True)

            CustomData_obj = data.custom_data_class(column_data_types={
                    "arm_name": str,
                    "metric_name": str,
                    "mean": np.float64,
                    "sem": np.float64,
                    "trial_index": np.int64,
                    "start_time": pd.Timestamp,
                    "end_time": pd.Timestamp,
                    "n": np.int64,
                    "frac_nonnull": np.float64,
                    "random_split": np.int64,
                    "fidelities": str,  # Dictionary stored as json
                    "x1": np.float64,
                    "x2": np.float64,
                    "x3": np.float64,
                },required_columns={"trial_index","arm_name","mean","sem","x1","x2","x3","metric_name"},)
            
            data_obj = CustomData_obj(df=df2,description="Dataframe of stuff")
            data_obj.df
            
            model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
                experiment=AxClient_obj.experiment,
                data=data_obj,
            )

            ObservationFeatures_lis = []
            n = Resolution_int
            # Sampling the Mean Function
            # 30 = 0:27
            # 40 = 1:03
            # 50 = 2:00
            # 60 = 3:32
            # Sampling the Acquisition Function
            # 20 = 1:24
            # 30 = 4:27
            # 40 = 10:58
            # 50 = 21:00

            x1_arr = np.linspace(OptimisationSetup_obj.Parameters_lis[0]['bounds'][0],OptimisationSetup_obj.Parameters_lis[0]['bounds'][1],n)
            x2_arr = np.linspace(OptimisationSetup_obj.Parameters_lis[1]['bounds'][0],OptimisationSetup_obj.Parameters_lis[1]['bounds'][1],n)
            x3_arr = np.linspace(OptimisationSetup_obj.Parameters_lis[2]['bounds'][0],OptimisationSetup_obj.Parameters_lis[2]['bounds'][1],n)

            three_lis = []
            for x2_sca in x2_arr:
                two_lis = []
                for x1_sca in x1_arr:
                    one_lis = []
                    for x3_sca in x3_arr:
                        t = np.datetime64("now")
                        if TypeOfJob_str == "mean":
                            pred_lis = model_bridge_with_GPEI.predict([observation.ObservationFeatures(parameters={f"{OptimisationSetup_obj.Parameters_lis[0]['name']}":x1_sca,f"{OptimisationSetup_obj.Parameters_lis[1]['name']}":x2_sca,f"{OptimisationSetup_obj.Parameters_lis[2]['name']}":x3_sca},start_time=t,end_time=t)])
                            pred_sca = pred_lis[0]["t1"][0]
                        elif TypeOfJob_str == "covariance":
                            pred_lis = model_bridge_with_GPEI.predict([observation.ObservationFeatures(parameters={f"{OptimisationSetup_obj.Parameters_lis[0]['name']}":x1_sca,f"{OptimisationSetup_obj.Parameters_lis[1]['name']}":x2_sca,f"{OptimisationSetup_obj.Parameters_lis[2]['name']}":x3_sca},start_time=t,end_time=t)])
                            pred_sca = pred_lis[1]["t1"]["t1"][0] # Covariance evaluations
                        elif TypeOfJob_str == "acquisition":
                            pred_lis = model_bridge_with_GPEI.evaluate_acquisition_function([observation.ObservationFeatures(parameters={f"{OptimisationSetup_obj.Parameters_lis[0]['name']}":x1_sca,f"{OptimisationSetup_obj.Parameters_lis[1]['name']}":x2_sca,f"{OptimisationSetup_obj.Parameters_lis[2]['name']}":x3_sca},start_time=t,end_time=t)])
                            pred_sca = pred_lis[0] # Acquisition function evaluations
                        one_lis.append(pred_sca)
                    two_lis.append(one_lis)
                three_lis.append(two_lis)

            three_arr = np.array(three_lis)
            three_arr

            X, Y, Z = np.meshgrid(np.linspace(OptimisationSetup_obj.Parameters_lis[0]['bounds'][0],OptimisationSetup_obj.Parameters_lis[0]['bounds'][1],n), np.linspace(OptimisationSetup_obj.Parameters_lis[1]['bounds'][0],OptimisationSetup_obj.Parameters_lis[1]['bounds'][1],n), np.linspace(OptimisationSetup_obj.Parameters_lis[2]['bounds'][0],OptimisationSetup_obj.Parameters_lis[2]['bounds'][1],n))
            dataset = three_arr

            fig = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=dataset.flatten(),
                isomin=dataset.min(),
                isomax=dataset.max(),
                opacity=0.1, # needs to be small to see through all surfaces
                surface_count=20, # needs to be a large number for good volume rendering
                colorscale='plasma',
                ))

            fig.update_layout(
                autosize=False,
                width=750, 
                height=750,
                margin=dict(l=65, r=50, b=65, t=90),
                scene=dict(
                    xaxis_title='x1',
                    yaxis_title='x2',
                    zaxis_title='x3',
                ),
            )

            fig.show()