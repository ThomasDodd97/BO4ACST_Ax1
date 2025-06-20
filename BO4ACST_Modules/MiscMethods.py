import os
from pathlib import Path

class MiscMethods_class(object):
    def __init__(self):
        self.name = "MiscMethods"
        self.ChemicalDependencyFileLocation_str = "/BO4ACST_Dependencies/MiscellaneousDependencies/data_2024-10-04_ChemicalData.json"
        self.RootPackageLocation_str = str(Path(os.path.abspath(__file__)).parent.absolute().parent.absolute())

    def jsonOpener_func(self,path_str:str)->dict:
        """
        This function takes a path to a json style dictionary, converts it to a python style
        dictionary and then returns this.
        Takes:
            path_str = string, the path to the json style dictionary.
        Returns:
            foobar_dict = dictionary, the python style dictionary outputted.
        """
        import json
        with open(f'{path_str}') as f:
            foobar_dict = json.load(f)
        return foobar_dict

    def NumericUserInputRetriever_func(self,question_str:str)->float:
        """
        This function takes a question that should have some sort of numerical answer
        and poses it to the user. It then looks at a users answer, decides whether it
        is numeric or string based, before either posing the same question again, or
        returning the successfully retrieved numeric value back as a float.
        Takes:
            questions_str = string, the question to be posed to a user
        Returns:
            UserInput_flt = string, an acceptably numeric answer to the question is returned
        """
        print(question_str)
        while 1 == 1:
            try:
                UserInput_flt = float(input(question_str))
                break
            except ValueError:
                print('Please enter a number.')
        print("\t\tInput: ", UserInput_flt)
        return UserInput_flt
    
    def StringUserInputRetriever_func(self,question_str:str, ans1_str:str, ans2_str:str)->str:
        """ 
        This function takes a question that should have specific string based commands
        as answers and makes sure that the user has a command worth something before
        returning it to the main script.
        Takes:
            question_str = string, a question to be posed to a user
            ans1_str = string, a suitable answer by the user that stops the script
            ans2_str = string, a suitable answer by the user that continues the script
        Returns:
            UserInput_str = string, an acceptable text-based answer to the question is returned
        """
        while 1 == 1:
            print(question_str)
            UserInput_str = input(question_str)
            print("\t\tInput: ", UserInput_str)
            if UserInput_str == ans1_str:
                break
            elif UserInput_str == ans2_str:
                break
        return UserInput_str
    
    def CheckpointUserInputRetriever_func(self,question_str:str,ans1_str:str):
        """
        This function asks the user to confirm by a keypress that they would like for
        the program to continue- this gives the user time to carry out the actions
        previously requested of them by the program.
        Takes:
            question_str = string, a question to be posed to a user
            ans1_str = string, a suitable answer by the user that stops the script
        """
        print(question_str)
        while 1 == 1:
            try:
                UserInput_flt = str(input(question_str))
                if (ans1_str == UserInput_flt) == False:
                    raise ValueError('Input not equal to a suitable command.')
                break
            except ValueError:
                print('Please confirm. (y)')
        print("\t\tInput: ", UserInput_flt)