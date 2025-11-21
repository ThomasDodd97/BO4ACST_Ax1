import numpy as np
from sympy import Point3D, Plane, Line, Point2D
from scipy.stats import qmc

def PythagorasTheoremHypotenuse_func(VarA,VarO):
    """
    A function which finds the hypotenuse using pythagoras theorem.
    Takes:
        VarA = float, adjacent length.
        VarO = float, opposite length.
    Returns:
        . = float, hypotenuse length.
    """
    return np.sqrt(abs((VarA**2)+(VarO**2)))

def PythagorasTheoremAdjacentOpposite_func(VarAO,VarC):
    """
    A function which takes the hypotenuse and either the adjacent or opposite,
    and returns a short sides length.
    Takes:
        VarAO = float, adjacent or opposite length.
        VarC = float, hypotenuse length.
    Returns:
        . = float, short length of a triangle.
    """
    return np.sqrt(abs((VarC**2)-(VarAO**2)))

# Setting up the tetrahedron characteristics for sampling within:
PointA = Point3D(0, 0, 0)
PointB = Point3D(0.5, PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5), 0)
PointC = Point3D(1, 0, 0)
PointD = Point3D(0.5, PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5)*(1/3), PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5))
# Face A
A_label_str = "A"
A_color_str = "red"
A_X_coords_arr = np.array([float(PointA[0]),float(PointB[0]),float(PointC[0])])
A_Y_coords_arr = np.array([float(PointA[1]),float(PointB[1]),float(PointC[1])])
A_Z_coords_arr = np.array([float(PointA[2]),float(PointB[2]),float(PointC[2])])
A_plane = Plane(PointA, PointB, PointC)
# Face B
B_label_str = "B"
B_color_str = "blue"
B_X_coords_arr = np.array([float(PointA[0]),float(PointB[0]),float(PointD[0])])
B_Y_coords_arr = np.array([float(PointA[1]),float(PointB[1]),float(PointD[1])])
B_Z_coords_arr = np.array([float(PointA[2]),float(PointB[2]),float(PointD[2])])
B_plane = Plane(PointA, PointB, PointD)
# Face C
C_label_str = "C"
C_color_str = "green"
C_X_coords_arr = np.array([float(PointB[0]),float(PointD[0]),float(PointC[0])])
C_Y_coords_arr = np.array([float(PointB[1]),float(PointD[1]),float(PointC[1])])
C_Z_coords_arr = np.array([float(PointB[2]),float(PointD[2]),float(PointC[2])])
C_plane = Plane(PointB, PointD, PointC)
# Face D
D_label_str = "D"
D_color_str = "yellow"
D_X_coords_arr = np.array([float(PointA[0]),float(PointD[0]),float(PointC[0])])
D_Y_coords_arr = np.array([float(PointA[1]),float(PointD[1]),float(PointC[1])])
D_Z_coords_arr = np.array([float(PointA[2]),float(PointD[2]),float(PointC[2])])
D_plane = Plane(PointA, PointD, PointC)
# Line A
A_line = Line(PointA, PointB)
# Line B
B_line = Line(PointC, PointB)
# Line C
C_line = Line(PointA, PointC)

def PointWithinLineChecker_func(Point,LowerBound,UpperBound):
    if Point > LowerBound:
        if Point < UpperBound:
            return 1
        else:
            return 0
    else:
        return 0

def PointsWithinLineCounter_func(Points,LowerBound,UpperBound):
    PointsInCounter = 0
    for point in Points:
        PointsInCounter += PointWithinLineChecker_func(point,LowerBound,UpperBound)
    return PointsInCounter

def PointsWithinLineReturner_func(Points,LowerBound,UpperBound):
    PointsIn = []
    for point in Points:
        if PointWithinLineChecker_func(point,LowerBound,UpperBound) == 1:
            PointsIn.append(point)
    PointsIn = np.array(PointsIn)
    return PointsIn

def PointWithinTetrahedronChecker_func(Point,Plane1,Plane2,Plane3,Plane4):
    if Plane1.equation(x=Point[0],y=Point[1],z=Point[2])>0:
        # print("Point is without the A plane of the tetrahedron.")
        # print("Cancelling analysis.")
        return 0
    else:
        # print("Point is within the A plane of the tetrahedron.")
        # print("Continuing analysis.")
        if Plane2.equation(x=Point[0],y=Point[1],z=Point[2])>0:
            # print("Point is within the B plane of the tetrahedron.")
            # print("Continuing analysis.")
            if Plane3.equation(x=Point[0],y=Point[1],z=Point[2])>0:
                # print("Point is without the C plane of the tetrahedron.")
                # print("Cancelling analysis.")
                return 0
            else:
                # print("Point is within the C plane of the tetrahedron")
                # print("Continuing analysis.")
                if Plane4.equation(x=Point[0],y=Point[1],z=Point[2])>0:
                    # print("Point is within the D plane of the tetrahedron.")
                    # print("Succesful analysis.")
                    return 1
                else:
                    # print("Point is without the D plane of the tetrahedron.")
                    # print("Cancelling analysis.")
                    return 0
        else:
            # print("Point is without the B plane of the tetrahedron.")
            # print("Cancelling analysis.")
            return 0

def PointsWithinTetrahedronCounter_func(Points,Plane1,Plane2,Plane3,Plane4):
    PointsInCounter = 0
    for point in Points:
        PointsInCounter += PointWithinTetrahedronChecker_func(point,Plane1,Plane2,Plane3,Plane4)
    return PointsInCounter

def PointsWithinTetrahedronReturner_func(Points,Plane1,Plane2,Plane3,Plane4):
    PointsIn = []
    for point in Points:
        if PointWithinTetrahedronChecker_func(point,Plane1,Plane2,Plane3,Plane4) == 1:
            PointsIn.append(point)
    PointsIn = np.array(PointsIn)
    return PointsIn

def LinePointsRepresentativityEvaluator_func(Points):
    OverallDistance = 0
    for i_point in Points:
        centrepoint = i_point
        for j_point in Points:
            OverallDistance += abs(centrepoint-j_point)
    return OverallDistance

def TetrahedronPointsRepresentativityEvaluator_func(Points):
    OverallDistance = 0
    for i_point in Points:
        centrepoint = Point3D(i_point)
        for j_point in Points:
            testpoint = Point3D(j_point)
            OverallDistance += float(centrepoint.distance(testpoint))
    return OverallDistance

def CoordinateConverterFromXyzToQs(xyzCoords_mat):
    qs_lis = []
    for row in xyzCoords_mat.T:
        x_flt = row[0:1][0]
        q2_flt = x_flt

        y_flt = row[1:2][0]
        A_line_intersect_sympy = Line(Point3D(np.array([x_flt,float(0),float(0)])),Point3D(np.array([x_flt,float(1),float(0)]))).intersection(A_line)
        B_line_intersect_sympy = Line(Point3D(np.array([x_flt,float(0),float(0)])),Point3D(np.array([x_flt,float(1),float(0)]))).intersection(B_line)
        AB_line_intersects = [float(A_line_intersect_sympy[0][1]),float(B_line_intersect_sympy[0][1])]
        AB_True_intersect = np.min(AB_line_intersects)
        q3_flt = y_flt/AB_True_intersect

        z_flt = row[2:3][0]
        B_plane_intersect_sympy = Line(Point3D(np.array([x_flt,y_flt,float(0)])),Point3D(np.array([x_flt,y_flt,float(1)]))).intersection(B_plane)
        C_plane_intersect_sympy = Line(Point3D(np.array([x_flt,y_flt,float(0)])),Point3D(np.array([x_flt,y_flt,float(1)]))).intersection(C_plane)
        D_plane_intersect_sympy = Line(Point3D(np.array([x_flt,y_flt,float(0)])),Point3D(np.array([x_flt,y_flt,float(1)]))).intersection(D_plane)
        BCD_line_intersects = [float(B_plane_intersect_sympy[0][2]),float(C_plane_intersect_sympy[0][2]),float(D_plane_intersect_sympy[0][2])]
        BCD_True_intersect = np.min(BCD_line_intersects)
        q4_flt = z_flt / BCD_True_intersect

        qs_lis.append([q2_flt,q3_flt,q4_flt])
    qs_mat = np.array(qs_lis).T
    return qs_mat

def CoordinateConverterFromQsToXyz(qs_mat):
    xyz_lis = []
    for row in qs_mat.T:
        q2_flt = row[0:1][0]
        x_flt = q2_flt

        q3_flt = row[1:2][0]
        A_line_intersect_sympy = Line(Point3D(np.array([x_flt,0,float(0)])),Point3D(np.array([x_flt,1,float(0)]))).intersection(A_line)
        B_line_intersect_sympy = Line(Point3D(np.array([x_flt,0,float(0)])),Point3D(np.array([x_flt,1,float(0)]))).intersection(B_line)
        AB_line_intersects = [float(A_line_intersect_sympy[0][1]),float(B_line_intersect_sympy[0][1])]
        AB_True_intersect_flt = np.min(AB_line_intersects)
        y_flt = q3_flt*AB_True_intersect_flt

        q4_flt = row[2:3][0]
        B_plane_intersect_sympy = Line(Point3D(np.array([x_flt,y_flt,float(0)])),Point3D(np.array([x_flt,y_flt,float(1)]))).intersection(B_plane)
        C_plane_intersect_sympy = Line(Point3D(np.array([x_flt,y_flt,float(0)])),Point3D(np.array([x_flt,y_flt,float(1)]))).intersection(C_plane)
        D_plane_intersect_sympy = Line(Point3D(np.array([x_flt,y_flt,float(0)])),Point3D(np.array([x_flt,y_flt,float(1)]))).intersection(D_plane)
        BCD_line_intersects = [float(B_plane_intersect_sympy[0][2]),float(C_plane_intersect_sympy[0][2]),float(D_plane_intersect_sympy[0][2])]
        BCD_True_intersect = np.min(BCD_line_intersects)
        z_flt = q4_flt*BCD_True_intersect
        xyz_lis.append([x_flt,y_flt,z_flt])

    xyz_mat = np.array(xyz_lis).T
    return xyz_mat

def CoordinateConverterFromXyzToAs(xyz_mat):
    as_lis = []
    for row in xyz_mat.T:
        Point = Point3D(row)
        l1 = float(A_plane.distance(Point))
        l2 = float(B_plane.distance(Point))
        l3 = float(C_plane.distance(Point))
        l4 = float(D_plane.distance(Point))
        Distances_lis = np.array([l1,l2,l3,l4])
        TotalOfDistances_flt = np.sum(Distances_lis)
        a_lis = []
        for distance_flt in Distances_lis:
            a_lis.append(distance_flt/TotalOfDistances_flt)
        as_lis.append(a_lis)
    as_mat = np.array(as_lis).T
    return as_mat

def AsMatrixCompiler_func(as_mat,a1_arr):
    return np.concat((a1_arr.T,as_mat.T),axis=1)

def DetectNoPI1D_func(array,low_flt:float,high_flt:float)->tuple[int,np.ndarray]:
    """
    A function for detecting the number of points inside the parameter
    space of interest. In this case, a 1D line.
    Takes:
        array = array, a set of points distributed a 1D space
        low_flt = float, the lower bound for a point to be considered within the range
        high_flt = float, the higher bound below which a point is considered within the range
    Returns:
        NoPI_int = float, the number of points considered within the space
        PointsIn_arr = array, the set of points considered within the space
    """
    import numpy as np
    NoPI_int = int(0)
    PointsIn_lis = []
    for _ in array:
        if _ > low_flt and _ < high_flt:
            NoPI_int += 1
            PointsIn_lis.append(_)
    PointsIn_arr = np.array(PointsIn_lis)
    return NoPI_int,PointsIn_arr

def NearestSobolBaseFinder_func(NoSD_int:int)->tuple[int,int]:
    """
    This function takes the number of samples desired and finds the
    nearest sobol base and the number of samples that base would yield.
    Takes:
        NoSD_int = integer, the number of samples desired within the space
    Returns:
        base_int = integer, the base value required by a sobol sampler
                            deemed to yield the number of samples closest
                            to the desired number of samples.
        NoS_int = integer, the number of samples the base will yield
    """
    import numpy as np
    # Making an array of number of samples to be taken at each base_int value in sobol sequence
    SobolSampleSeq_li = []
    for _ in range(100):
        SobolSampleSeq_li.append(2 ** _)
    # Find the index (base_int) of the nearest value between sobol sequence number of samples array and the
    #  scalar for double the number of samples needed in the equilateral triangle
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx,array[idx]
    # Set the base_int value for the sobol sampler, i.e. Set base_int (which is x) where 2 ** x
    # is equal to the number of samples the sobol sequence can deal with for an attempt
    base_int,NoS_int = find_nearest(SobolSampleSeq_li,NoSD_int)
    return base_int,NoS_int

def SobolFixer(abstract_arr:np.ndarray,low_flt:float,high_flt:float):
    """
    Function takes an abstract set of sobol samples between 0 and 1
    and turns them into samples between a low and a high point.
    Takes:
        abstract_arr = array, the sobol sampled array between 0 and 1
        low_flt = float, the lower boundary of the dimensions parameter space
        high_flt = scalar, the upper boundary of the dimensions parameter space
    Returns:
        fixed_arr = array, the normalised array between lower and upper bounds of parameter space
    """
    import numpy as np
    delta_flt = high_flt - low_flt
    fixed_lis = []
    for number in abstract_arr:
        fixed_lis.append(low_flt + (number * delta_flt))
    fixed_arr = np.array(fixed_lis)
    return fixed_arr

def DetectNoPI2DSquare_func(fixed_s1_arr:np.ndarray,fixed_s2_arr:np.ndarray,dims_li:list)->tuple[int,list,list]:
    """
    A function for detecting the number of points inside the parameter
    space of interest. In this case, a 2D square.
    Takes:
        fixed_s1_arr = array, a set of points generated in and around the parameter space
        fixed_s2_arr = array, a set of points generated in and around the parameter space
        dims_li = list, a list of dimensions involved (i.e. 3 for 3D problem)
    Returns:
        NoPI_int = integer, the number of points in the parameter space
        s1PointsIn_lis = list, the points within the parameter space
        s2PointsIn_lis = list, the points within the parameter space
    """
    NoPI_int = 0
    s1PointsIn_lis = []
    s2PointsIn_lis = []
    for s1_sca,s2_sca in zip(fixed_s1_arr,fixed_s2_arr):
        if s1_sca > dims_li[0].bounds[0] and s1_sca < dims_li[0].bounds[1] and s2_sca > dims_li[1].bounds[0] and s2_sca < dims_li[1].bounds[1]:
            NoPI_int += 1
            s1PointsIn_lis.append(s1_sca)
            s2PointsIn_lis.append(s2_sca)
    return NoPI_int,s1PointsIn_lis,s2PointsIn_lis

def DetectNoPI3DCube_func(fixed_x1_arr:np.ndarray,fixed_x2_arr:np.ndarray,fixed_x3_arr:np.ndarray,Parameters_lis:list)->tuple[int,list,list,list]:
    """
    A function for detecting the number of points inside the parameter
    space of interest. In this case, a 3D cube.
    Takes:
        fixed_x1_arr = array, a set of points generated in and around the parameter space
        fixed_x2_arr = array, a set of points generated in and around the parameter space
        fixed_x3_arr = array, a set of points generated in and around the parameter space
        Parameters_lis = list, a list of dimensions involved (i.e. 3 for 3D problem)
    Returns:
        NoPI_int = integer, the number of points in the parameter space
        x1PointsIn_lis = list, the points within the parameter space
        x2PointsIn_lis = list, the points within the parameter space
        x3PointsIn_lis = list, the points within the parameter space
    """
    NoPI_int = int(0)
    x1PointsIn_lis = []
    x2PointsIn_lis = []
    x3PointsIn_lis = []
    for x1_sca,x2_sca,x3_sca in zip(fixed_x1_arr,fixed_x2_arr,fixed_x3_arr):
        if x1_sca > Parameters_lis[0].bounds[0] and x1_sca < Parameters_lis[0].bounds[1] and x2_sca > Parameters_lis[1].bounds[0] and x2_sca < Parameters_lis[1].bounds[1] and x3_sca > Parameters_lis[2].bounds[0] and x3_sca < Parameters_lis[2].bounds[1]:
            NoPI_int += 1
            x1PointsIn_lis.append(x1_sca)
            x2PointsIn_lis.append(x2_sca)
            x3PointsIn_lis.append(x3_sca)
    return NoPI_int,x1PointsIn_lis,x2PointsIn_lis,x3PointsIn_lis

class OneDim_class():
    def __init__(self):
        self.name = "Inner Class - One Dimensional Sampler"
    def GridSampler1D_func(self,NoSD_int:float,Parameters_lis:list)->np.ndarray:
        """
        A function for generating a grid sample across a 1D space.
        Takes:
            NoSD_int = integer, number of samples desired across the space.
            dim_li = list, a list of dimensions involved (i.e. 1D for 1D problem)
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        import numpy as np
        # delta_sca = Parameters_lis[0].bounds[1] - Parameters_lis[0].bounds[0]
        delta_sca = Parameters_lis[0].bounds[1] - Parameters_lis[0].bounds[0]
        segments_sca = delta_sca / (NoSD_int - 1)
        s_li = []
        for i in range(NoSD_int):
            if i == 0:
                s_li.append(Parameters_lis[0].bounds[0])
            elif i == 1:
                s_li.append(Parameters_lis[0].bounds[0] + segments_sca)
            elif i > 1:
                s_li.append(s_li[len(s_li)-1] + segments_sca)
        x_arr = np.array(s_li)
        x_mat = np.array([x_arr])
        return x_mat
    def PseudorandomSampler1D_func(self,NoSD_flt:float,Parameters_lis:list)->np.ndarray:
        """
        This is a function for generating pseudorandom sampling across a
        1D parameter space.
        Takes:
            NoSD_flt = float, number of samples desired across the space.
            dim_li = list, a list of dimensions involved (i.e. 1D for 1D problem)
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        import numpy as np
        s_li = np.random.uniform(Parameters_lis[0].bounds[0],Parameters_lis[0].bounds[1],NoSD_flt)
        x_arr = np.array(s_li)
        # x_mat = np.array([x_arr])
        return x_arr
    def QuasirandomSampler1D_func(self,NoSD_flt:float,Parameters_lis:list)->np.ndarray:
        """
        Function that samples quasirandomly from across the 1D parameter space
        in question. The method is sobolian in nature. (sobol sampling with a rejection algorithm)
        Takes:
            NoSD_flt = float, number of samples desired across the space.
            dim_li = list, a list of dimensions involved (i.e. 1D for 1D problem)
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        import numpy as np
        from scipy.stats import qmc
        delta_sca = Parameters_lis[0].bounds[1] - Parameters_lis[0].bounds[0]
        step_sca = delta_sca / 100
        i = 0
        NoPI = 0
        MaxAttempts = 1000
        while i < MaxAttempts:
            if i == 0:
                low_sca = Parameters_lis[0].bounds[0]
                high_sca = Parameters_lis[0].bounds[1]
            if (i % 250) == 0:
                low_sca = Parameters_lis[0].bounds[0]
                high_sca = Parameters_lis[0].bounds[1]
            base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_flt)
            if NoS_sca < NoSD_flt:
                base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_flt * 2)
            sampler_obj = qmc.Sobol(d=len(Parameters_lis))
            sample_arr = sampler_obj.random_base2(m=base_sca)
            abstract_arr = sample_arr[:,0]
            fixed_arr = SobolFixer(abstract_arr,low_sca,high_sca)
            NoPI_sca,PointsIn_arr = DetectNoPI1D_func(fixed_arr,Parameters_lis[0].bounds[0],Parameters_lis[0].bounds[1])
            if NoSD_flt == NoPI_sca:
                break
            low_sca = low_sca - step_sca
            high_sca = high_sca + step_sca
            i += 1
        x1_arr = np.array(PointsIn_arr)
        x_mat = np.array([x1_arr])
        return x_mat
    def Plot1D_func(self,x_mat:list):
        from matplotlib import pyplot as plt
        plt.close("all")
        plt.rcParams["figure.figsize"] = [12,1]
        plt.hlines(1,0,1)
        plt.eventplot(x_mat[0], orientation='horizontal', colors='b')
        plt.xlabel("x1")
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        plt.show()
class TwoDim_class():
    def __init__(self):
            self.name = "Inner Class - Two Dimensional Sampler"
            self.Sampler1D = OneDim_class()
    def GridSampler2D_func(self,NoSD_int:int,Parameters_lis:list)->np.ndarray:
        """
        This function grid samples across a two dimensional parameter space.
        Takes:
            Parameters_lis = list, the dimensions e.g. [x1, x2]
            NoSD_int = integer, the number of samples desired
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        import numpy as np
        dims_sca = len(Parameters_lis)
        GridSampleSeq_arr = []
        for _ in range(100):
            GridSampleSeq_arr.append(_ ** dims_sca)
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return GridSampleSeq_arr[idx]
        NoS_sca = find_nearest(GridSampleSeq_arr,NoSD_int)
        VerticePoints_sca = int(round(NoS_sca ** (1 / dims_sca),1)) # 3
        x_vars = []
        for i,dim_xx in zip(range(dims_sca),Parameters_lis):
            VarName_str = f"x_{i}_arr"
            value_arr = []
            delta_sca = dim_xx.bounds[1] - dim_xx.bounds[0]
            step_sca = delta_sca / (VerticePoints_sca - 1)
            start_sca = dim_xx.bounds[0]
            for _ in range(VerticePoints_sca):
                if _ == 0:
                    value_arr.append(start_sca)
                    start_sca = start_sca + step_sca
                elif _ > 0:
                    value_arr.append(start_sca)
                    start_sca = start_sca + step_sca
            globals()[VarName_str] = value_arr
            x_vars.append(globals()[VarName_str])
        x_arr = np.meshgrid(*x_vars)
        x_final_arr = []
        for i,array in zip(range(dims_sca),x_arr):
            VarName_str = f"x{i+1}_arr"
            value_arr = np.array(list(np.concatenate(array,axis=0).flat))
            globals()[VarName_str] = value_arr
            x_final_arr.append(globals()[VarName_str])
        print(f"Samples gridded = {NoS_sca}")
        x1_arr = np.array(x_final_arr[0])
        x2_arr = np.array(x_final_arr[1])
        x_mat = np.array([x1_arr,x2_arr])
        return x_mat
    def PseudorandomSampler2D_func(self,NoSD_flt:float,dims_li:list)->np.ndarray:
        """
        This function pseudorandomly samples from across a 2D space.
        Takes:
            NoSD_flt = float, number of samples desired across the space.
            dim_li = list, a list of dimensions involved (i.e. 2 for 2D problem)
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        x1_arr = self.Sampler1D.PseudorandomSampler1D_func(NoSD_flt,[dims_li[0]])
        x2_arr = self.Sampler1D.PseudorandomSampler1D_func(NoSD_flt,[dims_li[1]])
        x_mat = np.array([x1_arr,x2_arr])
        return x_mat
    def QuasirandomSampler2D_func(self,NoSD_flt:float,dims_li:list)->np.ndarray:
        """
        This function quasirandomly samples across a 2D space. (sobol sampling with a rejection algorithm)
        Takes:
            NoSD_flt = float, number of samples desired across the space.
            dim_li = list, a list of dimensions involved (i.e. 2 for 2D problem)
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        import numpy as np
        from scipy.stats import qmc
        high_s1_sca = dims_li[0].bounds[1]
        high_s2_sca = dims_li[1].bounds[1]
        low_s1_sca = dims_li[0].bounds[0]
        low_s2_sca = dims_li[1].bounds[0]
        delta_s1_sca = dims_li[0].bounds[1] - dims_li[0].bounds[0]
        step_s1_sca = delta_s1_sca / 100
        delta_s2_sca = dims_li[1].bounds[1] - dims_li[1].bounds[0]
        step_s2_sca = delta_s2_sca / 100
        i = 0
        NoPI = 0
        MaxAttempts = 1000
        while i < MaxAttempts:
            if i == 0 or (i % 250) == 0:
                c_low_s1_sca = low_s1_sca
                c_low_s2_sca = low_s2_sca
                c_high_s1_sca = high_s1_sca
                c_high_s2_sca = high_s2_sca
            base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_flt)
            if NoS_sca < NoSD_flt:
                base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_flt * 2)
            sampler_obj = qmc.Sobol(d=len(dims_li))
            sample_arr = sampler_obj.random_base2(m=base_sca)
            abstract_s1_arr = sample_arr[:,0]
            abstract_s2_arr = sample_arr[:,1]
            fixed_s1_arr = SobolFixer(abstract_s1_arr,c_low_s1_sca,c_high_s1_sca)
            fixed_s2_arr = SobolFixer(abstract_s2_arr,c_low_s2_sca,c_high_s2_sca)
            NoPI_sca,s1PointsIn_arr,s2PointsIn_arr, = DetectNoPI2DSquare_func(fixed_s1_arr,fixed_s2_arr,dims_li)
            if NoSD_flt == NoPI_sca:
                break
            c_low_s1_sca = c_low_s1_sca - step_s1_sca
            c_high_s1_sca = c_high_s1_sca + step_s1_sca
            c_low_s2_sca = c_low_s2_sca - step_s2_sca
            c_high_s2_sca = c_high_s2_sca + step_s2_sca
            i += 1
        s1_arr = np.array(s1PointsIn_arr)
        s2_arr = np.array(s2PointsIn_arr)
        x_mat = np.array([s1_arr,s2_arr])
        return x_mat
    def Plot2D_func(self,x_mat):
        from matplotlib import pyplot as plt
        plt.close("all")
        plt.rcParams["figure.figsize"] = [3,3]
        plt.scatter(x_mat[0],x_mat[1])
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

class ThreeDim_class():
    def __init__(self):
        self.name = "Inner Class - Three Dimensional Sampler"
        self.Sampler1D = OneDim_class()
    def GridSampler3D_func(self,NoS_desired_flt:float,Parameters_lis:list)->np.ndarray:
        """
        This function take the number of desired samples and the dimensions across
        which these should span and then grid samples them accordingly.
        Takes:
            Parameters_lis = array, the dimensions e.g. x1, x2, x3
            NoS_desired_flt = float, the number of samples desired
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        import numpy as np
        dims_sca = len(Parameters_lis)
        GridSampleSeq_arr = []
        for _ in range(100):
            GridSampleSeq_arr.append(_ ** dims_sca)
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return GridSampleSeq_arr[idx]
        NoS_sca = find_nearest(GridSampleSeq_arr,NoS_desired_flt)
        VerticePoints_sca = int(round(NoS_sca ** (1 / dims_sca),1))
        x_vars = []
        for i,dim_xx in zip(range(dims_sca),Parameters_lis):
            VarName_str = f"x_{i}_arr"
            value_arr = []
            delta_sca = dim_xx.bounds[1] - dim_xx.bounds[0]
            step_sca = delta_sca / (dims_sca - 1)
            start_sca = dim_xx.bounds[0]
            for _ in range(VerticePoints_sca):
                if _ == 0:
                    value_arr.append(start_sca)
                    start_sca = start_sca + step_sca
                elif _ > 0:
                    value_arr.append(start_sca)
                    start_sca = start_sca + step_sca
            globals()[VarName_str] = value_arr
            x_vars.append(globals()[VarName_str])
        x_arr = np.meshgrid(*x_vars)
        x_final_arr = []
        for i,array in zip(range(dims_sca),x_arr):
            VarName_str = f"x{i+1}_arr"
            value_arr = np.array(list(np.concatenate(array,axis=0).flat))
            globals()[VarName_str] = value_arr
            x_final_arr.append(globals()[VarName_str])
        print(f"Samples gridded = {NoS_sca}")
        x1_arr = np.array(x_final_arr[0])
        x2_arr = np.array(x_final_arr[1])
        x3_arr = np.array(x_final_arr[2])
        x_mat = np.array([x1_arr,x2_arr,x3_arr])
        return x_mat
    def PseudorandomSampler3D_func(self,NoSD_flt:float,dims_li:list)->np.ndarray:
        """
        This function pseudorandomly samples from across a 3D space.
        Takes:
            NoSD_flt = float, number of samples desired across the space.
            dim_li = list, a list of dimensions involved (i.e. 3 for 3D problem)
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        x1_arr = self.Sampler1D.PseudorandomSampler1D_func(NoSD_flt,[dims_li[0]])
        x2_arr = self.Sampler1D.PseudorandomSampler1D_func(NoSD_flt,[dims_li[1]])
        x3_arr = self.Sampler1D.PseudorandomSampler1D_func(NoSD_flt,[dims_li[2]])
        x_mat = np.array([x1_arr,x2_arr,x3_arr])
        return x_mat
    def QuasirandomSampler3D_func(self,NoSD_flt:float,Parameters_lis:list)->np.ndarray:
        """
        This function quasirandomly samples across a 3D space (sobol sampling with a rejection algorithm)
        Takes:
            NoSD_flt = scalar, number of samples desired across the space.
            dim_li = list, a list of dimensions involved (i.e. 3 for 3D problem)
        Returns:
            x_mat = matrix, a matrix of the x arrays desired
        """
        import numpy as np
        from scipy.stats import qmc
        high_x1_sca = Parameters_lis[0].bounds[1]
        high_x2_sca = Parameters_lis[1].bounds[1]
        high_x3_sca = Parameters_lis[2].bounds[1]
        low_x1_sca = Parameters_lis[0].bounds[0]
        low_x2_sca = Parameters_lis[1].bounds[0]
        low_x3_sca = Parameters_lis[2].bounds[0]
        delta_x1_sca = Parameters_lis[0].bounds[1] - Parameters_lis[0].bounds[0]
        step_x1_sca = delta_x1_sca / 100
        delta_x2_sca = Parameters_lis[1].bounds[1] - Parameters_lis[1].bounds[0]
        step_x2_sca = delta_x2_sca / 100
        delta_x3_sca = Parameters_lis[2].bounds[1] - Parameters_lis[2].bounds[0]
        step_x3_sca = delta_x3_sca / 100
        i = 0
        NoPI = 0
        MaxAttempts = 1000
        while i < MaxAttempts:
            if i == 0 or (i % 250) == 0:
                c_low_x1_sca = low_x1_sca
                c_low_x2_sca = low_x2_sca
                c_low_x3_sca = low_x3_sca
                c_high_x1_sca = high_x1_sca
                c_high_x2_sca = high_x2_sca
                c_high_x3_sca = high_x3_sca
            base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_flt)
            if NoS_sca < NoSD_flt:
                base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_flt * 2)
            sampler_obj = qmc.Sobol(d=len(Parameters_lis))
            sample_arr = sampler_obj.random_base2(m=base_sca)
            abstract_x1_arr = sample_arr[:,0]
            abstract_x2_arr = sample_arr[:,1]
            abstract_x3_arr = sample_arr[:,2]
            fixed_x1_arr = SobolFixer(abstract_x1_arr,c_low_x1_sca,c_high_x1_sca)
            fixed_x2_arr = SobolFixer(abstract_x2_arr,c_low_x2_sca,c_high_x2_sca)
            fixed_x3_arr = SobolFixer(abstract_x3_arr,c_low_x3_sca,c_high_x3_sca)
            NoPI_sca,x1PointsIn_arr,x2PointsIn_arr,x3PointsIn_arr = DetectNoPI3DCube_func(fixed_x1_arr,fixed_x2_arr,fixed_x3_arr,Parameters_lis)
            if NoSD_flt == NoPI_sca:
                break
            c_low_x1_sca = c_low_x1_sca - step_x1_sca
            c_high_x1_sca = c_high_x1_sca + step_x1_sca
            c_low_x2_sca = c_low_x2_sca - step_x2_sca
            c_high_x2_sca = c_high_x2_sca + step_x2_sca
            c_low_x3_sca = c_low_x3_sca - step_x3_sca
            c_high_x3_sca = c_high_x3_sca + step_x3_sca
            i += 1
        x1_arr = np.array(x1PointsIn_arr)
        x2_arr = np.array(x2PointsIn_arr)
        x3_arr = np.array(x3PointsIn_arr)
        x_mat = np.array([x1_arr,x2_arr,x3_arr])
        return x_mat
    def Plot3D_func(self,x_mat):
        from matplotlib import pyplot as plt
        fig=plt.figure(figsize=[3,3])
        ax=plt.axes(projection='3d')
        ax.scatter(x_mat[0],x_mat[1],x_mat[2])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")

class McIntersiteProjTh_class():
    def __init__(self):
        self.name = "Inner Class - Crombecq Exploratory Sampler with threshold based system"
    def McIntersiteProjTh_Tetra_func(self,OptimisationSetup_obj,AxClient_obj):
        # Pulling the Trials So Far
        TrialsThusFar_df = AxClient_obj.summarize()

        # Names of Dimensions
        DimensionNames_lis = []
        for i in OptimisationSetup_obj.Parameters_lis:
            DimensionNames_lis.append(i.name)

        # Matrix of Trials So Far
        q_lis = []
        for i in DimensionNames_lis:
            q_lis.append(np.array(TrialsThusFar_df[i]))
        q_mat = np.array(q_lis)

        xyz_mat = CoordinateConverterFromQsToXyz(q_mat[1:4])

        mixed_mat = np.concatenate((np.expand_dims(q_mat[0],axis=0),xyz_mat),axis=0)

        # Dimension Bounds
        b_lis = []
        for i in OptimisationSetup_obj.Parameters_lis:
            b_lis.append(i.bounds)

        # Number of Dimensions
        d_flt = len(b_lis)

        # Number of Points Sampled
        n_flt = len(TrialsThusFar_df)

        # Number of Points Candidated
        c_flt = 250*n_flt

        # Alpha Hyperparameter
        alpha_flt = 0.5

        # Pre-Candidate Matrix
        pre_y_lis = []
        for i in range(3):
            pre_y_lis.append(np.random.uniform(0.00000000001,1,c_flt))
        pre_y_mat = np.array(pre_y_lis)

        filtered_y_lis = []
        for row in pre_y_mat.T:
            point = Point3D(row[0],row[1],row[2])
            if PointWithinTetrahedronChecker_func(point,A_plane,B_plane,C_plane,D_plane) == 1:
                filtered_y_lis.append(row)
        filtered_y_mat = np.array(filtered_y_lis).T

        ci_flt = len(filtered_y_mat[0])
        # print(ci_flt) # 22s for 97 points at 10*n_flt
        # print(ci_flt)  # 2m30s for 876 points at 100*n_flt
        # print(ci_flt)  # 12m15s for 4534 points at 500*n_flt
        print(ci_flt)  # 12m15s for 4534 points at 250*n_flt
    
        q1_lis = np.random.uniform(b_lis[0][0],b_lis[0][1],ci_flt)

        y_mat = np.concatenate((np.expand_dims(q1_lis,axis=0),filtered_y_mat),axis=0)

        # d_min Parameter
        d_min_flt = (2*alpha_flt)/n_flt

        # mc-intersite_proj_th
        y_results = []
        for y_row in y_mat.T:
            gobbles_lis = []
            degooks_lis = []
            for x_row in mixed_mat.T:
                gobbles_lis.append(np.linalg.norm(x_row-y_row,ord=np.inf))
                degooks_lis.append(np.linalg.norm(x_row-y_row,ord=2))
            gobble_flt = np.min(gobbles_lis)
            degook_flt = degooks_lis[np.argmin(gobbles_lis)]
            if gobble_flt < d_min_flt:
                y_results.append(0)
            else:
                y_results.append(degook_flt)

        # Suggested Point
        y_coord_lis = []
        for dimension in y_mat:
            y_coord_lis.append([dimension[np.argmax(y_results)]])
        y_coord_mat = np.array(y_coord_lis)

        qs_mat = CoordinateConverterFromXyzToQs(y_coord_mat[1:4])

        return np.concatenate((np.expand_dims(y_coord_mat[0],axis=0),qs_mat),axis=0)


    def McIntersiteProjTh_func(self,OptimisationSetup_obj,AxClient_obj):
        # Pulling the Trials So Far
        TrialsThusFar_df = AxClient_obj.summarize()

        # Names of Dimensions
        DimensionNames_lis = []
        for i in OptimisationSetup_obj.Parameters_lis:
            DimensionNames_lis.append(i.name)

        # Matrix of Trials So Far
        x_lis = []
        for i in DimensionNames_lis:
            x_lis.append(np.array(TrialsThusFar_df[i]))
        x_mat = np.array(x_lis)

        # Dimension Bounds
        b_lis = []
        for i in OptimisationSetup_obj.Parameters_lis:
            b_lis.append(i.bounds)

        # Number of Dimensions
        d_flt = len(b_lis)

        # Number of Points Sampled
        n_flt = len(TrialsThusFar_df)

        # Number of Points Candidated
        c_flt = 100*n_flt

        # Alpha Hyperparameter
        alpha_flt = 0.5

        # Candidate Matrix
        y_lis = []
        for i in b_lis:
            y_lis.append(np.random.uniform(i[0],i[1],c_flt))
        y_mat = np.array(y_lis)

        # d_min Parameter
        d_min_flt = (2*alpha_flt)/n_flt

        # mc-intersite_proj_th
        y_results = []
        for y_row in y_mat.T:
            gobbles_lis = []
            degooks_lis = []
            for x_row in x_mat.T:
                gobbles_lis.append(np.linalg.norm(x_row-y_row,ord=np.inf))
                degooks_lis.append(np.linalg.norm(x_row-y_row,ord=2))
            gobble_flt = np.min(gobbles_lis)
            degook_flt = degooks_lis[np.argmin(gobbles_lis)]
            if gobble_flt < d_min_flt:
                y_results.append(0)
            else:
                y_results.append(degook_flt)
        
        # Suggested Point
        y_coord_lis = []
        for dimension in y_mat:
            y_coord_lis.append([dimension[np.argmax(y_results)]])
        y_coord_mat = np.array(y_coord_lis)
        return y_coord_mat.T

class McIntersiteProj_class():
    def __init__(self):
        self.name = "Inner Class - Crombecq Exploratory Sampler"
        self.Sampler1D = OneDim_class()
    def PointIntersiteDistance_func(self,x_mat,CandidatePoint_arr):
        """
        Function which takes the current matrix of values and returns a single criteria by the name
        of maximin also known as the minimum of the l2 norm (euclidean norm).
        https://doi.org/10.1016/j.ejor.2011.05.032
        """
        x_mat = x_mat.T
        l2Norm_arr = np.empty(0)
        for i in range(len(x_mat[0,:])):
            AbsVal_arr = np.empty(0)
            for d in range(len(x_mat[:,0])):
                AbsVal_arr = np.append(arr=AbsVal_arr,values=(np.abs(x_mat[d,i] - CandidatePoint_arr[d]))**2)
            l2Norm_arr = np.append(arr=l2Norm_arr,values=np.sqrt(np.sum(AbsVal_arr)))
        Pointl2Norm_flt = np.min(l2Norm_arr)
        return Pointl2Norm_flt
    def PointProjectedDistance_func(self,x_mat,CandidatePoint_arr):
        """
        Function which takes the current matrix of values and returns a single criteria by the name
        of projected distance.
        https://doi.org/10.1016/j.ejor.2011.05.032
        """
        x_mat = x_mat.T
        ProjectedDist_arr = np.empty(0)
        for i in range(len(x_mat[0,:])):
            AbsVal_arr = np.empty(0)
            for d in range(len(x_mat[:,0])):
                AbsVal_arr = np.append(arr=AbsVal_arr,values=np.abs(x_mat[d,i] - CandidatePoint_arr[d]))
            ProjectedDist_arr = np.append(arr=ProjectedDist_arr,values=np.min(AbsVal_arr))
        PointProjectedDist_flt = np.min(ProjectedDist_arr)
        return PointProjectedDist_flt
    def mc_intersite_proj(self,Parameters_lis,x_mat,CandidatePoint_arr):
        """
        https://doi.org/10.1016/j.ejor.2011.05.032
        """
        Pointl2Norm_flt = self.PointIntersiteDistance_func(x_mat,CandidatePoint_arr)
        PointProjectedDist_flt = self.PointProjectedDistance_func(x_mat,CandidatePoint_arr)
        a_flt = ((((len(x_mat[0,:])+1) ** (1/len(Parameters_lis)))-1)/2 * Pointl2Norm_flt)
        b_flt = (((len(x_mat[0,:])+1)/2) * PointProjectedDist_flt)
        mc_intersite_proj_flt = a_flt + b_flt
        return mc_intersite_proj_flt
    def CandidateGenerator_func(self,CandidateSamples_flt,Parameters_lis):
        Candidate_mat = np.empty(shape=(0,len(Parameters_lis)))
        for i in range(CandidateSamples_flt):
            Candidate_lis = []
            for j in range(len(Parameters_lis)):
                Candidate_lis.append(np.random.uniform(Parameters_lis[j].bounds[0],Parameters_lis[j].bounds[1]))
            Candidate_arr = np.array([Candidate_lis])
            Candidate_mat = np.append(arr=Candidate_mat,values=Candidate_arr,axis=0)
        return Candidate_mat
    def GetNextTrials_func(self,OptimisationSetup_obj,AxClient_obj):
        TrialsThusFar_df = AxClient_obj.summarize()
        ParameterNames_lis = []
        for i in OptimisationSetup_obj.Parameters_lis:
            ParameterNames_lis.append(i.name)
        Trials_df = TrialsThusFar_df[ParameterNames_lis]
        CurrentTrials_mat = Trials_df.to_numpy()
        CandidateSamples_flt = 10000
        CandidateMatrix_mat = self.CandidateGenerator_func(CandidateSamples_flt,OptimisationSetup_obj.Parameters_lis)
        FantasizedTrials_mat = CurrentTrials_mat
        NewTrials_arr = []
        for i in range(OptimisationSetup_obj.NoOfTrialsPerIteration_int):
            Evaluated_mc_intersite_proj_arr = np.empty(0)
            for CandidatePoint_arr in CandidateMatrix_mat:
                mc_intersite_proj_flt = self.mc_intersite_proj(OptimisationSetup_obj.Parameters_lis,FantasizedTrials_mat,CandidatePoint_arr)
                Evaluated_mc_intersite_proj_arr = np.append(arr=Evaluated_mc_intersite_proj_arr,values=mc_intersite_proj_flt)
            BestCandidate_arr = CandidateMatrix_mat[np.argmax(Evaluated_mc_intersite_proj_arr)]
            FantasizedTrials_mat = np.append(arr=FantasizedTrials_mat,values=[BestCandidate_arr],axis=0)
            NewTrials_arr.append(BestCandidate_arr)
        NewTrials_mat = np.array(NewTrials_arr)
        return NewTrials_mat

class custom_class():
    def __init__(self):
        self.name = "Inner Class - Custom Samplers"
    def LineTetrahedronSampler_func(self,NoSD_flt:float,Parameters_lis:list):
        # Setting some threshold hyperparameters for tuning process:
        # When these are all set to 25 expect a good responsible sampling but
        # half an hours rejection sampling.
        # Set to 1 for a 3 minute quick run which doesn't do the hottest in
        # terms of representativity.
        Threshold1DSystemTune = 25
        Theshold3DSystemTune = 25
        Threshold4DSystemTune = 25

        base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_flt * 10)
        MaxAttempts = 10000

        # Sweeping the q1 parameter in the linear space
        # for the best set of lows and highs

        q1_low_successful_arr = []
        q1_high_successful_arr = []

        i = 0
        while i < MaxAttempts:
            if i == 0 or (i % 100) == 0:
                c_low_q1_sca = Parameters_lis[0].bounds[0]
                c_high_q1_sca = Parameters_lis[0].bounds[1]
            sampler_obj = qmc.Sobol(d=4)
            sample_arr = sampler_obj.random_base2(m=base_sca)
            abstract_q1_arr = sample_arr[:,0]
            fixed_q1_arr = SobolFixer(abstract_q1_arr,c_low_q1_sca,c_high_q1_sca)
            NoPI1DL = PointsWithinLineCounter_func(fixed_q1_arr,Parameters_lis[0].bounds[0],Parameters_lis[0].bounds[1])
            if NoPI1DL == NoSD_flt:
                q1_low_successful_arr.append(c_low_q1_sca)
                q1_high_successful_arr.append(c_high_q1_sca)
            if len(q1_low_successful_arr) == Threshold1DSystemTune:
                break
            c_low_q1_sca = c_low_q1_sca - (abs(Parameters_lis[0].bounds[1] - Parameters_lis[0].bounds[0]))/10
            c_high_q1_sca = c_high_q1_sca + (abs(Parameters_lis[0].bounds[1] - Parameters_lis[0].bounds[0]))/10

            i += 1

        q1_low_avgsuccess_sca = np.average(np.array(q1_low_successful_arr))
        q1_high_avgsuccess_sca = np.average(np.array(q1_high_successful_arr))

        # Sweeping the x,y,z parameters in the tetrahedral space
        # for the best set of lows and highs

        x_low_successful_arr = []
        x_high_successful_arr = []
        y_low_successful_arr = []
        y_high_successful_arr = []
        z_low_successful_arr = []
        z_high_successful_arr = []

        i = 0
        while i < MaxAttempts:
            if i == 0 or (i % 15) == 0:
                c_low_x_sca = 0
                c_high_x_sca = 1
                c_low_y_sca = 0
                c_high_y_sca = PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5)
                c_low_z_sca = 0
                c_high_z_sca = PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5)
            sampler_obj = qmc.Sobol(d=4)
            sample_arr = sampler_obj.random_base2(m=base_sca)
            abstract_x_arr = sample_arr[:,1]
            abstract_y_arr = sample_arr[:,2]
            abstract_z_arr = sample_arr[:,3]
            fixed_x_arr = SobolFixer(abstract_x_arr,c_low_x_sca,c_high_x_sca)
            fixed_y_arr = SobolFixer(abstract_y_arr,c_low_y_sca,c_high_y_sca)
            fixed_z_arr = SobolFixer(abstract_z_arr,c_low_z_sca,c_high_z_sca)
            fixed_xyz_mat = np.array([fixed_x_arr,fixed_y_arr,fixed_z_arr])
            NoPI3DT_sca = PointsWithinTetrahedronCounter_func(fixed_xyz_mat.T,A_plane,B_plane,C_plane,D_plane)
            if NoPI3DT_sca == NoSD_flt:
                x_low_successful_arr.append(c_low_x_sca)
                x_high_successful_arr.append(c_high_x_sca)
                y_low_successful_arr.append(c_low_y_sca)
                y_high_successful_arr.append(c_high_y_sca)
                z_low_successful_arr.append(c_low_z_sca)
                z_high_successful_arr.append(c_high_z_sca)
            if len(x_low_successful_arr) == Theshold3DSystemTune:
                break
            c_low_x_sca = c_low_x_sca - (abs(1 - 0))/100
            c_high_x_sca = c_high_x_sca + (abs(1 - 0))/100
            c_low_y_sca = c_low_y_sca - (abs(PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5) - 0))/100
            c_high_y_sca = c_high_y_sca + (abs(PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5) - 0))/100
            c_low_z_sca = c_low_z_sca - (abs(PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5) - 0))/100
            c_high_z_sca = c_high_z_sca + (abs(PythagorasTheoremAdjacentOpposite_func(VarAO=1,VarC=0.5) - 0))/100
            i += 1

        x_low_avgsuccess_sca = np.average(np.array(x_low_successful_arr))
        x_high_avgsuccess_sca = np.average(np.array(x_high_successful_arr))
        y_low_avgsuccess_sca = np.average(np.array(y_low_successful_arr))
        y_high_avgsuccess_sca = np.average(np.array(y_high_successful_arr))
        z_low_avgsuccess_sca = np.average(np.array(z_low_successful_arr))
        z_high_avgsuccess_sca = np.average(np.array(z_high_successful_arr))

        # Leveraging the best lows and highs for both spaces to obtain a sobol
        # sampled array that suits both the line and tetrahedron spaces.

        Representativitys = []
        PI1DL3DTs = []

        i = 0
        while i < MaxAttempts:
            sampler_obj = qmc.Sobol(d=4)
            sample_arr = sampler_obj.random_base2(m=base_sca)
            abstract_q1_arr = sample_arr[:,0]
            abstract_x_arr = sample_arr[:,1]
            abstract_y_arr = sample_arr[:,2]
            abstract_z_arr = sample_arr[:,3]
            fixed_q1_arr = SobolFixer(abstract_q1_arr,q1_low_avgsuccess_sca,q1_high_avgsuccess_sca)
            fixed_x_arr = SobolFixer(abstract_x_arr,x_low_avgsuccess_sca,x_high_avgsuccess_sca)
            fixed_y_arr = SobolFixer(abstract_y_arr,y_low_avgsuccess_sca,y_high_avgsuccess_sca)
            fixed_z_arr = SobolFixer(abstract_z_arr,z_low_avgsuccess_sca,z_high_avgsuccess_sca)
            fixed_xyz_mat = np.array([fixed_x_arr,fixed_y_arr,fixed_z_arr])
            NoPI1DL_sca = PointsWithinLineCounter_func(fixed_q1_arr,Parameters_lis[0].bounds[0],Parameters_lis[0].bounds[1])
            PI1DL = PointsWithinLineReturner_func(fixed_q1_arr,Parameters_lis[0].bounds[0],Parameters_lis[0].bounds[1])
            NoPI3DT_sca = PointsWithinTetrahedronCounter_func(fixed_xyz_mat.T,A_plane,B_plane,C_plane,D_plane)
            PI3DT = PointsWithinTetrahedronReturner_func(fixed_xyz_mat.T,A_plane,B_plane,C_plane,D_plane)
            if NoPI1DL_sca == NoSD_flt and NoPI3DT_sca == NoSD_flt:
                LineRepresentativity = LinePointsRepresentativityEvaluator_func(PI1DL)
                TetrahedralRepresentativity = TetrahedronPointsRepresentativityEvaluator_func(PI3DT)
                Representativitys.append(LineRepresentativity+TetrahedralRepresentativity)
                PI1DL3DT = np.array([PI1DL,PI3DT.T[0],PI3DT.T[1],PI3DT.T[2]])
                PI1DL3DTs.append(PI1DL3DT)
            if len(Representativitys) == Threshold4DSystemTune:
                break
            i += 1

        PI1DL3DT_Final = PI1DL3DTs[np.argmax(Representativitys)]
        q1_arr = PI1DL3DT_Final[[0],:]
        xyzCoords_mat = PI1DL3DT_Final[[1,2,3],:]
        qs_mat = CoordinateConverterFromXyzToQs(xyzCoords_mat)
        full_qs_mat = np.concat((q1_arr.T,qs_mat.T),axis=1).T
        return full_qs_mat

class Sampler_class(object):
    def __init__(self):
        self.name = "Outer Class - Sampler Tool"
        self.one = OneDim_class()
        self.two = TwoDim_class()
        self.three = ThreeDim_class()
        self.custom = custom_class()
        self.McIntersiteProj = McIntersiteProj_class()
        self.McIntersiteProjTh = McIntersiteProjTh_class()